from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from mrun.cli import main as mrun_main
from mrun.decompiler._json import canonical_json_bytes, canonical_sha256
from mrun.decompiler.cli import main as decompiler_main
from mrun.decompiler.emitter import (
    ArtifactEmissionError,
    ArtifactVerificationError,
    ExecutionCertificationUnavailable,
    NativeSourceComponentEmitter,
    build_component_artifact,
    certify_component_artifact,
    inspect_component_eligibility,
    open_component_artifact,
)
from mrun.decompiler.source import SourceMutationError

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4}


def _tensor_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count * _DTYPE_BYTES[dtype]


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    cursor = 0
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for name in sorted(tensors):
        dtype, shape = tensors[name]
        byte_count = _tensor_bytes(dtype, shape)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + byte_count],
        }
        cursor += byte_count
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    payload = bytes((index * 17 + 3) % 251 for index in range(cursor))
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def _config(family: str, *, tied: bool) -> dict[str, object]:
    architecture = {
        "qwen2": "Qwen2ForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
        "llama": "LlamaForCausalLM",
    }[family]
    config: dict[str, object] = {
        "architectures": [architecture],
        "model_type": family,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
        "hidden_act": "silu",
        "tie_word_embeddings": tied,
        "bos_token_id": 1,
        "eos_token_id": [2, 3],
        "pad_token_id": 0,
        "attention_dropout": 0.0,
        "use_cache": True,
    }
    if family == "qwen3":
        config["attention_bias"] = False
    if family == "llama":
        config.update({"attention_bias": False, "mlp_bias": False, "pretraining_tp": 1})
    return config


def _tensors(family: str, *, tied: bool) -> dict[str, tuple[str, tuple[int, ...]]]:
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {
        "model.embed_tokens.weight": ("BF16", (8, 4)),
        "model.norm.weight": ("BF16", (4,)),
        "model.layers.0.input_layernorm.weight": ("BF16", (4,)),
        "model.layers.0.post_attention_layernorm.weight": ("BF16", (4,)),
        "model.layers.0.self_attn.q_proj.weight": ("BF16", (4, 4)),
        "model.layers.0.self_attn.k_proj.weight": ("BF16", (2, 4)),
        "model.layers.0.self_attn.v_proj.weight": ("BF16", (2, 4)),
        "model.layers.0.self_attn.o_proj.weight": ("BF16", (4, 4)),
        "model.layers.0.mlp.gate_proj.weight": ("BF16", (8, 4)),
        "model.layers.0.mlp.up_proj.weight": ("BF16", (8, 4)),
        "model.layers.0.mlp.down_proj.weight": ("BF16", (4, 8)),
    }
    if family == "qwen2":
        tensors.update(
            {
                "model.layers.0.self_attn.q_proj.bias": ("BF16", (4,)),
                "model.layers.0.self_attn.k_proj.bias": ("BF16", (2,)),
                "model.layers.0.self_attn.v_proj.bias": ("BF16", (2,)),
            }
        )
    if family == "qwen3":
        tensors.update(
            {
                "model.layers.0.self_attn.q_norm.weight": ("BF16", (2,)),
                "model.layers.0.self_attn.k_norm.weight": ("BF16", (2,)),
            }
        )
    if not tied:
        tensors["lm_head.weight"] = ("BF16", (8, 4))
    return tensors


def _write_model(
    root: Path,
    family: str,
    *,
    tied: bool,
    dtype_override: str | None = None,
) -> None:
    root.mkdir()
    tensors = _tensors(family, tied=tied)
    if dtype_override is not None:
        dtype, shape = tensors["model.layers.0.self_attn.q_proj.weight"]
        del dtype
        tensors["model.layers.0.self_attn.q_proj.weight"] = (dtype_override, shape)
    (root / "config.json").write_text(
        json.dumps(_config(family, tied=tied), sort_keys=True), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, sort_keys=True), encoding="utf-8"
    )
    _write_safetensors(root / "model.safetensors", tensors)


@pytest.mark.parametrize(
    ("family", "tied"),
    [("qwen2", True), ("qwen3", False), ("llama", False)],
)
def test_build_reopen_is_byte_complete_for_supported_dense_families(
    tmp_path: Path, family: str, tied: bool
) -> None:
    source_root = tmp_path / family
    _write_model(source_root, family, tied=tied)

    record = build_component_artifact(
        source_root,
        tmp_path / "artifacts",
        source_id=f"Test/{family}",
        resolved_revision="a" * 40,
    )
    artifact = open_component_artifact(record.path)

    assert record.verified_reopen
    assert not record.execution_certified
    assert artifact.manifest["status"] == "built-unexecuted"
    assert not artifact.manifest["execution_certified"]
    detached_manifest = artifact.manifest
    detached_manifest["status"] = "caller-mutated"
    assert artifact.manifest["status"] == "built-unexecuted"
    assert artifact.manifest["artifact_codec"]["supported_source_contracts"] == [
        {
            "codec_id": "raw-float",
            "codec_version": "1.0.0",
            "stored_dtypes": ["BF16", "F16", "F32", "F64"],
            "byte_order": "little",
            "packing": "none",
        }
    ]
    allocations = artifact.manifest["allocations"]
    assert len(allocations) == len(artifact.tensor_index.tensors)
    assert record.emitted_blob_bytes == artifact.tensor_index.total_tensor_bytes
    for allocation in allocations:
        with (source_root / allocation["source_file"]).open("rb") as handle:
            handle.seek(allocation["source_byte_offset"])
            expected = handle.read(allocation["source_byte_length"])
        assert (record.path / allocation["blob"]["path"]).read_bytes() == expected

    assets = artifact.manifest["assets"]
    assert {item["source_path"] for item in assets} == {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    for asset in assets:
        assert (record.path / asset["artifact_path"]).read_bytes() == (
            source_root / asset["source_path"]
        ).read_bytes()

    aliases = artifact.ir_bundle.physical_weights.alias_classes
    if tied:
        assert len(aliases) == 1
        assert aliases[0].logical_names == ("lm_head.weight", "token_embedding.weight")
        alias_allocations = [
            item for item in allocations if item["allocation_id"] == aliases[0].allocation_id
        ]
        assert len(alias_allocations) == 1
    else:
        assert not aliases


def test_artifact_identity_is_path_independent_and_overwrite_is_refused(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    _write_model(source_root, "qwen2", tied=True)
    first = build_component_artifact(source_root, tmp_path / "first", source_id="Test/Portable")
    second = build_component_artifact(source_root, tmp_path / "second", source_id="Test/Portable")

    assert first.artifact_id == second.artifact_id
    assert (first.path / "manifest.json").read_bytes() == (
        second.path / "manifest.json"
    ).read_bytes()
    with pytest.raises(ArtifactEmissionError, match="overwrite"):
        build_component_artifact(source_root, tmp_path / "first", source_id="Test/Portable")


def test_source_mutation_during_streaming_aborts_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "artifacts"
    _write_model(source_root, "llama", tied=False)
    import mrun.decompiler.emitter as emitter_module

    original = emitter_module._copy_source_range
    mutated = False

    def mutate_after_copy(*args: object, **kwargs: object) -> tuple[str, int]:
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            mutated = True
            (source_root / "tokenizer.json").write_text('{"changed":true}\n', encoding="utf-8")
        return result

    monkeypatch.setattr(emitter_module, "_copy_source_range", mutate_after_copy)
    with pytest.raises(SourceMutationError, match="changed after freeze"):
        build_component_artifact(source_root, output_root)
    assert output_root.is_dir()
    assert list(output_root.iterdir()) == []


def test_blob_extra_file_and_alias_tampering_fail_closed(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_model(source_root, "qwen2", tied=True)

    blob_record = build_component_artifact(source_root, tmp_path / "blob-out")
    blob_manifest = json.loads((blob_record.path / "manifest.json").read_bytes())
    blob_path = blob_record.path / blob_manifest["allocations"][0]["blob"]["path"]
    blob = bytearray(blob_path.read_bytes())
    blob[0] ^= 0xFF
    blob_path.write_bytes(blob)
    with pytest.raises(ArtifactVerificationError, match="content hash mismatch"):
        open_component_artifact(blob_record.path)

    extra_record = build_component_artifact(source_root, tmp_path / "extra-out")
    (extra_record.path / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="inventory"):
        open_component_artifact(extra_record.path)

    alias_record = build_component_artifact(source_root, tmp_path / "alias-out")
    manifest_path = alias_record.path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["alias_classes"][0]["logical_names"][1] = "tampered.weight"
    manifest["artifact_id"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "artifact_id"}
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    renamed = alias_record.path.parent / manifest["artifact_id"]
    alias_record.path.rename(renamed)
    with pytest.raises(ArtifactVerificationError, match="alias classes"):
        open_component_artifact(renamed)


def test_symlink_and_output_inside_source_are_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_model(source_root, "llama", tied=False)
    with pytest.raises(ArtifactEmissionError, match="inside the frozen source"):
        build_component_artifact(source_root, source_root / "artifacts")
    assert not (source_root / "artifacts").exists()

    record = build_component_artifact(source_root, tmp_path / "artifacts")
    blob_manifest = json.loads((record.path / "manifest.json").read_bytes())
    blob_path = record.path / blob_manifest["allocations"][0]["blob"]["path"]
    blob_path.unlink()
    blob_path.symlink_to(record.path / "source.json")
    with pytest.raises(ArtifactVerificationError, match="symbolic links"):
        open_component_artifact(record.path)


def test_integrity_certification_never_claims_execution(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_model(source_root, "qwen3", tied=False)
    record = build_component_artifact(source_root, tmp_path / "artifacts")

    certification = certify_component_artifact(record.path)
    assert certification.status == "passed"
    assert certification.scope == "artifact-integrity-only"
    assert not certification.execution_performed
    assert not certification.execution_certified
    with pytest.raises(ExecutionCertificationUnavailable, match="no registered"):
        certify_component_artifact(record.path, require_execution=True)


def test_packed_or_integer_source_is_precisely_ineligible(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_model(source_root, "llama", tied=False, dtype_override="I32")
    from mrun.decompiler import decompile_source

    result = decompile_source(source_root)
    eligibility = inspect_component_eligibility(result)
    assert not eligibility["eligible"]
    assert result.report.failures[0].code == "unsupported_variant"
    assert "unregistered_source_codec" in str(result.report.failures[0].details)
    with pytest.raises(ArtifactEmissionError, match="successful U2"):
        NativeSourceComponentEmitter().build(result, tmp_path / "artifacts")


def test_cli_inspect_build_certify_report_and_execution_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    _write_model(source_root, "qwen2", tied=True)

    assert mrun_main(["decompile", "inspect", str(source_root)]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["status"] == "supported"
    assert inspection["component_emission"]["eligible"]

    assert decompiler_main(["build", str(source_root), "--output-root", str(artifact_root)]) == 0
    built = json.loads(capsys.readouterr().out)
    artifact_path = Path(built["build"]["path"])
    assert built["status"] == "built-unexecuted"

    assert decompiler_main(["certify", str(artifact_path)]) == 0
    certified = json.loads(capsys.readouterr().out)
    assert certified["status"] == "integrity-certified"
    assert not certified["certification"]["execution_certified"]

    assert decompiler_main(["report", str(artifact_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "verified-built-unexecuted"
    assert not report["artifact_report"]["execution_certified"]

    assert decompiler_main(["certify", str(artifact_path), "--require-execution"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "execution_certification_unavailable"
    assert error["error"]["gate"] == "G8"
