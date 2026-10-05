from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mrun.decompiler import ArtifactEmissionError, build_component_artifact
from mrun.decompiler.cli import main as decompiler_main
from mrun.decompiler.mlx_native import (
    SOURCE_MLX_BUILDER_ABI,
    SOURCE_MLX_MAPPING_ABI,
    SOURCE_MLX_NATIVE_SCHEMA,
    SOURCE_MLX_Q4_BUILDER_ABI,
    SOURCE_MLX_Q4_CODEC,
    SOURCE_MLX_Q4_NATIVE_SCHEMA,
    SOURCE_MLX_Q4_NUMERICAL_CONTRACT,
    MLXSourceQ4Engine,
    SourceMlxArtifactError,
    SourceMlxLoweringError,
    VerifiedSourceMlxArtifact,
    VerifiedSourceMlxQ4Artifact,
    build_source_mlx_artifact,
    build_source_mlx_q4_artifact,
)


def _config(
    family: str,
    *,
    tied: bool,
    hidden: int = 8,
    intermediate: int = 16,
    vocab: int = 11,
) -> dict[str, object]:
    architecture = {
        "qwen2": "Qwen2ForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
        "llama": "LlamaForCausalLM",
    }[family]
    config: dict[str, object] = {
        "architectures": [architecture],
        "model_type": family,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": hidden // 4,
        "vocab_size": vocab,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
        "hidden_act": "silu",
        "tie_word_embeddings": tied,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "attention_dropout": 0.0,
        "use_cache": True,
    }
    if family == "qwen3":
        config["attention_bias"] = False
    if family == "llama":
        config.update({"attention_bias": False, "mlp_bias": False, "pretraining_tp": 1})
    return config


def _weights(
    family: str,
    *,
    tied: bool,
    hidden: int = 8,
    intermediate: int = 16,
    vocab: int = 11,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(700 + {"qwen2": 2, "qwen3": 3, "llama": 4}[family])

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.05

    prefix = "model.layers.0"
    head_dim = hidden // 4
    kv_width = 2 * head_dim
    weights = {
        "model.embed_tokens.weight": random(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        f"{prefix}.input_layernorm.weight": torch.ones(hidden),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden),
        f"{prefix}.self_attn.q_proj.weight": random(hidden, hidden),
        f"{prefix}.self_attn.k_proj.weight": random(kv_width, hidden),
        f"{prefix}.self_attn.v_proj.weight": random(kv_width, hidden),
        f"{prefix}.self_attn.o_proj.weight": random(hidden, hidden),
        f"{prefix}.mlp.gate_proj.weight": random(intermediate, hidden),
        f"{prefix}.mlp.up_proj.weight": random(intermediate, hidden),
        f"{prefix}.mlp.down_proj.weight": random(hidden, intermediate),
    }
    if family == "qwen2":
        weights.update(
            {
                f"{prefix}.self_attn.q_proj.bias": random(hidden),
                f"{prefix}.self_attn.k_proj.bias": random(kv_width),
                f"{prefix}.self_attn.v_proj.bias": random(kv_width),
            }
        )
    if family == "qwen3":
        weights.update(
            {
                f"{prefix}.self_attn.q_norm.weight": torch.ones(head_dim),
                f"{prefix}.self_attn.k_norm.weight": torch.ones(head_dim),
            }
        )
    if not tied:
        weights["lm_head.weight"] = random(vocab, hidden)
    return weights


def _write_model(
    root: Path,
    family: str,
    *,
    tied: bool,
    hidden: int = 8,
    intermediate: int = 16,
    vocab: int = 11,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    root.mkdir()
    config = _config(
        family,
        tied=tied,
        hidden=hidden,
        intermediate=intermediate,
        vocab=vocab,
    )
    weights = {
        name: value.to(dtype)
        for name, value in _weights(
            family,
            tied=tied,
            hidden=hidden,
            intermediate=intermediate,
            vocab=vocab,
        ).items()
    }
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, sort_keys=True), encoding="utf-8"
    )
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    return weights


def _build(
    tmp_path: Path, family: str, *, tied: bool
) -> tuple[dict[str, torch.Tensor], Path, VerifiedSourceMlxArtifact]:
    source = tmp_path / family
    weights = _write_model(source, family, tied=tied)
    canonical = build_component_artifact(
        source,
        tmp_path / "canonical",
        source_id=f"Test/{family}",
        resolved_revision="b" * 40,
    )
    record = build_source_mlx_artifact(canonical.path, tmp_path / "native")
    return weights, canonical.path, VerifiedSourceMlxArtifact(record.path)


def _build_q4(
    tmp_path: Path, family: str, *, tied: bool
) -> tuple[dict[str, torch.Tensor], Path, VerifiedSourceMlxQ4Artifact]:
    pytest.importorskip("mlx.core")
    source = tmp_path / family
    weights = _write_model(
        source,
        family,
        tied=tied,
        hidden=64,
        intermediate=128,
    )
    canonical = build_component_artifact(
        source,
        tmp_path / "canonical",
        source_id=f"Test/{family}",
        resolved_revision="c" * 40,
    )
    record = build_source_mlx_q4_artifact(canonical.path, tmp_path / "native-q4")
    return weights, canonical.path, VerifiedSourceMlxQ4Artifact(record.path)


@pytest.mark.parametrize(
    ("family", "tied", "roles"),
    [
        ("qwen2", True, {"body", "norm", "lexical_shared"}),
        ("qwen3", False, {"body", "norm", "ingress", "egress"}),
        ("llama", False, {"body", "norm", "ingress", "egress"}),
    ],
)
def test_direct_source_mlx_lowering_is_exact_complete_and_role_separated(
    tmp_path: Path,
    family: str,
    tied: bool,
    roles: set[str],
) -> None:
    source_weights, canonical_path, artifact = _build(tmp_path, family, tied=tied)
    manifest = artifact.manifest

    assert manifest["schema"] == SOURCE_MLX_NATIVE_SCHEMA
    assert manifest["status"] == "native-lowered-unexecuted"
    assert not manifest["execution_certified"]
    assert manifest["native_runtime_candidate"]
    assert not manifest["production_runtime_eligible"]
    assert manifest["recipe"]["builder_abi"] == SOURCE_MLX_BUILDER_ABI
    assert manifest["recipe"]["mapping_abi"] == SOURCE_MLX_MAPPING_ABI
    assert manifest["source"]["direct_from_canonical_source"]
    assert not manifest["source"]["intermediate_qstore"]
    assert {item["role"] for item in manifest["shards"]} == roles

    emitted: dict[str, torch.Tensor] = {}
    source_allocation_ids: set[str] = set()
    for shard in manifest["shards"]:
        with safe_open(artifact.path / shard["filename"], framework="pt", device="cpu") as handle:
            for parameter in shard["parameters"]:
                name = parameter["name"]
                emitted[name] = handle.get_tensor(name)
                source_allocation_ids.add(parameter["source_allocation_id"])
    assert set(emitted) == set(source_weights)
    assert all(torch.equal(emitted[name], value) for name, value in source_weights.items())
    assert len(source_allocation_ids) == len(source_weights)
    assert manifest["coverage"]["emitted_parameter_count"] == len(source_weights)
    assert manifest["coverage"]["source_allocation_count"] == len(source_weights)
    if tied:
        assert "lm_head.weight" not in emitted
        assert manifest["source"]["tied_lexical_allocation"]

    repeated = build_source_mlx_artifact(canonical_path, tmp_path / "native")
    assert repeated.path == artifact.path
    assert repeated.artifact_sha256 == artifact.artifact_sha256
    assert not repeated.production_runtime_eligible


def test_direct_source_mlx_artifact_rejects_tampered_shard(tmp_path: Path) -> None:
    _weights_value, _canonical, artifact = _build(tmp_path, "qwen2", tied=True)
    shard = artifact.path / artifact.manifest["shards"][0]["filename"]
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(SourceMlxArtifactError, match="hash"):
        VerifiedSourceMlxArtifact(artifact.path)


def test_direct_source_raw_v1_mapping_artifact_remains_strictly_openable(
    tmp_path: Path,
) -> None:
    _weights_value, _canonical, artifact = _build(tmp_path, "qwen2", tied=True)
    manifest_path = artifact.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recipe"]["mapping_abi"] = "mrun-dense-hf-source-name-to-mlx-lm-v1"
    manifest["source"].pop("runtime_numerical_compatibility")
    import hashlib

    canonical_recipe = json.dumps(
        manifest["recipe"], sort_keys=True, separators=(",", ":")
    ).encode()
    manifest["build_key_sha256"] = hashlib.sha256(canonical_recipe).hexdigest()
    manifest.pop("artifact_sha256")
    canonical_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["artifact_sha256"] = hashlib.sha256(canonical_manifest).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    reopened = VerifiedSourceMlxArtifact(artifact.path)
    assert reopened.manifest["recipe"]["mapping_abi"].endswith("-v1")
    reopened.assert_unchanged()


def test_direct_source_current_mapping_requires_numerical_compatibility(
    tmp_path: Path,
) -> None:
    _weights_value, _canonical, artifact = _build(tmp_path, "qwen2", tied=True)
    manifest_path = artifact.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"].pop("runtime_numerical_compatibility")
    import hashlib

    manifest.pop("artifact_sha256")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(SourceMlxArtifactError, match="lineage"):
        VerifiedSourceMlxArtifact(artifact.path)


def test_direct_source_mlx_rejects_non_default_rope(tmp_path: Path) -> None:
    source = tmp_path / "qwen2"
    _write_model(source, "qwen2", tied=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["rope_scaling"] = {"rope_type": "linear", "factor": 2.0}
    (source / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    with pytest.raises(ArtifactEmissionError, match="successful U2"):
        build_component_artifact(source, tmp_path / "canonical")


def test_direct_source_mlx_cli_build_and_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "qwen2"
    _write_model(source, "qwen2", tied=True)
    canonical = build_component_artifact(source, tmp_path / "canonical")

    assert (
        decompiler_main(
            ["lower-mlx", str(canonical.path), "--output-root", str(tmp_path / "native")]
        )
        == 0
    )
    lowered = json.loads(capsys.readouterr().out)
    assert lowered["status"] == "native-lowered-unexecuted"
    assert not lowered["build"]["production_runtime_eligible"]

    assert decompiler_main(["verify-mlx", lowered["build"]["path"]]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["artifact_sha256"] == lowered["build"]["artifact_sha256"]
    assert not verified["execution_certified"]
    assert not verified["production_runtime_eligible"]


@pytest.mark.parametrize(
    ("family", "tied", "roles"),
    [
        ("qwen2", True, {"body", "norm", "lexical_shared"}),
        ("qwen3", False, {"body", "norm", "ingress", "egress"}),
        ("llama", False, {"body", "norm", "ingress", "egress"}),
    ],
)
def test_direct_source_q4_is_role_separated_alias_preserving_and_no_qstore(
    tmp_path: Path,
    family: str,
    tied: bool,
    roles: set[str],
) -> None:
    source_weights, canonical_path, artifact = _build_q4(tmp_path, family, tied=tied)
    manifest = artifact.manifest

    assert manifest["schema"] == SOURCE_MLX_Q4_NATIVE_SCHEMA
    assert manifest["status"] == "native-lowered-approximate-unexecuted"
    assert manifest["approximate_quantized"]
    assert not manifest["execution_certified"]
    assert not manifest["production_runtime_eligible"]
    assert manifest["numerical_contract"] == SOURCE_MLX_Q4_NUMERICAL_CONTRACT
    assert manifest["recipe"]["builder_abi"] == SOURCE_MLX_Q4_BUILDER_ABI
    assert manifest["recipe"]["codec"] == SOURCE_MLX_Q4_CODEC
    assert manifest["recipe"]["direct_from_canonical_source"] is True
    assert manifest["recipe"]["intermediate_qstore"] is False
    assert manifest["source"]["direct_from_canonical_source"] is True
    assert manifest["source"]["intermediate_qstore"] is False
    assert manifest["quantization"]["source_codec"].endswith("canonical-source-v1")
    assert {item["role"] for item in manifest["shards"]} == roles
    assert manifest["coverage"]["auxiliary_source_exact"] is True
    assert manifest["quantization"]["max_abs_error"] > 0
    assert manifest["quantization"]["rmse"] > 0
    assert (
        manifest["coverage"]["emitted_parameter_bytes"]
        < manifest["coverage"]["source_allocation_bytes"]
    )

    emitted: dict[str, torch.Tensor] = {}
    descriptors: dict[str, dict[str, object]] = {}
    allocation_ids: set[str] = set()
    for shard in manifest["shards"]:
        with safe_open(artifact.path / shard["filename"], framework="pt", device="cpu") as handle:
            for parameter in shard["parameters"]:
                emitted[parameter["name"]] = handle.get_tensor(parameter["name"])
                descriptors[parameter["name"]] = parameter
                allocation_ids.add(parameter["source_allocation_id"])

    matrix_names = {name for name, value in source_weights.items() if value.ndim == 2}
    auxiliary_names = set(source_weights) - matrix_names
    expected_names = set(auxiliary_names)
    for name in matrix_names:
        base = name.removesuffix(".weight")
        expected_names.update({f"{base}.weight", f"{base}.scales", f"{base}.biases"})
    assert set(emitted) == expected_names
    for name in auxiliary_names:
        assert torch.equal(emitted[name], source_weights[name])
        assert descriptors[name]["encoding"] == "source-exact"
    for name in matrix_names:
        base = name.removesuffix(".weight")
        rows, columns = source_weights[name].shape
        assert emitted[f"{base}.weight"].dtype == torch.uint32
        assert list(emitted[f"{base}.weight"].shape) == [rows, columns // 8]
        assert list(emitted[f"{base}.scales"].shape) == [rows, columns // 64]
        assert list(emitted[f"{base}.biases"].shape) == [rows, columns // 64]
    assert len(allocation_ids) == len(source_weights)
    if tied:
        assert manifest["source"]["tied_lexical_allocation"]
        assert "lm_head.weight" not in emitted
        embed = descriptors["model.embed_tokens.weight"]
        assert embed["logical_names"] == ["lm_head.weight", "token_embedding.weight"]

    repeated = build_source_mlx_q4_artifact(canonical_path, tmp_path / "native-q4")
    assert repeated.path == artifact.path
    assert repeated.artifact_sha256 == artifact.artifact_sha256
    assert repeated.approximate_quantized
    assert not repeated.production_runtime_eligible


def test_direct_source_q4_rejects_tamper_and_resigned_lineage(tmp_path: Path) -> None:
    _weights_value, _canonical_path, artifact = _build_q4(tmp_path, "qwen2", tied=True)
    shard = artifact.path / artifact.manifest["shards"][0]["filename"]
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)
    with pytest.raises(SourceMlxArtifactError, match="hash"):
        VerifiedSourceMlxQ4Artifact(artifact.path)

    lineage = tmp_path / "lineage"
    lineage.mkdir()
    _weights_value, _canonical_path, fresh = _build_q4(lineage, "qwen2", tied=True)
    manifest_path = fresh.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["tokenizer_custody_sha256"] = "0" * 64
    manifest.pop("artifact_sha256")
    import hashlib

    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(SourceMlxArtifactError, match="lineage"):
        VerifiedSourceMlxQ4Artifact(fresh.path)


def test_direct_source_q4_rejects_unsupported_width_and_f64(tmp_path: Path) -> None:
    pytest.importorskip("mlx.core")
    narrow = tmp_path / "narrow"
    _write_model(narrow, "qwen2", tied=True)
    canonical = build_component_artifact(narrow, tmp_path / "canonical-narrow")
    with pytest.raises(SourceMlxLoweringError, match="divisible"):
        build_source_mlx_q4_artifact(canonical.path, tmp_path / "native-narrow")

    double = tmp_path / "double"
    _write_model(
        double,
        "qwen2",
        tied=True,
        hidden=64,
        intermediate=128,
        dtype=torch.float64,
    )
    canonical_double = build_component_artifact(double, tmp_path / "canonical-double")
    with pytest.raises(SourceMlxLoweringError, match="BF16, F16, or F32"):
        build_source_mlx_q4_artifact(canonical_double.path, tmp_path / "native-double")


def test_direct_source_q4_cli_and_capability_boundary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("mlx.core")
    source = tmp_path / "qwen2"
    _write_model(source, "qwen2", tied=True, hidden=64, intermediate=128)
    canonical = build_component_artifact(source, tmp_path / "canonical")
    assert (
        decompiler_main(
            ["lower-mlx-q4", str(canonical.path), "--output-root", str(tmp_path / "q4")]
        )
        == 0
    )
    lowered = json.loads(capsys.readouterr().out)
    assert lowered["status"] == "native-lowered-approximate-unexecuted"
    assert lowered["build"]["approximate_quantized"]
    assert not lowered["build"]["production_runtime_eligible"]
    assert decompiler_main(["verify-mlx-q4", lowered["build"]["path"]]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["approximate_quantized"]
    assert not verified["execution_certified"]
    assert not verified["production_runtime_eligible"]

    shell = object.__new__(MLXSourceQ4Engine)
    shell.approximate_quantized = True
    shell.compact_fused_weights = True
    capabilities = shell.capabilities()
    assert capabilities.approximate_quantized
    assert capabilities.compact_fused_weights
