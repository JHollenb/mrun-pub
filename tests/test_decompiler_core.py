from __future__ import annotations

import json
import shutil
import struct
from copy import deepcopy
from pathlib import Path

import pytest

from mrun.decompiler import (
    AdapterRegistry,
    DecompileResult,
    FrozenSourceBundle,
    IRValidationError,
    PhysicalWeightIR,
    Qwen2Adapter,
    SafetensorsTensorIndexer,
    SourceMutationError,
    SourcePolicy,
    TensorIndex,
    TensorIndexError,
    decompile_source,
    freeze_source,
)
from mrun.decompiler._json import canonical_json, canonical_sha256
from mrun.decompiler.source import SourceCustodyError, SourcePolicyError

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4}


def _tensor_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count * _DTYPE_BYTES[dtype]


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...]]],
    *,
    ranges: dict[str, tuple[int, int]] | None = None,
) -> None:
    cursor = 0
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for name in sorted(tensors):
        dtype, shape = tensors[name]
        byte_count = _tensor_bytes(dtype, shape)
        start, end = (cursor, cursor + byte_count) if ranges is None else ranges[name]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, end],
        }
        cursor = max(cursor + byte_count, end)
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    payload_bytes = max(
        (entry["data_offsets"][1] for key, entry in header.items() if key != "__metadata__"),
        default=0,
    )
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + bytes(payload_bytes))


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
    config_update: dict[str, object] | None = None,
    tensor_update: dict[str, tuple[str, tuple[int, ...]]] | None = None,
    remove_tensors: tuple[str, ...] = (),
) -> dict[str, tuple[str, tuple[int, ...]]]:
    root.mkdir()
    config = _config(family, tied=tied)
    config.update(config_update or {})
    tensors = _tensors(family, tied=tied)
    tensors.update(tensor_update or {})
    for name in remove_tensors:
        tensors.pop(name, None)
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, sort_keys=True),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"do_sample": False, "max_new_tokens": 32}, sort_keys=True),
        encoding="utf-8",
    )
    _write_safetensors(root / "model.safetensors", tensors)
    return tensors


def test_qwen2_tied_decompile_has_total_raw_coverage_and_round_trips(
    tmp_path: Path,
) -> None:
    tensors = _write_model(tmp_path / "qwen2", "qwen2", tied=True)
    result = decompile_source(
        tmp_path / "qwen2",
        source_id="Test/TinyQwen2",
        resolved_revision="a" * 40,
    )

    assert result.succeeded
    assert result.report.status == "decoded"
    assert result.report.universal_level == "U2"
    assert result.report.selected_adapter_id == "mrun.hf.qwen2-dense"
    assert result.report.coverage is not None and result.report.coverage.complete
    assert result.report.coverage.source_tensor_count == len(tensors)
    assert result.ir_bundle is not None
    physical = result.ir_bundle.physical_weights
    assert {item.source_name for item in physical.classifications} == set(tensors)
    assert len(physical.alias_classes) == 1
    assert physical.alias_classes[0].logical_names == (
        "lm_head.weight",
        "token_embedding.weight",
    )
    lexical_views = {
        view.logical_name: view.allocation_id
        for view in physical.views
        if view.logical_name in {"lm_head.weight", "token_embedding.weight"}
    }
    assert len(set(lexical_views.values())) == 1
    assert len(result.ir_bundle.state.slots) == 3
    assert result.ir_bundle.io.chat_templates[0].template_id == "default"
    assert "G7-native-artifact-emission-and-reopen" in result.report.pending_gates

    restored = DecompileResult.from_dict(result.as_dict())
    assert restored.as_dict() == result.as_dict()
    assert json.loads(json.dumps(result.as_dict(), sort_keys=True)) == result.as_dict()


@pytest.mark.parametrize(
    ("family", "adapter_id", "required_view"),
    [
        ("qwen3", "mrun.hf.qwen3-dense", "layers.0.attention.q_norm.weight"),
        ("llama", "mrun.hf.llama-dense-baseline", "layers.0.attention.q_proj.weight"),
    ],
)
def test_qwen3_and_llama_untied_adapters_are_exact_and_total(
    tmp_path: Path,
    family: str,
    adapter_id: str,
    required_view: str,
) -> None:
    root = tmp_path / family
    tensors = _write_model(root, family, tied=False)
    result = decompile_source(root, source_id=f"Test/{family}")

    assert result.succeeded
    assert result.report.selected_adapter_id == adapter_id
    assert result.report.coverage is not None
    assert result.report.coverage.classified_tensor_count == len(tensors)
    assert result.ir_bundle is not None
    assert not result.ir_bundle.physical_weights.alias_classes
    assert required_view in {view.logical_name for view in result.ir_bundle.physical_weights.views}
    assert result.ir_bundle.model.dimensions.num_key_value_heads == 1


@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_declared_attention_and_llama_mlp_bias_variants_are_total(
    tmp_path: Path, family: str
) -> None:
    bias_tensors = {
        "model.layers.0.self_attn.q_proj.bias": ("BF16", (4,)),
        "model.layers.0.self_attn.k_proj.bias": ("BF16", (2,)),
        "model.layers.0.self_attn.v_proj.bias": ("BF16", (2,)),
        "model.layers.0.self_attn.o_proj.bias": ("BF16", (4,)),
    }
    config_update: dict[str, object] = {"attention_bias": True}
    if family == "llama":
        config_update["mlp_bias"] = True
        bias_tensors.update(
            {
                "model.layers.0.mlp.gate_proj.bias": ("BF16", (8,)),
                "model.layers.0.mlp.up_proj.bias": ("BF16", (8,)),
                "model.layers.0.mlp.down_proj.bias": ("BF16", (4,)),
            }
        )
    root = tmp_path / family
    tensors = _write_model(
        root,
        family,
        tied=False,
        config_update=config_update,
        tensor_update=bias_tensors,
    )
    result = decompile_source(root)

    assert result.succeeded
    assert result.report.coverage is not None
    assert result.report.coverage.classified_tensor_count == len(tensors)
    assert result.ir_bundle is not None
    logical_names = {view.logical_name for view in result.ir_bundle.physical_weights.views}
    assert "layers.0.attention.o_proj.bias" in logical_names
    if family == "llama":
        assert "layers.0.mlp.down_proj.bias" in logical_names


def test_llama_identity_markers_are_validated_not_rejected_as_unknown(tmp_path: Path) -> None:
    root = tmp_path / "llama-markers"
    _write_model(
        root,
        "llama",
        tied=False,
        config_update={"is_llama_config": True, "rope_interleaved": False},
    )
    assert decompile_source(root).succeeded

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["rope_interleaved"] = True
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    result = decompile_source(root)
    assert not result.succeeded
    assert any(
        feature.code == "invalid_semantic_config"
        for match in result.report.match_results
        for feature in match.unsupported_features
    )


def test_padded_lexical_rows_and_serialized_rotary_buffers_are_explicit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    _write_model(
        root,
        "llama",
        tied=False,
        tensor_update={
            "model.embed_tokens.weight": ("BF16", (10, 4)),
            "lm_head.weight": ("BF16", (10, 4)),
            "model.rotary_emb.inv_freq": ("F32", (1,)),
            "model.rotary_emb.cos_cached": ("F32", (64, 2)),
        },
    )
    result = decompile_source(root)

    assert result.succeeded
    assert result.ir_bundle is not None
    io = result.ir_bundle.io
    assert {mapper.kind for mapper in io.row_mappers} == {"padded-identity"}
    assert all(mapper.unreachable_rows == (8, 9) for mapper in io.row_mappers)
    classifications = {
        item.source_name: item for item in result.ir_bundle.physical_weights.classifications
    }
    assert classifications["model.rotary_emb.inv_freq"].disposition == "buffer"
    assert classifications["model.rotary_emb.cos_cached"].disposition == "ignored"
    rotary = next(
        operation
        for operation in result.ir_bundle.model.operations
        if operation.operation_id == "layers.0.attention.rotary"
    )
    assert rotary.parameters == ("rotary.inv_freq",)


def test_source_and_tensor_identities_are_path_independent(tmp_path: Path) -> None:
    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    _write_model(first_root, "qwen3", tied=False)
    shutil.copytree(first_root, second_root)

    first = freeze_source(first_root, source_id="Test/Portable", resolved_revision="b" * 40)
    second = freeze_source(second_root, source_id="Test/Portable", resolved_revision="b" * 40)
    first_index = SafetensorsTensorIndexer().build(first)
    second_index = SafetensorsTensorIndexer().build(second)

    assert first.fingerprint == second.fingerprint
    assert first.as_dict() == second.as_dict()
    assert first_index.fingerprint == second_index.fingerprint
    assert first_index.as_dict() == second_index.as_dict()
    detached = FrozenSourceBundle.from_dict(first.as_dict())
    assert detached.as_dict() == first.as_dict()
    assert detached.attach_root(second_root).as_dict() == first.as_dict()
    assert TensorIndex.from_dict(first_index.as_dict()).as_dict() == first_index.as_dict()


def test_hf_local_dir_transport_metadata_is_not_a_model_asset(tmp_path: Path) -> None:
    root = tmp_path / "local-dir"
    _write_model(root, "qwen3", tied=False)
    metadata = root / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / ".gitattributes.lock").write_bytes(b"")
    (metadata / "config.json.metadata").write_text(
        f"{'c' * 40}\nignored-etag\nignored-timestamp\n", encoding="utf-8"
    )

    supplied = freeze_source(
        root,
        source_id="Test/LocalDirQwen3",
        resolved_revision="c" * 40,
        policy=SourcePolicy(require_immutable_revision=True),
    )
    discovered = freeze_source(
        root,
        source_id="Test/LocalDirQwen3",
        policy=SourcePolicy(require_immutable_revision=True),
    )
    decoded = decompile_source(
        root,
        source_id="Test/LocalDirQwen3",
        resolved_revision="c" * 40,
        policy=SourcePolicy(require_immutable_revision=True),
    )

    assert supplied.resolved_revision == discovered.resolved_revision == "c" * 40
    assert supplied.fingerprint == discovered.fingerprint
    assert all(not item.path.startswith(".cache/") for item in supplied.files)
    assert decoded.succeeded

    (metadata / ".gitattributes.lock").write_bytes(b"ambient transport state changed")
    supplied.assert_unchanged()


def test_immutable_huggingface_snapshot_symlinks_are_frozen_by_content(
    tmp_path: Path,
) -> None:
    materialized = tmp_path / "materialized"
    _write_model(materialized, "qwen2", tied=True)
    repository = tmp_path / "models--Test--TinyQwen"
    blobs = repository / "blobs"
    snapshot = repository / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for index, source_path in enumerate(sorted(materialized.iterdir())):
        content = source_path.read_bytes()
        blob_name = f"{index + 1:040x}"
        blob_path = blobs / blob_name
        blob_path.write_bytes(content)
        (snapshot / source_path.name).symlink_to(Path("../../blobs") / blob_name)

    frozen = freeze_source(
        snapshot,
        source_id="Test/TinyQwen",
        policy=SourcePolicy(require_immutable_revision=True),
    )
    decoded = decompile_source(
        snapshot,
        source_id="Test/TinyQwen",
        policy=SourcePolicy(require_immutable_revision=True),
    )

    assert frozen.resolved_revision == "a" * 40
    assert frozen.revision_immutable
    assert frozen.file_path("model.safetensors").parent == blobs
    assert decoded.succeeded
    frozen.assert_unchanged()

    tokenizer_link = snapshot / "tokenizer.json"
    original_target = tokenizer_link.readlink()
    replacement = blobs / ("f" * 40)
    replacement.write_bytes((snapshot / "tokenizer.json").read_bytes())
    tokenizer_link.unlink()
    tokenizer_link.symlink_to(Path("../../blobs") / replacement.name)
    with pytest.raises(SourceMutationError, match="filesystem identity changed"):
        frozen.assert_unchanged()
    tokenizer_link.unlink()
    tokenizer_link.symlink_to(original_target)


def test_huggingface_snapshot_symlink_must_target_its_blob_store(tmp_path: Path) -> None:
    materialized = tmp_path / "materialized"
    _write_model(materialized, "qwen2", tied=True)
    repository = tmp_path / "models--Test--TinyQwen"
    snapshot = repository / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    outside = tmp_path / ("c" * 40)
    outside.write_bytes(b"not a model asset")
    for source_path in materialized.iterdir():
        destination = snapshot / source_path.name
        if source_path.name == "tokenizer.json":
            destination.symlink_to(outside)
        else:
            destination.write_bytes(source_path.read_bytes())

    with pytest.raises(SourceCustodyError, match="non-symlink|escapes"):
        freeze_source(snapshot)


def test_tensor_index_reads_headers_not_tensor_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "model"
    _write_model(root, "llama", tied=False)
    source = freeze_source(root)
    import mrun.decompiler.tensor_index as tensor_index_module

    reads: list[tuple[int, int]] = []
    original = tensor_index_module._read_exact_at

    def observed(descriptor: int, offset: int, length: int, *, field: str) -> bytes:
        reads.append((offset, length))
        return original(descriptor, offset, length, field=field)

    monkeypatch.setattr(tensor_index_module, "_read_exact_at", observed)
    index = SafetensorsTensorIndexer().build(source)
    shard = index.shards[0]

    assert reads == [(0, 8), (8, shard.header_bytes)]
    assert sum(length for _, length in reads) == shard.data_offset
    assert shard.data_bytes == index.total_tensor_bytes


def test_source_mutation_is_rejected_after_freeze(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _write_model(root, "llama", tied=False)
    source = freeze_source(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["hidden_size"] = 6
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SourceMutationError, match="changed after freeze"):
        source.assert_unchanged()


def test_symlink_pickle_and_remote_code_policies_fail_closed(tmp_path: Path) -> None:
    symlink_root = tmp_path / "symlink"
    _write_model(symlink_root, "llama", tied=False)
    (symlink_root / "linked-tokenizer.json").symlink_to(symlink_root / "tokenizer.json")
    with pytest.raises(SourceCustodyError, match="non-symlink"):
        freeze_source(symlink_root)

    pickle_root = tmp_path / "pickle"
    _write_model(pickle_root, "llama", tied=False)
    (pickle_root / "pytorch_model.bin").write_bytes(b"not-pickle")
    with pytest.raises(SourcePolicyError, match="pickle"):
        freeze_source(pickle_root)

    code_root = tmp_path / "code"
    _write_model(code_root, "llama", tied=False)
    (code_root / "modeling_custom.py").write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(SourcePolicyError, match="custom code"):
        freeze_source(code_root)

    auto_root = tmp_path / "auto"
    _write_model(
        auto_root,
        "llama",
        tied=False,
        config_update={"auto_map": {"AutoModel": "modeling_custom.CustomModel"}},
    )
    with pytest.raises(SourcePolicyError, match="auto_map"):
        freeze_source(auto_root)


def test_promotion_policy_requires_immutable_revision(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _write_model(root, "llama", tied=False)
    with pytest.raises(SourcePolicyError, match="immutable"):
        freeze_source(
            root,
            resolved_revision="main",
            policy=SourcePolicy(require_immutable_revision=True),
        )


def test_sharded_index_requires_exact_shard_and_tensor_ownership(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_config("llama", tied=False)), encoding="utf-8")
    all_tensors = _tensors("llama", tied=False)
    names = sorted(all_tensors)
    first = {name: all_tensors[name] for name in names[::2]}
    second = {name: all_tensors[name] for name in names[1::2]}
    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    _write_safetensors(root / first_name, first)
    _write_safetensors(root / second_name, second)
    weight_map = {name: first_name for name in first} | {name: second_name for name in second}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}, sort_keys=True),
        encoding="utf-8",
    )
    source = freeze_source(root)
    index = SafetensorsTensorIndexer().build(source)
    assert len(index.shards) == 2
    assert {record.source_name for record in index.tensors} == set(all_tensors)

    wrong = deepcopy(weight_map)
    wrong[names[0]] = second_name
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wrong}, sort_keys=True), encoding="utf-8"
    )
    changed = freeze_source(root)
    with pytest.raises(TensorIndexError, match="wrong shard"):
        SafetensorsTensorIndexer().build(changed)


def test_index_rejects_missing_shards_multiple_files_without_index_and_range_gaps(
    tmp_path: Path,
) -> None:
    no_index = tmp_path / "no-index"
    _write_model(no_index, "llama", tied=False)
    shutil.copyfile(no_index / "model.safetensors", no_index / "extra.safetensors")
    source = freeze_source(no_index)
    with pytest.raises(TensorIndexError, match="require one coherent index"):
        SafetensorsTensorIndexer().build(source)

    missing = tmp_path / "missing"
    _write_model(missing, "llama", tied=False)
    (missing / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"model.embed_tokens.weight": "missing-00002.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    missing_source = freeze_source(missing)
    with pytest.raises(TensorIndexError, match="index/shard inventory mismatch"):
        SafetensorsTensorIndexer().build(missing_source)

    gap = tmp_path / "gap"
    gap.mkdir()
    (gap / "config.json").write_text(json.dumps(_config("llama", tied=False)), encoding="utf-8")
    tensors = {"a": ("BF16", (1,)), "b": ("BF16", (1,))}
    _write_safetensors(
        gap / "model.safetensors",
        tensors,
        ranges={"a": (0, 2), "b": (4, 6)},
    )
    gap_source = freeze_source(gap)
    with pytest.raises(TensorIndexError, match="gap"):
        SafetensorsTensorIndexer().build(gap_source)


@pytest.mark.parametrize(
    ("mutation", "expected_code", "expected_gate"),
    [
        ("unknown_tensor", "source_coverage_failure", "G2"),
        ("missing_projection", "source_coverage_failure", "G2"),
        ("missing_untied_head", "alias_evidence_failure", "G4"),
        ("sliding_window", "unsupported_variant", "G1"),
        ("unknown_config", "unsupported_variant", "G1"),
        ("packed_dtype", "unsupported_variant", "G1"),
        ("shape_mismatch", "semantic_configuration_failure", "G3"),
        ("tied_duplicate_head", "alias_evidence_failure", "G4"),
        ("mrope", "unsupported_variant", "G1"),
    ],
)
def test_semantic_negative_corpus_is_structured_and_fail_closed(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
    expected_gate: str,
) -> None:
    root = tmp_path / mutation
    config_update: dict[str, object] = {}
    tensor_update: dict[str, tuple[str, tuple[int, ...]]] = {}
    remove: tuple[str, ...] = ()
    if mutation == "unknown_tensor":
        tensor_update["model.layers.0.attention_magic.weight"] = ("BF16", (4, 4))
    elif mutation == "missing_projection":
        remove = ("model.layers.0.self_attn.k_proj.weight",)
    elif mutation == "missing_untied_head":
        remove = ("lm_head.weight",)
    elif mutation == "sliding_window":
        config_update.update(
            {"use_sliding_window": True, "sliding_window": 32, "layer_types": ["sliding_attention"]}
        )
    elif mutation == "unknown_config":
        config_update["mystery_attention_policy"] = "surprise"
    elif mutation == "packed_dtype":
        tensor_update["model.layers.0.self_attn.q_proj.weight"] = ("I32", (4, 4))
    elif mutation == "shape_mismatch":
        tensor_update["model.layers.0.self_attn.q_proj.weight"] = ("BF16", (6, 4))
    elif mutation == "tied_duplicate_head":
        config_update["tie_word_embeddings"] = True
    elif mutation == "mrope":
        config_update["use_mrope"] = True
    _write_model(
        root,
        "llama",
        tied=False,
        config_update=config_update,
        tensor_update=tensor_update,
        remove_tensors=remove,
    )

    result = decompile_source(root)

    assert not result.succeeded
    assert result.ir_bundle is None
    assert result.report.status == "unsupported"
    assert result.report.failures[0].code == expected_code
    assert result.report.failures[0].gate == expected_gate
    assert result.report.fingerprint == result.report.fingerprint


def test_unknown_architecture_and_ambiguous_adapters_are_stable(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown"
    _write_model(
        unknown,
        "llama",
        tied=False,
        config_update={"model_type": "future_decoder", "architectures": ["FutureDecoder"]},
    )
    no_match = decompile_source(unknown)
    assert no_match.report.failures[0].code == "unsupported_architecture"
    assert no_match.report.universal_level == "U1"

    class FirstQwen2(Qwen2Adapter):
        adapter_id = "test.qwen2.first"

    class SecondQwen2(Qwen2Adapter):
        adapter_id = "test.qwen2.second"

    qwen = tmp_path / "ambiguous"
    _write_model(qwen, "qwen2", tied=True)
    first_registry = AdapterRegistry((FirstQwen2(), SecondQwen2()))
    second_registry = AdapterRegistry((SecondQwen2(), FirstQwen2()))
    first = decompile_source(qwen, registry=first_registry)
    second = decompile_source(qwen, registry=second_registry)
    assert first.report.failures[0].code == "ambiguous_adapter"
    assert first.report.as_dict() == second.report.as_dict()


def test_round_trip_rejects_unknown_fields_and_fingerprint_tamper(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _write_model(root, "qwen3", tied=False)
    result = decompile_source(root)
    assert result.succeeded

    source_payload = result.source.as_dict() if result.source is not None else {}
    source_payload["newer_unknown_field"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        FrozenSourceBundle.from_dict(source_payload)

    report_payload = result.report.as_dict()
    report_payload["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        type(result.report).from_dict(report_payload)


def test_deserialized_index_and_physical_ir_revalidate_internal_ranges_and_views(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    _write_model(root, "llama", tied=False)
    result = decompile_source(root)
    assert result.succeeded and result.tensor_index is not None and result.ir_bundle is not None

    index_payload = result.tensor_index.as_dict()
    index_payload["tensors"][0]["byte_offset"] += 1
    index_identity = {key: value for key, value in index_payload.items() if key != "fingerprint"}
    index_payload["fingerprint"] = canonical_sha256(index_identity)
    with pytest.raises(ValueError, match="range identity|gap or overlap"):
        TensorIndex.from_dict(index_payload)

    physical_payload = result.ir_bundle.physical_weights.as_dict()
    physical_payload["views"][0]["logical_shape"] = [999]
    physical_identity = {
        key: value for key, value in physical_payload.items() if key != "fingerprint"
    }
    physical_payload["fingerprint"] = canonical_sha256(physical_identity)
    with pytest.raises(IRValidationError, match="view shape"):
        PhysicalWeightIR.from_dict(physical_payload)


def test_canonical_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    from mrun.decompiler._json import strict_json_loads

    with pytest.raises(ValueError, match="duplicate key"):
        strict_json_loads('{"a":1,"a":2}', field="test")
    with pytest.raises(ValueError, match="non-finite"):
        strict_json_loads('{"a":NaN}', field="test")
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
