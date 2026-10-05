from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mrun.engine import olmoe_cuda
from mrun.engine.kernels import qstore_build
from mrun.engine.kernels.qstore import QStore, inspect_qstore_manifest_identity
from mrun.store_provenance import (
    SEMANTIC_DERIVED_SCHEMA,
    build_derived_provenance,
    build_source_provenance,
    verify_source_provenance,
)


def test_qstore_runtime_resolves_validated_alias_chains_and_rejects_cycles() -> None:
    store = QStore.__new__(QStore)
    store.blocks = {
        "head": {"alias": "tied"},
        "tied": {"alias": "embed"},
        "embed": {"kind": "qrow", "shape": [2, 2]},
    }
    assert store._resolve("head") is store.blocks["embed"]

    store.blocks["embed"] = {"alias": "head"}
    with pytest.raises(RuntimeError, match="cyclic QStore alias"):
        store._resolve("head")


def _tiny_qwen_checkpoint(
    root: Path,
    *,
    tied: bool = True,
    serialize_lm_head: bool = False,
    mismatched_lm_head: bool = False,
) -> Path:
    root.mkdir(parents=True)
    config = {
        "model_type": "qwen2",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 8,
        "vocab_size": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": tied,
    }
    embedding = torch.randn(8, 4)
    tensors = {
        "model.embed_tokens.weight": embedding,
        "model.norm.weight": torch.randn(4),
        "model.layers.0.input_layernorm.weight": torch.randn(4),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(4),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 4),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 4),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 4),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4),
        "model.layers.0.mlp.up_proj.weight": torch.randn(8, 4),
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 8),
    }
    if serialize_lm_head:
        lm_head = embedding.clone()
        if mismatched_lm_head:
            lm_head[0, 0] += 1
        tensors["lm_head.weight"] = lm_head
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(tensors, root / "model.safetensors")
    return root


def _patch_qwen_resolution(monkeypatch: pytest.MonkeyPatch, checkpoint: Path) -> None:
    shard = checkpoint / "model.safetensors"
    monkeypatch.setattr(qstore_build, "find_safetensors", lambda _name: [shard])
    monkeypatch.setattr(
        qstore_build,
        "resolve_model",
        lambda _name: SimpleNamespace(name="tiny-qwen", hf_id="Test/TinyQwen"),
    )
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda _path: SimpleNamespace(model_type="qwen2")),
    )


def _tiny_olmoe_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True)
    config = {
        "model_type": "olmoe",
        "num_hidden_layers": 1,
        "num_experts": 2,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_experts_per_tok": 1,
    }
    tensors = {}
    for expert in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        tensors[f"{prefix}.gate_proj.weight"] = torch.randn(8, 4)
        tensors[f"{prefix}.up_proj.weight"] = torch.randn(8, 4)
        tensors[f"{prefix}.down_proj.weight"] = torch.randn(4, 8)
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, root / shard_name)
    index = {"metadata": {}, "weight_map": {name: shard_name for name in tensors}}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def _flip_first_byte(path: Path) -> None:
    with path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))


def _resign_qstore_manifest(store: Path, manifest: dict) -> None:
    semantic_manifest = {key: value for key, value in manifest.items() if key != "derived"}
    manifest["derived"] = build_derived_provenance(
        store,
        qstore_build.QSTORE_FILES,
        semantic_manifest=semantic_manifest,
    )
    (store / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _built_tiny_qstore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    checkpoint = _tiny_qwen_checkpoint(tmp_path / "model")
    _patch_qwen_resolution(monkeypatch, checkpoint)
    stores = tmp_path / "stores"
    return stores, qstore_build.build("tiny", out_root=stores, store_dir_name="Tiny")


def test_checkpoint_hash_binds_config_index_and_every_shard(tmp_path: Path) -> None:
    checkpoint = _tiny_olmoe_checkpoint(tmp_path / "model")
    shard = checkpoint / "model-00001-of-00001.safetensors"
    first = build_source_provenance(
        checkpoint,
        [shard],
        model_name="tiny-olmoe",
        hf_id="Test/TinyOLMoE",
        revision="abc123",
    )
    _flip_first_byte(shard)
    second = build_source_provenance(
        checkpoint,
        [shard],
        model_name="tiny-olmoe",
        hf_id="Test/TinyOLMoE",
        revision="abc123",
    )

    assert first["source_checkpoint_sha256"] != second["source_checkpoint_sha256"]
    assert first["config"]["sha256"] == second["config"]["sha256"]
    assert first["index"]["sha256"] == second["index"]["sha256"]
    assert first["safetensors"][0]["sha256"] != second["safetensors"][0]["sha256"]


def test_source_verification_is_portable_across_revision_discovery(
    tmp_path: Path,
) -> None:
    checkpoint = _tiny_qwen_checkpoint(tmp_path / "model")
    shard = checkpoint / "model.safetensors"
    source = build_source_provenance(
        checkpoint,
        [shard],
        model_name="tiny-qwen",
        hf_id="Test/TinyQwen",
        revision="abc123",
    )
    same_content_from_direct_directory = deepcopy(source)
    same_content_from_direct_directory["revision"] = {
        "kind": "unknown",
        "commits": [],
    }

    verify_source_provenance(source, same_content_from_direct_directory)


def test_qstore_build_is_atomic_content_bound_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _tiny_qwen_checkpoint(tmp_path / "model")
    _patch_qwen_resolution(monkeypatch, checkpoint)
    stores = tmp_path / "stores"
    store = qstore_build.build("tiny", out_root=stores, store_dir_name="Tiny")
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == qstore_build.QSTORE_SCHEMA
    assert manifest["source"]["hf_id"] == "Test/TinyQwen"
    assert len(manifest["source"]["safetensors"]) == 1
    assert manifest["builder"]["quantization"] == qstore_build.QSTORE_QUANTIZATION
    assert {record["name"] for record in manifest["builder"]["source_files"]} == {
        "mrun/engine/kernels/qstore_build.py",
        "mrun/store_provenance.py",
    }
    assert len(manifest["builder"]["source_bundle_sha256"]) == 64
    assert {record["name"] for record in manifest["derived"]["files"]} == set(
        qstore_build.QSTORE_FILES
    )
    assert manifest["derived"]["schema_version"] == SEMANTIC_DERIVED_SCHEMA
    assert len(manifest["derived"]["manifest_semantic_sha256"]) == 64
    preflight = inspect_qstore_manifest_identity(manifest)
    assert preflight["semantic_identity_verified"]
    assert not preflight["content_identity_verified"]
    assert not preflight["blob_identity_verified"]
    assert qstore_build.build("tiny", out_root=stores, store_dir_name="Tiny") == store

    _flip_first_byte(store / "weights.i8")
    with pytest.raises(RuntimeError, match="derived-file hash mismatch"):
        qstore_build.build("tiny", out_root=stores, store_dir_name="Tiny")


def test_tied_qstore_deduplicates_only_with_exact_source_and_encoded_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _tiny_qwen_checkpoint(
        tmp_path / "tied-model",
        tied=True,
        serialize_lm_head=True,
    )
    _patch_qwen_resolution(monkeypatch, checkpoint)
    stores = tmp_path / "stores"
    tied_store = qstore_build.build("tiny", out_root=stores, store_dir_name="Tied")
    tied_manifest = json.loads((tied_store / "manifest.json").read_text(encoding="utf-8"))

    assert tied_manifest["tie_word_embeddings"] is True
    assert tied_manifest["blocks"]["lm_head"] == {"alias": "embed"}
    binding = tied_manifest["lexical_weight_binding"]
    assert binding["schema_version"] == qstore_build.LEXICAL_BINDING_SCHEMA
    assert binding["disposition"] == "verified-duplicate-alias"
    assert binding["source_proof"]["kind"] == "shape-dtype-byte-length-sha256-identity"
    assert binding["encoded_proof"]["kind"] == "shape-dtype-byte-length-sha256-identity"
    assert binding["alias_saved_bytes"] == 8 * 4 + 8 * 4

    untied_checkpoint = _tiny_qwen_checkpoint(
        tmp_path / "untied-model",
        tied=False,
        serialize_lm_head=True,
    )
    _patch_qwen_resolution(monkeypatch, untied_checkpoint)
    untied_store = qstore_build.build("tiny", out_root=stores, store_dir_name="Untied")
    untied_manifest = json.loads(
        (untied_store / "manifest.json").read_text(encoding="utf-8")
    )

    assert untied_manifest["tie_word_embeddings"] is False
    assert untied_manifest["blocks"]["lm_head"]["kind"] == "qrow"
    assert untied_manifest["lexical_weight_binding"]["disposition"] == (
        "distinct-physical-blocks"
    )
    assert untied_manifest["lexical_weight_binding"]["alias_saved_bytes"] == 0
    tied_bytes = sum((tied_store / name).stat().st_size for name in qstore_build.QSTORE_FILES)
    untied_bytes = sum((untied_store / name).stat().st_size for name in qstore_build.QSTORE_FILES)
    assert untied_bytes - tied_bytes == binding["alias_saved_bytes"]


def test_tied_qstore_refuses_serialized_lexical_mismatch_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _tiny_qwen_checkpoint(
        tmp_path / "model",
        tied=True,
        serialize_lm_head=True,
        mismatched_lm_head=True,
    )
    _patch_qwen_resolution(monkeypatch, checkpoint)
    stores = tmp_path / "stores"

    with pytest.raises(RuntimeError, match="source tensors are not byte-identical"):
        qstore_build.build("tiny", out_root=stores, store_dir_name="Tiny")

    assert not (stores / "Tiny").exists()
    assert list(stores.glob(".Tiny.building-*")) == []


def test_qstore_refuses_to_infer_tying_without_config_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _tiny_qwen_checkpoint(tmp_path / "model", tied=False)
    _patch_qwen_resolution(monkeypatch, checkpoint)
    stores = tmp_path / "stores"

    with pytest.raises(RuntimeError, match="refusing to infer tied lexical weights"):
        qstore_build.build("tiny", out_root=stores, store_dir_name="Tiny")

    assert not (stores / "Tiny").exists()


def test_tied_qstore_refuses_codec_output_mismatch_even_for_equal_sources() -> None:
    binding = qstore_build.LexicalWeightBinding({"tie_word_embeddings": True})
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    binding.observe_source("embed", source)
    binding.observe_encoded(
        "embed",
        weights=np.zeros((2, 4), dtype=np.int8),
        scales=np.ones(2, dtype=np.float32),
    )
    binding.observe_source("lm_head", source.clone())
    assert not binding.observe_encoded(
        "lm_head",
        weights=np.ones((2, 4), dtype=np.int8),
        scales=np.ones(2, dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="encoded payloads are not byte-identical"):
        binding.finalize({"embed": {"kind": "qrow"}})


def test_qstore_v3_open_verifies_semantic_and_blob_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores, store = _built_tiny_qstore(tmp_path, monkeypatch)

    opened = QStore("Tiny", root=stores)

    assert opened.content_identity_verified
    assert opened.store_identity["blob_identity_verified"]
    assert opened.identity_status == "content-addressed-semantic-v2-files-verified"
    assert opened.source_checkpoint_sha256 == opened.man["source"]["source_checkpoint_sha256"]
    assert opened.derived_store_sha256 == opened.man["derived"]["derived_store_sha256"]
    assert opened.manifest_semantic_sha256 == opened.man["derived"]["manifest_semantic_sha256"]

    _flip_first_byte(store / "weights.i8")
    with pytest.raises(RuntimeError, match="derived-file hash mismatch"):
        QStore("Tiny", root=stores)


def test_open_qstore_fails_closed_if_verified_files_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores, store = _built_tiny_qstore(tmp_path, monkeypatch)
    opened = QStore("Tiny", root=stores)
    _flip_first_byte(store / "weights.i8")

    with pytest.raises(RuntimeError, match="files changed after content verification"):
        opened.assert_content_identity_unchanged()
    assert not opened.content_identity_verified
    assert not opened.store_identity["blob_identity_verified"]
    assert opened.identity_status == "verified-store-files-changed"

    with pytest.raises(RuntimeError, match="derived-file hash mismatch"):
        opened.reverify_content_identity()
    assert not opened.content_identity_verified
    assert not opened.store_identity["blob_identity_verified"]
    assert opened.identity_status == "fresh-content-verification-failed"


def test_qstore_v3_rejects_manifest_semantic_tamper_before_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores, store = _built_tiny_qstore(tmp_path, monkeypatch)
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["blocks"]["L0.q"]["w_off"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="semantic manifest digest mismatch"):
        QStore("Tiny", root=stores)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("alias-cycle", "alias cycle"),
        ("shape", "weight length does not match"),
        ("overlap", "ranges overlap"),
        ("bounds", "exceeds file bounds"),
    ),
)
def test_qstore_v3_rejects_resigned_invalid_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    stores, store = _built_tiny_qstore(tmp_path, monkeypatch)
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    qrows = [
        block
        for block in manifest["blocks"].values()
        if isinstance(block, dict) and block.get("kind") == "qrow"
    ]
    if mutation == "alias-cycle":
        manifest["blocks"]["lm_head"] = {"alias": "lm_head"}
    elif mutation == "shape":
        qrows[0]["shape"][1] += 1
    elif mutation == "overlap":
        qrows[1]["w_off"] = qrows[0]["w_off"]
    elif mutation == "bounds":
        max(qrows, key=lambda block: block["w_off"])["w_off"] = (
            store / "weights.i8"
        ).stat().st_size + 1
    else:  # pragma: no cover - guarded by the parametrization
        raise AssertionError(mutation)
    _resign_qstore_manifest(store, manifest)

    with pytest.raises(RuntimeError, match=message):
        QStore("Tiny", root=stores)


def test_qstore_v2_and_legacy_remain_readable_but_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores, store = _built_tiny_qstore(tmp_path, monkeypatch)
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "mrun-qstore-int8-v2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    opened = QStore("Tiny", root=stores)

    assert not opened.content_identity_verified
    assert opened.identity_status == "declared-content-only-unverified"
    assert opened.source_checkpoint_sha256 == manifest["source"]["source_checkpoint_sha256"]
    assert opened.derived_store_sha256 == manifest["derived"]["derived_store_sha256"]


def test_qstore_rejects_source_change_and_preserves_legacy_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _tiny_qwen_checkpoint(tmp_path / "model")
    _patch_qwen_resolution(monkeypatch, checkpoint)
    stores = tmp_path / "stores"
    store = qstore_build.build("tiny", out_root=stores, store_dir_name="Tiny")
    _flip_first_byte(checkpoint / "model.safetensors")
    with pytest.raises(RuntimeError, match="source provenance mismatch"):
        qstore_build.build("tiny", out_root=stores, store_dir_name="Tiny")

    legacy = stores / "Legacy"
    legacy.mkdir()
    marker = legacy / "do-not-overwrite"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete QStore"):
        qstore_build.build("tiny", out_root=stores, store_dir_name="Legacy")
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert store.is_dir()


def test_olmoe_store_binds_source_and_derived_data(tmp_path: Path) -> None:
    checkpoint = _tiny_olmoe_checkpoint(tmp_path / "model")
    store = tmp_path / "store"
    manifest = olmoe_cuda.build_fp8_store(
        checkpoint,
        store,
        model_name="tiny-olmoe",
        hf_id="Test/TinyOLMoE",
        revision="abc123",
    )

    assert manifest["schema_version"] == olmoe_cuda.STORE_SCHEMA
    assert manifest["source"]["hf_id"] == "Test/TinyOLMoE"
    assert manifest["derived"]["files"][0]["sha256"] == manifest["data_sha256"]
    assert (
        olmoe_cuda.build_fp8_store(
            checkpoint,
            store,
            model_name="tiny-olmoe",
            hf_id="Test/TinyOLMoE",
            revision="abc123",
        )["data_sha256"]
        == manifest["data_sha256"]
    )

    _flip_first_byte(store / "experts.fp8")
    with pytest.raises(RuntimeError, match="derived-file hash mismatch"):
        olmoe_cuda.build_fp8_store(
            checkpoint,
            store,
            model_name="tiny-olmoe",
            hf_id="Test/TinyOLMoE",
            revision="abc123",
        )


def test_olmoe_failed_build_is_atomic_and_legacy_directory_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _tiny_olmoe_checkpoint(tmp_path / "model")
    store = tmp_path / "store"

    def fail_quantization(_source: torch.Tensor):
        raise RuntimeError("synthetic quantization failure")

    monkeypatch.setattr(olmoe_cuda, "quantize_fp8_weight", fail_quantization)
    with pytest.raises(RuntimeError, match="synthetic quantization failure"):
        olmoe_cuda.build_fp8_store(
            checkpoint,
            store,
            model_name="tiny-olmoe",
            hf_id="Test/TinyOLMoE",
            revision="abc123",
        )
    assert not store.exists()
    assert list(tmp_path.glob(".store.building-*")) == []

    store.mkdir()
    marker = store / "do-not-overwrite"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete OLMoE store"):
        olmoe_cuda.build_fp8_store(
            checkpoint,
            store,
            model_name="tiny-olmoe",
            hf_id="Test/TinyOLMoE",
            revision="abc123",
        )
    assert marker.read_text(encoding="utf-8") == "preserve"
