from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mrun.decompiler import build_component_artifact
from mrun.decompiler.cli import main as decompiler_main
from mrun.decompiler.cuda_native import (
    SOURCE_CUDA_INT8_BUILDER_ABI,
    SOURCE_CUDA_INT8_CODEC,
    SOURCE_CUDA_INT8_NATIVE_SCHEMA,
    SOURCE_CUDA_INT8_NUMERICAL_CONTRACT,
    SourceCudaInt8ArtifactError,
    SourceCudaInt8LoweringError,
    VerifiedSourceCudaInt8Artifact,
    build_source_cuda_int8_artifact,
)
from mrun.engine.dense_qstore_cuda import DenseQStoreTarget
from mrun.engine.kernels.source_cuda_int8 import DirectSourceCudaInt8Store
from mrun.runtime.native_backends import compiled_identity_from_dense_cuda_engine


def _config(*, family: str = "qwen2", tied: bool = True) -> dict[str, object]:
    architecture = {
        "qwen2": "Qwen2ForCausalLM",
        "llama": "LlamaForCausalLM",
    }[family]
    config: dict[str, object] = {
        "architectures": [architecture],
        "model_type": family,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 11,
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
    if family == "llama":
        config.update({"attention_bias": False, "mlp_bias": False, "pretraining_tp": 1})
    return config


def _write_model(
    root: Path, *, family: str = "qwen2", tied: bool = True
) -> dict[str, torch.Tensor]:
    root.mkdir()
    generator = torch.Generator().manual_seed(8128)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.05

    prefix = "model.layers.0"
    weights = {
        "model.embed_tokens.weight": random(11, 8),
        "model.norm.weight": torch.ones(8),
        f"{prefix}.input_layernorm.weight": torch.ones(8),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(8),
        f"{prefix}.self_attn.q_proj.weight": random(8, 8),
        f"{prefix}.self_attn.k_proj.weight": random(4, 8),
        f"{prefix}.self_attn.v_proj.weight": random(4, 8),
        f"{prefix}.self_attn.o_proj.weight": random(8, 8),
        f"{prefix}.mlp.gate_proj.weight": random(16, 8),
        f"{prefix}.mlp.up_proj.weight": random(16, 8),
        f"{prefix}.mlp.down_proj.weight": random(8, 16),
    }
    if family == "qwen2":
        weights.update(
            {
                f"{prefix}.self_attn.q_proj.bias": random(8),
                f"{prefix}.self_attn.k_proj.bias": random(4),
                f"{prefix}.self_attn.v_proj.bias": random(4),
            }
        )
    if not tied:
        weights["lm_head.weight"] = random(11, 8)
    (root / "config.json").write_text(
        json.dumps(_config(family=family, tied=tied), sort_keys=True), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, sort_keys=True), encoding="utf-8"
    )
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    return weights


def _build(
    tmp_path: Path,
) -> tuple[dict[str, torch.Tensor], Path, VerifiedSourceCudaInt8Artifact]:
    weights = _write_model(tmp_path / "source")
    canonical = build_component_artifact(
        tmp_path / "source",
        tmp_path / "canonical",
        source_id="Test/Qwen2",
        resolved_revision="d" * 40,
    )
    built = build_source_cuda_int8_artifact(canonical.path, tmp_path / "cuda")
    artifact = VerifiedSourceCudaInt8Artifact(
        built.path, source_artifact=canonical.path, verify_quantized_values=True
    )
    return weights, canonical.path, artifact


def test_direct_cuda_int8_is_role_separated_alias_preserving_and_source_exact(
    tmp_path: Path,
) -> None:
    source_weights, canonical_path, artifact = _build(tmp_path)
    manifest = artifact.manifest
    assert manifest["schema"] == SOURCE_CUDA_INT8_NATIVE_SCHEMA
    assert manifest["recipe"]["builder_abi"] == SOURCE_CUDA_INT8_BUILDER_ABI
    assert manifest["recipe"]["codec"] == SOURCE_CUDA_INT8_CODEC
    assert manifest["numerical_contract"] == SOURCE_CUDA_INT8_NUMERICAL_CONTRACT
    assert manifest["source"]["direct_from_canonical_source"] is True
    assert manifest["source"]["intermediate_qstore"] is False
    assert set(artifact.components) == {"body", "norm", "lexical_shared"}
    assert artifact.blocks["lm_head"]["alias"] == "embed"
    assert artifact.blocks["embed"]["logical_names"] == [
        "lm_head.weight",
        "token_embedding.weight",
    ]
    assert manifest["coverage"]["source_allocation_count"] == len(source_weights)
    assert manifest["coverage"]["all_blob_bytes_covered_once"] is True
    assert artifact.physical_bytes < sum(value.numel() * 4 for value in source_weights.values())

    embed = artifact.blocks["embed"]
    role = embed["role"]
    weight_path = artifact.path / artifact.components[role]["blobs"]["weights.i8"]["path"]
    scale_path = artifact.path / artifact.components[role]["blobs"]["scales.f32"]["path"]
    codes = np.memmap(weight_path, mode="r", dtype=np.int8)
    scales = np.memmap(scale_path, mode="r", dtype=np.float32)
    source = source_weights["model.embed_tokens.weight"]
    expected_scales = source.abs().amax(dim=1) / 127.0
    expected_codes = torch.round(source / expected_scales[:, None]).clamp(-127, 127).to(torch.int8)
    start = embed["w_off"]
    scale_start = embed["s_off"] // 4
    assert np.array_equal(
        np.asarray(codes[start : start + source.numel()]).reshape(source.shape),
        expected_codes.numpy(),
    )
    assert np.array_equal(
        np.asarray(scales[scale_start : scale_start + source.shape[0]]),
        expected_scales.numpy(),
    )

    repeated = build_source_cuda_int8_artifact(canonical_path, tmp_path / "cuda")
    assert repeated.path == artifact.path
    assert repeated.artifact_sha256 == artifact.artifact_sha256
    assert repeated.intermediate_qstore is False
    assert repeated.production_runtime_eligible is False


def test_direct_cuda_store_reuses_kernels_without_qstore_artifact(tmp_path: Path) -> None:
    _source_weights, canonical_path, artifact = _build(tmp_path)
    store = DirectSourceCudaInt8Store(
        artifact,
        source_artifact=canonical_path,
        device="cpu",
        compute_dtype="bf16",
        compact_cache_mb=1.0,
        pin_fp32_aux=True,
        require_triton=False,
    )
    try:
        store.prepare_fully_resident()
        assert store.compact_page("embed") is store.compact_page("lm_head")
        assert store.fp32("norm.final").dtype == torch.float32
        assert store.snapshot()["fully_resident"] is True
        assert store.snapshot()["intermediate_qstore"] is False
        target = DenseQStoreTarget(
            store,
            max_seq_len=16,
            semantic_token_count=11,
        )
        result = target.forward(torch.tensor([1, 2]), target.empty_cache(1), return_logits=True)
        assert result.hidden.shape == (1, 2, 8)
        assert result.logits is not None and result.logits.shape == (1, 2, 11)
        assert result.top1.shape == (1, 2)
        page = store.compact_page("lm_head")
        start, stop, weights = next(store.row_blocks("lm_head", bs=page.out_features))
        assert (start, stop) == (0, page.out_features)
        assert weights.dtype == torch.float32
        torch.testing.assert_close(
            weights,
            page.codes.float() * page.scales[:, None],
            rtol=0.0,
            atol=0.0,
        )
        assert torch.count_nonzero(weights != weights.to(torch.bfloat16).float()) > 0
        assert not (artifact.path / "weights.i8").exists()
        assert not (artifact.path / "qstore.json").exists()
    finally:
        store.close()


def test_direct_cuda_artifact_rejects_blob_tamper_and_resigned_lineage(tmp_path: Path) -> None:
    _weights, canonical_path, artifact = _build(tmp_path)
    first = next(
        artifact.path / record["path"]
        for component in artifact.components.values()
        for record in component["blobs"].values()
    )
    payload = bytearray(first.read_bytes())
    payload[-1] ^= 1
    first.write_bytes(payload)
    with pytest.raises(SourceCudaInt8ArtifactError, match="hash"):
        VerifiedSourceCudaInt8Artifact(artifact.path, source_artifact=canonical_path)

    second_root = tmp_path / "second"
    second_root.mkdir()
    _weights, _canonical, fresh = _build(second_root)
    manifest_path = fresh.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["tokenizer_custody_sha256"] = "0" * 64
    manifest.pop("artifact_sha256")
    manifest["artifact_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(SourceCudaInt8ArtifactError, match="lineage"):
        VerifiedSourceCudaInt8Artifact(fresh.path)


def test_direct_cuda_cli_and_compiled_identity_have_no_qstore_lineage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_model(tmp_path / "source")
    canonical = build_component_artifact(tmp_path / "source", tmp_path / "canonical")
    assert (
        decompiler_main(
            [
                "lower-cuda-int8",
                str(canonical.path),
                "--output-root",
                str(tmp_path / "cuda"),
            ]
        )
        == 0
    )
    lowered = json.loads(capsys.readouterr().out)
    assert lowered["build"]["intermediate_qstore"] is False
    assert (
        decompiler_main(
            [
                "verify-cuda-int8",
                lowered["build"]["path"],
                "--source-artifact",
                str(canonical.path),
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["artifact_sha256"] == lowered["build"]["artifact_sha256"]

    artifact = VerifiedSourceCudaInt8Artifact(
        lowered["build"]["path"], source_artifact=canonical.path
    )
    store = DirectSourceCudaInt8Store(
        artifact,
        source_artifact=canonical.path,
        device="cpu",
        compact_cache_mb=1.0,
        require_triton=False,
    )
    engine = SimpleNamespace(
        direct_artifact=artifact,
        store=store,
        cfg=store.cfg,
        name="qwen2.5-0.5b-instruct",
        arch="qwen2",
        max_seq_len=32,
        semantic_token_count=11,
        assert_content_identity_unchanged=store.assert_content_identity_unchanged,
    )
    try:
        identity = compiled_identity_from_dense_cuda_engine(engine)
    finally:
        store.close()
    assert identity.component_graph_sha256 == artifact.artifact_sha256
    assert all(component.codec_id == SOURCE_CUDA_INT8_CODEC for component in identity.components)
    assert "direct-no-qstore" in identity.compiler_abi
    assert "qstore-component" not in identity.compiler_abi


def test_direct_cuda_v1_rejects_non_qwen2_source(tmp_path: Path) -> None:
    _write_model(tmp_path / "llama", family="llama", tied=False)
    canonical = build_component_artifact(tmp_path / "llama", tmp_path / "canonical")
    with pytest.raises(SourceCudaInt8LoweringError, match="closed to dense Qwen2"):
        build_source_cuda_int8_artifact(canonical.path, tmp_path / "cuda")
