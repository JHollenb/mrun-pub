from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mrun.decompiler import build_component_artifact
from mrun.decompiler.cli import main as decompiler_main
from mrun.decompiler.mlx_hybrid import (
    SOURCE_MLX_HYBRID_BF16_CODEC,
    SOURCE_MLX_HYBRID_BF16_NUMERICAL_CONTRACT,
    SOURCE_MLX_HYBRID_NATIVE_SCHEMA,
    SOURCE_MLX_HYBRID_Q8_CODEC,
    SOURCE_MLX_HYBRID_Q8_NUMERICAL_CONTRACT,
    MLXSourceHybridBF16Engine,
    MLXSourceHybridQ8Engine,
    VerifiedSourceMlxHybridBF16Artifact,
    VerifiedSourceMlxHybridQ8Artifact,
    build_source_mlx_hybrid_artifact,
)
from mrun.decompiler.mlx_native import SourceMlxArtifactError, SourceMlxLoweringError


def _write_qwen2(root: Path, *, tied: bool = True, dtype: torch.dtype = torch.bfloat16) -> None:
    hidden = 64
    intermediate = 128
    vocab = 128
    root.mkdir()
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
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
    generator = torch.Generator().manual_seed(144)

    def random(*shape: int) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * 0.05).to(dtype)

    prefix = "model.layers.0"
    weights = {
        "model.embed_tokens.weight": random(vocab, hidden),
        "model.norm.weight": torch.ones(hidden, dtype=dtype),
        f"{prefix}.input_layernorm.weight": torch.ones(hidden, dtype=dtype),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden, dtype=dtype),
        f"{prefix}.self_attn.q_proj.weight": random(hidden, hidden),
        f"{prefix}.self_attn.k_proj.weight": random(hidden // 2, hidden),
        f"{prefix}.self_attn.v_proj.weight": random(hidden // 2, hidden),
        f"{prefix}.self_attn.o_proj.weight": random(hidden, hidden),
        f"{prefix}.self_attn.q_proj.bias": random(hidden),
        f"{prefix}.self_attn.k_proj.bias": random(hidden // 2),
        f"{prefix}.self_attn.v_proj.bias": random(hidden // 2),
        f"{prefix}.mlp.gate_proj.weight": random(intermediate, hidden),
        f"{prefix}.mlp.up_proj.weight": random(intermediate, hidden),
        f"{prefix}.mlp.down_proj.weight": random(hidden, intermediate),
    }
    if not tied:
        weights["lm_head.weight"] = random(vocab, hidden)
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, sort_keys=True), encoding="utf-8"
    )
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})


def _canonical(tmp_path: Path, *, tied: bool = True, dtype: torch.dtype = torch.bfloat16):
    source = tmp_path / "source"
    _write_qwen2(source, tied=tied, dtype=dtype)
    return build_component_artifact(
        source,
        tmp_path / "canonical",
        source_id="Test/Qwen2-Hybrid",
        resolved_revision="d" * 40,
    )


@pytest.mark.parametrize(
    ("precision", "verifier", "codec", "contract", "lexical_encoding"),
    [
        (
            "q8",
            VerifiedSourceMlxHybridQ8Artifact,
            SOURCE_MLX_HYBRID_Q8_CODEC,
            SOURCE_MLX_HYBRID_Q8_NUMERICAL_CONTRACT,
            "mlx-affine-q8-g64",
        ),
        (
            "bf16",
            VerifiedSourceMlxHybridBF16Artifact,
            SOURCE_MLX_HYBRID_BF16_CODEC,
            SOURCE_MLX_HYBRID_BF16_NUMERICAL_CONTRACT,
            "source-exact-bf16",
        ),
    ],
)
def test_role_hybrid_artifact_is_explicit_alias_preserving_and_fail_closed(
    tmp_path: Path,
    precision: str,
    verifier: type,
    codec: str,
    contract: str,
    lexical_encoding: str,
) -> None:
    pytest.importorskip("mlx.core")
    canonical = _canonical(tmp_path)
    record = build_source_mlx_hybrid_artifact(
        canonical.path, tmp_path / "native", lexical_precision=precision
    )
    artifact = verifier(record.path)
    manifest = artifact.manifest
    assert manifest["schema"] == SOURCE_MLX_HYBRID_NATIVE_SCHEMA
    assert artifact.codec == codec
    assert artifact.numerical_contract == contract
    assert artifact.lexical_precision == precision
    assert artifact.body_bits == 4
    assert artifact.lexical_bits == (8 if precision == "q8" else 16)
    assert manifest["recipe"]["role_codecs"] == {
        "body": "mlx-affine-q4-g64",
        "lexical_shared": lexical_encoding,
        "norm": "source-exact-bf16",
    }
    assert {item["role"] for item in manifest["shards"]} == {
        "body",
        "lexical_shared",
        "norm",
    }
    assert not manifest["execution_certified"]
    assert not manifest["production_runtime_eligible"]
    assert manifest["source"]["tied_lexical_allocation"]
    assert manifest["source"]["direct_from_canonical_source"]
    assert not manifest["source"]["intermediate_qstore"]

    descriptors: list[dict[str, object]] = []
    for shard in manifest["shards"]:
        with safe_open(artifact.path / shard["filename"], framework="pt") as handle:
            for parameter in shard["parameters"]:
                assert parameter["name"] in handle.keys()
                descriptors.append(parameter)
    lexical = [
        item
        for item in descriptors
        if item["logical_names"] == ["lm_head.weight", "token_embedding.weight"]
    ]
    assert lexical
    assert {item["source_allocation_id"] for item in lexical}.__len__() == 1
    assert {item["encoding"] for item in lexical} == {lexical_encoding}
    assert all(
        item["encoding"] == "mlx-affine-q4-g64"
        for item in descriptors
        if item["source_tensor"].startswith("model.layers") and len(item["source_shape"]) == 2
    )
    assert all(
        item["encoding"] == "source-exact-bf16"
        for item in descriptors
        if item["source_tensor"].endswith("norm.weight")
        or "layernorm.weight" in item["source_tensor"]
    )
    repeated = build_source_mlx_hybrid_artifact(
        canonical.path, tmp_path / "native", lexical_precision=precision
    )
    assert repeated.path == artifact.path
    assert repeated.artifact_sha256 == artifact.artifact_sha256


def test_hybrid_q8_and_bf16_have_distinct_artifact_and_backend_identity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mlx.core")
    canonical = _canonical(tmp_path)
    q8_record = build_source_mlx_hybrid_artifact(
        canonical.path, tmp_path / "native", lexical_precision="q8"
    )
    bf16_record = build_source_mlx_hybrid_artifact(
        canonical.path, tmp_path / "native", lexical_precision="bf16"
    )
    assert q8_record.artifact_sha256 != bf16_record.artifact_sha256
    assert q8_record.path != bf16_record.path
    assert q8_record.shard_bytes < bf16_record.shard_bytes
    assert q8_record.lexical_rmse > 0
    assert bf16_record.lexical_rmse == 0
    assert MLXSourceHybridQ8Engine.backend != MLXSourceHybridBF16Engine.backend
    with pytest.raises(SourceMlxArtifactError, match="differs from backend"):
        VerifiedSourceMlxHybridQ8Artifact(bf16_record.path)
    with pytest.raises(SourceMlxArtifactError, match="differs from backend"):
        VerifiedSourceMlxHybridBF16Artifact(q8_record.path)


def test_hybrid_artifact_rejects_tamper_and_unsupported_sources(tmp_path: Path) -> None:
    pytest.importorskip("mlx.core")
    canonical = _canonical(tmp_path)
    record = build_source_mlx_hybrid_artifact(canonical.path, tmp_path / "native")
    artifact = VerifiedSourceMlxHybridQ8Artifact(record.path)
    shard = artifact.path / artifact.manifest["shards"][0]["filename"]
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)
    with pytest.raises(SourceMlxArtifactError, match="hash"):
        VerifiedSourceMlxHybridQ8Artifact(artifact.path)

    untied_root = tmp_path / "untied"
    untied_root.mkdir()
    untied = _canonical(untied_root, tied=False)
    with pytest.raises(SourceMlxLoweringError, match="tied dense Qwen2"):
        build_source_mlx_hybrid_artifact(untied.path, tmp_path / "untied-native")

    f32_root = tmp_path / "f32"
    f32_root.mkdir()
    f32 = _canonical(f32_root, dtype=torch.float32)
    with pytest.raises(SourceMlxLoweringError, match="uniform canonical BF16"):
        build_source_mlx_hybrid_artifact(f32.path, tmp_path / "f32-native")


def test_hybrid_cli_preserves_precision_and_promotion_boundary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("mlx.core")
    canonical = _canonical(tmp_path)
    assert (
        decompiler_main(
            [
                "lower-mlx-hybrid",
                str(canonical.path),
                "--output-root",
                str(tmp_path / "native"),
                "--lexical-precision",
                "q8",
            ]
        )
        == 0
    )
    lowered = json.loads(capsys.readouterr().out)
    assert lowered["status"] == "native-lowered-approximate-unexecuted"
    assert lowered["build"]["lexical_precision"] == "q8"
    assert lowered["build"]["production_runtime_eligible"] is False
    assert decompiler_main(["verify-mlx-hybrid", lowered["build"]["path"]]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["artifact_sha256"] == lowered["build"]["artifact_sha256"]
    assert verified["lexical_precision"] == "q8"
    assert verified["execution_certified"] is False
    assert verified["production_runtime_eligible"] is False
