from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mrun.decompiler import (
    SourceMlxLoweringError,
    VerifiedSourceMlxArtifact,
    build_component_artifact,
    build_source_mlx_artifact,
    build_source_mlx_q4_artifact,
    decompile_source,
    lower_component_artifact_to_reference,
    run_g8_reference_parity,
)


def _config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["GPTNeoXForCausalLM"],
        "attention_bias": True,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "hidden_act": "gelu",
        "hidden_dropout": 0.0,
        "hidden_size": 8,
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "layer_norm_eps": 1e-5,
        "max_position_embeddings": 32,
        "model_type": "gpt_neox",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "rotary_emb_base": 10_000,
        "rotary_pct": 0.5,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "use_cache": True,
        "use_parallel_residual": True,
        "vocab_size": 13,
    }
    config.update(updates)
    return config


def _weights(*, attention_bias: bool = True) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(9017)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.04

    prefix = "gpt_neox.layers.0"
    weights = {
        "embed_out.weight": random(13, 8),
        "gpt_neox.embed_in.weight": random(13, 8),
        "gpt_neox.final_layer_norm.bias": random(8),
        "gpt_neox.final_layer_norm.weight": torch.ones(8),
        f"{prefix}.input_layernorm.bias": random(8),
        f"{prefix}.input_layernorm.weight": torch.ones(8),
        f"{prefix}.attention.query_key_value.weight": random(24, 8),
        f"{prefix}.attention.dense.weight": random(8, 8),
        f"{prefix}.post_attention_layernorm.bias": random(8),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(8),
        f"{prefix}.mlp.dense_h_to_4h.bias": random(32),
        f"{prefix}.mlp.dense_h_to_4h.weight": random(32, 8),
        f"{prefix}.mlp.dense_4h_to_h.bias": random(8),
        f"{prefix}.mlp.dense_4h_to_h.weight": random(8, 32),
    }
    if attention_bias:
        weights[f"{prefix}.attention.query_key_value.bias"] = random(24)
        weights[f"{prefix}.attention.dense.bias"] = random(8)
    return weights


def _write_model(
    root: Path,
    *,
    config_updates: dict[str, object] | None = None,
    attention_bias: bool = True,
    extra_weights: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    root.mkdir()
    config = _config(attention_bias=attention_bias, **(config_updates or {}))
    weights = _weights(attention_bias=attention_bias)
    weights.update(extra_weights or {})
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    return weights


def test_gpt_neox_decompiles_fused_qkv_partial_rope_and_untied_lexical_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pythia"
    weights = _write_model(source)

    result = decompile_source(
        source,
        source_id="Test/pythia",
        resolved_revision="a" * 40,
    )

    assert result.succeeded and result.ir_bundle is not None
    assert result.report.selected_adapter_id == "mrun.hf.gpt-neox-pythia"
    assert result.report.coverage is not None and result.report.coverage.complete
    assert result.report.coverage.source_tensor_count == len(weights)
    physical = result.ir_bundle.physical_weights
    assert not physical.alias_classes
    views = {item.logical_name: item for item in physical.views}
    assert views["token_embedding.weight"].allocation_id != views["lm_head.weight"].allocation_id
    assert "layers.0.attention.query_key_value.weight" in views
    assert "layers.0.attention_norm.bias" in views
    assert "layers.0.mlp.dense_4h_to_h.bias" in views
    operations = {item.operation_id: item for item in result.ir_bundle.model.operations}
    assert operations["layers.0.attention.unpack_qkv"].attributes == {
        "head_dim": 4,
        "num_attention_heads": 2,
        "packing": "per-head-qkv",
    }
    assert operations["layers.0.attention.rotary"].attributes["rotary_dim"] == 2
    assert operations["layers.0.mlp_norm"].inputs == ("embedding.hidden",)
    assert operations["layers.0.mlp.gelu"].kind == "gelu-erf"


@pytest.mark.parametrize(
    ("config_updates", "extra_weights", "failure_code"),
    [
        ({"tie_word_embeddings": True}, {}, "unsupported_variant"),
        ({"hidden_act": "gelu_new"}, {}, "unsupported_variant"),
        ({"rotary_pct": 0.3}, {}, "unsupported_variant"),
        ({"mystery_neox_mode": True}, {}, "unsupported_variant"),
        (
            {},
            {"gpt_neox.layers.0.attention.unknown.weight": torch.zeros(1)},
            "source_coverage_failure",
        ),
    ],
)
def test_gpt_neox_variants_fail_closed(
    tmp_path: Path,
    config_updates: dict[str, object],
    extra_weights: dict[str, torch.Tensor],
    failure_code: str,
) -> None:
    source = tmp_path / "invalid"
    _write_model(source, config_updates=config_updates, extra_weights=extra_weights)

    result = decompile_source(source)

    assert not result.succeeded
    assert result.report.failures[0].code == failure_code


@pytest.mark.parametrize("parallel", [True, False])
def test_gpt_neox_reference_matches_transformers_and_certifies_g8(
    tmp_path: Path, parallel: bool
) -> None:
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "pythia"
    weights = _write_model(source, config_updates={"use_parallel_residual": parallel})
    canonical = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(canonical.path)
    tokens = torch.tensor([[0, 3, 7, 12], [4, 2, 9, 1]], dtype=torch.int64)

    reference = executable.forward(tokens)
    certification = run_g8_reference_parity(executable, [[[0]], tokens.tolist()])

    hf_config = transformers.GPTNeoXConfig(**_config(use_parallel_residual=parallel))
    hf_config._attn_implementation = "eager"
    hf_model = transformers.GPTNeoXForCausalLM(hf_config).eval()
    # Transformers 5.14 renamed the native output-head attribute. The legacy
    # source checkpoint still uses embed_out; preserve the tensor and strict loading.
    native_weights = dict(weights)
    if "lm_head.weight" in hf_model.state_dict() and "embed_out.weight" in native_weights:
        native_weights["lm_head.weight"] = native_weights.pop("embed_out.weight")
    hf_model.load_state_dict(native_weights, strict=True)
    with torch.inference_mode():
        expected = hf_model(tokens).logits
    torch.testing.assert_close(reference.logits, expected, rtol=2e-5, atol=2e-6)
    assert certification.maximum_absolute_error == 0.0
    assert certification.source_logits_fingerprints == certification.ir_logits_fingerprints
    mlp_norm = next(
        item
        for item in executable.artifact.ir_bundle.model.operations
        if item.operation_id == "layers.0.mlp_norm"
    )
    expected_input = "embedding.hidden" if parallel else "layers.0.attention_residual.hidden"
    assert mlp_norm.inputs == (expected_input,)


def test_gpt_neox_direct_unquantized_mlx_artifact_loads_and_runs(
    tmp_path: Path,
) -> None:
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.utils import load_model

    source = tmp_path / "pythia"
    weights = _write_model(source)
    canonical = build_component_artifact(
        source,
        tmp_path / "canonical",
        source_id="Test/pythia",
        resolved_revision="b" * 40,
    )
    executable = lower_component_artifact_to_reference(canonical.path)
    record = build_source_mlx_artifact(canonical.path, tmp_path / "native")
    artifact = VerifiedSourceMlxArtifact(record.path)

    assert artifact.source["architecture"] == "gpt_neox"
    assert artifact.source["runtime_numerical_compatibility"] == (
        "mlx-lm-gpt-neox-approximate-gelu-bounded-parity-required"
    )
    assert artifact.config["rotary_pct"] == 0.5
    assert artifact.config["rotary_emb_base"] == 10_000.0
    assert {item["role"] for item in artifact.manifest["shards"]} == {
        "body",
        "egress",
        "ingress",
        "norm",
    }
    emitted: dict[str, torch.Tensor] = {}
    for shard in artifact.manifest["shards"]:
        with safe_open(artifact.path / shard["filename"], framework="pt") as handle:
            emitted.update({name: handle.get_tensor(name) for name in handle.keys()})
    assert set(emitted) == set(weights)
    assert all(torch.equal(emitted[name], weights[name]) for name in weights)

    model, _config_value = load_model(artifact.path, lazy=False, strict=True)
    tokens = torch.tensor([[0, 3, 7, 12]], dtype=torch.int64)
    native = model(mx.array(tokens.numpy()))
    mx.eval(native)
    native_logits = torch.from_numpy(np.array(native.astype(mx.float32), copy=False))
    reference_logits = executable.forward(tokens).logits.to(torch.float32)
    assert native_logits.shape == reference_logits.shape
    assert torch.isfinite(native_logits).all()
    # mlx-lm intentionally uses approximate GELU, so this is a bounded native parity check rather
    # than an exact G8 claim.
    torch.testing.assert_close(native_logits, reference_logits, rtol=2e-3, atol=2e-4)


def test_gpt_neox_reference_accepts_biasless_attention_but_mlx_rejects_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "biasless"
    _write_model(source, attention_bias=False)
    canonical = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(canonical.path)

    certification = run_g8_reference_parity(executable, [[[1, 2, 3]]])
    assert certification.maximum_absolute_error == 0.0
    with pytest.raises(SourceMlxLoweringError, match="requires attention projection biases"):
        build_source_mlx_artifact(canonical.path, tmp_path / "native")


def test_gpt_neox_direct_q4_fails_closed_until_mapping_is_certified(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pythia"
    _write_model(source)
    canonical = build_component_artifact(source, tmp_path / "canonical")

    with pytest.raises(SourceMlxLoweringError, match="no certified mapping"):
        build_source_mlx_q4_artifact(canonical.path, tmp_path / "native-q4")
