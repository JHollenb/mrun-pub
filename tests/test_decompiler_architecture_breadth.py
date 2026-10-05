from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
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


def _mistral_config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["MistralForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 4,
        "hidden_act": "silu",
        "hidden_size": 8,
        "intermediate_size": 16,
        "max_position_embeddings": 32,
        "mlp_bias": False,
        "model_type": "mistral",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "use_cache": True,
        "vocab_size": 13,
    }
    config.update(updates)
    return config


def _mistral_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1907)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.04

    prefix = "model.layers.0"
    return {
        "model.embed_tokens.weight": random(13, 8),
        "model.norm.weight": torch.ones(8),
        "lm_head.weight": random(13, 8),
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


def _write_mistral(root: Path, **config_updates: object) -> dict[str, torch.Tensor]:
    root.mkdir()
    weights = _mistral_weights()
    (root / "config.json").write_text(
        json.dumps(_mistral_config(**config_updates), sort_keys=True), encoding="utf-8"
    )
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    return weights


def test_mistral_default_attention_decompiles_and_matches_transformers(
    tmp_path: Path,
) -> None:
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "mistral"
    weights = _write_mistral(source)
    canonical = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(canonical.path)
    tokens = torch.tensor([[0, 3, 7, 12], [4, 2, 9, 1]], dtype=torch.int64)

    certification = run_g8_reference_parity(executable, [[[0]], tokens.tolist()])
    hf_config = transformers.MistralConfig(**_mistral_config())
    hf_config._attn_implementation = "eager"
    hf_model = transformers.MistralForCausalLM(hf_config).eval()
    hf_model.load_state_dict(weights, strict=True)
    with torch.inference_mode():
        expected = hf_model(tokens).logits
    observed = executable.forward(tokens).logits

    torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-6)
    assert certification.maximum_absolute_error == 0.0
    assert executable.identity.architecture_id == "mistral-dense-causal-decoder"


def test_mistral_sliding_window_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "mistral-windowed"
    _write_mistral(source, sliding_window=16)

    result = decompile_source(source)

    assert not result.succeeded
    assert result.report.failures[0].code == "unsupported_variant"
    assert "sliding_attention" in json.dumps(result.report.failures[0].details)


def test_mistral_direct_unquantized_mlx_is_bounded_and_loadable(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.utils import load_model

    source = tmp_path / "mistral"
    _write_mistral(source)
    canonical = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(canonical.path)
    record = build_source_mlx_artifact(canonical.path, tmp_path / "native")
    artifact = VerifiedSourceMlxArtifact(record.path)

    assert artifact.source["architecture"] == "mistral"
    model, _config = load_model(artifact.path, lazy=False, strict=True)
    tokens = torch.tensor([[0, 3, 7, 12]], dtype=torch.int64)
    native = model(mx.array(tokens.numpy().astype(np.int32, copy=False)))
    mx.eval(native)
    native_logits = torch.from_numpy(np.array(native.astype(mx.float32), copy=False))
    reference_logits = executable.forward(tokens).logits.to(torch.float32)
    torch.testing.assert_close(native_logits, reference_logits, rtol=3e-3, atol=3e-4)


def _gemma_config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["GemmaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 4,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 8,
        "intermediate_size": 16,
        "max_position_embeddings": 32,
        "mlp_bias": False,
        "model_type": "gemma",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "use_cache": True,
        "vocab_size": 13,
    }
    config.update(updates)
    return config


def _gemma_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2402)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.04

    prefix = "model.layers.0"
    return {
        "model.embed_tokens.weight": random(13, 8),
        "model.norm.weight": random(8),
        f"{prefix}.input_layernorm.weight": random(8),
        f"{prefix}.post_attention_layernorm.weight": random(8),
        f"{prefix}.self_attn.q_proj.weight": random(8, 8),
        f"{prefix}.self_attn.k_proj.weight": random(4, 8),
        f"{prefix}.self_attn.v_proj.weight": random(4, 8),
        f"{prefix}.self_attn.o_proj.weight": random(8, 8),
        f"{prefix}.mlp.gate_proj.weight": random(16, 8),
        f"{prefix}.mlp.up_proj.weight": random(16, 8),
        f"{prefix}.mlp.down_proj.weight": random(8, 16),
    }


def _write_gemma(root: Path, **config_updates: object) -> dict[str, torch.Tensor]:
    root.mkdir()
    weights = _gemma_weights()
    (root / "config.json").write_text(
        json.dumps(_gemma_config(**config_updates), sort_keys=True), encoding="utf-8"
    )
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    return weights


def test_gemma1_offset_norm_embedding_scale_and_geglu_are_exact(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "gemma"
    weights = _write_gemma(source)
    canonical = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(canonical.path)
    tokens = torch.tensor([[0, 3, 7, 12], [4, 2, 9, 1]], dtype=torch.int64)

    certification = run_g8_reference_parity(executable, [[[0]], tokens.tolist()])
    operations = {
        item.operation_id: item for item in executable.artifact.ir_bundle.model.operations
    }
    assert operations["embedding_scale"].attributes["scalar"] == pytest.approx(8**0.5)
    assert operations["layers.0.attention_norm"].kind == "gemma-rms-norm"
    assert operations["layers.0.mlp.gelu"].kind == "gelu-tanh"

    hf_config = transformers.GemmaConfig(**_gemma_config())
    hf_config._attn_implementation = "eager"
    hf_model = transformers.GemmaForCausalLM(hf_config).eval()
    state = {**weights, "lm_head.weight": weights["model.embed_tokens.weight"]}
    hf_model.load_state_dict(state, strict=True)
    with torch.inference_mode():
        expected = hf_model(tokens).logits
    observed = executable.forward(tokens).logits
    torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-6)
    assert certification.maximum_absolute_error == 0.0


@pytest.mark.parametrize(
    "updates",
    [
        {"hidden_act": "gelu"},
        {"hidden_activation": "gelu"},
        {"attention_bias": True},
        {"mystery_gemma_mode": True},
    ],
)
def test_gemma1_unsupported_variants_fail_closed(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    source = tmp_path / "invalid-gemma"
    _write_gemma(source, **updates)

    result = decompile_source(source)

    assert not result.succeeded
    assert result.report.failures[0].code == "unsupported_variant"


def test_gemma1_direct_mlx_is_explicitly_unregistered(tmp_path: Path) -> None:
    source = tmp_path / "gemma"
    _write_gemma(source)
    canonical = build_component_artifact(source, tmp_path / "canonical")

    with pytest.raises(SourceMlxLoweringError, match="no direct MLX target"):
        build_source_mlx_artifact(canonical.path, tmp_path / "native")


def _phi_config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["PhiForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "embd_pdrop": 0.0,
        "eos_token_id": 2,
        "hidden_act": "gelu_new",
        "hidden_size": 8,
        "intermediate_size": 16,
        "layer_norm_eps": 1e-5,
        "max_position_embeddings": 32,
        "model_type": "phi",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.5,
        "qk_layernorm": False,
        "resid_pdrop": 0.0,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "use_cache": True,
        "vocab_size": 13,
    }
    config.update(updates)
    return config


def _phi_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(31415)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.04

    prefix = "model.layers.0"
    return {
        "model.embed_tokens.weight": random(13, 8),
        "model.final_layernorm.weight": torch.ones(8),
        "model.final_layernorm.bias": random(8),
        "lm_head.weight": random(13, 8),
        "lm_head.bias": random(13),
        f"{prefix}.input_layernorm.weight": torch.ones(8),
        f"{prefix}.input_layernorm.bias": random(8),
        f"{prefix}.self_attn.q_proj.weight": random(8, 8),
        f"{prefix}.self_attn.q_proj.bias": random(8),
        f"{prefix}.self_attn.k_proj.weight": random(8, 8),
        f"{prefix}.self_attn.k_proj.bias": random(8),
        f"{prefix}.self_attn.v_proj.weight": random(8, 8),
        f"{prefix}.self_attn.v_proj.bias": random(8),
        f"{prefix}.self_attn.dense.weight": random(8, 8),
        f"{prefix}.self_attn.dense.bias": random(8),
        f"{prefix}.mlp.fc1.weight": random(16, 8),
        f"{prefix}.mlp.fc1.bias": random(16),
        f"{prefix}.mlp.fc2.weight": random(8, 16),
        f"{prefix}.mlp.fc2.bias": random(8),
    }


def _write_phi(root: Path, **config_updates: object) -> dict[str, torch.Tensor]:
    root.mkdir()
    weights = _phi_weights()
    (root / "config.json").write_text(
        json.dumps(_phi_config(**config_updates), sort_keys=True), encoding="utf-8"
    )
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    return weights


def test_phi_parallel_residual_partial_rope_matches_transformers(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "phi"
    weights = _write_phi(source)
    canonical = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(canonical.path)
    tokens = torch.tensor([[0, 3, 7, 12], [4, 2, 9, 1]], dtype=torch.int64)

    certification = run_g8_reference_parity(executable, [[[0]], tokens.tolist()])
    operations = {
        item.operation_id: item for item in executable.artifact.ir_bundle.model.operations
    }
    assert operations["layers.0.attention.rotary"].attributes["rotary_dim"] == 2
    assert operations["layers.0.mlp.fc1"].inputs == ("layers.0.input_norm.hidden",)
    assert operations["layers.0.parallel_sum"].inputs == (
        "layers.0.attention.output",
        "layers.0.mlp.output",
    )

    hf_config = transformers.PhiConfig(**_phi_config())
    hf_config._attn_implementation = "eager"
    hf_model = transformers.PhiForCausalLM(hf_config).eval()
    hf_model.load_state_dict(weights, strict=True)
    with torch.inference_mode():
        expected = hf_model(tokens).logits
    observed = executable.forward(tokens).logits
    torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-6)
    assert certification.maximum_absolute_error == 0.0


@pytest.mark.parametrize(
    "updates",
    [
        {"hidden_act": "gelu"},
        {"qk_layernorm": True},
        {"tie_word_embeddings": True},
        {"partial_rotary_factor": 0.25},
        {"mystery_phi_mode": True},
    ],
)
def test_phi_unsupported_variants_fail_closed(tmp_path: Path, updates: dict[str, object]) -> None:
    source = tmp_path / "invalid-phi"
    _write_phi(source, **updates)

    result = decompile_source(source)

    assert not result.succeeded
    assert result.report.failures[0].code == "unsupported_variant"


def test_phi_direct_unquantized_mlx_is_bounded_and_loadable(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.utils import load_model

    source = tmp_path / "phi"
    _write_phi(source)
    canonical = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(canonical.path)
    record = build_source_mlx_artifact(canonical.path, tmp_path / "native")
    artifact = VerifiedSourceMlxArtifact(record.path)

    assert artifact.source["architecture"] == "phi"
    assert artifact.source["runtime_numerical_compatibility"] == (
        "mlx-lm-phi-approximate-gelu-bounded-parity-required"
    )
    model, _config = load_model(artifact.path, lazy=False, strict=True)
    tokens = torch.tensor([[0, 3, 7, 12]], dtype=torch.int64)
    native = model(mx.array(tokens.numpy().astype(np.int32, copy=False)))
    mx.eval(native)
    native_logits = torch.from_numpy(np.array(native.astype(mx.float32), copy=False))
    reference_logits = executable.forward(tokens).logits.to(torch.float32)
    torch.testing.assert_close(native_logits, reference_logits, rtol=3e-3, atol=3e-4)


def test_phi_padded_output_requires_matching_bias_capacity(tmp_path: Path) -> None:
    source = tmp_path / "phi-padded-output"
    source.mkdir()
    weights = _phi_weights()
    weights["lm_head.weight"] = torch.cat(
        (weights["lm_head.weight"], torch.zeros((1, 8), dtype=torch.float32)), dim=0
    )
    (source / "config.json").write_text(json.dumps(_phi_config(), sort_keys=True), encoding="utf-8")
    save_file(weights, source / "model.safetensors", metadata={"format": "pt"})

    result = decompile_source(source)

    assert not result.succeeded
    assert result.report.failures[0].code == "io_contract_failure"
    assert "bias" in result.report.failures[0].message


@pytest.mark.parametrize("family", ["mistral", "phi"])
def test_uncertified_architecture_q4_paths_fail_closed(tmp_path: Path, family: str) -> None:
    source = tmp_path / family
    if family == "mistral":
        _write_mistral(source)
    else:
        _write_phi(source)
    canonical = build_component_artifact(source, tmp_path / "canonical")

    with pytest.raises(SourceMlxLoweringError, match="no certified mapping"):
        build_source_mlx_q4_artifact(canonical.path, tmp_path / "native-q4")
