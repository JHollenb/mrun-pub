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
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": 0.0,
        "bos_token_id": 0,
        "embd_pdrop": 0.0,
        "eos_token_id": 0,
        "layer_norm_epsilon": 1e-5,
        "model_type": "gpt2",
        "n_ctx": 32,
        "n_embd": 8,
        "n_head": 2,
        "n_inner": 16,
        "n_layer": 1,
        "n_positions": 32,
        "reorder_and_upcast_attn": False,
        "resid_pdrop": 0.0,
        "scale_attn_by_inverse_layer_idx": False,
        "scale_attn_weights": True,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "use_cache": True,
        "vocab_size": 13,
    }
    config.update(updates)
    return config


def _weights(*, tied: bool = True, intermediate: int = 16) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2112)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.04

    prefix = "transformer.h.0"
    weights = {
        "transformer.wte.weight": random(13, 8),
        "transformer.wpe.weight": random(32, 8),
        "transformer.ln_f.weight": torch.ones(8),
        "transformer.ln_f.bias": random(8),
        f"{prefix}.attn.bias": torch.tril(torch.ones(32, 32))[None, None],
        f"{prefix}.attn.c_attn.weight": random(8, 24),
        f"{prefix}.attn.c_attn.bias": random(24),
        f"{prefix}.attn.c_proj.weight": random(8, 8),
        f"{prefix}.attn.c_proj.bias": random(8),
        f"{prefix}.ln_1.weight": torch.ones(8),
        f"{prefix}.ln_1.bias": random(8),
        f"{prefix}.ln_2.weight": torch.ones(8),
        f"{prefix}.ln_2.bias": random(8),
        f"{prefix}.mlp.c_fc.weight": random(8, intermediate),
        f"{prefix}.mlp.c_fc.bias": random(intermediate),
        f"{prefix}.mlp.c_proj.weight": random(intermediate, 8),
        f"{prefix}.mlp.c_proj.bias": random(8),
    }
    if not tied:
        weights["lm_head.weight"] = random(13, 8)
    return weights


def _write_model(
    root: Path,
    *,
    tied: bool = True,
    config_updates: dict[str, object] | None = None,
    extra_weights: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    root.mkdir()
    config = _config(tie_word_embeddings=tied, **(config_updates or {}))
    raw_inner = config["n_inner"]
    intermediate = 32 if raw_inner is None else int(raw_inner)
    weights = _weights(tied=tied, intermediate=intermediate)
    weights.update(extra_weights or {})
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    return weights


def test_gpt2_decompiles_absolute_positions_conv1d_and_serialized_causal_mask(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gpt2"
    weights = _write_model(source)

    result = decompile_source(source, source_id="Test/gpt2", resolved_revision="2" * 40)

    assert result.succeeded and result.ir_bundle is not None
    assert result.report.selected_adapter_id == "mrun.hf.gpt2"
    assert result.report.coverage is not None and result.report.coverage.complete
    assert result.report.coverage.source_tensor_count == len(weights)
    physical = result.ir_bundle.physical_weights
    assert len(physical.alias_classes) == 1
    alias = physical.alias_classes[0]
    assert alias.logical_names == ("lm_head.weight", "token_embedding.weight")
    views = {item.logical_name: item for item in physical.views}
    assert views["position_embedding.weight"].logical_shape == (32, 8)
    assert views["layers.0.attention.causal_mask"].parameter_kind == "buffer"
    operations = {item.operation_id: item for item in result.ir_bundle.model.operations}
    assert operations["position_embedding"].kind == "absolute-position-embedding-add"
    assert operations["layers.0.attention.c_attn"].kind == "conv1d-linear"
    assert operations["layers.0.attention.split_qkv"].attributes == {"split_width": 8}
    assert operations["layers.0.mlp.gelu"].kind == "gelu-tanh"


@pytest.mark.parametrize(
    ("config_updates", "extra_weights"),
    [
        ({"activation_function": "gelu"}, {}),
        ({"scale_attn_weights": False}, {}),
        ({"scale_attn_by_inverse_layer_idx": True}, {}),
        ({"reorder_and_upcast_attn": True}, {}),
        ({"n_ctx": 16}, {}),
        ({"mystery_gpt2_mode": True}, {}),
        ({}, {"transformer.h.0.attn.masked_bias": torch.tensor(-1e4)}),
    ],
)
def test_gpt2_variants_fail_closed(
    tmp_path: Path,
    config_updates: dict[str, object],
    extra_weights: dict[str, torch.Tensor],
) -> None:
    source = tmp_path / "invalid"
    _write_model(source, config_updates=config_updates, extra_weights=extra_weights)

    result = decompile_source(source)

    assert not result.succeeded
    assert result.report.failures[0].code in {"unsupported_variant", "source_coverage_failure"}


@pytest.mark.parametrize("tied", [True, False])
def test_gpt2_reference_matches_transformers_and_certifies_exact_g8(
    tmp_path: Path, tied: bool
) -> None:
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "gpt2"
    weights = _write_model(source, tied=tied)
    canonical = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(canonical.path)
    tokens = torch.tensor([[0, 3, 7, 12], [4, 2, 9, 1]], dtype=torch.int64)

    result = executable.forward(tokens)
    certification = run_g8_reference_parity(executable, [[[0]], tokens.tolist()])

    hf_config = transformers.GPT2Config(**_config(tie_word_embeddings=tied))
    hf_config._attn_implementation = "eager"
    hf_model = transformers.GPT2LMHeadModel(hf_config).eval()
    state = dict(weights)
    state.pop("transformer.h.0.attn.bias")
    if tied:
        state["lm_head.weight"] = weights["transformer.wte.weight"]
    hf_model.load_state_dict(state, strict=True)
    with torch.inference_mode():
        expected = hf_model(tokens).logits
    torch.testing.assert_close(result.logits, expected, rtol=2e-5, atol=2e-6)
    assert certification.maximum_absolute_error == 0.0
    assert certification.source_logits_fingerprints == certification.ir_logits_fingerprints


def test_gpt2_rejects_serialized_head_when_config_declares_tie(tmp_path: Path) -> None:
    source = tmp_path / "invalid-alias"
    _write_model(source, extra_weights={"lm_head.weight": torch.zeros(13, 8)})

    result = decompile_source(source)

    assert not result.succeeded
    assert result.report.failures[0].code == "alias_evidence_failure"


def test_gpt2_direct_unquantized_mlx_mapping_loads_and_runs(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.utils import load_model

    source = tmp_path / "gpt2"
    source_weights = _write_model(source, config_updates={"n_inner": 32})
    canonical = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(canonical.path)
    record = build_source_mlx_artifact(canonical.path, tmp_path / "native")
    artifact = VerifiedSourceMlxArtifact(record.path)

    assert artifact.source["architecture"] == "gpt2"
    assert artifact.source["runtime_numerical_compatibility"] == (
        "mlx-lm-gpt2-gelu-approx-native-parity-required"
    )
    emitted: dict[str, torch.Tensor] = {}
    descriptors: dict[str, dict[str, object]] = {}
    for shard in artifact.manifest["shards"]:
        with safe_open(artifact.path / shard["filename"], framework="pt") as handle:
            for item in shard["parameters"]:
                emitted[item["name"]] = handle.get_tensor(item["name"])
                descriptors[item["name"]] = item
    assert "wte.weight" in emitted and "transformer.wte.weight" not in emitted
    assert set(item["source_tensor"] for item in descriptors.values()) == set(source_weights)
    assert all(
        torch.equal(emitted[name.removeprefix("transformer.")], value)
        for name, value in source_weights.items()
    )

    model, _loaded_config = load_model(artifact.path, lazy=False, strict=True)
    tokens = torch.tensor([[0, 3, 7, 12]], dtype=torch.int64)
    native = model(mx.array(tokens.numpy().astype(np.int32, copy=False)))
    mx.eval(native)
    native_logits = torch.from_numpy(np.array(native.astype(mx.float32), copy=False))
    reference_logits = executable.forward(tokens).logits.to(torch.float32)
    torch.testing.assert_close(native_logits, reference_logits, rtol=3e-3, atol=3e-4)


def test_gpt2_direct_mlx_rejects_untied_or_noncanonical_causal_mask(
    tmp_path: Path,
) -> None:
    untied = tmp_path / "untied"
    _write_model(untied, tied=False, config_updates={"n_inner": 32})
    untied_artifact = build_component_artifact(untied, tmp_path / "untied-canonical")
    with pytest.raises(SourceMlxLoweringError, match="requires tied lexical"):
        build_source_mlx_artifact(untied_artifact.path, tmp_path / "untied-native")

    bad_mask = tmp_path / "bad-mask"
    _write_model(
        bad_mask,
        config_updates={"n_inner": 32},
        extra_weights={"transformer.h.0.attn.bias": torch.ones(1, 1, 32, 32)},
    )
    bad_artifact = build_component_artifact(bad_mask, tmp_path / "bad-canonical")
    with pytest.raises(SourceMlxLoweringError, match="source mask is not canonical"):
        build_source_mlx_artifact(bad_artifact.path, tmp_path / "bad-native")


def test_gpt2_direct_q4_fails_closed_until_mapping_is_certified(tmp_path: Path) -> None:
    source = tmp_path / "gpt2"
    _write_model(source, config_updates={"n_inner": 32})
    canonical = build_component_artifact(source, tmp_path / "canonical")

    with pytest.raises(SourceMlxLoweringError, match="no certified mapping"):
        build_source_mlx_q4_artifact(canonical.path, tmp_path / "native-q4")
