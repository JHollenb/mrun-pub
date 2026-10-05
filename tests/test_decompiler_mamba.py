from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mrun.decompiler import build_component_artifact, decompile_source, open_component_artifact
from mrun.decompiler.ir import ModelDimensionsIR


def _config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["MambaForCausalLM"],
        "bos_token_id": 0,
        "conv_kernel": 3,
        "expand": 2,
        "hidden_act": "silu",
        "hidden_size": 8,
        "intermediate_size": 16,
        "layer_norm_epsilon": 1e-5,
        "model_type": "mamba",
        "num_hidden_layers": 2,
        "pad_token_id": 0,
        "residual_in_fp32": True,
        "state_size": 4,
        "time_step_rank": 2,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "use_bias": False,
        "use_cache": True,
        "use_conv_bias": True,
        "vocab_size": 13,
    }
    config.update(updates)
    return config


def _weights(*, tied: bool = True) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(314159)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.04

    output: dict[str, torch.Tensor] = {
        "backbone.embeddings.weight": random(16, 8),
        "backbone.norm_f.weight": torch.ones(8, dtype=torch.float32),
    }
    if not tied:
        output["lm_head.weight"] = random(16, 8)
    for layer in range(2):
        prefix = f"backbone.layers.{layer}"
        mixer = f"{prefix}.mixer"
        output.update(
            {
                f"{prefix}.norm.weight": torch.ones(8, dtype=torch.float32),
                f"{mixer}.in_proj.weight": random(32, 8),
                f"{mixer}.conv1d.weight": random(16, 1, 3),
                f"{mixer}.conv1d.bias": random(16),
                f"{mixer}.x_proj.weight": random(10, 16),
                f"{mixer}.dt_proj.weight": random(16, 2),
                f"{mixer}.dt_proj.bias": random(16),
                f"{mixer}.A_log": random(16, 4),
                f"{mixer}.D": random(16),
                f"{mixer}.out_proj.weight": random(8, 16),
            }
        )
    return output


def _write_model(
    root: Path,
    *,
    tied: bool = True,
    config_updates: dict[str, object] | None = None,
    weight_updates: dict[str, torch.Tensor] | None = None,
) -> None:
    root.mkdir()
    config = _config(tie_word_embeddings=tied, **(config_updates or {}))
    weights = _weights(tied=tied)
    weights.update(weight_updates or {})
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})


def test_mamba_decompiles_without_inventing_attention_or_kv(tmp_path: Path) -> None:
    source = tmp_path / "mamba"
    _write_model(source)

    built = build_component_artifact(source, tmp_path / "artifacts")
    bundle = open_component_artifact(built.path).ir_bundle
    dimensions = bundle.model.dimensions

    assert bundle.model.architecture_id == "mamba1-selective-state-space-causal-decoder"
    assert dimensions.num_attention_heads == 0
    assert dimensions.num_key_value_heads == 0
    assert dimensions.head_dim == 0
    assert dimensions.max_position_embeddings == 0
    assert not any("k_cache" in item or "v_cache" in item for item in bundle.model.state_refs)
    assert bundle.model.state_refs == (
        "layers.0.conv_state",
        "layers.0.recurrent_state",
        "layers.1.conv_state",
        "layers.1.recurrent_state",
        "sequence_length",
    )
    assert {item.kind for item in bundle.state.slots} == {
        "committed-sequence-length",
        "mamba-causal-convolution-window",
        "mamba-selective-scan-state",
    }
    assert bundle.state.commit_protocol.protocol_id == "mrun-transactional-recurrent-state-v1"
    assert all("prefix" in item.provisional_representation for item in bundle.state.slots)
    operations = {item.operation_id: item for item in bundle.model.operations}
    assert operations["layers.0.mixer.causal_conv"].attributes == {
        "activation": "silu",
        "kernel_size": 3,
        "state_slot": "layers.0.conv_state",
    }
    assert operations["layers.0.mixer.selective_scan"].attributes["state_size"] == 4
    assert operations["layers.0.residual"].attributes["residual_accumulation"] == "float32"


def test_mamba_tied_lexical_mapping_preserves_one_physical_allocation(tmp_path: Path) -> None:
    source = tmp_path / "mamba"
    _write_model(source)

    built = build_component_artifact(source, tmp_path / "artifacts")
    artifact = open_component_artifact(built.path)
    physical = artifact.ir_bundle.physical_weights
    views = {item.logical_name: item for item in physical.views}

    assert views["token_embedding.weight"].allocation_id == views["lm_head.weight"].allocation_id
    assert views["token_embedding.weight"].logical_shape == (16, 8)
    assert len(physical.alias_classes) == 1
    assert physical.alias_classes[0].logical_names == (
        "lm_head.weight",
        "token_embedding.weight",
    )
    io = artifact.ir_bundle.io
    assert io.text_spaces[0].token_count == 13
    assert io.text_spaces[0].physical_row_count == 16
    assert io.row_mappers[0].unreachable_rows == (13, 14, 15)


def test_mamba_untied_readout_is_owned_explicitly(tmp_path: Path) -> None:
    source = tmp_path / "mamba"
    _write_model(source, tied=False)

    built = build_component_artifact(source, tmp_path / "artifacts")
    physical = open_component_artifact(built.path).ir_bundle.physical_weights
    views = {item.logical_name: item for item in physical.views}

    assert views["token_embedding.weight"].allocation_id != views["lm_head.weight"].allocation_id
    assert physical.alias_classes == ()


@pytest.mark.parametrize(
    "config_updates",
    [
        {"architectures": ["MambaModel"]},
        {"mystery_recurrence": True},
        {"ssm_cfg": {"layer": "mamba2"}},
        {"hidden_act": "gelu"},
        {"d_model": 9},
        {"use_mambapy": True},
    ],
)
def test_mamba_semantic_variants_fail_closed(
    tmp_path: Path, config_updates: dict[str, object]
) -> None:
    source = tmp_path / "mamba"
    _write_model(source, config_updates=config_updates)

    result = decompile_source(source)

    assert not result.succeeded
    assert result.report.failures[0].code == "unsupported_variant"


def test_mamba_unexplained_tensor_and_missing_state_fail_closed(tmp_path: Path) -> None:
    unexplained = tmp_path / "unexplained"
    _write_model(
        unexplained,
        weight_updates={"backbone.layers.0.mixer.unknown": torch.ones(1)},
    )
    result = decompile_source(unexplained)
    assert not result.succeeded
    assert result.report.failures[0].code == "source_coverage_failure"

    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "config.json").write_text(json.dumps(_config(), sort_keys=True), encoding="utf-8")
    weights = _weights()
    del weights["backbone.layers.1.mixer.A_log"]
    save_file(weights, missing / "model.safetensors", metadata={"format": "pt"})
    result = decompile_source(missing)
    assert not result.succeeded
    assert result.report.failures[0].code == "source_coverage_failure"


def test_architecture_neutral_dimensions_allow_only_coherent_zero_attention() -> None:
    dimensions = ModelDimensionsIR(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=0,
        num_key_value_heads=0,
        head_dim=0,
        vocab_size=13,
        physical_vocab_rows=16,
        max_position_embeddings=0,
    )
    assert ModelDimensionsIR.from_dict(dimensions.as_dict()) == dimensions

    payload = dimensions.as_dict()
    payload["num_attention_heads"] = 1
    with pytest.raises(ValueError, match="all be zero or all be positive"):
        ModelDimensionsIR.from_dict(payload)
