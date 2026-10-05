from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import mrun.decompiler.reference as reference_module
from mrun.decompiler import (
    build_component_artifact,
    decompile_source,
    lower_component_artifact_to_reference,
    run_g8_reference_parity,
)

_CACHED_MAMBA_130M = Path(
    "/Users/jakeholl/.cache/huggingface/hub/"
    "models--state-spaces--mamba-130m-hf/snapshots/"
    "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
)


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
        "use_associative_scan": False,
        "use_bias": False,
        "use_cache": True,
        "use_conv_bias": True,
        "vocab_size": 16,
    }
    config.update(updates)
    return config


def _weights(
    *,
    tied: bool = True,
    use_bias: bool = False,
    use_conv_bias: bool = True,
    time_step_rank: int = 2,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(271828)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.04

    output: dict[str, torch.Tensor] = {
        "backbone.embeddings.weight": random(16, 8),
        "backbone.norm_f.weight": torch.linspace(0.9, 1.1, 8, dtype=torch.float32),
    }
    if not tied:
        output["lm_head.weight"] = random(16, 8)
    for layer in range(2):
        prefix = f"backbone.layers.{layer}"
        mixer = f"{prefix}.mixer"
        output.update(
            {
                f"{prefix}.norm.weight": torch.linspace(0.95, 1.05, 8, dtype=torch.float32),
                f"{mixer}.in_proj.weight": random(32, 8),
                f"{mixer}.conv1d.weight": random(16, 1, 3),
                f"{mixer}.x_proj.weight": random(time_step_rank + 8, 16),
                f"{mixer}.dt_proj.weight": random(16, time_step_rank),
                f"{mixer}.dt_proj.bias": random(16),
                f"{mixer}.A_log": random(16, 4),
                f"{mixer}.D": random(16),
                f"{mixer}.out_proj.weight": random(8, 16),
            }
        )
        if use_bias:
            output[f"{mixer}.in_proj.bias"] = random(32)
            output[f"{mixer}.out_proj.bias"] = random(8)
        if use_conv_bias:
            output[f"{mixer}.conv1d.bias"] = random(16)
    return output


def _write_model(root: Path, **config_updates: object) -> None:
    root.mkdir()
    config = _config(**config_updates)
    raw_rank = config["time_step_rank"]
    time_step_rank = 1 if raw_rank == "auto" else raw_rank
    assert type(time_step_rank) is int
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    save_file(
        _weights(
            tied=bool(config["tie_word_embeddings"]),
            use_bias=bool(config["use_bias"]),
            use_conv_bias=bool(config["use_conv_bias"]),
            time_step_rank=time_step_rank,
        ),
        root / "model.safetensors",
        metadata={"format": "pt"},
    )


def _build_reference(tmp_path: Path):
    source = tmp_path / "mamba"
    _write_model(source)
    built = build_component_artifact(source, tmp_path / "artifacts")
    return lower_component_artifact_to_reference(built.path)


def test_mamba_reference_executes_exact_g8_without_a_positional_ceiling(
    tmp_path: Path,
) -> None:
    executable = _build_reference(tmp_path)
    tokens = torch.tensor(
        [
            [index % 16 for index in range(19)],
            [(3 * index + 1) % 16 for index in range(19)],
        ],
        dtype=torch.int64,
    )

    result = executable.forward(tokens)

    assert result.logits.shape == (2, 19, 16)
    assert result.hidden_states.shape == (2, 19, 8)
    assert torch.isfinite(result.logits).all()
    assert executable.identity.architecture_id == ("mamba1-selective-state-space-causal-decoder")
    assert executable.identity.source_model_type == "mamba"
    assert executable.identity.state_mode == "stateless-full-sequence"
    assert {
        "mamba-rms-norm",
        "mamba-causal-depthwise-convolution",
        "mamba-selective-scan",
    }.issubset(executable.identity.operation_kinds)

    certification = run_g8_reference_parity(
        executable,
        [
            [[1]],
            [[1, 4, 7, 2, 9]],
            [[2, 3, 5, 7], [6, 8, 10, 12]],
        ],
    )
    assert certification.maximum_absolute_error == 0.0
    assert certification.maximum_relative_error == 0.0
    assert certification.source_logits_fingerprints == (certification.ir_logits_fingerprints)


def test_mamba_reference_matches_transformers_full_sequence(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    mamba_config = getattr(transformers, "MambaConfig", None)
    mamba_model = getattr(transformers, "MambaForCausalLM", None)
    if mamba_config is None or mamba_model is None:
        pytest.skip("installed Transformers does not expose Mamba")

    executable = _build_reference(tmp_path)
    config = mamba_config(
        vocab_size=16,
        hidden_size=8,
        state_size=4,
        num_hidden_layers=2,
        layer_norm_epsilon=1e-5,
        expand=2,
        conv_kernel=3,
        use_bias=False,
        use_conv_bias=True,
        residual_in_fp32=True,
        time_step_rank=2,
        use_cache=False,
        use_associative_scan=False,
        tie_word_embeddings=True,
    )
    model = mamba_model(config)
    source_weights = _weights()
    source_weights["lm_head.weight"] = source_weights["backbone.embeddings.weight"]
    incompatible = model.load_state_dict(source_weights, strict=False)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    model.tie_weights()
    model.eval()
    tokens = torch.tensor([[1, 4, 7, 2, 9], [3, 5, 8, 13, 0]], dtype=torch.int64)

    # Compare one CPU oracle backend: oneDNN may choose a different reduction
    # order for the source and reference convolution layouts on Linux.
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        expected = model(
            input_ids=tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        actual = executable.forward(tokens)

    torch.testing.assert_close(actual.logits, expected.logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual.hidden_states,
        expected.hidden_states[-1],
        rtol=0.0,
        atol=0.0,
    )


def test_mamba_reference_executes_optional_bias_and_activation_residual_variant(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mamba-variant"
    _write_model(
        source,
        residual_in_fp32=False,
        tie_word_embeddings=False,
        time_step_rank="auto",
        use_bias=True,
        use_conv_bias=False,
    )
    built = build_component_artifact(source, tmp_path / "variant-artifacts")
    executable = lower_component_artifact_to_reference(built.path)

    certification = run_g8_reference_parity(
        executable,
        [[[1, 3, 5, 7]], [[2, 4, 6], [8, 10, 12]]],
    )

    assert certification.maximum_absolute_error == 0.0
    assert (
        executable.store.tensor("lm_head.weight").data_ptr()
        != executable.store.tensor("token_embedding.weight").data_ptr()
    )


def test_mamba_reference_state_protocol_fails_closed(tmp_path: Path) -> None:
    executable = _build_reference(tmp_path)
    state = executable.artifact.ir_bundle.state
    invalid_state = SimpleNamespace(
        slots=state.slots,
        initialization=state.initialization,
        prefill_updates=state.prefill_updates,
        decode_updates=state.decode_updates,
        commit_protocol=replace(
            state.commit_protocol,
            stale_epoch_rule="allow-write-before-epoch-check",
        ),
        capacity_equations=state.capacity_equations,
    )

    with pytest.raises(reference_module.ReferenceLoweringError, match="commit protocol"):
        reference_module._validate_mamba_state(
            invalid_state,
            executable.artifact.ir_bundle.model.dimensions,
            executable.artifact.source.config,
        )


@pytest.mark.skipif(
    __import__("os").environ.get("MRUN_RUN_MODEL_TESTS") != "1" or not _CACHED_MAMBA_130M.exists(), reason="cached state-spaces/mamba-130m-hf is absent"
)
def test_cached_mamba_130m_snapshot_decompiles_read_only() -> None:
    result = decompile_source(_CACHED_MAMBA_130M)

    assert result.succeeded, result.report.as_dict()
    assert result.ir_bundle is not None
    assert result.ir_bundle.model.architecture_id == ("mamba1-selective-state-space-causal-decoder")
    assert result.ir_bundle.state.commit_protocol.protocol_id == (
        "mrun-transactional-recurrent-state-v1"
    )
