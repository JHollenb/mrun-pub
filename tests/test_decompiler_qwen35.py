from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mrun.decompiler import (
    build_component_artifact,
    decompile_source,
    lower_component_artifact_to_reference,
    run_g8_reference_parity,
)
from mrun.decompiler.qwen35_stateful import (
    Qwen35TransactionalReference,
    certify_qwen35_stateful_two_token_parity,
)
from mrun.decompiler.reference import ReferenceExecutionError


def _config() -> dict[str, object]:
    text = {
        "model_type": "qwen3_5_text",
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 17,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "full_attention_interval": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "mlp_only_layers": [],
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "tie_word_embeddings": True,
        "use_cache": True,
        "eos_token_id": 2,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.75,
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 1],
        },
    }
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "transformers_version": "5.14.1",
        "image_token_id": 4,
        "video_token_id": 5,
        "vision_start_token_id": 6,
        "vision_end_token_id": 7,
        "vision_config": {"model_type": "qwen3_5_vision"},
        "text_config": text,
    }


def _weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(35)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.04

    output = {
        "model.language_model.embed_tokens.weight": random(17, 16),
        "model.language_model.norm.weight": torch.zeros(16),
        "model.visual.patch_embed.proj.weight": random(2, 2),
        "mtp.fc.weight": random(16, 32),
    }
    for layer in range(4):
        source = f"model.language_model.layers.{layer}"
        output.update(
            {
                f"{source}.input_layernorm.weight": torch.zeros(16),
                f"{source}.post_attention_layernorm.weight": torch.zeros(16),
                f"{source}.mlp.gate_proj.weight": random(24, 16),
                f"{source}.mlp.up_proj.weight": random(24, 16),
                f"{source}.mlp.down_proj.weight": random(16, 24),
            }
        )
        if layer == 3:
            output.update(
                {
                    f"{source}.self_attn.q_proj.weight": random(32, 16),
                    f"{source}.self_attn.k_proj.weight": random(8, 16),
                    f"{source}.self_attn.v_proj.weight": random(8, 16),
                    f"{source}.self_attn.o_proj.weight": random(16, 16),
                    f"{source}.self_attn.q_norm.weight": torch.zeros(8),
                    f"{source}.self_attn.k_norm.weight": torch.zeros(8),
                }
            )
        else:
            output.update(
                {
                    f"{source}.linear_attn.in_proj_qkv.weight": random(24, 16),
                    f"{source}.linear_attn.in_proj_z.weight": random(8, 16),
                    f"{source}.linear_attn.in_proj_a.weight": random(2, 16),
                    f"{source}.linear_attn.in_proj_b.weight": random(2, 16),
                    f"{source}.linear_attn.conv1d.weight": random(24, 1, 4),
                    f"{source}.linear_attn.A_log": random(2),
                    f"{source}.linear_attn.dt_bias": random(2),
                    f"{source}.linear_attn.norm.weight": torch.ones(4),
                    f"{source}.linear_attn.out_proj.weight": random(16, 8),
                }
            )
    return output


def _write_model(root: Path, *, extra_tensor: bool = False) -> dict[str, torch.Tensor]:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_config(), sort_keys=True), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    weights = _weights()
    if extra_tensor:
        weights["model.language_model.unowned.weight"] = torch.ones(1)
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    return weights


def test_qwen35_hybrid_text_adapter_has_total_scoped_coverage(tmp_path: Path) -> None:
    root = tmp_path / "qwen35"
    weights = _write_model(root)
    result = decompile_source(root, source_id="Test/TinyQwen3.5")

    assert result.succeeded
    assert result.report.selected_adapter_id == "mrun.hf.qwen3_5-hybrid-text"
    assert result.report.coverage is not None and result.report.coverage.complete
    assert result.report.coverage.source_tensor_count == len(weights)
    assert result.ir_bundle is not None
    bundle = result.ir_bundle
    ignored = {
        item.source_name
        for item in bundle.physical_weights.classifications
        if item.disposition == "ignored"
    }
    assert ignored == {"model.visual.patch_embed.proj.weight", "mtp.fc.weight"}
    kinds = {operation.kind for operation in bundle.model.operations}
    assert {
        "qwen35-gated-delta-recurrence",
        "qwen35-partial-interleaved-mrope",
        "qwen35-rms-norm",
    } <= kinds
    slot_kinds = {slot.kind for slot in bundle.state.slots}
    assert {
        "causal-depthwise-convolution-state",
        "gated-delta-recurrent-matrix",
        "paged-k-cache",
        "paged-v-cache",
        "position-counter",
    } == slot_kinds
    assert bundle.state.commit_protocol.protocol_id == "mrun-transactional-hybrid-state-v1"


def test_qwen35_text_scope_fails_closed_on_unowned_tensor(tmp_path: Path) -> None:
    root = tmp_path / "qwen35-extra"
    _write_model(root, extra_tensor=True)
    result = decompile_source(root)

    assert not result.succeeded
    assert result.report.status == "unsupported"
    assert "unowned" in json.dumps(result.report.as_dict())


def test_qwen35_adapter_fails_closed_on_layer_schedule_disagreement(tmp_path: Path) -> None:
    root = tmp_path / "qwen35-schedule"
    _write_model(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["text_config"]["layer_types"][2] = "full_attention"
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = decompile_source(root)
    assert not result.succeeded
    assert "full_attention_interval" in json.dumps(result.report.as_dict())


def test_qwen35_stateless_reference_has_exact_source_ir_parity(tmp_path: Path) -> None:
    root = tmp_path / "qwen35-reference"
    _write_model(root)
    built = build_component_artifact(root, tmp_path / "artifacts")
    executable = lower_component_artifact_to_reference(built.path)

    output = executable.forward([[1, 3, 5], [2, 4, 6]])
    assert output.logits.shape == (2, 3, 17)
    assert output.hidden_states.shape == (2, 3, 16)
    assert torch.isfinite(output.logits).all()
    assert executable.identity.source_model_type == "qwen3_5"
    assert executable.identity.stored_parameter_dtype == "mixed[F32]"

    certification = run_g8_reference_parity(
        executable,
        [
            [[1]],
            [[1, 3, 5]],
            [[2, 4, 6], [7, 8, 9]],
        ],
    )
    assert certification.execution_certified
    assert certification.maximum_absolute_error == 0.0
    assert certification.source_logits_fingerprints == certification.ir_logits_fingerprints


def test_qwen35_stateful_two_token_parity_and_transactional_branches(tmp_path: Path) -> None:
    root = tmp_path / "qwen35-stateful"
    _write_model(root)
    built = build_component_artifact(root, tmp_path / "artifacts")
    executable = lower_component_artifact_to_reference(built.path)
    full = executable.forward([[1, 3, 5, 7, 9]])
    receipt = certify_qwen35_stateful_two_token_parity(executable, [[1, 3, 5]], [[7]], [[9]])
    assert receipt.certified
    assert receipt.maximum_absolute_error <= receipt.absolute_tolerance

    runtime = Qwen35TransactionalReference(executable, batch=1, capacity=16)
    prefill_logits = runtime.prefill([[1, 3, 5]])
    torch.testing.assert_close(
        prefill_logits,
        full.logits[:, :3],
        atol=receipt.absolute_tolerance,
        rtol=receipt.relative_tolerance,
    )

    branches = runtime.stage_branches(([[7]], [[8]]))
    assert branches[0].parent_fingerprint == branches[1].parent_fingerprint
    parent = runtime.snapshot()
    runtime.rollback(branches[1])
    assert runtime.snapshot() == parent
    committed = runtime.commit(branches[0])
    assert committed.length == 4 and committed.epoch == parent.epoch + 1
    torch.testing.assert_close(
        branches[0].logits[:, 0],
        full.logits[:, 3],
        atol=receipt.absolute_tolerance,
        rtol=receipt.relative_tolerance,
    )

    with pytest.raises(ReferenceExecutionError, match="already committed or rolled back"):
        runtime.commit(branches[1])

    second = runtime.stage([[9]])
    torch.testing.assert_close(
        second.logits[:, 0],
        full.logits[:, 4],
        atol=receipt.absolute_tolerance,
        rtol=receipt.relative_tolerance,
    )
    final = runtime.commit(second)
    assert final.length == 5


def test_qwen35_stateful_prefix_commit_discards_unaccepted_tail(tmp_path: Path) -> None:
    root = tmp_path / "qwen35-prefix-commit"
    _write_model(root)
    built = build_component_artifact(root, tmp_path / "artifacts")
    runtime = Qwen35TransactionalReference(built.path, batch=1, capacity=8)

    proposal = runtime.stage([[1, 2, 3]])
    committed = runtime.commit(proposal, accepted_tokens=2)
    assert committed.length == 2
    continuation = runtime.stage([[4]])
    executable = lower_component_artifact_to_reference(built.path)
    full = executable.forward([[1, 2, 4]])
    receipt = certify_qwen35_stateful_two_token_parity(executable, [[1, 2]], [[4]], [[5]])
    assert receipt.certified
    torch.testing.assert_close(
        continuation.logits[:, 0],
        full.logits[:, 2],
        atol=receipt.absolute_tolerance,
        rtol=receipt.relative_tolerance,
    )
