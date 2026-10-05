from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mrun.decompiler import (
    NativeLoweringUnavailable,
    ReferenceExecutable,
    ReferenceExecutionError,
    ReferenceLoweringError,
    ReferenceParityError,
    build_component_artifact,
    certify_component_artifact,
    lower_component_artifact_to_reference,
    promote_reference_target,
    require_native_lowering,
    run_g8_reference_parity,
)


def _config(family: str, *, tied: bool) -> dict[str, object]:
    architecture = {
        "qwen2": "Qwen2ForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
        "llama": "LlamaForCausalLM",
    }[family]
    config: dict[str, object] = {
        "architectures": [architecture],
        "model_type": family,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
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
    if family == "qwen3":
        config["attention_bias"] = False
    if family == "llama":
        config.update({"attention_bias": False, "mlp_bias": False, "pretraining_tp": 1})
    return config


def _weights(
    family: str,
    *,
    tied: bool,
    mixed_dtype: bool = False,
    serialized_rope: bool = False,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1000 + {"qwen2": 2, "qwen3": 3, "llama": 4}[family])

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.08

    output: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": random(11, 8),
        "model.norm.weight": torch.ones(8, dtype=torch.float32),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        output.update(
            {
                f"{prefix}.input_layernorm.weight": torch.ones(8, dtype=torch.float32),
                f"{prefix}.post_attention_layernorm.weight": torch.ones(8, dtype=torch.float32),
                f"{prefix}.self_attn.q_proj.weight": random(8, 8),
                f"{prefix}.self_attn.k_proj.weight": random(4, 8),
                f"{prefix}.self_attn.v_proj.weight": random(4, 8),
                f"{prefix}.self_attn.o_proj.weight": random(8, 8),
                f"{prefix}.mlp.gate_proj.weight": random(16, 8),
                f"{prefix}.mlp.up_proj.weight": random(16, 8),
                f"{prefix}.mlp.down_proj.weight": random(8, 16),
            }
        )
        if family == "qwen2":
            output.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": random(8),
                    f"{prefix}.self_attn.k_proj.bias": random(4),
                    f"{prefix}.self_attn.v_proj.bias": random(4),
                }
            )
        if family == "qwen3":
            output.update(
                {
                    f"{prefix}.self_attn.q_norm.weight": torch.ones(2, dtype=torch.float32),
                    f"{prefix}.self_attn.k_norm.weight": torch.ones(2, dtype=torch.float32),
                }
            )
    if not tied:
        output["lm_head.weight"] = random(11, 8)
    if serialized_rope:
        output["model.rotary_emb.inv_freq"] = torch.tensor([1.0], dtype=torch.float32)
    if mixed_dtype:
        output["model.layers.0.self_attn.q_proj.weight"] = output[
            "model.layers.0.self_attn.q_proj.weight"
        ].to(torch.float16)
    return output


def _write_model(
    root: Path,
    family: str,
    *,
    tied: bool,
    mixed_dtype: bool = False,
    serialized_rope: bool = False,
) -> None:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(_config(family, tied=tied), sort_keys=True), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, sort_keys=True), encoding="utf-8"
    )
    save_file(
        _weights(
            family,
            tied=tied,
            mixed_dtype=mixed_dtype,
            serialized_rope=serialized_rope,
        ),
        root / "model.safetensors",
        metadata={"format": "pt"},
    )


@pytest.mark.parametrize(
    ("family", "tied", "serialized_rope"),
    [
        ("qwen2", True, False),
        ("qwen3", False, False),
        ("llama", False, True),
    ],
)
def test_dense_reference_target_runs_exact_source_vs_ir_g8(
    tmp_path: Path, family: str, tied: bool, serialized_rope: bool
) -> None:
    source = tmp_path / family
    _write_model(source, family, tied=tied, serialized_rope=serialized_rope)
    built = build_component_artifact(source, tmp_path / "artifacts")

    executable = lower_component_artifact_to_reference(built.path)
    output = executable.forward(torch.tensor([[1, 4, 7], [2, 3, 5]], dtype=torch.int64))

    assert output.logits.shape == (2, 3, 11)
    assert output.hidden_states.shape == (2, 3, 8)
    assert torch.isfinite(output.logits).all()
    assert not executable.identity.execution_certified
    assert not executable.identity.production_runtime_eligible
    assert executable.identity.native_lowering_status == "not-lowered"

    certification = run_g8_reference_parity(
        executable,
        [
            [[1]],
            [[1, 4, 7]],
            [[2, 3, 5], [6, 7, 8]],
        ],
    )
    assert certification.execution_performed
    assert certification.execution_certified
    assert not certification.production_runtime_eligible
    assert certification.maximum_absolute_error == 0.0
    assert certification.source_logits_fingerprints == certification.ir_logits_fingerprints

    promotion = promote_reference_target(executable, certification)
    assert promotion.status == "promoted-reference-only"
    assert not promotion.production_runtime_eligible
    with pytest.raises(NativeLoweringUnavailable, match="does not lower"):
        require_native_lowering(promotion)


def test_tied_lexical_views_preserve_object_identity(tmp_path: Path) -> None:
    source = tmp_path / "qwen2"
    _write_model(source, "qwen2", tied=True)
    built = build_component_artifact(source, tmp_path / "artifacts")
    executable = lower_component_artifact_to_reference(built.path)

    embedding = executable.store.tensor("token_embedding.weight")
    readout = executable.store.tensor("lm_head.weight")

    assert embedding is readout
    assert embedding.data_ptr() == readout.data_ptr()


def test_reference_target_rejects_mixed_executable_dtypes(tmp_path: Path) -> None:
    source = tmp_path / "llama"
    _write_model(source, "llama", tied=False, mixed_dtype=True)
    built = build_component_artifact(source, tmp_path / "artifacts")

    with pytest.raises(ReferenceLoweringError, match="one stored dtype"):
        lower_component_artifact_to_reference(built.path)


def test_execution_rechecks_blob_hash_after_lowering(tmp_path: Path) -> None:
    source = tmp_path / "qwen3"
    _write_model(source, "qwen3", tied=False)
    built = build_component_artifact(source, tmp_path / "artifacts")
    executable = lower_component_artifact_to_reference(built.path)
    manifest = executable.artifact.manifest
    blob = executable.artifact.directory / manifest["allocations"][0]["blob"]["path"]
    payload = bytearray(blob.read_bytes())
    payload[0] ^= 0x01
    blob.write_bytes(payload)

    with pytest.raises(ReferenceExecutionError, match="content check"):
        executable.forward([[1, 2]])


def test_input_domain_and_context_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "qwen2"
    _write_model(source, "qwen2", tied=True)
    built = build_component_artifact(source, tmp_path / "artifacts")
    executable = lower_component_artifact_to_reference(built.path)

    with pytest.raises(ReferenceExecutionError, match="outside"):
        executable.forward([[11]])
    with pytest.raises(ReferenceExecutionError, match="integer"):
        executable.forward(torch.tensor([[1.0]]))
    with pytest.raises(ReferenceExecutionError, match="context"):
        executable.forward(torch.zeros((1, 65), dtype=torch.int64))


def test_g8_mismatch_never_yields_a_promotion_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "llama"
    _write_model(source, "llama", tied=False)
    built = build_component_artifact(source, tmp_path / "artifacts")
    executable = lower_component_artifact_to_reference(built.path)
    original = ReferenceExecutable.forward

    def shifted(self: ReferenceExecutable, token_ids: torch.Tensor) -> object:
        result = original(self, token_ids)
        return type(result)(logits=result.logits + 1.0, hidden_states=result.hidden_states)

    monkeypatch.setattr(ReferenceExecutable, "forward", shifted)
    with pytest.raises(ReferenceParityError, match="differ"):
        run_g8_reference_parity(executable, [[[1, 2, 3]]])


def test_integrity_record_cannot_promote_reference_execution(tmp_path: Path) -> None:
    source = tmp_path / "qwen2"
    _write_model(source, "qwen2", tied=True)
    built = build_component_artifact(source, tmp_path / "artifacts")
    executable = lower_component_artifact_to_reference(built.path)
    structural = certify_component_artifact(built.path)

    with pytest.raises(TypeError, match="execution certification"):
        promote_reference_target(executable, structural)  # type: ignore[arg-type]


def test_lowering_reopens_instead_of_trusting_mutated_object(tmp_path: Path) -> None:
    source = tmp_path / "qwen3"
    _write_model(source, "qwen3", tied=False)
    built = build_component_artifact(source, tmp_path / "artifacts")
    first = lower_component_artifact_to_reference(built.path)

    # Passing the object cannot bypass canonical inventory/hash validation: the API reopens its
    # directory and derives a fresh target identity from disk.
    second = lower_component_artifact_to_reference(first.artifact)
    assert second.artifact is not first.artifact
    assert second.identity == first.identity
