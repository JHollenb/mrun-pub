from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mrun.decompiler import (
    ArtifactVerificationError,
    SourceCudaInt8LoweringError,
    VerifiedSourceMlxArtifact,
    build_component_artifact,
    build_source_cuda_int8_artifact,
    build_source_mlx_artifact,
    decompile_source,
    lower_component_artifact_to_reference,
    open_component_artifact,
    run_g8_reference_parity,
    run_g13_mixtral_semantic_parity,
)

HIDDEN = 8
INTERMEDIATE = 12
LAYERS = 2
EXPERTS = 3
TOP_K = 2
VOCAB = 11
PHYSICAL_ROWS = 13


def _config(*, tied: bool, **updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["MixtralForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": [2, 3],
        "head_dim": 4,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "max_position_embeddings": 64,
        "model_type": "mixtral",
        "num_attention_heads": 2,
        "num_experts_per_tok": TOP_K,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": 1,
        "num_local_experts": EXPERTS,
        "output_router_logits": False,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"},
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.0,
        "sliding_window": None,
        "tie_word_embeddings": tied,
        "torch_dtype": "float32",
        "use_cache": True,
        "vocab_size": VOCAB,
    }
    config.update(updates)
    return config


def _weights(*, tied: bool) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0xA11CE)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02

    weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": random(PHYSICAL_ROWS, HIDDEN),
        "model.norm.weight": torch.ones(HIDDEN, dtype=torch.float32),
    }
    if not tied:
        weights["lm_head.weight"] = random(PHYSICAL_ROWS, HIDDEN)
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}"
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": torch.ones(HIDDEN),
                f"{prefix}.post_attention_layernorm.weight": torch.ones(HIDDEN),
                f"{prefix}.self_attn.q_proj.weight": random(HIDDEN, HIDDEN),
                f"{prefix}.self_attn.k_proj.weight": random(4, HIDDEN),
                f"{prefix}.self_attn.v_proj.weight": random(4, HIDDEN),
                f"{prefix}.self_attn.o_proj.weight": random(HIDDEN, HIDDEN),
                f"{prefix}.block_sparse_moe.gate.weight": random(EXPERTS, HIDDEN),
            }
        )
        for expert in range(EXPERTS):
            expert_prefix = f"{prefix}.block_sparse_moe.experts.{expert}"
            weights.update(
                {
                    f"{expert_prefix}.w1.weight": random(INTERMEDIATE, HIDDEN),
                    f"{expert_prefix}.w2.weight": random(HIDDEN, INTERMEDIATE),
                    f"{expert_prefix}.w3.weight": random(INTERMEDIATE, HIDDEN),
                }
            )
    return weights


def _write_mixtral(
    root: Path,
    *,
    tied: bool = False,
    config_updates: dict[str, object] | None = None,
    weights: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    root.mkdir()
    config = _config(tied=tied, **(config_updates or {}))
    tensors = weights if weights is not None else _weights(tied=tied)
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, sort_keys=True), encoding="utf-8"
    )
    (root / "generation_config.json").write_text(
        json.dumps({"do_sample": False}, sort_keys=True), encoding="utf-8"
    )
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    return tensors


def test_mixtral_u2_inventory_topology_state_io_and_canonical_reopen(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixtral"
    tensors = _write_mixtral(source)

    result = decompile_source(source)

    assert result.succeeded
    assert result.report.universal_level == "U2"
    assert result.report.selected_adapter_id == "mrun.hf.mixtral-sparse-moe"
    assert result.report.coverage is not None and result.report.coverage.complete
    assert result.report.coverage.source_tensor_count == len(tensors)
    assert result.ir_bundle is not None
    bundle = result.ir_bundle
    physical = bundle.physical_weights
    assert {item.source_name for item in physical.classifications} == set(tensors)
    assert all(item.disposition == "parameter" for item in physical.classifications)
    roles = {view.semantic_role for view in physical.views}
    assert "moe-router-routed-only" in roles
    assert {
        "moe-routed-expert-gate",
        "moe-routed-expert-up",
        "moe-routed-expert-down",
    } <= roles
    assert not any("shared" in view.logical_name for view in physical.views)

    match = next(
        item
        for item in result.report.match_results
        if item.adapter_id == "mrun.hf.mixtral-sparse-moe"
    )
    evidence = {item.predicate: item for item in match.evidence}
    assert evidence["tensor.anchor.layer0_router"].matched
    assert evidence["tensor.anchor.layer0_expert0_w1"].matched

    operations = {item.operation_id: item for item in bundle.model.operations}
    router = operations["layers.0.moe.router"]
    assert router.kind == "moe-router-linear"
    assert router.attributes == {
        "expert_scope": "routed-only",
        "num_routed_experts": EXPERTS,
        "num_shared_experts": 0,
        "router_bias": False,
        "weight_orientation": "experts-hidden",
    }
    top_k = operations["layers.0.moe.top_k"]
    assert top_k.kind == "moe-top-k-softmax"
    assert top_k.attributes["softmax_dtype"] == "float32"
    assert top_k.attributes["num_experts_per_token"] == TOP_K
    assert top_k.attributes["renormalize_selected_probabilities"] is True
    dispatch = operations["layers.0.moe.dispatch"]
    assert len(dispatch.outputs) == EXPERTS
    assert dispatch.attributes["num_shared_experts"] == 0
    combine = operations["layers.0.moe.combine"]
    assert combine.kind == "moe-weighted-scatter-add"
    assert combine.attributes["accumulation_order"] == "source-expert-index-order"
    for expert in range(EXPERTS):
        expert_prefix = f"layers.0.moe.routed_experts.{expert}"
        assert operations[f"{expert_prefix}.gate_proj"].attributes["expert_id"] == expert
        assert operations[f"{expert_prefix}.down_proj"].parameters == (
            f"{expert_prefix}.down_proj.weight",
        )

    assert {slot.slot_id for slot in bundle.state.slots} == {
        "position",
        *(f"layers.{layer}.{kind}_cache" for layer in range(LAYERS) for kind in ("k", "v")),
    }
    row_mappers = {item.mapper_id: item for item in bundle.io.row_mappers}
    assert row_mappers["tokens-to-input-rows"].kind == "padded-identity"
    assert row_mappers["tokens-to-output-rows"].kind == "padded-identity"
    assert row_mappers["tokens-to-input-rows"].token_count == VOCAB
    assert row_mappers["tokens-to-input-rows"].row_count == PHYSICAL_ROWS
    assert row_mappers["tokens-to-input-rows"].unreachable_rows == (11, 12)

    record = build_component_artifact(source, tmp_path / "canonical")
    reopened = open_component_artifact(record.path)
    assert record.verified_reopen
    assert reopened.artifact_id == record.artifact_id
    assert reopened.ir_bundle.fingerprint == bundle.fingerprint
    assert len(reopened.manifest["allocations"]) == len(tensors)
    assert reopened.manifest["status"] == "built-unexecuted"
    assert reopened.manifest["execution_certified"] is False

    executable = lower_component_artifact_to_reference(record.path)
    forward = executable.forward(torch.tensor([[1, 4, 7], [2, 3, 5]], dtype=torch.int64))
    assert forward.logits.shape == (2, 3, VOCAB)
    assert forward.hidden_states.shape == (2, 3, HIDDEN)
    assert torch.isfinite(forward.logits).all()
    certification = run_g8_reference_parity(
        executable,
        [
            [[1]],
            [[1, 4, 7]],
            [[2, 3, 5], [6, 7, 8]],
        ],
    )
    assert certification.execution_certified
    assert certification.maximum_absolute_error == 0.0
    assert certification.maximum_relative_error == 0.0
    assert certification.source_logits_fingerprints == certification.ir_logits_fingerprints
    semantic = run_g13_mixtral_semantic_parity(
        executable,
        [
            [[1]],
            [[1, 4, 7]],
            [[2, 3, 5], [6, 7, 8]],
        ],
    )
    assert semantic.execution_certified
    assert not semantic.production_runtime_eligible
    assert semantic.maximum_absolute_error == 0.0
    assert semantic.maximum_relative_error == 0.0
    assert semantic.source_observation_fingerprints == semantic.ir_observation_fingerprints
    assert "layers.0.moe.selected_experts" in semantic.observation_names
    assert "layers.1.moe.routed_experts.2.output" in semantic.observation_names
    native_record = build_source_mlx_artifact(record.path, tmp_path / "mlx")
    native_artifact = VerifiedSourceMlxArtifact(native_record.path)
    assert native_record.verified_reopen
    assert native_artifact.source["architecture"] == "mixtral"
    assert native_artifact.config["vocab_size"] == PHYSICAL_ROWS
    assert native_artifact.source["runtime_numerical_compatibility"] == (
        "mlx-lm-mixtral-topk-switchglu-bounded-parity-required"
    )
    with pytest.raises(SourceCudaInt8LoweringError, match="intentionally closed"):
        build_source_cuda_int8_artifact(record.path, tmp_path / "cuda")


def test_mixtral_direct_unquantized_mlx_is_bounded_and_loadable(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.utils import load_model

    source = tmp_path / "mixtral-mlx"
    _write_mixtral(source)
    canonical = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(canonical.path)
    record = build_source_mlx_artifact(canonical.path, tmp_path / "native")
    artifact = VerifiedSourceMlxArtifact(record.path)

    model, _loaded_config = load_model(artifact.path, lazy=False, strict=True)
    tokens = torch.tensor([[1, 4, 7, 2]], dtype=torch.int64)
    native = model(mx.array(tokens.numpy().astype(np.int32, copy=False)))
    mx.eval(native)
    native_logits = torch.from_numpy(np.array(native[..., :VOCAB].astype(mx.float32), copy=False))
    reference_logits = executable.forward(tokens).logits.to(torch.float32)

    torch.testing.assert_close(native_logits, reference_logits, rtol=3e-3, atol=3e-4)


@pytest.mark.parametrize("tied", [False, True])
def test_mixtral_lexical_tie_is_evidence_bound(tmp_path: Path, tied: bool) -> None:
    source = tmp_path / f"mixtral-tied-{tied}"
    _write_mixtral(source, tied=tied)

    result = decompile_source(source)

    assert result.succeeded and result.ir_bundle is not None
    physical = result.ir_bundle.physical_weights
    aliases = physical.alias_classes
    views = {item.logical_name: item for item in physical.views}
    if tied:
        assert len(aliases) == 1
        assert aliases[0].logical_names == ("lm_head.weight", "token_embedding.weight")
        assert (
            views["lm_head.weight"].allocation_id == views["token_embedding.weight"].allocation_id
        )
        assert aliases[0].evidence.kind == "missing-serialized-tied-readout"
    else:
        assert not aliases
        assert (
            views["lm_head.weight"].allocation_id != views["token_embedding.weight"].allocation_id
        )


@pytest.mark.parametrize(
    ("tied", "serialize_head"),
    [(True, True), (False, False)],
)
def test_mixtral_contradictory_lexical_alias_evidence_is_rejected(
    tmp_path: Path, tied: bool, serialize_head: bool
) -> None:
    weights = _weights(tied=tied)
    if serialize_head:
        weights["lm_head.weight"] = torch.zeros(PHYSICAL_ROWS, HIDDEN)
    else:
        weights.pop("lm_head.weight")
    source = tmp_path / f"mixtral-alias-conflict-{tied}"
    _write_mixtral(source, tied=tied, weights=weights)

    result = decompile_source(source)

    assert not result.succeeded
    assert result.report.failures[0].code == "alias_evidence_failure"


@pytest.mark.parametrize(
    ("updates", "feature"),
    [
        ({"future_router_mode": "v2"}, "unknown_config_keys"),
        ({"architectures": ["FutureMixtralForCausalLM"]}, "architecture_declaration"),
        ({"architectures": ["MixtralModel"]}, "architecture_declaration"),
        ({"attention_bias": True}, "invalid_semantic_config"),
        ({"num_experts_per_tok": EXPERTS}, "invalid_semantic_config"),
        ({"output_router_logits": True}, "router_output_contract"),
        ({"quantization_config": {"bits": 4}}, "unregistered_source_codec"),
        ({"rope_scaling": {"factor": 2.0, "rope_type": "linear"}}, "rope_scaling"),
        ({"router_jitter_noise": 0.1}, "stochastic_router"),
        ({"sliding_window": 32}, "sliding_attention"),
    ],
)
def test_mixtral_unregistered_semantic_variants_fail_closed(
    tmp_path: Path, updates: dict[str, object], feature: str
) -> None:
    source = tmp_path / feature
    _write_mixtral(source, config_updates=updates)

    result = decompile_source(source)

    assert not result.succeeded
    assert result.report.failures[0].code == "unsupported_variant"
    assert feature in json.dumps(result.report.failures[0].details)


def test_mixtral_unknown_shared_expert_and_fused_source_schema_are_rejected(
    tmp_path: Path,
) -> None:
    unknown_weights = _weights(tied=False)
    unknown_name = "model.layers.0.block_sparse_moe.shared_expert.w1.weight"
    unknown_weights[unknown_name] = torch.zeros(INTERMEDIATE, HIDDEN)
    unknown_source = tmp_path / "unknown-shared"
    _write_mixtral(unknown_source, weights=unknown_weights)

    unknown = decompile_source(unknown_source)

    assert not unknown.succeeded
    assert unknown.report.failures[0].code == "source_coverage_failure"
    assert unknown_name in unknown.report.failures[0].details["unexplained_tensors"]

    fused_weights = _weights(tied=False)
    for expert in range(EXPERTS):
        prefix = f"model.layers.0.block_sparse_moe.experts.{expert}"
        for projection in ("w1", "w2", "w3"):
            fused_weights.pop(f"{prefix}.{projection}.weight")
    fused_weights["model.layers.0.mlp.experts.gate_up_proj"] = torch.zeros(
        EXPERTS, 2 * INTERMEDIATE, HIDDEN
    )
    fused_weights["model.layers.0.mlp.experts.down_proj"] = torch.zeros(
        EXPERTS, HIDDEN, INTERMEDIATE
    )
    fused_source = tmp_path / "fused-source"
    _write_mixtral(fused_source, weights=fused_weights)

    fused = decompile_source(fused_source)

    assert not fused.succeeded
    assert fused.report.failures[0].code == "source_coverage_failure"
    assert "missing_tensors" in fused.report.failures[0].details


def test_mixtral_canonical_artifact_rejects_blob_tamper_and_extra_inventory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixtral"
    _write_mixtral(source)
    tampered = build_component_artifact(source, tmp_path / "tampered")
    manifest = json.loads((tampered.path / "manifest.json").read_text(encoding="utf-8"))
    blob_path = tampered.path / manifest["allocations"][0]["blob"]["path"]
    blob = bytearray(blob_path.read_bytes())
    blob[0] ^= 0xFF
    blob_path.write_bytes(blob)

    with pytest.raises(ArtifactVerificationError, match="content hash mismatch"):
        open_component_artifact(tampered.path)

    extra = build_component_artifact(source, tmp_path / "extra")
    (extra.path / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="inventory"):
        open_component_artifact(extra.path)


def test_mixtral_g13_covers_router_ties_and_empty_expert_dispatch(tmp_path: Path) -> None:
    weights = _weights(tied=False)
    for layer in range(LAYERS):
        weights[f"model.layers.{layer}.block_sparse_moe.gate.weight"] = torch.zeros(
            EXPERTS, HIDDEN, dtype=torch.float32
        )
    source = tmp_path / "mixtral-router-ties"
    _write_mixtral(source, weights=weights)
    record = build_component_artifact(source, tmp_path / "canonical")
    executable = lower_component_artifact_to_reference(record.path)
    token_ids = torch.tensor([[1, 4, 7], [2, 3, 5]], dtype=torch.int64)

    certification = run_g13_mixtral_semantic_parity(executable, [token_ids])
    trace = executable.trace(
        token_ids,
        [
            "layers.0.moe.selected_experts",
            *(f"layers.0.moe.routed_experts.{expert}.output" for expert in range(EXPERTS)),
        ],
    )

    selected = trace["layers.0.moe.selected_experts"]
    assert torch.equal(selected, selected[:1, :1].expand_as(selected))
    empty_experts = [
        expert
        for expert in range(EXPERTS)
        if trace[f"layers.0.moe.routed_experts.{expert}.output"].shape[0] == 0
    ]
    assert len(empty_experts) == EXPERTS - TOP_K
    assert certification.maximum_absolute_error == 0.0
