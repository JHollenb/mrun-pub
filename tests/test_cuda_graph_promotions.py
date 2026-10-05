from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.compiler.identity import bind_loaded_qstore_identity
from mrun.compiler.promotions import (
    CUDA_GRAPH_PROMOTION_REGISTRY_SCHEMA,
    CudaGraphPromotionRecord,
    build_cuda_graph_promotion_record,
    cuda_graph_promotion_registry_payload,
    default_cuda_graph_promotion_registry,
    load_cuda_graph_promotions,
    select_cuda_graph_promotion,
)
from mrun.testing.qstore_identity import (
    install_verified_test_identity,
    verified_test_manifest,
)


def _engine():
    store = SimpleNamespace(
        compute_dtype=torch.bfloat16,
        man=verified_test_manifest(
            {
                "model_name": "tiny",
                "arch": "qwen2",
                "dtype": "int8",
                "config": {"vocab_size": 32},
                "blocks": {"lm_head": {"kind": "qrow"}},
            }
        ),
    )
    install_verified_test_identity(store)
    return SimpleNamespace(
        backend="dense-qstore-cuda",
        device="cuda:0",
        name="tiny",
        store=store,
        cuda_graph_runtime_environment={
            "runtime_source_sha256": "7" * 64,
            "runtime_distribution_sha256": "6" * 64,
            "wheel_sha256": "8" * 64,
            "torch_version": "test-torch",
            "torch_cuda_version": "test-cuda",
            "cuda_driver_version": 1,
            "triton_version": "test-triton",
            "device": "cuda:0",
            "gpu_name": "test-gpu",
            "gpu_capability": [8, 9],
            "gpu_total_memory_bytes": 16_000_000,
            "store_reverified_at_ns": 1,
        },
    )


def _record(engine, **overrides):
    identity = bind_loaded_qstore_identity(engine)
    values = {
        "promotion_id": "test-promotion",
        "source_checkpoint_sha256": identity.model_revision,
        "derived_store_sha256": identity.store_fingerprint,
        "manifest_semantic_sha256": identity.manifest_semantic_sha256,
        "identity_certificate_sha256": identity.identity_certificate_sha256,
        "activation_dtype": "bf16",
        "weight_dtype": "int8",
        "accumulator_dtype": "fp32",
        "runtime_source_sha256": "7" * 64,
        "runtime_distribution_sha256": "6" * 64,
        "evidence_wheel_sha256": "8" * 64,
        "evidence_file_sha256": "9" * 64,
        "torch_version": "test-torch",
        "torch_cuda_version": "test-cuda",
        "gpu_name": "test-gpu",
        "gpu_capability": (8, 9),
        "batch_size": 1,
        "sequence_length": 3,
        "union_candidate_count": 2,
        "capture_total_residency_bytes": 1_000,
        "capture_setup_ms": 10.0,
        "break_even_replays": 4,
        "measured_speedup": 3.0,
        "ratio_ci95": (2.5, 3.5),
        "exact_parity": True,
        "wins": 10,
        "trials": 10,
    }
    values.update(overrides)
    return CudaGraphPromotionRecord(**values)


def test_exact_promotion_requires_identity_runtime_shape_amortization_and_vram():
    engine = _engine()
    record = _record(engine)
    selected = select_cuda_graph_promotion(
        engine,
        np.asarray([1, 2, 3]),
        (4, 5),
        expected_replays=4,
        minimum_improvement_percent=1.0,
        promotions=(record,),
        free_cuda_bytes=record.required_free_bytes,
    )

    assert selected.selected
    assert selected.promotion == record
    assert selected.as_dict()["promotion_fingerprint"] == record.fingerprint


@pytest.mark.parametrize(
    ("token_ids", "expected_replays", "free_cuda_bytes", "blocker"),
    (
        (np.asarray([1, 2]), 4, 10_000, "no-exact-promotion-record"),
        (np.asarray([1, 2, 3]), 3, 10_000, "below break-even"),
        (np.asarray([1, 2, 3]), 4, 1, "free CUDA memory"),
    ),
)
def test_promotion_falls_back_when_one_admission_dimension_does_not_match(
    token_ids,
    expected_replays,
    free_cuda_bytes,
    blocker,
):
    engine = _engine()
    decision = select_cuda_graph_promotion(
        engine,
        token_ids,
        (4, 5),
        expected_replays=expected_replays,
        promotions=(_record(engine),),
        free_cuda_bytes=free_cuda_bytes,
    )

    assert not decision.selected
    assert any(blocker in value for value in decision.blockers)


def test_promotion_record_rejects_nonexact_or_losing_evidence():
    engine = _engine()
    with pytest.raises(ValueError, match="exact replay parity"):
        _record(engine, exact_parity=False)
    with pytest.raises(ValueError, match="every paired trial"):
        _record(engine, wins=9)
    with pytest.raises(ValueError, match="performance"):
        _record(engine, ratio_ci95=(0.9, 3.5))


def test_promotion_record_is_derived_from_bound_passing_hardware_evidence():
    engine = _engine()
    identity = bind_loaded_qstore_identity(engine)
    plan = SimpleNamespace(
        model_revision=identity.model_revision,
        store_fingerprint=identity.store_fingerprint,
        content_identity_verified=True,
        precision=SimpleNamespace(
            activation_dtype="bf16",
            weight_dtype="int8",
            accumulator_dtype="fp32",
        ),
        shape=SimpleNamespace(actual_batch=1, sequence_length=3),
    )
    campaign = SimpleNamespace(
        fingerprint="a" * 64,
        union_token_ids=(4, 5),
        base_bundle=SimpleNamespace(plan=plan),
    )
    benchmark = SimpleNamespace(
        campaign_fingerprint=campaign.fingerprint,
        union_candidate_count=2,
        capture_improvement_demonstrated=True,
        output_parity=SimpleNamespace(exact=True),
        matched_eager_to_cuda_graph=SimpleNamespace(
            wins=10,
            ci95=(2.5, 3.5),
        ),
        trials=10,
        break_even_replays=4,
        runtime_environment={
            **engine.cuda_graph_runtime_environment,
            "gpu_capability": (8, 9),
        },
        capture_executor_evidence={
            "capture_total_residency_delta_bytes": 1_000,
        },
        capture_setup_ms=10.0,
        speedup=3.0,
    )

    record = build_cuda_graph_promotion_record(
        engine,
        campaign,
        benchmark,
        promotion_id="derived-test",
        evidence_file_sha256="9" * 64,
    )
    registry = cuda_graph_promotion_registry_payload((record,))

    assert record.source_checkpoint_sha256 == identity.model_revision
    assert record.runtime_distribution_sha256 == "6" * 64
    assert registry["records"][0]["promotion_fingerprint"] == record.fingerprint


@pytest.mark.parametrize(
    ("environment_field", "replacement"),
    (
        ("runtime_distribution_sha256", "5" * 64),
        ("wheel_sha256", "4" * 64),
    ),
)
def test_promotion_rejects_runtime_payload_or_wheel_drift(
    environment_field,
    replacement,
):
    engine = _engine()
    record = _record(engine)
    engine.cuda_graph_runtime_environment[environment_field] = replacement

    decision = select_cuda_graph_promotion(
        engine,
        np.asarray([1, 2, 3]),
        (4, 5),
        expected_replays=4,
        promotions=(record,),
        free_cuda_bytes=record.required_free_bytes,
    )

    assert not decision.selected
    assert decision.blockers == ("no-exact-promotion-record",)


def test_external_registry_round_trips_and_binds_fingerprint(tmp_path, monkeypatch):
    engine = _engine()
    record = _record(engine)
    registry = tmp_path / "cuda-graph-promotions.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": CUDA_GRAPH_PROMOTION_REGISTRY_SCHEMA,
                "records": [record.as_dict()],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MRUN_CUDA_GRAPH_PROMOTIONS", str(registry))

    assert default_cuda_graph_promotion_registry() == registry
    assert load_cuda_graph_promotions() == (record,)
    decision = select_cuda_graph_promotion(
        engine,
        np.asarray([1, 2, 3]),
        (4, 5),
        expected_replays=4,
        free_cuda_bytes=record.required_free_bytes,
    )
    assert decision.selected
    assert decision.promotion == record


def test_external_registry_fails_closed_on_tampering(tmp_path):
    engine = _engine()
    payload = _record(engine).as_dict()
    payload["capture_setup_ms"] = 11.0
    registry = tmp_path / "cuda-graph-promotions.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": CUDA_GRAPH_PROMOTION_REGISTRY_SCHEMA,
                "records": [payload],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_cuda_graph_promotions(registry)
