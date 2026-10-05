from __future__ import annotations

import pytest

from mrun.compiler.phase_evidence import (
    ContinuousBatchComputeEvidence,
    ContinuousBatchEvidence,
    PrecisionOversizedEvidence,
    ProfileDrivenFusionEvidence,
    StatefulDecodeEvidence,
    TargetAlignedK4Evidence,
)


def test_p4_direct_artifact_is_compute_evidence_not_service_evidence() -> None:
    cells = []
    for batch in (2, 4, 8, 16, 32):
        cells.append(
            {
                "batch_size": batch,
                "independent_b1_exact_winners": True,
                "independent_b1_max_abs_score_delta": 1e-6,
                "max_abs_score_delta": 0.0,
                "row_interference_passed": True,
                "p95_graph_seconds": 0.005,
                "speedup_lower_95": 2.0,
                "executor_evidence": {"capture_peak_allocated_delta_bytes": 123},
            }
        )
    evidence = ContinuousBatchComputeEvidence.from_gate_artifact(
        {
            "schema": "mrun-resident-continuous-batch-cuda-gate-v1",
            "qualified": True,
            "numerical_contract": "row-stable",
            "atol": 2e-4,
            "arena_bytes": 456,
            "cells": cells,
        }
    )
    assert evidence.batch_sizes == (2, 4, 8, 16, 32)
    assert evidence.p95_execution_ms == 5.0
    combined = ContinuousBatchEvidence.from_compute_and_service_artifacts(
        evidence,
        {
            "schema": "mrun-resident-continuous-batch-service-cuda-gate-v1",
            "qualified": True,
            "numerical_contract": "row-stable",
            "cancellation_passed": True,
            "refill_passed": True,
            "peak_vram_bytes": 789,
            "cells": [
                {
                    "batch_size": batch,
                    "max_abs_score_delta": 0.0,
                    "speedup_lower_95": 1.5,
                }
                for batch in (2, 4, 8, 16, 32)
            ],
            "service_telemetry": {
                "fallback_rows": 0,
                "captured_max_width": 32,
                "execution_p50_ms": 4.0,
                "execution_p95_ms": 5.0,
                "queue_p95_ms": 2.0,
            },
        },
    )
    assert combined.peak_vram_bytes == 789


def test_p4_evidence_requires_complete_family() -> None:
    evidence = ContinuousBatchEvidence(
        (2, 4, 8, 16, 32), True, True, 1.0, 2.0, 0.5, 1.1, 100
    )
    assert evidence.batch_sizes[-1] == 32
    with pytest.raises(ValueError, match="complete"):
        ContinuousBatchEvidence((2, 4), True, True, 1.0, 2.0, 0.5, 1.1, 100)


def test_p5_evidence_requires_more_than_five_percent() -> None:
    assert ProfileDrivenFusionEvidence(9, True, 2, 100, True, 1.06).bytes_removed == 100
    with pytest.raises(ValueError, match="1.05"):
        ProfileDrivenFusionEvidence(9, True, 2, 100, True, 1.05)


def test_p6_evidence_requires_full_stress_contract() -> None:
    evidence = StatefulDecodeEvidence(
        256, 1_000, 32, True, True, True, True, True, True, True, 1.01
    )
    assert evidence.flat_allocations
    with pytest.raises(ValueError, match="floors"):
        StatefulDecodeEvidence(
            255, 1_000, 32, True, True, True, True, True, True, True, 1.01
        )


def test_p7_evidence_requires_two_useful_outputs_per_pass() -> None:
    assert TargetAlignedK4Evidence(100, 201, True, True, True, 1.01).useful_outputs == 201
    with pytest.raises(ValueError, match="two useful"):
        TargetAlignedK4Evidence(100, 199, True, True, True, 1.01)


def test_p8_evidence_is_all_or_nothing() -> None:
    assert PrecisionOversizedEvidence(True, True, True, True, True, True).quality_passed
    with pytest.raises(ValueError, match="every"):
        PrecisionOversizedEvidence(True, False, True, True, True, True)
