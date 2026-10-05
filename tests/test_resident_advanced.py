from __future__ import annotations

import pytest
import torch

from mrun.compiler.graph_passes import FusionPlan, FusionRegion
from mrun.compiler.profile_fusion import FusionRegionProfile, select_profiled_fusion_regions
from mrun.runtime.resident_advanced import (
    PrecisionBackend,
    ResidentCapturedBatchFamily,
    ResidentCapturedBatchService,
    ResidentPrecisionFamily,
    TargetAlignedK4Runtime,
    TargetAlignedPredictorManifest,
    TransactionalCapturedDecodeLane,
)


class _Template:
    def __init__(self) -> None:
        self.generation = 0
        self.rows: tuple[tuple[int, ...], ...] = ()
        self.selected: tuple[int, ...] = ()
        self.request_id = ""

    def rebind(self, ids_list, token_ids, *, request_id):
        self.generation += 1
        self.rows = tuple(tuple(row) for row in ids_list)
        self.selected = tuple(token_ids)
        self.request_id = request_id
        return self.generation

    def execute(self, *, expected_generation, request_id):
        if expected_generation != self.generation or request_id != self.request_id:
            raise RuntimeError("stale")
        return torch.tensor(
            [[sum(row) + selected for selected in self.selected] for row in self.rows],
            dtype=torch.float32,
        )


def test_p4_captured_batch_family_preserves_row_identity() -> None:
    family = ResidentCapturedBatchFamily({(2, 3, 2): _Template()})
    result = family.execute(
        ((1, 2, 3), (4, 5, 6)),
        (10, 20),
        request_ids=("a", "b"),
        request_id="cohort-1",
    )
    assert result.request_ids == ("a", "b")
    assert result.scores == ((16.0, 26.0), (25.0, 35.0))
    assert family.telemetry() == {"physical_dispatches": 1, "logical_rows": 2}
    with pytest.raises(ValueError, match="not admitted"):
        family.execute(((1,),), (2,), request_ids=("a",), request_id="bad")


def test_p4_captured_batch_service_cohorts_cancels_and_falls_back() -> None:
    family = ResidentCapturedBatchFamily({(2, 3, 2): _Template()})
    with ResidentCapturedBatchService(
        family,
        eager_fallback=lambda row, selected: tuple(sum(row) + value for value in selected),
        max_queue_delay_seconds=0.01,
    ) as service:
        first = service.submit((1, 2, 3), (10, 20), request_id="a")
        cancelled = service.submit((9, 9, 9), (30, 40), request_id="cancel")
        assert cancelled.cancel()
        second = service.submit((4, 5, 6), (10, 20), request_id="b")
        assert first.result(timeout=1) == (16.0, 26.0)
        assert second.result(timeout=1) == (25.0, 35.0)
        fallback = service.submit((2, 2, 2), (1, 3), request_id="single")
        assert fallback.result(timeout=1) == (7, 9)
        telemetry = service.telemetry()
        assert telemetry["captured_rows"] == 2
        assert telemetry["captured_dispatches"] == 1
        assert telemetry["captured_mean_width"] == 2
        assert telemetry["fallback_rows"] == 1
        assert telemetry["cancelled_rows"] == 1
        assert telemetry["queue_p95_ms"] is not None


def test_p5_profile_driven_fusion_rejects_unprofitable_legal_regions() -> None:
    fast = FusionRegion("fast", ("a", "b"), ("x",), ("y",), ("w",))
    slow = FusionRegion("slow", ("c", "d"), ("y",), ("z",), ("v",))
    legal = FusionPlan((fast, slow), ("e",), (), 2, 2)
    selected = select_profiled_fusion_regions(
        legal,
        (
            FusionRegionProfile(
                "fast", (2.0, 2.1), (1.0, 1.1), 100, 50, True, 1, True
            ),
            FusionRegionProfile(
                "slow", (1.0, 1.0), (1.1, 1.2), 100, 80, True, 1, True
            ),
        ),
    )
    assert selected.selected_region_ids == ("fast",)
    assert set(selected.selected_plan.unfused_node_ids) == {"c", "d", "e"}


class _DecodeBackend:
    def __init__(self) -> None:
        self.allocation_count = 4
        self.generation = 0
        self.bound: dict[int, tuple[int, int, str]] = {}

    def initialize(self, *, slot, prefix, request_id):
        self.bound[slot] = (prefix[-1] if prefix else 0, len(prefix), request_id)

    def bind(self, *, slot, token_id, position, request_id):
        self.generation += 1
        self.bound[slot] = (token_id, position, request_id)
        return self.generation

    def replay(self, *, slot, generation, request_id):
        token, _position, owner = self.bound[slot]
        if generation != self.generation or owner != request_id:
            raise RuntimeError("stale")
        return token + 1

    def commit(self, *, slot, accepted_count):
        assert accepted_count == 1

    def rollback(self, *, slot):
        assert slot in self.bound

    def release(self, *, slot, request_id):
        assert self.bound[slot][2] == request_id
        del self.bound[slot]


def test_p6_transactional_captured_decode_is_flat_and_rollback_safe() -> None:
    lane = TransactionalCapturedDecodeLane(_DecodeBackend(), slots=2, capacity=8)
    lane.allocate("a", (1, 2))
    first = lane.step("a", 2)
    assert first.token_id == 3
    assert first.after.committed_length == 3
    rolled_back = lane.step("a", 3, commit=False)
    assert rolled_back.after.committed_length == 3
    for _ in range(4):
        lane.step("a", 3)
    assert lane.telemetry()["allocation_count"] == 2
    lane.release("a")


def test_p7_k4_applies_target_correction_and_reports_useful_outputs() -> None:
    runtime = TargetAlignedK4Runtime(
        lambda _prefix, _k: (10, 11, 12, 13),
        lambda _prefix, _proposals: (10, 11, 99, 100, 101),
    )
    step = runtime.step((1, 2))
    assert step.accepted_proposals == 2
    assert step.outputs == (10, 11, 99)
    assert runtime.telemetry()["clears_two_outputs_per_pass_gate"] is True


def test_p8_precision_family_is_exact_and_fail_closed() -> None:
    digest = "a" * 64
    fp8 = PrecisionBackend(
        "moe-fp8",
        "fp8",
        "qwen3-moe",
        "decode",
        "contract-a",
        "greedy-token",
        1_000,
        100,
        50,
        digest,
        lambda: 1,
        True,
    )
    int4 = PrecisionBackend(
        "moe-int4",
        "int4",
        "qwen3-moe",
        "decode",
        "contract-b",
        "greedy-token",
        500,
        50,
        25,
        digest,
        lambda: 2,
        False,
    )
    family = ResidentPrecisionFamily((fp8, int4))
    assert (
        family.select(
            architecture="qwen3-moe",
            precision="fp8",
            workload="decode",
            output_contract="greedy-token",
            free_bytes=100,
            numerical_contract="contract-a",
        )
        is fp8
    )
    with pytest.raises(PermissionError, match="not promoted"):
        family.select(
            architecture="qwen3-moe",
            precision="int4",
            workload="decode",
            output_contract="greedy-token",
            free_bytes=100,
            numerical_contract="contract-b",
        )
    with pytest.raises(MemoryError):
        family.select(
            architecture="qwen3-moe",
            precision="fp8",
            workload="decode",
            output_contract="greedy-token",
            free_bytes=99,
            numerical_contract="contract-a",
        )


def test_p7_target_aligned_predictor_manifest_is_content_bound() -> None:
    manifest = TargetAlignedPredictorManifest(
        "qwen-k4",
        "a" * 64,
        "b" * 64,
        "target-hidden-state-distillation-v1",
    )
    assert manifest.proposal_tokens == 4
    with pytest.raises(ValueError, match="SHA-256"):
        TargetAlignedPredictorManifest("bad", "x", "b" * 64, "training")
