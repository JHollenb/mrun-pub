from __future__ import annotations

import pytest

from mrun.compiler import (
    CohortRequest,
    ResidencyNode,
    ShapeBucketPolicy,
    form_amortization_groups,
    schedule_for_residency,
)


def test_cohorts_preserve_contracts_deadlines_and_separate_physical_savings() -> None:
    policy = ShapeBucketPolicy(coalescing_window_ns=10)
    requests = tuple(
        CohortRequest(
            request_id=f"r{index}",
            arena_fingerprint="arena-a" if index < 3 else "arena-b",
            numerical_contract="exact",
            output_contract="selected",
            sequence_length=5 + index % 2,
            selected_row_count=2,
            deadline_ns=100 + index,
        )
        for index in range(5)
    )
    plan = form_amortization_groups(requests, policy)

    assert plan.logical_requests == 5
    assert plan.physical_cohorts == 2
    assert plan.groups[0].request_ids == ("r0", "r1", "r2")
    assert plan.groups[0].bucket == (4, 16, 2)
    assert plan.as_dict()["semantic_work_deleted"] == 0
    assert plan.as_dict()["weight_traversals_amortized"] == 3


def test_cohort_refuses_duplicate_ids_and_oversized_shapes() -> None:
    request = CohortRequest("same", "arena", "exact", "selected", 1, 2, 1)
    with pytest.raises(ValueError, match="globally unique"):
        form_amortization_groups((request, request))
    oversized = CohortRequest("large", "arena", "exact", "selected", 129, 2, 1)
    with pytest.raises(OverflowError, match="sequence"):
        form_amortization_groups((oversized,))


def test_residency_scheduler_preserves_causality_and_prefers_hot_resources() -> None:
    nodes = (
        ResidencyNode("a", (), ("page-1",)),
        ResidencyNode("b", (), ("page-2",)),
        ResidencyNode("c", ("a",), ("page-1",)),
        ResidencyNode("d", ("b",), ("page-2",)),
    )
    schedule = schedule_for_residency(nodes)
    assert schedule.node_order == ("a", "c", "b", "d")
    assert schedule.resource_transitions == 2
    assert schedule.residency_reuses == 2


def test_residency_scheduler_refuses_cycles() -> None:
    with pytest.raises(ValueError, match="cycle"):
        schedule_for_residency(
            (
                ResidencyNode("a", ("b",), ("page",)),
                ResidencyNode("b", ("a",), ("page",)),
            )
        )
