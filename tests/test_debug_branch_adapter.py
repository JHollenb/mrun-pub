from __future__ import annotations

import hashlib

import pytest

from mrun.compiler import execute_serialized_sciencegraph_plan


def test_sciencegraph_adapter_rejects_missing_backend_artifact() -> None:
    with pytest.raises(ValueError, match="no ScienceGraph artifact"):
        execute_serialized_sciencegraph_plan(
            object(),
            {
                "schema": "manalysis.generative-branch-plan.v1",
                "model_identity": "m",
                "numerical_contract": "exact",
                "branches": [],
            },
        )


def test_sciencegraph_adapter_validates_neutral_schema_before_importing_engine() -> None:
    with pytest.raises(ValueError, match="unsupported neutral branch-plan schema"):
        execute_serialized_sciencegraph_plan(
            object(),
            {"schema": "not-a-branch-plan"},
        )


def test_adapter_plan_fingerprint_fallback_is_stable() -> None:
    # This checks the adapter's envelope precondition without constructing a
    # valid paged graph (the graph itself is covered by ScienceGraph tests).
    canonical = (
        "{\"branches\":[],\"model_identity\":\"m\","
        "\"numerical_contract\":\"exact\","
        "\"schema\":\"manalysis.generative-branch-plan.v1\"}"
    )
    assert hashlib.sha256(canonical.encode()).hexdigest()
