from __future__ import annotations

import json

import pytest

from mrun.science.qwen3_moe_page_trace import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    TRACE_SCHEMA,
    PageAccessTrace,
    PageCacheBudget,
    PageTraceError,
    PageTraceEvent,
    bounded_real_run_request,
    load_page_access_trace,
    replay_page_policy_ab,
)


def _scan_then_decode_trace() -> PageAccessTrace:
    # Two quota-local hot pages are hidden in one-pass prefill scans. Global LRU fills with
    # both pages from each layer and flushes the first layer. Transient-frequency admits only
    # each layer's hottest page, preserving exactly the later decode working set.
    return PageAccessTrace(
        layer_count=2,
        page_bytes=100,
        source={"run_id": "hand-checkable", "model": "toy-qwen3-moe"},
        events=(
            PageTraceEvent(phase="prefill", step=0, layer=0, pages=((0, 3), (1, 1))),
            PageTraceEvent(phase="prefill", step=0, layer=1, pages=((0, 3), (1, 1))),
            PageTraceEvent(phase="decode", step=0, layer=0, pages=((0, 1),)),
            PageTraceEvent(phase="decode", step=0, layer=1, pages=((0, 1),)),
            PageTraceEvent(phase="decode", step=1, layer=0, pages=((0, 1),)),
            PageTraceEvent(phase="decode", step=1, layer=1, pages=((0, 1),)),
        ),
    )


def test_hand_trace_reports_exact_policy_delta_under_same_budget() -> None:
    report = replay_page_policy_ab(
        _scan_then_decode_trace(),
        PageCacheBudget(page_bytes=100, capacity_pages=2),
    )

    baseline = report["arms"]["baseline"]
    candidate = report["arms"]["candidate"]
    assert baseline["name"] == BASELINE_ARM
    assert candidate["name"] == CANDIDATE_ARM
    assert baseline["page_requests"] == candidate["page_requests"] == 8
    assert baseline["page_hits"] == 2
    assert baseline["page_misses"] == 6
    assert baseline["evictions"] == 4
    assert baseline["h2d_bytes"] == 600
    assert candidate["page_hits"] == 4
    assert candidate["page_misses"] == 4
    assert candidate["evictions"] == 0
    assert candidate["h2d_bytes"] == 400
    assert candidate["transient_prefill_admitted_pages"] == 2
    assert report["comparison"] == {
        "candidate_minus_baseline_hits": 2,
        "candidate_minus_baseline_misses": -2,
        "candidate_minus_baseline_evictions": -4,
        "signed_h2d_bytes_saved": 200,
        "upper_bound_h2d_bytes_saved": 200,
        "upper_bound_h2d_gb_saved": 2e-07,
        "upper_bound_transfer_reduction_fraction": pytest.approx(1 / 3),
        "transfer_only_speedup_upper_bound": 1.5,
        "candidate_has_lower_h2d": True,
    }


def test_layer_frequency_order_preserves_hot_page_across_layer_spill() -> None:
    trace = PageAccessTrace(
        layer_count=2,
        page_bytes=16,
        events=(
            PageTraceEvent(phase="decode", step=0, layer=0, pages=((0, 3), (1, 1), (2, 1))),
            PageTraceEvent(phase="decode", step=0, layer=1, pages=((0, 1),)),
            PageTraceEvent(phase="decode", step=1, layer=0, pages=((0, 1),)),
        ),
    )
    report = replay_page_policy_ab(trace, PageCacheBudget(page_bytes=16, capacity_pages=3))

    assert report["arms"]["baseline"]["page_hits"] == 0
    assert report["arms"]["candidate"]["page_hits"] == 1
    assert report["arms"]["candidate"]["over_quota_evictions"] == 1


def test_trace_round_trip_fingerprint_and_scalar_only_custody(tmp_path) -> None:
    trace = _scan_then_decode_trace()
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace.as_dict()))

    loaded = load_page_access_trace(path)
    assert loaded == trace
    assert loaded.trace_sha256 == trace.trace_sha256
    tampered = trace.as_dict()
    tampered["events"][0]["pages"][0][1] = 99
    with pytest.raises(PageTraceError, match="trace_sha256"):
        PageAccessTrace.from_dict(tampered)
    with pytest.raises(PageTraceError, match="tensor/list payloads"):
        PageAccessTrace(
            layer_count=1,
            page_bytes=1,
            events=(PageTraceEvent("decode", 0, 0, ((0, 1),)),),
            source={"raw_tensor": [1, 2, 3]},  # type: ignore[dict-item]
        )


def test_budget_floors_bytes_and_refuses_an_oversized_active_set() -> None:
    budget = PageCacheBudget.from_bytes(page_bytes=100, capacity_bytes=250)
    assert budget.as_dict() == {
        "page_bytes": 100,
        "capacity_pages": 2,
        "allocated_bytes": 200,
        "requested_bytes": 250,
        "unallocated_bytes": 50,
    }
    trace = PageAccessTrace(
        layer_count=1,
        page_bytes=100,
        events=(PageTraceEvent("decode", 0, 0, ((0, 1), (1, 1), (2, 1))),),
    )
    with pytest.raises(PageTraceError, match="requests 3 pages"):
        replay_page_policy_ab(trace, budget)


def test_bounded_request_is_non_submitting_parity_guarded_and_artifact_light() -> None:
    report = replay_page_policy_ab(
        _scan_then_decode_trace(),
        PageCacheBudget(page_bytes=100, capacity_pages=2),
    )
    request = bounded_real_run_request(report)

    assert request["schema"].endswith("request-v1")
    assert request["submission_authorized"] is False
    assert request["scheduler"]["strict_preflight"] is True
    assert request["scheduler"]["needs"] == {"cuda": True}
    assert request["physical_geometry"]["model_loads"] == 1
    assert request["physical_geometry"]["maximum_physical_forwards"] == 4
    assert request["stop_conditions"]["total_wall_seconds"] == 840
    assert request["required_parity"]["route_page_sequence_sha256_equal"] is True
    assert "weights" in request["retention"]["drop"]


def test_schema_and_page_stride_mismatch_are_refused() -> None:
    payload = _scan_then_decode_trace().as_dict()
    payload["schema"] = "unknown"
    with pytest.raises(PageTraceError, match="unsupported"):
        PageAccessTrace.from_dict(payload)
    with pytest.raises(PageTraceError, match="differs"):
        replay_page_policy_ab(
            _scan_then_decode_trace(),
            PageCacheBudget(page_bytes=99, capacity_pages=2),
        )
    assert TRACE_SCHEMA == "mrun-qwen3-moe-page-trace-v1"


def test_loader_refuses_an_unbounded_artifact_before_reading(tmp_path, monkeypatch) -> None:
    path = tmp_path / "too-large.json"
    path.write_text("{}")
    real_stat = path.stat()

    class OversizedStat:
        st_size = 32 * 1024 * 1024 + 1

    monkeypatch.setattr(type(path), "stat", lambda _self: OversizedStat())
    with pytest.raises(PageTraceError, match="bounded loader limit"):
        load_page_access_trace(path)
    assert real_stat.st_size == 2
