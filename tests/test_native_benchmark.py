from __future__ import annotations

from dataclasses import replace

import pytest

from mrun.runtime.benchmark import (
    NativeBenchmarkCampaign,
    NativeBenchmarkKey,
    NativeBenchmarkSample,
    NativeRoofline,
    WorkKind,
    compare_campaigns,
)


def _key(**changes) -> NativeBenchmarkKey:
    values = {
        "model_fingerprint": "a" * 64,
        "artifact_fingerprint": "b" * 64,
        "lowering_fingerprint": "c" * 64,
        "placement_fingerprint": "d" * 64,
        "device_fingerprint": "e" * 64,
        "backend_id": "mlx-native",
        "numerical_contract": "bf16-source-bounded-v1",
        "codec_id": "bf16",
        "state_abi": "transformer-kv-v1",
        "output_contract": "next-token-argmax",
        "work_kind": WorkKind.AUTONOMOUS_DECODE,
        "batch_size": 1,
        "prompt_tokens_per_row": 128,
        "context_tokens_before_work": 128,
        "requested_output_tokens_per_row": 32,
        "residency": "fully-resident",
        "arrival_shape": "isolated-singleton",
        "session_shape": "fresh",
    }
    values.update(changes)
    return NativeBenchmarkKey(**values)


def _sample(
    index: int,
    elapsed: float,
    *,
    key: NativeBenchmarkKey | None = None,
    page_loads: int = 0,
) -> NativeBenchmarkSample:
    selected = key or _key()
    return NativeBenchmarkSample(
        key=selected,
        acquisition_index=index,
        launch_order="baseline-first" if index % 2 == 0 else "candidate-first",
        elapsed_seconds=elapsed,
        work_units=32,
        committed_output_tokens=(32 if selected.work_kind is WorkKind.AUTONOMOUS_DECODE else 0),
        input_positions=32 if selected.work_kind is WorkKind.PREFILL else 0,
        physical_weight_bytes_read=1000,
        kv_bytes_read=200,
        kv_bytes_written=20,
        host_to_device_bytes=0,
        device_to_host_bytes=8,
        peak_resident_bytes=4096,
        average_power_watts=20.0,
        unexpected_page_loads=page_loads,
    )


def test_key_fingerprint_separates_quantity_batch_context_and_numerical_contract() -> None:
    baseline = _key()
    assert baseline.fingerprint == _key().fingerprint
    assert baseline.fingerprint != _key(batch_size=8).fingerprint
    assert baseline.fingerprint != _key(context_tokens_before_work=8192).fingerprint
    assert baseline.fingerprint != _key(work_kind=WorkKind.TEACHER_FORCED_DECODE).fingerprint
    assert baseline.fingerprint != _key(numerical_contract="q4-quality-v1").fingerprint


def test_sample_enforces_autonomous_teacher_forced_and_prefill_units() -> None:
    autonomous = _sample(0, 2.0)
    assert autonomous.units_per_second == 16.0
    assert autonomous.joules_per_unit == 1.25
    assert autonomous.as_dict()["work_unit_kind"] == ("autonomous-committed-decode-tokens")

    with pytest.raises(ValueError, match="committed output"):
        replace(autonomous, committed_output_tokens=31)

    teacher_key = _key(work_kind=WorkKind.TEACHER_FORCED_DECODE)
    teacher = _sample(0, 1.0, key=teacher_key)
    assert teacher.committed_output_tokens == 0
    with pytest.raises(ValueError, match="teacher-forced"):
        replace(teacher, committed_output_tokens=32)

    prefill_key = _key(
        work_kind=WorkKind.PREFILL,
        requested_output_tokens_per_row=0,
        prompt_tokens_per_row=32,
    )
    prefill = _sample(0, 0.5, key=prefill_key)
    assert prefill.input_positions == prefill.work_units == 32
    with pytest.raises(ValueError, match="input positions"):
        replace(prefill, input_positions=31)


def test_campaign_requires_one_exact_key_and_ordered_acquisitions() -> None:
    campaign = NativeBenchmarkCampaign((_sample(0, 2.0), _sample(1, 1.0)))
    assert campaign.median_units_per_second == 24.0
    assert campaign.p95_units_per_second == pytest.approx(31.2)
    assert campaign.all_comparable

    with pytest.raises(ValueError, match="contiguous acquisition"):
        NativeBenchmarkCampaign((_sample(1, 1.0), _sample(0, 2.0)))
    with pytest.raises(ValueError, match="exact benchmark key"):
        NativeBenchmarkCampaign((_sample(0, 1.0), _sample(1, 1.0, key=_key(batch_size=2))))


def test_equal_work_comparison_allows_backend_change_but_not_batch_or_quantity() -> None:
    baseline = NativeBenchmarkCampaign((_sample(0, 2.0), _sample(1, 2.0)))
    candidate_key = _key(
        artifact_fingerprint="f" * 64,
        lowering_fingerprint="1" * 64,
        placement_fingerprint="2" * 64,
        backend_id="mlx-native-paged",
    )
    candidate = NativeBenchmarkCampaign(
        (_sample(0, 1.0, key=candidate_key), _sample(1, 1.0, key=candidate_key))
    )
    comparison = compare_campaigns(baseline, candidate)
    assert comparison["candidate_over_baseline_median_ratio"] == 2.0
    assert comparison["both_comparable"]

    different_batch = NativeBenchmarkCampaign(
        (
            _sample(0, 1.0, key=_key(batch_size=2)),
            _sample(1, 1.0, key=_key(batch_size=2)),
        )
    )
    with pytest.raises(ValueError, match="equal work"):
        compare_campaigns(baseline, different_batch)


def test_page_load_or_fallback_marks_sample_noncomparable() -> None:
    campaign = NativeBenchmarkCampaign((_sample(0, 1.0), _sample(1, 1.0, page_loads=1)))
    assert not campaign.all_comparable
    assert not campaign.samples[1].comparable


def test_roofline_reports_overlap_and_serial_bounds_without_claiming_speed() -> None:
    roofline = NativeRoofline(
        weight_bytes=1_000,
        kv_read_bytes=200,
        kv_write_bytes=100,
        transfer_bytes=50,
        floating_point_operations=2_000,
        memory_bandwidth_bytes_per_second=1_000,
        transfer_bandwidth_bytes_per_second=100,
        compute_operations_per_second=4_000,
        fixed_seconds_per_unit=0.1,
    )
    assert roofline.memory_seconds == 1.3
    assert roofline.transfer_seconds == 0.5
    assert roofline.compute_seconds == 0.5
    assert roofline.ideal_overlap_seconds == pytest.approx(1.4)
    assert roofline.serial_seconds == pytest.approx(2.4)
    assert roofline.ideal_overlap_ceiling_units_per_second == pytest.approx(1 / 1.4)
    assert roofline.serial_ceiling_units_per_second == pytest.approx(1 / 2.4)
    assert roofline.utilization(0.5) == pytest.approx(0.7)
