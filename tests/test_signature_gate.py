"""mrun.testing.signature — planning ladder + the backend capability gate (pure, no models)."""

from __future__ import annotations

import pytest

from mrun.policy import HostCaps
from mrun.testing.signature import (
    SIGNATURE_HEADROOM_MB,
    SIGNATURE_MAX_THREADS,
    compare_signatures,
    plan_signature,
    reservation_for,
)

BEAST = HostCaps(name="beast", ram_mb=66508.0, vram_mb=16000.0, has_cuda=True, cpus=32)
TINY = HostCaps(name="tiny", ram_mb=8_000.0, cpus=4)


# --------------------------------------------------------------------------- planning
# NOTE: these tests are LADDER-AGNOSTIC on purpose. The rung order in
# _DTYPE_DEVICE_LADDER is an evolving policy owned by the signature arc (it moved
# fp32-cpu-first -> bf16-cuda-first on 2026-07-17); hardcoding a rung here made the
# tests break on a legitimate policy change. Test the mechanics, not the policy.
def test_small_model_admits_first_rung():
    from mrun.testing.signature import _DTYPE_DEVICE_LADDER
    p = plan_signature("qwen2.5-0.5b", host=BEAST)
    canon = {"fp32": "float32", "bf16": "bfloat16"}
    first_dt, first_dev = _DTYPE_DEVICE_LADDER[0]
    # a 0.5B fits every rung, so the FIRST rung must win
    assert p.dtype == canon[first_dt] and p.device == first_dev and p.backend == "hf"
    assert p.threads == SIGNATURE_MAX_THREADS  # un-throttled from mrun's default 4


def test_big_model_falls_back_down_ladder():
    # 14B fp32 (~56GB + headroom) fails the 0.70 comfort budget on a 66GB box, and its
    # bf16 weights (~28GB) exceed the 16GB card, so it must land on a cpu bf16 rung.
    p = plan_signature("qwen3-14b", host=BEAST)
    assert p.dtype == "bfloat16" and p.device == "cpu"


def test_headroom_applied():
    from mrun.policy import plan_run
    p = plan_signature("qwen2.5-0.5b", host=BEAST)
    rev = {"float32": "fp32", "bfloat16": "bf16"}
    base = plan_run("qwen2.5-0.5b", task="forward", host=BEAST, backend="hf",
                    dtype=rev[p.dtype], device=p.device)
    assert p.ram_limit_mb == pytest.approx(base.ram_limit_mb + SIGNATURE_HEADROOM_MB, abs=1.0)


def test_nothing_fits_raises():
    with pytest.raises(MemoryError):
        plan_signature("qwen3-14b", host=TINY)


def test_reservation_shape():
    resv, plan = reservation_for("qwen2.5-0.5b", host=BEAST)
    assert resv["source"] == "plan_signature"
    assert resv["ram_mb"] == pytest.approx(plan.ram_limit_mb)
    assert resv["cpu_threads"] == plan.threads




# --------------------------------------------------------------------------- gate fixtures
def _payload(*, present=True, dep=-1.5, spec=0.8, depth=4, behavior=0.9,
             null_acc=0.2, measurable=True, capability="coding"):
    leg_meas = {"measurable": True}
    return {
        "capability": capability,
        "model_name": "m",
        "config": {"dtype": "float32"},
        "primitives": {
            "long_range_variable_binding": {
                "present": ({**leg_meas, "present": present,
                             "scrambled_null": {"n": 16, "accuracy": null_acc}}
                            if measurable else {"measurable": False, "reason": "x"}),
                "causal_use": {**leg_meas, "dependence": dep, "specificity": spec},
                "depth": {**leg_meas, "max_distance_surviving": depth},
            },
        },
        "coverage": {"behavior_accuracy": behavior,
                     "behavior_components": {"binding_acc_d1": behavior}},
    }


def test_gate_passes_identical():
    v = compare_signatures(_payload(), _payload())
    assert v["pass"], v["failures"]
    assert not v["failures"]


def test_gate_fails_present_flip():
    v = compare_signatures(_payload(), _payload(present=False))
    assert not v["pass"]
    assert any("present: True -> False" in f for f in v["failures"])


def test_gate_fails_dependence_lost():
    v = compare_signatures(_payload(dep=-1.5), _payload(dep=0.1))
    assert any("need lost" in f for f in v["failures"])


def test_gate_fails_specificity_lost():
    v = compare_signatures(_payload(spec=0.8), _payload(spec=-0.2))
    assert any("localization lost" in f for f in v["failures"])


def test_gate_allows_noise_near_zero():
    # ref spec 0.05 (below eps): candidate flipping sign is NOT a failure
    v = compare_signatures(_payload(spec=0.05), _payload(spec=-0.03))
    assert v["pass"], v["failures"]


def test_gate_fails_depth_shrink():
    v = compare_signatures(_payload(depth=4), _payload(depth=2))
    assert any("depth shrank" in f for f in v["failures"])


def test_gate_fails_behavior_drop():
    v = compare_signatures(_payload(behavior=0.9), _payload(behavior=0.7))
    assert any("behavior_accuracy" in f for f in v["failures"])


def test_gate_fails_null_resurrection():
    v = compare_signatures(_payload(null_acc=0.2), _payload(null_acc=0.6))
    assert any("null resurrected" in f for f in v["failures"])


def test_gate_warns_reference_null_elevated():
    v = compare_signatures(_payload(null_acc=0.6), _payload(null_acc=0.6))
    assert any("REFERENCE null" in w for w in v["warnings"])


def test_gate_fails_measurability_regression():
    v = compare_signatures(_payload(measurable=True), _payload(measurable=False))
    assert any("lost in candidate" in f for f in v["failures"])


def test_gate_warns_capability_gained():
    v = compare_signatures(_payload(present=False), _payload(present=True))
    assert v["pass"]  # gains never fail
    assert any("gained" in w for w in v["warnings"])


def test_gate_skips_unmeasurable_in_both():
    v = compare_signatures(_payload(measurable=False), _payload(measurable=False))
    assert v["pass"], v["failures"]


def test_gate_capability_mismatch_raises():
    with pytest.raises(ValueError):
        compare_signatures(_payload(capability="coding"),
                           _payload(capability="systematic_reasoning"))
