"""Reservation sizing module — ladder cases live here so clamp/raise changes diff cleanly."""

from mrun.protocol import kill_ceiling_mb
from mrun.server.reservation import (
    SizedReservation,
    effective_reservation,
    history_admission_reservation,
    resolve_reservation,
    size_reservation,
)
from mrun.server.scheduler import fits_some_host_capacity


def test_ladder_declared_wins():
    r = resolve_reservation({"ram_mb": 1234}, {"ram_peak_mb": 500}, {"ram_mb": 999})
    assert (r.ram_mb, r.source) == (1234, "declared")


def test_ladder_history_beats_estimate_with_safety_factor():
    # n<3: max peak x 1.3 (1s-sampled peaks read low), 512-rounded
    r = resolve_reservation(None, {"ram_peak_mb": 2000, "wall_s": 60}, {"ram_mb": 999})
    assert r.ram_mb == 3072  # round_up_512(2000 * 1.3)
    assert r.source == "history"
    # n>=3: stable p95 basis x 1.25
    r = resolve_reservation(None, {"ram_peak_mb": 2000, "wall_s": 60, "n": 5}, None)
    assert r.ram_mb == 2560  # round_up_512(2000 * 1.25)


def test_ladder_estimate_floors_history():
    r = resolve_reservation(None, {"ram_peak_mb": 500, "wall_s": 60}, {"ram_mb": 999})
    assert r.ram_mb == 1024  # estimate floors history, 512-rounded


def test_ladder_generalized_history():
    r = resolve_reservation(
        None, None, None, generalized={"p95_ram_mb": 4000, "n": 5, "p50_wall_s": 30}
    )
    assert r.ram_mb == 6144  # round_up_512(4000 * 1.5)
    assert r.source == "generalized-history"


def test_ladder_estimate_then_default():
    r = resolve_reservation(None, None, {"ram_mb": 999})
    assert (r.ram_mb, r.source) == (999, "estimated")
    r = resolve_reservation(None, None, None, needs_cuda=True)
    assert (r.ram_mb, r.vram_mb, r.source) == (4096, 4096, "probe-default")


def test_size_reservation_wraps_ladder():
    sized = size_reservation(
        {"ram_mb": 1234},
        exact_history={"ram_peak_mb": 500},
        client_estimate={"ram_mb": 999},
    )
    assert isinstance(sized, SizedReservation)
    assert sized.reservation.ram_mb == 1234
    assert sized.reservation.source == "declared"
    assert sized.warnings == []
    assert sized.clamps == []


def test_effective_reservation_declared_beats_plan():
    job = {
        "reservation": {"ram_mb": 8192, "source": "declared"},
        "plans": {"h1": {"ram_limit_mb": 2000, "est_vram_mb": 0, "threads": 2}},
    }
    assert effective_reservation(job, "h1").ram_mb == 8192


def test_effective_reservation_plan_overrides_default():
    job = {
        "reservation": {"ram_mb": 8192, "source": "default"},
        "plans": {"h1": {"ram_limit_mb": 2000, "est_vram_mb": 1000, "threads": 2}},
    }
    res = effective_reservation(job, "h1")
    assert (res.ram_mb, res.vram_mb, res.cpu_threads, res.source) == (2000, 1000, 2, "plan")


def test_history_admission_right_sizes_overdeclared():
    job = {"reservation": {"ram_mb": 40000, "vram_mb": 0, "source": "declared"}}
    out = history_admission_reservation(
        job, {"n": 5, "p95_ram_mb": 3000}, {"tree_rss_mb": 2500}
    )
    assert out is not None
    assert out["ram_mb"] == 6144  # max(3000*2, 2500*2) rounded up to 512
    assert out["source"] == "history-admission"


def test_history_admission_requires_declared_and_n3():
    job = {"reservation": {"ram_mb": 40000, "source": "default"}}
    assert history_admission_reservation(job, {"n": 5, "p95_ram_mb": 3000}) is None
    job = {"reservation": {"ram_mb": 40000, "source": "declared"}}
    assert history_admission_reservation(job, {"n": 2, "p95_ram_mb": 3000}) is None


# ------------------------------------------------------------------ phase 1: clamps


def test_kill_ceiling_absolute_grace_for_small_reservations():
    assert kill_ceiling_mb(250) == 762  # 250 + 512, not 275
    assert kill_ceiling_mb(20000) == 22000.0  # x1.1 dominates
    assert kill_ceiling_mb(0) == 0.0


def test_declared_overask_clamped_to_family_p95():
    # the 48GB-ask / 3.4GB-RSS pattern: family p95 3328MB, n>=3. Now clamped to the
    # measured-history grant (p95 x FAMILY_P95_FACTOR=1.5), 512-rounded, floored by
    # observed peak x 1.1 (== p95 here since no max_ram_mb given).
    sized = size_reservation(
        {"ram_mb": 48000},
        family_history={"n": 5, "p95_ram_mb": 3328},
    )
    assert sized.reservation.source == "declared-clamped"
    assert sized.reservation.ram_mb == 5120  # round_up_512(3328 * 1.5)
    assert sized.clamps and sized.clamps[0]["field"] == "ram_mb"
    assert sized.warnings


def test_declared_within_history_ceiling_not_clamped():
    # a declared ask already <= the measured-history grant is left alone: 4000 < the
    # 5120 ceiling from family p95 3328 x1.5, so no clamp (the old 3x tripwire is gone).
    sized = size_reservation(
        {"ram_mb": 4000},
        family_history={"n": 5, "p95_ram_mb": 3328},
    )
    assert sized.reservation.source == "declared"
    assert sized.reservation.ram_mb == 4000
    assert not sized.clamps


def test_declared_clamp_needs_family_n3():
    sized = size_reservation(
        {"ram_mb": 48000},
        family_history={"n": 2, "p95_ram_mb": 3328},
    )
    assert sized.reservation.source == "declared"
    assert sized.reservation.ram_mb == 48000


def test_declared_underask_raised_to_measured_peak():
    # 8GB ask vs 8.9GB measured peak: old ceiling 8800 < peak — the agent would
    # kill a correct run. Raised to peak x 1.2.
    sized = size_reservation(
        {"ram_mb": 8000},
        exact_history={"ram_peak_mb": 8970},
    )
    assert sized.reservation.source == "declared-raised"
    assert sized.reservation.ram_mb == 11264  # ceil(8970*1.2 / 512) * 512
    assert sized.clamps


def test_declared_underask_within_ceiling_not_raised():
    sized = size_reservation(
        {"ram_mb": 8000},
        exact_history={"ram_peak_mb": 8000},
    )
    assert sized.reservation.source == "declared"
    assert sized.reservation.ram_mb == 8000


def test_declared_vram_underask_raised_to_measured_peak():
    sized = size_reservation(
        {"ram_mb": 3000, "vram_mb": 1349},
        exact_history={"ram_peak_mb": 800, "vram_peak_mb": 4092},
    )
    assert sized.reservation.source == "declared-raised"
    assert sized.reservation.ram_mb == 3000
    assert sized.reservation.vram_mb == 5120
    assert sized.clamps == [
        {
            "field": "vram_mb",
            "from": 1349.0,
            "to": 5120.0,
            "why": (
                "declared VRAM kill ceiling 1861MB < measured peak 4092MB; "
                "raised to peak x 1.2"
            ),
        }
    ]


def test_declared_vram_overask_clamped():
    sized = size_reservation(
        {"ram_mb": 4000, "vram_mb": 1500},
        family_history={"n": 5, "p95_ram_mb": 3500, "p95_vram_mb": 400},
    )
    assert sized.reservation.vram_mb == 1024  # max(round_up(400*2)=1024, 512)
    assert sized.reservation.source == "declared-clamped"


def test_effective_reservation_clamped_may_shrink_via_plan_never_grow():
    job = {
        "reservation": {"ram_mb": 6656, "source": "declared-clamped"},
        "plans": {"h1": {"ram_limit_mb": 3000, "est_vram_mb": 0, "threads": 2}},
    }
    assert effective_reservation(job, "h1").ram_mb == 3000
    job["plans"]["h1"]["ram_limit_mb"] = 20000
    assert effective_reservation(job, "h1").ram_mb == 6656


# --------------------------------------------------- phase 4: history clamp (declared)


def test_history_clamp_exact_preferred_over_family():
    # the 1.5-3x over-ask the old 3x tripwire let through: 12000 declared, exact-config
    # history says p95 4000 -> clamp to p95 x EXACT_P95_FACTOR (1.25). Exact beats family.
    sized = size_reservation(
        {"ram_mb": 12000},
        exact_history={"n": 5, "ram_peak_mb": 4000, "max_ram_mb": 4200},
        family_history={"n": 9, "p95_ram_mb": 8000, "max_ram_mb": 9000},
    )
    assert sized.reservation.source == "declared-clamped"
    assert sized.reservation.ram_mb == 5120  # round_up_512(4000 * 1.25)
    assert sized.clamps and sized.clamps[0]["field"] == "ram_mb"
    assert sized.warnings


def test_history_clamp_family_when_no_exact():
    # no exact history -> clamp to family p95 x FAMILY_P95_FACTOR (1.5)
    sized = size_reservation(
        {"ram_mb": 12000},
        family_history={"n": 5, "p95_ram_mb": 6000, "max_ram_mb": 6500},
    )
    assert sized.reservation.source == "declared-clamped"
    assert sized.reservation.ram_mb == 9216  # round_up_512(6000 * 1.5)


def test_history_clamp_no_history_no_clamp():
    # n<3 family and no exact -> nothing trustworthy to clamp against
    sized = size_reservation(
        {"ram_mb": 12000},
        family_history={"n": 2, "p95_ram_mb": 3000, "max_ram_mb": 3000},
    )
    assert sized.reservation.source == "declared"
    assert sized.reservation.ram_mb == 12000
    assert not sized.clamps
    # and truly no history at all
    sized = size_reservation({"ram_mb": 12000})
    assert sized.reservation.source == "declared"
    assert sized.reservation.ram_mb == 12000


def test_history_clamp_never_below_observed_peak():
    # skewed family: p95 3000 but a 9000 peak was observed. p95 x1.5 = 4500 would clamp
    # below the worst run's kill line, so the floor (max 9000 x 1.1) wins instead.
    sized = size_reservation(
        {"ram_mb": 12000},
        family_history={"n": 10, "p95_ram_mb": 3000, "max_ram_mb": 9000},
    )
    assert sized.reservation.source == "declared-clamped"
    assert sized.reservation.ram_mb == 10240  # round_up_512(9000 * 1.1), not 4608
    assert sized.reservation.ram_mb >= 9000 * 1.1


def test_history_clamp_opt_out_pinned():
    # source 'declared-pinned' bypasses the clamp entirely
    sized = size_reservation(
        {"ram_mb": 40000, "source": "declared-pinned"},
        family_history={"n": 5, "p95_ram_mb": 3000, "max_ram_mb": 3200},
    )
    assert sized.reservation.source == "declared-pinned"
    assert sized.reservation.ram_mb == 40000
    assert not sized.clamps
    # a `pin` flag on the declared dict works too
    sized = size_reservation(
        {"ram_mb": 40000, "pin": True},
        family_history={"n": 5, "p95_ram_mb": 3000, "max_ram_mb": 3200},
    )
    assert sized.reservation.source == "declared-pinned"
    assert sized.reservation.ram_mb == 40000


def test_history_clamp_pin_warns_when_below_measured_peak():
    # pin bypasses the raise too, but a pin under a measured peak is flagged
    sized = size_reservation(
        {"ram_mb": 2000, "source": "declared-pinned"},
        exact_history={"n": 5, "ram_peak_mb": 5000, "max_ram_mb": 5000},
    )
    assert sized.reservation.source == "declared-pinned"
    assert sized.reservation.ram_mb == 2000
    assert sized.warnings and "may be killed" in sized.warnings[0]


def test_vram_keeps_three_x_tripwire():
    # VRAM is only clamped above 3x family p95 (to 2x); RAM has no history here.
    within = size_reservation(
        {"ram_mb": 3000, "vram_mb": 12000},
        family_history={"n": 5, "p95_vram_mb": 6000, "max_vram_mb": 6500},
    )
    assert within.reservation.vram_mb == 12000
    assert within.reservation.ram_mb == 3000
    over = size_reservation(
        {"ram_mb": 3000, "vram_mb": 20000},
        family_history={"n": 5, "p95_vram_mb": 6000, "max_vram_mb": 6500},
    )
    assert over.reservation.source == "declared-clamped"
    assert over.reservation.vram_mb == 12288  # round_up_512(6000 * 2)


# ------------------------------------------------------------- phase 1: preflight


_HOSTS = [
    {
        "name": "beast",
        "ram_total_mb": 66509,
        "vram_total_mb": 16376,
        "cpu_threads": 32,
        "caps": {"cuda": True},
    },
    {
        "name": "mbp1",
        "ram_total_mb": 19327,
        "vram_total_mb": 0,
        "cpu_threads": 11,
        "caps": {"mps": True},
    },
]


def test_preflight_rejects_impossible_vram():
    # 18GB VRAM on a 16GB card (2026-07-31): never fits, must be told at submit
    job = {
        "needs": {"cuda": True},
        "reservation": {"ram_mb": 8000, "vram_mb": 18000, "source": "declared"},
    }
    ok, reasons = fits_some_host_capacity(job, _HOSTS)
    assert not ok
    assert any("vram" in r for r in reasons)


def test_preflight_rejects_job_vram_ceiling_over_contract():
    job = {
        "needs": {"cuda": True},
        "config": {"max_vram_ceiling_mb": 15_500},
        "reservation": {"ram_mb": 8000, "vram_mb": 14_100, "source": "declared"},
    }
    ok, reasons = fits_some_host_capacity(job, _HOSTS)
    assert not ok
    assert any("configured max 15500MB" in reason for reason in reasons)


def test_preflight_rejects_ram_beyond_any_host():
    job = {"needs": {}, "reservation": {"ram_mb": 64000, "source": "declared"}}
    ok, reasons = fits_some_host_capacity(job, _HOSTS)
    assert not ok
    assert len(reasons) == 2  # one per candidate


def test_preflight_accepts_fitting_job_ignoring_telemetry():
    # no telemetry on the host rows at all — static totals only
    job = {"needs": {}, "reservation": {"ram_mb": 44000, "source": "declared"}}
    ok, reasons = fits_some_host_capacity(job, _HOSTS)
    assert ok and reasons == []


def test_preflight_uses_host_plan_over_stale_generalized_reservation():
    # A family-history prior can be larger than the selected host-specific plan. The
    # plan is the execution contract for non-declared jobs and must be used for the
    # static submit-time fit check too, otherwise a valid CUDA probe is rejected before
    # the scheduler can lease the host.
    job = {
        "needs": {"cuda": True},
        "reservation": {
            "ram_mb": 6656,
            "vram_mb": 14848,
            "cpu_threads": 4,
            "source": "generalized-history",
        },
        "plans": {
            "beast": {
                "ram_limit_mb": 4172.7,
                "est_vram_mb": 2799.8,
                "threads": 4,
            }
        },
    }
    ok, reasons = fits_some_host_capacity(job, _HOSTS)
    assert ok and reasons == []


def test_preflight_reports_missing_pin_and_caps():
    job = {"needs": {"host": "ghost"}, "reservation": {"ram_mb": 100}}
    ok, reasons = fits_some_host_capacity(job, _HOSTS)
    assert not ok and "not registered" in reasons[0]
    job = {"needs": {"cuda": True, "mps": True}, "reservation": {"ram_mb": 100}}
    ok, reasons = fits_some_host_capacity(job, _HOSTS)
    assert not ok and "caps" in reasons[0]


def test_preflight_requires_payload_sandbox_capability():
    job = {
        "needs": {"cuda": True, "payload_sandbox_v1": True},
        "reservation": {"ram_mb": 100, "vram_mb": 100},
    }
    ok, reasons = fits_some_host_capacity(job, _HOSTS)
    assert not ok
    assert reasons == ["no registered host has caps: cuda, payload_sandbox_v1"]

    sandbox_hosts = [
        {
            **_HOSTS[0],
            "caps": {**_HOSTS[0]["caps"], "payload_sandbox_v1": True},
        },
        _HOSTS[1],
    ]
    ok, reasons = fits_some_host_capacity(job, sandbox_hosts)
    assert ok and reasons == []


# ------------------------------------------------------- phase 3: measured sizing


def test_family_key_shapes():
    from mrun.protocol import family_key_for

    assert family_key_for("e", ["python", "x.py"], {"model": "qwen", "task_family": "score"}) \
        == "model:qwen:score"
    assert family_key_for("atlas", ["uv", "run", "scripts/prep.py", "--layer", "7"], {}) \
        == "cmd:atlas:prep.py"
    # per-item args don't split the family
    assert family_key_for("atlas", ["uv", "run", "scripts/prep.py", "--layer", "9"], {}) \
        == "cmd:atlas:prep.py"
    assert family_key_for("bench", ["python", "-c", "print(1)"], None) == "cmd:bench:python"


def test_exact_stats_n_aware(tmp_path):
    from mrun.server.db import DB

    db = DB(tmp_path / "mrun.db")
    for _i, peak in enumerate([1000.0, 1200.0]):
        db.add_estimate({
            "client_run_id": "run-x", "experiment": "e", "status": "succeeded",
            "ram_peak_mb": peak, "wall_s": 5.0, "family_key": "cmd:e:s.py",
        })
    s = db.exact_stats("run-x")
    assert s["n"] == 2 and s["ram_peak_mb"] == 1200.0  # n<3 -> max peak
    db.add_estimate({
        "client_run_id": "run-x", "experiment": "e", "status": "succeeded",
        "ram_peak_mb": 1100.0, "wall_s": 5.0, "family_key": "cmd:e:s.py",
    })
    s = db.exact_stats("run-x")
    assert s["n"] == 3 and s["ram_peak_mb"] == 1200.0  # p95 of [1000,1100,1200]
    assert db.exact_stats("nope") is None


def test_killed_rows_excluded_from_history_stats(tmp_path):
    from mrun.server.db import DB

    db = DB(tmp_path / "mrun.db")
    for i in range(3):
        db.add_estimate({
            "client_run_id": f"r{i}", "experiment": "e", "status": "succeeded",
            "ram_peak_mb": 1000.0, "family_key": "cmd:e:s.py",
        })
    db.add_estimate({
        "client_run_id": "r-k", "experiment": "e", "status": "killed_ram",
        "ram_peak_mb": 9000.0, "family_key": "cmd:e:s.py", "kill_state": "killed_ram",
    })
    s = db.history_stats("cmd:e:s.py")
    assert s["n"] == 3 and s["p95_ram_mb"] == 1000.0
