"""Pure admission-control tests — synthetic hosts/telemetry, no server."""

from __future__ import annotations

import time

from mrun.server.scheduler import (
    admissible,
    history_admission_reservation,
    host_eta_s,
    pick_job_for_host,
    rank_hosts,
    resident_yield_requests,
    resolve_reservation,
    swap_limit_for,
)


def _host(
    name="h1",
    ram=32_000,
    vram=0.0,
    vram_free=None,
    cuda=False,
    ram_free=28_000,
    ts_age=0.0,
    running=(),
    cpus=16,
    os_name="unknown",
):
    return {
        "name": name,
        "os": os_name,
        "ram_total_mb": ram,
        "vram_total_mb": vram,
        "cpu_threads": cpus,
        "caps": {"cuda": cuda, "mps": not cuda, "cpu": True},
        "telemetry": {
            "ts": time.time() - ts_age,
            "ram_free_mb": ram_free,
            "vram_free_mb": vram if vram_free is None else vram_free,
            "disk_free_gb": 500.0,
            "running": list(running),
        },
    }


def _job(ram_mb=4000, vram_mb=0.0, cuda=False, host_pin=None, threads=4, est_wall_s=None):
    return {
        "needs": {"cuda": cuda, "host": host_pin},
        "reservation": {
            "ram_mb": ram_mb,
            "vram_mb": vram_mb,
            "cpu_threads": threads,
            "est_wall_s": est_wall_s,
        },
    }


def test_fits_when_room():
    ok, why = admissible(_host(), _job(ram_mb=4000), [])
    assert ok, why


def test_rank_hosts_prefers_exact_template_then_warm_arena() -> None:
    job = _job()
    job["config"] = {
        "resident_arena_key": "arena-a",
        "graph_template_key": "template-a",
    }
    cold = _host(name="cold")
    warm_arena = _host(name="arena")
    warm_template = _host(name="template")
    warm_arena["telemetry"]["executable_inventory"] = {"warm_arena_keys": ["arena-a"]}
    warm_template["telemetry"]["executable_inventory"] = {
        "warm_arena_keys": ["arena-a"],
        "warm_graph_template_keys": ["template-a"],
    }
    assert rank_hosts(
        job,
        [cold, warm_arena, warm_template],
        {"cold": [], "arena": [], "template": []},
    ) == ["template", "arena", "cold"]


def test_guarded_payload_requires_updated_agent_capability():
    job = _job(ram_mb=4000)
    job["needs"]["payload_custody_v2"] = True
    old_agent = _host()
    ok, why = admissible(old_agent, job, [])
    assert not ok
    assert why == "needs payload custody v2 agent"

    updated_agent = _host()
    updated_agent["caps"]["payload_custody_v2"] = True
    ok, why = admissible(updated_agent, job, [])
    assert ok, why


def test_guarded_debugger_requires_updated_agent_capability():
    job = _job(ram_mb=4000)
    job["needs"]["saturn_debug_credential_v1"] = True
    old_agent = _host()
    ok, why = admissible(old_agent, job, [])
    assert not ok
    assert why == "needs Saturn job debugger credential v1 agent"

    updated_agent = _host()
    updated_agent["caps"]["saturn_debug_credential_v1"] = True
    ok, why = admissible(updated_agent, job, [])
    assert ok, why


def test_guarded_payload_sandbox_fails_closed_on_legacy_agent():
    job = _job(ram_mb=4000)
    job["needs"]["payload_sandbox_v1"] = True
    old_agent = _host()
    ok, why = admissible(old_agent, job, [])
    assert not ok
    assert why == "needs payload sandbox v1 agent"

    updated_agent = _host()
    updated_agent["caps"]["payload_sandbox_v1"] = True
    ok, why = admissible(updated_agent, job, [])
    assert ok, why


def test_reservations_block_not_free_ram():
    # host still shows lots of FREE ram (job admitted but not malloc'ing yet) — the
    # committed reservation must still block the second job
    running = [_job(ram_mb=20_000)]
    ok, why = admissible(_host(ram_free=30_000), _job(ram_mb=8000), running)
    assert not ok
    assert "committed" in why


def test_history_admission_reservation_unwedges_overdeclared_active_job():
    host = _host(
        name="beast",
        os_name="linux",
        ram=66_509,
        ram_free=57_512,
        cuda=True,
        vram=16_376,
        cpus=16,
        running=[{"job_id": "active", "tree_rss_mb": 3400.0, "vram_mb": 960.0}],
    )
    active = _job(ram_mb=30_000, vram_mb=9000, cuda=True, threads=10)
    active["job_id"] = "active"
    active["reservation"]["source"] = "declared"
    candidate = _job(ram_mb=22_000, vram_mb=10_000, cuda=True, threads=6)

    ok, why = admissible(host, candidate, [active])
    assert not ok
    assert "ram" in why or "vram" in why

    admission = history_admission_reservation(
        active,
        {"n": 3, "p95_ram_mb": 3376.6, "p95_vram_mb": 918.0},
        {"tree_rss_mb": 3400.0, "vram_mb": 960.0},
    )
    assert admission is not None
    active = {**active, "admission_reservation": admission}

    ok, why = admissible(host, candidate, [active])
    assert ok, why


def test_margin_reserved():
    # 32GB host, margin = max(15%, 2GiB) = 4.8GB -> a 28GB job never fits
    ok, why = admissible(_host(), _job(ram_mb=28_000), [])
    assert not ok


def test_baseline_counts_non_mrun_load():
    # host has only 4GB free and no mrun jobs -> baseline eats the capacity
    ok, why = admissible(_host(ram_free=4_000), _job(ram_mb=10_000), [])
    assert not ok
    assert "baseline" in why


def test_stale_telemetry_blocks():
    ok, why = admissible(_host(ts_age=120.0), _job(), [])
    assert not ok
    assert "stale" in why


def test_needs_cuda_and_pin():
    assert not admissible(_host(cuda=False), _job(cuda=True), [])[0]
    assert admissible(_host(cuda=True, vram=16_000), _job(cuda=True), [])[0]
    assert not admissible(_host(name="mac"), _job(host_pin="beast"), [])[0]
    assert admissible(_host(name="beast"), _job(host_pin="beast"), [])[0]


def test_vram_committed():
    host = _host(cuda=True, vram=16_000)
    running = [_job(vram_mb=12_000)]
    ok, why = admissible(host, _job(vram_mb=6_000), running)
    assert not ok
    assert "vram" in why


def test_vram_baseline_counts_non_mrun_gpu_load():
    host = _host(cuda=True, vram=16_000, vram_free=4_000)
    ok, why = admissible(host, _job(vram_mb=6_000, cuda=True), [])
    assert not ok
    assert "vram" in why and "baseline" in why


def test_orphaned_agent_rss_remains_in_baseline():
    host = _host(
        ram=32_000,
        ram_free=20_000,
        running=[{"job_id": "lost", "tree_rss_mb": 8_000.0, "vram_mb": 0.0}],
    )
    ok, why = admissible(host, _job(ram_mb=14_000), [])
    assert not ok
    assert "ram" in why and "baseline" in why


def test_orphaned_agent_vram_remains_in_baseline():
    host = _host(
        cuda=True,
        vram=16_000,
        vram_free=6_000,
        running=[{"job_id": "lost", "tree_rss_mb": 0.0, "vram_mb": 8_000.0}],
    )
    ok, why = admissible(host, _job(vram_mb=6_000, cuda=True), [])
    assert not ok
    assert "vram" in why and "baseline" in why


def test_cpu_threads_hard():
    ok, why = admissible(_host(cpus=8), _job(threads=6), [_job(threads=4)])
    assert not ok
    assert "cpu" in why


def test_pick_job_order():
    now = time.time()
    host = _host()
    blocked = _job(ram_mb=40_000)
    blocked.update({"created_ts": now - 600, "priority": 100, "job_id": "blocked"})
    ready = _job(ram_mb=1000)
    ready.update({"created_ts": now, "priority": 0, "job_id": "ready"})
    jobs = [blocked, ready]  # priority/created order supplied by DB.jobs()
    assert jobs[0]["priority"] > jobs[1]["priority"]
    assert jobs[0]["created_ts"] < jobs[1]["created_ts"]
    assert not admissible(host, blocked, [])[0]
    picked = pick_job_for_host(host, jobs, [])
    assert picked is ready  # first admissible, not first in queue


def test_pick_job_returns_none_when_all_queued_jobs_are_inadmissible():
    now = time.time()
    host = _host()
    blocked_by_ram = _job(ram_mb=40_000)
    blocked_by_ram.update({"created_ts": now - 600, "priority": 100, "job_id": "blocked-by-ram"})
    blocked_by_cuda = _job(ram_mb=1000, cuda=True)
    blocked_by_cuda.update({"created_ts": now - 300, "priority": 50, "job_id": "blocked-by-cuda"})

    assert pick_job_for_host(host, [blocked_by_ram, blocked_by_cuda], [], now=now) is None


def test_aged_pinned_job_stops_resource_backfill_until_host_drains():
    now = time.time()
    host = _host(name="beast", cuda=True, vram=16_000)
    active = _job(ram_mb=12_000, vram_mb=4000, cuda=True, host_pin="beast")
    blocked = _job(ram_mb=10_000, vram_mb=10_000, cuda=True, host_pin="beast")
    blocked.update({"created_ts": now - 600, "priority": 80, "job_id": "blocked"})
    backfill = _job(ram_mb=1000, vram_mb=500, cuda=True, host_pin="beast")
    backfill.update({"created_ts": now, "priority": 10, "job_id": "backfill"})

    assert pick_job_for_host(host, [blocked, backfill], [active], now=now) is None
    assert pick_job_for_host(host, [blocked, backfill], [], now=now) is blocked


def test_recent_pinned_job_allows_bounded_backfill():
    now = time.time()
    host = _host(name="beast", cuda=True, vram=16_000)
    active = _job(ram_mb=12_000, vram_mb=4000, cuda=True, host_pin="beast")
    blocked = _job(ram_mb=10_000, vram_mb=10_000, cuda=True, host_pin="beast")
    blocked.update({"created_ts": now - 60, "priority": 80, "job_id": "blocked"})
    backfill = _job(ram_mb=1000, vram_mb=500, cuda=True, host_pin="beast")
    backfill.update({"created_ts": now, "priority": 10, "job_id": "backfill"})

    assert pick_job_for_host(host, [blocked, backfill], [active], now=now) is backfill


def test_impossible_or_unpinned_job_does_not_create_drain_barrier():
    now = time.time()
    host = _host(name="beast", cuda=True, vram=16_000)
    active = _job(ram_mb=4000, vram_mb=5000, cuda=True, host_pin="beast")
    backfill = _job(ram_mb=1000, vram_mb=500, cuda=True, host_pin="beast")
    backfill.update({"created_ts": now, "priority": 10, "job_id": "backfill"})

    impossible = _job(ram_mb=40_000, vram_mb=20_000, cuda=True, host_pin="beast")
    impossible.update({"created_ts": now - 600, "priority": 80, "job_id": "impossible"})
    assert pick_job_for_host(host, [impossible, backfill], [active], now=now) is backfill

    unpinned = _job(ram_mb=10_000, vram_mb=10_000, cuda=True)
    unpinned.update({"created_ts": now - 600, "priority": 80, "job_id": "unpinned"})
    assert pick_job_for_host(host, [unpinned, backfill], [active], now=now) is backfill


def test_higher_priority_finite_job_requests_cooperative_resident_yield():
    now = time.time()
    host = _host(
        name="beast",
        ram=66_509,
        ram_free=55_000,
        cuda=True,
        vram=16_376,
        vram_free=2_992,
        running=[{"job_id": "resident", "tree_rss_mb": 10_000, "vram_mb": 13_384}],
    )
    host["limits"] = {"vram_margin_mb": 300}
    resident = _job(ram_mb=24_000, vram_mb=13_500, cuda=True, host_pin="beast")
    resident.update(
        {
            "job_id": "resident",
            "priority": 0,
            "config": {
                "resident_worker": True,
                "preemptible_resident": True,
                "resident_yield_after_s": 60,
            },
        }
    )
    finite = _job(ram_mb=4_000, vram_mb=2_000, cuda=True, host_pin="beast")
    finite.update(
        {
            "job_id": "finite",
            "experiment": "finite-proof",
            "created_ts": now,
            "priority": 20,
            "config": {},
        }
    )

    requests = resident_yield_requests([host], [finite], {"beast": [resident]}, now=now)

    assert requests["resident"]["blocker_job_id"] == "finite"
    assert requests["resident"]["blocked_reason"].startswith("vram:")


def test_finite_saturn_resident_program_can_request_service_yield():
    now = time.time()
    host = _host(
        name="beast",
        ram=66_509,
        ram_free=37_000,
        cuda=True,
        vram=16_376,
        vram_free=2_992,
        running=[{"job_id": "service", "tree_rss_mb": 15_000, "vram_mb": 13_384}],
    )
    host["limits"] = {"vram_margin_mb": 300}
    service = _job(ram_mb=24_000, vram_mb=13_500, cuda=True, host_pin="beast")
    service.update(
        {
            "job_id": "service",
            "priority": 0,
            "config": {
                "resident_worker": True,
                "preemptible_resident": True,
                "resident_yield_after_s": 60,
            },
        }
    )
    finite = _job(ram_mb=32_000, vram_mb=13_500, cuda=True, host_pin="beast")
    finite.update(
        {
            "job_id": "finite-saturn",
            "experiment": "finite-saturn-program",
            "created_ts": now,
            "priority": 100,
            "config": {
                "resident_worker": True,
                "scheduling_contract": {"execution_model": "finite-resident-program"},
            },
        }
    )

    requests = resident_yield_requests([host], [finite], {"beast": [service]}, now=now)

    assert requests["service"]["blocker_job_id"] == "finite-saturn"


def test_resident_yield_waits_for_quantum_at_equal_priority():
    now = time.time()
    host = _host(name="beast", cuda=True, vram=16_376)
    host["limits"] = {"vram_margin_mb": 300}
    resident = _job(ram_mb=24_000, vram_mb=13_500, cuda=True, host_pin="beast")
    resident.update(
        {
            "job_id": "resident",
            "priority": 0,
            "config": {
                "resident_worker": True,
                "preemptible_resident": True,
                "resident_yield_after_s": 60,
            },
        }
    )
    finite = _job(ram_mb=4_000, vram_mb=2_000, cuda=True, host_pin="beast")
    finite.update({"job_id": "finite", "created_ts": now - 30, "priority": 0, "config": {}})
    assert not resident_yield_requests([host], [finite], {"beast": [resident]}, now=now)

    finite["created_ts"] = now - 61
    assert "resident" in resident_yield_requests([host], [finite], {"beast": [resident]}, now=now)


def test_cofitting_job_does_not_request_resident_yield():
    now = time.time()
    host = _host(name="beast", cuda=True, vram=16_376)
    host["limits"] = {"vram_margin_mb": 300}
    resident = _job(ram_mb=8_000, vram_mb=10_000, cuda=True, host_pin="beast")
    resident.update(
        {
            "job_id": "resident",
            "priority": 0,
            "config": {"resident_worker": True, "preemptible_resident": True},
        }
    )
    finite = _job(ram_mb=1_000, vram_mb=500, cuda=True, host_pin="beast")
    finite.update({"job_id": "finite", "created_ts": now - 600, "priority": 20, "config": {}})

    assert not resident_yield_requests([host], [finite], {"beast": [resident]}, now=now)


def test_nonpreemptible_resident_never_receives_yield_request():
    now = time.time()
    host = _host(name="beast", cuda=True, vram=16_376)
    resident = _job(ram_mb=24_000, vram_mb=13_500, cuda=True, host_pin="beast")
    resident.update(
        {
            "job_id": "resident",
            "priority": 0,
            "config": {"resident_worker": True},
        }
    )
    finite = _job(ram_mb=4_000, vram_mb=2_000, cuda=True, host_pin="beast")
    finite.update({"job_id": "finite", "created_ts": now - 600, "priority": 100, "config": {}})

    assert not resident_yield_requests([host], [finite], {"beast": [resident]}, now=now)


def test_reservation_ladder():
    # declared wins
    r = resolve_reservation({"ram_mb": 1234}, {"ram_peak_mb": 500}, {"ram_mb": 999})
    assert r.source == "declared" and r.ram_mb == 1234
    # history next (n<3 -> max peak x1.3), floored at the first-principles client
    # estimate, 512-rounded
    r = resolve_reservation(None, {"ram_peak_mb": 500, "wall_s": 60}, {"ram_mb": 999})
    assert r.source == "history" and r.ram_mb == 1024 and r.est_wall_s == 60
    r = resolve_reservation(None, {"ram_peak_mb": 2000, "wall_s": 60}, {"ram_mb": 999})
    assert r.source == "history" and r.ram_mb == 3072  # round_up_512(2000*1.3)
    # n>=3 -> stable p95 basis with the tighter factor
    r = resolve_reservation(None, {"ram_peak_mb": 2000, "wall_s": 60, "n": 4}, None)
    assert r.source == "history" and r.ram_mb == 2560  # round_up_512(2000*1.25)
    # tiny observed Python jobs still need a sane execution floor
    r = resolve_reservation(None, {"ram_peak_mb": 5, "wall_s": 1}, None)
    assert r.source == "history" and r.ram_mb == 512
    # generalized history carries both RAM and VRAM when exact-config history is absent
    r = resolve_reservation(
        None,
        None,
        {"ram_mb": 999, "vram_mb": 111},
        generalized={"p95_ram_mb": 2000, "p95_vram_mb": 1000, "p50_wall_s": 60},
        needs_cuda=True,
    )
    assert r.source == "generalized-history"
    assert r.ram_mb == 3072 and r.vram_mb == 1536  # p95 x 1.5, 512-rounded
    r = resolve_reservation(
        None,
        None,
        {"ram_mb": 999},
        generalized={"p95_ram_mb": 2000, "p95_vram_mb": 1000},
    )
    assert r.source == "generalized-history" and r.vram_mb == 0
    # client estimate next
    r = resolve_reservation(None, None, {"ram_mb": 999})
    assert r.source == "estimated" and r.ram_mb == 999
    # small probe default last — growth-on-kill covers the miss
    r = resolve_reservation(None, None, None, needs_cuda=True)
    assert r.source == "probe-default" and r.ram_mb == 4096 and r.vram_mb == 4096


# ---------------------------------------------------------------- P0 crash-safety gates


def test_committed_at_kill_ceiling():
    # Two jobs that fit at 1.0x reservation but not at the 1.1x kill ceiling must not
    # co-admit: each may legally grow to ceiling before its guard fires.
    # 32GB host, margin 4.8GB -> budget 27.2GB. Two 13GB jobs: 1.0x commit = 26 (fits),
    # ceiling commit = 28.6 (must refuse the second).
    running = [_job(ram_mb=13_000)]
    ok, why = admissible(_host(ram_free=31_000), _job(ram_mb=13_000), running)
    assert not ok
    assert "ceiling" in why


def test_swap_pressure_refusal(monkeypatch):
    monkeypatch.setenv("MRUN_SWAP_MAX_MB", "2048")
    monkeypatch.delenv("MRUN_HOST_SWAP_MAX_MB", raising=False)
    host = _host()
    host["telemetry"]["swap_used_mb"] = 5000.0
    ok, why = admissible(host, _job(ram_mb=1000), [])
    assert not ok and "swap" in why

    host = _host()
    host["telemetry"]["mem_pressure"] = 2
    ok, why = admissible(host, _job(ram_mb=1000), [])
    assert not ok and "pressure" in why

    host = _host()
    host["telemetry"]["mem_pressure"] = 1  # normal
    host["telemetry"]["swap_used_mb"] = 100.0
    assert admissible(host, _job(ram_mb=1000), [])[0]


def test_linux_sticky_swap_with_healthy_available_ram_is_allowed(monkeypatch):
    monkeypatch.setenv("MRUN_SWAP_MAX_MB", "2048")
    monkeypatch.delenv("MRUN_HOST_SWAP_MAX_MB", raising=False)
    host = _host(
        name="beast",
        os_name="linux",
        ram=66_509,
        ram_free=57_512,
        cuda=True,
        vram=16_376,
    )
    host["telemetry"]["swap_used_mb"] = 2_887.0

    ok, why = admissible(host, _job(ram_mb=30_000, vram_mb=13_900, cuda=True), [])

    assert ok, why


def test_linux_swap_with_low_available_ram_is_refused(monkeypatch):
    monkeypatch.setenv("MRUN_SWAP_MAX_MB", "2048")
    monkeypatch.delenv("MRUN_HOST_SWAP_MAX_MB", raising=False)
    host = _host(os_name="linux", ram=64_000, ram_free=3_000)
    host["telemetry"]["swap_used_mb"] = 5_000.0

    ok, why = admissible(host, _job(ram_mb=1_000), [])

    assert not ok
    assert "swap" in why


def test_linux_explicit_memory_pressure_still_refuses_with_free_ram(monkeypatch):
    monkeypatch.setenv("MRUN_SWAP_MAX_MB", "2048")
    host = _host(os_name="linux", ram=64_000, ram_free=53_000)
    host["telemetry"]["swap_used_mb"] = 5_000.0
    host["telemetry"]["mem_pressure"] = 2

    ok, why = admissible(host, _job(ram_mb=1_000), [])

    assert not ok
    assert "pressure" in why


def test_darwin_swap_fallback_remains_strict(monkeypatch):
    monkeypatch.setenv("MRUN_SWAP_MAX_MB", "2048")
    monkeypatch.delenv("MRUN_HOST_SWAP_MAX_MB", raising=False)
    host = _host(os_name="darwin", ram=64_000, ram_free=53_000)
    host["telemetry"]["swap_used_mb"] = 5_000.0

    ok, why = admissible(host, _job(ram_mb=1_000), [])

    assert not ok
    assert "swap" in why


def test_per_host_swap_limit(monkeypatch):
    monkeypatch.setenv("MRUN_HOST_SWAP_MAX_MB", '{"beast": 8192, "mbp1": 2048}')
    beast = _host(name="beast")
    beast["telemetry"]["swap_used_mb"] = 5000.0
    assert swap_limit_for(beast) == 8192.0
    assert admissible(beast, _job(ram_mb=1000), [])[0]

    mbp = _host(name="mbp1")
    mbp["telemetry"]["swap_used_mb"] = 5000.0
    assert swap_limit_for(mbp) == 2048.0
    ok, why = admissible(mbp, _job(ram_mb=1000), [])
    assert not ok and "swap" in why


def test_normal_pressure_signal_allows_accumulated_swap(monkeypatch):
    monkeypatch.setenv("MRUN_HOST_SWAP_MAX_MB", '{"mbp1": 2048}')
    mbp = _host(name="mbp1", ram=19_327, ram_free=8_044)
    mbp["telemetry"]["swap_used_mb"] = 3_155.0
    mbp["telemetry"]["mem_pressure"] = 1

    ok, why = admissible(mbp, _job(ram_mb=2_048, threads=1), [])

    assert ok, why


def test_per_host_margin_override(monkeypatch):
    # 18GB mac: default margin 2.7GB admits an 11GB job; the mbp1 override (25%/4GB)
    # must refuse it.
    job = _job(ram_mb=12_000)
    host = _host(name="mbp1", ram=18_000, ram_free=17_000)
    assert admissible(host, job, [])[0]
    monkeypatch.setenv("MRUN_HOST_MARGINS", '{"mbp1": {"fraction": 0.25, "min_mb": 4096}}')
    ok, why = admissible(host, job, [])
    assert not ok and "margin" in why
    # other hosts keep the default
    assert admissible(_host(name="beast", ram=18_000, ram_free=17_000), job, [])[0]


def test_external_rss_not_double_counted():
    # A local (external) run's live RSS shows in (total-free); folding its heartbeat RSS
    # into running_rss keeps it out of the baseline term (it's already in committed).
    ext = {
        "needs": {},
        "reservation": {"ram_mb": 6_000, "cpu_threads": 4},
        "payload_kind": "external",
        "external_rss_mb": 5_000.0,
    }
    # 32GB host, 5GB of the external run is in use -> free 23GB (baseline 4GB real).
    # Without the fold: baseline = 9GB, committed 6.6 -> avail 11.6, 10GB ceiling 11 fits barely.
    # With the fold: baseline 4GB, committed 6.6, margin 4.8 -> avail 16.6 -> 10GB job fits.
    ok, why = admissible(_host(ram_free=23_000), _job(ram_mb=10_000), [ext])
    assert ok, why


def test_external_vram_not_double_counted():
    ext = {
        "needs": {"cuda": True},
        "reservation": {"ram_mb": 1_000, "vram_mb": 6_000, "cpu_threads": 1},
        "payload_kind": "external",
        "external_vram_mb": 5_000.0,
    }
    host = _host(cuda=True, vram=16_000, vram_free=7_000)

    ok, why = admissible(host, _job(ram_mb=1_000, vram_mb=3_000, cuda=True), [ext])

    assert ok, why


# ---------------------------------------------------------------- P1 plan-aware placement


def _plan(backend="hf", device="cpu", ram_limit=3000.0, vram=0.0, threads=4, weights_gb=2.0):
    return {
        "backend": backend,
        "device": device,
        "ram_limit_mb": ram_limit,
        "est_vram_mb": vram,
        "threads": threads,
        "weights_gb": weights_gb,
        "max_batch": 16,
        "dtype": "float32",
    }


def test_per_host_plan_overrides_reservation():
    from mrun.server.scheduler import effective_reservation

    job = _job(ram_mb=8192)
    job["reservation"]["source"] = "default"
    job["plans"] = {"h1": _plan(ram_limit=3000.0)}
    res = effective_reservation(job, "h1")
    assert res.ram_mb == 3000.0 and res.source == "plan"
    # declared always wins
    job["reservation"]["source"] = "declared"
    assert effective_reservation(job, "h1").ram_mb == 8192


def test_rank_hosts_prefers_cuda_then_warm():
    from mrun.server.scheduler import rank_hosts

    mac = _host(name="mbp1", ram=18_000, ram_free=15_000)
    beast = _host(name="beast", ram=61_000, ram_free=55_000, cuda=True, vram=16_000)
    job = _job(ram_mb=3000)
    job["plans"] = {
        "mbp1": _plan(backend="hf", device="cpu"),
        "beast": _plan(backend="hf", device="cuda"),
    }
    ranked = rank_hosts(job, [mac, beast], {"mbp1": [], "beast": []})
    assert ranked[0] == "beast"

    # two cpu hosts, one warm -> warm wins
    h2 = _host(name="h2", ram=18_000, ram_free=15_000)
    job2 = _job(ram_mb=3000)
    job2["config"] = {"model": "m"}
    job2["plans"] = {"mbp1": _plan(), "h2": _plan()}
    ranked = rank_hosts(job2, [mac, h2], {"mbp1": [], "h2": []}, warm_map={"h2": ["weights"]})
    assert ranked[0] == "h2"


def test_best_host_leasing_and_starvation():
    from mrun.server.scheduler import pick_job_for_host

    mac = _host(name="mbp1", ram=61_000, ram_free=55_000)
    beast = _host(name="beast", ram=61_000, ram_free=55_000, cuda=True, vram=16_000)
    job = _job(ram_mb=3000)
    job["plans"] = {
        "mbp1": _plan(device="cpu"),
        "beast": _plan(device="cuda"),
    }
    job["created_ts"] = time.time()
    ctx = {"all_hosts": [mac, beast], "active_by_host": {"mbp1": [], "beast": []}}
    # fresh job: the mac polls -> not its best host -> nothing granted
    assert pick_job_for_host(mac, [job], [], **ctx) is None
    # the best host polls -> granted
    assert pick_job_for_host(beast, [job], [], **ctx) is job
    # starved job: any admissible host takes it
    job["created_ts"] = time.time() - 120
    assert pick_job_for_host(mac, [job], [], **ctx) is job


def test_pick_job_normalizes_content_custody_model_for_warm_lookup():
    from mrun.server.scheduler import pick_job_for_host

    host = _host(name="beast", ram=61_000, ram_free=55_000)
    job = _job(ram_mb=3000)
    model = {
        "source_id": "Qwen/Qwen2.5-1.5B",
        "revision": "frozen-revision",
        "assets": [{"path": "config.json", "sha256": "a" * 64}],
    }
    job["config"] = {"model": model}
    seen = []

    assert (
        pick_job_for_host(
            host,
            [job],
            [],
            warm_lookup=lambda key: seen.append(key) or {},
        )
        is job
    )
    assert seen == ["Qwen/Qwen2.5-1.5B"]


def test_pick_job_prefers_structured_model_name_for_warm_lookup():
    from mrun.server.scheduler import pick_job_for_host

    host = _host(name="beast", ram=61_000, ram_free=55_000)
    job = _job(ram_mb=3000)
    job["config"] = {
        "model": {
            "model_name": "qwen2.5-coder-1.5b-instruct",
            "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        }
    }
    seen = []

    assert (
        pick_job_for_host(
            host,
            [job],
            [],
            warm_lookup=lambda key: seen.append(key) or {"beast": ["weights"]},
        )
        is job
    )
    assert seen == ["qwen2.5-coder-1.5b-instruct"]


def test_cold_model_charged_against_models_mount():
    host = _host()
    host["models_mount"] = "/mnt/big"
    host["telemetry"]["disks"] = [
        {"mount": "/", "total_gb": 200.0, "free_gb": 100.0},
        {"mount": "/mnt/big", "total_gb": 2000.0, "free_gb": 10.0},
    ]
    job = _job(ram_mb=1000)
    job["plans"] = {"h1": _plan(weights_gb=30.0)}
    # cold: 1 (disk_gb) + 30 (weights) > 10 - 5 -> refuse
    ok, why = admissible(host, job, [], warm_kinds=None)
    assert not ok and "cold model" in why
    assert "5.0GB reserve" in why
    assert "5.0GB available" in why
    # warm: no download charge -> fits
    ok, why = admissible(host, job, [], warm_kinds=["weights"])
    assert ok, why


def test_cold_artifact_uses_selected_mount_for_disk_admission():
    host = _host()
    host["models_mount"] = "/mnt/big"
    host["telemetry"]["disks"] = [
        {"mount": "/mnt/big", "total_gb": 2000.0, "free_gb": 10.0},
        {"mount": "/mnt/ssd1tb", "total_gb": 1000.0, "free_gb": 100.0},
    ]
    job = _job(ram_mb=1000)
    job["plans"] = {
        "h1": {
            **_plan(weights_gb=30.0),
            "artifact_mount": "/mnt/ssd1tb",
            "artifact_locator": {"mount": "/mnt/ssd1tb", "path": "/mnt/ssd1tb/qstore"},
        }
    }

    ok, why = admissible(host, job, [], warm_kinds=None)
    assert ok, why


def test_bound_artifact_requires_exact_warm_identity():
    host = _host()
    host["models_mount"] = "/mnt/big"
    host["telemetry"]["disks"] = [
        {"mount": "/mnt/big", "total_gb": 2000.0, "free_gb": 10.0},
    ]
    job = _job(ram_mb=1000)
    job["plans"] = {
        "h1": {
            **_plan(backend="paged", weights_gb=30.0),
            "artifact_id": "artifact:target",
        }
    }

    ok, why = admissible(host, job, [], warm_kinds=["artifact:other"])
    assert not ok and "cold model" in why
    ok, why = admissible(host, job, [], warm_kinds=["artifact:target"])
    assert ok, why


def test_host_eta():
    now = time.time()
    jobs = [
        {"reservation": {"est_wall_s": 100}, "started_ts": now - 40},
        {"reservation": {"est_wall_s": 30}, "started_ts": now - 10},
    ]
    eta = host_eta_s(_host(), jobs)
    assert 55 <= eta <= 61
    assert host_eta_s(_host(), [{"reservation": {}, "started_ts": now}]) is None
    assert host_eta_s(_host(), []) == 0.0


# ------------------------------------------------------- operator host limits (phase 2)


def test_margin_from_db_limits_beats_env_and_default():
    from mrun.server.reservation import margin_for

    host = _host(ram=32_000)
    assert margin_for(host) == 4800  # default 15%
    host["limits"] = {"ram_margin_mb": 6000}
    assert margin_for(host) == 6000
    host["limits"] = {"ram_margin_mb": 2000, "ram_margin_fraction": 0.25}
    assert margin_for(host) == 8000  # max(fraction*total, mb)
    host["limits"] = {"ram_margin_mb": 100}  # below the 1GB never-crash floor
    assert margin_for(host) == 1024


def test_vram_margin_from_db_limits():
    from mrun.server.reservation import vram_margin_for

    host = _host(vram=16_000)
    assert vram_margin_for(host) == 1024
    host["limits"] = {"vram_margin_mb": 2048}
    assert vram_margin_for(host) == 2048


def test_disabled_host_refuses_admission():
    host = _host()
    host["limits"] = {"enabled": False}
    ok, why = admissible(host, _job(ram_mb=1000), [])
    assert not ok and "disabled" in why


def test_max_concurrent_limit():
    host = _host()
    host["limits"] = {"max_concurrent": 1}
    ok, why = admissible(host, _job(ram_mb=1000), [_job(ram_mb=1000)])
    assert not ok and "max_concurrent" in why
    ok, why = admissible(host, _job(ram_mb=1000), [])
    assert ok, why


def test_host_limits_roundtrip_via_db(tmp_path):
    import os

    from mrun.server.db import DB

    os.environ["MRUN_SERVER_DATA"] = str(tmp_path)
    db = DB()
    db.upsert_host(
        "h1",
        {
            "os": "linux",
            "arch": "x86_64",
            "caps": {"cpu": True},
            "ram_total_mb": 32_000,
            "vram_total_mb": 0,
            "cpu_threads": 16,
            "disk_total_gb": 100,
        },
    )
    db.set_setting("host_limits:h1", {"ram_margin_mb": 6000, "max_concurrent": 2})
    rows = db.host_rows()
    assert rows[0]["limits"] == {"ram_margin_mb": 6000, "max_concurrent": 2}
