"""Admission control — reservations, not instantaneous free RAM.

A job admitted but not yet malloc'ing still blocks the next one (its reservation is
committed). Live telemetry only feeds the *baseline* term (what the host uses outside
mrun jobs — OS, user apps) and the staleness check (a sleeping mac is not schedulable).
"""

from __future__ import annotations

import math
import time
from typing import Any

from ..protocol import (
    MEM_PRESSURE_WARN,
    RAM_KILL_FACTOR,
    TELEMETRY_STALE_S,
    Reservation,
    kill_ceiling_mb,
    swap_signal_is_critical,
)
from .executable_inventory import ExecutableInventory
from .reservation import (  # noqa: F401 — re-exported: admission callers import from here
    ADMISSION_HISTORY_SAFETY_FACTOR,
    ADMISSION_LIVE_SAFETY_FACTOR,
    ADMISSION_MIN_RAM_MB,
    ADMISSION_MIN_VRAM_MB,
    ADMISSION_ROUND_MB,
    HISTORY_RESERVATION_MIN_RAM_MB,
    HISTORY_RESERVATION_MIN_VRAM_MB,
    RAM_MARGIN_FRACTION,
    RAM_MARGIN_MIN_MB,
    VRAM_MARGIN_MB,
    SizedReservation,
    _round_up_mb,
    effective_reservation,
    history_admission_reservation,
    host_ram_cap_mb,
    host_vram_cap_mb,
    hosts_satisfying_needs,
    margin_for,
    plan_for_host,
    resolve_reservation,
    size_reservation,
    swap_limit_for,
    vram_margin_for,
)


def _fresh(telemetry: dict[str, Any] | None, now: float | None = None) -> bool:
    if not telemetry or telemetry.get("ts") is None:
        return False
    return (now or time.time()) - float(telemetry["ts"]) <= TELEMETRY_STALE_S


def _committed_reservation(job: dict[str, Any]) -> Reservation:
    return Reservation.from_dict(job.get("admission_reservation") or job.get("reservation"))


def _configured_vram_ceiling_mb(job: dict[str, Any]) -> float | None:
    """Return an optional caller-owned hard VRAM ceiling.

    This is deliberately a job contract rather than a scheduler-wide GPU size:
    a 15.5 GB ceiling is appropriate for Diffusion View's 16 GB Beast card, but
    must not constrain a future host with a larger accelerator. Invalid optional
    values are ignored here; the ordinary physical-capacity checks remain in force.
    """
    raw = (job.get("config") or {}).get("max_vram_ceiling_mb")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def model_inventory_keys(model: Any) -> tuple[str, ...]:
    """Return canonical inventory lookup candidates for a job model binding.

    Guarded jobs can carry a structured content-custody record. Stringifying that
    record is neither a stable identity nor the model key agents publish in their
    inventory, so prefer its explicit registry/name fields and fall back through
    source identifiers.
    """

    if isinstance(model, str):
        value = model.strip()
        return (value,) if value else ()
    if not isinstance(model, dict):
        return ()
    values = []
    for field in ("model_name", "name", "model", "model_id", "source_id", "hf_id"):
        value = model.get(field)
        if isinstance(value, str) and value.strip() and value.strip() not in values:
            values.append(value.strip())
    return tuple(values)


def warm_artifacts_for_model(model: Any, warm_lookup: Any) -> dict[str, list[str]]:
    """Resolve warm artifacts without ever using a stringified model dictionary."""

    if warm_lookup is None:
        return {}
    for key in model_inventory_keys(model):
        found = warm_lookup(key) or {}
        if found:
            return found
    return {}


def admissible(
    host: dict[str, Any],
    job: dict[str, Any],
    active_jobs: list[dict[str, Any]],
    *,
    now: float | None = None,
    warm_kinds: list[str] | None = None,
) -> tuple[bool, str]:
    """Can ``job`` start on ``host`` right now? Returns (ok, reason)."""
    needs = job.get("needs") or {}
    res = effective_reservation(job, host.get("name") or "")
    tel = host.get("telemetry")

    if needs.get("host") and needs["host"] != host["name"]:
        return False, f"pinned to {needs['host']}"
    limits = host.get("limits") or {}
    if limits.get("enabled") is False:
        return False, "host disabled by operator"
    max_concurrent = limits.get("max_concurrent")
    if max_concurrent is not None and len(active_jobs) >= int(max_concurrent):
        return False, f"concurrency: at operator max_concurrent {int(max_concurrent)}"
    caps = host.get("caps") or {}
    if needs.get("cuda") and not caps.get("cuda"):
        return False, "needs cuda"
    if needs.get("mps") and not caps.get("mps"):
        return False, "needs mps"
    if needs.get("payload_custody_v2") and not caps.get("payload_custody_v2"):
        return False, "needs payload custody v2 agent"
    if needs.get("saturn_debug_credential_v1") and not caps.get("saturn_debug_credential_v1"):
        return False, "needs Saturn job debugger credential v1 agent"
    if needs.get("payload_sandbox_v1") and not caps.get("payload_sandbox_v1"):
        return False, "needs payload sandbox v1 agent"
    if not _fresh(tel, now):
        return False, "telemetry stale (host asleep/offline)"

    # Swap usage is sticky after pressure clears. When a host supplies an explicit
    # pressure level, trust that signal and let reservation/margin math account for
    # current free RAM. Hosts without a pressure signal retain the conservative swap
    # ceiling (macOS thrash measured by the watchdog on 2026-07-15).
    swap_used = float(tel.get("swap_used_mb") or 0.0)
    pressure = tel.get("mem_pressure")
    if pressure is not None and int(pressure) >= MEM_PRESSURE_WARN:
        return False, f"host under memory pressure: level {pressure}"
    swap_over_limit = swap_used > swap_limit_for(host)
    swap_is_critical = swap_signal_is_critical(
        system=str(host.get("os") or "unknown"),
        available_mb=float(tel.get("ram_free_mb") or 0.0),
        total_mb=float(host.get("ram_total_mb") or 0.0),
    )
    if pressure is None and swap_over_limit and swap_is_critical:
        return False, f"host under memory pressure: swap {swap_used:.0f}MB in use"

    ram_total = float(host.get("ram_total_mb") or 0.0)
    margin = margin_for(host)
    # Commit each active job at its KILL CEILING (max of x1.1 and +1GB absolute), not its
    # reservation: N jobs each legally growing to their ceiling must still fit what
    # admission budgeted.
    committed = sum(kill_ceiling_mb(_committed_reservation(j).ram_mb) for j in active_jobs)
    # baseline: uncommitted consumption = (total - free) - what active, committed jobs
    # measure now. Agent telemetry may still contain a process whose scheduler lease expired.
    # Such an orphan has no reservation in ``active_jobs``; subtracting its RSS here would
    # make the process disappear from both baseline and committed accounting and could admit
    # a conflicting job. Only fold telemetry rows that still have an active scheduler record.
    active_job_ids = {str(j["job_id"]) for j in active_jobs if j.get("job_id") is not None}
    running_rss = sum(
        float(r.get("tree_rss_mb") or 0.0)
        for r in (tel.get("running") or [])
        if str(r.get("job_id")) in active_job_ids
    )
    # External (locally-guarded) jobs don't appear in agent telemetry; their heartbeat RSS
    # is folded in so their live usage isn't double-counted against the baseline.
    running_rss += sum(
        float(j.get("external_rss_mb") or 0.0)
        for j in active_jobs
        if j.get("payload_kind") == "external"
    )
    baseline = max(0.0, (ram_total - float(tel.get("ram_free_mb") or 0.0)) - running_rss)
    avail = ram_total - baseline - committed - margin
    if kill_ceiling_mb(res.ram_mb) > avail:
        return False, (
            f"ram: need {kill_ceiling_mb(res.ram_mb):.0f}MB (ceiling) > avail {avail:.0f}MB "
            f"(total {ram_total:.0f} - baseline {baseline:.0f} - committed {committed:.0f} "
            f"- margin {margin:.0f})"
        )

    if res.vram_mb > 0:
        vram_ceiling = res.vram_mb * RAM_KILL_FACTOR
        configured_ceiling = _configured_vram_ceiling_mb(job)
        if configured_ceiling is not None and vram_ceiling > configured_ceiling:
            return False, (
                f"vram: job ceiling {vram_ceiling:.0f}MB > configured max "
                f"{configured_ceiling:.0f}MB"
            )
        vram_total = float(host.get("vram_total_mb") or 0.0)
        vram_committed = sum(
            _committed_reservation(j).vram_mb * RAM_KILL_FACTOR for j in active_jobs
        )
        vram_free = tel.get("vram_free_mb")
        vram_baseline = 0.0
        if vram_free is not None:
            running_vram = sum(
                float(r.get("vram_mb") or 0.0)
                for r in (tel.get("running") or [])
                if str(r.get("job_id")) in active_job_ids
            )
            running_vram += sum(
                float(j.get("external_vram_mb") or 0.0)
                for j in active_jobs
                if j.get("payload_kind") == "external"
            )
            vram_baseline = max(0.0, (vram_total - float(vram_free or 0.0)) - running_vram)
        vram_margin = vram_margin_for(host)
        vram_avail = vram_total - vram_baseline - vram_committed - vram_margin
        if vram_ceiling > vram_avail:
            return False, (
                f"vram: need {vram_ceiling:.0f}MB (ceiling) > "
                f"avail {vram_avail:.0f}MB (total {vram_total:.0f} - baseline "
                f"{vram_baseline:.0f} - committed {vram_committed:.0f} "
                f"- margin {vram_margin:.0f})"
            )

    cpu_total = int(host.get("cpu_threads") or 0)
    cpu_committed = sum(_committed_reservation(j).cpu_threads for j in active_jobs)
    if res.cpu_threads + cpu_committed > cpu_total:
        return False, f"cpu: need {res.cpu_threads} + committed {cpu_committed} > {cpu_total}"

    # disk: charge the models-mount for a cold model download on top of the job's own
    # disk ask. The agent reports per-mount free space (beast has several drives).
    disk_need = res.disk_gb
    plan = plan_for_host(job, host.get("name") or "")
    if plan and plan.get("weights_gb") and not _is_warm(job, host, warm_kinds):
        disk_need += float(plan["weights_gb"])
    disk_free = _artifact_mount_free_gb(host, plan)
    disk_reserve = 5.0
    disk_available = max(0.0, disk_free - disk_reserve) if disk_free is not None else None
    if disk_available is not None and disk_need > disk_available:
        mount_label = (
            (plan or {}).get("artifact_mount")
            or ((plan or {}).get("artifact_locator") or {}).get("mount")
            or "models mount"
        )
        return False, (
            f"disk: need {disk_need:.1f}GB (incl. cold model) > "
            f"{disk_available:.1f}GB available on {mount_label} "
            f"({disk_free:.1f}GB free - {disk_reserve:.1f}GB reserve)"
        )
    return True, "ok"


def fits_some_host_capacity(
    job: dict[str, Any], hosts: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    """Could ``job`` EVER run on any registered host? Static totals only.

    Deliberately ignores telemetry, staleness and active jobs — an asleep mbp1 must not
    cause a false reject. This is the submit-time preflight that turns a physically
    impossible ask (18GB VRAM on a 16GB card, 2026-07-31) into an error instead of a
    silently-forever-queued job.
    """
    needs = job.get("needs") or {}
    candidates = hosts_satisfying_needs(hosts, needs)
    if not candidates:
        if needs.get("host"):
            return False, [f"pinned host {needs['host']!r} is not registered"]
        wanted = [
            k
            for k in (
                "cuda",
                "mps",
                "payload_custody_v2",
                "saturn_debug_credential_v1",
                "payload_sandbox_v1",
            )
            if needs.get(k)
        ]
        return False, [f"no registered host has caps: {', '.join(wanted) or 'any'}"]
    reasons = []
    for h in candidates:
        name = h.get("name") or "?"
        host_reasons = []
        ram_cap = host_ram_cap_mb(h)
        # A client-computed per-host plan is the exact admission/execution contract
        # for non-declared jobs. The generic history reservation can be deliberately
        # conservative (and may describe a different backend); applying it here
        # rejected a 0.5B CUDA plan with a stale 14GB family-VRAM prior before the
        # already-shipped 2.8GB host plan had a chance to take effect.
        effective = effective_reservation(job, name)
        if kill_ceiling_mb(effective.ram_mb) > ram_cap:
            host_reasons.append(
                f"ram ceiling {kill_ceiling_mb(effective.ram_mb):.0f}MB > "
                f"{ram_cap:.0f}MB (total - margin)"
            )
        if effective.vram_mb > 0:
            vram_ceiling = effective.vram_mb * RAM_KILL_FACTOR
            configured_ceiling = _configured_vram_ceiling_mb(job)
            if configured_ceiling is not None and vram_ceiling > configured_ceiling:
                host_reasons.append(
                    f"vram ceiling {vram_ceiling:.0f}MB > configured max "
                    f"{configured_ceiling:.0f}MB"
                )
            vram_cap = host_vram_cap_mb(h)
            if vram_ceiling > vram_cap:
                host_reasons.append(
                    f"vram ceiling {vram_ceiling:.0f}MB > "
                    f"{vram_cap:.0f}MB (total - margin)"
                )
        if effective.cpu_threads > int(h.get("cpu_threads") or 0):
            host_reasons.append(
                f"cpu {effective.cpu_threads} > {int(h.get('cpu_threads') or 0)} threads"
            )
        if not host_reasons:
            return True, []
        reasons.append(f"{name}: {'; '.join(host_reasons)}")
    return False, reasons


def _is_warm(job: dict[str, Any], host: dict[str, Any], warm_kinds: list[str] | None) -> bool:
    if warm_kinds is None:
        return False
    plan = plan_for_host(job, host.get("name") or "")
    if plan and plan.get("artifact_id"):
        # Once a plan names exact bytes, a different qstore variant is not warm
        # enough. Legacy plans without artifact identity retain kind-level behavior.
        return str(plan["artifact_id"]) in warm_kinds
    if plan and plan.get("backend") in {
        "paged",
        "qwen3-moe-cuda",
        "moe-qstore-cuda",
    }:
        return any(kind == "qstore" or kind.startswith("qstore:") for kind in warm_kinds)
    return bool(warm_kinds)


def _models_mount_free_gb(host: dict[str, Any]) -> float | None:
    """Free space on the mount holding the host's models root (falls back to the legacy
    single disk_free_gb figure)."""
    tel = host.get("telemetry") or {}
    mount = host.get("models_mount")
    for d in tel.get("disks") or []:
        if d.get("mount") == mount:
            return float(d.get("free_gb") or 0.0)
    return float(tel["disk_free_gb"]) if tel.get("disk_free_gb") is not None else None


def _artifact_mount_free_gb(host: dict[str, Any], plan: dict[str, Any] | None) -> float | None:
    """Free space for the selected artifact locator, with legacy fallback.

    Older plans have no locator and retain the models-root behavior. New plans may
    provide ``artifact_mount`` or ``artifact_locator.mount`` without changing any
    existing submission fields.
    """
    plan = plan or {}
    locator = plan.get("artifact_locator") or {}
    mount = locator.get("mount") or plan.get("artifact_mount")
    if not mount:
        return _models_mount_free_gb(host)
    for disk in (host.get("telemetry") or {}).get("disks") or []:
        if disk.get("mount") == mount:
            return float(disk.get("free_gb") or 0.0)
    return None


# How long a queued job waits for its BEST host before any admissible host may take it.
STARVATION_S = 60.0
# A pinned, resource-blocked job gets bounded backfill before its host must drain.
BACKFILL_DRAIN_S = 300.0
# A preemptible resident may set a smaller per-job value.  Keeping the scheduler
# default aligned with the existing drain barrier prevents an equal-priority
# finite job from immediately churning a useful warm service unless its owner
# explicitly opts into a shorter service quantum.
RESIDENT_YIELD_AFTER_S = BACKFILL_DRAIN_S
_YIELD_RESOURCE_PREFIXES = ("ram:", "vram:", "cpu:", "concurrency:")

# Relative throughput priors, paged=1.0. Anchored to measured standalone knees on the
# same (model, T=32) point where those exist. The `ane` label means Core ML here; later
# MLComputePlan evidence placed about 98% of that path on the GPU, so this is a Core ML
# throughput prior, not an independent Neural Engine resource claim.
_BACKEND_SPEED = {
    "multifabric": 4.0,
    "mlx": 5.0,  # 4208/826 tok/s standalone knee; head-to-head 2196.8 vs coreml 1694.5
    "ane": 4.0,  # DEMOTED: slower than mlx and ~98% GPU-placed, not Neural Engine
    "qwen3-moe-cuda": 3.0,
    "moe-qstore-cuda": 3.0,
    "hf": 2.0,
    "paged": 1.0,
}


def _resident_yield_after_s(job: dict[str, Any]) -> float:
    raw = (job.get("config") or {}).get("resident_yield_after_s", RESIDENT_YIELD_AFTER_S)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return RESIDENT_YIELD_AFTER_S
    return max(0.0, value) if math.isfinite(value) else RESIDENT_YIELD_AFTER_S


def _released_resident_commit(job: dict[str, Any]) -> dict[str, Any]:
    """Keep telemetry attribution while simulating a released resident lease.

    Simply removing the active row makes its current process-tree RSS/VRAM look
    like external baseline load, so the candidate still appears blocked.  A zero
    commit bearing the same job id subtracts that live row from the baseline and
    models the state after the cooperative process exits.
    """

    released = dict(job)
    released["admission_reservation"] = {
        "ram_mb": 0.0,
        "vram_mb": 0.0,
        "cpu_threads": 0,
        "disk_gb": 0.0,
    }
    return released


def resident_yield_requests(
    hosts: list[dict[str, Any]],
    queued: list[dict[str, Any]],
    active_by_host: dict[str, list[dict[str, Any]]],
    *,
    warm_lookup=None,
    now: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Plan cooperative resident releases needed by blocked finite jobs.

    This is deliberately a signal, not preemption: mrun never kills the resident.
    The resident's control plane observes the durable request, finishes its current
    safe unit, stops claiming work, and exits normally.  A request is produced only
    when the finite job is not currently admissible anywhere and becomes admissible
    on one host after releasing opted-in resident commits.
    """

    current = time.time() if now is None else float(now)
    requests: dict[str, dict[str, Any]] = {}
    for candidate in queued:
        candidate_id = candidate.get("job_id")
        candidate_config = candidate.get("config") or {}
        # ``resident_worker`` describes the payload's in-process execution model,
        # not necessarily an indefinite service lease.  Finite Saturn programs
        # use it too.  Only a resident that explicitly opted into the revocable
        # service protocol should be excluded as a blocker candidate itself.
        if not candidate_id or (
            candidate_config.get("resident_worker") and candidate_config.get("preemptible_resident")
        ):
            continue

        model = candidate_config.get("model")
        warm_map = warm_artifacts_for_model(model, warm_lookup) if model else {}

        # A runnable job needs no eviction.  This also prevents yielding a
        # resident on one host when another host can take the finite job now.
        runnable = False
        for host in hosts:
            try:
                ok, _ = admissible(
                    host,
                    candidate,
                    active_by_host.get(host["name"], []),
                    now=current,
                    warm_kinds=(warm_map or {}).get(host["name"]),
                )
            except Exception:  # noqa: BLE001 - malformed rows never trigger a yield
                ok = False
            if ok:
                runnable = True
                break
        if runnable:
            continue

        waited_s = max(0.0, current - float(candidate.get("created_ts") or current))
        candidate_priority = int(candidate.get("priority") or 0)
        planned = False
        for host in hosts:
            active = active_by_host.get(host["name"], [])
            if not active:
                continue
            warm_kinds = (warm_map or {}).get(host["name"])
            try:
                ok, blocked_reason = admissible(
                    host,
                    candidate,
                    active,
                    now=current,
                    warm_kinds=warm_kinds,
                )
            except Exception:  # noqa: BLE001
                continue
            if ok or not blocked_reason.startswith(_YIELD_RESOURCE_PREFIXES):
                continue

            eligible: list[dict[str, Any]] = []
            for resident in active:
                config = resident.get("config") or {}
                resident_id = resident.get("job_id")
                if (
                    not resident_id
                    or resident_id in requests
                    or not config.get("resident_worker")
                    or not config.get("preemptible_resident")
                ):
                    continue
                resident_priority = int(resident.get("priority") or 0)
                if candidate_priority <= resident_priority and waited_s < _resident_yield_after_s(
                    resident
                ):
                    continue
                eligible.append(resident)
            if not eligible:
                continue

            # Prefer the lowest-priority resident, then the largest VRAM commit:
            # it minimizes service disruption while usually releasing the card in
            # one safe handoff.  Multiple releases remain supported when required.
            eligible.sort(
                key=lambda job: (
                    int(job.get("priority") or 0),
                    -_committed_reservation(job).vram_mb,
                    str(job.get("job_id") or ""),
                )
            )
            released_ids: set[str] = set()
            for resident in eligible:
                released_ids.add(str(resident["job_id"]))
                simulated = [
                    _released_resident_commit(job)
                    if str(job.get("job_id")) in released_ids
                    else job
                    for job in active
                ]
                try:
                    fits_after_yield, _ = admissible(
                        host,
                        candidate,
                        simulated,
                        now=current,
                        warm_kinds=warm_kinds,
                    )
                except Exception:  # noqa: BLE001
                    fits_after_yield = False
                if not fits_after_yield:
                    continue
                for released_id in sorted(released_ids):
                    released = next(
                        job for job in eligible if str(job.get("job_id")) == released_id
                    )
                    requests[released_id] = {
                        "schema": "mrun.resident-yield-request.v1",
                        "resident_job_id": released_id,
                        "blocker_job_id": str(candidate_id),
                        "blocker_experiment": candidate.get("experiment"),
                        "host": host["name"],
                        "blocked_reason": blocked_reason,
                        "blocker_priority": candidate_priority,
                        "resident_priority": int(released.get("priority") or 0),
                        "waited_s": round(waited_s, 3),
                    }
                planned = True
                break
            if planned:
                break
    return requests


def _host_score(
    job: dict[str, Any],
    host: dict[str, Any],
    active_jobs: list[dict[str, Any]],
    warm_kinds: list[str] | None,
) -> float:
    """Bigger = better place to run this job. Pure arithmetic, unit-testable."""
    plan = plan_for_host(job, host.get("name") or "") or {}
    speed = _BACKEND_SPEED.get(plan.get("backend") or "", 1.5)
    if plan.get("device") == "cuda":
        speed += 2.0  # cuda-hf beats cpu-hf
    warm = 1.0 if _is_warm(job, host, warm_kinds) else 0.0
    eta = host_eta_s(host, active_jobs)
    eta_penalty = min((eta or 0.0) / 600.0, 2.0) if active_jobs else 0.0
    tel = host.get("telemetry") or {}
    total = float(host.get("ram_total_mb") or 1.0)
    headroom = max(0.0, float(tel.get("ram_free_mb") or 0.0)) / total  # 0..1 tiebreak
    config = job.get("config") or {}
    inventory = ExecutableInventory.from_telemetry(tel)
    affinity = inventory.affinity(
        arena_key=config.get("resident_arena_key"),
        template_key=config.get("graph_template_key"),
    )
    executable_affinity = {
        "exact-warm-template": 20.0,
        "warm-arena": 10.0,
        "cold": 0.0,
    }[affinity]
    # Soft preference dominates backend-speed deltas but not admissibility; a
    # preferred-but-full host loses the job through the normal starvation valve.
    prefer = 12.0 if (job.get("needs") or {}).get("prefer_host") == host.get("name") else 0.0
    return speed * 10.0 + warm * 5.0 + executable_affinity + prefer - eta_penalty + headroom


def rank_hosts(
    job: dict[str, Any],
    hosts: list[dict[str, Any]],
    active_by_host: dict[str, list[dict[str, Any]]],
    warm_map: dict[str, list[str]] | None = None,
    *,
    now: float | None = None,
) -> list[str]:
    """Admissible hosts for ``job``, best first."""
    scored = []
    for h in hosts:
        active = active_by_host.get(h["name"], [])
        warm = (warm_map or {}).get(h["name"])
        try:
            ok, _ = admissible(h, job, active, now=now, warm_kinds=warm)
        except Exception:  # noqa: BLE001
            continue
        if ok:
            scored.append((_host_score(job, h, active, warm), h["name"]))
    return [name for _, name in sorted(scored, reverse=True)]


def pick_job_for_host(
    host: dict[str, Any],
    queued: list[dict[str, Any]],
    active_jobs: list[dict[str, Any]],
    *,
    all_hosts: list[dict[str, Any]] | None = None,
    active_by_host: dict[str, list[dict[str, Any]]] | None = None,
    warm_lookup=None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """First queued job this host should run (queue is priority/created ordered).

    With fleet context (``all_hosts``), a job goes to its BEST admissible host: the
    polling host only wins if it ranks first, unless the job has starved past
    ``STARVATION_S`` — then any admissible host takes it (a busy best host must not
    wedge the queue)."""
    now = now or time.time()
    for job in queued:
        model = (job.get("config") or {}).get("model")
        # Guarded jobs may bind a complete content-custody model object rather
        # than the legacy string model id.  Inventory keys are strings; passing
        # the object through wedges the whole lease poll in SQLite before the
        # scheduler can consider any later queue row.
        warm_map = warm_artifacts_for_model(model, warm_lookup) if model else None
        warm_kinds = (warm_map or {}).get(host["name"]) if warm_map else None
        try:
            ok, reason = admissible(
                host,
                job,
                active_jobs,
                warm_kinds=warm_kinds,
            )
        except Exception as exc:  # noqa: BLE001 — one malformed row must not wedge the host
            print(f"admission error on {job.get('job_id')}: {exc}; skipping", flush=True)
            continue
        if not ok:
            needs = job.get("needs") or {}
            waited_s = now - float(job.get("created_ts") or now)
            resource_blocked = reason.startswith(("ram:", "vram:", "cpu:", "concurrency:"))
            if (
                active_jobs
                and needs.get("host") == host["name"]
                and waited_s > BACKFILL_DRAIN_S
                and resource_blocked
            ):
                try:
                    fits_when_drained, _ = admissible(
                        host,
                        job,
                        [],
                        warm_kinds=warm_kinds,
                    )
                except Exception:  # noqa: BLE001 — malformed jobs never create a barrier
                    fits_when_drained = False
                if fits_when_drained:
                    return None
            continue
        if all_hosts and active_by_host is not None:
            starved = (now - float(job.get("created_ts") or now)) > STARVATION_S
            if not starved:
                ranked = rank_hosts(job, all_hosts, active_by_host, warm_map, now=now)
                if ranked and ranked[0] != host["name"]:
                    continue  # its best host is alive and has room — let that one take it
        return job
    return None


def host_eta_s(host: dict[str, Any], active_jobs: list[dict[str, Any]]) -> float | None:
    """max(est_wall_s - elapsed) over running jobs; None when any lacks an estimate."""
    now = time.time()
    etas = []
    for j in active_jobs:
        est = (j.get("reservation") or {}).get("est_wall_s")
        if est is None:
            return None
        started = j.get("started_ts") or now
        etas.append(max(0.0, float(est) - (now - float(started))))
    return max(etas) if etas else 0.0
