"""Reservation sizing — one module owning every path that produces a Reservation.

Admission math (can this job start on this host NOW) stays in ``scheduler``; this
module answers the prior question: how big should the job's reservation be, and
from which evidence (declared ask, exact history, family history, client estimate,
default). ``size_reservation`` is the single entry point used at submit time;
``effective_reservation`` refines per-host at lease time; ``history_admission_reservation``
derives a smaller commit-only figure for over-declared active jobs.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from ..protocol import (
    PROBE_DEFAULT_RAM_MB,
    PROBE_DEFAULT_VRAM_MB,
    RAM_KILL_FACTOR,
    SWAP_ADMISSION_MAX_MB,
    Reservation,
    kill_ceiling_mb,
)

RAM_MARGIN_FRACTION = 0.15
RAM_MARGIN_MIN_MB = 2048.0
VRAM_MARGIN_MB = 1024.0
ADMISSION_HISTORY_SAFETY_FACTOR = 2.0
ADMISSION_LIVE_SAFETY_FACTOR = 2.0
ADMISSION_ROUND_MB = 512.0
ADMISSION_MIN_RAM_MB = 2048.0
ADMISSION_MIN_VRAM_MB = 512.0
HISTORY_RESERVATION_MIN_RAM_MB = 512.0
HISTORY_RESERVATION_MIN_VRAM_MB = 512.0

# Declared-ask hygiene. A declared reservation is capped at the figure MEASURED history
# would grant this job (see the history clamp below), not left as-is until it exceeds a
# loose 3x-family-p95 tripwire. The old tripwire let the common 1.5-3x over-ask through
# (median reserved:peak 2.4x RAM / 1.5x VRAM over the last 200 succeeded jobs, measured
# 2026-09-23; 48GB asks vs 3.4GB RSS were the extreme). The clamp is floored so it never
# drops below the worst OBSERVED peak's kill line, and an ask whose kill ceiling sits
# under the measured exact-history peak is RAISED instead of letting the agent kill it
# just short of its known peak. Callers who know a spike is coming pin the ask (below).
DECLARED_RAISE_PEAK_TOLERANCE = 1.05
DECLARED_RAISE_TARGET = 1.2

# History-clamp knobs. Trustworthy history == n>=3 (matches db.history_stats). The clamp
# ceiling reuses the same measured-history grant factors the ladder itself applies, so a
# clamped declared ask lands exactly where a history-sized reservation would: exact-config
# p95 x EXACT_P95_FACTOR, else family p95 x FAMILY_P95_FACTOR, but never under the worst
# observed peak x RAM_KILL_FACTOR. Opt-out marker bypasses clamp AND raise.
HISTORY_CLAMP_MIN_N = 3
DECLARED_PINNED_SOURCE = "declared-pinned"
VRAM_CLAMP_TRIGGER = 3.0
VRAM_CLAMP_TARGET = 2.0

# Measured-history sizing factors. Exact history with n>=3 sizes off a stable p95;
# below that the max peak gets a fatter factor because 1s-sampled peaks read LOW
# (measured 976 vs 2162MB). Family history is a coarser prior -> widest factor.
EXACT_P95_FACTOR = 1.25
EXACT_PEAK_FACTOR = 1.3
FAMILY_P95_FACTOR = 1.5


# The "never crash a compute host" invariant: no operator setting may shrink the RAM
# margin below this — the OS always keeps at least 1GB nothing can reserve.
RAM_MARGIN_FLOOR_MB = 1024.0


def margin_for(host: dict[str, Any]) -> float:
    """RAM admission margin for a host.

    Precedence: DB-persisted operator limits (``host["limits"]``, editable live from
    the dashboard) > ``MRUN_HOST_MARGINS`` env JSON (deprecated, needs a container
    restart) > default ``max(0.15 x total, 2048MB)``. Never below RAM_MARGIN_FLOOR_MB.
    """
    ram_total = float(host.get("ram_total_mb") or 0.0)
    limits = host.get("limits") or {}
    limit_mb = limits.get("ram_margin_mb")
    limit_fraction = limits.get("ram_margin_fraction")
    if limit_mb is not None or limit_fraction is not None:
        if limit_fraction is None:
            return max(float(limit_mb), RAM_MARGIN_FLOOR_MB)
        return max(
            float(limit_fraction) * ram_total,
            float(limit_mb or 0.0),
            RAM_MARGIN_FLOOR_MB,
        )
    fraction, min_mb = RAM_MARGIN_FRACTION, RAM_MARGIN_MIN_MB
    raw = os.environ.get("MRUN_HOST_MARGINS")
    if raw:
        try:
            override = json.loads(raw).get(host.get("name") or "") or {}
            fraction = float(override.get("fraction", fraction))
            min_mb = float(override.get("min_mb", min_mb))
        except (ValueError, TypeError, AttributeError):
            pass
    return max(fraction * ram_total, min_mb, RAM_MARGIN_FLOOR_MB)


def vram_margin_for(host: dict[str, Any]) -> float:
    limits = host.get("limits") or {}
    if limits.get("vram_margin_mb") is not None:
        return max(0.0, float(limits["vram_margin_mb"]))
    return VRAM_MARGIN_MB


def swap_limit_for(host: dict[str, Any]) -> float:
    """Swap admission ceiling with optional per-host JSON overrides."""
    limit = float(os.environ.get("MRUN_SWAP_MAX_MB") or SWAP_ADMISSION_MAX_MB)
    raw = os.environ.get("MRUN_HOST_SWAP_MAX_MB")
    if raw:
        try:
            value = json.loads(raw).get(host.get("name") or "")
            if value is not None:
                limit = float(value)
        except (ValueError, TypeError, AttributeError):
            pass
    return limit


def hosts_satisfying_needs(
    hosts: list[dict[str, Any]], needs: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Hosts a job could ever run on given its pin and capability needs."""
    needs = needs or {}
    out = []
    for h in hosts:
        if needs.get("host") and needs["host"] != h.get("name"):
            continue
        caps = h.get("caps") or {}
        if needs.get("cuda") and not caps.get("cuda"):
            continue
        if needs.get("mps") and not caps.get("mps"):
            continue
        if needs.get("payload_custody_v2") and not caps.get("payload_custody_v2"):
            continue
        if needs.get("saturn_debug_credential_v1") and not caps.get(
            "saturn_debug_credential_v1"
        ):
            continue
        if needs.get("payload_sandbox_v1") and not caps.get("payload_sandbox_v1"):
            continue
        out.append(h)
    return out


def host_ram_cap_mb(host: dict[str, Any]) -> float:
    """Largest RAM kill-ceiling this host could ever admit (empty host, static totals)."""
    return float(host.get("ram_total_mb") or 0.0) - margin_for(host)


def host_vram_cap_mb(host: dict[str, Any]) -> float:
    total = float(host.get("vram_total_mb") or 0.0)
    return max(0.0, total - vram_margin_for(host)) if total else 0.0


@dataclass
class SizedReservation:
    """Result of ``size_reservation``: the reservation plus how we got there."""

    reservation: Reservation
    warnings: list[str] = field(default_factory=list)
    clamps: list[dict[str, Any]] = field(default_factory=list)


def size_reservation(
    declared: dict[str, Any] | None,
    *,
    exact_history: dict[str, Any] | None = None,
    family_history: dict[str, Any] | None = None,
    client_estimate: dict[str, Any] | None = None,
    needs_cuda: bool = False,
    hosts: list[dict[str, Any]] | None = None,
    needs: dict[str, Any] | None = None,
) -> SizedReservation:
    """Size a job's reservation from the best available evidence.

    Wraps the ladder in ``resolve_reservation``, then sanity-checks a *declared* winner
    against measurement: it is CLAMPED down to the figure measured history would grant
    (exact-config p95, else family p95 x its ladder factor, never below the worst
    observed peak's kill line), and RAISED up when its kill ceiling sits under the
    measured exact-history peak (the agent would otherwise kill a correct run just short
    of its known peak). A caller that pins the ask (``source == 'declared-pinned'`` or a
    ``pin``/``no_clamp`` flag) bypasses both.
    """
    reservation = resolve_reservation(
        declared,
        exact_history,
        client_estimate,
        needs_cuda=needs_cuda,
        generalized=family_history,
    )
    sized = SizedReservation(reservation=reservation)
    if reservation.source != "declared":
        return sized

    if _is_pinned(declared):
        # Explicit opt-out: a caller that knows this run will spike above its measured
        # history pins the ask. Pin bypasses BOTH the clamp and the raise; warn when the
        # pin sits under a measured peak so a self-inflicted kill isn't silent.
        reservation.source = DECLARED_PINNED_SOURCE
        _warn_pin_below_measured_peak(sized, reservation, exact_history)
        return sized

    est = client_estimate or {}
    # History clamp: cap an over-declared ask at the figure measured history would grant
    # this job, floored by the client first-principles estimate and never below the worst
    # observed peak's kill line, so no run already seen for this key would be killed.
    ram_ceiling = _history_clamp_ceiling(
        exact_history, family_history, ram=True,
        floor=max(float(est.get("ram_mb") or 0.0), ADMISSION_MIN_RAM_MB),
    )
    if ram_ceiling and ram_ceiling < reservation.ram_mb:
        _record_clamp(sized, reservation, "ram_mb", ram_ceiling)
        reservation.ram_mb = ram_ceiling
        reservation.source = "declared-clamped"

    # VRAM keeps the original 3x-family-p95 tripwire. Replaying 507 jobs showed the tighter
    # history clamp leaves the VRAM reserved:peak ratio unchanged (1.51x median, last 200)
    # while adding one kill (a 2.4x spike over every prior peak), so only RAM is tightened.
    family_n = int((family_history or {}).get("n") or 0)
    family_p95_vram = float((family_history or {}).get("p95_vram_mb") or 0.0)
    if family_n >= HISTORY_CLAMP_MIN_N and family_p95_vram > 0 and reservation.vram_mb > 0 and (
        reservation.vram_mb > family_p95_vram * VRAM_CLAMP_TRIGGER
    ):
        clamped_vram = max(
            _round_up_mb(family_p95_vram * VRAM_CLAMP_TARGET),
            float(est.get("vram_mb") or 0.0),
            ADMISSION_MIN_VRAM_MB,
        )
        if clamped_vram < reservation.vram_mb:
            _record_clamp(sized, reservation, "vram_mb", clamped_vram)
            reservation.vram_mb = clamped_vram
            reservation.source = "declared-clamped"

    exact_peak = float((exact_history or {}).get("ram_peak_mb") or 0.0)
    if exact_peak > 0 and kill_ceiling_mb(reservation.ram_mb) < (
        exact_peak * DECLARED_RAISE_PEAK_TOLERANCE
    ):
        raised = _round_up_mb(exact_peak * DECLARED_RAISE_TARGET)
        if raised > reservation.ram_mb:
            sized.clamps.append({
                "field": "ram_mb",
                "from": reservation.ram_mb,
                "to": raised,
                "why": (
                    f"declared kill ceiling {kill_ceiling_mb(reservation.ram_mb):.0f}MB "
                    f"< measured peak {exact_peak:.0f}MB; raised to peak x "
                    f"{DECLARED_RAISE_TARGET}"
                ),
            })
            reservation.ram_mb = raised
            reservation.source = "declared-raised"

    exact_vram_peak = float((exact_history or {}).get("vram_peak_mb") or 0.0)
    if exact_vram_peak > 0 and kill_ceiling_mb(reservation.vram_mb) < (
        exact_vram_peak * DECLARED_RAISE_PEAK_TOLERANCE
    ):
        raised_vram = _round_up_mb(exact_vram_peak * DECLARED_RAISE_TARGET)
        if raised_vram > reservation.vram_mb:
            sized.clamps.append({
                "field": "vram_mb",
                "from": reservation.vram_mb,
                "to": raised_vram,
                "why": (
                    f"declared VRAM kill ceiling {kill_ceiling_mb(reservation.vram_mb):.0f}MB "
                    f"< measured peak {exact_vram_peak:.0f}MB; raised to peak x "
                    f"{DECLARED_RAISE_TARGET}"
                ),
            })
            reservation.vram_mb = raised_vram
            reservation.source = "declared-raised"

    for c in sized.clamps:
        sized.warnings.append(
            f"reservation {c['field']} {c['from']:.0f} -> {c['to']:.0f}MB: {c['why']}"
        )
    return sized


def resolve_reservation(
    declared: dict[str, Any] | None,
    history: dict[str, Any] | None,
    client_estimate: dict[str, Any] | None,
    *,
    needs_cuda: bool = False,
    generalized: dict[str, Any] | None = None,
) -> Reservation:
    """Reservation ladder: declared > exact-config history > generalized history
    (p95 over the model+task family, n>=3) > client first-principles estimate >
    conservative default. The winning source is recorded on the reservation."""
    if declared and declared.get("ram_mb"):
        r = Reservation.from_dict(declared)
        r.source = "declared"
        return r
    if history and history.get("ram_peak_mb"):
        # The observed peak is 1s-sampled and mmap/page-cache dependent — it can read LOW
        # (measured: qwen fp32 976MB sampled vs 2162MB on rerun). The first-principles
        # client estimate is a floor history may refine upward but never undercut.
        # n-aware: >=3 observations size off a stable p95, fewer off the max peak with
        # a fatter factor (``db.exact_stats`` supplies the matching basis).
        exact_factor = (
            EXACT_P95_FACTOR if int(history.get("n") or 1) >= 3 else EXACT_PEAK_FACTOR
        )
        est = client_estimate or {}
        est_vram = float(est.get("vram_mb") or 0.0)
        history_vram = float(history.get("vram_peak_mb") or 0.0) if needs_cuda or est_vram else 0.0
        return Reservation(
            ram_mb=_round_up_mb(max(
                float(history["ram_peak_mb"]) * exact_factor,
                float(est.get("ram_mb") or 0.0),
                HISTORY_RESERVATION_MIN_RAM_MB,
            )),
            vram_mb=_round_up_mb(max(
                history_vram * exact_factor,
                est_vram,
                HISTORY_RESERVATION_MIN_VRAM_MB if history_vram or est_vram else 0.0,
            )),
            cpu_threads=int((declared or {}).get("cpu_threads", 4)),
            est_wall_s=float(history["wall_s"]) if history.get("wall_s") else None,
            source="history",
        )
    if generalized and generalized.get("p95_ram_mb"):
        est = client_estimate or {}
        est_vram = float(est.get("vram_mb") or 0.0)
        history_vram = (
            float(generalized.get("p95_vram_mb") or 0.0) if needs_cuda or est_vram else 0.0
        )
        return Reservation(
            ram_mb=_round_up_mb(max(
                float(generalized["p95_ram_mb"]) * FAMILY_P95_FACTOR,
                float(est.get("ram_mb") or 0.0),
                HISTORY_RESERVATION_MIN_RAM_MB,
            )),
            vram_mb=_round_up_mb(max(
                history_vram * FAMILY_P95_FACTOR,
                est_vram,
                HISTORY_RESERVATION_MIN_VRAM_MB if history_vram or est_vram else 0.0,
            )),
            cpu_threads=int(est.get("cpu_threads") or 4),
            est_wall_s=generalized.get("p50_wall_s"),
            source="generalized-history",
        )
    if client_estimate and client_estimate.get("ram_mb"):
        r = Reservation.from_dict(client_estimate)
        r.source = "estimated"
        return r
    # Unknown workload: a small PROBE, not a conservative block. Kill ceiling gives it
    # +1GB grace, and a ceiling kill auto-retries with a reservation grown from the
    # measured peak — so first runs are cheap to admit instead of queue-blocking.
    return Reservation(
        ram_mb=PROBE_DEFAULT_RAM_MB,
        vram_mb=PROBE_DEFAULT_VRAM_MB if needs_cuda else 0.0,
        source="probe-default",
    )


def plan_for_host(job: dict[str, Any], host_name: str) -> dict[str, Any] | None:
    """The client-computed RunPlan for this job on this host, if one was shipped."""
    return (job.get("plans") or {}).get(host_name)


def effective_reservation(job: dict[str, Any], host_name: str) -> Reservation:
    """Reservation to admit/enforce on this host: a declared reservation always wins;
    otherwise the per-host plan (computed against the host's REAL caps) overrides the
    generic reservation's sizing. A server-clamped declared ask may still be refined
    DOWNWARD by a plan, never back up past the clamp."""
    res = Reservation.from_dict(job.get("reservation"))
    if res.source in ("declared", "declared-raised"):
        return res
    plan = plan_for_host(job, host_name)
    if res.source == "declared-clamped":
        if plan and float(plan.get("ram_limit_mb") or 0.0) and (
            float(plan["ram_limit_mb"]) < res.ram_mb
        ):
            res.ram_mb = float(plan["ram_limit_mb"])
            res.vram_mb = min(res.vram_mb, float(plan.get("est_vram_mb") or res.vram_mb))
            res.cpu_threads = int(plan.get("threads") or res.cpu_threads)
            res.source = "plan"
        return res
    if plan:
        res.ram_mb = float(plan.get("ram_limit_mb") or res.ram_mb)
        res.vram_mb = float(plan.get("est_vram_mb") or 0.0)
        res.cpu_threads = int(plan.get("threads") or res.cpu_threads)
        res.source = "plan"
    return res


def history_admission_reservation(
    job: dict[str, Any],
    history: dict[str, Any] | None,
    running: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Smaller commit-only reservation for over-declared active jobs.

    This is intentionally not the execution reservation. The agent keeps enforcing the
    reservation it was granted; admission may use this derived number to avoid one
    stale declared estimate wedging the queue after repeated successful measurements.
    """
    if not history or int(history.get("n") or 0) < 3 or not history.get("p95_ram_mb"):
        return None
    res = Reservation.from_dict(job.get("reservation"))
    if res.source not in ("declared", "declared-clamped", "declared-raised"):
        return None

    running = running or {}
    ram_mb = max(
        _round_up_mb(float(history["p95_ram_mb"]) * ADMISSION_HISTORY_SAFETY_FACTOR),
        _round_up_mb(float(running.get("tree_rss_mb") or 0.0) * ADMISSION_LIVE_SAFETY_FACTOR),
        ADMISSION_MIN_RAM_MB,
    )
    vram_mb = res.vram_mb
    if res.vram_mb > 0 and history.get("p95_vram_mb") is not None:
        vram_mb = max(
            _round_up_mb(
                float(history.get("p95_vram_mb") or 0.0) * ADMISSION_HISTORY_SAFETY_FACTOR
            ),
            _round_up_mb(float(running.get("vram_mb") or 0.0) * ADMISSION_LIVE_SAFETY_FACTOR),
            ADMISSION_MIN_VRAM_MB,
        )

    if ram_mb >= res.ram_mb and vram_mb >= res.vram_mb:
        return None
    right_sized = Reservation(
        ram_mb=min(res.ram_mb, ram_mb),
        vram_mb=min(res.vram_mb, vram_mb),
        cpu_threads=res.cpu_threads,
        disk_gb=res.disk_gb,
        est_wall_s=res.est_wall_s,
        source="history-admission",
    )
    return right_sized.as_dict()


def _round_up_mb(value: float, unit: float = ADMISSION_ROUND_MB) -> float:
    if value <= 0:
        return 0.0
    return float(math.ceil(value / unit) * unit)


def _is_pinned(declared: dict[str, Any] | None) -> bool:
    """Whether the caller opted this declared ask out of the history clamp/raise."""
    d = declared or {}
    return bool(
        d.get("source") == DECLARED_PINNED_SOURCE
        or d.get("pin")
        or d.get("no_clamp")
    )


def _clamp_basis(
    exact_history: dict[str, Any] | None,
    family_history: dict[str, Any] | None,
    *,
    ram: bool,
) -> tuple[float, float, float] | None:
    """(quantile_peak, observed_max_peak, factor) for the history clamp, preferring
    exact-config history (n>=3) over the coarser model/script family (n>=3). ``None``
    when no trustworthy history exists for this resource.

    ``observed_max_peak`` falls back to the quantile when the stats row predates the
    ``max_*`` fields (older DBs) — never below the quantile, so the floor stays sane.
    """
    q_exact = "ram_peak_mb" if ram else "vram_peak_mb"
    q_family = "p95_ram_mb" if ram else "p95_vram_mb"
    max_key = "max_ram_mb" if ram else "max_vram_mb"
    e = exact_history or {}
    if int(e.get("n") or 0) >= HISTORY_CLAMP_MIN_N and e.get(q_exact):
        q = float(e[q_exact])
        return q, max(float(e.get(max_key) or q), q), EXACT_P95_FACTOR
    f = family_history or {}
    if int(f.get("n") or 0) >= HISTORY_CLAMP_MIN_N and f.get(q_family):
        q = float(f[q_family])
        return q, max(float(f.get(max_key) or q), q), FAMILY_P95_FACTOR
    return None


def _history_clamp_ceiling(
    exact_history: dict[str, Any] | None,
    family_history: dict[str, Any] | None,
    *,
    ram: bool,
    floor: float,
) -> float:
    """Ceiling to clamp an over-declared ask to: the measured-history grant
    (quantile x ladder factor), floored by the client estimate, and NEVER below the
    worst observed peak's kill line (x RAM_KILL_FACTOR). 0.0 when no trustworthy history
    exists (nothing to clamp against)."""
    basis = _clamp_basis(exact_history, family_history, ram=ram)
    if basis is None:
        return 0.0
    quantile, observed_max, factor = basis
    return _round_up_mb(max(
        quantile * factor,
        observed_max * RAM_KILL_FACTOR,
        floor,
    ))


def _record_clamp(
    sized: SizedReservation, reservation: Reservation, field: str, to_val: float
) -> None:
    """Record a clamp decision on the reservation the same way sources are recorded."""
    frm = float(getattr(reservation, field))
    label = "VRAM" if field == "vram_mb" else "RAM"
    sized.clamps.append({
        "field": field,
        "from": frm,
        "to": to_val,
        "why": (
            f"declared {frm:.0f}MB {label} > measured-history grant {to_val:.0f}MB; "
            f"clamped (never below worst observed peak x {RAM_KILL_FACTOR})"
        ),
    })


def _warn_pin_below_measured_peak(
    sized: SizedReservation, reservation: Reservation, exact_history: dict[str, Any] | None
) -> None:
    peak = float((exact_history or {}).get("ram_peak_mb") or 0.0)
    if peak and kill_ceiling_mb(reservation.ram_mb) < peak:
        sized.warnings.append(
            f"pinned reservation kill ceiling {kill_ceiling_mb(reservation.ram_mb):.0f}MB "
            f"< measured peak {peak:.0f}MB: pin bypasses clamp AND raise; may be killed"
        )
