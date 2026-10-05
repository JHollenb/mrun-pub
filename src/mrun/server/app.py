"""mrun scheduler — FastAPI app (docker on zima, port 9025).

Run: ``uvicorn mrun.server.app:app --host 0.0.0.0 --port 9025``
(or ``python -m mrun.server``). Auth: shared token in ``X-Mrun-Token`` when
``MRUN_TOKEN`` is required by default; local development may explicitly opt out.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import threading
import time
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request, Response

from .. import __version__
from ..diagnostics import normalize_external_result, redact_command
from ..ids import new_job_id
from ..protocol import (
    ACTIVE_STATES,
    AWAITING_PAYLOAD,
    KILL_CEILING_ABS_MB,
    KILLED_RAM,
    KILLED_VRAM,
    LEASE_EXPIRY_S,
    LEASE_POLL_HOLD_S,
    PROTOCOL_VERSION,
    QUEUED,
    RAM_KILL_FACTOR,
    TELEMETRY_INTERVAL_S,
    TERMINAL_STATES,
    Reservation,
    family_key_for,
)
from .db import DB, GuardedAdmissionError, ProtectedScopeError, data_dir
from .reservation import _round_up_mb, host_ram_cap_mb, host_vram_cap_mb
from .scheduler import (
    admissible,
    effective_reservation,
    fits_some_host_capacity,
    history_admission_reservation,
    host_eta_s,
    hosts_satisfying_needs,
    pick_job_for_host,
    plan_for_host,
    resident_yield_requests,
    size_reservation,
    warm_artifacts_for_model,
)

app = FastAPI(title="mrun scheduler", version=__version__)
db = DB()

# Minimal ops UI (hosts / queue / log tail) — same JSON API, one static file, LAN-only.
_STATIC = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC):
    from fastapi.responses import FileResponse

    @app.get("/ui")
    def ui() -> Any:
        return FileResponse(os.path.join(_STATIC, "index.html"))


_wake = threading.Condition()  # notified on submit/finish -> lease long-polls re-check

# Saturn's debugger broker is a small control-plane mailbox. The worker remains
# the authority over tensors and StateCuts; mrun only routes authenticated
# requests while the worker is alive.
_saturn_debug_lock = threading.Condition()
_saturn_debug_channels: dict[str, dict[str, Any]] = {}
_SATURN_DEBUG_STALE_S = 30.0
_SATURN_DEBUG_RETENTION_S = 300.0
_SATURN_DEBUG_SCOPES = (
    "debug:poll",
    "debug:register",
    "debug:respond",
)
_SATURN_DEBUG_WORKER_ROUTE = re.compile(r"^/api/jobs/[^/]+/debug(?:/(?:register|respond))?$")


def _wake_all() -> None:
    with _wake:
        _wake.notify_all()


_CONTROL_SETTING = "control"


def _control_record() -> dict[str, Any]:
    raw = db.get_setting(_CONTROL_SETTING, {})
    return raw if isinstance(raw, dict) else {}


def _draining() -> bool:
    return bool(_control_record().get("draining"))


def _active_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for state in sorted(ACTIVE_STATES):
        jobs.extend(db.jobs(state=state))
    return jobs


def _control_state() -> dict[str, Any]:
    record = _control_record()
    draining = bool(record.get("draining"))
    active = _active_jobs()
    queued = db.jobs(state=QUEUED)
    return {
        "draining": draining,
        "safe_to_shutdown": draining and not active,
        "active": len(active),
        "queued": len(queued),
        "active_jobs": [
            {
                "job_id": j["job_id"],
                "experiment": j.get("experiment"),
                "state": j.get("state"),
                "host": j.get("assigned_host"),
            }
            for j in active
        ],
        "reason": record.get("reason"),
        "updated_ts": record.get("updated_ts"),
    }


def _active_jobs_for_admission(
    host_row: dict[str, Any], active_jobs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Annotate active rows with history-sized commit reservations when available.

    The DB reservation remains the agent-enforced ceiling. This derived field only
    affects admission math for other jobs competing for the same host.
    """
    telemetry = host_row.get("telemetry") or {}
    running_by_job = {
        str(r.get("job_id")): r for r in telemetry.get("running") or [] if r.get("job_id")
    }
    out: list[dict[str, Any]] = []
    for job in active_jobs:
        history = db.history_stats(
            family_key_for(str(job.get("experiment") or ""), job.get("cmd"), job.get("config"))
        )
        admission = history_admission_reservation(
            job,
            history,
            running_by_job.get(str(job.get("job_id"))),
        )
        if admission:
            job = dict(job)
            job["admission_reservation"] = admission
        out.append(job)
    return out


def _active_by_host_for_admission(
    all_hosts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    raw_active_by_host = {x["name"]: db.active_jobs_on(x["name"]) for x in all_hosts}
    return {
        x["name"]: _active_jobs_for_admission(x, raw_active_by_host.get(x["name"], []))
        for x in all_hosts
    }


_ADMISSION_PREFIXES = ("waiting:", "unschedulable:")


def _short_detail(prefix: str, reasons: list[str]) -> str:
    detail = prefix + " " + "; ".join(reasons[:6])
    if len(reasons) > 6:
        detail += f"; +{len(reasons) - 6} more"
    return detail[:1000]


def _admission_status(
    job: dict[str, Any],
    all_hosts: list[dict[str, Any]],
    active_by_host: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, str, dict[str, Any]]:
    """Explain queued-job admission without rejecting the job.

    ``None`` means it can start on at least one host right now. ``waiting`` means a
    registered host could run it after active reservations or current host pressure
    clears. ``unschedulable`` means no currently registered host can satisfy the static
    reservation contract, even before live telemetry and active allocations are counted.
    """
    needs = job.get("needs") or {}
    host_pin = needs.get("host")
    if host_pin:
        hosts = [h for h in all_hosts if h["name"] == host_pin]
        if not hosts:
            detail = f"unschedulable: pinned host {host_pin!r} is not registered"
            return detail, "unschedulable", {"host_pin": host_pin, "current_reasons": []}
    else:
        hosts = all_hosts
    if not hosts:
        return "waiting: no registered hosts", "waiting", {"current_reasons": []}

    model = (job.get("config") or {}).get("model")
    warm_map = warm_artifacts_for_model(model, db.hosts_with_artifacts) if model else {}
    current_reasons: list[str] = []
    empty_reasons: list[str] = []
    fits_when_empty = False
    for h in hosts:
        warm = warm_map.get(h["name"])
        try:
            ok, reason = admissible(
                h,
                job,
                active_by_host.get(h["name"], []),
                warm_kinds=warm,
            )
        except Exception as exc:  # noqa: BLE001
            ok, reason = False, f"admission error: {exc}"
        if ok:
            return None, "admissible", {"current_reasons": current_reasons}
        current_reasons.append(f"{h['name']}: {reason}")

        try:
            empty_ok, empty_reason = admissible(h, job, [], warm_kinds=warm)
        except Exception as exc:  # noqa: BLE001
            empty_ok, empty_reason = False, f"admission error: {exc}"
        if empty_ok:
            fits_when_empty = True
        else:
            empty_reasons.append(f"{h['name']}: {empty_reason}")

    if fits_when_empty:
        return (
            _short_detail("waiting:", current_reasons),
            "waiting",
            {"current_reasons": current_reasons, "empty_reasons": empty_reasons},
        )
    # ``admissible(..., active_jobs=[])`` still subtracts the host's live external
    # baseline. That is useful for launch decisions, but it is not a permanent
    # capacity verdict: a process can exit, a driver can release VRAM, or a stale
    # telemetry sample can be replaced. Use the static preflight here so a job that
    # fits an empty physical host is reported as waiting rather than unschedulable.
    static_ok, _static_reasons = fits_some_host_capacity(job, all_hosts)
    if static_ok:
        return (
            _short_detail("waiting:", current_reasons),
            "waiting",
            {
                "current_reasons": current_reasons,
                "empty_reasons": empty_reasons,
                "static_capacity": "fits",
            },
        )
    return (
        _short_detail("unschedulable:", empty_reasons or current_reasons),
        "unschedulable",
        {"current_reasons": current_reasons, "empty_reasons": empty_reasons},
    )


def _refresh_queue_admission_details(
    queued: list[dict[str, Any]] | None = None,
    *,
    all_hosts: list[dict[str, Any]] | None = None,
    active_by_host: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    queued = queued if queued is not None else db.jobs(state=QUEUED)
    all_hosts = all_hosts if all_hosts is not None else db.host_rows()
    active_by_host = (
        active_by_host if active_by_host is not None else (_active_by_host_for_admission(all_hosts))
    )
    out: list[dict[str, Any]] = []
    for job in queued:
        if job.get("state") != QUEUED:
            continue
        detail, reason, payload = _admission_status(job, all_hosts, active_by_host)
        if db.set_status_detail_if_changed(
            job["job_id"],
            detail,
            clear_prefixes=_ADMISSION_PREFIXES,
        ):
            db.add_event(
                "admission.status",
                job_id=job["job_id"],
                state=QUEUED,
                reason=reason,
                detail=detail,
                payload=payload,
            )
            job = dict(job)
            job["status_detail"] = detail
        out.append(job)
    # Yield planning needs the complete queue even when this refresh was called
    # with one newly submitted row.  The signal lives on the active resident's
    # mutable metadata and never changes guarded request identity or kill state.
    _refresh_resident_yield_requests(
        db.jobs(state=QUEUED),
        all_hosts=all_hosts,
        active_by_host=active_by_host,
    )
    return out


_YIELD_REQUEST_KEY = "resident_yield_request"
_YIELD_IDENTITY_FIELDS = (
    "schema",
    "resident_job_id",
    "blocker_job_id",
    "host",
    "blocker_priority",
    "resident_priority",
)


def _same_yield_request(current: Any, desired: dict[str, Any]) -> bool:
    return isinstance(current, dict) and all(
        current.get(field) == desired.get(field) for field in _YIELD_IDENTITY_FIELDS
    )


def _refresh_resident_yield_requests(
    queued: list[dict[str, Any]],
    *,
    all_hosts: list[dict[str, Any]],
    active_by_host: dict[str, list[dict[str, Any]]],
) -> None:
    queued_ids = {str(job["job_id"]) for job in queued if job.get("job_id")}
    desired = resident_yield_requests(
        all_hosts,
        queued,
        active_by_host,
        warm_lookup=db.hosts_with_artifacts,
    )
    active = {
        str(job["job_id"]): job
        for jobs in active_by_host.values()
        for job in jobs
        if job.get("job_id")
        and (job.get("config") or {}).get("resident_worker")
        and (job.get("config") or {}).get("preemptible_resident")
    }
    for job_id, job in active.items():
        current = (job.get("meta") or {}).get(_YIELD_REQUEST_KEY)
        request = desired.get(job_id)
        if request is not None:
            if _same_yield_request(current, request):
                continue
            request = {**request, "requested_ts": time.time()}
            db.merge_meta(job_id, {_YIELD_REQUEST_KEY: request})
            db.add_event(
                "job.resident_yield_requested",
                job_id=job_id,
                host=job.get("assigned_host"),
                state=job.get("state"),
                reason="finite-job-blocked",
                detail=(
                    f"yield at next safe boundary for {request['blocker_job_id']}: "
                    f"{request['blocked_reason']}"
                )[:1000],
                payload=request,
            )
        elif isinstance(current, dict):
            # Once issued, keep the cooperative stop request durable while its
            # blocker is still queued.  A transient stale/missing telemetry
            # sample during a scheduler restart must not withdraw the handoff
            # before the resident control plane observes it.
            if str(current.get("blocker_job_id") or "") in queued_ids:
                continue
            db.merge_meta(job_id, {_YIELD_REQUEST_KEY: None})
            db.add_event(
                "job.resident_yield_cleared",
                job_id=job_id,
                host=job.get("assigned_host"),
                state=job.get("state"),
                reason="blocker-cleared",
                payload={"previous": current},
            )


@app.middleware("http")
async def _auth(request: Request, call_next):
    token = os.environ.get("MRUN_TOKEN")
    if request.url.path.startswith("/api"):
        if not token:
            local_peer = request.client is not None and request.client.host in {
                "127.0.0.1", "::1", "localhost", "testclient",
            }
            if os.environ.get("MRUN_ALLOW_UNAUTHENTICATED") != "1" or not local_peer:
                return Response(status_code=503, content="configure MRUN_TOKEN")
            return await call_next(request)
        scoped_debug_worker = bool(
            request.headers.get("x-mrun-debug-credential")
            and request.headers.get("x-mrun-debug-credential-id")
            and _SATURN_DEBUG_WORKER_ROUTE.fullmatch(request.url.path)
            and (
                request.method == "GET"
                or request.url.path.endswith("/register")
                or request.url.path.endswith("/respond")
            )
        )
        if not scoped_debug_worker and not hmac.compare_digest(
            request.headers.get("x-mrun-token", ""), token
        ):
            return Response(status_code=401, content="bad token")
    return await call_next(request)


def _agent_credential_valid(request: Request) -> bool:
    """A general MRUN_TOKEN is never sufficient for guarded agent authority."""

    expected = os.environ.get("MRUN_AGENT_TOKEN")
    supplied = request.headers.get("x-mrun-agent-token")
    return bool(
        expected
        and supplied
        and hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8"))
    )


_AGENT_AUTH_CAPABILITIES = frozenset(
    {
        "payload_custody_v2",
        "saturn_debug_credential_v1",
        "payload_sandbox_v1",
    }
)


def _agent_auth_capable(payload: dict[str, Any] | None) -> bool:
    caps = (payload or {}).get("caps") or {}
    return any(bool(caps.get(name)) for name in _AGENT_AUTH_CAPABILITIES)


def _registered_host(host: str) -> dict[str, Any] | None:
    return next((row for row in db.host_rows() if row.get("name") == host), None)


def _require_registered_agent_auth(request: Request, host: str) -> None:
    existing = _registered_host(host)
    if existing is not None and _agent_auth_capable(existing):
        if not _agent_credential_valid(request):
            raise HTTPException(403, "guarded-capable host requires the agent credential")


def _guarded_agent_required(job: dict[str, Any]) -> bool:
    """Return whether agent-originated mutations need independent authority.

    Admission, rather than payload kind, is the protected-scope boundary.  The
    custody flag is retained as a defensive fallback for any partially migrated
    row, while legacy unguarded shipped jobs remain ordinary jobs.
    """

    return job.get("admission") is not None or bool(
        (job.get("payload_custody") or {}).get("required")
    )


def _guarded_agent_headers(request: Request, job: dict[str, Any]) -> None:
    """Authorize one guarded, lease-bound agent action or fail closed."""

    if not _guarded_agent_required(job):
        return
    if not (job.get("payload_custody") or {}).get("required"):
        raise HTTPException(409, "guarded job has no lease-capability custody binding")
    if not _agent_credential_valid(request):
        raise HTTPException(403, "guarded action requires the agent credential")
    lease_identity = request.headers.get("x-mrun-lease-id")
    lease_capability = request.headers.get("x-mrun-lease-capability")
    if not lease_identity or not lease_capability:
        raise HTTPException(403, "guarded action requires the lease capability")
    if len(lease_identity) > 200 or len(lease_capability) > 500:
        raise HTTPException(403, "guarded lease authorization is invalid")
    try:
        db.authorize_guarded_agent_lease(
            job["job_id"],
            lease_identity=lease_identity,
            lease_capability=lease_capability,
        )
    except (KeyError, GuardedAdmissionError) as exc:
        raise HTTPException(403, f"guarded lease authorization failed: {exc}") from exc


def _guarded_debug_headers(
    request: Request,
    job: dict[str, Any],
    *,
    scope: str,
) -> str | None:
    """Authorize only the guarded worker side of one debugger mailbox.

    Ordinary jobs continue to use the normal client token.  Supplying a narrow
    credential for an ordinary job is rejected because the auth middleware may
    have admitted the request specifically for route-level verification.  This
    route authorization is not a same-UID process-isolation boundary.
    """

    credential_id = request.headers.get("x-mrun-debug-credential-id")
    credential = request.headers.get("x-mrun-debug-credential")
    if not _guarded_agent_required(job):
        if credential_id or credential:
            raise HTTPException(403, "ordinary job cannot use debugger credential auth")
        return None
    if not credential_id or not credential:
        raise HTTPException(403, "guarded debugger action requires its job credential")
    if len(credential_id) > 200 or len(credential) > 500:
        raise HTTPException(403, "guarded debugger credential is invalid")
    try:
        db.authorize_guarded_debug_credential(
            job["job_id"],
            credential_id=credential_id,
            credential=credential,
            scope=scope,
        )
    except (KeyError, GuardedAdmissionError) as exc:
        if job.get("state") in TERMINAL_STATES:
            with _saturn_debug_lock:
                _saturn_debug_channels.pop(str(job.get("job_id")), None)
                _saturn_debug_lock.notify_all()
        raise HTTPException(403, f"guarded debugger authorization failed: {exc}") from exc
    return credential_id


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.get("/api/version")
def version() -> dict[str, Any]:
    return {"version": __version__, "protocol": PROTOCOL_VERSION}


@app.get("/api/control")
def control() -> dict[str, Any]:
    """Persistent operator controls for drain/safe-shutdown workflows."""
    return _control_state()


@app.post("/api/control")
def set_control(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    draining = payload.get("draining")
    if not isinstance(draining, bool):
        raise HTTPException(422, "draining must be a boolean")
    reason = payload.get("reason") or ("shutdown-drain" if draining else "resume")
    if not isinstance(reason, str):
        raise HTTPException(422, "reason must be a string")
    record = {
        "draining": draining,
        "reason": reason[:200],
        "updated_ts": time.time(),
    }
    db.set_setting(_CONTROL_SETTING, record)
    db.add_event(
        "control.drain" if draining else "control.resume",
        reason=record["reason"],
        payload=record,
    )
    _refresh_queue_admission_details()
    _wake_all()
    return _control_state()


# --------------------------------------------------------------------- agents
@app.post("/api/agents/register")
def register(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if int(payload.get("protocol_version") or 0) != PROTOCOL_VERSION:
        raise HTTPException(409, f"protocol mismatch: server={PROTOCOL_VERSION}")
    host = payload.get("host")
    if not isinstance(host, str) or not host:
        raise HTTPException(422, "host required")
    existing = _registered_host(host)
    if _agent_auth_capable(payload) or _agent_auth_capable(existing):
        if not _agent_credential_valid(request):
            raise HTTPException(
                403, "guarded-capable agent registration requires the agent credential"
            )
    db.upsert_host(host, payload)
    db.add_event("host.register", host=host, payload=payload)
    _refresh_queue_admission_details()
    _wake_all()
    return {
        "telemetry_interval_s": TELEMETRY_INTERVAL_S,
        "lease_hold_s": LEASE_POLL_HOLD_S,
        "protocol_version": PROTOCOL_VERSION,
    }


@app.post("/api/agents/{host}/telemetry")
def telemetry(host: str, request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    _require_registered_agent_auth(request, host)
    # An agent whose host row was deleted (UI remove while it slept) must not write
    # orphan telemetry forever — tell it to re-register. Current agents ignore the
    # extra key (harmless); updated agents call register() on seeing it.
    if not any(h["name"] == host for h in db.host_rows()):
        return {"kill": [], "reregister": True}
    raw_running = payload.get("running") or []
    now = time.time()
    sanitized_running: list[dict[str, Any]] = []
    authorized_guarded_ids: set[str] = set()
    for item in raw_running:
        if not isinstance(item, dict):
            continue
        running = dict(item)
        authorization = running.pop("lease_authorization", None)
        job_id = running.get("job_id")
        existing = db.job(job_id) if isinstance(job_id, str) else None
        if existing is not None and _guarded_agent_required(existing):
            authority = authorization if isinstance(authorization, dict) else {}
            try:
                if existing.get("assigned_host") != host:
                    raise GuardedAdmissionError(
                        "guarded telemetry host does not own the active lease"
                    )
                if not _agent_credential_valid(request):
                    raise GuardedAdmissionError("guarded telemetry requires the agent credential")
                db.authorize_guarded_agent_lease(
                    job_id,
                    lease_identity=str(authority.get("lease_id") or ""),
                    lease_capability=str(authority.get("capability") or ""),
                    now=now,
                )
            except (KeyError, GuardedAdmissionError) as exc:
                # Reject this protected row before it can influence host snapshots,
                # job telemetry events, or lease renewal.  Other ordinary rows and
                # aggregate host telemetry remain useful during rolling upgrades.
                db.add_event(
                    "job.guarded_telemetry_rejected",
                    job_id=job_id,
                    host=host,
                    state=existing.get("state"),
                    reason="invalid-lease-authorization",
                    detail=str(exc),
                )
                continue
            authorized_guarded_ids.add(job_id)
        sanitized_running.append(running)
    sanitized_payload = dict(payload)
    sanitized_payload["running"] = sanitized_running
    db.record_telemetry(host, sanitized_payload)
    db.add_event("host.telemetry", host=host, payload=sanitized_payload)
    for running in sanitized_running:
        if running.get("job_id"):
            db.add_event(
                "job.telemetry",
                job_id=str(running["job_id"]),
                host=host,
                payload=running,
            )
    # telemetry doubles as lease renewal for the jobs it reports
    running_ids = {r.get("job_id") for r in sanitized_running}
    kills = []
    for job in db.active_jobs_on(host):
        if job["job_id"] in running_ids:
            if _guarded_agent_required(job):
                if job["job_id"] in authorized_guarded_ids:
                    db.update_job(job["job_id"], lease_expires_ts=now + LEASE_EXPIRY_S)
            else:
                db.update_job(job["job_id"], lease_expires_ts=now + LEASE_EXPIRY_S)
        if job.get("kill_requested"):
            kills.append(job["job_id"])
    lost = db.expire_leases()
    if lost:
        _requeue_lost(lost)
        _wake_all()
    _refresh_queue_admission_details()
    return {"kill": kills}


@app.post("/api/agents/{host}/lease")
def lease(host: str, request: Request) -> Response:
    """Long-poll for work: hold up to LEASE_POLL_HOLD_S, 204 when nothing fits."""
    deadline = time.time() + LEASE_POLL_HOLD_S
    guarded_agent = _agent_credential_valid(request)
    _require_registered_agent_auth(request, host)
    while True:
        newly_lost = db.expire_leases()
        if newly_lost:
            _requeue_lost(newly_lost)
        if _draining():
            remaining = deadline - time.time()
            if remaining <= 0:
                return Response(status_code=204)
            with _wake:
                _wake.wait(timeout=min(remaining, 2.0))
            continue
        all_hosts = db.host_rows()
        hosts = {h["name"]: h for h in all_hosts}
        h = hosts.get(host)
        if h is not None:
            active_by_host = _active_by_host_for_admission(all_hosts)
            queued = _refresh_queue_admission_details(
                db.jobs(state=QUEUED),
                all_hosts=all_hosts,
                active_by_host=active_by_host,
            )
            # Old agents retain ordinary-job compatibility during a rolling upgrade,
            # but can never receive the one response that discloses guarded authority.
            # A malformed/legacy guarded command row has no sealed-payload binding
            # from which to mint a lease capability, so it is never leasable.
            queued = [
                job
                for job in queued
                if not (
                    job.get("admission") is not None
                    and not (job.get("payload_custody") or {}).get("required")
                )
            ]
            if not guarded_agent:
                queued = [job for job in queued if not _guarded_agent_required(job)]
            job = pick_job_for_host(
                h,
                queued,
                active_by_host.get(host, []),
                all_hosts=all_hosts,
                active_by_host=active_by_host,
                warm_lookup=db.hosts_with_artifacts,
            )
            if job is not None:
                if _draining():
                    continue
                # Stamp the chosen per-host plan and its effective reservation so agent
                # enforcement, committed math and calibration all use ONE sizing.
                plan = plan_for_host(job, host)
                if job.get("admission") is not None:
                    submitted_plans = job.get("plans")
                    submitted_plan = (
                        submitted_plans.get(host) if isinstance(submitted_plans, dict) else None
                    )
                    if plan != submitted_plan:
                        db.add_event(
                            "job.plan_binding_refused",
                            job_id=job["job_id"],
                            host=host,
                            state=QUEUED,
                            reason="selected-plan-not-submitted",
                        )
                        continue
                res = effective_reservation(job, host)
                guarded = _guarded_agent_required(job)
                lease_identity = secrets.token_urlsafe(24) if guarded else None
                lease_capability = secrets.token_urlsafe(32) if guarded else None
                debug_requested = bool(
                    guarded and (job.get("needs") or {}).get("saturn_debug_credential_v1")
                )
                debug_credential_id = f"dbg-{secrets.token_hex(12)}" if debug_requested else None
                debug_credential = secrets.token_urlsafe(32) if debug_requested else None
                lease_expires_ts = time.time() + LEASE_EXPIRY_S
                claimed = db.claim_job(
                    job["job_id"],
                    host=host,
                    lease_expires_ts=lease_expires_ts,
                    reservation=res.as_dict(),
                    plan=plan,
                    lease_identity=lease_identity,
                    lease_capability=lease_capability,
                    debug_credential_id=debug_credential_id,
                    debug_credential=debug_credential,
                    debug_scopes=_SATURN_DEBUG_SCOPES if debug_requested else (),
                )
                if claimed:
                    granted = db.job(job["job_id"])
                    if granted is None:  # pragma: no cover - conditional UPDATE invariant
                        raise RuntimeError("claimed job disappeared")
                    if guarded:
                        # This is the only serialization containing the raw capability.
                        # It is deliberately absent from DB job dictionaries, status,
                        # receipts, events, and the eventual child environment.
                        granted["agent_authorization"] = {
                            "lease_id": lease_identity,
                            "capability": lease_capability,
                            "expires_ts": lease_expires_ts,
                        }
                    if debug_requested:
                        # This narrower bearer is the only scheduler authority
                        # payload code receives.  It cannot submit commands,
                        # read job data, send telemetry, or act on another job.
                        granted["debug_authorization"] = {
                            "schema": "mrun.job-debug-credential-v1",
                            "credential_id": debug_credential_id,
                            "credential": debug_credential,
                            "scopes": list(_SATURN_DEBUG_SCOPES),
                            "expires_ts": lease_expires_ts,
                        }
                    db.add_event(
                        "job.claimed",
                        job_id=job["job_id"],
                        host=host,
                        state="assigned",
                        payload={
                            "reservation": res.as_dict(),
                            "plan": plan,
                            "debug_credential": (
                                {
                                    "schema": "mrun.job-debug-credential-v1",
                                    "credential_id": debug_credential_id,
                                    "scopes": list(_SATURN_DEBUG_SCOPES),
                                    "expires_ts": lease_expires_ts,
                                }
                                if debug_requested
                                else None
                            ),
                        },
                    )
                    return Response(content=json.dumps(granted), media_type="application/json")
                # Another lease request won the conditional update after our queue read.
                # Recompute admission so this agent may claim a different fitting job.
                db.add_event(
                    "job.claim_race",
                    job_id=job["job_id"],
                    host=host,
                    state=QUEUED,
                    reason="claim-race",
                )
                continue
        remaining = deadline - time.time()
        if remaining <= 0:
            return Response(status_code=204)
        with _wake:
            _wake.wait(timeout=min(remaining, 2.0))


def _payload_custody_is_exact(job: dict[str, Any]) -> bool:
    custody = job.get("payload_custody") or {}
    if not custody.get("required"):
        return True
    declared = custody.get("declared") or {}
    sealed = custody.get("sealed") or {}
    executed = custody.get("executed") or {}
    identity = (declared.get("sha256"), declared.get("size_bytes"))
    return (
        identity[0] is not None
        and identity == (sealed.get("sha256"), sealed.get("size_bytes"))
        and identity == (executed.get("sha256"), executed.get("size_bytes"))
        and executed.get("host") == job.get("assigned_host")
    )


def _finish_job(job: dict[str, Any], state: str, payload: dict[str, Any]) -> None:
    """Terminal transition: persist result, write the estimates-calibration row, wake."""
    if state == "succeeded" and not _payload_custody_is_exact(job):
        raise HTTPException(409, "guarded success requires exact executed payload custody")
    if state == "succeeded" and job.get("admission") is not None:
        assigned_host = job.get("assigned_host")
        plans = job.get("plans")
        submitted_plan = (
            plans.get(assigned_host)
            if isinstance(plans, dict) and isinstance(assigned_host, str)
            else None
        )
        if job.get("plan") != submitted_plan:
            raise HTTPException(409, "guarded success selected plan is not exactly submitted")
    result = payload.get("result") or {}
    db.update_job(
        job["job_id"],
        state=state,
        finished_ts=time.time(),
        result_json_obj=payload.get("result"),
        status_detail=payload.get("detail"),
    )
    plan = job.get("plan") or {}
    config = job.get("config") or {}
    reservation = job.get("reservation") or {}
    db.add_estimate(
        {
            "client_run_id": job["client_run_id"],
            "experiment": job["experiment"],
            "model": config.get("model"),
            "host": job.get("assigned_host"),
            "status": state,
            "ram_peak_mb": result.get("peak_rss_mb"),
            "vram_peak_mb": result.get("peak_vram_mb"),
            "wall_s": result.get("elapsed_s"),
            "backend": plan.get("backend"),
            "dtype": plan.get("dtype"),
            "task_family": config.get("task_family") or "forward",
            "est_ram_mb": plan.get("est_ram_mb"),
            # Written on EVERY terminal state: killed peaks (kill_state set) are floors
            # for growth-on-kill; only succeeded rows enter the p95 sizing statistics.
            "family_key": family_key_for(job["experiment"], job.get("cmd"), config),
            "reserved_ram_mb": reservation.get("ram_mb"),
            "reserved_vram_mb": reservation.get("vram_mb"),
            "kill_state": state if state in (KILLED_RAM, KILLED_VRAM) else None,
        }
    )
    db.add_event(
        "job.finished",
        job_id=job["job_id"],
        host=job.get("assigned_host"),
        state=state,
        reason=state,
        detail=payload.get("detail"),
        payload={"result": result, "plan": plan, "config": config},
    )
    if state in (KILLED_RAM, KILLED_VRAM):
        _maybe_auto_retry(job, state, result)
    _wake_all()  # capacity freed -> re-check queue


def _max_grantable_ram_mb(job: dict[str, Any]) -> float:
    """Largest RAM reservation whose kill ceiling still fits some candidate host."""
    hosts = db.host_rows()
    candidates = hosts_satisfying_needs(hosts, job.get("needs")) or hosts
    cap = max((host_ram_cap_mb(h) for h in candidates), default=0.0)
    return max(0.0, min(cap / RAM_KILL_FACTOR, cap - KILL_CEILING_ABS_MB))


def _max_grantable_vram_mb(job: dict[str, Any]) -> float:
    hosts = db.host_rows()
    candidates = hosts_satisfying_needs(hosts, job.get("needs")) or hosts
    cap = max((host_vram_cap_mb(h) for h in candidates), default=0.0)
    return max(0.0, cap / RAM_KILL_FACTOR)


def _reservation_recommendations(
    job: dict[str, Any], hosts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return conservative per-host reservations that can pass static capacity checks."""
    recommendations: list[dict[str, Any]] = []
    for host in hosts_satisfying_needs(hosts, job.get("needs")):
        ram_cap = host_ram_cap_mb(host)
        max_ram = max(0.0, min(ram_cap / RAM_KILL_FACTOR, ram_cap - KILL_CEILING_ABS_MB))
        vram_cap = host_vram_cap_mb(host)
        max_vram = vram_cap / RAM_KILL_FACTOR if vram_cap else 0.0
        reservation = {
            "ram_mb": math.floor(max_ram / 512.0) * 512.0,
            "vram_mb": math.floor(max_vram / 512.0) * 512.0 if max_vram else 0.0,
            "cpu_threads": int(host.get("cpu_threads") or 0),
        }
        recommendations.append(
            {
                "host": host.get("name"),
                "reservation": reservation,
                "reason": "largest conservative static reservation for this host",
            }
        )
    return recommendations


AUTO_RETRY_MAX_ATTEMPTS = 2
AUTO_RETRY_PEAK_FACTOR = 1.3
AUTO_RETRY_GROW_FACTOR = 1.5


def _maybe_auto_retry(job: dict[str, Any], state: str, result: dict[str, Any]) -> None:
    """Ceiling-killed job -> clone with a reservation grown from the MEASURED peak.

    This is what makes the small probe default safe: a first run killed at its ceiling
    costs one retry, not a manual resubmit. killed_ram/killed_vram states come only
    from the agent guard/cgroup ceiling — a host-pressure sentinel kill surfaces as
    ``failed`` — so growth always has a real measurement to work from.
    Opt-out: config.retry_on_kill = false. Guarded/external jobs never auto-clone
    (idempotency custody / not ours to run).
    """
    if job.get("admission") is not None or job.get("payload_kind") == "external":
        return
    config = job.get("config") or {}
    if config.get("retry_on_kill") is False:
        return
    meta = job.get("meta") or {}
    attempt = int(meta.get("auto_retry_attempt") or 0)
    if attempt >= AUTO_RETRY_MAX_ATTEMPTS:
        return
    res = Reservation.from_dict(job.get("reservation"))
    if state == KILLED_RAM:
        peak = float(result.get("peak_rss_mb") or 0.0)
        target = _round_up_mb(
            max(peak * AUTO_RETRY_PEAK_FACTOR, res.ram_mb * AUTO_RETRY_GROW_FACTOR)
        )
        grantable = math.floor(_max_grantable_ram_mb(job) / 512.0) * 512.0
        new_ram = min(target, max(grantable, 0.0))
        if new_ram <= res.ram_mb:
            db.add_event(
                "job.auto_retry_exhausted",
                job_id=job["job_id"],
                state=state,
                reason="exceeds fleet capacity",
                detail=f"peak {peak:.0f}MB needs > any host's grantable RAM",
            )
            return
        res.ram_mb = new_ram
    else:  # KILLED_VRAM
        vpeak = float(result.get("peak_vram_mb") or 0.0)
        target = _round_up_mb(
            max(vpeak * AUTO_RETRY_PEAK_FACTOR, res.vram_mb * AUTO_RETRY_GROW_FACTOR)
        )
        grantable = math.floor(_max_grantable_vram_mb(job) / 512.0) * 512.0
        new_vram = min(target, max(grantable, 0.0))
        if new_vram <= res.vram_mb:
            db.add_event(
                "job.auto_retry_exhausted",
                job_id=job["job_id"],
                state=state,
                reason="exceeds fleet capacity",
                detail=f"peak vram {vpeak:.0f}MB needs > any host's grantable VRAM",
            )
            return
        res.vram_mb = new_vram
    retry_job = dict(job)
    retry_job["reservation"] = res.as_dict()
    hosts = db.host_rows()
    candidates = hosts_satisfying_needs(hosts, retry_job.get("needs"))
    if candidates:
        fits, reasons = fits_some_host_capacity(retry_job, hosts)
        if not fits:
            db.add_event(
                "job.auto_retry_exhausted",
                job_id=job["job_id"],
                state=state,
                reason="reservation cannot fit fleet",
                detail="automatic retry was not queued because no host can admit it",
                payload={
                    "reservation": res.as_dict(),
                    "reasons": reasons,
                    "recommendations": _reservation_recommendations(retry_job, hosts),
                },
            )
            return
    res.source = "grown-on-kill"
    clone = _clone_job(
        job,
        reservation=res.as_dict(),
        meta={"auto_retry_of": job["job_id"], "auto_retry_attempt": attempt + 1},
        reason="auto-retry-on-kill",
    )
    if clone is None:
        return
    db.merge_meta(job["job_id"], {"auto_retry_job_id": clone["job_id"]})
    db.add_event(
        "job.auto_retry",
        job_id=clone["job_id"],
        state=clone["state"],
        reason=state,
        payload={
            "source_job_id": job["job_id"],
            "attempt": attempt + 1,
            "reservation": res.as_dict(),
        },
    )


@app.post("/api/jobs/{job_id}/events")
def job_event(job_id: str, request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    _guarded_agent_headers(request, job)
    state = payload.get("state")
    # Only an ASSIGNED/PREPARING/RUNNING job may take agent events: terminal states are
    # final (no resurrection, no duplicate estimates rows) and a queued job has no agent.
    if job["state"] not in ACTIVE_STATES:
        raise HTTPException(409, f"job is {job['state']!r}; events only apply to active jobs")
    now = time.time()
    phase = payload.get("phase")
    if phase is not None and state is None:
        if not isinstance(phase, str) or not phase.strip():
            raise HTTPException(400, "phase must be a non-empty string")
        db.add_event(
            "job.agent_phase",
            job_id=job_id,
            host=job.get("assigned_host"),
            state=job["state"],
            reason=phase,
            detail=payload.get("detail"),
            payload=payload,
        )
        return {"ok": True}
    if state in ("preparing", "running"):
        if state == "running" and not _payload_custody_is_exact(job):
            raise HTTPException(409, "guarded running requires exact executed payload custody")
        fields: dict[str, Any] = {"state": state, "lease_expires_ts": now + LEASE_EXPIRY_S}
        if state == "running" and not job.get("started_ts"):
            fields["started_ts"] = now
        db.update_job(job_id, **fields)
        db.add_event(
            "job.agent_event",
            job_id=job_id,
            host=job.get("assigned_host"),
            state=state,
            reason=state,
            payload=payload,
        )
    elif state in TERMINAL_STATES:
        _finish_job(job, state, payload)
    else:
        raise HTTPException(400, f"bad state {state!r}")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
def get_job_events(job_id: str, limit: int = Query(200, ge=1, le=10_000)) -> list[dict[str, Any]]:
    if db.job(job_id) is None:
        raise HTTPException(404, "unknown job")
    return db.events(job_id=job_id, limit=limit)


@app.get("/api/events")
def events(
    job_id: str | None = None,
    host: str | None = None,
    kind: str | None = None,
    limit: int = Query(500, ge=1, le=10_000),
) -> list[dict[str, Any]]:
    return db.events(job_id=job_id, host=host, kind=kind, limit=limit)


# ---------------------------------------------------------------- local (external) runs
# A guarded run executing OUTSIDE the agent (mx run --local on a dev box) registers here
# so its reservation counts against the host's committed RAM. Heartbeats renew the lease;
# an abandoned run goes `lost` via the normal lease expiry. Registration is best-effort
# client-side — the scheduler being down never blocks dev work.
def _require_external_local_run(job: dict[str, Any]) -> None:
    if job.get("payload_kind") != "external" or job.get("admission") is not None:
        raise HTTPException(409, "local-run endpoint requires an external local-run record")


def _external_command(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("command")
    if raw is None:
        raw = payload.get("cmd")
    if isinstance(raw, list) and raw and all(isinstance(value, str) for value in raw):
        return redact_command(raw)
    return ["<external>"]


@app.post("/api/local-runs")
def register_local_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    host = payload.get("host")
    _require(isinstance(host, str) and host != "", "host required")
    experiment = payload.get("experiment") or "local"
    reservation = _numeric_dict(payload.get("reservation"), "reservation") or {}
    from ..protocol import Reservation

    res = Reservation.from_dict(reservation)
    res.source = "declared"
    job = {
        "job_id": new_job_id(),
        "client_run_id": payload.get("client_run_id") or f"local:{experiment}",
        "experiment": experiment,
        # Born running (never QUEUED — a lease poll must never hand an external job to an
        # agent). assigned_host/lease land in the update right after the insert.
        "state": "running",
        "needs": {"host": host},
        "reservation": res.as_dict(),
        "payload_kind": "external",
        "cmd": _external_command(payload),
        "config": payload.get("config") or {},
    }
    db.insert_job(job)
    db.update_job(
        job["job_id"],
        state="running",
        assigned_host=host,
        started_ts=time.time(),
        lease_expires_ts=time.time() + LEASE_EXPIRY_S,
    )
    db.add_event(
        "local_run.register",
        job_id=job["job_id"],
        host=host,
        state="running",
        payload={"reservation": res.as_dict(), "config": job["config"]},
    )
    _refresh_queue_admission_details()
    return {"job_id": job["job_id"], "reservation": res.as_dict()}


@app.post("/api/local-runs/{job_id}/heartbeat")
def local_run_heartbeat(job_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    _require_external_local_run(job)
    if job["state"] not in ACTIVE_STATES:
        raise HTTPException(409, f"job is {job['state']!r}")
    rss = payload.get("tree_rss_mb")
    fields: dict[str, Any] = {"lease_expires_ts": time.time() + LEASE_EXPIRY_S}
    if rss is not None:
        try:
            fields["external_rss_mb"] = float(rss)
        except (TypeError, ValueError):
            pass
    vram = payload.get("tree_vram_mb")
    if vram is not None:
        try:
            fields["external_vram_mb"] = float(vram)
        except (TypeError, ValueError):
            pass
    db.update_job(job_id, **fields)
    db.add_event(
        "local_run.heartbeat",
        job_id=job_id,
        host=job.get("assigned_host"),
        state=job.get("state"),
        payload={k: v for k, v in fields.items() if k != "lease_expires_ts"},
    )
    _refresh_queue_admission_details()
    return {"ok": True}


@app.post("/api/local-runs/{job_id}/finish")
def local_run_finish(job_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    _require_external_local_run(job)
    if job["state"] not in ACTIVE_STATES:
        raise HTTPException(409, f"job is {job['state']!r}; already terminal")
    state = payload.get("state") or "succeeded"
    if state not in TERMINAL_STATES:
        raise HTTPException(400, f"bad state {state!r}")
    if state != "succeeded":
        payload = dict(payload)
        payload["result"] = normalize_external_result(
            state=state,
            result=payload.get("result"),
            detail=payload.get("detail"),
            command=job.get("cmd"),
        )
    _finish_job(job, state, payload)
    db.add_event(
        "local_run.finish",
        job_id=job_id,
        host=job.get("assigned_host"),
        state=state,
        payload=payload,
    )
    return {"ok": True}


@app.post("/api/jobs/{job_id}/logs")
def push_logs(
    job_id: str, request: Request, offset: int = Query(...), body: bytes = Body(...)
) -> dict[str, Any]:
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    _guarded_agent_headers(request, job)
    path = data_dir() / "logs" / f"{job_id}.log"
    size = path.stat().st_size if path.exists() else 0
    if offset != size:
        # idempotent append: agent re-syncs from our authoritative offset
        db.add_event(
            "job.logs_resync",
            job_id=job_id,
            reason="offset-mismatch",
            payload={"requested_offset": offset, "server_offset": size},
        )
        return {"next_offset": size, "resync": True}
    with open(path, "ab") as f:
        f.write(body)
    db.add_event(
        "job.logs_append",
        job_id=job_id,
        payload={"offset": offset, "bytes": len(body), "next_offset": size + len(body)},
    )
    return {"next_offset": size + len(body)}


@app.get("/api/payloads/{job_id}")
def get_payload(job_id: str, request: Request) -> Response:
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    _guarded_agent_headers(request, job)
    sealed = (job.get("payload_custody") or {}).get("sealed")
    if sealed is None:
        raise HTTPException(425, "payload is not sealed")
    path = data_dir() / "payloads" / f"{job_id}.tgz"
    if not path.exists():
        raise HTTPException(404, "no payload")
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != sealed["sha256"] or len(body) != sealed["size_bytes"]:
        raise HTTPException(500, "sealed payload file failed custody verification")
    return Response(
        content=body,
        media_type="application/gzip",
        headers={
            "X-MRun-Payload-SHA256": digest,
            "X-MRun-Payload-Size": str(len(body)),
        },
    )


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise HTTPException(422, msg)


def _optional_audit_message(raw: Any, name: str) -> str | None:
    """Validate bounded human context without making prose part of job semantics."""

    if raw is None:
        return None
    _require(isinstance(raw, str), f"{name} must be a string or null")
    value = raw.strip()
    _require(bool(value), f"{name} must not be empty")
    _require(len(value) <= 500, f"{name} must be at most 500 characters")
    return value


def _numeric_dict(raw: Any, name: str) -> dict[str, Any] | None:
    """Validate a reservation-shaped dict: numeric fields must be numbers (a str ram_mb
    used to poison the queue and 500 every lease poll — admission compares floats)."""
    if raw is None:
        return None
    _require(isinstance(raw, dict), f"{name} must be an object")
    out: dict[str, Any] = {}
    for key, val in raw.items():
        if key in ("ram_mb", "vram_mb", "disk_gb", "est_wall_s") and val is not None:
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                raise HTTPException(422, f"{name}.{key} must be a number, got {val!r}") from None
            _require(math.isfinite(out[key]), f"{name}.{key} must be finite")
        elif key == "cpu_threads" and val is not None:
            try:
                out[key] = int(val)
            except (TypeError, ValueError):
                raise HTTPException(422, f"{name}.{key} must be an int, got {val!r}") from None
        else:
            out[key] = val
    return out


_ADMISSION_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADMISSION_CLAIM_MAX_TTL_S = 900.0


def _admission_identity(payload: dict[str, Any]) -> tuple[str, str]:
    claim_key = payload.get("claim_key")
    owner_token = payload.get("owner_token")
    _require(
        isinstance(claim_key, str) and _ADMISSION_TOKEN.fullmatch(claim_key) is not None,
        "claim_key must be a safe 1-200 character token",
    )
    _require(
        isinstance(owner_token, str) and _ADMISSION_TOKEN.fullmatch(owner_token) is not None,
        "owner_token must be a safe 1-200 character token",
    )
    return claim_key, owner_token


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _admission_scope(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    scope = payload.get("scope")
    _require(isinstance(scope, dict), "scope must be an object")
    _require(
        set(scope) == {"experiment", "config_selector"},
        "scope must contain exactly experiment and config_selector",
    )
    experiment = scope.get("experiment")
    selector = scope.get("config_selector")
    _require(
        isinstance(experiment, str) and 0 < len(experiment) <= 200,
        "scope.experiment must be a 1-200 character string",
    )
    _require(
        isinstance(selector, dict) and 0 < len(selector) <= 32,
        "scope.config_selector must be a non-empty object with at most 32 fields",
    )
    _require(
        all(
            isinstance(key, str) and _ADMISSION_TOKEN.fullmatch(key) is not None for key in selector
        ),
        "scope.config_selector keys must be safe tokens",
    )
    normalized = {"experiment": experiment, "config_selector": selector}
    _require(len(_canonical_json(normalized)) <= 16_384, "scope is too large")
    return normalized, _json_sha256(normalized)


def _positive_epoch(raw: Any, name: str = "fencing_epoch") -> int:
    _require(not isinstance(raw, bool) and isinstance(raw, int), f"{name} must be an int")
    _require(raw > 0, f"{name} must be positive")
    return raw


def _payload_declaration(payload: dict[str, Any]) -> tuple[str, int]:
    declared = payload.get("payload")
    _require(isinstance(declared, dict), "shipped guarded job requires payload declaration")
    _require(
        set(declared) == {"sha256", "size_bytes"},
        "payload declaration must contain exactly sha256 and size_bytes",
    )
    sha256 = declared.get("sha256")
    size_bytes = declared.get("size_bytes")
    _require(
        isinstance(sha256, str) and _SHA256.fullmatch(sha256) is not None,
        "payload.sha256 must be a lowercase SHA-256 hex digest",
    )
    _require(
        not isinstance(size_bytes, bool) and isinstance(size_bytes, int) and size_bytes >= 0,
        "payload.size_bytes must be a non-negative int",
    )
    from ..protocol import MAX_PAYLOAD_BYTES

    _require(size_bytes <= MAX_PAYLOAD_BYTES, "declared payload is too large")
    return sha256, size_bytes


@app.post("/api/admission-claims/acquire")
def acquire_admission_claim(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Atomically acquire/renew a scheduler-wide preflight -> submit lease."""

    claim_key, owner_token = _admission_identity(payload)
    ttl_s = payload.get("ttl_s")
    _require(
        not isinstance(ttl_s, bool) and isinstance(ttl_s, (int, float)),
        "ttl_s must be a number",
    )
    ttl = float(ttl_s)
    _require(
        math.isfinite(ttl) and 0.0 < ttl <= _ADMISSION_CLAIM_MAX_TTL_S,
        f"ttl_s must be in (0, {_ADMISSION_CLAIM_MAX_TTL_S:g}]",
    )
    metadata = payload.get("metadata") or {}
    _require(isinstance(metadata, dict), "metadata must be an object")
    _require(
        len(json.dumps(metadata, sort_keys=True)) <= 16_384,
        "metadata is too large",
    )
    scope, scope_sha256 = _admission_scope(payload)
    try:
        record = db.acquire_admission_claim(
            claim_key,
            owner_token,
            ttl_s=ttl,
            scope=scope,
            scope_sha256=scope_sha256,
            metadata=metadata,
        )
    except GuardedAdmissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    if record is None:
        raise HTTPException(409, "admission claim is held by another live owner")
    db.add_event(
        "admission.claim_acquired",
        reason="acquired-or-renewed",
        payload={
            "claim_key": claim_key,
            "expires_ts": record["expires_ts"],
            "metadata": metadata,
        },
    )
    return {
        "schema": "mrun.admission-claim.v2",
        "claim_key": claim_key,
        "owner_token": owner_token,
        "acquired_ts": record["acquired_ts"],
        "expires_ts": record["expires_ts"],
        "fencing_epoch": record["fencing_epoch"],
        "scope": record["scope"],
        "scope_sha256": record["scope_sha256"],
    }


@app.post("/api/admission-claims/release")
def release_admission_claim(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Release only the exact caller-owned claim; wrong owners fail closed."""

    claim_key, owner_token = _admission_identity(payload)
    fencing_epoch = _positive_epoch(payload.get("fencing_epoch"))
    if not db.release_admission_claim(claim_key, owner_token, fencing_epoch):
        raise HTTPException(409, "admission claim is absent or owned by another client")
    db.add_event(
        "admission.claim_released",
        reason="released",
        payload={"claim_key": claim_key},
    )
    return {
        "schema": "mrun.admission-claim-release.v2",
        "claim_key": claim_key,
        "fencing_epoch": fencing_epoch,
        "released": True,
    }


def _normalized_job(
    payload: dict[str, Any], *, guarded: bool
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Validate one request and separate submitted identity from resolved sizing.

    The idempotency digest covers only the normalized bytes the client submitted (plus
    the server-required custody capability), never history-dependent reservation output.
    Resolved reservation and later selected plan are separately exposed in receipts.
    """

    client_run_id = payload.get("client_run_id")
    _require(isinstance(client_run_id, str) and client_run_id != "", "client_run_id required")
    experiment = payload.get("experiment", "unnamed")
    _require(
        isinstance(experiment, str) and 0 < len(experiment) <= 200,
        "experiment must be a 1-200 character string",
    )
    cmd = payload.get("cmd") or []
    _require(
        isinstance(cmd, list) and cmd and all(isinstance(c, str) for c in cmd),
        "cmd must be a non-empty list of strings",
    )
    needs = payload.get("needs") or {}
    _require(isinstance(needs, dict), "needs must be an object")
    needs = dict(needs)
    config = payload.get("config") or {}
    _require(isinstance(config, dict), "config must be an object")
    note = _optional_audit_message(payload.get("note"), "note")
    payload_kind = payload.get("payload_kind", "cmd")
    _require(payload_kind in ("cmd", "shipped"), "payload_kind must be cmd or shipped")
    if guarded:
        _require(payload_kind == "shipped", "guarded jobs require payload_kind shipped")
    env_alias = payload.get("env_alias")
    _require(env_alias is None or isinstance(env_alias, str), "env_alias must be a string or null")
    try:
        priority = max(-100, min(100, int(payload.get("priority") or 0)))
    except (TypeError, ValueError):
        raise HTTPException(422, "priority must be an int") from None
    timeout_s = payload.get("timeout_s")
    if timeout_s is not None:
        try:
            timeout_s = float(timeout_s)
        except (TypeError, ValueError):
            raise HTTPException(422, "timeout_s must be a number") from None
        _require(
            math.isfinite(timeout_s) and timeout_s > 0,
            "timeout_s must be finite and positive",
        )
    plans = payload.get("plans")
    if plans is not None and not (
        isinstance(plans, dict) and all(isinstance(v, dict) for v in plans.values())
    ):
        raise HTTPException(422, "plans must be an object of host -> plan objects")
    declared_reservation = _numeric_dict(payload.get("reservation"), "reservation")
    client_estimate = _numeric_dict(payload.get("client_estimate"), "client_estimate")
    declared_payload: dict[str, Any] | None = None
    custody_required = guarded and payload_kind == "shipped"
    if guarded and payload_kind == "shipped":
        sha256, size_bytes = _payload_declaration(payload)
        declared_payload = {"sha256": sha256, "size_bytes": size_bytes}
        # This is server-owned, not a client opt-in.  Only updated agents advertise it.
        needs["payload_custody_v2"] = True

    submitted_request = {
        "schema": "mrun.normalized-job-request.v2" if guarded else "mrun.normalized-job-request.v1",
        "experiment": experiment,
        "client_run_id": client_run_id,
        "cmd": list(cmd),
        "needs": needs,
        "reservation": declared_reservation,
        "client_estimate": client_estimate,
        "payload_kind": payload_kind,
        "payload": declared_payload,
        "env_alias": env_alias,
        "config": config,
        "timeout_s": timeout_s,
        "priority": priority,
        "plans": plans,
    }
    submitted_request_sha256 = _json_sha256(submitted_request)
    # Family history exists for EVERY job now (model+task, or experiment+script for
    # the model-less majority) — the measurement feedback loop no longer needs a
    # config.model to engage.
    # Resident Saturn workers have a different memory geometry from one-shot
    # Diffusers/offload jobs. Do not let stale family history for the latter
    # inflate a resident request past the host's static CUDA envelope; the
    # intent-first client estimate already carries the resident profile and
    # exact history can still refine this identity on subsequent runs.
    generalized = (
        None
        if bool(config.get("resident_worker"))
        else db.history_stats(family_key_for(experiment, cmd, config))
    )
    sized = size_reservation(
        declared_reservation,
        exact_history=db.exact_stats(client_run_id),
        family_history=generalized,
        client_estimate=client_estimate,
        needs_cuda=bool(needs.get("cuda")),
        hosts=db.host_rows(),
        needs=needs,
    )
    reservation = sized.reservation
    job = {
        "job_id": new_job_id(),
        "client_run_id": client_run_id,
        "experiment": experiment,
        "state": AWAITING_PAYLOAD if payload_kind == "shipped" else QUEUED,
        "needs": needs,
        "reservation": reservation.as_dict(),
        "payload_kind": payload_kind,
        "env_alias": env_alias,
        "cmd": cmd,
        "config": config,
        "timeout_s": timeout_s,
        "priority": priority,
        "plans": plans,
        "meta": ({"queue_note": note} if note is not None else {}),
        "request_json": _canonical_json(submitted_request),
        "request_sha256": submitted_request_sha256,
        "custody_required": custody_required,
        "payload_declared_sha256": (
            declared_payload["sha256"] if declared_payload is not None else None
        ),
        "payload_declared_size": (
            declared_payload["size_bytes"] if declared_payload is not None else None
        ),
        # Transient (not a DB column): surfaced in the submit response + clamp event;
        # the durable marker is reservation.source = declared-clamped / declared-raised.
        "reservation_clamps": sized.clamps,
    }
    return job, submitted_request, submitted_request_sha256


def _guarded_job_receipt(job: dict[str, Any], *, created: bool | None) -> dict[str, Any]:
    plans = job.get("plans")
    assigned_host = job.get("assigned_host")
    selected_plan = job.get("plan")
    submitted_plan_present = (
        isinstance(plans, dict) and isinstance(assigned_host, str) and assigned_host in plans
    )
    expected_plan = (
        plans.get(assigned_host)
        if isinstance(plans, dict) and isinstance(assigned_host, str)
        else None
    )
    plan_binding = {
        "assigned_host": assigned_host,
        "selected_plan": selected_plan,
        "submitted_plan": expected_plan,
        "submitted_plan_present": submitted_plan_present,
        "matches_submitted_plan": (
            selected_plan == expected_plan if assigned_host is not None else None
        ),
        "binding_status": (
            "pending_assignment"
            if assigned_host is None
            else (
                "exact_submitted_plan"
                if submitted_plan_present and selected_plan == expected_plan
                else (
                    "explicit_no_submitted_plan"
                    if not submitted_plan_present and selected_plan is None
                    else "mismatch"
                )
            )
        ),
    }
    return {
        "schema": "mrun.guarded-job-receipt.v2",
        "job_id": job["job_id"],
        "state": job["state"],
        "created": created,
        "idempotent_replay": (not created) if created is not None else None,
        "admission": job.get("admission"),
        "submitted_request": job.get("submitted_request"),
        "submitted_request_sha256": job.get("submitted_request_sha256"),
        "resolved_execution": {
            "reservation": job.get("reservation"),
            "plans": plans,
            "plan_binding": plan_binding,
        },
        "payload_custody": job.get("payload_custody"),
        "meta": job.get("meta") or {},
        "assigned_host": assigned_host,
        "result": job.get("result"),
        "created_ts": job.get("created_ts"),
        "started_ts": job.get("started_ts"),
        "finished_ts": job.get("finished_ts"),
    }


# --------------------------------------------------------------------- client
def _preflight_or_422(job: dict[str, Any], payload: dict[str, Any]) -> None:
    """Reject a job no registered host could EVER fit, at submit time.

    A silently-unschedulable ask otherwise queues forever (46GB ask cancelled after
    days, 2026-07-26; 18GB VRAM on a 16GB card, 2026-07-31). ``allow_unschedulable``
    queues anyway — for pre-registering work before its host joins the fleet."""
    if payload.get("allow_unschedulable"):
        return
    hosts = db.host_rows()
    if not hosts:
        return  # empty fleet: nothing to judge against
    if not hosts_satisfying_needs(hosts, job.get("needs")):
        # A missing pin target / capability host is temporal (it may register later);
        # the queued job carries an `unschedulable:` status_detail. Only *capacity*
        # impossibility on existing candidates is permanent enough to hard-reject.
        return
    ok, reasons = fits_some_host_capacity(job, hosts)
    if not ok:
        raise HTTPException(
            422,
            {
                "error": (
                    "reservation can never fit any registered host "
                    "(pass allow_unschedulable=true to queue anyway)"
                ),
                "reasons": reasons,
                "recommendations": _reservation_recommendations(job, hosts),
                "reservation": job["reservation"],
            },
        )


def _record_clamps(job: dict[str, Any]) -> None:
    if job.get("reservation_clamps"):
        db.add_event(
            "job.reservation_clamped",
            job_id=job["job_id"],
            state=job["state"],
            payload={"clamps": job["reservation_clamps"]},
        )


def _admission_outlook(job: dict[str, Any]) -> dict[str, Any]:
    all_hosts = db.host_rows()
    active_by_host = _active_by_host_for_admission(all_hosts)
    _detail, status, info = _admission_status(job, all_hosts, active_by_host)
    return {
        "status": status,
        "reasons": info.get("current_reasons") or [],
        "clamps": job.get("reservation_clamps") or [],
        "recommendations": (
            _reservation_recommendations(job, all_hosts) if status == "unschedulable" else []
        ),
    }


@app.post("/api/jobs")
def submit(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    job, _submitted, _submitted_sha = _normalized_job(payload, guarded=False)
    _preflight_or_422(job, payload)
    try:
        db.insert_job(job)
    except ProtectedScopeError as exc:
        raise HTTPException(409, f"protected scope requires POST /api/jobs/guarded: {exc}") from exc
    db.add_event(
        "job.submit",
        job_id=job["job_id"],
        state=job["state"],
        payload={
            "experiment": job["experiment"],
            "client_run_id": job["client_run_id"],
            "needs": job["needs"],
            "reservation": job["reservation"],
            "config": job["config"],
            "priority": job["priority"],
            "payload_kind": job["payload_kind"],
            "note": (job.get("meta") or {}).get("queue_note"),
        },
    )
    _record_clamps(job)
    if job["state"] in (QUEUED, AWAITING_PAYLOAD):
        _refresh_queue_admission_details([job])
        _wake_all()
    return {
        "job_id": job["job_id"],
        "state": job["state"],
        "reservation": job["reservation"],
        "note": (job.get("meta") or {}).get("queue_note"),
        "admission_outlook": _admission_outlook(job),
    }


@app.post("/api/jobs/guarded")
def submit_guarded(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    admission = payload.get("admission")
    _require(isinstance(admission, dict), "admission must be an object")
    _require(
        set(admission)
        == {
            "claim_key",
            "owner_token",
            "fencing_epoch",
            "idempotency_key",
        },
        "admission must contain exactly claim_key, owner_token, fencing_epoch, idempotency_key",
    )
    claim_key, owner_token = _admission_identity(admission)
    fencing_epoch = _positive_epoch(admission.get("fencing_epoch"))
    idempotency_key = admission.get("idempotency_key")
    _require(
        isinstance(idempotency_key, str)
        and _ADMISSION_TOKEN.fullmatch(idempotency_key) is not None,
        "idempotency_key must be a safe 1-200 character token",
    )
    job, _submitted, submitted_sha = _normalized_job(payload, guarded=True)
    _preflight_or_422(job, payload)
    try:
        admitted, created = db.admit_guarded_job(
            job,
            claim_key=claim_key,
            owner_token=owner_token,
            fencing_epoch=fencing_epoch,
            idempotency_key=idempotency_key,
            request_sha256=submitted_sha,
        )
    except GuardedAdmissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    if created:
        note = (job.get("meta") or {}).get("queue_note")
        db.add_event(
            "job.submit_guarded",
            job_id=admitted["job_id"],
            state=admitted["state"],
            payload={
                "admission": admitted["admission"],
                "submitted_request_sha256": submitted_sha,
                "payload_custody": admitted["payload_custody"],
                "note": note,
            },
        )
        _record_clamps({**job, "job_id": admitted["job_id"], "state": admitted["state"]})
        if admitted["state"] == QUEUED:
            _refresh_queue_admission_details([admitted])
            _wake_all()
    receipt = _guarded_job_receipt(admitted, created=created)
    receipt["admission_outlook"] = _admission_outlook(
        {**admitted, "reservation_clamps": job.get("reservation_clamps")}
    )
    return receipt


@app.put("/api/jobs/{job_id}/payload")
async def put_payload(
    job_id: str,
    request: Request,
    owner_token: str | None = Header(None, alias="X-MRun-Admission-Owner"),
    epoch_header: str | None = Header(None, alias="X-MRun-Admission-Epoch"),
) -> dict[str, Any]:
    from ..protocol import MAX_PAYLOAD_BYTES

    body = await request.body()
    if len(body) > MAX_PAYLOAD_BYTES:
        raise HTTPException(413, "payload too large")
    fencing_epoch: int | None = None
    if epoch_header is not None:
        try:
            fencing_epoch = int(epoch_header)
        except ValueError:
            raise HTTPException(422, "X-MRun-Admission-Epoch must be an int") from None
        _require(fencing_epoch > 0, "X-MRun-Admission-Epoch must be positive")
    sha256 = hashlib.sha256(body).hexdigest()
    try:
        job, created = db.seal_job_payload(
            job_id,
            owner_token=owner_token,
            fencing_epoch=fencing_epoch,
            sha256=sha256,
            size_bytes=len(body),
            body=body,
            path=data_dir() / "payloads" / f"{job_id}.tgz",
        )
    except KeyError as exc:
        raise HTTPException(404, "unknown job") from exc
    except GuardedAdmissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    db.add_event(
        "job.payload_sealed" if created else "job.payload_seal_replay",
        job_id=job_id,
        state=job["state"],
        payload={"sha256": sha256, "size_bytes": len(body), "created": created},
    )
    if created:
        _refresh_queue_admission_details([job])
        _wake_all()
    return {
        "schema": "mrun.payload-seal-receipt.v2",
        "job_id": job_id,
        "state": job["state"],
        "created": created,
        "idempotent_replay": not created,
        "payload_custody": job["payload_custody"],
    }


@app.post("/api/jobs/{job_id}/payload/executed")
def report_payload_executed(
    job_id: str, request: Request, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    existing = db.job(job_id)
    if existing is None:
        raise HTTPException(404, "unknown job")
    _guarded_agent_headers(request, existing)
    _require(
        set(payload) == {"host", "sha256", "size_bytes"},
        "payload executed report must contain exactly host, sha256, size_bytes",
    )
    host = payload.get("host")
    sha256 = payload.get("sha256")
    size_bytes = payload.get("size_bytes")
    _require(isinstance(host, str) and host != "", "host required")
    _require(
        isinstance(sha256, str) and _SHA256.fullmatch(sha256) is not None,
        "sha256 must be a lowercase SHA-256 hex digest",
    )
    _require(
        not isinstance(size_bytes, bool) and isinstance(size_bytes, int) and size_bytes >= 0,
        "size_bytes must be a non-negative int",
    )
    try:
        job, created = db.report_executed_payload(
            job_id,
            host=host,
            sha256=sha256,
            size_bytes=size_bytes,
        )
    except KeyError as exc:
        raise HTTPException(404, "unknown job") from exc
    except GuardedAdmissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    db.add_event(
        "job.payload_executed" if created else "job.payload_executed_replay",
        job_id=job_id,
        host=host,
        state=job["state"],
        payload={"sha256": sha256, "size_bytes": size_bytes, "created": created},
    )
    return {
        "schema": "mrun.payload-executed-receipt.v2",
        "job_id": job_id,
        "state": job["state"],
        "created": created,
        "idempotent_replay": not created,
        "payload_custody": job["payload_custody"],
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.post("/api/jobs/{job_id}/debug/register")
def register_saturn_debugger(
    job_id: str, request: Request, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """Register an instrumented Saturn worker with the job control broker."""
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    credential_id = _guarded_debug_headers(request, job, scope="debug:register")
    capability = payload.get("capability")
    if (
        not isinstance(capability, dict)
        or capability.get("schema") != "saturn-debugger-capability-v1"
    ):
        raise HTTPException(422, "invalid Saturn debugger capability")
    if capability.get("job_id") != job_id:
        raise HTTPException(409, "debugger capability job identity mismatch")
    with _saturn_debug_lock:
        existing = _saturn_debug_channels.get(job_id)
        if existing is not None:
            same_credential = existing.get("credential_id") == credential_id
            same_session = (existing.get("capability") or {}).get("session_id") == capability.get(
                "session_id"
            )
            if same_credential and not same_session:
                raise HTTPException(409, "debugger credential is already bound to another session")
            if same_credential and same_session:
                existing["last_seen_ts"] = time.time()
                return {
                    "ok": True,
                    "job_id": job_id,
                    "capability": existing["capability"],
                    "idempotent": True,
                }
            # A newly verified credential means the scheduler issued a new
            # lease attempt.  Replace the stale attempt's in-memory mailbox.
        _saturn_debug_channels[job_id] = {
            "capability": capability,
            "credential_id": credential_id,
            "pending": {},
            "responses": {},
            "registered_ts": time.time(),
            "last_seen_ts": time.time(),
        }
        _saturn_debug_lock.notify_all()
    db.add_event(
        "saturn.debug.registered",
        job_id=job_id,
        host=job.get("assigned_host"),
        payload={
            "capability": capability,
            "debug_credential_id": credential_id,
        },
    )
    return {"ok": True, "job_id": job_id, "capability": capability}


@app.post("/api/jobs/{job_id}/debug")
def send_saturn_debug_command(job_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Route one authenticated client request to the registered worker."""
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.get("state") not in ACTIVE_STATES:
        with _saturn_debug_lock:
            _saturn_debug_channels.pop(job_id, None)
            _saturn_debug_lock.notify_all()
        raise HTTPException(409, "Saturn debugger job is not active")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise HTTPException(422, "debug request requires request_id")
    with _saturn_debug_lock:
        channel = _saturn_debug_channels.get(job_id)
        if channel is None:
            raise HTTPException(404, "job has no Saturn debug capability")
        now = time.time()
        last_seen = float(channel.get("last_seen_ts") or channel.get("registered_ts") or 0.0)
        if now - last_seen > _SATURN_DEBUG_RETENTION_S:
            _saturn_debug_channels.pop(job_id, None)
            _saturn_debug_lock.notify_all()
            raise HTTPException(410, "Saturn debugger mailbox expired")
        channel["last_seen_ts"] = now
        old = channel["responses"].get(request_id)
        if old is not None:
            return old
        channel["pending"][request_id] = dict(payload)
        deadline = time.time() + 30.0
        _saturn_debug_lock.notify_all()
        while request_id not in channel["responses"] and time.time() < deadline:
            _saturn_debug_lock.wait(timeout=max(0.01, deadline - time.time()))
        response = channel["responses"].get(request_id)
        channel["pending"].pop(request_id, None)
        if len(channel["responses"]) > 1024:
            oldest = next(iter(channel["responses"]))
            channel["responses"].pop(oldest, None)
    if response is None:
        raise HTTPException(504, "Saturn debugger worker did not acknowledge before timeout")
    return response


@app.get("/api/jobs/{job_id}/debug")
def poll_saturn_debug_command(
    job_id: str, request: Request, wait_s: float = Query(0.0, ge=0.0, le=30.0)
) -> dict[str, Any]:
    """Worker-side safe-point poll; mrun never interprets model state."""
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    credential_id = _guarded_debug_headers(request, job, scope="debug:poll")
    deadline = time.time() + wait_s
    with _saturn_debug_lock:
        channel = _saturn_debug_channels.get(job_id)
        if channel is None:
            raise HTTPException(404, "job has no Saturn debug capability")
        if channel.get("credential_id") != credential_id:
            raise HTTPException(403, "debugger mailbox belongs to another lease")
        channel["last_seen_ts"] = time.time()
        while not channel["pending"] and time.time() < deadline:
            _saturn_debug_lock.wait(timeout=max(0.01, deadline - time.time()))
        if not channel["pending"]:
            return {"schema": "saturn-debugger-poll-v1", "request": None}
        request_id = next(iter(channel["pending"]))
        return {
            "schema": "saturn-debugger-poll-v1",
            "request": channel["pending"][request_id],
        }


@app.post("/api/jobs/{job_id}/debug/respond")
def respond_saturn_debug_command(
    job_id: str, request: Request, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    credential_id = _guarded_debug_headers(request, job, scope="debug:respond")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise HTTPException(422, "debug response requires request_id")
    with _saturn_debug_lock:
        channel = _saturn_debug_channels.get(job_id)
        if channel is not None and channel.get("credential_id") != credential_id:
            raise HTTPException(403, "debugger mailbox belongs to another lease")
        if channel is None or request_id not in channel["pending"]:
            raise HTTPException(409, "debug request is not pending")
        channel["last_seen_ts"] = time.time()
        channel["responses"][request_id] = dict(payload)
        _saturn_debug_lock.notify_all()
    return {"ok": True, "request_id": request_id}


@app.get("/api/jobs/{job_id}/receipt")
def guarded_job_receipt(job_id: str) -> dict[str, Any]:
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.get("admission") is None:
        raise HTTPException(409, "job is not guarded")
    return _guarded_job_receipt(job, created=None)


@app.get("/api/jobs")
def list_jobs(
    client_run_id: str | None = None,
    state: str | None = None,
    debuggable: bool = False,
) -> list[dict[str, Any]]:
    jobs = db.jobs(state=state, client_run_id=client_run_id)
    states = {str(job.get("job_id")): job.get("state") for job in jobs}
    with _saturn_debug_lock:
        now = time.time()
        pruned = False
        for channel_job_id, channel in list(_saturn_debug_channels.items()):
            last_seen = float(channel.get("last_seen_ts") or channel.get("registered_ts") or 0.0)
            if (
                now - last_seen > _SATURN_DEBUG_RETENTION_S
                or states.get(channel_job_id) in TERMINAL_STATES
            ):
                _saturn_debug_channels.pop(channel_job_id, None)
                pruned = True
        if pruned:
            _saturn_debug_lock.notify_all()
        registered = {
            job_id
            for job_id, channel in _saturn_debug_channels.items()
            if now - float(channel.get("last_seen_ts") or channel.get("registered_ts") or 0.0)
            <= _SATURN_DEBUG_STALE_S
        }
    result = [{**job, "debugger_capable": job.get("job_id") in registered} for job in jobs]
    if debuggable:
        return [job for job in result if job["debugger_capable"]]
    return result


def _clone_job(
    job: dict[str, Any],
    *,
    reservation: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    reason: str,
    priority: int | None = None,
) -> dict[str, Any] | None:
    """Re-queue a copy of ``job`` (shared by restart, auto-retry and lost-requeue).

    Returns the new job row, or None when the job cannot be cloned (external record,
    guarded idempotency, or its shipped payload archive is gone).
    """
    payload_kind = job.get("payload_kind") or "cmd"
    if payload_kind == "external" or job.get("admission") is not None:
        return None
    shipped_body: bytes | None = None
    if payload_kind == "shipped":
        src = data_dir() / "payloads" / f"{job['job_id']}.tgz"
        if not src.exists():
            return None
        shipped_body = src.read_bytes()
    new_id = new_job_id()
    new_job = {
        "job_id": new_id,
        "client_run_id": job.get("client_run_id") or f"{reason}:{job['job_id']}",
        "experiment": job.get("experiment") or reason,
        "state": AWAITING_PAYLOAD if payload_kind == "shipped" else QUEUED,
        "needs": job.get("needs") or {},
        "reservation": reservation or job.get("reservation") or {},
        "payload_kind": payload_kind,
        "env_alias": job.get("env_alias"),
        "cmd": job.get("cmd") or [],
        "config": job.get("config") or {},
        "timeout_s": job.get("timeout_s"),
        "priority": priority if priority is not None else int(job.get("priority") or 0),
        "plans": job.get("plans"),
        "meta": {"queue_note": (job.get("meta") or {}).get("queue_note")}
        if (job.get("meta") or {}).get("queue_note")
        else {},
    }
    db.insert_job(new_job)
    if shipped_body is not None:
        new_job, _created = db.seal_job_payload(
            new_id,
            owner_token=None,
            fencing_epoch=None,
            sha256=hashlib.sha256(shipped_body).hexdigest(),
            size_bytes=len(shipped_body),
            body=shipped_body,
            path=data_dir() / "payloads" / f"{new_id}.tgz",
        )
    if meta:
        db.merge_meta(new_id, meta)
        new_job = db.job(new_id) or new_job
    _refresh_queue_admission_details([new_job])
    _wake_all()
    return new_job


def _requeue_lost(job_ids: list[str]) -> None:
    """A lost job (agent went silent mid-lease) re-drives ONCE automatically (I3)."""
    for jid in job_ids:
        job = db.job(jid)
        if not job or job.get("state") != "lost":
            continue
        if (job.get("meta") or {}).get("requeued_after_lost"):
            continue
        clone = _clone_job(
            job,
            meta={"requeued_after_lost": 1, "auto_retry_of": jid},
            reason="requeue-after-lost",
        )
        if clone is None:
            continue
        db.merge_meta(jid, {"auto_retry_job_id": clone["job_id"]})
        db.add_event(
            "job.requeued_after_lost",
            job_id=clone["job_id"],
            state=clone["state"],
            reason="lease expired",
            payload={"source_job_id": jid},
        )


@app.post("/api/jobs/{job_id}/restart")
def restart_job(job_id: str, payload: dict[str, Any] | None = Body(None)) -> dict[str, Any]:
    """Clone a terminal job into a new queued job.

    Terminal states remain immutable; restart is a fresh job with a source link in meta.
    Active/queued jobs are refused to avoid accidental duplicate execution.
    """
    payload = payload or {}
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job["state"] not in TERMINAL_STATES:
        raise HTTPException(409, f"job is {job['state']!r}; only terminal jobs restart")
    if job.get("admission") is not None:
        raise HTTPException(
            409, "guarded jobs cannot bypass idempotency through the restart endpoint"
        )
    payload_kind = job.get("payload_kind") or "cmd"
    if payload_kind == "external":
        raise HTTPException(409, "external local-run records cannot be restarted")
    try:
        priority = max(-100, min(100, int(payload.get("priority", job.get("priority") or 0))))
    except (TypeError, ValueError):
        raise HTTPException(422, "priority must be an int") from None

    new_job = _clone_job(
        job,
        meta={"restart_of": job_id, "restart_source_state": job["state"]},
        reason="restart",
        priority=priority,
    )
    if new_job is None:
        raise HTTPException(409, "original shipped payload is missing")
    new_id = new_job["job_id"]
    db.add_event(
        "job.restart",
        job_id=new_id,
        state=new_job["state"],
        reason="operator",
        payload={"source_job_id": job_id, "source_state": job["state"], "priority": priority},
    )
    db.add_event(
        "job.restarted",
        job_id=job_id,
        state=job["state"],
        reason="operator",
        payload={"new_job_id": new_id},
    )
    return {
        "job_id": new_id,
        "state": new_job["state"],
        "source_job_id": job_id,
        "reservation": new_job["reservation"],
        "draining": _draining(),
    }


@app.get("/api/jobs/{job_id}/logs")
async def read_logs(job_id: str, offset: int = 0, wait_s: float = 0.0) -> Response:
    """Read or long-poll a job log without occupying a worker thread.

    Log followers commonly wait with no new bytes available.  Keeping that wait
    on the event loop prevents a group of followers from exhausting Starlette's
    bounded sync-worker pool and starving health, telemetry, and job endpoints.
    """
    offset = max(0, offset)  # negative seek would 500
    path = data_dir() / "logs" / f"{job_id}.log"
    deadline = time.time() + min(wait_s, 15.0)
    while True:
        size = path.stat().st_size if path.exists() else 0
        if size > offset or time.time() >= deadline:
            chunk = b""
            if size > offset:
                with open(path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read()
            return Response(
                content=chunk,
                media_type="application/octet-stream",
                headers={"X-Next-Offset": str(offset + len(chunk))},
            )
        await asyncio.sleep(0.5)


@app.post("/api/jobs/{job_id}/priority")
def set_priority(job_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Re-prioritize a QUEUED job (the UI's drag-to-reorder). Same clamp as submit."""
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job["state"] != QUEUED:
        raise HTTPException(409, f"job is {job['state']!r}; only queued jobs re-prioritize")
    try:
        priority = max(-100, min(100, int(payload.get("priority"))))
    except (TypeError, ValueError):
        raise HTTPException(422, "priority must be an int") from None
    db.update_job(job_id, priority=priority)
    db.add_event(
        "job.priority",
        job_id=job_id,
        state=QUEUED,
        payload={"priority": priority},
    )
    _wake_all()  # queue order changed -> waiting lease polls re-check
    return {"job_id": job_id, "priority": priority}


@app.post("/api/jobs/{job_id}/meta")
def set_meta(job_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Attach run metadata (mx run_id, tracking project, …) — the harness phones this
    home best-effort so the UI can deep-link MLflow/Trackio/MinIO. Merged inside one
    DB transaction (a concurrent links-scrape write must not be lost). LAN-trusted like
    cancel/submit; scalar values only, capped."""
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    patch = {
        k: v
        for k, v in payload.items()
        if isinstance(k, str) and isinstance(v, (str, int, float, bool)) and len(str(v)) < 512
    }
    if len(dict(job.get("meta") or {})) + len(patch) > 32:
        raise HTTPException(422, "meta too large (32 keys max)")
    meta = db.merge_meta(job_id, patch)
    db.add_event(
        "job.meta",
        job_id=job_id,
        state=job.get("state"),
        payload={"patch": patch, "meta": meta},
    )
    return {"ok": True, "meta": meta}


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str, payload: dict[str, Any] | None = Body(None)) -> dict[str, Any]:
    payload = payload or {}
    _require(isinstance(payload, dict), "cancel payload must be an object")
    _require(set(payload) <= {"reason"}, "cancel payload may contain only reason")
    reason = _optional_audit_message(payload.get("reason"), "reason")
    job = db.job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if reason is not None and job["state"] in ({QUEUED} | ACTIVE_STATES):
        db.merge_meta(job_id, {"cancellation_reason": reason})
    if job["state"] == QUEUED:
        db.update_job(job_id, state="cancelled", finished_ts=time.time())
        db.add_event(
            "job.cancelled",
            job_id=job_id,
            state="cancelled",
            reason=reason or "user",
        )
        response: dict[str, Any] = {"state": "cancelled"}
        if reason is not None:
            response["reason"] = reason
        return response
    if job["state"] in ACTIVE_STATES:
        db.update_job(job_id, kill_requested=1)
        db.add_event(
            "job.cancel_requested",
            job_id=job_id,
            host=job.get("assigned_host"),
            state=job["state"],
            reason=reason or "user",
        )
        response = {
            "state": job["state"],
            "kill_requested": True,
        }
        if reason is not None:
            response["reason"] = reason
        return response
    return {"state": job["state"]}


@app.get("/api/hosts")
def hosts() -> list[dict[str, Any]]:
    out = []
    for h in db.host_rows():
        active = db.active_jobs_on(h["name"])
        h["active_jobs"] = [
            {k: j[k] for k in ("job_id", "experiment", "state", "reservation", "started_ts")}
            for j in active
        ]
        h["committed_ram_mb"] = sum(
            float((j.get("reservation") or {}).get("ram_mb") or 0) for j in active
        )
        h["eta_s"] = host_eta_s(h, active)
        h["models"] = [
            {
                k: r.get(k)
                for k in (
                    "model",
                    "kind",
                    "bytes",
                    "path",
                    "artifact_id",
                    "artifact_kind",
                    "variant",
                    "mount",
                    "manifest_sha256",
                    "content_hash",
                    "source_model",
                    "locator",
                )
            }
            for r in db.inventory(h["name"])
        ]
        from .reservation import margin_for, vram_margin_for

        h["admission"] = {
            "ram_margin_mb": margin_for(h),
            "vram_margin_mb": vram_margin_for(h),
            "enabled": (h.get("limits") or {}).get("enabled", True),
            "max_concurrent": (h.get("limits") or {}).get("max_concurrent"),
        }
        out.append(h)
    return out


_HOST_LIMIT_KEYS = {
    "ram_margin_mb",
    "ram_margin_fraction",
    "vram_margin_mb",
    "max_concurrent",
    "enabled",
}


def _host_or_404(name: str) -> dict[str, Any]:
    for h in db.host_rows():
        if h["name"] == name:
            return h
    raise HTTPException(404, "unknown host")


def _host_limits_view(host: dict[str, Any]) -> dict[str, Any]:
    from .reservation import margin_for, vram_margin_for

    limits = host.get("limits") or {}
    return {
        "host": host["name"],
        "limits": limits,
        "effective": {
            "ram_margin_mb": margin_for(host),
            "vram_margin_mb": vram_margin_for(host),
            "max_concurrent": limits.get("max_concurrent"),
            "enabled": limits.get("enabled", True),
        },
    }


@app.get("/api/hosts/{name}/limits")
def get_host_limits(name: str) -> dict[str, Any]:
    return _host_limits_view(_host_or_404(name))


@app.put("/api/hosts/{name}/limits")
def put_host_limits(name: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Operator limits, persisted in the settings table and applied LIVE — no container
    restart. A key set to null clears it back to defaults; PUT replaces the whole set."""
    _host_or_404(name)
    unknown = set(payload) - _HOST_LIMIT_KEYS
    _require(not unknown, f"unknown limit keys: {sorted(unknown)}")
    limits: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if key == "enabled":
            _require(isinstance(value, bool), "enabled must be a boolean")
            limits[key] = value
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise HTTPException(422, f"{key} must be a number") from None
        _require(math.isfinite(num), f"{key} must be finite")
        if key == "ram_margin_mb":
            # The never-crash floor: at least 1GB of every host stays unreservable.
            _require(num >= 1024, "ram_margin_mb must be >= 1024 (1GB host floor)")
        elif key == "ram_margin_fraction":
            _require(0.0 <= num < 1.0, "ram_margin_fraction must be in [0, 1)")
        elif key == "vram_margin_mb":
            _require(num >= 0, "vram_margin_mb must be >= 0")
        elif key == "max_concurrent":
            _require(num >= 1 and num == int(num), "max_concurrent must be an integer >= 1")
            limits[key] = int(num)
            continue
        limits[key] = num
    db.set_setting(f"host_limits:{name}", limits)
    db.add_event("host.limits", host=name, reason="operator", payload={"limits": limits})
    _refresh_queue_admission_details()
    _wake_all()
    return _host_limits_view(_host_or_404(name))


@app.delete("/api/hosts/{name}")
def delete_host(name: str) -> dict[str, Any]:
    """Forget a host row. For stale duplicates after an agent rename (e.g. the same Mac
    registered as both ``jakes-macbook-pro`` and ``mbp1``). Refused while the row still
    has active jobs, or queued jobs pinned to it via needs.host (they would strand
    unschedulable forever). If the agent is merely asleep, it re-registers on wake via
    the telemetry ``reregister`` signal."""
    hosts = {h["name"] for h in db.host_rows()}
    if name not in hosts:
        raise HTTPException(404, "unknown host")
    active = db.active_jobs_on(name)
    if active:
        raise HTTPException(409, f"host has {len(active)} active job(s)")
    pinned = [j for j in db.jobs(state=QUEUED) if (j.get("needs") or {}).get("host") == name]
    if pinned:
        ids = ", ".join(j["job_id"] for j in pinned[:5])
        raise HTTPException(
            409, f"{len(pinned)} queued job(s) pinned to this host ({ids}) — cancel them first"
        )
    db.delete_host(name)
    db.add_event("host.delete", host=name, reason="operator")
    _refresh_queue_admission_details()
    return {"ok": True, "deleted": name}


# ------------------------------------------------------------------- model inventory
# Warm-cache inventory: which host already has which model or store bytes.
@app.post("/api/agents/{host}/inventory")
def inventory(host: str, request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    _require_registered_agent_auth(request, host)
    rows = payload.get("models")
    _require(isinstance(rows, list), "models must be a list")
    db.replace_inventory(host, rows)
    db.add_event(
        "host.inventory",
        host=host,
        payload={"models": rows, "n": len(rows)},
    )
    _refresh_queue_admission_details()
    return {"ok": True, "n": len(rows)}


@app.get("/api/models")
def models(host: str | None = None) -> list[dict[str, Any]]:
    return db.inventory(host)


@app.get("/api/calibration")
def calibration() -> list[dict[str, Any]]:
    """Per-task-family est-vs-peak ratios — tune estimate_activation_mb from these."""
    return db.calibration()


@app.get("/api/history/{client_run_id}")
def history(client_run_id: str, family_key: str | None = None) -> dict[str, Any]:
    """Read-only reservation history for client-side semantic preflight.

    The client uses the n-aware exact basis when available and may optionally request the
    generalized family history. This endpoint exposes existing scheduler measurements; it does
    not admit, reserve, or mutate a job.
    """
    return {
        "exact": db.history_for(client_run_id),
        "exact_stats": db.exact_stats(client_run_id),
        "generalized": db.history_stats(family_key) if family_key else None,
    }


@app.get("/api/scheduler-disk")
def scheduler_disk() -> dict[str, Any]:
    """Free space on the zima data volume (the scheduler's own host)."""

    u = shutil.disk_usage(str(data_dir()))
    return {"total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1)}
