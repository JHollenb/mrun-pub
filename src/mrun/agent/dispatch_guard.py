"""Dispatch-time guards for the host agent — the two things admission cannot know.

Admission (server-side) validates a job's declared reservation against other *declared*
reservations. Two real-world gaps remain, both measured on beast 2026-07-17:

GAP 1 — the GPU may be *actually* occupied at dispatch even though no declared
reservation says so: a finishing job's CUDA context is not yet released, a cross-session
process holds memory, or a context leaked. Launching into it is an instant CUDA OOM
(measured: 5 jobs dispatched into a GPU holding 8.7-12.5GB → died in <0.1s). So right
before launching any job that declares ``vram_mb>0`` we query the GPU's *actual* free
VRAM and, if it is short, HOLD the lease with backoff rather than launch — and only if
still blocked past a window do we release the lease (never a hard FAILED — the job
lapses cleanly to ``lost`` and can be re-driven).

GAP 2 — payload-download race: on an idle host the scheduler can hand out a lease the
instant a job is submitted, before the submit's payload PUT has landed. The agent's
single GET then 404s and the job dies in ~0.03s with "payload download failed: HTTP 404".
Retrying the GET with backoff closes the race (and retires the keepalive/force-queue
workaround family).

The core routines take injected ``query`` / ``fetch`` / ``sleep`` / ``clock`` / ``log``
callables so they unit-test with fakes and carry zero import weight; the thin
``guard_vram_or_release`` / real-fetch adapters wire them to nvml + the Api client.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

# ------------------------------------------------------------------- GAP 1: VRAM


@dataclass
class GuardConfig:
    """Tunables for the dispatch-time VRAM guard (all overridable via AgentConfig)."""

    safety_margin_mb: float = 500.0
    hold_window_s: float = 120.0
    # backoff between actual-free-VRAM re-checks while holding; last value repeats until
    # the window is spent.
    backoff_s: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 60.0)


@dataclass
class GuardDecision:
    launch: bool
    reason: str
    attempts: int = 0
    waited_s: float = 0.0
    free_mb: float | None = None


def check_vram_before_launch(
    declared_vram_mb: float,
    query_free_mb: Callable[[], float | None],
    *,
    cfg: GuardConfig | None = None,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    log: Callable[[str], None] | None = None,
    job_id: str = "job",
) -> GuardDecision:
    """Decide whether it is safe to launch a job needing ``declared_vram_mb`` of VRAM.

    Polls ``query_free_mb`` (ACTUAL free VRAM incl. other processes' contexts). Launches
    as soon as ``free >= declared + safety_margin``. Otherwise HOLDS with backoff, logging
    every hold decision, up to ``hold_window_s``; if still short past the window, returns a
    ``launch=False`` (requeue/release) decision. ``declared<=0`` launches immediately
    without querying; an un-queryable GPU fails *open* (launch) — matching the pre-guard
    behaviour, never worse.
    """
    cfg = cfg or GuardConfig()
    _log = log or (lambda _m: None)
    margin = cfg.safety_margin_mb
    need = declared_vram_mb + margin

    if declared_vram_mb <= 0:
        return GuardDecision(launch=True, reason="no vram declared", attempts=0)

    start = clock()
    attempt = 0
    while True:
        attempt += 1
        free = query_free_mb()
        if free is None:
            # Cannot determine actual free VRAM (no nvml / nvidia-smi). The pre-guard
            # world launched blind here, so fail open — but say so loudly.
            _log(
                f"{job_id} VRAM-UNQUERYABLE declared={declared_vram_mb:.0f}MB "
                f"— launching blind (no nvml/nvidia-smi)"
            )
            return GuardDecision(
                launch=True,
                reason="vram unqueryable; launching blind",
                attempts=attempt,
                waited_s=clock() - start,
                free_mb=None,
            )
        if free >= need:
            waited = clock() - start
            _log(
                f"{job_id} VRAM-OK declared={declared_vram_mb:.0f}MB free={free:.0f}MB "
                f"(margin={margin:.0f}MB, need={need:.0f}MB) after {waited:.1f}s "
                f"/ {attempt} check(s)"
            )
            return GuardDecision(
                launch=True, reason="vram available", attempts=attempt,
                waited_s=waited, free_mb=free,
            )

        elapsed = clock() - start
        remaining = cfg.hold_window_s - elapsed
        short = need - free
        if remaining <= 0:
            _log(
                f"{job_id} VRAM-RELEASE giving up after {elapsed:.1f}s "
                f"(window={cfg.hold_window_s:.0f}s): declared={declared_vram_mb:.0f}MB "
                f"free={free:.0f}MB short={short:.0f}MB — releasing lease "
                f"(job lapses to 'lost' and can be re-driven)"
            )
            return GuardDecision(
                launch=False,
                reason=(
                    f"actual free VRAM {free:.0f}MB < declared {declared_vram_mb:.0f}MB "
                    f"+ margin {margin:.0f}MB after {cfg.hold_window_s:.0f}s hold"
                ),
                attempts=attempt,
                waited_s=elapsed,
                free_mb=free,
            )

        idx = min(attempt - 1, len(cfg.backoff_s) - 1)
        backoff = min(cfg.backoff_s[idx], remaining)
        _log(
            f"{job_id} VRAM-HOLD attempt={attempt} declared={declared_vram_mb:.0f}MB "
            f"free={free:.0f}MB need={need:.0f}MB short={short:.0f}MB "
            f"waited={elapsed:.1f}s backoff={backoff:.1f}s"
        )
        sleep(backoff)


def actual_free_vram_mb() -> float | None:
    """ACTUAL free VRAM on GPU 0 (MB), counting every process' usage — None if no GPU.

    Prefers nvml (already loaded by hostinfo); falls back to a ``nvidia-smi`` subprocess so
    the guard still works if pynvml is missing. Returns None (not 0) when there is no CUDA
    device or the query fails, so the caller can fail *open* rather than block forever.
    """
    try:
        from .hostinfo import _NVML

        if _NVML is not None:
            h = _NVML.nvmlDeviceGetHandleByIndex(0)
            return float(_NVML.nvmlDeviceGetMemoryInfo(h).free) / 1e6
    except Exception:  # noqa: BLE001 — fall through to nvidia-smi
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5.0,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0].strip())
    except Exception:  # noqa: BLE001
        pass
    return None


def guard_vram_or_release(job_run) -> bool:  # noqa: ANN001 — JobRun, avoid import cycle
    """Adapter: gate ``job_run``'s launch on ACTUAL free VRAM. True = safe to launch;
    False = the lease was released (caller must NOT launch and must NOT post a terminal
    'failed' event — the job lapses to 'lost' via lease-expiry)."""
    import time

    res = job_run.res
    declared = float(getattr(res, "vram_mb", 0.0) or 0.0)
    if declared <= 0:
        return True
    cfg = guard_config_from(job_run.cfg)
    decision = check_vram_before_launch(
        declared,
        actual_free_vram_mb,
        cfg=cfg,
        sleep=time.sleep,
        clock=time.monotonic,
        log=lambda m: print(f"mrun-agent: {m}", flush=True),
        job_id=job_run.job_id,
    )
    # Preserve the final dispatch decision for the agent's structured phase event.  The
    # old path only printed this to the agent console, which disappeared when the lease
    # later expired as ``lost``.
    job_run.dispatch_guard = {
        "launch": decision.launch,
        "reason": decision.reason,
        "attempts": decision.attempts,
        "waited_s": round(decision.waited_s, 3),
        "free_mb": decision.free_mb,
        "declared_vram_mb": declared,
        "safety_margin_mb": cfg.safety_margin_mb,
    }
    return decision.launch


def guard_config_from(agent_cfg) -> GuardConfig:  # noqa: ANN001 — AgentConfig, optional fields
    """Build a GuardConfig from an AgentConfig, honouring any overrides it carries."""
    backoff = getattr(agent_cfg, "vram_backoff_s", None)
    return GuardConfig(
        safety_margin_mb=float(getattr(agent_cfg, "vram_safety_margin_mb", 500.0)),
        hold_window_s=float(getattr(agent_cfg, "vram_hold_window_s", 120.0)),
        backoff_s=tuple(backoff) if backoff else GuardConfig.backoff_s,
    )


# ---------------------------------------------------------------- GAP 2: payload


@dataclass
class PayloadRetryConfig:
    attempts: int = 6
    window_s: float = 30.0
    base_delay_s: float = 1.0
    max_delay_s: float = 8.0
    # statuses (besides 5xx, always retried) that mean "not landed yet, try again"
    retry_statuses: tuple[int, ...] = field(default_factory=lambda: (404, 425, 429))


def fetch_payload_with_retry(
    fetch: Callable[[], tuple[int, bytes]],
    *,
    cfg: PayloadRetryConfig | None = None,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    log: Callable[[str], None] | None = None,
    job_id: str = "job",
) -> bytes:
    """GET the payload with backoff, absorbing the instant-dispatch race.

    ``fetch()`` returns ``(status, body)``. Returns the body on any 2xx. Retries on 404
    (payload PUT not landed yet) / 425 / 429 / 5xx and transport exceptions with
    exponential backoff, up to ``attempts`` tries within ``window_s``. Any other 4xx
    (real error) fails fast. On HTTP exhaustion raises ``RuntimeError`` keeping the
    legacy ``payload download failed: HTTP <status>`` prefix so existing log-scrapers
    still match.
    """
    cfg = cfg or PayloadRetryConfig()
    _log = log or (lambda _m: None)
    start = clock()
    delay = cfg.base_delay_s
    status: int | None = None
    last_error: BaseException | None = None
    for attempt in range(1, cfg.attempts + 1):
        try:
            status, body = fetch()
            last_error = None
        except (ConnectionError, TimeoutError, OSError) as exc:
            status, body = None, b""
            last_error = exc
        if status is not None and 200 <= status < 300:
            if attempt > 1:
                _log(
                    f"{job_id} payload GET ok on attempt {attempt} "
                    f"after {clock() - start:.1f}s ({len(body)} bytes)"
                )
            return body
        retryable = last_error is not None or (
            status is not None and (status in cfg.retry_statuses or 500 <= status < 600)
        )
        if not retryable:
            raise RuntimeError(f"payload download failed: HTTP {status}")
        elapsed = clock() - start
        if attempt >= cfg.attempts or elapsed >= cfg.window_s:
            break
        wait = min(delay, cfg.max_delay_s, cfg.window_s - elapsed)
        label = (
            f"transport {type(last_error).__name__}: {last_error}"
            if last_error is not None
            else f"HTTP {status}"
        )
        _log(
            f"{job_id} payload not ready ({label}); retry {attempt}/{cfg.attempts} "
            f"in {wait:.1f}s (elapsed {elapsed:.1f}s)"
        )
        sleep(wait)
        delay *= 2
    if last_error is not None:
        raise RuntimeError(
            f"payload download failed: transport {type(last_error).__name__}: "
            f"{last_error} after {cfg.attempts} attempts / {clock() - start:.1f}s"
        )
    raise RuntimeError(
        f"payload download failed: HTTP {status} after {cfg.attempts} attempts "
        f"/ {clock() - start:.1f}s"
    )


def payload_config_from(agent_cfg) -> PayloadRetryConfig:  # noqa: ANN001
    """Build a PayloadRetryConfig from an AgentConfig, honouring any overrides."""
    return PayloadRetryConfig(
        attempts=int(getattr(agent_cfg, "payload_retry_attempts", 6)),
        window_s=float(getattr(agent_cfg, "payload_retry_window_s", 30.0)),
    )
