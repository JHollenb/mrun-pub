"""RAM discipline — monitor RSS on every job; guard with a SIZE-MATCHED limit, not raw
"pages free" (CLAUDE.md §3.6). Extracted from the copies in ``routing_discriminator.py``,
``capability_harness.py``, ``model_structure_spectrometer.py``.

This project runs memory-starved (qwen forwards OOM-kill when two compete), so a guard that
aborts a job at a cooperative phase boundary *before* it thrashes swap is worth more than a
post-mortem. Set the ceiling with the ``RSS_LIMIT_MB`` env var and call
``check_rss("label")`` at each phase boundary. Use the subprocess/fleet poller for independent,
sampled containment between boundaries (1 s by default); it is not an OS-enforced allocator cap.
"""
from __future__ import annotations

import os
import resource
import sys

# Optional cooperative ceiling; a job calling check_rss() over this raises MemoryError instead
# of being SIGKILL'd mid-forward (which leaves no traceback). 0/unset ⇒ monitor only, no abort.
RSS_LIMIT_MB = float(os.environ.get("RSS_LIMIT_MB", "0")) or None


def _rss_limit_mb() -> float | None:
    """Resolve the current process ceiling.

    ``apply_plan`` installs ``RSS_LIMIT_MB`` after :mod:`mrun.guard` may already have
    been imported, so the environment must be read at the check boundary rather than
    only once at module import.  Keep the exported ``RSS_LIMIT_MB`` value as the
    fallback for callers that set it directly.
    """
    configured = os.environ.get("RSS_LIMIT_MB", "").strip()
    return (float(configured) or None) if configured else RSS_LIMIT_MB


def rss_mb() -> float:
    """Current process RSS in MB. Uses psutil if present (true current RSS); otherwise falls
    back to ``getrusage`` peak RSS (conservative — never under-reports), handling the
    macOS-bytes vs Linux-KB unit difference."""
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports ru_maxrss in BYTES; Linux in KILOBYTES.
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def rss_peak_mb() -> float:
    """Peak process RSS in MB (``getrusage`` high-water mark, monotonic). Use this for an
    OOM guard that should trip on the worst moment, not the instantaneous reading. Handles the
    macOS-bytes vs Linux-KB unit difference."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def rss_gb() -> float:
    return rss_mb() / 1024


def cpu_seconds() -> float:
    """Process CPU time (user + system) in seconds, from ``getrusage``. Pair with
    ``rss_peak_mb`` to record the cost of a run (both are RUSAGE_SELF high-water counters, so
    they are only clean when the run owns its process — see ``experiment.isolated_modes``)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def check_rss(label: str = "", *, limit_mb: float | None = None) -> float:
    """Return current RSS (MB); raise MemoryError if it exceeds the limit. The limit defaults
    to the ``RSS_LIMIT_MB`` env var; pass ``limit_mb`` to override per-call."""
    limit = limit_mb if limit_mb is not None else _rss_limit_mb()
    cur = rss_mb()
    if limit and cur > limit:
        raise MemoryError(f"RSS {cur:.0f} MB exceeds limit {limit:.0f} MB at {label!r}")
    return cur


class ram_guard:
    """Context manager that checks RSS on enter and exit and reports the delta.

        with ram_guard("content leg"):
            ...   # checks both boundaries; prints the entry-to-exit current-RSS delta
    """

    def __init__(self, label: str = "", *, limit_mb: float | None = None, verbose: bool = True):
        self.label = label
        self.limit_mb = limit_mb
        self.verbose = verbose
        self.start = 0.0

    def __enter__(self) -> ram_guard:
        self.start = check_rss(f"{self.label} (enter)", limit_mb=self.limit_mb)
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        end = rss_mb()
        if self.verbose:
            print(f"[ram] {self.label}: {self.start:.0f}→{end:.0f} MB "
                  f"({end - self.start:+.0f} MB)", flush=True)
        # Do not hide an exception already raised by the guarded phase.  On a clean
        # exit, however, the boundary is a real guard: allocations made inside the
        # phase must not be allowed to cross the configured ceiling silently.
        limit = self.limit_mb if self.limit_mb is not None else _rss_limit_mb()
        if exc_type is None and limit and end > limit:
            raise MemoryError(
                f"RSS {end:.0f} MB exceeds limit {limit:.0f} MB at "
                f"{self.label + ' (exit)'!r}"
            )


def set_thread_env(n: int = 4) -> None:
    """Pin the math-library thread pools (idempotent; only sets vars not already set). The run
    prefix usually sets OMP_NUM_THREADS already; call this for scripts launched without it."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))
