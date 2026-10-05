"""orchestration — crash-safe subprocess stage runner with RAM/timeout guards.

Generalized from ``experiments/stress-suite/run_stress_suite.py`` so the lab runner and the
stress suite share one copy of "run a stage in its own process, poll the process-tree RSS, kill
on ``--ram-limit-mb`` or ``--timeout-s``, capture the log". Each stage gets a clean peak RSS and
one crash never takes down the suite (CLAUDE.md: monitor RAM/CPU on every job).
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .diagnostics import execution_diagnostics, failure_record, log_tail
from .os_limit import cleanup_strict_scope, prepare_os_memory_limit, strict_scope_oom_killed


@dataclass
class StageRun:
    cmd: list[str]
    returncode: int
    status: str  # ok | failed | killed_ram | timeout
    elapsed_s: float
    peak_rss_mb: float
    log_path: str | None = None
    os_memory_limit_backend: str = "off"
    failure: dict | None = None
    diagnostics: dict | None = None


def _tree_rss_mb(pid: int) -> float:
    try:
        import psutil
    except Exception:
        return 0.0
    try:
        proc = psutil.Process(pid)
        procs = [proc] + proc.children(recursive=True)
        return sum(p.memory_info().rss for p in procs if p.is_running()) / (1024 * 1024)
    except Exception:
        return 0.0


def run_stage(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    log_path: Path | None = None,
    ram_limit_mb: float | None = None,
    timeout_s: float | None = None,
    poll_s: float = 1.0,
    inherit_output: bool = False,
    on_spawn=None,
    os_memory_limit_mode: str = "off",
) -> StageRun:
    """Run ``cmd`` as a child process group, polling RSS; SIGKILL on an observed breach.

    ``inherit_output`` streams the child to the parent's stdout/stderr (interactive
    guarded runs) instead of a log file / devnull. ``on_spawn(pid)`` is called once with
    the child's pid (fleet-visibility heartbeats sample the tree from it). RSS enforcement
    is sampled at ``poll_s`` (1 s by default), so a shorter-lived spike can escape observation;
    use an OS limit/cgroup when a strict allocation ceiling is required.
    """
    cmd = [str(c) for c in cmd]
    full_env = {**os.environ, **(env or {})}
    log_handle = None
    start = time.time()
    peak = 0.0
    status = "ok"
    launch = None
    proc: subprocess.Popen | None = None
    failure: dict | None = None
    phases: list[dict[str, object]] = []

    def phase(name: str) -> None:
        phases.append({"name": name, "ts": time.time()})

    try:
        if log_path is not None:
            log_handle = open(log_path, "w")
        phase("spawn")
        launch = prepare_os_memory_limit(
            cmd,
            limit_mb=ram_limit_mb,
            mode=os_memory_limit_mode,
        )
        if inherit_output:
            stdout, stderr = None, None  # inherit the parent's terminal
        else:
            stdout = log_handle or subprocess.DEVNULL
            stderr = subprocess.STDOUT if log_handle else subprocess.DEVNULL
        proc = subprocess.Popen(
            launch.command, cwd=str(cwd) if cwd else None, env=full_env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        phase("running")
        if on_spawn is not None:
            try:
                on_spawn(proc.pid)
            except Exception:  # noqa: BLE001
                pass
        phase("monitoring")
        while True:
            ret = proc.poll()
            peak = max(peak, _tree_rss_mb(proc.pid))
            if ret is not None:
                if strict_scope_oom_killed(launch, ret):
                    status = "killed_ram"
                    failure = failure_record(
                        kind="killed_ram",
                        phase="monitoring",
                        message="strict cgroup scope exited by SIGKILL at memory.max",
                        command=cmd,
                        cwd=cwd,
                        returncode=ret,
                        resources={"peak_rss_mb": round(peak, 1)},
                    )
                else:
                    status = "ok" if ret == 0 else "failed"
                    if status == "failed":
                        failure = failure_record(
                            kind="process_exit",
                            phase="process",
                            message=f"child exited with return code {ret}",
                            command=cmd,
                            cwd=cwd,
                            returncode=ret,
                            resources={"peak_rss_mb": round(peak, 1)},
                        )
                break
            if ram_limit_mb is not None and peak > ram_limit_mb:
                status = "killed_ram"
                failure = failure_record(
                    kind="killed_ram",
                    phase="monitoring",
                    message=f"tree RSS {peak:.0f}MB > ceiling {ram_limit_mb:.0f}MB",
                    command=cmd,
                    cwd=cwd,
                    resources={"peak_rss_mb": round(peak, 1), "ceiling_mb": ram_limit_mb},
                )
            elif timeout_s is not None and (time.time() - start) > timeout_s:
                status = "timeout"
                failure = failure_record(
                    kind="timeout",
                    phase="monitoring",
                    message=f"exceeded {timeout_s:.0f}s",
                    command=cmd,
                    cwd=cwd,
                    resources={"peak_rss_mb": round(peak, 1), "timeout_s": timeout_s},
                )
            else:
                time.sleep(poll_s)
                continue
            phase("terminating")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            proc.wait()
            cleanup_strict_scope(launch)
            break
    except Exception as exc:  # noqa: BLE001 — return a diagnosable failed stage
        status = "failed"
        failure = failure_record(
            kind="spawn_error" if proc is None else "runner_exception",
            phase="spawn" if proc is None else "monitoring",
            message="local stage runner failed",
            command=cmd,
            cwd=cwd,
            exception=exc,
            resources={"peak_rss_mb": round(peak, 1)},
        )
        if proc is not None and proc.poll() is None:
            phase("terminating")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=10)
            except Exception:
                pass
    finally:
        if log_handle is not None:
            log_handle.close()
    phase("finished")
    if failure is not None and log_path is not None:
        tail = log_tail(log_path)
        if tail is not None:
            failure["log_tail"] = tail
    diagnostics = execution_diagnostics(
        command=cmd,
        cwd=cwd,
        phase=phases[-1]["name"] if phases else "finished",
        phases=phases,
        log_path=log_path,
    )
    return StageRun(
        cmd=cmd,
        returncode=proc.returncode if proc is not None and proc.returncode is not None else -1,
        status=status,
        elapsed_s=round(time.time() - start, 2),
        peak_rss_mb=round(peak, 1),
        log_path=str(log_path) if log_path else None,
        os_memory_limit_backend=launch.backend if launch is not None else "off",
        failure=failure,
        diagnostics=diagnostics,
    )
