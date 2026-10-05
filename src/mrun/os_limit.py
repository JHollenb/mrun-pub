"""Strict operating-system memory boundaries for launched jobs.

Linux cgroup v2 is the only supported strict backend.  The command is placed in a
transient systemd user scope and a bootstrap running *inside* that scope verifies the
effective ``memory.max`` and ``memory.swap.max`` files before executing the workload.

Do not substitute ``RLIMIT_AS``: it limits virtual address space, not resident memory,
and breaks CUDA plus mmap-backed model stores.  macOS ``RLIMIT_RSS`` is advisory, so it
is deliberately not advertised as a strict backend.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_MODES = {"off", "auto", "required"}
CGROUP_V2_BACKEND = "linux-cgroup-v2"


@dataclass(frozen=True)
class OSMemoryLimitLaunch:
    command: list[str]
    backend: str
    strict: bool
    unit_name: str | None = None


def _mode(value: str | None) -> str:
    normalized = str(value or "off").strip().lower()
    if normalized not in _MODES:
        raise ValueError(
            f"os memory limit mode must be one of {sorted(_MODES)}, got {value!r}"
        )
    return normalized


def _memory_controller_available(cgroup_root: Path) -> bool:
    try:
        return "memory" in (cgroup_root / "cgroup.controllers").read_text().split()
    except OSError:
        return False


def _unit_name(identity: str | None) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", identity or uuid.uuid4().hex).strip("-.")
    return f"mrun-{safe[:48] or uuid.uuid4().hex}.scope"


def prepare_os_memory_limit(
    command: Sequence[str],
    *,
    limit_mb: float | None,
    mode: str = "off",
    identity: str | None = None,
    system: str | None = None,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    systemd_run: str | None = None,
) -> OSMemoryLimitLaunch:
    """Return the command required to launch under the selected OS memory boundary.

    ``auto`` activates cgroup v2 on Linux when the local prerequisites are visible and
    otherwise keeps the sampled guard. ``required`` raises instead of falling back.
    The in-scope bootstrap performs the final, fail-closed verification; merely finding
    ``systemd-run`` is not treated as proof that the controller was delegated.
    """
    selected = _mode(mode)
    raw = [str(part) for part in command]
    if selected == "off" or limit_mb is None:
        return OSMemoryLimitLaunch(raw, "off", False)
    if limit_mb <= 0:
        raise ValueError("strict OS memory limit must be positive")

    host_system = system or platform.system()
    runner = systemd_run if systemd_run is not None else shutil.which("systemd-run")
    supported = (
        host_system == "Linux"
        and runner is not None
        and _memory_controller_available(cgroup_root)
    )
    if not supported:
        if selected == "required":
            raise RuntimeError(
                "strict OS memory limit requested, but Linux cgroup v2 + systemd-run "
                "is unavailable; macOS RLIMIT_RSS is advisory and RLIMIT_AS is unsafe "
                "for mmap/CUDA workloads"
            )
        return OSMemoryLimitLaunch(raw, "sampled-only", False)

    limit_bytes = int(limit_mb * 1024 * 1024)
    unit = _unit_name(identity)
    wrapped = [
        runner,
        "--user",
        "--scope",
        "--quiet",
        f"--unit={unit}",
        "--property=MemoryAccounting=yes",
        f"--property=MemoryMax={limit_bytes}",
        "--property=MemorySwapMax=0",
        "--property=OOMPolicy=kill",
        "--",
        sys.executable,
        "-m",
        "mrun.os_limit",
        "exec",
        "--limit-bytes",
        str(limit_bytes),
        "--",
        *raw,
    ]
    return OSMemoryLimitLaunch(wrapped, CGROUP_V2_BACKEND, True, unit)


def strict_scope_oom_killed(
    launch: OSMemoryLimitLaunch,
    returncode: int,
    *,
    run=subprocess.run,
) -> bool:
    """Read systemd's authoritative scope result, then release failed-unit state.

    ``systemd-run --scope`` returns 255 for a cgroup OOM on the production host, so
    return-code heuristics alone cannot distinguish ``memory.max`` from launch failure.
    """
    if not launch.strict or not launch.unit_name:
        return False
    systemctl = shutil.which("systemctl") or "systemctl"
    result = ""
    try:
        completed = run(
            [
                systemctl,
                "--user",
                "show",
                launch.unit_name,
                "--property=Result",
                "--value",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            result = completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        try:
            run(
                [systemctl, "--user", "reset-failed", launch.unit_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    if result:
        return result == "oom-kill"
    return returncode in (-9, 137)


def cleanup_strict_scope(launch: OSMemoryLimitLaunch, *, run=subprocess.run) -> None:
    """Release retained transient scope state after an externally classified stop."""
    if not launch.strict or not launch.unit_name:
        return
    systemctl = shutil.which("systemctl") or "systemctl"
    try:
        run(
            [systemctl, "--user", "reset-failed", launch.unit_name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _current_cgroup_path(proc_cgroup: Path = Path("/proc/self/cgroup")) -> str:
    for line in proc_cgroup.read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            return fields[2]
    raise RuntimeError("process is not in a unified cgroup v2 hierarchy")


def verify_current_cgroup_limit(
    expected_bytes: int,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
) -> Path:
    """Fail unless this process is inside the requested strict cgroup boundary."""
    relative = _current_cgroup_path(proc_cgroup).lstrip("/")
    root = cgroup_root.resolve()
    scope = (root / relative).resolve()
    if scope != root and root not in scope.parents:
        raise RuntimeError(f"cgroup path escaped controller root: {scope}")
    try:
        raw_max = (scope / "memory.max").read_text().strip()
        raw_swap = (scope / "memory.swap.max").read_text().strip()
        raw_oom_group = (scope / "memory.oom.group").read_text().strip()
    except OSError as exc:
        raise RuntimeError(f"cgroup memory controller is not active for {scope}") from exc
    if raw_max == "max" or int(raw_max) > expected_bytes:
        raise RuntimeError(
            f"cgroup memory.max={raw_max!r}, expected no more than {expected_bytes}"
        )
    if raw_swap != "0":
        raise RuntimeError(f"cgroup memory.swap.max={raw_swap!r}, expected '0'")
    if raw_oom_group != "1":
        raise RuntimeError(f"cgroup memory.oom.group={raw_oom_group!r}, expected '1'")
    return scope


def _exec_in_verified_scope(limit_bytes: int, command: list[str]) -> None:
    if not command:
        raise RuntimeError("strict memory bootstrap received an empty command")
    verify_current_cgroup_limit(limit_bytes)
    os.environ["MRUN_OS_MEMORY_LIMIT_BACKEND"] = CGROUP_V2_BACKEND
    os.environ["MRUN_OS_MEMORY_LIMIT_BYTES"] = str(limit_bytes)
    os.execvpe(command[0], command, os.environ)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mrun.os_limit")
    sub = parser.add_subparsers(dest="action", required=True)
    execute = sub.add_parser("exec")
    execute.add_argument("--limit-bytes", type=int, required=True)
    execute.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    _exec_in_verified_scope(args.limit_bytes, command)
    return 127  # os.execvpe does not return


if __name__ == "__main__":  # pragma: no cover - exercised by Linux integration
    raise SystemExit(_main())
