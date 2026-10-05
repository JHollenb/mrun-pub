"""Fail-closed Linux mount/PID isolation for shipped job payloads.

The mrun agent may hold a broad agent credential while a job child receives only
job-scoped authority.  Environment filtering alone is not a security boundary for
same-UID code: an unsandboxed child can inspect the agent through ``/proc`` or read
its home-directory configuration.  This module builds a bubblewrap boundary that:

* creates new user, PID, IPC, UTS, and cgroup namespaces;
* mounts a fresh procfs, hides host homes, user runtime state, other job workdirs,
  unapproved data mounts, and sysfs;
* exposes only the current workdir writable and approved model roots read-only;
* gives shared package caches an ephemeral overlay so payload writes do not persist;
* exposes only selected accelerator device nodes; and
* disables nested user namespaces after setup.

Host networking remains shared because debugger workers must reach the scheduler
mailbox.  Consequently this is filesystem/process/credential isolation, not a
network-exfiltration boundary or protection from arbitrary hostile processes that
already execute outside mrun under the agent's Unix UID.
"""

from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PAYLOAD_SANDBOX_CAPABILITY = "payload_sandbox_v1"
PAYLOAD_SANDBOX_BACKEND = "linux-bubblewrap-v1"
PAYLOAD_SANDBOX_REQUEST_SCHEMA = "mrun-payload-sandbox-request-v1"


class PayloadSandboxError(RuntimeError):
    """Raised when a required job sandbox cannot be constructed exactly."""


@dataclass(frozen=True)
class PayloadSandboxLaunch:
    command: list[str]
    backend: str
    strict: bool
    profile: Mapping[str, Any]


def sandbox_requested(job: Mapping[str, Any]) -> bool:
    return bool((job.get("needs") or {}).get(PAYLOAD_SANDBOX_CAPABILITY))


@functools.lru_cache(maxsize=4)
def _probe_cached(bwrap: str, system: str) -> bool:
    if system != "Linux":
        return False
    try:
        completed = subprocess.run(
            [
                bwrap,
                "--die-with-parent",
                "--new-session",
                "--unshare-user",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
                "--unshare-cgroup",
                "--share-net",
                "--disable-userns",
                "--assert-userns-disabled",
                "--ro-bind",
                "/usr",
                "/usr",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--",
                "/usr/bin/true",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def payload_sandbox_supported(*, bwrap: str | None = None, system: str | None = None) -> bool:
    executable = bwrap or shutil.which("bwrap")
    return bool(executable) and _probe_cached(str(executable), system or platform.system())


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolved_existing(path: str | os.PathLike[str], *, label: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise PayloadSandboxError(f"{label} must be an absolute path")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise PayloadSandboxError(f"{label} does not exist: {raw}") from exc
    return resolved


def _approved_roots(requested: Sequence[str], allowed: Sequence[str]) -> tuple[Path, ...]:
    allowed_roots = tuple(
        _resolved_existing(value, label="configured sandbox root") for value in allowed
    )
    if not allowed_roots:
        raise PayloadSandboxError(
            "payload sandbox has no configured read-only model/artifact roots"
        )
    selected: list[Path] = []
    for raw in requested:
        target = _resolved_existing(raw, label="requested sandbox read path")
        candidates = [root for root in allowed_roots if _within(target, root)]
        if not candidates:
            raise PayloadSandboxError(f"requested sandbox path escapes configured roots: {target}")
        # Mount the narrowest configured root containing the requested object. This
        # preserves Hugging Face snapshot -> blobs symlink closure without exposing
        # unrelated host mounts or user homes.
        selected.append(max(candidates, key=lambda value: len(value.parts)))
    deduplicated: list[Path] = []
    for root in sorted(set(selected), key=lambda value: (len(value.parts), str(value))):
        if any(_within(root, parent) for parent in deduplicated):
            continue
        deduplicated.append(root)
    return tuple(deduplicated)


def _parents(path: Path) -> list[str]:
    values: list[str] = []
    current = path.parent
    while current != current.parent:
        values.append(str(current))
        current = current.parent
    return list(reversed(values))


def _directory_args(path: Path) -> list[str]:
    result: list[str] = []
    for parent in _parents(path):
        result.extend(("--dir", parent))
    if path.is_dir():
        result.extend(("--dir", str(path)))
    return result


def _request(job: Mapping[str, Any]) -> dict[str, Any]:
    config = job.get("config") or {}
    request = config.get("payload_sandbox") if isinstance(config, Mapping) else None
    if not isinstance(request, Mapping):
        raise PayloadSandboxError("sandbox-capable job needs config.payload_sandbox declaration")
    if request.get("schema") != PAYLOAD_SANDBOX_REQUEST_SCHEMA:
        raise PayloadSandboxError("unsupported payload sandbox request schema")
    if request.get("required") is not True:
        raise PayloadSandboxError("payload sandbox request must be fail-closed")
    paths = request.get("read_only_paths")
    if (
        not isinstance(paths, Sequence)
        or isinstance(paths, (str, bytes))
        or not paths
        or any(not isinstance(value, str) or not value for value in paths)
    ):
        raise PayloadSandboxError("payload sandbox request needs non-empty read_only_paths")
    return dict(request)


def prepare_payload_sandbox(
    command: Sequence[str],
    *,
    job: Mapping[str, Any],
    work_dir: Path,
    env_path: Path | None,
    allowed_read_roots: Sequence[str],
    shared_cache_paths: Sequence[str] = (),
    bwrap: str | None = None,
    uv_executable: str | None = None,
    system: str | None = None,
) -> PayloadSandboxLaunch:
    """Wrap one required job command in a strict bubblewrap profile."""

    raw = [str(value) for value in command]
    if not sandbox_requested(job):
        return PayloadSandboxLaunch(raw, "off", False, {"requested": False})
    if job.get("payload_kind") != "shipped":
        raise PayloadSandboxError("payload sandbox v1 requires a scheduler-sealed shipped payload")
    executable = bwrap or shutil.which("bwrap")
    host_system = system or platform.system()
    if not executable or not payload_sandbox_supported(bwrap=str(executable), system=host_system):
        raise PayloadSandboxError(
            "payload sandbox was required but Linux bubblewrap/user namespaces failed "
            "their execution probe"
        )
    request = _request(job)
    work = _resolved_existing(work_dir, label="job work directory")
    environment = (
        _resolved_existing(env_path, label="agent environment") if env_path is not None else None
    )
    roots = _approved_roots(request["read_only_paths"], allowed_read_roots)
    caches: list[Path] = []
    for raw_cache in shared_cache_paths:
        try:
            cache = _resolved_existing(raw_cache, label="shared package cache")
        except PayloadSandboxError:
            continue
        if any(_within(cache, root) for root in roots):
            caches.append(cache)

    uv = uv_executable or shutil.which("uv")
    uv_path = _resolved_existing(uv, label="uv executable") if uv is not None else None
    devices = sorted(
        {
            path.resolve()
            for pattern in ("/dev/nvidia*", "/dev/dri", "/dev/kfd")
            for path in Path("/").glob(pattern.lstrip("/"))
            if path.exists()
        },
        key=str,
    )

    wrapped = [
        str(executable),
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--share-net",
        "--disable-userns",
        "--assert-userns-disabled",
        "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/dev/shm",
        "--tmpfs",
        "/home",
        "--tmpfs",
        "/root",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/var/tmp",
        "--tmpfs",
        "/run",
        "--tmpfs",
        "/mnt",
        "--tmpfs",
        "/media",
        "--tmpfs",
        "/sys",
    ]
    mounts: list[tuple[Path, bool]] = [
        *((root, False) for root in roots),
        *((cache, False) for cache in caches),
    ]
    if environment is not None:
        mounts.append((environment, False))
    if uv_path is not None:
        mounts.append((uv_path, False))
    mounts.append((work, True))
    for source, writable in mounts:
        wrapped.extend(_directory_args(source))
        wrapped.extend(("--bind" if writable else "--ro-bind", str(source), str(source)))
    # Caches remain readable from the approved model root but every mutation goes
    # to an invisible tmpfs overlay discarded with the job namespace.
    for cache in caches:
        wrapped.extend(("--tmp-overlay", str(cache)))
    for device in devices:
        wrapped.extend(("--dev-bind", str(device), str(device)))
    wrapped.extend(("--chdir", str(work), "--", *raw))
    profile = {
        "schema": "mrun-payload-sandbox-profile-v1",
        "backend": PAYLOAD_SANDBOX_BACKEND,
        "requested": True,
        "strict": True,
        "network_namespace": "shared",
        "process_namespace": "private",
        "user_namespace": "private-and-nesting-disabled",
        "host_home_visible": False,
        "host_user_bus_visible": False,
        "other_job_workdirs_visible": False,
        "work_dir_writable": True,
        "read_only_root_count": len(roots),
        "ephemeral_cache_count": len(caches),
        "accelerator_device_count": len(devices),
    }
    return PayloadSandboxLaunch(
        command=wrapped,
        backend=PAYLOAD_SANDBOX_BACKEND,
        strict=True,
        profile=profile,
    )


__all__ = [
    "PAYLOAD_SANDBOX_BACKEND",
    "PAYLOAD_SANDBOX_CAPABILITY",
    "PAYLOAD_SANDBOX_REQUEST_SCHEMA",
    "PayloadSandboxError",
    "PayloadSandboxLaunch",
    "payload_sandbox_supported",
    "prepare_payload_sandbox",
    "sandbox_requested",
]
