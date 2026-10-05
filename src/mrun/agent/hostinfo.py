"""Host capability + telemetry sampling for the agent."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil

from .. import __version__
from ..paths import models_root
from ..protocol import PROTOCOL_VERSION


def _nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


_NVML = _nvml()


def _smi_memory_mb() -> tuple[float, float] | None:
    """(total, free) MiB via nvidia-smi when the optional pynvml package is absent."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
        total, free = (float(value.strip()) for value in result.stdout.splitlines()[0].split(","))
        return total, free
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _smi_process_vram_mb(pids: set[int]) -> float:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total = 0.0
    for line in result.stdout.splitlines():
        try:
            pid_text, memory_text = line.split(",", maxsplit=1)
            if int(pid_text.strip()) in pids:
                total += float(memory_text.strip())
        except ValueError:
            continue
    return total


def disks() -> list[dict[str, Any]]:
    """Every real mounted filesystem: beast has several external drives + a 2TB internal;
    admission must check the mount that will actually take the bytes."""
    seen: dict[str, dict[str, Any]] = {}
    for part in psutil.disk_partitions(all=False):
        if not part.device.startswith("/dev"):
            continue
        if part.mountpoint.startswith(("/System/Volumes/", "/boot", "/private/var/vm")):
            if part.mountpoint != "/System/Volumes/Data":
                continue
        try:
            u = shutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        if u.total < 1e9:  # skip recovery/efi slivers
            continue
        # dedupe by device (APFS exposes one container many ways) — keep the shortest mount
        cur = seen.get(part.device)
        if cur is None or len(part.mountpoint) < len(cur["mount"]):
            seen[part.device] = {
                "mount": part.mountpoint,
                "total_gb": round(u.total / 1e9, 1),
                "free_gb": round(u.free / 1e9, 1),
            }
    return sorted(seen.values(), key=lambda d: d["mount"])


def models_mount() -> str:
    """The mount holding the models root (where downloads/qstore builds land)."""
    from ..paths import models_root

    root = str(models_root())
    mounts = [d["mount"] for d in disks()]
    best = "/"
    for m in mounts:
        if root.startswith(m.rstrip("/") + "/") or root == m:
            if len(m) > len(best):
                best = m
    return best


def _inventory_store_roots(configured_root: str | Path) -> list[Path]:
    """Return configured and conventional sibling roots containing derived stores.

    Beast historically kept source checkpoints under ``/mnt/big/llm-models`` but
    materialized QStores under the sibling ``/mnt/big/qstores`` directory.  The
    configured default only covers ``<models_root>/qstores``, so inventory silently
    omitted those valid variants.  Keep the explicit root authoritative while also
    discovering the sibling layout used by the existing fleet.
    """
    candidates = [Path(configured_root).expanduser()]
    model_roots = [Path(models_root())]
    model_roots.extend(
        Path(value).expanduser()
        for value in os.environ.get("LLM_MODELS_EXTRA_ROOT", "").split(os.pathsep)
        if value.strip()
    )
    for root in model_roots:
        candidates.extend((root / "qstores", root.parent / "qstores"))

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def model_inventory(host: str | None = None) -> list[dict[str, Any]]:
    """Advertise logical models plus every discovered local artifact variant.

    The legacy ``model``/``kind``/``bytes``/``path`` fields remain stable for scheduler
    placement.  Artifact IDs, variants, mounts and locators are additive metadata; MLflow
    on Zima remains the logical registry and Beast remains the source of local bytes.
    """
    from ..catalog import artifact_from_weights, scan_store_artifacts
    from ..models import REGISTRY, find_safetensors
    from ..paths import stores_root

    mounts = [row["mount"] for row in disks()]
    out: list[dict[str, Any]] = []
    for name, spec in REGISTRY.items():
        try:
            files = find_safetensors(spec)
        except Exception:  # noqa: BLE001
            files = []
        artifact = artifact_from_weights(files, model=name, mounts=mounts, host=host)
        if artifact is not None:
            out.append(artifact.as_dict())
    out.extend(
        artifact.as_dict()
        for store_root in _inventory_store_roots(stores_root())
        for artifact in scan_store_artifacts(
            store_root, registry=REGISTRY, mounts=mounts, host=host
        )
    )
    return out


def register_payload(
    host: str,
    *,
    guarded_custody: bool = False,
    payload_sandbox: bool = False,
) -> dict[str, Any]:
    smi_memory = _smi_memory_mb() if _NVML is None else None
    has_cuda = _NVML is not None or smi_memory is not None
    vram_total_mb = 0.0
    if has_cuda:
        if _NVML is not None:
            h = _NVML.nvmlDeviceGetHandleByIndex(0)
            vram_total_mb = _NVML.nvmlDeviceGetMemoryInfo(h).total / 1e6
        else:
            vram_total_mb = smi_memory[0]  # type: ignore[index]
    is_asi = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {
        "host": host,
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "caps": {
            "cuda": has_cuda,
            "mps": is_asi,
            "ane": is_asi,
            "cpu": True,
            # A binary-new agent without the independent credential remains a safe
            # ordinary-job agent; the scheduler must not send it guarded authority.
            "payload_custody_v2": guarded_custody,
            # The same binary boundary that protects shipped payload custody now
            # supports a separate least-authority debugger bearer for job children.
            "saturn_debug_credential_v1": guarded_custody,
            # Advertised only after the agent's bubblewrap/user-namespace execution
            # probe succeeds and an operator model-root allowlist is configured.
            "payload_sandbox_v1": payload_sandbox,
        },
        "ram_total_mb": psutil.virtual_memory().total / 1e6,
        "vram_total_mb": vram_total_mb,
        "cpu_threads": os.cpu_count() or 1,
        "disk_total_gb": shutil.disk_usage(str(os.path.expanduser("~"))).total / 1e9,
        "models_mount": models_mount(),
        "agent_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
    }


def mem_pressure_level() -> int | None:
    """Darwin memorystatus pressure: 1 normal, 2 warn, 4 critical. None elsewhere."""
    if platform.system() != "Darwin":
        return None
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return int(out.stdout.strip())
    except Exception:
        return None


EXTERNAL_PROC_MIN_RSS_MB = 512.0


def top_external_processes(
    exclude_pids: frozenset[int] | set[int] = frozenset(), limit: int = 5
) -> list[dict[str, Any]]:
    """Largest non-mrun processes (>=512MB RSS) — the out-of-band baseline breakdown.

    Admission already charges unattributed consumption as baseline (measured 44.7GB on
    beast, issue I9); this makes it VISIBLE on the dashboard instead of a mystery number.
    """
    me = os.getpid()
    out: list[dict[str, Any]] = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = p.info
            if info["pid"] == me or info["pid"] in exclude_pids:
                continue
            mem = info.get("memory_info")
            rss_mb = (mem.rss / 1e6) if mem else 0.0
            if rss_mb >= EXTERNAL_PROC_MIN_RSS_MB:
                out.append(
                    {"pid": info["pid"], "name": info.get("name") or "?", "rss_mb": round(rss_mb)}
                )
        except (psutil.Error, OSError):
            continue
    out.sort(key=lambda r: -r["rss_mb"])
    return out[:limit]


def telemetry_payload(
    running: list[dict[str, Any]],
    *,
    exclude_pids: frozenset[int] | set[int] = frozenset(),
    pressure_external: bool = False,
) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    vram_free_mb = None
    if _NVML is not None:
        h = _NVML.nvmlDeviceGetHandleByIndex(0)
        vram_free_mb = _NVML.nvmlDeviceGetMemoryInfo(h).free / 1e6
    else:
        smi_memory = _smi_memory_mb()
        if smi_memory is not None:
            vram_free_mb = smi_memory[1]
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = 0.0
    try:
        from ..engine.resident_worker import resident_executable_inventory

        executable_inventory = resident_executable_inventory()
    except Exception:  # noqa: BLE001 - telemetry must survive optional runtime imports
        executable_inventory = {}
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "ram_free_mb": vm.available / 1e6,
        "swap_used_mb": swap.used / 1e6,
        "swap_total_mb": swap.total / 1e6,
        "mem_pressure": mem_pressure_level(),
        "vram_free_mb": vram_free_mb,
        "disk_free_gb": shutil.disk_usage(str(os.path.expanduser("~"))).free / 1e9,
        "disks": disks(),
        "load1": load1,
        "running": running,
        "executable_inventory": executable_inventory,
        "top_external": top_external_processes(exclude_pids),
        "pressure_external": pressure_external,
    }


def tree_vram_mb(pids: set[int]) -> float:
    """VRAM used by any of ``pids`` (nvml compute procs); 0 when no GPU."""
    if not pids:
        return 0.0
    if _NVML is None:
        return _smi_process_vram_mb(pids)
    total = 0.0
    try:
        h = _NVML.nvmlDeviceGetHandleByIndex(0)
        for p in _NVML.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid in pids and p.usedGpuMemory:
                total += p.usedGpuMemory / 1e6
    except Exception:
        return 0.0
    return total
