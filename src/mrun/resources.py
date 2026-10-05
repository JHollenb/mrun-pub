"""Resource accounting and disk-space guards for experiment runs."""

from __future__ import annotations

import os
import resource
import shutil
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import default_hub_root
from .paths import models_root

GIB = 1024**3
DEFAULT_MIN_FREE_BYTES = 5 * GIB


@dataclass
class ResourceMetrics:
    wall_s: float
    cpu_s: float
    rss_peak_mb: float
    rss_current_mb: float
    disk_free_before_bytes: int
    disk_free_after_bytes: int
    output_bytes: int
    hf_cache_bytes: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def disk_free_bytes(path: str | Path) -> int:
    return int(shutil.disk_usage(_nearest_existing_path(path)).free)


def require_free_space(
    path: str | Path,
    *,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    label: str = "disk",
) -> int:
    free = disk_free_bytes(path)
    if free < min_free_bytes:
        raise RuntimeError(
            f"{label} has {format_bytes(free)} free, below required {format_bytes(min_free_bytes)}"
        )
    return free


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return int(root.stat().st_size)
    total = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        for name in files:
            item = Path(current) / name
            if item.is_symlink():
                continue
            try:
                total += int(item.stat().st_size)
            except OSError:
                continue
    return total


def output_size_bytes(paths: Iterable[str | Path]) -> int:
    return sum(directory_size_bytes(path) for path in paths)


def hf_cache_bytes() -> int:
    roots = {default_hub_root(), models_root()}
    return sum(directory_size_bytes(root) for root in roots)


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def rss_peak_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024 * 1024)
    return float(rss) / 1024


def rss_current_mb() -> float:
    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
            return float(resident_pages * os.sysconf("SC_PAGE_SIZE")) / (1024 * 1024)
        except (OSError, IndexError, ValueError):
            pass
    return rss_peak_mb()


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


class ResourceMonitor:
    """Context manager that records wall, CPU, RSS, disk, output, and HF-cache usage."""

    def __init__(
        self,
        *,
        disk_path: str | Path = ".",
        output_paths: Iterable[str | Path] = (),
    ) -> None:
        self.disk_path = Path(disk_path)
        self.output_paths = [Path(path) for path in output_paths]
        self._cpu_start = 0.0
        self._wall_start = 0.0
        self._disk_before = 0
        self.metrics: ResourceMetrics | None = None

    def __enter__(self) -> ResourceMonitor:
        self._cpu_start = cpu_seconds()
        self._wall_start = time.time()
        self._disk_before = disk_free_bytes(self.disk_path)
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.finish()

    def finish(self) -> ResourceMetrics:
        if self.metrics is None:
            self.metrics = ResourceMetrics(
                wall_s=float(time.time() - self._wall_start),
                cpu_s=float(cpu_seconds() - self._cpu_start),
                rss_peak_mb=rss_peak_mb(),
                rss_current_mb=rss_current_mb(),
                disk_free_before_bytes=int(self._disk_before),
                disk_free_after_bytes=disk_free_bytes(self.disk_path),
                output_bytes=output_size_bytes(self.output_paths),
                hf_cache_bytes=hf_cache_bytes(),
            )
        return self.metrics


def _nearest_existing_path(path: str | Path) -> Path:
    current = Path(path).expanduser()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current
