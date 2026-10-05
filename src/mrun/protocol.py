"""Wire protocol shared by server, agent and client — plain dataclasses + dicts.

Kept pydantic-free so importing the client never drags a web stack in. All payloads
travel as JSON dicts; these dataclasses are the single source of field names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PROTOCOL_VERSION = 2

# job states
QUEUED = "queued"
AWAITING_PAYLOAD = "awaiting_payload"
ASSIGNED = "assigned"
PREPARING = "preparing"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
KILLED_RAM = "killed_ram"
KILLED_VRAM = "killed_vram"
TIMEOUT = "timeout"
CANCELLED = "cancelled"
LOST = "lost"

TERMINAL_STATES = {SUCCEEDED, FAILED, KILLED_RAM, KILLED_VRAM, TIMEOUT, CANCELLED, LOST}
ACTIVE_STATES = {ASSIGNED, PREPARING, RUNNING}

# how a completed stage status maps to a job state
STAGE_TO_STATE = {
    "ok": SUCCEEDED,
    "failed": FAILED,
    "killed_ram": KILLED_RAM,
    "killed_vram": KILLED_VRAM,
    "timeout": TIMEOUT,
    "cancelled": CANCELLED,
}

TELEMETRY_STALE_S = 30.0
LEASE_EXPIRY_S = 60.0
TELEMETRY_INTERVAL_S = 5.0
LEASE_POLL_HOLD_S = 20.0
LOG_PUSH_INTERVAL_S = 2.0
RAM_KILL_FACTOR = 1.1  # kill at reservation x this
# Small honest reservations get absolute headroom on top of the multiplicative factor:
# a 768MB ask killed at 845MB (measured 2026-07-31) punishes exactly the callers who
# right-size. 512MB covers every measured under-kill gap (768->1088 was the worst,
# +320MB); a larger grace triples the admission commit of small jobs and starves
# co-admission. Bigger misses are auto-retried with a grown reservation instead.
# RAM only — VRAM margins are too thin on a 16GB card for an absolute grace.
KILL_CEILING_ABS_MB = 512.0


def kill_ceiling_mb(res_ram_mb: float) -> float:
    """Agent RAM kill line AND the per-job figure admission budgets for."""
    if res_ram_mb <= 0:
        return 0.0
    return max(res_ram_mb * RAM_KILL_FACTOR, res_ram_mb + KILL_CEILING_ABS_MB)
# Admission refusals when a host is already under memory pressure (P0 crash-safety):
# swap in active use means the host is past its RAM — the reservation math can no longer
# be trusted (macOS thrashes instead of OOM-killing; measured watchdog panic 2026-07-15).
SWAP_ADMISSION_MAX_MB = 2048.0
MEM_PRESSURE_WARN = 2  # darwin kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
MEM_PRESSURE_CRITICAL = 4
SWAP_SENTINEL_GROWTH_MB = 4096.0  # agent-side: swap grew this much since a job started -> kill
# The sentinel only shoots a job big enough to plausibly BE the pressure — a 61MB
# watcher killed twice for host-wide swap (measured 2026-07-30) is a bystander.
SENTINEL_MIN_VICTIM_RSS_MB = 512.0
LINUX_AVAILABLE_FLOOR_FRACTION = 0.10
LINUX_AVAILABLE_FLOOR_MIN_MB = 4096.0
DEFAULT_RESERVATION_RAM_MB = 8192.0
DEFAULT_RESERVATION_VRAM_MB = 4096.0
# Unknown workloads start SMALL and are auto-retried with a grown reservation if killed
# (the +1GB absolute kill grace makes the first-run ceiling 5120MB). Measured basis:
# the old 8192 default queue-blocked beast for days while median job RSS was ~3GB.
PROBE_DEFAULT_RAM_MB = 4096.0
PROBE_DEFAULT_VRAM_MB = 4096.0
HISTORY_SAFETY_FACTOR = 1.2


def family_key_for(
    experiment: str, cmd: list[Any] | None, config: dict[str, Any] | None
) -> str:
    """Reservation-history grouping key that ALWAYS exists.

    Model-shaped work groups by (model, task_family). Script-shaped work — the ~72% of
    jobs that carry no config.model and previously produced calibration-dead estimate
    rows — groups by (experiment, script basename), so repeated submits of the same
    stub share measured history without hashing per-item arguments apart.
    """
    config = config or {}
    task = str(config.get("task_family") or "forward")
    model = config.get("model")
    if model:
        return f"model:{model}:{task}"
    script = ""
    for c in cmd or []:
        s = str(c)
        if s.endswith(".py"):
            script = s.rsplit("/", 1)[-1]
            break
    if not script and cmd:
        script = str(cmd[0]).rsplit("/", 1)[-1]
    return f"cmd:{experiment}:{script}"
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024


def swap_signal_is_critical(
    *, system: str, available_mb: float, total_mb: float
) -> bool:
    """Whether an already-triggered swap signal denotes current host pressure.

    Darwin swap is a hard pressure signal. Linux commonly retains cold pages in
    swap after pressure has cleared, so require low currently-available RAM too.
    The scheduler and the agent sentinel share this predicate to avoid disagreeing
    about an idle Linux host.
    """
    if system.lower() != "linux":
        return True
    available_floor = max(
        LINUX_AVAILABLE_FLOOR_MIN_MB,
        total_mb * LINUX_AVAILABLE_FLOOR_FRACTION,
    )
    return available_mb < available_floor


@dataclass
class Reservation:
    ram_mb: float = DEFAULT_RESERVATION_RAM_MB
    vram_mb: float = 0.0
    cpu_threads: int = 4
    disk_gb: float = 1.0
    est_wall_s: float | None = None
    source: str = "default"  # declared | history | estimated | default

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Reservation:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Needs:
    cuda: bool = False
    mps: bool = False
    host: str | None = None  # hard pin
    prefer_host: str | None = None  # soft preference: scored up, migrates when starved

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Needs:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class JobSpec:
    """What the client submits. ``cmd`` may contain ``{env}`` which the agent replaces
    with the local path of ``env_alias`` from its config; shipped-payload jobs run with
    cwd = the extracted payload dir."""

    experiment: str
    client_run_id: str
    cmd: list[str]
    env_alias: str | None = None
    payload_kind: str = "cmd"  # cmd | shipped
    needs: dict[str, Any] = field(default_factory=dict)
    reservation: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    timeout_s: float | None = None
    priority: int = 0
