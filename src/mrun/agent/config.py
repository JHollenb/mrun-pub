"""Agent config — JSON at ~/.mrun/agent.json (override with MRUN_AGENT_CONFIG).

Example:
{
  "server_url": "http://127.0.0.1:9025",
  "token": "…",
  "agent_token": "independent guarded-agent credential",
  "max_concurrent": 1,
  "os_memory_limit_mode": "auto",
  "payload_sandbox_mode": "auto",
  "payload_sandbox_roots": ["/srv/models"],
  "payload_sandbox_caches": [
    "/srv/models/.uv-cache",
    "/srv/models/.torch-cache"
  ],
  "model_roots": ["/srv/models"],
  "envs": {
    "runtime": "/srv/runtime"
  }
}

``envs`` maps a job's ``env_alias`` to the LOCAL uv project whose environment runs the
payload ("code travels, environments don't" — each host keeps its own torch build).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    server_url: str | None = None
    token: str | None = None
    # Independent of the general MRUN token. Without it this agent deliberately
    # remains eligible only for ordinary/legacy jobs during a rolling upgrade.
    agent_token: str | None = None
    # 0 makes a telemetry-only agent. Any positive value enables leasing; actual
    # co-admission is decided by scheduler RAM/VRAM/CPU reservation math.
    max_concurrent: int = 2
    envs: dict[str, str] = field(default_factory=dict)
    # Additional materialized Hugging Face roots. Kept in agent.json so managed
    # launchd/systemd services do not depend on a cached service environment.
    model_roots: list[str] = field(default_factory=list)
    host: str | None = None  # override the detected hostname
    work_root: str = "~/.local/state/mrun/work"
    spool_root: str = "~/.local/state/mrun/spool"
    # Linux jobs default to a verified cgroup-v2 memory.max scope. On unsupported
    # platforms auto retains the sampled guard; required fails before workload exec.
    os_memory_limit_mode: str = "auto"
    # Required jobs are admitted only to agents that can build the bubblewrap
    # process/mount boundary. Roots are operator-controlled safe model/artifact stores;
    # a submitted job may request a child path but cannot expand this allowlist.
    payload_sandbox_mode: str = "auto"
    payload_sandbox_roots: list[str] = field(default_factory=list)
    # Shared package caches are mounted through an ephemeral per-job overlay. Reads use
    # the warm lower directory while payload writes disappear with the namespace.
    payload_sandbox_caches: list[str] = field(default_factory=list)
    # dispatch-time VRAM guard (see agent/dispatch_guard.py) — don't launch a vram job
    # into a GPU that is ACTUALLY occupied; hold the lease with backoff instead.
    vram_safety_margin_mb: float = 500.0
    # Hold window is QUEUE semantics, not failure semantics: each JobRun holds in its own
    # thread (executor spawns one thread per job; a holding job never blocks other
    # dispatches), so a long window turns "GPU busy" into "wait in line" instead of
    # lapsing the job to `lost` after 2 minutes — which on a contended card (two sessions
    # sharing beast) turned every queued job into a manual re-drive. 2h covers the long
    # trainings that actually occupy the card; the 60s backoff cap keeps polls cheap.
    vram_hold_window_s: float = 7200.0
    vram_backoff_s: list[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 40.0, 60.0])
    # payload-download retry (absorbs the idle-host instant-dispatch race)
    payload_retry_attempts: int = 6
    payload_retry_window_s: float = 30.0

    def env_path(self, alias: str | None) -> Path | None:
        if not alias:
            return None
        p = self.envs.get(alias)
        return Path(p).expanduser() if p else None

    def work_dir(self, job_id: str) -> Path:
        d = Path(self.work_root).expanduser() / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config() -> AgentConfig:
    path = Path(
        os.environ.get("MRUN_AGENT_CONFIG", str(Path.home() / ".mrun" / "agent.json"))
    ).expanduser()
    cfg = AgentConfig()
    if path.exists():
        data = json.loads(path.read_text())
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    # Managed services can provision the independent credential through a root-owned
    # EnvironmentFile without rewriting the agent's ordinary JSON configuration.
    # The executor explicitly strips this variable from every child environment.
    service_agent_token = os.environ.get("MRUN_AGENT_TOKEN")
    if service_agent_token:
        cfg.agent_token = service_agent_token
    return cfg
