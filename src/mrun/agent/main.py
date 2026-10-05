"""mrun host agent — register, heartbeat telemetry, long-poll leases, execute under guard.

Run: ``python -m mrun.agent`` (launchd on macs, systemd user unit on beast).
Pull model: the agent calls out to the scheduler; nothing connects in. Scheduler down
=> running jobs keep being guarded, events retry with backoff.
"""

from __future__ import annotations

import os
import platform
import threading
import time
from pathlib import Path

from ..client.api import Api
from ..protocol import (
    MEM_PRESSURE_CRITICAL,
    SENTINEL_MIN_VICTIM_RSS_MB,
    SWAP_SENTINEL_GROWTH_MB,
    TELEMETRY_INTERVAL_S,
    swap_signal_is_critical,
)
from .config import AgentConfig, load_config
from .executor import JobRun
from .hostinfo import (
    disks,
    mem_pressure_level,
    model_inventory,
    register_payload,
    telemetry_payload,
)

INVENTORY_INTERVAL_S = 600.0


def _swap_growth_is_critical(
    *, system: str, growth_mb: float, available_mb: float, total_mb: float
) -> bool:
    """Darwin swap growth is pressure; Linux swap can just be cold-page reclamation."""
    if growth_mb <= SWAP_SENTINEL_GROWTH_MB:
        return False
    return swap_signal_is_critical(
        system=system,
        available_mb=available_mb,
        total_mb=total_mb,
    )


class Agent:
    def __init__(self, cfg: AgentConfig | None = None) -> None:
        self.cfg = cfg or load_config()
        self._apply_model_roots()
        self.host = self.cfg.host or platform.node().split(".")[0].lower()
        self.api = Api(
            self.cfg.server_url,
            token=self.cfg.token,
            agent_token=self.cfg.agent_token,
        )
        self.runs: dict[str, JobRun] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._swap_floor_mb: float | None = None  # swap when the first job started
        self._pressure_external = False  # pressure with no plausible mrun victim

    def _apply_model_roots(self) -> None:
        roots = [
            path for path in os.environ.get("LLM_MODELS_EXTRA_ROOT", "").split(os.pathsep) if path
        ]
        roots.extend(str(Path(path).expanduser()) for path in self.cfg.model_roots)
        # Beast keeps model families on separate mounted volumes.  The service config
        # historically named only one of them, which made complete checkpoints look
        # absent even though they were present at the mount root (for example
        # /mnt/big/Qwen3.8-27B).  Linux agents can safely add mounted data volumes as
        # shallow lookup roots; find_safetensors still requires an exact model directory
        # and a complete shard set, so unrelated directories are not loaded.
        if platform.system() == "Linux":
            roots.extend(
                str(row["mount"])
                for row in disks()
                if str(row.get("mount") or "").startswith("/mnt/")
            )
        deduplicated = list(dict.fromkeys(roots))
        if deduplicated:
            os.environ["LLM_MODELS_EXTRA_ROOT"] = os.pathsep.join(deduplicated)

    # -- loops -------------------------------------------------------------------
    def register(self) -> None:
        from ..payload_sandbox import payload_sandbox_supported

        sandbox_enabled = (
            str(self.cfg.payload_sandbox_mode).lower() != "off"
            and bool(self.cfg.payload_sandbox_roots)
            and payload_sandbox_supported()
        )
        payload = register_payload(
            self.host,
            guarded_custody=bool(self.cfg.agent_token),
            payload_sandbox=sandbox_enabled,
        )
        resp = self.api.retry_json(
            "POST", "/api/agents/register", json_body=payload, max_wait_s=86400
        )
        print(f"mrun-agent: registered {self.host} -> {self.api.base_url} ({resp})", flush=True)
        self.push_inventory()

    def push_inventory(self) -> None:
        """Report which models this host has bytes for (warm-cache placement)."""
        try:
            rows = model_inventory(self.host)
            self.api.json("POST", f"/api/agents/{self.host}/inventory", json_body={"models": rows})
            self._last_inventory_ts = time.time()
            print(f"mrun-agent: inventory pushed ({len(rows)} entries)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"mrun-agent: inventory push failed ({exc})", flush=True)

    def pressure_watch(self) -> None:
        """Host-level backstop, independent of the scheduler: if swap grew hard since a
        job started, or Darwin pressure hits critical, kill the largest-RSS job before the
        HOST dies (macOS thrashes to a watchdog panic instead of OOM-killing — measured
        2026-07-15). Per-job RSS ceilings can all hold while the host still drowns
        (baseline drift, Metal allocations invisible to RSS)."""
        import psutil

        with self._lock:
            runs = list(self.runs.values())
        if not runs:
            self._swap_floor_mb = None
            return
        swap_mb = psutil.swap_memory().used / 1e6
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1e6
        total_mb = memory.total / 1e6
        if self._swap_floor_mb is None:
            self._swap_floor_mb = swap_mb
            return
        pressure = mem_pressure_level()
        grew = swap_mb - self._swap_floor_mb
        if _swap_growth_is_critical(
            system=platform.system(),
            growth_mb=grew,
            available_mb=available_mb,
            total_mb=total_mb,
        ) or (pressure is not None and pressure >= MEM_PRESSURE_CRITICAL):
            victim = max(runs, key=lambda r: r.cur_rss_mb)
            # A 61MB watcher cannot be why the host is drowning (measured collateral
            # kill x2, 2026-07-30). No plausible mrun victim -> the pressure is
            # out-of-band; report it instead of shooting a bystander.
            if victim.cur_rss_mb < SENTINEL_MIN_VICTIM_RSS_MB:
                if not self._pressure_external:
                    print(
                        f"mrun-agent: PRESSURE (out-of-band) — swap +{grew:.0f}MB "
                        f"available={available_mb:.0f}MB pressure={pressure}; largest "
                        f"mrun job is only {victim.cur_rss_mb:.0f}MB — not killing",
                        flush=True,
                    )
                self._pressure_external = True
                self._swap_floor_mb = swap_mb
                return
            if not victim.pressure_kill.is_set():
                print(
                    f"mrun-agent: PRESSURE SENTINEL — swap +{grew:.0f}MB "
                    f"available={available_mb:.0f}MB pressure={pressure}; "
                    f"killing {victim.job_id} "
                    f"(rss {victim.cur_rss_mb:.0f}MB)",
                    flush=True,
                )
                victim.pressure_kill.set()
            self._swap_floor_mb = swap_mb  # re-arm against the new floor
        else:
            self._pressure_external = False

    def telemetry_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.pressure_watch()
            except Exception as exc:  # noqa: BLE001 — the sentinel must never kill the loop
                print(f"mrun-agent: pressure watch error ({exc})", flush=True)
            try:
                with self._lock:
                    running = [r.running_stat() for r in self.runs.values()]
                    job_pids = set().union(*(r.tree_pids for r in self.runs.values()), set())
                resp = self.api.json(
                    "POST",
                    f"/api/agents/{self.host}/telemetry",
                    json_body=telemetry_payload(
                        running,
                        exclude_pids=job_pids,
                        pressure_external=self._pressure_external,
                    ),
                )
                if (resp or {}).get("reregister"):
                    # our host row was deleted server-side (UI remove while asleep) —
                    # re-register instead of writing orphan telemetry forever
                    print("mrun-agent: server forgot us; re-registering", flush=True)
                    self.register()
                for job_id in (resp or {}).get("kill") or []:
                    with self._lock:
                        run = self.runs.get(job_id)
                    if run:
                        print(f"mrun-agent: kill order for {job_id}", flush=True)
                        run.kill_flag.set()
            except Exception as exc:  # noqa: BLE001
                self.api.base_url = None  # re-resolve next time
                print(f"mrun-agent: telemetry failed ({exc}); retrying", flush=True)
            self._stop.wait(TELEMETRY_INTERVAL_S)

    def lease_loop(self) -> None:
        if self.cfg.max_concurrent <= 0:
            # telemetry-only host (e.g. an agent on zima reporting disks) — never lease
            print("mrun-agent: max_concurrent=0 — telemetry only, not leasing", flush=True)
            self._stop.wait()
            return
        while not self._stop.is_set():
            reaped = self._reap()
            if reaped:
                self.push_inventory()  # a finished job may have downloaded/built a model
            elif time.time() - getattr(self, "_last_inventory_ts", 0.0) > INVENTORY_INTERVAL_S:
                self.push_inventory()
            try:
                job = self.api.json("POST", f"/api/agents/{self.host}/lease", timeout_s=40.0)
            except Exception as exc:  # noqa: BLE001
                self.api.base_url = None
                print(f"mrun-agent: lease poll failed ({exc}); backing off", flush=True)
                time.sleep(5.0)
                continue
            if not job:
                continue  # 204 — poll again
            print(
                f"mrun-agent: leased {job['job_id']} ({job.get('experiment')}) "
                f"reservation={job.get('reservation')}",
                flush=True,
            )
            run = JobRun(self.api, self.cfg, job)
            with self._lock:
                self.runs[job["job_id"]] = run
            run.start()

    def _reap(self) -> bool:
        with self._lock:
            done = [jid for jid, r in self.runs.items() if not r.thread.is_alive()]
            for jid in done:
                del self.runs[jid]
        return bool(done)

    def run_forever(self) -> None:
        self.register()
        t = threading.Thread(target=self.telemetry_loop, daemon=True, name="telemetry")
        t.start()
        try:
            self.lease_loop()
        except KeyboardInterrupt:
            print("mrun-agent: stopping (running jobs keep their guards)", flush=True)
            self._stop.set()


def main() -> None:
    Agent().run_forever()


if __name__ == "__main__":
    main()
