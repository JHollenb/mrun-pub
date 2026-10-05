"""Job executor — the fleet generalization of ``orchestrate.run_stage``.

Child runs in its own process group; a 1 s poller tracks tree-RSS (+ tree-VRAM on cuda
hosts) and SIGKILLs the group at reservation x RAM_KILL_FACTOR — killing at the
*reservation* (not host exhaustion) is what keeps admission sound. Log bytes stream to
the scheduler every 2 s at authoritative offsets; a terminal event carries the StageRun-
style result that becomes the estimates-calibration row.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import signal
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from ..client.api import Api
from ..diagnostics import execution_diagnostics, failure_record, redact_text
from ..os_limit import (
    OSMemoryLimitLaunch,
    cleanup_strict_scope,
    prepare_os_memory_limit,
    strict_scope_oom_killed,
)
from ..payload_sandbox import (
    PayloadSandboxLaunch,
    prepare_payload_sandbox,
    sandbox_requested,
)
from ..protocol import LOG_PUSH_INTERVAL_S, RAM_KILL_FACTOR, Reservation, kill_ceiling_mb
from .config import AgentConfig
from .hostinfo import tree_vram_mb


def _execution_receipt_requested(job: dict[str, Any]) -> bool:
    """Persist an executed-payload receipt for guarded or explicitly v2-capable jobs."""
    return bool(
        (job.get("payload_custody") or {}).get("required")
        or (job.get("needs") or {}).get("payload_custody_v2")
    )


class JobRun:
    """One executing job. ``kill_flag`` is set by the telemetry loop on server order."""

    def __init__(self, api: Api, cfg: AgentConfig, job: dict[str, Any]) -> None:
        self.api = api
        self.cfg = cfg
        self.job = dict(job)
        self._lease_authorization = self.job.pop("agent_authorization", None)
        self._debug_authorization = self.job.pop("debug_authorization", None)
        self.job_id = job["job_id"]
        self.res = Reservation.from_dict(job.get("reservation"))
        self.kill_flag = threading.Event()
        self.pressure_kill = threading.Event()  # host swap/pressure sentinel (agent-side)
        self.started = time.time()
        self.peak_rss_mb = 0.0
        self.peak_vram_mb = 0.0
        self.cur_rss_mb = 0.0
        self.cur_vram_mb = 0.0
        self.tree_pids: set[int] = set()
        self.proc: subprocess.Popen | None = None
        self.executed_payload: dict[str, Any] | None = None
        self.command: list[str] = [str(c) for c in self.job.get("cmd") or []]
        self.work_dir: Path | None = None
        self.phase = "created"
        self.phase_history: list[dict[str, Any]] = []
        self.dispatch_guard: dict[str, Any] | None = None
        self.os_memory_limit_backend = "off"
        self.os_memory_limit_launch = OSMemoryLimitLaunch([], "off", False)
        self.payload_sandbox_backend = "off"
        self.payload_sandbox_launch = PayloadSandboxLaunch([], "off", False, {"requested": False})
        self.thread = threading.Thread(target=self._run, name=f"job-{self.job_id}", daemon=True)
        self.done = threading.Event()

    # -- public ---------------------------------------------------------------
    def start(self) -> None:
        self.thread.start()

    def running_stat(self) -> dict[str, Any]:
        stat = {
            "job_id": self.job_id,
            "tree_rss_mb": round(self.cur_rss_mb, 1),
            "vram_mb": round(self.cur_vram_mb, 1),
            "elapsed_s": round(time.time() - self.started, 1),
        }
        if isinstance(self._lease_authorization, dict):
            # The server strips this before telemetry persistence/exposure. It is used
            # only to stop a general client token from extending a guarded lease.
            stat["lease_authorization"] = {
                "lease_id": self._lease_authorization.get("lease_id"),
                "capability": self._lease_authorization.get("capability"),
            }
        return stat

    # -- internals -------------------------------------------------------------
    def _guarded_headers(self) -> dict[str, str] | None:
        required = bool((self.job.get("payload_custody") or {}).get("required"))
        if not required:
            return None
        authority = self._lease_authorization
        if not isinstance(authority, dict):
            raise RuntimeError("guarded lease response omitted agent authorization")
        lease_id = authority.get("lease_id")
        capability = authority.get("capability")
        if not isinstance(lease_id, str) or not isinstance(capability, str):
            raise RuntimeError("guarded lease response contained invalid authorization")
        return {
            "X-MRun-Lease-ID": lease_id,
            "X-MRun-Lease-Capability": capability,
        }

    def _event(
        self,
        state: str | None,
        *,
        result: dict[str, Any] | None = None,
        detail: str | None = None,
        phase: str | None = None,
    ) -> None:
        from ..client.api import ApiError

        payload: dict[str, Any] = {"result": result, "detail": detail}
        if state is not None:
            payload["state"] = state
        if phase is not None:
            payload["phase"] = phase
        try:
            self.api.retry_json(
                "POST",
                f"/api/jobs/{self.job_id}/events",
                json_body=payload,
                headers=self._guarded_headers(),
                max_wait_s=3600,
            )
        except ApiError as exc:
            # 409 = job no longer active server-side (lost/cancelled while we ran) — the
            # server's word is final; don't crash the reporter thread over it.
            print(f"mrun-agent: event {state} for {self.job_id} rejected: {exc}", flush=True)

    def _phase(
        self,
        name: str,
        *,
        detail: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.phase = name
        self.phase_history.append({"name": name, "ts": time.time()})
        self._event(None, phase=name, detail=detail, result=payload)

    def _prepare(self) -> tuple[list[str], Path]:
        work = self.cfg.work_dir(self.job_id)
        if self.job.get("payload_kind") == "shipped":
            # GAP 2: retry the GET — an idle host can be handed a lease before the submit's
            # payload PUT lands (instant-dispatch race → HTTP 404 → 0.03s "download failed").
            from .dispatch_guard import fetch_payload_with_retry, payload_config_from

            body = fetch_payload_with_retry(
                lambda: self.api.request(
                    "GET",
                    f"/api/payloads/{self.job_id}",
                    headers=self._guarded_headers(),
                )[:2],
                cfg=payload_config_from(self.cfg),
                sleep=time.sleep,
                clock=time.monotonic,
                log=lambda m: print(f"mrun-agent: {m}", flush=True),
                job_id=self.job_id,
            )
            sha256 = hashlib.sha256(body).hexdigest()
            size_bytes = len(body)
            custody = self.job.get("payload_custody") or {}
            sealed = custody.get("sealed")
            if sealed is None:
                # Rolling upgrade compatibility: an updated agent may briefly poll an
                # old scheduler that has no custody schema. Such a scheduler cannot
                # create a guarded job, so only ordinary legacy work may use this path.
                if (self.job.get("needs") or {}).get("payload_custody_v2"):
                    raise RuntimeError("guarded shipped payload has no server seal")
            else:
                if sealed.get("sha256") != sha256 or sealed.get("size_bytes") != size_bytes:
                    raise RuntimeError(
                        "downloaded shipped payload differs from scheduler seal: "
                        f"got {sha256}/{size_bytes}, expected "
                        f"{sealed.get('sha256')}/{sealed.get('size_bytes')}"
                    )
                if _execution_receipt_requested(self.job):
                    host = self.cfg.host or platform.node().split(".")[0].lower()
                    receipt = self.api.retry_json(
                        "POST",
                        f"/api/jobs/{self.job_id}/payload/executed",
                        json_body={"host": host, "sha256": sha256, "size_bytes": size_bytes},
                        headers=self._guarded_headers(),
                        max_wait_s=3600,
                    )
                    executed = (receipt.get("payload_custody") or {}).get("executed")
                    if (
                        executed is None
                        or executed.get("sha256") != sha256
                        or executed.get("size_bytes") != size_bytes
                        or executed.get("host") != host
                    ):
                        raise RuntimeError("scheduler returned an invalid executed-payload receipt")
                    self.executed_payload = {
                        "sha256": sha256,
                        "size_bytes": size_bytes,
                        "host": host,
                    }
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
                tar.extractall(work, filter="data")
        cmd = [str(c) for c in self.job.get("cmd") or []]
        env_path_for_alias = getattr(self.cfg, "env_path", None)
        env_path = (
            env_path_for_alias(self.job.get("env_alias")) if callable(env_path_for_alias) else None
        )
        cmd = [c.replace("{env}", str(env_path)) if "{env}" in c else c for c in cmd]
        if any("{env}" in str(c) for c in self.job.get("cmd") or []) and env_path is None:
            raise RuntimeError(f"env alias {self.job.get('env_alias')!r} not in agent config")
        if platform.system() == "Darwin":
            cmd = ["caffeinate", "-dims", *cmd]  # a mac must not sleep mid-job
        self.command = cmd
        self.work_dir = work
        return cmd, work

    def _child_env(self, work: Path | None = None) -> dict[str, str]:
        isolated = sandbox_requested(self.job)
        if isolated:
            # Same-UID isolation starts with a positive environment allowlist. Broad
            # service, cloud, registry, SSH, and scheduler credentials never enter the
            # namespace even if a managed service supplied them to the agent.
            allowed = {
                "LANG",
                "LANGUAGE",
                "LC_ALL",
                "LC_CTYPE",
                "TZ",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "CUDA_MODULE_LOADING",
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
                "LD_LIBRARY_PATH",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "NO_PROXY",
                "no_proxy",
            }
            env = {key: value for key, value in os.environ.items() if key in allowed}
        else:
            env = dict(os.environ)
        # These are scheduler/agent boundary credentials. Even when a service manager
        # supplied them in the parent environment, payload code must never inherit them.
        for secret_name in (
            "MRUN_TOKEN",
            "MRUN_SERVER_TOKEN",
            "MRUN_AGENT_TOKEN",
            "MRUN_LEASE_CAPABILITY",
            "MRUN_LEASE_ID",
            "MRUN_DEBUG_CREDENTIAL",
            "MRUN_DEBUG_CREDENTIAL_ID",
        ):
            env.pop(secret_name, None)
        threads = str(int(self.res.cpu_threads or 4))
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[var] = threads
        # The agent service may inherit a host-level command shim (Beast currently
        # exposes /snap/bin/mrun, which is not executable from the agent's scope).
        # An env alias identifies the local checkout; make its virtualenv the first
        # command search path so bare `mrun`, `python`, and sibling entry points use
        # the same environment as the editable source tree.
        env_path_for_alias = getattr(self.cfg, "env_path", None)
        env_path = (
            env_path_for_alias(self.job.get("env_alias")) if callable(env_path_for_alias) else None
        )
        if env_path is not None:
            env_bin = env_path / ".venv" / "bin"
            project_bin = env_path / "bin"
            if isolated:
                uv = shutil.which("uv")
                uv_bin = str(Path(uv).parent) if uv else ""
                prior_path = os.pathsep.join(
                    value
                    for value in (
                        uv_bin,
                        "/usr/local/sbin",
                        "/usr/local/bin",
                        "/usr/sbin",
                        "/usr/bin",
                        "/sbin",
                        "/bin",
                    )
                    if value
                )
            else:
                prior_path = env.get("PATH", "")
            env["PATH"] = os.pathsep.join(str(path) for path in (env_bin, project_bin) if path) + (
                os.pathsep + prior_path if prior_path else ""
            )
        elif isolated:
            env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        env["RSS_LIMIT_MB"] = str(int(kill_ceiling_mb(self.res.ram_mb)))
        # The child's VRAM ceiling = its declared reservation (NOT the kill factor): a
        # resident-cache sizer that reads physical free VRAM would overfill a shared card
        # past the reservation and trip the kill line (I11/I2, killed_vram on bf16-validate).
        # Hand the declared reservation down so in-process budgeters cap to what admission
        # granted, leaving the kill factor as pure margin.
        if self.res.vram_mb:
            env["VRAM_LIMIT_MB"] = str(int(self.res.vram_mb))
        env["MRUN_JOB_ID"] = self.job_id
        # A shipped Python entrypoint normally puts only its own script directory on
        # sys.path. Make the payload root importable so `scripts/run.py` can import a
        # sibling package identically on macOS and Linux (I36).
        if work is not None:
            prior_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                f"{work}{os.pathsep}{prior_pythonpath}" if prior_pythonpath else str(work)
            )
            if isolated:
                sandbox_home = work / ".sandbox-home"
                sandbox_tmp = work / ".sandbox-tmp"
                sandbox_cache = work / ".sandbox-cache"
                for directory in (sandbox_home, sandbox_tmp, sandbox_cache):
                    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                env["HOME"] = str(sandbox_home)
                env["TMPDIR"] = str(sandbox_tmp)
                env["XDG_CACHE_HOME"] = str(sandbox_cache)
                env["XDG_CONFIG_HOME"] = str(sandbox_home / ".config")
                env["PYTHONPYCACHEPREFIX"] = str(sandbox_cache / "pycache")
        # The worker receives the scheduler address for lease-scoped debugger calls.
        # General client and independent agent credentials never enter the child.
        if self.cfg.server_url:
            env["MRUN_SERVER_URL"] = self.cfg.server_url
        debug_authority = getattr(self, "_debug_authorization", None)
        debug_requested = bool((self.job.get("needs") or {}).get("saturn_debug_credential_v1"))
        if isinstance(debug_authority, dict):
            credential_id = debug_authority.get("credential_id")
            credential = debug_authority.get("credential")
            if (
                debug_authority.get("schema") != "mrun.job-debug-credential-v1"
                or not isinstance(credential_id, str)
                or not credential_id
                or not isinstance(credential, str)
                or not credential
            ):
                raise RuntimeError("guarded debugger lease response is malformed")
            env["MRUN_DEBUG_CREDENTIAL_ID"] = credential_id
            env["MRUN_DEBUG_CREDENTIAL"] = credential
        elif debug_requested:
            raise RuntimeError("debugger-capable job lease omitted its narrow debugger credential")
        # The scheduler-chosen plan drives execution: plan_run in the child short-circuits
        # on MRUN_PLAN, so the sizing that won admission is the one that runs.
        plan = self.job.get("plan")
        if plan:
            import json as _json

            env["MRUN_PLAN"] = _json.dumps(plan)
            if plan.get("dtype"):
                env["MRUN_TORCH_DTYPE"] = str(plan["dtype"])
            if plan.get("max_batch"):
                env["GATHER_MAX_BATCH"] = str(int(plan["max_batch"]))
        return env

    def _run(self) -> None:
        log_path = self.cfg.work_dir(self.job_id) / "job.log"
        status = "failed"
        returncode = -1
        detail = None
        failure: dict[str, Any] | None = None
        log_handle = None
        pusher: threading.Thread | None = None
        try:
            self._event("preparing")
            self._phase("payload.prepare")
            cmd, work = self._prepare()
            # GAP 1: never launch a VRAM job into a GPU that is ACTUALLY occupied (finishing
            # job's CUDA context not yet freed / cross-session process / leaked context →
            # instant OOM). Hold the lease with backoff; if still blocked past the window,
            # release WITHOUT a terminal 'failed' event — the lease lapses to 'lost' and the
            # job can be re-driven (not a hard failure).
            from .dispatch_guard import guard_vram_or_release

            self._phase(
                "dispatch.vram_guard",
                payload={
                    "declared_vram_mb": float(self.res.vram_mb or 0.0),
                    "safety_margin_mb": float(getattr(self.cfg, "vram_safety_margin_mb", 500.0)),
                },
            )
            if not guard_vram_or_release(self):
                decision = self.dispatch_guard or {}
                self._phase(
                    "dispatch.vram_release",
                    detail=str(decision.get("reason") or "actual VRAM remained unavailable"),
                    payload=decision,
                )
                self.done.set()
                return
            self._phase("process.launch")
            log_handle = open(log_path, "wb")
            env_path_for_alias = getattr(self.cfg, "env_path", None)
            env_path = (
                env_path_for_alias(self.job.get("env_alias"))
                if callable(env_path_for_alias)
                else None
            )
            sandbox_launch = prepare_payload_sandbox(
                cmd,
                job=self.job,
                work_dir=work,
                env_path=env_path,
                allowed_read_roots=self.cfg.payload_sandbox_roots,
                shared_cache_paths=self.cfg.payload_sandbox_caches,
            )
            self.payload_sandbox_backend = sandbox_launch.backend
            self.payload_sandbox_launch = sandbox_launch
            launch = prepare_os_memory_limit(
                sandbox_launch.command,
                limit_mb=kill_ceiling_mb(self.res.ram_mb),
                mode=self.cfg.os_memory_limit_mode,
                identity=f"job-{self.job_id}",
            )
            self.os_memory_limit_backend = launch.backend
            self.os_memory_limit_launch = launch
            self.proc = subprocess.Popen(
                launch.command,
                cwd=str(work),
                env=self._child_env(work),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._event("running")
            self._phase("process.monitor")
            pusher = threading.Thread(target=self._push_logs_loop, args=(log_path,), daemon=True)
            pusher.start()
            status, returncode, detail = self._guard_loop()
            self._phase("process.finalize")
            if status != "ok":
                detail = detail or f"child exited with return code {returncode}"
                failure = failure_record(
                    kind={
                        "failed": "process_exit",
                        "killed_ram": "killed_ram",
                        "killed_vram": "killed_vram",
                        "timeout": "timeout",
                        "cancelled": "cancelled",
                    }.get(status, "process_failure"),
                    phase=self.phase,
                    message=detail,
                    command=self.command,
                    cwd=self.work_dir,
                    returncode=returncode,
                    resources={
                        "peak_rss_mb": round(self.peak_rss_mb, 1),
                        "peak_vram_mb": round(self.peak_vram_mb, 1),
                        "ram_ceiling_mb": kill_ceiling_mb(self.res.ram_mb),
                        "vram_ceiling_mb": (
                            self.res.vram_mb * RAM_KILL_FACTOR if self.res.vram_mb else None
                        ),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            detail = f"agent error: {redact_text(exc)}"
            failure = failure_record(
                kind="agent_exception",
                phase=self.phase,
                message="agent failed while preparing or executing the job",
                command=self.command,
                cwd=self.work_dir,
                exception=exc,
                resources={
                    "peak_rss_mb": round(self.peak_rss_mb, 1),
                    "peak_vram_mb": round(self.peak_vram_mb, 1),
                },
            )
            if self.proc is not None and self.proc.poll() is None:
                self._phase("process.abort", detail=detail)
                self._kill_group()
        finally:
            self.done.set()
            if log_handle is not None:
                try:
                    log_handle.close()
                except Exception:  # noqa: BLE001
                    pass
            if pusher is not None:
                pusher.join(timeout=10)
            try:
                self._push_logs(log_path)  # final flush
            except Exception:  # noqa: BLE001
                pass
            if failure is not None:
                from ..diagnostics import log_tail

                tail = log_tail(log_path)
                if tail is not None:
                    failure["log_tail"] = tail
            if self.phase != "finished":
                self.phase_history.append({"name": "finished", "ts": time.time()})
                self.phase = "finished"
        result = {
            "status": status,
            "returncode": returncode,
            "elapsed_s": round(time.time() - self.started, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "executed_payload": self.executed_payload,
            "os_memory_limit_backend": self.os_memory_limit_backend,
            "payload_sandbox_backend": self.payload_sandbox_backend,
            "payload_sandbox": dict(self.payload_sandbox_launch.profile),
            "diagnostics": execution_diagnostics(
                command=self.command,
                cwd=self.work_dir,
                phase=self.phase,
                phases=self.phase_history,
                log_path=log_path,
            ),
            "failure": failure,
        }
        from ..protocol import STAGE_TO_STATE

        self._event(STAGE_TO_STATE.get(status, "failed"), result=result, detail=detail)

    def _tree(self) -> tuple[float, set[int]]:
        try:
            proc = psutil.Process(self.proc.pid)
            procs = [proc] + proc.children(recursive=True)
        except Exception:  # noqa: BLE001
            return 0.0, set()
        rss = 0.0
        pids: set[int] = set()
        for p in procs:
            try:
                if not p.is_running():
                    continue
                pids.add(p.pid)
                rss += p.memory_info().rss
            except (psutil.Error, OSError):
                # Process-tree membership races with short-lived children. One vanished
                # wrapper must not zero the whole tree or RAM admission/enforcement lies.
                continue
        return rss / (1024 * 1024), pids

    def _guard_loop(self) -> tuple[str, int, str | None]:
        ram_ceiling = kill_ceiling_mb(self.res.ram_mb)
        vram_ceiling = self.res.vram_mb * RAM_KILL_FACTOR if self.res.vram_mb else None
        timeout_s = self.job.get("timeout_s") or (
            self.res.est_wall_s * 4 if self.res.est_wall_s else 24 * 3600
        )
        while True:
            ret = self.proc.poll()
            rss, pids = self._tree()
            self.tree_pids = pids  # lets telemetry attribute host RSS to mrun vs external
            self.cur_rss_mb = rss
            self.peak_rss_mb = max(self.peak_rss_mb, rss)
            vram = tree_vram_mb(pids)
            self.cur_vram_mb = vram
            self.peak_vram_mb = max(self.peak_vram_mb, vram)
            if ret is not None:
                if strict_scope_oom_killed(self.os_memory_limit_launch, ret):
                    return (
                        "killed_ram",
                        ret,
                        "strict cgroup scope exited by SIGKILL at memory.max",
                    )
                return ("ok" if ret == 0 else "failed"), ret, None
            status = detail = None
            if rss > ram_ceiling:
                status, detail = "killed_ram", f"tree RSS {rss:.0f}MB > ceiling {ram_ceiling:.0f}MB"
            elif vram_ceiling and vram > vram_ceiling:
                status, detail = (
                    "killed_vram",
                    f"tree VRAM {vram:.0f}MB > ceiling {vram_ceiling:.0f}MB",
                )
            elif time.time() - self.started > timeout_s:
                status, detail = "timeout", f"exceeded {timeout_s:.0f}s"
            elif self.pressure_kill.is_set():
                status, detail = (
                    "killed_ram",
                    "host pressure sentinel (swap growth / memory pressure critical)",
                )
            elif self.kill_flag.is_set():
                status, detail = "cancelled", "cancelled by server order"
            if status is None:
                time.sleep(1.0)
                continue
            self._kill_group()
            cleanup_strict_scope(self.os_memory_limit_launch)
            return status, -9, detail

    def _kill_group(self) -> None:
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass

    # -- log streaming ----------------------------------------------------------
    def _push_logs_loop(self, log_path: Path) -> None:
        offset = 0
        while not self.done.is_set():
            offset = self._push_logs(log_path, offset)
            self.done.wait(LOG_PUSH_INTERVAL_S)

    def _push_logs(self, log_path: Path, offset: int | None = None) -> int:
        if offset is None:
            offset = getattr(self, "_log_offset", 0)
        try:
            size = log_path.stat().st_size if log_path.exists() else 0
            while offset < size:
                with open(log_path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read(1024 * 1024)
                status, body, _ = self.api.request(
                    "POST",
                    f"/api/jobs/{self.job_id}/logs?offset={offset}",
                    raw_body=chunk,
                    headers=self._guarded_headers(),
                )
                if status >= 400:
                    break
                import json as _json

                resp = _json.loads(body)
                offset = int(resp.get("next_offset", offset))
                if resp.get("resync"):
                    continue
        except Exception:  # noqa: BLE001
            pass  # next tick retries; offsets make this safe
        self._log_offset = offset
        return offset
