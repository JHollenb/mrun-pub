"""Fleet-visible local runs — register a locally-guarded run with the scheduler.

A run executing on a dev box outside the agent (``mx run --local``) still consumes RAM
the fleet must account for: without this, a local run + an admitted fleet job can jointly
exhaust the host (the measured 2026-07-15 crash class). ``local_run(...)`` registers the
reservation, heartbeats the live tree-RSS, and reports the terminal result — all
best-effort: if the scheduler is unreachable the run proceeds self-guarded and nothing
raises.

    from mrun.client.localrun import local_run

    with local_run("my-exp", reservation={"ram_mb": 6000}, pid_fn=lambda: child.pid):
        ...  # the guarded work
"""

from __future__ import annotations

import platform
import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..diagnostics import failure_record, normalize_external_result
from ..protocol import TELEMETRY_INTERVAL_S
from .api import Api


def _tree_pids(pid: int | None) -> set[int]:
    if not pid:
        return set()
    try:
        import psutil

        proc = psutil.Process(pid)
        procs = [proc] + proc.children(recursive=True)
        return {p.pid for p in procs if p.is_running()}
    except Exception:  # noqa: BLE001
        return set()


def _tree_rss_mb(pid: int | None) -> float:
    pids = _tree_pids(pid)
    if not pids:
        return 0.0
    try:
        import psutil

        return sum(psutil.Process(p).memory_info().rss for p in pids) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


def _tree_vram_mb(pid: int | None) -> float:
    try:
        from ..agent.hostinfo import tree_vram_mb

        return tree_vram_mb(_tree_pids(pid))
    except Exception:  # noqa: BLE001
        return 0.0


class LocalRunHandle:
    """One registered local run. All scheduler I/O is swallow-on-failure."""

    def __init__(
        self,
        experiment: str,
        *,
        reservation: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        client_run_id: str | None = None,
        host: str | None = None,
        pid_fn: Callable[[], int | None] | None = None,
        command: Iterable[object] | str | bytes | None = None,
        cwd: str | Path | None = None,
        api: Api | None = None,
    ) -> None:
        self.experiment = experiment
        self.reservation = reservation or {}
        self.config = config or {}
        self.client_run_id = client_run_id
        self.host = host or platform.node().split(".")[0].lower()
        self.pid_fn = pid_fn
        if isinstance(command, (str, bytes)):
            self.command = [str(command)]
        elif command is not None:
            self.command = [str(value) for value in command]
        else:
            self.command = None
        self.cwd = str(cwd) if cwd is not None else None
        self.api = api or Api(timeout_s=5.0)
        self.job_id: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self) -> None:
        try:
            payload: dict[str, Any] = {
                "host": self.host,
                "experiment": self.experiment,
                "client_run_id": self.client_run_id,
                "reservation": self.reservation,
                "config": self.config,
            }
            if self.command is not None:
                payload["command"] = self.command
            if self.cwd is not None:
                payload["cwd"] = self.cwd
            resp = self.api.json(
                "POST",
                "/api/local-runs",
                json_body=payload,
            )
            self.job_id = (resp or {}).get("job_id")
        except Exception:  # noqa: BLE001
            self.job_id = None  # scheduler down/off-LAN: run stays self-guarded only
        if self.job_id:
            self._thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True, name=f"localrun-{self.job_id}"
            )
            self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                pid = self.pid_fn() if self.pid_fn else None
                self.api.json(
                    "POST",
                    f"/api/local-runs/{self.job_id}/heartbeat",
                    json_body={
                        "tree_rss_mb": round(_tree_rss_mb(pid), 1),
                        "tree_vram_mb": round(_tree_vram_mb(pid), 1),
                    },
                )
            except Exception:  # noqa: BLE001
                self.api.base_url = None  # re-resolve next tick
            self._stop.wait(TELEMETRY_INTERVAL_S)

    def finish(self, state: str = "succeeded", *, result: dict[str, Any] | None = None,
               detail: str | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.job_id:
            return
        if state != "succeeded":
            result = normalize_external_result(
                state=state,
                result=result,
                detail=detail,
                command=self.command,
                cwd=self.cwd,
            )
        try:
            self.api.json(
                "POST",
                f"/api/local-runs/{self.job_id}/finish",
                json_body={"state": state, "result": result, "detail": detail},
            )
        except Exception:  # noqa: BLE001
            pass  # lease expiry marks it lost; never fail the run over bookkeeping


@contextmanager
def local_run(
    experiment: str,
    *,
    reservation: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    client_run_id: str | None = None,
    pid_fn: Callable[[], int | None] | None = None,
    command: Iterable[object] | str | bytes | None = None,
    cwd: str | Path | None = None,
    api: Api | None = None,
):
    """Context manager: register on entry, finish on exit (failed on exception).

    The body should set/refresh what ``pid_fn`` returns (the guarded child's pid) so
    heartbeats carry real tree-RSS. Call ``handle.finish(state, result=...)`` yourself
    before exit for a richer terminal state (killed_ram/timeout + peak stats).
    """
    handle = LocalRunHandle(
        experiment,
        reservation=reservation,
        config=config,
        client_run_id=client_run_id,
        pid_fn=pid_fn,
        command=command,
        cwd=cwd,
        api=api,
    )
    handle.register()
    finished = False
    original_finish = handle.finish

    def _finish_once(state: str = "succeeded", **kw: Any) -> None:
        nonlocal finished
        if not finished:
            finished = True
            original_finish(state, **kw)

    handle.finish = _finish_once  # type: ignore[method-assign]
    try:
        yield handle
    except BaseException as exc:
        failure = failure_record(
            kind="local_exception",
            phase="body",
            message="local run body raised an exception",
            exception=exc,
        )
        _finish_once("failed", detail=failure["message"], result={"failure": failure})
        raise
    else:
        _finish_once("succeeded")
