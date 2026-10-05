"""Local guarded execution — run an explicit command under a RunPlan's guard.

This is the one code path both local runs and the fleet agent use: build (or accept) a
``RunPlan``-shaped reservation, then execute the work in its own process group with the
sampled tree-RSS SIGKILL guard (``orchestrate.run_stage``) so a sustained overage observed
by the poller dies with a clear ``killed_ram`` instead of taking down the host.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .guard import set_thread_env
from .orchestrate import StageRun, run_stage

DEFAULT_TIMEOUT_S = 24 * 3600


@dataclass(frozen=True)
class RunResult:
    stage: StageRun
    log_path: str | None
    run_dir: str | None = None

    @property
    def ok(self) -> bool:
        return self.stage.status == "ok"

    @property
    def status(self) -> str:
        return self.stage.status






def run(
    target: Sequence[str],
    *,
    model: str | None = None,
    ram_limit_mb: float | None = None,
    timeout_s: float | None = None,
    threads: int | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    log_path: str | Path | None = None,
    echo: bool = True,
    os_memory_limit_mode: str | None = None,
) -> RunResult:
    """Run ``target`` locally under the RAM/timeout guard.

    Pass a nonempty command argument list. The guard ceiling comes from an explicit
    ram_limit_mb or the policy plan for model. Without either, execution is
    monitor-only and emits a warning.
    """
    from .policy import HostCaps, plan_run

    env = dict(env or {})
    if ram_limit_mb is None and model is not None:
        plan = plan_run(model, host=HostCaps.detect())
        ram_limit_mb = plan.ram_limit_mb
        threads = threads or plan.threads
    if threads:
        # pin BLAS pools in THIS process env so the child inherits them
        set_thread_env(threads)
    if ram_limit_mb is None:
        print(
            "mrun.run: no ram_limit_mb and no model given — running UNGUARDED "
            "(pass model=... or ram_limit_mb=... to enable the kill ceiling)",
            file=sys.stderr,
        )

    if isinstance(target, (str, Path)) or not target:
        raise TypeError("target must be a nonempty command argument list")
    cmd = [str(c) for c in target]

    if log_path is None:
        name = Path(cmd[0]).name
        log_path = Path.cwd() / f"mrun-{name}.log"
    log_path = Path(log_path)

    stage = run_stage(
        cmd,
        cwd=Path(cwd) if cwd else None,
        env=env,
        log_path=log_path,
        ram_limit_mb=ram_limit_mb,
        timeout_s=timeout_s or DEFAULT_TIMEOUT_S,
        os_memory_limit_mode=(
            os_memory_limit_mode
            or os.environ.get("MRUN_OS_MEMORY_LIMIT_MODE", "off")
        ),
    )
    if echo:
        print(
            f"mrun.run: {stage.status} rc={stage.returncode} "
            f"wall={stage.elapsed_s}s peak_rss={stage.peak_rss_mb}MB log={log_path}",
            file=sys.stderr,
        )
        if stage.failure:
            print(
                "mrun.run: failure "
                f"kind={stage.failure.get('kind')} "
                f"phase={stage.failure.get('phase')} "
                f"message={stage.failure.get('message')}",
                file=sys.stderr,
            )
    return RunResult(stage=stage, log_path=str(log_path))
