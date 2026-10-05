"""mrun — centralized model runner/trainer library + fleet scheduler client.

One source of truth for: engine backends (HF, paged QStore, dense QStore CUDA, resident
OLMoE CUDA, paged Qwen3 MoE CUDA, MLX/Metal, experimental Core ML, and multifabric), model
registry, resource estimation, RAM/thread guards, run policy, guarded local execution, and
remote submission to the fleet scheduler.
"""

from .client.submit import RemoteResult, attach, remote_run
from .estimate import (
    MemoryEstimate,
    RunRecord,
    estimate_memory,
    estimate_resources,
    load_run_history,
    model_param_count,
)
from .execute import RunResult, run
from .guard import (
    RSS_LIMIT_MB,
    check_rss,
    cpu_seconds,
    ram_guard,
    rss_mb,
    rss_peak_mb,
    set_thread_env,
)
from .models import ModelSpec, resolve_model
from .orchestrate import StageRun, run_stage
from .policy import HostCaps, RunPlan, engine_from_plan, plan_run
from .resources import ResourceMonitor, require_free_space

__version__ = "0.1.2"


def __getattr__(name):
    # Lazy: importing mrun must stay light (agent/server hosts may lack torch).
    if name in {
        "CausalFamilyContinuationExample",
        "CausalFamilyExample",
        "close_pooled_engines",
        "evaluate_causal_family_batch",
        "evaluate_causal_family_continuation_batch",
        "open_engine",
    }:
        from . import engine

        return getattr(engine, name)
    if name in {"EngineReport", "engine_facts"}:
        from .engine import report

        return getattr(report, name)
    raise AttributeError(f"module 'mrun' has no attribute {name!r}")

__all__ = [
    "HostCaps",
    "RemoteResult",
    "attach",
    "remote_run",
    "MemoryEstimate",
    "ModelSpec",
    "ResourceMonitor",
    "RSS_LIMIT_MB",
    "RunPlan",
    "RunRecord",
    "RunResult",
    "StageRun",
    "check_rss",
    "CausalFamilyExample",
    "CausalFamilyContinuationExample",
    "close_pooled_engines",
    "engine_from_plan",
    "engine_facts",
    "EngineReport",
    "open_engine",
    "plan_run",
    "run",
    "cpu_seconds",
    "estimate_memory",
    "estimate_resources",
    "evaluate_causal_family_batch",
    "evaluate_causal_family_continuation_batch",
    "load_run_history",
    "model_param_count",
    "ram_guard",
    "require_free_space",
    "resolve_model",
    "rss_mb",
    "rss_peak_mb",
    "run_stage",
    "set_thread_env",
]
