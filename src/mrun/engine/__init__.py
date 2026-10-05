"""Model backend entry points."""

from __future__ import annotations

import atexit
import os
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from typing import Any

from .base import (
    EngineCapabilities,
    GenerationEngine,
    ModelEngine,
    PagedExpertEngine,
    apply_patch_ops,
    layer_maps_for_global_neurons,
    length_ops_buckets,
    summarize_forced_choice_rows,
)
from .hf import HFEngine
from .profiles import (
    EngineProfile,
    canonical_backend,
    engine_profiles,
    profile_for_backend,
    profile_id_for_backend,
)

_OPENED: list[weakref.ref] = []
_ATEXIT_ARMED = False


def _emit_engine_reports() -> None:
    """At process exit, write one engine graph per engine that was opened (issue I48).

    A reporting layer nobody calls is worth nothing: the two 2026-07-24 failures (an "ane"
    backend silently running 100% paged, and a Core ML path placing 98% of its ops on the
    GPU) were both knowable at runtime and recorded nowhere. This costs a JSON write per
    engine and needs no caller changes. Timing-per-stage still requires an explicit
    EngineReport; what this guarantees is that every run says WHAT executed.

    Off by default outside a job so ad-hoc shells are not littered: set MRUN_ENGINE_REPORT=1,
    or run under the agent, which sets MRUN_JOB_ID.
    """
    if not (os.environ.get("MRUN_ENGINE_REPORT") == "1" or os.environ.get("MRUN_JOB_ID")):
        return
    out_dir = os.environ.get("MRUN_REPORT_DIR") or "results"
    for index, ref in enumerate(_OPENED, start=1):
        engine = ref()
        if engine is None:
            continue
        try:
            from .report import EngineReport

            job = os.environ.get("MRUN_JOB_ID", "local")
            rep = EngineReport(
                engine,
                run_id=f"{job}-{index:02d}-{getattr(engine, 'backend', '?')}",
                note="auto-emitted at exit",
                started_at=getattr(engine, "_mrun_opened_at", None),
                start_proc=getattr(engine, "_mrun_open_proc_stats", None),
            )
            rep.save(out_dir, print_graph=False)
        except Exception:  # noqa: BLE001 — never fail a run over bookkeeping at shutdown
            continue


def _register_engine(engine: Any) -> Any:
    global _ATEXIT_ARMED
    try:
        from .report import _proc_stats

        engine._mrun_opened_at = time.perf_counter()
        engine._mrun_open_proc_stats = _proc_stats()
    except Exception:  # noqa: BLE001 — reporting is optional for constrained/test engines
        pass
    try:
        _OPENED.append(weakref.ref(engine))
        if not _ATEXIT_ARMED:
            atexit.register(_emit_engine_reports)
            _ATEXIT_ARMED = True
    except TypeError:  # engine not weak-referenceable (test doubles); reporting is optional
        pass
    return engine


_POOL: dict[tuple, tuple[ModelEngine, Callable[[], None]]] = {}
_POOL_LOCK = threading.RLock()


def _pool_backend_name(backend: str) -> str | None:
    normalized = backend.strip().lower().replace("_", "-")
    if normalized in {"apple", "apple-speed"}:
        return "mlx"
    # A multifabric engine owns several children. Pooling both the composite and its children
    # makes close ordering ambiguous and can release one child twice.
    if normalized == "multifabric":
        return None
    return normalized


def _pool_key(model_name: str, backend: str, kwargs: dict) -> tuple | None:
    """Identity of a reusable engine, or None if it must not be pooled.

    Only engines opened with identical construction arguments may share an instance, because
    those arguments (int4/int2 store choice, cache budget, stores_dir, or an explicit linked
    image path) change what the engine IS. Anything unhashable in kwargs means we cannot prove
    identity, so we do not pool.
    """
    if backend.strip().lower().replace("_", "-") in {
        "cuda-source-int8",
        "cuda-source-int8-compact-head",
    }:
        # Canonical/native paths are mutable locators, not identities. The production route
        # owns one engine and its strict source/blob guard instead of sharing a path-keyed pool.
        return None
    normalized_kwargs = dict(kwargs)
    component_graph = normalized_kwargs.get("component_graph")
    if (
        backend
        in {
            "paged",
            "dense-qstore-cuda",
            "dense-cuda",
            "mlx-component",
            "mlx-component-q4",
            "metal-component",
            "metal-component-q4",
        }
        and component_graph is not None
    ):
        # A mutable path is not an engine identity.  Validate the graph/manifests now and
        # pool on its semantic fingerprint; the engine performs a post-open file guard on
        # every hit, while unopened component blobs remain hash-gated on first use.
        from .kernels.composite_qstore import inspect_component_graph_fingerprint

        normalized_kwargs["component_graph"] = (
            "component-graph",
            inspect_component_graph_fingerprint(component_graph),
        )
        normalized_kwargs.setdefault("output_contract", "full_logits")
        component_budgets = normalized_kwargs.get("component_cache_mb")
        if isinstance(component_budgets, Mapping):
            normalized_kwargs["component_cache_mb"] = tuple(
                sorted((str(role), float(value)) for role, value in component_budgets.items())
            )
    try:
        key = (
            model_name,
            backend,
            tuple(sorted((key, value) for key, value in normalized_kwargs.items())),
        )
        hash(key)
        return key
    except TypeError:
        return None


def _component_fingerprint_from_pool_key(key: tuple) -> str | None:
    """Return the inspected component custody identity embedded in a pool key."""

    for name, value in key[2]:
        if (
            name == "component_graph"
            and isinstance(value, tuple)
            and len(value) == 2
            and value[0] == "component-graph"
            and isinstance(value[1], str)
        ):
            return value[1]
    return None


def _validate_opened_engine_pool_identity(engine: ModelEngine, key: tuple) -> None:
    """Close the inspect/open TOCTOU gap before a component engine enters the pool."""

    expected = _component_fingerprint_from_pool_key(key)
    if expected is None:
        return
    composite = getattr(engine, "composite_store", None)
    actual = getattr(composite, "composite_fingerprint_sha256", None)
    if actual is None:
        # The native MLX executor owns a strict graph reader rather than a CompositeQStore.
        # Both values are the inspected graph/manifests/blob custody fingerprint.
        graph = getattr(engine, "graph", None)
        actual = getattr(graph, "custody_fingerprint_sha256", None)
    if actual != expected:
        raise RuntimeError(
            "component graph changed between pool-key inspection and engine open "
            f"(inspected {expected!r}, opened {actual!r})"
        )


def _close_failed_pool_open(engine: ModelEngine) -> None:
    """Best-effort cleanup for an engine rejected before close deferral is installed."""

    close = getattr(engine, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # noqa: BLE001 — retain the identity/admission error
        pass


def _defer_close_while_pooled(engine: ModelEngine) -> Callable[[], None] | None:
    """Keep context-manager and explicit ``close`` calls from invalidating a pooled engine."""

    def deferred_close() -> None:
        return None

    try:
        real_close = engine.close
        engine.close = deferred_close  # type: ignore[method-assign]
    except (AttributeError, TypeError):
        return None
    return real_close


def open_engine(
    model_name: str, *, backend: str = "hf", fresh: bool = False, **kwargs: Any
) -> ModelEngine:
    """Open a backend, reusing a warm instance when one matches.

    Loading weights is the dominant cost of a short job: every job forks a fresh process and
    `mx` smoke-then-real pays TWO full loads per submit (OPEN-WORK estimates 2-10x on the
    short-job tail). Within a process, opening the same (model, backend, kwargs) twice is pure
    waste — the second load produces an identical object.

    Reuse is safe here because engine state is either shared-by-design or shared-desirably:
    the QStore dequant LRU and Core ML compiled-model cache are caches (that is the point),
    the scoring/fallback counters are cumulative telemetry, and a disabled accelerator
    (`_ane_ok=False` after a failed parity self-check) MUST persist — re-enabling a path
    measured to return wrong answers would be a regression. Weight patches are per-call
    arguments, not engine state, so they cannot leak between callers.

    Pass ``fresh=True`` (or set MRUN_ENGINE_REUSE=0) when a run genuinely needs an unshared
    instance — e.g. measuring cold-start cost, which reuse would otherwise hide.
    """
    reuse = not fresh and os.environ.get("MRUN_ENGINE_REUSE", "1") != "0"
    pool_backend = _pool_backend_name(backend) if reuse else None
    key = _pool_key(model_name, pool_backend, kwargs) if pool_backend is not None else None
    if key is None:
        return _register_engine(_open_engine_impl(model_name, backend=backend, **kwargs))
    # Serialize pooled construction. Besides preventing duplicate/leaked ring workers, this
    # intentionally prevents concurrent cold opens from briefly exceeding admission memory.
    with _POOL_LOCK:
        entry = _POOL.get(key)
        if entry is not None:
            assert_unchanged = getattr(entry[0], "assert_content_identity_unchanged", None)
            if callable(assert_unchanged):
                assert_unchanged()
            return entry[0]
        engine = _open_engine_impl(model_name, backend=backend, **kwargs)
        try:
            _validate_opened_engine_pool_identity(engine, key)
        except BaseException:
            _close_failed_pool_open(engine)
            raise
        engine = _register_engine(engine)
        real_close = _defer_close_while_pooled(engine)
        if real_close is not None:
            _POOL[key] = (engine, real_close)
        return engine


def close_pooled_engines() -> None:
    """Drop pooled engines (frees weights/caches). Safe to call between phases."""
    with _POOL_LOCK:
        entries = list(_POOL.values())
        _POOL.clear()
        # Keep construction excluded until every old engine has actually released its
        # component stores, ring workers, and mmaps.  Clearing the dictionary and dropping
        # the lock first creates a transient double-residency window: another thread can
        # cold-open the same model while ``real_close`` is still draining the old one.
        for engine, real_close in entries:
            try:
                delattr(engine, "close")
            except (AttributeError, TypeError):
                pass
            try:
                real_close()
            except Exception:  # noqa: BLE001
                pass


def _open_engine_impl(model_name: str, *, backend: str = "hf", **kwargs: Any) -> ModelEngine:
    backend = backend.lower()
    if backend == "hf":
        return HFEngine(model_name, **kwargs)
    if backend == "paged":
        from .paged import PagedEngine

        return PagedEngine(model_name, **kwargs)
    if backend in {"paged-fp32", "paged_fp32", "paged-lossless", "paged_lossless"}:
        from .paged import PagedEngine

        requested = kwargs.pop("fp32", True)
        if requested is not True:
            raise ValueError("paged-fp32 cannot be opened with fp32=False")
        return PagedEngine(model_name, fp32=True, **kwargs)
    if backend in {"paged-fp16", "paged_fp16", "paged-bf16", "paged_bf16"}:
        from .paged import PagedEngine

        requested_backend_dtype = "fp16" if "fp16" in backend else "bf16"
        requested = kwargs.pop("compute_dtype", requested_backend_dtype)
        if requested not in {requested_backend_dtype, None}:
            raise ValueError(
                f"{backend} is fixed to {requested_backend_dtype} arithmetic; "
                f"received compute_dtype={requested!r}"
            )
        return PagedEngine(
            model_name,
            fp32=True,
            compute_dtype=requested_backend_dtype,
            **kwargs,
        )
    if backend in {"dense-qstore-cuda", "dense_qstore_cuda", "dense-cuda"}:
        from .dense_qstore_cuda import DenseQStoreCudaEngine

        return DenseQStoreCudaEngine(model_name, **kwargs)
    if backend in {"cuda-source-int8", "cuda_source_int8"}:
        from .dense_qstore_cuda import DenseSourceCudaInt8Engine

        return DenseSourceCudaInt8Engine(model_name, **kwargs)
    if backend in {
        "cuda-source-int8-compact-head",
        "cuda_source_int8_compact_head",
    }:
        from .dense_qstore_cuda import DenseSourceCudaInt8CompactHeadEngine

        return DenseSourceCudaInt8CompactHeadEngine(model_name, **kwargs)
    if backend in {"olmoe-cuda", "olmoe_cuda"}:
        from .olmoe_cuda import OLMoECudaEngine

        return OLMoECudaEngine(model_name, **kwargs)
    if backend in {
        "qwen3-moe-cuda",
        "qwen3_moe_cuda",
        "moe-qstore-cuda",
        "moe_qstore_cuda",
    }:
        from .qwen3_moe_cuda import Qwen3MoeCudaEngine

        return Qwen3MoeCudaEngine(model_name, **kwargs)
    if backend in {"moe-stream", "moe_stream"}:
        from .moe_stream import MoEStreamEngine

        return MoEStreamEngine(model_name, **kwargs)
    if backend in {"apple", "apple-speed"}:
        # Named Apple speed path. MEASURED 2026-07-24 (Qwen2.5-0.5B, B=8/T=9, warm median
        # of 5): mlx 2196.8 tok/s / 0.62 s cold vs coreml 1694.5 tok/s / 66.93 s cold —
        # 1.30x faster warm, ~108x faster cold, and none of the Core ML tax (19.3 s compile
        # per shape, 943 MB per package, fp16->fp32 upcast in the native binding, opaque
        # placement). Core ML is reachable as backend="ane" for experiments only.
        backend = "mlx"
    if backend == "mlx":
        from .mlx import open_mlx_engine

        return open_mlx_engine(model_name, **kwargs)
    if backend in {"mlx-q4", "mlx_q4"}:
        # int4 g64 quantized-resident Metal path (mx.quantized_matmul). Opt-in SPEED backend:
        # fp32 numpy stays the canonical oracle; gate capability, not per-position argmax.
        from .mlx import open_mlx_engine

        return open_mlx_engine(model_name, quantized="q4g64", **kwargs)
    if backend in {
        "mlx-component",
        "mlx_component",
        "metal-component",
        "metal_component",
    }:
        from .mlx_component import MLXComponentEngine

        return MLXComponentEngine(model_name, **kwargs)
    if backend in {
        "mlx-component-q4",
        "mlx_component_q4",
        "metal-component-q4",
        "metal_component_q4",
    }:
        from .mlx_component import MLXComponentQ4Engine

        return MLXComponentQ4Engine(model_name, **kwargs)
    if backend in {"ane", "coreml"}:
        # Core ML accelerator-eligible speed backend (qwen2/llama, fp16). The legacy `ane`
        # name does not prove placement: compute_units=ALL may select CPU/GPU/ANE/mixed.
        # coremltools is optional, so fall back to paged on availability misses.
        try:
            from .ane import ANEPagedEngine
        except Exception:
            from .paged import PagedEngine

            return PagedEngine(model_name, **kwargs)
        engine = ANEPagedEngine(model_name, **kwargs)
        engine._requested_backend_alias = backend
        return engine
    if backend == "multifabric":
        # Split a batch across CPU paged + MLX/Metal and run them concurrently. Core ML is
        # excluded by default because measured placement shares the GPU with MLX.
        from .multifabric import open_multifabric_engine

        return open_multifabric_engine(model_name, **kwargs)
    raise ValueError(
        "backend must be one of: hf, paged, paged-fp32, paged-fp16, paged-bf16, "
        "dense-qstore-cuda, cuda-source-int8, "
        "cuda-source-int8-compact-head, olmoe-cuda, "
        "qwen3-moe-cuda, moe-stream, mlx, mlx-q4, mlx-component, mlx-component-q4, "
        "metal-component, metal-component-q4, apple, apple-speed, ane, coreml, "
        "multifabric"
    )


from .causal_family import (  # noqa: E402
    CausalFamilyContinuationExample,
    CausalFamilyExample,
    evaluate_causal_family_batch,
    evaluate_causal_family_continuation_batch,
)
from .continuous import (  # noqa: E402 - public serving API after backend registry construction
    ContinuousBatchingPolicy,
    ContinuousPagedRequest,
    ContinuousPagedResult,
    ContinuousPagedService,
    ContinuousServiceTelemetry,
    ContinuousServingCapabilities,
    ContinuousStepRecord,
    LatencyDistribution,
    resolve_continuous_serving_capabilities,
)
from .graph_pool import (  # noqa: E402
    BoundGraphResult,
    GraphTemplate,
    GraphTemplateKey,
    GraphTemplatePool,
    ResidentQStoreArena,
    ResidentQStoreArenaKey,
)
from .kernels.body_only_qstore import (  # noqa: E402
    build_body_only_qstore,
    inspect_body_only_qstore,
)
from .kernels.lexical import (  # noqa: E402
    LexicalBinding,
    LexicalComponent,
    LexicalValues,
    LexicalWeights,
    train_lexical_bridge,
)
from .kernels.lexical_compiler import (  # noqa: E402
    LexicalCompilationResult,
    compile_lexical_component,
    inspect_lexical_compilation,
    load_lexical_compilation,
)
from .kernels.token_address_map import (  # noqa: E402
    TokenAddressMap,
    inspect_token_address_map,
)
from .qwen3_moe_statecut import (  # noqa: E402
    Qwen3MoeKVStateCut,
    Qwen3MoeKVStateCutBranch,
    Qwen3MoeKVStateCutContinuation,
    Qwen3MoeKVStateCutDescriptor,
    Qwen3MoeKVStateCutReceipt,
)
from .resident_worker import (  # noqa: E402
    ResidentModelWorker,
    ResidentSelectedScoreRequest,
    ResidentSelectedScoreResponse,
    build_dense_resident_worker,
    resident_executable_inventory,
)
from .statecut import (  # noqa: E402
    STATECUT_ADJUDICATION_SCHEMA,
    STATECUT_CONTINUATION_SCHEMA,
    STATECUT_RECEIPT_SCHEMA,
    STATECUT_SCHEMA,
    STATECUT_SCREENING_SCHEMA,
    PagedKVStateCut,
    PagedKVStateCutAdjudication,
    PagedKVStateCutBranch,
    PagedKVStateCutContinuation,
    PagedKVStateCutDescriptor,
    PagedKVStateCutProjectionAccounting,
    PagedKVStateCutProjectionMemory,
    PagedKVStateCutReceipt,
    PagedKVStateCutScreening,
    prefill_paged_kv_statecut,
)

__all__ = [
    "ContinuousBatchingPolicy",
    "BoundGraphResult",
    "CausalFamilyExample",
    "ContinuousPagedRequest",
    "ContinuousPagedResult",
    "ContinuousPagedService",
    "ContinuousStepRecord",
    "ContinuousServiceTelemetry",
    "ContinuousServingCapabilities",
    "EngineCapabilities",
    "GenerationEngine",
    "GraphTemplate",
    "GraphTemplateKey",
    "GraphTemplatePool",
    "HFEngine",
    "ModelEngine",
    "PagedExpertEngine",
    "PagedKVStateCut",
    "PagedKVStateCutAdjudication",
    "PagedKVStateCutBranch",
    "PagedKVStateCutContinuation",
    "PagedKVStateCutDescriptor",
    "PagedKVStateCutProjectionAccounting",
    "PagedKVStateCutProjectionMemory",
    "PagedKVStateCutReceipt",
    "PagedKVStateCutScreening",
    "Qwen3MoeKVStateCut",
    "Qwen3MoeKVStateCutBranch",
    "Qwen3MoeKVStateCutContinuation",
    "Qwen3MoeKVStateCutDescriptor",
    "Qwen3MoeKVStateCutReceipt",
    "ResidentModelWorker",
    "ResidentQStoreArena",
    "ResidentQStoreArenaKey",
    "ResidentSelectedScoreRequest",
    "ResidentSelectedScoreResponse",
    "apply_patch_ops",
    "build_dense_resident_worker",
    "close_pooled_engines",
    "layer_maps_for_global_neurons",
    "length_ops_buckets",
    "LatencyDistribution",
    "LexicalBinding",
    "LexicalComponent",
    "LexicalValues",
    "LexicalWeights",
    "LexicalCompilationResult",
    "TokenAddressMap",
    "compile_lexical_component",
    "inspect_lexical_compilation",
    "load_lexical_compilation",
    "build_body_only_qstore",
    "inspect_body_only_qstore",
    "inspect_token_address_map",
    "open_engine",
    "EngineProfile",
    "canonical_backend",
    "engine_profiles",
    "profile_for_backend",
    "profile_id_for_backend",
    "prefill_paged_kv_statecut",
    "evaluate_causal_family_batch",
    "evaluate_causal_family_continuation_batch",
    "CausalFamilyContinuationExample",
    "resolve_continuous_serving_capabilities",
    "resident_executable_inventory",
    "summarize_forced_choice_rows",
    "train_lexical_bridge",
    "STATECUT_SCHEMA",
    "STATECUT_SCREENING_SCHEMA",
    "STATECUT_ADJUDICATION_SCHEMA",
    "STATECUT_CONTINUATION_SCHEMA",
    "STATECUT_RECEIPT_SCHEMA",
]
