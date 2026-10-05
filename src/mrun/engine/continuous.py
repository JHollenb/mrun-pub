"""Continuous greedy serving over the transactional Paged component reactor.

The component reactor owns one physical wave.  This module owns the longer-lived request
lifecycle: admission, per-request KV authority, greedy selection, explicit commit, refill,
cancellation, deadlines, and bounded service telemetry.  No callback commits state; one
coordinator thread is the sole commit and lease-release authority.

The route resolver is deliberately fail closed.  Generic engine capability booleans do not
authorize this adapter: only the exact CPU :class:`PagedEngine` implementation, an eligible
architecture, a full-vocabulary output view, and the versioned scratch-only component route
produce a route record.  That record is revalidated before every dispatch.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
import torch

from mrun.compiler import (
    ExecutionMode,
    OutputContract,
    bind_versioned_kv_state,
    build_paged_qstore_plan,
    decompose_work_plan,
    lower_work_template,
)

from .kernels import paged_forward as pf
from .paged_reactor import (
    PagedComponentReactor,
    PagedReactorPayload,
    PagedReactorResult,
)
from .reactor import (
    ComponentSubmission,
    ReactorDeadlineExceeded,
    template_compatibility_key,
)

_MAX_REQUEST_ID_UTF8_BYTES = 256


class ContinuousServiceError(RuntimeError):
    """Base class for continuous-serving failures."""


class ContinuousRouteUnavailable(ContinuousServiceError):
    """No exact, trusted continuous-serving adapter matches the engine."""


class ContinuousServiceClosed(ContinuousServiceError):
    """Admission was attempted after shutdown began."""


class ContinuousAdmissionError(ContinuousServiceError):
    """A request failed synchronous service admission."""


class ContinuousBackpressureError(ContinuousAdmissionError):
    """The bounded active-request or KV-byte budget is exhausted."""


class ContinuousDuplicateRequestError(ContinuousAdmissionError):
    """A request ID was already used in this service lifetime."""


class ContinuousOutputError(ContinuousServiceError):
    """A step returned malformed or unsafe logits."""


class ContinuousRequestDeadlineExceeded(TimeoutError, ContinuousServiceError):
    """A request expired before its next token could be committed."""

    def __init__(self, request_id: str) -> None:
        super().__init__(f"continuous request deadline exceeded: {request_id}")
        self.request_id = request_id


class _CommitDeadlineExpired(RuntimeError):
    """A provisional commit crossed its request deadline and was rolled back."""


class ContinuousBatchingPolicy(str, Enum):
    """Per-request coalescing preference."""

    ADAPTIVE = "adaptive"
    PREFER_BATCH = "prefer_batch"
    LATENCY = "latency"


@dataclass(frozen=True, slots=True)
class ContinuousServingCapabilities:
    """Immutable exact route record, not a generic positive capability flag."""

    schema: Literal["mrun-serving-route-v1"]
    route_id: str
    engine_class: str
    engine_instance_id: int
    store_instance_id: int
    engine_backend: Literal["paged"]
    adapter_abi: Literal["paged-continuous-service-v1"]
    architecture: Literal["qwen2", "llama", "qwen3"]
    fabric: Literal["cpu"]
    output_contract: Literal["last_token_logits"]
    component_output_view: Literal["full_logits"]
    state_abi: Literal["versioned-provisional-kv-v1"]
    state_owner: Literal["per-request-b1"]
    binding_fields: tuple[str, ...]
    mutation: Literal["scratch-only"]
    commit_abi: Literal["explicit-prefix-per-request-v1"]
    slot_authority: Literal["cache-issued-aba-safe-v1"]
    batching_abi: Literal["paged-component-reactor-v1"]
    pooled_arithmetic: tuple[pf.PagedPooledArithmetic, ...]
    identity_snapshot: tuple[tuple[str, str], ...]
    promotion_status: Literal["local-candidate"]
    promotion_gaps: tuple[str, ...]

    def assert_current(
        self,
        engine: Any,
        arithmetic: pf.PagedPooledArithmetic,
        *,
        revalidate_content: bool = True,
    ) -> None:
        """Revalidate the exact engine/store snapshot bound by this route."""

        engine_class = f"{type(engine).__module__}.{type(engine).__qualname__}"
        if engine_class != self.engine_class or id(engine) != self.engine_instance_id:
            raise ContinuousRouteUnavailable("continuous route engine identity drifted")
        store = getattr(engine, "store", None)
        if store is None or id(store) != self.store_instance_id:
            raise ContinuousRouteUnavailable("continuous route store identity drifted")
        if str(getattr(engine, "backend", "")) != self.engine_backend:
            raise ContinuousRouteUnavailable("continuous route backend drifted")
        if str(getattr(engine, "arch", "")) != self.architecture:
            raise ContinuousRouteUnavailable("continuous route architecture drifted")
        device = torch.device(getattr(engine, "device", "cpu"))
        store_device = torch.device(getattr(store, "device", "cpu"))
        if device.type != self.fabric or store_device.type != self.fabric:
            raise ContinuousRouteUnavailable("continuous route fabric drifted")
        component_view = getattr(engine, "component_output_contract", None) or "full_logits"
        if component_view != self.component_output_view:
            raise ContinuousRouteUnavailable("continuous route component output view drifted")
        if arithmetic not in self.pooled_arithmetic:
            raise ContinuousRouteUnavailable(
                f"continuous route does not admit pooled arithmetic {arithmetic!r}"
            )
        current_identity = _route_identity_snapshot(engine)
        if current_identity != self.identity_snapshot:
            raise ContinuousRouteUnavailable("continuous route content identity fields drifted")
        if revalidate_content:
            assert_unchanged = getattr(engine, "assert_content_identity_unchanged", None)
            if not callable(assert_unchanged):
                raise ContinuousRouteUnavailable(
                    "continuous route requires content-identity revalidation"
                )
            assert_unchanged()


def _route_identity_snapshot(engine: Any) -> tuple[tuple[str, str], ...]:
    store = getattr(engine, "store", None)
    manifest = getattr(store, "man", None)
    if not isinstance(manifest, dict):
        raise ContinuousRouteUnavailable("paged continuous route requires a QStore manifest")
    identity_keys = (
        "model_name",
        "model_revision",
        "store_fingerprint",
        "manifest_semantic_sha256",
        "identity_certificate_sha256",
        "source_identity_status",
        "store_identity_status",
    )
    values = [(key, str(manifest.get(key, ""))) for key in identity_keys]
    composite = getattr(engine, "composite_store", None)
    values.append(
        (
            "component_graph",
            str(getattr(composite, "composite_fingerprint_sha256", "monolithic")),
        )
    )
    return tuple(values)


def resolve_continuous_serving_capabilities(
    engine: Any,
) -> ContinuousServingCapabilities:
    """Resolve the sole local continuous route or deny the engine.

    The exact concrete class check prevents aliases, subclasses, and test doubles from gaining
    authority merely by copying ``backend='paged'`` and generic capability booleans.
    """

    from .paged import PagedEngine

    if type(engine) is not PagedEngine:
        raise ContinuousRouteUnavailable(
            "continuous serving requires the exact mrun.engine.paged.PagedEngine class"
        )
    if str(getattr(engine, "backend", "")) != "paged":
        raise ContinuousRouteUnavailable("continuous serving requires the actual paged backend")
    architecture = str(getattr(engine, "arch", ""))
    if architecture not in {"qwen2", "llama", "qwen3"}:
        raise ContinuousRouteUnavailable(
            f"continuous serving is unavailable for paged architecture {architecture!r}"
        )
    store = getattr(engine, "store", None)
    if store is None:
        raise ContinuousRouteUnavailable("continuous serving requires an opened paged store")
    device = torch.device(getattr(engine, "device", "cpu"))
    store_device = torch.device(getattr(store, "device", "cpu"))
    if device.type != "cpu" or store_device.type != "cpu":
        raise ContinuousRouteUnavailable(
            "the current continuous component route is CPU-only; native CUDA is unpromoted"
        )
    if not callable(getattr(engine, "_can_batch", None)) or not engine._can_batch():  # noqa: SLF001
        raise ContinuousRouteUnavailable("paged engine cannot execute pooled transformer waves")
    component_view = getattr(engine, "component_output_contract", None) or "full_logits"
    if component_view != "full_logits":
        raise ContinuousRouteUnavailable(
            "open-vocabulary continuous generation requires the full_logits component view"
        )
    execution_lock = getattr(engine, "_execution_lock", None)
    if not callable(getattr(execution_lock, "acquire", None)) or not callable(
        getattr(execution_lock, "release", None)
    ):
        raise ContinuousRouteUnavailable("paged engine lacks its execution transaction lock")
    capabilities = engine.capabilities()
    if not bool(getattr(capabilities, "transactional_kv", False)) or not bool(
        getattr(capabilities, "speculative_blocks", False)
    ):
        raise ContinuousRouteUnavailable("paged engine lacks transactional scratch-only KV")
    semantic_tokens = getattr(engine, "semantic_token_count", None)
    if (
        isinstance(semantic_tokens, bool)
        or not isinstance(semantic_tokens, Integral)
        or int(semantic_tokens) <= 0
    ):
        raise ContinuousRouteUnavailable("paged engine lacks a positive semantic token domain")
    identity = _route_identity_snapshot(engine)
    assert_unchanged = getattr(engine, "assert_content_identity_unchanged", None)
    if not callable(assert_unchanged):
        raise ContinuousRouteUnavailable("paged engine lacks content-identity revalidation")
    assert_unchanged()
    engine_class = f"{type(engine).__module__}.{type(engine).__qualname__}"
    route_document = {
        "schema": "mrun-serving-route-v1",
        "engine_class": engine_class,
        "engine_backend": "paged",
        "architecture": architecture,
        "fabric": "cpu",
        "state_abi": "versioned-provisional-kv-v1",
        "batching_abi": "paged-component-reactor-v1",
        "identity": identity,
    }
    route_id = hashlib.sha256(
        json.dumps(route_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ContinuousServingCapabilities(
        schema="mrun-serving-route-v1",
        route_id=route_id,
        engine_class=engine_class,
        engine_instance_id=id(engine),
        store_instance_id=id(store),
        engine_backend="paged",
        adapter_abi="paged-continuous-service-v1",
        architecture=cast(Literal["qwen2", "llama", "qwen3"], architecture),
        fabric="cpu",
        output_contract="last_token_logits",
        component_output_view="full_logits",
        state_abi="versioned-provisional-kv-v1",
        state_owner="per-request-b1",
        binding_fields=("epoch", "lengths", "capacity", "cache_id", "storage_signature"),
        mutation="scratch-only",
        commit_abi="explicit-prefix-per-request-v1",
        slot_authority="cache-issued-aba-safe-v1",
        batching_abi="paged-component-reactor-v1",
        pooled_arithmetic=("row_stable", "row_stable_split"),
        identity_snapshot=identity,
        promotion_status="local-candidate",
        promotion_gaps=(
            "production promotion record",
            "long-context service curve",
            "native CUDA/MLX/MoE adapter",
            "byte admission before caller-owned cache allocation",
        ),
    )


@dataclass(frozen=True, slots=True)
class ContinuousPagedRequest:
    """One autonomous greedy request and optional pre-existing B1 KV authority."""

    request_id: str
    engine: Any = field(repr=False, compare=False)
    input_ids: tuple[int, ...]
    max_new_tokens: int
    cache: pf.BatchedPagedKVCache | None = field(default=None, repr=False, compare=False)
    slot_lease: pf.PagedKVSlotLease | None = field(default=None, repr=False, compare=False)
    kv_capacity: int | None = None
    eos_token_id: int | None = None
    deadline: float | None = None
    batching_policy: ContinuousBatchingPolicy | str = ContinuousBatchingPolicy.ADAPTIVE

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise TypeError("continuous request_id must be a non-empty string")
        if self.request_id != self.request_id.strip():
            raise ValueError("continuous request_id cannot have surrounding whitespace")
        if len(self.request_id.encode("utf-8")) > _MAX_REQUEST_ID_UTF8_BYTES:
            raise ValueError(
                f"continuous request_id cannot exceed {_MAX_REQUEST_ID_UTF8_BYTES} UTF-8 bytes"
            )
        if self.engine is None:
            raise TypeError("continuous request requires an engine owner")
        raw_ids = tuple(self.input_ids)
        if not raw_ids:
            raise ValueError("continuous request input_ids cannot be empty")
        normalized_ids: list[int] = []
        for token in raw_ids:
            if isinstance(token, bool) or not isinstance(token, Integral):
                raise TypeError("continuous request token IDs must be integers")
            normalized_ids.append(int(token))
        object.__setattr__(self, "input_ids", tuple(normalized_ids))
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, Integral)
            or int(self.max_new_tokens) <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        object.__setattr__(self, "max_new_tokens", int(self.max_new_tokens))
        if (self.cache is None) != (self.slot_lease is None):
            raise ValueError("cache and slot_lease must either both be present or both be absent")
        if self.kv_capacity is not None and (
            isinstance(self.kv_capacity, bool)
            or not isinstance(self.kv_capacity, Integral)
            or int(self.kv_capacity) <= 0
        ):
            raise ValueError("kv_capacity must be a positive integer or None")
        if self.kv_capacity is not None:
            object.__setattr__(self, "kv_capacity", int(self.kv_capacity))
        if self.eos_token_id is not None:
            if isinstance(self.eos_token_id, bool) or not isinstance(self.eos_token_id, Integral):
                raise TypeError("eos_token_id must be an integer or None")
            object.__setattr__(self, "eos_token_id", int(self.eos_token_id))
        if self.deadline is not None:
            if isinstance(self.deadline, bool) or not isinstance(self.deadline, (int, float)):
                raise TypeError("deadline must be a finite monotonic timestamp or None")
            normalized_deadline = float(self.deadline)
            if not math.isfinite(normalized_deadline):
                raise ValueError("deadline must be a finite monotonic timestamp or None")
            object.__setattr__(self, "deadline", normalized_deadline)
        try:
            policy = (
                self.batching_policy
                if isinstance(self.batching_policy, ContinuousBatchingPolicy)
                else ContinuousBatchingPolicy(str(self.batching_policy))
            )
        except ValueError as exc:
            choices = ", ".join(policy.value for policy in ContinuousBatchingPolicy)
            raise ValueError(f"batching_policy must be one of: {choices}") from exc
        object.__setattr__(self, "batching_policy", policy)


@dataclass(frozen=True, slots=True)
class ContinuousStepRecord:
    step_index: int
    reactor_batch_id: int
    reactor_dispatch_id: int
    wave_width: int
    input_count: int
    token_id: int
    committed_cache_length: int
    refill: bool


@dataclass(frozen=True, slots=True)
class ContinuousPagedResult:
    request_id: str
    generated_token_ids: tuple[int, ...]
    completion_reason: Literal["max_new_tokens", "eos_token"]
    committed_input_count: int
    starting_cache_length: int
    final_cache_length: int
    final_cache_epoch: int
    pending_token_id: int
    step_count: int
    ttft_seconds: float
    inter_token_seconds: tuple[float, ...]
    request_latency_seconds: float
    route_id: str
    pooled_arithmetic: pf.PagedPooledArithmetic
    step_records: tuple[ContinuousStepRecord, ...]


@dataclass(frozen=True, slots=True)
class LatencyDistribution:
    observation_count: int
    window_count: int
    window_capacity: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    p50: float | None
    p95: float | None
    p99: float | None


class _BoundedLatency:
    __slots__ = ("capacity", "observation_count", "values")

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.observation_count = 0
        self.values: deque[float] = deque(maxlen=capacity)

    def add(self, value: float) -> None:
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError("latency observations must be finite and non-negative")
        self.observation_count += 1
        self.values.append(normalized)

    @staticmethod
    def _quantile(sorted_values: tuple[float, ...], q: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = (len(sorted_values) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return sorted_values[lower]
        fraction = position - lower
        return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction

    def snapshot(self) -> LatencyDistribution:
        ordered = tuple(sorted(self.values))
        if not ordered:
            return LatencyDistribution(
                observation_count=self.observation_count,
                window_count=0,
                window_capacity=self.capacity,
                minimum=None,
                maximum=None,
                mean=None,
                p50=None,
                p95=None,
                p99=None,
            )
        return LatencyDistribution(
            observation_count=self.observation_count,
            window_count=len(ordered),
            window_capacity=self.capacity,
            minimum=ordered[0],
            maximum=ordered[-1],
            mean=sum(ordered) / len(ordered),
            p50=self._quantile(ordered, 0.50),
            p95=self._quantile(ordered, 0.95),
            p99=self._quantile(ordered, 0.99),
        )


@dataclass(frozen=True, slots=True)
class ContinuousServiceTelemetry:
    admitted: int
    active: int
    completed: int
    cancelled: int
    deadline_expired: int
    failed: int
    shutdown_terminated: int
    rejected_admission: int
    rejected_backpressure: int
    rejected_duplicate: int
    rejected_request_history: int
    steps_submitted: int
    steps_succeeded: int
    steps_failed: int
    steps_cancelled: int
    steps_abandoned: int
    steps_active: int
    generated_tokens: int
    refills: int
    leases_released: int
    aggregate_template_cache_hits: int
    aggregate_template_cache_misses: int
    template_cache_size: int
    template_cache_capacity: int
    template_cache_evictions: int
    request_history_size: int
    request_history_capacity: int
    pooled_attention_global_prefix_kv_logical_bytes_max: int
    pooled_attention_request_local_prefix_kv_logical_bytes_max: int
    pooled_attention_explicit_live_prefix_kv_peak_bytes_max: int
    pooled_aggregate_provisional_delta_bytes_max: int
    active_committed_kv_arena_bytes: int
    peak_active_committed_kv_arena_bytes: int
    max_committed_kv_arena_bytes: int | None
    ttft: LatencyDistribution
    inter_token: LatencyDistribution
    request_latency: LatencyDistribution
    step_latency: LatencyDistribution
    accepting: bool
    stopped: bool
    route_id: str
    route_promotion_status: str
    pooled_arithmetic: pf.PagedPooledArithmetic
    reactor: Any

    @property
    def reconciled(self) -> bool:
        terminal = (
            self.completed
            + self.cancelled
            + self.deadline_expired
            + self.failed
            + self.shutdown_terminated
        )
        accounted_steps = (
            self.steps_succeeded + self.steps_failed + self.steps_cancelled + self.steps_abandoned
        )
        reactor_terminal = (
            self.reactor.succeeded
            + self.reactor.failed
            + self.reactor.cancelled
            + self.reactor.deadline_expired
        )
        return bool(
            self.admitted == self.active + terminal
            and self.leases_released == terminal
            and self.steps_submitted == accounted_steps + self.steps_active
            and self.steps_submitted == self.reactor.submitted
            and self.generated_tokens == self.steps_succeeded
            # Reactor outcome counters are recorded immediately before Future publication,
            # while ``outstanding`` falls in the Future callback.  Admit that bounded overlap
            # during a live snapshot; at idle this collapses to exact equality.
            and reactor_terminal <= self.reactor.submitted
            and self.reactor.submitted <= reactor_terminal + self.reactor.outstanding
        )


class _RequestFuture(Future[ContinuousPagedResult]):
    def __init__(self, owner: ContinuousPagedService, request_id: str) -> None:
        super().__init__()
        self._owner = owner
        self._request_id = request_id

    def cancel(self) -> bool:
        return self._owner._cancel_public_future(self)  # noqa: SLF001

    def _cancel_under_owner(self) -> bool:
        return super().cancel()


@dataclass(slots=True)
class _Session:
    request: ContinuousPagedRequest
    future: _RequestFuture
    cache: pf.BatchedPagedKVCache
    lease: pf.PagedKVSlotLease
    kv_handle: str
    kv_bytes: int
    service_allocated_cache: bool
    arrival_at: float
    starting_length: int
    current_ids: tuple[int, ...]
    expected_epoch: int
    expected_length: int
    expected_storage_signature: tuple[tuple[Any, ...], ...]
    generated: list[int] = field(default_factory=list)
    step_records: list[ContinuousStepRecord] = field(default_factory=list)
    committed_input_count: int = 0
    phase: str = "ready"
    step_future: Future[PagedReactorResult] | None = None
    step_sequence: int = 0
    step_accounted: bool = True
    step_started_at: float | None = None
    last_token_at: float | None = None
    first_token_latency: float | None = None
    inter_token_seconds: list[float] = field(default_factory=list)
    cancel_requested: bool = False
    cancel_reason: Literal["cancelled", "shutdown"] = "cancelled"
    lease_released: bool = False


@dataclass(frozen=True, slots=True)
class _StepDone:
    request_id: str
    step_sequence: int
    future: Future[PagedReactorResult]


class ContinuousPagedService:
    """Bounded autonomous greedy service over one exact Paged engine owner."""

    def __init__(
        self,
        engine: Any,
        *,
        max_active_requests: int,
        max_batch_size: int = 32,
        max_batch_delay_seconds: float = 0.001,
        max_committed_kv_arena_bytes: int | None = None,
        pooled_arithmetic: pf.PagedPooledArithmetic = "row_stable_split",
        telemetry_history: int = 256,
        max_aggregate_templates: int = 256,
        max_request_history: int = 65_536,
        clock: Callable[[], float] = time.monotonic,
        thread_name: str = "mrun-continuous-paged-service",
    ) -> None:
        if (
            isinstance(max_active_requests, bool)
            or not isinstance(max_active_requests, Integral)
            or int(max_active_requests) <= 0
        ):
            raise ValueError("max_active_requests must be a positive integer")
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, Integral)
            or int(max_batch_size) <= 0
            or int(max_batch_size) > int(max_active_requests)
        ):
            raise ValueError(
                "max_batch_size must be positive and no larger than max_active_requests"
            )
        if max_committed_kv_arena_bytes is not None and (
            isinstance(max_committed_kv_arena_bytes, bool)
            or not isinstance(max_committed_kv_arena_bytes, Integral)
            or int(max_committed_kv_arena_bytes) <= 0
        ):
            raise ValueError("max_committed_kv_arena_bytes must be a positive integer or None")
        if (
            isinstance(telemetry_history, bool)
            or not isinstance(telemetry_history, Integral)
            or int(telemetry_history) <= 0
        ):
            raise ValueError("telemetry_history must be a positive integer")
        if (
            isinstance(max_aggregate_templates, bool)
            or not isinstance(max_aggregate_templates, Integral)
            or int(max_aggregate_templates) <= 0
        ):
            raise ValueError("max_aggregate_templates must be a positive integer")
        if (
            isinstance(max_request_history, bool)
            or not isinstance(max_request_history, Integral)
            or int(max_request_history) < int(max_active_requests)
        ):
            raise ValueError(
                "max_request_history must be an integer no smaller than max_active_requests"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._engine = engine
        self._route = resolve_continuous_serving_capabilities(engine)
        self._pooled_arithmetic = pooled_arithmetic
        self._route.assert_current(engine, pooled_arithmetic)
        self._max_active = int(max_active_requests)
        self._max_committed_kv_arena_bytes = (
            int(max_committed_kv_arena_bytes) if max_committed_kv_arena_bytes is not None else None
        )
        self._max_request_history = int(max_request_history)
        self._max_template_cache = int(max_aggregate_templates)
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._events: deque[_StepDone] = deque()
        self._active: dict[str, _Session] = {}
        self._seen_request_ids: set[str] = set()
        self._template_cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._accepting = True
        self._shutdown_mode: Literal["drain", "abort"] | None = None
        self._stopped = threading.Event()

        self._admitted = 0
        self._completed = 0
        self._cancelled = 0
        self._deadline_expired = 0
        self._failed = 0
        self._shutdown_terminated = 0
        self._rejected_admission = 0
        self._rejected_backpressure = 0
        self._rejected_duplicate = 0
        self._rejected_request_history = 0
        self._steps_submitted = 0
        self._steps_succeeded = 0
        self._steps_failed = 0
        self._steps_cancelled = 0
        self._steps_abandoned = 0
        self._generated_tokens = 0
        self._refills = 0
        self._leases_released = 0
        self._aggregate_template_cache_hits = 0
        self._aggregate_template_cache_misses = 0
        self._template_cache_evictions = 0
        self._pooled_attention_global_prefix_kv_logical_bytes_max = 0
        self._pooled_attention_request_local_prefix_kv_logical_bytes_max = 0
        self._pooled_attention_explicit_live_prefix_kv_peak_bytes_max = 0
        self._pooled_aggregate_provisional_delta_bytes_max = 0
        self._active_committed_kv_arena_bytes = 0
        self._peak_active_committed_kv_arena_bytes = 0
        self._ttft = _BoundedLatency(int(telemetry_history))
        self._inter_token = _BoundedLatency(int(telemetry_history))
        self._request_latency = _BoundedLatency(int(telemetry_history))
        self._step_latency = _BoundedLatency(int(telemetry_history))

        self._reactor = PagedComponentReactor(
            max_batch_size=int(max_batch_size),
            max_pending=int(max_active_requests),
            max_batch_delay_seconds=max_batch_delay_seconds,
            telemetry_history=int(telemetry_history),
            max_aggregate_templates=max_aggregate_templates,
            pooled_arithmetic=pooled_arithmetic,
            clock=clock,
            thread_name=f"{thread_name}-reactor",
        )
        self._coordinator = threading.Thread(
            target=self._coordinator_main,
            name=thread_name,
            daemon=True,
        )
        self._coordinator.start()

    @property
    def capabilities(self) -> ContinuousServingCapabilities:
        return self._route

    def submit(self, request: ContinuousPagedRequest) -> Future[ContinuousPagedResult]:
        return self.submit_many((request,))[0]

    def submit_many(
        self,
        requests: Sequence[ContinuousPagedRequest],
    ) -> tuple[Future[ContinuousPagedResult], ...]:
        cohort = tuple(requests)
        if not cohort:
            raise ValueError("continuous request cohort cannot be empty")
        if any(not isinstance(request, ContinuousPagedRequest) for request in cohort):
            raise TypeError("requests must contain ContinuousPagedRequest values")

        with self._condition:
            if not self._accepting:
                self._rejected_admission += len(cohort)
                raise ContinuousServiceClosed("continuous service is shutting down")
            # Admission must remain live while another request owns the engine transaction lock.
            # The coordinator performs the full content guard immediately before every dispatch.
            self._route.assert_current(
                self._engine,
                self._pooled_arithmetic,
                revalidate_content=False,
            )
            request_ids = tuple(request.request_id for request in cohort)
            if len(request_ids) != len(set(request_ids)) or any(
                request_id in self._seen_request_ids for request_id in request_ids
            ):
                self._rejected_admission += len(cohort)
                self._rejected_duplicate += len(cohort)
                raise ContinuousDuplicateRequestError(
                    "continuous request IDs are unique for one service lifetime"
                )
            if len(self._seen_request_ids) + len(cohort) > self._max_request_history:
                self._rejected_admission += len(cohort)
                self._rejected_backpressure += len(cohort)
                self._rejected_request_history += len(cohort)
                raise ContinuousBackpressureError(
                    "continuous request-history capacity is exhausted; create a new service "
                    "incarnation before admitting more globally unique request IDs"
                )
            if len(self._active) + len(cohort) > self._max_active:
                self._rejected_admission += len(cohort)
                self._rejected_backpressure += len(cohort)
                raise ContinuousBackpressureError(
                    f"continuous service would have {len(self._active) + len(cohort)} active "
                    f"requests (limit {self._max_active})"
                )

            validated: list[
                tuple[
                    ContinuousPagedRequest,
                    pf.BatchedPagedKVCache | None,
                    pf.PagedKVSlotLease | None,
                    int,
                    int,
                ]
            ] = []
            try:
                active_cache_ids = {id(session.cache) for session in self._active.values()}
                active_lease_ids = {session.lease.lease_id for session in self._active.values()}
                supplied_caches = tuple(
                    request.cache for request in cohort if request.cache is not None
                )
                supplied_leases = tuple(
                    request.slot_lease for request in cohort if request.slot_lease is not None
                )
                if len({id(cache) for cache in supplied_caches}) != len(supplied_caches) or any(
                    id(cache) in active_cache_ids for cache in supplied_caches
                ):
                    raise ContinuousAdmissionError(
                        "one paged KV cache cannot back multiple continuous requests"
                    )
                if len({lease.lease_id for lease in supplied_leases}) != len(
                    supplied_leases
                ) or any(lease.lease_id in active_lease_ids for lease in supplied_leases):
                    raise ContinuousAdmissionError(
                        "one paged KV slot lease cannot back multiple continuous requests"
                    )
                for request in cohort:
                    validated.append(self._validate_request_for_admission(request))
                cohort_bytes = sum(item[4] for item in validated)
                if (
                    self._max_committed_kv_arena_bytes is not None
                    and self._active_committed_kv_arena_bytes + cohort_bytes
                    > self._max_committed_kv_arena_bytes
                ):
                    raise ContinuousBackpressureError(
                        "continuous service would own "
                        f"{self._active_committed_kv_arena_bytes + cohort_bytes} committed KV "
                        "arena bytes "
                        f"(limit {self._max_committed_kv_arena_bytes}); transient attention "
                        "and provisional-delta scratch is separately instrumented, not admitted "
                        "by this arena budget"
                    )
            except BaseException as exc:
                self._rejected_admission += len(cohort)
                if isinstance(exc, ContinuousBackpressureError):
                    self._rejected_backpressure += len(cohort)
                raise

            created: list[tuple[pf.BatchedPagedKVCache, pf.PagedKVSlotLease]] = []
            sessions: list[_Session] = []
            now = self._clock()
            try:
                for request, cache, lease, capacity, expected_bytes in validated:
                    service_allocated = cache is None
                    if cache is None:
                        cache = self._allocate_cache(capacity)
                        lease = cache.mint_slot_lease()
                        created.append((cache, lease))
                    assert lease is not None
                    actual_bytes = self._cache_bytes(cache)
                    if actual_bytes != expected_bytes:
                        raise ContinuousAdmissionError(
                            "paged KV allocation bytes changed after admission planning"
                        )
                    future = _RequestFuture(self, request.request_id)
                    kv_handle = f"kv:continuous:{self._route.route_id[:12]}:{request.request_id}"
                    observation = bind_versioned_kv_state(cache, (kv_handle,))
                    starting_length = int(observation.lengths[0])
                    required = starting_length + len(request.input_ids) + request.max_new_tokens - 1
                    if required > cache.capacity:
                        raise ContinuousAdmissionError(
                            "paged KV state changed during admission and no longer has capacity"
                        )
                    sessions.append(
                        _Session(
                            request=request,
                            future=future,
                            cache=cache,
                            lease=lease,
                            kv_handle=kv_handle,
                            kv_bytes=actual_bytes,
                            service_allocated_cache=service_allocated,
                            arrival_at=now,
                            starting_length=starting_length,
                            current_ids=request.input_ids,
                            expected_epoch=observation.epoch,
                            expected_length=starting_length,
                            expected_storage_signature=observation.storage_signature,
                        )
                    )
            except BaseException:
                for cache, lease in created:
                    try:
                        cache.release_slot_lease(lease)
                    except Exception:
                        pass
                self._rejected_admission += len(cohort)
                raise

            for session in sessions:
                request_id = session.request.request_id
                self._active[request_id] = session
                self._seen_request_ids.add(request_id)
                self._active_committed_kv_arena_bytes += session.kv_bytes
            self._peak_active_committed_kv_arena_bytes = max(
                self._peak_active_committed_kv_arena_bytes,
                self._active_committed_kv_arena_bytes,
            )
            self._admitted += len(sessions)
            self._condition.notify_all()
            return tuple(session.future for session in sessions)

    def telemetry(self) -> ContinuousServiceTelemetry:
        with self._condition:
            snapshot = ContinuousServiceTelemetry(
                admitted=self._admitted,
                active=len(self._active),
                completed=self._completed,
                cancelled=self._cancelled,
                deadline_expired=self._deadline_expired,
                failed=self._failed,
                shutdown_terminated=self._shutdown_terminated,
                rejected_admission=self._rejected_admission,
                rejected_backpressure=self._rejected_backpressure,
                rejected_duplicate=self._rejected_duplicate,
                rejected_request_history=self._rejected_request_history,
                steps_submitted=self._steps_submitted,
                steps_succeeded=self._steps_succeeded,
                steps_failed=self._steps_failed,
                steps_cancelled=self._steps_cancelled,
                steps_abandoned=self._steps_abandoned,
                steps_active=sum(not session.step_accounted for session in self._active.values()),
                generated_tokens=self._generated_tokens,
                refills=self._refills,
                leases_released=self._leases_released,
                aggregate_template_cache_hits=self._aggregate_template_cache_hits,
                aggregate_template_cache_misses=self._aggregate_template_cache_misses,
                template_cache_size=len(self._template_cache),
                template_cache_capacity=self._max_template_cache,
                template_cache_evictions=self._template_cache_evictions,
                request_history_size=len(self._seen_request_ids),
                request_history_capacity=self._max_request_history,
                pooled_attention_global_prefix_kv_logical_bytes_max=(
                    self._pooled_attention_global_prefix_kv_logical_bytes_max
                ),
                pooled_attention_request_local_prefix_kv_logical_bytes_max=(
                    self._pooled_attention_request_local_prefix_kv_logical_bytes_max
                ),
                pooled_attention_explicit_live_prefix_kv_peak_bytes_max=(
                    self._pooled_attention_explicit_live_prefix_kv_peak_bytes_max
                ),
                pooled_aggregate_provisional_delta_bytes_max=(
                    self._pooled_aggregate_provisional_delta_bytes_max
                ),
                active_committed_kv_arena_bytes=(self._active_committed_kv_arena_bytes),
                peak_active_committed_kv_arena_bytes=(self._peak_active_committed_kv_arena_bytes),
                max_committed_kv_arena_bytes=self._max_committed_kv_arena_bytes,
                ttft=self._ttft.snapshot(),
                inter_token=self._inter_token.snapshot(),
                request_latency=self._request_latency.snapshot(),
                step_latency=self._step_latency.snapshot(),
                accepting=self._accepting,
                stopped=self._stopped.is_set(),
                route_id=self._route.route_id,
                route_promotion_status=self._route.promotion_status,
                pooled_arithmetic=self._pooled_arithmetic,
                reactor=self._reactor.telemetry(),
            )
            if not snapshot.reconciled:
                raise AssertionError("continuous service request telemetry does not reconcile")
            return snapshot

    def wait_idle(self, timeout: float | None = None) -> bool:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be finite and non-negative")
        end = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._active:
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_pending: bool = False,
        timeout: float | None = None,
    ) -> bool:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be finite and non-negative")
        with self._condition:
            self._accepting = False
            requested = "abort" if cancel_pending else "drain"
            if self._shutdown_mode is None or requested == "abort":
                self._shutdown_mode = requested
            if self._shutdown_mode == "abort":
                for session in tuple(self._active.values()):
                    self._request_shutdown_cancel_locked(session)
            draining = self._shutdown_mode == "drain"
            self._condition.notify_all()
        if draining:
            # A PREFER_BATCH step may already be sleeping inside the reactor.  Flush only the
            # queue that exists now; later drain refills carry their own bypass marker.
            self._reactor.flush_pending()
        if wait and threading.current_thread() is not self._coordinator:
            self._coordinator.join(timeout)
        return self._stopped.is_set()

    def close(self) -> None:
        self.shutdown(wait=True)

    def __enter__(self) -> ContinuousPagedService:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.shutdown(wait=True, cancel_pending=exc is not None)

    def _cancel_public_future(self, future: _RequestFuture) -> bool:
        with self._condition:
            session = self._active.get(future._request_id)  # noqa: SLF001
            if session is None:
                return future.cancelled()
            if session.future is not future or session.phase == "committing":
                return False
            if session.cancel_requested:
                # Cancellation is a first-winner terminal intent.  Preserve whether the user
                # or abort shutdown won while retaining Future.cancel() idempotence.
                return future.cancelled()
            cancelled = future._cancel_under_owner()
            if not cancelled:
                return False
            session.cancel_requested = True
            session.cancel_reason = "cancelled"
            self._cancel_session_work_locked(session)
            self._condition.notify_all()
            return True

    def _request_shutdown_cancel_locked(self, session: _Session) -> None:
        if session.phase == "committing" or session.cancel_requested:
            return
        session.cancel_requested = True
        session.cancel_reason = "shutdown"
        session.future._cancel_under_owner()
        self._cancel_session_work_locked(session)

    def _cancel_session_work_locked(self, session: _Session) -> None:
        step = session.step_future
        if step is None:
            self._terminate_locked(session, session.cancel_reason)
            return
        if step.cancel():
            self._account_step_locked(session, "cancelled")
            self._terminate_locked(session, session.cancel_reason)

    def _account_step_locked(
        self,
        session: _Session,
        outcome: Literal["succeeded", "failed", "cancelled", "abandoned"],
    ) -> None:
        if session.step_accounted:
            raise AssertionError("continuous reactor step was accounted more than once")
        if outcome == "succeeded":
            self._steps_succeeded += 1
        elif outcome == "failed":
            self._steps_failed += 1
        elif outcome == "cancelled":
            self._steps_cancelled += 1
        else:
            self._steps_abandoned += 1
        session.step_accounted = True

    def _validate_request_for_admission(
        self,
        request: ContinuousPagedRequest,
    ) -> tuple[
        ContinuousPagedRequest,
        pf.BatchedPagedKVCache | None,
        pf.PagedKVSlotLease | None,
        int,
        int,
    ]:
        if request.engine is not self._engine:
            raise ContinuousAdmissionError("continuous request belongs to a foreign engine")
        semantic_tokens = int(self._engine.semantic_token_count)
        if min(request.input_ids) < 0 or max(request.input_ids) >= semantic_tokens:
            raise ContinuousAdmissionError(
                f"continuous request token IDs must be inside [0, {semantic_tokens})"
            )
        if request.eos_token_id is not None and not 0 <= request.eos_token_id < semantic_tokens:
            raise ContinuousAdmissionError(f"eos_token_id must be inside [0, {semantic_tokens})")
        cache = request.cache
        lease = request.slot_lease
        if cache is not None:
            self._validate_supplied_cache(cache, cast(pf.PagedKVSlotLease, lease))
            starting_length = int(cache.lengths[0])
            capacity = int(cache.capacity)
            if request.kv_capacity is not None and request.kv_capacity != capacity:
                raise ContinuousAdmissionError(
                    "request kv_capacity does not match its supplied cache"
                )
        else:
            starting_length = 0
            minimum = len(request.input_ids) + request.max_new_tokens - 1
            capacity = request.kv_capacity or max(1, minimum)
        required = starting_length + len(request.input_ids) + request.max_new_tokens - 1
        if required > capacity:
            raise ContinuousAdmissionError(
                f"continuous request needs KV capacity {required}, but owns {capacity}"
            )
        configured_context = int(
            self._engine.cfg.get(
                "max_position_embeddings",
                self._engine.cfg.get("max_seq_len", capacity),
            )
        )
        if capacity > configured_context:
            raise ContinuousAdmissionError(
                f"KV capacity {capacity} exceeds configured context {configured_context}"
            )
        expected_bytes = self._planned_cache_bytes(capacity)
        if cache is not None and self._cache_bytes(cache) != expected_bytes:
            raise ContinuousAdmissionError("supplied cache byte size does not match model anatomy")
        return request, cache, lease, capacity, expected_bytes

    def _validate_supplied_cache(
        self,
        cache: pf.BatchedPagedKVCache,
        lease: pf.PagedKVSlotLease,
    ) -> None:
        if not isinstance(cache, pf.BatchedPagedKVCache) or cache.B != 1:
            raise ContinuousAdmissionError("continuous requests require one B=1 paged KV cache")
        expected = (
            int(self._engine.cfg["num_hidden_layers"]),
            1,
            cache.capacity,
            int(self._engine.cfg["num_key_value_heads"]),
            int(self._engine.cfg["head_dim"]),
        )
        if tuple(cache.k.shape) != expected or tuple(cache.v.shape) != expected:
            raise ContinuousAdmissionError(f"supplied paged KV cache shape must be {expected}")
        if cache.k.dtype != torch.float32 or cache.v.dtype != torch.float32:
            raise ContinuousAdmissionError("supplied paged KV cache must use fp32")
        engine_device = torch.device(self._engine.device)
        cache_device = cache.k.device
        same_engine_device = (
            cache_device.type == "cpu" and engine_device.type == "cpu"
            if self._route.fabric == "cpu"
            else cache_device == engine_device
        )
        if cache.v.device != cache_device or not same_engine_device:
            raise ContinuousAdmissionError("supplied cache and engine must share one device")
        cache.validate_slot_lease(lease)

    def _planned_cache_bytes(self, capacity: int) -> int:
        cfg = self._engine.cfg
        return (
            2
            * int(cfg["num_hidden_layers"])
            * int(capacity)
            * int(cfg["num_key_value_heads"])
            * int(cfg["head_dim"])
            * torch.empty((), dtype=torch.float32).element_size()
        )

    @staticmethod
    def _cache_bytes(cache: pf.BatchedPagedKVCache) -> int:
        return cache.k.numel() * cache.k.element_size() + cache.v.numel() * cache.v.element_size()

    def _allocate_cache(self, capacity: int) -> pf.BatchedPagedKVCache:
        cfg = self._engine.cfg
        return pf.BatchedPagedKVCache(
            int(cfg["num_hidden_layers"]),
            1,
            int(cfg["num_key_value_heads"]),
            int(cfg["head_dim"]),
            capacity=capacity,
            device=self._engine.device,
        )

    def _coordinator_main(self) -> None:
        try:
            while True:
                with self._condition:
                    while (
                        not self._events
                        and not any(session.phase == "ready" for session in self._active.values())
                        and not (self._shutdown_mode is not None and not self._active)
                    ):
                        self._condition.wait()

                    events = tuple(self._events)
                    self._events.clear()
                    for event in events:
                        self._process_step_done_locked(event)

                    if self._shutdown_mode == "abort":
                        for session in tuple(self._active.values()):
                            self._request_shutdown_cancel_locked(session)

                    self._dispatch_ready_locked()
                    if self._shutdown_mode is not None and not self._active:
                        break
        except BaseException as exc:
            fatal_error = exc
        else:
            fatal_error = None
        finally:
            self._reactor.shutdown(
                wait=True,
                cancel_pending=self._shutdown_mode == "abort" or fatal_error is not None,
            )
            with self._condition:
                if fatal_error is not None:
                    # The reactor is now quiescent, so no executing child can still carry a
                    # request's lease or read its cache while terminal cleanup releases it.
                    for session in tuple(self._active.values()):
                        if not session.step_accounted:
                            future = session.step_future
                            outcome = (
                                "cancelled"
                                if future is not None and future.cancelled()
                                else "abandoned"
                                if session.cancel_requested
                                else "failed"
                            )
                            self._account_step_locked(session, outcome)
                        if session.cancel_requested:
                            self._terminate_locked(session, session.cancel_reason)
                        else:
                            self._terminate_locked(session, "failed", fatal_error)
                self._accepting = False
                self._stopped.set()
                self._condition.notify_all()

    def _dispatch_ready_locked(self) -> None:
        ready = tuple(session for session in self._active.values() if session.phase == "ready")
        if not ready:
            return
        now = self._clock()
        prepared: list[tuple[_Session, ComponentSubmission[PagedReactorPayload]]] = []
        for session in ready:
            if session.cancel_requested or session.future.cancelled():
                self._terminate_locked(session, session.cancel_reason)
                continue
            if session.request.deadline is not None and session.request.deadline <= now:
                self._terminate_locked(
                    session,
                    "deadline",
                    ContinuousRequestDeadlineExceeded(session.request.request_id),
                )
                continue
            try:
                prepared.append((session, self._prepare_step_submission(session)))
            except BaseException as exc:
                self._terminate_locked(session, "failed", exc)
        if not prepared:
            return

        groups: dict[Any, list[int]] = {}
        for index, (_session, submission) in enumerate(prepared):
            key = template_compatibility_key(
                submission.template,
                submission.lowered_template,
            )
            groups.setdefault(key, []).append(index)
        active_count = len(self._active)
        for indices in groups.values():
            final_index = indices[-1]
            session, submission = prepared[final_index]
            should_flush = self._shutdown_mode == "drain" or len(indices) >= 2
            if len(indices) == 1 and not should_flush:
                policy = cast(ContinuousBatchingPolicy, session.request.batching_policy)
                should_flush = policy is ContinuousBatchingPolicy.LATENCY or (
                    policy is ContinuousBatchingPolicy.ADAPTIVE and active_count == 1
                )
            if should_flush:
                prepared[final_index] = (
                    session,
                    replace(submission, bypass_batch_delay=True),
                )

        try:
            futures = self._reactor.submit_many(
                tuple(submission for _session, submission in prepared)
            )
        except BaseException as exc:
            for session, _submission in prepared:
                self._terminate_locked(session, "failed", exc)
            return

        submitted_at = self._clock()
        for (session, _submission), future in zip(prepared, futures, strict=True):
            session.phase = "inflight"
            session.step_sequence += 1
            session.step_future = future
            session.step_accounted = False
            session.step_started_at = submitted_at
            if session.generated:
                self._refills += 1
            self._steps_submitted += 1
            request_id = session.request.request_id
            step_sequence = session.step_sequence
            future.add_done_callback(
                lambda completed, rid=request_id, seq=step_sequence: self._enqueue_step_done(
                    rid,
                    seq,
                    completed,
                )
            )

    def _prepare_step_submission(
        self,
        session: _Session,
    ) -> ComponentSubmission[PagedReactorPayload]:
        self._route.assert_current(self._engine, self._pooled_arithmetic)
        ids = np.asarray(session.current_ids, dtype=np.int64)
        session.cache.validate_slot_lease(session.lease)
        state_binding = bind_versioned_kv_state(session.cache, (session.kv_handle,))
        if (
            state_binding.epoch != session.expected_epoch
            or state_binding.lengths != (session.expected_length,)
            or state_binding.storage_signature != session.expected_storage_signature
        ):
            raise RuntimeError("continuous request KV state changed outside the service")
        mode = ExecutionMode.PREFILL if int(session.cache.lengths[0]) == 0 else ExecutionMode.DECODE
        plan = build_paged_qstore_plan(
            self._engine,
            [ids],
            execution_mode=mode,
            output_contract=OutputContract.LAST_TOKEN_LOGITS,
            request_ids=(session.request.request_id,),
            request_slots=(0,),
            kv_read_handles=(session.kv_handle,),
            kv_write_handles=(session.kv_handle,),
            kv_capacity=session.cache.capacity,
        )
        template, binding = decompose_work_plan(plan)
        artifact = self._template_cache.get(template.fingerprint)
        if artifact is None:
            lowered = lower_work_template(template, "paged-qstore")
            self._template_cache[template.fingerprint] = (template, lowered)
            self._template_cache.move_to_end(template.fingerprint)
            while len(self._template_cache) > self._max_template_cache:
                self._template_cache.popitem(last=False)
                self._template_cache_evictions += 1
        else:
            cached_template, lowered = artifact
            if cached_template != template:
                raise RuntimeError("continuous template fingerprint collision")
            self._template_cache.move_to_end(template.fingerprint)
        if state_binding.handles != plan.kv_read_handles:
            raise RuntimeError("continuous state observation handle drifted")
        payload = PagedReactorPayload(
            engine=self._engine,
            ids=ids,
            state_binding=state_binding,
            slot_lease=session.lease,
        )
        return ComponentSubmission(
            template=template,
            binding=binding,
            payload=payload,
            lowered_template=lowered,
            deadline=session.request.deadline,
        )

    def _enqueue_step_done(
        self,
        request_id: str,
        step_sequence: int,
        future: Future[PagedReactorResult],
    ) -> None:
        with self._condition:
            self._events.append(_StepDone(request_id, step_sequence, future))
            self._condition.notify_all()

    def _process_step_done_locked(self, event: _StepDone) -> None:
        session = self._active.get(event.request_id)
        if (
            session is None
            or session.step_sequence != event.step_sequence
            or session.step_future is not event.future
        ):
            return
        session.step_future = None
        now = self._clock()
        if session.step_started_at is not None:
            self._step_latency.add(max(0.0, now - session.step_started_at))
        session.step_started_at = None
        if session.cancel_requested or session.future.cancelled():
            self._account_step_locked(
                session,
                "cancelled" if event.future.cancelled() else "abandoned",
            )
            self._terminate_locked(session, session.cancel_reason)
            return
        try:
            result = event.future.result()
        except CancelledError:
            self._account_step_locked(session, "cancelled")
            self._terminate_locked(session, session.cancel_reason)
            return
        except ReactorDeadlineExceeded:
            self._account_step_locked(session, "failed")
            self._terminate_locked(
                session,
                "deadline",
                ContinuousRequestDeadlineExceeded(session.request.request_id),
            )
            return
        except BaseException as exc:
            self._account_step_locked(session, "failed")
            self._terminate_locked(session, "failed", exc)
            return

        if session.request.deadline is not None and session.request.deadline <= now:
            self._account_step_locked(session, "abandoned")
            self._terminate_locked(
                session,
                "deadline",
                ContinuousRequestDeadlineExceeded(session.request.request_id),
            )
            return
        try:
            token_id = self._validate_and_select_token(result)
            commit_started = self._clock()
            if session.request.deadline is not None and session.request.deadline <= commit_started:
                self._account_step_locked(session, "abandoned")
                self._terminate_locked(
                    session,
                    "deadline",
                    ContinuousRequestDeadlineExceeded(session.request.request_id),
                )
                return
            session.phase = "committing"
            accepted = len(session.current_ids)
            committed, emitted_at = self._commit_before_deadline(
                session,
                result,
                accepted,
            )
            if committed != (accepted,):
                raise RuntimeError("continuous step committed the wrong token count")
            observation = bind_versioned_kv_state(session.cache, (session.kv_handle,))
            if observation.epoch != session.expected_epoch + 1 or observation.lengths != (
                session.expected_length + accepted,
            ):
                raise RuntimeError("continuous commit produced an unexpected KV state version")
            session.expected_epoch = observation.epoch
            session.expected_length = int(observation.lengths[0])
            session.expected_storage_signature = observation.storage_signature
        except _CommitDeadlineExpired:
            self._account_step_locked(session, "abandoned")
            self._terminate_locked(
                session,
                "deadline",
                ContinuousRequestDeadlineExceeded(session.request.request_id),
            )
            return
        except BaseException as exc:
            self._account_step_locked(session, "failed")
            self._terminate_locked(session, "failed", exc)
            return

        session.generated.append(token_id)
        session.committed_input_count += accepted
        self._generated_tokens += 1
        self._account_step_locked(session, "succeeded")
        if session.last_token_at is None:
            ttft = max(0.0, emitted_at - session.arrival_at)
            session.first_token_latency = ttft
            self._ttft.add(ttft)
        else:
            interval = max(0.0, emitted_at - session.last_token_at)
            session.inter_token_seconds.append(interval)
            self._inter_token.add(interval)
        session.last_token_at = emitted_at
        evidence = result.evidence
        session.step_records.append(
            ContinuousStepRecord(
                step_index=session.step_sequence,
                reactor_batch_id=int(evidence["component_reactor_batch_id"]),
                reactor_dispatch_id=int(evidence["component_reactor_dispatch_id"]),
                wave_width=int(evidence["component_reactor_wave_width"]),
                input_count=accepted,
                token_id=token_id,
                committed_cache_length=session.expected_length,
                refill=session.step_sequence > 1,
            )
        )

        completed_for_limit = len(session.generated) >= session.request.max_new_tokens
        completed_for_eos = (
            session.request.eos_token_id is not None and token_id == session.request.eos_token_id
        )
        if completed_for_limit or completed_for_eos:
            reason: Literal["max_new_tokens", "eos_token"] = (
                "eos_token" if completed_for_eos else "max_new_tokens"
            )
            if session.first_token_latency is None:
                raise AssertionError("completed continuous request has no first-token latency")
            request_latency = max(0.0, emitted_at - session.arrival_at)
            result_value = ContinuousPagedResult(
                request_id=session.request.request_id,
                generated_token_ids=tuple(session.generated),
                completion_reason=reason,
                committed_input_count=session.committed_input_count,
                starting_cache_length=session.starting_length,
                final_cache_length=session.expected_length,
                final_cache_epoch=session.expected_epoch,
                pending_token_id=token_id,
                step_count=session.step_sequence,
                ttft_seconds=cast(float, session.first_token_latency),
                inter_token_seconds=tuple(session.inter_token_seconds),
                request_latency_seconds=request_latency,
                route_id=self._route.route_id,
                pooled_arithmetic=self._pooled_arithmetic,
                step_records=tuple(session.step_records),
            )
            self._terminate_locked(session, "completed", result=result_value)
            return
        session.current_ids = (token_id,)
        session.phase = "ready"

    def _commit_before_deadline(
        self,
        session: _Session,
        result: PagedReactorResult,
        accepted: int,
    ) -> tuple[tuple[int, ...], float]:
        """Commit and choose the token-publication time as one guarded transaction.

        A request without a deadline takes the ordinary commit path without backup copies.
        With a deadline, the exact target tail is backed up while holding the cache lock.  If
        the first post-commit clock sample is at or beyond the deadline, committed bytes,
        lengths, and epoch are restored before the lock is released.  Compliant readers can
        therefore observe either the old state or the accepted commit, never a late commit.
        """

        deadline = session.request.deadline
        if deadline is None:
            return result.commit(accepted), self._clock()

        cache = session.cache
        start = session.expected_length
        stop = start + accepted
        with cache._lock:  # noqa: SLF001 - cache lock is the transaction boundary
            previous_epoch = int(cache.epoch)
            previous_lengths = cache.lengths.copy()
            previous_k = cache.k[:, 0, start:stop].clone()
            previous_v = cache.v[:, 0, start:stop].clone()
            try:
                committed = result.commit(accepted)
                emitted_at = self._clock()
                if deadline <= emitted_at:
                    raise _CommitDeadlineExpired
            except BaseException:
                try:
                    with torch.no_grad():
                        cache.k[:, 0, start:stop].copy_(previous_k)
                        cache.v[:, 0, start:stop].copy_(previous_v)
                    cache.lengths = previous_lengths
                    cache.epoch = previous_epoch
                except BaseException as rollback_error:
                    cache._poisoned_reason = (  # noqa: SLF001
                        "continuous deadline commit rollback failed"
                    )
                    raise RuntimeError(
                        "continuous deadline commit rollback failed; cache is poisoned"
                    ) from rollback_error
                raise
        return committed, emitted_at

    def _validate_and_select_token(self, result: PagedReactorResult) -> int:
        if not isinstance(result, PagedReactorResult):
            raise ContinuousOutputError("continuous reactor returned the wrong result type")
        if result.output_contract != OutputContract.LAST_TOKEN_LOGITS.value:
            raise ContinuousOutputError("continuous reactor returned the wrong output contract")
        evidence = result.evidence
        if (
            evidence.get("pooled_arithmetic") != self._pooled_arithmetic
            or evidence.get("aggregate_pooled_arithmetic") != self._pooled_arithmetic
        ):
            raise ContinuousOutputError("continuous reactor arithmetic evidence drifted")
        if evidence.get("component_reactor_one_pooled_traversal") is not True:
            raise ContinuousOutputError("continuous reactor lacks one-traversal evidence")
        global_prefix = evidence.get("pooled_attention_global_prefix_kv_logical_bytes")
        local_prefix = evidence.get("pooled_attention_request_local_prefix_kv_logical_bytes_max")
        explicit_live_prefix_peak = evidence.get(
            "pooled_attention_explicit_live_prefix_kv_peak_bytes"
        )
        provisional = evidence.get("pooled_aggregate_provisional_delta_bytes")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for value in (
                global_prefix,
                local_prefix,
                explicit_live_prefix_peak,
                provisional,
            )
        ):
            raise ContinuousOutputError("continuous reactor scratch evidence is malformed")
        if self._pooled_arithmetic == "row_stable_split" and int(global_prefix) != 0:
            raise ContinuousOutputError(
                "row_stable_split unexpectedly allocated global prefix K/V scratch"
            )
        if self._pooled_arithmetic == "row_stable_split":
            raw_parent_lengths = evidence.get("pooled_attention_parent_lengths")
            raw_token_count = evidence.get("pooled_attention_token_count")
            raw_wave_width = evidence.get("component_reactor_wave_width")
            if (
                not isinstance(raw_parent_lengths, (list, tuple))
                or not raw_parent_lengths
                or any(
                    isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
                    for value in raw_parent_lengths
                )
                or isinstance(raw_token_count, bool)
                or not isinstance(raw_token_count, Integral)
                or int(raw_token_count) <= 0
                or isinstance(raw_wave_width, bool)
                or not isinstance(raw_wave_width, Integral)
                or int(raw_wave_width) != len(raw_parent_lengths)
            ):
                raise ContinuousOutputError(
                    "row_stable_split prefix-shape scratch evidence is malformed"
                )
            parent_lengths = tuple(int(value) for value in raw_parent_lengths)
            token_count = int(raw_token_count)
            wave_width = int(raw_wave_width)
            expected_local_logical = (
                2
                * (max(parent_lengths) + token_count)
                * int(self._engine.cfg["num_key_value_heads"])
                * int(self._engine.cfg["head_dim"])
                * torch.empty((), dtype=torch.float32).element_size()
            )
            expected_live_peak = (
                4
                * (max(parent_lengths) + token_count)
                * int(self._engine.cfg["num_attention_heads"])
                * int(self._engine.cfg["head_dim"])
                * torch.empty((), dtype=torch.float32).element_size()
            )
            expected_provisional = (
                2
                * int(self._engine.cfg["num_hidden_layers"])
                * wave_width
                * token_count
                * int(self._engine.cfg["num_key_value_heads"])
                * int(self._engine.cfg["head_dim"])
                * torch.empty((), dtype=torch.float32).element_size()
            )
            if int(local_prefix) != expected_local_logical:
                raise ContinuousOutputError(
                    "row_stable_split request-local logical K/V evidence drifted"
                )
            if int(explicit_live_prefix_peak) != expected_live_peak:
                raise ContinuousOutputError(
                    "row_stable_split explicit live prefix K/V peak evidence drifted: "
                    f"observed {int(explicit_live_prefix_peak)}, expected {expected_live_peak}"
                )
            if int(provisional) != expected_provisional:
                raise ContinuousOutputError(
                    "row_stable_split aggregate provisional K/V evidence drifted"
                )
        self._pooled_attention_global_prefix_kv_logical_bytes_max = max(
            self._pooled_attention_global_prefix_kv_logical_bytes_max,
            int(global_prefix),
        )
        self._pooled_attention_request_local_prefix_kv_logical_bytes_max = max(
            self._pooled_attention_request_local_prefix_kv_logical_bytes_max,
            int(local_prefix),
        )
        self._pooled_attention_explicit_live_prefix_kv_peak_bytes_max = max(
            self._pooled_attention_explicit_live_prefix_kv_peak_bytes_max,
            int(explicit_live_prefix_peak),
        )
        self._pooled_aggregate_provisional_delta_bytes_max = max(
            self._pooled_aggregate_provisional_delta_bytes_max,
            int(provisional),
        )
        cache_hit = evidence.get("aggregate_template_cache_hit")
        if not isinstance(cache_hit, bool):
            raise ContinuousOutputError("continuous aggregate-template evidence is malformed")
        if cache_hit:
            self._aggregate_template_cache_hits += 1
        else:
            self._aggregate_template_cache_misses += 1
        logits = result.outputs
        vocab_rows = int(self._engine.cfg["vocab_size"])
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, vocab_rows):
            raise ContinuousOutputError(f"continuous logits must have shape (1, {vocab_rows})")
        if logits.dtype != torch.float32 or logits.device.type != "cpu":
            raise ContinuousOutputError("continuous logits must be CPU fp32")
        semantic = int(self._engine.semantic_token_count)
        if not bool(torch.isfinite(logits).all().item()):
            raise ContinuousOutputError("continuous logits contain non-finite values")
        semantic_logits = logits[0, :semantic]
        token_id = int(torch.argmax(semantic_logits).item())
        if token_id < 0 or token_id >= semantic:
            raise ContinuousOutputError("continuous greedy token escaped the semantic domain")
        return token_id

    def _terminate_locked(
        self,
        session: _Session,
        reason: Literal["completed", "cancelled", "deadline", "failed", "shutdown"],
        exception: BaseException | None = None,
        *,
        result: ContinuousPagedResult | None = None,
    ) -> None:
        request_id = session.request.request_id
        if self._active.get(request_id) is not session:
            return
        if not session.step_accounted:
            raise AssertionError("continuous request terminated with an unaccounted reactor step")
        session.phase = "terminal"
        release_error: BaseException | None = None
        if not session.lease_released:
            try:
                session.cache.release_slot_lease(session.lease)
                session.lease_released = True
                self._leases_released += 1
            except BaseException as exc:
                release_error = exc
        self._active.pop(request_id, None)
        self._active_committed_kv_arena_bytes -= session.kv_bytes
        if self._active_committed_kv_arena_bytes < 0:
            raise AssertionError("continuous committed KV arena byte count became negative")
        request_latency = max(0.0, self._clock() - session.arrival_at)
        self._request_latency.add(request_latency)

        if release_error is not None:
            reason = "failed"
            exception = release_error
            result = None
        if reason == "completed":
            if result is None:
                raise AssertionError("completed continuous request requires a result")
            result = replace(result, request_latency_seconds=request_latency)
            self._completed += 1
            if not session.future.done():
                session.future.set_result(result)
        elif reason == "cancelled":
            self._cancelled += 1
            if not session.future.done():
                session.future._cancel_under_owner()
        elif reason == "shutdown":
            self._shutdown_terminated += 1
            if not session.future.done():
                session.future._cancel_under_owner()
        elif reason == "deadline":
            self._deadline_expired += 1
            if not session.future.done():
                session.future.set_exception(
                    exception or ContinuousRequestDeadlineExceeded(request_id)
                )
        else:
            self._failed += 1
            if not session.future.done():
                session.future.set_exception(
                    exception or ContinuousServiceError("continuous request failed")
                )
        self._condition.notify_all()


__all__ = [
    "ContinuousAdmissionError",
    "ContinuousBackpressureError",
    "ContinuousBatchingPolicy",
    "ContinuousDuplicateRequestError",
    "ContinuousOutputError",
    "ContinuousPagedRequest",
    "ContinuousPagedResult",
    "ContinuousPagedService",
    "ContinuousRequestDeadlineExceeded",
    "ContinuousRouteUnavailable",
    "ContinuousServiceClosed",
    "ContinuousServiceError",
    "ContinuousServiceTelemetry",
    "ContinuousServingCapabilities",
    "ContinuousStepRecord",
    "LatencyDistribution",
    "resolve_continuous_serving_capabilities",
]
