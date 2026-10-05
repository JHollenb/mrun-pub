"""Component-stationary reactor integration for transactional paged inference.

The generic :mod:`mrun.engine.reactor` owns admission and compatible queue formation.  This
module is the paged backend boundary: it turns compatible B=1 stateful dispatches into one
physical ``B x K`` weight traversal while retaining one independently committable KV delta per
request.

The executor deliberately does not call ``PagedEngine.execute_workplan_stateful``.  That entry
point is a correct single-cache adapter and therefore cannot pool independent request arenas.
Instead, every child artifact and runtime binding is revalidated under one engine transaction,
all cache locks are acquired in their immutable canonical order, and exactly one
``paged_forward_block_pooled`` call performs the physical work.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, cast

import numpy as np
import torch

from mrun.compiler.executable import (
    ProvisionalKVDeltaBinding,
    VersionedKVStateBinding,
    WorkPlanExecutionResult,
    _bind_provisional_delta,
    _state_shape,
    _state_storage_signature,
    _validate_inputs,
    _validate_numerical_contract,
    _validate_runtime_component_binding,
    _validate_runtime_configuration,
    _validate_runtime_identity,
    _validate_stateful_outputs,
    candidate_outputs_from_values,
    commit_provisional_kv_delta,
)
from mrun.compiler.ir import (
    DenseWorkPlan,
    DispatchBinding,
    ExecutionMode,
    OutputContract,
    ShapeBucket,
    WorkTemplate,
)
from mrun.compiler.lowering import LoweredWorkPlan, LoweredWorkTemplate, lower_work_template
from mrun.compiler.memory import MemoryPlan, plan_qstore_memory

from .kernels import paged_forward as pf
from .reactor import BatchItemResult, ComponentBatch, ComponentReactor

_POOLED_ARITHMETIC_LAYOUT_PREFIX = "paged-pooled-arithmetic:"


class PagedReactorValidationError(RuntimeError):
    """A pooled dispatch failed a paged-specific fail-closed invariant."""


def _validate_pooled_arithmetic(value: object) -> pf.PagedPooledArithmetic:
    if not isinstance(value, str) or value not in {
        "packed",
        "row_stable",
        "row_stable_split",
        "batch_invariant",
    }:
        raise ValueError(
            "pooled_arithmetic must be 'packed', 'row_stable', 'row_stable_split', or "
            "'batch_invariant'"
        )
    return cast(pf.PagedPooledArithmetic, value)


@dataclass(frozen=True, slots=True)
class PagedReactorPayload:
    """Opaque dynamic payload paired with one B=1 source-template binding.

    ``ids`` is copied into a read-only int64 vector at construction.  Mutable committed state
    remains outside the source template and is named by both a versioned state observation and
    a cache-issued slot capability.  Merely constructing this value does not prove that the
    observation or capability is still current; the executor proves both atomically.
    """

    engine: Any = field(repr=False, compare=False)
    ids: np.ndarray = field(repr=False, compare=False)
    state_binding: VersionedKVStateBinding = field(repr=False, compare=False)
    slot_lease: pf.PagedKVSlotLease

    def __post_init__(self) -> None:
        if self.engine is None:
            raise TypeError("paged reactor payload requires an engine")
        if not isinstance(self.state_binding, VersionedKVStateBinding):
            raise TypeError("paged reactor payload requires a VersionedKVStateBinding")
        if not isinstance(self.slot_lease, pf.PagedKVSlotLease):
            raise TypeError("paged reactor payload requires a cache-issued PagedKVSlotLease")
        cache = self.state_binding.state
        if not isinstance(cache, pf.BatchedPagedKVCache):
            raise TypeError("paged reactor payload state must be a BatchedPagedKVCache")
        if cache.B != 1 or len(self.state_binding.lengths) != 1:
            raise ValueError("paged reactor payload requires an independently owned B=1 cache")
        if self.state_binding.cache_id != self.slot_lease.cache_id:
            raise ValueError("paged reactor payload lease and state binding name different caches")
        raw_ids = np.asarray(self.ids)
        if raw_ids.dtype.kind not in {"i", "u"}:
            raise TypeError("paged reactor token IDs must be integers")
        if raw_ids.ndim != 1 or raw_ids.size <= 0:
            raise ValueError("paged reactor token IDs must be one nonempty row")
        normalized = np.array(raw_ids, dtype=np.int64, copy=True)
        normalized.setflags(write=False)
        object.__setattr__(self, "ids", normalized)


@dataclass(frozen=True, slots=True)
class PagedReactorResult:
    """One scratch-only B=1 result with an explicit independent commit operation."""

    execution: WorkPlanExecutionResult
    plan: DenseWorkPlan = field(repr=False)
    state_binding: VersionedKVStateBinding = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.execution.plan_fingerprint != self.plan.fingerprint:
            raise ValueError("paged reactor result plan identity is inconsistent")
        if self.execution.provisional_delta is None:
            raise ValueError("paged reactor result requires an uncommitted KV delta")
        if self.execution.provisional_delta.state is not self.state_binding.state:
            raise ValueError("paged reactor result delta belongs to a different state")

    @property
    def outputs(self) -> Any:
        return self.execution.outputs

    @property
    def evidence(self) -> dict[str, Any]:
        return self.execution.evidence

    @property
    def provisional_delta(self) -> ProvisionalKVDeltaBinding:
        delta = self.execution.provisional_delta
        assert delta is not None
        return delta

    @property
    def output_contract(self) -> str:
        return self.execution.output_contract

    @property
    def plan_fingerprint(self) -> str:
        return self.execution.plan_fingerprint

    def commit(self, accepted_count: int) -> tuple[int, ...]:
        """Commit exactly this request's accepted provisional prefix.

        Execution never calls this method.  Zero is an explicit transaction that consumes the
        delta and advances the cache epoch; duplicate, stale, foreign, or released-lease commits
        fail through the ordinary WorkPlan/KV transaction validator.
        """

        if isinstance(accepted_count, bool) or not isinstance(accepted_count, Integral):
            raise TypeError("accepted_count must be an integer")
        normalized = int(accepted_count)
        if normalized < 0 or normalized > self.provisional_delta.token_count:
            raise ValueError("accepted_count is outside the provisional token block")
        result = commit_provisional_kv_delta(
            self.state_binding,
            self.provisional_delta,
            (normalized,),
            plan=self.plan,
        )
        return tuple(int(value) for value in result)


@dataclass(frozen=True, slots=True)
class _PreparedChild:
    binding: DispatchBinding
    payload: PagedReactorPayload
    plan: DenseWorkPlan
    lowered: LoweredWorkPlan
    ids: np.ndarray
    cache: pf.BatchedPagedKVCache


@dataclass(frozen=True, slots=True)
class _AggregateArtifact:
    template: WorkTemplate
    lowered_template: LoweredWorkTemplate


class PagedComponentBatchExecutor:
    """Execute one strictly compatible component batch as one paged traversal."""

    def __init__(
        self,
        *,
        max_aggregate_templates: int = 256,
        pooled_arithmetic: pf.PagedPooledArithmetic = "packed",
    ) -> None:
        if (
            isinstance(max_aggregate_templates, bool)
            or not isinstance(max_aggregate_templates, int)
            or max_aggregate_templates <= 0
        ):
            raise ValueError("max_aggregate_templates must be a positive integer")
        self._pooled_arithmetic = _validate_pooled_arithmetic(pooled_arithmetic)
        self._max_aggregate_templates = max_aggregate_templates
        self._aggregate_templates: OrderedDict[str, _AggregateArtifact] = OrderedDict()
        self._aggregate_lock = threading.Lock()

    @property
    def pooled_arithmetic(self) -> pf.PagedPooledArithmetic:
        """Validated numerical lane fixed for this executor's lifetime."""

        return self._pooled_arithmetic

    @property
    def aggregate_template_cache_size(self) -> int:
        with self._aggregate_lock:
            return len(self._aggregate_templates)

    def execute_batch(
        self,
        batch: ComponentBatch[PagedReactorPayload],
    ) -> Sequence[BatchItemResult[PagedReactorResult]]:
        """Revalidate, pool once, and return one independently committable child result."""

        source, lowered_template, prepared = self._prepare_without_store_access(batch)
        engine = prepared[0].payload.engine
        self._validate_single_engine_owner(prepared)
        child_errors = self._cross_child_capability_errors(prepared)

        engine_lock = getattr(engine, "_execution_lock", None)
        if not callable(getattr(engine_lock, "acquire", None)) or not callable(
            getattr(engine_lock, "release", None)
        ):
            raise TypeError("paged reactor engine must expose its execution transaction lock")

        unique_caches = tuple({id(child.cache): child.cache for child in prepared}.values())
        ordered_caches = sorted(
            unique_caches,
            key=lambda cache: (cache._lock_order_key, id(cache)),  # noqa: SLF001
        )
        with engine_lock:
            with ExitStack() as stack:
                for cache in ordered_caches:
                    stack.enter_context(cache._lock)  # noqa: SLF001 - global cache order

                runtime_evidence: list[dict[str, Any] | None] = [None] * len(prepared)
                for index, child in enumerate(prepared):
                    if child_errors[index] is not None:
                        continue
                    try:
                        runtime_evidence[index] = self._validate_child_under_locks(
                            engine,
                            lowered_template,
                            child,
                        )
                    except Exception as exc:  # one stale request must not poison its siblings
                        child_errors[index] = exc

                valid_indices = tuple(
                    index for index, error in enumerate(child_errors) if error is None
                )
                if not valid_indices:
                    return self._failure_outcomes(child_errors)
                valid = tuple(prepared[index] for index in valid_indices)
                caches = tuple(child.cache for child in valid)
                leases = tuple(child.payload.slot_lease for child in valid)
                union = self._stable_output_union(source.output_contract, valid)
                composite = getattr(engine, "composite_store", None)
                if composite is not None and union:
                    composite.vocab.validate_token_ids(union)

                try:
                    aggregate_plan, aggregate_lowered, aggregate_memory, cache_hit = (
                        self._aggregate_wave_artifacts(
                            source,
                            lowered_template,
                            valid,
                            union,
                        )
                    )
                    before = tuple(
                        (_state_shape(cache), _state_storage_signature(cache)) for cache in caches
                    )
                    scratch_telemetry: list[pf.PagedPooledScratchTelemetry] = []
                    try:
                        pooled_output, raw_deltas = pf.paged_forward_block_pooled(
                            engine.store,
                            np.stack([child.ids for child in valid]),
                            caches,
                            leases,
                            output_contract=self._kernel_output_contract(source.output_contract),
                            selected_rows=union,
                            selected_row_groups=(
                                self._selected_row_groups(source.output_contract, valid)
                                if self.pooled_arithmetic in {"row_stable", "row_stable_split"}
                                else ()
                            ),
                            last_only=self._last_only(source.output_contract),
                            arithmetic=self.pooled_arithmetic,
                            scratch_observer=scratch_telemetry.append,
                        )
                    finally:
                        after = tuple(
                            (_state_shape(cache), _state_storage_signature(cache))
                            for cache in caches
                        )
                        if after != before:
                            for cache, old, new in zip(caches, before, after, strict=True):
                                if old != new:
                                    cache._poisoned_reason = (  # noqa: SLF001
                                        "paged component reactor observed an auto-commit or "
                                        "committed-state mutation"
                                    )
                            raise RuntimeError(
                                "paged component reactor traversal mutated committed KV state"
                            )
                    if len(raw_deltas) != len(valid):
                        raise RuntimeError(
                            "pooled paged kernel returned the wrong child-delta count"
                        )
                    if len(scratch_telemetry) != 1:
                        raise RuntimeError(
                            "pooled paged kernel did not emit exactly one scratch telemetry record"
                        )
                    scratch = scratch_telemetry[0]
                    if scratch.arithmetic != self.pooled_arithmetic:
                        raise RuntimeError("pooled paged scratch telemetry named the wrong lane")
                    child_outputs = self._split_outputs(
                        source.output_contract,
                        pooled_output,
                        valid,
                        union,
                    )
                except Exception as exc:
                    for index in valid_indices:
                        child_errors[index] = exc
                    return self._failure_outcomes(child_errors)

                outcomes: list[BatchItemResult[PagedReactorResult] | None] = [None] * len(prepared)
                for index, error in enumerate(child_errors):
                    if error is not None:
                        outcomes[index] = BatchItemResult.failure(error)
                for valid_offset, (child, output, raw_delta) in enumerate(
                    zip(valid, child_outputs, raw_deltas, strict=True)
                ):
                    original_index = valid_indices[valid_offset]
                    try:
                        _validate_stateful_outputs(child.plan, output)
                        provisional = _bind_provisional_delta(
                            child.payload.state_binding,
                            raw_delta,
                            plan=child.plan,
                            outputs=output,
                        )
                        child_runtime = runtime_evidence[original_index]
                        if child_runtime is None:
                            raise AssertionError("valid paged child has no runtime evidence")
                        execution = WorkPlanExecutionResult(
                            plan_fingerprint=child.plan.fingerprint,
                            executable_key=child.lowered.executable_key,
                            output_contract=child.plan.output_contract.value,
                            outputs=output,
                            evidence={
                                "implementation_status": child.lowered.implementation_status,
                                "runtime_implementation_status": "paged-component-reactor",
                                "engine_backend": str(getattr(engine, "backend", "unknown")),
                                **child_runtime,
                                "runtime_identity_bound": True,
                                "runtime_configuration_bound": True,
                                "state_binding_verified": True,
                                "slot_lease_verified": True,
                                "provisional_kv_emitted": True,
                                "work_template_fingerprint": source.fingerprint,
                                "dispatch_binding_fingerprint": child.binding.fingerprint,
                                "dispatch_metadata": dict(child.binding.dispatch_metadata),
                                "lowering_abi": lowered_template.lowering_abi,
                                "lowered_template_artifact_sha256": (
                                    lowered_template.artifact_sha256
                                ),
                                "reused_lowered_template": True,
                                "component_reactor_batch_id": batch.batch_id,
                                "component_reactor_dispatch_id": batch.dispatch_ids[original_index],
                                "component_reactor_wave_width": len(valid),
                                "component_reactor_one_pooled_traversal": True,
                                "pooled_arithmetic": self.pooled_arithmetic,
                                "pooled_attention_global_prefix_kv_logical_bytes": (
                                    scratch.global_prefix_kv_logical_bytes
                                ),
                                "pooled_attention_request_local_prefix_kv_logical_bytes_max": (
                                    scratch.request_local_prefix_kv_logical_bytes_max
                                ),
                                "pooled_attention_explicit_live_prefix_kv_peak_bytes": (
                                    scratch.explicit_live_prefix_kv_peak_bytes
                                ),
                                "pooled_attention_parent_lengths": list(scratch.parent_lengths),
                                "pooled_attention_token_count": scratch.token_count,
                                "pooled_aggregate_provisional_delta_bytes": (
                                    scratch.aggregate_provisional_delta_bytes
                                ),
                                "stable_output_union": list(union),
                                "aggregate_plan_fingerprint": aggregate_plan.fingerprint,
                                "aggregate_template_fingerprint": (
                                    WorkTemplate.from_plan(aggregate_plan).fingerprint
                                ),
                                "aggregate_executable_key": aggregate_lowered.executable_key,
                                "aggregate_template_cache_hit": cache_hit,
                                "aggregate_memory_plan": aggregate_memory.as_dict(),
                                "aggregate_actual_batch": aggregate_plan.shape.actual_batch,
                                "aggregate_output_union_count": len(union),
                                "aggregate_pooled_arithmetic": self.pooled_arithmetic,
                                "actual_batch": 1,
                                "sequence_length": child.plan.shape.sequence_length,
                                "numerical_contract": child.plan.numerical_contract,
                                "kv_parent_epoch": provisional.parent_epoch,
                                "kv_parent_lengths": list(provisional.parent_lengths),
                                "kv_cache_id": provisional.cache_id,
                                "provisional_plan_fingerprint": (provisional.plan_fingerprint),
                            },
                            provisional_delta=provisional,
                        )
                        outcomes[original_index] = BatchItemResult.success(
                            PagedReactorResult(
                                execution=execution,
                                plan=child.plan,
                                state_binding=child.payload.state_binding,
                            )
                        )
                    except Exception as exc:
                        outcomes[original_index] = BatchItemResult.failure(exc)

                if any(outcome is None for outcome in outcomes):
                    raise AssertionError("paged component executor left an outcome unresolved")
                return tuple(outcome for outcome in outcomes if outcome is not None)

    @staticmethod
    def _prepare_without_store_access(
        batch: ComponentBatch[PagedReactorPayload],
    ) -> tuple[WorkTemplate, LoweredWorkTemplate, tuple[_PreparedChild, ...]]:
        """Canonicalize every serialized artifact before touching engine/store state."""

        if not isinstance(batch, ComponentBatch):
            raise TypeError("paged executor requires a ComponentBatch")
        if batch.dispatch_width <= 0:
            raise ValueError("paged executor batch cannot be empty")
        source = WorkTemplate.from_json(batch.template.to_json())
        if source != batch.template or source.fingerprint != batch.template.fingerprint:
            raise PagedReactorValidationError("source WorkTemplate is not canonical")
        if source.shape.actual_batch != 1:
            raise ValueError("paged component pooling currently requires B=1 source templates")
        if source.execution_mode not in {ExecutionMode.PREFILL, ExecutionMode.DECODE}:
            raise ValueError("paged component pooling requires a stateful source template")
        if source.capture.requested:
            raise ValueError("paged component pooling does not admit graph capture")
        if any(
            value.startswith(_POOLED_ARITHMETIC_LAYOUT_PREFIX)
            for value in source.compute_layout_ids
        ):
            raise ValueError("source templates cannot forge the reactor pooled-arithmetic layout")
        if batch.lowered_template is None:
            raise TypeError("paged component pooling requires a lowered WorkTemplate")
        lowered_template = LoweredWorkTemplate.from_json(batch.lowered_template.to_json())
        if lowered_template != batch.lowered_template:
            raise PagedReactorValidationError("lowered WorkTemplate is not canonical")
        canonical_lowering = lower_work_template(source, "paged-qstore")
        if lowered_template != canonical_lowering:
            raise PagedReactorValidationError(
                "lowered schedule is not the canonical paged-qstore lowering"
            )

        prepared: list[_PreparedChild] = []
        for binding, payload in zip(batch.bindings, batch.payloads, strict=True):
            if not isinstance(payload, PagedReactorPayload):
                raise TypeError("paged component batch payload has the wrong type")
            canonical_binding = DispatchBinding.from_json(binding.to_json())
            if canonical_binding != binding:
                raise PagedReactorValidationError("DispatchBinding is not canonical")
            plan = source.bind(canonical_binding)
            lowered = lowered_template.bind(plan)
            if lowered.backend != "paged-qstore":
                raise RuntimeError("paged component executor requires paged-qstore lowering")
            if lowered.implementation_status != "eager-adapter":
                raise RuntimeError("paged component executor requires an eager-adapter schedule")
            if lowered.capture_requested or lowered.capture_ready or lowered.capture_executed:
                raise RuntimeError("paged component executor cannot execute a capture schedule")
            cache = payload.state_binding.state
            if not isinstance(cache, pf.BatchedPagedKVCache) or cache.B != 1:
                raise TypeError("paged component child must own one singleton paged KV cache")
            prepared.append(
                _PreparedChild(
                    binding=canonical_binding,
                    payload=payload,
                    plan=plan,
                    lowered=lowered,
                    ids=payload.ids,
                    cache=cache,
                )
            )
        return source, lowered_template, tuple(prepared)

    @staticmethod
    def _validate_single_engine_owner(prepared: tuple[_PreparedChild, ...]) -> None:
        """One reactor wave belongs to one physical engine/store owner.

        Source templates intentionally exclude engine object identity.  Consequently, callers
        must dedicate a :class:`PagedComponentReactor` to one engine owner; if distinct owners
        enter the same compatible queue, the entire mixed wave fails before either store is
        touched.
        """

        engine = prepared[0].payload.engine
        if str(getattr(engine, "backend", "")) != "paged":
            raise RuntimeError("paged component executor requires a PagedEngine backend")
        if any(child.payload.engine is not engine for child in prepared):
            raise RuntimeError("paged component wave cannot mix engine instances")

    @staticmethod
    def _cross_child_capability_errors(
        prepared: tuple[_PreparedChild, ...],
    ) -> list[BaseException | None]:
        """Isolate colliding request authorities without discarding valid siblings."""

        errors: list[BaseException | None] = [None] * len(prepared)

        def mark_duplicates(values: Sequence[Any], message: str) -> None:
            groups: dict[Any, list[int]] = {}
            for index, value in enumerate(values):
                groups.setdefault(value, []).append(index)
            for indices in groups.values():
                if len(indices) <= 1:
                    continue
                for index in indices:
                    errors[index] = PagedReactorValidationError(message)

        mark_duplicates(
            [id(child.cache) for child in prepared],
            "paged component wave cannot reuse one cache in multiple children",
        )
        mark_duplicates(
            [child.cache.cache_id for child in prepared],
            "paged component wave contains a duplicate cache identity",
        )
        mark_duplicates(
            [child.payload.slot_lease.lease_id for child in prepared],
            "paged component wave contains a duplicate slot lease",
        )
        mark_duplicates(
            [child.binding.request_ids[0] for child in prepared],
            "paged component wave contains a duplicate request ID",
        )
        mark_duplicates(
            [child.binding.kv_read_handles[0] for child in prepared],
            "paged component wave contains a duplicate KV handle",
        )
        return errors

    @staticmethod
    def _failure_outcomes(
        errors: Sequence[BaseException | None],
    ) -> tuple[BatchItemResult[PagedReactorResult], ...]:
        if any(error is None for error in errors):
            raise AssertionError("cannot publish a failure vector with unresolved children")
        return tuple(BatchItemResult.failure(error) for error in errors if error is not None)

    @staticmethod
    def _validate_child_under_locks(
        engine: Any,
        lowered_template: LoweredWorkTemplate,
        child: _PreparedChild,
    ) -> dict[str, Any]:
        # Rebind once more inside the transaction so neither a source nor lowered artifact can
        # bypass the exact concrete-plan relationship used by runtime validation.
        rebound = WorkTemplate.from_json(WorkTemplate.from_plan(child.plan).to_json())
        if rebound.fingerprint != lowered_template.template_fingerprint:
            raise RuntimeError("child plan changed its source-template identity")
        rebound_plan = rebound.bind(DispatchBinding.from_json(child.binding.to_json()))
        rebound_lowered = lowered_template.bind(rebound_plan)
        if rebound_plan.fingerprint != child.plan.fingerprint:
            raise RuntimeError("child WorkPlan reconstruction is not lossless")
        if rebound_lowered.plan_fingerprint != rebound_plan.fingerprint:
            raise RuntimeError("child lowered schedule does not name its concrete WorkPlan")

        _validate_runtime_identity(engine, rebound_plan)
        runtime_configuration = _validate_runtime_configuration(
            engine,
            rebound_plan,
            reported_fabric=rebound_lowered.reported_fabric,
        )
        runtime_component = _validate_runtime_component_binding(engine, rebound_plan)
        checked_rows = _validate_inputs(engine, rebound_plan, (child.ids,))
        if len(checked_rows) != 1 or not np.array_equal(checked_rows[0], child.ids):
            raise RuntimeError("paged child token reconstruction changed its values")
        _validate_numerical_contract(engine, rebound_plan)
        child.payload.state_binding.validate_for_plan(rebound_plan)
        child.cache._validate_slot_lease_unlocked(  # noqa: SLF001
            child.payload.slot_lease
        )
        if child.payload.slot_lease.row != 0:
            raise ValueError("paged singleton slot lease must name row zero")
        if child.payload.state_binding.cache_id != child.cache.cache_id:
            raise RuntimeError("paged state binding cache identity changed")
        return {
            **runtime_configuration,
            **runtime_component,
            "reported_fabric": rebound_lowered.reported_fabric,
            "placement_verified": rebound_lowered.placement_verified,
            "content_identity_verified": rebound_lowered.content_identity_verified,
        }

    @staticmethod
    def _stable_output_union(
        contract: OutputContract,
        prepared: tuple[_PreparedChild, ...],
    ) -> tuple[int, ...]:
        if contract is OutputContract.SELECTED_TOKEN_ROWS:
            values = (token for child in prepared for token in child.binding.required_output_rows)
        elif contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
            values = (
                token
                for child in prepared
                for row in child.binding.candidate_token_ids
                for token in row
            )
        else:
            return ()
        return tuple(dict.fromkeys(int(token) for token in values))

    @staticmethod
    def _selected_row_groups(
        contract: OutputContract,
        prepared: tuple[_PreparedChild, ...],
    ) -> tuple[tuple[int, ...], ...]:
        if contract is OutputContract.SELECTED_TOKEN_ROWS:
            return tuple(child.binding.required_output_rows for child in prepared)
        if contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
            return tuple(child.binding.candidate_token_ids[0] for child in prepared)
        return ()

    @staticmethod
    def _kernel_output_contract(contract: OutputContract) -> pf.PagedBlockOutputContract:
        if contract in {
            OutputContract.SELECTED_TOKEN_ROWS,
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        }:
            return "selected_token_rows"
        if contract is OutputContract.HIDDEN_STATE_ONLY:
            return "hidden_state_only"
        if contract in {OutputContract.FULL_LOGITS, OutputContract.LAST_TOKEN_LOGITS}:
            return "full_logits"
        raise NotImplementedError(
            f"paged reactor output contract {contract.value!r} is unavailable"
        )

    @staticmethod
    def _last_only(contract: OutputContract) -> bool:
        return contract in {
            OutputContract.LAST_TOKEN_LOGITS,
            OutputContract.SELECTED_TOKEN_ROWS,
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        }

    def _aggregate_wave_artifacts(
        self,
        source: WorkTemplate,
        lowered_template: LoweredWorkTemplate,
        prepared: tuple[_PreparedChild, ...],
        union: tuple[int, ...],
    ) -> tuple[DenseWorkPlan, LoweredWorkPlan, MemoryPlan, bool]:
        width = len(prepared)
        metadata = dict(source.metadata)
        if source.output_contract in {
            OutputContract.SELECTED_TOKEN_ROWS,
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        }:
            metadata["logical_head_row_count"] = len(union)
        aggregate_template = WorkTemplate(
            model_name=source.model_name,
            model_revision=source.model_revision,
            store_fingerprint=source.store_fingerprint,
            execution_mode=source.execution_mode,
            precision=source.precision,
            shape=ShapeBucket(
                actual_batch=width,
                # The eager pooled kernel allocates exact B, so a padded power-of-two bucket
                # would make admission describe memory the physical wave never allocates.
                batch_bucket=width,
                sequence_length=source.shape.sequence_length,
                sequence_bucket=source.shape.sequence_bucket,
            ),
            output_contract=source.output_contract,
            numerical_contract=source.numerical_contract,
            prefix_state_count=0,
            kv_read_handle_count=width,
            kv_write_handle_count=width,
            required_output_row_count=(
                len(union) if source.output_contract is OutputContract.SELECTED_TOKEN_ROWS else 0
            ),
            candidate_row_counts=(
                tuple(len(child.binding.candidate_token_ids[0]) for child in prepared)
                if source.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
                else ()
            ),
            candidate_union_count=(
                len(union)
                if source.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
                else 0
            ),
            page_sequence=source.page_sequence,
            cache_admission=source.cache_admission,
            compute_layout_ids=(
                *source.compute_layout_ids,
                f"{_POOLED_ARITHMETIC_LAYOUT_PREFIX}{self.pooled_arithmetic}",
            ),
            structured_operator_ids=source.structured_operator_ids,
            capture=source.capture,
            metadata=tuple(metadata.items()),
        )
        with self._aggregate_lock:
            artifact = self._aggregate_templates.get(aggregate_template.fingerprint)
            cache_hit = artifact is not None
            if artifact is None:
                artifact = _AggregateArtifact(
                    template=aggregate_template,
                    lowered_template=lower_work_template(
                        aggregate_template,
                        lowered_template.backend,
                    ),
                )
                self._aggregate_templates[aggregate_template.fingerprint] = artifact
                self._aggregate_templates.move_to_end(aggregate_template.fingerprint)
                while len(self._aggregate_templates) > self._max_aggregate_templates:
                    self._aggregate_templates.popitem(last=False)
            else:
                self._aggregate_templates.move_to_end(aggregate_template.fingerprint)

        aggregate_binding = DispatchBinding(
            template_fingerprint=artifact.template.fingerprint,
            request_ids=tuple(child.binding.request_ids[0] for child in prepared),
            request_slots=tuple(range(width)),
            kv_read_handles=tuple(child.binding.kv_read_handles[0] for child in prepared),
            kv_write_handles=tuple(child.binding.kv_write_handles[0] for child in prepared),
            required_output_rows=(
                union if source.output_contract is OutputContract.SELECTED_TOKEN_ROWS else ()
            ),
            candidate_token_ids=(
                tuple(child.binding.candidate_token_ids[0] for child in prepared)
                if source.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
                else ()
            ),
        )
        aggregate_plan = artifact.template.bind(aggregate_binding)
        aggregate_lowered = artifact.lowered_template.bind(aggregate_plan)
        manifest = getattr(prepared[0].payload.engine.store, "man", None)
        if not isinstance(manifest, Mapping):
            raise RuntimeError("paged reactor cannot plan aggregate memory without a manifest")
        aggregate_memory = plan_qstore_memory(aggregate_plan, dict(manifest))
        return aggregate_plan, aggregate_lowered, aggregate_memory, cache_hit

    @staticmethod
    def _split_outputs(
        contract: OutputContract,
        pooled: torch.Tensor,
        prepared: tuple[_PreparedChild, ...],
        union: tuple[int, ...],
    ) -> tuple[Any, ...]:
        if not isinstance(pooled, torch.Tensor) or int(pooled.shape[0]) != len(prepared):
            raise RuntimeError("pooled paged output has the wrong batch dimension")
        if contract in {
            OutputContract.FULL_LOGITS,
            OutputContract.LAST_TOKEN_LOGITS,
            OutputContract.HIDDEN_STATE_ONLY,
        }:
            return tuple(pooled[index : index + 1].clone() for index in range(len(prepared)))

        offsets = {token: index for index, token in enumerate(union)}
        if contract is OutputContract.SELECTED_TOKEN_ROWS:
            results: list[torch.Tensor] = []
            for index, child in enumerate(prepared):
                columns = torch.as_tensor(
                    [offsets[token] for token in child.binding.required_output_rows],
                    dtype=torch.long,
                    device=pooled.device,
                )
                results.append(pooled[index : index + 1].index_select(1, columns).clone())
            return tuple(results)

        if contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
            candidate_results: list[tuple[dict[str, Any], ...]] = []
            for index, child in enumerate(prepared):
                candidates = child.binding.candidate_token_ids[0]
                columns = torch.as_tensor(
                    [offsets[token] for token in candidates],
                    dtype=torch.long,
                    device=pooled.device,
                )
                values = pooled[index].index_select(0, columns).clone()
                candidate_results.append(candidate_outputs_from_values((values,), (candidates,)))
            return tuple(candidate_results)
        raise NotImplementedError(
            f"paged reactor output contract {contract.value!r} is unavailable"
        )


class PagedComponentReactor(ComponentReactor[PagedReactorPayload, PagedReactorResult]):
    """Convenience reactor preconfigured with the paged pooled batch executor."""

    def __init__(
        self,
        *,
        max_batch_size: int = 32,
        max_pending: int = 1024,
        max_batch_delay_seconds: float = 0.001,
        telemetry_history: int = 128,
        max_aggregate_templates: int = 256,
        pooled_arithmetic: pf.PagedPooledArithmetic = "packed",
        clock: Callable[[], float] = time.monotonic,
        thread_name: str = "mrun-paged-component-reactor",
    ) -> None:
        executor = PagedComponentBatchExecutor(
            max_aggregate_templates=max_aggregate_templates,
            pooled_arithmetic=pooled_arithmetic,
        )
        self.paged_executor = executor
        super().__init__(
            executor,
            max_batch_size=max_batch_size,
            max_pending=max_pending,
            max_batch_delay_seconds=max_batch_delay_seconds,
            telemetry_history=telemetry_history,
            clock=clock,
            thread_name=thread_name,
        )


__all__ = [
    "PagedComponentBatchExecutor",
    "PagedComponentReactor",
    "PagedReactorPayload",
    "PagedReactorResult",
    "PagedReactorValidationError",
]
