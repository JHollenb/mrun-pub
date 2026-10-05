"""Decode-only compatible batching over fixed dense-CUDA K/V slots.

The production generation coordinator already owns queueing, refill, cancellation, deadlines,
and publication ordering.  This module contributes only the optional physical execution lane:
independent B1 states retain their ordinary authorities, while compatible one-token decodes share
one model traversal and one scratch-only K/V delta.  No committed prefix is copied to form a batch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from .contracts import (
    CompatibleBatchLaneIdentity,
    DecodeWork,
    NativeOutput,
    OutputMode,
    PrefillWork,
    PromotionStatus,
    ProvisionalStep,
    StateObservation,
)
from .dense_cuda import (
    DenseCudaBatchProvisionalAuthority,
    DenseCudaKVSlotCache,
    DenseCudaKVSlotPool,
    DenseCudaNativeRuntime,
    DenseCudaRuntimeError,
    DenseCudaState,
    _next_tokens,
)

CUDA_COMPATIBLE_BATCH_ABI = "cuda-slot-pool-decode-batch-v1"
CUDA_COMPATIBLE_BATCH_NUMERICAL_SUFFIX = "cuda-slot-pool-compatible-batch-v1"


@dataclass(frozen=True, slots=True)
class DenseCudaCompatibleBatchTelemetry:
    lane_id: str
    dispatches: int
    physical_forwards: int
    provisional_rows: int
    committed_rows: int
    abandoned_rows: int
    failed_dispatches: int
    singleton_bypasses: int
    max_width: int
    width_histogram: tuple[tuple[int, int], ...]
    scratch_peak_bytes: int
    active_scratches: int
    pool_slots: int
    pool_active_slots: int


class _DenseCudaBatchScratch:
    """Shared result lifetime with one independently consumable authority per row."""

    __slots__ = (
        "_active_rows",
        "_bindings",
        "_lock",
        "_on_consume",
        "_on_release",
        "_result",
        "scratch_id",
    )

    def __init__(
        self,
        *,
        result: Any,
        states: Sequence[DenseCudaState],
        parents: Sequence[StateObservation],
        on_consume: Callable[[bool, int], None],
        on_release: Callable[[], None],
    ) -> None:
        self.scratch_id = f"cuda-batch-scratch-{uuid4().hex}"
        self._result = result
        self._active_rows = set(range(len(states)))
        self._bindings = tuple(
            (
                state.state_id,
                state._cache.cache_id,  # noqa: SLF001 - runtime lane owns cache authority
                state._cache.slot,  # noqa: SLF001
                state._cache.slot_generation,  # noqa: SLF001
                parent.epoch,
                parent.lengths[0],
            )
            for state, parent in zip(states, parents, strict=True)
        )
        self._on_consume = on_consume
        self._on_release = on_release
        self._lock = threading.Lock()

    @property
    def result(self) -> Any | None:
        with self._lock:
            return self._result

    def install_row(
        self,
        cache: Any,
        *,
        row: int,
        parent: StateObservation,
        accepted_count: int,
    ) -> Any:
        with self._lock:
            if row not in self._active_rows or self._result is None:
                raise DenseCudaRuntimeError("CUDA compatible-batch row is no longer provisional")
            if not isinstance(cache, DenseCudaKVSlotCache):
                raise DenseCudaRuntimeError("CUDA compatible-batch authority lost its slot cache")
            state_id, cache_id, slot, generation, epoch, length = self._bindings[row]
            if (
                parent.state_id != state_id
                or cache.cache_id != cache_id
                or cache.slot != slot
                or cache.slot_generation != generation
                or parent.epoch != epoch
                or parent.lengths != (length,)
            ):
                raise DenseCudaRuntimeError("CUDA compatible-batch row binding drifted")
            delta = getattr(self._result, "delta", None)
            if delta is None:
                raise DenseCudaRuntimeError("CUDA compatible-batch result lost its K/V delta")
            # Do not consume the row until the runtime has checked the post-commit observation.
            # A failed physical copy therefore retains the exact shared scratch for retry.
            return cache.commit_batch_row(
                delta,
                row=row,
                parent_epoch=epoch,
                parent_length=length,
                accepted_count=accepted_count,
            )

    def consume(self, row: int, *, committed: bool, accepted_count: int = 0) -> None:
        released = False
        with self._lock:
            if row not in self._active_rows:
                raise DenseCudaRuntimeError("CUDA compatible-batch row was already consumed")
            self._active_rows.remove(row)
            if not self._active_rows:
                self._result = None
                released = True
        self._on_consume(committed, accepted_count)
        if released:
            self._on_release()

    def discard(self) -> None:
        """Release an unpublished result after step construction or validation failed."""

        with self._lock:
            if self._result is None:
                return
            self._active_rows.clear()
            self._result = None
        self._on_release()


class DenseCudaCompatibleBatchLane:
    """Opt-in ragged-length CUDA decode pooling over independently terminal B1 slots.

    The engine seam is::

        engine.forward_decode_slots(input_ids, indexed_cache)

    ``input_ids`` is int64 ``[B, 1]``. ``indexed_cache`` exposes the pool's full physical K/V
    tensors, logical per-request lengths, and device-local physical row indices.  The result must
    expose ``top1`` ``[B, 1]`` and a scratch-only ``delta`` whose per-layer K/V tensors are
    ``[B, 1, Hkv, D]``. Unit tests may inject the same callable contract without a live CUDA
    device.
    """

    def __init__(
        self,
        runtime: DenseCudaNativeRuntime,
        *,
        max_batch_size: int = 8,
        max_slots: int | None = None,
        max_queue_delay_seconds: float = 0.002,
        max_scratch_bytes: int | None = None,
        batch_executor: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(runtime, DenseCudaNativeRuntime):
            raise TypeError("CUDA compatible batching requires DenseCudaNativeRuntime")
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise TypeError("max_batch_size must be a strict integer")
        if max_batch_size <= 1:
            raise ValueError("CUDA compatible batching requires max_batch_size > 1")
        if max_slots is None:
            max_slots = max_batch_size
        if isinstance(max_slots, bool) or not isinstance(max_slots, int):
            raise TypeError("max_slots must be a strict integer or None")
        if max_slots < max_batch_size:
            raise ValueError("CUDA K/V slot count cannot be smaller than batch width")
        if isinstance(max_queue_delay_seconds, bool) or not isinstance(
            max_queue_delay_seconds, (int, float)
        ):
            raise TypeError("max_queue_delay_seconds must be a finite non-negative number")
        queue_delay = float(max_queue_delay_seconds)
        if not np.isfinite(queue_delay) or queue_delay < 0:
            raise ValueError("max_queue_delay_seconds must be finite and non-negative")
        if max_scratch_bytes is None:
            max_scratch_bytes = max(
                int(runtime._placement.workspace_bytes),  # noqa: SLF001
                int(runtime._placement.state.bytes_per_token) * max_batch_size,  # noqa: SLF001
                1,
            )
        if (
            isinstance(max_scratch_bytes, bool)
            or not isinstance(max_scratch_bytes, int)
            or max_scratch_bytes <= 0
        ):
            raise ValueError("max_scratch_bytes must be a positive integer or None")

        runtime._assert_execution_binding()  # noqa: SLF001 - lane binds exact incarnation
        forward = batch_executor or getattr(runtime._engine, "forward_decode_slots", None)  # noqa: SLF001
        if not callable(forward):
            raise DenseCudaRuntimeError(
                "CUDA compatible batching requires engine.forward_decode_slots"
            )
        if (
            batch_executor is None
            and runtime._head_execution_binding[5]  # noqa: SLF001
            != "segmented-flash-gqa-decode-v1"
        ):
            raise DenseCudaRuntimeError(
                "production CUDA compatible batching requires segmented-flash-gqa-decode-v1"
            )
        self._runtime = runtime
        self._executor = forward
        self._pool = runtime.attach_kv_slot_pool(max_slots=max_slots)
        numerical_contract = str(getattr(runtime._engine, "numerical_contract", ""))  # noqa: SLF001
        self._identity = CompatibleBatchLaneIdentity(
            lane_id=f"cuda-compatible-batch-{uuid4().hex}",
            runtime_id=runtime.route.runtime_id,
            lane_abi=CUDA_COMPATIBLE_BATCH_ABI,
            numerical_contract=(
                f"{numerical_contract}+{CUDA_COMPATIBLE_BATCH_NUMERICAL_SUFFIX}"
            ),
            promotion_status=PromotionStatus.EXPERIMENTAL,
            max_batch_size=max_batch_size,
            max_queue_delay_seconds=queue_delay,
            max_scratch_bytes=max_scratch_bytes,
            supports_ragged_dispatch=True,
        )
        self._lock = threading.Lock()
        self._dispatches = 0
        self._physical_forwards = 0
        self._provisional_rows = 0
        self._committed_rows = 0
        self._abandoned_rows = 0
        self._failed_dispatches = 0
        self._singleton_bypasses = 0
        self._max_width = 0
        self._width_histogram: dict[int, int] = {}
        self._scratch_peak_bytes = 0
        self._active_scratches = 0

    @property
    def identity(self) -> CompatibleBatchLaneIdentity:
        return self._identity

    @property
    def pool(self) -> DenseCudaKVSlotPool:
        return self._pool

    def _record_forward(
        self,
        *,
        width: int,
        scratch_bytes: int = 0,
        singleton: bool = False,
    ) -> None:
        with self._lock:
            self._physical_forwards += 1
            self._provisional_rows += width
            self._max_width = max(self._max_width, width)
            self._width_histogram[width] = self._width_histogram.get(width, 0) + 1
            self._scratch_peak_bytes = max(self._scratch_peak_bytes, scratch_bytes)
            if singleton:
                self._singleton_bypasses += 1

    def _scratch_consumed(self, committed: bool, _accepted_count: int) -> None:
        with self._lock:
            if committed:
                self._committed_rows += 1
            else:
                self._abandoned_rows += 1

    def _scratch_released(self) -> None:
        with self._lock:
            if self._active_scratches <= 0:
                raise DenseCudaRuntimeError("CUDA compatible-batch scratch accounting underflow")
            self._active_scratches -= 1

    @staticmethod
    def _compatible(work: PrefillWork | DecodeWork) -> bool:
        return (
            isinstance(work, DecodeWork)
            and work.parent.batch_size == 1
            and work.parent.lengths[0] > 0
            and len(work.token_rows) == 1
            and len(work.token_rows[0]) == 1
            and work.output.mode is OutputMode.NEXT_TOKEN_ARGMAX
        )

    def _execute_batch(
        self,
        indexed: Sequence[tuple[int, DecodeWork]],
    ) -> tuple[tuple[int, ProvisionalStep], ...]:
        width = len(indexed)
        if width <= 1:
            raise ValueError("a physical CUDA compatible batch requires at least two rows")
        works = tuple(work for _index, work in indexed)
        states = tuple(self._runtime._state(work.state) for work in works)  # noqa: SLF001
        if len({state.state_id for state in states}) != width:
            raise DenseCudaRuntimeError("CUDA compatible batch cannot contain a state twice")
        step_ids = tuple(f"step-{uuid4().hex}" for _ in works)
        scratch: _DenseCudaBatchScratch | None = None
        started = time.perf_counter()
        try:
            with ExitStack() as stack:
                for state in sorted(states, key=lambda value: value.state_id):
                    stack.enter_context(state._lock)  # noqa: SLF001
                self._runtime._assert_execution_binding()  # noqa: SLF001
                parents: list[StateObservation] = []
                caches: list[DenseCudaKVSlotCache] = []
                for state, work, step_id in zip(states, works, step_ids, strict=True):
                    current = state._observe_unlocked()  # noqa: SLF001
                    if current != work.parent:
                        raise DenseCudaRuntimeError(
                            "CUDA compatible-batch parent is stale or does not bind its state"
                        )
                    if state._pending_step_id is not None:  # noqa: SLF001
                        raise DenseCudaRuntimeError(
                            "CUDA compatible-batch state already has provisional work"
                        )
                    cache = state._cache  # noqa: SLF001
                    if not isinstance(cache, DenseCudaKVSlotCache) or cache._pool is not self._pool:  # noqa: SLF001
                        raise DenseCudaRuntimeError(
                            "CUDA compatible-batch state does not own a lane K/V slot"
                        )
                    if current.lengths[0] + 1 > current.capacity:
                        raise OverflowError("CUDA compatible-batch decode exceeds state capacity")
                    parents.append(current)
                    caches.append(cache)
                    state._pending_step_id = step_id  # noqa: SLF001
                if len({cache.slot for cache in caches}) != width:
                    raise DenseCudaRuntimeError("CUDA compatible batch contains a duplicate slot")

                ids = np.asarray(
                    [work.token_rows[0] for work in works],
                    dtype=np.int64,
                )
                if ids.shape != (width, 1):
                    raise DenseCudaRuntimeError("CUDA compatible decode input is not [B,1]")
                if ids.min() < 0 or ids.max() >= self._runtime._semantic_token_count:  # noqa: SLF001
                    raise ValueError("input token IDs escape the semantic token domain")
                indexed_cache = self._pool.indexed_cache(
                    caches,
                    parent_lengths=tuple(parent.lengths[0] for parent in parents),
                )
                try:
                    result = self._executor(ids, indexed_cache)
                finally:
                    # The engine keeps only a diagnostic last-cache pointer.  Retaining this
                    # ephemeral indexed facade would pin the route's pool after runtime close.
                    self._runtime._forget_engine_cache(indexed_cache)  # noqa: SLF001
                indexed_cache.assert_bound()
                tokens = _next_tokens(
                    result.top1,
                    semantic_token_count=self._runtime._semantic_token_count,  # noqa: SLF001
                    batch_size=width,
                )
                delta = getattr(result, "delta", None)
                if delta is None or int(getattr(delta, "token_count", 0)) != 1:
                    raise DenseCudaRuntimeError(
                        "CUDA compatible decode did not return a one-token K/V delta"
                    )
                if (
                    str(getattr(delta, "cache_id", "")) != indexed_cache.cache_id
                    or int(getattr(delta, "parent_epoch", -1)) != indexed_cache.epoch
                    or tuple(getattr(delta, "parent_lengths", ()))
                    != tuple(int(value) for value in indexed_cache.lengths)
                ):
                    raise DenseCudaRuntimeError(
                        "CUDA compatible decode delta does not bind its indexed cache facade"
                    )
                keys = tuple(getattr(delta, "keys", ()))
                values = tuple(getattr(delta, "values", ()))
                if len(keys) != self._pool.num_layers or len(values) != self._pool.num_layers:
                    raise DenseCudaRuntimeError("CUDA compatible decode K/V layer count drifted")
                expected_delta_shape = (
                    width,
                    1,
                    self._pool.num_kv_heads,
                    self._pool.head_dim,
                )
                if any(
                    tuple(key.shape) != expected_delta_shape
                    or tuple(value.shape) != expected_delta_shape
                    or key.dtype != self._pool.dtype
                    or value.dtype != self._pool.dtype
                    or str(key.device) != str(self._pool.device)
                    or str(value.device) != str(self._pool.device)
                    for key, value in zip(keys, values, strict=True)
                ):
                    raise DenseCudaRuntimeError("CUDA compatible decode K/V geometry drifted")
                scratch_bytes = int(getattr(result, "kv_delta_bytes", 0))
                if scratch_bytes <= 0:
                    scratch_bytes = sum(
                        int(tensor.numel()) * int(tensor.element_size())
                        for tensor in (*keys, *values)
                    )
                if scratch_bytes > self._identity.max_scratch_bytes:
                    raise DenseCudaRuntimeError(
                        "CUDA compatible decode exceeded its admitted scratch bound"
                    )
                scratch = _DenseCudaBatchScratch(
                    result=result,
                    states=states,
                    parents=parents,
                    on_consume=self._scratch_consumed,
                    on_release=self._scratch_released,
                )
                with self._lock:
                    self._active_scratches += 1
                produced = tuple(
                    (
                        index,
                        ProvisionalStep(
                            runtime_id=self._runtime.route.runtime_id,
                            step_id=step_id,
                            request_ids=work.request_ids,
                            state=state,
                            parent=parent,
                            token_counts=(1,),
                            output=NativeOutput(
                                mode=OutputMode.NEXT_TOKEN_ARGMAX,
                                token_ids=(token,),
                            ),
                            authority=DenseCudaBatchProvisionalAuthority(
                                runtime_id=self._runtime.route.runtime_id,
                                step_id=step_id,
                                state=state,
                                scratch=scratch,
                                row=row,
                            ),
                        ),
                    )
                    for row, ((index, work), state, parent, step_id, token) in enumerate(
                        zip(indexed, states, parents, step_ids, tokens, strict=True)
                    )
                )
        except BaseException:
            for state, step_id in zip(states, step_ids, strict=True):
                with state._lock:  # noqa: SLF001
                    if state._pending_step_id == step_id:  # noqa: SLF001
                        state._pending_step_id = None  # noqa: SLF001
            if scratch is not None:
                scratch.discard()
            raise

        elapsed = time.perf_counter() - started
        head_workspace = int(
            getattr(
                getattr(self._runtime._engine, "target", None),  # noqa: SLF001
                "reranked_head_working_bytes_peak",
                0,
            )
        )
        self._runtime._record_compatible_decode(  # noqa: SLF001
            rows=width,
            elapsed=elapsed,
            workspace_bytes=scratch_bytes + head_workspace,
        )
        self._record_forward(width=width, scratch_bytes=scratch_bytes + head_workspace)
        return produced

    def execute(
        self,
        works: Sequence[PrefillWork | DecodeWork],
    ) -> tuple[ProvisionalStep, ...]:
        cohort = tuple(works)
        if not cohort:
            raise ValueError("CUDA compatible-batch work cohort cannot be empty")
        if len(cohort) > self._identity.max_batch_size:
            raise ValueError("CUDA compatible-batch cohort exceeds its declared maximum width")
        if any(not isinstance(work, (PrefillWork, DecodeWork)) for work in cohort):
            raise TypeError("CUDA compatible-batch cohort must contain native work values")
        if any(work.parent.batch_size != 1 for work in cohort):
            raise ValueError("CUDA compatible batching pools independent B1 states only")
        if any(work.parent.runtime_id != self._identity.runtime_id for work in cohort):
            raise DenseCudaRuntimeError("CUDA compatible-batch work belongs to another runtime")
        request_ids = tuple(work.request_ids[0] for work in cohort)
        state_ids = tuple(work.parent.state_id for work in cohort)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("CUDA compatible-batch request IDs must be unique")
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("CUDA compatible-batch state IDs must be unique")
        with self._lock:
            self._dispatches += 1

        compatible = [
            (index, work)
            for index, work in enumerate(cohort)
            if self._compatible(work)
        ]
        compatible_indices = {index for index, _work in compatible}
        produced: list[tuple[int, ProvisionalStep]] = []
        try:
            if len(compatible) > 1:
                produced.extend(
                    self._execute_batch(
                        tuple((index, work) for index, work in compatible)  # type: ignore[misc]
                    )
                )
            for index, work in enumerate(cohort):
                if index in compatible_indices and len(compatible) > 1:
                    continue
                step = (
                    self._runtime.prefill(work)
                    if isinstance(work, PrefillWork)
                    else self._runtime.decode(work)
                )
                produced.append((index, step))
                self._record_forward(width=1, singleton=True)
        except BaseException as primary:
            cleanup_errors: list[BaseException] = []
            for _index, step in reversed(produced):
                try:
                    self._runtime.abandon(step)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            with self._lock:
                self._failed_dispatches += 1
            if cleanup_errors:
                details = "; ".join(str(error) for error in cleanup_errors)
                raise DenseCudaRuntimeError(
                    f"CUDA compatible-batch cleanup failed after {primary}: {details}"
                ) from primary
            raise
        produced.sort(key=lambda item: item[0])
        return tuple(step for _index, step in produced)

    def telemetry(self) -> DenseCudaCompatibleBatchTelemetry:
        with self._lock:
            return DenseCudaCompatibleBatchTelemetry(
                lane_id=self._identity.lane_id,
                dispatches=self._dispatches,
                physical_forwards=self._physical_forwards,
                provisional_rows=self._provisional_rows,
                committed_rows=self._committed_rows,
                abandoned_rows=self._abandoned_rows,
                failed_dispatches=self._failed_dispatches,
                singleton_bypasses=self._singleton_bypasses,
                max_width=self._max_width,
                width_histogram=tuple(sorted(self._width_histogram.items())),
                scratch_peak_bytes=self._scratch_peak_bytes,
                active_scratches=self._active_scratches,
                pool_slots=self._pool.max_slots,
                pool_active_slots=self._pool.active_slots,
            )


__all__ = [
    "CUDA_COMPATIBLE_BATCH_ABI",
    "CUDA_COMPATIBLE_BATCH_NUMERICAL_SUFFIX",
    "DenseCudaCompatibleBatchLane",
    "DenseCudaCompatibleBatchTelemetry",
]
