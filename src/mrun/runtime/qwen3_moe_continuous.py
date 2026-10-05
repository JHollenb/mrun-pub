"""Native continuous-serving bridge for the decomposed Qwen3 MoE CUDA engine.

The Qwen executor currently exposes a ``StaticKVCache`` with one scalar length.  That is enough
for a fixed, equal-length batch, but it cannot represent a continuously refilled ragged cohort.
This module keeps that limitation explicit instead of padding shorter rows or silently changing
attention semantics:

* :class:`Qwen3MoeCudaNativeRuntime` gives every request an independent B1 authority backed by a
  row of one fixed CUDA K/V arena.  Ordinary prefill/decode works against the current engine.
* :class:`Qwen3MoeCompatibleBatchLane` plugs into :class:`NativeGenerationService`, whose existing
  coordinator owns admission, stop state, cancellation, publication, and continuous refill.
* true pooled decode is enabled only by an explicit :class:`Qwen3MoeRaggedDecodeExecutor`.  Its
  contract carries physical slots and per-row lengths, the two facts missing from today's scalar
  ``StaticKVCache.length`` API.  There is no padded or prefix-copy fallback carrying a speed claim.

The shared arena makes row removal cheap: the scheduler compacts each dispatch logically and
passes a dense request order plus a physical-slot indirection vector to the executor.  A finished
row returns its slot to the arena; the next admitted request can reuse it without moving another
request's K/V prefix.
"""

from __future__ import annotations

import heapq
import math
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from .contracts import (
    CommitResult,
    CompatibleBatchLaneIdentity,
    DecodeWork,
    NativeOutput,
    OutputMode,
    PrefillWork,
    PromotionStatus,
    ProvisionalStep,
    RuntimeRoute,
    RuntimeTelemetry,
    StateObservation,
)

QWEN3_MOE_SLOT_KV_ABI = "qwen3-moe-static-kv-slot-pool-v1"
QWEN3_MOE_CONTINUOUS_BATCH_ABI = "qwen3-moe-ragged-slot-decode-v1"
QWEN3_MOE_CONTINUOUS_NUMERICAL_CONTRACT = "qwen3-moe-row-stable-greedy-v1"
QWEN3_MOE_ATTENTION_LENGTH_BUCKET_POLICY = (
    "qwen3-moe-attention-min32-pow2-parent-plus-one-v1"
)
QWEN3_MOE_UNPARTITIONED_ATTENTION_LENGTH_POLICY = (
    "qwen3-moe-attention-unpartitioned-max-parent-plus-one-v1"
)


class Qwen3MoeContinuousRuntimeError(RuntimeError):
    """The Qwen continuous-serving state or executor contract was violated."""


def _strict_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _strict_nonnegative_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return normalized


def _strict_bool(value: bool, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be boolean")
    return value


def _attention_length_bucket(parent_length: int) -> int:
    """Mirror the segmented-GQA kernel's compile-time sequence bucket exactly."""

    sequence_length = _strict_positive_int(parent_length, "parent_length") + 1
    return max(32, 1 << (sequence_length - 1).bit_length())


@dataclass(slots=True)
class _LayerKVView:
    key: Any
    value: Any


@dataclass(slots=True)
class _StaticKVRowView:
    """Duck-typed B1 view accepted by ``Qwen3MoeDecodeRuntime.forward``."""

    layers: list[_LayerKVView]
    batch_size: int
    capacity: int
    length: int = 0

    @property
    def device_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            for tensor in (layer.key, layer.value):
                total += int(tensor.numel()) * int(tensor.element_size())
        return total


@dataclass(frozen=True, slots=True)
class Qwen3MoeKVPoolTelemetry:
    max_slots: int
    max_context_tokens: int
    active_slots: int
    peak_active_slots: int
    allocations: int
    releases: int
    slot_reuses: int
    kv_resident_bytes: int


class Qwen3MoeKVSlotPool:
    """One fixed physical K/V allocation leased as independently versioned B1 rows."""

    def __init__(
        self,
        decode_runtime: Any,
        *,
        max_slots: int = 8,
        max_context_tokens: int,
        kv_bytes_per_token: int | None = None,
    ) -> None:
        self.max_slots = _strict_positive_int(max_slots, "max_slots")
        self.max_context_tokens = _strict_positive_int(
            max_context_tokens,
            "max_context_tokens",
        )
        new_cache = getattr(decode_runtime, "new_cache", None)
        if not callable(new_cache):
            raise TypeError("Qwen decode runtime must expose new_cache(batch_size, capacity)")
        root = new_cache(batch_size=self.max_slots, capacity=self.max_context_tokens)
        if (
            getattr(root, "batch_size", None) != self.max_slots
            or getattr(root, "capacity", None) != self.max_context_tokens
            or not getattr(root, "layers", None)
        ):
            raise Qwen3MoeContinuousRuntimeError(
                "Qwen K/V factory returned a cache with the wrong fixed arena shape"
            )
        for layer in root.layers:
            if not hasattr(layer, "key") or not hasattr(layer, "value"):
                raise Qwen3MoeContinuousRuntimeError("Qwen K/V layer lacks key/value tensors")
            if int(layer.key.shape[0]) != self.max_slots or int(layer.value.shape[0]) != max_slots:
                raise Qwen3MoeContinuousRuntimeError("Qwen K/V layer batch dimension drifted")

        self._decode_runtime = decode_runtime
        self._cache = root
        self._lock = threading.RLock()
        self._free = list(range(self.max_slots))
        heapq.heapify(self._free)
        self._slot_generations = [0] * self.max_slots
        self._active: dict[int, str] = {}
        self._allocations = 0
        self._releases = 0
        self._slot_reuses = 0
        self._peak_active = 0
        self._closed = False
        inferred = self._infer_kv_bytes_per_token()
        if kv_bytes_per_token is None:
            self.kv_bytes_per_token = inferred
        else:
            self.kv_bytes_per_token = _strict_positive_int(
                kv_bytes_per_token,
                "kv_bytes_per_token",
            )
            if inferred and self.kv_bytes_per_token != inferred:
                raise ValueError(
                    "kv_bytes_per_token disagrees with the physical Qwen K/V allocation"
                )

    @property
    def cache(self) -> Any:
        """The full fixed arena; only a bound ragged executor may consume it."""

        return self._cache

    @property
    def device_bytes(self) -> int:
        explicit = getattr(self._cache, "device_bytes", None)
        if explicit is not None:
            return int(explicit)
        return self.max_slots * self.max_context_tokens * self.kv_bytes_per_token

    def _infer_kv_bytes_per_token(self) -> int:
        total = 0
        try:
            for layer in self._cache.layers:
                for tensor in (layer.key, layer.value):
                    token = tensor[0, ..., 0, :]
                    total += int(token.numel()) * int(token.element_size())
        except (AttributeError, IndexError, TypeError):
            return 0
        if total <= 0:
            raise Qwen3MoeContinuousRuntimeError("Qwen K/V tensors have no per-token storage")
        return total

    def _row_view(self, slot: int, capacity: int) -> _StaticKVRowView:
        return _StaticKVRowView(
            layers=[
                _LayerKVView(
                    key=layer.key[slot : slot + 1],
                    value=layer.value[slot : slot + 1],
                )
                for layer in self._cache.layers
            ],
            batch_size=1,
            capacity=capacity,
        )

    def acquire(self, state_id: str, capacity: int) -> tuple[int, int, _StaticKVRowView]:
        capacity = _strict_positive_int(capacity, "capacity")
        if capacity > self.max_context_tokens:
            raise OverflowError("requested Qwen K/V capacity exceeds the fixed slot arena")
        with self._lock:
            if self._closed:
                raise Qwen3MoeContinuousRuntimeError("Qwen K/V slot pool is closed")
            if not self._free:
                raise Qwen3MoeContinuousRuntimeError("Qwen K/V slot pool is exhausted")
            slot = heapq.heappop(self._free)
            generation = self._slot_generations[slot] + 1
            if generation > 1:
                self._slot_reuses += 1
            self._slot_generations[slot] = generation
            self._active[slot] = state_id
            self._allocations += 1
            self._peak_active = max(self._peak_active, len(self._active))
        return slot, generation, self._row_view(slot, capacity)

    def release(self, *, state_id: str, slot: int, storage_generation: int) -> None:
        with self._lock:
            if self._active.get(slot) != state_id:
                raise Qwen3MoeContinuousRuntimeError("Qwen K/V slot authority is stale")
            if self._slot_generations[slot] != storage_generation:
                raise Qwen3MoeContinuousRuntimeError("Qwen K/V slot generation is stale")
            del self._active[slot]
            heapq.heappush(self._free, slot)
            self._releases += 1

    def assert_binding(self, *, state_id: str, slot: int, storage_generation: int) -> None:
        with self._lock:
            if (
                self._active.get(slot) != state_id
                or self._slot_generations[slot] != storage_generation
            ):
                raise Qwen3MoeContinuousRuntimeError("Qwen K/V row no longer owns its slot")

    def telemetry(self) -> Qwen3MoeKVPoolTelemetry:
        with self._lock:
            return Qwen3MoeKVPoolTelemetry(
                max_slots=self.max_slots,
                max_context_tokens=self.max_context_tokens,
                active_slots=len(self._active),
                peak_active_slots=self._peak_active,
                allocations=self._allocations,
                releases=self._releases,
                slot_reuses=self._slot_reuses,
                kv_resident_bytes=self.device_bytes,
            )

    def close(self) -> None:
        with self._lock:
            if self._active:
                raise Qwen3MoeContinuousRuntimeError(
                    "cannot close the Qwen K/V pool with live row authorities"
                )
            self._closed = True


class Qwen3MoeCudaState:
    """One request's committed-length authority over a shared physical K/V row."""

    def __init__(
        self,
        runtime: Qwen3MoeCudaNativeRuntime,
        *,
        owner_id: str,
        capacity: int,
    ) -> None:
        self._runtime = runtime
        self.runtime_id = runtime.route.runtime_id
        self.state_id = f"qwen3-state-{uuid4().hex}"
        self.owner_id = owner_id
        self.capacity = capacity
        self.generation = 0
        self.epoch = 0
        self._lock = threading.RLock()
        self._pending: _Qwen3MoeAuthority | None = None
        self._released = False
        self.slot, self.storage_generation, self._cache = runtime.pool.acquire(
            self.state_id,
            capacity,
        )

    def _observe_unlocked(self) -> StateObservation:
        if self._released:
            raise Qwen3MoeContinuousRuntimeError("Qwen K/V state was released")
        self._runtime.pool.assert_binding(
            state_id=self.state_id,
            slot=self.slot,
            storage_generation=self.storage_generation,
        )
        return StateObservation(
            runtime_id=self.runtime_id,
            state_id=self.state_id,
            generation=self.generation,
            epoch=self.epoch,
            lengths=(int(self._cache.length),),
            capacity=self.capacity,
            state_abi=QWEN3_MOE_SLOT_KV_ABI,
            storage_generation=self.storage_generation,
        )

    def observe(self) -> StateObservation:
        with self._lock:
            return self._observe_unlocked()


@dataclass(slots=True)
class _Qwen3MoeAuthority:
    runtime_id: str
    step_id: str
    state: Qwen3MoeCudaState
    parent: StateObservation
    token_count: int
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class Qwen3MoeRuntimeRates:
    prefill_positions_per_second: float | None
    decode_tokens_per_second: float | None


class Qwen3MoeCudaNativeRuntime:
    """Transactional B1 runtime over fixed Qwen K/V slots.

    The current Qwen forward writes only ``[cache.length:end]``.  We treat that uncommitted tail
    as provisional storage and restore the scalar length before publishing a step.  Commit only
    advances the scalar authority; abandon leaves the unreachable tail ignored.  This avoids a
    full K/V prefix copy while preserving the common runtime's explicit commit boundary.
    """

    def __init__(
        self,
        engine: Any,
        *,
        route: RuntimeRoute,
        max_slots: int = 8,
        max_context_tokens: int,
        kv_bytes_per_token: int | None = None,
        model_resident_bytes: int | None = None,
        synchronize: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        b1_executor: Callable[[Qwen3MoeCudaState, tuple[int, ...]], int] | None = None,
        owns_engine: bool = False,
    ) -> None:
        if not isinstance(route, RuntimeRoute):
            raise TypeError("route must be a RuntimeRoute")
        decode_runtime = getattr(engine, "runtime", None)
        if decode_runtime is None:
            require_runtime = getattr(engine, "_require_runtime", None)
            if callable(require_runtime):
                decode_runtime = require_runtime()
        if decode_runtime is None or not callable(getattr(decode_runtime, "forward", None)):
            raise TypeError("engine must own a live Qwen decode runtime")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if type(owns_engine) is not bool:
            raise TypeError("owns_engine must be boolean")

        self.route = route
        self._engine = engine
        self._decode_runtime = decode_runtime
        self.pool = Qwen3MoeKVSlotPool(
            decode_runtime,
            max_slots=max_slots,
            max_context_tokens=max_context_tokens,
            kv_bytes_per_token=kv_bytes_per_token,
        )
        self._clock = clock
        self._synchronize = synchronize or self._default_synchronize
        self._b1_executor = b1_executor or self._default_b1_executor
        self._owns_engine = owns_engine
        if model_resident_bytes is None:
            working_set_mb = getattr(engine, "working_set_mb", 0.0)
            model_resident_bytes = int(float(working_set_mb) * 1e6)
        if (
            isinstance(model_resident_bytes, bool)
            or not isinstance(model_resident_bytes, int)
            or model_resident_bytes < 0
        ):
            raise ValueError("model_resident_bytes must be a non-negative integer or None")
        self._model_resident_bytes = model_resident_bytes
        self._lock = threading.RLock()
        self._states: dict[str, Qwen3MoeCudaState] = {}
        self._prefill_calls = 0
        self._prefill_tokens = 0
        self._prefill_seconds = 0.0
        self._decode_calls = 0
        self._decode_tokens = 0
        self._decode_seconds = 0.0
        self._provisional_steps = 0
        self._commits = 0
        self._abandons = 0
        self._committed_tokens = 0
        self._closed = False

    def _default_synchronize(self) -> None:
        device = getattr(self._decode_runtime, "device", None)
        if str(device).startswith("cuda"):
            import torch

            torch.cuda.synchronize(device)

    def _default_b1_executor(
        self,
        state: Qwen3MoeCudaState,
        token_ids: tuple[int, ...],
    ) -> int:
        import torch

        device = getattr(self._decode_runtime, "device", getattr(self._engine, "device", "cuda"))
        inputs = torch.as_tensor((token_ids,), device=device, dtype=torch.long)
        result = self._decode_runtime.forward(inputs, cache=state._cache)
        logits = getattr(result, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != 1:
            raise Qwen3MoeContinuousRuntimeError(
                "Qwen B1 forward must return final logits with shape [1,vocab]"
            )
        return int(torch.argmax(logits[0]).item())

    def _state(self, value: Any) -> Qwen3MoeCudaState:
        if not isinstance(value, Qwen3MoeCudaState) or value._runtime is not self:  # noqa: SLF001
            raise Qwen3MoeContinuousRuntimeError("foreign Qwen state authority")
        with self._lock:
            if self._states.get(value.state_id) is not value:
                raise Qwen3MoeContinuousRuntimeError("stale Qwen state authority")
        return value

    def allocate_state(
        self,
        *,
        owner_id: str,
        batch_size: int,
        capacity: int,
    ) -> Qwen3MoeCudaState:
        if batch_size != 1:
            raise ValueError("Qwen continuous runtime allocates independent B1 states only")
        with self._lock:
            if self._closed:
                raise Qwen3MoeContinuousRuntimeError("Qwen continuous runtime is closed")
        state = Qwen3MoeCudaState(self, owner_id=owner_id, capacity=capacity)
        with self._lock:
            self._states[state.state_id] = state
        return state

    def _validate_work(
        self,
        work: PrefillWork | DecodeWork,
    ) -> tuple[Qwen3MoeCudaState, tuple[int, ...]]:
        state = self._state(work.state)
        if work.parent.batch_size != 1 or len(work.token_rows) != 1:
            raise ValueError("Qwen continuous work must be B1")
        if work.output.mode is not OutputMode.NEXT_TOKEN_ARGMAX:
            raise NotImplementedError("Qwen continuous runtime currently supports raw argmax only")
        return state, tuple(work.token_rows[0])

    def _execute(self, work: PrefillWork | DecodeWork) -> ProvisionalStep:
        state, token_ids = self._validate_work(work)
        with state._lock:  # noqa: SLF001 - runtime owns state authority
            current = state._observe_unlocked()  # noqa: SLF001
            if current != work.parent:
                raise Qwen3MoeContinuousRuntimeError("Qwen work binds a stale parent")
            if state._pending is not None:  # noqa: SLF001
                raise Qwen3MoeContinuousRuntimeError("Qwen state already has provisional work")
            parent_length = current.lengths[0]
            state._cache.length = parent_length  # noqa: SLF001
            self._synchronize()
            started = self._clock()
            try:
                next_token = int(self._b1_executor(state, token_ids))
                self._synchronize()
                elapsed = self._clock() - started
            finally:
                # The newly written tail stays as scratch, but it is not semantically visible.
                state._cache.length = parent_length  # noqa: SLF001
            elapsed = _strict_nonnegative_float(elapsed, "B1 execution time")
            if next_token < 0:
                raise Qwen3MoeContinuousRuntimeError("Qwen executor returned a negative token ID")
            step_id = f"qwen3-step-{uuid4().hex}"
            authority = _Qwen3MoeAuthority(
                runtime_id=self.route.runtime_id,
                step_id=step_id,
                state=state,
                parent=current,
                token_count=len(token_ids),
            )
            state._pending = authority  # noqa: SLF001
            try:
                step = ProvisionalStep(
                    runtime_id=self.route.runtime_id,
                    step_id=step_id,
                    request_ids=work.request_ids,
                    state=state,
                    parent=current,
                    token_counts=(len(token_ids),),
                    output=NativeOutput(mode=work.output.mode, token_ids=(next_token,)),
                    authority=authority,
                )
            except BaseException:
                state._pending = None  # noqa: SLF001
                raise
        with self._lock:
            if isinstance(work, PrefillWork):
                self._prefill_calls += 1
                self._prefill_tokens += len(token_ids)
                self._prefill_seconds += elapsed
            else:
                self._decode_calls += 1
                self._decode_tokens += len(token_ids)
                self._decode_seconds += elapsed
            self._provisional_steps += 1
        return step

    def prefill(self, work: PrefillWork) -> ProvisionalStep:
        if not isinstance(work, PrefillWork):
            raise TypeError("prefill requires PrefillWork")
        return self._execute(work)

    def decode(self, work: DecodeWork) -> ProvisionalStep:
        if not isinstance(work, DecodeWork):
            raise TypeError("decode requires DecodeWork")
        return self._execute(work)

    def commit(
        self,
        step: ProvisionalStep,
        accepted_counts: Sequence[int],
    ) -> CommitResult:
        if not isinstance(step, ProvisionalStep):
            raise TypeError("commit requires a ProvisionalStep")
        authority = step.authority
        if (
            not isinstance(authority, _Qwen3MoeAuthority)
            or authority.runtime_id != self.route.runtime_id
        ):
            raise Qwen3MoeContinuousRuntimeError("foreign Qwen provisional authority")
        state = self._state(authority.state)
        accepted = tuple(accepted_counts)
        if accepted != (authority.token_count,):
            raise ValueError("ordinary Qwen generation commits the complete input row")
        with state._lock:  # noqa: SLF001
            if authority.consumed or state._pending is not authority:  # noqa: SLF001
                raise Qwen3MoeContinuousRuntimeError("Qwen provisional authority was consumed")
            before = state._observe_unlocked()  # noqa: SLF001
            if before != authority.parent or before != step.parent:
                raise Qwen3MoeContinuousRuntimeError("Qwen commit parent drifted")
            state._cache.length = before.lengths[0] + authority.token_count  # noqa: SLF001
            state.epoch += 1
            authority.consumed = True
            state._pending = None  # noqa: SLF001
            after = state._observe_unlocked()  # noqa: SLF001
        with self._lock:
            self._commits += 1
            self._committed_tokens += authority.token_count
        return CommitResult(
            runtime_id=self.route.runtime_id,
            step_id=step.step_id,
            state_id=state.state_id,
            accepted_counts=accepted,
            before=before,
            after=after,
            state_bytes_written=authority.token_count * self.pool.kv_bytes_per_token,
        )

    def abandon(self, step: ProvisionalStep) -> None:
        if not isinstance(step, ProvisionalStep):
            raise TypeError("abandon requires a ProvisionalStep")
        authority = step.authority
        if (
            not isinstance(authority, _Qwen3MoeAuthority)
            or authority.runtime_id != self.route.runtime_id
        ):
            raise Qwen3MoeContinuousRuntimeError("foreign Qwen provisional authority")
        state = self._state(authority.state)
        with state._lock:  # noqa: SLF001
            if authority.consumed or state._pending is not authority:  # noqa: SLF001
                raise Qwen3MoeContinuousRuntimeError("Qwen provisional authority was consumed")
            if state._observe_unlocked() != authority.parent:  # noqa: SLF001
                raise Qwen3MoeContinuousRuntimeError("Qwen abandon parent drifted")
            state._cache.length = authority.parent.lengths[0]  # noqa: SLF001
            authority.consumed = True
            state._pending = None  # noqa: SLF001
        with self._lock:
            self._abandons += 1

    def release_state(self, state: Any) -> None:
        owned = self._state(state)
        with owned._lock:  # noqa: SLF001
            if owned._pending is not None:  # noqa: SLF001
                raise Qwen3MoeContinuousRuntimeError("cannot release Qwen state with pending work")
            self.pool.release(
                state_id=owned.state_id,
                slot=owned.slot,
                storage_generation=owned.storage_generation,
            )
            owned._released = True  # noqa: SLF001
        with self._lock:
            del self._states[owned.state_id]

    def _record_batched_decode(self, *, rows: int, elapsed: float) -> None:
        with self._lock:
            self._decode_calls += rows
            self._decode_tokens += rows
            self._decode_seconds += elapsed
            self._provisional_steps += rows

    def rates(self) -> Qwen3MoeRuntimeRates:
        with self._lock:
            return Qwen3MoeRuntimeRates(
                prefill_positions_per_second=(
                    self._prefill_tokens / self._prefill_seconds
                    if self._prefill_seconds > 0
                    else None
                ),
                decode_tokens_per_second=(
                    self._decode_tokens / self._decode_seconds if self._decode_seconds > 0 else None
                ),
            )

    def telemetry(self) -> RuntimeTelemetry:
        with self._lock:
            pool = self.pool.telemetry()
            return RuntimeTelemetry(
                runtime_id=self.route.runtime_id,
                route_backend_id=self.route.backend_id,
                model_fingerprint=self.route.model_fingerprint,
                placement_fingerprint=self.route.placement_fingerprint,
                prefill_calls=self._prefill_calls,
                prefill_tokens=self._prefill_tokens,
                prefill_seconds=self._prefill_seconds,
                decode_calls=self._decode_calls,
                decode_tokens=self._decode_tokens,
                decode_seconds=self._decode_seconds,
                provisional_steps=self._provisional_steps,
                commits=self._commits,
                abandons=self._abandons,
                committed_tokens=self._committed_tokens,
                model_resident_bytes=self._model_resident_bytes,
                kv_resident_bytes=pool.kv_resident_bytes,
                extra_counters=(
                    ("qwen_kv_active_slots", pool.active_slots),
                    ("qwen_kv_allocations", pool.allocations),
                    ("qwen_kv_peak_active_slots", pool.peak_active_slots),
                    ("qwen_kv_releases", pool.releases),
                    ("qwen_kv_slot_reuses", pool.slot_reuses),
                ),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._states:
                raise Qwen3MoeContinuousRuntimeError(
                    "cannot close Qwen continuous runtime with live states"
                )
            self._closed = True
        self.pool.close()
        if self._owns_engine:
            close = getattr(self._engine, "close", None)
            if callable(close):
                close()


@dataclass(frozen=True, slots=True)
class Qwen3MoeDecodeBatch:
    """Exact one-token row-indirected batch submitted to the native executor seam."""

    dispatch_id: str
    request_ids: tuple[str, ...]
    physical_slots: tuple[int, ...]
    storage_generations: tuple[int, ...]
    parent_lengths: tuple[int, ...]
    input_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        width = len(self.request_ids)
        if not self.dispatch_id or width < 1:
            raise ValueError("ragged Qwen decode requires an identity and at least one row")
        fields = (
            self.physical_slots,
            self.storage_generations,
            self.parent_lengths,
            self.input_token_ids,
        )
        if any(len(values) != width for values in fields):
            raise ValueError("ragged Qwen decode fields must have one value per request")
        if len(set(self.request_ids)) != width or len(set(self.physical_slots)) != width:
            raise ValueError("ragged Qwen decode requests and physical slots must be unique")
        if any(value < 0 for values in fields for value in values):
            raise ValueError("ragged Qwen decode integer fields must be non-negative")
        if any(value <= 0 for value in self.storage_generations):
            raise ValueError("ragged Qwen decode storage generations must be positive")
        if any(value <= 0 for value in self.parent_lengths):
            raise ValueError("ragged Qwen decode requires committed non-empty prefixes")

    @property
    def width(self) -> int:
        return len(self.request_ids)

    @property
    def row_bindings(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            zip(
                self.physical_slots,
                self.storage_generations,
                self.parent_lengths,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class Qwen3MoeDecodeBatchResult:
    """Selected tokens plus an order/binding receipt for one physical traversal."""

    dispatch_id: str
    row_bindings: tuple[tuple[int, int, int], ...]
    next_token_ids: tuple[int, ...]


class Qwen3MoeRaggedDecodeExecutor(Protocol):
    """Minimal engine seam required for true ragged continuous decode.

    ``execute`` must consume the full shared slot cache without materializing or padding K/V
    prefixes.  It writes exactly one provisional K/V position at ``parent_lengths[row]`` for
    each physical slot, selects one raw-greedy token per row on device, and leaves all scalar
    state lengths untouched.  Rows must be independently committable after return.
    """

    def execute(
        self,
        batch: Qwen3MoeDecodeBatch,
        *,
        cache: Any,
    ) -> Qwen3MoeDecodeBatchResult: ...


class Qwen3MoeNativeRaggedDecodeExecutor:
    """Bind the continuous lane to ``Qwen3MoeDecodeRuntime.forward_decode_rows``."""

    def __init__(self, engine: Any) -> None:
        decode_runtime = getattr(engine, "runtime", None)
        if decode_runtime is None:
            require_runtime = getattr(engine, "_require_runtime", None)
            if callable(require_runtime):
                decode_runtime = require_runtime()
        forward = getattr(decode_runtime, "forward_decode_rows", None)
        if decode_runtime is None or not callable(forward):
            raise Qwen3MoeContinuousRuntimeError(
                "Qwen continuous decode requires runtime.forward_decode_rows"
            )
        self._decode_runtime = decode_runtime
        self._forward = forward

    def execute(
        self,
        batch: Qwen3MoeDecodeBatch,
        *,
        cache: Any,
    ) -> Qwen3MoeDecodeBatchResult:
        import torch

        device = getattr(self._decode_runtime, "device", None)
        input_token_ids = torch.tensor(
            batch.input_token_ids,
            device=device,
            dtype=torch.long,
        )
        selected = self._forward(
            input_token_ids,
            cache=cache,
            physical_slots=batch.physical_slots,
            parent_lengths=batch.parent_lengths,
        )
        if (
            not isinstance(selected, torch.Tensor)
            or selected.dtype != torch.long
            or selected.device != input_token_ids.device
            or tuple(selected.shape) != (batch.width,)
        ):
            raise Qwen3MoeContinuousRuntimeError(
                "Qwen native ragged decode must return on-device int64 token IDs"
            )
        next_token_ids = tuple(int(value) for value in selected.detach().cpu().tolist())
        return Qwen3MoeDecodeBatchResult(
            dispatch_id=batch.dispatch_id,
            row_bindings=batch.row_bindings,
            next_token_ids=next_token_ids,
        )


@dataclass(frozen=True, slots=True)
class Qwen3MoeBatchTelemetry:
    """Cumulative physical work and attention-length dispatch custody for one lane.

    ``attention_length_bucket_histogram`` counts completed physical subdispatches, not rows.
    Token-iteration estimates model the segmented-GQA loop as ``rows * sequence_bucket``;
    they deliberately exclude the rest of the transformer body and are not timing claims.
    """

    lane_id: str
    singleton_ragged_decode_enabled: bool
    attention_length_bucketing_enabled: bool
    attention_length_bucket_policy: str
    dispatches: int
    physical_decode_calls: int
    physical_subdispatches: int
    decoded_rows: int
    decode_seconds: float
    decode_row_seconds: float
    max_width: int
    width_histogram: tuple[tuple[int, int], ...]
    prefill_fallback_rows: int
    singleton_decode_fallback_rows: int
    singleton_ragged_decode_rows: int
    logical_row_compactions: int
    noncontiguous_slot_dispatches: int
    refilled_rows: int
    attention_length_bucket_histogram: tuple[tuple[int, int], ...]
    split_decode_waves: int
    logical_token_iteration_estimate: int
    unpartitioned_token_iteration_estimate: int
    token_iteration_savings_estimate: int

    @property
    def aggregate_tokens_per_second(self) -> float | None:
        return self.decoded_rows / self.decode_seconds if self.decode_seconds > 0 else None

    @property
    def per_stream_tokens_per_second(self) -> float | None:
        return self.decoded_rows / self.decode_row_seconds if self.decode_row_seconds > 0 else None

    @property
    def token_iteration_savings_fraction(self) -> float | None:
        if self.unpartitioned_token_iteration_estimate <= 0:
            return None
        return (
            self.token_iteration_savings_estimate
            / self.unpartitioned_token_iteration_estimate
        )


class Qwen3MoeCompatibleBatchLane:
    """Dense logical cohorts over row-indirected fixed Qwen K/V slots.

    Singleton decode remains on the scalar runtime unless explicitly enabled and recorded in the
    immutable lane identity.  Optional attention-length partitioning is a separate, default-off
    physical dispatch policy: it groups decode rows by the exact compile-time bucket used by the
    segmented-GQA kernel while retaining the logical cohort's output order and transaction.
    Prefill always uses the scalar authority, including mixed waves.
    """

    def __init__(
        self,
        runtime: Qwen3MoeCudaNativeRuntime,
        executor: Qwen3MoeRaggedDecodeExecutor,
        *,
        max_batch_size: int = 8,
        max_queue_delay_seconds: float = 0.002,
        promotion_status: PromotionStatus = PromotionStatus.EXPERIMENTAL,
        enable_singleton_ragged_decode: bool = False,
        enable_attention_length_bucketing: bool = False,
    ) -> None:
        if not isinstance(runtime, Qwen3MoeCudaNativeRuntime):
            raise TypeError("Qwen compatible batching requires Qwen3MoeCudaNativeRuntime")
        execute = getattr(executor, "execute", None)
        if not callable(execute):
            raise TypeError("ragged Qwen executor must expose execute(batch, cache=...)")
        width = _strict_positive_int(max_batch_size, "max_batch_size")
        if width <= 1 or width > runtime.pool.max_slots:
            raise ValueError("Qwen compatible batch width must be in 2..pool.max_slots")
        queue_delay = _strict_nonnegative_float(
            max_queue_delay_seconds,
            "max_queue_delay_seconds",
        )
        singleton_ragged_decode = _strict_bool(
            enable_singleton_ragged_decode,
            "enable_singleton_ragged_decode",
        )
        attention_length_bucketing = _strict_bool(
            enable_attention_length_bucketing,
            "enable_attention_length_bucketing",
        )
        self._runtime = runtime
        self._executor = executor
        self._identity = CompatibleBatchLaneIdentity(
            lane_id=f"qwen3-compatible-batch-{uuid4().hex}",
            runtime_id=runtime.route.runtime_id,
            lane_abi=QWEN3_MOE_CONTINUOUS_BATCH_ABI,
            numerical_contract=QWEN3_MOE_CONTINUOUS_NUMERICAL_CONTRACT,
            promotion_status=promotion_status,
            max_batch_size=width,
            max_queue_delay_seconds=queue_delay,
            max_scratch_bytes=max(runtime.pool.kv_bytes_per_token * width, 1),
            supports_ragged_dispatch=True,
            dispatches_singletons=singleton_ragged_decode,
        )
        self._lock = threading.RLock()
        self._attention_length_bucketing_enabled = attention_length_bucketing
        self._attention_length_bucket_policy = (
            QWEN3_MOE_ATTENTION_LENGTH_BUCKET_POLICY
            if attention_length_bucketing
            else QWEN3_MOE_UNPARTITIONED_ATTENTION_LENGTH_POLICY
        )
        self._dispatches = 0
        self._physical_decode_calls = 0
        self._decoded_rows = 0
        self._decode_seconds = 0.0
        self._decode_row_seconds = 0.0
        self._max_width = 0
        self._width_histogram: dict[int, int] = {}
        self._prefill_fallback_rows = 0
        self._singleton_decode_fallback_rows = 0
        self._singleton_ragged_decode_rows = 0
        self._logical_row_compactions = 0
        self._noncontiguous_slot_dispatches = 0
        self._refilled_rows = 0
        self._attention_length_bucket_histogram: dict[int, int] = {}
        self._split_decode_waves = 0
        self._logical_token_iteration_estimate = 0
        self._unpartitioned_token_iteration_estimate = 0
        self._token_iteration_savings_estimate = 0

    @property
    def identity(self) -> CompatibleBatchLaneIdentity:
        return self._identity

    @property
    def attention_length_bucket_policy(self) -> str:
        """Immutable physical dispatch policy recorded alongside the lane identity."""

        return self._attention_length_bucket_policy

    def _decode_subdispatches(
        self,
        indexed: Sequence[tuple[int, DecodeWork]],
    ) -> tuple[tuple[int, tuple[tuple[int, DecodeWork], ...]], ...]:
        row_buckets = tuple(
            _attention_length_bucket(work.parent.lengths[0]) for _index, work in indexed
        )
        if not self._attention_length_bucketing_enabled:
            return ((max(row_buckets), tuple(indexed)),)
        partitioned: dict[int, list[tuple[int, DecodeWork]]] = {}
        for item, bucket in zip(indexed, row_buckets, strict=True):
            partitioned.setdefault(bucket, []).append(item)
        return tuple(
            (bucket, tuple(partitioned[bucket])) for bucket in sorted(partitioned)
        )

    def _record_decode_plan(
        self,
        subdispatches: Sequence[tuple[int, Sequence[tuple[int, DecodeWork]]]],
    ) -> None:
        logical_iterations = sum(
            bucket * len(indexed) for bucket, indexed in subdispatches
        )
        rows = sum(len(indexed) for _bucket, indexed in subdispatches)
        unpartitioned_iterations = max(bucket for bucket, _indexed in subdispatches) * rows
        savings = unpartitioned_iterations - logical_iterations
        if savings < 0:
            raise AssertionError("Qwen attention-length partition increased its loop estimate")
        with self._lock:
            self._split_decode_waves += int(len(subdispatches) > 1)
            self._logical_token_iteration_estimate += logical_iterations
            self._unpartitioned_token_iteration_estimate += unpartitioned_iterations
            self._token_iteration_savings_estimate += savings

    def _execute_decode_batch(
        self,
        indexed: Sequence[tuple[int, DecodeWork]],
        *,
        attention_length_bucket: int,
        logical_decode_width: int,
    ) -> tuple[tuple[int, ProvisionalStep], ...]:
        states = tuple(self._runtime._state(work.state) for _index, work in indexed)  # noqa: SLF001
        if len({state.state_id for state in states}) != len(states):
            raise Qwen3MoeContinuousRuntimeError("Qwen decode cohort contains a state twice")
        with ExitStack() as stack:
            for state in sorted(states, key=lambda value: value.state_id):
                stack.enter_context(state._lock)  # noqa: SLF001
            parents: list[StateObservation] = []
            for state, (_index, work) in zip(states, indexed, strict=True):
                current = state._observe_unlocked()  # noqa: SLF001
                if current != work.parent:
                    raise Qwen3MoeContinuousRuntimeError("Qwen batch work binds a stale parent")
                if state._pending is not None:  # noqa: SLF001
                    raise Qwen3MoeContinuousRuntimeError(
                        "Qwen batch state already has provisional work"
                    )
                if work.output.mode is not OutputMode.NEXT_TOKEN_ARGMAX:
                    raise NotImplementedError("Qwen ragged batching supports raw argmax only")
                if len(work.token_rows) != 1 or len(work.token_rows[0]) != 1:
                    raise ValueError("Qwen ragged decode consumes exactly one token per row")
                parents.append(current)

            dispatch = Qwen3MoeDecodeBatch(
                dispatch_id=f"qwen3-dispatch-{uuid4().hex}",
                request_ids=tuple(work.request_ids[0] for _index, work in indexed),
                physical_slots=tuple(state.slot for state in states),
                storage_generations=tuple(state.storage_generation for state in states),
                parent_lengths=tuple(parent.lengths[0] for parent in parents),
                input_token_ids=tuple(work.token_rows[0][0] for _index, work in indexed),
            )
            expected_bucket = max(
                _attention_length_bucket(parent_length)
                for parent_length in dispatch.parent_lengths
            )
            if attention_length_bucket != expected_bucket:
                raise Qwen3MoeContinuousRuntimeError(
                    "Qwen attention-length subdispatch does not match its kernel bucket"
                )
            self._runtime._synchronize()  # noqa: SLF001 - lane shares exact timing boundary
            started = self._runtime._clock()  # noqa: SLF001
            try:
                result = self._executor.execute(dispatch, cache=self._runtime.pool.cache)
                self._runtime._synchronize()  # noqa: SLF001
                elapsed = self._runtime._clock() - started  # noqa: SLF001
            except BaseException:
                for state, parent in zip(states, parents, strict=True):
                    state._cache.length = parent.lengths[0]  # noqa: SLF001
                raise
            elapsed = _strict_nonnegative_float(elapsed, "ragged decode time")
            if not isinstance(result, Qwen3MoeDecodeBatchResult):
                raise TypeError("ragged Qwen executor returned the wrong result type")
            if (
                result.dispatch_id != dispatch.dispatch_id
                or result.row_bindings != dispatch.row_bindings
                or len(result.next_token_ids) != dispatch.width
                or any(token < 0 for token in result.next_token_ids)
            ):
                raise Qwen3MoeContinuousRuntimeError(
                    "ragged Qwen executor result does not bind the submitted row order"
                )
            if any(
                state._cache.length != parent.lengths[0]
                for state, parent in zip(states, parents, strict=True)
            ):  # noqa: SLF001
                raise Qwen3MoeContinuousRuntimeError(
                    "ragged Qwen executor mutated a committed scalar K/V length"
                )

            produced: list[tuple[int, ProvisionalStep]] = []
            installed: list[_Qwen3MoeAuthority] = []
            try:
                for (output_index, work), state, parent, token in zip(
                    indexed,
                    states,
                    parents,
                    result.next_token_ids,
                    strict=True,
                ):
                    step_id = f"qwen3-step-{uuid4().hex}"
                    authority = _Qwen3MoeAuthority(
                        runtime_id=self._runtime.route.runtime_id,
                        step_id=step_id,
                        state=state,
                        parent=parent,
                        token_count=1,
                    )
                    state._pending = authority  # noqa: SLF001
                    installed.append(authority)
                    produced.append(
                        (
                            output_index,
                            ProvisionalStep(
                                runtime_id=self._runtime.route.runtime_id,
                                step_id=step_id,
                                request_ids=work.request_ids,
                                state=state,
                                parent=parent,
                                token_counts=(1,),
                                output=NativeOutput(
                                    mode=work.output.mode,
                                    token_ids=(int(token),),
                                ),
                                authority=authority,
                            ),
                        )
                    )
            except BaseException:
                for authority in installed:
                    authority.state._pending = None  # noqa: SLF001
                raise

        slots = dispatch.physical_slots
        compacted = sum(slot != dense_row for dense_row, slot in enumerate(slots))
        with self._lock:
            self._physical_decode_calls += 1
            self._decoded_rows += dispatch.width
            if logical_decode_width == 1:
                self._singleton_ragged_decode_rows += 1
            self._decode_seconds += elapsed
            self._decode_row_seconds += dispatch.width * elapsed
            self._max_width = max(self._max_width, dispatch.width)
            self._width_histogram[dispatch.width] = self._width_histogram.get(dispatch.width, 0) + 1
            self._logical_row_compactions += compacted
            if slots != tuple(range(dispatch.width)):
                self._noncontiguous_slot_dispatches += 1
            self._refilled_rows += sum(value > 1 for value in dispatch.storage_generations)
            self._attention_length_bucket_histogram[attention_length_bucket] = (
                self._attention_length_bucket_histogram.get(attention_length_bucket, 0) + 1
            )
        self._runtime._record_batched_decode(rows=dispatch.width, elapsed=elapsed)  # noqa: SLF001
        return tuple(produced)

    def execute(
        self,
        works: Sequence[PrefillWork | DecodeWork],
    ) -> tuple[ProvisionalStep, ...]:
        cohort = tuple(works)
        if not cohort:
            raise ValueError("Qwen compatible cohort cannot be empty")
        if len(cohort) > self._identity.max_batch_size:
            raise ValueError("Qwen compatible cohort exceeds its declared width")
        if any(not isinstance(work, (PrefillWork, DecodeWork)) for work in cohort):
            raise TypeError("Qwen compatible cohort contains unsupported work")
        with self._lock:
            self._dispatches += 1

        outputs: list[ProvisionalStep | None] = [None] * len(cohort)
        try:
            decode_items = tuple(
                (index, work) for index, work in enumerate(cohort) if isinstance(work, DecodeWork)
            )
            if len(decode_items) >= 2 or (
                len(decode_items) == 1 and self._identity.dispatches_singletons
            ):
                subdispatches = self._decode_subdispatches(decode_items)
                for attention_length_bucket, subdispatch in subdispatches:
                    for index, step in self._execute_decode_batch(
                        subdispatch,
                        attention_length_bucket=attention_length_bucket,
                        logical_decode_width=len(decode_items),
                    ):
                        outputs[index] = step
                self._record_decode_plan(subdispatches)
            elif decode_items:
                index, work = decode_items[0]
                outputs[index] = self._runtime.decode(work)
                with self._lock:
                    self._singleton_decode_fallback_rows += 1

            for index, work in enumerate(cohort):
                if isinstance(work, PrefillWork):
                    outputs[index] = self._runtime.prefill(work)
                    with self._lock:
                        self._prefill_fallback_rows += 1
        except BaseException as primary:
            cleanup_errors: list[BaseException] = []
            for output in reversed(outputs):
                if output is None:
                    continue
                try:
                    self._runtime.abandon(output)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                detail = "; ".join(str(error) for error in cleanup_errors)
                raise Qwen3MoeContinuousRuntimeError(
                    f"Qwen compatible-lane cleanup failed after {primary}: {detail}"
                ) from primary
            raise
        if any(output is None for output in outputs):
            raise AssertionError("Qwen compatible lane failed to populate every logical row")
        return tuple(output for output in outputs if output is not None)

    def telemetry(self) -> Qwen3MoeBatchTelemetry:
        with self._lock:
            return Qwen3MoeBatchTelemetry(
                lane_id=self._identity.lane_id,
                singleton_ragged_decode_enabled=self._identity.dispatches_singletons,
                attention_length_bucketing_enabled=(
                    self._attention_length_bucketing_enabled
                ),
                attention_length_bucket_policy=self._attention_length_bucket_policy,
                dispatches=self._dispatches,
                physical_decode_calls=self._physical_decode_calls,
                physical_subdispatches=self._physical_decode_calls,
                decoded_rows=self._decoded_rows,
                decode_seconds=self._decode_seconds,
                decode_row_seconds=self._decode_row_seconds,
                max_width=self._max_width,
                width_histogram=tuple(sorted(self._width_histogram.items())),
                prefill_fallback_rows=self._prefill_fallback_rows,
                singleton_decode_fallback_rows=self._singleton_decode_fallback_rows,
                singleton_ragged_decode_rows=self._singleton_ragged_decode_rows,
                logical_row_compactions=self._logical_row_compactions,
                noncontiguous_slot_dispatches=self._noncontiguous_slot_dispatches,
                refilled_rows=self._refilled_rows,
                attention_length_bucket_histogram=tuple(
                    sorted(self._attention_length_bucket_histogram.items())
                ),
                split_decode_waves=self._split_decode_waves,
                logical_token_iteration_estimate=self._logical_token_iteration_estimate,
                unpartitioned_token_iteration_estimate=(
                    self._unpartitioned_token_iteration_estimate
                ),
                token_iteration_savings_estimate=(
                    self._token_iteration_savings_estimate
                ),
            )


@dataclass(slots=True)
class Qwen3MoeContinuousStack:
    """The exact runtime, batch lane, and existing production generation coordinator."""

    runtime: Qwen3MoeCudaNativeRuntime
    lane: Qwen3MoeCompatibleBatchLane
    service: Any = field(repr=False)

    def close(self) -> None:
        self.service.close()


def build_qwen3_moe_continuous_service(
    engine: Any,
    *,
    route: RuntimeRoute,
    max_context_tokens: int,
    semantic_token_count: int,
    decode_executor: Qwen3MoeRaggedDecodeExecutor | None = None,
    max_batch_size: int = 8,
    max_new_tokens: int = 4096,
    max_queue_delay_seconds: float = 0.002,
    event_queue_capacity: int = 64,
    owns_engine: bool = False,
    enable_singleton_ragged_decode: bool = False,
    enable_attention_length_bucketing: bool = False,
) -> Qwen3MoeContinuousStack:
    """Bind Qwen with independent default-off B1 and attention-bucket opt-ins."""

    from .inference.service import NativeGenerationService

    singleton_ragged_decode = _strict_bool(
        enable_singleton_ragged_decode,
        "enable_singleton_ragged_decode",
    )
    attention_length_bucketing = _strict_bool(
        enable_attention_length_bucketing,
        "enable_attention_length_bucketing",
    )
    runtime = Qwen3MoeCudaNativeRuntime(
        engine,
        route=route,
        max_slots=max_batch_size,
        max_context_tokens=max_context_tokens,
        owns_engine=owns_engine,
    )
    try:
        native_executor = decode_executor or Qwen3MoeNativeRaggedDecodeExecutor(engine)
        lane = Qwen3MoeCompatibleBatchLane(
            runtime,
            native_executor,
            max_batch_size=max_batch_size,
            max_queue_delay_seconds=max_queue_delay_seconds,
            enable_singleton_ragged_decode=singleton_ragged_decode,
            enable_attention_length_bucketing=attention_length_bucketing,
        )
        service = NativeGenerationService(
            runtime,
            max_context_tokens=max_context_tokens,
            semantic_token_count=semantic_token_count,
            max_active_requests=max_batch_size,
            supported_output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,),
            max_new_tokens=max_new_tokens,
            event_queue_capacity=event_queue_capacity,
            owns_runtime=True,
            compatible_batch_lane=lane,
            thread_name="mrun-qwen3-moe-continuous",
        )
    except BaseException:
        runtime.close()
        raise
    return Qwen3MoeContinuousStack(runtime=runtime, lane=lane, service=service)


__all__ = [
    "QWEN3_MOE_ATTENTION_LENGTH_BUCKET_POLICY",
    "QWEN3_MOE_CONTINUOUS_BATCH_ABI",
    "QWEN3_MOE_CONTINUOUS_NUMERICAL_CONTRACT",
    "QWEN3_MOE_SLOT_KV_ABI",
    "QWEN3_MOE_UNPARTITIONED_ATTENTION_LENGTH_POLICY",
    "Qwen3MoeBatchTelemetry",
    "Qwen3MoeCompatibleBatchLane",
    "Qwen3MoeContinuousRuntimeError",
    "Qwen3MoeContinuousStack",
    "Qwen3MoeCudaNativeRuntime",
    "Qwen3MoeCudaState",
    "Qwen3MoeDecodeBatch",
    "Qwen3MoeDecodeBatchResult",
    "Qwen3MoeKVPoolTelemetry",
    "Qwen3MoeKVSlotPool",
    "Qwen3MoeNativeRaggedDecodeExecutor",
    "Qwen3MoeRaggedDecodeExecutor",
    "Qwen3MoeRuntimeRates",
    "build_qwen3_moe_continuous_service",
]
