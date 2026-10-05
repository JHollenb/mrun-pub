"""Transactional backend-neutral adapter for the native dense CUDA executor.

The dense QStore engine already computes K/V into an immutable ``KVDelta`` and exposes an
epoch-checked cache commit.  This module binds that concrete mechanism to the public native
runtime ABI without copying model tensors through a framework-neutral compatibility layer.
Greedy argmax retains its established no-logit-materialization fast path.  Stochastic selection
materializes final-position logits only on CUDA, applies the complete policy there, and copies
one selected int64 ID per row to the host.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from .contracts import (
    BackendCapabilities,
    CommitResult,
    CompiledModelIdentity,
    DecodeWork,
    DeviceDescriptor,
    NativeOutput,
    OutputMode,
    PlacementPlan,
    PrefillWork,
    ProvisionalStep,
    RuntimeRoute,
    RuntimeTelemetry,
    SamplingRequest,
    StateForkResult,
    StateObservation,
    WorkloadSpec,
)
from .placement import validate_placement_plan
from .sampling import sampling_adjustments, stateless_uniform, validate_sampling_domain


class DenseCudaRuntimeError(RuntimeError):
    """The concrete CUDA runtime rejected stale, foreign, or unsupported work."""


def _name(value: str, field_name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _strict_counts(values: Sequence[int], *, expected: int) -> tuple[int, ...]:
    counts = tuple(values)
    if len(counts) != expected:
        raise ValueError("accepted_counts must contain one value per state row")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise TypeError("accepted_counts must contain strict integers")
    return tuple(int(value) for value in counts)


def _next_tokens(top1: Any, *, semantic_token_count: int, batch_size: int) -> tuple[int, ...]:
    """Copy only one selected ID per row from device; never materialize the logit matrix."""

    selected = top1[:, -1]
    if hasattr(selected, "detach"):
        selected = selected.detach()
    if hasattr(selected, "to"):
        selected = selected.to("cpu")
    if hasattr(selected, "numpy"):
        selected = selected.numpy()
    array = np.asarray(selected, dtype=np.int64).reshape(-1)
    if array.size != batch_size:
        raise DenseCudaRuntimeError("native top1 result does not align with the state batch")
    values = tuple(int(value) for value in array.tolist())
    if any(value < 0 or value >= semantic_token_count for value in values):
        raise DenseCudaRuntimeError("native top1 escaped the semantic token domain")
    return values


def _sample_torch_row(
    logits: Any,
    request: SamplingRequest,
    *,
    semantic_token_count: int,
) -> Any:
    """Select one token without moving a vocabulary-sized object off its Torch device."""

    import torch

    validate_sampling_domain(request, semantic_token_count)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise DenseCudaRuntimeError("native sampling requires one device logit vector")
    if int(logits.shape[0]) < semantic_token_count:
        raise DenseCudaRuntimeError("native logit width is smaller than the semantic domain")
    scores = logits[:semantic_token_count].float().clone()
    adjustments = sampling_adjustments(request)
    if adjustments:
        indices = torch.tensor(
            [token for token, _ in adjustments],
            dtype=torch.long,
            device=scores.device,
        )
        values = torch.tensor(
            [value for _, value in adjustments],
            dtype=torch.float32,
            device=scores.device,
        )
        scores.index_add_(0, indices, values)

    finite = torch.isfinite(scores).all()
    policy = request.policy
    if policy.temperature == 0.0:
        selected = torch.argmax(scores)
    else:
        # Subtracting the maximum before temperature scaling prevents positive overflow.  Very
        # small temperatures may underflow losing candidates to -inf, which correctly assigns
        # them zero probability rather than silently producing NaNs.
        scaled = (scores - torch.max(scores)) / policy.temperature
        filtered = scaled
        ordered_tokens = None
        if policy.top_k or policy.top_p < 1.0:
            order = torch.argsort(scaled, descending=True, stable=True)
            ordered = torch.index_select(scaled, 0, order)
            if policy.top_k:
                ranks = torch.arange(
                    semantic_token_count,
                    dtype=torch.long,
                    device=scores.device,
                )
                ordered = torch.where(
                    ranks < policy.top_k,
                    ordered,
                    torch.full_like(ordered, float("-inf")),
                )
            base_probabilities = torch.softmax(ordered, dim=0)
            if policy.top_p < 1.0:
                cumulative = torch.cumsum(base_probabilities, dim=0)
                keep = cumulative - base_probabilities < policy.top_p
                ordered = torch.where(
                    keep,
                    ordered,
                    torch.full_like(ordered, float("-inf")),
                )
            filtered = ordered
            ordered_tokens = order
        probabilities = torch.softmax(filtered, dim=0)
        cumulative = torch.cumsum(probabilities, dim=0)
        threshold = torch.tensor(
            stateless_uniform(policy.seed, request.rng_counter),
            dtype=torch.float32,
            device=scores.device,
        )
        selected_position = torch.searchsorted(cumulative, threshold, right=False)
        selected_position = torch.clamp(selected_position, max=semantic_token_count - 1)
        selected = (
            selected_position if ordered_tokens is None else ordered_tokens[selected_position]
        )
        finite = finite & torch.isfinite(probabilities).all() & (cumulative[-1] > 0)

    # Encode numerical failure as an out-of-domain sentinel.  The existing one-token D2H path
    # then rejects it, avoiding a second host synchronization solely for a validity boolean.
    sentinel = torch.full_like(selected, semantic_token_count)
    return torch.where(finite, selected, sentinel)


def _sample_torch_rows(
    logits: Any,
    requests: Sequence[SamplingRequest],
    *,
    semantic_token_count: int,
) -> tuple[int, ...]:
    import torch

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise DenseCudaRuntimeError("native sampling requires device logits shaped [B,1,V]")
    if int(logits.shape[1]) != 1 or int(logits.shape[0]) != len(requests):
        raise DenseCudaRuntimeError("native sampling logits do not align with sampling rows")
    selected = torch.stack(
        [
            _sample_torch_row(
                logits[row, -1],
                request,
                semantic_token_count=semantic_token_count,
            )
            for row, request in enumerate(requests)
        ]
    )[:, None]
    return _next_tokens(
        selected,
        semantic_token_count=semantic_token_count,
        batch_size=len(requests),
    )


def _dense_cuda_execution_binding(engine: Any) -> tuple[Any, ...]:
    target = getattr(engine, "target", None)
    return (
        str(getattr(engine, "backend", "")),
        str(getattr(engine, "head_execution_mode", "exact-fp32-row-blocks")),
        str(getattr(engine, "head_execution_abi", "exact-fp32-row-blocks-v1")),
        str(getattr(engine, "numerical_contract", "")),
        bool(getattr(target, "experimental_reranked_head", False)),
        str(getattr(target, "decode_attention_mode", "established")),
        getattr(target, "decode_attention_tile", 64),
        str(getattr(target, "body_fusion_mode", "established")),
    )


def _cache_storage_signature(cache: Any) -> tuple[Any, ...]:
    """Bind the authority to exact K/V allocations, not merely compatible shapes."""

    def tensor_signature(tensor: Any) -> tuple[Any, ...]:
        pointer = tensor.data_ptr() if hasattr(tensor, "data_ptr") else id(tensor)
        return (
            id(tensor),
            int(pointer),
            tuple(int(value) for value in tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
        )

    return (
        str(cache.cache_id),
        tuple(tensor_signature(tensor) for tensor in cache.keys),
        tuple(tensor_signature(tensor) for tensor in cache.values),
    )


def _cache_storage_pointers(cache: Any) -> tuple[int, ...]:
    """Return concrete allocation addresses solely for in-backend alias rejection."""

    pointers: list[int] = []
    for tensor in (*cache.keys, *cache.values):
        pointer = tensor.data_ptr() if hasattr(tensor, "data_ptr") else id(tensor)
        pointers.append(int(pointer))
    return tuple(pointers)


@dataclass(frozen=True, slots=True)
class _DenseCudaCacheCommitStats:
    accepted_counts: tuple[int, ...]
    kv_write_bytes: int
    epoch_before: int
    epoch_after: int


@dataclass(frozen=True, slots=True)
class _DenseCudaCacheInstallStats:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    installed_lengths: tuple[int, ...]
    kv_copy_bytes: int
    epoch_before: int
    epoch_after: int


class DenseCudaKVSlotCache:
    """One B1 lease over a row of a fixed route-owned CUDA K/V allocation.

    The tensor views retain the ordinary dense-engine cache shape ``[1, T, Hkv, D]`` so exact
    prefill and singleton decode need no compatibility adapter.  ``cache_id`` and
    ``slot_generation`` are minted for every lease: reusing the same physical row cannot revive a
    released authority or make a stale provisional delta valid again.
    """

    def __init__(
        self,
        pool: DenseCudaKVSlotPool,
        *,
        slot: int,
        slot_generation: int,
        capacity: int,
    ) -> None:
        self._pool = pool
        self.slot = int(slot)
        self.slot_generation = int(slot_generation)
        self.num_layers = pool.num_layers
        self.batch_size = 1
        self.max_seq_len = int(capacity)
        self.num_kv_heads = pool.num_kv_heads
        self.head_dim = pool.head_dim
        self.dtype = pool.dtype
        self.device = pool.device
        self.keys = [
            tensor[self.slot : self.slot + 1, : self.max_seq_len]
            for tensor in pool.keys
        ]
        self.values = [
            tensor[self.slot : self.slot + 1, : self.max_seq_len]
            for tensor in pool.values
        ]
        self.lengths = np.zeros(1, dtype=np.int64)
        self.epoch = 0
        self.cache_id = f"{pool.pool_id}.slot-{self.slot}.generation-{self.slot_generation}"
        self._active = True
        self._lock = threading.RLock()

    def _require_active(self) -> None:
        if not self._active or not self._pool._binds(self):  # noqa: SLF001
            raise DenseCudaRuntimeError("CUDA K/V slot lease is stale or released")

    @property
    def committed_bytes(self) -> int:
        self._require_active()
        rows = int(self.lengths[0]) * self.num_layers * self.num_kv_heads * self.head_dim * 2
        return rows * int(self.keys[0].element_size())

    @property
    def allocated_bytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in (*self.keys, *self.values)
        )

    def _validate_delta(self, delta: Any) -> None:
        self._require_active()
        if str(getattr(delta, "cache_id", "")) != self.cache_id:
            raise DenseCudaRuntimeError("K/V delta belongs to another CUDA slot lease")
        if int(getattr(delta, "parent_epoch", -1)) != self.epoch:
            raise DenseCudaRuntimeError("K/V delta epoch is stale for this CUDA slot")
        if tuple(getattr(delta, "parent_lengths", ())) != tuple(
            int(value) for value in self.lengths
        ):
            raise DenseCudaRuntimeError("K/V delta parent length is stale for this CUDA slot")
        token_count = int(getattr(delta, "token_count", 0))
        expected = (1, token_count, self.num_kv_heads, self.head_dim)
        keys = tuple(getattr(delta, "keys", ()))
        values = tuple(getattr(delta, "values", ()))
        if token_count <= 0 or len(keys) != self.num_layers or len(values) != self.num_layers:
            raise DenseCudaRuntimeError("K/V delta geometry differs from the CUDA slot pool")
        for key, value in zip(keys, values, strict=True):
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise DenseCudaRuntimeError("K/V delta row shape differs from the CUDA slot")
            if key.dtype != self.dtype or value.dtype != self.dtype:
                raise DenseCudaRuntimeError("K/V delta dtype differs from the CUDA slot")
            if str(key.device) != str(self.device) or str(value.device) != str(self.device):
                raise DenseCudaRuntimeError("K/V delta device differs from the CUDA slot")

    def commit(self, delta: Any, accepted_counts: Sequence[int]) -> _DenseCudaCacheCommitStats:
        counts = _strict_counts(accepted_counts, expected=1)
        with self._lock:
            self._validate_delta(delta)
            count = counts[0]
            token_count = int(delta.token_count)
            if count < 0 or count > token_count:
                raise ValueError("accepted count outside provisional block")
            start = int(self.lengths[0])
            if start + count > self.max_seq_len:
                raise OverflowError("CUDA K/V slot capacity exceeded")
            epoch_before = self.epoch
            if count:
                for target, source in zip(self.keys, delta.keys, strict=True):
                    target[0, start : start + count].copy_(source[0, :count])
                for target, source in zip(self.values, delta.values, strict=True):
                    target[0, start : start + count].copy_(source[0, :count])
            self.lengths[0] = start + count
            self.epoch += 1
            rows = count * self.num_layers * self.num_kv_heads * self.head_dim * 2
            return _DenseCudaCacheCommitStats(
                accepted_counts=counts,
                kv_write_bytes=rows * int(self.keys[0].element_size()),
                epoch_before=epoch_before,
                epoch_after=self.epoch,
            )

    def commit_batch_row(
        self,
        delta: Any,
        *,
        row: int,
        parent_epoch: int,
        parent_length: int,
        accepted_count: int,
    ) -> _DenseCudaCacheCommitStats:
        """Install one independently accepted row from shared decode scratch."""

        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            raise ValueError("batch scratch row must be a non-negative integer")
        if isinstance(accepted_count, bool) or not isinstance(accepted_count, int):
            raise TypeError("accepted_count must be a strict integer")
        with self._lock:
            self._require_active()
            if self.epoch != parent_epoch or int(self.lengths[0]) != parent_length:
                raise DenseCudaRuntimeError("CUDA batch row parent is stale")
            token_count = int(getattr(delta, "token_count", 0))
            keys = tuple(getattr(delta, "keys", ()))
            values = tuple(getattr(delta, "values", ()))
            if token_count != 1 or len(keys) != self.num_layers or len(values) != self.num_layers:
                raise DenseCudaRuntimeError("CUDA batch scratch must contain one token per layer")
            if accepted_count < 0 or accepted_count > token_count:
                raise ValueError("accepted count outside provisional batch row")
            if parent_length + accepted_count > self.max_seq_len:
                raise OverflowError("CUDA K/V slot capacity exceeded")
            for key, value in zip(keys, values, strict=True):
                expected_tail = (token_count, self.num_kv_heads, self.head_dim)
                if row >= int(key.shape[0]) or row >= int(value.shape[0]):
                    raise DenseCudaRuntimeError("CUDA batch scratch row lies outside its delta")
                if tuple(key.shape[1:]) != expected_tail or tuple(value.shape[1:]) != expected_tail:
                    raise DenseCudaRuntimeError("CUDA batch scratch K/V geometry drifted")
                if key.dtype != self.dtype or value.dtype != self.dtype:
                    raise DenseCudaRuntimeError("CUDA batch scratch dtype differs from its slot")
                if str(key.device) != str(self.device) or str(value.device) != str(self.device):
                    raise DenseCudaRuntimeError("CUDA batch scratch device differs from its slot")
            epoch_before = self.epoch
            if accepted_count:
                for target, source in zip(self.keys, keys, strict=True):
                    target[0, parent_length].copy_(source[row, 0])
                for target, source in zip(self.values, values, strict=True):
                    target[0, parent_length].copy_(source[row, 0])
            self.lengths[0] = parent_length + accepted_count
            self.epoch += 1
            rows = accepted_count * self.num_layers * self.num_kv_heads * self.head_dim * 2
            return _DenseCudaCacheCommitStats(
                accepted_counts=(accepted_count,),
                kv_write_bytes=rows * int(self.keys[0].element_size()),
                epoch_before=epoch_before,
                epoch_after=self.epoch,
            )

    def install_requests(
        self,
        source: Any,
        *,
        source_indices: Sequence[int],
        target_indices: Sequence[int],
    ) -> _DenseCudaCacheInstallStats:
        sources = tuple(int(value) for value in source_indices)
        targets = tuple(int(value) for value in target_indices)
        if sources != (0,) or targets != (0,):
            raise ValueError("a B1 CUDA slot copy requires source and target row zero")
        with self._lock:
            self._require_active()
            if int(getattr(source, "batch_size", 0)) != 1:
                raise ValueError("CUDA slot copy source must be B1")
            length = int(source.lengths[0])
            if length > self.max_seq_len:
                raise OverflowError("source prefix exceeds CUDA slot capacity")
            if len(source.keys) != self.num_layers or len(source.values) != self.num_layers:
                raise DenseCudaRuntimeError("CUDA slot copy layer count drifted")
            epoch_before = self.epoch
            if length:
                for target, source_tensor in zip(self.keys, source.keys, strict=True):
                    target[0, :length].copy_(source_tensor[0, :length])
                for target, source_tensor in zip(self.values, source.values, strict=True):
                    target[0, :length].copy_(source_tensor[0, :length])
            self.lengths[0] = length
            self.epoch += 1
            rows = length * self.num_layers * self.num_kv_heads * self.head_dim * 2
            return _DenseCudaCacheInstallStats(
                source_indices=sources,
                target_indices=targets,
                installed_lengths=(length,),
                kv_copy_bytes=rows * int(self.keys[0].element_size()),
                epoch_before=epoch_before,
                epoch_after=self.epoch,
            )


class DenseCudaKVSlotPool:
    """Fixed per-layer ``[slot, capacity, Hkv, D]`` storage with ABA-safe B1 leases."""

    def __init__(self, backing_cache: Any, *, max_slots: int, capacity: int) -> None:
        if isinstance(max_slots, bool) or not isinstance(max_slots, int) or max_slots <= 0:
            raise ValueError("max_slots must be a positive integer")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if int(getattr(backing_cache, "batch_size", 0)) != max_slots:
            raise ValueError("CUDA slot-pool backing batch differs from max_slots")
        if int(getattr(backing_cache, "max_seq_len", 0)) != capacity:
            raise ValueError("CUDA slot-pool backing context differs from capacity")
        self.pool_id = f"cuda-kv-pool-{uuid4().hex}"
        self.max_slots = int(max_slots)
        self.capacity = int(capacity)
        self.num_layers = int(backing_cache.num_layers)
        self.num_kv_heads = int(backing_cache.num_kv_heads)
        self.head_dim = int(backing_cache.head_dim)
        self.dtype = backing_cache.dtype
        self.device = backing_cache.device
        self.keys = tuple(backing_cache.keys)
        self.values = tuple(backing_cache.values)
        self._backing_cache = backing_cache
        self._generations = [0] * self.max_slots
        self._leases: list[DenseCudaKVSlotCache | None] = [None] * self.max_slots
        self._lock = threading.RLock()
        self._closed = False

    @property
    def allocated_bytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in (*self.keys, *self.values)
        )

    @property
    def active_slots(self) -> int:
        with self._lock:
            return sum(lease is not None for lease in self._leases)

    def _binds(self, cache: DenseCudaKVSlotCache) -> bool:
        with self._lock:
            return (
                not self._closed
                and 0 <= cache.slot < self.max_slots
                and self._leases[cache.slot] is cache
                and self._generations[cache.slot] == cache.slot_generation
            )

    def acquire(self, *, capacity: int) -> DenseCudaKVSlotCache:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("CUDA K/V slot capacity must be an integer")
        if capacity <= 0 or capacity > self.capacity:
            raise ValueError("CUDA K/V slot capacity lies outside the fixed pool")
        with self._lock:
            if self._closed:
                raise DenseCudaRuntimeError("CUDA K/V slot pool is closed")
            try:
                slot = self._leases.index(None)
            except ValueError as exc:
                raise MemoryError("CUDA K/V slot pool is exhausted") from exc
            self._generations[slot] += 1
            cache = DenseCudaKVSlotCache(
                self,
                slot=slot,
                slot_generation=self._generations[slot],
                capacity=capacity,
            )
            self._leases[slot] = cache
            return cache

    def release(self, cache: DenseCudaKVSlotCache) -> None:
        if not isinstance(cache, DenseCudaKVSlotCache) or cache._pool is not self:  # noqa: SLF001
            raise DenseCudaRuntimeError("CUDA K/V slot belongs to another pool")
        # Cache-to-pool is the same order used by commit validation.  Reversing it here can
        # deadlock a terminal operation against a concurrent release attempt.
        with cache._lock, self._lock:  # noqa: SLF001
            if not self._binds(cache):
                raise DenseCudaRuntimeError("CUDA K/V slot lease is stale or released")
            self._leases[cache.slot] = None
            cache._active = False  # noqa: SLF001

    def indexed_cache(
        self,
        caches: Sequence[DenseCudaKVSlotCache],
        *,
        parent_lengths: Sequence[int],
    ) -> DenseCudaIndexedKVCache:
        """Build the exact cache-shaped facade consumed by ``forward_decode_slots``."""

        return DenseCudaIndexedKVCache(
            self,
            caches=tuple(caches),
            parent_lengths=tuple(parent_lengths),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if any(lease is not None for lease in self._leases):
                raise DenseCudaRuntimeError("cannot close CUDA K/V slot pool with live leases")
            self._closed = True
            self._backing_cache = None
            self.keys = ()
            self.values = ()


class DenseCudaIndexedKVCache:
    """Immutable logical batch view over selected rows of a fixed CUDA K/V pool.

    The dense engine deliberately accepts its ordinary cache protocol for indexed decode.  This
    facade supplies logical ``lengths`` and a device-local ``row_indices`` vector while retaining
    the pool's physical per-layer ``[slot, capacity, Hkv, D]`` tensors.  It has no commit method:
    the only mutation authority remains each B1 slot lease.
    """

    def __init__(
        self,
        pool: DenseCudaKVSlotPool,
        *,
        caches: tuple[DenseCudaKVSlotCache, ...],
        parent_lengths: tuple[int, ...],
    ) -> None:
        if not caches or len(caches) != len(parent_lengths):
            raise ValueError("indexed CUDA cache requires one parent length per slot")
        if len({cache.slot for cache in caches}) != len(caches):
            raise DenseCudaRuntimeError("indexed CUDA cache cannot select a slot twice")
        normalized_lengths = tuple(int(value) for value in parent_lengths)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in parent_lengths
        ):
            raise TypeError("indexed CUDA cache parent lengths must be strict integers")
        with pool._lock:  # noqa: SLF001 - facade is minted by its owning pool
            if pool._closed:  # noqa: SLF001
                raise DenseCudaRuntimeError("CUDA K/V slot pool is closed")
            for cache, length in zip(caches, normalized_lengths, strict=True):
                if cache._pool is not pool or not pool._binds(cache):  # noqa: SLF001
                    raise DenseCudaRuntimeError("indexed CUDA cache contains a stale slot lease")
                if length != int(cache.lengths[0]) or length < 0:
                    raise DenseCudaRuntimeError("indexed CUDA cache parent length is stale")
                if length + 1 > cache.max_seq_len:
                    raise OverflowError("indexed CUDA decode exceeds a slot's capacity")

            self._pool = pool
            self._caches = caches
            self._bindings = tuple(
                (cache.cache_id, cache.slot, cache.slot_generation, cache.epoch)
                for cache in caches
            )
            self.keys = pool.keys
            self.values = pool.values
            self.num_layers = pool.num_layers
            self.batch_size = len(caches)
            self.max_seq_len = pool.capacity
            self.num_kv_heads = pool.num_kv_heads
            self.head_dim = pool.head_dim
            self.dtype = pool.dtype
            self.device = pool.device
        self.lengths = np.asarray(normalized_lengths, dtype=np.int64)
        self.epoch = 0
        self.cache_id = f"{pool.pool_id}.indexed-{uuid4().hex}"
        self._lock = threading.RLock()
        import torch

        self.row_indices = torch.as_tensor(
            tuple(cache.slot for cache in caches),
            device=self.device,
            dtype=torch.long,
        )
        self._row_indices_signature = (
            int(self.row_indices.data_ptr()),
            tuple(self.row_indices.shape),
            self.row_indices.dtype,
            self.row_indices.device,
            int(self.row_indices._version),  # noqa: SLF001 - detects in-place mapping mutation
        )

    @property
    def allocated_bytes(self) -> int:
        return self._pool.allocated_bytes

    @property
    def committed_bytes(self) -> int:
        rows = sum(int(value) for value in self.lengths)
        elements = rows * self.num_layers * self.num_kv_heads * self.head_dim * 2
        return elements * int(self.keys[0].element_size())

    def assert_bound(self) -> None:
        """Reject lease ABA, logical-length mutation, and physical-pool drift."""

        with self._lock, self._pool._lock:  # noqa: SLF001
            if self.epoch != 0:
                raise DenseCudaRuntimeError("indexed CUDA cache epoch mutated during decode")
            if tuple(int(value) for value in self.lengths) != tuple(
                int(cache.lengths[0]) for cache in self._caches
            ):
                raise DenseCudaRuntimeError("indexed CUDA cache lengths mutated during decode")
            row_indices_signature = (
                int(self.row_indices.data_ptr()),
                tuple(self.row_indices.shape),
                self.row_indices.dtype,
                self.row_indices.device,
                int(self.row_indices._version),  # noqa: SLF001
            )
            if row_indices_signature != self._row_indices_signature:
                raise DenseCudaRuntimeError("indexed CUDA cache row mapping mutated during decode")
            for cache, binding in zip(self._caches, self._bindings, strict=True):
                expected = (cache.cache_id, cache.slot, cache.slot_generation, cache.epoch)
                if expected != binding or not self._pool._binds(cache):  # noqa: SLF001
                    raise DenseCudaRuntimeError("indexed CUDA cache slot binding drifted")


class DenseCudaState:
    """Opaque authority over one fixed, request-isolated CUDA K/V arena."""

    __slots__ = (
        "_cache",
        "_capacity",
        "_generation",
        "_lock",
        "_owner_id",
        "_pending_step_id",
        "_released",
        "_runtime_id",
        "_state_abi",
        "_state_id",
        "_storage_generation",
        "_storage_signature",
    )

    def __init__(
        self,
        *,
        runtime_id: str,
        state_id: str,
        owner_id: str,
        state_abi: str,
        capacity: int,
        generation: int,
        cache: Any,
    ) -> None:
        self._runtime_id = runtime_id
        self._state_id = state_id
        self._owner_id = owner_id
        self._state_abi = state_abi
        self._capacity = int(capacity)
        self._cache = cache
        self._generation = int(generation)
        self._storage_generation = int(getattr(cache, "slot_generation", 0))
        self._storage_signature = _cache_storage_signature(cache)
        self._pending_step_id: str | None = None
        self._released = False
        self._lock = threading.RLock()

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def state_id(self) -> str:
        return self._state_id

    @property
    def owner_id(self) -> str:
        return self._owner_id

    def _observe_unlocked(self) -> StateObservation:
        if self._released:
            raise DenseCudaRuntimeError("state authority has been released")
        if _cache_storage_signature(self._cache) != self._storage_signature:
            raise DenseCudaRuntimeError("state K/V backing storage identity changed")
        lengths = tuple(int(value) for value in self._cache.lengths)
        return StateObservation(
            runtime_id=self._runtime_id,
            state_id=self._state_id,
            generation=self._generation,
            epoch=int(self._cache.epoch),
            lengths=lengths,
            capacity=self._capacity,
            state_abi=self._state_abi,
            storage_generation=self._storage_generation,
        )

    def observe(self) -> StateObservation:
        with self._lock:
            return self._observe_unlocked()


class DenseCudaProvisionalAuthority:
    """Single-use runtime-owned handle retaining the device-resident provisional delta."""

    __slots__ = ("_consumed", "_result", "_runtime_id", "_state", "_step_id")

    def __init__(
        self,
        *,
        runtime_id: str,
        step_id: str,
        state: DenseCudaState,
        result: Any,
    ) -> None:
        self._runtime_id = runtime_id
        self._step_id = step_id
        self._state = state
        self._result = result
        self._consumed = False

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def step_id(self) -> str:
        return self._step_id


class DenseCudaBatchProvisionalAuthority:
    """Per-row terminal authority over one shared CUDA decode scratch allocation."""

    __slots__ = (
        "_consumed",
        "_row",
        "_runtime_id",
        "_scratch",
        "_state",
        "_step_id",
    )

    def __init__(
        self,
        *,
        runtime_id: str,
        step_id: str,
        state: DenseCudaState,
        scratch: Any,
        row: int,
    ) -> None:
        self._runtime_id = runtime_id
        self._step_id = step_id
        self._state = state
        self._scratch = scratch
        self._row = int(row)
        self._consumed = False

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def step_id(self) -> str:
        return self._step_id


class DenseCudaNativeRuntime:
    """Concrete ``ModelRuntime`` over ``DenseQStoreCudaEngine`` and fixed CUDA K/V arenas."""

    def __init__(
        self,
        engine: Any,
        *,
        route: RuntimeRoute,
        placement: PlacementPlan,
        semantic_token_count: int,
        state_abi: str,
        supported_output_modes: Sequence[OutputMode] = (
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ),
        admitted_body_workspace_bytes: int = 0,
        cache_factory: Callable[[int, int], Any] | None = None,
        owns_engine: bool = False,
    ) -> None:
        if not isinstance(route, RuntimeRoute) or not isinstance(placement, PlacementPlan):
            raise TypeError("dense CUDA runtime requires canonical route and placement values")
        if route.placement_fingerprint != placement.fingerprint:
            raise ValueError("runtime route does not bind the supplied placement")
        if route.model_fingerprint != placement.model_fingerprint:
            raise ValueError("runtime route and placement bind different models")
        if route.backend_id != placement.backend_id:
            raise ValueError("runtime route and placement bind different backends")
        if route.device_id != placement.device_id:
            raise ValueError("runtime route and placement bind different devices")
        if state_abi != placement.state.state_abi:
            raise ValueError("runtime state ABI differs from placement")
        if isinstance(semantic_token_count, bool) or not isinstance(semantic_token_count, int):
            raise TypeError("semantic_token_count must be an integer")
        if semantic_token_count <= 0:
            raise ValueError("semantic_token_count must be positive")
        engine_backend = str(getattr(engine, "backend", ""))
        if engine_backend != route.backend_id:
            raise ValueError(
                f"engine backend {engine_backend!r} differs from route {route.backend_id!r}"
            )
        engine_limit = int(getattr(engine, "max_seq_len", 0))
        if engine_limit < placement.state.max_context_tokens:
            raise ValueError("engine context limit is smaller than the admitted placement")
        engine_tokens = int(getattr(engine, "semantic_token_count", semantic_token_count))
        if engine_tokens != semantic_token_count:
            raise ValueError("engine semantic token domain differs from the compiled model")
        output_modes = tuple(supported_output_modes)
        if not output_modes or any(not isinstance(mode, OutputMode) for mode in output_modes):
            raise TypeError("supported_output_modes must contain OutputMode values")
        if len(set(output_modes)) != len(output_modes):
            raise ValueError("supported_output_modes must be unique")
        if any(
            mode not in (OutputMode.NEXT_TOKEN_ARGMAX, OutputMode.NEXT_TOKEN_SAMPLE)
            for mode in output_modes
        ):
            raise ValueError("dense CUDA runtime received an unsupported output capability")
        if (
            isinstance(admitted_body_workspace_bytes, bool)
            or not isinstance(admitted_body_workspace_bytes, int)
            or admitted_body_workspace_bytes < 0
        ):
            raise TypeError("admitted_body_workspace_bytes must be a non-negative integer")
        if admitted_body_workspace_bytes > placement.workspace_bytes:
            raise ValueError("admitted body workspace exceeds total placement workspace")

        self._engine = engine
        self._route = route
        self._placement = placement
        self._semantic_token_count = semantic_token_count
        self._state_abi = state_abi
        self._supported_output_modes = frozenset(output_modes)
        self._admitted_body_workspace_bytes = admitted_body_workspace_bytes
        self._compact_head_route = route.backend_id == "cuda-source-int8-compact-head"
        self._head_execution_binding = _dense_cuda_execution_binding(engine)
        decode_attention_mode = self._head_execution_binding[5]
        decode_attention_tile = self._head_execution_binding[6]
        body_fusion_mode = self._head_execution_binding[7]
        if decode_attention_mode not in {
            "established",
            "segmented-flash-gqa-decode-v1",
        }:
            raise DenseCudaRuntimeError("dense CUDA decode attention mode is not admitted")
        if (
            isinstance(decode_attention_tile, bool)
            or not isinstance(decode_attention_tile, int)
            or decode_attention_tile < 16
            or decode_attention_tile > 256
            or decode_attention_tile & (decode_attention_tile - 1)
        ):
            raise DenseCudaRuntimeError("dense CUDA decode attention tile is not admitted")
        if body_fusion_mode not in {"established", "residual-rms-swiglu-v1"}:
            raise DenseCudaRuntimeError("dense CUDA body fusion mode is not admitted")
        if self._compact_head_route:
            decode_contract_suffix = (
                ""
                if decode_attention_mode == "established"
                else "+segmented-flash-gqa-decode-v1"
            )
            body_contract_suffix = (
                "" if body_fusion_mode == "established" else "+residual-rms-swiglu-v1"
            )
            expected = (
                "cuda-source-int8-compact-head",
                "semantic-prefix-w8a16-top2-fp32-rerank",
                "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1",
                (
                    "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1+"
                    "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"
                    f"{decode_contract_suffix}{body_contract_suffix}"
                ),
                True,
                decode_attention_mode,
                decode_attention_tile,
                body_fusion_mode,
            )
            if self._head_execution_binding != expected:
                raise DenseCudaRuntimeError(
                    "compact CUDA runtime route does not match the admitted head identity"
                )
            if self._supported_output_modes != {OutputMode.NEXT_TOKEN_ARGMAX}:
                raise DenseCudaRuntimeError("compact CUDA runtime must be greedy argmax only")
            self._forward_argmax = getattr(engine, "forward_last_top1", None)
            if not callable(self._forward_argmax):
                raise DenseCudaRuntimeError(
                    "compact CUDA runtime requires bounded final-row execution"
                )
        else:
            if (
                self._head_execution_binding[1] == ("semantic-prefix-w8a16-top2-fp32-rerank")
                or self._head_execution_binding[0] == "cuda-source-int8-compact-head"
            ):
                raise DenseCudaRuntimeError(
                    "compact CUDA head cannot execute through a non-compact runtime route"
                )
            self._forward_argmax = getattr(engine, "forward_block", None)
            if not callable(self._forward_argmax):
                raise DenseCudaRuntimeError("dense CUDA runtime requires block execution")
        self._cache_factory = cache_factory or self._default_cache_factory
        self._kv_slot_pool: DenseCudaKVSlotPool | None = None
        self._owns_engine = bool(owns_engine)
        self._states: dict[str, DenseCudaState] = {}
        self._next_state_generation = 1
        self._lock = threading.RLock()
        self._closed = False
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
        self._device_to_host_bytes = 0
        self._workspace_peak_bytes = 0
        self._state_forks = 0
        self._state_fork_tokens = 0
        self._state_fork_bytes = 0

    @classmethod
    def bind(
        cls,
        engine: Any,
        *,
        model: CompiledModelIdentity,
        workload: WorkloadSpec,
        capabilities: BackendCapabilities,
        device: DeviceDescriptor,
        placement: PlacementPlan,
        admitted_body_workspace_bytes: int = 0,
        cache_factory: Callable[[int, int], Any] | None = None,
        owns_engine: bool = False,
    ) -> DenseCudaNativeRuntime:
        """Validate the complete compiler/capability/placement chain and create a live route."""

        validate_placement_plan(placement, model, workload, capabilities, device)
        if workload.output_mode not in (
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ):
            raise NotImplementedError(
                "dense CUDA native runtime promotes only native next-token selection"
            )
        if str(getattr(engine, "arch", "")) != model.architecture:
            raise ValueError("engine architecture differs from the compiled model")
        numerical_contract = str(getattr(engine, "numerical_contract", ""))
        if numerical_contract != workload.numerical_contract:
            raise ValueError("engine numerical contract differs from the admitted workload")
        route = RuntimeRoute(
            runtime_id=f"dense-cuda-{uuid4().hex}",
            model_fingerprint=model.fingerprint,
            capability_fingerprint=capabilities.fingerprint,
            placement_fingerprint=placement.fingerprint,
            backend_id=capabilities.backend_id,
            device_id=device.device_id,
            promotion_status=capabilities.promotion_status,
        )
        return cls(
            engine,
            route=route,
            placement=placement,
            semantic_token_count=model.semantic_token_count,
            state_abi=model.state_abi,
            supported_output_modes=capabilities.output_modes,
            admitted_body_workspace_bytes=admitted_body_workspace_bytes,
            cache_factory=cache_factory,
            owns_engine=owns_engine,
        )

    @property
    def route(self) -> RuntimeRoute:
        return self._route

    def _default_cache_factory(self, batch_size: int, capacity: int) -> Any:
        from mrun.engine.dense_qstore_cuda import DenseQStoreKVCache

        return DenseQStoreKVCache.for_store(
            self._engine.store,
            batch_size=batch_size,
            max_seq_len=capacity,
        )

    def _assert_execution_binding(self) -> None:
        if _dense_cuda_execution_binding(self._engine) != self._head_execution_binding:
            raise DenseCudaRuntimeError(
                "dense CUDA head/decode execution identity changed after runtime binding"
            )

    def attach_kv_slot_pool(self, *, max_slots: int) -> DenseCudaKVSlotPool:
        """Replace per-state allocation with one fixed pool before the first state exists."""

        if isinstance(max_slots, bool) or not isinstance(max_slots, int) or max_slots <= 1:
            raise ValueError("CUDA compatible batching requires at least two K/V slots")
        with self._lock:
            self._require_open()
            if self._states:
                raise DenseCudaRuntimeError(
                    "CUDA K/V slot pool must attach before state allocation"
                )
            if self._kv_slot_pool is not None:
                raise DenseCudaRuntimeError("dense CUDA runtime already has a K/V slot pool")
            capacity = self._placement.state.max_context_tokens
            backing = self._cache_factory(max_slots, capacity)
            pool = DenseCudaKVSlotPool(
                backing,
                max_slots=max_slots,
                capacity=capacity,
            )
            expected = self._placement.state.bytes_per_token * max_slots * capacity
            if pool.allocated_bytes != expected:
                raise DenseCudaRuntimeError(
                    "CUDA K/V slot-pool bytes differ from state ABI accounting "
                    f"({pool.allocated_bytes} != {expected})"
                )
            self._kv_slot_pool = pool
            return pool

    def _new_cache_unlocked(self, batch_size: int, capacity: int) -> Any:
        pool = self._kv_slot_pool
        if pool is not None:
            if batch_size != 1:
                raise ValueError("a fixed CUDA K/V slot pool issues B1 state authorities only")
            return pool.acquire(capacity=capacity)
        return self._cache_factory(batch_size, capacity)

    def _record_compatible_decode(
        self,
        *,
        rows: int,
        elapsed: float,
        workspace_bytes: int,
    ) -> None:
        """Account one physical batch forward and its independently provisional rows."""

        with self._lock:
            self._provisional_steps += rows
            self._device_to_host_bytes += rows * 8
            self._workspace_peak_bytes = max(self._workspace_peak_bytes, workspace_bytes)
            self._decode_calls += 1
            self._decode_tokens += rows
            self._decode_seconds += elapsed

    def _require_open(self) -> None:
        if self._closed:
            raise DenseCudaRuntimeError("native CUDA runtime is closed")

    def _forget_engine_cache(self, cache: Any) -> None:
        """Drop the engine's diagnostic cache reference only when it names ``cache``.

        Dense CUDA engines remember the most recently executed cache for standalone engine
        telemetry.  Runtime state owns that arena, so the diagnostic pointer must not extend
        its lifetime after release.  Identity matters here: another live state may have
        replaced ``_last_cache`` between this state's final execution and its release.
        """

        if getattr(self._engine, "_last_cache", None) is cache:
            self._engine._last_cache = None  # noqa: SLF001 - runtime releases engine custody

    def _state(self, handle: Any) -> DenseCudaState:
        with self._lock:
            self._require_open()
            if not isinstance(handle, DenseCudaState):
                raise TypeError("state was not issued by the dense CUDA runtime")
            if handle.runtime_id != self._route.runtime_id:
                raise DenseCudaRuntimeError("state belongs to another runtime")
            if self._states.get(handle.state_id) is not handle:
                raise DenseCudaRuntimeError("state authority is stale or released")
            return handle

    def _mint_state_generation_unlocked(self) -> int:
        generation = self._next_state_generation
        self._next_state_generation += 1
        return generation

    def allocate_state(
        self,
        *,
        owner_id: str,
        batch_size: int,
        capacity: int,
    ) -> DenseCudaState:
        owner_id = _name(owner_id, "owner_id")
        for value, field_name in ((batch_size, "batch_size"), (capacity, "capacity")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if batch_size > self._placement.state.max_batch_size:
            raise ValueError("batch_size exceeds the admitted placement")
        if capacity > self._placement.state.max_context_tokens:
            raise ValueError("capacity exceeds the admitted placement")
        with self._lock:
            self._require_open()
            cache = self._new_cache_unlocked(batch_size, capacity)
            if int(cache.batch_size) != batch_size or int(cache.max_seq_len) != capacity:
                raise DenseCudaRuntimeError("cache factory violated the requested arena shape")
            state_id = f"state-{uuid4().hex}"
            state = DenseCudaState(
                runtime_id=self._route.runtime_id,
                state_id=state_id,
                owner_id=owner_id,
                state_abi=self._state_abi,
                capacity=capacity,
                generation=0,
                cache=cache,
            )
            self._states[state_id] = state
            return state

    def fork_state(
        self,
        source: Any,
        *,
        parent: StateObservation,
        owner_id: str,
        capacity: int,
    ) -> StateForkResult:
        """Copy every committed source row into a fresh native cache allocation."""

        if not isinstance(parent, StateObservation):
            raise TypeError("state fork parent must be a StateObservation")
        owner_id = _name(owner_id, "owner_id")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0 or capacity > self._placement.state.max_context_tokens:
            raise ValueError("fork capacity lies outside the admitted placement")
        resolved = self._state(source)
        with self._lock, resolved._lock:  # noqa: SLF001 - runtime owns state authority
            self._require_open()
            if self._states.get(resolved.state_id) is not resolved:
                raise DenseCudaRuntimeError("source state authority is stale or released")
            current = resolved._observe_unlocked()  # noqa: SLF001
            if resolved._pending_step_id is not None:  # noqa: SLF001
                raise DenseCudaRuntimeError("cannot fork state with pending provisional work")
            if current != parent:
                raise DenseCudaRuntimeError("state fork parent is stale or does not bind source")
            if current.state_abi != self._state_abi:
                raise DenseCudaRuntimeError("source state ABI differs from the runtime")
            if max(current.lengths) > capacity:
                raise OverflowError("committed source prefix exceeds fork capacity")
            source_cache = resolved._cache  # noqa: SLF001
            if (
                int(source_cache.batch_size) != current.batch_size
                or int(source_cache.max_seq_len) != current.capacity
            ):
                raise DenseCudaRuntimeError("source cache geometry differs from state authority")
            source_expected_bytes = (
                self._placement.state.bytes_per_token * current.batch_size * current.capacity
            )
            if int(getattr(source_cache, "allocated_bytes", -1)) != source_expected_bytes:
                raise DenseCudaRuntimeError("source cache bytes differ from state ABI accounting")
            source_pointers = _cache_storage_pointers(source_cache)
            if len(set(source_pointers)) != len(source_pointers):
                raise DenseCudaRuntimeError("source cache contains aliased backing storage")

            cache = self._new_cache_unlocked(current.batch_size, capacity)
            if int(cache.batch_size) != current.batch_size or int(cache.max_seq_len) != capacity:
                raise DenseCudaRuntimeError("cache factory violated the fork arena shape")
            if any(int(length) != 0 for length in cache.lengths) or int(cache.epoch) != 0:
                raise DenseCudaRuntimeError("fork cache factory must return fresh empty state")
            expected_bytes = self._placement.state.bytes_per_token * current.batch_size * capacity
            if int(getattr(cache, "allocated_bytes", -1)) != expected_bytes:
                raise DenseCudaRuntimeError(
                    "fork cache bytes differ from placement accounting "
                    f"({getattr(cache, 'allocated_bytes', None)} != {expected_bytes})"
                )
            target_pointers = _cache_storage_pointers(cache)
            if len(set(target_pointers)) != len(target_pointers):
                raise DenseCudaRuntimeError("fork cache contains aliased backing storage")
            if set(target_pointers).intersection(source_pointers):
                raise DenseCudaRuntimeError("fork cache aliases source backing storage")
            target_signature = _cache_storage_signature(cache)
            install = getattr(cache, "install_requests", None)
            if not callable(install):
                raise DenseCudaRuntimeError("native cache lacks exact-prefix install support")
            indices = tuple(range(current.batch_size))
            stats = install(
                source_cache,
                source_indices=indices,
                target_indices=indices,
            )
            if tuple(stats.installed_lengths) != current.lengths:
                raise DenseCudaRuntimeError("native cache did not preserve fork row lengths")
            expected_copied = sum(current.lengths) * self._placement.state.bytes_per_token
            copied = int(stats.kv_copy_bytes)
            if copied != expected_copied:
                raise DenseCudaRuntimeError(
                    "native cache fork bytes differ from state ABI accounting"
                )
            if _cache_storage_signature(cache) != target_signature:
                raise DenseCudaRuntimeError("fork cache backing storage changed during copy")
            if resolved._observe_unlocked() != current:  # noqa: SLF001
                raise DenseCudaRuntimeError("native fork mutated its source state")

            state_id = f"state-{uuid4().hex}"
            forked_state = DenseCudaState(
                runtime_id=self._route.runtime_id,
                state_id=state_id,
                owner_id=owner_id,
                state_abi=self._state_abi,
                capacity=capacity,
                generation=self._mint_state_generation_unlocked(),
                cache=cache,
            )
            forked = forked_state._observe_unlocked()  # noqa: SLF001
            if forked.lengths != current.lengths:
                raise DenseCudaRuntimeError("forked state observation lost committed row lengths")
            result = StateForkResult(
                runtime_id=self._route.runtime_id,
                source=current,
                forked=forked,
                state=forked_state,
                state_bytes_copied=copied,
            )
            self._states[state_id] = forked_state
            self._state_forks += 1
            self._state_fork_tokens += sum(current.lengths)
            self._state_fork_bytes += copied
            return result

    def _execute(self, work: PrefillWork | DecodeWork, *, phase: str) -> ProvisionalStep:
        self._assert_execution_binding()
        if work.output.mode not in self._supported_output_modes:
            raise NotImplementedError(
                f"dense CUDA route does not admit {work.output.mode.value!r} output"
            )
        state = self._state(work.state)
        with state._lock:  # noqa: SLF001 - runtime owns the opaque state implementation
            current = state._observe_unlocked()  # noqa: SLF001
            if work.parent != current:
                raise DenseCudaRuntimeError("work parent is stale or does not bind this state")
            if state._pending_step_id is not None:  # noqa: SLF001
                raise DenseCudaRuntimeError("state already has an unconsumed provisional step")
            widths = {len(row) for row in work.token_rows}
            if len(widths) != 1:
                raise NotImplementedError("dense CUDA native runtime requires equal-width rows")
            ids = np.asarray(work.token_rows, dtype=np.int64)
            if ids.size and (ids.min() < 0 or ids.max() >= self._semantic_token_count):
                raise ValueError("input token IDs escape the semantic token domain")
            step_id = f"step-{uuid4().hex}"
            state._pending_step_id = step_id  # noqa: SLF001
            try:
                started = time.perf_counter()
                if work.output.mode is OutputMode.NEXT_TOKEN_ARGMAX:
                    if self._compact_head_route:
                        result = self._forward_argmax(ids, state._cache)
                    else:
                        result = self._forward_argmax(
                            ids,
                            state._cache,
                            return_logits=False,
                        )
                else:
                    forward_last_logits = getattr(self._engine, "forward_last_logits", None)
                    if not callable(forward_last_logits):
                        raise DenseCudaRuntimeError(
                            "dense CUDA engine lacks bounded final-logit sampling execution"
                        )
                    result = forward_last_logits(ids, state._cache)  # noqa: SLF001
                elapsed = time.perf_counter() - started
                if work.output.mode is OutputMode.NEXT_TOKEN_ARGMAX:
                    tokens = _next_tokens(
                        result.top1,
                        semantic_token_count=self._semantic_token_count,
                        batch_size=current.batch_size,
                    )
                else:
                    logits = getattr(result, "logits", None)
                    if (
                        logits is None
                        or getattr(getattr(logits, "device", None), "type", None) != "cuda"
                    ):
                        raise DenseCudaRuntimeError(
                            "native CUDA sampling logits must remain on the CUDA device"
                        )
                    tokens = _sample_torch_rows(
                        logits,
                        work.output.sampling,
                        semantic_token_count=self._semantic_token_count,
                    )
                token_counts = tuple(len(row) for row in work.token_rows)
                authority = DenseCudaProvisionalAuthority(
                    runtime_id=self._route.runtime_id,
                    step_id=step_id,
                    state=state,
                    result=result,
                )
                step = ProvisionalStep(
                    runtime_id=self._route.runtime_id,
                    step_id=step_id,
                    request_ids=work.request_ids,
                    state=state,
                    parent=current,
                    token_counts=token_counts,
                    output=NativeOutput(
                        mode=work.output.mode,
                        token_ids=tokens,
                    ),
                    authority=authority,
                )
            except BaseException:
                state._pending_step_id = None  # noqa: SLF001
                raise
        with self._lock:
            self._provisional_steps += 1
            self._device_to_host_bytes += len(tokens) * 8
            head_workspace_bytes = int(
                getattr(
                    getattr(self._engine, "target", None),
                    "reranked_head_working_bytes_peak",
                    0,
                )
            )
            self._workspace_peak_bytes = max(
                self._workspace_peak_bytes,
                int(getattr(result, "kv_delta_bytes", 0)) + head_workspace_bytes,
            )
            count = sum(token_counts)
            if phase == "prefill":
                self._prefill_calls += 1
                self._prefill_tokens += count
                self._prefill_seconds += elapsed
            else:
                self._decode_calls += 1
                self._decode_tokens += count
                self._decode_seconds += elapsed
        return step

    def prefill(self, work: PrefillWork) -> ProvisionalStep:
        if not isinstance(work, PrefillWork):
            raise TypeError("prefill requires PrefillWork")
        return self._execute(work, phase="prefill")

    def decode(self, work: DecodeWork) -> ProvisionalStep:
        if not isinstance(work, DecodeWork):
            raise TypeError("decode requires DecodeWork")
        return self._execute(work, phase="decode")

    def _authority(
        self,
        step: ProvisionalStep,
    ) -> DenseCudaProvisionalAuthority | DenseCudaBatchProvisionalAuthority:
        if not isinstance(step, ProvisionalStep):
            raise TypeError("terminal operation requires ProvisionalStep")
        if step.runtime_id != self._route.runtime_id:
            raise DenseCudaRuntimeError("provisional step belongs to another runtime")
        authority = step.authority
        if not isinstance(
            authority,
            (DenseCudaProvisionalAuthority, DenseCudaBatchProvisionalAuthority),
        ):
            raise TypeError("provisional authority was not issued by this runtime")
        if authority._consumed:  # noqa: SLF001
            raise DenseCudaRuntimeError("provisional authority has already been consumed")
        if authority._state is not step.state:  # noqa: SLF001
            raise DenseCudaRuntimeError("provisional authority and step bind different state")
        return authority

    def commit(
        self,
        step: ProvisionalStep,
        accepted_counts: Sequence[int],
    ) -> CommitResult:
        authority = self._authority(step)
        state = self._state(step.state)
        counts = _strict_counts(accepted_counts, expected=step.parent.batch_size)
        if any(
            count < 0 or count > available
            for count, available in zip(counts, step.token_counts, strict=True)
        ):
            raise ValueError("accepted count lies outside the provisional token block")
        with state._lock:  # noqa: SLF001
            before = state._observe_unlocked()  # noqa: SLF001
            if before != step.parent:
                raise DenseCudaRuntimeError("state changed after provisional execution")
            if state._pending_step_id != step.step_id:  # noqa: SLF001
                raise DenseCudaRuntimeError("state does not own this pending step")
            if isinstance(authority, DenseCudaBatchProvisionalAuthority):
                scratch = authority._scratch  # noqa: SLF001
                if scratch is None:
                    raise DenseCudaRuntimeError("batch authority lost its shared CUDA scratch")
                stats = scratch.install_row(
                    state._cache,  # noqa: SLF001
                    row=authority._row,  # noqa: SLF001
                    parent=step.parent,
                    accepted_count=counts[0],
                )
            else:
                result = authority._result  # noqa: SLF001
                if result is None:
                    raise DenseCudaRuntimeError("provisional authority lost its execution result")
                stats = self._engine.commit_block(state._cache, result, counts)  # noqa: SLF001
            after = state._observe_unlocked()  # noqa: SLF001
            # The authority is the sole runtime owner of provisional device scratch.  Detach the
            # result only after commit and its postcondition both succeed so a retained, consumed
            # ProvisionalStep cannot pin the K/V delta or hidden/output tensors indefinitely.
            if isinstance(authority, DenseCudaBatchProvisionalAuthority):
                scratch.consume(
                    authority._row,  # noqa: SLF001
                    committed=True,
                    accepted_count=counts[0],
                )
                authority._scratch = None  # noqa: SLF001
            else:
                authority._result = None  # noqa: SLF001
            authority._consumed = True  # noqa: SLF001
            state._pending_step_id = None  # noqa: SLF001
        receipt = CommitResult(
            runtime_id=self._route.runtime_id,
            step_id=step.step_id,
            state_id=state.state_id,
            accepted_counts=counts,
            before=before,
            after=after,
            state_bytes_written=int(stats.kv_write_bytes),
        )
        with self._lock:
            self._commits += 1
            self._committed_tokens += sum(counts)
        return receipt

    def abandon(self, step: ProvisionalStep) -> None:
        authority = self._authority(step)
        state = self._state(step.state)
        with state._lock:  # noqa: SLF001
            if state._observe_unlocked() != step.parent:  # noqa: SLF001
                raise DenseCudaRuntimeError("state changed after provisional execution")
            if state._pending_step_id != step.step_id:  # noqa: SLF001
                raise DenseCudaRuntimeError("state does not own this pending step")
            # Abandon has no state mutation to preserve.  Drop provisional device scratch while
            # holding the state transaction lock, before publishing the terminal authority state.
            if isinstance(authority, DenseCudaBatchProvisionalAuthority):
                scratch = authority._scratch  # noqa: SLF001
                if scratch is None:
                    raise DenseCudaRuntimeError("batch authority lost its shared CUDA scratch")
                scratch.consume(authority._row, committed=False)  # noqa: SLF001
                authority._scratch = None  # noqa: SLF001
            else:
                authority._result = None  # noqa: SLF001
            authority._consumed = True  # noqa: SLF001
            state._pending_step_id = None  # noqa: SLF001
        with self._lock:
            self._abandons += 1

    def release_state(self, state: Any) -> None:
        resolved = self._state(state)
        with self._lock, resolved._lock:  # noqa: SLF001
            if resolved._pending_step_id is not None:  # noqa: SLF001
                raise DenseCudaRuntimeError("cannot release state with a pending provisional step")
            if self._states.pop(resolved.state_id, None) is not resolved:
                raise DenseCudaRuntimeError("state was already released")
            cache = resolved._cache  # noqa: SLF001
            resolved._released = True  # noqa: SLF001
            resolved._cache = None  # noqa: SLF001
            self._forget_engine_cache(cache)
            if self._kv_slot_pool is not None:
                self._kv_slot_pool.release(cache)

    def telemetry(self) -> RuntimeTelemetry:
        with self._lock:
            self._require_open()
            kv_resident = (
                self._kv_slot_pool.allocated_bytes
                if self._kv_slot_pool is not None
                else sum(
                    int(getattr(state._cache, "allocated_bytes", 0))  # noqa: SLF001
                    for state in self._states.values()
                )
            )
            return RuntimeTelemetry(
                runtime_id=self._route.runtime_id,
                route_backend_id=self._route.backend_id,
                model_fingerprint=self._route.model_fingerprint,
                placement_fingerprint=self._route.placement_fingerprint,
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
                device_to_host_bytes=self._device_to_host_bytes,
                model_resident_bytes=self._placement.model_resident_bytes,
                kv_resident_bytes=kv_resident,
                workspace_peak_bytes=self._workspace_peak_bytes,
                extra_counters=(
                    ("admitted_body_workspace_bytes", self._admitted_body_workspace_bytes),
                    (
                        "kv_slot_pool_active_slots",
                        0 if self._kv_slot_pool is None else self._kv_slot_pool.active_slots,
                    ),
                    (
                        "kv_slot_pool_capacity_slots",
                        0 if self._kv_slot_pool is None else self._kv_slot_pool.max_slots,
                    ),
                    ("live_states", len(self._states)),
                    ("state_fork_bytes", self._state_fork_bytes),
                    ("state_fork_tokens", self._state_fork_tokens),
                    ("state_forks", self._state_forks),
                ),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            pending: list[str] = []
            for state in self._states.values():
                with state._lock:  # noqa: SLF001
                    if state._pending_step_id is not None:  # noqa: SLF001
                        pending.append(state.state_id)
            if pending:
                raise DenseCudaRuntimeError(
                    f"cannot close runtime with pending provisional states: {pending!r}"
                )
            states = tuple(self._states.values())
            self._states.clear()
            for state in states:
                with state._lock:  # noqa: SLF001
                    cache = state._cache  # noqa: SLF001
                    state._released = True  # noqa: SLF001
                    state._cache = None  # noqa: SLF001
                    self._forget_engine_cache(cache)
                    if self._kv_slot_pool is not None:
                        self._kv_slot_pool.release(cache)
            if self._kv_slot_pool is not None:
                self._kv_slot_pool.close()
            self._closed = True
        if self._owns_engine:
            self._engine.close()


__all__ = [
    "DenseCudaBatchProvisionalAuthority",
    "DenseCudaIndexedKVCache",
    "DenseCudaKVSlotCache",
    "DenseCudaKVSlotPool",
    "DenseCudaNativeRuntime",
    "DenseCudaProvisionalAuthority",
    "DenseCudaRuntimeError",
    "DenseCudaState",
]
