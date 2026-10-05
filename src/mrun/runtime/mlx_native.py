"""Fixed-arena transactional runtime for native MLX component executors.

MLX's ordinary ``KVCache`` mutates as it decodes and grows storage in 256-token chunks.  The
adapter below discovers the exact native K/V geometry once, allocates fixed per-request arenas,
and treats the suffix beyond the committed offset as provisional scratch.  Commit advances the
logical offset; abandon trims the suffix.  No model weights, logits, or K/V arrays cross through
Torch or NumPy on the hot path—only selected token IDs are returned to the coordinator.

The exact default state lane is deliberately B1.  Compatible-request packing and chunked prefill
are separately identified, opt-in bounded-numerical shapes.  Neither is promotion-equivalent to
B1 until cross-entropy, task, and generated-trajectory gates pass.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

import numpy as np

from .contracts import (
    BackendCapabilities,
    CommitResult,
    CompatibleBatchLaneIdentity,
    CompiledModelIdentity,
    DecodeWork,
    DeviceDescriptor,
    GreedyBlockVerification,
    GreedyBlockVerifyWork,
    GreedyProposalBeginWork,
    GreedyProposalTransaction,
    NativeOutput,
    OutputMode,
    OutputRequest,
    PlacementPlan,
    PrefillWork,
    PromotionStatus,
    ProvisionalStep,
    RuntimeRoute,
    RuntimeTelemetry,
    SamplingRequest,
    StateForkResult,
    StateObservation,
    WorkloadSpec,
)
from .mlx_paged_attention import MlxPagedDecodeAttentionLane
from .mlx_paged_kv import MLX_PAGED_KV_CACHE_ABI
from .placement import validate_placement_plan
from .sampling import sampling_adjustments, stateless_uniform, validate_sampling_domain


class MlxNativeRuntimeError(RuntimeError):
    """The MLX runtime rejected stale, foreign, or physically inconsistent work."""


MLX_COMPATIBLE_BATCH_ABI = "mrun-mlx-compatible-cow-batch-v1"
MLX_COMPATIBLE_BATCH_NUMERICAL_CONTRACT = "mlx-lm-packed-compatible-cow-bounded-numerical-v1"
MLX_COMPATIBLE_CHUNKED_PREFILL_NUMERICAL_CONTRACT = (
    "mlx-lm-packed-compatible-cow-chunked-prefill-bounded-numerical-v1"
)
MLX_PREFILL_EXECUTION_SHAPE_ABI = "mrun-mlx-transactional-prefill-shape-v1"
MLX_CHUNKED_PREFILL_NUMERICAL_CONTRACT = "mlx-lm-transactional-chunked-prefill-bounded-numerical-v1"
MLX_GREEDY_BLOCK_ABI = "mrun-mlx-transactional-greedy-block-v1"
MLX_GREEDY_PROPOSAL_ABI = "mrun-mlx-in-place-greedy-proposal-v1"


@dataclass(frozen=True, slots=True)
class MlxPrefillExecutionShape:
    """Declared numerical identity for optional long-prompt execution chunking."""

    chunk_size: int | None
    base_numerical_contract: str
    numerical_contract: str
    promotion_status: PromotionStatus
    execution_abi: str = MLX_PREFILL_EXECUTION_SHAPE_ABI

    def __post_init__(self) -> None:
        if self.execution_abi != MLX_PREFILL_EXECUTION_SHAPE_ABI:
            raise ValueError("unsupported MLX prefill execution-shape ABI")
        if self.chunk_size is not None and (
            isinstance(self.chunk_size, bool)
            or not isinstance(self.chunk_size, int)
            or self.chunk_size <= 0
        ):
            raise ValueError("MLX prefill chunk_size must be a positive integer or None")
        _name(self.base_numerical_contract, "base_numerical_contract")
        _name(self.numerical_contract, "numerical_contract")
        if not isinstance(self.promotion_status, PromotionStatus):
            raise TypeError("prefill execution-shape promotion_status must be PromotionStatus")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "execution_abi": self.execution_abi,
                "chunk_size": self.chunk_size,
                "base_numerical_contract": self.base_numerical_contract,
                "numerical_contract": self.numerical_contract,
                "promotion_status": self.promotion_status.value,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _sample_mlx_row(
    mx: Any,
    logits: Any,
    request: SamplingRequest,
    *,
    semantic_token_count: int,
) -> Any:
    """Select one token while keeping the vocabulary vector in unified GPU memory."""

    validate_sampling_domain(request, semantic_token_count)
    if int(getattr(logits, "ndim", -1)) != 1 or int(logits.shape[0]) < semantic_token_count:
        raise MlxNativeRuntimeError("native MLX sampling requires one complete logit vector")
    scores = logits[:semantic_token_count].astype(mx.float32)
    adjustments = sampling_adjustments(request)
    if adjustments:
        indices = mx.array([token for token, _ in adjustments], dtype=mx.int32)
        values = mx.array([value for _, value in adjustments], dtype=mx.float32)
        scores = scores.at[indices].add(values)

    finite = mx.all(mx.isfinite(scores))
    policy = request.policy
    if policy.temperature == 0.0:
        selected = mx.argmax(scores)
    else:
        scaled = (scores - mx.max(scores)) / policy.temperature
        filtered = scaled
        ordered_tokens = None
        if policy.top_k or policy.top_p < 1.0:
            order = mx.argsort(-scaled)
            ordered = scaled[order]
            if policy.top_k:
                ranks = mx.arange(semantic_token_count)
                ordered = mx.where(
                    ranks < policy.top_k,
                    ordered,
                    mx.full_like(ordered, float("-inf")),
                )
            base_probabilities = mx.softmax(ordered)
            if policy.top_p < 1.0:
                cumulative = mx.cumsum(base_probabilities)
                keep = cumulative - base_probabilities < policy.top_p
                ordered = mx.where(
                    keep,
                    ordered,
                    mx.full_like(ordered, float("-inf")),
                )
            filtered = ordered
            ordered_tokens = order
        probabilities = mx.softmax(filtered)
        cumulative = mx.cumsum(probabilities)
        threshold = mx.array(
            stateless_uniform(policy.seed, request.rng_counter),
            dtype=mx.float32,
        )
        selected_position = mx.minimum(
            mx.sum(cumulative < threshold),
            mx.array(semantic_token_count - 1),
        )
        selected = (
            selected_position if ordered_tokens is None else ordered_tokens[selected_position]
        )
        finite = finite & mx.all(mx.isfinite(probabilities)) & (cumulative[-1] > 0)
    sentinel = mx.array(semantic_token_count, dtype=mx.int64)
    return mx.where(finite, selected.astype(mx.int64), sentinel)


@dataclass(frozen=True, slots=True)
class MlxLayerCacheSpec:
    kv_heads: int
    key_head_dim: int
    value_head_dim: int
    key_dtype: Any
    value_dtype: Any
    key_element_bytes: int
    value_element_bytes: int

    @property
    def bytes_per_token(self) -> int:
        return self.kv_heads * (
            self.key_head_dim * self.key_element_bytes
            + self.value_head_dim * self.value_element_bytes
        )


@dataclass(frozen=True, slots=True)
class MlxStateLayout:
    layers: tuple[MlxLayerCacheSpec, ...]
    dtype_name: str
    bytes_per_token: int


def inspect_mlx_state_layout(engine: Any) -> MlxStateLayout:
    """Measure native post-projection K/V geometry before placement is admitted."""

    try:
        from mlx_lm.models.cache import KVCache
    except ImportError as exc:
        raise RuntimeError("MLX native runtime requires mlx-lm") from exc
    mx = engine._mx
    layer_count = int(engine.n_layer)
    probe = [KVCache() for _ in range(layer_count)]
    body = getattr(engine.model, "model", None)
    if body is None:
        raise MlxNativeRuntimeError("MLX model does not expose its decoder body")
    hidden = body(mx.array([[0]]), cache=probe)
    mx.eval(hidden, *[value for cache in probe for value in (cache.keys, cache.values)])
    layers: list[MlxLayerCacheSpec] = []
    dtype_names: set[str] = set()
    for cache in probe:
        key_elements = int(np.prod(cache.keys.shape, dtype=np.int64))
        value_elements = int(np.prod(cache.values.shape, dtype=np.int64))
        key_element_bytes = int(cache.keys.nbytes) // key_elements
        value_element_bytes = int(cache.values.nbytes) // value_elements
        key_name = str(cache.keys.dtype).rsplit(".", maxsplit=1)[-1]
        value_name = str(cache.values.dtype).rsplit(".", maxsplit=1)[-1]
        dtype_names.update((key_name, value_name))
        layers.append(
            MlxLayerCacheSpec(
                kv_heads=int(cache.keys.shape[1]),
                key_head_dim=int(cache.keys.shape[3]),
                value_head_dim=int(cache.values.shape[3]),
                key_dtype=cache.keys.dtype,
                value_dtype=cache.values.dtype,
                key_element_bytes=key_element_bytes,
                value_element_bytes=value_element_bytes,
            )
        )
    if len(layers) != layer_count:
        raise MlxNativeRuntimeError("MLX cache probe returned the wrong layer count")
    dtype_name = next(iter(dtype_names)) if len(dtype_names) == 1 else "+".join(sorted(dtype_names))
    return MlxStateLayout(
        layers=tuple(layers),
        dtype_name=dtype_name,
        bytes_per_token=sum(layer.bytes_per_token for layer in layers),
    )


def _name(value: str, field_name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _strict_single_count(values: Sequence[int], *, maximum: int) -> int:
    counts = tuple(values)
    if len(counts) != 1:
        raise ValueError("accepted_counts must contain exactly one B1 value")
    value = counts[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("accepted_counts must contain strict integers")
    if value < 0 or value > maximum:
        raise ValueError("accepted count lies outside the provisional token block")
    return int(value)


def _cache_signature(caches: Sequence[Any]) -> tuple[Any, ...]:
    signatures: list[Any] = []
    for cache in caches:
        custom = getattr(cache, "storage_signature", None)
        if callable(custom):
            signatures.append(custom())
            continue
        keys = getattr(cache, "keys", None)
        values = getattr(cache, "values", None)
        if keys is None or values is None:
            raise MlxNativeRuntimeError("MLX cache has no fixed K/V backing arrays")
        signatures.append(
            (
                id(keys),
                tuple(int(value) for value in keys.shape),
                str(keys.dtype),
                id(values),
                tuple(int(value) for value in values.shape),
                str(values.dtype),
            )
        )
    return tuple(signatures)


def _cache_bytes(caches: Sequence[Any]) -> int:
    total = 0
    for cache in caches:
        value = getattr(cache, "nbytes", 0)
        total += int(value() if callable(value) else value)
    return total


def _cache_evaluation_arrays(caches: Sequence[Any]) -> tuple[Any, ...]:
    """Return realization dependencies without forcing optional dense cache views."""

    arrays: list[Any] = []
    for cache in caches:
        custom = getattr(cache, "evaluation_arrays", None)
        if callable(custom):
            values = tuple(custom())
            if not values:
                raise MlxNativeRuntimeError("MLX cache returned no evaluation dependencies")
            arrays.extend(values)
        else:
            arrays.extend((cache.keys, cache.values))
    return tuple(arrays)


class FixedMlxKVCache:
    """mlx-lm compatible, exact global-attention cache with fixed backing storage."""

    __slots__ = (
        "_capacity",
        "_create_attention_mask",
        "_mx",
        "keys",
        "offset",
        "values",
    )

    def __init__(
        self,
        *,
        mx: Any,
        create_attention_mask: Callable[..., Any],
        capacity: int,
        kv_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        key_dtype: Any,
        value_dtype: Any,
    ) -> None:
        self._capacity = int(capacity)
        self._create_attention_mask = create_attention_mask
        self._mx = mx
        self.keys = mx.zeros((1, kv_heads, capacity, key_head_dim), dtype=key_dtype)
        self.values = mx.zeros((1, kv_heads, capacity, value_head_dim), dtype=value_dtype)
        self.offset = 0

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        token_count = int(keys.shape[2])
        if token_count <= 0 or self.offset + token_count > self._capacity:
            raise OverflowError("native MLX K/V append exceeds the fixed arena")
        if (
            int(keys.shape[0]) != 1
            or tuple(keys.shape[:2]) != tuple(self.keys.shape[:2])
            or int(keys.shape[3]) != int(self.keys.shape[3])
            or tuple(values.shape[:2]) != tuple(self.values.shape[:2])
            or int(values.shape[3]) != int(self.values.shape[3])
            or keys.dtype != self.keys.dtype
            or values.dtype != self.values.dtype
        ):
            raise MlxNativeRuntimeError("model K/V output differs from the fixed arena geometry")
        start = self.offset
        self.offset += token_count
        self.keys[..., start : self.offset, :] = keys
        self.values[..., start : self.offset, :] = values
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    def make_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._create_attention_mask(*args, offset=self.offset, **kwargs)

    def trim(self, count: int) -> int:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("trim count must be a non-negative integer")
        trimmed = min(self.offset, count)
        self.offset -= trimmed
        return trimmed

    def reset(self, offset: int) -> None:
        if offset < 0 or offset > self._capacity:
            raise ValueError("reset offset lies outside the fixed arena")
        self.offset = int(offset)

    def copy_committed_prefix_from(
        self,
        source: FixedMlxKVCache,
        length: int,
    ) -> None:
        """Copy one exact committed layer prefix without materializing it on the host."""

        if not isinstance(source, FixedMlxKVCache):
            raise TypeError("MLX cache fork source must use the fixed native cache ABI")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("MLX cache fork length must be a non-negative integer")
        if self.offset != 0:
            raise MlxNativeRuntimeError("MLX cache fork target must be empty")
        if self._mx is not source._mx:
            raise MlxNativeRuntimeError("MLX cache fork crosses an execution context boundary")
        if length > source.offset:
            raise MlxNativeRuntimeError("MLX cache fork exceeds the committed source prefix")
        if length > self._capacity:
            raise OverflowError("MLX cache fork exceeds the target capacity")
        if (
            tuple(self.keys.shape[:2]) != tuple(source.keys.shape[:2])
            or int(self.keys.shape[3]) != int(source.keys.shape[3])
            or tuple(self.values.shape[:2]) != tuple(source.values.shape[:2])
            or int(self.values.shape[3]) != int(source.values.shape[3])
            or self.keys.dtype != source.keys.dtype
            or self.values.dtype != source.values.dtype
        ):
            raise MlxNativeRuntimeError("MLX cache fork crosses a K/V layout boundary")
        if self.keys is source.keys or self.values is source.values:
            raise MlxNativeRuntimeError("MLX cache fork target aliases source backing storage")
        if length:
            self.keys[..., :length, :] = source.keys[..., :length, :]
            self.values[..., :length, :] = source.values[..., :length, :]
            self._mx.eval(self.keys, self.values)
        self.offset = length

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return True

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def state(self) -> tuple[Any, Any]:
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)

    def storage_signature(self) -> tuple[Any, ...]:
        return (
            id(self.keys),
            tuple(int(value) for value in self.keys.shape),
            str(self.keys.dtype),
            id(self.values),
            tuple(int(value) for value in self.values.shape),
            str(self.values.dtype),
        )

    def validate_row_copy_from_batch(
        self,
        source: FixedMlxBatchKVCache,
        *,
        row: int,
        start: int,
        count: int,
    ) -> None:
        """Validate a COW suffix installation without changing logical or physical state."""

        if not isinstance(source, FixedMlxBatchKVCache):
            raise TypeError("MLX batch commit source must use the fixed batch-cache ABI")
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in (row, start, count)
        ):
            raise TypeError("MLX batch commit row/start/count must be strict integers")
        if row < 0 or row >= source.batch_size or start < 0 or count < 0:
            raise ValueError("MLX batch commit slice lies outside the source arena")
        if self._mx is not source._mx:
            raise MlxNativeRuntimeError("MLX batch commit crosses an execution context boundary")
        if self.offset != start or start + count > self._capacity:
            raise MlxNativeRuntimeError("MLX batch commit does not extend the committed prefix")
        if start + count > source.offset:
            raise MlxNativeRuntimeError("MLX batch commit exceeds the provisional source suffix")
        if (
            tuple(self.keys.shape[1:2]) != tuple(source.keys.shape[1:2])
            or int(self.keys.shape[3]) != int(source.keys.shape[3])
            or tuple(self.values.shape[1:2]) != tuple(source.values.shape[1:2])
            or int(self.values.shape[3]) != int(source.values.shape[3])
            or self.keys.dtype != source.keys.dtype
            or self.values.dtype != source.values.dtype
        ):
            raise MlxNativeRuntimeError("MLX batch commit crosses a K/V layout boundary")

    def write_row_from_batch(
        self,
        source: FixedMlxBatchKVCache,
        *,
        row: int,
        start: int,
        count: int,
    ) -> None:
        """Write an already validated suffix without advancing the visible cache offset."""

        self.validate_row_copy_from_batch(source, row=row, start=start, count=count)
        if count:
            stop = start + count
            self.keys[..., start:stop, :] = source.keys[row : row + 1, ..., start:stop, :]
            self.values[..., start:stop, :] = source.values[row : row + 1, ..., start:stop, :]


class FixedMlxBatchKVCache:
    """Ephemeral dense MLX cache used only by the opt-in compatible-request lane.

    The batch arena owns a copy of every committed prefix.  Model execution appends provisional
    K/V here, never to any request's authoritative B1 arena.
    """

    __slots__ = (
        "_capacity",
        "_create_attention_mask",
        "_mx",
        "batch_size",
        "keys",
        "offset",
        "values",
    )

    def __init__(
        self,
        *,
        mx: Any,
        create_attention_mask: Callable[..., Any],
        capacity: int,
        batch_size: int,
        kv_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        key_dtype: Any,
        value_dtype: Any,
    ) -> None:
        self._capacity = int(capacity)
        self._create_attention_mask = create_attention_mask
        self._mx = mx
        self.batch_size = int(batch_size)
        if self._capacity <= 0 or self.batch_size <= 1:
            raise ValueError("fixed MLX batch cache requires positive capacity and batch_size > 1")
        self.keys = mx.zeros(
            (self.batch_size, kv_heads, self._capacity, key_head_dim),
            dtype=key_dtype,
        )
        self.values = mx.zeros(
            (self.batch_size, kv_heads, self._capacity, value_head_dim),
            dtype=value_dtype,
        )
        self.offset = 0

    def copy_committed_row_from(
        self,
        source: FixedMlxKVCache,
        *,
        row: int,
        length: int,
    ) -> None:
        if not isinstance(source, FixedMlxKVCache):
            raise TypeError("MLX batch prefix source must use the fixed B1 cache ABI")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (row, length)):
            raise TypeError("MLX batch prefix row/length must be strict integers")
        if row < 0 or row >= self.batch_size or length < 0:
            raise ValueError("MLX batch prefix slice lies outside the target arena")
        if self._mx is not source._mx:
            raise MlxNativeRuntimeError("MLX batch prefix copy crosses an execution context")
        if length > source.offset or length > self._capacity:
            raise OverflowError("MLX batch prefix exceeds its source or target capacity")
        if (
            tuple(self.keys.shape[1:2]) != tuple(source.keys.shape[1:2])
            or int(self.keys.shape[3]) != int(source.keys.shape[3])
            or tuple(self.values.shape[1:2]) != tuple(source.values.shape[1:2])
            or int(self.values.shape[3]) != int(source.values.shape[3])
            or self.keys.dtype != source.keys.dtype
            or self.values.dtype != source.values.dtype
        ):
            raise MlxNativeRuntimeError("MLX batch prefix copy crosses a K/V layout boundary")
        if length:
            self.keys[row : row + 1, ..., :length, :] = source.keys[..., :length, :]
            self.values[row : row + 1, ..., :length, :] = source.values[..., :length, :]

    def seal_prefix(self, length: int) -> None:
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("MLX batch prefix length must be a non-negative integer")
        if self.offset != 0 or length > self._capacity:
            raise MlxNativeRuntimeError("MLX batch prefix can only be sealed once")
        self.offset = length

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        token_count = int(keys.shape[2])
        if token_count <= 0 or self.offset + token_count > self._capacity:
            raise OverflowError("native MLX batch K/V append exceeds the ephemeral arena")
        if (
            int(keys.shape[0]) != self.batch_size
            or tuple(keys.shape[:2]) != tuple(self.keys.shape[:2])
            or int(keys.shape[3]) != int(self.keys.shape[3])
            or tuple(values.shape[:2]) != tuple(self.values.shape[:2])
            or int(values.shape[3]) != int(self.values.shape[3])
            or keys.dtype != self.keys.dtype
            or values.dtype != self.values.dtype
        ):
            raise MlxNativeRuntimeError("model K/V output differs from batch arena geometry")
        start = self.offset
        self.offset += token_count
        self.keys[..., start : self.offset, :] = keys
        self.values[..., start : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def make_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._create_attention_mask(*args, offset=self.offset, **kwargs)

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return False

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def state(self) -> tuple[Any, Any]:
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)

    def storage_signature(self) -> tuple[Any, ...]:
        return (
            id(self.keys),
            tuple(int(value) for value in self.keys.shape),
            str(self.keys.dtype),
            id(self.values),
            tuple(int(value) for value in self.values.shape),
            str(self.values.dtype),
        )


class MlxNativeState:
    """Opaque B1 authority over a list of fixed per-layer unified-memory arenas."""

    __slots__ = (
        "_caches",
        "_capacity",
        "_committed_length",
        "_epoch",
        "_generation",
        "_lock",
        "_owner_id",
        "_pending_count",
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
        caches: Sequence[Any],
    ) -> None:
        self._runtime_id = runtime_id
        self._state_id = state_id
        self._owner_id = owner_id
        self._state_abi = state_abi
        self._capacity = int(capacity)
        self._caches = tuple(caches)
        if not self._caches:
            raise MlxNativeRuntimeError("MLX state requires at least one layer cache")
        self._storage_signature = _cache_signature(self._caches)
        self._committed_length = 0
        self._epoch = 0
        self._generation = int(generation)
        self._storage_generation = 0
        self._pending_step_id: str | None = None
        self._pending_count = 0
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

    def _verify_storage_unlocked(self) -> None:
        if self._released:
            raise MlxNativeRuntimeError("state authority has been released")
        if _cache_signature(self._caches) != self._storage_signature:
            raise MlxNativeRuntimeError("state K/V backing storage identity changed")
        expected_offset = self._committed_length + self._pending_count
        if any(int(cache.offset) != expected_offset for cache in self._caches):
            raise MlxNativeRuntimeError("layer cache offsets diverged from logical state")

    def _observe_unlocked(self) -> StateObservation:
        self._verify_storage_unlocked()
        return StateObservation(
            runtime_id=self._runtime_id,
            state_id=self._state_id,
            generation=self._generation,
            epoch=self._epoch,
            lengths=(self._committed_length,),
            capacity=self._capacity,
            state_abi=self._state_abi,
            storage_generation=self._storage_generation,
        )

    def observe(self) -> StateObservation:
        with self._lock:
            return self._observe_unlocked()


class MlxProvisionalAuthority:
    __slots__ = ("_consumed", "_runtime_id", "_state", "_step_id")

    def __init__(self, *, runtime_id: str, step_id: str, state: MlxNativeState) -> None:
        self._runtime_id = runtime_id
        self._step_id = step_id
        self._state = state
        self._consumed = False

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def step_id(self) -> str:
        return self._step_id


class MlxGreedyBlockAuthority(MlxProvisionalAuthority):
    """Exact terminal authority for one all-position greedy verification suffix."""

    __slots__ = ()


class MlxGreedyProposalAuthority(MlxProvisionalAuthority):
    """Versioned authority over one in-place, multi-forward draft suffix."""

    __slots__ = (
        "_input_token_ids",
        "_predicted_token_ids",
        "_sealed",
        "_version",
    )

    def __init__(self, *, runtime_id: str, step_id: str, state: MlxNativeState) -> None:
        super().__init__(runtime_id=runtime_id, step_id=step_id, state=state)
        self._input_token_ids: tuple[int, ...] = ()
        self._predicted_token_ids: tuple[int, ...] = ()
        self._sealed = False
        self._version = 0


class _MlxBatchScratch:
    """Shared lifetime for one ephemeral merged cache and its independent child rows."""

    __slots__ = (
        "_active_rows",
        "_caches",
        "_lock",
        "_on_consume",
        "_on_release",
        "scratch_id",
    )

    def __init__(
        self,
        *,
        caches: Sequence[Any],
        row_count: int,
        on_consume: Callable[[bool, int], None],
        on_release: Callable[[], None],
    ) -> None:
        self.scratch_id = f"scratch-{uuid4().hex}"
        self._caches = tuple(caches)
        self._active_rows = set(range(row_count))
        self._on_consume = on_consume
        self._on_release = on_release
        self._lock = threading.Lock()

    @property
    def caches(self) -> tuple[Any, ...]:
        with self._lock:
            if not self._caches:
                raise MlxNativeRuntimeError("MLX compatible-batch scratch was already released")
            return self._caches

    def install_row(
        self,
        targets: Sequence[Any],
        *,
        row: int,
        start: int,
        count: int,
    ) -> None:
        with self._lock:
            if row not in self._active_rows or not self._caches:
                raise MlxNativeRuntimeError("MLX compatible-batch row is no longer provisional")
            if len(targets) != len(self._caches):
                raise MlxNativeRuntimeError("MLX compatible-batch layer count drifted")
            for target, source in zip(targets, self._caches, strict=True):
                validate = getattr(target, "validate_row_copy_from_batch", None)
                write = getattr(target, "write_row_from_batch", None)
                if not callable(validate) or not callable(write):
                    raise MlxNativeRuntimeError(
                        "B1 cache lacks compatible-batch suffix installation support"
                    )
                validate(source, row=row, start=start, count=count)
            for target, source in zip(targets, self._caches, strict=True):
                target.write_row_from_batch(source, row=row, start=start, count=count)
            if count:
                evaluator = getattr(targets[0], "_mx", None)
                if evaluator is not None:
                    evaluator.eval(
                        *[value for target in targets for value in (target.keys, target.values)]
                    )
            for target in targets:
                reset = getattr(target, "reset", None)
                if not callable(reset):
                    raise MlxNativeRuntimeError("B1 cache lacks offset finalization support")
                reset(start + count)

    def consume(self, row: int, *, committed: bool, accepted_count: int = 0) -> None:
        released = False
        with self._lock:
            if row not in self._active_rows:
                raise MlxNativeRuntimeError("MLX compatible-batch row was already consumed")
            self._active_rows.remove(row)
            if not self._active_rows:
                self._caches = ()
                released = True
        self._on_consume(committed, accepted_count)
        if released:
            self._on_release()

    def discard(self) -> None:
        """Drop an unpublished scratch after result construction failed."""

        with self._lock:
            if not self._caches:
                return
            self._active_rows.clear()
            self._caches = ()
        self._on_release()


class MlxBatchProvisionalAuthority(MlxProvisionalAuthority):
    """Per-row terminal authority over one shared ephemeral MLX batch scratch."""

    __slots__ = ("_row", "_scratch")

    def __init__(
        self,
        *,
        runtime_id: str,
        step_id: str,
        state: MlxNativeState,
        scratch: _MlxBatchScratch,
        row: int,
    ) -> None:
        super().__init__(runtime_id=runtime_id, step_id=step_id, state=state)
        self._scratch = scratch
        self._row = int(row)


@dataclass(frozen=True, slots=True)
class MlxCompatibleBatchTelemetry:
    lane_id: str
    dispatches: int
    physical_forwards: int
    provisional_rows: int
    committed_rows: int
    abandoned_rows: int
    failed_dispatches: int
    singleton_bypasses: int
    scratch_limited_bypasses: int
    max_width: int
    width_histogram: tuple[tuple[int, int], ...]
    prefix_bytes_copied: int
    suffix_bytes_committed: int
    scratch_peak_bytes: int
    active_scratches: int

    def __post_init__(self) -> None:
        _name(self.lane_id, "lane_id")
        for field_name in (
            "dispatches",
            "physical_forwards",
            "provisional_rows",
            "committed_rows",
            "abandoned_rows",
            "failed_dispatches",
            "singleton_bypasses",
            "scratch_limited_bypasses",
            "max_width",
            "prefix_bytes_copied",
            "suffix_bytes_committed",
            "scratch_peak_bytes",
            "active_scratches",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        histogram = tuple(self.width_histogram)
        if any(
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for width, count in histogram
        ):
            raise ValueError("width_histogram must contain positive integer pairs")
        if tuple(sorted(histogram)) != histogram or len({width for width, _ in histogram}) != len(
            histogram
        ):
            raise ValueError("width_histogram must be sorted with unique widths")


class MlxNativeRuntime:
    """Backend-neutral transactional B1 runtime over an MLX component engine."""

    def __init__(
        self,
        engine: Any,
        *,
        route: RuntimeRoute,
        placement: PlacementPlan,
        semantic_token_count: int,
        state_abi: str,
        cache_factory: Callable[[int], Sequence[Any]] | None = None,
        executor: Callable[[tuple[int, ...], Sequence[Any]], int] | None = None,
        greedy_block_executor: (
            Callable[[tuple[int, ...], Sequence[Any]], Sequence[int]] | None
        ) = None,
        greedy_proposal_executor: (Callable[[tuple[int, ...], Sequence[Any]], int] | None) = None,
        greedy_proposal_seal_executor: (
            Callable[[tuple[int, ...], Sequence[Any]], None] | None
        ) = None,
        sampling_executor: (
            Callable[[tuple[int, ...], Sequence[Any], SamplingRequest], int] | None
        ) = None,
        prefill_chunk_size: int | None = None,
        prefill_chunk_executor: (
            Callable[
                [tuple[int, ...], Sequence[Any], OutputRequest | None],
                int | None,
            ]
            | None
        ) = None,
        state_layout: MlxStateLayout | None = None,
        paged_decode_attention_lane: MlxPagedDecodeAttentionLane | None = None,
        owns_cache_factory: bool = False,
        owns_engine: bool = False,
    ) -> None:
        if not isinstance(route, RuntimeRoute) or not isinstance(placement, PlacementPlan):
            raise TypeError("MLX runtime requires canonical route and placement values")
        if route.placement_fingerprint != placement.fingerprint:
            raise ValueError("runtime route does not bind the supplied placement")
        if route.model_fingerprint != placement.model_fingerprint:
            raise ValueError("runtime route and placement bind different models")
        if route.backend_id != placement.backend_id or route.device_id != placement.device_id:
            raise ValueError("runtime route and placement bind different backend/device")
        if state_abi != placement.state.state_abi:
            raise ValueError("runtime state ABI differs from placement")
        if isinstance(semantic_token_count, bool) or not isinstance(semantic_token_count, int):
            raise TypeError("semantic_token_count must be an integer")
        if semantic_token_count <= 0:
            raise ValueError("semantic_token_count must be positive")
        if str(getattr(engine, "backend", "")) != route.backend_id:
            raise ValueError("engine backend differs from the runtime route")
        engine_limit = int(getattr(engine, "context_size", getattr(engine, "max_seq_len", 0)))
        if engine_limit < placement.state.max_context_tokens:
            raise ValueError("engine context limit is smaller than the admitted placement")
        engine_token_count = int(getattr(engine, "semantic_token_count", semantic_token_count))
        if engine_token_count != semantic_token_count:
            raise ValueError("engine semantic token domain differs from the compiled model")
        if prefill_chunk_size is not None and (
            isinstance(prefill_chunk_size, bool)
            or not isinstance(prefill_chunk_size, int)
            or prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer or None")
        if (
            prefill_chunk_size is not None
            and prefill_chunk_size > placement.state.max_context_tokens
        ):
            raise ValueError("prefill_chunk_size exceeds the admitted context")
        if prefill_chunk_executor is not None and prefill_chunk_size is None:
            raise ValueError("prefill_chunk_executor requires an explicit prefill_chunk_size")
        if paged_decode_attention_lane is not None:
            if not isinstance(paged_decode_attention_lane, MlxPagedDecodeAttentionLane):
                raise TypeError("paged_decode_attention_lane has an unsupported implementation")
            if prefill_chunk_size is not None:
                raise ValueError(
                    "Metal paged decode and chunked prefill cannot share one execution identity"
                )
            if getattr(cache_factory, "cache_abi", None) != MLX_PAGED_KV_CACHE_ABI or not bool(
                getattr(cache_factory, "paged_decode_attention", False)
            ):
                raise ValueError(
                    "Metal paged decode requires its explicitly admitted BF16 page factory"
                )
            identity = paged_decode_attention_lane.identity
            if (
                getattr(cache_factory, "page_size", None) != identity.page_size
                or getattr(cache_factory, "page_count", None) != identity.page_count
                or len(tuple(getattr(cache_factory, "pools", ()))) != identity.layer_count
            ):
                raise ValueError(
                    "Metal paged-decode identity differs from its physical page-factory geometry"
                )
        if type(owns_cache_factory) is not bool:
            raise TypeError("owns_cache_factory must be boolean")
        if owns_cache_factory and (
            cache_factory is None or not callable(getattr(cache_factory, "close", None))
        ):
            raise ValueError("owned cache_factory must expose close()")

        self._engine = engine
        self._placement = placement
        self._semantic_token_count = semantic_token_count
        self._state_abi = state_abi
        self._state_layout = state_layout
        if cache_factory is None:
            self._state_layout = state_layout or inspect_mlx_state_layout(engine)
        if self._state_layout is not None:
            if self._state_layout.bytes_per_token != placement.state.bytes_per_token:
                raise ValueError(
                    "measured MLX state bytes/token differ from placement "
                    f"({self._state_layout.bytes_per_token} != "
                    f"{placement.state.bytes_per_token})"
                )
            if self._state_layout.dtype_name != placement.state.dtype:
                raise ValueError(
                    "measured MLX state dtype differs from placement "
                    f"({self._state_layout.dtype_name!r} != {placement.state.dtype!r})"
                )
        self._cache_factory = cache_factory or self._default_cache_factory
        self._owns_cache_factory = owns_cache_factory
        self._paged_decode_attention_lane = paged_decode_attention_lane
        self._executor = executor or self._default_executor
        self._greedy_block_executor = greedy_block_executor or self._default_greedy_block_executor
        self._greedy_proposal_executor = greedy_proposal_executor or self._executor
        self._greedy_proposal_seal_executor = (
            greedy_proposal_seal_executor or self._default_greedy_proposal_seal_executor
        )
        self._sampling_executor = sampling_executor or self._default_sampling_executor
        self._prefill_chunk_size = prefill_chunk_size
        self._prefill_chunk_executor = (
            prefill_chunk_executor or self._default_prefill_chunk_executor
        )
        base_numerical_contract = str(getattr(engine, "numerical_contract", ""))
        self._prefill_execution_shape = MlxPrefillExecutionShape(
            chunk_size=prefill_chunk_size,
            base_numerical_contract=base_numerical_contract,
            numerical_contract=(
                base_numerical_contract
                if prefill_chunk_size is None
                else MLX_CHUNKED_PREFILL_NUMERICAL_CONTRACT
            ),
            promotion_status=(
                route.promotion_status
                if prefill_chunk_size is None
                else PromotionStatus.EXPERIMENTAL
            ),
        )
        if paged_decode_attention_lane is not None:
            identity = paged_decode_attention_lane.identity
            if route.effective_numerical_contract is None:
                self._route = replace(
                    route,
                    promotion_status=PromotionStatus.EXPERIMENTAL,
                    effective_numerical_contract=identity.numerical_contract,
                    execution_shape_fingerprint=identity.fingerprint,
                )
            elif (
                route.promotion_status is not PromotionStatus.EXPERIMENTAL
                or route.effective_numerical_contract != identity.numerical_contract
                or route.execution_shape_fingerprint != identity.fingerprint
            ):
                raise ValueError(
                    "Metal paged-decode route does not bind its effective execution identity"
                )
            else:
                self._route = route
        elif prefill_chunk_size is None:
            self._route = route
        else:
            shape = self._prefill_execution_shape
            if route.effective_numerical_contract is None:
                self._route = replace(
                    route,
                    promotion_status=PromotionStatus.EXPERIMENTAL,
                    effective_numerical_contract=shape.numerical_contract,
                    execution_shape_fingerprint=shape.fingerprint,
                )
            elif (
                route.promotion_status is not PromotionStatus.EXPERIMENTAL
                or route.effective_numerical_contract != shape.numerical_contract
                or route.execution_shape_fingerprint != shape.fingerprint
            ):
                raise ValueError(
                    "chunked MLX prefill route does not bind its effective execution shape"
                )
            else:
                self._route = route
        self._owns_engine = bool(owns_engine)
        self._states: dict[str, MlxNativeState] = {}
        self._next_state_generation = 1
        self._lock = threading.RLock()
        self._chunk_lock = threading.Lock()
        self._closed = False
        self._prefill_calls = 0
        self._prefill_tokens = 0
        self._prefill_seconds = 0.0
        self._chunked_prefill_calls = 0
        self._prefill_chunks = 0
        self._prefill_chunk_failures = 0
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
        self._greedy_block_verifications = 0
        self._greedy_block_tokens = 0
        self._greedy_block_commits = 0
        self._greedy_block_abandons = 0
        self._greedy_block_accepted_tokens = 0
        self._greedy_block_rejected_tokens = 0
        self._greedy_block_failures = 0
        self._greedy_proposal_transactions = 0
        self._greedy_proposal_advances = 0
        self._greedy_proposal_seals = 0
        self._greedy_proposal_tokens = 0
        self._greedy_proposal_selected_tokens = 0
        self._greedy_proposal_commits = 0
        self._greedy_proposal_abandons = 0
        self._greedy_proposal_accepted_tokens = 0
        self._greedy_proposal_rejected_tokens = 0
        self._greedy_proposal_abandoned_tokens = 0
        self._greedy_proposal_failures = 0
        self._compatible_batch_lane: MlxCompatibleBatchLane | None = None

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
        cache_factory: Callable[[int], Sequence[Any]] | None = None,
        executor: Callable[[tuple[int, ...], Sequence[Any]], int] | None = None,
        greedy_block_executor: (
            Callable[[tuple[int, ...], Sequence[Any]], Sequence[int]] | None
        ) = None,
        greedy_proposal_executor: (Callable[[tuple[int, ...], Sequence[Any]], int] | None) = None,
        greedy_proposal_seal_executor: (
            Callable[[tuple[int, ...], Sequence[Any]], None] | None
        ) = None,
        sampling_executor: (
            Callable[[tuple[int, ...], Sequence[Any], SamplingRequest], int] | None
        ) = None,
        prefill_chunk_size: int | None = None,
        prefill_chunk_executor: (
            Callable[
                [tuple[int, ...], Sequence[Any], OutputRequest | None],
                int | None,
            ]
            | None
        ) = None,
        state_layout: MlxStateLayout | None = None,
        paged_decode_attention_lane: MlxPagedDecodeAttentionLane | None = None,
        owns_cache_factory: bool = False,
        owns_engine: bool = False,
    ) -> MlxNativeRuntime:
        validate_placement_plan(placement, model, workload, capabilities, device)
        if workload.output_mode not in (
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ):
            raise NotImplementedError("MLX native runtime promotes only native next-token output")
        if workload.max_batch_size != 1:
            raise NotImplementedError(
                "transactional MLX state is B1; native batch scheduling is a separate lane"
            )
        if str(getattr(engine, "arch", "")) != model.architecture:
            raise ValueError("engine architecture differs from the compiled model")
        if str(getattr(engine, "numerical_contract", "")) != workload.numerical_contract:
            raise ValueError("engine numerical contract differs from the admitted workload")
        route = RuntimeRoute(
            runtime_id=f"mlx-{uuid4().hex}",
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
            cache_factory=cache_factory,
            executor=executor,
            greedy_block_executor=greedy_block_executor,
            greedy_proposal_executor=greedy_proposal_executor,
            greedy_proposal_seal_executor=greedy_proposal_seal_executor,
            sampling_executor=sampling_executor,
            prefill_chunk_size=prefill_chunk_size,
            prefill_chunk_executor=prefill_chunk_executor,
            state_layout=state_layout,
            paged_decode_attention_lane=paged_decode_attention_lane,
            owns_cache_factory=owns_cache_factory,
            owns_engine=owns_engine,
        )

    @property
    def route(self) -> RuntimeRoute:
        return self._route

    @property
    def prefill_execution_shape(self) -> MlxPrefillExecutionShape:
        return self._prefill_execution_shape

    @property
    def paged_decode_attention_identity(self) -> Any | None:
        lane = self._paged_decode_attention_lane
        return None if lane is None else lane.identity

    def paged_decode_attention_telemetry(self) -> Any | None:
        lane = self._paged_decode_attention_lane
        return None if lane is None else lane.telemetry()

    def _default_cache_factory(self, capacity: int) -> Sequence[FixedMlxKVCache]:
        from mlx_lm.models.cache import create_attention_mask

        mx = self._engine._mx
        if self._state_layout is None:
            raise MlxNativeRuntimeError("MLX state layout was not measured")
        return tuple(
            FixedMlxKVCache(
                mx=mx,
                create_attention_mask=create_attention_mask,
                capacity=capacity,
                kv_heads=layer.kv_heads,
                key_head_dim=layer.key_head_dim,
                value_head_dim=layer.value_head_dim,
                key_dtype=layer.key_dtype,
                value_dtype=layer.value_dtype,
            )
            for layer in self._state_layout.layers
        )

    def _default_executor(self, ids: tuple[int, ...], caches: Sequence[Any]) -> int:
        mx = self._engine._mx
        inputs = mx.array(np.asarray(ids, dtype=np.int64))[None, :]
        logits = self._engine.model(inputs, cache=caches)
        token = mx.argmax(logits[0, -1, : self._semantic_token_count], axis=-1)
        mx.eval(token, *_cache_evaluation_arrays(caches))
        value = int(token.item())
        if value < 0 or value >= self._semantic_token_count:
            raise MlxNativeRuntimeError("native MLX argmax escaped the semantic token domain")
        return value

    def _default_greedy_block_executor(
        self,
        ids: tuple[int, ...],
        caches: Sequence[Any],
    ) -> tuple[int, ...]:
        """Select every position on device and transfer only the selected integer vector."""

        mx = self._engine._mx
        inputs = mx.array(np.asarray(ids, dtype=np.int64))[None, :]
        logits = self._engine.model(inputs, cache=caches)
        selected = mx.argmax(logits[0, :, : self._semantic_token_count], axis=-1)
        mx.eval(selected, *_cache_evaluation_arrays(caches))
        raw = selected.tolist()
        if not isinstance(raw, list):
            raise MlxNativeRuntimeError("native MLX block argmax returned a non-vector result")
        return tuple(raw)

    def _default_greedy_proposal_seal_executor(
        self,
        ids: tuple[int, ...],
        caches: Sequence[Any],
    ) -> None:
        """Append one final draft input while realizing K/V only, without host selection."""

        mx = self._engine._mx
        inputs = mx.array(np.asarray(ids, dtype=np.int64))[None, :]
        self._engine.model(inputs, cache=caches)
        mx.eval(*_cache_evaluation_arrays(caches))

    def _default_sampling_executor(
        self,
        ids: tuple[int, ...],
        caches: Sequence[Any],
        request: SamplingRequest,
    ) -> int:
        mx = self._engine._mx
        inputs = mx.array(np.asarray(ids, dtype=np.int64))[None, :]
        logits = self._engine.model(inputs, cache=caches)
        token = _sample_mlx_row(
            mx,
            logits[0, -1],
            request,
            semantic_token_count=self._semantic_token_count,
        )
        mx.eval(token, *_cache_evaluation_arrays(caches))
        value = int(token.item())
        if value < 0 or value >= self._semantic_token_count:
            raise MlxNativeRuntimeError("native MLX sampling produced invalid probabilities")
        return value

    def _default_prefill_chunk_executor(
        self,
        ids: tuple[int, ...],
        caches: Sequence[Any],
        output: OutputRequest | None,
    ) -> int | None:
        """Evaluate one bounded prefill chunk; only a final chunk performs token selection."""

        mx = self._engine._mx
        inputs = mx.array(np.asarray(ids, dtype=np.int64))[None, :]
        logits = self._engine.model(inputs, cache=caches)
        cache_values = _cache_evaluation_arrays(caches)
        if output is None:
            # Realize only the provisional K/V dependencies.  The intermediate vocabulary head
            # is deliberately not selected or materialized on the host.
            mx.eval(*cache_values)
            return None
        if output.mode is OutputMode.NEXT_TOKEN_ARGMAX:
            selected = mx.argmax(logits[0, -1, : self._semantic_token_count], axis=-1)
        elif output.mode is OutputMode.NEXT_TOKEN_SAMPLE:
            selected = _sample_mlx_row(
                mx,
                logits[0, -1],
                output.sampling[0],
                semantic_token_count=self._semantic_token_count,
            )
        else:
            raise NotImplementedError("chunked MLX prefill supports next-token output only")
        mx.eval(selected, *cache_values)
        return int(selected.item())

    def _execute_chunked_prefill(
        self,
        ids: tuple[int, ...],
        caches: Sequence[Any],
        output: OutputRequest,
    ) -> int:
        chunk_size = self._prefill_chunk_size
        if chunk_size is None or len(ids) <= chunk_size:
            raise ValueError("chunked prefill requires an input larger than its configured chunk")
        chunks = tuple(ids[start : start + chunk_size] for start in range(0, len(ids), chunk_size))
        try:
            selected: int | None = None
            for index, chunk in enumerate(chunks):
                final = index == len(chunks) - 1
                value = self._prefill_chunk_executor(
                    chunk,
                    caches,
                    output if final else None,
                )
                if final:
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise MlxNativeRuntimeError(
                            "final MLX prefill chunk did not select one strict token ID"
                        )
                    selected = int(value)
                elif value is not None:
                    raise MlxNativeRuntimeError(
                        "intermediate MLX prefill chunk unexpectedly selected a token"
                    )
                with self._chunk_lock:
                    self._prefill_chunks += 1
        except BaseException:
            with self._chunk_lock:
                self._prefill_chunk_failures += 1
            raise
        if selected is None:
            raise MlxNativeRuntimeError("chunked MLX prefill produced no final token")
        with self._chunk_lock:
            self._chunked_prefill_calls += 1
        return selected

    def _require_open(self) -> None:
        if self._closed:
            raise MlxNativeRuntimeError("native MLX runtime is closed")

    def _state(self, handle: Any) -> MlxNativeState:
        with self._lock:
            self._require_open()
            if not isinstance(handle, MlxNativeState):
                raise TypeError("state was not issued by the MLX runtime")
            if handle.runtime_id != self._route.runtime_id:
                raise MlxNativeRuntimeError("state belongs to another runtime")
            if self._states.get(handle.state_id) is not handle:
                raise MlxNativeRuntimeError("state authority is stale or released")
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
    ) -> MlxNativeState:
        owner_id = _name(owner_id, "owner_id")
        if batch_size != 1 or isinstance(batch_size, bool):
            raise ValueError("transactional MLX state currently requires batch_size=1")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0 or capacity > self._placement.state.max_context_tokens:
            raise ValueError("capacity lies outside the admitted placement")
        with self._lock:
            self._require_open()
            caches: tuple[Any, ...] = ()
            try:
                caches = tuple(self._cache_factory(capacity))
                if not caches or any(int(cache.offset) != 0 for cache in caches):
                    raise MlxNativeRuntimeError("cache factory must return empty fixed arenas")
                actual_bytes = _cache_bytes(caches)
                expected_bytes = self._placement.state.bytes_per_token * capacity
                if actual_bytes != expected_bytes:
                    raise MlxNativeRuntimeError(
                        "fixed MLX arena bytes differ from placement accounting "
                        f"({actual_bytes} != {expected_bytes})"
                    )
                state_id = f"state-{uuid4().hex}"
                state = MlxNativeState(
                    runtime_id=self._route.runtime_id,
                    state_id=state_id,
                    owner_id=owner_id,
                    state_abi=self._state_abi,
                    capacity=capacity,
                    generation=0,
                    caches=caches,
                )
            except BaseException as exc:
                if caches:
                    try:
                        self._release_caches(caches)
                    except BaseException as cleanup_exc:
                        exc.add_note(f"cache allocation cleanup also failed: {cleanup_exc}")
                raise
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
        """Copy the committed B1 prefix into fresh fixed unified-memory arenas."""

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
                raise MlxNativeRuntimeError("source state authority is stale or released")
            current = resolved._observe_unlocked()  # noqa: SLF001
            if resolved._pending_step_id is not None:  # noqa: SLF001
                raise MlxNativeRuntimeError("cannot fork state with pending provisional work")
            if current != parent:
                raise MlxNativeRuntimeError("state fork parent is stale or does not bind source")
            if current.state_abi != self._state_abi:
                raise MlxNativeRuntimeError("source state ABI differs from the runtime")
            length = current.lengths[0]
            if length > capacity:
                raise OverflowError("committed source prefix exceeds fork capacity")
            source_caches = resolved._caches  # noqa: SLF001
            source_expected_bytes = self._placement.state.bytes_per_token * current.capacity
            if _cache_bytes(source_caches) != source_expected_bytes:
                raise MlxNativeRuntimeError(
                    "source MLX arena bytes differ from state ABI accounting"
                )
            source_signatures = _cache_signature(source_caches)
            if len(set(source_signatures)) != len(source_signatures):
                raise MlxNativeRuntimeError("source MLX layers alias backing storage")

            caches: tuple[Any, ...] = ()
            try:
                caches = tuple(self._cache_factory(capacity))
                if len(caches) != len(source_caches):
                    raise MlxNativeRuntimeError("fork cache factory changed the layer count")
                if not caches or any(int(cache.offset) != 0 for cache in caches):
                    raise MlxNativeRuntimeError("fork cache factory must return empty fixed arenas")
                actual_bytes = _cache_bytes(caches)
                expected_bytes = self._placement.state.bytes_per_token * capacity
                if actual_bytes != expected_bytes:
                    raise MlxNativeRuntimeError(
                        "fork MLX arena bytes differ from placement accounting "
                        f"({actual_bytes} != {expected_bytes})"
                    )
                target_signatures = _cache_signature(caches)
                if len(set(target_signatures)) != len(target_signatures):
                    raise MlxNativeRuntimeError("fork MLX layers alias backing storage")
                if set(target_signatures).intersection(source_signatures):
                    raise MlxNativeRuntimeError("fork cache aliases source backing storage")
                physical_copy_bytes: list[int] = []
                copy_byte_override = True
                for target, source_cache in zip(
                    caches,
                    source_caches,
                    strict=True,
                ):
                    if target is source_cache:
                        raise MlxNativeRuntimeError("fork cache aliases source backing storage")
                    copy_prefix = getattr(target, "copy_committed_prefix_from", None)
                    if not callable(copy_prefix):
                        raise MlxNativeRuntimeError(
                            "fixed MLX cache lacks exact-prefix copy support"
                        )
                    copy_prefix(source_cache, length)
                    if int(target.offset) != length:
                        raise MlxNativeRuntimeError("MLX cache fork did not preserve prefix length")
                    target_copy_bytes = getattr(target, "last_prefix_copy_bytes", None)
                    if (
                        isinstance(target_copy_bytes, bool)
                        or not isinstance(target_copy_bytes, int)
                        or target_copy_bytes < 0
                    ):
                        copy_byte_override = False
                    else:
                        physical_copy_bytes.append(target_copy_bytes)
                if _cache_signature(caches) != target_signatures:
                    raise MlxNativeRuntimeError("fork cache backing storage changed during copy")
                if resolved._observe_unlocked() != current:  # noqa: SLF001
                    raise MlxNativeRuntimeError("native fork mutated its source state")

                state_id = f"state-{uuid4().hex}"
                forked_state = MlxNativeState(
                    runtime_id=self._route.runtime_id,
                    state_id=state_id,
                    owner_id=owner_id,
                    state_abi=self._state_abi,
                    capacity=capacity,
                    generation=self._mint_state_generation_unlocked(),
                    caches=caches,
                )
                forked_state._committed_length = length  # noqa: SLF001
                forked_state._epoch = 1  # noqa: SLF001 - fork installs one logical prefix
                forked = forked_state._observe_unlocked()  # noqa: SLF001
                copied = (
                    sum(physical_copy_bytes)
                    if copy_byte_override
                    else length * self._placement.state.bytes_per_token
                )
                result = StateForkResult(
                    runtime_id=self._route.runtime_id,
                    source=current,
                    forked=forked,
                    state=forked_state,
                    state_bytes_copied=copied,
                )
            except BaseException as exc:
                if caches:
                    try:
                        self._release_caches(caches)
                    except BaseException as cleanup_exc:
                        exc.add_note(f"fork cache cleanup also failed: {cleanup_exc}")
                raise
            self._states[state_id] = forked_state
            self._state_forks += 1
            self._state_fork_tokens += length
            self._state_fork_bytes += copied
            return result

    @staticmethod
    def _reset_caches(state: MlxNativeState, offset: int) -> None:
        for cache in state._caches:  # noqa: SLF001
            reset = getattr(cache, "reset", None)
            if callable(reset):
                reset(offset)
            else:
                cache.offset = offset

    def _validate_cache_release(self, caches: Sequence[Any]) -> None:
        factory_validator = getattr(self._cache_factory, "validate_release_caches", None)
        if callable(factory_validator):
            factory_validator(caches)
            return
        for cache in caches:
            validator = getattr(cache, "validate_release", None)
            if callable(validator):
                validator()

    def _release_caches(self, caches: Sequence[Any]) -> None:
        """Release optional pooled-cache authority without changing fixed-arena behavior."""

        factory_release = getattr(self._cache_factory, "release_caches", None)
        if callable(factory_release):
            factory_release(caches)
            return
        self._validate_cache_release(caches)
        for cache in reversed(tuple(caches)):
            release = getattr(cache, "release", None)
            if callable(release):
                release()

    def _execute(self, work: PrefillWork | DecodeWork, *, phase: str) -> ProvisionalStep:
        if work.output.mode not in (
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ):
            raise NotImplementedError("MLX native runtime supports only native next-token output")
        state = self._state(work.state)
        with state._lock:  # noqa: SLF001 - runtime owns the opaque state implementation
            current = state._observe_unlocked()  # noqa: SLF001
            if work.parent != current:
                raise MlxNativeRuntimeError("work parent is stale or does not bind this state")
            if state._pending_step_id is not None:  # noqa: SLF001
                raise MlxNativeRuntimeError("state already has an unconsumed provisional step")
            if len(work.token_rows) != 1:
                raise NotImplementedError("transactional MLX state is B1")
            ids = tuple(int(value) for value in work.token_rows[0])
            if any(value < 0 or value >= self._semantic_token_count for value in ids):
                raise ValueError("input token IDs escape the semantic token domain")
            step_id = f"step-{uuid4().hex}"
            state._pending_step_id = step_id  # noqa: SLF001
            try:
                started = time.perf_counter()
                if (
                    phase == "prefill"
                    and self._prefill_chunk_size is not None
                    and len(ids) > self._prefill_chunk_size
                ):
                    token = self._execute_chunked_prefill(ids, state._caches, work.output)  # noqa: SLF001
                elif work.output.mode is OutputMode.NEXT_TOKEN_ARGMAX:
                    token = int(self._executor(ids, state._caches))  # noqa: SLF001
                else:
                    token = int(  # noqa: SLF001
                        self._sampling_executor(
                            ids,
                            state._caches,
                            work.output.sampling[0],
                        )
                    )
                elapsed = time.perf_counter() - started
                if token < 0 or token >= self._semantic_token_count:
                    raise MlxNativeRuntimeError("native MLX selection escaped the token domain")
                state._pending_count = len(ids)  # noqa: SLF001
                state._verify_storage_unlocked()  # noqa: SLF001
                authority = MlxProvisionalAuthority(
                    runtime_id=self._route.runtime_id,
                    step_id=step_id,
                    state=state,
                )
                step = ProvisionalStep(
                    runtime_id=self._route.runtime_id,
                    step_id=step_id,
                    request_ids=work.request_ids,
                    state=state,
                    parent=current,
                    token_counts=(len(ids),),
                    output=NativeOutput(
                        mode=work.output.mode,
                        token_ids=(token,),
                    ),
                    authority=authority,
                )
            except BaseException:
                self._reset_caches(state, state._committed_length)  # noqa: SLF001
                state._pending_count = 0  # noqa: SLF001
                state._pending_step_id = None  # noqa: SLF001
                raise
        with self._lock:
            self._provisional_steps += 1
            self._device_to_host_bytes += 8
            self._workspace_peak_bytes = max(
                self._workspace_peak_bytes,
                self._placement.state.bytes_per_token * len(ids),
            )
            if phase == "prefill":
                self._prefill_calls += 1
                self._prefill_tokens += len(ids)
                self._prefill_seconds += elapsed
            else:
                self._decode_calls += 1
                self._decode_tokens += len(ids)
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

    def verify_greedy_block(
        self,
        work: GreedyBlockVerifyWork,
    ) -> GreedyBlockVerification:
        """Run one B1 teacher-forced block and keep its complete K/V suffix provisional."""

        if not isinstance(work, GreedyBlockVerifyWork):
            raise TypeError("greedy block verification requires GreedyBlockVerifyWork")
        state = self._state(work.state)
        try:
            with state._lock:  # noqa: SLF001 - runtime owns the opaque state implementation
                current = state._observe_unlocked()  # noqa: SLF001
                if work.parent != current:
                    raise MlxNativeRuntimeError(
                        "greedy block parent is stale or does not bind this state"
                    )
                if state._pending_step_id is not None:  # noqa: SLF001
                    raise MlxNativeRuntimeError("state already has an unconsumed provisional step")
                ids = tuple(work.token_ids)
                if any(value >= self._semantic_token_count for value in ids):
                    raise ValueError("input token IDs escape the semantic token domain")
                step_id = f"verify-{uuid4().hex}"
                state._pending_step_id = step_id  # noqa: SLF001
                try:
                    started = time.perf_counter()
                    raw_predictions = tuple(
                        self._greedy_block_executor(ids, state._caches)  # noqa: SLF001
                    )
                    elapsed = time.perf_counter() - started
                    if len(raw_predictions) != len(ids):
                        raise MlxNativeRuntimeError(
                            "native MLX block verifier must select every input position"
                        )
                    if any(
                        isinstance(value, bool) or not isinstance(value, (int, np.integer))
                        for value in raw_predictions
                    ):
                        raise MlxNativeRuntimeError(
                            "native MLX block verifier returned a non-integer token"
                        )
                    predictions = tuple(int(value) for value in raw_predictions)
                    if any(
                        value < 0 or value >= self._semantic_token_count for value in predictions
                    ):
                        raise MlxNativeRuntimeError(
                            "native MLX block selection escaped the token domain"
                        )
                    state._pending_count = len(ids)  # noqa: SLF001
                    state._verify_storage_unlocked()  # noqa: SLF001
                    authority = MlxGreedyBlockAuthority(
                        runtime_id=self._route.runtime_id,
                        step_id=step_id,
                        state=state,
                    )
                    verification = GreedyBlockVerification(
                        runtime_id=self._route.runtime_id,
                        step_id=step_id,
                        request_id=work.request_id,
                        state=state,
                        parent=current,
                        input_token_ids=ids,
                        predicted_token_ids=predictions,
                        authority=authority,
                    )
                except BaseException:
                    self._reset_caches(state, state._committed_length)  # noqa: SLF001
                    state._pending_count = 0  # noqa: SLF001
                    state._pending_step_id = None  # noqa: SLF001
                    raise
        except BaseException:
            with self._lock:
                self._greedy_block_failures += 1
            raise
        with self._lock:
            self._greedy_block_verifications += 1
            self._greedy_block_tokens += len(ids)
            self._provisional_steps += 1
            self._decode_calls += 1
            self._decode_tokens += len(ids)
            self._decode_seconds += elapsed
            self._device_to_host_bytes += 8 * len(predictions)
            self._workspace_peak_bytes = max(
                self._workspace_peak_bytes,
                self._placement.state.bytes_per_token * len(ids),
            )
        return verification

    def begin_greedy_proposal(
        self,
        work: GreedyProposalBeginWork,
    ) -> GreedyProposalTransaction:
        """Reserve one authoritative draft state for a versioned provisional suffix."""

        if not isinstance(work, GreedyProposalBeginWork):
            raise TypeError("greedy proposal begin requires GreedyProposalBeginWork")
        state = self._state(work.state)
        try:
            with state._lock:  # noqa: SLF001
                current = state._observe_unlocked()  # noqa: SLF001
                if current != work.parent:
                    raise MlxNativeRuntimeError(
                        "greedy proposal parent is stale or does not bind this state"
                    )
                if state._pending_step_id is not None:  # noqa: SLF001
                    raise MlxNativeRuntimeError("state already has an unconsumed provisional step")
                step_id = f"proposal-{uuid4().hex}"
                authority = MlxGreedyProposalAuthority(
                    runtime_id=self._route.runtime_id,
                    step_id=step_id,
                    state=state,
                )
                transaction = GreedyProposalTransaction(
                    runtime_id=self._route.runtime_id,
                    step_id=step_id,
                    request_id=work.request_id,
                    state=state,
                    parent=current,
                    input_token_ids=(),
                    predicted_token_ids=(),
                    sealed=False,
                    authority=authority,
                )
                state._pending_count = 0  # noqa: SLF001
                state._pending_step_id = step_id  # noqa: SLF001
                try:
                    state._verify_storage_unlocked()  # noqa: SLF001
                except BaseException:
                    # Beginning a transaction must either publish a usable reservation or leave
                    # the state completely unreserved.  No K/V has been appended yet, so clearing
                    # the pending identity is the exact rollback for this failure boundary.
                    state._pending_count = 0  # noqa: SLF001
                    state._pending_step_id = None  # noqa: SLF001
                    authority._consumed = True  # noqa: SLF001
                    raise
        except BaseException:
            with self._lock:
                self._greedy_proposal_failures += 1
            raise
        with self._lock:
            self._greedy_proposal_transactions += 1
            self._provisional_steps += 1
        return transaction

    def _greedy_proposal_authority(
        self,
        transaction: GreedyProposalTransaction,
    ) -> MlxGreedyProposalAuthority:
        if not isinstance(transaction, GreedyProposalTransaction):
            raise TypeError("proposal operation requires GreedyProposalTransaction")
        if transaction.runtime_id != self._route.runtime_id:
            raise MlxNativeRuntimeError("greedy proposal belongs to another runtime")
        authority = transaction.authority
        if type(authority) is not MlxGreedyProposalAuthority:
            raise TypeError("greedy proposal authority was not issued by this runtime")
        if authority._consumed:  # noqa: SLF001
            raise MlxNativeRuntimeError("greedy proposal authority has already been consumed")
        if authority._state is not transaction.state:  # noqa: SLF001
            raise MlxNativeRuntimeError(
                "greedy proposal authority and transaction bind different state"
            )
        if (
            authority._version != transaction.version  # noqa: SLF001
            or authority._sealed != transaction.sealed  # noqa: SLF001
            or authority._input_token_ids != transaction.input_token_ids  # noqa: SLF001
            or authority._predicted_token_ids  # noqa: SLF001
            != transaction.predicted_token_ids
        ):
            raise MlxNativeRuntimeError("greedy proposal transaction snapshot is stale")
        return authority

    def _extend_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        input_token_id: int,
        *,
        select_next: bool,
    ) -> GreedyProposalTransaction:
        if isinstance(input_token_id, bool) or not isinstance(input_token_id, int):
            raise TypeError("greedy proposal input token must be a strict integer")
        token_id = int(input_token_id)
        if token_id < 0 or token_id >= self._semantic_token_count:
            raise ValueError("greedy proposal input escapes the semantic token domain")
        try:
            authority = self._greedy_proposal_authority(transaction)
            if transaction.sealed:
                raise MlxNativeRuntimeError("sealed greedy proposal cannot be extended")
            state = self._state(transaction.state)
            with state._lock:  # noqa: SLF001
                current = state._observe_unlocked()  # noqa: SLF001
                if (
                    current != transaction.parent
                    or state._pending_step_id != transaction.step_id  # noqa: SLF001
                    or state._pending_count != transaction.input_count  # noqa: SLF001
                ):
                    raise MlxNativeRuntimeError(
                        "state no longer owns this greedy proposal transaction"
                    )
                if current.lengths[0] + transaction.input_count + 1 > current.capacity:
                    raise OverflowError("greedy proposal transaction exceeds state capacity")
                old_pending = state._pending_count  # noqa: SLF001
                old_offset = state._committed_length + old_pending  # noqa: SLF001
                try:
                    started = time.perf_counter()
                    if select_next:
                        raw_prediction = self._greedy_proposal_executor(  # noqa: SLF001
                            (token_id,),
                            state._caches,  # noqa: SLF001
                        )
                        elapsed = time.perf_counter() - started
                        if isinstance(raw_prediction, bool) or not isinstance(
                            raw_prediction,
                            (int, np.integer),
                        ):
                            raise MlxNativeRuntimeError(
                                "native MLX proposal selected a non-integer token"
                            )
                        prediction = int(raw_prediction)
                        if prediction < 0 or prediction >= self._semantic_token_count:
                            raise MlxNativeRuntimeError(
                                "native MLX proposal selection escaped the token domain"
                            )
                        predictions = transaction.predicted_token_ids + (prediction,)
                    else:
                        self._greedy_proposal_seal_executor(  # noqa: SLF001
                            (token_id,),
                            state._caches,  # noqa: SLF001
                        )
                        elapsed = time.perf_counter() - started
                        predictions = transaction.predicted_token_ids
                    inputs = transaction.input_token_ids + (token_id,)
                    state._pending_count = old_pending + 1  # noqa: SLF001
                    state._verify_storage_unlocked()  # noqa: SLF001
                    next_transaction = GreedyProposalTransaction(
                        runtime_id=self._route.runtime_id,
                        step_id=transaction.step_id,
                        request_id=transaction.request_id,
                        state=state,
                        parent=transaction.parent,
                        input_token_ids=inputs,
                        predicted_token_ids=predictions,
                        sealed=not select_next,
                        authority=authority,
                    )
                    authority._version += 1  # noqa: SLF001
                    authority._input_token_ids = inputs  # noqa: SLF001
                    authority._predicted_token_ids = predictions  # noqa: SLF001
                    authority._sealed = not select_next  # noqa: SLF001
                except BaseException:
                    self._reset_caches(state, old_offset)
                    state._pending_count = old_pending  # noqa: SLF001
                    state._verify_storage_unlocked()  # noqa: SLF001
                    raise
        except BaseException:
            with self._lock:
                self._greedy_proposal_failures += 1
            raise
        with self._lock:
            self._decode_calls += 1
            self._decode_tokens += 1
            self._decode_seconds += elapsed
            self._greedy_proposal_tokens += 1
            self._workspace_peak_bytes = max(
                self._workspace_peak_bytes,
                self._placement.state.bytes_per_token,
            )
            if select_next:
                self._greedy_proposal_advances += 1
                self._greedy_proposal_selected_tokens += 1
                self._device_to_host_bytes += 8
            else:
                self._greedy_proposal_seals += 1
        return next_transaction

    def advance_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        input_token_id: int,
    ) -> GreedyProposalTransaction:
        return self._extend_greedy_proposal(
            transaction,
            input_token_id,
            select_next=True,
        )

    def seal_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        input_token_id: int,
    ) -> GreedyProposalTransaction:
        return self._extend_greedy_proposal(
            transaction,
            input_token_id,
            select_next=False,
        )

    def commit_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        accepted_input_count: int,
    ) -> CommitResult:
        authority = self._greedy_proposal_authority(transaction)
        if isinstance(accepted_input_count, bool) or not isinstance(accepted_input_count, int):
            raise TypeError("accepted_input_count must be a strict integer")
        accepted = int(accepted_input_count)
        if accepted < 0 or accepted > transaction.input_count:
            raise ValueError("accepted input count lies outside the proposal transaction")
        state = self._state(transaction.state)
        terminal_error: BaseException | None = None
        with state._lock:  # noqa: SLF001
            before = state._observe_unlocked()  # noqa: SLF001
            if (
                before != transaction.parent
                or state._pending_step_id != transaction.step_id  # noqa: SLF001
                or state._pending_count != transaction.input_count  # noqa: SLF001
            ):
                raise MlxNativeRuntimeError("state no longer owns this greedy proposal transaction")
            rejected = transaction.input_count - accepted
            try:
                for cache in state._caches:  # noqa: SLF001
                    if int(cache.trim(rejected)) != rejected:
                        raise MlxNativeRuntimeError(
                            "MLX cache could not trim the rejected proposal suffix"
                        )
            except BaseException as exc:
                self._reset_caches(state, state._committed_length)  # noqa: SLF001
                state._pending_count = 0  # noqa: SLF001
                state._pending_step_id = None  # noqa: SLF001
                authority._consumed = True  # noqa: SLF001
                state._verify_storage_unlocked()  # noqa: SLF001
                terminal_error = exc
            if terminal_error is None:
                state._pending_count = 0  # noqa: SLF001
                state._committed_length += accepted  # noqa: SLF001
                state._epoch += 1  # noqa: SLF001
                state._pending_step_id = None  # noqa: SLF001
                authority._consumed = True  # noqa: SLF001
                after = state._observe_unlocked()  # noqa: SLF001
        if terminal_error is not None:
            with self._lock:
                self._greedy_proposal_failures += 1
            raise terminal_error
        written = accepted * self._placement.state.bytes_per_token
        receipt = CommitResult(
            runtime_id=self._route.runtime_id,
            step_id=transaction.step_id,
            state_id=state.state_id,
            accepted_counts=(accepted,),
            before=before,
            after=after,
            state_bytes_written=written,
        )
        with self._lock:
            self._commits += 1
            self._committed_tokens += accepted
            self._greedy_proposal_commits += 1
            self._greedy_proposal_accepted_tokens += accepted
            self._greedy_proposal_rejected_tokens += rejected
        return receipt

    def abandon_greedy_proposal(self, transaction: GreedyProposalTransaction) -> None:
        authority = self._greedy_proposal_authority(transaction)
        state = self._state(transaction.state)
        with state._lock:  # noqa: SLF001
            if (
                state._observe_unlocked() != transaction.parent  # noqa: SLF001
                or state._pending_step_id != transaction.step_id  # noqa: SLF001
                or state._pending_count != transaction.input_count  # noqa: SLF001
            ):
                raise MlxNativeRuntimeError("state no longer owns this greedy proposal transaction")
            self._reset_caches(state, state._committed_length)  # noqa: SLF001
            state._pending_count = 0  # noqa: SLF001
            state._pending_step_id = None  # noqa: SLF001
            authority._consumed = True  # noqa: SLF001
            state._verify_storage_unlocked()  # noqa: SLF001
        with self._lock:
            self._abandons += 1
            self._greedy_proposal_abandons += 1
            self._greedy_proposal_abandoned_tokens += transaction.input_count

    def _authority(self, step: ProvisionalStep) -> MlxProvisionalAuthority:
        if not isinstance(step, ProvisionalStep):
            raise TypeError("terminal operation requires ProvisionalStep")
        if step.runtime_id != self._route.runtime_id:
            raise MlxNativeRuntimeError("provisional step belongs to another runtime")
        authority = step.authority
        if not isinstance(authority, MlxProvisionalAuthority):
            raise TypeError("provisional authority was not issued by this runtime")
        if isinstance(authority, MlxGreedyBlockAuthority):
            raise TypeError("greedy block authority cannot terminate an ordinary step")
        if isinstance(authority, MlxGreedyProposalAuthority):
            raise TypeError("greedy proposal authority cannot terminate an ordinary step")
        if authority._consumed:  # noqa: SLF001
            raise MlxNativeRuntimeError("provisional authority has already been consumed")
        if authority._state is not step.state:  # noqa: SLF001
            raise MlxNativeRuntimeError("provisional authority and step bind different state")
        return authority

    def _greedy_block_authority(
        self,
        verification: GreedyBlockVerification,
    ) -> MlxGreedyBlockAuthority:
        if not isinstance(verification, GreedyBlockVerification):
            raise TypeError("terminal operation requires GreedyBlockVerification")
        if verification.runtime_id != self._route.runtime_id:
            raise MlxNativeRuntimeError("greedy block belongs to another runtime")
        authority = verification.authority
        if type(authority) is not MlxGreedyBlockAuthority:
            raise TypeError("greedy block authority was not issued by this runtime")
        if authority._consumed:  # noqa: SLF001
            raise MlxNativeRuntimeError("greedy block authority has already been consumed")
        if authority._state is not verification.state:  # noqa: SLF001
            raise MlxNativeRuntimeError(
                "greedy block authority and verification bind different state"
            )
        return authority

    def commit_greedy_block(
        self,
        verification: GreedyBlockVerification,
        accepted_input_count: int,
    ) -> CommitResult:
        authority = self._greedy_block_authority(verification)
        if isinstance(accepted_input_count, bool) or not isinstance(accepted_input_count, int):
            raise TypeError("accepted_input_count must be a strict integer")
        accepted = int(accepted_input_count)
        if accepted < 0 or accepted > verification.token_count:
            raise ValueError("accepted input count lies outside the verified token block")
        state = self._state(verification.state)
        with state._lock:  # noqa: SLF001
            before = state._observe_unlocked()  # noqa: SLF001
            if (
                before != verification.parent or state._pending_step_id != verification.step_id  # noqa: SLF001
            ):
                raise MlxNativeRuntimeError("state no longer owns this greedy block")
            rejected = verification.token_count - accepted
            for cache in state._caches:  # noqa: SLF001
                if int(cache.trim(rejected)) != rejected:
                    raise MlxNativeRuntimeError(
                        "MLX cache could not trim the rejected block suffix"
                    )
            state._pending_count = 0  # noqa: SLF001
            state._committed_length += accepted  # noqa: SLF001
            state._epoch += 1  # noqa: SLF001
            authority._consumed = True  # noqa: SLF001
            state._pending_step_id = None  # noqa: SLF001
            after = state._observe_unlocked()  # noqa: SLF001
        written = accepted * self._placement.state.bytes_per_token
        receipt = CommitResult(
            runtime_id=self._route.runtime_id,
            step_id=verification.step_id,
            state_id=state.state_id,
            accepted_counts=(accepted,),
            before=before,
            after=after,
            state_bytes_written=written,
        )
        with self._lock:
            self._commits += 1
            self._committed_tokens += accepted
            self._greedy_block_commits += 1
            self._greedy_block_accepted_tokens += accepted
            self._greedy_block_rejected_tokens += rejected
        return receipt

    def abandon_greedy_block(self, verification: GreedyBlockVerification) -> None:
        authority = self._greedy_block_authority(verification)
        state = self._state(verification.state)
        with state._lock:  # noqa: SLF001
            if state._observe_unlocked() != verification.parent:  # noqa: SLF001
                raise MlxNativeRuntimeError("state changed after greedy block verification")
            if state._pending_step_id != verification.step_id:  # noqa: SLF001
                raise MlxNativeRuntimeError("state does not own this pending greedy block")
            for cache in state._caches:  # noqa: SLF001
                if int(cache.trim(verification.token_count)) != verification.token_count:
                    raise MlxNativeRuntimeError("MLX cache could not abandon the greedy block")
            state._pending_count = 0  # noqa: SLF001
            authority._consumed = True  # noqa: SLF001
            state._pending_step_id = None  # noqa: SLF001
            state._verify_storage_unlocked()  # noqa: SLF001
        with self._lock:
            self._abandons += 1
            self._greedy_block_abandons += 1

    def _commit_batch_row(
        self,
        step: ProvisionalStep,
        authority: MlxBatchProvisionalAuthority,
        state: MlxNativeState,
        accepted: int,
    ) -> CommitResult:
        """Install one accepted COW suffix into its independent authoritative B1 arena."""

        with state._lock:  # noqa: SLF001
            before = state._observe_unlocked()  # noqa: SLF001
            if before != step.parent or state._pending_step_id != step.step_id:  # noqa: SLF001
                raise MlxNativeRuntimeError("state no longer owns this batch provisional row")
            if state._pending_count != 0:  # noqa: SLF001
                raise MlxNativeRuntimeError("batch provisional row unexpectedly mutated B1 K/V")
            authority._scratch.install_row(  # noqa: SLF001
                state._caches,  # noqa: SLF001
                row=authority._row,  # noqa: SLF001
                start=before.lengths[0],
                count=accepted,
            )
            state._committed_length += accepted  # noqa: SLF001
            state._epoch += 1  # noqa: SLF001
            authority._consumed = True  # noqa: SLF001
            state._pending_step_id = None  # noqa: SLF001
            after = state._observe_unlocked()  # noqa: SLF001
        authority._scratch.consume(  # noqa: SLF001
            authority._row,  # noqa: SLF001
            committed=True,
            accepted_count=accepted,
        )
        written = accepted * self._placement.state.bytes_per_token
        with self._lock:
            self._commits += 1
            self._committed_tokens += accepted
        return CommitResult(
            runtime_id=self._route.runtime_id,
            step_id=step.step_id,
            state_id=state.state_id,
            accepted_counts=(accepted,),
            before=before,
            after=after,
            state_bytes_written=written,
        )

    def commit(
        self,
        step: ProvisionalStep,
        accepted_counts: Sequence[int],
    ) -> CommitResult:
        authority = self._authority(step)
        state = self._state(step.state)
        accepted = _strict_single_count(accepted_counts, maximum=step.token_counts[0])
        if isinstance(authority, MlxBatchProvisionalAuthority):
            return self._commit_batch_row(step, authority, state, accepted)
        with state._lock:  # noqa: SLF001
            before = state._observe_unlocked()  # noqa: SLF001
            if before != step.parent or state._pending_step_id != step.step_id:  # noqa: SLF001
                raise MlxNativeRuntimeError("state no longer owns this provisional step")
            rejected = step.token_counts[0] - accepted
            for cache in state._caches:  # noqa: SLF001
                if int(cache.trim(rejected)) != rejected:
                    raise MlxNativeRuntimeError("MLX cache could not trim the rejected suffix")
            state._pending_count = 0  # noqa: SLF001
            state._committed_length += accepted  # noqa: SLF001
            state._epoch += 1  # noqa: SLF001
            authority._consumed = True  # noqa: SLF001
            state._pending_step_id = None  # noqa: SLF001
            after = state._observe_unlocked()  # noqa: SLF001
        written = accepted * self._placement.state.bytes_per_token
        receipt = CommitResult(
            runtime_id=self._route.runtime_id,
            step_id=step.step_id,
            state_id=state.state_id,
            accepted_counts=(accepted,),
            before=before,
            after=after,
            state_bytes_written=written,
        )
        with self._lock:
            self._commits += 1
            self._committed_tokens += accepted
        return receipt

    def abandon(self, step: ProvisionalStep) -> None:
        authority = self._authority(step)
        state = self._state(step.state)
        if isinstance(authority, MlxBatchProvisionalAuthority):
            with state._lock:  # noqa: SLF001
                if state._observe_unlocked() != step.parent:  # noqa: SLF001
                    raise MlxNativeRuntimeError("state changed after batch provisional execution")
                if state._pending_step_id != step.step_id:  # noqa: SLF001
                    raise MlxNativeRuntimeError("state does not own this pending batch row")
                if state._pending_count != 0:  # noqa: SLF001
                    raise MlxNativeRuntimeError("batch provisional row unexpectedly mutated B1 K/V")
                authority._consumed = True  # noqa: SLF001
                state._pending_step_id = None  # noqa: SLF001
                state._verify_storage_unlocked()  # noqa: SLF001
            authority._scratch.consume(  # noqa: SLF001
                authority._row,  # noqa: SLF001
                committed=False,
            )
            with self._lock:
                self._abandons += 1
            return
        with state._lock:  # noqa: SLF001
            if state._observe_unlocked() != step.parent:  # noqa: SLF001
                raise MlxNativeRuntimeError("state changed after provisional execution")
            if state._pending_step_id != step.step_id:  # noqa: SLF001
                raise MlxNativeRuntimeError("state does not own this pending step")
            for cache in state._caches:  # noqa: SLF001
                if int(cache.trim(step.token_counts[0])) != step.token_counts[0]:
                    raise MlxNativeRuntimeError("MLX cache could not abandon the suffix")
            state._pending_count = 0  # noqa: SLF001
            authority._consumed = True  # noqa: SLF001
            state._pending_step_id = None  # noqa: SLF001
            state._verify_storage_unlocked()  # noqa: SLF001
        with self._lock:
            self._abandons += 1

    def release_state(self, state: Any) -> None:
        resolved = self._state(state)
        with self._lock, resolved._lock:  # noqa: SLF001
            if resolved._pending_step_id is not None:  # noqa: SLF001
                raise MlxNativeRuntimeError("cannot release state with pending work")
            self._release_caches(resolved._caches)  # noqa: SLF001
            if self._states.pop(resolved.state_id, None) is not resolved:
                raise MlxNativeRuntimeError("state was already released")
            resolved._released = True  # noqa: SLF001
            resolved._caches = ()  # noqa: SLF001

    def telemetry(self) -> RuntimeTelemetry:
        with self._lock:
            self._require_open()
            with self._chunk_lock:
                chunked_prefill_calls = self._chunked_prefill_calls
                prefill_chunks = self._prefill_chunks
                prefill_chunk_failures = self._prefill_chunk_failures
            batch = (
                None
                if self._compatible_batch_lane is None
                else self._compatible_batch_lane.telemetry()
            )
            paged_decode = (
                None
                if self._paged_decode_attention_lane is None
                else self._paged_decode_attention_lane.telemetry()
            )
            paged_pool = None
            if getattr(self._cache_factory, "cache_abi", None) == MLX_PAGED_KV_CACHE_ABI:
                paged_telemetry = getattr(self._cache_factory, "telemetry", None)
                if not callable(paged_telemetry):
                    raise MlxNativeRuntimeError(
                        "paged MLX cache factory does not expose pool telemetry"
                    )
                paged_pool = paged_telemetry()
            extras = {
                "chunked_prefill_calls": chunked_prefill_calls,
                "chunked_prefill_chunk_size": self._prefill_chunk_size or 0,
                "chunked_prefill_chunks": prefill_chunks,
                "chunked_prefill_enabled": int(self._prefill_chunk_size is not None),
                "chunked_prefill_failures": prefill_chunk_failures,
                "live_states": len(self._states),
                "greedy_block_abandons": self._greedy_block_abandons,
                "greedy_block_accepted_tokens": self._greedy_block_accepted_tokens,
                "greedy_block_commits": self._greedy_block_commits,
                "greedy_block_failures": self._greedy_block_failures,
                "greedy_block_rejected_tokens": self._greedy_block_rejected_tokens,
                "greedy_block_tokens": self._greedy_block_tokens,
                "greedy_block_verifications": self._greedy_block_verifications,
                "greedy_proposal_abandoned_tokens": self._greedy_proposal_abandoned_tokens,
                "greedy_proposal_abandons": self._greedy_proposal_abandons,
                "greedy_proposal_accepted_tokens": self._greedy_proposal_accepted_tokens,
                "greedy_proposal_advances": self._greedy_proposal_advances,
                "greedy_proposal_commits": self._greedy_proposal_commits,
                "greedy_proposal_failures": self._greedy_proposal_failures,
                "greedy_proposal_prefix_copy_bytes": 0,
                "greedy_proposal_rejected_tokens": self._greedy_proposal_rejected_tokens,
                "greedy_proposal_replay_forwards": 0,
                "greedy_proposal_seals": self._greedy_proposal_seals,
                "greedy_proposal_selected_tokens": self._greedy_proposal_selected_tokens,
                "greedy_proposal_tokens": self._greedy_proposal_tokens,
                "greedy_proposal_transactions": self._greedy_proposal_transactions,
                "state_fork_bytes": self._state_fork_bytes,
                "state_fork_tokens": self._state_fork_tokens,
                "state_forks": self._state_forks,
                "transactional_batch": 1,
            }
            if paged_pool is not None:
                extras.update(
                    {
                        "paged_kv_active_caches": paged_pool.active_caches,
                        "paged_kv_concatenate_materializations": (
                            paged_pool.concatenate_materializations
                        ),
                        "paged_kv_cow_copy_bytes": paged_pool.cow_copy_bytes,
                        "paged_kv_failures": paged_pool.failures,
                        "paged_kv_free_bytes": paged_pool.free_bytes,
                        "paged_kv_full_page_prefix_shares": (paged_pool.full_page_prefix_shares),
                        "paged_kv_leaked_pages": paged_pool.leaked_pages,
                        "paged_kv_live_bytes": paged_pool.live_bytes,
                        "paged_kv_logical_capacity_bytes": (paged_pool.logical_capacity_bytes),
                        "paged_kv_materialized_bytes": paged_pool.materialized_bytes,
                        "paged_kv_paged_decode_appends": paged_pool.paged_decode_appends,
                        "paged_kv_page_table_materializations": (
                            paged_pool.page_table_materializations
                        ),
                        "paged_kv_dense_materializations_avoided": (
                            paged_pool.dense_materializations_avoided
                        ),
                        "paged_kv_partial_tail_copy_bytes": (paged_pool.partial_tail_copy_bytes),
                        "paged_kv_physical_bytes": paged_pool.physical_bytes,
                        "paged_kv_pool_count": paged_pool.pool_count,
                        "paged_kv_quarantine_reconciliation_failures": (
                            paged_pool.quarantine_reconciliation_failures
                        ),
                        "paged_kv_quarantine_reconciliations": (
                            paged_pool.quarantine_reconciliations
                        ),
                        "paged_kv_quarantined_bytes": paged_pool.quarantined_bytes,
                        "paged_kv_quarantined_pages": paged_pool.quarantined_pages,
                        "paged_kv_reconciled": int(paged_pool.reconciled),
                        "paged_kv_reserved_bytes": paged_pool.reserved_bytes,
                        "paged_kv_shared_bytes": paged_pool.shared_bytes,
                    }
                )
            if paged_decode is not None:
                extras.update(
                    {
                        "paged_decode_attention_enabled": 1,
                        "paged_decode_attention_installed_layers": (paged_decode.installed_layers),
                        "paged_decode_attention_calls": paged_decode.paged_decode_calls,
                        "paged_decode_attention_attended_tokens": (paged_decode.attended_tokens),
                        "paged_decode_attention_logical_pages_read": (
                            paged_decode.logical_pages_read
                        ),
                        "paged_decode_attention_page_table_entries_uploaded": (
                            paged_decode.page_table_entries_uploaded
                        ),
                        "paged_decode_attention_dense_materializations_avoided": (
                            paged_decode.dense_materializations_avoided
                        ),
                        "paged_decode_attention_prefill_dense_calls": (
                            paged_decode.prefill_dense_calls
                        ),
                        "paged_decode_attention_uncached_dense_calls": (
                            paged_decode.uncached_dense_calls
                        ),
                        "paged_decode_attention_qwen2_calls": paged_decode.qwen2_calls,
                        "paged_decode_attention_mixtral_calls": paged_decode.mixtral_calls,
                        "paged_decode_attention_failures": paged_decode.failures,
                        "paged_decode_attention_max_attended_tokens": (
                            paged_decode.max_attended_tokens
                        ),
                    }
                )
            else:
                extras["paged_decode_attention_enabled"] = 0
            if batch is not None:
                extras.update(
                    {
                        "compatible_batch_abandoned_rows": batch.abandoned_rows,
                        "compatible_batch_active_scratches": batch.active_scratches,
                        "compatible_batch_committed_rows": batch.committed_rows,
                        "compatible_batch_dispatches": batch.dispatches,
                        "compatible_batch_failed_dispatches": batch.failed_dispatches,
                        "compatible_batch_max_width": batch.max_width,
                        "compatible_batch_physical_forwards": batch.physical_forwards,
                        "compatible_batch_prefix_bytes_copied": batch.prefix_bytes_copied,
                        "compatible_batch_provisional_rows": batch.provisional_rows,
                        "compatible_batch_scratch_peak_bytes": batch.scratch_peak_bytes,
                        "compatible_batch_scratch_limited_bypasses": (
                            batch.scratch_limited_bypasses
                        ),
                        "compatible_batch_singleton_bypasses": batch.singleton_bypasses,
                        "compatible_batch_suffix_bytes_committed": (batch.suffix_bytes_committed),
                    }
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
                kv_resident_bytes=(
                    paged_pool.physical_bytes
                    if paged_pool is not None
                    else sum(
                        _cache_bytes(state._caches)  # noqa: SLF001
                        for state in self._states.values()
                    )
                ),
                workspace_peak_bytes=self._workspace_peak_bytes,
                extra_counters=tuple(sorted(extras.items())),
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
                raise MlxNativeRuntimeError(
                    f"cannot close runtime with pending provisional states: {pending!r}"
                )
            states = tuple(self._states.values())
            for state in states:
                with state._lock:  # noqa: SLF001
                    self._validate_cache_release(state._caches)  # noqa: SLF001
            for state in states:
                with state._lock:  # noqa: SLF001
                    self._release_caches(state._caches)  # noqa: SLF001
                    state._released = True  # noqa: SLF001
                    state._caches = ()  # noqa: SLF001
            self._states.clear()
            if self._owns_cache_factory:
                self._cache_factory.close()
            if self._paged_decode_attention_lane is not None:
                self._paged_decode_attention_lane.close()
            self._closed = True
        if self._owns_engine:
            self._engine.close()


class MlxCompatibleBatchLane:
    """Opt-in compatible-request pooling over COW merged MLX K/V.

    The packed MLX forward is intentionally identified as bounded-numerical and experimental:
    it is not the exact B1 arithmetic contract.  Ragged dispatches are split into physical waves
    with equal phase, committed length, input length, and output mode.  Singleton waves call the
    runtime's unchanged B1 executor.  Production promotion requires explicit CE, task, and
    generated-trajectory evidence for this exact lane identity.
    """

    def __init__(
        self,
        runtime: MlxNativeRuntime,
        *,
        max_batch_size: int = 8,
        max_queue_delay_seconds: float = 0.002,
        max_scratch_bytes: int | None = None,
        batch_cache_factory: Callable[[int, int], Sequence[Any]] | None = None,
        batch_executor: (
            Callable[
                [tuple[tuple[int, ...], ...], Sequence[Any], OutputRequest | None],
                Sequence[int] | None,
            ]
            | None
        ) = None,
    ) -> None:
        if not isinstance(runtime, MlxNativeRuntime):
            raise TypeError("MLX compatible batching requires an exact MlxNativeRuntime")
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise TypeError("max_batch_size must be a strict integer")
        if max_batch_size <= 1:
            raise ValueError("compatible batching requires max_batch_size > 1")
        if isinstance(max_queue_delay_seconds, bool) or not isinstance(
            max_queue_delay_seconds, (int, float)
        ):
            raise TypeError("max_queue_delay_seconds must be a finite non-negative number")
        queue_delay = float(max_queue_delay_seconds)
        if not np.isfinite(queue_delay) or queue_delay < 0:
            raise ValueError("max_queue_delay_seconds must be finite and non-negative")
        default_scratch_bytes = (
            runtime._placement.state.bytes_per_token  # noqa: SLF001
            * runtime._placement.state.max_context_tokens  # noqa: SLF001
            * max_batch_size
        )
        if max_scratch_bytes is None:
            max_scratch_bytes = default_scratch_bytes
        if (
            isinstance(max_scratch_bytes, bool)
            or not isinstance(max_scratch_bytes, int)
            or max_scratch_bytes <= 0
        ):
            raise ValueError("max_scratch_bytes must be a positive integer or None")
        self._runtime = runtime
        self._batch_cache_factory = batch_cache_factory or self._default_batch_cache_factory
        self._batch_executor = batch_executor or self._default_batch_executor
        self._identity = CompatibleBatchLaneIdentity(
            lane_id=f"mlx-compatible-batch-{uuid4().hex}",
            runtime_id=runtime.route.runtime_id,
            lane_abi=MLX_COMPATIBLE_BATCH_ABI,
            numerical_contract=(
                MLX_COMPATIBLE_BATCH_NUMERICAL_CONTRACT
                if runtime.prefill_execution_shape.chunk_size is None
                else MLX_COMPATIBLE_CHUNKED_PREFILL_NUMERICAL_CONTRACT
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
        self._scratch_limited_bypasses = 0
        self._max_width = 0
        self._width_histogram: dict[int, int] = {}
        self._prefix_bytes_copied = 0
        self._suffix_bytes_committed = 0
        self._scratch_peak_bytes = 0
        self._active_scratches = 0
        with runtime._lock:  # noqa: SLF001 - explicit attachment to one runtime incarnation
            runtime._require_open()  # noqa: SLF001
            if runtime._compatible_batch_lane is not None:  # noqa: SLF001
                raise MlxNativeRuntimeError(
                    "native MLX runtime already has a compatible-request batch lane"
                )
            runtime._compatible_batch_lane = self  # noqa: SLF001

    @property
    def identity(self) -> CompatibleBatchLaneIdentity:
        return self._identity

    def _default_batch_cache_factory(
        self,
        capacity: int,
        batch_size: int,
    ) -> Sequence[FixedMlxBatchKVCache]:
        from mlx_lm.models.cache import create_attention_mask

        layout = self._runtime._state_layout  # noqa: SLF001
        if layout is None:
            raise MlxNativeRuntimeError("MLX state layout was not measured")
        mx = self._runtime._engine._mx  # noqa: SLF001
        return tuple(
            FixedMlxBatchKVCache(
                mx=mx,
                create_attention_mask=create_attention_mask,
                capacity=capacity,
                batch_size=batch_size,
                kv_heads=layer.kv_heads,
                key_head_dim=layer.key_head_dim,
                value_head_dim=layer.value_head_dim,
                key_dtype=layer.key_dtype,
                value_dtype=layer.value_dtype,
            )
            for layer in layout.layers
        )

    def _default_batch_executor(
        self,
        rows: tuple[tuple[int, ...], ...],
        caches: Sequence[Any],
        output: OutputRequest | None,
    ) -> Sequence[int] | None:
        mx = self._runtime._engine._mx  # noqa: SLF001
        inputs = mx.array(np.stack(rows, axis=0))
        logits = self._runtime._engine.model(inputs, cache=caches)  # noqa: SLF001
        semantic_tokens = self._runtime._semantic_token_count  # noqa: SLF001
        cache_values = _cache_evaluation_arrays(caches)
        if output is None:
            mx.eval(*cache_values)
            return None
        if output.mode is OutputMode.NEXT_TOKEN_ARGMAX:
            selected = mx.argmax(logits[:, -1, :semantic_tokens], axis=-1)
        elif output.mode is OutputMode.NEXT_TOKEN_SAMPLE:
            selected = mx.stack(
                [
                    _sample_mlx_row(
                        mx,
                        logits[row, -1],
                        request,
                        semantic_token_count=semantic_tokens,
                    )
                    for row, request in enumerate(output.sampling)
                ]
            )
        else:  # defensive: compatibility validation should make this unreachable
            raise NotImplementedError("MLX compatible batching supports next-token output only")
        mx.eval(
            selected,
            *cache_values,
        )
        return tuple(int(value) for value in np.asarray(selected).reshape(-1).tolist())

    @staticmethod
    def _compatibility_key(work: PrefillWork | DecodeWork) -> tuple[Any, ...]:
        return (
            type(work),
            work.parent.lengths[0],
            len(work.token_rows[0]),
            work.output.mode,
        )

    def _record_forward(
        self,
        *,
        width: int,
        prefix_bytes: int = 0,
        scratch_bytes: int = 0,
        singleton: bool = False,
        provisional_rows: int | None = None,
    ) -> None:
        with self._lock:
            self._physical_forwards += 1
            self._provisional_rows += width if provisional_rows is None else provisional_rows
            self._max_width = max(self._max_width, width)
            self._width_histogram[width] = self._width_histogram.get(width, 0) + 1
            self._prefix_bytes_copied += prefix_bytes
            self._scratch_peak_bytes = max(self._scratch_peak_bytes, scratch_bytes)
            if singleton:
                self._singleton_bypasses += 1

    def _record_scratch_limited_bypass(self) -> None:
        with self._lock:
            self._scratch_limited_bypasses += 1

    def _scratch_consumed(self, committed: bool, accepted_count: int) -> None:
        with self._lock:
            if committed:
                self._committed_rows += 1
                self._suffix_bytes_committed += (
                    accepted_count * self._runtime._placement.state.bytes_per_token  # noqa: SLF001
                )
            else:
                self._abandoned_rows += 1

    def _scratch_released(self) -> None:
        with self._lock:
            if self._active_scratches <= 0:
                raise MlxNativeRuntimeError("compatible-batch scratch accounting underflow")
            self._active_scratches -= 1

    def _execute_batch(
        self,
        indexed: Sequence[tuple[int, PrefillWork | DecodeWork]],
    ) -> tuple[tuple[int, ProvisionalStep], ...]:
        width = len(indexed)
        if width <= 1:
            raise ValueError("physical compatible batch requires at least two rows")
        works = tuple(work for _index, work in indexed)
        states = tuple(self._runtime._state(work.state) for work in works)  # noqa: SLF001
        if len({state.state_id for state in states}) != width:
            raise MlxNativeRuntimeError("compatible batch cannot contain a state twice")
        parent_length = works[0].parent.lengths[0]
        token_count = len(works[0].token_rows[0])
        capacity = parent_length + token_count
        expected_scratch_bytes = (
            self._runtime._placement.state.bytes_per_token  # noqa: SLF001
            * capacity
            * width
        )
        if expected_scratch_bytes > self._identity.max_scratch_bytes:
            raise MlxNativeRuntimeError("compatible-batch scratch exceeds its declared byte bound")
        step_ids = tuple(f"step-{uuid4().hex}" for _ in works)
        caches: tuple[Any, ...] = ()
        scratch: _MlxBatchScratch | None = None
        chunked_prefill = bool(
            isinstance(works[0], PrefillWork)
            and self._runtime._prefill_chunk_size is not None  # noqa: SLF001
            and token_count > self._runtime._prefill_chunk_size  # noqa: SLF001
        )
        physical_forwards = 1
        successful_chunks = 0
        started = time.perf_counter()
        try:
            with ExitStack() as stack:
                for state in sorted(states, key=lambda value: value.state_id):
                    stack.enter_context(state._lock)  # noqa: SLF001
                for state, work, step_id in zip(states, works, step_ids, strict=True):
                    current = state._observe_unlocked()  # noqa: SLF001
                    if current != work.parent:
                        raise MlxNativeRuntimeError(
                            "compatible-batch parent is stale or does not bind its B1 state"
                        )
                    if state._pending_step_id is not None:  # noqa: SLF001
                        raise MlxNativeRuntimeError(
                            "compatible-batch state already has provisional work"
                        )
                    state._pending_step_id = step_id  # noqa: SLF001
                    state._pending_count = 0  # noqa: SLF001

                caches = tuple(self._batch_cache_factory(capacity, width))
                source_layer_count = len(states[0]._caches)  # noqa: SLF001
                if len(caches) != source_layer_count or not caches:
                    raise MlxNativeRuntimeError(
                        "compatible-batch cache factory changed the MLX layer count"
                    )
                expected_bytes = expected_scratch_bytes
                actual_bytes = _cache_bytes(caches)
                if actual_bytes != expected_bytes:
                    raise MlxNativeRuntimeError(
                        "compatible-batch scratch bytes differ from placement accounting "
                        f"({actual_bytes} != {expected_bytes})"
                    )
                for layer, batch_cache in enumerate(caches):
                    copy_row = getattr(batch_cache, "copy_committed_row_from", None)
                    seal = getattr(batch_cache, "seal_prefix", None)
                    if not callable(copy_row) or not callable(seal):
                        raise MlxNativeRuntimeError(
                            "batch cache lacks COW prefix-copy/finalization support"
                        )
                    for row, state in enumerate(states):
                        copy_row(
                            state._caches[layer],  # noqa: SLF001
                            row=row,
                            length=parent_length,
                        )
                    seal(parent_length)
                evaluator = getattr(caches[0], "_mx", None)
                if evaluator is not None:
                    evaluator.eval(*_cache_evaluation_arrays(caches))

                mode = works[0].output.mode
                output = OutputRequest(
                    mode,
                    sampling=(
                        tuple(work.output.sampling[0] for work in works)
                        if mode is OutputMode.NEXT_TOKEN_SAMPLE
                        else ()
                    ),
                )
                rows = tuple(work.token_rows[0] for work in works)
                if chunked_prefill:
                    chunk_size = self._runtime._prefill_chunk_size  # noqa: SLF001
                    assert chunk_size is not None
                    chunk_rows = tuple(
                        tuple(row[start : start + chunk_size] for row in rows)
                        for start in range(0, token_count, chunk_size)
                    )
                    physical_forwards = len(chunk_rows)
                    selected: Sequence[int] | None = None
                    for chunk_index, chunk in enumerate(chunk_rows):
                        final = chunk_index == len(chunk_rows) - 1
                        value = self._batch_executor(
                            chunk,
                            caches,
                            output if final else None,
                        )
                        successful_chunks += 1
                        if final:
                            selected = value
                        elif value is not None:
                            raise MlxNativeRuntimeError(
                                "intermediate compatible prefill chunk selected tokens"
                            )
                    if selected is None:
                        raise MlxNativeRuntimeError(
                            "final compatible prefill chunk selected no tokens"
                        )
                    tokens = tuple(int(value) for value in selected)
                else:
                    selected = self._batch_executor(rows, caches, output)
                    if selected is None:
                        raise MlxNativeRuntimeError(
                            "compatible-batch executor selected no final tokens"
                        )
                    tokens = tuple(int(value) for value in selected)
                if len(tokens) != width or any(
                    value < 0 or value >= self._runtime._semantic_token_count  # noqa: SLF001
                    for value in tokens
                ):
                    raise MlxNativeRuntimeError(
                        "compatible-batch executor returned an invalid token row"
                    )
                if any(int(cache.offset) != capacity for cache in caches):
                    raise MlxNativeRuntimeError(
                        "compatible-batch executor did not append the requested token block"
                    )
                for state, work in zip(states, works, strict=True):
                    if state._observe_unlocked() != work.parent:  # noqa: SLF001
                        raise MlxNativeRuntimeError(
                            "compatible-batch forward mutated authoritative B1 state"
                        )

                scratch = _MlxBatchScratch(
                    caches=caches,
                    row_count=width,
                    on_consume=self._scratch_consumed,
                    on_release=self._scratch_released,
                )
                with self._lock:
                    self._active_scratches += 1
                results: list[tuple[int, ProvisionalStep]] = []
                for row, ((index, work), state, step_id, token) in enumerate(
                    zip(indexed, states, step_ids, tokens, strict=True)
                ):
                    authority = MlxBatchProvisionalAuthority(
                        runtime_id=self._runtime.route.runtime_id,
                        step_id=step_id,
                        state=state,
                        scratch=scratch,
                        row=row,
                    )
                    results.append(
                        (
                            index,
                            ProvisionalStep(
                                runtime_id=self._runtime.route.runtime_id,
                                step_id=step_id,
                                request_ids=work.request_ids,
                                state=state,
                                parent=work.parent,
                                token_counts=(token_count,),
                                output=NativeOutput(mode=mode, token_ids=(token,)),
                                authority=authority,
                            ),
                        )
                    )
        except BaseException:
            if chunked_prefill:
                with self._runtime._chunk_lock:  # noqa: SLF001
                    self._runtime._prefill_chunks += successful_chunks  # noqa: SLF001
                    self._runtime._prefill_chunk_failures += 1  # noqa: SLF001
            for state, step_id in zip(states, step_ids, strict=True):
                with state._lock:  # noqa: SLF001
                    if state._pending_step_id == step_id:  # noqa: SLF001
                        state._pending_step_id = None  # noqa: SLF001
                        state._pending_count = 0  # noqa: SLF001
                        state._verify_storage_unlocked()  # noqa: SLF001
            if scratch is not None:
                scratch.discard()
            raise

        elapsed = time.perf_counter() - started
        prefix_bytes = (
            width * parent_length * self._runtime._placement.state.bytes_per_token  # noqa: SLF001
        )
        scratch_bytes = _cache_bytes(caches)
        if chunked_prefill:
            with self._runtime._chunk_lock:  # noqa: SLF001
                self._runtime._chunked_prefill_calls += 1  # noqa: SLF001
                self._runtime._prefill_chunks += physical_forwards  # noqa: SLF001
        with self._runtime._lock:  # noqa: SLF001
            self._runtime._provisional_steps += width  # noqa: SLF001
            self._runtime._device_to_host_bytes += 8 * width  # noqa: SLF001
            self._runtime._workspace_peak_bytes = max(  # noqa: SLF001
                self._runtime._workspace_peak_bytes,  # noqa: SLF001
                scratch_bytes,
            )
            if isinstance(works[0], PrefillWork):
                self._runtime._prefill_calls += physical_forwards  # noqa: SLF001
                self._runtime._prefill_tokens += width * token_count  # noqa: SLF001
                self._runtime._prefill_seconds += elapsed  # noqa: SLF001
            else:
                self._runtime._decode_calls += 1  # noqa: SLF001
                self._runtime._decode_tokens += width * token_count  # noqa: SLF001
                self._runtime._decode_seconds += elapsed  # noqa: SLF001
        for forward in range(physical_forwards):
            self._record_forward(
                width=width,
                prefix_bytes=prefix_bytes if forward == 0 else 0,
                scratch_bytes=scratch_bytes,
                provisional_rows=width if forward == physical_forwards - 1 else 0,
            )
        return tuple(results)

    def execute(
        self,
        works: Sequence[PrefillWork | DecodeWork],
    ) -> tuple[ProvisionalStep, ...]:
        cohort = tuple(works)
        if not cohort:
            raise ValueError("compatible-batch work cohort cannot be empty")
        if len(cohort) > self._identity.max_batch_size:
            raise ValueError("compatible-batch cohort exceeds its declared maximum width")
        if any(not isinstance(work, (PrefillWork, DecodeWork)) for work in cohort):
            raise TypeError("compatible-batch cohort must contain native work values")
        if any(work.parent.batch_size != 1 for work in cohort):
            raise ValueError("compatible batching pools independent B1 states only")
        if any(work.parent.runtime_id != self._identity.runtime_id for work in cohort):
            raise MlxNativeRuntimeError("compatible-batch work belongs to another runtime")
        if len({work.request_ids[0] for work in cohort}) != len(cohort):
            raise ValueError("compatible-batch request IDs must be unique")
        if len({work.parent.state_id for work in cohort}) != len(cohort):
            raise ValueError("compatible-batch state IDs must be unique")
        with self._lock:
            self._dispatches += 1

        buckets: dict[tuple[Any, ...], list[tuple[int, PrefillWork | DecodeWork]]] = {}
        for index, work in enumerate(cohort):
            buckets.setdefault(self._compatibility_key(work), []).append((index, work))
        produced: list[tuple[int, ProvisionalStep]] = []
        try:
            for indexed in buckets.values():
                sample_work = indexed[0][1]
                row_scratch_bytes = (
                    self._runtime._placement.state.bytes_per_token  # noqa: SLF001
                    * (sample_work.parent.lengths[0] + len(sample_work.token_rows[0]))
                )
                memory_width = self._identity.max_scratch_bytes // row_scratch_bytes
                wave_width = max(1, min(len(indexed), memory_width))
                if wave_width == 1 and len(indexed) > 1:
                    self._record_scratch_limited_bypass()
                for start in range(0, len(indexed), wave_width):
                    wave = indexed[start : start + wave_width]
                    if len(wave) == 1:
                        index, work = wave[0]
                        step = (
                            self._runtime.prefill(work)
                            if isinstance(work, PrefillWork)
                            else self._runtime.decode(work)
                        )
                        produced.append((index, step))
                        self._record_forward(width=1, singleton=True)
                    else:
                        produced.extend(self._execute_batch(wave))
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
                raise MlxNativeRuntimeError(
                    f"compatible-batch cleanup failed after {primary}: {details}"
                ) from primary
            raise
        produced.sort(key=lambda item: item[0])
        return tuple(step for _index, step in produced)

    def telemetry(self) -> MlxCompatibleBatchTelemetry:
        with self._lock:
            return MlxCompatibleBatchTelemetry(
                lane_id=self._identity.lane_id,
                dispatches=self._dispatches,
                physical_forwards=self._physical_forwards,
                provisional_rows=self._provisional_rows,
                committed_rows=self._committed_rows,
                abandoned_rows=self._abandoned_rows,
                failed_dispatches=self._failed_dispatches,
                singleton_bypasses=self._singleton_bypasses,
                scratch_limited_bypasses=self._scratch_limited_bypasses,
                max_width=self._max_width,
                width_histogram=tuple(sorted(self._width_histogram.items())),
                prefix_bytes_copied=self._prefix_bytes_copied,
                suffix_bytes_committed=self._suffix_bytes_committed,
                scratch_peak_bytes=self._scratch_peak_bytes,
                active_scratches=self._active_scratches,
            )


__all__ = [
    "MLX_CHUNKED_PREFILL_NUMERICAL_CONTRACT",
    "MLX_COMPATIBLE_BATCH_ABI",
    "MLX_COMPATIBLE_BATCH_NUMERICAL_CONTRACT",
    "MLX_GREEDY_BLOCK_ABI",
    "MLX_GREEDY_PROPOSAL_ABI",
    "MLX_PREFILL_EXECUTION_SHAPE_ABI",
    "FixedMlxBatchKVCache",
    "FixedMlxKVCache",
    "MlxBatchProvisionalAuthority",
    "MlxCompatibleBatchLane",
    "MlxCompatibleBatchTelemetry",
    "MlxGreedyBlockAuthority",
    "MlxGreedyProposalAuthority",
    "MlxLayerCacheSpec",
    "MlxNativeRuntime",
    "MlxNativeRuntimeError",
    "MlxNativeState",
    "MlxProvisionalAuthority",
    "MlxPrefillExecutionShape",
    "MlxStateLayout",
    "inspect_mlx_state_layout",
]
