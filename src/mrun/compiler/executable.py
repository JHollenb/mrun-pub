"""Fail-closed eager execution for lowered WorkPlan v2 schedules."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .graph_passes import GraphCompilation
from .identity import bind_loaded_qstore_identity
from .ir import DenseWorkPlan, DispatchBinding, ExecutionMode, OutputContract, WorkTemplate
from .lowering import (
    LoweredWorkPlan,
    LoweredWorkTemplate,
    lower_work_plan,
    lower_work_template,
)


def _state_shape(state: Any) -> tuple[int, tuple[int, ...], int, int]:
    try:
        epoch_value = state.epoch
        length_values = tuple(state.lengths)
    except (AttributeError, TypeError) as exc:
        raise TypeError("KV state must expose integer epoch and lengths") from exc
    if isinstance(epoch_value, bool) or not isinstance(epoch_value, Integral):
        raise TypeError("KV state epoch must be an integer")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in length_values):
        raise TypeError("KV state lengths must be an integer sequence")
    epoch = int(epoch_value)
    lengths = tuple(int(value) for value in length_values)
    capacity_value = getattr(state, "capacity", getattr(state, "max_seq_len", None))
    batch_value = getattr(state, "B", getattr(state, "batch_size", None))
    if isinstance(capacity_value, bool) or not isinstance(capacity_value, Integral):
        raise TypeError("KV state must expose integer capacity or max_seq_len")
    if isinstance(batch_value, bool) or not isinstance(batch_value, Integral):
        raise TypeError("KV state must expose integer B or batch_size")
    capacity = int(capacity_value)
    batch_size = int(batch_value)
    if epoch < 0 or capacity <= 0 or batch_size <= 0:
        raise ValueError("KV state epoch, capacity, and batch dimensions are invalid")
    if len(lengths) != batch_size:
        raise ValueError("KV state lengths do not match its batch dimension")
    if any(length < 0 or length > capacity for length in lengths):
        raise ValueError("KV state lengths are outside its capacity")
    return epoch, lengths, capacity, batch_size


def _state_handles(handles: Sequence[str]) -> tuple[str, ...]:
    if isinstance(handles, (str, bytes, bytearray)):
        raise TypeError("versioned KV state handles must be a sequence of strings")
    normalized = tuple(handles)
    if any(type(value) is not str for value in normalized):
        raise TypeError("versioned KV state handles must contain only strings")
    if not normalized or any(not value or value.strip() != value for value in normalized):
        raise ValueError("versioned KV state handles must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("versioned KV state handles must be unique")
    return normalized


def _canonical_torch_dtype(dtype: torch.dtype) -> str:
    names = {
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.int8: "int8",
        torch.uint8: "uint8",
    }
    float8 = getattr(torch, "float8_e4m3fn", None)
    if isinstance(float8, torch.dtype):
        names[float8] = "fp8"
    return names.get(dtype, str(dtype).removeprefix("torch."))


def _state_cache_id(state: Any) -> str:
    value = getattr(state, "cache_id", None)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("KV state must expose a non-empty immutable cache_id")
    return value


def _state_storage_tensors(state: Any) -> tuple[torch.Tensor, ...]:
    k = getattr(state, "k", None)
    v = getattr(state, "v", None)
    if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
        return (k, v)
    keys = getattr(state, "keys", None)
    values = getattr(state, "values", None)
    if isinstance(keys, (tuple, list)) and isinstance(values, (tuple, list)):
        tensors = (*keys, *values)
        if tensors and all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            return tuple(tensors)
    raise TypeError("KV state must expose paged k/v tensors or dense keys/values tensors")


def _tensor_signature(tensors: Sequence[torch.Tensor]) -> tuple[tuple[Any, ...], ...]:
    try:
        return tuple(
            (
                id(tensor),
                int(tensor.data_ptr()),
                tuple(int(value) for value in tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                bool(tensor.is_contiguous()),
                int(tensor._version),  # noqa: SLF001 - zero-copy mutation guard
            )
            for tensor in tensors
        )
    except RuntimeError as exc:
        raise TypeError("KV tensors must track mutation versions") from exc


def _tensor_storage_range(tensor: torch.Tensor) -> tuple[str, int, int]:
    if tensor.layout is not torch.strided:
        raise TypeError("KV tensors must use strided storage")
    start = int(tensor.untyped_storage().data_ptr()) + int(tensor.storage_offset()) * int(
        tensor.element_size()
    )
    end = start + int(tensor.numel()) * int(tensor.element_size())
    return str(tensor.device), start, end


def _require_nonoverlapping_tensors(
    tensors: Sequence[torch.Tensor],
    *,
    field: str,
) -> None:
    ranges = [_tensor_storage_range(tensor) for tensor in tensors]
    for index, (device, start, end) in enumerate(ranges):
        for other_device, other_start, other_end in ranges[index + 1 :]:
            if device == other_device and max(start, other_start) < min(end, other_end):
                raise ValueError(f"{field} tensors must not alias or overlap")


def _state_storage_signature(state: Any) -> tuple[tuple[Any, ...], ...]:
    return _tensor_signature(_state_storage_tensors(state))


def _validate_state_storage_for_plan(state: Any, plan: DenseWorkPlan) -> None:
    poisoned_reason = getattr(state, "_poisoned_reason", None)
    if poisoned_reason is not None:
        raise RuntimeError(f"KV state is poisoned: {poisoned_reason}")
    if "paged-transformer" in plan.structured_operator_ids:
        from ..engine.kernels.paged_forward import BatchedPagedKVCache

        if not isinstance(state, BatchedPagedKVCache):
            raise TypeError(
                "stateful paged WorkPlan execution requires BatchedPagedKVCache's "
                "scratch-then-commit transaction protocol"
            )
    metadata = dict(plan.metadata)
    layers = int(metadata["kv_num_layers"])
    batch = plan.shape.actual_batch
    capacity = int(metadata["kv_capacity"])
    kv_heads = int(metadata["kv_num_heads"])
    head_dim = int(metadata["kv_head_dim"])
    expected_dtype = str(metadata["kv_dtype"])
    tensors = _state_storage_tensors(state)
    if hasattr(state, "k"):
        expected_shape = (layers, batch, capacity, kv_heads, head_dim)
        if len(tensors) != 2 or any(tuple(tensor.shape) != expected_shape for tensor in tensors):
            raise ValueError(f"paged KV state tensors must have shape {expected_shape}")
    else:
        expected_shape = (batch, capacity, kv_heads, head_dim)
        if len(tensors) != layers * 2 or any(
            tuple(tensor.shape) != expected_shape for tensor in tensors
        ):
            raise ValueError(
                f"dense KV state must expose {layers} key/value tensors of shape {expected_shape}"
            )
    dtypes = {_canonical_torch_dtype(tensor.dtype) for tensor in tensors}
    if dtypes != {expected_dtype}:
        raise ValueError(f"KV state dtype {sorted(dtypes)!r} does not match {expected_dtype!r}")
    devices = {str(tensor.device) for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("KV state tensors must share one device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("KV state tensors must be contiguous")
    if any(tensor.requires_grad for tensor in tensors):
        raise ValueError("KV state tensors must not require gradients")
    _require_nonoverlapping_tensors(tensors, field="KV state")
    expected_device = str(metadata.get("engine_device", ""))
    actual_device = next(iter(devices))
    if not _same_runtime_device(actual_device, expected_device):
        raise ValueError("KV state device does not match the WorkPlan engine device")
    lock = getattr(state, "_lock", None)
    if not callable(getattr(lock, "acquire", None)) or not callable(getattr(lock, "release", None)):
        raise TypeError("KV state must expose a transaction lock")


@dataclass(frozen=True)
class VersionedKVStateBinding:
    """An immutable observation of one mutable committed KV arena.

    Identity, layout, and ordinary Torch mutation versions are pinned. Native execution assumes
    all writers honor the cache lock and Torch's mutation APIs; hostile foreign-memory or ``.data``
    writes outside that trusted boundary are not a supported concurrency model. Untrusted generic
    adapters receive the stronger byte-for-byte rollback check in :class:`_PagedDispatchRollback`.
    """

    state: Any = field(repr=False, compare=False)
    handles: tuple[str, ...]
    epoch: int
    lengths: tuple[int, ...]
    capacity: int
    cache_id: str
    storage_signature: tuple[tuple[Any, ...], ...]

    @property
    def parent_epoch(self) -> int:
        return self.epoch

    @property
    def parent_lengths(self) -> tuple[int, ...]:
        return self.lengths

    @classmethod
    def capture(cls, state: Any, handles: Sequence[str]) -> VersionedKVStateBinding:
        return bind_versioned_kv_state(state, handles)

    def assert_current(self) -> None:
        poisoned_reason = getattr(self.state, "_poisoned_reason", None)
        if poisoned_reason is not None:
            raise RuntimeError(f"versioned KV state is poisoned: {poisoned_reason}")
        epoch, lengths, capacity, batch_size = _state_shape(self.state)
        if len(self.handles) != batch_size:
            raise ValueError("KV binding handles do not match the state batch dimension")
        if (epoch, lengths, capacity) != (self.epoch, self.lengths, self.capacity):
            raise RuntimeError("versioned KV state binding is stale")
        if _state_storage_signature(self.state) != self.storage_signature:
            raise RuntimeError("versioned KV state backing storage changed")

    def validate_for_plan(self, plan: DenseWorkPlan) -> None:
        if plan.execution_mode not in {ExecutionMode.PREFILL, ExecutionMode.DECODE}:
            raise ValueError("versioned KV state can bind only a prefill/decode WorkPlan")
        self.assert_current()
        if self.handles != plan.kv_read_handles or self.handles != plan.kv_write_handles:
            raise ValueError("versioned KV state handles do not match the WorkPlan")
        if self.capacity != int(dict(plan.metadata)["kv_capacity"]):
            raise ValueError("versioned KV state capacity does not match the WorkPlan")
        if _state_cache_id(self.state) != self.cache_id:
            raise RuntimeError("versioned KV state cache identity changed")
        _validate_state_storage_for_plan(self.state, plan)
        if len(self.lengths) != plan.shape.actual_batch:
            raise ValueError("versioned KV state batch does not match the WorkPlan")
        if plan.execution_mode is ExecutionMode.PREFILL and any(self.lengths):
            raise ValueError("prefill requires an empty committed KV state")
        if plan.execution_mode is ExecutionMode.DECODE and any(
            length <= 0 for length in self.lengths
        ):
            raise ValueError("decode requires a committed prefix for every KV state row")
        if any(length + plan.shape.sequence_length > self.capacity for length in self.lengths):
            raise OverflowError("planned token block exceeds versioned KV state capacity")


@dataclass(frozen=True)
class ProvisionalKVDeltaBinding:
    """An uncommitted delta pinned to the exact state version it was derived from."""

    state: Any = field(repr=False, compare=False)
    handles: tuple[str, ...]
    delta: Any = field(repr=False, compare=False)
    plan_fingerprint: str
    parent_epoch: int
    parent_lengths: tuple[int, ...]
    token_count: int
    capacity: int
    cache_id: str
    state_storage_signature: tuple[tuple[Any, ...], ...]
    tensor_signature: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class _PagedDispatchRollback:
    """Private full-arena rollback for non-native stateful adapter implementations.

    The production :class:`PagedEngine` path is statically tied to the scratch-only kernel and
    does not pay this copy. Compatibility adapters are untrusted same-process code, so their
    committed arena is snapshotted before dispatch and restored before an auto-commit failure is
    surfaced. The restored cache remains poisoned because arbitrary adapter side effects outside
    this typed state object cannot be rolled back safely.
    """

    state: Any = field(repr=False)
    k: torch.Tensor = field(repr=False)
    v: torch.Tensor = field(repr=False)
    lengths: np.ndarray = field(repr=False)
    epoch: int
    capacity: int
    batch_size: int
    cache_id: str
    lock: Any = field(repr=False, compare=False)
    lock_order_key: str
    poisoned_reason: str | None
    slot_generations: tuple[int, ...]
    active_slot_leases: tuple[tuple[int, Any], ...]

    @classmethod
    def capture(cls, state: Any) -> _PagedDispatchRollback:
        from ..engine.kernels.paged_forward import BatchedPagedKVCache

        if not isinstance(state, BatchedPagedKVCache):
            raise TypeError("stateful compatibility rollback requires BatchedPagedKVCache")
        with torch.inference_mode(False):
            k = state.k.detach().clone(memory_format=torch.contiguous_format)
            v = state.v.detach().clone(memory_format=torch.contiguous_format)
        return cls(
            state=state,
            k=k,
            v=v,
            lengths=state.lengths.copy(),
            epoch=int(state.epoch),
            capacity=int(state.capacity),
            batch_size=int(state.B),
            cache_id=_state_cache_id(state),
            lock=state._lock,  # noqa: SLF001 - part of the typed cache protocol
            lock_order_key=str(state._lock_order_key),  # noqa: SLF001
            poisoned_reason=state._poisoned_reason,  # noqa: SLF001
            slot_generations=tuple(int(value) for value in state._slot_generations),  # noqa: SLF001
            active_slot_leases=tuple(state._active_slot_leases.items()),  # noqa: SLF001
        )

    def matches_current(self) -> bool:
        """Compare the complete typed cache protocol, including committed bytes.

        Tensor version counters catch ordinary Torch writes cheaply. Compatibility adapters are
        not trusted to obey that API, however, so their already-paid rollback snapshot is also an
        exact content witness. This detects raw ``.data`` writes that deliberately bypass
        ``Tensor._version`` without adding a full-arena copy to the native paged path.
        """

        state = self.state
        try:
            return bool(
                isinstance(state.k, torch.Tensor)
                and isinstance(state.v, torch.Tensor)
                and torch.equal(state.k, self.k)
                and torch.equal(state.v, self.v)
                and isinstance(state.lengths, np.ndarray)
                and np.array_equal(state.lengths, self.lengths)
                and state.epoch == self.epoch
                and state.capacity == self.capacity
                and state.B == self.batch_size
                and state.cache_id == self.cache_id
                and state._lock is self.lock  # noqa: SLF001
                and state._lock_order_key == self.lock_order_key  # noqa: SLF001
                and state._poisoned_reason == self.poisoned_reason  # noqa: SLF001
                and tuple(state._slot_generations) == self.slot_generations  # noqa: SLF001
                and tuple(state._active_slot_leases.items())  # noqa: SLF001
                == self.active_slot_leases
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def restore_and_poison(self, reason: str) -> None:
        state = self.state
        # Replace the adapter-visible tensor objects outright. This also recovers from an
        # adapter that resized or rebound the prior arena rather than merely writing into it.
        with torch.inference_mode(False):
            state.k = self.k.detach().clone(memory_format=torch.contiguous_format)
            state.v = self.v.detach().clone(memory_format=torch.contiguous_format)
        state.lengths = self.lengths.copy()
        state.epoch = self.epoch
        state.capacity = self.capacity
        state.B = self.batch_size
        state.cache_id = self.cache_id
        state._lock = self.lock  # noqa: SLF001
        state._lock_order_key = self.lock_order_key  # noqa: SLF001
        state._slot_generations = list(self.slot_generations)  # noqa: SLF001
        state._active_slot_leases = dict(self.active_slot_leases)  # noqa: SLF001
        state._poisoned_reason = reason  # noqa: SLF001


def _native_paged_scratch_executor(engine: Any) -> bool:
    """True only for the shipped, non-overridden PagedEngine transaction adapter."""

    from ..engine.paged import PagedEngine

    resolved_executor = getattr(engine, "execute_workplan_stateful", None)
    return (
        isinstance(engine, PagedEngine)
        and getattr(engine, "stateful_workplan_adapter_abi", None) == "mrun-paged-scratch-only-v1"
        and getattr(type(engine), "execute_workplan_stateful", None)
        is PagedEngine.execute_workplan_stateful
        and getattr(resolved_executor, "__func__", None) is PagedEngine.execute_workplan_stateful
    )


def bind_versioned_kv_state(
    state: Any,
    handles: Sequence[str],
) -> VersionedKVStateBinding:
    """Capture an explicit, fail-closed binding to the state's current version."""

    normalized_handles = _state_handles(handles)
    lock = getattr(state, "_lock", None)
    if not callable(getattr(lock, "acquire", None)) or not callable(getattr(lock, "release", None)):
        raise TypeError("KV state must expose a transaction lock")
    with lock:
        epoch, lengths, capacity, batch_size = _state_shape(state)
        if len(normalized_handles) != batch_size:
            raise ValueError("KV state requires one handle per batch row")
        return VersionedKVStateBinding(
            state=state,
            handles=normalized_handles,
            epoch=epoch,
            lengths=lengths,
            capacity=capacity,
            cache_id=_state_cache_id(state),
            storage_signature=_state_storage_signature(state),
        )


@dataclass(frozen=True)
class WorkPlanExecutionResult:
    plan_fingerprint: str
    executable_key: str
    output_contract: str
    outputs: Any
    evidence: dict[str, Any]
    provisional_delta: ProvisionalKVDeltaBinding | None = None


def _runtime_token_limit(engine: Any) -> int:
    engine_semantic = getattr(engine, "semantic_token_count", None)
    if isinstance(engine_semantic, Integral) and not isinstance(engine_semantic, bool):
        if int(engine_semantic) > 0:
            return int(engine_semantic)
    composite = getattr(engine, "composite_store", None)
    semantic = getattr(getattr(composite, "vocab", None), "token_count", None)
    if semantic is not None:
        limit = int(semantic)
    else:
        manifest = getattr(getattr(engine, "store", None), "man", None)
        config = manifest.get("config") if isinstance(manifest, Mapping) else None
        limit = int(config.get("vocab_size", 0)) if isinstance(config, Mapping) else 0
    if limit <= 0:
        raise RuntimeError("runtime engine has no positive semantic token limit")
    return limit


def _validate_inputs(
    engine: Any,
    plan: DenseWorkPlan,
    ids_list: Sequence[np.ndarray | Sequence[int]],
) -> list[np.ndarray]:
    raw_rows = [np.asarray(ids) for ids in ids_list]
    if any(row.dtype.kind not in {"i", "u"} for row in raw_rows):
        raise TypeError("runtime token IDs must be integer values, not coerced floats or booleans")
    rows = [row.astype(np.int64, copy=False) for row in raw_rows]
    if len(rows) != plan.shape.actual_batch:
        raise ValueError("runtime batch does not match the plan")
    if any(row.ndim != 1 or int(row.size) != plan.shape.sequence_length for row in rows):
        raise ValueError("runtime rows do not match the plan sequence length")
    runtime_limit = _runtime_token_limit(engine)
    if plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
        output_token_ids = plan.required_output_rows
    elif plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
        output_token_ids = tuple(
            token for candidates in plan.candidate_token_ids for token in candidates
        )
    else:
        output_token_ids = ()
    if any(token < 0 or token >= runtime_limit for token in output_token_ids):
        raise ValueError(
            f"WorkPlan output token IDs must be inside runtime semantic token space "
            f"[0, {runtime_limit})"
        )
    planned_limit = dict(plan.metadata).get("input_token_limit")
    if planned_limit is not None and int(planned_limit) != runtime_limit:
        raise RuntimeError("runtime semantic token limit does not match the WorkPlan")
    if any(row.size and ((row < 0).any() or (row >= runtime_limit).any()) for row in rows):
        raise ValueError(f"runtime token IDs must be inside [0, {runtime_limit})")
    return rows


def _validate_numerical_contract(engine: Any, plan: DenseWorkPlan) -> None:
    engine_contract = str(getattr(engine, "numerical_contract", plan.numerical_contract))
    supported_contracts = tuple(
        str(value) for value in getattr(engine, "supported_numerical_contracts", (engine_contract,))
    )
    if plan.numerical_contract not in supported_contracts:
        raise RuntimeError(
            f"engine numerical contract set {supported_contracts!r} does not match/include "
            f"plan contract {plan.numerical_contract!r}"
        )


def _validate_runtime_identity(engine: Any, plan: DenseWorkPlan) -> None:
    guard = getattr(engine, "assert_content_identity_unchanged", None)
    if not callable(guard):
        guard = getattr(getattr(engine, "store", None), "assert_content_identity_unchanged", None)
    if callable(guard):
        guard()
    identity = bind_loaded_qstore_identity(engine)
    if (
        plan.model_name != identity.model_name
        or plan.model_revision != identity.model_revision
        or plan.store_fingerprint != identity.store_fingerprint
    ):
        raise RuntimeError("runtime engine identity does not match the WorkPlan")


def _runtime_activation_dtype(engine: Any) -> str:
    store = getattr(engine, "store", None)
    value = str(getattr(store, "compute_dtype", "float32")).removeprefix("torch.")
    return {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
    }.get(value, value)


def _canonical_fabric(value: Any) -> str:
    device = str(value).lower()
    if device.startswith("cuda"):
        return "cuda"
    if device.startswith("mps"):
        return "mps"
    if device.startswith("cpu"):
        return "cpu"
    return device


def _same_runtime_device(actual: Any, expected: Any) -> bool:
    """Match exact accelerator ordinals while normalizing PyTorch's CPU index erasure."""

    actual_device = str(actual).lower()
    expected_device = str(expected).lower()
    actual_fabric = _canonical_fabric(actual_device)
    expected_fabric = _canonical_fabric(expected_device)
    return bool(
        actual_fabric == expected_fabric
        and (expected_fabric == "cpu" or actual_device == expected_device)
    )


def _runtime_paged_residency(engine: Any) -> tuple[str, dict[str, Any]]:
    composite = getattr(engine, "composite_store", None)
    if composite is not None:
        snapshot_method = getattr(composite, "snapshot", None)
        snapshot = snapshot_method() if callable(snapshot_method) else {}
        if not isinstance(snapshot, Mapping):
            raise TypeError("CompositeQStore snapshot must be an object")
        providers = getattr(composite, "_providers", {})
        if not isinstance(providers, Mapping) or "body" not in providers:
            raise RuntimeError("CompositeQStore runtime has no body provider")
        raw_budgets = snapshot.get("component_cache_budget_bytes")
        if isinstance(raw_budgets, Mapping):
            budgets = {str(role): int(value) for role, value in raw_budgets.items()}
        else:
            budgets = {
                str(role): max(0, int(getattr(provider.store, "_cache_budget", 0)))
                for role, provider in providers.items()
            }
        body = providers["body"].store
        policy = str(getattr(body, "_cache_policy", "unknown"))
        total_budget = snapshot.get("cache_budget_bytes", sum(budgets.values()))
        if isinstance(total_budget, bool) or not isinstance(total_budget, Integral):
            raise TypeError("CompositeQStore runtime cache budget must be an integer")
        ring_allocated = getattr(composite, "ring_allocated_bytes", None)
        return (
            f"component-aggregate-{policy}",
            {
                "weight_cache_budget_bytes": max(0, int(total_budget)),
                "weight_cache_policy": policy,
                "component_cache_budgets_json": json.dumps(
                    budgets,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "ring_staging_bytes": (int(ring_allocated()) if callable(ring_allocated) else 0),
            },
        )
    store = engine.store
    ring_allocated = getattr(store, "ring_allocated_bytes", None)
    policy = str(getattr(store, "_cache_policy", "unknown"))
    return (
        policy,
        {
            "weight_cache_budget_bytes": max(0, int(getattr(store, "_cache_budget", 0))),
            "weight_cache_policy": policy,
            "component_cache_budgets_json": "",
            "ring_staging_bytes": int(ring_allocated()) if callable(ring_allocated) else 0,
        },
    )


def _validate_runtime_configuration(
    engine: Any,
    plan: DenseWorkPlan,
    *,
    reported_fabric: str,
) -> dict[str, str]:
    """Bind execution precision, storage format, and fabric to the compiled plan."""

    store = getattr(engine, "store", None)
    manifest = getattr(store, "man", None)
    if not isinstance(manifest, Mapping):
        raise RuntimeError("runtime engine has no QStore manifest for configuration binding")
    runtime_activation_dtype = _runtime_activation_dtype(engine)
    runtime_weight_dtype = str(manifest.get("dtype", ""))
    runtime_config = manifest.get("config", {})
    runtime_output_rows = (
        int(runtime_config.get("vocab_size", 0)) if isinstance(runtime_config, Mapping) else 0
    )
    runtime_device = str(getattr(engine, "device", ""))
    runtime_fabric = _canonical_fabric(runtime_device)
    expected_device = str(dict(plan.metadata).get("engine_device", ""))
    expected_fabric = _canonical_fabric(expected_device)
    lowered_fabric = _canonical_fabric(reported_fabric)
    if runtime_activation_dtype != plan.precision.activation_dtype:
        raise RuntimeError(
            "runtime activation dtype does not match the WorkPlan "
            f"({runtime_activation_dtype!r} != {plan.precision.activation_dtype!r})"
        )
    if runtime_weight_dtype != plan.precision.weight_dtype:
        raise RuntimeError(
            "runtime QStore weight dtype does not match the WorkPlan "
            f"({runtime_weight_dtype!r} != {plan.precision.weight_dtype!r})"
        )
    if int(dict(plan.metadata).get("configured_output_row_count", 0)) != runtime_output_rows:
        raise RuntimeError("runtime configured output rows do not match the WorkPlan")
    if not expected_fabric or not _same_runtime_device(runtime_device, expected_device):
        raise RuntimeError(
            "runtime device does not match the WorkPlan "
            f"({runtime_device!r} != {expected_device!r})"
        )
    if runtime_fabric != lowered_fabric:
        raise RuntimeError(
            "runtime fabric does not match the lowered schedule "
            f"({runtime_fabric!r} != {lowered_fabric!r})"
        )
    if str(getattr(engine, "backend", "")) == "paged":
        runtime_cache_admission, runtime_residency = _runtime_paged_residency(engine)
        if runtime_cache_admission != plan.cache_admission:
            raise RuntimeError("runtime weight-cache policy does not match the WorkPlan")
        metadata = dict(plan.metadata)
        for key, value in runtime_residency.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"runtime {key} does not match the WorkPlan")
    return {
        "runtime_activation_dtype": runtime_activation_dtype,
        "runtime_weight_dtype": runtime_weight_dtype,
        "runtime_device": runtime_device,
        "runtime_fabric": runtime_fabric,
    }


def _validate_runtime_component_binding(
    engine: Any,
    plan: DenseWorkPlan,
) -> dict[str, Any]:
    metadata = dict(plan.metadata)
    expected_contract = metadata.get("component_output_contract")
    composite = getattr(engine, "composite_store", None)
    runtime_contract = getattr(engine, "component_output_contract", None)
    if expected_contract is None:
        if composite is not None or runtime_contract is not None:
            raise RuntimeError("runtime composite routing is absent from the WorkPlan")
        return {"runtime_component_binding": False}
    if composite is None:
        raise RuntimeError("WorkPlan requires a CompositeQStore runtime")
    if str(runtime_contract) != str(expected_contract):
        raise RuntimeError("runtime component output contract does not match the WorkPlan")
    assert_unchanged = getattr(composite, "assert_content_identity_unchanged", None)
    if not callable(assert_unchanged):
        raise RuntimeError("CompositeQStore runtime has no post-verification artifact guard")
    assert_unchanged()
    runtime_graph_fingerprint = getattr(
        composite,
        "composite_fingerprint_sha256",
        getattr(composite, "composite_graph_fingerprint", None),
    )
    runtime_vocab_fingerprint = getattr(composite, "vocab_manifest_sha256", None)
    if str(runtime_graph_fingerprint) != str(metadata.get("component_graph_fingerprint")):
        raise RuntimeError("runtime component graph fingerprint does not match the WorkPlan")
    if str(runtime_vocab_fingerprint) != str(metadata.get("vocab_manifest_sha256")):
        raise RuntimeError("runtime vocabulary manifest does not match the WorkPlan")
    return {
        "runtime_component_binding": True,
        "runtime_component_output_contract": str(runtime_contract),
        "runtime_component_graph_fingerprint": str(runtime_graph_fingerprint),
        "runtime_vocab_manifest_sha256": str(runtime_vocab_fingerprint),
    }


def _output_pushdown_available(
    engine: Any,
    plan: DenseWorkPlan,
) -> bool:
    contract = plan.output_contract
    method = {
        OutputContract.LAST_TOKEN_LOGITS: "last_logits_batch",
        OutputContract.SELECTED_TOKEN_ROWS: "selected_last_logits_batch",
        OutputContract.CANDIDATE_ARGMAX_AND_MARGIN: "candidate_logits_batch",
        OutputContract.HIDDEN_STATE_ONLY: "hidden_states_batch",
    }.get(contract)
    if contract is not OutputContract.HIDDEN_STATE_ONLY and not bool(
        dict(plan.metadata).get("output_pushdown", False)
    ):
        return False
    return method is not None and callable(getattr(engine, method, None))


def _candidate_outputs(
    logits: list[torch.Tensor],
    candidates: tuple[tuple[int, ...], ...],
) -> tuple[dict[str, Any], ...]:
    values = [
        row_logits[-1].index_select(
            0,
            torch.as_tensor(
                row_candidates,
                dtype=torch.long,
                device=row_logits.device,
            ),
        )
        for row_logits, row_candidates in zip(logits, candidates, strict=True)
    ]
    return candidate_outputs_from_values(values, candidates)


def candidate_outputs_from_values(
    values: Sequence[torch.Tensor],
    candidates: tuple[tuple[int, ...], ...],
) -> tuple[dict[str, Any], ...]:
    outputs: list[dict[str, Any]] = []
    for row_values, row_candidates in zip(values, candidates, strict=True):
        normalized = row_values.float()
        if not bool(torch.isfinite(normalized).all()):
            raise RuntimeError("candidate logits must be finite")
        top = torch.topk(normalized, k=2)
        winner_offset = int(top.indices[0])
        runner_offset = int(top.indices[1])
        if winner_offset == runner_offset or float(top.values[0]) < float(top.values[1]):
            raise RuntimeError("candidate winner/runner ordering is invalid")
        outputs.append(
            {
                "winner_token_id": int(row_candidates[winner_offset]),
                "runner_up_token_id": int(row_candidates[runner_offset]),
                "winner_logit": float(top.values[0]),
                "runner_up_logit": float(top.values[1]),
                "margin": float(top.values[0] - top.values[1]),
                "candidate_token_ids": tuple(int(token) for token in row_candidates),
            }
        )
    return tuple(outputs)


def _require_floating_output(
    value: Any,
    *,
    shape: tuple[int, ...],
    field: str,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tuple(tensor.shape) != shape:
        raise RuntimeError(f"{field} shape {tuple(tensor.shape)} does not match {shape}")
    if not tensor.dtype.is_floating_point:
        raise RuntimeError(f"{field} must use a floating-point dtype")
    return tensor


def _validate_score_logits(
    logits: Sequence[Any],
    plan: DenseWorkPlan,
) -> tuple[torch.Tensor, ...]:
    metadata = dict(plan.metadata)
    vocab_rows = int(metadata.get("configured_output_row_count", 0))
    if vocab_rows <= 0:
        raise RuntimeError("WorkPlan has no positive logical vocabulary width")
    expected = (plan.shape.sequence_length, vocab_rows)
    return tuple(
        _require_floating_output(value, shape=expected, field="score logits") for value in logits
    )


def execute_direct_contract(
    engine: Any,
    plan: DenseWorkPlan,
    ids_list: Sequence[np.ndarray | Sequence[int]],
    *,
    labels_list: Sequence[np.ndarray | Sequence[int]] | None = None,
) -> Any:
    """Execute only the plan's numerical/output contract on an established eager engine."""

    if plan.execution_mode is not ExecutionMode.SCORE:
        raise ValueError("stateful WorkPlans require an explicit VersionedKVStateBinding")
    _validate_runtime_identity(engine, plan)
    _validate_runtime_configuration(
        engine,
        plan,
        reported_fabric=_canonical_fabric(getattr(engine, "device", "")),
    )
    _validate_runtime_component_binding(engine, plan)
    rows = _validate_inputs(engine, plan, ids_list)
    _validate_numerical_contract(engine, plan)

    if plan.output_contract is OutputContract.HIDDEN_STATE_ONLY:
        states = list(engine.hidden_states_batch(rows))
        if len(states) != len(rows) or any(not row_states for row_states in states):
            raise RuntimeError("engine did not return hidden states for every input row")
        hidden_width = int(dict(plan.metadata).get("logical_hidden_width", 0))
        if hidden_width <= 0:
            raise RuntimeError("WorkPlan has no positive logical hidden width")
        final_states = [
            _require_floating_output(
                row_states[-1],
                shape=(plan.shape.sequence_length, hidden_width),
                field="hidden-state output",
            )
            for row_states in states
        ]
        return torch.stack(final_states)

    pushdown = bool(dict(plan.metadata).get("output_pushdown", False))
    last_logits_batch = getattr(engine, "last_logits_batch", None)
    if (
        pushdown
        and plan.output_contract is OutputContract.LAST_TOKEN_LOGITS
        and callable(last_logits_batch)
    ):
        value = torch.as_tensor(last_logits_batch(rows))
        width = int(dict(plan.metadata).get("logical_head_row_count", 0))
        return _require_floating_output(
            value,
            shape=(plan.shape.actual_batch, width),
            field="last-token logits",
        )
    selected_last_logits_batch = getattr(engine, "selected_last_logits_batch", None)
    if (
        pushdown
        and plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS
        and callable(selected_last_logits_batch)
    ):
        value = torch.as_tensor(selected_last_logits_batch(rows, plan.required_output_rows))
        return _require_floating_output(
            value,
            shape=(plan.shape.actual_batch, len(plan.required_output_rows)),
            field="selected-token logits",
        )
    candidate_logits_batch = getattr(engine, "candidate_logits_batch", None)
    if (
        pushdown
        and plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
        and callable(candidate_logits_batch)
    ):
        values = tuple(candidate_logits_batch(rows, plan.candidate_token_ids))
        if len(values) != plan.shape.actual_batch:
            raise RuntimeError("candidate backend returned the wrong batch size")
        checked = tuple(
            _require_floating_output(
                value,
                shape=(len(candidates),),
                field="candidate logits",
            )
            for value, candidates in zip(values, plan.candidate_token_ids, strict=True)
        )
        return candidate_outputs_from_values(checked, plan.candidate_token_ids)

    raw_logits = list(engine.logits_batch(rows))
    if len(raw_logits) != len(rows):
        raise RuntimeError("engine did not return one logits tensor per input row")
    logits = _validate_score_logits(raw_logits, plan)

    if plan.output_contract is OutputContract.FULL_LOGITS:
        return tuple(logits)
    if plan.output_contract is OutputContract.LAST_TOKEN_LOGITS:
        return torch.stack([row[-1] for row in logits])
    if plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
        return torch.stack(
            [
                row[-1].index_select(
                    0,
                    torch.as_tensor(
                        plan.required_output_rows,
                        dtype=torch.long,
                        device=row.device,
                    ),
                )
                for row in logits
            ]
        )
    if plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
        return _candidate_outputs(logits, plan.candidate_token_ids)
    if plan.output_contract is OutputContract.LOSS_ONLY:
        if labels_list is None:
            label_rows = rows
        else:
            label_rows = []
            token_limit = _runtime_token_limit(engine)
            for labels in labels_list:
                raw_labels = np.asarray(labels)
                if raw_labels.dtype.kind not in {"i", "u"}:
                    raise TypeError("loss labels must be integer token IDs")
                normalized_labels = raw_labels.astype(np.int64, copy=False)
                if normalized_labels.size and (
                    (normalized_labels < 0).any() or (normalized_labels >= token_limit).any()
                ):
                    raise ValueError("loss labels are outside the runtime token domain")
                label_rows.append(normalized_labels)
        if len(label_rows) != len(rows) or any(
            labels.ndim != 1 or labels.size != row.size
            for labels, row in zip(label_rows, rows, strict=True)
        ):
            raise ValueError("loss labels must match the planned batch and sequence shape")
        if plan.shape.sequence_length < 2:
            raise ValueError("loss_only requires a sequence length of at least two")
        losses = [
            F.cross_entropy(
                row_logits[:-1].float(),
                torch.as_tensor(labels[1:], dtype=torch.long, device=row_logits.device),
                reduction="sum",
            )
            for row_logits, labels in zip(logits, label_rows, strict=True)
        ]
        return torch.stack(losses).sum() / (
            plan.shape.actual_batch * (plan.shape.sequence_length - 1)
        )
    raise NotImplementedError(f"the eager adapter does not implement {plan.output_contract.value}")


def _bind_provisional_delta(
    state_binding: VersionedKVStateBinding,
    raw_delta: Any,
    *,
    plan: DenseWorkPlan,
    outputs: Any,
) -> ProvisionalKVDeltaBinding:
    try:
        parent_epoch_value = raw_delta.parent_epoch
        parent_lengths_value = raw_delta.parent_lengths
        raw_token_count_value = raw_delta.token_count
        raw_cache_id = raw_delta.cache_id
    except AttributeError as exc:
        raise TypeError(
            "stateful dispatch delta must expose parent_epoch, parent_lengths, cache_id, "
            "and token_count"
        ) from exc
    if isinstance(parent_epoch_value, bool) or not isinstance(parent_epoch_value, Integral):
        raise TypeError("stateful dispatch delta parent_epoch must be an integer")
    if isinstance(raw_token_count_value, bool) or not isinstance(raw_token_count_value, Integral):
        raise TypeError("stateful dispatch delta token_count must be an integer")
    try:
        raw_parent_lengths = tuple(parent_lengths_value)
    except TypeError as exc:
        raise TypeError(
            "stateful dispatch delta parent_lengths must be an integer sequence"
        ) from exc
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in raw_parent_lengths
    ):
        raise TypeError("stateful dispatch delta parent_lengths must be an integer sequence")
    parent_lengths = tuple(int(value) for value in raw_parent_lengths)
    parent_epoch = int(parent_epoch_value)
    token_count_value = int(raw_token_count_value)
    if parent_epoch != state_binding.epoch:
        raise RuntimeError("provisional KV delta parent epoch does not match its state binding")
    if parent_lengths != state_binding.lengths:
        raise RuntimeError("provisional KV delta parent lengths do not match its state binding")
    if token_count_value != plan.shape.sequence_length:
        raise RuntimeError("provisional KV delta token count does not match the WorkPlan")
    if not isinstance(raw_cache_id, str) or raw_cache_id != state_binding.cache_id:
        raise RuntimeError("provisional KV delta belongs to a different cache identity")
    metadata = dict(plan.metadata)
    layers = int(metadata["kv_num_layers"])
    batch = plan.shape.actual_batch
    token_count = plan.shape.sequence_length
    kv_heads = int(metadata["kv_num_heads"])
    head_dim = int(metadata["kv_head_dim"])
    expected_dtype = str(metadata["kv_dtype"])
    delta_k = getattr(raw_delta, "k", None)
    delta_v = getattr(raw_delta, "v", None)
    if isinstance(delta_k, torch.Tensor) and isinstance(delta_v, torch.Tensor):
        tensors = (delta_k, delta_v)
        expected_shape = (layers, batch, token_count, kv_heads, head_dim)
        if any(tuple(tensor.shape) != expected_shape for tensor in tensors):
            raise RuntimeError(f"paged provisional KV tensors must have shape {expected_shape}")
    else:
        keys = getattr(raw_delta, "keys", None)
        values = getattr(raw_delta, "values", None)
        if not isinstance(keys, (tuple, list)) or not isinstance(values, (tuple, list)):
            raise TypeError("provisional KV delta has no paged or dense tensor payload")
        tensors = (*keys, *values)
        expected_shape = (batch, token_count, kv_heads, head_dim)
        if (
            len(keys) != layers
            or len(values) != layers
            or any(
                not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape
                for tensor in tensors
            )
        ):
            raise RuntimeError(
                f"dense provisional KV delta must expose {layers} key/value tensors "
                f"of shape {expected_shape}"
            )
    if {_canonical_torch_dtype(tensor.dtype) for tensor in tensors} != {expected_dtype}:
        raise RuntimeError("provisional KV delta dtype does not match the WorkPlan")
    if len({str(tensor.device) for tensor in tensors}) != 1 or any(
        not tensor.is_contiguous() for tensor in tensors
    ):
        raise RuntimeError("provisional KV delta tensors must be contiguous on one device")
    if any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("provisional KV delta tensors must not require gradients")
    _require_nonoverlapping_tensors(tensors, field="provisional KV delta")
    state_tensors = _state_storage_tensors(state_binding.state)
    _require_nonoverlapping_tensors(
        (*state_tensors, *tensors),
        field="committed and provisional KV",
    )
    if isinstance(outputs, torch.Tensor):
        _require_nonoverlapping_tensors(
            (*state_tensors, *tensors, outputs),
            field="committed KV, provisional KV, and output scratch",
        )
    state_device = str(state_tensors[0].device)
    if str(tensors[0].device) != state_device:
        raise RuntimeError("provisional KV delta device does not match its bound cache")
    return ProvisionalKVDeltaBinding(
        state=state_binding.state,
        handles=state_binding.handles,
        delta=raw_delta,
        plan_fingerprint=plan.fingerprint,
        parent_epoch=parent_epoch,
        parent_lengths=parent_lengths,
        token_count=token_count_value,
        capacity=state_binding.capacity,
        cache_id=raw_cache_id,
        tensor_signature=_tensor_signature(tensors),
        state_storage_signature=state_binding.storage_signature,
    )


def _validate_stateful_outputs(plan: DenseWorkPlan, outputs: Any) -> None:
    batch = plan.shape.actual_batch
    token_count = plan.shape.sequence_length
    contract = plan.output_contract
    if contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
        if (
            not isinstance(outputs, tuple)
            or len(outputs) != batch
            or any(not isinstance(row, Mapping) for row in outputs)
        ):
            raise RuntimeError("stateful candidate output does not match the WorkPlan")
        expected_keys = {
            "winner_token_id",
            "runner_up_token_id",
            "winner_logit",
            "runner_up_logit",
            "margin",
            "candidate_token_ids",
        }
        for row, candidates in zip(outputs, plan.candidate_token_ids, strict=True):
            if set(row) != expected_keys:
                raise RuntimeError("stateful candidate output fields do not match the WorkPlan")
            candidate_values = row["candidate_token_ids"]
            if isinstance(candidate_values, (str, bytes, bytearray)) or not isinstance(
                candidate_values,
                Sequence,
            ):
                raise RuntimeError("stateful candidate IDs must be a non-string sequence")
            raw_returned_candidates = tuple(candidate_values)
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in raw_returned_candidates
            ):
                raise RuntimeError("stateful candidate IDs are malformed")
            returned_candidates = tuple(int(value) for value in raw_returned_candidates)
            if returned_candidates != candidates:
                raise RuntimeError("stateful candidate IDs do not match the WorkPlan")
            winner = row["winner_token_id"]
            runner = row["runner_up_token_id"]
            if (
                isinstance(winner, bool)
                or isinstance(runner, bool)
                or not isinstance(winner, Integral)
                or not isinstance(runner, Integral)
                or int(winner) not in candidates
                or int(runner) not in candidates
                or int(winner) == int(runner)
            ):
                raise RuntimeError("stateful candidate winners are invalid")
            numeric = (row["winner_logit"], row["runner_up_logit"], row["margin"])
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in numeric
            ):
                raise RuntimeError("stateful candidate logits must be finite numbers")
            expected_margin = float(row["winner_logit"]) - float(row["runner_up_logit"])
            if not math.isclose(float(row["margin"]), expected_margin, rel_tol=1e-6, abs_tol=1e-6):
                raise RuntimeError("stateful candidate margin is inconsistent")
            if expected_margin < 0:
                raise RuntimeError("stateful candidate winner must not trail the runner-up")
        return
    if not isinstance(outputs, torch.Tensor):
        raise RuntimeError("stateful tensor output must be a torch.Tensor")
    if outputs.requires_grad or outputs.grad_fn is not None:
        raise RuntimeError("stateful tensor output must be detached from autograd")
    if outputs.dtype is not torch.float32 or outputs.device.type != "cpu":
        raise RuntimeError("stateful tensor output must be CPU fp32")
    if not outputs.is_contiguous():
        raise RuntimeError("stateful tensor output must be contiguous")
    shape_value = getattr(outputs, "shape", None)
    try:
        shape = tuple(int(value) for value in shape_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("stateful tensor output has no concrete shape") from exc
    expected_prefix = {
        OutputContract.FULL_LOGITS: (batch, token_count),
        OutputContract.LAST_TOKEN_LOGITS: (batch,),
        OutputContract.SELECTED_TOKEN_ROWS: (batch,),
        OutputContract.HIDDEN_STATE_ONLY: (batch, token_count),
    }.get(contract)
    expected_rank = {
        OutputContract.FULL_LOGITS: 3,
        OutputContract.LAST_TOKEN_LOGITS: 2,
        OutputContract.SELECTED_TOKEN_ROWS: 2,
        OutputContract.HIDDEN_STATE_ONLY: 3,
    }.get(contract)
    if expected_prefix is None or expected_rank is None:
        raise NotImplementedError(f"stateful output contract {contract.value!r} is unavailable")
    if len(shape) != expected_rank or shape[: len(expected_prefix)] != expected_prefix:
        raise RuntimeError(
            f"stateful {contract.value} output shape {shape} does not match the WorkPlan"
        )
    metadata = dict(plan.metadata)
    expected_width = {
        OutputContract.FULL_LOGITS: int(metadata.get("logical_head_row_count", 0)),
        OutputContract.LAST_TOKEN_LOGITS: int(metadata.get("logical_head_row_count", 0)),
        OutputContract.SELECTED_TOKEN_ROWS: len(plan.required_output_rows),
        OutputContract.HIDDEN_STATE_ONLY: int(metadata.get("logical_hidden_width", 0)),
    }[contract]
    if expected_width <= 0 or shape[-1] != expected_width:
        raise RuntimeError(
            f"stateful {contract.value} output width {shape[-1]} does not match {expected_width}"
        )


def _accepted_counts(
    values: Sequence[int],
    *,
    batch_size: int,
    token_count: int,
) -> tuple[int, ...]:
    counts: list[int] = []
    for raw_value in values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, np.integer)):
            raise TypeError("accepted counts must be integers")
        counts.append(int(raw_value))
    normalized = tuple(counts)
    if len(normalized) != batch_size:
        raise ValueError("accepted count must be supplied for every KV state row")
    if any(value < 0 or value > token_count for value in normalized):
        raise ValueError("accepted count is outside the provisional token block")
    return normalized


def _commit_provisional_kv_delta_unlocked(
    state_binding: VersionedKVStateBinding,
    provisional_delta: ProvisionalKVDeltaBinding,
    accepted_counts: Sequence[int],
    *,
    plan: DenseWorkPlan,
) -> Any:
    """Explicitly commit an accepted prefix; execution itself never calls this helper."""

    if provisional_delta.state is not state_binding.state:
        raise ValueError("provisional KV delta belongs to a different state object")
    if provisional_delta.handles != state_binding.handles:
        raise ValueError("provisional KV delta handles do not match the state binding")
    if provisional_delta.cache_id != state_binding.cache_id:
        raise ValueError("provisional KV delta belongs to a different cache identity")
    if provisional_delta.state_storage_signature != state_binding.storage_signature:
        raise ValueError("provisional KV delta belongs to different committed KV storage")
    if provisional_delta.plan_fingerprint != plan.fingerprint:
        raise ValueError("provisional KV delta belongs to a different WorkPlan")
    raw_tensors = _state_storage_tensors(provisional_delta.delta)
    if _tensor_signature(raw_tensors) != provisional_delta.tensor_signature:
        raise RuntimeError("provisional KV delta tensors changed after execution")
    state_binding.validate_for_plan(plan)
    state_binding.assert_current()
    if (
        provisional_delta.parent_epoch != state_binding.epoch
        or provisional_delta.parent_lengths != state_binding.lengths
        or provisional_delta.capacity != state_binding.capacity
    ):
        raise RuntimeError("provisional KV delta does not belong to the current binding version")
    counts = _accepted_counts(
        accepted_counts,
        batch_size=len(state_binding.lengths),
        token_count=provisional_delta.token_count,
    )
    expected_lengths = tuple(
        length + count for length, count in zip(state_binding.lengths, counts, strict=True)
    )
    if any(length > state_binding.capacity for length in expected_lengths):
        raise OverflowError("accepted KV prefix exceeds state capacity")

    from ..engine.kernels.paged_forward import BatchedPagedKVCache, commit_block

    if not isinstance(state_binding.state, BatchedPagedKVCache):
        raise TypeError(
            "WorkPlan v2 commit requires BatchedPagedKVCache's validated atomic protocol"
        )
    result = commit_block(state_binding.state, provisional_delta.delta, counts)

    epoch, lengths, capacity, _batch_size = _state_shape(state_binding.state)
    if epoch != state_binding.epoch + 1 or lengths != expected_lengths:
        raise RuntimeError("KV commit did not produce the promised epoch/length transition")
    if capacity != state_binding.capacity:
        raise RuntimeError("KV commit changed the bound state capacity")
    return result


def commit_provisional_kv_delta(
    state_binding: VersionedKVStateBinding,
    provisional_delta: ProvisionalKVDeltaBinding,
    accepted_counts: Sequence[int],
    *,
    plan: DenseWorkPlan,
) -> Any:
    """Commit only the named plan's delta while holding the cache transaction lock."""

    lock = state_binding.state._lock  # noqa: SLF001 - state contract owns this lock
    with lock:
        return _commit_provisional_kv_delta_unlocked(
            state_binding,
            provisional_delta,
            accepted_counts,
            plan=plan,
        )


def execute_lowered_plan(
    engine: Any,
    plan: DenseWorkPlan,
    lowered: LoweredWorkPlan,
    ids_list: Sequence[np.ndarray | Sequence[int]],
    *,
    labels_list: Sequence[np.ndarray | Sequence[int]] | None = None,
    graph_compilation: GraphCompilation | None = None,
    state_binding: VersionedKVStateBinding | None = None,
) -> WorkPlanExecutionResult:
    """Execute the supported eager or captured subset; schedule-only lowerers refuse."""

    if lowered.plan_fingerprint != plan.fingerprint:
        raise ValueError("lowered schedule does not belong to this WorkPlan")
    canonical_lowering = lower_work_plan(plan, lowered.backend)
    canonical_template_lowering = lower_work_template(
        WorkTemplate.from_plan(plan),
        lowered.backend,
    ).bind(plan)
    if lowered not in {canonical_lowering, canonical_template_lowering}:
        raise ValueError("lowered schedule is not the canonical lowering of this WorkPlan")
    if lowered.implementation_status != "eager-adapter":
        raise RuntimeError(
            f"{lowered.backend} is {lowered.implementation_status}; no executable is available"
        )
    if lowered.capture_executed:
        raise RuntimeError("capture execution is a runtime verdict, not a serialized claim")
    stateful = plan.execution_mode in {ExecutionMode.PREFILL, ExecutionMode.DECODE}
    if stateful:
        if not isinstance(state_binding, VersionedKVStateBinding):
            raise TypeError("prefill/decode execution requires a VersionedKVStateBinding")
        if graph_compilation is not None:
            raise ValueError("prefill/decode OpGraph execution is not modeled in WorkPlan v2")
    elif state_binding is not None:
        raise ValueError("score execution does not accept a KV state binding")
    rewrite_ids: tuple[str, ...] = ()
    graph_fingerprint: str | None = None
    if graph_compilation is not None:
        graph = graph_compilation.graph
        if (
            graph.model_name != plan.model_name
            or graph.model_revision != plan.model_revision
            or graph.store_fingerprint != plan.store_fingerprint
            or graph.numerical_contract != plan.numerical_contract
        ):
            raise ValueError("graph compilation does not belong to this WorkPlan")
        certificate = graph_compilation.rewrite_certificate
        if certificate is None and plan.output_contract in {
            OutputContract.LAST_TOKEN_LOGITS,
            OutputContract.SELECTED_TOKEN_ROWS,
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            OutputContract.HIDDEN_STATE_ONLY,
        }:
            raise ValueError("optimized graph execution requires an output-demand certificate")
        if plan.output_contract in {
            OutputContract.LAST_TOKEN_LOGITS,
            OutputContract.SELECTED_TOKEN_ROWS,
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            OutputContract.HIDDEN_STATE_ONLY,
        } and not _output_pushdown_available(engine, plan):
            raise RuntimeError("engine cannot execute the certified output-demand graph")
        if certificate is not None:
            if certificate.output_contract != plan.output_contract.value:
                raise ValueError("graph rewrite output contract does not match this WorkPlan")
            rewrite_ids = certificate.rewrite_ids
        if plan.output_contract in {
            OutputContract.SELECTED_TOKEN_ROWS,
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        }:
            head_nodes = [node for node in graph.nodes if node.node_id == "output.vocab_projection"]
            if len(head_nodes) != 1 or not head_nodes[0].parameters:
                raise ValueError("output-sliced graph has no vocabulary projection")
            if head_nodes[0].parameters[0].access != "rows":
                raise ValueError("output-sliced graph did not bind row-addressable head access")
        graph_fingerprint = graph_compilation.fingerprint
    compatible_engines = {
        "cuda-qstore": {"dense-qstore-cuda"},
        "paged-qstore": {"paged"},
    }
    engine_backend = str(getattr(engine, "backend", "unknown"))
    if engine_backend not in compatible_engines.get(lowered.backend, {engine_backend}):
        raise RuntimeError(
            f"lowered backend {lowered.backend!r} is incompatible with engine {engine_backend!r}"
        )
    _validate_runtime_identity(engine, plan)
    runtime_configuration = _validate_runtime_configuration(
        engine,
        plan,
        reported_fabric=lowered.reported_fabric,
    )
    runtime_component_binding = _validate_runtime_component_binding(engine, plan)
    rows = _validate_inputs(engine, plan, ids_list)
    capture_executor: Any | None = None
    capture_metadata: Mapping[str, Any] | None = None
    provisional_delta: ProvisionalKVDeltaBinding | None = None
    if stateful:
        assert state_binding is not None
        if lowered.backend == "paged-qstore":
            from ..engine.kernels.paged_forward import BatchedPagedKVCache

            if not isinstance(state_binding.state, BatchedPagedKVCache):
                raise TypeError("paged-qstore stateful execution requires BatchedPagedKVCache")
        if plan.capture.requested:
            raise ValueError("prefill/decode execution cannot request graph capture")
        if labels_list is not None:
            raise ValueError("prefill/decode stateful dispatch does not accept labels")
        _validate_numerical_contract(engine, plan)
        state_binding.validate_for_plan(plan)
        execute_stateful = getattr(engine, "execute_workplan_stateful", None)
        if not callable(execute_stateful):
            raise RuntimeError("engine does not implement execute_workplan_stateful")
        engine_lock = getattr(engine, "_execution_lock", None)
        if not callable(getattr(engine_lock, "acquire", None)) or not callable(
            getattr(engine_lock, "release", None)
        ):
            raise TypeError("stateful engine must expose its execution transaction lock")
        lock = state_binding.state._lock  # noqa: SLF001 - state contract owns this lock
        # One global order for every supported entry point: engine lease, then cache lease.
        # Both are RLocks because the production engine/kernel also guard their public APIs.
        with engine_lock:
            with lock:
                # The binding was checked once before waiting for the transaction locks. Recheck
                # after both leases are held so a concurrent commit/rebind can never become the
                # state against which this request actually dispatches.
                state_binding.validate_for_plan(plan)
                state_before = _state_shape(state_binding.state)
                storage_before = _state_storage_signature(state_binding.state)
                cache_id_before = _state_cache_id(state_binding.state)
                rollback = (
                    None
                    if _native_paged_scratch_executor(engine)
                    else _PagedDispatchRollback.capture(state_binding.state)
                )
                try:
                    with torch.no_grad():
                        dispatched = execute_stateful(plan, rows, state_binding)
                finally:
                    try:
                        state_after = _state_shape(state_binding.state)
                        storage_after = _state_storage_signature(state_binding.state)
                        cache_id_after = _state_cache_id(state_binding.state)
                        mutated = (
                            state_after != state_before
                            or storage_after != storage_before
                            or cache_id_after != cache_id_before
                            or (rollback is not None and not rollback.matches_current())
                        )
                    except Exception:
                        mutated = True
                    if mutated:
                        reason = "stateful executor violated scratch-only dispatch"
                        if rollback is not None:
                            rollback.restore_and_poison(reason)
                        else:
                            state_binding.state._poisoned_reason = reason  # noqa: SLF001
                        raise RuntimeError(
                            "stateful WorkPlan dispatch mutated or auto-committed the "
                            "bound KV state"
                        )
        if not isinstance(dispatched, tuple) or len(dispatched) != 2:
            raise TypeError(
                "execute_workplan_stateful must return (contract_output, provisional_delta)"
            )
        outputs, raw_delta = dispatched
        _validate_stateful_outputs(plan, outputs)
        provisional_delta = _bind_provisional_delta(
            state_binding,
            raw_delta,
            plan=plan,
            outputs=outputs,
        )
    elif plan.capture.requested:
        if not lowered.capture_ready:
            raise RuntimeError("WorkPlan requested graph capture but its lowerer is not ready")
        if plan.output_contract is not OutputContract.SELECTED_TOKEN_ROWS:
            raise NotImplementedError(
                "captured WorkPlan execution currently requires selected_token_rows"
            )
        if labels_list is not None:
            raise ValueError("captured selected-row execution does not accept labels")
        prepare_capture = getattr(engine, "prepare_selected_last_cuda_graph", None)
        if not callable(prepare_capture):
            raise RuntimeError("engine cannot prepare the requested CUDA Graph executor")
        capture_executor = prepare_capture(rows, plan.required_output_rows)
        execute_capture = getattr(capture_executor, "execute", None)
        if not callable(execute_capture):
            close = getattr(capture_executor, "close", None)
            if callable(close):
                close()
            raise RuntimeError("CUDA Graph executor does not expose execute()")
        raw_capture_metadata = getattr(capture_executor, "evidence", None)
        if not isinstance(raw_capture_metadata, Mapping):
            close = getattr(capture_executor, "close", None)
            if callable(close):
                close()
            raise RuntimeError("CUDA Graph executor does not expose evidence")
        try:
            outputs = torch.as_tensor(execute_capture())
            raw_capture_metadata = getattr(capture_executor, "evidence", None)
            if not isinstance(raw_capture_metadata, Mapping):
                raise RuntimeError("CUDA Graph executor lost its runtime evidence")
            capture_metadata = dict(raw_capture_metadata)
        finally:
            close = getattr(capture_executor, "close", None)
            if callable(close):
                close()
    else:
        outputs = execute_direct_contract(
            engine,
            plan,
            rows,
            labels_list=labels_list,
        )

    return WorkPlanExecutionResult(
        plan_fingerprint=plan.fingerprint,
        executable_key=lowered.executable_key,
        output_contract=plan.output_contract.value,
        outputs=outputs,
        evidence={
            "implementation_status": lowered.implementation_status,
            "runtime_implementation_status": (
                "stateful-eager-adapter"
                if stateful
                else "cuda-graph"
                if capture_executor is not None
                else "eager-adapter"
            ),
            "engine_backend": engine_backend,
            "reported_fabric": runtime_configuration["runtime_fabric"],
            "graph_replay": capture_executor is not None,
            "capture_requested": plan.capture.requested,
            "capture_ready": lowered.capture_ready,
            "capture_executed": capture_executor is not None,
            "capture_metadata": capture_metadata,
            "placement_verified": lowered.placement_verified,
            "content_identity_verified": lowered.content_identity_verified,
            "runtime_identity_bound": True,
            "runtime_configuration_bound": True,
            **runtime_configuration,
            **runtime_component_binding,
            "actual_batch": len(rows),
            "sequence_length": plan.shape.sequence_length,
            "numerical_contract": plan.numerical_contract,
            "output_pushdown": _output_pushdown_available(engine, plan),
            "graph_compilation_fingerprint": graph_fingerprint,
            "graph_rewrite_ids": rewrite_ids,
            "graph_dispatch_verified": graph_compilation is not None,
            "state_binding_verified": stateful,
            "provisional_kv_emitted": provisional_delta is not None,
            "kv_parent_epoch": (
                None if provisional_delta is None else provisional_delta.parent_epoch
            ),
            "kv_parent_lengths": (
                None if provisional_delta is None else list(provisional_delta.parent_lengths)
            ),
            "kv_cache_id": None if provisional_delta is None else provisional_delta.cache_id,
            "provisional_plan_fingerprint": (
                None if provisional_delta is None else provisional_delta.plan_fingerprint
            ),
        },
        provisional_delta=provisional_delta,
    )


def execute_lowered_template(
    engine: Any,
    template: WorkTemplate,
    binding: DispatchBinding,
    lowered_template: LoweredWorkTemplate,
    ids_list: Sequence[np.ndarray | Sequence[int]],
    *,
    labels_list: Sequence[np.ndarray | Sequence[int]] | None = None,
    graph_compilation: GraphCompilation | None = None,
    state_binding: VersionedKVStateBinding | None = None,
) -> WorkPlanExecutionResult:
    """Attach and execute one dispatch without recompiling its structural schedule.

    The attachment is intentionally fail-closed: both the request binding and lowered
    artifact must name the exact template fingerprint, then the ordinary concrete executor
    revalidates model identity, runtime configuration, token domains, and mutable KV state.
    """

    lowered_template.verify_integrity()
    plan = template.bind(binding)
    lowered = lowered_template.bind(plan)
    result = execute_lowered_plan(
        engine,
        plan,
        lowered,
        ids_list,
        labels_list=labels_list,
        graph_compilation=graph_compilation,
        state_binding=state_binding,
    )
    return WorkPlanExecutionResult(
        plan_fingerprint=result.plan_fingerprint,
        executable_key=result.executable_key,
        output_contract=result.output_contract,
        outputs=result.outputs,
        evidence={
            **result.evidence,
            "work_template_fingerprint": template.fingerprint,
            "dispatch_binding_fingerprint": binding.fingerprint,
            "lowering_abi": lowered_template.lowering_abi,
            "lowered_template_artifact_sha256": lowered_template.artifact_sha256,
            "reused_lowered_template": True,
        },
        provisional_delta=result.provisional_delta,
    )
