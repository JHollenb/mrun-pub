"""Transactional fixed-state MLX runtime for original Mamba causal language models.

Transformer K/V and a selective-state-space recurrence have different physical laws.  K/V grows
with every committed token; Mamba retains one convolution window and one recurrence per layer.
This runtime therefore keeps recurrent state out of the K/V arena implementation and gives it an
independent, fixed-per-row ABI.

Committed cache containers are never passed to a model forward.  MLX's immutable committed arrays
may be shared as read-only inputs by fresh scratch containers; mlx-lm's Mamba block replaces the
scratch entries with newly computed arrays.  Full-prefix commit installs that scratch, abandon
drops it, and a partial-prefix commit deterministically replays only the accepted tokens from the
unchanged committed parent.  The latter is deliberately rare but makes the generic accepted-prefix
contract exact without retaining one multi-megabyte recurrence snapshot per provisional token.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
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
    OutputRequest,
    PlacementPlan,
    PrefillWork,
    PromotionStatus,
    ProvisionalStep,
    RuntimeRoute,
    RuntimeTelemetry,
    StateForkResult,
    StateObservation,
    WorkloadSpec,
)
from .mlx_native import _sample_mlx_row
from .placement import validate_placement_plan


class MlxMambaRuntimeError(RuntimeError):
    """The recurrent runtime could not preserve its custody or transaction contract."""


MLX_MAMBA_PREFILL_EXECUTION_SHAPE_ABI = "mrun-mlx-mamba1-prefill-shape-v1"
MLX_MAMBA_CHUNK_WORKSPACE_ABI = "mrun-mlx-mamba1-conservative-tensor-workspace-v1"
MLX_MAMBA_CHUNKED_PREFILL_NUMERICAL_CONTRACT = (
    "mlx-source-component-f32-mamba1-chunked-sequential-selective-scan-v1"
)


@dataclass(frozen=True, slots=True)
class MlxMambaPrefillExecutionShape:
    """Route identity and admitted tensor workspace for bounded Mamba prefill segments."""

    chunk_size: int | None
    base_numerical_contract: str
    numerical_contract: str
    promotion_status: PromotionStatus
    activation_workspace_bytes: int = 0
    logits_workspace_bytes: int = 0
    execution_abi: str = MLX_MAMBA_PREFILL_EXECUTION_SHAPE_ABI
    workspace_abi: str = MLX_MAMBA_CHUNK_WORKSPACE_ABI

    def __post_init__(self) -> None:
        if self.execution_abi != MLX_MAMBA_PREFILL_EXECUTION_SHAPE_ABI:
            raise ValueError("unsupported MLX Mamba prefill execution-shape ABI")
        if self.workspace_abi != MLX_MAMBA_CHUNK_WORKSPACE_ABI:
            raise ValueError("unsupported MLX Mamba chunk-workspace ABI")
        _name(self.base_numerical_contract, "base_numerical_contract")
        _name(self.numerical_contract, "numerical_contract")
        if not isinstance(self.promotion_status, PromotionStatus):
            raise TypeError("Mamba prefill promotion_status must be PromotionStatus")
        for field_name in ("activation_workspace_bytes", "logits_workspace_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.chunk_size is None:
            if self.numerical_contract != self.base_numerical_contract:
                raise ValueError("unchunked Mamba prefill must preserve the base contract")
            if self.activation_workspace_bytes or self.logits_workspace_bytes:
                raise ValueError("unchunked Mamba prefill cannot claim chunk workspace")
        else:
            if (
                isinstance(self.chunk_size, bool)
                or not isinstance(self.chunk_size, int)
                or self.chunk_size <= 0
            ):
                raise ValueError("Mamba prefill chunk_size must be a positive integer or None")
            if self.numerical_contract == self.base_numerical_contract:
                raise ValueError("chunked Mamba prefill requires a distinct numerical contract")
            if self.promotion_status is not PromotionStatus.EXPERIMENTAL:
                raise ValueError("chunked Mamba prefill is not eligible for implicit promotion")
            if self.activation_workspace_bytes <= 0 or self.logits_workspace_bytes <= 0:
                raise ValueError("chunked Mamba prefill requires bounded activation/logit charges")

    @property
    def workspace_bytes(self) -> int:
        return self.activation_workspace_bytes + self.logits_workspace_bytes

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "execution_abi": self.execution_abi,
                "workspace_abi": self.workspace_abi,
                "chunk_size": self.chunk_size,
                "base_numerical_contract": self.base_numerical_contract,
                "numerical_contract": self.numerical_contract,
                "promotion_status": self.promotion_status.value,
                "activation_workspace_bytes": self.activation_workspace_bytes,
                "logits_workspace_bytes": self.logits_workspace_bytes,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


CacheTuple = tuple[Any, ...]
CacheFactory = Callable[[], Sequence[Any]]
CacheClone = Callable[[Sequence[Any]], Sequence[Any]]
CacheInstall = Callable[[Sequence[Any], Sequence[Any]], None]
CacheBytes = Callable[[Sequence[Any]], int]
CacheRelease = Callable[[Sequence[Any]], None]
StepExecutor = Callable[[tuple[int, ...], Sequence[Any], OutputRequest], int]
AdvanceExecutor = Callable[[tuple[int, ...], Sequence[Any]], None]
PrefillChunkExecutor = Callable[
    [tuple[int, ...], Sequence[Any], OutputRequest | None],
    int | None,
]


def _name(value: Any, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_config_int(config: Mapping[str, Any], field: str) -> int:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Mamba config {field!r} must be a positive integer")
    return value


def build_mlx_mamba_prefill_execution_shape(
    config: Mapping[str, Any],
    *,
    chunk_size: int | None,
    base_numerical_contract: str,
    unchunked_promotion_status: PromotionStatus,
) -> MlxMambaPrefillExecutionShape:
    """Build the canonical, conservative B1 tensor-workspace charge for one segment.

    mlx-lm's sequential scan builds several ``I×S`` intermediates per token before realization.
    The formula deliberately charges six such state-sized values plus the visible projections,
    residuals, convolution window, scan boundary and full F32 vocabulary result for every token in
    the terminal segment.  It is conservative relative to buffer reuse; allocator metadata and
    unrelated process memory remain the caller's explicit headroom.
    """

    _name(base_numerical_contract, "base_numerical_contract")
    if not isinstance(unchunked_promotion_status, PromotionStatus):
        raise TypeError("unchunked_promotion_status must be PromotionStatus")
    if chunk_size is None:
        return MlxMambaPrefillExecutionShape(
            chunk_size=None,
            base_numerical_contract=base_numerical_contract,
            numerical_contract=base_numerical_contract,
            promotion_status=unchunked_promotion_status,
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Mamba prefill chunk_size must be a positive integer or None")
    layers = _positive_config_int(config, "num_hidden_layers")
    hidden = _positive_config_int(config, "hidden_size")
    intermediate = _positive_config_int(config, "intermediate_size")
    state_size = _positive_config_int(config, "state_size")
    conv_kernel = _positive_config_int(config, "conv_kernel")
    time_rank = _positive_config_int(config, "time_step_rank")
    vocab_size = _positive_config_int(config, "vocab_size")

    per_token_per_layer = (
        4 * hidden
        + 12 * intermediate
        + 2 * (time_rank + 2 * state_size)
        + 6 * intermediate * state_size
    )
    boundary_per_layer = (conv_kernel - 1) * intermediate + 2 * intermediate * state_size
    activation_elements = (
        layers * (chunk_size * per_token_per_layer + boundary_per_layer) + 2 * chunk_size * hidden
    )
    # F32 model/state arithmetic is an admitted precondition; token IDs are int64.
    activation_workspace_bytes = 4 * activation_elements + 8 * chunk_size
    # The terminal model result is ``chunk×vocab`` F32.  Reserve another 128 bytes/vocabulary
    # entry for every visible tensor in the worst lazy sampling graph: adjustment indices and
    # values, adjusted/scaled/ordered/filtered score copies, conservative int64 order/rank
    # vectors, two softmax paths, cumulative vectors and their boolean masks.  Kernel-private
    # allocator state remains explicit service headroom; argmax uses less but shares this route.
    logits_workspace_bytes = (4 * chunk_size + 128) * vocab_size
    return MlxMambaPrefillExecutionShape(
        chunk_size=chunk_size,
        base_numerical_contract=base_numerical_contract,
        numerical_contract=MLX_MAMBA_CHUNKED_PREFILL_NUMERICAL_CONTRACT,
        promotion_status=PromotionStatus.EXPERIMENTAL,
        activation_workspace_bytes=activation_workspace_bytes,
        logits_workspace_bytes=logits_workspace_bytes,
    )


def _strict_single_count(values: Sequence[int], *, maximum: int) -> int:
    counts = tuple(values)
    if len(counts) != 1:
        raise ValueError("Mamba accepted_counts must contain exactly one row")
    accepted = counts[0]
    if isinstance(accepted, bool) or not isinstance(accepted, int):
        raise TypeError("accepted count must be an integer")
    if accepted < 0 or accepted > maximum:
        raise ValueError("accepted count lies outside the provisional token prefix")
    return accepted


def _array_state(cache: Any) -> tuple[Any, Any]:
    values = getattr(cache, "state", None)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise MlxMambaRuntimeError("Mamba layer cache must expose exactly two state arrays")
    conv, recurrent = values
    if conv is None or recurrent is None:
        raise MlxMambaRuntimeError("Mamba layer cache contains unallocated recurrent state")
    return conv, recurrent


def _default_cache_bytes(caches: Sequence[Any]) -> int:
    total = 0
    if not caches:
        raise MlxMambaRuntimeError("Mamba state requires at least one layer cache")
    for cache in caches:
        for value in _array_state(cache):
            nbytes = getattr(value, "nbytes", None)
            if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
                raise MlxMambaRuntimeError("Mamba state array has no valid physical byte size")
            total += int(nbytes)
    if total <= 0:
        raise MlxMambaRuntimeError("Mamba state has no positive physical byte size")
    return total


def _cache_arrays(caches: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(value for cache in caches for value in _array_state(cache))


def _array_storage_signature(value: Any) -> tuple[Any, ...]:
    """Seal one immutable MLX array without copying it or reading it back to the host."""

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    nbytes = getattr(value, "nbytes", None)
    if shape is None or dtype is None or isinstance(nbytes, bool) or not isinstance(nbytes, int):
        raise MlxMambaRuntimeError("Mamba state array has no stable structural identity")
    try:
        resolved_shape = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as exc:
        raise MlxMambaRuntimeError("Mamba state array shape is malformed") from exc
    if any(dimension < 0 for dimension in resolved_shape) or nbytes < 0:
        raise MlxMambaRuntimeError("Mamba state array geometry is malformed")
    return (id(value), resolved_shape, str(dtype), int(nbytes))


def _cache_payload_signature(caches: Sequence[Any]) -> tuple[Any, ...] | None:
    """Return an exact production-array seal, or ``None`` for an injected opaque test cache."""

    payload: list[tuple[Any, ...]] = []
    for cache in caches:
        values = getattr(cache, "state", None)
        if values is None:
            return None
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise MlxMambaRuntimeError("Mamba layer cache must expose exactly two state arrays")
        payload.append(tuple(_array_storage_signature(value) for value in values))
    return tuple(payload)


def _cache_storage_signature(caches: Sequence[Any]) -> tuple[Any, ...]:
    """Bind cache-container authority and, for ArraysCache, every immutable array identity."""

    resolved = tuple(caches)
    return (
        tuple((id(cache), type(cache).__module__, type(cache).__qualname__) for cache in resolved),
        _cache_payload_signature(resolved),
    )


def _fork_copy_bytes(
    source: Sequence[Any],
    forked: Sequence[Any],
    *,
    opaque_charge: int,
) -> int:
    """Measure newly owned recurrent arrays; opaque injected caches are charged conservatively."""

    if _cache_payload_signature(source) is None or _cache_payload_signature(forked) is None:
        return opaque_charge
    source_arrays = {id(value) for cache in source for value in _array_state(cache)}
    return sum(
        int(value.nbytes)
        for cache in forked
        for value in _array_state(cache)
        if id(value) not in source_arrays
    )


def _model_parameter_signature(model: Any) -> tuple[Any, ...] | None:
    """Seal the loaded immutable MLX parameter leaves without device-to-host reads."""

    if model is None:
        return None
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    root = parameters()
    leaves: list[tuple[Any, ...]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                visit(value[key], (*path, str(key)))
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))
            return
        if all(hasattr(value, attribute) for attribute in ("shape", "dtype", "nbytes")):
            leaves.append((*path, *_array_storage_signature(value)))
            return
        raise MlxMambaRuntimeError(
            f"Mamba parameter tree has an unsupported leaf at {'.'.join(path)!r}"
        )

    visit(root, ())
    if not leaves:
        raise MlxMambaRuntimeError("Mamba model exposes no immutable parameter leaves")
    return tuple(leaves)


class MlxMambaState:
    """Opaque B1 authority over one fixed-size committed Mamba recurrence."""

    __slots__ = (
        "_cache_bytes",
        "_caches",
        "_capacity",
        "_committed_length",
        "_container_signature",
        "_epoch",
        "_expected_bytes",
        "_generation",
        "_lock",
        "_owner_id",
        "_pending_authority",
        "_pending_step_id",
        "_released",
        "_runtime_id",
        "_state_abi",
        "_state_id",
        "_storage_signature",
        "_storage_generation",
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
        expected_bytes: int,
        cache_bytes: CacheBytes,
    ) -> None:
        self._runtime_id = runtime_id
        self._state_id = state_id
        self._owner_id = owner_id
        self._state_abi = state_abi
        self._capacity = capacity
        self._generation = generation
        self._caches = tuple(caches)
        self._expected_bytes = expected_bytes
        self._cache_bytes = cache_bytes
        self._container_signature = tuple(id(cache) for cache in self._caches)
        self._storage_signature = _cache_storage_signature(self._caches)
        self._committed_length = 0
        self._epoch = 0
        self._storage_generation = 0
        self._pending_step_id: str | None = None
        self._pending_authority: MlxMambaProvisionalAuthority | None = None
        self._released = False
        self._lock = threading.RLock()
        self._verify_storage_unlocked()

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
            raise MlxMambaRuntimeError("Mamba state authority has been released")
        if tuple(id(cache) for cache in self._caches) != self._container_signature:
            raise MlxMambaRuntimeError("Mamba committed cache containers changed identity")
        if _cache_storage_signature(self._caches) != self._storage_signature:
            raise MlxMambaRuntimeError("Mamba committed cache array identity changed")
        actual = self._cache_bytes(self._caches)
        if actual != self._expected_bytes:
            raise MlxMambaRuntimeError(
                "Mamba committed state differs from fixed-state accounting "
                f"({actual} != {self._expected_bytes})"
            )

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


class MlxMambaProvisionalAuthority:
    """Single-use authority over an isolated recurrent-state scratch."""

    __slots__ = (
        "_consumed",
        "_input_ids",
        "_issued_step",
        "_runtime_id",
        "_scratch",
        "_scratch_signature",
        "_state",
        "_step_id",
    )

    def __init__(
        self,
        *,
        runtime_id: str,
        step_id: str,
        state: MlxMambaState,
        input_ids: tuple[int, ...],
        scratch: Sequence[Any],
    ) -> None:
        self._runtime_id = runtime_id
        self._step_id = step_id
        self._state = state
        self._input_ids = input_ids
        self._scratch = tuple(scratch)
        self._scratch_signature = _cache_storage_signature(self._scratch)
        self._issued_step: ProvisionalStep | None = None
        self._consumed = False

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def step_id(self) -> str:
        return self._step_id


class MlxMambaRuntime:
    """B1 native MLX executor with fixed recurrent state and exact transactions."""

    def __init__(
        self,
        engine: Any,
        *,
        route: RuntimeRoute,
        placement: PlacementPlan,
        semantic_token_count: int,
        state_abi: str,
        cache_factory: CacheFactory | None = None,
        cache_clone: CacheClone | None = None,
        cache_install: CacheInstall | None = None,
        cache_bytes: CacheBytes | None = None,
        cache_release: CacheRelease | None = None,
        executor: StepExecutor | None = None,
        advance_executor: AdvanceExecutor | None = None,
        prefill_execution_shape: MlxMambaPrefillExecutionShape | None = None,
        prefill_chunk_executor: PrefillChunkExecutor | None = None,
        max_decode_tokens: int = 1,
        owns_engine: bool = False,
    ) -> None:
        if route.placement_fingerprint != placement.fingerprint:
            raise ValueError("Mamba route does not bind the supplied placement")
        if route.model_fingerprint != placement.model_fingerprint:
            raise ValueError("Mamba route and placement bind different models")
        if route.backend_id != placement.backend_id or route.device_id != placement.device_id:
            raise ValueError("Mamba route and placement bind different backend/device identities")
        if placement.state.state_abi != state_abi:
            raise ValueError("Mamba runtime state ABI differs from placement")
        if placement.state.bytes_per_token != 0 or placement.state.fixed_bytes_per_row <= 0:
            raise ValueError("Mamba placement must use only fixed per-row recurrent state")
        if placement.workspace_bytes < 2 * placement.state.fixed_bytes_per_row:
            raise ValueError("Mamba placement omits the two exact transaction scratch states")
        if str(getattr(engine, "backend", "")) != route.backend_id:
            raise ValueError("Mamba engine backend differs from its route")
        if str(getattr(engine, "arch", "")) != "mamba":
            raise ValueError("fixed-state MLX runtime requires a Mamba engine")
        if (
            isinstance(semantic_token_count, bool)
            or not isinstance(semantic_token_count, int)
            or semantic_token_count <= 0
        ):
            raise ValueError("semantic_token_count must be a positive integer")
        if int(getattr(engine, "semantic_token_count", 0)) != semantic_token_count:
            raise ValueError("Mamba engine token domain differs from the compiled model")
        if int(getattr(engine, "context_size", 0)) < placement.state.max_context_tokens:
            raise ValueError("Mamba service context limit is smaller than placement")
        if type(owns_engine) is not bool:
            raise TypeError("owns_engine must be boolean")
        if (
            isinstance(max_decode_tokens, bool)
            or not isinstance(max_decode_tokens, int)
            or max_decode_tokens <= 0
        ):
            raise ValueError("max_decode_tokens must be a positive integer")

        self._engine = engine
        self._placement = placement
        self._semantic_token_count = semantic_token_count
        self._state_abi = state_abi
        config = self._engine.artifact.config
        self._expected_layer_count = int(config["num_hidden_layers"])
        self._expected_intermediate_size = int(config["intermediate_size"])
        self._expected_state_size = int(config["state_size"])
        self._expected_conv_kernel = int(config["conv_kernel"])
        base_numerical_contract = str(getattr(self._engine, "numerical_contract", ""))
        if prefill_execution_shape is not None and not isinstance(
            prefill_execution_shape,
            MlxMambaPrefillExecutionShape,
        ):
            raise TypeError("prefill_execution_shape must be a MlxMambaPrefillExecutionShape")
        expected_prefill_shape = build_mlx_mamba_prefill_execution_shape(
            config,
            chunk_size=(
                None if prefill_execution_shape is None else prefill_execution_shape.chunk_size
            ),
            base_numerical_contract=base_numerical_contract,
            unchunked_promotion_status=route.promotion_status,
        )
        if prefill_execution_shape is None:
            prefill_execution_shape = expected_prefill_shape
        elif (
            not isinstance(prefill_execution_shape, MlxMambaPrefillExecutionShape)
            or prefill_execution_shape != expected_prefill_shape
            or prefill_execution_shape.fingerprint != expected_prefill_shape.fingerprint
        ):
            raise ValueError("Mamba prefill shape differs from canonical config/workspace geometry")
        if (
            prefill_execution_shape.chunk_size is not None
            and prefill_execution_shape.chunk_size > placement.state.max_context_tokens
        ):
            raise ValueError("Mamba prefill chunk size exceeds the admitted context")
        if (
            prefill_execution_shape.chunk_size is not None
            and max_decode_tokens > prefill_execution_shape.chunk_size
        ):
            raise ValueError("Mamba decode verification width exceeds the chunk workspace")
        if prefill_chunk_executor is not None and prefill_execution_shape.chunk_size is None:
            raise ValueError("prefill_chunk_executor requires an explicit Mamba chunk shape")
        required_workspace = (
            2 * placement.state.fixed_bytes_per_row + prefill_execution_shape.workspace_bytes
        )
        if placement.workspace_bytes < required_workspace:
            raise ValueError(
                "Mamba placement omits transactional scratch or chunk tensor workspace"
            )
        if prefill_execution_shape.chunk_size is None:
            if route.effective_numerical_contract is not None:
                raise ValueError("unchunked Mamba route cannot carry a chunk execution identity")
        elif route.effective_numerical_contract is None:
            route = replace(
                route,
                promotion_status=PromotionStatus.EXPERIMENTAL,
                effective_numerical_contract=prefill_execution_shape.numerical_contract,
                execution_shape_fingerprint=prefill_execution_shape.fingerprint,
            )
        elif (
            route.promotion_status is not PromotionStatus.EXPERIMENTAL
            or route.effective_numerical_contract != prefill_execution_shape.numerical_contract
            or route.execution_shape_fingerprint != prefill_execution_shape.fingerprint
        ):
            raise ValueError("chunked Mamba route does not bind its exact execution shape")
        self._route = route
        self._prefill_execution_shape = prefill_execution_shape
        self._max_decode_tokens = max_decode_tokens
        self._engine_execution_identity = self._current_engine_execution_identity()
        self._model_parameter_signature = _model_parameter_signature(
            getattr(self._engine, "model", None)
        )
        self._cache_bytes = cache_bytes or _default_cache_bytes
        self._cache_factory = cache_factory or self._default_cache_factory
        self._cache_clone = cache_clone or self._default_cache_clone
        self._cache_install = cache_install or self._default_cache_install
        self._cache_release = cache_release or self._default_cache_release
        self._executor = executor or self._default_executor
        self._advance_executor = advance_executor or self._default_advance_executor
        self._prefill_chunk_executor = (
            prefill_chunk_executor or self._default_prefill_chunk_executor
        )
        self._owns_engine = owns_engine
        self._states: dict[str, MlxMambaState] = {}
        self._next_generation = 1
        self._lock = threading.RLock()
        # The admitted workspace contains exactly one full transaction scratch and one optional
        # accepted-prefix replay scratch.  Keep at most one provisional transaction outstanding
        # so concurrent callers cannot silently multiply that physical reservation.
        self._scratch_slot = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._chunk_lock = threading.Lock()
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
        self._prefix_replay_forwards = 0
        self._prefix_replay_tokens = 0
        self._state_forks = 0
        self._state_fork_tokens = 0
        self._state_fork_bytes_copied = 0
        self._cleanup_failures = 0
        self._chunked_prefill_calls = 0
        self._prefill_chunks = 0
        self._prefill_chunk_failures = 0
        self._prefix_replay_chunks = 0

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
        owns_engine: bool = False,
        **runtime_options: Any,
    ) -> MlxMambaRuntime:
        custody_guard = getattr(engine, "assert_content_identity_unchanged", None)
        if not callable(custody_guard):
            raise ValueError("Mamba runtime binding requires an executable custody guard")
        custody_guard()
        validate_placement_plan(placement, model, workload, capabilities, device)
        if model.architecture != "mamba" or str(getattr(engine, "arch", "")) != "mamba":
            raise ValueError("Mamba runtime binding requires matching Mamba identities")
        if model.state_bytes_per_token != 0 or model.state_fixed_bytes_per_row <= 0:
            raise ValueError("compiled Mamba identity lacks fixed recurrent-state accounting")
        if workload.max_batch_size != 1:
            raise NotImplementedError("transactional Mamba runtime currently requires B1")
        if workload.output_mode not in (
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ):
            raise NotImplementedError("Mamba runtime promotes native next-token output only")
        if str(getattr(engine, "numerical_contract", "")) != workload.numerical_contract:
            raise ValueError("Mamba numerical contract differs from admitted workload")
        route = RuntimeRoute(
            runtime_id=f"mlx-mamba-{uuid4().hex}",
            model_fingerprint=model.fingerprint,
            capability_fingerprint=capabilities.fingerprint,
            placement_fingerprint=placement.fingerprint,
            backend_id=capabilities.backend_id,
            device_id=device.device_id,
            promotion_status=capabilities.promotion_status,
        )
        admitted_decode_tokens = max(1, workload.verify_tokens)
        requested_decode_tokens = runtime_options.get(
            "max_decode_tokens",
            admitted_decode_tokens,
        )
        if (
            isinstance(requested_decode_tokens, bool)
            or not isinstance(requested_decode_tokens, int)
            or requested_decode_tokens <= 0
            or requested_decode_tokens > admitted_decode_tokens
        ):
            raise ValueError("max_decode_tokens exceeds the admitted verification width")
        runtime_options["max_decode_tokens"] = requested_decode_tokens
        return cls(
            engine,
            route=route,
            placement=placement,
            semantic_token_count=model.semantic_token_count,
            state_abi=model.state_abi,
            owns_engine=owns_engine,
            **runtime_options,
        )

    @property
    def route(self) -> RuntimeRoute:
        return self._route

    @property
    def prefill_execution_shape(self) -> MlxMambaPrefillExecutionShape:
        return self._prefill_execution_shape

    def _default_cache_factory(self) -> CacheTuple:
        mx = self._engine._mx
        caches = tuple(self._engine.model.make_cache())
        if len(caches) != self._expected_layer_count:
            raise MlxMambaRuntimeError("Mamba model cache layer count differs from artifact")
        for cache in caches:
            cache[0] = mx.zeros(
                (1, self._expected_conv_kernel - 1, self._expected_intermediate_size),
                dtype=mx.float32,
            )
            cache[1] = mx.zeros(
                (1, self._expected_intermediate_size, self._expected_state_size),
                dtype=mx.float32,
            )
        mx.eval(*_cache_arrays(caches))
        return caches

    def _default_cache_clone(self, caches: Sequence[Any]) -> CacheTuple:
        clones = tuple(self._engine.model.make_cache())
        if len(clones) != len(caches):
            raise MlxMambaRuntimeError("Mamba cache clone changed the layer count")
        for target, source in zip(clones, caches, strict=True):
            conv, recurrent = _array_state(source)
            # MLX arrays are immutable.  Sharing the committed inputs is safe because the Mamba
            # block replaces each ArraysCache entry with newly computed arrays.
            target.state = [conv, recurrent]
        return clones

    @staticmethod
    def _default_cache_install(targets: Sequence[Any], sources: Sequence[Any]) -> None:
        if len(targets) != len(sources):
            raise MlxMambaRuntimeError("Mamba state install changed the layer count")
        replacements = [list(_array_state(source)) for source in sources]
        previous = [list(_array_state(target)) for target in targets]
        try:
            for target, replacement in zip(targets, replacements, strict=True):
                target.state = replacement
        except BaseException:
            for target, original in zip(targets, previous, strict=True):
                target.state = original
            raise

    @staticmethod
    def _default_cache_release(caches: Sequence[Any]) -> None:
        for cache in caches:
            if hasattr(cache, "state"):
                cache.state = [None, None]

    def _default_executor(
        self,
        ids: tuple[int, ...],
        caches: Sequence[Any],
        output: OutputRequest,
    ) -> int:
        mx = self._engine._mx
        inputs = mx.array(np.asarray(ids, dtype=np.int64))[None, :]
        logits = self._engine.model(inputs, cache=caches)
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
            raise NotImplementedError("Mamba executor supports only next-token output")
        mx.eval(selected, *_cache_arrays(caches))
        value = int(selected.item())
        if value < 0 or value >= self._semantic_token_count:
            raise MlxMambaRuntimeError("Mamba selection escaped the semantic token domain")
        return value

    def _default_advance_executor(self, ids: tuple[int, ...], caches: Sequence[Any]) -> None:
        mx = self._engine._mx
        inputs = mx.array(np.asarray(ids, dtype=np.int64))[None, :]
        self._engine.model(inputs, cache=caches)
        mx.eval(*_cache_arrays(caches))

    def _default_prefill_chunk_executor(
        self,
        ids: tuple[int, ...],
        caches: Sequence[Any],
        output: OutputRequest | None,
    ) -> int | None:
        """Advance one bounded scratch segment; only the terminal segment selects a token."""

        mx = self._engine._mx
        inputs = mx.array(np.asarray(ids, dtype=np.int64))[None, :]
        logits = self._engine.model(inputs, cache=caches)
        cache_arrays = _cache_arrays(caches)
        if output is None:
            mx.eval(*cache_arrays)
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
            raise NotImplementedError("chunked Mamba prefill supports next-token output only")
        mx.eval(selected, *cache_arrays)
        value = int(selected.item())
        if value < 0 or value >= self._semantic_token_count:
            raise MlxMambaRuntimeError("chunked Mamba selection escaped the token domain")
        return value

    def _execute_chunked_prefill(
        self,
        ids: tuple[int, ...],
        caches: Sequence[Any],
        output: OutputRequest,
    ) -> int:
        chunk_size = self._prefill_execution_shape.chunk_size
        if chunk_size is None:
            raise ValueError("chunked Mamba prefill requires an admitted segment shape")
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
                        raise MlxMambaRuntimeError(
                            "terminal Mamba prefill segment did not select one strict token"
                        )
                    selected = value
                elif value is not None:
                    raise MlxMambaRuntimeError(
                        "intermediate Mamba prefill segment selected an unexpected token"
                    )
                with self._chunk_lock:
                    self._prefill_chunks += 1
        except BaseException:
            with self._chunk_lock:
                self._prefill_chunk_failures += 1
            raise
        if selected is None or selected < 0 or selected >= self._semantic_token_count:
            with self._chunk_lock:
                self._prefill_chunk_failures += 1
            raise MlxMambaRuntimeError("chunked Mamba prefill produced no valid final token")
        with self._chunk_lock:
            self._chunked_prefill_calls += 1
        return selected

    def _execute_chunked_advance(
        self,
        ids: tuple[int, ...],
        caches: Sequence[Any],
    ) -> None:
        chunk_size = self._prefill_execution_shape.chunk_size
        if chunk_size is None:
            self._advance_executor(ids, caches)
            return
        try:
            for start in range(0, len(ids), chunk_size):
                value = self._prefill_chunk_executor(
                    ids[start : start + chunk_size],
                    caches,
                    None,
                )
                if value is not None:
                    raise MlxMambaRuntimeError("Mamba replay segment unexpectedly selected a token")
                with self._chunk_lock:
                    self._prefix_replay_chunks += 1
        except BaseException:
            with self._chunk_lock:
                self._prefill_chunk_failures += 1
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise MlxMambaRuntimeError("Mamba runtime is closed")

    def _current_engine_execution_identity(self) -> tuple[Any, ...]:
        artifact = getattr(self._engine, "artifact", None)
        config = getattr(artifact, "config", None)
        if not isinstance(config, Mapping):
            raise MlxMambaRuntimeError("Mamba engine artifact config is unavailable")
        return (
            id(getattr(self._engine, "model", None)),
            id(artifact),
            str(getattr(artifact, "artifact_sha256", "")),
            str(getattr(artifact, "source_dtype", "")),
            str(getattr(self._engine, "backend", "")),
            str(getattr(self._engine, "arch", "")),
            getattr(self._engine, "semantic_token_count", None),
            getattr(self._engine, "context_size", None),
            str(getattr(self._engine, "numerical_contract", "")),
            tuple(
                config.get(field)
                for field in (
                    "num_hidden_layers",
                    "hidden_size",
                    "intermediate_size",
                    "state_size",
                    "conv_kernel",
                    "time_step_rank",
                    "vocab_size",
                )
            ),
            bool(getattr(self._engine, "_closed", False)),
        )

    def _assert_engine_execution_unchanged(self, *, full_parameters: bool = False) -> None:
        if self._current_engine_execution_identity() != self._engine_execution_identity:
            raise MlxMambaRuntimeError("Mamba engine execution identity changed after binding")
        if (
            full_parameters
            and _model_parameter_signature(getattr(self._engine, "model", None))
            != self._model_parameter_signature
        ):
            raise MlxMambaRuntimeError("Mamba loaded parameter identity changed after binding")

    def _state(self, handle: Any) -> MlxMambaState:
        with self._lock:
            self._require_open()
            if not isinstance(handle, MlxMambaState):
                raise TypeError("state was not issued by the Mamba runtime")
            if handle.runtime_id != self._route.runtime_id:
                raise MlxMambaRuntimeError("Mamba state belongs to another runtime")
            if self._states.get(handle.state_id) is not handle:
                raise MlxMambaRuntimeError("Mamba state authority is stale or released")
            return handle

    def _release_best_effort(self, caches: Sequence[Any]) -> None:
        """Drop an obsolete cache without ever making transaction outcome ambiguous."""

        if not caches:
            return
        # Invalid injected factories/cloners may repeat one cache container.  Cleanup is a
        # terminal ownership action, so never invoke a release callback twice for one object.
        unique: list[Any] = []
        seen: set[int] = set()
        for cache in caches:
            cache_id = id(cache)
            if cache_id not in seen:
                seen.add(cache_id)
                unique.append(cache)
        try:
            self._cache_release(tuple(unique))
        except BaseException:
            # Cache cleanup is terminal: the authority referencing these containers has already
            # been invalidated (or was never published).  Re-raising here would make a committed
            # transaction look unsuccessful and encourage an unsafe retry.  Python reference
            # release still drops the containers; expose callback failures in telemetry.
            with self._cleanup_lock:
                self._cleanup_failures += 1

    def _checked_clone(
        self,
        source: Sequence[Any],
        clone: Sequence[Any],
        field: str,
    ) -> CacheTuple:
        resolved = tuple(clone)
        source_ids = {id(cache) for cache in source}
        try:
            checked = self._checked_caches(resolved, field)
            if any(id(cache) in source_ids for cache in checked):
                raise MlxMambaRuntimeError(f"{field} aliases a committed cache container")
            return checked
        except BaseException:
            disposable = tuple(cache for cache in resolved if id(cache) not in source_ids)
            self._release_best_effort(disposable)
            raise

    def _checked_caches(self, caches: Sequence[Any], field: str) -> CacheTuple:
        resolved = tuple(caches)
        if len({id(cache) for cache in resolved}) != len(resolved):
            raise MlxMambaRuntimeError(f"{field} repeats a cache container")
        actual = self._cache_bytes(resolved)
        expected = self._placement.state.fixed_bytes_per_row
        if actual != expected:
            raise MlxMambaRuntimeError(
                f"{field} differs from fixed-state accounting ({actual} != {expected})"
            )
        array_backed = tuple(hasattr(cache, "state") for cache in resolved)
        if any(array_backed) and not all(array_backed):
            raise MlxMambaRuntimeError(f"{field} mixes opaque and array-backed caches")
        if all(array_backed):
            if len(resolved) != self._expected_layer_count:
                raise MlxMambaRuntimeError(f"{field} changed the Mamba layer count")
            expected_shapes = (
                (1, self._expected_conv_kernel - 1, self._expected_intermediate_size),
                (1, self._expected_intermediate_size, self._expected_state_size),
            )
            for cache in resolved:
                values = _array_state(cache)
                for value, expected_shape in zip(values, expected_shapes, strict=True):
                    signature = _array_storage_signature(value)
                    if signature[1] != expected_shape or signature[2] != "mlx.core.float32":
                        raise MlxMambaRuntimeError(
                            f"{field} changed the registered F32 recurrent-state geometry"
                        )
        return resolved

    def allocate_state(
        self,
        *,
        owner_id: str,
        batch_size: int,
        capacity: int,
    ) -> MlxMambaState:
        owner_id = _name(owner_id, "owner_id")
        if isinstance(batch_size, bool) or batch_size != 1:
            raise ValueError("transactional Mamba state requires batch_size=1")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0 or capacity > self._placement.state.max_context_tokens:
            raise ValueError("Mamba capacity lies outside the admitted service limit")
        with self._lock:
            self._require_open()
            # State geometry and initial cache contents belong to the bound executable.  Refuse
            # allocation after a model/config swap instead of publishing an authority that can
            # only fail later at its first forward.
            self._assert_engine_execution_unchanged(full_parameters=True)
            caches: CacheTuple = ()
            try:
                caches = tuple(self._cache_factory())
                caches = self._checked_caches(caches, "allocated Mamba state")
                state_id = f"state-{uuid4().hex}"
                state = MlxMambaState(
                    runtime_id=self._route.runtime_id,
                    state_id=state_id,
                    owner_id=owner_id,
                    state_abi=self._state_abi,
                    capacity=capacity,
                    generation=0,
                    caches=caches,
                    expected_bytes=self._placement.state.fixed_bytes_per_row,
                    cache_bytes=self._cache_bytes,
                )
                self._assert_engine_execution_unchanged(full_parameters=True)
            except BaseException:
                if caches:
                    self._release_best_effort(caches)
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
        if not isinstance(parent, StateObservation):
            raise TypeError("Mamba fork parent must be a StateObservation")
        owner_id = _name(owner_id, "owner_id")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("fork capacity must be an integer")
        if capacity <= 0 or capacity > self._placement.state.max_context_tokens:
            raise ValueError("fork capacity lies outside the admitted service limit")
        resolved = self._state(source)
        with self._lock, resolved._lock:  # noqa: SLF001
            self._assert_engine_execution_unchanged(full_parameters=True)
            current = resolved._observe_unlocked()  # noqa: SLF001
            if current != parent:
                raise MlxMambaRuntimeError("Mamba fork parent is stale")
            if resolved._pending_step_id is not None:  # noqa: SLF001
                raise MlxMambaRuntimeError("cannot fork Mamba state with pending work")
            if current.lengths[0] > capacity:
                raise OverflowError("Mamba committed prefix exceeds fork capacity")
            caches: CacheTuple = ()
            try:
                caches = self._checked_clone(
                    resolved._caches,  # noqa: SLF001
                    self._cache_clone(resolved._caches),  # noqa: SLF001
                    "forked Mamba state",
                )
                state_id = f"state-{uuid4().hex}"
                generation = self._next_generation
                self._next_generation += 1
                forked_state = MlxMambaState(
                    runtime_id=self._route.runtime_id,
                    state_id=state_id,
                    owner_id=owner_id,
                    state_abi=self._state_abi,
                    capacity=capacity,
                    generation=generation,
                    caches=caches,
                    expected_bytes=self._placement.state.fixed_bytes_per_row,
                    cache_bytes=self._cache_bytes,
                )
            except BaseException:
                if caches:
                    self._release_best_effort(caches)
                raise
            forked_state._committed_length = current.lengths[0]  # noqa: SLF001
            forked_state._epoch = 1  # noqa: SLF001
            forked = forked_state._observe_unlocked()  # noqa: SLF001
            state_bytes_copied = _fork_copy_bytes(
                resolved._caches,  # noqa: SLF001
                forked_state._caches,  # noqa: SLF001
                opaque_charge=self._placement.state.fixed_bytes_per_row,
            )
            self._states[state_id] = forked_state
            self._state_forks += 1
            self._state_fork_tokens += current.lengths[0]
            self._state_fork_bytes_copied += state_bytes_copied
            return StateForkResult(
                runtime_id=self._route.runtime_id,
                source=current,
                forked=forked,
                state=forked_state,
                state_bytes_copied=state_bytes_copied,
            )

    def _execute(self, work: PrefillWork | DecodeWork, *, phase: str) -> ProvisionalStep:
        if work.output.mode not in (
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ):
            raise NotImplementedError("Mamba runtime supports only next-token output")
        # A full parameter-tree seal costs measurable Python time on every K1 decode.  Recheck it
        # at every prefill/fork boundary while keeping decode's per-token custody check O(1): MLX
        # parameter arrays are immutable and the model/root artifact identities remain sealed.
        self._assert_engine_execution_unchanged(full_parameters=phase == "prefill")
        if not self._scratch_slot.acquire(blocking=False):
            raise MlxMambaRuntimeError(
                "Mamba transaction scratch reservation already has provisional work"
            )
        published = False
        try:
            state = self._state(work.state)
            with state._lock:  # noqa: SLF001
                current = state._observe_unlocked()  # noqa: SLF001
                if current != work.parent:
                    raise MlxMambaRuntimeError("Mamba work parent is stale")
                if state._pending_step_id is not None:  # noqa: SLF001
                    raise MlxMambaRuntimeError("Mamba state already has provisional work")
                if len(work.token_rows) != 1:
                    raise NotImplementedError("transactional Mamba state is B1")
                ids = tuple(int(value) for value in work.token_rows[0])
                if any(value < 0 or value >= self._semantic_token_count for value in ids):
                    raise ValueError("input token IDs escape the Mamba semantic token domain")
                if phase == "decode" and len(ids) > self._max_decode_tokens:
                    raise MlxMambaRuntimeError(
                        "Mamba decode block exceeds the admitted verification width"
                    )
                scratch: CacheTuple = ()
                step_id = f"step-{uuid4().hex}"
                state._pending_step_id = step_id  # noqa: SLF001
                try:
                    scratch = self._checked_clone(
                        state._caches,  # noqa: SLF001
                        self._cache_clone(state._caches),  # noqa: SLF001
                        "Mamba transaction scratch",
                    )
                    started = time.perf_counter()
                    chunk_size = self._prefill_execution_shape.chunk_size
                    if phase == "prefill" and chunk_size is not None:
                        token = self._execute_chunked_prefill(ids, scratch, work.output)
                    else:
                        token = int(self._executor(ids, scratch, work.output))
                    elapsed = time.perf_counter() - started
                    self._checked_caches(scratch, "advanced Mamba transaction scratch")
                    self._assert_engine_execution_unchanged(full_parameters=phase == "prefill")
                    # A clone may share immutable MLX array inputs, but it may never replace the
                    # committed container payload while executing provisionally.
                    state._verify_storage_unlocked()  # noqa: SLF001
                    if token < 0 or token >= self._semantic_token_count:
                        raise MlxMambaRuntimeError("Mamba selection escaped the token domain")
                    authority = MlxMambaProvisionalAuthority(
                        runtime_id=self._route.runtime_id,
                        step_id=step_id,
                        state=state,
                        input_ids=ids,
                        scratch=scratch,
                    )
                    step = ProvisionalStep(
                        runtime_id=self._route.runtime_id,
                        step_id=step_id,
                        request_ids=work.request_ids,
                        state=state,
                        parent=current,
                        token_counts=(len(ids),),
                        output=NativeOutput(mode=work.output.mode, token_ids=(token,)),
                        authority=authority,
                    )
                    authority._issued_step = step  # noqa: SLF001
                    state._pending_authority = authority  # noqa: SLF001
                    published = True
                except BaseException:
                    if scratch:
                        self._release_best_effort(scratch)
                    state._pending_authority = None  # noqa: SLF001
                    state._pending_step_id = None  # noqa: SLF001
                    raise
        except BaseException:
            if not published:
                self._scratch_slot.release()
            raise
        with self._lock:
            self._provisional_steps += 1
            self._device_to_host_bytes += 8
            self._workspace_peak_bytes = max(
                self._workspace_peak_bytes,
                self._placement.state.fixed_bytes_per_row
                + self._prefill_execution_shape.workspace_bytes,
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

    def _authority(self, step: ProvisionalStep) -> MlxMambaProvisionalAuthority:
        if not isinstance(step, ProvisionalStep):
            raise TypeError("Mamba transaction requires a ProvisionalStep")
        authority = step.authority
        if type(authority) is not MlxMambaProvisionalAuthority:
            raise MlxMambaRuntimeError("step authority was not issued by the Mamba runtime")
        if (
            authority.runtime_id != self._route.runtime_id
            or step.runtime_id != self._route.runtime_id
            or authority.step_id != step.step_id
            or authority._state is not step.state  # noqa: SLF001
            or authority._issued_step is not step  # noqa: SLF001
            or authority._consumed  # noqa: SLF001
        ):
            raise MlxMambaRuntimeError("Mamba provisional authority is foreign or consumed")
        return authority

    def commit(
        self,
        step: ProvisionalStep,
        accepted_counts: Sequence[int],
    ) -> CommitResult:
        authority = self._authority(step)
        state = self._state(step.state)
        accepted = _strict_single_count(accepted_counts, maximum=step.token_counts[0])
        replay: CacheTuple = ()
        with self._lock, state._lock:  # noqa: SLF001
            self._require_open()
            if self._states.get(state.state_id) is not state:
                raise MlxMambaRuntimeError("Mamba state authority is stale or released")
            before = state._observe_unlocked()  # noqa: SLF001
            if (  # noqa: SLF001
                before != step.parent
                or state._pending_step_id != step.step_id
                or state._pending_authority is not authority
                or authority._consumed
                or authority._issued_step is not step
                or _cache_storage_signature(authority._scratch) != authority._scratch_signature
            ):
                raise MlxMambaRuntimeError("Mamba state no longer owns this provisional step")
            try:
                selected: Sequence[Any] | None
                if accepted == 0:
                    selected = None
                elif accepted == step.token_counts[0]:
                    selected = authority._scratch  # noqa: SLF001
                else:
                    # Replay must be computed by the exact parameter tree that produced the
                    # provisional continuation.  It is rare and already performs another model
                    # forward, so pay the full custody seal here rather than inheriting decode's
                    # intentionally O(1) fast-path check.
                    self._assert_engine_execution_unchanged(full_parameters=True)
                    replay = self._checked_clone(
                        state._caches,  # noqa: SLF001
                        self._cache_clone(state._caches),  # noqa: SLF001
                        "Mamba accepted-prefix replay scratch",
                    )
                    self._execute_chunked_advance(
                        authority._input_ids[:accepted],  # noqa: SLF001
                        replay,
                    )
                    self._assert_engine_execution_unchanged(full_parameters=True)
                    selected = self._checked_caches(
                        replay,
                        "advanced Mamba accepted-prefix replay",
                    )
                    state._verify_storage_unlocked()  # noqa: SLF001
                if selected is not None:
                    selected_payload = _cache_payload_signature(selected)
                    self._cache_install(state._caches, selected)  # noqa: SLF001
                    installed_payload = _cache_payload_signature(state._caches)  # noqa: SLF001
                    if selected_payload is not None and installed_payload != selected_payload:
                        raise MlxMambaRuntimeError(
                            "Mamba state install did not preserve the selected array identity"
                        )
                    state._storage_signature = _cache_storage_signature(  # noqa: SLF001
                        state._caches  # noqa: SLF001
                    )
                    state._verify_storage_unlocked()  # noqa: SLF001
            except BaseException:
                if replay:
                    self._release_best_effort(replay)
                raise
            state._committed_length += accepted  # noqa: SLF001
            state._epoch += 1  # noqa: SLF001
            state._pending_step_id = None  # noqa: SLF001
            state._pending_authority = None  # noqa: SLF001
            authority._consumed = True  # noqa: SLF001
            retired_scratch = authority._scratch  # noqa: SLF001
            authority._scratch = ()  # noqa: SLF001
            authority._issued_step = None  # noqa: SLF001
            after = state._observe_unlocked()  # noqa: SLF001
            self._commits += 1
            self._committed_tokens += accepted
            if 0 < accepted < step.token_counts[0]:
                self._prefix_replay_forwards += 1
                self._prefix_replay_tokens += accepted
                self._workspace_peak_bytes = max(
                    self._workspace_peak_bytes,
                    2 * self._placement.state.fixed_bytes_per_row
                    + self._prefill_execution_shape.workspace_bytes,
                )
        self._release_best_effort(retired_scratch)
        if replay:
            self._release_best_effort(replay)
        self._scratch_slot.release()
        return CommitResult(
            runtime_id=self._route.runtime_id,
            step_id=step.step_id,
            state_id=state.state_id,
            accepted_counts=(accepted,),
            before=before,
            after=after,
            state_bytes_written=(self._placement.state.fixed_bytes_per_row if accepted else 0),
        )

    def abandon(self, step: ProvisionalStep) -> None:
        authority = self._authority(step)
        state = self._state(step.state)
        with self._lock, state._lock:  # noqa: SLF001
            self._require_open()
            if self._states.get(state.state_id) is not state:
                raise MlxMambaRuntimeError("Mamba state authority is stale or released")
            if state._observe_unlocked() != step.parent:  # noqa: SLF001
                raise MlxMambaRuntimeError("Mamba state changed after provisional execution")
            if (  # noqa: SLF001
                state._pending_step_id != step.step_id
                or state._pending_authority is not authority
                or _cache_storage_signature(authority._scratch) != authority._scratch_signature
            ):
                raise MlxMambaRuntimeError("Mamba state does not own this pending step")
            retired_scratch = authority._scratch  # noqa: SLF001
            authority._scratch = ()  # noqa: SLF001
            authority._issued_step = None  # noqa: SLF001
            authority._consumed = True  # noqa: SLF001
            state._pending_authority = None  # noqa: SLF001
            state._pending_step_id = None  # noqa: SLF001
            state._verify_storage_unlocked()  # noqa: SLF001
            self._abandons += 1
        self._release_best_effort(retired_scratch)
        self._scratch_slot.release()

    def release_state(self, state: Any) -> None:
        resolved = self._state(state)
        with self._lock, resolved._lock:  # noqa: SLF001
            if resolved._pending_step_id is not None:  # noqa: SLF001
                raise MlxMambaRuntimeError("cannot release Mamba state with pending work")
            resolved._verify_storage_unlocked()  # noqa: SLF001
            if self._states.pop(resolved.state_id, None) is not resolved:
                raise MlxMambaRuntimeError("Mamba state was already released")
            retired = resolved._caches  # noqa: SLF001
            resolved._released = True  # noqa: SLF001
            resolved._caches = ()  # noqa: SLF001
        self._release_best_effort(retired)

    def _recurrent_residency_unlocked(self) -> tuple[int, int]:
        """Return physical immutable-array bytes and conservative logical state bytes."""

        logical = len(self._states) * self._placement.state.fixed_bytes_per_row
        arrays: dict[int, int] = {}
        for state in self._states.values():
            with state._lock:  # noqa: SLF001
                state._verify_storage_unlocked()  # noqa: SLF001
                payload = _cache_payload_signature(state._caches)  # noqa: SLF001
                if payload is None:
                    return logical, logical
                for cache in state._caches:  # noqa: SLF001
                    for value in _array_state(cache):
                        arrays.setdefault(id(value), int(value.nbytes))
        return sum(arrays.values()), logical

    def telemetry(self) -> RuntimeTelemetry:
        with self._lock:
            self._require_open()
            recurrent_bytes, recurrent_logical_bytes = self._recurrent_residency_unlocked()
            with self._cleanup_lock:
                cleanup_failures = self._cleanup_failures
            with self._chunk_lock:
                chunked_prefill_calls = self._chunked_prefill_calls
                prefill_chunks = self._prefill_chunks
                prefill_chunk_failures = self._prefill_chunk_failures
                prefix_replay_chunks = self._prefix_replay_chunks
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
                # The shared schema predates recurrent models.  Keep this compatibility counter
                # charged while publishing the unambiguous recurrent name below.
                kv_resident_bytes=recurrent_bytes,
                workspace_peak_bytes=self._workspace_peak_bytes,
                extra_counters=(
                    ("mamba_chunk_size", self._prefill_execution_shape.chunk_size or 0),
                    (
                        "mamba_chunk_tensor_workspace_bytes",
                        self._prefill_execution_shape.workspace_bytes,
                    ),
                    ("mamba_chunked_prefill_calls", chunked_prefill_calls),
                    ("mamba_prefill_chunk_failures", prefill_chunk_failures),
                    ("mamba_prefill_chunks", prefill_chunks),
                    ("mamba_prefix_replay_forwards", self._prefix_replay_forwards),
                    ("mamba_prefix_replay_chunks", prefix_replay_chunks),
                    ("mamba_prefix_replay_tokens", self._prefix_replay_tokens),
                    ("mamba_recurrent_state_logical_bytes", recurrent_logical_bytes),
                    ("mamba_recurrent_state_resident_bytes", recurrent_bytes),
                    ("mamba_state_fork_bytes_copied", self._state_fork_bytes_copied),
                    ("mamba_state_fork_tokens", self._state_fork_tokens),
                    ("mamba_state_forks", self._state_forks),
                    ("mamba_cleanup_failures", cleanup_failures),
                    ("transactional_batch", 1),
                ),
            )

    def close(self) -> None:
        retired: list[CacheTuple] = []
        with self._lock:
            if self._closed:
                return
            states = tuple(sorted(self._states.values(), key=lambda value: value.state_id))
            with ExitStack() as locks:
                for state in states:
                    locks.enter_context(state._lock)  # noqa: SLF001
                pending: list[str] = []
                for state in states:
                    if state._pending_step_id is not None:  # noqa: SLF001
                        pending.append(state.state_id)
                if pending:
                    raise MlxMambaRuntimeError(
                        f"cannot close Mamba runtime with pending states: {pending!r}"
                    )
                # Validate every authority before invalidating any of them.  This keeps close
                # all-or-nothing even when one cache was externally tampered.
                for state in states:
                    state._verify_storage_unlocked()  # noqa: SLF001
                for state in states:
                    retired.append(state._caches)  # noqa: SLF001
                    state._released = True  # noqa: SLF001
                    state._caches = ()  # noqa: SLF001
                self._states.clear()
                self._closed = True
        for caches in retired:
            self._release_best_effort(caches)
        if self._owns_engine:
            self._engine.close()


__all__ = [
    "MLX_MAMBA_CHUNKED_PREFILL_NUMERICAL_CONTRACT",
    "MLX_MAMBA_CHUNK_WORKSPACE_ABI",
    "MLX_MAMBA_PREFILL_EXECUTION_SHAPE_ABI",
    "MlxMambaPrefillExecutionShape",
    "MlxMambaProvisionalAuthority",
    "MlxMambaRuntime",
    "MlxMambaRuntimeError",
    "MlxMambaState",
    "build_mlx_mamba_prefill_execution_shape",
]
