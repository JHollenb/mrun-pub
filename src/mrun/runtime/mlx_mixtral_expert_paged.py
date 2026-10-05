"""Explicit MLX sparse-MoE slice backed by authenticated q4 expert pages.

The classes here are intentionally not registered in the default native loader.  This is a
model-weight/MoE-block slice, not a full serving runtime: attention, KV state, token IO, and a
full-model performance gate remain outside its claim boundary.  Its placement contract is
tiered and therefore separate from the fully resident :class:`PlacementPlan` contract.
"""

from __future__ import annotations

import platform
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from mrun.decompiler.mlx_mixtral_expert_store import (
    MIXTRAL_EXPERT_STORE_BITS,
    MIXTRAL_EXPERT_STORE_CODEC,
    MIXTRAL_EXPERT_STORE_GROUP_SIZE,
    MIXTRAL_EXPERT_STORE_MODE,
    MIXTRAL_EXPERT_STORE_NUMERICAL_CONTRACT,
    MIXTRAL_EXPERT_STORE_RUNTIME_ABI,
    VerifiedMixtralExpertStore,
)
from mrun.engine.mlx_component import _canonical_json_bytes, _sha256_bytes

MIXTRAL_EXPERT_PAGED_PLACEMENT_SCHEMA = "mrun-mlx-mixtral-tiered-placement-v1"
MIXTRAL_EXPERT_PAGED_HARDWARE_SCHEMA = "mrun-mlx-mixtral-hardware-evidence-v1"
MIXTRAL_EXPERT_PAGED_BACKEND_ID = "mlx-mixtral-q4-expert-paged-slice"


class MixtralExpertPagedRuntimeError(RuntimeError):
    """Base error for the explicit Mixtral expert-paged runtime slice."""


class MixtralExpertCacheError(MixtralExpertPagedRuntimeError):
    """The expert cache cannot complete an authenticated, bounded operation."""


class MixtralTieredPlacementError(MixtralExpertPagedRuntimeError):
    """The separate tiered weight-placement contract is invalid or cannot fit."""


class MixtralExpertPagedHardwareError(MixtralExpertPagedRuntimeError):
    """The live host cannot prove the exact Metal hardware contract for this slice."""


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _require_mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - exercised on non-Apple hosts
        raise MixtralExpertPagedRuntimeError(
            "the Mixtral expert-paged runtime requires the optional MLX dependency"
        ) from exc
    return mx


@dataclass(frozen=True, slots=True)
class MixtralExpertPagedHardwareEvidence:
    """Live, fingerprinted evidence for the only supported execution device."""

    platform_system: str
    machine: str
    mlx_version: str
    default_device: str
    metal_available: bool
    device_name: str
    architecture: str
    memory_size_bytes: int
    max_recommended_working_set_size_bytes: int
    max_buffer_length_bytes: int
    resource_limit: int
    schema_version: str = MIXTRAL_EXPERT_PAGED_HARDWARE_SCHEMA
    backend_id: str = MIXTRAL_EXPERT_PAGED_BACKEND_ID
    real_checkpoint_execution_certified: bool = False
    production_runtime_eligible: bool = False
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != MIXTRAL_EXPERT_PAGED_HARDWARE_SCHEMA:
            raise ValueError("Mixtral hardware-evidence schema is unsupported")
        if self.backend_id != MIXTRAL_EXPERT_PAGED_BACKEND_ID:
            raise ValueError("Mixtral hardware evidence belongs to another backend")
        if self.platform_system != "Darwin" or self.machine not in {"arm64", "aarch64"}:
            raise ValueError("Mixtral expert paging requires Apple silicon on Darwin")
        if not self.mlx_version or self.mlx_version.strip() != self.mlx_version:
            raise ValueError("Mixtral hardware evidence has no canonical MLX version")
        if "gpu" not in self.default_device.lower() or self.metal_available is not True:
            raise ValueError("Mixtral expert paging requires the live default Metal GPU")
        if not self.device_name or not self.architecture:
            raise ValueError("Mixtral hardware evidence is missing the Metal device identity")
        for field_name in (
            "memory_size_bytes",
            "max_recommended_working_set_size_bytes",
            "max_buffer_length_bytes",
            "resource_limit",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if self.max_recommended_working_set_size_bytes > self.memory_size_bytes:
            raise ValueError("recommended Metal working set exceeds physical unified memory")
        if self.max_buffer_length_bytes > self.memory_size_bytes:
            raise ValueError("Metal maximum buffer length exceeds physical unified memory")
        if self.real_checkpoint_execution_certified or self.production_runtime_eligible:
            raise ValueError("hardware evidence cannot promote this unverified runtime slice")
        object.__setattr__(
            self,
            "fingerprint",
            _sha256_bytes(_canonical_json_bytes(self.payload())),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_id": self.backend_id,
            "platform_system": self.platform_system,
            "machine": self.machine,
            "mlx_version": self.mlx_version,
            "default_device": self.default_device,
            "metal_available": self.metal_available,
            "device_name": self.device_name,
            "architecture": self.architecture,
            "memory_size_bytes": self.memory_size_bytes,
            "max_recommended_working_set_size_bytes": (self.max_recommended_working_set_size_bytes),
            "max_buffer_length_bytes": self.max_buffer_length_bytes,
            "resource_limit": self.resource_limit,
            "real_checkpoint_execution_certified": self.real_checkpoint_execution_certified,
            "production_runtime_eligible": self.production_runtime_eligible,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.payload(), "fingerprint": self.fingerprint}


def probe_mixtral_expert_paged_hardware() -> MixtralExpertPagedHardwareEvidence:
    """Probe real Metal state and fail closed when any required fact is unavailable."""

    mx = _require_mlx()
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        raise MixtralExpertPagedHardwareError(
            "Mixtral expert paging requires a live Apple-silicon Darwin host"
        )
    metal = getattr(mx, "metal", None)
    available = getattr(metal, "is_available", None)
    if not callable(available) or available() is not True:
        raise MixtralExpertPagedHardwareError("MLX cannot prove that Metal is available")
    default_device = str(mx.default_device())
    if "gpu" not in default_device.lower():
        raise MixtralExpertPagedHardwareError("the live MLX default device is not the Metal GPU")
    try:
        info = dict(mx.device_info())
    except Exception as exc:
        raise MixtralExpertPagedHardwareError("MLX cannot report live Metal device facts") from exc

    def text_field(name: str) -> str:
        value = info.get(name)
        if not isinstance(value, str) or not value or value.strip() != value:
            raise MixtralExpertPagedHardwareError(
                f"MLX device evidence is missing canonical {name}"
            )
        return value

    def integer_field(name: str) -> int:
        value = info.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MixtralExpertPagedHardwareError(f"MLX device evidence is missing positive {name}")
        return int(value)

    try:
        mlx_version = package_version("mlx")
    except PackageNotFoundError as exc:  # pragma: no cover - malformed installation
        raise MixtralExpertPagedHardwareError(
            "the installed MLX version is unidentifiable"
        ) from exc
    try:
        return MixtralExpertPagedHardwareEvidence(
            platform_system=platform.system(),
            machine=platform.machine(),
            mlx_version=mlx_version,
            default_device=default_device,
            metal_available=True,
            device_name=text_field("device_name"),
            architecture=text_field("architecture"),
            memory_size_bytes=integer_field("memory_size"),
            max_recommended_working_set_size_bytes=integer_field(
                "max_recommended_working_set_size"
            ),
            max_buffer_length_bytes=integer_field("max_buffer_length"),
            resource_limit=integer_field("resource_limit"),
        )
    except ValueError as exc:
        raise MixtralExpertPagedHardwareError(str(exc)) from exc


def validate_mixtral_expert_paged_hardware(
    evidence: MixtralExpertPagedHardwareEvidence,
) -> None:
    if not isinstance(evidence, MixtralExpertPagedHardwareEvidence):
        raise MixtralExpertPagedHardwareError(
            "Mixtral runtime construction requires explicit live hardware evidence"
        )
    current = probe_mixtral_expert_paged_hardware()
    if evidence != current or evidence.fingerprint != current.fingerprint:
        raise MixtralExpertPagedHardwareError(
            "Mixtral hardware evidence is stale, forged, or belongs to another host"
        )


@dataclass(frozen=True, slots=True)
class MixtralTieredPlacement:
    """Weight-only tiered accounting; deliberately not a resident ``PlacementPlan``.

    ``total_reserved_bytes`` covers evaluated skeleton tensors, the maximum evaluated expert
    cache payload, workspace, and headroom.  Expert-store files are authenticated backing bytes,
    not claimed resident bytes.  KV state is explicitly absent from this slice.
    """

    artifact_sha256: str
    source_model_fingerprint: str
    hardware_fingerprint: str
    device_id: str
    memory_domain: str
    skeleton_resident_bytes: int
    expert_cache_capacity_bytes: int
    expert_load_staging_bytes: int
    expert_page_bytes: int
    expert_store_bytes: int
    workspace_bytes: int
    headroom_bytes: int
    total_reserved_bytes: int
    memory_budget_bytes: int
    schema_version: str = MIXTRAL_EXPERT_PAGED_PLACEMENT_SCHEMA
    backend_id: str = MIXTRAL_EXPERT_PAGED_BACKEND_ID
    runtime_abi: str = MIXTRAL_EXPERT_STORE_RUNTIME_ABI
    codec: str = MIXTRAL_EXPERT_STORE_CODEC
    fully_resident: bool = False
    includes_kv_state: bool = False
    performance_claim_valid: bool = False
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_sha256",
            "source_model_fingerprint",
            "hardware_fingerprint",
        ):
            value = str(getattr(self, field_name))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if self.schema_version != MIXTRAL_EXPERT_PAGED_PLACEMENT_SCHEMA:
            raise ValueError("tiered placement schema is unsupported")
        if self.backend_id != MIXTRAL_EXPERT_PAGED_BACKEND_ID:
            raise ValueError("tiered placement backend is unsupported")
        if self.runtime_abi != MIXTRAL_EXPERT_STORE_RUNTIME_ABI:
            raise ValueError("tiered placement runtime ABI is unsupported")
        if self.codec != MIXTRAL_EXPERT_STORE_CODEC:
            raise ValueError("tiered placement codec is unsupported")
        if self.device_id != "metal:0" or self.memory_domain != "unified":
            raise ValueError("this runtime slice supports only the explicit Metal unified route")
        for field_name in (
            "skeleton_resident_bytes",
            "expert_cache_capacity_bytes",
            "expert_load_staging_bytes",
            "expert_page_bytes",
            "expert_store_bytes",
            "total_reserved_bytes",
            "memory_budget_bytes",
        ):
            _positive_int(getattr(self, field_name), field_name)
        for field_name in ("workspace_bytes", "headroom_bytes"):
            _nonnegative_int(getattr(self, field_name), field_name)
        if self.expert_cache_capacity_bytes < self.expert_page_bytes:
            raise ValueError("tiered placement cache cannot hold one expert page")
        if self.expert_load_staging_bytes != self.expert_page_bytes:
            raise ValueError("tiered placement must reserve one transactional load page")
        if self.expert_cache_capacity_bytes > self.expert_store_bytes:
            raise ValueError("tiered placement cache exceeds its complete expert store")
        expected_total = (
            self.skeleton_resident_bytes
            + self.expert_cache_capacity_bytes
            + self.expert_load_staging_bytes
            + self.workspace_bytes
            + self.headroom_bytes
        )
        if self.total_reserved_bytes != expected_total:
            raise ValueError("tiered placement reserved-byte accounting is inconsistent")
        if self.total_reserved_bytes > self.memory_budget_bytes:
            raise ValueError("tiered placement exceeds its memory budget")
        if self.fully_resident or self.includes_kv_state or self.performance_claim_valid:
            raise ValueError("tiered slice cannot claim resident/full-runtime performance status")
        fingerprint = _sha256_bytes(_canonical_json_bytes(self.payload()))
        object.__setattr__(self, "fingerprint", fingerprint)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_id": self.backend_id,
            "runtime_abi": self.runtime_abi,
            "codec": self.codec,
            "artifact_sha256": self.artifact_sha256,
            "source_model_fingerprint": self.source_model_fingerprint,
            "hardware_fingerprint": self.hardware_fingerprint,
            "device_id": self.device_id,
            "memory_domain": self.memory_domain,
            "skeleton_resident_bytes": self.skeleton_resident_bytes,
            "expert_cache_capacity_bytes": self.expert_cache_capacity_bytes,
            "expert_load_staging_bytes": self.expert_load_staging_bytes,
            "expert_page_bytes": self.expert_page_bytes,
            "expert_store_bytes": self.expert_store_bytes,
            "workspace_bytes": self.workspace_bytes,
            "headroom_bytes": self.headroom_bytes,
            "total_reserved_bytes": self.total_reserved_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "fully_resident": self.fully_resident,
            "includes_kv_state": self.includes_kv_state,
            "performance_claim_valid": self.performance_claim_valid,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.payload(), "fingerprint": self.fingerprint}


def plan_mixtral_tiered_placement(
    artifact: VerifiedMixtralExpertStore,
    *,
    hardware_evidence: MixtralExpertPagedHardwareEvidence,
    expert_cache_capacity_bytes: int,
    memory_budget_bytes: int,
    workspace_bytes: int = 0,
    headroom_bytes: int = 0,
) -> MixtralTieredPlacement:
    validate_mixtral_expert_paged_hardware(hardware_evidence)
    requested_cache = _positive_int(expert_cache_capacity_bytes, "expert_cache_capacity_bytes")
    budget = _positive_int(memory_budget_bytes, "memory_budget_bytes")
    workspace = _nonnegative_int(workspace_bytes, "workspace_bytes")
    headroom = _nonnegative_int(headroom_bytes, "headroom_bytes")
    if requested_cache > artifact.expert_store_tensor_bytes:
        raise MixtralTieredPlacementError(
            "expert cache budget exceeds the complete authenticated expert store"
        )
    if requested_cache % artifact.expert_page_tensor_bytes:
        raise MixtralTieredPlacementError(
            "expert cache budget must contain an exact whole number of expert pages"
        )
    cache_capacity = requested_cache
    if cache_capacity < artifact.expert_page_tensor_bytes:
        raise MixtralTieredPlacementError(
            "expert cache budget cannot hold one authenticated expert page"
        )
    staging = artifact.expert_page_tensor_bytes
    total = artifact.skeleton_tensor_bytes + cache_capacity + staging + workspace + headroom
    if total > budget:
        raise MixtralTieredPlacementError(
            "tiered Mixtral weight placement exceeds budget: "
            f"skeleton={artifact.skeleton_tensor_bytes}, cache={cache_capacity}, "
            f"staging={staging}, "
            f"workspace={workspace}, headroom={headroom}, total={total}, budget={budget}"
        )
    if budget > hardware_evidence.max_recommended_working_set_size_bytes:
        raise MixtralTieredPlacementError(
            "tiered Mixtral memory budget exceeds the live Metal recommended working set"
        )
    return MixtralTieredPlacement(
        artifact_sha256=artifact.artifact_sha256,
        source_model_fingerprint=str(artifact.source["model_fingerprint"]),
        hardware_fingerprint=hardware_evidence.fingerprint,
        device_id="metal:0",
        memory_domain="unified",
        skeleton_resident_bytes=artifact.skeleton_tensor_bytes,
        expert_cache_capacity_bytes=cache_capacity,
        expert_load_staging_bytes=staging,
        expert_page_bytes=artifact.expert_page_tensor_bytes,
        expert_store_bytes=artifact.expert_store_tensor_bytes,
        workspace_bytes=workspace,
        headroom_bytes=headroom,
        total_reserved_bytes=total,
        memory_budget_bytes=budget,
    )


def validate_mixtral_tiered_placement(
    placement: MixtralTieredPlacement,
    artifact: VerifiedMixtralExpertStore,
    hardware_evidence: MixtralExpertPagedHardwareEvidence,
) -> None:
    if not isinstance(placement, MixtralTieredPlacement):
        raise MixtralTieredPlacementError("Mixtral route requires its separate tiered placement")
    expected = plan_mixtral_tiered_placement(
        artifact,
        hardware_evidence=hardware_evidence,
        expert_cache_capacity_bytes=placement.expert_cache_capacity_bytes,
        memory_budget_bytes=placement.memory_budget_bytes,
        workspace_bytes=placement.workspace_bytes,
        headroom_bytes=placement.headroom_bytes,
    )
    if placement != expected or placement.fingerprint != expected.fingerprint:
        raise MixtralTieredPlacementError(
            "tiered placement is stale, forged, or belongs to another artifact"
        )


def _manifest_tensor_records(parameters: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(parameter["name"]): dict(parameter) for parameter in parameters}


def _mlx_dtype_code(value: Any) -> str:
    mapping = {
        "mlx.core.bfloat16": "BF16",
        "mlx.core.float16": "F16",
        "mlx.core.float32": "F32",
        "mlx.core.uint32": "U32",
    }
    try:
        return mapping[str(value.dtype)]
    except KeyError as exc:
        raise MixtralExpertPagedRuntimeError(f"unsupported MLX tensor dtype {value.dtype}") from exc


class ResidentMixtralSkeleton:
    """Evaluated, immutable MLX arrays for every authenticated non-expert parameter."""

    def __init__(self, artifact: VerifiedMixtralExpertStore):
        mx = _require_mlx()
        tensors: dict[str, Any] = {}
        for shard in artifact.skeleton_shards:
            with artifact.open_verified_member(str(shard["filename"])) as handle:
                loaded = mx.load(handle, format="safetensors")
                if not isinstance(loaded, dict):
                    raise MixtralExpertPagedRuntimeError(
                        "skeleton shard did not load as a tensor map"
                    )
                expected = _manifest_tensor_records(shard["parameters"])
                if set(loaded) != set(expected):
                    raise MixtralExpertPagedRuntimeError("skeleton shard inventory changed at load")
                for name, value in loaded.items():
                    record = expected[name]
                    if (
                        tuple(int(item) for item in value.shape) != tuple(record["shape"])
                        or _mlx_dtype_code(value) != record["dtype"]
                        or int(value.nbytes) != int(record["logical_bytes"])
                        or name in tensors
                    ):
                        raise MixtralExpertPagedRuntimeError(
                            "skeleton tensor identity/shape/dtype changed at load"
                        )
                # MLX is lazy.  Evaluation must happen while the authenticated descriptor is
                # open; the context then rehashes that same descriptor before any arrays escape.
                mx.eval(*loaded.values())
            tensors.update(loaded)
        if set(tensors) != set(artifact.skeleton_tensor_names):
            raise MixtralExpertPagedRuntimeError("resident skeleton inventory is incomplete")
        resident_bytes = sum(int(value.nbytes) for value in tensors.values())
        if resident_bytes != artifact.skeleton_tensor_bytes:
            raise MixtralExpertPagedRuntimeError("resident skeleton byte accounting is invalid")
        self._artifact_sha256 = artifact.artifact_sha256
        self._tensors = MappingProxyType(tensors)
        self._resident_bytes = resident_bytes
        self._active_operations = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset(self._tensors)

    def _acquire_tensor(self, name: str) -> Any:
        with self._lock:
            if self._closed:
                raise MixtralExpertPagedRuntimeError("resident skeleton is closed")
            try:
                value = self._tensors[name]
            except KeyError as exc:
                raise MixtralExpertPagedRuntimeError(
                    f"resident skeleton has no tensor {name!r}"
                ) from exc
            self._active_operations += 1
            return value

    def _acquire_router(self, layer: int) -> Any:
        return self._acquire_tensor(f"model.layers.{int(layer)}.block_sparse_moe.gate.weight")

    def _release_operation(self) -> None:
        with self._lock:
            if self._active_operations <= 0:
                raise AssertionError("resident skeleton operation count underflow")
            self._active_operations -= 1

    def router_shape(self, layer: int) -> tuple[int, ...]:
        router = self._acquire_router(layer)
        try:
            return tuple(int(value) for value in router.shape)
        finally:
            self._release_operation()

    def close(self) -> None:
        with self._lock:
            if self._active_operations:
                raise MixtralExpertPagedRuntimeError(
                    "cannot close resident skeleton during an active operation"
                )
            self._closed = True
            self._tensors = MappingProxyType({})
            self._resident_bytes = 0


@dataclass(frozen=True, slots=True)
class QuantizedProjection:
    weight: Any
    scales: Any
    biases: Any
    input_features: int
    output_features: int

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in (self.weight, self.scales, self.biases))


def _quantized_linear(value: Any, projection: QuantizedProjection) -> Any:
    mx = _require_mlx()
    if int(value.shape[-1]) != projection.input_features:
        raise MixtralExpertPagedRuntimeError("quantized projection input width is invalid")
    output = mx.quantized_matmul(
        value,
        projection.weight,
        scales=projection.scales,
        biases=projection.biases,
        transpose=True,
        group_size=MIXTRAL_EXPERT_STORE_GROUP_SIZE,
        bits=MIXTRAL_EXPERT_STORE_BITS,
        mode=MIXTRAL_EXPERT_STORE_MODE,
    )
    if int(output.shape[-1]) != projection.output_features:
        raise MixtralExpertPagedRuntimeError("quantized projection output width is invalid")
    return output


@dataclass(frozen=True, slots=True)
class QuantizedMixtralExpert:
    layer: int
    expert: int
    w1: QuantizedProjection
    w2: QuantizedProjection
    w3: QuantizedProjection

    @property
    def nbytes(self) -> int:
        return self.w1.nbytes + self.w2.nbytes + self.w3.nbytes


@dataclass(slots=True)
class _CacheEntry:
    page: QuantizedMixtralExpert
    nbytes: int
    leases: int = 0


@dataclass(slots=True)
class _MutableCacheStats:
    page_requests: int = 0
    page_hits: int = 0
    page_misses: int = 0
    page_load_failures: int = 0
    page_evictions: int = 0
    loaded_bytes: int = 0
    peak_resident_bytes: int = 0
    logical_staging_bytes: int = 0
    peak_logical_staging_bytes: int = 0
    peak_logical_live_bytes: int = 0
    requests_started: int = 0
    requests_completed: int = 0

    def as_dict(self, *, resident_bytes: int, active_requests: int) -> dict[str, Any]:
        return {
            "page_requests": self.page_requests,
            "page_hits": self.page_hits,
            "page_misses": self.page_misses,
            "page_load_failures": self.page_load_failures,
            "page_evictions": self.page_evictions,
            "loaded_bytes": self.loaded_bytes,
            "peak_resident_bytes": self.peak_resident_bytes,
            "resident_bytes": resident_bytes,
            "logical_staging_bytes": self.logical_staging_bytes,
            "peak_logical_staging_bytes": self.peak_logical_staging_bytes,
            "logical_live_bytes": resident_bytes + self.logical_staging_bytes,
            "peak_logical_live_bytes": self.peak_logical_live_bytes,
            "active_requests": active_requests,
            "requests_started": self.requests_started,
            "requests_completed": self.requests_completed,
            "hit_rate": self.page_hits / max(1, self.page_requests),
        }


class MixtralExpertLease(AbstractContextManager[Any]):
    def __init__(
        self,
        cache: BoundedMixtralExpertCache,
        entries: Mapping[tuple[int, int], QuantizedMixtralExpert],
    ) -> None:
        self._cache = cache
        self._entries = MappingProxyType(dict(entries))
        self._active = True

    def __enter__(self) -> MixtralExpertLease:
        if not self._active:
            raise MixtralExpertCacheError("expert lease is no longer active")
        return self

    def page_nbytes(self, key: tuple[int, int]) -> int:
        if not self._active:
            raise MixtralExpertCacheError("expert lease is no longer active")
        return self._entries[key].nbytes

    def forward(self, key: tuple[int, int], value: Any) -> Any:
        """Execute one complete expert without exposing cache-owned page arrays."""

        if not self._active:
            raise MixtralExpertCacheError("expert lease is no longer active")
        mx = _require_mlx()
        page = self._entries[key]
        gate = _quantized_linear(value, page.w1)
        up = _quantized_linear(value, page.w3)
        activated = (gate * mx.sigmoid(gate)) * up
        output = _quantized_linear(activated, page.w2)
        finite = mx.all(mx.isfinite(output))
        mx.eval(output, finite)
        if not bool(finite.item()):
            raise MixtralExpertPagedRuntimeError(
                "quantized Mixtral expert produced non-finite output"
            )
        return output

    @property
    def keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._entries)

    def release(self) -> None:
        if not self._active:
            raise MixtralExpertCacheError("expert lease was released more than once")
        self._cache._release(tuple(self._entries))
        self._active = False
        # Do not let a released lease become an unaccounted second owner of page arrays.
        self._entries = MappingProxyType({})

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


class MixtralCacheRequest(AbstractContextManager[Any]):
    def __init__(self, cache: BoundedMixtralExpertCache):
        self._cache = cache
        self._active = False
        self._leases = 0

    def __enter__(self) -> MixtralCacheRequest:
        if self._active:
            raise MixtralExpertCacheError("cache request cannot be entered twice")
        self._cache._begin_request()
        self._active = True
        return self

    def lease(self, keys: Sequence[tuple[int, int]]) -> MixtralExpertLease:
        if not self._active:
            raise MixtralExpertCacheError("cache request is not active")
        lease = self._cache.lease(keys)
        self._leases += 1
        return _RequestLease(self, lease)

    def _lease_released(self) -> None:
        self._leases -= 1
        if self._leases < 0:
            raise AssertionError("cache request lease count underflow")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._active:
            raise MixtralExpertCacheError("cache request is no longer active")
        if self._leases:
            raise MixtralExpertCacheError("cache request exited with active expert leases")
        self._active = False
        self._cache._end_request()


class _RequestLease(MixtralExpertLease):
    def __init__(self, request: MixtralCacheRequest, lease: MixtralExpertLease):
        self._request = request
        self._delegate = lease
        self._active = True

    def page_nbytes(self, key: tuple[int, int]) -> int:
        return self._delegate.page_nbytes(key)

    def forward(self, key: tuple[int, int], value: Any) -> Any:
        return self._delegate.forward(key, value)

    @property
    def keys(self) -> tuple[tuple[int, int], ...]:
        return self._delegate.keys

    def release(self) -> None:
        if not self._active:
            raise MixtralExpertCacheError("expert request lease was released more than once")
        self._delegate.release()
        self._active = False
        self._request._lease_released()


class BoundedMixtralExpertCache:
    """Byte-bounded, lease-aware LRU over immutable expert page files."""

    def __init__(self, artifact: VerifiedMixtralExpertStore, *, capacity_bytes: int):
        capacity = _positive_int(capacity_bytes, "capacity_bytes")
        if capacity < artifact.expert_page_tensor_bytes:
            raise MixtralExpertCacheError("expert cache cannot hold one page")
        if capacity > artifact.expert_store_tensor_bytes:
            raise MixtralExpertCacheError("expert cache capacity exceeds complete store size")
        if capacity % artifact.expert_page_tensor_bytes:
            raise MixtralExpertCacheError(
                "expert cache capacity must contain an exact whole number of pages"
            )
        self.artifact = artifact
        self.capacity_bytes = capacity
        self.staging_capacity_bytes = artifact.expert_page_tensor_bytes
        self._condition = threading.Condition(threading.RLock())
        self._entries: OrderedDict[tuple[int, int], _CacheEntry] = OrderedDict()
        self._resident_bytes = 0
        self._active_requests = 0
        self._closed = False
        self._stats = _MutableCacheStats()

    @property
    def resident_bytes(self) -> int:
        with self._condition:
            return self._resident_bytes

    @property
    def cached_keys(self) -> tuple[tuple[int, int], ...]:
        with self._condition:
            return tuple(self._entries)

    def stats(self) -> dict[str, Any]:
        with self._condition:
            return self._stats.as_dict(
                resident_bytes=self._resident_bytes,
                active_requests=self._active_requests,
            )

    def request(self) -> MixtralCacheRequest:
        return MixtralCacheRequest(self)

    def _begin_request(self) -> None:
        with self._condition:
            if self._closed:
                raise MixtralExpertCacheError("expert cache is closed")
            self._active_requests += 1
            self._stats.requests_started += 1

    def _end_request(self) -> None:
        with self._condition:
            if self._active_requests <= 0:
                raise AssertionError("expert cache request count underflow")
            self._active_requests -= 1
            self._stats.requests_completed += 1
            self._condition.notify_all()

    def _normalize_keys(self, keys: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        normalized: set[tuple[int, int]] = set()
        layers = self.artifact.topology["num_hidden_layers"]
        experts = self.artifact.topology["num_local_experts"]
        for key in keys:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in key)
            ):
                raise MixtralExpertCacheError("expert cache key must be an integer pair")
            layer, expert = key
            if not 0 <= layer < layers or not 0 <= expert < experts:
                raise MixtralExpertCacheError("expert cache key is outside artifact topology")
            normalized.add((layer, expert))
        if not normalized:
            raise MixtralExpertCacheError("expert cache lease cannot be empty")
        return tuple(sorted(normalized))

    def _load_page(self, key: tuple[int, int]) -> QuantizedMixtralExpert:
        mx = _require_mlx()
        layer, expert = key
        record = self.artifact.expert_page(layer, expert)
        with self.artifact.open_verified_member(str(record["filename"])) as handle:
            loaded = mx.load(handle, format="safetensors")
            if not isinstance(loaded, dict):
                raise MixtralExpertCacheError("expert page did not load as a tensor map")
            expected = _manifest_tensor_records(record["tensors"])
            if set(loaded) != set(expected):
                raise MixtralExpertCacheError("expert page inventory changed at load")
            for name, value in loaded.items():
                tensor = expected[name]
                if (
                    tuple(int(item) for item in value.shape) != tuple(tensor["shape"])
                    or _mlx_dtype_code(value) != tensor["dtype"]
                    or int(value.nbytes) != int(tensor["logical_bytes"])
                ):
                    raise MixtralExpertCacheError(
                        "expert page tensor identity/shape/dtype changed at load"
                    )
            mx.eval(*loaded.values())
        auxiliaries = [
            value
            for name, value in loaded.items()
            if name.endswith(".scales") or name.endswith(".biases")
        ]
        finite = mx.array(True)
        for value in auxiliaries:
            finite = mx.logical_and(finite, mx.all(mx.isfinite(value)))
        mx.eval(finite)
        if not bool(finite.item()):
            raise MixtralExpertCacheError("expert page contains non-finite quantization metadata")
        hidden = self.artifact.topology["hidden_size"]
        intermediate = self.artifact.topology["intermediate_size"]

        def projection(name: str, input_features: int, output_features: int) -> QuantizedProjection:
            return QuantizedProjection(
                weight=loaded[f"{name}.weight"],
                scales=loaded[f"{name}.scales"],
                biases=loaded[f"{name}.biases"],
                input_features=input_features,
                output_features=output_features,
            )

        page = QuantizedMixtralExpert(
            layer=layer,
            expert=expert,
            w1=projection("w1", hidden, intermediate),
            w2=projection("w2", intermediate, hidden),
            w3=projection("w3", hidden, intermediate),
        )
        if page.nbytes != int(record["tensor_bytes"]):
            raise MixtralExpertCacheError("expert page resident-byte accounting is invalid")
        return page

    def lease(self, keys: Sequence[tuple[int, int]]) -> MixtralExpertLease:
        normalized = self._normalize_keys(keys)
        with self._condition:
            if self._closed:
                raise MixtralExpertCacheError("expert cache is closed")
            hits = tuple(key for key in normalized if key in self._entries)
            misses = tuple(key for key in normalized if key not in self._entries)
            staged_bytes = len(misses) * self.artifact.expert_page_tensor_bytes
            if staged_bytes > self.staging_capacity_bytes:
                raise MixtralExpertCacheError(
                    "one cache lease may stage at most one missing expert page"
                )
            protected = set(normalized)
            victims: list[tuple[int, int]] = []
            projected = self._resident_bytes + staged_bytes
            for key, entry in self._entries.items():
                if projected <= self.capacity_bytes:
                    break
                if key not in protected and entry.leases == 0:
                    victims.append(key)
                    projected -= entry.nbytes
            if projected > self.capacity_bytes:
                raise MixtralExpertCacheError(
                    "expert cache cannot evict a leased or request-protected page"
                )
            self._stats.logical_staging_bytes = staged_bytes
            self._stats.peak_logical_staging_bytes = max(
                self._stats.peak_logical_staging_bytes,
                staged_bytes,
            )
            self._stats.peak_logical_live_bytes = max(
                self._stats.peak_logical_live_bytes,
                self._resident_bytes + staged_bytes,
            )
            staged: dict[tuple[int, int], QuantizedMixtralExpert] = {}
            try:
                for key in misses:
                    staged[key] = self._load_page(key)
                actual_staged_bytes = sum(page.nbytes for page in staged.values())
                if actual_staged_bytes != staged_bytes:
                    raise MixtralExpertCacheError("staged expert bytes differ from the manifest")
            except Exception:
                self._stats.page_load_failures += 1
                self._stats.logical_staging_bytes = 0
                raise
            snapshot = OrderedDict(
                (
                    key,
                    _CacheEntry(page=entry.page, nbytes=entry.nbytes, leases=entry.leases),
                )
                for key, entry in self._entries.items()
            )
            resident_snapshot = self._resident_bytes
            try:
                for key in victims:
                    entry = self._entries.pop(key)
                    self._resident_bytes -= entry.nbytes
                for key, page in staged.items():
                    self._entries[key] = _CacheEntry(page=page, nbytes=page.nbytes)
                    self._resident_bytes += page.nbytes
                selected: dict[tuple[int, int], QuantizedMixtralExpert] = {}
                for key in normalized:
                    entry = self._entries[key]
                    entry.leases += 1
                    self._entries.move_to_end(key)
                    selected[key] = entry.page
                if self._resident_bytes > self.capacity_bytes:
                    raise AssertionError("expert cache exceeded its byte capacity")
            except BaseException:
                self._entries = snapshot
                self._resident_bytes = resident_snapshot
                self._stats.logical_staging_bytes = 0
                raise
            self._stats.logical_staging_bytes = 0
            self._stats.page_requests += len(normalized)
            self._stats.page_hits += len(hits)
            self._stats.page_misses += len(misses)
            self._stats.page_evictions += len(victims)
            self._stats.loaded_bytes += staged_bytes
            self._stats.peak_resident_bytes = max(
                self._stats.peak_resident_bytes, self._resident_bytes
            )
            return MixtralExpertLease(self, selected)

    def _release(self, keys: tuple[tuple[int, int], ...]) -> None:
        with self._condition:
            for key in keys:
                entry = self._entries.get(key)
                if entry is None or entry.leases <= 0:
                    raise MixtralExpertCacheError("expert lease release state is invalid")
            for key in keys:
                self._entries[key].leases -= 1
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._active_requests or any(entry.leases for entry in self._entries.values()):
                raise MixtralExpertCacheError("cannot close expert cache during an active request")
            self._closed = True
            self._entries.clear()
            self._resident_bytes = 0
            self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class MixtralSparseBlockResult:
    output: Any
    router_logits: Any
    routing_weights: Any
    selected_experts: Any
    dispatch_rows: tuple[int, ...]


class MlxMixtralSparseBlock:
    """Classic routed-only Mixtral block using q4 ``mx.quantized_matmul`` experts."""

    def __init__(
        self,
        *,
        layer: int,
        skeleton: ResidentMixtralSkeleton,
        cache: BoundedMixtralExpertCache,
    ) -> None:
        mx = _require_mlx()
        topology = cache.artifact.topology
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not (0 <= layer < topology["num_hidden_layers"])
        ):
            raise MixtralExpertPagedRuntimeError("sparse block layer is outside topology")
        if skeleton._artifact_sha256 != cache.artifact.artifact_sha256:
            raise MixtralExpertPagedRuntimeError(
                "sparse block skeleton and expert cache belong to different artifacts"
            )
        router_weight = skeleton._acquire_router(layer)
        expected_shape = (topology["num_local_experts"], topology["hidden_size"])
        try:
            if tuple(int(value) for value in router_weight.shape) != expected_shape:
                raise MixtralExpertPagedRuntimeError("router weight shape differs from topology")
            if _mlx_dtype_code(router_weight) not in {"BF16", "F16", "F32"}:
                raise MixtralExpertPagedRuntimeError("router weight dtype is unsupported")
            finite = mx.all(mx.isfinite(router_weight))
            mx.eval(finite)
            if not bool(finite.item()):
                raise MixtralExpertPagedRuntimeError("router weight contains non-finite values")
        finally:
            skeleton._release_operation()
        self.layer = layer
        self.skeleton = skeleton
        self.cache = cache
        self.hidden_size = topology["hidden_size"]
        self.intermediate_size = topology["intermediate_size"]
        self.num_experts = topology["num_local_experts"]
        self.top_k = topology["num_experts_per_token"]

    @staticmethod
    def _qlinear(value: Any, projection: QuantizedProjection) -> Any:
        return _quantized_linear(value, projection)

    def _forward_with_router(
        self,
        hidden_states: Any,
        router_weight: Any,
    ) -> MixtralSparseBlockResult:
        mx = _require_mlx()
        if len(hidden_states.shape) != 3 or int(hidden_states.shape[-1]) != self.hidden_size:
            raise MixtralExpertPagedRuntimeError(
                "Mixtral sparse block requires [batch, tokens, hidden] input"
            )
        if _mlx_dtype_code(hidden_states) not in {"BF16", "F16", "F32"}:
            raise MixtralExpertPagedRuntimeError("sparse block input dtype is unsupported")
        finite_input = mx.all(mx.isfinite(hidden_states))
        mx.eval(finite_input)
        if not bool(finite_input.item()):
            raise MixtralExpertPagedRuntimeError("sparse block input contains non-finite values")

        router_logits = hidden_states @ router_weight.T
        probabilities = mx.softmax(router_logits.astype(mx.float32), axis=-1, precise=True)
        # Stable descending sort gives deterministic expert-index tie breaking.  Softmax-before-
        # top-k plus selected-mass renormalization is the canonical adapter contract.
        selected = mx.stop_gradient(mx.argsort(-probabilities, axis=-1)[..., : self.top_k])
        selected_mass = mx.take_along_axis(probabilities, selected, axis=-1)
        denominator = mx.sum(selected_mass, axis=-1, keepdims=True)
        routing_weights = (selected_mass / denominator).astype(router_logits.dtype)
        valid_routes = mx.logical_and(mx.all(mx.isfinite(routing_weights)), mx.all(denominator > 0))
        mx.eval(selected, routing_weights, valid_routes)
        if not bool(valid_routes.item()):
            raise MixtralExpertPagedRuntimeError("router selected non-finite or zero mass")

        batch, tokens, _hidden = (int(value) for value in hidden_states.shape)
        flattened = hidden_states.reshape(batch * tokens, self.hidden_size)
        selected_flat = selected.reshape(batch * tokens, self.top_k)
        weights_flat = routing_weights.reshape(batch * tokens, self.top_k)
        selected_host = np.asarray(selected_flat, dtype=np.int64)
        output = mx.zeros((batch * tokens, self.hidden_size), dtype=hidden_states.dtype)
        dispatch_rows: list[int] = []
        with self.cache.request() as request:
            for expert in range(self.num_experts):
                top_k_indices, token_indices = np.nonzero(selected_host.T == expert)
                dispatch_rows.append(int(token_indices.size))
                if not token_indices.size:
                    continue
                token_ids = mx.array(token_indices.astype(np.int32, copy=False))
                route_slots = mx.array(top_k_indices.astype(np.int32, copy=False))
                expert_input = mx.take(flattened, token_ids, axis=0)
                with request.lease(((self.layer, expert),)) as lease:
                    expert_output = lease.forward((self.layer, expert), expert_input)
                    coefficients = weights_flat[token_ids, route_slots].astype(expert_output.dtype)
                    weighted = expert_output * coefficients[:, None]
                    # Expert-major iteration is the canonical source-expert-index accumulation
                    # order; ArrayAt.add preserves duplicate-token scatter-add semantics.
                    output = output.at[token_ids].add(weighted.astype(output.dtype))
                    # Materialize before releasing the page so lazy graphs cannot extend its
                    # lifetime beyond the cache accounting/lease boundary.
                    mx.eval(output)
                del expert_output, coefficients, weighted
        output = output.reshape(batch, tokens, self.hidden_size)
        finite_output = mx.all(mx.isfinite(output))
        mx.eval(output, router_logits, routing_weights, selected, finite_output)
        if not bool(finite_output.item()):
            raise MixtralExpertPagedRuntimeError("Mixtral sparse block produced non-finite output")
        return MixtralSparseBlockResult(
            output=output,
            router_logits=router_logits,
            routing_weights=routing_weights,
            selected_experts=selected,
            dispatch_rows=tuple(dispatch_rows),
        )

    def forward(self, hidden_states: Any) -> MixtralSparseBlockResult:
        router_weight = self.skeleton._acquire_router(self.layer)
        try:
            return self._forward_with_router(hidden_states, router_weight)
        finally:
            self.skeleton._release_operation()

    def __call__(self, hidden_states: Any) -> Any:
        return self.forward(hidden_states).output


class MixtralExpertPagedRuntimeSlice:
    """Opened skeleton/cache/block factory with an explicit non-full-model claim boundary."""

    backend_id = MIXTRAL_EXPERT_PAGED_BACKEND_ID
    runtime_abi = MIXTRAL_EXPERT_STORE_RUNTIME_ABI
    numerical_contract = MIXTRAL_EXPERT_STORE_NUMERICAL_CONTRACT
    native_runtime_candidate = True
    production_runtime_eligible = False
    full_model_runtime = False
    performance_claim_valid = False

    def __init__(
        self,
        artifact: VerifiedMixtralExpertStore,
        *,
        placement: MixtralTieredPlacement,
        hardware_evidence: MixtralExpertPagedHardwareEvidence,
    ) -> None:
        validate_mixtral_tiered_placement(placement, artifact, hardware_evidence)
        skeleton = ResidentMixtralSkeleton(artifact)
        if skeleton.resident_bytes != placement.skeleton_resident_bytes:
            skeleton.close()
            raise MixtralTieredPlacementError("loaded skeleton differs from placement accounting")
        try:
            cache = BoundedMixtralExpertCache(
                artifact, capacity_bytes=placement.expert_cache_capacity_bytes
            )
        except BaseException:
            skeleton.close()
            raise
        self.artifact = artifact
        self.hardware_evidence = hardware_evidence
        self.placement = placement
        self.skeleton = skeleton
        self.cache = cache
        self._closed = False
        self._lock = threading.RLock()

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        placement: MixtralTieredPlacement,
        hardware_evidence: MixtralExpertPagedHardwareEvidence,
    ) -> MixtralExpertPagedRuntimeSlice:
        return cls(
            VerifiedMixtralExpertStore(path),
            placement=placement,
            hardware_evidence=hardware_evidence,
        )

    def block(self, layer: int) -> MlxMixtralSparseBlock:
        with self._lock:
            if self._closed:
                raise MixtralExpertPagedRuntimeError("Mixtral expert-paged slice is closed")
            return MlxMixtralSparseBlock(
                layer=layer,
                skeleton=self.skeleton,
                cache=self.cache,
            )

    def accounting(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise MixtralExpertPagedRuntimeError("Mixtral expert-paged slice is closed")
            cache_stats = self.cache.stats()
            return {
                "schema": "mrun-mlx-mixtral-expert-paged-accounting-v1",
                "artifact_sha256": self.artifact.artifact_sha256,
                "hardware_evidence": self.hardware_evidence.as_dict(),
                "hardware_fingerprint": self.hardware_evidence.fingerprint,
                "placement_fingerprint": self.placement.fingerprint,
                "skeleton_resident_bytes": self.skeleton.resident_bytes,
                "expert_cache_capacity_bytes": self.cache.capacity_bytes,
                "expert_load_staging_bytes": self.cache.staging_capacity_bytes,
                "expert_cache_resident_bytes": cache_stats["resident_bytes"],
                "logical_live_tensor_bytes": (
                    self.skeleton.resident_bytes + cache_stats["logical_live_bytes"]
                ),
                "logical_peak_tensor_bytes": (
                    self.skeleton.resident_bytes + cache_stats["peak_logical_live_bytes"]
                ),
                "logical_reserved_tensor_bytes": (
                    self.placement.skeleton_resident_bytes
                    + self.placement.expert_cache_capacity_bytes
                    + self.placement.expert_load_staging_bytes
                ),
                "expert_store_bytes": self.artifact.expert_store_tensor_bytes,
                "workspace_bytes": self.placement.workspace_bytes,
                "headroom_bytes": self.placement.headroom_bytes,
                "total_reserved_bytes": self.placement.total_reserved_bytes,
                "memory_budget_bytes": self.placement.memory_budget_bytes,
                "includes_kv_state": False,
                "full_model_runtime": False,
                "performance_claim_valid": False,
                "cache": cache_stats,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            # Cache close is deliberately first and fail-closed: an active request leaves the
            # runtime and its skeleton intact rather than producing a half-closed object.
            self.cache.close()
            self.skeleton.close()
            self._closed = True


__all__ = [
    "MIXTRAL_EXPERT_PAGED_BACKEND_ID",
    "MIXTRAL_EXPERT_PAGED_HARDWARE_SCHEMA",
    "MIXTRAL_EXPERT_PAGED_PLACEMENT_SCHEMA",
    "BoundedMixtralExpertCache",
    "MixtralCacheRequest",
    "MixtralExpertCacheError",
    "MixtralExpertLease",
    "MixtralExpertPagedRuntimeError",
    "MixtralExpertPagedHardwareError",
    "MixtralExpertPagedHardwareEvidence",
    "MixtralExpertPagedRuntimeSlice",
    "MixtralSparseBlockResult",
    "MixtralTieredPlacement",
    "MixtralTieredPlacementError",
    "MlxMixtralSparseBlock",
    "QuantizedMixtralExpert",
    "QuantizedProjection",
    "ResidentMixtralSkeleton",
    "plan_mixtral_tiered_placement",
    "probe_mixtral_expert_paged_hardware",
    "validate_mixtral_expert_paged_hardware",
    "validate_mixtral_tiered_placement",
]
