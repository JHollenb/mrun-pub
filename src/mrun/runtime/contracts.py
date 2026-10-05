"""Backend-neutral contracts for native model execution.

This module deliberately contains no Torch, MLX, CUDA, HTTP, tokenizer, or server types.  It
defines the stable boundary between compiled model custody, capability/placement selection,
mutable model state, and a native executor.  Concrete backends retain their tensors and device
objects behind :class:`StateHandle` and :class:`ProvisionalAuthority`.

The common layer owns semantic validation and canonical identity.  A backend still must reject
foreign or stale opaque authorities by exact runtime ownership; structural ``Protocol`` checks
alone never grant execution authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

COMPILED_MODEL_SCHEMA = "mrun-compiled-runtime-model-v2"
BACKEND_CAPABILITIES_SCHEMA = "mrun-native-backend-capabilities-v1"
DEVICE_DESCRIPTOR_SCHEMA = "mrun-native-device-descriptor-v1"
WORKLOAD_SPEC_SCHEMA = "mrun-native-workload-spec-v1"
PLACEMENT_PLAN_SCHEMA = "mrun-native-placement-plan-v2"
RUNTIME_ROUTE_SCHEMA = "mrun-native-runtime-route-v1"
STATE_OBSERVATION_SCHEMA = "mrun-native-state-observation-v1"
RUNTIME_TELEMETRY_SCHEMA = "mrun-native-runtime-telemetry-v1"
COMPATIBLE_BATCH_LANE_SCHEMA = "mrun-compatible-batch-lane-v1"
GREEDY_BLOCK_VERIFICATION_SCHEMA = "mrun-greedy-block-verification-v1"
GREEDY_PROPOSAL_TRANSACTION_SCHEMA = "mrun-greedy-proposal-transaction-v1"
NATIVE_SAMPLING_ABI = "mrun-splitmix64-openai-sampling-v1"


class MemoryDomain(str, Enum):
    """Physical memory domain visible to a native executor."""

    HOST = "host"
    UNIFIED = "unified"
    CUDA = "cuda"


class Residency(str, Enum):
    """How one physical allocation is expected to remain available."""

    RESIDENT = "resident"
    HOST_PINNED = "host-pinned"
    MEMORY_MAPPED = "memory-mapped"


class FallbackPolicy(str, Enum):
    """Whether a route may leave its declared native placement."""

    DENY = "deny"
    CORRECTNESS_ONLY = "correctness-only"


class OutputMode(str, Enum):
    """Executor-visible output semantics, including on-device token selection."""

    FULL_LOGITS = "full-logits"
    LAST_LOGITS = "last-logits"
    SELECTED_LOGITS = "selected-logits"
    CANDIDATE_ARGMAX_MARGIN = "candidate-argmax-margin"
    NEXT_TOKEN_ARGMAX = "next-token-argmax"
    NEXT_TOKEN_SAMPLE = "next-token-sample"
    HIDDEN_STATE = "hidden-state"


class PromotionStatus(str, Enum):
    """Evidence posture of an implementation/capability document."""

    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    PRODUCTION = "production"


JsonScalar = str | int | float | bool | None


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: str, field_name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _nonnegative_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return normalized


def _names(values: Sequence[str], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    normalized = tuple(_require_name(value, f"{field_name}[]") for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")
    return normalized


def _string_pairs(
    values: Sequence[tuple[str, str]],
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    normalized = tuple(
        (
            _require_name(key, f"{field_name}.key"),
            _require_name(value, f"{field_name}[{key!r}]"),
        )
        for key, value in values
    )
    keys = [key for key, _ in normalized]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{field_name} keys must be unique")
    return tuple(sorted(normalized))


def _counter_pairs(
    values: Sequence[tuple[str, int]],
    field_name: str,
) -> tuple[tuple[str, int], ...]:
    normalized = tuple(
        (
            _require_name(key, f"{field_name}.key"),
            _nonnegative_int(value, f"{field_name}[{key!r}]"),
        )
        for key, value in values
    )
    keys = [key for key, _ in normalized]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{field_name} keys must be unique")
    return tuple(sorted(normalized))


def _coerce_enum(value: str | Enum, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}") from exc


@dataclass(frozen=True, slots=True)
class BlobIdentity:
    """Content identity for one immutable physical blob."""

    blob_id: str
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "blob_id", _require_name(self.blob_id, "blob_id"))
        object.__setattr__(self, "sha256", _require_sha256(self.sha256, "blob sha256"))
        object.__setattr__(
            self,
            "byte_count",
            _positive_int(self.byte_count, "blob byte_count"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "blob_id": self.blob_id,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True, slots=True)
class CompiledComponent:
    """One logical component mapped to one alias-aware physical allocation."""

    component_id: str
    role: str
    allocation_id: str
    codec_id: str
    layout_id: str
    physical_bytes: int
    blobs: tuple[BlobIdentity, ...]

    def __post_init__(self) -> None:
        for field_name in ("component_id", "role", "allocation_id", "codec_id", "layout_id"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "physical_bytes",
            _positive_int(self.physical_bytes, "component physical_bytes"),
        )
        blobs = tuple(self.blobs)
        if not blobs or any(not isinstance(blob, BlobIdentity) for blob in blobs):
            raise TypeError("compiled component blobs must be non-empty BlobIdentity values")
        if len({blob.blob_id for blob in blobs}) != len(blobs):
            raise ValueError("compiled component blob IDs must be unique")
        if sum(blob.byte_count for blob in blobs) != self.physical_bytes:
            raise ValueError("compiled component blob bytes must equal physical_bytes")
        object.__setattr__(self, "blobs", tuple(sorted(blobs, key=lambda blob: blob.blob_id)))

    def physical_identity(self) -> tuple[Any, ...]:
        return (
            self.allocation_id,
            self.codec_id,
            self.layout_id,
            self.physical_bytes,
            tuple((blob.blob_id, blob.sha256, blob.byte_count) for blob in self.blobs),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "role": self.role,
            "allocation_id": self.allocation_id,
            "codec_id": self.codec_id,
            "layout_id": self.layout_id,
            "physical_bytes": self.physical_bytes,
            "blobs": [blob.as_dict() for blob in self.blobs],
        }


@dataclass(frozen=True, slots=True)
class CompiledModelIdentity:
    """Complete immutable identity consumed by backend capability matching."""

    model_name: str
    architecture: str
    source_revision_sha256: str
    semantic_model_sha256: str
    component_graph_sha256: str
    vocab_manifest_sha256: str
    compiler_abi: str
    components: tuple[CompiledComponent, ...]
    operator_ids: tuple[str, ...]
    state_abi: str
    state_dtype: str
    state_bytes_per_token: int
    max_context_tokens: int
    semantic_token_count: int
    state_fixed_bytes_per_row: int = 0
    schema_version: str = COMPILED_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != COMPILED_MODEL_SCHEMA:
            raise ValueError(f"unsupported compiled-model schema: {self.schema_version}")
        for field_name in (
            "model_name",
            "architecture",
            "compiler_abi",
            "state_abi",
            "state_dtype",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        for field_name in (
            "source_revision_sha256",
            "semantic_model_sha256",
            "component_graph_sha256",
            "vocab_manifest_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        components = tuple(self.components)
        if not components or any(not isinstance(value, CompiledComponent) for value in components):
            raise TypeError("compiled model components must be non-empty CompiledComponent values")
        if len({component.component_id for component in components}) != len(components):
            raise ValueError("compiled model component IDs must be unique")
        by_allocation: dict[str, tuple[Any, ...]] = {}
        for component in components:
            physical = component.physical_identity()
            previous = by_allocation.setdefault(component.allocation_id, physical)
            if previous != physical:
                raise ValueError(
                    "logical components sharing an allocation_id must have identical physical "
                    "identity"
                )
        object.__setattr__(
            self,
            "components",
            tuple(sorted(components, key=lambda value: value.component_id)),
        )
        object.__setattr__(self, "operator_ids", _names(self.operator_ids, "operator_ids"))
        object.__setattr__(
            self,
            "state_bytes_per_token",
            _nonnegative_int(self.state_bytes_per_token, "state_bytes_per_token"),
        )
        object.__setattr__(
            self,
            "state_fixed_bytes_per_row",
            _nonnegative_int(
                self.state_fixed_bytes_per_row,
                "state_fixed_bytes_per_row",
            ),
        )
        if self.state_bytes_per_token == 0 and self.state_fixed_bytes_per_row == 0:
            raise ValueError("compiled model must declare token-scaled or fixed per-row state")
        object.__setattr__(
            self,
            "max_context_tokens",
            _positive_int(self.max_context_tokens, "max_context_tokens"),
        )
        object.__setattr__(
            self,
            "semantic_token_count",
            _positive_int(self.semantic_token_count, "semantic_token_count"),
        )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.as_dict())

    @property
    def physical_allocation_bytes(self) -> int:
        allocations: dict[str, int] = {}
        for component in self.components:
            allocations.setdefault(component.allocation_id, component.physical_bytes)
        return sum(allocations.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "source_revision_sha256": self.source_revision_sha256,
            "semantic_model_sha256": self.semantic_model_sha256,
            "component_graph_sha256": self.component_graph_sha256,
            "vocab_manifest_sha256": self.vocab_manifest_sha256,
            "compiler_abi": self.compiler_abi,
            "components": [component.as_dict() for component in self.components],
            "operator_ids": list(self.operator_ids),
            "state_abi": self.state_abi,
            "state_dtype": self.state_dtype,
            "state_bytes_per_token": self.state_bytes_per_token,
            "state_fixed_bytes_per_row": self.state_fixed_bytes_per_row,
            "max_context_tokens": self.max_context_tokens,
            "semantic_token_count": self.semantic_token_count,
        }


@dataclass(frozen=True, slots=True)
class CodecCapability:
    """One physical codec/layout pair consumed without compatibility materialization."""

    codec_id: str
    layout_id: str
    component_roles: tuple[str, ...]
    native_direct: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "codec_id", _require_name(self.codec_id, "codec_id"))
        object.__setattr__(self, "layout_id", _require_name(self.layout_id, "layout_id"))
        object.__setattr__(
            self,
            "component_roles",
            _names(self.component_roles, "component_roles"),
        )
        if type(self.native_direct) is not bool:
            raise TypeError("native_direct must be boolean")

    def supports(self, component: CompiledComponent, *, require_native: bool) -> bool:
        return bool(
            self.codec_id == component.codec_id
            and self.layout_id == component.layout_id
            and component.role in self.component_roles
            and (self.native_direct or not require_native)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "codec_id": self.codec_id,
            "layout_id": self.layout_id,
            "component_roles": list(self.component_roles),
            "native_direct": self.native_direct,
        }


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Canonical implementation capability document, independent of a live runtime."""

    backend_id: str
    backend_abi: str
    implementation_version: str
    fabric: str
    memory_domain: MemoryDomain | str
    architectures: tuple[str, ...]
    operator_ids: tuple[str, ...]
    codecs: tuple[CodecCapability, ...]
    state_abis: tuple[str, ...]
    output_modes: tuple[OutputMode | str, ...]
    numerical_contracts: tuple[str, ...]
    max_context_tokens: int
    max_batch_size: int
    max_verify_tokens: int
    transactional_state: bool
    scratch_only_steps: bool
    independently_committable_rows: bool
    supports_ragged_batches: bool
    telemetry_counters: tuple[str, ...]
    promotion_status: PromotionStatus | str = PromotionStatus.EXPERIMENTAL
    schema_version: str = BACKEND_CAPABILITIES_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BACKEND_CAPABILITIES_SCHEMA:
            raise ValueError(f"unsupported backend-capabilities schema: {self.schema_version}")
        for field_name in (
            "backend_id",
            "backend_abi",
            "implementation_version",
            "fabric",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "memory_domain",
            _coerce_enum(self.memory_domain, MemoryDomain, "memory_domain"),
        )
        object.__setattr__(
            self,
            "promotion_status",
            _coerce_enum(self.promotion_status, PromotionStatus, "promotion_status"),
        )
        for field_name in (
            "architectures",
            "operator_ids",
            "state_abis",
            "numerical_contracts",
            "telemetry_counters",
        ):
            object.__setattr__(self, field_name, _names(getattr(self, field_name), field_name))
        codecs = tuple(self.codecs)
        if not codecs or any(not isinstance(value, CodecCapability) for value in codecs):
            raise TypeError("backend codecs must be non-empty CodecCapability values")
        codec_keys = [(value.codec_id, value.layout_id, value.component_roles) for value in codecs]
        if len(set(codec_keys)) != len(codecs):
            raise ValueError("backend codec capabilities must be unique")
        object.__setattr__(
            self,
            "codecs",
            tuple(sorted(codecs, key=lambda value: (value.codec_id, value.layout_id))),
        )
        output_modes = tuple(
            _coerce_enum(value, OutputMode, "output_modes") for value in self.output_modes
        )
        if not output_modes or len(set(output_modes)) != len(output_modes):
            raise ValueError("output_modes must contain unique supported modes")
        object.__setattr__(self, "output_modes", tuple(sorted(output_modes, key=lambda x: x.value)))
        for field_name in ("max_context_tokens", "max_batch_size"):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "max_verify_tokens",
            _nonnegative_int(self.max_verify_tokens, "max_verify_tokens"),
        )
        for field_name in (
            "transactional_state",
            "scratch_only_steps",
            "independently_committable_rows",
            "supports_ragged_batches",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be boolean")

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_id": self.backend_id,
            "backend_abi": self.backend_abi,
            "implementation_version": self.implementation_version,
            "fabric": self.fabric,
            "memory_domain": self.memory_domain.value,
            "architectures": list(self.architectures),
            "operator_ids": list(self.operator_ids),
            "codecs": [codec.as_dict() for codec in self.codecs],
            "state_abis": list(self.state_abis),
            "output_modes": [mode.value for mode in self.output_modes],
            "numerical_contracts": list(self.numerical_contracts),
            "max_context_tokens": self.max_context_tokens,
            "max_batch_size": self.max_batch_size,
            "max_verify_tokens": self.max_verify_tokens,
            "transactional_state": self.transactional_state,
            "scratch_only_steps": self.scratch_only_steps,
            "independently_committable_rows": self.independently_committable_rows,
            "supports_ragged_batches": self.supports_ragged_batches,
            "telemetry_counters": list(self.telemetry_counters),
            "promotion_status": self.promotion_status.value,
        }


@dataclass(frozen=True, slots=True)
class DeviceDescriptor:
    """One exact hardware/memory snapshot used during placement."""

    device_id: str
    fabric: str
    memory_domain: MemoryDomain | str
    total_bytes: int
    available_bytes: int
    machine_fingerprint: str
    attributes: tuple[tuple[str, str], ...] = ()
    schema_version: str = DEVICE_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DEVICE_DESCRIPTOR_SCHEMA:
            raise ValueError(f"unsupported device-descriptor schema: {self.schema_version}")
        for field_name in ("device_id", "fabric"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "memory_domain",
            _coerce_enum(self.memory_domain, MemoryDomain, "memory_domain"),
        )
        object.__setattr__(self, "total_bytes", _positive_int(self.total_bytes, "total_bytes"))
        object.__setattr__(
            self,
            "available_bytes",
            _nonnegative_int(self.available_bytes, "available_bytes"),
        )
        if self.available_bytes > self.total_bytes:
            raise ValueError("available_bytes cannot exceed total_bytes")
        object.__setattr__(
            self,
            "machine_fingerprint",
            _require_sha256(self.machine_fingerprint, "machine_fingerprint"),
        )
        object.__setattr__(
            self,
            "attributes",
            _string_pairs(self.attributes, "device attributes"),
        )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "device_id": self.device_id,
            "fabric": self.fabric,
            "memory_domain": self.memory_domain.value,
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "machine_fingerprint": self.machine_fingerprint,
            "attributes": {key: value for key, value in self.attributes},
        }


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Backend-neutral service shape and semantics used for capability matching."""

    max_batch_size: int
    max_context_tokens: int
    verify_tokens: int
    output_mode: OutputMode | str
    numerical_contract: str
    state_abi: str
    required_component_roles: tuple[str, ...]
    require_native_codecs: bool = True
    workspace_bytes: int = 0
    headroom_bytes: int = 0
    fallback_policy: FallbackPolicy | str = FallbackPolicy.DENY
    schema_version: str = WORKLOAD_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != WORKLOAD_SPEC_SCHEMA:
            raise ValueError(f"unsupported workload schema: {self.schema_version}")
        object.__setattr__(
            self,
            "max_batch_size",
            _positive_int(self.max_batch_size, "max_batch_size"),
        )
        object.__setattr__(
            self,
            "max_context_tokens",
            _positive_int(self.max_context_tokens, "max_context_tokens"),
        )
        object.__setattr__(
            self,
            "verify_tokens",
            _nonnegative_int(self.verify_tokens, "verify_tokens"),
        )
        object.__setattr__(
            self,
            "output_mode",
            _coerce_enum(self.output_mode, OutputMode, "output_mode"),
        )
        object.__setattr__(
            self,
            "fallback_policy",
            _coerce_enum(self.fallback_policy, FallbackPolicy, "fallback_policy"),
        )
        for field_name in ("numerical_contract", "state_abi"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "required_component_roles",
            _names(self.required_component_roles, "required_component_roles"),
        )
        if type(self.require_native_codecs) is not bool:
            raise TypeError("require_native_codecs must be boolean")
        object.__setattr__(
            self,
            "workspace_bytes",
            _nonnegative_int(self.workspace_bytes, "workspace_bytes"),
        )
        object.__setattr__(
            self,
            "headroom_bytes",
            _nonnegative_int(self.headroom_bytes, "headroom_bytes"),
        )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_batch_size": self.max_batch_size,
            "max_context_tokens": self.max_context_tokens,
            "verify_tokens": self.verify_tokens,
            "output_mode": self.output_mode.value,
            "numerical_contract": self.numerical_contract,
            "state_abi": self.state_abi,
            "required_component_roles": list(self.required_component_roles),
            "require_native_codecs": self.require_native_codecs,
            "workspace_bytes": self.workspace_bytes,
            "headroom_bytes": self.headroom_bytes,
            "fallback_policy": self.fallback_policy.value,
        }


@dataclass(frozen=True, slots=True)
class ComponentPlacement:
    """Placement of one unique physical allocation and all of its logical roles."""

    allocation_id: str
    component_ids: tuple[str, ...]
    roles: tuple[str, ...]
    codec_id: str
    layout_id: str
    memory_domain: MemoryDomain | str
    residency: Residency | str
    physical_bytes: int

    def __post_init__(self) -> None:
        for field_name in ("allocation_id", "codec_id", "layout_id"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "component_ids", _names(self.component_ids, "component_ids"))
        object.__setattr__(self, "roles", _names(self.roles, "roles"))
        object.__setattr__(
            self,
            "memory_domain",
            _coerce_enum(self.memory_domain, MemoryDomain, "memory_domain"),
        )
        object.__setattr__(
            self,
            "residency",
            _coerce_enum(self.residency, Residency, "residency"),
        )
        object.__setattr__(
            self,
            "physical_bytes",
            _positive_int(self.physical_bytes, "physical_bytes"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "component_ids": list(self.component_ids),
            "roles": list(self.roles),
            "codec_id": self.codec_id,
            "layout_id": self.layout_id,
            "memory_domain": self.memory_domain.value,
            "residency": self.residency.value,
            "physical_bytes": self.physical_bytes,
        }


@dataclass(frozen=True, slots=True)
class StatePlacement:
    state_abi: str
    dtype: str
    memory_domain: MemoryDomain | str
    bytes_per_token: int
    reserved_bytes: int
    max_batch_size: int
    max_context_tokens: int
    fixed_bytes_per_row: int = 0

    def __post_init__(self) -> None:
        for field_name in ("state_abi", "dtype"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "memory_domain",
            _coerce_enum(self.memory_domain, MemoryDomain, "memory_domain"),
        )
        for field_name in ("bytes_per_token", "reserved_bytes", "fixed_bytes_per_row"):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_int(getattr(self, field_name), field_name),
            )
        for field_name in ("max_batch_size", "max_context_tokens"):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name),
            )
        if self.bytes_per_token == 0 and self.fixed_bytes_per_row == 0:
            raise ValueError("state placement must reserve token-scaled or fixed per-row state")
        expected = self.max_batch_size * (
            self.fixed_bytes_per_row + self.bytes_per_token * self.max_context_tokens
        )
        if self.reserved_bytes != expected:
            raise ValueError(
                "state reserved_bytes must equal batch * "
                "(fixed_bytes_per_row + bytes_per_token * context)"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_abi": self.state_abi,
            "dtype": self.dtype,
            "memory_domain": self.memory_domain.value,
            "bytes_per_token": self.bytes_per_token,
            "fixed_bytes_per_row": self.fixed_bytes_per_row,
            "reserved_bytes": self.reserved_bytes,
            "max_batch_size": self.max_batch_size,
            "max_context_tokens": self.max_context_tokens,
        }


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    """Content-addressed result of fail-closed capability and resident-memory planning."""

    model_fingerprint: str
    capability_fingerprint: str
    device_fingerprint: str
    workload_fingerprint: str
    backend_id: str
    device_id: str
    components: tuple[ComponentPlacement, ...]
    state: StatePlacement
    workspace_bytes: int
    headroom_bytes: int
    model_resident_bytes: int
    total_reserved_bytes: int
    memory_budget_bytes: int
    fully_resident: bool
    fallback_policy: FallbackPolicy | str
    performance_claim_valid: bool
    schema_version: str = PLACEMENT_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PLACEMENT_PLAN_SCHEMA:
            raise ValueError(f"unsupported placement-plan schema: {self.schema_version}")
        for field_name in (
            "model_fingerprint",
            "capability_fingerprint",
            "device_fingerprint",
            "workload_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        for field_name in ("backend_id", "device_id"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        components = tuple(self.components)
        if not components or any(not isinstance(value, ComponentPlacement) for value in components):
            raise TypeError("placement components must be non-empty ComponentPlacement values")
        if len({value.allocation_id for value in components}) != len(components):
            raise ValueError("placement must name each physical allocation exactly once")
        object.__setattr__(
            self,
            "components",
            tuple(sorted(components, key=lambda value: value.allocation_id)),
        )
        if not isinstance(self.state, StatePlacement):
            raise TypeError("placement state must be a StatePlacement")
        for field_name in (
            "workspace_bytes",
            "headroom_bytes",
            "model_resident_bytes",
            "total_reserved_bytes",
            "memory_budget_bytes",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_int(getattr(self, field_name), field_name),
            )
        if self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        computed_model = sum(value.physical_bytes for value in self.components)
        if computed_model != self.model_resident_bytes:
            raise ValueError("model_resident_bytes does not match component placements")
        computed_total = (
            self.model_resident_bytes
            + self.state.reserved_bytes
            + self.workspace_bytes
            + self.headroom_bytes
        )
        if computed_total != self.total_reserved_bytes:
            raise ValueError("total_reserved_bytes does not match placement accounting")
        if self.total_reserved_bytes > self.memory_budget_bytes:
            raise ValueError("placement exceeds its memory budget")
        for field_name in ("fully_resident", "performance_claim_valid"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be boolean")
        object.__setattr__(
            self,
            "fallback_policy",
            _coerce_enum(self.fallback_policy, FallbackPolicy, "fallback_policy"),
        )
        if self.performance_claim_valid and not self.fully_resident:
            raise ValueError("a non-resident placement cannot carry a native performance claim")

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_fingerprint": self.model_fingerprint,
            "capability_fingerprint": self.capability_fingerprint,
            "device_fingerprint": self.device_fingerprint,
            "workload_fingerprint": self.workload_fingerprint,
            "backend_id": self.backend_id,
            "device_id": self.device_id,
            "components": [component.as_dict() for component in self.components],
            "state": self.state.as_dict(),
            "workspace_bytes": self.workspace_bytes,
            "headroom_bytes": self.headroom_bytes,
            "model_resident_bytes": self.model_resident_bytes,
            "total_reserved_bytes": self.total_reserved_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "fully_resident": self.fully_resident,
            "fallback_policy": self.fallback_policy.value,
            "performance_claim_valid": self.performance_claim_valid,
        }


@dataclass(frozen=True, slots=True)
class RuntimeRoute:
    """Exact live binding of model, implementation, device, and placement identities."""

    runtime_id: str
    model_fingerprint: str
    capability_fingerprint: str
    placement_fingerprint: str
    backend_id: str
    device_id: str
    promotion_status: PromotionStatus | str
    effective_numerical_contract: str | None = None
    execution_shape_fingerprint: str | None = None
    schema_version: str = RUNTIME_ROUTE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_ROUTE_SCHEMA:
            raise ValueError(f"unsupported runtime-route schema: {self.schema_version}")
        for field_name in ("runtime_id", "backend_id", "device_id"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        for field_name in (
            "model_fingerprint",
            "capability_fingerprint",
            "placement_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "promotion_status",
            _coerce_enum(self.promotion_status, PromotionStatus, "promotion_status"),
        )
        if (self.effective_numerical_contract is None) != (
            self.execution_shape_fingerprint is None
        ):
            raise ValueError(
                "runtime route effective numerical contract and execution shape must co-occur"
            )
        if self.effective_numerical_contract is not None:
            object.__setattr__(
                self,
                "effective_numerical_contract",
                _require_name(
                    self.effective_numerical_contract,
                    "effective_numerical_contract",
                ),
            )
            object.__setattr__(
                self,
                "execution_shape_fingerprint",
                _require_sha256(
                    self.execution_shape_fingerprint,
                    "execution_shape_fingerprint",
                ),
            )


@dataclass(frozen=True, slots=True)
class CompatibleBatchLaneIdentity:
    """Explicit identity for an opt-in compatible-request execution lane.

    A lane is deliberately separate from :class:`BackendCapabilities`: attaching one to a
    generation service cannot silently promote or alter the runtime's exact B1 contract.  The
    numerical contract and promotion status describe the pooled arithmetic itself.
    ``dispatches_singletons`` is an execution-policy fact: false preserves the runtime's direct
    B1 route, while true authorizes the coordinator to submit singleton work to this lane.
    """

    lane_id: str
    runtime_id: str
    lane_abi: str
    numerical_contract: str
    promotion_status: PromotionStatus | str
    max_batch_size: int
    max_queue_delay_seconds: float
    max_scratch_bytes: int
    supports_ragged_dispatch: bool
    schema_version: str = COMPATIBLE_BATCH_LANE_SCHEMA
    dispatches_singletons: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != COMPATIBLE_BATCH_LANE_SCHEMA:
            raise ValueError(f"unsupported compatible-batch lane schema: {self.schema_version}")
        for field_name in ("lane_id", "runtime_id", "lane_abi", "numerical_contract"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "promotion_status",
            _coerce_enum(self.promotion_status, PromotionStatus, "promotion_status"),
        )
        object.__setattr__(
            self,
            "max_batch_size",
            _positive_int(self.max_batch_size, "max_batch_size"),
        )
        object.__setattr__(
            self,
            "max_queue_delay_seconds",
            _nonnegative_float(self.max_queue_delay_seconds, "max_queue_delay_seconds"),
        )
        object.__setattr__(
            self,
            "max_scratch_bytes",
            _positive_int(self.max_scratch_bytes, "max_scratch_bytes"),
        )
        if type(self.supports_ragged_dispatch) is not bool:
            raise TypeError("supports_ragged_dispatch must be boolean")
        if type(self.dispatches_singletons) is not bool:
            raise TypeError("dispatches_singletons must be boolean")


@dataclass(frozen=True, slots=True)
class StateObservation:
    """Neutral version observation for a backend-owned mutable state allocation."""

    runtime_id: str
    state_id: str
    generation: int
    epoch: int
    lengths: tuple[int, ...]
    capacity: int
    state_abi: str
    storage_generation: int
    schema_version: str = STATE_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATE_OBSERVATION_SCHEMA:
            raise ValueError(f"unsupported state-observation schema: {self.schema_version}")
        for field_name in ("runtime_id", "state_id", "state_abi"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        for field_name in ("generation", "epoch", "storage_generation"):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "capacity", _positive_int(self.capacity, "capacity"))
        lengths = tuple(_nonnegative_int(value, "lengths[]") for value in self.lengths)
        if not lengths:
            raise ValueError("state lengths cannot be empty")
        if any(value > self.capacity for value in lengths):
            raise ValueError("state lengths cannot exceed capacity")
        object.__setattr__(self, "lengths", lengths)

    @property
    def batch_size(self) -> int:
        return len(self.lengths)


@runtime_checkable
class StateHandle(Protocol):
    """Opaque backend-owned state authority exposed to the common coordinator."""

    @property
    def runtime_id(self) -> str: ...

    @property
    def state_id(self) -> str: ...

    @property
    def owner_id(self) -> str: ...

    def observe(self) -> StateObservation: ...


@dataclass(frozen=True, slots=True)
class StateForkResult:
    """Receipt for an exact committed-prefix copy into fresh backend-owned state.

    The receipt deliberately exposes only opaque state authority and neutral observations.
    Backends must copy committed K/V prefixes directly inside their native memory domain; no
    tensor, array, pointer, or serialized cache payload is part of this contract.
    """

    runtime_id: str
    source: StateObservation
    forked: StateObservation
    state: StateHandle = field(repr=False, compare=False)
    state_bytes_copied: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_id", _require_name(self.runtime_id, "runtime_id"))
        if not isinstance(self.source, StateObservation) or not isinstance(
            self.forked,
            StateObservation,
        ):
            raise TypeError("state fork source/forked values must be StateObservation values")
        if not isinstance(self.state, StateHandle):
            raise TypeError("state fork requires an opaque StateHandle")
        if self.source.runtime_id != self.runtime_id or self.forked.runtime_id != self.runtime_id:
            raise ValueError("state fork observations must belong to the receipt runtime")
        if self.state.runtime_id != self.runtime_id or self.state.state_id != self.forked.state_id:
            raise ValueError("forked state authority does not match its observation")
        if self.source.state_id == self.forked.state_id:
            raise ValueError("state fork must mint a distinct state authority")
        if self.forked.generation <= self.source.generation:
            raise ValueError("state fork must mint a fresh authority generation")
        if self.source.state_abi != self.forked.state_abi:
            raise ValueError("state fork cannot cross a state ABI boundary")
        if self.source.batch_size != self.forked.batch_size:
            raise ValueError("state fork must preserve source rows")
        if self.source.lengths != self.forked.lengths:
            raise ValueError("state fork must preserve the exact committed prefix lengths")
        if any(length > self.forked.capacity for length in self.source.lengths):
            raise ValueError("forked state capacity cannot hold the committed prefix")
        object.__setattr__(
            self,
            "state_bytes_copied",
            _nonnegative_int(self.state_bytes_copied, "state_bytes_copied"),
        )


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """Immutable token-selection policy with a concrete request-owned seed.

    The native executor never consumes a process-global framework RNG.  ``seed`` and the
    per-step counter in :class:`SamplingRequest` completely identify the uniform variate for a
    token, so coordinator interleaving cannot change a request's trajectory.
    """

    seed: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: tuple[tuple[int, float], ...] = ()
    sampling_abi: str = NATIVE_SAMPLING_ABI

    def __post_init__(self) -> None:
        if self.sampling_abi != NATIVE_SAMPLING_ABI:
            raise ValueError(f"unsupported native sampling ABI: {self.sampling_abi!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("sampling seed must be a strict integer")
        if self.seed < -(1 << 63) or self.seed > (1 << 63) - 1:
            raise ValueError("sampling seed must fit in a signed 64-bit integer")
        for field_name, minimum, maximum in (
            ("temperature", 0.0, 2.0),
            ("top_p", 0.0, 1.0),
            ("frequency_penalty", -2.0, 2.0),
            ("presence_penalty", -2.0, 2.0),
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite number")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < minimum or normalized > maximum:
                raise ValueError(f"{field_name} must be in [{minimum}, {maximum}]")
            if field_name == "top_p" and normalized == 0.0:
                raise ValueError("top_p must be greater than zero")
            object.__setattr__(self, field_name, normalized)
        object.__setattr__(self, "top_k", _nonnegative_int(self.top_k, "top_k"))
        biases: list[tuple[int, float]] = []
        for token_id, bias in self.logit_bias:
            token = _nonnegative_int(token_id, "logit_bias token ID")
            if isinstance(bias, bool) or not isinstance(bias, (int, float)):
                raise TypeError("logit_bias values must be finite numbers")
            normalized = float(bias)
            if not math.isfinite(normalized) or normalized < -100.0 or normalized > 100.0:
                raise ValueError("logit_bias values must be in [-100, 100]")
            biases.append((token, normalized))
        if len({token for token, _ in biases}) != len(biases):
            raise ValueError("logit_bias token IDs must be unique")
        object.__setattr__(self, "logit_bias", tuple(sorted(biases)))

    @property
    def raw_argmax_equivalent(self) -> bool:
        """Whether selection is exactly the established unmodified argmax path."""

        return bool(
            self.temperature == 0.0
            and self.frequency_penalty == 0.0
            and self.presence_penalty == 0.0
            and not self.logit_bias
        )


@dataclass(frozen=True, slots=True)
class SamplingRequest:
    """Dynamic row-local inputs for one stateless native token selection."""

    policy: SamplingPolicy
    token_counts: tuple[tuple[int, int], ...]
    rng_counter: int

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SamplingPolicy):
            raise TypeError("sampling request requires SamplingPolicy")
        counts = tuple(
            (
                _nonnegative_int(token_id, "token_counts token ID"),
                _positive_int(count, "token_counts count"),
            )
            for token_id, count in self.token_counts
        )
        if len({token for token, _ in counts}) != len(counts):
            raise ValueError("token_counts token IDs must be unique")
        object.__setattr__(self, "token_counts", tuple(sorted(counts)))
        object.__setattr__(
            self,
            "rng_counter",
            _nonnegative_int(self.rng_counter, "rng_counter"),
        )


@dataclass(frozen=True, slots=True)
class OutputRequest:
    mode: OutputMode | str
    selected_token_ids: tuple[int, ...] = ()
    candidate_token_ids: tuple[tuple[int, ...], ...] = ()
    sampling: tuple[SamplingRequest, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _coerce_enum(self.mode, OutputMode, "output mode"))
        selected = tuple(
            _nonnegative_int(value, "selected_token_ids[]") for value in self.selected_token_ids
        )
        if len(set(selected)) != len(selected):
            raise ValueError("selected_token_ids must be unique")
        candidates = tuple(
            tuple(_nonnegative_int(value, "candidate_token_ids[][]") for value in row)
            for row in self.candidate_token_ids
        )
        if any(len(row) < 2 or len(set(row)) != len(row) for row in candidates):
            raise ValueError("candidate rows require at least two unique token IDs")
        if self.mode is OutputMode.SELECTED_LOGITS and not selected:
            raise ValueError("selected-logits output requires selected_token_ids")
        if self.mode is not OutputMode.SELECTED_LOGITS and selected:
            raise ValueError("selected_token_ids are legal only for selected-logits output")
        if self.mode is OutputMode.CANDIDATE_ARGMAX_MARGIN and not candidates:
            raise ValueError("candidate output requires candidate_token_ids")
        if self.mode is not OutputMode.CANDIDATE_ARGMAX_MARGIN and candidates:
            raise ValueError("candidate_token_ids are legal only for candidate output")
        sampling = tuple(self.sampling)
        if any(not isinstance(value, SamplingRequest) for value in sampling):
            raise TypeError("sampling rows must contain SamplingRequest values")
        if self.mode is OutputMode.NEXT_TOKEN_SAMPLE and not sampling:
            raise ValueError("next-token-sample output requires row sampling requests")
        if self.mode is not OutputMode.NEXT_TOKEN_SAMPLE and sampling:
            raise ValueError("sampling rows are legal only for next-token-sample output")
        object.__setattr__(self, "selected_token_ids", selected)
        object.__setattr__(self, "candidate_token_ids", candidates)
        object.__setattr__(self, "sampling", sampling)


@dataclass(frozen=True, slots=True)
class HostBuffer:
    """Explicit diagnostic host materialization; never a hidden native tensor escape."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _require_name(self.dtype, "host buffer dtype"))
        shape = tuple(_positive_int(value, "host buffer shape[]") for value in self.shape)
        if not shape:
            raise ValueError("host buffer shape cannot be empty")
        if type(self.data) is not bytes:
            raise TypeError("host buffer data must be immutable bytes")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True, slots=True)
class NativeOutput:
    """Framework-free output returned with a provisional state delta."""

    mode: OutputMode | str
    token_ids: tuple[int, ...] = ()
    values: HostBuffer | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _coerce_enum(self.mode, OutputMode, "output mode"))
        tokens = tuple(_nonnegative_int(value, "output token_ids[]") for value in self.token_ids)
        token_mode = self.mode in (
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        )
        if token_mode != bool(tokens):
            raise ValueError("next-token output requires tokens and other modes cannot carry them")
        if token_mode and self.values is not None:
            raise ValueError("next-token output cannot hide a host logit materialization")
        if not token_mode and not isinstance(self.values, HostBuffer):
            raise TypeError("non-token outputs require an explicit HostBuffer")
        object.__setattr__(self, "token_ids", tokens)


def _normalize_work_rows(
    request_ids: Sequence[str],
    token_rows: Sequence[Sequence[int]],
    parent: StateObservation,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    normalized_requests = _names(request_ids, "request_ids")
    rows = tuple(
        tuple(_nonnegative_int(value, "token_rows[][]") for value in row) for row in token_rows
    )
    if not rows or any(not row for row in rows):
        raise ValueError("token_rows must contain non-empty token sequences")
    if len(rows) != parent.batch_size or len(normalized_requests) != parent.batch_size:
        raise ValueError("request IDs, token rows, and state batch must align")
    return normalized_requests, rows


def _validate_work_state(state: StateHandle, parent: StateObservation) -> None:
    if not isinstance(state, StateHandle):
        raise TypeError("work requires a StateHandle")
    if state.runtime_id != parent.runtime_id or state.state_id != parent.state_id:
        raise ValueError("state handle identity does not match its parent observation")


@dataclass(frozen=True, slots=True)
class PrefillWork:
    request_ids: tuple[str, ...]
    token_rows: tuple[tuple[int, ...], ...]
    state: StateHandle = field(repr=False, compare=False)
    parent: StateObservation
    output: OutputRequest

    def __post_init__(self) -> None:
        if not isinstance(self.parent, StateObservation):
            raise TypeError("prefill parent must be a StateObservation")
        if any(self.parent.lengths):
            raise ValueError("prefill requires empty committed state")
        _validate_work_state(self.state, self.parent)
        request_ids, rows = _normalize_work_rows(
            self.request_ids,
            self.token_rows,
            self.parent,
        )
        if not isinstance(self.output, OutputRequest):
            raise TypeError("prefill output must be an OutputRequest")
        if self.output.mode is OutputMode.NEXT_TOKEN_SAMPLE and (
            len(self.output.sampling) != self.parent.batch_size
        ):
            raise ValueError("sampling rows must align with the state batch")
        if any(len(row) > self.parent.capacity for row in rows):
            raise OverflowError("prefill token rows exceed state capacity")
        object.__setattr__(self, "request_ids", request_ids)
        object.__setattr__(self, "token_rows", rows)


@dataclass(frozen=True, slots=True)
class DecodeWork:
    request_ids: tuple[str, ...]
    token_rows: tuple[tuple[int, ...], ...]
    state: StateHandle = field(repr=False, compare=False)
    parent: StateObservation
    output: OutputRequest

    def __post_init__(self) -> None:
        if not isinstance(self.parent, StateObservation):
            raise TypeError("decode parent must be a StateObservation")
        if any(length <= 0 for length in self.parent.lengths):
            raise ValueError("decode requires a committed prefix for every row")
        _validate_work_state(self.state, self.parent)
        request_ids, rows = _normalize_work_rows(
            self.request_ids,
            self.token_rows,
            self.parent,
        )
        if not isinstance(self.output, OutputRequest):
            raise TypeError("decode output must be an OutputRequest")
        if self.output.mode is OutputMode.NEXT_TOKEN_SAMPLE and (
            len(self.output.sampling) != self.parent.batch_size
        ):
            raise ValueError("sampling rows must align with the state batch")
        if any(
            length + len(row) > self.parent.capacity
            for length, row in zip(self.parent.lengths, rows, strict=True)
        ):
            raise OverflowError("decode token rows exceed state capacity")
        object.__setattr__(self, "request_ids", request_ids)
        object.__setattr__(self, "token_rows", rows)


@dataclass(frozen=True, slots=True)
class GreedyBlockVerifyWork:
    """Transactional B1 teacher-forced block verification request.

    Every input position produces its own raw-greedy target prediction.  The state delta remains
    provisional until an explicit prefix is committed; this is deliberately distinct from
    :class:`DecodeWork`, whose token output contains one final prediction per batch row.
    """

    request_id: str
    token_ids: tuple[int, ...]
    state: StateHandle = field(repr=False, compare=False)
    parent: StateObservation

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _require_name(self.request_id, "request_id"))
        if not isinstance(self.parent, StateObservation):
            raise TypeError("greedy block parent must be a StateObservation")
        if self.parent.batch_size != 1:
            raise ValueError("greedy block verification is strictly B1")
        if self.parent.lengths[0] <= 0:
            raise ValueError("greedy block verification requires a committed prefix")
        _validate_work_state(self.state, self.parent)
        tokens = tuple(_nonnegative_int(value, "token_ids[]") for value in self.token_ids)
        if not tokens:
            raise ValueError("greedy block token_ids cannot be empty")
        if self.parent.lengths[0] + len(tokens) > self.parent.capacity:
            raise OverflowError("greedy block token IDs exceed state capacity")
        object.__setattr__(self, "token_ids", tokens)


@dataclass(frozen=True, slots=True)
class GreedyProposalBeginWork:
    """Begin an in-place B1 draft proposal transaction at one exact committed parent."""

    request_id: str
    state: StateHandle = field(repr=False, compare=False)
    parent: StateObservation

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _require_name(self.request_id, "request_id"))
        if not isinstance(self.parent, StateObservation):
            raise TypeError("greedy proposal parent must be a StateObservation")
        if self.parent.batch_size != 1:
            raise ValueError("greedy proposal transactions are strictly B1")
        if self.parent.lengths[0] <= 0:
            raise ValueError("greedy proposal transaction requires a committed prefix")
        _validate_work_state(self.state, self.parent)


@runtime_checkable
class ProvisionalAuthority(Protocol):
    """Opaque, backend-issued authority consumed exactly once by commit or abandon."""

    @property
    def runtime_id(self) -> str: ...

    @property
    def step_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ProvisionalStep:
    """Scratch-only execution result bound to one exact parent state observation."""

    runtime_id: str
    step_id: str
    request_ids: tuple[str, ...]
    state: StateHandle = field(repr=False, compare=False)
    parent: StateObservation
    token_counts: tuple[int, ...]
    output: NativeOutput
    authority: ProvisionalAuthority = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for field_name in ("runtime_id", "step_id"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        if not isinstance(self.parent, StateObservation):
            raise TypeError("provisional parent must be a StateObservation")
        _validate_work_state(self.state, self.parent)
        request_ids = _names(self.request_ids, "request_ids")
        if len(request_ids) != self.parent.batch_size:
            raise ValueError("provisional request IDs must align with state batch")
        counts = tuple(_positive_int(value, "token_counts[]") for value in self.token_counts)
        if len(counts) != self.parent.batch_size:
            raise ValueError("provisional token counts must align with state batch")
        if any(
            length + count > self.parent.capacity
            for length, count in zip(self.parent.lengths, counts, strict=True)
        ):
            raise OverflowError("provisional delta exceeds state capacity")
        if not isinstance(self.output, NativeOutput):
            raise TypeError("provisional output must be a NativeOutput")
        if self.output.mode in (OutputMode.NEXT_TOKEN_ARGMAX, OutputMode.NEXT_TOKEN_SAMPLE) and (
            len(self.output.token_ids) != self.parent.batch_size
        ):
            raise ValueError("next-token output must contain one token per state row")
        if not isinstance(self.authority, ProvisionalAuthority):
            raise TypeError("provisional step requires a ProvisionalAuthority")
        if self.authority.runtime_id != self.runtime_id or self.authority.step_id != self.step_id:
            raise ValueError("provisional authority identity does not match the step")
        if self.runtime_id != self.parent.runtime_id:
            raise ValueError("provisional runtime does not match parent state")
        object.__setattr__(self, "request_ids", request_ids)
        object.__setattr__(self, "token_counts", counts)


@dataclass(frozen=True, slots=True)
class GreedyBlockVerification:
    """Opaque provisional B1 suffix plus one on-device argmax result per input position.

    ``predicted_token_ids[i]`` is the target model's raw argmax after consuming
    ``input_token_ids[i]`` under the preceding teacher-forced prefix.  Only these selected token
    IDs cross the backend boundary; the contract has no field capable of carrying host logits.
    """

    runtime_id: str
    step_id: str
    request_id: str
    state: StateHandle = field(repr=False, compare=False)
    parent: StateObservation
    input_token_ids: tuple[int, ...]
    predicted_token_ids: tuple[int, ...]
    authority: ProvisionalAuthority = field(repr=False, compare=False)
    schema_version: str = GREEDY_BLOCK_VERIFICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GREEDY_BLOCK_VERIFICATION_SCHEMA:
            raise ValueError(f"unsupported greedy-block schema: {self.schema_version}")
        for field_name in ("runtime_id", "step_id", "request_id"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        if not isinstance(self.parent, StateObservation):
            raise TypeError("greedy block parent must be a StateObservation")
        if self.parent.batch_size != 1 or self.parent.lengths[0] <= 0:
            raise ValueError("greedy block verification must bind committed B1 state")
        _validate_work_state(self.state, self.parent)
        inputs = tuple(
            _nonnegative_int(value, "input_token_ids[]") for value in self.input_token_ids
        )
        predictions = tuple(
            _nonnegative_int(value, "predicted_token_ids[]") for value in self.predicted_token_ids
        )
        if not inputs:
            raise ValueError("greedy block input_token_ids cannot be empty")
        if len(predictions) != len(inputs):
            raise ValueError("greedy block must return one prediction per input position")
        if self.parent.lengths[0] + len(inputs) > self.parent.capacity:
            raise OverflowError("greedy block provisional suffix exceeds state capacity")
        if not isinstance(self.authority, ProvisionalAuthority):
            raise TypeError("greedy block verification requires a ProvisionalAuthority")
        if self.authority.runtime_id != self.runtime_id or self.authority.step_id != self.step_id:
            raise ValueError("greedy block authority identity does not match the verification")
        if self.runtime_id != self.parent.runtime_id:
            raise ValueError("greedy block runtime does not match parent state")
        object.__setattr__(self, "input_token_ids", inputs)
        object.__setattr__(self, "predicted_token_ids", predictions)

    @property
    def token_count(self) -> int:
        return len(self.input_token_ids)


@dataclass(frozen=True, slots=True)
class GreedyProposalTransaction:
    """Versioned snapshot of an in-place provisional draft K/V suffix.

    An unsealed transaction has one raw-greedy prediction for every appended input.  ``seal``
    appends exactly one final input without selecting another token, so a sealed transaction has
    one more input than prediction.  Only the latest snapshot is executable or terminal.
    """

    runtime_id: str
    step_id: str
    request_id: str
    state: StateHandle = field(repr=False, compare=False)
    parent: StateObservation
    input_token_ids: tuple[int, ...]
    predicted_token_ids: tuple[int, ...]
    sealed: bool
    authority: ProvisionalAuthority = field(repr=False, compare=False)
    schema_version: str = GREEDY_PROPOSAL_TRANSACTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GREEDY_PROPOSAL_TRANSACTION_SCHEMA:
            raise ValueError(f"unsupported greedy-proposal schema: {self.schema_version}")
        for field_name in ("runtime_id", "step_id", "request_id"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        if not isinstance(self.parent, StateObservation):
            raise TypeError("greedy proposal parent must be a StateObservation")
        if self.parent.batch_size != 1 or self.parent.lengths[0] <= 0:
            raise ValueError("greedy proposal transaction must bind committed B1 state")
        _validate_work_state(self.state, self.parent)
        inputs = tuple(
            _nonnegative_int(value, "input_token_ids[]") for value in self.input_token_ids
        )
        predictions = tuple(
            _nonnegative_int(value, "predicted_token_ids[]") for value in self.predicted_token_ids
        )
        if type(self.sealed) is not bool:
            raise TypeError("greedy proposal sealed flag must be boolean")
        if self.sealed:
            if not inputs or len(inputs) != len(predictions) + 1:
                raise ValueError("sealed greedy proposal requires one final unselected input")
        elif len(inputs) != len(predictions):
            raise ValueError("open greedy proposal requires one prediction per input")
        if self.parent.lengths[0] + len(inputs) > self.parent.capacity:
            raise OverflowError("greedy proposal suffix exceeds state capacity")
        if not isinstance(self.authority, ProvisionalAuthority):
            raise TypeError("greedy proposal transaction requires a ProvisionalAuthority")
        if self.authority.runtime_id != self.runtime_id or self.authority.step_id != self.step_id:
            raise ValueError("greedy proposal authority identity does not match transaction")
        if self.runtime_id != self.parent.runtime_id:
            raise ValueError("greedy proposal runtime does not match parent state")
        object.__setattr__(self, "input_token_ids", inputs)
        object.__setattr__(self, "predicted_token_ids", predictions)

    @property
    def version(self) -> int:
        return len(self.input_token_ids)

    @property
    def input_count(self) -> int:
        return len(self.input_token_ids)

    @property
    def latest_prediction(self) -> int:
        if not self.predicted_token_ids:
            raise ValueError("greedy proposal transaction has no selected token")
        return self.predicted_token_ids[-1]


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Terminal receipt for one explicit accepted prefix of a provisional step."""

    runtime_id: str
    step_id: str
    state_id: str
    accepted_counts: tuple[int, ...]
    before: StateObservation
    after: StateObservation
    state_bytes_written: int

    def __post_init__(self) -> None:
        for field_name in ("runtime_id", "step_id", "state_id"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        if not isinstance(self.before, StateObservation) or not isinstance(
            self.after,
            StateObservation,
        ):
            raise TypeError("commit before/after must be StateObservation values")
        if (
            self.runtime_id != self.before.runtime_id
            or self.runtime_id != self.after.runtime_id
            or self.state_id != self.before.state_id
            or self.state_id != self.after.state_id
        ):
            raise ValueError("commit identity does not match before/after observations")
        if (
            self.after.generation != self.before.generation
            or self.after.capacity != self.before.capacity
            or self.after.state_abi != self.before.state_abi
            or self.after.storage_generation != self.before.storage_generation
        ):
            raise ValueError("commit cannot replace state authority or backing storage")
        if self.after.epoch != self.before.epoch + 1:
            raise ValueError("commit must advance the state epoch exactly once")
        accepted = tuple(
            _nonnegative_int(value, "accepted_counts[]") for value in self.accepted_counts
        )
        if len(accepted) != self.before.batch_size:
            raise ValueError("accepted_counts must align with state batch")
        expected_lengths = tuple(
            length + count for length, count in zip(self.before.lengths, accepted, strict=True)
        )
        if self.after.lengths != expected_lengths:
            raise ValueError("commit lengths do not equal the accepted provisional prefix")
        object.__setattr__(self, "accepted_counts", accepted)
        object.__setattr__(
            self,
            "state_bytes_written",
            _nonnegative_int(self.state_bytes_written, "state_bytes_written"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeTelemetry:
    """Shared meanings for native runtime counters; backend extras remain namespaced."""

    runtime_id: str
    route_backend_id: str
    model_fingerprint: str
    placement_fingerprint: str
    prefill_calls: int = 0
    prefill_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_calls: int = 0
    decode_tokens: int = 0
    decode_seconds: float = 0.0
    provisional_steps: int = 0
    commits: int = 0
    abandons: int = 0
    committed_tokens: int = 0
    physical_weight_bytes_read: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    model_resident_bytes: int = 0
    kv_resident_bytes: int = 0
    workspace_peak_bytes: int = 0
    unexpected_fallbacks: int = 0
    unexpected_page_loads: int = 0
    extra_counters: tuple[tuple[str, int], ...] = ()
    schema_version: str = RUNTIME_TELEMETRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_TELEMETRY_SCHEMA:
            raise ValueError(f"unsupported runtime-telemetry schema: {self.schema_version}")
        for field_name in ("runtime_id", "route_backend_id"):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        for field_name in ("model_fingerprint", "placement_fingerprint"):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        float_fields = ("prefill_seconds", "decode_seconds")
        for field_name in float_fields:
            object.__setattr__(
                self,
                field_name,
                _nonnegative_float(getattr(self, field_name), field_name),
            )
        integer_fields = (
            "prefill_calls",
            "prefill_tokens",
            "decode_calls",
            "decode_tokens",
            "provisional_steps",
            "commits",
            "abandons",
            "committed_tokens",
            "physical_weight_bytes_read",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "model_resident_bytes",
            "kv_resident_bytes",
            "workspace_peak_bytes",
            "unexpected_fallbacks",
            "unexpected_page_loads",
        )
        for field_name in integer_fields:
            object.__setattr__(
                self,
                field_name,
                _nonnegative_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "extra_counters",
            _counter_pairs(self.extra_counters, "extra_counters"),
        )


@runtime_checkable
class ModelRuntime(Protocol):
    """Live model execution protocol; all backend tensors remain private."""

    @property
    def route(self) -> RuntimeRoute: ...

    def allocate_state(
        self,
        *,
        owner_id: str,
        batch_size: int,
        capacity: int,
    ) -> StateHandle: ...

    def prefill(self, work: PrefillWork) -> ProvisionalStep: ...

    def decode(self, work: DecodeWork) -> ProvisionalStep: ...

    def commit(
        self,
        step: ProvisionalStep,
        accepted_counts: Sequence[int],
    ) -> CommitResult: ...

    def abandon(self, step: ProvisionalStep) -> None: ...

    def release_state(self, state: StateHandle) -> None: ...

    def telemetry(self) -> RuntimeTelemetry: ...

    def close(self) -> None: ...


@runtime_checkable
class GreedyBlockVerifier(Protocol):
    """Optional transactional B1 capability for all-position raw-greedy verification.

    The verifier must leave the whole input block provisional.  ``commit_greedy_block`` accepts
    only a prefix of that input block (including an empty prefix), while abandon restores the
    exact parent observation.  Implementations must reject stale, foreign, consumed, or
    ordinary-step authorities rather than relying on structural protocol compatibility.
    """

    def verify_greedy_block(
        self,
        work: GreedyBlockVerifyWork,
    ) -> GreedyBlockVerification: ...

    def commit_greedy_block(
        self,
        verification: GreedyBlockVerification,
        accepted_input_count: int,
    ) -> CommitResult: ...

    def abandon_greedy_block(self, verification: GreedyBlockVerification) -> None: ...


@runtime_checkable
class GreedyProposalTransactionRuntime(Protocol):
    """Optional in-place draft cursor with exact rollback and accepted-prefix promotion."""

    def begin_greedy_proposal(
        self,
        work: GreedyProposalBeginWork,
    ) -> GreedyProposalTransaction: ...

    def advance_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        input_token_id: int,
    ) -> GreedyProposalTransaction: ...

    def seal_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        input_token_id: int,
    ) -> GreedyProposalTransaction: ...

    def commit_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        accepted_input_count: int,
    ) -> CommitResult: ...

    def abandon_greedy_proposal(self, transaction: GreedyProposalTransaction) -> None: ...


@runtime_checkable
class CompatibleBatchLane(Protocol):
    """Optional pooled-forward lane over independent runtime-owned B1 states.

    ``execute`` must return one independently consumable provisional step for every input work
    item in the same order.  A conforming implementation may partition ragged input into stricter
    physical waves, but it may not mutate committed state before the corresponding returned step
    is explicitly committed.
    """

    @property
    def identity(self) -> CompatibleBatchLaneIdentity: ...

    def execute(
        self,
        works: Sequence[PrefillWork | DecodeWork],
    ) -> tuple[ProvisionalStep, ...]: ...


@runtime_checkable
class ForkableModelRuntime(ModelRuntime, Protocol):
    """Optional native capability for row-preserving exact-prefix state copies."""

    def fork_state(
        self,
        source: StateHandle,
        *,
        parent: StateObservation,
        owner_id: str,
        capacity: int,
    ) -> StateForkResult: ...


@runtime_checkable
class ExecutionBackend(Protocol):
    """Factory/planner interface implemented by CPU, MLX/Metal, and CUDA backends."""

    def capabilities(self, device: DeviceDescriptor) -> BackendCapabilities: ...

    def plan(
        self,
        model: CompiledModelIdentity,
        workload: WorkloadSpec,
        device: DeviceDescriptor,
        *,
        memory_budget_bytes: int | None = None,
    ) -> PlacementPlan: ...

    def open(
        self,
        model: CompiledModelIdentity,
        placement: PlacementPlan,
    ) -> ModelRuntime: ...


__all__ = [
    "BACKEND_CAPABILITIES_SCHEMA",
    "COMPATIBLE_BATCH_LANE_SCHEMA",
    "COMPILED_MODEL_SCHEMA",
    "DEVICE_DESCRIPTOR_SCHEMA",
    "PLACEMENT_PLAN_SCHEMA",
    "RUNTIME_ROUTE_SCHEMA",
    "RUNTIME_TELEMETRY_SCHEMA",
    "STATE_OBSERVATION_SCHEMA",
    "WORKLOAD_SPEC_SCHEMA",
    "BackendCapabilities",
    "BlobIdentity",
    "CodecCapability",
    "CompatibleBatchLane",
    "CompatibleBatchLaneIdentity",
    "CommitResult",
    "CompiledComponent",
    "CompiledModelIdentity",
    "ComponentPlacement",
    "DecodeWork",
    "DeviceDescriptor",
    "ExecutionBackend",
    "FallbackPolicy",
    "ForkableModelRuntime",
    "HostBuffer",
    "MemoryDomain",
    "ModelRuntime",
    "NATIVE_SAMPLING_ABI",
    "NativeOutput",
    "OutputMode",
    "OutputRequest",
    "PlacementPlan",
    "PrefillWork",
    "PromotionStatus",
    "ProvisionalAuthority",
    "ProvisionalStep",
    "Residency",
    "RuntimeRoute",
    "RuntimeTelemetry",
    "SamplingPolicy",
    "SamplingRequest",
    "StateHandle",
    "StateForkResult",
    "StateObservation",
    "StatePlacement",
    "WorkloadSpec",
]
