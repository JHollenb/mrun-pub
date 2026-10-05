"""Immutable, content-addressed work-plan IR.

The IR records static execution decisions. Runtime tensors stay outside the plan so the
same serialized plan can be validated, lowered, cached, and replayed across requests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from numbers import Integral
from typing import Any

WORKPLAN_SCHEMA = "mrun-dense-workplan-v3"
LEGACY_WORKPLAN_SCHEMAS = frozenset({"mrun-dense-workplan-v1", "mrun-dense-workplan-v2"})
WORK_TEMPLATE_SCHEMA = "mrun-work-template-v2"
LEGACY_WORK_TEMPLATE_SCHEMAS = frozenset({"mrun-work-template-v1"})
DISPATCH_BINDING_SCHEMA = "mrun-dispatch-binding-v1"
CAPTURE_RESOURCE_ESTIMATE_SCHEMA = "mrun-capture-resource-estimate-v1"
TYPED_CAPTURE_SPEC_SCHEMA = "mrun-typed-capture-spec-v1"
CAPTURE_CAPABILITY_SCHEMA = "mrun-capture-capability-v1"

_ACTIVATION_DTYPES = {"fp16", "bf16", "fp32", "int8", "uint8"}
_WEIGHT_DTYPES = {"fp16", "bf16", "fp32", "int8", "int4", "int3", "int2", "fp8"}
_ACCUMULATOR_DTYPES = {"fp16", "bf16", "fp32", "int32", "fp32-class"}
_KV_DTYPES = {"fp16", "bf16", "fp32", "fp8", "int8", "uint8", "int4", "int3", "int2"}

# WorkPlan metadata is intentionally open-ended because callers may attach request
# provenance, tracing labels, or experiment annotations.  A WorkTemplate is a cache key and
# executable contract, so it must be closed over only the fields the compiler/runtime
# currently interpret structurally.  New structural metadata must be added here deliberately;
# everything else travels in DispatchBinding and cannot fragment or poison the template cache.
_REUSABLE_WORK_TEMPLATE_METADATA_KEYS = frozenset(
    {
        "blob_identity_verified",
        "blob_records_sha256",
        "builder_source_bundle_sha256",
        "component_cache_budgets_json",
        "component_graph_fingerprint",
        "component_output_contract",
        "composite_qstore",
        "configured_output_row_count",
        "engine_backend",
        "engine_device",
        "head_output_pushdown",
        "identity_certificate_sha256",
        "input_token_limit",
        "kv_capacity",
        "kv_dtype",
        "kv_head_dim",
        "kv_num_heads",
        "kv_num_layers",
        "logical_head_access",
        "logical_head_row_count",
        "logical_hidden_width",
        "manifest_semantic_sha256",
        "max_seq_len",
        "op_graph_status",
        "output_pushdown",
        "ring_staging_bytes",
        "source_identity_status",
        "store_identity_status",
        "vocab_manifest_sha256",
        "weight_cache_budget_bytes",
        "weight_cache_policy",
        "work_floor_status",
    }
)

JsonScalar = str | int | float | bool | None


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"serialized JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"serialized JSON contains non-finite number {value!r}")


def _strict_json_loads(payload: str | bytes | bytearray, *, field: str) -> dict[str, Any]:
    decoded = json.loads(
        payload,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(decoded, dict):
        raise TypeError(f"serialized {field} must be a JSON object")
    return decoded


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    field: str,
) -> None:
    if any(type(key) is not str for key in payload):
        raise TypeError(f"{field} field names must be strings")
    actual = set(payload)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {sorted(unknown)!r}")
    if missing:
        raise ValueError(f"{field} is missing required fields: {sorted(missing)!r}")


def _strict_string(value: Any, *, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    return value


def _strict_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a boolean")
    return value


def _strict_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    return value


def _strict_list(value: Any, *, field: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{field} must be an array")
    return value


def _strict_string_list(value: Any, *, field: str) -> tuple[str, ...]:
    values = _strict_list(value, field=field)
    return tuple(
        _strict_string(item, field=f"{field}[{index}]") for index, item in enumerate(values)
    )


def _strict_string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence of strings")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be a sequence of strings") from exc
    if any(type(item) is not str for item in values):
        raise TypeError(f"{field} must contain only strings")
    return values


def _strict_int_tuple(value: Any, *, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence of integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be a sequence of integers") from exc
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in values):
        raise TypeError(f"{field} must contain only integers")
    return tuple(int(item) for item in values)


def _strict_nested_int_tuple(value: Any, *, field: str) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence of integer sequences")
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be a sequence of integer sequences") from exc
    return tuple(
        _strict_int_tuple(row, field=f"{field}[{index}]") for index, row in enumerate(rows)
    )


def _is_sha256_digest(value: str) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class ExecutionMode(str, Enum):
    SCORE = "score"
    PREFILL = "prefill"
    DECODE = "decode"


class OutputContract(str, Enum):
    FULL_LOGITS = "full_logits"
    LAST_TOKEN_LOGITS = "last_token_logits"
    SELECTED_TOKEN_ROWS = "selected_token_rows"
    CANDIDATE_ARGMAX_AND_MARGIN = "candidate_argmax_and_margin"
    LOSS_ONLY = "loss_only"
    HIDDEN_STATE_ONLY = "hidden_state_only"
    SELECTED_CAPTURE = "selected_capture"


class CaptureStateOwner(str, Enum):
    RESIDUAL = "residual"
    ATTENTION = "attention"
    MLP = "mlp"
    KV = "kv"
    RECURRENT = "recurrent"
    MODULATION = "modulation"
    ROUTER = "router"
    SAMPLER = "sampler"
    SCHEDULER = "scheduler"


class CaptureKind(str, Enum):
    RAW_SELECTED_ROW = "raw_selected_row"
    MOMENTS = "moments"
    TOP_K = "top_k"
    PROJECTION = "projection"
    COUNT_SKETCH = "count_sketch"
    JVP = "jvp"
    VJP = "vjp"
    PATCH_EFFECT = "patch_effect"


class CaptureRetention(str, Enum):
    """Compact artifact forms; intentionally no full-tensor retention mode exists."""

    METADATA_ONLY = "metadata_only"
    METADATA_AND_DIFFS = "metadata_and_diffs"
    COMPACT_REDUCTION = "compact_reduction"


class InterventionAuthority(str, Enum):
    OBSERVE_ONLY = "observe_only"
    DELETE = "delete"
    REPLACE = "replace"
    ADD = "add"
    TRANSACTIONAL_STATE_WRITE = "transactional_state_write"


def _coerce_enum(value: str | Enum, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}") from exc


def _metadata_items(
    metadata: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
) -> tuple[tuple[str, JsonScalar], ...]:
    if metadata is None:
        return ()
    items = metadata.items() if isinstance(metadata, Mapping) else metadata
    normalized: list[tuple[str, JsonScalar]] = []
    for raw_key, value in items:
        if type(raw_key) is not str:
            raise TypeError("metadata keys must be strings")
        key = raw_key
        if not key:
            raise ValueError("metadata keys must be non-empty")
        if type(value) not in {str, int, float, bool, type(None)}:
            raise TypeError(f"metadata value for {key!r} must be a JSON scalar")
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"metadata value for {key!r} must be finite")
        normalized.append((key, value))
    keys = [key for key, _ in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("metadata keys must be unique")
    return tuple(sorted(normalized))


def _partition_workplan_metadata(
    metadata: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
) -> tuple[tuple[tuple[str, JsonScalar], ...], tuple[tuple[str, JsonScalar], ...]]:
    items = _metadata_items(metadata)
    reusable = tuple(
        (key, value) for key, value in items if key in _REUSABLE_WORK_TEMPLATE_METADATA_KEYS
    )
    dispatch = tuple(
        (key, value) for key, value in items if key not in _REUSABLE_WORK_TEMPLATE_METADATA_KEYS
    )
    return reusable, dispatch


def _work_template_metadata_items(
    metadata: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
) -> tuple[tuple[str, JsonScalar], ...]:
    items = _metadata_items(metadata)
    unknown = sorted(key for key, _ in items if key not in _REUSABLE_WORK_TEMPLATE_METADATA_KEYS)
    if unknown:
        raise ValueError(f"template metadata contains non-reusable keys: {unknown!r}")
    return items


def _dispatch_metadata_items(
    metadata: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
) -> tuple[tuple[str, JsonScalar], ...]:
    items = _metadata_items(metadata)
    structural = sorted(key for key, _ in items if key in _REUSABLE_WORK_TEMPLATE_METADATA_KEYS)
    if structural:
        raise ValueError(f"dispatch metadata contains structural template keys: {structural!r}")
    return items


def _metadata_nonnegative_int(metadata: Mapping[str, JsonScalar], key: str) -> int | None:
    if key not in metadata:
        return None
    value = metadata[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} metadata must be a non-negative integer")
    return value


def _validate_redundant_head_metadata(
    *,
    output_contract: OutputContract,
    metadata: Mapping[str, JsonScalar],
    required_output_row_count: int,
    candidate_union_count: int,
) -> None:
    """Cross-check cached head-shape claims against the canonical dispatch shape."""

    logical_count = _metadata_nonnegative_int(metadata, "logical_head_row_count")
    configured_count = _metadata_nonnegative_int(metadata, "configured_output_row_count")
    expected_access: str
    expected_count: int | None
    if output_contract in {
        OutputContract.HIDDEN_STATE_ONLY,
        OutputContract.SELECTED_CAPTURE,
    }:
        expected_access = "none"
        expected_count = 0
    elif output_contract is OutputContract.SELECTED_TOKEN_ROWS:
        expected_access = "rows"
        expected_count = required_output_row_count
    elif output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
        expected_access = "rows"
        expected_count = candidate_union_count
    else:
        expected_access = "all"
        expected_count = configured_count

    compact_count = (
        required_output_row_count
        if output_contract is OutputContract.SELECTED_TOKEN_ROWS
        else candidate_union_count
        if output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
        else None
    )
    if (
        configured_count is not None
        and compact_count is not None
        and compact_count > configured_count
    ):
        raise ValueError(
            "output-contract cardinality exceeds configured_output_row_count metadata "
            f"({compact_count} > {configured_count})"
        )

    if logical_count is not None and expected_count is not None and logical_count != expected_count:
        raise ValueError(
            "logical_head_row_count metadata does not match the output-contract cardinality "
            f"({logical_count} != {expected_count})"
        )
    if "logical_head_access" in metadata:
        access = metadata["logical_head_access"]
        if type(access) is not str:
            raise ValueError("logical_head_access metadata must be a string")
        if access != expected_access:
            raise ValueError(
                "logical_head_access metadata does not match the output contract "
                f"({access!r} != {expected_access!r})"
            )


@dataclass(frozen=True)
class PrecisionPolicy:
    activation_dtype: str
    weight_dtype: str
    accumulator_dtype: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str
            for value in (self.activation_dtype, self.weight_dtype, self.accumulator_dtype)
        ):
            raise TypeError("precision dtypes must be strings")
        if self.activation_dtype not in _ACTIVATION_DTYPES:
            raise ValueError(f"unsupported activation dtype: {self.activation_dtype}")
        if self.weight_dtype not in _WEIGHT_DTYPES:
            raise ValueError(f"unsupported weight dtype: {self.weight_dtype}")
        if self.accumulator_dtype not in _ACCUMULATOR_DTYPES:
            raise ValueError(f"unsupported accumulator dtype: {self.accumulator_dtype}")

    def as_dict(self) -> dict[str, str]:
        return {
            "activation_dtype": self.activation_dtype,
            "weight_dtype": self.weight_dtype,
            "accumulator_dtype": self.accumulator_dtype,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrecisionPolicy:
        return cls(
            activation_dtype=_strict_string(
                payload["activation_dtype"], field="PrecisionPolicy.activation_dtype"
            ),
            weight_dtype=_strict_string(
                payload["weight_dtype"], field="PrecisionPolicy.weight_dtype"
            ),
            accumulator_dtype=_strict_string(
                payload["accumulator_dtype"], field="PrecisionPolicy.accumulator_dtype"
            ),
        )


@dataclass(frozen=True)
class ShapeBucket:
    actual_batch: int
    batch_bucket: int
    sequence_length: int
    sequence_bucket: int

    def __post_init__(self) -> None:
        for field_name in (
            "actual_batch",
            "batch_bucket",
            "sequence_length",
            "sequence_bucket",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
            object.__setattr__(self, field_name, int(value))
        if self.actual_batch <= 0 or self.sequence_length <= 0:
            raise ValueError("actual batch and sequence length must be positive")
        if self.batch_bucket < self.actual_batch:
            raise ValueError("batch bucket cannot be smaller than the actual batch")
        if self.sequence_bucket < self.sequence_length:
            raise ValueError("sequence bucket cannot be smaller than the actual sequence")

    @property
    def live_token_rows(self) -> int:
        return self.actual_batch * self.sequence_length

    def as_dict(self) -> dict[str, int]:
        return {
            "actual_batch": self.actual_batch,
            "batch_bucket": self.batch_bucket,
            "sequence_length": self.sequence_length,
            "sequence_bucket": self.sequence_bucket,
            "live_token_rows": self.live_token_rows,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ShapeBucket:
        return cls(
            actual_batch=payload["actual_batch"],
            batch_bucket=payload["batch_bucket"],
            sequence_length=payload["sequence_length"],
            sequence_bucket=payload["sequence_bucket"],
        )


@dataclass(frozen=True)
class CaptureContract:
    requested: bool = False
    static_shapes: bool = True
    stable_addresses: bool = False
    graph_safe: bool = False

    def __post_init__(self) -> None:
        for field_name in ("requested", "static_shapes", "stable_addresses", "graph_safe"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"capture {field_name} must be boolean")

    @property
    def eligible(self) -> bool:
        return self.static_shapes and self.stable_addresses and self.graph_safe

    @property
    def refusal_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.static_shapes:
            reasons.append("shapes are dynamic")
        if not self.stable_addresses:
            reasons.append("device addresses are not proven stable")
        if not self.graph_safe:
            reasons.append("one or more operations are not proven graph-safe")
        return tuple(reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "static_shapes": self.static_shapes,
            "stable_addresses": self.stable_addresses,
            "graph_safe": self.graph_safe,
            "eligible": self.eligible,
            "refusal_reasons": list(self.refusal_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaptureContract:
        return cls(
            requested=payload.get("requested", False),
            static_shapes=payload.get("static_shapes", True),
            stable_addresses=payload.get("stable_addresses", False),
            graph_safe=payload.get("graph_safe", False),
        )


def _canonical_string(value: Any, *, field: str) -> str:
    result = _strict_string(value, field=field)
    if not result or result.strip() != result:
        raise ValueError(f"{field} must be a canonical non-empty string")
    return result


def _canonical_string_set(
    values: Sequence[str],
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(
        _canonical_string(value, field=f"{field}[{index}]")
        for index, value in enumerate(_strict_string_tuple(values, field=field))
    )
    if not allow_empty and not normalized:
        raise ValueError(f"{field} must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must contain unique values")
    return tuple(sorted(normalized))


def _canonical_index_set(values: Sequence[int], *, field: str) -> tuple[int, ...]:
    normalized = _strict_int_tuple(values, field=field)
    if any(value < 0 for value in normalized):
        raise ValueError(f"{field} must contain only non-negative indices")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must contain unique indices")
    return tuple(sorted(normalized))


def _optional_nonnegative_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    result = _strict_int(value, field=field)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _enum_set(
    values: Sequence[str | Enum],
    enum_type: type[Enum],
    *,
    field: str,
) -> tuple[Enum, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field} must be a sequence") from exc
    if not raw_values:
        raise ValueError(f"{field} must be non-empty")
    normalized = tuple(_coerce_enum(value, enum_type, field) for value in raw_values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must contain unique values")
    return tuple(sorted(normalized, key=lambda value: value.value))


@dataclass(frozen=True)
class CaptureResourceEstimate:
    """Static upper-bound inputs used before a model or capture tape is allocated."""

    model_loads: int
    prefix_calls: int
    suffix_calls: int
    backward_calls: int
    device_bytes: int
    retained_artifact_bytes: int
    expected_liveness_steps: int
    schema_version: str = CAPTURE_RESOURCE_ESTIMATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAPTURE_RESOURCE_ESTIMATE_SCHEMA:
            raise ValueError(f"unsupported capture resource schema: {self.schema_version}")
        for field_name in (
            "model_loads",
            "prefix_calls",
            "suffix_calls",
            "backward_calls",
            "device_bytes",
            "retained_artifact_bytes",
            "expected_liveness_steps",
        ):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"capture estimate {field_name} must be an integer")
            if value < 0:
                raise ValueError(f"capture estimate {field_name} must be non-negative")
        if self.device_bytes <= 0:
            raise ValueError("capture estimate device_bytes must be positive")
        if self.retained_artifact_bytes <= 0:
            raise ValueError("capture estimate retained_artifact_bytes must be positive")
        if self.expected_liveness_steps <= 0:
            raise ValueError("capture estimate expected_liveness_steps must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_loads": self.model_loads,
            "prefix_calls": self.prefix_calls,
            "suffix_calls": self.suffix_calls,
            "backward_calls": self.backward_calls,
            "device_bytes": self.device_bytes,
            "retained_artifact_bytes": self.retained_artifact_bytes,
            "expected_liveness_steps": self.expected_liveness_steps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaptureResourceEstimate:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "model_loads",
                "prefix_calls",
                "suffix_calls",
                "backward_calls",
                "device_bytes",
                "retained_artifact_bytes",
                "expected_liveness_steps",
            },
            field="CaptureResourceEstimate",
        )
        return cls(
            model_loads=_strict_int(
                payload["model_loads"], field="CaptureResourceEstimate.model_loads"
            ),
            prefix_calls=_strict_int(
                payload["prefix_calls"], field="CaptureResourceEstimate.prefix_calls"
            ),
            suffix_calls=_strict_int(
                payload["suffix_calls"], field="CaptureResourceEstimate.suffix_calls"
            ),
            backward_calls=_strict_int(
                payload["backward_calls"], field="CaptureResourceEstimate.backward_calls"
            ),
            device_bytes=_strict_int(
                payload["device_bytes"], field="CaptureResourceEstimate.device_bytes"
            ),
            retained_artifact_bytes=_strict_int(
                payload["retained_artifact_bytes"],
                field="CaptureResourceEstimate.retained_artifact_bytes",
            ),
            expected_liveness_steps=_strict_int(
                payload["expected_liveness_steps"],
                field="CaptureResourceEstimate.expected_liveness_steps",
            ),
            schema_version=_strict_string(
                payload["schema_version"], field="CaptureResourceEstimate.schema_version"
            ),
        )


@dataclass(frozen=True)
class TypedCaptureSpec:
    """One fully addressed, reduction-first observation or intervention sink.

    The spec contains only immutable control metadata. Tensor values, projections, and patch
    operands are separately content-addressed runtime inputs and are deliberately inexpressible
    in this schema.
    """

    capture_id: str
    semantic_role: str
    physical_port: str
    state_owner: CaptureStateOwner
    capture_kind: CaptureKind
    dtype: str
    accumulator_dtype: str
    numerical_contract: str
    intended_consumer: str
    terminal_assay: str
    on_device_reduction: bool
    retention: CaptureRetention
    max_retained_bytes: int
    specimen_row_ids: tuple[str, ...]
    parent_statecut_id: str
    branch_id: str
    source_hashes: tuple[str, ...]
    intervention_authority: InterventionAuthority
    required_engine_capabilities: tuple[str, ...]
    resource_estimate: CaptureResourceEstimate
    layer_index: int | None = None
    block_index: int | None = None
    head_indices: tuple[int, ...] = ()
    expert_indices: tuple[int, ...] = ()
    token_indices: tuple[int, ...] = ()
    token_span: tuple[int, ...] = ()
    cursor_index: int | None = None
    schema_version: str = TYPED_CAPTURE_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TYPED_CAPTURE_SPEC_SCHEMA:
            raise ValueError(f"unsupported typed capture schema: {self.schema_version}")
        for field_name in (
            "capture_id",
            "semantic_role",
            "physical_port",
            "numerical_contract",
            "intended_consumer",
            "terminal_assay",
            "branch_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_string(getattr(self, field_name), field=f"capture {field_name}"),
            )
        object.__setattr__(
            self,
            "state_owner",
            _coerce_enum(self.state_owner, CaptureStateOwner, "capture state_owner"),
        )
        object.__setattr__(
            self,
            "capture_kind",
            _coerce_enum(self.capture_kind, CaptureKind, "capture kind"),
        )
        object.__setattr__(
            self,
            "retention",
            _coerce_enum(self.retention, CaptureRetention, "capture retention"),
        )
        object.__setattr__(
            self,
            "intervention_authority",
            _coerce_enum(
                self.intervention_authority,
                InterventionAuthority,
                "capture intervention_authority",
            ),
        )
        if type(self.on_device_reduction) is not bool:
            raise TypeError("capture on_device_reduction must be boolean")
        if not self.on_device_reduction:
            raise ValueError("typed captures must reduce or select on device before retention")
        if self.dtype not in _ACTIVATION_DTYPES:
            raise ValueError(f"unsupported capture dtype: {self.dtype}")
        if self.accumulator_dtype not in _ACCUMULATOR_DTYPES:
            raise ValueError(f"unsupported capture accumulator dtype: {self.accumulator_dtype}")
        if type(self.max_retained_bytes) is not int or self.max_retained_bytes <= 0:
            raise ValueError("capture max_retained_bytes must be a positive integer")
        if not isinstance(self.resource_estimate, CaptureResourceEstimate):
            raise TypeError("capture resource_estimate must be a CaptureResourceEstimate")
        if self.resource_estimate.retained_artifact_bytes > self.max_retained_bytes:
            raise ValueError("capture retained-artifact estimate exceeds its byte cap")
        object.__setattr__(
            self,
            "layer_index",
            _optional_nonnegative_int(self.layer_index, field="capture layer_index"),
        )
        object.__setattr__(
            self,
            "block_index",
            _optional_nonnegative_int(self.block_index, field="capture block_index"),
        )
        object.__setattr__(
            self,
            "cursor_index",
            _optional_nonnegative_int(self.cursor_index, field="capture cursor_index"),
        )
        for field_name in ("head_indices", "expert_indices", "token_indices"):
            object.__setattr__(
                self,
                field_name,
                _canonical_index_set(getattr(self, field_name), field=f"capture {field_name}"),
            )
        token_span = _strict_int_tuple(self.token_span, field="capture token_span")
        if token_span:
            if len(token_span) != 2 or token_span[0] < 0 or token_span[1] <= token_span[0]:
                raise ValueError("capture token_span must be an increasing [start, end) pair")
        object.__setattr__(self, "token_span", token_span)
        if not self.token_indices and not self.token_span and self.cursor_index is None:
            raise ValueError("capture requires exact token indices, span, or cursor")
        object.__setattr__(
            self,
            "specimen_row_ids",
            tuple(
                _canonical_string(value, field=f"capture specimen_row_ids[{index}]")
                for index, value in enumerate(
                    _strict_string_tuple(self.specimen_row_ids, field="capture specimen_row_ids")
                )
            ),
        )
        if not self.specimen_row_ids or len(self.specimen_row_ids) != len(
            set(self.specimen_row_ids)
        ):
            raise ValueError("capture specimen_row_ids must be non-empty and unique")
        if not _is_sha256_digest(self.parent_statecut_id):
            raise ValueError("capture parent_statecut_id must be a SHA-256 identity")
        source_hashes = _canonical_string_set(
            self.source_hashes,
            field="capture source_hashes",
        )
        if any(not _is_sha256_digest(value) for value in source_hashes):
            raise ValueError("capture source_hashes must contain only SHA-256 identities")
        object.__setattr__(self, "source_hashes", source_hashes)
        object.__setattr__(
            self,
            "required_engine_capabilities",
            _canonical_string_set(
                self.required_engine_capabilities,
                field="capture required_engine_capabilities",
            ),
        )

    @cached_property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capture_id": self.capture_id,
            "semantic_role": self.semantic_role,
            "physical_port": self.physical_port,
            "state_owner": self.state_owner.value,
            "capture_kind": self.capture_kind.value,
            "dtype": self.dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "numerical_contract": self.numerical_contract,
            "intended_consumer": self.intended_consumer,
            "terminal_assay": self.terminal_assay,
            "on_device_reduction": self.on_device_reduction,
            "retention": self.retention.value,
            "max_retained_bytes": self.max_retained_bytes,
            "specimen_row_ids": list(self.specimen_row_ids),
            "parent_statecut_id": self.parent_statecut_id,
            "branch_id": self.branch_id,
            "source_hashes": list(self.source_hashes),
            "intervention_authority": self.intervention_authority.value,
            "required_engine_capabilities": list(self.required_engine_capabilities),
            "resource_estimate": self.resource_estimate.as_dict(),
            "layer_index": self.layer_index,
            "block_index": self.block_index,
            "head_indices": list(self.head_indices),
            "expert_indices": list(self.expert_indices),
            "token_indices": list(self.token_indices),
            "token_span": list(self.token_span),
            "cursor_index": self.cursor_index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TypedCaptureSpec:
        expected = {
            "schema_version",
            "capture_id",
            "semantic_role",
            "physical_port",
            "state_owner",
            "capture_kind",
            "dtype",
            "accumulator_dtype",
            "numerical_contract",
            "intended_consumer",
            "terminal_assay",
            "on_device_reduction",
            "retention",
            "max_retained_bytes",
            "specimen_row_ids",
            "parent_statecut_id",
            "branch_id",
            "source_hashes",
            "intervention_authority",
            "required_engine_capabilities",
            "resource_estimate",
            "layer_index",
            "block_index",
            "head_indices",
            "expert_indices",
            "token_indices",
            "token_span",
            "cursor_index",
        }
        _require_exact_keys(payload, expected, field="TypedCaptureSpec")
        resource_estimate = payload["resource_estimate"]
        if type(resource_estimate) is not dict:
            raise TypeError("TypedCaptureSpec.resource_estimate must be an object")
        return cls(
            capture_id=_strict_string(payload["capture_id"], field="TypedCaptureSpec.capture_id"),
            semantic_role=_strict_string(
                payload["semantic_role"], field="TypedCaptureSpec.semantic_role"
            ),
            physical_port=_strict_string(
                payload["physical_port"], field="TypedCaptureSpec.physical_port"
            ),
            state_owner=_strict_string(
                payload["state_owner"], field="TypedCaptureSpec.state_owner"
            ),  # type: ignore[arg-type]
            capture_kind=_strict_string(
                payload["capture_kind"], field="TypedCaptureSpec.capture_kind"
            ),  # type: ignore[arg-type]
            dtype=_strict_string(payload["dtype"], field="TypedCaptureSpec.dtype"),
            accumulator_dtype=_strict_string(
                payload["accumulator_dtype"], field="TypedCaptureSpec.accumulator_dtype"
            ),
            numerical_contract=_strict_string(
                payload["numerical_contract"], field="TypedCaptureSpec.numerical_contract"
            ),
            intended_consumer=_strict_string(
                payload["intended_consumer"], field="TypedCaptureSpec.intended_consumer"
            ),
            terminal_assay=_strict_string(
                payload["terminal_assay"], field="TypedCaptureSpec.terminal_assay"
            ),
            on_device_reduction=_strict_bool(
                payload["on_device_reduction"], field="TypedCaptureSpec.on_device_reduction"
            ),
            retention=_strict_string(payload["retention"], field="TypedCaptureSpec.retention"),  # type: ignore[arg-type]
            max_retained_bytes=_strict_int(
                payload["max_retained_bytes"], field="TypedCaptureSpec.max_retained_bytes"
            ),
            specimen_row_ids=_strict_string_list(
                payload["specimen_row_ids"], field="TypedCaptureSpec.specimen_row_ids"
            ),
            parent_statecut_id=_strict_string(
                payload["parent_statecut_id"], field="TypedCaptureSpec.parent_statecut_id"
            ),
            branch_id=_strict_string(payload["branch_id"], field="TypedCaptureSpec.branch_id"),
            source_hashes=_strict_string_list(
                payload["source_hashes"], field="TypedCaptureSpec.source_hashes"
            ),
            intervention_authority=_strict_string(
                payload["intervention_authority"],
                field="TypedCaptureSpec.intervention_authority",
            ),  # type: ignore[arg-type]
            required_engine_capabilities=_strict_string_list(
                payload["required_engine_capabilities"],
                field="TypedCaptureSpec.required_engine_capabilities",
            ),
            resource_estimate=CaptureResourceEstimate.from_dict(resource_estimate),
            layer_index=_optional_nonnegative_int(
                payload["layer_index"], field="TypedCaptureSpec.layer_index"
            ),
            block_index=_optional_nonnegative_int(
                payload["block_index"], field="TypedCaptureSpec.block_index"
            ),
            head_indices=_strict_int_tuple(
                payload["head_indices"], field="TypedCaptureSpec.head_indices"
            ),
            expert_indices=_strict_int_tuple(
                payload["expert_indices"], field="TypedCaptureSpec.expert_indices"
            ),
            token_indices=_strict_int_tuple(
                payload["token_indices"], field="TypedCaptureSpec.token_indices"
            ),
            token_span=_strict_int_tuple(
                payload["token_span"], field="TypedCaptureSpec.token_span"
            ),
            cursor_index=_optional_nonnegative_int(
                payload["cursor_index"], field="TypedCaptureSpec.cursor_index"
            ),
            schema_version=_strict_string(
                payload["schema_version"], field="TypedCaptureSpec.schema_version"
            ),
        )


@dataclass(frozen=True)
class CaptureCapability:
    """Exact backend capability used to fail closed before capture lowering."""

    backend: str
    runtime_entrypoint: str
    semantic_roles: tuple[str, ...]
    physical_ports: tuple[str, ...]
    state_owners: tuple[CaptureStateOwner, ...]
    capture_kinds: tuple[CaptureKind, ...]
    dtypes: tuple[str, ...]
    accumulator_dtypes: tuple[str, ...]
    numerical_contracts: tuple[str, ...]
    intervention_authorities: tuple[InterventionAuthority, ...]
    engine_capabilities: tuple[str, ...]
    supports_on_device_reduction: bool
    supports_backward: bool
    max_specs: int
    max_retained_bytes: int
    schema_version: str = CAPTURE_CAPABILITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAPTURE_CAPABILITY_SCHEMA:
            raise ValueError(f"unsupported capture capability schema: {self.schema_version}")
        object.__setattr__(
            self, "backend", _canonical_string(self.backend, field="capture backend")
        )
        object.__setattr__(
            self,
            "runtime_entrypoint",
            _canonical_string(self.runtime_entrypoint, field="capture runtime_entrypoint"),
        )
        for field_name in (
            "semantic_roles",
            "physical_ports",
            "dtypes",
            "accumulator_dtypes",
            "numerical_contracts",
            "engine_capabilities",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_string_set(getattr(self, field_name), field=f"capture {field_name}"),
            )
        object.__setattr__(
            self,
            "state_owners",
            _enum_set(
                self.state_owners,
                CaptureStateOwner,
                field="capture capability state_owners",
            ),
        )
        object.__setattr__(
            self,
            "capture_kinds",
            _enum_set(
                self.capture_kinds,
                CaptureKind,
                field="capture capability capture_kinds",
            ),
        )
        object.__setattr__(
            self,
            "intervention_authorities",
            _enum_set(
                self.intervention_authorities,
                InterventionAuthority,
                field="capture capability intervention_authorities",
            ),
        )
        for field_name in ("supports_on_device_reduction", "supports_backward"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"capture capability {field_name} must be boolean")
        for field_name in ("max_specs", "max_retained_bytes"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"capture capability {field_name} must be a positive integer")
        if any(value not in _ACTIVATION_DTYPES for value in self.dtypes):
            raise ValueError("capture capability contains an unsupported dtype")
        if any(value not in _ACCUMULATOR_DTYPES for value in self.accumulator_dtypes):
            raise ValueError("capture capability contains an unsupported accumulator dtype")

    def validate_specs(self, specs: Sequence[TypedCaptureSpec]) -> None:
        normalized = tuple(specs)
        if not normalized:
            raise ValueError("selected_capture requires at least one typed capture spec")
        if len(normalized) > self.max_specs:
            raise ValueError("typed capture count exceeds the backend capability")
        retained_bytes = sum(spec.max_retained_bytes for spec in normalized)
        if retained_bytes <= 0 or retained_bytes > self.max_retained_bytes:
            raise ValueError("aggregate capture retained-byte caps exceed the backend capability")
        engine_capabilities = set(self.engine_capabilities)
        for spec in normalized:
            if not isinstance(spec, TypedCaptureSpec):
                raise TypeError("capture_specs must contain only TypedCaptureSpec values")
            if spec.semantic_role not in self.semantic_roles:
                raise ValueError(f"capture {spec.capture_id!r} semantic role is not supported")
            if spec.physical_port not in self.physical_ports:
                raise ValueError(f"capture {spec.capture_id!r} physical port is not supported")
            if spec.state_owner not in self.state_owners:
                raise ValueError(f"capture {spec.capture_id!r} state owner is not supported")
            if spec.capture_kind not in self.capture_kinds:
                raise ValueError(f"capture {spec.capture_id!r} kind is not supported")
            if (
                spec.dtype not in self.dtypes
                or spec.accumulator_dtype not in self.accumulator_dtypes
            ):
                raise ValueError(f"capture {spec.capture_id!r} precision is not supported")
            if spec.numerical_contract not in self.numerical_contracts:
                raise ValueError(f"capture {spec.capture_id!r} numerical contract is not supported")
            if spec.intervention_authority not in self.intervention_authorities:
                raise ValueError(
                    f"capture {spec.capture_id!r} intervention authority is not supported"
                )
            if spec.on_device_reduction and not self.supports_on_device_reduction:
                raise ValueError("backend capability cannot perform required on-device reduction")
            if (
                spec.capture_kind in {CaptureKind.JVP, CaptureKind.VJP}
                and not self.supports_backward
            ):
                raise ValueError("JVP/VJP capture requires a backward-capable backend")
            missing = set(spec.required_engine_capabilities) - engine_capabilities
            if missing:
                raise ValueError(
                    f"capture {spec.capture_id!r} requires unavailable engine capabilities: "
                    f"{sorted(missing)!r}"
                )

    @cached_property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "runtime_entrypoint": self.runtime_entrypoint,
            "semantic_roles": list(self.semantic_roles),
            "physical_ports": list(self.physical_ports),
            "state_owners": [value.value for value in self.state_owners],
            "capture_kinds": [value.value for value in self.capture_kinds],
            "dtypes": list(self.dtypes),
            "accumulator_dtypes": list(self.accumulator_dtypes),
            "numerical_contracts": list(self.numerical_contracts),
            "intervention_authorities": [value.value for value in self.intervention_authorities],
            "engine_capabilities": list(self.engine_capabilities),
            "supports_on_device_reduction": self.supports_on_device_reduction,
            "supports_backward": self.supports_backward,
            "max_specs": self.max_specs,
            "max_retained_bytes": self.max_retained_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaptureCapability:
        expected = {
            "schema_version",
            "backend",
            "runtime_entrypoint",
            "semantic_roles",
            "physical_ports",
            "state_owners",
            "capture_kinds",
            "dtypes",
            "accumulator_dtypes",
            "numerical_contracts",
            "intervention_authorities",
            "engine_capabilities",
            "supports_on_device_reduction",
            "supports_backward",
            "max_specs",
            "max_retained_bytes",
        }
        _require_exact_keys(payload, expected, field="CaptureCapability")
        return cls(
            backend=_strict_string(payload["backend"], field="CaptureCapability.backend"),
            runtime_entrypoint=_strict_string(
                payload["runtime_entrypoint"], field="CaptureCapability.runtime_entrypoint"
            ),
            semantic_roles=_strict_string_list(
                payload["semantic_roles"], field="CaptureCapability.semantic_roles"
            ),
            physical_ports=_strict_string_list(
                payload["physical_ports"], field="CaptureCapability.physical_ports"
            ),
            state_owners=_strict_string_list(
                payload["state_owners"], field="CaptureCapability.state_owners"
            ),  # type: ignore[arg-type]
            capture_kinds=_strict_string_list(
                payload["capture_kinds"], field="CaptureCapability.capture_kinds"
            ),  # type: ignore[arg-type]
            dtypes=_strict_string_list(payload["dtypes"], field="CaptureCapability.dtypes"),
            accumulator_dtypes=_strict_string_list(
                payload["accumulator_dtypes"], field="CaptureCapability.accumulator_dtypes"
            ),
            numerical_contracts=_strict_string_list(
                payload["numerical_contracts"], field="CaptureCapability.numerical_contracts"
            ),
            intervention_authorities=_strict_string_list(
                payload["intervention_authorities"],
                field="CaptureCapability.intervention_authorities",
            ),  # type: ignore[arg-type]
            engine_capabilities=_strict_string_list(
                payload["engine_capabilities"], field="CaptureCapability.engine_capabilities"
            ),
            supports_on_device_reduction=_strict_bool(
                payload["supports_on_device_reduction"],
                field="CaptureCapability.supports_on_device_reduction",
            ),
            supports_backward=_strict_bool(
                payload["supports_backward"], field="CaptureCapability.supports_backward"
            ),
            max_specs=_strict_int(payload["max_specs"], field="CaptureCapability.max_specs"),
            max_retained_bytes=_strict_int(
                payload["max_retained_bytes"], field="CaptureCapability.max_retained_bytes"
            ),
            schema_version=_strict_string(
                payload["schema_version"], field="CaptureCapability.schema_version"
            ),
        )


@dataclass(frozen=True)
class DenseWorkPlan:
    model_name: str
    model_revision: str
    store_fingerprint: str
    execution_mode: ExecutionMode
    precision: PrecisionPolicy
    shape: ShapeBucket
    output_contract: OutputContract
    numerical_contract: str
    request_ids: tuple[str, ...]
    request_slots: tuple[int, ...]
    prefix_state_ids: tuple[str, ...] = ()
    kv_read_handles: tuple[str, ...] = ()
    kv_write_handles: tuple[str, ...] = ()
    required_output_rows: tuple[int, ...] = ()
    candidate_token_ids: tuple[tuple[int, ...], ...] = ()
    page_sequence: tuple[str, ...] = ()
    cache_admission: str = "default"
    compute_layout_ids: tuple[str, ...] = ()
    structured_operator_ids: tuple[str, ...] = ()
    capture: CaptureContract = field(default_factory=CaptureContract)
    capture_specs: tuple[TypedCaptureSpec, ...] = ()
    capture_capability: CaptureCapability | None = None
    metadata: tuple[tuple[str, JsonScalar], ...] = ()
    schema_version: str = WORKPLAN_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.precision, PrecisionPolicy):
            raise TypeError("precision must be a PrecisionPolicy")
        if not isinstance(self.shape, ShapeBucket):
            raise TypeError("shape must be a ShapeBucket")
        if not isinstance(self.capture, CaptureContract):
            raise TypeError("capture must be a CaptureContract")
        if isinstance(self.capture_specs, (str, bytes, bytearray)):
            raise TypeError("capture_specs must be a sequence of TypedCaptureSpec values")
        try:
            capture_specs = tuple(self.capture_specs)
        except TypeError as exc:
            raise TypeError("capture_specs must be a sequence of TypedCaptureSpec values") from exc
        if any(not isinstance(spec, TypedCaptureSpec) for spec in capture_specs):
            raise TypeError("capture_specs must contain only TypedCaptureSpec values")
        capture_ids = [spec.capture_id for spec in capture_specs]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("capture_specs must have unique capture IDs")
        object.__setattr__(
            self, "capture_specs", tuple(sorted(capture_specs, key=lambda x: x.capture_id))
        )
        if self.capture_capability is not None and not isinstance(
            self.capture_capability, CaptureCapability
        ):
            raise TypeError("capture_capability must be a CaptureCapability or None")
        object.__setattr__(
            self,
            "execution_mode",
            _coerce_enum(self.execution_mode, ExecutionMode, "execution_mode"),
        )
        object.__setattr__(
            self,
            "output_contract",
            _coerce_enum(self.output_contract, OutputContract, "output_contract"),
        )
        for field_name in (
            "request_ids",
            "prefix_state_ids",
            "kv_read_handles",
            "kv_write_handles",
            "page_sequence",
            "compute_layout_ids",
            "structured_operator_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_string_tuple(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self,
            "request_slots",
            _strict_int_tuple(self.request_slots, field="request_slots"),
        )
        object.__setattr__(
            self,
            "required_output_rows",
            _strict_int_tuple(self.required_output_rows, field="required_output_rows"),
        )
        object.__setattr__(
            self,
            "candidate_token_ids",
            _strict_nested_int_tuple(self.candidate_token_ids, field="candidate_token_ids"),
        )
        metadata = dict(_metadata_items(self.metadata))
        if self.execution_mode in {ExecutionMode.PREFILL, ExecutionMode.DECODE}:
            for key in ("op_graph_status", "work_floor_status"):
                declared = metadata.get(key)
                if declared not in {None, "not-modeled-v1"}:
                    raise ValueError(f"stateful {key} must be 'not-modeled-v1'")
                metadata[key] = "not-modeled-v1"
        object.__setattr__(self, "metadata", _metadata_items(metadata))
        if self.schema_version != WORKPLAN_SCHEMA:
            raise ValueError(f"unsupported work-plan schema: {self.schema_version}")
        for name, value in (
            ("model_name", self.model_name),
            ("model_revision", self.model_revision),
            ("store_fingerprint", self.store_fingerprint),
            ("numerical_contract", self.numerical_contract),
            ("cache_admission", self.cache_admission),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be a string")
            if not value:
                raise ValueError(f"{name} must be non-empty")
        batch = self.shape.actual_batch
        if any(not value or value.strip() != value for value in self.request_ids):
            raise ValueError("request_ids must be canonical non-empty strings")
        if len(self.request_ids) != batch:
            raise ValueError("request_ids must have one entry per actual batch row")
        if len(self.request_slots) != batch:
            raise ValueError("request_slots must have one entry per actual batch row")
        if len(set(self.request_ids)) != batch:
            raise ValueError("request_ids must be unique")
        if len(set(self.request_slots)) != batch or any(slot < 0 for slot in self.request_slots):
            raise ValueError("request_slots must be unique non-negative integers")
        if self.output_contract is OutputContract.SELECTED_CAPTURE:
            if not self.capture_specs or self.capture_capability is None:
                raise ValueError(
                    "selected_capture requires typed capture specs and an exact backend capability"
                )
            if self.required_output_rows or self.candidate_token_ids:
                raise ValueError("selected_capture cannot carry vocabulary output selections")
            request_ids = set(self.request_ids)
            for spec in self.capture_specs:
                if not set(spec.specimen_row_ids).issubset(request_ids):
                    raise ValueError(
                        f"capture {spec.capture_id!r} names specimen rows outside the WorkPlan"
                    )
                if spec.numerical_contract != self.numerical_contract:
                    raise ValueError(
                        f"capture {spec.capture_id!r} numerical contract differs from its WorkPlan"
                    )
                if any(index >= self.shape.sequence_length for index in spec.token_indices):
                    raise ValueError(f"capture {spec.capture_id!r} token index exceeds the plan")
                if spec.token_span and spec.token_span[1] > self.shape.sequence_length:
                    raise ValueError(f"capture {spec.capture_id!r} token span exceeds the plan")
                if (
                    spec.cursor_index is not None
                    and spec.cursor_index >= self.shape.sequence_length
                ):
                    raise ValueError(f"capture {spec.capture_id!r} cursor exceeds the plan")
            self.capture_capability.validate_specs(self.capture_specs)
        elif self.capture_specs or self.capture_capability is not None:
            raise ValueError(
                "typed capture specs and capabilities are legal only for selected_capture"
            )
        for field_name, handles in (
            ("prefix_state_ids", self.prefix_state_ids),
            ("kv_read_handles", self.kv_read_handles),
            ("kv_write_handles", self.kv_write_handles),
        ):
            if handles and len(handles) != batch:
                raise ValueError(f"{field_name} must be empty or have one entry per batch row")
            if any(not handle or handle.strip() != handle for handle in handles):
                raise ValueError(f"{field_name} must contain canonical non-empty strings")
        if self.execution_mode in {ExecutionMode.PREFILL, ExecutionMode.DECODE}:
            if self.output_contract is OutputContract.LOSS_ONLY:
                raise ValueError("loss_only is unavailable for prefill/decode WorkPlans")
            if self.prefix_state_ids:
                raise ValueError("stateful WorkPlans do not admit prefix_state_ids")
            if self.request_slots != tuple(range(batch)):
                raise ValueError("stateful request_slots must be contiguous and row ordered")
            if len(self.kv_read_handles) != batch or len(self.kv_write_handles) != batch:
                raise ValueError("stateful WorkPlans require one KV read/write handle per row")
            all_kv_handles = (*self.kv_read_handles, *self.kv_write_handles)
            if any(not handle.strip() for handle in all_kv_handles):
                raise ValueError("stateful KV handles must be non-empty")
            if self.kv_read_handles != self.kv_write_handles:
                raise ValueError("stateful KV read/write handles must match row by row")
            if len(set(self.kv_read_handles)) != batch:
                raise ValueError("stateful KV handles must be unique per row")
            kv_capacity = dict(self.metadata).get("kv_capacity")
            if isinstance(kv_capacity, bool) or not isinstance(kv_capacity, int):
                raise ValueError("stateful WorkPlans require integer kv_capacity metadata")
            if kv_capacity < self.shape.sequence_length:
                raise ValueError("kv_capacity cannot be smaller than the planned token block")
            kv_dtype = dict(self.metadata).get("kv_dtype")
            if not isinstance(kv_dtype, str) or kv_dtype not in _KV_DTYPES:
                raise ValueError("stateful WorkPlans require supported kv_dtype metadata")
            for key in ("kv_num_layers", "kv_num_heads", "kv_head_dim"):
                value = dict(self.metadata).get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"stateful WorkPlans require positive integer {key} metadata")
        elif self.prefix_state_ids or self.kv_read_handles or self.kv_write_handles:
            raise ValueError("score WorkPlans cannot claim prefix or KV state bindings")
        input_token_limit = dict(self.metadata).get("input_token_limit")
        if input_token_limit is not None and (
            isinstance(input_token_limit, bool)
            or not isinstance(input_token_limit, int)
            or input_token_limit <= 0
        ):
            raise ValueError("input_token_limit metadata must be a positive integer")
        if any(row < 0 for row in self.required_output_rows):
            raise ValueError("required output rows must be non-negative")
        if input_token_limit is not None and any(
            row >= input_token_limit for row in self.required_output_rows
        ):
            raise ValueError(
                f"required output rows must be inside semantic token space [0, {input_token_limit})"
            )
        if len(set(self.required_output_rows)) != len(self.required_output_rows):
            raise ValueError("required output rows must be unique")
        if self.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
            if not self.required_output_rows:
                raise ValueError("selected_token_rows requires required_output_rows")
        if self.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
            if len(self.candidate_token_ids) != batch:
                raise ValueError("candidate contract requires candidate IDs for every batch row")
            if any(len(row) < 2 or len(row) != len(set(row)) for row in self.candidate_token_ids):
                raise ValueError(
                    "candidate margin requires at least two distinct, non-duplicate candidates "
                    "per row"
                )
            if any(token < 0 for row in self.candidate_token_ids for token in row):
                raise ValueError("candidate token IDs must be non-negative")
            if input_token_limit is not None and any(
                token >= input_token_limit for row in self.candidate_token_ids for token in row
            ):
                raise ValueError(
                    "candidate token IDs must be inside semantic token space "
                    f"[0, {input_token_limit})"
                )
        elif self.candidate_token_ids:
            raise ValueError("candidate_token_ids are only legal for the candidate output contract")
        if (
            self.output_contract
            in {
                OutputContract.SELECTED_TOKEN_ROWS,
                OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            }
            and input_token_limit is None
        ):
            raise ValueError(
                f"{self.output_contract.value} requires input_token_limit metadata "
                "to bind the semantic output domain"
            )
        _validate_redundant_head_metadata(
            output_contract=self.output_contract,
            metadata=dict(self.metadata),
            required_output_row_count=len(self.required_output_rows),
            candidate_union_count=len(
                {token for row_candidates in self.candidate_token_ids for token in row_candidates}
            ),
        )

    @cached_property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @cached_property
    def executable_contract_fingerprint(self) -> str:
        """Fingerprint static executable structure without request-scoped bindings.

        A full plan fingerprint remains the provenance identity for one concrete workload.
        The reusable identity is the canonical :class:`WorkTemplate` identity. Vocabulary
        IDs are dispatch data; their cardinalities remain structural because they determine
        output and workspace shapes.
        """

        return WorkTemplate.from_plan(self).fingerprint

    @property
    def content_addressed(self) -> bool:
        return _is_sha256_digest(self.model_revision) and _is_sha256_digest(self.store_fingerprint)

    @property
    def content_identity_verified(self) -> bool:
        """Whether a loaded-store certificate, not digest syntax, verified the content."""

        metadata = dict(self.metadata)
        return (
            self.content_addressed
            and metadata.get("source_identity_status") == "content-addressed-verified"
            and metadata.get("store_identity_status") == "content-addressed-semantic-verified"
            and metadata.get("blob_identity_verified") is True
            and all(
                _is_sha256_digest(str(metadata.get(field_name, "")))
                for field_name in (
                    "identity_certificate_sha256",
                    "manifest_semantic_sha256",
                    "builder_source_bundle_sha256",
                    "blob_records_sha256",
                )
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "store_fingerprint": self.store_fingerprint,
            "content_addressed": self.content_addressed,
            "content_identity_verified": self.content_identity_verified,
            "execution_mode": self.execution_mode.value,
            "precision": self.precision.as_dict(),
            "shape": self.shape.as_dict(),
            "output_contract": self.output_contract.value,
            "numerical_contract": self.numerical_contract,
            "request_ids": list(self.request_ids),
            "request_slots": list(self.request_slots),
            "prefix_state_ids": list(self.prefix_state_ids),
            "kv_read_handles": list(self.kv_read_handles),
            "kv_write_handles": list(self.kv_write_handles),
            "required_output_rows": list(self.required_output_rows),
            "candidate_token_ids": [list(row) for row in self.candidate_token_ids],
            "page_sequence": list(self.page_sequence),
            "cache_admission": self.cache_admission,
            "compute_layout_ids": list(self.compute_layout_ids),
            "structured_operator_ids": list(self.structured_operator_ids),
            "capture": self.capture.as_dict(),
            "capture_specs": [spec.as_dict() for spec in self.capture_specs],
            "capture_capability": (
                None if self.capture_capability is None else self.capture_capability.as_dict()
            ),
            "metadata": {key: value for key, value in self.metadata},
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the canonical, human-auditable plan representation."""

        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DenseWorkPlan:
        """Reconstruct and fully revalidate a plan from its serialized form."""

        schema = str(payload.get("schema_version", ""))
        if schema in LEGACY_WORKPLAN_SCHEMAS:
            state_fields = ("prefix_state_ids", "kv_read_handles", "kv_write_handles")
            if schema == "mrun-dense-workplan-v1" and (
                str(payload.get("execution_mode", "")) != ExecutionMode.SCORE.value
                or any(payload.get(field_name) for field_name in state_fields)
            ):
                raise ValueError(
                    f"legacy stateful WorkPlan schema {schema!r} cannot be migrated; rebuild as "
                    f"{WORKPLAN_SCHEMA!r} so KV semantics are explicit"
                )
            if str(payload.get("output_contract", "")) == OutputContract.SELECTED_CAPTURE.value:
                raise ValueError("legacy selected_capture WorkPlans have no typed v3 migration")
            migrated = dict(payload)
            migrated["schema_version"] = WORKPLAN_SCHEMA
            migrated["capture_specs"] = []
            migrated["capture_capability"] = None
            raw_metadata = migrated.get("metadata", {})
            if not isinstance(raw_metadata, Mapping):
                raise TypeError("metadata must be an object")
            metadata_copy = dict(raw_metadata)
            metadata_copy["legacy_schema_migrated_from"] = schema
            migrated["metadata"] = metadata_copy
            payload = migrated

        precision = payload.get("precision")
        shape = payload.get("shape")
        capture = payload.get("capture", {})
        capture_specs = payload.get("capture_specs", ())
        capture_capability = payload.get("capture_capability")
        metadata = payload.get("metadata", {})
        if not isinstance(precision, Mapping):
            raise TypeError("precision must be an object")
        if not isinstance(shape, Mapping):
            raise TypeError("shape must be an object")
        if not isinstance(capture, Mapping):
            raise TypeError("capture must be an object")
        if not isinstance(capture_specs, Sequence) or isinstance(
            capture_specs, (str, bytes, bytearray)
        ):
            raise TypeError("capture_specs must be an array")
        if capture_capability is not None and not isinstance(capture_capability, Mapping):
            raise TypeError("capture_capability must be an object or null")
        if any(not isinstance(spec, Mapping) for spec in capture_specs):
            raise TypeError("capture_specs must contain only objects")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be an object")
        return cls(
            model_name=_strict_string(payload["model_name"], field="DenseWorkPlan.model_name"),
            model_revision=_strict_string(
                payload["model_revision"], field="DenseWorkPlan.model_revision"
            ),
            store_fingerprint=_strict_string(
                payload["store_fingerprint"], field="DenseWorkPlan.store_fingerprint"
            ),
            execution_mode=_strict_string(
                payload["execution_mode"], field="DenseWorkPlan.execution_mode"
            ),  # type: ignore[arg-type]
            precision=PrecisionPolicy.from_dict(precision),
            shape=ShapeBucket.from_dict(shape),
            output_contract=_strict_string(
                payload["output_contract"], field="DenseWorkPlan.output_contract"
            ),  # type: ignore[arg-type]
            numerical_contract=_strict_string(
                payload["numerical_contract"], field="DenseWorkPlan.numerical_contract"
            ),
            request_ids=_strict_string_tuple(
                payload["request_ids"], field="DenseWorkPlan.request_ids"
            ),
            request_slots=_strict_int_tuple(
                payload["request_slots"], field="DenseWorkPlan.request_slots"
            ),
            prefix_state_ids=_strict_string_tuple(
                payload.get("prefix_state_ids", ()), field="DenseWorkPlan.prefix_state_ids"
            ),
            kv_read_handles=_strict_string_tuple(
                payload.get("kv_read_handles", ()), field="DenseWorkPlan.kv_read_handles"
            ),
            kv_write_handles=_strict_string_tuple(
                payload.get("kv_write_handles", ()), field="DenseWorkPlan.kv_write_handles"
            ),
            required_output_rows=_strict_int_tuple(
                payload.get("required_output_rows", ()),
                field="DenseWorkPlan.required_output_rows",
            ),
            candidate_token_ids=_strict_nested_int_tuple(
                payload.get("candidate_token_ids", ()),
                field="DenseWorkPlan.candidate_token_ids",
            ),
            page_sequence=_strict_string_tuple(
                payload.get("page_sequence", ()), field="DenseWorkPlan.page_sequence"
            ),
            cache_admission=_strict_string(
                payload.get("cache_admission", "default"),
                field="DenseWorkPlan.cache_admission",
            ),
            compute_layout_ids=_strict_string_tuple(
                payload.get("compute_layout_ids", ()),
                field="DenseWorkPlan.compute_layout_ids",
            ),
            structured_operator_ids=_strict_string_tuple(
                payload.get("structured_operator_ids", ()),
                field="DenseWorkPlan.structured_operator_ids",
            ),
            capture=CaptureContract.from_dict(capture),
            capture_specs=tuple(TypedCaptureSpec.from_dict(spec) for spec in capture_specs),
            capture_capability=(
                None
                if capture_capability is None
                else CaptureCapability.from_dict(capture_capability)
            ),
            metadata=_metadata_items(metadata),
            schema_version=_strict_string(
                payload.get("schema_version", WORKPLAN_SCHEMA),
                field="DenseWorkPlan.schema_version",
            ),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> DenseWorkPlan:
        return cls.from_dict(_strict_json_loads(payload, field="DenseWorkPlan"))


@dataclass(frozen=True)
class WorkTemplate:
    """Binding-free executable contract shared by compatible dispatches.

    Every field that can alter lowering, allocation, numerical behavior, or output shape is
    structural. Request identity, mutable-state handles, and the *values* of vocabulary rows
    live in :class:`DispatchBinding`. Their cardinalities remain here because current kernels
    allocate from them.
    """

    model_name: str
    model_revision: str
    store_fingerprint: str
    execution_mode: ExecutionMode
    precision: PrecisionPolicy
    shape: ShapeBucket
    output_contract: OutputContract
    numerical_contract: str
    prefix_state_count: int = 0
    kv_read_handle_count: int = 0
    kv_write_handle_count: int = 0
    required_output_row_count: int = 0
    candidate_row_counts: tuple[int, ...] = ()
    candidate_union_count: int = 0
    page_sequence: tuple[str, ...] = ()
    cache_admission: str = "default"
    compute_layout_ids: tuple[str, ...] = ()
    structured_operator_ids: tuple[str, ...] = ()
    capture: CaptureContract = field(default_factory=CaptureContract)
    capture_specs: tuple[TypedCaptureSpec, ...] = ()
    capture_capability: CaptureCapability | None = None
    metadata: tuple[tuple[str, JsonScalar], ...] = ()
    schema_version: str = WORK_TEMPLATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_mode",
            _coerce_enum(self.execution_mode, ExecutionMode, "execution_mode"),
        )
        object.__setattr__(
            self,
            "output_contract",
            _coerce_enum(self.output_contract, OutputContract, "output_contract"),
        )
        if not isinstance(self.precision, PrecisionPolicy):
            raise TypeError("template precision must be a PrecisionPolicy")
        if not isinstance(self.shape, ShapeBucket):
            raise TypeError("template shape must be a ShapeBucket")
        if not isinstance(self.capture, CaptureContract):
            raise TypeError("template capture must be a CaptureContract")
        if isinstance(self.capture_specs, (str, bytes, bytearray)):
            raise TypeError("template capture_specs must be a sequence")
        try:
            capture_specs = tuple(self.capture_specs)
        except TypeError as exc:
            raise TypeError("template capture_specs must be a sequence") from exc
        if any(not isinstance(spec, TypedCaptureSpec) for spec in capture_specs):
            raise TypeError("template capture_specs must contain TypedCaptureSpec values")
        capture_ids = [spec.capture_id for spec in capture_specs]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("template capture_specs must have unique capture IDs")
        object.__setattr__(
            self,
            "capture_specs",
            tuple(sorted(capture_specs, key=lambda spec: spec.capture_id)),
        )
        if self.capture_capability is not None and not isinstance(
            self.capture_capability, CaptureCapability
        ):
            raise TypeError("template capture_capability must be a CaptureCapability or None")
        object.__setattr__(self, "metadata", _work_template_metadata_items(self.metadata))
        object.__setattr__(
            self,
            "candidate_row_counts",
            _strict_int_tuple(self.candidate_row_counts, field="template candidate_row_counts"),
        )
        for field_name in ("page_sequence", "compute_layout_ids", "structured_operator_ids"):
            object.__setattr__(
                self,
                field_name,
                _strict_string_tuple(getattr(self, field_name), field=f"template {field_name}"),
            )
        if self.schema_version != WORK_TEMPLATE_SCHEMA:
            raise ValueError(f"unsupported work-template schema: {self.schema_version}")
        for field_name, value in (
            ("model_name", self.model_name),
            ("model_revision", self.model_revision),
            ("store_fingerprint", self.store_fingerprint),
            ("numerical_contract", self.numerical_contract),
            ("cache_admission", self.cache_admission),
        ):
            if type(value) is not str:
                raise TypeError(f"template {field_name} must be a string")
            if not value:
                raise ValueError(f"template {field_name} must be non-empty")
        counts = {
            "prefix_state_count": self.prefix_state_count,
            "kv_read_handle_count": self.kv_read_handle_count,
            "kv_write_handle_count": self.kv_write_handle_count,
            "required_output_row_count": self.required_output_row_count,
            "candidate_union_count": self.candidate_union_count,
        }
        for field_name, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"template {field_name} must be a non-negative integer")
        batch = self.shape.actual_batch
        for field_name, count in (
            ("prefix_state_count", self.prefix_state_count),
            ("kv_read_handle_count", self.kv_read_handle_count),
            ("kv_write_handle_count", self.kv_write_handle_count),
        ):
            if count not in {0, batch}:
                raise ValueError(f"template {field_name} must be zero or actual batch")
        if self.execution_mode in {ExecutionMode.PREFILL, ExecutionMode.DECODE}:
            if self.prefix_state_count:
                raise ValueError("stateful templates do not admit prefix-state bindings")
            if self.kv_read_handle_count != batch or self.kv_write_handle_count != batch:
                raise ValueError("stateful templates require one KV read/write handle per row")
        elif self.prefix_state_count or self.kv_read_handle_count or self.kv_write_handle_count:
            raise ValueError("score templates cannot claim prefix or KV bindings")
        if self.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
            if self.required_output_row_count <= 0:
                raise ValueError("selected-token templates require output rows")
        if self.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
            if len(self.candidate_row_counts) != batch:
                raise ValueError("candidate templates require one row width per batch row")
            if any(count < 2 for count in self.candidate_row_counts):
                raise ValueError("candidate template row widths must be at least two")
            if not (
                max(self.candidate_row_counts)
                <= self.candidate_union_count
                <= sum(self.candidate_row_counts)
            ):
                raise ValueError("candidate template union cardinality is impossible")
        elif self.candidate_row_counts or self.candidate_union_count:
            raise ValueError("candidate shape is legal only for the candidate output contract")
        token_limit = dict(self.metadata).get("input_token_limit")
        if isinstance(token_limit, int) and not isinstance(token_limit, bool):
            if self.required_output_row_count > token_limit:
                raise ValueError("template output-row count exceeds the semantic token space")
            if self.candidate_union_count > token_limit:
                raise ValueError("template candidate union exceeds the semantic token space")
        _validate_redundant_head_metadata(
            output_contract=self.output_contract,
            metadata=dict(self.metadata),
            required_output_row_count=self.required_output_row_count,
            candidate_union_count=self.candidate_union_count,
        )
        # DenseWorkPlan remains the single semantic validator. A deterministic representative
        # catches malformed static metadata and prevents the template schema from drifting.
        self._materialize(self._representative_binding())

    @classmethod
    def from_plan(cls, plan: DenseWorkPlan) -> WorkTemplate:
        reusable_metadata, _ = _partition_workplan_metadata(plan.metadata)
        return cls(
            model_name=plan.model_name,
            model_revision=plan.model_revision,
            store_fingerprint=plan.store_fingerprint,
            execution_mode=plan.execution_mode,
            precision=plan.precision,
            shape=plan.shape,
            output_contract=plan.output_contract,
            numerical_contract=plan.numerical_contract,
            prefix_state_count=len(plan.prefix_state_ids),
            kv_read_handle_count=len(plan.kv_read_handles),
            kv_write_handle_count=len(plan.kv_write_handles),
            required_output_row_count=len(plan.required_output_rows),
            candidate_row_counts=tuple(len(row) for row in plan.candidate_token_ids),
            candidate_union_count=len({token for row in plan.candidate_token_ids for token in row}),
            page_sequence=plan.page_sequence,
            cache_admission=plan.cache_admission,
            compute_layout_ids=plan.compute_layout_ids,
            structured_operator_ids=plan.structured_operator_ids,
            capture=plan.capture,
            capture_specs=plan.capture_specs,
            capture_capability=plan.capture_capability,
            metadata=reusable_metadata,
        )

    def _representative_candidates(self) -> tuple[tuple[int, ...], ...]:
        if not self.candidate_row_counts:
            return ()
        cursor = 0
        rows: list[tuple[int, ...]] = []
        for count in self.candidate_row_counts:
            rows.append(
                tuple((cursor + offset) % self.candidate_union_count for offset in range(count))
            )
            cursor = (cursor + count) % self.candidate_union_count
        return tuple(rows)

    def _representative_binding(self) -> DispatchBinding:
        batch = self.shape.actual_batch
        request_ids: list[str] = []
        for spec in self.capture_specs:
            for row_id in spec.specimen_row_ids:
                if row_id not in request_ids:
                    request_ids.append(row_id)
        if len(request_ids) > batch:
            raise ValueError("template capture rows exceed the batch cardinality")
        for index in range(batch - len(request_ids)):
            candidate = f"template-request-{index}"
            while candidate in request_ids:
                candidate = f"{candidate}-next"
            request_ids.append(candidate)
        return DispatchBinding(
            template_fingerprint=self.fingerprint,
            request_ids=tuple(request_ids),
            request_slots=tuple(range(batch)),
            prefix_state_ids=tuple(
                f"template-prefix-{index}" for index in range(self.prefix_state_count)
            ),
            kv_read_handles=tuple(
                f"template-kv-{index}" for index in range(self.kv_read_handle_count)
            ),
            kv_write_handles=tuple(
                f"template-kv-{index}" for index in range(self.kv_write_handle_count)
            ),
            required_output_rows=tuple(range(self.required_output_row_count)),
            candidate_token_ids=self._representative_candidates(),
        )

    def _materialize(self, binding: DispatchBinding) -> DenseWorkPlan:
        metadata = dict(self.metadata)
        overlap = set(metadata).intersection(dict(binding.dispatch_metadata))
        if overlap:
            raise ValueError(
                f"dispatch metadata shadows structural template keys: {sorted(overlap)!r}"
            )
        metadata.update(binding.dispatch_metadata)
        return DenseWorkPlan(
            model_name=self.model_name,
            model_revision=self.model_revision,
            store_fingerprint=self.store_fingerprint,
            execution_mode=self.execution_mode,
            precision=self.precision,
            shape=self.shape,
            output_contract=self.output_contract,
            numerical_contract=self.numerical_contract,
            request_ids=binding.request_ids,
            request_slots=binding.request_slots,
            prefix_state_ids=binding.prefix_state_ids,
            kv_read_handles=binding.kv_read_handles,
            kv_write_handles=binding.kv_write_handles,
            required_output_rows=binding.required_output_rows,
            candidate_token_ids=binding.candidate_token_ids,
            page_sequence=self.page_sequence,
            cache_admission=self.cache_admission,
            compute_layout_ids=self.compute_layout_ids,
            structured_operator_ids=self.structured_operator_ids,
            capture=self.capture,
            capture_specs=self.capture_specs,
            capture_capability=self.capture_capability,
            metadata=metadata,
        )

    def bind(self, binding: DispatchBinding) -> DenseWorkPlan:
        """Attach one request binding and recover the fully validated concrete WorkPlan."""

        binding.validate_for(self)
        plan = self._materialize(binding)
        if WorkTemplate.from_plan(plan).fingerprint != self.fingerprint:
            raise RuntimeError("materialized WorkPlan changed its WorkTemplate identity")
        return plan

    @cached_property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "store_fingerprint": self.store_fingerprint,
            "execution_mode": self.execution_mode.value,
            "precision": self.precision.as_dict(),
            "shape": self.shape.as_dict(),
            "output_contract": self.output_contract.value,
            "numerical_contract": self.numerical_contract,
            "dispatch_shape": {
                "prefix_state_count": self.prefix_state_count,
                "kv_read_handle_count": self.kv_read_handle_count,
                "kv_write_handle_count": self.kv_write_handle_count,
                "required_output_row_count": self.required_output_row_count,
                "candidate_row_counts": list(self.candidate_row_counts),
                "candidate_union_count": self.candidate_union_count,
            },
            "page_sequence": list(self.page_sequence),
            "cache_admission": self.cache_admission,
            "compute_layout_ids": list(self.compute_layout_ids),
            "structured_operator_ids": list(self.structured_operator_ids),
            "capture": self.capture.as_dict(),
            "capture_specs": [spec.as_dict() for spec in self.capture_specs],
            "capture_capability": (
                None if self.capture_capability is None else self.capture_capability.as_dict()
            ),
            "metadata": {key: value for key, value in self.metadata},
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkTemplate:
        schema = str(payload.get("schema_version", ""))
        if schema in LEGACY_WORK_TEMPLATE_SCHEMAS:
            if str(payload.get("output_contract", "")) == OutputContract.SELECTED_CAPTURE.value:
                raise ValueError("legacy selected_capture WorkTemplates have no typed v2 migration")
            migrated = dict(payload)
            migrated["schema_version"] = WORK_TEMPLATE_SCHEMA
            migrated["capture_specs"] = []
            migrated["capture_capability"] = None
            payload = migrated
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "model_name",
                "model_revision",
                "store_fingerprint",
                "execution_mode",
                "precision",
                "shape",
                "output_contract",
                "numerical_contract",
                "dispatch_shape",
                "page_sequence",
                "cache_admission",
                "compute_layout_ids",
                "structured_operator_ids",
                "capture",
                "capture_specs",
                "capture_capability",
                "metadata",
            },
            field="WorkTemplate",
        )
        precision = payload.get("precision")
        shape = payload.get("shape")
        dispatch_shape = payload.get("dispatch_shape")
        capture = payload.get("capture")
        capture_specs = payload.get("capture_specs")
        capture_capability = payload.get("capture_capability")
        metadata = payload.get("metadata")
        if type(precision) is not dict:
            raise TypeError("template precision must be an object")
        if type(shape) is not dict:
            raise TypeError("template shape must be an object")
        if type(dispatch_shape) is not dict:
            raise TypeError("template dispatch_shape must be an object")
        if type(capture) is not dict:
            raise TypeError("template capture must be an object")
        if type(capture_specs) is not list or any(type(spec) is not dict for spec in capture_specs):
            raise TypeError("template capture_specs must be an array of objects")
        if capture_capability is not None and type(capture_capability) is not dict:
            raise TypeError("template capture_capability must be an object or null")
        if type(metadata) is not dict:
            raise TypeError("template metadata must be an object")
        _require_exact_keys(
            precision,
            {"activation_dtype", "weight_dtype", "accumulator_dtype"},
            field="WorkTemplate.precision",
        )
        _require_exact_keys(
            shape,
            {
                "actual_batch",
                "batch_bucket",
                "sequence_length",
                "sequence_bucket",
                "live_token_rows",
            },
            field="WorkTemplate.shape",
        )
        _require_exact_keys(
            dispatch_shape,
            {
                "prefix_state_count",
                "kv_read_handle_count",
                "kv_write_handle_count",
                "required_output_row_count",
                "candidate_row_counts",
                "candidate_union_count",
            },
            field="WorkTemplate.dispatch_shape",
        )
        _require_exact_keys(
            capture,
            {
                "requested",
                "static_shapes",
                "stable_addresses",
                "graph_safe",
                "eligible",
                "refusal_reasons",
            },
            field="WorkTemplate.capture",
        )
        actual_batch = _strict_int(shape["actual_batch"], field="WorkTemplate.shape.actual_batch")
        sequence_length = _strict_int(
            shape["sequence_length"], field="WorkTemplate.shape.sequence_length"
        )
        live_token_rows = _strict_int(
            shape["live_token_rows"], field="WorkTemplate.shape.live_token_rows"
        )
        if live_token_rows != actual_batch * sequence_length:
            raise ValueError("WorkTemplate.shape.live_token_rows is not canonical")
        capture_contract = CaptureContract(
            requested=_strict_bool(capture["requested"], field="WorkTemplate.capture.requested"),
            static_shapes=_strict_bool(
                capture["static_shapes"], field="WorkTemplate.capture.static_shapes"
            ),
            stable_addresses=_strict_bool(
                capture["stable_addresses"], field="WorkTemplate.capture.stable_addresses"
            ),
            graph_safe=_strict_bool(capture["graph_safe"], field="WorkTemplate.capture.graph_safe"),
        )
        if _strict_bool(capture["eligible"], field="WorkTemplate.capture.eligible") != (
            capture_contract.eligible
        ):
            raise ValueError("WorkTemplate.capture.eligible is not canonical")
        claimed_reasons = _strict_string_list(
            capture["refusal_reasons"], field="WorkTemplate.capture.refusal_reasons"
        )
        if claimed_reasons != capture_contract.refusal_reasons:
            raise ValueError("WorkTemplate.capture.refusal_reasons are not canonical")
        candidate_counts = tuple(
            _strict_int(item, field=f"WorkTemplate.dispatch_shape.candidate_row_counts[{index}]")
            for index, item in enumerate(
                _strict_list(
                    dispatch_shape["candidate_row_counts"],
                    field="WorkTemplate.dispatch_shape.candidate_row_counts",
                )
            )
        )
        return cls(
            model_name=_strict_string(payload["model_name"], field="WorkTemplate.model_name"),
            model_revision=_strict_string(
                payload["model_revision"], field="WorkTemplate.model_revision"
            ),
            store_fingerprint=_strict_string(
                payload["store_fingerprint"], field="WorkTemplate.store_fingerprint"
            ),
            execution_mode=_strict_string(
                payload["execution_mode"], field="WorkTemplate.execution_mode"
            ),  # type: ignore[arg-type]
            precision=PrecisionPolicy(
                activation_dtype=_strict_string(
                    precision["activation_dtype"],
                    field="WorkTemplate.precision.activation_dtype",
                ),
                weight_dtype=_strict_string(
                    precision["weight_dtype"], field="WorkTemplate.precision.weight_dtype"
                ),
                accumulator_dtype=_strict_string(
                    precision["accumulator_dtype"],
                    field="WorkTemplate.precision.accumulator_dtype",
                ),
            ),
            shape=ShapeBucket(
                actual_batch=actual_batch,
                batch_bucket=_strict_int(
                    shape["batch_bucket"], field="WorkTemplate.shape.batch_bucket"
                ),
                sequence_length=sequence_length,
                sequence_bucket=_strict_int(
                    shape["sequence_bucket"], field="WorkTemplate.shape.sequence_bucket"
                ),
            ),
            output_contract=_strict_string(
                payload["output_contract"], field="WorkTemplate.output_contract"
            ),  # type: ignore[arg-type]
            numerical_contract=_strict_string(
                payload["numerical_contract"], field="WorkTemplate.numerical_contract"
            ),
            prefix_state_count=_strict_int(
                dispatch_shape["prefix_state_count"],
                field="WorkTemplate.dispatch_shape.prefix_state_count",
            ),
            kv_read_handle_count=_strict_int(
                dispatch_shape["kv_read_handle_count"],
                field="WorkTemplate.dispatch_shape.kv_read_handle_count",
            ),
            kv_write_handle_count=_strict_int(
                dispatch_shape["kv_write_handle_count"],
                field="WorkTemplate.dispatch_shape.kv_write_handle_count",
            ),
            required_output_row_count=_strict_int(
                dispatch_shape["required_output_row_count"],
                field="WorkTemplate.dispatch_shape.required_output_row_count",
            ),
            candidate_row_counts=candidate_counts,
            candidate_union_count=_strict_int(
                dispatch_shape["candidate_union_count"],
                field="WorkTemplate.dispatch_shape.candidate_union_count",
            ),
            page_sequence=_strict_string_list(
                payload["page_sequence"], field="WorkTemplate.page_sequence"
            ),
            cache_admission=_strict_string(
                payload["cache_admission"], field="WorkTemplate.cache_admission"
            ),
            compute_layout_ids=_strict_string_list(
                payload["compute_layout_ids"], field="WorkTemplate.compute_layout_ids"
            ),
            structured_operator_ids=_strict_string_list(
                payload["structured_operator_ids"],
                field="WorkTemplate.structured_operator_ids",
            ),
            capture=capture_contract,
            capture_specs=tuple(TypedCaptureSpec.from_dict(spec) for spec in capture_specs),
            capture_capability=(
                None
                if capture_capability is None
                else CaptureCapability.from_dict(capture_capability)
            ),
            metadata=_work_template_metadata_items(metadata),
            schema_version=_strict_string(
                payload["schema_version"], field="WorkTemplate.schema_version"
            ),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> WorkTemplate:
        return cls.from_dict(_strict_json_loads(payload, field="WorkTemplate"))


@dataclass(frozen=True)
class DispatchBinding:
    """Immutable request/state/provenance attachment for one compatible WorkTemplate."""

    template_fingerprint: str
    request_ids: tuple[str, ...]
    request_slots: tuple[int, ...]
    prefix_state_ids: tuple[str, ...] = ()
    kv_read_handles: tuple[str, ...] = ()
    kv_write_handles: tuple[str, ...] = ()
    required_output_rows: tuple[int, ...] = ()
    candidate_token_ids: tuple[tuple[int, ...], ...] = ()
    dispatch_metadata: tuple[tuple[str, JsonScalar], ...] = ()
    schema_version: str = DISPATCH_BINDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_ids",
            _strict_string_tuple(self.request_ids, field="dispatch request_ids"),
        )
        object.__setattr__(
            self,
            "request_slots",
            _strict_int_tuple(self.request_slots, field="dispatch request_slots"),
        )
        for field_name in ("prefix_state_ids", "kv_read_handles", "kv_write_handles"):
            object.__setattr__(
                self,
                field_name,
                _strict_string_tuple(getattr(self, field_name), field=f"dispatch {field_name}"),
            )
        object.__setattr__(
            self,
            "required_output_rows",
            _strict_int_tuple(self.required_output_rows, field="dispatch required_output_rows"),
        )
        object.__setattr__(
            self,
            "candidate_token_ids",
            _strict_nested_int_tuple(
                self.candidate_token_ids, field="dispatch candidate_token_ids"
            ),
        )
        object.__setattr__(
            self,
            "dispatch_metadata",
            _dispatch_metadata_items(self.dispatch_metadata),
        )
        if self.schema_version != DISPATCH_BINDING_SCHEMA:
            raise ValueError(f"unsupported dispatch-binding schema: {self.schema_version}")
        if not _is_sha256_digest(self.template_fingerprint):
            raise ValueError("dispatch binding requires a canonical template fingerprint")
        if any(not value or value.strip() != value for value in self.request_ids):
            raise ValueError("dispatch request IDs must be canonical non-empty strings")
        for field_name in ("prefix_state_ids", "kv_read_handles", "kv_write_handles"):
            if any(not value or value.strip() != value for value in getattr(self, field_name)):
                raise ValueError(f"dispatch {field_name} must contain canonical non-empty strings")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("dispatch request IDs must be unique")
        if len(set(self.request_slots)) != len(self.request_slots) or any(
            slot < 0 for slot in self.request_slots
        ):
            raise ValueError("dispatch request slots must be unique non-negative integers")
        if len(set(self.required_output_rows)) != len(self.required_output_rows) or any(
            value < 0 for value in self.required_output_rows
        ):
            raise ValueError("dispatch output rows must be unique non-negative integers")

    @classmethod
    def from_plan(
        cls,
        plan: DenseWorkPlan,
        *,
        template: WorkTemplate | None = None,
    ) -> DispatchBinding:
        actual_template = WorkTemplate.from_plan(plan)
        if template is not None and template.fingerprint != actual_template.fingerprint:
            raise ValueError("WorkPlan does not belong to the supplied WorkTemplate")
        _, dispatch_metadata = _partition_workplan_metadata(plan.metadata)
        return cls(
            template_fingerprint=actual_template.fingerprint,
            request_ids=plan.request_ids,
            request_slots=plan.request_slots,
            prefix_state_ids=plan.prefix_state_ids,
            kv_read_handles=plan.kv_read_handles,
            kv_write_handles=plan.kv_write_handles,
            required_output_rows=plan.required_output_rows,
            candidate_token_ids=plan.candidate_token_ids,
            dispatch_metadata=dispatch_metadata,
        )

    def validate_for(self, template: WorkTemplate) -> None:
        if self.template_fingerprint != template.fingerprint:
            raise ValueError("dispatch binding belongs to a different WorkTemplate")
        batch = template.shape.actual_batch
        expected_counts = {
            "request_ids": batch,
            "request_slots": batch,
            "prefix_state_ids": template.prefix_state_count,
            "kv_read_handles": template.kv_read_handle_count,
            "kv_write_handles": template.kv_write_handle_count,
            "required_output_rows": template.required_output_row_count,
        }
        for field_name, count in expected_counts.items():
            if len(getattr(self, field_name)) != count:
                raise ValueError(f"dispatch {field_name} cardinality does not match its template")
        row_counts = tuple(len(row) for row in self.candidate_token_ids)
        if row_counts != template.candidate_row_counts:
            raise ValueError("dispatch candidate row widths do not match its template")
        union_count = len({token for row in self.candidate_token_ids for token in row})
        if union_count != template.candidate_union_count:
            raise ValueError("dispatch candidate union cardinality does not match its template")
        _dispatch_metadata_items(self.dispatch_metadata)

    @cached_property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "template_fingerprint": self.template_fingerprint,
            "request_ids": list(self.request_ids),
            "request_slots": list(self.request_slots),
            "prefix_state_ids": list(self.prefix_state_ids),
            "kv_read_handles": list(self.kv_read_handles),
            "kv_write_handles": list(self.kv_write_handles),
            "required_output_rows": list(self.required_output_rows),
            "candidate_token_ids": [list(row) for row in self.candidate_token_ids],
            "dispatch_metadata": {key: value for key, value in self.dispatch_metadata},
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DispatchBinding:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "template_fingerprint",
                "request_ids",
                "request_slots",
                "prefix_state_ids",
                "kv_read_handles",
                "kv_write_handles",
                "required_output_rows",
                "candidate_token_ids",
                "dispatch_metadata",
            },
            field="DispatchBinding",
        )
        request_slots = tuple(
            _strict_int(item, field=f"DispatchBinding.request_slots[{index}]")
            for index, item in enumerate(
                _strict_list(payload["request_slots"], field="DispatchBinding.request_slots")
            )
        )
        output_rows = tuple(
            _strict_int(item, field=f"DispatchBinding.required_output_rows[{index}]")
            for index, item in enumerate(
                _strict_list(
                    payload["required_output_rows"],
                    field="DispatchBinding.required_output_rows",
                )
            )
        )
        raw_candidate_rows = _strict_list(
            payload["candidate_token_ids"], field="DispatchBinding.candidate_token_ids"
        )
        dispatch_metadata = payload["dispatch_metadata"]
        if type(dispatch_metadata) is not dict:
            raise TypeError("DispatchBinding.dispatch_metadata must be an object")
        candidate_rows: list[tuple[int, ...]] = []
        for row_index, raw_row in enumerate(raw_candidate_rows):
            row = _strict_list(
                raw_row,
                field=f"DispatchBinding.candidate_token_ids[{row_index}]",
            )
            candidate_rows.append(
                tuple(
                    _strict_int(
                        token,
                        field=f"DispatchBinding.candidate_token_ids[{row_index}][{token_index}]",
                    )
                    for token_index, token in enumerate(row)
                )
            )
        return cls(
            template_fingerprint=_strict_string(
                payload["template_fingerprint"], field="DispatchBinding.template_fingerprint"
            ),
            request_ids=_strict_string_list(
                payload["request_ids"], field="DispatchBinding.request_ids"
            ),
            request_slots=request_slots,
            prefix_state_ids=_strict_string_list(
                payload["prefix_state_ids"], field="DispatchBinding.prefix_state_ids"
            ),
            kv_read_handles=_strict_string_list(
                payload["kv_read_handles"], field="DispatchBinding.kv_read_handles"
            ),
            kv_write_handles=_strict_string_list(
                payload["kv_write_handles"], field="DispatchBinding.kv_write_handles"
            ),
            required_output_rows=output_rows,
            candidate_token_ids=tuple(candidate_rows),
            dispatch_metadata=_dispatch_metadata_items(dispatch_metadata),
            schema_version=_strict_string(
                payload["schema_version"], field="DispatchBinding.schema_version"
            ),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> DispatchBinding:
        return cls.from_dict(_strict_json_loads(payload, field="DispatchBinding"))


def decompose_work_plan(plan: DenseWorkPlan) -> tuple[WorkTemplate, DispatchBinding]:
    """Split one concrete WorkPlan into an exact reusable template and attachment."""

    template = WorkTemplate.from_plan(plan)
    binding = DispatchBinding.from_plan(plan, template=template)
    if template.bind(binding).fingerprint != plan.fingerprint:
        raise RuntimeError("WorkPlan template/binding decomposition is not lossless")
    return template, binding


def build_dense_work_plan(
    *,
    model_name: str,
    model_revision: str,
    store_fingerprint: str,
    batch_size: int,
    sequence_length: int,
    batch_bucket: int | None = None,
    sequence_bucket: int | None = None,
    execution_mode: ExecutionMode | str = ExecutionMode.SCORE,
    output_contract: OutputContract | str = OutputContract.LAST_TOKEN_LOGITS,
    numerical_contract: str = "torch-batched-established",
    activation_dtype: str = "bf16",
    weight_dtype: str = "int8",
    accumulator_dtype: str = "fp32",
    request_ids: Sequence[str] | None = None,
    request_slots: Sequence[int] | None = None,
    prefix_state_ids: Sequence[str] = (),
    kv_read_handles: Sequence[str] = (),
    kv_write_handles: Sequence[str] = (),
    required_output_rows: Sequence[int] = (),
    candidate_token_ids: Sequence[Sequence[int]] = (),
    page_sequence: Sequence[str] = (),
    cache_admission: str = "default",
    compute_layout_ids: Sequence[str] = (),
    structured_operator_ids: Sequence[str] = (),
    capture: CaptureContract | None = None,
    capture_specs: Sequence[TypedCaptureSpec] = (),
    capture_capability: CaptureCapability | None = None,
    metadata: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None = None,
) -> DenseWorkPlan:
    """Build and validate a dense plan from explicit, serialization-safe fields."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be an integer")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, Integral):
        raise TypeError("sequence_length must be an integer")
    batch_size = int(batch_size)
    sequence_length = int(sequence_length)
    request_ids = (
        tuple(f"request-{index}" for index in range(batch_size))
        if request_ids is None
        else request_ids
    )
    request_slots = tuple(range(batch_size)) if request_slots is None else request_slots
    return DenseWorkPlan(
        model_name=model_name,
        model_revision=model_revision,
        store_fingerprint=store_fingerprint,
        execution_mode=execution_mode,  # type: ignore[arg-type]
        precision=PrecisionPolicy(
            activation_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            accumulator_dtype=accumulator_dtype,
        ),
        shape=ShapeBucket(
            actual_batch=batch_size,
            batch_bucket=batch_bucket or batch_size,
            sequence_length=sequence_length,
            sequence_bucket=sequence_bucket or sequence_length,
        ),
        output_contract=output_contract,  # type: ignore[arg-type]
        numerical_contract=numerical_contract,
        request_ids=_strict_string_tuple(request_ids, field="request_ids"),
        request_slots=_strict_int_tuple(request_slots, field="request_slots"),
        prefix_state_ids=_strict_string_tuple(prefix_state_ids, field="prefix_state_ids"),
        kv_read_handles=_strict_string_tuple(kv_read_handles, field="kv_read_handles"),
        kv_write_handles=_strict_string_tuple(kv_write_handles, field="kv_write_handles"),
        required_output_rows=_strict_int_tuple(required_output_rows, field="required_output_rows"),
        candidate_token_ids=_strict_nested_int_tuple(
            candidate_token_ids, field="candidate_token_ids"
        ),
        page_sequence=_strict_string_tuple(page_sequence, field="page_sequence"),
        cache_admission=cache_admission,
        compute_layout_ids=_strict_string_tuple(compute_layout_ids, field="compute_layout_ids"),
        structured_operator_ids=_strict_string_tuple(
            structured_operator_ids, field="structured_operator_ids"
        ),
        capture=capture or CaptureContract(),
        capture_specs=tuple(capture_specs),
        capture_capability=capture_capability,
        metadata=_metadata_items(metadata),
    )
