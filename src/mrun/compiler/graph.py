"""Typed, immutable operation graphs for WorkPlan compilation.

The graph is a semantic layer below :class:`~mrun.compiler.ir.DenseWorkPlan`.
Runtime tensors and request handles do not belong here: an ``OpGraph`` describes
the statically compilable dataflow, parameter accesses, and observable effects.

This module deliberately does not execute graphs.  Backends may fuse, place, and
schedule nodes, but the original graph remains available as the auditable semantic
contract against which those transformations are checked.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from math import prod
from typing import Any, TypeAlias

from .identity import manifest_declaration_matches
from .ir import DenseWorkPlan, ExecutionMode, OutputContract

OPGRAPH_SCHEMA = "mrun-opgraph-v1"

JsonScalar: TypeAlias = str | int | float | bool | None
FrozenJson: TypeAlias = JsonScalar | tuple["FrozenJson", ...] | tuple[tuple[str, "FrozenJson"], ...]

_PARAMETER_ACCESSES = {"all", "rows", "runtime_rows"}
_REGION_CHANNELS = {"w", "s", "e"}


class StorageClass(str, Enum):
    """Semantic ownership of a graph value."""

    INPUT = "input"
    ACTIVATION = "activation"
    PARAMETER = "parameter"
    STATE = "state"
    OUTPUT = "output"


class EffectKind(str, Enum):
    """Observable behavior that prevents unconstrained node elimination/reordering."""

    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    CAPTURE = "capture"
    BARRIER = "barrier"


class OpKind(str, Enum):
    """Operation kinds in the semantic graph.

    Kinds are intentionally backend-neutral.  A backend lowering may represent a
    connected set of these operations as one fused kernel or compiled region.
    """

    IDENTITY = "identity"
    EMBEDDING = "embedding"
    RMS_NORM = "rms_norm"
    LINEAR = "linear"
    BIAS_ADD = "bias_add"
    RESHAPE_HEADS = "reshape_heads"
    ROPE = "rope"
    GQA_REPEAT = "gqa_repeat"
    ATTENTION_SCORES = "attention_scores"
    CAUSAL_SOFTMAX = "causal_softmax"
    ATTENTION_VALUES = "attention_values"
    MERGE_HEADS = "merge_heads"
    RESIDUAL_ADD = "residual_add"
    SILU = "silu"
    MULTIPLY = "multiply"
    TAKE_LAST = "take_last"
    SELECT_ROWS = "select_rows"
    VOCAB_PROJECTION = "vocab_projection"
    TOPK_MARGIN = "topk_margin"
    CROSS_ENTROPY = "cross_entropy"
    GENERIC_PARAMETER_STREAM = "generic_parameter_stream"


def _coerce_enum(value: str | Enum, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}") from exc


def _require_name(value: Any, field_name: str) -> str:
    result = str(value)
    if not result or result.strip() != result:
        raise ValueError(f"{field_name} must be a non-empty, surrounding-whitespace-free string")
    return result


def _freeze_json(value: Any, *, path: str) -> FrozenJson:
    """Convert JSON-compatible data to a canonical immutable representation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} cannot contain non-finite floats")
        return value
    if isinstance(value, Enum):
        return _freeze_json(value.value, path=path)
    if isinstance(value, Mapping):
        items: list[tuple[str, FrozenJson]] = []
        seen: set[str] = set()
        for raw_key, raw_value in value.items():
            key = _require_name(raw_key, f"{path} key")
            if key in seen:
                raise ValueError(f"{path} contains duplicate key {key!r}")
            seen.add(key)
            items.append((key, _freeze_json(raw_value, path=f"{path}.{key}")))
        return tuple(sorted(items))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible values, got {type(value).__name__}")


def _looks_like_frozen_mapping(value: Any) -> bool:
    return isinstance(value, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value
    )


def freeze_json_mapping(
    value: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
    *,
    field_name: str = "attributes",
) -> tuple[tuple[str, FrozenJson], ...]:
    """Canonicalize a mapping for immutable IR fields.

    This helper is public so graph passes can construct nodes without depending on
    private normalization details.
    """

    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    normalized: list[tuple[str, FrozenJson]] = []
    seen: set[str] = set()
    for raw_key, raw_value in items:
        key = _require_name(raw_key, f"{field_name} key")
        if key in seen:
            raise ValueError(f"{field_name} contains duplicate key {key!r}")
        seen.add(key)
        normalized.append((key, _freeze_json(raw_value, path=f"{field_name}.{key}")))
    return tuple(sorted(normalized))


def _thaw_json(value: FrozenJson) -> Any:
    if _looks_like_frozen_mapping(value):
        return {key: _thaw_json(item) for key, item in value}  # type: ignore[misc]
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def thaw_json_mapping(value: Sequence[tuple[str, FrozenJson]]) -> dict[str, Any]:
    """Return the ordinary JSON representation of frozen attributes."""

    return {key: _thaw_json(item) for key, item in value}


def _shape(value: Sequence[int], field_name: str) -> tuple[int, ...]:
    result = tuple(int(dimension) for dimension in value)
    if any(dimension <= 0 for dimension in result):
        raise ValueError(f"{field_name} dimensions must be positive")
    return result


@dataclass(frozen=True)
class TensorSpec:
    """Static tensor value in SSA form."""

    value_id: str
    shape: tuple[int, ...]
    dtype: str
    layout: str
    storage_class: StorageClass
    memory_space: str = "logical"
    alias_of: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_id", _require_name(self.value_id, "value_id"))
        object.__setattr__(self, "shape", _shape(self.shape, f"{self.value_id}.shape"))
        object.__setattr__(self, "dtype", _require_name(self.dtype, f"{self.value_id}.dtype"))
        object.__setattr__(self, "layout", _require_name(self.layout, f"{self.value_id}.layout"))
        object.__setattr__(
            self,
            "storage_class",
            _coerce_enum(self.storage_class, StorageClass, "storage_class"),
        )
        object.__setattr__(
            self,
            "memory_space",
            _require_name(self.memory_space, f"{self.value_id}.memory_space"),
        )
        if self.alias_of is not None:
            object.__setattr__(
                self,
                "alias_of",
                _require_name(self.alias_of, f"{self.value_id}.alias_of"),
            )
            if self.alias_of == self.value_id:
                raise ValueError(f"{self.value_id!r} cannot alias itself")

    @property
    def element_count(self) -> int:
        return prod(self.shape)

    def as_dict(self) -> dict[str, Any]:
        return {
            "value_id": self.value_id,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "storage_class": self.storage_class.value,
            "memory_space": self.memory_space,
            "alias_of": self.alias_of,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TensorSpec:
        shape = payload.get("shape")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
            raise TypeError("tensor shape must be an array")
        return cls(
            value_id=str(payload["value_id"]),
            shape=tuple(int(value) for value in shape),
            dtype=str(payload["dtype"]),
            layout=str(payload["layout"]),
            storage_class=str(payload["storage_class"]),  # type: ignore[arg-type]
            memory_space=str(payload.get("memory_space", "logical")),
            alias_of=(None if payload.get("alias_of") is None else str(payload["alias_of"])),
        )


@dataclass(frozen=True)
class ParameterRef:
    """Logical parameter access with resolved physical storage identity.

    ``logical_name`` preserves model semantics while ``physical_name`` and
    ``regions`` make aliases (for example tied ``lm_head``/``embed`` weights)
    explicit for storage accounting.
    """

    logical_name: str
    physical_name: str
    kind: str
    shape: tuple[int, ...]
    regions: tuple[tuple[str, int, int], ...]
    access: str = "all"
    row_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "logical_name",
            _require_name(self.logical_name, "parameter logical_name"),
        )
        object.__setattr__(
            self,
            "physical_name",
            _require_name(self.physical_name, "parameter physical_name"),
        )
        object.__setattr__(self, "kind", _require_name(self.kind, "parameter kind"))
        object.__setattr__(
            self,
            "shape",
            _shape(self.shape, f"parameter {self.logical_name}.shape"),
        )
        access = str(self.access)
        if access not in _PARAMETER_ACCESSES:
            raise ValueError(
                f"parameter access must be one of {sorted(_PARAMETER_ACCESSES)}, got {access!r}"
            )
        object.__setattr__(self, "access", access)
        rows = tuple(int(value) for value in self.row_indices)
        object.__setattr__(self, "row_indices", rows)
        if access == "rows":
            if not rows:
                raise ValueError("row-selected parameter access requires row_indices")
            if len(rows) != len(set(rows)):
                raise ValueError("row-selected parameter indices must be unique")
            if any(row < 0 or row >= self.shape[0] for row in rows):
                raise ValueError(
                    f"row-selected indices for {self.logical_name!r} exceed shape {self.shape}"
                )
        elif rows:
            raise ValueError(f"parameter access {access!r} cannot carry row_indices")

        normalized_regions: list[tuple[str, int, int]] = []
        for raw_region in self.regions:
            if len(raw_region) != 3:
                raise ValueError("parameter regions must be (channel, offset, length) triples")
            channel, raw_offset, raw_length = raw_region
            channel = str(channel)
            offset = int(raw_offset)
            length = int(raw_length)
            if channel not in _REGION_CHANNELS:
                raise ValueError(f"unknown parameter region channel: {channel!r}")
            if offset < 0 or length <= 0:
                raise ValueError(
                    "parameter region offsets must be non-negative and lengths positive"
                )
            normalized_regions.append((channel, offset, length))
        if not normalized_regions:
            raise ValueError(f"parameter {self.logical_name!r} has no physical storage region")
        if len(normalized_regions) != len(set(normalized_regions)):
            raise ValueError(f"parameter {self.logical_name!r} has duplicate storage regions")
        object.__setattr__(
            self,
            "regions",
            tuple(sorted(normalized_regions)),
        )

    @property
    def is_alias(self) -> bool:
        return self.logical_name != self.physical_name

    @property
    def physical_bytes(self) -> int:
        return sum(length for _, _, length in self.regions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "physical_name": self.physical_name,
            "kind": self.kind,
            "shape": list(self.shape),
            "regions": [
                {"channel": channel, "offset": offset, "length": length}
                for channel, offset, length in self.regions
            ],
            "access": self.access,
            "row_indices": list(self.row_indices),
            "is_alias": self.is_alias,
            "physical_bytes": self.physical_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ParameterRef:
        shape = payload.get("shape")
        raw_regions = payload.get("regions")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
            raise TypeError("parameter shape must be an array")
        if not isinstance(raw_regions, Sequence) or isinstance(
            raw_regions, (str, bytes, bytearray)
        ):
            raise TypeError("parameter regions must be an array")
        regions: list[tuple[str, int, int]] = []
        for region in raw_regions:
            if isinstance(region, Mapping):
                regions.append(
                    (
                        str(region["channel"]),
                        int(region["offset"]),
                        int(region["length"]),
                    )
                )
            elif isinstance(region, Sequence) and not isinstance(region, (str, bytes, bytearray)):
                if len(region) != 3:
                    raise ValueError("parameter region arrays must have three entries")
                regions.append((str(region[0]), int(region[1]), int(region[2])))
            else:
                raise TypeError("parameter regions must contain objects or triples")
        return cls(
            logical_name=str(payload["logical_name"]),
            physical_name=str(payload["physical_name"]),
            kind=str(payload["kind"]),
            shape=tuple(int(value) for value in shape),
            regions=tuple(regions),
            access=str(payload.get("access", "all")),
            row_indices=tuple(int(value) for value in payload.get("row_indices", ())),
        )


@dataclass(frozen=True)
class Effect:
    """Versioned observable resource access."""

    kind: EffectKind
    resource_id: str
    version: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(self.kind, EffectKind, "effect kind"))
        object.__setattr__(
            self,
            "resource_id",
            _require_name(self.resource_id, "effect resource_id"),
        )
        if self.version is not None and int(self.version) < 0:
            raise ValueError("effect versions must be non-negative")
        if self.version is not None:
            object.__setattr__(self, "version", int(self.version))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "resource_id": self.resource_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Effect:
        return cls(
            kind=str(payload["kind"]),  # type: ignore[arg-type]
            resource_id=str(payload["resource_id"]),
            version=None if payload.get("version") is None else int(payload["version"]),
        )


@dataclass(frozen=True)
class OpNode:
    """One backend-neutral operation in topological SSA order."""

    node_id: str
    kind: OpKind
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    parameters: tuple[ParameterRef, ...] = ()
    params: tuple[tuple[str, FrozenJson], ...] = ()
    effects: tuple[Effect, ...] = ()
    control_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _require_name(self.node_id, "node_id"))
        object.__setattr__(self, "kind", _coerce_enum(self.kind, OpKind, "operation kind"))
        for field_name in ("inputs", "outputs", "control_inputs"):
            normalized = tuple(
                _require_name(value, f"{self.node_id}.{field_name}")
                for value in getattr(self, field_name)
            )
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{self.node_id}.{field_name} must be unique")
            object.__setattr__(self, field_name, normalized)
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(
            self,
            "params",
            freeze_json_mapping(self.params, field_name=f"{self.node_id}.params"),
        )
        object.__setattr__(self, "effects", tuple(self.effects))
        if not self.outputs and not self.effects:
            raise ValueError(f"operation {self.node_id!r} must produce a value or an effect")
        logical_parameters = [parameter.logical_name for parameter in self.parameters]
        if len(logical_parameters) != len(set(logical_parameters)):
            raise ValueError(f"operation {self.node_id!r} repeats a logical parameter")
        effect_keys = [(effect.kind, effect.resource_id, effect.version) for effect in self.effects]
        if len(effect_keys) != len(set(effect_keys)):
            raise ValueError(f"operation {self.node_id!r} repeats an effect")

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "parameters": [parameter.as_dict() for parameter in self.parameters],
            "params": thaw_json_mapping(self.params),
            "effects": [effect.as_dict() for effect in self.effects],
            "control_inputs": list(self.control_inputs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OpNode:
        raw_parameters = payload.get("parameters", ())
        raw_effects = payload.get("effects", ())
        params = payload.get("params", {})
        if not isinstance(raw_parameters, Sequence) or isinstance(
            raw_parameters, (str, bytes, bytearray)
        ):
            raise TypeError("node parameters must be an array")
        if not isinstance(raw_effects, Sequence) or isinstance(
            raw_effects, (str, bytes, bytearray)
        ):
            raise TypeError("node effects must be an array")
        if not isinstance(params, Mapping):
            raise TypeError("node params must be an object")
        return cls(
            node_id=str(payload["node_id"]),
            kind=str(payload["kind"]),  # type: ignore[arg-type]
            inputs=tuple(str(value) for value in payload.get("inputs", ())),
            outputs=tuple(str(value) for value in payload.get("outputs", ())),
            parameters=tuple(
                ParameterRef.from_dict(value)
                if isinstance(value, Mapping)
                else _raise_type("node parameters must contain objects")
                for value in raw_parameters
            ),
            params=freeze_json_mapping(params),
            effects=tuple(
                Effect.from_dict(value)
                if isinstance(value, Mapping)
                else _raise_type("node effects must contain objects")
                for value in raw_effects
            ),
            control_inputs=tuple(str(value) for value in payload.get("control_inputs", ())),
        )


@dataclass(frozen=True)
class OpGraph:
    """Validated semantic dataflow graph.

    Nodes are serialized in topological order.  Every non-input tensor has exactly
    one producer and every node may consume only graph inputs or values produced by
    earlier nodes.  These constraints make cycles, dangling references, and
    accidental reassignment fail at construction time.
    """

    model_name: str
    model_revision: str
    store_fingerprint: str
    architecture: str
    numerical_contract: str
    values: tuple[TensorSpec, ...]
    nodes: tuple[OpNode, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    metadata: tuple[tuple[str, FrozenJson], ...] = ()
    schema_version: str = OPGRAPH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != OPGRAPH_SCHEMA:
            raise ValueError(f"unsupported OpGraph schema: {self.schema_version}")
        for field_name in (
            "model_name",
            "model_revision",
            "store_fingerprint",
            "architecture",
            "numerical_contract",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(
            self,
            "inputs",
            tuple(_require_name(value, "graph input") for value in self.inputs),
        )
        object.__setattr__(
            self,
            "outputs",
            tuple(_require_name(value, "graph output") for value in self.outputs),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field_name="graph metadata"),
        )
        self._validate()

    def _validate(self) -> None:
        value_ids = [value.value_id for value in self.values]
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("graph tensor value IDs must be unique")
        values = {value.value_id: value for value in self.values}
        if not self.inputs:
            raise ValueError("graph must have at least one input")
        if not self.outputs:
            raise ValueError("graph must have at least one output")
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError("graph inputs must be unique")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError("graph outputs must be unique")
        missing_inputs = sorted(set(self.inputs) - values.keys())
        missing_outputs = sorted(set(self.outputs) - values.keys())
        if missing_inputs:
            raise ValueError(f"graph inputs have no TensorSpec: {missing_inputs}")
        if missing_outputs:
            raise ValueError(f"graph outputs have no TensorSpec: {missing_outputs}")

        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node IDs must be unique")
        graph_inputs = set(self.inputs)
        produced: dict[str, str] = {}
        seen_nodes: set[str] = set()
        physical_parameters: dict[
            str, tuple[str, tuple[int, ...], tuple[tuple[str, int, int], ...]]
        ] = {}
        for node in self.nodes:
            unavailable_controls = [
                dependency for dependency in node.control_inputs if dependency not in seen_nodes
            ]
            if unavailable_controls:
                raise ValueError(
                    f"node {node.node_id!r} has non-topological control dependencies "
                    f"{unavailable_controls}"
                )
            for value_id in node.inputs:
                if value_id not in values:
                    raise ValueError(f"node {node.node_id!r} consumes unknown value {value_id!r}")
                if value_id not in graph_inputs and value_id not in produced:
                    raise ValueError(
                        f"node {node.node_id!r} consumes {value_id!r} before its producer"
                    )
            for value_id in node.outputs:
                if value_id not in values:
                    raise ValueError(f"node {node.node_id!r} produces unknown value {value_id!r}")
                if value_id in graph_inputs:
                    raise ValueError(
                        f"node {node.node_id!r} cannot overwrite graph input {value_id!r}"
                    )
                previous = produced.get(value_id)
                if previous is not None:
                    raise ValueError(
                        f"SSA violation: {value_id!r} is produced by {previous!r} "
                        f"and {node.node_id!r}"
                    )
                produced[value_id] = node.node_id
            for parameter in node.parameters:
                physical_contract = (
                    parameter.kind,
                    parameter.shape,
                    parameter.regions,
                )
                existing = physical_parameters.setdefault(
                    parameter.physical_name,
                    physical_contract,
                )
                if existing != physical_contract:
                    raise ValueError(
                        f"physical parameter {parameter.physical_name!r} has inconsistent "
                        "kind, shape, or storage regions"
                    )
            seen_nodes.add(node.node_id)

        orphan_values = sorted(set(values) - graph_inputs - produced.keys())
        if orphan_values:
            raise ValueError(f"non-input graph values have no producer: {orphan_values}")
        unavailable_outputs = sorted(
            value_id
            for value_id in self.outputs
            if value_id not in graph_inputs and value_id not in produced
        )
        if unavailable_outputs:
            raise ValueError(f"graph outputs are unavailable: {unavailable_outputs}")

        for value in self.values:
            if value.alias_of is None:
                continue
            target = values.get(value.alias_of)
            if target is None:
                raise ValueError(
                    f"value {value.value_id!r} aliases unknown value {value.alias_of!r}"
                )
            if value.dtype != target.dtype:
                raise ValueError(f"alias {value.value_id!r} must preserve dtype {target.dtype!r}")
            if value.element_count != target.element_count:
                raise ValueError(
                    f"alias {value.value_id!r} must preserve element count {target.element_count}"
                )
            if value.memory_space != target.memory_space:
                raise ValueError(
                    f"alias {value.value_id!r} must stay in memory space {target.memory_space!r}"
                )
            chain: set[str] = {value.value_id}
            current = target
            while current.alias_of is not None:
                if current.value_id in chain:
                    raise ValueError(f"tensor alias cycle involving {current.value_id!r}")
                chain.add(current.value_id)
                next_target = values.get(current.alias_of)
                if next_target is None:
                    raise ValueError(
                        f"value {current.value_id!r} aliases unknown value {current.alias_of!r}"
                    )
                current = next_target

    @property
    def tensor_map(self) -> dict[str, TensorSpec]:
        return {value.value_id: value for value in self.values}

    @property
    def node_map(self) -> dict[str, OpNode]:
        return {node.node_id: node for node in self.nodes}

    @property
    def producer_map(self) -> dict[str, str]:
        return {value_id: node.node_id for node in self.nodes for value_id in node.outputs}

    @property
    def consumer_map(self) -> dict[str, tuple[str, ...]]:
        consumers: dict[str, list[str]] = {value.value_id: [] for value in self.values}
        for node in self.nodes:
            for value_id in node.inputs:
                consumers[value_id].append(node.node_id)
        return {value_id: tuple(node_ids) for value_id, node_ids in consumers.items()}

    @property
    def parameter_refs(self) -> tuple[ParameterRef, ...]:
        return tuple(parameter for node in self.nodes for parameter in node.parameters)

    @cached_property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "store_fingerprint": self.store_fingerprint,
            "architecture": self.architecture,
            "numerical_contract": self.numerical_contract,
            "values": [value.as_dict() for value in self.values],
            "nodes": [node.as_dict() for node in self.nodes],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "metadata": thaw_json_mapping(self.metadata),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OpGraph:
        raw_values = payload.get("values")
        raw_nodes = payload.get("nodes")
        metadata = payload.get("metadata", {})
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes, bytearray)):
            raise TypeError("graph values must be an array")
        if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes, bytearray)):
            raise TypeError("graph nodes must be an array")
        if not isinstance(metadata, Mapping):
            raise TypeError("graph metadata must be an object")
        graph = cls(
            model_name=str(payload["model_name"]),
            model_revision=str(payload["model_revision"]),
            store_fingerprint=str(payload["store_fingerprint"]),
            architecture=str(payload["architecture"]),
            numerical_contract=str(payload["numerical_contract"]),
            values=tuple(
                TensorSpec.from_dict(value)
                if isinstance(value, Mapping)
                else _raise_type("graph values must contain objects")
                for value in raw_values
            ),
            nodes=tuple(
                OpNode.from_dict(node)
                if isinstance(node, Mapping)
                else _raise_type("graph nodes must contain objects")
                for node in raw_nodes
            ),
            inputs=tuple(str(value) for value in payload.get("inputs", ())),
            outputs=tuple(str(value) for value in payload.get("outputs", ())),
            metadata=freeze_json_mapping(metadata),
            schema_version=str(payload.get("schema_version", OPGRAPH_SCHEMA)),
        )
        claimed_fingerprint = payload.get("graph_fingerprint")
        if claimed_fingerprint is not None and str(claimed_fingerprint) != graph.fingerprint:
            raise ValueError("OpGraph fingerprint mismatch")
        return graph

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> OpGraph:
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise TypeError("serialized OpGraph must be a JSON object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class DemandRewriteCertificate:
    """Serializable proof record for output-driven graph specialization.

    This is a structural certificate, not a proof of floating-point equivalence.
    ``rewrite_ids`` name the algebraic rules used; numerical parity remains an
    independent promotion requirement.
    """

    source_fingerprint: str
    rewritten_fingerprint: str
    output_contract: str
    rewrite_ids: tuple[str, ...]
    removed_node_ids: tuple[str, ...]
    added_node_ids: tuple[str, ...]
    claims: tuple[str, ...]
    schema_version: str = "mrun-output-demand-certificate-v1"

    def __post_init__(self) -> None:
        for field_name in (
            "source_fingerprint",
            "rewritten_fingerprint",
            "output_contract",
            "schema_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        for field_name in (
            "rewrite_ids",
            "removed_node_ids",
            "added_node_ids",
            "claims",
        ):
            values = tuple(
                _require_name(value, f"certificate {field_name}")
                for value in getattr(self, field_name)
            )
            if len(values) != len(set(values)):
                raise ValueError(f"certificate {field_name} must be unique")
            object.__setattr__(self, field_name, values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "rewritten_fingerprint": self.rewritten_fingerprint,
            "output_contract": self.output_contract,
            "rewrite_ids": list(self.rewrite_ids),
            "removed_node_ids": list(self.removed_node_ids),
            "added_node_ids": list(self.added_node_ids),
            "claims": list(self.claims),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DemandRewriteCertificate:
        return cls(
            source_fingerprint=str(payload["source_fingerprint"]),
            rewritten_fingerprint=str(payload["rewritten_fingerprint"]),
            output_contract=str(payload["output_contract"]),
            rewrite_ids=tuple(str(value) for value in payload.get("rewrite_ids", ())),
            removed_node_ids=tuple(str(value) for value in payload.get("removed_node_ids", ())),
            added_node_ids=tuple(str(value) for value in payload.get("added_node_ids", ())),
            claims=tuple(str(value) for value in payload.get("claims", ())),
            schema_version=str(
                payload.get(
                    "schema_version",
                    "mrun-output-demand-certificate-v1",
                )
            ),
        )


def _raise_type(message: str) -> Any:
    raise TypeError(message)


def _manifest_blocks(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    blocks = manifest.get("blocks")
    if not isinstance(blocks, Mapping):
        raise TypeError("manifest blocks must be an object")
    return blocks


def _manifest_parameter(
    manifest: Mapping[str, Any],
    logical_name: str,
    *,
    access: str = "all",
    row_indices: Sequence[int] = (),
) -> ParameterRef:
    blocks = _manifest_blocks(manifest)
    current = logical_name
    seen: set[str] = set()
    while True:
        if current in seen:
            raise ValueError(f"manifest parameter alias cycle at {current!r}")
        seen.add(current)
        raw = blocks.get(current)
        if not isinstance(raw, Mapping):
            raise KeyError(f"manifest has no parameter {current!r}")
        alias = raw.get("alias")
        if alias is None:
            physical_name = current
            physical = raw
            break
        current = _require_name(alias, f"manifest alias for {current}")

    raw_shape = physical.get("shape")
    if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes, bytearray)):
        raise TypeError(f"manifest parameter {physical_name!r} has no array shape")
    shape = tuple(int(value) for value in raw_shape)
    regions_list: list[tuple[str, int, int]] = []
    for channel in ("w", "s", "e"):
        length = int(physical.get(f"{channel}_len", 0))
        if (
            not length
            and channel == "w"
            and str(physical.get("kind", "")).startswith("qrow")
            and physical.get("row_bytes") is not None
        ):
            length = shape[0] * int(physical["row_bytes"])
        elif (
            not length
            and channel == "s"
            and str(physical.get("kind", "")).startswith("qrow")
            and physical.get("n_groups") is not None
        ):
            length = shape[0] * int(physical["n_groups"]) * 4
        if length > 0:
            regions_list.append(
                (
                    channel,
                    int(physical.get(f"{channel}_off", 0)),
                    length,
                )
            )
    return ParameterRef(
        logical_name=logical_name,
        physical_name=physical_name,
        kind=str(physical.get("kind", "unknown")),
        shape=shape,
        regions=tuple(regions_list),
        access=access,
        row_indices=tuple(int(value) for value in row_indices),
    )


def _manifest_has(manifest: Mapping[str, Any], logical_name: str) -> bool:
    return logical_name in _manifest_blocks(manifest)


class _GraphBuilder:
    def __init__(self) -> None:
        self.values: list[TensorSpec] = []
        self.nodes: list[OpNode] = []
        self.inputs: list[str] = []

    def add_input(
        self,
        value_id: str,
        shape: Sequence[int],
        dtype: str,
        layout: str,
    ) -> str:
        self.values.append(
            TensorSpec(
                value_id=value_id,
                shape=tuple(shape),
                dtype=dtype,
                layout=layout,
                storage_class=StorageClass.INPUT,
            )
        )
        self.inputs.append(value_id)
        return value_id

    def add_node(
        self,
        *,
        node_id: str,
        kind: OpKind,
        inputs: Sequence[str],
        output_specs: Sequence[TensorSpec],
        parameters: Sequence[ParameterRef] = (),
        params: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        effects: Sequence[Effect] = (),
        control_inputs: Sequence[str] = (),
    ) -> tuple[str, ...]:
        self.values.extend(output_specs)
        outputs = tuple(spec.value_id for spec in output_specs)
        self.nodes.append(
            OpNode(
                node_id=node_id,
                kind=kind,
                inputs=tuple(inputs),
                outputs=outputs,
                parameters=tuple(parameters),
                params=freeze_json_mapping(params, field_name=f"{node_id}.params"),
                effects=tuple(effects),
                control_inputs=tuple(control_inputs),
            )
        )
        return outputs


def _activation(
    value_id: str,
    shape: Sequence[int],
    dtype: str,
    layout: str,
    *,
    storage_class: StorageClass = StorageClass.ACTIVATION,
    alias_of: str | None = None,
) -> TensorSpec:
    return TensorSpec(
        value_id=value_id,
        shape=tuple(shape),
        dtype=dtype,
        layout=layout,
        storage_class=storage_class,
        alias_of=alias_of,
    )


def _positive_config(config: Mapping[str, Any], key: str) -> int:
    value = int(config.get(key, 0))
    if value <= 0:
        raise ValueError(f"manifest config {key!r} must be positive")
    return value


def _validate_matrix(
    parameter: ParameterRef,
    *,
    expected_out: int,
    expected_in: int,
) -> None:
    expected = (expected_out, expected_in)
    if parameter.shape != expected:
        raise ValueError(
            f"parameter {parameter.logical_name!r} has shape {parameter.shape}, expected {expected}"
        )


def _validate_vector(parameter: ParameterRef, expected: int) -> None:
    if parameter.shape != (expected,):
        raise ValueError(
            f"parameter {parameter.logical_name!r} has shape {parameter.shape}, "
            f"expected {(expected,)}"
        )


def _build_qwen_family_graph(
    plan: DenseWorkPlan,
    manifest: Mapping[str, Any],
    architecture: str,
) -> OpGraph:
    if plan.execution_mode is ExecutionMode.DECODE:
        raise NotImplementedError(
            "OpGraph decode requires explicit versioned KV state inputs/outputs"
        )
    if plan.output_contract is OutputContract.SELECTED_CAPTURE:
        raise NotImplementedError(
            "selected_capture requires a typed capture specification not present in WorkPlan v2"
        )
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("manifest config must be an object")
    hidden = _positive_config(config, "hidden_size")
    layers = _positive_config(config, "num_hidden_layers")
    heads = _positive_config(config, "num_attention_heads")
    kv_heads = _positive_config(config, "num_key_value_heads")
    head_dim = int(config.get("head_dim", hidden // heads))
    intermediate = _positive_config(config, "intermediate_size")
    vocab = _positive_config(config, "vocab_size")
    if head_dim <= 0 or heads * head_dim != hidden:
        raise ValueError("manifest attention heads/head_dim must exactly cover hidden_size")
    if heads % kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    batch = plan.shape.batch_bucket
    sequence = plan.shape.sequence_bucket
    activation_dtype = plan.precision.activation_dtype
    builder = _GraphBuilder()
    token_ids = builder.add_input(
        "input.token_ids",
        (batch, sequence),
        "int64",
        "BT",
    )

    embed = _manifest_parameter(manifest, "embed", access="runtime_rows")
    _validate_matrix(embed, expected_out=vocab, expected_in=hidden)
    (residual,) = builder.add_node(
        node_id="embed.lookup",
        kind=OpKind.EMBEDDING,
        inputs=(token_ids,),
        output_specs=(
            _activation("embed.hidden", (batch, sequence, hidden), activation_dtype, "BTH"),
        ),
        parameters=(embed,),
        params={"padding": "right", "vocabulary_size": vocab},
    )

    def add_norm(
        *,
        node_id: str,
        input_id: str,
        parameter_name: str,
        output_id: str,
        value_shape: tuple[int, ...],
        axis_size: int,
        layout: str,
    ) -> str:
        weight = _manifest_parameter(manifest, parameter_name)
        _validate_vector(weight, axis_size)
        (output,) = builder.add_node(
            node_id=node_id,
            kind=OpKind.RMS_NORM,
            inputs=(input_id,),
            output_specs=(_activation(output_id, value_shape, activation_dtype, layout),),
            parameters=(weight,),
            params={
                "axis": -1,
                "epsilon": float(config.get("rms_norm_eps", 1e-6)),
            },
        )
        return output

    def add_linear(
        *,
        layer: int,
        role: str,
        input_id: str,
        input_width: int,
        output_width: int,
        output_shape: tuple[int, ...],
        layout: str,
    ) -> str:
        prefix = f"layer.{layer}.{role}"
        parameter_name = f"L{layer}.{role}"
        weight = _manifest_parameter(manifest, parameter_name)
        _validate_matrix(weight, expected_out=output_width, expected_in=input_width)
        raw_output_id = f"{prefix}.linear"
        (output,) = builder.add_node(
            node_id=f"{prefix}.linear",
            kind=OpKind.LINEAR,
            inputs=(input_id,),
            output_specs=(_activation(raw_output_id, output_shape, activation_dtype, layout),),
            parameters=(weight,),
            params={
                "accumulator_dtype": plan.precision.accumulator_dtype,
                "role": role,
            },
        )
        bias_name = f"{parameter_name}.bias"
        if _manifest_has(manifest, bias_name):
            bias = _manifest_parameter(manifest, bias_name)
            _validate_vector(bias, output_width)
            biased_id = f"{prefix}.biased"
            (output,) = builder.add_node(
                node_id=f"{prefix}.bias",
                kind=OpKind.BIAS_ADD,
                inputs=(output,),
                output_specs=(_activation(biased_id, output_shape, activation_dtype, layout),),
                parameters=(bias,),
                params={"axis": -1, "role": role},
            )
        return output

    for layer in range(layers):
        layer_prefix = f"layer.{layer}"
        norm1 = add_norm(
            node_id=f"{layer_prefix}.attn_norm",
            input_id=residual,
            parameter_name=f"L{layer}.ln1",
            output_id=f"{layer_prefix}.attn_norm.out",
            value_shape=(batch, sequence, hidden),
            axis_size=hidden,
            layout="BTH",
        )
        q_width = heads * head_dim
        kv_width = kv_heads * head_dim
        q = add_linear(
            layer=layer,
            role="q",
            input_id=norm1,
            input_width=hidden,
            output_width=q_width,
            output_shape=(batch, sequence, q_width),
            layout="BTH",
        )
        k = add_linear(
            layer=layer,
            role="k",
            input_id=norm1,
            input_width=hidden,
            output_width=kv_width,
            output_shape=(batch, sequence, kv_width),
            layout="BTH",
        )
        v = add_linear(
            layer=layer,
            role="v",
            input_id=norm1,
            input_width=hidden,
            output_width=kv_width,
            output_shape=(batch, sequence, kv_width),
            layout="BTH",
        )

        q_heads_id = f"{layer_prefix}.q.heads"
        k_heads_id = f"{layer_prefix}.k.heads"
        v_heads_id = f"{layer_prefix}.v.heads"
        (q_heads,) = builder.add_node(
            node_id=f"{layer_prefix}.q.reshape",
            kind=OpKind.RESHAPE_HEADS,
            inputs=(q,),
            output_specs=(
                _activation(
                    q_heads_id,
                    (batch, heads, sequence, head_dim),
                    activation_dtype,
                    "BHSD",
                    alias_of=q,
                ),
            ),
            params={"heads": heads, "head_dim": head_dim},
        )
        (k_heads,) = builder.add_node(
            node_id=f"{layer_prefix}.k.reshape",
            kind=OpKind.RESHAPE_HEADS,
            inputs=(k,),
            output_specs=(
                _activation(
                    k_heads_id,
                    (batch, kv_heads, sequence, head_dim),
                    activation_dtype,
                    "BHSD",
                    alias_of=k,
                ),
            ),
            params={"heads": kv_heads, "head_dim": head_dim},
        )
        (v_heads,) = builder.add_node(
            node_id=f"{layer_prefix}.v.reshape",
            kind=OpKind.RESHAPE_HEADS,
            inputs=(v,),
            output_specs=(
                _activation(
                    v_heads_id,
                    (batch, kv_heads, sequence, head_dim),
                    activation_dtype,
                    "BHSD",
                    alias_of=v,
                ),
            ),
            params={"heads": kv_heads, "head_dim": head_dim},
        )

        q_norm_name = f"L{layer}.q_norm"
        k_norm_name = f"L{layer}.k_norm"
        has_q_norm = _manifest_has(manifest, q_norm_name)
        has_k_norm = _manifest_has(manifest, k_norm_name)
        if has_q_norm != has_k_norm:
            raise ValueError(f"layer {layer} must provide both q_norm and k_norm or neither")
        if has_q_norm:
            q_heads = add_norm(
                node_id=f"{layer_prefix}.q.head_norm",
                input_id=q_heads,
                parameter_name=q_norm_name,
                output_id=f"{layer_prefix}.q.head_norm.out",
                value_shape=(batch, heads, sequence, head_dim),
                axis_size=head_dim,
                layout="BHSD",
            )
            k_heads = add_norm(
                node_id=f"{layer_prefix}.k.head_norm",
                input_id=k_heads,
                parameter_name=k_norm_name,
                output_id=f"{layer_prefix}.k.head_norm.out",
                value_shape=(batch, kv_heads, sequence, head_dim),
                axis_size=head_dim,
                layout="BHSD",
            )

        (q_rope,) = builder.add_node(
            node_id=f"{layer_prefix}.q.rope",
            kind=OpKind.ROPE,
            inputs=(q_heads,),
            output_specs=(
                _activation(
                    f"{layer_prefix}.q.rope.out",
                    (batch, heads, sequence, head_dim),
                    activation_dtype,
                    "BHSD",
                ),
            ),
            params={
                "head_dim": head_dim,
                "theta": float(config.get("rope_theta", 10000.0)),
            },
        )
        (k_rope,) = builder.add_node(
            node_id=f"{layer_prefix}.k.rope",
            kind=OpKind.ROPE,
            inputs=(k_heads,),
            output_specs=(
                _activation(
                    f"{layer_prefix}.k.rope.out",
                    (batch, kv_heads, sequence, head_dim),
                    activation_dtype,
                    "BHSD",
                ),
            ),
            params={
                "head_dim": head_dim,
                "theta": float(config.get("rope_theta", 10000.0)),
            },
        )

        if kv_heads != heads:
            (k_rope,) = builder.add_node(
                node_id=f"{layer_prefix}.k.gqa_repeat",
                kind=OpKind.GQA_REPEAT,
                inputs=(k_rope,),
                output_specs=(
                    _activation(
                        f"{layer_prefix}.k.gqa_repeat.out",
                        (batch, heads, sequence, head_dim),
                        activation_dtype,
                        "BHSD",
                    ),
                ),
                params={"repeat": heads // kv_heads},
            )
            (v_heads,) = builder.add_node(
                node_id=f"{layer_prefix}.v.gqa_repeat",
                kind=OpKind.GQA_REPEAT,
                inputs=(v_heads,),
                output_specs=(
                    _activation(
                        f"{layer_prefix}.v.gqa_repeat.out",
                        (batch, heads, sequence, head_dim),
                        activation_dtype,
                        "BHSD",
                    ),
                ),
                params={"repeat": heads // kv_heads},
            )

        (scores,) = builder.add_node(
            node_id=f"{layer_prefix}.attention.scores",
            kind=OpKind.ATTENTION_SCORES,
            inputs=(q_rope, k_rope),
            output_specs=(
                _activation(
                    f"{layer_prefix}.attention.scores.out",
                    (batch, heads, sequence, sequence),
                    "fp32",
                    "BHSS",
                ),
            ),
            params={"scale": head_dim**-0.5},
        )
        (probabilities,) = builder.add_node(
            node_id=f"{layer_prefix}.attention.softmax",
            kind=OpKind.CAUSAL_SOFTMAX,
            inputs=(scores,),
            output_specs=(
                _activation(
                    f"{layer_prefix}.attention.softmax.out",
                    (batch, heads, sequence, sequence),
                    "fp32",
                    "BHSS",
                ),
            ),
            params={"causal": True, "right_padding": True},
        )
        (head_context,) = builder.add_node(
            node_id=f"{layer_prefix}.attention.values",
            kind=OpKind.ATTENTION_VALUES,
            inputs=(probabilities, v_heads),
            output_specs=(
                _activation(
                    f"{layer_prefix}.attention.values.out",
                    (batch, sequence, heads, head_dim),
                    activation_dtype,
                    "BSHD",
                ),
            ),
        )
        (merged_context,) = builder.add_node(
            node_id=f"{layer_prefix}.attention.merge_heads",
            kind=OpKind.MERGE_HEADS,
            inputs=(head_context,),
            output_specs=(
                _activation(
                    f"{layer_prefix}.attention.merged",
                    (batch, sequence, hidden),
                    activation_dtype,
                    "BTH",
                    alias_of=head_context,
                ),
            ),
            params={"hidden_size": hidden},
        )
        attention_output = add_linear(
            layer=layer,
            role="o",
            input_id=merged_context,
            input_width=hidden,
            output_width=hidden,
            output_shape=(batch, sequence, hidden),
            layout="BTH",
        )
        (attention_residual,) = builder.add_node(
            node_id=f"{layer_prefix}.attention.residual",
            kind=OpKind.RESIDUAL_ADD,
            inputs=(residual, attention_output),
            output_specs=(
                _activation(
                    f"{layer_prefix}.attention.residual.out",
                    (batch, sequence, hidden),
                    activation_dtype,
                    "BTH",
                ),
            ),
        )
        norm2 = add_norm(
            node_id=f"{layer_prefix}.mlp_norm",
            input_id=attention_residual,
            parameter_name=f"L{layer}.ln2",
            output_id=f"{layer_prefix}.mlp_norm.out",
            value_shape=(batch, sequence, hidden),
            axis_size=hidden,
            layout="BTH",
        )
        gate = add_linear(
            layer=layer,
            role="gate",
            input_id=norm2,
            input_width=hidden,
            output_width=intermediate,
            output_shape=(batch, sequence, intermediate),
            layout="BTI",
        )
        up = add_linear(
            layer=layer,
            role="up",
            input_id=norm2,
            input_width=hidden,
            output_width=intermediate,
            output_shape=(batch, sequence, intermediate),
            layout="BTI",
        )
        (activated_gate,) = builder.add_node(
            node_id=f"{layer_prefix}.mlp.silu",
            kind=OpKind.SILU,
            inputs=(gate,),
            output_specs=(
                _activation(
                    f"{layer_prefix}.mlp.silu.out",
                    (batch, sequence, intermediate),
                    activation_dtype,
                    "BTI",
                ),
            ),
        )
        (gated_hidden,) = builder.add_node(
            node_id=f"{layer_prefix}.mlp.multiply",
            kind=OpKind.MULTIPLY,
            inputs=(activated_gate, up),
            output_specs=(
                _activation(
                    f"{layer_prefix}.mlp.gated",
                    (batch, sequence, intermediate),
                    activation_dtype,
                    "BTI",
                ),
            ),
        )
        down = add_linear(
            layer=layer,
            role="down",
            input_id=gated_hidden,
            input_width=intermediate,
            output_width=hidden,
            output_shape=(batch, sequence, hidden),
            layout="BTH",
        )
        (residual,) = builder.add_node(
            node_id=f"{layer_prefix}.mlp.residual",
            kind=OpKind.RESIDUAL_ADD,
            inputs=(attention_residual, down),
            output_specs=(
                _activation(
                    f"{layer_prefix}.output",
                    (batch, sequence, hidden),
                    activation_dtype,
                    "BTH",
                ),
            ),
        )

    final_hidden = add_norm(
        node_id="final.norm",
        input_id=residual,
        parameter_name="norm.final",
        output_id="final.hidden",
        value_shape=(batch, sequence, hidden),
        axis_size=hidden,
        layout="BTH",
    )
    outputs: tuple[str, ...]
    if plan.output_contract is OutputContract.HIDDEN_STATE_ONLY:
        outputs = (final_hidden,)
    else:
        use_last = plan.output_contract in {
            OutputContract.LAST_TOKEN_LOGITS,
            OutputContract.SELECTED_TOKEN_ROWS,
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        }
        head_input = final_hidden
        head_batch_shape: tuple[int, ...] = (batch, sequence)
        if use_last:
            (head_input,) = builder.add_node(
                node_id="output.take_last_hidden",
                kind=OpKind.TAKE_LAST,
                inputs=(final_hidden,),
                output_specs=(
                    _activation(
                        "output.last_hidden",
                        (batch, hidden),
                        activation_dtype,
                        "BH",
                    ),
                ),
                params={"right_padding": True},
            )
            head_batch_shape = (batch,)

        head_rows: tuple[int, ...] = ()
        if plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
            head_rows = plan.required_output_rows
        elif plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
            head_rows = tuple(
                dict.fromkeys(
                    token for row_candidates in plan.candidate_token_ids for token in row_candidates
                )
            )
        access = "rows" if head_rows else "all"
        head = _manifest_parameter(
            manifest,
            "lm_head",
            access=access,
            row_indices=head_rows,
        )
        _validate_matrix(head, expected_out=vocab, expected_in=hidden)
        head_width = len(head_rows) if head_rows else vocab
        logits_id = (
            "output.candidate_union_logits"
            if plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
            else "output.logits"
        )
        logits_class = (
            StorageClass.ACTIVATION
            if plan.output_contract
            in {OutputContract.CANDIDATE_ARGMAX_AND_MARGIN, OutputContract.LOSS_ONLY}
            else StorageClass.OUTPUT
        )
        (logits,) = builder.add_node(
            node_id="output.vocab_projection",
            kind=OpKind.VOCAB_PROJECTION,
            inputs=(head_input,),
            output_specs=(
                _activation(
                    logits_id,
                    (*head_batch_shape, head_width),
                    "fp32",
                    "BTV" if len(head_batch_shape) == 2 else "BV",
                    storage_class=logits_class,
                ),
            ),
            parameters=(head,),
            params={
                "access": access,
                "accumulator_dtype": "fp32",
                "logical_parameter": "lm_head",
                "output_contract": plan.output_contract.value,
            },
        )
        if plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
            candidate_outputs = (
                _activation(
                    "output.winner_token_id",
                    (batch,),
                    "int64",
                    "B",
                    storage_class=StorageClass.OUTPUT,
                ),
                _activation(
                    "output.runner_up_token_id",
                    (batch,),
                    "int64",
                    "B",
                    storage_class=StorageClass.OUTPUT,
                ),
                _activation(
                    "output.winner_logit",
                    (batch,),
                    "fp32",
                    "B",
                    storage_class=StorageClass.OUTPUT,
                ),
                _activation(
                    "output.runner_up_logit",
                    (batch,),
                    "fp32",
                    "B",
                    storage_class=StorageClass.OUTPUT,
                ),
                _activation(
                    "output.margin",
                    (batch,),
                    "fp32",
                    "B",
                    storage_class=StorageClass.OUTPUT,
                ),
            )
            outputs = builder.add_node(
                node_id="output.candidate_topk_margin",
                kind=OpKind.TOPK_MARGIN,
                inputs=(logits,),
                output_specs=candidate_outputs,
                params={
                    "candidate_token_ids": plan.candidate_token_ids,
                    "union_token_ids": head_rows,
                    "live_batch": plan.shape.actual_batch,
                    "top_k": 2,
                },
            )
        elif plan.output_contract is OutputContract.LOSS_ONLY:
            labels = builder.add_input(
                "input.labels",
                (batch, sequence),
                "int64",
                "BT",
            )
            outputs = builder.add_node(
                node_id="output.cross_entropy",
                kind=OpKind.CROSS_ENTROPY,
                inputs=(logits, labels),
                output_specs=(
                    _activation(
                        "output.loss",
                        (),
                        "fp32",
                        "scalar",
                        storage_class=StorageClass.OUTPUT,
                    ),
                ),
                params={
                    "live_batch": plan.shape.actual_batch,
                    "live_sequence": plan.shape.sequence_length,
                    "reduction": "mean",
                    "shift": 1,
                },
            )
        else:
            outputs = (logits,)

    used_parameters = {
        parameter.logical_name for node in builder.nodes for parameter in node.parameters
    }
    auxiliary_pattern = re.compile(
        r"^L\d+\.(?:ln1|ln2|q_norm|k_norm|"
        r"(?:q|k|v|o|gate|up|down)\.bias)$"
    )
    expected_auxiliary = {
        str(name)
        for name in _manifest_blocks(manifest)
        if auxiliary_pattern.fullmatch(str(name)) or str(name) == "norm.final"
    }
    unmodeled_auxiliary = sorted(expected_auxiliary - used_parameters)
    if unmodeled_auxiliary:
        raise ValueError(
            f"Qwen-family graph left manifest norm/bias parameters unmodeled: {unmodeled_auxiliary}"
        )

    return OpGraph(
        model_name=plan.model_name,
        model_revision=plan.model_revision,
        store_fingerprint=plan.store_fingerprint,
        architecture=architecture,
        numerical_contract=plan.numerical_contract,
        values=tuple(builder.values),
        nodes=tuple(builder.nodes),
        inputs=tuple(builder.inputs),
        outputs=outputs,
        metadata=freeze_json_mapping(
            {
                "activation_dtype": activation_dtype,
                "batch_bucket": batch,
                "graph_semantics": "executable-qwen-family",
                "live_batch": plan.shape.actual_batch,
                "live_sequence": plan.shape.sequence_length,
                "output_contract": plan.output_contract.value,
                "sequence_bucket": sequence,
                "vocabulary_size": vocab,
                "weight_dtype": plan.precision.weight_dtype,
            }
        ),
    )


def _qwen_graph_body(graph: OpGraph) -> tuple[list[TensorSpec], list[OpNode], list[str]]:
    """Extract the canonical transformer body through ``final.hidden``."""

    body_nodes = [node for node in graph.nodes if not node.node_id.startswith("output.")]
    body_value_ids = {value_id for node in body_nodes for value_id in (*node.inputs, *node.outputs)}
    body_inputs = [value_id for value_id in graph.inputs if value_id in body_value_ids]
    body_values = [value for value in graph.values if value.value_id in body_value_ids]
    if "final.hidden" not in body_value_ids:
        raise ValueError("Qwen graph body has no final.hidden value")
    return body_values, body_nodes, body_inputs


def _parameter_rows(
    parameter: ParameterRef,
    row_indices: Sequence[int],
) -> ParameterRef:
    rows = tuple(int(value) for value in row_indices)
    return ParameterRef(
        logical_name=parameter.logical_name,
        physical_name=parameter.physical_name,
        kind=parameter.kind,
        shape=parameter.shape,
        regions=parameter.regions,
        access="rows" if rows else "all",
        row_indices=rows,
    )


def _tail_context(
    graph: OpGraph,
) -> tuple[int, int, int, int, str]:
    final_hidden = graph.tensor_map.get("final.hidden")
    if final_hidden is None or len(final_hidden.shape) != 3:
        raise ValueError("Qwen graph final.hidden must have shape [batch, sequence, hidden]")
    batch, sequence, hidden = final_hidden.shape
    metadata = thaw_json_mapping(graph.metadata)
    vocab = int(metadata.get("vocabulary_size", 0))
    if vocab <= 0:
        head_nodes = [
            node for node in graph.nodes if node.kind is OpKind.VOCAB_PROJECTION and node.parameters
        ]
        if head_nodes:
            vocab = int(head_nodes[-1].parameters[0].shape[0])
    if vocab <= 0:
        raise ValueError("Qwen graph does not record a positive vocabulary size")
    return batch, sequence, hidden, vocab, final_hidden.dtype


def _append_candidate_outputs(
    builder: _GraphBuilder,
    *,
    logits: str,
    batch: int,
    plan: DenseWorkPlan,
    union_token_ids: tuple[int, ...],
) -> tuple[str, ...]:
    return builder.add_node(
        node_id="output.candidate_topk_margin",
        kind=OpKind.TOPK_MARGIN,
        inputs=(logits,),
        output_specs=(
            _activation(
                "output.winner_token_id",
                (batch,),
                "int64",
                "B",
                storage_class=StorageClass.OUTPUT,
            ),
            _activation(
                "output.runner_up_token_id",
                (batch,),
                "int64",
                "B",
                storage_class=StorageClass.OUTPUT,
            ),
            _activation(
                "output.winner_logit",
                (batch,),
                "fp32",
                "B",
                storage_class=StorageClass.OUTPUT,
            ),
            _activation(
                "output.runner_up_logit",
                (batch,),
                "fp32",
                "B",
                storage_class=StorageClass.OUTPUT,
            ),
            _activation(
                "output.margin",
                (batch,),
                "fp32",
                "B",
                storage_class=StorageClass.OUTPUT,
            ),
        ),
        params={
            "candidate_token_ids": plan.candidate_token_ids,
            "union_token_ids": union_token_ids,
            "live_batch": plan.shape.actual_batch,
            "top_k": 2,
        },
    )


def _source_graph_from_qwen_graph(
    optimized_graph: OpGraph,
    plan: DenseWorkPlan,
    full_head: ParameterRef,
) -> OpGraph:
    """Build the canonical unsliced full-vocabulary source program."""

    values, nodes, inputs = _qwen_graph_body(optimized_graph)
    batch, sequence, hidden, vocab, activation_dtype = _tail_context(optimized_graph)
    builder = _GraphBuilder()
    builder.values = values
    builder.nodes = nodes
    builder.inputs = inputs
    (full_logits,) = builder.add_node(
        node_id="output.source_full_vocab",
        kind=OpKind.VOCAB_PROJECTION,
        inputs=("final.hidden",),
        output_specs=(
            _activation(
                "output.source_full_logits",
                (batch, sequence, vocab),
                "fp32",
                "BTV",
                storage_class=(
                    StorageClass.OUTPUT
                    if plan.output_contract is OutputContract.FULL_LOGITS
                    else StorageClass.ACTIVATION
                ),
            ),
        ),
        parameters=(full_head,),
        params={
            "access": "all",
            "accumulator_dtype": "fp32",
            "logical_parameter": "lm_head",
            "semantic_source": "unsliced-full-vocabulary",
        },
    )
    outputs: tuple[str, ...]
    if plan.output_contract is OutputContract.FULL_LOGITS:
        outputs = (full_logits,)
    elif plan.output_contract is OutputContract.HIDDEN_STATE_ONLY:
        # The full head intentionally remains dead in the canonical source.  The
        # demand pass supplies the structural proof that it may be removed.
        outputs = ("final.hidden",)
    elif plan.output_contract is OutputContract.LOSS_ONLY:
        labels = builder.add_input(
            "input.labels",
            (batch, sequence),
            "int64",
            "BT",
        )
        outputs = builder.add_node(
            node_id="output.cross_entropy",
            kind=OpKind.CROSS_ENTROPY,
            inputs=(full_logits, labels),
            output_specs=(
                _activation(
                    "output.loss",
                    (),
                    "fp32",
                    "scalar",
                    storage_class=StorageClass.OUTPUT,
                ),
            ),
            params={
                "live_batch": plan.shape.actual_batch,
                "live_sequence": plan.shape.sequence_length,
                "reduction": "mean",
                "shift": 1,
            },
        )
    else:
        (last_logits,) = builder.add_node(
            node_id="output.source_take_last_logits",
            kind=OpKind.TAKE_LAST,
            inputs=(full_logits,),
            output_specs=(
                _activation(
                    "output.source_last_logits",
                    (batch, vocab),
                    "fp32",
                    "BV",
                    storage_class=(
                        StorageClass.OUTPUT
                        if plan.output_contract is OutputContract.LAST_TOKEN_LOGITS
                        else StorageClass.ACTIVATION
                    ),
                ),
            ),
            params={"right_padding": True},
        )
        if plan.output_contract is OutputContract.LAST_TOKEN_LOGITS:
            outputs = (last_logits,)
        else:
            selected_ids = (
                plan.required_output_rows
                if plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS
                else tuple(
                    dict.fromkeys(
                        token for candidates in plan.candidate_token_ids for token in candidates
                    )
                )
            )
            (selected_logits,) = builder.add_node(
                node_id="output.source_select_vocab_rows",
                kind=OpKind.SELECT_ROWS,
                inputs=(last_logits,),
                output_specs=(
                    _activation(
                        "output.source_selected_logits",
                        (batch, len(selected_ids)),
                        "fp32",
                        "BV",
                        storage_class=(
                            StorageClass.OUTPUT
                            if plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS
                            else StorageClass.ACTIVATION
                        ),
                    ),
                ),
                params={
                    "axis": -1,
                    "row_indices": selected_ids,
                    "semantic_source": "host-selection-after-full-head",
                },
            )
            outputs = (
                (selected_logits,)
                if plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS
                else _append_candidate_outputs(
                    builder,
                    logits=selected_logits,
                    batch=batch,
                    plan=plan,
                    union_token_ids=selected_ids,
                )
            )
    metadata = thaw_json_mapping(optimized_graph.metadata)
    metadata.update(
        {
            "graph_semantics": "canonical-unsliced-qwen-family",
            "output_demand_state": "source",
            "source_activation_dtype": activation_dtype,
            "vocabulary_size": vocab,
        }
    )
    return OpGraph(
        model_name=optimized_graph.model_name,
        model_revision=optimized_graph.model_revision,
        store_fingerprint=optimized_graph.store_fingerprint,
        architecture=optimized_graph.architecture,
        numerical_contract=optimized_graph.numerical_contract,
        values=tuple(builder.values),
        nodes=tuple(builder.nodes),
        inputs=tuple(builder.inputs),
        outputs=outputs,
        metadata=freeze_json_mapping(metadata),
    )


def _optimized_tail_from_source(
    source_graph: OpGraph,
    plan: DenseWorkPlan,
    full_head: ParameterRef,
) -> OpGraph:
    values, nodes, inputs = _qwen_graph_body(source_graph)
    batch, sequence, hidden, vocab, activation_dtype = _tail_context(source_graph)
    builder = _GraphBuilder()
    builder.values = values
    builder.nodes = nodes
    builder.inputs = inputs
    outputs: tuple[str, ...]
    if plan.output_contract is OutputContract.HIDDEN_STATE_ONLY:
        outputs = ("final.hidden",)
    else:
        (last_hidden,) = builder.add_node(
            node_id="output.take_last_hidden",
            kind=OpKind.TAKE_LAST,
            inputs=("final.hidden",),
            output_specs=(
                _activation(
                    "output.last_hidden",
                    (batch, hidden),
                    activation_dtype,
                    "BH",
                ),
            ),
            params={"right_padding": True},
        )
        selected_ids: tuple[int, ...] = ()
        if plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
            selected_ids = plan.required_output_rows
        elif plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
            selected_ids = tuple(
                dict.fromkeys(
                    token for candidates in plan.candidate_token_ids for token in candidates
                )
            )
        head = _parameter_rows(full_head, selected_ids)
        head_width = len(selected_ids) if selected_ids else vocab
        projection_output = (
            "output.candidate_union_logits"
            if plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
            else "output.logits"
        )
        (logits,) = builder.add_node(
            node_id="output.vocab_projection",
            kind=OpKind.VOCAB_PROJECTION,
            inputs=(last_hidden,),
            output_specs=(
                _activation(
                    projection_output,
                    (batch, head_width),
                    "fp32",
                    "BV",
                    storage_class=(
                        StorageClass.ACTIVATION
                        if plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
                        else StorageClass.OUTPUT
                    ),
                ),
            ),
            parameters=(head,),
            params={
                "access": head.access,
                "accumulator_dtype": "fp32",
                "logical_parameter": "lm_head",
                "output_contract": plan.output_contract.value,
            },
        )
        outputs = (
            _append_candidate_outputs(
                builder,
                logits=logits,
                batch=batch,
                plan=plan,
                union_token_ids=selected_ids,
            )
            if plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
            else (logits,)
        )
    metadata = thaw_json_mapping(source_graph.metadata)
    metadata.update(
        {
            "graph_semantics": "output-demand-rewritten-qwen-family",
            "output_demand_state": "rewritten",
            "vocabulary_size": vocab,
        }
    )
    return OpGraph(
        model_name=source_graph.model_name,
        model_revision=source_graph.model_revision,
        store_fingerprint=source_graph.store_fingerprint,
        architecture=source_graph.architecture,
        numerical_contract=source_graph.numerical_contract,
        values=tuple(builder.values),
        nodes=tuple(builder.nodes),
        inputs=tuple(builder.inputs),
        outputs=outputs,
        metadata=freeze_json_mapping(metadata),
    )


def rewrite_output_demand(
    source_graph: OpGraph,
    plan: DenseWorkPlan,
) -> tuple[OpGraph, DemandRewriteCertificate]:
    """Specialize a canonical source graph to the requested output's causal cone."""

    if source_graph.model_name != plan.model_name:
        raise ValueError("source graph model does not match WorkPlan")
    if source_graph.model_revision != plan.model_revision:
        raise ValueError("source graph revision does not match WorkPlan")
    if source_graph.store_fingerprint != plan.store_fingerprint:
        raise ValueError("source graph store does not match WorkPlan")
    metadata = thaw_json_mapping(source_graph.metadata)
    if metadata.get("output_demand_state") != "source":
        raise ValueError("rewrite_output_demand requires a canonical unsliced source graph")
    if metadata.get("output_contract") != plan.output_contract.value:
        raise ValueError("source graph output contract does not match WorkPlan")

    source_head_nodes = [
        node for node in source_graph.nodes if node.node_id == "output.source_full_vocab"
    ]
    if len(source_head_nodes) != 1 or len(source_head_nodes[0].parameters) != 1:
        raise ValueError("canonical source graph must contain one full-vocabulary head")
    full_head = source_head_nodes[0].parameters[0]
    if full_head.access != "all":
        raise ValueError("canonical source vocabulary head must access all rows")

    identity_contract = plan.output_contract in {
        OutputContract.FULL_LOGITS,
        OutputContract.LOSS_ONLY,
    }
    if identity_contract:
        rewritten = source_graph
        rewrite_ids = (
            "identity-full-vocabulary-demand"
            if plan.output_contract is OutputContract.FULL_LOGITS
            else "identity-loss-requires-full-vocabulary",
        )
        claims = (
            "loss retains full-vocabulary normalization"
            if plan.output_contract is OutputContract.LOSS_ONLY
            else "full-logits demand retains the source graph",
        )
    else:
        rewritten = _optimized_tail_from_source(source_graph, plan, full_head)
        if plan.output_contract is OutputContract.LAST_TOKEN_LOGITS:
            rewrite_ids = ("take-last-through-row-wise-linear-head",)
            claims = ("only last-token hidden rows reach the full-vocabulary head",)
        elif plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
            rewrite_ids = (
                "take-last-through-row-wise-linear-head",
                "vocabulary-row-selection-into-parameter-access",
            )
            claims = (
                "only requested vocabulary rows are read by the head",
                "selected logits preserve requested row order",
            )
        elif plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
            rewrite_ids = (
                "take-last-through-row-wise-linear-head",
                "candidate-union-into-parameter-access",
                "top2-margin-depends-only-on-candidate-logits",
            )
            claims = (
                "only the stable union of candidate vocabulary rows is read",
                "winner and margin depend only on each row's candidate logits",
            )
        elif plan.output_contract is OutputContract.HIDDEN_STATE_ONLY:
            rewrite_ids = ("backward-slice-unreachable-vocabulary-head",)
            claims = ("the vocabulary head is unreachable from hidden-state output",)
        else:  # selected capture already fails in the Qwen source builder
            raise NotImplementedError(f"no output-demand rewrite for {plan.output_contract.value}")

    source_ids = [node.node_id for node in source_graph.nodes]
    rewritten_ids = [node.node_id for node in rewritten.nodes]
    removed = tuple(node_id for node_id in source_ids if node_id not in rewritten_ids)
    added = tuple(node_id for node_id in rewritten_ids if node_id not in source_ids)
    certificate = DemandRewriteCertificate(
        source_fingerprint=source_graph.fingerprint,
        rewritten_fingerprint=rewritten.fingerprint,
        output_contract=plan.output_contract.value,
        rewrite_ids=rewrite_ids,
        removed_node_ids=removed,
        added_node_ids=added,
        claims=claims,
    )
    return rewritten, certificate


def _build_generic_manifest_graph(
    plan: DenseWorkPlan,
    manifest: Mapping[str, Any],
    original_architecture: str,
) -> OpGraph:
    """Build an explicitly non-executable resource graph for fixtures/audits."""

    builder = _GraphBuilder()
    current = builder.add_input("input.generic_seed", (1,), "uint8", "opaque")
    blocks = _manifest_blocks(manifest)
    for index, logical_name in enumerate(sorted(str(name) for name in blocks)):
        parameter = _manifest_parameter(manifest, logical_name)
        output_id = f"generic.parameter.{index}.token"
        (current,) = builder.add_node(
            node_id=f"generic.parameter.{index}",
            kind=OpKind.GENERIC_PARAMETER_STREAM,
            inputs=(current,),
            output_specs=(_activation(output_id, (1,), "uint8", "opaque"),),
            parameters=(parameter,),
            params={
                "executable": False,
                "logical_name": logical_name,
                "semantic_status": "resource-order-only",
            },
        )
    if not builder.nodes:
        (current,) = builder.add_node(
            node_id="generic.empty_manifest",
            kind=OpKind.IDENTITY,
            inputs=(current,),
            output_specs=(_activation("generic.empty.output", (1,), "uint8", "opaque"),),
            params={"executable": False, "semantic_status": "empty-manifest"},
        )
    return OpGraph(
        model_name=plan.model_name,
        model_revision=plan.model_revision,
        store_fingerprint=plan.store_fingerprint,
        architecture=f"generic-manifest:{original_architecture or 'unspecified'}",
        numerical_contract=plan.numerical_contract,
        values=tuple(builder.values),
        nodes=tuple(builder.nodes),
        inputs=tuple(builder.inputs),
        outputs=(current,),
        metadata=freeze_json_mapping(
            {
                "executable": False,
                "graph_semantics": "generic-manifest-resource-graph",
                "reason": "unsupported or unspecified architecture accepted by explicit opt-in",
            }
        ),
    )


def _validate_manifest_identity(plan: DenseWorkPlan, manifest: Mapping[str, Any]) -> None:
    source = manifest.get("source")
    if isinstance(source, Mapping):
        revision = source.get("source_checkpoint_sha256")
        if not manifest_declaration_matches(plan.model_revision, revision):
            raise ValueError("manifest source identity does not match WorkPlan model_revision")
    derived = manifest.get("derived")
    if isinstance(derived, Mapping):
        fingerprint = derived.get("derived_store_sha256")
        if not manifest_declaration_matches(plan.store_fingerprint, fingerprint):
            raise ValueError("manifest derived identity does not match WorkPlan store_fingerprint")


def build_op_graph(
    plan: DenseWorkPlan,
    manifest: Mapping[str, Any],
    *,
    allow_generic_manifest: bool = False,
) -> OpGraph:
    """Build a validated semantic graph from a WorkPlan and QStore manifest.

    Qwen2, Qwen3, and Llama manifests use the executable Qwen-family graph.  Other
    architectures fail closed unless ``allow_generic_manifest`` is explicitly set;
    that opt-in returns a clearly marked, non-executable resource graph intended for
    fixture validation and manifest inspection only.
    """

    if not isinstance(plan, DenseWorkPlan):
        raise TypeError("plan must be a DenseWorkPlan")
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be an object")
    _validate_manifest_identity(plan, manifest)
    architecture = str(manifest.get("arch", "")).strip().lower()
    if architecture in {"qwen2", "qwen3", "llama"}:
        return _build_qwen_family_graph(plan, manifest, architecture)
    if allow_generic_manifest:
        return _build_generic_manifest_graph(plan, manifest, architecture)
    label = architecture or "unspecified"
    raise NotImplementedError(
        f"OpGraph builder does not support architecture {label!r}; "
        "pass allow_generic_manifest=True only for a non-executable resource graph"
    )


def build_qwen_family_graph(
    plan: DenseWorkPlan,
    manifest: Mapping[str, Any],
) -> OpGraph:
    """Build a Qwen-family graph and reject every other architecture."""

    graph = build_op_graph(plan, manifest, allow_generic_manifest=False)
    if graph.architecture not in {"qwen2", "qwen3", "llama"}:  # defensive API invariant
        raise RuntimeError("build_qwen_family_graph returned a non-Qwen graph")
    return graph


def build_source_op_graph(
    plan: DenseWorkPlan,
    manifest: Mapping[str, Any],
) -> OpGraph:
    """Build the canonical unsliced graph used as a rewrite audit baseline.

    The current canonical source is defined for Qwen2/Qwen3/Llama score and
    prefill graphs. It always materializes the full vocabulary head before any
    last-token, selected-row, or candidate projection.
    """

    optimized = build_qwen_family_graph(plan, manifest)
    full_head = _manifest_parameter(manifest, "lm_head", access="all")
    return _source_graph_from_qwen_graph(optimized, plan, full_head)


def build_output_demand_graph(
    plan: DenseWorkPlan,
    manifest: Mapping[str, Any],
) -> tuple[OpGraph, OpGraph, DemandRewriteCertificate]:
    """Build canonical and rewritten graphs plus their structural certificate."""

    source = build_source_op_graph(plan, manifest)
    rewritten, certificate = rewrite_output_demand(source, plan)
    return source, rewritten, certificate


__all__ = [
    "DemandRewriteCertificate",
    "Effect",
    "EffectKind",
    "FrozenJson",
    "OPGRAPH_SCHEMA",
    "OpGraph",
    "OpKind",
    "OpNode",
    "ParameterRef",
    "StorageClass",
    "TensorSpec",
    "build_op_graph",
    "build_output_demand_graph",
    "build_qwen_family_graph",
    "build_source_op_graph",
    "freeze_json_mapping",
    "rewrite_output_demand",
    "thaw_json_mapping",
]
