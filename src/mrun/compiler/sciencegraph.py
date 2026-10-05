"""Typed intervention graphs for shared-prefix causal analysis.

This module compiles a family of branch-local interventions over one exact
prompt into a deterministic :class:`BranchPack`.  The pack is backend-neutral:
it records the semantic fork/cut, the stable candidate union, row-local replay
descriptors, authenticated tensor payloads, and immutable fan-out maps.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any

SCIENCEGRAPH_SCHEMA = "mrun-intervention-sciencegraph-v2"
BRANCH_PACK_SCHEMA = "mrun-intervention-branch-pack-v2"
SCIENCEGRAPH_EXECUTION_SCHEMA = "mrun-intervention-sciencegraph-execution-v2"
SCIENCEGRAPH_BENCHMARK_SCHEMA = "mrun-intervention-sciencegraph-benchmark-v2"

ReplayValue = float | str | None
ReplayOp = tuple[str, tuple[int, ...], ReplayValue]
FrozenLayerPatchMap = tuple[tuple[int, tuple[ReplayOp, ...]], ...]


def _sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_id(value: str, *, field: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _canonical_ints(values: Sequence[int], *, field: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be an integer sequence")
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in result):
        raise TypeError(f"{field} must contain only integers")
    return tuple(int(value) for value in result)


def _require_keys(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    field: str,
) -> None:
    keys = set(payload)
    allowed = required | (optional or set())
    if missing := sorted(required - keys):
        raise ValueError(f"{field} is missing required keys: {missing}")
    if extra := sorted(keys - allowed):
        raise ValueError(f"{field} contains unsupported keys: {extra}")


class InterventionPort(str, Enum):
    """Disassembled activation boundary at which an intervention is applied."""

    MLP_DOWN_INPUT = "mlp_down_input"
    ATTENTION_HEAD_OUTPUT = "attention_head_output"
    RESIDUAL_OUTPUT = "residual_output"
    KEY_PROJECTION = "key_projection"
    VALUE_PROJECTION = "value_projection"


class InterventionOp(str, Enum):
    """Operations supported by the paged intervention ABI."""

    ZERO = "zero"
    SCALE = "scale"
    GLOBAL_MEAN = "global_mean"
    POSITION_MEAN = "position_mean"
    ADD_AMP = "add_amp"
    PROJECTION_REMOVE = "proj_remove"
    POSITION_REPLACE = "position_replace"


def _tensor_sha256(value: Any) -> str:
    try:
        import torch

        tensor = value.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise TypeError("payload binding must be a detached-compatible torch tensor") from exc
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class TensorPayloadSpec:
    """Content identity for one tensor-valued intervention operand."""

    payload_id: str
    sha256: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload_id", _canonical_id(self.payload_id, field="payload_id"))
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise ValueError("payload sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "dtype", _canonical_id(self.dtype, field="payload dtype"))
        object.__setattr__(self, "shape", _canonical_ints(self.shape, field="payload shape"))
        if any(size <= 0 for size in self.shape):
            raise ValueError("payload dimensions must be positive")

    @classmethod
    def from_tensor(cls, payload_id: str, value: Any) -> TensorPayloadSpec:
        try:
            shape = tuple(int(size) for size in value.shape)
            dtype = str(value.dtype)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("payload value must be a torch tensor") from exc
        return cls(payload_id, _tensor_sha256(value), dtype, shape)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TensorPayloadSpec:
        _require_keys(
            payload,
            required={"payload_id", "sha256", "dtype", "shape"},
            field="TensorPayloadSpec",
        )
        return cls(
            payload_id=payload["payload_id"],
            sha256=payload["sha256"],
            dtype=payload["dtype"],
            shape=tuple(payload["shape"]),
        )

    def validate_binding(self, value: Any) -> None:
        actual = TensorPayloadSpec.from_tensor(self.payload_id, value)
        if actual != self:
            raise ValueError(f"tensor payload binding drift for {self.payload_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_id": self.payload_id,
            "sha256": self.sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class StateCut:
    """The earliest layer input shared by every branch in a fork."""

    layer: int
    prompt_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, Integral):
            raise TypeError("StateCut.layer must be an integer")
        object.__setattr__(self, "layer", int(self.layer))
        if self.layer < 0:
            raise ValueError("StateCut.layer must be non-negative")
        if (
            type(self.prompt_sha256) is not str
            or len(self.prompt_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.prompt_sha256)
        ):
            raise ValueError("StateCut.prompt_sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "prompt_sha256": self.prompt_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StateCut:
        _require_keys(
            payload,
            required={"layer", "prompt_sha256"},
            field="StateCut",
        )
        return cls(layer=payload["layer"], prompt_sha256=payload["prompt_sha256"])


@dataclass(frozen=True)
class Intervention:
    """One immutable branch-local edit at a typed model port."""

    layer: int
    indices: tuple[int, ...]
    op: InterventionOp | str = InterventionOp.ZERO
    port: InterventionPort | str = InterventionPort.MLP_DOWN_INPUT
    value: float | None = None
    payload_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, Integral):
            raise TypeError("intervention layer must be an integer")
        object.__setattr__(self, "layer", int(self.layer))
        object.__setattr__(self, "indices", _canonical_ints(self.indices, field="indices"))
        try:
            object.__setattr__(self, "op", InterventionOp(self.op))
            object.__setattr__(self, "port", InterventionPort(self.port))
        except ValueError as exc:
            raise ValueError("unsupported intervention operation or port") from exc
        if self.layer < 0:
            raise ValueError("interventions require a non-negative layer")
        if len(self.indices) != len(set(self.indices)) or any(index < 0 for index in self.indices):
            raise ValueError("intervention indices must be unique and non-negative")
        indexed_ops = {
            InterventionOp.ZERO,
            InterventionOp.SCALE,
            InterventionOp.GLOBAL_MEAN,
            InterventionOp.POSITION_MEAN,
            InterventionOp.ADD_AMP,
            InterventionOp.POSITION_REPLACE,
        }
        tensor_ops = {
            InterventionOp.GLOBAL_MEAN,
            InterventionOp.POSITION_MEAN,
            InterventionOp.ADD_AMP,
            InterventionOp.PROJECTION_REMOVE,
            InterventionOp.POSITION_REPLACE,
        }
        if self.op in indexed_ops and not self.indices:
            raise ValueError(f"{self.op.value} interventions require at least one index")
        if self.op is InterventionOp.PROJECTION_REMOVE and self.indices:
            raise ValueError("projection removal takes a direction payload, not indices")
        if self.op in {InterventionOp.ZERO, InterventionOp.SCALE} and self.port not in {
            InterventionPort.MLP_DOWN_INPUT,
            InterventionPort.ATTENTION_HEAD_OUTPUT,
        }:
            raise ValueError("zero/scale are supported only at MLP or head-output ports")
        if self.op in {
            InterventionOp.GLOBAL_MEAN,
            InterventionOp.POSITION_MEAN,
            InterventionOp.ADD_AMP,
        } and self.port not in {
            InterventionPort.MLP_DOWN_INPUT,
            InterventionPort.ATTENTION_HEAD_OUTPUT,
        }:
            raise ValueError("tensor replacement/addition is unsupported at this port")
        if self.op is InterventionOp.PROJECTION_REMOVE and (
            self.port is not InterventionPort.RESIDUAL_OUTPUT
        ):
            raise ValueError("projection removal requires the residual-output port")
        if self.op is InterventionOp.POSITION_REPLACE and self.port not in {
            InterventionPort.KEY_PROJECTION,
            InterventionPort.VALUE_PROJECTION,
        }:
            raise ValueError("position replacement requires a K/V projection port")
        if self.port in {
            InterventionPort.KEY_PROJECTION,
            InterventionPort.VALUE_PROJECTION,
        } and self.op is not InterventionOp.POSITION_REPLACE:
            raise ValueError("K/V projection ports require position replacement")
        if self.op is InterventionOp.SCALE:
            if isinstance(self.value, bool) or not isinstance(self.value, Real):
                raise TypeError("scale interventions require a finite numeric value")
            scalar = float(self.value)
            if not float("-inf") < scalar < float("inf"):
                raise ValueError("scale intervention value must be finite")
            object.__setattr__(self, "value", scalar)
        elif self.value is not None:
            raise ValueError(f"{self.op.value} interventions do not accept a scalar value")
        if self.op in tensor_ops:
            object.__setattr__(
                self,
                "payload_id",
                _canonical_id(self.payload_id, field="payload_id"),  # type: ignore[arg-type]
            )
        elif self.payload_id is not None:
            raise ValueError(f"{self.op.value} interventions do not accept a tensor payload")

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "port": self.port.value,
            "op": self.op.value,
            "indices": list(self.indices),
            "value": self.value,
            "payload_id": self.payload_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Intervention:
        _require_keys(
            payload,
            required={"layer", "indices", "op", "port"},
            optional={"value", "payload_id"},
            field="Intervention",
        )
        return cls(
            layer=payload["layer"],
            indices=tuple(payload["indices"]),
            op=payload["op"],
            port=payload["port"],
            value=payload.get("value"),
            payload_id=payload.get("payload_id"),
        )

    def replay_tuple(self) -> ReplayOp:
        operand: ReplayValue = self.payload_id if self.payload_id is not None else self.value
        return self.op.value, self.indices, operand


@dataclass(frozen=True)
class InterventionBranch:
    """One logical consumer of the shared state."""

    branch_id: str
    candidate_token_ids: tuple[int, ...]
    interventions: tuple[Intervention, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_id", _canonical_id(self.branch_id, field="branch_id"))
        object.__setattr__(
            self,
            "candidate_token_ids",
            _canonical_ints(self.candidate_token_ids, field="candidate_token_ids"),
        )
        if not self.candidate_token_ids:
            raise ValueError("each branch requires at least one candidate token")
        if len(self.candidate_token_ids) != len(set(self.candidate_token_ids)):
            raise ValueError("candidate token IDs must be unique within each branch")
        if any(token < 0 for token in self.candidate_token_ids):
            raise ValueError("candidate token IDs must be non-negative")
        interventions = tuple(self.interventions)
        if any(not isinstance(item, Intervention) for item in interventions):
            raise TypeError("interventions must contain Intervention values")
        if len({(item.layer, item.port, item.indices) for item in interventions}) != len(
            interventions
        ):
            raise ValueError("a branch cannot edit the same typed index set twice")
        object.__setattr__(self, "interventions", interventions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "candidate_token_ids": list(self.candidate_token_ids),
            "interventions": [item.to_dict() for item in self.interventions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InterventionBranch:
        _require_keys(
            payload,
            required={"branch_id", "candidate_token_ids", "interventions"},
            field="InterventionBranch",
        )
        return cls(
            branch_id=payload["branch_id"],
            candidate_token_ids=tuple(payload["candidate_token_ids"]),
            interventions=tuple(
                Intervention.from_dict(item) for item in payload.get("interventions", ())
            ),
        )


@dataclass(frozen=True)
class InterventionFork:
    """Logical fan-out from one exact shared prompt state."""

    cut: StateCut
    branches: tuple[InterventionBranch, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cut, StateCut):
            raise TypeError("fork cut must be a StateCut")
        branches = tuple(self.branches)
        if not branches or any(not isinstance(branch, InterventionBranch) for branch in branches):
            raise ValueError("fork requires InterventionBranch values")
        ids = tuple(branch.branch_id for branch in branches)
        if len(ids) != len(set(ids)):
            raise ValueError("branch IDs must be unique")
        for branch in branches:
            if any(item.layer < self.cut.layer for item in branch.interventions):
                raise ValueError("an intervention cannot precede its shared StateCut")
        earliest = min(
            (item.layer for branch in branches for item in branch.interventions),
            default=0,
        )
        if self.cut.layer != earliest:
            raise ValueError("StateCut must equal the earliest branch intervention layer")
        object.__setattr__(self, "branches", branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cut": self.cut.to_dict(),
            "branches": [branch.to_dict() for branch in self.branches],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InterventionFork:
        _require_keys(payload, required={"cut", "branches"}, field="InterventionFork")
        return cls(
            cut=StateCut.from_dict(payload["cut"]),
            branches=tuple(InterventionBranch.from_dict(item) for item in payload["branches"]),
        )


@dataclass(frozen=True)
class BranchPack:
    """Physical branch batch plus immutable logical result projections."""

    model_identity: str
    numerical_contract: str
    fork: InterventionFork
    candidate_union: tuple[int, ...]
    candidate_projections: tuple[tuple[int, ...], ...]
    row_patch_maps: tuple[FrozenLayerPatchMap, ...]
    row_head_patch_maps: tuple[FrozenLayerPatchMap, ...]
    row_resid_patch_maps: tuple[FrozenLayerPatchMap, ...]
    row_key_patch_maps: tuple[FrozenLayerPatchMap, ...]
    row_value_patch_maps: tuple[FrozenLayerPatchMap, ...]
    payload_specs: tuple[TensorPayloadSpec, ...] = ()
    schema_version: str = BRANCH_PACK_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_identity", _canonical_id(self.model_identity, field="model_identity")
        )
        object.__setattr__(
            self,
            "numerical_contract",
            _canonical_id(self.numerical_contract, field="numerical_contract"),
        )
        if self.schema_version != BRANCH_PACK_SCHEMA:
            raise ValueError("unsupported BranchPack schema")
        object.__setattr__(
            self,
            "candidate_union",
            _canonical_ints(self.candidate_union, field="candidate_union"),
        )
        if not self.candidate_union or len(self.candidate_union) != len(set(self.candidate_union)):
            raise ValueError("candidate_union must be non-empty and unique")
        if any(token < 0 for token in self.candidate_union):
            raise ValueError("candidate_union must be non-negative")
        projections = tuple(
            _canonical_ints(row, field="candidate projection")
            for row in self.candidate_projections
        )
        object.__setattr__(self, "candidate_projections", projections)
        if len(self.candidate_projections) != len(self.fork.branches):
            raise ValueError("candidate projections must align with branches")
        for branch, projection in zip(
            self.fork.branches, self.candidate_projections, strict=True
        ):
            try:
                projected = tuple(self.candidate_union[index] for index in projection)
            except IndexError as exc:
                raise ValueError("candidate projection falls outside the stable union") from exc
            if projected != branch.candidate_token_ids:
                raise ValueError("candidate projection does not reconstruct its branch")
        for field_name in (
            "row_patch_maps",
            "row_head_patch_maps",
            "row_resid_patch_maps",
            "row_key_patch_maps",
            "row_value_patch_maps",
        ):
            if len(getattr(self, field_name)) != len(self.fork.branches):
                raise ValueError(f"{field_name} must align with branches")
        def expected_maps(port: InterventionPort) -> tuple[FrozenLayerPatchMap, ...]:
            rows = []
            for branch in self.fork.branches:
                by_layer: dict[int, list[ReplayOp]] = {}
                for intervention in branch.interventions:
                    if intervention.port is port:
                        by_layer.setdefault(intervention.layer, []).append(
                            intervention.replay_tuple()
                        )
                rows.append(
                    tuple((layer, tuple(by_layer[layer])) for layer in sorted(by_layer))
                )
            return tuple(rows)

        expected_by_field = {
            "row_patch_maps": expected_maps(InterventionPort.MLP_DOWN_INPUT),
            "row_head_patch_maps": expected_maps(InterventionPort.ATTENTION_HEAD_OUTPUT),
            "row_resid_patch_maps": expected_maps(InterventionPort.RESIDUAL_OUTPUT),
            "row_key_patch_maps": expected_maps(InterventionPort.KEY_PROJECTION),
            "row_value_patch_maps": expected_maps(InterventionPort.VALUE_PROJECTION),
        }
        for field_name, expected in expected_by_field.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} diverges from the semantic intervention fork")
        specs = tuple(self.payload_specs)
        if any(not isinstance(spec, TensorPayloadSpec) for spec in specs):
            raise TypeError("payload_specs must contain TensorPayloadSpec values")
        ids = tuple(spec.payload_id for spec in specs)
        if len(ids) != len(set(ids)):
            raise ValueError("payload IDs must be unique")
        referenced = {
            item.payload_id
            for branch in self.fork.branches
            for item in branch.interventions
            if item.payload_id is not None
        }
        if referenced != set(ids):
            raise ValueError("payload specifications must exactly cover referenced payload IDs")
        object.__setattr__(self, "payload_specs", specs)

    @property
    def fingerprint(self) -> str:
        return _sha256(self.to_dict(include_fingerprint=False))

    def patch_maps(self) -> tuple[dict[int, list[ReplayOp]], ...]:
        return tuple(
            {layer: list(ops) for layer, ops in row}
            for row in self.row_patch_maps
        )

    def head_patch_maps(self) -> tuple[dict[int, list[ReplayOp]], ...]:
        return tuple(
            {layer: list(ops) for layer, ops in row}
            for row in self.row_head_patch_maps
        )

    def resid_patch_maps(self) -> tuple[dict[int, list[ReplayOp]], ...]:
        return tuple(
            {layer: list(ops) for layer, ops in row}
            for row in self.row_resid_patch_maps
        )

    def key_patch_maps(self) -> tuple[dict[int, list[ReplayOp]], ...]:
        return tuple(
            {layer: list(ops) for layer, ops in row}
            for row in self.row_key_patch_maps
        )

    def value_patch_maps(self) -> tuple[dict[int, list[ReplayOp]], ...]:
        return tuple(
            {layer: list(ops) for layer, ops in row}
            for row in self.row_value_patch_maps
        )

    def resolve_patch_maps(
        self,
        payload_bindings: Mapping[str, Any],
    ) -> tuple[
        tuple[dict[int, list[tuple[str, Any, Any]]], ...],
        tuple[dict[int, list[tuple[str, Any, Any]]], ...],
        tuple[dict[int, list[tuple[str, Any, Any]]], ...],
    ]:
        """Validate tensor custody and build the concrete paged replay ABI."""

        if set(payload_bindings) != {spec.payload_id for spec in self.payload_specs}:
            raise ValueError("tensor payload bindings must exactly match the BranchPack")
        frozen_bindings: dict[str, Any] = {}
        for spec in self.payload_specs:
            try:
                frozen = payload_bindings[spec.payload_id].detach().clone()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise TypeError("tensor payload binding cannot be frozen") from exc
            spec.validate_binding(frozen)
            frozen_bindings[spec.payload_id] = frozen

        def resolve(
            rows: tuple[FrozenLayerPatchMap, ...],
            *,
            residual: bool,
        ) -> tuple[dict[int, list[tuple[str, Any, Any]]], ...]:
            resolved_rows = []
            for row in rows:
                by_layer: dict[int, list[tuple[str, Any, Any]]] = {}
                for layer, ops in row:
                    concrete = []
                    for op, indices, operand in ops:
                        if isinstance(operand, str):
                            tensor = frozen_bindings[operand]
                            concrete.append(
                                (op, tensor, None) if residual else (op, indices, tensor)
                            )
                        else:
                            concrete.append((op, indices, operand))
                    by_layer[layer] = concrete
                resolved_rows.append(by_layer)
            return tuple(resolved_rows)

        return (
            resolve(self.row_patch_maps, residual=False),
            resolve(self.row_head_patch_maps, residual=False),
            resolve(self.row_resid_patch_maps, residual=True),
        )

    def resolve_all_patch_maps(
        self,
        payload_bindings: Mapping[str, Any],
    ) -> tuple[
        tuple[dict[int, list[tuple[str, Any, Any]]], ...],
        tuple[dict[int, list[tuple[str, Any, Any]]], ...],
        tuple[dict[int, list[tuple[str, Any, Any]]], ...],
        tuple[dict[int, list[tuple[str, Any, Any]]], ...],
        tuple[dict[int, list[tuple[str, Any, Any]]], ...],
    ]:
        """Resolve every physical port, including K/V projection replacement."""

        base = self.resolve_patch_maps(payload_bindings)

        def resolve_kv(
            rows: tuple[FrozenLayerPatchMap, ...],
        ) -> tuple[dict[int, list[tuple[str, Any, Any]]], ...]:
            resolved_rows = []
            for row in rows:
                by_layer: dict[int, list[tuple[str, Any, Any]]] = {}
                for layer, ops in row:
                    concrete = []
                    for op, positions, operand in ops:
                        if not isinstance(operand, str):
                            raise ValueError("K/V position replacement requires a tensor payload")
                        concrete.append((op, positions, payload_bindings[operand].detach().clone()))
                    by_layer[layer] = concrete
                resolved_rows.append(by_layer)
            return tuple(resolved_rows)

        return (*base, resolve_kv(self.row_key_patch_maps), resolve_kv(self.row_value_patch_maps))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema": self.schema_version,
            "model_identity": self.model_identity,
            "numerical_contract": self.numerical_contract,
            "fork": self.fork.to_dict(),
            "candidate_union": list(self.candidate_union),
            "candidate_projections": [list(row) for row in self.candidate_projections],
            "payload_specs": [spec.to_dict() for spec in self.payload_specs],
        }
        for key, maps in (
            ("row_patch_maps", self.row_patch_maps),
            ("row_head_patch_maps", self.row_head_patch_maps),
            ("row_resid_patch_maps", self.row_resid_patch_maps),
            ("row_key_patch_maps", self.row_key_patch_maps),
            ("row_value_patch_maps", self.row_value_patch_maps),
        ):
            if key in {"row_key_patch_maps", "row_value_patch_maps"} and not any(maps):
                continue
            body[key] = [
                [
                    {
                        "layer": layer,
                        "ops": [
                            {"op": op, "indices": list(indices), "value": value}
                            for op, indices, value in ops
                        ],
                    }
                    for layer, ops in row
                ]
                for row in maps
            ]
        if include_fingerprint:
            body["fingerprint"] = _sha256(body)
        return body

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BranchPack:
        _require_keys(
            payload,
            required={
                "schema",
                "model_identity",
                "numerical_contract",
                "fork",
                "candidate_union",
                "candidate_projections",
                "row_patch_maps",
                "row_head_patch_maps",
                "row_resid_patch_maps",
                "payload_specs",
            },
            optional={"fingerprint", "row_key_patch_maps", "row_value_patch_maps"},
            field="BranchPack",
        )
        def maps(key: str) -> tuple[FrozenLayerPatchMap, ...]:
            if key not in payload:
                return tuple(() for _ in payload["fork"]["branches"])
            return tuple(
                tuple(
                    (
                        item["layer"],
                        tuple(
                            (op["op"], tuple(op["indices"]), op.get("value"))
                            for op in item["ops"]
                        ),
                    )
                    for item in row
                )
                for row in payload[key]
            )

        result = cls(
            model_identity=payload["model_identity"],
            numerical_contract=payload["numerical_contract"],
            fork=InterventionFork.from_dict(payload["fork"]),
            candidate_union=tuple(payload["candidate_union"]),
            candidate_projections=tuple(
                tuple(row) for row in payload["candidate_projections"]
            ),
            row_patch_maps=maps("row_patch_maps"),
            row_head_patch_maps=maps("row_head_patch_maps"),
            row_resid_patch_maps=maps("row_resid_patch_maps"),
            row_key_patch_maps=maps("row_key_patch_maps"),
            row_value_patch_maps=maps("row_value_patch_maps"),
            payload_specs=tuple(
                TensorPayloadSpec.from_dict(item) for item in payload.get("payload_specs", ())
            ),
            schema_version=payload["schema"],
        )
        expected = payload.get("fingerprint")
        if expected is not None and expected != result.fingerprint:
            raise ValueError("BranchPack fingerprint mismatch")
        return result


@dataclass(frozen=True)
class InterventionScienceGraph:
    """Content-addressed semantic graph and its first physical BranchPack."""

    prompt_token_ids: tuple[int, ...]
    branch_pack: BranchPack
    schema_version: str = SCIENCEGRAPH_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_token_ids",
            _canonical_ints(self.prompt_token_ids, field="prompt_token_ids"),
        )
        if not self.prompt_token_ids or any(token < 0 for token in self.prompt_token_ids):
            raise ValueError("prompt token IDs must be non-empty and non-negative")
        if self.schema_version != SCIENCEGRAPH_SCHEMA:
            raise ValueError("unsupported InterventionScienceGraph schema")
        if self.branch_pack.fork.cut.prompt_sha256 != _prompt_sha256(self.prompt_token_ids):
            raise ValueError("StateCut prompt identity does not match graph tokens")

    @property
    def fingerprint(self) -> str:
        return _sha256(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        body = {
            "schema": self.schema_version,
            "prompt_token_ids": list(self.prompt_token_ids),
            "branch_pack": self.branch_pack.to_dict(),
        }
        if include_fingerprint:
            body["fingerprint"] = _sha256(body)
        return body

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InterventionScienceGraph:
        _require_keys(
            payload,
            required={"schema", "prompt_token_ids", "branch_pack"},
            optional={"fingerprint"},
            field="InterventionScienceGraph",
        )
        result = cls(
            prompt_token_ids=tuple(payload["prompt_token_ids"]),
            branch_pack=BranchPack.from_dict(payload["branch_pack"]),
            schema_version=payload["schema"],
        )
        expected = payload.get("fingerprint")
        if expected is not None and expected != result.fingerprint:
            raise ValueError("InterventionScienceGraph fingerprint mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> InterventionScienceGraph:
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
            raise ValueError("invalid InterventionScienceGraph JSON") from exc
        if not isinstance(value, Mapping):
            raise TypeError("InterventionScienceGraph JSON must contain an object")
        return cls.from_dict(value)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def write_json(self, path: str | Path) -> Path:
        """Atomically persist a checksum-bearing graph artifact."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(self.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> InterventionScienceGraph:
        return cls.from_json(Path(path).read_bytes())


@dataclass(frozen=True)
class InterventionBranchResult:
    """One branch's selected scores projected back from the stable union."""

    branch_id: str
    candidate_token_ids: tuple[int, ...]
    scores: tuple[float, ...]
    winner_token_id: int
    margin: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "candidate_token_ids": list(self.candidate_token_ids),
            "scores": list(self.scores),
            "winner_token_id": self.winner_token_id,
            "margin": self.margin,
        }


def _prompt_sha256(prompt_token_ids: Sequence[int]) -> str:
    payload = {"token_ids": list(_canonical_ints(prompt_token_ids, field="prompt_token_ids"))}
    return _sha256(payload)


def bind_sciencegraph_model_identity(engine: Any) -> str:
    """Return the canonical loaded-QStore identity used by ScienceGraph artifacts."""

    from .identity import bind_loaded_qstore_identity

    bound = bind_loaded_qstore_identity(engine)
    return f"{bound.model_name}@{bound.model_revision}#{bound.store_fingerprint}"


def compile_intervention_sciencegraph(
    *,
    model_identity: str,
    numerical_contract: str,
    prompt_token_ids: Sequence[int],
    branches: Sequence[InterventionBranch],
    payload_specs: Sequence[TensorPayloadSpec] = (),
) -> InterventionScienceGraph:
    """Compile one prompt's branch-local edits into a deterministic row pack."""

    prompt = _canonical_ints(prompt_token_ids, field="prompt_token_ids")
    branch_values = tuple(branches)
    if not branch_values:
        raise ValueError("at least one intervention branch is required")
    intervention_layers = [
        item.layer for branch in branch_values for item in branch.interventions
    ]
    cut = StateCut(
        layer=min(intervention_layers, default=0),
        prompt_sha256=_prompt_sha256(prompt),
    )
    fork = InterventionFork(cut=cut, branches=branch_values)
    candidate_union = tuple(
        dict.fromkeys(
            token for branch in branch_values for token in branch.candidate_token_ids
        )
    )
    offsets = {token: index for index, token in enumerate(candidate_union)}
    projections = tuple(
        tuple(offsets[token] for token in branch.candidate_token_ids) for branch in branch_values
    )
    def freeze_port(port: InterventionPort) -> tuple[FrozenLayerPatchMap, ...]:
        rows = []
        for branch in branch_values:
            by_layer: dict[int, list[ReplayOp]] = {}
            for intervention in branch.interventions:
                if intervention.port is port:
                    by_layer.setdefault(intervention.layer, []).append(
                        intervention.replay_tuple()
                    )
            rows.append(tuple((layer, tuple(by_layer[layer])) for layer in sorted(by_layer)))
        return tuple(rows)

    pack = BranchPack(
        model_identity=model_identity,
        numerical_contract=numerical_contract,
        fork=fork,
        candidate_union=candidate_union,
        candidate_projections=projections,
        row_patch_maps=freeze_port(InterventionPort.MLP_DOWN_INPUT),
        row_head_patch_maps=freeze_port(InterventionPort.ATTENTION_HEAD_OUTPUT),
        row_resid_patch_maps=freeze_port(InterventionPort.RESIDUAL_OUTPUT),
        row_key_patch_maps=freeze_port(InterventionPort.KEY_PROJECTION),
        row_value_patch_maps=freeze_port(InterventionPort.VALUE_PROJECTION),
        payload_specs=tuple(payload_specs),
    )
    return InterventionScienceGraph(prompt_token_ids=prompt, branch_pack=pack)


def execute_intervention_sciencegraph(
    engine: Any,
    graph: InterventionScienceGraph,
    *,
    model_identity: str,
    numerical_contract: str,
    payload_bindings: Mapping[str, Any] | None = None,
    max_branch_batch: int | None = None,
) -> dict[str, Any]:
    """Execute a BranchPack through an exact StateCut or row-local fallback.

    The paged engine materializes the common prefix once, resumes every distinct suffix in one
    weight traversal, and pushes the stable candidate union into ``lm_head``. Other engines may
    use the MLP-only full-logit fallback, but never silently drop unsupported typed ports.
    """

    if not isinstance(graph, InterventionScienceGraph):
        raise TypeError("graph must be an InterventionScienceGraph")
    pack = graph.branch_pack
    if _canonical_id(model_identity, field="model_identity") != pack.model_identity:
        raise ValueError("runtime model identity does not match the BranchPack")
    store = getattr(engine, "store", None)
    if isinstance(getattr(store, "man", None), Mapping):
        actual_model_identity = bind_sciencegraph_model_identity(engine)
        if actual_model_identity != pack.model_identity:
            raise ValueError("loaded engine identity does not match the BranchPack")
    if (
        _canonical_id(numerical_contract, field="numerical_contract")
        != pack.numerical_contract
    ):
        raise ValueError("runtime numerical contract does not match the BranchPack")
    bindings = {} if payload_bindings is None else payload_bindings
    patch_rows, head_rows, resid_rows, key_rows, value_rows = (
        pack.resolve_all_patch_maps(bindings)
    )
    prompt_rows = [graph.prompt_token_ids] * len(pack.fork.branches)
    physical = getattr(engine, "selected_last_intervention_branches", None)
    if callable(physical):
        union_scores, runtime_telemetry = physical(
            graph.prompt_token_ids,
            cut_layer=pack.fork.cut.layer,
            token_ids=pack.candidate_union,
            patch_ops_by_layer_rows=list(patch_rows),
            head_patch_ops_by_layer_rows=list(head_rows),
            resid_patch_ops_by_layer_rows=list(resid_rows),
            key_patch_ops_by_layer_rows=list(key_rows),
            value_patch_ops_by_layer_rows=list(value_rows),
            max_branch_batch=max_branch_batch,
        )
        output_path = "statecut_selected_union"
        suffix_traversals = int(runtime_telemetry.get("suffix_weight_traversals", 1))
        physical_forward_calls = 1 + suffix_traversals
        physical_weight_traversals = 1 + suffix_traversals
    else:
        if any(head_rows) or any(resid_rows) or any(key_rows) or any(value_rows):
            raise NotImplementedError(
                "engine cannot lower one or more typed row-local interventions"
            )
        forward_rows = getattr(engine, "forward_patched_rows", None)
        if not callable(forward_rows):
            raise NotImplementedError("engine has no row-local intervention batch primitive")
        full_rows = forward_rows(prompt_rows, list(patch_rows))
        try:
            import torch

            indices = torch.as_tensor(pack.candidate_union, dtype=torch.long)
            union_scores = torch.stack(
                [row[-1].index_select(0, indices) for row in full_rows]
            )
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("row-local engine returned malformed logits") from exc
        output_path = "full_logits_fallback"
        physical_forward_calls = 1
        physical_weight_traversals = 1
        runtime_telemetry = {"shared_prefix_materialized": False}
    if tuple(union_scores.shape) != (len(pack.fork.branches), len(pack.candidate_union)):
        raise RuntimeError("row-local engine returned the wrong selected-score shape")

    results = []
    for row_index, (branch, projection) in enumerate(
        zip(pack.fork.branches, pack.candidate_projections, strict=True)
    ):
        scores = tuple(float(union_scores[row_index, index]) for index in projection)
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        winner_index = order[0]
        margin = scores[winner_index] - scores[order[1]] if len(order) > 1 else None
        results.append(
            InterventionBranchResult(
                branch_id=branch.branch_id,
                candidate_token_ids=branch.candidate_token_ids,
                scores=scores,
                winner_token_id=branch.candidate_token_ids[winner_index],
                margin=margin,
            )
        )
    result_payload = {
        "schema": SCIENCEGRAPH_EXECUTION_SCHEMA,
        "graph_fingerprint": graph.fingerprint,
        "branch_pack_fingerprint": pack.fingerprint,
        "results": [result.to_dict() for result in results],
        "telemetry": {
            "logical_branches": len(results),
            "physical_forward_calls": physical_forward_calls,
            "physical_weight_traversals": physical_weight_traversals,
            "candidate_union_count": len(pack.candidate_union),
            "model_identity": pack.model_identity,
            "numerical_contract": pack.numerical_contract,
            "output_path": output_path,
            "row_local_intervention_fused": True,
            **runtime_telemetry,
        },
    }
    result_payload["fingerprint"] = _sha256(result_payload)
    return result_payload


def benchmark_intervention_sciencegraph(
    engine: Any,
    graph: InterventionScienceGraph,
    *,
    model_identity: str,
    numerical_contract: str,
    payload_bindings: Mapping[str, Any] | None = None,
    warmups: int = 1,
    repeats: int = 5,
    atol: float = 2e-4,
    min_speedup: float = 1.05,
    max_branch_batch: int | None = None,
) -> dict[str, Any]:
    """Compare complete independent branches with the compiled StateCut graph."""

    if isinstance(warmups, bool) or not isinstance(warmups, Integral) or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats < 3:
        raise ValueError("repeats must be an integer of at least three")
    if not isinstance(atol, Real) or float(atol) < 0:
        raise ValueError("atol must be non-negative")
    if not isinstance(min_speedup, Real) or float(min_speedup) <= 0:
        raise ValueError("min_speedup must be positive")
    bindings = {} if payload_bindings is None else payload_bindings
    pack = graph.branch_pack
    mlp_rows, head_rows, resid_rows, key_rows, value_rows = (
        pack.resolve_all_patch_maps(bindings)
    )
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - mrun requires both packages
        raise RuntimeError("ScienceGraph benchmarking requires numpy and torch") from exc
    prompt = np.asarray(graph.prompt_token_ids, dtype=np.int64)

    def independent() -> tuple[tuple[float, ...], ...]:
        rows = []
        selected_branch = getattr(engine, "selected_last_intervention_branches", None)
        for index, branch in enumerate(pack.fork.branches):
            if callable(selected_branch):
                selected_scores, _telemetry = selected_branch(
                    prompt,
                    cut_layer=0,
                    token_ids=branch.candidate_token_ids,
                    patch_ops_by_layer_rows=[mlp_rows[index]],
                    head_patch_ops_by_layer_rows=[head_rows[index]],
                    resid_patch_ops_by_layer_rows=[resid_rows[index]],
                    key_patch_ops_by_layer_rows=[key_rows[index]],
                    value_patch_ops_by_layer_rows=[value_rows[index]],
                    max_branch_batch=1,
                )
                rows.append(tuple(float(value) for value in selected_scores[0]))
                continue
            kwargs: dict[str, Any] = {"patch_ops_by_layer": mlp_rows[index] or None}
            if head_rows[index]:
                kwargs["head_patch_ops_by_layer"] = head_rows[index]
            if resid_rows[index]:
                kwargs["resid_patch_ops_by_layer"] = resid_rows[index]
            if key_rows[index] or value_rows[index]:
                raise NotImplementedError(
                    "independent K/V benchmarking requires the ScienceGraph branch API"
                )
            output, _acts, _captured = engine.forward_patched(prompt, **kwargs)
            indices = torch.as_tensor(branch.candidate_token_ids, dtype=torch.long)
            rows.append(tuple(float(value) for value in output[-1].index_select(0, indices)))
        return tuple(rows)

    def compiled() -> tuple[tuple[float, ...], ...]:
        result = execute_intervention_sciencegraph(
            engine,
            graph,
            model_identity=model_identity,
            numerical_contract=numerical_contract,
            payload_bindings=bindings,
            max_branch_batch=max_branch_batch,
        )
        return tuple(tuple(row["scores"]) for row in result["results"])

    for _ in range(int(warmups)):
        independent()
        compiled()

    independent_samples = []
    compiled_samples = []
    last_independent: tuple[tuple[float, ...], ...] = ()
    last_compiled: tuple[tuple[float, ...], ...] = ()
    for repeat in range(int(repeats)):
        legs = (("independent", independent), ("compiled", compiled))
        if repeat % 2:
            legs = tuple(reversed(legs))
        for label, function in legs:
            started = time.perf_counter()
            values = function()
            elapsed = time.perf_counter() - started
            if label == "independent":
                independent_samples.append(elapsed)
                last_independent = values
            else:
                compiled_samples.append(elapsed)
                last_compiled = values

    if len(last_independent) != len(last_compiled):
        raise RuntimeError("benchmark legs returned different branch counts")
    if any(
        len(expected) != len(actual)
        for expected, actual in zip(last_independent, last_compiled, strict=True)
    ):
        raise RuntimeError("benchmark legs returned different candidate widths")
    deltas = [
        abs(expected - actual)
        for expected_row, actual_row in zip(last_independent, last_compiled, strict=True)
        for expected, actual in zip(expected_row, actual_row, strict=True)
    ]
    max_abs = max(deltas, default=0.0)
    exact_winners = all(
        max(range(len(expected)), key=lambda index: (expected[index], -index))
        == max(range(len(actual)), key=lambda index: (actual[index], -index))
        for expected, actual in zip(last_independent, last_compiled, strict=True)
    )
    independent_median = statistics.median(independent_samples)
    compiled_median = statistics.median(compiled_samples)
    speedup = independent_median / compiled_median
    paired_speedups = [
        independent_sample / compiled_sample
        for independent_sample, compiled_sample in zip(
            independent_samples, compiled_samples, strict=True
        )
    ]
    log_ratios = [math.log(value) for value in paired_speedups]
    log_mean = statistics.mean(log_ratios)
    log_standard_error = statistics.stdev(log_ratios) / math.sqrt(len(log_ratios))
    paired_lower_95 = math.exp(log_mean - 1.96 * log_standard_error)
    result = {
        "schema": SCIENCEGRAPH_BENCHMARK_SCHEMA,
        "graph_fingerprint": graph.fingerprint,
        "branch_pack_fingerprint": pack.fingerprint,
        "independent_seconds": independent_samples,
        "compiled_seconds": compiled_samples,
        "independent_median_seconds": independent_median,
        "compiled_median_seconds": compiled_median,
        "speedup": speedup,
        "paired_speedups": paired_speedups,
        "paired_speedup_median": statistics.median(paired_speedups),
        "paired_speedup_min": min(paired_speedups),
        "paired_log_ratio_lower_95": paired_lower_95,
        "paired_interval_method": "normal-approximation-on-log-ratios",
        "max_abs_score_delta": max_abs,
        "exact_winners": exact_winners,
        "atol": float(atol),
        "min_speedup": float(min_speedup),
        "qualified": (
            max_abs <= float(atol)
            and exact_winners
            and paired_lower_95 >= float(min_speedup)
        ),
        "logical_branches": len(pack.fork.branches),
        "acquisition": "alternating-independent-compiled",
    }
    result["fingerprint"] = _sha256(result)
    return result


__all__ = [
    "BRANCH_PACK_SCHEMA",
    "SCIENCEGRAPH_BENCHMARK_SCHEMA",
    "SCIENCEGRAPH_EXECUTION_SCHEMA",
    "SCIENCEGRAPH_SCHEMA",
    "BranchPack",
    "Intervention",
    "InterventionBranch",
    "InterventionBranchResult",
    "InterventionFork",
    "InterventionOp",
    "InterventionPort",
    "InterventionScienceGraph",
    "StateCut",
    "TensorPayloadSpec",
    "benchmark_intervention_sciencegraph",
    "bind_sciencegraph_model_identity",
    "compile_intervention_sciencegraph",
    "execute_intervention_sciencegraph",
]
