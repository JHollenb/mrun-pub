"""Physical resource graphs and scoped structural work-floor records.

These records are deliberately narrower than universal complexity lower bounds.  They
describe the parameter and output extent that remains mandatory inside one validated
``OpGraph`` under explicit storage assumptions.  Runtime traffic, elapsed time, RSS,
and energy require independent measurement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from .graph import (
    DemandRewriteCertificate,
    OpGraph,
    OpKind,
    ParameterRef,
    thaw_json_mapping,
)
from .ir import DenseWorkPlan, OutputContract

PHYSICAL_RESOURCE_GRAPH_SCHEMA = "mrun-physical-resource-graph-v1"
WORK_FLOOR_SCHEMA = "mrun-work-floor-v1"
WORK_FLOOR_COMPARISON_SCHEMA = "mrun-work-floor-comparison-v1"

_DTYPE_BITS = {
    "bool": 8,
    "uint8": 8,
    "int8": 8,
    "fp8": 8,
    "int2": 2,
    "int3": 3,
    "int4": 4,
    "fp16": 16,
    "bf16": 16,
    "fp32": 32,
    "fp32-class": 32,
    "int32": 32,
    "int64": 64,
}


def _require_name(value: Any, field_name: str) -> str:
    result = str(value)
    if not result or result.strip() != result:
        raise ValueError(f"{field_name} must be a non-empty string without outer whitespace")
    return result


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, order=True)
class RegionSlice:
    """One byte interval inside a QStore channel."""

    channel: str
    offset: int
    length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", _require_name(self.channel, "region channel"))
        object.__setattr__(self, "offset", int(self.offset))
        object.__setattr__(self, "length", int(self.length))
        if self.offset < 0 or self.length <= 0:
            raise ValueError("region slices require non-negative offsets and positive lengths")

    @property
    def end(self) -> int:
        return self.offset + self.length

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "offset": self.offset,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RegionSlice:
        return cls(
            channel=str(payload["channel"]),
            offset=int(payload["offset"]),
            length=int(payload["length"]),
        )


def _merge_slices(slices: Iterable[RegionSlice]) -> tuple[RegionSlice, ...]:
    merged: list[RegionSlice] = []
    for current in sorted(set(slices)):
        if not merged:
            merged.append(current)
            continue
        previous = merged[-1]
        if current.channel == previous.channel and current.offset <= previous.end:
            merged[-1] = RegionSlice(
                previous.channel,
                previous.offset,
                max(previous.end, current.end) - previous.offset,
            )
        else:
            merged.append(current)
    return tuple(merged)


def _covered_bytes(start: int, end: int, slices: Iterable[RegionSlice]) -> int:
    covered = 0
    for interval in slices:
        overlap_start = max(start, interval.offset)
        overlap_end = min(end, interval.end)
        if overlap_end > overlap_start:
            covered += overlap_end - overlap_start
    return covered


def _region_id(channel: str, offset: int, length: int) -> str:
    return f"{channel}:{offset}:{length}"


@dataclass(frozen=True)
class PhysicalParameterAccess:
    """One graph node's logical access to one physical parameter."""

    access_id: str
    node_id: str
    logical_name: str
    physical_name: str
    parameter_kind: str
    access_mode: str
    row_indices: tuple[int, ...]
    region_ids: tuple[str, ...]
    required_extent_floor_bytes: int
    runtime_dependent: bool

    def __post_init__(self) -> None:
        for field_name in (
            "access_id",
            "node_id",
            "logical_name",
            "physical_name",
            "parameter_kind",
            "access_mode",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        rows = tuple(int(value) for value in self.row_indices)
        if len(rows) != len(set(rows)) or any(row < 0 for row in rows):
            raise ValueError("physical access row indices must be unique and non-negative")
        object.__setattr__(self, "row_indices", rows)
        regions = tuple(str(value) for value in self.region_ids)
        if not regions or len(regions) != len(set(regions)):
            raise ValueError("physical access must reference unique storage regions")
        object.__setattr__(self, "region_ids", regions)
        object.__setattr__(
            self,
            "required_extent_floor_bytes",
            int(self.required_extent_floor_bytes),
        )
        if self.required_extent_floor_bytes <= 0:
            raise ValueError("physical access byte extent must be positive")
        object.__setattr__(self, "runtime_dependent", bool(self.runtime_dependent))
        if self.access_mode == "rows" and not self.row_indices:
            raise ValueError("row-selected physical access requires row indices")
        if self.access_mode != "rows" and self.row_indices:
            raise ValueError("only row-selected physical access may carry row indices")
        if self.runtime_dependent != (self.access_mode == "runtime_rows"):
            raise ValueError("runtime-dependent flag does not match physical access mode")

    def as_dict(self) -> dict[str, Any]:
        return {
            "access_id": self.access_id,
            "node_id": self.node_id,
            "logical_name": self.logical_name,
            "physical_name": self.physical_name,
            "parameter_kind": self.parameter_kind,
            "access_mode": self.access_mode,
            "row_indices": list(self.row_indices),
            "region_ids": list(self.region_ids),
            "required_extent_floor_bytes": self.required_extent_floor_bytes,
            "runtime_dependent": self.runtime_dependent,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhysicalParameterAccess:
        return cls(
            access_id=str(payload["access_id"]),
            node_id=str(payload["node_id"]),
            logical_name=str(payload["logical_name"]),
            physical_name=str(payload["physical_name"]),
            parameter_kind=str(payload["parameter_kind"]),
            access_mode=str(payload["access_mode"]),
            row_indices=tuple(int(value) for value in payload.get("row_indices", ())),
            region_ids=tuple(str(value) for value in payload["region_ids"]),
            required_extent_floor_bytes=int(payload["required_extent_floor_bytes"]),
            runtime_dependent=bool(payload["runtime_dependent"]),
        )


@dataclass(frozen=True)
class PhysicalRegionFloor:
    """Mandatory byte floor for one full physical storage region."""

    region_id: str
    channel: str
    offset: int
    length: int
    access_ids: tuple[str, ...]
    static_slices: tuple[RegionSlice, ...]
    runtime_row_widths: tuple[int, ...]
    static_covered_bytes: int
    runtime_additional_floor_bytes: int
    mandatory_floor_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _require_name(self.region_id, "region_id"))
        object.__setattr__(self, "channel", _require_name(self.channel, "region channel"))
        for field_name in (
            "offset",
            "length",
            "static_covered_bytes",
            "runtime_additional_floor_bytes",
            "mandatory_floor_bytes",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        if self.offset < 0 or self.length <= 0:
            raise ValueError("physical regions require non-negative offsets and positive lengths")
        if self.region_id != _region_id(self.channel, self.offset, self.length):
            raise ValueError("physical region ID does not match its byte interval")
        access_ids = tuple(str(value) for value in self.access_ids)
        if not access_ids or len(access_ids) != len(set(access_ids)):
            raise ValueError("physical region access IDs must be non-empty and unique")
        object.__setattr__(self, "access_ids", access_ids)
        slices = _merge_slices(self.static_slices)
        if slices != self.static_slices:
            raise ValueError("physical region static slices must be canonical and merged")
        if any(
            item.channel != self.channel
            or item.offset < self.offset
            or item.end > self.offset + self.length
            for item in slices
        ):
            raise ValueError("physical region static slice lies outside the full region")
        widths = tuple(sorted(set(int(value) for value in self.runtime_row_widths)))
        if any(width <= 0 or width > self.length for width in widths):
            raise ValueError("runtime row widths must fit inside their physical region")
        object.__setattr__(self, "runtime_row_widths", widths)
        expected_static = sum(item.length for item in slices)
        if self.static_covered_bytes != expected_static:
            raise ValueError("physical region static byte total is inconsistent")
        if self.runtime_additional_floor_bytes < 0:
            raise ValueError("runtime additional byte floor cannot be negative")
        if (
            self.mandatory_floor_bytes
            != self.static_covered_bytes + self.runtime_additional_floor_bytes
            or self.mandatory_floor_bytes > self.length
        ):
            raise ValueError("physical region mandatory byte floor is inconsistent")

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "channel": self.channel,
            "offset": self.offset,
            "length": self.length,
            "access_ids": list(self.access_ids),
            "static_slices": [item.as_dict() for item in self.static_slices],
            "runtime_row_widths": list(self.runtime_row_widths),
            "static_covered_bytes": self.static_covered_bytes,
            "runtime_additional_floor_bytes": self.runtime_additional_floor_bytes,
            "mandatory_floor_bytes": self.mandatory_floor_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhysicalRegionFloor:
        return cls(
            region_id=str(payload["region_id"]),
            channel=str(payload["channel"]),
            offset=int(payload["offset"]),
            length=int(payload["length"]),
            access_ids=tuple(str(value) for value in payload["access_ids"]),
            static_slices=tuple(
                RegionSlice.from_dict(value) for value in payload.get("static_slices", ())
            ),
            runtime_row_widths=tuple(int(value) for value in payload.get("runtime_row_widths", ())),
            static_covered_bytes=int(payload["static_covered_bytes"]),
            runtime_additional_floor_bytes=int(payload["runtime_additional_floor_bytes"]),
            mandatory_floor_bytes=int(payload["mandatory_floor_bytes"]),
        )


@dataclass(frozen=True)
class PhysicalResourceGraph:
    """Hypergraph between logical parameter accesses and physical byte regions."""

    graph_fingerprint: str
    accesses: tuple[PhysicalParameterAccess, ...]
    regions: tuple[PhysicalRegionFloor, ...]
    mandatory_unique_parameter_bytes: int
    per_access_parameter_extent_bytes: int
    unresolved_runtime_access_count: int
    alias_access_count: int
    schema_version: str = PHYSICAL_RESOURCE_GRAPH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PHYSICAL_RESOURCE_GRAPH_SCHEMA:
            raise ValueError(f"unsupported physical-resource graph schema: {self.schema_version}")
        object.__setattr__(
            self,
            "graph_fingerprint",
            _require_name(self.graph_fingerprint, "graph_fingerprint"),
        )
        access_ids = [item.access_id for item in self.accesses]
        region_ids = [item.region_id for item in self.regions]
        if len(access_ids) != len(set(access_ids)):
            raise ValueError("physical-resource graph access IDs must be unique")
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("physical-resource graph region IDs must be unique")
        access_set = set(access_ids)
        region_set = set(region_ids)
        for access in self.accesses:
            if not set(access.region_ids).issubset(region_set):
                raise ValueError("physical access refers to an unknown region")
        for region in self.regions:
            if not set(region.access_ids).issubset(access_set):
                raise ValueError("physical region refers to an unknown access")
        expected_unique = _unique_parameter_floor(self.regions)
        expected_extent = sum(item.required_extent_floor_bytes for item in self.accesses)
        expected_runtime = sum(item.runtime_dependent for item in self.accesses)
        expected_aliases = sum(item.logical_name != item.physical_name for item in self.accesses)
        for field_name, expected in (
            ("mandatory_unique_parameter_bytes", expected_unique),
            ("per_access_parameter_extent_bytes", expected_extent),
            ("unresolved_runtime_access_count", expected_runtime),
            ("alias_access_count", expected_aliases),
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
            if getattr(self, field_name) != expected:
                raise ValueError(f"physical-resource graph {field_name} is inconsistent")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_fingerprint": self.graph_fingerprint,
            "accesses": [item.as_dict() for item in self.accesses],
            "regions": [item.as_dict() for item in self.regions],
            "mandatory_unique_parameter_bytes": self.mandatory_unique_parameter_bytes,
            "per_access_parameter_extent_bytes": self.per_access_parameter_extent_bytes,
            "unresolved_runtime_access_count": self.unresolved_runtime_access_count,
            "alias_access_count": self.alias_access_count,
        }

    @cached_property
    def fingerprint(self) -> str:
        return _hash_payload(self._payload_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._payload_dict(),
            "physical_resource_graph_fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhysicalResourceGraph:
        result = cls(
            graph_fingerprint=str(payload["graph_fingerprint"]),
            accesses=tuple(
                PhysicalParameterAccess.from_dict(value) for value in payload.get("accesses", ())
            ),
            regions=tuple(
                PhysicalRegionFloor.from_dict(value) for value in payload.get("regions", ())
            ),
            mandatory_unique_parameter_bytes=int(payload["mandatory_unique_parameter_bytes"]),
            per_access_parameter_extent_bytes=int(payload["per_access_parameter_extent_bytes"]),
            unresolved_runtime_access_count=int(payload["unresolved_runtime_access_count"]),
            alias_access_count=int(payload["alias_access_count"]),
            schema_version=str(payload.get("schema_version", PHYSICAL_RESOURCE_GRAPH_SCHEMA)),
        )
        claimed = payload.get("physical_resource_graph_fingerprint")
        if claimed is not None and str(claimed) != result.fingerprint:
            raise ValueError("physical-resource graph fingerprint mismatch")
        return result


def _unique_parameter_floor(regions: Iterable[PhysicalRegionFloor]) -> int:
    """Union static bytes and conservatively combine unresolved runtime rows."""

    materialized = tuple(regions)
    static_by_channel: dict[str, list[RegionSlice]] = {}
    runtime_by_channel: dict[
        str,
        list[tuple[int, int, int]],
    ] = {}
    for region in materialized:
        static_by_channel.setdefault(region.channel, []).extend(region.static_slices)
        if region.runtime_row_widths:
            runtime_by_channel.setdefault(region.channel, []).append(
                (
                    region.offset,
                    region.offset + region.length,
                    region.runtime_additional_floor_bytes,
                )
            )
    total = sum(
        item.length for slices in static_by_channel.values() for item in _merge_slices(slices)
    )
    for intervals in runtime_by_channel.values():
        component_end = -1
        component_floor = 0
        for start, end, floor in sorted(intervals):
            if start >= component_end:
                total += component_floor
                component_end = end
                component_floor = floor
            else:
                component_end = max(component_end, end)
                component_floor = max(component_floor, floor)
        total += component_floor
    return total


@dataclass(frozen=True)
class HeadFloorAttainment:
    """Exact output-head demand floor inside the declared row-wise algorithm."""

    scope_node_ids: tuple[str, ...]
    required_evaluation_rows: int
    realized_evaluation_rows: int
    required_vocabulary_rows: int
    realized_vocabulary_rows: int
    required_scalar_outputs: int
    realized_scalar_outputs: int
    required_parameter_extent_bytes: int
    realized_parameter_extent_bytes: int
    complete: bool
    attained: bool
    excluded_claims: tuple[str, ...] = (
        "not-a-whole-model-work-floor",
        "not-runtime-traffic",
        "not-a-latency-floor",
        "not-a-flop-complexity-proof",
        "not-an-energy-floor",
    )

    def __post_init__(self) -> None:
        scope = tuple(str(value) for value in self.scope_node_ids)
        if len(scope) != len(set(scope)):
            raise ValueError("head-floor scope node IDs must be unique")
        object.__setattr__(self, "scope_node_ids", scope)
        for field_name in (
            "required_evaluation_rows",
            "realized_evaluation_rows",
            "required_vocabulary_rows",
            "realized_vocabulary_rows",
            "required_scalar_outputs",
            "realized_scalar_outputs",
            "required_parameter_extent_bytes",
            "realized_parameter_extent_bytes",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
            if getattr(self, field_name) < 0:
                raise ValueError(f"head-floor {field_name} cannot be negative")
        object.__setattr__(self, "complete", bool(self.complete))
        object.__setattr__(self, "attained", bool(self.attained))
        expected_attained = self.complete and (
            self.required_evaluation_rows == self.realized_evaluation_rows
            and self.required_vocabulary_rows == self.realized_vocabulary_rows
            and self.required_scalar_outputs == self.realized_scalar_outputs
            and self.required_parameter_extent_bytes == self.realized_parameter_extent_bytes
        )
        if self.attained != expected_attained:
            raise ValueError("head-floor attainment flag is inconsistent")
        claims = tuple(str(value) for value in self.excluded_claims)
        if not claims or len(claims) != len(set(claims)):
            raise ValueError("head-floor excluded claims must be non-empty and unique")
        object.__setattr__(self, "excluded_claims", claims)

    @property
    def excess_scalar_outputs(self) -> int:
        return self.realized_scalar_outputs - self.required_scalar_outputs

    @property
    def excess_parameter_extent_bytes(self) -> int:
        return self.realized_parameter_extent_bytes - self.required_parameter_extent_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_node_ids": list(self.scope_node_ids),
            "required_evaluation_rows": self.required_evaluation_rows,
            "realized_evaluation_rows": self.realized_evaluation_rows,
            "required_vocabulary_rows": self.required_vocabulary_rows,
            "realized_vocabulary_rows": self.realized_vocabulary_rows,
            "required_scalar_outputs": self.required_scalar_outputs,
            "realized_scalar_outputs": self.realized_scalar_outputs,
            "required_parameter_extent_bytes": self.required_parameter_extent_bytes,
            "realized_parameter_extent_bytes": self.realized_parameter_extent_bytes,
            "excess_scalar_outputs": self.excess_scalar_outputs,
            "excess_parameter_extent_bytes": self.excess_parameter_extent_bytes,
            "complete": self.complete,
            "attained": self.attained,
            "excluded_claims": list(self.excluded_claims),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HeadFloorAttainment:
        result = cls(
            scope_node_ids=tuple(str(value) for value in payload.get("scope_node_ids", ())),
            required_evaluation_rows=int(payload["required_evaluation_rows"]),
            realized_evaluation_rows=int(payload["realized_evaluation_rows"]),
            required_vocabulary_rows=int(payload["required_vocabulary_rows"]),
            realized_vocabulary_rows=int(payload["realized_vocabulary_rows"]),
            required_scalar_outputs=int(payload["required_scalar_outputs"]),
            realized_scalar_outputs=int(payload["realized_scalar_outputs"]),
            required_parameter_extent_bytes=int(payload["required_parameter_extent_bytes"]),
            realized_parameter_extent_bytes=int(payload["realized_parameter_extent_bytes"]),
            complete=bool(payload["complete"]),
            attained=bool(payload["attained"]),
            excluded_claims=tuple(str(value) for value in payload.get("excluded_claims", ())),
        )
        if (
            payload.get("excess_scalar_outputs") is not None
            and int(payload["excess_scalar_outputs"]) != result.excess_scalar_outputs
        ):
            raise ValueError("head-floor excess scalar output count mismatch")
        if (
            payload.get("excess_parameter_extent_bytes") is not None
            and int(payload["excess_parameter_extent_bytes"])
            != result.excess_parameter_extent_bytes
        ):
            raise ValueError("head-floor excess parameter extent mismatch")
        return result


@dataclass(frozen=True)
class WorkFloorCertificate:
    """Scoped structural floor for one operation graph."""

    graph_fingerprint: str
    physical_resource_graph_fingerprint: str
    output_contract: str
    numerical_contract: str
    mandatory_unique_parameter_bytes: int
    per_access_parameter_extent_bytes: int
    required_output_extent_bytes: int
    critical_path_node_count: int
    critical_path_node_ids: tuple[str, ...]
    observable_effect_count: int
    head_demand: HeadFloorAttainment
    assumptions: tuple[str, ...]
    limits: tuple[str, ...]
    floor_scope: str = "declared-opgraph-access-and-output-v1"
    schema_version: str = WORK_FLOOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != WORK_FLOOR_SCHEMA:
            raise ValueError(f"unsupported work-floor schema: {self.schema_version}")
        for field_name in (
            "graph_fingerprint",
            "physical_resource_graph_fingerprint",
            "output_contract",
            "numerical_contract",
            "floor_scope",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        for field_name in (
            "mandatory_unique_parameter_bytes",
            "per_access_parameter_extent_bytes",
            "required_output_extent_bytes",
            "critical_path_node_count",
            "observable_effect_count",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        path = tuple(str(value) for value in self.critical_path_node_ids)
        if self.critical_path_node_count != len(path):
            raise ValueError("critical path node count is inconsistent")
        object.__setattr__(self, "critical_path_node_ids", path)
        for field_name in ("assumptions", "limits"):
            values = tuple(str(value) for value in getattr(self, field_name))
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"work-floor {field_name} must be unique and non-empty")
            object.__setattr__(self, field_name, values)
        if not self.assumptions:
            raise ValueError("work-floor certificate requires explicit assumptions")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_fingerprint": self.graph_fingerprint,
            "physical_resource_graph_fingerprint": (self.physical_resource_graph_fingerprint),
            "output_contract": self.output_contract,
            "numerical_contract": self.numerical_contract,
            "mandatory_unique_parameter_bytes": self.mandatory_unique_parameter_bytes,
            "per_access_parameter_extent_bytes": self.per_access_parameter_extent_bytes,
            "required_output_extent_bytes": self.required_output_extent_bytes,
            "critical_path_node_count": self.critical_path_node_count,
            "critical_path_node_ids": list(self.critical_path_node_ids),
            "observable_effect_count": self.observable_effect_count,
            "head_demand": self.head_demand.as_dict(),
            "assumptions": list(self.assumptions),
            "limits": list(self.limits),
            "floor_scope": self.floor_scope,
        }

    @cached_property
    def fingerprint(self) -> str:
        return _hash_payload(self._payload_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._payload_dict(),
            "work_floor_fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkFloorCertificate:
        result = cls(
            graph_fingerprint=str(payload["graph_fingerprint"]),
            physical_resource_graph_fingerprint=str(payload["physical_resource_graph_fingerprint"]),
            output_contract=str(payload["output_contract"]),
            numerical_contract=str(payload["numerical_contract"]),
            mandatory_unique_parameter_bytes=int(payload["mandatory_unique_parameter_bytes"]),
            per_access_parameter_extent_bytes=int(payload["per_access_parameter_extent_bytes"]),
            required_output_extent_bytes=int(payload["required_output_extent_bytes"]),
            critical_path_node_count=int(payload["critical_path_node_count"]),
            critical_path_node_ids=tuple(
                str(value) for value in payload.get("critical_path_node_ids", ())
            ),
            observable_effect_count=int(payload["observable_effect_count"]),
            head_demand=HeadFloorAttainment.from_dict(payload["head_demand"]),
            assumptions=tuple(str(value) for value in payload.get("assumptions", ())),
            limits=tuple(str(value) for value in payload.get("limits", ())),
            floor_scope=str(
                payload.get(
                    "floor_scope",
                    "declared-opgraph-access-and-output-v1",
                )
            ),
            schema_version=str(payload.get("schema_version", WORK_FLOOR_SCHEMA)),
        )
        claimed = payload.get("work_floor_fingerprint")
        if claimed is not None and str(claimed) != result.fingerprint:
            raise ValueError("work-floor certificate fingerprint mismatch")
        return result


@dataclass(frozen=True)
class WorkFloorAnalysis:
    resources: PhysicalResourceGraph
    certificate: WorkFloorCertificate

    def __post_init__(self) -> None:
        if self.resources.graph_fingerprint != self.certificate.graph_fingerprint:
            raise ValueError("work-floor resources and certificate bind different graphs")
        if self.resources.fingerprint != self.certificate.physical_resource_graph_fingerprint:
            raise ValueError("work-floor certificate does not bind its resource graph")
        if (
            self.resources.mandatory_unique_parameter_bytes
            != self.certificate.mandatory_unique_parameter_bytes
            or self.resources.per_access_parameter_extent_bytes
            != self.certificate.per_access_parameter_extent_bytes
        ):
            raise ValueError("work-floor certificate parameter totals are inconsistent")

    @cached_property
    def fingerprint(self) -> str:
        return _hash_payload(self._payload_dict())

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "resources": self.resources.as_dict(),
            "certificate": self.certificate.as_dict(),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._payload_dict(),
            "work_floor_analysis_fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkFloorAnalysis:
        result = cls(
            resources=PhysicalResourceGraph.from_dict(payload["resources"]),
            certificate=WorkFloorCertificate.from_dict(payload["certificate"]),
        )
        claimed = payload.get("work_floor_analysis_fingerprint")
        if claimed is not None and str(claimed) != result.fingerprint:
            raise ValueError("work-floor analysis fingerprint mismatch")
        return result


@dataclass(frozen=True)
class WorkFloorComparison:
    """Source-versus-rewritten structural floor comparison."""

    source: WorkFloorAnalysis
    rewritten: WorkFloorAnalysis
    output_contract: str
    rewrite_ids: tuple[str, ...]
    comparison_basis: str = "structural-extent-not-runtime-traffic"
    schema_version: str = WORK_FLOOR_COMPARISON_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != WORK_FLOOR_COMPARISON_SCHEMA:
            raise ValueError(f"unsupported work-floor comparison schema: {self.schema_version}")
        object.__setattr__(
            self,
            "output_contract",
            _require_name(self.output_contract, "output_contract"),
        )
        object.__setattr__(
            self,
            "comparison_basis",
            _require_name(self.comparison_basis, "comparison_basis"),
        )
        rewrite_ids = tuple(str(value) for value in self.rewrite_ids)
        if len(rewrite_ids) != len(set(rewrite_ids)):
            raise ValueError("work-floor rewrite IDs must be unique")
        object.__setattr__(self, "rewrite_ids", rewrite_ids)
        if self.rewritten.certificate.output_contract != self.output_contract:
            raise ValueError("rewritten work floor has the wrong output contract")

    @property
    def source_minus_rewritten_unique_parameter_bytes(self) -> int:
        return (
            self.source.certificate.mandatory_unique_parameter_bytes
            - self.rewritten.certificate.mandatory_unique_parameter_bytes
        )

    @property
    def source_minus_rewritten_per_access_parameter_bytes(self) -> int:
        return (
            self.source.certificate.per_access_parameter_extent_bytes
            - self.rewritten.certificate.per_access_parameter_extent_bytes
        )

    @property
    def source_minus_rewritten_output_bytes(self) -> int:
        return (
            self.source.certificate.required_output_extent_bytes
            - self.rewritten.certificate.required_output_extent_bytes
        )

    @property
    def source_minus_rewritten_critical_path_nodes(self) -> int:
        return (
            self.source.certificate.critical_path_node_count
            - self.rewritten.certificate.critical_path_node_count
        )

    @property
    def source_head_excess_scalar_outputs(self) -> int:
        return self.source.certificate.head_demand.excess_scalar_outputs

    @property
    def rewritten_head_excess_scalar_outputs(self) -> int:
        return self.rewritten.certificate.head_demand.excess_scalar_outputs

    @property
    def source_head_excess_parameter_extent_bytes(self) -> int:
        return self.source.certificate.head_demand.excess_parameter_extent_bytes

    @property
    def rewritten_head_excess_parameter_extent_bytes(self) -> int:
        return self.rewritten.certificate.head_demand.excess_parameter_extent_bytes

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.as_dict(),
            "rewritten": self.rewritten.as_dict(),
            "output_contract": self.output_contract,
            "rewrite_ids": list(self.rewrite_ids),
            "comparison_basis": self.comparison_basis,
            "source_minus_rewritten_unique_parameter_bytes": (
                self.source_minus_rewritten_unique_parameter_bytes
            ),
            "source_minus_rewritten_per_access_parameter_bytes": (
                self.source_minus_rewritten_per_access_parameter_bytes
            ),
            "source_minus_rewritten_output_bytes": (self.source_minus_rewritten_output_bytes),
            "source_minus_rewritten_critical_path_nodes": (
                self.source_minus_rewritten_critical_path_nodes
            ),
            "source_head_excess_scalar_outputs": (self.source_head_excess_scalar_outputs),
            "rewritten_head_excess_scalar_outputs": (self.rewritten_head_excess_scalar_outputs),
            "source_head_excess_parameter_extent_bytes": (
                self.source_head_excess_parameter_extent_bytes
            ),
            "rewritten_head_excess_parameter_extent_bytes": (
                self.rewritten_head_excess_parameter_extent_bytes
            ),
        }

    @cached_property
    def fingerprint(self) -> str:
        return _hash_payload(self._payload_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._payload_dict(),
            "work_floor_comparison_fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkFloorComparison:
        result = cls(
            source=WorkFloorAnalysis.from_dict(payload["source"]),
            rewritten=WorkFloorAnalysis.from_dict(payload["rewritten"]),
            output_contract=str(payload["output_contract"]),
            rewrite_ids=tuple(str(value) for value in payload.get("rewrite_ids", ())),
            comparison_basis=str(
                payload.get(
                    "comparison_basis",
                    "structural-extent-not-runtime-traffic",
                )
            ),
            schema_version=str(payload.get("schema_version", WORK_FLOOR_COMPARISON_SCHEMA)),
        )
        claimed = payload.get("work_floor_comparison_fingerprint")
        if claimed is not None and str(claimed) != result.fingerprint:
            raise ValueError("work-floor comparison fingerprint mismatch")
        for field_name in (
            "source_minus_rewritten_unique_parameter_bytes",
            "source_minus_rewritten_per_access_parameter_bytes",
            "source_minus_rewritten_output_bytes",
            "source_minus_rewritten_critical_path_nodes",
            "source_head_excess_scalar_outputs",
            "rewritten_head_excess_scalar_outputs",
            "source_head_excess_parameter_extent_bytes",
            "rewritten_head_excess_parameter_extent_bytes",
        ):
            claimed_value = payload.get(field_name)
            if claimed_value is not None and int(claimed_value) != getattr(result, field_name):
                raise ValueError(f"work-floor comparison {field_name} mismatch")
        return result


def _parameter_region_slices(
    parameter: ParameterRef,
) -> tuple[
    dict[tuple[str, int, int], tuple[RegionSlice, ...]],
    dict[tuple[str, int, int], int],
    int,
]:
    static: dict[tuple[str, int, int], tuple[RegionSlice, ...]] = {}
    runtime_widths: dict[tuple[str, int, int], int] = {}
    extent = 0
    rows = int(parameter.shape[0])
    for channel, offset, length in parameter.regions:
        key = (channel, offset, length)
        if parameter.access == "all":
            static[key] = (RegionSlice(channel, offset, length),)
            extent += length
            continue
        if length % rows:
            raise ValueError(
                f"row-addressed region for {parameter.logical_name!r} does not divide "
                f"evenly across {rows} rows"
            )
        row_width = length // rows
        if parameter.access == "rows":
            static[key] = _merge_slices(
                RegionSlice(channel, offset + row * row_width, row_width)
                for row in parameter.row_indices
            )
        elif parameter.access == "runtime_rows":
            runtime_widths[key] = row_width
        else:  # pragma: no cover - ParameterRef already validates this
            raise ValueError(f"unknown parameter access mode: {parameter.access}")
        extent += row_width * (len(parameter.row_indices) if parameter.access == "rows" else 1)
    return static, runtime_widths, extent


def _runtime_additional_floor(
    *,
    offset: int,
    length: int,
    static_slices: tuple[RegionSlice, ...],
    row_widths: tuple[int, ...],
) -> int:
    additions: list[int] = []
    for width in row_widths:
        if length % width:
            raise ValueError("runtime row width does not divide its physical region")
        minimum = width
        for row_offset in range(offset, offset + length, width):
            uncovered = width - _covered_bytes(
                row_offset,
                row_offset + width,
                static_slices,
            )
            minimum = min(minimum, uncovered)
            if minimum == 0:
                break
        additions.append(minimum)
    # Multiple runtime accesses may legally name the same row.  The maximum of
    # their individual minima is a conservative unique-byte floor.
    return max(additions, default=0)


def build_physical_resource_graph(graph: OpGraph) -> PhysicalResourceGraph:
    """Resolve graph parameter accesses into a deterministic physical hypergraph."""

    accesses: list[PhysicalParameterAccess] = []
    region_accesses: dict[tuple[str, int, int], list[str]] = {}
    region_static: dict[tuple[str, int, int], list[RegionSlice]] = {}
    region_runtime_widths: dict[tuple[str, int, int], list[int]] = {}

    for node in graph.nodes:
        for parameter_index, parameter in enumerate(node.parameters):
            access_id = f"{node.node_id}:{parameter_index}:{parameter.logical_name}"
            static, runtime_widths, extent = _parameter_region_slices(parameter)
            keys = tuple(parameter.regions)
            for key in keys:
                region_accesses.setdefault(key, []).append(access_id)
                region_static.setdefault(key, []).extend(static.get(key, ()))
                if key in runtime_widths:
                    region_runtime_widths.setdefault(key, []).append(runtime_widths[key])
            accesses.append(
                PhysicalParameterAccess(
                    access_id=access_id,
                    node_id=node.node_id,
                    logical_name=parameter.logical_name,
                    physical_name=parameter.physical_name,
                    parameter_kind=parameter.kind,
                    access_mode=parameter.access,
                    row_indices=parameter.row_indices,
                    region_ids=tuple(
                        _region_id(channel, offset, length) for channel, offset, length in keys
                    ),
                    required_extent_floor_bytes=extent,
                    runtime_dependent=parameter.access == "runtime_rows",
                )
            )

    keys = sorted(region_accesses)
    global_static_by_channel: dict[str, tuple[RegionSlice, ...]] = {}
    for channel, _, _ in keys:
        if channel in global_static_by_channel:
            continue
        global_static_by_channel[channel] = _merge_slices(
            item for key, values in region_static.items() if key[0] == channel for item in values
        )

    regions: list[PhysicalRegionFloor] = []
    for channel, offset, length in keys:
        key = (channel, offset, length)
        static_slices = _merge_slices(region_static.get(key, ()))
        widths = tuple(sorted(set(region_runtime_widths.get(key, ()))))
        static_bytes = sum(item.length for item in static_slices)
        runtime_additional = _runtime_additional_floor(
            offset=offset,
            length=length,
            static_slices=global_static_by_channel[channel],
            row_widths=widths,
        )
        regions.append(
            PhysicalRegionFloor(
                region_id=_region_id(channel, offset, length),
                channel=channel,
                offset=offset,
                length=length,
                access_ids=tuple(region_accesses[key]),
                static_slices=static_slices,
                runtime_row_widths=widths,
                static_covered_bytes=static_bytes,
                runtime_additional_floor_bytes=runtime_additional,
                mandatory_floor_bytes=static_bytes + runtime_additional,
            )
        )

    regions_tuple = tuple(regions)
    return PhysicalResourceGraph(
        graph_fingerprint=graph.fingerprint,
        accesses=tuple(accesses),
        regions=regions_tuple,
        mandatory_unique_parameter_bytes=_unique_parameter_floor(regions_tuple),
        per_access_parameter_extent_bytes=sum(
            item.required_extent_floor_bytes for item in accesses
        ),
        unresolved_runtime_access_count=sum(item.runtime_dependent for item in accesses),
        alias_access_count=sum(item.logical_name != item.physical_name for item in accesses),
    )


def _output_extent_bytes(graph: OpGraph) -> int:
    tensors = graph.tensor_map
    bits = 0
    for output_id in graph.outputs:
        tensor = tensors[output_id]
        dtype_bits = _DTYPE_BITS.get(tensor.dtype.lower())
        if dtype_bits is None:
            raise ValueError(f"work-floor output dtype {tensor.dtype!r} has no byte-width rule")
        bits += tensor.element_count * dtype_bits
    return (bits + 7) // 8


def _critical_path(graph: OpGraph) -> tuple[str, ...]:
    producers = graph.producer_map
    paths: dict[str, tuple[str, ...]] = {}
    for node in graph.nodes:
        predecessor_ids = [producers[value_id] for value_id in node.inputs if value_id in producers]
        predecessor_ids.extend(node.control_inputs)
        if predecessor_ids:
            predecessor = max(
                predecessor_ids,
                key=lambda value: (len(paths[value]), paths[value]),
            )
            paths[node.node_id] = (*paths[predecessor], node.node_id)
        else:
            paths[node.node_id] = (node.node_id,)
    return max(paths.values(), key=lambda value: (len(value), value), default=())


def _required_head_demand(
    plan: DenseWorkPlan,
    vocabulary_size: int,
) -> tuple[int, int, int]:
    if plan.output_contract in {
        OutputContract.FULL_LOGITS,
        OutputContract.LOSS_ONLY,
    }:
        evaluation_rows = plan.shape.live_token_rows
        vocabulary_rows = vocabulary_size
        return evaluation_rows, vocabulary_rows, evaluation_rows * vocabulary_rows
    if plan.output_contract is OutputContract.LAST_TOKEN_LOGITS:
        evaluation_rows = plan.shape.actual_batch
        vocabulary_rows = vocabulary_size
        return evaluation_rows, vocabulary_rows, evaluation_rows * vocabulary_rows
    if plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
        evaluation_rows = plan.shape.actual_batch
        vocabulary_rows = len(plan.required_output_rows)
        return evaluation_rows, vocabulary_rows, evaluation_rows * vocabulary_rows
    if plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
        evaluation_rows = plan.shape.actual_batch
        vocabulary_rows = len({token_id for row in plan.candidate_token_ids for token_id in row})
        # The exact public demand is ragged: each batch row needs only its own
        # distinct candidate logits.  B * |union| is the current dense-union
        # realization, not a lower bound when candidate sets differ by row.
        scalar_outputs = sum(len(set(row)) for row in plan.candidate_token_ids)
        return evaluation_rows, vocabulary_rows, scalar_outputs
    if plan.output_contract is OutputContract.HIDDEN_STATE_ONLY:
        return 0, 0, 0
    raise ValueError(f"unsupported head-floor output contract: {plan.output_contract.value}")


def _head_floor(graph: OpGraph, plan: DenseWorkPlan) -> HeadFloorAttainment:
    output_nodes = tuple(node for node in graph.nodes if node.node_id.startswith("output."))
    head_nodes = tuple(node for node in graph.nodes if node.kind is OpKind.VOCAB_PROJECTION)
    metadata = thaw_json_mapping(graph.metadata)
    vocabulary_size = int(metadata.get("vocabulary_size", 0))
    if not vocabulary_size and head_nodes and head_nodes[0].parameters:
        vocabulary_size = int(head_nodes[0].parameters[0].shape[0])
    (
        required_evaluations,
        required_vocabulary_rows,
        required_scalars,
    ) = _required_head_demand(
        plan,
        vocabulary_size,
    )

    if plan.output_contract is OutputContract.HIDDEN_STATE_ONLY and not head_nodes:
        return HeadFloorAttainment(
            scope_node_ids=tuple(node.node_id for node in output_nodes),
            required_evaluation_rows=0,
            realized_evaluation_rows=0,
            required_vocabulary_rows=0,
            realized_vocabulary_rows=0,
            required_scalar_outputs=0,
            realized_scalar_outputs=0,
            required_parameter_extent_bytes=0,
            realized_parameter_extent_bytes=0,
            complete=True,
            attained=True,
        )
    if len(head_nodes) != 1 or len(head_nodes[0].parameters) != 1:
        return HeadFloorAttainment(
            scope_node_ids=tuple(node.node_id for node in output_nodes),
            required_evaluation_rows=required_evaluations,
            realized_evaluation_rows=0,
            required_vocabulary_rows=required_vocabulary_rows,
            realized_vocabulary_rows=0,
            required_scalar_outputs=required_scalars,
            realized_scalar_outputs=0,
            required_parameter_extent_bytes=0,
            realized_parameter_extent_bytes=0,
            complete=False,
            attained=False,
        )

    head = head_nodes[0]
    parameter = head.parameters[0]
    if not head.inputs:
        raise ValueError("vocabulary projection has no activation input")
    input_tensor = graph.tensor_map[head.inputs[0]]
    if len(input_tensor.shape) < 2:
        raise ValueError("vocabulary projection input must include a hidden dimension")
    realized_evaluations = 1
    for dimension in input_tensor.shape[:-1]:
        realized_evaluations *= dimension
    if parameter.access == "all":
        realized_vocabulary_rows = int(parameter.shape[0])
    elif parameter.access == "rows":
        realized_vocabulary_rows = len(parameter.row_indices)
    else:
        realized_vocabulary_rows = 0
    complete = parameter.access in {"all", "rows"}

    full_parameter_bytes = parameter.physical_bytes
    parameter_rows = int(parameter.shape[0])
    if full_parameter_bytes % parameter_rows:
        complete = False
        required_parameter_bytes = 0
        realized_parameter_bytes = 0
    else:
        row_bytes = full_parameter_bytes // parameter_rows
        required_parameter_bytes = required_vocabulary_rows * row_bytes
        realized_parameter_bytes = realized_vocabulary_rows * row_bytes

    return HeadFloorAttainment(
        scope_node_ids=tuple(node.node_id for node in output_nodes),
        required_evaluation_rows=required_evaluations,
        realized_evaluation_rows=realized_evaluations,
        required_vocabulary_rows=required_vocabulary_rows,
        realized_vocabulary_rows=realized_vocabulary_rows,
        required_scalar_outputs=required_scalars,
        realized_scalar_outputs=realized_evaluations * realized_vocabulary_rows,
        required_parameter_extent_bytes=required_parameter_bytes,
        realized_parameter_extent_bytes=realized_parameter_bytes,
        complete=complete,
        attained=complete
        and required_evaluations == realized_evaluations
        and required_vocabulary_rows == realized_vocabulary_rows
        and required_scalars == realized_evaluations * realized_vocabulary_rows,
    )


def analyze_work_floor(
    graph: OpGraph,
    plan: DenseWorkPlan,
) -> WorkFloorAnalysis:
    """Build a scoped structural floor for ``graph``."""

    resources = build_physical_resource_graph(graph)
    output_contract = plan.output_contract.value
    critical_path = _critical_path(graph)
    effects = sum(len(node.effects) for node in graph.nodes)
    limits: list[str] = []
    if resources.unresolved_runtime_access_count:
        limits.append("runtime-parameter-row-identities-unresolved")
    if output_contract == "full_logits":
        limits.append("full-logits-retain-full-vocabulary-output")
    elif output_contract == "loss_only":
        limits.append("loss-retains-full-vocabulary-normalizer")
    if any(
        parameter.access == "all" and parameter.kind.startswith("qrow")
        for parameter in graph.parameter_refs
    ):
        limits.append("dense-full-parameter-accesses-remain")
    if effects:
        limits.append("observable-effects-remain")

    certificate = WorkFloorCertificate(
        graph_fingerprint=graph.fingerprint,
        physical_resource_graph_fingerprint=resources.fingerprint,
        output_contract=output_contract,
        numerical_contract=graph.numerical_contract,
        mandatory_unique_parameter_bytes=resources.mandatory_unique_parameter_bytes,
        per_access_parameter_extent_bytes=(resources.per_access_parameter_extent_bytes),
        required_output_extent_bytes=_output_extent_bytes(graph),
        critical_path_node_count=len(critical_path),
        critical_path_node_ids=critical_path,
        observable_effect_count=effects,
        head_demand=_head_floor(graph, plan),
        assumptions=(
            "parameter-regions-use-one-address-space-per-channel",
            "row-addressed-regions-are-uniformly-partitioned",
            "runtime-row-accesses-require-at-least-one-row",
            "unique-parameter-floor-assumes-empty-residency-and-perfect-retention",
            "parameter-extents-are-structural-not-observed-traffic",
            "output-extent-is-logical-public-output-not-observed-write-traffic",
            "critical-path-is-node-count-not-time",
        ),
        limits=tuple(limits),
    )
    return WorkFloorAnalysis(resources=resources, certificate=certificate)


def compare_work_floors(
    source_graph: OpGraph,
    rewritten_graph: OpGraph,
    plan: DenseWorkPlan,
    rewrite_certificate: DemandRewriteCertificate | None = None,
) -> WorkFloorComparison:
    """Compare source and rewritten structural floors with fingerprint binding."""

    rewrite_ids: tuple[str, ...] = ()
    output_contract = plan.output_contract.value
    if rewrite_certificate is not None:
        if rewrite_certificate.source_fingerprint != source_graph.fingerprint:
            raise ValueError("work-floor rewrite certificate source mismatch")
        if rewrite_certificate.rewritten_fingerprint != rewritten_graph.fingerprint:
            raise ValueError("work-floor rewrite certificate graph mismatch")
        if rewrite_certificate.output_contract != output_contract:
            raise ValueError("work-floor rewrite certificate output mismatch")
        rewrite_ids = rewrite_certificate.rewrite_ids
    elif source_graph.fingerprint != rewritten_graph.fingerprint:
        raise ValueError("different work-floor graphs require a rewrite certificate")

    return WorkFloorComparison(
        source=analyze_work_floor(source_graph, plan),
        rewritten=analyze_work_floor(rewritten_graph, plan),
        output_contract=output_contract,
        rewrite_ids=rewrite_ids,
    )


__all__ = [
    "PHYSICAL_RESOURCE_GRAPH_SCHEMA",
    "WORK_FLOOR_COMPARISON_SCHEMA",
    "WORK_FLOOR_SCHEMA",
    "PhysicalParameterAccess",
    "PhysicalRegionFloor",
    "PhysicalResourceGraph",
    "RegionSlice",
    "WorkFloorAnalysis",
    "WorkFloorCertificate",
    "WorkFloorComparison",
    "analyze_work_floor",
    "build_physical_resource_graph",
    "compare_work_floors",
]
