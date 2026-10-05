"""Deterministic graph analyses for the mrun operation graph.

These passes intentionally stop short of backend code generation.  They answer four
questions that are useful before accelerator hardware is available:

* which nodes are demanded by an output contract;
* which pure, single-consumer regions are safe fusion candidates;
* which tensor lifetimes can share storage; and
* how a graph should be split between two backends under explicit unary and transfer
  costs.

The algorithms are conservative.  In particular, effectful/control-dependent nodes
form fusion barriers, liveness intervals are inclusive, and binary placement is called
``exact`` only because its stated pairwise objective is solved by an s-t minimum cut.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from typing import Any

from .graph import (
    DemandRewriteCertificate,
    OpGraph,
    OpNode,
    StorageClass,
    TensorSpec,
)

GRAPH_COMPILATION_SCHEMA = "mrun-graph-compilation-v1"
_FLOW_EPSILON = 1e-12


def _enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _params(node: OpNode) -> dict[str, Any]:
    return {str(key): value for key, value in node.params}


def _ordered_nodes(graph: OpGraph) -> tuple[OpNode, ...]:
    """Return the graph's already validated deterministic topological order."""

    return tuple(graph.nodes)


def _value_map(graph: OpGraph) -> dict[str, TensorSpec]:
    return {value.value_id: value for value in graph.values}


def _producer_map(graph: OpGraph) -> dict[str, str]:
    return dict(graph.producer_map)


def _consumer_map(graph: OpGraph) -> dict[str, tuple[str, ...]]:
    return {value_id: tuple(consumers) for value_id, consumers in graph.consumer_map.items()}


def _node_is_barrier(node: OpNode) -> bool:
    """Recognize explicit and semantic fusion barriers conservatively."""

    if node.effects or node.control_inputs:
        return True
    params = _params(node)
    if bool(params.get("barrier", False)) or bool(params.get("fusion_barrier", False)):
        return True
    kind = _enum_value(node.kind).lower()
    barrier_fragments = (
        "barrier",
        "synchron",
        "state_read",
        "state_write",
        "commit",
        "rollback",
        "transfer",
        "dispatch",
    )
    return any(fragment in kind for fragment in barrier_fragments)


def _alias_closure(
    value_ids: Iterable[str],
    values: Mapping[str, TensorSpec],
) -> set[str]:
    kept = set(value_ids)
    pending = list(kept)
    while pending:
        value_id = pending.pop()
        value = values.get(value_id)
        if value is None or value.alias_of is None or value.alias_of in kept:
            continue
        kept.add(value.alias_of)
        pending.append(value.alias_of)
    return kept


@dataclass(frozen=True)
class SliceReport:
    roots: tuple[str, ...]
    retained_effect_nodes: tuple[str, ...]
    retained_barrier_nodes: tuple[str, ...]
    kept_node_ids: tuple[str, ...]
    removed_node_ids: tuple[str, ...]
    kept_value_ids: tuple[str, ...]
    removed_value_ids: tuple[str, ...]

    @property
    def nodes_removed(self) -> int:
        return len(self.removed_node_ids)

    @property
    def values_removed(self) -> int:
        return len(self.removed_value_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "roots": list(self.roots),
            "retained_effect_nodes": list(self.retained_effect_nodes),
            "retained_barrier_nodes": list(self.retained_barrier_nodes),
            "kept_node_ids": list(self.kept_node_ids),
            "removed_node_ids": list(self.removed_node_ids),
            "kept_value_ids": list(self.kept_value_ids),
            "removed_value_ids": list(self.removed_value_ids),
            "nodes_removed": self.nodes_removed,
            "values_removed": self.values_removed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SliceReport:
        return cls(
            roots=tuple(str(value) for value in payload["roots"]),
            retained_effect_nodes=tuple(
                str(value) for value in payload.get("retained_effect_nodes", ())
            ),
            retained_barrier_nodes=tuple(
                str(value) for value in payload.get("retained_barrier_nodes", ())
            ),
            kept_node_ids=tuple(str(value) for value in payload["kept_node_ids"]),
            removed_node_ids=tuple(str(value) for value in payload["removed_node_ids"]),
            kept_value_ids=tuple(str(value) for value in payload["kept_value_ids"]),
            removed_value_ids=tuple(str(value) for value in payload["removed_value_ids"]),
        )


@dataclass(frozen=True)
class BackwardSliceResult:
    graph: OpGraph
    report: SliceReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.as_dict(),
            "report": self.report.as_dict(),
        }


def backward_slice(
    graph: OpGraph,
    roots: Iterable[str] | None = None,
    *,
    retain_effects: bool = True,
    retain_barriers: bool = True,
) -> BackwardSliceResult:
    """Slice ``graph`` backwards from tensor roots.

    Effectful nodes and explicit barriers are retained by default even when their
    values are not demanded.  Their data and control predecessors are retained too,
    which preserves the prerequisites of state transitions rather than merely keeping
    a disconnected side-effect node.
    """

    nodes = _ordered_nodes(graph)
    node_by_id = {node.node_id: node for node in nodes}
    values = _value_map(graph)
    producers = _producer_map(graph)
    selected_roots = tuple(dict.fromkeys(graph.outputs if roots is None else roots))
    if not selected_roots:
        raise ValueError("backward slice requires at least one root value")
    unknown_roots = [root for root in selected_roots if root not in values]
    if unknown_roots:
        raise ValueError(f"unknown graph roots: {', '.join(unknown_roots)}")

    kept_nodes: set[str] = set()
    demanded_values: set[str] = set(selected_roots)
    pending_values = list(reversed(selected_roots))
    pending_nodes: list[str] = []

    effect_nodes = tuple(node.node_id for node in nodes if node.effects)
    barrier_nodes = tuple(node.node_id for node in nodes if _node_is_barrier(node))
    if retain_effects:
        pending_nodes.extend(reversed(effect_nodes))
    if retain_barriers:
        pending_nodes.extend(reversed(barrier_nodes))

    def demand_node(node_id: str) -> None:
        if node_id in kept_nodes:
            return
        node = node_by_id.get(node_id)
        if node is None:
            raise ValueError(f"unknown control dependency: {node_id}")
        kept_nodes.add(node_id)
        pending_values.extend(reversed(node.inputs))
        pending_nodes.extend(reversed(node.control_inputs))

    while pending_values or pending_nodes:
        while pending_nodes:
            demand_node(pending_nodes.pop())
        if not pending_values:
            continue
        value_id = pending_values.pop()
        value = values[value_id]
        if value.alias_of is not None and value.alias_of not in demanded_values:
            demanded_values.add(value.alias_of)
            pending_values.append(value.alias_of)
        producer_id = producers.get(value_id)
        if producer_id is not None:
            demand_node(producer_id)

    kept_node_tuple = tuple(node.node_id for node in nodes if node.node_id in kept_nodes)
    kept_node_set = set(kept_node_tuple)
    referenced_values = set(selected_roots)
    for node in nodes:
        if node.node_id not in kept_node_set:
            continue
        referenced_values.update(node.inputs)
        referenced_values.update(node.outputs)
    referenced_values = _alias_closure(referenced_values, values)
    selected_inputs = tuple(value_id for value_id in graph.inputs if value_id in referenced_values)
    if not selected_inputs:
        # OpGraph deliberately requires at least one declared input.  A retained
        # effect-only component may have no data dependency, so retain the first
        # caller-owned input as a harmless signature anchor.
        selected_inputs = graph.inputs[:1]
        referenced_values.update(selected_inputs)
    kept_value_tuple = tuple(
        value.value_id for value in graph.values if value.value_id in referenced_values
    )
    removed_node_tuple = tuple(node.node_id for node in nodes if node.node_id not in kept_node_set)
    removed_value_tuple = tuple(
        value.value_id for value in graph.values if value.value_id not in referenced_values
    )

    sliced = replace(
        graph,
        nodes=tuple(node for node in nodes if node.node_id in kept_node_set),
        values=tuple(value for value in graph.values if value.value_id in referenced_values),
        inputs=selected_inputs,
        outputs=selected_roots,
    )
    report = SliceReport(
        roots=selected_roots,
        retained_effect_nodes=tuple(
            node_id for node_id in effect_nodes if node_id in kept_node_set
        ),
        retained_barrier_nodes=tuple(
            node_id for node_id in barrier_nodes if node_id in kept_node_set
        ),
        kept_node_ids=kept_node_tuple,
        removed_node_ids=removed_node_tuple,
        kept_value_ids=kept_value_tuple,
        removed_value_ids=removed_value_tuple,
    )
    return BackwardSliceResult(graph=sliced, report=report)


@dataclass(frozen=True)
class FusionRegion:
    region_id: str
    node_ids: tuple[str, ...]
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    parameter_names: tuple[str, ...]

    @property
    def launches_removed(self) -> int:
        return max(0, len(self.node_ids) - 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "node_ids": list(self.node_ids),
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "parameter_names": list(self.parameter_names),
            "launches_removed": self.launches_removed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FusionRegion:
        return cls(
            region_id=str(payload["region_id"]),
            node_ids=tuple(str(value) for value in payload["node_ids"]),
            input_value_ids=tuple(str(value) for value in payload["input_value_ids"]),
            output_value_ids=tuple(str(value) for value in payload["output_value_ids"]),
            parameter_names=tuple(str(value) for value in payload["parameter_names"]),
        )


@dataclass(frozen=True)
class FusionPlan:
    regions: tuple[FusionRegion, ...]
    unfused_node_ids: tuple[str, ...]
    barrier_node_ids: tuple[str, ...]
    candidate_edges: int
    accepted_edges: int

    @property
    def launches_removed(self) -> int:
        return sum(region.launches_removed for region in self.regions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "regions": [region.as_dict() for region in self.regions],
            "unfused_node_ids": list(self.unfused_node_ids),
            "barrier_node_ids": list(self.barrier_node_ids),
            "candidate_edges": self.candidate_edges,
            "accepted_edges": self.accepted_edges,
            "launches_removed": self.launches_removed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FusionPlan:
        return cls(
            regions=tuple(FusionRegion.from_dict(region) for region in payload.get("regions", ())),
            unfused_node_ids=tuple(str(value) for value in payload.get("unfused_node_ids", ())),
            barrier_node_ids=tuple(str(value) for value in payload.get("barrier_node_ids", ())),
            candidate_edges=int(payload.get("candidate_edges", 0)),
            accepted_edges=int(payload.get("accepted_edges", 0)),
        )


def _node_edges(
    graph: OpGraph,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    nodes = _ordered_nodes(graph)
    successors = {node.node_id: set() for node in nodes}
    predecessors = {node.node_id: set() for node in nodes}
    producers = _producer_map(graph)
    for node in nodes:
        for value_id in node.inputs:
            producer_id = producers.get(value_id)
            if producer_id is None:
                continue
            successors[producer_id].add(node.node_id)
            predecessors[node.node_id].add(producer_id)
        for control_id in node.control_inputs:
            successors[control_id].add(node.node_id)
            predecessors[node.node_id].add(control_id)
    return successors, predecessors


def _transitive_successors(
    nodes: Sequence[OpNode],
    successors: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    reachable = {node.node_id: set() for node in nodes}
    for node in reversed(nodes):
        node_reachable = reachable[node.node_id]
        for successor in successors[node.node_id]:
            node_reachable.add(successor)
            node_reachable.update(reachable[successor])
    return reachable


def _is_connected(
    node_ids: set[str],
    successors: Mapping[str, set[str]],
    predecessors: Mapping[str, set[str]],
) -> bool:
    if not node_ids:
        return False
    seen: set[str] = set()
    pending = [min(node_ids)]
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        pending.extend((successors[node_id] | predecessors[node_id]) & node_ids)
    return seen == node_ids


def _is_convex(
    node_ids: set[str],
    all_node_ids: set[str],
    reachable: Mapping[str, set[str]],
) -> bool:
    """A set is convex when no path between two members leaves the set."""

    outside = all_node_ids - node_ids
    for candidate in outside:
        has_predecessor_in_region = any(candidate in reachable[member] for member in node_ids)
        if not has_predecessor_in_region:
            continue
        if any(member in reachable[candidate] for member in node_ids):
            return False
    return True


def _internal_values_are_single_consumer(
    node_ids: set[str],
    node_by_id: Mapping[str, OpNode],
    consumers: Mapping[str, tuple[str, ...]],
    graph_outputs: set[str],
) -> bool:
    for node_id in node_ids:
        for value_id in node_by_id[node_id].outputs:
            internal_consumers = tuple(
                consumer for consumer in consumers.get(value_id, ()) if consumer in node_ids
            )
            if not internal_consumers:
                continue
            if value_id in graph_outputs:
                return False
            if len(consumers.get(value_id, ())) != 1 or len(internal_consumers) != 1:
                return False
    return True


def _make_fusion_region(
    graph: OpGraph,
    node_ids: tuple[str, ...],
) -> FusionRegion:
    node_set = set(node_ids)
    node_by_id = {node.node_id: node for node in graph.nodes}
    producers = _producer_map(graph)
    consumers = _consumer_map(graph)
    inputs: list[str] = []
    outputs: list[str] = []
    parameters: list[str] = []
    for node_id in node_ids:
        node = node_by_id[node_id]
        for value_id in node.inputs:
            if producers.get(value_id) not in node_set and value_id not in inputs:
                inputs.append(value_id)
        for value_id in node.outputs:
            outside = any(consumer not in node_set for consumer in consumers.get(value_id, ()))
            if outside or value_id in graph.outputs or not consumers.get(value_id):
                if value_id not in outputs:
                    outputs.append(value_id)
        for parameter in node.parameters:
            if parameter.logical_name not in parameters:
                parameters.append(parameter.logical_name)
    digest = hashlib.sha256("\0".join(node_ids).encode("utf-8")).hexdigest()[:16]
    return FusionRegion(
        region_id=f"fusion-{digest}",
        node_ids=node_ids,
        input_value_ids=tuple(inputs),
        output_value_ids=tuple(outputs),
        parameter_names=tuple(parameters),
    )


def find_fusion_regions(
    graph: OpGraph,
    *,
    max_region_size: int = 32,
    can_fuse: Callable[[OpNode, OpNode], bool] | None = None,
) -> FusionPlan:
    """Find deterministic conservative fusion candidates.

    A region is connected and graph-convex.  Every value that becomes internal to the
    region has exactly one global consumer, and no effectful/barrier node can enter a
    region.  This deliberately excludes profitable but more difficult multi-output or
    effect-aware fusion patterns.
    """

    if max_region_size < 2:
        raise ValueError("max_region_size must be at least two")
    nodes = _ordered_nodes(graph)
    node_by_id = {node.node_id: node for node in nodes}
    topo_index = {node.node_id: index for index, node in enumerate(nodes)}
    successors, predecessors = _node_edges(graph)
    reachable = _transitive_successors(nodes, successors)
    all_node_ids = set(node_by_id)
    consumers = _consumer_map(graph)
    graph_outputs = set(graph.outputs)
    barriers = {node.node_id for node in nodes if _node_is_barrier(node)}
    # A control dependency names an externally observable scheduling boundary.
    # Keep both endpoints out of a fusion region so a future contraction cannot
    # invalidate the control-input identity or move work across that boundary.
    barriers.update(control_id for node in nodes for control_id in node.control_inputs)
    assigned: set[str] = set()
    regions: list[FusionRegion] = []
    candidate_edges = 0
    accepted_edges = 0

    for seed in nodes:
        if seed.node_id in assigned or seed.node_id in barriers:
            continue
        region = {seed.node_id}
        while len(region) < max_region_size:
            candidates = sorted(
                {
                    successor
                    for node_id in region
                    for successor in successors[node_id]
                    if successor not in assigned
                    and successor not in region
                    and successor not in barriers
                },
                key=topo_index.__getitem__,
            )
            extended = False
            for candidate_id in candidates:
                candidate_edges += 1
                candidate = node_by_id[candidate_id]
                if can_fuse is not None and not any(
                    can_fuse(node_by_id[pred], candidate)
                    for pred in predecessors[candidate_id] & region
                ):
                    continue
                proposal = region | {candidate_id}
                if not _is_connected(proposal, successors, predecessors):
                    continue
                if not _is_convex(proposal, all_node_ids, reachable):
                    continue
                if not _internal_values_are_single_consumer(
                    proposal,
                    node_by_id,
                    consumers,
                    graph_outputs,
                ):
                    continue
                region = proposal
                accepted_edges += 1
                extended = True
                break
            if not extended:
                break
        if len(region) < 2:
            continue
        ordered_region = tuple(sorted(region, key=topo_index.__getitem__))
        regions.append(_make_fusion_region(graph, ordered_region))
        assigned.update(region)

    fused_nodes = {node_id for region in regions for node_id in region.node_ids}
    return FusionPlan(
        regions=tuple(regions),
        unfused_node_ids=tuple(node.node_id for node in nodes if node.node_id not in fused_nodes),
        barrier_node_ids=tuple(node.node_id for node in nodes if node.node_id in barriers),
        candidate_edges=candidate_edges,
        accepted_edges=accepted_edges,
    )


_DTYPE_BITS = {
    "bool": 8,
    "uint8": 8,
    "int8": 8,
    "int2": 2,
    "int3": 3,
    "int4": 4,
    "fp8": 8,
    "float8": 8,
    "fp16": 16,
    "float16": 16,
    "half": 16,
    "bf16": 16,
    "bfloat16": 16,
    "fp32": 32,
    "float32": 32,
    "int32": 32,
    "fp32-class": 32,
    "fp64": 64,
    "float64": 64,
    "int64": 64,
}


def _tensor_bytes(value: TensorSpec) -> int:
    dtype = _enum_value(value.dtype).lower()
    try:
        bits = _DTYPE_BITS[dtype]
    except KeyError as exc:
        raise ValueError(f"cannot size tensor dtype {dtype!r}") from exc
    elements = math.prod(value.shape)
    return (elements * bits + 7) // 8


def _aligned_size(size: int, alignment: int) -> int:
    return ((size + alignment - 1) // alignment) * alignment


def _canonical_alias(
    value_id: str,
    values: Mapping[str, TensorSpec],
) -> str:
    seen: set[str] = set()
    current = value_id
    while values[current].alias_of is not None:
        if current in seen:
            raise ValueError(f"alias cycle at {value_id}")
        seen.add(current)
        target = values[current].alias_of
        if target not in values:
            raise ValueError(f"unknown alias target {target!r}")
        current = target
    return current


@dataclass(frozen=True)
class ValueLifetime:
    value_id: str
    canonical_value_id: str
    start: int
    end: int
    logical_bytes: int
    allocated_bytes: int
    buffer_id: int
    memory_space: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "value_id": self.value_id,
            "canonical_value_id": self.canonical_value_id,
            "start": self.start,
            "end": self.end,
            "logical_bytes": self.logical_bytes,
            "allocated_bytes": self.allocated_bytes,
            "buffer_id": self.buffer_id,
            "memory_space": self.memory_space,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ValueLifetime:
        return cls(
            value_id=str(payload["value_id"]),
            canonical_value_id=str(payload["canonical_value_id"]),
            start=int(payload["start"]),
            end=int(payload["end"]),
            logical_bytes=int(payload["logical_bytes"]),
            allocated_bytes=int(payload["allocated_bytes"]),
            buffer_id=int(payload["buffer_id"]),
            memory_space=str(payload.get("memory_space", "logical")),
        )


@dataclass(frozen=True)
class BufferSlot:
    buffer_id: int
    capacity_bytes: int
    value_ids: tuple[str, ...]
    memory_space: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "buffer_id": self.buffer_id,
            "capacity_bytes": self.capacity_bytes,
            "value_ids": list(self.value_ids),
            "memory_space": self.memory_space,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BufferSlot:
        return cls(
            buffer_id=int(payload["buffer_id"]),
            capacity_bytes=int(payload["capacity_bytes"]),
            value_ids=tuple(str(value) for value in payload["value_ids"]),
            memory_space=str(payload.get("memory_space", "logical")),
        )


@dataclass(frozen=True)
class LivenessPlan:
    alignment_bytes: int
    lifetimes: tuple[ValueLifetime, ...]
    buffers: tuple[BufferSlot, ...]
    logical_bytes: int
    naive_reserved_bytes: int
    reserved_bytes: int
    peak_live_bytes: int

    @property
    def reuse_savings_bytes(self) -> int:
        return self.naive_reserved_bytes - self.reserved_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "alignment_bytes": self.alignment_bytes,
            "lifetimes": [lifetime.as_dict() for lifetime in self.lifetimes],
            "buffers": [buffer.as_dict() for buffer in self.buffers],
            "logical_bytes": self.logical_bytes,
            "naive_reserved_bytes": self.naive_reserved_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_live_bytes": self.peak_live_bytes,
            "reuse_savings_bytes": self.reuse_savings_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LivenessPlan:
        return cls(
            alignment_bytes=int(payload["alignment_bytes"]),
            lifetimes=tuple(
                ValueLifetime.from_dict(value) for value in payload.get("lifetimes", ())
            ),
            buffers=tuple(BufferSlot.from_dict(value) for value in payload.get("buffers", ())),
            logical_bytes=int(payload["logical_bytes"]),
            naive_reserved_bytes=int(payload["naive_reserved_bytes"]),
            reserved_bytes=int(payload["reserved_bytes"]),
            peak_live_bytes=int(payload["peak_live_bytes"]),
        )


def allocate_liveness(
    graph: OpGraph,
    *,
    alignment_bytes: int = 64,
) -> LivenessPlan:
    """Allocate reusable buffers from inclusive tensor lifetime intervals.

    Graph inputs are caller-owned and therefore excluded.  Produced aliases share the
    canonical value's interval and buffer.  Because intervals are inclusive, a buffer
    can be reused only when ``previous.end < next.start``.
    """

    if alignment_bytes <= 0:
        raise ValueError("alignment_bytes must be positive")
    nodes = _ordered_nodes(graph)
    node_index = {node.node_id: index for index, node in enumerate(nodes)}
    producers = _producer_map(graph)
    consumers = _consumer_map(graph)
    values = _value_map(graph)
    graph_inputs = set(graph.inputs)
    graph_outputs = set(graph.outputs)
    terminal_position = len(nodes)

    groups: dict[str, list[str]] = {}
    for value in graph.values:
        canonical = _canonical_alias(value.value_id, values)
        groups.setdefault(canonical, []).append(value.value_id)

    intervals: list[tuple[int, int, str, tuple[str, ...], int, int, str]] = []
    for canonical, members_list in groups.items():
        members = tuple(members_list)
        if canonical in graph_inputs:
            continue
        producer_positions = [
            node_index[producers[value_id]] for value_id in members if value_id in producers
        ]
        if not producer_positions:
            continue
        start = min(producer_positions)
        consumer_positions = [
            node_index[consumer_id]
            for value_id in members
            for consumer_id in consumers.get(value_id, ())
        ]
        end = max(consumer_positions, default=start)
        if any(
            value_id in graph_outputs
            or values[value_id].storage_class in {StorageClass.STATE, StorageClass.OUTPUT}
            for value_id in members
        ):
            end = max(end, terminal_position)
        logical_size = max(_tensor_bytes(values[value_id]) for value_id in members)
        allocation_size = _aligned_size(logical_size, alignment_bytes)
        memory_space = values[canonical].memory_space
        intervals.append(
            (
                start,
                end,
                canonical,
                members,
                logical_size,
                allocation_size,
                memory_space,
            )
        )
    intervals.sort(key=lambda item: (item[0], item[2]))

    capacities: list[int] = []
    memory_spaces: list[str] = []
    assignments: dict[str, int] = {}
    members_by_buffer: dict[int, list[str]] = {}
    active: dict[int, int] = {}
    available: set[int] = set()

    for (
        start,
        end,
        canonical,
        members,
        _logical_size,
        allocation_size,
        memory_space,
    ) in intervals:
        for buffer_id, active_end in tuple(active.items()):
            if active_end < start:
                del active[buffer_id]
                available.add(buffer_id)
        compatible = {
            buffer_id for buffer_id in available if memory_spaces[buffer_id] == memory_space
        }
        if compatible:
            buffer_id = min(
                compatible,
                key=lambda candidate: (
                    max(capacities[candidate], allocation_size) - capacities[candidate],
                    max(capacities[candidate], allocation_size),
                    candidate,
                ),
            )
            available.remove(buffer_id)
            capacities[buffer_id] = max(capacities[buffer_id], allocation_size)
        else:
            buffer_id = len(capacities)
            capacities.append(allocation_size)
            memory_spaces.append(memory_space)
        active[buffer_id] = end
        assignments[canonical] = buffer_id
        members_by_buffer.setdefault(buffer_id, []).extend(members)

    lifetimes: list[ValueLifetime] = []
    for (
        start,
        end,
        canonical,
        members,
        logical_size,
        allocation_size,
        memory_space,
    ) in intervals:
        buffer_id = assignments[canonical]
        for value_id in members:
            lifetimes.append(
                ValueLifetime(
                    value_id=value_id,
                    canonical_value_id=canonical,
                    start=start,
                    end=end,
                    logical_bytes=logical_size,
                    allocated_bytes=allocation_size,
                    buffer_id=buffer_id,
                    memory_space=memory_space,
                )
            )
    value_order = {value.value_id: index for index, value in enumerate(graph.values)}
    lifetimes.sort(key=lambda value: value_order[value.value_id])

    peak_live = 0
    for position in range(terminal_position + 1):
        peak_live = max(
            peak_live,
            sum(
                allocation_size
                for (
                    start,
                    end,
                    _canonical,
                    _members,
                    _logical,
                    allocation_size,
                    _memory_space,
                ) in intervals
                if start <= position <= end
            ),
        )
    buffers = tuple(
        BufferSlot(
            buffer_id=buffer_id,
            capacity_bytes=capacity,
            value_ids=tuple(members_by_buffer.get(buffer_id, ())),
            memory_space=memory_spaces[buffer_id],
        )
        for buffer_id, capacity in enumerate(capacities)
    )
    return LivenessPlan(
        alignment_bytes=alignment_bytes,
        lifetimes=tuple(lifetimes),
        buffers=buffers,
        logical_bytes=sum(item[4] for item in intervals),
        naive_reserved_bytes=sum(item[5] for item in intervals),
        reserved_bytes=sum(capacities),
        peak_live_bytes=peak_live,
    )


def _finite_nonnegative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _unary_costs(
    raw: Mapping[str, float] | Sequence[float],
    backend_a: str,
    backend_b: str,
    node_id: str,
) -> tuple[float, float]:
    if isinstance(raw, Mapping):
        try:
            cost_a = raw[backend_a]
            cost_b = raw[backend_b]
        except KeyError as exc:
            raise ValueError(f"missing backend cost for node {node_id}") from exc
    else:
        if len(raw) != 2:
            raise ValueError(f"node {node_id} costs must have exactly two entries")
        cost_a, cost_b = raw
    return (
        _finite_nonnegative(float(cost_a), f"{node_id}/{backend_a} cost"),
        _finite_nonnegative(float(cost_b), f"{node_id}/{backend_b} cost"),
    )


def _dependency_transfer_cost(
    transfer_costs: Mapping[object, float | Sequence[float]] | None,
    *,
    producer_id: str,
    consumer_id: str,
    value_id: str,
) -> tuple[float, float]:
    if transfer_costs is None:
        return 0.0, 0.0
    key = (producer_id, consumer_id, value_id)
    raw = transfer_costs.get(key, transfer_costs.get(value_id, 0.0))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != 2:
            raise ValueError(f"transfer cost for {key!r} must have two directions")
        cost_ab, cost_ba = raw
    else:
        cost_ab = cost_ba = raw
    return (
        _finite_nonnegative(float(cost_ab), f"{key!r} {producer_id}->{consumer_id} cost"),
        _finite_nonnegative(float(cost_ba), f"{key!r} {consumer_id}->{producer_id} cost"),
    )


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: float


class _Dinic:
    def __init__(self, node_count: int) -> None:
        self.adjacency: list[list[_FlowEdge]] = [[] for _ in range(node_count)]

    def add_edge(self, source: int, target: int, capacity: float) -> None:
        if capacity <= _FLOW_EPSILON:
            return
        forward = _FlowEdge(target, len(self.adjacency[target]), capacity)
        reverse = _FlowEdge(source, len(self.adjacency[source]), 0.0)
        self.adjacency[source].append(forward)
        self.adjacency[target].append(reverse)

    def _send(
        self,
        node: int,
        sink: int,
        flow: float,
        level: list[int],
        cursor: list[int],
    ) -> float:
        if node == sink:
            return flow
        while cursor[node] < len(self.adjacency[node]):
            edge = self.adjacency[node][cursor[node]]
            if edge.capacity > _FLOW_EPSILON and level[edge.to] == level[node] + 1:
                pushed = self._send(
                    edge.to,
                    sink,
                    min(flow, edge.capacity),
                    level,
                    cursor,
                )
                if pushed > _FLOW_EPSILON:
                    edge.capacity -= pushed
                    reverse = self.adjacency[edge.to][edge.reverse]
                    reverse.capacity += pushed
                    return pushed
            cursor[node] += 1
        return 0.0

    def maximum_flow(self, source: int, sink: int) -> float:
        total = 0.0
        node_count = len(self.adjacency)
        while True:
            level = [-1] * node_count
            level[source] = 0
            queue: deque[int] = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.adjacency[node]:
                    if edge.capacity > _FLOW_EPSILON and level[edge.to] < 0:
                        level[edge.to] = level[node] + 1
                        queue.append(edge.to)
            if level[sink] < 0:
                break
            cursor = [0] * node_count
            while True:
                pushed = self._send(source, sink, math.inf, level, cursor)
                if pushed <= _FLOW_EPSILON:
                    break
                total += pushed
        return total

    def source_partition(self, source: int) -> set[int]:
        reachable = {source}
        pending = [source]
        while pending:
            node = pending.pop()
            for edge in self.adjacency[node]:
                if edge.capacity > _FLOW_EPSILON and edge.to not in reachable:
                    reachable.add(edge.to)
                    pending.append(edge.to)
        return reachable


@dataclass(frozen=True)
class PlacementCut:
    value_id: str
    producer_node_id: str
    consumer_node_id: str
    source_backend: str
    target_backend: str
    cost: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost", float(self.cost))

    def as_dict(self) -> dict[str, Any]:
        return {
            "value_id": self.value_id,
            "producer_node_id": self.producer_node_id,
            "consumer_node_id": self.consumer_node_id,
            "source_backend": self.source_backend,
            "target_backend": self.target_backend,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlacementCut:
        return cls(
            value_id=str(payload["value_id"]),
            producer_node_id=str(payload["producer_node_id"]),
            consumer_node_id=str(payload["consumer_node_id"]),
            source_backend=str(payload["source_backend"]),
            target_backend=str(payload["target_backend"]),
            cost=float(payload["cost"]),
        )


@dataclass(frozen=True)
class BinaryPlacement:
    backend_a: str
    backend_b: str
    assignments: tuple[tuple[str, str], ...]
    unary_cost: float
    transfer_cost: float
    total_cost: float
    minimum_cut_cost: float
    cuts: tuple[PlacementCut, ...]
    exact: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "unary_cost",
            "transfer_cost",
            "total_cost",
            "minimum_cut_cost",
        ):
            object.__setattr__(self, field_name, float(getattr(self, field_name)))

    def assignment_map(self) -> dict[str, str]:
        return dict(self.assignments)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_a": self.backend_a,
            "backend_b": self.backend_b,
            "assignments": {key: value for key, value in self.assignments},
            "unary_cost": self.unary_cost,
            "transfer_cost": self.transfer_cost,
            "total_cost": self.total_cost,
            "minimum_cut_cost": self.minimum_cut_cost,
            "cuts": [cut.as_dict() for cut in self.cuts],
            "exact": self.exact,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BinaryPlacement:
        assignments = payload["assignments"]
        if not isinstance(assignments, Mapping):
            raise TypeError("placement assignments must be an object")
        return cls(
            backend_a=str(payload["backend_a"]),
            backend_b=str(payload["backend_b"]),
            assignments=tuple(
                (str(node_id), str(backend)) for node_id, backend in assignments.items()
            ),
            unary_cost=float(payload["unary_cost"]),
            transfer_cost=float(payload["transfer_cost"]),
            total_cost=float(payload["total_cost"]),
            minimum_cut_cost=float(payload["minimum_cut_cost"]),
            cuts=tuple(PlacementCut.from_dict(cut) for cut in payload.get("cuts", ())),
            exact=bool(payload.get("exact", True)),
        )


def place_binary_backends(
    graph: OpGraph,
    *,
    backend_a: str,
    backend_b: str,
    node_costs: Mapping[str, Mapping[str, float] | Sequence[float]],
    transfer_costs: Mapping[object, float | Sequence[float]] | None = None,
    fixed_placements: Mapping[str, str] | None = None,
) -> BinaryPlacement:
    """Solve exact two-backend placement for unary plus dependency-cut costs.

    ``transfer_costs[value_id]`` applies to every producer-consumer dependency
    carrying that value.  A ``(producer, consumer, value)`` key overrides it.  A scalar
    is symmetric; a two-item sequence is ``(A->B, B->A)``.  This pairwise objective is
    submodular and is solved exactly by deterministic s-t minimum cut.
    """

    if not backend_a or not backend_b or backend_a == backend_b:
        raise ValueError("binary placement requires two distinct backend names")
    nodes = _ordered_nodes(graph)
    node_ids = {node.node_id for node in nodes}
    missing = [node.node_id for node in nodes if node.node_id not in node_costs]
    if missing:
        raise ValueError(f"missing costs for nodes: {', '.join(missing)}")
    fixed = dict(fixed_placements or {})
    unknown_fixed = sorted(set(fixed) - node_ids)
    if unknown_fixed:
        raise ValueError(f"unknown fixed-placement nodes: {', '.join(unknown_fixed)}")
    invalid_fixed = sorted(
        node_id for node_id, backend in fixed.items() if backend not in {backend_a, backend_b}
    )
    if invalid_fixed:
        raise ValueError("fixed placements use an unknown backend for: " + ", ".join(invalid_fixed))

    unary: dict[str, tuple[float, float]] = {
        node.node_id: _unary_costs(
            node_costs[node.node_id],
            backend_a,
            backend_b,
            node.node_id,
        )
        for node in nodes
    }
    producers = _producer_map(graph)
    consumers = _consumer_map(graph)
    dependencies: list[tuple[str, str, str, float, float]] = []
    for value in graph.values:
        producer_id = producers.get(value.value_id)
        if producer_id is None:
            continue
        for consumer_id in consumers.get(value.value_id, ()):
            cost_ab, cost_ba = _dependency_transfer_cost(
                transfer_costs,
                producer_id=producer_id,
                consumer_id=consumer_id,
                value_id=value.value_id,
            )
            dependencies.append((producer_id, consumer_id, value.value_id, cost_ab, cost_ba))
    finite_objective = sum(cost for pair in unary.values() for cost in pair) + sum(
        cost_ab + cost_ba for _, _, _, cost_ab, cost_ba in dependencies
    )
    hard_constraint = finite_objective + 1.0

    source = len(nodes)
    sink = source + 1
    flow = _Dinic(len(nodes) + 2)
    index = {node.node_id: position for position, node in enumerate(nodes)}
    # Source side means backend A.  Cutting S->node pays B's unary cost;
    # cutting node->T pays A's unary cost.
    for node in nodes:
        cost_a, cost_b = unary[node.node_id]
        if fixed.get(node.node_id) == backend_a:
            cost_b += hard_constraint
        elif fixed.get(node.node_id) == backend_b:
            cost_a += hard_constraint
        flow.add_edge(source, index[node.node_id], cost_b)
        flow.add_edge(index[node.node_id], sink, cost_a)
    for producer_id, consumer_id, _value_id, cost_ab, cost_ba in dependencies:
        flow.add_edge(index[producer_id], index[consumer_id], cost_ab)
        flow.add_edge(index[consumer_id], index[producer_id], cost_ba)

    minimum_cut = flow.maximum_flow(source, sink)
    source_side = flow.source_partition(source)
    assignment_map = {
        node.node_id: (backend_a if index[node.node_id] in source_side else backend_b)
        for node in nodes
    }
    for node_id, backend in fixed.items():
        if assignment_map[node_id] != backend:
            raise RuntimeError("minimum-cut hard placement constraint was not respected")

    unary_total = sum(
        unary[node.node_id][0]
        if assignment_map[node.node_id] == backend_a
        else unary[node.node_id][1]
        for node in nodes
    )
    cuts: list[PlacementCut] = []
    for producer_id, consumer_id, value_id, cost_ab, cost_ba in dependencies:
        producer_backend = assignment_map[producer_id]
        consumer_backend = assignment_map[consumer_id]
        if producer_backend == consumer_backend:
            continue
        cost = cost_ab if producer_backend == backend_a else cost_ba
        cuts.append(
            PlacementCut(
                value_id=value_id,
                producer_node_id=producer_id,
                consumer_node_id=consumer_id,
                source_backend=producer_backend,
                target_backend=consumer_backend,
                cost=cost,
            )
        )
    transfer_total = float(sum(cut.cost for cut in cuts))
    total = unary_total + transfer_total
    if not math.isclose(total, minimum_cut, rel_tol=1e-9, abs_tol=1e-8):
        raise RuntimeError(
            f"minimum-cut accounting mismatch: recomputed={total}, flow={minimum_cut}"
        )
    return BinaryPlacement(
        backend_a=backend_a,
        backend_b=backend_b,
        assignments=tuple((node.node_id, assignment_map[node.node_id]) for node in nodes),
        unary_cost=unary_total,
        transfer_cost=transfer_total,
        total_cost=total,
        minimum_cut_cost=minimum_cut,
        cuts=tuple(cuts),
    )


@dataclass(frozen=True)
class GraphCompilation:
    source_graph_fingerprint: str
    graph: OpGraph
    slice_report: SliceReport
    fusion_plan: FusionPlan
    liveness_plan: LivenessPlan
    rewrite_certificate: DemandRewriteCertificate | None = None
    placement: BinaryPlacement | None = None
    metadata: tuple[tuple[str, Any], ...] = ()
    schema_version: str = GRAPH_COMPILATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GRAPH_COMPILATION_SCHEMA:
            raise ValueError(f"unsupported graph compilation schema: {self.schema_version}")
        if self.rewrite_certificate is not None:
            if self.rewrite_certificate.source_fingerprint != self.source_graph_fingerprint:
                raise ValueError("rewrite certificate source graph fingerprint mismatch")
            if self.rewrite_certificate.rewritten_fingerprint != self.graph.fingerprint:
                raise ValueError("rewrite certificate rewritten graph fingerprint mismatch")
            graph_output_contract = dict(self.graph.metadata).get("output_contract")
            if (
                graph_output_contract is not None
                and self.rewrite_certificate.output_contract != graph_output_contract
            ):
                raise ValueError("rewrite certificate output contract mismatch")
        graph_node_ids = tuple(node.node_id for node in self.graph.nodes)
        graph_node_set = set(graph_node_ids)
        graph_value_ids = tuple(value.value_id for value in self.graph.values)
        graph_value_set = set(graph_value_ids)
        if self.slice_report.roots != self.graph.outputs:
            raise ValueError("slice report roots do not match compiled graph outputs")
        if self.slice_report.kept_node_ids != graph_node_ids:
            raise ValueError("slice report kept nodes do not match compiled graph")
        if self.slice_report.kept_value_ids != graph_value_ids:
            raise ValueError("slice report kept values do not match compiled graph")
        if graph_node_set.intersection(self.slice_report.removed_node_ids):
            raise ValueError("slice report removed nodes overlap compiled graph")
        if graph_value_set.intersection(self.slice_report.removed_value_ids):
            raise ValueError("slice report removed values overlap compiled graph")

        fusion_node_ids = tuple(
            node_id for region in self.fusion_plan.regions for node_id in region.node_ids
        )
        if len(fusion_node_ids) != len(set(fusion_node_ids)):
            raise ValueError("fusion regions contain duplicate graph nodes")
        if set(fusion_node_ids).intersection(self.fusion_plan.unfused_node_ids):
            raise ValueError("fused and unfused graph nodes overlap")
        covered_fusion_nodes = set(fusion_node_ids).union(self.fusion_plan.unfused_node_ids)
        if covered_fusion_nodes != graph_node_set:
            raise ValueError("fusion plan does not cover exactly the compiled graph nodes")
        if not set(self.fusion_plan.barrier_node_ids).issubset(graph_node_set):
            raise ValueError("fusion plan contains an unknown barrier node")
        if set(self.fusion_plan.barrier_node_ids).intersection(fusion_node_ids):
            raise ValueError("fusion plan places a barrier inside a fused region")
        if (
            self.fusion_plan.candidate_edges < 0
            or self.fusion_plan.accepted_edges < 0
            or self.fusion_plan.accepted_edges > self.fusion_plan.candidate_edges
        ):
            raise ValueError("fusion plan edge counts are inconsistent")
        region_ids = [region.region_id for region in self.fusion_plan.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("fusion region IDs must be unique")
        for region in self.fusion_plan.regions:
            if not region.node_ids:
                raise ValueError("fusion regions cannot be empty")
            if not set(region.input_value_ids).issubset(graph_value_set):
                raise ValueError("fusion region contains an unknown input value")
            if not set(region.output_value_ids).issubset(graph_value_set):
                raise ValueError("fusion region contains an unknown output value")

        expected_liveness = allocate_liveness(
            self.graph,
            alignment_bytes=self.liveness_plan.alignment_bytes,
        )
        if self.liveness_plan != expected_liveness:
            raise ValueError("liveness plan does not match compiled graph")

        if self.placement is not None:
            assignment_ids = [node_id for node_id, _ in self.placement.assignments]
            if len(assignment_ids) != len(set(assignment_ids)):
                raise ValueError("placement contains duplicate node assignments")
            if set(assignment_ids) != graph_node_set:
                raise ValueError("placement does not cover exactly the compiled graph nodes")
            allowed_backends = {self.placement.backend_a, self.placement.backend_b}
            if len(allowed_backends) != 2:
                raise ValueError("placement backends must be distinct")
            assignment_map = self.placement.assignment_map()
            if not set(assignment_map.values()).issubset(allowed_backends):
                raise ValueError("placement contains an unknown backend")
            if not math.isclose(
                self.placement.transfer_cost,
                sum(cut.cost for cut in self.placement.cuts),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("placement transfer cost does not match its cuts")
            if not math.isclose(
                self.placement.total_cost,
                self.placement.unary_cost + self.placement.transfer_cost,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("placement total cost is inconsistent")
            if not math.isclose(
                self.placement.minimum_cut_cost,
                self.placement.total_cost,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("placement minimum-cut cost is inconsistent")
            node_map = self.graph.node_map
            for cut in self.placement.cuts:
                producer = node_map.get(cut.producer_node_id)
                consumer = node_map.get(cut.consumer_node_id)
                if (
                    producer is None
                    or consumer is None
                    or cut.value_id not in producer.outputs
                    or cut.value_id not in consumer.inputs
                ):
                    raise ValueError("placement cut is not a compiled graph dependency")
                if (
                    assignment_map[producer.node_id] != cut.source_backend
                    or assignment_map[consumer.node_id] != cut.target_backend
                    or cut.source_backend == cut.target_backend
                ):
                    raise ValueError("placement cut does not match node assignments")
        keys = [key for key, _ in self.metadata]
        if len(keys) != len(set(keys)):
            raise ValueError("graph compilation metadata keys must be unique")

    @cached_property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "graph": self.graph.as_dict(),
            "rewrite_certificate": (
                None if self.rewrite_certificate is None else self.rewrite_certificate.as_dict()
            ),
            "slice_report": self.slice_report.as_dict(),
            "fusion_plan": self.fusion_plan.as_dict(),
            "liveness_plan": self.liveness_plan.as_dict(),
            "placement": None if self.placement is None else self.placement.as_dict(),
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
    def from_dict(cls, payload: Mapping[str, Any]) -> GraphCompilation:
        placement = payload.get("placement")
        rewrite_certificate = payload.get("rewrite_certificate")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("graph compilation metadata must be an object")
        return cls(
            source_graph_fingerprint=str(payload["source_graph_fingerprint"]),
            graph=OpGraph.from_dict(payload["graph"]),
            slice_report=SliceReport.from_dict(payload["slice_report"]),
            fusion_plan=FusionPlan.from_dict(payload["fusion_plan"]),
            liveness_plan=LivenessPlan.from_dict(payload["liveness_plan"]),
            rewrite_certificate=(
                None
                if rewrite_certificate is None
                else DemandRewriteCertificate.from_dict(rewrite_certificate)
            ),
            placement=(None if placement is None else BinaryPlacement.from_dict(placement)),
            metadata=tuple(sorted((str(key), value) for key, value in metadata.items())),
            schema_version=str(payload.get("schema_version", GRAPH_COMPILATION_SCHEMA)),
        )

    @classmethod
    def from_json(cls, payload: str) -> GraphCompilation:
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise TypeError("graph compilation JSON must encode an object")
        return cls.from_dict(decoded)


def compile_graph(
    graph: OpGraph,
    roots: Iterable[str] | None = None,
    *,
    retain_effects: bool = True,
    retain_barriers: bool = True,
    max_fusion_region_size: int = 32,
    can_fuse: Callable[[OpNode, OpNode], bool] | None = None,
    alignment_bytes: int = 64,
    backend_a: str | None = None,
    backend_b: str | None = None,
    node_costs: Mapping[str, Mapping[str, float] | Sequence[float]] | None = None,
    transfer_costs: Mapping[object, float | Sequence[float]] | None = None,
    fixed_placements: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    source_graph_fingerprint: str | None = None,
    rewrite_certificate: DemandRewriteCertificate | None = None,
) -> GraphCompilation:
    """Run the hardware-independent graph analysis pipeline."""

    if (
        rewrite_certificate is not None
        and rewrite_certificate.rewritten_fingerprint != graph.fingerprint
    ):
        raise ValueError("rewrite certificate does not describe the graph being compiled")
    sliced = backward_slice(
        graph,
        roots,
        retain_effects=retain_effects,
        retain_barriers=retain_barriers,
    )
    fusion = find_fusion_regions(
        sliced.graph,
        max_region_size=max_fusion_region_size,
        can_fuse=can_fuse,
    )
    liveness = allocate_liveness(sliced.graph, alignment_bytes=alignment_bytes)
    placement: BinaryPlacement | None = None
    placement_requested = any(value is not None for value in (backend_a, backend_b, node_costs))
    if placement_requested:
        if backend_a is None or backend_b is None or node_costs is None:
            raise ValueError("backend_a, backend_b, and node_costs are all required for placement")
        placement = place_binary_backends(
            sliced.graph,
            backend_a=backend_a,
            backend_b=backend_b,
            node_costs=node_costs,
            transfer_costs=transfer_costs,
            fixed_placements=fixed_placements,
        )
    return GraphCompilation(
        source_graph_fingerprint=source_graph_fingerprint or graph.fingerprint,
        graph=sliced.graph,
        slice_report=sliced.report,
        fusion_plan=fusion,
        liveness_plan=liveness,
        rewrite_certificate=rewrite_certificate,
        placement=placement,
        metadata=tuple(sorted((str(key), value) for key, value in (metadata or {}).items())),
    )
