"""Deterministic resource-aware scheduling over an already-valid causal DAG."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class ResidencyNode:
    node_id: str
    dependencies: tuple[str, ...]
    resource_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("residency node ID must be non-empty")
        for field in ("dependencies", "resource_ids"):
            values = tuple(str(value) for value in getattr(self, field))
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"{field} must contain unique non-empty identities")
            object.__setattr__(self, field, values)
        if self.node_id in self.dependencies:
            raise ValueError("residency node cannot depend on itself")


@dataclass(frozen=True)
class ResidencySchedule:
    node_order: tuple[str, ...]
    resource_transitions: int
    logical_resource_references: int
    unique_resource_count: int

    @property
    def residency_reuses(self) -> int:
        return max(0, self.logical_resource_references - self.resource_transitions)


def schedule_for_residency(nodes: tuple[ResidencyNode, ...]) -> ResidencySchedule:
    """Topologically schedule ready nodes, preferring the currently hot resource set."""

    by_id = OrderedDict((node.node_id, node) for node in nodes)
    if len(by_id) != len(nodes):
        raise ValueError("residency node IDs must be unique")
    known = set(by_id)
    for node in nodes:
        if not set(node.dependencies) <= known:
            raise ValueError("residency dependency references an unknown node")
    completed: set[str] = set()
    current_resources: set[str] = set()
    order: list[str] = []
    transitions = 0
    while len(order) < len(nodes):
        ready = [
            node
            for node in nodes
            if node.node_id not in completed and set(node.dependencies) <= completed
        ]
        if not ready:
            raise ValueError("residency graph contains a dependency cycle")
        selected = max(
            ready,
            key=lambda node: (
                len(current_resources.intersection(node.resource_ids)),
                -list(by_id).index(node.node_id),
            ),
        )
        next_resources = set(selected.resource_ids)
        transitions += len(next_resources - current_resources)
        current_resources = next_resources
        completed.add(selected.node_id)
        order.append(selected.node_id)
    all_resources = {resource for node in nodes for resource in node.resource_ids}
    return ResidencySchedule(
        node_order=tuple(order),
        resource_transitions=transitions,
        logical_resource_references=sum(len(node.resource_ids) for node in nodes),
        unique_resource_count=len(all_resources),
    )


__all__ = ["ResidencyNode", "ResidencySchedule", "schedule_for_residency"]
