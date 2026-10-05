"""Execution traces that bind semantic graph resources to QStore calls."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .executable import WorkPlanExecutionResult, execute_lowered_plan
from .graph_passes import GraphCompilation
from .ir import DenseWorkPlan
from .lowering import LoweredWorkPlan


@dataclass(frozen=True)
class GraphResourceTrace:
    graph_compilation_fingerprint: str
    expected_logical_resources: tuple[str, ...]
    observed_logical_resources: tuple[str, ...]
    missing_resources: tuple[str, ...]
    unexpected_resources: tuple[str, ...]
    exact_order: bool
    topological_order_valid: bool
    execution_evidence: dict[str, Any]
    trace_basis: str = "measured-qstore-method-interposition"

    @property
    def expected_count(self) -> int:
        return len(self.expected_logical_resources)

    @property
    def observed_count(self) -> int:
        return len(self.observed_logical_resources)

    @property
    def complete(self) -> bool:
        return (self.exact_order or self.topological_order_valid) and (
            not self.missing_resources and not self.unexpected_resources
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_compilation_fingerprint": self.graph_compilation_fingerprint,
            "expected_logical_resources": list(self.expected_logical_resources),
            "observed_logical_resources": list(self.observed_logical_resources),
            "expected_count": self.expected_count,
            "observed_count": self.observed_count,
            "missing_resources": list(self.missing_resources),
            "unexpected_resources": list(self.unexpected_resources),
            "exact_order": self.exact_order,
            "topological_order_valid": self.topological_order_valid,
            "complete": self.complete,
            "execution_evidence": self.execution_evidence,
            "trace_basis": self.trace_basis,
        }


def _counter_difference(left: Counter[str], right: Counter[str]) -> tuple[str, ...]:
    return tuple(
        resource
        for resource in sorted(left)
        for _ in range(max(0, left[resource] - right[resource]))
    )


def _parameter_order_is_topological(
    graph_compilation: GraphCompilation,
    observed: tuple[str, ...],
) -> bool:
    graph = graph_compilation.graph
    parameter_nodes: dict[str, str] = {}
    for node in graph.nodes:
        for parameter in node.parameters:
            if parameter.logical_name in parameter_nodes:
                return observed == tuple(ref.logical_name for ref in graph.parameter_refs)
            parameter_nodes[parameter.logical_name] = node.node_id
    if set(observed) != set(parameter_nodes) or len(observed) != len(parameter_nodes):
        return False

    positions = {resource: index for index, resource in enumerate(observed)}
    producer_nodes = graph.producer_map
    upstream_parameters: dict[str, set[str]] = {}
    for node in graph.nodes:
        predecessors = {
            producer_nodes[value_id] for value_id in node.inputs if value_id in producer_nodes
        }
        predecessors.update(node.control_inputs)
        inherited: set[str] = set()
        for predecessor in predecessors:
            inherited.update(upstream_parameters[predecessor])
            inherited.update(
                parameter.logical_name for parameter in graph.node_map[predecessor].parameters
            )
        node_parameters = tuple(parameter.logical_name for parameter in node.parameters)
        for resource in node_parameters:
            if any(positions[predecessor] >= positions[resource] for predecessor in inherited):
                return False
        if any(
            positions[left] >= positions[right]
            for left, right in zip(node_parameters, node_parameters[1:], strict=False)
        ):
            return False
        upstream_parameters[node.node_id] = inherited
    return True


def trace_qstore_graph_execution(
    engine: Any,
    plan: DenseWorkPlan,
    lowered: LoweredWorkPlan,
    graph_compilation: GraphCompilation,
    ids_list: Sequence[np.ndarray | Sequence[int]],
) -> GraphResourceTrace:
    """Execute once and compare graph parameter order with observed QStore access.

    The helper temporarily interposes only the four logical resource entry points used
    by paged QStore execution. It restores the store object in ``finally`` and must not
    be used concurrently with another request on the same engine instance.
    """

    store = getattr(engine, "store", None)
    if store is None:
        raise TypeError("graph resource tracing requires an engine with a QStore")
    method_names = ("embed_rows", "matmul", "fp32", "row_blocks")
    missing_methods = [name for name in method_names if not callable(getattr(store, name, None))]
    if missing_methods:
        raise TypeError("QStore does not expose traceable methods: " + ", ".join(missing_methods))

    expected = tuple(
        parameter.logical_name
        for node in graph_compilation.graph.nodes
        for parameter in node.parameters
    )
    observed: list[str] = []
    instance_state: dict[str, tuple[bool, Any]] = {}
    for method_name in method_names:
        had_instance_attribute = method_name in vars(store)
        instance_state[method_name] = (
            had_instance_attribute,
            vars(store).get(method_name),
        )
        original = getattr(store, method_name)

        def traced(
            *args: Any,
            _method_name: str = method_name,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            logical_name = args[0] if args else kwargs.get("name")
            if logical_name is None:
                raise RuntimeError(f"{_method_name} trace could not resolve a resource name")
            observed.append(str(logical_name))
            return _original(*args, **kwargs)

        setattr(store, method_name, traced)

    execution: WorkPlanExecutionResult
    try:
        execution = execute_lowered_plan(
            engine,
            plan,
            lowered,
            ids_list,
            graph_compilation=graph_compilation,
        )
    finally:
        for method_name, (had_instance_attribute, previous) in instance_state.items():
            if had_instance_attribute:
                setattr(store, method_name, previous)
            else:
                delattr(store, method_name)

    expected_counter = Counter(expected)
    observed_tuple = tuple(observed)
    observed_counter = Counter(observed_tuple)
    topological_order_valid = _parameter_order_is_topological(
        graph_compilation,
        observed_tuple,
    )
    return GraphResourceTrace(
        graph_compilation_fingerprint=graph_compilation.fingerprint,
        expected_logical_resources=expected,
        observed_logical_resources=observed_tuple,
        missing_resources=_counter_difference(expected_counter, observed_counter),
        unexpected_resources=_counter_difference(observed_counter, expected_counter),
        exact_order=observed_tuple == expected,
        topological_order_valid=topological_order_valid,
        execution_evidence=execution.evidence,
    )
