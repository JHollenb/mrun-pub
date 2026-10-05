"""Adapter from the neutral debugger branch plan to the paged ScienceGraph.

``mrun`` owns physical execution, while ``manalysis`` owns the portable plan
and trace schemas.  Keeping this adapter mapping-based avoids a dependency
cycle between the two packages.  The adapter accepts a serialized
``BranchPlan`` with a backend artifact under either ``backend.sciencegraph``
or ``metadata.sciencegraph`` and returns a neutral result envelope.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from numbers import Integral
import re
from typing import Any

from .sciencegraph import InterventionScienceGraph, execute_intervention_sciencegraph

SCIENCEGRAPH_BRANCH_ADAPTER_SCHEMA = "mrun.sciencegraph-branch-adapter-result.v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _plan_fingerprint(payload: Mapping[str, Any]) -> str:
    declared = payload.get("fingerprint")
    if isinstance(declared, str) and declared:
        return declared
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _backend_artifact(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    backend = plan.get("backend")
    if isinstance(backend, Mapping):
        artifact = backend.get("sciencegraph")
        if isinstance(artifact, Mapping):
            return artifact
    metadata = plan.get("metadata")
    if isinstance(metadata, Mapping):
        artifact = metadata.get("sciencegraph")
        if isinstance(artifact, Mapping):
            return artifact
    raise ValueError(
        "serialized BranchPlan has no ScienceGraph artifact; expected "
        "backend.sciencegraph or metadata.sciencegraph"
    )


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _declared_layer(shared_cut: Mapping[str, Any]) -> int | None:
    """Read the layer identity from the neutral cut when it is serialized.

    Older neutral plans only carried ``cut_id=layer-N``.  Newer plans may put
    the same identity in ``trajectory_position`` or ``abi``.  Supporting both
    keeps old plans loadable while making a supplied identity auditable.
    """

    layers: list[int] = []
    cut_id = shared_cut.get("cut_id")
    if isinstance(cut_id, str):
        match = re.fullmatch(r"layer-(\d+)", cut_id)
        if match:
            layers.append(int(match.group(1)))
    for field in ("trajectory_position", "abi"):
        nested = shared_cut.get(field)
        if not isinstance(nested, Mapping):
            continue
        for key in ("layer", "layer_index", "cut_layer"):
            if key in nested:
                value = nested[key]
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f"shared_cut.{field}.{key} must be an integer")
                layers.append(int(value))
    if layers and any(layer != layers[0] for layer in layers[1:]):
        raise ValueError("neutral shared cut layer declarations disagree")
    return layers[0] if layers else None


def _validate_neutral_contract(
    plan: Mapping[str, Any],
    graph: InterventionScienceGraph,
    *,
    model_identity: str | None,
    numerical_contract: str | None,
    explicit_max_batch: int | None,
) -> tuple[str, str, int, dict[str, Any]]:
    """Validate the neutral identity and row ABI before physical execution."""

    pack = graph.branch_pack
    declared_model = plan.get("model_identity")
    declared_contract = plan.get("numerical_contract")
    if not isinstance(declared_model, str) or not declared_model:
        raise ValueError("neutral BranchPlan must declare model_identity")
    if not isinstance(declared_contract, str) or not declared_contract:
        raise ValueError("neutral BranchPlan must declare numerical_contract")
    if declared_model != pack.model_identity:
        raise ValueError("neutral plan model identity does not match the ScienceGraph")
    if declared_contract != pack.numerical_contract:
        raise ValueError("neutral plan numerical contract does not match the ScienceGraph")
    if model_identity is not None and model_identity != declared_model:
        raise ValueError("adapter model identity conflicts with the neutral BranchPlan")
    if numerical_contract is not None and numerical_contract != declared_contract:
        raise ValueError("adapter numerical contract conflicts with the neutral BranchPlan")

    shared_cut = plan.get("shared_cut")
    if not isinstance(shared_cut, Mapping):
        raise ValueError("neutral BranchPlan must declare shared_cut")
    for field in ("cut_id", "component_id", "port"):
        value = shared_cut.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"shared_cut.{field} must be a non-empty string")
    declared_layer = _declared_layer(shared_cut)
    if declared_layer is not None and declared_layer != pack.fork.cut.layer:
        raise ValueError(
            "neutral shared cut layer does not match the ScienceGraph StateCut"
        )
    trajectory_position = shared_cut.get("trajectory_position")
    if trajectory_position is not None and not isinstance(trajectory_position, Mapping):
        raise ValueError("shared_cut.trajectory_position must be a mapping")
    abi = shared_cut.get("abi")
    if abi is not None and not isinstance(abi, Mapping):
        raise ValueError("shared_cut.abi must be a mapping")
    if isinstance(abi, Mapping):
        for key, expected in (
            ("model_identity", pack.model_identity),
            ("numerical_contract", pack.numerical_contract),
        ):
            if key in abi and abi[key] != expected:
                raise ValueError(f"shared_cut.abi.{key} does not match the ScienceGraph")
    graph_ids = tuple(item.branch_id for item in pack.fork.branches)
    declared_cut_ids = shared_cut.get("branch_ids", ())
    if declared_cut_ids:
        if not isinstance(declared_cut_ids, (list, tuple)):
            raise ValueError("shared_cut.branch_ids must be a sequence")
        if tuple(declared_cut_ids) != graph_ids:
            raise ValueError("shared_cut.branch_ids do not match ScienceGraph order")

    declared_branches = plan.get("branches")
    if not isinstance(declared_branches, (list, tuple)):
        raise ValueError("neutral BranchPlan must declare a branch sequence")
    if len(declared_branches) != len(graph_ids):
        raise ValueError("neutral branch count does not match the ScienceGraph")
    row_slots: list[int] = []
    compatibility_keys: list[str] = []
    route_keys: list[str] = []
    fork_ids: list[str] = []
    for index, (item, graph_id) in enumerate(zip(declared_branches, graph_ids, strict=True)):
        if not isinstance(item, Mapping):
            raise ValueError(f"neutral branch {index} must be a mapping")
        if item.get("branch_id") != graph_id:
            raise ValueError("neutral branch order does not match ScienceGraph order")
        row_slot = item.get("row_slot")
        if isinstance(row_slot, bool) or not isinstance(row_slot, Integral):
            raise ValueError(f"neutral branch {graph_id!r} row_slot must be an integer")
        row_slots.append(int(row_slot))
        compatibility = item.get("compatibility_key", "")
        route = item.get("route_key")
        fork_id = item.get("fork_id")
        if not isinstance(compatibility, str) or not compatibility:
            raise ValueError(f"neutral branch {graph_id!r} compatibility_key is missing")
        if not isinstance(route, str) or not route:
            raise ValueError(f"neutral branch {graph_id!r} route_key is missing")
        if not isinstance(fork_id, str) or not fork_id:
            raise ValueError(f"neutral branch {graph_id!r} fork_id is missing")
        compatibility_keys.append(compatibility)
        route_keys.append(route)
        fork_ids.append(fork_id)
        parent = item.get("parent_branch_id")
        if parent is not None and parent not in graph_ids:
            raise ValueError(f"neutral branch {graph_id!r} names an unknown parent branch")
    expected_slots = list(range(len(graph_ids)))
    if row_slots != expected_slots:
        raise ValueError(
            "neutral row_slot sequence does not match physical ScienceGraph rows"
        )
    if len(set(compatibility_keys)) != 1 or len(set(route_keys)) != 1:
        raise ValueError(
            "ScienceGraph adapter requires one explicit compatibility and route bucket"
        )
    if len(set(fork_ids)) != 1:
        raise ValueError("ScienceGraph adapter requires one fork_id for the shared cut")

    has_declared_max = "max_batch_size" in plan
    declared_max = plan.get("max_batch_size", 1)
    declared_max = _positive_int(declared_max, field="max_batch_size")
    if explicit_max_batch is None:
        resolved_max = declared_max
        max_source = "neutral_plan" if has_declared_max else "neutral_plan_default"
    else:
        resolved_max = _positive_int(explicit_max_batch, field="max_branch_batch")
        max_source = "adapter_override"
    identity = {
        "model_identity": pack.model_identity,
        "numerical_contract": pack.numerical_contract,
        "graph_state_cut": pack.fork.cut.to_dict(),
        "neutral_shared_cut": dict(shared_cut),
        "branch_order": list(graph_ids),
        "row_slots": row_slots,
        "compatibility_key": compatibility_keys[0],
        "route_key": route_keys[0],
        "fork_id": fork_ids[0],
        "max_batch_size": resolved_max,
        "max_batch_size_source": max_source,
    }
    return declared_model, declared_contract, resolved_max, identity


def execute_serialized_sciencegraph_plan(
    engine: Any,
    plan: Mapping[str, Any],
    *,
    model_identity: str | None = None,
    numerical_contract: str | None = None,
    payload_bindings: Mapping[str, Any] | None = None,
    max_branch_batch: int | None = None,
) -> dict[str, Any]:
    """Execute one neutral plan through the existing StateCut fast path.

    The adapter is intentionally fail-closed.  An unsupported typed port or
    missing physical primitive is reported by the underlying ScienceGraph
    executor; this function never replaces it with a hidden scalar loop.
    """

    if not isinstance(plan, Mapping):
        raise TypeError("plan must be a serialized mapping")
    schema = plan.get("schema")
    if schema != "manalysis.generative-branch-plan.v1":
        raise ValueError(f"unsupported neutral branch-plan schema: {schema!r}")
    artifact = _backend_artifact(plan)
    graph = InterventionScienceGraph.from_dict(artifact)
    resolved_model_identity, resolved_contract, resolved_max_batch, execution_identity = (
        _validate_neutral_contract(
            plan,
            graph,
            model_identity=model_identity,
            numerical_contract=numerical_contract,
            explicit_max_batch=max_branch_batch,
        )
    )
    graph_ids = tuple(item.branch_id for item in graph.branch_pack.fork.branches)

    result = execute_intervention_sciencegraph(
        engine,
        graph,
        model_identity=resolved_model_identity,
        numerical_contract=resolved_contract,
        payload_bindings=payload_bindings,
        max_branch_batch=resolved_max_batch,
    )
    by_branch = {row["branch_id"]: row for row in result["results"]}
    telemetry = dict(result.get("telemetry", {}))
    telemetry.update(
        {
            "adapter": "sciencegraph",
            "neutral_plan_schema": schema,
            "logical_branch_order": list(graph_ids),
            "physical_backend": "mrun.paged.sciencegraph",
            "max_branch_batch": resolved_max_batch,
            "max_branch_batch_source": execution_identity["max_batch_size_source"],
        }
    )
    return {
        "schema": SCIENCEGRAPH_BRANCH_ADAPTER_SCHEMA,
        "plan_fingerprint": _plan_fingerprint(plan),
        "graph_fingerprint": result["graph_fingerprint"],
        "branch_outputs": by_branch,
        "branch_summaries": {
            branch_id: {
                "winner_token_id": row["winner_token_id"],
                "margin": row["margin"],
                "candidate_token_ids": row["candidate_token_ids"],
            }
            for branch_id, row in by_branch.items()
        },
        "telemetry": telemetry,
        "parity": {
            "winner_and_margin_source": "mrun.intervention-sciencegraph",
            "row_order_verified": True,
            "execution_identity_verified": True,
        },
        "execution_identity": execution_identity,
        "unsupported": [],
        "fingerprint": hashlib.sha256(
            _canonical(
                {
                    "schema": SCIENCEGRAPH_BRANCH_ADAPTER_SCHEMA,
                    "plan_fingerprint": _plan_fingerprint(plan),
                    "graph_fingerprint": result["graph_fingerprint"],
                    "execution_identity": execution_identity,
                    "branch_outputs": by_branch,
                    "telemetry": telemetry,
                }
            ).encode("utf-8")
        ).hexdigest(),
    }
