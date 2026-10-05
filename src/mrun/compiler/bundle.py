"""Compilation bundles joining semantic, schedule, memory, and cost contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from .cost import CostAssumptions, CostEstimate, estimate_workplan_cost
from .graph import build_op_graph, build_output_demand_graph
from .graph_passes import GraphCompilation, compile_graph
from .identity import validate_plan_identity_certificate
from .ir import DenseWorkPlan, ExecutionMode
from .lowering import LoweredWorkPlan, lower_work_plan
from .memory import MemoryPlan, plan_qstore_memory
from .work_floor import (
    WorkFloorComparison,
    analyze_work_floor,
    compare_work_floors,
)

COMPILATION_BUNDLE_SCHEMA = "mrun-compilation-bundle-v2"
COMPILER_VERSION = "mrun-workplan-graph-4"


@dataclass(frozen=True)
class CompilationBundle:
    plan: DenseWorkPlan
    lowered: LoweredWorkPlan
    memory: MemoryPlan | None = None
    cost: CostEstimate | None = None
    graph: GraphCompilation | None = None
    work_floor: WorkFloorComparison | None = None
    compiler_version: str = COMPILER_VERSION
    schema_version: str = COMPILATION_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != COMPILATION_BUNDLE_SCHEMA:
            raise ValueError(f"unsupported compilation-bundle schema: {self.schema_version}")
        if self.lowered.plan_fingerprint != self.plan.fingerprint:
            raise ValueError("lowered schedule does not belong to bundle WorkPlan")
        if self.lowered.content_identity_verified != self.plan.content_identity_verified:
            raise ValueError("lowered content-identity verdict does not match bundle WorkPlan")
        if self.plan.execution_mode in {ExecutionMode.PREFILL, ExecutionMode.DECODE} and (
            self.graph is not None or self.work_floor is not None
        ):
            raise ValueError("prefill/decode bundles cannot claim an OpGraph or work-floor model")
        if self.graph is not None:
            graph = self.graph.graph
            if (
                graph.model_name != self.plan.model_name
                or graph.model_revision != self.plan.model_revision
                or graph.store_fingerprint != self.plan.store_fingerprint
                or graph.numerical_contract != self.plan.numerical_contract
            ):
                raise ValueError("compiled graph does not belong to bundle WorkPlan")
        if self.work_floor is not None:
            if self.graph is None:
                raise ValueError("work-floor comparison requires a compiled graph")
            if (
                self.work_floor.source.resources.graph_fingerprint
                != self.graph.source_graph_fingerprint
                or self.work_floor.rewritten.resources.graph_fingerprint
                != self.graph.graph.fingerprint
            ):
                raise ValueError("work-floor comparison does not belong to compiled graph")
            if self.work_floor.output_contract != self.plan.output_contract.value:
                raise ValueError("work-floor comparison has the wrong output contract")
            expected_rewrite_ids = (
                ()
                if self.graph.rewrite_certificate is None
                else self.graph.rewrite_certificate.rewrite_ids
            )
            if self.work_floor.rewrite_ids != expected_rewrite_ids:
                raise ValueError("work-floor rewrite IDs do not match compiled graph")
            expected_rewritten_floor = analyze_work_floor(self.graph.graph, self.plan)
            if self.work_floor.rewritten != expected_rewritten_floor:
                raise ValueError("rewritten work-floor analysis does not match graph")
            if self.graph.rewrite_certificate is None and (
                self.work_floor.source != expected_rewritten_floor
            ):
                raise ValueError("unrewritten work-floor source does not match graph")
        if not self.compiler_version:
            raise ValueError("compiler_version must be non-empty")

    def _payload_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "compiler_version": self.compiler_version,
            "plan": self.plan.as_dict(),
            "lowered": self.lowered.as_dict(),
            "memory": None if self.memory is None else self.memory.as_dict(),
            "cost": None if self.cost is None else self.cost.as_dict(),
            "graph": None if self.graph is None else self.graph.as_dict(),
        }
        if self.work_floor is not None:
            payload["work_floor"] = self.work_floor.as_dict()
        return payload

    @cached_property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._payload_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload_dict(), "bundle_fingerprint": self.fingerprint}

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CompilationBundle:
        plan_payload = payload.get("plan")
        lowered_payload = payload.get("lowered")
        memory_payload = payload.get("memory")
        cost_payload = payload.get("cost")
        graph_payload = payload.get("graph")
        work_floor_payload = payload.get("work_floor")
        if not isinstance(plan_payload, dict) or not isinstance(lowered_payload, dict):
            raise TypeError("bundle plan and lowered schedule must be objects")
        bundle = cls(
            plan=DenseWorkPlan.from_dict(plan_payload),
            lowered=LoweredWorkPlan.from_dict(lowered_payload),
            memory=(
                None
                if memory_payload is None
                else MemoryPlan.from_dict(memory_payload)
                if isinstance(memory_payload, dict)
                else _raise_type("bundle memory must be an object or null")
            ),
            cost=(
                None
                if cost_payload is None
                else CostEstimate.from_dict(cost_payload)
                if isinstance(cost_payload, dict)
                else _raise_type("bundle cost must be an object or null")
            ),
            graph=(
                None
                if graph_payload is None
                else GraphCompilation.from_dict(graph_payload)
                if isinstance(graph_payload, dict)
                else _raise_type("bundle graph must be an object or null")
            ),
            work_floor=(
                None
                if work_floor_payload is None
                else WorkFloorComparison.from_dict(work_floor_payload)
                if isinstance(work_floor_payload, dict)
                else _raise_type("bundle work_floor must be an object or null")
            ),
            compiler_version=str(payload.get("compiler_version", COMPILER_VERSION)),
            schema_version=str(payload.get("schema_version", COMPILATION_BUNDLE_SCHEMA)),
        )
        claimed = payload.get("bundle_fingerprint")
        if claimed is not None and str(claimed) != bundle.fingerprint:
            raise ValueError("compilation bundle fingerprint mismatch")
        return bundle

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> CompilationBundle:
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise TypeError("serialized compilation bundle must be a JSON object")
        return cls.from_dict(decoded)


def _raise_type(message: str) -> Any:
    raise TypeError(message)


def compile_work_plan(
    plan: DenseWorkPlan,
    backend: str,
    *,
    manifest: dict[str, Any] | None = None,
    cost_assumptions: CostAssumptions | None = None,
) -> CompilationBundle:
    """Lower a WorkPlan and optionally attach manifest-derived resource/cost estimates."""

    if cost_assumptions is not None and manifest is None:
        raise ValueError("cost estimation requires a QStore manifest")
    validate_plan_identity_certificate(plan, manifest)
    lowered = lower_work_plan(plan, backend)
    memory = None if manifest is None else plan_qstore_memory(plan, manifest)
    graph: GraphCompilation | None = None
    work_floor: WorkFloorComparison | None = None
    stateful = plan.execution_mode in {ExecutionMode.PREFILL, ExecutionMode.DECODE}
    if stateful and cost_assumptions is not None:
        raise ValueError(
            "stateful WorkPlan cost is not modeled: committed-prefix length is runtime state"
        )
    if manifest is not None and not stateful:
        architecture = str(manifest.get("arch", "")).strip().lower()
        if architecture in {"qwen2", "qwen3", "llama"}:
            source_graph, rewritten_graph, certificate = build_output_demand_graph(
                plan,
                manifest,
            )
            graph = compile_graph(
                rewritten_graph,
                source_graph_fingerprint=source_graph.fingerprint,
                rewrite_certificate=certificate,
                metadata={
                    "analysis_status": "hardware-independent",
                    "placement_status": "not-requested",
                },
            )
            work_floor = compare_work_floors(
                source_graph,
                rewritten_graph,
                plan,
                certificate,
            )
        else:
            resource_graph = build_op_graph(
                plan,
                manifest,
                allow_generic_manifest=True,
            )
            graph = compile_graph(
                resource_graph,
                metadata={
                    "analysis_status": "resource-graph-only",
                    "placement_status": "not-requested",
                },
            )
            work_floor = compare_work_floors(
                resource_graph,
                resource_graph,
                plan,
            )
    cost = (
        None
        if cost_assumptions is None or memory is None or manifest is None
        else estimate_workplan_cost(plan, lowered, manifest, memory, cost_assumptions)
    )
    return CompilationBundle(
        plan=plan,
        lowered=lowered,
        memory=memory,
        cost=cost,
        graph=graph,
        work_floor=work_floor,
    )
