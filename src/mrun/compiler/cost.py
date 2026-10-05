"""Roofline-style WorkPlan cost estimates with explicit assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ir import DenseWorkPlan, OutputContract
from .lowering import LoweredWorkPlan
from .memory import MemoryPlan, resolve_manifest_block

COST_ESTIMATE_SCHEMA = "mrun-cost-estimate-v1"


@dataclass(frozen=True)
class CostAssumptions:
    peak_compute_ops_per_s: float
    memory_bandwidth_bytes_per_s: float
    launch_overhead_s: float = 0.0
    boundary_overhead_s: float = 0.0
    label: str = "user-supplied"

    def __post_init__(self) -> None:
        if self.peak_compute_ops_per_s <= 0:
            raise ValueError("peak_compute_ops_per_s must be positive")
        if self.memory_bandwidth_bytes_per_s <= 0:
            raise ValueError("memory_bandwidth_bytes_per_s must be positive")
        if self.launch_overhead_s < 0 or self.boundary_overhead_s < 0:
            raise ValueError("overhead assumptions cannot be negative")
        if not self.label:
            raise ValueError("assumption label must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_compute_ops_per_s": self.peak_compute_ops_per_s,
            "memory_bandwidth_bytes_per_s": self.memory_bandwidth_bytes_per_s,
            "launch_overhead_s": self.launch_overhead_s,
            "boundary_overhead_s": self.boundary_overhead_s,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CostAssumptions:
        return cls(
            peak_compute_ops_per_s=float(payload["peak_compute_ops_per_s"]),
            memory_bandwidth_bytes_per_s=float(payload["memory_bandwidth_bytes_per_s"]),
            launch_overhead_s=float(payload.get("launch_overhead_s", 0.0)),
            boundary_overhead_s=float(payload.get("boundary_overhead_s", 0.0)),
            label=str(payload.get("label", "user-supplied")),
        )


@dataclass(frozen=True)
class CostEstimate:
    operation_count: int
    bytes_moved: int
    compute_floor_s: float
    bandwidth_floor_s: float
    dispatch_overhead_s: float
    boundary_overhead_s: float
    estimated_total_s: float
    arithmetic_intensity_ops_per_byte: float
    assumptions: CostAssumptions
    estimate_basis: str = "roofline-design-estimate-not-measured"
    schema_version: str = COST_ESTIMATE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operation_count": self.operation_count,
            "bytes_moved": self.bytes_moved,
            "compute_floor_s": self.compute_floor_s,
            "bandwidth_floor_s": self.bandwidth_floor_s,
            "dispatch_overhead_s": self.dispatch_overhead_s,
            "boundary_overhead_s": self.boundary_overhead_s,
            "estimated_total_s": self.estimated_total_s,
            "arithmetic_intensity_ops_per_byte": self.arithmetic_intensity_ops_per_byte,
            "assumptions": self.assumptions.as_dict(),
            "estimate_basis": self.estimate_basis,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CostEstimate:
        schema = str(payload.get("schema_version", COST_ESTIMATE_SCHEMA))
        if schema != COST_ESTIMATE_SCHEMA:
            raise ValueError(f"unsupported cost-estimate schema: {schema}")
        assumptions = payload.get("assumptions")
        if not isinstance(assumptions, dict):
            raise TypeError("cost assumptions must be an object")
        return cls(
            operation_count=int(payload["operation_count"]),
            bytes_moved=int(payload["bytes_moved"]),
            compute_floor_s=float(payload["compute_floor_s"]),
            bandwidth_floor_s=float(payload["bandwidth_floor_s"]),
            dispatch_overhead_s=float(payload["dispatch_overhead_s"]),
            boundary_overhead_s=float(payload["boundary_overhead_s"]),
            estimated_total_s=float(payload["estimated_total_s"]),
            arithmetic_intensity_ops_per_byte=float(payload["arithmetic_intensity_ops_per_byte"]),
            assumptions=CostAssumptions.from_dict(assumptions),
            estimate_basis=str(
                payload.get("estimate_basis", "roofline-design-estimate-not-measured")
            ),
            schema_version=schema,
        )


def _head_operation_count(plan: DenseWorkPlan, vocab: int, hidden: int) -> int:
    head_output_pushdown = bool(dict(plan.metadata).get("head_output_pushdown", False))
    if not head_output_pushdown and plan.output_contract in {
        OutputContract.LAST_TOKEN_LOGITS,
        OutputContract.SELECTED_TOKEN_ROWS,
        OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
    }:
        return 2 * plan.shape.live_token_rows * vocab * hidden
    if plan.output_contract in {OutputContract.FULL_LOGITS, OutputContract.LOSS_ONLY}:
        rows = plan.shape.live_token_rows
        columns = vocab
    elif plan.output_contract is OutputContract.LAST_TOKEN_LOGITS:
        rows = plan.shape.actual_batch
        columns = vocab
    elif plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
        rows = plan.shape.actual_batch
        columns = len(plan.required_output_rows)
    elif plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
        # The row-addressable head computes the union once for the batch. Each
        # batch row consumes the resulting candidate columns independently.
        columns = len(
            {token for row_candidates in plan.candidate_token_ids for token in row_candidates}
        )
        rows = plan.shape.actual_batch
    elif plan.output_contract is OutputContract.HIDDEN_STATE_ONLY:
        return 0
    else:
        return 0
    return 2 * rows * columns * hidden


def estimate_workplan_cost(
    plan: DenseWorkPlan,
    lowered: LoweredWorkPlan,
    manifest: dict[str, Any],
    memory: MemoryPlan,
    assumptions: CostAssumptions,
) -> CostEstimate:
    """Estimate a dense transformer region without presenting it as measured performance."""

    live_rows = plan.shape.live_token_rows
    operations = 0
    for name in plan.page_sequence:
        try:
            physical_name, block = resolve_manifest_block(manifest, name)
        except KeyError:
            continue
        shape = block.get("shape")
        if (
            not str(block.get("kind", "")).startswith("qrow")
            or not isinstance(shape, list)
            or len(shape) != 2
        ):
            continue
        if name == "lm_head":
            operations += _head_operation_count(plan, int(shape[0]), int(shape[1]))
            continue
        # An aliased lm_head must still execute separately from the embedding lookup.
        if physical_name == "embed" and name != "lm_head":
            continue
        operations += 2 * live_rows * int(shape[0]) * int(shape[1])

    config = manifest.get("config", {})
    layers = int(config.get("num_hidden_layers", 0))
    hidden = int(config.get("hidden_size", 0))
    heads = int(config.get("num_attention_heads", 0))
    head_dim = int(config.get("head_dim", hidden // max(heads, 1)))
    batch = plan.shape.actual_batch
    sequence = plan.shape.sequence_length
    attention_ops = 4 * layers * batch * heads * sequence * sequence * head_dim
    elementwise_ops = 10 * layers * live_rows * hidden
    operations += attention_ops + elementwise_ops

    bytes_moved = (
        memory.estimated_compact_read_bytes
        + memory.output_bytes
        + memory.kv_allocated_bytes
        + memory.workspace_bytes
    )
    compute_floor = operations / assumptions.peak_compute_ops_per_s
    bandwidth_floor = bytes_moved / assumptions.memory_bandwidth_bytes_per_s
    dispatch = len(lowered.steps) * assumptions.launch_overhead_s
    boundaries = sum(
        left.target != right.target
        for left, right in zip(lowered.steps, lowered.steps[1:], strict=False)
    )
    boundary = boundaries * assumptions.boundary_overhead_s
    total = max(compute_floor, bandwidth_floor) + dispatch + boundary
    intensity = operations / max(bytes_moved, 1)
    return CostEstimate(
        operation_count=operations,
        bytes_moved=bytes_moved,
        compute_floor_s=compute_floor,
        bandwidth_floor_s=bandwidth_floor,
        dispatch_overhead_s=dispatch,
        boundary_overhead_s=boundary,
        estimated_total_s=total,
        arithmetic_intensity_ops_per_byte=intensity,
        assumptions=assumptions,
    )
