"""Decisive evidence contracts for resident ScienceGraph phases P4-P8."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContinuousBatchComputeEvidence:
    """Direct executor evidence; deliberately excludes service queue promotion."""

    batch_sizes: tuple[int, ...]
    numerical_contract: str
    parity_passed: bool
    row_interference_passed: bool
    max_independent_b1_delta: float
    p95_execution_ms: float
    throughput_speedup_lower_95: float
    arena_bytes: int
    peak_vram_delta_bytes: int

    def __post_init__(self) -> None:
        if self.batch_sizes != (2, 4, 8, 16, 32):
            raise ValueError("P4 compute evidence requires B={2,4,8,16,32}")
        if not self.numerical_contract:
            raise ValueError("P4 compute evidence requires a numerical contract")
        if not self.parity_passed or not self.row_interference_passed:
            raise ValueError("P4 compute parity and interference gates are required")
        if not math.isfinite(self.max_independent_b1_delta) or self.max_independent_b1_delta < 0:
            raise ValueError("P4 independent-B1 delta must be finite and non-negative")
        _positive(self.p95_execution_ms, "p95_execution_ms")
        if _positive(self.throughput_speedup_lower_95, "throughput_speedup_lower_95") <= 1:
            raise ValueError("P4 compute throughput confidence bound must exceed one")
        if self.arena_bytes <= 0 or self.peak_vram_delta_bytes <= 0:
            raise ValueError("P4 compute evidence requires arena and peak VRAM bytes")

    @classmethod
    def from_gate_artifact(cls, payload: Mapping[str, Any]) -> ContinuousBatchComputeEvidence:
        if payload.get("schema") != "mrun-resident-continuous-batch-cuda-gate-v1":
            raise ValueError("unsupported P4 continuous-batch artifact")
        cells = tuple(payload.get("cells", ()))
        batches = tuple(int(cell["batch_size"]) for cell in cells)
        atol = float(payload["atol"])
        parity = bool(payload.get("qualified")) and all(
            bool(cell["independent_b1_exact_winners"])
            and float(cell["independent_b1_max_abs_score_delta"]) <= atol
            and float(cell["max_abs_score_delta"]) == 0.0
            for cell in cells
        )
        return cls(
            batch_sizes=batches,
            numerical_contract=str(payload["numerical_contract"]),
            parity_passed=parity,
            row_interference_passed=all(
                bool(cell["row_interference_passed"]) for cell in cells
            ),
            max_independent_b1_delta=max(
                float(cell["independent_b1_max_abs_score_delta"]) for cell in cells
            ),
            p95_execution_ms=max(float(cell["p95_graph_seconds"]) for cell in cells) * 1_000,
            throughput_speedup_lower_95=min(
                float(cell["speedup_lower_95"]) for cell in cells
            ),
            arena_bytes=int(payload["arena_bytes"]),
            peak_vram_delta_bytes=max(
                int(cell["executor_evidence"]["capture_peak_allocated_delta_bytes"])
                for cell in cells
            ),
        )


def _positive(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


@dataclass(frozen=True)
class ContinuousBatchEvidence:
    batch_sizes: tuple[int, ...]
    parity_passed: bool
    row_interference_passed: bool
    p50_execution_ms: float
    p95_execution_ms: float
    p95_queue_ms: float
    throughput_speedup_lower_95: float
    peak_vram_bytes: int

    def __post_init__(self) -> None:
        if self.batch_sizes != (2, 4, 8, 16, 32):
            raise ValueError("P4 evidence requires the complete B={2,4,8,16,32} family")
        if not self.parity_passed or not self.row_interference_passed:
            raise ValueError("P4 requires row parity and interference gates")
        for field in ("p50_execution_ms", "p95_execution_ms", "p95_queue_ms"):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        if self.p95_execution_ms < self.p50_execution_ms:
            raise ValueError("P4 p95 execution latency cannot be below p50")
        _positive(self.throughput_speedup_lower_95, "throughput_speedup_lower_95")
        if self.throughput_speedup_lower_95 <= 1:
            raise ValueError("P4 throughput confidence bound must exceed one")
        if self.peak_vram_bytes <= 0:
            raise ValueError("P4 peak VRAM must be recorded")

    @classmethod
    def from_compute_and_service_artifacts(
        cls,
        compute: ContinuousBatchComputeEvidence,
        service_payload: Mapping[str, Any],
    ) -> ContinuousBatchEvidence:
        if (
            service_payload.get("schema")
            != "mrun-resident-continuous-batch-service-cuda-gate-v1"
            or not bool(service_payload.get("qualified"))
        ):
            raise ValueError("P4 service artifact is absent or unqualified")
        if service_payload.get("numerical_contract") != compute.numerical_contract:
            raise ValueError("P4 compute and service numerical contracts differ")
        cells = tuple(service_payload.get("cells", ()))
        batches = tuple(int(cell["batch_size"]) for cell in cells)
        if batches != compute.batch_sizes:
            raise ValueError("P4 compute and service batch families differ")
        telemetry = service_payload["service_telemetry"]
        if (
            not bool(service_payload.get("cancellation_passed"))
            or not bool(service_payload.get("refill_passed"))
            or int(telemetry["fallback_rows"]) != 0
            or int(telemetry["captured_max_width"]) != 32
        ):
            raise ValueError("P4 service cancellation, refill, fallback, or occupancy gate failed")
        return cls(
            batch_sizes=batches,
            parity_passed=compute.parity_passed
            and all(float(cell["max_abs_score_delta"]) == 0 for cell in cells),
            row_interference_passed=compute.row_interference_passed,
            p50_execution_ms=float(telemetry["execution_p50_ms"]),
            p95_execution_ms=float(telemetry["execution_p95_ms"]),
            p95_queue_ms=float(telemetry["queue_p95_ms"]),
            throughput_speedup_lower_95=min(
                float(cell["speedup_lower_95"]) for cell in cells
            ),
            peak_vram_bytes=int(service_payload["peak_vram_bytes"]),
        )


@dataclass(frozen=True)
class ProfileDrivenFusionEvidence:
    shape_cells: int
    parity_passed: bool
    kernel_count_reduction: int
    bytes_removed: int
    compile_graph_factorial_complete: bool
    speedup_lower_95: float

    def __post_init__(self) -> None:
        if self.shape_cells < 9:
            raise ValueError("P5 requires at least the 3x3 shape matrix")
        if not self.parity_passed or not self.compile_graph_factorial_complete:
            raise ValueError("P5 parity and compile-by-graph factorial are required")
        if self.kernel_count_reduction <= 0 or self.bytes_removed <= 0:
            raise ValueError("P5 must remove kernels and bytes")
        if _positive(self.speedup_lower_95, "speedup_lower_95") <= 1.05:
            raise ValueError("P5 confidence bound must exceed 1.05x")


@dataclass(frozen=True)
class StatefulDecodeEvidence:
    generated_tokens: int
    interleaved_slot_operations: int
    sustained_decode_steps: int
    token_parity_passed: bool
    state_parity_passed: bool
    cancellation_passed: bool
    rollback_passed: bool
    refill_passed: bool
    overflow_passed: bool
    flat_allocations: bool
    b1_itl_speedup_lower_95: float

    def __post_init__(self) -> None:
        if self.generated_tokens < 256 or self.interleaved_slot_operations < 1_000:
            raise ValueError("P6 token and interleaved-operation floors were not met")
        if self.sustained_decode_steps < 32:
            raise ValueError("P6 requires at least 32 sustained decode steps")
        required = (
            self.token_parity_passed,
            self.state_parity_passed,
            self.cancellation_passed,
            self.rollback_passed,
            self.refill_passed,
            self.overflow_passed,
            self.flat_allocations,
        )
        if not all(required):
            raise ValueError("P6 transactional decode gates did not all pass")
        if _positive(self.b1_itl_speedup_lower_95, "b1_itl_speedup_lower_95") <= 1:
            raise ValueError("P6 B1 ITL confidence bound must exceed one")


@dataclass(frozen=True)
class TargetAlignedK4Evidence:
    target_passes: int
    useful_outputs: int
    exact_transactional_behavior: bool
    long_horizon_agreement: bool
    rejection_overhead_included: bool
    complete_wall_speedup_lower_95: float

    def __post_init__(self) -> None:
        if self.target_passes <= 0 or self.useful_outputs / self.target_passes < 2:
            raise ValueError("P7 requires at least two useful outputs per target pass")
        if not (
            self.exact_transactional_behavior
            and self.long_horizon_agreement
            and self.rejection_overhead_included
        ):
            raise ValueError("P7 correctness and rejection-overhead gates are required")
        if _positive(self.complete_wall_speedup_lower_95, "complete_wall_speedup_lower_95") <= 1:
            raise ValueError("P7 complete-wall confidence bound must exceed one")


@dataclass(frozen=True)
class PrecisionOversizedEvidence:
    fp8_dense_promoted: bool
    int4_dense_promoted: bool
    route_first_moe_promoted: bool
    quality_passed: bool
    active_bytes_recorded: bool
    cache_pressure_sweep_passed: bool

    def __post_init__(self) -> None:
        if not all(
            (
                self.fp8_dense_promoted,
                self.int4_dense_promoted,
                self.route_first_moe_promoted,
                self.quality_passed,
                self.active_bytes_recorded,
                self.cache_pressure_sweep_passed,
            )
        ):
            raise ValueError("P8 requires every precision, quality, byte, and pressure gate")


__all__ = [
    "ContinuousBatchComputeEvidence",
    "ContinuousBatchEvidence",
    "PrecisionOversizedEvidence",
    "ProfileDrivenFusionEvidence",
    "StatefulDecodeEvidence",
    "TargetAlignedK4Evidence",
]
