"""Canonical, backend-neutral performance evidence for native model execution.

The schema makes quantities that are routinely conflated impossible to serialize under one
unlabelled ``tokens_per_second`` field.  A sample binds the complete executable tuple and declares
whether its work units are autonomous committed tokens, teacher-forced positions, or prefill input
positions.  The roofline helper separates ideal overlap from a serial traffic bound; neither is a
performance claim until a hardware sample is attached.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

NATIVE_BENCHMARK_KEY_SCHEMA = "mrun-native-benchmark-key-v1"
NATIVE_BENCHMARK_SAMPLE_SCHEMA = "mrun-native-benchmark-sample-v1"
NATIVE_BENCHMARK_CAMPAIGN_SCHEMA = "mrun-native-benchmark-campaign-v1"
NATIVE_ROOFLINE_SCHEMA = "mrun-native-roofline-v1"


class WorkKind(str, Enum):
    AUTONOMOUS_DECODE = "autonomous-committed-decode-tokens"
    TEACHER_FORCED_DECODE = "teacher-forced-decode-positions"
    PREFILL = "prefill-input-positions"


class ThermalState(str, Enum):
    NOMINAL = "nominal"
    FAIR = "fair"
    SERIOUS = "serious"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _name(value: str, field: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _sha256(value: str, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _nonnegative_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _positive_float(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return result


def _nonnegative_float(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


@dataclass(frozen=True, slots=True)
class NativeBenchmarkKey:
    """Complete executable/workload identity for one comparable sample population."""

    model_fingerprint: str
    artifact_fingerprint: str
    lowering_fingerprint: str
    placement_fingerprint: str
    device_fingerprint: str
    backend_id: str
    numerical_contract: str
    codec_id: str
    state_abi: str
    output_contract: str
    work_kind: WorkKind | str
    batch_size: int
    prompt_tokens_per_row: int
    context_tokens_before_work: int
    requested_output_tokens_per_row: int
    residency: str
    arrival_shape: str
    session_shape: str
    schema_version: str = NATIVE_BENCHMARK_KEY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_BENCHMARK_KEY_SCHEMA:
            raise ValueError("unsupported native benchmark key schema")
        for field in (
            "model_fingerprint",
            "artifact_fingerprint",
            "lowering_fingerprint",
            "placement_fingerprint",
            "device_fingerprint",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        for field in (
            "backend_id",
            "numerical_contract",
            "codec_id",
            "state_abi",
            "output_contract",
            "residency",
            "arrival_shape",
            "session_shape",
        ):
            object.__setattr__(self, field, _name(getattr(self, field), field))
        try:
            work_kind = (
                self.work_kind
                if isinstance(self.work_kind, WorkKind)
                else WorkKind(str(self.work_kind))
            )
        except ValueError as exc:
            raise ValueError("unsupported native benchmark work kind") from exc
        object.__setattr__(self, "work_kind", work_kind)
        object.__setattr__(self, "batch_size", _positive_int(self.batch_size, "batch_size"))
        for field in (
            "prompt_tokens_per_row",
            "context_tokens_before_work",
            "requested_output_tokens_per_row",
        ):
            object.__setattr__(self, field, _nonnegative_int(getattr(self, field), field))
        if work_kind is WorkKind.PREFILL:
            if self.prompt_tokens_per_row <= 0:
                raise ValueError("prefill benchmarks require positive prompt_tokens_per_row")
        elif self.requested_output_tokens_per_row <= 0:
            raise ValueError("decode benchmarks require requested output tokens")

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_fingerprint": self.model_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "lowering_fingerprint": self.lowering_fingerprint,
            "placement_fingerprint": self.placement_fingerprint,
            "device_fingerprint": self.device_fingerprint,
            "backend_id": self.backend_id,
            "numerical_contract": self.numerical_contract,
            "codec_id": self.codec_id,
            "state_abi": self.state_abi,
            "output_contract": self.output_contract,
            "work_kind": self.work_kind.value,
            "batch_size": self.batch_size,
            "prompt_tokens_per_row": self.prompt_tokens_per_row,
            "context_tokens_before_work": self.context_tokens_before_work,
            "requested_output_tokens_per_row": self.requested_output_tokens_per_row,
            "residency": self.residency,
            "arrival_shape": self.arrival_shape,
            "session_shape": self.session_shape,
        }


@dataclass(frozen=True, slots=True)
class NativeBenchmarkSample:
    key: NativeBenchmarkKey
    acquisition_index: int
    launch_order: str
    elapsed_seconds: float
    work_units: int
    committed_output_tokens: int
    input_positions: int
    physical_weight_bytes_read: int
    kv_bytes_read: int
    kv_bytes_written: int
    host_to_device_bytes: int
    device_to_host_bytes: int
    peak_resident_bytes: int
    average_power_watts: float | None = None
    thermal_state: ThermalState | str = ThermalState.UNKNOWN
    unexpected_fallbacks: int = 0
    unexpected_page_loads: int = 0
    schema_version: str = NATIVE_BENCHMARK_SAMPLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_BENCHMARK_SAMPLE_SCHEMA:
            raise ValueError("unsupported native benchmark sample schema")
        if not isinstance(self.key, NativeBenchmarkKey):
            raise TypeError("key must be a NativeBenchmarkKey")
        object.__setattr__(
            self, "acquisition_index", _nonnegative_int(self.acquisition_index, "acquisition_index")
        )
        object.__setattr__(self, "launch_order", _name(self.launch_order, "launch_order"))
        object.__setattr__(
            self, "elapsed_seconds", _positive_float(self.elapsed_seconds, "elapsed_seconds")
        )
        object.__setattr__(self, "work_units", _positive_int(self.work_units, "work_units"))
        for field in (
            "committed_output_tokens",
            "input_positions",
            "physical_weight_bytes_read",
            "kv_bytes_read",
            "kv_bytes_written",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "peak_resident_bytes",
            "unexpected_fallbacks",
            "unexpected_page_loads",
        ):
            object.__setattr__(self, field, _nonnegative_int(getattr(self, field), field))
        if self.average_power_watts is not None:
            object.__setattr__(
                self,
                "average_power_watts",
                _positive_float(self.average_power_watts, "average_power_watts"),
            )
        try:
            thermal = (
                self.thermal_state
                if isinstance(self.thermal_state, ThermalState)
                else ThermalState(str(self.thermal_state))
            )
        except ValueError as exc:
            raise ValueError("unsupported thermal state") from exc
        object.__setattr__(self, "thermal_state", thermal)

        if self.key.work_kind is WorkKind.AUTONOMOUS_DECODE:
            if self.work_units != self.committed_output_tokens:
                raise ValueError("autonomous work units must equal committed output tokens")
        elif self.key.work_kind is WorkKind.PREFILL:
            if self.work_units != self.input_positions or self.committed_output_tokens:
                raise ValueError("prefill work units must equal input positions and emit no tokens")
        elif self.committed_output_tokens:
            raise ValueError("teacher-forced positions cannot be labelled committed output tokens")

    @property
    def units_per_second(self) -> float:
        return self.work_units / self.elapsed_seconds

    @property
    def joules_per_unit(self) -> float | None:
        if self.average_power_watts is None:
            return None
        return self.average_power_watts * self.elapsed_seconds / self.work_units

    @property
    def comparable(self) -> bool:
        return not self.unexpected_fallbacks and not self.unexpected_page_loads

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key": self.key.as_dict(),
            "key_fingerprint": self.key.fingerprint,
            "acquisition_index": self.acquisition_index,
            "launch_order": self.launch_order,
            "elapsed_seconds": self.elapsed_seconds,
            "work_units": self.work_units,
            "work_unit_kind": self.key.work_kind.value,
            "units_per_second": self.units_per_second,
            "committed_output_tokens": self.committed_output_tokens,
            "input_positions": self.input_positions,
            "physical_weight_bytes_read": self.physical_weight_bytes_read,
            "kv_bytes_read": self.kv_bytes_read,
            "kv_bytes_written": self.kv_bytes_written,
            "host_to_device_bytes": self.host_to_device_bytes,
            "device_to_host_bytes": self.device_to_host_bytes,
            "peak_resident_bytes": self.peak_resident_bytes,
            "average_power_watts": self.average_power_watts,
            "joules_per_unit": self.joules_per_unit,
            "thermal_state": self.thermal_state.value,
            "unexpected_fallbacks": self.unexpected_fallbacks,
            "unexpected_page_loads": self.unexpected_page_loads,
            "comparable": self.comparable,
        }


@dataclass(frozen=True, slots=True)
class NativeBenchmarkCampaign:
    """Repeated acquisition-ordered observations for exactly one benchmark key."""

    samples: tuple[NativeBenchmarkSample, ...]
    schema_version: str = NATIVE_BENCHMARK_CAMPAIGN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_BENCHMARK_CAMPAIGN_SCHEMA:
            raise ValueError("unsupported native benchmark campaign schema")
        samples = tuple(self.samples)
        if len(samples) < 2 or any(
            not isinstance(sample, NativeBenchmarkSample) for sample in samples
        ):
            raise ValueError("a native benchmark campaign requires at least two samples")
        if len({sample.key.fingerprint for sample in samples}) != 1:
            raise ValueError("campaign samples must bind one exact benchmark key")
        indices = tuple(sample.acquisition_index for sample in samples)
        if indices != tuple(range(len(samples))):
            raise ValueError("campaign samples must be in contiguous acquisition order")
        object.__setattr__(self, "samples", samples)

    @property
    def median_units_per_second(self) -> float:
        return statistics.median(sample.units_per_second for sample in self.samples)

    @property
    def p95_units_per_second(self) -> float:
        ordered = sorted(sample.units_per_second for sample in self.samples)
        position = (len(ordered) - 1) * 0.95
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    @property
    def all_comparable(self) -> bool:
        return all(sample.comparable for sample in self.samples)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key_fingerprint": self.samples[0].key.fingerprint,
            "sample_count": len(self.samples),
            "median_units_per_second": self.median_units_per_second,
            "p95_units_per_second": self.p95_units_per_second,
            "all_comparable": self.all_comparable,
            "samples": [sample.as_dict() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class NativeRoofline:
    """Per-unit physical lower bounds under ideal overlap and fully serial traffic."""

    weight_bytes: int
    kv_read_bytes: int
    kv_write_bytes: int
    transfer_bytes: int
    floating_point_operations: int
    memory_bandwidth_bytes_per_second: float
    transfer_bandwidth_bytes_per_second: float
    compute_operations_per_second: float
    fixed_seconds_per_unit: float = 0.0
    schema_version: str = NATIVE_ROOFLINE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_ROOFLINE_SCHEMA:
            raise ValueError("unsupported native roofline schema")
        for field in (
            "weight_bytes",
            "kv_read_bytes",
            "kv_write_bytes",
            "transfer_bytes",
            "floating_point_operations",
        ):
            object.__setattr__(self, field, _nonnegative_int(getattr(self, field), field))
        for field in (
            "memory_bandwidth_bytes_per_second",
            "transfer_bandwidth_bytes_per_second",
            "compute_operations_per_second",
        ):
            object.__setattr__(self, field, _positive_float(getattr(self, field), field))
        object.__setattr__(
            self,
            "fixed_seconds_per_unit",
            _nonnegative_float(self.fixed_seconds_per_unit, "fixed_seconds_per_unit"),
        )
        if not any(
            (
                self.weight_bytes,
                self.kv_read_bytes,
                self.kv_write_bytes,
                self.transfer_bytes,
                self.floating_point_operations,
                self.fixed_seconds_per_unit,
            )
        ):
            raise ValueError("roofline requires at least one non-zero cost")

    @property
    def memory_seconds(self) -> float:
        return (
            self.weight_bytes + self.kv_read_bytes + self.kv_write_bytes
        ) / self.memory_bandwidth_bytes_per_second

    @property
    def transfer_seconds(self) -> float:
        return self.transfer_bytes / self.transfer_bandwidth_bytes_per_second

    @property
    def compute_seconds(self) -> float:
        return self.floating_point_operations / self.compute_operations_per_second

    @property
    def ideal_overlap_seconds(self) -> float:
        return self.fixed_seconds_per_unit + max(
            self.memory_seconds, self.transfer_seconds, self.compute_seconds
        )

    @property
    def serial_seconds(self) -> float:
        return (
            self.fixed_seconds_per_unit
            + self.memory_seconds
            + self.transfer_seconds
            + self.compute_seconds
        )

    @property
    def ideal_overlap_ceiling_units_per_second(self) -> float:
        return 1.0 / self.ideal_overlap_seconds

    @property
    def serial_ceiling_units_per_second(self) -> float:
        return 1.0 / self.serial_seconds

    def utilization(self, measured_units_per_second: float) -> float:
        measured = _positive_float(measured_units_per_second, "measured_units_per_second")
        return measured / self.ideal_overlap_ceiling_units_per_second

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "weight_bytes": self.weight_bytes,
            "kv_read_bytes": self.kv_read_bytes,
            "kv_write_bytes": self.kv_write_bytes,
            "transfer_bytes": self.transfer_bytes,
            "floating_point_operations": self.floating_point_operations,
            "memory_bandwidth_bytes_per_second": self.memory_bandwidth_bytes_per_second,
            "transfer_bandwidth_bytes_per_second": self.transfer_bandwidth_bytes_per_second,
            "compute_operations_per_second": self.compute_operations_per_second,
            "fixed_seconds_per_unit": self.fixed_seconds_per_unit,
            "memory_seconds": self.memory_seconds,
            "transfer_seconds": self.transfer_seconds,
            "compute_seconds": self.compute_seconds,
            "ideal_overlap_seconds": self.ideal_overlap_seconds,
            "serial_seconds": self.serial_seconds,
            "ideal_overlap_ceiling_units_per_second": (self.ideal_overlap_ceiling_units_per_second),
            "serial_ceiling_units_per_second": self.serial_ceiling_units_per_second,
        }


def compare_campaigns(
    baseline: NativeBenchmarkCampaign,
    candidate: NativeBenchmarkCampaign,
) -> dict[str, Any]:
    """Compare campaigns only when their workload tuple differs solely by executable identity."""

    if not isinstance(baseline, NativeBenchmarkCampaign) or not isinstance(
        candidate, NativeBenchmarkCampaign
    ):
        raise TypeError("compare_campaigns requires NativeBenchmarkCampaign values")
    baseline_key = baseline.samples[0].key.as_dict()
    candidate_key = candidate.samples[0].key.as_dict()
    executable_fields = {
        "artifact_fingerprint",
        "lowering_fingerprint",
        "placement_fingerprint",
        "backend_id",
        "codec_id",
        "numerical_contract",
    }
    workload_baseline = {
        key: value for key, value in baseline_key.items() if key not in executable_fields
    }
    workload_candidate = {
        key: value for key, value in candidate_key.items() if key not in executable_fields
    }
    if workload_baseline != workload_candidate:
        raise ValueError("campaigns do not have equal work and cannot produce a speed ratio")
    return {
        "baseline_key_fingerprint": baseline.samples[0].key.fingerprint,
        "candidate_key_fingerprint": candidate.samples[0].key.fingerprint,
        "work_kind": baseline.samples[0].key.work_kind.value,
        "baseline_median_units_per_second": baseline.median_units_per_second,
        "candidate_median_units_per_second": candidate.median_units_per_second,
        "candidate_over_baseline_median_ratio": (
            candidate.median_units_per_second / baseline.median_units_per_second
        ),
        "both_comparable": baseline.all_comparable and candidate.all_comparable,
    }


__all__ = [
    "NATIVE_BENCHMARK_CAMPAIGN_SCHEMA",
    "NATIVE_BENCHMARK_KEY_SCHEMA",
    "NATIVE_BENCHMARK_SAMPLE_SCHEMA",
    "NATIVE_ROOFLINE_SCHEMA",
    "NativeBenchmarkCampaign",
    "NativeBenchmarkKey",
    "NativeBenchmarkSample",
    "NativeRoofline",
    "ThermalState",
    "WorkKind",
    "compare_campaigns",
]
