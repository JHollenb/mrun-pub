"""Hardware-scoped promotion records for the primary CUDA Graph campaign route."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .capture_benchmark import cuda_graph_runtime_environment
from .evidence import evidence_payload_sha256
from .identity import bind_loaded_qstore_identity

CUDA_GRAPH_PROMOTION_SCHEMA = "mrun-cuda-graph-primary-promotion-v1"
CUDA_GRAPH_PROMOTION_REGISTRY_SCHEMA = "mrun-cuda-graph-promotion-registry-v1"


def _sha256(value: object, field_name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class CudaGraphPromotionRecord:
    promotion_id: str
    source_checkpoint_sha256: str
    derived_store_sha256: str
    manifest_semantic_sha256: str
    identity_certificate_sha256: str
    activation_dtype: str
    weight_dtype: str
    accumulator_dtype: str
    runtime_source_sha256: str
    runtime_distribution_sha256: str
    evidence_wheel_sha256: str
    evidence_file_sha256: str
    torch_version: str
    torch_cuda_version: str
    gpu_name: str
    gpu_capability: tuple[int, int]
    batch_size: int
    sequence_length: int
    union_candidate_count: int
    capture_total_residency_bytes: int
    capture_setup_ms: float
    break_even_replays: int
    measured_speedup: float
    ratio_ci95: tuple[float, float]
    exact_parity: bool
    wins: int
    trials: int
    residency_safety_factor: float = 1.2
    schema_version: str = CUDA_GRAPH_PROMOTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CUDA_GRAPH_PROMOTION_SCHEMA:
            raise ValueError(f"unsupported CUDA Graph promotion schema: {self.schema_version}")
        if not self.promotion_id:
            raise ValueError("promotion_id must be non-empty")
        for field_name in (
            "source_checkpoint_sha256",
            "derived_store_sha256",
            "manifest_semantic_sha256",
            "identity_certificate_sha256",
            "runtime_source_sha256",
            "runtime_distribution_sha256",
            "evidence_wheel_sha256",
            "evidence_file_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )
        if (
            self.activation_dtype not in {"bf16", "fp16"}
            or self.weight_dtype != "int8"
            or self.accumulator_dtype != "fp32"
        ):
            raise ValueError("promotion precision contract is unsupported")
        if not self.torch_version or not self.torch_cuda_version or not self.gpu_name:
            raise ValueError("promotion runtime identity is incomplete")
        capability = tuple(int(value) for value in self.gpu_capability)
        if len(capability) != 2 or min(capability) < 0:
            raise ValueError("promotion GPU capability is invalid")
        object.__setattr__(self, "gpu_capability", capability)
        for field_name in (
            "batch_size",
            "sequence_length",
            "union_candidate_count",
            "capture_total_residency_bytes",
            "break_even_replays",
            "wins",
            "trials",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if self.batch_size != 1:
            raise ValueError("candidate campaign promotion currently supports batch one")
        setup_ms = float(self.capture_setup_ms)
        speedup = float(self.measured_speedup)
        ci95 = tuple(float(value) for value in self.ratio_ci95)
        safety = float(self.residency_safety_factor)
        if (
            not math.isfinite(setup_ms)
            or setup_ms <= 0
            or not math.isfinite(speedup)
            or speedup <= 1
            or len(ci95) != 2
            or any(not math.isfinite(value) or value <= 1 for value in ci95)
            or ci95[0] > ci95[1]
            or not math.isfinite(safety)
            or safety < 1
        ):
            raise ValueError("promotion performance/residency evidence is invalid")
        object.__setattr__(self, "capture_setup_ms", setup_ms)
        object.__setattr__(self, "measured_speedup", speedup)
        object.__setattr__(self, "ratio_ci95", ci95)
        object.__setattr__(self, "residency_safety_factor", safety)
        if type(self.exact_parity) is not bool or not self.exact_parity:
            raise ValueError("promotion requires exact replay parity")
        if self.wins != self.trials:
            raise ValueError("promotion requires every paired trial to favor graph replay")

    @property
    def fingerprint(self) -> str:
        return evidence_payload_sha256(self.as_dict(include_fingerprint=False))

    @property
    def required_free_bytes(self) -> int:
        return math.ceil(self.capture_total_residency_bytes * self.residency_safety_factor)

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "promotion_id": self.promotion_id,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "derived_store_sha256": self.derived_store_sha256,
            "manifest_semantic_sha256": self.manifest_semantic_sha256,
            "identity_certificate_sha256": self.identity_certificate_sha256,
            "activation_dtype": self.activation_dtype,
            "weight_dtype": self.weight_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "runtime_source_sha256": self.runtime_source_sha256,
            "runtime_distribution_sha256": self.runtime_distribution_sha256,
            "evidence_wheel_sha256": self.evidence_wheel_sha256,
            "evidence_file_sha256": self.evidence_file_sha256,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "gpu_name": self.gpu_name,
            "gpu_capability": list(self.gpu_capability),
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "union_candidate_count": self.union_candidate_count,
            "capture_total_residency_bytes": self.capture_total_residency_bytes,
            "capture_setup_ms": self.capture_setup_ms,
            "break_even_replays": self.break_even_replays,
            "measured_speedup": self.measured_speedup,
            "ratio_ci95": list(self.ratio_ci95),
            "exact_parity": self.exact_parity,
            "wins": self.wins,
            "trials": self.trials,
            "residency_safety_factor": self.residency_safety_factor,
        }
        if include_fingerprint:
            payload["promotion_fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CudaGraphPromotionRecord:
        record = cls(
            promotion_id=str(payload["promotion_id"]),
            source_checkpoint_sha256=str(payload["source_checkpoint_sha256"]),
            derived_store_sha256=str(payload["derived_store_sha256"]),
            manifest_semantic_sha256=str(payload["manifest_semantic_sha256"]),
            identity_certificate_sha256=str(payload["identity_certificate_sha256"]),
            activation_dtype=str(payload["activation_dtype"]),
            weight_dtype=str(payload["weight_dtype"]),
            accumulator_dtype=str(payload["accumulator_dtype"]),
            runtime_source_sha256=str(payload["runtime_source_sha256"]),
            runtime_distribution_sha256=str(payload["runtime_distribution_sha256"]),
            evidence_wheel_sha256=str(payload["evidence_wheel_sha256"]),
            evidence_file_sha256=str(payload["evidence_file_sha256"]),
            torch_version=str(payload["torch_version"]),
            torch_cuda_version=str(payload["torch_cuda_version"]),
            gpu_name=str(payload["gpu_name"]),
            gpu_capability=tuple(int(value) for value in payload["gpu_capability"]),
            batch_size=int(payload["batch_size"]),
            sequence_length=int(payload["sequence_length"]),
            union_candidate_count=int(payload["union_candidate_count"]),
            capture_total_residency_bytes=int(payload["capture_total_residency_bytes"]),
            capture_setup_ms=float(payload["capture_setup_ms"]),
            break_even_replays=int(payload["break_even_replays"]),
            measured_speedup=float(payload["measured_speedup"]),
            ratio_ci95=tuple(float(value) for value in payload["ratio_ci95"]),
            exact_parity=payload["exact_parity"],
            wins=int(payload["wins"]),
            trials=int(payload["trials"]),
            residency_safety_factor=float(payload.get("residency_safety_factor", 1.2)),
            schema_version=str(payload.get("schema_version", CUDA_GRAPH_PROMOTION_SCHEMA)),
        )
        claimed = payload.get("promotion_fingerprint")
        if claimed is not None and str(claimed) != record.fingerprint:
            raise ValueError("CUDA Graph promotion fingerprint mismatch")
        return record


@dataclass(frozen=True)
class CudaGraphPromotionDecision:
    selected: bool
    reason: str
    expected_replays: int
    promotion: CudaGraphPromotionRecord | None = None
    blockers: tuple[str, ...] = ()
    free_cuda_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "reason": self.reason,
            "expected_replays": self.expected_replays,
            "promotion_id": None if self.promotion is None else self.promotion.promotion_id,
            "promotion_fingerprint": (
                None if self.promotion is None else self.promotion.fingerprint
            ),
            "required_free_bytes": (
                None if self.promotion is None else self.promotion.required_free_bytes
            ),
            "free_cuda_bytes": self.free_cuda_bytes,
            "blockers": list(self.blockers),
        }


# Built-ins are intentionally empty. Promotion records are installed beside the runtime
# rather than embedded in the wheel, so a record can bind the exact wheel without creating
# a self-referential hash.
BUILTIN_CUDA_GRAPH_PROMOTIONS: tuple[CudaGraphPromotionRecord, ...] = ()


def cuda_graph_promotion_registry_payload(
    records: Sequence[CudaGraphPromotionRecord],
) -> dict[str, Any]:
    """Build a canonical, fingerprinted external promotion registry payload."""

    normalized = tuple(records)
    if any(not isinstance(record, CudaGraphPromotionRecord) for record in normalized):
        raise TypeError("CUDA Graph promotion registry accepts only promotion records")
    ids = [record.promotion_id for record in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("CUDA Graph promotion registry has duplicate promotion IDs")
    return {
        "schema_version": CUDA_GRAPH_PROMOTION_REGISTRY_SCHEMA,
        "records": [record.as_dict() for record in normalized],
    }


def build_cuda_graph_promotion_record(
    engine: Any,
    campaign: Any,
    benchmark: Any,
    *,
    promotion_id: str,
    evidence_file_sha256: str,
    residency_safety_factor: float = 1.2,
) -> CudaGraphPromotionRecord:
    """Derive one exact promotion record from a passing hardware benchmark."""

    if benchmark.campaign_fingerprint != campaign.fingerprint:
        raise ValueError("promotion benchmark does not bind the supplied campaign")
    if benchmark.union_candidate_count != len(campaign.union_token_ids):
        raise ValueError("promotion benchmark union does not bind the supplied campaign")
    if benchmark.capture_improvement_demonstrated is not True:
        raise ValueError("promotion benchmark did not demonstrate a capture improvement")
    if benchmark.output_parity.exact is not True:
        raise ValueError("promotion benchmark lacks exact eager/replay parity")
    if benchmark.matched_eager_to_cuda_graph.wins != benchmark.trials:
        raise ValueError("promotion benchmark did not win every paired trial")
    break_even = benchmark.break_even_replays
    if break_even is None:
        raise ValueError("promotion benchmark has no finite replay break-even")

    identity = bind_loaded_qstore_identity(engine)
    if not identity.content_identity_verified:
        raise ValueError("promotion requires verified QStore content identity")
    plan = campaign.base_bundle.plan
    if (
        plan.model_revision != identity.model_revision
        or plan.store_fingerprint != identity.store_fingerprint
        or not plan.content_identity_verified
    ):
        raise ValueError("promotion campaign is not bound to the loaded QStore identity")
    environment = dict(benchmark.runtime_environment)
    observed_environment = cuda_graph_runtime_environment(engine)
    for key in (
        "runtime_source_sha256",
        "runtime_distribution_sha256",
        "wheel_sha256",
        "torch_version",
        "torch_cuda_version",
        "device",
        "gpu_name",
        "gpu_capability",
    ):
        benchmark_value = environment.get(key)
        observed_value = observed_environment.get(key)
        if key == "gpu_capability":
            benchmark_value = tuple(int(value) for value in benchmark_value)
            observed_value = tuple(int(value) for value in observed_value)
        if benchmark_value != observed_value:
            raise ValueError(f"promotion runtime environment drifted after benchmark: {key}")
    wheel_sha256 = environment.get("wheel_sha256")
    if wheel_sha256 is None:
        raise ValueError("promotion evidence requires MRUN_WHEEL_SHA256")
    capture_evidence = benchmark.capture_executor_evidence

    return CudaGraphPromotionRecord(
        promotion_id=promotion_id,
        source_checkpoint_sha256=identity.model_revision,
        derived_store_sha256=identity.store_fingerprint,
        manifest_semantic_sha256=identity.manifest_semantic_sha256,
        identity_certificate_sha256=identity.identity_certificate_sha256,
        activation_dtype=plan.precision.activation_dtype,
        weight_dtype=plan.precision.weight_dtype,
        accumulator_dtype=plan.precision.accumulator_dtype,
        runtime_source_sha256=str(environment["runtime_source_sha256"]),
        runtime_distribution_sha256=str(environment["runtime_distribution_sha256"]),
        evidence_wheel_sha256=str(wheel_sha256),
        evidence_file_sha256=evidence_file_sha256,
        torch_version=str(environment["torch_version"]),
        torch_cuda_version=str(environment["torch_cuda_version"]),
        gpu_name=str(environment["gpu_name"]),
        gpu_capability=tuple(int(value) for value in environment["gpu_capability"]),
        batch_size=plan.shape.actual_batch,
        sequence_length=plan.shape.sequence_length,
        union_candidate_count=len(campaign.union_token_ids),
        capture_total_residency_bytes=int(capture_evidence["capture_total_residency_delta_bytes"]),
        capture_setup_ms=benchmark.capture_setup_ms,
        break_even_replays=break_even,
        measured_speedup=benchmark.speedup,
        ratio_ci95=tuple(benchmark.matched_eager_to_cuda_graph.ci95),
        exact_parity=benchmark.output_parity.exact,
        wins=benchmark.matched_eager_to_cuda_graph.wins,
        trials=benchmark.trials,
        residency_safety_factor=residency_safety_factor,
    )


def default_cuda_graph_promotion_registry() -> Path:
    configured = os.environ.get("MRUN_CUDA_GRAPH_PROMOTIONS")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".config" / "mrun" / "cuda_graph_promotions.json"


def load_cuda_graph_promotions(
    path: str | Path | None = None,
) -> tuple[CudaGraphPromotionRecord, ...]:
    """Load an external evidence registry without creating a wheel-hash cycle."""

    registry_path = (
        default_cuda_graph_promotion_registry() if path is None else Path(path).expanduser()
    )
    if not registry_path.exists():
        return BUILTIN_CUDA_GRAPH_PROMOTIONS
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CUDA Graph promotion registry must be an object")
    if payload.get("schema_version") != CUDA_GRAPH_PROMOTION_REGISTRY_SCHEMA:
        raise ValueError("CUDA Graph promotion registry schema mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("CUDA Graph promotion registry records must be a list")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("CUDA Graph promotion registry records must be objects")
    loaded = tuple(CudaGraphPromotionRecord.from_dict(record) for record in records)
    ids = [record.promotion_id for record in loaded]
    if len(ids) != len(set(ids)):
        raise ValueError("CUDA Graph promotion registry has duplicate promotion IDs")
    return (*BUILTIN_CUDA_GRAPH_PROMOTIONS, *loaded)


def _activation_dtype(engine: Any) -> str:
    value = str(getattr(getattr(engine, "store", None), "compute_dtype", "")).removeprefix("torch.")
    return {"bfloat16": "bf16", "float16": "fp16"}.get(value, value)


def select_cuda_graph_promotion(
    engine: Any,
    token_ids: np.ndarray | Sequence[int],
    union_candidate_ids: Sequence[int],
    *,
    expected_replays: int,
    minimum_improvement_percent: float = 0.0,
    promotions: Sequence[CudaGraphPromotionRecord] | None = None,
    free_cuda_bytes: int | None = None,
) -> CudaGraphPromotionDecision:
    """Select only an exact hardware/software/shape promotion with enough amortization."""

    expected = int(expected_replays)
    if expected <= 0:
        raise ValueError("expected_replays must be positive")
    threshold = float(minimum_improvement_percent)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("minimum improvement threshold must be finite and non-negative")
    values = np.asarray(token_ids, dtype=np.int64)
    union = tuple(int(value) for value in union_candidate_ids)
    if values.ndim != 1 or not values.size or not union or len(union) != len(set(union)):
        raise ValueError("promotion selection requires one static input and a unique union")
    identity = bind_loaded_qstore_identity(engine)
    if not identity.content_identity_verified:
        return CudaGraphPromotionDecision(
            False,
            "QStore content identity is not verified",
            expected,
            blockers=("verified-qstore-v3-required",),
        )
    environment = cuda_graph_runtime_environment(engine)
    store = engine.store
    manifest = store.man
    metadata = {
        "source_checkpoint_sha256": identity.model_revision,
        "derived_store_sha256": identity.store_fingerprint,
        "manifest_semantic_sha256": identity.manifest_semantic_sha256,
        "identity_certificate_sha256": identity.identity_certificate_sha256,
        "activation_dtype": _activation_dtype(engine),
        "weight_dtype": str(manifest.get("dtype", "")),
        "accumulator_dtype": "fp32",
        "runtime_source_sha256": str(environment["runtime_source_sha256"]),
        "runtime_distribution_sha256": str(environment["runtime_distribution_sha256"]),
        "torch_version": str(environment["torch_version"]),
        "torch_cuda_version": str(environment["torch_cuda_version"]),
        "gpu_name": str(environment["gpu_name"]),
        "gpu_capability": tuple(int(value) for value in environment["gpu_capability"]),
        "batch_size": 1,
        "sequence_length": int(values.size),
        "union_candidate_count": len(union),
    }
    records = tuple(load_cuda_graph_promotions() if promotions is None else promotions)
    matching = [
        record
        for record in records
        if all(getattr(record, key) == value for key, value in metadata.items())
    ]
    runtime_wheel_sha256 = environment.get("wheel_sha256")
    if runtime_wheel_sha256 is not None:
        runtime_wheel_sha256 = _sha256(
            runtime_wheel_sha256,
            "runtime_environment.wheel_sha256",
        )
        matching = [
            record for record in matching if record.evidence_wheel_sha256 == runtime_wheel_sha256
        ]
    if not matching:
        return CudaGraphPromotionDecision(
            False,
            "no exact CUDA Graph promotion matches this store/runtime/shape",
            expected,
            blockers=("no-exact-promotion-record",),
        )
    record = matching[0]
    blockers = []
    if record.ratio_ci95[0] <= 1.0 + threshold / 100.0:
        blockers.append("promotion interval does not clear the requested improvement threshold")
    if expected < record.break_even_replays:
        blockers.append(
            f"expected replays {expected} are below break-even {record.break_even_replays}"
        )
    if free_cuda_bytes is None:
        import torch

        device = str(environment["device"])
        free_cuda_bytes = int(torch.cuda.mem_get_info(device)[0])
    if free_cuda_bytes < record.required_free_bytes:
        blockers.append("free CUDA memory is below the promotion residency safety requirement")
    if blockers:
        return CudaGraphPromotionDecision(
            False,
            "exact promotion exists but its runtime admission gate failed",
            expected,
            promotion=record,
            blockers=tuple(blockers),
            free_cuda_bytes=free_cuda_bytes,
        )
    return CudaGraphPromotionDecision(
        True,
        "exact promotion, amortization, and VRAM gates passed",
        expected,
        promotion=record,
        free_cuda_bytes=free_cuda_bytes,
    )
