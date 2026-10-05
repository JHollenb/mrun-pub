"""Launch-isolated CUDA Graph evidence for candidate campaigns.

The control and treatment are two methods on one prepared capture executor:
``execute_eager_control()`` runs the retained operation eagerly, while ``execute()``
replays the captured graph.  They therefore share static inputs, zero-KV state, output
buffers, pinned QStore resources, and numerical projection.  Compilation, capture,
residency setup, and runtime binding all happen before timing.

This is intentionally narrower than the ordinary campaign benchmark.  It answers
"what did graph replay itself buy?" rather than comparing streaming eager execution
with the complete resident/captured runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .benchmark import OutputParity, _compare_outputs, _reset_max_dequant_block, _sync_engine
from .campaign import (
    CandidateCampaign,
    CandidateReadout,
    compile_candidate_campaign,
    validate_candidate_campaign_engine_binding,
)
from .campaign_benchmark import (
    CampaignLegStats,
    PairedCampaignContrast,
    _leg_stats,
    _paired_contrast,
)
from .ir import CaptureContract

CUDA_GRAPH_CAMPAIGN_BENCHMARK_SCHEMA = "mrun-cuda-graph-campaign-benchmark-v1"
_BENCHMARK_BASIS = "paired-shared-executor-matched-eager-control-vs-cuda-graph-replay"
_MATCHED_CONTROL_BASIS = "same-operation-static-inputs-kv-and-pinned-resources-without-graph-replay"
_CUDA_GRAPH_RUNTIME_FILES = (
    "compiler/campaign.py",
    "compiler/capture_benchmark.py",
    "compiler/identity.py",
    "compiler/lowering.py",
    "engine/dense_qstore_cuda.py",
    "engine/kernels/dense_qstore_cuda.py",
    "engine/kernels/qstore.py",
)


def _require_sha256(value: Any, field_name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _same_float(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _freeze_json(value: Any, field_name: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{field_name} keys must be non-empty strings")
            frozen[key] = _freeze_json(item, field_name)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, field_name) for item in value)
    raise TypeError(f"{field_name} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _json_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _thaw_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cuda_graph_runtime_source_sha256() -> str:
    """Hash the exact packaged sources that define capture, identity, and replay."""

    package_root = Path(__file__).resolve().parents[1]
    records = []
    for logical_name in _CUDA_GRAPH_RUNTIME_FILES:
        path = package_root / logical_name
        records.append(
            {
                "name": logical_name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cuda_graph_runtime_distribution_sha256() -> str:
    """Hash every installed Python source in the mrun runtime payload."""

    package_root = Path(__file__).resolve().parents[1]
    records = []
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        records.append(
            {
                "name": path.relative_to(package_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not records:
        raise RuntimeError("cannot identify the installed mrun runtime payload")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cuda_graph_runtime_environment(engine: Any) -> dict[str, Any]:
    declared = getattr(engine, "cuda_graph_runtime_environment", None)
    if isinstance(declared, Mapping):
        return dict(declared)

    import torch

    device = torch.device(str(getattr(engine, "device", "cuda")))
    if device.type != "cuda":
        raise RuntimeError("CUDA Graph evidence requires a CUDA runtime device")
    if device.index is None:
        raise RuntimeError("CUDA Graph evidence requires a concrete CUDA device ordinal")
    properties = torch.cuda.get_device_properties(device)
    try:
        import triton

        triton_version = str(triton.__version__)
    except (ImportError, AttributeError):
        triton_version = None
    driver_version = None
    get_driver_version = getattr(torch._C, "_cuda_getDriverVersion", None)
    if callable(get_driver_version):
        driver_version = int(get_driver_version())
    wheel_sha256 = os.environ.get("MRUN_WHEEL_SHA256")
    if wheel_sha256 is not None:
        wheel_sha256 = _require_sha256(wheel_sha256, "MRUN_WHEEL_SHA256")
    return {
        "runtime_source_sha256": cuda_graph_runtime_source_sha256(),
        "runtime_distribution_sha256": cuda_graph_runtime_distribution_sha256(),
        "wheel_sha256": wheel_sha256,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_driver_version": driver_version,
        "triton_version": triton_version,
        "device": str(device),
        "gpu_name": str(properties.name),
        "gpu_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "gpu_total_memory_bytes": int(properties.total_memory),
        "store_reverified_at_ns": int(
            getattr(getattr(engine, "store", None), "content_verified_at_ns", 0)
        ),
    }


def _parity_from_dict(payload: Mapping[str, Any]) -> OutputParity:
    return OutputParity(
        allclose=_require_bool(payload["allclose"], "output_parity.allclose"),
        exact=_require_bool(payload["exact"], "output_parity.exact"),
        compared_values=_require_positive_int(
            payload["compared_values"],
            "output_parity.compared_values",
        ),
        max_abs_error=float(payload["max_abs_error"]),
        max_rel_error=float(payload["max_rel_error"]),
    )


def _capture_semantic_payload(campaign: CandidateCampaign) -> dict[str, Any]:
    graph = campaign.base_bundle.graph
    if graph is None or graph.rewrite_certificate is None:
        raise ValueError("CUDA Graph campaign benchmark requires a certified graph")
    semantic_plan = replace(campaign.base_bundle.plan, capture=CaptureContract())
    return {
        "schema": "mrun-cuda-graph-campaign-semantics-v1",
        "input_binding": campaign.input_binding.as_dict(),
        "readouts": [readout.as_dict() for readout in campaign.readouts],
        "union_token_ids": list(campaign.union_token_ids),
        "sharing": campaign.sharing.as_dict(),
        "plan_without_capture": semantic_plan.as_dict(),
        "source_graph_fingerprint": graph.source_graph_fingerprint,
        "rewritten_graph_fingerprint": graph.graph.fingerprint,
        "rewrite_ids": list(graph.rewrite_certificate.rewrite_ids),
    }


def validate_cuda_graph_campaign(campaign: CandidateCampaign) -> str:
    """Validate capture legality and return the workload's capture-neutral identity."""

    plan = campaign.base_bundle.plan
    lowered = campaign.base_bundle.lowered
    if lowered.backend != "cuda-qstore":
        raise ValueError("CUDA Graph campaign must use the CUDA QStore lowerer")
    if not plan.capture.requested:
        raise ValueError("CUDA Graph campaign must request graph capture")
    if not plan.capture.eligible:
        raise ValueError("CUDA Graph campaign lacks static shape/address/graph-safe proofs")
    if not lowered.capture_ready:
        raise ValueError("CUDA Graph campaign lowerer is not capture-ready")
    if lowered.capture_executed:
        raise ValueError("serialized lowering cannot claim runtime graph execution")
    payload = _capture_semantic_payload(campaign)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_executor_evidence(
    evidence: Mapping[str, Any],
    *,
    minimum_execution_count: int,
) -> None:
    if not isinstance(evidence, Mapping) or not evidence:
        raise ValueError("capture executor must expose non-empty evidence")
    for key in (
        "capture_ready",
        "capture_executed",
        "graph_replay",
        "capture_matched_eager_control_available",
        "capture_static_shapes",
        "capture_stable_addresses",
        "capture_graph_safe",
        "capture_residency_non_evictable",
        "stable_addresses_verified",
        "capture_resource_addresses_verified",
    ):
        if evidence.get(key) is not True:
            raise ValueError(f"capture executor evidence must expose {key}=true")
    for key in ("capture_full_logits", "capture_stateful_kv", "capture_decode"):
        if evidence.get(key) is not False:
            raise ValueError(f"capture executor evidence must expose {key}=false")
    if evidence.get("capture_mode") != "selected-last-stateless-score":
        raise ValueError("capture executor evidence has the wrong capture mode")
    if evidence.get("capture_matched_eager_control_basis") != _MATCHED_CONTROL_BASIS:
        raise ValueError("capture executor evidence has no matched eager-control proof")
    if evidence.get("capture_executor_closed") is not False:
        raise ValueError("capture executor was closed before evidence snapshot")
    if _require_positive_int(evidence.get("capture_count"), "capture_count") != 1:
        raise ValueError("capture executor must represent exactly one graph capture")
    replay_count = _require_nonnegative_int(
        evidence.get("capture_replay_count"),
        "capture_replay_count",
    )
    control_count = _require_nonnegative_int(
        evidence.get("capture_matched_eager_control_count"),
        "capture_matched_eager_control_count",
    )
    if replay_count < minimum_execution_count or control_count < minimum_execution_count:
        raise ValueError("capture executor evidence does not cover every benchmark execution")
    for key in ("capture_backend", "graph_backend"):
        if not isinstance(evidence.get(key), str) or not evidence[key]:
            raise ValueError(f"capture executor evidence must identify {key}")
    setup_ms = float(evidence.get("capture_setup_ms", -1.0))
    if not math.isfinite(setup_ms) or setup_ms <= 0:
        raise ValueError("capture executor evidence must record positive capture_setup_ms")
    for key in (
        "capture_resident_bytes",
        "capture_total_allocated_delta_bytes",
        "capture_total_reserved_delta_bytes",
        "capture_peak_allocated_delta_bytes",
        "capture_total_residency_delta_bytes",
    ):
        _require_nonnegative_int(evidence.get(key), key)
    if evidence["capture_total_residency_delta_bytes"] < evidence["capture_resident_bytes"]:
        raise ValueError("capture total residency cannot be smaller than pinned QStore tensors")
    budget = evidence.get("capture_residency_budget_bytes")
    if budget is not None:
        _require_positive_int(budget, "capture_residency_budget_bytes")
        if evidence["capture_total_residency_delta_bytes"] > budget:
            raise ValueError("capture total residency exceeds its declared budget")


def _leg_evidence(
    *,
    route: str,
    graph_replay: bool,
    capture_executed: bool,
    executor_evidence_sha256: str,
    compilation_fingerprint: str,
    input_token_sha256: str,
    reported_fabric: str,
) -> dict[str, Any]:
    return {
        "body_execution_count": 1,
        "execution_route": route,
        "shared_capture_executor": True,
        "capture_requested": True,
        "capture_ready": True,
        "capture_executed": capture_executed,
        "graph_replay": graph_replay,
        "capture_executor_evidence_sha256": executor_evidence_sha256,
        "graph_compilation_fingerprint": compilation_fingerprint,
        "input_token_sha256": input_token_sha256,
        "reported_fabric": reported_fabric,
    }


def _validate_leg_evidence(
    eager_evidence: Mapping[str, Any],
    graph_evidence: Mapping[str, Any],
    *,
    executor_evidence_sha256: str,
    compilation_fingerprint: str,
    input_token_sha256: str,
    reported_fabric: str,
) -> None:
    for field_name, evidence in (
        ("matched_eager_control_runtime_evidence", eager_evidence),
        ("cuda_graph_runtime_evidence", graph_evidence),
    ):
        if not isinstance(evidence, Mapping):
            raise TypeError(f"{field_name} must be an object")
        if evidence.get("body_execution_count") != 1:
            raise ValueError(f"{field_name} must record one body execution")
        if evidence.get("shared_capture_executor") is not True:
            raise ValueError(f"{field_name} must bind the shared capture executor")
        if evidence.get("capture_requested") is not True:
            raise ValueError(f"{field_name} must record capture preparation")
        if evidence.get("capture_ready") is not True:
            raise ValueError(f"{field_name} must record capture readiness")
        if evidence.get("capture_executor_evidence_sha256") != executor_evidence_sha256:
            raise ValueError(f"{field_name} has the wrong capture executor identity")
        if evidence.get("graph_compilation_fingerprint") != compilation_fingerprint:
            raise ValueError(f"{field_name} has the wrong compilation identity")
        if evidence.get("input_token_sha256") != input_token_sha256:
            raise ValueError(f"{field_name} has the wrong input identity")
        if evidence.get("reported_fabric") != reported_fabric:
            raise ValueError(f"{field_name} has the wrong reported fabric")

    if eager_evidence.get("execution_route") != "matched-eager-control":
        raise ValueError("matched eager-control evidence has the wrong route")
    if eager_evidence.get("capture_executed") is not False:
        raise ValueError("matched eager control cannot claim capture_executed=true")
    if eager_evidence.get("graph_replay") is not False:
        raise ValueError("matched eager control cannot claim graph_replay=true")
    if graph_evidence.get("execution_route") != "cuda-graph-replay":
        raise ValueError("CUDA Graph evidence has the wrong route")
    if graph_evidence.get("capture_executed") is not True:
        raise ValueError("CUDA Graph evidence must expose capture_executed=true")
    if graph_evidence.get("graph_replay") is not True:
        raise ValueError("CUDA Graph evidence must expose graph_replay=true")


@dataclass(frozen=True)
class CudaGraphCampaignBenchmarkResult:
    """Paired raw measurements and runtime proof for one shared capture executor."""

    semantic_fingerprint: str
    campaign_fingerprint: str
    compilation_fingerprint: str
    input_token_sha256: str
    query_count: int
    union_candidate_count: int
    matched_eager_control: CampaignLegStats
    cuda_graph: CampaignLegStats
    matched_eager_to_cuda_graph: PairedCampaignContrast
    output_parity: OutputParity
    capture_executor_evidence: Mapping[str, Any]
    matched_eager_control_runtime_evidence: Mapping[str, Any]
    cuda_graph_runtime_evidence: Mapping[str, Any]
    runtime_environment: Mapping[str, Any]
    minimum_improvement_percent: float
    capture_improvement_demonstrated: bool
    trials: int
    warmup: int
    reported_fabric: str
    benchmark_basis: str = _BENCHMARK_BASIS
    schema_version: str = CUDA_GRAPH_CAMPAIGN_BENCHMARK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CUDA_GRAPH_CAMPAIGN_BENCHMARK_SCHEMA:
            raise ValueError(f"unsupported CUDA Graph benchmark schema: {self.schema_version}")
        for field_name in (
            "semantic_fingerprint",
            "campaign_fingerprint",
            "compilation_fingerprint",
            "input_token_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )

        query_count = int(self.query_count)
        union_count = int(self.union_candidate_count)
        trials = int(self.trials)
        warmup = int(self.warmup)
        if query_count < 2:
            raise ValueError("CUDA Graph campaign benchmark requires at least two queries")
        if union_count < 2:
            raise ValueError("CUDA Graph campaign union requires at least two candidates")
        if trials <= 0 or warmup < 0:
            raise ValueError("trials must be positive and warmup must be non-negative")
        if (
            len(self.matched_eager_control.samples_ms) != trials
            or len(self.cuda_graph.samples_ms) != trials
        ):
            raise ValueError("CUDA Graph benchmark leg sample counts do not match trials")
        object.__setattr__(self, "query_count", query_count)
        object.__setattr__(self, "union_candidate_count", union_count)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "warmup", warmup)

        contrast = self.matched_eager_to_cuda_graph
        if (
            contrast.numerator_leg != "matched_eager_control"
            or contrast.denominator_leg != "cuda_graph"
        ):
            raise ValueError("CUDA Graph benchmark contrast leg identity is inconsistent")
        expected_ratios = tuple(
            eager / graph
            for eager, graph in zip(
                self.matched_eager_control.samples_ms,
                self.cuda_graph.samples_ms,
                strict=True,
            )
        )
        if len(contrast.ratios) != trials or any(
            not _same_float(observed, expected)
            for observed, expected in zip(
                contrast.ratios,
                expected_ratios,
                strict=True,
            )
        ):
            raise ValueError("CUDA Graph paired ratios do not match raw samples")
        expected_reduction = (
            1.0 - self.cuda_graph.median_ms / self.matched_eager_control.median_ms
        ) * 100.0
        if not _same_float(
            contrast.latency_reduction_percent,
            expected_reduction,
        ):
            raise ValueError("CUDA Graph latency reduction does not match raw leg medians")

        parity = self.output_parity
        if (
            type(parity.allclose) is not bool
            or type(parity.exact) is not bool
            or type(parity.compared_values) is not int
            or parity.compared_values <= 0
            or not math.isfinite(parity.max_abs_error)
            or parity.max_abs_error < 0
            or not math.isfinite(parity.max_rel_error)
            or parity.max_rel_error < 0
            or (parity.exact and not parity.allclose)
        ):
            raise ValueError("CUDA Graph output parity evidence is inconsistent")

        executor_evidence = _freeze_json(
            self.capture_executor_evidence,
            "capture_executor_evidence",
        )
        eager_evidence = _freeze_json(
            self.matched_eager_control_runtime_evidence,
            "matched_eager_control_runtime_evidence",
        )
        graph_evidence = _freeze_json(
            self.cuda_graph_runtime_evidence,
            "cuda_graph_runtime_evidence",
        )
        runtime_environment = _freeze_json(
            self.runtime_environment,
            "runtime_environment",
        )
        _validate_executor_evidence(
            executor_evidence,
            minimum_execution_count=1 + warmup + trials,
        )
        evidence_sha256 = _json_digest(executor_evidence)
        fabric = str(self.reported_fabric)
        if not fabric:
            raise ValueError("reported_fabric must be non-empty")
        _validate_leg_evidence(
            eager_evidence,
            graph_evidence,
            executor_evidence_sha256=evidence_sha256,
            compilation_fingerprint=self.compilation_fingerprint,
            input_token_sha256=self.input_token_sha256,
            reported_fabric=fabric,
        )
        object.__setattr__(self, "capture_executor_evidence", executor_evidence)
        object.__setattr__(
            self,
            "matched_eager_control_runtime_evidence",
            eager_evidence,
        )
        object.__setattr__(self, "cuda_graph_runtime_evidence", graph_evidence)
        for key in (
            "runtime_source_sha256",
            "runtime_distribution_sha256",
            "torch_version",
            "torch_cuda_version",
            "device",
            "gpu_name",
            "gpu_capability",
            "gpu_total_memory_bytes",
            "store_reverified_at_ns",
        ):
            if runtime_environment.get(key) in (None, "", ()):
                raise ValueError(f"CUDA Graph runtime environment is missing {key}")
        _require_sha256(
            runtime_environment["runtime_source_sha256"],
            "runtime_environment.runtime_source_sha256",
        )
        _require_sha256(
            runtime_environment["runtime_distribution_sha256"],
            "runtime_environment.runtime_distribution_sha256",
        )
        capability = runtime_environment["gpu_capability"]
        if (
            not isinstance(capability, tuple)
            or len(capability) != 2
            or any(type(value) is not int or value < 0 for value in capability)
        ):
            raise ValueError("CUDA Graph runtime environment has invalid GPU capability")
        if (
            type(runtime_environment["gpu_total_memory_bytes"]) is not int
            or runtime_environment["gpu_total_memory_bytes"] <= 0
        ):
            raise ValueError("CUDA Graph runtime environment has invalid GPU memory")
        if (
            type(runtime_environment["store_reverified_at_ns"]) is not int
            or runtime_environment["store_reverified_at_ns"] <= 0
        ):
            raise ValueError("CUDA Graph runtime environment has no fresh store verification")
        object.__setattr__(self, "runtime_environment", runtime_environment)
        object.__setattr__(self, "reported_fabric", fabric)

        threshold = float(self.minimum_improvement_percent)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("minimum improvement threshold must be finite and non-negative")
        object.__setattr__(self, "minimum_improvement_percent", threshold)
        expected_verdict = parity.exact and contrast.ci95[0] > 1.0 + threshold / 100.0
        if (
            type(self.capture_improvement_demonstrated) is not bool
            or self.capture_improvement_demonstrated != expected_verdict
        ):
            raise ValueError("capture improvement verdict is inconsistent with paired evidence")
        if self.benchmark_basis != _BENCHMARK_BASIS:
            raise ValueError("unsupported CUDA Graph benchmark basis")

    @property
    def speedup(self) -> float:
        """Paired median control/replay duration ratio; above one favors replay."""

        return self.matched_eager_to_cuda_graph.median_ratio

    @property
    def latency_reduction_percent(self) -> float:
        return self.matched_eager_to_cuda_graph.latency_reduction_percent

    @property
    def maximum_temporal_drift_factor(self) -> float:
        factors = []
        for leg in (self.matched_eager_control, self.cuda_graph):
            ratio = leg.temporal_drift_ratio
            factors.append(max(ratio, 1.0 / ratio) if ratio else math.inf)
        return max(factors)

    @property
    def capture_setup_ms(self) -> float:
        return float(self.capture_executor_evidence["capture_setup_ms"])

    @property
    def per_replay_savings_ms(self) -> float:
        return self.matched_eager_control.median_ms - self.cuda_graph.median_ms

    @property
    def break_even_replays(self) -> int | None:
        savings = self.per_replay_savings_ms
        if savings <= 0:
            return None
        return max(1, math.ceil(self.capture_setup_ms / savings))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic_fingerprint": self.semantic_fingerprint,
            "campaign_fingerprint": self.campaign_fingerprint,
            "compilation_fingerprint": self.compilation_fingerprint,
            "input_token_sha256": self.input_token_sha256,
            "query_count": self.query_count,
            "union_candidate_count": self.union_candidate_count,
            "matched_eager_control": self.matched_eager_control.as_dict(),
            "cuda_graph": self.cuda_graph.as_dict(),
            "matched_eager_to_cuda_graph": (self.matched_eager_to_cuda_graph.as_dict()),
            "output_parity": self.output_parity.as_dict(),
            "capture_executor_evidence": _thaw_json(self.capture_executor_evidence),
            "matched_eager_control_runtime_evidence": _thaw_json(
                self.matched_eager_control_runtime_evidence
            ),
            "cuda_graph_runtime_evidence": _thaw_json(self.cuda_graph_runtime_evidence),
            "runtime_environment": _thaw_json(self.runtime_environment),
            "minimum_improvement_percent": self.minimum_improvement_percent,
            "capture_improvement_demonstrated": (self.capture_improvement_demonstrated),
            "speedup": self.speedup,
            "latency_reduction_percent": self.latency_reduction_percent,
            "maximum_temporal_drift_factor": self.maximum_temporal_drift_factor,
            "capture_setup_ms": self.capture_setup_ms,
            "per_replay_savings_ms": self.per_replay_savings_ms,
            "break_even_replays": self.break_even_replays,
            "trials": self.trials,
            "warmup": self.warmup,
            "reported_fabric": self.reported_fabric,
            "benchmark_basis": self.benchmark_basis,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> CudaGraphCampaignBenchmarkResult:
        executor_evidence = payload.get("capture_executor_evidence")
        eager_evidence = payload.get("matched_eager_control_runtime_evidence")
        graph_evidence = payload.get("cuda_graph_runtime_evidence")
        runtime_environment = payload.get("runtime_environment")
        if not all(
            isinstance(value, Mapping)
            for value in (
                executor_evidence,
                eager_evidence,
                graph_evidence,
                runtime_environment,
            )
        ):
            raise TypeError("serialized CUDA Graph runtime evidence must be objects")
        result = cls(
            semantic_fingerprint=str(payload["semantic_fingerprint"]),
            campaign_fingerprint=str(payload["campaign_fingerprint"]),
            compilation_fingerprint=str(payload["compilation_fingerprint"]),
            input_token_sha256=str(payload["input_token_sha256"]),
            query_count=int(payload["query_count"]),
            union_candidate_count=int(payload["union_candidate_count"]),
            matched_eager_control=CampaignLegStats.from_dict(payload["matched_eager_control"]),
            cuda_graph=CampaignLegStats.from_dict(payload["cuda_graph"]),
            matched_eager_to_cuda_graph=PairedCampaignContrast.from_dict(
                payload["matched_eager_to_cuda_graph"]
            ),
            output_parity=_parity_from_dict(payload["output_parity"]),
            capture_executor_evidence=executor_evidence,
            matched_eager_control_runtime_evidence=eager_evidence,
            cuda_graph_runtime_evidence=graph_evidence,
            runtime_environment=runtime_environment,
            minimum_improvement_percent=float(payload["minimum_improvement_percent"]),
            capture_improvement_demonstrated=_require_bool(
                payload["capture_improvement_demonstrated"],
                "capture_improvement_demonstrated",
            ),
            trials=int(payload["trials"]),
            warmup=int(payload["warmup"]),
            reported_fabric=str(payload["reported_fabric"]),
            benchmark_basis=str(payload.get("benchmark_basis", _BENCHMARK_BASIS)),
            schema_version=str(
                payload.get(
                    "schema_version",
                    CUDA_GRAPH_CAMPAIGN_BENCHMARK_SCHEMA,
                )
            ),
        )
        for field_name, expected in (
            ("speedup", result.speedup),
            ("latency_reduction_percent", result.latency_reduction_percent),
            ("maximum_temporal_drift_factor", result.maximum_temporal_drift_factor),
            ("capture_setup_ms", result.capture_setup_ms),
            ("per_replay_savings_ms", result.per_replay_savings_ms),
        ):
            claimed = payload.get(field_name)
            if claimed is not None and not _same_float(float(claimed), expected):
                raise ValueError(f"serialized CUDA Graph {field_name} is inconsistent")
        claimed_break_even = payload.get("break_even_replays")
        if claimed_break_even is not None and int(claimed_break_even) != result.break_even_replays:
            raise ValueError("serialized CUDA Graph break_even_replays is inconsistent")
        return result

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
    ) -> CudaGraphCampaignBenchmarkResult:
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise TypeError("serialized CUDA Graph campaign benchmark must be an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class _MatchedCaptureRunners:
    eager_control: Callable[[], Any]
    cuda_graph: Callable[[], Any]


def _run_matched_capture_benchmark(
    engine: Any,
    runners: _MatchedCaptureRunners,
    *,
    executor_evidence: Callable[[], Mapping[str, Any]],
    semantic_fingerprint: str,
    campaign_fingerprint: str,
    compilation_fingerprint: str,
    input_token_sha256: str,
    query_count: int,
    union_candidate_count: int,
    warmup: int,
    trials: int,
    rtol: float,
    atol: float,
    minimum_improvement_percent: float,
    reported_fabric: str,
    runtime_environment: Mapping[str, Any],
) -> CudaGraphCampaignBenchmarkResult:
    if warmup < 0 or trials <= 0:
        raise ValueError("warmup must be non-negative and trials must be positive")
    if rtol < 0 or atol < 0:
        raise ValueError("parity tolerances must be non-negative")
    if minimum_improvement_percent < 0:
        raise ValueError("minimum improvement threshold must be non-negative")

    # Correctness precedes warmup/timing and also proves both executor methods callable.
    eager_reference = runners.eager_control()
    graph_reference = runners.cuda_graph()
    parity = _compare_outputs(
        eager_reference,
        graph_reference,
        rtol=rtol,
        atol=atol,
    )

    paths = ("matched_eager_control", "cuda_graph")

    def execute(path: str) -> Any:
        return runners.eager_control() if path == "matched_eager_control" else runners.cuda_graph()

    for warmup_index in range(warmup):
        rotation = warmup_index % len(paths)
        for path in paths[rotation:] + paths[:rotation]:
            execute(path)

    samples: dict[str, list[float]] = {path: [] for path in paths}
    max_blocks: dict[str, list[float]] = {path: [] for path in paths}
    for trial in range(trials):
        rotation = trial % len(paths)
        for path in paths[rotation:] + paths[:rotation]:
            _sync_engine(engine)
            telemetry_reset = _reset_max_dequant_block(engine)
            started = time.perf_counter()
            execute(path)
            _sync_engine(engine)
            samples[path].append((time.perf_counter() - started) * 1000.0)
            if telemetry_reset:
                store = getattr(engine, "store", None)
                observed = getattr(store, "max_block_bytes", None)
                if observed is not None:
                    max_blocks[path].append(float(observed) / 1e6)

    # Evidence is dynamic: snapshot only after every measured method invocation.
    raw_executor_evidence = executor_evidence()
    if not isinstance(raw_executor_evidence, Mapping):
        raise RuntimeError("CUDA Graph executor lost its evidence mapping")
    frozen_executor_evidence = _freeze_json(
        dict(raw_executor_evidence),
        "capture_executor_evidence",
    )
    _validate_executor_evidence(
        frozen_executor_evidence,
        minimum_execution_count=1 + warmup + trials,
    )
    executor_evidence_sha256 = _json_digest(frozen_executor_evidence)
    eager_stats = _leg_stats(
        samples["matched_eager_control"],
        max_blocks["matched_eager_control"],
    )
    graph_stats = _leg_stats(samples["cuda_graph"], max_blocks["cuda_graph"])
    contrast = _paired_contrast(
        "matched_eager_control",
        "cuda_graph",
        eager_stats,
        graph_stats,
    )
    eager_evidence = _leg_evidence(
        route="matched-eager-control",
        graph_replay=False,
        capture_executed=False,
        executor_evidence_sha256=executor_evidence_sha256,
        compilation_fingerprint=compilation_fingerprint,
        input_token_sha256=input_token_sha256,
        reported_fabric=reported_fabric,
    )
    graph_evidence = _leg_evidence(
        route="cuda-graph-replay",
        graph_replay=True,
        capture_executed=True,
        executor_evidence_sha256=executor_evidence_sha256,
        compilation_fingerprint=compilation_fingerprint,
        input_token_sha256=input_token_sha256,
        reported_fabric=reported_fabric,
    )
    threshold = float(minimum_improvement_percent)
    return CudaGraphCampaignBenchmarkResult(
        semantic_fingerprint=semantic_fingerprint,
        campaign_fingerprint=campaign_fingerprint,
        compilation_fingerprint=compilation_fingerprint,
        input_token_sha256=input_token_sha256,
        query_count=query_count,
        union_candidate_count=union_candidate_count,
        matched_eager_control=eager_stats,
        cuda_graph=graph_stats,
        matched_eager_to_cuda_graph=contrast,
        output_parity=parity,
        capture_executor_evidence=frozen_executor_evidence,
        matched_eager_control_runtime_evidence=eager_evidence,
        cuda_graph_runtime_evidence=graph_evidence,
        runtime_environment=runtime_environment,
        minimum_improvement_percent=threshold,
        capture_improvement_demonstrated=(
            parity.exact and contrast.ci95[0] > 1.0 + threshold / 100.0
        ),
        trials=trials,
        warmup=warmup,
        reported_fabric=reported_fabric,
    )


def benchmark_cuda_graph_campaign(
    engine: Any,
    campaign: CandidateCampaign,
    token_ids: np.ndarray | Sequence[int],
    *,
    capture_warmup: int = 3,
    warmup: int = 3,
    trials: int = 20,
    rtol: float = 0.0,
    atol: float = 0.0,
    minimum_improvement_percent: float = 0.0,
) -> CudaGraphCampaignBenchmarkResult:
    """Prepare one capture executor and pair its eager-control and replay methods."""

    semantic_fingerprint = validate_cuda_graph_campaign(campaign)
    reverify = getattr(getattr(engine, "store", None), "reverify_content_identity", None)
    if not callable(reverify):
        raise RuntimeError("CUDA Graph evidence requires fresh QStore blob verification")
    reverified = reverify()
    if (
        not isinstance(reverified, Mapping)
        or reverified.get("content_identity_verified") is not True
        or reverified.get("blob_identity_verified") is not True
    ):
        raise RuntimeError("CUDA Graph QStore fresh-verification gate failed")
    runtime_configuration = validate_candidate_campaign_engine_binding(engine, campaign)
    runtime_environment = cuda_graph_runtime_environment(engine)
    values = np.ascontiguousarray(np.asarray(token_ids, dtype=np.int64))
    if values.ndim != 1 or not values.size:
        raise ValueError("CUDA Graph benchmark token IDs must be one-dimensional")
    if not campaign.input_binding.matches(values):
        raise ValueError("benchmark tokens do not match the campaign input binding")
    if capture_warmup < 1:
        raise ValueError("capture_warmup must be positive")
    prepare = getattr(engine, "prepare_selected_last_cuda_graph", None)
    if not callable(prepare):
        raise RuntimeError("engine cannot prepare a CUDA Graph executor")

    # Capture/residency setup is outside correctness, warmup, and timed acquisition.
    executor = prepare(
        [values],
        campaign.union_token_ids,
        warmup=capture_warmup,
    )
    eager_control = getattr(executor, "execute_eager_control", None)
    graph_replay = getattr(executor, "execute", None)
    close = getattr(executor, "close", None)
    if not callable(close):
        raise RuntimeError("CUDA Graph executor has no close method")

    def executor_evidence() -> Mapping[str, Any]:
        evidence = getattr(executor, "evidence", None)
        if not isinstance(evidence, Mapping):
            raise RuntimeError("CUDA Graph executor has no evidence mapping")
        return evidence

    try:
        if not callable(eager_control):
            raise RuntimeError("CUDA Graph executor has no matched eager-control method")
        if not callable(graph_replay):
            raise RuntimeError("CUDA Graph executor has no replay method")
        _validate_executor_evidence(executor_evidence(), minimum_execution_count=0)
        graph = campaign.base_bundle.graph
        if graph is None:
            raise RuntimeError("CUDA Graph campaign lost its compiled graph")
        reported_fabric = runtime_configuration["runtime_fabric"]
        return _run_matched_capture_benchmark(
            engine,
            _MatchedCaptureRunners(
                eager_control=eager_control,
                cuda_graph=graph_replay,
            ),
            executor_evidence=executor_evidence,
            semantic_fingerprint=semantic_fingerprint,
            campaign_fingerprint=campaign.fingerprint,
            compilation_fingerprint=graph.fingerprint,
            input_token_sha256=campaign.input_binding.token_sha256,
            query_count=campaign.sharing.query_count,
            union_candidate_count=campaign.sharing.union_candidate_count,
            warmup=warmup,
            trials=trials,
            rtol=rtol,
            atol=atol,
            minimum_improvement_percent=minimum_improvement_percent,
            reported_fabric=reported_fabric,
            runtime_environment=runtime_environment,
        )
    finally:
        close()


def compile_and_benchmark_cuda_graph_campaign(
    engine: Any,
    token_ids: np.ndarray | Sequence[int],
    readouts: Sequence[CandidateReadout],
    *,
    capture_warmup: int = 3,
    warmup: int = 3,
    trials: int = 20,
    rtol: float = 0.0,
    atol: float = 0.0,
    minimum_improvement_percent: float = 0.0,
) -> CudaGraphCampaignBenchmarkResult:
    """Compile one capture campaign, prepare once, and measure matched replay."""

    campaign = compile_candidate_campaign(
        engine,
        token_ids,
        readouts,
        capture_requested=True,
    )
    return benchmark_cuda_graph_campaign(
        engine,
        campaign,
        token_ids,
        capture_warmup=capture_warmup,
        warmup=warmup,
        trials=trials,
        rtol=rtol,
        atol=atol,
        minimum_improvement_percent=minimum_improvement_percent,
    )
