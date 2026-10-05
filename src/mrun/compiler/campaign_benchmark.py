"""Paired latency, parity, and resource evidence for candidate campaigns."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .benchmark import (
    OutputParity,
    _compare_outputs,
    _duration_summary,
    _max_dequant_block_mb,
    _paired_median_bootstrap_ci95,
    _reset_max_dequant_block,
    _sync_engine,
)
from .bundle import CompilationBundle, compile_work_plan
from .campaign import (
    CandidateCampaign,
    CandidateCampaignExecutionResult,
    CandidateReadout,
    CandidateReadoutResult,
    prepare_candidate_campaign,
)
from .executable import candidate_outputs_from_values, execute_lowered_plan
from .ir import ExecutionMode, OutputContract

_RATIO_SEMANTICS = (
    "paired-duration-ratio:numerator-ms/denominator-ms;greater-than-one-favors-denominator"
)


def _require_finite_nonnegative(value: Any, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _parity_from_dict(payload: Mapping[str, Any]) -> OutputParity:
    return OutputParity(
        allclose=bool(payload["allclose"]),
        exact=bool(payload["exact"]),
        compared_values=int(payload["compared_values"]),
        max_abs_error=float(payload["max_abs_error"]),
        max_rel_error=float(payload["max_rel_error"]),
    )


@dataclass(frozen=True)
class CampaignLegStats:
    """Raw samples and robust summaries for one complete campaign execution."""

    samples_ms: tuple[float, ...]
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    max_dequant_block_mb: float | None

    @property
    def median_absolute_deviation_ms(self) -> float:
        values = np.asarray(self.samples_ms, dtype=np.float64)
        return float(np.median(np.abs(values - np.median(values))))

    @property
    def temporal_drift_ratio(self) -> float:
        """Late/early median ratio in acquisition order; diagnostic, not a correction."""

        half = len(self.samples_ms) // 2
        if half == 0:
            return 1.0
        early = np.asarray(self.samples_ms[:half], dtype=np.float64)
        late = np.asarray(self.samples_ms[-half:], dtype=np.float64)
        early_median = float(np.median(early))
        return float(np.median(late)) / early_median if early_median else math.inf

    def __post_init__(self) -> None:
        samples = tuple(
            _require_finite_nonnegative(value, "campaign latency sample")
            for value in self.samples_ms
        )
        if not samples:
            raise ValueError("campaign leg requires at least one latency sample")
        object.__setattr__(self, "samples_ms", samples)
        expected = _duration_summary(samples)
        for field_name, observed, wanted in zip(
            ("median_ms", "p95_ms", "min_ms", "max_ms"),
            (
                self.median_ms,
                self.p95_ms,
                self.min_ms,
                self.max_ms,
            ),
            expected,
            strict=True,
        ):
            value = _require_finite_nonnegative(observed, field_name)
            if not _same_float(value, wanted):
                raise ValueError(f"{field_name} does not match campaign samples")
            object.__setattr__(self, field_name, value)
        if self.max_dequant_block_mb is not None:
            object.__setattr__(
                self,
                "max_dequant_block_mb",
                _require_finite_nonnegative(
                    self.max_dequant_block_mb,
                    "max_dequant_block_mb",
                ),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples_ms": list(self.samples_ms),
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "max_dequant_block_mb": self.max_dequant_block_mb,
            "median_absolute_deviation_ms": self.median_absolute_deviation_ms,
            "temporal_drift_ratio": self.temporal_drift_ratio,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CampaignLegStats:
        block = payload.get("max_dequant_block_mb")
        result = cls(
            samples_ms=tuple(float(value) for value in payload["samples_ms"]),
            median_ms=float(payload["median_ms"]),
            p95_ms=float(payload["p95_ms"]),
            min_ms=float(payload["min_ms"]),
            max_ms=float(payload["max_ms"]),
            max_dequant_block_mb=None if block is None else float(block),
        )
        for field_name, expected in (
            ("median_absolute_deviation_ms", result.median_absolute_deviation_ms),
            ("temporal_drift_ratio", result.temporal_drift_ratio),
        ):
            claimed = payload.get(field_name)
            if claimed is not None and not _same_float(float(claimed), expected):
                raise ValueError(f"{field_name} does not match campaign samples")
        return result


@dataclass(frozen=True)
class PairedCampaignContrast:
    """A paired numerator/denominator latency contrast."""

    numerator_leg: str
    denominator_leg: str
    ratios: tuple[float, ...]
    median_ratio: float
    ci95: tuple[float, float]
    wins: int
    latency_reduction_percent: float
    ratio_semantics: str = _RATIO_SEMANTICS

    @property
    def paired_median_latency_reduction_percent(self) -> float:
        """Reduction derived from the paired median ratio, favoring the denominator."""

        return (1.0 - 1.0 / self.median_ratio) * 100.0

    def __post_init__(self) -> None:
        numerator = str(self.numerator_leg)
        denominator = str(self.denominator_leg)
        if not numerator or not denominator or numerator == denominator:
            raise ValueError("paired contrast leg names must be non-empty and distinct")
        object.__setattr__(self, "numerator_leg", numerator)
        object.__setattr__(self, "denominator_leg", denominator)
        ratios = tuple(float(value) for value in self.ratios)
        if not ratios or any(not math.isfinite(value) or value <= 0 for value in ratios):
            raise ValueError("paired contrast ratios must be finite and positive")
        object.__setattr__(self, "ratios", ratios)
        expected_median = float(np.median(np.asarray(ratios, dtype=np.float64)))
        median = float(self.median_ratio)
        if not _same_float(median, expected_median):
            raise ValueError("paired contrast median does not match ratios")
        object.__setattr__(self, "median_ratio", median)
        ci95 = tuple(float(value) for value in self.ci95)
        expected_ci95 = _paired_median_bootstrap_ci95(ratios)
        if len(ci95) != 2 or any(
            not _same_float(observed, expected)
            for observed, expected in zip(ci95, expected_ci95, strict=True)
        ):
            raise ValueError("paired contrast interval does not match ratios")
        object.__setattr__(self, "ci95", ci95)
        wins = int(self.wins)
        if wins != sum(value > 1.0 for value in ratios):
            raise ValueError("paired contrast win count does not match ratios")
        object.__setattr__(self, "wins", wins)
        reduction = float(self.latency_reduction_percent)
        if not math.isfinite(reduction):
            raise ValueError("paired contrast latency reduction must be finite")
        object.__setattr__(self, "latency_reduction_percent", reduction)
        if self.ratio_semantics != _RATIO_SEMANTICS:
            raise ValueError("unsupported paired contrast ratio semantics")

    def as_dict(self) -> dict[str, Any]:
        return {
            "numerator_leg": self.numerator_leg,
            "denominator_leg": self.denominator_leg,
            "ratios": list(self.ratios),
            "median_ratio": self.median_ratio,
            "ci95": list(self.ci95),
            "wins": self.wins,
            "latency_reduction_percent": self.latency_reduction_percent,
            "paired_median_latency_reduction_percent": (
                self.paired_median_latency_reduction_percent
            ),
            "ratio_semantics": self.ratio_semantics,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PairedCampaignContrast:
        result = cls(
            numerator_leg=str(payload["numerator_leg"]),
            denominator_leg=str(payload["denominator_leg"]),
            ratios=tuple(float(value) for value in payload["ratios"]),
            median_ratio=float(payload["median_ratio"]),
            ci95=tuple(float(value) for value in payload["ci95"]),
            wins=int(payload["wins"]),
            latency_reduction_percent=float(payload["latency_reduction_percent"]),
            ratio_semantics=str(payload.get("ratio_semantics", _RATIO_SEMANTICS)),
        )
        claimed = payload.get("paired_median_latency_reduction_percent")
        if claimed is not None and not _same_float(
            float(claimed),
            result.paired_median_latency_reduction_percent,
        ):
            raise ValueError("paired median latency reduction does not match paired ratios")
        return result


@dataclass(frozen=True)
class CandidateCampaignBenchmarkResult:
    """Three-leg evidence for independent, manual-union, and compiled execution."""

    campaign_fingerprint: str
    compilation_fingerprint: str
    rewrite_ids: tuple[str, ...]
    query_count: int
    reference_candidate_count: int
    union_candidate_count: int
    duplicate_candidate_reference_count: int
    shared_candidate_token_count: int
    max_candidate_multiplicity: int
    overlap_edge_count: int
    overlap_component_count: int
    independent: CampaignLegStats
    manual_union: CampaignLegStats
    compiled: CampaignLegStats
    independent_to_manual: PairedCampaignContrast
    independent_to_compiled: PairedCampaignContrast
    manual_to_compiled: PairedCampaignContrast
    independent_manual_parity: OutputParity
    independent_compiled_parity: OutputParity
    manual_compiled_parity: OutputParity
    minimum_improvement_percent: float
    improvement_demonstrated: bool
    trials: int
    warmup: int
    reported_fabric: str
    benchmark_basis: str = "measured-independent-vs-manual-union-vs-compiled-candidate-campaign"

    @property
    def manual_compiled_latency_verdict(self) -> str:
        low, high = self.manual_to_compiled.ci95
        if high < 1.0:
            return "compiled-slower"
        if low > 1.0:
            return "compiled-faster"
        return "no-detected-difference"

    @property
    def maximum_temporal_drift_factor(self) -> float:
        factors = []
        for leg in (self.independent, self.manual_union, self.compiled):
            ratio = leg.temporal_drift_ratio
            factors.append(max(ratio, 1.0 / ratio) if ratio else math.inf)
        return max(factors)

    def __post_init__(self) -> None:
        for field_name in ("campaign_fingerprint", "compilation_fingerprint"):
            value = str(getattr(self, field_name))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
            object.__setattr__(self, field_name, value)
        rewrite_ids = tuple(str(value) for value in self.rewrite_ids)
        if len(rewrite_ids) != len(set(rewrite_ids)) or any(not value for value in rewrite_ids):
            raise ValueError("campaign rewrite IDs must be unique and non-empty")
        object.__setattr__(self, "rewrite_ids", rewrite_ids)
        for field_name in (
            "query_count",
            "reference_candidate_count",
            "union_candidate_count",
            "duplicate_candidate_reference_count",
            "shared_candidate_token_count",
            "max_candidate_multiplicity",
            "overlap_edge_count",
            "overlap_component_count",
            "trials",
            "warmup",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative")
            object.__setattr__(self, field_name, value)
        if self.query_count < 2:
            raise ValueError("candidate campaign benchmark requires at least two queries")
        if self.union_candidate_count < 2:
            raise ValueError("candidate campaign union must contain at least two candidates")
        if (
            self.reference_candidate_count < 2 * self.query_count
            or self.reference_candidate_count < self.union_candidate_count
            or self.duplicate_candidate_reference_count
            != self.reference_candidate_count - self.union_candidate_count
            or self.shared_candidate_token_count > self.union_candidate_count
            or self.shared_candidate_token_count > self.duplicate_candidate_reference_count
            or not 1 <= self.max_candidate_multiplicity <= self.query_count
            or not 1 <= self.overlap_component_count <= self.query_count
            or self.overlap_edge_count > self.query_count * (self.query_count - 1) // 2
            or self.overlap_component_count < self.query_count - self.overlap_edge_count
        ):
            raise ValueError("candidate campaign count evidence is inconsistent")
        no_overlap = self.shared_candidate_token_count == 0
        if (
            no_overlap != (self.max_candidate_multiplicity == 1)
            or no_overlap != (self.duplicate_candidate_reference_count == 0)
            or no_overlap != (self.overlap_edge_count == 0)
            or no_overlap != (self.overlap_component_count == self.query_count)
        ):
            raise ValueError("candidate campaign overlap evidence is inconsistent")
        if self.trials <= 0 or any(
            len(leg.samples_ms) != self.trials
            for leg in (self.independent, self.manual_union, self.compiled)
        ):
            raise ValueError("campaign leg sample counts do not match trials")
        expected_contrasts = (
            (
                self.independent_to_manual,
                "independent",
                "manual_union",
                self.independent,
                self.manual_union,
            ),
            (
                self.independent_to_compiled,
                "independent",
                "compiled",
                self.independent,
                self.compiled,
            ),
            (
                self.manual_to_compiled,
                "manual_union",
                "compiled",
                self.manual_union,
                self.compiled,
            ),
        )
        for (
            contrast,
            numerator_name,
            denominator_name,
            numerator,
            denominator,
        ) in expected_contrasts:
            if (
                contrast.numerator_leg != numerator_name
                or contrast.denominator_leg != denominator_name
            ):
                raise ValueError("campaign contrast leg identity is inconsistent")
            expected_ratios = tuple(
                left / right
                for left, right in zip(
                    numerator.samples_ms,
                    denominator.samples_ms,
                    strict=True,
                )
            )
            if any(
                not _same_float(observed, expected)
                for observed, expected in zip(
                    contrast.ratios,
                    expected_ratios,
                    strict=True,
                )
            ):
                raise ValueError("campaign contrast ratios do not match leg samples")
            expected_reduction = (1.0 - denominator.median_ms / numerator.median_ms) * 100.0
            if not _same_float(
                contrast.latency_reduction_percent,
                expected_reduction,
            ):
                raise ValueError("campaign contrast latency reduction does not match leg medians")
        threshold = _require_finite_nonnegative(
            self.minimum_improvement_percent,
            "minimum_improvement_percent",
        )
        object.__setattr__(self, "minimum_improvement_percent", threshold)
        for parity in (
            self.independent_manual_parity,
            self.independent_compiled_parity,
            self.manual_compiled_parity,
        ):
            if (
                parity.compared_values < 0
                or not math.isfinite(parity.max_abs_error)
                or parity.max_abs_error < 0
                or not math.isfinite(parity.max_rel_error)
                or parity.max_rel_error < 0
                or (parity.exact and not parity.allclose)
            ):
                raise ValueError("campaign parity evidence is inconsistent")
        required_ratio = 1.0 + threshold / 100.0
        expected_improvement = (
            self.independent_compiled_parity.allclose
            and self.manual_compiled_parity.exact
            and self.independent_to_compiled.ci95[0] > required_ratio
        )
        if bool(self.improvement_demonstrated) != expected_improvement:
            raise ValueError("campaign improvement verdict is inconsistent")
        fabric = str(self.reported_fabric)
        basis = str(self.benchmark_basis)
        if not fabric or not basis:
            raise ValueError("campaign fabric and benchmark basis must be non-empty")
        object.__setattr__(self, "reported_fabric", fabric)
        object.__setattr__(self, "benchmark_basis", basis)

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_fingerprint": self.campaign_fingerprint,
            "compilation_fingerprint": self.compilation_fingerprint,
            "rewrite_ids": list(self.rewrite_ids),
            "query_count": self.query_count,
            "reference_candidate_count": self.reference_candidate_count,
            "union_candidate_count": self.union_candidate_count,
            "duplicate_candidate_reference_count": (self.duplicate_candidate_reference_count),
            "shared_candidate_token_count": self.shared_candidate_token_count,
            "max_candidate_multiplicity": self.max_candidate_multiplicity,
            "overlap_edge_count": self.overlap_edge_count,
            "overlap_component_count": self.overlap_component_count,
            "independent": self.independent.as_dict(),
            "manual_union": self.manual_union.as_dict(),
            "compiled": self.compiled.as_dict(),
            "independent_to_manual": self.independent_to_manual.as_dict(),
            "independent_to_compiled": self.independent_to_compiled.as_dict(),
            "manual_to_compiled": self.manual_to_compiled.as_dict(),
            "independent_manual_parity": self.independent_manual_parity.as_dict(),
            "independent_compiled_parity": self.independent_compiled_parity.as_dict(),
            "manual_compiled_parity": self.manual_compiled_parity.as_dict(),
            "minimum_improvement_percent": self.minimum_improvement_percent,
            "improvement_demonstrated": self.improvement_demonstrated,
            "manual_compiled_latency_verdict": (self.manual_compiled_latency_verdict),
            "maximum_temporal_drift_factor": (self.maximum_temporal_drift_factor),
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
    ) -> CandidateCampaignBenchmarkResult:
        result = cls(
            campaign_fingerprint=str(payload["campaign_fingerprint"]),
            compilation_fingerprint=str(payload["compilation_fingerprint"]),
            rewrite_ids=tuple(str(value) for value in payload["rewrite_ids"]),
            query_count=int(payload["query_count"]),
            reference_candidate_count=int(payload["reference_candidate_count"]),
            union_candidate_count=int(payload["union_candidate_count"]),
            duplicate_candidate_reference_count=int(payload["duplicate_candidate_reference_count"]),
            shared_candidate_token_count=int(payload["shared_candidate_token_count"]),
            max_candidate_multiplicity=int(payload["max_candidate_multiplicity"]),
            overlap_edge_count=int(payload["overlap_edge_count"]),
            overlap_component_count=int(payload["overlap_component_count"]),
            independent=CampaignLegStats.from_dict(payload["independent"]),
            manual_union=CampaignLegStats.from_dict(payload["manual_union"]),
            compiled=CampaignLegStats.from_dict(payload["compiled"]),
            independent_to_manual=PairedCampaignContrast.from_dict(
                payload["independent_to_manual"]
            ),
            independent_to_compiled=PairedCampaignContrast.from_dict(
                payload["independent_to_compiled"]
            ),
            manual_to_compiled=PairedCampaignContrast.from_dict(payload["manual_to_compiled"]),
            independent_manual_parity=_parity_from_dict(payload["independent_manual_parity"]),
            independent_compiled_parity=_parity_from_dict(payload["independent_compiled_parity"]),
            manual_compiled_parity=_parity_from_dict(payload["manual_compiled_parity"]),
            minimum_improvement_percent=float(payload["minimum_improvement_percent"]),
            improvement_demonstrated=bool(payload["improvement_demonstrated"]),
            trials=int(payload["trials"]),
            warmup=int(payload["warmup"]),
            reported_fabric=str(payload["reported_fabric"]),
            benchmark_basis=str(payload["benchmark_basis"]),
        )
        claimed_verdict = payload.get("manual_compiled_latency_verdict")
        if (
            claimed_verdict is not None
            and str(claimed_verdict) != result.manual_compiled_latency_verdict
        ):
            raise ValueError("manual/compiled latency verdict is inconsistent")
        claimed_drift = payload.get("maximum_temporal_drift_factor")
        if claimed_drift is not None and not _same_float(
            float(claimed_drift),
            result.maximum_temporal_drift_factor,
        ):
            raise ValueError("maximum campaign temporal drift is inconsistent")
        return result


@dataclass(frozen=True)
class _CampaignRunners:
    independent: Callable[[int], Any]
    manual_union: Callable[[], Any]
    compiled: Callable[[], Any]


def _readout_result(
    readout: CandidateReadout,
    values: torch.Tensor,
) -> CandidateReadoutResult:
    scores = torch.as_tensor(values)
    if scores.ndim != 1 or int(scores.shape[0]) != len(readout.candidate_token_ids):
        raise RuntimeError("candidate readout returned an invalid score shape")
    fp32_scores = scores.detach().float()
    summary = candidate_outputs_from_values(
        (fp32_scores,),
        (readout.candidate_token_ids,),
    )[0]
    return CandidateReadoutResult(
        query_id=readout.query_id,
        candidate_token_ids=readout.candidate_token_ids,
        candidate_logits=tuple(float(value) for value in fp32_scores.cpu()),
        winner_token_id=int(summary["winner_token_id"]),
        runner_up_token_id=int(summary["runner_up_token_id"]),
        winner_logit=float(summary["winner_logit"]),
        runner_up_logit=float(summary["runner_up_logit"]),
        margin=float(summary["margin"]),
    )


def _project_union_result(
    campaign: CandidateCampaign,
    union_scores: torch.Tensor,
    *,
    execution_route: str,
) -> CandidateCampaignExecutionResult:
    scores = torch.as_tensor(union_scores)
    if scores.ndim != 1 or int(scores.shape[0]) != len(campaign.union_token_ids):
        raise RuntimeError("manual campaign union returned an invalid score shape")
    offsets = {token: index for index, token in enumerate(campaign.union_token_ids)}
    readout_results = tuple(
        _readout_result(
            readout,
            scores.index_select(
                0,
                torch.as_tensor(
                    [offsets[token] for token in readout.candidate_token_ids],
                    dtype=torch.long,
                    device=scores.device,
                ),
            ),
        )
        for readout in campaign.readouts
    )
    return CandidateCampaignExecutionResult(
        campaign_fingerprint=campaign.fingerprint,
        union_token_ids=campaign.union_token_ids,
        union_logits=tuple(float(value) for value in scores.detach().cpu()),
        readouts=readout_results,
        evidence={
            "body_execution_count": 1,
            "execution_route": execution_route,
            "candidate_reference_count": (campaign.sharing.candidate_reference_count),
            "union_candidate_count": campaign.sharing.union_candidate_count,
            "query_reduction_semantics": ("local-candidate-order-then-torch-topk-2"),
        },
    )


def _normalized_readouts(
    results: Sequence[CandidateReadoutResult],
) -> tuple[dict[str, Any], ...]:
    return tuple(result.as_dict() for result in results)


def _compile_independent_readout_bundles(
    engine: Any,
    campaign: CandidateCampaign,
    token_ids: np.ndarray,
) -> tuple[tuple[CandidateReadout, CompilationBundle], ...]:
    plan_builder = getattr(engine, "build_work_plan", None)
    if not callable(plan_builder):
        raise ValueError("engine does not expose WorkPlan compilation")
    store = getattr(engine, "store", None)
    manifest = getattr(store, "man", None)
    if not isinstance(manifest, dict):
        raise ValueError("candidate campaign benchmark requires a QStore manifest")
    lowering_backend = campaign.base_bundle.lowered.backend
    expected_engine_backend = {
        "paged-qstore": "paged",
        "cuda-qstore": "dense-qstore-cuda",
    }.get(lowering_backend)
    if expected_engine_backend is None:
        raise RuntimeError("candidate campaign benchmark has an unsupported lowerer")
    if str(getattr(engine, "backend", "")) != expected_engine_backend:
        raise RuntimeError(
            "candidate campaign benchmark engine does not match the compiled lowerer"
        )
    compiled: list[tuple[CandidateReadout, CompilationBundle]] = []
    for index, readout in enumerate(campaign.readouts):
        plan = plan_builder(
            [token_ids],
            execution_mode=ExecutionMode.SCORE,
            output_contract=OutputContract.SELECTED_TOKEN_ROWS,
            required_output_rows=readout.candidate_token_ids,
            request_ids=(f"campaign-independent-{index}",),
        )
        bundle = compile_work_plan(
            plan,
            lowering_backend,
            manifest=manifest,
        )
        if bundle.graph is None or bundle.graph.rewrite_certificate is None:
            raise RuntimeError("independent campaign readout requires a certified graph")
        compiled.append((readout, bundle))
    return tuple(compiled)


def benchmark_candidate_campaign(
    engine: Any,
    campaign: CandidateCampaign,
    token_ids: np.ndarray | Sequence[int],
    *,
    warmup: int = 1,
    trials: int = 5,
    rtol: float = 1e-5,
    atol: float = 2e-5,
    minimum_improvement_percent: float = 1.0,
) -> CandidateCampaignBenchmarkResult:
    """Measure independent readouts, one manual union, and the compiled campaign.

    Plan/graph compilation occurs before correctness establishment and timing. Each
    independent leg executes every readout's own graph-bound selected-row WorkPlan
    sequentially. Its query execution order rotates per trial, but results return in the
    campaign's canonical readout order. The manual control calls one selected-row union
    head directly, and the compiled leg uses :func:`execute_candidate_campaign`.
    """

    values = np.ascontiguousarray(np.asarray(token_ids, dtype=np.int64))
    if values.ndim != 1 or not values.size:
        raise ValueError("campaign benchmark token IDs must be one-dimensional")
    if not campaign.input_binding.matches(values):
        raise ValueError("benchmark token IDs do not match the compiled campaign input binding")
    selected = getattr(engine, "selected_last_logits_batch", None)
    if not callable(selected):
        raise RuntimeError("engine cannot execute a manual selected-row union")
    independent_bundles = _compile_independent_readout_bundles(
        engine,
        campaign,
        values,
    )
    prepared_campaign = prepare_candidate_campaign(
        engine,
        campaign,
        values,
    )

    def run_independent(rotation: int) -> tuple[dict[str, Any], ...]:
        ordered = independent_bundles[rotation:] + independent_bundles[:rotation]
        by_query_id: dict[str, CandidateReadoutResult] = {}
        for readout, bundle in ordered:
            if bundle.graph is None:
                raise RuntimeError("independent candidate bundle lost its graph")
            execution = execute_lowered_plan(
                engine,
                bundle.plan,
                bundle.lowered,
                [values],
                graph_compilation=bundle.graph,
            )
            scores = torch.as_tensor(execution.outputs)
            if scores.ndim != 2 or tuple(scores.shape) != (
                1,
                len(readout.candidate_token_ids),
            ):
                raise RuntimeError("independent candidate WorkPlan returned an invalid score shape")
            by_query_id[readout.query_id] = _readout_result(
                readout,
                scores[0],
            )
        return _normalized_readouts(
            tuple(by_query_id[readout.query_id] for readout in campaign.readouts)
        )

    def run_manual_union() -> tuple[dict[str, Any], ...]:
        scores = torch.as_tensor(selected([values], campaign.union_token_ids))
        if scores.ndim != 2 or tuple(scores.shape) != (
            1,
            len(campaign.union_token_ids),
        ):
            raise RuntimeError("manual candidate union returned an invalid score shape")
        result = _project_union_result(
            campaign,
            scores[0],
            execution_route="manual-selected-row-union",
        )
        return _normalized_readouts(result.readouts)

    def run_compiled() -> tuple[dict[str, Any], ...]:
        result = prepared_campaign.execute()
        return _normalized_readouts(result.readouts)

    graph = campaign.base_bundle.graph
    if graph is None or graph.rewrite_certificate is None:
        raise RuntimeError("candidate campaign has no certified graph")
    sharing = campaign.sharing
    try:
        return _run_paired_campaign_benchmark(
            engine,
            _CampaignRunners(
                independent=run_independent,
                manual_union=run_manual_union,
                compiled=run_compiled,
            ),
            campaign_fingerprint=campaign.fingerprint,
            compilation_fingerprint=graph.fingerprint,
            rewrite_ids=graph.rewrite_certificate.rewrite_ids,
            query_count=sharing.query_count,
            reference_candidate_count=sharing.candidate_reference_count,
            union_candidate_count=sharing.union_candidate_count,
            duplicate_candidate_reference_count=(sharing.duplicate_candidate_reference_count),
            shared_candidate_token_count=sharing.shared_candidate_token_count,
            max_candidate_multiplicity=sharing.max_candidate_multiplicity,
            overlap_edge_count=len(sharing.overlap_edges),
            overlap_component_count=sharing.overlap_component_count,
            reported_fabric=campaign.base_bundle.lowered.reported_fabric,
            warmup=warmup,
            trials=trials,
            rtol=rtol,
            atol=atol,
            minimum_improvement_percent=minimum_improvement_percent,
        )
    finally:
        prepared_campaign.close()


def _leg_stats(
    samples: Sequence[float],
    max_blocks: Sequence[float],
) -> CampaignLegStats:
    median, p95, minimum, maximum = _duration_summary(samples)
    return CampaignLegStats(
        samples_ms=tuple(float(value) for value in samples),
        median_ms=median,
        p95_ms=p95,
        min_ms=minimum,
        max_ms=maximum,
        max_dequant_block_mb=max(max_blocks) if max_blocks else None,
    )


def _paired_contrast(
    numerator_name: str,
    denominator_name: str,
    numerator: CampaignLegStats,
    denominator: CampaignLegStats,
) -> PairedCampaignContrast:
    ratios = tuple(
        left / right if right else math.inf
        for left, right in zip(
            numerator.samples_ms,
            denominator.samples_ms,
            strict=True,
        )
    )
    median_ratio = float(np.median(np.asarray(ratios, dtype=np.float64)))
    return PairedCampaignContrast(
        numerator_leg=numerator_name,
        denominator_leg=denominator_name,
        ratios=ratios,
        median_ratio=median_ratio,
        ci95=_paired_median_bootstrap_ci95(ratios),
        wins=sum(value > 1.0 for value in ratios),
        latency_reduction_percent=(
            (1.0 - denominator.median_ms / numerator.median_ms) * 100.0
            if numerator.median_ms
            else -math.inf
        ),
    )


def _run_paired_campaign_benchmark(
    engine: Any,
    runners: _CampaignRunners,
    *,
    campaign_fingerprint: str,
    compilation_fingerprint: str,
    rewrite_ids: tuple[str, ...],
    query_count: int,
    reference_candidate_count: int,
    union_candidate_count: int,
    duplicate_candidate_reference_count: int,
    shared_candidate_token_count: int,
    max_candidate_multiplicity: int,
    overlap_edge_count: int,
    overlap_component_count: int,
    reported_fabric: str,
    warmup: int,
    trials: int,
    rtol: float,
    atol: float,
    minimum_improvement_percent: float,
) -> CandidateCampaignBenchmarkResult:
    """Run already-bound campaign legs under one balanced timing protocol."""

    if query_count <= 0:
        raise ValueError("candidate campaign must contain at least one query")
    if reference_candidate_count < union_candidate_count:
        raise ValueError("union candidate count cannot exceed reference occurrences")
    if union_candidate_count < 2:
        raise ValueError("candidate campaign union must contain at least two candidates")
    if warmup < 0 or trials <= 0:
        raise ValueError("warmup must be non-negative and trials must be positive")
    if rtol < 0 or atol < 0:
        raise ValueError("parity tolerances must be non-negative")
    if minimum_improvement_percent < 0:
        raise ValueError("minimum improvement threshold must be non-negative")

    # Correctness is established outside timing. The independent route is authoritative;
    # rotation zero preserves canonical query order for this reference.
    independent_output = runners.independent(0)
    manual_output = runners.manual_union()
    compiled_output = runners.compiled()
    independent_manual_parity = _compare_outputs(
        independent_output,
        manual_output,
        rtol=rtol,
        atol=atol,
    )
    independent_compiled_parity = _compare_outputs(
        independent_output,
        compiled_output,
        rtol=rtol,
        atol=atol,
    )
    manual_compiled_parity = _compare_outputs(
        manual_output,
        compiled_output,
        rtol=0.0,
        atol=0.0,
    )

    paths = ("independent", "manual_union", "compiled")

    def run_path(path: str, query_rotation: int) -> Any:
        if path == "independent":
            return runners.independent(query_rotation)
        if path == "manual_union":
            return runners.manual_union()
        return runners.compiled()

    for warmup_index in range(warmup):
        leg_rotation = warmup_index % len(paths)
        order = paths[leg_rotation:] + paths[:leg_rotation]
        query_rotation = warmup_index % query_count
        for path in order:
            run_path(path, query_rotation)

    samples: dict[str, list[float]] = {path: [] for path in paths}
    max_blocks: dict[str, list[float]] = {path: [] for path in paths}
    for trial in range(trials):
        leg_rotation = trial % len(paths)
        order = paths[leg_rotation:] + paths[:leg_rotation]
        query_rotation = trial % query_count
        for path in order:
            _sync_engine(engine)
            telemetry_reset = _reset_max_dequant_block(engine)
            started = time.perf_counter()
            run_path(path, query_rotation)
            _sync_engine(engine)
            samples[path].append((time.perf_counter() - started) * 1000.0)
            if telemetry_reset:
                observed = _max_dequant_block_mb(engine)
                if observed is not None:
                    max_blocks[path].append(observed)

    independent = _leg_stats(samples["independent"], max_blocks["independent"])
    manual_union = _leg_stats(samples["manual_union"], max_blocks["manual_union"])
    compiled = _leg_stats(samples["compiled"], max_blocks["compiled"])
    independent_to_manual = _paired_contrast(
        "independent",
        "manual_union",
        independent,
        manual_union,
    )
    independent_to_compiled = _paired_contrast(
        "independent",
        "compiled",
        independent,
        compiled,
    )
    manual_to_compiled = _paired_contrast(
        "manual_union",
        "compiled",
        manual_union,
        compiled,
    )
    required_ratio = 1.0 + minimum_improvement_percent / 100.0
    return CandidateCampaignBenchmarkResult(
        campaign_fingerprint=campaign_fingerprint,
        compilation_fingerprint=compilation_fingerprint,
        rewrite_ids=rewrite_ids,
        query_count=query_count,
        reference_candidate_count=reference_candidate_count,
        union_candidate_count=union_candidate_count,
        duplicate_candidate_reference_count=duplicate_candidate_reference_count,
        shared_candidate_token_count=shared_candidate_token_count,
        max_candidate_multiplicity=max_candidate_multiplicity,
        overlap_edge_count=overlap_edge_count,
        overlap_component_count=overlap_component_count,
        independent=independent,
        manual_union=manual_union,
        compiled=compiled,
        independent_to_manual=independent_to_manual,
        independent_to_compiled=independent_to_compiled,
        manual_to_compiled=manual_to_compiled,
        independent_manual_parity=independent_manual_parity,
        independent_compiled_parity=independent_compiled_parity,
        manual_compiled_parity=manual_compiled_parity,
        minimum_improvement_percent=minimum_improvement_percent,
        improvement_demonstrated=(
            independent_compiled_parity.allclose
            and manual_compiled_parity.exact
            and independent_to_compiled.ci95[0] > required_ratio
        ),
        trials=trials,
        warmup=warmup,
        reported_fabric=reported_fabric,
    )
