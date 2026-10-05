"""Factorial ablation for graph sharing and selected-output pushdown.

The ordinary candidate-campaign benchmark isolates graph sharing by comparing
independent selected-row readouts with one manual or compiled selected-row union.
This module adds the complementary 2x2 experiment:

* independent versus union execution; and
* full-head versus selected-head output.

The four cells make the combined result auditable.  They distinguish a compiler
that merely coalesces requests from one that also removes unnecessary vocabulary
work, while retaining paired raw samples and correctness checks against the
established full-head route.
"""

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
    _max_dequant_block_mb,
    _paired_median_bootstrap_ci95,
    _reset_max_dequant_block,
    _sync_engine,
)
from .campaign import (
    CandidateCampaign,
    CandidateReadoutResult,
    prepare_candidate_campaign,
)
from .campaign_benchmark import (
    CampaignLegStats,
    PairedCampaignContrast,
    _leg_stats,
    _normalized_readouts,
    _paired_contrast,
    _project_union_result,
    _readout_result,
)

_INTERACTION_SEMANTICS = (
    "(full_head_independent/full_head_union)/"
    "(selected_independent/compiled_union);one-means-equal-sharing-effect"
)


def _require_sha256(value: str, field_name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


def _same_float(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _require_serialized_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_serialized_positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _parity_from_dict(payload: Mapping[str, Any]) -> OutputParity:
    return OutputParity(
        allclose=_require_serialized_bool(payload["allclose"], "parity.allclose"),
        exact=_require_serialized_bool(payload["exact"], "parity.exact"),
        compared_values=_require_serialized_positive_int(
            payload["compared_values"],
            "parity.compared_values",
        ),
        max_abs_error=float(payload["max_abs_error"]),
        max_rel_error=float(payload["max_rel_error"]),
    )


@dataclass(frozen=True)
class CandidateCampaignFactorialResult:
    """Paired four-cell evidence for sharing and output-pushdown effects."""

    campaign_fingerprint: str
    compilation_fingerprint: str
    query_count: int
    full_head_independent: CampaignLegStats
    full_head_union: CampaignLegStats
    selected_independent: CampaignLegStats
    compiled_union: CampaignLegStats
    sharing_at_full_head: PairedCampaignContrast
    sharing_at_selected_head: PairedCampaignContrast
    pushdown_at_independent: PairedCampaignContrast
    pushdown_at_union: PairedCampaignContrast
    combined: PairedCampaignContrast
    full_head_union_parity: OutputParity
    selected_independent_parity: OutputParity
    compiled_union_parity: OutputParity
    interaction_ratios: tuple[float, ...]
    interaction_median: float
    interaction_ci95: tuple[float, float]
    minimum_improvement_percent: float
    sharing_improvement_demonstrated: bool
    pushdown_improvement_demonstrated: bool
    combined_improvement_demonstrated: bool
    trials: int
    warmup: int
    reported_fabric: str
    benchmark_basis: str = "measured-2x2-full-vs-selected-by-independent-vs-compiled-union"
    interaction_semantics: str = _INTERACTION_SEMANTICS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "campaign_fingerprint",
            _require_sha256(self.campaign_fingerprint, "campaign_fingerprint"),
        )
        object.__setattr__(
            self,
            "compilation_fingerprint",
            _require_sha256(self.compilation_fingerprint, "compilation_fingerprint"),
        )
        query_count = int(self.query_count)
        trials = int(self.trials)
        warmup = int(self.warmup)
        if query_count < 2:
            raise ValueError("factorial campaign requires at least two queries")
        if trials <= 0 or warmup < 0:
            raise ValueError("trials must be positive and warmup must be non-negative")
        object.__setattr__(self, "query_count", query_count)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "warmup", warmup)

        legs = (
            self.full_head_independent,
            self.full_head_union,
            self.selected_independent,
            self.compiled_union,
        )
        if any(len(leg.samples_ms) != trials for leg in legs):
            raise ValueError("factorial leg sample counts do not match trials")

        expected_contrasts = (
            (
                self.sharing_at_full_head,
                "full_head_independent",
                "full_head_union",
                self.full_head_independent,
                self.full_head_union,
            ),
            (
                self.sharing_at_selected_head,
                "selected_independent",
                "compiled_union",
                self.selected_independent,
                self.compiled_union,
            ),
            (
                self.pushdown_at_independent,
                "full_head_independent",
                "selected_independent",
                self.full_head_independent,
                self.selected_independent,
            ),
            (
                self.pushdown_at_union,
                "full_head_union",
                "compiled_union",
                self.full_head_union,
                self.compiled_union,
            ),
            (
                self.combined,
                "full_head_independent",
                "compiled_union",
                self.full_head_independent,
                self.compiled_union,
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
                raise ValueError("factorial contrast leg identity is inconsistent")
            expected = tuple(
                left / right
                for left, right in zip(
                    numerator.samples_ms,
                    denominator.samples_ms,
                    strict=True,
                )
            )
            if any(
                not _same_float(observed, wanted)
                for observed, wanted in zip(
                    contrast.ratios,
                    expected,
                    strict=True,
                )
            ):
                raise ValueError("factorial contrast ratios do not match leg samples")

        ratios = tuple(float(value) for value in self.interaction_ratios)
        if len(ratios) != trials or any(not math.isfinite(value) or value <= 0 for value in ratios):
            raise ValueError("factorial interaction ratios must be finite and positive")
        expected_ratios = tuple(
            (full_independent / full_union) / (selected_independent / compiled_union)
            for full_independent, full_union, selected_independent, compiled_union in zip(
                self.full_head_independent.samples_ms,
                self.full_head_union.samples_ms,
                self.selected_independent.samples_ms,
                self.compiled_union.samples_ms,
                strict=True,
            )
        )
        if any(
            not _same_float(observed, wanted)
            for observed, wanted in zip(ratios, expected_ratios, strict=True)
        ):
            raise ValueError("factorial interaction ratios do not match leg samples")
        object.__setattr__(self, "interaction_ratios", ratios)
        expected_median = float(np.median(np.asarray(ratios, dtype=np.float64)))
        if not _same_float(self.interaction_median, expected_median):
            raise ValueError("factorial interaction median does not match ratios")
        object.__setattr__(self, "interaction_median", expected_median)
        expected_ci = _paired_median_bootstrap_ci95(ratios)
        observed_ci = tuple(float(value) for value in self.interaction_ci95)
        if len(observed_ci) != 2 or any(
            not _same_float(observed, wanted)
            for observed, wanted in zip(observed_ci, expected_ci, strict=True)
        ):
            raise ValueError("factorial interaction interval does not match ratios")
        object.__setattr__(self, "interaction_ci95", observed_ci)

        threshold = float(self.minimum_improvement_percent)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("minimum improvement threshold must be finite and non-negative")
        object.__setattr__(self, "minimum_improvement_percent", threshold)

        for parity in (
            self.full_head_union_parity,
            self.selected_independent_parity,
            self.compiled_union_parity,
        ):
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
                raise ValueError("factorial parity evidence is inconsistent")

        required_ratio = 1.0 + threshold / 100.0
        expected_verdicts = (
            (
                "sharing_improvement_demonstrated",
                self.selected_independent_parity.allclose
                and self.compiled_union_parity.allclose
                and self.sharing_at_selected_head.ci95[0] > required_ratio,
            ),
            (
                "pushdown_improvement_demonstrated",
                self.full_head_union_parity.allclose
                and self.compiled_union_parity.allclose
                and self.pushdown_at_union.ci95[0] > required_ratio,
            ),
            (
                "combined_improvement_demonstrated",
                self.compiled_union_parity.allclose and self.combined.ci95[0] > required_ratio,
            ),
        )
        for field_name, expected in expected_verdicts:
            observed = getattr(self, field_name)
            if type(observed) is not bool or observed != expected:
                raise ValueError(f"{field_name} is inconsistent with paired evidence")
        if not str(self.reported_fabric) or not str(self.benchmark_basis):
            raise ValueError("factorial fabric and benchmark basis must be non-empty")
        object.__setattr__(self, "reported_fabric", str(self.reported_fabric))
        object.__setattr__(self, "benchmark_basis", str(self.benchmark_basis))
        if self.interaction_semantics != _INTERACTION_SEMANTICS:
            raise ValueError("unsupported factorial interaction semantics")

    @property
    def maximum_temporal_drift_factor(self) -> float:
        factors = []
        for leg in (
            self.full_head_independent,
            self.full_head_union,
            self.selected_independent,
            self.compiled_union,
        ):
            ratio = leg.temporal_drift_ratio
            factors.append(max(ratio, 1.0 / ratio) if ratio else math.inf)
        return max(factors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_fingerprint": self.campaign_fingerprint,
            "compilation_fingerprint": self.compilation_fingerprint,
            "query_count": self.query_count,
            "full_head_independent": self.full_head_independent.as_dict(),
            "full_head_union": self.full_head_union.as_dict(),
            "selected_independent": self.selected_independent.as_dict(),
            "compiled_union": self.compiled_union.as_dict(),
            "sharing_at_full_head": self.sharing_at_full_head.as_dict(),
            "sharing_at_selected_head": self.sharing_at_selected_head.as_dict(),
            "pushdown_at_independent": self.pushdown_at_independent.as_dict(),
            "pushdown_at_union": self.pushdown_at_union.as_dict(),
            "combined": self.combined.as_dict(),
            "full_head_union_parity": self.full_head_union_parity.as_dict(),
            "selected_independent_parity": self.selected_independent_parity.as_dict(),
            "compiled_union_parity": self.compiled_union_parity.as_dict(),
            "interaction_ratios": list(self.interaction_ratios),
            "interaction_median": self.interaction_median,
            "interaction_ci95": list(self.interaction_ci95),
            "interaction_semantics": self.interaction_semantics,
            "minimum_improvement_percent": self.minimum_improvement_percent,
            "sharing_improvement_demonstrated": (self.sharing_improvement_demonstrated),
            "pushdown_improvement_demonstrated": (self.pushdown_improvement_demonstrated),
            "combined_improvement_demonstrated": (self.combined_improvement_demonstrated),
            "trials": self.trials,
            "warmup": self.warmup,
            "reported_fabric": self.reported_fabric,
            "benchmark_basis": self.benchmark_basis,
            "maximum_temporal_drift_factor": self.maximum_temporal_drift_factor,
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
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateCampaignFactorialResult:
        result = cls(
            campaign_fingerprint=str(payload["campaign_fingerprint"]),
            compilation_fingerprint=str(payload["compilation_fingerprint"]),
            query_count=int(payload["query_count"]),
            full_head_independent=CampaignLegStats.from_dict(payload["full_head_independent"]),
            full_head_union=CampaignLegStats.from_dict(payload["full_head_union"]),
            selected_independent=CampaignLegStats.from_dict(payload["selected_independent"]),
            compiled_union=CampaignLegStats.from_dict(payload["compiled_union"]),
            sharing_at_full_head=PairedCampaignContrast.from_dict(payload["sharing_at_full_head"]),
            sharing_at_selected_head=PairedCampaignContrast.from_dict(
                payload["sharing_at_selected_head"]
            ),
            pushdown_at_independent=PairedCampaignContrast.from_dict(
                payload["pushdown_at_independent"]
            ),
            pushdown_at_union=PairedCampaignContrast.from_dict(payload["pushdown_at_union"]),
            combined=PairedCampaignContrast.from_dict(payload["combined"]),
            full_head_union_parity=_parity_from_dict(payload["full_head_union_parity"]),
            selected_independent_parity=_parity_from_dict(payload["selected_independent_parity"]),
            compiled_union_parity=_parity_from_dict(payload["compiled_union_parity"]),
            interaction_ratios=tuple(float(value) for value in payload["interaction_ratios"]),
            interaction_median=float(payload["interaction_median"]),
            interaction_ci95=tuple(float(value) for value in payload["interaction_ci95"]),
            interaction_semantics=str(payload.get("interaction_semantics", _INTERACTION_SEMANTICS)),
            minimum_improvement_percent=float(payload["minimum_improvement_percent"]),
            sharing_improvement_demonstrated=_require_serialized_bool(
                payload["sharing_improvement_demonstrated"],
                "sharing_improvement_demonstrated",
            ),
            pushdown_improvement_demonstrated=_require_serialized_bool(
                payload["pushdown_improvement_demonstrated"],
                "pushdown_improvement_demonstrated",
            ),
            combined_improvement_demonstrated=_require_serialized_bool(
                payload["combined_improvement_demonstrated"],
                "combined_improvement_demonstrated",
            ),
            trials=int(payload["trials"]),
            warmup=int(payload["warmup"]),
            reported_fabric=str(payload["reported_fabric"]),
            benchmark_basis=str(payload["benchmark_basis"]),
        )
        claimed_drift = payload.get("maximum_temporal_drift_factor")
        if claimed_drift is not None and not _same_float(
            float(claimed_drift),
            result.maximum_temporal_drift_factor,
        ):
            raise ValueError("factorial maximum temporal drift is inconsistent")
        return result

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
    ) -> CandidateCampaignFactorialResult:
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise TypeError("serialized factorial campaign must be an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class _FactorialRunners:
    full_head_independent: Callable[[int], Any]
    full_head_union: Callable[[], Any]
    selected_independent: Callable[[int], Any]
    compiled_union: Callable[[], Any]


def _run_factorial_campaign_benchmark(
    engine: Any,
    runners: _FactorialRunners,
    *,
    campaign_fingerprint: str,
    compilation_fingerprint: str,
    query_count: int,
    reported_fabric: str,
    warmup: int,
    trials: int,
    rtol: float,
    atol: float,
    minimum_improvement_percent: float,
) -> CandidateCampaignFactorialResult:
    """Run four already-bound cells under one rotating paired protocol."""

    if query_count < 2:
        raise ValueError("factorial campaign requires at least two queries")
    if warmup < 0 or trials <= 0:
        raise ValueError("warmup must be non-negative and trials must be positive")
    if rtol < 0 or atol < 0:
        raise ValueError("parity tolerances must be non-negative")
    if minimum_improvement_percent < 0:
        raise ValueError("minimum improvement threshold must be non-negative")

    reference = runners.full_head_independent(0)
    full_union_output = runners.full_head_union()
    selected_independent_output = runners.selected_independent(0)
    compiled_output = runners.compiled_union()
    full_union_parity = _compare_outputs(
        reference,
        full_union_output,
        rtol=rtol,
        atol=atol,
    )
    selected_independent_parity = _compare_outputs(
        reference,
        selected_independent_output,
        rtol=rtol,
        atol=atol,
    )
    compiled_union_parity = _compare_outputs(
        reference,
        compiled_output,
        rtol=rtol,
        atol=atol,
    )

    paths = (
        "full_head_independent",
        "full_head_union",
        "selected_independent",
        "compiled_union",
    )

    def run_path(path: str, query_rotation: int) -> Any:
        if path == "full_head_independent":
            return runners.full_head_independent(query_rotation)
        if path == "full_head_union":
            return runners.full_head_union()
        if path == "selected_independent":
            return runners.selected_independent(query_rotation)
        return runners.compiled_union()

    for warmup_index in range(warmup):
        rotation = warmup_index % len(paths)
        order = paths[rotation:] + paths[:rotation]
        query_rotation = warmup_index % query_count
        for path in order:
            run_path(path, query_rotation)

    samples: dict[str, list[float]] = {path: [] for path in paths}
    max_blocks: dict[str, list[float]] = {path: [] for path in paths}
    for trial in range(trials):
        rotation = trial % len(paths)
        order = paths[rotation:] + paths[:rotation]
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

    full_head_independent = _leg_stats(
        samples["full_head_independent"],
        max_blocks["full_head_independent"],
    )
    full_head_union = _leg_stats(
        samples["full_head_union"],
        max_blocks["full_head_union"],
    )
    selected_independent = _leg_stats(
        samples["selected_independent"],
        max_blocks["selected_independent"],
    )
    compiled_union = _leg_stats(
        samples["compiled_union"],
        max_blocks["compiled_union"],
    )
    sharing_at_full_head = _paired_contrast(
        "full_head_independent",
        "full_head_union",
        full_head_independent,
        full_head_union,
    )
    sharing_at_selected_head = _paired_contrast(
        "selected_independent",
        "compiled_union",
        selected_independent,
        compiled_union,
    )
    pushdown_at_independent = _paired_contrast(
        "full_head_independent",
        "selected_independent",
        full_head_independent,
        selected_independent,
    )
    pushdown_at_union = _paired_contrast(
        "full_head_union",
        "compiled_union",
        full_head_union,
        compiled_union,
    )
    combined = _paired_contrast(
        "full_head_independent",
        "compiled_union",
        full_head_independent,
        compiled_union,
    )
    interaction_ratios = tuple(
        (full_independent / full_union) / (selected_independent_value / compiled_value)
        for full_independent, full_union, selected_independent_value, compiled_value in zip(
            full_head_independent.samples_ms,
            full_head_union.samples_ms,
            selected_independent.samples_ms,
            compiled_union.samples_ms,
            strict=True,
        )
    )
    required_ratio = 1.0 + minimum_improvement_percent / 100.0
    return CandidateCampaignFactorialResult(
        campaign_fingerprint=campaign_fingerprint,
        compilation_fingerprint=compilation_fingerprint,
        query_count=query_count,
        full_head_independent=full_head_independent,
        full_head_union=full_head_union,
        selected_independent=selected_independent,
        compiled_union=compiled_union,
        sharing_at_full_head=sharing_at_full_head,
        sharing_at_selected_head=sharing_at_selected_head,
        pushdown_at_independent=pushdown_at_independent,
        pushdown_at_union=pushdown_at_union,
        combined=combined,
        full_head_union_parity=full_union_parity,
        selected_independent_parity=selected_independent_parity,
        compiled_union_parity=compiled_union_parity,
        interaction_ratios=interaction_ratios,
        interaction_median=float(np.median(interaction_ratios)),
        interaction_ci95=_paired_median_bootstrap_ci95(interaction_ratios),
        minimum_improvement_percent=minimum_improvement_percent,
        sharing_improvement_demonstrated=(
            selected_independent_parity.allclose
            and compiled_union_parity.allclose
            and sharing_at_selected_head.ci95[0] > required_ratio
        ),
        pushdown_improvement_demonstrated=(
            full_union_parity.allclose
            and compiled_union_parity.allclose
            and pushdown_at_union.ci95[0] > required_ratio
        ),
        combined_improvement_demonstrated=(
            compiled_union_parity.allclose and combined.ci95[0] > required_ratio
        ),
        trials=trials,
        warmup=warmup,
        reported_fabric=reported_fabric,
    )


def benchmark_candidate_campaign_factorial(
    engine: Any,
    campaign: CandidateCampaign,
    token_ids: np.ndarray | Sequence[int],
    *,
    warmup: int = 1,
    trials: int = 5,
    rtol: float = 1e-5,
    atol: float = 2e-5,
    minimum_improvement_percent: float = 1.0,
) -> CandidateCampaignFactorialResult:
    """Measure the full/selected by independent/union 2x2 campaign ablation."""

    values = np.ascontiguousarray(np.asarray(token_ids, dtype=np.int64))
    if values.ndim != 1 or not values.size:
        raise ValueError("campaign factorial token IDs must be one-dimensional")
    if not campaign.input_binding.matches(values):
        raise ValueError("factorial token IDs do not match the compiled campaign input binding")
    full_head = getattr(engine, "logits_batch", None)
    selected = getattr(engine, "selected_last_logits_batch", None)
    if not callable(full_head) or not callable(selected):
        raise RuntimeError(
            "factorial campaign requires full-head and selected-head batch execution"
        )
    prepared = prepare_candidate_campaign(engine, campaign, values)

    def full_scores() -> torch.Tensor:
        outputs = full_head([values])
        if len(outputs) != 1:
            raise RuntimeError("full-head factorial route returned the wrong batch size")
        logits = torch.as_tensor(outputs[0])
        if logits.ndim != 2 or int(logits.shape[-1]) <= max(campaign.union_token_ids):
            raise RuntimeError("full-head factorial route returned invalid logits")
        return logits[-1]

    def full_head_independent(rotation: int) -> tuple[dict[str, Any], ...]:
        ordered = campaign.readouts[rotation:] + campaign.readouts[:rotation]
        by_query_id: dict[str, CandidateReadoutResult] = {}
        for readout in ordered:
            scores = full_scores().index_select(
                0,
                torch.as_tensor(
                    readout.candidate_token_ids,
                    dtype=torch.long,
                ),
            )
            by_query_id[readout.query_id] = _readout_result(readout, scores)
        return _normalized_readouts(
            tuple(by_query_id[readout.query_id] for readout in campaign.readouts)
        )

    def full_head_union() -> tuple[dict[str, Any], ...]:
        scores = full_scores().index_select(
            0,
            torch.as_tensor(campaign.union_token_ids, dtype=torch.long),
        )
        projected = _project_union_result(
            campaign,
            scores,
            execution_route="manual-full-head-union",
        )
        return _normalized_readouts(projected.readouts)

    def selected_independent(rotation: int) -> tuple[dict[str, Any], ...]:
        ordered = campaign.readouts[rotation:] + campaign.readouts[:rotation]
        by_query_id: dict[str, CandidateReadoutResult] = {}
        for readout in ordered:
            scores = torch.as_tensor(selected([values], readout.candidate_token_ids))
            if scores.ndim != 2 or tuple(scores.shape) != (
                1,
                len(readout.candidate_token_ids),
            ):
                raise RuntimeError("selected-head factorial route returned invalid logits")
            by_query_id[readout.query_id] = _readout_result(
                readout,
                scores[0],
            )
        return _normalized_readouts(
            tuple(by_query_id[readout.query_id] for readout in campaign.readouts)
        )

    def compiled_union() -> tuple[dict[str, Any], ...]:
        return _normalized_readouts(prepared.execute().readouts)

    graph = campaign.base_bundle.graph
    if graph is None or graph.rewrite_certificate is None:
        raise RuntimeError("factorial candidate campaign has no certified graph")
    try:
        return _run_factorial_campaign_benchmark(
            engine,
            _FactorialRunners(
                full_head_independent=full_head_independent,
                full_head_union=full_head_union,
                selected_independent=selected_independent,
                compiled_union=compiled_union,
            ),
            campaign_fingerprint=campaign.fingerprint,
            compilation_fingerprint=graph.fingerprint,
            query_count=len(campaign.readouts),
            reported_fabric=campaign.base_bundle.lowered.reported_fabric,
            warmup=warmup,
            trials=trials,
            rtol=rtol,
            atol=atol,
            minimum_improvement_percent=minimum_improvement_percent,
        )
    finally:
        prepared.close()
