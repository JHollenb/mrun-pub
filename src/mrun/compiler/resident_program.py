"""Fail-closed P0-P8 status evaluator for the resident ScienceGraph proposal."""

from __future__ import annotations

from dataclasses import dataclass

from .phase_evidence import (
    ContinuousBatchEvidence,
    PrecisionOversizedEvidence,
    ProfileDrivenFusionEvidence,
    StatefulDecodeEvidence,
    TargetAlignedK4Evidence,
)
from .resident_promotions import ResidentTemplatePromotion, RuntimeCapabilityPromotion


@dataclass(frozen=True)
class MilestoneStatus:
    milestone: str
    promoted: bool
    reason: str


@dataclass(frozen=True)
class ResidentScienceGraphProgramStatus:
    milestones: tuple[MilestoneStatus, ...]

    @property
    def first_release_complete(self) -> bool:
        return all(item.promoted for item in self.milestones[:4])

    @property
    def complete_research_program_promoted(self) -> bool:
        return all(item.promoted for item in self.milestones)


def evaluate_resident_sciencegraph_program(
    *,
    automatic_campaign_eligible: int,
    automatic_campaign_fallbacks: int,
    automatic_campaign_parity: bool,
    automatic_campaign_suite_speedup: float,
    template_promotions: tuple[ResidentTemplatePromotion, ...],
    intervention_sciencegraph_qualified: bool,
    capability_promotions: tuple[RuntimeCapabilityPromotion, ...] = (),
    continuous_batch_evidence: ContinuousBatchEvidence | None = None,
    fusion_evidence: ProfileDrivenFusionEvidence | None = None,
    stateful_decode_evidence: StatefulDecodeEvidence | None = None,
    target_aligned_k4_evidence: TargetAlignedK4Evidence | None = None,
    precision_evidence: PrecisionOversizedEvidence | None = None,
) -> ResidentScienceGraphProgramStatus:
    required_shapes = {
        (1, sequence, selected)
        for sequence in (1, 5, 16)
        for selected in (2, 6, 16)
    }
    promoted_shapes = {record.shape_bucket for record in template_promotions}
    capabilities = {record.capability for record in capability_promotions}
    p0 = (
        automatic_campaign_eligible > 0
        and automatic_campaign_fallbacks >= 0
        and automatic_campaign_parity
        and automatic_campaign_suite_speedup > 1
    )
    p1 = any(record.shape_bucket == (1, 5, 6) for record in template_promotions)
    p2 = required_shapes <= promoted_shapes
    p3 = intervention_sciencegraph_qualified
    milestones = [
        MilestoneStatus("P0", p0, "automatic campaign suite gate"),
        MilestoneStatus("P1", p1, "rebindable B1/S5/K6 promotion"),
        MilestoneStatus("P2", p2, "nine-cell resident template family"),
        MilestoneStatus("P3", p3, "experiment-suite quotienting gate"),
    ]
    milestones.extend(
        (
            MilestoneStatus(
                "P4",
                continuous_batch_evidence is not None
                and "continuous_batch_capture" in capabilities,
                "captured continuous-batch evidence and promotion",
            ),
            MilestoneStatus(
                "P5",
                fusion_evidence is not None and "profile_driven_fusion" in capabilities,
                "profile-driven fusion evidence and promotion",
            ),
            MilestoneStatus(
                "P6",
                stateful_decode_evidence is not None
                and "stateful_decode_capture" in capabilities,
                "stateful captured-decode evidence and promotion",
            ),
            MilestoneStatus(
                "P7",
                target_aligned_k4_evidence is not None and "target_aligned_k4" in capabilities,
                "target-aligned K4 evidence and promotion",
            ),
        )
    )
    precision_capabilities = {"fp8_dense", "int4_dense", "route_first_moe"}
    milestones.append(
        MilestoneStatus(
            "P8",
            precision_evidence is not None and precision_capabilities <= capabilities,
            "precision/oversized evidence and FP8, int4, route-first promotions",
        )
    )
    return ResidentScienceGraphProgramStatus(tuple(milestones))


__all__ = [
    "MilestoneStatus",
    "ResidentScienceGraphProgramStatus",
    "evaluate_resident_sciencegraph_program",
]
