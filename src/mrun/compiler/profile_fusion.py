"""Profile-driven selection over the compiler's conservative legal fusion regions."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .graph_passes import FusionPlan, FusionRegion


@dataclass(frozen=True)
class FusionRegionProfile:
    region_id: str
    unfused_seconds: tuple[float, ...]
    fused_seconds: tuple[float, ...]
    unfused_bytes: int
    fused_bytes: int
    parity_passed: bool
    kernel_count_reduction: int
    compile_graph_factorial_complete: bool

    def __post_init__(self) -> None:
        if (
            not self.region_id
            or not self.unfused_seconds
            or not self.fused_seconds
            or len(self.unfused_seconds) != len(self.fused_seconds)
        ):
            raise ValueError("fusion profile is incomplete")
        if len(self.unfused_seconds) < 2:
            raise ValueError("fusion selection requires at least two paired samples")
        for samples in (self.unfused_seconds, self.fused_seconds):
            if any(not math.isfinite(value) or value <= 0 for value in samples):
                raise ValueError("fusion timing samples must be finite and positive")
        if self.unfused_bytes < 0 or self.fused_bytes < 0:
            raise ValueError("fusion byte counts must be non-negative")
        if self.kernel_count_reduction < 0:
            raise ValueError("fusion kernel-count reduction must be non-negative")

    @property
    def speedup(self) -> float:
        return _median(self.unfused_seconds) / _median(self.fused_seconds)

    @property
    def bytes_removed(self) -> int:
        return max(0, self.unfused_bytes - self.fused_bytes)

    @property
    def speedup_lower_95(self) -> float:
        ratios = [
            unfused / fused
            for unfused, fused in zip(
                self.unfused_seconds,
                self.fused_seconds,
                strict=True,
            )
        ]
        logs = [math.log(value) for value in ratios]
        mean = statistics.fmean(logs)
        standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
        return math.exp(mean - 1.96 * standard_error)


@dataclass(frozen=True)
class ProfileDrivenFusionPlan:
    legal_plan: FusionPlan
    selected_plan: FusionPlan
    profiles: tuple[FusionRegionProfile, ...]
    minimum_speedup: float

    @property
    def selected_region_ids(self) -> tuple[str, ...]:
        return tuple(region.region_id for region in self.selected_plan.regions)


def select_profiled_fusion_regions(
    legal_plan: FusionPlan,
    profiles: tuple[FusionRegionProfile, ...],
    *,
    minimum_speedup: float = 1.05,
) -> ProfileDrivenFusionPlan:
    """Select legal regions with parity, less traffic, and measured complete-region gain."""

    if not math.isfinite(minimum_speedup) or minimum_speedup <= 1:
        raise ValueError("profile-driven fusion requires a speedup threshold above one")
    legal_by_id = {region.region_id: region for region in legal_plan.regions}
    profile_by_id = {profile.region_id: profile for profile in profiles}
    if len(profile_by_id) != len(profiles):
        raise ValueError("fusion region profiles must be unique")
    unknown = set(profile_by_id) - set(legal_by_id)
    if unknown:
        raise ValueError(f"profile references illegal fusion regions: {sorted(unknown)}")
    selected: list[FusionRegion] = []
    rejected_nodes: list[str] = list(legal_plan.unfused_node_ids)
    accepted_edges = 0
    for region in legal_plan.regions:
        profile = profile_by_id.get(region.region_id)
        if (
            profile is not None
            and profile.parity_passed
            and profile.compile_graph_factorial_complete
            and profile.kernel_count_reduction > 0
            and profile.bytes_removed > 0
            and profile.speedup_lower_95 > minimum_speedup
        ):
            selected.append(region)
            accepted_edges += region.launches_removed
        else:
            rejected_nodes.extend(region.node_ids)
    selected_plan = FusionPlan(
        regions=tuple(selected),
        unfused_node_ids=tuple(dict.fromkeys(rejected_nodes)),
        barrier_node_ids=legal_plan.barrier_node_ids,
        candidate_edges=legal_plan.candidate_edges,
        accepted_edges=accepted_edges,
    )
    return ProfileDrivenFusionPlan(
        legal_plan=legal_plan,
        selected_plan=selected_plan,
        profiles=profiles,
        minimum_speedup=float(minimum_speedup),
    )


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


__all__ = [
    "FusionRegionProfile",
    "ProfileDrivenFusionPlan",
    "select_profiled_fusion_regions",
]
