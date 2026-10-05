"""Fail-closed native capability matching and resident placement planning."""

from __future__ import annotations

from collections import defaultdict

from .contracts import (
    BackendCapabilities,
    CompiledComponent,
    CompiledModelIdentity,
    ComponentPlacement,
    DeviceDescriptor,
    FallbackPolicy,
    PlacementPlan,
    Residency,
    StatePlacement,
    WorkloadSpec,
)


class CapabilityMismatch(RuntimeError):
    """A backend cannot execute the requested model/workload contract."""

    def __init__(self, gaps: tuple[str, ...]) -> None:
        if not gaps:
            raise ValueError("CapabilityMismatch requires at least one gap")
        self.gaps = gaps
        super().__init__("native capability mismatch: " + "; ".join(gaps))


class PlacementError(RuntimeError):
    """A capability-compatible workload cannot fit the declared placement budget."""


def capability_gaps(
    model: CompiledModelIdentity,
    workload: WorkloadSpec,
    capabilities: BackendCapabilities,
    device: DeviceDescriptor,
) -> tuple[str, ...]:
    """Return deterministic reasons why a backend/device route is incompatible.

    This check performs no model allocation or conversion.  In particular, a backend does not
    gain native codec authority merely because it could materialize an unsupported format through
    a host FP32 compatibility path.
    """

    gaps: list[str] = []
    if capabilities.fabric != device.fabric:
        gaps.append(
            f"backend fabric {capabilities.fabric!r} does not match device {device.fabric!r}"
        )
    if capabilities.memory_domain is not device.memory_domain:
        gaps.append(
            "backend memory domain "
            f"{capabilities.memory_domain.value!r} does not match device "
            f"{device.memory_domain.value!r}"
        )
    if model.architecture not in capabilities.architectures:
        gaps.append(f"architecture {model.architecture!r} is unsupported")

    missing_operators = sorted(set(model.operator_ids) - set(capabilities.operator_ids))
    if missing_operators:
        gaps.append(f"missing operator IDs: {missing_operators!r}")

    if workload.state_abi != model.state_abi:
        gaps.append(
            f"workload state ABI {workload.state_abi!r} does not match model {model.state_abi!r}"
        )
    if workload.state_abi not in capabilities.state_abis:
        gaps.append(f"state ABI {workload.state_abi!r} is unsupported")
    if workload.output_mode not in capabilities.output_modes:
        gaps.append(f"output mode {workload.output_mode.value!r} is unsupported")
    if workload.numerical_contract not in capabilities.numerical_contracts:
        gaps.append(f"numerical contract {workload.numerical_contract!r} is unsupported")
    if workload.max_context_tokens > model.max_context_tokens:
        gaps.append(
            f"context {workload.max_context_tokens} exceeds model limit {model.max_context_tokens}"
        )
    if workload.max_context_tokens > capabilities.max_context_tokens:
        gaps.append(
            f"context {workload.max_context_tokens} exceeds backend limit "
            f"{capabilities.max_context_tokens}"
        )
    if workload.max_batch_size > capabilities.max_batch_size:
        gaps.append(
            f"batch {workload.max_batch_size} exceeds backend limit {capabilities.max_batch_size}"
        )
    if workload.verify_tokens > capabilities.max_verify_tokens:
        gaps.append(
            f"verify width {workload.verify_tokens} exceeds backend limit "
            f"{capabilities.max_verify_tokens}"
        )

    if not capabilities.transactional_state:
        gaps.append("backend lacks transactional state")
    if not capabilities.scratch_only_steps:
        gaps.append("backend steps are not scratch-only")
    if workload.max_batch_size > 1 and not capabilities.independently_committable_rows:
        gaps.append("backend cannot commit pooled rows independently")

    model_roles = {component.role for component in model.components}
    missing_roles = sorted(set(workload.required_component_roles) - model_roles)
    if missing_roles:
        gaps.append(f"model lacks required component roles: {missing_roles!r}")

    required = [
        component
        for component in model.components
        if component.role in workload.required_component_roles
    ]
    for component in required:
        if not any(
            codec.supports(component, require_native=workload.require_native_codecs)
            for codec in capabilities.codecs
        ):
            qualifier = "native " if workload.require_native_codecs else ""
            gaps.append(
                f"component {component.component_id!r} has no {qualifier}codec/layout support "
                f"for {component.codec_id!r}/{component.layout_id!r}"
            )

    return tuple(dict.fromkeys(gaps))


def require_capabilities(
    model: CompiledModelIdentity,
    workload: WorkloadSpec,
    capabilities: BackendCapabilities,
    device: DeviceDescriptor,
) -> None:
    """Raise before allocation when a route is not semantically executable."""

    gaps = capability_gaps(model, workload, capabilities, device)
    if gaps:
        raise CapabilityMismatch(gaps)


def _selected_allocations(
    model: CompiledModelIdentity,
    workload: WorkloadSpec,
) -> tuple[tuple[str, tuple[CompiledComponent, ...]], ...]:
    selected_ids = {
        component.allocation_id
        for component in model.components
        if component.role in workload.required_component_roles
    }
    groups: dict[str, list[CompiledComponent]] = defaultdict(list)
    for component in model.components:
        if component.allocation_id in selected_ids:
            groups[component.allocation_id].append(component)
    return tuple(
        (allocation_id, tuple(sorted(components, key=lambda item: item.component_id)))
        for allocation_id, components in sorted(groups.items())
    )


def plan_resident_placement(
    model: CompiledModelIdentity,
    workload: WorkloadSpec,
    capabilities: BackendCapabilities,
    device: DeviceDescriptor,
    *,
    memory_budget_bytes: int | None = None,
) -> PlacementPlan:
    """Build a fully resident plan and reject capacity shortfalls before backend open.

    Host-tier/offload planning intentionally is not an implicit fallback here.  It needs a
    separately named planner and route because its service ceiling and physical traffic differ
    materially from a resident route.
    """

    require_capabilities(model, workload, capabilities, device)
    if memory_budget_bytes is not None and (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, int)
        or memory_budget_bytes <= 0
    ):
        raise ValueError("memory_budget_bytes must be a positive integer or None")
    effective_budget = device.available_bytes
    if memory_budget_bytes is not None:
        effective_budget = min(effective_budget, memory_budget_bytes)
    if effective_budget <= 0:
        raise PlacementError("device has no available native placement budget")

    placements: list[ComponentPlacement] = []
    for allocation_id, components in _selected_allocations(model, workload):
        physical = components[0]
        placements.append(
            ComponentPlacement(
                allocation_id=allocation_id,
                component_ids=tuple(component.component_id for component in components),
                roles=tuple(sorted({component.role for component in components})),
                codec_id=physical.codec_id,
                layout_id=physical.layout_id,
                memory_domain=device.memory_domain,
                residency=Residency.RESIDENT,
                physical_bytes=physical.physical_bytes,
            )
        )

    state_bytes = workload.max_batch_size * (
        model.state_fixed_bytes_per_row + model.state_bytes_per_token * workload.max_context_tokens
    )
    state = StatePlacement(
        state_abi=model.state_abi,
        dtype=model.state_dtype,
        memory_domain=device.memory_domain,
        bytes_per_token=model.state_bytes_per_token,
        fixed_bytes_per_row=model.state_fixed_bytes_per_row,
        reserved_bytes=state_bytes,
        max_batch_size=workload.max_batch_size,
        max_context_tokens=workload.max_context_tokens,
    )
    model_bytes = sum(component.physical_bytes for component in placements)
    total = model_bytes + state_bytes + workload.workspace_bytes + workload.headroom_bytes
    if total > effective_budget:
        raise PlacementError(
            "fully resident native placement exceeds budget: "
            f"model={model_bytes}, state={state_bytes}, workspace={workload.workspace_bytes}, "
            f"headroom={workload.headroom_bytes}, total={total}, budget={effective_budget}"
        )

    performance_valid = bool(
        workload.require_native_codecs and workload.fallback_policy is FallbackPolicy.DENY
    )
    return PlacementPlan(
        model_fingerprint=model.fingerprint,
        capability_fingerprint=capabilities.fingerprint,
        device_fingerprint=device.fingerprint,
        workload_fingerprint=workload.fingerprint,
        backend_id=capabilities.backend_id,
        device_id=device.device_id,
        components=tuple(placements),
        state=state,
        workspace_bytes=workload.workspace_bytes,
        headroom_bytes=workload.headroom_bytes,
        model_resident_bytes=model_bytes,
        total_reserved_bytes=total,
        memory_budget_bytes=effective_budget,
        fully_resident=True,
        fallback_policy=workload.fallback_policy,
        performance_claim_valid=performance_valid,
    )


def validate_placement_plan(
    plan: PlacementPlan,
    model: CompiledModelIdentity,
    workload: WorkloadSpec,
    capabilities: BackendCapabilities,
    device: DeviceDescriptor,
) -> None:
    """Recompute the canonical plan before a backend allocates or opens artifacts."""

    expected = plan_resident_placement(
        model,
        workload,
        capabilities,
        device,
        memory_budget_bytes=plan.memory_budget_bytes,
    )
    if plan != expected or plan.fingerprint != expected.fingerprint:
        raise PlacementError("placement plan is stale, forged, or belongs to another route")


__all__ = [
    "CapabilityMismatch",
    "PlacementError",
    "capability_gaps",
    "plan_resident_placement",
    "require_capabilities",
    "validate_placement_plan",
]
