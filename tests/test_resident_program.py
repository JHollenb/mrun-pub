from __future__ import annotations

from mrun.compiler import (
    BUILTIN_RUNTIME_CAPABILITY_PROMOTIONS,
    ContinuousBatchEvidence,
    PrecisionOversizedEvidence,
    ProfileDrivenFusionEvidence,
    ResidentTemplatePromotion,
    RuntimeCapabilityPromotion,
    StatefulDecodeEvidence,
    TargetAlignedK4Evidence,
    evaluate_resident_sciencegraph_program,
    select_runtime_capability_promotion,
)


def _template(shape: tuple[int, int, int]) -> ResidentTemplatePromotion:
    return ResidentTemplatePromotion(
        promotion_id=f"template-{shape}",
        arena_fingerprint="a" * 64,
        template_implementation_sha256="b" * 64,
        mutable_binding_schema_sha256="c" * 64,
        numerical_contract="exact",
        output_contract="selected",
        shape_bucket=shape,
        device_identity="fake",
        runtime_identity_sha256="d" * 64,
        arena_bytes=10_000,
        template_bytes=500,
        setup_ms=1,
        rebind_ms=0.01,
        replay_speedup_lower_95=1.1,
        lane_count=1,
        contamination_trials=1_000,
        cancellation_passed=True,
        exact_parity=True,
        evidence_sha256="e" * 64,
        wheel_sha256="f" * 64,
    )


def _capability(name: str) -> RuntimeCapabilityPromotion:
    return RuntimeCapabilityPromotion(
        capability=name,
        implementation_sha256="1" * 64,
        evidence_sha256="2" * 64,
        numerical_contract="exact",
        device_identity="fake",
        parity_passed=True,
        performance_lower_95=1.1,
    )


def test_builtin_p4_capability_promotion_is_hardware_scoped() -> None:
    record = select_runtime_capability_promotion(
        BUILTIN_RUNTIME_CAPABILITY_PROMOTIONS,
        capability="continuous_batch_capture",
        numerical_contract="row-stable-triton-v1",
        device_identity="NVIDIA GeForce RTX 4080|cc8.9|vram16718168064",
    )
    assert record is not None
    assert record.performance_lower_95 > 13
    assert (
        select_runtime_capability_promotion(
            BUILTIN_RUNTIME_CAPABILITY_PROMOTIONS,
            capability="continuous_batch_capture",
            numerical_contract="row-stable-triton-v1",
            device_identity="another-gpu",
        )
        is None
    )


def test_program_status_distinguishes_first_release_from_later_research_promotions() -> None:
    templates = tuple(
        _template((1, sequence, selected))
        for sequence in (1, 5, 16)
        for selected in (2, 6, 16)
    )
    first_release = evaluate_resident_sciencegraph_program(
        automatic_campaign_eligible=100,
        automatic_campaign_fallbacks=3,
        automatic_campaign_parity=True,
        automatic_campaign_suite_speedup=2,
        template_promotions=templates,
        intervention_sciencegraph_qualified=True,
    )
    assert first_release.first_release_complete is True
    assert first_release.complete_research_program_promoted is False
    assert [item.milestone for item in first_release.milestones if not item.promoted] == [
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
    ]

    capabilities = tuple(
        _capability(name)
        for name in (
            "continuous_batch_capture",
            "profile_driven_fusion",
            "stateful_decode_capture",
            "target_aligned_k4",
            "fp8_dense",
            "int4_dense",
            "route_first_moe",
        )
    )
    complete = evaluate_resident_sciencegraph_program(
        automatic_campaign_eligible=100,
        automatic_campaign_fallbacks=0,
        automatic_campaign_parity=True,
        automatic_campaign_suite_speedup=2,
        template_promotions=templates,
        intervention_sciencegraph_qualified=True,
        capability_promotions=capabilities,
        continuous_batch_evidence=ContinuousBatchEvidence(
            (2, 4, 8, 16, 32), True, True, 1, 2, 0.5, 1.1, 100
        ),
        fusion_evidence=ProfileDrivenFusionEvidence(9, True, 1, 1, True, 1.1),
        stateful_decode_evidence=StatefulDecodeEvidence(
            256, 1_000, 32, True, True, True, True, True, True, True, 1.1
        ),
        target_aligned_k4_evidence=TargetAlignedK4Evidence(
            100, 200, True, True, True, 1.1
        ),
        precision_evidence=PrecisionOversizedEvidence(True, True, True, True, True, True),
    )
    assert complete.complete_research_program_promoted is True


def test_capability_records_without_decisive_phase_evidence_do_not_complete_program() -> None:
    capabilities = tuple(
        _capability(name)
        for name in (
            "continuous_batch_capture",
            "profile_driven_fusion",
            "stateful_decode_capture",
            "target_aligned_k4",
            "fp8_dense",
            "int4_dense",
            "route_first_moe",
        )
    )
    status = evaluate_resident_sciencegraph_program(
        automatic_campaign_eligible=1,
        automatic_campaign_fallbacks=0,
        automatic_campaign_parity=True,
        automatic_campaign_suite_speedup=2,
        template_promotions=(),
        intervention_sciencegraph_qualified=True,
        capability_promotions=capabilities,
    )
    assert not any(item.promoted for item in status.milestones[4:])
