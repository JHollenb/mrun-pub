from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mrun.compiler import (
    CaptureCapability,
    CaptureKind,
    CaptureResourceEstimate,
    CaptureRetention,
    CaptureStateOwner,
    DenseWorkPlan,
    InterventionAuthority,
    OutputContract,
    TypedCaptureSpec,
    build_dense_work_plan,
    lower_work_plan,
)


def _estimate(*, retained: int = 256) -> CaptureResourceEstimate:
    return CaptureResourceEstimate(
        model_loads=1,
        prefix_calls=1,
        suffix_calls=1,
        backward_calls=0,
        device_bytes=4096,
        retained_artifact_bytes=retained,
        expected_liveness_steps=2,
    )


def _spec(capture_id: str, row_id: str, *, port: str) -> TypedCaptureSpec:
    return TypedCaptureSpec(
        capture_id=capture_id,
        semantic_role="post-rope-query",
        physical_port=port,
        state_owner=CaptureStateOwner.ATTENTION,
        capture_kind=CaptureKind.PROJECTION,
        dtype="bf16",
        accumulator_dtype="fp32",
        numerical_contract="same-codec-w8a16",
        intended_consumer="carry-suffix",
        terminal_assay="carry-margin",
        on_device_reduction=True,
        retention=CaptureRetention.METADATA_AND_DIFFS,
        max_retained_bytes=512,
        specimen_row_ids=(row_id,),
        parent_statecut_id="c" * 64,
        branch_id=f"branch-{capture_id}",
        source_hashes=("d" * 64,),
        intervention_authority=InterventionAuthority.OBSERVE_ONLY,
        required_engine_capabilities=("post-rope", "selected-capture"),
        resource_estimate=_estimate(),
        layer_index=3,
        token_indices=(2,),
    )


def _capability(*, max_retained_bytes: int = 4096, backend: str = "cuda-qstore"):
    return CaptureCapability(
        backend=backend,
        runtime_entrypoint="selected_capture_batch",
        semantic_roles=("post-rope-query",),
        physical_ports=("layer.3.q.rope.out", "layer.4.q.rope.out"),
        state_owners=(CaptureStateOwner.ATTENTION,),
        capture_kinds=(CaptureKind.PROJECTION,),
        dtypes=("bf16",),
        accumulator_dtypes=("fp32",),
        numerical_contracts=("same-codec-w8a16",),
        intervention_authorities=(InterventionAuthority.OBSERVE_ONLY,),
        engine_capabilities=("selected-capture", "post-rope"),
        supports_on_device_reduction=True,
        supports_backward=False,
        max_specs=4,
        max_retained_bytes=max_retained_bytes,
    )


def _plan(specs, capability=None):
    return build_dense_work_plan(
        model_name="tiny",
        model_revision="a" * 64,
        store_fingerprint="b" * 64,
        batch_size=2,
        sequence_length=3,
        output_contract=OutputContract.SELECTED_CAPTURE,
        numerical_contract="same-codec-w8a16",
        request_ids=("row-a", "row-b"),
        request_slots=(0, 1),
        activation_dtype="bf16",
        weight_dtype="int8",
        accumulator_dtype="fp32",
        page_sequence=("embed", "L3.q", "L4.q", "norm.final"),
        capture_specs=specs,
        capture_capability=_capability() if capability is None else capability,
        metadata={
            "engine_device": "cuda",
            "logical_head_access": "none",
            "logical_head_row_count": 0,
            "configured_output_row_count": 8,
        },
    )


def test_typed_capture_plan_is_canonical_metadata_only_and_lowerable() -> None:
    first_spec = _spec("q3", "row-a", port="layer.3.q.rope.out")
    second_spec = _spec("q4", "row-b", port="layer.4.q.rope.out")
    first = _plan((second_spec, first_spec))
    second = _plan((first_spec, second_spec))

    assert [spec.capture_id for spec in first.capture_specs] == ["q3", "q4"]
    assert first.fingerprint == second.fingerprint
    assert DenseWorkPlan.from_json(first.to_json()) == first
    assert first.schema_version == "mrun-dense-workplan-v3"

    lowered = lower_work_plan(first, "dense-qstore-cuda")
    output_step = lowered.steps[-1]
    params = dict(output_step.params)
    assert output_step.operation == "selected_capture_output"
    assert params["capability_fingerprint"] == first.capture_capability.fingerprint
    assert params["aggregate_retained_byte_cap"] == 1024
    assert params["payload_storage"] == "external-content-addressed-runtime-inputs-only"
    assert "capture-retained-byte-cap-enforced" in lowered.evidence_requirements

    serialized = first.to_json()
    assert "tensor_values" not in serialized
    assert len(serialized) < 10_000


def test_typed_capture_fails_closed_on_missing_or_inexact_capability() -> None:
    spec = _spec("q3", "row-a", port="layer.3.q.rope.out")
    with pytest.raises(ValueError, match="typed capture specs and an exact backend capability"):
        build_dense_work_plan(
            model_name="tiny",
            model_revision="a" * 64,
            store_fingerprint="b" * 64,
            batch_size=2,
            sequence_length=3,
            output_contract=OutputContract.SELECTED_CAPTURE,
            numerical_contract="same-codec-w8a16",
            request_ids=("row-a", "row-b"),
            capture_specs=(spec,),
        )

    with pytest.raises(ValueError, match="aggregate capture retained-byte caps"):
        _plan(
            (spec, _spec("q4", "row-b", port="layer.4.q.rope.out")),
            _capability(max_retained_bytes=900),
        )

    with pytest.raises(ValueError, match="physical port is not supported"):
        _plan((replace(spec, physical_port="layer.9.q.rope.out"),))

    valid = _plan((spec,))
    with pytest.raises(ValueError, match="capability backend does not match"):
        lower_work_plan(
            replace(valid, capture_capability=_capability(backend="paged-qstore")), "cuda"
        )


def test_capture_schema_rejects_tensor_like_payload_fields_and_legacy_authority() -> None:
    spec_payload = _spec("q3", "row-a", port="layer.3.q.rope.out").as_dict()
    spec_payload["tensor_values"] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="unknown fields"):
        TypedCaptureSpec.from_dict(spec_payload)

    legacy = build_dense_work_plan(
        model_name="tiny",
        model_revision="a" * 64,
        store_fingerprint="b" * 64,
        batch_size=1,
        sequence_length=1,
    ).as_dict()
    legacy["schema_version"] = "mrun-dense-workplan-v2"
    legacy.pop("capture_specs")
    legacy.pop("capture_capability")
    migrated = DenseWorkPlan.from_dict(json.loads(json.dumps(legacy)))
    assert migrated.schema_version == "mrun-dense-workplan-v3"
    assert migrated.capture_specs == ()
    assert migrated.capture_capability is None

    legacy["output_contract"] = OutputContract.SELECTED_CAPTURE.value
    with pytest.raises(ValueError, match="no typed v3 migration"):
        DenseWorkPlan.from_dict(legacy)
