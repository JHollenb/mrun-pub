"""Tests for the typed MARS scene-state/debugger contract."""

from __future__ import annotations

import pytest

from mrun.diffusion import (
    SceneIntervention,
    SceneState,
    SceneStateDebugger,
    SceneStateError,
)


def _state() -> SceneState:
    return (
        SceneState()
        .set_field("subject", "color", "red", kind="attribute", provenance="dataset")
        .set_field("subject", "shape", "circle", kind="attribute", provenance="dataset")
        .set_field("object", "color", "blue", kind="attribute", provenance="dataset")
        .set_field("object", "shape", "square", kind="attribute", provenance="dataset")
    )


def test_scene_state_is_content_addressed_and_typed() -> None:
    first = _state()
    second = _state()
    assert first.fingerprint == second.fingerprint
    assert first.to_payload()["slots"]["subject"]["fields"]["color"]["kind"] == "attribute"
    assert first.field("subject", "color").provenance == "dataset"


def test_scene_debugger_records_field_swap_and_preserves_other_fields() -> None:
    before = _state()
    after, record = SceneStateDebugger.apply(
        before,
        SceneIntervention.swap("subject", "object", "color"),
    )
    assert after.field("subject", "color").value == "blue"
    assert after.field("object", "color").value == "red"
    assert after.field("subject", "shape").value == "circle"
    assert after.field("object", "shape").value == "square"
    assert record.changed_addresses == ("slots.object.color", "slots.subject.color")
    assert record.before_fingerprint != record.after_fingerprint
    assert record.after_generation == record.before_generation + 1


def test_scene_debugger_supports_clear_and_address_permutation() -> None:
    before = _state()
    cleared, clear_record = SceneStateDebugger.apply(
        before,
        SceneIntervention.clear("subject", "shape"),
    )
    assert "shape" not in cleared.slots["subject"].fields
    assert clear_record.changed_addresses == ("slots.subject.shape",)

    permuted, permutation_record = SceneStateDebugger.apply(
        before,
        SceneIntervention.permute({"subject": "object", "object": "subject"}),
    )
    assert permuted.field("subject", "color").value == "blue"
    assert permuted.field("object", "color").value == "red"
    assert permutation_record.changed_addresses == (
        "slots.object.color",
        "slots.object.shape",
        "slots.subject.color",
        "slots.subject.shape",
    )


def test_scene_state_interventions_fail_closed() -> None:
    with pytest.raises(SceneStateError):
        _state().clear_field("unknown", "color")
    with pytest.raises(SceneStateError):
        _state().swap_fields("subject", "object", "missing")
    with pytest.raises(SceneStateError):
        _state().permute_slots({"subject": "object"})
