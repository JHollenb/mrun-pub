from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.compiler import (
    LOWERING_ABI,
    DispatchBinding,
    LoweredWorkTemplate,
    OutputContract,
    ScheduleStep,
    WorkTemplate,
    WorkTemplateCache,
    build_dense_qstore_plan,
    build_dense_work_plan,
    decompose_work_plan,
    execute_lowered_template,
    lower_work_template,
)
from mrun.testing.qstore_identity import (
    install_verified_test_identity,
    verified_test_manifest,
)


def _plan(**overrides):
    values = {
        "model_name": "tiny",
        "model_revision": "a" * 64,
        "store_fingerprint": "b" * 64,
        "batch_size": 2,
        "sequence_length": 3,
        "batch_bucket": 4,
        "activation_dtype": "bf16",
        "weight_dtype": "int8",
        "accumulator_dtype": "fp32",
        "request_ids": ("first-a", "first-b"),
        "request_slots": (5, 7),
        "output_contract": OutputContract.SELECTED_TOKEN_ROWS,
        "required_output_rows": (1, 5),
        "metadata": {"engine_device": "cuda", "input_token_limit": 8},
    }
    values.update(overrides)
    return build_dense_work_plan(**values)


def test_template_binding_decomposition_is_lossless_canonical_and_serializable():
    first = _plan()
    second = _plan(
        request_ids=("second-a", "second-b"),
        request_slots=(11, 13),
        required_output_rows=(2, 6),
    )

    first_template, first_binding = decompose_work_plan(first)
    second_template, second_binding = decompose_work_plan(second)

    assert first.fingerprint != second.fingerprint
    assert first_template == second_template
    assert first_template.fingerprint == first.executable_contract_fingerprint
    assert first_template.fingerprint == second.executable_contract_fingerprint
    assert first_binding.fingerprint != second_binding.fingerprint
    assert first_template.bind(first_binding) == first
    assert first_template.bind(second_binding) == second
    assert WorkTemplate.from_json(first_template.to_json(indent=2)) == first_template
    assert DispatchBinding.from_json(first_binding.to_json(indent=2)) == first_binding


def test_request_metadata_moves_to_binding_without_fragmenting_or_leaking_template_cache():
    structural = {"engine_device": "cuda", "input_token_limit": 8}
    first = _plan(
        metadata={
            **structural,
            "owner": "alice",
            "request_trace_id": "secret-first",
        }
    )
    second = _plan(
        request_ids=("second-a", "second-b"),
        metadata={
            **structural,
            "owner": "bob",
            "request_trace_id": "secret-second",
        },
    )

    first_template, first_binding = decompose_work_plan(first)
    second_template, second_binding = decompose_work_plan(second)

    assert first_template == second_template
    assert first_template.fingerprint == second_template.fingerprint
    assert dict(first_template.metadata) == structural
    assert dict(first_binding.dispatch_metadata) == {
        "owner": "alice",
        "request_trace_id": "secret-first",
    }
    assert dict(second_binding.dispatch_metadata) == {
        "owner": "bob",
        "request_trace_id": "secret-second",
    }
    assert first_template.bind(first_binding) == first
    assert second_template.bind(second_binding) == second
    assert DispatchBinding.from_json(first_binding.to_json()) == first_binding
    lowered_json = lower_work_template(first_template, "cuda").to_json()
    assert "alice" not in lowered_json
    assert "secret-first" not in lowered_json


def test_template_and_binding_metadata_schemas_reject_wrong_partition():
    template, binding = decompose_work_plan(_plan())
    forged_template = template.as_dict()
    forged_template["metadata"]["request_trace_id"] = "must-not-be-cached"
    with pytest.raises(ValueError, match="non-reusable keys"):
        WorkTemplate.from_dict(forged_template)

    with pytest.raises(ValueError, match="structural template keys"):
        replace(binding, dispatch_metadata=(("engine_device", "cpu"),))


def test_binding_rejects_wrong_template_shape_and_semantic_token_domain():
    template, binding = decompose_work_plan(_plan())
    other_template, _ = decompose_work_plan(_plan(required_output_rows=(1, 3, 5)))

    with pytest.raises(ValueError, match="different WorkTemplate"):
        template.bind(replace(binding, template_fingerprint=other_template.fingerprint))
    with pytest.raises(ValueError, match="cardinality"):
        template.bind(replace(binding, required_output_rows=(1,)))
    with pytest.raises(ValueError, match="semantic token space"):
        template.bind(replace(binding, required_output_rows=(1, 8)))


def test_candidate_union_cardinality_is_part_of_the_reusable_template_identity():
    shared = {
        "output_contract": OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        "required_output_rows": (),
        "metadata": {"engine_device": "cuda", "input_token_limit": 16},
    }
    overlapping, _ = decompose_work_plan(_plan(candidate_token_ids=((1, 2), (2, 3)), **shared))
    disjoint, _ = decompose_work_plan(_plan(candidate_token_ids=((4, 5), (6, 7)), **shared))

    assert overlapping.candidate_row_counts == disjoint.candidate_row_counts == (2, 2)
    assert overlapping.candidate_union_count == 3
    assert disjoint.candidate_union_count == 4
    assert overlapping.fingerprint != disjoint.fingerprint


@pytest.mark.parametrize(
    "plan",
    (
        lambda: _plan(
            metadata={
                "engine_device": "cuda",
                "input_token_limit": 8,
                "logical_head_access": "rows",
                "logical_head_row_count": 3,
            }
        ),
        lambda: _plan(
            output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            required_output_rows=(),
            candidate_token_ids=((1, 2), (2, 3)),
            metadata={
                "engine_device": "cuda",
                "input_token_limit": 8,
                "logical_head_access": "rows",
                "logical_head_row_count": 4,
            },
        ),
        lambda: _plan(
            output_contract=OutputContract.FULL_LOGITS,
            required_output_rows=(),
            metadata={
                "configured_output_row_count": 8,
                "engine_device": "cuda",
                "logical_head_access": "all",
                "logical_head_row_count": 7,
            },
        ),
        lambda: _plan(
            metadata={
                "engine_device": "cuda",
                "input_token_limit": 8,
                "logical_head_access": "all",
                "logical_head_row_count": 2,
            }
        ),
        lambda: _plan(
            metadata={
                "configured_output_row_count": 1,
                "engine_device": "cuda",
                "input_token_limit": 8,
                "logical_head_access": "rows",
                "logical_head_row_count": 2,
            }
        ),
    ),
)
def test_workplan_rejects_inconsistent_redundant_head_metadata(plan):
    with pytest.raises(ValueError, match="logical_head|output-contract cardinality"):
        plan()


def test_template_parser_cross_checks_selected_and_candidate_cardinality_metadata():
    selected, _ = decompose_work_plan(
        _plan(
            metadata={
                "engine_device": "cuda",
                "input_token_limit": 8,
                "logical_head_access": "rows",
                "logical_head_row_count": 2,
            }
        )
    )
    forged_selected = selected.as_dict()
    forged_selected["dispatch_shape"]["required_output_row_count"] = 3
    with pytest.raises(ValueError, match="logical_head_row_count"):
        WorkTemplate.from_dict(forged_selected)

    candidate, _ = decompose_work_plan(
        _plan(
            output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            required_output_rows=(),
            candidate_token_ids=((1, 2), (2, 3)),
            metadata={
                "engine_device": "cuda",
                "input_token_limit": 8,
                "logical_head_access": "rows",
                "logical_head_row_count": 3,
            },
        )
    )
    forged_candidate = candidate.as_dict()
    forged_candidate["dispatch_shape"]["candidate_union_count"] = 4
    with pytest.raises(ValueError, match="logical_head_row_count"):
        WorkTemplate.from_dict(forged_candidate)


def test_lowered_template_cache_reuses_one_schedule_without_retaining_request_provenance():
    first = _plan()
    second = _plan(
        request_ids=("second-a", "second-b"),
        request_slots=(11, 13),
        required_output_rows=(2, 6),
    )
    template, first_binding = decompose_work_plan(first)
    _, second_binding = decompose_work_plan(second)
    lowered_template = lower_work_template(template, "cuda")
    restored = LoweredWorkTemplate.from_json(lowered_template.to_json(indent=2))
    cache = WorkTemplateCache(max_entries=1, max_bytes=1_000_000)

    assert restored == lowered_template
    assert cache.put(lowered_template)
    first_bound = cache.get_bound(lowered_template.executable_key, template, first_binding)
    second_bound = cache.get_bound(lowered_template.executable_key, template, second_binding)
    assert first_bound is not None and second_bound is not None
    first_plan, first_lowered = first_bound
    second_plan, second_lowered = second_bound
    assert first_plan == first
    assert second_plan == second
    assert first_lowered.plan_fingerprint == first.fingerprint
    assert second_lowered.plan_fingerprint == second.fingerprint
    assert first_lowered.executable_key == second_lowered.executable_key
    serialized_schedule = json.dumps(lowered_template.as_dict(), sort_keys=True)
    assert "first-a" not in serialized_schedule
    assert "second-a" not in serialized_schedule
    assert cache.stats().hits == 2


def test_lowered_template_deserialization_and_binding_reject_forged_identity_verdicts():
    plan = _plan()
    template, _ = decompose_work_plan(plan)
    lowered = lower_work_template(template, "cuda")
    forged_key = lowered.as_dict()
    forged_key["executable_key"] = "0" * 64

    with pytest.raises(ValueError, match="lookup key"):
        LoweredWorkTemplate.from_dict(forged_key)
    with pytest.raises(ValueError, match="artifact digest"):
        replace(lowered, content_identity_verified=not plan.content_identity_verified)
    with pytest.raises(ValueError, match="artifact digest"):
        replace(lowered, reported_fabric="forged", artifact_sha256="")


def test_lowered_template_lookup_key_is_namespaced_by_the_lowering_abi():
    template, _ = decompose_work_plan(_plan())
    lowered = lower_work_template(template, "cuda")
    legacy_key = hashlib.sha256(f"{lowered.backend}:{template.fingerprint}".encode()).hexdigest()

    assert lowered.lowering_abi == LOWERING_ABI
    assert lowered.executable_key != legacy_key
    with pytest.raises(ValueError, match="ABI"):
        replace(lowered, lowering_abi="mrun-work-template-lowering-abi-v999")


def test_schedule_step_deep_freeze_preserves_arrays_objects_and_cache_artifacts():
    source = {
        "array": [1, {"leaf": [2, 3]}],
        "empty_array": [],
        "empty_object": {},
    }
    step = ScheduleStep("host", "deep_freeze", (("config", source),))
    canonical = step.as_dict()
    source["array"][1]["leaf"].append(4)
    source["new"] = "poison"

    assert step.as_dict() == canonical
    assert canonical["params"]["config"]["empty_array"] == []
    assert canonical["params"]["config"]["empty_object"] == {}
    assert ScheduleStep.from_dict(canonical) == step
    frozen_object = dict(step.params)["config"]
    with pytest.raises(FrozenInstanceError):
        frozen_object.items = ()
    frozen_array = dict(frozen_object.items)["array"]
    with pytest.raises(TypeError):
        frozen_array[0] = 99

    template, _ = decompose_work_plan(_plan())
    lowered = lower_work_template(template, "cuda")
    exported = lowered.as_dict()
    transformer = next(
        item for item in exported["steps"] if item["operation"] == "dense_transformer_region"
    )
    transformer["params"]["layout_ids"].append("poison")
    lowered.verify_integrity()
    assert "poison" not in lowered.to_json()
    cache = WorkTemplateCache(max_entries=1, max_bytes=1_000_000)
    assert cache.put(lowered)
    assert cache.get(lowered.executable_key) is lowered


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("model_name",), None),
        (("dispatch_shape", "required_output_row_count"), 2.0),
        (("dispatch_shape", "candidate_union_count"), True),
        (("shape", "batch_bucket"), 4.0),
        (("capture", "requested"), 1),
        (("precision", "weight_dtype"), None),
        (("page_sequence",), [None]),
        (("metadata",), {"non_finite": float("nan")}),
    ),
)
def test_work_template_parser_rejects_lossy_or_noncanonical_values(path, value):
    template, _ = decompose_work_plan(_plan())
    payload = template.as_dict()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((TypeError, ValueError)):
        WorkTemplate.from_json(json.dumps(payload))


def test_work_template_parser_rejects_unknown_duplicate_and_forged_derived_fields():
    template, _ = decompose_work_plan(_plan())
    unknown = template.as_dict()
    unknown["fingerprint"] = template.fingerprint
    with pytest.raises(ValueError, match="unknown fields"):
        WorkTemplate.from_dict(unknown)

    nested_unknown = template.as_dict()
    nested_unknown["shape"]["future_shape"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        WorkTemplate.from_dict(nested_unknown)

    forged_rows = template.as_dict()
    forged_rows["shape"]["live_token_rows"] += 1
    with pytest.raises(ValueError, match="live_token_rows"):
        WorkTemplate.from_dict(forged_rows)

    forged_capture = template.as_dict()
    forged_capture["capture"]["eligible"] = not forged_capture["capture"]["eligible"]
    with pytest.raises(ValueError, match="eligible"):
        WorkTemplate.from_dict(forged_capture)

    with pytest.raises(ValueError, match="duplicate key"):
        WorkTemplate.from_json('{"schema_version":"first","schema_version":"second"}')


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("template_fingerprint", None),
        ("request_ids", ["a", 2]),
        ("request_slots", [5, 7.0]),
        ("required_output_rows", [True, 5]),
        ("candidate_token_ids", None),
        ("schema_version", False),
    ),
)
def test_dispatch_binding_parser_rejects_lossy_values(field, value):
    _, binding = decompose_work_plan(_plan())
    payload = binding.as_dict()
    payload[field] = value
    with pytest.raises((TypeError, ValueError)):
        DispatchBinding.from_json(json.dumps(payload))


def test_dispatch_binding_parser_rejects_unknown_and_duplicate_fields():
    _, binding = decompose_work_plan(_plan())
    payload = binding.as_dict()
    payload["binding_fingerprint"] = binding.fingerprint
    with pytest.raises(ValueError, match="unknown fields"):
        DispatchBinding.from_dict(payload)
    with pytest.raises(ValueError, match="duplicate key"):
        DispatchBinding.from_json('{"schema_version":"first","schema_version":"second"}')


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("implementation_status", "forged-adapter"),
        ("reported_fabric", "forged-fabric"),
        ("placement_verified", False),
        ("content_identity_verified", True),
        ("capture_requested", True),
        ("capture_ready", True),
        ("capture_refusal_reasons", ["forged"]),
        ("evidence_requirements", ["forged"]),
    ),
)
def test_lowered_template_artifact_digest_binds_every_scalar_and_sequence(field, replacement):
    template, _ = decompose_work_plan(_plan())
    lowered = lower_work_template(template, "cuda")
    payload = lowered.as_dict()
    payload[field] = replacement

    with pytest.raises(ValueError, match="artifact digest"):
        LoweredWorkTemplate.from_dict(payload)


def test_lowered_template_digest_binds_steps_and_parser_is_strict():
    template, _ = decompose_work_plan(_plan())
    lowered = lower_work_template(template, "cuda")

    changed_step = lowered.as_dict()
    changed_step["steps"][0]["operation"] = "skip_identity_check"
    with pytest.raises(ValueError, match="artifact digest"):
        LoweredWorkTemplate.from_dict(changed_step)

    changed_param = lowered.as_dict()
    changed_param["steps"][0]["params"]["model_revision"] = "c" * 64
    with pytest.raises(ValueError, match="artifact digest"):
        LoweredWorkTemplate.from_dict(changed_param)

    unknown = lowered.as_dict()
    unknown["future_field"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        LoweredWorkTemplate.from_dict(unknown)

    unknown_step = lowered.as_dict()
    unknown_step["steps"][0]["future_field"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        LoweredWorkTemplate.from_dict(unknown_step)

    non_finite = lowered.as_dict()
    non_finite["steps"][0]["params"]["bad"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        LoweredWorkTemplate.from_dict(non_finite)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("backend", None),
        ("placement_verified", 1),
        ("capture_requested", "false"),
        ("capture_ready", None),
        ("capture_refusal_reasons", [None]),
        ("artifact_sha256", None),
    ),
)
def test_lowered_template_json_parser_rejects_lossy_values(field, value):
    template, _ = decompose_work_plan(_plan())
    payload = lower_work_template(template, "cuda").as_dict()
    payload[field] = value
    with pytest.raises((TypeError, ValueError)):
        LoweredWorkTemplate.from_json(json.dumps(payload))


def test_lowered_template_json_parser_rejects_duplicate_fields():
    with pytest.raises(ValueError, match="duplicate key"):
        LoweredWorkTemplate.from_json('{"schema_version":"first","schema_version":"second"}')


def test_forged_lowered_template_fails_before_cache_lookup_and_binding():
    plan = _plan()
    template, binding = decompose_work_plan(plan)

    forged_put = lower_work_template(template, "cuda")
    object.__setattr__(forged_put, "reported_fabric", "forged")
    cache = WorkTemplateCache(max_entries=1, max_bytes=1_000_000)
    with pytest.raises(ValueError, match="artifact digest"):
        cache.put(forged_put)

    cached = lower_work_template(template, "cuda")
    assert cache.put(cached)
    object.__setattr__(cached, "reported_fabric", "forged")
    with pytest.raises(ValueError, match="artifact digest"):
        cache.get(cached.executable_key)

    forged_bind = lower_work_template(template, "cuda")
    object.__setattr__(forged_bind, "reported_fabric", "forged")
    with pytest.raises(ValueError, match="artifact digest"):
        forged_bind.bind(template.bind(binding))


def test_stateful_lowered_template_contains_only_a_binding_contract_not_kv_handles():
    plan = build_dense_work_plan(
        model_name="tiny",
        model_revision="a" * 64,
        store_fingerprint="b" * 64,
        batch_size=2,
        sequence_length=1,
        execution_mode="decode",
        request_ids=("decode-a", "decode-b"),
        kv_read_handles=("secret-kv-a", "secret-kv-b"),
        kv_write_handles=("secret-kv-a", "secret-kv-b"),
        structured_operator_ids=("paged-transformer", "transactional-kv"),
        metadata={
            "engine_device": "cpu",
            "kv_capacity": 16,
            "kv_dtype": "fp32",
            "kv_num_layers": 1,
            "kv_num_heads": 1,
            "kv_head_dim": 4,
        },
    )
    template, _ = decompose_work_plan(plan)
    lowered = lower_work_template(template, "paged")
    state_step = next(step for step in lowered.steps if step.operation == "bind_versioned_kv_state")
    params = dict(state_step.params)

    assert params["binding_schema"] == "mrun-dispatch-binding-v1"
    assert params["handle_count"] == 2
    assert params["request_slot_count"] == 2
    assert "read_handles" not in params
    assert "write_handles" not in params
    assert "secret-kv-a" not in lowered.to_json()


class _Store:
    compute_dtype = torch.bfloat16
    man = verified_test_manifest(
        {
            "model_name": "tiny",
            "dtype": "int8",
            "config": {
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "intermediate_size": 8,
                "vocab_size": 8,
            },
        }
    )

    def __init__(self) -> None:
        install_verified_test_identity(self)

    def has(self, _name: str) -> bool:
        return True


class _Engine:
    backend = "dense-qstore-cuda"
    device = torch.device("cuda")
    name = "tiny"
    n_layer = 1
    max_seq_len = 16
    numerical_contract = "torch-batched-established"
    spec = SimpleNamespace(name="tiny")

    def __init__(self) -> None:
        self.store = _Store()

    def logits_batch(self, rows):
        return [
            torch.arange(len(row) * 8, dtype=torch.float32).reshape(len(row), 8) + index * 100
            for index, row in enumerate(rows)
        ]


def test_execute_lowered_template_reuses_the_artifact_but_preserves_dispatch_provenance():
    engine = _Engine()
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    first = build_dense_qstore_plan(
        engine,
        rows,
        request_ids=("first-a", "first-b"),
    )
    second = build_dense_qstore_plan(
        engine,
        rows,
        request_ids=("second-a", "second-b"),
    )
    template, first_binding = decompose_work_plan(first)
    second_template, second_binding = decompose_work_plan(second)
    lowered = lower_work_template(template, "dense-qstore-cuda")

    assert second_template == template
    first_result = execute_lowered_template(engine, template, first_binding, lowered, rows)
    second_result = execute_lowered_template(engine, template, second_binding, lowered, rows)

    assert torch.equal(first_result.outputs, second_result.outputs)
    assert first_result.plan_fingerprint == first.fingerprint
    assert second_result.plan_fingerprint == second.fingerprint
    assert first_result.executable_key == second_result.executable_key == lowered.executable_key
    assert first_result.evidence["work_template_fingerprint"] == template.fingerprint
    assert first_result.evidence["dispatch_binding_fingerprint"] == first_binding.fingerprint
    assert second_result.evidence["dispatch_binding_fingerprint"] == second_binding.fingerprint
    assert first_result.evidence["reused_lowered_template"] is True


def test_execute_lowered_template_verifies_artifact_before_materializing_dispatch():
    engine = _Engine()
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    plan = build_dense_qstore_plan(engine, rows)
    template, binding = decompose_work_plan(plan)
    lowered = lower_work_template(template, "cuda")
    object.__setattr__(lowered, "reported_fabric", "forged")
    invalid_binding = replace(binding, required_output_rows=(1,))

    with pytest.raises(ValueError, match="artifact digest"):
        execute_lowered_template(engine, template, invalid_binding, lowered, rows)
