from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.compiler import (
    CaptureContract,
    DenseWorkPlan,
    ExecutionMode,
    OutputContract,
    WorkPlanCache,
    build_dense_qstore_plan,
    build_dense_work_plan,
    execute_lowered_plan,
    lower_work_plan,
)
from mrun.engine.ane import ANEPagedEngine
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
        "request_ids": ("a", "b"),
        "metadata": {"z": 1, "a": "first", "engine_device": "cuda"},
    }
    values.update(overrides)
    return build_dense_work_plan(**values)


class _FakeStore:
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
                "vocab_size": 8,
            },
        }
    )

    def __init__(self) -> None:
        install_verified_test_identity(self)

    def has(self, name: str) -> bool:
        return name in {
            "L0.q",
            "L0.k",
            "L0.v",
            "L0.o",
            "L0.gate",
            "L0.up",
            "L0.down",
            "norm.final",
            "lm_head",
        }


class _FakeDenseEngine:
    backend = "dense-qstore-cuda"
    device = torch.device("cuda")
    name = "tiny"
    n_layer = 1
    max_seq_len = 16
    numerical_contract = "torch-batched-established"
    store = _FakeStore()
    spec = SimpleNamespace(name="tiny")

    def logits_batch(self, rows):
        outputs = []
        for index, row in enumerate(rows):
            base = torch.arange(len(row) * 8, dtype=torch.float32).reshape(len(row), 8)
            outputs.append(base + index * 100)
        return outputs


class _FakeCapturedDenseEngine(_FakeDenseEngine):
    subset_head_numerical_contract = "torch-batched-established+selected-head-fp32-v1"

    def execute_workplan_stateful(self, *_args, **_kwargs):
        raise AssertionError("the dense stateful lowerer must reject before execution")

    def capabilities(self):
        return SimpleNamespace(transactional_kv=True, speculative_blocks=True)

    supported_numerical_contracts = (
        _FakeDenseEngine.numerical_contract,
        subset_head_numerical_contract,
    )

    def __init__(self):
        self.capture_preparations = 0
        self.capture_replays = 0

    def selected_last_logits_batch(self, rows, token_ids):
        indices = torch.as_tensor(token_ids, dtype=torch.long)
        logits = self.logits_batch(rows)
        return torch.stack([row[-1].index_select(0, indices) for row in logits])

    def prepare_selected_last_cuda_graph(self, rows, token_ids, warmup=3):
        engine = self
        bound_rows = tuple(np.asarray(row, dtype=np.int64).copy() for row in rows)
        bound_tokens = tuple(int(value) for value in token_ids)
        self.capture_preparations += 1

        class _Executor:
            evidence = {
                "graph_backend": "fake-cuda-graph",
                "stable_addresses_verified": True,
                "capture_warmups": warmup,
            }

            def execute(self):
                engine.capture_replays += 1
                return engine.selected_last_logits_batch(bound_rows, bound_tokens)

        return _Executor()


def test_workplan_fingerprint_is_canonical_and_contract_sensitive():
    first = _plan(metadata={"z": 1, "a": "first"})
    second = _plan(metadata=(("a", "first"), ("z", 1)))
    changed = _plan(output_contract=OutputContract.FULL_LOGITS)

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed.fingerprint
    assert first.content_addressed
    assert not first.content_identity_verified
    assert json.loads(json.dumps(first.as_dict()))["shape"]["live_token_rows"] == 6


def test_legacy_v1_migration_is_narrowly_score_only() -> None:
    score_payload = _plan().as_dict()
    score_payload["schema_version"] = "mrun-dense-workplan-v1"
    migrated = DenseWorkPlan.from_dict(score_payload)

    assert migrated.schema_version == "mrun-dense-workplan-v3"
    assert dict(migrated.metadata)["legacy_schema_migrated_from"] == ("mrun-dense-workplan-v1")

    stateful_payload = dict(score_payload)
    stateful_payload["execution_mode"] = "decode"
    stateful_payload["kv_read_handles"] = ["kv:a", "kv:b"]
    stateful_payload["kv_write_handles"] = ["kv:a", "kv:b"]
    with pytest.raises(ValueError, match="legacy stateful"):
        DenseWorkPlan.from_dict(stateful_payload)


def test_content_identity_requires_actual_sha256_digests():
    descriptive_only = _plan(
        model_revision="source-abc",
        store_fingerprint="store-def",
    )

    assert not descriptive_only.content_identity_verified
    assert not lower_work_plan(descriptive_only, "cuda").content_identity_verified


def test_output_contracts_fail_closed_during_plan_validation():
    with pytest.raises(ValueError, match="requires required_output_rows"):
        _plan(output_contract=OutputContract.SELECTED_TOKEN_ROWS)
    with pytest.raises(ValueError, match="candidate IDs"):
        _plan(output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN)
    with pytest.raises(ValueError, match="at least two distinct"):
        _plan(
            output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            candidate_token_ids=((1, 1), (2, 3)),
        )


@pytest.mark.parametrize(
    ("output_contract", "output_fields", "message"),
    (
        (
            OutputContract.SELECTED_TOKEN_ROWS,
            {"required_output_rows": (0, 8)},
            "required output rows must be inside semantic token space",
        ),
        (
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            {"candidate_token_ids": ((0, 8), (1, 2))},
            "candidate token IDs must be inside semantic token space",
        ),
    ),
)
def test_workplan_direct_construction_rejects_output_ids_at_semantic_vocab_boundary(
    output_contract,
    output_fields,
    message,
):
    base = _plan(metadata={"input_token_limit": 8})

    with pytest.raises(ValueError, match=message):
        replace(base, output_contract=output_contract, **output_fields)


@pytest.mark.parametrize(
    ("output_contract", "output_fields"),
    (
        (
            OutputContract.SELECTED_TOKEN_ROWS,
            {"required_output_rows": (0, 7)},
        ),
        (
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            {"candidate_token_ids": ((0, 7), (1, 2))},
        ),
    ),
)
def test_workplan_direct_construction_requires_a_bound_semantic_output_domain(
    output_contract,
    output_fields,
):
    base = _plan()

    with pytest.raises(ValueError, match="requires input_token_limit metadata"):
        replace(base, output_contract=output_contract, **output_fields)


@pytest.mark.parametrize(
    ("output_contract", "field_name", "field_value", "message"),
    (
        (
            OutputContract.SELECTED_TOKEN_ROWS,
            "required_output_rows",
            [0, 8],
            "required output rows must be inside semantic token space",
        ),
        (
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            "candidate_token_ids",
            [[0, 8], [1, 2]],
            "candidate token IDs must be inside semantic token space",
        ),
    ),
)
def test_workplan_deserialization_rejects_output_ids_at_semantic_vocab_boundary(
    output_contract,
    field_name,
    field_value,
    message,
):
    payload = _plan(metadata={"input_token_limit": 8}).as_dict()
    payload["output_contract"] = output_contract.value
    payload[field_name] = field_value

    with pytest.raises(ValueError, match=message):
        DenseWorkPlan.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("output_contract", "output_fields", "mutated_field", "mutated_value"),
    (
        (
            OutputContract.SELECTED_TOKEN_ROWS,
            {"required_output_rows": (1, 5)},
            "required_output_rows",
            (1, 8),
        ),
        (
            OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            {"candidate_token_ids": ((1, 3), (4, 6))},
            "candidate_token_ids",
            ((1, 8), (4, 6)),
        ),
    ),
)
def test_runtime_rechecks_output_ids_against_the_engine_semantic_domain(
    output_contract,
    output_fields,
    mutated_field,
    mutated_value,
):
    engine = _FakeDenseEngine()
    engine.store = _FakeStore()
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    plan = build_dense_qstore_plan(
        engine,
        rows,
        output_contract=output_contract,
        **output_fields,
    )
    lowered = lower_work_plan(plan, "dense-qstore-cuda")

    # Simulate an in-process caller bypassing the frozen dataclass after lowering. Runtime
    # admission must still stop the corrupted row domain before dispatching the backend.
    object.__setattr__(plan, mutated_field, mutated_value)

    with pytest.raises(ValueError, match="runtime semantic token space"):
        execute_lowered_plan(engine, plan, lowered, rows)


def test_cuda_lowering_reports_capture_eligibility_without_serialized_execution_claim():
    plan = _plan(
        capture=CaptureContract(
            requested=True,
            static_shapes=True,
            stable_addresses=True,
            graph_safe=True,
        )
    )
    lowered = lower_work_plan(plan, "cuda-qstore")

    assert lowered.implementation_status == "eager-adapter"
    assert lowered.reported_fabric == "cuda"
    assert not lowered.capture_ready
    assert not lowered.capture_executed
    assert (
        "CUDA Graph runtime requires a loaded-store identity certificate"
        in lowered.capture_refusal_reasons
    )
    assert "cuda-graph-eager-replay-parity" in lowered.evidence_requirements
    assert "cuda-graph-capture-executed" in lowered.evidence_requirements
    assert [step.operation for step in lowered.steps][-1] == "last_token_full_vocabulary_head"


def test_cuda_lowering_rejects_stateful_dense_until_versioned_executor_is_integrated():
    engine = _FakeCapturedDenseEngine()
    plan = build_dense_qstore_plan(
        engine,
        [np.asarray([1, 2, 3])],
        execution_mode=ExecutionMode.DECODE,
        output_contract=OutputContract.FULL_LOGITS,
        kv_capacity=16,
        capture_requested=True,
        stable_addresses=True,
        graph_safe=True,
    )
    with pytest.raises(NotImplementedError, match="dense CUDA stateful WorkPlans"):
        lower_work_plan(plan, "cuda-qstore")


def test_rocm_and_coreml_lowerers_preserve_unmeasured_boundaries():
    rocm = lower_work_plan(_plan(), "rocm")
    assert rocm.implementation_status == "schedule-only"
    assert rocm.reported_fabric == "rocm-unmeasured"
    assert not rocm.placement_verified

    coreml_plan = _plan(
        activation_dtype="fp16",
        weight_dtype="fp16",
        accumulator_dtype="fp32-class",
        batch_bucket=2,
    )
    coreml = lower_work_plan(coreml_plan, "ane")
    assert coreml.backend == "coreml"
    assert coreml.reported_fabric == "coreml-unverified"
    assert not coreml.placement_verified
    assert "mlcomputeplan-placement" in coreml.evidence_requirements


def test_lowered_plan_cache_is_lru_and_byte_accounted():
    first = lower_work_plan(_plan(), "cuda")
    second = lower_work_plan(
        _plan(output_contract=OutputContract.FULL_LOGITS),
        "cuda",
    )
    cache = WorkPlanCache(max_entries=1, max_bytes=1_000_000)

    assert cache.put(first)
    assert cache.get(first.executable_key) == first
    assert cache.put(second)
    assert cache.get(first.executable_key) is None
    assert cache.get(second.executable_key) == second
    stats = cache.stats()
    assert stats.entries == 1
    assert stats.bytes == second.estimated_bytes
    assert stats.hits == 2
    assert stats.misses == 1
    assert stats.evictions == 1


def test_dense_qstore_adapter_and_candidate_execution():
    engine = _FakeDenseEngine()
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    plan = build_dense_qstore_plan(
        engine,
        rows,
        output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        candidate_token_ids=((1, 3, 2), (4, 6, 5)),
    )
    lowered = lower_work_plan(plan, "dense-qstore-cuda")
    result = execute_lowered_plan(engine, plan, lowered, rows)

    assert plan.model_revision == engine.store.source_checkpoint_sha256
    assert plan.store_fingerprint == engine.store.derived_store_sha256
    assert plan.content_identity_verified
    assert plan.shape.batch_bucket == 2
    assert "L0.q" in plan.page_sequence and "lm_head" in plan.page_sequence
    assert result.outputs[0]["winner_token_id"] == 3
    assert result.outputs[0]["runner_up_token_id"] == 2
    assert result.outputs[0]["margin"] == pytest.approx(1.0)
    assert result.outputs[1]["winner_token_id"] == 6
    assert result.evidence["graph_replay"] is False
    assert result.evidence["runtime_identity_bound"] is True
    assert result.evidence["runtime_configuration_bound"] is True
    assert result.evidence["runtime_activation_dtype"] == "bf16"
    assert result.evidence["runtime_weight_dtype"] == "int8"
    assert result.evidence["runtime_fabric"] == "cuda"


def test_lowered_execution_refuses_runtime_precision_weight_and_device_drift():
    engine = _FakeDenseEngine()
    engine.store = _FakeStore()
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    plan = build_dense_qstore_plan(engine, rows)
    lowered = lower_work_plan(plan, "cuda")

    engine.store.compute_dtype = torch.float16
    with pytest.raises(RuntimeError, match="activation dtype"):
        execute_lowered_plan(engine, plan, lowered, rows)

    engine.store.compute_dtype = torch.bfloat16
    engine.store.man = {**_FakeStore.man, "dtype": "int4"}
    with pytest.raises(RuntimeError, match="semantic manifest digest"):
        execute_lowered_plan(engine, plan, lowered, rows)

    engine.store.man = dict(_FakeStore.man)
    engine.device = torch.device("cpu")
    with pytest.raises(RuntimeError, match="runtime device"):
        execute_lowered_plan(engine, plan, lowered, rows)


def test_lowered_execution_rejects_a_different_runtime_store_identity():
    engine = _FakeDenseEngine()
    engine.store = _FakeStore()
    engine.store.man = {
        **_FakeStore.man,
        "derived": {"derived_store_sha256": "c" * 64},
    }
    engine.store.derived_store_sha256 = "c" * 64
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    plan = _plan()
    lowered = lower_work_plan(plan, "cuda")

    with pytest.raises(RuntimeError, match="semantic"):
        execute_lowered_plan(engine, plan, lowered, rows)


def test_legacy_store_identity_is_explicit_and_non_promotable():
    engine = _FakeDenseEngine()
    engine.store = _FakeStore()
    engine.store.man = {
        "model_name": "tiny",
        "dtype": "int8",
        "config": {
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "vocab_size": 8,
        },
    }
    engine.store.content_identity_verified = False
    engine.store.identity_status = "legacy-unverified"
    engine.store.source_checkpoint_sha256 = None
    engine.store.derived_store_sha256 = None
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    plan = build_dense_qstore_plan(engine, rows)
    lowered = lower_work_plan(plan, "cuda")

    assert not plan.content_identity_verified
    assert not lowered.content_identity_verified
    assert dict(plan.metadata)["source_identity_status"] == "legacy-unverified"
    assert dict(plan.metadata)["store_identity_status"] == "legacy-unverified"
    assert "content-addressed-model-and-store" in lowered.evidence_requirements


def test_declared_legacy_digests_are_diagnostic_not_verified_identity():
    engine = _FakeDenseEngine()
    engine.store = _FakeStore()
    engine.store.content_identity_verified = False
    engine.store.identity_status = "declared-content-only-unverified"
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]

    plan = build_dense_qstore_plan(engine, rows)

    assert plan.model_revision == (
        f"declared-unverified:{engine.store.man['source']['source_checkpoint_sha256']}"
    )
    assert plan.store_fingerprint == (
        f"declared-unverified:{engine.store.man['derived']['derived_store_sha256']}"
    )
    assert not plan.content_identity_verified
    assert dict(plan.metadata)["source_identity_status"] == "declared-unverified"
    assert dict(plan.metadata)["store_identity_status"] == "declared-unverified"


def test_selected_rows_and_numerical_contract_gate():
    engine = _FakeDenseEngine()
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    plan = build_dense_qstore_plan(
        engine,
        rows,
        output_contract=OutputContract.SELECTED_TOKEN_ROWS,
        required_output_rows=(1, 5),
    )
    lowered = lower_work_plan(plan, "cuda")
    assert "cuda-graph-eager-replay-parity" not in lowered.evidence_requirements
    assert "cuda-graph-capture-executed" not in lowered.evidence_requirements
    result = execute_lowered_plan(engine, plan, lowered, rows)
    assert result.outputs.shape == (2, 2)
    assert torch.equal(result.outputs[0], torch.tensor([17.0, 21.0]))

    engine.numerical_contract = "different-contract"
    with pytest.raises(RuntimeError, match="does not match"):
        execute_lowered_plan(engine, plan, lowered, rows)


def test_captured_selected_rows_execute_with_runtime_capture_evidence():
    engine = _FakeCapturedDenseEngine()
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    plan = build_dense_qstore_plan(
        engine,
        rows,
        output_contract=OutputContract.SELECTED_TOKEN_ROWS,
        required_output_rows=(1, 5),
        capture_requested=True,
        stable_addresses=True,
        graph_safe=True,
    )
    lowered = lower_work_plan(plan, "cuda-qstore")

    result = execute_lowered_plan(engine, plan, lowered, rows)

    assert torch.equal(
        result.outputs,
        torch.tensor([[17.0, 21.0], [117.0, 121.0]]),
    )
    assert engine.capture_preparations == 1
    assert engine.capture_replays == 1
    assert result.evidence["graph_replay"] is True
    assert result.evidence["capture_requested"] is True
    assert result.evidence["capture_ready"] is True
    assert result.evidence["capture_executed"] is True
    assert result.evidence["runtime_implementation_status"] == "cuda-graph"
    assert result.evidence["capture_metadata"]["stable_addresses_verified"] is True


def test_schedule_only_backend_cannot_execute():
    engine = _FakeDenseEngine()
    rows = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    plan = build_dense_qstore_plan(engine, rows)
    schedule = lower_work_plan(plan, "rocm")
    with pytest.raises(RuntimeError, match="schedule-only"):
        execute_lowered_plan(engine, plan, schedule, rows)


def test_coreml_evidence_never_infers_ane_placement():
    engine = object.__new__(ANEPagedEngine)
    engine._requested_backend_alias = "coreml"
    engine._compiled_shapes = {(4, 8), (1, 3)}
    engine._last_fallback = False

    evidence = engine.execution_evidence()
    assert evidence["reported_fabric"] == "coreml-unverified"
    # The request is reported, never inferred. ALL remains the default: MLComputePlan measured
    # it placing 1296/1327 ops on the GPU, but CPU_AND_NE failed the parity gate at 0/8 argmax.
    # A bare instance has no coremltools bound, so it must say so rather than guess.
    assert evidence["compute_units_request"] == "unavailable"
    # placement is still not verified per-run — that is the point of this test
    assert evidence["placement_verified"] is False
    assert evidence["preferred_devices"] is None
    assert evidence["compiled_shapes"] == [[1, 3], [4, 8]]


def test_coreml_backend_alias_records_the_requested_name(monkeypatch):
    import mrun.engine.ane as ane_module
    from mrun.engine import open_engine

    class FakeCoreMLEngine:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs
            self._requested_backend_alias = "ane"

    monkeypatch.setattr(ane_module, "ANEPagedEngine", FakeCoreMLEngine)
    engine = open_engine("tiny", backend="coreml", marker=17)
    assert engine.model_name == "tiny"
    assert engine.kwargs == {"marker": 17}
    assert engine._requested_backend_alias == "coreml"
