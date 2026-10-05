from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mrun.compiler import (
    CompilationArtifactStore,
    CompilationBundle,
    CostAssumptions,
    DenseWorkPlan,
    EvidenceRecord,
    LoweredWorkPlan,
    OutputContract,
    benchmark_eager_plan,
    benchmark_output_slicing,
    build_dense_work_plan,
    build_paged_qstore_plan,
    compile_work_plan,
    evaluate_promotion,
    evidence_payload_sha256,
    execute_lowered_plan,
    verify_output_pushdown_parity,
)
from mrun.testing.qstore_identity import (
    install_verified_test_identity,
    verified_test_manifest,
)


class _PagedStore:
    compute_dtype = torch.float32
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
            "blocks": {
                "embed": {
                    "kind": "qrow",
                    "shape": [8, 4],
                    "w_off": 0,
                    "w_len": 32,
                    "s_off": 0,
                    "s_len": 32,
                },
                "L0.q": {
                    "kind": "qrow",
                    "shape": [4, 4],
                    "w_off": 32,
                    "w_len": 16,
                    "s_off": 32,
                    "s_len": 16,
                },
                "norm.final": {
                    "kind": "fp32",
                    "shape": [4],
                    "e_off": 0,
                    "e_len": 16,
                },
                "lm_head": {"alias": "embed"},
            },
        }
    )

    def __init__(self) -> None:
        install_verified_test_identity(self)

    def has(self, name: str) -> bool:
        return name in self.man["blocks"]


class _PagedEngine:
    backend = "paged"
    device = torch.device("cpu")
    name = "tiny"
    n_layer = 1
    hidden = 4
    inter = 8
    numerical_contract = "paged-qstore-established"
    store = _PagedStore()
    spec = SimpleNamespace(name="tiny")
    working_set_mb = 0.125

    def logits_batch(self, rows):
        return [
            torch.arange(len(row) * 8, dtype=torch.float32).reshape(len(row), 8) + index * 100
            for index, row in enumerate(rows)
        ]

    def hidden_states_batch(self, rows):
        return [
            [
                torch.full((len(row), 4), float(index)),
                torch.full((len(row), 4), float(index + 1)),
            ]
            for index, row in enumerate(rows)
        ]


class _PushdownPagedEngine(_PagedEngine):
    last_head_numerical_contract = "paged-qstore-last-head-fp32-v1"
    supported_numerical_contracts = (
        _PagedEngine.numerical_contract,
        last_head_numerical_contract,
    )

    def last_logits_batch(self, rows):
        result = torch.stack([row[-1] for row in self.logits_batch(rows)])
        result[0, 0] += 1e-6
        return result


def _rows():
    return [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]


def _paged_plan(**kwargs):
    return build_paged_qstore_plan(_PagedEngine(), _rows(), **kwargs)


def _compile(plan, backend="paged", **kwargs):
    return compile_work_plan(
        plan,
        backend,
        manifest=_PagedStore.man,
        **kwargs,
    )


def test_typed_plan_and_lowered_round_trips_are_fingerprint_stable():
    plan = _paged_plan()
    restored = DenseWorkPlan.from_json(plan.to_json(indent=2))
    lowered = _compile(plan).lowered
    restored_lowered = LoweredWorkPlan.from_dict(lowered.as_dict())

    assert restored == plan
    assert restored.fingerprint == plan.fingerprint
    assert restored_lowered == lowered
    assert lowered.backend == "paged-qstore"
    assert lowered.implementation_status == "eager-adapter"
    assert lowered.reported_fabric == "cpu"


def test_bundle_rejects_a_forged_lowered_content_identity_verdict():
    plan = build_paged_qstore_plan(
        _PagedEngine(),
        _rows(),
    )
    legacy_payload = plan.as_dict()
    legacy_payload["model_revision"] = "unversioned:tiny"
    legacy_payload["store_fingerprint"] = "unfingerprinted:tiny"
    legacy_payload["content_identity_verified"] = False
    legacy_plan = DenseWorkPlan.from_dict(legacy_payload)
    lowered = compile_work_plan(legacy_plan, "paged").lowered
    forged_payload = lowered.as_dict()
    forged_payload["content_identity_verified"] = True
    forged = LoweredWorkPlan.from_dict(forged_payload)

    with pytest.raises(ValueError, match="content-identity verdict"):
        CompilationBundle(plan=legacy_plan, lowered=forged)


def test_executable_contract_can_match_while_concrete_lowered_cache_keys_remain_distinct():
    first = build_paged_qstore_plan(
        _PagedEngine(),
        _rows(),
        request_ids=("first-a", "first-b"),
    )
    second = build_paged_qstore_plan(
        _PagedEngine(),
        _rows(),
        request_ids=("second-a", "second-b"),
    )

    assert first.fingerprint != second.fingerprint
    assert first.executable_contract_fingerprint == second.executable_contract_fingerprint
    assert _compile(first).lowered.executable_key != _compile(second).lowered.executable_key


def test_paged_adapter_executes_candidate_loss_and_hidden_contracts():
    engine = _PagedEngine()
    rows = _rows()
    candidate = build_paged_qstore_plan(
        engine,
        rows,
        output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        candidate_token_ids=((1, 3, 2), (4, 6, 5)),
    )
    candidate_result = execute_lowered_plan(
        engine,
        candidate,
        _compile(candidate).lowered,
        rows,
    )
    assert candidate_result.outputs[0]["winner_token_id"] == 3
    assert candidate_result.outputs[1]["winner_token_id"] == 6

    loss_plan = build_paged_qstore_plan(engine, rows, output_contract=OutputContract.LOSS_ONLY)
    loss = execute_lowered_plan(
        engine,
        loss_plan,
        _compile(loss_plan).lowered,
        rows,
    ).outputs
    expected = (
        torch.stack(
            [
                F.cross_entropy(logits[:-1], torch.as_tensor(row[1:]), reduction="sum")
                for logits, row in zip(engine.logits_batch(rows), rows, strict=True)
            ]
        ).sum()
        / 4
    )
    assert torch.equal(loss, expected)

    hidden_plan = build_paged_qstore_plan(
        engine,
        rows,
        output_contract=OutputContract.HIDDEN_STATE_ONLY,
    )
    hidden = execute_lowered_plan(
        engine,
        hidden_plan,
        _compile(hidden_plan).lowered,
        rows,
    ).outputs
    assert hidden.shape == (2, 3, 4)
    assert torch.equal(hidden[1], torch.full((3, 4), 2.0))


def test_bundle_memory_cost_and_artifact_round_trip(tmp_path):
    plan = _paged_plan()
    assumptions = CostAssumptions(
        peak_compute_ops_per_s=1e12,
        memory_bandwidth_bytes_per_s=100e9,
        launch_overhead_s=1e-6,
        boundary_overhead_s=2e-6,
        label="test-only",
    )
    bundle = compile_work_plan(
        plan,
        "paged-qstore",
        manifest=_PagedStore.man,
        cost_assumptions=assumptions,
    )
    assert bundle.memory is not None
    assert bundle.memory.durable_store_bytes == 112
    assert bundle.memory.scheduled_page_bytes == 112
    assert bundle.memory.peak_promoted_weight_bytes == 128
    assert bundle.memory.kv_allocated_bytes == 0
    assert bundle.cost is not None
    assert bundle.cost.operation_count > 0
    assert bundle.cost.estimated_total_s > 0
    assert bundle.work_floor is not None
    assert not bundle.work_floor.rewritten.certificate.head_demand.complete
    assert CompilationBundle.from_json(bundle.to_json()) == bundle

    store = CompilationArtifactStore(tmp_path)
    record = store.save(bundle)
    assert record.path.exists()
    assert store.load(bundle.fingerprint) == bundle

    wrapper = json.loads(record.path.read_text())
    wrapper["bundle"]["plan"]["model_name"] = "tampered"
    record.path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.load(bundle.fingerprint)


def test_evidence_gate_never_promotes_missing_or_legacy_claims():
    bundle = _compile(_paged_plan())
    absent = evaluate_promotion(bundle, target="candidate")
    assert not absent.promotable
    assert any("missing evidence" in blocker for blocker in absent.blockers)

    records = [
        EvidenceRecord(
            requirement,
            True,
            "test-suite",
            subject_fingerprint=bundle.fingerprint,
            result_fingerprint=evidence_payload_sha256({"requirement": requirement}),
        )
        for requirement in bundle.lowered.evidence_requirements
    ]
    candidate = evaluate_promotion(bundle, records, target="candidate")
    assert candidate.promotable
    production = evaluate_promotion(bundle, records, target="production")
    assert not production.promotable
    assert "missing evidence: compilation-artifact-checksum" in production.blockers

    legacy_engine = _PagedEngine()
    legacy_engine.store = _PagedStore()
    legacy_engine.store.man = {**_PagedStore.man, "source": {}, "derived": {}}
    legacy_engine.store.content_identity_verified = False
    legacy_engine.store.identity_status = "legacy-unverified"
    legacy_engine.store.source_checkpoint_sha256 = None
    legacy_engine.store.derived_store_sha256 = None
    legacy = compile_work_plan(build_paged_qstore_plan(legacy_engine, _rows()), "paged")
    report = evaluate_promotion(legacy, records, target="reference")
    assert not report.promotable
    assert "model/store content identity is not verified" in report.blockers


def test_sha_shaped_ghost_plan_is_content_addressed_but_never_verified_or_promotable():
    ghost = build_dense_work_plan(
        model_name="ghost",
        model_revision="a" * 64,
        store_fingerprint="b" * 64,
        batch_size=1,
        sequence_length=1,
    )
    bundle = compile_work_plan(ghost, "paged")

    assert ghost.content_addressed
    assert not ghost.content_identity_verified
    assert not bundle.lowered.content_identity_verified
    report = evaluate_promotion(bundle, target="reference")
    assert not report.promotable
    assert "model/store content identity is not verified" in report.blockers


def test_verified_plan_requires_its_semantic_manifest_and_bound_evidence():
    plan = _paged_plan()
    with pytest.raises(ValueError, match="requires its QStore manifest"):
        compile_work_plan(plan, "paged")

    bundle = _compile(plan)
    unbound = [
        EvidenceRecord(requirement, True, "trust-me")
        for requirement in bundle.lowered.evidence_requirements
    ]
    report = evaluate_promotion(bundle, unbound, target="candidate")
    assert not report.promotable
    assert any("unbound evidence" in blocker for blocker in report.blockers)


def test_eager_benchmark_proves_exact_direct_adapter_parity():
    engine = _PagedEngine()
    plan = _paged_plan()
    lowered = _compile(plan).lowered
    benchmark = benchmark_eager_plan(
        engine,
        plan,
        lowered,
        _rows(),
        warmup=0,
        trials=2,
    )

    assert benchmark.parity.exact
    assert benchmark.parity.allclose
    assert benchmark.parity.compared_values == 16
    assert benchmark.reported_fabric == "cpu"
    assert benchmark.working_set_mb == pytest.approx(0.125)


def test_output_pushdown_has_a_distinct_contract_and_full_head_gate():
    engine = _PushdownPagedEngine()
    plan = build_paged_qstore_plan(engine, _rows())
    bundle = _compile(plan)

    assert plan.numerical_contract == engine.last_head_numerical_contract
    assert dict(plan.metadata)["output_pushdown"] is True
    assert "full-head-output-contract-parity" in bundle.lowered.evidence_requirements
    parity = verify_output_pushdown_parity(engine, plan, _rows())
    assert parity.allclose
    assert not parity.exact
    assert parity.max_abs_error > 0


def test_output_slicing_benchmark_uses_an_unsliced_equal_output_baseline():
    engine = _PushdownPagedEngine()
    plan = build_paged_qstore_plan(engine, _rows())
    lowered = _compile(plan).lowered

    result = benchmark_output_slicing(
        engine,
        plan,
        lowered,
        _rows(),
        warmup=0,
        trials=2,
    )

    assert result.output_contract == OutputContract.LAST_TOKEN_LOGITS.value
    assert len(result.baseline_samples_ms) == 2
    assert len(result.sliced_samples_ms) == 2
    assert len(result.paired_speedups) == 2
    assert result.parity.allclose
    assert not result.parity.exact
    assert result.max_dequant_block_mb == pytest.approx(0.125)
    assert "working_set_mb" not in result.as_dict()


def test_intn_manifest_kinds_plan_without_a_compute_dtype_attribute():
    engine = _PagedEngine()
    manifest = copy.deepcopy(_PagedStore.man)
    manifest["dtype"] = "int4"
    manifest["blocks"]["embed"]["kind"] = "qrow4"
    manifest["blocks"]["L0.q"]["kind"] = "qrow4"
    engine.store = SimpleNamespace(
        man=manifest,
        has=lambda name: name in manifest["blocks"],
    )

    plan = build_paged_qstore_plan(engine, _rows())
    bundle = compile_work_plan(plan, "paged", manifest=manifest)
    assert plan.precision.weight_dtype == "int4"
    assert plan.precision.activation_dtype == "fp32"
    assert bundle.memory is not None
    assert bundle.memory.peak_promoted_weight_bytes == 128
    assert bundle.cost is None


@pytest.mark.parametrize(
    ("dtype", "kind", "row_bytes", "expected_weight_bytes"),
    [
        ("int2", "qrow2", 2, 16),
        ("int3", "qrow3", 3, 24),
        ("int4", "qrow4", 4, 32),
    ],
)
def test_packed_row_manifests_account_for_implicit_weight_and_scale_lengths(
    dtype,
    kind,
    row_bytes,
    expected_weight_bytes,
):
    engine = _PagedEngine()
    manifest = copy.deepcopy(_PagedStore.man)
    manifest["dtype"] = dtype
    manifest["blocks"] = {
        "embed": {
            "kind": kind,
            "shape": [8, 4],
            "w_off": 0,
            "row_bytes": row_bytes,
            "s_off": 0,
            "n_groups": 2,
            "group_size": 2,
        },
        "L0.q": {
            "kind": kind,
            "shape": [4, 4],
            "w_off": expected_weight_bytes,
            "row_bytes": row_bytes,
            "s_off": 64,
            "n_groups": 2,
            "group_size": 2,
        },
        "norm.final": {
            "kind": "fp32",
            "shape": [4],
            "e_off": 0,
            "e_len": 16,
        },
        "lm_head": {"alias": "embed"},
    }
    engine.store = SimpleNamespace(
        man=manifest,
        has=lambda name: name in manifest["blocks"],
    )

    plan = build_paged_qstore_plan(engine, _rows())
    memory = compile_work_plan(plan, "paged", manifest=manifest).memory

    assert memory is not None
    # Physical embed/head bytes are deduplicated even though both are logical uses:
    # embed weights + q weights + two 8-byte-per-row scale regions + fp32 norm.
    assert memory.durable_store_bytes == (
        expected_weight_bytes + 4 * row_bytes + 8 * 2 * 4 + 4 * 2 * 4 + 16
    )


def test_candidate_cost_uses_batch_times_union_of_head_rows():
    engine = _PagedEngine()
    engine.candidate_logits_batch = lambda rows, candidates: None
    plan = build_paged_qstore_plan(
        engine,
        _rows(),
        output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        candidate_token_ids=((1, 2), (3, 4)),
    )
    assumptions = CostAssumptions(1e12, 1e11)
    memory = compile_work_plan(plan, "paged", manifest=_PagedStore.man).memory
    assert memory is not None

    # Isolate the head by comparing against the same hidden-state-only body.
    candidate = compile_work_plan(
        plan,
        "paged",
        manifest=_PagedStore.man,
        cost_assumptions=assumptions,
    ).cost
    hidden_plan = build_paged_qstore_plan(
        engine,
        _rows(),
        output_contract=OutputContract.HIDDEN_STATE_ONLY,
    )
    hidden = compile_work_plan(
        hidden_plan,
        "paged",
        manifest=_PagedStore.man,
        cost_assumptions=assumptions,
    ).cost
    assert candidate is not None and hidden is not None
    assert candidate.operation_count - hidden.operation_count == 2 * 2 * 4 * 4


def test_qstore_adapter_rejects_output_rows_outside_the_manifest_vocab():
    with pytest.raises(ValueError, match=r"semantic token space \[0, 8\)"):
        build_paged_qstore_plan(
            _PagedEngine(),
            _rows(),
            output_contract=OutputContract.SELECTED_TOKEN_ROWS,
            required_output_rows=(8,),
        )
