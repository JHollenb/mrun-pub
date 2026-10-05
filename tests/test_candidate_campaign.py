from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.compiler import (
    Effect,
    EffectKind,
    analyze_work_floor,
    compile_graph,
)
from mrun.compiler.campaign import (
    CampaignInputBinding,
    CandidateCampaign,
    CandidateCampaignExecutionResult,
    CandidateReadout,
    CandidateReadoutResult,
    compile_candidate_campaign,
    execute_candidate_campaign,
    prepare_candidate_campaign,
)
from mrun.compiler.executable import candidate_outputs_from_values
from mrun.testing.qstore_identity import (
    install_verified_test_identity,
    verified_test_manifest,
)


def _qrow(shape, offset):
    rows, columns = shape
    return {
        "kind": "qrow",
        "shape": [rows, columns],
        "w_off": offset,
        "w_len": rows * columns,
        "s_off": offset * 4,
        "s_len": rows * 4,
    }


def _fp32(length, offset):
    return {
        "kind": "fp32",
        "shape": [length],
        "e_off": offset,
        "e_len": length * 4,
    }


def _tiny_qwen_manifest():
    return verified_test_manifest(
        {
            "model_name": "tiny-qwen",
            "arch": "qwen2",
            "dtype": "int8",
            "config": {
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "intermediate_size": 8,
                "vocab_size": 8,
                "rms_norm_eps": 1e-6,
            },
            "blocks": {
                "embed": _qrow((8, 4), 0),
                "L0.ln1": _fp32(4, 0),
                "L0.q": _qrow((4, 4), 32),
                "L0.q.bias": _fp32(4, 4),
                "L0.k": _qrow((2, 4), 48),
                "L0.k.bias": _fp32(2, 8),
                "L0.v": _qrow((2, 4), 56),
                "L0.v.bias": _fp32(2, 10),
                "L0.o": _qrow((4, 4), 64),
                "L0.ln2": _fp32(4, 12),
                "L0.gate": _qrow((8, 4), 80),
                "L0.up": _qrow((8, 4), 112),
                "L0.down": _qrow((4, 8), 144),
                "norm.final": _fp32(4, 16),
                "lm_head": {"alias": "embed"},
            },
        }
    )


class _CampaignStore:
    compute_dtype = torch.float32

    def __init__(self):
        self.man = _tiny_qwen_manifest()
        install_verified_test_identity(self)

    def has(self, name):
        return name in self.man["blocks"]


class _CampaignEngine:
    backend = "paged"
    device = torch.device("cpu")
    name = "tiny-qwen"
    n_layer = 1
    hidden = 4
    inter = 8
    numerical_contract = "paged-qstore-established"
    subset_head_numerical_contract = "paged-qstore-subset-head-fp32-v1"
    supported_numerical_contracts = (
        numerical_contract,
        subset_head_numerical_contract,
    )

    def __init__(self):
        self.store = _CampaignStore()
        self.spec = SimpleNamespace(name=self.name)
        self.selected_body_calls = 0
        self.selected_rows: list[tuple[int, ...]] = []

    def build_work_plan(self, ids_list, **kwargs):
        from mrun.compiler import build_paged_qstore_plan

        return build_paged_qstore_plan(self, ids_list, **kwargs)

    def selected_last_logits_batch(self, ids_list, token_ids):
        self.selected_body_calls += 1
        rows = tuple(int(token) for token in token_ids)
        self.selected_rows.append(rows)
        return torch.stack(
            [
                torch.as_tensor(
                    [float(np.asarray(ids).sum() + 2 * token) for token in rows],
                    dtype=torch.float32,
                )
                for ids in ids_list
            ]
        )


class _FakeCudaGraphExecutor:
    def __init__(self, engine, ids_list, token_ids):
        self.engine = engine
        self.ids_list = tuple(np.asarray(ids, dtype=np.int64).copy() for ids in ids_list)
        self.token_ids = tuple(int(value) for value in token_ids)
        self.closed = False
        self.evidence = {
            "graph_backend": "fake-cuda-graph",
            "stable_addresses_verified": True,
            "capture_warmups": 3,
        }

    def execute(self):
        if self.closed:
            raise RuntimeError("closed")
        self.engine.capture_replays += 1
        return torch.stack(
            [
                torch.as_tensor(
                    [float(np.asarray(ids).sum() + 2 * token) for token in self.token_ids],
                    dtype=torch.float32,
                )
                for ids in self.ids_list
            ]
        )

    def close(self):
        if not self.closed:
            self.closed = True
            self.engine.capture_closes += 1


class _DenseCampaignEngine(_CampaignEngine):
    backend = "dense-qstore-cuda"
    device = torch.device("cuda")
    max_seq_len = 16
    numerical_contract = "torch-batched-established"
    subset_head_numerical_contract = "torch-batched-established+selected-head-fp32-v1"
    supported_numerical_contracts = (
        numerical_contract,
        subset_head_numerical_contract,
    )

    def __init__(self):
        super().__init__()
        self.store.compute_dtype = torch.bfloat16
        self.capture_preparations = 0
        self.capture_replays = 0
        self.capture_closes = 0

    def build_work_plan(self, ids_list, **kwargs):
        from mrun.compiler import build_dense_qstore_plan

        return build_dense_qstore_plan(self, ids_list, **kwargs)

    def prepare_selected_last_cuda_graph(self, ids_list, token_ids, warmup=3):
        assert warmup == 3
        self.capture_preparations += 1
        return _FakeCudaGraphExecutor(self, ids_list, token_ids)


def _readouts():
    return (
        CandidateReadout("alpha", (5, 1, 3)),
        CandidateReadout("beta", (3, 7, 1)),
    )


def test_campaign_primary_runtime_selection_is_scoped_and_fail_closed():
    from mrun.cli import _resolve_campaign_runtime

    base = {
        "backend": "auto",
        "cuda_graph": None,
        "int2": False,
        "int3": False,
        "int4": False,
    }
    cuda = _resolve_campaign_runtime(
        SimpleNamespace(**base),
        cuda_available=True,
    )
    assert cuda["backend"] == "dense-qstore-cuda"
    assert cuda["capture_requested"] is False
    assert cuda["selection_mode"] == "promotion-gated-auto"
    assert cuda["selected_for_primary_runtime"] is False
    assert cuda["scope"] == "static-stateless-selected-vocabulary-campaign"

    cpu = _resolve_campaign_runtime(
        SimpleNamespace(**base),
        cuda_available=False,
    )
    assert cpu["backend"] == "paged"
    assert cpu["capture_requested"] is False

    packed = _resolve_campaign_runtime(
        SimpleNamespace(**{**base, "int4": True}),
        cuda_available=True,
    )
    assert packed["backend"] == "paged"
    assert packed["capture_requested"] is False

    explicit_dense = _resolve_campaign_runtime(
        SimpleNamespace(**{**base, "backend": "dense-qstore-cuda"}),
        cuda_available=True,
    )
    assert explicit_dense["capture_requested"] is False

    with pytest.raises(ValueError, match="requires a CUDA"):
        _resolve_campaign_runtime(
            SimpleNamespace(**{**base, "backend": "paged", "cuda_graph": True}),
            cuda_available=False,
        )


def _independent_results(engine, token_ids, readouts):
    results = {}
    for readout in readouts:
        logits = engine.selected_last_logits_batch(
            [token_ids],
            readout.candidate_token_ids,
        )[0]
        results[readout.query_id] = (
            tuple(float(value) for value in logits),
            candidate_outputs_from_values(
                [logits],
                (readout.candidate_token_ids,),
            )[0],
        )
    return results


def test_campaign_shares_one_body_and_preserves_query_local_top_two():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    readouts = _readouts()
    engine = _CampaignEngine()
    independent = _independent_results(engine, token_ids, readouts)
    assert engine.selected_body_calls == len(readouts)

    engine.selected_body_calls = 0
    engine.selected_rows.clear()
    campaign = compile_candidate_campaign(engine, token_ids, readouts)
    result = execute_candidate_campaign(engine, campaign, token_ids)

    assert campaign.union_token_ids == (5, 1, 3, 7)
    assert campaign.base_bundle.plan.required_output_rows == (5, 1, 3, 7)
    assert campaign.sharing.candidate_reference_count == 6
    assert campaign.sharing.union_candidate_count == 4
    assert campaign.sharing.eliminated_body_evaluations == 1
    assert campaign.sharing.eliminated_head_score_evaluations == 2
    assert tuple(projection.union_offsets for projection in campaign.sharing.query_projections) == (
        (0, 1, 2),
        (2, 3, 1),
    )
    assert engine.selected_body_calls == 1
    assert engine.selected_rows == [(5, 1, 3, 7)]
    assert result.evidence["body_execution_count"] == 1

    by_query = {readout.query_id: readout for readout in result.readouts}
    for readout in readouts:
        expected_logits, expected = independent[readout.query_id]
        actual = by_query[readout.query_id]
        assert actual.candidate_token_ids == readout.candidate_token_ids
        assert actual.candidate_logits == expected_logits
        assert actual.winner_token_id == expected["winner_token_id"]
        assert actual.runner_up_token_id == expected["runner_up_token_id"]
        assert actual.margin == expected["margin"]

    # Token 7 is the union winner, but alpha must reduce only its own ordered set.
    assert by_query["alpha"].winner_token_id == 5
    assert by_query["alpha"].runner_up_token_id == 3
    assert by_query["beta"].winner_token_id == 7
    assert by_query["beta"].runner_up_token_id == 3


def test_campaign_and_result_round_trip_with_stable_fingerprints():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _CampaignEngine()
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())

    assert CandidateReadout.from_dict(_readouts()[0].as_dict()) == _readouts()[0]
    binding = CampaignInputBinding.from_token_ids(token_ids)
    assert CampaignInputBinding.from_dict(binding.as_dict()) == binding

    restored = CandidateCampaign.from_json(campaign.to_json(indent=2))
    assert restored == campaign
    assert restored.fingerprint == campaign.fingerprint

    result = execute_candidate_campaign(engine, restored, token_ids)
    restored_result = CandidateCampaignExecutionResult.from_dict(result.as_dict())
    assert restored_result == result


def test_dense_cuda_campaign_preserves_all_graph_and_work_floor_guards():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _DenseCampaignEngine()
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())
    result = execute_candidate_campaign(engine, campaign, token_ids)

    assert campaign.base_bundle.lowered.backend == "cuda-qstore"
    assert campaign.base_bundle.lowered.reported_fabric == "cuda"
    assert campaign.base_bundle.lowered.implementation_status == "eager-adapter"
    assert not campaign.base_bundle.lowered.capture_executed
    assert "full-head-output-contract-parity" in (
        campaign.base_bundle.lowered.evidence_requirements
    )
    assert dict(campaign.base_bundle.plan.metadata)["output_pushdown"] is True
    assert campaign.base_bundle.plan.numerical_contract == (engine.subset_head_numerical_contract)
    graph = campaign.base_bundle.graph
    assert graph is not None
    assert not any(node.effects for node in graph.graph.nodes)
    head = next(
        node for node in graph.graph.nodes if node.node_id == "output.vocab_projection"
    ).parameters[0]
    assert head.access == "rows"
    assert head.row_indices == campaign.union_token_ids
    assert campaign.base_bundle.work_floor is not None
    demand = campaign.base_bundle.work_floor.rewritten.certificate.head_demand
    assert demand.complete and demand.attained
    assert engine.selected_body_calls == 1
    assert result.evidence["reported_fabric"] == "cuda"
    assert result.evidence["runtime_configuration_bound"] is True
    assert result.evidence["runtime_activation_dtype"] == "bf16"
    assert result.evidence["runtime_weight_dtype"] == "int8"
    assert result.evidence["runtime_fabric"] == "cuda"
    assert CandidateCampaign.from_json(campaign.to_json()) == campaign


def test_dense_cuda_campaign_capture_is_required_prepared_once_and_replayed():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _DenseCampaignEngine()
    campaign = compile_candidate_campaign(
        engine,
        token_ids,
        _readouts(),
        capture_requested=True,
    )

    assert campaign.base_bundle.plan.capture.requested
    assert campaign.base_bundle.plan.capture.eligible
    assert campaign.base_bundle.lowered.capture_ready
    assert not campaign.base_bundle.lowered.capture_executed

    prepared = prepare_candidate_campaign(engine, campaign, token_ids)
    assert engine.capture_preparations == 1
    assert engine.capture_replays == 0
    assert engine.selected_body_calls == 0

    first = prepared.execute()
    second = prepared.execute()
    assert first == second
    assert engine.capture_replays == 2
    assert engine.selected_body_calls == 0
    assert first.evidence["graph_replay"] is True
    assert first.evidence["capture_requested"] is True
    assert first.evidence["capture_ready"] is True
    assert first.evidence["capture_executed"] is True
    assert first.evidence["runtime_implementation_status"] == "cuda-graph"
    assert first.evidence["capture_metadata"]["stable_addresses_verified"] is True
    prepared.close()
    assert engine.capture_closes == 1
    with pytest.raises(RuntimeError, match="closed"):
        prepared.execute()


def test_capture_requested_campaign_fails_closed_without_engine_executor():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _DenseCampaignEngine()
    engine.prepare_selected_last_cuda_graph = None
    campaign = compile_candidate_campaign(
        engine,
        token_ids,
        _readouts(),
        capture_requested=True,
    )

    with pytest.raises(RuntimeError, match="cannot prepare"):
        prepare_candidate_campaign(engine, campaign, token_ids)
    assert engine.selected_body_calls == 0
    assert engine.capture_replays == 0


def test_capture_requested_campaign_rejects_declared_only_store_identity():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _DenseCampaignEngine()
    engine.store.content_identity_verified = False
    engine.store.identity_status = "declared-content-only-unverified"

    with pytest.raises(ValueError, match="blob-verified"):
        compile_candidate_campaign(
            engine,
            token_ids,
            _readouts(),
            capture_requested=True,
        )


def test_capture_requested_campaign_refuses_non_cuda_backend():
    with pytest.raises(ValueError, match="requires dense-qstore-cuda"):
        compile_candidate_campaign(
            _CampaignEngine(),
            np.asarray([1, 2, 3], dtype=np.int64),
            _readouts(),
            capture_requested=True,
        )


def test_campaign_runtime_refuses_precision_weight_and_fabric_drift_before_dispatch():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _DenseCampaignEngine()
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())

    engine.store.compute_dtype = torch.float16
    with pytest.raises(RuntimeError, match="activation dtype"):
        execute_candidate_campaign(engine, campaign, token_ids)

    engine.store.compute_dtype = torch.bfloat16
    engine.store.man = {**_tiny_qwen_manifest(), "dtype": "int4"}
    with pytest.raises(RuntimeError, match="semantic manifest digest"):
        execute_candidate_campaign(engine, campaign, token_ids)

    engine.store.man = _tiny_qwen_manifest()
    engine.device = torch.device("cpu")
    with pytest.raises(RuntimeError, match="runtime device"):
        execute_candidate_campaign(engine, campaign, token_ids)
    assert engine.selected_body_calls == 0


def test_campaign_runtime_refuses_cross_backend_execution_before_dispatch():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    paged = _CampaignEngine()
    dense = _DenseCampaignEngine()
    paged_campaign = compile_candidate_campaign(paged, token_ids, _readouts())
    dense_campaign = compile_candidate_campaign(dense, token_ids, _readouts())

    with pytest.raises(RuntimeError, match="backend does not match"):
        execute_candidate_campaign(dense, paged_campaign, token_ids)
    with pytest.raises(RuntimeError, match="backend does not match"):
        execute_candidate_campaign(paged, dense_campaign, token_ids)
    assert paged.selected_body_calls == 0
    assert dense.selected_body_calls == 0


def test_campaign_refuses_changed_runtime_tokens_before_execution():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _CampaignEngine()
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())

    with pytest.raises(ValueError, match="input binding"):
        execute_candidate_campaign(
            engine,
            campaign,
            np.asarray([1, 2, 4], dtype=np.int64),
        )
    assert engine.selected_body_calls == 0


def test_campaign_refuses_duplicate_queries_candidates_and_out_of_vocab_tokens():
    with pytest.raises(ValueError, match="unique inside"):
        CandidateReadout("duplicate-candidate", (1, 1))

    engine = _CampaignEngine()
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    duplicate_queries = (
        CandidateReadout("same", (1, 2)),
        CandidateReadout("same", (2, 3)),
    )
    with pytest.raises(ValueError, match="query IDs must be unique"):
        compile_candidate_campaign(engine, token_ids, duplicate_queries)

    out_of_vocab = (
        CandidateReadout("first", (1, 2)),
        CandidateReadout("second", (2, 8)),
    )
    with pytest.raises(ValueError, match="candidate token exceeds"):
        compile_candidate_campaign(engine, token_ids, out_of_vocab)
    with pytest.raises(ValueError, match="input token exceeds"):
        compile_candidate_campaign(
            engine,
            np.asarray([1, 2, 8], dtype=np.int64),
            _readouts(),
        )


def test_campaign_constructor_refuses_stateful_and_effectful_bundles():
    engine = _CampaignEngine()
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())
    bundle = campaign.base_bundle

    # WorkPlan v2 now rejects the malformed score/state mixture before it can reach the
    # campaign constructor; this is a stronger boundary than the historical downstream gate.
    with pytest.raises(ValueError, match="score WorkPlans cannot claim prefix"):
        replace(
            bundle.plan,
            prefix_state_ids=("prefix-state",),
        )

    compilation = bundle.graph
    assert compilation is not None
    certificate = compilation.rewrite_certificate
    assert certificate is not None
    first_node, *remaining_nodes = compilation.graph.nodes
    effectful_graph = replace(
        compilation.graph,
        nodes=(
            replace(
                first_node,
                effects=(Effect(EffectKind.STATE_READ, "test-state", 0),),
            ),
            *remaining_nodes,
        ),
    )
    effectful_certificate = replace(
        certificate,
        rewritten_fingerprint=effectful_graph.fingerprint,
    )
    effectful_compilation = compile_graph(
        effectful_graph,
        source_graph_fingerprint=compilation.source_graph_fingerprint,
        rewrite_certificate=effectful_certificate,
        metadata=dict(compilation.metadata),
    )
    assert bundle.work_floor is not None
    effectful_floor = replace(
        bundle.work_floor,
        rewritten=analyze_work_floor(effectful_graph, bundle.plan),
    )
    effectful_bundle = replace(
        bundle,
        graph=effectful_compilation,
        work_floor=effectful_floor,
    )
    with pytest.raises(ValueError, match="effect-free"):
        replace(campaign, base_bundle=effectful_bundle)


def test_campaign_serialized_tampering_is_rejected():
    engine = _CampaignEngine()
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())

    tampered_union = campaign.as_dict()
    tampered_union["union_token_ids"] = [1, 5, 3, 7]
    with pytest.raises(ValueError, match="stable first-use order"):
        CandidateCampaign.from_dict(tampered_union)

    tampered_fingerprint = campaign.as_dict()
    tampered_fingerprint["campaign_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        CandidateCampaign.from_dict(tampered_fingerprint)

    result = execute_candidate_campaign(engine, campaign, token_ids)
    tampered_result = result.as_dict()
    tampered_result["evidence"]["body_execution_count"] = 2
    with pytest.raises(ValueError, match="one body execution"):
        CandidateCampaignExecutionResult.from_dict(tampered_result)


def test_prepared_campaign_is_pure_then_dispatches_once_per_replay():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _CampaignEngine()
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())

    prepared = prepare_candidate_campaign(engine, campaign, token_ids)
    assert engine.selected_body_calls == 0

    first = prepared.execute()
    assert engine.selected_body_calls == 1
    second = prepared.execute()
    assert engine.selected_body_calls == 2
    assert second == first
    assert first.evidence["dispatch_prepared"] is True


def test_prepared_campaign_copies_input_and_matches_one_shot_bit_for_bit():
    canonical = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _CampaignEngine()
    campaign = compile_candidate_campaign(engine, canonical, _readouts())
    one_shot = execute_candidate_campaign(engine, campaign, canonical.copy())

    engine.selected_body_calls = 0
    caller_owned = canonical.copy()
    prepared = prepare_candidate_campaign(engine, campaign, caller_owned)
    assert engine.selected_body_calls == 0
    caller_owned[:] = np.asarray([7, 7, 7], dtype=np.int64)

    replayed = prepared.execute()
    assert engine.selected_body_calls == 1
    assert replayed == one_shot
    assert replayed.union_logits == one_shot.union_logits
    assert replayed.readouts == one_shot.readouts


def test_preparation_refuses_a_mismatched_token_binding_without_dispatch():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _CampaignEngine()
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())

    with pytest.raises(ValueError, match="input binding"):
        prepare_candidate_campaign(
            engine,
            campaign,
            np.asarray([1, 2, 4], dtype=np.int64),
        )
    assert engine.selected_body_calls == 0


def test_execution_evidence_is_deeply_immutable_and_forged_top_two_is_rejected():
    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _CampaignEngine()
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())
    result = execute_candidate_campaign(engine, campaign, token_ids)

    with pytest.raises(TypeError):
        result.evidence["body_execution_count"] = 2  # type: ignore[index]
    rewrite_ids = result.evidence["graph_rewrite_ids"]
    assert isinstance(rewrite_ids, tuple)

    forged = result.readouts[0].as_dict()
    forged.update(
        {
            "winner_token_id": 3,
            "runner_up_token_id": 1,
            "winner_logit": forged["candidate_logits"][2],
            "runner_up_logit": forged["candidate_logits"][1],
            "margin": (forged["candidate_logits"][2] - forged["candidate_logits"][1]),
        }
    )
    with pytest.raises(ValueError, match="valid top two"):
        CandidateReadoutResult.from_dict(forged)


def test_campaign_replay_cli_loads_prepares_once_and_replays(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    from mrun import cli as cli_module
    from mrun import engine as engine_module

    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _CampaignEngine()
    engine.encode = lambda _prompts, add_special_tokens=False: [token_ids.copy()]
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())
    artifact = tmp_path / "campaign.json"
    artifact.write_text(campaign.to_json(indent=2))
    output = tmp_path / "replay.json"
    opened: list[tuple[tuple, dict]] = []

    def fake_open(*args, **kwargs):
        opened.append((args, kwargs))
        return nullcontext(engine)

    monkeypatch.setattr(engine_module, "open_engine", fake_open)

    status = cli_module._campaign_replay(
        SimpleNamespace(
            artifact=artifact,
            prompt="same exact input",
            replays=3,
            compact_cache_mb=0.0,
            compute_dtype=None,
            out=output,
        )
    )

    assert status == 0
    assert engine.selected_body_calls == 3
    payload = json.loads(output.read_text())
    assert payload["campaign_fingerprint"] == campaign.fingerprint
    assert payload["engine_backend"] == "paged"
    assert payload["compiled_reported_fabric"] == "cpu"
    assert payload["reported_fabric"] == "cpu"
    assert payload["runtime_configuration"] == {
        "runtime_activation_dtype": "fp32",
        "runtime_weight_dtype": "int8",
        "runtime_device": "cpu",
        "runtime_fabric": "cpu",
    }
    assert payload["content_identity_verified"] is True
    assert payload["replay_count"] == 3
    assert payload["deterministic_public_results"] is True
    assert len(payload["replay_samples_ms"]) == 3
    assert opened == [
        (
            ("tiny-qwen",),
            {"backend": "paged", "device": "cpu"},
        )
    ]


def test_campaign_replay_derives_dense_dtype_and_refuses_override(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    from mrun import cli as cli_module
    from mrun import engine as engine_module

    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _DenseCampaignEngine()
    engine.encode = lambda _prompts, add_special_tokens=False: [token_ids.copy()]
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())
    artifact = tmp_path / "dense-campaign.json"
    artifact.write_text(campaign.to_json())
    output = tmp_path / "dense-replay.json"
    opened: list[dict] = []

    def fake_open(*_args, **kwargs):
        opened.append(kwargs)
        return nullcontext(engine)

    monkeypatch.setattr(engine_module, "open_engine", fake_open)
    base = {
        "artifact": artifact,
        "prompt": "same exact input",
        "replays": 1,
        "compact_cache_mb": 64.0,
        "out": output,
    }

    assert cli_module._campaign_replay(SimpleNamespace(**base, compute_dtype=None)) == 0
    assert opened == [
        {
            "backend": "dense-qstore-cuda",
            "device": "cuda",
            "compute_dtype": "bf16",
            "compact_cache_mb": 64.0,
        }
    ]
    assert (
        json.loads(output.read_text())["runtime_configuration"]["runtime_activation_dtype"]
        == "bf16"
    )

    opened.clear()
    assert cli_module._campaign_replay(SimpleNamespace(**base, compute_dtype="fp16")) == 1
    assert opened == []
    assert (
        cli_module._campaign_replay(
            SimpleNamespace(
                **{**base, "compact_cache_mb": -1.0},
                compute_dtype=None,
            )
        )
        == 1
    )
    assert opened == []


@pytest.mark.parametrize(
    ("weight_dtype", "flag"),
    (("int2", "int2"), ("int3", "int3"), ("int4", "int4")),
)
def test_campaign_replay_derives_paged_quantized_store_variants(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    weight_dtype: str,
    flag: str,
):
    from mrun import cli as cli_module
    from mrun import engine as engine_module

    token_ids = np.asarray([1, 2, 3], dtype=np.int64)
    engine = _CampaignEngine()
    engine.store.man = {**_tiny_qwen_manifest(), "dtype": weight_dtype}
    engine.store.content_identity_verified = False
    engine.store.identity_status = "packed-test-store-unverified"
    engine.encode = lambda _prompts, add_special_tokens=False: [token_ids.copy()]
    campaign = compile_candidate_campaign(engine, token_ids, _readouts())
    artifact = tmp_path / f"{weight_dtype}-campaign.json"
    artifact.write_text(campaign.to_json())
    opened: list[dict] = []

    def fake_open(*_args, **kwargs):
        opened.append(kwargs)
        return nullcontext(engine)

    monkeypatch.setattr(engine_module, "open_engine", fake_open)
    assert (
        cli_module._campaign_replay(
            SimpleNamespace(
                artifact=artifact,
                prompt="same exact input",
                replays=1,
                compact_cache_mb=0.0,
                compute_dtype=None,
                out=tmp_path / f"{weight_dtype}-replay.json",
            )
        )
        == 0
    )
    assert opened == [
        {
            "backend": "paged",
            "device": "cpu",
            flag: True,
        }
    ]
