from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.compiler import (
    Intervention,
    InterventionBranch,
    InterventionScienceGraph,
    TensorPayloadSpec,
    benchmark_intervention_sciencegraph,
    compile_intervention_sciencegraph,
    execute_intervention_sciencegraph,
    execute_serialized_sciencegraph_plan,
)
from mrun.engine.kernels.paged_forward import _apply_patch_ops_by_row


def _graph():
    return compile_intervention_sciencegraph(
        model_identity="qwen-test@revision",
        numerical_contract="same-codec-w8a16",
        prompt_token_ids=(10, 11, 12),
        branches=(
            InterventionBranch("clean", (7, 3)),
            InterventionBranch(
                "delete-l4",
                (3, 9),
                (Intervention(layer=4, indices=(1, 5)),),
            ),
            InterventionBranch(
                "scale-l2",
                (7, 9),
                (Intervention(layer=2, indices=(8,), op="scale", value=0.5),),
            ),
        ),
    )


def test_compile_intervention_sciencegraph_is_deterministic_and_projects_union(tmp_path) -> None:
    graph = _graph()
    again = _graph()
    pack = graph.branch_pack

    assert graph.fingerprint == again.fingerprint
    assert pack.fingerprint == again.branch_pack.fingerprint
    assert pack.fork.cut.layer == 2
    assert pack.candidate_union == (7, 3, 9)
    assert pack.candidate_projections == ((0, 1), (1, 2), (0, 2))
    assert pack.patch_maps() == (
        {},
        {4: [("zero", (1, 5), None)]},
        {2: [("scale", (8,), 0.5)]},
    )
    assert graph.to_dict()["fingerprint"] == graph.fingerprint
    assert InterventionScienceGraph.from_json(graph.to_json()) == graph
    artifact = graph.write_json(tmp_path / "nested" / "graph.json")
    assert InterventionScienceGraph.read_json(artifact) == graph
    assert not list(artifact.parent.glob("*.tmp"))
    tampered = copy.deepcopy(graph.to_dict())
    tampered["prompt_token_ids"][0] = 99
    with pytest.raises(ValueError, match="prompt identity|fingerprint"):
        InterventionScienceGraph.from_dict(tampered)
    divergent = copy.deepcopy(graph.branch_pack.to_dict())
    divergent["row_patch_maps"][1][0]["ops"][0]["op"] = "scale"
    divergent["row_patch_maps"][1][0]["ops"][0]["value"] = 1.0
    with pytest.raises(ValueError, match="diverges"):
        type(graph.branch_pack).from_dict(divergent)


def test_sciencegraph_rejects_ambiguous_or_nonfinite_interventions() -> None:
    with pytest.raises(ValueError, match="unique"):
        InterventionBranch("bad", (1, 1))
    with pytest.raises(ValueError, match="finite"):
        Intervention(layer=0, indices=(1,), op="scale", value=float("nan"))
    with pytest.raises(ValueError, match="same typed index set"):
        InterventionBranch(
            "bad",
            (1,),
            (
                Intervention(layer=0, indices=(2,)),
                Intervention(layer=0, indices=(2,), op="scale", value=0.5),
            ),
        )


def test_tensor_payloads_and_all_typed_ports_resolve_fail_closed() -> None:
    mean = torch.tensor([0.25, -0.5], dtype=torch.float32)
    direction = torch.tensor([1.0, 0.0, -1.0, 0.5], dtype=torch.float32)
    specs = (
        TensorPayloadSpec.from_tensor("mean-v1", mean),
        TensorPayloadSpec.from_tensor("direction-v1", direction),
    )
    graph = compile_intervention_sciencegraph(
        model_identity="qwen-test@revision",
        numerical_contract="same-codec-w8a16",
        prompt_token_ids=(1, 2),
        payload_specs=specs,
        branches=(
            InterventionBranch(
                "ports",
                (3, 4),
                (
                    Intervention(
                        layer=1,
                        indices=(0,),
                        port="attention_head_output",
                        op="global_mean",
                        payload_id="mean-v1",
                    ),
                    Intervention(
                        layer=2,
                        indices=(),
                        port="residual_output",
                        op="proj_remove",
                        payload_id="direction-v1",
                    ),
                ),
            ),
        ),
    )
    mlp, head, resid = graph.branch_pack.resolve_patch_maps(
        {"mean-v1": mean, "direction-v1": direction}
    )
    assert mlp == ({},)
    assert head[0][1][0][0:2] == ("global_mean", (0,))
    torch.testing.assert_close(head[0][1][0][2], mean)
    assert head[0][1][0][2].data_ptr() != mean.data_ptr()
    assert resid[0][2][0][0] == "proj_remove"
    torch.testing.assert_close(resid[0][2][0][1], direction)
    with pytest.raises(ValueError, match="binding drift"):
        graph.branch_pack.resolve_patch_maps({"mean-v1": mean + 1, "direction-v1": direction})
    with pytest.raises(ValueError, match="exactly match"):
        graph.branch_pack.resolve_patch_maps({"mean-v1": mean})


def test_row_local_paged_patch_isolates_clean_and_distinct_branches() -> None:
    source = torch.arange(2 * 3 * 5, dtype=torch.float32).view(2, 3, 5)
    result = _apply_patch_ops_by_row(
        source,
        {
            0: [("zero", (1, 3), None)],
            1: [("scale", (0, 4), 0.5)],
        },
    )

    torch.testing.assert_close(source[:, :, 2], result[:, :, 2])
    assert torch.count_nonzero(result[0, :, (1, 3)]) == 0
    torch.testing.assert_close(result[1, :, (0, 4)], source[1, :, (0, 4)] * 0.5)
    assert torch.equal(source, torch.arange(30, dtype=torch.float32).view(2, 3, 5))


class _SelectedEngine:
    def __init__(self) -> None:
        self.calls = []

    def selected_last_intervention_branches(self, prompt_ids, **kwargs):
        self.calls.append((prompt_ids, kwargs))
        full = torch.tensor(
            [
                [5.0, 1.0, -1.0],
                [2.0, 4.0, 3.0],
                [0.0, -2.0, 6.0],
            ]
        )
        patch_rows = kwargs["patch_ops_by_layer_rows"]
        if len(patch_rows) == 3:
            scores = full
        else:
            condition = 0 if not patch_rows[0] else 1 if 4 in patch_rows[0] else 2
            offsets = {7: 0, 3: 1, 9: 2}
            scores = full[condition : condition + 1].index_select(
                1,
                torch.as_tensor([offsets[token] for token in kwargs["token_ids"]]),
            )
        return scores, {
            "shared_prefix_materialized": True,
            "cut_layer": kwargs["cut_layer"],
        }

    def forward_patched(
        self,
        ids,
        *,
        patch_ops_by_layer=None,
        head_patch_ops_by_layer=None,
        resid_patch_ops_by_layer=None,
    ):
        assert head_patch_ops_by_layer is None
        assert resid_patch_ops_by_layer is None
        logits = torch.zeros(len(ids), 10)
        if not patch_ops_by_layer:
            logits[-1, 7], logits[-1, 3], logits[-1, 9] = 5.0, 1.0, -1.0
        elif 4 in patch_ops_by_layer:
            logits[-1, 7], logits[-1, 3], logits[-1, 9] = 2.0, 4.0, 3.0
        else:
            logits[-1, 7], logits[-1, 3], logits[-1, 9] = 0.0, -2.0, 6.0
        return logits, [], {}


def test_execute_sciencegraph_fans_selected_union_back_to_logical_branches() -> None:
    engine = _SelectedEngine()
    result = execute_intervention_sciencegraph(
        engine,
        _graph(),
        model_identity="qwen-test@revision",
        numerical_contract="same-codec-w8a16",
    )

    assert len(engine.calls) == 1
    prompt_ids, kwargs = engine.calls[0]
    assert prompt_ids == (10, 11, 12)
    assert kwargs["patch_ops_by_layer_rows"][0] == {}
    assert kwargs["token_ids"] == (7, 3, 9)
    assert [row["winner_token_id"] for row in result["results"]] == [7, 3, 9]
    assert [row["margin"] for row in result["results"]] == [4.0, 1.0, 6.0]
    assert result["telemetry"] == {
        "logical_branches": 3,
        "physical_forward_calls": 2,
        "physical_weight_traversals": 2,
        "candidate_union_count": 3,
        "model_identity": "qwen-test@revision",
        "numerical_contract": "same-codec-w8a16",
        "output_path": "statecut_selected_union",
        "row_local_intervention_fused": True,
        "shared_prefix_materialized": True,
        "cut_layer": 2,
    }
    assert result["fingerprint"]


def test_neutral_branch_plan_adapter_preserves_sciencegraph_rows_and_telemetry() -> None:
    graph = _graph()
    neutral_plan = {
        "schema": "manalysis.generative-branch-plan.v1",
        "plan_id": "sciencegraph-plan",
        "model_identity": "qwen-test@revision",
        "numerical_contract": "same-codec-w8a16",
        "shared_cut": {
            "cut_id": "layer-2",
            "component_id": "block-2",
            "port": "residual",
            "branch_ids": [branch.branch_id for branch in graph.branch_pack.fork.branches],
        },
        "branches": [
            {
                "branch_id": branch.branch_id,
                "row_slot": index,
                "compatibility_key": "paged",
                "route_key": "default",
                "fork_id": "fork-2",
            }
            for index, branch in enumerate(graph.branch_pack.fork.branches)
        ],
        "metadata": {"sciencegraph": graph.to_dict()},
    }
    result = execute_serialized_sciencegraph_plan(
        _SelectedEngine(),
        neutral_plan,
    )
    assert list(result["branch_outputs"]) == ["clean", "delete-l4", "scale-l2"]
    assert result["telemetry"]["adapter"] == "sciencegraph"
    assert result["telemetry"]["physical_backend"] == "mrun.paged.sciencegraph"
    assert result["parity"]["row_order_verified"] is True
    assert result["telemetry"]["max_branch_batch"] == 1
    assert result["telemetry"]["max_branch_batch_source"] == "neutral_plan_default"
    assert result["parity"]["execution_identity_verified"] is True


def test_neutral_branch_plan_adapter_uses_declared_batch_and_validates_row_abi() -> None:
    graph = _graph()
    neutral_plan = {
        "schema": "manalysis.generative-branch-plan.v1",
        "plan_id": "sciencegraph-plan-batched",
        "model_identity": "qwen-test@revision",
        "numerical_contract": "same-codec-w8a16",
        "shared_cut": {
            "cut_id": "layer-2",
            "component_id": "block-2",
            "port": "residual",
            "trajectory_position": {"layer": 2},
            "abi": {
                "model_identity": "qwen-test@revision",
                "numerical_contract": "same-codec-w8a16",
            },
            "branch_ids": [branch.branch_id for branch in graph.branch_pack.fork.branches],
        },
        "branches": [
            {
                "branch_id": branch.branch_id,
                "row_slot": index,
                "compatibility_key": "paged",
                "route_key": "default",
                "fork_id": "fork-2",
            }
            for index, branch in enumerate(graph.branch_pack.fork.branches)
        ],
        "max_batch_size": 4,
        "metadata": {"sciencegraph": graph.to_dict()},
    }
    engine = _SelectedEngine()
    result = execute_serialized_sciencegraph_plan(engine, neutral_plan)
    assert result["telemetry"]["max_branch_batch"] == 4
    assert result["execution_identity"]["row_slots"] == [0, 1, 2]
    assert result["execution_identity"]["fork_id"] == "fork-2"
    assert engine.calls[-1][1]["max_branch_batch"] == 4

    broken = copy.deepcopy(neutral_plan)
    broken["branches"][1]["row_slot"] = 0
    with pytest.raises(ValueError, match="row_slot"):
        execute_serialized_sciencegraph_plan(_SelectedEngine(), broken)

    broken = copy.deepcopy(neutral_plan)
    broken["shared_cut"]["trajectory_position"]["layer"] = 3
    with pytest.raises(ValueError, match="shared cut layer"):
        execute_serialized_sciencegraph_plan(_SelectedEngine(), broken)

    broken = copy.deepcopy(neutral_plan)
    broken["model_identity"] = "other-model@revision"
    with pytest.raises(ValueError, match="model identity"):
        execute_serialized_sciencegraph_plan(_SelectedEngine(), broken)


def test_neutral_branch_plan_explicit_batch_override_is_audited() -> None:
    graph = _graph()
    neutral_plan = {
        "schema": "manalysis.generative-branch-plan.v1",
        "plan_id": "sciencegraph-plan-override",
        "model_identity": "qwen-test@revision",
        "numerical_contract": "same-codec-w8a16",
        "shared_cut": {
            "cut_id": "layer-2",
            "component_id": "block-2",
            "port": "residual",
            "branch_ids": [branch.branch_id for branch in graph.branch_pack.fork.branches],
        },
        "branches": [
            {
                "branch_id": branch.branch_id,
                "row_slot": index,
                "compatibility_key": "paged",
                "route_key": "default",
                "fork_id": "fork-2",
            }
            for index, branch in enumerate(graph.branch_pack.fork.branches)
        ],
        "max_batch_size": 4,
        "metadata": {"sciencegraph": graph.to_dict()},
    }
    engine = _SelectedEngine()
    result = execute_serialized_sciencegraph_plan(
        engine,
        neutral_plan,
        max_branch_batch=2,
    )
    assert result["telemetry"]["max_branch_batch"] == 2
    assert result["telemetry"]["max_branch_batch_source"] == "adapter_override"


def test_execute_sciencegraph_refuses_wrong_runtime_identity() -> None:
    with pytest.raises(ValueError, match="model identity"):
        execute_intervention_sciencegraph(
            _SelectedEngine(),
            _graph(),
            model_identity="other-model@revision",
            numerical_contract="same-codec-w8a16",
        )


def test_sciencegraph_benchmark_compares_complete_independent_and_compiled_legs() -> None:
    result = benchmark_intervention_sciencegraph(
        _SelectedEngine(),
        _graph(),
        model_identity="qwen-test@revision",
        numerical_contract="same-codec-w8a16",
        warmups=0,
        repeats=3,
        min_speedup=0.0001,
    )
    assert result["exact_winners"] is True
    assert result["max_abs_score_delta"] == 0.0
    assert result["qualified"] is True
    assert len(result["independent_seconds"]) == len(result["compiled_seconds"]) == 3
    assert result["fingerprint"]


def test_paged_selected_branch_api_pushes_row_maps_and_candidate_union(monkeypatch) -> None:
    from mrun.engine import paged as paged_module
    from mrun.engine.paged import PagedEngine

    calls = []

    def fake_batched(_store, ids_list, **kwargs):
        calls.append((ids_list, kwargs))
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]]), np.asarray([2, 2])

    class _Store:
        def embed_rows(self, name, ids):
            assert name == "lm_head"
            assert list(ids) == [7, 9]
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(paged_module.pf, "batched_paged_logits", fake_batched)
    engine = object.__new__(PagedEngine)
    engine.arch = "qwen2"
    engine.store = _Store()
    engine.composite_store = None
    rows = [np.asarray([1, 2]), np.asarray([1, 2])]
    maps = [None, {1: [("zero", (2,), None)]}]

    scores = engine.selected_last_patched_rows(rows, maps, (7, 9))

    torch.testing.assert_close(scores, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert calls[0][1]["return_hidden"] is True
    assert calls[0][1]["patch_ops_by_layer_rows"] is maps


class _TwoLayerStore:
    cfg = {
        "hidden_size": 4,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "intermediate_size": 6,
        "vocab_size": 10,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
    }

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(23)
        hidden, intermediate = 4, 6
        self.embedding = torch.randn(10, hidden, generator=generator) * 0.2
        self.head = torch.randn(10, hidden, generator=generator) * 0.2
        self.weights = {}
        self.norms = {"norm.final": torch.ones(hidden)}
        for layer in range(2):
            self.weights.update(
                {
                    f"L{layer}.q": torch.randn(hidden, hidden, generator=generator) * 0.1,
                    f"L{layer}.k": torch.randn(2, hidden, generator=generator) * 0.1,
                    f"L{layer}.v": torch.randn(2, hidden, generator=generator) * 0.1,
                    f"L{layer}.o": torch.randn(hidden, hidden, generator=generator) * 0.1,
                    f"L{layer}.gate": torch.randn(intermediate, hidden, generator=generator) * 0.1,
                    f"L{layer}.up": torch.randn(intermediate, hidden, generator=generator) * 0.1,
                    f"L{layer}.down": torch.randn(hidden, intermediate, generator=generator) * 0.1,
                }
            )
            self.norms[f"L{layer}.ln1"] = torch.ones(hidden)
            self.norms[f"L{layer}.ln2"] = torch.ones(hidden)

    def embed_rows(self, name, ids):
        rows = torch.as_tensor(np.asarray(ids), dtype=torch.long)
        return (self.embedding if name == "embed" else self.head).index_select(0, rows)

    def fp32(self, name):
        return self.norms[name]

    def matmul(self, name, value):
        return value @ self.weights[name].T

    def has(self, _name):
        return False

    def row_blocks(self, name, bs=4):
        assert name == "lm_head"
        for start in range(0, len(self.head), bs):
            end = min(start + bs, len(self.head))
            yield start, end, self.head[start:end]


def test_physical_statecut_prefix_reuse_matches_complete_batched_branches() -> None:
    from mrun.engine.kernels import paged_forward as pf
    from mrun.engine.paged import PagedEngine

    store = _TwoLayerStore()
    prompt = np.asarray([1, 2, 3], dtype=np.int64)
    mlp_rows = [None, {1: [("zero", (0, 2), None)]}, None, None]
    head_rows = [None, None, {1: [("scale", (1,), 0.25)]}, None]
    direction = torch.tensor([1.0, 0.0, -0.5, 0.25])
    resid_rows = [None, None, None, {1: [("proj_remove", direction, None)]}]
    branch_inputs = [prompt] * 4
    full_hidden, _ = pf.batched_paged_logits(
        store,
        branch_inputs,
        last_only=True,
        return_hidden=True,
        patch_ops_by_layer_rows=mlp_rows,
        head_patch_ops_by_layer_rows=head_rows,
        resid_patch_ops_by_layer_rows=resid_rows,
    )
    expected = full_hidden.float() @ store.embed_rows("lm_head", [4, 7]).T

    engine = object.__new__(PagedEngine)
    engine.arch = "qwen2"
    engine.store = store
    engine.composite_store = None
    engine.n_layer = 2
    actual, telemetry = engine.selected_last_intervention_branches(
        prompt,
        cut_layer=1,
        token_ids=(4, 7),
        patch_ops_by_layer_rows=mlp_rows,
        head_patch_ops_by_layer_rows=head_rows,
        resid_patch_ops_by_layer_rows=resid_rows,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
    assert telemetry["shared_prefix_materialized"] is True
    assert telemetry["physical_layer_weight_traversals"] == 2
    assert telemetry["independent_layer_weight_traversals"] == 8

    chunked, chunked_telemetry = engine.selected_last_intervention_branches(
        prompt,
        cut_layer=1,
        token_ids=(4, 7),
        patch_ops_by_layer_rows=mlp_rows,
        head_patch_ops_by_layer_rows=head_rows,
        resid_patch_ops_by_layer_rows=resid_rows,
        max_branch_batch=2,
    )
    torch.testing.assert_close(chunked, expected, rtol=0, atol=2e-7)
    assert chunked_telemetry["suffix_weight_traversals"] == 2
    assert chunked_telemetry["physical_layer_weight_traversals"] == 3


def test_kv_projection_ports_compile_resolve_and_round_trip() -> None:
    key = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
    value = torch.tensor([[-0.1, 0.2], [-0.3, 0.4], [-0.5, 0.6]], dtype=torch.float32)
    graph = compile_intervention_sciencegraph(
        model_identity="tiny-qwen@23",
        numerical_contract="fp32-test",
        prompt_token_ids=(1, 2, 3),
        payload_specs=(
            TensorPayloadSpec.from_tensor("key-donor", key),
            TensorPayloadSpec.from_tensor("value-donor", value),
        ),
        branches=(
            InterventionBranch("clean", (4, 7)),
            InterventionBranch(
                "kv-repair",
                (4, 7),
                (
                    Intervention(
                        layer=1,
                        indices=(0, 2),
                        port="key_projection",
                        op="position_replace",
                        payload_id="key-donor",
                    ),
                    Intervention(
                        layer=1,
                        indices=(0, 2),
                        port="value_projection",
                        op="position_replace",
                        payload_id="value-donor",
                    ),
                ),
            ),
        ),
    )

    mlp, head, resid, key_rows, value_rows = graph.branch_pack.resolve_all_patch_maps(
        {"key-donor": key, "value-donor": value}
    )
    assert mlp == ({}, {})
    assert head == ({}, {})
    assert resid == ({}, {})
    assert key_rows[1][1][0][0:2] == ("position_replace", (0, 2))
    assert value_rows[1][1][0][0:2] == ("position_replace", (0, 2))
    torch.testing.assert_close(key_rows[1][1][0][2], key)
    torch.testing.assert_close(value_rows[1][1][0][2], value)
    assert InterventionScienceGraph.from_json(graph.to_json()) == graph


def test_kv_statecut_branches_match_complete_batched_execution() -> None:
    from mrun.engine.kernels import paged_forward as pf
    from mrun.engine.paged import PagedEngine

    store = _TwoLayerStore()
    prompt = np.asarray([1, 2, 3], dtype=np.int64)
    key = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    value = torch.tensor([[-0.1, 0.2], [-0.3, 0.4], [-0.5, 0.6]])
    key_rows = [None, {1: [("position_replace", (0, 2), key)]}, None]
    value_rows = [None, None, {1: [("position_replace", (0, 2), value)]}]
    branch_inputs = [prompt] * 3
    full_hidden, _ = pf.batched_paged_logits(
        store,
        branch_inputs,
        last_only=True,
        return_hidden=True,
        key_patch_ops_by_layer_rows=key_rows,
        value_patch_ops_by_layer_rows=value_rows,
    )
    expected = full_hidden.float() @ store.embed_rows("lm_head", [4, 7]).T

    engine = object.__new__(PagedEngine)
    engine.arch = "qwen2"
    engine.store = store
    engine.composite_store = None
    engine.n_layer = 2
    actual, telemetry = engine.selected_last_intervention_branches(
        prompt,
        cut_layer=1,
        token_ids=(4, 7),
        patch_ops_by_layer_rows=[None] * 3,
        head_patch_ops_by_layer_rows=[None] * 3,
        resid_patch_ops_by_layer_rows=[None] * 3,
        key_patch_ops_by_layer_rows=key_rows,
        value_patch_ops_by_layer_rows=value_rows,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
    assert telemetry["physical_layer_weight_traversals"] == 2
    assert telemetry["independent_layer_weight_traversals"] == 6


def test_paged_engine_captures_raw_kv_projection_bank() -> None:
    from mrun.engine.paged import PagedEngine

    engine = object.__new__(PagedEngine)
    engine.arch = "qwen2"
    engine.store = _TwoLayerStore()
    engine.composite_store = None
    engine.n_layer = 2
    keys, values = engine.kv_projections(np.asarray([1, 2, 3], dtype=np.int64))

    assert len(keys) == len(values) == 2
    assert all(tuple(tensor.shape) == (3, 2) for tensor in (*keys, *values))
    assert all(torch.isfinite(tensor).all() for tensor in (*keys, *values))


def test_paged_engine_batches_raw_kv_projection_bank_with_true_lengths() -> None:
    from mrun.engine.paged import PagedEngine

    engine = object.__new__(PagedEngine)
    engine.arch = "qwen2"
    engine.store = _TwoLayerStore()
    engine.composite_store = None
    engine.n_layer = 2
    rows = engine.kv_projections_batch(
        [
            np.asarray([1, 2, 3], dtype=np.int64),
            np.asarray([4, 5], dtype=np.int64),
        ]
    )

    assert len(rows) == 2
    assert [tuple(value.shape) for value in (*rows[0][0], *rows[0][1])] == [(3, 2)] * 4
    assert [tuple(value.shape) for value in (*rows[1][0], *rows[1][1])] == [(2, 2)] * 4
    scalar = engine.kv_projections(np.asarray([1, 2, 3], dtype=np.int64))
    for actual, expected in zip((*rows[0][0], *rows[0][1]), (*scalar[0], *scalar[1]), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)


def test_complete_sciencegraph_benchmark_runs_against_real_paged_math() -> None:
    from mrun.engine.kernels import paged_forward as pf
    from mrun.engine.paged import PagedEngine

    direction = torch.tensor([1.0, 0.0, -0.5, 0.25])
    graph = compile_intervention_sciencegraph(
        model_identity="tiny-qwen@23",
        numerical_contract="fp32-test",
        prompt_token_ids=(1, 2, 3),
        payload_specs=(TensorPayloadSpec.from_tensor("direction", direction),),
        branches=(
            InterventionBranch("clean", (4, 7)),
            InterventionBranch("mlp", (4, 7), (Intervention(layer=1, indices=(0, 2)),)),
            InterventionBranch(
                "head",
                (4, 7),
                (
                    Intervention(
                        layer=1,
                        indices=(1,),
                        port="attention_head_output",
                        op="scale",
                        value=0.25,
                    ),
                ),
            ),
            InterventionBranch(
                "residual",
                (4, 7),
                (
                    Intervention(
                        layer=1,
                        indices=(),
                        port="residual_output",
                        op="proj_remove",
                        payload_id="direction",
                    ),
                ),
            ),
        ),
    )
    engine = object.__new__(PagedEngine)
    engine.arch = "qwen2"
    engine.store = _TwoLayerStore()
    engine.composite_store = None
    engine.n_layer = 2
    engine._paged_logits = pf.paged_logits

    benchmark = benchmark_intervention_sciencegraph(
        engine,
        graph,
        model_identity="tiny-qwen@23",
        numerical_contract="fp32-test",
        payload_bindings={"direction": direction},
        warmups=0,
        repeats=3,
        atol=2e-6,
        min_speedup=0.0001,
    )
    assert benchmark["max_abs_score_delta"] <= 2e-6
    assert benchmark["exact_winners"] is True
    assert benchmark["qualified"] is True


def test_sciencegraph_cli_replays_a_bound_artifact(tmp_path, monkeypatch, capsys) -> None:
    from mrun import cli
    from mrun import compiler as compiler_module
    from mrun import engine as engine_module

    artifact = _graph().write_json(tmp_path / "graph.json")
    fake = _SelectedEngine()
    fake.capabilities = lambda: SimpleNamespace(intervention_sciencegraph=True)

    @contextmanager
    def open_fake(*_args, **_kwargs):
        yield fake

    monkeypatch.setattr(engine_module, "open_engine", open_fake)
    monkeypatch.setattr(
        compiler_module,
        "bind_sciencegraph_model_identity",
        lambda _engine: "qwen-test@revision",
    )
    status = cli.main(
        [
            "sciencegraph",
            "qwen-test",
            str(artifact),
            "--numerical-contract",
            "same-codec-w8a16",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "mrun-intervention-sciencegraph-execution-v2"
