from __future__ import annotations

import itertools
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.compiler import (
    DemandRewriteCertificate,
    Effect,
    EffectKind,
    GraphCompilation,
    OpGraph,
    OpKind,
    OpNode,
    OutputContract,
    ParameterRef,
    RegionSlice,
    StorageClass,
    TensorSpec,
    WorkFloorComparison,
    allocate_liveness,
    backward_slice,
    build_dense_work_plan,
    build_op_graph,
    build_output_demand_graph,
    build_physical_resource_graph,
    compile_graph,
    compile_work_plan,
    execute_lowered_plan,
    find_fusion_regions,
    place_binary_backends,
    trace_qstore_graph_execution,
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
    return {
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


def _plan(
    contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
    *,
    candidates=((1, 3), (3, 4)),
):
    logical_head_rows = (
        0
        if contract is OutputContract.HIDDEN_STATE_ONLY
        else 2
        if contract is OutputContract.SELECTED_TOKEN_ROWS
        else len({token for row in candidates for token in row})
        if contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
        else 8
    )
    return build_dense_work_plan(
        model_name="tiny-qwen",
        model_revision="unversioned:tiny-qwen",
        store_fingerprint="unfingerprinted:tiny-qwen",
        batch_size=2,
        sequence_length=3,
        activation_dtype="fp32",
        weight_dtype="int8",
        accumulator_dtype="fp32",
        output_contract=contract,
        candidate_token_ids=(
            candidates if contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN else ()
        ),
        required_output_rows=((1, 4) if contract is OutputContract.SELECTED_TOKEN_ROWS else ()),
        numerical_contract="paged-qstore-subset-head-fp32-v1",
        cache_admission="unknown",
        structured_operator_ids=("paged-transformer",),
        metadata={
            "component_cache_budgets_json": "",
            "configured_output_row_count": 8,
            "engine_device": "cpu",
            "head_output_pushdown": True,
            "input_token_limit": 8,
            "logical_head_row_count": logical_head_rows,
            "logical_hidden_width": 4,
            "output_pushdown": True,
            "ring_staging_bytes": 0,
            "weight_cache_budget_bytes": 0,
            "weight_cache_policy": "unknown",
        },
    )


def _value(
    value_id,
    *,
    storage=StorageClass.ACTIVATION,
    memory_space="cpu",
):
    return TensorSpec(
        value_id=value_id,
        shape=(4,),
        dtype="fp32",
        layout="C",
        storage_class=storage,
        memory_space=memory_space,
    )


def _simple_graph(*, with_effect=False):
    values = (
        _value("x", storage=StorageClass.INPUT),
        _value("a"),
        _value("b"),
        _value("dead"),
        _value("state"),
    )
    nodes = [
        OpNode("make-a", OpKind.IDENTITY, inputs=("x",), outputs=("a",)),
        OpNode("make-b", OpKind.SILU, inputs=("a",), outputs=("b",)),
        OpNode("make-dead", OpKind.IDENTITY, inputs=("x",), outputs=("dead",)),
    ]
    if with_effect:
        nodes.append(
            OpNode(
                "write-state",
                OpKind.IDENTITY,
                inputs=("a",),
                outputs=("state",),
                effects=(Effect(EffectKind.STATE_WRITE, "kv", 1),),
            )
        )
    else:
        nodes.append(OpNode("make-state", OpKind.IDENTITY, inputs=("a",), outputs=("state",)))
    return OpGraph(
        model_name="toy",
        model_revision="r",
        store_fingerprint="s",
        architecture="toy",
        numerical_contract="exact",
        values=values,
        nodes=tuple(nodes),
        inputs=("x",),
        outputs=("b",),
    )


def test_qwen_candidate_rewrite_is_alias_aware_and_certificate_bound():
    plan = _plan()
    manifest = _tiny_qwen_manifest()
    source, rewritten, certificate = build_output_demand_graph(plan, manifest)

    assert certificate.source_fingerprint == source.fingerprint
    assert certificate.rewritten_fingerprint == rewritten.fingerprint
    assert "candidate-union-into-parameter-access" in certificate.rewrite_ids
    assert DemandRewriteCertificate.from_dict(certificate.as_dict()) == certificate

    source_head = source.node_map["output.source_full_vocab"].parameters[0]
    rewritten_head = rewritten.node_map["output.vocab_projection"].parameters[0]
    assert source_head.access == "all"
    assert rewritten_head.access == "rows"
    assert rewritten_head.row_indices == (1, 3, 4)
    assert rewritten_head.logical_name == "lm_head"
    assert rewritten_head.physical_name == "embed"
    assert rewritten_head.is_alias

    logical_resources = {parameter.logical_name for parameter in rewritten.parameter_refs}
    assert logical_resources == set(manifest["blocks"])
    compilation = compile_graph(
        rewritten,
        source_graph_fingerprint=source.fingerprint,
        rewrite_certificate=certificate,
    )
    assert compilation.liveness_plan.reserved_bytes < (
        compilation.liveness_plan.naive_reserved_bytes
    )
    assert GraphCompilation.from_json(compilation.to_json()) == compilation
    tampered = compilation.as_dict()
    assert tampered["rewrite_certificate"] is not None
    tampered["rewrite_certificate"]["rewritten_fingerprint"] = source.fingerprint
    with pytest.raises(
        ValueError,
        match="rewrite certificate rewritten graph fingerprint mismatch",
    ):
        GraphCompilation.from_dict(tampered)
    tampered = compilation.as_dict()
    tampered["liveness_plan"]["reserved_bytes"] += 64
    with pytest.raises(ValueError, match="liveness plan does not match compiled graph"):
        GraphCompilation.from_dict(tampered)
    tampered = compilation.as_dict()
    tampered["fusion_plan"]["unfused_node_ids"] = []
    with pytest.raises(
        ValueError,
        match="fusion plan does not cover exactly the compiled graph nodes",
    ):
        GraphCompilation.from_dict(tampered)


def test_candidate_work_floor_proves_only_the_output_head_scope():
    plan = _plan()
    bundle = compile_work_plan(
        plan,
        "paged-qstore",
        manifest=_tiny_qwen_manifest(),
    )
    assert bundle.work_floor is not None
    floor = bundle.work_floor
    source = floor.source.certificate.head_demand
    rewritten = floor.rewritten.certificate.head_demand

    assert source.required_evaluation_rows == 2
    assert source.realized_evaluation_rows == 6
    assert source.required_vocabulary_rows == 3
    assert source.realized_vocabulary_rows == 8
    # The two rows require {1,3} and {3,4}: four scalar logits, not the
    # six-element dense cross product B * |union|.
    assert source.required_scalar_outputs == 4
    assert source.realized_scalar_outputs == 48
    assert source.excess_scalar_outputs == 44
    assert source.excess_parameter_extent_bytes == 40
    assert not source.attained

    assert rewritten.required_scalar_outputs == 4
    assert rewritten.realized_scalar_outputs == 6
    assert rewritten.excess_scalar_outputs == 2
    assert rewritten.excess_parameter_extent_bytes == 0
    assert rewritten.complete
    assert not rewritten.attained
    assert floor.source_head_excess_scalar_outputs == 44
    assert floor.rewritten_head_excess_scalar_outputs == 2
    assert floor.source_minus_rewritten_per_access_parameter_bytes == 40
    assert floor.rewritten.resources.unresolved_runtime_access_count == 1
    assert WorkFloorComparison.from_dict(floor.as_dict()) == floor

    head_region = next(
        region for region in floor.rewritten.resources.regions if region.region_id == "w:0:32"
    )
    assert head_region.static_slices == (
        # Candidate rows 1, 3, and 4; the last two are coalesced.
        RegionSlice("w", 4, 4),
        RegionSlice("w", 12, 8),
    )

    tampered = bundle.as_dict()
    tampered_head = tampered["work_floor"]["rewritten"]["certificate"]["head_demand"]
    tampered_head["attained"] = True
    with pytest.raises(ValueError, match="head-floor attainment flag is inconsistent"):
        type(bundle).from_dict(tampered)


def test_candidate_work_floor_attains_scalar_floor_for_shared_candidate_sets():
    plan = _plan(candidates=((1, 3), (1, 3)))
    bundle = compile_work_plan(
        plan,
        "paged-qstore",
        manifest=_tiny_qwen_manifest(),
    )

    assert bundle.work_floor is not None
    head = bundle.work_floor.rewritten.certificate.head_demand
    assert head.required_vocabulary_rows == 2
    assert head.realized_vocabulary_rows == 2
    assert head.required_scalar_outputs == 4
    assert head.realized_scalar_outputs == 4
    assert head.complete
    assert head.attained


def test_physical_resource_graph_unions_partially_overlapping_regions():
    graph = OpGraph(
        model_name="overlap",
        model_revision="r",
        store_fingerprint="s",
        architecture="generic",
        numerical_contract="exact",
        values=(
            _value("x", storage=StorageClass.INPUT),
            _value("a"),
            _value("b"),
        ),
        nodes=(
            OpNode(
                "first",
                OpKind.GENERIC_PARAMETER_STREAM,
                inputs=("x",),
                outputs=("a",),
                parameters=(
                    ParameterRef(
                        "first",
                        "first",
                        "fp32",
                        (2,),
                        (("e", 0, 8),),
                    ),
                ),
            ),
            OpNode(
                "second",
                OpKind.GENERIC_PARAMETER_STREAM,
                inputs=("a",),
                outputs=("b",),
                parameters=(
                    ParameterRef(
                        "second",
                        "second",
                        "fp32",
                        (2,),
                        (("e", 4, 8),),
                    ),
                ),
            ),
        ),
        inputs=("x",),
        outputs=("b",),
    )

    resources = build_physical_resource_graph(graph)
    assert resources.per_access_parameter_extent_bytes == 16
    assert resources.mandatory_unique_parameter_bytes == 12
    assert len(resources.regions) == 2
    assert type(resources).from_dict(resources.as_dict()) == resources


def test_graph_bound_execution_records_the_rewrite_route_and_refuses_missing_proof():
    plan = _plan(OutputContract.LAST_TOKEN_LOGITS)
    bundle = compile_work_plan(
        plan,
        "paged-qstore",
        manifest=_tiny_qwen_manifest(),
    )
    assert bundle.graph is not None

    class Engine:
        backend = "paged"
        device = "cpu"
        numerical_contract = plan.numerical_contract
        supported_numerical_contracts = (plan.numerical_contract,)
        store = SimpleNamespace(man=_tiny_qwen_manifest())

        @staticmethod
        def logits_batch(rows):
            return [
                torch.arange(len(row) * 8, dtype=torch.float32).reshape(len(row), 8) for row in rows
            ]

        @classmethod
        def last_logits_batch(cls, rows):
            return torch.stack([logits[-1] for logits in cls.logits_batch(rows)])

    rows = [np.asarray([1, 2, 3]), np.asarray([3, 2, 1])]
    result = execute_lowered_plan(
        Engine(),
        plan,
        bundle.lowered,
        rows,
        graph_compilation=bundle.graph,
    )
    assert result.evidence["graph_dispatch_verified"] is True
    assert result.evidence["graph_compilation_fingerprint"] == bundle.graph.fingerprint
    assert result.evidence["graph_rewrite_ids"] == ("take-last-through-row-wise-linear-head",)

    with pytest.raises(ValueError, match="requires an output-demand certificate"):
        execute_lowered_plan(
            Engine(),
            plan,
            bundle.lowered,
            rows,
            graph_compilation=replace(bundle.graph, rewrite_certificate=None),
        )


def test_qstore_resource_trace_binds_runtime_access_order_to_the_graph():
    plan = _plan(OutputContract.LAST_TOKEN_LOGITS)
    bundle = compile_work_plan(
        plan,
        "paged-qstore",
        manifest=_tiny_qwen_manifest(),
    )
    assert bundle.graph is not None

    class Store:
        man = _tiny_qwen_manifest()

        @staticmethod
        def embed_rows(_name, _rows):
            return None

        @staticmethod
        def matmul(_name, _values):
            return None

        @staticmethod
        def fp32(_name):
            return None

        @staticmethod
        def row_blocks(_name):
            return iter(())

    class Engine:
        backend = "paged"
        device = "cpu"
        numerical_contract = plan.numerical_contract
        supported_numerical_contracts = (plan.numerical_contract,)

        def __init__(self):
            self.store = Store()

        def last_logits_batch(self, rows):
            for parameter in bundle.graph.graph.parameter_refs:
                if parameter.logical_name == "embed":
                    self.store.embed_rows(parameter.logical_name, [0])
                elif parameter.logical_name == "lm_head":
                    self.store.row_blocks(parameter.logical_name)
                elif parameter.kind == "fp32":
                    self.store.fp32(parameter.logical_name)
                else:
                    self.store.matmul(parameter.logical_name, None)
            return torch.zeros((len(rows), 8), dtype=torch.float32)

        def logits_batch(self, rows):
            return [torch.zeros((len(row), 8), dtype=torch.float32) for row in rows]

    trace = trace_qstore_graph_execution(
        Engine(),
        plan,
        bundle.lowered,
        bundle.graph,
        [np.asarray([1, 2, 3]), np.asarray([3, 2, 1])],
    )
    assert trace.complete
    assert trace.exact_order
    assert trace.expected_count == len(_tiny_qwen_manifest()["blocks"])
    assert trace.observed_count == trace.expected_count


def test_output_rewrites_respect_loss_hidden_and_selected_contracts():
    manifest = _tiny_qwen_manifest()
    loss_source, loss_graph, loss_certificate = build_output_demand_graph(
        _plan(OutputContract.LOSS_ONLY),
        manifest,
    )
    assert loss_graph == loss_source
    assert loss_certificate.rewrite_ids == ("identity-loss-requires-full-vocabulary",)
    assert loss_graph.node_map["output.source_full_vocab"].parameters[0].access == "all"

    hidden_source, hidden_graph, hidden_certificate = build_output_demand_graph(
        _plan(OutputContract.HIDDEN_STATE_ONLY),
        manifest,
    )
    assert "output.source_full_vocab" in hidden_source.node_map
    assert "output.source_full_vocab" not in hidden_graph.node_map
    assert hidden_graph.outputs == ("final.hidden",)
    assert hidden_certificate.rewrite_ids == ("backward-slice-unreachable-vocabulary-head",)

    _, selected, _ = build_output_demand_graph(
        _plan(OutputContract.SELECTED_TOKEN_ROWS),
        manifest,
    )
    assert selected.node_map["output.vocab_projection"].parameters[0].row_indices == (
        1,
        4,
    )


def test_unsupported_architectures_fail_closed_unless_resource_graph_is_explicit():
    plan = _plan(OutputContract.LAST_TOKEN_LOGITS)
    manifest = _tiny_qwen_manifest()
    manifest["arch"] = "gpt_neox"

    with pytest.raises(NotImplementedError, match="gpt_neox"):
        build_op_graph(plan, manifest)
    generic = build_op_graph(plan, manifest, allow_generic_manifest=True)
    assert generic.architecture == "generic-manifest:gpt_neox"
    assert len(generic.parameter_refs) == len(manifest["blocks"])


def test_backward_slice_removes_pure_dead_work_and_retains_effect_ancestors():
    pure = backward_slice(_simple_graph())
    assert pure.report.removed_node_ids == ("make-dead", "make-state")
    assert pure.graph.outputs == ("b",)

    effectful = backward_slice(_simple_graph(with_effect=True))
    assert "write-state" in effectful.report.retained_effect_nodes
    assert "make-a" in effectful.report.kept_node_ids
    assert "make-dead" in effectful.report.removed_node_ids


def test_liveness_is_inclusive_and_never_reuses_across_memory_spaces():
    graph = OpGraph(
        model_name="toy",
        model_revision="r",
        store_fingerprint="s",
        architecture="toy",
        numerical_contract="exact",
        values=(
            _value("x", storage=StorageClass.INPUT),
            _value("a"),
            _value("b"),
            _value("gpu-output", storage=StorageClass.OUTPUT, memory_space="gpu"),
            _value("late-cpu"),
        ),
        nodes=(
            OpNode("n0", OpKind.IDENTITY, inputs=("x",), outputs=("a",)),
            OpNode("n1", OpKind.IDENTITY, inputs=("a",), outputs=("b",)),
            OpNode(
                "n2",
                OpKind.IDENTITY,
                inputs=("b",),
                outputs=("gpu-output", "late-cpu"),
            ),
        ),
        inputs=("x",),
        outputs=("gpu-output",),
    )
    allocation = allocate_liveness(graph, alignment_bytes=16)
    lifetimes = {value.value_id: value for value in allocation.lifetimes}

    assert lifetimes["a"].end == lifetimes["b"].start
    assert lifetimes["a"].buffer_id != lifetimes["b"].buffer_id
    assert lifetimes["a"].buffer_id == lifetimes["late-cpu"].buffer_id
    assert lifetimes["gpu-output"].buffer_id != lifetimes["late-cpu"].buffer_id
    assert {buffer.memory_space for buffer in allocation.buffers} == {"cpu", "gpu"}


def test_fusion_refuses_effect_barriers_and_binary_cut_matches_exhaustive():
    graph = _simple_graph(with_effect=True)
    fusion = find_fusion_regions(graph)
    assert "write-state" in fusion.barrier_node_ids
    assert all("write-state" not in region.node_ids for region in fusion.regions)

    sliced = backward_slice(graph).graph
    node_costs = {
        node.node_id: {
            "cpu": float(index + 1),
            "accelerator": float(len(sliced.nodes) - index),
        }
        for index, node in enumerate(sliced.nodes)
    }
    transfer_costs = {value.value_id: 0.75 for value in sliced.values}
    placement = place_binary_backends(
        sliced,
        backend_a="cpu",
        backend_b="accelerator",
        node_costs=node_costs,
        transfer_costs=transfer_costs,
    )

    producer = sliced.producer_map
    consumers = sliced.consumer_map
    exhaustive = []
    for choices in itertools.product(("cpu", "accelerator"), repeat=len(sliced.nodes)):
        assignment = dict(zip((node.node_id for node in sliced.nodes), choices, strict=True))
        cost = sum(node_costs[node_id][backend] for node_id, backend in assignment.items())
        for value_id, producer_id in producer.items():
            for consumer_id in consumers[value_id]:
                if assignment[producer_id] != assignment[consumer_id]:
                    cost += 0.75
        exhaustive.append(cost)
    assert placement.exact
    assert placement.total_cost == pytest.approx(min(exhaustive))


def test_graph_validation_rejects_dangling_and_duplicate_producers():
    values = (
        _value("x", storage=StorageClass.INPUT),
        _value("a"),
    )
    with pytest.raises(ValueError, match="unknown value"):
        OpGraph(
            model_name="toy",
            model_revision="r",
            store_fingerprint="s",
            architecture="toy",
            numerical_contract="exact",
            values=values,
            nodes=(OpNode("bad", OpKind.IDENTITY, inputs=("missing",), outputs=("a",)),),
            inputs=("x",),
            outputs=("a",),
        )
    with pytest.raises(ValueError, match="SSA violation"):
        OpGraph(
            model_name="toy",
            model_revision="r",
            store_fingerprint="s",
            architecture="toy",
            numerical_contract="exact",
            values=values,
            nodes=(
                OpNode("first", OpKind.IDENTITY, inputs=("x",), outputs=("a",)),
                OpNode("second", OpKind.IDENTITY, inputs=("x",), outputs=("a",)),
            ),
            inputs=("x",),
            outputs=("a",),
        )


def test_packed_row_parameters_are_sized_without_explicit_lengths():
    plan = _plan(OutputContract.LAST_TOKEN_LOGITS)
    manifest = _tiny_qwen_manifest()
    for block in manifest["blocks"].values():
        if block.get("kind") != "qrow":
            continue
        block["kind"] = "qrow4"
        block["row_bytes"] = 2
        block["n_groups"] = 1
        block.pop("w_len")
        block.pop("s_len")

    graph = build_op_graph(plan, manifest)
    q = next(parameter for parameter in graph.parameter_refs if parameter.logical_name == "L0.q")
    assert q.physical_bytes == 4 * 2 + 4 * 1 * 4
