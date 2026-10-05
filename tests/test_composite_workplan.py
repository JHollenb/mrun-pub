from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.compiler import (
    ExecutionMode,
    OutputContract,
    ProvisionalKVDeltaBinding,
    bind_versioned_kv_state,
    build_dense_qstore_plan,
    build_paged_qstore_plan,
    commit_provisional_kv_delta,
    compile_work_plan,
    execute_lowered_plan,
    lower_work_plan,
    plan_qstore_memory,
)
from mrun.compiler.executable import _native_paged_scratch_executor, _same_runtime_device
from mrun.engine.kernels import paged_forward as pf
from mrun.engine.paged import PagedEngine
from mrun.testing.qstore_identity import (
    install_verified_test_identity,
    verified_test_manifest,
)


def test_runtime_device_matching_normalizes_only_cpu_indices() -> None:
    assert _same_runtime_device("cpu", "cpu")
    assert _same_runtime_device("cpu", "cpu:0")
    assert _same_runtime_device("cpu:7", "cpu")
    assert _same_runtime_device("cuda:0", "cuda:0")
    assert _same_runtime_device("mps", "mps")
    assert not _same_runtime_device("cpu", "cuda:0")
    assert not _same_runtime_device("cuda", "cuda:0")
    assert not _same_runtime_device("cuda:0", "cuda")
    assert not _same_runtime_device("cuda:1", "cuda:0")


def test_native_scratch_executor_identity_rejects_instance_override() -> None:
    engine = object.__new__(PagedEngine)
    assert _native_paged_scratch_executor(engine)

    engine.execute_workplan_stateful = lambda *_args, **_kwargs: None
    assert not _native_paged_scratch_executor(engine)


class _Store:
    compute_dtype = torch.float32
    man = verified_test_manifest(
        {
            "model_name": "tiny-composite",
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


class _Engine:
    backend = "paged"
    device = torch.device("cpu")
    name = "tiny-composite"
    n_layer = 1
    numerical_contract = "paged-qstore-established"
    store = _Store()
    _execution_lock = threading.RLock()

    def capabilities(self):
        return SimpleNamespace(transactional_kv=True, speculative_blocks=True)

    def selected_last_logits_batch(self, rows, token_ids):
        return torch.zeros((len(rows), len(token_ids)), dtype=torch.float32)

    def candidate_logits_batch(self, rows, candidate_token_ids):
        return tuple(
            torch.zeros(len(candidates), dtype=torch.float32) for candidates in candidate_token_ids
        )

    def execute_workplan_stateful(self, plan, rows, state_binding):
        batch = len(rows)
        if plan.output_contract is OutputContract.LAST_TOKEN_LOGITS:
            output = torch.zeros((batch, 8), dtype=torch.float32)
        else:
            output = torch.zeros((batch, len(rows[0]), 8), dtype=torch.float32)
        delta = pf.PagedKVDelta(
            parent_epoch=state_binding.parent_epoch,
            parent_lengths=state_binding.parent_lengths,
            cache_id=state_binding.cache_id,
            token_count=len(rows[0]),
            k=torch.zeros((1, batch, len(rows[0]), 1, 2), dtype=torch.float32),
            v=torch.zeros((1, batch, len(rows[0]), 1, 2), dtype=torch.float32),
        )
        return output, delta


class _DenseEngine(_Engine):
    backend = "dense-qstore-cuda"
    max_seq_len = 8


class _State(pf.BatchedPagedKVCache):
    def __init__(self, lengths: tuple[int, ...], *, capacity: int = 8) -> None:
        super().__init__(1, len(lengths), 1, 2, capacity, torch.device("cpu"))
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.epoch = 3
        self.cache_id = "test-state-cache"


def _rows() -> list[np.ndarray]:
    return [np.asarray([1, 2]), np.asarray([3, 4])]


def test_logical_pages_and_head_rows_follow_output_semantics() -> None:
    engine = _Engine()
    hidden = build_paged_qstore_plan(
        engine,
        _rows(),
        output_contract=OutputContract.HIDDEN_STATE_ONLY,
    )
    selected = build_paged_qstore_plan(
        engine,
        _rows(),
        output_contract=OutputContract.SELECTED_TOKEN_ROWS,
        required_output_rows=(1, 5),
    )
    full = build_paged_qstore_plan(
        engine,
        _rows(),
        output_contract=OutputContract.FULL_LOGITS,
    )

    assert hidden.page_sequence[0] == "embed"
    assert "lm_head" not in hidden.page_sequence
    assert dict(hidden.metadata)["logical_head_access"] == "none"
    assert dict(hidden.metadata)["logical_head_row_count"] == 0
    assert selected.page_sequence[0] == "embed"
    assert selected.page_sequence[-1] == "lm_head"
    assert dict(selected.metadata)["logical_head_access"] == "rows"
    assert dict(selected.metadata)["logical_head_row_count"] == 2
    assert full.page_sequence[-1] == "lm_head"
    assert dict(full.metadata)["logical_head_access"] == "all"
    assert dict(full.metadata)["logical_head_row_count"] == 8


def test_composite_contract_metadata_does_not_promote_legacy_identity() -> None:
    engine = _Engine()
    engine.store = _Store()
    engine.store.content_identity_verified = False
    engine.store.identity_status = "component-graph-verified-source-legacy"
    engine.store.source_checkpoint_sha256 = None
    engine.store.derived_store_sha256 = None
    engine.component_output_contract = "selected_rows"
    engine.composite_store = SimpleNamespace(
        composite_fingerprint_sha256="c" * 64,
        vocab_manifest_sha256="d" * 64,
        vocab=SimpleNamespace(token_count=8),
        _providers={"body": SimpleNamespace(store=SimpleNamespace(_cache_policy="lru"))},
        snapshot=lambda: {
            "component_cache_budget_bytes": {"body": 0},
            "cache_budget_bytes": 0,
        },
        ring_allocated_bytes=lambda: 0,
        assert_content_identity_unchanged=lambda: None,
    )

    plan = build_paged_qstore_plan(
        engine,
        _rows(),
        output_contract=OutputContract.SELECTED_TOKEN_ROWS,
        required_output_rows=(1, 5),
    )
    metadata = dict(plan.metadata)

    assert not plan.content_identity_verified
    assert metadata["component_output_contract"] == "selected_rows"
    assert metadata["component_graph_fingerprint"] == "c" * 64
    assert metadata["vocab_manifest_sha256"] == "d" * 64

    engine.component_output_contract = "full_logits"
    with pytest.raises(RuntimeError, match="component output contract"):
        execute_lowered_plan(
            engine,
            plan,
            lower_work_plan(plan, "paged"),
            _rows(),
        )


def test_composite_residency_fallback_does_not_require_snapshot_fixture() -> None:
    engine = _Engine()
    engine.component_output_contract = "selected_rows"
    body_store = SimpleNamespace(_cache_policy="lru", _cache_budget=17)
    engine.composite_store = SimpleNamespace(
        composite_fingerprint_sha256="c" * 64,
        vocab_manifest_sha256="d" * 64,
        vocab=SimpleNamespace(token_count=8),
        _providers={"body": SimpleNamespace(store=body_store)},
        assert_content_identity_unchanged=lambda: None,
    )
    plan = build_paged_qstore_plan(
        engine,
        _rows(),
        output_contract=OutputContract.SELECTED_TOKEN_ROWS,
        required_output_rows=(1, 5),
    )

    assert dict(plan.metadata)["weight_cache_budget_bytes"] == 17
    result = execute_lowered_plan(
        engine,
        plan,
        lower_work_plan(plan, "paged"),
        _rows(),
    )
    assert tuple(result.outputs.shape) == (2, 2)


def test_stateful_plan_invariants_lowering_and_capacity_memory() -> None:
    plan = build_paged_qstore_plan(
        _Engine(),
        _rows(),
        execution_mode=ExecutionMode.DECODE,
        request_ids=("alpha", "beta"),
        kv_capacity=8,
    )
    metadata = dict(plan.metadata)

    assert plan.request_slots == (0, 1)
    assert plan.kv_read_handles == plan.kv_write_handles == ("kv:alpha", "kv:beta")
    assert not plan.prefix_state_ids
    assert metadata["kv_capacity"] == 8
    assert metadata["kv_dtype"] == "fp32"
    assert metadata["op_graph_status"] == "not-modeled-v1"
    assert metadata["work_floor_status"] == "not-modeled-v1"
    assert {"persistent-kv", "transactional-kv"} <= set(plan.structured_operator_ids)

    operations = [step.operation for step in lower_work_plan(plan, "paged").steps]
    assert "bind_versioned_kv_state" in operations
    assert "emit_provisional_kv_delta" in operations
    assert operations.index("bind_versioned_kv_state") < operations.index(
        "fused_batch_paged_transformer_region"
    )
    assert operations.index("emit_provisional_kv_delta") > operations.index(
        "last_token_full_vocabulary_head"
    )

    bundle = compile_work_plan(plan, "paged", manifest=_Store.man)
    assert bundle.graph is None
    assert bundle.work_floor is None
    assert bundle.memory is not None
    assert bundle.memory.kv_allocated_bytes == 1 * 2 * 8 * 1 * 2 * 2 * 4

    with pytest.raises(ValueError, match="kv_dtype"):
        replace(
            plan,
            metadata=tuple((key, value) for key, value in plan.metadata if key != "kv_dtype"),
        )

    with pytest.raises(ValueError, match="contiguous"):
        replace(plan, request_slots=(1, 0))
    with pytest.raises(ValueError, match="match row by row"):
        replace(plan, kv_write_handles=("kv:alpha", "different"))
    with pytest.raises(ValueError, match="non-empty"):
        replace(plan, kv_read_handles=("", "kv:beta"), kv_write_handles=("", "kv:beta"))
    with pytest.raises(ValueError, match="do not admit prefix"):
        replace(plan, prefix_state_ids=("prefix-a", "prefix-b"))
    with pytest.raises(ValueError, match="loss_only"):
        build_paged_qstore_plan(
            _Engine(),
            _rows(),
            execution_mode=ExecutionMode.PREFILL,
            output_contract=OutputContract.LOSS_ONLY,
            kv_capacity=8,
        )


def test_stateful_memory_uses_physical_backend_kv_dtype() -> None:
    paged = _Engine()
    paged.store = _Store()
    paged.store.compute_dtype = torch.bfloat16
    paged_plan = build_paged_qstore_plan(
        paged,
        _rows(),
        execution_mode=ExecutionMode.PREFILL,
        kv_capacity=8,
    )
    paged_memory = compile_work_plan(paged_plan, "paged", manifest=paged.store.man).memory

    dense = _DenseEngine()
    dense.store = _Store()
    dense.store.compute_dtype = torch.bfloat16
    dense_plan = build_dense_qstore_plan(
        dense,
        _rows(),
        execution_mode=ExecutionMode.PREFILL,
        kv_capacity=8,
    )
    dense_memory = plan_qstore_memory(dense_plan, dense.store.man)

    assert dict(paged_plan.metadata)["kv_dtype"] == "fp32"
    assert paged_memory is not None
    assert paged_memory.kv_allocated_bytes == 1 * 2 * 8 * 1 * 2 * 2 * 4
    assert dict(dense_plan.metadata)["kv_dtype"] == "bf16"
    assert dense_memory is not None
    assert dense_memory.kv_allocated_bytes == 1 * 2 * 8 * 1 * 2 * 2 * 2


def test_compact_read_and_candidate_output_account_for_actual_row_work() -> None:
    engine = _Engine()
    hidden = build_paged_qstore_plan(
        engine,
        _rows(),
        output_contract=OutputContract.HIDDEN_STATE_ONLY,
    )
    selected = build_paged_qstore_plan(
        engine,
        _rows(),
        output_contract=OutputContract.SELECTED_TOKEN_ROWS,
        required_output_rows=(1, 5),
    )
    full = build_paged_qstore_plan(engine, _rows(), output_contract=OutputContract.FULL_LOGITS)
    candidate = build_paged_qstore_plan(
        engine,
        _rows(),
        output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        candidate_token_ids=((0, 1, 2), (3, 4, 5)),
    )

    hidden_memory = plan_qstore_memory(hidden, engine.store.man)
    selected_memory = plan_qstore_memory(selected, engine.store.man)
    full_memory = plan_qstore_memory(full, engine.store.man)
    candidate_memory = plan_qstore_memory(candidate, engine.store.man)

    # The tied 8x4 embed/head occupies 64 compact bytes. Ingress addresses B*T=4 rows
    # (32 bytes), then selected U reads only two rows (16 bytes) while full U scans 64.
    assert (
        selected_memory.estimated_compact_read_bytes - hidden_memory.estimated_compact_read_bytes
        == 16
    )
    assert (
        full_memory.estimated_compact_read_bytes - hidden_memory.estimated_compact_read_bytes == 64
    )
    # Current candidate pushdown materializes [B, |global union|], not merely sum(row sizes).
    assert candidate_memory.output_bytes == 2 * 6 * 4


def test_stateful_lowered_cache_key_binds_request_handles_until_templates_exist() -> None:
    first = build_paged_qstore_plan(
        _Engine(),
        _rows(),
        execution_mode=ExecutionMode.DECODE,
        request_ids=("first-a", "first-b"),
        kv_capacity=8,
    )
    second = build_paged_qstore_plan(
        _Engine(),
        _rows(),
        execution_mode=ExecutionMode.DECODE,
        request_ids=("second-a", "second-b"),
        kv_capacity=8,
    )

    assert first.executable_contract_fingerprint == second.executable_contract_fingerprint
    assert (
        lower_work_plan(first, "paged").executable_key
        != lower_work_plan(second, "paged").executable_key
    )


def test_stateful_execution_returns_then_explicitly_commits_provisional_delta() -> None:
    engine = _Engine()
    rows = _rows()
    plan = build_paged_qstore_plan(
        engine,
        rows,
        execution_mode=ExecutionMode.DECODE,
        kv_capacity=8,
    )
    lowered = lower_work_plan(plan, "paged")
    state = _State((1, 2))
    binding = bind_versioned_kv_state(state, plan.kv_read_handles)

    result = execute_lowered_plan(
        engine,
        plan,
        lowered,
        rows,
        state_binding=binding,
    )

    assert result.outputs.shape == (2, 8)
    assert result.provisional_delta is not None
    assert result.provisional_delta.plan_fingerprint == plan.fingerprint
    assert result.provisional_delta.parent_epoch == 3
    assert result.provisional_delta.parent_lengths == (1, 2)
    assert state.epoch == 3
    assert state.lengths.tolist() == [1, 2]
    assert result.evidence["provisional_kv_emitted"] is True
    assert result.evidence["provisional_plan_fingerprint"] == plan.fingerprint

    assert commit_provisional_kv_delta(
        binding,
        result.provisional_delta,
        (2, 1),
        plan=plan,
    ) == (2, 1)
    assert state.epoch == 4
    assert state.lengths.tolist() == [3, 3]
    with pytest.raises(RuntimeError, match="stale"):
        commit_provisional_kv_delta(
            binding,
            result.provisional_delta,
            (0, 0),
            plan=plan,
        )


def test_state_binding_mode_semantics_and_auto_commit_fail_closed() -> None:
    engine = _Engine()
    rows = _rows()
    decode = build_paged_qstore_plan(
        engine,
        rows,
        execution_mode=ExecutionMode.DECODE,
        kv_capacity=8,
    )
    prefill = build_paged_qstore_plan(
        engine,
        rows,
        execution_mode=ExecutionMode.PREFILL,
        kv_capacity=8,
    )

    with pytest.raises(ValueError, match="committed prefix"):
        bind_versioned_kv_state(_State((0, 0)), decode.kv_read_handles).validate_for_plan(decode)
    with pytest.raises(ValueError, match="empty committed"):
        bind_versioned_kv_state(_State((1, 1)), prefill.kv_read_handles).validate_for_plan(prefill)

    class _AutoCommitEngine(_Engine):
        def execute_workplan_stateful(self, plan, runtime_rows, state_binding):
            output, delta = super().execute_workplan_stateful(
                plan,
                runtime_rows,
                state_binding,
            )
            state_binding.state.lengths += 1
            state_binding.state.epoch += 1
            return output, delta

    state = _State((1, 2))
    k_before = state.k.clone()
    v_before = state.v.clone()
    lengths_before = state.lengths.copy()
    epoch_before = state.epoch
    binding = bind_versioned_kv_state(state, decode.kv_read_handles)
    with pytest.raises(RuntimeError, match="auto-committed"):
        execute_lowered_plan(
            _AutoCommitEngine(),
            decode,
            lower_work_plan(decode, "paged"),
            rows,
            state_binding=binding,
        )
    assert state._poisoned_reason is not None
    assert state.epoch == epoch_before
    assert np.array_equal(state.lengths, lengths_before)
    assert torch.equal(state.k, k_before)
    assert torch.equal(state.v, v_before)
    with pytest.raises(RuntimeError, match="poisoned"):
        binding.validate_for_plan(decode)


def test_generic_stateful_adapter_raw_data_corruption_is_rolled_back_and_poisoned() -> None:
    class _RawDataCorruptingEngine(_Engine):
        def execute_workplan_stateful(self, plan, runtime_rows, state_binding):
            output, delta = super().execute_workplan_stateful(
                plan,
                runtime_rows,
                state_binding,
            )
            # Deliberately bypass Tensor._version. The compatibility rollback's byte witness
            # must still detect this unsupported adapter mutation.
            state_binding.state.k.data.add_(1.0)
            return output, delta

    engine = _RawDataCorruptingEngine()
    rows = _rows()
    plan = build_paged_qstore_plan(
        engine,
        rows,
        execution_mode=ExecutionMode.DECODE,
        kv_capacity=8,
    )
    state = _State((1, 2))
    before = (state.k.clone(), state.v.clone(), state.lengths.copy(), state.epoch)
    binding = bind_versioned_kv_state(state, plan.kv_read_handles)

    with pytest.raises(RuntimeError, match="mutated or auto-committed"):
        execute_lowered_plan(
            engine,
            plan,
            lower_work_plan(plan, "paged"),
            rows,
            state_binding=binding,
        )

    assert state._poisoned_reason is not None
    assert torch.equal(state.k, before[0])
    assert torch.equal(state.v, before[1])
    assert np.array_equal(state.lengths, before[2])
    assert state.epoch == before[3]


def test_state_binding_is_rechecked_after_transaction_lock_acquisition() -> None:
    class _ReplaceArenaOnSecondAcquire:
        def __init__(self, state: _State) -> None:
            self._inner = threading.RLock()
            self._state = state
            self._acquisitions = 0

        def acquire(self, *args, **kwargs):
            acquired = self._inner.acquire(*args, **kwargs)
            if acquired:
                self._acquisitions += 1
                if self._acquisitions == 2:
                    self._state.k = torch.zeros_like(self._state.k)
            return acquired

        def release(self) -> None:
            self._inner.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self.release()

    class _DispatchSpy(_Engine):
        def __init__(self) -> None:
            self.dispatched = False

        def execute_workplan_stateful(self, plan, runtime_rows, state_binding):
            self.dispatched = True
            return super().execute_workplan_stateful(plan, runtime_rows, state_binding)

    engine = _DispatchSpy()
    rows = _rows()
    plan = build_paged_qstore_plan(
        engine,
        rows,
        execution_mode=ExecutionMode.DECODE,
        kv_capacity=8,
    )
    state = _State((1, 2))
    state._lock = _ReplaceArenaOnSecondAcquire(state)
    binding = bind_versioned_kv_state(state, plan.kv_read_handles)

    with pytest.raises(RuntimeError, match="backing storage changed"):
        execute_lowered_plan(
            engine,
            plan,
            lower_work_plan(plan, "paged"),
            rows,
            state_binding=binding,
        )
    assert not engine.dispatched


def test_stateful_dispatch_rejects_autograd_and_output_delta_aliasing() -> None:
    rows = _rows()

    class _AutogradEngine(_Engine):
        def execute_workplan_stateful(self, plan, runtime_rows, state_binding):
            output, delta = super().execute_workplan_stateful(plan, runtime_rows, state_binding)
            return output.requires_grad_(True), delta

    autograd_engine = _AutogradEngine()
    autograd_plan = build_paged_qstore_plan(
        autograd_engine,
        rows,
        execution_mode=ExecutionMode.PREFILL,
        kv_capacity=2,
    )
    autograd_state = _State((0, 0), capacity=2)
    with pytest.raises(RuntimeError, match="detached from autograd"):
        execute_lowered_plan(
            autograd_engine,
            autograd_plan,
            lower_work_plan(autograd_plan, "paged"),
            rows,
            state_binding=bind_versioned_kv_state(
                autograd_state,
                autograd_plan.kv_read_handles,
            ),
        )

    class _OutputAliasEngine(_Engine):
        def execute_workplan_stateful(self, plan, runtime_rows, state_binding):
            shared = torch.zeros(16, dtype=torch.float32)
            delta = pf.PagedKVDelta(
                parent_epoch=state_binding.parent_epoch,
                parent_lengths=state_binding.parent_lengths,
                cache_id=state_binding.cache_id,
                token_count=len(runtime_rows[0]),
                k=shared[:8].view(1, 2, 2, 1, 2),
                v=torch.zeros((1, 2, 2, 1, 2), dtype=torch.float32),
            )
            return shared.view(2, 8), delta

    alias_engine = _OutputAliasEngine()
    alias_plan = build_paged_qstore_plan(
        alias_engine,
        rows,
        execution_mode=ExecutionMode.PREFILL,
        kv_capacity=2,
    )
    alias_state = _State((0, 0), capacity=2)
    with pytest.raises(ValueError, match="output scratch"):
        execute_lowered_plan(
            alias_engine,
            alias_plan,
            lower_work_plan(alias_plan, "paged"),
            rows,
            state_binding=bind_versioned_kv_state(alias_state, alias_plan.kv_read_handles),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required_output_rows", (True,)),
        ("required_output_rows", ("1",)),
        ("candidate_token_ids", ((0, "1"), (2, 3))),
        ("candidate_token_ids", ((0, True), (2, 3))),
    ],
)
def test_workplan_adapter_rejects_lossy_output_id_coercion(field: str, value: object) -> None:
    kwargs = {
        "output_contract": (
            OutputContract.SELECTED_TOKEN_ROWS
            if field == "required_output_rows"
            else OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
        ),
        field: value,
    }
    with pytest.raises(TypeError, match="integers"):
        build_paged_qstore_plan(_Engine(), _rows(), **kwargs)


def test_workplan_adapter_requires_concrete_candidate_sequences_and_boolean_capabilities() -> None:
    with pytest.raises(TypeError, match="integer sequence"):
        build_paged_qstore_plan(
            _Engine(),
            _rows(),
            output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            candidate_token_ids=(iter((0, 1)), (2, 3)),
        )

    class _TruthyCapabilityEngine(_Engine):
        def capabilities(self):
            return SimpleNamespace(transactional_kv="yes", speculative_blocks=1)

    with pytest.raises(NotImplementedError, match="does not advertise"):
        build_paged_qstore_plan(
            _TruthyCapabilityEngine(),
            _rows(),
            execution_mode=ExecutionMode.DECODE,
            kv_capacity=8,
        )


def test_stateful_candidate_output_rejects_bytes_as_an_integer_sequence() -> None:
    class _ByteCandidateEngine(_Engine):
        def execute_workplan_stateful(self, plan, runtime_rows, state_binding):
            _output, delta = super().execute_workplan_stateful(
                plan,
                runtime_rows,
                state_binding,
            )
            outputs = tuple(
                {
                    "winner_token_id": candidates[0],
                    "runner_up_token_id": candidates[1],
                    "winner_logit": 1.0,
                    "runner_up_logit": 0.0,
                    "margin": 1.0,
                    "candidate_token_ids": bytes(candidates),
                }
                for candidates in plan.candidate_token_ids
            )
            return outputs, delta

    engine = _ByteCandidateEngine()
    plan = build_paged_qstore_plan(
        engine,
        _rows(),
        execution_mode=ExecutionMode.PREFILL,
        output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
        candidate_token_ids=((0, 1), (2, 3)),
        kv_capacity=2,
    )
    state = _State((0, 0), capacity=2)

    with pytest.raises(RuntimeError, match="non-string sequence"):
        execute_lowered_plan(
            engine,
            plan,
            lower_work_plan(plan, "paged"),
            _rows(),
            state_binding=bind_versioned_kv_state(state, plan.kv_read_handles),
        )


def test_stateful_lowering_names_exact_cache_and_storage_identity_fields() -> None:
    plan = build_paged_qstore_plan(
        _Engine(),
        _rows(),
        execution_mode=ExecutionMode.DECODE,
        kv_capacity=8,
    )
    lowered = lower_work_plan(plan, "paged")
    bind_step = next(step for step in lowered.steps if step.operation == "bind_versioned_kv_state")
    emit_step = next(
        step for step in lowered.steps if step.operation == "emit_provisional_kv_delta"
    )
    bind_params = dict(bind_step.as_dict()["params"])
    emit_params = dict(emit_step.as_dict()["params"])

    assert bind_params["adapter_abi"] == "mrun-paged-scratch-only-v1"
    assert bind_params["cache_identity_source"] == "VersionedKVStateBinding.cache_id"
    assert bind_params["identity_fields"] == [
        "cache_id",
        "storage_signature",
        "epoch",
        "lengths",
    ]
    assert emit_params["parent_version_fields"] == [
        "cache_id",
        "storage_signature",
        "epoch",
        "lengths",
    ]


def test_explicit_commit_helper_supports_the_paged_cache_contract() -> None:
    cache = pf.BatchedPagedKVCache(
        nL=1,
        B=2,
        nKV=1,
        hd=2,
        capacity=8,
        device="cpu",
    )
    cache.lengths = np.asarray([1, 2], dtype=np.int64)
    cache.epoch = 3
    binding = bind_versioned_kv_state(cache, ("kv:a", "kv:b"))
    raw_delta = pf.PagedKVDelta(
        parent_epoch=3,
        parent_lengths=(1, 2),
        cache_id=cache.cache_id,
        k=torch.ones((1, 2, 2, 1, 2), dtype=torch.float32),
        v=torch.full((1, 2, 2, 1, 2), 2.0, dtype=torch.float32),
        token_count=2,
    )
    provisional = ProvisionalKVDeltaBinding(
        state=cache,
        handles=binding.handles,
        delta=raw_delta,
        plan_fingerprint="a" * 64,
        parent_epoch=3,
        parent_lengths=(1, 2),
        token_count=2,
        capacity=8,
        cache_id=cache.cache_id,
        tensor_signature=raw_delta.tensor_signature,
        state_storage_signature=binding.storage_signature,
    )

    plan = build_paged_qstore_plan(
        _Engine(),
        _rows(),
        execution_mode=ExecutionMode.DECODE,
        request_ids=("a", "b"),
        kv_capacity=8,
    )
    provisional = replace(provisional, plan_fingerprint=plan.fingerprint)
    assert commit_provisional_kv_delta(
        binding,
        provisional,
        (2, 1),
        plan=plan,
    ) == (2, 1)
    assert cache.epoch == 4
    assert cache.lengths.tolist() == [3, 3]
    assert torch.equal(cache.k[:, 0, 1:3], raw_delta.k[:, 0, :2])
    assert torch.equal(cache.v[:, 1, 2:3], raw_delta.v[:, 1, :1])


def test_state_binding_detects_backing_arena_replacement() -> None:
    state = _State((1, 2))
    plan = build_paged_qstore_plan(
        _Engine(),
        _rows(),
        execution_mode=ExecutionMode.DECODE,
        kv_capacity=8,
    )
    binding = bind_versioned_kv_state(state, plan.kv_read_handles)
    state.k = torch.zeros_like(state.k)

    with pytest.raises(RuntimeError, match="backing storage changed"):
        binding.validate_for_plan(plan)


def test_provisional_delta_cannot_rebind_to_replaced_committed_storage() -> None:
    engine = _Engine()
    plan = build_paged_qstore_plan(
        engine,
        _rows(),
        execution_mode=ExecutionMode.DECODE,
        kv_capacity=8,
    )
    state = _State((1, 2))
    original_binding = bind_versioned_kv_state(state, plan.kv_read_handles)
    result = execute_lowered_plan(
        engine,
        plan,
        lower_work_plan(plan, "paged"),
        _rows(),
        state_binding=original_binding,
    )
    assert result.provisional_delta is not None

    state.k = torch.zeros_like(state.k)
    state.v = torch.zeros_like(state.v)
    rebound = bind_versioned_kv_state(state, plan.kv_read_handles)
    with pytest.raises(ValueError, match="different committed KV storage"):
        commit_provisional_kv_delta(
            rebound,
            result.provisional_delta,
            (1, 1),
            plan=plan,
        )


def test_state_binding_rejects_lossy_version_type_coercion() -> None:
    with pytest.raises(TypeError, match="sequence of strings"):
        bind_versioned_kv_state(_State((1, 2)), "ab")

    state = _State((1, 2))
    state.epoch = True
    with pytest.raises(TypeError, match="epoch must be an integer"):
        bind_versioned_kv_state(state, ("kv:a", "kv:b"))

    state = _State((1, 2))
    state.lengths = np.asarray([1.5, 2.0], dtype=np.float32)
    with pytest.raises(TypeError, match="lengths must be an integer"):
        bind_versioned_kv_state(state, ("kv:a", "kv:b"))


def test_stateful_executor_cannot_return_committed_arena_as_provisional_delta() -> None:
    class _AliasingEngine(_Engine):
        def execute_workplan_stateful(self, plan, rows, state_binding):
            output = torch.zeros((len(rows), 8), dtype=torch.float32)
            state = state_binding.state
            return output, pf.PagedKVDelta(
                parent_epoch=state_binding.parent_epoch,
                parent_lengths=state_binding.parent_lengths,
                cache_id=state_binding.cache_id,
                k=state.k,
                v=state.v,
                token_count=len(rows[0]),
            )

    engine = _AliasingEngine()
    plan = build_paged_qstore_plan(
        engine,
        _rows(),
        execution_mode=ExecutionMode.PREFILL,
        kv_capacity=2,
    )
    state = _State((0, 0), capacity=2)
    binding = bind_versioned_kv_state(state, plan.kv_read_handles)

    with pytest.raises(ValueError, match="committed and provisional KV"):
        execute_lowered_plan(
            engine,
            plan,
            lower_work_plan(plan, "paged"),
            _rows(),
            state_binding=binding,
        )
