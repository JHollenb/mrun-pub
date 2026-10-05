from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from mrun.runtime import (
    ComponentPlacement,
    DecodeWork,
    ExactGreedySpeculativeRuntime,
    FallbackPolicy,
    GreedyBlockVerification,
    GreedyBlockVerifier,
    GreedyBlockVerifyWork,
    GreedyProposalBeginWork,
    GreedyProposalTransactionRuntime,
    MemoryDomain,
    MlxNativeRuntime,
    MlxNativeRuntimeError,
    NativeOutput,
    OutputMode,
    OutputRequest,
    PlacementPlan,
    PrefillWork,
    PromotionStatus,
    ProvisionalStep,
    Residency,
    RuntimeRoute,
    StateObservation,
    StatePlacement,
)


class _DeviceTensor:
    """Synthetic device value that loudly forbids NumPy/host materialization."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __getitem__(self, key: Any) -> _DeviceTensor:
        return _DeviceTensor(self.values[key])

    def __array__(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        raise AssertionError("full logits crossed the synthetic device boundary")


class _Selected:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)

    def item(self) -> Any:
        return self.values.item()

    def tolist(self) -> Any:
        return self.values.tolist()


class _FakeMx:
    def __init__(self) -> None:
        self.argmax_shapes: list[tuple[int, ...]] = []
        self.eval_calls = 0

    @staticmethod
    def array(values: np.ndarray) -> np.ndarray:
        return np.asarray(values)

    def argmax(self, values: _DeviceTensor, *, axis: int) -> _Selected:
        assert isinstance(values, _DeviceTensor)
        assert axis == -1
        self.argmax_shapes.append(values.shape)
        return _Selected(np.argmax(values.values, axis=axis))

    def eval(self, *_values: Any) -> None:
        self.eval_calls += 1


class _Cache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.offset = 0
        self.nbytes = capacity * 8
        self.marker = object()
        self.keys = object()
        self.values = object()

    def append(self, count: int) -> None:
        if self.offset + count > self.capacity:
            raise OverflowError
        self.offset += count

    def trim(self, count: int) -> int:
        if count > self.offset:
            return 0
        self.offset -= count
        return count

    def reset(self, offset: int) -> None:
        self.offset = offset

    def storage_signature(self) -> tuple[int, int]:
        return (id(self.marker), self.capacity)


class _Engine:
    backend = "mlx-component"
    arch = "qwen2"
    context_size = 8
    semantic_token_count = 10
    numerical_contract = "mlx-synthetic-v1"

    def __init__(self) -> None:
        self._mx = _FakeMx()
        self.model_inputs: list[tuple[int, ...]] = []

    def model(self, inputs: np.ndarray, *, cache: tuple[_Cache, ...]) -> _DeviceTensor:
        ids = np.asarray(inputs)[0]
        self.model_inputs.append(tuple(int(value) for value in ids))
        for layer in cache:
            layer.append(len(ids))
        logits = np.full((1, len(ids), self.semantic_token_count), -100.0)
        for position, token in enumerate(ids):
            logits[0, position, (int(token) + 1) % self.semantic_token_count] = 100.0
        return _DeviceTensor(logits)

    def close(self) -> None:
        pass


def _runtime(
    *,
    greedy_block_executor: Any = None,
    greedy_proposal_executor: Any = None,
    greedy_proposal_seal_executor: Any = None,
) -> tuple[MlxNativeRuntime, _Engine]:
    state = StatePlacement(
        state_abi="gqa-kv-v1",
        dtype="bfloat16",
        memory_domain=MemoryDomain.UNIFIED,
        bytes_per_token=16,
        reserved_bytes=16 * 8,
        max_batch_size=1,
        max_context_tokens=8,
    )
    component = ComponentPlacement(
        allocation_id="allocation.body",
        component_ids=("body",),
        roles=("body",),
        codec_id="mlx-affine-q8",
        layout_id="mlx-linear",
        memory_domain=MemoryDomain.UNIFIED,
        residency=Residency.RESIDENT,
        physical_bytes=100,
    )
    placement = PlacementPlan(
        model_fingerprint="a" * 64,
        capability_fingerprint="b" * 64,
        device_fingerprint="c" * 64,
        workload_fingerprint="d" * 64,
        backend_id="mlx-component",
        device_id="metal:0",
        components=(component,),
        state=state,
        workspace_bytes=0,
        headroom_bytes=0,
        model_resident_bytes=100,
        total_reserved_bytes=228,
        memory_budget_bytes=1_000,
        fully_resident=True,
        fallback_policy=FallbackPolicy.DENY,
        performance_claim_valid=True,
    )
    route = RuntimeRoute(
        runtime_id=f"runtime.{uuid4().hex}",
        model_fingerprint="a" * 64,
        capability_fingerprint="b" * 64,
        placement_fingerprint=placement.fingerprint,
        backend_id="mlx-component",
        device_id="metal:0",
        promotion_status=PromotionStatus.EXPERIMENTAL,
    )
    engine = _Engine()
    runtime = MlxNativeRuntime(
        engine,
        route=route,
        placement=placement,
        semantic_token_count=10,
        state_abi="gqa-kv-v1",
        cache_factory=lambda capacity: (_Cache(capacity), _Cache(capacity)),
        greedy_block_executor=greedy_block_executor,
        greedy_proposal_executor=greedy_proposal_executor,
        greedy_proposal_seal_executor=greedy_proposal_seal_executor,
    )
    return runtime, engine


def _prefilled(runtime: MlxNativeRuntime) -> tuple[Any, StateObservation]:
    state = runtime.allocate_state(owner_id="request", batch_size=1, capacity=8)
    parent = state.observe()
    step = runtime.prefill(
        PrefillWork(
            request_ids=("request",),
            token_rows=((1, 2),),
            state=state,
            parent=parent,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    runtime.commit(step, (2,))
    return state, state.observe()


def test_mlx_all_position_argmax_is_transactional_and_never_materializes_logits() -> None:
    runtime, engine = _runtime()
    assert isinstance(runtime, GreedyBlockVerifier)
    state, parent = _prefilled(runtime)

    verification = runtime.verify_greedy_block(
        GreedyBlockVerifyWork(
            request_id="request",
            token_ids=(3, 4, 9),
            state=state,
            parent=parent,
        )
    )

    assert verification.predicted_token_ids == (4, 5, 0)
    assert verification.token_count == 3
    assert state.observe() == parent
    receipt = runtime.commit_greedy_block(verification, 2)
    assert receipt.accepted_counts == (2,)
    assert receipt.after.lengths == (4,)
    assert engine._mx.argmax_shapes == [(10,), (3, 10)]
    telemetry = runtime.telemetry()
    assert telemetry.device_to_host_bytes == 8 + 3 * 8
    extras = dict(telemetry.extra_counters)
    assert extras["greedy_block_verifications"] == 1
    assert extras["greedy_block_tokens"] == 3
    assert extras["greedy_block_accepted_tokens"] == 2
    assert extras["greedy_block_rejected_tokens"] == 1


def test_mlx_greedy_block_abandon_restores_exact_parent_and_is_single_use() -> None:
    runtime, _engine = _runtime()
    state, parent = _prefilled(runtime)
    verification = runtime.verify_greedy_block(
        GreedyBlockVerifyWork(
            request_id="request",
            token_ids=(3, 4),
            state=state,
            parent=parent,
        )
    )

    runtime.abandon_greedy_block(verification)

    assert state.observe() == parent
    with pytest.raises(MlxNativeRuntimeError, match="already been consumed"):
        runtime.abandon_greedy_block(verification)
    assert dict(runtime.telemetry().extra_counters)["greedy_block_abandons"] == 1


def test_mlx_terminal_authorities_cannot_cross_ordinary_and_block_domains() -> None:
    runtime, _engine = _runtime()
    state, parent = _prefilled(runtime)
    verification = runtime.verify_greedy_block(
        GreedyBlockVerifyWork(
            request_id="request",
            token_ids=(3,),
            state=state,
            parent=parent,
        )
    )
    forged_ordinary = ProvisionalStep(
        runtime_id=verification.runtime_id,
        step_id=verification.step_id,
        request_ids=("request",),
        state=state,
        parent=parent,
        token_counts=(1,),
        output=NativeOutput(OutputMode.NEXT_TOKEN_ARGMAX, token_ids=(4,)),
        authority=verification.authority,
    )
    with pytest.raises(TypeError, match="greedy block authority"):
        runtime.commit(forged_ordinary, (1,))
    runtime.abandon_greedy_block(verification)

    parent = state.observe()
    ordinary = runtime.decode(
        DecodeWork(
            request_ids=("request",),
            token_rows=((3,),),
            state=state,
            parent=parent,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    forged_block = GreedyBlockVerification(
        runtime_id=ordinary.runtime_id,
        step_id=ordinary.step_id,
        request_id="request",
        state=state,
        parent=parent,
        input_token_ids=(3,),
        predicted_token_ids=(4,),
        authority=ordinary.authority,
    )
    with pytest.raises(TypeError, match="greedy block authority"):
        runtime.commit_greedy_block(forged_block, 1)
    runtime.abandon(ordinary)


def test_mlx_invalid_block_selection_rolls_back_cache_and_counts_failure() -> None:
    def invalid_executor(ids: tuple[int, ...], caches: tuple[_Cache, ...]) -> tuple[int, ...]:
        for cache in caches:
            cache.append(len(ids))
        return (10,) * len(ids)

    runtime, _engine = _runtime(greedy_block_executor=invalid_executor)
    state, parent = _prefilled(runtime)

    with pytest.raises(MlxNativeRuntimeError, match="escaped the token domain"):
        runtime.verify_greedy_block(
            GreedyBlockVerifyWork(
                request_id="request",
                token_ids=(3, 4),
                state=state,
                parent=parent,
            )
        )

    assert state.observe() == parent
    assert dict(runtime.telemetry().extra_counters)["greedy_block_failures"] == 1


def test_mlx_greedy_block_rejects_foreign_terminal_and_input_domain_authority() -> None:
    runtime, _engine = _runtime()
    foreign, _foreign_engine = _runtime()
    state, parent = _prefilled(runtime)
    verification = runtime.verify_greedy_block(
        GreedyBlockVerifyWork(
            request_id="request",
            token_ids=(3,),
            state=state,
            parent=parent,
        )
    )

    with pytest.raises(MlxNativeRuntimeError, match="another runtime"):
        foreign.commit_greedy_block(verification, 1)
    runtime.abandon_greedy_block(verification)
    assert state.observe() == parent

    with pytest.raises(ValueError, match="semantic token domain"):
        runtime.verify_greedy_block(
            GreedyBlockVerifyWork(
                request_id="request",
                token_ids=(10,),
                state=state,
                parent=parent,
            )
        )
    assert state.observe() == parent


def test_mlx_in_place_proposal_commits_prefix_without_copy_or_replay() -> None:
    runtime, engine = _runtime()
    assert isinstance(runtime, GreedyProposalTransactionRuntime)
    state, parent = _prefilled(runtime)
    transaction = runtime.begin_greedy_proposal(
        GreedyProposalBeginWork(
            request_id="request",
            state=state,
            parent=parent,
        )
    )
    first = runtime.advance_greedy_proposal(transaction, 3)
    second = runtime.advance_greedy_proposal(first, 4)
    third = runtime.advance_greedy_proposal(second, 9)

    assert first.predicted_token_ids == (4,)
    assert third.input_token_ids == (3, 4, 9)
    assert third.predicted_token_ids == (4, 5, 0)
    assert state.observe() == parent
    receipt = runtime.commit_greedy_proposal(third, 2)

    assert receipt.after.lengths == (4,)
    assert engine.model_inputs == [(1, 2), (3,), (4,), (9,)]
    extras = dict(runtime.telemetry().extra_counters)
    assert extras["state_forks"] == 0
    assert extras["greedy_proposal_prefix_copy_bytes"] == 0
    assert extras["greedy_proposal_replay_forwards"] == 0
    assert extras["greedy_proposal_advances"] == 3
    assert extras["greedy_proposal_accepted_tokens"] == 2
    assert extras["greedy_proposal_rejected_tokens"] == 1


def test_mlx_proposal_seal_appends_final_kv_without_argmax_or_host_token() -> None:
    runtime, engine = _runtime()
    state, parent = _prefilled(runtime)
    transaction = runtime.begin_greedy_proposal(
        GreedyProposalBeginWork(
            request_id="request",
            state=state,
            parent=parent,
        )
    )
    transaction = runtime.advance_greedy_proposal(transaction, 3)
    transaction = runtime.advance_greedy_proposal(transaction, 4)
    sealed = runtime.seal_greedy_proposal(transaction, 5)

    assert sealed.sealed
    assert sealed.input_token_ids == (3, 4, 5)
    assert sealed.predicted_token_ids == (4, 5)
    assert engine.model_inputs == [(1, 2), (3,), (4,), (5,)]
    assert engine._mx.argmax_shapes == [(10,), (10,), (10,)]
    before_terminal_bytes = runtime.telemetry().device_to_host_bytes
    receipt = runtime.commit_greedy_proposal(sealed, 3)

    assert receipt.after.lengths == (5,)
    assert before_terminal_bytes == 8 + 2 * 8
    assert dict(runtime.telemetry().extra_counters)["greedy_proposal_seals"] == 1


def test_mlx_proposal_cursor_is_latest_only_foreign_safe_and_single_use() -> None:
    runtime, _engine = _runtime()
    foreign, _foreign_engine = _runtime()
    state, parent = _prefilled(runtime)
    initial = runtime.begin_greedy_proposal(
        GreedyProposalBeginWork(
            request_id="request",
            state=state,
            parent=parent,
        )
    )
    latest = runtime.advance_greedy_proposal(initial, 3)

    with pytest.raises(MlxNativeRuntimeError, match="snapshot is stale"):
        runtime.advance_greedy_proposal(initial, 4)
    with pytest.raises(MlxNativeRuntimeError, match="another runtime"):
        foreign.commit_greedy_proposal(latest, 1)
    sealed = runtime.seal_greedy_proposal(latest, 4)
    with pytest.raises(MlxNativeRuntimeError, match="sealed"):
        runtime.advance_greedy_proposal(sealed, 5)
    runtime.abandon_greedy_proposal(sealed)

    assert state.observe() == parent
    with pytest.raises(MlxNativeRuntimeError, match="already been consumed"):
        runtime.abandon_greedy_proposal(sealed)


def test_mlx_proposal_mid_advance_failure_restores_latest_cursor_for_abandon() -> None:
    def failing_executor(ids: tuple[int, ...], caches: tuple[_Cache, ...]) -> int:
        for cache in caches:
            cache.append(len(ids))
        if ids == (4,):
            raise RuntimeError("injected proposal failure")
        return (ids[-1] + 1) % 10

    runtime, _engine = _runtime(greedy_proposal_executor=failing_executor)
    state, parent = _prefilled(runtime)
    initial = runtime.begin_greedy_proposal(
        GreedyProposalBeginWork(
            request_id="request",
            state=state,
            parent=parent,
        )
    )
    latest = runtime.advance_greedy_proposal(initial, 3)

    with pytest.raises(RuntimeError, match="injected proposal failure"):
        runtime.advance_greedy_proposal(latest, 4)

    assert state.observe() == parent
    runtime.abandon_greedy_proposal(latest)
    assert state.observe() == parent
    extras = dict(runtime.telemetry().extra_counters)
    assert extras["greedy_proposal_advances"] == 1
    assert extras["greedy_proposal_failures"] == 1


def test_mlx_proposal_authority_cannot_terminate_an_ordinary_step() -> None:
    runtime, _engine = _runtime()
    state, parent = _prefilled(runtime)
    transaction = runtime.begin_greedy_proposal(
        GreedyProposalBeginWork(
            request_id="request",
            state=state,
            parent=parent,
        )
    )
    transaction = runtime.advance_greedy_proposal(transaction, 3)
    forged = ProvisionalStep(
        runtime_id=transaction.runtime_id,
        step_id=transaction.step_id,
        request_ids=("request",),
        state=state,
        parent=parent,
        token_counts=(1,),
        output=NativeOutput(OutputMode.NEXT_TOKEN_ARGMAX, token_ids=(4,)),
        authority=transaction.authority,
    )

    with pytest.raises(TypeError, match="proposal authority"):
        runtime.commit(forged, (1,))
    runtime.abandon_greedy_proposal(transaction)


def test_mlx_end_to_end_speculation_has_zero_draft_copy_and_zero_replay() -> None:
    target, target_engine = _runtime()
    draft, draft_engine = _runtime()
    runtime = ExactGreedySpeculativeRuntime(
        target,
        draft,
        semantic_token_count=10,
        proposal_tokens=2,
    )
    state = runtime.allocate_state(owner_id="chat", capacity=8)
    pending = runtime.prefill(state, request_id="request", token_ids=(1, 2)).pending_token_id

    step = runtime.step(state, request_id="request", current_token_id=pending)

    assert step.output_token_ids == (4, 5, 6)
    assert target_engine.model_inputs == [(1, 2), (3, 4, 5)]
    assert draft_engine.model_inputs == [(1, 2), (3,), (4,), (5,)]
    telemetry = runtime.telemetry()
    assert telemetry.optimized_draft_steps == 1
    assert telemetry.proposal_state_forks == 0
    assert telemetry.proposal_prefix_bytes_copied == 0
    assert telemetry.draft_sync_block_calls == 0
    assert telemetry.draft_transaction_advances == 2
    assert telemetry.draft_transaction_seals == 1
    draft_extras = dict(draft.telemetry().extra_counters)
    assert draft_extras["state_forks"] == 0
    assert draft_extras["greedy_block_verifications"] == 0
    assert draft_extras["greedy_proposal_prefix_copy_bytes"] == 0
    assert draft_extras["greedy_proposal_replay_forwards"] == 0
