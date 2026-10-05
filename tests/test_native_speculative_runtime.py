from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from mrun.runtime import (
    CommitResult,
    DecodeWork,
    ExactGreedySpeculativeRuntime,
    GreedyBlockVerification,
    GreedyBlockVerifyWork,
    GreedyProposalBeginWork,
    GreedyProposalTransaction,
    NativeOutput,
    OutputMode,
    PrefillWork,
    ProvisionalStep,
    RuntimeTelemetry,
    SpeculativeRuntimeError,
    StateForkResult,
    StateObservation,
)


class _Authority:
    def __init__(self, runtime_id: str, step_id: str, state: _State, *, block: bool) -> None:
        self.runtime_id = runtime_id
        self.step_id = step_id
        self.state = state
        self.block = block
        self.consumed = False


class _ProposalAuthority:
    def __init__(self, runtime_id: str, step_id: str, state: _State) -> None:
        self.runtime_id = runtime_id
        self.step_id = step_id
        self.state = state
        self.consumed = False
        self.version = 0
        self.sealed = False
        self.inputs: tuple[int, ...] = ()
        self.predictions: tuple[int, ...] = ()


class _State:
    def __init__(
        self,
        *,
        runtime_id: str,
        owner_id: str,
        capacity: int,
        generation: int,
        tokens: Sequence[int] = (),
        epoch: int = 0,
    ) -> None:
        self.runtime_id = runtime_id
        self.state_id = f"state-{uuid4().hex}"
        self.owner_id = owner_id
        self.capacity = capacity
        self.generation = generation
        self.epoch = epoch
        self.tokens = list(tokens)
        self.pending: tuple[int, ...] = ()
        self.pending_step_id: str | None = None
        self.released = False

    def observe(self) -> StateObservation:
        if self.released:
            raise RuntimeError("released")
        return StateObservation(
            runtime_id=self.runtime_id,
            state_id=self.state_id,
            generation=self.generation,
            epoch=self.epoch,
            lengths=(len(self.tokens),),
            capacity=self.capacity,
            state_abi="fake-kv-v1",
            storage_generation=0,
        )


class _FakeGreedyRuntime:
    def __init__(
        self, name: str, predictor: Callable[[int], int], *, token_count: int = 32
    ) -> None:
        self._runtime_id = f"{name}-{uuid4().hex}"
        self._route = SimpleNamespace(runtime_id=self._runtime_id)
        self._predictor = predictor
        self._token_count = token_count
        self._states: dict[str, _State] = {}
        self._generation = 1
        self.fail_verify_once = False
        self.fail_block_commit_once = False
        self.fail_proposal_advance_once = False
        self.fail_proposal_seal_once = False
        self.fail_proposal_commit_once = False
        self.block_abandons = 0
        self.block_verifications = 0
        self.fork_calls = 0
        self.proposal_begins = 0
        self.proposal_advances = 0
        self.proposal_seals = 0
        self.proposal_commits = 0
        self.proposal_abandons = 0
        self.releases = 0
        self.closed = False

    @property
    def route(self) -> Any:
        return self._route

    def _state(self, state: Any) -> _State:
        if not isinstance(state, _State) or state.runtime_id != self._runtime_id:
            raise RuntimeError("foreign state")
        if self._states.get(state.state_id) is not state or state.released:
            raise RuntimeError("stale state")
        return state

    def allocate_state(self, *, owner_id: str, batch_size: int, capacity: int) -> _State:
        if batch_size != 1:
            raise ValueError("B1")
        state = _State(
            runtime_id=self._runtime_id,
            owner_id=owner_id,
            capacity=capacity,
            generation=self._generation,
        )
        self._generation += 1
        self._states[state.state_id] = state
        return state

    def fork_state(
        self,
        source: Any,
        *,
        parent: StateObservation,
        owner_id: str,
        capacity: int,
    ) -> StateForkResult:
        resolved = self._state(source)
        if resolved.observe() != parent or resolved.pending_step_id is not None:
            raise RuntimeError("stale fork")
        forked = _State(
            runtime_id=self._runtime_id,
            owner_id=owner_id,
            capacity=capacity,
            generation=max(self._generation, parent.generation + 1),
            tokens=resolved.tokens,
            epoch=1,
        )
        self._generation = forked.generation + 1
        self._states[forked.state_id] = forked
        self.fork_calls += 1
        return StateForkResult(
            runtime_id=self._runtime_id,
            source=parent,
            forked=forked.observe(),
            state=forked,
            state_bytes_copied=len(resolved.tokens) * 8,
        )

    def _execute(self, work: PrefillWork | DecodeWork) -> ProvisionalStep:
        state = self._state(work.state)
        if state.observe() != work.parent or state.pending_step_id is not None:
            raise RuntimeError("stale work")
        ids = tuple(work.token_rows[0])
        step_id = f"step-{uuid4().hex}"
        state.pending = ids
        state.pending_step_id = step_id
        authority = _Authority(self._runtime_id, step_id, state, block=False)
        return ProvisionalStep(
            runtime_id=self._runtime_id,
            step_id=step_id,
            request_ids=work.request_ids,
            state=state,
            parent=work.parent,
            token_counts=(len(ids),),
            output=NativeOutput(
                OutputMode.NEXT_TOKEN_ARGMAX,
                token_ids=(self._predictor(ids[-1]) % self._token_count,),
            ),
            authority=authority,
        )

    def prefill(self, work: PrefillWork) -> ProvisionalStep:
        return self._execute(work)

    def decode(self, work: DecodeWork) -> ProvisionalStep:
        return self._execute(work)

    def _ordinary_authority(self, step: ProvisionalStep) -> _Authority:
        authority = step.authority
        if not isinstance(authority, _Authority) or authority.block or authority.consumed:
            raise RuntimeError("invalid ordinary authority")
        return authority

    def commit(self, step: ProvisionalStep, accepted_counts: Sequence[int]) -> CommitResult:
        authority = self._ordinary_authority(step)
        accepted = tuple(accepted_counts)[0]
        return self._commit(
            authority.state,
            authority,
            parent=step.parent,
            accepted=accepted,
        )

    def abandon(self, step: ProvisionalStep) -> None:
        authority = self._ordinary_authority(step)
        self._abandon(authority.state, authority, step.parent)

    def verify_greedy_block(self, work: GreedyBlockVerifyWork) -> GreedyBlockVerification:
        if self.fail_verify_once:
            self.fail_verify_once = False
            raise RuntimeError("injected verify failure")
        state = self._state(work.state)
        self.block_verifications += 1
        if state.observe() != work.parent or state.pending_step_id is not None:
            raise RuntimeError("stale verification")
        step_id = f"verify-{uuid4().hex}"
        state.pending = tuple(work.token_ids)
        state.pending_step_id = step_id
        authority = _Authority(self._runtime_id, step_id, state, block=True)
        return GreedyBlockVerification(
            runtime_id=self._runtime_id,
            step_id=step_id,
            request_id=work.request_id,
            state=state,
            parent=work.parent,
            input_token_ids=work.token_ids,
            predicted_token_ids=tuple(
                self._predictor(token) % self._token_count for token in work.token_ids
            ),
            authority=authority,
        )

    def _block_authority(self, verification: GreedyBlockVerification) -> _Authority:
        authority = verification.authority
        if not isinstance(authority, _Authority) or not authority.block or authority.consumed:
            raise RuntimeError("invalid block authority")
        return authority

    def commit_greedy_block(
        self,
        verification: GreedyBlockVerification,
        accepted_input_count: int,
    ) -> CommitResult:
        if self.fail_block_commit_once:
            self.fail_block_commit_once = False
            raise RuntimeError("injected block commit failure")
        authority = self._block_authority(verification)
        return self._commit(
            authority.state,
            authority,
            parent=verification.parent,
            accepted=accepted_input_count,
        )

    def abandon_greedy_block(self, verification: GreedyBlockVerification) -> None:
        authority = self._block_authority(verification)
        self._abandon(authority.state, authority, verification.parent)
        self.block_abandons += 1

    def begin_greedy_proposal(
        self,
        work: GreedyProposalBeginWork,
    ) -> GreedyProposalTransaction:
        state = self._state(work.state)
        if state.observe() != work.parent or state.pending_step_id is not None:
            raise RuntimeError("stale proposal begin")
        step_id = f"proposal-{uuid4().hex}"
        authority = _ProposalAuthority(self._runtime_id, step_id, state)
        state.pending = ()
        state.pending_step_id = step_id
        self.proposal_begins += 1
        return GreedyProposalTransaction(
            runtime_id=self._runtime_id,
            step_id=step_id,
            request_id=work.request_id,
            state=state,
            parent=work.parent,
            input_token_ids=(),
            predicted_token_ids=(),
            sealed=False,
            authority=authority,
        )

    def _proposal_authority(
        self,
        transaction: GreedyProposalTransaction,
    ) -> _ProposalAuthority:
        authority = transaction.authority
        if (
            not isinstance(authority, _ProposalAuthority)
            or authority.consumed
            or authority.state is not transaction.state
        ):
            raise RuntimeError("invalid proposal authority")
        if (
            authority.version != transaction.version
            or authority.sealed != transaction.sealed
            or authority.inputs != transaction.input_token_ids
            or authority.predictions != transaction.predicted_token_ids
        ):
            raise RuntimeError("stale proposal snapshot")
        return authority

    def advance_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        input_token_id: int,
    ) -> GreedyProposalTransaction:
        authority = self._proposal_authority(transaction)
        if transaction.sealed:
            raise RuntimeError("sealed proposal")
        if self.fail_proposal_advance_once:
            self.fail_proposal_advance_once = False
            raise RuntimeError("injected proposal advance failure")
        state = authority.state
        if state.observe() != transaction.parent or state.pending_step_id != transaction.step_id:
            raise RuntimeError("stale proposal state")
        inputs = transaction.input_token_ids + (input_token_id,)
        predictions = transaction.predicted_token_ids + (
            self._predictor(input_token_id) % self._token_count,
        )
        next_transaction = GreedyProposalTransaction(
            runtime_id=self._runtime_id,
            step_id=transaction.step_id,
            request_id=transaction.request_id,
            state=state,
            parent=transaction.parent,
            input_token_ids=inputs,
            predicted_token_ids=predictions,
            sealed=False,
            authority=authority,
        )
        state.pending = inputs
        authority.inputs = inputs
        authority.predictions = predictions
        authority.version += 1
        self.proposal_advances += 1
        return next_transaction

    def seal_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        input_token_id: int,
    ) -> GreedyProposalTransaction:
        authority = self._proposal_authority(transaction)
        if transaction.sealed:
            raise RuntimeError("sealed proposal")
        if self.fail_proposal_seal_once:
            self.fail_proposal_seal_once = False
            raise RuntimeError("injected proposal seal failure")
        state = authority.state
        if state.observe() != transaction.parent or state.pending_step_id != transaction.step_id:
            raise RuntimeError("stale proposal state")
        inputs = transaction.input_token_ids + (input_token_id,)
        sealed = GreedyProposalTransaction(
            runtime_id=self._runtime_id,
            step_id=transaction.step_id,
            request_id=transaction.request_id,
            state=state,
            parent=transaction.parent,
            input_token_ids=inputs,
            predicted_token_ids=transaction.predicted_token_ids,
            sealed=True,
            authority=authority,
        )
        state.pending = inputs
        authority.inputs = inputs
        authority.version += 1
        authority.sealed = True
        self.proposal_seals += 1
        return sealed

    def commit_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        accepted_input_count: int,
    ) -> CommitResult:
        if self.fail_proposal_commit_once:
            self.fail_proposal_commit_once = False
            raise RuntimeError("injected proposal commit failure")
        authority = self._proposal_authority(transaction)
        receipt = self._commit(
            authority.state,
            authority,
            parent=transaction.parent,
            accepted=accepted_input_count,
        )
        self.proposal_commits += 1
        return receipt

    def abandon_greedy_proposal(self, transaction: GreedyProposalTransaction) -> None:
        authority = self._proposal_authority(transaction)
        self._abandon(authority.state, authority, transaction.parent)
        self.proposal_abandons += 1

    @staticmethod
    def _commit(
        state: _State,
        authority: _Authority,
        *,
        parent: StateObservation,
        accepted: int,
    ) -> CommitResult:
        if state.observe() != parent or state.pending_step_id != authority.step_id:
            raise RuntimeError("stale commit")
        if accepted < 0 or accepted > len(state.pending):
            raise ValueError("accepted")
        before = state.observe()
        state.tokens.extend(state.pending[:accepted])
        state.pending = ()
        state.pending_step_id = None
        state.epoch += 1
        authority.consumed = True
        return CommitResult(
            runtime_id=state.runtime_id,
            step_id=authority.step_id,
            state_id=state.state_id,
            accepted_counts=(accepted,),
            before=before,
            after=state.observe(),
            state_bytes_written=accepted * 8,
        )

    @staticmethod
    def _abandon(
        state: _State,
        authority: _Authority,
        parent: StateObservation,
    ) -> None:
        if state.observe() != parent or state.pending_step_id != authority.step_id:
            raise RuntimeError("stale abandon")
        state.pending = ()
        state.pending_step_id = None
        authority.consumed = True

    def release_state(self, state: Any) -> None:
        resolved = self._state(state)
        if resolved.pending_step_id is not None:
            raise RuntimeError("pending")
        self._states.pop(resolved.state_id)
        resolved.released = True
        self.releases += 1

    def telemetry(self) -> RuntimeTelemetry:
        return RuntimeTelemetry(
            runtime_id=self._runtime_id,
            route_backend_id="fake",
            model_fingerprint="a" * 64,
            placement_fingerprint="b" * 64,
        )

    def close(self) -> None:
        if self._states:
            raise RuntimeError("live states")
        self.closed = True

    def committed_for_owner(self, owner_id: str) -> tuple[int, ...]:
        matches = [state for state in self._states.values() if state.owner_id == owner_id]
        assert len(matches) == 1
        return tuple(matches[0].tokens)


class _LegacyDraftProxy:
    """Expose the original fork/block API without the optional proposal transaction methods."""

    def __init__(self, runtime: _FakeGreedyRuntime) -> None:
        self.runtime = runtime

    @property
    def route(self) -> Any:
        return self.runtime.route

    def allocate_state(self, **kwargs: Any) -> _State:
        return self.runtime.allocate_state(**kwargs)

    def fork_state(self, source: Any, **kwargs: Any) -> StateForkResult:
        return self.runtime.fork_state(source, **kwargs)

    def prefill(self, work: PrefillWork) -> ProvisionalStep:
        return self.runtime.prefill(work)

    def decode(self, work: DecodeWork) -> ProvisionalStep:
        return self.runtime.decode(work)

    def commit(self, step: ProvisionalStep, accepted_counts: Sequence[int]) -> CommitResult:
        return self.runtime.commit(step, accepted_counts)

    def abandon(self, step: ProvisionalStep) -> None:
        self.runtime.abandon(step)

    def verify_greedy_block(self, work: GreedyBlockVerifyWork) -> GreedyBlockVerification:
        return self.runtime.verify_greedy_block(work)

    def commit_greedy_block(
        self,
        verification: GreedyBlockVerification,
        accepted_input_count: int,
    ) -> CommitResult:
        return self.runtime.commit_greedy_block(verification, accepted_input_count)

    def abandon_greedy_block(self, verification: GreedyBlockVerification) -> None:
        self.runtime.abandon_greedy_block(verification)

    def release_state(self, state: Any) -> None:
        self.runtime.release_state(state)

    def telemetry(self) -> RuntimeTelemetry:
        return self.runtime.telemetry()

    def close(self) -> None:
        self.runtime.close()


class _ProposalOnlyDraftProxy:
    """Expose the optimized draft protocol without the legacy block-replay capability."""

    def __init__(self, runtime: _FakeGreedyRuntime) -> None:
        self.runtime = runtime

    @property
    def route(self) -> Any:
        return self.runtime.route

    def allocate_state(self, **kwargs: Any) -> _State:
        return self.runtime.allocate_state(**kwargs)

    def fork_state(self, source: Any, **kwargs: Any) -> StateForkResult:
        return self.runtime.fork_state(source, **kwargs)

    def prefill(self, work: PrefillWork) -> ProvisionalStep:
        return self.runtime.prefill(work)

    def decode(self, work: DecodeWork) -> ProvisionalStep:
        return self.runtime.decode(work)

    def commit(self, step: ProvisionalStep, accepted_counts: Sequence[int]) -> CommitResult:
        return self.runtime.commit(step, accepted_counts)

    def abandon(self, step: ProvisionalStep) -> None:
        self.runtime.abandon(step)

    def begin_greedy_proposal(
        self,
        work: GreedyProposalBeginWork,
    ) -> GreedyProposalTransaction:
        return self.runtime.begin_greedy_proposal(work)

    def advance_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        input_token_id: int,
    ) -> GreedyProposalTransaction:
        return self.runtime.advance_greedy_proposal(transaction, input_token_id)

    def seal_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        input_token_id: int,
    ) -> GreedyProposalTransaction:
        return self.runtime.seal_greedy_proposal(transaction, input_token_id)

    def commit_greedy_proposal(
        self,
        transaction: GreedyProposalTransaction,
        accepted_input_count: int,
    ) -> CommitResult:
        return self.runtime.commit_greedy_proposal(transaction, accepted_input_count)

    def abandon_greedy_proposal(self, transaction: GreedyProposalTransaction) -> None:
        self.runtime.abandon_greedy_proposal(transaction)

    def release_state(self, state: Any) -> None:
        self.runtime.release_state(state)

    def telemetry(self) -> RuntimeTelemetry:
        return self.runtime.telemetry()

    def close(self) -> None:
        self.runtime.close()


def _runtime_pair(
    *,
    draft_predictor: Callable[[int], int] | None = None,
    proposal_tokens: int = 3,
) -> tuple[ExactGreedySpeculativeRuntime, _FakeGreedyRuntime, _FakeGreedyRuntime]:
    target = _FakeGreedyRuntime("target", lambda token: token + 1)
    draft = _FakeGreedyRuntime("draft", draft_predictor or (lambda token: token + 1))
    runtime = ExactGreedySpeculativeRuntime(
        target,
        draft,
        semantic_token_count=32,
        proposal_tokens=proposal_tokens,
    )
    return runtime, target, draft


def test_exact_speculation_accepts_full_draft_and_emits_target_bonus() -> None:
    runtime, target, draft = _runtime_pair()
    assert runtime.uses_in_place_draft_transactions
    state = runtime.allocate_state(owner_id="chat", capacity=16)
    prefill = runtime.prefill(state, request_id="request", token_ids=(1, 2))

    assert prefill.pending_token_id == 3
    step = runtime.step(state, request_id="request", current_token_id=3)

    assert step.draft_token_ids == (4, 5, 6)
    assert step.target_prediction_ids == (4, 5, 6, 7)
    assert step.accepted_draft_tokens == 3
    assert step.output_token_ids == (4, 5, 6, 7)
    assert step.all_draft_tokens_accepted
    assert target.committed_for_owner("chat.target") == (1, 2, 3, 4, 5, 6)
    assert draft.committed_for_owner("chat.draft") == (1, 2, 3, 4, 5, 6)
    telemetry = runtime.telemetry()
    assert telemetry.draft_proposed_tokens == 3
    assert telemetry.accepted_draft_tokens == 3
    assert telemetry.acceptance_rate == 1.0
    assert telemetry.full_accept_steps == 1
    assert telemetry.bonus_tokens == 1
    assert telemetry.optimized_draft_steps == 1
    assert telemetry.fallback_draft_steps == 0
    assert telemetry.proposal_state_forks == telemetry.proposal_state_releases == 0
    assert telemetry.proposal_prefix_bytes_copied == 0
    assert telemetry.draft_sync_block_calls == 0
    assert telemetry.draft_transaction_advances == 3
    assert telemetry.draft_transaction_seals == 1
    assert telemetry.draft_transaction_commits == 1
    assert draft.fork_calls == 0
    assert draft.block_verifications == 0


def test_partial_acceptance_commits_only_shared_input_prefix_and_fork_is_independent() -> None:
    def draft_predictor(token: int) -> int:
        return {3: 4, 4: 9}.get(token, token + 1)

    runtime, target, draft = _runtime_pair(draft_predictor=draft_predictor)
    state = runtime.allocate_state(owner_id="chat", capacity=16)
    pending = runtime.prefill(state, request_id="request", token_ids=(1, 2)).pending_token_id
    step = runtime.step(state, request_id="request", current_token_id=pending)

    assert step.draft_token_ids == (4, 9, 10)
    assert step.target_prediction_ids == (4, 5, 10, 11)
    assert step.accepted_draft_tokens == 1
    assert step.output_token_ids == (4, 5)
    assert target.committed_for_owner("chat.target") == (1, 2, 3, 4)
    assert draft.committed_for_owner("chat.draft") == (1, 2, 3, 4)
    assert draft.proposal_advances == 3
    assert draft.proposal_seals == 0
    assert draft.proposal_commits == 1
    assert draft.block_verifications == 0

    fork = runtime.fork_state(
        state,
        parent=step.after,
        owner_id="branch",
        capacity=16,
    )
    branch_step = runtime.step(fork.state, request_id="branch-request", current_token_id=5)
    assert branch_step.before.committed_length == 4
    assert state.observe().committed_length == 4
    assert fork.state.observe().committed_length > state.observe().committed_length
    runtime.release_state(fork.state)
    assert runtime.telemetry().partial_accept_steps == 1


def test_target_verifier_failure_abandons_draft_transaction_and_state_remains_usable() -> None:
    runtime, target, draft = _runtime_pair()
    state = runtime.allocate_state(owner_id="chat", capacity=16)
    pending = runtime.prefill(state, request_id="request", token_ids=(1, 2)).pending_token_id
    before = state.observe()
    target.fail_verify_once = True

    with pytest.raises(SpeculativeRuntimeError, match="verification failed"):
        runtime.step(state, request_id="request", current_token_id=pending)

    assert state.observe() == before
    assert target.block_abandons == 0
    assert draft.proposal_abandons == 1
    retry = runtime.step(state, request_id="request", current_token_id=pending)
    assert retry.output_token_ids == (4, 5, 6, 7)
    assert runtime.telemetry().draft_transaction_abandons == 1


def test_split_backend_commit_poison_is_fail_closed_but_releasable() -> None:
    runtime, _target, draft = _runtime_pair()
    state = runtime.allocate_state(owner_id="chat", capacity=16)
    pending = runtime.prefill(state, request_id="request", token_ids=(1, 2)).pending_token_id
    draft.fail_proposal_commit_once = True

    with pytest.raises(SpeculativeRuntimeError, match="split paired state"):
        runtime.step(state, request_id="request", current_token_id=pending)

    observation = state.observe()
    assert observation.poisoned
    assert not observation.synchronized
    with pytest.raises(SpeculativeRuntimeError, match="poisoned"):
        runtime.step(state, request_id="request", current_token_id=4)
    telemetry = runtime.telemetry()
    assert telemetry.states_poisoned == 1
    assert telemetry.split_commit_failures == 1
    runtime.release_state(state)
    assert runtime.telemetry().states_live == 0


def test_draft_advance_failure_rolls_back_transaction_and_allows_retry() -> None:
    runtime, target, draft = _runtime_pair()
    state = runtime.allocate_state(owner_id="chat", capacity=16)
    pending = runtime.prefill(state, request_id="request", token_ids=(1, 2)).pending_token_id
    before = state.observe()
    draft.fail_proposal_advance_once = True

    with pytest.raises(SpeculativeRuntimeError, match="proposal transaction failed"):
        runtime.step(state, request_id="request", current_token_id=pending)

    assert state.observe() == before
    assert draft.proposal_abandons == 1
    assert target.block_verifications == 0
    retry = runtime.step(state, request_id="request", current_token_id=pending)
    assert retry.output_token_ids == (4, 5, 6, 7)


def test_draft_seal_failure_abandons_target_and_draft_without_state_drift() -> None:
    runtime, target, draft = _runtime_pair()
    state = runtime.allocate_state(owner_id="chat", capacity=16)
    pending = runtime.prefill(state, request_id="request", token_ids=(1, 2)).pending_token_id
    before = state.observe()
    draft.fail_proposal_seal_once = True

    with pytest.raises(SpeculativeRuntimeError, match="transaction seal failed"):
        runtime.step(state, request_id="request", current_token_id=pending)

    assert state.observe() == before
    assert target.block_abandons == 1
    assert draft.proposal_abandons == 1
    assert runtime.telemetry().verification_abandons == 1
    retry = runtime.step(state, request_id="request", current_token_id=pending)
    assert retry.output_token_ids == (4, 5, 6, 7)


def test_capacity_tail_falls_back_to_one_target_verified_token() -> None:
    runtime, _target, _draft = _runtime_pair(proposal_tokens=4)
    state = runtime.allocate_state(owner_id="chat", capacity=3)
    pending = runtime.prefill(state, request_id="request", token_ids=(1, 2)).pending_token_id

    step = runtime.step(state, request_id="request", current_token_id=pending)

    assert step.draft_token_ids == ()
    assert step.output_token_ids == (4,)
    assert step.after.committed_length == 3
    telemetry = runtime.telemetry()
    assert telemetry.ordinary_target_steps == 1
    assert telemetry.draft_transaction_advances == 0
    assert telemetry.draft_transaction_seals == 1
    assert telemetry.proposal_prefix_bytes_copied == 0
    assert telemetry.draft_sync_block_calls == 0


def test_step_rejects_a_token_not_selected_by_the_target_state() -> None:
    runtime, _target, _draft = _runtime_pair()
    state = runtime.allocate_state(owner_id="chat", capacity=8)
    prefill = runtime.prefill(state, request_id="request", token_ids=(1, 2))
    before = state.observe()

    with pytest.raises(SpeculativeRuntimeError, match="target-authoritative pending token"):
        runtime.step(state, request_id="request", current_token_id=9)

    assert state.observe() == before
    assert runtime.telemetry().proposal_state_forks == 0
    assert prefill.pending_token_id == before.pending_token_id == 3


def test_legacy_draft_preserves_fork_and_replay_compatibility_lane() -> None:
    target = _FakeGreedyRuntime("target", lambda token: token + 1)
    legacy_runtime = _FakeGreedyRuntime("legacy-draft", lambda token: token + 1)
    draft = _LegacyDraftProxy(legacy_runtime)
    runtime = ExactGreedySpeculativeRuntime(
        target,
        draft,
        semantic_token_count=32,
        proposal_tokens=2,
    )
    assert not runtime.uses_in_place_draft_transactions
    state = runtime.allocate_state(owner_id="chat", capacity=8)
    pending = runtime.prefill(state, request_id="request", token_ids=(1, 2)).pending_token_id

    step = runtime.step(state, request_id="request", current_token_id=pending)

    assert step.output_token_ids == (4, 5, 6)
    telemetry = runtime.telemetry()
    assert telemetry.optimized_draft_steps == 0
    assert telemetry.fallback_draft_steps == 1
    assert telemetry.proposal_state_forks == 1
    assert telemetry.proposal_prefix_bytes_copied > 0
    assert telemetry.draft_sync_block_calls == 1


def test_transactional_draft_does_not_require_legacy_block_replay_capability() -> None:
    target = _FakeGreedyRuntime("target", lambda token: token + 1)
    draft_runtime = _FakeGreedyRuntime("proposal-only-draft", lambda token: token + 1)
    draft = _ProposalOnlyDraftProxy(draft_runtime)
    runtime = ExactGreedySpeculativeRuntime(
        target,
        draft,
        semantic_token_count=32,
        proposal_tokens=2,
    )
    assert runtime.uses_in_place_draft_transactions
    state = runtime.allocate_state(owner_id="chat", capacity=8)
    pending = runtime.prefill(state, request_id="request", token_ids=(1, 2)).pending_token_id

    step = runtime.step(state, request_id="request", current_token_id=pending)

    assert step.output_token_ids == (4, 5, 6)
    assert draft_runtime.block_verifications == 0
    assert runtime.telemetry().draft_sync_block_calls == 0
