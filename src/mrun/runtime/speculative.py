"""Exact-greedy target/draft speculation over transactional native runtimes.

The coordinator never treats a draft token as authoritative.  A capable draft grows an in-place
versioned provisional suffix; older backends retain the isolated-fork compatibility lane.  The
target verifies ``(current, draft...)`` in one teacher-forced block, and both live states commit
exactly ``current`` plus the consecutive target-matching draft prefix.  The final emitted token
is always the target correction (or target bonus when every proposal matched).

Two independent backend commits cannot be made physically atomic.  This module therefore makes
the failure boundary explicit: any ambiguous or split terminal failure permanently poisons the
paired state.  A poisoned state can only be observed for diagnosis or released.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .contracts import (
    CommitResult,
    DecodeWork,
    ForkableModelRuntime,
    GreedyBlockVerification,
    GreedyBlockVerifier,
    GreedyBlockVerifyWork,
    GreedyProposalBeginWork,
    GreedyProposalTransaction,
    GreedyProposalTransactionRuntime,
    OutputMode,
    OutputRequest,
    PrefillWork,
    ProvisionalStep,
    StateHandle,
    StateObservation,
)

SPECULATIVE_RUNTIME_ABI = "mrun-exact-greedy-target-draft-v1"
SPECULATIVE_STATE_SCHEMA = "mrun-speculative-state-observation-v1"


class SpeculativeRuntimeError(RuntimeError):
    """The paired runtime rejected unsafe, stale, or semantically invalid work."""


def _name(value: str, field_name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _strict_nonnegative(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _strict_positive(value: int, field_name: str) -> int:
    normalized = _strict_nonnegative(value, field_name)
    if normalized == 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return normalized


@dataclass(frozen=True, slots=True)
class SpeculativeStateObservation:
    """Tensor-opaque paired target/draft state identity and synchronization status."""

    runtime_id: str
    state_id: str
    generation: int
    epoch: int
    target: StateObservation
    draft: StateObservation
    pending_token_id: int | None = None
    poisoned: bool = False
    poison_reason: str | None = None
    schema_version: str = SPECULATIVE_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SPECULATIVE_STATE_SCHEMA:
            raise ValueError(f"unsupported speculative-state schema: {self.schema_version}")
        object.__setattr__(self, "runtime_id", _name(self.runtime_id, "runtime_id"))
        object.__setattr__(self, "state_id", _name(self.state_id, "state_id"))
        object.__setattr__(
            self,
            "generation",
            _strict_nonnegative(self.generation, "generation"),
        )
        object.__setattr__(self, "epoch", _strict_nonnegative(self.epoch, "epoch"))
        if not isinstance(self.target, StateObservation) or not isinstance(
            self.draft,
            StateObservation,
        ):
            raise TypeError("speculative state requires target and draft observations")
        if self.target.batch_size != 1 or self.draft.batch_size != 1:
            raise ValueError("speculative target and draft state must both be B1")
        if self.target.capacity != self.draft.capacity and not self.poisoned:
            raise ValueError("speculative target and draft capacities must match")
        if self.pending_token_id is not None:
            object.__setattr__(
                self,
                "pending_token_id",
                _strict_nonnegative(self.pending_token_id, "pending_token_id"),
            )
        if type(self.poisoned) is not bool:
            raise TypeError("poisoned must be boolean")
        if self.poisoned:
            if self.poison_reason is None:
                raise ValueError("poisoned speculative state requires a reason")
            object.__setattr__(
                self,
                "poison_reason",
                _name(self.poison_reason, "poison_reason"),
            )
        elif self.poison_reason is not None:
            raise ValueError("healthy speculative state cannot carry a poison reason")
        if not self.poisoned and not self.synchronized:
            raise ValueError("healthy speculative target and draft prefixes must match")
        if not self.poisoned and self.synchronized:
            has_prefix = self.target.lengths[0] > 0
            if has_prefix != (self.pending_token_id is not None):
                raise ValueError(
                    "healthy speculative state must bind one pending token iff it has a prefix"
                )

    @property
    def synchronized(self) -> bool:
        return bool(
            self.target.capacity == self.draft.capacity
            and self.target.lengths == self.draft.lengths
        )

    @property
    def capacity(self) -> int:
        return self.target.capacity

    @property
    def committed_length(self) -> int:
        if not self.synchronized:
            raise SpeculativeRuntimeError("poisoned target and draft lengths have diverged")
        return self.target.lengths[0]


class SpeculativeState:
    """Opaque paired state handle.  Backend-owned state objects remain private."""

    __slots__ = (
        "_draft_state",
        "_epoch",
        "_generation",
        "_lock",
        "_owner_id",
        "_pending_token_id",
        "_poison_reason",
        "_poisoned",
        "_released",
        "_runtime",
        "_state_id",
        "_target_state",
        "_target_released",
        "_draft_released",
    )

    def __init__(
        self,
        *,
        runtime: ExactGreedySpeculativeRuntime,
        state_id: str,
        owner_id: str,
        generation: int,
        epoch: int,
        target_state: StateHandle,
        draft_state: StateHandle,
        pending_token_id: int | None = None,
    ) -> None:
        self._runtime = runtime
        self._state_id = state_id
        self._owner_id = owner_id
        self._generation = generation
        self._epoch = epoch
        self._target_state = target_state
        self._draft_state = draft_state
        self._target_released = False
        self._draft_released = False
        self._pending_token_id = pending_token_id
        self._poisoned = False
        self._poison_reason: str | None = None
        self._released = False
        self._lock = threading.RLock()

    @property
    def runtime_id(self) -> str:
        return self._runtime.runtime_id

    @property
    def state_id(self) -> str:
        return self._state_id

    @property
    def owner_id(self) -> str:
        return self._owner_id

    def observe(self) -> SpeculativeStateObservation:
        return self._runtime.observe_state(self)


@dataclass(frozen=True, slots=True)
class SpeculativePrefillResult:
    runtime_id: str
    request_id: str
    input_token_ids: tuple[int, ...]
    pending_token_id: int
    before: SpeculativeStateObservation
    after: SpeculativeStateObservation
    target_commit: CommitResult = field(repr=False)
    draft_commit: CommitResult = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_id", _name(self.runtime_id, "runtime_id"))
        object.__setattr__(self, "request_id", _name(self.request_id, "request_id"))
        inputs = tuple(
            _strict_nonnegative(value, "input_token_ids[]") for value in self.input_token_ids
        )
        if not inputs:
            raise ValueError("speculative prefill input cannot be empty")
        object.__setattr__(self, "input_token_ids", inputs)
        object.__setattr__(
            self,
            "pending_token_id",
            _strict_nonnegative(self.pending_token_id, "pending_token_id"),
        )
        if not isinstance(self.before, SpeculativeStateObservation) or not isinstance(
            self.after,
            SpeculativeStateObservation,
        ):
            raise TypeError("speculative prefill requires paired before/after observations")
        if self.before.runtime_id != self.runtime_id or self.after.runtime_id != self.runtime_id:
            raise ValueError("speculative prefill observations belong to another runtime")
        if self.before.state_id != self.after.state_id or self.before.committed_length != 0:
            raise ValueError("speculative prefill must initialize one empty paired state")
        if self.after.committed_length != len(inputs):
            raise ValueError("speculative prefill did not commit the complete input")
        if self.before.pending_token_id is not None:
            raise ValueError("empty speculative state cannot already carry a pending token")
        if self.after.pending_token_id != self.pending_token_id:
            raise ValueError("speculative prefill did not bind its target-selected token")
        if self.after.epoch != self.before.epoch + 1:
            raise ValueError("speculative prefill must advance the paired epoch exactly once")
        for receipt in (self.target_commit, self.draft_commit):
            if not isinstance(receipt, CommitResult) or receipt.accepted_counts != (len(inputs),):
                raise ValueError("speculative prefill requires complete backend commits")


@dataclass(frozen=True, slots=True)
class SpeculativeStepResult:
    """One exact target trajectory segment and its paired state transition."""

    runtime_id: str
    request_id: str
    current_token_id: int
    draft_token_ids: tuple[int, ...]
    target_prediction_ids: tuple[int, ...]
    accepted_draft_tokens: int
    output_token_ids: tuple[int, ...]
    before: SpeculativeStateObservation
    after: SpeculativeStateObservation
    target_commit: CommitResult = field(repr=False)
    draft_commit: CommitResult = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_id", _name(self.runtime_id, "runtime_id"))
        object.__setattr__(self, "request_id", _name(self.request_id, "request_id"))
        object.__setattr__(
            self,
            "current_token_id",
            _strict_nonnegative(self.current_token_id, "current_token_id"),
        )
        drafts = tuple(
            _strict_nonnegative(value, "draft_token_ids[]") for value in self.draft_token_ids
        )
        predictions = tuple(
            _strict_nonnegative(value, "target_prediction_ids[]")
            for value in self.target_prediction_ids
        )
        outputs = tuple(
            _strict_nonnegative(value, "output_token_ids[]") for value in self.output_token_ids
        )
        accepted = _strict_nonnegative(self.accepted_draft_tokens, "accepted_draft_tokens")
        if len(predictions) != len(drafts) + 1:
            raise ValueError("target verification must predict every speculative block position")
        if accepted > len(drafts):
            raise ValueError("accepted draft count exceeds the proposed block")
        if any(drafts[index] != predictions[index] for index in range(accepted)):
            raise ValueError("accepted draft prefix does not match the target trajectory")
        if accepted < len(drafts) and drafts[accepted] == predictions[accepted]:
            raise ValueError("speculative acceptance must include every consecutive match")
        expected_outputs = drafts[:accepted] + (predictions[accepted],)
        if outputs != expected_outputs:
            raise ValueError("speculative outputs are not the target-correct trajectory")
        object.__setattr__(self, "draft_token_ids", drafts)
        object.__setattr__(self, "target_prediction_ids", predictions)
        object.__setattr__(self, "accepted_draft_tokens", accepted)
        object.__setattr__(self, "output_token_ids", outputs)
        if not isinstance(self.before, SpeculativeStateObservation) or not isinstance(
            self.after,
            SpeculativeStateObservation,
        ):
            raise TypeError("speculative step requires paired before/after observations")
        if self.before.state_id != self.after.state_id:
            raise ValueError("speculative step cannot replace paired state authority")
        if self.before.pending_token_id != self.current_token_id:
            raise ValueError("speculative step current token is not state-authoritative")
        if self.after.pending_token_id != outputs[-1]:
            raise ValueError("speculative step did not bind its final target prediction")
        committed = accepted + 1
        if self.after.committed_length != self.before.committed_length + committed:
            raise ValueError("speculative step committed the wrong input prefix")
        if self.after.epoch != self.before.epoch + 1:
            raise ValueError("speculative step must advance the paired epoch exactly once")
        for receipt in (self.target_commit, self.draft_commit):
            if not isinstance(receipt, CommitResult) or receipt.accepted_counts != (committed,):
                raise ValueError("target and draft must commit the identical input prefix")

    @property
    def all_draft_tokens_accepted(self) -> bool:
        return self.accepted_draft_tokens == len(self.draft_token_ids)


@dataclass(frozen=True, slots=True)
class SpeculativeForkResult:
    runtime_id: str
    source: SpeculativeStateObservation
    forked: SpeculativeStateObservation
    state: SpeculativeState = field(repr=False, compare=False)
    state_bytes_copied: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_id", _name(self.runtime_id, "runtime_id"))
        if not isinstance(self.source, SpeculativeStateObservation) or not isinstance(
            self.forked,
            SpeculativeStateObservation,
        ):
            raise TypeError("speculative fork requires paired observations")
        if not isinstance(self.state, SpeculativeState):
            raise TypeError("speculative fork requires SpeculativeState authority")
        if self.source.runtime_id != self.runtime_id or self.forked.runtime_id != self.runtime_id:
            raise ValueError("speculative fork observations belong to another runtime")
        if self.source.state_id == self.forked.state_id:
            raise ValueError("speculative fork must mint a new state identity")
        if self.forked.generation <= self.source.generation:
            raise ValueError("speculative fork must mint a fresh generation")
        if self.source.committed_length != self.forked.committed_length:
            raise ValueError("speculative fork must preserve the exact committed prefix")
        if self.source.pending_token_id != self.forked.pending_token_id:
            raise ValueError("speculative fork must preserve the pending target token")
        object.__setattr__(
            self,
            "state_bytes_copied",
            _strict_nonnegative(self.state_bytes_copied, "state_bytes_copied"),
        )


@dataclass(frozen=True, slots=True)
class SpeculativeTelemetry:
    runtime_id: str
    configured_proposal_tokens: int
    states_allocated: int
    states_forked: int
    states_released: int
    states_live: int
    states_poisoned: int
    prefill_calls: int
    speculative_steps: int
    draft_proposal_forwards: int
    draft_proposed_tokens: int
    target_block_calls: int
    target_verified_tokens: int
    draft_sync_block_calls: int
    draft_sync_tokens: int
    optimized_draft_steps: int
    fallback_draft_steps: int
    draft_transaction_begins: int
    draft_transaction_advances: int
    draft_transaction_seals: int
    draft_transaction_commits: int
    draft_transaction_abandons: int
    accepted_draft_tokens: int
    emitted_tokens: int
    full_accept_steps: int
    partial_accept_steps: int
    zero_accept_steps: int
    correction_tokens: int
    bonus_tokens: int
    ordinary_target_steps: int
    proposal_state_forks: int
    proposal_state_releases: int
    proposal_prefix_bytes_copied: int
    verification_abandons: int
    failures: int
    split_commit_failures: int
    execution_abi: str = SPECULATIVE_RUNTIME_ABI

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_id", _name(self.runtime_id, "runtime_id"))
        if self.execution_abi != SPECULATIVE_RUNTIME_ABI:
            raise ValueError(f"unsupported speculative runtime ABI: {self.execution_abi}")
        for field_name in (
            "configured_proposal_tokens",
            "states_allocated",
            "states_forked",
            "states_released",
            "states_live",
            "states_poisoned",
            "prefill_calls",
            "speculative_steps",
            "draft_proposal_forwards",
            "draft_proposed_tokens",
            "target_block_calls",
            "target_verified_tokens",
            "draft_sync_block_calls",
            "draft_sync_tokens",
            "optimized_draft_steps",
            "fallback_draft_steps",
            "draft_transaction_begins",
            "draft_transaction_advances",
            "draft_transaction_seals",
            "draft_transaction_commits",
            "draft_transaction_abandons",
            "accepted_draft_tokens",
            "emitted_tokens",
            "full_accept_steps",
            "partial_accept_steps",
            "zero_accept_steps",
            "correction_tokens",
            "bonus_tokens",
            "ordinary_target_steps",
            "proposal_state_forks",
            "proposal_state_releases",
            "proposal_prefix_bytes_copied",
            "verification_abandons",
            "failures",
            "split_commit_failures",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_nonnegative(getattr(self, field_name), field_name),
            )
        if self.configured_proposal_tokens == 0:
            raise ValueError("configured_proposal_tokens must be positive")

    @property
    def acceptance_rate(self) -> float:
        if self.draft_proposed_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.draft_proposed_tokens


class ExactGreedySpeculativeRuntime:
    """Paired target/draft runtime with exact target-verified greedy output semantics."""

    def __init__(
        self,
        target: Any,
        draft: Any,
        *,
        semantic_token_count: int,
        proposal_tokens: int = 4,
        runtime_id: str | None = None,
        owns_runtimes: bool = False,
    ) -> None:
        if not isinstance(target, ForkableModelRuntime) or not isinstance(
            target,
            GreedyBlockVerifier,
        ):
            raise TypeError("target must be a forkable ModelRuntime with greedy block verification")
        if not isinstance(draft, ForkableModelRuntime):
            raise TypeError("draft must be a forkable ModelRuntime")
        transactional_draft = isinstance(draft, GreedyProposalTransactionRuntime)
        if not transactional_draft and not isinstance(draft, GreedyBlockVerifier):
            raise TypeError(
                "draft must support in-place greedy proposal transactions or block verification"
            )
        self._semantic_token_count = _strict_positive(
            semantic_token_count,
            "semantic_token_count",
        )
        self._proposal_tokens = _strict_positive(proposal_tokens, "proposal_tokens")
        self._runtime_id = _name(
            runtime_id or f"speculative-{uuid4().hex}",
            "runtime_id",
        )
        if type(owns_runtimes) is not bool:
            raise TypeError("owns_runtimes must be boolean")
        self._target = target
        self._draft = draft
        self._transactional_draft = draft if transactional_draft else None
        self._owns_runtimes = owns_runtimes
        self._states: dict[str, SpeculativeState] = {}
        self._next_generation = 1
        self._closed = False
        self._lock = threading.RLock()
        self._states_allocated = 0
        self._states_forked = 0
        self._states_released = 0
        self._states_poisoned = 0
        self._prefill_calls = 0
        self._speculative_steps = 0
        self._draft_proposal_forwards = 0
        self._draft_proposed_tokens = 0
        self._target_block_calls = 0
        self._target_verified_tokens = 0
        self._draft_sync_block_calls = 0
        self._draft_sync_tokens = 0
        self._optimized_draft_steps = 0
        self._fallback_draft_steps = 0
        self._draft_transaction_begins = 0
        self._draft_transaction_advances = 0
        self._draft_transaction_seals = 0
        self._draft_transaction_commits = 0
        self._draft_transaction_abandons = 0
        self._accepted_draft_tokens = 0
        self._emitted_tokens = 0
        self._full_accept_steps = 0
        self._partial_accept_steps = 0
        self._zero_accept_steps = 0
        self._correction_tokens = 0
        self._bonus_tokens = 0
        self._ordinary_target_steps = 0
        self._proposal_state_forks = 0
        self._proposal_state_releases = 0
        self._proposal_prefix_bytes_copied = 0
        self._verification_abandons = 0
        self._failures = 0
        self._split_commit_failures = 0

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def proposal_tokens(self) -> int:
        return self._proposal_tokens

    @property
    def uses_in_place_draft_transactions(self) -> bool:
        return self._transactional_draft is not None

    def _require_open(self) -> None:
        if self._closed:
            raise SpeculativeRuntimeError("speculative runtime is closed")

    def _mint_generation(self) -> int:
        with self._lock:
            generation = self._next_generation
            self._next_generation += 1
            return generation

    def _state(self, state: Any) -> SpeculativeState:
        if not isinstance(state, SpeculativeState) or state._runtime is not self:  # noqa: SLF001
            raise SpeculativeRuntimeError("speculative state belongs to another runtime")
        with self._lock:
            self._require_open()
            if self._states.get(state.state_id) is not state:
                raise SpeculativeRuntimeError("speculative state is stale or released")
        return state

    @staticmethod
    def _backend_prefixes_synchronized(
        target: StateObservation,
        draft: StateObservation,
    ) -> bool:
        return bool(
            target.batch_size == 1
            and draft.batch_size == 1
            and target.capacity == draft.capacity
            and target.lengths == draft.lengths
        )

    def _poison_locked(self, state: SpeculativeState, reason: str) -> None:
        if state._poisoned:  # noqa: SLF001
            return
        state._poisoned = True  # noqa: SLF001
        state._poison_reason = _name(reason, "poison reason")  # noqa: SLF001
        with self._lock:
            self._states_poisoned += 1

    def _observe_locked(self, state: SpeculativeState) -> SpeculativeStateObservation:
        if state._released:  # noqa: SLF001
            raise SpeculativeRuntimeError("speculative state was released")
        try:
            target = state._target_state.observe()  # noqa: SLF001
            draft = state._draft_state.observe()  # noqa: SLF001
        except BaseException as exc:
            self._poison_locked(state, f"backend state observation failed: {type(exc).__name__}")
            raise SpeculativeRuntimeError("could not observe paired backend state") from exc
        if not self._backend_prefixes_synchronized(target, draft):
            self._poison_locked(state, "target and draft committed prefixes diverged")
        return SpeculativeStateObservation(
            runtime_id=self._runtime_id,
            state_id=state.state_id,
            generation=state._generation,  # noqa: SLF001
            epoch=state._epoch,  # noqa: SLF001
            target=target,
            draft=draft,
            pending_token_id=state._pending_token_id,  # noqa: SLF001
            poisoned=state._poisoned,  # noqa: SLF001
            poison_reason=state._poison_reason,  # noqa: SLF001
        )

    def _healthy_observation_locked(
        self,
        state: SpeculativeState,
    ) -> SpeculativeStateObservation:
        observation = self._observe_locked(state)
        if observation.poisoned:
            raise SpeculativeRuntimeError(
                f"speculative state is poisoned: {observation.poison_reason}"
            )
        return observation

    def observe_state(self, state: SpeculativeState) -> SpeculativeStateObservation:
        resolved = self._state(state)
        with resolved._lock:  # noqa: SLF001
            return self._observe_locked(resolved)

    def _tokens(self, values: Sequence[int], field_name: str) -> tuple[int, ...]:
        tokens = tuple(values)
        if not tokens:
            raise ValueError(f"{field_name} cannot be empty")
        normalized: list[int] = []
        for value in tokens:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must contain strict integers")
            token = int(value)
            if token < 0 or token >= self._semantic_token_count:
                raise ValueError(f"{field_name} escapes the shared semantic token domain")
            normalized.append(token)
        return tuple(normalized)

    def _selected_token(self, step: ProvisionalStep, *, runtime_name: str) -> int:
        if not isinstance(step, ProvisionalStep):
            raise SpeculativeRuntimeError(f"{runtime_name} returned a non-provisional result")
        if step.output.mode is not OutputMode.NEXT_TOKEN_ARGMAX or len(step.output.token_ids) != 1:
            raise SpeculativeRuntimeError(
                f"{runtime_name} must return exactly one raw-greedy token"
            )
        token = step.output.token_ids[0]
        if token >= self._semantic_token_count:
            raise SpeculativeRuntimeError(
                f"{runtime_name} selection escaped the shared semantic token domain"
            )
        return token

    def _validate_verification(
        self,
        verification: GreedyBlockVerification,
        *,
        runtime: Any,
        parent: StateObservation,
        request_id: str,
        input_token_ids: tuple[int, ...],
        runtime_name: str,
    ) -> None:
        if not isinstance(verification, GreedyBlockVerification):
            raise SpeculativeRuntimeError(f"{runtime_name} returned a non-verification result")
        if (
            verification.runtime_id != runtime.route.runtime_id
            or verification.parent != parent
            or verification.request_id != request_id
            or verification.input_token_ids != input_token_ids
        ):
            raise SpeculativeRuntimeError(
                f"{runtime_name} verification does not bind the requested block"
            )
        if any(token >= self._semantic_token_count for token in verification.predicted_token_ids):
            raise SpeculativeRuntimeError(
                f"{runtime_name} verification escaped the shared semantic token domain"
            )

    @staticmethod
    def _validate_commit_receipt(
        receipt: CommitResult,
        *,
        runtime: Any,
        parent: StateObservation,
        after: StateObservation,
        accepted_count: int,
        runtime_name: str,
    ) -> None:
        if not isinstance(receipt, CommitResult):
            raise SpeculativeRuntimeError(f"{runtime_name} returned a non-commit receipt")
        if (
            receipt.runtime_id != runtime.route.runtime_id
            or receipt.before != parent
            or receipt.after != after
            or receipt.accepted_counts != (accepted_count,)
        ):
            raise SpeculativeRuntimeError(
                f"{runtime_name} commit receipt does not bind the paired transition"
            )

    def _record_failure(self, *, split_commit: bool = False) -> None:
        with self._lock:
            self._failures += 1
            if split_commit:
                self._split_commit_failures += 1

    def allocate_state(
        self,
        *,
        owner_id: str,
        capacity: int,
    ) -> SpeculativeState:
        owner = _name(owner_id, "owner_id")
        requested_capacity = _strict_positive(capacity, "capacity")
        with self._lock:
            self._require_open()
        target_state = self._target.allocate_state(
            owner_id=f"{owner}.target",
            batch_size=1,
            capacity=requested_capacity,
        )
        try:
            draft_state = self._draft.allocate_state(
                owner_id=f"{owner}.draft",
                batch_size=1,
                capacity=requested_capacity,
            )
        except BaseException:
            self._target.release_state(target_state)
            raise
        target_observation = target_state.observe()
        draft_observation = draft_state.observe()
        if not self._backend_prefixes_synchronized(target_observation, draft_observation):
            self._draft.release_state(draft_state)
            self._target.release_state(target_state)
            raise SpeculativeRuntimeError("new target and draft states are not synchronized")
        state = SpeculativeState(
            runtime=self,
            state_id=f"spec-state-{uuid4().hex}",
            owner_id=owner,
            generation=self._mint_generation(),
            epoch=0,
            target_state=target_state,
            draft_state=draft_state,
            pending_token_id=None,
        )
        with self._lock:
            self._require_open()
            self._states[state.state_id] = state
            self._states_allocated += 1
        return state

    @staticmethod
    def _argmax_output() -> OutputRequest:
        return OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX)

    def _abandon_standard_locked(
        self,
        state: SpeculativeState,
        runtime: Any,
        step: ProvisionalStep,
        runtime_name: str,
    ) -> None:
        try:
            runtime.abandon(step)
        except BaseException as exc:
            self._poison_locked(
                state,
                f"{runtime_name} ordinary provisional cleanup failed",
            )
            raise SpeculativeRuntimeError(
                f"could not abandon {runtime_name} provisional work"
            ) from exc

    def _abandon_verification_locked(
        self,
        state: SpeculativeState,
        runtime: Any,
        verification: GreedyBlockVerification,
        runtime_name: str,
    ) -> None:
        try:
            runtime.abandon_greedy_block(verification)
        except BaseException as exc:
            self._poison_locked(
                state,
                f"{runtime_name} greedy verification cleanup failed",
            )
            raise SpeculativeRuntimeError(
                f"could not abandon {runtime_name} greedy verification"
            ) from exc
        with self._lock:
            self._verification_abandons += 1

    def prefill(
        self,
        state: SpeculativeState,
        *,
        request_id: str,
        token_ids: Sequence[int],
    ) -> SpeculativePrefillResult:
        resolved = self._state(state)
        request = _name(request_id, "request_id")
        inputs = self._tokens(token_ids, "prefill token_ids")
        with resolved._lock:  # noqa: SLF001
            before = self._healthy_observation_locked(resolved)
            if before.committed_length != 0:
                raise SpeculativeRuntimeError("speculative prefill requires empty paired state")
            target_step: ProvisionalStep | None = None
            draft_step: ProvisionalStep | None = None
            try:
                target_step = self._target.prefill(
                    PrefillWork(
                        request_ids=(request,),
                        token_rows=(inputs,),
                        state=resolved._target_state,  # noqa: SLF001
                        parent=before.target,
                        output=self._argmax_output(),
                    )
                )
                target_pending = self._selected_token(target_step, runtime_name="target")
                draft_step = self._draft.prefill(
                    PrefillWork(
                        request_ids=(request,),
                        token_rows=(inputs,),
                        state=resolved._draft_state,  # noqa: SLF001
                        parent=before.draft,
                        output=self._argmax_output(),
                    )
                )
                self._selected_token(draft_step, runtime_name="draft")
            except BaseException as exc:
                cleanup_errors: list[BaseException] = []
                for runtime, pending, runtime_name in (
                    (self._draft, draft_step, "draft"),
                    (self._target, target_step, "target"),
                ):
                    if pending is not None:
                        try:
                            self._abandon_standard_locked(
                                resolved,
                                runtime,
                                pending,
                                runtime_name,
                            )
                        except BaseException as cleanup_exc:
                            cleanup_errors.append(cleanup_exc)
                self._record_failure()
                if cleanup_errors:
                    raise SpeculativeRuntimeError(
                        "speculative prefill failed and provisional cleanup was incomplete"
                    ) from cleanup_errors[0]
                raise SpeculativeRuntimeError("speculative prefill execution failed") from exc
            assert target_step is not None and draft_step is not None
            try:
                target_commit = self._target.commit(target_step, (len(inputs),))
            except BaseException as exc:
                for runtime, pending, runtime_name in (
                    (self._draft, draft_step, "draft"),
                    (self._target, target_step, "target"),
                ):
                    try:
                        self._abandon_standard_locked(
                            resolved,
                            runtime,
                            pending,
                            runtime_name,
                        )
                    except BaseException:
                        pass
                self._poison_locked(resolved, "target prefill commit was ambiguous")
                self._record_failure()
                raise SpeculativeRuntimeError("target prefill commit failed") from exc
            try:
                draft_commit = self._draft.commit(draft_step, (len(inputs),))
            except BaseException as exc:
                try:
                    self._abandon_standard_locked(resolved, self._draft, draft_step, "draft")
                except BaseException:
                    pass
                self._poison_locked(resolved, "target committed prefill before draft failure")
                self._record_failure(split_commit=True)
                raise SpeculativeRuntimeError("draft prefill commit split paired state") from exc
            resolved._epoch += 1  # noqa: SLF001
            resolved._pending_token_id = target_pending  # noqa: SLF001
            after = self._healthy_observation_locked(resolved)
            if after.committed_length != len(inputs):
                self._poison_locked(resolved, "prefill backend commit lengths were inconsistent")
                self._record_failure(split_commit=True)
                raise SpeculativeRuntimeError("paired prefill committed inconsistent lengths")
            try:
                self._validate_commit_receipt(
                    target_commit,
                    runtime=self._target,
                    parent=before.target,
                    after=after.target,
                    accepted_count=len(inputs),
                    runtime_name="target",
                )
                self._validate_commit_receipt(
                    draft_commit,
                    runtime=self._draft,
                    parent=before.draft,
                    after=after.draft,
                    accepted_count=len(inputs),
                    runtime_name="draft",
                )
            except BaseException:
                self._poison_locked(resolved, "prefill commit receipts were inconsistent")
                self._record_failure(split_commit=True)
                raise
            result = SpeculativePrefillResult(
                runtime_id=self._runtime_id,
                request_id=request,
                input_token_ids=inputs,
                pending_token_id=target_pending,
                before=before,
                after=after,
                target_commit=target_commit,
                draft_commit=draft_commit,
            )
        with self._lock:
            self._prefill_calls += 1
        return result

    def _validate_draft_transaction(
        self,
        transaction: GreedyProposalTransaction,
        *,
        state: SpeculativeState,
        parent: StateObservation,
        request_id: str,
        input_token_ids: tuple[int, ...],
        predicted_token_ids: tuple[int, ...],
        sealed: bool,
    ) -> None:
        if not isinstance(transaction, GreedyProposalTransaction):
            raise SpeculativeRuntimeError("draft returned a non-transaction result")
        if (
            transaction.runtime_id != self._draft.route.runtime_id
            or transaction.state is not state._draft_state  # noqa: SLF001
            or transaction.parent != parent
            or transaction.request_id != request_id
            or transaction.input_token_ids != input_token_ids
            or transaction.predicted_token_ids != predicted_token_ids
            or transaction.sealed is not sealed
        ):
            raise SpeculativeRuntimeError(
                "draft proposal transaction does not bind its exact cursor"
            )
        if any(token >= self._semantic_token_count for token in predicted_token_ids):
            raise SpeculativeRuntimeError(
                "draft proposal transaction escaped the shared semantic token domain"
            )

    def _abandon_draft_transaction_locked(
        self,
        state: SpeculativeState,
        transaction: GreedyProposalTransaction,
    ) -> None:
        draft = self._transactional_draft
        if draft is None:
            raise SpeculativeRuntimeError("draft transaction capability disappeared")
        try:
            draft.abandon_greedy_proposal(transaction)
        except BaseException as exc:
            self._poison_locked(state, "draft proposal transaction cleanup failed")
            raise SpeculativeRuntimeError("could not abandon draft proposal transaction") from exc
        with self._lock:
            self._draft_transaction_abandons += 1

    def _draft_transaction_proposals_locked(
        self,
        state: SpeculativeState,
        *,
        parent: StateObservation,
        request_id: str,
        current_token_id: int,
        proposal_count: int,
    ) -> tuple[GreedyProposalTransaction, tuple[int, ...]]:
        draft = self._transactional_draft
        if draft is None:
            raise SpeculativeRuntimeError("draft transaction capability disappeared")
        transaction: GreedyProposalTransaction | None = None
        proposals: list[int] = []
        try:
            transaction = draft.begin_greedy_proposal(
                GreedyProposalBeginWork(
                    request_id=request_id,
                    state=state._draft_state,  # noqa: SLF001
                    parent=parent,
                )
            )
            self._validate_draft_transaction(
                transaction,
                state=state,
                parent=parent,
                request_id=request_id,
                input_token_ids=(),
                predicted_token_ids=(),
                sealed=False,
            )
            with self._lock:
                self._draft_transaction_begins += 1
            current = current_token_id
            for _index in range(proposal_count):
                previous_inputs = transaction.input_token_ids
                previous_predictions = transaction.predicted_token_ids
                transaction = draft.advance_greedy_proposal(transaction, current)
                expected_inputs = previous_inputs + (current,)
                if len(transaction.predicted_token_ids) != len(expected_inputs):
                    raise SpeculativeRuntimeError(
                        "draft transaction did not select its appended input"
                    )
                expected_predictions = previous_predictions + (transaction.predicted_token_ids[-1],)
                self._validate_draft_transaction(
                    transaction,
                    state=state,
                    parent=parent,
                    request_id=request_id,
                    input_token_ids=expected_inputs,
                    predicted_token_ids=expected_predictions,
                    sealed=False,
                )
                current = transaction.latest_prediction
                proposals.append(current)
                with self._lock:
                    self._draft_transaction_advances += 1
                    self._draft_proposal_forwards += 1
        except BaseException as exc:
            if transaction is not None:
                try:
                    self._abandon_draft_transaction_locked(state, transaction)
                except BaseException as cleanup_exc:
                    raise SpeculativeRuntimeError(
                        "draft transaction failed and cleanup was incomplete"
                    ) from cleanup_exc
            raise SpeculativeRuntimeError("draft proposal transaction failed") from exc
        assert transaction is not None
        return transaction, tuple(proposals)

    def _draft_proposals_locked(
        self,
        state: SpeculativeState,
        *,
        parent: StateObservation,
        request_id: str,
        current_token_id: int,
        proposal_count: int,
    ) -> tuple[int, ...]:
        if proposal_count == 0:
            return ()
        fork = self._draft.fork_state(
            state._draft_state,  # noqa: SLF001
            parent=parent,
            owner_id=f"{state.owner_id}.proposal.{uuid4().hex}",
            capacity=parent.capacity,
        )
        with self._lock:
            self._proposal_state_forks += 1
            self._proposal_prefix_bytes_copied += fork.state_bytes_copied
        proposal_state = fork.state
        proposal_parent = fork.forked
        proposals: list[int] = []
        current = current_token_id
        operation_error: BaseException | None = None
        pending: ProvisionalStep | None = None
        try:
            for index in range(proposal_count):
                pending = self._draft.decode(
                    DecodeWork(
                        request_ids=(f"{request_id}.proposal.{index}",),
                        token_rows=((current,),),
                        state=proposal_state,
                        parent=proposal_parent,
                        output=self._argmax_output(),
                    )
                )
                proposed = self._selected_token(pending, runtime_name="draft")
                commit = self._draft.commit(pending, (1,))
                pending = None
                if commit.before != proposal_parent or commit.accepted_counts != (1,):
                    raise SpeculativeRuntimeError(
                        "draft proposal commit returned an invalid receipt"
                    )
                proposal_parent = commit.after
                proposals.append(proposed)
                current = proposed
                with self._lock:
                    self._draft_proposal_forwards += 1
        except BaseException as exc:
            operation_error = exc
            if pending is not None:
                try:
                    self._draft.abandon(pending)
                    pending = None
                except BaseException:
                    pass
        release_error: BaseException | None = None
        try:
            self._draft.release_state(proposal_state)
        except BaseException as exc:
            release_error = exc
        else:
            with self._lock:
                self._proposal_state_releases += 1
        if release_error is not None:
            self._poison_locked(state, "draft proposal fork could not be released")
            raise SpeculativeRuntimeError("draft proposal fork cleanup failed") from release_error
        if operation_error is not None:
            raise SpeculativeRuntimeError("draft proposal generation failed") from operation_error
        return tuple(proposals)

    def step(
        self,
        state: SpeculativeState,
        *,
        request_id: str,
        current_token_id: int,
        proposal_tokens: int | None = None,
    ) -> SpeculativeStepResult:
        resolved = self._state(state)
        request = _name(request_id, "request_id")
        current = self._tokens((current_token_id,), "current_token_id")[0]
        configured = (
            self._proposal_tokens
            if proposal_tokens is None
            else _strict_positive(proposal_tokens, "proposal_tokens")
        )
        with resolved._lock:  # noqa: SLF001
            before = self._healthy_observation_locked(resolved)
            if before.committed_length <= 0:
                raise SpeculativeRuntimeError("speculative step requires a committed prefix")
            if before.pending_token_id != current:
                raise SpeculativeRuntimeError(
                    "current token does not match the target-authoritative pending token"
                )
            remaining = before.capacity - before.committed_length
            if remaining <= 0:
                raise OverflowError("speculative state has no remaining input capacity")
            proposal_count = min(configured, remaining - 1)
            optimized = self._transactional_draft is not None
            draft_transaction: GreedyProposalTransaction | None = None
            try:
                if optimized:
                    draft_transaction, proposals = self._draft_transaction_proposals_locked(
                        resolved,
                        parent=before.draft,
                        request_id=request,
                        current_token_id=current,
                        proposal_count=proposal_count,
                    )
                else:
                    proposals = self._draft_proposals_locked(
                        resolved,
                        parent=before.draft,
                        request_id=request,
                        current_token_id=current,
                        proposal_count=proposal_count,
                    )
            except BaseException:
                self._record_failure()
                raise
            input_block = (current,) + proposals
            target_verification: GreedyBlockVerification | None = None
            draft_verification: GreedyBlockVerification | None = None
            try:
                target_verification = self._target.verify_greedy_block(
                    GreedyBlockVerifyWork(
                        request_id=request,
                        token_ids=input_block,
                        state=resolved._target_state,  # noqa: SLF001
                        parent=before.target,
                    )
                )
                self._validate_verification(
                    target_verification,
                    runtime=self._target,
                    parent=before.target,
                    request_id=request,
                    input_token_ids=input_block,
                    runtime_name="target",
                )
                with self._lock:
                    self._target_block_calls += 1
                    self._target_verified_tokens += len(input_block)
                if not optimized:
                    draft_verification = self._draft.verify_greedy_block(
                        GreedyBlockVerifyWork(
                            request_id=request,
                            token_ids=input_block,
                            state=resolved._draft_state,  # noqa: SLF001
                            parent=before.draft,
                        )
                    )
                    self._validate_verification(
                        draft_verification,
                        runtime=self._draft,
                        parent=before.draft,
                        request_id=request,
                        input_token_ids=input_block,
                        runtime_name="draft",
                    )
                    with self._lock:
                        self._draft_sync_block_calls += 1
                        self._draft_sync_tokens += len(input_block)
            except BaseException as exc:
                cleanup_errors: list[BaseException] = []
                if draft_transaction is not None:
                    try:
                        self._abandon_draft_transaction_locked(
                            resolved,
                            draft_transaction,
                        )
                    except BaseException as cleanup_exc:
                        cleanup_errors.append(cleanup_exc)
                for runtime, verification, runtime_name in (
                    (self._draft, draft_verification, "draft"),
                    (self._target, target_verification, "target"),
                ):
                    if verification is not None:
                        try:
                            self._abandon_verification_locked(
                                resolved,
                                runtime,
                                verification,
                                runtime_name,
                            )
                        except BaseException as cleanup_exc:
                            cleanup_errors.append(cleanup_exc)
                self._record_failure()
                if cleanup_errors:
                    raise SpeculativeRuntimeError(
                        "speculative verification failed and cleanup was incomplete"
                    ) from cleanup_errors[0]
                raise SpeculativeRuntimeError("speculative block verification failed") from exc
            assert target_verification is not None
            if not optimized:
                assert draft_verification is not None
            predictions = target_verification.predicted_token_ids
            accepted = 0
            while accepted < len(proposals) and proposals[accepted] == predictions[accepted]:
                accepted += 1
            committed_inputs = accepted + 1
            outputs = proposals[:accepted] + (predictions[accepted],)
            if optimized and accepted == len(proposals):
                assert draft_transaction is not None
                draft = self._transactional_draft
                assert draft is not None
                try:
                    draft_transaction = draft.seal_greedy_proposal(
                        draft_transaction,
                        input_block[-1],
                    )
                    self._validate_draft_transaction(
                        draft_transaction,
                        state=resolved,
                        parent=before.draft,
                        request_id=request,
                        input_token_ids=input_block,
                        predicted_token_ids=proposals,
                        sealed=True,
                    )
                    with self._lock:
                        self._draft_transaction_seals += 1
                except BaseException as exc:
                    cleanup_errors: list[BaseException] = []
                    try:
                        self._abandon_draft_transaction_locked(
                            resolved,
                            draft_transaction,
                        )
                    except BaseException as cleanup_exc:
                        cleanup_errors.append(cleanup_exc)
                    try:
                        self._abandon_verification_locked(
                            resolved,
                            self._target,
                            target_verification,
                            "target",
                        )
                    except BaseException as cleanup_exc:
                        cleanup_errors.append(cleanup_exc)
                    self._record_failure()
                    if cleanup_errors:
                        raise SpeculativeRuntimeError(
                            "draft transaction seal failed and cleanup was incomplete"
                        ) from cleanup_errors[0]
                    raise SpeculativeRuntimeError("draft proposal transaction seal failed") from exc
            try:
                target_commit = self._target.commit_greedy_block(
                    target_verification,
                    committed_inputs,
                )
            except BaseException as exc:
                if draft_transaction is not None:
                    try:
                        self._abandon_draft_transaction_locked(
                            resolved,
                            draft_transaction,
                        )
                    except BaseException:
                        pass
                if draft_verification is not None:
                    try:
                        self._abandon_verification_locked(
                            resolved,
                            self._draft,
                            draft_verification,
                            "draft",
                        )
                    except BaseException:
                        pass
                try:
                    self._abandon_verification_locked(
                        resolved,
                        self._target,
                        target_verification,
                        "target",
                    )
                except BaseException:
                    pass
                self._poison_locked(resolved, "target speculative commit was ambiguous")
                self._record_failure()
                raise SpeculativeRuntimeError("target speculative commit failed") from exc
            try:
                if optimized:
                    assert draft_transaction is not None
                    draft = self._transactional_draft
                    assert draft is not None
                    draft_commit = draft.commit_greedy_proposal(
                        draft_transaction,
                        committed_inputs,
                    )
                    with self._lock:
                        self._draft_transaction_commits += 1
                else:
                    assert draft_verification is not None
                    draft_commit = self._draft.commit_greedy_block(
                        draft_verification,
                        committed_inputs,
                    )
            except BaseException as exc:
                if draft_transaction is not None:
                    try:
                        self._abandon_draft_transaction_locked(
                            resolved,
                            draft_transaction,
                        )
                    except BaseException:
                        pass
                if draft_verification is not None:
                    try:
                        self._abandon_verification_locked(
                            resolved,
                            self._draft,
                            draft_verification,
                            "draft",
                        )
                    except BaseException:
                        pass
                self._poison_locked(resolved, "target committed before draft speculative failure")
                self._record_failure(split_commit=True)
                raise SpeculativeRuntimeError(
                    "draft speculative commit split paired state"
                ) from exc
            resolved._epoch += 1  # noqa: SLF001
            resolved._pending_token_id = outputs[-1]  # noqa: SLF001
            after = self._healthy_observation_locked(resolved)
            expected_length = before.committed_length + committed_inputs
            if after.committed_length != expected_length:
                self._poison_locked(resolved, "speculative commits advanced inconsistent lengths")
                self._record_failure(split_commit=True)
                raise SpeculativeRuntimeError("paired speculative commit length drifted")
            try:
                self._validate_commit_receipt(
                    target_commit,
                    runtime=self._target,
                    parent=before.target,
                    after=after.target,
                    accepted_count=committed_inputs,
                    runtime_name="target",
                )
                self._validate_commit_receipt(
                    draft_commit,
                    runtime=self._draft,
                    parent=before.draft,
                    after=after.draft,
                    accepted_count=committed_inputs,
                    runtime_name="draft",
                )
            except BaseException:
                self._poison_locked(resolved, "speculative commit receipts were inconsistent")
                self._record_failure(split_commit=True)
                raise
            result = SpeculativeStepResult(
                runtime_id=self._runtime_id,
                request_id=request,
                current_token_id=current,
                draft_token_ids=proposals,
                target_prediction_ids=predictions,
                accepted_draft_tokens=accepted,
                output_token_ids=outputs,
                before=before,
                after=after,
                target_commit=target_commit,
                draft_commit=draft_commit,
            )
        with self._lock:
            self._speculative_steps += 1
            if optimized:
                self._optimized_draft_steps += 1
            else:
                self._fallback_draft_steps += 1
            self._draft_proposed_tokens += len(proposals)
            self._accepted_draft_tokens += accepted
            self._emitted_tokens += len(outputs)
            if not proposals:
                self._ordinary_target_steps += 1
            elif accepted == len(proposals):
                self._full_accept_steps += 1
                self._bonus_tokens += 1
            elif accepted == 0:
                self._zero_accept_steps += 1
                self._correction_tokens += 1
            else:
                self._partial_accept_steps += 1
                self._correction_tokens += 1
        return result

    def fork_state(
        self,
        source: SpeculativeState,
        *,
        parent: SpeculativeStateObservation,
        owner_id: str,
        capacity: int,
    ) -> SpeculativeForkResult:
        resolved = self._state(source)
        if not isinstance(parent, SpeculativeStateObservation):
            raise TypeError("speculative fork parent must be SpeculativeStateObservation")
        owner = _name(owner_id, "owner_id")
        fork_capacity = _strict_positive(capacity, "capacity")
        with resolved._lock:  # noqa: SLF001
            current = self._healthy_observation_locked(resolved)
            if current != parent:
                raise SpeculativeRuntimeError("speculative fork parent is stale")
            target_fork = self._target.fork_state(
                resolved._target_state,  # noqa: SLF001
                parent=current.target,
                owner_id=f"{owner}.target",
                capacity=fork_capacity,
            )
            try:
                draft_fork = self._draft.fork_state(
                    resolved._draft_state,  # noqa: SLF001
                    parent=current.draft,
                    owner_id=f"{owner}.draft",
                    capacity=fork_capacity,
                )
            except BaseException:
                self._target.release_state(target_fork.state)
                raise
            forked_state = SpeculativeState(
                runtime=self,
                state_id=f"spec-state-{uuid4().hex}",
                owner_id=owner,
                generation=self._mint_generation(),
                epoch=1,
                target_state=target_fork.state,
                draft_state=draft_fork.state,
                pending_token_id=current.pending_token_id,
            )
            with forked_state._lock:  # noqa: SLF001
                forked_observation = self._observe_locked(forked_state)
            if forked_observation.poisoned:
                self._draft.release_state(draft_fork.state)
                self._target.release_state(target_fork.state)
                raise SpeculativeRuntimeError("forked target and draft prefixes diverged")
            with self._lock:
                self._states[forked_state.state_id] = forked_state
                self._states_forked += 1
            return SpeculativeForkResult(
                runtime_id=self._runtime_id,
                source=current,
                forked=forked_observation,
                state=forked_state,
                state_bytes_copied=(target_fork.state_bytes_copied + draft_fork.state_bytes_copied),
            )

    def release_state(self, state: SpeculativeState) -> None:
        resolved = self._state(state)
        errors: list[BaseException] = []
        with resolved._lock:  # noqa: SLF001
            for runtime, backend_state, released_field in (
                (
                    self._draft,
                    resolved._draft_state,  # noqa: SLF001
                    "_draft_released",
                ),
                (
                    self._target,
                    resolved._target_state,  # noqa: SLF001
                    "_target_released",
                ),
            ):
                if getattr(resolved, released_field):
                    continue
                try:
                    runtime.release_state(backend_state)
                except BaseException as exc:
                    errors.append(exc)
                else:
                    setattr(resolved, released_field, True)
            if resolved._draft_released and resolved._target_released:  # noqa: SLF001
                resolved._released = True  # noqa: SLF001
                with self._lock:
                    self._states.pop(resolved.state_id, None)
                    self._states_released += 1
            elif errors:
                self._poison_locked(resolved, "paired backend release was incomplete")
        if errors:
            self._record_failure()
            raise SpeculativeRuntimeError(
                "one or more paired backend states failed to release"
            ) from errors[0]

    def telemetry(self) -> SpeculativeTelemetry:
        with self._lock:
            self._require_open()
            return SpeculativeTelemetry(
                runtime_id=self._runtime_id,
                configured_proposal_tokens=self._proposal_tokens,
                states_allocated=self._states_allocated,
                states_forked=self._states_forked,
                states_released=self._states_released,
                states_live=len(self._states),
                states_poisoned=self._states_poisoned,
                prefill_calls=self._prefill_calls,
                speculative_steps=self._speculative_steps,
                draft_proposal_forwards=self._draft_proposal_forwards,
                draft_proposed_tokens=self._draft_proposed_tokens,
                target_block_calls=self._target_block_calls,
                target_verified_tokens=self._target_verified_tokens,
                draft_sync_block_calls=self._draft_sync_block_calls,
                draft_sync_tokens=self._draft_sync_tokens,
                optimized_draft_steps=self._optimized_draft_steps,
                fallback_draft_steps=self._fallback_draft_steps,
                draft_transaction_begins=self._draft_transaction_begins,
                draft_transaction_advances=self._draft_transaction_advances,
                draft_transaction_seals=self._draft_transaction_seals,
                draft_transaction_commits=self._draft_transaction_commits,
                draft_transaction_abandons=self._draft_transaction_abandons,
                accepted_draft_tokens=self._accepted_draft_tokens,
                emitted_tokens=self._emitted_tokens,
                full_accept_steps=self._full_accept_steps,
                partial_accept_steps=self._partial_accept_steps,
                zero_accept_steps=self._zero_accept_steps,
                correction_tokens=self._correction_tokens,
                bonus_tokens=self._bonus_tokens,
                ordinary_target_steps=self._ordinary_target_steps,
                proposal_state_forks=self._proposal_state_forks,
                proposal_state_releases=self._proposal_state_releases,
                proposal_prefix_bytes_copied=self._proposal_prefix_bytes_copied,
                verification_abandons=self._verification_abandons,
                failures=self._failures,
                split_commit_failures=self._split_commit_failures,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._states:
                raise SpeculativeRuntimeError("cannot close speculative runtime with live states")
            self._closed = True
        if not self._owns_runtimes:
            return
        self._target.close()
        if self._draft is not self._target:
            self._draft.close()


__all__ = [
    "SPECULATIVE_RUNTIME_ABI",
    "SPECULATIVE_STATE_SCHEMA",
    "ExactGreedySpeculativeRuntime",
    "SpeculativeForkResult",
    "SpeculativePrefillResult",
    "SpeculativeRuntimeError",
    "SpeculativeState",
    "SpeculativeStateObservation",
    "SpeculativeStepResult",
    "SpeculativeTelemetry",
]
