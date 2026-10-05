"""Production lifecycle coordinator over :class:`mrun.runtime.ModelRuntime`.

One coordinator thread is the sole state mutation authority.  The default advances one native B1
step at a time in round-robin order.  A separately supplied compatible-batch lane may coalesce a
bounded wave while retaining independently committable state authorities.  Singleton work keeps
the exact B1 bypass unless the lane identity explicitly opts into singleton dispatch.  A selected
token is never observable until its own provisional authority commits.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from uuid import uuid4

from mrun.runtime.contracts import (
    CommitResult,
    CompatibleBatchLane,
    DecodeWork,
    ForkableModelRuntime,
    ModelRuntime,
    OutputMode,
    OutputRequest,
    PrefillWork,
    ProvisionalStep,
    SamplingRequest,
    StateForkResult,
    StateHandle,
    StateObservation,
)

from .schemas import (
    CompatibleBatchServiceTelemetry,
    CompletedEvent,
    EventBackpressurePolicy,
    FinishReason,
    GenerationEvent,
    GenerationRequest,
    GenerationResult,
    GenerationServiceTelemetry,
    LatencyDistribution,
    StateRetentionOwner,
    TerminalEvent,
    TerminalStatus,
    TokenEvent,
)
from .session import NativeSessionStore, SessionCacheStatus, SessionLease


class GenerationServiceError(RuntimeError):
    """Base class for generation lifecycle failures."""


class GenerationServiceClosed(GenerationServiceError):
    """Admission was attempted after shutdown began."""


class GenerationAdmissionError(GenerationServiceError):
    """A request failed synchronous admission."""


class GenerationBackpressureError(GenerationAdmissionError):
    """A bounded active-request, history, or event budget was exhausted."""


class GenerationDuplicateRequestError(GenerationAdmissionError):
    """A request ID was already admitted by this service incarnation."""


class GenerationCancelled(GenerationServiceError):
    """The caller cancelled a live request."""

    def __init__(self, request_id: str, reason: str) -> None:
        super().__init__(f"generation request cancelled ({reason}): {request_id}")
        self.request_id = request_id
        self.reason = reason


class GenerationDeadlineExceeded(TimeoutError, GenerationServiceError):
    """A request crossed its absolute monotonic deadline before publication."""

    def __init__(self, request_id: str) -> None:
        super().__init__(f"generation request deadline exceeded: {request_id}")
        self.request_id = request_id


class GenerationShutdown(GenerationServiceError):
    """Abort shutdown terminated a live request."""

    def __init__(self, request_id: str) -> None:
        super().__init__(f"generation service shutdown terminated request: {request_id}")
        self.request_id = request_id


class GenerationExecutionError(GenerationServiceError):
    """The runtime violated the requested lifecycle or raised during execution."""

    def __init__(self, request_id: str, operation: str, cause: BaseException) -> None:
        super().__init__(f"generation {operation} failed for {request_id}: {cause}")
        self.request_id = request_id
        self.operation = operation
        self.cause = cause


class GenerationCleanupError(GenerationServiceError):
    """A terminal path could not consume provisional work or release state."""

    def __init__(
        self,
        request_id: str,
        primary: BaseException | None,
        cleanup_errors: Sequence[BaseException],
    ) -> None:
        details = "; ".join(str(error) for error in cleanup_errors)
        prefix = f" after {primary}" if primary is not None else ""
        super().__init__(f"generation cleanup failed for {request_id}{prefix}: {details}")
        self.request_id = request_id
        self.primary = primary
        self.cleanup_errors = tuple(cleanup_errors)


class GenerationEventTimeout(TimeoutError, GenerationServiceError):
    """No streamed event arrived within the caller's wait timeout."""


class ShutdownMode(str, Enum):
    DRAIN = "drain"
    ABORT = "abort"


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


class _BoundedLatency:
    __slots__ = ("capacity", "observation_count", "values")

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.observation_count = 0
        self.values: deque[float] = deque(maxlen=capacity)

    def add(self, value: float) -> None:
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError("latency observations must be finite and non-negative")
        self.observation_count += 1
        self.values.append(normalized)

    @staticmethod
    def _quantile(values: tuple[float, ...], q: float) -> float:
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        fraction = position - lower
        return values[lower] + (values[upper] - values[lower]) * fraction

    def snapshot(self) -> LatencyDistribution:
        ordered = tuple(sorted(self.values))
        if not ordered:
            return LatencyDistribution(
                observation_count=self.observation_count,
                window_count=0,
                window_capacity=self.capacity,
                minimum=None,
                maximum=None,
                mean=None,
                p50=None,
                p95=None,
                p99=None,
            )
        return LatencyDistribution(
            observation_count=self.observation_count,
            window_count=len(ordered),
            window_capacity=self.capacity,
            minimum=ordered[0],
            maximum=ordered[-1],
            mean=sum(ordered) / len(ordered),
            p50=self._quantile(ordered, 0.50),
            p95=self._quantile(ordered, 0.95),
            p99=self._quantile(ordered, 0.99),
        )


class _EventChannel:
    """Bounded event channel with one slot permanently reserved for terminal delivery."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._condition = threading.Condition(threading.Lock())
        self._events: deque[GenerationEvent] = deque()
        self._terminal = False
        self._high_watermark = 0

    @property
    def high_watermark(self) -> int:
        with self._condition:
            return self._high_watermark

    def publish_tokens(self, events: Sequence[TokenEvent]) -> bool:
        batch = tuple(events)
        if not batch:
            return True
        with self._condition:
            if self._terminal:
                return False
            # The final physical slot is never consumed by a token.  This guarantees that an
            # absent/slow consumer can still learn why its request was terminated.
            if len(self._events) + len(batch) > self.capacity - 1:
                return False
            self._events.extend(batch)
            self._high_watermark = max(self._high_watermark, len(self._events))
            self._condition.notify_all()
            return True

    def publish_terminal(self, event: CompletedEvent | TerminalEvent) -> None:
        with self._condition:
            if self._terminal:
                raise RuntimeError("terminal generation event was already published")
            if len(self._events) >= self.capacity:
                raise RuntimeError("reserved terminal event slot was lost")
            self._events.append(event)
            self._terminal = True
            self._high_watermark = max(self._high_watermark, len(self._events))
            self._condition.notify_all()

    def next(self, timeout: float | None) -> GenerationEvent:
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("event timeout must be a finite non-negative number or None")
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("event timeout must be a finite non-negative number or None")
        expires = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._events:
                if self._terminal:
                    raise StopIteration
                remaining = None if expires is None else expires - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise GenerationEventTimeout("timed out waiting for generation event")
                self._condition.wait(remaining)
            return self._events.popleft()


class GenerationStateHandoff:
    """One-shot ownership transfer for a successful retained native state.

    The handoff preserves the exact final observation and the pending model output that has not
    yet been appended to KV.  ``claim`` requires the same runtime object, preventing an opaque
    state authority from crossing backend/runtime incarnations.  Before claim, ``release`` is
    idempotent and remains the safe default.
    """

    def __init__(
        self,
        *,
        handoff_id: str,
        request_id: str,
        runtime: ModelRuntime,
        state: StateHandle,
        observation: StateObservation,
        pending_token_id: int,
        committed_token_ids: tuple[int, ...],
    ) -> None:
        self.handoff_id = handoff_id
        self.request_id = request_id
        self.runtime_id = observation.runtime_id
        self.observation = observation
        self.pending_token_id = pending_token_id
        ledger = tuple(committed_token_ids)
        if (
            not ledger
            or any(
                isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
                for value in ledger
            )
            or observation.batch_size != 1
            or observation.lengths != (len(ledger),)
        ):
            raise ValueError("state handoff token ledger must exactly match committed K/V")
        self.committed_token_ids = tuple(int(value) for value in ledger)
        self._runtime = runtime
        self._state: StateHandle | None = state
        self._claimed = False
        self._released = False
        self._lock = threading.Lock()

    @property
    def live(self) -> bool:
        with self._lock:
            return self._state is not None and not self._released and not self._claimed

    def claim(self, runtime: ModelRuntime) -> StateHandle:
        """Transfer the raw state authority exactly once to an exact runtime owner."""

        with self._lock:
            if runtime is not self._runtime:
                raise ValueError("state handoff can only be claimed by its exact runtime owner")
            if self._released:
                raise GenerationServiceError("state handoff was already released")
            if self._claimed or self._state is None:
                raise GenerationServiceError("state handoff was already claimed")
            state = self._state
            if state.observe() != self.observation:
                raise GenerationServiceError("retained state changed before handoff claim")
            self._state = None
            self._claimed = True
            return state

    def fork(
        self,
        runtime: ForkableModelRuntime,
        *,
        owner_id: str,
        capacity: int,
    ) -> StateForkResult:
        """Copy the retained committed prefix without surrendering source ownership.

        The returned fork is owned by the caller.  The handoff remains live until explicitly
        released or claimed, allowing a session store to mint several exact prefix children and
        then deterministically release the retained source.
        """

        with self._lock:
            if runtime is not self._runtime:
                raise ValueError("state handoff can only be forked by its exact runtime owner")
            if not isinstance(runtime, ForkableModelRuntime):
                raise TypeError("runtime does not implement exact-prefix state fork")
            if self._released:
                raise GenerationServiceError("state handoff was already released")
            if self._claimed or self._state is None:
                raise GenerationServiceError("state handoff was already claimed")
            if self._state.observe() != self.observation:
                raise GenerationServiceError("retained state changed before handoff fork")
            return runtime.fork_state(
                self._state,
                parent=self.observation,
                owner_id=owner_id,
                capacity=capacity,
            )

    def release(self) -> bool:
        """Release an unclaimed handoff; return false after an earlier release."""

        with self._lock:
            if self._claimed:
                raise GenerationServiceError(
                    "claimed state authority must be released by its owner"
                )
            if self._released:
                return False
            if self._state is None:
                raise GenerationServiceError("state handoff lost its state authority")
            self._runtime.release_state(self._state)
            self._state = None
            self._released = True
            return True


class GenerationHandle:
    """Caller handle for result, cancellation, events, and optional state handoff."""

    def __init__(
        self,
        service: NativeGenerationService,
        request_id: str,
        channel: _EventChannel,
    ) -> None:
        self.request_id = request_id
        self._service = service
        self._channel = channel
        self._future: Future[GenerationResult] = Future()
        self._handoff: GenerationStateHandoff | None = None
        self._handoff_lock = threading.Lock()

    @property
    def event_queue_capacity(self) -> int:
        return self._channel.capacity

    def cancel(self, reason: str = "caller") -> bool:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cancellation reason must be a non-empty string")
        return self._service._cancel(self.request_id, reason.strip())  # noqa: SLF001

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> GenerationResult:
        return self._future.result(timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout)

    def next_event(self, timeout: float | None = None) -> GenerationEvent:
        return self._channel.next(timeout)

    def iter_events(self, timeout: float | None = None) -> Iterator[GenerationEvent]:
        while True:
            try:
                event = self.next_event(timeout)
            except StopIteration:
                return
            yield event
            if isinstance(event, (CompletedEvent, TerminalEvent)):
                return

    def take_state_handoff(self) -> GenerationStateHandoff:
        """Take the successful request's retained state handoff exactly once."""

        # Waiting here makes the method safe immediately after submit and propagates failures.
        result = self.result()
        if not result.state_retained:
            raise GenerationServiceError("request did not retain state on success")
        if result.state_retention_owner is not StateRetentionOwner.HANDLE:
            raise GenerationServiceError("retained state is owned by the session store")
        with self._handoff_lock:
            if self._handoff is None:
                raise GenerationServiceError("state handoff was already taken")
            handoff = self._handoff
            self._handoff = None
            return handoff

    def release_retained_state(self) -> bool:
        """Release a successful handoff still owned by this handle.

        This is the safe path when a session layer decides not to claim a state it requested.
        Once :meth:`take_state_handoff` transfers the handoff, that caller owns release.
        """

        result = self.result()
        if not result.state_retained:
            return False
        if result.state_retention_owner is StateRetentionOwner.SESSION_STORE:
            return False
        with self._handoff_lock:
            if self._handoff is None:
                raise GenerationServiceError("state handoff was already taken or released")
            handoff = self._handoff
            released = handoff.release()
            self._handoff = None
            return released

    def _set_handoff(self, handoff: GenerationStateHandoff) -> None:
        with self._handoff_lock:
            if self._handoff is not None:
                raise RuntimeError("generation handle already owns a state handoff")
            self._handoff = handoff


@dataclass(frozen=True, slots=True)
class _HeldToken:
    token_id: int
    step_index: int
    committed_at: float
    state_epoch: int
    state_length: int


@dataclass(slots=True)
class _Session:
    request: GenerationRequest
    handle: GenerationHandle
    arrival_at: float
    state_capacity: int
    state: StateHandle | None = None
    prefix_lease: SessionLease | None = None
    current_ids: tuple[int, ...] = ()
    generated_raw: list[int] = field(default_factory=list)
    token_counts: dict[int, int] = field(default_factory=dict)
    visible: list[int] = field(default_factory=list)
    holdback: list[_HeldToken] = field(default_factory=list)
    publication_times: list[float] = field(default_factory=list)
    step_count: int = 0
    committed_input_count: int = 0
    final_observation: StateObservation | None = None
    cancel_status: TerminalStatus | None = None
    cancel_reason: str = ""
    terminal_decided: bool = False
    ready_at: float = 0.0
    ready_wall_at: float = 0.0


@dataclass(frozen=True, slots=True)
class _StopDecision:
    emit: tuple[_HeldToken, ...]
    reason: FinishReason | None
    matched: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class _PreparedStep:
    session: _Session
    parent: StateObservation
    output: OutputRequest
    work: PrefillWork | DecodeWork
    operation: str
    via_batch_lane: bool = False


class NativeGenerationService:
    """Bounded native generation service over one exact ``ModelRuntime`` incarnation."""

    def __init__(
        self,
        runtime: ModelRuntime,
        *,
        max_context_tokens: int,
        semantic_token_count: int,
        max_active_requests: int,
        supported_output_modes: Sequence[OutputMode],
        max_new_tokens: int = 4096,
        event_queue_capacity: int = 64,
        max_request_history: int = 65_536,
        telemetry_history: int = 512,
        backpressure_policy: EventBackpressurePolicy | str = (
            EventBackpressurePolicy.TERMINATE_REQUEST
        ),
        clock: Callable[[], float] = time.monotonic,
        owns_runtime: bool = False,
        session_store: NativeSessionStore | None = None,
        compatible_batch_lane: CompatibleBatchLane | None = None,
        thread_name: str = "mrun-native-generation",
    ) -> None:
        if not isinstance(runtime, ModelRuntime):
            raise TypeError("runtime must implement the backend-neutral ModelRuntime protocol")
        self._runtime = runtime
        self._route = runtime.route
        self._max_context = _positive_int(max_context_tokens, "max_context_tokens")
        self._semantic_tokens = _positive_int(semantic_token_count, "semantic_token_count")
        self._max_active = _positive_int(max_active_requests, "max_active_requests")
        output_modes = tuple(supported_output_modes)
        if not output_modes or any(not isinstance(mode, OutputMode) for mode in output_modes):
            raise TypeError("supported_output_modes must contain OutputMode values")
        if len(set(output_modes)) != len(output_modes):
            raise ValueError("supported_output_modes must be unique")
        if OutputMode.NEXT_TOKEN_ARGMAX not in output_modes:
            raise ValueError("generation service requires next-token-argmax support")
        self._supported_output_modes = output_modes
        self._max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self._event_capacity = _positive_int(event_queue_capacity, "event_queue_capacity")
        if self._event_capacity < 2:
            raise ValueError("event_queue_capacity must reserve token and terminal slots")
        self._max_request_history = _positive_int(
            max_request_history,
            "max_request_history",
        )
        if self._max_request_history < self._max_active:
            raise ValueError("max_request_history cannot be smaller than max_active_requests")
        history = _positive_int(telemetry_history, "telemetry_history")
        try:
            policy = (
                backpressure_policy
                if isinstance(backpressure_policy, EventBackpressurePolicy)
                else EventBackpressurePolicy(str(backpressure_policy))
            )
        except ValueError as exc:
            raise ValueError("unsupported event backpressure policy") from exc
        if not callable(clock):
            raise TypeError("clock must be callable")
        if type(owns_runtime) is not bool:
            raise TypeError("owns_runtime must be boolean")
        if session_store is not None:
            if not isinstance(session_store, NativeSessionStore):
                raise TypeError("session_store must be NativeSessionStore or None")
            if not isinstance(runtime, ForkableModelRuntime) or not session_store.binds_runtime(
                runtime
            ):
                raise ValueError("session_store must bind the service's exact forkable runtime")
            if session_store.identity.semantic_token_count != self._semantic_tokens:
                raise ValueError("session store semantic domain differs from generation service")
            if owns_runtime:
                raise ValueError(
                    "a runtime-owning generation service cannot outlive retained session state"
                )
        if compatible_batch_lane is not None:
            if not isinstance(compatible_batch_lane, CompatibleBatchLane):
                raise TypeError("compatible_batch_lane must implement CompatibleBatchLane")
            identity = compatible_batch_lane.identity
            if identity.runtime_id != self._route.runtime_id:
                raise ValueError("compatible batch lane binds a different runtime incarnation")
            if identity.max_batch_size > self._max_active:
                raise ValueError(
                    "compatible batch width cannot exceed the service active-request bound"
                )
        self._backpressure_policy = policy
        self._clock = clock
        self._owns_runtime = owns_runtime
        self._session_store = session_store
        self._compatible_batch_lane = compatible_batch_lane
        self._service_id = f"generation.{uuid4().hex}"
        self._condition = threading.Condition(threading.RLock())
        self._active: dict[str, _Session] = {}
        self._admitting: set[str] = set()
        self._ready: deque[str] = deque()
        self._seen: set[str] = set()
        self._accepting = True
        self._shutdown_mode: ShutdownMode | None = None
        self._stopped = threading.Event()

        self._admitted = 0
        self._completed = 0
        self._cancelled = 0
        self._deadline_exceeded = 0
        self._failed = 0
        self._shutdown_terminated = 0
        self._backpressure_terminated = 0
        self._rejected_admission = 0
        self._rejected_duplicate = 0
        self._rejected_capacity = 0
        self._forward_started = 0
        self._forward_succeeded = 0
        self._forward_failed = 0
        self._provisional_steps = 0
        self._commits = 0
        self._abandons_attempted = 0
        self._abandons_succeeded = 0
        self._releases_attempted = 0
        self._releases_succeeded = 0
        self._cleanup_failures = 0
        self._model_tokens = 0
        self._streamed_tokens = 0
        self._event_high_watermark = 0
        self._ttft = _BoundedLatency(history)
        self._itl = _BoundedLatency(history)
        self._request_latency = _BoundedLatency(history)
        self._forward_latency = _BoundedLatency(history)
        self._batch_dispatches = 0
        self._batch_dispatched_rows = 0
        self._batch_singleton_bypasses = 0
        self._batch_max_width = 0
        self._batch_width_histogram: dict[int, int] = {}
        self._batch_commits = 0
        self._batch_abandons = 0
        self._batch_queue_delay = _BoundedLatency(history)
        self._batch_forward_latency = _BoundedLatency(history)

        self._worker = threading.Thread(target=self._worker_main, name=thread_name, daemon=True)
        self._worker.start()

    @property
    def service_id(self) -> str:
        return self._service_id

    @property
    def runtime_id(self) -> str:
        return self._route.runtime_id

    @property
    def session_store(self) -> NativeSessionStore | None:
        return self._session_store

    @property
    def supported_output_modes(self) -> tuple[OutputMode, ...]:
        """Executor output modes authorized for this exact service route."""

        return self._supported_output_modes

    def submit(self, request: GenerationRequest) -> GenerationHandle:
        return self.submit_many((request,))[0]

    def submit_many(self, requests: Sequence[GenerationRequest]) -> tuple[GenerationHandle, ...]:
        cohort = tuple(requests)
        if not cohort:
            raise ValueError("generation request cohort cannot be empty")
        if any(not isinstance(request, GenerationRequest) for request in cohort):
            raise TypeError("requests must contain GenerationRequest values")
        ids = tuple(request.request_id for request in cohort)
        with self._condition:
            if not self._accepting:
                self._rejected_admission += len(cohort)
                raise GenerationServiceClosed("generation service is shutting down")
            if len(set(ids)) != len(ids) or any(
                request_id in self._seen or request_id in self._admitting for request_id in ids
            ):
                self._rejected_admission += len(cohort)
                self._rejected_duplicate += len(cohort)
                raise GenerationDuplicateRequestError(
                    "request IDs are unique for one service lifetime"
                )
            if len(self._seen) + len(self._admitting) + len(cohort) > self._max_request_history:
                self._rejected_admission += len(cohort)
                self._rejected_capacity += len(cohort)
                raise GenerationBackpressureError("request-history capacity is exhausted")
            if len(self._active) + len(self._admitting) + len(cohort) > self._max_active:
                self._rejected_admission += len(cohort)
                self._rejected_capacity += len(cohort)
                raise GenerationBackpressureError("active-request capacity is exhausted")

            now = self._now()
            validated: list[tuple[GenerationRequest, int, int]] = []
            try:
                for request in cohort:
                    validated.append(self._validate_admission(request, now))
            except BaseException:
                self._rejected_admission += len(cohort)
                raise

            self._admitting.update(ids)

        resources: list[tuple[SessionLease | None, StateHandle | None]] = []
        sessions: list[_Session] = []
        try:
            for request, state_capacity, event_capacity in validated:
                lease: SessionLease | None = None
                state: StateHandle | None = None
                if request.session_id is not None:
                    assert self._session_store is not None
                    lease = self._session_store.acquire(
                        session_id=request.session_id,
                        request_id=request.request_id,
                        identity=self._session_store.identity,
                        prompt_token_ids=request.input_ids,
                        state_capacity=state_capacity,
                    )
                    resources.append((lease, None))
                    state = self._session_store.claim_for_generation(lease)
                    resources[-1] = (lease, state)
                else:
                    resources.append((None, None))
                channel = _EventChannel(event_capacity)
                handle = GenerationHandle(self, request.request_id, channel)
                sessions.append(
                    _Session(
                        request=request,
                        handle=handle,
                        arrival_at=now,
                        state_capacity=state_capacity,
                        state=state,
                        prefix_lease=lease,
                        current_ids=(
                            lease.suffix_token_ids
                            if lease is not None and lease.hit
                            else request.input_ids
                        ),
                        token_counts=dict(Counter(request.input_ids)),
                        ready_at=now,
                        ready_wall_at=time.monotonic(),
                    )
                )
        except BaseException as primary:
            cleanup_error: BaseException | None = None
            try:
                self._rollback_admission_resources(resources)
            except BaseException as exc:
                cleanup_error = exc
            with self._condition:
                self._admitting.difference_update(ids)
                self._rejected_admission += len(cohort)
                self._condition.notify_all()
            if cleanup_error is not None:
                raise cleanup_error from primary
            raise

        rejected_during_admission = False
        with self._condition:
            self._admitting.difference_update(ids)
            if not self._accepting:
                self._rejected_admission += len(cohort)
                rejected_during_admission = True
            else:
                for session in sessions:
                    request_id = session.request.request_id
                    self._active[request_id] = session
                    self._ready.append(request_id)
                    self._seen.add(request_id)
                self._admitted += len(sessions)
            self._condition.notify_all()
        if rejected_during_admission:
            self._rollback_admission_resources(resources)
            raise GenerationServiceClosed("generation service shut down during admission")
        return tuple(session.handle for session in sessions)

    def _rollback_admission_resources(
        self,
        resources: Sequence[tuple[SessionLease | None, StateHandle | None]],
    ) -> None:
        errors: list[BaseException] = []
        for lease, state in reversed(tuple(resources)):
            if state is not None:
                with self._condition:
                    self._releases_attempted += 1
                try:
                    self._runtime.release_state(state)
                except BaseException as exc:
                    errors.append(exc)
                else:
                    with self._condition:
                        self._releases_succeeded += 1
            if lease is not None and self._session_store is not None:
                try:
                    self._session_store.abort(
                        lease,
                        claimed_state_released=True,
                    )
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            with self._condition:
                self._cleanup_failures += len(errors)
            raise GenerationCleanupError("session-admission", None, errors)

    def _validate_admission(
        self,
        request: GenerationRequest,
        now: float,
    ) -> tuple[GenerationRequest, int, int]:
        if (
            request.sampling is not None
            and not request.sampling.raw_argmax_equivalent
            and OutputMode.NEXT_TOKEN_SAMPLE not in self._supported_output_modes
        ):
            raise GenerationAdmissionError(
                "request requires next-token-sample, but this route supports greedy argmax only"
            )
        if request.max_new_tokens > self._max_new_tokens:
            self._rejected_capacity += 1
            raise GenerationAdmissionError(
                f"max_new_tokens exceeds service limit {self._max_new_tokens}"
            )
        required_capacity = len(request.input_ids) + request.max_new_tokens - 1
        state_capacity = request.state_capacity or required_capacity
        if state_capacity < required_capacity:
            self._rejected_capacity += 1
            raise GenerationAdmissionError(
                "state_capacity cannot hold prompt plus generated-token decode inputs"
            )
        if state_capacity > self._max_context:
            self._rejected_capacity += 1
            raise GenerationAdmissionError(
                f"state_capacity exceeds context limit {self._max_context}"
            )
        all_rows = (request.input_ids, *request.stop_sequences, request.eos_token_ids)
        if any(token >= self._semantic_tokens for row in all_rows for token in row):
            raise GenerationAdmissionError(
                f"request token IDs must be below semantic token count {self._semantic_tokens}"
            )
        if request.sampling is not None:
            if any(token >= self._semantic_tokens for token, _bias in request.sampling.logit_bias):
                raise GenerationAdmissionError(
                    "sampling logit_bias escapes the semantic token domain"
                )
        event_capacity = request.event_queue_capacity or self._event_capacity
        if event_capacity > self._event_capacity:
            self._rejected_capacity += 1
            raise GenerationAdmissionError(
                f"event_queue_capacity exceeds service limit {self._event_capacity}"
            )
        if request.deadline is not None and request.deadline <= now:
            raise GenerationDeadlineExceeded(request.request_id)
        if request.session_id is not None:
            if self._session_store is None:
                raise GenerationAdmissionError(
                    "session_id requires a configured exact-prefix session store"
                )
            if not request.retain_state_on_success:
                raise GenerationAdmissionError(
                    "session requests must retain their successful native state"
                )
        if request.retain_state_on_success and self._owns_runtime:
            raise GenerationAdmissionError(
                "retain_state_on_success is unavailable when the service owns runtime shutdown"
            )
        return request, state_capacity, event_capacity

    def _cancel(self, request_id: str, reason: str) -> bool:
        with self._condition:
            session = self._active.get(request_id)
            if session is None or session.handle.done() or session.terminal_decided:
                return False
            if session.cancel_status is None:
                session.cancel_status = TerminalStatus.CANCELLED
                session.cancel_reason = reason
                self._condition.notify_all()
            return True

    def shutdown(
        self,
        mode: ShutdownMode | str = ShutdownMode.DRAIN,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        try:
            normalized = mode if isinstance(mode, ShutdownMode) else ShutdownMode(str(mode))
        except ValueError as exc:
            raise ValueError("shutdown mode must be drain or abort") from exc
        if type(wait) is not bool:
            raise TypeError("wait must be boolean")
        with self._condition:
            self._accepting = False
            if self._shutdown_mode is None or normalized is ShutdownMode.ABORT:
                self._shutdown_mode = normalized
            if self._shutdown_mode is ShutdownMode.ABORT:
                for session in self._active.values():
                    if session.cancel_status is None and not session.terminal_decided:
                        session.cancel_status = TerminalStatus.SHUTDOWN
                        session.cancel_reason = "service abort"
            self._condition.notify_all()
        if wait:
            self._worker.join(timeout)
        return self._stopped.is_set()

    def close(self) -> None:
        self.shutdown(ShutdownMode.ABORT, wait=True)

    def __enter__(self) -> NativeGenerationService:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def telemetry(self) -> GenerationServiceTelemetry:
        with self._condition:
            compatible_batch = None
            if self._compatible_batch_lane is not None:
                compatible_batch = CompatibleBatchServiceTelemetry(
                    identity=self._compatible_batch_lane.identity,
                    dispatches=self._batch_dispatches,
                    dispatched_rows=self._batch_dispatched_rows,
                    singleton_bypasses=self._batch_singleton_bypasses,
                    max_width=self._batch_max_width,
                    width_histogram=tuple(sorted(self._batch_width_histogram.items())),
                    commits=self._batch_commits,
                    abandons=self._batch_abandons,
                    queue_delay=self._batch_queue_delay.snapshot(),
                    forward_latency=self._batch_forward_latency.snapshot(),
                )
            return GenerationServiceTelemetry(
                service_id=self._service_id,
                runtime_id=self._route.runtime_id,
                backend_id=self._route.backend_id,
                admitted=self._admitted,
                active=len(self._active),
                completed=self._completed,
                cancelled=self._cancelled,
                deadline_exceeded=self._deadline_exceeded,
                failed=self._failed,
                shutdown_terminated=self._shutdown_terminated,
                backpressure_terminated=self._backpressure_terminated,
                rejected_admission=self._rejected_admission,
                rejected_duplicate=self._rejected_duplicate,
                rejected_capacity=self._rejected_capacity,
                forward_steps_started=self._forward_started,
                forward_steps_succeeded=self._forward_succeeded,
                forward_steps_failed=self._forward_failed,
                forward_steps_active=(
                    self._forward_started - self._forward_succeeded - self._forward_failed
                ),
                provisional_steps=self._provisional_steps,
                commits=self._commits,
                abandons_attempted=self._abandons_attempted,
                abandons_succeeded=self._abandons_succeeded,
                state_releases_attempted=self._releases_attempted,
                state_releases_succeeded=self._releases_succeeded,
                cleanup_failures=self._cleanup_failures,
                model_generated_tokens=self._model_tokens,
                streamed_tokens=self._streamed_tokens,
                event_queue_high_watermark=self._event_high_watermark,
                event_queue_capacity=self._event_capacity,
                backpressure_policy=self._backpressure_policy,
                accepting=self._accepting,
                stopped=self._stopped.is_set(),
                ttft=self._ttft.snapshot(),
                inter_token=self._itl.snapshot(),
                request_latency=self._request_latency.snapshot(),
                forward_latency=self._forward_latency.snapshot(),
                compatible_batch=compatible_batch,
            )

    def _worker_main(self) -> None:
        try:
            while True:
                sessions = self._next_ready_cohort()
                if not sessions:
                    return
                try:
                    if self._compatible_batch_lane is not None and (
                        len(sessions) > 1
                        or self._compatible_batch_lane.identity.dispatches_singletons
                    ):
                        continuing = self._advance_batch(sessions)
                    else:
                        if self._compatible_batch_lane is not None:
                            with self._condition:
                                self._batch_singleton_bypasses += 1
                        continuing = (sessions[0],) if self._advance(sessions[0]) else ()
                except BaseException as exc:  # coordinator survival is a service invariant
                    for session in sessions:
                        if session.request.request_id in self._active:
                            self._terminate(
                                session,
                                TerminalStatus.FAILED,
                                GenerationExecutionError(
                                    session.request.request_id,
                                    "coordinator",
                                    exc,
                                ),
                            )
                    continuing = ()
                if continuing:
                    with self._condition:
                        ready_at = self._now() if self._compatible_batch_lane is not None else 0.0
                        ready_wall_at = time.monotonic()
                        for session in continuing:
                            if session.request.request_id in self._active:
                                if self._compatible_batch_lane is not None:
                                    session.ready_at = ready_at
                                    session.ready_wall_at = ready_wall_at
                                self._ready.append(session.request.request_id)
                        self._condition.notify_all()
        finally:
            if self._owns_runtime:
                try:
                    self._runtime.close()
                except BaseException:
                    with self._condition:
                        self._cleanup_failures += 1
            self._stopped.set()

    def _next_ready_cohort(self) -> tuple[_Session, ...]:
        """Take one bounded scheduling wave without delaying true singleton traffic."""

        with self._condition:
            while True:
                while self._ready and self._ready[0] not in self._active:
                    self._ready.popleft()
                if not self._ready:
                    if not self._active and not self._admitting and not self._accepting:
                        return ()
                    self._condition.wait()
                    continue
                lane = self._compatible_batch_lane
                if lane is None:
                    request_id = self._ready.popleft()
                    session = self._active.get(request_id)
                    return () if session is None else (session,)

                identity = lane.identity
                available = min(len(self._ready), identity.max_batch_size)
                if available == 1 and len(self._active) > 1 and identity.max_queue_delay_seconds:
                    first = self._active.get(self._ready[0])
                    if first is None:
                        self._ready.popleft()
                        continue
                    remaining = (
                        first.ready_wall_at + identity.max_queue_delay_seconds - time.monotonic()
                    )
                    if remaining > 0:
                        self._condition.wait(remaining)
                        continue
                width = available
                selected: list[_Session] = []
                for _ in range(width):
                    request_id = self._ready.popleft()
                    session = self._active.get(request_id)
                    if session is not None:
                        selected.append(session)
                if not selected:
                    continue
                dispatched_at = self._now()
                actual_width = len(selected)
                self._batch_dispatches += 1
                self._batch_dispatched_rows += actual_width
                self._batch_max_width = max(self._batch_max_width, actual_width)
                self._batch_width_histogram[actual_width] = (
                    self._batch_width_histogram.get(actual_width, 0) + 1
                )
                for session in selected:
                    self._batch_queue_delay.add(max(dispatched_at - session.ready_at, 0.0))
                return tuple(selected)

    def _prepare_step(
        self,
        session: _Session,
        *,
        via_batch_lane: bool = False,
    ) -> _PreparedStep | None:
        signal = self._terminal_signal(session)
        if signal is not None:
            status, error = signal
            self._terminate(session, status, error)
            return None
        if session.state is None:
            try:
                session.state = self._runtime.allocate_state(
                    owner_id=f"{self._service_id}:{session.request.request_id}",
                    batch_size=1,
                    capacity=session.state_capacity,
                )
            except BaseException as exc:
                self._terminate(
                    session,
                    TerminalStatus.FAILED,
                    GenerationExecutionError(session.request.request_id, "allocate_state", exc),
                )
                return None
            signal = self._terminal_signal(session)
            if signal is not None:
                status, error = signal
                self._terminate(session, status, error)
                return None

        assert session.state is not None
        try:
            parent = session.state.observe()
            policy = session.request.sampling
            if policy is None or policy.raw_argmax_equivalent:
                output = OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX)
            else:
                output = OutputRequest(
                    OutputMode.NEXT_TOKEN_SAMPLE,
                    sampling=(
                        SamplingRequest(
                            policy=policy,
                            token_counts=tuple(sorted(session.token_counts.items())),
                            rng_counter=len(session.generated_raw),
                        ),
                    ),
                )
            if not any(parent.lengths):
                work: PrefillWork | DecodeWork = PrefillWork(
                    request_ids=(session.request.request_id,),
                    token_rows=(session.current_ids,),
                    state=session.state,
                    parent=parent,
                    output=output,
                )
                operation = "prefill"
            else:
                work = DecodeWork(
                    request_ids=(session.request.request_id,),
                    token_rows=(session.current_ids,),
                    state=session.state,
                    parent=parent,
                    output=output,
                )
                operation = "decode"
        except BaseException as exc:
            self._terminate(
                session,
                TerminalStatus.FAILED,
                GenerationExecutionError(session.request.request_id, "prepare", exc),
            )
            return None
        return _PreparedStep(
            session=session,
            parent=parent,
            output=output,
            work=work,
            operation=operation,
            via_batch_lane=via_batch_lane,
        )

    def _advance(self, session: _Session) -> bool:
        prepared = self._prepare_step(session)
        if prepared is None:
            return False
        started = self._now()
        with self._condition:
            self._forward_started += 1
        try:
            step = (
                self._runtime.prefill(prepared.work)
                if prepared.operation == "prefill"
                else self._runtime.decode(prepared.work)
            )
            if not isinstance(step, ProvisionalStep):
                raise TypeError("runtime did not return ProvisionalStep")
        except BaseException as exc:
            ended = self._now()
            with self._condition:
                self._forward_failed += 1
                self._forward_latency.add(max(ended - started, 0.0))
            self._terminate(
                session,
                TerminalStatus.FAILED,
                GenerationExecutionError(session.request.request_id, prepared.operation, exc),
            )
            return False
        ended = self._now()
        with self._condition:
            self._forward_succeeded += 1
            self._provisional_steps += 1
            self._forward_latency.add(max(ended - started, 0.0))
        return self._finalize_step(prepared, step)

    def _advance_batch(self, sessions: Sequence[_Session]) -> tuple[_Session, ...]:
        lane = self._compatible_batch_lane
        if lane is None:
            raise RuntimeError("compatible batch dispatch requires an attached lane")
        prepared = tuple(
            value
            for session in sessions
            if (value := self._prepare_step(session, via_batch_lane=True)) is not None
        )
        if not prepared:
            return ()
        if len(prepared) == 1 and not lane.identity.dispatches_singletons:
            with self._condition:
                self._batch_singleton_bypasses += 1
            return (prepared[0].session,) if self._execute_prepared_b1(prepared[0]) else ()

        started = self._now()
        with self._condition:
            self._forward_started += len(prepared)
        steps: tuple[ProvisionalStep, ...] = ()
        try:
            returned = lane.execute(tuple(value.work for value in prepared))
            steps = tuple(returned)
            if len(steps) != len(prepared) or any(
                not isinstance(step, ProvisionalStep) for step in steps
            ):
                raise TypeError("compatible batch lane did not return one provisional row per work")
        except BaseException as exc:
            ended = self._now()
            cleanup_errors: list[BaseException] = []
            for step in reversed(steps):
                if not isinstance(step, ProvisionalStep):
                    continue
                with self._condition:
                    self._abandons_attempted += 1
                try:
                    self._runtime.abandon(step)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                else:
                    with self._condition:
                        self._abandons_succeeded += 1
                        self._batch_abandons += 1
            with self._condition:
                self._forward_failed += len(prepared)
                latency = max(ended - started, 0.0)
                self._forward_latency.add(latency)
                self._batch_forward_latency.add(latency)
                if cleanup_errors:
                    self._cleanup_failures += len(cleanup_errors)
            failure: BaseException = exc
            if cleanup_errors:
                failure = GenerationCleanupError("compatible-batch", exc, cleanup_errors)
            for value in prepared:
                self._terminate(
                    value.session,
                    TerminalStatus.FAILED,
                    GenerationExecutionError(
                        value.session.request.request_id,
                        "compatible_batch",
                        failure,
                    ),
                )
            return ()
        ended = self._now()
        with self._condition:
            self._forward_succeeded += len(prepared)
            self._provisional_steps += len(prepared)
            latency = max(ended - started, 0.0)
            self._forward_latency.add(latency)
            self._batch_forward_latency.add(latency)
        continuing: list[_Session] = []
        for value, step in zip(prepared, steps, strict=True):
            if self._finalize_step(value, step):
                continuing.append(value.session)
        return tuple(continuing)

    def _execute_prepared_b1(self, prepared: _PreparedStep) -> bool:
        """Preserve the exact runtime path when a planned wave collapses to one live row."""

        started = self._now()
        with self._condition:
            self._forward_started += 1
        try:
            step = (
                self._runtime.prefill(prepared.work)
                if prepared.operation == "prefill"
                else self._runtime.decode(prepared.work)
            )
            if not isinstance(step, ProvisionalStep):
                raise TypeError("runtime did not return ProvisionalStep")
        except BaseException as exc:
            ended = self._now()
            with self._condition:
                self._forward_failed += 1
                latency = max(ended - started, 0.0)
                self._forward_latency.add(latency)
                self._batch_forward_latency.add(latency)
            self._terminate(
                prepared.session,
                TerminalStatus.FAILED,
                GenerationExecutionError(
                    prepared.session.request.request_id,
                    prepared.operation,
                    exc,
                ),
            )
            return False
        ended = self._now()
        with self._condition:
            self._forward_succeeded += 1
            self._provisional_steps += 1
            latency = max(ended - started, 0.0)
            self._forward_latency.add(latency)
            self._batch_forward_latency.add(latency)
        return self._finalize_step(prepared, step)

    def _finalize_step(self, prepared: _PreparedStep, step: ProvisionalStep) -> bool:
        session = prepared.session
        parent = prepared.parent
        output = prepared.output

        signal = self._terminal_signal(session)
        if signal is not None:
            status, error = signal
            if prepared.via_batch_lane:
                with self._condition:
                    self._batch_abandons += 1
            self._terminate(session, status, error, step=step)
            return False

        expected_counts = (len(session.current_ids),)
        try:
            if step.runtime_id != self._route.runtime_id:
                raise ValueError("provisional step runtime identity drifted")
            if step.request_ids != (session.request.request_id,):
                raise ValueError("provisional step request identity drifted")
            if step.parent != parent or step.state is not session.state:
                raise ValueError("provisional step state parent drifted")
            if step.token_counts != expected_counts:
                raise ValueError("provisional token counts differ from submitted input")
            if step.output.mode is not output.mode:
                raise ValueError("runtime returned a different token-selection output mode")
            if len(step.output.token_ids) != 1:
                raise ValueError("runtime returned anything other than one B1 token")
            token_id = step.output.token_ids[0]
            if token_id >= self._semantic_tokens:
                raise ValueError("runtime selected a token outside the semantic token domain")
            receipt = self._runtime.commit(step, expected_counts)
            self._validate_commit(session, parent, receipt, expected_counts)
            observed = session.state.observe()
            if observed != receipt.after:
                raise ValueError("state observation differs from commit receipt")
        except BaseException as exc:
            if prepared.via_batch_lane:
                with self._condition:
                    self._batch_abandons += 1
            self._terminate(
                session,
                TerminalStatus.FAILED,
                GenerationExecutionError(session.request.request_id, "commit", exc),
                step=step,
            )
            return False
        with self._condition:
            self._commits += 1
            if prepared.via_batch_lane:
                self._batch_commits += 1

        committed_at = self._now()
        signal = self._terminal_signal(session, now=committed_at)
        if signal is not None:
            status, error = signal
            self._terminate(session, status, error)
            return False

        session.step_count += 1
        session.committed_input_count += expected_counts[0]
        session.generated_raw.append(token_id)
        session.token_counts[token_id] = session.token_counts.get(token_id, 0) + 1
        session.final_observation = receipt.after
        session.current_ids = (token_id,)
        held = _HeldToken(
            token_id=token_id,
            step_index=session.step_count - 1,
            committed_at=committed_at,
            state_epoch=receipt.after.epoch,
            state_length=receipt.after.lengths[0],
        )
        session.holdback.append(held)
        with self._condition:
            self._model_tokens += 1

        decision = self._stop_decision(session)
        published_at = self._now()
        with self._condition:
            signal = self._terminal_signal(session, now=published_at)
            if signal is None and decision.reason is not None:
                session.terminal_decided = True
            published = (
                self._publish_visible(session, decision.emit, published_at)
                if signal is None
                else False
            )
            if signal is None and not published:
                session.terminal_decided = True
        if signal is not None:
            status, error = signal
            self._terminate(session, status, error)
            return False
        if not published:
            self._terminate(
                session,
                TerminalStatus.BACKPRESSURE,
                GenerationBackpressureError(
                    "stream event queue exhausted; request terminated without blocking other "
                    "native work"
                ),
            )
            return False
        if decision.reason is not None:
            self._complete(session, decision.reason, decision.matched)
            return False
        return True

    def _validate_commit(
        self,
        session: _Session,
        parent: StateObservation,
        receipt: CommitResult,
        accepted: tuple[int, ...],
    ) -> None:
        if not isinstance(receipt, CommitResult):
            raise TypeError("runtime commit did not return CommitResult")
        if receipt.runtime_id != self._route.runtime_id:
            raise ValueError("commit receipt runtime identity drifted")
        if receipt.before != parent or receipt.accepted_counts != accepted:
            raise ValueError("commit receipt does not bind the accepted provisional prefix")
        if session.state is None or receipt.state_id != session.state.state_id:
            raise ValueError("commit receipt state identity drifted")

    def _stop_decision(self, session: _Session) -> _StopDecision:
        request = session.request
        values = tuple(item.token_id for item in session.holdback)
        targets: list[tuple[tuple[int, ...], FinishReason, int]] = []
        for ordinal, sequence in enumerate(request.stop_sequences):
            targets.append((sequence, FinishReason.STOP_SEQUENCE, ordinal))
        offset = len(targets)
        for ordinal, token in enumerate(request.eos_token_ids):
            targets.append(((token,), FinishReason.EOS_TOKEN, offset + ordinal))
        matches = [target for target in targets if values[-len(target[0]) :] == target[0]]
        if matches:
            # Longest simultaneous match explains the maximum withheld suffix.  At equal length,
            # EOS wins over a duplicate user sequence, then request order is stable.
            sequence, reason, _ = min(
                matches,
                key=lambda item: (-len(item[0]), item[1] is not FinishReason.EOS_TOKEN, item[2]),
            )
            split = len(session.holdback) - len(sequence)
            safe = tuple(session.holdback[:split])
            stopped = tuple(session.holdback[split:])
            session.holdback.clear()
            emit = (*safe, *stopped) if request.include_stop_tokens else safe
            return _StopDecision(tuple(emit), reason, sequence)

        if len(session.generated_raw) >= request.max_new_tokens:
            emit = tuple(session.holdback)
            session.holdback.clear()
            return _StopDecision(emit, FinishReason.MAX_NEW_TOKENS, None)

        longest_prefix = 0
        for sequence, _reason, _ordinal in targets:
            for length in range(1, len(sequence)):
                if length <= len(values) and values[-length:] == sequence[:length]:
                    longest_prefix = max(longest_prefix, length)
        flush_count = len(session.holdback) - longest_prefix
        emit = tuple(session.holdback[:flush_count])
        if flush_count:
            del session.holdback[:flush_count]
        return _StopDecision(emit, None, None)

    def _publish_visible(
        self,
        session: _Session,
        held: Sequence[_HeldToken],
        published_at: float,
    ) -> bool:
        batch = tuple(held)
        if not batch:
            return True
        starting_index = len(session.visible)
        events = tuple(
            TokenEvent(
                request_id=session.request.request_id,
                token_id=item.token_id,
                token_index=starting_index + index,
                step_index=item.step_index,
                committed_at=item.committed_at,
                published_at=published_at,
                state_epoch=item.state_epoch,
                state_length=item.state_length,
            )
            for index, item in enumerate(batch)
        )
        if session.request.stream and not session.handle._channel.publish_tokens(events):  # noqa: SLF001
            return False
        session.visible.extend(item.token_id for item in batch)
        for _item in batch:
            session.publication_times.append(published_at)
        with self._condition:
            if session.request.stream:
                self._streamed_tokens += len(events)
                self._event_high_watermark = max(
                    self._event_high_watermark,
                    session.handle._channel.high_watermark,  # noqa: SLF001
                )
            published_start = len(session.publication_times) - len(batch)
            for index in range(published_start, len(session.publication_times)):
                timestamp = session.publication_times[index]
                if index == 0:
                    self._ttft.add(max(timestamp - session.arrival_at, 0.0))
                else:
                    self._itl.add(max(timestamp - session.publication_times[index - 1], 0.0))
        return True

    def _terminal_signal(
        self,
        session: _Session,
        *,
        now: float | None = None,
    ) -> tuple[TerminalStatus, BaseException] | None:
        with self._condition:
            status = session.cancel_status
            reason = session.cancel_reason
        if status is TerminalStatus.CANCELLED:
            return status, GenerationCancelled(session.request.request_id, reason or "caller")
        if status is TerminalStatus.SHUTDOWN:
            return status, GenerationShutdown(session.request.request_id)
        timestamp = self._now() if now is None else now
        if session.request.deadline is not None and timestamp >= session.request.deadline:
            return (
                TerminalStatus.DEADLINE_EXCEEDED,
                GenerationDeadlineExceeded(session.request.request_id),
            )
        return None

    def _complete(
        self,
        session: _Session,
        reason: FinishReason,
        matched: tuple[int, ...] | None,
    ) -> None:
        if session.state is None or session.final_observation is None or not session.generated_raw:
            self._terminate(
                session,
                TerminalStatus.FAILED,
                GenerationExecutionError(
                    session.request.request_id,
                    "complete",
                    RuntimeError("successful terminal state is incomplete"),
                ),
            )
            return
        handoff: GenerationStateHandoff | None = None
        cleanup_errors: list[BaseException] = []
        retention_owner = StateRetentionOwner.NONE
        session_cache_status = SessionCacheStatus.DISABLED
        committed_ledger = (
            *session.request.input_ids,
            *session.generated_raw[:-1],
        )
        if session.request.retain_state_on_success:
            handoff = GenerationStateHandoff(
                handoff_id=f"handoff.{uuid4().hex}",
                request_id=session.request.request_id,
                runtime=self._runtime,
                state=session.state,
                observation=session.final_observation,
                pending_token_id=session.generated_raw[-1],
                committed_token_ids=committed_ledger,
            )
            session.state = None
            if session.prefix_lease is not None:
                assert self._session_store is not None
                session_cache_status = session.prefix_lease.status
                try:
                    self._session_store.install(session.prefix_lease, handoff)
                except BaseException as exc:
                    cleanup_errors.append(exc)
                    try:
                        handoff.release()
                    except BaseException as release_exc:
                        cleanup_errors.append(release_exc)
                    try:
                        self._session_store.abort(
                            session.prefix_lease,
                            claimed_state_released=True,
                        )
                    except BaseException as abort_exc:
                        cleanup_errors.append(abort_exc)
                else:
                    retention_owner = StateRetentionOwner.SESSION_STORE
                session.prefix_lease = None
            else:
                retention_owner = StateRetentionOwner.HANDLE
        else:
            cleanup_errors.extend(self._release_state(session))
        if cleanup_errors:
            with self._condition:
                self._cleanup_failures += len(cleanup_errors)
            self._publish_failure(
                session,
                TerminalStatus.FAILED,
                GenerationCleanupError(session.request.request_id, None, cleanup_errors),
            )
            return

        ended = self._now()
        ttft = (
            max(session.publication_times[0] - session.arrival_at, 0.0)
            if session.publication_times
            else None
        )
        intervals = tuple(
            max(right - left, 0.0)
            for left, right in zip(
                session.publication_times,
                session.publication_times[1:],
                strict=False,
            )
        )
        observation = session.final_observation
        result = GenerationResult(
            request_id=session.request.request_id,
            token_ids=tuple(session.visible),
            finish_reason=reason,
            matched_stop_sequence=matched,
            prompt_token_count=len(session.request.input_ids),
            model_generated_token_count=len(session.generated_raw),
            committed_input_token_count=session.committed_input_count,
            final_state_length=observation.lengths[0],
            final_state_epoch=observation.epoch,
            pending_token_id=session.generated_raw[-1],
            step_count=session.step_count,
            state_capacity=session.state_capacity,
            ttft_seconds=ttft,
            inter_token_seconds=intervals,
            request_latency_seconds=max(ended - session.arrival_at, 0.0),
            runtime_id=self._route.runtime_id,
            backend_id=self._route.backend_id,
            state_retained=handoff is not None,
            state_handoff_id=handoff.handoff_id if handoff is not None else None,
            state_retention_owner=retention_owner,
            session_id=session.request.session_id,
            session_cache_status=session_cache_status,
        )
        if handoff is not None and retention_owner is StateRetentionOwner.HANDLE:
            session.handle._set_handoff(handoff)  # noqa: SLF001
        event = CompletedEvent(session.request.request_id, result, ended)
        session.handle._channel.publish_terminal(event)  # noqa: SLF001
        with self._condition:
            self._active.pop(session.request.request_id, None)
            self._completed += 1
            self._request_latency.add(result.request_latency_seconds)
            self._event_high_watermark = max(
                self._event_high_watermark,
                session.handle._channel.high_watermark,  # noqa: SLF001
            )
            self._condition.notify_all()
        session.handle._future.set_result(result)  # noqa: SLF001

    def _terminate(
        self,
        session: _Session,
        status: TerminalStatus,
        error: BaseException,
        *,
        step: ProvisionalStep | None = None,
    ) -> None:
        with self._condition:
            session.terminal_decided = True
        cleanup_errors = self._cleanup(session, step)
        if cleanup_errors:
            error = GenerationCleanupError(session.request.request_id, error, cleanup_errors)
            status = TerminalStatus.FAILED
        self._publish_failure(session, status, error)

    def _cleanup(
        self,
        session: _Session,
        step: ProvisionalStep | None,
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        if step is not None:
            with self._condition:
                self._abandons_attempted += 1
            try:
                self._runtime.abandon(step)
            except BaseException as exc:
                errors.append(exc)
            else:
                with self._condition:
                    self._abandons_succeeded += 1
        errors.extend(self._release_state(session))
        if session.prefix_lease is not None and self._session_store is not None:
            try:
                self._session_store.abort(
                    session.prefix_lease,
                    claimed_state_released=True,
                )
            except BaseException as exc:
                errors.append(exc)
            session.prefix_lease = None
        if errors:
            with self._condition:
                self._cleanup_failures += len(errors)
        return errors

    def _release_state(self, session: _Session) -> list[BaseException]:
        if session.state is None:
            return []
        state = session.state
        session.state = None
        with self._condition:
            self._releases_attempted += 1
        try:
            self._runtime.release_state(state)
        except BaseException as exc:
            return [exc]
        with self._condition:
            self._releases_succeeded += 1
        return []

    def _publish_failure(
        self,
        session: _Session,
        status: TerminalStatus,
        error: BaseException,
    ) -> None:
        ended = self._now()
        event = TerminalEvent(
            request_id=session.request.request_id,
            status=status,
            error_code=type(error).__name__,
            message=str(error),
            created_at=ended,
        )
        session.handle._channel.publish_terminal(event)  # noqa: SLF001
        with self._condition:
            self._active.pop(session.request.request_id, None)
            if status is TerminalStatus.CANCELLED:
                self._cancelled += 1
            elif status is TerminalStatus.DEADLINE_EXCEEDED:
                self._deadline_exceeded += 1
            elif status is TerminalStatus.SHUTDOWN:
                self._shutdown_terminated += 1
            elif status is TerminalStatus.BACKPRESSURE:
                self._backpressure_terminated += 1
            else:
                self._failed += 1
            self._request_latency.add(max(ended - session.arrival_at, 0.0))
            self._event_high_watermark = max(
                self._event_high_watermark,
                session.handle._channel.high_watermark,  # noqa: SLF001
            )
            self._condition.notify_all()
        session.handle._future.set_exception(error)  # noqa: SLF001

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value) or value < 0:
            raise ValueError("service clock must return finite non-negative monotonic timestamps")
        return value


__all__ = [
    "GenerationAdmissionError",
    "GenerationBackpressureError",
    "GenerationCancelled",
    "GenerationCleanupError",
    "GenerationDeadlineExceeded",
    "GenerationDuplicateRequestError",
    "GenerationEventTimeout",
    "GenerationExecutionError",
    "GenerationHandle",
    "GenerationServiceClosed",
    "GenerationServiceError",
    "GenerationShutdown",
    "GenerationStateHandoff",
    "NativeGenerationService",
    "ShutdownMode",
]
