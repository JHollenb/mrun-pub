from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

import pytest

from mrun.runtime.contracts import (
    CommitResult,
    DecodeWork,
    NativeOutput,
    OutputMode,
    PrefillWork,
    PromotionStatus,
    ProvisionalStep,
    RuntimeRoute,
    RuntimeTelemetry,
    SamplingPolicy,
    StateForkResult,
    StateObservation,
)
from mrun.runtime.inference import (
    CompletedEvent,
    FinishReason,
    GenerationAdmissionError,
    GenerationBackpressureError,
    GenerationCancelled,
    GenerationCleanupError,
    GenerationDeadlineExceeded,
    GenerationDuplicateRequestError,
    GenerationExecutionError,
    GenerationRequest,
    GenerationServiceError,
    GenerationShutdown,
    NativeGenerationService,
    NativeSessionStore,
    SessionBusyError,
    SessionCacheStatus,
    SessionCleanupError,
    SessionIdentity,
    SessionIdentityError,
    SessionIntegrityError,
    SessionPrefixMismatch,
    ShutdownMode,
    StateRetentionOwner,
    TerminalEvent,
    TerminalStatus,
    TokenEvent,
)


@dataclass(slots=True)
class _FakeAuthority:
    runtime_id: str
    step_id: str
    state: _FakeState
    parent: StateObservation
    counts: tuple[int, ...]
    consumed: bool = False


class _FakeState:
    def __init__(
        self,
        runtime: _FakeRuntime,
        *,
        owner_id: str,
        capacity: int,
        generation: int = 0,
    ) -> None:
        self.runtime = runtime
        self.runtime_id = runtime.route.runtime_id
        self.state_id = f"state.{uuid4().hex}"
        self.owner_id = owner_id
        self.capacity = capacity
        self.generation = generation
        self.epoch = 0
        self.length = 0
        self.output_index = 0
        self.pending: _FakeAuthority | None = None
        self.released = False

    def observe(self) -> StateObservation:
        if self.released:
            raise RuntimeError("state released")
        return StateObservation(
            runtime_id=self.runtime_id,
            state_id=self.state_id,
            generation=self.generation,
            epoch=self.epoch,
            lengths=(self.length,),
            capacity=self.capacity,
            state_abi="fake-kv-v1",
            storage_generation=0,
        )


class _FakeRuntime:
    def __init__(self, scripts: dict[str, tuple[int, ...]] | None = None) -> None:
        self.route = RuntimeRoute(
            runtime_id=f"runtime.{uuid4().hex}",
            model_fingerprint="a" * 64,
            capability_fingerprint="b" * 64,
            placement_fingerprint="c" * 64,
            backend_id="fake-native",
            device_id="fake:0",
            promotion_status=PromotionStatus.CANDIDATE,
        )
        self.scripts = scripts or {}
        self.allocated: list[_FakeState] = []
        self.released: list[str] = []
        self.forward_started = threading.Event()
        self.forward_gate: threading.Event | None = None
        self.forward_error: BaseException | None = None
        self.commit_error: BaseException | None = None
        self.commit_calls = 0
        self.abandon_calls = 0
        self.release_calls = 0
        self.release_error: BaseException | None = None
        self.release_errors_by_state_id: dict[str, BaseException] = {}
        self.close_calls = 0
        self.fork_started = threading.Event()
        self.fork_gate: threading.Event | None = None
        self.fork_calls: list[tuple[StateObservation, int]] = []
        self.work_calls: list[tuple[str, tuple[int, ...], int]] = []
        self.output_requests = []

    def allocate_state(self, *, owner_id: str, batch_size: int, capacity: int) -> _FakeState:
        assert batch_size == 1
        state = _FakeState(self, owner_id=owner_id, capacity=capacity)
        self.allocated.append(state)
        return state

    def _execute(self, work: PrefillWork | DecodeWork) -> ProvisionalStep:
        state = work.state
        assert isinstance(state, _FakeState)
        if state.pending is not None:
            raise RuntimeError("pending work")
        if state.observe() != work.parent:
            raise RuntimeError("stale parent")
        self.work_calls.append(
            (
                "prefill" if isinstance(work, PrefillWork) else "decode",
                tuple(work.token_rows[0]),
                work.parent.lengths[0],
            )
        )
        self.output_requests.append(work.output)
        self.forward_started.set()
        if self.forward_gate is not None:
            assert self.forward_gate.wait(5)
        if self.forward_error is not None:
            raise self.forward_error
        script = self.scripts.get(work.request_ids[0], (1, 2, 3, 4, 5, 6, 7, 8))
        if state.output_index >= len(script):
            raise RuntimeError("fake script exhausted")
        step_id = f"step.{uuid4().hex}"
        authority = _FakeAuthority(
            runtime_id=self.route.runtime_id,
            step_id=step_id,
            state=state,
            parent=work.parent,
            counts=(len(work.token_rows[0]),),
        )
        state.pending = authority
        return ProvisionalStep(
            runtime_id=self.route.runtime_id,
            step_id=step_id,
            request_ids=work.request_ids,
            state=state,
            parent=work.parent,
            token_counts=authority.counts,
            output=NativeOutput(
                mode=work.output.mode,
                token_ids=(script[state.output_index],),
            ),
            authority=authority,
        )

    def prefill(self, work: PrefillWork) -> ProvisionalStep:
        return self._execute(work)

    def decode(self, work: DecodeWork) -> ProvisionalStep:
        return self._execute(work)

    def commit(self, step: ProvisionalStep, accepted_counts: Any) -> CommitResult:
        self.commit_calls += 1
        authority = step.authority
        assert isinstance(authority, _FakeAuthority)
        state = authority.state
        if self.commit_error is not None:
            raise self.commit_error
        if authority.consumed or state.pending is not authority:
            raise RuntimeError("stale authority")
        accepted = tuple(accepted_counts)
        if accepted != authority.counts:
            raise RuntimeError("wrong accepted count")
        before = state.observe()
        state.length += accepted[0]
        state.epoch += 1
        state.output_index += 1
        authority.consumed = True
        state.pending = None
        after = state.observe()
        return CommitResult(
            runtime_id=self.route.runtime_id,
            step_id=step.step_id,
            state_id=state.state_id,
            accepted_counts=accepted,
            before=before,
            after=after,
            state_bytes_written=accepted[0] * 8,
        )

    def abandon(self, step: ProvisionalStep) -> None:
        self.abandon_calls += 1
        authority = step.authority
        assert isinstance(authority, _FakeAuthority)
        if authority.consumed or authority.state.pending is not authority:
            raise RuntimeError("authority already consumed")
        authority.consumed = True
        authority.state.pending = None

    def fork_state(
        self,
        source: _FakeState,
        *,
        parent: StateObservation,
        owner_id: str,
        capacity: int,
    ) -> StateForkResult:
        if source.observe() != parent:
            raise RuntimeError("stale fork parent")
        self.fork_calls.append((parent, capacity))
        self.fork_started.set()
        if self.fork_gate is not None:
            assert self.fork_gate.wait(5)
        forked = _FakeState(
            self,
            owner_id=owner_id,
            capacity=capacity,
            generation=parent.generation + 1,
        )
        forked.length = parent.lengths[0]
        self.allocated.append(forked)
        return StateForkResult(
            runtime_id=self.route.runtime_id,
            source=parent,
            forked=forked.observe(),
            state=forked,
            state_bytes_copied=forked.length * 8,
        )

    def release_state(self, state: Any) -> None:
        self.release_calls += 1
        assert isinstance(state, _FakeState)
        if self.release_error is not None:
            raise self.release_error
        if error := self.release_errors_by_state_id.get(state.state_id):
            raise error
        if state.pending is not None:
            raise RuntimeError("pending state")
        if state.released:
            raise RuntimeError("double release")
        state.released = True
        self.released.append(state.state_id)

    def telemetry(self) -> RuntimeTelemetry:
        return RuntimeTelemetry(
            runtime_id=self.route.runtime_id,
            route_backend_id=self.route.backend_id,
            model_fingerprint=self.route.model_fingerprint,
            placement_fingerprint=self.route.placement_fingerprint,
        )

    def close(self) -> None:
        self.close_calls += 1


class _TickClock:
    def __init__(self, start: float = 0.0, step: float = 0.25) -> None:
        self.value = start
        self.step = step
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            result = self.value
            self.value += self.step
            return result

    def set(self, value: float) -> None:
        with self.lock:
            self.value = value


def _service(
    runtime: _FakeRuntime,
    **kwargs: Any,
) -> NativeGenerationService:
    kwargs.setdefault(
        "supported_output_modes",
        (OutputMode.NEXT_TOKEN_ARGMAX, OutputMode.NEXT_TOKEN_SAMPLE),
    )
    return NativeGenerationService(
        runtime,
        max_context_tokens=32,
        semantic_token_count=32,
        max_active_requests=4,
        max_new_tokens=16,
        event_queue_capacity=16,
        **kwargs,
    )


def _events(handle: Any) -> list[Any]:
    return list(handle.iter_events(timeout=2))


def _session_identity(runtime: _FakeRuntime, *, model_id: str = "toy-chat") -> SessionIdentity:
    return SessionIdentity(
        model_id=model_id,
        model_fingerprint=runtime.route.model_fingerprint,
        chat_template_sha256="d" * 64,
        semantic_token_count=32,
        route_id="route.test",
        runtime_id=runtime.route.runtime_id,
        capability_fingerprint=runtime.route.capability_fingerprint,
        placement_fingerprint=runtime.route.placement_fingerprint,
        backend_id=runtime.route.backend_id,
        device_id=runtime.route.device_id,
        state_abi="fake-kv-v1",
        execution_shape_fingerprint=runtime.route.execution_shape_fingerprint,
    )


def _session_service(
    runtime: _FakeRuntime,
    *,
    clock: Any | None = None,
    ttl_seconds: float = 60.0,
    max_entries: int = 4,
    max_bytes: int = 4096,
) -> tuple[NativeGenerationService, NativeSessionStore]:
    store = NativeSessionStore(
        runtime,
        identity=_session_identity(runtime),
        state_bytes_per_token=8,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
        max_bytes=max_bytes,
        **({} if clock is None else {"clock": clock}),
    )
    service = _service(runtime, session_store=store)
    return service, store


def test_session_identity_requires_the_live_execution_shape_fingerprint() -> None:
    runtime = _FakeRuntime()
    runtime.route = replace(
        runtime.route,
        promotion_status=PromotionStatus.EXPERIMENTAL,
        effective_numerical_contract="chunked-test-v1",
        execution_shape_fingerprint="e" * 64,
    )
    matching = _session_identity(runtime)
    assert matching.execution_shape_fingerprint == "e" * 64

    with pytest.raises(SessionIdentityError, match="live runtime route"):
        NativeSessionStore(
            runtime,
            identity=replace(matching, execution_shape_fingerprint=None),
            state_bytes_per_token=8,
        )

    store = NativeSessionStore(
        runtime,
        identity=matching,
        state_bytes_per_token=8,
    )
    assert store.identity.fingerprint == matching.fingerprint
    store.close()


def test_session_store_accounts_fixed_recurrent_state_independent_of_capacity() -> None:
    runtime = _FakeRuntime()
    store = NativeSessionStore(
        runtime,
        identity=_session_identity(runtime),
        state_bytes_per_token=0,
        state_fixed_bytes_per_row=64,
        max_entries=2,
        max_bytes=128,
    )
    lease = store.acquire(
        session_id="mamba-session",
        request_id="request-1",
        identity=store.identity,
        prompt_token_ids=(1, 2),
        state_capacity=32_768,
    )
    assert store.telemetry().reserved_bytes == 64
    assert store.abort(lease)
    assert store.telemetry().reserved_bytes == 0
    store.close()


def test_streamed_and_nonstreamed_generation_are_token_identical_and_post_commit() -> None:
    runtime = _FakeRuntime({"stream": (4, 5, 6), "plain": (4, 5, 6)})
    service = _service(runtime)
    streamed = service.submit(GenerationRequest("stream", (1, 2), 3, stream=True))
    plain = service.submit(GenerationRequest("plain", (1, 2), 3, stream=False))
    streamed_result = streamed.result(timeout=2)
    plain_result = plain.result(timeout=2)
    stream_events = _events(streamed)
    plain_events = _events(plain)

    assert streamed_result.token_ids == plain_result.token_ids == (4, 5, 6)
    assert streamed_result.finish_reason is FinishReason.MAX_NEW_TOKENS
    token_events = [event for event in stream_events if isinstance(event, TokenEvent)]
    assert tuple(event.token_id for event in token_events) == streamed_result.token_ids
    assert [event.state_epoch for event in token_events] == [1, 2, 3]
    assert all(event.published_at >= event.committed_at for event in token_events)
    assert isinstance(stream_events[-1], CompletedEvent)
    assert plain_events == [CompletedEvent("plain", plain_result, plain_events[0].created_at)]
    assert runtime.commit_calls == 6
    assert runtime.release_calls == 2
    assert service.shutdown(wait=True)
    assert service.telemetry().reconciled


def test_sampling_policy_is_row_local_counted_and_countered_only_after_commit() -> None:
    runtime = _FakeRuntime({"sample": (4, 4, 5)})
    service = _service(runtime)
    policy = SamplingPolicy(
        seed=8675309,
        temperature=0.7,
        top_p=0.9,
        top_k=8,
        frequency_penalty=0.25,
        presence_penalty=-0.5,
        logit_bias=((7, 1.25),),
    )
    result = service.submit(GenerationRequest("sample", (1, 1, 2), 3, sampling=policy)).result(
        timeout=2
    )

    assert result.token_ids == (4, 4, 5)
    assert [request.mode for request in runtime.output_requests] == [
        OutputMode.NEXT_TOKEN_SAMPLE,
        OutputMode.NEXT_TOKEN_SAMPLE,
        OutputMode.NEXT_TOKEN_SAMPLE,
    ]
    dynamic = [request.sampling[0] for request in runtime.output_requests]
    assert [request.rng_counter for request in dynamic] == [0, 1, 2]
    assert dynamic[0].token_counts == ((1, 2), (2, 1))
    assert dynamic[1].token_counts == ((1, 2), (2, 1), (4, 1))
    assert dynamic[2].token_counts == ((1, 2), (2, 1), (4, 2))
    assert all(request.policy is policy for request in dynamic)
    assert service.shutdown(wait=True)


def test_argmax_only_service_rejects_sampling_synchronously_before_state_allocation() -> None:
    runtime = _FakeRuntime({"greedy": (4,)})
    service = _service(
        runtime,
        supported_output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,),
    )
    assert service.supported_output_modes == (OutputMode.NEXT_TOKEN_ARGMAX,)

    with pytest.raises(GenerationAdmissionError, match="greedy argmax only"):
        service.submit(
            GenerationRequest(
                "sampled",
                (1,),
                1,
                sampling=SamplingPolicy(seed=17, temperature=0.7),
            )
        )
    assert runtime.allocated == []
    assert runtime.work_calls == []

    result = service.submit(
        GenerationRequest(
            "greedy",
            (1,),
            1,
            sampling=SamplingPolicy(seed=19, temperature=0.0),
        )
    ).result(timeout=2)
    assert result.token_ids == (4,)
    assert runtime.output_requests[0].mode is OutputMode.NEXT_TOKEN_ARGMAX
    assert service.telemetry().rejected_admission == 1
    assert service.shutdown(wait=True)


def test_exact_stop_sequence_overlap_is_hidden_across_decode_boundaries() -> None:
    runtime = _FakeRuntime({"stop": (7, 8, 9, 10)})
    service = _service(runtime)
    handle = service.submit(
        GenerationRequest(
            "stop",
            (1,),
            4,
            eos_token_ids=(9,),
            stop_sequences=((8, 9), (9,)),
        )
    )
    result = handle.result(timeout=2)
    events = _events(handle)

    assert result.token_ids == (7,)
    assert result.finish_reason is FinishReason.STOP_SEQUENCE
    assert result.matched_stop_sequence == (8, 9)
    assert result.model_generated_token_count == 3
    assert result.final_state_length == 3  # prompt plus the first two generated decode inputs
    assert [event.token_id for event in events if isinstance(event, TokenEvent)] == [7]
    assert runtime.release_calls == 1
    service.shutdown()


def test_partial_stop_prefix_flushes_at_max_tokens_and_eos_set_can_be_included() -> None:
    runtime = _FakeRuntime({"partial": (7, 8), "eos": (3, 4)})
    service = _service(runtime)
    partial = service.submit(GenerationRequest("partial", (1,), 2, stop_sequences=((8, 9),)))
    eos = service.submit(
        GenerationRequest(
            "eos",
            (1,),
            4,
            eos_token_ids=(4, 11),
            include_stop_tokens=True,
        )
    )
    partial_result = partial.result(timeout=2)
    eos_result = eos.result(timeout=2)
    assert partial_result.token_ids == (7, 8)
    assert partial_result.finish_reason is FinishReason.MAX_NEW_TOKENS
    assert eos_result.token_ids == (3, 4)
    assert eos_result.finish_reason is FinishReason.EOS_TOKEN
    assert eos_result.matched_stop_sequence == (4,)
    service.shutdown()


def test_slow_stream_consumer_triggers_explicit_fail_fast_backpressure_cleanup() -> None:
    runtime = _FakeRuntime({"slow": (1, 2, 3, 4)})
    service = NativeGenerationService(
        runtime,
        max_context_tokens=16,
        semantic_token_count=16,
        max_active_requests=1,
        supported_output_modes=(
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ),
        event_queue_capacity=3,  # two token slots plus the reserved terminal slot
    )
    handle = service.submit(GenerationRequest("slow", (0,), 4))
    with pytest.raises(GenerationBackpressureError, match="event queue exhausted"):
        handle.result(timeout=2)
    events = _events(handle)
    assert [event.token_id for event in events if isinstance(event, TokenEvent)] == [1, 2]
    assert isinstance(events[-1], TerminalEvent)
    assert events[-1].status is TerminalStatus.BACKPRESSURE
    assert runtime.release_calls == 1
    telemetry = service.telemetry()
    assert telemetry.backpressure_terminated == 1
    assert telemetry.event_queue_high_watermark == 3
    assert telemetry.reconciled
    service.shutdown()


def test_cancellation_during_forward_abandons_before_publication_and_releases_state() -> None:
    runtime = _FakeRuntime({"cancel": (4,)})
    runtime.forward_gate = threading.Event()
    service = _service(runtime)
    handle = service.submit(GenerationRequest("cancel", (1, 2), 1))
    assert runtime.forward_started.wait(2)
    live = service.telemetry()
    assert live.forward_steps_active == 1
    assert live.reconciled
    assert handle.cancel("disconnect")
    runtime.forward_gate.set()
    with pytest.raises(GenerationCancelled, match="disconnect"):
        handle.result(timeout=2)
    events = _events(handle)
    assert len(events) == 1 and isinstance(events[0], TerminalEvent)
    assert events[0].status is TerminalStatus.CANCELLED
    assert runtime.commit_calls == 0
    assert runtime.abandon_calls == 1
    assert runtime.release_calls == 1
    telemetry = service.telemetry()
    assert telemetry.abandons_attempted == telemetry.abandons_succeeded == 1
    assert telemetry.reconciled
    service.shutdown()


def test_deadline_crossed_during_forward_abandons_provisional_state() -> None:
    clock = _TickClock(start=10.0, step=0.0)
    runtime = _FakeRuntime({"deadline": (4,)})
    runtime.forward_gate = threading.Event()
    service = _service(runtime, clock=clock)
    handle = service.submit(GenerationRequest("deadline", (1,), 1, deadline=15.0))
    assert runtime.forward_started.wait(2)
    clock.set(20.0)
    runtime.forward_gate.set()
    with pytest.raises(GenerationDeadlineExceeded):
        handle.result(timeout=2)
    assert runtime.abandon_calls == 1
    assert runtime.release_calls == 1
    assert service.telemetry().deadline_exceeded == 1
    service.shutdown()


def test_commit_error_is_abandoned_and_released_without_a_token_event() -> None:
    runtime = _FakeRuntime({"broken": (4,)})
    runtime.commit_error = RuntimeError("injected stale commit")
    service = _service(runtime)
    handle = service.submit(GenerationRequest("broken", (1,), 1))
    with pytest.raises(GenerationExecutionError, match="injected stale commit"):
        handle.result(timeout=2)
    events = _events(handle)
    assert len(events) == 1 and isinstance(events[0], TerminalEvent)
    assert runtime.abandon_calls == 1
    assert runtime.release_calls == 1
    assert service.telemetry().failed == 1
    service.shutdown()


def test_forward_failure_releases_state_and_cleanup_failure_is_explicit() -> None:
    forward_runtime = _FakeRuntime({"forward": (4,)})
    forward_runtime.forward_error = RuntimeError("injected forward error")
    forward_service = _service(forward_runtime)
    forward = forward_service.submit(GenerationRequest("forward", (1,), 1))
    with pytest.raises(GenerationExecutionError, match="injected forward error"):
        forward.result(timeout=2)
    assert forward_runtime.abandon_calls == 0
    assert forward_runtime.release_calls == 1
    forward_service.shutdown()

    cleanup_runtime = _FakeRuntime({"cleanup": (4,)})
    cleanup_runtime.release_error = RuntimeError("injected release error")
    cleanup_service = _service(cleanup_runtime)
    cleanup = cleanup_service.submit(GenerationRequest("cleanup", (1,), 1))
    with pytest.raises(GenerationCleanupError, match="injected release error"):
        cleanup.result(timeout=2)
    telemetry = cleanup_service.telemetry()
    assert telemetry.state_releases_attempted == 1
    assert telemetry.state_releases_succeeded == 0
    assert telemetry.cleanup_failures == 1
    assert telemetry.failed == 1
    assert telemetry.reconciled
    cleanup_service.shutdown()


def test_abort_shutdown_during_forward_abandons_and_drain_rejects_new_admission() -> None:
    runtime = _FakeRuntime({"abort": (4,)})
    runtime.forward_gate = threading.Event()
    service = _service(runtime)
    handle = service.submit(GenerationRequest("abort", (1,), 1))
    assert runtime.forward_started.wait(2)
    assert not service.shutdown(ShutdownMode.ABORT, wait=False)
    runtime.forward_gate.set()
    assert service.shutdown(ShutdownMode.ABORT, wait=True, timeout=2)
    with pytest.raises(GenerationShutdown):
        handle.result(timeout=2)
    with pytest.raises(GenerationServiceError):
        service.submit(GenerationRequest("late", (1,), 1))
    assert runtime.abandon_calls == runtime.release_calls == 1


def test_admission_is_atomic_and_enforces_context_domain_history_and_duplicates() -> None:
    runtime = _FakeRuntime({"one": (1,)})
    runtime.forward_gate = threading.Event()
    service = NativeGenerationService(
        runtime,
        max_context_tokens=4,
        semantic_token_count=8,
        max_active_requests=1,
        supported_output_modes=(
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ),
        max_new_tokens=2,
        event_queue_capacity=4,
        max_request_history=2,
    )
    with pytest.raises(GenerationAdmissionError, match="state_capacity"):
        service.submit(GenerationRequest("context", (1, 2, 3, 4), 2))
    with pytest.raises(GenerationAdmissionError, match="semantic token"):
        service.submit(GenerationRequest("domain", (8,), 1))
    first = service.submit(GenerationRequest("one", (1,), 1))
    assert runtime.forward_started.wait(2)
    with pytest.raises(GenerationDuplicateRequestError):
        service.submit(GenerationRequest("one", (1,), 1))
    with pytest.raises(GenerationBackpressureError, match="active-request"):
        service.submit(GenerationRequest("two", (1,), 1))
    runtime.forward_gate.set()
    first.result(timeout=2)
    service.shutdown()


def test_retained_state_is_an_explicit_exact_one_shot_handoff() -> None:
    runtime = _FakeRuntime({"retain": (4, 5)})
    service = _service(runtime)
    handle = service.submit(GenerationRequest("retain", (1, 2), 2, retain_state_on_success=True))
    result = handle.result(timeout=2)
    assert result.state_retained
    assert result.state_handoff_id is not None
    assert runtime.release_calls == 0
    handoff = handle.take_state_handoff()
    assert handoff.handoff_id == result.state_handoff_id
    assert handoff.pending_token_id == 5
    fork = handoff.fork(runtime, owner_id="session.child", capacity=4)
    assert fork.source == handoff.observation
    assert fork.forked.lengths == handoff.observation.lengths
    assert handoff.live
    runtime.release_state(fork.state)
    state = handoff.claim(runtime)
    assert state.observe() == handoff.observation
    with pytest.raises(GenerationServiceError, match="already claimed"):
        handoff.claim(runtime)
    runtime.release_state(state)
    assert runtime.release_calls == 2
    service.shutdown()


def test_injected_clock_produces_exact_ttft_and_reconciled_latency_telemetry() -> None:
    clock = _TickClock()
    runtime = _FakeRuntime({"timed": (4,)})
    service = _service(runtime, clock=clock)
    handle = service.submit(GenerationRequest("timed", (1,), 1))
    result = handle.result(timeout=2)
    assert result.ttft_seconds == pytest.approx(1.75)
    assert result.request_latency_seconds == pytest.approx(2.0)
    telemetry = service.telemetry()
    assert telemetry.ttft.observation_count == 1
    assert telemetry.ttft.mean == pytest.approx(1.75)
    assert telemetry.forward_latency.mean == pytest.approx(0.25)
    assert telemetry.request_latency.mean == pytest.approx(2.0)
    assert telemetry.reconciled
    service.shutdown()


def test_session_miss_then_exact_prefix_hit_decodes_only_suffix_and_replaces_source() -> None:
    runtime = _FakeRuntime({"cold": (4, 5), "warm": (6, 7)})
    service, store = _session_service(runtime)

    cold = service.submit(
        GenerationRequest(
            "cold",
            (1, 2),
            2,
            stream=False,
            retain_state_on_success=True,
            session_id="thread-1",
        )
    ).result(timeout=2)
    assert cold.session_cache_status is SessionCacheStatus.MISS
    assert cold.state_retention_owner is StateRetentionOwner.SESSION_STORE
    assert cold.state_retained
    assert store.inspect("thread-1").committed_token_count == 3
    assert store.inspect("thread-1").pending_token_id == 5
    assert runtime.work_calls[:2] == [("prefill", (1, 2), 0), ("decode", (4,), 2)]

    warm_prompt = (1, 2, 4, 10, 11)
    warm_handle = service.submit(
        GenerationRequest(
            "warm",
            warm_prompt,
            2,
            stream=False,
            retain_state_on_success=True,
            session_id="thread-1",
        )
    )
    warm = warm_handle.result(timeout=2)
    assert warm.session_cache_status is SessionCacheStatus.HIT
    assert warm.state_retention_owner is StateRetentionOwner.SESSION_STORE
    assert not warm_handle.release_retained_state()
    assert runtime.fork_calls[0][0].lengths == (3,)
    assert runtime.fork_calls[0][1] == 6
    # Cached K/V covers (1, 2, 4).  The previous pending output 5 is not replayed or
    # silently treated as committed; only the newly rendered suffix (10, 11) is decoded.
    assert runtime.work_calls[2:] == [("decode", (10, 11), 3), ("decode", (6,), 5)]
    snapshot = store.inspect("thread-1")
    assert snapshot is not None
    assert snapshot.committed_token_count == 6
    assert snapshot.pending_token_id == 7
    assert snapshot.charge_bytes == 6 * 8
    telemetry = store.telemetry()
    assert telemetry.hits == telemetry.misses == telemetry.installs - 1 == 1
    assert telemetry.forks == 1
    assert telemetry.fork_tokens == 3
    assert telemetry.fork_bytes == 24
    assert telemetry.budget_reconciled
    assert runtime.release_calls == 1  # replaced cold source

    assert service.shutdown()
    store.close()
    assert runtime.release_calls == 2


def test_cross_session_content_index_uses_longest_exact_prefix_and_replays_on_miss() -> None:
    runtime = _FakeRuntime(
        {
            "short": (2, 31),
            "long-z": (3, 30),
            "long-a": (3, 29),
            "boundary": (23, 28),
            "boundary-b": (3, 27),
            "target": (4,),
            "boundary-target": (7,),
            "equal": (5,),
            "cold": (6,),
        }
    )
    service, store = _session_service(runtime, max_entries=12)

    for request_id, session_id, prompt in (
        ("short", "short", (1,)),
        ("long-z", "zzz-long", (1, 2)),
        ("long-a", "aaa-long", (1, 2)),
        ("boundary", "decimal-boundary", (1,)),
        ("boundary-b", "decimal-boundary-b", (12,)),
    ):
        result = service.submit(
            GenerationRequest(
                request_id,
                prompt,
                2,
                stream=False,
                retain_state_on_success=True,
                session_id=session_id,
            )
        ).result(timeout=2)
        assert result.session_cache_status is SessionCacheStatus.MISS

    fork_count = len(runtime.fork_calls)
    wrong_identity = replace(store.identity, chat_template_sha256="e" * 64)
    with pytest.raises(SessionIdentityError, match="bound identity"):
        store.acquire(
            session_id="wrong-template",
            request_id="wrong-template",
            identity=wrong_identity,
            prompt_token_ids=(1, 2, 3, 10),
            state_capacity=4,
        )
    assert len(runtime.fork_calls) == fork_count

    # Both long entries cover (1, 2, 3).  Longest-prefix lookup wins over (1, 2),
    # then lexical session ID is the deterministic tie-breaker.  Length framing also keeps token
    # rows such as (1, 23) and (12, 3) distinct despite the same naive decimal concatenation.
    selected_source_id = store._entries["aaa-long"].observation.state_id  # noqa: SLF001
    target = service.submit(
        GenerationRequest(
            "target",
            (1, 2, 3, 10, 11),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="new-thread",
        )
    ).result(timeout=2)
    assert target.session_cache_status is SessionCacheStatus.HIT
    assert runtime.fork_calls[-1][0].state_id == selected_source_id
    assert runtime.fork_calls[-1][0].lengths == (3,)
    assert runtime.work_calls[-1] == ("decode", (10, 11), 3)

    boundary_source_id = store._entries["decimal-boundary-b"].observation.state_id  # noqa: SLF001
    boundary_target = service.submit(
        GenerationRequest(
            "boundary-target",
            (12, 3, 10),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="boundary-target",
        )
    ).result(timeout=2)
    assert boundary_target.session_cache_status is SessionCacheStatus.HIT
    assert runtime.fork_calls[-1][0].state_id == boundary_source_id

    # A full-ledger equality is not a strict extension.  With no shorter retained exact
    # prefix for (1, 23), admission deliberately replays the prompt from a cold state.
    equal = service.submit(
        GenerationRequest(
            "equal",
            (1, 23),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="equal-thread",
        )
    ).result(timeout=2)
    assert equal.session_cache_status is SessionCacheStatus.MISS
    assert runtime.work_calls[-1] == ("prefill", (1, 23), 0)

    cold = service.submit(
        GenerationRequest(
            "cold",
            (9, 8),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="cold-thread",
        )
    ).result(timeout=2)
    assert cold.session_cache_status is SessionCacheStatus.MISS
    assert runtime.work_calls[-1] == ("prefill", (9, 8), 0)

    telemetry = store.telemetry()
    assert telemetry.cross_session_prefix_hits == 2
    assert telemetry.cross_session_prefix_tokens == 5
    assert telemetry.cross_session_prefix_bytes == 40
    assert telemetry.hits == 2
    assert telemetry.misses == 7
    assert telemetry.retired_entries == telemetry.pinned_sources == 0
    assert telemetry.reserved_slots == 0
    assert telemetry.budget_reconciled
    assert service.shutdown()
    store.close()


def test_session_prefix_identity_and_tamper_checks_fail_closed() -> None:
    runtime = _FakeRuntime({"seed": (4,)})
    service, store = _session_service(runtime)
    service.submit(
        GenerationRequest(
            "seed",
            (1, 2),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="strict",
        )
    ).result(timeout=2)

    with pytest.raises(SessionPrefixMismatch, match="strictly extend"):
        service.submit(
            GenerationRequest(
                "wrong-prefix",
                (1, 7, 8),
                1,
                retain_state_on_success=True,
                session_id="strict",
            )
        )
    with pytest.raises(SessionPrefixMismatch, match="strictly extend"):
        service.submit(
            GenerationRequest(
                "same-prefix",
                (1, 2),
                1,
                retain_state_on_success=True,
                session_id="strict",
            )
        )

    wrong_identity = _session_identity(runtime, model_id="other-model")
    with pytest.raises(SessionIdentityError, match="bound identity"):
        store.acquire(
            session_id="other",
            request_id="cross-model",
            identity=wrong_identity,
            prompt_token_ids=(1,),
            state_capacity=1,
        )

    # Deliberate in-process corruption is detected from the independently stored ledger digest.
    store._entries["strict"].committed_token_ids = (1, 3)  # noqa: SLF001
    with pytest.raises(SessionIntegrityError, match="identity/ledger"):
        store.acquire(
            session_id="strict",
            request_id="tampered",
            identity=store.identity,
            prompt_token_ids=(1, 3, 9),
            state_capacity=3,
        )
    assert store.inspect("strict") is None
    telemetry = store.telemetry()
    assert telemetry.prefix_rejections == 2
    assert telemetry.identity_rejections == 1
    assert telemetry.integrity_rejections == 1
    assert service.shutdown()
    store.close()


def test_session_single_active_lease_and_cancel_keep_the_prior_source_reusable() -> None:
    runtime = _FakeRuntime({"seed": (4,), "blocked": (5,), "retry": (6,)})
    service, store = _session_service(runtime)
    service.submit(
        GenerationRequest(
            "seed",
            (1,),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="shared",
        )
    ).result(timeout=2)
    original = store.inspect("shared")
    assert original is not None

    runtime.forward_started.clear()
    runtime.forward_gate = threading.Event()
    blocked = service.submit(
        GenerationRequest(
            "blocked",
            (1, 8),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="shared",
        )
    )
    assert runtime.forward_started.wait(2)
    with pytest.raises(SessionBusyError, match="active continuation"):
        service.submit(
            GenerationRequest(
                "concurrent",
                (1, 9),
                1,
                retain_state_on_success=True,
                session_id="shared",
            )
        )
    assert blocked.cancel("test-cancel")
    runtime.forward_gate.set()
    with pytest.raises(GenerationCancelled):
        blocked.result(timeout=2)

    after_cancel = store.inspect("shared")
    assert after_cancel is not None
    assert after_cancel.ledger_sha256 == original.ledger_sha256
    runtime.forward_gate = None
    retry = service.submit(
        GenerationRequest(
            "retry",
            (1, 10),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="shared",
        )
    ).result(timeout=2)
    assert retry.session_cache_status is SessionCacheStatus.HIT
    telemetry = store.telemetry()
    assert telemetry.busy_rejections == 1
    assert telemetry.aborts >= 1
    assert telemetry.active_leases == 0
    assert service.shutdown()
    store.close()


def test_cross_session_pin_survives_evict_replace_and_cancel_without_aba() -> None:
    runtime = _FakeRuntime({"seed": (2, 3), "replacement": (8,)})
    service, store = _session_service(runtime, max_entries=6)
    service.submit(
        GenerationRequest(
            "seed",
            (1,),
            2,
            stream=False,
            retain_state_on_success=True,
            session_id="source",
        )
    ).result(timeout=2)
    original = store.inspect("source")
    assert original is not None
    original_entry_id = store._entries["source"].entry_id  # noqa: SLF001
    original_state_id = store._entries["source"].observation.state_id  # noqa: SLF001

    runtime.fork_started.clear()
    runtime.fork_gate = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            store.acquire,
            session_id="cross-target",
            request_id="cross-target",
            identity=store.identity,
            prompt_token_ids=(1, 2, 9),
            state_capacity=3,
        )
        assert runtime.fork_started.wait(2)
        pinned = store.telemetry()
        assert pinned.entries == pinned.pinned_sources == 1
        assert pinned.retired_entries == 0
        assert pinned.active_leases == pinned.reserved_slots == 1

        assert store.evict("source")
        retired = store.telemetry()
        assert retired.entries == 0
        assert retired.retired_entries == retired.pinned_sources == 1
        assert retired.stored_bytes == original.charge_bytes
        assert runtime.release_calls == 0

        replacement = service.submit(
            GenerationRequest(
                "replacement",
                (7,),
                1,
                stream=False,
                retain_state_on_success=True,
                session_id="source",
            )
        ).result(timeout=2)
        assert replacement.session_cache_status is SessionCacheStatus.MISS
        replacement_snapshot = store.inspect("source")
        assert replacement_snapshot is not None
        replacement_entry_id = store._entries["source"].entry_id  # noqa: SLF001
        assert replacement_entry_id != original_entry_id

        runtime.fork_gate.set()
        lease = future.result(timeout=2)

    assert original_state_id in runtime.released
    assert store._entries["source"].entry_id == replacement_entry_id  # noqa: SLF001
    forked_state = store.claim_for_generation(lease)
    assert forked_state is not None
    runtime.release_state(forked_state)
    # This is the coordinator's exact cancellation cleanup path after a cross-session hit.
    assert store.abort(lease, claimed_state_released=True)

    telemetry = store.telemetry()
    assert telemetry.entries == 1
    assert telemetry.retired_entries == telemetry.pinned_sources == 0
    assert telemetry.active_leases == telemetry.reserved_slots == telemetry.reserved_bytes == 0
    assert telemetry.cross_session_prefix_hits == 1
    assert telemetry.aborts == 1
    assert telemetry.manual_evictions == 1
    assert telemetry.budget_reconciled
    assert service.shutdown()
    store.close()


def test_cross_session_ttl_retires_but_does_not_release_a_forking_source() -> None:
    clock = _ManualClock()
    runtime = _FakeRuntime({"seed": (2, 3)})
    service, store = _session_service(
        runtime,
        clock=clock,
        ttl_seconds=1,
        max_entries=4,
    )
    service.submit(
        GenerationRequest(
            "seed",
            (1,),
            2,
            stream=False,
            retain_state_on_success=True,
            session_id="source",
        )
    ).result(timeout=2)

    runtime.fork_started.clear()
    runtime.fork_gate = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            store.acquire,
            session_id="ttl-target",
            request_id="ttl-target",
            identity=store.identity,
            prompt_token_ids=(1, 2, 9),
            state_capacity=3,
        )
        assert runtime.fork_started.wait(2)
        clock.value = 2
        during = store.telemetry()
        assert during.entries == 0
        assert during.retired_entries == during.pinned_sources == 1
        assert during.ttl_evictions == 1
        assert runtime.release_calls == 0
        runtime.fork_gate.set()
        lease = future.result(timeout=2)

    assert runtime.release_calls == 1
    state = store.claim_for_generation(lease)
    assert state is not None
    runtime.release_state(state)
    assert store.abort(lease, claimed_state_released=True)
    after = store.telemetry()
    assert after.entries == after.retired_entries == after.pinned_sources == 0
    assert after.stored_bytes == after.reserved_bytes == after.reserved_slots == 0
    assert after.cross_session_prefix_hits == 1
    assert after.budget_reconciled
    assert service.shutdown()
    store.close()


def test_cross_session_source_release_failure_stays_charged_until_close_retry() -> None:
    runtime = _FakeRuntime({"seed": (2, 3)})
    service, store = _session_service(runtime, max_entries=4)
    service.submit(
        GenerationRequest(
            "seed",
            (1,),
            2,
            stream=False,
            retain_state_on_success=True,
            session_id="source",
        )
    ).result(timeout=2)
    source_snapshot = store.inspect("source")
    assert source_snapshot is not None
    source_state_id = store._entries["source"].observation.state_id  # noqa: SLF001

    runtime.fork_started.clear()
    runtime.fork_gate = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            store.acquire,
            session_id="failed-target",
            request_id="failed-target",
            identity=store.identity,
            prompt_token_ids=(1, 2, 9),
            state_capacity=3,
        )
        assert runtime.fork_started.wait(2)
        assert store.evict("source")
        runtime.release_errors_by_state_id[source_state_id] = RuntimeError("source release failed")
        runtime.fork_gate.set()
        with pytest.raises(SessionCleanupError, match="source release failed"):
            future.result(timeout=2)

    # The fork child was cleaned, while the failed source remains an opaque retained authority.
    assert source_state_id not in runtime.released
    assert len(runtime.released) == 1
    failed = store.telemetry()
    assert failed.entries == 0
    assert failed.retired_entries == 1
    assert failed.pinned_sources == 0
    assert failed.stored_bytes == source_snapshot.charge_bytes
    assert failed.active_leases == failed.reserved_bytes == failed.reserved_slots == 0
    assert failed.cleanup_failures == 1
    assert failed.cross_session_prefix_hits == 0
    assert failed.cross_session_prefix_tokens == failed.cross_session_prefix_bytes == 0
    assert failed.poisoned
    assert failed.budget_reconciled

    assert service.shutdown()
    runtime.release_errors_by_state_id.clear()
    store.close()
    closed = store.telemetry()
    assert closed.entries == closed.retired_entries == closed.stored_bytes == 0
    assert not closed.accepting
    assert source_state_id in runtime.released


def test_same_session_replacement_release_failure_rolls_back_new_state_and_stays_charged() -> None:
    runtime = _FakeRuntime({"seed": (2, 3), "extend": (4,)})
    service, store = _session_service(runtime, max_entries=4)
    service.submit(
        GenerationRequest(
            "seed",
            (1,),
            2,
            stream=False,
            retain_state_on_success=True,
            session_id="source",
        )
    ).result(timeout=2)
    source = store.inspect("source")
    assert source is not None
    source_state_id = store._entries["source"].observation.state_id  # noqa: SLF001
    runtime.release_errors_by_state_id[source_state_id] = RuntimeError("replacement release failed")

    failed = service.submit(
        GenerationRequest(
            "extend",
            (1, 2, 9),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="source",
        )
    )
    with pytest.raises(GenerationCleanupError, match="replacement release failed"):
        failed.result(timeout=2)

    telemetry = store.telemetry()
    assert telemetry.entries == 0
    assert telemetry.retired_entries == 1
    assert telemetry.stored_bytes == source.charge_bytes
    assert telemetry.active_leases == telemetry.reserved_bytes == telemetry.reserved_slots == 0
    assert telemetry.replacements == 0
    assert telemetry.cleanup_failures == 1
    assert telemetry.poisoned
    assert telemetry.budget_reconciled
    assert source_state_id not in runtime.released
    assert len(runtime.released) == 1  # completed replacement child rolled back

    assert service.shutdown()
    runtime.release_errors_by_state_id.clear()
    store.close()
    assert source_state_id in runtime.released


class _ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_session_ttl_lru_and_capacity_accounting_are_deterministic() -> None:
    clock = _ManualClock()
    runtime = _FakeRuntime(
        {
            "s1": (2,),
            "s2": (3,),
            "touch": (4,),
            "s3": (5,),
        }
    )
    service, store = _session_service(
        runtime,
        clock=clock,
        ttl_seconds=10,
        max_entries=2,
        max_bytes=64,
    )

    for request_id, session_id in (("s1", "one"), ("s2", "two")):
        service.submit(
            GenerationRequest(
                request_id,
                (1,),
                1,
                stream=False,
                retain_state_on_success=True,
                session_id=session_id,
            )
        ).result(timeout=2)
    service.submit(
        GenerationRequest(
            "touch",
            (1, 9),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="one",
        )
    ).result(timeout=2)
    service.submit(
        GenerationRequest(
            "s3",
            (1,),
            1,
            stream=False,
            retain_state_on_success=True,
            session_id="three",
        )
    ).result(timeout=2)

    assert store.inspect("one") is not None
    assert store.inspect("two") is None
    assert store.inspect("three") is not None
    before_ttl = store.telemetry()
    assert before_ttl.lru_evictions == 1
    assert before_ttl.entries == 2
    assert before_ttl.stored_bytes == 24  # capacities 2 and 1, charged at eight bytes/token
    assert before_ttl.reserved_bytes == 0
    assert before_ttl.budget_reconciled

    clock.value = 11.0
    after_ttl = store.telemetry()
    assert after_ttl.entries == 0
    assert after_ttl.stored_bytes == 0
    assert after_ttl.ttl_evictions == 2
    assert service.shutdown()
    store.close()
