from __future__ import annotations

import threading
from dataclasses import dataclass, replace

import pytest

from mrun.compiler import (
    OutputContract,
    build_dense_work_plan,
    decompose_work_plan,
    lower_work_template,
)
from mrun.engine.reactor import (
    BatchItemResult,
    ComponentBatch,
    ComponentReactor,
    ComponentSubmission,
    ReactorBackpressureError,
    ReactorClosedError,
    ReactorDeadlineExceeded,
    ReactorExecutorProtocolError,
    template_compatibility_key,
)


@dataclass(frozen=True)
class _Payload:
    value: str


def _template_binding(
    request_id: str,
    *,
    request_slot: int = 0,
    output_row: int = 1,
    numerical_contract: str = "reactor-test",
):
    plan = build_dense_work_plan(
        model_name="reactor-tiny",
        model_revision="a" * 64,
        store_fingerprint="b" * 64,
        batch_size=1,
        sequence_length=1,
        output_contract=OutputContract.SELECTED_TOKEN_ROWS,
        numerical_contract=numerical_contract,
        request_ids=(request_id,),
        request_slots=(request_slot,),
        required_output_rows=(output_row,),
        metadata={"engine_device": "cpu", "input_token_limit": 16},
    )
    return decompose_work_plan(plan)


class _RecordingExecutor:
    def __init__(self) -> None:
        self.batches: list[ComponentBatch[_Payload]] = []
        self.lock = threading.Lock()

    def execute_batch(self, batch: ComponentBatch[_Payload]):
        with self.lock:
            self.batches.append(batch)
        return tuple(BatchItemResult.success(payload.value) for payload in batch.payloads)


def test_reactor_coalesces_only_strictly_compatible_templates_and_routes_in_order():
    first_template, first = _template_binding("first", output_row=1)
    _, second = _template_binding("second", request_slot=1, output_row=2)
    other_template, other = _template_binding(
        "other", output_row=3, numerical_contract="different-contract"
    )
    executor = _RecordingExecutor()

    with ComponentReactor(
        executor,
        max_batch_size=4,
        max_pending=8,
        max_batch_delay_seconds=0.02,
    ) as reactor:
        first_payload = _Payload("first")
        other_payload = _Payload("other")
        second_payload = _Payload("second")
        first_future = reactor.submit(first_template, first, first_payload)
        other_future = reactor.submit(other_template, other, other_payload)
        second_future = reactor.submit(first_template, second, second_payload)

        assert first_future.result(timeout=1) == "first"
        assert second_future.result(timeout=1) == "second"
        assert other_future.result(timeout=1) == "other"
        assert reactor.wait_idle(timeout=1)
        telemetry = reactor.telemetry()

    assert [batch.dispatch_width for batch in executor.batches] == [2, 1]
    assert [binding.request_ids[0] for binding in executor.batches[0].bindings] == [
        "first",
        "second",
    ]
    assert executor.batches[0].payloads == (_Payload("first"), _Payload("second"))
    assert executor.batches[0].payloads[0] is first_payload
    assert executor.batches[0].payloads[1] is second_payload
    assert len({batch.compatibility_key for batch in executor.batches}) == 2
    assert telemetry.submitted == 3
    assert telemetry.unique_templates == 2
    assert telemetry.template_reuses == 1
    assert telemetry.coalesced_dispatches == 1
    assert telemetry.batch_width_histogram == ((1, 1), (2, 1))
    assert telemetry.batch_width_mean == pytest.approx(1.5)
    assert telemetry.queue_wait_samples == 3
    assert telemetry.queue_wait_seconds_max >= telemetry.queue_wait_seconds_mean >= 0
    assert telemetry.request_rows == 3


def test_explicit_delay_bypass_dispatches_immediately_and_collects_queued_siblings():
    template, first = _template_binding("waiting")
    _, urgent = _template_binding("urgent", request_slot=1, output_row=2)
    executor = _RecordingExecutor()

    with ComponentReactor(
        executor,
        max_batch_size=4,
        max_pending=4,
        max_batch_delay_seconds=60,
    ) as reactor:
        waiting = reactor.submit(template, first, _Payload("waiting"))
        bypassed = reactor.submit(
            template,
            urgent,
            _Payload("urgent"),
            bypass_batch_delay=True,
        )
        assert waiting.result(timeout=1) == "waiting"
        assert bypassed.result(timeout=1) == "urgent"
        telemetry = reactor.telemetry()

    assert len(executor.batches) == 1
    assert executor.batches[0].bypass_batch_delays == (False, True)
    assert telemetry.bypass_submitted == 1
    assert telemetry.bypass_dispatched == 1
    assert telemetry.bypass_batches == 1
    assert telemetry.bypass_singleton_batches == 0
    assert telemetry.recent_batches[-1].bypass_requested == 1
    assert telemetry.recent_batches[-1].batch_delay_bypassed
    assert len(telemetry.recent_batches[-1].queue_wait_seconds) == 2


def test_explicit_flush_drains_only_the_current_queue_without_counting_as_bypass():
    first_template, first = _template_binding("flush-first")
    second_template, second = _template_binding(
        "flush-second",
        numerical_contract="flush-other",
    )
    executor = _RecordingExecutor()

    with ComponentReactor(
        executor,
        max_batch_size=4,
        max_pending=4,
        max_batch_delay_seconds=60,
    ) as reactor:
        first_future = reactor.submit(first_template, first, _Payload("first"))
        second_future = reactor.submit(second_template, second, _Payload("second"))
        assert reactor.flush_pending() == 2
        assert first_future.result(timeout=1) == "first"
        assert second_future.result(timeout=1) == "second"
        telemetry = reactor.telemetry()

    assert telemetry.batches == 2
    assert telemetry.bypass_batches == telemetry.bypass_singleton_batches == 0
    assert all(record.dispatch_trigger == "explicit_flush" for record in telemetry.recent_batches)


def test_atomic_cohort_validation_and_admission_are_all_or_none():
    template, first = _template_binding("first")
    _, second = _template_binding("second", request_slot=1, output_row=2)
    _, invalid = _template_binding("invalid", numerical_contract="other")
    executor = _RecordingExecutor()

    with ComponentReactor(
        executor,
        max_batch_size=2,
        max_pending=2,
        max_batch_delay_seconds=60,
    ) as reactor:
        with pytest.raises(ValueError, match="different WorkTemplate"):
            reactor.submit_many(
                (
                    ComponentSubmission(template, first, _Payload("first")),
                    ComponentSubmission(template, invalid, _Payload("invalid")),
                )
            )
        assert reactor.telemetry().submitted == 0

        futures = reactor.submit_many(
            (
                ComponentSubmission(template, first, _Payload("first")),
                ComponentSubmission(
                    template,
                    second,
                    _Payload("second"),
                    bypass_batch_delay=True,
                ),
            )
        )
        assert tuple(future.result(timeout=1) for future in futures) == ("first", "second")
        telemetry = reactor.telemetry()

    assert len(executor.batches) == 1
    assert executor.batches[0].dispatch_width == 2
    assert telemetry.recent_batches[-1].dispatch_trigger == "batch_full"
    assert not telemetry.recent_batches[-1].batch_delay_bypassed
    assert telemetry.bypass_dispatched == 1
    assert telemetry.bypass_batches == 0


def test_aged_work_dispatches_before_a_new_incompatible_bypass_request():
    class ManualClock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

    clock = ManualClock()
    aged_template, aged_binding = _template_binding("aged")
    urgent_template, urgent_binding = _template_binding(
        "urgent-other",
        numerical_contract="urgent-other",
    )
    executor = _RecordingExecutor()

    with ComponentReactor(
        executor,
        max_batch_size=4,
        max_pending=4,
        max_batch_delay_seconds=10,
        clock=clock,
    ) as reactor:
        aged = reactor.submit(aged_template, aged_binding, _Payload("aged"))
        clock.value = 11.0
        urgent = reactor.submit(
            urgent_template,
            urgent_binding,
            _Payload("urgent"),
            bypass_batch_delay=True,
        )
        assert aged.result(timeout=1) == "aged"
        assert urgent.result(timeout=1) == "urgent"
        telemetry = reactor.telemetry()

    assert [batch.payloads[0].value for batch in executor.batches] == ["aged", "urgent"]
    assert [record.dispatch_trigger for record in telemetry.recent_batches] == [
        "delay_elapsed",
        "batch_delay_bypass",
    ]


def test_delay_bypass_flag_is_strictly_boolean_before_admission():
    template, binding = _template_binding("invalid-bypass")
    executor = _RecordingExecutor()

    with ComponentReactor(executor, max_batch_size=1, max_pending=1) as reactor:
        with pytest.raises(TypeError, match="bypass_batch_delay"):
            reactor.submit(
                template,
                binding,
                _Payload("invalid"),
                bypass_batch_delay=1,  # type: ignore[arg-type]
            )
        assert reactor.telemetry().submitted == 0


def test_source_and_lowered_templates_are_separate_and_lowered_identity_is_complete():
    template, binding = _template_binding("source")
    lowered = lower_work_template(template, "paged")
    changed_schedule = lower_work_template(template, "cuda")
    executor = _RecordingExecutor()

    assert template_compatibility_key(template) != template_compatibility_key(template, lowered)
    assert template_compatibility_key(template, lowered) != template_compatibility_key(
        template, changed_schedule
    )

    with ComponentReactor(
        executor,
        max_batch_size=4,
        max_pending=4,
        max_batch_delay_seconds=0.01,
    ) as reactor:
        futures = (
            reactor.submit(template, binding, _Payload("source")),
            reactor.submit(
                template,
                binding,
                _Payload("source"),
                lowered_template=lowered,
            ),
            reactor.submit(
                template,
                binding,
                _Payload("source"),
                lowered_template=changed_schedule,
            ),
        )
        assert [future.result(timeout=1) for future in futures] == ["source"] * 3

    assert [batch.dispatch_width for batch in executor.batches] == [1, 1, 1]
    assert len({batch.compatibility_key for batch in executor.batches}) == 3


def test_executor_preserves_per_item_success_and_exception():
    template, first = _template_binding("first")
    _, second = _template_binding("second", request_slot=1, output_row=2)
    marker = LookupError("one row failed")

    class MixedExecutor:
        def execute_batch(self, batch: ComponentBatch[_Payload]):
            assert batch.dispatch_width == 2
            assert batch.payloads == (_Payload("first"), _Payload("second"))
            return (BatchItemResult.success(41), BatchItemResult.failure(marker))

    with ComponentReactor(
        MixedExecutor(),
        max_batch_size=2,
        max_pending=2,
        max_batch_delay_seconds=1,
    ) as reactor:
        successful = reactor.submit(template, first, _Payload("first"))
        failed = reactor.submit(template, second, _Payload("second"))
        assert successful.result(timeout=1) == 41
        with pytest.raises(LookupError, match="one row failed"):
            failed.result(timeout=1)
        assert failed.exception() is marker
        telemetry = reactor.telemetry()

    assert telemetry.succeeded == 1
    assert telemetry.failed == 1
    assert telemetry.deadline_expired == 0
    assert telemetry.recent_batches[-1].succeeded == 1
    assert telemetry.recent_batches[-1].failed == 1


@pytest.mark.parametrize("returned_count", [0, 1, 3])
def test_malformed_executor_cardinality_fails_the_whole_batch(returned_count: int):
    template, first = _template_binding("first")
    _, second = _template_binding("second", request_slot=1, output_row=2)

    class MalformedExecutor:
        def execute_batch(self, batch: ComponentBatch[_Payload]):
            return tuple(BatchItemResult.success(index) for index in range(returned_count))

    with ComponentReactor(
        MalformedExecutor(),
        max_batch_size=2,
        max_pending=2,
        max_batch_delay_seconds=1,
    ) as reactor:
        futures = (
            reactor.submit(template, first, _Payload("first")),
            reactor.submit(template, second, _Payload("second")),
        )
        for future in futures:
            with pytest.raises(ReactorExecutorProtocolError, match="outcomes"):
                future.result(timeout=1)


def test_binding_validation_happens_before_admission():
    template, _ = _template_binding("first")
    _, wrong_binding = _template_binding("wrong", numerical_contract="other")
    executor = _RecordingExecutor()

    with ComponentReactor(executor, max_batch_size=1, max_pending=1) as reactor:
        with pytest.raises(ValueError, match="different WorkTemplate"):
            reactor.submit(template, wrong_binding, _Payload("wrong"))
        assert reactor.telemetry().submitted == 0


def test_lowered_schedule_cannot_bypass_source_binding_cardinality_validation():
    template, binding = _template_binding("forged")
    lowered = lower_work_template(template, "paged")
    forged = replace(binding, required_output_rows=())
    executor = _RecordingExecutor()

    with ComponentReactor(executor, max_batch_size=1, max_pending=1) as reactor:
        with pytest.raises(ValueError, match="cardinality"):
            reactor.submit(
                template,
                forged,
                _Payload("forged"),
                lowered_template=lowered,
            )
        with pytest.raises(TypeError, match="source WorkTemplate"):
            reactor.submit(lowered, binding, _Payload("lowered-only"))  # type: ignore[arg-type]
        assert reactor.telemetry().submitted == 0


class _FirstBatchBlockingExecutor:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.payloads: list[_Payload] = []

    def execute_batch(self, batch: ComponentBatch[_Payload]):
        self.calls += 1
        self.payloads.extend(batch.payloads)
        if self.calls == 1:
            self.started.set()
            assert self.release.wait(2)
        return tuple(BatchItemResult.success(payload.value) for payload in batch.payloads)


def test_backpressure_counts_executing_and_queued_and_cancellation_releases_capacity():
    template, first = _template_binding("first")
    _, second = _template_binding("second", request_slot=1, output_row=2)
    _, third = _template_binding("third", request_slot=2, output_row=3)
    executor = _FirstBatchBlockingExecutor()

    reactor = ComponentReactor(
        executor,
        max_batch_size=1,
        max_pending=2,
        max_batch_delay_seconds=0,
    )
    try:
        first_future = reactor.submit(template, first, _Payload("first"))
        assert executor.started.wait(1)
        second_future = reactor.submit(template, second, _Payload("second"))
        with pytest.raises(ReactorBackpressureError, match="limit 2"):
            reactor.submit(template, third, _Payload("rejected"))

        assert second_future.cancel()
        third_future = reactor.submit(template, third, _Payload("third"))
        executor.release.set()
        assert first_future.result(timeout=1) == "first"
        assert third_future.result(timeout=1) == "third"
        assert reactor.wait_idle(timeout=1)
        telemetry = reactor.telemetry()
    finally:
        executor.release.set()
        reactor.shutdown(wait=True, cancel_pending=True)

    assert telemetry.rejected_backpressure == 1
    assert telemetry.cancelled == 1
    assert telemetry.outstanding == 0
    assert executor.payloads == [_Payload("first"), _Payload("third")]


def test_admission_and_completion_deadlines_never_publish_late_success():
    template, binding = _template_binding("deadline")

    class ManualClock:
        value = 10.0

        def __call__(self) -> float:
            return self.value

    clock = ManualClock()

    class AdvancingExecutor:
        def execute_batch(self, batch: ComponentBatch[_Payload]):
            clock.value = 20.0
            return (BatchItemResult.success("too late"),)

    with ComponentReactor(
        AdvancingExecutor(),
        max_batch_size=1,
        max_pending=2,
        max_batch_delay_seconds=0,
        clock=clock,
    ) as reactor:
        already_expired = reactor.submit(template, binding, _Payload("expired"), deadline=9.0)
        late = reactor.submit(template, binding, _Payload("late"), deadline=15.0)

        with pytest.raises(ReactorDeadlineExceeded, match="deadline"):
            already_expired.result(timeout=1)
        with pytest.raises(ReactorDeadlineExceeded, match="deadline"):
            late.result(timeout=1)
        assert reactor.wait_idle(timeout=1)
        telemetry = reactor.telemetry()

    assert telemetry.deadline_expired == 2
    assert telemetry.succeeded == 0
    assert telemetry.dispatched == 1


def test_shutdown_cancels_queued_work_drains_running_work_and_closes_admission():
    template, first = _template_binding("running")
    _, second = _template_binding("queued", request_slot=1, output_row=2)
    _, rejected = _template_binding("rejected", request_slot=2, output_row=3)
    executor = _FirstBatchBlockingExecutor()
    reactor = ComponentReactor(
        executor,
        max_batch_size=1,
        max_pending=3,
        max_batch_delay_seconds=0,
    )

    running = reactor.submit(template, first, _Payload("running"))
    assert executor.started.wait(1)
    queued = reactor.submit(template, second, _Payload("queued"))
    assert not reactor.shutdown(wait=False, cancel_pending=True)
    assert queued.cancelled()
    with pytest.raises(ReactorClosedError):
        reactor.submit(template, rejected, _Payload("rejected"))

    executor.release.set()
    assert running.result(timeout=1) == "running"
    assert reactor.shutdown(wait=True, timeout=1)
    telemetry = reactor.telemetry()

    assert telemetry.stopped
    assert not telemetry.accepting
    assert telemetry.cancelled == 1
    assert telemetry.rejected_closed == 1
    assert telemetry.outstanding == 0
    assert executor.payloads == [_Payload("running")]


def test_executor_exception_is_delivered_to_every_binding_and_shutdown_is_idempotent():
    template, first = _template_binding("first")
    _, second = _template_binding("second", request_slot=1, output_row=2)
    marker = OSError("backend failed")

    class RaisingExecutor:
        def execute_batch(self, batch: ComponentBatch[_Payload]):
            raise marker

    reactor = ComponentReactor(
        RaisingExecutor(),
        max_batch_size=2,
        max_pending=2,
        max_batch_delay_seconds=1,
    )
    futures = (
        reactor.submit(template, first, _Payload("first")),
        reactor.submit(template, second, _Payload("second")),
    )
    for future in futures:
        with pytest.raises(OSError, match="backend failed"):
            future.result(timeout=1)
        assert future.exception() is marker
    assert reactor.shutdown(wait=True)
    assert reactor.shutdown(wait=True)


def test_batch_item_result_requires_one_unambiguous_outcome():
    with pytest.raises(ValueError, match="exactly one"):
        BatchItemResult()
    with pytest.raises(ValueError, match="exactly one"):
        BatchItemResult(_value=1, exception=ValueError("both"))
    assert BatchItemResult.success(None).value is None
