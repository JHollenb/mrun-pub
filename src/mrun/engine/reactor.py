"""Local component-stationary batching reactor.

The reactor is deliberately independent of any engine implementation.  It owns admission,
queueing, compatibility, and result routing; an injected :class:`ComponentBatchExecutor`
owns tensor assembly and execution.  This boundary lets a resident, streamed, or distributed
backend consume the same scheduling contract without putting concurrent callers inside a
stateful engine.

The invariants are intentionally narrow:

* one admitted dispatch owns exactly one ``Future`` and exactly one terminal outcome;
* every binding is validated against its source ``WorkTemplate`` before admission;
* an optional lowered schedule must name that exact source template, and only canonically
  identical source/lowered pairs share an executor call;
* every binding has one opaque payload at the same executor-batch index, so execution never
  depends on an external request-ID registry;
* executor result index ``i`` belongs to binding index ``i``; a malformed result vector fails
  the whole batch instead of guessing;
* cancellation succeeds only before dispatch, as specified by ``concurrent.futures.Future``;
* no successful value is published after its monotonic deadline;
* the admission bound covers queued *and* executing dispatches; and
* shutdown stops admission immediately and either drains or cancels work that has not begun.

This is a dispatch batching layer, not a tensor concatenation policy.  A binding may already
represent more than one model row, so telemetry records both dispatch width and request-row
width.  The executor decides how compatible bindings become a physical ``B x K`` wave.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar, cast

from mrun.compiler.ir import DispatchBinding, WorkTemplate
from mrun.compiler.lowering import LoweredWorkTemplate

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")

_DispatchTrigger = Literal[
    "shutdown",
    "delay_disabled",
    "delay_elapsed",
    "deadline_guard",
    "batch_full",
    "batch_delay_bypass",
    "explicit_flush",
]


class ComponentReactorError(RuntimeError):
    """Base class for reactor-owned failures."""


class ReactorClosedError(ComponentReactorError):
    """Raised when a caller submits after shutdown began."""


class ReactorBackpressureError(ComponentReactorError):
    """Raised when the bounded outstanding-dispatch budget is exhausted."""


class ReactorDeadlineExceeded(TimeoutError, ComponentReactorError):
    """A dispatch expired before a successful result could be published."""

    def __init__(self, request_ids: tuple[str, ...]) -> None:
        rendered = ", ".join(request_ids) if request_ids else "<unidentified>"
        super().__init__(f"component dispatch deadline exceeded: {rendered}")
        self.request_ids = request_ids


class ReactorExecutorProtocolError(ComponentReactorError):
    """The injected executor returned an ambiguous or malformed result vector."""


class ReactorWorkerFailed(ComponentReactorError):
    """The reactor loop itself failed and could no longer safely route work."""


_MISSING = object()


@dataclass(frozen=True, slots=True)
class BatchItemResult(Generic[ResultT]):
    """One explicit executor outcome.

    A wrapper is used instead of treating ``BaseException`` values specially: model callers
    are then free to return arbitrary Python values, and a successful ``None`` remains
    distinguishable from a missing result.
    """

    _value: object = field(default=_MISSING, repr=False)
    exception: BaseException | None = None

    def __post_init__(self) -> None:
        has_value = self._value is not _MISSING
        if has_value == (self.exception is not None):
            raise ValueError("batch item must contain exactly one of value or exception")
        if self.exception is not None and not isinstance(self.exception, BaseException):
            raise TypeError("batch item exception must derive from BaseException")

    @classmethod
    def success(cls, value: ResultT) -> BatchItemResult[ResultT]:
        return cls(_value=value)

    @classmethod
    def failure(cls, exception: BaseException) -> BatchItemResult[ResultT]:
        if not isinstance(exception, BaseException):
            raise TypeError("batch item exception must derive from BaseException")
        return cls(exception=exception)

    @property
    def succeeded(self) -> bool:
        return self.exception is None

    @property
    def value(self) -> ResultT:
        if self.exception is not None:
            raise RuntimeError("failed batch item has no value") from self.exception
        return cast(ResultT, self._value)


@dataclass(frozen=True, slots=True)
class TemplateCompatibilityKey:
    """Canonical, type-separated identity used for queue coalescing."""

    kind: str
    template_fingerprint: str
    canonical_digest: str


def template_compatibility_key(
    template: WorkTemplate,
    lowered_template: LoweredWorkTemplate | None = None,
) -> TemplateCompatibilityKey:
    """Return the strict reactor identity for a source/lowered pair.

    The source template is the binding-validation authority.  A paired lowered template
    additionally contains backend and schedule verdicts, so its complete canonical document
    is hashed.  A lowered template can never be admitted alone: it does not carry the binding
    cardinalities required by :meth:`DispatchBinding.validate_for`.
    """

    if not isinstance(template, WorkTemplate):
        raise TypeError("template must be a source WorkTemplate")
    if lowered_template is None:
        return TemplateCompatibilityKey(
            kind="work-template",
            template_fingerprint=template.fingerprint,
            canonical_digest=template.fingerprint,
        )
    if not isinstance(lowered_template, LoweredWorkTemplate):
        raise TypeError("lowered_template must be a LoweredWorkTemplate or None")
    if lowered_template.template_fingerprint != template.fingerprint:
        raise ValueError("lowered WorkTemplate belongs to a different source WorkTemplate")
    digest = hashlib.sha256(lowered_template.to_json().encode("utf-8")).hexdigest()
    return TemplateCompatibilityKey(
        kind="lowered-work-template",
        template_fingerprint=template.fingerprint,
        canonical_digest=digest,
    )


@dataclass(frozen=True, slots=True)
class ComponentBatch(Generic[PayloadT]):
    """Ordered batch handed to the injected executor."""

    batch_id: int
    compatibility_key: TemplateCompatibilityKey
    template: WorkTemplate
    lowered_template: LoweredWorkTemplate | None
    bindings: tuple[DispatchBinding, ...]
    payloads: tuple[PayloadT, ...]
    dispatch_ids: tuple[int, ...]
    deadlines: tuple[float | None, ...]
    bypass_batch_delays: tuple[bool, ...]

    def __post_init__(self) -> None:
        expected_key = template_compatibility_key(self.template, self.lowered_template)
        if self.compatibility_key != expected_key:
            raise ValueError("component batch compatibility key does not match its templates")
        width = len(self.bindings)
        if len(self.payloads) != width:
            raise ValueError("component batch payloads must align with bindings")
        if (
            len(self.dispatch_ids) != width
            or len(self.deadlines) != width
            or len(self.bypass_batch_delays) != width
        ):
            raise ValueError("component batch metadata must align with bindings")
        if any(not isinstance(value, bool) for value in self.bypass_batch_delays):
            raise TypeError("component batch delay-bypass metadata must be boolean")
        for binding in self.bindings:
            binding.validate_for(self.template)

    @property
    def dispatch_width(self) -> int:
        return len(self.bindings)

    @property
    def request_row_width(self) -> int:
        return sum(len(binding.request_ids) for binding in self.bindings)


@dataclass(frozen=True, slots=True)
class ComponentSubmission(Generic[PayloadT]):
    """One immutable dispatch description for scalar or atomic-cohort admission."""

    template: WorkTemplate
    binding: DispatchBinding
    payload: PayloadT
    lowered_template: LoweredWorkTemplate | None = None
    deadline: float | None = None
    bypass_batch_delay: bool = False


class ComponentBatchExecutor(Protocol[PayloadT, ResultT]):
    """Backend boundary for one strictly compatible execution wave."""

    def execute_batch(self, batch: ComponentBatch[PayloadT]) -> Sequence[BatchItemResult[ResultT]]:
        """Return exactly one explicit outcome for each binding, in binding order."""


@dataclass(frozen=True, slots=True)
class BatchTelemetry:
    batch_id: int
    compatibility_key: TemplateCompatibilityKey
    dispatch_width: int
    request_row_width: int
    coalesced_dispatches: int
    bypass_requested: int
    dispatch_trigger: _DispatchTrigger
    batch_delay_bypassed: bool
    queue_wait_seconds: tuple[float, ...]
    queue_wait_seconds_min: float
    queue_wait_seconds_mean: float
    queue_wait_seconds_max: float
    execution_seconds: float
    succeeded: int
    failed: int
    deadline_expired: int


@dataclass(frozen=True, slots=True)
class ReactorTelemetry:
    submitted: int
    rejected_backpressure: int
    rejected_closed: int
    succeeded: int
    failed: int
    cancelled: int
    deadline_expired: int
    batches: int
    dispatched: int
    request_rows: int
    batch_width_total: int
    batch_width_max: int
    batch_width_histogram: tuple[tuple[int, int], ...]
    queue_wait_samples: int
    queue_wait_seconds_total: float
    queue_wait_seconds_max: float
    unique_templates: int
    template_reuses: int
    coalesced_dispatches: int
    bypass_submitted: int
    bypass_dispatched: int
    bypass_batches: int
    bypass_singleton_batches: int
    outstanding: int
    queued: int
    executing: int
    accepting: bool
    stopped: bool
    worker_failure: str | None
    recent_batches: tuple[BatchTelemetry, ...]

    @property
    def batch_width_mean(self) -> float:
        return self.batch_width_total / self.batches if self.batches else 0.0

    @property
    def queue_wait_seconds_mean(self) -> float:
        if not self.queue_wait_samples:
            return 0.0
        return self.queue_wait_seconds_total / self.queue_wait_samples

    @property
    def template_reuse_rate(self) -> float:
        return self.template_reuses / self.submitted if self.submitted else 0.0


@dataclass(slots=True)
class _QueuedDispatch(Generic[PayloadT, ResultT]):
    dispatch_id: int
    key: TemplateCompatibilityKey
    template: WorkTemplate
    lowered_template: LoweredWorkTemplate | None
    binding: DispatchBinding
    payload: PayloadT
    future: Future[ResultT]
    enqueued_at: float
    deadline: float | None
    bypass_batch_delay: bool


class ComponentReactor(Generic[PayloadT, ResultT]):
    """A bounded, single-owner local batching reactor.

    ``deadline`` values passed to :meth:`submit` are absolute values in the reactor clock's
    domain (``time.monotonic()`` by default).  ``max_pending`` counts every admitted future
    that is not terminal, including work already inside the executor.
    """

    def __init__(
        self,
        executor: ComponentBatchExecutor[PayloadT, ResultT],
        *,
        max_batch_size: int = 32,
        max_pending: int = 1024,
        max_batch_delay_seconds: float = 0.001,
        telemetry_history: int = 128,
        clock: Callable[[], float] = time.monotonic,
        thread_name: str = "mrun-component-reactor",
    ) -> None:
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        if max_batch_size > max_pending:
            raise ValueError("max_batch_size cannot exceed max_pending")
        if (
            isinstance(max_batch_delay_seconds, bool)
            or not isinstance(max_batch_delay_seconds, (int, float))
            or not math.isfinite(max_batch_delay_seconds)
            or max_batch_delay_seconds < 0
        ):
            raise ValueError("max_batch_delay_seconds must be finite and non-negative")
        if (
            isinstance(telemetry_history, bool)
            or not isinstance(telemetry_history, int)
            or telemetry_history < 0
        ):
            raise ValueError("telemetry_history must be a non-negative integer")
        execute_batch = getattr(executor, "execute_batch", None)
        if not callable(execute_batch):
            raise TypeError("executor must implement execute_batch(batch)")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._executor = executor
        self._max_batch_size = int(max_batch_size)
        self._max_pending = int(max_pending)
        self._max_batch_delay = float(max_batch_delay_seconds)
        self._telemetry_history = int(telemetry_history)
        self._clock = clock

        self._condition = threading.Condition(threading.RLock())
        self._queue: deque[_QueuedDispatch[PayloadT, ResultT]] = deque()
        self._active: dict[int, _QueuedDispatch[PayloadT, ResultT]] = {}
        self._stopped = threading.Event()
        self._accepting = True
        self._shutdown_requested = False
        self._flush_watermark: int | None = None
        self._worker_failure: str | None = None
        self._next_dispatch_id = 1
        self._next_batch_id = 1
        self._outstanding = 0
        self._executing = 0

        self._submitted = 0
        self._rejected_backpressure = 0
        self._rejected_closed = 0
        self._succeeded = 0
        self._failed = 0
        self._cancelled = 0
        self._deadline_expired = 0
        self._batches = 0
        self._dispatched = 0
        self._request_rows = 0
        self._batch_width_total = 0
        self._batch_width_max = 0
        self._batch_width_histogram: Counter[int] = Counter()
        self._queue_wait_samples = 0
        self._queue_wait_seconds_total = 0.0
        self._queue_wait_seconds_max = 0.0
        self._seen_templates: set[TemplateCompatibilityKey] = set()
        self._template_reuses = 0
        self._coalesced_dispatches = 0
        self._bypass_submitted = 0
        self._bypass_dispatched = 0
        self._bypass_batches = 0
        self._bypass_singleton_batches = 0
        self._recent_batches: deque[BatchTelemetry] = deque(maxlen=self._telemetry_history or None)

        self._worker = threading.Thread(target=self._worker_main, name=thread_name, daemon=True)
        self._worker.start()

    def submit(
        self,
        template: WorkTemplate,
        binding: DispatchBinding,
        payload: PayloadT,
        *,
        lowered_template: LoweredWorkTemplate | None = None,
        deadline: float | None = None,
        bypass_batch_delay: bool = False,
    ) -> Future[ResultT]:
        """Admit one binding/payload pair or reject it under backpressure.

        Invalid template/binding pairs are rejected before they consume admission budget.
        ``payload`` is retained opaquely and never copied or inspected; callers must treat it
        as immutable for the lifetime of the returned future.  A paired lowered schedule is
        accepted only after its source fingerprint matches ``template``.
        An already-expired deadline is represented as a terminal future, preserving the
        one-submission/one-future contract used for all admitted work.
        """

        return self.submit_many(
            (
                ComponentSubmission(
                    template=template,
                    binding=binding,
                    payload=payload,
                    lowered_template=lowered_template,
                    deadline=deadline,
                    bypass_batch_delay=bypass_batch_delay,
                ),
            )
        )[0]

    def submit_many(
        self,
        submissions: Sequence[ComponentSubmission[PayloadT]],
    ) -> tuple[Future[ResultT], ...]:
        """Validate and admit one ordered cohort atomically.

        Either every member consumes admission budget and receives a future or none do.  The
        reactor still groups only strict compatibility matches; a cohort may therefore contain
        several eventual physical batches.  Marking its final compatible member for delay bypass
        is a safe way for a coordinator to flush an already assembled wave without racing the
        worker between scalar submissions.
        """

        cohort = tuple(submissions)
        if not cohort:
            raise ValueError("component submission cohort cannot be empty")
        prepared: list[
            tuple[
                ComponentSubmission[PayloadT],
                TemplateCompatibilityKey,
                float | None,
                Future[ResultT],
            ]
        ] = []
        for submission in cohort:
            if not isinstance(submission, ComponentSubmission):
                raise TypeError("submissions must contain ComponentSubmission values")
            key = template_compatibility_key(
                submission.template,
                submission.lowered_template,
            )
            self._validate_binding(submission.template, submission.binding)
            if submission.lowered_template is not None:
                submission.lowered_template.bind(submission.template.bind(submission.binding))
            normalized_deadline = self._validate_deadline(submission.deadline)
            if not isinstance(submission.bypass_batch_delay, bool):
                raise TypeError("bypass_batch_delay must be a boolean")
            prepared.append((submission, key, normalized_deadline, Future()))

        now = self._clock()
        expired: list[tuple[_QueuedDispatch[PayloadT, ResultT], Future[ResultT]]] = []
        with self._condition:
            width = len(prepared)
            if not self._accepting:
                self._rejected_closed += width
                raise ReactorClosedError("component reactor is shutting down")
            if self._outstanding + width > self._max_pending:
                self._rejected_backpressure += width
                raise ReactorBackpressureError(
                    f"component reactor would have {self._outstanding + width} outstanding "
                    f"dispatches (limit {self._max_pending})"
                )

            futures: list[Future[ResultT]] = []
            for submission, key, normalized_deadline, future in prepared:
                dispatch_id = self._next_dispatch_id
                self._next_dispatch_id += 1
                entry = _QueuedDispatch(
                    dispatch_id=dispatch_id,
                    key=key,
                    template=submission.template,
                    lowered_template=submission.lowered_template,
                    binding=submission.binding,
                    payload=submission.payload,
                    future=future,
                    enqueued_at=now,
                    deadline=normalized_deadline,
                    bypass_batch_delay=submission.bypass_batch_delay,
                )
                self._submitted += 1
                self._bypass_submitted += int(submission.bypass_batch_delay)
                self._outstanding += 1
                if key in self._seen_templates:
                    self._template_reuses += 1
                else:
                    self._seen_templates.add(key)
                future.add_done_callback(self._future_done)
                futures.append(future)

                if normalized_deadline is not None and normalized_deadline <= now:
                    if not future.set_running_or_notify_cancel():
                        raise AssertionError("newly admitted future was unexpectedly cancelled")
                    self._deadline_expired += 1
                    expired.append((entry, future))
                else:
                    self._queue.append(entry)
            self._condition.notify_all()

        for entry, future in expired:
            future.set_exception(ReactorDeadlineExceeded(entry.binding.request_ids))
        return tuple(futures)

    def telemetry(self) -> ReactorTelemetry:
        """Return an immutable, internally consistent point-in-time snapshot."""

        with self._condition:
            queued = sum(not entry.future.done() for entry in self._queue)
            return ReactorTelemetry(
                submitted=self._submitted,
                rejected_backpressure=self._rejected_backpressure,
                rejected_closed=self._rejected_closed,
                succeeded=self._succeeded,
                failed=self._failed,
                cancelled=self._cancelled,
                deadline_expired=self._deadline_expired,
                batches=self._batches,
                dispatched=self._dispatched,
                request_rows=self._request_rows,
                batch_width_total=self._batch_width_total,
                batch_width_max=self._batch_width_max,
                batch_width_histogram=tuple(sorted(self._batch_width_histogram.items())),
                queue_wait_samples=self._queue_wait_samples,
                queue_wait_seconds_total=self._queue_wait_seconds_total,
                queue_wait_seconds_max=self._queue_wait_seconds_max,
                unique_templates=len(self._seen_templates),
                template_reuses=self._template_reuses,
                coalesced_dispatches=self._coalesced_dispatches,
                bypass_submitted=self._bypass_submitted,
                bypass_dispatched=self._bypass_dispatched,
                bypass_batches=self._bypass_batches,
                bypass_singleton_batches=self._bypass_singleton_batches,
                outstanding=self._outstanding,
                queued=queued,
                executing=self._executing,
                accepting=self._accepting,
                stopped=self._stopped.is_set(),
                worker_failure=self._worker_failure,
                recent_batches=tuple(self._recent_batches),
            )

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until every admitted future is terminal."""

        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be finite and non-negative")
        end = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._outstanding:
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def flush_pending(self) -> int:
        """Wake all dispatches that are queued at this call's linearization point.

        The reactor remains open for admission.  A dispatch admitted later is not itself
        flush-marked, although it may naturally coalesce with an already marked compatible
        sibling.  The return value is the number of live queued entries covered by this call.
        """

        with self._condition:
            queued = tuple(entry for entry in self._queue if not entry.future.done())
            if not queued:
                return 0
            watermark = max(entry.dispatch_id for entry in queued)
            self._flush_watermark = max(self._flush_watermark or 0, watermark)
            self._condition.notify_all()
            return len(queued)

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_pending: bool = False,
        timeout: float | None = None,
    ) -> bool:
        """Stop admission, then drain or cancel dispatches that have not started.

        Running executor calls cannot be interrupted safely.  ``cancel_pending=True`` uses
        ordinary ``Future.cancel`` for queued requests and lets the current batch finish.
        The returned boolean says whether the worker has stopped.
        """

        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be finite and non-negative")
        with self._condition:
            self._accepting = False
            self._shutdown_requested = True
            to_cancel = tuple(self._queue) if cancel_pending else ()
            if cancel_pending:
                self._queue.clear()
            self._condition.notify_all()

        for entry in to_cancel:
            entry.future.cancel()

        if wait and threading.current_thread() is not self._worker:
            self._worker.join(timeout)
        return self._stopped.is_set()

    def close(self) -> None:
        self.shutdown(wait=True)

    def __enter__(self) -> ComponentReactor[PayloadT, ResultT]:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.shutdown(wait=True)

    @staticmethod
    def _validate_binding(template: WorkTemplate, binding: DispatchBinding) -> None:
        if not isinstance(template, WorkTemplate):
            raise TypeError("template must be a source WorkTemplate")
        if not isinstance(binding, DispatchBinding):
            raise TypeError("binding must be a DispatchBinding")
        binding.validate_for(template)

    @staticmethod
    def _validate_deadline(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise TypeError("deadline must be a finite monotonic timestamp")
        normalized = float(deadline)
        if not math.isfinite(normalized):
            raise ValueError("deadline must be a finite monotonic timestamp")
        return normalized

    def _future_done(self, future: Future[ResultT]) -> None:
        with self._condition:
            self._outstanding -= 1
            if self._outstanding < 0:
                raise AssertionError("reactor outstanding count became negative")
            if future.cancelled():
                self._cancelled += 1
            self._condition.notify_all()

    def _worker_main(self) -> None:
        try:
            self._worker_loop()
        except BaseException as exc:  # keep all admitted futures terminal on reactor bugs
            self._abort_worker(exc)
        finally:
            with self._condition:
                self._accepting = False
                self._stopped.set()
                self._condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                prepared = self._wait_for_batch_locked()
                if prepared is None:
                    return
                batch, entries, queue_waits, dispatch_trigger = prepared

            started = self._clock()
            outcomes = self._execute(batch)
            finished = self._clock()
            execution_seconds = max(0.0, finished - started)
            published = self._apply_completion_deadlines(entries, outcomes, finished)
            self._record_finished_batch(
                batch,
                entries,
                queue_waits,
                execution_seconds,
                published,
                dispatch_trigger,
            )

            for entry, outcome in zip(entries, published, strict=True):
                if outcome.succeeded:
                    entry.future.set_result(outcome.value)
                else:
                    assert outcome.exception is not None
                    entry.future.set_exception(outcome.exception)

    def _wait_for_batch_locked(
        self,
    ) -> (
        tuple[
            ComponentBatch[PayloadT],
            tuple[_QueuedDispatch[PayloadT, ResultT], ...],
            tuple[float, ...],
            _DispatchTrigger,
        ]
        | None
    ):
        while True:
            now = self._clock()
            expired = self._take_expired_locked(now)
            if expired:
                # These futures are already claimed RUNNING, preventing a cancellation race.
                self._deadline_expired += len(expired)
                self._condition.release()
                try:
                    for entry in expired:
                        entry.future.set_exception(
                            ReactorDeadlineExceeded(entry.binding.request_ids)
                        )
                finally:
                    self._condition.acquire()
                continue

            self._drop_cancelled_locked()
            if self._shutdown_requested and not self._queue:
                return None

            seed, wait_seconds, dispatch_trigger = self._select_seed_locked(now)
            if seed is None:
                self._condition.wait(wait_seconds)
                continue
            assert dispatch_trigger is not None

            delay_bypassed = dispatch_trigger == "batch_delay_bypass"
            selected = self._remove_compatible_locked(
                seed.key,
                required_dispatch_id=seed.dispatch_id if delay_bypassed else None,
            )
            # A cancelled/expired bypass marker cannot cause its compatible siblings to run.
            # Claim the causal seed before every sibling so a successful bypass batch always
            # contains the dispatch that requested it.  Ordinary triggers retain queue-order
            # claiming and cancellation semantics.
            bypass_seed_claimed = False
            if delay_bypassed:
                seed_now = self._clock()
                if seed.deadline is not None and seed.deadline <= seed_now:
                    self._restore_queue_locked(selected)
                    continue
                if not seed.future.set_running_or_notify_cancel():
                    self._restore_queue_locked(entry for entry in selected if entry is not seed)
                    continue
                bypass_seed_claimed = True
            entries: list[_QueuedDispatch[PayloadT, ResultT]] = []
            predispatch_expired: list[_QueuedDispatch[PayloadT, ResultT]] = []
            now = self._clock()
            for entry in selected:
                if entry is seed and bypass_seed_claimed:
                    claimed = True
                else:
                    claimed = entry.future.set_running_or_notify_cancel()
                if not claimed:
                    continue
                if (
                    entry.deadline is not None
                    and entry.deadline <= now
                    and not (delay_bypassed and entry is seed)
                ):
                    predispatch_expired.append(entry)
                else:
                    entries.append(entry)

            if predispatch_expired:
                self._deadline_expired += len(predispatch_expired)
                self._condition.release()
                try:
                    for entry in predispatch_expired:
                        entry.future.set_exception(
                            ReactorDeadlineExceeded(entry.binding.request_ids)
                        )
                finally:
                    self._condition.acquire()
            if not entries:
                continue

            batch_id = self._next_batch_id
            self._next_batch_id += 1
            started = self._clock()
            queue_waits = tuple(max(0.0, started - entry.enqueued_at) for entry in entries)
            entry_tuple = tuple(entries)
            for entry in entry_tuple:
                self._active[entry.dispatch_id] = entry
            self._executing += len(entry_tuple)

            batch = ComponentBatch(
                batch_id=batch_id,
                compatibility_key=seed.key,
                template=entries[0].template,
                lowered_template=entries[0].lowered_template,
                bindings=tuple(entry.binding for entry in entries),
                payloads=tuple(entry.payload for entry in entries),
                dispatch_ids=tuple(entry.dispatch_id for entry in entries),
                deadlines=tuple(entry.deadline for entry in entries),
                bypass_batch_delays=tuple(entry.bypass_batch_delay for entry in entries),
            )
            self._batches += 1
            self._dispatched += len(entries)
            self._request_rows += batch.request_row_width
            self._batch_width_total += len(entries)
            self._batch_width_max = max(self._batch_width_max, len(entries))
            self._batch_width_histogram[len(entries)] += 1
            self._coalesced_dispatches += max(0, len(entries) - 1)
            bypass_count = sum(entry.bypass_batch_delay for entry in entries)
            self._bypass_dispatched += bypass_count
            if delay_bypassed:
                if seed not in entries or not seed.bypass_batch_delay:
                    raise AssertionError("causal batch-delay bypass lost its seed dispatch")
                self._bypass_batches += 1
                self._bypass_singleton_batches += int(len(entries) == 1)
            self._queue_wait_samples += len(queue_waits)
            self._queue_wait_seconds_total += sum(queue_waits)
            if queue_waits:
                self._queue_wait_seconds_max = max(self._queue_wait_seconds_max, max(queue_waits))
            return batch, entry_tuple, queue_waits, dispatch_trigger

    def _take_expired_locked(self, now: float) -> tuple[_QueuedDispatch[PayloadT, ResultT], ...]:
        retained: deque[_QueuedDispatch[PayloadT, ResultT]] = deque()
        expired: list[_QueuedDispatch[PayloadT, ResultT]] = []
        while self._queue:
            entry = self._queue.popleft()
            if entry.future.done():
                continue
            if entry.deadline is not None and entry.deadline <= now:
                if entry.future.set_running_or_notify_cancel():
                    expired.append(entry)
            else:
                retained.append(entry)
        self._queue = retained
        return tuple(expired)

    def _drop_cancelled_locked(self) -> None:
        if any(entry.future.done() for entry in self._queue):
            self._queue = deque(entry for entry in self._queue if not entry.future.done())

    def _select_seed_locked(
        self, now: float
    ) -> tuple[
        _QueuedDispatch[PayloadT, ResultT] | None,
        float | None,
        _DispatchTrigger | None,
    ]:
        if not self._queue:
            return None, None, None
        oldest = self._queue[0]
        if self._shutdown_requested:
            return oldest, 0.0, "shutdown"
        if self._max_batch_delay == 0:
            return oldest, 0.0, "delay_disabled"
        if now >= oldest.enqueued_at + self._max_batch_delay:
            return oldest, 0.0, "delay_elapsed"

        deadline_entries = tuple(entry for entry in self._queue if entry.deadline is not None)
        if deadline_entries:
            earliest = min(deadline_entries, key=lambda entry: cast(float, entry.deadline))
            if cast(float, earliest.deadline) <= now + self._max_batch_delay:
                return earliest, 0.0, "deadline_guard"

        counts: Counter[TemplateCompatibilityKey] = Counter(entry.key for entry in self._queue)
        full_keys = {key for key, count in counts.items() if count >= self._max_batch_size}
        if full_keys:
            return next(entry for entry in self._queue if entry.key in full_keys), 0.0, "batch_full"

        bypass = next((entry for entry in self._queue if entry.bypass_batch_delay), None)
        if bypass is not None:
            return bypass, 0.0, "batch_delay_bypass"

        if self._flush_watermark is not None:
            flush_entry = next(
                (
                    entry
                    for entry in self._queue
                    if entry.dispatch_id <= cast(int, self._flush_watermark)
                ),
                None,
            )
            if flush_entry is not None:
                return flush_entry, 0.0, "explicit_flush"
            self._flush_watermark = None

        wake_at = oldest.enqueued_at + self._max_batch_delay
        if deadline_entries:
            earliest_deadline = min(cast(float, entry.deadline) for entry in deadline_entries)
            wake_at = min(wake_at, earliest_deadline - self._max_batch_delay)
        return None, max(0.0, wake_at - now), None

    def _remove_compatible_locked(
        self,
        key: TemplateCompatibilityKey,
        *,
        required_dispatch_id: int | None = None,
    ) -> tuple[_QueuedDispatch[PayloadT, ResultT], ...]:
        compatible = tuple(entry for entry in self._queue if entry.key == key)
        selected = list(compatible[: self._max_batch_size])
        if required_dispatch_id is not None:
            required = next(
                (entry for entry in compatible if entry.dispatch_id == required_dispatch_id),
                None,
            )
            if required is None:
                raise AssertionError("required compatible dispatch is no longer queued")
            if all(entry.dispatch_id != required_dispatch_id for entry in selected):
                selected = [*compatible[: self._max_batch_size - 1], required]
        selected_ids = {entry.dispatch_id for entry in selected}
        ordered_selected: list[_QueuedDispatch[PayloadT, ResultT]] = []
        retained: deque[_QueuedDispatch[PayloadT, ResultT]] = deque()
        while self._queue:
            entry = self._queue.popleft()
            if entry.dispatch_id in selected_ids:
                ordered_selected.append(entry)
            else:
                retained.append(entry)
        self._queue = retained
        return tuple(ordered_selected)

    def _restore_queue_locked(
        self,
        entries: Sequence[_QueuedDispatch[PayloadT, ResultT]],
    ) -> None:
        """Restore unclaimed entries after a causal bypass seed loses its claim race."""

        restored = tuple(entry for entry in entries if not entry.future.done())
        self._queue = deque(
            sorted(
                (*self._queue, *restored),
                key=lambda entry: entry.dispatch_id,
            )
        )

    def _execute(self, batch: ComponentBatch[PayloadT]) -> tuple[BatchItemResult[ResultT], ...]:
        try:
            raw = tuple(self._executor.execute_batch(batch))
        except BaseException as exc:
            return tuple(BatchItemResult.failure(exc) for _ in batch.bindings)
        if len(raw) != len(batch.bindings):
            error = ReactorExecutorProtocolError(
                f"executor returned {len(raw)} outcomes for {len(batch.bindings)} bindings"
            )
            return tuple(BatchItemResult.failure(error) for _ in batch.bindings)
        if any(not isinstance(outcome, BatchItemResult) for outcome in raw):
            error = ReactorExecutorProtocolError(
                "executor outcomes must all be BatchItemResult values"
            )
            return tuple(BatchItemResult.failure(error) for _ in batch.bindings)
        return cast(tuple[BatchItemResult[ResultT], ...], raw)

    @staticmethod
    def _apply_completion_deadlines(
        entries: tuple[_QueuedDispatch[PayloadT, ResultT], ...],
        outcomes: tuple[BatchItemResult[ResultT], ...],
        finished: float,
    ) -> tuple[BatchItemResult[ResultT], ...]:
        published: list[BatchItemResult[ResultT]] = []
        for entry, outcome in zip(entries, outcomes, strict=True):
            if entry.deadline is not None and entry.deadline <= finished:
                expired = ReactorDeadlineExceeded(entry.binding.request_ids)
                if outcome.exception is not None:
                    expired.__cause__ = outcome.exception
                published.append(BatchItemResult.failure(expired))
            else:
                published.append(outcome)
        return tuple(published)

    def _record_finished_batch(
        self,
        batch: ComponentBatch[PayloadT],
        entries: tuple[_QueuedDispatch[PayloadT, ResultT], ...],
        queue_waits: tuple[float, ...],
        execution_seconds: float,
        outcomes: tuple[BatchItemResult[ResultT], ...],
        dispatch_trigger: _DispatchTrigger,
    ) -> None:
        succeeded = sum(outcome.succeeded for outcome in outcomes)
        deadline_expired = sum(
            isinstance(outcome.exception, ReactorDeadlineExceeded) for outcome in outcomes
        )
        failed = len(outcomes) - succeeded - deadline_expired
        telemetry = BatchTelemetry(
            batch_id=batch.batch_id,
            compatibility_key=batch.compatibility_key,
            dispatch_width=batch.dispatch_width,
            request_row_width=batch.request_row_width,
            coalesced_dispatches=max(0, batch.dispatch_width - 1),
            bypass_requested=sum(batch.bypass_batch_delays),
            dispatch_trigger=dispatch_trigger,
            batch_delay_bypassed=dispatch_trigger == "batch_delay_bypass",
            queue_wait_seconds=queue_waits,
            queue_wait_seconds_min=min(queue_waits),
            queue_wait_seconds_mean=sum(queue_waits) / len(queue_waits),
            queue_wait_seconds_max=max(queue_waits),
            execution_seconds=execution_seconds,
            succeeded=succeeded,
            failed=failed,
            deadline_expired=deadline_expired,
        )
        with self._condition:
            self._succeeded += succeeded
            self._failed += failed
            self._deadline_expired += deadline_expired
            self._executing -= len(entries)
            for entry in entries:
                self._active.pop(entry.dispatch_id, None)
            if self._telemetry_history:
                self._recent_batches.append(telemetry)
            self._condition.notify_all()

    def _abort_worker(self, cause: BaseException) -> None:
        with self._condition:
            self._accepting = False
            self._shutdown_requested = True
            self._worker_failure = f"{type(cause).__name__}: {cause}"
            pending = list(self._active.values())
            self._active.clear()
            self._executing = 0
            while self._queue:
                entry = self._queue.popleft()
                if entry.future.set_running_or_notify_cancel():
                    pending.append(entry)
            self._failed += len(pending)
            self._condition.notify_all()

        for entry in pending:
            failure = ReactorWorkerFailed(
                f"component reactor worker failed before dispatch {entry.dispatch_id} completed"
            )
            failure.__cause__ = cause
            entry.future.set_exception(failure)


__all__ = [
    "BatchItemResult",
    "BatchTelemetry",
    "ComponentBatch",
    "ComponentBatchExecutor",
    "ComponentReactor",
    "ComponentReactorError",
    "ComponentSubmission",
    "ReactorBackpressureError",
    "ReactorClosedError",
    "ReactorDeadlineExceeded",
    "ReactorExecutorProtocolError",
    "ReactorTelemetry",
    "ReactorWorkerFailed",
    "TemplateCompatibilityKey",
    "template_compatibility_key",
]
