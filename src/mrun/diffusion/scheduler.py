"""Continuous compatible batching for program sessions."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any
from uuid import uuid4

from .program import ProgramRuntime, ProgramSession, ProgramStepResult


@dataclass(frozen=True, slots=True)
class BatchDispatch:
    """One physical wave selected by the compatibility scheduler."""

    session_ids: tuple[str, ...]
    compatibility_key: tuple[Any, ...]
    dispatch_id: str = ""
    queue_delays_s: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class _QueueEntry:
    """Admission metadata retained until a row commits or is cancelled."""

    session_id: str
    enqueued_at: float
    deadline_at: float | None = None
    priority: int = 0


class ContinuousBatchScheduler:
    """Queue row-local sessions and drain compatible waves.

    Requests remain independent program sessions.  The scheduler only groups
    rows whose immutable denoise binding matches, and caps each physical wave
    so latency-sensitive requests are not trapped behind an unbounded batch.
    """

    def __init__(
        self,
        runtime: ProgramRuntime,
        *,
        max_batch_size: int = 8,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.runtime = runtime
        self.max_batch_size = int(max_batch_size)
        self._queue: deque[str] = deque()
        self._pending: set[str] = set()
        self._inflight: set[str] = set()
        self._entries: dict[str, _QueueEntry] = {}
        self._clock = clock or time.monotonic
        self._lock = RLock()
        self._events: deque[dict[str, Any]] = deque()
        self._queue_delays_s: list[float] = []
        self._execution_times_s: list[float] = []
        self._submitted = 0
        self._cancelled = 0
        self._expired = 0
        self.physical_waves = 0
        self.physical_rows = 0

    def enqueue(
        self,
        session_id: str,
        *,
        deadline_s: float | None = None,
        deadline_at: float | None = None,
        priority: int = 0,
        now: float | None = None,
    ) -> None:
        """Admit one ready session with optional deadline and priority metadata.

        Admission is fail-closed: the session is validated before any queue
        mutation.  ``deadline_s`` is relative to the scheduler clock; callers
        that already own a monotonic deadline may pass ``deadline_at`` instead.
        The payload and program state remain owned by ``ProgramRuntime``.
        """

        if deadline_s is not None and deadline_at is not None:
            raise ValueError("provide deadline_s or deadline_at, not both")
        if deadline_s is not None and deadline_s <= 0:
            raise ValueError("deadline_s must be positive when supplied")
        if deadline_at is not None and deadline_at <= 0:
            raise ValueError("deadline_at must be positive when supplied")
        if isinstance(priority, bool):
            raise ValueError("priority must be an integer")
        session = self.runtime.get_session(session_id)
        if not isinstance(session, ProgramSession):  # defensive for custom registries
            raise TypeError("runtime returned a non-program session")
        session.binding()  # fail before mutating the queue if context is not ready
        admitted_at = float(self._clock() if now is None else now)
        resolved_deadline = (
            float(deadline_at)
            if deadline_at is not None
            else admitted_at + float(deadline_s)
            if deadline_s is not None
            else None
        )
        with self._lock:
            if session_id in self._pending or session_id in self._inflight:
                raise ValueError(f"session {session_id!r} is already queued")
            self._queue.append(session_id)
            self._pending.add(session_id)
            self._entries[session_id] = _QueueEntry(
                session_id=session_id,
                enqueued_at=admitted_at,
                deadline_at=resolved_deadline,
                priority=int(priority),
            )
            self._submitted += 1
            self._events.append(
                {
                    "event": "admitted",
                    "session_id": session_id,
                    "enqueued_at": admitted_at,
                    "deadline_at": resolved_deadline,
                    "priority": int(priority),
                }
            )

    def refill(
        self,
        session_ids: Iterable[str],
        *,
        deadline_s: float | None = None,
        priority: int = 0,
        now: float | None = None,
    ) -> tuple[str, ...]:
        """Admit a replacement wave after completed/cancelled rows free slots."""

        admitted: list[str] = []
        for session_id in session_ids:
            self.enqueue(
                str(session_id),
                deadline_s=deadline_s,
                priority=priority,
                now=now,
            )
            admitted.append(str(session_id))
        return tuple(admitted)

    def cancel(self, session_id: str, *, reason: str = "client") -> bool:
        """Cancel a queued row without touching its program/session state.

        An in-flight row cannot be cancelled by this queue-level API; the
        physical backend must expose cooperative cancellation for that case.
        Returning ``False`` makes that distinction explicit to a transport.
        """

        with self._lock:
            if session_id not in self._pending or session_id in self._inflight:
                return False
            self._pending.remove(session_id)
            self._entries.pop(session_id, None)
            self._queue = deque(item for item in self._queue if item != session_id)
            self._cancelled += 1
            self._events.append(
                {"event": "cancelled", "session_id": session_id, "reason": str(reason)}
            )
            return True

    def pending(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._queue)

    def _expire(self, now: float) -> tuple[str, ...]:
        expired: list[str] = []
        for session_id in tuple(self._queue):
            entry = self._entries.get(session_id)
            if entry is None or entry.deadline_at is None or entry.deadline_at > now:
                continue
            if session_id in self._inflight:
                continue
            self._pending.discard(session_id)
            self._entries.pop(session_id, None)
            expired.append(session_id)
        if expired:
            expired_set = set(expired)
            self._queue = deque(item for item in self._queue if item not in expired_set)
            self._expired += len(expired)
            self._events.extend(
                {"event": "expired", "session_id": session_id, "at": now}
                for session_id in expired
            )
        return tuple(expired)

    def _batches(self, *, dispatch_now: float) -> list[BatchDispatch]:
        grouped: dict[tuple[Any, ...], list[str]] = {}
        order: list[tuple[Any, ...]] = []
        for session_id in self._queue:
            if session_id in self._inflight or session_id not in self._pending:
                continue
            key = self.runtime.get_session(session_id).binding().compatibility_key
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(session_id)
        batches: list[BatchDispatch] = []
        for key in order:
            rows = sorted(
                grouped[key],
                key=lambda session_id: (
                    -self._entries[session_id].priority,
                    self._entries[session_id].enqueued_at,
                    session_id,
                ),
            )
            for start in range(0, len(rows), self.max_batch_size):
                session_ids = tuple(rows[start : start + self.max_batch_size])
                batches.append(
                    BatchDispatch(
                        session_ids=session_ids,
                        compatibility_key=key,
                        dispatch_id=f"dispatch-{uuid4().hex}",
                        queue_delays_s=tuple(
                            max(0.0, dispatch_now - self._entries[session_id].enqueued_at)
                            for session_id in session_ids
                        ),
                    )
                )
        return batches

    def flush(
        self,
        *,
        max_batches: int | None = None,
        now: float | None = None,
        worker_id: str = "worker-0",
        **step_options: Any,
    ) -> tuple[ProgramStepResult, ...]:
        """Execute queued compatible waves and remove only committed rows.

        ``step_options`` are forwarded unchanged to every physical wave.  This
        keeps transport-level controls (steps, size, guidance, generators, and
        backend-specific knobs) composable with continuous admission instead
        of forcing callers to choose between scheduling and explicit stepping.
        """

        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive when supplied")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        dispatch_now = float(self._clock() if now is None else now)
        with self._lock:
            self._expire(dispatch_now)
            batches = self._batches(dispatch_now=dispatch_now)
            if max_batches is not None:
                batches = batches[:max_batches]
            for batch in batches:
                self._inflight.update(batch.session_ids)
        results: list[ProgramStepResult] = []
        completed_ids: list[str] = []
        try:
            for batch in batches:
                started = time.perf_counter()
                wave_results = self.runtime.step_batch(batch.session_ids, **step_options)
                execution_wall_s = max(0.0, time.perf_counter() - started)
                if len(wave_results) != len(batch.session_ids):
                    raise RuntimeError(
                        "runtime returned a different row count from the scheduled wave"
                    )
                queue_delays = batch.queue_delays_s
                results.extend(
                    replace(
                        result,
                        telemetry={
                            **dict(result.telemetry),
                            "scheduler_dispatch_id": batch.dispatch_id,
                            "scheduler_queue_delay_s": queue_delays[index],
                            "scheduler_batch_size": len(batch.session_ids),
                            "scheduler_execution_wall_s": execution_wall_s,
                            "scheduler_worker_id": worker_id,
                        },
                    )
                    for index, result in enumerate(wave_results)
                )
                completed_ids.extend(batch.session_ids)
                with self._lock:
                    self.physical_waves += 1
                    self.physical_rows += len(batch.session_ids)
                    self._queue_delays_s.extend(queue_delays)
                    self._execution_times_s.append(execution_wall_s)
                    self._events.append(
                        {
                            "event": "dispatched",
                            "dispatch_id": batch.dispatch_id,
                            "worker_id": worker_id,
                            "session_ids": list(batch.session_ids),
                            "batch_size": len(batch.session_ids),
                            "queue_delays_s": list(queue_delays),
                            "execution_wall_s": execution_wall_s,
                        }
                    )
        finally:
            completed = set(completed_ids)
            with self._lock:
                self._inflight.difference_update(
                    session_id
                    for batch in batches
                    for session_id in batch.session_ids
                )
                if completed:
                    self._queue = deque(item for item in self._queue if item not in completed)
                    self._pending.difference_update(completed)
                    for session_id in completed:
                        self._entries.pop(session_id, None)
        return tuple(results)

    def drain_events(self) -> tuple[dict[str, Any], ...]:
        """Return and clear transport-safe queue events since the last read."""

        with self._lock:
            events = tuple(self._events)
            self._events.clear()
            return events

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
        return float(ordered[index])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending": len(self._queue),
                "inflight": len(self._inflight),
                "physical_waves": self.physical_waves,
                "physical_rows": self.physical_rows,
                "submitted": self._submitted,
                "cancelled": self._cancelled,
                "expired": self._expired,
                "queue_delay_samples": len(self._queue_delays_s),
                "queue_delay_p50_s": self._percentile(self._queue_delays_s, 0.50),
                "queue_delay_p95_s": self._percentile(self._queue_delays_s, 0.95),
                "execution_samples": len(self._execution_times_s),
                "execution_p50_s": self._percentile(self._execution_times_s, 0.50),
                "execution_p95_s": self._percentile(self._execution_times_s, 0.95),
            }


__all__ = ["BatchDispatch", "ContinuousBatchScheduler"]
