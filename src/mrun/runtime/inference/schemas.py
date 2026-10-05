"""Stable schemas for backend-neutral greedy generation.

The inference service deliberately exchanges token IDs rather than tokenizer text.  Text
decoding, chat templates, and transport framing belong above this boundary; native tensor and
KV objects remain below it.  All timestamps use the monotonic clock supplied to the service.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Literal

from ..contracts import CompatibleBatchLaneIdentity, SamplingPolicy
from .session import SessionCacheStatus

GENERATION_REQUEST_SCHEMA = "mrun-generation-request-v1"
GENERATION_RESULT_SCHEMA = "mrun-generation-result-v1"
GENERATION_EVENT_SCHEMA = "mrun-generation-event-v1"
GENERATION_TELEMETRY_SCHEMA = "mrun-generation-telemetry-v1"

_MAX_REQUEST_ID_UTF8_BYTES = 256


def _strict_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _strict_nonnegative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _finite_nonnegative(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return normalized


def _token_row(values: tuple[int, ...], field_name: str, *, empty: bool = False) -> tuple[int, ...]:
    row = tuple(_strict_nonnegative_int(value, f"{field_name}[]") for value in values)
    if not empty and not row:
        raise ValueError(f"{field_name} cannot be empty")
    return row


class FinishReason(str, Enum):
    """Successful request terminal condition."""

    MAX_NEW_TOKENS = "max_new_tokens"
    EOS_TOKEN = "eos_token"
    STOP_SEQUENCE = "stop_sequence"


class EventBackpressurePolicy(str, Enum):
    """Behavior when a streaming consumer exhausts its bounded token slots.

    ``TERMINATE_REQUEST`` is intentionally fail-fast.  Blocking the sole backend coordinator
    behind an absent network client would create cross-request head-of-line blocking.
    """

    TERMINATE_REQUEST = "terminate_request"


class TerminalStatus(str, Enum):
    """Exhaustive terminal accounting categories."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    FAILED = "failed"
    SHUTDOWN = "shutdown"
    BACKPRESSURE = "backpressure"


class StateRetentionOwner(str, Enum):
    """Owner responsible for releasing a successful retained native state."""

    NONE = "none"
    HANDLE = "handle"
    SESSION_STORE = "session_store"


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One autonomous B1 native token-generation request.

    Stop tokens are hidden by default.  When hidden, the streamer retains only the longest
    suffix that could still become a stop sequence and publishes every other token immediately.
    ``deadline`` is an absolute timestamp in the service's monotonic clock domain.
    """

    request_id: str
    input_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_ids: tuple[int, ...] = ()
    stop_sequences: tuple[tuple[int, ...], ...] = ()
    include_stop_tokens: bool = False
    stream: bool = True
    retain_state_on_success: bool = False
    session_id: str | None = None
    deadline: float | None = None
    state_capacity: int | None = None
    event_queue_capacity: int | None = None
    sampling: SamplingPolicy | None = None
    schema_version: Literal["mrun-generation-request-v1"] = GENERATION_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_REQUEST_SCHEMA:
            raise ValueError(f"unsupported generation-request schema: {self.schema_version}")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise TypeError("request_id must be a non-empty string")
        if self.request_id.strip() != self.request_id:
            raise ValueError("request_id cannot have surrounding whitespace")
        if len(self.request_id.encode("utf-8")) > _MAX_REQUEST_ID_UTF8_BYTES:
            raise ValueError(f"request_id cannot exceed {_MAX_REQUEST_ID_UTF8_BYTES} UTF-8 bytes")
        object.__setattr__(self, "input_ids", _token_row(self.input_ids, "input_ids"))
        object.__setattr__(
            self,
            "max_new_tokens",
            _strict_positive_int(self.max_new_tokens, "max_new_tokens"),
        )
        eos = tuple(
            sorted(
                {_strict_nonnegative_int(value, "eos_token_ids[]") for value in self.eos_token_ids}
            )
        )
        object.__setattr__(self, "eos_token_ids", eos)
        sequences = tuple(
            _token_row(tuple(sequence), "stop_sequences[]") for sequence in self.stop_sequences
        )
        if len(set(sequences)) != len(sequences):
            raise ValueError("stop_sequences must be unique")
        object.__setattr__(self, "stop_sequences", sequences)
        for field_name in ("include_stop_tokens", "stream", "retain_state_on_success"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be boolean")
        if self.session_id is not None:
            if (
                type(self.session_id) is not str
                or not self.session_id
                or self.session_id.strip() != self.session_id
                or len(self.session_id.encode("utf-8")) > 256
            ):
                raise ValueError("session_id must be a canonical string of at most 256 UTF-8 bytes")
        if self.deadline is not None:
            if isinstance(self.deadline, bool) or not isinstance(self.deadline, (int, float)):
                raise TypeError("deadline must be a finite monotonic timestamp or None")
            deadline = float(self.deadline)
            if not math.isfinite(deadline):
                raise ValueError("deadline must be a finite monotonic timestamp or None")
            object.__setattr__(self, "deadline", deadline)
        if self.state_capacity is not None:
            object.__setattr__(
                self,
                "state_capacity",
                _strict_positive_int(self.state_capacity, "state_capacity"),
            )
        if self.event_queue_capacity is not None:
            capacity = _strict_positive_int(
                self.event_queue_capacity,
                "event_queue_capacity",
            )
            if capacity < 2:
                raise ValueError(
                    "event_queue_capacity must be at least 2 (one token slot and one reserved "
                    "terminal slot)"
                )
            object.__setattr__(self, "event_queue_capacity", capacity)
        if self.sampling is not None and not isinstance(self.sampling, SamplingPolicy):
            raise TypeError("sampling must be SamplingPolicy or None")


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One visible token, published only after its producing state step committed."""

    request_id: str
    token_id: int
    token_index: int
    step_index: int
    committed_at: float
    published_at: float
    state_epoch: int
    state_length: int
    schema_version: Literal["mrun-generation-event-v1"] = GENERATION_EVENT_SCHEMA
    event_type: Literal["token"] = "token"

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_EVENT_SCHEMA:
            raise ValueError(f"unsupported generation-event schema: {self.schema_version}")
        if self.event_type != "token":
            raise ValueError("TokenEvent event_type must be 'token'")
        if not self.request_id:
            raise ValueError("token event request_id cannot be empty")
        for field_name in ("token_id", "token_index", "step_index", "state_epoch", "state_length"):
            object.__setattr__(
                self,
                field_name,
                _strict_nonnegative_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "committed_at",
            _finite_nonnegative(self.committed_at, "committed_at"),
        )
        object.__setattr__(
            self,
            "published_at",
            _finite_nonnegative(self.published_at, "published_at"),
        )
        if self.published_at < self.committed_at:
            raise ValueError("published_at cannot precede committed_at")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Successful receipt after state release or an explicit retained-state transfer."""

    request_id: str
    token_ids: tuple[int, ...]
    finish_reason: FinishReason | str
    matched_stop_sequence: tuple[int, ...] | None
    prompt_token_count: int
    model_generated_token_count: int
    committed_input_token_count: int
    final_state_length: int
    final_state_epoch: int
    pending_token_id: int
    step_count: int
    state_capacity: int
    ttft_seconds: float | None
    inter_token_seconds: tuple[float, ...]
    request_latency_seconds: float
    runtime_id: str
    backend_id: str
    state_retained: bool = False
    state_handoff_id: str | None = None
    state_retention_owner: StateRetentionOwner | str = StateRetentionOwner.NONE
    session_id: str | None = None
    session_cache_status: SessionCacheStatus | str = SessionCacheStatus.DISABLED
    schema_version: Literal["mrun-generation-result-v1"] = GENERATION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_RESULT_SCHEMA:
            raise ValueError(f"unsupported generation-result schema: {self.schema_version}")
        if not self.request_id or not self.runtime_id or not self.backend_id:
            raise ValueError("result request/runtime/backend identities cannot be empty")
        object.__setattr__(self, "token_ids", _token_row(self.token_ids, "token_ids", empty=True))
        try:
            reason = (
                self.finish_reason
                if isinstance(self.finish_reason, FinishReason)
                else FinishReason(str(self.finish_reason))
            )
        except ValueError as exc:
            raise ValueError("unsupported finish_reason") from exc
        object.__setattr__(self, "finish_reason", reason)
        if self.matched_stop_sequence is not None:
            object.__setattr__(
                self,
                "matched_stop_sequence",
                _token_row(self.matched_stop_sequence, "matched_stop_sequence"),
            )
        if reason is FinishReason.MAX_NEW_TOKENS and self.matched_stop_sequence is not None:
            raise ValueError("max-new-tokens completion cannot carry a matched stop sequence")
        if reason is not FinishReason.MAX_NEW_TOKENS and self.matched_stop_sequence is None:
            raise ValueError("stop completion requires matched_stop_sequence")
        for field_name in (
            "prompt_token_count",
            "model_generated_token_count",
            "committed_input_token_count",
            "final_state_length",
            "final_state_epoch",
            "pending_token_id",
            "step_count",
            "state_capacity",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_nonnegative_int(getattr(self, field_name), field_name),
            )
        if self.prompt_token_count <= 0 or self.model_generated_token_count <= 0:
            raise ValueError("successful generation must have prompt and generated tokens")
        if self.step_count != self.model_generated_token_count:
            raise ValueError("step_count must equal model_generated_token_count for native B1")
        if self.final_state_length > self.state_capacity:
            raise ValueError("final_state_length cannot exceed state_capacity")
        if self.ttft_seconds is not None:
            object.__setattr__(
                self,
                "ttft_seconds",
                _finite_nonnegative(self.ttft_seconds, "ttft_seconds"),
            )
        intervals = tuple(
            _finite_nonnegative(value, "inter_token_seconds[]")
            for value in self.inter_token_seconds
        )
        expected_intervals = max(len(self.token_ids) - 1, 0)
        if len(intervals) != expected_intervals:
            raise ValueError("inter_token_seconds must have one interval after each visible token")
        object.__setattr__(self, "inter_token_seconds", intervals)
        object.__setattr__(
            self,
            "request_latency_seconds",
            _finite_nonnegative(self.request_latency_seconds, "request_latency_seconds"),
        )
        if type(self.state_retained) is not bool:
            raise TypeError("state_retained must be boolean")
        if self.state_retained != (self.state_handoff_id is not None):
            raise ValueError("state_retained and state_handoff_id must agree")
        if self.state_handoff_id is not None and not self.state_handoff_id:
            raise ValueError("state_handoff_id cannot be empty")
        try:
            retention_owner = (
                self.state_retention_owner
                if isinstance(self.state_retention_owner, StateRetentionOwner)
                else StateRetentionOwner(str(self.state_retention_owner))
            )
        except ValueError as exc:
            raise ValueError("unsupported state_retention_owner") from exc
        object.__setattr__(self, "state_retention_owner", retention_owner)
        if self.state_retained != (retention_owner is not StateRetentionOwner.NONE):
            raise ValueError("retained-state owner must agree with state_retained")
        try:
            cache_status = (
                self.session_cache_status
                if isinstance(self.session_cache_status, SessionCacheStatus)
                else SessionCacheStatus(str(self.session_cache_status))
            )
        except ValueError as exc:
            raise ValueError("unsupported session_cache_status") from exc
        object.__setattr__(self, "session_cache_status", cache_status)
        if cache_status is SessionCacheStatus.DISABLED:
            if self.session_id is not None or retention_owner is StateRetentionOwner.SESSION_STORE:
                raise ValueError("disabled session cache cannot carry session-owned state")
        else:
            if (
                type(self.session_id) is not str
                or not self.session_id
                or retention_owner is not StateRetentionOwner.SESSION_STORE
            ):
                raise ValueError("session cache hit/miss requires session-store retained state")


@dataclass(frozen=True, slots=True)
class CompletedEvent:
    request_id: str
    result: GenerationResult
    created_at: float
    schema_version: Literal["mrun-generation-event-v1"] = GENERATION_EVENT_SCHEMA
    event_type: Literal["completed"] = "completed"

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_EVENT_SCHEMA:
            raise ValueError(f"unsupported generation-event schema: {self.schema_version}")
        if self.event_type != "completed":
            raise ValueError("CompletedEvent event_type must be 'completed'")
        if self.result.request_id != self.request_id:
            raise ValueError("completed event result identity mismatch")
        object.__setattr__(self, "created_at", _finite_nonnegative(self.created_at, "created_at"))


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    """Non-success terminal event with transport-safe error information."""

    request_id: str
    status: TerminalStatus | str
    error_code: str
    message: str
    created_at: float
    schema_version: Literal["mrun-generation-event-v1"] = GENERATION_EVENT_SCHEMA
    event_type: Literal["terminal"] = "terminal"

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_EVENT_SCHEMA:
            raise ValueError(f"unsupported generation-event schema: {self.schema_version}")
        if self.event_type != "terminal":
            raise ValueError("TerminalEvent event_type must be 'terminal'")
        if not self.request_id or not self.error_code or not self.message:
            raise ValueError("terminal event identity, error_code, and message cannot be empty")
        try:
            status = (
                self.status
                if isinstance(self.status, TerminalStatus)
                else TerminalStatus(str(self.status))
            )
        except ValueError as exc:
            raise ValueError("unsupported terminal status") from exc
        if status is TerminalStatus.COMPLETED:
            raise ValueError("completed requests require CompletedEvent")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "created_at", _finite_nonnegative(self.created_at, "created_at"))


GenerationEvent = TokenEvent | CompletedEvent | TerminalEvent


@dataclass(frozen=True, slots=True)
class LatencyDistribution:
    observation_count: int
    window_count: int
    window_capacity: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    p50: float | None
    p95: float | None
    p99: float | None

    def __post_init__(self) -> None:
        for field_name in ("observation_count", "window_count"):
            object.__setattr__(
                self,
                field_name,
                _strict_nonnegative_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "window_capacity",
            _strict_positive_int(self.window_capacity, "window_capacity"),
        )
        if self.window_count > self.window_capacity:
            raise ValueError("window_count cannot exceed window_capacity")
        if self.window_count > self.observation_count:
            raise ValueError("window_count cannot exceed observation_count")
        statistic_names = ("minimum", "maximum", "mean", "p50", "p95", "p99")
        statistics = tuple(getattr(self, name) for name in statistic_names)
        if self.window_count == 0:
            if any(value is not None for value in statistics):
                raise ValueError("empty latency windows cannot carry statistics")
            return
        if any(value is None for value in statistics):
            raise ValueError("non-empty latency windows require all statistics")
        normalized = tuple(
            _finite_nonnegative(value, name)
            for name, value in zip(statistic_names, statistics, strict=True)
        )
        for name, value in zip(statistic_names, normalized, strict=True):
            object.__setattr__(self, name, value)
        minimum, maximum, mean, p50, p95, p99 = normalized
        if not minimum <= p50 <= p95 <= p99 <= maximum:
            raise ValueError("latency quantiles must be ordered within minimum/maximum")
        if not minimum <= mean <= maximum:
            raise ValueError("latency mean must be within minimum/maximum")


@dataclass(frozen=True, slots=True)
class CompatibleBatchServiceTelemetry:
    """Scheduler-visible telemetry for one explicitly attached compatible batch lane."""

    identity: CompatibleBatchLaneIdentity
    dispatches: int
    dispatched_rows: int
    singleton_bypasses: int
    max_width: int
    width_histogram: tuple[tuple[int, int], ...]
    commits: int
    abandons: int
    queue_delay: LatencyDistribution
    forward_latency: LatencyDistribution

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CompatibleBatchLaneIdentity):
            raise TypeError("compatible-batch telemetry requires a lane identity")
        for field_name in (
            "dispatches",
            "dispatched_rows",
            "singleton_bypasses",
            "max_width",
            "commits",
            "abandons",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_nonnegative_int(getattr(self, field_name), field_name),
            )
        histogram = tuple(self.width_histogram)
        normalized = tuple(
            (
                _strict_positive_int(width, "width_histogram width"),
                _strict_positive_int(count, "width_histogram count"),
            )
            for width, count in histogram
        )
        if tuple(sorted(normalized)) != normalized or len(
            {width for width, _count in normalized}
        ) != len(normalized):
            raise ValueError("width_histogram must be sorted with unique widths")
        if sum(width * count for width, count in normalized) != self.dispatched_rows:
            raise ValueError("compatible-batch width histogram does not reconcile dispatched rows")
        if normalized and max(width for width, _count in normalized) != self.max_width:
            raise ValueError("compatible-batch max width disagrees with its histogram")
        if not normalized and self.max_width:
            raise ValueError("empty compatible-batch histogram requires max_width=0")
        if self.commits + self.abandons > self.dispatched_rows:
            raise ValueError("compatible-batch terminal rows exceed dispatched rows")
        object.__setattr__(self, "width_histogram", normalized)
        for field_name in ("queue_delay", "forward_latency"):
            if not isinstance(getattr(self, field_name), LatencyDistribution):
                raise TypeError(f"{field_name} must be a LatencyDistribution")


@dataclass(frozen=True, slots=True)
class GenerationServiceTelemetry:
    """Deterministic, internally reconciled service snapshot."""

    service_id: str
    runtime_id: str
    backend_id: str
    admitted: int
    active: int
    completed: int
    cancelled: int
    deadline_exceeded: int
    failed: int
    shutdown_terminated: int
    backpressure_terminated: int
    rejected_admission: int
    rejected_duplicate: int
    rejected_capacity: int
    forward_steps_started: int
    forward_steps_succeeded: int
    forward_steps_failed: int
    forward_steps_active: int
    provisional_steps: int
    commits: int
    abandons_attempted: int
    abandons_succeeded: int
    state_releases_attempted: int
    state_releases_succeeded: int
    cleanup_failures: int
    model_generated_tokens: int
    streamed_tokens: int
    event_queue_high_watermark: int
    event_queue_capacity: int
    backpressure_policy: EventBackpressurePolicy | str
    accepting: bool
    stopped: bool
    ttft: LatencyDistribution
    inter_token: LatencyDistribution
    request_latency: LatencyDistribution
    forward_latency: LatencyDistribution
    compatible_batch: CompatibleBatchServiceTelemetry | None = None
    schema_version: Literal["mrun-generation-telemetry-v1"] = GENERATION_TELEMETRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_TELEMETRY_SCHEMA:
            raise ValueError(f"unsupported generation-telemetry schema: {self.schema_version}")
        if not self.service_id or not self.runtime_id or not self.backend_id:
            raise ValueError("telemetry service/runtime/backend identities cannot be empty")
        counter_names = (
            "admitted",
            "active",
            "completed",
            "cancelled",
            "deadline_exceeded",
            "failed",
            "shutdown_terminated",
            "backpressure_terminated",
            "rejected_admission",
            "rejected_duplicate",
            "rejected_capacity",
            "forward_steps_started",
            "forward_steps_succeeded",
            "forward_steps_failed",
            "forward_steps_active",
            "provisional_steps",
            "commits",
            "abandons_attempted",
            "abandons_succeeded",
            "state_releases_attempted",
            "state_releases_succeeded",
            "cleanup_failures",
            "model_generated_tokens",
            "streamed_tokens",
            "event_queue_high_watermark",
        )
        for field_name in counter_names:
            object.__setattr__(
                self,
                field_name,
                _strict_nonnegative_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "event_queue_capacity",
            _strict_positive_int(self.event_queue_capacity, "event_queue_capacity"),
        )
        if self.event_queue_high_watermark > self.event_queue_capacity:
            raise ValueError("event queue high-watermark cannot exceed capacity")
        try:
            policy = (
                self.backpressure_policy
                if isinstance(self.backpressure_policy, EventBackpressurePolicy)
                else EventBackpressurePolicy(str(self.backpressure_policy))
            )
        except ValueError as exc:
            raise ValueError("unsupported event backpressure policy") from exc
        object.__setattr__(self, "backpressure_policy", policy)
        for field_name in ("accepting", "stopped"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be boolean")
        for field_name in ("ttft", "inter_token", "request_latency", "forward_latency"):
            if not isinstance(getattr(self, field_name), LatencyDistribution):
                raise TypeError(f"{field_name} must be a LatencyDistribution")
        if self.compatible_batch is not None and not isinstance(
            self.compatible_batch, CompatibleBatchServiceTelemetry
        ):
            raise TypeError("compatible_batch must be CompatibleBatchServiceTelemetry or None")

    @property
    def reconciled(self) -> bool:
        terminal = (
            self.completed
            + self.cancelled
            + self.deadline_exceeded
            + self.failed
            + self.shutdown_terminated
            + self.backpressure_terminated
        )
        return bool(
            self.admitted == self.active + terminal
            and self.forward_steps_started
            == (
                self.forward_steps_succeeded + self.forward_steps_failed + self.forward_steps_active
            )
            and self.commits <= self.provisional_steps
            and self.abandons_succeeded <= self.abandons_attempted
            and self.state_releases_succeeded <= self.state_releases_attempted
        )


__all__ = [
    "GENERATION_EVENT_SCHEMA",
    "GENERATION_REQUEST_SCHEMA",
    "GENERATION_RESULT_SCHEMA",
    "GENERATION_TELEMETRY_SCHEMA",
    "CompletedEvent",
    "CompatibleBatchServiceTelemetry",
    "EventBackpressurePolicy",
    "FinishReason",
    "GenerationEvent",
    "GenerationRequest",
    "GenerationResult",
    "GenerationServiceTelemetry",
    "LatencyDistribution",
    "StateRetentionOwner",
    "TerminalEvent",
    "TerminalStatus",
    "TokenEvent",
]
