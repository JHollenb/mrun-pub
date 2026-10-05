"""Dependency-free operational controls for the single-host inference data plane.

The rate limiter deliberately keys an in-memory digest of the direct peer address.  It never
trusts forwarding headers and never emits a client address.  The lifecycle logger accepts a
closed set of aggregate fields so prompts, message content, token IDs, credentials, and session
identifiers cannot accidentally enter request logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import secrets
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass


class InferenceRateLimitError(RuntimeError):
    """A direct client exceeded the bounded token bucket contract."""

    def __init__(self, message: str, *, code: str, retry_after_seconds: int) -> None:
        self.code = code
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RateLimitSnapshot:
    clients: int
    allowed_requests: int
    rejected_requests: int
    charged_tokens: int
    oversized_rejections: int
    client_capacity_rejections: int


@dataclass(slots=True)
class _TokenBucket:
    tokens: float
    refilled_at: float
    last_seen_at: float


class PerIpTokenLimiter:
    """Bounded direct-peer token buckets intended for one inference process.

    Token budgets are charged up front from rendered prompt length plus requested completion
    length.  They are intentionally not refunded: the limiter bounds admitted compute rather than
    attempting to reconstruct post-hoc billing from cancellation and stop behavior.
    """

    def __init__(
        self,
        *,
        tokens_per_minute: int,
        burst_tokens: int,
        max_clients: int,
        idle_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for value, field in (
            (tokens_per_minute, "tokens_per_minute"),
            (burst_tokens, "burst_tokens"),
            (max_clients, "max_clients"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if (
            isinstance(idle_seconds, bool)
            or not isinstance(idle_seconds, (int, float))
            or not math.isfinite(float(idle_seconds))
            or float(idle_seconds) <= 0
        ):
            raise ValueError("idle_seconds must be a finite positive number")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.tokens_per_minute = int(tokens_per_minute)
        self.burst_tokens = int(burst_tokens)
        self.max_clients = int(max_clients)
        self.idle_seconds = float(idle_seconds)
        self._tokens_per_second = self.tokens_per_minute / 60.0
        self._clock = clock
        self._key_secret = secrets.token_bytes(32)
        self._buckets: dict[bytes, _TokenBucket] = {}
        self._lock = threading.Lock()
        self._last_now: float | None = None
        self._allowed_requests = 0
        self._rejected_requests = 0
        self._charged_tokens = 0
        self._oversized_rejections = 0
        self._client_capacity_rejections = 0

    def _key(self, client_identifier: str) -> bytes:
        if type(client_identifier) is not str or not client_identifier:
            client_identifier = "unknown-direct-peer"
        return hashlib.blake2b(
            client_identifier.encode("utf-8", errors="replace"),
            key=self._key_secret,
            digest_size=16,
        ).digest()

    def _now_unlocked(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now) or now < 0:
            raise RuntimeError("rate-limit clock returned an invalid timestamp")
        if self._last_now is not None and now < self._last_now:
            raise RuntimeError("rate-limit clock regressed")
        self._last_now = now
        return now

    def _prune_unlocked(self, now: float) -> None:
        expired = [
            key
            for key, bucket in self._buckets.items()
            if now - bucket.last_seen_at >= self.idle_seconds
        ]
        for key in expired:
            self._buckets.pop(key, None)

    def charge(self, client_identifier: str, token_cost: int) -> None:
        if isinstance(token_cost, bool) or not isinstance(token_cost, int) or token_cost <= 0:
            raise ValueError("token_cost must be a positive integer")
        key = self._key(client_identifier)
        with self._lock:
            now = self._now_unlocked()
            if token_cost > self.burst_tokens:
                self._rejected_requests += 1
                self._oversized_rejections += 1
                raise InferenceRateLimitError(
                    "request token budget exceeds the per-client burst limit",
                    code="token_budget_exceeds_burst",
                    retry_after_seconds=60,
                )
            bucket = self._buckets.get(key)
            if bucket is None:
                self._prune_unlocked(now)
                if len(self._buckets) >= self.max_clients:
                    self._rejected_requests += 1
                    self._client_capacity_rejections += 1
                    raise InferenceRateLimitError(
                        "rate-limit client capacity is exhausted",
                        code="rate_limit_client_capacity",
                        retry_after_seconds=max(1, math.ceil(self.idle_seconds)),
                    )
                bucket = _TokenBucket(
                    tokens=float(self.burst_tokens),
                    refilled_at=now,
                    last_seen_at=now,
                )
                self._buckets[key] = bucket
            elapsed = now - bucket.refilled_at
            bucket.tokens = min(
                float(self.burst_tokens),
                bucket.tokens + elapsed * self._tokens_per_second,
            )
            bucket.refilled_at = now
            bucket.last_seen_at = now
            if bucket.tokens + 1e-12 < token_cost:
                deficit = token_cost - bucket.tokens
                self._rejected_requests += 1
                raise InferenceRateLimitError(
                    "per-client token rate limit exceeded",
                    code="token_rate_limit_exceeded",
                    retry_after_seconds=max(
                        1,
                        math.ceil(deficit / self._tokens_per_second),
                    ),
                )
            bucket.tokens -= token_cost
            self._allowed_requests += 1
            self._charged_tokens += token_cost

    def snapshot(self) -> RateLimitSnapshot:
        with self._lock:
            now = self._now_unlocked()
            self._prune_unlocked(now)
            return RateLimitSnapshot(
                clients=len(self._buckets),
                allowed_requests=self._allowed_requests,
                rejected_requests=self._rejected_requests,
                charged_tokens=self._charged_tokens,
                oversized_rejections=self._oversized_rejections,
                client_capacity_rejections=self._client_capacity_rejections,
            )


_LIFECYCLE_EVENTS = {
    "admitted",
    "cancelled",
    "completed",
    "disconnected",
    "failed",
    "rejected",
}
_LIFECYCLE_FIELDS = {
    "request_id",
    "model_id",
    "route_id",
    "stream",
    "status",
    "error_code",
    "prompt_token_count",
    "requested_completion_tokens",
    "completion_token_count",
    "latency_ms",
    "session_requested",
    "session_cache",
    "rate_limit_token_cost",
}


class PrivacySafeLifecycleLogger:
    """Emit JSON lifecycle records through a closed, content-free schema."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("lifecycle logger must be logging.Logger or None")
        if not callable(wall_clock):
            raise TypeError("wall_clock must be callable")
        self._logger = logger or logging.getLogger("mrun.inference.lifecycle")
        self._wall_clock = wall_clock

    def emit(self, event: str, **fields: object) -> None:
        if event not in _LIFECYCLE_EVENTS:
            raise ValueError("unsupported inference lifecycle event")
        unknown = set(fields) - _LIFECYCLE_FIELDS
        if unknown:
            raise ValueError(f"privacy-safe lifecycle schema rejected fields: {sorted(unknown)!r}")
        timestamp = float(self._wall_clock())
        if not math.isfinite(timestamp) or timestamp < 0:
            raise RuntimeError("lifecycle wall clock returned an invalid timestamp")
        payload: dict[str, object] = {
            "schema": "mrun-inference-request-lifecycle-v1",
            "event": event,
            "timestamp": timestamp,
        }
        for key, value in fields.items():
            if value is None:
                continue
            if not isinstance(value, str | int | float | bool):
                raise TypeError(f"lifecycle field {key!r} must be a JSON scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"lifecycle field {key!r} must be finite")
            if isinstance(value, str) and (len(value) > 512 or "\n" in value or "\r" in value):
                raise ValueError(f"lifecycle field {key!r} is not a bounded canonical string")
            payload[key] = value
        self._logger.info(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )


_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


@dataclass(frozen=True, slots=True)
class PrometheusSample:
    name: str
    metric_type: str
    help: str
    value: int | float

    def __post_init__(self) -> None:
        if not _METRIC_NAME.fullmatch(self.name):
            raise ValueError(f"invalid Prometheus metric name: {self.name!r}")
        if self.metric_type not in {"counter", "gauge"}:
            raise ValueError("Prometheus metric_type must be counter or gauge")
        if type(self.help) is not str or not self.help or "\n" in self.help:
            raise ValueError("Prometheus help must be a non-empty single line")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise TypeError("Prometheus value must be numeric")
        if not math.isfinite(float(self.value)):
            raise ValueError("Prometheus value must be finite")


def prometheus_text(samples: Iterable[PrometheusSample]) -> str:
    ordered = sorted(tuple(samples), key=lambda sample: sample.name)
    if len({sample.name for sample in ordered}) != len(ordered):
        raise ValueError("Prometheus samples must have unique names")
    lines: list[str] = []
    for sample in ordered:
        help_text = sample.help.replace("\\", "\\\\")
        value = str(sample.value) if isinstance(sample.value, int) else format(sample.value, ".17g")
        lines.extend(
            (
                f"# HELP {sample.name} {help_text}",
                f"# TYPE {sample.name} {sample.metric_type}",
                f"{sample.name} {value}",
            )
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "InferenceRateLimitError",
    "PerIpTokenLimiter",
    "PrivacySafeLifecycleLogger",
    "PrometheusSample",
    "RateLimitSnapshot",
    "prometheus_text",
]
