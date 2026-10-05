"""Separate OpenAI-compatible inference data-plane application.

The fleet scheduler in :mod:`mrun.server.app` is intentionally not imported here.  This process
owns one warm native generation service and translates chat text at the boundary; the hot model
runtime receives semantic token IDs only.
"""

import asyncio
import hmac
import json
import logging
import secrets
import threading
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from mrun.runtime import NATIVE_SAMPLING_ABI, OutputMode, PromotionStatus, SamplingPolicy
from mrun.runtime.inference import (
    CompletedEvent,
    FinishReason,
    GenerationAdmissionError,
    GenerationBackpressureError,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationEventTimeout,
    GenerationRequest,
    GenerationServiceClosed,
    GenerationServiceError,
    GenerationShutdown,
    SessionBusyError,
    SessionCapacityError,
    SessionIdentityError,
    SessionIntegrityError,
    SessionPrefixMismatch,
    SessionStoreClosed,
    TerminalEvent,
    TerminalStatus,
    TokenEvent,
)

from .chat import BoundChatTokenizer, ChatValidationError, IncrementalTextDecoder
from .openai_protocol import (
    OpenAIChatRequest,
    Usage,
    completion_chunk,
    completion_response,
    error_envelope,
    parse_chat_completion_request,
)
from .operations import (
    InferenceRateLimitError,
    PerIpTokenLimiter,
    PrivacySafeLifecycleLogger,
    PrometheusSample,
    prometheus_text,
)
from .web_ui import browser_chat_html


@dataclass(frozen=True, slots=True)
class InferenceAppConfig:
    max_request_bytes: int = 1_048_576
    max_http_requests: int = 256
    default_max_tokens: int = 256
    event_poll_seconds: float = 0.25
    request_timeout_seconds: float | None = None
    expose_metrics: bool = True
    serve_browser_ui: bool = True
    rate_limit_tokens_per_minute: int | None = 12_000
    rate_limit_burst_tokens: int = 4_096
    rate_limit_max_clients: int = 1_024
    rate_limit_idle_seconds: float = 600.0

    def __post_init__(self) -> None:
        for value, field in (
            (self.max_request_bytes, "max_request_bytes"),
            (self.max_http_requests, "max_http_requests"),
            (self.default_max_tokens, "default_max_tokens"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if (
            isinstance(self.event_poll_seconds, bool)
            or not isinstance(self.event_poll_seconds, (int, float))
            or self.event_poll_seconds <= 0
        ):
            raise ValueError("event_poll_seconds must be positive")
        if self.request_timeout_seconds is not None and (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive or None")
        if type(self.expose_metrics) is not bool:
            raise TypeError("expose_metrics must be boolean")
        if type(self.serve_browser_ui) is not bool:
            raise TypeError("serve_browser_ui must be boolean")
        if self.rate_limit_tokens_per_minute is not None and (
            isinstance(self.rate_limit_tokens_per_minute, bool)
            or not isinstance(self.rate_limit_tokens_per_minute, int)
            or self.rate_limit_tokens_per_minute <= 0
        ):
            raise ValueError("rate_limit_tokens_per_minute must be positive or None")
        for value, field in (
            (self.rate_limit_burst_tokens, "rate_limit_burst_tokens"),
            (self.rate_limit_max_clients, "rate_limit_max_clients"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if (
            isinstance(self.rate_limit_idle_seconds, bool)
            or not isinstance(self.rate_limit_idle_seconds, (int, float))
            or self.rate_limit_idle_seconds <= 0
        ):
            raise ValueError("rate_limit_idle_seconds must be positive")


class _HttpAdmission:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.active = 0
        self.rejected = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self.active >= self.capacity:
                self.rejected += 1
                return False
            self.active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self.active <= 0:
                raise RuntimeError("HTTP admission accounting underflow")
            self.active -= 1

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.active, self.rejected


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def strict_json_object(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            data,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ChatValidationError(f"request body is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ChatValidationError("request body must be a JSON object")
    return payload


async def _read_limited_body(request: Any, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > maximum:
            raise ChatValidationError("request body exceeds the configured byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _sse(payload: Mapping[str, Any] | str) -> bytes:
    body = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    return f"data: {body}\n\n".encode()


def _finish_reason(reason: FinishReason) -> str:
    return "length" if reason is FinishReason.MAX_NEW_TOKENS else "stop"


class InferenceHost:
    """Text/protocol owner over one warm backend-neutral generation service."""

    def __init__(
        self,
        *,
        tokenizer: BoundChatTokenizer,
        generation_service: Any,
        route_id: str,
        promotion_status: PromotionStatus,
        config: InferenceAppConfig | None = None,
        lifecycle_logger: logging.Logger | None = None,
    ) -> None:
        if type(route_id) is not str or not route_id:
            raise ValueError("route_id must be a non-empty string")
        if not isinstance(promotion_status, PromotionStatus):
            raise TypeError("promotion_status must be PromotionStatus")
        if not callable(getattr(generation_service, "submit", None)) or not callable(
            getattr(generation_service, "telemetry", None)
        ):
            raise TypeError("generation_service must expose submit and telemetry")
        self.tokenizer = tokenizer
        self.generation_service = generation_service
        self.route_id = route_id
        self.promotion_status = promotion_status
        declared_output_modes = getattr(generation_service, "supported_output_modes", None)
        if declared_output_modes is None:
            raise TypeError("generation_service must expose supported_output_modes")
        output_modes = tuple(declared_output_modes)
        if not output_modes or any(not isinstance(mode, OutputMode) for mode in output_modes):
            raise TypeError(
                "generation_service supported_output_modes must contain OutputMode values"
            )
        if len(set(output_modes)) != len(output_modes):
            raise ValueError("generation_service supported_output_modes must be unique")
        if OutputMode.NEXT_TOKEN_ARGMAX not in output_modes:
            raise ValueError("chat inference requires next-token-argmax support")
        self.supported_output_modes = output_modes
        self.session_store = getattr(generation_service, "session_store", None)
        if self.session_store is not None:
            self.session_store.identity.validate_chat_boundary(
                model_id=tokenizer.model_id,
                chat_template_sha256=tokenizer.chat_template_sha256,
                semantic_token_count=tokenizer.semantic_token_count,
                route_id=route_id,
            )
        self.config = config or InferenceAppConfig()
        self._admission = _HttpAdmission(self.config.max_http_requests)
        self._rate_limiter = (
            None
            if self.config.rate_limit_tokens_per_minute is None
            else PerIpTokenLimiter(
                tokens_per_minute=self.config.rate_limit_tokens_per_minute,
                burst_tokens=self.config.rate_limit_burst_tokens,
                max_clients=self.config.rate_limit_max_clients,
                idle_seconds=self.config.rate_limit_idle_seconds,
            )
        )
        self._lifecycle = PrivacySafeLifecycleLogger(lifecycle_logger)
        self._ready = True
        self._draining = False
        self._lock = threading.Lock()
        self.requests_started = 0
        self.requests_completed = 0
        self.requests_failed = 0
        self.disconnect_cancellations = 0
        self.lifecycle_logging_failures = 0

    @property
    def model_id(self) -> str:
        return self.tokenizer.model_id

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready and not self._draining

    @property
    def sessions_enabled(self) -> bool:
        return self.session_store is not None

    @property
    def sampling_enabled(self) -> bool:
        return OutputMode.NEXT_TOKEN_SAMPLE in self.supported_output_modes

    def begin_drain(self) -> None:
        with self._lock:
            self._draining = True

    def mark_unready(self) -> None:
        with self._lock:
            self._ready = False

    def prepare(
        self,
        payload: Any,
    ) -> tuple[OpenAIChatRequest, tuple[int, ...], GenerationRequest, str, int]:
        if not self.ready:
            raise GenerationServiceClosed("inference service is draining or not ready")
        request = parse_chat_completion_request(
            payload,
            loaded_model=self.model_id,
            semantic_token_count=self.tokenizer.semantic_token_count,
            default_max_tokens=self.config.default_max_tokens,
        )
        if request.session_id is not None and self.session_store is None:
            raise ChatValidationError(
                "session_id requires a configured native exact-prefix store",
                code="sessions_unavailable",
            )
        rendered = self.tokenizer.render(request.messages, max_new_tokens=request.max_tokens)
        token_stops = self.tokenizer.encode_stop_text(request.stop_strings)
        eos = getattr(self.tokenizer.tokenizer, "eos_token_id", None)
        if eos is None:
            eos_ids: tuple[int, ...] = ()
        elif isinstance(eos, int) and not isinstance(eos, bool):
            eos_ids = (int(eos),)
        elif isinstance(eos, (list, tuple)):
            eos_ids = tuple(
                sorted(
                    {
                        int(value)
                        for value in eos
                        if isinstance(value, int) and not isinstance(value, bool)
                    }
                )
            )
        else:
            raise ChatValidationError(
                "tokenizer exposes an invalid EOS identity",
                code="model_identity_mismatch",
            )
        if any(value < 0 or value >= self.tokenizer.semantic_token_count for value in eos_ids):
            raise ChatValidationError(
                "tokenizer EOS escapes the semantic token domain",
                code="model_identity_mismatch",
            )
        request_id = f"chat.{uuid4().hex}"
        created = int(time.time())
        deadline = (
            None
            if self.config.request_timeout_seconds is None
            else time.monotonic() + self.config.request_timeout_seconds
        )
        generation = GenerationRequest(
            request_id=request_id,
            input_ids=rendered.token_ids,
            max_new_tokens=request.max_tokens,
            eos_token_ids=eos_ids,
            stop_sequences=token_stops,
            include_stop_tokens=False,
            stream=request.stream,
            retain_state_on_success=bool(request.session_id),
            session_id=request.session_id,
            deadline=deadline,
            state_capacity=len(rendered.token_ids) + request.max_tokens - 1,
            sampling=SamplingPolicy(
                seed=(
                    request.sampling.seed
                    if request.sampling.seed is not None
                    else secrets.randbits(63)
                ),
                temperature=request.sampling.temperature,
                top_p=request.sampling.top_p,
                top_k=request.sampling.top_k,
                frequency_penalty=request.sampling.frequency_penalty,
                presence_penalty=request.sampling.presence_penalty,
                logit_bias=request.sampling.logit_bias,
            ),
        )
        return request, rendered.token_ids, generation, request_id, created

    def acquire_request(self) -> None:
        if not self._admission.acquire():
            raise GenerationBackpressureError("HTTP request concurrency is exhausted")
        with self._lock:
            self.requests_started += 1

    def finish_request(self, *, failed: bool) -> None:
        self._admission.release()
        with self._lock:
            if failed:
                self.requests_failed += 1
            else:
                self.requests_completed += 1

    def charge_token_budget(self, client_identifier: str, token_cost: int) -> None:
        if self._rate_limiter is not None:
            self._rate_limiter.charge(client_identifier, token_cost)

    def record_disconnect(self) -> None:
        with self._lock:
            self.disconnect_cancellations += 1

    def log_lifecycle(self, event: str, **fields: object) -> None:
        try:
            self._lifecycle.emit(event, **fields)
        except Exception:
            # Logging must never take down generation.  The failure itself is observable without
            # reflecting rejected content or credentials into a fallback log record.
            with self._lock:
                self.lifecycle_logging_failures += 1

    def metrics_text(self) -> str:
        service = self.generation_service.telemetry()
        active_http, rejected_http = self._admission.snapshot()
        rate = self._rate_limiter.snapshot() if self._rate_limiter is not None else None
        with self._lock:
            values: dict[str, tuple[str, str, int | float]] = {
                "mrun_inference_ready": (
                    "gauge",
                    "Whether this process currently admits new inference requests.",
                    int(self._ready and not self._draining),
                ),
                "mrun_inference_http_active": (
                    "gauge",
                    "HTTP chat requests currently holding an admission slot.",
                    active_http,
                ),
                "mrun_inference_http_rejected_total": (
                    "counter",
                    "HTTP requests rejected because concurrency was exhausted.",
                    rejected_http,
                ),
                "mrun_inference_requests_started_total": (
                    "counter",
                    "Authenticated chat requests entering bounded HTTP admission.",
                    self.requests_started,
                ),
                "mrun_inference_requests_completed_total": (
                    "counter",
                    "Chat requests completing without a transport or generation failure.",
                    self.requests_completed,
                ),
                "mrun_inference_requests_failed_total": (
                    "counter",
                    "Chat requests ending in rejection, cancellation, or failure.",
                    self.requests_failed,
                ),
                "mrun_inference_disconnect_cancellations_total": (
                    "counter",
                    "Generation requests cancelled after their HTTP client disconnected.",
                    self.disconnect_cancellations,
                ),
                "mrun_inference_lifecycle_logging_failures_total": (
                    "counter",
                    "Privacy-safe lifecycle records dropped because logging failed.",
                    self.lifecycle_logging_failures,
                ),
                "mrun_generation_active": (
                    "gauge",
                    "Native generation requests currently active.",
                    int(service.active),
                ),
                "mrun_generation_admitted_total": (
                    "counter",
                    "Native generation requests admitted by the coordinator.",
                    int(service.admitted),
                ),
                "mrun_generation_completed_total": (
                    "counter",
                    "Native generation requests completed successfully.",
                    int(service.completed),
                ),
                "mrun_generation_cancelled_total": (
                    "counter",
                    "Native generation requests cancelled.",
                    int(service.cancelled),
                ),
                "mrun_generation_failed_total": (
                    "counter",
                    "Native generation requests failed.",
                    int(service.failed),
                ),
                "mrun_generation_model_tokens_total": (
                    "counter",
                    "Tokens selected by the native generation coordinator.",
                    int(service.model_generated_tokens),
                ),
                "mrun_generation_streamed_tokens_total": (
                    "counter",
                    "Committed tokens published to streaming consumers.",
                    int(service.streamed_tokens),
                ),
                "mrun_inference_rate_limit_clients": (
                    "gauge",
                    "Direct-peer token buckets retained in bounded memory.",
                    0 if rate is None else rate.clients,
                ),
                "mrun_inference_rate_limit_allowed_total": (
                    "counter",
                    "Requests admitted by the per-peer token limiter.",
                    0 if rate is None else rate.allowed_requests,
                ),
                "mrun_inference_rate_limit_rejected_total": (
                    "counter",
                    "Requests rejected by the per-peer token limiter.",
                    0 if rate is None else rate.rejected_requests,
                ),
                "mrun_inference_rate_limit_charged_tokens_total": (
                    "counter",
                    "Rendered prompt plus requested completion tokens charged at admission.",
                    0 if rate is None else rate.charged_tokens,
                ),
                "mrun_inference_rate_limit_oversized_total": (
                    "counter",
                    "Requests whose token budget exceeded one bucket burst.",
                    0 if rate is None else rate.oversized_rejections,
                ),
                "mrun_inference_rate_limit_client_capacity_total": (
                    "counter",
                    "New direct peers rejected because the bounded bucket table was full.",
                    0 if rate is None else rate.client_capacity_rejections,
                ),
            }
            if self.session_store is not None:
                sessions = self.session_store.telemetry()
                values.update(
                    {
                        "mrun_session_entries": (
                            "gauge",
                            "Content-indexed exact-prefix session entries available for reuse.",
                            sessions.entries,
                        ),
                        "mrun_session_retired_entries": (
                            "gauge",
                            "Opaque session authorities retired from lookup but awaiting release.",
                            sessions.retired_entries,
                        ),
                        "mrun_session_active_leases": (
                            "gauge",
                            "Exact-prefix session continuations currently leased.",
                            sessions.active_leases,
                        ),
                        "mrun_session_stored_bytes": (
                            "gauge",
                            "Native state bytes charged to retained sessions.",
                            sessions.stored_bytes,
                        ),
                        "mrun_session_reserved_bytes": (
                            "gauge",
                            "Native state bytes reserved by active session leases.",
                            sessions.reserved_bytes,
                        ),
                        "mrun_session_reserved_slots": (
                            "gauge",
                            "Retained-state slots reserved by active session leases.",
                            sessions.reserved_slots,
                        ),
                        "mrun_session_pinned_sources": (
                            "gauge",
                            "Exact-prefix source pins protecting in-flight native forks.",
                            sessions.pinned_sources,
                        ),
                        "mrun_session_hits_total": (
                            "counter",
                            "Session continuations served from an exact-prefix hit.",
                            sessions.hits,
                        ),
                        "mrun_session_misses_total": (
                            "counter",
                            "Session continuations requiring a fresh prefix.",
                            sessions.misses,
                        ),
                        "mrun_session_cross_prefix_hits_total": (
                            "counter",
                            "Cross-session continuations served from a content-addressed prefix.",
                            sessions.cross_session_prefix_hits,
                        ),
                        "mrun_session_cross_prefix_tokens_total": (
                            "counter",
                            "Committed prefix tokens reused across session IDs.",
                            sessions.cross_session_prefix_tokens,
                        ),
                        "mrun_session_cross_prefix_bytes_total": (
                            "counter",
                            "Native state bytes copied for cross-session exact-prefix forks.",
                            sessions.cross_session_prefix_bytes,
                        ),
                        "mrun_session_prefix_rejections_total": (
                            "counter",
                            "Session requests rejected for non-exact prompt extension.",
                            sessions.prefix_rejections,
                        ),
                        "mrun_session_evictions_total": (
                            "counter",
                            "Session entries evicted by TTL, LRU, or explicit removal.",
                            (
                                sessions.ttl_evictions
                                + sessions.lru_evictions
                                + sessions.manual_evictions
                            ),
                        ),
                        "mrun_session_cleanup_failures_total": (
                            "counter",
                            "Session native-state cleanup failures.",
                            sessions.cleanup_failures,
                        ),
                        "mrun_session_poisoned": (
                            "gauge",
                            "Whether a session cleanup invariant poisoned the store.",
                            int(sessions.poisoned),
                        ),
                        "mrun_session_budget_reconciled": (
                            "gauge",
                            "Whether retained plus reserved byte and slot budgets reconcile.",
                            int(sessions.budget_reconciled),
                        ),
                    }
                )
            batch = getattr(service, "compatible_batch", None)
            if batch is not None:
                values.update(
                    {
                        "mrun_generation_batch_dispatches_total": (
                            "counter",
                            "Compatible-request scheduler waves dispatched.",
                            batch.dispatches,
                        ),
                        "mrun_generation_batch_rows_total": (
                            "counter",
                            "Request rows dispatched through the compatible scheduler.",
                            batch.dispatched_rows,
                        ),
                        "mrun_generation_batch_singleton_bypasses_total": (
                            "counter",
                            "Scheduler waves preserving the exact singleton runtime path.",
                            batch.singleton_bypasses,
                        ),
                        "mrun_generation_batch_width_max": (
                            "gauge",
                            "Largest compatible scheduler wave observed.",
                            batch.max_width,
                        ),
                        "mrun_generation_batch_commits_total": (
                            "counter",
                            "Compatible-lane rows committed independently.",
                            batch.commits,
                        ),
                        "mrun_generation_batch_abandons_total": (
                            "counter",
                            "Compatible-lane rows abandoned independently.",
                            batch.abandons,
                        ),
                        "mrun_generation_batch_queue_delay_seconds_p95": (
                            "gauge",
                            "Rolling p95 compatible scheduler queue delay in seconds.",
                            batch.queue_delay.p95 or 0.0,
                        ),
                        "mrun_generation_batch_forward_seconds_p95": (
                            "gauge",
                            "Rolling p95 compatible wave forward latency in seconds.",
                            batch.forward_latency.p95 or 0.0,
                        ),
                    }
                )
        return prometheus_text(
            PrometheusSample(name, metric_type, help_text, value)
            for name, (metric_type, help_text, value) in values.items()
        )


def _http_error(error: BaseException) -> tuple[int, str, str, str]:
    if isinstance(error, ChatValidationError):
        status = 404 if error.code == "model_not_found" else 400
        return status, "invalid_request_error", error.code, str(error)
    if isinstance(error, GenerationBackpressureError):
        return 429, "rate_limit_error", "capacity_exhausted", str(error)
    if isinstance(error, InferenceRateLimitError):
        return 429, "rate_limit_error", error.code, str(error)
    if isinstance(error, SessionBusyError):
        return 409, "invalid_request_error", "session_busy", str(error)
    if isinstance(error, SessionCapacityError):
        return 429, "rate_limit_error", "session_capacity_exhausted", str(error)
    if isinstance(error, SessionPrefixMismatch):
        return 409, "invalid_request_error", "session_prefix_mismatch", str(error)
    if isinstance(error, SessionIdentityError):
        return 409, "invalid_request_error", "session_identity_mismatch", str(error)
    if isinstance(error, SessionIntegrityError):
        return 409, "invalid_request_error", "session_integrity_error", str(error)
    if isinstance(error, SessionStoreClosed):
        return 503, "service_unavailable", "session_store_closed", str(error)
    if isinstance(error, GenerationDeadlineExceeded):
        return 504, "deadline_exceeded", "deadline_exceeded", str(error)
    if isinstance(error, GenerationCancelled):
        return 499, "invalid_request_error", "client_closed_request", "request was cancelled"
    if isinstance(error, (GenerationServiceClosed, GenerationShutdown)):
        return 503, "service_unavailable", "service_unavailable", str(error)
    if isinstance(error, GenerationAdmissionError):
        return 400, "invalid_request_error", "generation_admission_error", str(error)
    incident = f"incident.{uuid4().hex}"
    return 500, "server_error", incident, "internal inference failure"


def _direct_client_identifier(request: Any) -> str:
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return host if type(host) is str and host else "unknown-direct-peer"


async def _await_generation_result(
    request: Any,
    handle: Any,
    *,
    request_id: str,
    poll_seconds: float,
) -> Any:
    """Wait for a non-streaming result while retaining transport cancellation authority."""

    while not handle.done():
        if await request.is_disconnected():
            handle.cancel("client_disconnected")
            raise GenerationCancelled(request_id, "client_disconnected")
        await asyncio.sleep(poll_seconds)
    return await asyncio.to_thread(handle.result)


def create_inference_app(
    host: InferenceHost,
    *,
    bearer_token: str | None = None,
):
    """Create a FastAPI application without importing server dependencies at package import."""

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import (
            HTMLResponse,
            JSONResponse,
            PlainTextResponse,
            StreamingResponse,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional server extra
        raise RuntimeError("inference HTTP requires mrun[server]") from exc
    if not isinstance(host, InferenceHost):
        raise TypeError("host must be InferenceHost")
    if bearer_token is not None and (type(bearer_token) is not str or not bearer_token):
        raise ValueError("bearer_token must be a non-empty string or None")

    app = FastAPI(title="mrun inference", docs_url=None, redoc_url=None)
    app.state.inference_host = host

    def authorized(request: Request) -> bool:
        if bearer_token is None:
            return True
        authorization = request.headers.get("authorization", "")
        prefix = "Bearer "
        supplied = authorization[len(prefix) :] if authorization.startswith(prefix) else ""
        return hmac.compare_digest(supplied, bearer_token)

    def json_error(error: BaseException):
        status, error_type, code, message = _http_error(error)
        headers = {"Cache-Control": "no-store"}
        if isinstance(error, InferenceRateLimitError):
            headers["Retry-After"] = str(error.retry_after_seconds)
        return JSONResponse(
            error_envelope(message, error_type=error_type, code=code),
            status_code=status,
            headers=headers,
        )

    @app.get("/", response_class=HTMLResponse)
    @app.get("/ui", response_class=HTMLResponse)
    async def browser_chat():
        if not host.config.serve_browser_ui:
            return JSONResponse({"detail": "not found"}, status_code=404)
        nonce = secrets.token_urlsafe(18)
        return HTMLResponse(
            browser_chat_html(nonce),
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; "
                    f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                    "connect-src 'self'; img-src 'self' data:; "
                    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
                    "form-action 'self'"
                ),
                "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
            },
        )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        if not host.ready:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return {
            "status": "ready",
            "model": host.model_id,
            "route_id": host.route_id,
            "template_sha256": host.tokenizer.chat_template_sha256,
        }

    @app.get("/v1/models")
    async def models(request: Request):
        if not authorized(request):
            return JSONResponse(
                error_envelope(
                    "invalid bearer credential",
                    error_type="authentication_error",
                    code="invalid_api_key",
                ),
                status_code=401,
            )
        batch_telemetry = getattr(host.generation_service.telemetry(), "compatible_batch", None)
        batch_identity = None if batch_telemetry is None else batch_telemetry.identity
        session_telemetry = None if host.session_store is None else host.session_store.telemetry()
        sampling_controls = ["greedy"]
        if host.sampling_enabled:
            sampling_controls.extend(
                [
                    "temperature",
                    "top_p",
                    "top_k",
                    "frequency_penalty",
                    "presence_penalty",
                    "logit_bias",
                    "seed",
                ]
            )
        return {
            "object": "list",
            "data": [
                {
                    "id": host.model_id,
                    "object": "model",
                    "owned_by": "mrun",
                    "mrun": {
                        "context_size": host.tokenizer.context_size,
                        "semantic_token_count": host.tokenizer.semantic_token_count,
                        "route_id": host.route_id,
                        "promotion_status": host.promotion_status.value,
                        "output_modes": [mode.value for mode in host.supported_output_modes],
                        "sampling": sampling_controls,
                        "sampling_abi": NATIVE_SAMPLING_ABI if host.sampling_enabled else None,
                        "streaming": True,
                        "sessions": host.sessions_enabled,
                        "session_store": (
                            None
                            if session_telemetry is None
                            else {
                                "identity_fingerprint": session_telemetry.identity_fingerprint,
                                "state_abi": host.session_store.identity.state_abi,
                                "entries": session_telemetry.entries,
                                "retired_entries": session_telemetry.retired_entries,
                                "active_leases": session_telemetry.active_leases,
                                "pinned_sources": session_telemetry.pinned_sources,
                                "stored_bytes": session_telemetry.stored_bytes,
                                "reserved_bytes": session_telemetry.reserved_bytes,
                                "reserved_slots": session_telemetry.reserved_slots,
                                "max_entries": session_telemetry.max_entries,
                                "max_bytes": session_telemetry.max_bytes,
                                "budget_reconciled": session_telemetry.budget_reconciled,
                                "accepting": session_telemetry.accepting,
                                "poisoned": session_telemetry.poisoned,
                                "cross_session_prefix": {
                                    "hits": session_telemetry.cross_session_prefix_hits,
                                    "tokens": session_telemetry.cross_session_prefix_tokens,
                                    "bytes": session_telemetry.cross_session_prefix_bytes,
                                },
                            }
                        ),
                        "compatible_batch": (
                            None
                            if batch_identity is None
                            else {
                                "lane_abi": batch_identity.lane_abi,
                                "numerical_contract": batch_identity.numerical_contract,
                                "promotion_status": batch_identity.promotion_status.value,
                                "max_batch_size": batch_identity.max_batch_size,
                                "max_queue_delay_seconds": (batch_identity.max_queue_delay_seconds),
                                "max_scratch_bytes": batch_identity.max_scratch_bytes,
                                "ragged_dispatch": batch_identity.supports_ragged_dispatch,
                            }
                        ),
                        "rate_limit": (
                            None
                            if host.config.rate_limit_tokens_per_minute is None
                            else {
                                "tokens_per_minute": host.config.rate_limit_tokens_per_minute,
                                "burst_tokens": host.config.rate_limit_burst_tokens,
                                "identity": "direct_peer",
                            }
                        ),
                    },
                }
            ],
        }

    @app.get("/metrics")
    async def metrics(request: Request):
        if not host.config.expose_metrics:
            return JSONResponse({"detail": "not found"}, status_code=404)
        if not authorized(request):
            return JSONResponse(
                error_envelope(
                    "invalid bearer credential",
                    error_type="authentication_error",
                    code="invalid_api_key",
                ),
                status_code=401,
            )
        return PlainTextResponse(host.metrics_text(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if not authorized(request):
            return JSONResponse(
                error_envelope(
                    "invalid bearer credential",
                    error_type="authentication_error",
                    code="invalid_api_key",
                ),
                status_code=401,
            )
        lifecycle_started = time.monotonic()
        request_id = f"http.{uuid4().hex}"
        parsed: OpenAIChatRequest | None = None
        prompt_ids: tuple[int, ...] = ()
        acquired = False
        try:
            host.acquire_request()
            acquired = True
            body = await _read_limited_body(request, host.config.max_request_bytes)
            payload = strict_json_object(body)
            parsed, prompt_ids, generation, request_id, created = host.prepare(payload)
            token_cost = len(prompt_ids) + parsed.max_tokens
            host.charge_token_budget(_direct_client_identifier(request), token_cost)
            handle = host.generation_service.submit(generation)
            host.log_lifecycle(
                "admitted",
                request_id=request_id,
                model_id=host.model_id,
                route_id=host.route_id,
                stream=parsed.stream,
                status="admitted",
                prompt_token_count=len(prompt_ids),
                requested_completion_tokens=parsed.max_tokens,
                session_requested=parsed.session_id is not None,
                rate_limit_token_cost=token_cost,
            )
        except asyncio.CancelledError:
            if acquired:
                host.finish_request(failed=True)
            host.record_disconnect()
            host.log_lifecycle(
                "disconnected",
                request_id=request_id,
                model_id=host.model_id,
                route_id=host.route_id,
                stream=None if parsed is None else parsed.stream,
                status="cancelled",
                error_code="client_closed_request",
                prompt_token_count=len(prompt_ids) if prompt_ids else None,
                requested_completion_tokens=(None if parsed is None else parsed.max_tokens),
                latency_ms=max(0, int((time.monotonic() - lifecycle_started) * 1000)),
                session_requested=(None if parsed is None else parsed.session_id is not None),
            )
            raise
        except Exception as exc:
            if acquired:
                host.finish_request(failed=True)
            _status, _error_type, code, _message = _http_error(exc)
            host.log_lifecycle(
                "rejected",
                request_id=request_id,
                model_id=host.model_id,
                route_id=host.route_id,
                stream=None if parsed is None else parsed.stream,
                status="rejected",
                error_code=code,
                prompt_token_count=len(prompt_ids) if prompt_ids else None,
                requested_completion_tokens=(None if parsed is None else parsed.max_tokens),
                latency_ms=max(0, int((time.monotonic() - lifecycle_started) * 1000)),
                session_requested=(None if parsed is None else parsed.session_id is not None),
            )
            return json_error(exc)

        assert parsed is not None
        completion_id = "chatcmpl_" + request_id.removeprefix("chat.")
        decoder = IncrementalTextDecoder(
            host.tokenizer,
            stop_strings=parsed.stop_strings,
            include_stop=False,
        )

        if not parsed.stream:
            try:
                result = await _await_generation_result(
                    request,
                    handle,
                    request_id=request_id,
                    poll_seconds=host.config.event_poll_seconds,
                )
                for token_id in result.token_ids:
                    delta = decoder.push(token_id)
                    if delta.stopped:
                        break
                decoder.finish()
                if result.state_retained and result.session_id is None:
                    handle.release_retained_state()
                response = completion_response(
                    completion_id=completion_id,
                    created=created,
                    model=parsed.model,
                    text=decoder.emitted_text,
                    finish_reason=(
                        "stop" if decoder.stopped else _finish_reason(result.finish_reason)
                    ),
                    usage=Usage(
                        prompt_tokens=len(prompt_ids),
                        completion_tokens=len(result.token_ids),
                    ),
                    request_id=request_id,
                    route_id=host.route_id,
                    session_cache=result.session_cache_status.value,
                )
                host.finish_request(failed=False)
                host.log_lifecycle(
                    "completed",
                    request_id=request_id,
                    model_id=host.model_id,
                    route_id=host.route_id,
                    stream=False,
                    status="completed",
                    prompt_token_count=len(prompt_ids),
                    requested_completion_tokens=parsed.max_tokens,
                    completion_token_count=len(result.token_ids),
                    latency_ms=max(0, int((time.monotonic() - lifecycle_started) * 1000)),
                    session_requested=parsed.session_id is not None,
                    session_cache=result.session_cache_status.value,
                )
                return JSONResponse(response, headers={"Cache-Control": "no-store"})
            except asyncio.CancelledError:
                handle.cancel("http_task_cancelled")
                host.record_disconnect()
                host.finish_request(failed=True)
                host.log_lifecycle(
                    "disconnected",
                    request_id=request_id,
                    model_id=host.model_id,
                    route_id=host.route_id,
                    stream=False,
                    status="cancelled",
                    error_code="client_closed_request",
                    prompt_token_count=len(prompt_ids),
                    requested_completion_tokens=parsed.max_tokens,
                    latency_ms=max(0, int((time.monotonic() - lifecycle_started) * 1000)),
                    session_requested=parsed.session_id is not None,
                )
                raise
            except Exception as exc:
                if isinstance(exc, GenerationCancelled) and exc.reason == "client_disconnected":
                    host.record_disconnect()
                host.finish_request(failed=True)
                _status, _error_type, code, _message = _http_error(exc)
                host.log_lifecycle(
                    (
                        "disconnected"
                        if isinstance(exc, GenerationCancelled)
                        and exc.reason == "client_disconnected"
                        else "failed"
                    ),
                    request_id=request_id,
                    model_id=host.model_id,
                    route_id=host.route_id,
                    stream=False,
                    status="failed",
                    error_code=code,
                    prompt_token_count=len(prompt_ids),
                    requested_completion_tokens=parsed.max_tokens,
                    latency_ms=max(0, int((time.monotonic() - lifecycle_started) * 1000)),
                    session_requested=parsed.session_id is not None,
                )
                return json_error(exc)

        async def stream_events() -> AsyncIterator[bytes]:
            failed = True
            released = False
            seen_tokens = 0
            terminal_event = "failed"
            terminal_status = "failed"
            terminal_code = "stream_incomplete"
            session_cache: str | None = None
            disconnect_recorded = False
            try:
                yield _sse(
                    completion_chunk(
                        completion_id=completion_id,
                        created=created,
                        model=parsed.model,
                        delta={"role": "assistant"},
                        finish_reason=None,
                    )
                )
                while True:
                    if await request.is_disconnected():
                        handle.cancel("client_disconnected")
                        host.record_disconnect()
                        disconnect_recorded = True
                        terminal_event = "disconnected"
                        terminal_status = "cancelled"
                        terminal_code = "client_closed_request"
                        return
                    try:
                        event = await asyncio.to_thread(
                            handle.next_event,
                            host.config.event_poll_seconds,
                        )
                    except GenerationEventTimeout:
                        continue
                    if isinstance(event, TokenEvent):
                        seen_tokens += 1
                        text = decoder.push(event.token_id)
                        if text.text:
                            yield _sse(
                                completion_chunk(
                                    completion_id=completion_id,
                                    created=created,
                                    model=parsed.model,
                                    delta={"content": text.text},
                                    finish_reason=None,
                                )
                            )
                        if text.stopped:
                            handle.cancel("text_stop_matched")
                            usage = Usage(
                                prompt_tokens=len(prompt_ids),
                                completion_tokens=seen_tokens,
                            )
                            yield _sse(
                                completion_chunk(
                                    completion_id=completion_id,
                                    created=created,
                                    model=parsed.model,
                                    delta={},
                                    finish_reason="stop",
                                    usage=(usage if parsed.stream_options.include_usage else None),
                                )
                            )
                            yield _sse("[DONE]")
                            failed = False
                            terminal_event = "completed"
                            terminal_status = "completed"
                            terminal_code = "stop"
                            return
                    elif isinstance(event, CompletedEvent):
                        tail = decoder.finish()
                        if tail.text:
                            yield _sse(
                                completion_chunk(
                                    completion_id=completion_id,
                                    created=created,
                                    model=parsed.model,
                                    delta={"content": tail.text},
                                    finish_reason=None,
                                )
                            )
                        result = event.result
                        session_cache = result.session_cache_status.value
                        if result.state_retained and result.session_id is None:
                            handle.release_retained_state()
                            released = True
                        usage = Usage(
                            prompt_tokens=len(prompt_ids),
                            completion_tokens=len(result.token_ids),
                        )
                        yield _sse(
                            completion_chunk(
                                completion_id=completion_id,
                                created=created,
                                model=parsed.model,
                                delta={},
                                finish_reason=_finish_reason(result.finish_reason),
                                usage=usage if parsed.stream_options.include_usage else None,
                            )
                        )
                        yield _sse("[DONE]")
                        failed = False
                        terminal_event = "completed"
                        terminal_status = "completed"
                        terminal_code = result.finish_reason.value
                        return
                    elif isinstance(event, TerminalEvent):
                        if event.status is TerminalStatus.DEADLINE_EXCEEDED:
                            error_type = "deadline_exceeded"
                            code = "deadline_exceeded"
                            message = "generation deadline exceeded"
                        elif event.status is TerminalStatus.BACKPRESSURE:
                            error_type = "rate_limit_error"
                            code = "consumer_backpressure"
                            message = "stream consumer could not keep up"
                        elif event.status in {
                            TerminalStatus.CANCELLED,
                            TerminalStatus.SHUTDOWN,
                        }:
                            error_type = "service_unavailable"
                            code = event.status.value
                            message = "generation was cancelled"
                        else:
                            error_type = "server_error"
                            code = f"incident.{uuid4().hex}"
                            message = "internal inference failure"
                        terminal_event = (
                            "cancelled"
                            if event.status in {TerminalStatus.CANCELLED, TerminalStatus.SHUTDOWN}
                            else "failed"
                        )
                        terminal_status = event.status.value
                        terminal_code = code
                        yield _sse(
                            error_envelope(
                                message,
                                error_type=error_type,
                                code=code,
                            )
                        )
                        yield _sse("[DONE]")
                        return
                    else:  # pragma: no cover - union is exhaustive; retain fail-closed behavior
                        raise RuntimeError("generation service returned an unknown event")
            except (asyncio.CancelledError, GeneratorExit):
                handle.cancel("stream_cancelled")
                if not disconnect_recorded:
                    host.record_disconnect()
                    disconnect_recorded = True
                terminal_event = "disconnected"
                terminal_status = "cancelled"
                terminal_code = "client_closed_request"
                raise
            except Exception as exc:
                handle.cancel("stream_failure")
                _status, error_type, code, message = _http_error(exc)
                terminal_event = "failed"
                terminal_status = "failed"
                terminal_code = code
                yield _sse(error_envelope(message, error_type=error_type, code=code))
                yield _sse("[DONE]")
            finally:
                if not released and handle.done():
                    try:
                        result = handle.result()
                        if result.state_retained and result.session_id is None:
                            handle.release_retained_state()
                    except (GenerationCancelled, GenerationServiceError, TimeoutError):
                        pass
                host.log_lifecycle(
                    terminal_event,
                    request_id=request_id,
                    model_id=host.model_id,
                    route_id=host.route_id,
                    stream=True,
                    status=terminal_status,
                    error_code=terminal_code,
                    prompt_token_count=len(prompt_ids),
                    requested_completion_tokens=parsed.max_tokens,
                    completion_token_count=seen_tokens,
                    latency_ms=max(0, int((time.monotonic() - lifecycle_started) * 1000)),
                    session_requested=parsed.session_id is not None,
                    session_cache=session_cache,
                )
                host.finish_request(failed=failed)

        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )

    return app


__all__ = [
    "InferenceAppConfig",
    "InferenceHost",
    "create_inference_app",
    "strict_json_object",
]
