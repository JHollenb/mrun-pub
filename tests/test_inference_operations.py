from __future__ import annotations

import asyncio
import json
import logging
from io import StringIO

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mrun.inference import (
    BoundChatTokenizer,
    InferenceAppConfig,
    InferenceHost,
    canonical_chat_template_sha256,
    create_inference_app,
)
from mrun.inference.app import _await_generation_result
from mrun.inference.operations import (
    InferenceRateLimitError,
    PerIpTokenLimiter,
    PrivacySafeLifecycleLogger,
)
from mrun.runtime import OutputMode, PromotionStatus
from mrun.runtime.inference import (
    CompletedEvent,
    FinishReason,
    GenerationCancelled,
    GenerationResult,
    TokenEvent,
)


class _Tokenizer:
    chat_template = "{{ messages }}<assistant>"
    eos_token_id = 9

    def get_chat_template(self, chat_template=None):
        return self.chat_template

    def apply_chat_template(self, messages, **_kwargs):
        return [1, 2, len(messages)]

    def encode(self, text: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        return [8] if text else []

    def decode(self, ids, **_kwargs):
        return "".join({3: "Hel", 4: "lo"}.get(value, "?") for value in ids)


def _result(request_id: str) -> GenerationResult:
    return GenerationResult(
        request_id=request_id,
        token_ids=(3, 4),
        finish_reason=FinishReason.MAX_NEW_TOKENS,
        matched_stop_sequence=None,
        prompt_token_count=3,
        model_generated_token_count=2,
        committed_input_token_count=4,
        final_state_length=4,
        final_state_epoch=2,
        pending_token_id=4,
        step_count=2,
        state_capacity=4,
        ttft_seconds=0.01,
        inter_token_seconds=(0.01,),
        request_latency_seconds=0.02,
        runtime_id="runtime.test",
        backend_id="fake-native",
    )


class _Handle:
    def __init__(self, request_id: str, *, stream: bool) -> None:
        self._result = _result(request_id)
        self._done = not stream
        self.cancelled = False
        self._events = [
            TokenEvent(request_id, 3, 0, 0, 1.0, 1.0, 1, 3),
            TokenEvent(request_id, 4, 1, 1, 1.1, 1.1, 2, 4),
            CompletedEvent(request_id, self._result, 1.2),
        ]

    def result(self):
        self._done = True
        return self._result

    def next_event(self, _timeout=None):
        event = self._events.pop(0)
        if isinstance(event, CompletedEvent):
            self._done = True
        return event

    def cancel(self, _reason="caller"):
        self.cancelled = True
        self._done = True
        return True

    def done(self):
        return self._done

    def release_retained_state(self):
        return False


class _Telemetry:
    active = 0
    admitted = 0
    completed = 0
    cancelled = 0
    failed = 0
    model_generated_tokens = 0
    streamed_tokens = 0


class _Service:
    session_store = None
    supported_output_modes = (
        OutputMode.NEXT_TOKEN_ARGMAX,
        OutputMode.NEXT_TOKEN_SAMPLE,
    )

    def __init__(self) -> None:
        self.requests = []

    def submit(self, request):
        self.requests.append(request)
        return _Handle(request.request_id, stream=request.stream)

    def telemetry(self):
        return _Telemetry()


def _host(
    *,
    config: InferenceAppConfig | None = None,
    logger: logging.Logger | None = None,
) -> InferenceHost:
    tokenizer = _Tokenizer()
    return InferenceHost(
        tokenizer=BoundChatTokenizer(
            tokenizer,
            model_id="toy-chat",
            semantic_token_count=10,
            context_size=64,
            expected_chat_template_sha256=canonical_chat_template_sha256(tokenizer.chat_template),
        ),
        generation_service=_Service(),
        route_id="route.test",
        promotion_status=PromotionStatus.CANDIDATE,
        config=config,
        lifecycle_logger=logger,
    )


def _request(content: str = "private-user-text") -> dict[str, object]:
    return {
        "model": "toy-chat",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2,
    }


def test_browser_chat_is_self_contained_and_keeps_bearer_auth_at_api_boundary() -> None:
    app = create_inference_app(_host(), bearer_token="server-secret-never-rendered")
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert client.get("/ui").status_code == 200
        assert client.get("/v1/models").status_code == 401

    assert "server-secret-never-rendered" not in page.text
    assert "https://" not in page.text
    assert "AbortController" in page.text
    assert 'id="regenerate"' in page.text
    assert 'id="temperature"' in page.text
    assert 'id="top-p"' in page.text
    assert 'id="top-k"' in page.text
    assert 'fetch("/v1/chat/completions"' in page.text
    csp = page.headers["content-security-policy"]
    nonce = csp.split("script-src 'nonce-", maxsplit=1)[1].split("'", maxsplit=1)[0]
    assert f'nonce="{nonce}"' in page.text
    assert "unsafe-inline" not in csp
    assert page.headers["cache-control"] == "no-store"


def test_direct_peer_token_limiter_refills_isolates_clients_and_bounds_cardinality() -> None:
    now = [0.0]
    limiter = PerIpTokenLimiter(
        tokens_per_minute=60,
        burst_tokens=5,
        max_clients=2,
        idle_seconds=10,
        clock=lambda: now[0],
    )
    limiter.charge("192.0.2.1", 5)
    with pytest.raises(InferenceRateLimitError) as depleted:
        limiter.charge("192.0.2.1", 1)
    assert depleted.value.code == "token_rate_limit_exceeded"
    assert depleted.value.retry_after_seconds == 1
    limiter.charge("192.0.2.2", 5)
    with pytest.raises(InferenceRateLimitError) as full:
        limiter.charge("192.0.2.3", 1)
    assert full.value.code == "rate_limit_client_capacity"

    now[0] = 11.0
    limiter.charge("192.0.2.3", 1)
    limiter.charge("192.0.2.1", 1)
    snapshot = limiter.snapshot()
    assert snapshot.clients == 2
    assert snapshot.allowed_requests == 4
    assert snapshot.rejected_requests == 2
    assert snapshot.charged_tokens == 12
    with pytest.raises(InferenceRateLimitError, match="burst") as oversized:
        limiter.charge("192.0.2.1", 6)
    assert oversized.value.code == "token_budget_exceeds_burst"


def test_chat_rate_limit_ignores_forwarding_header_and_exports_prometheus_metadata() -> None:
    host = _host(
        config=InferenceAppConfig(
            rate_limit_tokens_per_minute=60,
            rate_limit_burst_tokens=5,
        )
    )
    with TestClient(create_inference_app(host)) as client:
        first = client.post(
            "/v1/chat/completions",
            json=_request(),
            headers={"x-forwarded-for": "192.0.2.1"},
        )
        second = client.post(
            "/v1/chat/completions",
            json=_request(),
            headers={"x-forwarded-for": "198.51.100.2"},
        )
        metrics = client.get("/metrics")

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "token_rate_limit_exceeded"
    assert int(second.headers["retry-after"]) >= 1
    assert metrics.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "# HELP mrun_inference_ready" in metrics.text
    assert "# TYPE mrun_inference_ready gauge" in metrics.text
    assert "# TYPE mrun_inference_rate_limit_rejected_total counter" in metrics.text
    assert "mrun_inference_rate_limit_rejected_total 1" in metrics.text


def test_lifecycle_log_schema_contains_counts_but_never_request_content_or_token_ids() -> None:
    output = StringIO()
    logger = logging.Logger("mrun-test-lifecycle", level=logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(output))
    secret_content = "PRIVATE-CONTENT-5f58dca2"
    host = _host(logger=logger)

    with TestClient(create_inference_app(host)) as client:
        response = client.post("/v1/chat/completions", json=_request(secret_content))
    assert response.status_code == 200

    raw = output.getvalue()
    assert secret_content not in raw
    assert "messages" not in raw
    assert "content" not in raw
    assert "token_ids" not in raw
    assert "authorization" not in raw
    records = [json.loads(line) for line in raw.splitlines()]
    assert [record["event"] for record in records] == ["admitted", "completed"]
    assert records[0]["prompt_token_count"] == 3
    assert records[1]["completion_token_count"] == 2


def test_streaming_lifecycle_has_one_admission_and_one_content_free_terminal_record() -> None:
    output = StringIO()
    logger = logging.Logger("mrun-test-stream-lifecycle", level=logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(output))
    host = _host(logger=logger)

    with TestClient(create_inference_app(host)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_request("STREAM-PRIVATE-CONTENT") | {"stream": True},
        )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["event"] for record in records] == ["admitted", "completed"]
    assert records[-1]["stream"] is True
    assert records[-1]["completion_token_count"] == 2
    assert "STREAM-PRIVATE-CONTENT" not in output.getvalue()


def test_privacy_logger_rejects_every_field_outside_closed_schema() -> None:
    logger = PrivacySafeLifecycleLogger(logging.Logger("closed-schema"), wall_clock=lambda: 1.0)
    with pytest.raises(ValueError, match="rejected fields"):
        logger.emit("admitted", request_id="safe", content="must-not-log")


def test_nonstreaming_disconnect_cancels_generation_before_waiting_for_result() -> None:
    class Request:
        async def is_disconnected(self) -> bool:
            return True

    class Handle:
        cancelled = False
        result_called = False

        def done(self) -> bool:
            return False

        def cancel(self, reason: str) -> bool:
            assert reason == "client_disconnected"
            self.cancelled = True
            return True

        def result(self):
            self.result_called = True
            raise AssertionError("disconnected request must not wait for a result")

    handle = Handle()
    with pytest.raises(GenerationCancelled, match="client_disconnected"):
        asyncio.run(
            _await_generation_result(
                Request(),
                handle,
                request_id="chat.test",
                poll_seconds=0.001,
            )
        )
    assert handle.cancelled
    assert not handle.result_called
