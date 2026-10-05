from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mrun.inference import (
    BoundChatTokenizer,
    InferenceAppConfig,
    InferenceHost,
    canonical_chat_template_sha256,
    create_inference_app,
    strict_json_object,
)
from mrun.runtime import OutputMode, PromotionStatus
from mrun.runtime.inference import (
    CompletedEvent,
    FinishReason,
    GenerationResult,
    SessionIdentity,
    SessionIdentityError,
    SessionStoreTelemetry,
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
        return "".join({3: "Hel", 4: "lo"}.get(value, f"<{value}>") for value in ids)


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
        ttft_seconds=0.1,
        inter_token_seconds=(0.1,),
        request_latency_seconds=0.2,
        runtime_id="runtime.test",
        backend_id="mlx-component",
    )


class _Handle:
    def __init__(self, request_id: str, *, stream: bool) -> None:
        self._result = _result(request_id)
        self._done = not stream
        self.cancelled = False
        self.events = deque(
            [
                TokenEvent(request_id, 3, 0, 0, 1.0, 1.1, 1, 3),
                TokenEvent(request_id, 4, 1, 1, 1.2, 1.3, 2, 4),
                CompletedEvent(request_id, self._result, 1.4),
            ]
            if stream
            else []
        )

    def result(self):
        self._done = True
        return self._result

    def next_event(self, _timeout=None):
        event = self.events.popleft()
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


@dataclass
class _Telemetry:
    active: int = 0
    admitted: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    model_generated_tokens: int = 0
    streamed_tokens: int = 0


class _Service:
    def __init__(
        self,
        session_store: Any | None = None,
        supported_output_modes: tuple[OutputMode, ...] = (
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ),
    ) -> None:
        self.requests = []
        self.session_store = session_store
        self.supported_output_modes = supported_output_modes

    def submit(self, request):
        self.requests.append(request)
        return _Handle(request.request_id, stream=request.stream)

    def telemetry(self):
        return _Telemetry(admitted=len(self.requests), completed=len(self.requests))


def _host(
    *,
    session_store: Any | None = None,
    supported_output_modes: tuple[OutputMode, ...] = (
        OutputMode.NEXT_TOKEN_ARGMAX,
        OutputMode.NEXT_TOKEN_SAMPLE,
    ),
    promotion_status: PromotionStatus = PromotionStatus.CANDIDATE,
    **config_updates: Any,
) -> tuple[InferenceHost, _Service]:
    tokenizer = _Tokenizer()
    bound = BoundChatTokenizer(
        tokenizer,
        model_id="toy-chat",
        semantic_token_count=10,
        context_size=64,
        expected_chat_template_sha256=canonical_chat_template_sha256(tokenizer.chat_template),
    )
    service = _Service(session_store, supported_output_modes)
    host = InferenceHost(
        tokenizer=bound,
        generation_service=service,
        route_id="route.test",
        promotion_status=promotion_status,
        config=InferenceAppConfig(**config_updates),
    )
    return host, service


class _SessionStore:
    def __init__(
        self,
        identity: SessionIdentity,
        telemetry: SessionStoreTelemetry | None = None,
    ) -> None:
        self.identity = identity
        self._telemetry = telemetry or _session_telemetry(identity)

    def telemetry(self) -> SessionStoreTelemetry:
        return self._telemetry


def _session_identity(*, model_id: str = "toy-chat") -> SessionIdentity:
    return SessionIdentity(
        model_id=model_id,
        model_fingerprint="a" * 64,
        chat_template_sha256=canonical_chat_template_sha256(_Tokenizer.chat_template),
        semantic_token_count=10,
        route_id="route.test",
        runtime_id="runtime.test",
        capability_fingerprint="b" * 64,
        placement_fingerprint="c" * 64,
        backend_id="fake-native",
        device_id="fake:0",
        state_abi="fake-kv-v1",
    )


def _session_telemetry(identity: SessionIdentity) -> SessionStoreTelemetry:
    return SessionStoreTelemetry(
        identity_fingerprint=identity.fingerprint,
        entries=2,
        active_leases=1,
        stored_bytes=320,
        reserved_bytes=64,
        reserved_slots=1,
        retired_entries=1,
        pinned_sources=1,
        max_entries=8,
        max_bytes=1024,
        hits=5,
        misses=7,
        installs=9,
        aborts=1,
        busy_rejections=2,
        identity_rejections=3,
        prefix_rejections=4,
        capacity_rejections=1,
        integrity_rejections=0,
        forks=5,
        fork_tokens=15,
        fork_bytes=120,
        cross_session_prefix_hits=2,
        cross_session_prefix_tokens=6,
        cross_session_prefix_bytes=48,
        ttl_evictions=1,
        lru_evictions=2,
        replacements=3,
        manual_evictions=4,
        cleanup_failures=0,
        accepting=True,
        poisoned=False,
    )


def _request(**updates):
    payload = {
        "model": "toy-chat",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 2,
    }
    payload.update(updates)
    return payload


def test_nonstreaming_chat_runs_exact_template_generation_and_response() -> None:
    host, service = _host()
    with TestClient(create_inference_app(host)) as client:
        response = client.post("/v1/chat/completions", json=_request())
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert service.requests[0].input_ids == (1, 2, 1)
    assert not service.requests[0].stream


def test_streaming_chat_emits_role_committed_deltas_usage_and_done() -> None:
    host, service = _host()
    with TestClient(create_inference_app(host)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_request(stream=True, stream_options={"include_usage": True}),
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"role":"assistant"' in response.text
    assert '"content":"Hel"' in response.text
    assert '"content":"lo"' in response.text
    assert '"total_tokens":5' in response.text
    assert "data: [DONE]" in response.text
    assert service.requests[0].stream


def test_auth_readiness_models_metrics_and_drain_are_separate_from_scheduler() -> None:
    host, _service = _host()
    app = create_inference_app(host, bearer_token="secret")
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/models").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        models = client.get("/v1/models", headers=headers)
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "toy-chat"
        assert models.json()["data"][0]["mrun"]["promotion_status"] == "candidate"
        metrics = client.get("/metrics", headers=headers)
        assert "mrun_inference_ready 1" in metrics.text
        host.begin_drain()
        assert client.get("/readyz").status_code == 503
        assert (
            client.post("/v1/chat/completions", headers=headers, json=_request()).status_code == 503
        )


def test_models_advertises_argmax_only_route_without_sampling_controls() -> None:
    host, _service = _host(
        supported_output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,),
        promotion_status=PromotionStatus.EXPERIMENTAL,
    )
    with TestClient(create_inference_app(host)) as client:
        model = client.get("/v1/models").json()["data"][0]["mrun"]

    assert model["output_modes"] == ["next-token-argmax"]
    assert model["promotion_status"] == "experimental"
    assert model["sampling"] == ["greedy"]
    assert model["sampling_abi"] is None


def test_host_rejects_untyped_promotion_status() -> None:
    with pytest.raises(TypeError, match="promotion_status must be PromotionStatus"):
        _host(promotion_status="candidate")  # type: ignore[arg-type]


def test_content_prefix_session_telemetry_reconciles_models_and_prometheus() -> None:
    identity = _session_identity()
    sessions = _session_telemetry(identity)
    assert sessions.budget_reconciled
    host, _service = _host(session_store=_SessionStore(identity, sessions))
    with TestClient(create_inference_app(host)) as client:
        model = client.get("/v1/models").json()["data"][0]["mrun"]
        assert model["sessions"] is True
        assert model["session_store"] == {
            "identity_fingerprint": identity.fingerprint,
            "state_abi": "fake-kv-v1",
            "entries": 2,
            "retired_entries": 1,
            "active_leases": 1,
            "pinned_sources": 1,
            "stored_bytes": 320,
            "reserved_bytes": 64,
            "reserved_slots": 1,
            "max_entries": 8,
            "max_bytes": 1024,
            "budget_reconciled": True,
            "accepting": True,
            "poisoned": False,
            "cross_session_prefix": {"hits": 2, "tokens": 6, "bytes": 48},
        }

        metrics = client.get("/metrics").text
        for sample in (
            "mrun_session_entries 2",
            "mrun_session_retired_entries 1",
            "mrun_session_reserved_slots 1",
            "mrun_session_pinned_sources 1",
            "mrun_session_cross_prefix_hits_total 2",
            "mrun_session_cross_prefix_tokens_total 6",
            "mrun_session_cross_prefix_bytes_total 48",
            "mrun_session_budget_reconciled 1",
        ):
            assert sample in metrics


def test_request_boundary_accepts_sampling_and_rejects_context_body_and_duplicate_json() -> None:
    host, service = _host(max_request_bytes=256)
    with TestClient(create_inference_app(host)) as client:
        sampling = client.post(
            "/v1/chat/completions",
            json=_request(temperature=0.5),
        )
        assert sampling.status_code == 200
        assert service.requests[-1].sampling is not None
        assert service.requests[-1].sampling.temperature == 0.5
        context = client.post(
            "/v1/chat/completions",
            json=_request(max_tokens=63),
        )
        assert context.status_code == 400
        assert context.json()["error"]["code"] == "context_length_exceeded"
        oversized = client.post(
            "/v1/chat/completions",
            content=b"{" + b" " * 300 + b"}",
            headers={"content-type": "application/json"},
        )
        assert oversized.status_code == 400

    with pytest.raises(Exception, match="duplicate JSON key"):
        strict_json_object(b'{"model":"a","model":"b"}')


def test_host_resolves_complete_sampling_policy_and_concrete_request_seed() -> None:
    host, _service = _host()
    _parsed, _prompt, generation, _request_id, _created = host.prepare(
        _request(
            temperature=0.75,
            top_p=0.91,
            top_k=7,
            seed=-123,
            frequency_penalty=0.4,
            presence_penalty=-0.2,
            logit_bias={"3": 1.5},
        )
    )
    assert generation.sampling is not None
    assert generation.sampling.seed == -123
    assert generation.sampling.temperature == 0.75
    assert generation.sampling.top_p == 0.91
    assert generation.sampling.top_k == 7
    assert generation.sampling.frequency_penalty == 0.4
    assert generation.sampling.presence_penalty == -0.2
    assert generation.sampling.logit_bias == ((3, 1.5),)

    _parsed, _prompt, generated_seed, _request_id, _created = host.prepare(
        _request(temperature=1.0)
    )
    assert generated_seed.sampling is not None
    assert 0 <= generated_seed.sampling.seed < 1 << 63


def test_host_sessions_are_explicitly_bound_or_rejected_before_generation() -> None:
    disabled, disabled_service = _host()
    with TestClient(create_inference_app(disabled)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_request(session_id="thread-1"),
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "sessions_unavailable"
    assert disabled_service.requests == []

    enabled, _service = _host(session_store=_SessionStore(_session_identity()))
    parsed, _prompt, generation, _request_id, _created = enabled.prepare(
        _request(session_id="thread-1")
    )
    assert parsed.session_id == generation.session_id == "thread-1"
    assert generation.retain_state_on_success
    assert enabled.sessions_enabled

    with pytest.raises(SessionIdentityError, match="model/template/domain/route"):
        _host(session_store=_SessionStore(_session_identity(model_id="wrong-model")))
