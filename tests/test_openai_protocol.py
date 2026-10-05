from __future__ import annotations

import pytest

from mrun.inference import (
    ChatValidationError,
    Usage,
    completion_chunk,
    completion_response,
    error_envelope,
    parse_chat_completion_request,
)


def _payload(**updates):
    payload = {
        "model": "toy-chat",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(updates)
    return payload


def test_request_parser_accepts_supported_sampling_stream_and_session_subset() -> None:
    request = parse_chat_completion_request(
        _payload(
            stream=True,
            max_tokens=42,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            seed=123,
            stop=["END", "User:"],
            frequency_penalty=0.25,
            presence_penalty=-0.5,
            logit_bias={"3": -2, 4: 1.5},
            stream_options={"include_usage": True},
            session_id="conversation-1",
        ),
        loaded_model="toy-chat",
        semantic_token_count=10,
    )
    assert request.stream
    assert request.max_tokens == 42
    assert request.sampling.temperature == 0.7
    assert request.sampling.logit_bias == ((3, -2.0), (4, 1.5))
    assert request.stop_strings == ("END", "User:")
    assert request.stream_options.include_usage
    assert request.session_id == "conversation-1"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"unknown": 1}, "unsupported request fields"),
        ({"model": "other"}, "not loaded"),
        ({"stream": 1}, "stream must be boolean"),
        ({"max_tokens": True}, "max_tokens must be an integer"),
        ({"n": 2}, "exactly one"),
        ({"temperature": float("nan")}, "finite"),
        ({"temperature": 3}, "temperature must be in"),
        ({"seed": 1 << 63}, "signed 64-bit"),
        ({"top_p": 0}, "top_p must be in"),
        ({"top_k": -1}, "at least 0"),
        ({"stop": []}, "one to four"),
        ({"stop": ["x", "x"]}, "must be unique"),
        ({"logit_bias": {"10": 1}}, "semantic domain"),
        ({"logit_bias": {"03": 1, "3": 2}}, "duplicate normalized"),
        ({"stream_options": {"other": True}}, "supports only"),
        ({"session_id": " bad"}, "canonical string"),
    ],
)
def test_request_parser_fails_closed(updates, message: str) -> None:
    with pytest.raises(ChatValidationError, match=message):
        parse_chat_completion_request(
            _payload(**updates),
            loaded_model="toy-chat",
            semantic_token_count=10,
        )


def test_model_not_found_retains_public_error_code() -> None:
    with pytest.raises(ChatValidationError) as error:
        parse_chat_completion_request(
            _payload(model="other"),
            loaded_model="toy-chat",
            semantic_token_count=10,
        )
    assert error.value.code == "model_not_found"


def test_openai_response_and_stream_chunk_shapes() -> None:
    usage = Usage(prompt_tokens=5, completion_tokens=2)
    response = completion_response(
        completion_id="chatcmpl_1",
        created=123,
        model="toy-chat",
        text="Hello",
        finish_reason="stop",
        usage=usage,
        request_id="request-1",
        route_id="route-1",
        session_cache="miss",
    )
    assert response["object"] == "chat.completion"
    assert response["usage"]["total_tokens"] == 7
    assert response["choices"][0]["message"]["content"] == "Hello"

    chunk = completion_chunk(
        completion_id="chatcmpl_1",
        created=123,
        model="toy-chat",
        delta={"content": "He"},
        finish_reason=None,
    )
    assert chunk["object"] == "chat.completion.chunk"
    assert "usage" not in chunk
    final = completion_chunk(
        completion_id="chatcmpl_1",
        created=123,
        model="toy-chat",
        delta={},
        finish_reason="stop",
        usage=usage,
    )
    assert final["usage"]["total_tokens"] == 7
    assert error_envelope("bad", error_type="invalid_request_error", code="bad")["error"] == {
        "message": "bad",
        "type": "invalid_request_error",
        "param": None,
        "code": "bad",
    }
