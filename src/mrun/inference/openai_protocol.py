"""Strict OpenAI-compatible chat request/response subset with no server dependency."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .chat import ChatMessage, ChatValidationError, validate_chat_messages


def _strict_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChatValidationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ChatValidationError(f"{field} must be at least {minimum}")
    return int(value)


def _finite_float(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChatValidationError(f"{field} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ChatValidationError(f"{field} must be finite")
    below = normalized <= minimum if exclusive_minimum else normalized < minimum
    if below or normalized > maximum:
        interval = "(" if exclusive_minimum else "["
        raise ChatValidationError(f"{field} must be in {interval}{minimum}, {maximum}]")
    return normalized


@dataclass(frozen=True, slots=True)
class SamplingOptions:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class StreamOptions:
    include_usage: bool = False


@dataclass(frozen=True, slots=True)
class OpenAIChatRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    stream: bool
    max_tokens: int
    sampling: SamplingOptions
    stop_strings: tuple[str, ...]
    stream_options: StreamOptions
    session_id: str | None


_REQUEST_FIELDS = {
    "model",
    "messages",
    "stream",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "stream_options",
    "session_id",
    "n",
}


def _parse_stops(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    values = (raw,) if type(raw) is str else raw
    if not isinstance(values, list | tuple) or not values or len(values) > 4:
        raise ChatValidationError("stop must be a string or an array of one to four strings")
    if any(type(value) is not str or not value for value in values):
        raise ChatValidationError("stop strings must be non-empty strings")
    if len(set(values)) != len(values):
        raise ChatValidationError("stop strings must be unique")
    return tuple(values)


def _parse_logit_bias(raw: Any, *, semantic_token_count: int) -> tuple[tuple[int, float], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ChatValidationError("logit_bias must be an object keyed by token ID")
    parsed: list[tuple[int, float]] = []
    for raw_token, raw_bias in raw.items():
        if type(raw_token) is int:
            token = raw_token
        elif type(raw_token) is str and raw_token.isascii() and raw_token.isdigit():
            token = int(raw_token)
        else:
            raise ChatValidationError("logit_bias keys must be base-10 token IDs")
        if token < 0 or token >= semantic_token_count:
            raise ChatValidationError("logit_bias token ID escapes the semantic domain")
        bias = _finite_float(raw_bias, f"logit_bias[{token}]", minimum=-100.0, maximum=100.0)
        parsed.append((token, bias))
    if len({token for token, _ in parsed}) != len(parsed):
        raise ChatValidationError("logit_bias contains duplicate normalized token IDs")
    return tuple(sorted(parsed))


def _parse_stream_options(raw: Any) -> StreamOptions:
    if raw is None:
        return StreamOptions()
    if not isinstance(raw, Mapping) or set(raw) - {"include_usage"}:
        raise ChatValidationError("stream_options supports only include_usage")
    include_usage = raw.get("include_usage", False)
    if type(include_usage) is not bool:
        raise ChatValidationError("stream_options.include_usage must be boolean")
    return StreamOptions(include_usage=include_usage)


def parse_chat_completion_request(
    payload: Any,
    *,
    loaded_model: str,
    semantic_token_count: int,
    default_max_tokens: int = 256,
) -> OpenAIChatRequest:
    """Parse the supported request subset and reject every semantics-changing unknown."""

    if not isinstance(payload, Mapping):
        raise ChatValidationError("request body must be a JSON object")
    unknown = sorted(set(payload) - _REQUEST_FIELDS)
    if unknown:
        raise ChatValidationError(f"unsupported request fields: {unknown!r}")
    model = payload.get("model")
    if type(model) is not str or not model:
        raise ChatValidationError("model must be a non-empty string")
    if model != loaded_model:
        raise ChatValidationError(f"model {model!r} is not loaded", code="model_not_found")
    messages = validate_chat_messages(payload.get("messages"))
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise ChatValidationError("stream must be boolean")
    max_tokens = _strict_int(payload.get("max_tokens", default_max_tokens), "max_tokens", minimum=1)
    n = _strict_int(payload.get("n", 1), "n", minimum=1)
    if n != 1:
        raise ChatValidationError("this service supports exactly one completion choice")
    seed = payload.get("seed")
    if seed is not None:
        seed = _strict_int(seed, "seed")
        if seed < -(1 << 63) or seed > (1 << 63) - 1:
            raise ChatValidationError("seed must fit in a signed 64-bit integer")
    session_id = payload.get("session_id")
    if session_id is not None and (
        type(session_id) is not str
        or not session_id
        or session_id.strip() != session_id
        or len(session_id) > 256
    ):
        raise ChatValidationError("session_id must be a canonical string of at most 256 chars")
    top_k = _strict_int(payload.get("top_k", 0), "top_k", minimum=0)
    sampling = SamplingOptions(
        temperature=_finite_float(
            payload.get("temperature", 0.0),
            "temperature",
            minimum=0.0,
            maximum=2.0,
        ),
        top_p=_finite_float(
            payload.get("top_p", 1.0),
            "top_p",
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
        ),
        top_k=top_k,
        seed=seed,
        frequency_penalty=_finite_float(
            payload.get("frequency_penalty", 0.0),
            "frequency_penalty",
            minimum=-2.0,
            maximum=2.0,
        ),
        presence_penalty=_finite_float(
            payload.get("presence_penalty", 0.0),
            "presence_penalty",
            minimum=-2.0,
            maximum=2.0,
        ),
        logit_bias=_parse_logit_bias(
            payload.get("logit_bias"),
            semantic_token_count=semantic_token_count,
        ),
    )
    return OpenAIChatRequest(
        model=model,
        messages=messages,
        stream=stream,
        max_tokens=max_tokens,
        sampling=sampling,
        stop_strings=_parse_stops(payload.get("stop")),
        stream_options=_parse_stream_options(payload.get("stream_options")),
        session_id=session_id,
    )


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        for value, field in (
            (self.prompt_tokens, "prompt_tokens"),
            (self.completion_tokens, "completion_tokens"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


def error_envelope(message: str, *, error_type: str, code: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def completion_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str,
    usage: Usage,
    request_id: str,
    route_id: str,
    session_cache: str,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage.as_dict(),
        "mrun": {
            "request_id": request_id,
            "route_id": route_id,
            "session_cache": session_cache,
        },
    }


def completion_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: Mapping[str, str],
    finish_reason: str | None,
    usage: Usage | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": dict(delta),
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage.as_dict()
    return payload


__all__ = [
    "OpenAIChatRequest",
    "SamplingOptions",
    "StreamOptions",
    "Usage",
    "completion_chunk",
    "completion_response",
    "error_envelope",
    "parse_chat_completion_request",
]
