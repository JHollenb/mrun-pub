"""Exact chat-template rendering and conservative incremental detokenization.

This module owns text/token semantics only.  It does not know about HTTP, Torch, MLX, CUDA, or
mutable K/V state.  A tokenizer and its chat template are executable model identity: callers must
bind the expected template digest at process startup and send only semantic token IDs to the
generation service.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class ChatValidationError(ValueError):
    """A public chat request cannot be represented under the bound tokenizer contract."""

    def __init__(self, message: str, *, code: str = "invalid_request_error") -> None:
        self.code = code
        super().__init__(message)


def canonical_chat_template_sha256(content: str) -> str:
    """Match the canonical digest used by ``decompiler.ChatTemplateIR``."""

    if type(content) is not str or not content:
        raise ValueError("chat template content must be a non-empty string")
    encoded = json.dumps(
        {"content": content},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ChatValidationError(f"unsupported chat role {self.role!r}")
        if type(self.content) is not str:
            raise ChatValidationError("message content must be a string")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def validate_chat_messages(
    messages: Sequence[Mapping[str, Any] | ChatMessage],
) -> tuple[ChatMessage, ...]:
    """Validate the initial text-only, alternating system/user/assistant chat subset."""

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise ChatValidationError("messages must be a non-empty array")
    normalized: list[ChatMessage] = []
    for index, raw in enumerate(messages):
        if isinstance(raw, ChatMessage):
            message = raw
        else:
            if not isinstance(raw, Mapping) or set(raw) != {"role", "content"}:
                raise ChatValidationError(
                    f"messages[{index}] must contain exactly role and string content"
                )
            role = raw.get("role")
            content = raw.get("content")
            if type(role) is not str or type(content) is not str:
                raise ChatValidationError(f"messages[{index}] role and content must be strings")
            message = ChatMessage(role=role, content=content)
        normalized.append(message)

    system_positions = [
        index for index, message in enumerate(normalized) if message.role == "system"
    ]
    if system_positions not in ([], [0]):
        raise ChatValidationError("a system message is legal only once and in the first position")
    dialogue = normalized[1:] if system_positions else normalized
    if not dialogue or dialogue[0].role != "user":
        raise ChatValidationError("the dialogue must begin with a user message")
    expected = "user"
    for message in dialogue:
        if message.role != expected:
            raise ChatValidationError("user and assistant messages must alternate")
        expected = "assistant" if expected == "user" else "user"
    if dialogue[-1].role != "user":
        raise ChatValidationError("chat completion requires a final user message")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class RenderedChat:
    model_id: str
    messages: tuple[ChatMessage, ...]
    token_ids: tuple[int, ...]
    template_sha256: str
    semantic_token_count: int


def _strict_token_ids(value: Any, *, field: str) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping) and "input_ids" in value:
        value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
    if (
        isinstance(value, Sequence)
        and len(value) == 1
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        value = value[0]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ChatValidationError(f"{field} must produce a non-empty token sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ChatValidationError(f"{field} produced a non-integer token ID")
    return tuple(int(item) for item in value)


class BoundChatTokenizer:
    """Tokenizer facade bound to one model, template, semantic domain, and context limit."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        model_id: str,
        semantic_token_count: int,
        context_size: int,
        expected_chat_template_sha256: str | None = None,
        expected_legacy_raw_chat_template_sha256: str | None = None,
        template_name: str | None = None,
        template_content: str | None = None,
    ) -> None:
        if type(model_id) is not str or not model_id or model_id.strip() != model_id:
            raise ValueError("model_id must be a canonical non-empty string")
        for value, field in (
            (semantic_token_count, "semantic_token_count"),
            (context_size, "context_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.semantic_token_count = int(semantic_token_count)
        self.context_size = int(context_size)
        if template_content is not None and (
            type(template_content) is not str or not template_content
        ):
            raise ValueError("template_content must be a non-empty string or None")
        if template_name is not None and template_content is not None:
            raise ValueError("template_name and template_content are mutually exclusive")
        self.template_name = template_name
        self.template_content = template_content
        self.chat_template = self._resolve_template()
        self.chat_template_sha256 = canonical_chat_template_sha256(self.chat_template)
        self.legacy_raw_chat_template_sha256 = hashlib.sha256(
            self.chat_template.encode("utf-8")
        ).hexdigest()
        if (
            expected_chat_template_sha256 is not None
            and expected_legacy_raw_chat_template_sha256 is not None
        ):
            raise ValueError("pass exactly one canonical or legacy raw chat-template identity")
        if (
            expected_chat_template_sha256 is not None
            and self.chat_template_sha256 != expected_chat_template_sha256
        ):
            raise ChatValidationError(
                "runtime chat template differs from compiled model identity",
                code="model_identity_mismatch",
            )
        if (
            expected_legacy_raw_chat_template_sha256 is not None
            and self.legacy_raw_chat_template_sha256 != expected_legacy_raw_chat_template_sha256
        ):
            raise ChatValidationError(
                "runtime chat template differs from legacy component-graph identity",
                code="model_identity_mismatch",
            )
        self._startup_probe()

    def _resolve_template(self) -> str:
        if self.template_content is not None:
            return self.template_content
        resolver = getattr(self.tokenizer, "get_chat_template", None)
        if callable(resolver):
            try:
                template = resolver(chat_template=self.template_name)
            except TypeError:
                template = resolver(self.template_name)
        else:
            raw = getattr(self.tokenizer, "chat_template", None)
            if isinstance(raw, Mapping):
                selected = self.template_name or "default"
                template = raw.get(selected)
            else:
                template = raw
        if type(template) is not str or not template:
            raise ChatValidationError(
                "the model tokenizer has no selected chat template",
                code="model_not_chat_capable",
            )
        return template

    def _startup_probe(self) -> None:
        probe = (
            ChatMessage("system", "Answer carefully."),
            ChatMessage("user", "Reply with one word."),
        )
        token_ids = self._apply_template(probe)
        self._validate_domain(token_ids, field="chat-template startup probe")

    def _apply_template(self, messages: Sequence[ChatMessage]) -> tuple[int, ...]:
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            raise ChatValidationError(
                "the model tokenizer cannot execute its chat template",
                code="model_not_chat_capable",
            )
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": None,
        }
        if self.template_content is not None:
            kwargs["chat_template"] = self.template_content
        elif self.template_name is not None:
            kwargs["chat_template"] = self.template_name
        try:
            rendered = apply_template([message.as_dict() for message in messages], **kwargs)
        except (TypeError, ValueError) as exc:
            raise ChatValidationError(f"chat template rejected the messages: {exc}") from exc
        return _strict_token_ids(rendered, field="chat template")

    def _validate_domain(self, token_ids: Sequence[int], *, field: str) -> None:
        if any(token < 0 or token >= self.semantic_token_count for token in token_ids):
            raise ChatValidationError(
                f"{field} escaped the compiled semantic token domain",
                code="model_identity_mismatch",
            )

    def render(
        self,
        messages: Sequence[Mapping[str, Any] | ChatMessage],
        *,
        max_new_tokens: int,
    ) -> RenderedChat:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ChatValidationError("max_tokens must be an integer")
        if max_new_tokens <= 0:
            raise ChatValidationError("max_tokens must be positive")
        normalized = validate_chat_messages(messages)
        token_ids = self._apply_template(normalized)
        self._validate_domain(token_ids, field="rendered chat")
        if len(token_ids) + max_new_tokens > self.context_size:
            raise ChatValidationError(
                "rendered prompt plus max_tokens exceeds the model context",
                code="context_length_exceeded",
            )
        return RenderedChat(
            model_id=self.model_id,
            messages=normalized,
            token_ids=token_ids,
            template_sha256=self.chat_template_sha256,
            semantic_token_count=self.semantic_token_count,
        )

    def encode_stop_text(self, values: Sequence[str]) -> tuple[tuple[int, ...], ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ChatValidationError("stop text must be an array of strings")
        encode = getattr(self.tokenizer, "encode", None)
        if not callable(encode):
            raise ChatValidationError("tokenizer cannot encode stop text")
        rows: list[tuple[int, ...]] = []
        for value in values:
            if type(value) is not str or not value:
                raise ChatValidationError("stop strings must be non-empty")
            try:
                row = _strict_token_ids(
                    encode(value, add_special_tokens=False),
                    field="stop string",
                )
            except (TypeError, ValueError) as exc:
                if isinstance(exc, ChatValidationError):
                    raise
                raise ChatValidationError(f"tokenizer rejected a stop string: {exc}") from exc
            self._validate_domain(row, field="stop string")
            rows.append(row)
        return tuple(rows)

    def decode(self, token_ids: Sequence[int]) -> str:
        self._validate_domain(token_ids, field="generated output")
        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            raise ChatValidationError("tokenizer cannot decode generated output")
        try:
            value = decode(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            value = decode(list(token_ids), skip_special_tokens=True)
        if type(value) is not str:
            raise ChatValidationError("tokenizer decode did not return text")
        return value


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str
    stopped: bool
    matched_stop: str | None = None


class IncrementalTextDecoder:
    """Decode accumulated IDs while retaining every possible stop/UTF-8 boundary suffix."""

    def __init__(
        self,
        tokenizer: BoundChatTokenizer,
        *,
        stop_strings: Sequence[str] = (),
        include_stop: bool = False,
    ) -> None:
        if isinstance(stop_strings, (str, bytes)):
            raise ChatValidationError("stop_strings must be an array")
        stops = tuple(stop_strings)
        if any(type(value) is not str or not value for value in stops):
            raise ChatValidationError("stop strings must be non-empty strings")
        if len(set(stops)) != len(stops):
            raise ChatValidationError("stop strings must be unique")
        if type(include_stop) is not bool:
            raise TypeError("include_stop must be boolean")
        self.tokenizer = tokenizer
        self.stop_strings = stops
        self.include_stop = include_stop
        self.token_ids: list[int] = []
        self.emitted_text = ""
        self.stopped = False
        self.matched_stop: str | None = None

    def _earliest_stop(self, text: str) -> tuple[int, str] | None:
        matches = [
            (position, stop) for stop in self.stop_strings if (position := text.find(stop)) >= 0
        ]
        return min(matches, key=lambda item: (item[0], len(item[1]), item[1])) if matches else None

    def _holdback(self, text: str) -> int:
        held = 0
        for stop in self.stop_strings:
            maximum = min(len(stop) - 1, len(text))
            for length in range(maximum, 0, -1):
                if text.endswith(stop[:length]):
                    held = max(held, length)
                    break
        # Byte-level tokenizers often expose U+FFFD until a later token completes UTF-8.  Never
        # publish that unstable suffix; the complete decode may replace it on the next push.
        replacement = len(text) - len(text.rstrip("\ufffd"))
        return max(held, replacement)

    def _advance(self, *, final: bool) -> TextDelta:
        decoded = self.tokenizer.decode(self.token_ids)
        match = self._earliest_stop(decoded)
        if match is not None:
            position, stop = match
            end = position + (len(stop) if self.include_stop else 0)
            visible = decoded[:end]
            stopped = True
            matched = stop
        else:
            held = 0 if final else self._holdback(decoded)
            visible = decoded[: len(decoded) - held] if held else decoded
            stopped = False
            matched = None
        if not visible.startswith(self.emitted_text):
            raise ChatValidationError(
                "tokenizer rewrote already published text; incremental contract is unsafe",
                code="model_identity_mismatch",
            )
        delta = visible[len(self.emitted_text) :]
        self.emitted_text = visible
        if stopped:
            self.stopped = True
            self.matched_stop = matched
        return TextDelta(text=delta, stopped=stopped, matched_stop=matched)

    def push(self, token_id: int) -> TextDelta:
        if self.stopped:
            raise ChatValidationError("cannot decode tokens after a stop matched")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token_id must be an integer")
        if token_id < 0 or token_id >= self.tokenizer.semantic_token_count:
            raise ChatValidationError("generated token escaped the semantic domain")
        self.token_ids.append(int(token_id))
        return self._advance(final=False)

    def finish(self) -> TextDelta:
        if self.stopped:
            return TextDelta(text="", stopped=True, matched_stop=self.matched_stop)
        return self._advance(final=True)


__all__ = [
    "BoundChatTokenizer",
    "ChatMessage",
    "ChatValidationError",
    "IncrementalTextDecoder",
    "RenderedChat",
    "TextDelta",
    "canonical_chat_template_sha256",
    "validate_chat_messages",
]
