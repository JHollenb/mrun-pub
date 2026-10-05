from __future__ import annotations

import hashlib
from typing import Any

import pytest

from mrun.inference import (
    BoundChatTokenizer,
    ChatValidationError,
    IncrementalTextDecoder,
    canonical_chat_template_sha256,
    validate_chat_messages,
)


class _Tokenizer:
    chat_template = "{{ messages }}<assistant>"
    eos_token_id = 9

    def __init__(self) -> None:
        self.decode_text: dict[tuple[int, ...], str] = {}
        self.last_apply: tuple[list[dict[str, str]], dict[str, Any]] | None = None

    def get_chat_template(self, chat_template: str | None = None) -> str:
        assert chat_template in (None, "default")
        return self.chat_template

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        self.last_apply = (messages, kwargs)
        return [1, 2, len(messages)]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) % 10 for character in text]

    def decode(self, ids: list[int], **_kwargs: Any) -> str:
        return self.decode_text.get(tuple(ids), "".join(str(value) for value in ids))


def _bound(tokenizer: _Tokenizer | None = None, **kwargs: Any) -> BoundChatTokenizer:
    tokenizer = tokenizer or _Tokenizer()
    return BoundChatTokenizer(
        tokenizer,
        model_id="toy-chat",
        semantic_token_count=10,
        context_size=32,
        expected_chat_template_sha256=canonical_chat_template_sha256(tokenizer.chat_template),
        **kwargs,
    )


def test_bound_chat_tokenizer_executes_exact_template_and_context_contract() -> None:
    tokenizer = _Tokenizer()
    bound = _bound(tokenizer)
    rendered = bound.render(
        [
            {"role": "system", "content": "Be exact."},
            {"role": "user", "content": "Why?"},
        ],
        max_new_tokens=8,
    )
    assert rendered.token_ids == (1, 2, 2)
    assert rendered.template_sha256 == canonical_chat_template_sha256(tokenizer.chat_template)
    assert tokenizer.last_apply is not None
    assert tokenizer.last_apply[1] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": None,
    }

    with pytest.raises(ChatValidationError, match="exceeds the model context") as error:
        BoundChatTokenizer(
            tokenizer,
            model_id="toy-chat",
            semantic_token_count=10,
            context_size=4,
        ).render([{"role": "user", "content": "Why?"}], max_new_tokens=2)
    assert error.value.code == "context_length_exceeded"


def test_chat_template_and_semantic_domain_mismatches_fail_at_the_boundary() -> None:
    with pytest.raises(ChatValidationError, match="differs from compiled") as error:
        BoundChatTokenizer(
            _Tokenizer(),
            model_id="toy-chat",
            semantic_token_count=10,
            context_size=32,
            expected_chat_template_sha256="0" * 64,
        )
    assert error.value.code == "model_identity_mismatch"

    tokenizer = _Tokenizer()
    tokenizer.apply_chat_template = lambda *_args, **_kwargs: [1, 10]
    with pytest.raises(ChatValidationError, match="semantic token domain"):
        _bound(tokenizer)


def test_legacy_component_graph_raw_template_identity_is_explicitly_bound() -> None:
    tokenizer = _Tokenizer()
    raw_digest = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    bound = BoundChatTokenizer(
        tokenizer,
        model_id="toy-chat",
        semantic_token_count=10,
        context_size=32,
        expected_legacy_raw_chat_template_sha256=raw_digest,
    )

    assert bound.legacy_raw_chat_template_sha256 == raw_digest
    with pytest.raises(ChatValidationError, match="legacy component-graph"):
        BoundChatTokenizer(
            tokenizer,
            model_id="toy-chat",
            semantic_token_count=10,
            context_size=32,
            expected_legacy_raw_chat_template_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="exactly one"):
        BoundChatTokenizer(
            tokenizer,
            model_id="toy-chat",
            semantic_token_count=10,
            context_size=32,
            expected_chat_template_sha256=canonical_chat_template_sha256(tokenizer.chat_template),
            expected_legacy_raw_chat_template_sha256=raw_digest,
        )


def test_external_template_content_is_executed_and_identity_bound() -> None:
    tokenizer = _Tokenizer()
    external = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    bound = BoundChatTokenizer(
        tokenizer,
        model_id="toy-base",
        semantic_token_count=10,
        context_size=32,
        expected_chat_template_sha256=canonical_chat_template_sha256(external),
        template_content=external,
    )

    rendered = bound.render([{"role": "user", "content": "hello"}], max_new_tokens=1)
    assert rendered.template_sha256 == canonical_chat_template_sha256(external)
    assert tokenizer.last_apply is not None
    assert tokenizer.last_apply[1]["chat_template"] == external

    with pytest.raises(ValueError, match="mutually exclusive"):
        BoundChatTokenizer(
            tokenizer,
            model_id="toy-base",
            semantic_token_count=10,
            context_size=32,
            template_name="default",
            template_content=external,
        )


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "x"}],
        [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}],
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        [{"role": "user", "content": ["multimodal"]}],
        [{"role": "developer", "content": "x"}],
        [{"role": "user", "content": "x", "name": "extra"}],
    ],
)
def test_chat_message_subset_rejects_ambiguous_or_unsupported_inputs(messages: Any) -> None:
    with pytest.raises(ChatValidationError):
        validate_chat_messages(messages)


def test_stop_text_encoding_is_exact_and_domain_checked() -> None:
    bound = _bound()
    assert bound.encode_stop_text(["AB", "C"]) == ((5, 6), (7,))
    with pytest.raises(ChatValidationError, match="non-empty"):
        bound.encode_stop_text([""])


def test_incremental_decoder_holds_possible_stop_prefix_until_resolved() -> None:
    tokenizer = _Tokenizer()
    tokenizer.decode_text = {
        (1,): "hello E",
        (1, 2): "hello EN",
        (1, 2, 3): "hello END tail",
    }
    decoder = IncrementalTextDecoder(_bound(tokenizer), stop_strings=("END",))
    assert decoder.push(1).text == "hello "
    assert decoder.push(2).text == ""
    stopped = decoder.push(3)
    assert stopped == stopped.__class__(text="", stopped=True, matched_stop="END")
    assert decoder.emitted_text == "hello "
    with pytest.raises(ChatValidationError, match="after a stop"):
        decoder.push(4)


def test_incremental_decoder_holds_replacement_suffix_and_finishes_exactly() -> None:
    tokenizer = _Tokenizer()
    tokenizer.decode_text = {
        (1,): "caf\ufffd",
        (1, 2): "café",
    }
    decoder = IncrementalTextDecoder(_bound(tokenizer))
    assert decoder.push(1).text == "caf"
    assert decoder.push(2).text == "é"
    assert decoder.finish().text == ""
    assert decoder.emitted_text == "café"


def test_incremental_decoder_detects_rewrites_of_published_text() -> None:
    tokenizer = _Tokenizer()
    tokenizer.decode_text = {(1,): "hello", (1, 2): "hullo"}
    decoder = IncrementalTextDecoder(_bound(tokenizer))
    assert decoder.push(1).text == "hello"
    with pytest.raises(ChatValidationError, match="rewrote already published"):
        decoder.push(2)
