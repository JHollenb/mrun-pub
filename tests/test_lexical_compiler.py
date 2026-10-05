from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mrun.engine.kernels.lexical import LexicalBinding, LexicalValues, LexicalWeights
from mrun.engine.kernels.lexical_compiler import (
    compile_lexical_component,
    load_lexical_compilation,
)


class _Backend:
    def to_str(self) -> str:
        return '{"model":"tiny"}'


class _Tokenizer:
    backend_tokenizer = _Backend()
    chat_template = "tiny"
    bos_token_id = None
    eos_token_id = 2
    pad_token_id = 2
    unk_token_id = None
    sep_token_id = cls_token_id = mask_token_id = None

    def __init__(self, tokens: tuple[str, ...], decompositions: dict[str, list[int]]) -> None:
        self.tokens = tokens
        self.decompositions = decompositions

    def __len__(self) -> int:
        return len(self.tokens)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[int(token_id)]

    def decode(self, ids, **_kwargs) -> str:
        return "".join(self.tokens[int(token_id)] for token_id in ids)

    def encode(self, text: str, **_kwargs) -> list[int]:
        return list(self.decompositions.get(text, []))

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("tiny", encoding="utf-8")


CFG = {
    "vocab_size": 8,
    "hidden_size": 2,
    "num_hidden_layers": 1,
    "intermediate_size": 4,
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}


def _source_binding(tmp_path: Path) -> LexicalBinding:
    source = _Tokenizer(
        ("a", "b", "<eos>"),
        {"a": [0], "b": [1], "ab": [0, 1], "<eos>": [2]},
    )
    values = LexicalValues.create(tmp_path / "values", tokenizer=source)
    matrix = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    weights = LexicalWeights.create(
        tmp_path / "weights",
        embed=matrix,
        lm_head=None,
        body_config=CFG,
        architecture="tiny",
    )
    return LexicalBinding.create(
        tmp_path / "binding",
        values=values,
        weights=weights,
        input_rows=[0, 1, 2],
        output_rows=list(range(3)),
        output_token_count=3,
    )


def test_compiler_preserves_shared_rows_and_decomposes_new_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    source_binding = _source_binding(tmp_path)
    target = _Tokenizer(
        ("a", "ab", "<eos>", "new"),
        {"a": [0], "ab": [0, 1], "<eos>": [2], "new": []},
    )

    result = compile_lexical_component(
        tmp_path / "compiled",
        source_binding=source_binding,
        target_tokenizer=target,
        body_config=CFG,
        architecture="tiny",
        seed=7,
    )

    np.testing.assert_array_equal(result.component.embed[0], source_binding.weights.embed[0])
    np.testing.assert_allclose(result.component.embed[1], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(result.component.embed[2], source_binding.weights.embed[2])
    assert result.mapping[1]["kind"] == "decomposed"
    assert result.mapping[3]["kind"].startswith("fallback")
    assert result.manifest["mapping"]["kind_counts"]["exact"] == 2

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda path, local_files_only=True: target),
    )
    loaded = load_lexical_compilation(
        result.root,
        body_config=CFG,
        architecture="tiny",
    )
    assert loaded.semantic_sha256 == result.semantic_sha256
    assert loaded.source_ids(1) == (0, 1)


def test_compiler_rejects_body_abi_mismatch(tmp_path: Path) -> None:
    source_binding = _source_binding(tmp_path)
    target = _Tokenizer(("a", "b", "<eos>"), {"a": [0], "b": [1], "<eos>": [2]})

    try:
        compile_lexical_component(
            tmp_path / "compiled",
            source_binding=source_binding,
            target_tokenizer=target,
            body_config={**CFG, "hidden_size": 3},
            architecture="tiny",
        )
    except ValueError as exc:
        assert "body ABI" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("body ABI mismatch was not rejected")
