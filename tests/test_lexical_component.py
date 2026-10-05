from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mrun.engine.kernels.lexical import (
    LexicalBinding,
    LexicalComponent,
    LexicalQStoreView,
    LexicalValues,
    LexicalWeights,
    body_abi,
    load_separated_lexical,
    train_lexical_bridge,
)


class _Backend:
    def to_str(self) -> str:
        return '{"model":"tiny"}'


class _Tokenizer:
    backend_tokenizer = _Backend()
    chat_template = "tiny"
    bos_token_id = 0
    eos_token_id = 3
    pad_token_id = 3
    unk_token_id = 1
    sep_token_id = cls_token_id = mask_token_id = None

    def __len__(self) -> int:
        return 4

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return ("a", "b", "c", "<eos>")[token_id]

    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("tiny", encoding="utf-8")


class _Body:
    cfg = {
        "vocab_size": 8,
        "hidden_size": 2,
        "num_hidden_layers": 1,
        "intermediate_size": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    }
    man = {"arch": "qwen2"}
    device = "cpu"

    def embed_rows(self, name, ids):
        return torch.full((len(ids), 2), 9.0)

    def row_blocks(self, name, bs=8192):
        yield 0, 8, torch.full((8, 2), 9.0)

    def selected_rows_fp32(self, name, ids):
        return torch.full((len(ids), 2), 9.0)

    def close(self):
        pass


def _component(tmp_path: Path) -> LexicalComponent:
    tokenizer = _Tokenizer()
    matrix = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    return LexicalComponent.create(
        tmp_path / "lexical",
        tokenizer=tokenizer,
        embed=matrix,
        lm_head=matrix + 10,
        body_config=_Body.cfg,
        architecture="qwen2",
    )


def test_lexical_view_replaces_only_vocab_operations(tmp_path: Path) -> None:
    component = _component(tmp_path)
    view = LexicalQStoreView(_Body(), component)

    assert view.cfg["vocab_size"] == 4
    torch.testing.assert_close(view.embed_rows("embed", np.array([1])), torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(
        view.selected_rows_fp32("lm_head", [1]), torch.tensor([[12.0, 13.0]])
    )
    assert next(view.row_blocks("lm_head"))[1] == 4
    assert view.embed_rows("L0.q", [0])[0, 0].item() == 9.0


def test_lexical_artifact_binds_body_abi_and_files(tmp_path: Path, monkeypatch) -> None:
    component = _component(tmp_path)
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda path, local_files_only=True: _Tokenizer()),
    )
    loaded = LexicalComponent.load(
        component.root, body_config=_Body.cfg, architecture="qwen2"
    )
    assert loaded.token_count == 4
    assert loaded.manifest["body_abi_sha256"] == body_abi(_Body.cfg, "qwen2")["semantic_sha256"]

    with pytest.raises(ValueError, match="body ABI"):
        LexicalComponent.load(
            component.root,
            body_config={**_Body.cfg, "hidden_size": 3},
            architecture="qwen2",
        )

    (component.root / "embed.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="missing or changed"):
        LexicalComponent.load(
            component.root, body_config=_Body.cfg, architecture="qwen2"
        )


def test_lexical_bridge_freezes_body() -> None:
    body = torch.nn.Linear(3, 3, bias=False)
    embedding = torch.nn.Embedding(4, 3)
    output = torch.nn.Linear(3, 4, bias=False)
    original = body.weight.detach().clone()
    batches = [torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])] * 2

    losses = train_lexical_bridge(body, embedding, output, batches, steps=2)

    assert len(losses) == 2
    torch.testing.assert_close(body.weight, original)
    assert any(parameter.grad is not None for parameter in embedding.parameters())


def test_separated_values_weights_and_binding_round_trip(tmp_path: Path, monkeypatch) -> None:
    tokenizer = _Tokenizer()
    values = LexicalValues.create(tmp_path / "values", tokenizer=tokenizer)
    weights = LexicalWeights.create(
        tmp_path / "weights",
        embed=torch.arange(10, dtype=torch.float32).reshape(5, 2),
        lm_head=torch.arange(10, dtype=torch.float32).reshape(5, 2) + 20,
        body_config=_Body.cfg,
        architecture="qwen2",
    )
    binding = LexicalBinding.create(
        tmp_path / "binding",
        values=values,
        weights=weights,
        input_rows=[4, 2, 0, 3],
        output_rows=[1, 3, 4, 0],
    )

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda path, local_files_only=True: _Tokenizer()),
    )
    loaded = load_separated_lexical(
        values_path=values.root,
        weights_path=weights.root,
        binding_path=binding.root,
        body_config=_Body.cfg,
        architecture="qwen2",
    )
    view = LexicalQStoreView(_Body(), loaded)
    torch.testing.assert_close(view.embed_rows("embed", [0]), torch.tensor([[8.0, 9.0]]))
    torch.testing.assert_close(
        view.selected_rows_fp32("lm_head", [0]), torch.tensor([[22.0, 23.0]])
    )
    assert [end for _, end, _ in view.row_blocks("lm_head", bs=2)] == [2, 4]

    padded_binding = LexicalBinding.create(
        tmp_path / "padded-binding",
        values=values,
        weights=weights,
        input_rows=[4, 2, 0, 3],
        output_rows=[0, 1, 2, 3, 4],
        output_token_count=5,
    )
    padded_view = LexicalQStoreView(
        _Body(),
        LexicalBinding.load(
            padded_binding.root,
            values=values,
            weights=weights,
        ),
    )
    assert padded_view.cfg["vocab_size"] == 5
    assert [end for _, end, _ in padded_view.row_blocks("lm_head", bs=2)] == [2, 4, 5]
