from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from mrun.engine.kernels import qstore_int2, qstore_int3, qstore_int4


def _checkpoint(root: Path, *, mismatch: bool = False) -> Path:
    root.mkdir(parents=True)
    config = {
        "model_type": "qwen2",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 8,
        "vocab_size": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
    }
    generator = torch.Generator().manual_seed(91)
    embedding = torch.randn(8, 4, generator=generator)
    lm_head = embedding.clone()
    if mismatch:
        lm_head[0, 0] += 1
    tensors = {
        "model.embed_tokens.weight": embedding,
        "lm_head.weight": lm_head,
        "model.norm.weight": torch.randn(4, generator=generator),
        "model.layers.0.input_layernorm.weight": torch.randn(4, generator=generator),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(4, generator=generator),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4, generator=generator),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 4, generator=generator),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 4, generator=generator),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 4, generator=generator),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4, generator=generator),
        "model.layers.0.mlp.up_proj.weight": torch.randn(8, 4, generator=generator),
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 8, generator=generator),
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    shard = root / "model.safetensors"
    save_file(tensors, shard)
    return shard


def _patch_resolution(
    monkeypatch: pytest.MonkeyPatch,
    module,
    shard: Path,
) -> None:
    monkeypatch.setattr(module, "find_safetensors", lambda _name: [shard])
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda _path: SimpleNamespace(model_type="qwen2")),
    )


@pytest.mark.parametrize(
    ("module", "suffix", "saved_bytes"),
    (
        (qstore_int2, "int2", 8 + 8 * 4),
        (qstore_int3, "int3", 8 * 2 + 8 * 4),
        (qstore_int4, "int4", 8 * 2 + 8 * 4),
    ),
)
def test_low_bit_builders_share_verified_lexical_alias_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    suffix: str,
    saved_bytes: int,
) -> None:
    shard = _checkpoint(tmp_path / "model")
    _patch_resolution(monkeypatch, module, shard)

    store = module.build("tiny", out_root=tmp_path / "stores", store_dir_name="Tiny")
    assert store == tmp_path / "stores" / f"Tiny-{suffix}"
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["tie_word_embeddings"] is True
    assert manifest["blocks"]["lm_head"] == {"alias": "embed"}
    binding = manifest["lexical_weight_binding"]
    assert binding["disposition"] == "verified-duplicate-alias"
    assert binding["alias_saved_bytes"] == saved_bytes


@pytest.mark.parametrize(
    ("module", "suffix"),
    (
        (qstore_int2, "int2"),
        (qstore_int3, "int3"),
        (qstore_int4, "int4"),
    ),
)
def test_low_bit_lexical_mismatch_fails_before_output_is_touched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    suffix: str,
) -> None:
    shard = _checkpoint(tmp_path / "model", mismatch=True)
    _patch_resolution(monkeypatch, module, shard)

    with pytest.raises(RuntimeError, match="source tensors are not byte-identical"):
        module.build("tiny", out_root=tmp_path / "stores", store_dir_name="Tiny")

    assert not (tmp_path / "stores" / f"Tiny-{suffix}").exists()
