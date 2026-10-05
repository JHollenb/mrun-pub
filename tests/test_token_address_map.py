from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mrun.engine.kernels.token_address_map import TokenAddressMap


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

    def __len__(self) -> int:
        return 3

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return ("a", "b", "<eos>")[token_id]


def _source(root: Path) -> Path:
    root.mkdir()
    (root / "weights.i8").write_bytes(np.arange(12, dtype=np.int8).tobytes())
    (root / "scales.f32").write_bytes(np.arange(6, dtype=np.float32).tobytes())
    (root / "extras.f32").write_bytes(np.arange(2, dtype=np.float32).tobytes())
    manifest = {
        "model_name": "tiny",
        "arch": "qwen2",
        "dtype": "int8",
        "config": {
            "vocab_size": 4,
            "hidden_size": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
        },
        "blocks": {
            "embed": {
                "kind": "qrow",
                "shape": [4, 2],
                "w_off": 0,
                "w_len": 8,
                "s_off": 0,
                "s_len": 16,
            },
            "norm.final": {"kind": "fp32", "shape": [2], "e_off": 0, "e_len": 8},
            "lm_head": {"alias": "embed"},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_token_address_map_binds_tokens_rows_and_padded_qstore_rows(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    artifact = TokenAddressMap.create(
        tmp_path / "map",
        qstore_root=source,
        tokenizer=_Tokenizer(),
    )

    assert artifact.token_count == 3
    assert artifact.output_token_count == 4
    assert artifact.manifest["tied"] is True
    assert artifact.manifest["padded_output_rows"] == {"start": 3, "stop": 4, "count": 1}
    assert artifact.lookup_token("b") == [1]
    assert artifact.address(2, space="embed") == {
        "space": "embed",
        "token_id": 2,
        "row": 2,
        "provider": "source",
        "physical_root": "embed",
        "weights_file": "weights.i8",
        "weights_byte_offset": 4,
        "scales_file": "scales.f32",
        "scale_byte_offset": 8,
    }
    assert artifact.address(2, space="lm_head")["physical_root"] == "embed"
    assert artifact.address(3, space="lm_head")["row"] == 3

    loaded = TokenAddressMap.load(
        artifact.root,
        tokenizer=_Tokenizer(),
        qstore_root=source,
    )
    assert loaded.semantic_sha256 == artifact.semantic_sha256


def test_linked_map_routes_tied_lexical_address_to_overlay(tmp_path: Path) -> None:
    base = _source(tmp_path / "base")
    extension = _source(tmp_path / "extension")
    extension_manifest_path = extension / "manifest.json"
    extension_manifest = json.loads(extension_manifest_path.read_text(encoding="utf-8"))
    extension_manifest["blocks"].pop("lm_head")
    extension_manifest["linked_image"] = {
        "extension_id": "tiny-extension",
        "overlay_blocks": ["embed"],
        "selection": "all_changed",
        "representation": "sparse_extension_qstore_blocks",
    }
    extension_manifest_path.write_text(json.dumps(extension_manifest), encoding="utf-8")

    artifact = TokenAddressMap.create_linked(
        tmp_path / "linked-map",
        base_qstore_root=base,
        extension_qstore_root=extension,
        tokenizer=_Tokenizer(),
    )
    assert artifact.manifest["map_kind"] == "linked_extension"
    assert artifact.manifest["address_spaces"]["embed"]["provider"] == "extension"
    assert artifact.manifest["address_spaces"]["lm_head"]["provider"] == "extension"
    assert artifact.address(2, space="lm_head")["provider"] == "extension"

    loaded = TokenAddressMap.load(
        artifact.root,
        tokenizer=_Tokenizer(),
        qstore_root=base,
        extension_qstore_root=extension,
    )
    assert loaded.manifest["extension"]["extension_id"] == "tiny-extension"
