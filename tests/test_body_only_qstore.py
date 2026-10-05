from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mrun.engine.kernels.body_only_qstore import (
    BODY_ONLY_SCHEMA,
    build_body_only_qstore,
    inspect_body_only_qstore,
)
from mrun.engine.kernels.qstore import QStore


def _legacy_store(root: Path) -> Path:
    root.mkdir()
    weights = np.arange(12, dtype=np.int8)
    scales = np.arange(6, dtype=np.float32) + 1.0
    extras = np.arange(4, dtype=np.float32) + 10.0
    (root / "weights.i8").write_bytes(weights.tobytes())
    (root / "scales.f32").write_bytes(scales.tobytes())
    (root / "extras.f32").write_bytes(extras.tobytes())
    manifest = {
        "model_name": "tiny",
        "arch": "qwen2",
        "dtype": "int8",
        "tie_word_embeddings": True,
        "config": {
            "vocab_size": 2,
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
                "shape": [2, 2],
                "w_off": 0,
                "w_len": 4,
                "s_off": 0,
                "s_len": 8,
            },
            "L0.q": {
                "kind": "qrow",
                "shape": [2, 2],
                "w_off": 4,
                "w_len": 4,
                "s_off": 8,
                "s_len": 8,
            },
            "norm.final": {"kind": "fp32", "shape": [2], "e_off": 0, "e_len": 8},
            "lm_head": {"alias": "embed"},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_body_only_repack_removes_lexical_bytes_and_preserves_body(tmp_path: Path) -> None:
    source = _legacy_store(tmp_path / "source")
    output = build_body_only_qstore(source, tmp_path / "body")

    evidence = inspect_body_only_qstore(output)
    assert evidence["schema_version"] == BODY_ONLY_SCHEMA
    assert evidence["lexical_blocks_present"] == []
    assert evidence["file_bytes"] == {
        "weights.i8": 4,
        "scales.f32": 8,
        "extras.f32": 8,
    }
    assert evidence["body_only"]["removed_logical_blocks"] == ["embed", "lm_head"]

    body_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(body_manifest["blocks"]) == {"L0.q", "norm.final"}
    body = QStore("body", root=tmp_path)
    np.testing.assert_array_equal(
        body.w[body_manifest["blocks"]["L0.q"]["w_off"] :][:4],
        np.arange(4, 8, dtype=np.int8),
    )
    body.close()
