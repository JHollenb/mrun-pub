import json

import numpy as np
import torch

from mrun.engine.kernels.qstore import QStoreInt2, _dequant_int2
from mrun.engine.kernels.qstore_int2 import _quant_row_int2


def test_quant_row_int2_roundtrip_is_groupwise_ternary():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((5, 17)).astype(np.float32)
    packed, scales = _quant_row_int2(W, G=4)

    assert packed.dtype == np.uint8
    assert packed.shape == (5, (17 * 2 + 7) // 8)
    assert scales.shape == (5, 5)

    deq = _dequant_int2(packed, scales, W.shape[1], G=4)
    per_col_scale = np.repeat(scales, 4, axis=1)[:, : W.shape[1]]
    ratio = deq / per_col_scale
    rounded = np.round(ratio).astype(np.int8)
    assert np.allclose(ratio, rounded, atol=1e-6)
    assert set(np.unique(rounded)).issubset({-1, 0, 1})


def test_qstore_int2_matmul_streams_chunks_and_matches_full_dequant(tmp_path):
    rng = np.random.default_rng(1)
    W = rng.standard_normal((7, 5)).astype(np.float32)
    packed, scales = _quant_row_int2(W, G=4)

    store_dir = tmp_path / "toy-int2"
    store_dir.mkdir()
    (store_dir / "weights.i2").write_bytes(packed.tobytes())
    (store_dir / "scales.f32").write_bytes(scales.astype(np.float32).tobytes())
    (store_dir / "extras.f32").write_bytes(np.zeros(1, dtype=np.float32).tobytes())
    (store_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": "toy",
                "arch": "qwen2",
                "dtype": "int2",
                "group_size": 4,
                "matmul_chunk_rows": 2,
                "config": {},
                "blocks": {
                    "W": {
                        "kind": "qrow2",
                        "shape": [7, 5],
                        "w_off": 0,
                        "row_bytes": packed.shape[1],
                        "s_off": 0,
                        "n_groups": scales.shape[1],
                        "group_size": 4,
                    }
                },
            }
        )
    )

    store = QStoreInt2("toy", root=tmp_path)
    torch.manual_seed(0)  # unseeded randn made chunked-vs-full allclose flaky
    x = torch.randn(3, 4, 5)
    got = store.matmul("W", x)
    max_chunk_bytes = 2 * 5 * 4
    assert store.max_block_bytes <= max_chunk_bytes

    full = x @ store.weight("W").T
    assert torch.allclose(got, full)

    rows = store.embed_rows("W", np.asarray([[0, 2], [6, 1]], dtype=np.int64))
    assert rows.shape == (2, 2, 5)
    assert torch.allclose(rows[0, 1], store.weight("W")[2])
