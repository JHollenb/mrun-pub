"""Resident weight-cache + fast torch dequant (int8 qrow path).

Non-model-gated: builds a tiny synthetic int8 store on disk and exercises the reader
directly, so the cache/eviction/parity contract is covered without a cached HF model.
The end-to-end paged-vs-HF argmax-exact gate lives in ``test_paged_parity.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mrun.engine.kernels.qstore import QStore, _dequant_qrow
from mrun.engine.paged import _auto_cache_mb


def _build_store(tmp: Path, blocks_shapes: dict[str, tuple[int, int]], seed: int = 0):
    """Write a minimal qrow-only store (weights.i8/scales.f32/extras.f32 + manifest)."""
    rng = np.random.default_rng(seed)
    w_parts, s_parts, blocks = [], [], {}
    w_off = s_off = 0
    ref: dict[str, np.ndarray] = {}
    for name, (out, inn) in blocks_shapes.items():
        q = rng.integers(-128, 128, size=(out, inn), dtype=np.int8)
        sc = rng.random(out).astype(np.float32) + 0.1
        blocks[name] = {
            "kind": "qrow", "shape": [out, inn],
            "w_off": w_off, "w_len": out * inn,
            "s_off": s_off, "s_len": out * 4,
        }
        ref[name] = q.astype(np.float32) * sc[:, None]        # numpy reference dequant
        w_parts.append(q.reshape(-1))
        s_parts.append(sc)
        w_off += out * inn
        s_off += out * 4
    d = tmp / "synthstore"
    d.mkdir()
    np.concatenate(w_parts).astype(np.int8).tofile(d / "weights.i8")
    np.concatenate(s_parts).astype(np.float32).tofile(d / "scales.f32")
    np.zeros(1, np.float32).tofile(d / "extras.f32")
    (d / "manifest.json").write_text(json.dumps({
        "arch": "qwen2", "config": {"hidden_size": 8}, "blocks": blocks,
    }))
    return d, ref


def test_fast_dequant_bit_identical_to_numpy(tmp_path: Path):
    _, ref = _build_store(tmp_path, {"L0.q": (32, 16), "L0.up": (48, 16)})
    store = QStore("synthstore", root=tmp_path)
    for name, want in ref.items():
        got = store.weight(name).numpy()
        assert np.array_equal(got, want), f"{name}: fast torch dequant not bit-exact vs numpy"


def test_helper_matches_numpy_reference():
    rng = np.random.default_rng(1)
    q = rng.integers(-128, 128, size=(10, 7), dtype=np.int8)
    sc = rng.random(10).astype(np.float32) + 0.1
    ref = q.astype(np.float32) * sc[:, None]
    got = _dequant_qrow(q.copy(), sc.copy()).numpy()
    assert np.array_equal(got, ref)


def test_cache_off_by_default_no_reuse(tmp_path: Path):
    _build_store(tmp_path, {"L0.q": (16, 8)})
    store = QStore("synthstore", root=tmp_path)      # cache_mb defaults to 0
    a = store.weight("L0.q")
    b = store.weight("L0.q")
    assert a is not b                                # streaming: fresh tensor each call


def test_cache_hit_returns_same_tensor(tmp_path: Path):
    _build_store(tmp_path, {"L0.q": (16, 8)})
    store = QStore("synthstore", root=tmp_path, cache_mb=8.0)
    a = store.weight("L0.q")
    b = store.weight("L0.q")
    assert a is b                                    # cache hit: identical object (=> bit-exact)


def test_cache_lru_eviction_respects_budget(tmp_path: Path, monkeypatch):
    # LRU is now the OPT-IN policy (default pin-fill; measured 2026-07-24: LRU gets
    # 0 hit bytes on the periodic paged forward). Same eviction contract under the flag.
    monkeypatch.setenv("MRUN_QSTORE_CACHE_POLICY", "lru")
    # three 16x256 fp32 blocks = 16*256*4 = 16384 B each; budget fits ~2.
    shapes = {f"L0.{n}": (16, 256) for n in ("q", "k", "v")}
    _build_store(tmp_path, shapes)
    block_bytes = 16 * 256 * 4
    store = QStore("synthstore", root=tmp_path, cache_mb=(2.5 * block_bytes) / 1e6)
    store.weight("L0.q")
    store.weight("L0.k")
    store.weight("L0.v")                             # q should be evicted (LRU)
    assert store._cache_bytes <= store._cache_budget
    assert len(store._cache) == 2
    assert "L0.q" not in store._cache
    assert "L0.v" in store._cache


def test_cache_pin_fill_default_retains_first_fill(tmp_path: Path):
    # default policy: admit until full, never evict (Belady-optimal on cyclic access)
    shapes = {f"L0.{n}": (16, 256) for n in ("q", "k", "v")}
    _build_store(tmp_path, shapes)
    block_bytes = 16 * 256 * 4
    store = QStore("synthstore", root=tmp_path, cache_mb=(2.5 * block_bytes) / 1e6)
    store.weight("L0.q")
    store.weight("L0.k")
    store.weight("L0.v")                             # does not fit; NOT admitted
    assert sorted(store._cache) == ["L0.k", "L0.q"]
    assert store._cache_bytes <= store._cache_budget


def test_cache_never_stores_block_bigger_than_budget(tmp_path: Path):
    _build_store(tmp_path, {"L0.q": (64, 256)})               # 64KB block
    store = QStore("synthstore", root=tmp_path, cache_mb=(4 * 1024) / 1e6)  # 4KB budget
    store.weight("L0.q")
    assert len(store._cache) == 0                    # oversized block skipped (no evict-loop hang)


def test_compute_dtype_matmul_accepts_fp32_norm_output(tmp_path: Path):
    _build_store(tmp_path, {"L0.q": (16, 8)})
    store = QStore("synthstore", root=tmp_path, compute_dtype="bf16")

    output = store.matmul("L0.q", torch.ones((1, 8), dtype=torch.float32))

    assert output.dtype is torch.bfloat16


def test_compute_dtype_applies_to_streamed_row_blocks(tmp_path: Path):
    _build_store(tmp_path, {"lm_head": (16, 8)})
    store = QStore("synthstore", root=tmp_path, compute_dtype="bf16")

    _start, _end, rows = next(store.row_blocks("lm_head", bs=4))

    assert rows.dtype is torch.bfloat16


def test_embed_rows_supports_batched_indices(tmp_path: Path):
    _build_store(tmp_path, {"embed": (16, 8)})
    store = QStore("synthstore", root=tmp_path)

    rows = store.embed_rows("embed", np.asarray([[1, 2], [3, 4]], dtype=np.int64))

    assert rows.shape == (2, 2, 8)


def test_cuda_auto_cache_does_not_spend_ram_budget_on_vram(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("RSS_LIMIT_MB", "64000")

    assert _auto_cache_mb() == 0.0
