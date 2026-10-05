"""qstore_int2.py - experimental ternary int2 store builder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ...models import find_safetensors, store_name
from ...paths import stores_root
from .qstore_build import (
    _arch_config,
    _canon,
    _is_fp32_block,
    preflight_lexical_binding,
    rss_mb,
)

GROUP = 128
MATMUL_CHUNK_ROWS = 1024


def _pack_int2_codes(codes: np.ndarray) -> np.ndarray:
    """Pack uint2 codes [out,in] into little-endian row bitstreams."""
    out, inn = codes.shape
    bits = ((codes[:, :, None] >> np.arange(2, dtype=np.uint8)) & 1).reshape(out, inn * 2)
    row_bits = inn * 2
    row_bytes = (row_bits + 7) // 8
    pad = row_bytes * 8 - row_bits
    if pad:
        bits = np.concatenate([bits, np.zeros((out, pad), dtype=np.uint8)], axis=1)
    return np.packbits(bits, axis=1, bitorder="little")


def _quant_row_int2(W: np.ndarray, G: int = GROUP):
    """Per-row, group-wise ternary int2. Returns (packed, scales).

    This is intentionally experimental: use a TWN-style threshold so the two-bit store keeps
    a sparse ternary approximation instead of pinning every group to its single max value.
    """
    out, inn = W.shape
    ng = (inn + G - 1) // G
    scales = np.empty((out, ng), np.float32)
    q = np.empty((out, inn), np.int8)
    for g in range(ng):
        c0, c1 = g * G, min((g + 1) * G, inn)
        blk = W[:, c0:c1]
        abs_blk = np.abs(blk)
        active = abs_blk >= (0.7 * abs_blk.mean(axis=1))[:, None]
        count = active.sum(axis=1)
        weighted = (abs_blk * active).sum(axis=1)
        s = np.divide(
            weighted,
            count,
            out=np.ones_like(weighted, dtype=np.float32),
            where=count > 0,
        )
        s = np.where(s == 0.0, 1.0, s).astype(np.float32)
        scales[:, g] = s
        q[:, c0:c1] = np.where(active, np.sign(blk), 0).astype(np.int8)
    codes = (q + 1).astype(np.uint8)                                  # -1..1 -> 0..2; 3 unused
    return np.ascontiguousarray(_pack_int2_codes(codes)), np.ascontiguousarray(scales)


def build(model_name: str, *, out_root: Path | None = None, store_dir_name: str | None = None) -> Path:
    """Build the experimental int2 store for ``model_name`` under ``out_root``."""
    from safetensors import safe_open
    from transformers import AutoConfig

    files = find_safetensors(model_name)
    if not files:
        raise FileNotFoundError(f"no safetensors for {model_name!r}")
    snap = files[0].parent
    cfg = AutoConfig.from_pretrained(str(snap))
    raw = json.loads((snap / "config.json").read_text())
    arch = raw.get("model_type", getattr(cfg, "model_type", "?"))
    if arch not in ("qwen2", "llama", "gpt_neox", "mamba"):
        raise NotImplementedError(f"arch {arch!r} not supported (qwen2/llama/gpt_neox/mamba)")

    def encode_lexical(tensor):
        packed, scales = _quant_row_int2(tensor.to(dtype=torch.float32).numpy())
        return {"weights": packed, "scales": scales}

    lexical_binding, lexical_manifest = preflight_lexical_binding(
        files, arch, raw, encode_lexical
    )

    root = Path(out_root) if out_root is not None else stores_root()
    dir_name = store_dir_name or store_name(model_name)
    out = root / f"{dir_name}-int2"
    out.mkdir(parents=True, exist_ok=True)
    w_path, s_path, e_path = out / "weights.i2", out / "scales.f32", out / "extras.f32"
    wf, sf, ef = open(w_path, "wb"), open(s_path, "wb"), open(e_path, "wb")
    w_off = s_off = e_off = 0
    blocks: dict[str, dict] = {}
    print(f">>> int2 build - {model_name}  arch={arch}  G={GROUP}  -> {out}")
    n_q = n_fp = 0
    for f in sorted(files):
        with safe_open(str(f), framework="pt") as st:
            for key in st.keys():
                name = _canon(key, arch)
                if name is None:
                    continue
                if not lexical_binding.should_write(name):
                    continue
                W = st.get_tensor(key).to(dtype=torch.float32).numpy()
                if _is_fp32_block(name):
                    arr = np.ascontiguousarray(W.astype(np.float32))
                    ef.write(arr.tobytes())
                    blocks[name] = {"kind": "fp32", "shape": list(W.shape),
                                    "e_off": e_off, "e_len": arr.nbytes}
                    e_off += arr.nbytes; n_fp += 1
                else:
                    if W.ndim != 2:
                        raise ValueError(f"{name}: {W.shape}")
                    out_, inn = W.shape
                    packed, scales = _quant_row_int2(W)
                    wf.write(packed.tobytes()); sf.write(scales.tobytes())
                    blocks[name] = {"kind": "qrow2", "shape": [out_, inn],
                                    "w_off": w_off, "row_bytes": packed.shape[1],
                                    "s_off": s_off, "n_groups": scales.shape[1],
                                    "group_size": GROUP}
                    w_off += packed.nbytes; s_off += scales.nbytes; n_q += 1
                del W
        print(f"    {f.name}: blocks={len(blocks)}  RSS={rss_mb():.0f}MB")
    wf.close(); sf.close(); ef.close()

    tie = lexical_binding.declared_tied
    if tie:
        blocks["lm_head"] = {"alias": "embed"}
    manifest = {
        "model_name": model_name, "arch": arch, "dtype": "int2", "group_size": GROUP,
        "matmul_chunk_rows": MATMUL_CHUNK_ROWS, "tie_word_embeddings": tie,
        "lexical_weight_binding": lexical_manifest,
        "experimental": True,
        "quantization": {
            "scheme": "twn_ternary",
            "levels": [-1, 0, 1],
            "threshold": "0.7 * mean(abs(group))",
        },
        "config": _arch_config(arch, raw),
        "blocks": blocks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    disk = sum(p.stat().st_size for p in (w_path, s_path, e_path)) / 1e6
    print(f"  q={n_q} fp32={n_fp}  int2 store={disk:.0f}MB  -> {out}")
    return out
