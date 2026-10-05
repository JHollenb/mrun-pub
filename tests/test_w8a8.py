"""W8A8 kernel parity + correctness (CPU, no model needed).

Guards the invariant the engine relies on: the ONLY numerical difference between the W8A8 path
and the fp32-dequant path is activation quantization, and protecting the outlier columns keeps
argmax exact. Also checks that ``torch._int_mm`` and the int32 fallback agree bit-for-bit.
"""
from __future__ import annotations

import torch

from mrun.engine.kernels.w8a8 import _int8_gemm, w8a8_matmul


def _setup(out=64, inn=128, T=5, n_outlier=3, seed=0):
    torch.manual_seed(seed)
    q_w = torch.randint(-127, 128, (out, inn), dtype=torch.int8)
    sc_w = torch.rand(out) * 0.02 + 1e-3
    W = q_w.float() * sc_w[:, None]          # fp32-dequant reference weight
    x = torch.randn(T, inn)
    x[:, torch.randperm(inn)[:n_outlier]] *= 300.0   # LLM.int8-style activation outliers
    ref = x @ W.T                            # the fp32-dequant path (parity target)
    return q_w, sc_w, x, ref


def test_int_mm_matches_fallback():
    q_x = torch.randint(-127, 128, (7, 128), dtype=torch.int8)
    q_w = torch.randint(-127, 128, (64, 128), dtype=torch.int8)
    a = _int8_gemm(q_x, q_w, use_int_mm=True)
    b = _int8_gemm(q_x, q_w, use_int_mm=False)
    assert a.dtype == torch.int32 and b.dtype == torch.int32
    assert torch.equal(a, b)


def test_outlier_protection_shrinks_error_and_keeps_argmax():
    q_w, sc_w, x, ref = _setup(n_outlier=3)
    # no protection: outliers blow up the per-tensor int8 scale
    d0 = (w8a8_matmul(q_w, sc_w, x, protect_k=0) - ref).abs().max().item()
    # protecting the 3 outlier columns collapses the error by >50x
    d = (w8a8_matmul(q_w, sc_w, x, protect_k=4) - ref).abs().max().item()
    assert d0 > 20.0
    assert d < 0.5
    assert d < d0 / 20


def test_argmax_exact_with_protection():
    q_w, sc_w, x, ref = _setup(out=128, inn=256, T=8, n_outlier=4)
    y = w8a8_matmul(q_w, sc_w, x, protect_k=8)
    assert torch.equal(y.argmax(-1), ref.argmax(-1))


def test_int_mm_vs_fallback_full_matmul_equal():
    q_w, sc_w, x, _ = _setup()
    y1 = w8a8_matmul(q_w, sc_w, x, protect_k=8, use_int_mm=True)
    y2 = w8a8_matmul(q_w, sc_w, x, protect_k=8, use_int_mm=False)
    assert (y1 - y2).abs().max().item() == 0.0


def test_static_protect_list_matches_topk():
    q_w, sc_w, x, _ = _setup(n_outlier=3)
    col_max = x.abs().amax(dim=0)
    idx = torch.topk(col_max, 4).indices
    y_static = w8a8_matmul(q_w, sc_w, x, protect_idx=idx)
    y_topk = w8a8_matmul(q_w, sc_w, x, protect_k=4)
    assert (y_static - y_topk).abs().max().item() == 0.0
