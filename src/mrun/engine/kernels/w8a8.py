"""W8A8 (int8 weight x int8 activation) matmul — opt-in fast path for the paged QStore.

The paged engine stores weights per-output-channel int8 (`qrow`) and, every forward,
*dequantizes them to fp32* before a plain `x @ W.T`. That dequant (int8->fp32 widen +
per-row scale multiply) is ~72-79% of the CPU weight-step (cl-0141: the paged bottleneck
is dequant/data-movement, not the matmul flops). W8A8 skips it: quantize the ACTIVATION to
int8 too and do an int8xint8->int32 GEMM, then rescale by (weight_row_scale * act_scale).

Both paths use the SAME int8 weights, so the ONLY numerical difference W8A8 introduces vs
the existing fp32-dequant path is the *activation* quantization. That is the clean parity
target (`w8a8_matmul` vs `QStore.matmul`): argmax must be identical.

The hard part is activation outliers. A few residual-stream channels carry ~10^2-10^3x the
median channel magnitude (measured Qwen2.5-7B: top |act| 12074 vs median 18, 655x). A single
per-tensor int8 scale set by those outliers collapses every other channel to 0. Fix (LLM.int8
style, arXiv:2208.07339): route the outlier columns through an exact fp side-path and quantize
only the rest per-token. Outlier columns are found per-call (top-k by column max-abs) so the
same kernel is correct for every matmul (q/k/v/o/gate/up/down) regardless of its input dim; an
optional static per-block protect-list (from calibration) is also accepted.

Opt-in only: `enable_w8a8(store, ...)` rebinds `store.matmul` on a single instance. It touches
no other file and defaults are never engaged unless a caller flips the store into W8A8 mode.
"""
from __future__ import annotations

import types
from typing import Any

import numpy as np
import torch


def _int8_gemm(q_x: torch.Tensor, q_w: torch.Tensor, use_int_mm: bool) -> torch.Tensor:
    """[T,in] int8 @ [out,in] int8 -> [T,out] int32.

    Prefers ``torch._int_mm`` (tensor-core int8 IMMA on CUDA sm_80+/Ada, a native int32-accumulate
    kernel on CPU).

    CUDA gotcha (measured, RTX 4080 / cuBLASLt): ``_int_mm`` requires the row count M to be
    ``> 16`` AND a multiple of 32 (M=24/40/56 -> CUBLAS_STATUS_NOT_SUPPORTED; 32/64/512 OK). The
    K/N dims of every Qwen/Llama projection are already multiples of 32, so we only pad the token
    rows: append zero rows to the next multiple of 32 (>=32) and slice back — zero rows contribute
    exactly 0 to the int accumulation, so the result is bit-identical.

    Fallback (op missing, e.g. MPS, or shape rejected): CPU widens to int32 (exact). CUDA has no
    integer matmul kernel, so the fallback there widens to fp32 — NOT exact for large K (int8xint8
    summed over K>~2600 exceeds fp32's 2^24 integer range); it should never fire because padding
    makes ``_int_mm`` succeed.
    """
    if use_int_mm and hasattr(torch, "_int_mm") and q_x.device.type in ("cpu", "cuda"):
        try:
            wt = q_w.t().contiguous()
            if q_x.device.type == "cuda":
                M = q_x.shape[0]
                Mp = max(32, ((M + 31) // 32) * 32)
                if Mp != M:
                    pad = torch.zeros((Mp - M, q_x.shape[1]), dtype=torch.int8, device=q_x.device)
                    return torch._int_mm(torch.cat([q_x, pad], 0).contiguous(), wt)[:M]
            return torch._int_mm(q_x.contiguous(), wt)
        except (RuntimeError, NotImplementedError):
            pass  # shape/arch not supported -> fallback
    if q_x.device.type == "cuda":
        return (q_x.to(torch.float32) @ q_w.to(torch.float32).t())  # lossy last resort (see docstring)
    return q_x.to(torch.int32) @ q_w.to(torch.int32).t()


def w8a8_matmul(
    q_w: torch.Tensor,
    sc_w: torch.Tensor,
    x: torch.Tensor,
    *,
    protect_idx: torch.Tensor | None = None,
    protect_k: int = 0,
    use_int_mm: bool = True,
) -> torch.Tensor:
    """Compute ``x @ (q_w * sc_w[:,None]).T`` via an int8xint8 GEMM with an fp32 outlier side-path.

    Args:
        q_w: int8 weight ``[out, in]`` (per-output-channel symmetric).
        sc_w: fp32 per-row scale ``[out]`` (weight dequant = ``q_w * sc_w[:,None]``).
        x: activation ``[..., in]``.
        protect_idx: explicit outlier columns (calibration static list). If None and
            ``protect_k>0``, the top-``protect_k`` columns by batch column-max-abs are protected.
        protect_k: number of outlier columns to route through the exact fp32 side-path.
        use_int_mm: use ``torch._int_mm`` when available.

    Returns: fp32 ``[..., out]`` — the same value the fp32-dequant path would produce, minus the
    per-token quantization error on the *non-protected* activation channels.
    """
    out = int(q_w.shape[0])
    inn = int(q_w.shape[1])
    x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)          # [T, in]
    T = x2.shape[0]
    dev = x2.device
    q_w = q_w.to(dev)
    sc_w = sc_w.to(dev)

    if protect_idx is None and protect_k > 0:
        col_max = x2.abs().amax(dim=0)                         # [in]
        k = min(int(protect_k), inn)
        protect_idx = torch.topk(col_max, k).indices
    if protect_idx is not None:
        protect_idx = torch.as_tensor(protect_idx, dtype=torch.long, device=dev)
        if protect_idx.numel() == 0:
            protect_idx = None

    y = torch.zeros((T, out), dtype=torch.float32, device=dev)

    keep_scale = None
    if protect_idx is not None:
        # Exact fp32 side-path for the outlier columns (weight still int8; activation full-precision).
        Wp = q_w.index_select(1, protect_idx).to(torch.float32) * sc_w[:, None]   # [out, |P|]
        y += x2.index_select(1, protect_idx) @ Wp.T
        # mask that zeroes the protected columns so they don't inflate the per-token act scale
        keep_scale = torch.ones(inn, dtype=torch.float32, device=dev)
        keep_scale[protect_idx] = 0.0

    xq_src = x2 if keep_scale is None else x2 * keep_scale
    act_scale = xq_src.abs().amax(dim=1, keepdim=True) / 127.0                     # [T,1]
    act_scale = torch.where(act_scale == 0, torch.ones_like(act_scale), act_scale)
    q_x = torch.round(xq_src / act_scale).clamp_(-127, 127).to(torch.int8)         # [T, in]

    acc = _int8_gemm(q_x, q_w, use_int_mm)                                         # [T, out] int32
    y += acc.to(torch.float32) * act_scale * sc_w[None, :]                         # rescale
    return y.reshape(*x.shape[:-1], out)


# ---- raw int8 reader + opt-in QStore hook ---------------------------------------------------

def _raw_qrow(store: Any, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a qrow block as (int8 weight [out,in], fp32 row-scale [out]) WITHOUT dequant."""
    b = store._resolve(name)
    out, inn = b["shape"]
    q = np.asarray(store.w[b["w_off"]:b["w_off"] + out * inn], dtype=np.int8).reshape(out, inn)
    so = b["s_off"] // 4
    sc = np.asarray(store.s[so:so + out], dtype=np.float32)
    qt = torch.from_numpy(q.copy())
    sct = torch.from_numpy(sc.copy())
    if getattr(store, "device", "cpu") != "cpu":
        qt = qt.to(store.device)
        sct = sct.to(store.device)
    return qt, sct


def _w8a8_matmul_method(self: Any, name: str, x: torch.Tensor) -> torch.Tensor:
    b = self._resolve(name)
    if b.get("kind") != "qrow":
        return _ORIG_MATMUL[id(self)](name, x)          # fp32 blocks: unchanged base path
    cfg = self._w8a8
    q_w, sc_w = _raw_qrow(self, name)
    pidx = cfg["lists"].get(name)
    return w8a8_matmul(
        q_w, sc_w, x,
        protect_idx=pidx,
        protect_k=cfg["k"],
        use_int_mm=cfg["use_int_mm"],
    )


_ORIG_MATMUL: dict[int, Any] = {}


def enable_w8a8(
    store: Any,
    *,
    protect_k: int = 8,
    protect_lists: dict[str, Any] | None = None,
    use_int_mm: bool = True,
) -> Any:
    """Flip a single QStore instance into W8A8 mode by rebinding ``matmul``. Reversible via
    :func:`disable_w8a8`. ``protect_k`` outlier columns per call go through fp32; the rest
    quantize per-token to int8. Pass ``protect_lists={block_name: idx_tensor}`` to pin a static
    calibration list for specific blocks (overrides top-k for those)."""
    _ORIG_MATMUL[id(store)] = store.matmul
    store._w8a8 = {"k": int(protect_k), "lists": protect_lists or {}, "use_int_mm": bool(use_int_mm)}
    store.matmul = types.MethodType(_w8a8_matmul_method, store)
    return store


def disable_w8a8(store: Any) -> Any:
    """Restore the original fp32-dequant matmul on a store flipped by :func:`enable_w8a8`."""
    orig = _ORIG_MATMUL.pop(id(store), None)
    if orig is not None:
        store.matmul = orig
    if hasattr(store, "_w8a8"):
        del store._w8a8
    return store
