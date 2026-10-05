"""paged_forward.py — run a qwen2/llama/gpt_neox/mamba forward by demand-paging weights
from a :class:`QStore`.

This is the kernel the whole RAM-decoupling claim rests on: a scoring forward that NEVER
holds all parameters resident. Each weight matrix is dequantized from the 8-bit store, used
for exactly one layer, and freed. The mandatory heap working set is therefore O(largest
single matrix), not O(total params) — so a model of any size on disk runs in bounded RAM.

Mechanics parity: paged-int8 logits equal an HF model with the SAME per-output-channel int8
fake-quant applied, up to float reduction order (argmax-exact, max|Δ| ~1.5e-4). Batched twin
``batched_paged_logits`` pays the dominant dequant cost once per weight and runs B sequences
against it (17.5× at B=128, T=32 on Qwen2.5-0.5B — see notes/findings/HW-STACK.md).

Vendored from ram-decoupling/paged_forward.py; the discovery self-test (run/main/PagedModel)
and the store-root subclass were stripped — callers construct ``QStore(name, root=...)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from numbers import Integral
from threading import RLock
from typing import Literal
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F

from .batch_invariant import (
    batch_invariant_attention,
    batch_invariant_matmul,
    batch_invariant_rms_norm,
)
from .qstore import QStore, _row_stable_loaded_matmul

PagedBlockOutputContract = Literal[
    "full_logits",
    "hidden_state_only",
    "selected_token_rows",
]

PagedPooledArithmetic = Literal["packed", "row_stable", "row_stable_split", "batch_invariant"]

_PAGED_BLOCK_OUTPUT_CONTRACTS = frozenset(
    {
        "full_logits",
        "hidden_state_only",
        "selected_token_rows",
    }
)
_PAGED_POOLED_ARITHMETIC = frozenset(
    {"packed", "row_stable", "row_stable_split", "batch_invariant"}
)


@dataclass(frozen=True, slots=True)
class PagedPooledScratchTelemetry:
    """Byte accounting for pooled attention source assembly.

    The first two fields report logical pre-GQA source sizes, making the split lane's
    ``O(B * max(prefix))`` versus ``O(max(single prefix))`` distinction explicit.
    ``explicit_live_prefix_kv_peak_bytes`` is measured from the actual live K/V source
    tensors at allocation boundaries and includes GQA expansion plus exact-layout clones.
    The aggregate provisional delta is accounted separately.
    """

    arithmetic: PagedPooledArithmetic
    batch_size: int
    token_count: int
    parent_lengths: tuple[int, ...]
    global_prefix_kv_logical_bytes: int
    request_local_prefix_kv_logical_bytes_max: int
    explicit_live_prefix_kv_peak_bytes: int
    aggregate_provisional_delta_bytes: int


# =================================================================== qwen2/llama math
def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(var + eps))


def _streamed_lm_head_matmul(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Multiply hidden rows by one streamed unembedding block.

    Some paged paths keep the final normalized hidden state in FP32 while QStore streams
    the unembedding block in the requested BF16/FP16 compute dtype.  Torch does not promote
    matmul operands, so align the short-lived block with the hidden rows before multiplying.
    """

    if weight.dtype != hidden.dtype:
        weight = weight.to(dtype=hidden.dtype)
    return hidden @ weight.T


def _row_stable_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Preserve the exact singleton-request RMS reduction shape for every pooled row."""

    return torch.cat(
        tuple(_rms_norm(x[row : row + 1], w, eps) for row in range(int(x.shape[0]))),
        dim=0,
    )


def _rope_tables(T: int, hd: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))  # [hd/2]
    pos = torch.arange(T, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, hd]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [T, n, hd] ; cos/sin: [T, hd]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return x * cos + _rotate_half(x) * sin


def _apply_patch_ops(x: torch.Tensor, ops: list | tuple) -> torch.Tensor:
    """Apply the replay patch tuple contract to a paged activation tensor.

    The binding/replay tools represent interventions as ``(op, cols, vals)``
    tuples keyed by layer. This mirrors their hook-time semantics at the
    paged-engine tap point: the MLP activation immediately before ``down``.
    ``x`` is usually [T, intermediate] here, while the HF hook path sees
    [B, T, intermediate]; the indexing below preserves the same last-axis
    column behavior.
    """
    for op, cols_t, vals in ops:
        cols = torch.as_tensor(cols_t, dtype=torch.long, device=x.device)
        if op == "zero":
            x[..., cols] = 0.0
        elif op == "scale":
            x[..., cols] = x[..., cols] * float(vals)
        elif op == "global_mean":
            v = vals.to(device=x.device, dtype=x.dtype)  # type: ignore[union-attr]
            x[..., cols] = v
        elif op == "position_mean":
            v = vals.to(device=x.device, dtype=x.dtype)  # type: ignore[union-attr]
            if x.ndim == 2:
                x[..., cols] = v[: x.shape[0]]
            else:
                x[..., cols] = v[: x.shape[1]].unsqueeze(0)
        elif op == "add_amp":
            v = vals.to(device=x.device, dtype=x.dtype)  # type: ignore[union-attr]
            x[..., cols] = x[..., cols] + v
        elif op == "center":
            x[..., cols] = x[..., cols] - x[..., cols].mean(dim=-1, keepdim=True)
        else:
            raise ValueError(f"unknown patch op {op}")
    return x


def _compile_row_patch_maps(
    rows: Sequence[Mapping[int, list | tuple] | None] | None,
    *,
    batch_size: int,
) -> dict[int, dict[int, list | tuple]]:
    """Transpose row-local replay maps into one layer-local dispatch table."""

    if rows is None:
        return {}
    if len(rows) != batch_size:
        raise ValueError("row-local patch maps must align with the paged batch")
    by_layer: dict[int, dict[int, list | tuple]] = {}
    for row_index, row in enumerate(rows):
        if row is None:
            continue
        if not isinstance(row, Mapping):
            raise TypeError("each row-local patch map must be a mapping or None")
        for raw_layer, ops in row.items():
            if isinstance(raw_layer, bool) or not isinstance(raw_layer, Integral):
                raise TypeError("row-local patch layers must be integers")
            layer = int(raw_layer)
            if layer < 0:
                raise ValueError("row-local patch layers must be non-negative")
            if ops:
                by_layer.setdefault(layer, {})[row_index] = ops
    return by_layer


def _apply_patch_ops_by_row(
    x: torch.Tensor,
    row_ops: Mapping[int, list | tuple],
) -> torch.Tensor:
    """Apply distinct MLP edits to selected rows without contaminating neighbors."""

    if x.ndim < 2:
        raise ValueError("row-local patches require a batched activation tensor")
    if not row_ops:
        return x
    out = x.clone()
    for row_index, ops in row_ops.items():
        if not 0 <= row_index < x.shape[0]:
            raise IndexError("row-local patch index is outside the activation batch")
        out[row_index] = _apply_patch_ops(x[row_index].clone(), ops)
    return out


def _apply_head_patch_ops_by_row(
    x: torch.Tensor,
    row_ops: Mapping[int, list | tuple],
) -> torch.Tensor:
    """Apply distinct attention-head edits to selected batch rows."""

    if x.ndim != 4:
        raise ValueError("row-local head patches require [B,T,H,D] activations")
    out = x.clone()
    for row_index, ops in row_ops.items():
        if not 0 <= row_index < x.shape[0]:
            raise IndexError("row-local head patch index is outside the activation batch")
        out[row_index] = _apply_head_patch_ops(x[row_index].clone(), ops)
    return out


def _apply_projection_patch_ops(
    x: torch.Tensor,
    ops: list | tuple,
) -> torch.Tensor:
    """Replace selected token positions in a raw K/V projection tensor."""

    if x.ndim not in {2, 3}:
        raise ValueError("K/V projection patches require [T,D] or [B,T,D] activations")
    for op, positions_t, values in ops:
        if op != "position_replace":
            raise ValueError(f"unknown K/V projection patch op {op}")
        positions = torch.as_tensor(positions_t, dtype=torch.long, device=x.device)
        if positions.numel() == 0:
            raise ValueError("K/V projection replacement requires token positions")
        if int(positions.min()) < 0 or int(positions.max()) >= x.shape[-2]:
            raise IndexError("K/V projection token position is outside the activation")
        donor = torch.as_tensor(values, dtype=x.dtype, device=x.device)
        if x.ndim == 2:
            if donor.ndim == 3 and donor.shape[0] == 1:
                donor = donor[0]
            if donor.ndim != 2 or donor.shape[-1] != x.shape[-1]:
                raise ValueError("K/V donor must have shape [T,D] or [P,D]")
            selected = donor.index_select(0, positions) if donor.shape[0] == x.shape[0] else donor
            if tuple(selected.shape) != (positions.numel(), x.shape[-1]):
                raise ValueError("K/V donor rows do not align with selected token positions")
            x[positions] = selected
            continue
        if donor.ndim == 2:
            donor = donor.unsqueeze(0).expand(x.shape[0], -1, -1)
        if donor.ndim != 3 or donor.shape[0] != x.shape[0] or donor.shape[-1] != x.shape[-1]:
            raise ValueError("batched K/V donor must have shape [B,T,D] or [T,D]")
        selected = donor.index_select(1, positions) if donor.shape[1] == x.shape[1] else donor
        if tuple(selected.shape) != (x.shape[0], positions.numel(), x.shape[-1]):
            raise ValueError("batched K/V donor rows do not align with token positions")
        x[:, positions] = selected
    return x


def _apply_projection_patch_ops_by_row(
    x: torch.Tensor,
    row_ops: Mapping[int, list | tuple],
) -> torch.Tensor:
    """Apply distinct raw K/V projection replacements to selected batch rows."""

    if x.ndim != 3:
        raise ValueError("row-local K/V projection patches require [B,T,D] activations")
    out = x.clone()
    for row_index, ops in row_ops.items():
        if not 0 <= row_index < x.shape[0]:
            raise IndexError("row-local K/V patch index is outside the activation batch")
        out[row_index] = _apply_projection_patch_ops(x[row_index].clone(), ops)
    return out


def _apply_resid_patch_ops_by_row(
    x: torch.Tensor,
    row_ops: Mapping[int, list | tuple],
) -> torch.Tensor:
    """Apply distinct residual-stream edits to selected batch rows."""

    if x.ndim != 3:
        raise ValueError("row-local residual patches require [B,T,D] activations")
    out = x.clone()
    for row_index, ops in row_ops.items():
        if not 0 <= row_index < x.shape[0]:
            raise IndexError("row-local residual patch index is outside the activation batch")
        out[row_index] = _apply_resid_patch_ops(x[row_index].clone(), ops)
    return out


def _apply_resid_patch_ops(h: torch.Tensor, ops: list | tuple) -> torch.Tensor:
    """Apply replay-style patches to the RESIDUAL stream at a layer's output.

    ``h`` is [T, d] (or [B, T, d] in the batched path). This is the tap the dense
    causal-use legs hook via ``register_forward_hook`` on a decoder layer (output[0]),
    i.e. AFTER attn+MLP have been added back. ``("proj_remove", u, None)`` removes the
    rank-1 projection onto unit direction ``u`` (h -> h - (h·u)u) — the size-matched
    direction-removal control. ``u`` may be any nonzero vector; it is re-normalized here.
    """
    for op, vec, values in ops:
        if op == "proj_remove":
            u = torch.as_tensor(vec, dtype=torch.float32, device=h.device)
            u = u / (u.norm() + 1e-9)
            proj = (h.float() @ u).unsqueeze(-1) * u
            h = (h.float() - proj).to(h.dtype)
        elif op == "position_replace":
            if h.ndim not in {2, 3}:
                raise ValueError(
                    "residual position replacement requires [T,D] or [B,T,D] activations"
                )
            positions = torch.as_tensor(vec, dtype=torch.long, device=h.device)
            if positions.numel() == 0:
                raise ValueError("residual position replacement requires token positions")
            if int(positions.min()) < 0 or int(positions.max()) >= h.shape[-2]:
                raise IndexError("residual replacement position is outside the activation")
            donor = torch.as_tensor(values, dtype=h.dtype, device=h.device)
            if h.ndim == 2:
                if donor.ndim == 1:
                    donor = donor.unsqueeze(0)
                if tuple(donor.shape) != (positions.numel(), h.shape[-1]):
                    raise ValueError("residual donor rows do not align with token positions")
                h[positions] = donor
            else:
                if donor.ndim == 2:
                    donor = donor.unsqueeze(0).expand(h.shape[0], -1, -1)
                if tuple(donor.shape) != (
                    h.shape[0],
                    positions.numel(),
                    h.shape[-1],
                ):
                    raise ValueError(
                        "batched residual donor rows do not align with token positions"
                    )
                h[:, positions] = donor
        elif op == "position_lerp":
            if h.ndim not in {2, 3}:
                raise ValueError("residual position lerp requires [T,D] or [B,T,D] activations")
            positions = torch.as_tensor(vec, dtype=torch.long, device=h.device)
            if positions.numel() == 0:
                raise ValueError("residual position lerp requires token positions")
            if int(positions.min()) < 0 or int(positions.max()) >= h.shape[-2]:
                raise IndexError("residual lerp position is outside the activation")
            if not isinstance(values, Mapping):
                raise ValueError("residual position lerp requires a value/dose mapping")
            dose = float(values.get("dose", -1.0))
            if not math.isfinite(dose) or dose < 0.0 or dose > 1.0:
                raise ValueError("residual position lerp dose must lie in [0, 1]")
            donor = torch.as_tensor(values.get("value"), dtype=h.dtype, device=h.device)
            if h.ndim == 2:
                if donor.ndim == 1:
                    donor = donor.unsqueeze(0)
                if tuple(donor.shape) != (positions.numel(), h.shape[-1]):
                    raise ValueError("residual lerp donor rows do not align")
                h[positions] = h[positions] + dose * (donor - h[positions])
            else:
                if donor.ndim == 2:
                    donor = donor.unsqueeze(0).expand(h.shape[0], -1, -1)
                if tuple(donor.shape) != (
                    h.shape[0],
                    positions.numel(),
                    h.shape[-1],
                ):
                    raise ValueError("batched residual lerp donor rows do not align")
                h[:, positions] = h[:, positions] + dose * (donor - h[:, positions])
        elif op == "position_add":
            if h.ndim not in {2, 3}:
                raise ValueError("residual position add requires [T,D] or [B,T,D] activations")
            positions = torch.as_tensor(vec, dtype=torch.long, device=h.device)
            if positions.numel() == 0:
                raise ValueError("residual position add requires token positions")
            if int(positions.min()) < 0 or int(positions.max()) >= h.shape[-2]:
                raise IndexError("residual add position is outside the activation")
            delta = torch.as_tensor(values, dtype=h.dtype, device=h.device)
            if h.ndim == 2:
                if delta.ndim == 1:
                    delta = delta.unsqueeze(0)
                if tuple(delta.shape) != (positions.numel(), h.shape[-1]):
                    raise ValueError("residual add rows do not align with token positions")
                h[positions] = h[positions] + delta
            else:
                if delta.ndim == 2:
                    delta = delta.unsqueeze(0).expand(h.shape[0], -1, -1)
                if tuple(delta.shape) != (
                    h.shape[0],
                    positions.numel(),
                    h.shape[-1],
                ):
                    raise ValueError("batched residual add rows do not align")
                h[:, positions] = h[:, positions] + delta
        else:
            raise ValueError(f"unknown resid patch op {op}")
    return h


def _apply_head_patch_ops(x: torch.Tensor, ops: list | tuple) -> torch.Tensor:
    """Apply replay-style patches to per-head attention outputs before ``o_proj``.

    ``x`` is either [T, nH, hd] or [B, T, nH, hd]. The patched axis is the
    head axis, so ``("zero", heads, None)`` removes the complete value-flow
    vector for those heads before the output projection mixes heads back into
    the residual stream.
    """
    for op, heads_t, vals in ops:
        heads = torch.as_tensor(heads_t, dtype=torch.long, device=x.device)
        if op == "zero":
            x[..., heads, :] = 0.0
        elif op == "scale":
            x[..., heads, :] = x[..., heads, :] * float(vals)
        elif op == "global_mean":
            v = vals.to(device=x.device, dtype=x.dtype)  # type: ignore[union-attr]
            x[..., heads, :] = v
        elif op == "add_amp":
            v = vals.to(device=x.device, dtype=x.dtype)  # type: ignore[union-attr]
            x[..., heads, :] = x[..., heads, :] + v
        else:
            raise ValueError(f"unknown head patch op {op}")
    return x


@torch.no_grad()
def paged_logits(
    store: QStore,
    input_ids: np.ndarray,
    collect_hs: list | None = None,
    collect_acts: list | None = None,
    collect_attn: list | None = None,
    patch_ops_by_layer: dict[int, list] | None = None,
    capture_selected_maps: dict[int, dict] | None = None,
    captured_selected: dict[int, torch.Tensor] | None = None,
    head_patch_ops_by_layer: dict[int, list] | None = None,
    key_patch_ops_by_layer: dict[int, list] | None = None,
    value_patch_ops_by_layer: dict[int, list] | None = None,
    collect_key_out: list | None = None,
    collect_value_out: list | None = None,
    collect_head_out: list | None = None,
    resid_patch_ops_by_layer: dict[int, list] | None = None,
) -> torch.Tensor:
    """Full causal forward over a single sequence; returns logits [T, V]. Streams every weight.
    If collect_hs is a list, append the residual `h` after embed and after each layer (debug).
    If collect_attn is a list, append the per-head softmax attention `probs` [nH, T, T] for each
    layer — the same matrix HF returns via output_attentions (minus the batch dim), so a paged
    attention probe (sink/routing physiology) is directly comparable to a live HF one. The matrix
    is already materialized here (it is otherwise just freed), so exposing it is near-free.
    If collect_acts is a list, append the per-layer MLP activation `silu(gate)*up` [T, inter]
    (the down-projection input) for each layer — the same tap the physiology recorder hooks
    via arch.add_act_hook, so a paged tape is directly comparable to a live HF tape.

    ``patch_ops_by_layer`` accepts the existing replay hook tuple contract:
    ``{layer: [(op, cols, vals), ...]}``. Patches are applied at the same
    semantic point as the HF pre-hook on ``down_proj``: after the gated MLP
    activation is formed and before it is multiplied by ``down``.

    ``capture_selected_maps`` accepts ``{layer: {"locals": [...]}}``. When
    supplied with ``captured_selected``, the paged engine stores post-patch
    selected activations as float16 CPU tensors keyed by layer, matching the
    worker's replay-recorder sidecar expectation.
    """
    c = store.cfg
    d, nL = c["hidden_size"], c["num_hidden_layers"]
    nH, nKV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps, theta = c["rms_norm_eps"], c["rope_theta"]
    T = len(input_ids)
    rep = nH // nKV
    scale = hd**-0.5

    h = store.embed_rows("embed", input_ids).clone()  # [T, d] on store.device
    dev = h.device  # follows the store (cpu default)
    if collect_hs is not None:
        collect_hs.append(h.clone().cpu())
    cos, sin = _rope_tables(T, hd, theta)
    cos, sin = cos.to(dev), sin.to(dev)
    causal = torch.triu(torch.full((T, T), float("-inf"), device=dev), diagonal=1)

    for L in range(nL):
        x = _rms_norm(h, store.fp32(f"L{L}.ln1"), eps)
        q = store.matmul(f"L{L}.q", x)
        k = store.matmul(f"L{L}.k", x)
        v = store.matmul(f"L{L}.v", x)
        if store.has(f"L{L}.q.bias"):
            q = q + store.fp32(f"L{L}.q.bias")
            k = k + store.fp32(f"L{L}.k.bias")
            v = v + store.fp32(f"L{L}.v.bias")
        if collect_key_out is not None:
            collect_key_out.append(k.detach().clone().cpu())
        if collect_value_out is not None:
            collect_value_out.append(v.detach().clone().cpu())
        if key_patch_ops_by_layer and L in key_patch_ops_by_layer:
            k = _apply_projection_patch_ops(k, key_patch_ops_by_layer[L])
        if value_patch_ops_by_layer and L in value_patch_ops_by_layer:
            v = _apply_projection_patch_ops(v, value_patch_ops_by_layer[L])
        q = q.view(T, nH, hd)
        k = k.view(T, nKV, hd)
        v = v.view(T, nKV, hd)
        if store.has(f"L{L}.q_norm"):  # qwen3: per-head RMSNorm on q/k
            q = _rms_norm(
                q, store.fp32(f"L{L}.q_norm"), eps
            )  # over head_dim, BEFORE RoPE (HF order)
            k = _rms_norm(k, store.fp32(f"L{L}.k_norm"), eps)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if rep > 1:  # GQA expand
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # attention per head: [nH, T, hd]
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale + causal  # [nH,T,T]
        probs = torch.softmax(scores, dim=-1)
        if collect_attn is not None:
            collect_attn.append(probs.detach().clone().cpu())  # [nH, T, T]
        # Keep the softmax in FP32 for stability; narrowed stores provide BF16/FP16
        # values, so widen the small attention value tensor at this mixed-dtype seam.
        head_ctx = torch.matmul(probs, vh.float()).transpose(0, 1)  # [T, nH, hd]
        if head_patch_ops_by_layer and L in head_patch_ops_by_layer:
            head_ctx = _apply_head_patch_ops(head_ctx, head_patch_ops_by_layer[L])
        if collect_head_out is not None:
            collect_head_out.append(head_ctx.detach().clone().cpu())  # [T, nH, hd]
        ctx = head_ctx.contiguous().reshape(T, nH * hd)  # [T, d]
        attn_out = store.matmul(f"L{L}.o", ctx)
        if store.has(f"L{L}.o.bias"):
            attn_out = attn_out + store.fp32(f"L{L}.o.bias")
        del q, k, v, qh, kh, vh, scores, probs, head_ctx, ctx
        h = h + attn_out

        x2 = _rms_norm(h, store.fp32(f"L{L}.ln2"), eps)
        g = store.matmul(f"L{L}.gate", x2)
        u = store.matmul(f"L{L}.up", x2)
        hid = torch.nn.functional.silu(g) * u
        if patch_ops_by_layer and L in patch_ops_by_layer:
            hid = _apply_patch_ops(hid, patch_ops_by_layer[L])
        if capture_selected_maps and captured_selected is not None and L in capture_selected_maps:
            locals_ = capture_selected_maps[L].get("locals", [])
            if locals_:
                cols = torch.as_tensor(locals_, dtype=torch.long, device=hid.device)
                captured_selected[L] = hid[..., cols].detach().to(dtype=torch.float16, device="cpu")
        if collect_acts is not None:
            collect_acts.append(hid.detach().clone().cpu())  # [T, inter] post-gate activation
        h = h + store.matmul(f"L{L}.down", hid)
        del g, u, hid
        if resid_patch_ops_by_layer and L in resid_patch_ops_by_layer:
            h = _apply_resid_patch_ops(h, resid_patch_ops_by_layer[L])
        if collect_hs is not None:
            collect_hs.append(h.clone().cpu())

    h = _rms_norm(h, store.fp32("norm.final"), eps)
    V = c["vocab_size"]
    logits = torch.empty((T, V), dtype=torch.float32, device=dev)
    for start, end, Wblk in store.row_blocks("lm_head"):  # stream the unembedding
        logits[:, start:end] = _streamed_lm_head_matmul(h, Wblk)
        del Wblk
    return logits.cpu()  # nothing device-resident escapes


# =================================================================== qwen3.5 hybrid math
def _qwen35_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5's zero-centered RMSNorm, kept separate from the older Qwen norm."""

    dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
    return (x32 * (1.0 + w.float())).to(dtype)


def _qwen35_head_rms_norm(
    x: torch.Tensor, w: torch.Tensor, eps: float, head_dim: int
) -> torch.Tensor:
    shape = x.shape
    if shape[-1] % head_dim:
        raise ValueError("Qwen3.5 head RMSNorm width is not divisible by head_dim")
    return _qwen35_rms_norm(x.reshape(*shape[:-1], -1, head_dim), w, eps).reshape(shape)


def _qwen35_apply_mrope(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    positions: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Text-only Qwen3.5 partial mRoPE with device-local tables."""

    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("Qwen3.5 partial rotary dimension is invalid")
    q = query.reshape(query.shape[0], query.shape[1], -1, head_dim).transpose(1, 2)
    k = key.reshape(key.shape[0], key.shape[1], -1, head_dim).transpose(1, 2)
    indices = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=query.device)
    frequencies = 1.0 / (float(rope_theta) ** (indices / rotary_dim))
    angles = torch.outer(positions.to(torch.float32), frequencies)
    embedding = torch.cat((angles, angles), dim=-1)
    cosine = embedding.cos().to(q.dtype)[None, None]
    sine = embedding.sin().to(q.dtype)[None, None]

    def rotate(value: torch.Tensor) -> torch.Tensor:
        rotary = value[..., :rotary_dim]
        tail = value[..., rotary_dim:]
        return torch.cat((rotary * cosine + _rotate_half(rotary) * sine, tail), dim=-1)

    return rotate(q).transpose(1, 2).reshape(query.shape), rotate(k).transpose(1, 2).reshape(
        key.shape
    )


def _qwen35_gated_delta_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-local exact GDN recurrence; state is deliberately float32."""

    batch, sequence, _ = query.shape
    key_width = num_key_heads * key_head_dim
    value_width = num_value_heads * value_head_dim
    if query.shape[-1] != key_width or key.shape != query.shape:
        raise ValueError("Qwen3.5 GDN query/key geometry is invalid")
    if value.shape != (batch, sequence, value_width):
        raise ValueError("Qwen3.5 GDN value geometry is invalid")
    if a.shape != (batch, sequence, num_value_heads) or b.shape != a.shape:
        raise ValueError("Qwen3.5 GDN gate geometry is invalid")

    q = query.reshape(batch, sequence, num_key_heads, key_head_dim)
    k = key.reshape(batch, sequence, num_key_heads, key_head_dim)
    v = value.reshape(batch, sequence, num_value_heads, value_head_dim)
    q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    repeats = num_value_heads // num_key_heads
    if repeats > 1:
        q = q.repeat_interleave(repeats, dim=2)
        k = k.repeat_interleave(repeats, dim=2)
    q = q.float() * (key_head_dim**-0.5)
    k = k.float()
    v = v.float()
    beta = torch.sigmoid(b.float())
    decay = -torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float())
    state_shape = (batch, num_value_heads, key_head_dim, value_head_dim)
    state = (
        torch.zeros(state_shape, dtype=torch.float32, device=query.device)
        if initial_state is None
        else initial_state.to(device=query.device, dtype=torch.float32).clone()
    )
    if tuple(state.shape) != state_shape:
        raise ValueError("Qwen3.5 GDN recurrent state shape is invalid")
    output = torch.empty(
        (batch, sequence, num_value_heads, value_head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    for position in range(sequence):
        q_t = q[:, position]
        k_t = k[:, position]
        v_t = v[:, position]
        state = state * decay[:, position].exp().unsqueeze(-1).unsqueeze(-1)
        memory = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - memory) * beta[:, position].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, position] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    return output.to(query.dtype).reshape(batch, sequence, value_width), state


def _qwen35_gated_norm(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    head_dim: int,
) -> torch.Tensor:
    if value.shape != gate.shape or value.shape[-1] % head_dim:
        raise ValueError("Qwen3.5 gated RMSNorm geometry is invalid")
    shape = value.shape
    value_2d = value.reshape(-1, head_dim).float()
    gate_2d = gate.reshape(-1, head_dim).float()
    normalized = value_2d * torch.rsqrt(value_2d.square().mean(dim=-1, keepdim=True) + eps)
    normalized = normalized * weight.float() * F.silu(gate_2d)
    return normalized.to(value.dtype).reshape(shape)


@dataclass(slots=True)
class Qwen35PagedState:
    """Resident typed state for one or more equal-position Qwen3.5 requests.

    GDN convolution history, float32 recurrent matrices, and full-attention K/V are separate
    allocations by design.  This is the state boundary Saturn's component trace describes;
    none of the three is treated as a generic KV cache.
    """

    capacity: int
    pos: int
    batch_size: int
    dtype: torch.dtype
    device: torch.device
    conv: dict[int, torch.Tensor]
    recurrent: dict[int, torch.Tensor]
    key: dict[int, torch.Tensor]
    value: dict[int, torch.Tensor]

    @classmethod
    def create(cls, store: QStore, capacity: int, *, batch_size: int = 1) -> Qwen35PagedState:
        if capacity <= 0:
            raise ValueError("Qwen3.5 state capacity must be positive")
        if batch_size <= 0:
            raise ValueError("Qwen3.5 state batch_size must be positive")
        c = store.cfg
        device = torch.device(store.device)
        dtype = store.compute_dtype
        state = cls(
            capacity=int(capacity),
            pos=0,
            batch_size=int(batch_size),
            dtype=dtype,
            device=device,
            conv={},
            recurrent={},
            key={},
            value={},
        )
        for layer, layer_type in enumerate(c["layer_types"]):
            if layer_type == "linear_attention":
                state.conv[layer] = torch.zeros(
                    (
                        int(batch_size),
                        int(c["linear_conv_width"]),
                        int(c["linear_conv_kernel_dim"]),
                    ),
                    dtype=dtype,
                    device=device,
                )
                state.recurrent[layer] = torch.zeros(
                    (
                        int(batch_size),
                        int(c["linear_num_value_heads"]),
                        int(c["linear_key_head_dim"]),
                        int(c["linear_value_head_dim"]),
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            elif layer_type == "full_attention":
                shape = (
                    int(batch_size),
                    int(capacity),
                    int(c["num_key_value_heads"]),
                    int(c["head_dim"]),
                )
                state.key[layer] = torch.empty(shape, dtype=dtype, device=device)
                state.value[layer] = torch.empty_like(state.key[layer])
            else:
                raise ValueError(f"unknown Qwen3.5 layer type {layer_type!r}")
        return state


def _qwen35_body(
    store: QStore,
    input_ids: np.ndarray,
    state: Qwen35PagedState,
) -> torch.Tensor:
    ids = np.asarray(input_ids, dtype=np.int64)
    squeeze_batch = ids.ndim == 1
    if squeeze_batch:
        ids = ids[None, :]
    if ids.ndim != 2 or ids.size == 0 or ids.shape[1] == 0:
        raise ValueError("Qwen3.5 paged input_ids must be a non-empty rank-one or rank-two array")
    c = store.cfg
    dtype = state.dtype
    hidden = store.embed_rows("embed", ids).to(dtype)
    batch, sequence, _ = hidden.shape
    if batch != state.batch_size:
        raise ValueError(
            f"Qwen3.5 paged state batch {state.batch_size} does not match input batch {batch}"
        )
    start = state.pos
    end = start + sequence
    if end > state.capacity:
        raise RuntimeError(f"Qwen3.5 state overflow: {end} > {state.capacity}")
    eps = float(c["rms_norm_eps"])
    key_heads = int(c["linear_num_key_heads"])
    value_heads = int(c["linear_num_value_heads"])
    key_dim = int(c["linear_key_head_dim"])
    value_dim = int(c["linear_value_head_dim"])
    conv_kernel = int(c["linear_conv_kernel_dim"])
    rotary_dim = int(int(c["head_dim"]) * float(c["partial_rotary_factor"]))
    positions = torch.arange(start, end, dtype=torch.long, device=hidden.device)
    layer_types = tuple(c["layer_types"])
    for layer, layer_type in enumerate(layer_types):
        residual = hidden
        normed = _qwen35_rms_norm(hidden, store.fp32(f"L{layer}.ln1"), eps)
        if layer_type == "linear_attention":
            packed = store.matmul(f"L{layer}.in_proj_qkv", normed)
            old = state.conv[layer]
            sequence_values = torch.cat((old[:, :, 1:], packed.transpose(1, 2)), dim=-1)
            conv_weight = store.fp32(f"L{layer}.conv1d.weight").to(dtype)
            convolved = F.conv1d(
                sequence_values,
                conv_weight,
                groups=sequence_values.shape[1],
            )[:, :, -sequence:]
            state.conv[layer] = sequence_values[:, :, -conv_kernel:].contiguous()
            convolved = F.silu(convolved.transpose(1, 2))
            key_width = key_heads * key_dim
            value_width = value_heads * value_dim
            query, key, value = convolved.split((key_width, key_width, value_width), dim=-1)
            z = store.matmul(f"L{layer}.in_proj_z", normed)
            a = store.matmul(f"L{layer}.in_proj_a", normed)
            b = store.matmul(f"L{layer}.in_proj_b", normed)
            core, recurrent = _qwen35_gated_delta_recurrence(
                query,
                key,
                value,
                a,
                b,
                a_log=store.fp32(f"L{layer}.A_log"),
                dt_bias=store.fp32(f"L{layer}.dt_bias"),
                num_key_heads=key_heads,
                num_value_heads=value_heads,
                key_head_dim=key_dim,
                value_head_dim=value_dim,
                initial_state=state.recurrent[layer],
            )
            state.recurrent[layer] = recurrent
            core = _qwen35_gated_norm(
                core,
                z,
                store.fp32(f"L{layer}.gdn_norm"),
                eps=eps,
                head_dim=value_dim,
            )
            mixer_output = store.matmul(f"L{layer}.out_proj", core)
        elif layer_type == "full_attention":
            packed_query = store.matmul(f"L{layer}.q", normed)
            query, gate = packed_query.reshape(batch, sequence, -1, 2 * int(c["head_dim"])).chunk(
                2, dim=-1
            )
            query = query.reshape(batch, sequence, -1)
            gate = gate.reshape(batch, sequence, -1)
            key = store.matmul(f"L{layer}.k", normed)
            value = store.matmul(f"L{layer}.v", normed)
            query = _qwen35_head_rms_norm(
                query, store.fp32(f"L{layer}.q_norm"), eps, int(c["head_dim"])
            )
            key = _qwen35_head_rms_norm(
                key, store.fp32(f"L{layer}.k_norm"), eps, int(c["head_dim"])
            )
            query, key = _qwen35_apply_mrope(
                query,
                key,
                positions=positions,
                head_dim=int(c["head_dim"]),
                rotary_dim=rotary_dim,
                rope_theta=float(c["rope_theta"]),
            )
            query = query.reshape(
                batch, sequence, int(c["num_attention_heads"]), int(c["head_dim"])
            )
            key = key.reshape(batch, sequence, int(c["num_key_value_heads"]), int(c["head_dim"]))
            value = value.reshape(
                batch, sequence, int(c["num_key_value_heads"]), int(c["head_dim"])
            )
            state.key[layer][:, start:end] = key
            state.value[layer][:, start:end] = value
            keys = state.key[layer][:, :end]
            values = state.value[layer][:, :end]
            repeats = int(c["num_attention_heads"]) // int(c["num_key_value_heads"])
            if repeats > 1:
                keys = keys.repeat_interleave(repeats, dim=2)
                values = values.repeat_interleave(repeats, dim=2)
            qh = query.transpose(1, 2)
            kh = keys.transpose(1, 2)
            vh = values.transpose(1, 2)
            scores = torch.matmul(qh, kh.transpose(-1, -2)) * (int(c["head_dim"]) ** -0.5)
            qpos = positions.to(hidden.device)[None, None, :, None]
            kpos = torch.arange(end, device=hidden.device)[None, None, None, :]
            scores = scores.masked_fill(kpos > qpos, float("-inf"))
            probs = torch.softmax(scores.float(), dim=-1).to(dtype)
            context = torch.matmul(probs, vh).transpose(1, 2).reshape(batch, sequence, -1)
            if bool(c.get("attn_output_gate", True)):
                context = context * torch.sigmoid(gate)
            mixer_output = store.matmul(f"L{layer}.o", context)
        else:
            raise ValueError(f"unknown Qwen3.5 layer type {layer_type!r}")
        hidden = residual + mixer_output
        residual = hidden
        normed = _qwen35_rms_norm(hidden, store.fp32(f"L{layer}.ln2"), eps)
        gate = store.matmul(f"L{layer}.gate", normed)
        up = store.matmul(f"L{layer}.up", normed)
        mlp = F.silu(gate) * up
        hidden = residual + store.matmul(f"L{layer}.down", mlp)
    state.pos = end
    return hidden.squeeze(0) if squeeze_batch else hidden


def _qwen35_stream_head(store: QStore, hidden: torch.Tensor, *, last_only: bool) -> torch.Tensor:
    normalized = _qwen35_rms_norm(
        hidden, store.fp32("norm.final"), float(store.cfg["rms_norm_eps"])
    )
    if last_only:
        normalized = normalized[-1:] if normalized.ndim == 2 else normalized[:, -1:]
    batched = normalized.ndim == 3
    if normalized.ndim not in (2, 3):
        raise ValueError("Qwen3.5 streamed head expects [T,H] or [B,T,H] hidden states")
    rows = int(normalized.shape[0] if not batched else normalized.shape[0] * normalized.shape[1])
    vocab = int(store.cfg["vocab_size"])
    logits = torch.empty((rows, vocab), dtype=torch.float32, device=normalized.device)
    for start, end, weight in store.row_blocks("lm_head"):
        block = _streamed_lm_head_matmul(normalized.reshape(rows, -1), weight)
        logits[:, start:end] = block.float()
        del block, weight
    if batched:
        logits = logits.reshape(normalized.shape[0], normalized.shape[1], vocab)
        if last_only:
            logits = logits[:, 0]
    return logits.cpu()


@torch.no_grad()
def paged_logits_qwen35(
    store: QStore,
    input_ids: np.ndarray,
    **_unused: object,
) -> torch.Tensor:
    """Lossless paged Qwen3.5 forward over the exact hybrid state machine."""

    ids = np.asarray(input_ids, dtype=np.int64)
    state = Qwen35PagedState.create(store, max(1, int(ids.size)))
    hidden = _qwen35_body(store, ids, state)
    return _qwen35_stream_head(store, hidden, last_only=False)


@torch.no_grad()
def paged_logits_qwen35_kv(
    store: QStore,
    input_ids: np.ndarray,
    state: Qwen35PagedState,
) -> torch.Tensor:
    """Prefill or decode one request using persistent convolution/GDN/full-KV state."""

    hidden = _qwen35_body(store, np.asarray(input_ids, dtype=np.int64), state)
    return _qwen35_stream_head(store, hidden, last_only=True)[0]


@torch.no_grad()
def paged_logits_qwen35_kv_batch(
    store: QStore,
    input_ids: np.ndarray,
    state: Qwen35PagedState,
) -> torch.Tensor:
    """Run one equal-position Qwen3.5 prefill/decode step for a batch of requests.

    The hybrid convolution, recurrent matrix, and full-attention KV state all carry a batch
    dimension.  Equal-length rows share one streamed weight traversal; callers bucket ragged
    prompts before entering this kernel.
    """

    ids = np.asarray(input_ids, dtype=np.int64)
    if ids.ndim != 2 or ids.shape[0] != state.batch_size or ids.shape[1] == 0:
        raise ValueError("Qwen3.5 batched KV input must have shape [batch, tokens]")
    hidden = _qwen35_body(store, ids, state)
    return _qwen35_stream_head(store, hidden, last_only=True)


@torch.no_grad()
def paged_logits_qwen35_batch(
    store: QStore,
    input_ids: np.ndarray,
    *,
    last_only: bool = False,
) -> torch.Tensor:
    """Run a stateless equal-length Qwen3.5 batch through one weight stream."""

    ids = np.asarray(input_ids, dtype=np.int64)
    if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0:
        raise ValueError("Qwen3.5 batched input must have shape [batch, tokens]")
    state = Qwen35PagedState.create(store, int(ids.shape[1]), batch_size=int(ids.shape[0]))
    hidden = _qwen35_body(store, ids, state)
    return _qwen35_stream_head(store, hidden, last_only=last_only)


# =================================================================== qwen2/llama KV DECODE
def _rope_tables_at(positions: torch.Tensor, hd: int, theta: float):
    """RoPE cos/sin for explicit absolute positions (decode appends past the prefix)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)  # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, hd]
    return emb.cos(), emb.sin()


class PagedKVCache:
    """Fixed-arena per-request K/V for the paged qwen2/llama/qwen3 forward.

    Stores pre-GQA-expand K/V (``[nL, capacity, nKV, hd]`` fp32) on the store device.
    Fixed capacity (prompt + max_new_tokens) — no ``torch.cat`` growth, per the
    measured KV-lifetime lesson from the OLMoE wide runs."""

    def __init__(self, nL: int, nKV: int, hd: int, capacity: int, device):
        self.k = torch.zeros((nL, capacity, nKV, hd), dtype=torch.float32, device=device)
        self.v = torch.zeros_like(self.k)
        self.capacity = int(capacity)
        self.pos = 0


@torch.no_grad()
def paged_forward_kv(
    store: QStore,
    input_ids: np.ndarray,
    cache: PagedKVCache,
) -> torch.Tensor:
    """Prefill or single-token decode over persistent K/V; returns the FINAL position's
    logits [V] (cpu fp32). Streams each weight once per call — a decode step therefore
    runs the weight stream against ONE row instead of replaying the whole prefix, and
    the lm_head streams against one row instead of [T, V].

    Claim boundary: greedy-token parity with the full-replay path is the gate, not
    bit-exact logits — a [1, Ttot] attention row and a [T, T] row can differ in float
    reduction order (the measured packed-shape lesson). qwen2/llama/qwen3 only; no
    patch/capture taps on this path."""
    c = store.cfg
    d, nL = c["hidden_size"], c["num_hidden_layers"]
    nH, nKV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps, theta = c["rms_norm_eps"], c["rope_theta"]
    T_new = len(input_ids)
    P = cache.pos
    Ttot = P + T_new
    if Ttot > cache.capacity:
        raise RuntimeError(f"PagedKVCache overflow: {Ttot} > {cache.capacity}")
    rep = nH // nKV
    scale = hd**-0.5

    h = store.embed_rows("embed", input_ids).clone()  # [T_new, d]
    dev = h.device
    positions = torch.arange(P, Ttot)
    cos, sin = _rope_tables_at(positions, hd, theta)
    cos, sin = cos.to(dev), sin.to(dev)
    # query row i (global P+i) may attend keys [0, P+i]
    if T_new > 1:
        qpos = positions.to(dev)[:, None]  # [T_new, 1]
        kpos = torch.arange(Ttot, device=dev)[None, :]  # [1, Ttot]
        causal = torch.where(kpos > qpos, float("-inf"), 0.0)  # [T_new, Ttot]
    else:
        causal = None

    for L in range(nL):
        x = _rms_norm(h, store.fp32(f"L{L}.ln1"), eps)
        q = store.matmul(f"L{L}.q", x)
        k = store.matmul(f"L{L}.k", x)
        v = store.matmul(f"L{L}.v", x)
        if store.has(f"L{L}.q.bias"):
            q = q + store.fp32(f"L{L}.q.bias")
            k = k + store.fp32(f"L{L}.k.bias")
            v = v + store.fp32(f"L{L}.v.bias")
        q = q.view(T_new, nH, hd)
        k = k.view(T_new, nKV, hd)
        v = v.view(T_new, nKV, hd)
        if store.has(f"L{L}.q_norm"):  # qwen3: per-head RMSNorm before RoPE (HF order)
            q = _rms_norm(q, store.fp32(f"L{L}.q_norm"), eps)
            k = _rms_norm(k, store.fp32(f"L{L}.k_norm"), eps)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        cache.k[L, P:Ttot] = k
        cache.v[L, P:Ttot] = v
        keys = cache.k[L, :Ttot]  # [Ttot, nKV, hd]
        vals = cache.v[L, :Ttot]
        if rep > 1:  # GQA expand at attention time
            keys = keys.repeat_interleave(rep, dim=1)
            vals = vals.repeat_interleave(rep, dim=1)
        qh = q.transpose(0, 1)  # [nH, T_new, hd]
        kh = keys.transpose(0, 1)  # [nH, Ttot, hd]
        vh = vals.transpose(0, 1)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # [nH, T_new, Ttot]
        if causal is not None:
            scores = scores + causal
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, vh.float()).transpose(0, 1).reshape(T_new, nH * hd)
        attn_out = store.matmul(f"L{L}.o", ctx)
        if store.has(f"L{L}.o.bias"):
            attn_out = attn_out + store.fp32(f"L{L}.o.bias")
        del q, k, v, qh, kh, vh, scores, probs, ctx
        h = h + attn_out

        x2 = _rms_norm(h, store.fp32(f"L{L}.ln2"), eps)
        g = store.matmul(f"L{L}.gate", x2)
        u = store.matmul(f"L{L}.up", x2)
        hid = torch.nn.functional.silu(g) * u
        h = h + store.matmul(f"L{L}.down", hid)
        del g, u, hid

    cache.pos = Ttot
    h_last = _rms_norm(h[-1:], store.fp32("norm.final"), eps)  # [1, d]
    V = c["vocab_size"]
    logits = torch.empty((1, V), dtype=torch.float32, device=dev)
    for start, end, Wblk in store.row_blocks("lm_head"):  # one ROW against the stream
        logits[:, start:end] = _streamed_lm_head_matmul(h_last, Wblk)
        del Wblk
    return logits[0].cpu()


# =================================================================== qwen2/llama BATCHED
def _apply_rope_b(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, T, n, hd]; cos/sin: [T, hd]. Broadcast over batch and heads."""
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + _rotate_half(x) * sin


@torch.no_grad()
def batched_paged_logits(
    store: QStore,
    ids_list: list[np.ndarray],
    *,
    last_only: bool = True,
    collect_acts: bool = False,
    return_hidden: bool = False,
    patch_ops_by_layer: dict[int, list] | None = None,
    patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    capture_selected_maps: dict[int, dict] | None = None,
    captured_selected: dict[int, torch.Tensor] | None = None,
    head_patch_ops_by_layer: dict[int, list] | None = None,
    head_patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    key_patch_ops_by_layer: dict[int, list] | None = None,
    key_patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    value_patch_ops_by_layer: dict[int, list] | None = None,
    value_patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    collect_key_out: bool = False,
    collect_value_out: bool = False,
    resid_patch_ops_by_layer: dict[int, list] | None = None,
    resid_patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    initial_hidden: torch.Tensor | None = None,
    start_layer: int = 0,
    stop_layer: int | None = None,
    collect_head_out: bool = False,
    collect_hidden_states: bool = False,
    return_aux: bool = False,
):
    """Batch-amortized twin of ``paged_logits`` (qwen2/llama): stream + dequantize each weight
    ONCE and run B sequences against it. Pays the dominant dequant cost a single time (the matmul
    has ~1000x headroom), so throughput scales ~B x until the matmul knee. Measured Qwen2.5-0.5B:
    17.5x at B=128, T=32.

    Right-pad + causal + key-pad mask ⇒ row i equals ``paged_logits(store, ids_list[i])`` up to
    float reduction order (verified argmax-exact, max|Δ| 1.5e-4). Resident WEIGHT heap stays
    O(largest single matrix); only activations grow O(B·Tmax·d) — with batch, not model.

    Returns (last_only ? logits[B,V] at each last real token : logits[B,Tmax,V]), lengths[B];
    plus the per-layer gated-MLP act tape (list of [B,Tmax,inter]) if collect_acts.
    ``patch_ops_by_layer`` applies the same MLP down-input patch map to every row in the
    batch. ``patch_ops_by_layer_rows`` instead applies a distinct MLP patch map to each row;
    supplying both is rejected. Head and residual maps likewise accept either one shared map
    or one map per row. ``initial_hidden`` plus ``start_layer`` resumes from an exact layer-input
    StateCut; ``stop_layer`` returns an unnormalized layer-input state when ``return_hidden`` is
    true. ``collect_hidden_states`` records HF-layout states (embedding then
    block outputs, with the final entry replaced by the post-final-norm state).
    ``return_aux`` returns (logits, lengths, {acts, captured_selected, head_out,
    hidden_states}) regardless of the individual collect flags.
    """
    c = store.cfg
    d, nL = c["hidden_size"], c["num_hidden_layers"]
    nH, nKV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps, theta = c["rms_norm_eps"], c["rope_theta"]
    V = c["vocab_size"]
    rep = nH // nKV
    scale = hd**-0.5

    B = len(ids_list)
    if patch_ops_by_layer and patch_ops_by_layer_rows is not None:
        raise ValueError("shared and row-local MLP patch maps are mutually exclusive")
    if head_patch_ops_by_layer and head_patch_ops_by_layer_rows is not None:
        raise ValueError("shared and row-local head patch maps are mutually exclusive")
    if key_patch_ops_by_layer and key_patch_ops_by_layer_rows is not None:
        raise ValueError("shared and row-local key patch maps are mutually exclusive")
    if value_patch_ops_by_layer and value_patch_ops_by_layer_rows is not None:
        raise ValueError("shared and row-local value patch maps are mutually exclusive")
    if resid_patch_ops_by_layer and resid_patch_ops_by_layer_rows is not None:
        raise ValueError("shared and row-local residual patch maps are mutually exclusive")
    row_patch_ops = _compile_row_patch_maps(patch_ops_by_layer_rows, batch_size=B)
    row_head_patch_ops = _compile_row_patch_maps(head_patch_ops_by_layer_rows, batch_size=B)
    row_key_patch_ops = _compile_row_patch_maps(key_patch_ops_by_layer_rows, batch_size=B)
    row_value_patch_ops = _compile_row_patch_maps(value_patch_ops_by_layer_rows, batch_size=B)
    row_resid_patch_ops = _compile_row_patch_maps(resid_patch_ops_by_layer_rows, batch_size=B)
    if isinstance(start_layer, bool) or not isinstance(start_layer, Integral):
        raise TypeError("start_layer must be an integer")
    start = int(start_layer)
    stop = nL if stop_layer is None else stop_layer
    if isinstance(stop, bool) or not isinstance(stop, Integral):
        raise TypeError("stop_layer must be an integer")
    stop = int(stop)
    if not 0 <= start <= stop <= nL:
        raise ValueError("layer interval must satisfy 0 <= start <= stop <= num_layers")
    if stop < nL and not return_hidden:
        raise ValueError("partial layer execution requires return_hidden=True")
    if initial_hidden is None and start != 0:
        raise ValueError("nonzero start_layer requires initial_hidden")
    if collect_hidden_states and (start != 0 or stop != nL):
        raise ValueError("hidden-state tape collection requires a complete forward")
    for label, row_maps in (
        ("MLP", row_patch_ops),
        ("head", row_head_patch_ops),
        ("key", row_key_patch_ops),
        ("value", row_value_patch_ops),
        ("residual", row_resid_patch_ops),
    ):
        outside = sorted(layer for layer in row_maps if not start <= layer < stop)
        if outside:
            raise ValueError(
                f"row-local {label} patch layers fall outside the executed interval: {outside}"
            )
    lengths = np.array([len(x) for x in ids_list], dtype=np.int64)
    Tmax = int(lengths.max())
    ids_pad = np.zeros((B, Tmax), dtype=np.int64)
    real = torch.zeros((B, Tmax), dtype=torch.bool)
    for b, x in enumerate(ids_list):
        ids_pad[b, : len(x)] = np.asarray(x, dtype=np.int64)
        real[b, : len(x)] = True

    if initial_hidden is None:
        h = store.embed_rows("embed", ids_pad.reshape(-1)).clone().view(B, Tmax, d)
    else:
        if tuple(initial_hidden.shape) != (B, Tmax, d):
            raise ValueError("initial_hidden must have shape [B,Tmax,hidden_size]")
        h = initial_hidden.clone()
    dev = h.device  # follows the store (cpu default)
    cos, sin = _rope_tables(Tmax, hd, theta)
    cos, sin = cos.to(dev), sin.to(dev)
    causal = torch.triu(torch.full((Tmax, Tmax), float("-inf"), device=dev), diagonal=1)
    key_pad = torch.where(real, 0.0, float("-inf"))[:, None, None, :].to(dev)  # [B,1,1,Tmax]
    # acts only when asked: return_aux must NOT force the [B,Tmax,inter]×nL fp32 tape —
    # forward_patched_batch always passes return_aux=True and usually collect_acts=False;
    # collecting anyway was a measured 3.8GB (0.5B, B=16 T=512) / ~8.7GB (7B) peak-RSS spike
    # on the exact backend whose contract is O(largest matrix). aux returns `acts or []`.
    acts = [] if collect_acts else None
    head_out = [] if collect_head_out else None
    key_out = [] if collect_key_out else None
    value_out = [] if collect_value_out else None
    hidden_states = [h.detach().clone().cpu()] if collect_hidden_states else None
    captured = captured_selected if captured_selected is not None else {}

    for L in range(start, stop):
        x = _rms_norm(h, store.fp32(f"L{L}.ln1"), eps)
        q = store.matmul(f"L{L}.q", x)
        k = store.matmul(f"L{L}.k", x)
        v = store.matmul(f"L{L}.v", x)
        if store.has(f"L{L}.q.bias"):
            q = q + store.fp32(f"L{L}.q.bias")
            k = k + store.fp32(f"L{L}.k.bias")
            v = v + store.fp32(f"L{L}.v.bias")
        if key_out is not None:
            key_out.append(k.detach().clone().cpu())
        if value_out is not None:
            value_out.append(v.detach().clone().cpu())
        if key_patch_ops_by_layer and L in key_patch_ops_by_layer:
            k = _apply_projection_patch_ops(k, key_patch_ops_by_layer[L])
        elif L in row_key_patch_ops:
            k = _apply_projection_patch_ops_by_row(k, row_key_patch_ops[L])
        if value_patch_ops_by_layer and L in value_patch_ops_by_layer:
            v = _apply_projection_patch_ops(v, value_patch_ops_by_layer[L])
        elif L in row_value_patch_ops:
            v = _apply_projection_patch_ops_by_row(v, row_value_patch_ops[L])
        q = q.view(B, Tmax, nH, hd)
        k = k.view(B, Tmax, nKV, hd)
        if store.has(f"L{L}.q_norm"):  # qwen3: per-head RMSNorm on q/k
            q = _rms_norm(
                q, store.fp32(f"L{L}.q_norm"), eps
            )  # over head_dim, BEFORE RoPE (HF order)
            k = _rms_norm(k, store.fp32(f"L{L}.k_norm"), eps)
        q = _apply_rope_b(q, cos, sin)
        k = _apply_rope_b(k, cos, sin)
        v = v.view(B, Tmax, nKV, hd)
        if rep > 1:
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B,nH,T,hd]
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale + causal + key_pad
        probs = torch.softmax(scores, dim=-1)
        head_ctx = torch.matmul(probs, vh.float()).transpose(1, 2)  # [B,T,nH,hd]
        if head_patch_ops_by_layer and L in head_patch_ops_by_layer:
            head_ctx = _apply_head_patch_ops(head_ctx, head_patch_ops_by_layer[L])
        elif L in row_head_patch_ops:
            head_ctx = _apply_head_patch_ops_by_row(head_ctx, row_head_patch_ops[L])
        if head_out is not None:
            head_out.append(head_ctx.detach().clone().cpu())
        ctx = head_ctx.contiguous().reshape(B, Tmax, nH * hd)
        attn_out = store.matmul(f"L{L}.o", ctx)
        if store.has(f"L{L}.o.bias"):
            attn_out = attn_out + store.fp32(f"L{L}.o.bias")
        del q, k, v, qh, kh, vh, scores, probs, head_ctx, ctx
        h = h + attn_out

        x2 = _rms_norm(h, store.fp32(f"L{L}.ln2"), eps)
        g = store.matmul(f"L{L}.gate", x2)
        u = store.matmul(f"L{L}.up", x2)
        hid = torch.nn.functional.silu(g) * u
        if patch_ops_by_layer and L in patch_ops_by_layer:
            hid = _apply_patch_ops(hid, patch_ops_by_layer[L])
        elif L in row_patch_ops:
            hid = _apply_patch_ops_by_row(hid, row_patch_ops[L])
        if capture_selected_maps and L in capture_selected_maps:
            locals_ = capture_selected_maps[L].get("locals", [])
            if locals_:
                cols = torch.as_tensor(locals_, dtype=torch.long, device=hid.device)
                captured[L] = hid[..., cols].detach().to(dtype=torch.float16, device="cpu")
        if acts is not None:
            acts.append(hid.detach().clone().cpu())
        h = h + store.matmul(f"L{L}.down", hid)
        del g, u, hid
        if resid_patch_ops_by_layer and L in resid_patch_ops_by_layer:
            h = _apply_resid_patch_ops(h, resid_patch_ops_by_layer[L])
        elif L in row_resid_patch_ops:
            h = _apply_resid_patch_ops_by_row(h, row_resid_patch_ops[L])
        if hidden_states is not None:
            hidden_states.append(h.detach().clone().cpu())

    if stop == nL:
        h = _rms_norm(h, store.fp32("norm.final"), eps)
    if hidden_states is not None:
        # Match HF ``output_hidden_states``: its terminal entry is normalized rather
        # than the raw output of the last decoder block.
        hidden_states[-1] = h.detach().clone().cpu()
    if return_hidden:
        # Skip lm_head entirely — return the final hidden state. Lets a caller score against a
        # candidate-only column subset of lm_head (store.embed_rows("lm_head", tokens)), so the
        # 544 MB unembedding is NEVER streamed: the working set stays at the largest MLP block
        # (~17.4 MB on Qwen2.5-0.5B), below the 29.4 MB lm_head-row-block floor.
        # h_last stays on store.device: the subset-head caller matmuls it against
        # embed_rows("lm_head", ...) which is also on-device; scalars sync on float().
        h_last = h[torch.arange(B, device=dev), torch.from_numpy(lengths - 1).to(dev)]  # [B,d]
        out_h = h_last if last_only else h  # [B,d] or [B,Tmax,d]
        if return_aux:
            return (
                out_h,
                lengths,
                {
                    "acts": acts or [],
                    "captured_selected": captured,
                    "head_out": head_out or [],
                    "key_out": key_out or [],
                    "value_out": value_out or [],
                    "hidden_states": hidden_states or [],
                },
            )
        if collect_acts:
            return out_h, lengths, acts
        return out_h, lengths
    if last_only:
        h_last = h[torch.arange(B, device=dev), torch.from_numpy(lengths - 1).to(dev)]  # [B,d]
        logits = torch.empty((B, V), dtype=torch.float32, device=dev)
        for start, end, Wblk in store.row_blocks("lm_head"):
            logits[:, start:end] = _streamed_lm_head_matmul(h_last, Wblk)
            del Wblk
    else:
        logits = torch.empty((B, Tmax, V), dtype=torch.float32, device=dev)
        for start, end, Wblk in store.row_blocks("lm_head"):
            logits[:, :, start:end] = _streamed_lm_head_matmul(h, Wblk)
            del Wblk
    logits = logits.cpu()  # nothing device-resident escapes
    if return_aux:
        return (
            logits,
            lengths,
            {
                "acts": acts or [],
                "captured_selected": captured,
                "head_out": head_out or [],
                "key_out": key_out or [],
                "value_out": value_out or [],
                "hidden_states": hidden_states or [],
            },
        )
    if collect_acts:
        return logits, lengths, acts
    return logits, lengths


# ==================================================== qwen2/llama BATCHED KV + BLOCK VERIFY
def _rope_tables_rows(positions: torch.Tensor, hd: int, theta: float):
    """RoPE cos/sin for PER-ROW absolute positions. positions: [B, T] -> cos/sin [B, T, hd].
    Decode/verify rows sit at different depths (mixed prompt lengths, partial accepts), so
    each row needs its own position tables."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    freqs = positions.to(torch.float32)[..., None] * inv_freq  # [B, T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [B, T, hd]
    return emb.cos(), emb.sin()


def _apply_rope_rows(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, T, n, hd]; cos/sin: [B, T, hd] per-row tables. Broadcast over heads only."""
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    return x * cos + _rotate_half(x) * sin


@dataclass(frozen=True, slots=True)
class PagedKVSlotLease:
    """Cache-issued capability for one mutable paged-KV request slot.

    ``generation`` prevents ABA reuse after release while ``lease_id`` prevents callers from
    manufacturing authority from public cache metadata.  A lease is valid only while the exact
    record remains active in its issuing cache.
    """

    cache_id: str
    row: int
    generation: int
    lease_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cache_id, str)
            or not self.cache_id
            or self.cache_id.strip() != self.cache_id
        ):
            raise TypeError("paged KV slot lease cache_id must be non-empty")
        if isinstance(self.row, bool) or not isinstance(self.row, Integral) or int(self.row) < 0:
            raise TypeError("paged KV slot lease row must be a non-negative integer")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, Integral)
            or int(self.generation) <= 0
        ):
            raise TypeError("paged KV slot lease generation must be a positive integer")
        if not isinstance(self.lease_id, str) or not self.lease_id:
            raise TypeError("paged KV slot lease lease_id must be non-empty")


class BatchedPagedKVCache:
    """B request slots of COMMITTED K/V for the batched paged decode/verify path.

    Fixed arena ``[nL, B, capacity, nKV, hd]`` fp32 on the store device (pre-GQA-expand,
    like :class:`PagedKVCache`) — no ``torch.cat`` growth. ``lengths[b]`` is slot b's
    committed length; reads are always masked by committed length, so stale tails are
    never cleared. ``epoch`` implements the dense-qstore transactional contract: every
    committed mutation bumps it, and a provisional :class:`PagedKVDelta` may only commit
    against the exact (epoch, lengths) it was forwarded from."""

    def __init__(self, nL: int, B: int, nKV: int, hd: int, capacity: int, device):
        raw_dimensions = (nL, B, nKV, hd, capacity)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in raw_dimensions
        ):
            raise TypeError("paged KV cache dimensions must be integers")
        dimensions = tuple(int(value) for value in raw_dimensions)
        if min(dimensions) <= 0:
            raise ValueError("paged KV cache dimensions must be positive")
        nL, B, nKV, hd, capacity = dimensions
        # Version counters are the zero-copy committed-prefix mutation guard. Explicitly
        # disable an outer inference_mode so these long-lived state tensors retain one.
        with torch.inference_mode(False):
            self.k = torch.zeros((nL, B, capacity, nKV, hd), dtype=torch.float32, device=device)
            self.v = torch.zeros_like(self.k)
        self.B = int(B)
        self.capacity = int(capacity)
        self.lengths = np.zeros(self.B, dtype=np.int64)
        self.epoch = 0
        self.cache_id = uuid4().hex
        # This key never changes even if corrupt same-process code tampers with public cache_id.
        # Every operation that locks more than one cache sorts by it, so reversed request order
        # cannot introduce an AB/BA deadlock.
        self._lock_order_key = uuid4().hex
        self._lock = RLock()
        self._poisoned_reason: str | None = None
        self._slot_generations = [0 for _ in range(self.B)]
        self._active_slot_leases: dict[int, PagedKVSlotLease] = {}

    def assert_usable(self) -> None:
        if self._poisoned_reason is not None:
            raise RuntimeError(f"paged KV cache is poisoned: {self._poisoned_reason}")

    def _validate_slot_row(self, row: int) -> int:
        if isinstance(row, bool) or not isinstance(row, Integral):
            raise TypeError("paged KV slot row must be an integer")
        normalized = int(row)
        if normalized < 0 or normalized >= self.B:
            raise ValueError(f"paged KV slot row {normalized} is outside [0, {self.B})")
        return normalized

    def _validate_slot_lease_unlocked(self, lease: PagedKVSlotLease) -> None:
        if not isinstance(lease, PagedKVSlotLease):
            raise TypeError("slot lease must be a PagedKVSlotLease")
        if lease.cache_id != self.cache_id:
            raise RuntimeError("paged KV slot lease belongs to a different cache")
        row = self._validate_slot_row(lease.row)
        current_generation = self._slot_generations[row]
        if int(lease.generation) != current_generation:
            raise RuntimeError(
                f"stale paged KV slot lease generation {lease.generation}; "
                f"slot {row} is at generation {current_generation}"
            )
        active = self._active_slot_leases.get(row)
        if active is None:
            raise RuntimeError("paged KV slot lease has been released")
        if active != lease:
            raise RuntimeError("paged KV slot lease identity does not match the active lease")

    def mint_slot_lease(self, row: int = 0) -> PagedKVSlotLease:
        """Mint the sole active capability for ``row``.

        The cache, not the caller, chooses both incarnation and identity.  A row cannot be
        leased twice concurrently.
        """

        with self._lock:
            self.assert_usable()
            normalized = self._validate_slot_row(row)
            if normalized in self._active_slot_leases:
                raise RuntimeError(f"paged KV slot {normalized} already has an active lease")
            generation = self._slot_generations[normalized] + 1
            self._slot_generations[normalized] = generation
            lease = PagedKVSlotLease(
                cache_id=self.cache_id,
                row=normalized,
                generation=generation,
                lease_id=uuid4().hex,
            )
            self._active_slot_leases[normalized] = lease
            return lease

    def validate_slot_lease(self, lease: PagedKVSlotLease) -> None:
        """Fail closed unless ``lease`` is the exact active cache-issued record."""

        with self._lock:
            self.assert_usable()
            self._validate_slot_lease_unlocked(lease)

    def release_slot_lease(self, lease: PagedKVSlotLease) -> None:
        """Release an active capability; duplicate and stale releases fail closed."""

        with self._lock:
            self.assert_usable()
            self._validate_slot_lease_unlocked(lease)
            del self._active_slot_leases[int(lease.row)]


def _require_disjoint_tensor_storage(
    tensors: Sequence[torch.Tensor],
    *,
    field: str,
) -> None:
    ranges: list[tuple[str, int, int]] = []
    for tensor in tensors:
        if tensor.layout is not torch.strided:
            raise TypeError(f"{field} tensors must use strided storage")
        start = int(tensor.untyped_storage().data_ptr()) + int(tensor.storage_offset()) * int(
            tensor.element_size()
        )
        ranges.append(
            (
                str(tensor.device),
                start,
                start + int(tensor.numel()) * int(tensor.element_size()),
            )
        )
    for index, first in enumerate(ranges):
        for second in ranges[index + 1 :]:
            if first[0] == second[0] and max(first[1], second[1]) < min(first[2], second[2]):
                raise ValueError(f"{field} tensors must not alias or overlap")


def _require_separate_kv_storage(k: torch.Tensor, v: torch.Tensor, *, field: str) -> None:
    _require_disjoint_tensor_storage((k, v), field=field)


@dataclass(frozen=True)
class PagedKVDelta:
    """Uncommitted provisional K/V from one :func:`paged_forward_block`.

    Lives in its own scratch tensors ``[nL, B, K, nKV, hd]`` — the committed arena is
    untouched until :func:`commit_block`. ``parent_epoch``/``parent_lengths`` pin the
    exact committed state it was computed against (stale commits are rejected).

    The transaction protocol detects ordinary Torch mutation and raw mutation racing the staging
    copy. As with committed arenas, callers must not use ``.data``/foreign-memory writes outside
    the cache transaction boundary; that is intentionally outside the trusted-backend contract.
    """

    parent_epoch: int
    parent_lengths: tuple[int, ...]
    cache_id: str
    k: torch.Tensor
    v: torch.Tensor
    token_count: int
    slot_lease: PagedKVSlotLease | None = None
    tensor_signature: tuple[tuple[object, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.parent_epoch, bool) or not isinstance(self.parent_epoch, Integral):
            raise TypeError("provisional KV parent_epoch must be an integer")
        if not isinstance(self.parent_lengths, tuple) or any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for value in self.parent_lengths
        ):
            raise TypeError("provisional KV parent_lengths must be non-negative integers")
        if (
            not isinstance(self.cache_id, str)
            or not self.cache_id
            or self.cache_id.strip() != self.cache_id
        ):
            raise TypeError("provisional KV cache_id must be non-empty")
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, Integral):
            raise TypeError("provisional KV token_count must be an integer")
        if int(self.token_count) <= 0:
            raise ValueError("provisional KV token_count must be positive")
        if self.slot_lease is not None:
            if not isinstance(self.slot_lease, PagedKVSlotLease):
                raise TypeError("provisional KV slot_lease must be a PagedKVSlotLease")
            if self.slot_lease.cache_id != self.cache_id:
                raise ValueError("provisional KV slot lease belongs to a different cache")
            if len(self.parent_lengths) != 1 or self.slot_lease.row != 0:
                raise ValueError("lease-bound provisional KV must describe singleton row zero")
        if not isinstance(self.k, torch.Tensor) or not isinstance(self.v, torch.Tensor):
            raise TypeError("provisional KV payloads must be tensors")
        if self.k.requires_grad or self.v.requires_grad:
            raise ValueError("provisional KV tensors must not require gradients")
        _require_separate_kv_storage(self.k, self.v, field="provisional KV")
        try:
            signature = tuple(
                (
                    id(tensor),
                    int(tensor.data_ptr()),
                    tuple(int(value) for value in tensor.shape),
                    str(tensor.dtype),
                    str(tensor.device),
                    bool(tensor.is_contiguous()),
                    int(tensor._version),  # noqa: SLF001 - pins scratch contents
                )
                for tensor in (self.k, self.v)
            )
        except RuntimeError as exc:
            raise TypeError("provisional KV tensors must track mutation versions") from exc
        object.__setattr__(self, "tensor_signature", signature)


@dataclass(frozen=True)
class PagedKVForkPanel:
    """Copy-on-write K/V diffs for several branches over one immutable parent batch.

    ``k`` and ``v`` retain one aggregate
    ``[layers, branches * batch, tokens, kv_heads, head_dim]`` scratch allocation. They never
    contain or alias the committed parent prefix. Selecting the one branch that will commit
    creates only that branch's block diff; sibling parent state is never duplicated.
    """

    parent_epoch: int
    parent_lengths: tuple[int, ...]
    cache_id: str
    branch_count: int
    batch_size: int
    k: torch.Tensor
    v: torch.Tensor
    token_count: int = 1
    tensor_signature: tuple[tuple[object, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.parent_epoch, bool) or not isinstance(self.parent_epoch, Integral):
            raise TypeError("fork panel parent_epoch must be an integer")
        if not isinstance(self.parent_lengths, tuple) or any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for value in self.parent_lengths
        ):
            raise TypeError("fork panel parent_lengths must be non-negative integers")
        for field_name, value in (
            ("branch_count", self.branch_count),
            ("batch_size", self.batch_size),
            ("token_count", self.token_count),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"fork panel {field_name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"fork panel {field_name} must be positive")
        if len(self.parent_lengths) != int(self.batch_size):
            raise ValueError("fork panel parent lengths must match its parent batch")
        if not isinstance(self.cache_id, str) or not self.cache_id:
            raise TypeError("fork panel cache_id must be non-empty")
        if not isinstance(self.k, torch.Tensor) or not isinstance(self.v, torch.Tensor):
            raise TypeError("fork panel K/V payloads must be tensors")
        if self.k.requires_grad or self.v.requires_grad:
            raise ValueError("fork panel K/V tensors must not require gradients")
        _require_separate_kv_storage(self.k, self.v, field="fork panel K/V")
        if self.k.ndim != 5 or self.v.ndim != 5:
            raise ValueError("fork panel K/V tensors must have rank five")
        if tuple(self.k.shape) != tuple(self.v.shape):
            raise ValueError("fork panel K/V tensors must share one shape")
        if int(self.k.shape[1]) != int(self.branch_count) * int(self.batch_size):
            raise ValueError("fork panel row count must equal branches * parent batch")
        if int(self.k.shape[2]) != int(self.token_count):
            raise ValueError("fork panel K/V token axis must match token_count")
        if self.k.dtype != torch.float32 or self.v.dtype != torch.float32:
            raise ValueError("fork panel K/V tensors must use fp32")
        if self.k.device != self.v.device:
            raise ValueError("fork panel K/V tensors must share one device")
        if not self.k.is_contiguous() or not self.v.is_contiguous():
            raise ValueError("fork panel K/V tensors must be contiguous")
        object.__setattr__(self, "tensor_signature", self._current_signature())

    def _current_signature(self) -> tuple[tuple[object, ...], ...]:
        try:
            return tuple(
                (
                    id(tensor),
                    int(tensor.data_ptr()),
                    tuple(int(value) for value in tensor.shape),
                    str(tensor.dtype),
                    str(tensor.device),
                    bool(tensor.is_contiguous()),
                    int(tensor._version),  # noqa: SLF001 - guards the retained diff
                )
                for tensor in (self.k, self.v)
            )
        except RuntimeError as exc:
            raise TypeError("fork panel tensors must track mutation versions") from exc

    @property
    def retained_bytes(self) -> int:
        return int(self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size())

    @property
    def per_branch_bytes(self) -> int:
        return self.retained_bytes // int(self.branch_count)

    def select(self, branch_index: int) -> PagedKVDelta:
        """Materialize only one branch diff for the existing atomic commit primitive."""

        if isinstance(branch_index, bool) or not isinstance(branch_index, Integral):
            raise TypeError("branch_index must be an integer")
        normalized = int(branch_index)
        if normalized < 0 or normalized >= int(self.branch_count):
            raise IndexError("branch_index is outside the fork panel")
        if self._current_signature() != self.tensor_signature:
            raise RuntimeError("fork panel K/V diff changed after continuation")
        start = normalized * int(self.batch_size)
        end = start + int(self.batch_size)
        with torch.inference_mode(False):
            selected_k = self.k[:, start:end].detach().clone(memory_format=torch.contiguous_format)
            selected_v = self.v[:, start:end].detach().clone(memory_format=torch.contiguous_format)
        return PagedKVDelta(
            parent_epoch=int(self.parent_epoch),
            parent_lengths=self.parent_lengths,
            cache_id=self.cache_id,
            k=selected_k,
            v=selected_v,
            token_count=int(self.token_count),
        )


@torch.no_grad()
def _paged_forward_kv_batch_unlocked(
    store: QStore,
    ids_list: list[np.ndarray],
    cache: BatchedPagedKVCache,
    *,
    output_contract: PagedBlockOutputContract = "full_logits",
    selected_rows: Sequence[int] = (),
    resid_patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    key_patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    value_patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    capture_hidden_layers: Sequence[int] = (),
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Batched prefill or decode over per-slot committed K/V; returns each row's
    last-REAL-position logits [B, V] (cpu fp32). ONE weight traversal serves all B rows
    (mirrors ``batched_paged_logits``: right-pad + causal/key masks, GQA expand, qwen3
    q/k-norm, biases), and appended K/V goes straight into the committed arena — this is
    the non-speculative path, so every produced position is committed by construction.

    Rows may have unequal new lengths AND unequal committed pasts (mixed prompts,
    post-partial-accept decode): per-row RoPE position tables + a per-row causal mask
    (query at global position P_b+t sees keys ≤ P_b+t) handle both. Padded query rows
    are never written to the arena and their outputs are discarded.

    Claim boundary (house rule): greedy-token parity with the scalar ``paged_forward_kv``
    path, not bit-exact logits — packed-shape float reduction order differs (measured).
    qwen2/llama/qwen3 math only."""
    c = store.cfg
    contract = str(output_contract)
    if contract not in _PAGED_BLOCK_OUTPUT_CONTRACTS:
        choices = ", ".join(sorted(_PAGED_BLOCK_OUTPUT_CONTRACTS))
        raise ValueError(f"output_contract must be one of: {choices}")
    output_rows = tuple(int(value) for value in selected_rows)
    if contract == "selected_token_rows":
        if not output_rows:
            raise ValueError("selected_token_rows requires selected_rows")
        if len(output_rows) != len(set(output_rows)):
            raise ValueError("selected_rows must be unique")
        vocab_size = int(c["vocab_size"])
        if min(output_rows) < 0 or max(output_rows) >= vocab_size:
            raise ValueError(f"selected_rows must be inside [0, {vocab_size})")
    elif output_rows:
        raise ValueError("selected_rows are only legal for selected_token_rows")
    d, nL = c["hidden_size"], c["num_hidden_layers"]
    nH, nKV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps, theta = c["rms_norm_eps"], c["rope_theta"]
    B = len(ids_list)
    if B != cache.B:
        raise ValueError(f"batch {B} does not match cache slots {cache.B}")
    t_new = np.array([len(x) for x in ids_list], dtype=np.int64)
    if (t_new <= 0).any():
        raise ValueError("every row must supply at least one token")
    P = cache.lengths.copy()
    tot = P + t_new
    if (tot > cache.capacity).any():
        raise RuntimeError(f"BatchedPagedKVCache overflow: {tot.max()} > {cache.capacity}")
    Tmax = int(t_new.max())
    max_tot = int(tot.max())
    rep = nH // nKV
    scale = hd**-0.5
    row_resid_patch_ops = _compile_row_patch_maps(resid_patch_ops_by_layer_rows, batch_size=B)
    row_key_patch_ops = _compile_row_patch_maps(key_patch_ops_by_layer_rows, batch_size=B)
    row_value_patch_ops = _compile_row_patch_maps(value_patch_ops_by_layer_rows, batch_size=B)
    capture_layers = tuple(sorted(set(int(value) for value in capture_hidden_layers)))
    if any(layer < 0 or layer >= nL for layer in capture_layers):
        raise ValueError("captured hidden layer is outside the model")
    for label, row_maps in (
        ("residual", row_resid_patch_ops),
        ("key", row_key_patch_ops),
        ("value", row_value_patch_ops),
    ):
        outside = sorted(layer for layer in row_maps if layer < 0 or layer >= nL)
        if outside:
            raise ValueError(f"row-local {label} patch layers are outside the model: {outside}")
    captured_hidden: dict[int, torch.Tensor] = {}
    captured_hidden_after_patch: dict[int, torch.Tensor] = {}

    ids_pad = np.zeros((B, Tmax), dtype=np.int64)
    for b, x in enumerate(ids_list):
        ids_pad[b, : len(x)] = np.asarray(x, dtype=np.int64)
    h = store.embed_rows("embed", ids_pad.reshape(-1)).clone().view(B, Tmax, d)
    dev = h.device
    pos = torch.as_tensor(P)[:, None] + torch.arange(Tmax)[None, :]  # [B, Tmax] global pos
    cos, sin = _rope_tables_rows(pos, hd, theta)
    cos, sin = cos.to(dev), sin.to(dev)
    # query (b, t) at global position P_b+t attends keys ≤ P_b+t: causal + past-length +
    # key-pad in one mask (keys past a row's own extent are stale arena slots — excluded).
    kpos = torch.arange(max_tot, device=dev)
    allowed = kpos[None, None, :] <= pos.to(dev)[:, :, None]  # [B, Tmax, max_tot]
    mask = torch.where(allowed, 0.0, float("-inf"))[:, None, :, :]  # [B, 1, Tmax, max_tot]

    for L in range(nL):
        x = _rms_norm(h, store.fp32(f"L{L}.ln1"), eps)
        q = store.matmul(f"L{L}.q", x)
        k = store.matmul(f"L{L}.k", x)
        v = store.matmul(f"L{L}.v", x)
        if store.has(f"L{L}.q.bias"):
            q = q + store.fp32(f"L{L}.q.bias")
            k = k + store.fp32(f"L{L}.k.bias")
            v = v + store.fp32(f"L{L}.v.bias")
        if L in row_key_patch_ops:
            k = _apply_projection_patch_ops_by_row(k, row_key_patch_ops[L])
        if L in row_value_patch_ops:
            v = _apply_projection_patch_ops_by_row(v, row_value_patch_ops[L])
        q = q.view(B, Tmax, nH, hd)
        k = k.view(B, Tmax, nKV, hd)
        v = v.view(B, Tmax, nKV, hd)
        if store.has(f"L{L}.q_norm"):  # qwen3: pre-RoPE (HF order)
            q = _rms_norm(q, store.fp32(f"L{L}.q_norm"), eps)
            k = _rms_norm(k, store.fp32(f"L{L}.k_norm"), eps)
        q = _apply_rope_rows(q, cos, sin)
        k = _apply_rope_rows(k, cos, sin)
        for b in range(B):  # commit REAL rows only
            tb, pb = int(t_new[b]), int(P[b])
            cache.k[L, b, pb : pb + tb] = k[b, :tb]
            cache.v[L, b, pb : pb + tb] = v[b, :tb]
        keys = cache.k[L, :, :max_tot]  # [B, max_tot, nKV, hd]
        vals = cache.v[L, :, :max_tot]
        if rep > 1:  # GQA expand at attn time
            keys = keys.repeat_interleave(rep, dim=2)
            vals = vals.repeat_interleave(rep, dim=2)
        qh = q.transpose(1, 2)  # [B, nH, Tmax, hd]
        kh = keys.transpose(1, 2)  # [B, nH, max_tot, hd]
        vh = vals.transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale + mask
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, vh.float()).transpose(1, 2).reshape(B, Tmax, nH * hd)
        attn_out = store.matmul(f"L{L}.o", ctx)
        if store.has(f"L{L}.o.bias"):
            attn_out = attn_out + store.fp32(f"L{L}.o.bias")
        del q, k, v, qh, kh, vh, keys, vals, scores, probs, ctx
        h = h + attn_out

        x2 = _rms_norm(h, store.fp32(f"L{L}.ln2"), eps)
        g = store.matmul(f"L{L}.gate", x2)
        u = store.matmul(f"L{L}.up", x2)
        hid = torch.nn.functional.silu(g) * u
        h = h + store.matmul(f"L{L}.down", hid)
        del g, u, hid
        if L in capture_layers:
            captured_hidden[L] = (
                h[
                    torch.arange(B, device=dev),
                    torch.as_tensor(t_new - 1, device=dev),
                ]
                .detach()
                .float()
                .cpu()
            )
        if L in row_resid_patch_ops:
            h = _apply_resid_patch_ops_by_row(h, row_resid_patch_ops[L])
        if L in capture_layers:
            captured_hidden_after_patch[L] = (
                h[
                    torch.arange(B, device=dev),
                    torch.as_tensor(t_new - 1, device=dev),
                ]
                .detach()
                .float()
                .cpu()
            )

    cache.lengths = tot
    cache.epoch += 1  # committed mutation
    h_last = h[torch.arange(B, device=dev), torch.as_tensor(t_new - 1, device=dev)]  # [B, d]
    h_last = _rms_norm(h_last, store.fp32("norm.final"), eps)
    if contract == "hidden_state_only":
        result = h_last.detach().cpu()
    elif contract == "selected_token_rows":
        weights = store.embed_rows(
            "lm_head",
            np.asarray(output_rows, dtype=np.int64),
        ).to(device=dev, dtype=torch.float32)
        result = (h_last.float() @ weights.T).detach().cpu()
    else:
        V = c["vocab_size"]
        logits = torch.empty((B, V), dtype=torch.float32, device=dev)
        for start, end, Wblk in store.row_blocks("lm_head"):  # B rows vs the stream
            logits[:, start:end] = _streamed_lm_head_matmul(h_last, Wblk)
            del Wblk
        result = logits.cpu()
    if return_aux:
        return result, {
            "captured_hidden": captured_hidden,
            "captured_hidden_after_patch": captured_hidden_after_patch,
            "patched_layers": tuple(sorted(row_resid_patch_ops)),
            "patched_key_layers": tuple(sorted(row_key_patch_ops)),
            "patched_value_layers": tuple(sorted(row_value_patch_ops)),
        }
    return result


def paged_forward_kv_batch(
    store: QStore,
    ids_list: list[np.ndarray],
    cache: BatchedPagedKVCache,
    *,
    output_contract: PagedBlockOutputContract = "full_logits",
    selected_rows: Sequence[int] = (),
    resid_patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    key_patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    value_patch_ops_by_layer_rows: Sequence[Mapping[int, list | tuple] | None] | None = None,
    capture_hidden_layers: Sequence[int] = (),
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Run committed batched KV mutation under the cache's single-writer lease."""

    with cache._lock:  # noqa: SLF001 - kernel/cache form one transaction boundary
        cache.assert_usable()
        return _paged_forward_kv_batch_unlocked(
            store,
            ids_list,
            cache,
            output_contract=output_contract,
            selected_rows=selected_rows,
            resid_patch_ops_by_layer_rows=resid_patch_ops_by_layer_rows,
            key_patch_ops_by_layer_rows=key_patch_ops_by_layer_rows,
            value_patch_ops_by_layer_rows=value_patch_ops_by_layer_rows,
            capture_hidden_layers=capture_hidden_layers,
            return_aux=return_aux,
        )


@torch.no_grad()
def _paged_forward_block_rows_unlocked(
    store: QStore,
    tokens: np.ndarray,
    row_sources: Sequence[tuple[BatchedPagedKVCache, int]],
    *,
    output_contract: PagedBlockOutputContract = "full_logits",
    selected_rows: Sequence[int] = (),
    selected_row_groups: Sequence[Sequence[int]] = (),
    last_only: bool = False,
    row_stable_requests: bool = False,
    split_source_attention: bool = False,
    batch_invariant_requests: bool = False,
    scratch_arithmetic: PagedPooledArithmetic | None = None,
    scratch_observer: Callable[[PagedPooledScratchTelemetry], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """Provisional B×K target forward against committed K/V — the weight-stationary
    verification lever: ONE weight traversal scores B requests × K provisional positions.

    Provisional K/V goes to SCRATCH tensors returned to the cache-specific wrapper (the
    committed arenas are untouched); attention runs over [committed ‖ provisional] with a
    per-row causal mask, so nothing is visible until :func:`commit_block`.
    The default ``full_logits`` contract preserves the original return shape ``[B, K, V]``.
    ``hidden_state_only`` stops after final normalization and returns ``[B, K, H]`` without
    touching ``lm_head``. ``selected_token_rows`` returns ``[B, K, R]`` and accesses the
    head only through ``embed_rows('lm_head', selected_rows)``; it never traverses full head
    blocks. With ``last_only=True``, the corresponding output is ``[B, H|R|V]`` and only the
    final provisional position enters the readout. Position k's scores predict the token AFTER
    ``tokens[:, k]``.

    Every output is detached to CPU. Every mode returns the same provisional payload and leaves
    cache tensors, lengths, and epoch unchanged until an explicit :func:`commit_block`. Same
    token-parity (not bit-exact) claim boundary and qwen2/llama/qwen3 scope as
    :func:`paged_forward_kv_batch`."""
    c = store.cfg
    d, nL = c["hidden_size"], c["num_hidden_layers"]
    nH, nKV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps, theta = c["rms_norm_eps"], c["rope_theta"]
    contract = str(output_contract)
    if contract not in _PAGED_BLOCK_OUTPUT_CONTRACTS:
        choices = ", ".join(sorted(_PAGED_BLOCK_OUTPUT_CONTRACTS))
        raise ValueError(f"output_contract must be one of: {choices}")
    output_rows = tuple(int(value) for value in selected_rows)
    if contract == "selected_token_rows":
        if not output_rows:
            raise ValueError("selected_token_rows requires selected_rows")
        if len(output_rows) != len(set(output_rows)):
            raise ValueError("selected_rows must be unique")
        vocab_size = int(c["vocab_size"])
        if min(output_rows) < 0 or max(output_rows) >= vocab_size:
            raise ValueError(f"selected_rows must be inside [0, {vocab_size})")
    elif output_rows:
        raise ValueError("selected_rows are only legal for selected_token_rows")
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [B, K]")
    B, K = (int(s) for s in tokens.shape)
    sources = tuple(row_sources)
    if B != len(sources):
        raise ValueError(f"batch {B} does not match paged KV row sources {len(sources)}")
    if B <= 0:
        raise ValueError("provisional batch cannot be empty")
    if K <= 0:
        raise ValueError("provisional block cannot be empty")
    row_groups = tuple(tuple(int(value) for value in group) for group in selected_row_groups)
    if row_groups:
        if contract != "selected_token_rows" or not row_stable_requests:
            raise ValueError(
                "selected_row_groups require selected_token_rows with row-stable arithmetic"
            )
        if len(row_groups) != B:
            raise ValueError("selected_row_groups must contain one group per request")
        output_row_set = set(output_rows)
        for group in row_groups:
            if not group:
                raise ValueError("selected row groups cannot be empty")
            if len(group) != len(set(group)):
                raise ValueError("each selected row group must be unique")
            if not set(group).issubset(output_row_set):
                raise ValueError("selected row groups must be subsets of selected_rows")
    parent_lengths: list[int] = []
    capacities: list[int] = []
    for source_cache, row in sources:
        if not isinstance(source_cache, BatchedPagedKVCache):
            raise TypeError("paged KV row source must contain a BatchedPagedKVCache")
        normalized_row = source_cache._validate_slot_row(row)  # noqa: SLF001
        if source_cache.k.ndim != 5 or source_cache.v.ndim != 5:
            raise ValueError("paged KV cache tensors must have rank five")
        if (
            not isinstance(source_cache.lengths, np.ndarray)
            or source_cache.lengths.dtype.kind not in {"i", "u"}
            or tuple(source_cache.lengths.shape) != (source_cache.B,)
        ):
            raise TypeError("paged KV cache lengths must be one integer per cache row")
        expected_shape = (nL, source_cache.B, source_cache.capacity, nKV, hd)
        if (
            tuple(source_cache.k.shape) != expected_shape
            or tuple(source_cache.v.shape) != expected_shape
        ):
            raise ValueError(f"paged KV cache tensor shape must be {expected_shape}")
        if source_cache.k.dtype != torch.float32 or source_cache.v.dtype != torch.float32:
            raise ValueError("paged KV cache tensors must use fp32")
        if source_cache.k.device != source_cache.v.device:
            raise ValueError("paged KV cache tensors must share one device")
        length = int(source_cache.lengths[normalized_row])
        if length < 0 or length > source_cache.capacity:
            raise ValueError("paged KV committed length is outside cache capacity")
        parent_lengths.append(length)
        capacities.append(source_cache.capacity)
    P = np.asarray(parent_lengths, dtype=np.int64)
    if any(
        parent + K > capacity for parent, capacity in zip(parent_lengths, capacities, strict=True)
    ):
        raise RuntimeError("provisional block exceeds one or more paged KV cache capacities")
    max_past = int(P.max())
    max_tot = max_past + K
    rep = nH // nKV
    scale = hd**-0.5
    if split_source_attention and not row_stable_requests:
        raise ValueError("split-source attention requires row-stable request arithmetic")
    if batch_invariant_requests and row_stable_requests:
        raise ValueError("batch-invariant and row-stable arithmetic are separate contracts")
    if scratch_observer is not None and not callable(scratch_observer):
        raise TypeError("scratch_observer must be callable")
    if scratch_observer is not None:
        if scratch_arithmetic is None:
            raise ValueError("scratch telemetry requires a named pooled arithmetic lane")
    fp32_bytes = torch.empty((), dtype=torch.float32).element_size()
    global_prefix_kv_logical_bytes = (
        0
        if split_source_attention or batch_invariant_requests
        else 2 * B * max_tot * nKV * hd * fp32_bytes
    )
    request_local_prefix_kv_logical_bytes_max = (
        2 * max_tot * nKV * hd * fp32_bytes if split_source_attention else 0
    )
    explicit_live_prefix_kv_peak_bytes = 0

    def observe_live_prefix_kv(*tensors: torch.Tensor) -> None:
        nonlocal explicit_live_prefix_kv_peak_bytes
        if scratch_observer is None:
            return
        live_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        explicit_live_prefix_kv_peak_bytes = max(
            explicit_live_prefix_kv_peak_bytes,
            live_bytes,
        )

    h = store.embed_rows("embed", tokens.reshape(-1)).clone().view(B, K, d)
    dev = h.device
    if any(source_cache.k.device != dev for source_cache, _row in sources):
        raise ValueError("paged KV caches and model store must share one device")
    pos = torch.as_tensor(P)[:, None] + torch.arange(K)[None, :]  # [B, K] global positions
    cos, sin = _rope_tables_rows(pos, hd, theta)
    cos, sin = cos.to(dev), sin.to(dev)
    kpos = torch.arange(max_tot, device=dev)
    allowed = kpos[None, None, :] <= pos.to(dev)[:, :, None]  # [B, K, max_tot]
    mask = torch.where(allowed, 0.0, float("-inf"))[:, None, :, :]  # [B, 1, K, max_tot]
    req = torch.arange(B, device=dev)[:, None]
    ppos = pos.to(dev)  # provisional key slots

    with torch.inference_mode(False):
        delta_k = torch.zeros((nL, B, K, nKV, hd), dtype=torch.float32, device=dev)
        delta_v = torch.zeros_like(delta_k)

    if batch_invariant_requests:
        # One immutable weight traversal for all rows; each row's bits depend only on that row.
        def request_norm(x: torch.Tensor, w: torch.Tensor, norm_eps: float) -> torch.Tensor:
            return batch_invariant_rms_norm(x, w, norm_eps)

        def request_matmul(name: str, x: torch.Tensor) -> torch.Tensor:
            return batch_invariant_matmul(x, store.weight(name))

        # Group rows by the committed arena they read so the attention kernel can address each
        # arena in place (StateCut branches all read one parent; pooled requests read their own).
        grouped_rows: dict[int, list[int]] = {}
        for b, (source_cache, _row) in enumerate(sources):
            grouped_rows.setdefault(id(source_cache), []).append(b)
        # Upload every group's row/source/past index once per forward: a pageable H2D copy
        # synchronizes the stream, so per-layer uploads would serialize the whole traversal.
        ordered_rows = [b for rows_of_group in grouped_rows.values() for b in rows_of_group]
        index_table = torch.as_tensor(
            np.asarray(
                [
                    ordered_rows,
                    [int(sources[b][1]) for b in ordered_rows],
                    [int(P[b]) for b in ordered_rows],
                ],
                dtype=np.int32,
            ),
            device=dev,
        )
        source_groups: list[tuple[BatchedPagedKVCache, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        offset = 0
        for rows_of_group in grouped_rows.values():
            end = offset + len(rows_of_group)
            source_groups.append(
                (
                    sources[rows_of_group[0]][0],
                    index_table[0, offset:end],
                    index_table[1, offset:end],
                    index_table[2, offset:end],
                )
            )
            offset = end
    else:
        request_norm = _row_stable_rms_norm if row_stable_requests else _rms_norm
        request_matmul = store.matmul_row_stable if row_stable_requests else store.matmul

    for L in range(nL):
        x = request_norm(h, store.fp32(f"L{L}.ln1"), eps)
        q = request_matmul(f"L{L}.q", x)
        k = request_matmul(f"L{L}.k", x)
        v = request_matmul(f"L{L}.v", x)
        if store.has(f"L{L}.q.bias"):
            q = q + store.fp32(f"L{L}.q.bias")
            k = k + store.fp32(f"L{L}.k.bias")
            v = v + store.fp32(f"L{L}.v.bias")
        q = q.view(B, K, nH, hd)
        k = k.view(B, K, nKV, hd)
        v = v.view(B, K, nKV, hd)
        if store.has(f"L{L}.q_norm"):  # qwen3: pre-RoPE (HF order)
            q = request_norm(q, store.fp32(f"L{L}.q_norm"), eps)
            k = request_norm(k, store.fp32(f"L{L}.k_norm"), eps)
        q = _apply_rope_rows(q, cos, sin)
        k = _apply_rope_rows(k, cos, sin)
        delta_k[L] = k  # scratch, NOT the arena
        delta_v[L] = v
        keys: torch.Tensor | None = None
        vals: torch.Tensor | None = None
        if not split_source_attention and not batch_invariant_requests:
            # Batch-wide source assembly used by the established packed and row-stable lanes.
            keys = torch.zeros((B, max_tot, nKV, hd), dtype=torch.float32, device=dev)
            vals = torch.zeros_like(keys)
            observe_live_prefix_kv(keys, vals)
            if max_past:
                for b, (source_cache, row) in enumerate(sources):
                    past = int(P[b])
                    if not past:
                        continue
                    keys[b, :past] = source_cache.k[L, int(row), :past]
                    vals[b, :past] = source_cache.v[L, int(row), :past]
            keys[req, ppos] = k
            vals[req, ppos] = v
            if rep > 1:  # GQA expand at attn time
                expanded_keys = keys.repeat_interleave(rep, dim=2)
                observe_live_prefix_kv(keys, vals, expanded_keys)
                keys = expanded_keys
                del expanded_keys
                expanded_vals = vals.repeat_interleave(rep, dim=2)
                observe_live_prefix_kv(keys, vals, expanded_vals)
                vals = expanded_vals
                del expanded_vals
        if batch_invariant_requests:
            ctx4 = torch.zeros((B, K, nH, hd), dtype=torch.float32, device=dev)
            q_c = q.contiguous()
            k_c = k.contiguous()
            v_c = v.contiguous()
            for group_cache, group_rows, group_sources, group_past in source_groups:
                batch_invariant_attention(
                    q_c,
                    k_c,
                    v_c,
                    group_cache.k[L],
                    group_cache.v[L],
                    group_rows,
                    group_sources,
                    group_past,
                    out=ctx4,
                )
            ctx = ctx4.reshape(B, K, nH * hd)
            del q_c, k_c, v_c, ctx4
        elif row_stable_requests:
            # A pooled row with a shorter committed prefix must not inherit the longer sibling's
            # softmax reduction width.  Use its exact B=1 key extent and retain a singleton batch
            # axis; Q/K/V projections above still loaded every immutable component only once.
            contexts = []
            for row in range(B):
                total = int(P[row]) + K
                if split_source_attention:
                    source_cache, source_row = sources[row]
                    past = int(P[row])
                    # Assemble only this request's exact logical source.  No sibling padding or
                    # batch-wide committed-prefix buffer exists in this separately named lane.
                    key_row = torch.cat(
                        (source_cache.k[L, int(source_row), :past], k[row]),
                        dim=0,
                    ).unsqueeze(0)
                    value_row = torch.cat(
                        (source_cache.v[L, int(source_row), :past], v[row]),
                        dim=0,
                    ).unsqueeze(0)
                    observe_live_prefix_kv(key_row, value_row)
                    if rep > 1:
                        expanded_key_row = key_row.repeat_interleave(rep, dim=2)
                        observe_live_prefix_kv(key_row, value_row, expanded_key_row)
                        key_row = expanded_key_row
                        del expanded_key_row
                        expanded_value_row = value_row.repeat_interleave(rep, dim=2)
                        observe_live_prefix_kv(key_row, value_row, expanded_value_row)
                        value_row = expanded_value_row
                        del expanded_value_row
                else:
                    assert keys is not None and vals is not None
                    key_row = keys[row : row + 1, :total]
                    value_row = vals[row : row + 1, :total]
                # Slicing a global-max arena can preserve the longer sibling's leading stride,
                # which is enough to select a different B1 matmul kernel at K=1.  Materialize
                # each exact request-local shape before either attention reduction.
                # ``contiguous()`` may legally return the original view when all dimensions
                # carrying the inherited stride are singleton (notably K=1).  A forced clone
                # is required to make the physical request-local layout match independent B1.
                qh_row = (
                    q[row : row + 1].clone(memory_format=torch.contiguous_format).transpose(1, 2)
                )
                kh_row = key_row.clone(memory_format=torch.contiguous_format).transpose(1, 2)
                if split_source_attention:
                    observe_live_prefix_kv(key_row, value_row, kh_row)
                else:
                    assert keys is not None and vals is not None
                    observe_live_prefix_kv(keys, vals, kh_row)
                vh_row = value_row.clone(memory_format=torch.contiguous_format).transpose(1, 2)
                if split_source_attention:
                    observe_live_prefix_kv(key_row, value_row, kh_row, vh_row)
                else:
                    assert keys is not None and vals is not None
                    observe_live_prefix_kv(keys, vals, kh_row, vh_row)
                mask_row = mask[row : row + 1, :, :, :total].clone(
                    memory_format=torch.contiguous_format
                )
                scores_row = torch.matmul(qh_row, kh_row.transpose(-1, -2)) * scale + mask_row
                probs_row = torch.softmax(scores_row, dim=-1)
                contexts.append(
                    torch.matmul(probs_row, vh_row).transpose(1, 2).reshape(1, K, nH * hd)
                )
                del qh_row, kh_row, vh_row, mask_row, scores_row, probs_row
                if split_source_attention:
                    del key_row, value_row
            ctx = torch.cat(tuple(contexts), dim=0)
            del contexts
        else:
            assert keys is not None and vals is not None
            qh = q.transpose(1, 2)  # [B, nH, K, hd]
            kh = keys.transpose(1, 2)
            vh = vals.transpose(1, 2)
            scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale + mask
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, vh.float()).transpose(1, 2).reshape(B, K, nH * hd)
            del qh, kh, vh, scores, probs
        attn_out = request_matmul(f"L{L}.o", ctx)
        if store.has(f"L{L}.o.bias"):
            attn_out = attn_out + store.fp32(f"L{L}.o.bias")
        del q, k, v, keys, vals, ctx
        h = h + attn_out

        x2 = request_norm(h, store.fp32(f"L{L}.ln2"), eps)
        g = request_matmul(f"L{L}.gate", x2)
        u = request_matmul(f"L{L}.up", x2)
        hid = (
            torch.cat(
                tuple(
                    torch.nn.functional.silu(g[row : row + 1]) * u[row : row + 1]
                    for row in range(B)
                ),
                dim=0,
            )
            if row_stable_requests
            else torch.nn.functional.silu(g) * u
        )
        h = h + request_matmul(f"L{L}.down", hid)
        del g, u, hid

    h = request_norm(h, store.fp32("norm.final"), eps)  # [B, K, d]
    readout = h[:, -1] if last_only else h
    if contract == "hidden_state_only":
        output = readout
    elif contract == "selected_token_rows":
        weights = store.embed_rows(
            "lm_head",
            np.asarray(output_rows, dtype=np.int64),
        ).to(device=dev, dtype=torch.float32)
        expected_shape = (len(output_rows), d)
        if tuple(weights.shape) != expected_shape:
            raise RuntimeError(
                f"selected lm_head rows have shape {tuple(weights.shape)}, "
                f"expected {expected_shape}"
            )
        if row_groups:
            # Preserve each request's independent selected-head width/order while loading the
            # stable union only once.  Undemanded union columns remain zero and never escape the
            # reactor's per-child projection.
            output = torch.zeros(
                (*readout.shape[:-1], len(output_rows)),
                dtype=torch.float32,
                device=dev,
            )
            offsets = {token: index for index, token in enumerate(output_rows)}
            for row, group in enumerate(row_groups):
                columns = torch.as_tensor(
                    [offsets[token] for token in group],
                    dtype=torch.long,
                    device=dev,
                )
                request_weights = weights.index_select(0, columns).contiguous()
                request_output = readout[row : row + 1].float() @ request_weights.T
                output[row].index_copy_(-1, columns, request_output[0])
        elif batch_invariant_requests:
            output = batch_invariant_matmul(readout.float(), weights)
        else:
            output = (
                _row_stable_loaded_matmul(readout.float(), weights)
                if row_stable_requests
                else readout.float() @ weights.T
            )
    else:
        vocab_size = int(c["vocab_size"])
        output = torch.empty(
            (*readout.shape[:-1], vocab_size),
            dtype=torch.float32,
            device=dev,
        )
        for start, end, weights in store.row_blocks("lm_head"):  # B*K vs head stream
            if batch_invariant_requests:
                output[..., start:end] = batch_invariant_matmul(readout, weights)
            else:
                output[..., start:end] = (
                    _row_stable_loaded_matmul(readout, weights)
                    if row_stable_requests
                    else readout @ weights.T
                )
            del weights
    if scratch_observer is not None:
        assert scratch_arithmetic is not None
        scratch_observer(
            PagedPooledScratchTelemetry(
                arithmetic=scratch_arithmetic,
                batch_size=B,
                token_count=K,
                parent_lengths=tuple(parent_lengths),
                global_prefix_kv_logical_bytes=global_prefix_kv_logical_bytes,
                request_local_prefix_kv_logical_bytes_max=(
                    request_local_prefix_kv_logical_bytes_max
                ),
                explicit_live_prefix_kv_peak_bytes=explicit_live_prefix_kv_peak_bytes,
                aggregate_provisional_delta_bytes=(2 * nL * B * K * nKV * hd * fp32_bytes),
            )
        )
    return output.detach().cpu(), delta_k, delta_v, tuple(parent_lengths), K


def _paged_forward_block_unlocked(
    store: QStore,
    tokens: np.ndarray,
    cache: BatchedPagedKVCache,
    *,
    output_contract: PagedBlockOutputContract = "full_logits",
    selected_rows: Sequence[int] = (),
    last_only: bool = False,
) -> tuple[torch.Tensor, PagedKVDelta]:
    """Adapt the row-source kernel to the original one-cache batched API."""

    output, delta_k, delta_v, parent_lengths, token_count = _paged_forward_block_rows_unlocked(
        store,
        tokens,
        tuple((cache, row) for row in range(cache.B)),
        output_contract=output_contract,
        selected_rows=selected_rows,
        last_only=last_only,
    )
    return output, PagedKVDelta(
        parent_epoch=cache.epoch,
        parent_lengths=parent_lengths,
        cache_id=cache.cache_id,
        k=delta_k,
        v=delta_v,
        token_count=token_count,
    )


def paged_forward_block(
    store: QStore,
    tokens: np.ndarray,
    cache: BatchedPagedKVCache,
    *,
    output_contract: PagedBlockOutputContract = "full_logits",
    selected_rows: Sequence[int] = (),
    last_only: bool = False,
) -> tuple[torch.Tensor, PagedKVDelta]:
    """Run a scratch-only provisional block under the cache's single-writer lease."""

    with cache._lock:  # noqa: SLF001 - the kernel and cache implement one transaction
        cache.assert_usable()
        return _paged_forward_block_unlocked(
            store,
            tokens,
            cache,
            output_contract=output_contract,
            selected_rows=selected_rows,
            last_only=last_only,
        )


@torch.no_grad()
def paged_forward_statecut_branches(
    store: QStore,
    tokens: np.ndarray,
    cache: BatchedPagedKVCache,
    *,
    output_contract: PagedBlockOutputContract = "hidden_state_only",
    selected_rows: Sequence[int] = (),
    arithmetic: PagedPooledArithmetic = "batch_invariant",
) -> tuple[torch.Tensor, PagedKVForkPanel]:
    """Continue N branches one token from one immutable B-row parent StateCut.

    ``tokens`` has shape ``[branches, parent_batch]``. The parent row sources are referenced,
    never copied; all branch diffs share one aggregate scratch allocation and one immutable
    weight traversal. The batch-invariant lane is the default: every branch's bits are independent
    of the panel width, and the parent arena is read in place. ``row_stable_split`` keeps the
    earlier cuBLAS-B=1 contract for replaying older receipts. Nothing commits here.
    """

    if not isinstance(cache, BatchedPagedKVCache):
        raise TypeError("statecut parent must be a BatchedPagedKVCache")
    token_array = np.asarray(tokens, dtype=np.int64)
    if token_array.ndim != 2:
        raise ValueError("statecut branch tokens must have shape [branches,parent_batch]")
    branch_count, batch_size = (int(value) for value in token_array.shape)
    if branch_count <= 0:
        raise ValueError("statecut branch panel cannot be empty")
    if batch_size != cache.B:
        raise ValueError("statecut branch token rows do not match the parent batch")
    if arithmetic not in _PAGED_POOLED_ARITHMETIC:
        choices = ", ".join(sorted(_PAGED_POOLED_ARITHMETIC))
        raise ValueError(f"statecut arithmetic must be one of: {choices}")
    row_stable_requests = arithmetic in {"row_stable", "row_stable_split"}
    if row_stable_requests and not callable(getattr(store, "matmul_row_stable", None)):
        raise TypeError("row-stable statecut execution requires store.matmul_row_stable")
    batch_invariant_requests = arithmetic == "batch_invariant"

    with cache._lock:  # noqa: SLF001 - the kernel/cache form one transaction boundary
        cache.assert_usable()
        parent_epoch = int(cache.epoch)
        parent_lengths = tuple(int(value) for value in cache.lengths)
        parent_signature = tuple(
            (
                id(tensor),
                int(tensor.data_ptr()),
                tuple(int(value) for value in tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                bool(tensor.is_contiguous()),
                int(tensor._version),  # noqa: SLF001 - immutable-parent guard
            )
            for tensor in (cache.k, cache.v)
        )
        flat_tokens = token_array.reshape(branch_count * batch_size, 1)
        row_sources = tuple(
            (cache, row) for _branch in range(branch_count) for row in range(batch_size)
        )
        output, delta_k, delta_v, repeated_lengths, token_count = (
            _paged_forward_block_rows_unlocked(
                store,
                flat_tokens,
                row_sources,
                output_contract=output_contract,
                selected_rows=selected_rows,
                last_only=True,
                row_stable_requests=row_stable_requests,
                split_source_attention=arithmetic == "row_stable_split",
                batch_invariant_requests=batch_invariant_requests,
                scratch_arithmetic=arithmetic,
            )
        )
        expected_repeated = parent_lengths * branch_count
        if repeated_lengths != expected_repeated:
            raise RuntimeError("statecut branch execution returned mismatched parent lengths")
        parent_signature_after = tuple(
            (
                id(tensor),
                int(tensor.data_ptr()),
                tuple(int(value) for value in tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                bool(tensor.is_contiguous()),
                int(tensor._version),  # noqa: SLF001 - immutable-parent guard
            )
            for tensor in (cache.k, cache.v)
        )
        if (
            int(cache.epoch) != parent_epoch
            or tuple(int(value) for value in cache.lengths) != parent_lengths
            or parent_signature_after != parent_signature
        ):
            raise RuntimeError("statecut continuation mutated its immutable parent")

    panel = PagedKVForkPanel(
        parent_epoch=parent_epoch,
        parent_lengths=parent_lengths,
        cache_id=cache.cache_id,
        branch_count=branch_count,
        batch_size=batch_size,
        k=delta_k,
        v=delta_v,
        token_count=token_count,
    )
    return output.reshape(branch_count, batch_size, *output.shape[1:]), panel


@torch.no_grad()
def paged_forward_statecut_branch_blocks(
    store: QStore,
    tokens: np.ndarray,
    cache: BatchedPagedKVCache,
    *,
    output_contract: PagedBlockOutputContract = "hidden_state_only",
    selected_rows: Sequence[int] = (),
    arithmetic: PagedPooledArithmetic = "batch_invariant",
    last_only: bool = True,
) -> tuple[torch.Tensor, PagedKVForkPanel]:
    """Continue N branches by one provisional K-token block.

    ``tokens`` has shape ``[branches, parent_batch, K]``. All K tokens remain in
    branch-local scratch until :func:`commit_block` atomically admits either the
    complete block or a caller-selected prefix. The immutable parent is referenced,
    never copied. This is the block generalization of
    :func:`paged_forward_statecut_branches`; the one-token API remains its stable
    special case.
    """

    if not isinstance(cache, BatchedPagedKVCache):
        raise TypeError("statecut parent must be a BatchedPagedKVCache")
    token_array = np.asarray(tokens, dtype=np.int64)
    if token_array.ndim != 3:
        raise ValueError(
            "statecut branch block tokens must have shape [branches,parent_batch,tokens]"
        )
    branch_count, batch_size, token_count = (
        int(value) for value in token_array.shape
    )
    if branch_count <= 0 or token_count <= 0:
        raise ValueError("statecut branch block panel and token count must be non-empty")
    if batch_size != cache.B:
        raise ValueError("statecut branch block rows do not match the parent batch")
    if arithmetic not in _PAGED_POOLED_ARITHMETIC:
        choices = ", ".join(sorted(_PAGED_POOLED_ARITHMETIC))
        raise ValueError(f"statecut arithmetic must be one of: {choices}")
    row_stable_requests = arithmetic in {"row_stable", "row_stable_split"}
    if row_stable_requests and not callable(getattr(store, "matmul_row_stable", None)):
        raise TypeError("row-stable statecut execution requires store.matmul_row_stable")
    batch_invariant_requests = arithmetic == "batch_invariant"

    with cache._lock:  # noqa: SLF001 - the kernel/cache form one transaction boundary
        cache.assert_usable()
        parent_epoch = int(cache.epoch)
        parent_lengths = tuple(int(value) for value in cache.lengths)
        if any(length + token_count > cache.capacity for length in parent_lengths):
            raise RuntimeError("BatchedPagedKVCache overflow on provisional block")
        parent_signature = tuple(
            (
                id(tensor),
                int(tensor.data_ptr()),
                tuple(int(value) for value in tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                bool(tensor.is_contiguous()),
                int(tensor._version),  # noqa: SLF001 - immutable-parent guard
            )
            for tensor in (cache.k, cache.v)
        )
        flat_tokens = token_array.reshape(branch_count * batch_size, token_count)
        row_sources = tuple(
            (cache, row) for _branch in range(branch_count) for row in range(batch_size)
        )
        output, delta_k, delta_v, repeated_lengths, returned_token_count = (
            _paged_forward_block_rows_unlocked(
                store,
                flat_tokens,
                row_sources,
                output_contract=output_contract,
                selected_rows=selected_rows,
                last_only=last_only,
                row_stable_requests=row_stable_requests,
                split_source_attention=arithmetic == "row_stable_split",
                batch_invariant_requests=batch_invariant_requests,
                scratch_arithmetic=arithmetic,
            )
        )
        expected_repeated = parent_lengths * branch_count
        if repeated_lengths != expected_repeated or returned_token_count != token_count:
            raise RuntimeError("statecut block execution returned mismatched geometry")
        parent_signature_after = tuple(
            (
                id(tensor),
                int(tensor.data_ptr()),
                tuple(int(value) for value in tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                bool(tensor.is_contiguous()),
                int(tensor._version),  # noqa: SLF001 - immutable-parent guard
            )
            for tensor in (cache.k, cache.v)
        )
        if (
            int(cache.epoch) != parent_epoch
            or tuple(int(value) for value in cache.lengths) != parent_lengths
            or parent_signature_after != parent_signature
        ):
            raise RuntimeError("statecut block continuation mutated its immutable parent")

    panel = PagedKVForkPanel(
        parent_epoch=parent_epoch,
        parent_lengths=parent_lengths,
        cache_id=cache.cache_id,
        branch_count=branch_count,
        batch_size=batch_size,
        k=delta_k,
        v=delta_v,
        token_count=token_count,
    )
    return output.reshape(branch_count, batch_size, *output.shape[1:]), panel


@torch.no_grad()
def paged_forward_block_pooled(
    store: QStore,
    tokens: np.ndarray,
    caches: Sequence[BatchedPagedKVCache],
    leases: Sequence[PagedKVSlotLease],
    *,
    output_contract: PagedBlockOutputContract = "full_logits",
    selected_rows: Sequence[int] = (),
    selected_row_groups: Sequence[Sequence[int]] = (),
    last_only: bool = False,
    arithmetic: PagedPooledArithmetic = "packed",
    scratch_observer: Callable[[PagedPooledScratchTelemetry], None] | None = None,
) -> tuple[torch.Tensor, tuple[PagedKVDelta, ...]]:
    """Score independent singleton caches in one weight-stationary ``B x K`` traversal.

    Each request keeps its own B=1 arena and cache-issued slot lease.  All cache locks are
    acquired by an immutable global key, leases are validated while those locks remain held,
    and committed KV is read directly from each arena.  The kernel never commits.  Only each
    request's new-token KV slice is cloned into its returned ordinary :class:`PagedKVDelta`, so
    siblings may commit unequal prefixes independently with :func:`commit_block`.

    ``arithmetic='packed'`` retains the original BxK GEMM lane.  The explicit
    ``arithmetic='row_stable'`` lane still loads/dequantizes each immutable component once, but
    applies it with the exact singleton-request matmul shape and uses each request's own attention
    reduction width.  ``arithmetic='row_stable_split'`` keeps that numerical lane while assembling
    committed-plus-provisional K/V one exact request at a time, eliminating the batch-wide
    ``B x max(prefix + K)`` source buffers.  These are separately named contracts and are never
    silently selected by this API.

    ``selected_rows`` is one shared, stable row order for the physical traversal.  In the
    row-stable lane, optional ``selected_row_groups`` preserves each request's independent
    selected-head width/order while still reading/dequantizing that shared union once.  Forming
    the stable union and projecting outputs back to individual orders belongs to the reactor.
    """

    if not isinstance(arithmetic, str) or arithmetic not in _PAGED_POOLED_ARITHMETIC:
        choices = ", ".join(sorted(_PAGED_POOLED_ARITHMETIC))
        raise ValueError(f"pooled arithmetic must be one of: {choices}")
    row_stable_requests = arithmetic in {"row_stable", "row_stable_split"}
    split_source_attention = arithmetic == "row_stable_split"
    batch_invariant_requests = arithmetic == "batch_invariant"
    if row_stable_requests and not callable(getattr(store, "matmul_row_stable", None)):
        raise TypeError("row-stable pooled arithmetic requires store.matmul_row_stable")

    cache_tuple = tuple(caches)
    lease_tuple = tuple(leases)
    token_array = np.asarray(tokens, dtype=np.int64)
    if token_array.ndim != 2:
        raise ValueError("tokens must have shape [B, K]")
    batch = int(token_array.shape[0])
    if batch <= 0:
        raise ValueError("provisional batch cannot be empty")
    if len(cache_tuple) != batch or len(lease_tuple) != batch:
        raise ValueError("tokens, singleton caches, and slot leases must have equal batch width")
    if any(not isinstance(cache, BatchedPagedKVCache) for cache in cache_tuple):
        raise TypeError("pooled cache entries must be BatchedPagedKVCache instances")
    if any(not isinstance(lease, PagedKVSlotLease) for lease in lease_tuple):
        raise TypeError("pooled lease entries must be PagedKVSlotLease instances")
    if any(cache.B != 1 for cache in cache_tuple):
        raise ValueError("pooled paged execution requires one B=1 cache per request")
    if len({id(cache) for cache in cache_tuple}) != batch:
        raise ValueError("pooled paged execution requires distinct request caches")
    if len({cache.cache_id for cache in cache_tuple}) != batch:
        raise ValueError("pooled paged execution requires distinct cache identities")
    _require_disjoint_tensor_storage(
        tuple(tensor for cache in cache_tuple for tensor in (cache.k, cache.v)),
        field="pooled committed KV",
    )
    if len({lease.lease_id for lease in lease_tuple}) != batch:
        raise ValueError("duplicate pooled paged KV slot lease")

    lock_order = sorted(cache_tuple, key=lambda cache: (cache._lock_order_key, id(cache)))  # noqa: SLF001
    with ExitStack() as stack:
        for cache in lock_order:
            stack.enter_context(cache._lock)  # noqa: SLF001 - exact multi-cache lock order
        for cache, lease in zip(cache_tuple, lease_tuple, strict=True):
            cache.assert_usable()
            cache._validate_slot_lease_unlocked(lease)  # noqa: SLF001
            if lease.row != 0:
                raise ValueError("pooled singleton cache lease must name row zero")

        output, delta_k, delta_v, parent_lengths, token_count = _paged_forward_block_rows_unlocked(
            store,
            token_array,
            tuple((cache, 0) for cache in cache_tuple),
            output_contract=output_contract,
            selected_rows=selected_rows,
            selected_row_groups=selected_row_groups,
            last_only=last_only,
            row_stable_requests=row_stable_requests,
            split_source_attention=split_source_attention,
            batch_invariant_requests=batch_invariant_requests,
            scratch_arithmetic=arithmetic,
            scratch_observer=scratch_observer,
        )
        # A view into the pooled scratch would couple sibling lifetimes and is not contiguous.
        # Clone only the K provisional positions, never either committed arena.
        with torch.inference_mode(False):
            child_k = tuple(delta_k[:, row : row + 1].clone() for row in range(batch))
            child_v = tuple(delta_v[:, row : row + 1].clone() for row in range(batch))
        children = tuple(
            PagedKVDelta(
                parent_epoch=cache.epoch,
                parent_lengths=(parent_lengths[row],),
                cache_id=cache.cache_id,
                k=child_k[row],
                v=child_v[row],
                token_count=token_count,
                slot_lease=lease,
            )
            for row, (cache, lease) in enumerate(zip(cache_tuple, lease_tuple, strict=True))
        )
        return output, children


def _commit_block_unlocked(
    cache: BatchedPagedKVCache,
    delta: PagedKVDelta,
    accepted_lens,
) -> tuple[int, ...]:
    """Move each row's accepted provisional prefix into the committed arena.

    Transactional per the dense-qstore contract: the delta must have been forwarded from
    the cache's CURRENT epoch and committed lengths — a stale delta (any committed
    mutation since, including a second commit of the same delta) raises RuntimeError.
    Returns the accepted counts."""
    cache.assert_usable()
    if delta.cache_id != cache.cache_id:
        raise RuntimeError("KV delta belongs to a different paged cache")
    if delta.slot_lease is not None:
        cache._validate_slot_lease_unlocked(delta.slot_lease)  # noqa: SLF001
    counts_list: list[int] = []
    for value in accepted_lens:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("accepted length must be an integer")
        counts_list.append(int(value))
    counts = tuple(counts_list)
    if isinstance(cache.epoch, bool) or not isinstance(cache.epoch, Integral):
        raise TypeError("paged cache epoch must be an integer")
    if not isinstance(cache.lengths, np.ndarray) or cache.lengths.dtype.kind not in {"i", "u"}:
        raise TypeError("paged cache lengths must be an integer NumPy array")
    if tuple(cache.lengths.shape) != (cache.B,):
        raise ValueError("paged cache lengths must have one entry per batch row")
    if isinstance(delta.parent_epoch, bool) or not isinstance(delta.parent_epoch, Integral):
        raise TypeError("KV delta parent_epoch must be an integer")
    if not isinstance(delta.parent_lengths, tuple) or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in delta.parent_lengths
    ):
        raise TypeError("KV delta parent_lengths must be an integer tuple")
    if int(delta.parent_epoch) != int(cache.epoch):
        raise RuntimeError(
            f"stale KV delta epoch {delta.parent_epoch}; cache is at epoch {cache.epoch}"
        )
    if delta.parent_lengths != tuple(int(value) for value in cache.lengths):
        raise RuntimeError("KV delta parent lengths do not match committed cache")
    if isinstance(delta.token_count, bool) or not isinstance(delta.token_count, Integral):
        raise TypeError("KV delta token_count must be an integer")
    token_count = int(delta.token_count)
    if token_count <= 0:
        raise ValueError("KV delta token_count must be positive")
    if cache.k.ndim != 5 or cache.v.ndim != 5:
        raise ValueError("paged KV cache tensors must have rank five")
    expected_cache_shape = (
        int(cache.k.shape[0]),
        cache.B,
        cache.capacity,
        int(cache.k.shape[3]),
        int(cache.k.shape[4]),
    )
    if tuple(cache.k.shape) != expected_cache_shape or tuple(cache.v.shape) != expected_cache_shape:
        raise ValueError(f"KV cache tensor shape must be {expected_cache_shape}")
    if cache.k.dtype != torch.float32 or cache.v.dtype != cache.k.dtype:
        raise ValueError("paged KV cache tensors must share fp32 dtype")
    if cache.v.device != cache.k.device:
        raise ValueError("paged KV cache tensors must share one device")
    if not cache.k.is_contiguous() or not cache.v.is_contiguous():
        raise ValueError("paged KV cache tensors must be contiguous")
    expected_shape = (
        int(cache.k.shape[0]),
        cache.B,
        token_count,
        int(cache.k.shape[3]),
        int(cache.k.shape[4]),
    )
    try:
        current_signature = tuple(
            (
                id(tensor),
                int(tensor.data_ptr()),
                tuple(int(value) for value in tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                bool(tensor.is_contiguous()),
                int(tensor._version),  # noqa: SLF001 - detects post-forward mutation
            )
            for tensor in (delta.k, delta.v)
        )
    except RuntimeError as exc:
        raise TypeError("provisional KV tensors must track mutation versions") from exc
    if current_signature != delta.tensor_signature:
        raise RuntimeError("provisional KV delta tensors changed after creation")
    if any(tuple(tensor.shape) != expected_shape for tensor in (delta.k, delta.v)):
        raise ValueError(f"KV delta tensor shape must be {expected_shape}")
    if any(tensor.dtype != cache.k.dtype for tensor in (delta.k, delta.v)):
        raise ValueError("KV delta dtype does not match cache")
    if any(tensor.device != cache.k.device for tensor in (delta.k, delta.v)):
        raise ValueError("KV delta device does not match cache")
    if any(not tensor.is_contiguous() for tensor in (delta.k, delta.v)):
        raise ValueError("KV delta tensors must be contiguous")
    if any(tensor.requires_grad for tensor in (cache.k, cache.v, delta.k, delta.v)):
        raise ValueError("KV cache and delta tensors must not require gradients")
    _require_separate_kv_storage(cache.k, cache.v, field="committed KV")
    _require_separate_kv_storage(delta.k, delta.v, field="provisional KV")
    cache_storage = cache.k.untyped_storage().data_ptr(), cache.v.untyped_storage().data_ptr()
    delta_storage = delta.k.untyped_storage().data_ptr(), delta.v.untyped_storage().data_ptr()
    if set(cache_storage) & set(delta_storage):
        raise ValueError("provisional KV tensors must not alias committed KV storage")
    if len(counts) != cache.B:
        raise ValueError("accepted length must be supplied for every request slot")
    if any(value < 0 or value > token_count for value in counts):
        raise ValueError("accepted length outside provisional block")
    if any(int(cache.lengths[b]) + count > cache.capacity for b, count in enumerate(counts)):
        raise RuntimeError("BatchedPagedKVCache overflow on commit")

    # Stage from the caller-owned provisional tensors before touching committed storage.
    # A writer racing either clone increments the source version; the second signature check
    # then fails while the arena is still byte-for-byte unchanged. Once staged, the commit no
    # longer reads mutable external storage.
    cache_signature_before_stage = tuple(
        (
            id(tensor),
            int(tensor.data_ptr()),
            tuple(int(value) for value in tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
            bool(tensor.is_contiguous()),
            int(tensor._version),  # noqa: SLF001 - concurrent state mutation guard
        )
        for tensor in (cache.k, cache.v)
    )
    with torch.inference_mode(False):
        staged_k = delta.k.detach().clone(memory_format=torch.contiguous_format)
        staged_v = delta.v.detach().clone(memory_format=torch.contiguous_format)
    signature_after_stage = tuple(
        (
            id(tensor),
            int(tensor.data_ptr()),
            tuple(int(value) for value in tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
            bool(tensor.is_contiguous()),
            int(tensor._version),  # noqa: SLF001 - concurrent provisional mutation guard
        )
        for tensor in (delta.k, delta.v)
    )
    if signature_after_stage != current_signature:
        raise RuntimeError("provisional KV delta changed while staging; commit was not applied")
    cache_signature_after_stage = tuple(
        (
            id(tensor),
            int(tensor.data_ptr()),
            tuple(int(value) for value in tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
            bool(tensor.is_contiguous()),
            int(tensor._version),  # noqa: SLF001 - concurrent state mutation guard
        )
        for tensor in (cache.k, cache.v)
    )
    if cache_signature_after_stage != cache_signature_before_stage:
        raise RuntimeError("committed KV changed while staging; commit was not applied")
    for b, count in enumerate(counts):
        if not count:
            continue
        start = int(cache.lengths[b])
        cache.k[:, b, start : start + count] = staged_k[:, b, :count]
        cache.v[:, b, start : start + count] = staged_v[:, b, :count]
    cache.lengths = cache.lengths + np.asarray(counts, dtype=np.int64)
    cache.epoch += 1
    return counts


@torch.no_grad()
def commit_block(
    cache: BatchedPagedKVCache,
    delta: PagedKVDelta,
    accepted_lens,
) -> tuple[int, ...]:
    """Atomically validate and commit an accepted provisional prefix."""

    with cache._lock:  # noqa: SLF001 - the kernel and cache implement one transaction
        return _commit_block_unlocked(cache, delta, accepted_lens)


# =================================================================== gpt_neox (pythia) math
def _layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * w + b


def _tap(hid, L, patch_ops_by_layer, capture_selected_maps, captured_selected, collect_acts):
    """Apply the replay patch / capture / activation-collect contract at a down-proj input.
    Identical semantics across all paged forwards (the engine's per-neuron tap point)."""
    if patch_ops_by_layer and L in patch_ops_by_layer:
        hid = _apply_patch_ops(hid, patch_ops_by_layer[L])
    if capture_selected_maps and captured_selected is not None and L in capture_selected_maps:
        locals_ = capture_selected_maps[L].get("locals", [])
        if locals_:
            cols = torch.as_tensor(locals_, dtype=torch.long, device=hid.device)
            captured_selected[L] = hid[..., cols].detach().to(dtype=torch.float16, device="cpu")
    if collect_acts is not None:
        collect_acts.append(hid.detach().clone())
    return hid


@torch.no_grad()
def paged_logits_neox(
    store: QStore,
    input_ids: np.ndarray,
    collect_acts: list | None = None,
    collect_attn: list | None = None,
    patch_ops_by_layer: dict[int, list] | None = None,
    capture_selected_maps: dict[int, dict] | None = None,
    captured_selected: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """GPT-NeoX (pythia) paged forward. LayerNorm (with bias), packed per-head QKV, partial
    rotary on the first ``rotary_ndims`` of each head, parallel residual, erf-GELU MLP. The tap
    (patch/capture/collect) is the post-GELU activation feeding ``dense_4h_to_h`` (= ``L{L}.down``)."""
    c = store.cfg
    d, nL = c["hidden_size"], c["num_hidden_layers"]
    nH, hd, rot = c["num_attention_heads"], c["head_dim"], c["rotary_ndims"]
    eps, theta = c["layer_norm_eps"], c["rope_theta"]
    parallel = bool(c.get("use_parallel_residual", True))
    T = len(input_ids)
    scale = hd**-0.5

    h = store.embed_rows("embed", input_ids).clone()  # [T, d]
    cos, sin = _rope_tables(T, rot, theta)  # tables on the rotary dims only
    causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)

    for L in range(nL):
        x1 = _layer_norm(h, store.fp32(f"L{L}.ln1"), store.fp32(f"L{L}.ln1.bias"), eps)
        qkv = store.matmul(f"L{L}.qkv", x1) + store.fp32(f"L{L}.qkv.bias")  # [T, 3*d]
        qkv = qkv.view(T, nH, 3 * hd)  # per-head interleaved
        q, k, v = qkv.split(hd, dim=-1)  # each [T, nH, hd]
        q = torch.cat([_apply_rope(q[..., :rot], cos, sin), q[..., rot:]], dim=-1)
        k = torch.cat([_apply_rope(k[..., :rot], cos, sin), k[..., rot:]], dim=-1)
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # [nH,T,hd]
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale + causal
        probs = torch.softmax(scores, dim=-1)
        if collect_attn is not None:
            collect_attn.append(probs.detach().clone())  # [nH, T, T]
        ctx = torch.matmul(probs, vh.float()).transpose(0, 1).reshape(T, nH * hd)  # [T, d]
        attn_out = store.matmul(f"L{L}.o", ctx) + store.fp32(f"L{L}.o.bias")
        del q, k, v, qh, kh, vh, scores, probs, ctx

        mlp_base = h if parallel else (h + attn_out)  # parallel: MLP reads pre-attn h
        x2 = _layer_norm(mlp_base, store.fp32(f"L{L}.ln2"), store.fp32(f"L{L}.ln2.bias"), eps)
        hid = torch.nn.functional.gelu(
            store.matmul(f"L{L}.h_to_4h", x2) + store.fp32(f"L{L}.h_to_4h.bias")
        )  # [T, inter]
        hid = _tap(
            hid, L, patch_ops_by_layer, capture_selected_maps, captured_selected, collect_acts
        )
        mlp_out = store.matmul(f"L{L}.down", hid) + store.fp32(f"L{L}.down.bias")
        del hid
        h = h + attn_out + mlp_out  # parallel & sequential agree here

    h = _layer_norm(h, store.fp32("norm.final"), store.fp32("norm.final.bias"), eps)
    V = c["vocab_size"]
    logits = torch.empty((T, V), dtype=torch.float32)
    for start, end, Wblk in store.row_blocks("lm_head"):
        logits[:, start:end] = _streamed_lm_head_matmul(h, Wblk)
        del Wblk
    return logits


# =================================================================== mamba (SSM) math
@torch.no_grad()
def paged_logits_mamba(
    store: QStore,
    input_ids: np.ndarray,
    collect_acts: list | None = None,
    patch_ops_by_layer: dict[int, list] | None = None,
    capture_selected_maps: dict[int, dict] | None = None,
    captured_selected: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Mamba paged forward (slow sequential selective-scan, no CUDA kernels). RMSNorm pre-mixer,
    in_proj→(x,z), depthwise causal conv1d+silu, x_proj→(dt,B,C), dt_proj+softplus, the
    discretized scan, D skip, silu(z) gate. The tap is the gated-scan activation feeding
    ``out_proj`` (= ``L{L}.down``, width = d_inner). lm_head is tied to the embedding."""
    c = store.cfg
    d, nL = c["hidden_size"], c["num_hidden_layers"]
    d_inner, N = c["intermediate_size"], c["state_size"]
    K, dt_rank, eps = c["conv_kernel"], c["time_step_rank"], c["layer_norm_epsilon"]
    T = len(input_ids)

    h = store.embed_rows("embed", input_ids).clone()  # [T, d]

    for L in range(nL):
        x = _rms_norm(h, store.fp32(f"L{L}.norm"), eps)  # [T, d]
        proj = store.matmul(f"L{L}.in_proj", x)  # [T, 2*d_inner]
        xc, z = proj[:, :d_inner], proj[:, d_inner:]  # each [T, d_inner]

        convW = store.fp32(f"L{L}.conv1d.weight")  # [d_inner, 1, K] depthwise
        convB = store.fp32(f"L{L}.conv1d.bias")  # [d_inner]
        xt = xc.transpose(0, 1).unsqueeze(0)  # [1, d_inner, T]
        conv_out = torch.nn.functional.conv1d(xt, convW, convB, padding=K - 1, groups=d_inner)[
            ..., :T
        ]
        u = torch.nn.functional.silu(conv_out)[0].transpose(0, 1)  # [T, d_inner]
        del xt, conv_out

        ssm = store.matmul(f"L{L}.x_proj", u)  # [T, dt_rank+2N]
        dt_in = ssm[:, :dt_rank]
        B = ssm[:, dt_rank : dt_rank + N]  # [T, N]
        Cc = ssm[:, dt_rank + N : dt_rank + 2 * N]  # [T, N]
        dt = torch.nn.functional.softplus(
            store.matmul(f"L{L}.dt_proj", dt_in) + store.fp32(f"L{L}.dt_proj.bias")
        )  # [T, d_inner]
        A = -torch.exp(store.fp32(f"L{L}.A_log"))  # [d_inner, N]
        D = store.fp32(f"L{L}.D")  # [d_inner]

        state = torch.zeros(d_inner, N, dtype=torch.float32)
        ys = torch.empty(T, d_inner, dtype=torch.float32)
        for t in range(T):  # selective scan recurrence
            dt_t = dt[t]  # [d_inner]
            dA = torch.exp(dt_t[:, None] * A)  # [d_inner, N]
            dBu = (dt_t[:, None] * B[t][None, :]) * u[t][:, None]  # [d_inner, N]
            state = dA * state + dBu
            ys[t] = state @ Cc[t]  # [d_inner]
        y = ys + u * D[None, :]  # D skip
        hid = y * torch.nn.functional.silu(z)  # gate — the tap activation [T, d_inner]
        hid = _tap(
            hid, L, patch_ops_by_layer, capture_selected_maps, captured_selected, collect_acts
        )
        h = h + store.matmul(f"L{L}.down", hid)
        del hid, ys, state

    h = _rms_norm(h, store.fp32("norm.final"), eps)
    V = c["vocab_size"]
    logits = torch.empty((T, V), dtype=torch.float32)
    for start, end, Wblk in store.row_blocks("lm_head"):
        logits[:, start:end] = _streamed_lm_head_matmul(h, Wblk)
        del Wblk
    return logits
