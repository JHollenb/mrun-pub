"""Bounded-memory safetensors reads and native streamed MoE execution.

OLMoE, Qwen2/3-MoE, Mixtral and DeepSeek layouts share the existing streamed
forward implementations. Routing and residual traces remain available to
callers; scientific physiology reports and probe batteries are separate.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from ..fp8 import (
    DEQUANT_ROW_BUDGET,
    declared_block_size,
    dequantize_block_fp8,
    dequantize_fp4_packed,
    dequantize_fp8_rows,
    is_fp4_packed,
    resolve_block_size,
    scale_key_for,
    scale_to_float32,
)

#: fp32 elements per on-device dequantization band. The device analogue of
#: :data:`mrun.fp8.DEQUANT_ROW_BUDGET`, deliberately larger: on the host the quantity being
#: bounded is resident bytes, on a GPU it is kernel-launch count against a ~700 GB/s
#: elementwise wall, so a 16 MB band would spend a fifth of its time on launch overhead.
_DEVICE_DEQUANT_ELEMS = max(DEQUANT_ROW_BUDGET, 1 << 24)


def _dequant_banded_on_device(
    codes: torch.Tensor, factor: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    """``(codes.float() * factor).to(dtype)`` without a full-size fp32 temporary.

    ``factor`` is a 0-d scalar or a ``[rows, 1]`` fp32 tensor. Bit-identical to the naive
    expression: every element is still widened exactly, multiplied once in IEEE single, and
    rounded once to ``dtype``. Banding changes no element's arithmetic.
    """
    if codes.ndim != 2:  # 1-D / scalar-scaled oddities: no banding axis to speak of
        return (codes.to(torch.float32) * factor).to(dtype)
    rows, cols = int(codes.shape[0]), int(codes.shape[1])
    out = torch.empty((rows, cols), dtype=dtype, device=codes.device)
    band = max(1, _DEVICE_DEQUANT_ELEMS // max(1, cols))
    per_row = factor.ndim > 0
    for start in range(0, rows, band):
        stop = min(start + band, rows)
        chunk = codes[start:stop].to(torch.float32)
        chunk.mul_(factor[start:stop] if per_row else factor)
        out[start:stop] = chunk.to(dtype)
    return out


def _dequant_blocked_on_device(
    codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    block_rows: int,
    block_cols: int,
) -> torch.Tensor:
    """Block-scaled FP8 dequant straight into ``dtype``, banded over rows of the block grid.

    The prior implementation widened the WHOLE tensor to fp32, multiplied that into a second
    full-size fp32 tensor, then narrowed — ~10 bytes/param of transient to produce a
    2 bytes/param bf16 weight. The ragged branch was worse: it additionally materialized a full
    ``[rows, cols]`` expanded scale (~14 bytes/param), which is precisely the blowup the aligned
    branch's grid-broadcast was written to avoid. Banding holds the fp32 working set to one band
    while producing bit-identical output.
    """
    rows, cols = int(codes.shape[0]), int(codes.shape[1])
    out = torch.empty((rows, cols), dtype=dtype, device=codes.device)
    if rows % block_rows == 0 and cols % block_cols == 0:
        col_blocks = cols // block_cols
        band = max(block_rows, (_DEVICE_DEQUANT_ELEMS // max(1, cols)) // block_rows * block_rows)
        for start in range(0, rows, band):
            stop = min(start + band, rows)
            n_blocks = (stop - start) // block_rows
            first = start // block_rows
            chunk = codes[start:stop].reshape(n_blocks, block_rows, col_blocks, block_cols)
            chunk = chunk.to(torch.float32)
            chunk.mul_(scale[first : first + n_blocks].reshape(n_blocks, 1, col_blocks, 1))
            out[start:stop] = chunk.reshape(stop - start, cols).to(dtype)
        return out
    # Ragged final block: expand the scale over COLUMNS once (bounded by the scale's own row
    # count, ~2 MB on a 12288x6144 weight) and index rows per band, rather than materializing a
    # full-size expanded scale.
    col_index = torch.arange(cols, device=codes.device) // block_cols
    row_index = torch.arange(rows, device=codes.device) // block_rows
    scale_cols = scale.index_select(1, col_index)
    band = max(1, _DEVICE_DEQUANT_ELEMS // max(1, cols))
    for start in range(0, rows, band):
        stop = min(start + band, rows)
        chunk = codes[start:stop].to(torch.float32)
        chunk.mul_(scale_cols.index_select(0, row_index[start:stop]))
        out[start:stop] = chunk.to(dtype)
    return out


def _rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        import resource
        m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return m / 1e9 if m > 1e7 else m / 1e6


# ------------------------------------------------------------------ paged store
class PagedSafetensors:
    """mmap-backed pull-per-weight reader. ``get`` returns the tensor in NATIVE dtype on cpu;
    the forward casts to its compute (device, dtype) per use and frees after — nothing model-
    sized ever goes resident.

    FP8 EXCEPTION (load-bearing, not a convenience): a block-scaled fp8 checkpoint stores
    ``<key>.weight`` as raw e4m3 CODES with the real magnitudes held in a separate
    ``<key>.weight_scale_inv``. "Native dtype" for those keys is meaningless on its own — the
    forward's ``.to(dtype)`` cast would turn code bytes into numbers off by the block scale,
    silently, at every weight. So ``get`` DEQUANTIZES any key that has a companion scale tensor
    (:func:`mrun.fp8.dequantize_block_fp8`, chunked so no full-size fp32 temporary appears) and
    returns unquantized keys byte-identically, exactly as before.

    MEASURED (beast, Mixtral-8x7B, 2026-07-17): on Linux the mmap'd shard pages a pass has
    touched stay counted in the process RSS (16GB by layer 4 of 32, heading for min(model,
    RAM)) — the kernel would reclaim them under pressure, but the fleet's RSS kill gate reads
    the number at face value and killed the job. ``release()`` closes the shard handles
    (unmapping the files, which drops those pages from RSS immediately) and lets ``get``
    reopen lazily; the forward calls it once per layer, so resident mapped pages stay bounded
    by roughly one layer's weights. macOS hid this (13GB OLMoE smoke sat at 2.8GB RSS)."""

    def __init__(self, model_dir: Path | str, *, dequant_dtype: torch.dtype = torch.float32):
        from safetensors import safe_open
        self._safe_open = safe_open
        model_dir = Path(model_dir)
        self._dir = model_dir
        idx = model_dir / "model.safetensors.index.json"
        if idx.exists():
            self.weight_map = json.loads(idx.read_text())["weight_map"]
        else:  # single-shard save (tiny/test models): no index file
            single = model_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no safetensors index or single shard in {model_dir}")
            with safe_open(str(single), framework="pt") as h:
                self.weight_map = {k: single.name for k in h.keys()}
        self.handles: dict = {}
        # `handles` is mutated by `_handle` (any reader) and cleared by `release` (typically a
        # per-block forward hook). With a background prefetch thread those are different threads,
        # and an unguarded dict lets `release` unmap a shard while a worker is mid-`get_tensor`.
        # That race is why per-block release and prefetch were previously mutually exclusive.
        self._handles_lock = threading.Lock()
        self.cfg = json.loads((model_dir / "config.json").read_text())
        self.dequant_dtype = dequant_dtype
        # The checkpoint's own declared fp8 block geometry. Shapes alone cannot always recover
        # it, and guessing wrong is silent — read it rather than infer where possible.
        self.block_size = declared_block_size(self.cfg)

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def scale_key(self, key: str) -> str | None:
        """Name of ``key``'s companion fp8 scale tensor, or ``None`` when stored unquantized."""
        return scale_key_for(self.weight_map, key)

    def _handle(self, key: str):
        shard = self.weight_map[key]
        with self._handles_lock:
            h = self.handles.get(shard)
            if h is None:
                h = self.handles[shard] = self._safe_open(str(self._dir / shard), framework="pt")
            return h

    def raw(self, key: str) -> torch.Tensor:
        """The stored tensor exactly as written (fp8 codes stay codes). Prefer :meth:`get`."""
        return self._handle(key).get_tensor(key)

    def get_rows(self, key: str, row_ids) -> torch.Tensor:
        """``[len(row_ids), cols]`` — RANGE-READ only those rows of a 2-D weight.

        The point of this method is what it does NOT read. Scoring a handful of candidate tokens
        needs a handful of ``lm_head`` rows; ``get`` would pull the whole 151936x6144 matrix
        (1.9 GB) to use ~10 of its rows. ``safe_open(...).get_slice(key)[ids]`` range-reads
        exactly the requested rows — verified against a full-tensor gather for scattered,
        unsorted and duplicated ids, with order preserved.

        FP8 is handled by resolving each GATHERED row's own scale block: after a scattered
        gather a row's position in the full tensor is gone, so reusing the whole-tensor
        dequantizer here would silently apply the wrong block's scale.
        """
        ids = [int(value) for value in row_ids]
        rows = self._handle(key).get_slice(key)[ids] if ids else None
        if rows is None:
            shape = self._handle(key).get_slice(key).get_shape()
            return torch.empty((0, int(shape[1])), dtype=self.dequant_dtype)
        scale_key = self.scale_key(key)
        if scale_key is None:
            return rows
        full_shape = self._handle(key).get_slice(key).get_shape()
        scale = self.raw(scale_key)
        if is_fp4_packed(rows, scale):
            # packed fp4: scale is per OUTPUT ROW, so gather the matching scale rows
            index = torch.as_tensor(ids, dtype=torch.long)
            return dequantize_fp4_packed(
                rows, scale.index_select(0, index), dtype=self.dequant_dtype
            )
        if scale.ndim == 0:
            return (rows.to(torch.float32) * scale_to_float32(scale)).to(self.dequant_dtype)
        if scale.ndim == 1:                                   # per-output-row scale
            index = torch.as_tensor(ids, dtype=torch.long)
            return (
                rows.to(torch.float32) * scale_to_float32(scale).index_select(0, index)[:, None]
            ).to(self.dequant_dtype)
        block_rows, block_cols = resolve_block_size(
            int(full_shape[0]), int(full_shape[1]), scale.shape, self.block_size
        )
        scale_index = torch.as_tensor([i // block_rows for i in ids], dtype=torch.long)
        return dequantize_fp8_rows(
            rows,
            scale_to_float32(scale).index_select(0, scale_index),
            block_cols=block_cols,
            dtype=self.dequant_dtype,
        )

    def get(self, key: str) -> torch.Tensor:
        scale_key = self.scale_key(key)
        if scale_key is None:
            return self.raw(key)
        return dequantize_block_fp8(
            self.raw(key),
            self.raw(scale_key),
            dtype=self.dequant_dtype,
            block_size=self.block_size,
        )

    def get_on(self, key: str, device: str, dtype: torch.dtype) -> torch.Tensor:
        """``get(key).to(device, dtype)`` — but for an FP8 weight, dequantize ON ``device``.

        MEASURED (Qwen3-Coder-480B, beast RTX 4080, 2026-07-25): dequantizing on the CPU and
        shipping the result made the GPU sit at 3% utilization while ~9 cores ground through
        ~6 G parameter-dequants per layer — 11 s wall against 105 s of CPU per layer. The GEMMs
        were already on the GPU; the DEQUANT was the bottleneck, and it was on the wrong device.

        Doing it here instead sends the raw e4m3 CODES over PCIe (1 byte/param instead of 2 for
        bf16), expands the block scale on the device, and leaves the CPU with just the mmap
        read. Numerically identical to :meth:`get` followed by ``.to()`` up to the order of the
        fp32 multiply and the final cast.

        The device-side expansion is BANDED (:func:`_dequant_blocked_on_device`). It used to
        widen the whole tensor to fp32 and multiply that into a second full-size fp32 tensor —
        ~10 bytes/param of transient to produce a 2 bytes/param bf16 weight, and ~14 for the
        ragged branch. That cost is why FP8 paging measured only 1.3x on the mstack FLUX DiT
        rather than the 2.0x its halved byte count predicts: `io_ms` there is computed as
        `paged - resident`, so ~110 ms/step of this dequant was being reported as transfer.

        For a CPU device or an unquantized weight this is exactly the old path.
        """
        scale_key = self.scale_key(key)
        if scale_key is None:
            return self.raw(key).to(device=device, dtype=dtype)
        if str(device) == "cpu":
            return self.get(key).to(dtype)
        raw_codes = self.raw(key)
        raw_scale = self.raw(scale_key)
        if is_fp4_packed(raw_codes, raw_scale):
            # ship 0.5 byte/param codes + 1 byte/group scales; unpack + scale ON the device
            return dequantize_fp4_packed(
                raw_codes.to(device=device, non_blocking=True),
                raw_scale.to(device=device, non_blocking=True),
                dtype=dtype,
            )
        codes = raw_codes.to(device=device, non_blocking=True)
        scale = scale_to_float32(raw_scale).to(device=device, non_blocking=True)
        del raw_codes, raw_scale
        if scale.ndim == 0:
            return _dequant_banded_on_device(codes, scale, dtype=dtype)
        rows, cols = int(codes.shape[0]), int(codes.shape[1])
        if scale.ndim == 1:
            return _dequant_banded_on_device(codes, scale[:, None], dtype=dtype)
        block_rows, block_cols = resolve_block_size(rows, cols, scale.shape, self.block_size)
        return _dequant_blocked_on_device(
            codes, scale, dtype=dtype, block_rows=block_rows, block_cols=block_cols
        )

    def release(self) -> None:
        """Unmap all shards (drops their file-backed pages from RSS); reopened lazily.

        A handle already returned to a caller stays alive through its own reference, so a
        concurrent reader mid-``get_tensor`` is unaffected; the lock only keeps the dict itself
        consistent.
        """
        with self._handles_lock:
            self.handles.clear()


def resolve_model_dir(model: str) -> Path:
    """Direct dir (possibly a snapshot root) or an HF-cache id (offline)."""
    p = Path(os.path.expanduser(model))
    if p.is_dir():
        if not (p / "config.json").exists():
            cands = sorted(p.glob("**/config.json"))
            if not cands:
                raise FileNotFoundError(f"no config.json under {p}")
            p = cands[0].parent
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, local_files_only=True,
                allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model",
                                "*.txt", "merges*", "vocab*"]))


# ------------------------------------------------------------------ layout
class MoeLayout(NamedTuple):
    name: str        # olmoe | qwen2_moe | qwen3_moe | mixtral | deepseek_v4
    router: str      # sub-key of the router Linear under model.layers.{i}.
    gate: str        # expert gate-proj sub-key pattern, .format(e=expert)
    up: str
    down: str
    norm_topk: bool  # renormalize the kept top-k weights
    qk_norm: str     # none | full | per_head
    embed_key: str = "model.embed_tokens.weight"
    head_key: str = "lm_head.weight"


def detect_layout(ps: PagedSafetensors) -> MoeLayout:
    """Structural detection over ALL keys — layer 0 may be dense (qwen3_moe mlp_only_layers),
    so probing a fixed layer is not enough."""
    cfg = ps.cfg
    keys = ps.weight_map
    if ps.has("layers.0.attn.wq_a.weight") and ps.has("layers.0.ffn.experts.0.w1.weight"):
        # DeepSeek-V4 NATIVE release namespace (no `model.` prefix; V3-style module names).
        # The forward lives in mri.deepseek_v4_stream; router/gate patterns here serve key
        # probes and the physiology accounting (w1=gate, w3=up, w2=down, mixtral-style).
        return MoeLayout("deepseek_v4", "ffn.gate",
                         "ffn.experts.{e}.w1", "ffn.experts.{e}.w3", "ffn.experts.{e}.w2",
                         bool(cfg.get("norm_topk_prob", True)), "none",
                         embed_key="embed.weight", head_key="head.weight")
    if any(k.endswith(".block_sparse_moe.gate.weight") for k in keys):
        return MoeLayout("mixtral", "block_sparse_moe.gate",
                         "block_sparse_moe.experts.{e}.w1", "block_sparse_moe.experts.{e}.w3",
                         "block_sparse_moe.experts.{e}.w2", True, "none")
    if (any(".mlp.shared_expert." in k for k in keys)
            and any(".mlp.shared_expert_gate." in k for k in keys)):
        return MoeLayout(
            "qwen2_moe",
            "mlp.gate",
            "mlp.experts.{e}.gate_proj",
            "mlp.experts.{e}.up_proj",
            "mlp.experts.{e}.down_proj",
            bool(cfg.get("norm_topk_prob", True)),
            "none",
        )
    if (any(k.endswith(".mlp.gate.weight") for k in keys)
            and any(".mlp.experts.0.gate_proj." in k for k in keys)):
        H = cfg["num_attention_heads"]
        Dh = cfg.get("head_dim") or (cfg["hidden_size"] // H)
        qk = "none"
        if ps.has("model.layers.0.self_attn.q_norm.weight"):
            qk = "per_head" if ps.get("model.layers.0.self_attn.q_norm.weight").numel() == Dh else "full"
        name = "qwen3_moe" if qk == "per_head" else "olmoe"
        return MoeLayout(name, "mlp.gate", "mlp.experts.{e}.gate_proj",
                         "mlp.experts.{e}.up_proj", "mlp.experts.{e}.down_proj",
                         bool(cfg.get("norm_topk_prob", False)), qk)
    raise NotImplementedError(
        f"unrecognized MoE tensor layout (model_type={cfg.get('model_type')!r}); "
        "supported: olmoe/qwen2_moe/qwen3_moe (mlp.experts.N.gate_proj) and mixtral "
        "(block_sparse_moe.experts.N.w1).")


# ------------------------------------------------------------------ math (HF-exact)
def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """HF RMSNorm: fp32 math, cast back to input dtype BEFORE the weight multiply."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _rope_tables(T: int, dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    freqs = torch.outer(torch.arange(T, dtype=torch.float32, device=device), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(x, cos, sin):
    # x: [T, n, hd]; cos/sin: [T, hd]
    return x * cos[:, None, :] + _rotate_half(x) * sin[:, None, :]


# ------------------------------------------------------------------ streamed forward
@torch.no_grad()
def streamed_moe_forward(ps: PagedSafetensors, ids: torch.Tensor, *,
                         device: str = "cpu", dtype: torch.dtype = torch.float32,
                         capture_expert_stats: bool = False, return_logits: bool = False,
                         capture_hidden_states: bool = False,
                         capture_final_hidden: bool = False,
                         resid_patch_ops_by_layer: dict[int, list] | None = None,
                         ablate_component: tuple[int, str] | None = None,
                         ablate_embed_direction: torch.Tensor | None = None,
                         ablate_embed_alpha: float = 1.0,
                         zero_embedding: bool = False,
                         skip_lm_head: bool = False,
                         abort_rss_gb: float = 48.0, log=None) -> dict:
    """One prefill pass, pulling weights on demand. Captures the per-layer top-k routing
    trace; optionally per-(layer,expert,neuron) firing stats + down-write norms (for the
    crystallization face). Returns CE over the sequence (fp32 log-softmax).

    ANALYSIS TAPS (all default-off; a call that passes none is byte-identical to before):

    ``ablate_component=(layer, "attn"|"mlp")`` removes that component's WHOLE contribution to
    the residual at every position — the streamed equivalent of a forward hook returning
    ``zeros_like(out)``. Both are exact, not approximate: a decoder block adds attention and
    MLP into the residual and nothing else in the layer consumes them, so dropping the term is
    identical to zeroing it. The ablated component's weights are then never pulled at all,
    which on an MoE layer means the top-k experts are not streamed off disk — a removed
    component is CHEAPER than a live one, not more expensive.

    ``ablate_embed_direction`` projects a unit direction out of the embedding rows before layer
    0 (``h -> h - alpha (h.v) v``), the same edit an in-place ``embedding.weight -= (W v) vᵀ``
    makes without mutating anything. Accepts ``[d]`` or ``[d, k]``; a matrix removes the whole
    subspace (``h -> h - alpha (h V) Vᵀ``, columns assumed orthonormal). ``zero_embedding``
    destroys token identity outright — the must-break arm of an anti-vacuity gate.

    ``skip_lm_head`` returns no logits and no CE. On a 151936-row vocab the head is a 1.9 GB
    pull off disk and a ``[T, 151936]`` fp32 materialization per pass; a residual-stream
    measurement reads neither, and a 125-pass coupling sweep would otherwise stream a quarter
    of a terabyte of lm_head for nothing. ``capture_final_hidden`` returns just the
    post-final-norm state, which is what such a measurement actually wants.
    """
    lay = detect_layout(ps)
    if lay.name == "deepseek_v4":
        from .deepseek_v4_stream import dsv4_forward
        return dsv4_forward(
            ps, ids, device=device, dtype=dtype,
            capture_expert_stats=capture_expert_stats, return_logits=return_logits,
            capture_hidden_states=capture_hidden_states,
            capture_final_hidden=capture_final_hidden,
            resid_patch_ops_by_layer=resid_patch_ops_by_layer,
            ablate_component=ablate_component,
            ablate_embed_direction=ablate_embed_direction,
            ablate_embed_alpha=ablate_embed_alpha,
            zero_embedding=zero_embedding, skip_lm_head=skip_lm_head,
            abort_rss_gb=abort_rss_gb, log=log)
    cfg = ps.cfg
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    Hkv = cfg.get("num_key_value_heads", H)
    Dh = cfg.get("head_dim") or (cfg["hidden_size"] // H)
    n_exp = cfg.get("num_experts") or cfg.get("num_local_experts") or cfg["n_routed_experts"]
    k = cfg["num_experts_per_tok"]
    eps = cfg.get("rms_norm_eps") or 1e-5
    # rope_theta moved under rope_parameters in transformers 5.x saves; older checkpoints
    # keep it top-level. A silent fallback here cost a 1e-4 parity drift — read both, and
    # refuse scaled rope variants (yarn/linear) rather than quietly ignoring them.
    rp = cfg.get("rope_parameters") or {}
    theta = rp.get("rope_theta") or cfg.get("rope_theta") or 1e4
    rope_type = rp.get("rope_type", "default")
    scaling = cfg.get("rope_scaling")
    if rope_type != "default" or (isinstance(scaling, dict) and scaling.get("type", scaling.get("rope_type", "default")) != "default"):
        raise NotImplementedError(f"non-default rope (type={rope_type!r}, scaling={scaling!r}) not supported")
    act_name = cfg.get("hidden_act", "silu")
    if act_name != "silu":
        raise NotImplementedError(f"hidden_act={act_name!r}; only silu SwiGLU experts supported")
    T = int(ids.shape[0])
    sw = cfg.get("sliding_window")
    if sw and T > int(sw):
        raise NotImplementedError(
            f"sliding_window={sw} < T={T}: this forward uses a full causal mask; "
            "truncate the input or add windowed attention before trusting the trace")
    ids = ids.to(device)

    def w(key: str) -> torch.Tensor:
        # get_on dequantizes fp8 ON the compute device (see PagedSafetensors.get_on); for cpu
        # or unquantized weights it is the original `get(key).to(device, dtype)`.
        return ps.get_on(key, device, dtype)

    def lk(i: int, s: str) -> str:
        return f"model.layers.{i}.{s}"

    has_attn_bias = ps.has(lk(0, "self_attn.q_proj.bias"))
    if log:
        log(f"  layout={lay.name} L={L} H={H} Hkv={Hkv} Dh={Dh} n_exp={n_exp} k={k} "
            f"theta={theta} qk_norm={lay.qk_norm} norm_topk={lay.norm_topk} "
            f"device={device} dtype={dtype}")

    ablate_layer, ablate_kind = -1, ""
    if ablate_component is not None:
        ablate_layer, ablate_kind = int(ablate_component[0]), str(ablate_component[1])
        if ablate_kind not in ("attn", "mlp"):
            raise ValueError(f"ablate_component kind must be 'attn' or 'mlp', got {ablate_kind!r}")
        if not 0 <= ablate_layer < L:
            raise ValueError(f"ablate_component layer {ablate_layer} out of range 0..{L - 1}")

    emb = w("model.embed_tokens.weight")
    h = emb[ids]
    if zero_embedding:
        h = torch.zeros_like(h)
    elif ablate_embed_direction is not None:
        basis = torch.as_tensor(ablate_embed_direction, dtype=torch.float32, device=device)
        if basis.ndim == 1:
            basis = basis / (basis.norm() + 1e-9)
            basis = basis[:, None]
        hf = h.float()
        h = (hf - ablate_embed_alpha * (hf @ basis) @ basis.T).to(h.dtype)
        del hf, basis
    hidden_states = [h.detach().cpu()] if capture_hidden_states else []
    tied_head = not ps.has("lm_head.weight")
    if not tied_head:
        emb = None  # free now; the separate lm_head is pulled at the end
    cos, sin = _rope_tables(T, Dh, theta, device, dtype)
    causal = torch.full((T, T), float("-inf"), device=device, dtype=dtype).triu(1)
    topk_idx = np.zeros((L, T, k), dtype=np.int16)
    topk_w = np.zeros((L, T, k), dtype=np.float32)
    act_sum: dict = {}
    act_cnt: dict = {}
    write_norm: dict = {}

    t0 = time.time()
    for i in range(L):
        # An ablated component contributes exactly zero to the residual, so the whole block is
        # skipped rather than computed and multiplied by nothing — on an MoE layer that also
        # means its top-k experts are never streamed off disk.
        if not (i == ablate_layer and ablate_kind == "attn"):
            x = _rmsnorm(h, w(lk(i, "input_layernorm.weight")), eps)

            def proj(name, x=x, i=i):
                o = x @ w(lk(i, f"self_attn.{name}.weight")).T
                if has_attn_bias and ps.has(lk(i, f"self_attn.{name}.bias")):
                    o = o + w(lk(i, f"self_attn.{name}.bias"))
                return o
            q = proj("q_proj"); kk = proj("k_proj"); v = proj("v_proj")
            if lay.qk_norm == "full":                   # olmoe: over the flat q/k vector
                q = _rmsnorm(q, w(lk(i, "self_attn.q_norm.weight")), eps)
                kk = _rmsnorm(kk, w(lk(i, "self_attn.k_norm.weight")), eps)
            q = q.view(T, H, Dh)
            kk = kk.view(T, Hkv, Dh)
            v = v.view(T, Hkv, Dh)
            if lay.qk_norm == "per_head":               # qwen3: per head over Dh, BEFORE rope
                q = _rmsnorm(q, w(lk(i, "self_attn.q_norm.weight")), eps)
                kk = _rmsnorm(kk, w(lk(i, "self_attn.k_norm.weight")), eps)
            q = _apply_rope(q, cos, sin)
            kk = _apply_rope(kk, cos, sin)
            q, kk, v = q.transpose(0, 1), kk.transpose(0, 1), v.transpose(0, 1)   # [H,T,Dh]
            if Hkv != H:
                rep = H // Hkv
                kk = kk.repeat_interleave(rep, dim=0)
                v = v.repeat_interleave(rep, dim=0)
            scores = (q @ kk.transpose(-1, -2)) / (Dh ** 0.5) + causal
            probs = torch.softmax(scores.float(), dim=-1).to(dtype)           # HF eager: fp32
            ctx = (probs @ v).transpose(0, 1).reshape(T, H * Dh)
            o = ctx @ w(lk(i, "self_attn.o_proj.weight")).T
            if has_attn_bias and ps.has(lk(i, "self_attn.o_proj.bias")):
                o = o + w(lk(i, "self_attn.o_proj.bias"))
            h = h + o
            del scores, probs, ctx, q, kk, v, o, x

        ablate_mlp_here = i == ablate_layer and ablate_kind == "mlp"
        y = _rmsnorm(h, w(lk(i, "post_attention_layernorm.weight")), eps)
        if not ps.has(lk(i, f"{lay.router}.weight")):
            # dense layer (qwen3_moe mlp_only_layers / decoder_sparse_step): plain SwiGLU MLP
            if not ablate_mlp_here:
                g = w(lk(i, "mlp.gate_proj.weight")); u = w(lk(i, "mlp.up_proj.weight"))
                d = w(lk(i, "mlp.down_proj.weight"))
                h = h + (torch.nn.functional.silu(y @ g.T) * (y @ u.T)) @ d.T
                del g, u, d
            h = _apply_resid_patch_ops(h, (resid_patch_ops_by_layer or {}).get(i, []))
            if capture_hidden_states:
                hidden_states.append(h.detach().cpu())
            del y
            topk_idx[i] = -1  # sentinel: no routing at this layer
            ps.release()
            continue
        logits = y @ w(lk(i, f"{lay.router}.weight")).T
        rprobs = torch.softmax(logits.float(), dim=-1)                        # HF: fp32 router
        topw, topi = rprobs.topk(k, dim=-1)
        if lay.norm_topk:
            topw = topw / topw.sum(-1, keepdim=True)
        # Routing-weight dtype is LAYOUT-SPECIFIC in HF (5.13.1): Olmoe/Qwen3Moe routers cast
        # the kept weights to the model dtype BEFORE the expert multiply; Mixtral's router
        # returns fp32 and the experts multiply in fp32 with one cast at the add. Keeping
        # fp32 everywhere diverges from a real bf16 olmoe/qwen3 run at every MoE layer.
        if lay.name != "mixtral":
            topw = topw.to(dtype)
        topk_idx[i] = topi.cpu().numpy().astype(np.int16)
        topk_w[i] = topw.float().cpu().numpy().astype(np.float32)  # applied (rounded) weights

        moe = torch.zeros_like(h)
        # Routing is still traced under an mlp ablation (one small GEMM, already paid above) so
        # the physiology outputs stay well-defined; only the expert pulls are skipped.
        routed = [] if ablate_mlp_here else torch.unique(topi).tolist()
        for e in routed:
            sel = (topi == e)
            tok = sel.any(-1)
            w_e = (topw * sel.to(topw.dtype)).sum(-1)[tok]    # fp32 (mixtral) / model dtype (rest)
            xe = y[tok]
            g = w(lk(i, lay.gate.format(e=e) + ".weight"))
            u = w(lk(i, lay.up.format(e=e) + ".weight"))
            d = w(lk(i, lay.down.format(e=e) + ".weight"))
            # HF 5.x fuses gate/up into ONE [2I,h] GEMM at load (gate_up_proj); doing two
            # separate GEMMs hits different BLAS blocking and drifts ~1e-4 off the dense
            # reference. Replicate the fused GEMM for bit-tight parity.
            gu = xe @ torch.cat([g, u], dim=0).T
            gg, uu = gu.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gg) * uu
            ye = act @ d.T
            if lay.name == "mixtral":   # HF: fp32 weight x expert-out, single cast at the add
                moe[tok] += (w_e[:, None] * ye.float()).to(dtype)
            else:                       # HF: weight already model dtype, multiply in dtype
                moe[tok] += w_e[:, None] * ye
            if capture_expert_stats:
                act_sum[(i, e)] = act.float().abs().sum(dim=0).cpu().double().numpy()
                act_cnt[(i, e)] = int(tok.sum().item())
                write_norm[(i, e)] = torch.linalg.norm(d.float(), dim=0).cpu().numpy()
            del g, u, d, act, xe
        if lay.name == "qwen2_moe" and not ablate_mlp_here:
            shared_gate = w(lk(i, "mlp.shared_expert.gate_proj.weight"))
            shared_up = w(lk(i, "mlp.shared_expert.up_proj.weight"))
            shared_down = w(lk(i, "mlp.shared_expert.down_proj.weight"))
            shared_gu = y @ torch.cat([shared_gate, shared_up], dim=0).T
            shared_g, shared_u = shared_gu.chunk(2, dim=-1)
            shared = (torch.nn.functional.silu(shared_g) * shared_u) @ shared_down.T
            shared_weight = torch.sigmoid(
                y @ w(lk(i, "mlp.shared_expert_gate.weight")).T
            )
            moe = moe + shared_weight * shared
            del shared_gate, shared_up, shared_down, shared_gu, shared_g, shared_u, shared
        if capture_expert_stats:                       # dead experts: NaN, weights never pulled
            inter = next(iter(act_sum.values())).shape[0] if act_sum else 0
            for e in range(n_exp):
                if (i, e) not in act_sum:
                    act_sum[(i, e)] = np.full(inter, np.nan)
                    act_cnt[(i, e)] = 0
                    write_norm[(i, e)] = np.full(inter, np.nan)
        h = h + moe
        h = _apply_resid_patch_ops(h, (resid_patch_ops_by_layer or {}).get(i, []))
        if capture_hidden_states:
            hidden_states.append(h.detach().cpu())
        del moe, y, logits, rprobs
        ps.release()  # unmap touched shards: RSS stays O(one layer), see PagedSafetensors

        r = _rss_gb()
        if log and (i % 4 == 0 or i == L - 1):
            vr = (torch.cuda.memory_allocated() / 1e9) if device.startswith("cuda") else 0.0
            log(f"    layer {i:2d}/{L}  RSS={r:.2f}GB VRAM={vr:.2f}GB  {time.time()-t0:.0f}s")
        if r > abort_rss_gb:
            raise MemoryError(f"RSS {r:.1f}GB > {abort_rss_gb}GB abort")

    hn = _rmsnorm(h, w("model.norm.weight"), eps)
    if capture_hidden_states:
        # Match HF output_hidden_states: embedding + one entry per layer, with the final
        # entry replaced by the post-final-norm state.
        hidden_states[-1] = hn.detach().cpu()
    ce = float("nan")
    out = None
    if not skip_lm_head:
        head = w("lm_head.weight") if not tied_head else emb
        out = hn @ head.T
        del head, emb
        logp = torch.log_softmax(out.float(), dim=-1)
        if T > 1:
            ce = -logp[torch.arange(T - 1, device=device), ids[1:]].mean().item()
    res = dict(topk_idx=topk_idx, topk_w=topk_w, n_exp=int(n_exp), k=int(k), L=int(L),
               hidden=int(cfg["hidden_size"]), ce=ce, layout=lay.name,
               wall_s=round(time.time() - t0, 1), rss_gb=round(_rss_gb(), 2))
    if capture_expert_stats:
        res.update(act_sum=act_sum, act_cnt=act_cnt, write_norm=write_norm)
    if return_logits and out is not None:
        res["logits"] = out.float().cpu()
    if capture_hidden_states:
        res["hidden_states"] = hidden_states
    if capture_final_hidden:
        res["final_hidden"] = hn.detach().float().cpu()
    return res


def _apply_resid_patch_ops(h: torch.Tensor, ops: list | tuple) -> torch.Tensor:
    """Apply residual-stream interventions after a complete decoder block."""
    for op, vector, _values in ops:
        if op != "proj_remove":
            raise ValueError(f"unknown residual patch op {op!r}")
        direction = torch.as_tensor(vector, dtype=torch.float32, device=h.device)
        direction = direction / (direction.norm() + 1e-9)
        projection = (h.float() @ direction).unsqueeze(-1) * direction
        h = (h.float() - projection).to(h.dtype)
    return h


@torch.no_grad()
def streamed_moe_forward_batch(ps: PagedSafetensors, rows: list[torch.Tensor],
                               arms: list[tuple[int, tuple[int, str] | None]], *,
                               device: str = "cpu", dtype: torch.dtype = torch.float32,
                               ablate_embed_direction: torch.Tensor | None = None,
                               ablate_embed_alpha: float = 1.0,
                               zero_embedding: bool = False,
                               capture_routing: bool = False,
                               abort_rss_gb: float = 48.0, log=None) -> dict:
    """One weight-streaming pass serving many (sequence, component-ablation) arms.

    WHY THIS EXISTS (measured, Qwen3-Coder-480B on beast, 2026-07-25): the serial forward
    re-streams the model per call — 152 GB per forward at 1.44-2.2 GB/s of disk while the GPU
    idles at 6%. Pulling each layer's weights ONCE and looping arms in compute served 11
    ablation arms in 175.8 s against 1162.7 s serial (6.61x), bit-exact. The win exists only
    when the model outsizes the page cache: a 13 GB OLMoE on a 61 GB host is warm after its
    first arm and batching buys ~nothing (1.0x wall, measured the same day). Full PoC:
    experiments/2026-07-25-moe-stream-max-utilization.

    NO CROSS-SEQUENCE ATTENTION: every arm keeps its own ``[T, d]`` hidden state and runs its
    own causal attention — rows are never concatenated, so the prompt-leak the serial engine's
    docstring refuses cannot occur here. The ONLY thing arms share is the weight pull (and the
    embedding-tap arguments, which apply to every arm — component ablation is the per-arm axis).

    BIT-EXACT BY CONSTRUCTION, verified not assumed: the serial kernel accumulates a sequence's
    routed experts in ``torch.unique`` (sorted) order; this kernel walks the sorted UNION of
    experts across arms and applies each to the arms that routed to it, so every arm still adds
    its own experts in the same order, through the same fused cat-GEMM, with the same
    layout-specific dtype casts. Parity gates assert equality, not tolerance.

    ``arms`` entries are ``(row_index, None | (layer, "attn"|"mlp"))``. Returns
    ``final_last [n_arms, hidden]`` fp32 cpu — the post-final-norm state at each arm's last
    token, i.e. ``final_hidden(row)[-1]`` of the serial kernel per arm. ``capture_routing``
    adds ``routing``: one int16 ``[L, T, k]`` per arm (-1 = dense layer), same convention as
    the serial ``topk_idx``.
    """
    lay = detect_layout(ps)
    if lay.name == "deepseek_v4":
        from .deepseek_v4_stream import dsv4_forward_arms
        return dsv4_forward_arms(
            ps, list(rows), list(arms), device=device, dtype=dtype,
            ablate_embed_direction=ablate_embed_direction,
            ablate_embed_alpha=ablate_embed_alpha,
            zero_embedding=zero_embedding, capture_routing=capture_routing,
            abort_rss_gb=abort_rss_gb, log=log)
    cfg = ps.cfg
    L = int(cfg["num_hidden_layers"])
    H = int(cfg["num_attention_heads"])
    Hkv = int(cfg.get("num_key_value_heads", H))
    Dh = int(cfg.get("head_dim") or (cfg["hidden_size"] // H))
    k = int(cfg["num_experts_per_tok"])
    eps = cfg.get("rms_norm_eps") or 1e-5
    rp = cfg.get("rope_parameters") or {}
    theta = rp.get("rope_theta") or cfg.get("rope_theta") or 1e4
    rope_type = rp.get("rope_type", "default")
    scaling = cfg.get("rope_scaling")
    if rope_type != "default" or (isinstance(scaling, dict) and scaling.get("type", scaling.get("rope_type", "default")) != "default"):
        raise NotImplementedError(f"non-default rope (type={rope_type!r}, scaling={scaling!r}) not supported")
    if cfg.get("hidden_act", "silu") != "silu":
        raise NotImplementedError(f"hidden_act={cfg.get('hidden_act')!r}; only silu supported")
    sw = cfg.get("sliding_window")
    for a_row, ablate in arms:
        if not 0 <= int(a_row) < len(rows):
            raise ValueError(f"arm row {a_row} out of range for {len(rows)} rows")
        if ablate is not None:
            li, kind = int(ablate[0]), str(ablate[1])
            if kind not in ("attn", "mlp") or not 0 <= li < L:
                raise ValueError(f"bad arm ablation {ablate!r}")

    def w(key: str) -> torch.Tensor:
        return ps.get_on(key, device, dtype)

    def lk(i: int, s: str) -> str:
        return f"model.layers.{i}.{s}"

    has_attn_bias = ps.has(lk(0, "self_attn.q_proj.bias"))
    t0 = time.time()
    rows_t = []
    for r in rows:
        row = torch.as_tensor(np.asarray(r, dtype=np.int64), dtype=torch.long)
        if row.ndim != 1 or not row.numel():
            raise ValueError("each row must be a non-empty one-dimensional token array")
        if sw and int(row.shape[0]) > int(sw):
            raise NotImplementedError(
                f"sliding_window={sw} < T={int(row.shape[0])}: full causal mask only")
        rows_t.append(row.to(device))
    lengths = sorted({int(r.shape[0]) for r in rows_t})
    tables = {T: _rope_tables(T, Dh, theta, device, dtype) for T in lengths}
    masks = {T: torch.full((T, T), float("-inf"), device=device, dtype=dtype).triu(1)
             for T in lengths}

    emb = w("model.embed_tokens.weight")
    hs = []
    for a_row, _ablate in arms:
        h = emb[rows_t[a_row]].clone()
        if zero_embedding:
            h = torch.zeros_like(h)
        elif ablate_embed_direction is not None:
            basis = torch.as_tensor(ablate_embed_direction, dtype=torch.float32, device=device)
            if basis.ndim == 1:
                basis = (basis / (basis.norm() + 1e-9))[:, None]
            hf = h.float()
            h = (hf - ablate_embed_alpha * (hf @ basis) @ basis.T).to(h.dtype)
            del hf, basis
        hs.append(h)
    del emb
    ps.release()
    A = len(arms)
    routing = None
    if capture_routing:
        routing = [np.full((L, int(hs[j].shape[0]), k), -1, dtype=np.int16) for j in range(A)]

    for i in range(L):
        attn_arms = [j for j in range(A) if arms[j][1] != (i, "attn")]
        if attn_arms:
            ln_w = w(lk(i, "input_layernorm.weight"))
            names = ("q_proj", "k_proj", "v_proj", "o_proj")
            pw = {n: w(lk(i, f"self_attn.{n}.weight")) for n in names}
            pb = {n: w(lk(i, f"self_attn.{n}.bias")) for n in names
                  if has_attn_bias and ps.has(lk(i, f"self_attn.{n}.bias"))}
            qn = w(lk(i, "self_attn.q_norm.weight")) if lay.qk_norm != "none" else None
            kn = w(lk(i, "self_attn.k_norm.weight")) if lay.qk_norm != "none" else None
            for j in attn_arms:
                h = hs[j]
                T = int(h.shape[0])
                cos, sin = tables[T]
                x = _rmsnorm(h, ln_w, eps)

                def proj(name, x=x):
                    o = x @ pw[name].T
                    if name in pb:
                        o = o + pb[name]
                    return o
                q = proj("q_proj"); kk = proj("k_proj"); v = proj("v_proj")
                if lay.qk_norm == "full":
                    q = _rmsnorm(q, qn, eps)
                    kk = _rmsnorm(kk, kn, eps)
                q = q.view(T, H, Dh)
                kk = kk.view(T, Hkv, Dh)
                v = v.view(T, Hkv, Dh)
                if lay.qk_norm == "per_head":
                    q = _rmsnorm(q, qn, eps)
                    kk = _rmsnorm(kk, kn, eps)
                q = _apply_rope(q, cos, sin)
                kk = _apply_rope(kk, cos, sin)
                q, kk, v = q.transpose(0, 1), kk.transpose(0, 1), v.transpose(0, 1)
                if Hkv != H:
                    rep = H // Hkv
                    kk = kk.repeat_interleave(rep, dim=0)
                    v = v.repeat_interleave(rep, dim=0)
                scores = (q @ kk.transpose(-1, -2)) / (Dh ** 0.5) + masks[T]
                probs = torch.softmax(scores.float(), dim=-1).to(dtype)
                ctx = (probs @ v).transpose(0, 1).reshape(T, H * Dh)
                o = ctx @ pw["o_proj"].T
                if "o_proj" in pb:
                    o = o + pb["o_proj"]
                hs[j] = h + o
                del scores, probs, ctx, q, kk, v, o, x
            del pw, pb, ln_w, qn, kn

        post_ln = w(lk(i, "post_attention_layernorm.weight"))
        mlp_live = [j for j in range(A) if arms[j][1] != (i, "mlp")]
        if not ps.has(lk(i, f"{lay.router}.weight")):
            if mlp_live:
                g = w(lk(i, "mlp.gate_proj.weight")); u = w(lk(i, "mlp.up_proj.weight"))
                d = w(lk(i, "mlp.down_proj.weight"))
                for j in mlp_live:
                    y = _rmsnorm(hs[j], post_ln, eps)
                    hs[j] = hs[j] + (torch.nn.functional.silu(y @ g.T) * (y @ u.T)) @ d.T
                    del y
                del g, u, d
            del post_ln
            ps.release()
            continue

        router_w = w(lk(i, f"{lay.router}.weight"))
        ys, tops = [], []
        for j in range(A):
            # Routing is computed (and traced) for EVERY arm, including one whose mlp is
            # ablated at this layer — same as the serial kernel, which keeps the trace
            # well-defined and only skips the expert pulls.
            y = _rmsnorm(hs[j], post_ln, eps)
            logits = y @ router_w.T
            rprobs = torch.softmax(logits.float(), dim=-1)
            topw, topi = rprobs.topk(k, dim=-1)
            if lay.norm_topk:
                topw = topw / topw.sum(-1, keepdim=True)
            if lay.name != "mixtral":       # see the serial kernel's routing-dtype note
                topw = topw.to(dtype)
            ys.append(y)
            tops.append((topw, topi))
            if capture_routing:
                routing[j][i] = topi.cpu().numpy().astype(np.int16)
            del logits, rprobs
        del router_w, post_ln

        union = sorted({int(e) for j in mlp_live for e in torch.unique(tops[j][1]).tolist()})
        moes = {j: torch.zeros_like(hs[j]) for j in mlp_live}
        for e in union:
            g = w(lk(i, lay.gate.format(e=e) + ".weight"))
            u = w(lk(i, lay.up.format(e=e) + ".weight"))
            d = w(lk(i, lay.down.format(e=e) + ".weight"))
            gu_w = torch.cat([g, u], dim=0)
            for j in mlp_live:
                topw, topi = tops[j]
                sel = (topi == e)
                tok = sel.any(-1)
                if not bool(tok.any()):
                    continue
                w_e = (topw * sel.to(topw.dtype)).sum(-1)[tok]
                xe = ys[j][tok]
                gu = xe @ gu_w.T
                gg, uu = gu.chunk(2, dim=-1)
                ye = (torch.nn.functional.silu(gg) * uu) @ d.T
                if lay.name == "mixtral":
                    moes[j][tok] += (w_e[:, None] * ye.float()).to(dtype)
                else:
                    moes[j][tok] += w_e[:, None] * ye
                del gu, gg, uu, ye, xe, w_e
            del g, u, d, gu_w
        if lay.name == "qwen2_moe":
            shared_gate = w(lk(i, "mlp.shared_expert.gate_proj.weight"))
            shared_up = w(lk(i, "mlp.shared_expert.up_proj.weight"))
            shared_down = w(lk(i, "mlp.shared_expert.down_proj.weight"))
            shared_gate_w = w(lk(i, "mlp.shared_expert_gate.weight"))
            gu_w = torch.cat([shared_gate, shared_up], dim=0)
            for j in mlp_live:
                shared_gu = ys[j] @ gu_w.T
                shared_g, shared_u = shared_gu.chunk(2, dim=-1)
                shared = (torch.nn.functional.silu(shared_g) * shared_u) @ shared_down.T
                weight = torch.sigmoid(ys[j] @ shared_gate_w.T)
                moes[j] = moes[j] + weight * shared
                del shared_gu, shared_g, shared_u, shared, weight
            del shared_gate, shared_up, shared_down, shared_gate_w, gu_w
        for j in mlp_live:
            hs[j] = hs[j] + moes[j]
        del moes, ys, tops
        ps.release()

        r = _rss_gb()
        if log and (i % 4 == 0 or i == L - 1):
            log(f"    [batch] layer {i:2d}/{L}  arms={A}  RSS={r:.2f}GB  {time.time()-t0:.0f}s")
        if r > abort_rss_gb:
            raise MemoryError(f"RSS {r:.1f}GB > {abort_rss_gb}GB abort")

    norm_w = w("model.norm.weight")
    final_last = torch.stack(
        [_rmsnorm(hs[j], norm_w, eps)[-1].detach().float().cpu() for j in range(A)]
    )
    ps.release()
    out = dict(final_last=final_last, n_arms=A, layout=lay.name,
               wall_s=round(time.time() - t0, 2))
    if capture_routing:
        out["routing"] = routing
    return out


# ------------------------------------------------------------------ trace nulls (vendored, measured on OLMoE)
def lru_hit(stream: np.ndarray, budget: int):
    cache: OrderedDict[int, int] = OrderedDict()
    hits = tot = 0
    for t in range(stream.shape[0]):
        for e in stream[t]:
            e = int(e); tot += 1
            if e in cache:
                cache.move_to_end(e); hits += 1
            else:
                if len(cache) >= budget:
                    cache.popitem(last=False)
                cache[e] = 1
    return hits, tot


def _sweep_hit(idx: np.ndarray, budgets) -> dict:
    out = {}
    L = idx.shape[0]
    for b in budgets:
        h = t = 0
        for li in range(L):
            hh, tt = lru_hit(idx[li], b); h += hh; t += tt
        out[b] = h / t
    return out


def time_shuffle(idx: np.ndarray, rng) -> np.ndarray:
    """Tightest temporal null: permute token order per layer — keeps multiset + marginals,
    destroys only adjacency. real − shuffle = pure temporal locality."""
    out = np.empty_like(idx)
    for li in range(idx.shape[0]):
        out[li] = idx[li, rng.permutation(idx.shape[1])]
    return out


def uniform_iid(shape: tuple, n_exp: int, rng) -> np.ndarray:
    """Measured uniform floor: same trace shape, k DISTINCT experts drawn uniformly per token.
    The analytic k/n floor is wrong for an LRU of capacity b (a uniform trace hits ~b/n) —
    measure the floor with the same estimator instead of asserting it."""
    L, T, k = shape
    out = np.empty((L, T, k), dtype=np.int16)
    for li in range(L):
        for t in range(T):
            out[li, t] = rng.choice(n_exp, size=k, replace=False)
    return out


def locality_decomposition(topk_idx: np.ndarray, n_exp: int, k: int, *,
                           moe_inter: int, hidden: int, dtype_bytes: int = 2) -> dict:
    """LRU hit-rate sweep + null decomposition, all floors MEASURED with the same estimator:
    temporal_locality = hit(real) − hit(time_shuffle)   (order within the token stream)
    skew_share        = hit(time_shuffle) − hit(uniform) (marginal skew + k-set pairing;
                        both nulls are orderless with k distinct experts per token, so the
                        contrast is matched on everything except the routing distribution —
                        a flattened-stream null is NOT matched: duplicate picks inside a
                        token fabricate ~+0.07 hit on a provably zero-skew trace, measured)
    Miss-byte accounting: expert = 3 SwiGLU mats at dtype_bytes."""
    L = topk_idx.shape[0]
    if n_exp <= 16:
        budgets = list(range(k, n_exp))
    else:
        budgets = sorted({k, *[int(round(n_exp * f)) for f in (0.12, 0.19, 0.25, 0.5, 0.75)]})
        budgets = [b for b in budgets if k <= b < n_exp]
    expert_bytes = 3 * moe_inter * hidden * dtype_bytes
    real = _sweep_hit(topk_idx, budgets)
    shuf = _sweep_hit(time_shuffle(topk_idx, np.random.default_rng(2)), budgets)
    unif = _sweep_hit(uniform_iid(topk_idx.shape, n_exp, np.random.default_rng(0)), budgets)
    rows = []
    for b in budgets:
        miss = 1 - real[b]
        rows.append(dict(
            budget=b, resident_frac=round(b / n_exp, 3),
            hit=round(real[b], 4), hit_shuffle=round(shuf[b], 4),
            hit_uniform=round(unif[b], 4),
            temporal_locality=round(real[b] - shuf[b], 4),
            skew_share=round(shuf[b] - unif[b], 4),
            bytes_per_token_mb=round(miss * k * L * expert_bytes / 1e6, 1),
            divide_locality=round((n_exp / k) / max(1e-9, miss), 1)))
    return dict(expert_mb=round(expert_bytes / 1e6, 2), sweep=rows)


# ------------------------------------------------------------------ physiology (moe_live face)
_FALLBACK_PROSE = (
    "Measurement is the discipline of doubting your own instruments. A number that has not "
    "survived a null control is a rumor with a decimal point. Speed in this regime is fewer "
    "bytes dragged off the disk per token, and the model spends most of itself idle. ") * 40
