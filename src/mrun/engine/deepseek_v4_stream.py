"""Streamed DeepSeek-V4 forward for the moe-stream engine — native-checkpoint layout.

Ported op-for-op from transformers 5.13.1 ``modeling_deepseek_v4.py`` (the parity oracle for
this adapter), reading the upstream (V3-style) tensor names the release checkpoint actually
ships (``layers.N.attn.wq_a.weight`` …), NOT the HF module names — the HF names only exist
after transformers' ``conversion_mapping`` rewrites them at load time.

Architecture recap (V4-Flash: 43 layers, d=4096, 64 heads x 512, 256 experts top-6 + 1 shared):

* MLA-style attention: low-rank Q (``wq_a`` -> weighted RMS ``q_norm`` -> ``wq_b`` -> unweighted
  per-head RMS), single shared KV head (K==V), interleaved partial RoPE on the trailing
  ``qk_rope_head_dim`` dims, per-head learnable sink, inverse-RoPE on the attention output,
  grouped low-rank output projection (``wo_a`` block-diagonal, ``wo_b`` mix).
* Sparse-index attention: sliding window (128) everywhere; CSA layers add a gated-pool
  compressor (ratio 4, overlapping Ca/Cb series) whose entries are selected per query by a
  Lightning Indexer (top ``index_topk``); HCA layers add a non-overlapping ratio-128 compressor
  with pure causal visibility. Layer types come from the legacy ``compress_ratios`` list
  (0=sliding, 4=CSA, 128=HCA) or an explicit ``layer_types``.
* Hyper-connections: the residual is ``hc_mult`` parallel streams ``[T, hc, d]``; each sublayer
  site collapses them with learned ``pre`` weights, and re-mixes with ``post`` + a
  Sinkhorn-projected doubly-stochastic ``comb``. All HC math runs in fp32 (HF keeps those
  modules fp32 even under bf16). The TRUE input-embedding site is the ``[T, d]`` vector BEFORE
  stream expansion — the embedding taps (``ablate_embed_direction`` / ``zero_embedding``) apply
  there, which is exactly HF's ``inputs_embeds``.
* Router: ``sqrt(softplus(logits))`` scoring; noaux_tc top-6 (selection on scores + fp32
  ``gate.bias``, weights from the UNbiased scores, renormalized, x ``routed_scaling_factor``);
  the first ``num_hash_layers`` layers route by the frozen ``tid2eid[input_ids]`` table
  instead of top-k. One always-on shared SwiGLU expert; all expert gate/up pre-activations
  clamp at ``swiglu_limit``.

Known, deliberate divergence from the SHIPPED reference (``inference/model.py``): the official
stack QAT-simulates activation quantization (fp8 on the non-rope KV dims, Hadamard+fp4 in the
indexer). transformers 5.13.1 omits all activation quant-sim, and THAT forward is the one this
port matches — the parity gate is against transformers, not the tilelang stack.

Sequence-length caveat (stated, not hidden): the Lightning Indexer keeps ``min(index_topk,
n_windows)`` entries, so with ``index_topk=512`` and ratio 4 its learned SELECTION only starts
truncating at T > 2048 tokens. At battery-scale prompts (T <= ~512) every causally-valid entry
is kept and the indexer only enforces causality — the parity gates exercise that regime.
"""
from __future__ import annotations

import math
import time
from typing import NamedTuple

import numpy as np
import torch


def _rss_gb() -> float:
    from .moe_safetensors import _rss_gb as impl
    return impl()


# ------------------------------------------------------------------ config spec
class DSV4Spec(NamedTuple):
    n_layers: int
    hidden: int
    n_heads: int
    head_dim: int
    rope_dim: int            # qk_rope_head_dim
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    sliding_window: int
    layer_types: tuple[str, ...]        # sliding | csa | hca per layer
    csa_ratio: int
    hca_ratio: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    n_experts: int
    top_k: int
    n_hash_layers: int
    moe_inter: int
    swiglu_limit: float
    routed_scaling_factor: float
    rms_eps: float
    hc_mult: int
    hc_iters: int
    hc_eps: float
    theta_main: float
    theta_compress: float
    yarn_factor: float
    yarn_orig_max: int
    yarn_beta_fast: float
    yarn_beta_slow: float
    vocab: int


_RATIO_TO_TYPE = {0: "sliding", 4: "csa", 128: "hca"}
_HF_TYPE = {"sliding_attention": "sliding", "compressed_sparse_attention": "csa",
            "heavily_compressed_attention": "hca"}


def parse_dsv4_spec(cfg: dict) -> DSV4Spec:
    """Layer schedule + dims from a raw ``config.json`` dict, mirroring
    ``DeepseekV4Config.__post_init__`` (legacy ``compress_ratios`` / ``num_hash_layers`` /
    ``qk_rope_head_dim`` handling included)."""
    n = int(cfg["num_hidden_layers"])
    head_dim = int(cfg.get("head_dim") or 512)
    if cfg.get("layer_types"):
        layer_types = tuple(_HF_TYPE[t] for t in cfg["layer_types"][:n])
    elif cfg.get("compress_ratios") is not None:
        layer_types = tuple(_RATIO_TO_TYPE[int(r)] for r in cfg["compress_ratios"][:n])
    else:
        raise NotImplementedError("deepseek_v4 config has neither layer_types nor compress_ratios")
    rates = cfg.get("compress_rates") or {}
    csa_ratio = int(rates.get("compressed_sparse_attention", cfg.get("compress_rate_csa", 4)))
    hca_ratio = int(rates.get("heavily_compressed_attention", cfg.get("compress_rate_hca", 128)))
    if cfg.get("mlp_layer_types"):
        mlt = list(cfg["mlp_layer_types"][:n])
        n_hash = sum(1 for t in mlt if t == "hash_moe")
        if mlt != ["hash_moe"] * n_hash + ["moe"] * (n - n_hash):
            raise NotImplementedError(f"non-prefix hash_moe schedule unsupported: {mlt}")
    else:
        n_hash = int(cfg.get("num_hash_layers", 0))
    rope_dim = int(cfg.get("qk_rope_head_dim") or round(
        head_dim * float(cfg.get("partial_rotary_factor") or (64 / 512))))
    scaling = cfg.get("rope_scaling") or {}
    stype = scaling.get("type", scaling.get("rope_type", "default"))
    if scaling and stype != "yarn":
        raise NotImplementedError(f"deepseek_v4 rope_scaling type {stype!r} unsupported")
    return DSV4Spec(
        n_layers=n,
        hidden=int(cfg["hidden_size"]),
        n_heads=int(cfg["num_attention_heads"]),
        head_dim=head_dim,
        rope_dim=rope_dim,
        q_lora_rank=int(cfg["q_lora_rank"]),
        o_groups=int(cfg["o_groups"]),
        o_lora_rank=int(cfg["o_lora_rank"]),
        sliding_window=int(cfg["sliding_window"]),
        layer_types=layer_types,
        csa_ratio=csa_ratio,
        hca_ratio=hca_ratio,
        index_n_heads=int(cfg["index_n_heads"]),
        index_head_dim=int(cfg["index_head_dim"]),
        index_topk=int(cfg["index_topk"]),
        n_experts=int(cfg["n_routed_experts"]),
        top_k=int(cfg["num_experts_per_tok"]),
        n_hash_layers=n_hash,
        moe_inter=int(cfg["moe_intermediate_size"]),
        swiglu_limit=float(cfg.get("swiglu_limit", 0.0) or 0.0),
        routed_scaling_factor=float(cfg.get("routed_scaling_factor", 1.0)),
        rms_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        hc_mult=int(cfg["hc_mult"]),
        hc_iters=int(cfg["hc_sinkhorn_iters"]),
        hc_eps=float(cfg["hc_eps"]),
        theta_main=float(cfg.get("rope_theta", 1e4)),
        theta_compress=float(cfg.get("compress_rope_theta", 1e4)),
        yarn_factor=float(scaling.get("factor", 1.0)),
        yarn_orig_max=int(scaling.get("original_max_position_embeddings", 0)),
        yarn_beta_fast=float(scaling.get("beta_fast", 32)),
        yarn_beta_slow=float(scaling.get("beta_slow", 1)),
        vocab=int(cfg["vocab_size"]),
    )


# ------------------------------------------------------------------ rope (interleaved, yarn)
def _yarn_inv_freq(dim: int, base: float, factor: float, beta_fast: float, beta_slow: float,
                   original_max: int, device) -> torch.Tensor:
    """transformers 5.13.1 ``_compute_yarn_parameters`` inv_freq, truncate=True. The V4 config
    forces ``attention_factor=1.0`` for the compress rope, so no mscale multiplies cos/sin."""
    pos_freqs = base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (factor * pos_freqs)

    def correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp(
        (torch.arange(dim // 2, device=device, dtype=torch.float32) - low) / (high - low), 0, 1
    )
    extra_factor = 1 - ramp
    return inv_inter * (1 - extra_factor) + inv_extra * extra_factor


def dsv4_rope_tables(spec: DSV4Spec, n_pos: int, device, dtype) -> dict[str, tuple]:
    """cos/sin ``[n_pos, rope_dim // 2]`` per rope family (one value per interleaved pair,
    matching HF's DeepseekV4RotaryEmbedding output before ``repeat_interleave``)."""
    out = {}
    positions = torch.arange(n_pos, device=device, dtype=torch.float32)
    for name, theta in (("main", spec.theta_main), ("compress", spec.theta_compress)):
        if name == "compress" and spec.yarn_orig_max > 0 and spec.yarn_factor != 1.0:
            inv = _yarn_inv_freq(spec.rope_dim, theta, spec.yarn_factor, spec.yarn_beta_fast,
                                 spec.yarn_beta_slow, spec.yarn_orig_max, device)
        else:
            inv = 1.0 / (theta ** (
                torch.arange(0, spec.rope_dim, 2, device=device, dtype=torch.float32)
                / spec.rope_dim))
        freqs = torch.outer(positions, inv)                    # [n_pos, rope_dim/2]
        out[name] = (freqs.cos().to(dtype), freqs.sin().to(dtype))
    return out


def _apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF ``apply_rotary_pos_emb``: interleaved rotation of the TRAILING rope slice.

    ``x``: ``[..., D]`` with the rope dims last; ``cos``/``sin``: broadcastable to
    ``x[..., -rd:]`` after ``repeat_interleave(2, -1)`` (callers pre-shape, e.g. ``[T, 1, rd/2]``
    for per-head tensors ``[T, H, D]``)."""
    cos2 = cos.repeat_interleave(2, dim=-1)
    sin2 = sin.repeat_interleave(2, dim=-1)
    rd = cos2.shape[-1]
    nope, rope = x[..., :-rd], x[..., -rd:]
    rot = torch.stack((-rope[..., 1::2], rope[..., 0::2]), dim=-1).flatten(-2)
    rotated = ((rope.float() * cos2) + (rot.float() * sin2)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


# ------------------------------------------------------------------ small math
def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _unweighted_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def _hc_mix(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor,
            spec: DSV4Spec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HF DeepseekV4HyperConnection.forward for ``streams [T, hc, d]`` (fp32 throughout).
    Returns ``(collapsed [T, d], post [T, hc], comb [T, hc, hc])``."""
    hc = spec.hc_mult
    flat = streams.flatten(1).float()
    flat = _unweighted_rms(flat, spec.rms_eps)
    mixes = flat @ fn.float().T                                   # [T, (2+hc)*hc]
    pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.float().split([hc, hc, hc * hc])
    s0, s1, s2 = scale.float().unbind(0)
    pre = torch.sigmoid(pre_w * s0 + pre_b) + spec.hc_eps
    post = 2 * torch.sigmoid(post_w * s1 + post_b)
    comb = torch.softmax(comb_w.view(-1, hc, hc) * s2 + comb_b.view(hc, hc), dim=-1) + spec.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + spec.hc_eps)
    for _ in range(spec.hc_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + spec.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + spec.hc_eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=1).to(streams.dtype)
    return collapsed, post, comb


def _hc_recombine(streams: torch.Tensor, sub_out: torch.Tensor | None, post: torch.Tensor,
                  comb: torch.Tensor) -> torch.Tensor:
    """``post * out + comb^T @ streams``; ``sub_out=None`` = ablated sublayer (exact zero)."""
    dtype = streams.dtype
    carried = torch.matmul(comb.to(dtype).transpose(-1, -2), streams)
    if sub_out is None:
        return carried
    return post.to(dtype).unsqueeze(-1) * sub_out.unsqueeze(-2) + carried


def _hc_head_collapse(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor,
                      scale: torch.Tensor, spec: DSV4Spec) -> torch.Tensor:
    flat = streams.flatten(1).float()
    flat = _unweighted_rms(flat, spec.rms_eps)
    mixes = flat @ fn.float().T
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + spec.hc_eps
    return (pre.unsqueeze(-1) * streams).sum(dim=1).to(streams.dtype)


# ------------------------------------------------------------------ compressor + indexer
def _compress(x: torch.Tensor, wkv: torch.Tensor, wgate: torch.Tensor, ape: torch.Tensor,
              norm_w: torch.Tensor, ratio: int, head_dim: int, overlap: bool,
              cos_tab: torch.Tensor, sin_tab: torch.Tensor, eps: float) -> torch.Tensor:
    """Stateless single-prefill compressor (HF CSA/HCA compressor with no cache): compress every
    complete window of ``ratio`` tokens, drop the remainder. Returns ``[n_win, head_dim]``."""
    T = int(x.shape[0])
    usable = (T // ratio) * ratio
    if usable == 0:
        return x.new_zeros((0, head_dim))
    n_win = usable // ratio
    kv = (x @ wkv.T)[:usable].view(n_win, ratio, -1)
    gate = (x @ wgate.T)[:usable].view(n_win, ratio, -1) + ape
    if overlap:
        new_kv = kv.new_zeros((n_win, 2 * ratio, head_dim))
        new_gate = gate.new_full((n_win, 2 * ratio, head_dim), float("-inf"))
        new_kv[:, ratio:] = kv[..., head_dim:]
        new_gate[:, ratio:] = gate[..., head_dim:]
        if n_win > 1:
            new_kv[1:, :ratio] = kv[:-1, :, :head_dim]
            new_gate[1:, :ratio] = gate[:-1, :, :head_dim]
        kv, gate = new_kv, new_gate
    weights = gate.softmax(dim=1, dtype=torch.float32).to(kv.dtype)
    compressed = _rmsnorm((kv * weights).sum(dim=1), norm_w, eps)
    win_pos = torch.arange(n_win, device=x.device) * ratio
    return _apply_rope_interleaved(compressed, cos_tab[win_pos], sin_tab[win_pos])


def _hca_bias(T: int, n_win: int, ratio: int, ref: torch.Tensor) -> torch.Tensor:
    """Causal block bias ``[T, n_win]``: query t sees entry w iff ``w < (t+1)//ratio``."""
    thresh = (torch.arange(T, device=ref.device) + 1) // ratio
    entry = torch.arange(n_win, device=ref.device)
    bias = ref.new_zeros((T, n_win))
    return bias.masked_fill(entry[None, :] >= thresh[:, None], float("-inf"))


def _indexer_topk(x: torch.Tensor, q_res: torch.Tensor, idx_compressed: torch.Tensor,
                  wq_b: torch.Tensor, w_weights: torch.Tensor, spec: DSV4Spec,
                  cos_q: torch.Tensor, sin_q: torch.Tensor) -> torch.Tensor:
    """HF DeepseekV4Indexer scoring + top-k with causal invalidation. Returns ``[T, k]`` with
    ``-1`` marking picks past a query's causal threshold."""
    T = int(x.shape[0])
    W = int(idx_compressed.shape[0])
    q = (q_res @ wq_b.T).view(T, spec.index_n_heads, spec.index_head_dim)
    q = _apply_rope_interleaved(q, cos_q[:, None, :], sin_q[:, None, :])
    scores = torch.relu(q.float() @ idx_compressed.float().T) * (spec.index_head_dim ** -0.5)
    weights = (x @ w_weights.T).float() * (spec.index_n_heads ** -0.5)          # [T, H]
    index_scores = (scores * weights.unsqueeze(-1)).sum(dim=1)                  # [T, W]
    thresh = (torch.arange(T, device=x.device) + 1) // spec.csa_ratio
    entry = torch.arange(W, device=x.device)
    index_scores = index_scores.masked_fill(entry[None, :] >= thresh[:, None], float("-inf"))
    top_k = min(spec.index_topk, W)
    top_idx = index_scores.topk(top_k, dim=-1).indices                          # [T, k]
    invalid = top_idx >= thresh[:, None]
    return torch.where(invalid, torch.full_like(top_idx, -1), top_idx)


def _csa_bias(top_idx: torch.Tensor, T: int, n_win: int, ref: torch.Tensor) -> torch.Tensor:
    """Indexer picks -> block bias ``[T, n_win]``: 0 at kept valid entries, -inf elsewhere."""
    valid = top_idx >= 0
    safe = torch.where(valid, top_idx, torch.full_like(top_idx, n_win))
    bias = ref.new_full((T, n_win + 1), float("-inf"))
    bias.scatter_(-1, safe, 0.0)
    return bias[:, :n_win]


# ------------------------------------------------------------------ per-layer weight bundles
class _LayerKeys:
    """Key strings for one decoder layer in the native namespace."""

    def __init__(self, i: int):
        p = f"layers.{i}."
        a = p + "attn."
        self.attn_norm = p + "attn_norm.weight"
        self.ffn_norm = p + "ffn_norm.weight"
        self.hc_attn = (p + "hc_attn_fn", p + "hc_attn_base", p + "hc_attn_scale")
        self.hc_ffn = (p + "hc_ffn_fn", p + "hc_ffn_base", p + "hc_ffn_scale")
        self.wq_a = a + "wq_a.weight"
        self.q_norm = a + "q_norm.weight"
        self.wq_b = a + "wq_b.weight"
        self.wkv = a + "wkv.weight"
        self.kv_norm = a + "kv_norm.weight"
        self.wo_a = a + "wo_a.weight"
        self.wo_b = a + "wo_b.weight"
        self.sink = a + "attn_sink"
        self.comp_wkv = a + "compressor.wkv.weight"
        self.comp_wgate = a + "compressor.wgate.weight"
        self.comp_ape = a + "compressor.ape"
        self.comp_norm = a + "compressor.norm.weight"
        self.idx_wkv = a + "indexer.compressor.wkv.weight"
        self.idx_wgate = a + "indexer.compressor.wgate.weight"
        self.idx_ape = a + "indexer.compressor.ape"
        self.idx_norm = a + "indexer.compressor.norm.weight"
        self.idx_wq_b = a + "indexer.wq_b.weight"
        self.idx_weights = a + "indexer.weights_proj.weight"
        self.gate_w = p + "ffn.gate.weight"
        self.gate_bias = p + "ffn.gate.bias"
        self.tid2eid = p + "ffn.gate.tid2eid"
        self.shared_w1 = p + "ffn.shared_experts.w1.weight"
        self.shared_w2 = p + "ffn.shared_experts.w2.weight"
        self.shared_w3 = p + "ffn.shared_experts.w3.weight"
        self.expert = p + "ffn.experts.{e}.{w}.weight"


def _attention(x: torch.Tensor, lw: dict, spec: DSV4Spec, layer_type: str,
               tables: dict, sw_mask: torch.Tensor, eps: float) -> torch.Tensor:
    """One full attention sublayer on ``x [T, d]`` (single prefill, all positions)."""
    T = int(x.shape[0])
    pos = torch.arange(T, device=x.device)
    rope_name = "main" if layer_type == "sliding" else "compress"
    cos_tab, sin_tab = tables[rope_name]
    cos, sin = cos_tab[pos], sin_tab[pos]

    q_res = _rmsnorm(x @ lw["wq_a"].T, lw["q_norm"], eps)
    q = (q_res @ lw["wq_b"].T).view(T, spec.n_heads, spec.head_dim)
    q = _unweighted_rms(q, eps)
    q = _apply_rope_interleaved(q, cos[:, None, :], sin[:, None, :])

    kv = _rmsnorm(x @ lw["wkv"].T, lw["kv_norm"], eps)              # [T, head_dim]
    kv = _apply_rope_interleaved(kv, cos, sin)

    bias = None
    keys = kv
    if layer_type != "sliding":
        ratio = spec.csa_ratio if layer_type == "csa" else spec.hca_ratio
        compressed = _compress(
            x, lw["comp_wkv"], lw["comp_wgate"], lw["comp_ape"], lw["comp_norm"],
            ratio, spec.head_dim, overlap=(layer_type == "csa"),
            cos_tab=cos_tab, sin_tab=sin_tab, eps=eps)
        n_win = int(compressed.shape[0])
        if n_win > 0:
            if layer_type == "csa":
                idx_compressed = _compress(
                    x, lw["idx_wkv"], lw["idx_wgate"], lw["idx_ape"], lw["idx_norm"],
                    ratio, spec.index_head_dim, overlap=True,
                    cos_tab=cos_tab, sin_tab=sin_tab, eps=eps)
                top_idx = _indexer_topk(x, q_res, idx_compressed, lw["idx_wq_b"],
                                        lw["idx_weights"], spec, cos, sin)
                bias = _csa_bias(top_idx, T, n_win, kv)
            else:
                bias = _hca_bias(T, n_win, ratio, kv)
            keys = torch.cat([kv, compressed], dim=0)

    mask = sw_mask if bias is None else torch.cat([sw_mask, bias.to(sw_mask.dtype)], dim=-1)
    scale = spec.head_dim ** -0.5
    logits = q.transpose(0, 1) @ keys.T * scale + mask              # [H, T, K]
    sinks = lw["sink"].float().view(-1, 1, 1).expand(-1, T, 1).to(logits.dtype)
    combined = torch.cat([logits, sinks], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined, dim=-1)[..., :-1]               # HF: softmax in logit dtype
    ctx = (probs.to(keys.dtype) @ keys).transpose(0, 1)             # [T, H, head_dim]

    ctx = _apply_rope_interleaved(ctx, cos[:, None, :], -sin[:, None, :])
    # HF DeepseekV4GroupedLinear op-for-op: view weight [g, r, d_g] -> transpose -> bmm.
    grouped = ctx.reshape(T, spec.o_groups, -1)                     # [T, g, H*hd/g]
    d_g = grouped.shape[-1]
    wo_a = lw["wo_a"].view(spec.o_groups, -1, d_g).transpose(1, 2)  # [g, d_g, r]
    o = torch.bmm(grouped.transpose(0, 1), wo_a).transpose(0, 1)    # [T, g, r]
    return o.reshape(T, -1) @ lw["wo_b"].T


def _sliding_causal_mask(T: int, window: int, device, dtype) -> torch.Tensor:
    """``[T, T]``: 0 where ``kv <= q`` AND ``kv > q - window``, else -inf (HF overlay)."""
    q = torch.arange(T, device=device)[:, None]
    kv = torch.arange(T, device=device)[None, :]
    allowed = (kv <= q) & (kv > q - window)
    mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    return mask.masked_fill(allowed, 0.0)


# ------------------------------------------------------------------ forward (batched core)
@torch.no_grad()
def dsv4_forward_arms(ps, rows: list[torch.Tensor],
                      arms: list[tuple[int, tuple[int, str] | None]], *,
                      device: str = "cpu", dtype: torch.dtype = torch.float32,
                      ablate_embed_direction: torch.Tensor | None = None,
                      ablate_embed_alpha: float = 1.0,
                      zero_embedding: bool = False,
                      capture_routing: bool = False,
                      capture_hidden_states: bool = False,
                      capture_final_hidden: bool = False,
                      return_logits: bool = False,
                      skip_lm_head: bool = True,
                      abort_rss_gb: float = 48.0, log=None) -> dict:
    """One weight-streaming pass serving many (sequence, component-ablation) arms — the
    DeepSeek-V4 analogue of ``streamed_moe_forward_batch`` with the serial captures available
    on top (used by the serial wrapper with a single arm).

    Same contract as the generic kernel: every arm keeps its own ``[T, hc, d]`` stream stack
    and its own attention — rows are never concatenated. Embedding taps apply to every arm
    (at the TRUE input site: the ``[T, d]`` embedding before hc expansion); the per-arm axis
    is the whole-component ablation, realised as "the sublayer's output term is exactly zero
    in the hyper-connection recombination" (its weights are then never pulled).
    """
    spec = parse_dsv4_spec(ps.cfg)
    hc = spec.hc_mult
    eps = spec.rms_eps
    for a_row, ablate in arms:
        if not 0 <= int(a_row) < len(rows):
            raise ValueError(f"arm row {a_row} out of range for {len(rows)} rows")
        if ablate is not None:
            li, kind = int(ablate[0]), str(ablate[1])
            if kind not in ("attn", "mlp") or not 0 <= li < spec.n_layers:
                raise ValueError(f"bad arm ablation {ablate!r}")

    def w(key: str) -> torch.Tensor:
        return ps.get_on(key, device, dtype)

    t0 = time.time()
    rows_t = []
    for r in rows:
        row = torch.as_tensor(np.asarray(r, dtype=np.int64), dtype=torch.long)
        if row.ndim != 1 or not row.numel():
            raise ValueError("each row must be a non-empty one-dimensional token array")
        rows_t.append(row.to(device))
    lengths = sorted({int(r.shape[0]) for r in rows_t})
    n_pos = max(lengths)
    tables = dsv4_rope_tables(spec, n_pos, device, dtype)
    sw_masks = {T: _sliding_causal_mask(T, spec.sliding_window, device, dtype)
                for T in lengths}

    emb = w("embed.weight")
    A = len(arms)
    streams: list[torch.Tensor] = []
    embeds_tap: list[torch.Tensor] = []     # post-tap [T, d] (the true input site), per arm
    for a_row, _ablate in arms:
        h = emb[rows_t[a_row]].clone()
        if zero_embedding:
            h = torch.zeros_like(h)
        elif ablate_embed_direction is not None:
            basis = torch.as_tensor(ablate_embed_direction, dtype=torch.float32, device=device)
            if basis.ndim == 1:
                basis = (basis / (basis.norm() + 1e-9))[:, None]
            hf32 = h.float()
            h = (hf32 - ablate_embed_alpha * (hf32 @ basis) @ basis.T).to(h.dtype)
            del hf32, basis
        embeds_tap.append(h)
        streams.append(h.unsqueeze(1).expand(-1, hc, -1).contiguous())
    del emb
    ps.release()

    hidden_states = [[embeds_tap[j].detach().cpu()] for j in range(A)] \
        if capture_hidden_states else None
    routing = None
    if capture_routing:
        routing = [np.full((spec.n_layers, int(streams[j].shape[0]), spec.top_k), -1,
                           dtype=np.int16) for j in range(A)]
    topw_trace = [np.zeros((spec.n_layers, int(streams[j].shape[0]), spec.top_k),
                           dtype=np.float32) for j in range(A)] if capture_routing else None

    for i in range(spec.n_layers):
        keys_i = _LayerKeys(i)
        layer_type = spec.layer_types[i]
        hc_attn = tuple(w(k) for k in keys_i.hc_attn)
        hc_ffn = tuple(w(k) for k in keys_i.hc_ffn)

        # ---- attention site --------------------------------------------------------------
        attn_live = [j for j in range(A) if arms[j][1] != (i, "attn")]
        lw = None
        if attn_live:
            lw = {
                "wq_a": w(keys_i.wq_a), "q_norm": w(keys_i.q_norm), "wq_b": w(keys_i.wq_b),
                "wkv": w(keys_i.wkv), "kv_norm": w(keys_i.kv_norm),
                "wo_a": w(keys_i.wo_a), "wo_b": w(keys_i.wo_b), "sink": w(keys_i.sink),
            }
            if layer_type != "sliding":
                lw.update(comp_wkv=w(keys_i.comp_wkv), comp_wgate=w(keys_i.comp_wgate),
                          comp_ape=w(keys_i.comp_ape), comp_norm=w(keys_i.comp_norm))
            if layer_type == "csa":
                lw.update(idx_wkv=w(keys_i.idx_wkv), idx_wgate=w(keys_i.idx_wgate),
                          idx_ape=w(keys_i.idx_ape), idx_norm=w(keys_i.idx_norm),
                          idx_wq_b=w(keys_i.idx_wq_b), idx_weights=w(keys_i.idx_weights))
            attn_norm_w = w(keys_i.attn_norm)
        for j in range(A):
            S = streams[j]
            collapsed, post, comb = _hc_mix(S, hc_attn[0], hc_attn[1], hc_attn[2], spec)
            if j in attn_live:
                x = _rmsnorm(collapsed, attn_norm_w, eps)
                T = int(x.shape[0])
                out = _attention(x, lw, spec, layer_type, tables, sw_masks[T], eps)
            else:
                out = None
            streams[j] = _hc_recombine(S, out, post, comb)
        del lw, hc_attn

        # ---- ffn (MoE) site --------------------------------------------------------------
        ffn_norm_w = w(keys_i.ffn_norm)
        gate_w = w(keys_i.gate_w)
        is_hash = i < spec.n_hash_layers
        gate_bias = None if is_hash else w(keys_i.gate_bias).float()
        tid2eid = ps.get(keys_i.tid2eid).to(device) if is_hash else None
        mlp_live = [j for j in range(A) if arms[j][1] != (i, "mlp")]

        ys, posts, combs, tops = [], [], [], []
        for j in range(A):
            collapsed, post, comb = _hc_mix(streams[j], hc_ffn[0], hc_ffn[1], hc_ffn[2], spec)
            y = _rmsnorm(collapsed, ffn_norm_w, eps)
            logits_r = y @ gate_w.T
            scores = torch.sqrt(torch.nn.functional.softplus(logits_r.float()))
            if is_hash:
                topi = tid2eid[rows_t[arms[j][0]]].long()               # [T, k]
            else:
                topi = torch.topk(scores + gate_bias, spec.top_k, dim=-1, sorted=False).indices
            topw = scores.gather(1, topi)
            topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)
            topw = (topw * spec.routed_scaling_factor).to(dtype)
            ys.append(y)
            posts.append(post)
            combs.append(comb)
            tops.append((topw, topi))
            if capture_routing:
                routing[j][i] = topi.cpu().numpy().astype(np.int16)
                topw_trace[j][i] = topw.float().cpu().numpy()
            del logits_r, scores
        del gate_w, gate_bias, tid2eid

        moes = {j: torch.zeros_like(ys[j]) for j in mlp_live}
        union = sorted({int(e) for j in mlp_live for e in torch.unique(tops[j][1]).tolist()})
        for e in union:
            g = w(keys_i.expert.format(e=e, w="w1"))
            u = w(keys_i.expert.format(e=e, w="w3"))
            d = w(keys_i.expert.format(e=e, w="w2"))
            gu_w = torch.cat([g, u], dim=0)
            for j in mlp_live:
                topw, topi = tops[j]
                sel = topi == e
                tok = sel.any(-1)
                if not bool(tok.any()):
                    continue
                w_e = (topw * sel.to(topw.dtype)).sum(-1)[tok]
                gu = ys[j][tok] @ gu_w.T
                gg, uu = gu.chunk(2, dim=-1)
                if spec.swiglu_limit > 0:
                    gg = gg.clamp(max=spec.swiglu_limit)
                    uu = uu.clamp(min=-spec.swiglu_limit, max=spec.swiglu_limit)
                ye = (torch.nn.functional.silu(gg) * uu) @ d.T
                moes[j][tok] += w_e[:, None] * ye
                del gu, gg, uu, ye, w_e
            del g, u, d, gu_w
        if mlp_live:
            # HF DeepseekV4MLP runs SEPARATE gate/up GEMMs for the shared expert (only the
            # routed experts ship load-fused gate_up) — mirror that, not a fused cat-GEMM.
            s1 = w(keys_i.shared_w1)
            s3 = w(keys_i.shared_w3)
            s2 = w(keys_i.shared_w2)
            for j in mlp_live:
                gg = (ys[j] @ s1.T).clamp(max=spec.swiglu_limit) \
                    if spec.swiglu_limit > 0 else ys[j] @ s1.T
                uu = ys[j] @ s3.T
                if spec.swiglu_limit > 0:
                    uu = uu.clamp(min=-spec.swiglu_limit, max=spec.swiglu_limit)
                moes[j] = moes[j] + (torch.nn.functional.silu(gg) * uu) @ s2.T
                del gg, uu
            del s1, s2, s3
        for j in range(A):
            streams[j] = _hc_recombine(streams[j], moes.get(j), posts[j], combs[j])
            if capture_hidden_states:
                hidden_states[j].append(streams[j].detach().cpu())
        del moes, ys, posts, combs, tops
        ps.release()

        r = _rss_gb()
        if log and (i % 4 == 0 or i == spec.n_layers - 1):
            log(f"    [dsv4] layer {i:2d}/{spec.n_layers}  arms={A}  type={layer_type}  "
                f"RSS={r:.2f}GB  {time.time() - t0:.0f}s")
        if r > abort_rss_gb:
            raise MemoryError(f"RSS {r:.1f}GB > {abort_rss_gb}GB abort")

    # ---- head ----------------------------------------------------------------------------
    hh = tuple(w(k) for k in ("hc_head_fn", "hc_head_base", "hc_head_scale"))
    norm_w = w("norm.weight")
    finals = []
    for j in range(A):
        collapsed = _hc_head_collapse(streams[j], hh[0], hh[1], hh[2], spec)
        finals.append(_rmsnorm(collapsed, norm_w, eps))
        if capture_hidden_states:
            hidden_states[j][-1] = finals[j].detach().cpu()
    del hh, norm_w

    out: dict = dict(n_arms=A, layout="deepseek_v4", L=spec.n_layers, k=spec.top_k,
                     n_exp=spec.n_experts, hidden=spec.hidden,
                     wall_s=round(time.time() - t0, 2), rss_gb=round(_rss_gb(), 2))
    out["final_last"] = torch.stack([f[-1].detach().float().cpu() for f in finals])
    if capture_final_hidden:
        out["final_hidden_arms"] = [f.detach().float().cpu() for f in finals]
    if capture_routing:
        out["routing"] = routing
        out["routing_weights"] = topw_trace
    if capture_hidden_states:
        out["hidden_states_arms"] = hidden_states

    if not skip_lm_head:
        head = w("head.weight")
        logits_arms, ces = [], []
        for j in range(A):
            lg = finals[j] @ head.T
            logp = torch.log_softmax(lg.float(), dim=-1)
            ids_j = rows_t[arms[j][0]]
            Tj = int(ids_j.shape[0])
            ce = (-logp[torch.arange(Tj - 1, device=device), ids_j[1:]].mean().item()
                  if Tj > 1 else float("nan"))
            ces.append(ce)
            logits_arms.append(lg.float().cpu() if return_logits else None)
        del head
        ps.release()
        out["ce_arms"] = ces
        if return_logits:
            out["logits_arms"] = logits_arms
    ps.release()
    return out


@torch.no_grad()
def dsv4_forward(ps, ids: torch.Tensor, *,
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
    """Serial-surface wrapper: same result keys as ``streamed_moe_forward`` so
    ``MoEStreamEngine._run`` works unchanged.

    NOT SUPPORTED (loud, not silent): ``capture_expert_stats`` (per-expert neuron stats need a
    dedicated fp4-expert surface) and ``resid_patch_ops_by_layer`` (under hyper-connections
    "the residual" is an ``[T, hc, d]`` stack; a single-vector projection op is ill-defined
    until a prereg says which stream object it should edit).

    ``hidden_states`` shape note (length ``L + 1``, mirroring HF's ``output_hidden_states``):
    entry 0 is the post-tap embedding ``[T, d]`` (the true input site); entries ``1..L-1`` are
    the hyper-connection stream STACKS ``[T, hc, d]`` after layers ``0..L-2``; and the LAST
    entry is the post-final-norm collapsed state ``[T, d]``, which REPLACES layer ``L-1``'s
    stack (the ``hidden_states[-1] = ...`` house convention, same as ``moe_stream.py`` and
    ``paged_forward.py``). Layer ``L-1``'s raw stack is therefore NOT exposed — consumers that
    need every layer's stack must capture it themselves; comparisons over "all layers" must
    iterate ``1..L-1`` only (this off-by-one crashed the real-checkpoint parity gate once).
    """
    if capture_expert_stats:
        raise NotImplementedError("deepseek_v4 moe-stream: capture_expert_stats not implemented")
    if resid_patch_ops_by_layer:
        raise NotImplementedError(
            "deepseek_v4 moe-stream: residual patch ops are ill-defined on the hc stream stack")
    row = torch.as_tensor(np.asarray(ids, dtype=np.int64), dtype=torch.long)
    res = dsv4_forward_arms(
        ps, [row], [(0, ablate_component)],
        device=device, dtype=dtype,
        ablate_embed_direction=ablate_embed_direction,
        ablate_embed_alpha=ablate_embed_alpha,
        zero_embedding=zero_embedding,
        capture_routing=True,
        capture_hidden_states=capture_hidden_states,
        capture_final_hidden=True,
        return_logits=return_logits and not skip_lm_head,
        skip_lm_head=skip_lm_head,
        abort_rss_gb=abort_rss_gb, log=log)
    out = dict(topk_idx=res["routing"][0], topk_w=res["routing_weights"][0],
               n_exp=res["n_exp"], k=res["k"], L=res["L"], hidden=res["hidden"],
               ce=(res["ce_arms"][0] if "ce_arms" in res else float("nan")),
               layout="deepseek_v4", wall_s=res["wall_s"], rss_gb=res["rss_gb"])
    if return_logits and "logits_arms" in res and res["logits_arms"][0] is not None:
        out["logits"] = res["logits_arms"][0]
    if capture_hidden_states:
        out["hidden_states"] = res["hidden_states_arms"][0]
    if capture_final_hidden:
        out["final_hidden"] = res["final_hidden_arms"][0]
    return out
