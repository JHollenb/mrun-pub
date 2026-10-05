"""Parity gates for the DeepSeek-V4 moe-stream adapter on tiny random models.

The oracle is transformers' ``DeepseekV4ForCausalLM`` (5.13.1) — the SAME code the real
checkpoint loads through. A tiny config is built to exercise every architectural branch the
real model has: sliding + CSA + HCA layers, a sliding window SHORTER than the sequence, an
``index_topk`` SMALLER than the number of CSA windows (the learned-selection truncation the
real-scale gate cannot reach below T=2048), hash + noaux_tc routing, swiglu clamps that
actually clip, yarn compress rope, and hc_mult=4 hyper-connections.

Weights are saved in the NATIVE DeepSeek release namespace (``layers.N.attn.wq_a.weight`` …,
reversing transformers' conversion_mapping), because that is what the real checkpoint ships
and what the adapter reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
try:
    from transformers.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
except ImportError:  # pragma: no cover - old transformers
    pytest.skip("transformers lacks deepseek_v4", allow_module_level=True)

from mrun.engine.deepseek_v4_stream import dsv4_forward, dsv4_forward_arms, parse_dsv4_spec
from mrun.engine.moe_safetensors import PagedSafetensors, detect_layout, streamed_moe_forward

TINY = dict(
    vocab_size=97,
    hidden_size=64,
    moe_intermediate_size=16,
    num_hidden_layers=5,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=16,
    q_lora_rank=16,
    o_groups=2,
    o_lora_rank=8,
    sliding_window=6,
    index_n_heads=4,
    index_head_dim=8,
    index_topk=3,                      # < n_windows at T=40 -> selection truncation exercised
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    routed_scaling_factor=1.5,
    swiglu_limit=0.2,
    hc_mult=4,
    hc_sinkhorn_iters=5,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    compress_rope_theta=1000.0,
    max_position_embeddings=256,
    tie_word_embeddings=False,
    num_nextn_predict_layers=0,
)
# legacy-style fields, exactly how the real checkpoint's config.json spells them
TINY_LEGACY = dict(
    compress_ratios=[0, 0, 4, 128, 4, 0],   # 6 entries for 5 layers -> truncation exercised
    num_hash_layers=2,
    qk_rope_head_dim=8,
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
    rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32,
                  "beta_fast": 32, "beta_slow": 1},
)

# HF module-name suffix -> native suffix, applied inside a layer prefix.
_LAYER_RENAMES = [
    ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("attn_hc.fn", "hc_attn_fn"), ("attn_hc.base", "hc_attn_base"),
    ("attn_hc.scale", "hc_attn_scale"),
    ("ffn_hc.fn", "hc_ffn_fn"), ("ffn_hc.base", "hc_ffn_base"),
    ("ffn_hc.scale", "hc_ffn_scale"),
    ("self_attn.q_a_proj.weight", "attn.wq_a.weight"),
    ("self_attn.q_a_norm.weight", "attn.q_norm.weight"),
    ("self_attn.q_b_proj.weight", "attn.wq_b.weight"),
    ("self_attn.kv_proj.weight", "attn.wkv.weight"),
    ("self_attn.kv_norm.weight", "attn.kv_norm.weight"),
    ("self_attn.o_a_proj.weight", "attn.wo_a.weight"),
    ("self_attn.o_b_proj.weight", "attn.wo_b.weight"),
    ("self_attn.sinks", "attn.attn_sink"),
    ("self_attn.compressor.indexer.kv_proj.weight", "attn.indexer.compressor.wkv.weight"),
    ("self_attn.compressor.indexer.gate_proj.weight", "attn.indexer.compressor.wgate.weight"),
    ("self_attn.compressor.indexer.position_bias", "attn.indexer.compressor.ape"),
    ("self_attn.compressor.indexer.kv_norm.weight", "attn.indexer.compressor.norm.weight"),
    ("self_attn.compressor.indexer.q_b_proj.weight", "attn.indexer.wq_b.weight"),
    ("self_attn.compressor.indexer.scorer.weights_proj.weight",
     "attn.indexer.weights_proj.weight"),
    ("self_attn.compressor.kv_proj.weight", "attn.compressor.wkv.weight"),
    ("self_attn.compressor.gate_proj.weight", "attn.compressor.wgate.weight"),
    ("self_attn.compressor.position_bias", "attn.compressor.ape"),
    ("self_attn.compressor.kv_norm.weight", "attn.compressor.norm.weight"),
    ("mlp.gate.weight", "ffn.gate.weight"),
    ("mlp.gate.e_score_correction_bias", "ffn.gate.bias"),
    ("mlp.gate.tid2eid", "ffn.gate.tid2eid"),
    ("mlp.shared_experts.gate_proj.weight", "ffn.shared_experts.w1.weight"),
    ("mlp.shared_experts.up_proj.weight", "ffn.shared_experts.w2_up_placeholder"),
    ("mlp.shared_experts.down_proj.weight", "ffn.shared_experts.w2.weight"),
]
# fix: up_proj is w3 (mixtral naming), down_proj is w2
_LAYER_RENAMES = [(a, b.replace("w2_up_placeholder", "w3.weight")) for a, b in _LAYER_RENAMES]

_TOP_RENAMES = {
    "model.embed_tokens.weight": "embed.weight",
    "lm_head.weight": "head.weight",
    "model.norm.weight": "norm.weight",
    "model.hc_head.hc_fn": "hc_head_fn",
    "model.hc_head.hc_base": "hc_head_base",
    "model.hc_head.hc_scale": "hc_head_scale",
}


def _to_native(sd: dict) -> dict:
    out = {}
    for key, value in sd.items():
        if key in _TOP_RENAMES:
            out[_TOP_RENAMES[key]] = value.contiguous().clone()
            continue
        assert key.startswith("model.layers."), f"unmapped key {key}"
        rest = key[len("model."):]                       # layers.N.<suffix>
        layer_prefix = ".".join(rest.split(".")[:2])     # layers.N
        suffix = rest[len(layer_prefix) + 1:]
        if suffix == "mlp.experts.gate_up_proj":
            n_experts, two_i, _d = value.shape
            inter = two_i // 2
            for e in range(n_experts):
                out[f"{layer_prefix}.ffn.experts.{e}.w1.weight"] = value[e, :inter].contiguous().clone()
                out[f"{layer_prefix}.ffn.experts.{e}.w3.weight"] = value[e, inter:].contiguous().clone()
            continue
        if suffix == "mlp.experts.down_proj":
            for e in range(value.shape[0]):
                out[f"{layer_prefix}.ffn.experts.{e}.w2.weight"] = value[e].contiguous().clone()
            continue
        for hf_suffix, native_suffix in _LAYER_RENAMES:
            if suffix == hf_suffix:
                out[f"{layer_prefix}.{native_suffix}"] = value.contiguous().clone()
                break
        else:
            raise AssertionError(f"unmapped layer key {key}")
    return out


def _randomize(model: torch.nn.Module, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        with torch.no_grad():
            if "tid2eid" in name:
                p.copy_(torch.randint(0, TINY["n_routed_experts"],
                                      p.shape, generator=g, dtype=p.dtype))
            elif "inv_freq" in name:
                continue
            elif name.endswith((".scale", "hc_scale")):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
            elif "norm" in name and p.ndim == 1:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.15 * torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype))


def _build(tmp_path: Path, *, n_hash: int = 2, seed: int = 0):
    cfg_json = {"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4",
                **TINY, **TINY_LEGACY, "num_hash_layers": n_hash}
    (tmp_path / "config.json").write_text(json.dumps(cfg_json))
    config = DeepseekV4Config.from_pretrained(tmp_path)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = DeepseekV4ForCausalLM(config).float().eval()
    _randomize(model, seed=seed)
    from safetensors.torch import save_file
    save_file(_to_native(model.state_dict()), str(tmp_path / "model.safetensors"))
    ps = PagedSafetensors(tmp_path, dequant_dtype=torch.float32)
    return model, ps


def test_detect_layout_and_spec(tmp_path):
    _model, ps = _build(tmp_path)
    lay = detect_layout(ps)
    assert lay.name == "deepseek_v4"
    assert lay.embed_key == "embed.weight" and lay.head_key == "head.weight"
    spec = parse_dsv4_spec(ps.cfg)
    assert spec.layer_types == ("sliding", "sliding", "csa", "hca", "csa")
    assert spec.n_hash_layers == 2 and spec.hca_ratio == 8 and spec.csa_ratio == 4
    assert spec.rope_dim == 8 and spec.yarn_factor == 4.0


@pytest.mark.parametrize("T", [3, 11, 40])
def test_logits_parity_vs_transformers(tmp_path, T):
    """Full-stack fp32 logits vs the HF oracle at several lengths: T=3 (below one CSA window),
    T=11 (windows exist, indexer truncation off), T=40 (sliding window < T, HCA windows live,
    indexer top-3 < 10 windows -> learned selection truncates)."""
    model, ps = _build(tmp_path)
    ids = torch.randint(0, TINY["vocab_size"], (T,), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        ref = model(input_ids=ids[None]).logits[0].float()
    res = streamed_moe_forward(ps, ids, device="cpu", dtype=torch.float32, return_logits=True)
    got = res["logits"]
    diff = (got - ref).abs().max().item()
    assert diff <= 3e-5, f"T={T}: max logit diff {diff:.3e} > 3e-5"
    assert (got.argmax(-1) == ref.argmax(-1)).all(), "argmax disagreement vs HF oracle"
    assert res["layout"] == "deepseek_v4"
    assert np.isfinite(res["ce"])


def test_embed_subspace_tap_is_true_input_site(tmp_path):
    """The engine's embed_direction subspace removal must equal HF fed with the SAME edited
    inputs_embeds — i.e. under hyper-connections the tap really sits at the [T, d] embedding
    BEFORE stream expansion. Uses a hash-free config: HF cannot take inputs_embeds with hash
    routing (tid2eid needs input_ids)."""
    model, ps = _build(tmp_path, n_hash=0, seed=1)
    d = TINY["hidden_size"]
    g = torch.Generator().manual_seed(3)
    V, _ = torch.linalg.qr(torch.randn(d, 3, generator=g))
    ids = torch.randint(0, TINY["vocab_size"], (17,), generator=g)
    with torch.no_grad():
        emb = model.model.embed_tokens(ids[None]).float()
        edited = emb - (emb @ V) @ V.T
        ref = model(inputs_embeds=edited).logits[0].float()
    res = streamed_moe_forward(ps, ids, device="cpu", dtype=torch.float32,
                               return_logits=True, ablate_embed_direction=V)
    diff = (res["logits"] - ref).abs().max().item()
    assert diff <= 3e-5, f"embed-tap max logit diff {diff:.3e} > 3e-5"
    # and the tap actually changed something vs clean
    clean = streamed_moe_forward(ps, ids, device="cpu", dtype=torch.float32,
                                 return_logits=True)
    assert (res["logits"] - clean["logits"]).abs().max().item() > 1e-3


def test_zero_embedding_breaks_model(tmp_path):
    model, ps = _build(tmp_path)
    ids = torch.randint(0, TINY["vocab_size"], (12,), generator=torch.Generator().manual_seed(9))
    clean = streamed_moe_forward(ps, ids, device="cpu", dtype=torch.float32, return_logits=True)
    zeroed = streamed_moe_forward(ps, ids, device="cpu", dtype=torch.float32,
                                  return_logits=True, zero_embedding=True)
    assert (clean["logits"] - zeroed["logits"]).abs().max().item() > 1e-3
    # zeroed rows are position-independent up to attention: every position's logits identical?
    # (not exactly true because of position encodings; just require a large break)


def test_batch_matches_serial_and_component_ablation(tmp_path):
    _model, ps = _build(tmp_path)
    g = torch.Generator().manual_seed(11)
    rows = [torch.randint(0, TINY["vocab_size"], (n,), generator=g) for n in (9, 23)]
    serial = [dsv4_forward(ps, r, device="cpu", dtype=torch.float32,
                           capture_final_hidden=True, skip_lm_head=True) for r in rows]
    batch = dsv4_forward_arms(ps, rows, [(0, None), (1, None), (0, (2, "attn")), (1, (4, "mlp"))],
                              device="cpu", dtype=torch.float32, capture_routing=True)
    for j in range(2):
        assert torch.equal(batch["final_last"][j], serial[j]["final_hidden"][-1]), \
            "batched arm diverges from serial forward"
    # component-ablated arms must differ from their clean twins
    assert (batch["final_last"][2] - batch["final_last"][0]).abs().max() > 1e-4
    assert (batch["final_last"][3] - batch["final_last"][1]).abs().max() > 1e-4
    # routing trace shape/sentinel conventions
    assert batch["routing"][0].shape == (TINY["num_hidden_layers"], 9, TINY["num_experts_per_tok"])
    assert (batch["routing"][0] >= 0).all()   # every layer is MoE in V4


def _stream_engine(tmp_path, **build_kw):
    """Real MoEStreamEngine over the tiny native checkpoint; __init__ bypassed only to skip
    registry/tokenizer resolution (same pattern as test_moe_stream._stream_engine_at)."""
    from mrun.engine.moe_stream import MoEStreamEngine

    model, ps = _build(tmp_path, **build_kw)
    engine = object.__new__(MoEStreamEngine)
    engine.store = ps
    engine.layout = detect_layout(ps)
    engine.cfg = ps.cfg
    engine.device, engine.dtype = "cpu", torch.float32
    engine.abort_rss_gb, engine.log = 48.0, None
    engine.n_layer, engine.hidden = TINY["num_hidden_layers"], TINY["hidden_size"]
    engine.init_taps()
    return engine, model


def test_engine_surface_candidate_head_and_ablation(tmp_path):
    """The exact surface the hub-physiology runner rides: untied native head keys,
    candidate_logits_batch, hidden_states probe (post-tap embedding at index 0), and the
    ablation context restoring state."""
    engine, model = _stream_engine(tmp_path)
    assert engine.tied_lm_head is False
    assert engine._head_key() == "head.weight"

    rows = [np.arange(1, 9), np.arange(4, 20)]
    cands = ((5, 11, 23), (2, 9))
    clean = engine.candidate_logits_batch(rows, cands)
    assert [tuple(c.shape) for c in clean] == [(3,), (2,)]
    # candidate scores must equal full-logits gather (same forward, subset head)
    full = engine.logits(rows[0])
    torch.testing.assert_close(
        clean[0], full[-1, torch.as_tensor(cands[0])].float(), atol=3e-5, rtol=1e-5)

    d = TINY["hidden_size"]
    V, _ = torch.linalg.qr(torch.randn(d, 4, generator=torch.Generator().manual_seed(5)))
    probe = np.arange(2, 14)
    h0_clean = engine.hidden_states(probe)[0]
    with engine.ablation(embed_direction=V, lm_head=False):
        h0 = engine.hidden_states(probe)[0]
        ablated = engine.candidate_logits_batch(rows, cands)
    frac_before = float(((h0_clean.float() @ V) ** 2).sum() / (h0_clean.float() ** 2).sum())
    frac_after = float(((h0.float() @ V) ** 2).sum() / (h0.float() ** 2).sum())
    assert frac_before > 1e-3 and frac_after < 1e-6
    assert any((a - c).abs().max() > 1e-5 for a, c in zip(ablated, clean, strict=True))
    # context restored: scores match clean again
    again = engine.candidate_logits_batch(rows, cands)
    for a, c in zip(again, clean, strict=True):
        torch.testing.assert_close(a, c)
    with engine.ablation(zero_embedding=True):
        z0 = engine.hidden_states(probe)[0]
    assert float(z0.abs().max()) == 0.0


def test_hidden_states_contract_len_and_shapes(tmp_path):
    """Pin the hidden_states contract the parity gate and any per-layer consumer rides.

    Regression: the real-checkpoint parity gate assumed entries 1..L were ALL stacks and
    indexed hs[1+i] for every oracle layer; the last index is the post-final-norm collapsed
    [T,d] state, which broadcast against the oracle's [T,hc,d] and blew up only after ~25 min
    of streaming. Length + per-entry shapes are asserted here so it fails in 4 s instead.
    """
    _model, ps = _build(tmp_path)
    T, L, d, hc = 7, TINY["num_hidden_layers"], TINY["hidden_size"], TINY["hc_mult"]
    res = dsv4_forward(ps, torch.arange(1, T + 1), device="cpu", dtype=torch.float32,
                       skip_lm_head=True, capture_hidden_states=True)
    hs = res["hidden_states"]
    assert len(hs) == L + 1, f"expected L+1={L + 1} entries (HF convention), got {len(hs)}"
    assert tuple(hs[0].shape) == (T, d)                     # post-tap embedding
    for i in range(1, L):                                   # stacks after layers 0..L-2
        assert tuple(hs[i].shape) == (T, hc, d), (i, tuple(hs[i].shape))
    assert tuple(hs[-1].shape) == (T, d)                    # post-final-norm collapsed
    # the number of comparable layer stacks is exactly L-1
    assert len([h for h in hs if h.dim() == 3]) == L - 1


def test_hash_routing_uses_tid2eid(tmp_path):
    _model, ps = _build(tmp_path)
    ids = torch.randint(0, TINY["vocab_size"], (8,), generator=torch.Generator().manual_seed(13))
    res = dsv4_forward(ps, ids, device="cpu", dtype=torch.float32, skip_lm_head=True)
    for layer in range(2):   # each hash layer owns its own frozen table
        tid2eid = ps.get(f"layers.{layer}.ffn.gate.tid2eid")
        assert (res["topk_idx"][layer] == tid2eid[ids].numpy().astype(np.int16)).all()
    # non-hash layers may differ from the table
    assert res["topk_idx"].shape[0] == TINY["num_hidden_layers"]
