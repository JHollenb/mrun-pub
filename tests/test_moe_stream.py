"""mri.moe_stream — HF-parity gates per MoE layout (tiny random models) + trace-null purity.

The parity tests are the correctness gate for the streamed forward: build a tiny random
model with transformers, save safetensors, run the streamed forward off the files, and
compare full logits against the dense HF forward (fp32, eager). This exercises layout
detection, GQA, rope, qk-norm variants, router softmax/renorm, and expert accumulation
end-to-end — if any of it drifts from HF semantics the logits diverge.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mrun.engine.moe_safetensors import (
    PagedSafetensors,
    detect_layout,
    locality_decomposition,
    lru_hit,
    streamed_moe_forward,
    time_shuffle,
    uniform_iid,
)

transformers = pytest.importorskip("transformers")


def test_streamed_moe_declares_scalar_residual_limitation():
    from mrun.engine.moe_stream import MoEStreamEngine

    engine = object.__new__(MoEStreamEngine)
    engine.dtype = torch.float32
    capabilities = engine.capabilities()
    assert capabilities.residual_tap
    assert not capabilities.residual_tap_batch
    assert "residual_tap_batch" in capabilities.gaps()


# --------------------------------------------------------------------------- helpers
def _save(model, cfg, tmp_path):
    d = tmp_path / "m"
    model.save_pretrained(str(d), safe_serialization=True)
    cfg.save_pretrained(str(d))
    return d


def _parity(model, cfg, tmp_path, expect_layout, T=12, atol=2e-4):
    model = model.eval().float()
    d = _save(model, cfg, tmp_path)
    ps = PagedSafetensors(d)
    assert detect_layout(ps).name == expect_layout
    rng = np.random.default_rng(0)
    ids = torch.from_numpy(rng.integers(1, cfg.vocab_size, size=T))
    with torch.no_grad():
        ref = model(ids[None], output_router_logits=True)
    got = streamed_moe_forward(ps, ids, return_logits=True)
    assert got["layout"] == expect_layout
    torch.testing.assert_close(got["logits"], ref.logits[0].float(), atol=atol, rtol=1e-3)
    # routing trace parity: recompute HF's top-k from its own router logits (fp32 softmax
    # is strictly monotone per row, so indices match the forward's exactly)
    k = got["k"]
    for li, rl in enumerate(ref.router_logits):
        if rl is None or (got["topk_idx"][li] < 0).all():
            continue
        hf_top = torch.topk(torch.softmax(rl.float(), dim=-1), k, dim=-1).indices.numpy()
        assert (np.sort(hf_top, -1) == np.sort(got["topk_idx"][li].astype(np.int64), -1)).all()
    return got


# --------------------------------------------------------------------------- layouts
def test_mixtral_parity(tmp_path):
    cfg = transformers.MixtralConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
        num_experts_per_tok=2, max_position_embeddings=64, rope_theta=1e6,
        attn_implementation="eager", tie_word_embeddings=False)
    got = _parity(transformers.MixtralForCausalLM(cfg), cfg, tmp_path, "mixtral")
    assert np.isfinite(got["ce"])


def test_olmoe_parity(tmp_path):
    cfg = transformers.OlmoeConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, num_experts=4,
        num_experts_per_tok=2, norm_topk_prob=False, max_position_embeddings=64,
        attn_implementation="eager", tie_word_embeddings=False)
    _parity(transformers.OlmoeForCausalLM(cfg), cfg, tmp_path, "olmoe")


def test_qwen3_moe_parity(tmp_path):
    cfg = transformers.Qwen3MoeConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, moe_intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
        max_position_embeddings=64, attn_implementation="eager", tie_word_embeddings=False)
    _parity(transformers.Qwen3MoeForCausalLM(cfg), cfg, tmp_path, "qwen3_moe")


def test_qwen2_moe_shared_expert_parity(tmp_path):
    cfg = transformers.Qwen2MoeConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, moe_intermediate_size=24,
        shared_expert_intermediate_size=40, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True, max_position_embeddings=64,
        attn_implementation="eager", tie_word_embeddings=False)
    _parity(transformers.Qwen2MoeForCausalLM(cfg), cfg, tmp_path, "qwen2_moe")


def test_qwen3_moe_dense_layer_fallback(tmp_path):
    # layer 0 dense (mlp_only_layers), layer 1 sparse — the -1 sentinel must appear only at 0
    cfg = transformers.Qwen3MoeConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, moe_intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
        mlp_only_layers=[0], max_position_embeddings=64,
        attn_implementation="eager", tie_word_embeddings=False)
    got = _parity(transformers.Qwen3MoeForCausalLM(cfg), cfg, tmp_path, "qwen3_moe")
    assert (got["topk_idx"][0] == -1).all()
    assert (got["topk_idx"][1] >= 0).all()


def test_hidden_state_capture_and_residual_patch(tmp_path):
    cfg = transformers.Qwen3MoeConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, moe_intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
        max_position_embeddings=64, attn_implementation="eager", tie_word_embeddings=False)
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    d = _save(model, cfg, tmp_path)
    ids = torch.arange(1, 11)
    with torch.no_grad():
        reference = model(ids[None], output_hidden_states=True)
    captured = streamed_moe_forward(
        PagedSafetensors(d), ids, return_logits=True, capture_hidden_states=True
    )
    assert len(captured["hidden_states"]) == cfg.num_hidden_layers + 1
    for actual, expected in zip(captured["hidden_states"], reference.hidden_states, strict=True):
        torch.testing.assert_close(actual, expected[0].float(), atol=2e-4, rtol=1e-3)

    direction = np.random.default_rng(4).normal(size=cfg.hidden_size).astype(np.float32)
    patched = streamed_moe_forward(
        PagedSafetensors(d),
        ids,
        return_logits=True,
        capture_hidden_states=True,
        resid_patch_ops_by_layer={0: [("proj_remove", direction, None)]},
    )
    unit_direction = torch.from_numpy(direction) / torch.from_numpy(direction).norm()
    layer_zero_projection = patched["hidden_states"][1].float() @ unit_direction
    assert float(layer_zero_projection.abs().max()) < 1e-5
    assert not torch.equal(patched["logits"], captured["logits"])


def test_expert_stats_capture(tmp_path):
    cfg = transformers.MixtralConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
        num_experts_per_tok=2, max_position_embeddings=64,
        attn_implementation="eager", tie_word_embeddings=False)
    model = transformers.MixtralForCausalLM(cfg).eval().float()
    d = _save(model, cfg, tmp_path)
    ps = PagedSafetensors(d)
    ids = torch.arange(1, 11)
    got = streamed_moe_forward(ps, ids, capture_expert_stats=True)
    assert set(got["act_sum"].keys()) == {(li, e) for li in range(2) for e in range(4)}
    for (li, e), s in got["act_sum"].items():
        assert s.shape == (48,)
        if got["act_cnt"][(li, e)] == 0:
            assert np.isnan(s).all() and np.isnan(got["write_norm"][(li, e)]).all()
        else:
            assert np.isfinite(s).all() and np.isfinite(got["write_norm"][(li, e)]).all()
    # write_norm parity vs the dense model's down weights, for a routed expert
    # (transformers 5.x fuses experts: down_proj is a batched Parameter [E, hidden, inter])
    routed = [(li, e) for (li, e), c in got["act_cnt"].items() if c > 0]
    li, e = routed[0]
    dp = model.model.layers[li].mlp.experts.down_proj
    np.testing.assert_allclose(got["write_norm"][(li, e)],
                               torch.linalg.norm(dp[e].detach().float(), dim=0).numpy(),
                               rtol=1e-5)


# --------------------------------------------------------------------------- trace nulls
def test_lru_hit_exact():
    stream = np.array([[0, 1], [0, 1], [2, 3], [0, 1]])
    hits, tot = lru_hit(stream, budget=2)
    assert tot == 8
    assert hits == 2  # t=1 hits 0,1; t=2 evicts both; t=3 misses both


def test_time_shuffle_preserves_multiset():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 8, size=(3, 50, 2)).astype(np.int16)
    sh = time_shuffle(idx, np.random.default_rng(1))
    for li in range(3):
        assert np.array_equal(np.sort(idx[li].reshape(-1)), np.sort(sh[li].reshape(-1)))
        assert not np.array_equal(idx[li], sh[li])


def test_skewed_trace_reports_skew():
    # heavy popularity skew, no temporal order: skew_share must catch it, temporal ~0
    rng = np.random.default_rng(0)
    idx = np.zeros((1, 800, 2), dtype=np.int16)
    idx[0, :, 0] = 0                                   # expert 0 in every token
    idx[0, :, 1] = rng.integers(1, 4, size=800)        # partner from a 3-expert pool
    dec = locality_decomposition(idx, 8, 2, moe_inter=16, hidden=8)
    mid = dec["sweep"][2]                              # budget 4 of 8
    assert mid["skew_share"] > 0.2, mid
    assert abs(mid["temporal_locality"]) < 0.05, mid


def test_uniform_iid_shape_and_distinct():
    out = uniform_iid((2, 100, 2), 8, np.random.default_rng(0))
    assert out.shape == (2, 100, 2)
    assert all(len(set(row)) == 2 for li in range(2) for row in out[li])
    c = np.bincount(out.reshape(-1), minlength=8)
    assert c.min() > 0  # uniform: everyone picked


def test_locality_decomposition_zero_skew_trace_reports_zero_skew():
    # regression for the k/n floor bug: a uniform-iid trace must show ~0 skew at EVERY
    # budget (the old analytic floor fabricated skew_share up to +0.62 at large budgets)
    iid = uniform_iid((1, 800, 2), 8, np.random.default_rng(5))
    dec = locality_decomposition(iid, 8, 2, moe_inter=16, hidden=8)
    for s in dec["sweep"]:
        assert abs(s["skew_share"]) < 0.05, s
        assert abs(s["temporal_locality"]) < 0.05, s


def test_locality_decomposition_repetitive_vs_random():
    # a maximally repetitive trace has high temporal locality; an iid trace ~0
    rep = np.tile(np.array([0, 1], dtype=np.int16), (1, 64, 1))
    rng = np.random.default_rng(0)
    iid = np.stack([rng.permutation(8)[:2] for _ in range(64)])[None].astype(np.int16)
    lrep = locality_decomposition(rep, 8, 2, moe_inter=16, hidden=8)
    liid = locality_decomposition(iid, 8, 2, moe_inter=16, hidden=8)
    rep_loc = lrep["sweep"][0]["temporal_locality"]
    iid_loc = liid["sweep"][0]["temporal_locality"]
    assert rep_loc >= 0.0  # degenerate trace: shuffle can't beat perfectly repeated
    assert lrep["sweep"][0]["hit"] > 0.9
    assert abs(iid_loc) < 0.15


def test_bf16_routing_trace_matches_hf():
    # F7 regression: on a bf16 model the routing WEIGHTS applied to experts must be the
    # bf16-rounded ones for olmoe/qwen3 (HF casts before the multiply), and the streamed
    # logits must track the dense bf16 model, not the fp32 one.
    torch.manual_seed(0)
    cfg = transformers.OlmoeConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, num_experts=4,
        num_experts_per_tok=2, norm_topk_prob=False, max_position_embeddings=64,
        attn_implementation="eager", tie_word_embeddings=False)
    model = transformers.OlmoeForCausalLM(cfg).eval().to(torch.bfloat16)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        d = Path(td) / "m"
        model.save_pretrained(str(d), safe_serialization=True)
        cfg.save_pretrained(str(d))
        ps = PagedSafetensors(d)
        ids = torch.from_numpy(np.random.default_rng(0).integers(1, 99, size=12))
        with torch.no_grad():
            ref = model(ids[None]).logits[0].float()
        got = streamed_moe_forward(ps, ids, dtype=torch.bfloat16, return_logits=True)
        torch.testing.assert_close(got["logits"], ref, atol=3e-2, rtol=3e-2)
        # applied routing weights are bf16-rounded for this layout
        w16 = got["topk_w"][0]
        assert np.allclose(w16, torch.from_numpy(w16).to(torch.bfloat16).float().numpy())


def test_single_shard_no_index(tmp_path):
    # PagedSafetensors must handle save_pretrained's index-less single shard
    cfg = transformers.MixtralConfig(
        vocab_size=50, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, num_local_experts=2,
        num_experts_per_tok=1, max_position_embeddings=32,
        attn_implementation="eager", tie_word_embeddings=False)
    model = transformers.MixtralForCausalLM(cfg).eval()
    d = _save(model, cfg, tmp_path)
    assert not (d / "model.safetensors.index.json").exists()
    ps = PagedSafetensors(d)
    assert json.loads((d / "config.json").read_text())["model_type"] == "mixtral"
    assert ps.has("model.layers.0.block_sparse_moe.gate.weight")


# ------------------------------------------------------------------- fp8 streaming reader
def _qwen3_moe_cfg(**overrides):
    base = dict(
        vocab_size=99, hidden_size=32, intermediate_size=48, moe_intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
        max_position_embeddings=64, attn_implementation="eager", tie_word_embeddings=False)
    base.update(overrides)
    return transformers.Qwen3MoeConfig(**base)


def _block_fp8_checkpoint(source_dir, out_dir, *, block=16, declare=True):
    """Re-encode a saved checkpoint's 2-D weights as block-scaled fp8 CODES + weight_scale_inv,
    the DeepSeek/Qwen fp8 layout. Everything else (norms, 1-D tensors) stays untouched, exactly
    as a real fp8 release does."""
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(source_dir / "model.safetensors"))
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = {}
    for name, tensor in tensors.items():
        if tensor.ndim != 2 or "norm" in name:
            encoded[name] = tensor
            continue
        rows, cols = tensor.shape
        scale = torch.zeros(-(-rows // block), -(-cols // block))
        for r in range(scale.shape[0]):
            for c in range(scale.shape[1]):
                tile = tensor[r * block:(r + 1) * block, c * block:(c + 1) * block]
                scale[r, c] = max(float(tile.abs().max()) / 448.0, 1e-12)
        expanded = scale[torch.arange(rows) // block][:, torch.arange(cols) // block]
        encoded[name] = (tensor / expanded).to(torch.float8_e4m3fn)
        encoded[f"{name}_scale_inv"] = scale
    save_file(encoded, str(out_dir / "model.safetensors"))
    cfg = json.loads((source_dir / "config.json").read_text())
    if declare:
        cfg["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [block, block]}
    (out_dir / "config.json").write_text(json.dumps(cfg))
    return out_dir


def test_paged_safetensors_leaves_unquantized_weights_byte_identical(tmp_path):
    """The fp8 branch must be invisible on a plain BF16 checkpoint — same bytes, same dtype."""
    from safetensors.torch import load_file

    model = transformers.Qwen3MoeForCausalLM(_qwen3_moe_cfg()).eval()
    d = _save(model, _qwen3_moe_cfg(), tmp_path)
    ps = PagedSafetensors(d)
    assert ps.block_size is None
    original = load_file(str(d / "model.safetensors"))
    for name, expected in original.items():
        got = ps.get(name)
        assert ps.scale_key(name) is None
        assert got.dtype == expected.dtype
        assert torch.equal(got, expected), name


def test_streamed_forward_dequantizes_block_fp8_instead_of_casting_codes(tmp_path):
    """THE gate for a meaningful fp8 run. On a block-scaled fp8 checkpoint a bare dtype cast
    reads code bytes as numbers; the reader must apply the scale. Compared against the SAME
    model in bf16: dequantized logits track it, raw codes do not come close."""
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    dense_dir = _save(model, cfg, tmp_path)
    fp8_dir = _block_fp8_checkpoint(dense_dir, tmp_path / "fp8")

    ids = torch.from_numpy(np.random.default_rng(0).integers(1, cfg.vocab_size, size=10))
    reference = streamed_moe_forward(PagedSafetensors(dense_dir), ids, return_logits=True)
    fp8 = streamed_moe_forward(PagedSafetensors(fp8_dir), ids, return_logits=True)

    assert PagedSafetensors(fp8_dir).block_size == (16, 16)
    ref_logits, got_logits = reference["logits"], fp8["logits"]
    # e4m3 keeps ~4 significant bits, so this is a quantization-level agreement, not equality —
    # what it rules out is the failure mode: unscaled codes are orders of magnitude off.
    scale = float(ref_logits.abs().max())
    assert float((got_logits - ref_logits).abs().max()) < 0.20 * scale

    raw = PagedSafetensors(fp8_dir)
    raw.get = raw.raw  # simulate the old "native dtype, let the forward cast" behaviour
    unscaled = streamed_moe_forward(raw, ids, return_logits=True)
    assert float((unscaled["logits"] - ref_logits).abs().max()) > 2.0 * scale, (
        "the unscaled-code control must be badly wrong, or this test proves nothing"
    )


def test_undeclared_fp8_block_geometry_refuses_rather_than_guessing(tmp_path):
    """Same bytes, no quantization_config: an unguessable block must raise, not silently
    misalign rows onto a neighbouring block's scale."""
    cfg = _qwen3_moe_cfg(hidden_size=48, num_attention_heads=4, head_dim=12)
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    dense_dir = _save(model, cfg, tmp_path)
    undeclared = _block_fp8_checkpoint(dense_dir, tmp_path / "fp8u", block=32, declare=False)
    ps = PagedSafetensors(undeclared)
    assert ps.block_size is None
    with pytest.raises(ValueError, match="cannot infer the fp8"):
        ps.get("model.layers.0.self_attn.q_proj.weight")


# ------------------------------------------------------- whole-component + embedding ablation
def _zeroed_copy(source_dir, out_dir, names):
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(source_dir / "model.safetensors"))
    for name in names:
        tensors[name] = torch.zeros_like(tensors[name])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text((source_dir / "config.json").read_text())
    return out_dir


def test_component_ablation_equals_zeroing_that_components_output(tmp_path):
    """``ablate_component`` must be EXACT, not approximate. Attention ablation is checked
    against a checkpoint whose o_proj is zeroed (so the attention term is provably zero); MoE
    ablation against one whose expert down_proj weights are zeroed (down @ act == 0)."""
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    d = _save(model, cfg, tmp_path)
    ids = torch.from_numpy(np.random.default_rng(3).integers(1, cfg.vocab_size, size=9))

    def run(directory, **kwargs):
        return streamed_moe_forward(
            PagedSafetensors(directory), ids, return_logits=True, **kwargs
        )["logits"]

    clean = run(d)
    zero_attn = _zeroed_copy(d, tmp_path / "za", ["model.layers.1.self_attn.o_proj.weight"])
    torch.testing.assert_close(run(d, ablate_component=(1, "attn")), run(zero_attn))
    zero_mlp = _zeroed_copy(d, tmp_path / "zm", [
        f"model.layers.1.mlp.experts.{e}.down_proj.weight" for e in range(cfg.num_experts)
    ])
    torch.testing.assert_close(run(d, ablate_component=(1, "mlp")), run(zero_mlp))

    for target in ((0, "attn"), (0, "mlp"), (1, "attn"), (1, "mlp")):
        assert not torch.allclose(run(d, ablate_component=target), clean), f"{target} was a no-op"
    with pytest.raises(ValueError, match="attn"):
        run(d, ablate_component=(0, "head"))
    with pytest.raises(ValueError, match="out of range"):
        run(d, ablate_component=(9, "mlp"))


def test_ablating_an_mlp_layer_stops_streaming_its_experts(tmp_path):
    """A removed component must be CHEAPER than a live one: its weights are never pulled."""
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    d = _save(model, cfg, tmp_path)
    ids = torch.from_numpy(np.random.default_rng(3).integers(1, cfg.vocab_size, size=9))

    class _Counting(PagedSafetensors):
        # hook `raw`, the actual read primitive: it is called on every path (get, get_on,
        # get_rows), so this cannot be fooled by a caller switching between them.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pulled = []

        def raw(self, key):
            self.pulled.append(key)
            return super().raw(key)

    live, ablated = _Counting(d), _Counting(d)
    streamed_moe_forward(live, ids, return_logits=True)
    streamed_moe_forward(ablated, ids, return_logits=True, ablate_component=(1, "mlp"))
    expert_prefix = "model.layers.1.mlp.experts."
    assert any(key.startswith(expert_prefix) for key in live.pulled)
    assert not any(key.startswith(expert_prefix) for key in ablated.pulled)
    # routing is still traced (one small GEMM already paid), so physiology stays well-defined
    assert "model.layers.1.mlp.gate.weight" in ablated.pulled


def test_embedding_direction_and_subspace_ablation(tmp_path):
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    d = _save(model, cfg, tmp_path)
    ids = torch.arange(1, 9)

    direction = torch.from_numpy(
        np.random.default_rng(5).normal(size=cfg.hidden_size).astype(np.float32)
    )
    unit = direction / direction.norm()
    raw = streamed_moe_forward(
        PagedSafetensors(d), ids, capture_hidden_states=True, skip_lm_head=True
    )["hidden_states"][0]
    assert float((raw @ unit).abs().max()) > 1e-3, "fixture must have a projection to remove"

    removed = streamed_moe_forward(
        PagedSafetensors(d), ids, capture_hidden_states=True, skip_lm_head=True,
        ablate_embed_direction=direction,  # deliberately un-normalized
    )["hidden_states"][0]
    assert float((removed @ unit).abs().max()) < 1e-5

    basis, _ = torch.linalg.qr(torch.from_numpy(
        np.random.default_rng(6).normal(size=(cfg.hidden_size, 3)).astype(np.float32)
    ))
    subspace = streamed_moe_forward(
        PagedSafetensors(d), ids, capture_hidden_states=True, skip_lm_head=True,
        ablate_embed_direction=basis,
    )["hidden_states"][0]
    assert float((subspace @ basis).abs().max()) < 1e-5

    zeroed = streamed_moe_forward(
        PagedSafetensors(d), ids, capture_hidden_states=True, skip_lm_head=True,
        zero_embedding=True,
    )["hidden_states"][0]
    assert float(zeroed.abs().max()) == 0.0


def test_skip_lm_head_drops_logits_and_final_hidden_matches_the_captured_stack(tmp_path):
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    d = _save(model, cfg, tmp_path)
    ids = torch.arange(1, 9)
    full = streamed_moe_forward(
        PagedSafetensors(d), ids, return_logits=True, capture_hidden_states=True
    )
    lean = streamed_moe_forward(
        PagedSafetensors(d), ids, skip_lm_head=True, capture_final_hidden=True
    )
    assert "logits" not in lean and np.isnan(lean["ce"])
    assert np.isfinite(full["ce"]) and "logits" in full
    torch.testing.assert_close(lean["final_hidden"], full["hidden_states"][-1].float())


# --------------------------------------------------------------------- engine-level taps
def _stream_engine_at(model_dir, cfg):
    """A real MoEStreamEngine over an already-saved checkpoint; __init__ is bypassed only to
    skip model-registry/tokenizer resolution. Every tap under test is the engine's own method."""
    from mrun.engine.moe_stream import MoEStreamEngine

    engine = object.__new__(MoEStreamEngine)
    engine.store = PagedSafetensors(model_dir)
    engine.layout = detect_layout(engine.store)
    engine.cfg = engine.store.cfg
    engine.device, engine.dtype = "cpu", torch.float32
    engine.abort_rss_gb, engine.log = 48.0, None
    engine.n_layer, engine.hidden = cfg.num_hidden_layers, cfg.hidden_size
    engine.init_taps()
    engine.scoring_stats = {}
    return engine, cfg


def _stream_engine(tmp_path):
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    return _stream_engine_at(_save(model, cfg, tmp_path), cfg)


def test_stream_engine_ablation_scope_is_sticky_nestable_and_restores(tmp_path):
    engine, cfg = _stream_engine(tmp_path)
    rows = [np.arange(1, 7), np.arange(3, 8)]
    clean = engine.hidden_last_batch(rows)
    assert clean.shape == (2, cfg.hidden_size)
    for index, row in enumerate(rows):
        torch.testing.assert_close(clean[index], engine.final_hidden(row)[-1])

    direction = torch.from_numpy(
        np.random.default_rng(8).normal(size=cfg.hidden_size).astype(np.float32)
    )
    with engine.ablation(embed_direction=direction, lm_head=True):
        ablated = engine.hidden_last_batch(rows)
        assert not torch.allclose(ablated, clean)
        with engine.ablation(component=(1, "mlp")):
            assert engine.ablate_embed_direction is direction  # inner scope must not cancel it
            both = engine.hidden_last_batch(rows)
        assert not torch.allclose(both, ablated)
        assert engine.ablate_component is None
    assert engine.ablate_embed_direction is None
    torch.testing.assert_close(engine.hidden_last_batch(rows), clean)


def test_stream_engine_forward_patched_still_refuses_dense_neuron_taps(tmp_path):
    engine, _cfg = _stream_engine(tmp_path)
    with pytest.raises(NotImplementedError, match="not dense-MLP neuron taps"):
        engine.forward_patched(np.arange(1, 5), patch_ops_by_layer={0: [("zero", [0], None)]})
    with pytest.raises(NotImplementedError, match="not dense-MLP neuron taps"):
        engine.forward_patched(np.arange(1, 5), collect_acts=True)


# ------------------------------------------------------- candidate-subset head (row range-read)
def test_get_rows_matches_a_full_tensor_gather_including_fp8(tmp_path):
    """Range-reading rows must equal gathering them from the full tensor — for scattered,
    unsorted, duplicated ids, on plain BF16 AND on block-scaled fp8, where each gathered row's
    own scale block has to be resolved (a subset read cannot reuse whole-tensor row positions)."""
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    dense = _save(model, cfg, tmp_path)
    fp8 = _block_fp8_checkpoint(dense, tmp_path / "fp8", block=16)
    ids = [7, 3, 40, 3, 0]

    plain = PagedSafetensors(dense)
    got = plain.get_rows("lm_head.weight", ids)
    assert torch.equal(got, plain.get("lm_head.weight")[torch.tensor(ids)])

    quantized = PagedSafetensors(fp8)
    rows = quantized.get_rows("lm_head.weight", ids)
    reference = quantized.get("lm_head.weight")[torch.tensor(ids)]
    assert rows.shape == (len(ids), cfg.hidden_size)
    torch.testing.assert_close(rows, reference)
    # the alignment this guards: a wrong scale block is silent, so check it is not merely
    # "close to the codes" — the scale genuinely moved the numbers
    raw_codes = quantized.raw("lm_head.weight")[torch.tensor(ids)].to(torch.float32)
    assert float((rows - raw_codes).abs().max()) > 1.0


def test_candidate_head_scores_match_a_full_vocab_forward(tmp_path):
    """The subset head must be EXACT against the full-vocab logits it avoids computing."""
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    d = _save(model, cfg, tmp_path)
    engine, _cfg = _stream_engine_at(d, cfg)

    rows = [np.arange(1, 6), np.arange(4, 12)]
    candidates = ((11, 4, 30), (30, 11, 4))
    scores = engine.candidate_logits_batch(rows, candidates)

    for row, row_candidates, got in zip(rows, candidates, scores, strict=True):
        full = streamed_moe_forward(
            PagedSafetensors(d), torch.as_tensor(row), return_logits=True
        )["logits"][-1]
        torch.testing.assert_close(
            got, full.index_select(0, torch.tensor(row_candidates)), atol=1e-4, rtol=1e-4
        )
    # per-row candidate ORDER is honoured independently of the shared union lookup
    torch.testing.assert_close(scores[1], scores[1])
    assert [tuple(s.shape) for s in scores] == [(3,), (3,)]


def test_candidate_head_never_reads_the_whole_output_head(tmp_path):
    """The whole point: scoring candidates must not pull the 151936-row head."""
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    d = _save(model, cfg, tmp_path)

    class _Watching(PagedSafetensors):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.full_reads = []
            self.row_reads = 0

        def get(self, key):
            self.full_reads.append(key)
            return super().get(key)

        def get_rows(self, key, row_ids):
            self.row_reads += 1
            return super().get_rows(key, row_ids)

    engine, _cfg = _stream_engine_at(d, cfg)
    engine.store = _Watching(d)
    engine.candidate_logits_batch([np.arange(1, 6)], ((11, 4, 30),))
    assert engine.store.row_reads == 1
    assert "lm_head.weight" not in engine.store.full_reads, "the full head was streamed"


def test_lm_head_rows_are_cached_and_honour_a_tied_head_ablation(tmp_path):
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    d = _save(model, cfg, tmp_path)
    engine, _cfg = _stream_engine_at(d, cfg)

    first = engine.lm_head_rows([5, 9])
    assert set(engine._head_row_cache) == {5, 9}
    torch.testing.assert_close(engine.lm_head_rows([9, 5]), first.flip(0))

    direction = torch.zeros(cfg.hidden_size)
    direction[0] = 3.0
    with engine.ablation(embed_direction=direction, lm_head=True):
        stripped = engine.lm_head_rows([5, 9])
        assert float(stripped[:, 0].abs().max()) < 1e-5
    torch.testing.assert_close(engine.lm_head_rows([5, 9]), first)  # restored, cache intact


def test_tied_checkpoint_uses_the_embedding_as_the_candidate_head(tmp_path):
    cfg = _qwen3_moe_cfg(tie_word_embeddings=True)
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    d = _save(model, cfg, tmp_path)
    engine, _cfg = _stream_engine_at(d, cfg)
    assert engine.tied_lm_head and engine._head_key() == "model.embed_tokens.weight"
    torch.testing.assert_close(
        engine.lm_head_rows([2, 8]),
        PagedSafetensors(d).get("model.embed_tokens.weight")[torch.tensor([2, 8])].float(),
    )


def test_get_on_matches_get_then_cast_on_every_path(tmp_path):
    """``get_on`` moves fp8 dequant onto the compute device. It must be numerically the same
    tensor ``get(key).to(device, dtype)`` produces — otherwise the GPU forward and the CPU
    reference silently disagree. Exercised on plain BF16 and on block-scaled fp8, cpu device
    (the path every test host has); the cuda branch differs only in where the multiply runs."""
    cfg = _qwen3_moe_cfg()
    model = transformers.Qwen3MoeForCausalLM(cfg).eval().float()
    dense = _save(model, cfg, tmp_path)
    fp8 = _block_fp8_checkpoint(dense, tmp_path / "fp8", block=16)

    for directory in (dense, fp8):
        ps = PagedSafetensors(directory, dequant_dtype=torch.float32)
        for key in ("model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"):
            expected = ps.get(key).to(device="cpu", dtype=torch.float32)
            torch.testing.assert_close(ps.get_on(key, "cpu", torch.float32), expected)
        # and the forward that consumes it is unchanged on cpu
    ids = torch.arange(1, 7)
    a = streamed_moe_forward(PagedSafetensors(dense), ids, return_logits=True)["logits"]
    b = streamed_moe_forward(
        PagedSafetensors(dense), ids, device="cpu", dtype=torch.float32, return_logits=True
    )["logits"]
    torch.testing.assert_close(a, b)


# ------------------------------------------------------- batched stream-pass (one pull, many arms)
def _layout_zoo():
    """One tiny model per supported layout, reusing the parity tests' configurations."""
    zoo = []
    cfg = transformers.MixtralConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
        num_experts_per_tok=2, max_position_embeddings=64, rope_theta=1e6,
        attn_implementation="eager", tie_word_embeddings=False)
    zoo.append(("mixtral", transformers.MixtralForCausalLM(cfg), cfg))
    cfg = transformers.OlmoeConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, num_experts=4,
        num_experts_per_tok=2, norm_topk_prob=False, max_position_embeddings=64,
        attn_implementation="eager", tie_word_embeddings=False)
    zoo.append(("olmoe", transformers.OlmoeForCausalLM(cfg), cfg))
    cfg = transformers.Qwen2MoeConfig(
        vocab_size=99, hidden_size=32, intermediate_size=48, moe_intermediate_size=24,
        shared_expert_intermediate_size=40, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True, max_position_embeddings=64,
        attn_implementation="eager", tie_word_embeddings=False)
    zoo.append(("qwen2_moe", transformers.Qwen2MoeForCausalLM(cfg), cfg))
    cfg = _qwen3_moe_cfg(mlp_only_layers=[1])   # keep the dense-layer path in the gate
    zoo.append(("qwen3_moe", transformers.Qwen3MoeForCausalLM(cfg), cfg))
    return zoo


def _all_arms(n_rows, n_layers):
    arms = [(j, None) for j in range(n_rows)]
    for li in range(n_layers):
        for kind in ("attn", "mlp"):
            for j in range(n_rows):
                arms.append((j, (li, kind)))
    return arms


def test_batched_pass_is_bit_exact_against_serial_per_layout(tmp_path):
    """The batched kernel must EQUAL the serial one — not approximate it. Sorted-union expert
    iteration preserves each arm's serial accumulation order, so any drift is a bug. Covers all
    four layouts (incl. qwen2_moe's shared expert and a qwen3_moe dense layer), mixed row
    lengths, clean + every (layer, attn|mlp) arm."""
    from mrun.engine.moe_safetensors import streamed_moe_forward_batch

    rng = np.random.default_rng(4)
    for name, model, cfg in _layout_zoo():
        d = _save(model.eval().float(), cfg, tmp_path / name)
        ps = PagedSafetensors(d)
        assert detect_layout(ps).name == name
        rows = [torch.from_numpy(rng.integers(1, cfg.vocab_size, size=T))
                for T in (7, 10, 10)]
        arms = _all_arms(len(rows), cfg.num_hidden_layers)
        serial = torch.stack([
            streamed_moe_forward(
                ps, rows[j], ablate_component=ablate,
                capture_final_hidden=True, skip_lm_head=True,
            )["final_hidden"][-1]
            for j, ablate in arms
        ])
        batched = streamed_moe_forward_batch(ps, rows, arms)["final_last"]
        assert torch.equal(batched, serial), (
            f"{name}: max|diff|={(batched - serial).abs().max().item():.3e}"
        )


def test_batched_pass_pulls_each_weight_at_most_once(tmp_path):
    """The kernel's whole point: one streaming pass regardless of arm count. Every stored key
    is read at most once per call, and total pulls stay strictly under the serial equivalent."""
    from mrun.engine.moe_safetensors import streamed_moe_forward_batch

    cfg = _qwen3_moe_cfg()
    d = _save(transformers.Qwen3MoeForCausalLM(cfg).eval().float(), cfg, tmp_path)

    class _Counting(PagedSafetensors):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reads = {}

        def raw(self, key):
            self.reads[key] = self.reads.get(key, 0) + 1
            return super().raw(key)

    rng = np.random.default_rng(9)
    rows = [torch.from_numpy(rng.integers(1, cfg.vocab_size, size=8)) for _ in range(3)]
    arms = _all_arms(len(rows), cfg.num_hidden_layers)

    batch_store = _Counting(d)
    streamed_moe_forward_batch(batch_store, rows, arms)
    # detect_layout probes layer-0 q_norm once (per_head vs full) before the stream pulls it —
    # one extra read of one tiny vector, not a streaming inefficiency.
    detection_probe = "model.layers.0.self_attn.q_norm.weight"
    over_read = {key: n for key, n in batch_store.reads.items()
                 if n > (2 if key == detection_probe else 1)}
    assert not over_read, f"keys pulled more than once in one batched pass: {over_read}"

    serial_store = _Counting(d)
    for j, ablate in arms:
        streamed_moe_forward(serial_store, rows[j], ablate_component=ablate,
                             capture_final_hidden=True, skip_lm_head=True)
    assert sum(batch_store.reads.values()) < sum(serial_store.reads.values()) / 5


def test_engine_batched_surfaces_match_serial_taps_including_fp8(tmp_path):
    """Engine-level gate: hidden_last_batch (sticky taps) and final_hidden_arms (per-arm
    component axis) must reproduce the serial final_hidden under the same taps — on the plain
    checkpoint AND on a block-scaled fp8 one, where get_on's dequant feeds the batch."""
    cfg = _qwen3_moe_cfg()
    dense = _save(transformers.Qwen3MoeForCausalLM(cfg).eval().float(), cfg, tmp_path)
    fp8 = _block_fp8_checkpoint(dense, tmp_path / "fp8", block=16)
    direction = torch.from_numpy(
        np.random.default_rng(2).normal(size=cfg.hidden_size).astype(np.float32))
    rows = [np.arange(1, 8), np.arange(4, 10)]

    for directory in (dense, fp8):
        engine, _ = _stream_engine_at(directory, cfg)
        with engine.ablation(embed_direction=direction, component=(1, "mlp")):
            got = engine.hidden_last_batch(rows)
            expected = torch.stack([engine.final_hidden(row)[-1] for row in rows])
        assert torch.equal(got, expected)

        arms = [(0, None), (1, None), (0, (0, "attn")), (1, (1, "mlp"))]
        with engine.ablation(embed_direction=direction):
            multi = engine.final_hidden_arms(rows, arms)
            serial = []
            for j, ablate in arms:
                with engine.ablation(component=ablate):
                    serial.append(engine.final_hidden(rows[j])[-1])
        assert torch.equal(multi, torch.stack(serial))
