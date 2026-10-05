"""Real-checkpoint parity gate for the deepseek_v4 moe-stream adapter.

Two fully independent stacks, same prompts, fp32 CPU:

  ENGINE side  mrun's streamed forward (``mri.deepseek_v4_stream``) + mrun's fp8/fp4 dequant.
  ORACLE side  transformers 5.13.1's OWN ``DeepseekV4DecoderLayer`` / rotary / hyper-head
               modules, instantiated ONE LAYER AT A TIME (bounded memory: one layer's fp32
               experts ~26 GB on V4-Flash) and filled through transformers' OWN
               ``Fp8Dequantize`` (``integrations.finegrained_fp8``) — so the gate
               cross-validates BOTH the forward math and the dequant, nothing is shared.

The oracle builds its sliding-window mask manually (0 / finfo.min, ``kv<=q & kv>q-window``);
those semantics are certified against HF's real mask-builder end-to-end by the tiny-model gate
in tests/test_deepseek_v4_stream.py (which calls ``model(input_ids=...)`` outright).

Gates reported (prereg: experiments/2026-07-26-deepseek-hub-physiology/PLAN.md):
  G1  per-layer hyper-connection stream max|diff| on the short prompt
  G2  full-stack logits max|diff| on the short prompt (target <= 3e-5 fp32-vs-fp32)
  G3  argmax agreement across ALL positions of every prompt (needs >= 64 positions, all equal)

Usage (beast, through mrun):
  python -m mrun.mri.deepseek_v4_parity --model /mnt/big/llm-models/DeepSeek-V4-Flash
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

TAG = "DSV4PARITY"

LONG_TEXT = (
    "Measurement is the discipline of doubting your own instruments, and a number that has "
    "not survived a null control is a rumor with a decimal point. The streamed forward pulls "
    "one weight at a time off the disk, so the resident working set stays bounded by a single "
    "expert matrix rather than the whole model. In 1869 Mendeleev left gaps in his table and "
    "predicted the properties of gallium, scandium, and germanium before anyone had seen "
    "them; the gaps were the theory. def fibonacci(n):\n    a, b = 0, 1\n    for _ in "
    "range(n):\n        a, b = b, a + b\n    return a\n"
    "The primes begin 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, and the gaps between them "
    "grow like the logarithm. A sliding window of one hundred twenty-eight tokens sees only "
    "the recent past; the compressor folds every four tokens into one gated summary so the "
    "indexer can reach back further than the window allows, and the heavily compressed "
    "branch folds one hundred twenty-eight at a time. Paris is the capital of France, Tokyo "
    "is the capital of Japan, and Canberra, not Sydney, is the capital of Australia. "
    "When the tide goes out you discover who has been swimming naked; when the null control "
    "fires you discover which effects were plumbing. The quick brown fox jumps over the lazy "
    "dog while the five boxing wizards jump quickly. E = mc^2 relates rest energy to mass, "
    "and F = ma is not a definition but a discovery about the world. "
)


def log(msg: str) -> None:
    print(f"[dsv4parity] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ oracle
_LAYER_FILL = [
    # (native suffix under layers.N., HF DecoderLayer state-dict key)
    ("attn_norm.weight", "input_layernorm.weight"),
    ("ffn_norm.weight", "post_attention_layernorm.weight"),
    ("hc_attn_fn", "attn_hc.fn"), ("hc_attn_base", "attn_hc.base"),
    ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_fn", "ffn_hc.fn"), ("hc_ffn_base", "ffn_hc.base"),
    ("hc_ffn_scale", "ffn_hc.scale"),
    ("attn.wq_a.weight", "self_attn.q_a_proj.weight"),
    ("attn.q_norm.weight", "self_attn.q_a_norm.weight"),
    ("attn.wq_b.weight", "self_attn.q_b_proj.weight"),
    ("attn.wkv.weight", "self_attn.kv_proj.weight"),
    ("attn.kv_norm.weight", "self_attn.kv_norm.weight"),
    ("attn.wo_a.weight", "self_attn.o_a_proj.weight"),
    ("attn.wo_b.weight", "self_attn.o_b_proj.weight"),
    ("attn.attn_sink", "self_attn.sinks"),
    ("attn.compressor.wkv.weight", "self_attn.compressor.kv_proj.weight"),
    ("attn.compressor.wgate.weight", "self_attn.compressor.gate_proj.weight"),
    ("attn.compressor.ape", "self_attn.compressor.position_bias"),
    ("attn.compressor.norm.weight", "self_attn.compressor.kv_norm.weight"),
    ("attn.indexer.compressor.wkv.weight", "self_attn.compressor.indexer.kv_proj.weight"),
    ("attn.indexer.compressor.wgate.weight", "self_attn.compressor.indexer.gate_proj.weight"),
    ("attn.indexer.compressor.ape", "self_attn.compressor.indexer.position_bias"),
    ("attn.indexer.compressor.norm.weight", "self_attn.compressor.indexer.kv_norm.weight"),
    ("attn.indexer.wq_b.weight", "self_attn.compressor.indexer.q_b_proj.weight"),
    ("attn.indexer.weights_proj.weight", "self_attn.compressor.indexer.scorer.weights_proj.weight"),
    ("ffn.gate.weight", "mlp.gate.weight"),
    ("ffn.gate.bias", "mlp.gate.e_score_correction_bias"),
    ("ffn.gate.tid2eid", "mlp.gate.tid2eid"),
    ("ffn.shared_experts.w1.weight", "mlp.shared_experts.gate_proj.weight"),
    ("ffn.shared_experts.w3.weight", "mlp.shared_experts.up_proj.weight"),
    ("ffn.shared_experts.w2.weight", "mlp.shared_experts.down_proj.weight"),
]


class _OracleReader:
    """Raw-tensor reads off the shards + transformers' OWN dequant (nothing from mrun.fp8)."""

    def __init__(self, ps):
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize
        self.ps = ps
        self.deq = Fp8Dequantize(None)

    def get(self, key: str) -> torch.Tensor:
        raw = self.ps.raw(key)
        scale_key = key[: -len(".weight")] + ".scale" if key.endswith(".weight") else None
        if scale_key and self.ps.has(scale_key):
            return self.deq._dequantize_one(raw, self.ps.raw(scale_key),
                                            output_dtype=torch.float32)
        if raw.dtype in (torch.long, torch.int64, torch.int32):
            return raw
        return raw.to(torch.float32)


def _fill_layer(layer, reader: "_OracleReader", i: int) -> None:
    """Copy every checkpoint tensor of layer ``i`` into the HF module; assert full coverage
    both ways (an unfilled parameter would silently carry torch.empty garbage)."""
    wanted = dict(layer.state_dict())
    filled: set[str] = set()
    with torch.no_grad():
        for native_suffix, hf_key in _LAYER_FILL:
            native = f"layers.{i}.{native_suffix}"
            if not reader.ps.has(native):
                continue
            assert hf_key in wanted, f"layer {i}: no module slot for {native} -> {hf_key}"
            value = reader.get(native)
            wanted[hf_key].copy_(value.to(wanted[hf_key].dtype))
            filled.add(hf_key)
        n_experts = layer.mlp.experts.num_experts
        inter = layer.mlp.experts.intermediate_dim
        gate_up = layer.mlp.experts.gate_up_proj
        down = layer.mlp.experts.down_proj
        for e in range(n_experts):
            gate_up[e, :inter].copy_(reader.get(f"layers.{i}.ffn.experts.{e}.w1.weight"))
            gate_up[e, inter:].copy_(reader.get(f"layers.{i}.ffn.experts.{e}.w3.weight"))
            down[e].copy_(reader.get(f"layers.{i}.ffn.experts.{e}.w2.weight"))
        filled.update({"mlp.experts.gate_up_proj", "mlp.experts.down_proj"})
    missing = set(wanted) - filled
    assert not missing, f"layer {i}: oracle params never filled from checkpoint: {sorted(missing)}"


def _oracle_min_mask(T: int, window: int) -> torch.Tensor:
    q = torch.arange(T)[:, None]
    kv = torch.arange(T)[None, :]
    allowed = (kv <= q) & (kv > q - window)
    mask = torch.full((T, T), torch.finfo(torch.float32).min)
    return mask.masked_fill(allowed, 0.0)[None, None]


@torch.no_grad()
def run_oracle(model_dir: Path, ps, prompts: dict[str, torch.Tensor],
               rss_cap_gb: float) -> dict:
    """transformers-modules forward, one layer resident at a time. Returns per-prompt logits
    and per-layer stream stacks."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Config,
        DeepseekV4DecoderLayer,
        DeepseekV4HyperHead,
        DeepseekV4RMSNorm,
        DeepseekV4RotaryEmbedding,
    )
    from ..engine.moe_safetensors import _rss_gb

    config = DeepseekV4Config.from_pretrained(model_dir)
    config._attn_implementation = "eager"
    reader = _OracleReader(ps)
    rotary = DeepseekV4RotaryEmbedding(config).float().eval()

    emb = reader.get("embed.weight")
    states: dict[str, torch.Tensor] = {}
    posemb: dict[str, dict] = {}
    masks: dict[str, torch.Tensor] = {}
    for name, ids in prompts.items():
        h = emb[ids][None].float()                               # [1, T, d]
        states[name] = h.unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
        pos = torch.arange(ids.shape[0])[None]
        posemb[name] = {
            "main": rotary(h, position_ids=pos, layer_type="main"),
            "compress": rotary(h, position_ids=pos, layer_type="compress"),
            "_pos": pos,
        }
        masks[name] = _oracle_min_mask(int(ids.shape[0]), config.sliding_window)
    del emb
    ps.release()

    layer_streams: dict[str, list[torch.Tensor]] = {name: [] for name in prompts}
    peak_rss = [0.0]
    t0 = time.time()
    for i in range(config.num_hidden_layers):
        layer = DeepseekV4DecoderLayer(config, i).float().eval()
        _fill_layer(layer, reader, i)
        ps.release()
        for name, ids in prompts.items():
            pe = posemb[name]
            states[name] = layer(
                states[name],
                input_ids=ids[None],
                position_embeddings={"main": pe["main"], "compress": pe["compress"]},
                position_ids=pe["_pos"],
                attention_mask=masks[name],
                past_key_values=None,
            )
            layer_streams[name].append(states[name][0].detach().clone())
        # Sample the high-water mark while the fp32 layer is STILL LIVE. Sampling after
        # `del layer; gc.collect()` reports the post-free trough (measured 2026-07-26:
        # ~1.4 GB logged vs ~30 GB actual under `ps`), which made rss_cap_gb structurally
        # unable to fire on the peak it exists to bound.
        peak = _rss_gb()
        peak_rss[0] = max(peak_rss[0], peak)
        del layer
        gc.collect()
        rss = _rss_gb()
        log(f"oracle layer {i:2d}/{config.num_hidden_layers} "
            f"peak={peak:.1f}GB resid={rss:.1f}GB {time.time() - t0:.0f}s")
        if peak > rss_cap_gb:
            raise MemoryError(
                f"oracle peak RSS {peak:.1f}GB > {rss_cap_gb}GB (layer {i}); "
                f"post-free residual was {rss:.1f}GB")

    hyper_head = DeepseekV4HyperHead(config).float().eval()
    with torch.no_grad():
        hyper_head.hc_fn.copy_(reader.get("hc_head_fn"))
        hyper_head.hc_base.copy_(reader.get("hc_head_base"))
        hyper_head.hc_scale.copy_(reader.get("hc_head_scale"))
    norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps).float().eval()
    with torch.no_grad():
        norm.weight.copy_(reader.get("norm.weight"))
    head_w = reader.get("head.weight")
    logits = {}
    for name in prompts:
        final = norm(hyper_head(states[name]))
        logits[name] = (final[0] @ head_w.T).detach()
    del head_w
    ps.release()
    return {"logits": logits, "layer_streams": layer_streams,
            "peak_rss_gb": round(peak_rss[0], 2),
            "wall_s": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ main gate
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rss-cap-gb", type=float, default=52.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from ..engine.deepseek_v4_stream import dsv4_forward
    from ..engine.moe_safetensors import PagedSafetensors, detect_layout, resolve_model_dir

    model_dir = resolve_model_dir(args.model)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    ps = PagedSafetensors(model_dir, dequant_dtype=torch.float32)
    lay = detect_layout(ps)
    assert lay.name == "deepseek_v4", f"unexpected layout {lay.name}"

    prompts = {
        "short10": torch.as_tensor(tok("The capital of France is the city of",
                                       return_tensors="pt").input_ids[0][:10]),
        "facts40": torch.as_tensor(tok(
            "Water boils at one hundred degrees Celsius at sea level, and the chemical "
            "formula for table salt is NaCl. The third planet from the sun is",
            return_tensors="pt").input_ids[0][:40]),
        "long300": torch.as_tensor(tok(LONG_TEXT, return_tensors="pt").input_ids[0][:300]),
    }
    for name, ids in prompts.items():
        log(f"prompt {name}: {int(ids.shape[0])} tokens")

    # ---- engine side (mrun streamed forward, fp32 cpu) ------------------------------------
    engine_out: dict = {}
    for name, ids in prompts.items():
        t0 = time.time()
        res = dsv4_forward(
            ps, ids, device="cpu", dtype=torch.float32, return_logits=True,
            capture_hidden_states=(name == "short10"),
            abort_rss_gb=args.rss_cap_gb, log=log)
        engine_out[name] = res
        log(f"engine {name}: wall={res['wall_s']}s rss={res['rss_gb']}GB "
            f"ce={res['ce']:.4f} ({time.time() - t0:.0f}s)")

    # ---- oracle side (transformers modules + transformers dequant) ------------------------
    oracle = run_oracle(model_dir, ps, prompts, args.rss_cap_gb)

    # ---- gates ----------------------------------------------------------------------------
    report: dict = {
        "model": str(model_dir), "layout": lay.name, "dtype": "float32", "device": "cpu",
        "engine_wall_s": {n: engine_out[n]["wall_s"] for n in prompts},
        "engine_rss_gb": {n: engine_out[n]["rss_gb"] for n in prompts},
        "engine_ce": {n: round(engine_out[n]["ce"], 4) for n in prompts},
        "oracle_wall_s": oracle["wall_s"],
        "oracle_peak_rss_gb": oracle.get("peak_rss_gb"),
        "prompt_tokens": {n: int(prompts[n].shape[0]) for n in prompts},
        "gates": {},
    }
    # G1: per-layer stream diff on the short prompt.
    # CONTRACT (see dsv4_forward docstring): the engine mirrors HF's output_hidden_states —
    # entry 0 is the post-tap embedding [T,d]; entries 1..L-1 are the hc stream stacks
    # [T,hc,d] after layers 0..L-2; and the LAST entry is the post-final-norm collapsed
    # state [T,d], which REPLACES layer L-1's stack (same `hidden_states[-1] = ...` house
    # convention as moe_stream.py / paged_forward.py). So only L-1 stacks are comparable.
    # The final layer + hyper-head collapse are covered end-to-end by G2/G3 below, which is
    # why this is a reporting bound and not a coverage hole.
    hs = engine_out["short10"]["hidden_states"]
    oracle_streams = oracle["layer_streams"]["short10"]
    n_cmp = len(hs) - 2
    if n_cmp != len(oracle_streams) - 1:
        raise RuntimeError(
            f"hidden_states contract drift: engine exposes {n_cmp} comparable stacks "
            f"(len(hidden_states)={len(hs)}) but oracle recorded {len(oracle_streams)} layers")
    layer_diffs = []
    for i in range(n_cmp):
        mine, ref = hs[1 + i].float(), oracle_streams[i]
        if tuple(mine.shape) != tuple(ref.shape):
            raise RuntimeError(
                f"layer {i} stream shape mismatch: engine {tuple(mine.shape)} vs "
                f"oracle {tuple(ref.shape)} (expected [T, hc, d])")
        layer_diffs.append(float((mine - ref).abs().max()))
    report["gates"]["G1_layer_stream_max_abs_diff"] = [round(d, 9) for d in layer_diffs]
    report["gates"]["G1_worst"] = max(layer_diffs)
    report["gates"]["G1_layers_compared"] = n_cmp
    report["gates"]["G1_layers_total"] = len(oracle_streams)
    report["gates"]["G1_note"] = (
        "last layer's stack is not exposed by the HF-mirroring hidden_states contract; "
        "covered by G2/G3 end-to-end")

    # G2/G3: logits diff + argmax agreement over every position of every prompt
    total_pos = agree = 0
    per_prompt = {}
    for name in prompts:
        mine = engine_out[name]["logits"].float()
        ref = oracle["logits"][name].float()
        diff = float((mine - ref).abs().max())
        am_mine = mine.argmax(-1)
        am_ref = ref.argmax(-1)
        n_pos = int(am_mine.shape[0])
        n_agree = int((am_mine == am_ref).sum())
        per_prompt[name] = {"max_abs_logit_diff": diff, "argmax_agree": n_agree,
                            "n_positions": n_pos}
        total_pos += n_pos
        agree += n_agree
    report["gates"]["G2_per_prompt"] = per_prompt
    report["gates"]["G2_short10_max_abs_logit_diff"] = per_prompt["short10"]["max_abs_logit_diff"]
    report["gates"]["G3_argmax_positions"] = total_pos
    report["gates"]["G3_argmax_agree"] = agree
    report["gates"]["PASS_G2_3e-5"] = bool(
        max(p["max_abs_logit_diff"] for p in per_prompt.values()) <= 3e-5)
    report["gates"]["PASS_G3_all_argmax"] = bool(agree == total_pos and total_pos >= 64)

    payload = json.dumps(report, default=float)
    print(f"==={TAG}_BEGIN===")
    print(payload)
    print(f"==={TAG}_END===")
    if args.out:
        Path(args.out).write_text(payload)
    ok = report["gates"]["PASS_G2_3e-5"] and report["gates"]["PASS_G3_all_argmax"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
