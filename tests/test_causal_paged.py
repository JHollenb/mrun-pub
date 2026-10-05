"""Parity gates for the causal-leg port (``mrun.testing.causal_paged``).

Every ported causal helper is gated here: a DENSE fp32 HF oracle (raw ``nn.Module`` + the exact
``register_forward_hook`` interventions the dense signature uses) vs the PAGED int8 adapter, on
Qwen2.5-0.5B. A leg without a green parity gate does not merge (house rule from the fused-route /
qstore ports).

Two things are MEASURED and REPORTED, not silently passed:
  * the ablation-delta ``max|Δ|`` between dense fp32 and paged int8 (offset cancels in a delta, so
    the tolerance is tight ~quant-scale), and
  * the SIGN-DISAGREEMENT rate on the sign-fragile specificity/dependence quantities — int8 can
    flip a near-zero signed delta (documented caveat), printed ``k/N`` mismatches like the
    behavior-parity gates, and only bounded (not required to be 0).

Model-gated: set ``MRUN_RUN_MODEL_TESTS=1`` and have the Qwen2.5-0.5B paged store + HF snapshot
cached. Working set is ~29 MB paged + ~2 GB dense fp32 (0.5B) — the house 0.5B-scale gate; RAM is
watched via ``ram_guard``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mrun.guard import ram_guard  # noqa: E402
from mrun.models import store_name  # noqa: E402
from mrun.paths import stores_root  # noqa: E402

MODEL = "qwen2.5-0.5b"

requires_model = pytest.mark.skipif(
    os.environ.get("MRUN_RUN_MODEL_TESTS", os.environ.get("MODEL_EXPERIMENTS_RUN_MODEL_TESTS")) != "1",
    reason="set MRUN_RUN_MODEL_TESTS=1 to run cached model tests",
)


def _store_available() -> bool:
    return (stores_root() / store_name(MODEL)).joinpath("manifest.json").exists()


# Documented int8 tolerances (MEASURED on Qwen2.5-0.5B, this port's parity run):
#   * causal Δlogprob drift dense-fp32 vs paged-int8 lands within ~0.2 on 0.5B (offset cancels in
#     a delta; the residual is pure int8 weight noise). Bound generously — the load-bearing
#     property is SIGN preservation on a clear signal, which ``compare_signatures`` reads.
#   * a Δlogprob whose true magnitude is below SIGN_EPS is in the int8 sign-fragile zone (the
#     specificity caveat) — such deltas may flip sign and are REPORTED, never asserted.
INT8_DELTA_TOL = 0.30
SIGN_EPS = 0.05


def _sign_disagreement(dense, paged):
    """(k, N) sign mismatches counted ONLY on clearly-nonzero dense deltas (|d| > SIGN_EPS); the
    near-zero, sign-fragile deltas are excluded from the count and merely reported by the caller."""
    clear = [(a, b) for a, b in zip(dense, paged) if abs(a) > SIGN_EPS]
    k = sum(int((a < 0) != (b < 0)) for a, b in clear)
    return k, len(clear)


# --------------------------------------------------------------------------- dense fp32 oracle
class DenseOracle:
    """Raw HF fp32 module + the frozen dense-signature interventions (mirrors the rd.* helpers and
    capability_signature's hooks exactly), so the gate compares against a self-contained reference
    that needs no discovery-repo import."""

    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        c = model.config
        self.n_layer = c.num_hidden_layers
        self.n_head = c.num_attention_heads
        self.d_head = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)

    def ids(self, prompt):
        return self.tok(prompt, return_tensors="pt")["input_ids"]

    def _lp(self, logits, aid):
        return float(torch.log_softmax(logits[0, -1].float(), dim=-1)[aid].item())

    def answer_logprob(self, prompt, aid):
        with torch.no_grad():
            lg = self.model(self.ids(prompt)).logits
        return self._lp(lg, aid)

    def _run_with_hooks(self, prompt, aid, hooks):
        handles = []
        for mod, fn, kind in hooks:
            h = mod.register_forward_pre_hook(fn) if kind == "pre" else mod.register_forward_hook(fn)
            handles.append(h)
        try:
            with torch.no_grad():
                lg = self.model(self.ids(prompt)).logits
        finally:
            for h in handles:
                h.remove()
        return self._lp(lg, aid)

    # --- head-zero pre-hook on o_proj (zeros a head's columns in the concatenated input) ---
    def ablate_heads_logprob(self, prompts, heads):
        def head_hook(head_idx):
            s, e = head_idx * self.d_head, (head_idx + 1) * self.d_head

            def hook(_m, args):
                x = args[0].clone()
                x[..., s:e] = 0.0
                return (x,)
            return hook

        deltas = []
        for prompt, _exp, aid in prompts:
            clean = self.answer_logprob(prompt, aid)
            hooks = [(self.model.model.layers[l].self_attn.o_proj, head_hook(h), "pre")
                     for l, h in heads]
            deltas.append(self._run_with_hooks(prompt, aid, hooks) - clean)
        return deltas

    # --- rank-1 dir removal post-hook on the decoder layer output ---
    def remove_direction_delta(self, prompt, aid, layer, u):
        uu = (torch.as_tensor(u, dtype=torch.float32) / (torch.as_tensor(u, dtype=torch.float32).norm() + 1e-9))

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            proj = (h.float() @ uu).unsqueeze(-1) * uu
            h2 = (h.float() - proj).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

        clean = self.answer_logprob(prompt, aid)
        return self._run_with_hooks(prompt, aid, [(self.model.model.layers[layer], hook, "post")]) - clean

    # --- MLP-zero post-hook (zeros the whole MLP output) ---
    def mlp_zero_delta(self, prompt, aid, layer):
        def hook(_m, _a, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)

        clean = self.answer_logprob(prompt, aid)
        return self._run_with_hooks(prompt, aid, [(self.model.model.layers[layer].mlp, hook, "post")]) - clean

    def hidden_states(self, prompt):
        with torch.no_grad():
            out = self.model(self.ids(prompt), output_hidden_states=True)
        return [h[0].float() for h in out.hidden_states]     # [nL+1] of [T, hidden]

    def top1(self, prompt):
        with torch.no_grad():
            return int(self.model(self.ids(prompt)).logits[0, -1].argmax().item())


@pytest.fixture(scope="module")
def pair():
    if not _store_available():
        pytest.skip(f"no paged store for {MODEL} under {stores_root()}")
    from mrun.engine import open_engine
    from mrun.models import load_hf_model, load_tokenizer
    from mrun.testing.causal_paged import PagedCausal

    with ram_guard("causal-paged gate: load"):
        tok = load_tokenizer(MODEL)
        hf = load_hf_model(MODEL, torch_dtype=torch.float32, attn_implementation="eager").eval()
        eng = open_engine(MODEL, backend="paged")
    yield DenseOracle(hf, tok), PagedCausal(eng)
    eng.close() if hasattr(eng, "close") else None


# prompts with an in-context repeat (copy/induction) so answer_id occurs in context
_COPY = [
    "The capital of France is Paris. The capital of France is",
    "cat dog cat dog cat",
    "1 2 3 4 1 2 3",
    "red green blue red green",
]
_HEADS = [(0, 0), (5, 3), (10, 7), (13, 2)]


def _copy_prompts(oracle):
    out = []
    for p in _COPY:
        nxt = oracle.top1(p)
        out.append((p, oracle.tok.decode([nxt]), nxt))
    return out


# --------------------------------------------------------------------------- LEG 1: head ablation
@requires_model
def test_head_ablation_parity(pair):
    dense, paged = pair
    with ram_guard("head-ablation parity"):
        prompts = _copy_prompts(dense)
        dd = dense.ablate_heads_logprob(prompts, _HEADS)
        pd = paged.ablate_heads_logprob(prompts, _HEADS, "parity")
    maxdiff = max(abs(a - b) for a, b in zip(dd, pd))
    k, n = _sign_disagreement(dd, pd)
    print(f"\n[head-ablation] dense={np.round(dd, 4)}  paged={np.round(pd, 4)}")
    print(f"[head-ablation] max|Δ|={maxdiff:.2e}  sign-disagreement(clear)={k}/{n}")
    assert maxdiff < INT8_DELTA_TOL, f"head-ablation Δlogprob drift max|Δ|={maxdiff} > int8 tol"
    assert k == 0, f"head-ablation sign flipped on {k}/{n} CLEAR deltas (not the near-zero zone)"


# --------------------------------------------------------------------------- LEG 2: hidden states / probe-R²
@requires_model
def test_hidden_states_indexing_matches_hf(pair):
    """PagedEngine.hidden_states[i] == HF output_hidden_states[i] for EVERY i (int8 tol), INCLUDING
    the final-normed top entry. This is the L-vs-L+1 / embedding-included / final-norm gotcha: the
    dense depth-probe reads hidden_states[l+1]; a shifted or un-normed paged tap silently changes
    probe-R²."""
    dense, paged = pair
    prompt = "def add ( a , b ) : return a + b\ndef mul ( a , b ) : return"
    with ram_guard("hidden-states indexing"):
        dh = dense.hidden_states(prompt)
        ph = paged.collect_hidden_states(np.asarray(paged._ids(prompt), np.int64))
    assert len(ph) == len(dh) == dense.n_layer + 1, f"len {len(ph)} vs {len(dh)} vs nL+1"

    def _cos(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))

    # correspondence, NOT bit-equality: int8 weight noise scales the residual magnitude at depth,
    # so cosine (direction) is the right correspondence metric — a mis-INDEXED comparison (layer L
    # vs L+1) tanks cosine, while int8 drift leaves it ~1. Every index must line up.
    per_index_cos = [_cos(ph[i], dh[i]) for i in range(len(dh))]
    print(f"\n[hidden-states] per-index cosine(paged,dense): {[round(x, 3) for x in per_index_cos]}")
    for i, c in enumerate(per_index_cos):
        assert c > 0.9, f"hidden_states[{i}] cosine {c:.3f} — index does not correspond to HF layer"

    # ADVERSARIAL: a deliberate off-by-one (compare paged[i] to dense[i+1]) must FAIL the same bar,
    # proving the cosine gate actually discriminates the indexing it is meant to protect.
    shifted = [_cos(ph[i], dh[i + 1]) for i in range(len(dh) - 1)]
    assert min(shifted) < 0.9, "off-by-one shift still passed cosine — gate can't see mis-indexing"


@requires_model
def test_hidden_states_final_norm_is_load_bearing(pair):
    """Guard: the top entry REQUIRES the final RMSNorm. Compare the HF-normed top against the raw
    pre-norm paged residual (collect_hs without the norm) — they must DIVERGE, proving the norm in
    PagedEngine.hidden_states is doing real work, not a no-op."""
    dense, paged = pair
    prompt = "def add ( a , b ) : return a + b\ndef mul ( a , b ) : return"
    ids = np.asarray(paged._ids(prompt), np.int64)
    with ram_guard("final-norm guard"):
        dh = dense.hidden_states(prompt)
        raw: list = []
        paged.eng._paged_logits(paged.eng.store, ids, collect_hs=raw)
        normed = paged.collect_hidden_states(ids)
    # normed top matches HF; raw (pre-norm) top does NOT
    norm_gap = float((normed[-1].float() - dh[-1]).abs().max())
    raw_gap = float((raw[-1].float() - dh[-1]).abs().max())
    print(f"\n[final-norm] normed-top gap={norm_gap:.3f}  raw(pre-norm)-top gap={raw_gap:.3f}")
    assert norm_gap < raw_gap, "final norm should bring the top entry CLOSER to HF"
    assert raw_gap > 0.5, "pre-norm top should be clearly different from the HF normed top"


# --------------------------------------------------------------------------- LEG 3: direction ablation
@requires_model
def test_direction_removal_parity(pair):
    dense, paged = pair
    layer = 12
    torch.manual_seed(0)
    u = torch.randn(dense.model.config.hidden_size)
    with ram_guard("dir-removal parity"):
        prompts = _copy_prompts(dense)
        dd, pd = [], []
        for prompt, _e, aid in prompts:
            dd.append(dense.remove_direction_delta(prompt, aid, layer, u))
            pd.append(paged.remove_direction_logprob(paged._ids(prompt), aid, layer, u.numpy())
                      - paged.answer_logprob(paged._ids(prompt), aid))
    maxdiff = max(abs(a - b) for a, b in zip(dd, pd))
    k, n = _sign_disagreement(dd, pd)
    print(f"\n[dir-removal] dense={np.round(dd, 4)}  paged={np.round(pd, 4)}")
    print(f"[dir-removal] max|Δ|={maxdiff:.2e}  sign-disagreement(clear)={k}/{n}")
    assert maxdiff < INT8_DELTA_TOL, f"dir-removal Δlogprob drift max|Δ|={maxdiff} > int8 tol"
    assert k == 0, f"dir-removal sign flipped on {k}/{n} CLEAR deltas (not the near-zero zone)"


@requires_model
def test_mlp_zero_parity(pair):
    dense, paged = pair
    layer = 12
    with ram_guard("mlp-zero parity"):
        prompts = _copy_prompts(dense)
        dd, pd = [], []
        for prompt, _e, aid in prompts:
            dd.append(dense.mlp_zero_delta(prompt, aid, layer))
            pd.append(paged.ablate_mlp_layer_logprob(paged._ids(prompt), aid, layer)
                      - paged.answer_logprob(paged._ids(prompt), aid))
    maxdiff = max(abs(a - b) for a, b in zip(dd, pd))
    k, n = _sign_disagreement(dd, pd)
    print(f"\n[mlp-zero] dense={np.round(dd, 4)}  paged={np.round(pd, 4)}  max|Δ|={maxdiff:.2e}"
          f"  sign-disagreement(clear)={k}/{n}")
    assert maxdiff < INT8_DELTA_TOL, f"mlp-zero Δlogprob drift max|Δ|={maxdiff} > int8 tol"
    assert k == 0, f"mlp-zero sign flipped on {k}/{n} CLEAR deltas"


# --------------------------------------------------------------------------- LEG 4: present composition
@requires_model
def test_filter_prompts_parity(pair):
    """The greedy solved-filter picks the SAME prompts dense vs paged (present-leg gate)."""
    dense, paged = pair
    cands = [(p, dense.tok.decode([dense.top1(p)])) for p in _COPY]
    with ram_guard("filter-prompts parity"):
        # dense reference via top-1 argmax
        dense_keep = [(p, e) for (p, e) in cands if dense.top1(p) == dense.tok.encode(e)[0]]
        paged_keep = paged.filter_prompts(cands, "PARITY", max_prompts=len(cands))
    dset = {p for p, _ in dense_keep}
    pset = {p for p, _, _ in paged_keep}
    print(f"\n[filter-prompts] dense kept {len(dset)}  paged kept {len(pset)}  agree={dset == pset}")
    assert dset == pset, f"solved-filter disagreement: dense={dset} paged={pset}"


@requires_model
def test_find_induction_heads_overlap(pair):
    """Top induction heads by copy attention mass overlap strongly dense vs paged (int8 attention is
    approximate but top-1 mass is preserved)."""
    dense, paged = pair
    from collections import defaultdict

    prompts = _copy_prompts(dense)
    with ram_guard("induction-heads parity"):
        paged_heads = set(paged.find_induction_heads(prompts, top_n=6))
        # dense reference: same computation on HF attentions
        acc = defaultdict(float)
        cnt = 0
        for prompt, _e, aid in prompts:
            seq = dense.ids(prompt)[0].tolist()
            apos = next((i for i, t in enumerate(seq[:-1]) if t == aid), None)
            if apos is None:
                continue
            with torch.no_grad():
                attns = dense.model(dense.ids(prompt), output_attentions=True).attentions
            for l in range(dense.n_layer):
                a = attns[l][0, :, -1, :].float()
                for h in range(dense.n_head):
                    acc[(l, h)] += float(a[h, apos])
            cnt += 1
        top = sorted(acc, key=acc.get, reverse=True)[:6]
    dense_heads = set(top)
    overlap = len(dense_heads & paged_heads)
    print(f"\n[induction-heads] dense={sorted(dense_heads)}  paged={sorted(paged_heads)}  overlap={overlap}/6")
    assert overlap >= 4, f"induction-head top-6 overlap only {overlap}/6"


# --------------------------------------------------------------------------- compare_signatures gate
@requires_model
def test_causal_use_leg_compare_signatures(pair):
    """Assemble a causal_use leg payload (dependence + specificity, size-matched null) from the DENSE
    oracle and from the PAGED adapter, then gate them with the SAME ``compare_signatures`` the
    backend gate uses. A green verdict = the paged causal leg preserves the capability signal.
    Also reports the specificity sign-agreement (the int8 sign-fragility caveat)."""
    from mrun.testing.signature import compare_signatures

    dense, paged = pair
    rng_d = np.random.default_rng(0)
    rng_p = np.random.default_rng(0)
    prompts = _copy_prompts(dense)
    heads = _HEADS
    hset = set(heads)

    with ram_guard("compare_signatures causal leg"):
        # dense arm
        d_ind = dense.ablate_heads_logprob(prompts, heads)
        d_rnd = []
        allc = [(l, h) for l in range(dense.n_layer) for h in range(dense.n_head) if (l, h) not in hset]
        sums = [0.0] * len(prompts)
        for _ in range(3):
            idx = rng_d.choice(len(allc), size=len(heads), replace=False)
            for i, dv in enumerate(dense.ablate_heads_logprob(prompts, [allc[int(j)] for j in idx])):
                sums[i] += dv
        d_rnd = [s / 3 for s in sums]
        # paged arm
        p_ind = paged.ablate_heads_logprob(prompts, heads, "ind")
        p_rnd = paged.ablate_random_heads_logprob(prompts, len(heads), hset, rng_p, n_repeats=3)

    def _excess(target, null):
        return float(np.mean(null)) - float(np.mean(target))

    d_dep, d_spec = float(np.mean(d_ind)), -_excess(d_ind, d_rnd)
    p_dep, p_spec = float(np.mean(p_ind)), -_excess(p_ind, p_rnd)
    print(f"\n[compare_sig] dense dep={d_dep:.4f} spec={d_spec:.4f} | paged dep={p_dep:.4f} spec={p_spec:.4f}")
    spec_sign_agree = (d_spec > 0) == (p_spec > 0)
    print(f"[compare_sig] specificity sign agreement: {spec_sign_agree}  (int8 sign-fragility caveat)")

    def payload(dep, spec):
        return {
            "capability": "long_range_variable_binding",
            "model_name": MODEL,
            "config": {"dtype": "fp32/paged"},
            "primitives": {"lrvb": {
                "present": {"measurable": True, "present": True},
                "causal_use": {"measurable": True, "dependence": round(dep, 4), "specificity": round(spec, 4)},
                "depth": {"measurable": True, "max_distance_surviving": 1},
            }},
            "coverage": {},
        }

    verdict = compare_signatures(payload(d_dep, d_spec), payload(p_dep, p_spec))
    print(f"[compare_sig] verdict pass={verdict['pass']} failures={verdict['failures']}")
    assert verdict["pass"], verdict["failures"]
