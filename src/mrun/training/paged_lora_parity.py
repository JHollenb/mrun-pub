"""Parity gate: train the SAME LoRA via a dense fp32 HF base and via the paged int8 base,
then compare TRAINING OUTCOMES and ADAPTER GEOMETRY.

This is the MPS-contamination standard applied to ``paged_lora``. int8 in the frozen base WILL
shift the optimisation trajectory; the question this gate answers with numbers is whether that
shift stays within the run-to-run **seed-noise floor** (⇒ cleared for geometry-sensitive
instrument legs, like the CUDA path was) or blows past it ~6× (⇒ cleared for CAPABILITY work only,
like the MPS path was). Either verdict is acceptable — it must be *stated with the measured ratio*.

Design (isolates the int8 effect from seed noise):
  run A = dense fp32, seed S0        run B = dense fp32, seed S1        run C = paged int8, seed S0
  * A and C share a BIT-IDENTICAL LoRA init (C's init is copied into A) and see the IDENTICAL
    batch stream ⇒ ``angle(A, C)`` is *purely* the int8-base effect.
  * B uses a different seed (different init + batch order) ⇒ ``angle(A, B)`` is the seed-noise
    FLOOR at fixed (fp32) precision.
  * ratio = angle(A,C) / angle(A,B). ~1 ⇒ geometry-clean; ~6 ⇒ capability-only.

Outcome parity = held-out forced-choice accuracy of A vs C (vs the A-vs-B spread for context).

The bank is a self-contained transitive-comparison operator (the train-into-substrate operator):
generalising, forced-choice, multi-token answers scored by average answer-token log-prob. Hermetic
so the gate has no cross-repo dependency.
"""
from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .paged_lora import LoRAConfig, PagedLoRATrainer

# HF submodule attribute per LoRA site name.
_HF_ATTR = {
    "q": ("self_attn", "q_proj"), "k": ("self_attn", "k_proj"),
    "v": ("self_attn", "v_proj"), "o": ("self_attn", "o_proj"),
    "gate": ("mlp", "gate_proj"), "up": ("mlp", "up_proj"), "down": ("mlp", "down_proj"),
}

_NAMES = [
    " Bella", " Paloma", " Evan", " Milo", " Cleo", " Tom", " Reza", " Vera", " Suki",
    " Jack", " Mara", " Nia", " Omar", " Priya", " Quinn", " Rosa", " Sven", " Tara",
    " Uma", " Viktor", " Wren", " Xena", " Yousef", " Zaid", " Ana", " Beto", " Cira", " Dov",
]
_RELS = [("stronger", "weakest", "strongest"), ("faster", "slowest", "fastest"),
         ("taller", "shortest", "tallest"), ("older", "youngest", "oldest")]


@dataclass
class Probe:
    prompt: str
    correct: str
    distractors: list[str]


def make_comparison_bank(
    n_train: int, n_eval: int, *, seed: int = 0
) -> tuple[list[Probe], list[Probe]]:
    """Transitive-comparison operator. Each example: a random total order over k names, k-1 edges
    stated (shuffled + randomly oriented), ask for the extreme (strongest/weakest). Generalising —
    identities are re-randomised so the model learns the compose-then-rank operator, not pairs."""
    rng = random.Random(seed)

    def one() -> Probe:
        k = rng.choice([3, 3, 4])
        names = rng.sample(_NAMES, k + 3)          # k in chain + 3 filler distractors
        chain = names[:k]                           # index 0 = strongest ... k-1 = weakest
        rel, weak_w, strong_w = rng.choice(_RELS)
        edges = []                                  # adjacent edges make the order recoverable
        for i in range(k - 1):
            hi, lo = chain[i], chain[i + 1]
            if rng.random() < 0.5:
                edges.append(f"{hi.strip()} is {rel} than {lo.strip()}.")
            else:                                   # inverse relation, same information
                edges.append(f"{lo.strip()} is less {rel.rstrip('er')} than {hi.strip()}.")
        rng.shuffle(edges)
        ask_max = rng.random() < 0.5
        correct = chain[0] if ask_max else chain[-1]
        word = strong_w if ask_max else weak_w
        prompt = " ".join(edges) + f" The {word} one is"
        pool = [n for n in names if n != correct]
        distractors = rng.sample(pool, 3)
        return Probe(prompt, correct, distractors)

    seen: set[str] = set()
    out: list[Probe] = []
    while len(out) < n_train + n_eval:
        p = one()
        if p.prompt in seen:
            continue
        seen.add(p.prompt)
        out.append(p)
    return out[:n_train], out[n_train:n_train + n_eval]


# ------------------------------------------------------------------ batching / loss / eval -------
def _tok(tokenizer, s: str) -> list[int]:
    return tokenizer(s, add_special_tokens=False)["input_ids"]


def build_examples(probes: list[Probe], tokenizer) -> list[tuple[list[int], list[int]]]:
    """(prompt_ids, answer_ids) for the CORRECT answer — the training target."""
    ex = []
    for p in probes:
        pid, aid = _tok(tokenizer, p.prompt), _tok(tokenizer, p.correct)
        if aid:
            ex.append((pid, aid))
    return ex


def make_batches(examples, *, steps: int, batch: int, seed: int) -> list[list[int]]:
    """A FIXED stream of batches (indices into ``examples``). Feeding the identical stream to both
    the dense and paged trainers removes batch order as a divergence source at a shared seed."""
    rng = random.Random(seed)
    return [[rng.randrange(len(examples)) for _ in range(batch)] for _ in range(steps)]


def _masked_ce(logits: torch.Tensor, ids_pad: torch.Tensor, ans_mask: torch.Tensor) -> torch.Tensor:
    """CE over answer positions only. logits[:, t] predicts ids_pad[:, t+1]; ans_mask marks answer
    token positions (in target space). Matches _avg_answer_logprob indexing."""
    pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
    tgt = ids_pad[:, 1:].reshape(-1)
    keep = ans_mask[:, 1:].reshape(-1)
    return F.cross_entropy(pred[keep], tgt[keep])


def _collate(batch_ex, pad_id: int, device) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
    """Right-pad to the batch max (paged's own layout). Returns (ids_list, ids_pad, ans_mask)."""
    ids_list = [pid + aid for pid, aid in batch_ex]
    Tmax = max(len(s) for s in ids_list)
    B = len(batch_ex)
    ids_pad = torch.full((B, Tmax), pad_id, dtype=torch.long, device=device)
    ans_mask = torch.zeros((B, Tmax), dtype=torch.bool, device=device)
    for i, (pid, aid) in enumerate(batch_ex):
        s = pid + aid
        ids_pad[i, : len(s)] = torch.tensor(s, device=device)
        ans_mask[i, len(pid): len(s)] = True
    return ids_list, ids_pad, ans_mask


@torch.no_grad()
def eval_forced_choice(logits_fn: Callable[[list[list[int]]], torch.Tensor], probes, tokenizer,
                       *, chunk: int = 6) -> tuple[float, list[int]]:
    """Forced-choice: pick the candidate (correct + distractors) with the highest average
    answer-token log-prob. ``logits_fn(ids_list) -> [B, Tmax, V]`` (full vocab; logZ needed).
    Returns (accuracy, per-probe predicted-candidate-index) — the per-probe vector powers the
    paired (McNemar-style) disagreement floor, which stays non-degenerate when accuracy ties."""
    seqs, spans, groups = [], [], []
    for p in probes:
        pid = _tok(tokenizer, p.prompt)
        cand = [p.correct] + p.distractors
        g = []
        for c in cand:
            aid = _tok(tokenizer, c)
            g.append(len(seqs))
            seqs.append(pid + aid)
            spans.append((len(pid), len(pid) + len(aid)))
        groups.append((g, 0))                       # index 0 is always the correct candidate
    scores = [0.0] * len(seqs)
    for s in range(0, len(seqs), chunk):
        ids_list = seqs[s: s + chunk]
        logits = logits_fn(ids_list)
        lp = torch.log_softmax(logits.float(), dim=-1)
        for j, ids in enumerate(ids_list):
            p0, p1 = spans[s + j]
            tok = torch.tensor(ids)
            # logprob of answer token at pos t read from prediction at t-1
            val = sum(float(lp[j, t - 1, tok[t]]) for t in range(p0, p1))
            scores[s + j] = val / max(1, p1 - p0)
    preds: list[int] = []
    correct = 0
    for g, gold in groups:
        best = max(range(len(g)), key=lambda i: scores[g[i]])
        preds.append(best)
        correct += int(best == gold)
    return correct / max(1, len(groups)), preds


def _disagree(preds_a: list[int], preds_b: list[int]) -> float:
    """Fraction of probes where two runs pick a different winner (paired seed/int8 floor)."""
    return sum(int(a != b) for a, b in zip(preds_a, preds_b, strict=True)) / max(1, len(preds_a))


# ------------------------------------------------------------------ dense reference --------------
class _DenseLoRA(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, A: torch.Tensor, B: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = torch.nn.Parameter(A.detach().clone().float())
        self.B = torch.nn.Parameter(B.detach().clone().float())
        self.scale = scale

    def forward(self, x):
        base = self.base(x)
        d = (F.linear(x.float(), self.A) @ self.B.T) * self.scale
        return base + d.to(base.dtype)


def _inject_dense(model, init: dict, layers, targets, alpha: float, rank: int) -> dict:
    """Wrap the HF projections with a LoRA whose A,B are copied from ``init`` (paged's init dict),
    so a shared-seed dense run starts BIT-IDENTICAL to the paged run. Returns {(L,t): module}."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = {}
    scale = alpha / rank
    for L in layers:
        blk = model.model.layers[L]
        for t in targets:
            sub, attr = _HF_ATTR[t]
            base = getattr(getattr(blk, sub), attr)
            ab = init[(L, t)]
            w = _DenseLoRA(base, ab["A"], ab["B"], scale)
            setattr(getattr(blk, sub), attr, w)
            wrapped[(L, t)] = w
    return wrapped


# ------------------------------------------------------------------ RSS sampler ------------------
class _PeakRSS:
    def __init__(self):
        import psutil
        self._p = psutil.Process()
        self.peak = 0.0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._p.memory_info().rss / 1e6)
            time.sleep(0.25)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._th.join(timeout=1.0)


# ------------------------------------------------------------------ delta geometry ---------------
def _delta(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    return (B.detach().float() @ A.detach().float()) * scale       # [out, in]


def _principal_angles_deg(Wa: torch.Tensor, Wb: torch.Tensor, r: int) -> tuple[float, float]:
    """Mean/max principal angle (degrees) between the top-r column spaces of two [out,in] deltas."""
    Ua = torch.linalg.svd(Wa, full_matrices=False).U[:, :r]
    Ub = torch.linalg.svd(Wb, full_matrices=False).U[:, :r]
    s = torch.linalg.svd(Ua.T @ Ub, full_matrices=False).S.clamp(-1.0, 1.0)
    ang = torch.rad2deg(torch.arccos(s))
    return float(ang.mean()), float(ang.max())


def _rel_fro(Wa: torch.Tensor, Wb: torch.Tensor) -> float:
    denom = max(float(Wa.norm()), float(Wb.norm()), 1e-12)
    return float((Wa - Wb).norm() / denom)


def geometry(deltas_a: dict, deltas_b: dict, rank: int) -> dict:
    angs, maxes, rels = [], [], []
    for key in deltas_a:
        Wa, Wb = deltas_a[key], deltas_b[key]
        m, mx = _principal_angles_deg(Wa, Wb, rank)
        angs.append(m)
        maxes.append(mx)
        rels.append(_rel_fro(Wa, Wb))
    return {"mean_principal_angle_deg": float(np.mean(angs)),
            "max_principal_angle_deg": float(np.max(maxes)),
            "mean_rel_fro": float(np.mean(rels)), "n_sites": len(angs)}


# ------------------------------------------------------------------ train loops ------------------
@dataclass
class RunResult:
    eval_acc: float
    preds: list[int] = field(default_factory=list)  # per-probe forced-choice winner index
    deltas: dict = field(default_factory=dict)     # (L,t) -> [out,in] fp32
    s_per_step: float = 0.0
    peak_rss_mb: float = 0.0
    working_set_mb: float = 0.0
    n_trainable: int = 0
    extra: dict = field(default_factory=dict)


def _optimizer(params, lr):
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))


def _sched(step, total, warmup=0.08):
    w = max(1, int(total * warmup))
    return step / w if step < w else 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def run_paged(model, cfg: LoRAConfig, examples, batches, eval_probes, tokenizer, *,
              lr: float, stores_dir=None) -> tuple[RunResult, dict]:
    tr = PagedLoRATrainer(model, cfg, stores_dir=stores_dir)
    init = {k: {"A": v["A"].detach().clone(), "B": v["B"].detach().clone()}
            for k, v in tr.lora.items()}                # capture the shared init for the dense twin
    pad_id = 0
    opt = _optimizer(tr.parameters(), lr)
    total = len(batches)
    t0 = time.time()
    with _PeakRSS() as rss:
        for step, idxs in enumerate(batches):
            batch_ex = [examples[i] for i in idxs]
            ids_list, ids_pad, ans_mask = _collate(batch_ex, pad_id, tr.store.device)
            for g in opt.param_groups:
                g["lr"] = lr * _sched(step, total)
            opt.zero_grad(set_to_none=True)
            logits = tr.forward_logits(ids_list, grad=True)
            loss = _masked_ce(logits, ids_pad, ans_mask)
            loss.backward()
            opt.step()
        elapsed = time.time() - t0
        acc, preds = eval_forced_choice(lambda ids: tr.forward_logits(ids, grad=False),
                                        eval_probes, tokenizer)
    deltas = {k: _delta(v["A"], v["B"], cfg.alpha / cfg.rank) for k, v in tr.lora.items()}
    res = RunResult(eval_acc=acc, preds=preds, deltas=deltas, s_per_step=elapsed / max(1, total),
                    peak_rss_mb=rss.peak, working_set_mb=tr.working_set_mb,
                    n_trainable=tr.n_trainable())
    return res, init


def run_dense(model_name, init: dict, cfg: LoRAConfig, examples, batches, eval_probes, tokenizer, *,
              lr: float) -> RunResult:
    from ..models import load_hf_model
    hf = load_hf_model(model_name, torch_dtype=torch.float32, device="cpu").eval()
    layers = tuple(range(hf.config.num_hidden_layers)) if cfg.layers is None else cfg.layers
    wrapped = _inject_dense(hf, init, layers, cfg.targets, cfg.alpha, cfg.rank)
    params = [w.A for w in wrapped.values()] + [w.B for w in wrapped.values()]
    pad_id = hf.config.pad_token_id or hf.config.eos_token_id or 0
    opt = _optimizer(params, lr)
    total = len(batches)
    t0 = time.time()

    def logits_fn(ids_list):
        _, ids_pad, _ = _collate([(s, []) for s in ids_list], pad_id, "cpu")
        attn = (ids_pad != pad_id).long()
        for i, s in enumerate(ids_list):        # ensure exact-length attn even if a real token == pad
            attn[i, : len(s)] = 1
        pos = torch.arange(ids_pad.shape[1]).unsqueeze(0).expand(ids_pad.shape[0], -1)
        return hf(input_ids=ids_pad, attention_mask=attn, position_ids=pos).logits

    with _PeakRSS() as rss:
        for step, idxs in enumerate(batches):
            batch_ex = [examples[i] for i in idxs]
            ids_list, ids_pad_t, ans_mask = _collate(batch_ex, pad_id, "cpu")
            attn = torch.zeros_like(ids_pad_t)
            for i, s in enumerate(ids_list):
                attn[i, : len(s)] = 1
            pos = torch.arange(ids_pad_t.shape[1]).unsqueeze(0).expand(ids_pad_t.shape[0], -1)
            for g in opt.param_groups:
                g["lr"] = lr * _sched(step, total)
            opt.zero_grad(set_to_none=True)
            logits = hf(input_ids=ids_pad_t, attention_mask=attn, position_ids=pos).logits
            loss = _masked_ce(logits, ids_pad_t, ans_mask)
            loss.backward()
            opt.step()
        elapsed = time.time() - t0
        acc, preds = eval_forced_choice(logits_fn, eval_probes, tokenizer)
    deltas = {k: _delta(w.A, w.B, cfg.alpha / cfg.rank) for k, w in wrapped.items()}
    params_mb = sum(p.numel() for p in hf.parameters()) * 4 / 1e6
    res = RunResult(eval_acc=acc, preds=preds, deltas=deltas, s_per_step=elapsed / max(1, total),
                    peak_rss_mb=rss.peak, n_trainable=sum(p.numel() for p in params),
                    extra={"dense_param_mb": params_mb})
    return res       # hf (~2 GB fp32) is a local ⇒ freed on return; not captured past this point


# ------------------------------------------------------------------ the gate ---------------------
def run_parity(model_name="qwen2.5-0.5b", *, rank=8, alpha=16.0, targets=("q", "v"),
               n_train=160, n_eval=48, steps=40, batch=8, lr=2e-4,
               dense_seeds=(0, 1, 2), stores_dir=None, verbose=True) -> dict:
    from ..models import load_tokenizer
    tok = load_tokenizer(model_name)
    seed0 = dense_seeds[0]

    train_p, eval_p = make_comparison_bank(n_train, n_eval, seed=seed0)
    ex = build_examples(train_p, tok)

    def log(*a):
        if verbose:
            print(*a, flush=True)

    def batches_for(seed):
        return make_batches(ex, steps=steps, batch=batch, seed=seed)

    # base (untrained) accuracy for context
    cfg0 = LoRAConfig(rank=rank, alpha=alpha, targets=targets, seed=seed0)
    base_tr = PagedLoRATrainer(model_name, cfg0, stores_dir=stores_dir)
    base_acc, _ = eval_forced_choice(
        lambda ids: base_tr.forward_logits(ids, grad=False), eval_p, tok)
    log(f"[base] eval_acc={base_acc:.3f}")
    base_tr = None       # free the base store before the training runs (kept resident otherwise)

    # PAGED run C at seed0.
    log(f"[paged  C] seed {seed0} ...")
    paged_C, init0 = run_paged(model_name, cfg0, ex, batches_for(seed0), eval_p, tok,
                               lr=lr, stores_dir=stores_dir)
    log(f"[paged  C] eval_acc={paged_C.eval_acc:.3f} s/step={paged_C.s_per_step:.2f} "
        f"peak_rss={paged_C.peak_rss_mb:.0f}MB working_set={paged_C.working_set_mb:.1f}MB")

    # DENSE run at seed0 shares C's BIT-IDENTICAL init ⇒ any A-vs-C divergence is purely int8.
    log(f"[dense  {seed0}] (shared init with C) ...")
    dense_runs = {seed0: run_dense(model_name, init0, cfg0, ex, batches_for(seed0), eval_p, tok, lr=lr)}
    log(f"[dense  {seed0}] eval_acc={dense_runs[seed0].eval_acc:.3f} "
        f"s/step={dense_runs[seed0].s_per_step:.2f} peak_rss={dense_runs[seed0].peak_rss_mb:.0f}MB")

    # DENSE floor runs at the other seeds (own init + batch order) → run-to-run noise floor.
    for sd in dense_seeds[1:]:
        cfg = LoRAConfig(rank=rank, alpha=alpha, targets=targets, seed=sd)
        ftr = PagedLoRATrainer(model_name, cfg, stores_dir=stores_dir)
        init = {k: {"A": v["A"].detach().clone(), "B": v["B"].detach().clone()}
                for k, v in ftr.lora.items()}
        del ftr
        log(f"[dense  {sd}] (seed-noise floor) ...")
        dense_runs[sd] = run_dense(model_name, init, cfg, ex, batches_for(sd), eval_p, tok, lr=lr)
        log(f"[dense  {sd}] eval_acc={dense_runs[sd].eval_acc:.3f}")

    dA = dense_runs[seed0]
    others = [dense_runs[s] for s in dense_seeds[1:]]

    # --- GEOMETRY: adapter-subspace principal angle, int8 (A-vs-C) vs seed floor (dense-dense) ---
    geom_test = geometry(dA.deltas, paged_C.deltas, rank)
    geom_floor_pairs = [geometry(dA.deltas, o.deltas, rank) for o in others]
    geom_floor_ang = float(np.mean([g["mean_principal_angle_deg"] for g in geom_floor_pairs]))
    geom_ratio = geom_test["mean_principal_angle_deg"] / max(1e-9, geom_floor_ang)

    # --- OUTCOME: paired per-probe disagreement, int8 (A-vs-C) vs seed floor (dense-dense) -------
    # (aggregate accuracy ties at coarse n_eval and gives a degenerate 0 floor; the paired
    #  disagreement is the sensitive, non-degenerate measure.)
    out_test = _disagree(dA.preds, paged_C.preds)
    out_floor_vals = [_disagree(dA.preds, o.preds) for o in others]
    out_floor = float(np.mean(out_floor_vals)) if out_floor_vals else 0.0
    out_ratio = out_test / max(1e-9, out_floor)
    acc_gap = abs(paged_C.eval_acc - dA.eval_acc)

    geometry_clean = geom_ratio <= 2.0
    # outcome parity: int8 churn within ~2x the seed churn (paired), OR within eval granularity.
    outcome_parity = (out_test <= 2.0 * out_floor + 1e-9) or (out_test <= 1.0 / n_eval + 1e-9)

    result = {
        "model": model_name, "rank": rank, "alpha": alpha, "targets": list(targets),
        "n_train": n_train, "n_eval": n_eval, "steps": steps, "batch": batch, "lr": lr,
        "dense_seeds": list(dense_seeds), "n_sites": geom_test["n_sites"],
        "n_trainable": paged_C.n_trainable,
        "acc": {"base": base_acc, "paged_C": paged_C.eval_acc,
                **{f"dense_{s}": dense_runs[s].eval_acc for s in dense_seeds}},
        "outcome": {"paged_vs_dense_acc_gap": acc_gap,
                    "paired_disagree_test_A_vs_C": out_test,
                    "paired_disagree_floor_dense_dense": out_floor,
                    "paired_disagree_floor_vals": out_floor_vals,
                    "ratio_test_over_floor": out_ratio, "parity": bool(outcome_parity)},
        "geometry": {"test_A_vs_C": geom_test,
                     "floor_dense_dense_mean_angle_deg": geom_floor_ang,
                     "floor_pairs": geom_floor_pairs,
                     "ratio_test_over_floor": geom_ratio, "clean": bool(geometry_clean)},
        "throughput": {"paged_s_per_step": paged_C.s_per_step, "dense_s_per_step": dA.s_per_step,
                       "paged_over_dense": paged_C.s_per_step / max(1e-9, dA.s_per_step)},
        "memory_mb": {"paged_peak_rss": paged_C.peak_rss_mb, "paged_working_set": paged_C.working_set_mb,
                      "dense_peak_rss": dA.peak_rss_mb,
                      "dense_param_fp32": dA.extra.get("dense_param_mb"),
                      "adapter_params": paged_C.n_trainable * 4 / 1e6,
                      "adam_state": paged_C.n_trainable * 8 / 1e6},
        "verdict": ("geometry+capability" if (outcome_parity and geometry_clean)
                    else "capability-only" if outcome_parity else "FAIL"),
    }
    return result


def _main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--targets", default="q,v")
    ap.add_argument("--n-train", type=int, default=160)
    ap.add_argument("--n-eval", type=int, default=48)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run_parity(a.model, rank=a.rank, targets=tuple(a.targets.split(",")),
                     n_train=a.n_train, n_eval=a.n_eval, steps=a.steps, batch=a.batch, lr=a.lr)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    _main()
