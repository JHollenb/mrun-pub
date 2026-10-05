"""Paged twins of the dense causal-leg helpers — the CAUSAL-LEG PORT.

The capability signature's FORWARD / forced-choice legs already run on any engine
(``ctx.eng.score_forced_choice_many`` — the paged engine serves it directly). Its CAUSAL
legs (present-detection, dependence, specificity, probe-R² regressions, head/direction
ablations) still call dense helpers against a RAW HF module: ``rd.filter_prompts``,
``rd.find_induction_heads``, ``rd.ablate_heads_logprob``, ``rd.answer_logprob(_with_hooks)``,
and ``model(..., output_hidden_states=True)`` / ``register_forward_hook`` interventions.
That HF module must be resident in full — the exact cost the paged engine deletes.

This module is the ADAPTER (not new kernels): every dense helper is re-expressed as a call
onto the paged engine's already-unit-tested intervention surface —

    rd.ablate_heads_logprob          -> forward_patched(head_patch_ops_by_layer=...)   ("zero")
    rd.answer_logprob_with_hooks     -> forward_patched(resid_patch_ops_by_layer=...)  ("proj_remove")
        (dir-removal)                        or (patch_ops_by_layer=...)               ("zero", mlp)
    model(output_hidden_states=True) -> PagedEngine.hidden_states(ids)                 (collect_hs)
    forward_logits_and_attns         -> PagedEngine.forward_attns(ids)                 (collect_attn)
    filter_prompts / answer_logprob  -> PagedEngine.logits(ids)

so the whole capability-signature vector can be scored against a ``PagedEngine`` — O(largest
matrix) RAM, no full HF module — on models a 16GB card can't hold dense.

The function names + argument shapes MIRROR ``routing_discriminator`` (``rd``) so a caller can
swap ``rd.<fn>(ctx.model, ctx.tok, ...)`` for ``PagedCausal(ctx.eng).<fn>(...)`` mechanically.

Numerics: int8 weights ⇒ each Δlogprob matches the dense fp32 oracle only up to quant tolerance
(offset cancels in a delta, so the ablation *deltas* are tight ~1e-2). Two documented caveats:
  * SIGN FRAGILITY. A specificity/dependence delta whose true magnitude is near zero can flip
    sign under int8 (the same small-signed-quantity fragility as bf16). The parity harness must
    MEASURE and REPORT the sign-disagreement rate against dense fp32 — a documented tolerance,
    never a silent pass (see ``tests/test_causal_paged.py``).
  * HIDDEN-STATE INDEXING. ``hidden_states`` reproduces HF's ``[L+1, hidden]`` convention
    including the final RMSNorm on the top entry (see ``PagedEngine.hidden_states``).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..engine.paged import PagedEngine


def _lp_last(logits: torch.Tensor, aid: int) -> float:
    """log-softmax at the final position, upcast to fp32 (int8/bf16 log_softmax loses digits)."""
    return float(torch.log_softmax(logits[-1].float(), dim=-1)[aid].item())


class PagedCausal:
    """Adapter binding the dense causal helpers to a :class:`PagedEngine`.

    Every method takes the SAME arguments as its ``rd.*`` twin minus the leading
    ``(model, tokenizer)`` pair (the engine already carries both). Prompt triples keep the
    ``(prompt, expected, answer_id)`` shape the causal legs consume, so no leg needs reshaping.
    """

    def __init__(self, eng: PagedEngine):
        if not isinstance(eng, PagedEngine):
            raise TypeError(f"PagedCausal needs a PagedEngine, got {type(eng).__name__}")
        self.eng = eng
        self.tok = eng.tokenizer
        c = eng.cfg
        self.n_layer = int(c["num_hidden_layers"])
        self.n_head = int(c["num_attention_heads"])
        self.d_head = int(c["head_dim"])

    # ------------------------------------------------------------------ ids helpers
    def _ids(self, prompt: str) -> np.ndarray:
        return np.asarray(self.tok(prompt)["input_ids"], dtype=np.int64)

    # ------------------------------------------------------------------ logits / logprob
    def answer_logprob(self, prompt_ids: np.ndarray, answer_token_id: int) -> float:
        """Clean logprob of ``answer_token_id`` at the final position (``rd.answer_logprob``)."""
        return _lp_last(self.eng.logits(np.asarray(prompt_ids, np.int64)), answer_token_id)

    def answer_logprob_patched(
        self,
        prompt_ids: np.ndarray,
        answer_token_id: int,
        *,
        head_patch_ops_by_layer: dict[int, list] | None = None,
        resid_patch_ops_by_layer: dict[int, list] | None = None,
        patch_ops_by_layer: dict[int, list] | None = None,
    ) -> float:
        """Patched logprob — the paged analog of ``rd.answer_logprob_with_hooks``. The dense
        hooks map onto the paged intervention surface: head-zero -> ``head_patch_ops_by_layer``;
        residual rank-1 dir-removal -> ``resid_patch_ops_by_layer`` (``proj_remove``); MLP-zero
        -> ``patch_ops_by_layer`` (``zero`` on the down-proj input columns)."""
        logits, _acts, _cap = self.eng.forward_patched(
            np.asarray(prompt_ids, np.int64),
            head_patch_ops_by_layer=head_patch_ops_by_layer,
            resid_patch_ops_by_layer=resid_patch_ops_by_layer,
            patch_ops_by_layer=patch_ops_by_layer,
        )
        return _lp_last(logits, answer_token_id)

    # ------------------------------------------------------------------ prompt filtering
    def filter_prompts(
        self,
        candidates: list[tuple[str, str]],
        condition_name: str,
        max_prompts: int = 6,
    ) -> list[tuple[str, str, int]]:
        """Keep candidates whose top-1 next token == the (single-token) expected — greedy filter,
        exact mirror of ``rd.filter_prompts`` but scored through the paged forward."""
        kept: list[tuple[str, str, int]] = []
        for prompt, expected in candidates:
            exp_ids = self.tok.encode(expected)
            if len(exp_ids) != 1:
                exp_ids2 = self.tok.encode(expected.strip())
                if len(exp_ids2) == 1:
                    exp_ids = exp_ids2
                else:
                    continue
            answer_id = exp_ids[0]
            top1 = int(self.eng.logits(self._ids(prompt))[-1].argmax().item())
            if top1 == answer_id:
                kept.append((prompt, expected, answer_id))
            if len(kept) >= max_prompts:
                break
        return kept

    # ------------------------------------------------------------------ attention / induction
    def forward_logits_and_attns(self, prompt_ids: np.ndarray):
        """(logits[T,V], list[nL] of [nH,T,T]) — the paged twin of ``rd.forward_logits_and_attns``
        (which returns HF ``output_attentions`` minus its batch dim). int8 weights ⇒ attention is
        approximate vs fp32 (top-1 mass preserved)."""
        return self.eng.forward_attns(np.asarray(prompt_ids, np.int64))

    def find_induction_heads(
        self,
        copy_prompts: list[tuple[str, str, int]],
        top_n: int = 4,
    ) -> list[tuple[int, int]]:
        """Top-``top_n`` (layer, head) by attention mass, from the final query position onto the
        in-context answer token, averaged over copy prompts (exact port of
        ``rd.find_induction_heads`` — same fallback when no prompt has the answer in context)."""
        attn_on_answer = np.zeros((self.n_layer, self.n_head))
        count = 0
        for prompt, _expected, answer_id in copy_prompts:
            seq = self._ids(prompt).tolist()
            answer_pos = next((i for i, t in enumerate(seq[:-1]) if t == answer_id), None)
            if answer_pos is None:
                continue
            _, attns = self.forward_logits_and_attns(np.asarray(seq, np.int64))
            for l_idx in range(self.n_layer):
                a = attns[l_idx][:, -1, :]                       # [nH, k_len]
                attn_on_answer[l_idx] += a[:, answer_pos].detach().float().cpu().numpy()
            count += 1
        if count == 0:
            return [(2, 1), (3, 6), (3, 2), (8, 7)]              # rd's circuit-graph fallback
        attn_on_answer /= count
        top = np.argsort(attn_on_answer.flatten())[::-1][:top_n]
        return [(int(i // self.n_head), int(i % self.n_head)) for i in top]

    # ------------------------------------------------------------------ head ablation
    def ablate_heads_logprob(
        self,
        prompts: list[tuple[str, str, int]],
        heads_to_ablate: list[tuple[int, int]],
        label: str = "",
    ) -> list[float]:
        """Δlogprob (ablated − clean) per prompt from zeroing the given (layer, head) value-flows
        BEFORE ``o_proj`` — the paged twin of ``rd.ablate_heads_logprob`` (whose dense pre-hook
        zeros each head's columns in the o_proj input). Heads are grouped per layer into one
        ``head_patch_ops_by_layer`` map so the whole set is ablated in a single forward."""
        by_layer: dict[int, list[int]] = defaultdict(list)
        for (l, h) in heads_to_ablate:
            by_layer[l].append(h)
        head_patch = {l: [("zero", hs, None)] for l, hs in by_layer.items()}
        deltas = []
        for prompt, _expected, aid in prompts:
            ids = self._ids(prompt)
            lp_clean = self.answer_logprob(ids, aid)
            lp_abl = self.answer_logprob_patched(ids, aid, head_patch_ops_by_layer=head_patch)
            deltas.append(lp_abl - lp_clean)
        return deltas

    def ablate_random_heads_logprob(
        self,
        prompts: list[tuple[str, str, int]],
        n_heads_to_ablate: int,
        exclude_heads: set[tuple[int, int]],
        rng: np.random.Generator,
        n_repeats: int = 3,
    ) -> list[float]:
        """Size-matched random-head null: mean Δlogprob over ``n_repeats`` draws of
        ``n_heads_to_ablate`` random heads not in ``exclude_heads`` (port of
        ``rd.ablate_random_heads_logprob``; identical rng.choice draw order)."""
        all_candidates = [(l, h) for l in range(self.n_layer) for h in range(self.n_head)
                          if (l, h) not in exclude_heads]
        per_prompt_sums = [0.0] * len(prompts)
        for _ in range(n_repeats):
            idx_arr = rng.choice(
                len(all_candidates), size=min(n_heads_to_ablate, len(all_candidates)), replace=False
            )
            chosen = [all_candidates[int(i)] for i in idx_arr]
            for i, d in enumerate(self.ablate_heads_logprob(prompts, chosen, "random")):
                per_prompt_sums[i] += d
        return [s / n_repeats for s in per_prompt_sums]

    # ------------------------------------------------------------------ direction / MLP ablation
    def remove_direction_logprob(
        self,
        prompt_ids: np.ndarray,
        answer_token_id: int,
        layer: int,
        u: np.ndarray | torch.Tensor,
    ) -> float:
        """Patched logprob after removing the rank-1 projection onto unit direction ``u`` from the
        residual stream at ``layer``'s output — the paged twin of the dense ``_make_dir_removal_hook``
        (post-hook on the decoder layer, ``h -> h - (h·u)u``). ``u`` is re-normalized in-kernel, so
        the size-matched random-direction null uses the same call with a random ``u``."""
        u_np = u.detach().cpu().numpy() if isinstance(u, torch.Tensor) else np.asarray(u)
        return self.answer_logprob_patched(
            prompt_ids, answer_token_id,
            resid_patch_ops_by_layer={int(layer): [("proj_remove", u_np, None)]},
        )

    def ablate_mlp_layer_logprob(
        self,
        prompt_ids: np.ndarray,
        answer_token_id: int,
        layer: int,
    ) -> float:
        """Patched logprob after zeroing the ENTIRE MLP output at ``layer`` — twin of the dense
        ``make_mlp_zero_hook`` (which zeros the MLP module output). Zeroing every down-proj input
        column drives the MLP write to 0, the same residual contribution as zeroing its output."""
        cols = list(range(self.eng.inter))
        return self.answer_logprob_patched(
            prompt_ids, answer_token_id,
            patch_ops_by_layer={int(layer): [("zero", cols, None)]},
        )

    # ------------------------------------------------------------------ hidden states (probe-R²)
    def collect_hidden_states(self, prompt_ids: np.ndarray) -> list[torch.Tensor]:
        """HF-layout residual-stream hidden states ``[nL+1]`` of ``[T, hidden]`` (final RMSNorm on
        the top entry) — the paged replacement for ``model(..., output_hidden_states=True)`` that
        the depth-probe indexes as ``hidden_states[l+1] = output of layer l``. See
        :meth:`PagedEngine.hidden_states` for the indexing gotcha this resolves."""
        return self.eng.hidden_states(np.asarray(prompt_ids, np.int64))
