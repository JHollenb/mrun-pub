"""Shared engine implementation.

``BaseEngine`` holds the backend-agnostic methods every engine needs — tokenization,
prose prompts, forced-choice scoring, activation-tape capture, write-norm, and scalar
batch fallbacks. Backends (``HFEngine``, ``PagedEngine``, MLX, ANE) subclass it and
implement only the primitives that differ:

  required attributes set in __init__:  name, tokenizer, backend, supports_batch,
                                         n_layer, inter, hidden, working_set_mb
  required methods:                      logits, forward_acts, forward_patched, down_weight

Everything else is inherited. This keeps the single-definition discipline the discovery
engine had (one base class, not the scoring/capture logic copy-pasted per backend).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np
import torch

from .base import (
    EngineCapabilities,
    PatchBuilder,
    avg_answer_logprob,
    finalize_forced_choice_row,
    summarize_forced_choice_rows,
)


class BaseEngine:
    # Subclasses set these; defaults keep static checkers and bare instances happy.
    backend: str = "base"
    supports_batch: bool = False
    name: str = ""
    arch: str = "?"                  # model_type family (qwen2/llama/qwen3/gpt_neox/mamba/…)
    n_layer: int = 0
    inter: int = 0
    hidden: int = 0
    working_set_mb: float | None = None
    tokenizer: Any = None

    # -- lifecycle ---------------------------------------------------------------
    def __enter__(self) -> BaseEngine:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def close(self) -> None:
        return None

    # -- tokenization ------------------------------------------------------------
    def encode(self, prompts: list[str], *, add_special_tokens: bool = False) -> list[np.ndarray]:
        encoded = self.tokenizer(prompts, add_special_tokens=add_special_tokens)
        return [np.asarray(ids, dtype=np.int64) for ids in encoded["input_ids"]]

    def prose_ids(self, max_len: int = 64) -> list[np.ndarray]:
        prompts = [
            "The capital of France is",
            "In a short proof, the key idea is",
            "A reliable experiment should",
            "When the model answers carefully, it",
        ]
        return [ids[:max_len] for ids in self.encode(prompts) if len(ids)]

    # -- batch fallbacks (overridden by backends that truly fuse) ----------------
    def _count_scalar_fallback(self, method: str, rows: int) -> None:
        """Record a batch-API call that ran as a per-row loop.

        These fallbacks are correct but unfused: a caller that hands N rows to a batch
        method and silently gets N weight streams is the exact shape of the repeated
        per-item-eval incident (44% GPU util, ~2 h wall vs ~15 min at batch 64). Counting
        them here makes the degradation legible in execution evidence instead of visible
        only as a slow run. ``MRUN_SCALAR_FALLBACK_WARN=1`` also prints once per method.
        """
        stats = self.__dict__.setdefault("_scalar_fallbacks", {})
        entry = stats.setdefault(method, {"calls": 0, "rows": 0})
        entry["calls"] += 1
        entry["rows"] += int(rows)
        if entry["calls"] == 1 and os.environ.get("MRUN_SCALAR_FALLBACK_WARN") == "1":
            print(
                f"[mrun] scalar fallback: {self.backend}.{method} ran {rows} rows as a "
                f"per-row loop (backend does not fuse this call)",
                file=sys.stderr,
            )

    def scalar_fallback_stats(self) -> dict[str, dict[str, int]]:
        """Per-method {calls, rows} for batch APIs that degraded to per-row loops."""
        return dict(self.__dict__.get("_scalar_fallbacks", {}))

    def logits_batch(self, ids_list: list[np.ndarray]) -> list[torch.Tensor]:
        self._count_scalar_fallback("logits_batch", len(ids_list))
        return [self.logits(ids) for ids in ids_list]

    def forward_acts_batch(
        self, ids_list: list[np.ndarray]
    ) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        self._count_scalar_fallback("forward_acts_batch", len(ids_list))
        return [self.forward_acts(ids) for ids in ids_list]

    def hidden_states_batch(self, ids_list: list[np.ndarray]) -> list[list[torch.Tensor]]:
        """Scalar semantic fallback for residual capture.

        Backends report ``residual_tap_batch=True`` only when they override this with a
        genuinely fused implementation. Keeping the fallback here makes the public surface
        backward-compatible without pretending that a streamed scalar engine amortizes work.
        """

        hidden_states = getattr(self, "hidden_states", None)
        if not callable(hidden_states):
            raise NotImplementedError(f"{self.backend} does not implement residual capture")
        self._count_scalar_fallback("hidden_states_batch", len(ids_list))
        return [hidden_states(ids) for ids in ids_list]

    def forward_patched_batch(
        self,
        ids_list: list[np.ndarray],
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        head_patch_ops_by_layer: dict[int, list] | None = None,
        resid_patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
    ) -> list[tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]]:
        """Per-row ``(logits, acts, captured)``; one shared patch map applies to every row.
        Scalar fallback — engines that truly fuse (HF padded batch, paged weight stream)
        override this."""
        kwargs: dict[str, Any] = {
            "patch_ops_by_layer": patch_ops_by_layer,
            "selected_maps": selected_maps,
            "collect_acts": collect_acts,
        }
        if head_patch_ops_by_layer:
            # engines whose forward_patched has no head tap must fail TYPED, not with a
            # confusing TypeError from an unexpected kwarg (adversarial-review F7)
            import inspect

            sig = inspect.signature(self.forward_patched)
            if "head_patch_ops_by_layer" not in sig.parameters:
                raise NotImplementedError(
                    f"{self.backend} forward_patched does not implement head-output patches"
                )
            kwargs["head_patch_ops_by_layer"] = head_patch_ops_by_layer
        if resid_patch_ops_by_layer:
            import inspect

            sig = inspect.signature(self.forward_patched)
            if "resid_patch_ops_by_layer" not in sig.parameters:
                raise NotImplementedError(
                    f"{self.backend} forward_patched does not implement residual patches"
                )
            kwargs["resid_patch_ops_by_layer"] = resid_patch_ops_by_layer
        self._count_scalar_fallback("forward_patched_batch", len(ids_list))
        return [self.forward_patched(ids, **kwargs) for ids in ids_list]

    # -- capabilities (recorders branch on this, never on backend strings) --------
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            logits=True,
            logits_batch=bool(self.supports_batch),
            mlp_acts=True,
            mlp_acts_batch=bool(self.supports_batch),
        )

    # -- write norm (per-neuron L2 of the down/write columns, concat over layers) -
    def write_norm(self) -> np.ndarray:
        norms: list[np.ndarray] = []
        for layer in range(self.n_layer):
            Wd = self.down_weight(layer)            # [hidden, inter]; column n = neuron n's write
            norms.append(Wd.pow(2).sum(dim=0).sqrt().detach().cpu().numpy())
        return np.concatenate(norms).astype(np.float32)

    # -- forced-choice scoring ---------------------------------------------------
    def score_forced_choice_many(
        self,
        probes: list[dict[str, Any]],
        *,
        patch_builder: PatchBuilder | None = None,
        argmax_only: bool = False,
    ) -> dict[str, Any]:
        # Subset fast path (opt-in): for SINGLE-TOKEN forced choice the full-vocab softmax
        # denominator logZ at the last prompt position is shared by a probe's candidates, so it
        # CANCELS — the winner and margin are exact from candidate-only lm_head columns and the
        # 544 MB unembedding is never streamed (lower RAM floor). Only the absolute avg_logprob is
        # unnormalized, so this is gated behind argmax_only: pass it when you need winner/margin,
        # not the calibrated logprob. Requires a backend that implements the subset method (paged);
        # any other case falls through to the exact full path below.
        if argmax_only and patch_builder is None:
            subset = getattr(self, "score_forced_choice_argmax_subset", None)
            if subset is not None and self._all_single_token(probes):
                t0 = time.perf_counter()
                result = self._flag_low_margin_rows(subset(probes))
                self._note_scoring("subset_head", len(probes), len(result.get("rows", [])),
                                   time.perf_counter() - t0)
                return result

        # Batched fast path: with no per-probe patch and a batch-fusing backend, score every
        # (prompt+candidate) row through ONE weight stream instead of a forward per candidate.
        # Measured ~16x on Qwen2.5-0.5B; the avg-logprob numbers (and thus winners) are identical
        # to the scalar loop — only the weight-stream amortization differs. Patched scoring stays
        # scalar (per-probe patch ops do not fuse across the batch).
        if patch_builder is None and getattr(self, "supports_batch", False):
            # Prefix-KV variant (backends that implement it): the prompt is forwarded ONCE per
            # probe with a KV cache and every candidate reuses it, instead of recomputing the
            # prompt inside each (prompt+candidate) row. The method falls back to the flat
            # batched path internally when the token savings are too small to matter.
            prefix_kv = getattr(self, "score_forced_choice_prefix_kv", None)
            t0 = time.perf_counter()
            if prefix_kv is not None and os.environ.get("MRUN_PREFIX_KV", "1") != "0":
                result = self._flag_low_margin_rows(prefix_kv(probes))
                path = "prefix_kv"
            else:
                result = self._flag_low_margin_rows(self._score_forced_choice_batched(probes))
                path = "batched"
            self._note_scoring(path, len(probes), len(result.get("rows", [])),
                               time.perf_counter() - t0)
            return result

        _t_scalar = time.perf_counter()
        rows = []
        for idx, probe in enumerate(probes):
            prompt = str(probe["prompt"])
            candidates = _probe_candidates(probe)
            prompt_ids = self.encode([prompt], add_special_tokens=False)[0].tolist()
            scored = []
            for cand in candidates:
                answer = str(cand["answer"])
                answer_ids = self.encode([answer], add_special_tokens=False)[0].tolist()
                ids = np.asarray(prompt_ids + answer_ids, dtype=np.int64)
                ops = patch_builder(probe, cand) if patch_builder else None
                logits = self.forward_patched(ids, patch_ops_by_layer=ops)[0] if ops else self.logits(ids)
                scored.append(
                    {
                        "answer": answer,
                        "avg_logprob": avg_answer_logprob(logits, len(prompt_ids), answer_ids),
                    }
                )
            rows.append(
                finalize_forced_choice_row(
                    scored,
                    probe_id=str(probe.get("probe_id", idx)),
                    category=probe.get("category"),
                )
            )
        result = self._flag_low_margin_rows(
            {"summary": summarize_forced_choice_rows(rows), "rows": rows}
        )
        self._note_scoring("scalar", len(probes), len(rows), time.perf_counter() - _t_scalar)
        return result

    def _note_scoring(self, path: str, probes: int, rows: int, seconds: float) -> None:
        """Accumulate per-path scoring timing so runs are self-describing.

        Auto-emitted engine reports (I48) capture WHAT executed but, without this, no stage
        timing — so the leaderboard could only be filled by code that opted in, which in
        practice means benchmarks rather than real runs. Recording it at the scoring boundary
        makes ordinary work populate it: which path served the probes (subset / prefix-KV /
        batched / scalar), how many, and how long.
        """
        stats = self.__dict__.setdefault("_scoring_stats", {})
        e = stats.setdefault(path, {"calls": 0, "probes": 0, "rows": 0, "seconds": 0.0})
        e["calls"] += 1
        e["probes"] += int(probes)
        e["rows"] += int(rows)
        e["seconds"] += float(seconds)

    def scoring_stats(self) -> dict[str, dict[str, Any]]:
        """Per-scoring-path {calls, probes, rows, seconds} with derived probes/s."""
        out = {}
        for path, e in self.__dict__.get("_scoring_stats", {}).items():
            d = dict(e)
            d["seconds"] = round(float(e["seconds"]), 4)
            if e["seconds"] > 0:
                d["probes_per_s"] = round(e["probes"] / e["seconds"], 1)
            out[path] = d
        return out

    def _flag_low_margin_rows(self, result: dict[str, Any]) -> dict[str, Any]:
        """Mark rows whose winner is not resolvable above this backend's own numerical error.

        An approximate backend is only argmax-exact where the true margin CLEARS its error.
        The Core ML path measures max|dmargin| 0.462 vs paged (2026-07-24), so a probe with a
        0.2-margin can report the wrong winner on text that looks perfectly natural — the old
        "safe on real text" heuristic does not cover it, because naturalness and margin are
        independent. Rows below the floor get ``margin_below_backend_error``; the summary gets
        a count. Nothing is silently dropped or corrected: the caller decides whether to
        re-score those probes on an exact backend.
        """
        floor = float(getattr(self, "margin_floor", 0.0) or 0.0)
        if floor <= 0.0:
            return result
        flagged = 0
        for row in result.get("rows", []):
            if abs(float(row.get("margin", 0.0))) < floor:
                row["margin_below_backend_error"] = True
                flagged += 1
        if flagged:
            summary = result.setdefault("summary", {})
            summary["rows_below_backend_margin_error"] = flagged
            summary["backend_margin_floor"] = floor
        return result

    def _all_single_token(self, probes: list[dict[str, Any]]) -> bool:
        """True iff every candidate of every probe is a single token under this tokenizer — the
        case where the softmax logZ cancels exactly and the subset path is valid."""
        for probe in probes:
            for cand in _probe_candidates(probe):
                if len(self.encode([str(cand["answer"])], add_special_tokens=False)[0]) != 1:
                    return False
        return True

    def _score_forced_choice_batched(self, probes: list[dict[str, Any]]) -> dict[str, Any]:
        # Flatten every probe's candidates into one list of (prompt+answer) rows, run a single
        # logits_batch, then regroup. Same finalize/summarize as the scalar path.
        flat_ids: list[np.ndarray] = []
        layout: list[tuple[int, str, int, list[int]]] = []     # (probe_idx, answer, prompt_len, answer_ids)
        meta: list[tuple[str, str | None]] = []                # per probe: (probe_id, category)
        for idx, probe in enumerate(probes):
            prompt = str(probe["prompt"])
            prompt_ids = self.encode([prompt], add_special_tokens=False)[0].tolist()
            for cand in _probe_candidates(probe):
                answer = str(cand["answer"])
                answer_ids = self.encode([answer], add_special_tokens=False)[0].tolist()
                flat_ids.append(np.asarray(prompt_ids + answer_ids, dtype=np.int64))
                layout.append((idx, answer, len(prompt_ids), answer_ids))
            meta.append((str(probe.get("probe_id", idx)), probe.get("category")))

        logits_rows = self.logits_batch(flat_ids)

        scored_by_probe: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(probes))}
        for (probe_idx, answer, prompt_len, answer_ids), logits in zip(layout, logits_rows):
            scored_by_probe[probe_idx].append(
                {"answer": answer, "avg_logprob": avg_answer_logprob(logits, prompt_len, answer_ids)}
            )
        rows = [
            finalize_forced_choice_row(scored_by_probe[i], probe_id=meta[i][0], category=meta[i][1])
            for i in range(len(probes))
        ]
        return {"summary": summarize_forced_choice_rows(rows), "rows": rows}

    def capture_selected_activation_tape(
        self,
        ids_list: list[np.ndarray],
        selected_maps: dict[int, dict[str, Any]],
    ) -> dict[int, list[torch.Tensor]]:
        out: dict[int, list[torch.Tensor]] = {layer: [] for layer in selected_maps}
        for ids in ids_list:
            _logits, _acts, captured = self.forward_patched(ids, selected_maps=selected_maps)
            for layer, tensor in captured.items():
                out[layer].append(tensor)
        return out


def _probe_candidates(probe: dict[str, Any]) -> list[dict[str, Any]]:
    if "answers" in probe:
        return [{"answer": str(a)} if not isinstance(a, dict) else dict(a) for a in probe["answers"]]
    correct = probe.get("correct", probe.get("answer"))
    distractors = probe.get("distractors", [])
    if correct is None:
        raise ValueError("probe must provide 'answers' or 'correct'/'answer'")
    return [{"answer": str(correct)}, *[{"answer": str(x)} for x in distractors]]
