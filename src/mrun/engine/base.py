"""Backend interface and shared scoring/patching helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol

import numpy as np
import torch

PatchBuilder = Callable[[dict[str, Any], dict[str, Any]], dict[int, list] | None]

#: Sentinel for "leave this setting alone" in an engine's scoped-tap context manager (the
#: ``ablation`` methods). Distinct from None, which explicitly CLEARS a setting — so a component
#: ablation can nest inside an embedding ablation without silently cancelling it.
KEEP: Any = object()


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine's taps actually support — recorders branch on THIS, never on backend
    strings. A missing capability means the semantic tap is unavailable (the method raises);
    it is recorded in a payload's ``capability_gaps``, not silently approximated."""

    logits: bool = True
    logits_batch: bool = False
    mlp_acts: bool = True
    mlp_acts_batch: bool = False
    mlp_patch: bool = False
    mlp_patch_batch: bool = False
    head_patch: bool = False
    head_patch_batch: bool = False
    selected_capture: bool = False
    attentions: bool = False
    residual_tap: bool = False
    residual_tap_batch: bool = False
    residual_patch: bool = False
    raw_model: bool = False          # eng.model/eng.resolver exist (dCE/gradxact/weight ablation)
    autograd: bool = False
    exact_reference: bool = False    # fp32 whole-model forward (the correctness oracle)
    approximate_quantized: bool = False
    generation: bool = False
    generation_batch: bool = False
    persistent_kv: bool = False
    transactional_kv: bool = False
    speculative_blocks: bool = False
    compact_fused_weights: bool = False
    grouped_moe: bool = False
    paged_experts: bool = False
    cache_telemetry: bool = False
    compiled_workplan: bool = False
    graph_replay: bool = False
    resident_graph_rebinding: bool = False
    continuous_batching: bool = False
    profile_driven_fusion: bool = False
    captured_decode: bool = False
    target_aligned_multi_token: bool = False
    fp8_execution: bool = False
    int4_execution: bool = False
    route_first_moe: bool = False
    intervention_sciencegraph: bool = False

    def gaps(self) -> list[str]:
        """Names of unavailable capabilities — for run_manifest.capability_gaps."""
        return [f.name for f in fields(self) if not getattr(self, f.name)]


class ModelEngine(Protocol):
    backend: str
    name: str
    n_layer: int
    inter: int
    hidden: int
    working_set_mb: float | None
    supports_batch: bool

    def encode(self, prompts: list[str], *, add_special_tokens: bool = False) -> list[np.ndarray]: ...

    def prose_ids(self, max_len: int = 64) -> list[np.ndarray]: ...

    def logits(self, ids: np.ndarray) -> torch.Tensor: ...

    def forward_acts(self, ids: np.ndarray) -> tuple[torch.Tensor, list[torch.Tensor]]: ...

    def hidden_states(self, ids: np.ndarray) -> list[torch.Tensor]: ...

    def hidden_states_batch(self, ids_list: list[np.ndarray]) -> list[list[torch.Tensor]]: ...

    def down_weight(self, layer: int) -> torch.Tensor: ...

    def write_norm(self) -> np.ndarray: ...

    def capabilities(self) -> EngineCapabilities: ...

    def close(self) -> None: ...


class GenerationEngine(ModelEngine, Protocol):
    """Typed extension implemented by engines that advertise ``generation``."""

    def generate(
        self,
        prompt: str | Sequence[int] | np.ndarray,
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        return_text: bool = False,
    ) -> list[int] | str: ...

    def generate_batch(
        self,
        prompts: Sequence[str | Sequence[int] | np.ndarray],
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        return_text: bool = False,
    ) -> list[list[int]] | list[str]: ...


class PagedExpertEngine(GenerationEngine, Protocol):
    """Generation plus the lifecycle API advertised by ``paged_experts``."""

    def reset_page_cache(self, *, clear_pages: bool = True) -> None: ...

    def warm_cache(
        self,
        prompts: Sequence[str | Sequence[int] | np.ndarray],
        *,
        max_new_tokens: int = 1,
    ) -> dict[str, Any]: ...

    def cache_stats(self) -> dict[str, Any]: ...

    def runtime_report(self) -> dict[str, Any]: ...


def apply_patch_ops(x: torch.Tensor, ops: list | tuple | None) -> torch.Tensor:
    out = x.clone() if ops else x
    for op, cols_raw, vals in ops or []:
        cols = torch.as_tensor(cols_raw, dtype=torch.long, device=out.device)
        if op == "zero":
            out[..., cols] = 0.0
        elif op == "scale":
            out[..., cols] = out[..., cols] * float(vals)
        elif op == "global_mean":
            out[..., cols] = vals.to(device=out.device, dtype=out.dtype)
        elif op == "position_mean":
            v = vals.to(device=out.device, dtype=out.dtype)
            out[..., cols] = v[: out.shape[-2]].unsqueeze(0) if out.ndim == 3 else v[: out.shape[0]]
        elif op == "add_amp":
            out[..., cols] = out[..., cols] + vals.to(device=out.device, dtype=out.dtype)
        elif op == "center":
            out[..., cols] = out[..., cols] - out[..., cols].mean(dim=-1, keepdim=True)
        else:
            raise ValueError(f"unknown patch op {op!r}")
    return out


def capture_selected_columns(x: torch.Tensor, locals_: list[int] | tuple[int, ...]) -> torch.Tensor:
    cols = torch.as_tensor(locals_, dtype=torch.long, device=x.device)
    cap = x[..., cols].detach().to(dtype=torch.float16, device="cpu")
    if cap.ndim == 3 and cap.shape[0] == 1:
        cap = cap[0]
    return cap


def avg_answer_logprob(logits: torch.Tensor, prompt_len: int, answer_ids: list[int]) -> float:
    logprob = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for idx, token_id in enumerate(answer_ids):
        total += float(logprob[prompt_len + idx - 1, int(token_id)])
    return total / max(1, len(answer_ids))


def finalize_forced_choice_row(
    scored: list[dict[str, Any]],
    *,
    probe_id: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    correct_lp = float(scored[0]["avg_logprob"])
    wrong_rows = scored[1:]
    best_wrong = max(float(row["avg_logprob"]) for row in wrong_rows) if wrong_rows else float("-inf")
    row: dict[str, Any] = {
        "correct_avg_logprob": correct_lp,
        "best_distractor_avg_logprob": best_wrong,
        "margin": correct_lp - best_wrong,
        "correct_ranked_first": bool(correct_lp > best_wrong),
        "distractors": wrong_rows,
    }
    if probe_id is not None:
        row["probe_id"] = str(probe_id)
    if category is not None:
        row["category"] = str(category)
    return row


def summarize_forced_choice_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "accuracy": None,
            "mean_margin": None,
            "mean_correct_avg_logprob": None,
        }
    acc = np.asarray([1.0 if row["correct_ranked_first"] else 0.0 for row in rows])
    margins = np.asarray([float(row["margin"]) for row in rows])
    correct = np.asarray([float(row["correct_avg_logprob"]) for row in rows])
    return {
        "n": int(len(rows)),
        "accuracy": float(acc.mean()),
        "mean_margin": float(margins.mean()),
        "mean_correct_avg_logprob": float(correct.mean()),
    }


def length_ops_buckets(
    seq_lens: list[int] | np.ndarray,
    ops_keys: list[Any] | None = None,
    *,
    max_batch: int = 16,
    pad_tolerance: int = 16,
) -> list[list[int]]:
    seq_lens = [int(x) for x in seq_lens]
    ops_keys = ops_keys or ["none"] * len(seq_lens)
    by_key: dict[Any, list[int]] = {}
    for idx, key in enumerate(ops_keys):
        by_key.setdefault(key, []).append(idx)
    buckets: list[list[int]] = []
    for idxs in by_key.values():
        idxs.sort(key=lambda i: seq_lens[i])
        cur: list[int] = []
        for idx in idxs:
            too_wide = cur and seq_lens[idx] - seq_lens[cur[0]] > pad_tolerance
            too_large = len(cur) >= max_batch
            if cur and (too_wide or too_large):
                buckets.append(cur)
                cur = []
            cur.append(idx)
        if cur:
            buckets.append(cur)
    return buckets


def layer_maps_for_global_neurons(
    global_neurons: list[int] | tuple[int, ...] | np.ndarray,
    inter: int,
) -> dict[int, dict[str, Any]]:
    maps: dict[int, dict[str, Any]] = {}
    for pos, global_id in enumerate(global_neurons):
        layer, local = divmod(int(global_id), int(inter))
        row = maps.setdefault(layer, {"locals": [], "positions": []})
        row["locals"].append(local)
        row["positions"].append(pos)
    return maps
