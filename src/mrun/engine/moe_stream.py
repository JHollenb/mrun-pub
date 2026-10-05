"""Bounded-memory model-engine adapter for parity-gated streamed MoE forwards."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from ..models import resolve_model, snapshot_dir
from .moe_safetensors import (
    PagedSafetensors,
    detect_layout,
    resolve_model_dir,
    streamed_moe_forward,
    streamed_moe_forward_batch,
)
from ._base_impl import BaseEngine
from .base import KEEP, EngineCapabilities


def _compute_dtype(device: str, value: torch.dtype | str | None) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None or value == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    names = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return names[str(value).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported MoE stream dtype {value!r}") from exc


class MoEStreamEngine(BaseEngine):
    """Exact-structure MoE engine with O(one expert matrix) resident model memory.

    OLMoE, Mixtral, Qwen2-MoE (including its shared expert), and Qwen3-MoE layouts are
    parity-gated against Transformers. The adapter exposes logits and residual
    states/interventions. Expert-neuron activations remain a separate typed physiology surface
    because an MoE layer has no single dense MLP activation matrix.

    ``streamed_moe_forward`` owns one causal sequence and one routing trace per call —
    concatenating rows would leak attention across prompts, so ``residual_tap_batch`` remains
    false for the full-stack captures. The FINAL-HIDDEN surface, however, is genuinely
    batched: :meth:`hidden_last_batch` and :meth:`final_hidden_arms` go through
    ``streamed_moe_forward_batch``, which streams each weight ONCE and keeps every arm's
    attention on its own sequence (measured 6.61x wall on Qwen3-Coder-480B at 11 arms,
    bit-exact against the serial path; see
    experiments/2026-07-25-moe-stream-max-utilization). Because forced-choice scoring rides on
    :meth:`hidden_last_batch`, a whole battery under one ablation arm now costs ~one model
    stream instead of one per item.

    ANALYSIS TAPS: :meth:`ablation` scopes whole-component ``(layer, attn|mlp)`` removal,
    embedding-direction/subspace removal and ``zero_embedding`` over :meth:`hidden_states`,
    :meth:`final_hidden`, :meth:`hidden_last_batch`, :meth:`logits` and the candidate-subset
    head (:meth:`lm_head_rows` / :meth:`candidate_logits_batch`). The candidate head
    range-reads individual ``lm_head`` rows, so forced-choice scoring costs a few KB per probe
    rather than a 1.9 GB pull — that is what makes a battery affordable here.
    """

    backend = "moe-stream"
    supports_batch = False

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        dtype: torch.dtype | str | None = None,
        abort_rss_gb: float = 48.0,
        log: Any = None,
        **_ignored: Any,
    ) -> None:
        raw_path = Path(model_name).expanduser()
        if raw_path.is_dir():
            model_dir = resolve_model_dir(str(raw_path))
            resolved_name = raw_path.name
        else:
            spec = resolve_model(model_name)
            try:
                model_dir = snapshot_dir(spec)
            except FileNotFoundError:
                model_dir = resolve_model_dir(spec.hf_id)
            resolved_name = spec.name

        self.model_dir = model_dir
        self.store = PagedSafetensors(model_dir)
        self.layout = detect_layout(self.store)
        self.cfg = self.store.cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device is None and self.device == "cpu" and torch.version.cuda is not None:
            # cuda-BUILT torch with no usable card: the host expected CUDA and the run
            # degrades to CPU silently otherwise (masks a broken install). A cpu-only
            # torch build (the Mac, CI) is an ordinary cpu host — stay quiet there.
            print("warn: moe-stream defaulting to cpu (cuda-built torch but "
                  "torch.cuda unavailable)", flush=True)
        self.dtype = _compute_dtype(self.device, dtype)
        self.abort_rss_gb = float(abort_rss_gb)
        self.log = log
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.name = resolved_name
        self.arch = self.layout.name
        self.n_layer = int(self.cfg["num_hidden_layers"])
        self.hidden = int(self.cfg["hidden_size"])
        self.inter = int(
            self.cfg.get("moe_intermediate_size") or self.cfg.get("intermediate_size") or 0
        )
        self.working_set_mb = None
        self.last_result: dict[str, Any] | None = None
        # Sticky analysis taps (see `ablation`). Held on the engine rather than threaded through
        # every call so measurement code written against the fused MoE engine's surface runs
        # here unchanged.
        self.init_taps()

    def init_taps(self) -> None:
        """Reset every analysis-tap field to "no ablation".

        Public and idempotent on purpose: it is the single definition of this engine's tap
        state, so adding a tap cannot leave a construction path (or a test fixture that
        bypasses ``__init__`` to avoid registry/tokenizer resolution) silently missing a field.
        """
        self.ablate_component: tuple[int, str] | None = None
        self.ablate_embed_direction: torch.Tensor | None = None
        self.ablate_embed_alpha: float = 1.0
        self.zero_embedding: bool = False
        self.ablate_lm_head: bool = False
        self._head_row_cache: dict[int, torch.Tensor] = {}

    @contextmanager
    def ablation(
        self,
        *,
        component: tuple[int, str] | None | Any = KEEP,
        embed_direction: torch.Tensor | None | Any = KEEP,
        embed_alpha: float | Any = KEEP,
        zero_embedding: bool | Any = KEEP,
        lm_head: bool | None | Any = KEEP,
    ) -> Any:
        """Scope an ablation over every tap call made inside the block, then restore.

        Mirrors ``Qwen3MoeCudaEngine.ablation`` so one caller serves both MoE engines: any
        argument left unset keeps its current value (a component ablation nests inside an
        embedding ablation without cancelling it); pass ``None`` explicitly to clear one.

        ``lm_head`` also removes the direction from the candidate head rows
        (:meth:`lm_head_rows`) — what a TIED checkpoint's single shared tensor does implicitly.
        It defaults to the checkpoint's own tie flag.
        """
        previous = (
            self.ablate_component,
            self.ablate_embed_direction,
            self.ablate_embed_alpha,
            self.zero_embedding,
            self.ablate_lm_head,
        )
        if component is not KEEP:
            self.ablate_component = component
        if embed_direction is not KEEP:
            self.ablate_embed_direction = embed_direction
        if embed_alpha is not KEEP:
            self.ablate_embed_alpha = float(embed_alpha)
        if zero_embedding is not KEEP:
            self.zero_embedding = bool(zero_embedding)
        if lm_head is not KEEP:
            self.ablate_lm_head = self.tied_lm_head if lm_head is None else bool(lm_head)
        try:
            yield self
        finally:
            (
                self.ablate_component,
                self.ablate_embed_direction,
                self.ablate_embed_alpha,
                self.zero_embedding,
                self.ablate_lm_head,
            ) = previous

    def _run(
        self,
        ids: np.ndarray,
        *,
        capture_hidden_states: bool = False,
        capture_final_hidden: bool = False,
        resid_patch_ops_by_layer: dict[int, list] | None = None,
        capture_expert_stats: bool = False,
        skip_lm_head: bool = False,
    ) -> dict[str, Any]:
        row = np.asarray(ids, dtype=np.int64)
        if row.ndim != 1 or not row.size:
            raise ValueError("ids must be a non-empty one-dimensional token array")
        result = streamed_moe_forward(
            self.store,
            torch.as_tensor(row, dtype=torch.long),
            device=self.device,
            dtype=self.dtype,
            capture_expert_stats=capture_expert_stats,
            return_logits=not skip_lm_head,
            capture_hidden_states=capture_hidden_states,
            capture_final_hidden=capture_final_hidden,
            skip_lm_head=skip_lm_head,
            resid_patch_ops_by_layer=resid_patch_ops_by_layer,
            ablate_component=self.ablate_component,
            ablate_embed_direction=self.ablate_embed_direction,
            ablate_embed_alpha=self.ablate_embed_alpha,
            zero_embedding=self.zero_embedding,
            abort_rss_gb=self.abort_rss_gb,
            log=self.log,
        )
        self.last_result = result
        return result

    def logits(self, ids: np.ndarray) -> torch.Tensor:
        return self._run(ids)["logits"]

    def hidden_states(self, ids: np.ndarray) -> list[torch.Tensor]:
        return self._run(ids, capture_hidden_states=True, skip_lm_head=True)["hidden_states"]

    def final_hidden(self, ids: np.ndarray) -> torch.Tensor:
        """Post-final-norm hidden state ``[T, d]`` WITHOUT computing the output head. On a
        151936-row vocab the head is a 1.9 GB pull and a ``[T, vocab]`` fp32 materialization
        that a residual-stream measurement never reads."""
        return self._run(ids, capture_final_hidden=True, skip_lm_head=True)["final_hidden"]

    @property
    def tied_lm_head(self) -> bool:
        """A tied checkpoint ships no output-head tensor; the embedding serves both roles.
        Key names are layout-owned (DeepSeek-V4 native ships ``head.weight``/``embed.weight``)."""
        return not self.store.has(self.layout.head_key)

    def _head_key(self) -> str:
        return self.layout.embed_key if self.tied_lm_head else self.layout.head_key

    def lm_head_rows(self, token_ids: list[int] | np.ndarray) -> torch.Tensor:
        """Only the requested output-head rows ``[k, d]`` — never the whole head.

        On a 151936x6144 head that is a few KB per probe instead of a 1.9 GB pull, which is the
        entire reason forced-choice scoring is affordable on a streaming engine. Rows are cached
        because a battery reuses the same small candidate set across every ablation arm; the
        cache is keyed by token id and holds only distinct candidates (tens of rows).
        """
        wanted = [int(t) for t in np.asarray(token_ids, dtype=np.int64).reshape(-1)]
        missing = sorted({t for t in wanted if t not in self._head_row_cache})
        if missing:
            fetched = self.store.get_rows(self._head_key(), missing)
            for token, row in zip(missing, fetched, strict=True):
                self._head_row_cache[token] = row.to(torch.float32)
            self.store.release()
        rows = torch.stack([self._head_row_cache[t] for t in wanted])
        direction = self.ablate_embed_direction
        if direction is not None and self.ablate_lm_head:
            basis = torch.as_tensor(direction, dtype=torch.float32)
            if basis.ndim == 1:
                basis = (basis / (basis.norm() + 1e-9))[:, None]
            rows = rows - self.ablate_embed_alpha * (rows @ basis) @ basis.T
        return rows

    def selected_last_logits_batch(
        self,
        ids_list: list[np.ndarray],
        token_ids: list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        """``[B, len(token_ids)]`` last-position logits over a token SUBSET. Same contract as
        ``PagedEngine``/``Qwen3MoeCudaEngine``: one BATCHED hidden pass (one weight stream for
        all rows), one head-row lookup."""
        hidden = self.hidden_last_batch(ids_list).float()
        weights = self.lm_head_rows(list(token_ids)).to(hidden.device).float()
        return (hidden @ weights.T).detach().cpu()

    def candidate_logits_batch(
        self,
        ids_list: list[np.ndarray],
        candidate_token_ids: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Per-row candidate scores through one hidden pass per row and one union head lookup.

        Signature-compatible with the dense paged and fused MoE engines, so forced-choice
        scoring code (``manalysis.paged_concept.battery_accuracy_paged``) is engine-agnostic.
        The shared ``-logZ`` cancels inside a row's candidate set, so argmax and margins are
        exact without ever forming a full-vocab softmax — no ``[T, 151936]`` tensor is built
        anywhere on this path.

        Each row is still its own causal SEQUENCE (never concatenated — no cross-prompt
        attention), but all rows share one weight-streaming pass via ``hidden_last_batch``.
        """
        union = tuple(
            dict.fromkeys(
                token for row_candidates in candidate_token_ids for token in row_candidates
            )
        )
        union_scores = self.selected_last_logits_batch(ids_list, union)
        offsets = {token: index for index, token in enumerate(union)}
        return [
            row_scores.index_select(
                0,
                torch.as_tensor(
                    [offsets[token] for token in row_candidates],
                    dtype=torch.long,
                    device=row_scores.device,
                ),
            )
            for row_scores, row_candidates in zip(
                union_scores,
                candidate_token_ids,
                strict=True,
            )
        ]

    def hidden_last_batch(self, ids_list: list[np.ndarray]) -> torch.Tensor:
        """Post-final-norm state at each row's last token ``[N, d]``.

        Same name and contract as ``PagedEngine``/``Qwen3MoeCudaEngine`` — and since
        2026-07-25 genuinely BATCHED over the weight stream: all rows ride one
        ``streamed_moe_forward_batch`` pass (each weight pulled once, each row still its own
        causal sequence — no cross-prompt attention), under whatever sticky ablation taps are
        in scope. Cost is ~one model stream per CALL, no longer one per row."""
        rows = [np.asarray(ids, dtype=np.int64) for ids in ids_list]
        if not rows:
            return torch.empty((0, self.hidden), dtype=torch.float32)
        return self.final_hidden_arms(rows, [(j, self.ablate_component)
                                             for j in range(len(rows))])

    def final_hidden_arms(
        self,
        rows: list[np.ndarray],
        arms: list[tuple[int, tuple[int, str] | None]],
    ) -> torch.Tensor:
        """``[len(arms), d]`` final-hidden last-token states, one streaming pass for ALL arms.

        ``arms`` pairs a row index with a per-arm whole-component ablation (or ``None``) — the
        multi-arm surface a coupling sweep needs: clean + every ``(layer, attn|mlp)`` arm of
        every battery item in ONE pass instead of ``n_items * (1 + 2 * n_layers)`` serial
        forwards. Embedding-direction taps (``ablation(embed_direction=...)``,
        ``zero_embedding``) apply to every arm from the sticky state, exactly as they would to
        each serial call; a sticky COMPONENT tap does not leak in — the per-arm entry is the
        only component axis, so pass it explicitly per arm."""
        result = streamed_moe_forward_batch(
            self.store,
            [torch.as_tensor(np.asarray(r, dtype=np.int64)) for r in rows],
            arms,
            device=self.device,
            dtype=self.dtype,
            ablate_embed_direction=self.ablate_embed_direction,
            ablate_embed_alpha=self.ablate_embed_alpha,
            zero_embedding=self.zero_embedding,
            abort_rss_gb=self.abort_rss_gb,
            log=self.log,
        )
        return result["final_last"]

    def forward_patched(
        self,
        ids: np.ndarray,
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        resid_patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]:
        """``(logits [T, vocab], [], {})`` under residual-stream interventions.

        WHAT THIS CAN EXPRESS: ``resid_patch_ops_by_layer={layer: [("proj_remove", vector,
        None)]}`` — remove a direction from the whole residual at every position, applied AFTER
        a complete decoder block (post-attention AND post-MoE). ``proj_remove`` is the only op.

        WHAT IT CANNOT: whole-component ``(layer, attn|mlp)`` ablation. A residual patch fires
        once the block has already added both terms, so it cannot subtract one of them — and
        dense-MLP neuron taps (``patch_ops_by_layer``/``selected_maps``/``collect_acts``) do not
        exist here at all, because an MoE layer has no single dense activation matrix. Component
        ablation is a separate, exact tap: ``engine.ablation(component=(layer, "attn"|"mlp"))``,
        which drops the term inside the forward rather than trying to cancel it afterwards.
        """
        if patch_ops_by_layer or selected_maps or collect_acts:
            raise NotImplementedError(
                "moe-stream exposes residual interventions, not dense-MLP neuron taps"
            )
        result = self._run(ids, resid_patch_ops_by_layer=resid_patch_ops_by_layer)
        return result["logits"], [], {}

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            logits=True,
            residual_tap=True,
            exact_reference=self.dtype == torch.float32,
        )

    def forward_acts(self, _ids: np.ndarray) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError("an MoE layer has routed per-expert activations")

    def down_weight(self, _layer: int) -> torch.Tensor:
        raise NotImplementedError("an MoE layer has per-expert down weights")

    def close(self) -> None:
        self._head_row_cache.clear()
        self.store.release()
