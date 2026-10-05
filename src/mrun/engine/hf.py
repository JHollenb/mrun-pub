"""Hugging Face backend implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .. import arch
from ..models import ModelSpec, load_hf_model, load_tokenizer, resolve_model
from ._base_impl import BaseEngine, _probe_candidates
from .base import (
    apply_patch_ops,
    capture_selected_columns,
    finalize_forced_choice_row,
    summarize_forced_choice_rows,
)


class HFEngine(BaseEngine):
    backend = "hf"
    supports_batch = True

    def __init__(
        self,
        model_name: str | ModelSpec,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: str = "cpu",
        step: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.spec = resolve_model(model_name)
        self.name = self.spec.name
        self.device = device
        self.model = model or load_hf_model(
            self.spec,
            step=step,
            device=device,
            **(model_kwargs or {}),
        )
        self.tokenizer = tokenizer or load_tokenizer(self.spec, **(tokenizer_kwargs or {}))
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.resolver = arch.resolver(self.model)
        self.cfg = arch.dims(self.model)
        self.n_layer = int(self.cfg["n_layer"])
        self.inter = int(self.cfg["intermediate"])
        self.hidden = int(self.cfg["hidden"])
        self.working_set_mb = None
        # model_type family (qwen2/llama/qwen3/gpt_neox/mamba/gpt2/…) — the same key the paged
        # store manifest records, so capability checks read one field across backends.
        self.arch = str(
            getattr(getattr(self.model, "config", None), "model_type", None) or self.resolver.family
        )

    @classmethod
    def from_loaded(
        cls, model: Any, tokenizer: Any, *, name: str = "loaded", device: str = "cpu"
    ) -> HFEngine:
        spec = ModelSpec(name=name, hf_id=name, family="auto", label=name)
        return cls(spec, model=model, tokenizer=tokenizer, device=device)

    def logits(self, ids: np.ndarray) -> torch.Tensor:
        input_ids = self._ids_tensor(ids)
        with torch.no_grad():
            out = self.model(input_ids, use_cache=False)  # scoring never reads the KV cache
        return out.logits[0].detach().float().cpu()

    def logits_batch(self, ids_list: list[np.ndarray]) -> list[torch.Tensor]:
        input_ids, attention_mask, lengths = self._batch_tensors(ids_list)
        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
        return [out.logits[i, :length].detach().float().cpu() for i, length in enumerate(lengths)]

    def forward_acts(self, ids: np.ndarray) -> tuple[torch.Tensor, list[torch.Tensor]]:
        store: list[torch.Tensor | None] = [None] * self.n_layer
        handles = []
        try:
            for idx, block in enumerate(self.resolver.layers(self.model)):
                handles.append(
                    self.resolver.add_act_hook(
                        block, lambda act, i=idx: store.__setitem__(i, act.cpu())
                    )
                )
            logits = self.logits(ids)
        finally:
            for handle in handles:
                handle.remove()
        acts = [x.float() for x in store if x is not None]
        if len(acts) != self.n_layer:
            raise RuntimeError(f"captured {len(acts)} activation layers, expected {self.n_layer}")
        return logits, acts

    def forward_acts_batch(
        self, ids_list: list[np.ndarray]
    ) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        logits, lengths, acts_buf, _ = self._run_batch(ids_list, want_acts=True)
        out = []
        for i, length in enumerate(lengths):
            acts_i = [acts_buf[li][i, :length] for li in range(self.n_layer)]
            out.append((logits[i, :length].detach().float().cpu(), acts_i))
        return out

    def forward_attns(self, ids: np.ndarray) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Logits + per-layer softmax attention ``[nH, T, T]`` via ``output_attentions``.
        Mirrors the paged engine's contract (same matrix, minus the batch dim). Models load
        sdpa by default (2026-07-15), which returns NO attention probs — on an empty capture
        this flips the loaded model to eager in place (transformers 5
        ``set_attn_implementation``) and retries once; only genuinely prob-less archs raise."""
        input_ids = self._ids_tensor(ids)
        for attempt in (0, 1):
            with torch.no_grad():
                out = self.model(input_ids, output_attentions=True, use_cache=False)
            attns = [a[0].detach().float().cpu() for a in (out.attentions or [])]
            if len(attns) == self.n_layer:
                return out.logits[0].detach().float().cpu(), attns
            if attempt == 0:
                try:
                    self.model.set_attn_implementation("eager")
                except Exception:  # noqa: BLE001 — older transformers / exotic arch
                    break
        raise RuntimeError(
            f"captured {len(attns)} attention layers, expected {self.n_layer} "
            "(load the model with attn_implementation='eager' for this arch)"
        )

    def forward_acts_resid(
        self,
        ids: np.ndarray,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Logits + per-layer MLP acts + the PRE-final-norm residual after the last block —
        the same contract as ``PagedEngine.forward_acts_resid`` (HF's ``hidden_states[-1]``
        is post-final-norm, so the residual is captured by a pre-hook on the final norm)."""
        resid: list[torch.Tensor] = []
        norm = self._final_norm_module()

        def grab(_module: Any, inputs: tuple[Any, ...]) -> None:
            resid.append(inputs[0][0].detach().float().cpu())  # [T, hidden]

        handle = norm.register_forward_pre_hook(grab)
        try:
            logits, acts = self.forward_acts(ids)
        finally:
            handle.remove()
        if not resid:
            raise RuntimeError("final-norm pre-hook captured nothing")
        return logits, acts, resid[-1]

    def _final_norm_module(self) -> Any:
        for path in (
            "model.norm",
            "gpt_neox.final_layer_norm",
            "transformer.ln_f",
            "backbone.norm_f",
        ):
            node: Any = self.model
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if node is not None:
                return node
        raise NotImplementedError(f"no final-norm module found for arch {self.arch!r}")

    def forward_patched(
        self,
        ids: np.ndarray,
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        head_patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]:
        if head_patch_ops_by_layer:
            raise NotImplementedError(
                "HFEngine forward_patched does not implement head-output patches"
            )
        patch_ops_by_layer = patch_ops_by_layer or {}
        selected_maps = selected_maps or {}
        captured: dict[int, torch.Tensor] = {}
        acts: list[torch.Tensor | None] = [None] * self.n_layer
        handles = []

        def make_hook(layer_idx: int):
            def hook(act: torch.Tensor) -> torch.Tensor:
                patched = apply_patch_ops(act, patch_ops_by_layer.get(layer_idx))
                if collect_acts:
                    acts[layer_idx] = patched.detach().cpu()
                if layer_idx in selected_maps:
                    captured[layer_idx] = capture_selected_columns(
                        patched,
                        selected_maps[layer_idx]["locals"],
                    )
                return patched

            return hook

        try:
            for idx, block in enumerate(self.resolver.layers(self.model)):
                handles.append(self.resolver.add_patch_hook(block, make_hook(idx)))
            logits = self.logits(ids)
        finally:
            for handle in handles:
                handle.remove()
        return logits, [x.float() for x in acts if x is not None], captured

    # ---- batched: ONE padded multi-row forward, hooks installed once ----------------------
    def _run_batch(
        self,
        ids_list: list[np.ndarray],
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        head_patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        want_acts: bool = False,
    ) -> tuple[torch.Tensor, list[int], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Right-pad ``ids_list`` to one ``[B, Tmax]`` batch, install the patch/capture hooks
        ONCE, run a single forward, and return ``(logits[B,Tmax,V], lens, acts_buf, cap_buf)``
        holding the full ``[B, Tmax, ...]`` taps. Right-padding is safe: the forward is causal
        so real positions never attend to pad tokens; callers slice each row to ``lens[i]``."""
        if head_patch_ops_by_layer:
            raise NotImplementedError(
                "HFEngine batched forward does not implement head-output patches"
            )
        input_ids, attention_mask, lengths = self._batch_tensors(ids_list)
        acts_buf: dict[int, torch.Tensor] = {}
        cap_buf: dict[int, torch.Tensor] = {}
        handles = []
        hook_layers = set(patch_ops_by_layer or {}) | set(selected_maps or {})
        if want_acts:
            hook_layers.update(range(self.n_layer))

        def make_hook(layer_idx: int):
            ops = (patch_ops_by_layer or {}).get(layer_idx)
            locals_ = (selected_maps or {}).get(layer_idx, {}).get("locals", [])

            def hook(act: torch.Tensor) -> torch.Tensor:
                x = apply_patch_ops(act, ops)
                if want_acts:
                    acts_buf[layer_idx] = x.detach().float().cpu()  # [B, T, inter]
                if locals_:
                    cols = torch.as_tensor(locals_, dtype=torch.long, device=x.device)
                    cap_buf[layer_idx] = (
                        x[..., cols].detach().to(torch.float16).cpu()
                    )  # [B, T, n_sel]
                return x

            return hook

        try:
            for idx, block in enumerate(self.resolver.layers(self.model)):
                if idx not in hook_layers:
                    continue
                handles.append(self.resolver.add_patch_hook(block, make_hook(idx)))
            with torch.no_grad():
                out = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return out.logits, lengths, acts_buf, cap_buf

    def forward_patched_batch(
        self,
        ids_list: list[np.ndarray],
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        head_patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
    ) -> list[tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]]:
        logits, lengths, acts_buf, cap_buf = self._run_batch(
            ids_list,
            patch_ops_by_layer=patch_ops_by_layer,
            head_patch_ops_by_layer=head_patch_ops_by_layer,
            selected_maps=selected_maps,
            want_acts=collect_acts,
        )
        out = []
        for i, length in enumerate(lengths):
            acts_i = (
                [acts_buf[li][i, :length] for li in range(self.n_layer)] if collect_acts else []
            )
            cap_i = {li: cap_buf[li][i, :length] for li in cap_buf}
            out.append((logits[i, :length].detach().float().cpu(), acts_i, cap_i))
        return out

    def capabilities(self):
        from .base import EngineCapabilities

        return EngineCapabilities(
            logits=True,
            logits_batch=True,
            mlp_acts=True,
            mlp_acts_batch=True,
            mlp_patch=True,
            mlp_patch_batch=True,
            head_patch=False,  # HF hook path has no per-head pre-o_proj tap
            head_patch_batch=False,
            selected_capture=True,
            attentions=True,  # via output_attentions (eager attn on some archs)
            residual_tap=True,  # pre-final-norm capture via forward_acts_resid
            raw_model=True,  # eng.model / eng.resolver — dCE/gradxact/weight ablation
            autograd=True,
            exact_reference=True,
            approximate_quantized=False,
        )

    # ---- subset-lm_head scoring: skip the [B,T,151936] logits materialize + D2H copy ------
    # Port of the paged engine's proven path (paged.py) to hf: for SINGLE-TOKEN forced
    # choice the full-vocab softmax logZ is shared by a probe's candidates and CANCELS, so
    # winner+margin are exact from candidate-only lm_head rows. Selected via the same
    # ``score_forced_choice_many(..., argmax_only=True)`` dispatch in _base_impl.

    def _base_model(self) -> Any:
        for name in ("model", "transformer", "gpt_neox", "backbone"):
            node = getattr(self.model, name, None)
            if node is not None and callable(getattr(node, "forward", None)):
                return node
        raise NotImplementedError(f"no base (headless) submodule found for arch {self.arch!r}")

    def hidden_last_batch(self, ids_list: list[np.ndarray]) -> torch.Tensor:
        """Post-final-norm hidden state at each last real token ``[B, d]`` — runs the
        HEADLESS base model, so the full-vocab lm_head matmul and its fp32 logits buffer
        (the dominant VRAM/D2H cost at large B) never happen."""
        input_ids, attention_mask, lengths = self._batch_tensors(ids_list)
        base = self._base_model()
        with torch.no_grad():
            out = base(input_ids, attention_mask=attention_mask, use_cache=False)
        h = out.last_hidden_state  # [B, Tmax, d], post-final-norm (norm applied in-base)
        rows = torch.arange(h.size(0), device=h.device)
        idx = torch.as_tensor([n - 1 for n in lengths], device=h.device)
        return h[rows, idx].detach().float().cpu()

    def lm_head_rows(self, token_ids: list[int] | np.ndarray) -> torch.Tensor:
        """Only the requested output-embedding rows ``[k, d]`` (never the 544MB matrix)."""
        W = self.model.get_output_embeddings().weight
        idx = torch.as_tensor(np.asarray(token_ids, dtype=np.int64), device=W.device)
        return W[idx].detach().float().cpu()

    def score_forced_choice_argmax_subset(self, probes: list[dict[str, Any]]) -> dict[str, Any]:
        """Same contract + note as ``PagedEngine.score_forced_choice_argmax_subset``:
        winner+margin exact (shared -logZ cancels), avg_logprob unnormalized, raises on
        multi-token candidates (caller falls back to the full path)."""
        if self.arch == "mamba":
            # mamba's base forward takes no attention_mask and pad tokens run through the
            # recurrent state — keep mamba on the existing full path unchanged.
            return self._score_forced_choice_batched(probes)
        prompts: list[np.ndarray] = []
        layout: list[tuple[list[int], list[str], str, Any, int]] = []
        prompt_offsets: dict[tuple[int, ...], int] = {}
        for idx, probe in enumerate(probes):
            prompt_ids = self.encode([str(probe["prompt"])], add_special_tokens=False)[0].tolist()
            cand_tokens, answers = [], []
            for cand in _probe_candidates(probe):
                answer = str(cand["answer"])
                aid = self.encode([answer], add_special_tokens=False)[0].tolist()
                if len(aid) != 1:
                    raise ValueError(
                        f"score_forced_choice_argmax_subset needs single-token candidates; "
                        f"{answer!r} is {len(aid)} tokens"
                    )
                cand_tokens.append(int(aid[0]))
                answers.append(answer)
            prompt_key = tuple(int(value) for value in prompt_ids)
            prompt_offset = prompt_offsets.get(prompt_key)
            if prompt_offset is None:
                prompt_offset = len(prompts)
                prompt_offsets[prompt_key] = prompt_offset
                prompts.append(np.asarray(prompt_ids, dtype=np.int64))
            layout.append(
                (
                    cand_tokens,
                    answers,
                    str(probe.get("probe_id", idx)),
                    probe.get("category"),
                    prompt_offset,
                )
            )

        h_last = self.hidden_last_batch(prompts)  # [P, d] fp32 cpu
        # ONE union head gather + ONE matmul for the whole probe set, then slice per probe.
        # The per-probe loop this replaces paid a separate device->cpu row gather and a
        # [1,d]@[d,k] matmul per probe; suites reuse the same candidate alphabet (digits,
        # yes/no, A-D) so the union is typically a handful of rows for hundreds of probes.
        # ``paged.candidate_logits_batch`` already batches this way. NOT bit-identical to the
        # loop: one [P,U] matmul blocks/accumulates differently than P separate [1,d]@[d,k],
        # so fp32 non-associativity shows up — measured max|dlogit| 4.3e-6, max|dmargin| 6.2e-6
        # on a 24-probe/10-candidate set, winner 24/24 unchanged. That sits under the 2e-5
        # subset-head bar, which is the contract this path is gated on; it is not "identical".
        union = list(dict.fromkeys(tok for cand_tokens, *_rest in layout for tok in cand_tokens))
        offsets = {tok: i for i, tok in enumerate(union)}
        W_union = self.lm_head_rows(union)  # [U, d] fp32 cpu
        all_logits = h_last @ W_union.T  # [P, U]
        rows = []
        for cand_tokens, answers, pid, cat, prompt_offset in layout:
            logits = all_logits[prompt_offset, [offsets[t] for t in cand_tokens]]  # [k]
            scored = [
                {"answer": answers[j], "avg_logprob": float(logits[j])} for j in range(len(answers))
            ]
            rows.append(finalize_forced_choice_row(scored, probe_id=pid, category=cat))
        return {
            "summary": summarize_forced_choice_rows(rows),
            "rows": rows,
            "note": "subset lm_head: winner+margin exact (shared -logZ cancels); avg_logprob unnormalized",
            "automatic_campaign": {
                "eligible": len(layout),
                "fallback": 0,
                "logical_prompt_evaluations": len(layout),
                "physical_prompt_evaluations": len(prompts),
                "candidate_references": sum(len(item[0]) for item in layout),
                "candidate_union_rows": len(union),
            },
        }

    def down_weight(self, layer: int) -> torch.Tensor:
        block = self.resolver.layers(self.model)[int(layer)]
        return self.resolver.down_weight(block).detach().float().cpu()

    def write_norm(self) -> np.ndarray:
        # HF keeps its resolver-based write_norm (proven path); the BaseEngine default
        # (down_weight loop) is used by the paged/mlx/ane backends.
        return self.resolver.write_norm(self.model, self.cfg).astype(np.float32)

    def _ids_tensor(self, ids: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    # -- per-row patched batch (ablation suites) -------------------------------------
    def forward_patched_rows(
        self,
        ids_list: list[np.ndarray],
        patch_ops_by_layer_rows: list[dict[int, list] | None],
    ) -> list[torch.Tensor]:
        """One padded forward where EACH ROW carries its own patch condition — the ablation-
        suite primitive. ``forward_patched_batch`` applies one shared condition to the whole
        batch, so an N-condition suite costs N forwards; this costs ceil(N/B). Returns per-row
        fp32 logits ``[T_i, V]`` like ``logits_batch``. Parity: each row's logits are identical
        to a scalar ``forward_patched`` with that row's ops (tests/test_patched_rows.py)."""
        if len(patch_ops_by_layer_rows) != len(ids_list):
            raise ValueError("patch_ops_by_layer_rows must align with ids_list")
        # {layer: {row: ops}} so a layer's hook touches only the rows patched there
        by_layer: dict[int, dict[int, list]] = {}
        for r, by in enumerate(patch_ops_by_layer_rows):
            for layer, ops in (by or {}).items():
                if ops:
                    by_layer.setdefault(int(layer), {})[r] = ops
        input_ids, attention_mask, lengths = self._batch_tensors(ids_list)
        handles = []

        def make_hook(layer_idx: int):
            rows = by_layer[layer_idx]

            def hook(act: torch.Tensor) -> torch.Tensor:
                x = act.clone()
                for r, ops in rows.items():
                    x[r] = apply_patch_ops(act[r], ops)
                return x

            return hook

        try:
            for idx, block in enumerate(self.resolver.layers(self.model)):
                if idx in by_layer:
                    handles.append(self.resolver.add_patch_hook(block, make_hook(idx)))
            with torch.no_grad():
                out = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return [out.logits[i, :length].detach().float().cpu() for i, length in enumerate(lengths)]

    def selected_last_intervention_branches(
        self,
        prompt_ids: np.ndarray | tuple[int, ...],
        *,
        cut_layer: int,
        token_ids: tuple[int, ...] | list[int],
        patch_ops_by_layer_rows: list[dict[int, list] | None],
        head_patch_ops_by_layer_rows: list[dict[int, list] | None],
        resid_patch_ops_by_layer_rows: list[dict[int, list] | None],
        key_patch_ops_by_layer_rows: list[dict[int, list] | None],
        value_patch_ops_by_layer_rows: list[dict[int, list] | None],
        max_branch_batch: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Execute a typed ScienceGraph BranchPack on the loaded HF model.

        This is the dense CPU/CUDA authority path for raw K/V ScienceGraphs.  It keeps the
        semantic graph and payload custody owned by :mod:`mrun.compiler.sciencegraph`, packs
        branch-local hooks into bounded batches, and pushes the selected vocabulary union below
        the output head.  Unlike the paged executor it cannot reuse a StateCut, so telemetry says
        so explicitly instead of claiming prefix sharing.
        """

        del cut_layer  # HF runs complete forwards; the graph cut remains semantic evidence.
        rows = len(patch_ops_by_layer_rows)
        aligned = (
            head_patch_ops_by_layer_rows,
            resid_patch_ops_by_layer_rows,
            key_patch_ops_by_layer_rows,
            value_patch_ops_by_layer_rows,
        )
        if any(len(value) != rows for value in aligned):
            raise ValueError("typed intervention row maps must align")
        if any(any(value or {} for value in maps) for maps in aligned[:2]):
            raise NotImplementedError(
                "HF ScienceGraph currently supports raw K/V projection ports; "
                "head/residual ports remain on the paged reference path"
            )
        if any(value or {} for value in patch_ops_by_layer_rows):
            raise NotImplementedError(
                "mixed MLP and raw K/V HF ScienceGraphs are not yet supported"
            )
        if rows == 0:
            raise ValueError("ScienceGraph requires at least one branch row")
        branch_batch = rows if max_branch_batch is None else int(max_branch_batch)
        if branch_batch <= 0:
            raise ValueError("max_branch_batch must be positive")
        selected = tuple(int(value) for value in token_ids)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("selected token IDs must be non-empty and unique")

        prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=self.device).reshape(1, -1)
        layers = list(self.resolver.layers(self.model))
        if self.arch not in {"qwen2", "qwen3", "llama"}:
            raise NotImplementedError(
                f"raw K/V ScienceGraph is unsupported for HF architecture {self.arch!r}"
            )

        def pack_projection_rows(
            maps: list[dict[int, list] | None],
        ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
            packed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for layer_index in range(len(layers)):
                touched = [
                    index
                    for index, by_layer in enumerate(maps)
                    if (by_layer or {}).get(layer_index)
                ]
                if not touched:
                    continue
                first_ops = maps[touched[0]][layer_index]  # type: ignore[index]
                first_donor = torch.as_tensor(first_ops[0][2]).detach().cpu()
                if first_donor.ndim == 3 and first_donor.shape[0] == 1:
                    first_donor = first_donor[0]
                if first_donor.ndim != 2:
                    raise ValueError("raw K/V donor must have shape [T,D]")
                mask = torch.zeros(
                    (len(maps), prompt.shape[-1]), dtype=torch.bool
                )
                replacements = torch.zeros(
                    (len(maps), prompt.shape[-1], first_donor.shape[-1]),
                    dtype=first_donor.dtype,
                )
                for row_index in touched:
                    for op, positions_value, donor_value in maps[row_index][layer_index]:  # type: ignore[index]
                        if op != "position_replace":
                            raise ValueError(f"unsupported raw K/V intervention {op!r}")
                        positions = torch.as_tensor(positions_value, dtype=torch.long)
                        donor = torch.as_tensor(donor_value).detach().cpu()
                        if donor.ndim == 3 and donor.shape[0] == 1:
                            donor = donor[0]
                        if donor.ndim != 2 or donor.shape[-1] != replacements.shape[-1]:
                            raise ValueError("raw K/V donor must have shape [T,D]")
                        source = (
                            donor.index_select(0, positions)
                            if donor.shape[0] == prompt.shape[-1]
                            else donor
                        )
                        if tuple(source.shape) != (
                            positions.numel(),
                            replacements.shape[-1],
                        ):
                            raise ValueError("raw K/V donor rows do not align with positions")
                        mask[row_index, positions] = True
                        replacements[row_index, positions] = source.to(
                            replacements.dtype
                        )
                packed[layer_index] = (
                    mask.to(self.device),
                    replacements.to(self.device),
                )
            return packed

        def apply_projection_rows(
            output: torch.Tensor,
            packed: tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            mask, replacements = packed
            if replacements.shape[-1] != output.shape[-1]:
                raise ValueError("raw K/V donor width does not match projection output")
            return torch.where(
                mask.unsqueeze(-1), replacements.to(output.dtype), output
            )

        output_rows = []
        physical_calls = 0
        for start in range(0, rows, branch_batch):
            stop = min(rows, start + branch_batch)
            key_maps = key_patch_ops_by_layer_rows[start:stop]
            value_maps = value_patch_ops_by_layer_rows[start:stop]
            packed_keys = pack_projection_rows(key_maps)
            packed_values = pack_projection_rows(value_maps)
            handles = []
            try:
                for layer_index, block in enumerate(layers):
                    if layer_index in packed_keys:
                        handles.append(
                            block.self_attn.k_proj.register_forward_hook(
                                lambda _module, _args, output, rows=packed_keys[layer_index]: (
                                    apply_projection_rows(output, rows)
                                )
                            )
                        )
                    if layer_index in packed_values:
                        handles.append(
                            block.self_attn.v_proj.register_forward_hook(
                                lambda _module, _args, output, rows=packed_values[layer_index]: (
                                    apply_projection_rows(output, rows)
                                )
                            )
                        )
                input_ids = prompt.expand(stop - start, -1)
                with torch.no_grad():
                    base = getattr(self.model, "model", None)
                    head = self.model.get_output_embeddings()
                    if base is None or head is None:
                        raise NotImplementedError(
                            "selected-head ScienceGraph requires a separable base model and output head"
                        )
                    hidden = base(input_ids=input_ids, use_cache=False).last_hidden_state[:, -1]
                    indices = torch.as_tensor(selected, dtype=torch.long, device=head.weight.device)
                    weight = head.weight.index_select(0, indices)
                    scores = torch.nn.functional.linear(
                        hidden.to(torch.float64), weight.to(torch.float64), None
                    ).to(torch.float32)
                output_rows.append(scores.detach().float().cpu())
                physical_calls += 1
            finally:
                for handle in handles:
                    handle.remove()
        result = torch.cat(output_rows, dim=0)
        return result, {
            "shared_prefix_materialized": False,
            "suffix_weight_traversals": max(0, physical_calls - 1),
            "complete_forward_calls": physical_calls,
            "branch_batch": branch_batch,
            "full_vocabulary_logits_materialized": False,
            "runtime_device": str(self.device),
        }

    def ablation_suite_ce(
        self,
        prompt_ids_list: list[np.ndarray],
        conditions: list[dict[int, list] | None],
        *,
        max_batch: int | None = None,
    ) -> list[float]:
        """Mean next-token CE of every (condition x prompt-bank) cell — the dCE-atlas
        primitive. Rows are (condition, prompt) pairs batched together and chunked at the
        activation-guarded batch cap; pass ``None`` as a condition for the unpatched
        baseline. Returns one mean-CE per condition (averaged over the bank)."""
        from ..policy import resolve_max_batch

        if max_batch is None:
            seq_lens = [len(p) for p in prompt_ids_list]
            max_batch = resolve_max_batch(self.device, seq_lens, self.hidden)
        cells = [(ci, p) for ci in range(len(conditions)) for p in range(len(prompt_ids_list))]
        ce_sums = [0.0] * len(conditions)
        for start in range(0, len(cells), max_batch):
            chunk = cells[start : start + max_batch]
            ids = [prompt_ids_list[p] for _, p in chunk]
            ops = [conditions[ci] for ci, _ in chunk]
            for (ci, p), logits in zip(chunk, self.forward_patched_rows(ids, ops), strict=True):
                tgt = torch.as_tensor(prompt_ids_list[p][1:], dtype=torch.long)
                lp = torch.log_softmax(logits[:-1], dim=-1)
                ce_sums[ci] += float(-lp[torch.arange(len(tgt)), tgt].mean())
        return [s / max(1, len(prompt_ids_list)) for s in ce_sums]

    # -- prefix-KV forced choice ---------------------------------------------------
    def score_forced_choice_prefix_kv(self, probes: list[dict[str, Any]]) -> dict[str, Any]:
        """Forced choice with the prompt forwarded ONCE per probe (KV cache), every candidate
        forwarded as answer-tokens-only on top of its probe's cache. Two batched forwards
        total. Saves ~(n_cand-1)/n_cand of the prompt compute vs the flat batched path, which
        recomputes the prompt inside every (prompt+candidate) row; the avg-logprob math is the
        SAME computation (cached attention is exact), so winners/margins match the flat path
        to reduction noise (parity-gated in tests/test_prefix_kv.py).

        Falls back to the flat batched path when the savings are <25% of total tokens (short
        prompts / single candidates) — cache bookkeeping isn't free.

        Chunked at the device batch cap: phase 1 materializes [N, Tmax, vocab] logits, so an
        unchunked large probe list allocates gigabytes (measured: 4096 probes -> 15.07 GiB
        lm_head buffer, MPS crash). Chunks recurse through this method and merge rows."""
        from ..policy import resolve_max_batch

        prompt_ids = [
            self.encode([str(p["prompt"])], add_special_tokens=False)[0].tolist() for p in probes
        ]
        cap = resolve_max_batch(self.device, [len(p) for p in prompt_ids], self.hidden)
        if len(probes) > cap:
            chunk_probes = [
                probe if "probe_id" in probe else {**probe, "probe_id": str(i)}
                for i, probe in enumerate(probes)
            ]
            rows = []
            for start in range(0, len(chunk_probes), cap):
                rows.extend(
                    self.score_forced_choice_prefix_kv(chunk_probes[start : start + cap])["rows"]
                )
            return {
                "summary": summarize_forced_choice_rows(rows),
                "rows": rows,
                "note": "prefix-kv: chunked at batch cap",
            }
        layout: list[tuple[int, str, list[int]]] = []  # (probe_idx, answer, answer_ids)
        for idx, probe in enumerate(probes):
            for cand in _probe_candidates(probe):
                answer = str(cand["answer"])
                aids = self.encode([answer], add_special_tokens=False)[0].tolist()
                layout.append((idx, answer, aids))
        flat_tokens = sum(len(prompt_ids[i]) + len(a) for i, _, a in layout)
        kv_tokens = sum(len(p) for p in prompt_ids) + sum(len(a) for _, _, a in layout)
        if not layout or kv_tokens >= 0.75 * flat_tokens:
            return self._score_forced_choice_batched(probes)

        # phase 1: one batched prompt forward, keep the KV cache + last-position logits
        input_ids, attn, plens = self._batch_tensors(
            [np.asarray(p, dtype=np.int64) for p in prompt_ids]
        )
        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attn, use_cache=True)
        cache = out.past_key_values
        first_lp = [
            torch.log_softmax(out.logits[i, plens[i] - 1].detach().float(), dim=-1)
            for i in range(len(probes))
        ]
        del out

        # phase 2: expand the cache along batch (one row per candidate) and forward answers.
        # Right-padded prompt rows are handled by the expanded attention mask (cached pad
        # positions stay masked); RoPE positions are set explicitly per row so the answer
        # continues at its probe's true prompt length, not at the padded length.
        sel = torch.as_tensor([i for i, _, _ in layout], dtype=torch.long, device=self.device)
        cache.batch_select_indices(sel)
        n = len(layout)
        amax = max(len(a) for _, _, a in layout)
        pad = int(self.tokenizer.pad_token_id or 0)
        ans = torch.full((n, amax), pad, dtype=torch.long, device=self.device)
        ans_mask = torch.zeros((n, amax), dtype=torch.long, device=self.device)
        pos = torch.zeros((n, amax), dtype=torch.long, device=self.device)
        for r, (pi, _, aids) in enumerate(layout):
            ans[r, : len(aids)] = torch.as_tensor(aids, dtype=torch.long, device=self.device)
            ans_mask[r, : len(aids)] = 1
            pos[r] = plens[pi] + torch.arange(amax, device=self.device)
        attn2 = torch.cat([attn.index_select(0, sel), ans_mask], dim=1)
        with torch.no_grad():
            out2 = self.model(
                ans, attention_mask=attn2, past_key_values=cache, position_ids=pos, use_cache=True
            )

        # assemble: answer token 0 is predicted by the prompt's last position (phase 1);
        # token t>=1 by answer position t-1 (phase 2). Same mean as avg_answer_logprob.
        scored_by_probe: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(probes))}
        for r, (pi, answer, aids) in enumerate(layout):
            total = float(first_lp[pi][int(aids[0])])
            if len(aids) > 1:
                lp = torch.log_softmax(out2.logits[r, : len(aids) - 1].detach().float(), dim=-1)
                total += sum(float(lp[t, int(aids[t + 1])]) for t in range(len(aids) - 1))
            scored_by_probe[pi].append({"answer": answer, "avg_logprob": total / max(1, len(aids))})
        rows = [
            finalize_forced_choice_row(
                scored_by_probe[i],
                probe_id=str(probes[i].get("probe_id", i)),
                category=probes[i].get("category"),
            )
            for i in range(len(probes))
        ]
        return {
            "summary": summarize_forced_choice_rows(rows),
            "rows": rows,
            "note": "prefix-kv: prompt forwarded once per probe, candidates share the cache",
        }

    def _batch_tensors(
        self, ids_list: list[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        lengths = [int(len(ids)) for ids in ids_list]
        max_len = max(lengths)
        pad = int(self.tokenizer.pad_token_id or 0)
        arr = np.full((len(ids_list), max_len), pad, dtype=np.int64)
        mask = np.zeros((len(ids_list), max_len), dtype=np.int64)
        for i, ids in enumerate(ids_list):
            arr[i, : len(ids)] = ids
            mask[i, : len(ids)] = 1
        return (
            torch.as_tensor(arr, dtype=torch.long, device=self.device),
            torch.as_tensor(mask, dtype=torch.long, device=self.device),
            lengths,
        )
