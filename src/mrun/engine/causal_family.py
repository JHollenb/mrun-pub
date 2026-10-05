"""High-throughput row-batched causal-family execution for dense HF engines.

This is the engine-owned counterpart to manalysis's conditional-family assay.  It keeps the
scientific family/workload definitions in manalysis while making physical execution an explicit
mrun contract: capture clean pre-output attention contexts once, batch the family × example grid
as rows, apply row-specific source cuts and head repairs, and score only the declared candidates.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .hf import HFEngine

Family = Mapping[int, Sequence[int]]
ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class CausalFamilyExample:
    """One immutable forced-choice row and its source/query/donor addresses."""

    input_ids: tuple[int, ...]
    candidate_token_ids: tuple[int, ...]
    correct_candidate_index: int
    source_position: int
    query_position: int
    donor_index: int

    def __post_init__(self) -> None:
        if not self.input_ids or not self.candidate_token_ids:
            raise ValueError("causal-family examples require input and candidate token IDs")
        if not 0 <= self.correct_candidate_index < len(self.candidate_token_ids):
            raise ValueError("correct_candidate_index is outside candidate_token_ids")
        if not 0 <= self.source_position < len(self.input_ids):
            raise ValueError("source_position is outside input_ids")
        if not 0 <= self.query_position < len(self.input_ids):
            raise ValueError("query_position is outside input_ids")
        if self.donor_index < 0:
            raise ValueError("donor_index must be non-negative")


@dataclass(frozen=True, slots=True)
class CausalFamilyContinuationExample:
    """One reference parent-prefix row followed by one autonomous token commit.

    The intervention is applied only while executing ``input_ids``. This HF
    reference path replays that complete prefix; it does not materialize or
    reuse a shared StateCut. The resulting KV state is then consumed by an
    unpatched one-token continuation
    using that branch's own parent argmax. ``committed_token_id`` declares the
    expected common commit identity and must equal ``expected_parent_token_id``;
    public invariance remains an observed full-vocabulary condition;
    ``candidate_token_ids`` and ``correct_candidate_index`` identify the child
    route consumer.
    """

    input_ids: tuple[int, ...]
    committed_token_id: int
    expected_parent_token_id: int
    candidate_token_ids: tuple[int, ...]
    correct_candidate_index: int
    source_position: int
    query_position: int
    donor_index: int

    def __post_init__(self) -> None:
        token_scalars = (self.committed_token_id, self.expected_parent_token_id)
        token_sequences = (self.input_ids, self.candidate_token_ids)
        if (
            not self.input_ids
            or not self.candidate_token_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in token_scalars
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for sequence in token_sequences
                for value in sequence
            )
        ):
            raise ValueError("continuation examples require non-negative token IDs")
        if len(self.candidate_token_ids) < 2 or len(set(self.candidate_token_ids)) != len(
            self.candidate_token_ids
        ):
            raise ValueError("continuation candidates must be distinct with length at least two")
        if not 0 <= self.correct_candidate_index < len(self.candidate_token_ids):
            raise ValueError("correct_candidate_index is outside candidate_token_ids")
        if self.committed_token_id != self.expected_parent_token_id:
            raise ValueError("declared commit token must equal the expected parent token")
        if self.query_position != len(self.input_ids) - 1:
            raise ValueError("parent StateCut query_position must be the committed-prefix boundary")
        if not 0 <= self.source_position <= self.query_position:
            raise ValueError("source_position is outside the parent causal prefix")
        if self.donor_index < 0:
            raise ValueError("donor_index must be non-negative")


def _normalize_family(family: Family, *, layers: int, heads: int) -> dict[int, tuple[int, ...]]:
    normalized: dict[int, tuple[int, ...]] = {}
    for raw_layer, raw_heads in family.items():
        layer = int(raw_layer)
        selected = tuple(sorted({int(head) for head in raw_heads}))
        if not 0 <= layer < layers or not selected:
            raise ValueError("causal family contains an invalid or empty layer")
        if selected[0] < 0 or selected[-1] >= heads:
            raise ValueError("causal family contains an out-of-range head")
        normalized[layer] = selected
    if not normalized:
        raise ValueError("causal family must be non-empty")
    return normalized


def _attention_modules(engine: HFEngine) -> tuple[list[Any], list[Any]]:
    if engine.arch not in {"qwen2", "qwen3", "llama"}:
        raise NotImplementedError(
            "batched causal-family execution currently supports qwen2, qwen3, and llama"
        )
    blocks = list(engine.resolver.layers(engine.model))
    attention = [block.self_attn for block in blocks]
    projections = [module.o_proj for module in attention]
    return attention, projections


def _base_hidden(
    engine: HFEngine,
    ids_list: Sequence[np.ndarray],
    *,
    handles: Sequence[Any] = (),
) -> tuple[torch.Tensor, list[int]]:
    del handles  # handles stay alive by caller scope; explicit parameter documents the contract.
    input_ids, attention_mask, lengths = engine._batch_tensors(list(ids_list))
    with torch.inference_mode():
        output = engine._base_model()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return output.last_hidden_state, lengths


def _parent_cached_output(
    engine: HFEngine,
    examples: Sequence[CausalFamilyContinuationExample],
    *,
    handles: Sequence[Any] = (),
) -> tuple[Any, torch.Tensor]:
    """Execute one uniform parent prefix and retain its exact mutable KV state."""

    del handles  # caller scope owns hook lifetime; this names the dependency.
    ids = [np.asarray(row.input_ids, dtype=np.int64) for row in examples]
    input_ids, attention_mask, lengths = engine._batch_tensors(ids)
    if len(set(lengths)) != 1 or lengths[0] != input_ids.shape[1]:
        raise ValueError("closed-loop parent StateCut requires a uniform unpadded prefix batch")
    with torch.inference_mode():
        output = engine._base_model()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    if getattr(output, "past_key_values", None) is None:
        raise RuntimeError("parent StateCut execution did not return a KV state")
    return output, attention_mask


def _committed_child_hidden(
    engine: HFEngine,
    examples: Sequence[CausalFamilyContinuationExample],
    *,
    parent_output: Any,
    parent_attention_mask: torch.Tensor,
    committed_token_ids: Sequence[int],
) -> torch.Tensor:
    """Commit each branch's own public argmax and consume its parent KV state."""

    committed = torch.as_tensor(
        list(committed_token_ids),
        dtype=torch.long,
        device=engine.device,
    ).unsqueeze(1)
    if committed.shape != (len(examples), 1):
        raise ValueError("branch commit token IDs do not align with examples")
    child_mask = torch.cat(
        (
            parent_attention_mask,
            torch.ones(
                (len(examples), 1),
                dtype=parent_attention_mask.dtype,
                device=parent_attention_mask.device,
            ),
        ),
        dim=1,
    )
    with torch.inference_mode():
        child = engine._base_model()(
            input_ids=committed,
            attention_mask=child_mask,
            past_key_values=parent_output.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    hidden = child.last_hidden_state
    if hidden.shape[:2] != (len(examples), 1):
        raise RuntimeError("committed continuation returned the wrong hidden-state geometry")
    return hidden[:, 0]


def _score_full_vocabulary_hidden(
    engine: HFEngine,
    hidden: torch.Tensor,
    expected_token_ids: Sequence[int],
) -> tuple[dict[str, list[float] | list[int]], int]:
    """Project one hidden row per branch, summarize exactly, and retain no logits."""

    expected = torch.as_tensor(
        list(expected_token_ids),
        dtype=torch.long,
        device=hidden.device,
    )
    if hidden.ndim != 2 or expected.shape != (hidden.shape[0],):
        raise ValueError("full-vocabulary readout rows do not align")
    output_embedding = engine.model.get_output_embeddings()
    with torch.inference_mode():
        logits = output_embedding(hidden)
    if logits.ndim != 2 or logits.shape[0] != hidden.shape[0]:
        raise RuntimeError("full-vocabulary readout returned the wrong geometry")
    if expected.max().item() >= logits.shape[1]:
        raise ValueError("expected token falls outside the model vocabulary")
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    predicted = logits.argmax(dim=1)
    target = logits[rows, expected]
    top_values, top_indices = logits.topk(k=2, dim=1)
    alternative = torch.where(
        top_indices[:, 0] == expected,
        top_values[:, 1],
        top_values[:, 0],
    )
    margin = target - alternative
    materialized_bytes = logits.numel() * logits.element_size()
    result: dict[str, list[float] | list[int]] = {
        "predicted_token_ids": predicted.detach().cpu().tolist(),
        "expected_token_ids": expected.detach().cpu().tolist(),
        "correct": (predicted == expected).float().cpu().tolist(),
        "expected_token_margin_vs_full_vocabulary_rest": margin.float().cpu().tolist(),
    }
    del logits
    return result, materialized_bytes


def _remove(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()


def _clean_context_bank(
    engine: HFEngine,
    examples: Sequence[CausalFamilyExample],
    layers: Sequence[int],
    *,
    batch_size: int,
    progress: ProgressCallback | None,
) -> tuple[dict[int, torch.Tensor], int]:
    _attention, projections = _attention_modules(engine)
    bank: dict[int, torch.Tensor] = {}
    calls = 0
    for start in range(0, len(examples), batch_size):
        stop = min(len(examples), start + batch_size)
        chosen = examples[start:stop]
        ids = [np.asarray(row.input_ids, dtype=np.int64) for row in chosen]
        query = torch.as_tensor(
            [row.query_position for row in chosen], dtype=torch.long, device=engine.device
        )
        batch_rows = torch.arange(len(chosen), device=engine.device)
        handles = []
        for layer in layers:

            def capture(
                _module: Any,
                hook_args: tuple[Any, ...],
                *,
                _layer: int = layer,
                _rows: torch.Tensor = batch_rows,
                _query: torch.Tensor = query,
                _start: int = start,
                _stop: int = stop,
            ) -> None:
                selected = hook_args[0][_rows, _query].detach()
                if _layer not in bank:
                    bank[_layer] = torch.empty(
                        (len(examples), selected.shape[-1]),
                        dtype=selected.dtype,
                        device=selected.device,
                    )
                bank[_layer][_start:_stop] = selected

            handles.append(projections[layer].register_forward_pre_hook(capture))
        try:
            _base_hidden(engine, ids, handles=handles)
        finally:
            _remove(handles)
        calls += 1
        if progress is not None:
            progress(
                {
                    "phase": "capture",
                    "completed_rows": stop,
                    "total_rows": len(examples),
                    "physical_forward_calls": calls,
                }
            )
    return bank, calls


def _mask_hook(
    *,
    engine: HFEngine,
    row_families: Sequence[Mapping[int, tuple[int, ...]]],
    examples: Sequence[CausalFamilyExample],
    layer: int,
) -> Callable[..., tuple[tuple[Any, ...], dict[str, Any]]]:
    entries = [
        (row, head)
        for row, family in enumerate(row_families)
        for head in family.get(layer, ())
    ]
    row_index = torch.as_tensor([row for row, _head in entries], device=engine.device)
    head_index = torch.as_tensor([head for _row, head in entries], device=engine.device)
    query = torch.as_tensor(
        [examples[row].query_position for row, _head in entries], device=engine.device
    )
    source = torch.as_tensor(
        [examples[row].source_position for row, _head in entries], device=engine.device
    )

    def apply_mask(
        _module: Any,
        hook_args: tuple[Any, ...],
        hook_kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        attention_mask = hook_kwargs.get("attention_mask")
        if attention_mask is None:
            hidden = hook_kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError("attention module exposed neither a mask nor hidden states")
            tokens = hidden.shape[-2]
            expanded = torch.ones(
                (len(examples), int(engine.cfg["n_head"]), tokens, tokens),
                dtype=torch.bool,
                device=hidden.device,
            ).tril()
        else:
            expanded = attention_mask.expand(
                len(examples),
                int(engine.cfg["n_head"]),
                attention_mask.shape[-2],
                attention_mask.shape[-1],
            ).clone()
        blocked_value: bool | float
        if expanded.dtype == torch.bool:
            blocked_value = False
        else:
            blocked_value = torch.finfo(expanded.dtype).min
        expanded[row_index, head_index, query, source] = blocked_value
        hook_kwargs["attention_mask"] = expanded
        return hook_args, hook_kwargs

    return apply_mask


def _repair_hook(
    *,
    engine: HFEngine,
    row_families: Sequence[Mapping[int, tuple[int, ...]]],
    examples: Sequence[CausalFamilyExample],
    example_indices: Sequence[int],
    clean_bank: Mapping[int, torch.Tensor],
    layer: int,
    wrong_donor: bool,
) -> Callable[..., tuple[Any, ...]]:
    head_dim = int(
        getattr(engine.model.config, "head_dim", 0)
        or int(engine.cfg["hidden"]) // int(engine.cfg["n_head"])
    )
    rows = [row for row, family in enumerate(row_families) if family.get(layer)]

    def apply_repair(
        _module: Any,
        hook_args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        changed = hook_args[0].clone()
        for row in rows:
            family_heads = row_families[row][layer]
            columns = [
                column
                for head in family_heads
                for column in range(head * head_dim, (head + 1) * head_dim)
            ]
            original = example_indices[row]
            source_row = examples[row].donor_index if wrong_donor else original
            changed[row, examples[row].query_position, columns] = clean_bank[layer][
                source_row, columns
            ].to(changed.dtype)
        return (changed, *hook_args[1:])

    return apply_repair


def _score_hidden(
    engine: HFEngine,
    hidden: torch.Tensor,
    examples: Sequence[CausalFamilyExample],
    union_offsets: Mapping[int, int],
    union_weights: torch.Tensor,
) -> tuple[list[float], list[float]]:
    rows = torch.arange(len(examples), device=hidden.device)
    query = torch.as_tensor(
        [row.query_position for row in examples], dtype=torch.long, device=hidden.device
    )
    selected_hidden = hidden[rows, query]
    union_scores = selected_hidden @ union_weights.T
    offsets = torch.as_tensor(
        [[union_offsets[token] for token in row.candidate_token_ids] for row in examples],
        dtype=torch.long,
        device=hidden.device,
    )
    scores = union_scores.gather(1, offsets)
    labels = torch.as_tensor(
        [row.correct_candidate_index for row in examples],
        dtype=torch.long,
        device=hidden.device,
    )
    chosen = scores[rows, labels]
    alternatives = scores.clone()
    alternatives[rows, labels] = -torch.inf
    correct = (scores.argmax(dim=1) == labels).float()
    margin = chosen - alternatives.max(dim=1).values
    return correct.detach().float().cpu().tolist(), margin.detach().float().cpu().tolist()


def evaluate_causal_family_batch(
    engine: HFEngine,
    examples: Sequence[CausalFamilyExample],
    families: Sequence[Family],
    full_family: Family,
    *,
    arms: Sequence[str] = ("deletion", "repair", "wrong_repair"),
    max_batch: int = 64,
    capture_batch: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Evaluate selected causal arms over a family × example grid.

    Physical work is cell-batched. Clean pre-output attention contexts are captured once per
    example when a repair arm is requested; deletion-only execution skips that capture. The
    candidate vocabulary is projected once per physical batch. The returned family ordering
    exactly matches ``families``.
    """
    selected_arms = tuple(dict.fromkeys(str(arm) for arm in arms))
    allowed_arms = {"deletion", "repair", "wrong_repair"}
    if not selected_arms or any(arm not in allowed_arms for arm in selected_arms):
        raise ValueError(
            "arms must be a non-empty subset of deletion, repair, and wrong_repair"
        )
    if not isinstance(engine, HFEngine):
        raise TypeError("causal-family batch execution currently requires mrun HFEngine")
    if not examples or not families or max_batch <= 0:
        raise ValueError("examples, families, and max_batch must be non-empty/positive")
    if any(row.donor_index >= len(examples) for row in examples):
        raise ValueError("example donor_index is outside the workload")
    candidate_counts = {len(row.candidate_token_ids) for row in examples}
    if len(candidate_counts) != 1:
        raise ValueError("candidate counts must be identical across examples")
    layers = int(engine.n_layer)
    heads = int(engine.cfg["n_head"])
    normalized_full = _normalize_family(full_family, layers=layers, heads=heads)
    normalized = [
        _normalize_family(family, layers=layers, heads=heads) for family in families
    ]
    for family in normalized:
        if any(
            not set(selected).issubset(normalized_full.get(layer, ()))
            for layer, selected in family.items()
        ):
            raise ValueError("selected family is not contained in full_family")

    started = time.perf_counter()
    capture_size = min(len(examples), capture_batch or max_batch)
    if any(arm != "deletion" for arm in selected_arms):
        clean_bank, clean_calls = _clean_context_bank(
            engine,
            examples,
            sorted(normalized_full),
            batch_size=capture_size,
            progress=progress,
        )
    else:
        clean_bank, clean_calls = {}, 0
    union_tokens = tuple(
        dict.fromkeys(token for row in examples for token in row.candidate_token_ids)
    )
    union_offsets = {token: index for index, token in enumerate(union_tokens)}
    output_weight = engine.model.get_output_embeddings().weight
    union_index = torch.as_tensor(union_tokens, dtype=torch.long, device=output_weight.device)
    union_weights = output_weight.index_select(0, union_index)

    shape = (len(families), len(examples))
    results = {
        arm: {
            "correct": np.empty(shape, dtype=np.float32),
            "margin": np.empty(shape, dtype=np.float32),
        }
        for arm in selected_arms
    }
    cells = [
        (family, example)
        for family in range(len(families))
        for example in range(len(examples))
    ]
    intervention_calls = 0
    total_physical = clean_calls + len(selected_arms) * math.ceil(len(cells) / max_batch)
    attention, projections = _attention_modules(engine)
    for arm in results:
        for start in range(0, len(cells), max_batch):
            chunk = cells[start : start + max_batch]
            family_indices = [family for family, _example in chunk]
            example_indices = [example for _family, example in chunk]
            chosen_examples = [examples[index] for index in example_indices]
            row_families = [normalized[index] for index in family_indices]
            mask_families = (
                row_families if arm == "deletion" else [normalized_full] * len(chunk)
            )
            ids = [np.asarray(row.input_ids, dtype=np.int64) for row in chosen_examples]
            handles = []
            for layer in sorted({layer for family in mask_families for layer in family}):
                handles.append(
                    attention[layer].register_forward_pre_hook(
                        _mask_hook(
                            engine=engine,
                            row_families=mask_families,
                            examples=chosen_examples,
                            layer=layer,
                        ),
                        with_kwargs=True,
                    )
                )
            if arm != "deletion":
                for layer in sorted({layer for family in row_families for layer in family}):
                    handles.append(
                        projections[layer].register_forward_pre_hook(
                            _repair_hook(
                                engine=engine,
                                row_families=row_families,
                                examples=chosen_examples,
                                example_indices=example_indices,
                                clean_bank=clean_bank,
                                layer=layer,
                                wrong_donor=arm == "wrong_repair",
                            )
                        )
                    )
            try:
                hidden, _lengths = _base_hidden(engine, ids, handles=handles)
            finally:
                _remove(handles)
            correct, margin = _score_hidden(
                engine, hidden, chosen_examples, union_offsets, union_weights
            )
            for offset, (family_index, example_index) in enumerate(chunk):
                results[arm]["correct"][family_index, example_index] = correct[offset]
                results[arm]["margin"][family_index, example_index] = margin[offset]
            intervention_calls += 1
            if progress is not None:
                progress(
                    {
                        "phase": arm,
                        "completed_cells": min(len(cells), start + len(chunk)),
                        "total_cells": len(cells),
                        "physical_forward_calls": clean_calls + intervention_calls,
                        "total_physical_forward_calls": total_physical,
                    }
                )

    payload_rows = []
    for family_index in range(len(families)):
        payload_rows.append(
            {
                arm: {
                    "correct": results[arm]["correct"][family_index].tolist(),
                    "margin": results[arm]["margin"][family_index].tolist(),
                }
                for arm in results
            }
        )
    elapsed = time.perf_counter() - started
    return {
        "schema": "mrun-causal-family-batch-v1",
        "families": payload_rows,
        "telemetry": {
            "engine_backend": engine.backend,
            "device": str(engine.device),
            "dtype": str(next(engine.model.parameters()).dtype).removeprefix("torch."),
            "attention_implementation": str(
                getattr(engine.model.config, "_attn_implementation", "unknown")
            ),
            "examples": len(examples),
            "families": len(families),
            "logical_family_example_cells": len(cells),
            "arms": list(selected_arms),
            "logical_arm_rows": len(selected_arms) * len(cells),
            "max_physical_batch": max_batch,
            "capture_batch": capture_size,
            "clean_capture_forward_calls": clean_calls,
            "intervention_forward_calls": intervention_calls,
            "physical_forward_calls": clean_calls + intervention_calls,
            "full_vocabulary_logits_materialized": False,
            "candidate_union_size": len(union_tokens),
            "elapsed_seconds": elapsed,
            "logical_rows_per_second": (
                len(selected_arms) * len(cells) / elapsed if elapsed else None
            ),
        },
    }


def evaluate_causal_family_continuation_batch(
    engine: HFEngine,
    examples: Sequence[CausalFamilyContinuationExample],
    families: Sequence[Family],
    full_family: Family,
    *,
    arms: Sequence[str] = ("deletion", "repair", "wrong_repair"),
    max_batch: int = 64,
    capture_batch: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Replay an intervened parent prefix, commit one token, and score its child.

    Clean capture, deletion, correct repair, and wrong-donor repair execute at
    the parent prefix only. Hooks are removed before each branch's own public
    argmax is committed into that branch's KV state. Parent and child decisions
    are exact full-vocabulary last-row readouts summarized immediately; dense
    logits are never retained in the returned evidence.

    This is deliberately a correctness oracle, not the shared-prefix StateCut
    runtime. Every clean/intervention row executes its complete parent prefix;
    telemetry names and accounts that distinction explicitly.
    """

    selected_arms = tuple(dict.fromkeys(str(arm) for arm in arms))
    allowed_arms = {"deletion", "repair", "wrong_repair"}
    if not selected_arms or any(arm not in allowed_arms for arm in selected_arms):
        raise ValueError(
            "arms must be a non-empty subset of deletion, repair, and wrong_repair"
        )
    if not isinstance(engine, HFEngine):
        raise TypeError("closed-loop causal-family execution requires mrun HFEngine")
    if (
        not examples
        or not families
        or isinstance(max_batch, bool)
        or not isinstance(max_batch, int)
        or max_batch <= 0
    ):
        raise ValueError("examples, families, and max_batch must be non-empty/positive")
    if capture_batch is not None and (
        isinstance(capture_batch, bool)
        or not isinstance(capture_batch, int)
        or capture_batch <= 0
    ):
        raise ValueError("capture_batch must be a positive integer when supplied")
    if any(row.donor_index >= len(examples) for row in examples):
        raise ValueError("continuation donor_index is outside the workload")
    if len({len(row.input_ids) for row in examples}) != 1:
        raise ValueError("continuation workload must use one exact parent prefix width")
    candidate_counts = {len(row.candidate_token_ids) for row in examples}
    if len(candidate_counts) != 1:
        raise ValueError("continuation candidate counts must be identical")

    layers = int(engine.n_layer)
    heads = int(engine.cfg["n_head"])
    normalized_full = _normalize_family(full_family, layers=layers, heads=heads)
    normalized = [
        _normalize_family(family, layers=layers, heads=heads) for family in families
    ]
    for family in normalized:
        if any(
            not set(selected).issubset(normalized_full.get(layer, ()))
            for layer, selected in family.items()
        ):
            raise ValueError("selected continuation family is not contained in full_family")

    started = time.perf_counter()
    capture_size = min(len(examples), capture_batch or max_batch)
    attention, projections = _attention_modules(engine)
    clean_bank: dict[int, torch.Tensor] = {}
    clean = {
        "parent": {
            "predicted_token_ids": [],
            "expected_token_ids": [],
            "correct": [],
            "expected_token_margin_vs_full_vocabulary_rest": [],
        },
        "child": {
            "predicted_token_ids": [],
            "expected_token_ids": [],
            "correct": [],
            "expected_token_margin_vs_full_vocabulary_rest": [],
        },
    }
    full_vocabulary_bytes = 0
    peak_full_vocabulary_bytes = 0
    clean_parent_calls = 0
    clean_child_calls = 0
    for start in range(0, len(examples), capture_size):
        stop = min(len(examples), start + capture_size)
        chosen = examples[start:stop]
        query = torch.as_tensor(
            [row.query_position for row in chosen], dtype=torch.long, device=engine.device
        )
        batch_rows = torch.arange(len(chosen), device=engine.device)
        handles = []
        for layer in sorted(normalized_full):

            def capture(
                _module: Any,
                hook_args: tuple[Any, ...],
                *,
                _layer: int = layer,
                _rows: torch.Tensor = batch_rows,
                _query: torch.Tensor = query,
                _start: int = start,
                _stop: int = stop,
            ) -> None:
                selected = hook_args[0][_rows, _query].detach()
                if _layer not in clean_bank:
                    clean_bank[_layer] = torch.empty(
                        (len(examples), selected.shape[-1]),
                        dtype=selected.dtype,
                        device=selected.device,
                    )
                clean_bank[_layer][_start:_stop] = selected

            handles.append(projections[layer].register_forward_pre_hook(capture))
        try:
            parent_output, parent_mask = _parent_cached_output(
                engine, chosen, handles=handles
            )
        finally:
            _remove(handles)
        clean_parent_calls += 1
        parent_hidden = parent_output.last_hidden_state[batch_rows, query]
        parent_score, parent_bytes = _score_full_vocabulary_hidden(
            engine,
            parent_hidden,
            [row.expected_parent_token_id for row in chosen],
        )
        child_hidden = _committed_child_hidden(
            engine,
            chosen,
            parent_output=parent_output,
            parent_attention_mask=parent_mask,
            committed_token_ids=parent_score["predicted_token_ids"],
        )
        clean_child_calls += 1
        child_score, child_bytes = _score_full_vocabulary_hidden(
            engine,
            child_hidden,
            [
                row.candidate_token_ids[row.correct_candidate_index]
                for row in chosen
            ],
        )
        for field in clean["parent"]:
            clean["parent"][field].extend(parent_score[field])
            clean["child"][field].extend(child_score[field])
        full_vocabulary_bytes += parent_bytes + child_bytes
        peak_full_vocabulary_bytes = max(
            peak_full_vocabulary_bytes, parent_bytes, child_bytes
        )
        if progress is not None:
            progress(
                {
                    "phase": "clean-parent-capture-and-committed-child",
                    "completed_rows": stop,
                    "total_rows": len(examples),
                    "physical_forward_calls": clean_parent_calls + clean_child_calls,
                }
            )

    shape = (len(families), len(examples))
    results = {
        arm: {
            phase: {
                "predicted_token_ids": np.empty(shape, dtype=np.int64),
                "expected_token_ids": np.empty(shape, dtype=np.int64),
                "correct": np.empty(shape, dtype=np.float32),
                "expected_token_margin_vs_full_vocabulary_rest": np.empty(
                    shape, dtype=np.float32
                ),
            }
            for phase in ("parent", "child")
        }
        for arm in selected_arms
    }
    cells = [
        (family, example)
        for family in range(len(families))
        for example in range(len(examples))
    ]
    intervention_parent_calls = 0
    intervention_child_calls = 0
    for arm in selected_arms:
        for start in range(0, len(cells), max_batch):
            chunk = cells[start : start + max_batch]
            family_indices = [family for family, _example in chunk]
            example_indices = [example for _family, example in chunk]
            chosen = [examples[index] for index in example_indices]
            row_families = [normalized[index] for index in family_indices]
            mask_families = (
                row_families if arm == "deletion" else [normalized_full] * len(chunk)
            )
            handles = []
            for layer in sorted({layer for family in mask_families for layer in family}):
                handles.append(
                    attention[layer].register_forward_pre_hook(
                        _mask_hook(
                            engine=engine,
                            row_families=mask_families,
                            examples=chosen,
                            layer=layer,
                        ),
                        with_kwargs=True,
                    )
                )
            if arm != "deletion":
                for layer in sorted({layer for family in row_families for layer in family}):
                    handles.append(
                        projections[layer].register_forward_pre_hook(
                            _repair_hook(
                                engine=engine,
                                row_families=row_families,
                                examples=chosen,
                                example_indices=example_indices,
                                clean_bank=clean_bank,
                                layer=layer,
                                wrong_donor=arm == "wrong_repair",
                            )
                        )
                    )
            try:
                parent_output, parent_mask = _parent_cached_output(
                    engine, chosen, handles=handles
                )
            finally:
                _remove(handles)
            intervention_parent_calls += 1
            row_index = torch.arange(len(chosen), device=engine.device)
            query = torch.as_tensor(
                [row.query_position for row in chosen],
                dtype=torch.long,
                device=engine.device,
            )
            parent_score, parent_bytes = _score_full_vocabulary_hidden(
                engine,
                parent_output.last_hidden_state[row_index, query],
                [row.expected_parent_token_id for row in chosen],
            )
            child_hidden = _committed_child_hidden(
                engine,
                chosen,
                parent_output=parent_output,
                parent_attention_mask=parent_mask,
                committed_token_ids=parent_score["predicted_token_ids"],
            )
            intervention_child_calls += 1
            child_score, child_bytes = _score_full_vocabulary_hidden(
                engine,
                child_hidden,
                [
                    row.candidate_token_ids[row.correct_candidate_index]
                    for row in chosen
                ],
            )
            for offset, (family_index, example_index) in enumerate(chunk):
                for phase, score in (("parent", parent_score), ("child", child_score)):
                    for field in results[arm][phase]:
                        results[arm][phase][field][family_index, example_index] = score[
                            field
                        ][offset]
            full_vocabulary_bytes += parent_bytes + child_bytes
            peak_full_vocabulary_bytes = max(
                peak_full_vocabulary_bytes, parent_bytes, child_bytes
            )
            if progress is not None:
                progress(
                    {
                        "phase": f"{arm}-parent-and-committed-child",
                        "completed_cells": min(len(cells), start + len(chunk)),
                        "total_cells": len(cells),
                        "physical_forward_calls": (
                            clean_parent_calls
                            + clean_child_calls
                            + intervention_parent_calls
                            + intervention_child_calls
                        ),
                    }
                )

    payload_rows = []
    for family_index in range(len(families)):
        payload_rows.append(
            {
                arm: {
                    phase: {
                        field: values[family_index].tolist()
                        for field, values in results[arm][phase].items()
                    }
                    for phase in ("parent", "child")
                }
                for arm in selected_arms
            }
        )
    elapsed = time.perf_counter() - started
    clean_parent_rows = len(examples)
    intervention_parent_rows = len(selected_arms) * len(cells)
    full_prefix_parent_rows = clean_parent_rows + intervention_parent_rows
    parent_width = len(examples[0].input_ids)
    full_prefix_parent_token_positions = full_prefix_parent_rows * parent_width
    committed_child_token_positions = full_prefix_parent_rows
    return {
        "schema": "mrun-causal-family-committed-continuation-batch-v1",
        "clean": clean,
        "families": payload_rows,
        "telemetry": {
            "execution_path": "hf-full-prefix-continuation-reference-oracle",
            "reference_oracle": True,
            "production_path": False,
            "shared_prefix_statecut_materialized": False,
            "shared_prefix_reused_across_arms": False,
            "full_prefix_replayed_per_branch_row": True,
            "engine_backend": engine.backend,
            "device": str(engine.device),
            "dtype": str(next(engine.model.parameters()).dtype).removeprefix("torch."),
            "attention_implementation": str(
                getattr(engine.model.config, "_attn_implementation", "unknown")
            ),
            "examples": len(examples),
            "families": len(families),
            "logical_family_example_cells": len(cells),
            "arms": list(selected_arms),
            "logical_arm_rows": len(selected_arms) * len(cells),
            "max_physical_batch": max_batch,
            "capture_batch": capture_size,
            "clean_parent_forward_calls": clean_parent_calls,
            "clean_committed_child_forward_calls": clean_child_calls,
            "intervention_parent_forward_calls": intervention_parent_calls,
            "intervention_committed_child_forward_calls": intervention_child_calls,
            "physical_forward_calls": (
                clean_parent_calls
                + clean_child_calls
                + intervention_parent_calls
                + intervention_child_calls
            ),
            "parent_intervention_hooks_removed_before_commit": True,
            "branch_local_parent_kv_consumed_by_child": True,
            "branch_predicted_parent_argmax_committed": True,
            "forced_or_oracle_commit_used": False,
            "clean_full_prefix_parent_rows": clean_parent_rows,
            "intervention_full_prefix_parent_rows": intervention_parent_rows,
            "full_prefix_parent_rows": full_prefix_parent_rows,
            "full_prefix_parent_token_positions": full_prefix_parent_token_positions,
            "committed_child_token_positions": committed_child_token_positions,
            "teacher_token_positions": (
                full_prefix_parent_token_positions + committed_child_token_positions
            ),
            "shared_prefix_reused_token_positions": 0,
            "full_vocabulary_last_row_projection_calls": (
                clean_parent_calls
                + clean_child_calls
                + intervention_parent_calls
                + intervention_child_calls
            ),
            "full_vocabulary_logits_materialized_bytes": full_vocabulary_bytes,
            "peak_full_vocabulary_logits_bytes": peak_full_vocabulary_bytes,
            "retained_full_vocabulary_logits_bytes": 0,
            "elapsed_seconds": elapsed,
        },
    }


__all__ = [
    "CausalFamilyContinuationExample",
    "CausalFamilyExample",
    "evaluate_causal_family_batch",
    "evaluate_causal_family_continuation_batch",
]
