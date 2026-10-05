from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.engine import PagedKVStateCut, prefill_paged_kv_statecut
from mrun.engine.kernels import paged_forward as pf


class _TinyStore:
    cfg = {
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "intermediate_size": 6,
        "vocab_size": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
    }
    device = "cpu"

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(91)
        hidden = int(self.cfg["hidden_size"])
        intermediate = int(self.cfg["intermediate_size"])
        vocab = int(self.cfg["vocab_size"])
        self.embedding = torch.randn(vocab, hidden, generator=generator) * 0.2
        self.head = torch.randn(vocab, hidden, generator=generator) * 0.2
        self.weights = {
            "L0.q": torch.randn(hidden, hidden, generator=generator) * 0.1,
            "L0.k": torch.randn(2, hidden, generator=generator) * 0.1,
            "L0.v": torch.randn(2, hidden, generator=generator) * 0.1,
            "L0.o": torch.randn(hidden, hidden, generator=generator) * 0.1,
            "L0.gate": torch.randn(intermediate, hidden, generator=generator) * 0.1,
            "L0.up": torch.randn(intermediate, hidden, generator=generator) * 0.1,
            "L0.down": torch.randn(hidden, intermediate, generator=generator) * 0.1,
        }
        self.norms = {
            "L0.ln1": torch.tensor([0.8, 0.9, 1.0, 1.1]),
            "L0.ln2": torch.tensor([1.1, 1.0, 0.9, 0.8]),
            "norm.final": torch.tensor([0.7, 0.9, 1.1, 1.3]),
        }
        self.calls: list[tuple[str, str]] = []

    def embed_rows(self, name: str, ids: np.ndarray) -> torch.Tensor:
        self.calls.append(("embed_rows", name))
        table = self.embedding if name == "embed" else self.head
        return table.index_select(0, torch.as_tensor(np.asarray(ids), dtype=torch.long))

    def fp32(self, name: str) -> torch.Tensor:
        self.calls.append(("fp32", name))
        return self.norms[name]

    def matmul(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(("matmul", name))
        return value @ self.weights[name].T

    def weight(self, name: str) -> torch.Tensor:
        self.calls.append(("weight", name))
        return self.weights[name]

    def matmul_row_stable(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(("matmul_row_stable", name))
        weight = self.weights[name]
        return torch.cat(
            tuple(value[row : row + 1] @ weight.T for row in range(int(value.shape[0]))),
            dim=0,
        )

    def has(self, _name: str) -> bool:
        return False

    def row_blocks(
        self,
        name: str,
        bs: int = 3,
    ) -> Iterator[tuple[int, int, torch.Tensor]]:
        self.calls.append(("row_blocks", name))
        for start in range(0, self.head.shape[0], bs):
            end = min(start + bs, self.head.shape[0])
            yield start, end, self.head[start:end]


class _TinyEngine:
    backend = "paged"
    arch = "qwen2"
    numerical_contract = "tiny-row-stable"

    def __init__(self, *, transactional: bool = True) -> None:
        self.store = _TinyStore()
        self._execution_lock = threading.RLock()
        self._transactional = transactional

    def capabilities(self) -> SimpleNamespace:
        return SimpleNamespace(transactional_kv=self._transactional)


PARENTS = (
    np.asarray([1, 2], dtype=np.int64),
    np.asarray([3, 4], dtype=np.int64),
)


def _prefill(
    *,
    budget: int = 4096,
) -> tuple[_TinyEngine, PagedKVStateCut, torch.Tensor]:
    engine = _TinyEngine()
    cut, output = prefill_paged_kv_statecut(
        engine,
        PARENTS,
        retention_budget_bytes=budget,
        capacity=5,
    )
    return engine, cut, output


def _clone_cache(cache: pf.BatchedPagedKVCache) -> pf.BatchedPagedKVCache:
    copied = pf.BatchedPagedKVCache(
        int(cache.k.shape[0]),
        cache.B,
        int(cache.k.shape[3]),
        int(cache.k.shape[4]),
        cache.capacity,
        cache.k.device,
    )
    copied.k.copy_(cache.k)
    copied.v.copy_(cache.v)
    copied.lengths = cache.lengths.copy()
    copied.epoch = cache.epoch
    return copied


def _snapshot(
    cache: pf.BatchedPagedKVCache,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
    return cache.k.clone(), cache.v.clone(), cache.lengths.copy(), int(cache.epoch)


def _assert_snapshot(
    cache: pf.BatchedPagedKVCache,
    snapshot: tuple[torch.Tensor, torch.Tensor, np.ndarray, int],
) -> None:
    assert torch.equal(cache.k, snapshot[0])
    assert torch.equal(cache.v, snapshot[1])
    assert np.array_equal(cache.lengths, snapshot[2])
    assert cache.epoch == snapshot[3]


def _assert_metadata_only(value: object) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_metadata_only(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_metadata_only(child)
    else:
        assert not isinstance(value, (torch.Tensor, np.ndarray))


def test_prefill_seals_one_parent_and_skips_the_vocabulary_head() -> None:
    engine, cut, prefill_output = _prefill()

    assert prefill_output.shape == (2, 4)
    assert engine.store.calls.count(("row_blocks", "lm_head")) == 0
    assert tuple(cut.cache.lengths) == (2, 2)
    assert cut.cache.epoch == 1
    descriptor = cut.descriptor
    assert descriptor.parent_allocated_bytes == 2 * 1 * 2 * 5 * 1 * 2 * 4
    assert descriptor.parent_committed_bytes == 2 * 1 * 2 * 2 * 1 * 2 * 4
    payload = descriptor.to_dict()
    _assert_metadata_only(payload)
    assert payload["duplicated_parent_kv_bytes"] == 0
    assert payload["serialized_parent_kv_bytes"] == 0
    assert len(json.dumps(payload)) < 2000

    cut.abandon()


def test_batched_branches_match_one_branch_reference_and_commit_atomically() -> None:
    engine, cut, _prefill_output = _prefill()
    parent_before = _snapshot(cut.cache)
    reference_cache = _clone_cache(cut.cache)
    cut.fork("accept")
    cut.fork("counterfactual")
    continuation = cut.continue_one(
        {
            "accept": np.asarray([5, 6]),
            "counterfactual": np.asarray([6, 5]),
        }
    )

    _assert_snapshot(cut.cache, parent_before)
    reference_output, reference_panel = pf.paged_forward_statecut_branches(
        _TinyStore(),
        np.asarray([[5, 6]], dtype=np.int64),
        reference_cache,
    )
    torch.testing.assert_close(
        continuation.output_for("accept"),
        reference_output[0],
        rtol=0,
        atol=0,
    )
    pf.commit_block(reference_cache, reference_panel.select(0), (1, 1))

    receipt = cut.commit("accept")
    assert receipt.transition_verified
    assert receipt.committed_branch_id == "accept"
    assert receipt.abandoned_branch_ids == ("counterfactual",)
    assert receipt.parent_epoch_after == receipt.parent_epoch_before + 1
    assert receipt.parent_lengths_after == (3, 3)
    assert torch.equal(cut.cache.k, reference_cache.k)
    assert torch.equal(cut.cache.v, reference_cache.v)
    assert np.array_equal(cut.cache.lengths, reference_cache.lengths)
    assert cut.cache.epoch == reference_cache.epoch
    assert cut.cache._active_statecut_id is None
    with pytest.raises(RuntimeError, match="terminal"):
        cut.commit("counterfactual")


def test_three_token_branch_block_commits_once_or_restores_exact_parent() -> None:
    _engine, cut, _prefill_output = _prefill()
    parent_before = _snapshot(cut.cache)
    reference_cache = _clone_cache(cut.cache)
    tokens = np.asarray([[5, 6, 7], [6, 5, 7]], dtype=np.int64)
    cut.fork("route")
    continuation = cut.continue_block(
        {"route": tokens}
    )

    assert continuation.output_for("route").shape == (2, 4)
    assert cut._panel is not None and cut._panel.token_count == 3
    _assert_snapshot(cut.cache, parent_before)
    reference_output, reference_panel = pf.paged_forward_statecut_branch_blocks(
        _TinyStore(),
        tokens.reshape(1, 2, 3),
        reference_cache,
    )
    torch.testing.assert_close(
        continuation.output_for("route"), reference_output[0], rtol=0, atol=0
    )
    pf.commit_block(reference_cache, reference_panel.select(0), (3, 3))
    receipt = cut.commit("route")
    assert receipt.transition_verified
    assert receipt.parent_epoch_after == receipt.parent_epoch_before + 1
    assert receipt.parent_lengths_after == (5, 5)
    assert torch.equal(cut.cache.k, reference_cache.k)
    assert torch.equal(cut.cache.v, reference_cache.v)
    assert np.array_equal(cut.cache.lengths, reference_cache.lengths)
    assert cut.cache.epoch == reference_cache.epoch

    _engine, rejected, _prefill_output = _prefill()
    rejected_before = _snapshot(rejected.cache)
    rejected.fork("route")
    rejected.continue_block(
        {"route": np.asarray([[5, 6, 7], [6, 5, 7]], dtype=np.int64)}
    )
    abandoned = rejected.abandon()
    assert abandoned.exact_parent_restored and abandoned.parent_storage_unchanged
    _assert_snapshot(rejected.cache, rejected_before)


def test_branch_diffs_never_alias_parent_and_mutation_fails_before_commit() -> None:
    _engine, cut, _prefill_output = _prefill()
    before = _snapshot(cut.cache)
    cut.fork("candidate")
    continuation = cut.continue_one({"candidate": np.asarray([5, 6])})
    panel = cut._panel
    assert panel is not None
    assert panel.k.untyped_storage().data_ptr() != cut.cache.k.untyped_storage().data_ptr()
    assert panel.v.untyped_storage().data_ptr() != cut.cache.v.untyped_storage().data_ptr()
    assert continuation.parent_copy_bytes == 0

    panel.k.add_(1.0)
    _assert_snapshot(cut.cache, before)
    with pytest.raises(RuntimeError, match="changed after continuation"):
        cut.commit("candidate")
    abandon = cut.abandon()
    assert abandon.exact_parent_restored and abandon.parent_storage_unchanged
    _assert_snapshot(cut.cache, before)


def test_abandon_proves_exact_parent_restoration_and_serializes_metadata_only() -> None:
    _engine, cut, _prefill_output = _prefill()
    before = _snapshot(cut.cache)
    cut.fork("left")
    cut.fork("right")
    continuation = cut.continue_one(
        {"left": np.asarray([5, 6]), "right": np.asarray([6, 5])},
        output_contract="selected_token_rows",
        selected_rows=(1, 7),
    )
    receipt = cut.abandon()

    _assert_snapshot(cut.cache, before)
    assert receipt.action == "abandon"
    assert receipt.exact_parent_restored
    assert receipt.parent_storage_unchanged
    assert receipt.parent_guard_before_sha256 == receipt.parent_guard_after_sha256
    assert receipt.retained_diff_bytes == continuation.retained_diff_bytes
    assert receipt.physical_forwards == 1
    for payload in (continuation.to_dict(), receipt.to_dict()):
        _assert_metadata_only(payload)
        assert payload["serialized_parent_kv_bytes"] == 0
        assert payload["serialized_diff_tensor_bytes"] == 0
        assert len(json.dumps(payload)) < 3000


def test_epoch_and_parent_mutation_invalidate_every_branch() -> None:
    _engine, cut, _prefill_output = _prefill()
    cut.fork("candidate")
    cut.continue_one({"candidate": np.asarray([5, 6])})
    cut.cache.epoch += 1

    with pytest.raises(RuntimeError, match="immutable parent changed"):
        cut.commit("candidate")


def test_retention_budget_fails_before_parent_or_branch_allocation() -> None:
    engine = _TinyEngine()
    with pytest.raises(MemoryError, match="planned parent KV"):
        PagedKVStateCut.prefill(
            engine,
            PARENTS,
            retention_budget_bytes=191,
            capacity=5,
        )

    _engine, cut, _prefill_output = _prefill(budget=250)
    cut.fork("left")
    cut.fork("right")
    with pytest.raises(MemoryError, match="branch diff"):
        cut.continue_one(
            {"left": np.asarray([5, 6]), "right": np.asarray([6, 5])}
        )
    assert cut._panel is None
    cut.abandon()


def test_unsupported_engine_and_geometry_fail_closed() -> None:
    with pytest.raises(NotImplementedError, match="transactional KV"):
        PagedKVStateCut.prefill(
            _TinyEngine(transactional=False),
            PARENTS,
            retention_budget_bytes=4096,
        )
    engine = _TinyEngine()
    engine.arch = "mamba"
    with pytest.raises(NotImplementedError, match="qwen2"):
        PagedKVStateCut.prefill(
            engine,
            PARENTS,
            retention_budget_bytes=4096,
        )


def _continue_two_program_branches(
    cut: PagedKVStateCut,
) -> object:
    cut.fork("rank-2-positive", rank=2, route="late-mlp", sign=1, dose=0.25)
    cut.fork("rank-8-negative", rank=8, route="late-mlp", sign=-1, dose=0.5)
    return cut.continue_one(
        {
            "rank-2-positive": np.asarray([5, 6]),
            "rank-8-negative": np.asarray([6, 5]),
        }
    )


def _row_stable_projection(hidden: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    flat = hidden.reshape(-1, int(hidden.shape[-1]))
    result = torch.cat(
        tuple(flat[row : row + 1] @ weights.T for row in range(int(flat.shape[0]))),
        dim=0,
    )
    return result.reshape(*hidden.shape[:-1], int(weights.shape[0]))


def test_selected_row_screening_reuses_one_suffix_and_batches_program_axis() -> None:
    engine, cut, _prefill_output = _prefill()
    parent_before = _snapshot(cut.cache)
    continuation = _continue_two_program_branches(cut)
    body_calls_before = tuple(
        call for call in engine.store.calls if call[1].startswith("L")
    )

    screening = cut.screen_selected_rows(
        (7, 1, 4),
        device_scratch_budget_bytes=1024,
    )

    # Screening projects aligned head blocks, matching full-vocabulary
    # adjudication. A three-column GEMM can round differently from that block.
    assert screening.memory.head_row_chunk_size == engine.store.cfg["vocab_size"]
    expected = _row_stable_projection(
        continuation.batched_outputs,
        engine.store.head,
    ).index_select(-1, torch.tensor([7, 1, 4]))
    torch.testing.assert_close(screening.scores, expected, rtol=0, atol=0)
    assert screening.scores.shape == (2, 2, 3)
    assert not screening.global_argmax_established
    assert "cannot establish the global" in screening.scope_statement
    assert [branch.rank for branch in screening.branch_axis] == [2, 8]
    assert [branch.sign for branch in screening.branch_axis] == [1, -1]
    assert [branch.dose for branch in screening.branch_axis] == [0.25, 0.5]
    assert screening.accounting.logical_branches == 2
    assert screening.accounting.logical_suffix_rows == 4
    assert screening.accounting.logical_score_values == 12
    assert screening.accounting.physical_suffix_forwards_total == 1
    assert screening.accounting.physical_suffix_forwards_this_call == 0
    assert screening.accounting.suffix_replays == 0
    assert screening.accounting.physical_selected_head_gathers_this_call == 1
    assert tuple(call for call in engine.store.calls if call[1].startswith("L")) == (
        body_calls_before
    )
    assert engine.store.calls.count(("row_blocks", "lm_head")) == 0
    _assert_snapshot(cut.cache, parent_before)
    _assert_metadata_only(screening.to_dict())
    assert screening.to_dict()["serialized_score_tensor_bytes"] == 0
    cut.abandon()


def test_full_vocab_adjudication_reuses_hidden_states_and_verifies_selected_parity() -> None:
    engine, cut, _prefill_output = _prefill()
    parent_before = _snapshot(cut.cache)
    continuation = _continue_two_program_branches(cut)
    screening = cut.screen_selected_rows(
        (7, 1, 4),
        device_scratch_budget_bytes=1024,
    )
    body_calls_before = tuple(
        call for call in engine.store.calls if call[1].startswith("L")
    )

    adjudication = cut.adjudicate_full_vocabulary(
        device_scratch_budget_bytes=1024,
    )

    expected = _row_stable_projection(continuation.batched_outputs, engine.store.head)
    torch.testing.assert_close(adjudication.logits, expected, rtol=0, atol=0)
    assert adjudication.global_argmax_established
    assert adjudication.screened_rows == screening.selected_rows
    assert adjudication.selected_screening_parity_verified
    assert torch.equal(
        adjudication.logits.index_select(-1, torch.tensor(screening.selected_rows)),
        screening.scores,
    )
    assert adjudication.accounting.logical_score_values == 2 * 2 * 8
    assert adjudication.accounting.physical_suffix_forwards_total == 1
    assert adjudication.accounting.physical_suffix_forwards_this_call == 0
    assert adjudication.accounting.suffix_replays == 0
    assert adjudication.accounting.physical_full_head_traversals_this_call == 1
    assert adjudication.accounting.physical_head_blocks_this_call == 1
    assert tuple(call for call in engine.store.calls if call[1].startswith("L")) == (
        body_calls_before
    )
    assert engine.store.calls.count(("row_blocks", "lm_head")) == 1
    _assert_snapshot(cut.cache, parent_before)
    _assert_metadata_only(adjudication.to_dict())
    assert adjudication.to_dict()["serialized_logit_tensor_bytes"] == 0
    cut.abandon()


def test_projection_preflight_rejects_retention_and_device_pressure_before_head_reads() -> None:
    engine, cut, _prefill_output = _prefill(budget=350)
    continuation = _continue_two_program_branches(cut)
    assert continuation.retained_total_bytes == 288
    calls_before = tuple(engine.store.calls)

    with pytest.raises(MemoryError, match="retention budget before allocation"):
        cut.adjudicate_full_vocabulary(device_scratch_budget_bytes=1024)
    assert tuple(engine.store.calls) == calls_before

    with pytest.raises(MemoryError, match="one head row"):
        cut.screen_selected_rows((1, 4), device_scratch_budget_bytes=100)
    assert tuple(engine.store.calls) == calls_before
    assert cut._screening is None and cut._adjudication is None
    cut.abandon()


def test_projection_chunks_to_device_budget_and_reports_every_physical_head_page() -> None:
    engine, cut, _prefill_output = _prefill()
    _continue_two_program_branches(cut)

    screening = cut.screen_selected_rows(
        (7, 1, 4),
        device_scratch_budget_bytes=120,
    )
    assert screening.memory.head_row_chunk_size == 1
    assert screening.memory.planned_device_scratch_peak_bytes == 120
    assert screening.accounting.physical_selected_head_gathers_this_call == 3

    adjudication = cut.adjudicate_full_vocabulary(
        device_scratch_budget_bytes=120,
    )
    assert adjudication.memory.head_row_chunk_size == 1
    assert adjudication.accounting.physical_full_head_traversals_this_call == 1
    assert adjudication.accounting.physical_head_blocks_this_call == 8
    assert adjudication.selected_screening_parity_verified
    cut.abandon()


def test_frozen_head_mutation_invalidates_adjudication_without_suffix_replay() -> None:
    engine, cut, _prefill_output = _prefill()
    _continue_two_program_branches(cut)
    screening = cut.screen_selected_rows((1, 4), device_scratch_budget_bytes=1024)
    calls_before = tuple(engine.store.calls)
    engine.store.head.add_(1.0)

    with pytest.raises(RuntimeError, match="frozen LM head changed"):
        cut.adjudicate_full_vocabulary(device_scratch_budget_bytes=1024)
    assert tuple(engine.store.calls) == calls_before
    assert screening.accounting.physical_suffix_forwards_total == 1
    cut.abandon()
