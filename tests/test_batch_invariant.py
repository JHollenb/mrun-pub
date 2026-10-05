from __future__ import annotations

import numpy as np
import pytest
import torch
from test_kv_statecut import PARENTS, _TinyEngine, _TinyStore

from mrun.engine import prefill_paged_kv_statecut
from mrun.engine.kernels import paged_forward as pf
from mrun.engine.kernels.batch_invariant import (
    batch_invariant_attention,
    batch_invariant_attention_reference,
    batch_invariant_matmul,
    batch_invariant_rms_norm,
)


class _WeightStore(_TinyStore):
    """Tiny paged store that also exposes the dequantized-matrix seam used by the lane."""

    def weight(self, name: str) -> torch.Tensor:
        self.calls.append(("weight", name))
        return self.weights[name]


class _WeightEngine(_TinyEngine):
    def __init__(self) -> None:
        super().__init__()
        self.store = _WeightStore()


def test_matmul_matches_dense_and_int8_scaled_weights() -> None:
    generator = torch.Generator().manual_seed(3)
    activations = torch.randn(5, 7, generator=generator)
    weight = torch.randn(9, 7, generator=generator)
    torch.testing.assert_close(batch_invariant_matmul(activations, weight), activations @ weight.T)
    codes = torch.randint(-127, 128, (9, 7), generator=generator, dtype=torch.int8)
    scales = torch.rand(9, generator=generator) * 0.01
    expected = activations @ (codes.float() * scales[:, None]).T
    torch.testing.assert_close(batch_invariant_matmul(activations, codes, scales), expected)


def test_matmul_rows_do_not_depend_on_batch_composition() -> None:
    generator = torch.Generator().manual_seed(4)
    activations = torch.randn(6, 11, generator=generator)
    weight = torch.randn(13, 11, generator=generator)
    packed = batch_invariant_matmul(activations, weight)
    for row in range(activations.shape[0]):
        single = batch_invariant_matmul(activations[row : row + 1], weight)
        assert torch.equal(single[0], packed[row])


def test_rms_norm_matches_row_formula() -> None:
    generator = torch.Generator().manual_seed(5)
    activations = torch.randn(2, 3, 8, generator=generator)
    weight = torch.rand(8, generator=generator) + 0.5
    expected = activations * torch.rsqrt(activations.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
    torch.testing.assert_close(batch_invariant_rms_norm(activations, weight, 1e-6), expected)


def test_attention_reads_source_in_place_and_matches_masked_softmax() -> None:
    generator = torch.Generator().manual_seed(6)
    rows, tokens, heads, kv_heads, dim, capacity = 3, 2, 4, 2, 4, 9
    query = torch.randn(rows, tokens, heads, dim, generator=generator)
    new_keys = torch.randn(rows, tokens, kv_heads, dim, generator=generator)
    new_values = torch.randn(rows, tokens, kv_heads, dim, generator=generator)
    source_keys = torch.randn(2, capacity, kv_heads, dim, generator=generator)
    source_values = torch.randn(2, capacity, kv_heads, dim, generator=generator)
    row_ids, source_rows, past = [0, 1, 2], [1, 0, 1], [3, 5, 0]
    output = batch_invariant_attention(
        query, new_keys, new_values, source_keys, source_values, row_ids, source_rows, past
    )
    reference = batch_invariant_attention_reference(
        query, new_keys, new_values, source_keys, source_values, row_ids, source_rows, past
    )
    torch.testing.assert_close(output, reference)
    # Independent recomputation for row 1 with an explicit concatenated source.
    keys = torch.cat((source_keys[0, :5], new_keys[1]), dim=0).repeat_interleave(2, dim=1)
    values = torch.cat((source_values[0, :5], new_values[1]), dim=0).repeat_interleave(2, dim=1)
    for token in range(tokens):
        visible = 5 + token + 1
        scores = torch.einsum("hd,shd->hs", query[1, token], keys[:visible]) * dim**-0.5
        expected = torch.einsum("hs,shd->hd", scores.softmax(-1), values[:visible])
        torch.testing.assert_close(output[1, token], expected)


def test_statecut_batch_invariant_lane_matches_row_stable_and_panel_width() -> None:
    engine = _WeightEngine()
    cut, _ = prefill_paged_kv_statecut(engine, PARENTS, retention_budget_bytes=8192, capacity=5)
    reference_cache = cut.cache
    tokens = np.asarray([[5, 6], [6, 5], [7, 1]], dtype=np.int64)
    stable, _ = pf.paged_forward_statecut_branches(
        engine.store,
        tokens,
        reference_cache,
        output_contract="full_logits",
        arithmetic="row_stable_split",
    )
    invariant, panel = pf.paged_forward_statecut_branches(
        engine.store,
        tokens,
        reference_cache,
        output_contract="full_logits",
        arithmetic="batch_invariant",
    )
    torch.testing.assert_close(invariant, stable, rtol=1e-5, atol=1e-6)
    assert panel.branch_count == 3
    single, _ = pf.paged_forward_statecut_branches(
        engine.store,
        tokens[1:2],
        reference_cache,
        output_contract="full_logits",
        arithmetic="batch_invariant",
    )
    assert torch.equal(single[0], invariant[1])
    assert ("weight", "L0.q") in engine.store.calls
    cut.abandon()


def test_statecut_continue_one_accepts_named_batch_invariant_contract() -> None:
    engine = _WeightEngine()
    cut, _ = prefill_paged_kv_statecut(engine, PARENTS, retention_budget_bytes=8192, capacity=5)
    cut.fork("a")
    cut.fork("b")
    continuation = cut.continue_one(
        {"a": np.asarray([5, 6]), "b": np.asarray([6, 5])},
        arithmetic="batch_invariant",
    )
    assert continuation.output_for("a").shape == (2, 4)
    receipt = cut.commit("a")
    assert receipt.transition_verified


def test_statecut_defaults_to_batch_invariant_and_records_it() -> None:
    engine = _WeightEngine()
    cut, _ = prefill_paged_kv_statecut(engine, PARENTS, retention_budget_bytes=8192, capacity=5)
    cut.fork("a")
    continuation = cut.continue_one({"a": np.asarray([5, 6])})
    assert continuation.arithmetic == "batch_invariant"
    assert continuation.to_dict()["arithmetic"] == "batch_invariant"
    assert ("weight", "L0.q") in engine.store.calls
    assert ("matmul_row_stable", "L0.q") not in engine.store.calls
    cut.abandon()


def test_statecut_rejects_unknown_arithmetic() -> None:
    engine = _WeightEngine()
    cut, _ = prefill_paged_kv_statecut(engine, PARENTS, retention_budget_bytes=8192, capacity=5)
    cut.fork("a")
    with pytest.raises(ValueError, match="StateCut arithmetic"):
        cut.continue_one({"a": np.asarray([5, 6])}, arithmetic="packed")


def test_pooled_batch_invariant_lane_matches_packed_and_is_width_invariant() -> None:
    import test_pooled_paged_kernel as pooled_fixtures

    class _PooledWeightStore(pooled_fixtures._TinyStore):
        def weight(self, name: str) -> torch.Tensor:
            return self.weights[name]

    caches = pooled_fixtures._cache_pair()
    tokens = pooled_fixtures.TOKENS

    def run(selected: tuple[int, ...], arithmetic: str) -> torch.Tensor:
        subset = tuple(caches[index] for index in selected)
        leases = tuple(cache.mint_slot_lease() for cache in subset)
        try:
            output, _children = pf.paged_forward_block_pooled(
                _PooledWeightStore(),
                tokens[list(selected)],
                subset,
                leases,
                output_contract="full_logits",
                arithmetic=arithmetic,
            )
        finally:
            for cache, lease in zip(subset, leases, strict=True):
                cache.release_slot_lease(lease)
        return output

    packed = run((0, 1), "packed")
    invariant = run((0, 1), "batch_invariant")
    torch.testing.assert_close(invariant, packed, rtol=1e-5, atol=1e-6)
    assert torch.equal(run((1,), "batch_invariant")[0], invariant[1])
