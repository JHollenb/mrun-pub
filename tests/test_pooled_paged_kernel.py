from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from uuid import uuid4

import numpy as np
import pytest
import torch

from mrun.engine.kernels import paged_forward as pf


class _TinyStore:
    """Deterministic Qwen-shaped store that records physical component traversals."""

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

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(31)
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
        rows = torch.as_tensor(np.asarray(ids), dtype=torch.long)
        table = self.embedding if name == "embed" else self.head
        return table.index_select(0, rows)

    def fp32(self, name: str) -> torch.Tensor:
        self.calls.append(("fp32", name))
        return self.norms[name]

    def matmul(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(("matmul", name))
        return value @ self.weights[name].T

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


def _new_cache(*, length: int, epoch: int, seed: int) -> pf.BatchedPagedKVCache:
    cache = pf.BatchedPagedKVCache(1, 1, 1, 2, capacity=8, device="cpu")
    cache.lengths = np.asarray([length], dtype=np.int64)
    cache.epoch = epoch
    generator = torch.Generator().manual_seed(seed)
    cache.k[:, 0, :length] = torch.randn(1, length, 1, 2, generator=generator)
    cache.v[:, 0, :length] = torch.randn(1, length, 1, 2, generator=generator)
    return cache


def _cache_pair() -> tuple[pf.BatchedPagedKVCache, pf.BatchedPagedKVCache]:
    return _new_cache(length=1, epoch=7, seed=41), _new_cache(length=3, epoch=13, seed=43)


def _equivalent_batch(
    caches: Sequence[pf.BatchedPagedKVCache],
) -> pf.BatchedPagedKVCache:
    combined = pf.BatchedPagedKVCache(1, len(caches), 1, 2, capacity=8, device="cpu")
    combined.lengths = np.asarray([int(cache.lengths[0]) for cache in caches], dtype=np.int64)
    for row, cache in enumerate(caches):
        length = int(cache.lengths[0])
        combined.k[:, row, :length] = cache.k[:, 0, :length]
        combined.v[:, row, :length] = cache.v[:, 0, :length]
    return combined


def _snapshot(
    cache: pf.BatchedPagedKVCache,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
    return cache.k.clone(), cache.v.clone(), cache.lengths.copy(), cache.epoch


def _assert_snapshot(
    cache: pf.BatchedPagedKVCache,
    snapshot: tuple[torch.Tensor, torch.Tensor, np.ndarray, int],
) -> None:
    keys, values, lengths, epoch = snapshot
    assert torch.equal(cache.k, keys)
    assert torch.equal(cache.v, values)
    assert np.array_equal(cache.lengths, lengths)
    assert cache.epoch == epoch


TOKENS = np.asarray([[1, 2], [4, 5]], dtype=np.int64)


@pytest.mark.parametrize(
    ("output_contract", "selected_rows"),
    [
        ("full_logits", ()),
        ("hidden_state_only", ()),
        ("selected_token_rows", (6, 1, 4)),
    ],
)
@pytest.mark.parametrize("last_only", [False, True])
def test_pooled_matches_equivalent_batched_cache_for_every_output_contract(
    output_contract: pf.PagedBlockOutputContract,
    selected_rows: tuple[int, ...],
    last_only: bool,
) -> None:
    caches = _cache_pair()
    leases = tuple(cache.mint_slot_lease() for cache in caches)
    before = tuple(_snapshot(cache) for cache in caches)
    batched = _equivalent_batch(caches)

    pooled_store = _TinyStore()
    pooled, children = pf.paged_forward_block_pooled(
        pooled_store,
        TOKENS,
        caches,
        leases,
        output_contract=output_contract,
        selected_rows=selected_rows,
        last_only=last_only,
    )
    reference, combined_delta = pf.paged_forward_block(
        _TinyStore(),
        TOKENS,
        batched,
        output_contract=output_contract,
        selected_rows=selected_rows,
        last_only=last_only,
    )

    torch.testing.assert_close(pooled, reference, rtol=0, atol=0)
    assert len(children) == 2
    assert children[0].k.untyped_storage().data_ptr() != children[1].k.untyped_storage().data_ptr()
    assert children[0].v.untyped_storage().data_ptr() != children[1].v.untyped_storage().data_ptr()
    for row, (cache, lease, child) in enumerate(zip(caches, leases, children, strict=True)):
        assert child.cache_id == cache.cache_id
        assert child.parent_epoch == cache.epoch
        assert child.parent_lengths == (int(cache.lengths[0]),)
        assert child.slot_lease == lease
        assert child.k.is_contiguous() and child.v.is_contiguous()
        torch.testing.assert_close(child.k, combined_delta.k[:, row : row + 1], rtol=0, atol=0)
        torch.testing.assert_close(child.v, combined_delta.v[:, row : row + 1], rtol=0, atol=0)
        _assert_snapshot(cache, before[row])

    # Packed execution visits every body component once, regardless of request count.
    for name in ("L0.q", "L0.k", "L0.v", "L0.o", "L0.gate", "L0.up", "L0.down"):
        assert pooled_store.calls.count(("matmul", name)) == 1
    expected_head_calls = 1 if output_contract == "full_logits" else 0
    assert pooled_store.calls.count(("row_blocks", "lm_head")) == expected_head_calls
    expected_selected_calls = 1 if output_contract == "selected_token_rows" else 0
    assert pooled_store.calls.count(("embed_rows", "lm_head")) == expected_selected_calls


@pytest.mark.parametrize(
    ("output_contract", "selected_rows"),
    [
        ("full_logits", ()),
        ("hidden_state_only", ()),
        ("selected_token_rows", (6, 1, 4)),
    ],
)
@pytest.mark.parametrize("last_only", [False, True])
def test_row_stable_pool_is_bit_exact_to_independent_b1_with_unequal_histories(
    output_contract: pf.PagedBlockOutputContract,
    selected_rows: tuple[int, ...],
    last_only: bool,
) -> None:
    caches = _cache_pair()
    leases = tuple(cache.mint_slot_lease() for cache in caches)
    reference_outputs: list[torch.Tensor] = []
    reference_deltas: list[pf.PagedKVDelta] = []
    for row, cache in enumerate(caches):
        output, delta = pf.paged_forward_block(
            _TinyStore(),
            TOKENS[row : row + 1],
            cache,
            output_contract=output_contract,
            selected_rows=selected_rows,
            last_only=last_only,
        )
        reference_outputs.append(output)
        reference_deltas.append(delta)

    pooled_store = _TinyStore()
    pooled, children = pf.paged_forward_block_pooled(
        pooled_store,
        TOKENS,
        caches,
        leases,
        output_contract=output_contract,
        selected_rows=selected_rows,
        last_only=last_only,
        arithmetic="row_stable",
    )

    assert torch.equal(pooled, torch.cat(reference_outputs, dim=0))
    for child, reference in zip(children, reference_deltas, strict=True):
        assert torch.equal(child.k, reference.k)
        assert torch.equal(child.v, reference.v)
    for name in ("L0.q", "L0.k", "L0.v", "L0.o", "L0.gate", "L0.up", "L0.down"):
        assert pooled_store.calls.count(("matmul_row_stable", name)) == 1
        assert pooled_store.calls.count(("matmul", name)) == 0
    expected_head_calls = 1 if output_contract == "full_logits" else 0
    assert pooled_store.calls.count(("row_blocks", "lm_head")) == expected_head_calls
    expected_selected_calls = 1 if output_contract == "selected_token_rows" else 0
    assert pooled_store.calls.count(("embed_rows", "lm_head")) == expected_selected_calls


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("token_count", [1, 2, 4])
@pytest.mark.parametrize(
    ("output_contract", "selected_rows"),
    [
        ("full_logits", ()),
        ("hidden_state_only", ()),
        ("selected_token_rows", (6, 1, 4)),
    ],
)
@pytest.mark.parametrize("last_only", [False, True])
def test_row_stable_pool_is_exhaustively_b1_exact_across_batch_block_and_prefix_shapes(
    batch: int,
    token_count: int,
    output_contract: pf.PagedBlockOutputContract,
    selected_rows: tuple[int, ...],
    last_only: bool,
) -> None:
    length_grid = (0, 3, 1, 2)
    caches = tuple(
        _new_cache(length=length_grid[row], epoch=7 + row, seed=101 + row) for row in range(batch)
    )
    leases = tuple(cache.mint_slot_lease() for cache in caches)
    tokens = np.asarray(
        [[((row * 3) + step + 1) % 8 for step in range(token_count)] for row in range(batch)],
        dtype=np.int64,
    )
    references = tuple(
        pf.paged_forward_block(
            _TinyStore(),
            tokens[row : row + 1],
            cache,
            output_contract=output_contract,
            selected_rows=selected_rows,
            last_only=last_only,
        )
        for row, cache in enumerate(caches)
    )

    pooled, children = pf.paged_forward_block_pooled(
        _TinyStore(),
        tokens,
        caches,
        leases,
        output_contract=output_contract,
        selected_rows=selected_rows,
        last_only=last_only,
        arithmetic="row_stable",
    )

    assert torch.equal(pooled, torch.cat([reference[0] for reference in references], dim=0))
    for child, (_output, reference_delta) in zip(children, references, strict=True):
        assert torch.equal(child.k, reference_delta.k)
        assert torch.equal(child.v, reference_delta.v)


def test_packed_pool_remains_the_default_and_row_stable_is_explicit() -> None:
    caches = _cache_pair()
    leases = tuple(cache.mint_slot_lease() for cache in caches)
    default_store = _TinyStore()
    explicit_store = _TinyStore()

    default, _ = pf.paged_forward_block_pooled(default_store, TOKENS, caches, leases)
    explicit, _ = pf.paged_forward_block_pooled(
        explicit_store,
        TOKENS,
        caches,
        leases,
        arithmetic="packed",
    )

    assert torch.equal(default, explicit)
    assert not any(method == "matmul_row_stable" for method, _name in default_store.calls)
    assert default_store.calls == explicit_store.calls


@pytest.mark.parametrize("arithmetic", ["", "stable", "ROW_STABLE", True, None])
def test_unknown_pooled_arithmetic_fails_before_component_access(arithmetic: object) -> None:
    caches = _cache_pair()
    leases = tuple(cache.mint_slot_lease() for cache in caches)
    store = _TinyStore()

    with pytest.raises(ValueError, match="pooled arithmetic"):
        pf.paged_forward_block_pooled(
            store,
            TOKENS,
            caches,
            leases,
            arithmetic=arithmetic,  # type: ignore[arg-type]
        )

    assert store.calls == []


@pytest.mark.parametrize(
    "groups",
    [
        ((6, 1, 4),),
        ((6, 1, 4), (1, 7)),
        ((6, 1, 4), ()),
        ((6, 1, 4), (1, 1)),
    ],
)
def test_invalid_row_stable_selected_groups_fail_before_component_access(
    groups: tuple[tuple[int, ...], ...],
) -> None:
    caches = _cache_pair()
    leases = tuple(cache.mint_slot_lease() for cache in caches)
    store = _TinyStore()

    with pytest.raises(ValueError, match="selected row group|selected_row_groups"):
        pf.paged_forward_block_pooled(
            store,
            TOKENS,
            caches,
            leases,
            output_contract="selected_token_rows",
            selected_rows=(6, 1, 4),
            selected_row_groups=groups,
            arithmetic="row_stable",
        )

    assert store.calls == []


def test_selected_groups_are_not_accepted_by_the_default_packed_lane() -> None:
    caches = _cache_pair()
    leases = tuple(cache.mint_slot_lease() for cache in caches)
    store = _TinyStore()

    with pytest.raises(ValueError, match="row-stable arithmetic"):
        pf.paged_forward_block_pooled(
            store,
            TOKENS,
            caches,
            leases,
            output_contract="selected_token_rows",
            selected_rows=(6, 1, 4),
            selected_row_groups=((6, 1), (1, 4)),
        )

    assert store.calls == []


def test_child_deltas_commit_independently_with_unequal_and_zero_accepts() -> None:
    caches = _cache_pair()
    leases = tuple(cache.mint_slot_lease() for cache in caches)
    before = tuple(_snapshot(cache) for cache in caches)
    _output, children = pf.paged_forward_block_pooled(_TinyStore(), TOKENS, caches, leases)

    assert pf.commit_block(caches[0], children[0], (2,)) == (2,)
    assert caches[0].lengths.tolist() == [3]
    assert torch.equal(caches[0].k[:, 0, 1:3], children[0].k[:, 0, :2])
    _assert_snapshot(caches[1], before[1])

    assert pf.commit_block(caches[1], children[1], (0,)) == (0,)
    assert caches[1].lengths.tolist() == [3]
    assert caches[1].epoch == before[1][3] + 1
    assert torch.equal(caches[1].k, before[1][0])
    assert torch.equal(caches[1].v, before[1][1])

    # The first sibling's commit did not stale the second cache; its delta was independently
    # consumable.  Each individual cache now rejects replay by its own epoch.
    with pytest.raises(RuntimeError, match="stale KV delta"):
        pf.commit_block(caches[0], children[0], (0,))
    with pytest.raises(RuntimeError, match="stale KV delta"):
        pf.commit_block(caches[1], children[1], (0,))


def test_outer_inference_mode_still_returns_version_tracked_child_deltas() -> None:
    caches = _cache_pair()
    leases = tuple(cache.mint_slot_lease() for cache in caches)

    with torch.inference_mode():
        _output, children = pf.paged_forward_block_pooled(_TinyStore(), TOKENS, caches, leases)

    assert all(isinstance(child.k._version, int) for child in children)  # noqa: SLF001
    assert all(isinstance(child.v._version, int) for child in children)  # noqa: SLF001
    assert pf.commit_block(caches[0], children[0], (1,)) == (1,)


def test_slot_lease_capabilities_reject_foreign_swapped_duplicate_released_and_aba() -> None:
    first, second = _cache_pair()
    first_lease = first.mint_slot_lease()
    second_lease = second.mint_slot_lease()

    with pytest.raises(RuntimeError, match="different cache"):
        pf.paged_forward_block_pooled(
            _TinyStore(), TOKENS, (first, second), (second_lease, first_lease)
        )
    with pytest.raises(ValueError, match="duplicate pooled"):
        pf.paged_forward_block_pooled(
            _TinyStore(), TOKENS, (first, second), (first_lease, first_lease)
        )
    with pytest.raises(RuntimeError, match="active lease"):
        first.mint_slot_lease()

    first.release_slot_lease(first_lease)
    with pytest.raises(RuntimeError, match="released"):
        first.validate_slot_lease(first_lease)
    with pytest.raises(RuntimeError, match="released"):
        pf.paged_forward_block_pooled(_TinyStore(), TOKENS[:1], (first,), (first_lease,))

    replacement = first.mint_slot_lease()
    assert replacement.generation == first_lease.generation + 1
    assert replacement.lease_id != first_lease.lease_id
    with pytest.raises(RuntimeError, match="stale"):
        first.validate_slot_lease(first_lease)
    forged = pf.PagedKVSlotLease(
        cache_id=first.cache_id,
        row=0,
        generation=replacement.generation,
        lease_id=uuid4().hex,
    )
    with pytest.raises(RuntimeError, match="identity"):
        first.validate_slot_lease(forged)
    with pytest.raises(RuntimeError, match="different cache"):
        second.validate_slot_lease(replacement)


def test_swapped_or_released_child_delta_cannot_commit_and_mutates_nothing() -> None:
    caches = _cache_pair()
    leases = tuple(cache.mint_slot_lease() for cache in caches)
    _output, children = pf.paged_forward_block_pooled(_TinyStore(), TOKENS, caches, leases)
    before = tuple(_snapshot(cache) for cache in caches)

    with pytest.raises(RuntimeError, match="different paged cache"):
        pf.commit_block(caches[1], children[0], (1,))
    for cache, snapshot in zip(caches, before, strict=True):
        _assert_snapshot(cache, snapshot)

    caches[0].release_slot_lease(leases[0])
    with pytest.raises(RuntimeError, match="released"):
        pf.commit_block(caches[0], children[0], (1,))
    _assert_snapshot(caches[0], before[0])

    replacement = caches[0].mint_slot_lease()
    with pytest.raises(RuntimeError, match="stale"):
        pf.commit_block(caches[0], children[0], (1,))
    caches[0].release_slot_lease(replacement)
    _assert_snapshot(caches[0], before[0])


def test_reversed_pooled_calls_share_one_global_lock_order_without_deadlock() -> None:
    first, second = _cache_pair()
    first_lease = first.mint_slot_lease()
    second_lease = second.mint_slot_lease()
    barrier = threading.Barrier(3)
    results: list[torch.Tensor] = []
    errors: list[BaseException] = []

    def run(
        caches: tuple[pf.BatchedPagedKVCache, pf.BatchedPagedKVCache],
        leases: tuple[pf.PagedKVSlotLease, pf.PagedKVSlotLease],
        tokens: np.ndarray,
    ) -> None:
        try:
            barrier.wait(timeout=2)
            output, _children = pf.paged_forward_block_pooled(
                _TinyStore(),
                tokens,
                caches,
                leases,
                output_contract="hidden_state_only",
            )
            results.append(output)
        except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
            errors.append(exc)

    forward = threading.Thread(
        target=run,
        args=((first, second), (first_lease, second_lease), TOKENS),
        daemon=True,
    )
    reverse = threading.Thread(
        target=run,
        args=((second, first), (second_lease, first_lease), TOKENS[::-1].copy()),
        daemon=True,
    )
    forward.start()
    reverse.start()
    barrier.wait(timeout=2)
    forward.join(timeout=5)
    reverse.join(timeout=5)

    assert not forward.is_alive() and not reverse.is_alive(), "multi-cache lock order deadlocked"
    assert errors == []
    assert len(results) == 2
    torch.testing.assert_close(results[0], results[1].flip(0), rtol=0, atol=0)
