from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.engine.kernels import paged_forward as pf
from mrun.engine.paged import PagedEngine


def test_paged_cache_dimensions_reject_lossy_integer_coercion() -> None:
    with pytest.raises(TypeError, match="dimensions must be integers"):
        pf.BatchedPagedKVCache(True, 1, 1, 2, 4, "cpu")
    with pytest.raises(TypeError, match="dimensions must be integers"):
        pf.BatchedPagedKVCache(1, 1.5, 1, 2, 4, "cpu")


def test_pooled_dispatch_rejects_cross_cache_storage_alias_before_store_access() -> None:
    first = pf.BatchedPagedKVCache(1, 1, 1, 2, 4, "cpu")
    second = pf.BatchedPagedKVCache(1, 1, 1, 2, 4, "cpu")
    second.k = first.k
    leases = (first.mint_slot_lease(), second.mint_slot_lease())

    with pytest.raises(ValueError, match="pooled committed KV tensors"):
        pf.paged_forward_block_pooled(
            object(),
            np.asarray([[1], [2]], dtype=np.int64),
            (first, second),
            leases,
        )


def test_commit_stages_delta_and_rechecks_race_before_arena_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = pf.BatchedPagedKVCache(1, 1, 1, 2, 4, "cpu")
    cache.lengths = np.asarray([1], dtype=np.int64)
    delta = pf.PagedKVDelta(
        parent_epoch=cache.epoch,
        parent_lengths=(1,),
        cache_id=cache.cache_id,
        k=torch.ones((1, 1, 2, 1, 2), dtype=torch.float32),
        v=torch.full((1, 1, 2, 1, 2), 2.0, dtype=torch.float32),
        token_count=2,
    )
    before = (cache.k.clone(), cache.v.clone(), cache.lengths.copy(), cache.epoch)
    original_clone = torch.Tensor.clone

    def racing_clone(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        result = original_clone(tensor, *args, **kwargs)
        if tensor.data_ptr() == delta.v.data_ptr():
            delta.k.add_(1.0)
        return result

    monkeypatch.setattr(torch.Tensor, "clone", racing_clone)
    with pytest.raises(RuntimeError, match="changed while staging"):
        pf.commit_block(cache, delta, (2,))

    assert torch.equal(cache.k, before[0])
    assert torch.equal(cache.v, before[1])
    assert np.array_equal(cache.lengths, before[2])
    assert cache.epoch == before[3]


class _TinyRoutedStore:
    """Deterministic Qwen-shaped store with fail-closed logical head methods."""

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

    def __init__(self, *, allowed_head_methods: Sequence[str]) -> None:
        generator = torch.Generator().manual_seed(17)
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
        self.allowed_head_methods = frozenset(allowed_head_methods)
        self.calls: list[tuple[str, str]] = []

    def _guard_head(self, method: str, name: str) -> None:
        if name == "lm_head" and method not in self.allowed_head_methods:
            raise AssertionError(f"contract denied {method}({name!r})")

    def embed_rows(self, name: str, ids: np.ndarray) -> torch.Tensor:
        self._guard_head("embed_rows", name)
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

    def has(self, _name: str) -> bool:
        return False

    def row_blocks(
        self,
        name: str,
        bs: int = 3,
    ) -> Iterator[tuple[int, int, torch.Tensor]]:
        self._guard_head("row_blocks", name)
        self.calls.append(("row_blocks", name))
        for start in range(0, self.head.shape[0], bs):
            end = min(start + bs, self.head.shape[0])
            yield start, end, self.head[start:end]


def _cache() -> pf.BatchedPagedKVCache:
    cache = pf.BatchedPagedKVCache(
        nL=1,
        B=2,
        nKV=1,
        hd=2,
        capacity=8,
        device="cpu",
    )
    cache.lengths = np.asarray([1, 2], dtype=np.int64)
    cache.epoch = 7
    cache.k[0, 0, 0, 0] = torch.tensor([0.10, -0.20])
    cache.v[0, 0, 0, 0] = torch.tensor([0.30, 0.40])
    cache.k[0, 1, :2, 0] = torch.tensor([[0.50, -0.60], [0.70, 0.80]])
    cache.v[0, 1, :2, 0] = torch.tensor([[-0.90, 1.00], [1.10, -1.20]])
    return cache


def _snapshot(cache: pf.BatchedPagedKVCache) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
    return cache.k.clone(), cache.v.clone(), cache.lengths.copy(), cache.epoch


def _assert_unchanged(
    cache: pf.BatchedPagedKVCache,
    before: tuple[torch.Tensor, torch.Tensor, np.ndarray, int],
) -> None:
    keys, values, lengths, epoch = before
    assert torch.equal(cache.k, keys)
    assert torch.equal(cache.v, values)
    assert np.array_equal(cache.lengths, lengths)
    assert cache.epoch == epoch


TOKENS = np.asarray([[1, 2], [3, 4]], dtype=np.int64)


def _empty_prefill_cache() -> pf.BatchedPagedKVCache:
    return pf.BatchedPagedKVCache(
        nL=1,
        B=2,
        nKV=1,
        hd=2,
        capacity=4,
        device="cpu",
    )


def test_persistent_prefill_row_local_key_and_value_patches_are_isolated() -> None:
    ids = [np.asarray([1, 2], dtype=np.int64), np.asarray([3, 4], dtype=np.int64)]
    baseline = _empty_prefill_cache()
    patched = _empty_prefill_cache()
    store = _TinyRoutedStore(allowed_head_methods=("row_blocks",))
    plain = pf.paged_forward_kv_batch(store, ids, baseline)
    donor = torch.zeros((1, 2), dtype=torch.float32)
    edited = pf.paged_forward_kv_batch(
        store,
        ids,
        patched,
        key_patch_ops_by_layer_rows=[
            {0: [("position_replace", [0], donor)]},
            None,
        ],
        value_patch_ops_by_layer_rows=[
            {0: [("position_replace", [1], donor)]},
            None,
        ],
    )

    assert patched.lengths.tolist() == baseline.lengths.tolist() == [2, 2]
    assert patched.epoch == baseline.epoch == 1
    assert torch.count_nonzero(patched.k[0, 0, 0]) == 0
    assert torch.count_nonzero(patched.v[0, 0, 1]) == 0
    assert torch.equal(patched.k[0, 0, 1], baseline.k[0, 0, 1])
    assert torch.equal(patched.k[0, 1], baseline.k[0, 1])
    assert torch.equal(patched.v[0, 1], baseline.v[0, 1])
    assert not torch.equal(edited[0], plain[0])
    torch.testing.assert_close(edited[1], plain[1], rtol=0, atol=0)


def test_persistent_prefill_noop_maps_are_exact_and_bad_rows_fail_before_write() -> None:
    ids = [np.asarray([1, 2], dtype=np.int64), np.asarray([3, 4], dtype=np.int64)]
    baseline = _empty_prefill_cache()
    noop = _empty_prefill_cache()
    store = _TinyRoutedStore(allowed_head_methods=("row_blocks",))
    plain = pf.paged_forward_kv_batch(store, ids, baseline)
    same = pf.paged_forward_kv_batch(
        store,
        ids,
        noop,
        key_patch_ops_by_layer_rows=[None, None],
        value_patch_ops_by_layer_rows=[None, None],
    )
    torch.testing.assert_close(same, plain, rtol=0, atol=0)
    assert torch.equal(noop.k, baseline.k)
    assert torch.equal(noop.v, baseline.v)

    invalid = _empty_prefill_cache()
    before = _snapshot(invalid)
    with pytest.raises(ValueError, match="align"):
        pf.paged_forward_kv_batch(
            store,
            ids,
            invalid,
            key_patch_ops_by_layer_rows=[None],
        )
    _assert_unchanged(invalid, before)


def test_default_full_logits_and_hidden_contract_share_exact_normalized_body() -> None:
    full_store = _TinyRoutedStore(allowed_head_methods=("row_blocks",))
    full_cache = _cache()
    full_before = _snapshot(full_cache)
    full, full_delta = pf.paged_forward_block(full_store, TOKENS, full_cache)

    hidden_store = _TinyRoutedStore(allowed_head_methods=())
    hidden_cache = _cache()
    hidden_before = _snapshot(hidden_cache)
    hidden, hidden_delta = pf.paged_forward_block(
        hidden_store,
        TOKENS,
        hidden_cache,
        output_contract="hidden_state_only",
    )

    assert full.shape == (2, 2, 8)
    assert hidden.shape == (2, 2, 4)
    expected_full = hidden.float() @ hidden_store.head.T
    torch.testing.assert_close(full, expected_full, rtol=0, atol=2e-7)
    assert ("row_blocks", "lm_head") in full_store.calls
    assert all(name != "lm_head" for _method, name in hidden_store.calls)
    _assert_unchanged(full_cache, full_before)
    _assert_unchanged(hidden_cache, hidden_before)
    assert full_delta.parent_epoch == hidden_delta.parent_epoch == 7
    assert full_delta.parent_lengths == hidden_delta.parent_lengths == (1, 2)
    assert torch.equal(full_delta.k, hidden_delta.k)
    assert torch.equal(full_delta.v, hidden_delta.v)


def test_selected_rows_use_only_compact_head_lookup_and_remain_provisional() -> None:
    full_store = _TinyRoutedStore(allowed_head_methods=("row_blocks",))
    full, _ = pf.paged_forward_block(full_store, TOKENS, _cache())

    selected_store = _TinyRoutedStore(allowed_head_methods=("embed_rows",))
    cache = _cache()
    before = _snapshot(cache)
    selected_ids = (1, 5, 7)
    selected, delta = pf.paged_forward_block(
        selected_store,
        TOKENS,
        cache,
        output_contract="selected_token_rows",
        selected_rows=selected_ids,
    )

    assert selected.shape == (2, 2, 3)
    expected = full.index_select(2, torch.as_tensor(selected_ids))
    torch.testing.assert_close(selected, expected, rtol=0, atol=2e-7)
    assert ("embed_rows", "lm_head") in selected_store.calls
    assert ("row_blocks", "lm_head") not in selected_store.calls
    _assert_unchanged(cache, before)

    assert pf.commit_block(cache, delta, [2, 1]) == (2, 1)
    assert cache.lengths.tolist() == [3, 3]
    assert cache.epoch == 8
    assert torch.equal(cache.k[:, 0, 1:3], delta.k[:, 0, :2])
    assert torch.equal(cache.v[:, 1, 2:3], delta.v[:, 1, :1])
    with pytest.raises(RuntimeError, match="stale KV delta"):
        pf.commit_block(cache, delta, [0, 0])


def test_provisional_mutation_and_aliases_fail_before_commit() -> None:
    store = _TinyRoutedStore(allowed_head_methods=("row_blocks",))
    cache = _cache()
    before = _snapshot(cache)
    _output, delta = pf.paged_forward_block(store, TOKENS, cache)
    delta.k.add_(1)

    with pytest.raises(RuntimeError, match="changed after creation"):
        pf.commit_block(cache, delta, (2, 2))
    _assert_unchanged(cache, before)

    shared = torch.zeros((1, 2, 2, 1, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="must not alias"):
        pf.PagedKVDelta(
            parent_epoch=cache.epoch,
            parent_lengths=tuple(int(value) for value in cache.lengths),
            cache_id=cache.cache_id,
            k=shared,
            v=shared,
            token_count=2,
        )


def test_outer_inference_mode_keeps_cache_and_kernel_delta_version_tracked() -> None:
    with torch.inference_mode():
        cache = _cache()
        _output, delta = pf.paged_forward_block(
            _TinyRoutedStore(allowed_head_methods=("row_blocks",)),
            TOKENS,
            cache,
        )

    assert isinstance(cache.k._version, int)  # noqa: SLF001 - regression contract
    assert isinstance(delta.k._version, int)  # noqa: SLF001 - regression contract
    assert pf.commit_block(cache, delta, (1, 1)) == (1, 1)


def test_direct_commit_rejects_corrupt_cache_storage_without_partial_write() -> None:
    cache = _cache()
    _output, delta = pf.paged_forward_block(
        _TinyRoutedStore(allowed_head_methods=("row_blocks",)),
        TOKENS,
        cache,
    )
    before_k = cache.k.clone()
    before_lengths = cache.lengths.copy()
    cache.v = cache.k

    with pytest.raises(ValueError, match="must not alias"):
        pf.commit_block(cache, delta, (2, 2))
    assert torch.equal(cache.k, before_k)
    assert np.array_equal(cache.lengths, before_lengths)


def test_last_only_projects_only_the_final_provisional_position() -> None:
    full_store = _TinyRoutedStore(allowed_head_methods=("row_blocks",))
    all_logits, all_delta = pf.paged_forward_block(full_store, TOKENS, _cache())
    last_logits, last_delta = pf.paged_forward_block(
        _TinyRoutedStore(allowed_head_methods=("row_blocks",)),
        TOKENS,
        _cache(),
        last_only=True,
    )
    selected, selected_delta = pf.paged_forward_block(
        _TinyRoutedStore(allowed_head_methods=("embed_rows",)),
        TOKENS,
        _cache(),
        output_contract="selected_token_rows",
        selected_rows=(1, 5, 7),
        last_only=True,
    )

    assert last_logits.shape == (2, 8)
    assert selected.shape == (2, 3)
    torch.testing.assert_close(last_logits, all_logits[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(
        selected,
        all_logits[:, -1].index_select(1, torch.as_tensor((1, 5, 7))),
        rtol=0,
        atol=2e-7,
    )
    assert torch.equal(all_delta.k, last_delta.k)
    assert torch.equal(all_delta.v, last_delta.v)
    assert torch.equal(all_delta.k, selected_delta.k)
    assert torch.equal(all_delta.v, selected_delta.v)


@pytest.mark.parametrize(
    ("output_contract", "selected_rows", "message"),
    [
        ("unknown", (), "output_contract must be one of"),
        ("selected_token_rows", (), "requires selected_rows"),
        ("selected_token_rows", (1, 1), "must be unique"),
        ("selected_token_rows", (-1, 2), "must be inside"),
        ("selected_token_rows", (1, 8), "must be inside"),
        ("full_logits", (1,), "only legal"),
        ("hidden_state_only", (1,), "only legal"),
    ],
)
def test_invalid_output_contracts_fail_before_store_access(
    output_contract: str,
    selected_rows: tuple[int, ...],
    message: str,
) -> None:
    store = _TinyRoutedStore(allowed_head_methods=())
    cache = _cache()
    before = _snapshot(cache)

    with pytest.raises(ValueError, match=message):
        pf.paged_forward_block(
            store,
            TOKENS,
            cache,
            output_contract=output_contract,  # type: ignore[arg-type]
            selected_rows=selected_rows,
        )

    assert store.calls == []
    _assert_unchanged(cache, before)


def test_paged_engine_stateful_seam_returns_an_uncommitted_selected_delta() -> None:
    engine = object.__new__(PagedEngine)
    engine.arch = "qwen2"
    engine.store = _TinyRoutedStore(allowed_head_methods=("embed_rows",))
    engine.composite_store = None
    engine.component_output_contract = None
    cache = _cache()
    before = _snapshot(cache)
    binding = SimpleNamespace(
        state=cache,
        parent_epoch=cache.epoch,
        parent_lengths=tuple(int(value) for value in cache.lengths),
    )
    plan = SimpleNamespace(
        output_contract=SimpleNamespace(value="selected_token_rows"),
        required_output_rows=(1, 5, 7),
        candidate_token_ids=(),
    )

    output, delta = engine.execute_workplan_stateful(
        plan,
        [TOKENS[0], TOKENS[1]],
        binding,
    )

    assert output.shape == (2, 3)
    assert delta.parent_epoch == 7
    assert delta.parent_lengths == (1, 2)
    _assert_unchanged(cache, before)
    assert pf.commit_block(cache, delta, [2, 1]) == (2, 1)
