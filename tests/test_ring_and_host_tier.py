"""Ring prefetch (dense paged) and host page tier (qwen3-moe) unit tests.

The host-tier test runs everywhere via a stub store. The ring parity test needs a
CUDA-granted paged store and is gated like the other model tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from mrun.engine.qwen3_moe_cuda import HostPageTier


class _StubLayout:
    page_stride = 64


class _StubStore:
    """Duck-typed stand-in for PackedFP8ExpertStore: 3 layers x 4 experts."""

    layout = _StubLayout()
    layers = 3
    experts = 4

    def __init__(self):
        import tempfile

        rows = self.layers * self.experts
        self._data = (
            np.arange(rows * self.layout.page_stride, dtype=np.uint64)
            .astype(np.uint8)
            .reshape(rows, self.layout.page_stride)
        )
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.write(self._data.tobytes())
        self._tmp.flush()
        self.data_path = self._tmp.name  # warm() reads via pread, not mmap

    def _ensure_pages(self):
        return self._data

    def global_page(self, layer: int, expert: int) -> int:
        return layer * self.experts + expert


def test_host_tier_warm_serves_every_page_bit_exact():
    store = _StubStore()
    tier = HostPageTier(store, host_cache_mb=1.0, try_pin=False)
    tier.warm()
    assert tier.stats["warmed_pages"] == store.layers * store.experts
    target = torch.zeros((3, store.layout.page_stride), dtype=torch.uint8)
    tier.gather(1, [0, 2, 3], target)
    for i, expert in enumerate([0, 2, 3]):
        gid = store.global_page(1, expert)
        assert np.array_equal(target[i].numpy(), store._data[gid])
    assert tier.stats["served_pages"] == 3
    assert tier.stats["disk_pages"] == 0


def test_host_tier_admits_on_miss_within_budget():
    store = _StubStore()
    # room for 6 of 12 pages, no pre-warm
    tier = HostPageTier(store, host_cache_mb=6 * store.layout.page_stride / 1e6, try_pin=False)
    target = torch.zeros((2, store.layout.page_stride), dtype=torch.uint8)
    tier.gather(0, [1, 3], target)
    assert tier.stats["disk_pages"] == 2
    assert tier.stats["admitted_pages"] == 2
    tier.gather(0, [1, 3], target)  # second touch served from the tier
    assert tier.stats["served_pages"] == 2
    for i, expert in enumerate([1, 3]):
        assert np.array_equal(target[i].numpy(), store._data[store.global_page(0, expert)])


def test_pin_fill_cache_never_evicts_on_cyclic_access():
    """Measured basis (resident-knapsack PoC): LRU on the periodic paged forward gets
    0 hit bytes whenever budget < model; pin-fill retains first-fill = Belady-optimal."""
    from mrun.engine.kernels.qstore import QStore

    store = QStore.__new__(QStore)  # cache machinery only; no files needed
    store.set_cache_budget(0.001)  # 1000 bytes
    blocks = {f"b{i}": torch.zeros(100, dtype=torch.uint8) for i in range(20)}
    for _ in range(3):  # cyclic access, 20 x 100B > 1000B budget
        for name, W in blocks.items():
            if store._cache_get(name) is None:
                store._cache_put(name, W)
    kept = list(store._cache)
    assert kept == [f"b{i}" for i in range(10)], kept  # first-fill retained, no churn
    assert store._cache_bytes == 1000
    for name in kept:  # steady-state hits for the pinned set
        assert store._cache_get(name) is not None


def test_lru_policy_still_available_via_env(monkeypatch):
    from mrun.engine.kernels.qstore import QStore

    monkeypatch.setenv("MRUN_QSTORE_CACHE_POLICY", "lru")
    store = QStore.__new__(QStore)
    store.set_cache_budget(0.001)
    for i in range(20):
        store._cache_put(f"b{i}", torch.zeros(100, dtype=torch.uint8))
    assert list(store._cache) == [f"b{i}" for i in range(10, 20)]  # evicting tail


@pytest.mark.skipif(
    not os.environ.get("MRUN_RUN_MODEL_TESTS"),
    reason="needs MRUN_RUN_MODEL_TESTS=1 and a local qwen2.5-0.5b qstore",
)
def test_row_blocks_cache_hits_are_bit_exact():
    """lm_head chunks were the 27.6% cache blind spot; hits must be the same tensors."""
    from mrun.engine import open_engine

    model = os.environ.get("MRUN_RING_TEST_MODEL", "qwen2.5-0.5b")
    eng = open_engine(model, backend="paged")
    eng.store.set_cache_budget(4000)  # holds several 29 MB chunks
    first = [(s, e, W) for s, e, W in eng.store.row_blocks("lm_head")]
    second = [(s, e, W) for s, e, W in eng.store.row_blocks("lm_head")]
    hits = sum(1 for (_, _, a), (_, _, b) in zip(first, second, strict=True) if a is b)
    assert hits > 0, "no row_blocks cache hits under a 4 GB budget"
    for (_, _, a), (_, _, b) in zip(first, second, strict=True):
        if a is not b:
            assert torch.equal(a, b)


@pytest.mark.skipif(
    not os.environ.get("MRUN_RUN_MODEL_TESTS"),
    reason="needs MRUN_RUN_MODEL_TESTS=1 and a local qwen2.5-0.5b qstore",
)
def test_paged_kv_generate_token_parity_with_replay():
    """Persistent-KV decode must reproduce the replay path's greedy tokens (the gate
    is token parity, not bit-exact logits — packed-shape reduction-order lesson)."""
    from mrun.engine import open_engine

    model = os.environ.get("MRUN_RING_TEST_MODEL", "qwen2.5-0.5b")
    eng = open_engine(model, backend="paged")
    prompt = "The capital of France is"
    replay = eng.generate(prompt, max_new_tokens=12, kv=False)
    kv = eng.generate(prompt, max_new_tokens=12, kv=True)
    assert kv == replay, f"kv {kv} != replay {replay}"


def test_commit_block_transactional_semantics():
    """Ungated: epoch/parent-length staleness contract (dense-qstore semantics) and the
    accepted-prefix arena move, on a hand-built cache — no model store needed."""
    from mrun.engine.kernels.paged_forward import (
        BatchedPagedKVCache,
        PagedKVDelta,
        commit_block,
    )

    cache = BatchedPagedKVCache(nL=2, B=2, nKV=1, hd=4, capacity=8, device="cpu")
    delta = PagedKVDelta(
        parent_epoch=0,
        parent_lengths=(0, 0),
        cache_id=cache.cache_id,
        k=torch.arange(2 * 2 * 3 * 1 * 4, dtype=torch.float32).view(2, 2, 3, 1, 4),
        v=-torch.arange(2 * 2 * 3 * 1 * 4, dtype=torch.float32).view(2, 2, 3, 1, 4),
        token_count=3,
    )
    counts = commit_block(cache, delta, [3, 1])
    assert counts == (3, 1)
    assert cache.lengths.tolist() == [3, 1] and cache.epoch == 1
    assert torch.equal(cache.k[:, 0, :3], delta.k[:, 0, :3])
    assert torch.equal(cache.v[:, 1, :1], delta.v[:, 1, :1])
    assert torch.all(cache.k[:, 1, 1:] == 0)  # only the accepted prefix moved

    with pytest.raises(RuntimeError, match="stale"):
        commit_block(cache, delta, [0, 0])  # same delta again: epoch advanced

    stale_lengths = PagedKVDelta(
        parent_epoch=1,
        parent_lengths=(0, 0),
        cache_id=cache.cache_id,
        k=delta.k,
        v=delta.v,
        token_count=3,
    )
    with pytest.raises(RuntimeError, match="parent lengths"):
        commit_block(cache, stale_lengths, [1, 1])

    ok = PagedKVDelta(
        parent_epoch=1,
        parent_lengths=(3, 1),
        cache_id=cache.cache_id,
        k=delta.k,
        v=delta.v,
        token_count=3,
    )
    with pytest.raises(ValueError):
        commit_block(cache, ok, [4, 0])  # accepted length outside the block
    cache.lengths[0] = 7  # capacity 8: +2 overflows
    with pytest.raises(RuntimeError, match="overflow"):
        commit_block(
            cache,
            PagedKVDelta(1, (7, 1), cache.cache_id, delta.k, delta.v, 3),
            [2, 0],
        )


@pytest.mark.skipif(
    not os.environ.get("MRUN_RUN_MODEL_TESTS"),
    reason="needs MRUN_RUN_MODEL_TESTS=1 and a local qwen2.5-0.5b qstore",
)
def test_batched_paged_kv_row_parity_with_single_request():
    """Each row of a B=3 mixed-prompt batched KV greedy decode must equal the
    single-request generate(kv=True) tokens for that prompt (token parity, not
    bit-exact logits — packed-shape reduction-order lesson)."""
    from mrun.engine import open_engine
    from mrun.engine.kernels import paged_forward as pf

    model = os.environ.get("MRUN_RING_TEST_MODEL", "qwen2.5-0.5b")
    eng = open_engine(model, backend="paged")
    prompts = ["The capital of France is", "1+1=", "The quick brown fox jumps"]
    n_new = 8
    ids = [np.asarray(eng.encode([p], add_special_tokens=False)[0], np.int64) for p in prompts]
    c = eng.cfg
    cache = pf.BatchedPagedKVCache(
        eng.n_layer,
        len(ids),
        int(c["num_key_value_heads"]),
        int(c["head_dim"]),
        capacity=max(len(x) for x in ids) + n_new + 1,
        device=eng.device,
    )
    logits = pf.paged_forward_kv_batch(eng.store, ids, cache)  # batched prefill
    rows: list[list[int]] = [[] for _ in ids]
    nxt = logits.argmax(dim=-1)
    for _ in range(n_new):
        for b in range(len(ids)):
            rows[b].append(int(nxt[b]))
        logits = pf.paged_forward_kv_batch(
            eng.store, [np.asarray([int(nxt[b])], np.int64) for b in range(len(ids))], cache
        )
        nxt = logits.argmax(dim=-1)
    eos = getattr(eng.tokenizer, "eos_token_id", None)
    for b, prompt in enumerate(prompts):
        ref = eng.generate(prompt, max_new_tokens=n_new, kv=True)
        got = rows[b]
        if eos is not None and eos in got:
            got = got[: got.index(eos)]  # generate() stops at eos; batch keeps rolling
        assert got == ref, f"row {b} ({prompt!r}): batch {got} != single-request {ref}"


@pytest.mark.skipif(
    not os.environ.get("MRUN_RUN_MODEL_TESTS"),
    reason="needs MRUN_RUN_MODEL_TESTS=1 and a local qwen2.5-0.5b qstore",
)
def test_paged_forward_block_verify_parity():
    """Weight-stationary B×K verification: propose K=4 tokens from the model's own greedy
    path (oracle), forward_block + commit == sequential KV decode tokens. Also exercises
    a PARTIAL accept (rows at unequal committed depths) and stale-commit rejection.
    Gate is token parity, not bit-exact logits."""
    from mrun.engine import open_engine
    from mrun.engine.kernels import paged_forward as pf

    model = os.environ.get("MRUN_RING_TEST_MODEL", "qwen2.5-0.5b")
    eng = open_engine(model, backend="paged")
    c = eng.cfg
    nKV, hd = int(c["num_key_value_heads"]), int(c["head_dim"])
    prompts = ["The capital of France is", "1+1="]
    ids = [np.asarray(eng.encode([p], add_special_tokens=False)[0], np.int64) for p in prompts]
    n_ref = 9

    # oracle: sequential scalar KV decode (no eos handling — pure kernel path)
    refs: list[list[int]] = []
    for row_ids in ids:
        scalar = pf.PagedKVCache(
            eng.n_layer, nKV, hd, capacity=len(row_ids) + n_ref + 1, device=eng.device
        )
        logit_row = pf.paged_forward_kv(eng.store, row_ids, scalar)
        seq: list[int] = []
        for _ in range(n_ref):
            tok = int(logit_row.argmax().item())
            seq.append(tok)
            logit_row = pf.paged_forward_kv(eng.store, np.asarray([tok], np.int64), scalar)
        refs.append(seq)

    cache = pf.BatchedPagedKVCache(
        eng.n_layer,
        len(ids),
        nKV,
        hd,
        capacity=max(len(x) for x in ids) + n_ref + 4,
        device=eng.device,
    )
    prefill = pf.paged_forward_kv_batch(eng.store, ids, cache)
    for b in range(len(ids)):
        assert int(prefill[b].argmax()) == refs[b][0]

    # block 1: both rows propose their own greedy tokens 0..3 -> logits predict 1..4
    K = 4
    block = np.asarray([refs[0][0:K], refs[1][0:K]], np.int64)
    logits, delta = pf.paged_forward_block(eng.store, block, cache)
    assert logits.shape == (2, K, int(c["vocab_size"]))
    for b in range(2):
        pred = logits[b].argmax(dim=-1).tolist()
        assert pred == refs[b][1 : K + 1], f"row {b}: block {pred} != oracle {refs[b][1 : K + 1]}"

    # partial accept: row 0 takes all 4, row 1 only 2 -> rows now at UNEQUAL depths
    pf.commit_block(cache, delta, [4, 2])
    assert cache.lengths.tolist() == [len(ids[0]) + 4, len(ids[1]) + 2]

    # block 2 from unequal depths: row 0 proposes tokens 4..7, row 1 proposes 2..5
    block2 = np.asarray([refs[0][4:8], refs[1][2:6]], np.int64)
    logits2, delta2 = pf.paged_forward_block(eng.store, block2, cache)
    assert logits2[0].argmax(dim=-1).tolist() == refs[0][5:9]
    assert logits2[1].argmax(dim=-1).tolist() == refs[1][3:7]
    pf.commit_block(cache, delta2, [4, 4])

    with pytest.raises(RuntimeError, match="stale"):
        pf.commit_block(cache, delta2, [0, 0])  # already committed: epoch advanced


@pytest.mark.skipif(
    not (os.environ.get("MRUN_RUN_MODEL_TESTS") and torch.cuda.is_available()),
    reason="needs MRUN_RUN_MODEL_TESTS=1 and CUDA with a granted paged arch",
)
def test_ring_prefetch_logits_bit_exact_and_hitting():
    from mrun.engine import open_engine

    model = os.environ.get("MRUN_RING_TEST_MODEL", "qwen2.5-0.5b")
    eng = open_engine(model, backend="paged")
    assert eng.store.device != "cpu", "test requires GATHER_DEVICE_PAGED arch grant"
    ids = eng.encode(["The capital of France is"])[0]
    plain = [eng.logits(ids).clone() for _ in range(2)]
    eng.store.enable_ring()
    ringed = [eng.logits(ids).clone() for _ in range(3)]
    stats = eng.store.ring_stats()
    eng.store.disable_ring()
    for got in ringed:
        assert torch.equal(got, plain[0]), "ring path must be bit-exact"
    assert stats["hits"] + stats["late_hits"] > 0, f"ring never hit: {stats}"
