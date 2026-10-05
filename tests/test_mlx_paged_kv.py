from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from mrun.runtime import (
    ComponentPlacement,
    FallbackPolicy,
    MemoryDomain,
    MlxKVPagePool,
    MlxNativeRuntime,
    MlxPagedKVCacheFactory,
    MlxPagedKVCapacityError,
    MlxPagedKVError,
    OutputMode,
    OutputRequest,
    PlacementPlan,
    PrefillWork,
    PromotionStatus,
    Residency,
    RuntimeRoute,
    StatePlacement,
)


class _FakeMx:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.concatenate_calls = 0
        self.fail_concatenate_once = False
        self.fail_eval_once = False

    @staticmethod
    def zeros(shape: tuple[int, ...], *, dtype: Any) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def concatenate(self, values: tuple[np.ndarray, ...], *, axis: int) -> np.ndarray:
        self.concatenate_calls += 1
        if self.fail_concatenate_once:
            self.fail_concatenate_once = False
            raise RuntimeError("injected device concatenate failure")
        return np.concatenate(values, axis=axis)

    def eval(self, *_values: Any) -> None:
        self.eval_calls += 1
        if self.fail_eval_once:
            self.fail_eval_once = False
            raise RuntimeError("injected device evaluation failure")


def _pool(
    *,
    mx: _FakeMx | None = None,
    page_size: int = 2,
    page_count: int = 8,
    key_head_dim: int = 2,
    value_head_dim: int = 3,
) -> tuple[MlxKVPagePool, _FakeMx]:
    resolved = mx or _FakeMx()
    return (
        MlxKVPagePool(
            mx=resolved,
            create_attention_mask=lambda *args, offset, **kwargs: (args, offset, kwargs),
            page_size=page_size,
            page_count=page_count,
            kv_heads=1,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            key_dtype=np.float32,
            value_dtype=np.float32,
        ),
        resolved,
    )


def _rows(start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    keys = np.arange(start, start + count * 2, dtype=np.float32).reshape(1, 1, count, 2)
    values = np.arange(start + 100, start + 100 + count * 3, dtype=np.float32).reshape(
        1,
        1,
        count,
        3,
    )
    return keys, values


def test_paged_cache_appends_across_fixed_device_pages_and_reports_materialization() -> None:
    pool, mx = _pool(page_count=4)
    cache = pool.new_cache(capacity=5)
    keys, values = _rows(0, 3)

    returned_keys, returned_values = cache.update_and_fetch(keys, values)

    assert cache.offset == 3
    assert returned_keys.shape == (1, 1, 3, 2)
    assert returned_values.shape == (1, 1, 3, 3)
    assert np.array_equal(returned_keys, keys)
    assert np.array_equal(returned_values, values)
    assert cache.keys is returned_keys
    assert cache.values is returned_values
    assert cache.make_mask("q", return_array=True) == (
        ("q",),
        3,
        {"return_array": True},
    )
    assert cache.nbytes == 5 * 5 * 4
    telemetry = pool.telemetry()
    assert telemetry.physical_bytes == 4 * 2 * 5 * 4
    assert telemetry.reserved_pages == telemetry.live_page_references == 2
    assert telemetry.shared_bytes == 0
    assert telemetry.logical_capacity_pages == 3
    assert telemetry.logical_capacity_bytes == 5 * pool.bytes_per_token
    assert telemetry.concatenate_materializations == 1
    assert mx.concatenate_calls == 2  # one K concat plus one V concat
    assert telemetry.materialized_bytes == 3 * 5 * 4
    assert telemetry.reconciled

    cache.reset(1)
    assert cache.offset == 1
    assert len(cache.page_authorities) == 1
    assert np.array_equal(cache.keys, keys[..., :1, :])
    with pytest.raises(ValueError, match="must not grow"):
        cache.reset(2)
    assert pool.telemetry().resets == 1
    assert pool.telemetry().reconciled

    cache.release()
    after = pool.telemetry()
    assert after.free_pages == 4
    assert after.active_caches == after.reserved_pages == after.leaked_pages == 0
    assert after.reconciled
    pool.close()


def test_full_pages_share_zero_copy_and_partial_fork_tail_is_private() -> None:
    pool, _mx = _pool()
    source = pool.new_cache(capacity=8)
    target = pool.new_cache(capacity=8)
    keys, values = _rows(0, 5)
    source.update_and_fetch(keys, values)

    target.copy_committed_prefix_from(source, 5)

    source_pages = source.page_authorities
    target_pages = target.page_authorities
    assert target_pages[:2] == source_pages[:2]
    assert target_pages[2] != source_pages[2]
    assert target.last_prefix_copy_bytes == pool.bytes_per_token
    assert np.array_equal(target.keys, source.keys)
    assert np.array_equal(target.values, source.values)
    telemetry = pool.telemetry()
    assert telemetry.full_page_prefix_shares == 2
    assert telemetry.partial_tail_copy_tokens == 1
    assert telemetry.reserved_pages == 4
    assert telemetry.live_page_references == 6
    assert telemetry.shared_pages == 2
    assert telemetry.shared_bytes == 2 * pool.page_bytes
    assert telemetry.reconciled

    source_tail = source.keys.copy()
    append_keys, append_values = _rows(50, 1)
    target.update_and_fetch(append_keys, append_values)
    assert np.array_equal(source.keys, source_tail)
    assert target.offset == 6

    target.release()
    source.release()
    assert pool.telemetry().reconciled
    pool.close()


def test_trim_into_shared_full_page_cows_before_append() -> None:
    pool, _mx = _pool(page_count=6)
    source = pool.new_cache(capacity=6)
    branch = pool.new_cache(capacity=6)
    keys, values = _rows(0, 4)
    source.update_and_fetch(keys, values)
    branch.copy_committed_prefix_from(source, 4)
    shared_second = source.page_authorities[1]
    assert branch.page_authorities[1] == shared_second

    assert branch.trim(1) == 1
    before_source = source.keys.copy()
    append_keys, append_values = _rows(80, 1)
    branch.update_and_fetch(append_keys, append_values)

    assert branch.offset == source.offset == 4
    assert branch.page_authorities[0] == source.page_authorities[0]
    assert branch.page_authorities[1] != shared_second
    assert source.page_authorities[1] == shared_second
    assert np.array_equal(source.keys, before_source)
    assert np.array_equal(branch.keys[..., :3, :], source.keys[..., :3, :])
    assert np.array_equal(branch.keys[..., 3:, :], append_keys)
    telemetry = pool.telemetry()
    assert telemetry.cow_pages == 1
    assert telemetry.cow_copy_tokens == 1
    assert telemetry.cow_copy_bytes == pool.bytes_per_token
    assert telemetry.reconciled

    branch.release()
    source.release()
    pool.close()


def test_exhaustion_and_concatenate_failure_leave_page_tables_reconciled() -> None:
    pool, mx = _pool(page_count=2)
    owner = pool.new_cache(capacity=4)
    shared = pool.new_cache(capacity=4)
    blocked = pool.new_cache(capacity=4)
    keys, values = _rows(0, 4)
    owner.update_and_fetch(keys, values)
    shared.copy_committed_prefix_from(owner, 4)
    assert shared.last_prefix_copy_bytes == 0
    assert shared.page_authorities == owner.page_authorities
    assert pool.telemetry().free_pages == 0
    blocked_before = blocked.page_authorities

    one_key, one_value = _rows(20, 1)
    with pytest.raises(MlxPagedKVCapacityError, match="lacks private append pages"):
        blocked.update_and_fetch(one_key, one_value)
    assert blocked.offset == 0
    assert blocked.page_authorities == blocked_before == ()
    assert pool.telemetry().reconciled

    shared.release()
    owner.release()
    mx.fail_concatenate_once = True
    first_keys, first_values = _rows(30, 1)
    blocked.update_and_fetch(first_keys, first_values)
    second_keys, second_values = _rows(40, 2)
    with pytest.raises(RuntimeError, match="injected device concatenate failure"):
        blocked.update_and_fetch(second_keys, second_values)
    assert blocked.offset == 1
    assert len(blocked.page_authorities) == 1
    assert np.array_equal(blocked.keys, first_keys)
    assert pool.telemetry().reconciled

    blocked.release()
    pool.close()


def test_failed_device_cleanup_quarantines_pages_until_explicit_reconciliation() -> None:
    pool, mx = _pool(page_size=1, page_count=2)
    cache = pool.new_cache(capacity=2)
    mx.fail_concatenate_once = True
    mx.fail_eval_once = True
    keys, values = _rows(0, 2)

    with pytest.raises(RuntimeError, match="injected device concatenate failure") as raised:
        cache.update_and_fetch(keys, values)

    assert any("quarantined" in note for note in (raised.value.__notes__ or ()))
    assert cache.offset == 0
    assert cache.page_authorities == ()
    telemetry = pool.telemetry()
    assert telemetry.free_pages == 0
    assert telemetry.quarantined_pages == 2
    assert telemetry.quarantined_bytes == telemetry.physical_bytes
    assert telemetry.leaked_pages == 0
    assert telemetry.reconciled

    cache.release()
    with pytest.raises(MlxPagedKVError, match="quarantined"):
        pool.close()
    assert pool.reconcile_quarantined_pages() == 2
    telemetry = pool.telemetry()
    assert telemetry.free_pages == 2
    assert telemetry.quarantined_pages == 0
    assert telemetry.quarantine_reconciliations == 1
    assert telemetry.reconciled
    pool.close()


def test_factory_release_evaluates_every_layer_before_mutating_any_ledger() -> None:
    mx = _FakeMx()
    pools = tuple(_pool(mx=mx, page_count=2)[0] for _index in range(2))
    factory = MlxPagedKVCacheFactory(pools)
    caches = factory(2)
    keys, values = _rows(0, 1)
    for cache in caches:
        cache.update_and_fetch(keys, values)

    mx.fail_eval_once = True
    with pytest.raises(RuntimeError, match="injected device evaluation failure"):
        factory.release_caches(caches)

    assert all(cache.offset == 1 for cache in caches)
    assert all(pool.telemetry().active_caches == 1 for pool in pools)
    assert all(pool.telemetry().reserved_pages == 1 for pool in pools)
    assert all(pool.telemetry().reconciled for pool in pools)

    factory.release_caches(caches)
    assert factory.telemetry().active_caches == 0
    assert factory.telemetry().reserved_bytes == 0
    factory.close()


def test_generation_authority_rejects_aba_foreign_pool_and_double_release() -> None:
    pool, _mx = _pool(page_count=1)
    first = pool.new_cache(capacity=2)
    keys, values = _rows(0, 1)
    first.update_and_fetch(keys, values)
    stale = first.page_authorities[0]
    first.release()

    second = pool.new_cache(capacity=2)
    second.update_and_fetch(keys, values)
    current = second.page_authorities[0]
    assert current.slot == stale.slot
    assert current.generation > stale.generation
    with pytest.raises(MlxPagedKVError, match="ABA-reused"):
        pool.page_refcount(stale)
    with pytest.raises(MlxPagedKVError, match="released"):
        first.release()

    foreign_pool, _foreign_mx = _pool(page_count=2)
    foreign = foreign_pool.new_cache(capacity=2)
    with pytest.raises(MlxPagedKVError, match="pool identity"):
        foreign.copy_committed_prefix_from(second, 1)
    with pytest.raises(MlxPagedKVError, match="live, quarantined, or leaked"):
        pool.close()

    foreign.release()
    foreign_pool.close()
    second.release()
    assert pool.telemetry().reconciled
    pool.close()


def test_geometry_and_logical_capacity_fail_before_page_mutation() -> None:
    pool, _mx = _pool(page_count=2)
    with pytest.raises(MlxPagedKVCapacityError, match="exceeds its physical pool"):
        pool.new_cache(capacity=5)
    cache = pool.new_cache(capacity=4)
    malformed = np.ones((1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="rank-four"):
        cache.update_and_fetch(malformed, malformed)
    wrong_width = np.ones((1, 1, 1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="pool geometry"):
        cache.update_and_fetch(wrong_width, wrong_width)
    keys, values = _rows(0, 1)
    with pytest.raises(ValueError, match="pool geometry"):
        cache.update_and_fetch(keys.astype(np.float64), values.astype(np.float64))
    assert cache.offset == 0
    assert cache.page_authorities == ()
    assert pool.telemetry().reserved_pages == 0
    assert pool.telemetry().reconciled
    cache.release()
    pool.close()


class _Engine:
    backend = "mlx-component"
    arch = "qwen2"
    context_size = 8
    semantic_token_count = 10
    numerical_contract = "mlx-paged-test-v1"

    def close(self) -> None:
        pass


def _runtime_with_paged_factory() -> tuple[MlxNativeRuntime, MlxPagedKVCacheFactory]:
    mx = _FakeMx()
    pools = tuple(
        MlxKVPagePool(
            mx=mx,
            create_attention_mask=lambda *args, offset, **kwargs: (args, offset, kwargs),
            page_size=2,
            page_count=4,
            kv_heads=1,
            key_head_dim=1,
            value_head_dim=1,
            key_dtype=np.float32,
            value_dtype=np.float32,
        )
        for _index in range(2)
    )
    factory = MlxPagedKVCacheFactory(pools)
    state_placement = StatePlacement(
        state_abi="paged-bf16-kv-v1",
        dtype="float32",
        memory_domain=MemoryDomain.UNIFIED,
        bytes_per_token=16,
        reserved_bytes=16 * 8,
        max_batch_size=1,
        max_context_tokens=8,
    )
    component = ComponentPlacement(
        allocation_id="allocation.body",
        component_ids=("body",),
        roles=("body",),
        codec_id="mlx-affine-q8",
        layout_id="mlx-linear",
        memory_domain=MemoryDomain.UNIFIED,
        residency=Residency.RESIDENT,
        physical_bytes=100,
    )
    placement = PlacementPlan(
        model_fingerprint="a" * 64,
        capability_fingerprint="b" * 64,
        device_fingerprint="c" * 64,
        workload_fingerprint="d" * 64,
        backend_id="mlx-component",
        device_id="metal:0",
        components=(component,),
        state=state_placement,
        workspace_bytes=0,
        headroom_bytes=0,
        model_resident_bytes=100,
        total_reserved_bytes=228,
        memory_budget_bytes=1_000,
        fully_resident=True,
        fallback_policy=FallbackPolicy.DENY,
        performance_claim_valid=True,
    )
    route = RuntimeRoute(
        runtime_id=f"runtime.{uuid4().hex}",
        model_fingerprint="a" * 64,
        capability_fingerprint="b" * 64,
        placement_fingerprint=placement.fingerprint,
        backend_id="mlx-component",
        device_id="metal:0",
        promotion_status=PromotionStatus.EXPERIMENTAL,
    )

    def executor(ids: tuple[int, ...], caches: Any) -> int:
        for layer, cache in enumerate(caches):
            shape = (1, 1, len(ids), 1)
            keys = np.full(shape, layer + ids[-1], dtype=np.float32)
            values = np.full(shape, layer + ids[-1] + 10, dtype=np.float32)
            cache.update_and_fetch(keys, values)
        return (ids[-1] + 1) % 10

    return (
        MlxNativeRuntime(
            _Engine(),
            route=route,
            placement=placement,
            semantic_token_count=10,
            state_abi="paged-bf16-kv-v1",
            cache_factory=factory,
            executor=executor,
        ),
        factory,
    )


def test_runtime_factory_hook_reports_physical_pool_and_actual_fork_copy_bytes() -> None:
    runtime, factory = _runtime_with_paged_factory()
    source = runtime.allocate_state(owner_id="source", batch_size=1, capacity=8)
    step = runtime.prefill(
        PrefillWork(
            request_ids=("source",),
            token_rows=((1, 2, 3, 4, 5),),
            state=source,
            parent=source.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    runtime.commit(step, (5,))

    fork = runtime.fork_state(
        source,
        parent=source.observe(),
        owner_id="fork",
        capacity=8,
    )

    # Four complete prefix tokens share two pages per layer; only one tail token is copied.
    assert fork.state_bytes_copied == 16
    snapshot = factory.telemetry()
    assert snapshot.full_page_prefix_shares == 4
    assert snapshot.partial_tail_copy_bytes == 16
    telemetry = runtime.telemetry()
    extras = dict(telemetry.extra_counters)
    assert telemetry.kv_resident_bytes == snapshot.physical_bytes
    assert extras["state_fork_bytes"] == 16
    assert extras["paged_kv_reconciled"] == 1
    assert extras["paged_kv_shared_bytes"] > 0
    assert extras["paged_kv_concatenate_materializations"] > 0

    runtime.release_state(fork.state)
    runtime.release_state(source)
    assert factory.telemetry().active_caches == 0
    assert factory.telemetry().reserved_bytes == 0
    assert factory.telemetry().reconciled
    runtime.close()
    factory.close()


def test_real_mlx_paged_cache_returns_mlx_compatible_dense_views() -> None:
    mx = pytest.importorskip("mlx.core")
    pool = MlxKVPagePool(
        mx=mx,
        create_attention_mask=lambda *args, offset, **kwargs: (args, offset, kwargs),
        page_size=2,
        page_count=3,
        kv_heads=1,
        key_head_dim=4,
        value_head_dim=4,
        key_dtype=mx.bfloat16,
        value_dtype=mx.bfloat16,
    )
    cache = pool.new_cache(capacity=5)
    keys = mx.arange(12).reshape(1, 1, 3, 4).astype(mx.bfloat16)
    values = (keys + 1).astype(mx.bfloat16)

    returned_keys, returned_values = cache.update_and_fetch(keys, values)
    mx.eval(returned_keys, returned_values)

    assert returned_keys.shape == returned_values.shape == (1, 1, 3, 4)
    assert bool(mx.array_equal(returned_keys, keys).item())
    assert bool(mx.array_equal(returned_values, values).item())
    host_values = np.ones((1, 1, 1, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="device array domain"):
        cache.update_and_fetch(host_values, host_values)
    cache.release()
    pool.close()


def test_real_mlx_partial_fork_isolated_tail_stays_on_device() -> None:
    mx = pytest.importorskip("mlx.core")
    pool = MlxKVPagePool(
        mx=mx,
        create_attention_mask=lambda *args, offset, **kwargs: (args, offset, kwargs),
        page_size=2,
        page_count=6,
        kv_heads=1,
        key_head_dim=4,
        value_head_dim=4,
        key_dtype=mx.bfloat16,
        value_dtype=mx.bfloat16,
    )
    source = pool.new_cache(capacity=6)
    branch = pool.new_cache(capacity=6)
    source_keys = mx.arange(20).reshape(1, 1, 5, 4).astype(mx.bfloat16)
    source_values = (source_keys + 100).astype(mx.bfloat16)
    source.update_and_fetch(source_keys, source_values)

    branch.copy_committed_prefix_from(source, 5)
    assert branch.page_authorities[:2] == source.page_authorities[:2]
    assert branch.page_authorities[2] != source.page_authorities[2]
    append_keys = mx.full((1, 1, 1, 4), 77, dtype=mx.bfloat16)
    append_values = mx.full((1, 1, 1, 4), 88, dtype=mx.bfloat16)
    branch.update_and_fetch(append_keys, append_values)
    mx.eval(source.keys, source.values, branch.keys, branch.values)

    assert bool(mx.array_equal(source.keys, source_keys).item())
    assert bool(mx.array_equal(source.values, source_values).item())
    assert bool(mx.array_equal(branch.keys[..., :5, :], source_keys).item())
    assert bool(mx.array_equal(branch.values[..., :5, :], source_values).item())
    assert bool(mx.array_equal(branch.keys[..., 5:, :], append_keys).item())
    assert bool(mx.array_equal(branch.values[..., 5:, :], append_values).item())
    branch.release()
    source.release()
    assert pool.telemetry().reconciled
    pool.close()


def test_tiny_qwen2_metal_forward_matches_mlx_kv_cache() -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache, create_attention_mask
    from mlx_lm.models.qwen2 import Model, ModelArgs

    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=16,
            num_key_value_heads=1,
        )
    )
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    pool = MlxKVPagePool(
        mx=mx,
        create_attention_mask=create_attention_mask,
        page_size=2,
        page_count=4,
        kv_heads=1,
        key_head_dim=4,
        value_head_dim=4,
        key_dtype=mx.bfloat16,
        value_dtype=mx.bfloat16,
    )
    paged = pool.new_cache(capacity=6)
    reference = KVCache()
    prompt = mx.array([[1, 2, 3]])

    expected_prefill = model(prompt, cache=(reference,))
    actual_prefill = model(prompt, cache=(paged,))
    mx.eval(expected_prefill, actual_prefill)
    assert bool(mx.allclose(actual_prefill, expected_prefill, rtol=0, atol=0).item())

    token = mx.array([[4]])
    expected_decode = model(token, cache=(reference,))
    actual_decode = model(token, cache=(paged,))
    mx.eval(expected_decode, actual_decode)
    assert bool(mx.allclose(actual_decode, expected_decode, rtol=0, atol=0).item())
    assert paged.offset == reference.offset == 4
    assert paged.keys.dtype == paged.values.dtype == mx.bfloat16
    paged.release()
    pool.close()
