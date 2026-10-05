from __future__ import annotations

import threading
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from mrun.runtime import (
    MLX_CHUNKED_PREFILL_NUMERICAL_CONTRACT,
    BackendCapabilities,
    BlobIdentity,
    CodecCapability,
    CompiledComponent,
    CompiledModelIdentity,
    ComponentPlacement,
    DecodeWork,
    DeviceDescriptor,
    FallbackPolicy,
    FixedMlxBatchKVCache,
    FixedMlxKVCache,
    ForkableModelRuntime,
    MemoryDomain,
    MlxCompatibleBatchLane,
    MlxNativeRuntime,
    MlxNativeRuntimeError,
    ModelRuntime,
    OutputMode,
    OutputRequest,
    PlacementPlan,
    PrefillWork,
    PromotionStatus,
    Residency,
    RuntimeRoute,
    SamplingPolicy,
    SamplingRequest,
    StatePlacement,
    WorkloadSpec,
    plan_resident_placement,
)
from mrun.runtime.inference import (
    GenerationCancelled,
    GenerationRequest,
    NativeGenerationService,
)


class _FakeMlxCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.offset = 0
        self.marker = object()
        self.nbytes = capacity * 8
        self.data: list[tuple[str, int] | None] = [None] * capacity

    def append(self, count: int) -> None:
        if self.offset + count > self.capacity:
            raise OverflowError
        for index in range(self.offset, self.offset + count):
            self.data[index] = ("kv", index)
        self.offset += count

    def trim(self, count: int) -> int:
        trimmed = min(count, self.offset)
        self.offset -= trimmed
        return trimmed

    def reset(self, offset: int) -> None:
        self.offset = offset

    def copy_committed_prefix_from(self, source: _FakeMlxCache, length: int) -> None:
        if self.offset:
            raise RuntimeError("target not empty")
        if length > source.offset or length > self.capacity:
            raise OverflowError
        self.data[:length] = source.data[:length]
        self.offset = length

    def storage_signature(self) -> tuple[int, int]:
        return (id(self.marker), self.capacity)

    def validate_row_copy_from_batch(
        self,
        source: _FakeBatchCache,
        *,
        row: int,
        start: int,
        count: int,
    ) -> None:
        if not isinstance(source, _FakeBatchCache):
            raise TypeError("wrong fake batch cache")
        if self.offset != start or start + count > self.capacity:
            raise RuntimeError("invalid fake batch commit")
        if row < 0 or row >= source.batch_size or start + count > source.offset:
            raise RuntimeError("invalid fake batch source")

    def write_row_from_batch(
        self,
        source: _FakeBatchCache,
        *,
        row: int,
        start: int,
        count: int,
    ) -> None:
        self.validate_row_copy_from_batch(source, row=row, start=start, count=count)
        self.data[start : start + count] = source.data[row][start : start + count]


class _FakeBatchCache:
    def __init__(self, capacity: int, batch_size: int) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self.offset = 0
        self.marker = object()
        self.nbytes = capacity * batch_size * 8
        self.data: list[list[tuple[str, int, int] | None]] = [
            [None] * capacity for _ in range(batch_size)
        ]

    def copy_committed_row_from(
        self,
        source: _FakeMlxCache,
        *,
        row: int,
        length: int,
    ) -> None:
        if self.offset or length > source.offset:
            raise RuntimeError("invalid fake prefix copy")
        self.data[row][:length] = [
            None if value is None else ("prefix", row, index)
            for index, value in enumerate(source.data[:length])
        ]

    def seal_prefix(self, length: int) -> None:
        if self.offset:
            raise RuntimeError("fake prefix already sealed")
        self.offset = length

    def append(self, count: int) -> None:
        if self.offset + count > self.capacity:
            raise OverflowError
        for row in range(self.batch_size):
            for index in range(self.offset, self.offset + count):
                self.data[row][index] = ("batch", row, index)
        self.offset += count

    def storage_signature(self) -> tuple[int, int, int]:
        return (id(self.marker), self.capacity, self.batch_size)


class _DriftingFakeMlxCache(_FakeMlxCache):
    def copy_committed_prefix_from(self, source: _FakeMlxCache, length: int) -> None:
        super().copy_committed_prefix_from(source, length)
        self.marker = object()


class _FakeMx:
    def __init__(self) -> None:
        self.eval_calls = 0

    @staticmethod
    def zeros(shape: tuple[int, ...], *, dtype: Any) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def eval(self, *_values: Any) -> None:
        self.eval_calls += 1


class _FakeMlxEngine:
    backend = "mlx-component"
    arch = "qwen2"
    context_size = 8
    semantic_token_count = 10
    numerical_contract = "mlx-component-test-v1"

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _cache_factory(capacity: int) -> tuple[_FakeMlxCache, _FakeMlxCache]:
    return (_FakeMlxCache(capacity), _FakeMlxCache(capacity))


def _executor(ids: tuple[int, ...], caches: Any) -> int:
    for cache in caches:
        cache.append(len(ids))
    return (ids[-1] + 1) % 10


def _direct_runtime(
    *,
    owns_engine: bool = False,
    cache_factory: Any = _cache_factory,
    owns_cache_factory: bool = False,
    sampling_executor: Any = None,
    prefill_chunk_size: int | None = None,
    prefill_chunk_executor: Any = None,
    route_effective_numerical_contract: str | None = None,
    route_execution_shape_fingerprint: str | None = None,
    route_promotion_status: PromotionStatus = PromotionStatus.CANDIDATE,
) -> tuple[MlxNativeRuntime, _FakeMlxEngine]:
    state = StatePlacement(
        state_abi="gqa-kv-v1",
        dtype="bfloat16",
        memory_domain=MemoryDomain.UNIFIED,
        bytes_per_token=16,
        reserved_bytes=16 * 1 * 8,
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
        state=state,
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
        promotion_status=route_promotion_status,
        effective_numerical_contract=route_effective_numerical_contract,
        execution_shape_fingerprint=route_execution_shape_fingerprint,
    )
    engine = _FakeMlxEngine()
    runtime = MlxNativeRuntime(
        engine,
        route=route,
        placement=placement,
        semantic_token_count=10,
        state_abi="gqa-kv-v1",
        cache_factory=cache_factory,
        executor=_executor,
        sampling_executor=sampling_executor,
        prefill_chunk_size=prefill_chunk_size,
        prefill_chunk_executor=prefill_chunk_executor,
        owns_cache_factory=owns_cache_factory,
        owns_engine=owns_engine,
    )
    return runtime, engine


def test_mlx_unchunked_construction_preserves_base_route_identity() -> None:
    runtime, engine = _direct_runtime()

    assert runtime.prefill_execution_shape.chunk_size is None
    assert runtime.prefill_execution_shape.base_numerical_contract == engine.numerical_contract
    assert runtime.prefill_execution_shape.numerical_contract == engine.numerical_contract
    assert runtime.prefill_execution_shape.promotion_status is PromotionStatus.CANDIDATE
    assert runtime.route.promotion_status is PromotionStatus.CANDIDATE
    assert runtime.route.effective_numerical_contract is None
    assert runtime.route.execution_shape_fingerprint is None


def test_mlx_runtime_closes_an_explicitly_owned_cache_factory_once() -> None:
    class OwnedFactory:
        def __init__(self) -> None:
            self.close_calls = 0

        def __call__(self, capacity: int) -> tuple[_FakeMlxCache, _FakeMlxCache]:
            return _cache_factory(capacity)

        def close(self) -> None:
            self.close_calls += 1

    factory = OwnedFactory()
    runtime, engine = _direct_runtime(
        cache_factory=factory,
        owns_cache_factory=True,
        owns_engine=True,
    )
    runtime.close()
    runtime.close()

    assert factory.close_calls == 1
    assert engine.close_calls == 1


def test_mlx_runtime_rejects_owned_cache_factory_without_close() -> None:
    with pytest.raises(ValueError, match=r"must expose close\(\)"):
        _direct_runtime(owns_cache_factory=True)


def test_mlx_chunked_construction_rejects_a_mismatched_prebound_shape() -> None:
    with pytest.raises(ValueError, match="does not bind its effective execution shape"):
        _direct_runtime(
            prefill_chunk_size=2,
            route_effective_numerical_contract=MLX_CHUNKED_PREFILL_NUMERICAL_CONTRACT,
            route_execution_shape_fingerprint="f" * 64,
            route_promotion_status=PromotionStatus.EXPERIMENTAL,
        )


def test_fixed_mlx_cache_copies_prefix_inside_native_arrays_only() -> None:
    mx = _FakeMx()
    kwargs = {
        "mx": mx,
        "create_attention_mask": lambda *args, **kwargs: (args, kwargs),
        "kv_heads": 1,
        "key_head_dim": 2,
        "value_head_dim": 3,
        "key_dtype": np.float32,
        "value_dtype": np.float32,
    }
    source = FixedMlxKVCache(capacity=4, **kwargs)
    target = FixedMlxKVCache(capacity=3, **kwargs)
    keys = np.arange(6, dtype=np.float32).reshape(1, 1, 3, 2)
    values = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    source.update_and_fetch(keys, values)

    target.copy_committed_prefix_from(source, 2)

    assert target.offset == 2
    assert np.array_equal(target.keys[..., :2, :], source.keys[..., :2, :])
    assert np.array_equal(target.values[..., :2, :], source.values[..., :2, :])
    assert np.count_nonzero(target.keys[..., 2:, :]) == 0
    assert np.count_nonzero(target.values[..., 2:, :]) == 0
    assert mx.eval_calls == 1
    with pytest.raises(MlxNativeRuntimeError, match="target must be empty"):
        target.copy_committed_prefix_from(source, 1)
    incompatible = FixedMlxKVCache(
        capacity=3,
        **(kwargs | {"value_head_dim": 2}),
    )
    with pytest.raises(MlxNativeRuntimeError, match="layout boundary"):
        incompatible.copy_committed_prefix_from(source, 2)


def test_fixed_mlx_batch_cache_copies_prefix_and_installs_only_selected_row_suffix() -> None:
    mx = _FakeMx()
    kwargs = {
        "mx": mx,
        "create_attention_mask": lambda *args, **kwargs: (args, kwargs),
        "kv_heads": 1,
        "key_head_dim": 2,
        "value_head_dim": 2,
        "key_dtype": np.float32,
        "value_dtype": np.float32,
    }
    sources = (FixedMlxKVCache(capacity=4, **kwargs), FixedMlxKVCache(capacity=4, **kwargs))
    for row, source in enumerate(sources):
        source.update_and_fetch(
            np.full((1, 1, 2, 2), row + 1, dtype=np.float32),
            np.full((1, 1, 2, 2), row + 11, dtype=np.float32),
        )
    batch = FixedMlxBatchKVCache(capacity=3, batch_size=2, **kwargs)
    for row, source in enumerate(sources):
        batch.copy_committed_row_from(source, row=row, length=2)
    batch.seal_prefix(2)
    batch.update_and_fetch(
        np.asarray([[[[21.0, 22.0]]], [[[31.0, 32.0]]]], dtype=np.float32),
        np.asarray([[[[41.0, 42.0]]], [[[51.0, 52.0]]]], dtype=np.float32),
    )

    target = sources[0]
    target.validate_row_copy_from_batch(batch, row=1, start=2, count=1)
    target.write_row_from_batch(batch, row=1, start=2, count=1)
    assert target.offset == 2
    assert np.array_equal(target.keys[..., 2, :], np.asarray([[[31.0, 32.0]]]))
    assert sources[1].offset == 2
    assert np.all(sources[1].keys[..., :2, :] == 2)


def test_mlx_native_runtime_uses_fixed_suffix_as_provisional_scratch() -> None:
    runtime, _engine = _direct_runtime()
    assert isinstance(runtime, ModelRuntime)
    state = runtime.allocate_state(owner_id="chat.request", batch_size=1, capacity=8)
    parent = state.observe()
    step = runtime.prefill(
        PrefillWork(
            request_ids=("request",),
            token_rows=((1, 2, 3),),
            state=state,
            parent=parent,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )

    assert step.output.token_ids == (4,)
    assert state.observe() == parent
    assert {cache.offset for cache in state._caches} == {3}
    receipt = runtime.commit(step, (2,))
    assert receipt.after.lengths == (2,)
    assert receipt.after.epoch == 1
    assert receipt.state_bytes_written == 32
    assert {cache.offset for cache in state._caches} == {2}

    decode_parent = state.observe()
    decode = runtime.decode(
        DecodeWork(
            request_ids=("request",),
            token_rows=((5,),),
            state=state,
            parent=decode_parent,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    assert decode.output.token_ids == (6,)
    runtime.abandon(decode)
    assert state.observe() == decode_parent
    assert {cache.offset for cache in state._caches} == {2}

    telemetry = runtime.telemetry()
    assert telemetry.prefill_calls == 1
    assert telemetry.prefill_tokens == 3
    assert telemetry.decode_calls == 1
    assert telemetry.decode_tokens == 1
    assert telemetry.provisional_steps == 2
    assert telemetry.commits == 1
    assert telemetry.abandons == 1
    assert telemetry.committed_tokens == 2
    assert telemetry.device_to_host_bytes == 16
    assert telemetry.kv_resident_bytes == 128


def test_mlx_native_runtime_preserves_sample_mode_and_dynamic_policy() -> None:
    seen: list[SamplingRequest] = []

    def sample(ids: tuple[int, ...], caches: Any, request: SamplingRequest) -> int:
        seen.append(request)
        for cache in caches:
            cache.append(len(ids))
        return 7

    runtime, _engine = _direct_runtime(sampling_executor=sample)
    state = runtime.allocate_state(owner_id="sample", batch_size=1, capacity=8)
    sampling = SamplingRequest(
        SamplingPolicy(seed=12, temperature=0.8, top_p=0.9, top_k=4),
        ((1, 2),),
        3,
    )
    step = runtime.prefill(
        PrefillWork(
            request_ids=("sample",),
            token_rows=((1, 2),),
            state=state,
            parent=state.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_SAMPLE, sampling=(sampling,)),
        )
    )

    assert step.output.mode is OutputMode.NEXT_TOKEN_SAMPLE
    assert step.output.token_ids == (7,)
    assert seen == [sampling]
    assert state.observe().lengths == (0,)
    runtime.commit(step, (2,))
    assert state.observe().lengths == (2,)


def test_mlx_native_runtime_forks_only_the_exact_committed_b1_prefix() -> None:
    runtime, _engine = _direct_runtime()
    assert isinstance(runtime, ForkableModelRuntime)
    source = runtime.allocate_state(owner_id="source", batch_size=1, capacity=8)
    step = runtime.prefill(
        PrefillWork(
            request_ids=("source",),
            token_rows=((1, 2, 3),),
            state=source,
            parent=source.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    runtime.commit(step, (2,))
    parent = source.observe()
    assert all(cache.data[2] is not None for cache in source._caches)

    result = runtime.fork_state(
        source,
        parent=parent,
        owner_id="session-copy",
        capacity=4,
    )

    assert result.source == parent
    assert result.state is not source
    assert result.state.owner_id == "session-copy"
    assert result.forked == result.state.observe()
    assert result.forked.lengths == (2,)
    assert result.forked.capacity == 4
    assert result.forked.generation > parent.generation
    assert result.state_bytes_copied == 32
    for source_cache, forked_cache in zip(source._caches, result.state._caches, strict=True):
        assert forked_cache.data[:2] == source_cache.data[:2]
        assert forked_cache.data[2:] == [None, None]
        assert forked_cache.marker is not source_cache.marker
    assert source.observe() == parent

    fork_step = runtime.decode(
        DecodeWork(
            request_ids=("fork",),
            token_rows=((5,),),
            state=result.state,
            parent=result.forked,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    runtime.commit(fork_step, (1,))
    assert result.state.observe().lengths == (3,)
    assert source.observe() == parent
    counters = dict(runtime.telemetry().extra_counters)
    assert counters["state_forks"] == 1
    assert counters["state_fork_tokens"] == 2
    assert counters["state_fork_bytes"] == 32


def test_mlx_native_runtime_fork_rejects_pending_stale_capacity_foreign_and_released() -> None:
    runtime, _engine = _direct_runtime()
    source = runtime.allocate_state(owner_id="source", batch_size=1, capacity=4)
    empty = source.observe()
    pending = runtime.prefill(
        PrefillWork(
            request_ids=("source",),
            token_rows=((1, 2),),
            state=source,
            parent=empty,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    with pytest.raises(MlxNativeRuntimeError, match="pending"):
        runtime.fork_state(source, parent=empty, owner_id="copy", capacity=4)
    runtime.commit(pending, (2,))
    committed = source.observe()

    with pytest.raises(MlxNativeRuntimeError, match="stale"):
        runtime.fork_state(source, parent=empty, owner_id="copy", capacity=4)
    with pytest.raises(OverflowError, match="capacity"):
        runtime.fork_state(source, parent=committed, owner_id="copy", capacity=1)
    with pytest.raises(ValueError, match="admitted placement"):
        runtime.fork_state(source, parent=committed, owner_id="copy", capacity=9)

    other, _ = _direct_runtime()
    with pytest.raises(MlxNativeRuntimeError, match="another runtime"):
        other.fork_state(source, parent=committed, owner_id="copy", capacity=4)
    runtime.release_state(source)
    with pytest.raises(MlxNativeRuntimeError, match="stale or released"):
        runtime.fork_state(source, parent=committed, owner_id="copy", capacity=4)


def test_mlx_native_runtime_fork_rejects_abi_storage_and_target_alias_drift() -> None:
    runtime, _engine = _direct_runtime()
    source = runtime.allocate_state(owner_id="source", batch_size=1, capacity=4)
    source._state_abi = "foreign-kv-v2"
    with pytest.raises(MlxNativeRuntimeError, match="ABI"):
        runtime.fork_state(
            source,
            parent=source.observe(),
            owner_id="copy",
            capacity=4,
        )

    runtime2, _engine = _direct_runtime()
    damaged = runtime2.allocate_state(owner_id="source", batch_size=1, capacity=4)
    damaged_parent = damaged.observe()
    damaged._caches[0].marker = object()
    with pytest.raises(MlxNativeRuntimeError, match="backing storage identity changed"):
        runtime2.fork_state(
            damaged,
            parent=damaged_parent,
            owner_id="copy",
            capacity=4,
        )

    runtime3, _engine = _direct_runtime()
    aliased = runtime3.allocate_state(owner_id="source", batch_size=1, capacity=4)
    runtime3._cache_factory = lambda _capacity: aliased._caches
    with pytest.raises(MlxNativeRuntimeError, match="aliases source"):
        runtime3.fork_state(
            aliased,
            parent=aliased.observe(),
            owner_id="copy",
            capacity=4,
        )

    runtime4, _engine = _direct_runtime()
    drifting = runtime4.allocate_state(owner_id="source", batch_size=1, capacity=4)
    runtime4._cache_factory = lambda capacity: (
        _DriftingFakeMlxCache(capacity),
        _DriftingFakeMlxCache(capacity),
    )
    with pytest.raises(MlxNativeRuntimeError, match="changed during copy"):
        runtime4.fork_state(
            drifting,
            parent=drifting.observe(),
            owner_id="copy",
            capacity=4,
        )


def test_mlx_native_runtime_rejects_pending_stale_foreign_and_double_terminal_work() -> None:
    runtime, _engine = _direct_runtime()
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=4)
    parent = state.observe()
    work = PrefillWork(
        request_ids=("request",),
        token_rows=((1, 2),),
        state=state,
        parent=parent,
        output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
    )
    step = runtime.prefill(work)
    with pytest.raises(MlxNativeRuntimeError, match="unconsumed"):
        runtime.prefill(work)
    with pytest.raises(MlxNativeRuntimeError, match="pending"):
        runtime.release_state(state)
    with pytest.raises(TypeError, match="strict integers"):
        runtime.commit(step, (True,))
    runtime.commit(step, (2,))
    with pytest.raises(MlxNativeRuntimeError, match="already been consumed"):
        runtime.commit(step, (2,))
    with pytest.raises(MlxNativeRuntimeError, match="already been consumed"):
        runtime.abandon(step)
    with pytest.raises(MlxNativeRuntimeError, match="stale"):
        runtime.prefill(work)

    other, _ = _direct_runtime()
    with pytest.raises(MlxNativeRuntimeError, match="another runtime"):
        other.release_state(state)
    runtime.release_state(state)
    with pytest.raises(MlxNativeRuntimeError, match="released"):
        state.observe()


def test_mlx_native_runtime_rolls_back_partial_native_failure_and_binds_storage() -> None:
    calls = 0

    def broken_executor(ids: tuple[int, ...], caches: Any) -> int:
        nonlocal calls
        calls += 1
        caches[0].append(len(ids))
        if calls == 1:
            raise RuntimeError("injected MLX failure")
        for cache in caches[1:]:
            cache.append(len(ids))
        return 2

    runtime, _engine = _direct_runtime()
    runtime._executor = broken_executor
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=4)
    parent = state.observe()
    work = PrefillWork(
        request_ids=("request",),
        token_rows=((1,),),
        state=state,
        parent=parent,
        output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
    )
    with pytest.raises(RuntimeError, match="injected MLX failure"):
        runtime.prefill(work)
    assert {cache.offset for cache in state._caches} == {0}
    step = runtime.prefill(work)
    runtime.abandon(step)

    state._caches[0].marker = object()
    with pytest.raises(MlxNativeRuntimeError, match="backing storage identity changed"):
        state.observe()


def test_mlx_native_runtime_close_is_fail_closed_with_pending_work() -> None:
    runtime, engine = _direct_runtime(owns_engine=True)
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=4)
    step = runtime.prefill(
        PrefillWork(
            request_ids=("request",),
            token_rows=((1,),),
            state=state,
            parent=state.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    with pytest.raises(MlxNativeRuntimeError, match="pending provisional"):
        runtime.close()
    runtime.abandon(step)
    runtime.close()
    runtime.close()
    assert engine.close_calls == 1


def _batch_cache_factory(capacity: int, batch_size: int) -> tuple[_FakeBatchCache, ...]:
    return (_FakeBatchCache(capacity, batch_size), _FakeBatchCache(capacity, batch_size))


def test_mlx_compatible_batch_keeps_b1_kv_unchanged_until_independent_terminal_ops() -> None:
    runtime, _engine = _direct_runtime()
    calls: list[tuple[tuple[int, ...], ...]] = []

    def execute(rows: tuple[tuple[int, ...], ...], caches: Any, _output: Any) -> tuple[int, ...]:
        calls.append(rows)
        for cache in caches:
            cache.append(len(rows[0]))
        return tuple((row[-1] + 1) % 10 for row in rows)

    lane = MlxCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_queue_delay_seconds=0,
        batch_cache_factory=_batch_cache_factory,
        batch_executor=execute,
    )
    states = tuple(
        runtime.allocate_state(owner_id=f"row-{row}", batch_size=1, capacity=8) for row in range(2)
    )
    parents = tuple(state.observe() for state in states)
    before = tuple(tuple(cache.data[:] for cache in state._caches) for state in states)
    steps = lane.execute(
        tuple(
            PrefillWork(
                request_ids=(f"request-{row}",),
                token_rows=((row + 1, row + 2),),
                state=state,
                parent=parent,
                output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
            )
            for row, (state, parent) in enumerate(zip(states, parents, strict=True))
        )
    )

    assert calls == [((1, 2), (2, 3))]
    assert tuple(step.output.token_ids for step in steps) == ((3,), (4,))
    assert tuple(state.observe() for state in states) == parents
    assert tuple(tuple(cache.data[:] for cache in state._caches) for state in states) == before
    assert all(cache.offset == 0 for state in states for cache in state._caches)

    first = runtime.commit(steps[0], (1,))
    runtime.abandon(steps[1])
    assert first.after.lengths == (1,)
    assert states[0].observe().lengths == (1,)
    assert states[1].observe() == parents[1]
    assert all(cache.offset == 1 for cache in states[0]._caches)
    assert all(cache.offset == 0 for cache in states[1]._caches)
    telemetry = lane.telemetry()
    assert telemetry.physical_forwards == 1
    assert telemetry.width_histogram == ((2, 1),)
    assert telemetry.committed_rows == telemetry.abandoned_rows == 1
    assert telemetry.prefix_bytes_copied == 0
    assert telemetry.suffix_bytes_committed == 16
    assert telemetry.active_scratches == 0


def test_mlx_compatible_batch_partitions_ragged_prefill_and_decode_to_exact_b1() -> None:
    runtime, _engine = _direct_runtime()
    batch_calls = 0

    def unexpected_batch(_rows: Any, _caches: Any, _output: Any) -> tuple[int, ...]:
        nonlocal batch_calls
        batch_calls += 1
        raise AssertionError("ragged singleton buckets must use exact B1")

    lane = MlxCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        batch_cache_factory=_batch_cache_factory,
        batch_executor=unexpected_batch,
    )
    states = tuple(
        runtime.allocate_state(owner_id=f"ragged-{row}", batch_size=1, capacity=8)
        for row in range(2)
    )
    prefill = lane.execute(
        (
            PrefillWork(
                request_ids=("short",),
                token_rows=((1,),),
                state=states[0],
                parent=states[0].observe(),
                output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
            ),
            PrefillWork(
                request_ids=("long",),
                token_rows=((2, 3),),
                state=states[1],
                parent=states[1].observe(),
                output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
            ),
        )
    )
    runtime.commit(prefill[0], (1,))
    runtime.commit(prefill[1], (2,))

    decode = lane.execute(
        tuple(
            DecodeWork(
                request_ids=(f"decode-{row}",),
                token_rows=((5 + row,),),
                state=state,
                parent=state.observe(),
                output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
            )
            for row, state in enumerate(states)
        )
    )
    runtime.abandon(decode[0])
    runtime.abandon(decode[1])

    assert batch_calls == 0
    telemetry = lane.telemetry()
    assert telemetry.dispatches == 2
    assert telemetry.physical_forwards == 4
    assert telemetry.singleton_bypasses == 4
    assert telemetry.width_histogram == ((1, 4),)


def test_mlx_compatible_batch_respects_scratch_bound_with_b1_fallback() -> None:
    runtime, _engine = _direct_runtime()

    def unexpected(_rows: Any, _caches: Any, _output: Any) -> tuple[int, ...]:
        raise AssertionError("scratch-limited cohort must bypass packed arithmetic")

    lane = MlxCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_scratch_bytes=32,
        batch_cache_factory=_batch_cache_factory,
        batch_executor=unexpected,
    )
    states = tuple(
        runtime.allocate_state(owner_id=f"bounded-{row}", batch_size=1, capacity=8)
        for row in range(2)
    )
    steps = lane.execute(
        tuple(
            PrefillWork(
                request_ids=(f"bounded-{row}",),
                token_rows=((1, 2),),
                state=state,
                parent=state.observe(),
                output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
            )
            for row, state in enumerate(states)
        )
    )
    runtime.commit(steps[0], (2,))
    runtime.abandon(steps[1])

    telemetry = lane.telemetry()
    assert lane.identity.max_scratch_bytes == 32
    assert telemetry.scratch_limited_bypasses == 1
    assert telemetry.singleton_bypasses == 2
    assert telemetry.width_histogram == ((1, 2),)


def test_mlx_compatible_batch_preserves_row_local_sampling_ownership() -> None:
    runtime, _engine = _direct_runtime()
    observed: list[tuple[SamplingRequest, ...]] = []

    def execute(
        rows: tuple[tuple[int, ...], ...], caches: Any, output: OutputRequest
    ) -> tuple[int, ...]:
        observed.append(output.sampling)
        for cache in caches:
            cache.append(len(rows[0]))
        return (7, 8)

    lane = MlxCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        batch_cache_factory=_batch_cache_factory,
        batch_executor=execute,
    )
    states = tuple(
        runtime.allocate_state(owner_id=f"sample-{row}", batch_size=1, capacity=8)
        for row in range(2)
    )
    requests = (
        SamplingRequest(SamplingPolicy(seed=11, temperature=0.7), ((1, 2),), 3),
        SamplingRequest(SamplingPolicy(seed=29, temperature=1.2), ((2, 1),), 5),
    )
    steps = lane.execute(
        tuple(
            PrefillWork(
                request_ids=(f"sample-{row}",),
                token_rows=((1, 2),),
                state=state,
                parent=state.observe(),
                output=OutputRequest(OutputMode.NEXT_TOKEN_SAMPLE, sampling=(request,)),
            )
            for row, (state, request) in enumerate(zip(states, requests, strict=True))
        )
    )

    assert observed == [requests]
    assert tuple(step.output.mode for step in steps) == (
        OutputMode.NEXT_TOKEN_SAMPLE,
        OutputMode.NEXT_TOKEN_SAMPLE,
    )
    assert tuple(step.output.token_ids for step in steps) == ((7,), (8,))
    runtime.commit(steps[0], (2,))
    runtime.commit(steps[1], (2,))


def test_mlx_compatible_batch_chunks_long_prefill_without_touching_b1_rows() -> None:
    runtime, _engine = _direct_runtime(prefill_chunk_size=2)
    calls: list[tuple[int, bool]] = []

    def execute(
        rows: tuple[tuple[int, ...], ...],
        caches: Any,
        output: OutputRequest | None,
    ) -> tuple[int, ...] | None:
        calls.append((len(rows[0]), output is not None))
        for cache in caches:
            cache.append(len(rows[0]))
        return (6, 7) if output is not None else None

    lane = MlxCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        batch_cache_factory=_batch_cache_factory,
        batch_executor=execute,
    )
    assert "chunked-prefill" in lane.identity.numerical_contract
    assert lane.identity.promotion_status is PromotionStatus.EXPERIMENTAL
    states = tuple(
        runtime.allocate_state(owner_id=f"long-{row}", batch_size=1, capacity=8) for row in range(2)
    )
    parents = tuple(state.observe() for state in states)
    steps = lane.execute(
        tuple(
            PrefillWork(
                request_ids=(f"long-{row}",),
                token_rows=((1, 2, 3, 4, 5),),
                state=state,
                parent=parent,
                output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
            )
            for row, (state, parent) in enumerate(zip(states, parents, strict=True))
        )
    )

    assert calls == [(2, False), (2, False), (1, True)]
    assert tuple(state.observe() for state in states) == parents
    assert all(cache.offset == 0 for state in states for cache in state._caches)
    runtime.commit(steps[0], (5,))
    runtime.abandon(steps[1])
    assert states[0].observe().lengths == (5,)
    assert states[1].observe() == parents[1]
    telemetry = lane.telemetry()
    assert telemetry.physical_forwards == 3
    assert telemetry.provisional_rows == 2
    assert telemetry.width_histogram == ((2, 3),)
    counters = dict(runtime.telemetry().extra_counters)
    assert counters["chunked_prefill_calls"] == 1
    assert counters["chunked_prefill_chunks"] == 3


def test_mlx_chunked_prefill_selects_only_final_chunk_and_commits_once() -> None:
    calls: list[tuple[tuple[int, ...], OutputRequest | None]] = []

    def chunks(ids: tuple[int, ...], caches: Any, output: OutputRequest | None) -> int | None:
        calls.append((ids, output))
        for cache in caches:
            cache.append(len(ids))
        return 9 if output is not None else None

    runtime, _engine = _direct_runtime(
        prefill_chunk_size=2,
        prefill_chunk_executor=chunks,
    )
    assert runtime.prefill_execution_shape.chunk_size == 2
    assert runtime.prefill_execution_shape.base_numerical_contract == "mlx-component-test-v1"
    assert (
        runtime.prefill_execution_shape.numerical_contract == MLX_CHUNKED_PREFILL_NUMERICAL_CONTRACT
    )
    assert runtime.prefill_execution_shape.promotion_status is PromotionStatus.EXPERIMENTAL
    assert runtime.route.promotion_status is PromotionStatus.EXPERIMENTAL
    assert runtime.route.effective_numerical_contract == MLX_CHUNKED_PREFILL_NUMERICAL_CONTRACT
    assert runtime.route.execution_shape_fingerprint == runtime.prefill_execution_shape.fingerprint
    state = runtime.allocate_state(owner_id="chunked", batch_size=1, capacity=8)
    parent = state.observe()
    step = runtime.prefill(
        PrefillWork(
            request_ids=("chunked",),
            token_rows=((1, 2, 3, 4, 5),),
            state=state,
            parent=parent,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )

    assert [ids for ids, _output in calls] == [(1, 2), (3, 4), (5,)]
    assert [output is None for _ids, output in calls] == [True, True, False]
    assert step.output.token_ids == (9,)
    assert state.observe() == parent
    assert all(cache.offset == 5 for cache in state._caches)
    receipt = runtime.commit(step, (5,))
    assert receipt.after.lengths == (5,)
    counters = dict(runtime.telemetry().extra_counters)
    assert counters["chunked_prefill_calls"] == 1
    assert counters["chunked_prefill_chunks"] == 3
    assert counters["chunked_prefill_failures"] == 0


def test_mlx_chunked_prefill_failure_rolls_every_layer_back_to_committed_offset() -> None:
    calls = 0

    def broken(ids: tuple[int, ...], caches: Any, _output: OutputRequest | None) -> int | None:
        nonlocal calls
        calls += 1
        for cache in caches:
            cache.append(len(ids))
        if calls == 2:
            raise RuntimeError("injected second chunk failure")
        return None

    runtime, _engine = _direct_runtime(
        prefill_chunk_size=2,
        prefill_chunk_executor=broken,
    )
    state = runtime.allocate_state(owner_id="chunk-failure", batch_size=1, capacity=8)
    parent = state.observe()
    work = PrefillWork(
        request_ids=("chunk-failure",),
        token_rows=((1, 2, 3, 4, 5),),
        state=state,
        parent=parent,
        output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
    )

    with pytest.raises(RuntimeError, match="second chunk failure"):
        runtime.prefill(work)
    assert state.observe() == parent
    assert all(cache.offset == 0 for cache in state._caches)
    assert state._pending_step_id is None
    counters = dict(runtime.telemetry().extra_counters)
    assert counters["chunked_prefill_calls"] == 0
    assert counters["chunked_prefill_chunks"] == 1
    assert counters["chunked_prefill_failures"] == 1


def test_generation_service_batch_lane_cancels_one_cow_row_and_commits_the_other() -> None:
    runtime, _engine = _direct_runtime()
    started = threading.Event()
    release = threading.Event()

    def execute(rows: tuple[tuple[int, ...], ...], caches: Any, _output: Any) -> tuple[int, ...]:
        for cache in caches:
            cache.append(len(rows[0]))
        started.set()
        assert release.wait(2)
        return (7, 8)

    lane = MlxCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_queue_delay_seconds=0.01,
        batch_cache_factory=_batch_cache_factory,
        batch_executor=execute,
    )
    service = NativeGenerationService(
        runtime,
        max_context_tokens=8,
        semantic_token_count=10,
        max_active_requests=2,
        supported_output_modes=(
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ),
        max_new_tokens=2,
        compatible_batch_lane=lane,
    )
    cancelled, completed = service.submit_many(
        (
            GenerationRequest("cancel-row", (1, 2), 1, stream=False),
            GenerationRequest("commit-row", (2, 3), 1, stream=False),
        )
    )
    assert started.wait(2)
    assert cancelled.cancel("test-row-cancel")
    release.set()

    with pytest.raises(GenerationCancelled, match="test-row-cancel"):
        cancelled.result(timeout=2)
    assert completed.result(timeout=2).token_ids == (8,)
    telemetry = service.telemetry()
    assert telemetry.reconciled
    assert telemetry.compatible_batch is not None
    assert telemetry.compatible_batch.dispatches == 1
    assert telemetry.compatible_batch.dispatched_rows == 2
    assert telemetry.compatible_batch.width_histogram == ((2, 1),)
    assert telemetry.compatible_batch.commits == 1
    assert telemetry.compatible_batch.abandons == 1
    assert telemetry.compatible_batch.queue_delay.observation_count == 2
    assert telemetry.compatible_batch.forward_latency.observation_count == 1
    lane_telemetry = lane.telemetry()
    assert lane_telemetry.committed_rows == lane_telemetry.abandoned_rows == 1
    assert lane_telemetry.active_scratches == 0
    assert service.shutdown(wait=True, timeout=2)


def test_generation_service_true_singleton_bypasses_opt_in_batch_lane() -> None:
    runtime, _engine = _direct_runtime()

    def unexpected(_rows: Any, _caches: Any, _output: Any) -> tuple[int, ...]:
        raise AssertionError("singleton traffic must bypass packed arithmetic")

    lane = MlxCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_queue_delay_seconds=0.05,
        batch_cache_factory=_batch_cache_factory,
        batch_executor=unexpected,
    )
    service = NativeGenerationService(
        runtime,
        max_context_tokens=8,
        semantic_token_count=10,
        max_active_requests=2,
        supported_output_modes=(
            OutputMode.NEXT_TOKEN_ARGMAX,
            OutputMode.NEXT_TOKEN_SAMPLE,
        ),
        max_new_tokens=2,
        compatible_batch_lane=lane,
    )
    result = service.submit(GenerationRequest("singleton", (1, 2), 1, stream=False)).result(
        timeout=2
    )

    assert result.token_ids == (3,)
    telemetry = service.telemetry().compatible_batch
    assert telemetry is not None
    assert telemetry.dispatches == 1
    assert telemetry.width_histogram == ((1, 1),)
    assert telemetry.singleton_bypasses == 1
    assert telemetry.queue_delay.observation_count == 1
    assert lane.telemetry().dispatches == 0
    assert service.shutdown(wait=True, timeout=2)


def test_mlx_native_runtime_bind_validates_compiler_capability_and_placement_chain() -> None:
    blob = BlobIdentity(blob_id="body.safetensors", sha256="1" * 64, byte_count=100)
    component = CompiledComponent(
        component_id="body",
        role="body",
        allocation_id="allocation.body",
        codec_id="mlx-affine-q8",
        layout_id="mlx-linear",
        physical_bytes=100,
        blobs=(blob,),
    )
    model = CompiledModelIdentity(
        model_name="toy",
        architecture="qwen2",
        source_revision_sha256="2" * 64,
        semantic_model_sha256="3" * 64,
        component_graph_sha256="4" * 64,
        vocab_manifest_sha256="5" * 64,
        compiler_abi="compiler-v1",
        components=(component,),
        operator_ids=("qwen2.dense-decoder",),
        state_abi="gqa-kv-v1",
        state_dtype="bfloat16",
        state_bytes_per_token=16,
        max_context_tokens=8,
        semantic_token_count=10,
    )
    capabilities = BackendCapabilities(
        backend_id="mlx-component",
        backend_abi="mlx-component-v1",
        implementation_version="test",
        fabric="apple-gpu",
        memory_domain=MemoryDomain.UNIFIED,
        architectures=("qwen2",),
        operator_ids=("qwen2.dense-decoder",),
        codecs=(
            CodecCapability(
                codec_id="mlx-affine-q8",
                layout_id="mlx-linear",
                component_roles=("body",),
            ),
        ),
        state_abis=("gqa-kv-v1",),
        output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,),
        numerical_contracts=("mlx-component-test-v1",),
        max_context_tokens=8,
        max_batch_size=1,
        max_verify_tokens=1,
        transactional_state=True,
        scratch_only_steps=True,
        independently_committable_rows=True,
        supports_ragged_batches=False,
        telemetry_counters=("committed_tokens",),
        promotion_status=PromotionStatus.CANDIDATE,
    )
    device = DeviceDescriptor(
        device_id="metal:0",
        fabric="apple-gpu",
        memory_domain=MemoryDomain.UNIFIED,
        total_bytes=10_000,
        available_bytes=10_000,
        machine_fingerprint="6" * 64,
    )
    workload = WorkloadSpec(
        max_batch_size=1,
        max_context_tokens=8,
        verify_tokens=1,
        output_mode=OutputMode.NEXT_TOKEN_ARGMAX,
        numerical_contract="mlx-component-test-v1",
        state_abi="gqa-kv-v1",
        required_component_roles=("body",),
        workspace_bytes=100,
        headroom_bytes=100,
    )
    placement = plan_resident_placement(model, workload, capabilities, device)
    engine = _FakeMlxEngine()
    runtime = MlxNativeRuntime.bind(
        engine,
        model=model,
        workload=workload,
        capabilities=capabilities,
        device=device,
        placement=placement,
        cache_factory=_cache_factory,
        executor=_executor,
    )

    assert runtime.route.model_fingerprint == model.fingerprint
    assert runtime.route.capability_fingerprint == capabilities.fingerprint
    assert runtime.route.placement_fingerprint == placement.fingerprint
    assert runtime.route.backend_id == "mlx-component"
