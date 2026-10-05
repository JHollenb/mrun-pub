from __future__ import annotations

import gc
import threading
import weakref
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
import torch

from mrun.engine.dense_qstore_cuda import (
    DenseForwardResult,
    DenseQStoreCudaEngine,
    DenseQStoreKVCache,
    KVDelta,
)
from mrun.runtime import (
    BackendCapabilities,
    BlobIdentity,
    CodecCapability,
    CompiledComponent,
    CompiledModelIdentity,
    ComponentPlacement,
    DecodeWork,
    DenseCudaCompatibleBatchLane,
    DenseCudaIndexedKVCache,
    DenseCudaKVSlotCache,
    DenseCudaNativeRuntime,
    DenseCudaRuntimeError,
    DeviceDescriptor,
    FallbackPolicy,
    ForkableModelRuntime,
    MemoryDomain,
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


class _FakeDenseEngine:
    backend = "dense-qstore-cuda"
    arch = "qwen2"
    max_seq_len = 8
    semantic_token_count = 10
    numerical_contract = "fake-qstore-exact-v1"

    def __init__(self) -> None:
        self.close_calls = 0
        self.forward_block_calls = 0
        self.forward_decode_slots_calls = 0
        self.forward_last_top1_calls = 0
        self._last_cache: DenseQStoreKVCache | None = None
        self.indexed_decode_calls: list[tuple[Any, ...]] = []

    def _result(
        self,
        input_ids: np.ndarray,
        cache: DenseQStoreKVCache,
    ) -> DenseForwardResult:
        ids = torch.as_tensor(input_ids, dtype=torch.long)
        values = ids.to(torch.float32)
        key = torch.stack((values, values + 0.25), dim=-1).unsqueeze(2)
        value = torch.stack((values + 0.5, values + 0.75), dim=-1).unsqueeze(2)
        delta = KVDelta(
            parent_epoch=cache.epoch,
            parent_lengths=tuple(int(item) for item in cache.lengths),
            cache_id=cache.cache_id,
            keys=(key,),
            values=(value,),
            token_count=int(ids.shape[1]),
        )
        return DenseForwardResult(
            top1=(ids + 1) % self.semantic_token_count,
            delta=delta,
            hidden=torch.zeros((*ids.shape, 2), dtype=torch.float32),
            kv_read_bytes=cache.committed_bytes,
            kv_delta_bytes=delta.byte_count,
            wall_s=0.001,
        )

    def forward_block(
        self,
        input_ids: np.ndarray,
        cache: DenseQStoreKVCache,
        *,
        return_logits: bool,
    ) -> DenseForwardResult:
        assert not return_logits
        self.forward_block_calls += 1
        self._last_cache = cache
        return self._result(input_ids, cache)

    def forward_last_top1(
        self,
        input_ids: np.ndarray,
        cache: DenseQStoreKVCache,
    ) -> DenseForwardResult:
        self.forward_last_top1_calls += 1
        self._last_cache = cache
        result = self._result(input_ids, cache)
        return DenseForwardResult(
            top1=result.top1[:, -1:],
            delta=result.delta,
            hidden=result.hidden,
            kv_read_bytes=result.kv_read_bytes,
            kv_delta_bytes=result.kv_delta_bytes,
            wall_s=result.wall_s,
        )

    def forward_decode_slots(
        self,
        input_ids: np.ndarray,
        indexed_cache: Any,
    ) -> DenseForwardResult:
        self.forward_decode_slots_calls += 1
        self._last_cache = indexed_cache
        self.indexed_decode_calls.append(
            (
                tuple(np.asarray(input_ids).shape),
                indexed_cache.batch_size,
                tuple(int(value) for value in indexed_cache.lengths),
                tuple(int(value) for value in indexed_cache.row_indices.detach().cpu().tolist()),
                tuple(int(tensor.data_ptr()) for tensor in indexed_cache.keys),
            )
        )
        return self._result(input_ids, indexed_cache)

    def commit_block(
        self,
        cache: DenseQStoreKVCache,
        result: DenseForwardResult,
        accepted_counts: tuple[int, ...],
    ) -> Any:
        return cache.commit(result.delta, accepted_counts)

    def forward_last_logits(
        self,
        input_ids: np.ndarray,
        cache: DenseQStoreKVCache,
    ) -> DenseForwardResult:
        result = self.forward_block(input_ids, cache, return_logits=False)
        batch = int(np.asarray(input_ids).shape[0])
        return DenseForwardResult(
            top1=result.top1[:, -1:],
            delta=result.delta,
            hidden=result.hidden,
            kv_read_bytes=result.kv_read_bytes,
            kv_delta_bytes=result.kv_delta_bytes,
            wall_s=result.wall_s,
            logits=torch.zeros((batch, 1, self.semantic_token_count)),
        )

    def close(self) -> None:
        self.close_calls += 1


def _cache_factory(batch_size: int, capacity: int) -> DenseQStoreKVCache:
    return DenseQStoreKVCache(
        num_layers=1,
        batch_size=batch_size,
        max_seq_len=capacity,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )


def _direct_runtime(
    *,
    owns_engine: bool = False,
    compact_head: bool = False,
    production_slot_seam: bool = False,
    decode_attention_mode: str = "established",
    body_fusion_mode: str = "established",
    supported_output_modes: tuple[OutputMode, ...] = (
        OutputMode.NEXT_TOKEN_ARGMAX,
        OutputMode.NEXT_TOKEN_SAMPLE,
    ),
    admitted_body_workspace_bytes: int = 0,
) -> tuple[DenseCudaNativeRuntime, _FakeDenseEngine]:
    backend_id = "cuda-source-int8-compact-head" if compact_head else "dense-qstore-cuda"
    state = StatePlacement(
        state_abi="gqa-kv-v1",
        dtype="float32",
        memory_domain=MemoryDomain.CUDA,
        bytes_per_token=16,
        reserved_bytes=16 * 2 * 8,
        max_batch_size=2,
        max_context_tokens=8,
    )
    component = ComponentPlacement(
        allocation_id="allocation.body",
        component_ids=("body",),
        roles=("body",),
        codec_id="qrow-int8",
        layout_id="row-major",
        memory_domain=MemoryDomain.CUDA,
        residency=Residency.RESIDENT,
        physical_bytes=100,
    )
    placement = PlacementPlan(
        model_fingerprint="a" * 64,
        capability_fingerprint="b" * 64,
        device_fingerprint="c" * 64,
        workload_fingerprint="d" * 64,
        backend_id=backend_id,
        device_id="cuda:0",
        components=(component,),
        state=state,
        workspace_bytes=admitted_body_workspace_bytes,
        headroom_bytes=0,
        model_resident_bytes=100,
        total_reserved_bytes=356 + admitted_body_workspace_bytes,
        memory_budget_bytes=10_000,
        fully_resident=True,
        fallback_policy=FallbackPolicy.DENY,
        performance_claim_valid=True,
    )
    route = RuntimeRoute(
        runtime_id=f"runtime.{uuid4().hex}",
        model_fingerprint="a" * 64,
        capability_fingerprint="b" * 64,
        placement_fingerprint=placement.fingerprint,
        backend_id=backend_id,
        device_id="cuda:0",
        promotion_status=PromotionStatus.CANDIDATE,
    )
    engine = _FakeDenseEngine()
    if compact_head:
        engine.backend = backend_id
        engine.head_execution_mode = "semantic-prefix-w8a16-top2-fp32-rerank"
        engine.head_execution_abi = "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"
        execution_suffix = ""
        if decode_attention_mode != "established":
            execution_suffix += f"+{decode_attention_mode}"
        if body_fusion_mode != "established":
            execution_suffix += f"+{body_fusion_mode}"
        engine.numerical_contract = (
            "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1+"
            "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"
            f"{execution_suffix}"
        )
        engine.target = type(
            "CompactTarget",
            (),
            {
                "experimental_reranked_head": True,
                "reranked_head_working_bytes_peak": 1234,
                "decode_attention_mode": decode_attention_mode,
                "decode_attention_tile": 64,
                "body_fusion_mode": body_fusion_mode,
            },
        )()
    elif production_slot_seam:
        engine.numerical_contract += "+segmented-flash-gqa-decode-v1"

        def production_forward_last_top1(
            _target: Any,
            input_ids: np.ndarray,
            indexed_cache: DenseCudaIndexedKVCache,
        ) -> DenseForwardResult:
            engine.forward_decode_slots_calls += 1
            engine.indexed_decode_calls.append(
                (
                    tuple(np.asarray(input_ids).shape),
                    indexed_cache.batch_size,
                    tuple(int(value) for value in indexed_cache.lengths),
                    tuple(
                        int(value)
                        for value in indexed_cache.row_indices.detach().cpu().tolist()
                    ),
                    tuple(int(tensor.data_ptr()) for tensor in indexed_cache.keys),
                )
            )
            return engine._result(input_ids, indexed_cache)

        engine.target = type(
            "ProductionSlotTarget",
            (),
            {
                "body_fusion_mode": "established",
                "cfg": {"num_hidden_layers": 1},
                "decode_attention_mode": "segmented-flash-gqa-decode-v1",
                "decode_attention_tile": 64,
                "experimental_reranked_head": False,
                "forward_last_top1": production_forward_last_top1,
            },
        )()
        engine.forward_decode_slots = DenseQStoreCudaEngine.forward_decode_slots.__get__(engine)
    return (
        DenseCudaNativeRuntime(
            engine,
            route=route,
            placement=placement,
            semantic_token_count=10,
            state_abi="gqa-kv-v1",
            supported_output_modes=supported_output_modes,
            admitted_body_workspace_bytes=admitted_body_workspace_bytes,
            cache_factory=_cache_factory,
            owns_engine=owns_engine,
        ),
        engine,
    )


def test_dense_native_runtime_is_scratch_only_until_explicit_commit() -> None:
    runtime, engine = _direct_runtime()
    assert isinstance(runtime, ModelRuntime)
    state = runtime.allocate_state(owner_id="chat.request", batch_size=2, capacity=8)
    parent = state.observe()
    step = runtime.prefill(
        PrefillWork(
            request_ids=("request.0", "request.1"),
            token_rows=((1, 2), (3, 4)),
            state=state,
            parent=parent,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )

    assert step.output.token_ids == (3, 5)
    assert state.observe() == parent
    receipt = runtime.commit(step, (2, 1))
    assert receipt.before == parent
    assert receipt.after.lengths == (2, 1)
    assert receipt.after.epoch == 1
    assert receipt.state_bytes_written == 48
    assert torch.equal(state._cache.keys[0][0, :2, 0, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(state._cache.keys[0][1, :1, 0, 0], torch.tensor([3.0]))

    decode_parent = state.observe()
    decode = runtime.decode(
        DecodeWork(
            request_ids=("request.0", "request.1"),
            token_rows=((5,), (6,)),
            state=state,
            parent=decode_parent,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    assert decode.output.token_ids == (6, 7)
    runtime.abandon(decode)
    assert state.observe() == decode_parent

    telemetry = runtime.telemetry()
    assert telemetry.prefill_calls == 1
    assert telemetry.prefill_tokens == 4
    assert telemetry.decode_calls == 1
    assert telemetry.decode_tokens == 2
    assert telemetry.provisional_steps == 2
    assert telemetry.commits == 1
    assert telemetry.abandons == 1
    assert telemetry.committed_tokens == 3
    assert telemetry.device_to_host_bytes == 32
    assert telemetry.kv_resident_bytes == state._cache.allocated_bytes
    assert engine.forward_block_calls == 2
    assert engine.forward_last_top1_calls == 0


def test_dense_native_runtime_release_drops_engine_cache_custody() -> None:
    runtime, engine = _direct_runtime()
    state = runtime.allocate_state(owner_id="release", batch_size=1, capacity=4)
    step = runtime.prefill(
        PrefillWork(
            request_ids=("release",),
            token_rows=((1, 2),),
            state=state,
            parent=state.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    result_reference = weakref.ref(step.authority._result)
    runtime.commit(step, (2,))
    cache = state._cache
    cache_reference = weakref.ref(cache)

    assert step.authority._result is None
    assert result_reference() is None
    assert engine._last_cache is cache
    assert runtime.telemetry().kv_resident_bytes == cache.allocated_bytes

    runtime.release_state(state)

    assert state._cache is None
    assert engine._last_cache is None
    telemetry = runtime.telemetry()
    assert telemetry.kv_resident_bytes == 0
    assert dict(telemetry.extra_counters)["live_states"] == 0
    with pytest.raises(DenseCudaRuntimeError, match="released"):
        state.observe()

    del cache
    gc.collect()
    assert cache_reference() is None


def test_dense_native_runtime_release_preserves_replacement_engine_cache() -> None:
    runtime, engine = _direct_runtime()
    first = runtime.allocate_state(owner_id="first", batch_size=1, capacity=4)
    first_step = runtime.prefill(
        PrefillWork(
            request_ids=("first",),
            token_rows=((1,),),
            state=first,
            parent=first.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    runtime.commit(first_step, (1,))
    first_cache = first._cache

    second = runtime.allocate_state(owner_id="second", batch_size=1, capacity=4)
    second_step = runtime.prefill(
        PrefillWork(
            request_ids=("second",),
            token_rows=((2,),),
            state=second,
            parent=second.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    runtime.commit(second_step, (1,))
    second_cache = second._cache
    assert engine._last_cache is second_cache

    runtime.release_state(first)

    assert first._cache is None
    assert engine._last_cache is second_cache
    telemetry = runtime.telemetry()
    assert telemetry.kv_resident_bytes == second_cache.allocated_bytes
    assert dict(telemetry.extra_counters)["live_states"] == 1
    assert first_cache is not second_cache

    runtime.release_state(second)
    assert second._cache is None
    assert engine._last_cache is None
    assert runtime.telemetry().kv_resident_bytes == 0


def test_dense_native_runtime_uses_last_row_execution_only_for_compact_head() -> None:
    runtime, engine = _direct_runtime(
        compact_head=True,
        supported_output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,),
        admitted_body_workspace_bytes=5678,
    )
    state = runtime.allocate_state(owner_id="compact", batch_size=1, capacity=8)

    step = runtime.prefill(
        PrefillWork(
            request_ids=("compact",),
            token_rows=((1, 2, 3),),
            state=state,
            parent=state.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )

    assert step.output.token_ids == (4,)
    assert engine.forward_last_top1_calls == 1
    assert engine.forward_block_calls == 0
    assert runtime.telemetry().workspace_peak_bytes == 1234 + 48
    assert dict(runtime.telemetry().extra_counters)["admitted_body_workspace_bytes"] == 5678

    runtime.abandon(step)
    engine.head_execution_abi = "tampered-after-open"
    with pytest.raises(DenseCudaRuntimeError, match="changed after runtime binding"):
        runtime.prefill(
            PrefillWork(
                request_ids=("compact",),
                token_rows=((4,),),
                state=state,
                parent=state.observe(),
                output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
            )
        )


def test_dense_native_runtime_seals_decode_and_body_fusion_identity() -> None:
    runtime, engine = _direct_runtime(
        compact_head=True,
        decode_attention_mode="segmented-flash-gqa-decode-v1",
        body_fusion_mode="residual-rms-swiglu-v1",
        supported_output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,),
    )
    assert engine.numerical_contract.endswith(
        "+segmented-flash-gqa-decode-v1+residual-rms-swiglu-v1"
    )
    state = runtime.allocate_state(owner_id="fused", batch_size=1, capacity=8)
    parent = state.observe()
    engine.target.body_fusion_mode = "established"

    with pytest.raises(DenseCudaRuntimeError, match="changed after runtime binding"):
        runtime.prefill(
            PrefillWork(
                request_ids=("fused",),
                token_rows=((1,),),
                state=state,
                parent=parent,
                output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
            )
        )


def test_dense_native_runtime_rejects_sampling_outside_route_capabilities() -> None:
    runtime, engine = _direct_runtime(supported_output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,))
    state = runtime.allocate_state(owner_id="greedy-only", batch_size=1, capacity=8)
    sampling = SamplingRequest(SamplingPolicy(seed=7, temperature=0.5), (), 0)

    with pytest.raises(NotImplementedError, match="does not admit 'next-token-sample'"):
        runtime.prefill(
            PrefillWork(
                request_ids=("greedy-only",),
                token_rows=((1, 2),),
                state=state,
                parent=state.observe(),
                output=OutputRequest(OutputMode.NEXT_TOKEN_SAMPLE, sampling=(sampling,)),
            )
        )

    assert engine.forward_block_calls == 0
    assert state.observe().lengths == (0,)


def test_dense_native_sampling_refuses_silent_cpu_logit_fallback() -> None:
    runtime, _engine = _direct_runtime()
    state = runtime.allocate_state(owner_id="sample", batch_size=1, capacity=8)
    parent = state.observe()
    sampling = SamplingRequest(
        SamplingPolicy(seed=11, temperature=0.7),
        ((1, 1),),
        0,
    )
    with pytest.raises(DenseCudaRuntimeError, match="must remain on the CUDA device"):
        runtime.prefill(
            PrefillWork(
                request_ids=("sample",),
                token_rows=((1, 2),),
                state=state,
                parent=parent,
                output=OutputRequest(OutputMode.NEXT_TOKEN_SAMPLE, sampling=(sampling,)),
            )
        )
    assert state.observe() == parent


def test_dense_native_runtime_row_preserving_fork_copies_only_committed_prefixes() -> None:
    runtime, _engine = _direct_runtime()
    assert isinstance(runtime, ForkableModelRuntime)
    source = runtime.allocate_state(owner_id="source", batch_size=2, capacity=8)
    step = runtime.prefill(
        PrefillWork(
            request_ids=("source.0", "source.1"),
            token_rows=((1, 2), (3, 4)),
            state=source,
            parent=source.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    runtime.commit(step, (2, 1))
    parent = source.observe()
    source._cache.keys[0][0, 2:].fill_(91)
    source._cache.values[0][0, 2:].fill_(92)
    source._cache.keys[0][1, 1:].fill_(93)
    source._cache.values[0][1, 1:].fill_(94)

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
    assert result.forked.lengths == (2, 1)
    assert result.forked.capacity == 4
    assert result.forked.generation > parent.generation
    assert result.state_bytes_copied == 48
    assert torch.equal(
        result.state._cache.keys[0][0, :2],
        source._cache.keys[0][0, :2],
    )
    assert torch.equal(
        result.state._cache.keys[0][1, :1],
        source._cache.keys[0][1, :1],
    )
    assert torch.count_nonzero(result.state._cache.keys[0][0, 2:]) == 0
    assert torch.count_nonzero(result.state._cache.keys[0][1, 1:]) == 0
    assert source.observe() == parent

    fork_step = runtime.decode(
        DecodeWork(
            request_ids=("fork.0", "fork.1"),
            token_rows=((5,), (6,)),
            state=result.state,
            parent=result.forked,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    runtime.commit(fork_step, (1, 1))
    assert result.state.observe().lengths == (3, 2)
    assert source.observe() == parent
    counters = dict(runtime.telemetry().extra_counters)
    assert counters["state_forks"] == 1
    assert counters["state_fork_tokens"] == 3
    assert counters["state_fork_bytes"] == 48


def test_dense_native_runtime_fork_rejects_pending_stale_capacity_foreign_and_released() -> None:
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
    with pytest.raises(DenseCudaRuntimeError, match="pending"):
        runtime.fork_state(source, parent=empty, owner_id="copy", capacity=4)
    runtime.commit(pending, (2,))
    committed = source.observe()

    with pytest.raises(DenseCudaRuntimeError, match="stale"):
        runtime.fork_state(source, parent=empty, owner_id="copy", capacity=4)
    with pytest.raises(OverflowError, match="capacity"):
        runtime.fork_state(source, parent=committed, owner_id="copy", capacity=1)
    with pytest.raises(ValueError, match="admitted placement"):
        runtime.fork_state(source, parent=committed, owner_id="copy", capacity=9)

    other, _ = _direct_runtime()
    with pytest.raises(DenseCudaRuntimeError, match="another runtime"):
        other.fork_state(source, parent=committed, owner_id="copy", capacity=4)
    runtime.release_state(source)
    with pytest.raises(DenseCudaRuntimeError, match="stale or released"):
        runtime.fork_state(source, parent=committed, owner_id="copy", capacity=4)


def test_dense_native_runtime_fork_rejects_abi_storage_and_target_alias_drift() -> None:
    runtime, _engine = _direct_runtime()
    source = runtime.allocate_state(owner_id="source", batch_size=1, capacity=4)
    source._state_abi = "foreign-kv-v2"
    with pytest.raises(DenseCudaRuntimeError, match="ABI"):
        runtime.fork_state(
            source,
            parent=source.observe(),
            owner_id="copy",
            capacity=4,
        )

    runtime2, _engine = _direct_runtime()
    damaged = runtime2.allocate_state(owner_id="source", batch_size=1, capacity=4)
    damaged_parent = damaged.observe()
    damaged._cache.keys[0] = damaged._cache.keys[0].clone()
    with pytest.raises(DenseCudaRuntimeError, match="backing storage identity changed"):
        runtime2.fork_state(
            damaged,
            parent=damaged_parent,
            owner_id="copy",
            capacity=4,
        )

    runtime3, _engine = _direct_runtime()
    aliased = runtime3.allocate_state(owner_id="source", batch_size=1, capacity=4)
    runtime3._cache_factory = lambda _batch, _capacity: aliased._cache
    with pytest.raises(DenseCudaRuntimeError, match="aliases source"):
        runtime3.fork_state(
            aliased,
            parent=aliased.observe(),
            owner_id="copy",
            capacity=4,
        )

    runtime4, _engine = _direct_runtime()
    drifting = runtime4.allocate_state(owner_id="source", batch_size=1, capacity=4)

    def drifting_factory(batch_size: int, capacity: int) -> DenseQStoreKVCache:
        cache = _cache_factory(batch_size, capacity)
        install = cache.install_requests

        def drifting_install(*args: Any, **kwargs: Any) -> Any:
            stats = install(*args, **kwargs)
            cache.keys[0] = cache.keys[0].clone()
            return stats

        cache.install_requests = drifting_install  # type: ignore[method-assign]
        return cache

    runtime4._cache_factory = drifting_factory
    with pytest.raises(DenseCudaRuntimeError, match="changed during copy"):
        runtime4.fork_state(
            drifting,
            parent=drifting.observe(),
            owner_id="copy",
            capacity=4,
        )


def test_dense_native_runtime_rejects_stale_foreign_and_double_terminal_authority() -> None:
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
    with pytest.raises(DenseCudaRuntimeError, match="unconsumed"):
        runtime.prefill(work)
    with pytest.raises(DenseCudaRuntimeError, match="pending"):
        runtime.release_state(state)
    with pytest.raises(TypeError, match="strict integers"):
        runtime.commit(step, (True,))
    assert state.observe() == parent

    runtime.commit(step, (2,))
    with pytest.raises(DenseCudaRuntimeError, match="already been consumed"):
        runtime.commit(step, (2,))
    with pytest.raises(DenseCudaRuntimeError, match="already been consumed"):
        runtime.abandon(step)
    with pytest.raises(DenseCudaRuntimeError, match="stale"):
        runtime.prefill(work)

    other, _ = _direct_runtime()
    with pytest.raises(DenseCudaRuntimeError, match="another runtime"):
        other.release_state(state)
    runtime.release_state(state)
    with pytest.raises(DenseCudaRuntimeError, match="released"):
        state.observe()


def test_dense_native_runtime_close_is_fail_closed_with_pending_work() -> None:
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
    with pytest.raises(DenseCudaRuntimeError, match="pending provisional"):
        runtime.close()
    assert engine.close_calls == 0
    assert engine._last_cache is state._cache
    result_reference = weakref.ref(step.authority._result)
    runtime.abandon(step)
    assert step.authority._result is None
    assert result_reference() is None
    cache_reference = weakref.ref(state._cache)
    runtime.close()
    runtime.close()
    assert state._cache is None
    assert engine._last_cache is None
    assert engine.close_calls == 1
    gc.collect()
    assert cache_reference() is None


def test_dense_native_runtime_failed_commit_retains_retryable_provisional_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, engine = _direct_runtime()
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
    result = step.authority._result
    real_commit = engine.commit_block

    def fail_commit(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(engine, "commit_block", fail_commit)
    with pytest.raises(RuntimeError, match="injected commit failure"):
        runtime.commit(step, (1,))

    assert step.authority._result is result
    assert step.authority._consumed is False
    assert state._pending_step_id == step.step_id
    assert state.observe() == step.parent

    monkeypatch.setattr(engine, "commit_block", real_commit)
    runtime.commit(step, (1,))
    assert step.authority._result is None


def test_dense_native_runtime_binds_backing_storage_and_recovers_from_forward_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, engine = _direct_runtime()
    state = runtime.allocate_state(owner_id="owner", batch_size=1, capacity=4)
    parent = state.observe()
    work = PrefillWork(
        request_ids=("request",),
        token_rows=((1,),),
        state=state,
        parent=parent,
        output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
    )
    real_forward = runtime._forward_argmax

    def fail_forward(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected native failure")

    monkeypatch.setattr(runtime, "_forward_argmax", fail_forward)
    with pytest.raises(RuntimeError, match="injected native failure"):
        runtime.prefill(work)
    monkeypatch.setattr(runtime, "_forward_argmax", real_forward)
    step = runtime.prefill(work)
    runtime.abandon(step)

    state._cache.keys[0] = state._cache.keys[0].clone()
    with pytest.raises(DenseCudaRuntimeError, match="backing storage identity changed"):
        state.observe()


def test_dense_native_runtime_bind_validates_compiler_capability_and_placement_chain() -> None:
    blob = BlobIdentity(blob_id="body.bin", sha256="1" * 64, byte_count=100)
    component = CompiledComponent(
        component_id="body",
        role="body",
        allocation_id="allocation.body",
        codec_id="qrow-int8",
        layout_id="row-major",
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
        state_dtype="float32",
        state_bytes_per_token=16,
        max_context_tokens=8,
        semantic_token_count=10,
    )
    capabilities = BackendCapabilities(
        backend_id="dense-qstore-cuda",
        backend_abi="dense-cuda-v1",
        implementation_version="test",
        fabric="cuda-sm89",
        memory_domain=MemoryDomain.CUDA,
        architectures=("qwen2",),
        operator_ids=("qwen2.dense-decoder",),
        codecs=(
            CodecCapability(
                codec_id="qrow-int8",
                layout_id="row-major",
                component_roles=("body",),
            ),
        ),
        state_abis=("gqa-kv-v1",),
        output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,),
        numerical_contracts=("fake-qstore-exact-v1",),
        max_context_tokens=8,
        max_batch_size=2,
        max_verify_tokens=1,
        transactional_state=True,
        scratch_only_steps=True,
        independently_committable_rows=True,
        supports_ragged_batches=False,
        telemetry_counters=("committed_tokens",),
        promotion_status=PromotionStatus.CANDIDATE,
    )
    device = DeviceDescriptor(
        device_id="cuda:0",
        fabric="cuda-sm89",
        memory_domain=MemoryDomain.CUDA,
        total_bytes=10_000,
        available_bytes=10_000,
        machine_fingerprint="6" * 64,
    )
    workload = WorkloadSpec(
        max_batch_size=2,
        max_context_tokens=8,
        verify_tokens=1,
        output_mode=OutputMode.NEXT_TOKEN_ARGMAX,
        numerical_contract="fake-qstore-exact-v1",
        state_abi="gqa-kv-v1",
        required_component_roles=("body",),
        workspace_bytes=100,
        headroom_bytes=100,
    )
    placement = plan_resident_placement(model, workload, capabilities, device)
    engine = _FakeDenseEngine()
    runtime = DenseCudaNativeRuntime.bind(
        engine,
        model=model,
        workload=workload,
        capabilities=capabilities,
        device=device,
        placement=placement,
        cache_factory=_cache_factory,
    )

    assert runtime.route.model_fingerprint == model.fingerprint
    assert runtime.route.capability_fingerprint == capabilities.fingerprint
    assert runtime.route.placement_fingerprint == placement.fingerprint
    assert runtime.route.backend_id == "dense-qstore-cuda"


def _slot_batch_result(
    input_ids: np.ndarray,
    indexed_cache: DenseCudaIndexedKVCache,
) -> DenseForwardResult:
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    slot_ids = tuple(int(value) for value in indexed_cache.row_indices.detach().cpu().tolist())
    parent_lengths = tuple(int(value) for value in indexed_cache.lengths)
    assert tuple(ids.shape) == (len(slot_ids), 1)
    assert len(parent_lengths) == len(slot_ids)
    values = ids.to(torch.float32)
    key = torch.stack((values, values + 0.25), dim=-1).unsqueeze(2)
    value = torch.stack((values + 0.5, values + 0.75), dim=-1).unsqueeze(2)
    delta = KVDelta(
        parent_epoch=0,
        parent_lengths=parent_lengths,
        cache_id=indexed_cache.cache_id,
        keys=(key,),
        values=(value,),
        token_count=1,
    )
    return DenseForwardResult(
        top1=(ids + 1) % 10,
        delta=delta,
        hidden=torch.zeros((*ids.shape, 2), dtype=torch.float32),
        kv_read_bytes=0,
        kv_delta_bytes=delta.byte_count,
        wall_s=0.001,
    )


def _commit_prompt(
    runtime: DenseCudaNativeRuntime,
    state: Any,
    request_id: str,
    token_ids: tuple[int, ...],
) -> None:
    step = runtime.prefill(
        PrefillWork(
            request_ids=(request_id,),
            token_rows=(token_ids,),
            state=state,
            parent=state.observe(),
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )
    runtime.commit(step, (len(token_ids),))


def _decode_work(state: Any, request_id: str, token_id: int) -> DecodeWork:
    return DecodeWork(
        request_ids=(request_id,),
        token_rows=((token_id,),),
        state=state,
        parent=state.observe(),
        output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
    )


def test_dense_cuda_slot_pool_is_fixed_b1_storage_and_rejects_slot_aba() -> None:
    runtime, _engine = _direct_runtime()
    lane = DenseCudaCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_slots=2,
        batch_executor=_slot_batch_result,
    )
    first = runtime.allocate_state(owner_id="slot.0", batch_size=1, capacity=4)
    second = runtime.allocate_state(owner_id="slot.1", batch_size=1, capacity=4)
    first_cache = first._cache
    second_cache = second._cache

    assert isinstance(first_cache, DenseCudaKVSlotCache)
    assert isinstance(second_cache, DenseCudaKVSlotCache)
    assert (first_cache.slot, second_cache.slot) == (0, 1)
    assert first_cache.keys[0].shape == second_cache.keys[0].shape == (1, 4, 1, 2)
    assert first_cache.keys[0].data_ptr() != second_cache.keys[0].data_ptr()
    assert lane.pool.keys[0].shape == (2, 8, 1, 2)
    assert lane.pool.allocated_bytes == 16 * 2 * 8
    assert runtime.telemetry().kv_resident_bytes == lane.pool.allocated_bytes
    with pytest.raises(MemoryError, match="exhausted"):
        runtime.allocate_state(owner_id="slot.full", batch_size=1, capacity=4)

    old_cache_id = first_cache.cache_id
    old_generation = first_cache.slot_generation
    old_pointer = first_cache.keys[0].data_ptr()
    runtime.release_state(first)
    replacement = runtime.allocate_state(owner_id="slot.reused", batch_size=1, capacity=4)
    replacement_cache = replacement._cache

    assert isinstance(replacement_cache, DenseCudaKVSlotCache)
    assert replacement_cache.slot == 0
    assert replacement_cache.keys[0].data_ptr() == old_pointer
    assert replacement_cache.slot_generation == old_generation + 1
    assert replacement_cache.cache_id != old_cache_id
    with pytest.raises(DenseCudaRuntimeError, match="stale or released"):
        _ = first_cache.committed_bytes

    runtime.release_state(second)
    runtime.release_state(replacement)
    assert lane.pool.active_slots == 0
    assert runtime.telemetry().kv_resident_bytes == lane.pool.allocated_bytes
    runtime.close()


def test_dense_cuda_lane_default_path_passes_engine_indexed_cache_positionally() -> None:
    runtime, engine = _direct_runtime(production_slot_seam=True)
    lane = DenseCudaCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_slots=2,
    )
    first = runtime.allocate_state(owner_id="engine-seam.0", batch_size=1, capacity=8)
    second = runtime.allocate_state(owner_id="engine-seam.1", batch_size=1, capacity=8)
    _commit_prompt(runtime, first, "engine-seam.0", (1, 2))
    _commit_prompt(runtime, second, "engine-seam.1", (3,))

    steps = lane.execute(
        (
            _decode_work(first, "engine-seam.0", 4),
            _decode_work(second, "engine-seam.1", 5),
        )
    )

    assert engine.forward_decode_slots_calls == 1
    assert engine.indexed_decode_calls == [
        (
            (2, 1),
            2,
            (2, 1),
            (0, 1),
            tuple(int(tensor.data_ptr()) for tensor in lane.pool.keys),
        )
    ]
    assert engine._last_cache is None
    runtime.commit(steps[0], (1,))
    runtime.abandon(steps[1])
    assert first.observe().lengths == (3,)
    assert second.observe().lengths == (1,)


def test_dense_cuda_lane_default_path_requires_segmented_decode_contract() -> None:
    runtime, _engine = _direct_runtime()

    with pytest.raises(DenseCudaRuntimeError, match="requires segmented-flash"):
        DenseCudaCompatibleBatchLane(runtime, max_batch_size=2, max_slots=2)
    assert runtime._kv_slot_pool is None


def test_dense_cuda_lane_rejects_delta_not_bound_to_indexed_cache() -> None:
    runtime, _engine = _direct_runtime()

    def foreign_delta(
        input_ids: np.ndarray,
        indexed_cache: DenseCudaIndexedKVCache,
    ) -> DenseForwardResult:
        result = _slot_batch_result(input_ids, indexed_cache)
        delta = KVDelta(
            parent_epoch=result.delta.parent_epoch,
            parent_lengths=result.delta.parent_lengths,
            cache_id="foreign-indexed-cache",
            keys=result.delta.keys,
            values=result.delta.values,
            token_count=result.delta.token_count,
        )
        return DenseForwardResult(
            top1=result.top1,
            delta=delta,
            hidden=result.hidden,
            kv_read_bytes=result.kv_read_bytes,
            kv_delta_bytes=result.kv_delta_bytes,
            wall_s=result.wall_s,
        )

    lane = DenseCudaCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_slots=2,
        batch_executor=foreign_delta,
    )
    first = runtime.allocate_state(owner_id="foreign-delta.0", batch_size=1, capacity=8)
    second = runtime.allocate_state(owner_id="foreign-delta.1", batch_size=1, capacity=8)
    _commit_prompt(runtime, first, "foreign-delta.0", (1,))
    _commit_prompt(runtime, second, "foreign-delta.1", (2,))

    with pytest.raises(DenseCudaRuntimeError, match="does not bind its indexed cache"):
        lane.execute(
            (
                _decode_work(first, "foreign-delta.0", 3),
                _decode_work(second, "foreign-delta.1", 4),
            )
        )
    assert first._pending_step_id is second._pending_step_id is None
    assert lane.telemetry().active_scratches == 0


def test_dense_cuda_compatible_batch_is_prefix_copy_free_and_row_terminal() -> None:
    runtime, _engine = _direct_runtime()
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def execute(
        input_ids: np.ndarray,
        indexed_cache: DenseCudaIndexedKVCache,
    ) -> DenseForwardResult:
        assert indexed_cache._pool is lane.pool
        slot_ids = tuple(
            int(value) for value in indexed_cache.row_indices.detach().cpu().tolist()
        )
        parent_lengths = tuple(int(value) for value in indexed_cache.lengths)
        calls.append((slot_ids, parent_lengths))
        return _slot_batch_result(input_ids, indexed_cache)

    lane = DenseCudaCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_slots=2,
        batch_executor=execute,
    )
    first = runtime.allocate_state(owner_id="batch.0", batch_size=1, capacity=8)
    second = runtime.allocate_state(owner_id="batch.1", batch_size=1, capacity=8)
    _commit_prompt(runtime, first, "batch.0", (1, 2))
    _commit_prompt(runtime, second, "batch.1", (3,))
    parents = (first.observe(), second.observe())
    before_keys = tuple(tensor.clone() for tensor in lane.pool.keys)
    before_values = tuple(tensor.clone() for tensor in lane.pool.values)

    steps = lane.execute(
        (
            _decode_work(first, "batch.0", 5),
            _decode_work(second, "batch.1", 6),
        )
    )

    assert calls == [((0, 1), (2, 1))]
    assert tuple(step.output.token_ids for step in steps) == ((6,), (7,))
    assert (first.observe(), second.observe()) == parents
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_keys, lane.pool.keys, strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_values, lane.pool.values, strict=True)
    )
    scratch = steps[0].authority._scratch
    assert scratch is steps[1].authority._scratch
    result_reference = weakref.ref(scratch.result)

    runtime.commit(steps[0], (1,))
    assert first.observe().lengths == (3,)
    assert second.observe() == parents[1]
    assert torch.equal(
        first._cache.keys[0][0, :3, 0, 0],
        torch.tensor([1.0, 2.0, 5.0]),
    )
    assert result_reference() is not None
    runtime.abandon(steps[1])
    gc.collect()
    assert second.observe() == parents[1]
    assert scratch.result is None
    assert result_reference() is None
    assert all(step.authority._scratch is None for step in steps)

    telemetry = lane.telemetry()
    assert telemetry.dispatches == telemetry.physical_forwards == 1
    assert telemetry.provisional_rows == 2
    assert telemetry.committed_rows == telemetry.abandoned_rows == 1
    assert telemetry.active_scratches == 0
    assert telemetry.width_histogram == ((2, 1),)
    runtime_telemetry = runtime.telemetry()
    assert runtime_telemetry.decode_calls == 1
    assert runtime_telemetry.decode_tokens == 2
    assert runtime_telemetry.device_to_host_bytes == 32


def test_dense_cuda_batch_failed_commit_keeps_shared_scratch_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _engine = _direct_runtime()
    lane = DenseCudaCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_slots=2,
        batch_executor=_slot_batch_result,
    )
    first = runtime.allocate_state(owner_id="retry.0", batch_size=1, capacity=8)
    second = runtime.allocate_state(owner_id="retry.1", batch_size=1, capacity=8)
    _commit_prompt(runtime, first, "retry.0", (1,))
    _commit_prompt(runtime, second, "retry.1", (2,))
    steps = lane.execute(
        (
            _decode_work(first, "retry.0", 3),
            _decode_work(second, "retry.1", 4),
        )
    )
    shared = steps[0].authority._scratch
    original = DenseCudaKVSlotCache.commit_batch_row
    failed = False

    def fail_once(cache: DenseCudaKVSlotCache, *args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        if cache is first._cache and not failed:
            failed = True
            raise RuntimeError("injected slot-row commit failure")
        return original(cache, *args, **kwargs)

    monkeypatch.setattr(DenseCudaKVSlotCache, "commit_batch_row", fail_once)
    with pytest.raises(RuntimeError, match="injected slot-row commit failure"):
        runtime.commit(steps[0], (1,))

    assert steps[0].authority._scratch is shared
    assert steps[0].authority._consumed is False
    assert shared.result is not None
    assert first.observe() == steps[0].parent
    assert first._pending_step_id == steps[0].step_id
    assert second._pending_step_id == steps[1].step_id

    runtime.commit(steps[0], (1,))
    runtime.abandon(steps[1])
    assert first.observe().lengths == (2,)
    assert lane.telemetry().active_scratches == 0


def test_dense_cuda_batch_rejects_duplicates_stale_parents_and_recovers_forward_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _engine = _direct_runtime()
    lane = DenseCudaCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_slots=2,
        batch_executor=_slot_batch_result,
    )
    first = runtime.allocate_state(owner_id="reject.0", batch_size=1, capacity=8)
    second = runtime.allocate_state(owner_id="reject.1", batch_size=1, capacity=8)
    _commit_prompt(runtime, first, "reject.0", (1,))
    _commit_prompt(runtime, second, "reject.1", (2,))

    with pytest.raises(ValueError, match="request IDs must be unique"):
        lane.execute(
            (
                _decode_work(first, "duplicate", 3),
                _decode_work(second, "duplicate", 4),
            )
        )
    with pytest.raises(ValueError, match="state IDs must be unique"):
        lane.execute(
            (
                _decode_work(first, "duplicate-state.0", 3),
                _decode_work(first, "duplicate-state.1", 4),
            )
        )

    stale = _decode_work(first, "stale.0", 3)
    advance = _decode_work(first, "advance", 5)
    runtime.commit(runtime.decode(advance), (1,))
    second_parent = second.observe()
    with pytest.raises(DenseCudaRuntimeError, match="parent is stale"):
        lane.execute((stale, _decode_work(second, "stale.1", 4)))
    assert second.observe() == second_parent
    assert second._pending_step_id is None

    real_executor = lane._executor

    def fail_forward(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected compatible forward failure")

    monkeypatch.setattr(lane, "_executor", fail_forward)
    works = (
        _decode_work(first, "recover.0", 6),
        _decode_work(second, "recover.1", 7),
    )
    with pytest.raises(RuntimeError, match="injected compatible forward failure"):
        lane.execute(works)
    assert first._pending_step_id is second._pending_step_id is None
    assert first.observe() == works[0].parent
    assert second.observe() == works[1].parent

    monkeypatch.setattr(lane, "_executor", real_executor)
    recovered = lane.execute(works)
    runtime.abandon(recovered[0])
    runtime.abandon(recovered[1])
    assert lane.telemetry().failed_dispatches == 2
    assert lane.telemetry().active_scratches == 0


def test_dense_cuda_batch_pending_release_is_fail_closed_and_slot_is_new_generation() -> None:
    runtime, _engine = _direct_runtime()
    lane = DenseCudaCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_slots=2,
        batch_executor=_slot_batch_result,
    )
    first = runtime.allocate_state(owner_id="release.0", batch_size=1, capacity=8)
    second = runtime.allocate_state(owner_id="release.1", batch_size=1, capacity=8)
    _commit_prompt(runtime, first, "release.0", (1,))
    _commit_prompt(runtime, second, "release.1", (2,))
    steps = lane.execute(
        (
            _decode_work(first, "release.0", 3),
            _decode_work(second, "release.1", 4),
        )
    )
    old_slot = first._cache.slot
    old_generation = first._cache.slot_generation

    with pytest.raises(DenseCudaRuntimeError, match="pending provisional"):
        runtime.release_state(first)
    with pytest.raises(DenseCudaRuntimeError, match="pending provisional"):
        runtime.close()
    runtime.abandon(steps[0])
    runtime.abandon(steps[1])
    runtime.release_state(first)
    replacement = runtime.allocate_state(owner_id="release.replacement", batch_size=1, capacity=8)

    assert replacement._cache.slot == old_slot
    assert replacement._cache.slot_generation == old_generation + 1
    runtime.release_state(second)
    runtime.release_state(replacement)
    runtime.close()


def test_dense_cuda_generation_service_cancels_one_slot_row_and_commits_the_other() -> None:
    runtime, _engine = _direct_runtime()
    started = threading.Event()
    release = threading.Event()

    def blocking_execute(
        input_ids: np.ndarray,
        indexed_cache: DenseCudaIndexedKVCache,
    ) -> DenseForwardResult:
        started.set()
        if not release.wait(2):
            raise RuntimeError("test did not release CUDA batch executor")
        return _slot_batch_result(input_ids, indexed_cache)

    lane = DenseCudaCompatibleBatchLane(
        runtime,
        max_batch_size=2,
        max_slots=2,
        max_queue_delay_seconds=0.01,
        batch_executor=blocking_execute,
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
    try:
        cancelled, completed = service.submit_many(
            (
                GenerationRequest("cancel-slot", (1, 2), 2, stream=False),
                GenerationRequest("commit-slot", (2, 3), 2, stream=False),
            )
        )
        assert started.wait(2)
        assert cancelled.cancel("test-slot-cancel")
        release.set()

        with pytest.raises(GenerationCancelled, match="test-slot-cancel"):
            cancelled.result(timeout=2)
        assert completed.result(timeout=2).token_ids == (4, 5)
        telemetry = lane.telemetry()
        assert telemetry.committed_rows == telemetry.abandoned_rows == 1
        assert telemetry.active_scratches == 0
    finally:
        release.set()
        assert service.shutdown(wait=True, timeout=2)
    assert lane.pool.active_slots == 0
    runtime.close()
