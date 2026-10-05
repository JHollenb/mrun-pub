from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mrun.runtime.contracts import (
    DecodeWork,
    OutputMode,
    OutputRequest,
    PrefillWork,
    PromotionStatus,
    RuntimeRoute,
)
from mrun.runtime.inference import FinishReason, GenerationRequest, NativeGenerationService
from mrun.runtime.qwen3_moe_continuous import (
    Qwen3MoeCompatibleBatchLane,
    Qwen3MoeContinuousRuntimeError,
    Qwen3MoeCudaNativeRuntime,
    Qwen3MoeDecodeBatch,
    Qwen3MoeDecodeBatchResult,
    build_qwen3_moe_continuous_service,
)


@dataclass(slots=True)
class _LayerKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class _Cache:
    layers: list[_LayerKV]
    batch_size: int
    capacity: int
    length: int = 0

    @property
    def device_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in self.layers
            for tensor in (layer.key, layer.value)
        )


class _FakeDecodeRuntime:
    device = "cpu"

    def new_cache(self, batch_size: int, capacity: int) -> _Cache:
        return _Cache(
            layers=[
                _LayerKV(
                    key=torch.zeros((batch_size, 1, capacity, 1), dtype=torch.float32),
                    value=torch.zeros((batch_size, 1, capacity, 1), dtype=torch.float32),
                )
            ],
            batch_size=batch_size,
            capacity=capacity,
        )

    def forward(self, input_ids: torch.Tensor, *, cache: _Cache) -> Any:
        start = cache.length
        end = start + int(input_ids.shape[1])
        cache.layers[0].key[:, 0, start:end, 0].copy_(input_ids.float())
        cache.layers[0].value[:, 0, start:end, 0].copy_(input_ids.float() + 1000)
        cache.length = end
        next_token = int(input_ids[0, -1].item()) + 1
        logits = torch.full((1, 128), -1000.0)
        logits[0, next_token] = 1.0
        return SimpleNamespace(logits=logits)


class _FakeEngine:
    def __init__(self) -> None:
        self.runtime = _FakeDecodeRuntime()
        self.working_set_mb = 1.0


def _route() -> RuntimeRoute:
    return RuntimeRoute(
        runtime_id="qwen3-test-runtime",
        model_fingerprint="a" * 64,
        capability_fingerprint="b" * 64,
        placement_fingerprint="c" * 64,
        backend_id="qwen3-moe-cuda-test",
        device_id="cpu:test",
        promotion_status=PromotionStatus.EXPERIMENTAL,
    )


def _runtime(
    *,
    slots: int = 3,
    max_context_tokens: int = 16,
) -> Qwen3MoeCudaNativeRuntime:
    return Qwen3MoeCudaNativeRuntime(
        _FakeEngine(),
        route=_route(),
        max_slots=slots,
        max_context_tokens=max_context_tokens,
        synchronize=lambda: None,
    )


def _prefill(runtime: Qwen3MoeCudaNativeRuntime, state: Any, *tokens: int) -> Any:
    parent = state.observe()
    return runtime.prefill(
        PrefillWork(
            request_ids=(state.owner_id,),
            token_rows=(tuple(tokens),),
            state=state,
            parent=parent,
            output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
        )
    )


def _decode(runtime: Qwen3MoeCudaNativeRuntime, state: Any, token: int) -> DecodeWork:
    return DecodeWork(
        request_ids=(state.owner_id,),
        token_rows=((token,),),
        state=state,
        parent=state.observe(),
        output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
    )


def _prefilled_state(
    runtime: Qwen3MoeCudaNativeRuntime,
    owner_id: str,
    length: int,
) -> Any:
    state = runtime.allocate_state(
        owner_id=owner_id,
        batch_size=1,
        capacity=runtime.pool.max_context_tokens,
    )
    step = _prefill(runtime, state, *((1,) * length))
    runtime.commit(step, (length,))
    return state


class _RaggedExecutor:
    def __init__(self, *, block_first: bool = False) -> None:
        self.batches: list[Qwen3MoeDecodeBatch] = []
        self.request_calls: dict[str, int] = {}
        self.first_started = threading.Event()
        self.first_gate = threading.Event()
        if not block_first:
            self.first_gate.set()

    def execute(
        self,
        batch: Qwen3MoeDecodeBatch,
        *,
        cache: _Cache,
    ) -> Qwen3MoeDecodeBatchResult:
        self.batches.append(batch)
        if len(self.batches) == 1:
            self.first_started.set()
            assert self.first_gate.wait(5)
        selected: list[int] = []
        for request_id, slot, length, token in zip(
            batch.request_ids,
            batch.physical_slots,
            batch.parent_lengths,
            batch.input_token_ids,
            strict=True,
        ):
            cache.layers[0].key[slot, 0, length, 0] = token
            cache.layers[0].value[slot, 0, length, 0] = token + 1000
            call = self.request_calls.get(request_id, 0) + 1
            self.request_calls[request_id] = call
            selected.append(token + 1 if call == 1 else 99)
        return Qwen3MoeDecodeBatchResult(
            dispatch_id=batch.dispatch_id,
            row_bindings=batch.row_bindings,
            next_token_ids=tuple(selected),
        )


class _FailingRaggedExecutor:
    def __init__(self) -> None:
        self.batches: list[Qwen3MoeDecodeBatch] = []

    def execute(
        self,
        batch: Qwen3MoeDecodeBatch,
        *,
        cache: _Cache,
    ) -> Qwen3MoeDecodeBatchResult:
        self.batches.append(batch)
        slot = batch.physical_slots[0]
        length = batch.parent_lengths[0]
        cache.layers[0].key[slot, 0, length, 0] = 777
        raise RuntimeError("injected ragged failure")


class _FailingOnCallRaggedExecutor(_RaggedExecutor):
    def __init__(self, fail_call: int) -> None:
        super().__init__()
        self.fail_call = fail_call

    def execute(
        self,
        batch: Qwen3MoeDecodeBatch,
        *,
        cache: _Cache,
    ) -> Qwen3MoeDecodeBatchResult:
        if len(self.batches) + 1 == self.fail_call:
            self.batches.append(batch)
            slot = batch.physical_slots[0]
            length = batch.parent_lengths[0]
            cache.layers[0].key[slot, 0, length, 0] = 888
            raise RuntimeError("injected later bucket failure")
        return super().execute(batch, cache=cache)


def test_qwen_b1_tail_is_provisional_and_slot_generation_prevents_reuse() -> None:
    runtime = _runtime(slots=1)
    state = runtime.allocate_state(owner_id="request-a", batch_size=1, capacity=8)

    prefill = _prefill(runtime, state, 3, 4)
    assert prefill.output.token_ids == (5,)
    assert state.observe().lengths == (0,)
    assert state._cache.length == 0  # noqa: SLF001 - asserts the provisional ABI
    assert state._cache.layers[0].key[0, 0, :2, 0].tolist() == [3.0, 4.0]  # noqa: SLF001

    committed = runtime.commit(prefill, (2,))
    assert committed.before.lengths == (0,)
    assert committed.after.lengths == (2,)
    assert committed.state_bytes_written == 16

    decode = runtime.decode(_decode(runtime, state, 5))
    assert decode.output.token_ids == (6,)
    assert state.observe().lengths == (2,)
    runtime.abandon(decode)
    assert state.observe().lengths == (2,)

    first_slot = state.slot
    first_storage_generation = state.storage_generation
    runtime.release_state(state)
    replacement = runtime.allocate_state(owner_id="request-b", batch_size=1, capacity=8)
    assert replacement.slot == first_slot
    assert replacement.storage_generation == first_storage_generation + 1
    try:
        state.observe()
    except Qwen3MoeContinuousRuntimeError:
        pass
    else:
        raise AssertionError("released Qwen state remained observable")

    pool = runtime.pool.telemetry()
    assert pool.allocations == 2
    assert pool.releases == 1
    assert pool.slot_reuses == 1
    runtime.release_state(replacement)
    runtime.close()


def test_singleton_ragged_decode_is_strict_opt_in_and_default_stays_scalar() -> None:
    runtime = _runtime(slots=2)
    state = runtime.allocate_state(owner_id="request-default", batch_size=1, capacity=8)
    prefill = _prefill(runtime, state, 4)
    runtime.commit(prefill, (1,))
    executor = _RaggedExecutor()
    lane = Qwen3MoeCompatibleBatchLane(
        runtime,
        executor,
        max_batch_size=2,
        max_queue_delay_seconds=0,
    )

    step = lane.execute((_decode(runtime, state, 5),))[0]

    assert step.output.token_ids == (6,)
    assert executor.batches == []
    assert lane.identity.dispatches_singletons is False
    telemetry = lane.telemetry()
    assert telemetry.singleton_ragged_decode_enabled is False
    assert telemetry.singleton_decode_fallback_rows == 1
    assert telemetry.singleton_ragged_decode_rows == 0
    runtime.abandon(step)
    runtime.release_state(state)
    runtime.close()


def test_opt_in_singleton_ragged_commit_abandon_and_failure_cleanup() -> None:
    runtime = _runtime(slots=2)
    state = runtime.allocate_state(owner_id="request-opt-in", batch_size=1, capacity=8)
    prefill = _prefill(runtime, state, 3)
    runtime.commit(prefill, (1,))
    executor = _RaggedExecutor()
    lane = Qwen3MoeCompatibleBatchLane(
        runtime,
        executor,
        max_batch_size=2,
        max_queue_delay_seconds=0,
        enable_singleton_ragged_decode=True,
    )

    committed_step = lane.execute((_decode(runtime, state, 10),))[0]
    committed = runtime.commit(committed_step, (1,))
    assert committed.after.lengths == (2,)
    abandoned_step = lane.execute((_decode(runtime, state, 11),))[0]
    runtime.abandon(abandoned_step)
    assert state.observe().lengths == (2,)
    assert state._pending is None  # noqa: SLF001 - verifies lifecycle cleanup

    failing = Qwen3MoeCompatibleBatchLane(
        runtime,
        _FailingRaggedExecutor(),
        max_batch_size=2,
        max_queue_delay_seconds=0,
        enable_singleton_ragged_decode=True,
    )
    with pytest.raises(RuntimeError, match="injected ragged failure"):
        failing.execute((_decode(runtime, state, 12),))
    assert state.observe().lengths == (2,)
    assert state._cache.length == 2  # noqa: SLF001 - failed tail remains non-authoritative
    assert state._pending is None  # noqa: SLF001

    assert lane.identity.dispatches_singletons is True
    telemetry = lane.telemetry()
    assert telemetry.singleton_ragged_decode_enabled is True
    assert telemetry.singleton_decode_fallback_rows == 0
    assert telemetry.singleton_ragged_decode_rows == 2
    assert telemetry.width_histogram == ((1, 2),)
    runtime.release_state(state)
    runtime.close()


def test_opt_in_singleton_ragged_binds_reused_slot_generation() -> None:
    runtime = _runtime(slots=2)
    original = runtime.allocate_state(owner_id="request-original", batch_size=1, capacity=8)
    original_slot = original.slot
    original_generation = original.storage_generation
    runtime.release_state(original)

    replacement = runtime.allocate_state(owner_id="request-replacement", batch_size=1, capacity=8)
    prefill = _prefill(runtime, replacement, 8)
    runtime.commit(prefill, (1,))
    executor = _RaggedExecutor()
    lane = Qwen3MoeCompatibleBatchLane(
        runtime,
        executor,
        max_batch_size=2,
        max_queue_delay_seconds=0,
        enable_singleton_ragged_decode=True,
    )

    step = lane.execute((_decode(runtime, replacement, 9),))[0]

    batch = executor.batches[0]
    assert batch.physical_slots == (original_slot,)
    assert batch.storage_generations == (original_generation + 1,)
    assert batch.parent_lengths == (1,)
    assert batch.row_bindings == ((original_slot, original_generation + 1, 1),)
    runtime.abandon(step)
    runtime.release_state(replacement)
    runtime.close()


def test_opt_in_mixed_prefill_and_singleton_decode_uses_only_ragged_decode() -> None:
    runtime = _runtime(slots=2)
    fresh = runtime.allocate_state(owner_id="request-prefill", batch_size=1, capacity=8)
    decoding = runtime.allocate_state(owner_id="request-decode", batch_size=1, capacity=8)
    initial = _prefill(runtime, decoding, 6)
    runtime.commit(initial, (1,))
    prefill_work = PrefillWork(
        request_ids=(fresh.owner_id,),
        token_rows=((1, 2),),
        state=fresh,
        parent=fresh.observe(),
        output=OutputRequest(OutputMode.NEXT_TOKEN_ARGMAX),
    )
    executor = _RaggedExecutor()
    lane = Qwen3MoeCompatibleBatchLane(
        runtime,
        executor,
        max_batch_size=2,
        max_queue_delay_seconds=0,
        enable_singleton_ragged_decode=True,
    )

    prefill_step, decode_step = lane.execute(
        (prefill_work, _decode(runtime, decoding, 7))
    )

    assert prefill_step.output.token_ids == (3,)
    assert decode_step.output.token_ids == (8,)
    assert executor.batches[0].request_ids == (decoding.owner_id,)
    assert executor.batches[0].width == 1
    telemetry = lane.telemetry()
    assert telemetry.prefill_fallback_rows == 1
    assert telemetry.singleton_decode_fallback_rows == 0
    assert telemetry.singleton_ragged_decode_rows == 1
    runtime.commit(prefill_step, (2,))
    runtime.commit(decode_step, (1,))
    runtime.release_state(fresh)
    runtime.release_state(decoding)
    runtime.close()


def test_build_stack_propagates_singleton_opt_in_and_rejects_non_boolean() -> None:
    executor = _RaggedExecutor()
    stack = build_qwen3_moe_continuous_service(
        _FakeEngine(),
        route=_route(),
        max_context_tokens=16,
        semantic_token_count=128,
        decode_executor=executor,
        max_batch_size=2,
        max_queue_delay_seconds=0,
        enable_singleton_ragged_decode=True,
    )
    try:
        assert stack.lane.identity.dispatches_singletons is True
        assert stack.lane.telemetry().singleton_ragged_decode_enabled is True
    finally:
        stack.close()

    runtime = _runtime(slots=2)
    with pytest.raises(TypeError, match="enable_singleton_ragged_decode must be boolean"):
        Qwen3MoeCompatibleBatchLane(
            runtime,
            executor,
            max_batch_size=2,
            enable_singleton_ragged_decode=1,  # type: ignore[arg-type]
        )
    runtime.close()

    with pytest.raises(TypeError, match="enable_singleton_ragged_decode must be boolean"):
        build_qwen3_moe_continuous_service(
            _FakeEngine(),
            route=_route(),
            max_context_tokens=16,
            semantic_token_count=128,
            decode_executor=executor,
            max_batch_size=2,
            enable_singleton_ragged_decode=1,  # type: ignore[arg-type]
        )


def test_native_service_dispatches_b1_decode_to_opt_in_ragged_lane() -> None:
    runtime = _runtime(slots=2)
    executor = _RaggedExecutor()
    lane = Qwen3MoeCompatibleBatchLane(
        runtime,
        executor,
        max_batch_size=2,
        max_queue_delay_seconds=0,
        enable_singleton_ragged_decode=True,
    )
    service = NativeGenerationService(
        runtime,
        max_context_tokens=16,
        semantic_token_count=128,
        max_active_requests=2,
        supported_output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,),
        max_new_tokens=4,
        compatible_batch_lane=lane,
        owns_runtime=True,
    )
    try:
        result = service.submit(
            GenerationRequest(
                request_id="singleton-service",
                input_ids=(1,),
                max_new_tokens=2,
                stream=False,
            )
        ).result(5)

        assert result.token_ids == (2, 3)
        assert len(executor.batches) == 1
        assert executor.batches[0].width == 1
        assert executor.batches[0].request_ids == ("singleton-service",)
        lane_telemetry = lane.telemetry()
        assert lane_telemetry.prefill_fallback_rows == 1
        assert lane_telemetry.singleton_ragged_decode_rows == 1
        service_telemetry = service.telemetry().compatible_batch
        assert service_telemetry is not None
        assert service_telemetry.identity.dispatches_singletons is True
        assert service_telemetry.singleton_bypasses == 0
    finally:
        service.close()


def test_native_service_default_keeps_b1_on_scalar_runtime() -> None:
    runtime = _runtime(slots=2)
    executor = _RaggedExecutor()
    lane = Qwen3MoeCompatibleBatchLane(
        runtime,
        executor,
        max_batch_size=2,
        max_queue_delay_seconds=0,
    )
    service = NativeGenerationService(
        runtime,
        max_context_tokens=16,
        semantic_token_count=128,
        max_active_requests=2,
        supported_output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,),
        max_new_tokens=4,
        compatible_batch_lane=lane,
        owns_runtime=True,
    )
    try:
        result = service.submit(
            GenerationRequest(
                request_id="singleton-default",
                input_ids=(1,),
                max_new_tokens=2,
                stream=False,
            )
        ).result(5)

        assert result.token_ids == (2, 3)
        assert executor.batches == []
        assert lane.telemetry().dispatches == 0
        service_telemetry = service.telemetry().compatible_batch
        assert service_telemetry is not None
        assert service_telemetry.identity.dispatches_singletons is False
        assert service_telemetry.singleton_bypasses == 2
    finally:
        service.close()


def test_ragged_lane_compacts_logical_rows_without_copying_prefixes() -> None:
    runtime = _runtime(slots=3)
    states = [
        runtime.allocate_state(owner_id=f"request-{index}", batch_size=1, capacity=8)
        for index in range(3)
    ]
    for state, tokens in zip(states, ((1,), (2, 3), (4, 5, 6)), strict=True):
        step = _prefill(runtime, state, *tokens)
        runtime.commit(step, (len(tokens),))

    runtime.release_state(states[1])
    replacement = runtime.allocate_state(owner_id="request-refill", batch_size=1, capacity=8)
    refill = _prefill(runtime, replacement, 7, 8)
    runtime.commit(refill, (2,))

    executor = _RaggedExecutor()
    lane = Qwen3MoeCompatibleBatchLane(
        runtime,
        executor,
        max_batch_size=3,
        max_queue_delay_seconds=0,
    )
    ordered = (states[2], states[0], replacement)
    works = tuple(
        _decode(runtime, state, token) for state, token in zip(ordered, (20, 21, 22), strict=True)
    )
    steps = lane.execute(works)

    assert executor.batches[0].physical_slots == (2, 0, 1)
    assert executor.batches[0].parent_lengths == (3, 1, 2)
    assert tuple(step.output.token_ids[0] for step in steps) == (21, 22, 23)
    for step in steps:
        runtime.commit(step, (1,))
    assert tuple(state.observe().lengths[0] for state in ordered) == (4, 2, 3)

    telemetry = lane.telemetry()
    assert telemetry.physical_decode_calls == 1
    assert telemetry.decoded_rows == 3
    assert telemetry.width_histogram == ((3, 1),)
    assert telemetry.logical_row_compactions == 3
    assert telemetry.noncontiguous_slot_dispatches == 1
    assert telemetry.refilled_rows == 1
    assert telemetry.aggregate_tokens_per_second is not None
    assert telemetry.per_stream_tokens_per_second is not None

    for state in ordered:
        runtime.release_state(state)
    runtime.close()


def test_native_service_refills_finished_slot_and_preserves_request_stop_state() -> None:
    runtime = _runtime(slots=3)
    executor = _RaggedExecutor(block_first=True)
    lane = Qwen3MoeCompatibleBatchLane(
        runtime,
        executor,
        max_batch_size=3,
        max_queue_delay_seconds=0,
    )
    service = NativeGenerationService(
        runtime,
        max_context_tokens=16,
        semantic_token_count=128,
        max_active_requests=3,
        supported_output_modes=(OutputMode.NEXT_TOKEN_ARGMAX,),
        max_new_tokens=8,
        compatible_batch_lane=lane,
        owns_runtime=True,
    )
    try:
        handle_a, handle_b, handle_c = service.submit_many(
            (
                GenerationRequest(
                    request_id="a",
                    input_ids=(0,),
                    max_new_tokens=4,
                    stop_sequences=((99,),),
                    stream=False,
                ),
                GenerationRequest(
                    request_id="b",
                    input_ids=(98,),
                    max_new_tokens=1,
                    stream=False,
                ),
                GenerationRequest(
                    request_id="c",
                    input_ids=(9,),
                    max_new_tokens=4,
                    stop_sequences=((99,),),
                    stream=False,
                ),
            )
        )
        assert executor.first_started.wait(5)
        result_b = handle_b.result(5)
        assert result_b.token_ids == (99,)
        assert result_b.finish_reason is FinishReason.MAX_NEW_TOKENS

        # B's physical row is already free while A/C remain in the blocked decode wave.
        handle_d = service.submit(
            GenerationRequest(
                request_id="d",
                input_ids=(6,),
                max_new_tokens=1,
                stream=False,
            )
        )
        executor.first_gate.set()

        result_a = handle_a.result(5)
        result_c = handle_c.result(5)
        result_d = handle_d.result(5)
        assert result_a.token_ids == (1, 2)
        assert result_c.token_ids == (10, 11)
        assert result_a.finish_reason is FinishReason.STOP_SEQUENCE
        assert result_c.finish_reason is FinishReason.STOP_SEQUENCE
        assert result_a.matched_stop_sequence == (99,)
        assert result_c.matched_stop_sequence == (99,)
        assert result_d.token_ids == (7,)

        lane_telemetry = lane.telemetry()
        assert lane_telemetry.physical_decode_calls == 2
        assert lane_telemetry.decoded_rows == 4
        assert lane_telemetry.width_histogram == ((2, 2),)
        assert lane_telemetry.prefill_fallback_rows == 4
        pool = runtime.pool.telemetry()
        assert pool.allocations == 4
        assert pool.releases == 4
        assert pool.slot_reuses == 1
        assert service.telemetry().reconciled
    finally:
        service.close()
