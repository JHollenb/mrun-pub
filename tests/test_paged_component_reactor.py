from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrun.compiler import (
    DenseWorkPlan,
    DispatchBinding,
    ExecutionMode,
    OutputContract,
    bind_versioned_kv_state,
    build_paged_qstore_plan,
    decompose_work_plan,
    lower_work_template,
)
from mrun.engine.continuous import (
    ContinuousAdmissionError,
    ContinuousBackpressureError,
    ContinuousDuplicateRequestError,
    ContinuousOutputError,
    ContinuousPagedRequest,
    ContinuousPagedService,
    ContinuousRequestDeadlineExceeded,
    ContinuousRouteUnavailable,
)
from mrun.engine.kernels import paged_forward as pf
from mrun.engine.paged import PagedEngine
from mrun.engine.paged_reactor import (
    PagedComponentBatchExecutor,
    PagedComponentReactor,
    PagedReactorPayload,
    PagedReactorResult,
)
from mrun.testing.qstore_identity import (
    install_verified_test_identity,
    verified_test_manifest,
)

_CONFIG = {
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


def _qrow(shape: list[int], offset: int) -> dict[str, object]:
    rows = int(shape[0])
    width = int(shape[1])
    return {
        "kind": "qrow",
        "shape": shape,
        "w_off": offset,
        "w_len": rows * width,
        "s_off": offset,
        "s_len": rows * 4,
    }


_MANIFEST = verified_test_manifest(
    {
        "model_name": "reactor-tiny",
        "dtype": "int8",
        "config": _CONFIG,
        "blocks": {
            "embed": _qrow([8, 4], 0),
            "L0.q": _qrow([4, 4], 64),
            "L0.k": _qrow([2, 4], 96),
            "L0.v": _qrow([2, 4], 112),
            "L0.o": _qrow([4, 4], 128),
            "L0.gate": _qrow([6, 4], 160),
            "L0.up": _qrow([6, 4], 208),
            "L0.down": _qrow([4, 6], 256),
            "L0.ln1": {"kind": "fp32", "shape": [4], "e_off": 0, "e_len": 16},
            "L0.ln2": {"kind": "fp32", "shape": [4], "e_off": 16, "e_len": 16},
            "norm.final": {
                "kind": "fp32",
                "shape": [4],
                "e_off": 32,
                "e_len": 16,
            },
            "lm_head": _qrow([8, 4], 320),
        },
    }
)


class _TinyStore:
    """Deterministic Qwen-shaped store with physical traversal telemetry."""

    compute_dtype = torch.float32

    def __init__(self) -> None:
        self.man = _MANIFEST
        self.cfg = self.man["config"]
        self.device = torch.device("cpu")
        self._cache_policy = "test-static-pin"
        self._cache_budget = 0
        self.max_block_bytes = 96
        generator = torch.Generator().manual_seed(311)
        hidden = int(_CONFIG["hidden_size"])
        intermediate = int(_CONFIG["intermediate_size"])
        vocab = int(_CONFIG["vocab_size"])
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
        install_verified_test_identity(self)

    def has(self, name: str) -> bool:
        return name in self.man["blocks"]

    def ring_allocated_bytes(self) -> int:
        return 0

    def embed_rows(self, name: str, ids: np.ndarray) -> torch.Tensor:
        self.calls.append(("embed_rows", name))
        rows = torch.as_tensor(np.array(ids, copy=True), dtype=torch.long)
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

    def row_blocks(
        self,
        name: str,
        bs: int = 3,
    ) -> Iterator[tuple[int, int, torch.Tensor]]:
        self.calls.append(("row_blocks", name))
        for start in range(0, int(self.head.shape[0]), bs):
            end = min(start + bs, int(self.head.shape[0]))
            yield start, end, self.head[start:end]


class _TinyEngine:
    backend = "paged"
    numerical_contract = "paged-qstore-established"
    last_head_numerical_contract = "paged-qstore-last-head-fp32-v1"
    subset_head_numerical_contract = "paged-qstore-subset-head-fp32-v1"
    supported_numerical_contracts = (
        numerical_contract,
        last_head_numerical_contract,
        subset_head_numerical_contract,
        "paged-qstore-test-alternate",
    )

    def __init__(self) -> None:
        self.store = _TinyStore()
        self.device = self.store.device
        self.name = "reactor-tiny"
        self.spec = SimpleNamespace(name=self.name)
        self.semantic_token_count = 8
        self.n_layer = 1
        self.hidden = 4
        self.inter = 6
        self.arch = "qwen2"
        self.composite_store = None
        self.component_output_contract = None
        self._execution_lock = threading.RLock()

    def capabilities(self) -> SimpleNamespace:
        return SimpleNamespace(transactional_kv=True, speculative_blocks=True)

    def assert_content_identity_unchanged(self) -> None:
        self.store.assert_content_identity_unchanged()

    def execute_workplan_stateful(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("reactor bypassed the pooled kernel")

    # Planning uses method presence to bind the real output-pushdown/numerical contract.  The
    # reactor executes the pooled kernel directly, so these methods must never be called here.
    def last_logits_batch(self, _rows: object) -> torch.Tensor:
        raise AssertionError("reactor bypassed the pooled kernel")

    def selected_last_logits_batch(self, _rows: object, _selected: object) -> torch.Tensor:
        raise AssertionError("reactor bypassed the pooled kernel")

    def candidate_logits_batch(self, _rows: object, _candidates: object) -> torch.Tensor:
        raise AssertionError("reactor bypassed the pooled kernel")

    def hidden_states_batch(self, _rows: object) -> torch.Tensor:
        raise AssertionError("reactor bypassed the pooled kernel")


def _continuous_tiny_engine() -> PagedEngine:
    """Install the tiny deterministic store on the exact trusted PagedEngine class."""

    source = _TinyEngine()
    engine = object.__new__(PagedEngine)
    engine.__dict__.update(source.__dict__)
    engine.cfg = engine.store.cfg
    engine._closed = False
    return engine


def _new_cache(
    *,
    length: int,
    seed: int,
    capacity: int = 12,
) -> pf.BatchedPagedKVCache:
    cache = pf.BatchedPagedKVCache(1, 1, 1, 2, capacity=capacity, device="cpu")
    cache.lengths = np.asarray([length], dtype=np.int64)
    cache.epoch = seed
    generator = torch.Generator().manual_seed(seed)
    cache.k[:, 0, :length] = torch.randn(1, length, 1, 2, generator=generator)
    cache.v[:, 0, :length] = torch.randn(1, length, 1, 2, generator=generator)
    return cache


def _equivalent_cache(
    caches: Sequence[pf.BatchedPagedKVCache],
) -> pf.BatchedPagedKVCache:
    combined = pf.BatchedPagedKVCache(
        1,
        len(caches),
        1,
        2,
        capacity=caches[0].capacity,
        device="cpu",
    )
    combined.lengths = np.asarray([int(cache.lengths[0]) for cache in caches], dtype=np.int64)
    for row, cache in enumerate(caches):
        length = int(cache.lengths[0])
        combined.k[:, row, :length] = cache.k[:, 0, :length]
        combined.v[:, row, :length] = cache.v[:, 0, :length]
    return combined


def _snapshot(
    cache: pf.BatchedPagedKVCache,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
    return cache.k.clone(), cache.v.clone(), cache.lengths.copy(), int(cache.epoch)


def _assert_snapshot(
    cache: pf.BatchedPagedKVCache,
    snapshot: tuple[torch.Tensor, torch.Tensor, np.ndarray, int],
) -> None:
    keys, values, lengths, epoch = snapshot
    assert torch.equal(cache.k, keys)
    assert torch.equal(cache.v, values)
    assert np.array_equal(cache.lengths, lengths)
    assert cache.epoch == epoch


@dataclass(frozen=True)
class _Submission:
    plan: DenseWorkPlan
    template: object
    binding: DispatchBinding
    lowered: object
    payload: PagedReactorPayload
    cache: pf.BatchedPagedKVCache
    lease: pf.PagedKVSlotLease


def _submission(
    engine: _TinyEngine,
    index: int,
    *,
    output_contract: OutputContract = OutputContract.SELECTED_TOKEN_ROWS,
    selected_rows: tuple[int, ...] = (1, 3),
    candidates: tuple[int, ...] = (1, 3, 5),
    tokens: tuple[int, ...] = (2, 4),
    cache: pf.BatchedPagedKVCache | None = None,
    dispatch_metadata: dict[str, object] | None = None,
    numerical_contract: str | None = None,
) -> _Submission:
    actual_cache = cache or _new_cache(length=1 + index % 3, seed=41 + index)
    handle = f"kv:request-{index}"
    plan = build_paged_qstore_plan(
        engine,
        [np.asarray(tokens, dtype=np.int64)],
        execution_mode=ExecutionMode.DECODE,
        output_contract=output_contract,
        request_ids=(f"request-{index}",),
        request_slots=(0,),
        kv_read_handles=(handle,),
        kv_write_handles=(handle,),
        kv_capacity=actual_cache.capacity,
        required_output_rows=(
            selected_rows if output_contract is OutputContract.SELECTED_TOKEN_ROWS else ()
        ),
        candidate_token_ids=(
            (candidates,) if output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN else ()
        ),
    )
    if dispatch_metadata:
        plan = replace(
            plan,
            metadata=tuple((*plan.metadata, *dispatch_metadata.items())),
        )
    if numerical_contract is not None:
        plan = replace(plan, numerical_contract=numerical_contract)
    template, binding = decompose_work_plan(plan)
    lowered = lower_work_template(template, "paged-qstore")
    state_binding = bind_versioned_kv_state(actual_cache, plan.kv_read_handles)
    lease = actual_cache.mint_slot_lease()
    payload = PagedReactorPayload(
        engine=engine,
        ids=np.asarray(tokens, dtype=np.int64),
        state_binding=state_binding,
        slot_lease=lease,
    )
    return _Submission(plan, template, binding, lowered, payload, actual_cache, lease)


def _submit(reactor: PagedComponentReactor, item: _Submission):
    return reactor.submit(
        item.template,
        item.binding,
        item.payload,
        lowered_template=item.lowered,
    )


def test_continuous_route_rejects_generic_capability_lookalikes() -> None:
    with pytest.raises(ContinuousRouteUnavailable, match="exact"):
        ContinuousPagedService(
            _TinyEngine(),
            max_active_requests=1,
            max_batch_size=1,
        )


def test_continuous_request_id_has_a_bounded_utf8_representation() -> None:
    engine = _continuous_tiny_engine()
    with pytest.raises(ValueError, match="256 UTF-8 bytes"):
        ContinuousPagedRequest(
            request_id="é" * 129,
            engine=engine,
            input_ids=(1,),
            max_new_tokens=1,
        )


def test_continuous_service_runs_autonomous_prefill_refill_and_releases_lease() -> None:
    engine = _continuous_tiny_engine()
    cache = _new_cache(length=0, seed=501, capacity=8)
    initial_epoch = cache.epoch
    lease = cache.mint_slot_lease()
    request = ContinuousPagedRequest(
        request_id="autonomous-prefill",
        engine=engine,
        input_ids=(1, 2),
        max_new_tokens=3,
        cache=cache,
        slot_lease=lease,
        batching_policy="adaptive",
    )

    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=60,
    ) as service:
        result = service.submit(request).result(timeout=3)
        assert service.wait_idle(timeout=1)
        telemetry = service.telemetry()

    assert len(result.generated_token_ids) == 3
    assert result.pending_token_id == result.generated_token_ids[-1]
    assert result.committed_input_count == 4
    assert result.starting_cache_length == 0
    assert result.final_cache_length == 4
    assert cache.lengths.tolist() == [4]
    assert result.final_cache_epoch == cache.epoch == initial_epoch + 3
    assert result.step_count == 3
    with pytest.raises(RuntimeError, match="released"):
        cache.validate_slot_lease(lease)
    assert telemetry.admitted == telemetry.completed == 1
    assert telemetry.steps_submitted == telemetry.steps_succeeded == 3
    assert telemetry.generated_tokens == 3
    assert telemetry.refills == 2
    assert telemetry.leases_released == 1
    # Width one is already a full reactor batch here, so the causal dispatch trigger is
    # ``full`` even though the service requested latency bypass.
    assert telemetry.reactor.bypass_submitted == telemetry.reactor.bypass_dispatched == 3
    assert telemetry.reactor.bypass_singleton_batches == 0
    assert telemetry.reconciled


def test_continuous_adaptive_singleton_bypasses_delay_without_disabling_next_burst() -> None:
    engine = _continuous_tiny_engine()
    with ContinuousPagedService(
        engine,
        max_active_requests=2,
        max_batch_size=2,
        max_batch_delay_seconds=60,
    ) as service:
        singleton = service.submit(
            ContinuousPagedRequest(
                request_id="adaptive-singleton",
                engine=engine,
                input_ids=(1,),
                max_new_tokens=1,
                kv_capacity=1,
            )
        ).result(timeout=2)
        burst = service.submit_many(
            tuple(
                ContinuousPagedRequest(
                    request_id=f"adaptive-burst-{index}",
                    engine=engine,
                    input_ids=(index + 2,),
                    max_new_tokens=1,
                    kv_capacity=1,
                )
                for index in range(2)
            )
        )
        burst_results = tuple(future.result(timeout=2) for future in burst)
        telemetry = service.telemetry()

    assert len(singleton.generated_token_ids) == 1
    assert all(len(result.generated_token_ids) == 1 for result in burst_results)
    assert telemetry.reactor.batch_width_histogram == ((1, 1), (2, 1))
    assert telemetry.reactor.bypass_batches == 1
    assert telemetry.reactor.bypass_singleton_batches == 1
    assert telemetry.reactor.coalesced_dispatches == 1
    assert telemetry.reconciled


def _run_continuous_decode_leg(
    *,
    max_batch_size: int,
) -> tuple[
    tuple[object, ...],
    tuple[pf.BatchedPagedKVCache, ...],
    object,
]:
    engine = _continuous_tiny_engine()
    limits = (1, 2, 3, 4)
    caches = tuple(
        _new_cache(length=1 + index % 3, seed=700 + index, capacity=12)
        for index in range(len(limits))
    )
    requests = tuple(
        ContinuousPagedRequest(
            request_id=f"ragged-{index}",
            engine=engine,
            input_ids=((index + 2) % 8,),
            max_new_tokens=limit,
            cache=cache,
            slot_lease=cache.mint_slot_lease(),
            batching_policy="adaptive",
        )
        for index, (limit, cache) in enumerate(zip(limits, caches, strict=True))
    )
    with ContinuousPagedService(
        engine,
        max_active_requests=4,
        max_batch_size=max_batch_size,
        max_batch_delay_seconds=0.1,
    ) as service:
        futures = service.submit_many(requests)
        results = tuple(future.result(timeout=5) for future in futures)
        telemetry = service.telemetry()
    return results, caches, telemetry


def test_continuous_ragged_refill_is_exact_between_serial_and_pooled_services() -> None:
    serial_results, serial_caches, serial_telemetry = _run_continuous_decode_leg(max_batch_size=1)
    pooled_results, pooled_caches, pooled_telemetry = _run_continuous_decode_leg(max_batch_size=4)

    assert tuple(result.generated_token_ids for result in pooled_results) == tuple(
        result.generated_token_ids for result in serial_results
    )
    assert tuple(result.pending_token_id for result in pooled_results) == tuple(
        result.pending_token_id for result in serial_results
    )
    for serial, pooled in zip(serial_caches, pooled_caches, strict=True):
        assert torch.equal(serial.k, pooled.k)
        assert torch.equal(serial.v, pooled.v)
        assert np.array_equal(serial.lengths, pooled.lengths)
        assert serial.epoch == pooled.epoch
    assert serial_telemetry.generated_tokens == pooled_telemetry.generated_tokens == 10
    assert serial_telemetry.reactor.batch_width_max == 1
    assert pooled_telemetry.reactor.batch_width_max == 4
    assert pooled_telemetry.reactor.coalesced_dispatches > 0
    assert serial_telemetry.reconciled and pooled_telemetry.reconciled


def test_continuous_accepts_canonical_cpu_cache_for_explicit_cpu_zero_engine() -> None:
    engine = _continuous_tiny_engine()
    engine.device = torch.device("cpu:0")
    cache = _new_cache(length=0, seed=777, capacity=2)
    assert cache.k.device == torch.device("cpu")
    assert cache.k.device != engine.device
    lease = cache.mint_slot_lease()

    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
    ) as service:
        result = service.submit(
            ContinuousPagedRequest(
                request_id="canonical-cpu-device",
                engine=engine,
                input_ids=(1,),
                max_new_tokens=1,
                cache=cache,
                slot_lease=lease,
            )
        ).result(timeout=2)
        telemetry = service.telemetry()

    assert result.generated_token_ids
    assert result.committed_input_count == 1
    assert result.final_cache_length == 1
    assert result.final_cache_epoch == 778
    assert int(cache.lengths[0]) == 1
    assert telemetry.completed == telemetry.generated_tokens == telemetry.leases_released == 1
    assert telemetry.reconciled


def test_continuous_arrival_during_execution_joins_a_later_refill_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _continuous_tiny_engine()
    original = pf.paged_forward_block_pooled
    first_started = threading.Event()
    release_first = threading.Event()
    call_lock = threading.Lock()
    calls = 0

    def first_blocked_kernel(*args: object, **kwargs: object):
        nonlocal calls
        with call_lock:
            calls += 1
            call = calls
        if call == 1:
            first_started.set()
            assert release_first.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(pf, "paged_forward_block_pooled", first_blocked_kernel)
    caches = tuple(_new_cache(length=2, seed=760 + index, capacity=10) for index in range(3))
    requests = tuple(
        ContinuousPagedRequest(
            request_id=f"live-{index}",
            engine=engine,
            input_ids=((index + 1) % 8,),
            max_new_tokens=limit,
            cache=cache,
            slot_lease=cache.mint_slot_lease(),
        )
        for index, (limit, cache) in enumerate(zip((1, 3, 2), caches, strict=True))
    )
    service = ContinuousPagedService(
        engine,
        max_active_requests=3,
        max_batch_size=2,
        max_batch_delay_seconds=0.5,
    )
    try:
        first_futures = service.submit_many(requests[:2])
        assert first_started.wait(1)
        late_future = service.submit(requests[2])
        release_first.set()
        results = tuple(future.result(timeout=4) for future in (*first_futures, late_future))
        telemetry = service.telemetry()
    finally:
        release_first.set()
        service.shutdown(wait=True, cancel_pending=True)

    assert tuple(len(result.generated_token_ids) for result in results) == (1, 3, 2)
    assert telemetry.generated_tokens == 6
    assert telemetry.reactor.batch_width_histogram == ((2, 3),)
    assert telemetry.reactor.coalesced_dispatches == 3
    assert telemetry.reconciled


def test_continuous_drain_shutdown_keeps_internal_refill_open() -> None:
    engine = _continuous_tiny_engine()
    service = ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=60,
    )
    future = service.submit(
        ContinuousPagedRequest(
            request_id="drain-refill",
            engine=engine,
            input_ids=(1,),
            max_new_tokens=3,
            kv_capacity=4,
        )
    )

    assert service.shutdown(wait=True, timeout=3)
    result = future.result(timeout=1)
    telemetry = service.telemetry()

    assert len(result.generated_token_ids) == 3
    assert telemetry.completed == 1
    assert telemetry.steps_succeeded == 3
    assert telemetry.stopped
    assert not telemetry.accepting
    assert telemetry.reconciled


def test_continuous_drain_flushes_prefer_batch_singleton_without_waiting_delay() -> None:
    engine = _continuous_tiny_engine()
    service = ContinuousPagedService(
        engine,
        max_active_requests=2,
        max_batch_size=2,
        max_batch_delay_seconds=60,
    )
    future = service.submit(
        ContinuousPagedRequest(
            request_id="drain-prefer-batch",
            engine=engine,
            input_ids=(1,),
            max_new_tokens=1,
            kv_capacity=1,
            batching_policy="prefer_batch",
        )
    )

    wait_deadline = time.monotonic() + 1
    while service.telemetry().reactor.queued != 1 and time.monotonic() < wait_deadline:
        time.sleep(0.001)
    assert service.telemetry().reactor.queued == 1

    assert service.shutdown(wait=True, timeout=2)
    assert len(future.result(timeout=1).generated_token_ids) == 1
    telemetry = service.telemetry()
    assert telemetry.completed == telemetry.steps_succeeded == 1
    assert telemetry.reactor.bypass_batches == telemetry.reactor.bypass_singleton_batches == 0
    assert telemetry.reactor.recent_batches[-1].dispatch_trigger == "explicit_flush"
    assert telemetry.reconciled


def test_continuous_completed_request_id_cannot_be_reused_and_rejection_keeps_lease() -> None:
    engine = _continuous_tiny_engine()
    second_cache = _new_cache(length=1, seed=790, capacity=5)
    second_lease = second_cache.mint_slot_lease()
    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
    ) as service:
        service.submit(
            ContinuousPagedRequest(
                request_id="never-reuse",
                engine=engine,
                input_ids=(1,),
                max_new_tokens=1,
                kv_capacity=2,
            )
        ).result(timeout=2)
        with pytest.raises(ContinuousDuplicateRequestError):
            service.submit(
                ContinuousPagedRequest(
                    request_id="never-reuse",
                    engine=engine,
                    input_ids=(2,),
                    max_new_tokens=1,
                    cache=second_cache,
                    slot_lease=second_lease,
                )
            )
        telemetry = service.telemetry()

    second_cache.validate_slot_lease(second_lease)
    second_cache.release_slot_lease(second_lease)
    assert telemetry.admitted == telemetry.completed == 1
    assert telemetry.rejected_duplicate == 1
    assert telemetry.reconciled


def test_continuous_request_history_is_bounded_and_fails_closed_when_full() -> None:
    engine = _continuous_tiny_engine()
    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
        max_request_history=1,
    ) as service:
        service.submit(
            ContinuousPagedRequest(
                request_id="history-first",
                engine=engine,
                input_ids=(1,),
                max_new_tokens=1,
                kv_capacity=1,
            )
        ).result(timeout=2)
        with pytest.raises(ContinuousBackpressureError, match="request-history"):
            service.submit(
                ContinuousPagedRequest(
                    request_id="history-second",
                    engine=engine,
                    input_ids=(2,),
                    max_new_tokens=1,
                    kv_capacity=1,
                )
            )
        telemetry = service.telemetry()

    assert telemetry.request_history_size == telemetry.request_history_capacity == 1
    assert telemetry.rejected_request_history == telemetry.rejected_backpressure == 1
    assert telemetry.reconciled


def test_continuous_source_template_cache_is_bounded_lru() -> None:
    engine = _continuous_tiny_engine()
    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
        max_aggregate_templates=1,
    ) as service:
        for index, input_ids in enumerate(((1,), (1, 2))):
            service.submit(
                ContinuousPagedRequest(
                    request_id=f"template-lru-{index}",
                    engine=engine,
                    input_ids=input_ids,
                    max_new_tokens=1,
                    kv_capacity=len(input_ids),
                )
            ).result(timeout=2)
        telemetry = service.telemetry()

    assert telemetry.template_cache_size == telemetry.template_cache_capacity == 1
    assert telemetry.template_cache_evictions >= 1
    assert telemetry.reconciled


def test_continuous_executing_cancel_discards_scratch_before_releasing_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _continuous_tiny_engine()
    cache = _new_cache(length=2, seed=810, capacity=8)
    lease = cache.mint_slot_lease()
    snapshot = _snapshot(cache)
    started = threading.Event()
    release = threading.Event()
    original = pf.paged_forward_block_pooled

    def blocking_kernel(*args: object, **kwargs: object):
        started.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(pf, "paged_forward_block_pooled", blocking_kernel)
    service = ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
    )
    try:
        future = service.submit(
            ContinuousPagedRequest(
                request_id="cancel-inflight",
                engine=engine,
                input_ids=(3,),
                max_new_tokens=2,
                cache=cache,
                slot_lease=lease,
            )
        )
        assert started.wait(1)
        assert future.cancel()
        cache.validate_slot_lease(lease)
        release.set()
        assert service.wait_idle(timeout=2)
        with pytest.raises(CancelledError):
            future.result()
        telemetry = service.telemetry()
    finally:
        release.set()
        service.shutdown(wait=True, cancel_pending=True)

    _assert_snapshot(cache, snapshot)
    with pytest.raises(RuntimeError, match="released"):
        cache.validate_slot_lease(lease)
    assert telemetry.cancelled == 1
    assert telemetry.steps_succeeded == 0
    assert telemetry.steps_abandoned == 1
    assert telemetry.steps_cancelled == telemetry.steps_failed == telemetry.steps_active == 0
    assert telemetry.generated_tokens == 0
    assert telemetry.leases_released == 1
    assert telemetry.reconciled


@pytest.mark.parametrize(
    ("first_intent", "expected_cancelled", "expected_shutdown"),
    [
        ("user", 1, 0),
        ("shutdown", 0, 1),
    ],
)
def test_continuous_cancel_and_abort_preserve_first_terminal_intent(
    monkeypatch: pytest.MonkeyPatch,
    first_intent: str,
    expected_cancelled: int,
    expected_shutdown: int,
) -> None:
    engine = _continuous_tiny_engine()
    started = threading.Event()
    release = threading.Event()
    original = pf.paged_forward_block_pooled

    def blocking_kernel(*args: object, **kwargs: object):
        started.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(pf, "paged_forward_block_pooled", blocking_kernel)
    service = ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
    )
    future = service.submit(
        ContinuousPagedRequest(
            request_id=f"cancel-abort-{first_intent}",
            engine=engine,
            input_ids=(3,),
            max_new_tokens=1,
            kv_capacity=1,
        )
    )
    try:
        assert started.wait(1)
        if first_intent == "user":
            assert future.cancel()
            assert not service.shutdown(wait=False, cancel_pending=True)
        else:
            assert not service.shutdown(wait=False, cancel_pending=True)
            assert future.cancel()
        release.set()
        assert service.shutdown(wait=True, cancel_pending=True, timeout=2)
        with pytest.raises(CancelledError):
            future.result()
        telemetry = service.telemetry()
    finally:
        release.set()
        service.shutdown(wait=True, cancel_pending=True)

    assert telemetry.cancelled == expected_cancelled
    assert telemetry.shutdown_terminated == expected_shutdown
    assert telemetry.steps_abandoned == 1
    assert telemetry.steps_cancelled == telemetry.steps_failed == telemetry.steps_active == 0
    assert telemetry.reconciled


def test_continuous_queued_cancel_never_reaches_components_or_mutates_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _continuous_tiny_engine()
    caches = (
        _new_cache(length=2, seed=811, capacity=8),
        _new_cache(length=2, seed=812, capacity=8),
    )
    leases = tuple(cache.mint_slot_lease() for cache in caches)
    queued_snapshot = _snapshot(caches[1])
    started = threading.Event()
    release = threading.Event()
    original = pf.paged_forward_block_pooled
    calls = 0

    def blocking_first_kernel(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(pf, "paged_forward_block_pooled", blocking_first_kernel)
    service = ContinuousPagedService(
        engine,
        max_active_requests=2,
        max_batch_size=1,
        max_batch_delay_seconds=0,
    )
    try:
        futures = service.submit_many(
            tuple(
                ContinuousPagedRequest(
                    request_id=f"queued-cancel-{index}",
                    engine=engine,
                    input_ids=(index + 1,),
                    max_new_tokens=1,
                    cache=cache,
                    slot_lease=lease,
                )
                for index, (cache, lease) in enumerate(zip(caches, leases, strict=True))
            )
        )
        assert started.wait(1)
        assert futures[1].cancel()
        with pytest.raises(CancelledError):
            futures[1].result()
        with pytest.raises(RuntimeError, match="released"):
            caches[1].validate_slot_lease(leases[1])
        release.set()
        assert len(futures[0].result(timeout=2).generated_token_ids) == 1
        telemetry = service.telemetry()
    finally:
        release.set()
        service.shutdown(wait=True, cancel_pending=True)

    assert calls == 1
    _assert_snapshot(caches[1], queued_snapshot)
    assert telemetry.completed == telemetry.cancelled == 1
    assert telemetry.steps_submitted == 2
    assert telemetry.steps_succeeded == 1
    assert telemetry.steps_cancelled == 1
    assert telemetry.steps_abandoned == telemetry.steps_failed == telemetry.steps_active == 0
    assert telemetry.generated_tokens == 1
    assert telemetry.reactor.cancelled == 1
    assert telemetry.reconciled


def test_continuous_completion_deadline_discards_finished_scratch_without_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ManualClock:
        value = 10.0

        def __call__(self) -> float:
            return self.value

    clock = ManualClock()
    engine = _continuous_tiny_engine()
    cache = _new_cache(length=2, seed=820, capacity=8)
    lease = cache.mint_slot_lease()
    snapshot = _snapshot(cache)
    original = pf.paged_forward_block_pooled

    def expiring_kernel(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        clock.value = 20.0
        return result

    monkeypatch.setattr(pf, "paged_forward_block_pooled", expiring_kernel)
    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
        clock=clock,
    ) as service:
        future = service.submit(
            ContinuousPagedRequest(
                request_id="deadline-inflight",
                engine=engine,
                input_ids=(3,),
                max_new_tokens=1,
                cache=cache,
                slot_lease=lease,
                deadline=15.0,
            )
        )
        with pytest.raises(ContinuousRequestDeadlineExceeded):
            future.result(timeout=2)
        telemetry = service.telemetry()

    _assert_snapshot(cache, snapshot)
    assert telemetry.deadline_expired == 1
    assert telemetry.steps_succeeded == telemetry.generated_tokens == 0
    assert telemetry.steps_failed == 1
    assert telemetry.steps_cancelled == telemetry.steps_abandoned == telemetry.steps_active == 0
    assert telemetry.reconciled


def test_continuous_deadline_expiring_after_selection_but_before_commit_abandons_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ManualClock:
        value = 10.0

        def __call__(self) -> float:
            return self.value

    clock = ManualClock()
    engine = _continuous_tiny_engine()
    cache = _new_cache(length=2, seed=821, capacity=8)
    lease = cache.mint_slot_lease()
    snapshot = _snapshot(cache)
    original_select = ContinuousPagedService._validate_and_select_token

    def select_then_expire(
        service: ContinuousPagedService,
        result: object,
    ) -> int:
        token = original_select(service, result)  # type: ignore[arg-type]
        clock.value = 20.0
        return token

    monkeypatch.setattr(
        ContinuousPagedService,
        "_validate_and_select_token",
        select_then_expire,
    )
    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
        clock=clock,
    ) as service:
        future = service.submit(
            ContinuousPagedRequest(
                request_id="deadline-precommit",
                engine=engine,
                input_ids=(3,),
                max_new_tokens=1,
                cache=cache,
                slot_lease=lease,
                deadline=15.0,
            )
        )
        with pytest.raises(ContinuousRequestDeadlineExceeded):
            future.result(timeout=2)
        telemetry = service.telemetry()

    _assert_snapshot(cache, snapshot)
    assert telemetry.deadline_expired == 1
    assert telemetry.steps_abandoned == 1
    assert telemetry.steps_succeeded == telemetry.steps_failed == 0
    assert telemetry.reconciled


def test_continuous_deadline_crossing_during_commit_rolls_back_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ManualClock:
        value = 10.0

        def __call__(self) -> float:
            return self.value

    clock = ManualClock()
    engine = _continuous_tiny_engine()
    cache = _new_cache(length=2, seed=822, capacity=8)
    lease = cache.mint_slot_lease()
    snapshot = _snapshot(cache)
    original_commit = PagedReactorResult.commit

    def commit_then_expire(result: PagedReactorResult, accepted_count: int) -> tuple[int, ...]:
        committed = original_commit(result, accepted_count)
        clock.value = 20.0
        return committed

    monkeypatch.setattr(PagedReactorResult, "commit", commit_then_expire)
    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
        clock=clock,
    ) as service:
        future = service.submit(
            ContinuousPagedRequest(
                request_id="deadline-during-commit",
                engine=engine,
                input_ids=(3,),
                max_new_tokens=1,
                cache=cache,
                slot_lease=lease,
                deadline=15.0,
            )
        )
        with pytest.raises(ContinuousRequestDeadlineExceeded):
            future.result(timeout=2)
        telemetry = service.telemetry()

    _assert_snapshot(cache, snapshot)
    assert telemetry.deadline_expired == 1
    assert telemetry.generated_tokens == telemetry.steps_succeeded == 0
    assert telemetry.steps_abandoned == 1
    assert telemetry.steps_cancelled == telemetry.steps_failed == telemetry.steps_active == 0
    assert telemetry.leases_released == 1
    assert telemetry.reconciled


def test_continuous_detects_external_kv_mutation_between_refills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _continuous_tiny_engine()
    cache = _new_cache(length=1, seed=822, capacity=8)
    lease = cache.mint_slot_lease()
    original_prepare = ContinuousPagedService._prepare_step_submission
    calls = 0

    def mutate_before_refill(
        service: ContinuousPagedService,
        session: object,
    ):
        nonlocal calls
        calls += 1
        if calls == 2:
            with cache._lock:  # noqa: SLF001 - adversarial external mutation
                cache.lengths[0] += 1
        return original_prepare(service, session)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ContinuousPagedService,
        "_prepare_step_submission",
        mutate_before_refill,
    )
    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
    ) as service:
        future = service.submit(
            ContinuousPagedRequest(
                request_id="mutated-between-refills",
                engine=engine,
                input_ids=(3,),
                max_new_tokens=2,
                cache=cache,
                slot_lease=lease,
            )
        )
        with pytest.raises(RuntimeError, match="outside the service"):
            future.result(timeout=2)
        telemetry = service.telemetry()

    assert calls == 2
    assert telemetry.failed == 1
    assert telemetry.steps_submitted == telemetry.steps_succeeded == 1
    assert telemetry.generated_tokens == 1
    assert telemetry.reconciled


def test_continuous_nonfinite_logits_fail_before_argmax_or_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _continuous_tiny_engine()
    cache = _new_cache(length=1, seed=830, capacity=8)
    lease = cache.mint_slot_lease()
    snapshot = _snapshot(cache)
    original = pf.paged_forward_block_pooled

    def nan_kernel(*args: object, **kwargs: object):
        output, deltas = original(*args, **kwargs)
        return torch.full_like(output, float("nan")), deltas

    monkeypatch.setattr(pf, "paged_forward_block_pooled", nan_kernel)
    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
    ) as service:
        future = service.submit(
            ContinuousPagedRequest(
                request_id="nan-output",
                engine=engine,
                input_ids=(2,),
                max_new_tokens=1,
                cache=cache,
                slot_lease=lease,
            )
        )
        with pytest.raises(ContinuousOutputError, match="non-finite"):
            future.result(timeout=2)
        telemetry = service.telemetry()

    _assert_snapshot(cache, snapshot)
    assert telemetry.failed == 1
    assert telemetry.steps_succeeded == telemetry.generated_tokens == 0
    assert telemetry.reconciled


def test_continuous_admission_rejects_insufficient_capacity_without_taking_lease() -> None:
    engine = _continuous_tiny_engine()
    cache = _new_cache(length=3, seed=840, capacity=5)
    lease = cache.mint_slot_lease()
    snapshot = _snapshot(cache)

    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
    ) as service:
        with pytest.raises(ContinuousAdmissionError, match="needs KV capacity"):
            service.submit(
                ContinuousPagedRequest(
                    request_id="too-large",
                    engine=engine,
                    input_ids=(1, 2),
                    max_new_tokens=2,
                    cache=cache,
                    slot_lease=lease,
                )
            )
        telemetry = service.telemetry()

    cache.validate_slot_lease(lease)
    cache.release_slot_lease(lease)
    _assert_snapshot(cache, snapshot)
    assert telemetry.admitted == 0
    assert telemetry.rejected_admission == 1


def test_continuous_latency_distributions_use_one_deterministic_bounded_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ManualClock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

    clock = ManualClock()
    engine = _continuous_tiny_engine()
    original = pf.paged_forward_block_pooled

    def one_second_kernel(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        clock.value += 1.0
        return result

    monkeypatch.setattr(pf, "paged_forward_block_pooled", one_second_kernel)
    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=60,
        telemetry_history=2,
        clock=clock,
    ) as service:
        result = service.submit(
            ContinuousPagedRequest(
                request_id="clocked",
                engine=engine,
                input_ids=(1,),
                max_new_tokens=3,
                kv_capacity=4,
            )
        ).result(timeout=2)
        telemetry = service.telemetry()

    assert result.ttft_seconds == 1.0
    assert result.inter_token_seconds == (1.0, 1.0)
    assert result.request_latency_seconds == 3.0
    assert telemetry.ttft.observation_count == telemetry.ttft.window_count == 1
    assert telemetry.ttft.p50 == telemetry.ttft.p95 == telemetry.ttft.p99 == 1.0
    assert telemetry.inter_token.observation_count == telemetry.inter_token.window_count == 2
    assert telemetry.inter_token.mean == telemetry.inter_token.p50 == 1.0
    assert telemetry.step_latency.observation_count == 3
    assert telemetry.step_latency.window_count == 2
    assert telemetry.step_latency.minimum == telemetry.step_latency.maximum == 1.0
    assert telemetry.request_latency.p99 == 3.0
    assert telemetry.peak_active_committed_kv_arena_bytes == 2 * 1 * 4 * 1 * 2 * 4
    assert telemetry.active_committed_kv_arena_bytes == 0
    assert telemetry.reconciled


def test_continuous_result_reuses_exact_first_ttft_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ManualClock:
        value = 1_000_000.0

        def __call__(self) -> float:
            return self.value

    clock = ManualClock()
    engine = _continuous_tiny_engine()
    original = pf.paged_forward_block_pooled
    emitted_times = iter((1_000_000.000001, 101_000_000.0, 101_000_000.000001))

    def nonuniform_kernel(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        clock.value = next(emitted_times)
        return result

    monkeypatch.setattr(pf, "paged_forward_block_pooled", nonuniform_kernel)
    with ContinuousPagedService(
        engine,
        max_active_requests=1,
        max_batch_size=1,
        max_batch_delay_seconds=0,
        clock=clock,
    ) as service:
        result = service.submit(
            ContinuousPagedRequest(
                request_id="exact-first-ttft",
                engine=engine,
                input_ids=(1,),
                max_new_tokens=3,
                kv_capacity=3,
            )
        ).result(timeout=2)
        telemetry = service.telemetry()

    assert result.ttft_seconds == telemetry.ttft.minimum
    assert result.ttft_seconds == telemetry.ttft.maximum
    assert result.ttft_seconds == telemetry.ttft.mean
    assert result.ttft_seconds == telemetry.ttft.p50
    assert result.ttft_seconds == telemetry.ttft.p95
    assert result.ttft_seconds == telemetry.ttft.p99
    assert result.ttft_seconds == 1.00000761449337e-06
    assert telemetry.reconciled


def test_eight_b1_dispatches_coalesce_into_one_exact_transactional_wave() -> None:
    engine = _TinyEngine()
    row_demands = (
        (6, 1),
        (1, 4),
        (4, 2),
        (2, 7),
        (7, 0),
        (0, 3),
        (3, 5),
        (5, 6),
    )
    items = tuple(
        _submission(
            engine,
            index,
            selected_rows=rows,
            tokens=((index + 1) % 8, (index + 3) % 8),
            dispatch_metadata={"request_trace_id": f"trace-{index}", "owner": "reactor-test"},
        )
        for index, rows in enumerate(row_demands)
    )
    assert len({item.template.fingerprint for item in items}) == 1
    assert all(
        DispatchBinding.from_json(item.binding.to_json()).dispatch_metadata
        == item.binding.dispatch_metadata
        for item in items
    )
    snapshots = tuple(_snapshot(item.cache) for item in items)
    stable_union = tuple(dict.fromkeys(token for rows in row_demands for token in rows))
    combined = _equivalent_cache([item.cache for item in items])
    token_matrix = np.stack([item.payload.ids for item in items])
    reference, combined_delta = pf.paged_forward_block(
        _TinyStore(),
        token_matrix,
        combined,
        output_contract="selected_token_rows",
        selected_rows=stable_union,
        last_only=True,
    )

    with PagedComponentReactor(
        max_batch_size=8,
        max_pending=8,
        max_batch_delay_seconds=0.1,
    ) as reactor:
        futures = tuple(_submit(reactor, item) for item in items)
        results = tuple(future.result(timeout=3) for future in futures)
        telemetry = reactor.telemetry()
        assert reactor.paged_executor.aggregate_template_cache_size == 1

    assert telemetry.batches == 1
    assert telemetry.batch_width_histogram == ((8, 1),)
    assert telemetry.coalesced_dispatches == 7
    for name in ("L0.q", "L0.k", "L0.v", "L0.o", "L0.gate", "L0.up", "L0.down"):
        assert engine.store.calls.count(("matmul", name)) == 1
    assert engine.store.calls.count(("embed_rows", "lm_head")) == 1

    offsets = {token: offset for offset, token in enumerate(stable_union)}
    for index, (item, rows, result, snapshot) in enumerate(
        zip(items, row_demands, results, snapshots, strict=True)
    ):
        columns = torch.as_tensor([offsets[token] for token in rows])
        expected = reference[index : index + 1].index_select(1, columns)
        assert result.outputs.shape == (1, 2)
        torch.testing.assert_close(result.outputs, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            result.provisional_delta.delta.k,
            combined_delta.k[:, index : index + 1],
            rtol=0,
            atol=0,
        )
        assert result.plan_fingerprint == item.plan.fingerprint
        assert result.evidence["stable_output_union"] == list(stable_union)
        assert result.evidence["aggregate_actual_batch"] == 8
        assert result.evidence["aggregate_output_union_count"] == 8
        assert result.evidence["pooled_arithmetic"] == "packed"
        assert result.evidence["aggregate_pooled_arithmetic"] == "packed"
        assert result.evidence["dispatch_metadata"]["request_trace_id"] == f"trace-{index}"
        assert dict(result.plan.metadata)["request_trace_id"] == f"trace-{index}"
        memory = result.evidence["aggregate_memory_plan"]
        assert memory["kv_allocated_bytes"] == 1 * 8 * 12 * 1 * 2 * 2 * 4
        assert memory["output_bytes"] == 8 * 8 * 4
        _assert_snapshot(item.cache, snapshot)

    assert results[0].commit(2) == (2,)
    assert items[0].cache.lengths.tolist() == [snapshots[0][2][0] + 2]
    for item, snapshot in zip(items[1:], snapshots[1:], strict=True):
        _assert_snapshot(item.cache, snapshot)
    assert results[1].commit(0) == (0,)
    assert items[1].cache.lengths.tolist() == snapshots[1][2].tolist()
    assert items[1].cache.epoch == snapshots[1][3] + 1
    assert results[7].commit(1) == (1,)
    with pytest.raises(RuntimeError, match="stale"):
        results[0].commit(0)


def test_candidate_union_routes_back_to_original_order_and_exact_b1_contract() -> None:
    engine = _TinyEngine()
    candidate_rows = ((6, 1, 4), (4, 2, 6), (7, 1, 2))
    items = tuple(
        _submission(
            engine,
            index,
            output_contract=OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
            candidates=candidates,
        )
        for index, candidates in enumerate(candidate_rows)
    )
    stable_union = (6, 1, 4, 2, 7)
    combined = _equivalent_cache([item.cache for item in items])
    reference, _delta = pf.paged_forward_block(
        _TinyStore(),
        np.stack([item.payload.ids for item in items]),
        combined,
        output_contract="selected_token_rows",
        selected_rows=stable_union,
        last_only=True,
    )

    with PagedComponentReactor(
        max_batch_size=3,
        max_pending=3,
        max_batch_delay_seconds=0.1,
    ) as reactor:
        futures = tuple(_submit(reactor, item) for item in items)
        results = tuple(future.result(timeout=3) for future in futures)

    offsets = {token: index for index, token in enumerate(stable_union)}
    for row, (candidates, result) in enumerate(zip(candidate_rows, results, strict=True)):
        assert isinstance(result.outputs, tuple) and len(result.outputs) == 1
        summary = result.outputs[0]
        assert summary["candidate_token_ids"] == candidates
        values = reference[row].index_select(
            0,
            torch.as_tensor([offsets[token] for token in candidates]),
        )
        top = torch.topk(values, k=2)
        assert summary["winner_token_id"] == candidates[int(top.indices[0])]
        assert summary["runner_up_token_id"] == candidates[int(top.indices[1])]
        assert result.evidence["stable_output_union"] == list(stable_union)
    assert engine.store.calls.count(("embed_rows", "lm_head")) == 1


@pytest.mark.parametrize("arithmetic", ["row_stable", "row_stable_split"])
def test_row_stable_selected_union_preserves_each_independent_b1_head_shape_exactly(
    arithmetic: pf.PagedPooledArithmetic,
) -> None:
    engine = _TinyEngine()
    row_demands = (
        (0, 1, 2, 3),
        (2, 3, 4, 5),
        (0, 4, 6, 7),
        (1, 3, 5, 7),
    )
    items = tuple(
        _submission(
            engine,
            index,
            selected_rows=rows,
            tokens=((index + 1) % 8, (index + 3) % 8),
        )
        for index, rows in enumerate(row_demands)
    )
    references = tuple(
        pf.paged_forward_block(
            _TinyStore(),
            item.payload.ids[None, :],
            item.cache,
            output_contract="selected_token_rows",
            selected_rows=rows,
            last_only=True,
        )[0]
        for item, rows in zip(items, row_demands, strict=True)
    )

    with PagedComponentReactor(
        max_batch_size=4,
        max_pending=4,
        max_batch_delay_seconds=0.1,
        pooled_arithmetic=arithmetic,
    ) as reactor:
        futures = tuple(_submit(reactor, item) for item in items)
        results = tuple(future.result(timeout=3) for future in futures)
        assert reactor.paged_executor.pooled_arithmetic == arithmetic

    for result, reference in zip(results, references, strict=True):
        assert torch.equal(result.outputs, reference)
        assert result.evidence["pooled_arithmetic"] == arithmetic
        assert result.evidence["aggregate_pooled_arithmetic"] == arithmetic
        assert result.evidence["aggregate_output_union_count"] == 8
        if arithmetic == "row_stable_split":
            assert result.evidence["pooled_attention_global_prefix_kv_logical_bytes"] == 0
            assert (
                result.evidence["pooled_attention_request_local_prefix_kv_logical_bytes_max"] == 80
            )
            assert result.evidence["pooled_attention_explicit_live_prefix_kv_peak_bytes"] == 320
        else:
            assert result.evidence["pooled_attention_global_prefix_kv_logical_bytes"] == 320
            assert (
                result.evidence["pooled_attention_request_local_prefix_kv_logical_bytes_max"] == 0
            )
            assert result.evidence["pooled_attention_explicit_live_prefix_kv_peak_bytes"] == 800
        assert result.evidence["pooled_aggregate_provisional_delta_bytes"] == 128
    for name in ("L0.q", "L0.k", "L0.v", "L0.o", "L0.gate", "L0.up", "L0.down"):
        assert engine.store.calls.count(("matmul_row_stable", name)) == 1
        assert engine.store.calls.count(("matmul", name)) == 0
    assert engine.store.calls.count(("embed_rows", "lm_head")) == 1


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("token_count", [1, 2, 4])
@pytest.mark.parametrize(
    ("output_contract", "selected_rows", "last_only"),
    [
        ("full_logits", (), False),
        ("full_logits", (), True),
        ("hidden_state_only", (), False),
        ("selected_token_rows", (1, 3, 5), False),
    ],
)
def test_row_stable_split_is_exact_to_independent_b1_and_commits_identically(
    batch_size: int,
    token_count: int,
    output_contract: pf.PagedBlockOutputContract,
    selected_rows: tuple[int, ...],
    last_only: bool,
) -> None:
    pooled_caches = tuple(
        _new_cache(length=1 + row % 3, seed=900 + row) for row in range(batch_size)
    )
    reference_caches = tuple(
        _new_cache(length=1 + row % 3, seed=900 + row) for row in range(batch_size)
    )
    tokens = np.asarray(
        [[(row + offset + 1) % 8 for offset in range(token_count)] for row in range(batch_size)],
        dtype=np.int64,
    )
    references = tuple(
        pf.paged_forward_block(
            _TinyStore(),
            tokens[row : row + 1],
            reference_caches[row],
            output_contract=output_contract,
            selected_rows=selected_rows,
            last_only=last_only,
        )
        for row in range(batch_size)
    )
    snapshots = tuple(_snapshot(cache) for cache in pooled_caches)
    scratch: list[pf.PagedPooledScratchTelemetry] = []

    pooled_output, pooled_deltas = pf.paged_forward_block_pooled(
        _TinyStore(),
        tokens,
        pooled_caches,
        tuple(cache.mint_slot_lease() for cache in pooled_caches),
        output_contract=output_contract,
        selected_rows=selected_rows,
        last_only=last_only,
        arithmetic="row_stable_split",
        scratch_observer=scratch.append,
    )

    assert len(scratch) == 1
    assert scratch[0].global_prefix_kv_logical_bytes == 0
    expected_peak = 2 * (max(cache.lengths[0] for cache in pooled_caches) + token_count) * 1 * 2 * 4
    assert scratch[0].request_local_prefix_kv_logical_bytes_max == expected_peak
    expected_explicit_live_peak = (
        4 * (max(cache.lengths[0] for cache in pooled_caches) + token_count) * 2 * 2 * 4
    )
    assert scratch[0].explicit_live_prefix_kv_peak_bytes == expected_explicit_live_peak
    for row, (pooled_cache, reference_cache, reference, delta, snapshot) in enumerate(
        zip(
            pooled_caches,
            reference_caches,
            references,
            pooled_deltas,
            snapshots,
            strict=True,
        )
    ):
        reference_output, reference_delta = reference
        assert torch.equal(pooled_output[row : row + 1], reference_output)
        assert torch.equal(delta.k, reference_delta.k)
        assert torch.equal(delta.v, reference_delta.v)
        _assert_snapshot(pooled_cache, snapshot)
        assert pf.commit_block(reference_cache, reference_delta, (token_count,)) == (token_count,)
        assert pf.commit_block(pooled_cache, delta, (token_count,)) == (token_count,)
        assert torch.equal(pooled_cache.k, reference_cache.k)
        assert torch.equal(pooled_cache.v, reference_cache.v)
        assert np.array_equal(pooled_cache.lengths, reference_cache.lengths)
        assert pooled_cache.epoch == reference_cache.epoch


@pytest.mark.parametrize("pooled_arithmetic", ["", "stable", "ROW_STABLE", True, None])
def test_reactor_rejects_unknown_pooled_arithmetic_at_construction(
    pooled_arithmetic: object,
) -> None:
    with pytest.raises(ValueError, match="pooled_arithmetic"):
        PagedComponentBatchExecutor(
            pooled_arithmetic=pooled_arithmetic,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="pooled_arithmetic"):
        PagedComponentReactor(
            pooled_arithmetic=pooled_arithmetic,  # type: ignore[arg-type]
        )


def test_executor_pooled_arithmetic_is_read_only_after_construction() -> None:
    executor = PagedComponentBatchExecutor(pooled_arithmetic="row_stable")

    assert executor.pooled_arithmetic == "row_stable"
    with pytest.raises(AttributeError):
        executor.pooled_arithmetic = "packed"  # type: ignore[misc]
    assert executor.pooled_arithmetic == "row_stable"


def test_aggregate_template_and_executable_identity_bind_pooled_arithmetic() -> None:
    def execute(mode: pf.PagedPooledArithmetic):
        engine = _TinyEngine()
        items = tuple(_submission(engine, index) for index in range(2))
        with PagedComponentReactor(
            max_batch_size=2,
            max_pending=2,
            max_batch_delay_seconds=0.1,
            pooled_arithmetic=mode,
        ) as reactor:
            futures = tuple(_submit(reactor, item) for item in items)
            results = tuple(future.result(timeout=3) for future in futures)
            aggregate_layouts = tuple(
                artifact.template.compute_layout_ids
                for artifact in reactor.paged_executor._aggregate_templates.values()  # noqa: SLF001
            )
        return items, results, aggregate_layouts

    packed_items, packed, packed_layouts = execute("packed")
    stable_items, stable, stable_layouts = execute("row_stable")

    assert packed_items[0].template.fingerprint == stable_items[0].template.fingerprint
    assert packed[0].plan_fingerprint == stable[0].plan_fingerprint
    assert (
        packed[0].evidence["aggregate_template_fingerprint"]
        != stable[0].evidence["aggregate_template_fingerprint"]
    )
    assert (
        packed[0].evidence["aggregate_executable_key"]
        != stable[0].evidence["aggregate_executable_key"]
    )
    assert {result.evidence["pooled_arithmetic"] for result in packed} == {"packed"}
    assert {result.evidence["aggregate_pooled_arithmetic"] for result in packed} == {"packed"}
    assert {result.evidence["pooled_arithmetic"] for result in stable} == {"row_stable"}
    assert {result.evidence["aggregate_pooled_arithmetic"] for result in stable} == {"row_stable"}
    assert len(packed_layouts) == len(stable_layouts) == 1
    assert tuple(
        value for value in packed_layouts[0] if value.startswith("paged-pooled-arithmetic:")
    ) == ("paged-pooled-arithmetic:packed",)
    assert tuple(
        value for value in stable_layouts[0] if value.startswith("paged-pooled-arithmetic:")
    ) == ("paged-pooled-arithmetic:row_stable",)
    assert packed_layouts[0][:-1] == packed_items[0].template.compute_layout_ids
    assert stable_layouts[0][:-1] == stable_items[0].template.compute_layout_ids


def test_source_template_cannot_preclaim_reserved_pooled_arithmetic_layout() -> None:
    engine = _TinyEngine()
    item = _submission(engine, 0)
    forged_template = replace(
        item.template,
        compute_layout_ids=(
            *item.template.compute_layout_ids,
            "paged-pooled-arithmetic:row_stable",
        ),
    )
    forged = replace(
        item,
        template=forged_template,
        binding=replace(item.binding, template_fingerprint=forged_template.fingerprint),
        lowered=lower_work_template(forged_template, "paged-qstore"),
    )

    with PagedComponentReactor(
        max_batch_size=1,
        max_pending=1,
        max_batch_delay_seconds=0,
        pooled_arithmetic="row_stable",
    ) as reactor:
        with pytest.raises(ValueError, match="cannot forge"):
            _submit(reactor, forged).result(timeout=3)

    assert engine.store.calls == []


@pytest.mark.parametrize(
    ("contract", "shape"),
    [
        (OutputContract.FULL_LOGITS, (1, 2, 8)),
        (OutputContract.LAST_TOKEN_LOGITS, (1, 8)),
        (OutputContract.HIDDEN_STATE_ONLY, (1, 2, 4)),
    ],
)
def test_tensor_contracts_split_back_to_exact_b1_shapes(
    contract: OutputContract,
    shape: tuple[int, ...],
) -> None:
    engine = _TinyEngine()
    items = tuple(_submission(engine, index, output_contract=contract) for index in range(2))

    with PagedComponentReactor(
        max_batch_size=2,
        max_pending=2,
        max_batch_delay_seconds=0.1,
    ) as reactor:
        futures = tuple(_submit(reactor, item) for item in items)
        results = tuple(future.result(timeout=3) for future in futures)

    assert all(tuple(result.outputs.shape) == shape for result in results)
    assert all(
        result.provisional_delta.delta.slot_lease == item.lease
        for result, item in zip(results, items, strict=True)
    )


def test_stale_child_is_isolated_and_valid_sibling_still_executes_and_commits() -> None:
    engine = _TinyEngine()
    stale = _submission(engine, 0)
    valid = _submission(engine, 1)
    stale.cache.release_slot_lease(stale.lease)
    replacement = stale.cache.mint_slot_lease()
    stale_before = _snapshot(stale.cache)
    valid_before = _snapshot(valid.cache)

    with PagedComponentReactor(
        max_batch_size=2,
        max_pending=2,
        max_batch_delay_seconds=0.1,
    ) as reactor:
        stale_future = _submit(reactor, stale)
        valid_future = _submit(reactor, valid)
        with pytest.raises(RuntimeError, match="stale"):
            stale_future.result(timeout=3)
        result = valid_future.result(timeout=3)

    _assert_snapshot(stale.cache, stale_before)
    _assert_snapshot(valid.cache, valid_before)
    assert result.evidence["aggregate_actual_batch"] == 1
    assert result.commit(1) == (1,)
    _assert_snapshot(stale.cache, stale_before)
    stale.cache.release_slot_lease(replacement)


def test_queued_cancellation_never_executes_or_mutates_its_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _TinyEngine()
    running = _submission(engine, 0, output_contract=OutputContract.FULL_LOGITS)
    cancelled = _submission(engine, 1, output_contract=OutputContract.SELECTED_TOKEN_ROWS)
    cancelled_before = _snapshot(cancelled.cache)
    started = threading.Event()
    release = threading.Event()
    original = pf.paged_forward_block_pooled

    def blocking_kernel(*args: object, **kwargs: object):
        started.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(pf, "paged_forward_block_pooled", blocking_kernel)
    reactor = PagedComponentReactor(
        max_batch_size=1,
        max_pending=2,
        max_batch_delay_seconds=0,
    )
    try:
        running_future = _submit(reactor, running)
        assert started.wait(2)
        cancelled_future = _submit(reactor, cancelled)
        assert cancelled_future.cancel()
        release.set()
        running_future.result(timeout=3)
        assert reactor.wait_idle(timeout=3)
    finally:
        release.set()
        reactor.shutdown(wait=True, cancel_pending=True)

    assert cancelled_future.cancelled()
    _assert_snapshot(cancelled.cache, cancelled_before)


def test_mixed_engine_owner_fails_whole_wave_before_either_store_access() -> None:
    first_engine = _TinyEngine()
    second_engine = _TinyEngine()
    first = _submission(first_engine, 0)
    second = _submission(second_engine, 1)
    assert first.template.fingerprint == second.template.fingerprint

    with PagedComponentReactor(
        max_batch_size=2,
        max_pending=2,
        max_batch_delay_seconds=0.1,
    ) as reactor:
        futures = (_submit(reactor, first), _submit(reactor, second))
        for future in futures:
            with pytest.raises(RuntimeError, match="mix engine"):
                future.result(timeout=3)

    assert first_engine.store.calls == []
    assert second_engine.store.calls == []


def test_duplicate_and_foreign_slot_authority_fail_closed_before_traversal() -> None:
    engine = _TinyEngine()
    first = _submission(engine, 0)
    second = _submission(engine, 1)
    with pytest.raises(ValueError, match="different caches"):
        PagedReactorPayload(
            engine=engine,
            ids=second.payload.ids,
            state_binding=second.payload.state_binding,
            slot_lease=first.lease,
        )

    duplicate_plan = replace(
        second.plan,
        kv_read_handles=first.plan.kv_read_handles,
        kv_write_handles=first.plan.kv_write_handles,
    )
    duplicate_template, duplicate_binding = decompose_work_plan(duplicate_plan)
    assert duplicate_template.fingerprint == first.template.fingerprint
    duplicate = _Submission(
        duplicate_plan,
        duplicate_template,
        duplicate_binding,
        lower_work_template(duplicate_template, "paged-qstore"),
        PagedReactorPayload(
            engine=engine,
            ids=second.payload.ids,
            state_binding=first.payload.state_binding,
            slot_lease=first.lease,
        ),
        first.cache,
        first.lease,
    )

    with PagedComponentReactor(
        max_batch_size=2,
        max_pending=2,
        max_batch_delay_seconds=0.1,
    ) as reactor:
        futures = (_submit(reactor, first), _submit(reactor, duplicate))
        for future in futures:
            with pytest.raises(RuntimeError, match="reuse one cache|duplicate"):
                future.result(timeout=3)

    assert engine.store.calls == []


def test_output_numerical_and_kv_contracts_never_share_a_physical_wave() -> None:
    engine = _TinyEngine()
    selected = _submission(engine, 0)
    hidden = _submission(engine, 1, output_contract=OutputContract.HIDDEN_STATE_ONLY)
    alternate = _submission(
        engine,
        2,
        numerical_contract="paged-qstore-test-alternate",
    )
    different_capacity = _submission(
        engine,
        3,
        cache=_new_cache(length=1, seed=57, capacity=16),
    )
    assert (
        len(
            {
                selected.template.fingerprint,
                hidden.template.fingerprint,
                alternate.template.fingerprint,
                different_capacity.template.fingerprint,
            }
        )
        == 4
    )

    with PagedComponentReactor(
        max_batch_size=4,
        max_pending=4,
        max_batch_delay_seconds=0.02,
    ) as reactor:
        futures = tuple(
            _submit(reactor, item) for item in (selected, hidden, alternate, different_capacity)
        )
        tuple(future.result(timeout=3) for future in futures)
        telemetry = reactor.telemetry()

    assert telemetry.batches == 4
    assert telemetry.batch_width_histogram == ((1, 4),)
    assert engine.store.calls.count(("matmul", "L0.q")) == 4
