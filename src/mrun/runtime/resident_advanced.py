"""Unified P4-P8 resident-runtime orchestration over exact backend contracts.

This module does not pretend that one kernel serves every model.  It supplies the common
authorities that the dense, MoE, precision, and speculative backends plug into: fixed admitted
batch families, transactional captured-decode slots, target-aligned K4 correction, and exact
precision-family routing.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..compiler.temporal import TemporalStateArena, TemporalStateHandle


class RebindableTemplate(Protocol):
    def rebind(
        self,
        ids_list: Sequence[Sequence[int]],
        token_ids: Sequence[int],
        *,
        request_id: str,
    ) -> int: ...

    def execute(self, *, expected_generation: int, request_id: str) -> Any: ...


@dataclass(frozen=True)
class CapturedBatchResult:
    request_ids: tuple[str, ...]
    scores: tuple[tuple[float, ...], ...]
    batch_size: int
    generation: int
    template_bucket: tuple[int, int, int]


class ResidentCapturedBatchFamily:
    """P4: exact captured batches over a fixed promoted bucket family."""

    def __init__(
        self,
        templates: Mapping[tuple[int, int, int], RebindableTemplate],
        *,
        allowed_batches: tuple[int, ...] = (2, 4, 8, 16, 32),
    ) -> None:
        if not templates:
            raise ValueError("captured batch family requires at least one template")
        if allowed_batches != tuple(sorted(set(allowed_batches))) or min(allowed_batches) <= 1:
            raise ValueError(
                "captured batch sizes must be unique, increasing, and greater than one"
            )
        self._templates = dict(templates)
        self._allowed_batches = allowed_batches
        self._dispatches = 0
        self._rows = 0
        self._lock = threading.Lock()

    def execute(
        self,
        token_rows: Sequence[Sequence[int]],
        selected_token_ids: Sequence[int],
        *,
        request_ids: Sequence[str],
        request_id: str,
    ) -> CapturedBatchResult:
        rows = tuple(tuple(int(token) for token in row) for row in token_rows)
        row_ids = tuple(int(token) for token in selected_token_ids)
        logical_ids = tuple(str(value) for value in request_ids)
        if not rows or len(rows) != len(logical_ids) or len(set(logical_ids)) != len(logical_ids):
            raise ValueError("captured batch requires one unique request identity per row")
        if len(rows) not in self._allowed_batches:
            raise ValueError("captured batch width is not admitted")
        widths = {len(row) for row in rows}
        if len(widths) != 1 or not row_ids or len(row_ids) != len(set(row_ids)):
            raise ValueError("captured batch requires exact sequence and selected-row shapes")
        bucket = (len(rows), next(iter(widths)), len(row_ids))
        try:
            template = self._templates[bucket]
        except KeyError as exc:
            raise KeyError(f"captured batch bucket is not installed: {bucket}") from exc
        generation = template.rebind(rows, row_ids, request_id=request_id)
        output = torch.as_tensor(
            template.execute(expected_generation=generation, request_id=request_id),
            dtype=torch.float32,
        ).cpu()
        if tuple(output.shape) != (len(rows), len(row_ids)):
            raise RuntimeError("captured batch template returned the wrong shape")
        with self._lock:
            self._dispatches += 1
            self._rows += len(rows)
        return CapturedBatchResult(
            request_ids=logical_ids,
            scores=tuple(tuple(float(value) for value in row) for row in output.tolist()),
            batch_size=len(rows),
            generation=generation,
            template_bucket=bucket,
        )

    def telemetry(self) -> dict[str, int]:
        with self._lock:
            return {"physical_dispatches": self._dispatches, "logical_rows": self._rows}

    @property
    def allowed_batches(self) -> tuple[int, ...]:
        return self._allowed_batches


@dataclass
class _QueuedCapturedRow:
    request_id: str
    token_row: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    enqueued_at: float
    future: Future[tuple[float, ...]]


class ResidentCapturedBatchService:
    """Bounded P4 arrival queue over a captured batch family with exact eager fallback."""

    def __init__(
        self,
        family: ResidentCapturedBatchFamily,
        *,
        eager_fallback: Callable[[tuple[int, ...], tuple[int, ...]], Sequence[float]],
        max_queue_delay_seconds: float = 0.002,
        max_pending: int = 1_024,
        telemetry_window: int = 4_096,
    ) -> None:
        if max_queue_delay_seconds <= 0 or not math.isfinite(max_queue_delay_seconds):
            raise ValueError("captured batch queue delay must be finite and positive")
        if max_pending <= 0 or telemetry_window <= 0:
            raise ValueError("captured batch service bounds must be positive")
        self.family = family
        self._fallback = eager_fallback
        self._max_delay = float(max_queue_delay_seconds)
        self._max_pending = int(max_pending)
        self._queue: deque[_QueuedCapturedRow] = deque()
        self._request_ids: set[str] = set()
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._cohort_id = 0
        self._queue_ms: deque[float] = deque(maxlen=telemetry_window)
        self._execution_ms: deque[float] = deque(maxlen=telemetry_window)
        self._captured_rows = 0
        self._captured_dispatches = 0
        self._captured_width_sum = 0
        self._captured_width_max = 0
        self._fallback_rows = 0
        self._cancelled_rows = 0
        self._peak_pending = 0
        self._worker = threading.Thread(
            target=self._run,
            name="mrun-resident-captured-batch",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        token_row: Sequence[int],
        selected_token_ids: Sequence[int],
        *,
        request_id: str,
    ) -> Future[tuple[float, ...]]:
        row = tuple(int(value) for value in token_row)
        selected = tuple(int(value) for value in selected_token_ids)
        if (
            not request_id
            or not row
            or not selected
            or len(selected) != len(set(selected))
        ):
            raise ValueError("captured batch service request is incomplete")
        future: Future[tuple[float, ...]] = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("captured batch service is closed")
            if request_id in self._request_ids:
                raise ValueError("captured batch request ID is already active")
            if len(self._queue) >= self._max_pending:
                raise MemoryError("captured batch pending queue is full")
            self._request_ids.add(request_id)
            self._queue.append(
                _QueuedCapturedRow(
                    request_id,
                    row,
                    selected,
                    time.monotonic(),
                    future,
                )
            )
            self._peak_pending = max(self._peak_pending, len(self._queue))
            self._condition.notify()
        return future

    def close(self, *, wait: bool = True) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        if wait:
            self._worker.join()

    def telemetry(self) -> dict[str, float | int | None]:
        with self._condition:
            queue = tuple(self._queue_ms)
            execution = tuple(self._execution_ms)
            return {
                "captured_rows": self._captured_rows,
                "captured_dispatches": self._captured_dispatches,
                "captured_mean_width": (
                    self._captured_width_sum / self._captured_dispatches
                    if self._captured_dispatches
                    else None
                ),
                "captured_max_width": self._captured_width_max,
                "fallback_rows": self._fallback_rows,
                "cancelled_rows": self._cancelled_rows,
                "peak_pending": self._peak_pending,
                "pending": len(self._queue),
                "queue_p50_ms": _quantile(queue, 0.50),
                "queue_p95_ms": _quantile(queue, 0.95),
                "execution_p50_ms": _quantile(execution, 0.50),
                "execution_p95_ms": _quantile(execution, 0.95),
            }

    def __enter__(self) -> ResidentCapturedBatchService:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _run(self) -> None:
        while True:
            with self._condition:
                cohort = self._next_cohort_locked()
                if cohort is None:
                    if self._closed and not self._queue:
                        return
                    self._condition.wait(self._wait_seconds_locked())
                    continue
            self._execute(cohort)

    def _next_cohort_locked(self) -> tuple[_QueuedCapturedRow, ...] | None:
        self._discard_cancelled_locked()
        if not self._queue:
            return None
        first = self._queue[0]
        compatible = tuple(
            item
            for item in self._queue
            if len(item.token_row) == len(first.token_row)
            and item.selected_token_ids == first.selected_token_ids
            and not item.future.cancelled()
        )
        admitted = tuple(batch for batch in self.family.allowed_batches if batch <= len(compatible))
        expired = time.monotonic() - first.enqueued_at >= self._max_delay
        if len(compatible) >= max(self.family.allowed_batches):
            chosen = compatible[: max(self.family.allowed_batches)]
        elif admitted and (expired or self._closed):
            chosen = compatible[: max(admitted)]
        elif not admitted and (expired or self._closed):
            chosen = (first,)
        else:
            return None
        chosen_ids = {id(item) for item in chosen}
        self._queue = deque(item for item in self._queue if id(item) not in chosen_ids)
        return chosen

    def _wait_seconds_locked(self) -> float | None:
        if not self._queue:
            return None
        return max(0.0, self._max_delay - (time.monotonic() - self._queue[0].enqueued_at))

    def _discard_cancelled_locked(self) -> None:
        retained: deque[_QueuedCapturedRow] = deque()
        for item in self._queue:
            if item.future.cancelled():
                self._request_ids.discard(item.request_id)
                self._cancelled_rows += 1
            else:
                retained.append(item)
        self._queue = retained

    def _execute(self, cohort: tuple[_QueuedCapturedRow, ...]) -> None:
        started = time.perf_counter()
        dispatched_at = time.monotonic()
        try:
            if len(cohort) == 1:
                item = cohort[0]
                rows = (tuple(float(value) for value in self._fallback(
                    item.token_row, item.selected_token_ids
                )),)
                captured = False
            else:
                self._cohort_id += 1
                result = self.family.execute(
                    tuple(item.token_row for item in cohort),
                    cohort[0].selected_token_ids,
                    request_ids=tuple(item.request_id for item in cohort),
                    request_id=f"service-cohort-{self._cohort_id}",
                )
                rows = result.scores
                captured = True
            elapsed_ms = (time.perf_counter() - started) * 1_000
            with self._condition:
                self._execution_ms.append(elapsed_ms)
                if captured:
                    self._captured_rows += len(cohort)
                    self._captured_dispatches += 1
                    self._captured_width_sum += len(cohort)
                    self._captured_width_max = max(self._captured_width_max, len(cohort))
                else:
                    self._fallback_rows += 1
            for item, row in zip(cohort, rows, strict=True):
                if not item.future.cancelled():
                    item.future.set_result(tuple(row))
        except BaseException as exc:
            for item in cohort:
                if not item.future.cancelled():
                    item.future.set_exception(exc)
        finally:
            with self._condition:
                for item in cohort:
                    self._request_ids.discard(item.request_id)
                    if item.future.cancelled():
                        self._cancelled_rows += 1
                    else:
                        self._queue_ms.append((dispatched_at - item.enqueued_at) * 1_000)
                self._condition.notify_all()


def _quantile(values: tuple[float, ...], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


class CapturedDecodeBackend(Protocol):
    """Device backend for a fixed-address one-token decode slot family."""

    @property
    def allocation_count(self) -> int: ...

    def initialize(self, *, slot: int, prefix: tuple[int, ...], request_id: str) -> None: ...

    def bind(self, *, slot: int, token_id: int, position: int, request_id: str) -> int: ...

    def replay(self, *, slot: int, generation: int, request_id: str) -> int: ...

    def commit(self, *, slot: int, accepted_count: int) -> None: ...

    def rollback(self, *, slot: int) -> None: ...

    def release(self, *, slot: int, request_id: str) -> None: ...


class TorchCudaGraphDecodeBackend:
    """Concrete fixed-address CUDA Graph backend for one-token stateful decode.

    ``capture_step`` must be graph-safe and write any provisional K/V state using the supplied
    device scalar tensors.  Commit and rollback remain explicit host-side authority transitions;
    they must only publish or discard the provisional state written by the captured step.
    """

    def __init__(
        self,
        capture_step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        device: torch.device | str,
        initialize_state: Callable[[int, tuple[int, ...], str], None],
        commit_state: Callable[[int, int], None],
        rollback_state: Callable[[int], None],
        release_state: Callable[[int, str], None],
        warmup: int = 3,
    ) -> None:
        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("stateful CUDA Graph decode requires an available CUDA device")
        if warmup <= 0:
            raise ValueError("CUDA Graph decode warmup must be positive")
        self._device = resolved
        self._initialize_state = initialize_state
        self._commit_state = commit_state
        self._rollback_state = rollback_state
        self._release_state = release_state
        self._token = torch.zeros((), dtype=torch.int64, device=resolved)
        self._position = torch.zeros((), dtype=torch.int64, device=resolved)
        self._slot = torch.zeros((), dtype=torch.int64, device=resolved)
        warm_stream = torch.cuda.Stream(device=resolved)
        warm_stream.wait_stream(torch.cuda.current_stream(resolved))
        with torch.cuda.stream(warm_stream):
            for _ in range(warmup):
                capture_step(self._token, self._position, self._slot)
        torch.cuda.current_stream(resolved).wait_stream(warm_stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            output = capture_step(self._token, self._position, self._slot)
        if output.numel() != 1:
            raise ValueError("captured decode step must return one greedy token")
        self._output = output.reshape(())
        self._generation = 0
        self._bound: dict[int, tuple[int, str]] = {}
        self._lock = threading.RLock()

    @property
    def allocation_count(self) -> int:
        # Three mutable scalar bindings plus one captured scalar result. CUDA Graph private-pool
        # allocations happen at construction and are intentionally outside replay accounting.
        return 4

    def initialize(self, *, slot: int, prefix: tuple[int, ...], request_id: str) -> None:
        with self._lock:
            if slot in self._bound:
                raise RuntimeError("captured decode slot is already owned")
            self._initialize_state(int(slot), tuple(prefix), str(request_id))
            self._bound[int(slot)] = (0, str(request_id))

    def bind(self, *, slot: int, token_id: int, position: int, request_id: str) -> int:
        with self._lock:
            _generation, owner = self._require_owner(slot, request_id)
            self._generation += 1
            generation = self._generation
            self._token.fill_(int(token_id))
            self._position.fill_(int(position))
            self._slot.fill_(int(slot))
            self._bound[int(slot)] = (generation, owner)
            return generation

    def replay(self, *, slot: int, generation: int, request_id: str) -> int:
        with self._lock:
            current, _owner = self._require_owner(slot, request_id)
            if int(generation) != current:
                raise RuntimeError("captured decode binding generation is stale")
            self._graph.replay()
            return int(self._output.item())

    def commit(self, *, slot: int, accepted_count: int) -> None:
        with self._lock:
            if int(slot) not in self._bound:
                raise KeyError("unknown captured decode slot")
            self._commit_state(int(slot), int(accepted_count))

    def rollback(self, *, slot: int) -> None:
        with self._lock:
            if int(slot) not in self._bound:
                raise KeyError("unknown captured decode slot")
            self._rollback_state(int(slot))

    def release(self, *, slot: int, request_id: str) -> None:
        with self._lock:
            self._require_owner(slot, request_id)
            self._release_state(int(slot), str(request_id))
            del self._bound[int(slot)]

    def _require_owner(self, slot: int, request_id: str) -> tuple[int, str]:
        try:
            generation, owner = self._bound[int(slot)]
        except KeyError as exc:
            raise KeyError("unknown captured decode slot") from exc
        if owner != str(request_id):
            raise RuntimeError("captured decode slot belongs to another request")
        return generation, owner


@dataclass(frozen=True)
class CapturedDecodeStep:
    request_id: str
    token_id: int
    generation: int
    before: TemporalStateHandle
    after: TemporalStateHandle
    committed: bool


class TransactionalCapturedDecodeLane:
    """P6: state authority around a fixed-address captured one-token backend."""

    def __init__(self, backend: CapturedDecodeBackend, *, slots: int, capacity: int) -> None:
        self.backend = backend
        self.states = TemporalStateArena(slots=slots, capacity=capacity)
        self._initial_allocations = int(backend.allocation_count)
        self._steps = 0

    def allocate(self, request_id: str, prefix: tuple[int, ...]) -> TemporalStateHandle:
        handle = self.states.create(request_id, prefix)
        try:
            self.backend.initialize(
                slot=handle.slot,
                prefix=tuple(int(token) for token in prefix),
                request_id=request_id,
            )
        except BaseException:
            self.states.release(request_id)
            raise
        return handle

    def step(self, request_id: str, token_id: int, *, commit: bool = True) -> CapturedDecodeStep:
        before = self.states.observe(request_id)
        if before.committed_length >= before.capacity:
            raise OverflowError("captured decode state has reached its context capacity")
        generation = self.backend.bind(
            slot=before.slot,
            token_id=int(token_id),
            position=before.committed_length,
            request_id=request_id,
        )
        predicted = int(
            self.backend.replay(
                slot=before.slot,
                generation=generation,
                request_id=request_id,
            )
        )
        delta = self.states.begin(request_id, (int(token_id),))
        if commit:
            self.backend.commit(slot=before.slot, accepted_count=1)
            after = self.states.commit(delta, 1)
        else:
            self.backend.rollback(slot=before.slot)
            after = self.states.rollback(delta)
        if int(self.backend.allocation_count) != self._initial_allocations:
            raise RuntimeError("captured decode backend allocated after lane construction")
        self._steps += 1
        return CapturedDecodeStep(
            request_id=request_id,
            token_id=predicted,
            generation=generation,
            before=before,
            after=after,
            committed=commit,
        )

    def release(self, request_id: str) -> None:
        state = self.states.observe(request_id)
        self.backend.release(slot=state.slot, request_id=request_id)
        self.states.release(request_id)

    def telemetry(self) -> dict[str, int]:
        return {**self.states.evidence(), "decode_steps": self._steps}


@dataclass(frozen=True)
class K4Step:
    proposals: tuple[int, ...]
    target_predictions: tuple[int, ...]
    outputs: tuple[int, ...]
    accepted_proposals: int
    target_passes: int = 1

    @property
    def useful_outputs(self) -> int:
        return len(self.outputs)


class TargetAlignedK4Runtime:
    """P7: exact greedy K4 verification with target correction and explicit ceiling metrics."""

    def __init__(
        self,
        predictor: Callable[[tuple[int, ...], int], Sequence[int]],
        verifier: Callable[[tuple[int, ...], tuple[int, ...]], Sequence[int]],
        *,
        k: int = 4,
    ) -> None:
        if k != 4:
            raise ValueError("target-aligned runtime is deliberately fixed to K4")
        self.predictor = predictor
        self.verifier = verifier
        self.k = k
        self._steps = 0
        self._target_passes = 0
        self._useful_outputs = 0

    def step(self, prefix: Sequence[int]) -> K4Step:
        context = tuple(int(token) for token in prefix)
        proposals = tuple(int(token) for token in self.predictor(context, self.k))
        if len(proposals) != self.k:
            raise RuntimeError("K4 predictor must return exactly four proposals")
        target = tuple(int(token) for token in self.verifier(context, proposals))
        if len(target) != self.k + 1:
            raise RuntimeError("K4 verifier must return four position predictions plus a bonus")
        accepted = 0
        while accepted < self.k and proposals[accepted] == target[accepted]:
            accepted += 1
        outputs = (
            (*proposals, target[-1])
            if accepted == self.k
            else (*proposals[:accepted], target[accepted])
        )
        result = K4Step(
            proposals=proposals,
            target_predictions=target,
            outputs=tuple(outputs),
            accepted_proposals=accepted,
        )
        self._steps += 1
        self._target_passes += 1
        self._useful_outputs += result.useful_outputs
        return result

    def telemetry(self) -> dict[str, float | int | bool]:
        outputs_per_pass = self._useful_outputs / max(1, self._target_passes)
        return {
            "steps": self._steps,
            "target_passes": self._target_passes,
            "useful_outputs": self._useful_outputs,
            "useful_outputs_per_target_pass": outputs_per_pass,
            "clears_two_outputs_per_pass_gate": outputs_per_pass >= 2.0,
        }


@dataclass(frozen=True)
class TargetAlignedPredictorManifest:
    predictor_id: str
    target_model_sha256: str
    predictor_weights_sha256: str
    training_contract: str
    proposal_tokens: int = 4

    def __post_init__(self) -> None:
        if not self.predictor_id or not self.training_contract:
            raise ValueError("target-aligned predictor identity is incomplete")
        for field in ("target_model_sha256", "predictor_weights_sha256"):
            if re.fullmatch("[0-9a-f]{64}", getattr(self, field)) is None:
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if self.proposal_tokens != 4:
            raise ValueError("target-aligned predictor manifest must declare K4")


def build_target_aligned_k4_speculative_runtime(
    target: Any,
    predictor: Any,
    *,
    semantic_token_count: int,
    manifest: TargetAlignedPredictorManifest,
    owns_runtimes: bool = False,
) -> Any:
    """Bind a provenance-carrying K4 predictor to the exact transactional verifier runtime."""

    from .speculative import ExactGreedySpeculativeRuntime

    if manifest.proposal_tokens != 4:
        raise ValueError("target-aligned speculative runtime requires K4")
    runtime = ExactGreedySpeculativeRuntime(
        target,
        predictor,
        semantic_token_count=semantic_token_count,
        proposal_tokens=4,
        runtime_id=f"target-aligned-k4:{manifest.predictor_id}",
        owns_runtimes=owns_runtimes,
    )
    # The verifier runtime deliberately does not infer training provenance from backend types.
    # Attach the immutable manifest as the admission authority for evidence and promotion code.
    runtime.target_aligned_manifest = manifest
    return runtime


@dataclass(frozen=True)
class PrecisionBackend:
    name: str
    precision: str
    architecture: str
    workload: str
    numerical_contract: str
    output_contract: str
    store_bytes: int
    max_resident_bytes: int
    max_active_bytes: int
    quality_evidence_sha256: str
    execute: Callable[..., Any]
    promoted: bool

    def __post_init__(self) -> None:
        if not all(
            (
                self.name,
                self.precision,
                self.architecture,
                self.workload,
                self.numerical_contract,
                self.output_contract,
            )
        ):
            raise ValueError("precision backend identity is incomplete")
        if (
            self.store_bytes <= 0
            or self.max_resident_bytes <= 0
            or self.max_active_bytes <= 0
            or self.max_active_bytes > self.max_resident_bytes
        ):
            raise ValueError("precision backend byte accounting is invalid")
        if re.fullmatch("[0-9a-f]{64}", self.quality_evidence_sha256) is None:
            raise ValueError("precision quality evidence must be a lowercase SHA-256")


class ResidentPrecisionFamily:
    """P8: fail-closed routing across promoted FP8, int4, and route-first MoE backends."""

    def __init__(self, backends: Sequence[PrecisionBackend]) -> None:
        self._backends = {
            (item.architecture, item.precision, item.workload, item.output_contract): item
            for item in backends
        }
        if len(self._backends) != len(tuple(backends)):
            raise ValueError("precision family backend identities must be unique")

    def select(
        self,
        *,
        architecture: str,
        precision: str,
        workload: str,
        output_contract: str,
        free_bytes: int,
        numerical_contract: str,
    ) -> PrecisionBackend:
        try:
            backend = self._backends[(architecture, precision, workload, output_contract)]
        except KeyError as exc:
            raise LookupError("no exact resident precision backend") from exc
        if not backend.promoted:
            raise PermissionError("resident precision backend is not promoted")
        if backend.numerical_contract != numerical_contract:
            raise PermissionError("resident precision numerical contract does not match")
        if free_bytes < backend.max_resident_bytes:
            raise MemoryError("resident precision backend does not fit")
        return backend


def require_finite_speedup(value: float, *, minimum: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= minimum:
        raise ValueError(f"speedup must be finite and greater than {minimum}")
    return result


__all__ = [
    "CapturedBatchResult",
    "CapturedDecodeBackend",
    "CapturedDecodeStep",
    "K4Step",
    "PrecisionBackend",
    "ResidentCapturedBatchFamily",
    "ResidentCapturedBatchService",
    "ResidentPrecisionFamily",
    "TargetAlignedK4Runtime",
    "TargetAlignedPredictorManifest",
    "TorchCudaGraphDecodeBackend",
    "TransactionalCapturedDecodeLane",
    "build_target_aligned_k4_speculative_runtime",
    "require_finite_speedup",
]
