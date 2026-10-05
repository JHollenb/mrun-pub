"""Restricted resident selected-score worker with promotion-gated graph replay."""

from __future__ import annotations

import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..compiler.resident_promotions import (
    ResidentTemplatePromotion,
    select_resident_template_promotion,
)
from .graph_pool import (
    GraphTemplate,
    GraphTemplateKey,
    GraphTemplatePool,
    ResidentQStoreArena,
    ResidentQStoreArenaKey,
)

_WORKERS: weakref.WeakSet[ResidentModelWorker] = weakref.WeakSet()
_WORKERS_LOCK = threading.RLock()


@dataclass(frozen=True)
class ResidentSelectedScoreRequest:
    request_id: str
    token_rows: tuple[tuple[int, ...], ...]
    selected_token_ids: tuple[int, ...]
    numerical_contract: str
    deadline_ns: int

    def __post_init__(self) -> None:
        if not self.request_id or not self.token_rows or not self.selected_token_ids:
            raise ValueError("resident selected-score request is incomplete")
        widths = {len(row) for row in self.token_rows}
        if len(widths) != 1 or min(widths) <= 0:
            raise ValueError("resident selected-score rows require one non-empty exact shape")
        if len(self.selected_token_ids) != len(set(self.selected_token_ids)):
            raise ValueError("resident selected rows must be unique")
        if min(self.selected_token_ids) < 0 or not self.numerical_contract:
            raise ValueError("resident selected-score contract is invalid")
        if self.deadline_ns <= 0:
            raise ValueError("resident selected-score deadline must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResidentSelectedScoreRequest:
        expected = {
            "request_id",
            "token_rows",
            "selected_token_ids",
            "numerical_contract",
            "deadline_ns",
        }
        if set(payload) != expected:
            raise ValueError("resident selected-score request fields are not exact")
        return cls(
            request_id=str(payload["request_id"]),
            token_rows=tuple(tuple(int(value) for value in row) for row in payload["token_rows"]),
            selected_token_ids=tuple(int(value) for value in payload["selected_token_ids"]),
            numerical_contract=str(payload["numerical_contract"]),
            deadline_ns=int(payload["deadline_ns"]),
        )


@dataclass(frozen=True)
class ResidentSelectedScoreResponse:
    request_id: str
    scores: tuple[tuple[float, ...], ...]
    route: str
    setup_ms: float
    queue_ms: float
    execution_ms: float
    template_fingerprint: str | None
    generation: int | None
    fallback_reason: str | None


class ResidentModelWorker:
    """One typed resident model service; arbitrary code cannot enter this boundary."""

    def __init__(
        self,
        engine: Any,
        arena: ResidentQStoreArena,
        pool: GraphTemplatePool,
        promotions: tuple[ResidentTemplatePromotion, ...],
        template_factory: Any,
        *,
        device_identity: str,
        estimated_template_bytes: int,
    ) -> None:
        if estimated_template_bytes < 0:
            raise ValueError("estimated template bytes must be non-negative")
        self.engine = engine
        self.arena = arena
        self.pool = pool
        self.promotions = promotions
        self.template_factory = template_factory
        self.device_identity = device_identity
        self.estimated_template_bytes = int(estimated_template_bytes)
        self._requests = 0
        self._graph_requests = 0
        self._eager_requests = 0
        self._fallback_reasons: dict[str, int] = {}
        self._closed = False
        with _WORKERS_LOCK:
            _WORKERS.add(self)

    def _template_key(self, request: ResidentSelectedScoreRequest) -> GraphTemplateKey:
        return GraphTemplateKey(
            arena_fingerprint=self.arena.key.fingerprint,
            execution_mode="selected-last-stateless-score",
            output_contract="selected-token-rows",
            batch_size=len(request.token_rows),
            sequence_length=len(request.token_rows[0]),
            selected_row_count=len(request.selected_token_ids),
        )

    def execute(self, request: ResidentSelectedScoreRequest) -> ResidentSelectedScoreResponse:
        if self._closed:
            raise RuntimeError("resident model worker is closed")
        received_ns = time.monotonic_ns()
        if received_ns > request.deadline_ns:
            raise TimeoutError("resident selected-score request deadline elapsed before dispatch")
        key = self._template_key(request)
        promotion = select_resident_template_promotion(
            self.promotions,
            arena_fingerprint=key.arena_fingerprint,
            numerical_contract=request.numerical_contract,
            output_contract=key.output_contract,
            shape_bucket=(key.batch_size, key.sequence_length, key.selected_row_count),
            device_identity=self.device_identity,
        )
        started_ns = received_ns
        setup_ns = 0
        queue_ns = 0
        route = "eager-fallback"
        template_fingerprint = None
        generation = None
        fallback_reason = None
        if promotion is None:
            fallback_reason = "no-exact-template-promotion"
            selected = getattr(self.engine, "selected_last_logits_batch", None)
            if not callable(selected):
                raise RuntimeError("resident worker eager fallback is unavailable")
            raw = selected(
                [np.asarray(row, dtype=np.int64) for row in request.token_rows],
                request.selected_token_ids,
            )
            scores = torch.as_tensor(raw, dtype=torch.float32).cpu()
            self._eager_requests += 1
            self._fallback_reasons[fallback_reason] = (
                self._fallback_reasons.get(fallback_reason, 0) + 1
            )
        else:
            setup_started_ns = time.monotonic_ns()
            template = self.pool.get_or_create(
                key,
                lambda: self._create_template(key, promotion),
                estimated_template_bytes=promotion.template_bytes,
            )
            setup_ns = time.monotonic_ns() - setup_started_ns
            if time.monotonic_ns() >= request.deadline_ns:
                raise TimeoutError("resident request deadline elapsed during template setup")
            timeout_s = max(0.0, (request.deadline_ns - time.monotonic_ns()) / 1e9)
            queue_started_ns = time.monotonic_ns()
            lease = template.lease(timeout_s=timeout_s)
            queue_ns = time.monotonic_ns() - queue_started_ns
            if time.monotonic_ns() >= request.deadline_ns:
                lease.release()
                raise TimeoutError("resident request deadline elapsed while waiting for a lane")
            started_ns = time.monotonic_ns()
            with lease:
                bound = lease.execute(
                    request.token_rows,
                    request.selected_token_ids,
                    request_id=request.request_id,
                )
            scores = torch.as_tensor(bound.output, dtype=torch.float32).cpu()
            route = "resident-graph-template"
            template_fingerprint = bound.template_fingerprint
            generation = bound.generation
            self._graph_requests += 1
        if tuple(scores.shape) != (len(request.token_rows), len(request.selected_token_ids)):
            raise RuntimeError("resident worker returned a malformed selected-score matrix")
        completed_ns = time.monotonic_ns()
        self._requests += 1
        return ResidentSelectedScoreResponse(
            request_id=request.request_id,
            scores=tuple(tuple(float(value) for value in row) for row in scores.tolist()),
            route=route,
            setup_ms=setup_ns / 1e6,
            queue_ms=queue_ns / 1e6,
            execution_ms=(completed_ns - started_ns) / 1e6,
            template_fingerprint=template_fingerprint,
            generation=generation,
            fallback_reason=fallback_reason,
        )

    def _create_template(
        self,
        key: GraphTemplateKey,
        promotion: ResidentTemplatePromotion,
    ) -> GraphTemplate:
        template = self.template_factory(key, self.arena, promotion)
        if not isinstance(template, GraphTemplate):
            raise TypeError("resident worker template factory returned an invalid object")
        return template

    def inventory(self) -> dict[str, Any]:
        pool = self.pool.inventory()
        return {
            "warm_arena_keys": [self.arena.key.fingerprint],
            "arena_bytes": self.arena.resident_bytes,
            **pool,
            "request_count": self._requests,
            "graph_request_count": self._graph_requests,
            "eager_fallback_count": self._eager_requests,
            "fallback_reasons": dict(sorted(self._fallback_reasons.items())),
        }

    def close(self) -> None:
        if self._closed:
            return
        self.pool.close()
        self.arena.close()
        self._closed = True


def resident_executable_inventory() -> dict[str, Any]:
    """Aggregate process-local resident workers for host telemetry."""

    with _WORKERS_LOCK:
        inventories = [worker.inventory() for worker in tuple(_WORKERS) if not worker._closed]
    arena_keys = tuple(
        dict.fromkeys(key for inventory in inventories for key in inventory["warm_arena_keys"])
    )
    template_keys = tuple(
        dict.fromkeys(
            key for inventory in inventories for key in inventory["warm_graph_template_keys"]
        )
    )
    hits = sum(int(inventory["template_hit_count"]) for inventory in inventories)
    misses = sum(int(inventory["template_miss_count"]) for inventory in inventories)
    return {
        "warm_arena_keys": list(arena_keys),
        "warm_graph_template_keys": list(template_keys),
        "arena_bytes": sum(int(inventory["arena_bytes"]) for inventory in inventories),
        "template_bytes": sum(int(inventory["template_bytes"]) for inventory in inventories),
        "lane_count": sum(int(inventory["lane_count"]) for inventory in inventories),
        "queue_depth_by_bucket": {},
        "template_hit_rate": hits / max(1, hits + misses),
        "capture_count": sum(int(inventory["capture_count"]) for inventory in inventories),
        "replay_count": sum(int(inventory["replay_count"]) for inventory in inventories),
        "eviction_count": sum(int(inventory["eviction_count"]) for inventory in inventories),
    }


def build_dense_resident_worker(
    engine: Any,
    arena_key: ResidentQStoreArenaKey,
    promotions: tuple[ResidentTemplatePromotion, ...],
    *,
    device_identity: str,
    arena_budget_mb: float,
    template_budget_bytes: int,
) -> ResidentModelWorker:
    """Assemble the production dense-CUDA arena/template implementation."""

    from ..compiler.sciencegraph import bind_sciencegraph_model_identity

    actual_model_identity = bind_sciencegraph_model_identity(engine)
    if arena_key.model_identity != actual_model_identity:
        raise RuntimeError("resident arena key does not match the loaded QStore identity")
    actual_numerical_contract = str(getattr(engine, "numerical_contract", ""))
    if arena_key.numerical_contract != actual_numerical_contract:
        raise RuntimeError("resident arena key does not match the engine numerical contract")
    prepare_arena = getattr(engine, "prepare_resident_qstore_arena", None)
    prepare_template = getattr(engine, "prepare_rebindable_selected_last_cuda_graph", None)
    if not callable(prepare_arena) or not callable(prepare_template):
        raise TypeError("engine does not implement resident rebindable CUDA Graph execution")
    backend_arena = prepare_arena(residency_budget_mb=arena_budget_mb)
    arena = ResidentQStoreArena(
        arena_key,
        backend_arena,
        resident_bytes=int(backend_arena.resident_bytes),
    )

    def factory(
        key: GraphTemplateKey,
        owner: ResidentQStoreArena,
        promotion: ResidentTemplatePromotion,
    ) -> GraphTemplate:
        placeholder_rows = tuple(
            np.zeros(key.sequence_length, dtype=np.int64) for _ in range(key.batch_size)
        )
        placeholder_selected = tuple(range(key.selected_row_count))
        executors = tuple(
            prepare_template(
                placeholder_rows,
                placeholder_selected,
                arena=owner.backend_arena,
            )
            for _ in range(promotion.lane_count)
        )
        return GraphTemplate(
            key,
            owner,
            executors,
            template_bytes=promotion.template_bytes,
        )

    return ResidentModelWorker(
        engine,
        arena,
        GraphTemplatePool(max_template_bytes=template_budget_bytes),
        promotions,
        factory,
        device_identity=device_identity,
        estimated_template_bytes=max(
            (record.template_bytes for record in promotions),
            default=0,
        ),
    )


__all__ = [
    "ResidentModelWorker",
    "ResidentSelectedScoreRequest",
    "ResidentSelectedScoreResponse",
    "build_dense_resident_worker",
    "resident_executable_inventory",
]
