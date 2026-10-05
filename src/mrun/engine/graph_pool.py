"""Resident QStore arenas and fenced, rebindable graph-template lanes.

The pool is backend-neutral.  CUDA executors provide ``rebind`` and generation-checked
``execute`` methods; deterministic CPU fakes exercise the same ownership protocol.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any

GRAPH_TEMPLATE_KEY_SCHEMA = "mrun-graph-template-key-v1"
RESIDENT_ARENA_KEY_SCHEMA = "mrun-resident-qstore-arena-key-v1"


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _nonempty(value: object, field: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


@dataclass(frozen=True)
class ResidentQStoreArenaKey:
    model_identity: str
    numerical_contract: str
    device_identity: str
    runtime_distribution_sha256: str
    activation_dtype: str
    weight_dtype: str
    schema: str = RESIDENT_ARENA_KEY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESIDENT_ARENA_KEY_SCHEMA:
            raise ValueError("unsupported resident arena key schema")
        for field in (
            "model_identity",
            "numerical_contract",
            "device_identity",
            "runtime_distribution_sha256",
            "activation_dtype",
            "weight_dtype",
        ):
            object.__setattr__(self, field, _nonempty(getattr(self, field), field))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True)
class GraphTemplateKey:
    arena_fingerprint: str
    execution_mode: str
    output_contract: str
    batch_size: int
    sequence_length: int
    selected_row_count: int
    state_contract: str = "stateless"
    kernel_contract: str = "established"
    schema: str = GRAPH_TEMPLATE_KEY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GRAPH_TEMPLATE_KEY_SCHEMA:
            raise ValueError("unsupported graph template key schema")
        for field in (
            "arena_fingerprint",
            "execution_mode",
            "output_contract",
            "state_contract",
            "kernel_contract",
        ):
            object.__setattr__(self, field, _nonempty(getattr(self, field), field))
        for field in ("batch_size", "sequence_length", "selected_row_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{field} must be positive")
            object.__setattr__(self, field, int(value))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


class ResidentQStoreArena:
    """High-level authority over one backend arena and its template references."""

    def __init__(
        self,
        key: ResidentQStoreArenaKey,
        backend_arena: Any,
        *,
        resident_bytes: int,
    ) -> None:
        if resident_bytes <= 0:
            raise ValueError("resident arena bytes must be positive")
        self.key = key
        self.backend_arena = backend_arena
        self.resident_bytes = int(resident_bytes)
        self._template_references = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    def template_references(self) -> int:
        with self._lock:
            return self._template_references

    def retain_template(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("resident arena is closed")
            verify = getattr(self.backend_arena, "verify_stable_addresses", None)
            if callable(verify):
                verify()
            self._template_references += 1

    def release_template(self) -> None:
        with self._lock:
            if self._template_references <= 0:
                raise RuntimeError("resident arena template reference underflow")
            self._template_references -= 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._template_references:
                raise RuntimeError("cannot close resident arena with live templates")
            close = getattr(self.backend_arena, "close", None)
            if callable(close):
                close()
            self._closed = True


@dataclass(frozen=True)
class BoundGraphResult:
    request_id: str
    generation: int
    template_fingerprint: str
    lane_id: int
    output: Any


class _LaneLease(AbstractContextManager["_LaneLease"]):
    def __init__(self, template: GraphTemplate, lane: _GraphLane) -> None:
        self._template = template
        self._lane = lane
        self._released = False

    def execute(
        self,
        ids_list: Sequence[Any],
        token_ids: Sequence[int],
        *,
        request_id: str,
    ) -> BoundGraphResult:
        if self._released:
            raise RuntimeError("graph lane lease is released")
        epoch = int(self._lane.executor.rebind(ids_list, token_ids, request_id=request_id))
        output = self._lane.executor.execute(
            expected_generation=epoch,
            request_id=request_id,
        )
        if int(getattr(self._lane.executor, "generation", -1)) != epoch:
            raise RuntimeError("graph lane generation changed during replay")
        self._lane.replays += 1
        return BoundGraphResult(
            request_id=request_id,
            generation=epoch,
            template_fingerprint=self._template.key.fingerprint,
            lane_id=self._lane.lane_id,
            output=output,
        )

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._template._release(self._lane)  # noqa: SLF001 - lease ownership protocol

    def __exit__(self, *exc: object) -> None:
        self.release()


@dataclass
class _GraphLane:
    lane_id: int
    executor: Any
    leased: bool = False
    replays: int = 0


class GraphTemplate:
    """One shape template with isolated mutable buffers per lane."""

    def __init__(
        self,
        key: GraphTemplateKey,
        arena: ResidentQStoreArena,
        executors: Sequence[Any],
        *,
        template_bytes: int,
    ) -> None:
        if key.arena_fingerprint != arena.key.fingerprint:
            raise ValueError("graph template key does not bind its resident arena")
        if not executors:
            raise ValueError("graph template requires at least one lane")
        if template_bytes < 0:
            raise ValueError("template bytes must be non-negative")
        self.key = key
        self.arena = arena
        self.template_bytes = int(template_bytes)
        self._lanes = [_GraphLane(index, executor) for index, executor in enumerate(executors)]
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._lease_count = 0
        self._waiters = 0
        self.arena.retain_template()

    @property
    def lane_count(self) -> int:
        return len(self._lanes)

    @property
    def in_flight(self) -> int:
        with self._condition:
            return sum(lane.leased for lane in self._lanes)

    def lease(self, *, timeout_s: float | None = None) -> _LaneLease:
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("lane timeout must be non-negative")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        waiting = False
        with self._condition:
            while True:
                if self._closed:
                    if waiting:
                        self._waiters -= 1
                    raise RuntimeError("graph template is closed")
                for lane in self._lanes:
                    if not lane.leased:
                        if waiting:
                            self._waiters -= 1
                        lane.leased = True
                        self._lease_count += 1
                        return _LaneLease(self, lane)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if waiting:
                            self._waiters -= 1
                        raise TimeoutError("graph template lane acquisition timed out")
                    if not waiting:
                        self._waiters += 1
                        waiting = True
                    self._condition.wait(remaining)
                else:
                    if not waiting:
                        self._waiters += 1
                        waiting = True
                    self._condition.wait()

    def _release(self, lane: _GraphLane) -> None:
        with self._condition:
            if not lane.leased:
                raise RuntimeError("graph template lane was released twice")
            lane.leased = False
            self._condition.notify()

    def evidence(self) -> dict[str, Any]:
        with self._condition:
            return {
                "template_fingerprint": self.key.fingerprint,
                "arena_fingerprint": self.arena.key.fingerprint,
                "template_bytes": self.template_bytes,
                "lane_count": len(self._lanes),
                "in_flight": sum(lane.leased for lane in self._lanes),
                "queue_depth": self._waiters,
                "lease_count": self._lease_count,
                "replay_count": sum(lane.replays for lane in self._lanes),
                "closed": self._closed,
            }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if any(lane.leased for lane in self._lanes):
                raise RuntimeError("cannot close a graph template with leased lanes")
            for lane in self._lanes:
                close = getattr(lane.executor, "close", None)
                if callable(close):
                    close()
            self._closed = True
            self._condition.notify_all()
        self.arena.release_template()


class GraphTemplatePool:
    """Process-local LRU with exact identity, byte admission, and safe eviction."""

    def __init__(self, *, max_template_bytes: int, max_templates: int = 64) -> None:
        if max_template_bytes <= 0 or max_templates <= 0:
            raise ValueError("graph template pool budgets must be positive")
        self.max_template_bytes = int(max_template_bytes)
        self.max_templates = int(max_templates)
        self._templates: OrderedDict[str, GraphTemplate] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _evict_until(self, additional_bytes: int) -> None:
        while (
            len(self._templates) >= self.max_templates
            or self.template_bytes + additional_bytes > self.max_template_bytes
        ):
            victim_key = next(
                (key for key, value in self._templates.items() if value.in_flight == 0),
                None,
            )
            if victim_key is None:
                raise MemoryError("graph template pool has no evictable capacity")
            victim = self._templates.pop(victim_key)
            victim.close()
            self._evictions += 1

    @property
    def template_bytes(self) -> int:
        return sum(template.template_bytes for template in self._templates.values())

    def get_or_create(
        self,
        key: GraphTemplateKey,
        factory: Callable[[], GraphTemplate],
        *,
        estimated_template_bytes: int,
    ) -> GraphTemplate:
        if estimated_template_bytes < 0:
            raise ValueError("estimated template bytes must be non-negative")
        fingerprint = key.fingerprint
        with self._lock:
            existing = self._templates.get(fingerprint)
            if existing is not None:
                self._templates.move_to_end(fingerprint)
                self._hits += 1
                return existing
            self._misses += 1
            if estimated_template_bytes > self.max_template_bytes:
                raise MemoryError("one graph template exceeds the complete pool budget")
            self._evict_until(estimated_template_bytes)
            created = factory()
            if created.key != key:
                created.close()
                raise RuntimeError("graph template factory returned the wrong identity")
            if created.template_bytes > self.max_template_bytes:
                created.close()
                raise MemoryError("created graph template exceeds the complete pool budget")
            self._evict_until(created.template_bytes)
            self._templates[fingerprint] = created
            return created

    def inventory(self) -> dict[str, Any]:
        with self._lock:
            return {
                "warm_graph_template_keys": list(self._templates),
                "template_bytes": self.template_bytes,
                "template_count": len(self._templates),
                "lane_count": sum(value.lane_count for value in self._templates.values()),
                "queue_depth_by_bucket": {
                    key: value.evidence()["queue_depth"]
                    for key, value in self._templates.items()
                    if value.evidence()["queue_depth"]
                },
                "template_hit_count": self._hits,
                "template_miss_count": self._misses,
                "template_hit_rate": self._hits / max(1, self._hits + self._misses),
                "capture_count": self._misses,
                "replay_count": sum(
                    value.evidence()["replay_count"] for value in self._templates.values()
                ),
                "eviction_count": self._evictions,
            }

    def close(self) -> None:
        with self._lock:
            if any(value.in_flight for value in self._templates.values()):
                raise RuntimeError("cannot close graph pool with in-flight lanes")
            for template in self._templates.values():
                template.close()
            self._templates.clear()


__all__ = [
    "BoundGraphResult",
    "GRAPH_TEMPLATE_KEY_SCHEMA",
    "GraphTemplate",
    "GraphTemplateKey",
    "GraphTemplatePool",
    "RESIDENT_ARENA_KEY_SCHEMA",
    "ResidentQStoreArena",
    "ResidentQStoreArenaKey",
]
