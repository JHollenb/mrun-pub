"""Byte- and entry-bounded LRU for lowered work plans."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from .ir import DenseWorkPlan, DispatchBinding, WorkTemplate
from .lowering import LoweredWorkPlan, LoweredWorkTemplate


@dataclass(frozen=True)
class WorkPlanCacheStats:
    entries: int
    bytes: int
    hits: int
    misses: int
    evictions: int


class WorkPlanCache:
    def __init__(self, *, max_entries: int = 64, max_bytes: int = 16 * 1024 * 1024) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_entries <= 0
            or max_bytes <= 0
        ):
            raise ValueError("cache bounds must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, LoweredWorkPlan] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    def get(self, key: str) -> LoweredWorkPlan | None:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return value

    def put(self, value: LoweredWorkPlan) -> bool:
        if not isinstance(value, LoweredWorkPlan):
            raise TypeError("WorkPlanCache accepts only LoweredWorkPlan entries")
        expected_key = hashlib.sha256(
            f"{value.backend}:{value.plan_fingerprint}".encode()
        ).hexdigest()
        if value.executable_key != expected_key:
            raise ValueError("WorkPlanCache requires a full-plan executable key")
        with self._lock:
            key = value.executable_key
            size = value.estimated_bytes
            if size > self.max_bytes:
                return False
            previous = self._entries.get(key)
            if previous is not None and previous != value:
                raise RuntimeError("lowered WorkPlan executable-key collision")
            if previous is not None:
                self._entries.pop(key)
                self._bytes -= previous.estimated_bytes
            self._entries[key] = value
            self._bytes += size
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.estimated_bytes
                self._evictions += 1
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> WorkPlanCacheStats:
        with self._lock:
            return WorkPlanCacheStats(
                entries=len(self._entries),
                bytes=self._bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )


class WorkTemplateCache:
    """Thread-safe byte-bounded LRU of binding-free backend schedules.

    A hit can be attached to any :class:`DispatchBinding` with the exact template
    fingerprint. Concrete plan provenance is reconstructed on every attachment and never
    retained in the reusable cache entry.
    """

    def __init__(self, *, max_entries: int = 64, max_bytes: int = 16 * 1024 * 1024) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_entries <= 0
            or max_bytes <= 0
        ):
            raise ValueError("cache bounds must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, LoweredWorkTemplate] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    def get(self, key: str) -> LoweredWorkTemplate | None:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self._misses += 1
                return None
            value.verify_integrity()
            self._entries.move_to_end(key)
            self._hits += 1
            return value

    def get_bound(
        self,
        key: str,
        template: WorkTemplate,
        binding: DispatchBinding,
    ) -> tuple[DenseWorkPlan, LoweredWorkPlan] | None:
        """Return a newly proven concrete plan/schedule pair from one cached template."""

        lowered_template = self.get(key)
        if lowered_template is None:
            return None
        plan = template.bind(binding)
        return plan, lowered_template.bind(plan)

    def put(self, value: LoweredWorkTemplate) -> bool:
        if not isinstance(value, LoweredWorkTemplate):
            raise TypeError("WorkTemplateCache accepts only LoweredWorkTemplate entries")
        with self._lock:
            value.verify_integrity()
            key = value.executable_key
            size = value.estimated_bytes
            if size > self.max_bytes:
                return False
            previous = self._entries.get(key)
            if previous is not None:
                previous.verify_integrity()
                previous_size = previous.estimated_bytes
                if previous != value:
                    raise RuntimeError("lowered-template executable-key collision")
                self._entries.pop(key)
                self._bytes -= previous_size
            self._entries[key] = value
            self._bytes += size
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.estimated_bytes
                self._evictions += 1
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> WorkPlanCacheStats:
        with self._lock:
            return WorkPlanCacheStats(
                entries=len(self._entries),
                bytes=self._bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )
