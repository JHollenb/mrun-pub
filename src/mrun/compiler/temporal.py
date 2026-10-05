"""Reference temporal-state DAG with fenced provisional commit and rollback."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalStateHandle:
    state_id: str
    slot: int
    epoch: int
    committed_length: int
    capacity: int
    prefix_state_id: str | None = None


@dataclass(frozen=True)
class ProvisionalStateDelta:
    state_id: str
    parent_epoch: int
    tokens: tuple[int, ...]


class TemporalStateArena:
    """Fixed-slot reference authority used to gate native captured-decode implementations."""

    def __init__(self, *, slots: int, capacity: int) -> None:
        if slots <= 0 or capacity <= 0:
            raise ValueError("temporal arena geometry must be positive")
        self.slots = int(slots)
        self.capacity = int(capacity)
        self._free = list(range(slots))
        self._states: dict[str, dict[str, object]] = {}
        self._lock = threading.RLock()
        self._peak_active = 0
        self._commits = 0
        self._rollbacks = 0

    def create(self, state_id: str, prefix_tokens: tuple[int, ...] = ()) -> TemporalStateHandle:
        if not state_id:
            raise ValueError("state ID must be non-empty")
        prefix = tuple(int(token) for token in prefix_tokens)
        with self._lock:
            if state_id in self._states:
                raise ValueError("state ID already exists")
            if not self._free:
                raise MemoryError("temporal state arena has no free slot")
            if len(prefix) > self.capacity:
                raise OverflowError("prefix exceeds temporal state capacity")
            slot = self._free.pop(0)
            self._states[state_id] = {
                "slot": slot,
                "epoch": 0,
                "tokens": prefix,
                "prefix_state_id": None,
            }
            self._peak_active = max(self._peak_active, len(self._states))
            return self.observe(state_id)

    def fork(self, source_state_id: str, target_state_id: str) -> TemporalStateHandle:
        with self._lock:
            source = self._require(source_state_id)
            target = self.create(target_state_id, source["tokens"])  # type: ignore[arg-type]
            self._states[target_state_id]["prefix_state_id"] = source_state_id
            return TemporalStateHandle(
                state_id=target.state_id,
                slot=target.slot,
                epoch=target.epoch,
                committed_length=target.committed_length,
                capacity=target.capacity,
                prefix_state_id=source_state_id,
            )

    def observe(self, state_id: str) -> TemporalStateHandle:
        with self._lock:
            state = self._require(state_id)
            return TemporalStateHandle(
                state_id=state_id,
                slot=int(state["slot"]),
                epoch=int(state["epoch"]),
                committed_length=len(state["tokens"]),  # type: ignore[arg-type]
                capacity=self.capacity,
                prefix_state_id=state["prefix_state_id"],  # type: ignore[arg-type]
            )

    def begin(self, state_id: str, tokens: tuple[int, ...]) -> ProvisionalStateDelta:
        provisional = tuple(int(token) for token in tokens)
        if not provisional:
            raise ValueError("provisional state delta must contain at least one token")
        with self._lock:
            state = self._require(state_id)
            if len(state["tokens"]) + len(provisional) > self.capacity:  # type: ignore[arg-type]
                raise OverflowError("provisional state delta exceeds arena capacity")
            return ProvisionalStateDelta(
                state_id=state_id,
                parent_epoch=int(state["epoch"]),
                tokens=provisional,
            )

    def commit(self, delta: ProvisionalStateDelta, accepted_count: int) -> TemporalStateHandle:
        if accepted_count < 0 or accepted_count > len(delta.tokens):
            raise ValueError("accepted count lies outside the provisional delta")
        with self._lock:
            state = self._require(delta.state_id)
            if int(state["epoch"]) != delta.parent_epoch:
                raise RuntimeError("provisional state delta parent epoch is stale")
            committed = state["tokens"]  # type: ignore[assignment]
            state["tokens"] = (*committed, *delta.tokens[:accepted_count])
            state["epoch"] = int(state["epoch"]) + 1
            self._commits += 1
            return self.observe(delta.state_id)

    def rollback(self, delta: ProvisionalStateDelta) -> TemporalStateHandle:
        with self._lock:
            state = self._require(delta.state_id)
            if int(state["epoch"]) != delta.parent_epoch:
                raise RuntimeError("cannot roll back a stale provisional state delta")
            self._rollbacks += 1
            return self.observe(delta.state_id)

    def release(self, state_id: str) -> None:
        with self._lock:
            state = self._require(state_id)
            del self._states[state_id]
            self._free.append(int(state["slot"]))
            self._free.sort()

    def tokens(self, state_id: str) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._require(state_id)["tokens"])  # type: ignore[arg-type]

    def evidence(self) -> dict[str, int]:
        with self._lock:
            return {
                "slot_count": self.slots,
                "capacity_per_slot": self.capacity,
                "active_slots": len(self._states),
                "peak_active_slots": self._peak_active,
                "allocation_count": self.slots,
                "commit_count": self._commits,
                "rollback_count": self._rollbacks,
            }

    def _require(self, state_id: str) -> dict[str, object]:
        try:
            return self._states[state_id]
        except KeyError as exc:
            raise KeyError(f"unknown temporal state {state_id!r}") from exc


__all__ = [
    "ProvisionalStateDelta",
    "TemporalStateArena",
    "TemporalStateHandle",
]
