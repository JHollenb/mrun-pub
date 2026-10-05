"""Validated warm executable inventory advertised by resident workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutableInventory:
    warm_arena_keys: tuple[str, ...] = ()
    warm_graph_template_keys: tuple[str, ...] = ()
    arena_bytes: int = 0
    template_bytes: int = 0
    lane_count: int = 0
    template_hit_rate: float = 0.0
    capture_count: int = 0
    replay_count: int = 0
    eviction_count: int = 0

    def __post_init__(self) -> None:
        for field in ("warm_arena_keys", "warm_graph_template_keys"):
            values = tuple(str(value) for value in getattr(self, field))
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"{field} must contain unique non-empty identities")
            object.__setattr__(self, field, values)
        for field in (
            "arena_bytes",
            "template_bytes",
            "lane_count",
            "capture_count",
            "replay_count",
            "eviction_count",
        ):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, value)
        if not 0 <= float(self.template_hit_rate) <= 1:
            raise ValueError("template hit rate must lie inside [0,1]")

    @classmethod
    def from_telemetry(cls, telemetry: dict[str, Any] | None) -> ExecutableInventory:
        payload = (telemetry or {}).get("executable_inventory") or {}
        return cls(
            warm_arena_keys=tuple(payload.get("warm_arena_keys") or ()),
            warm_graph_template_keys=tuple(payload.get("warm_graph_template_keys") or ()),
            arena_bytes=int(payload.get("arena_bytes") or 0),
            template_bytes=int(payload.get("template_bytes") or 0),
            lane_count=int(payload.get("lane_count") or 0),
            template_hit_rate=float(payload.get("template_hit_rate") or 0.0),
            capture_count=int(payload.get("capture_count") or 0),
            replay_count=int(payload.get("replay_count") or 0),
            eviction_count=int(payload.get("eviction_count") or 0),
        )

    def affinity(self, *, arena_key: str | None, template_key: str | None) -> str:
        if template_key and template_key in self.warm_graph_template_keys:
            return "exact-warm-template"
        if arena_key and arena_key in self.warm_arena_keys:
            return "warm-arena"
        return "cold"


__all__ = ["ExecutableInventory"]
