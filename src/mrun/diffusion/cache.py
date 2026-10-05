"""Disk cache for prompt embeddings — safetensors, byte-exact round trip.

One file per cache key under the cache root. safetensors preserves dtype and
raw bytes exactly (bf16 stays bf16), which is what makes a cached embed safe to
feed back into a parity-gated generation. Writes are atomic (tmp + rename) so a
killed job can never leave a torn cache entry that a later job trusts.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .phase import PromptEmbeds


class EmbedCache:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        if not key or any(c not in "0123456789abcdef" for c in key):
            raise ValueError(f"cache key must be a hex digest, got {key!r}")
        return self.root / f"{key}.safetensors"

    def save(self, embeds: PromptEmbeds) -> Path:
        from safetensors.torch import save_file

        path = self.path(embeds.key)
        tmp = path.with_suffix(".tmp")
        metadata = {"meta": json.dumps(embeds.meta, sort_keys=True)}
        save_file(dict(embeds.tensors), str(tmp), metadata=metadata)
        os.replace(tmp, path)
        return path

    def load(self, key: str) -> PromptEmbeds | None:
        from safetensors import safe_open

        from .phase import PromptEmbeds

        path = self.path(key)
        if not path.is_file():
            return None
        tensors: dict[str, Any] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            raw_meta = (handle.metadata() or {}).get("meta")
            for name in handle.keys():
                tensors[name] = handle.get_tensor(name)
        meta = json.loads(raw_meta) if raw_meta else {}
        return PromptEmbeds(key=key, tensors=tensors, meta=meta)


def _canonical_key_payload(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=repr)


@dataclass(frozen=True, slots=True)
class TrajectoryCacheEntry:
    """A process-owned checkpoint and the identities that make it reusable."""

    key: str
    checkpoint: Any = field(repr=False, compare=False)
    model_identity: str
    conditioning_key: str
    schedule_fingerprint: str
    references: tuple[str, ...] = ()
    dependency_keys: tuple[str, ...] = ()


class TrajectoryCache:
    """Content-addressed trajectory/reference cache with explicit invalidation.

    The cache deliberately stores checkpoint objects in the owning process; it
    does not turn tensors into transport payloads.  A caller must include every
    value that can change the prefix in the cache key, and may attach dependency
    keys for reference/edit invalidation.  A hit is therefore a runtime claim
    about one declared prefix, not a promise that all edits share it.
    """

    def __init__(self, *, max_entries: int = 32) -> None:
        if isinstance(max_entries, bool) or int(max_entries) <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = int(max_entries)
        self._entries: dict[str, TrajectoryCacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._invalidations = 0

    @staticmethod
    def make_key(
        *,
        model_identity: str,
        conditioning_key: str,
        schedule_fingerprint: str,
        cut_step: int,
        resolution: tuple[int, int],
        references: Iterable[str] = (),
        numerical_contract: str = "scalar-authority",
        initial_latent_fingerprint: str = "",
        trajectory_abi: str = "mrun-diffusion-trajectory-v2",
    ) -> str:
        payload = {
            "schema": "mrun-trajectory-cache-key-v2",
            "model_identity": str(model_identity),
            "conditioning_key": str(conditioning_key),
            "schedule_fingerprint": str(schedule_fingerprint),
            "cut_step": int(cut_step),
            "resolution": [int(resolution[0]), int(resolution[1])],
            "references": sorted(str(value) for value in references),
            "numerical_contract": str(numerical_contract),
            # A full trajectory prefix is a function of the initial target
            # latent as well as the prompt/reference state.  Keep an explicit
            # empty value for legacy callers so their keys cannot collide
            # with the repaired ABI.
            "initial_latent_fingerprint": str(initial_latent_fingerprint),
            "trajectory_abi": str(trajectory_abi),
        }
        return hashlib.sha256(_canonical_key_payload(payload).encode("utf-8")).hexdigest()

    def put(
        self,
        checkpoint: Any,
        *,
        key: str,
        model_identity: str,
        conditioning_key: str,
        schedule_fingerprint: str,
        references: Iterable[str] = (),
        dependency_keys: Iterable[str] = (),
    ) -> TrajectoryCacheEntry:
        if not key or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("trajectory cache key must be a hex digest")
        entry = TrajectoryCacheEntry(
            key=key,
            checkpoint=checkpoint,
            model_identity=str(model_identity),
            conditioning_key=str(conditioning_key),
            schedule_fingerprint=str(schedule_fingerprint),
            references=tuple(sorted(str(value) for value in references)),
            dependency_keys=tuple(sorted(str(value) for value in dependency_keys)),
        )
        if key not in self._entries and len(self._entries) >= self.max_entries:
            oldest = next(iter(self._entries))
            del self._entries[oldest]
            self._evictions += 1
        self._entries[key] = entry
        return entry

    def get(
        self,
        key: str,
        *,
        model_identity: str | None = None,
        conditioning_key: str | None = None,
        schedule_fingerprint: str | None = None,
    ) -> TrajectoryCacheEntry | None:
        entry = self._entries.get(str(key))
        if entry is None:
            self._misses += 1
            return None
        expected = {
            "model_identity": model_identity,
            "conditioning_key": conditioning_key,
            "schedule_fingerprint": schedule_fingerprint,
        }
        for field_name, expected_value in expected.items():
            if expected_value is not None and getattr(entry, field_name) != str(expected_value):
                self._entries.pop(str(key), None)
                self._misses += 1
                self._invalidations += 1
                return None
        self._hits += 1
        return entry

    def invalidate(self, dependency_keys: Iterable[str]) -> tuple[str, ...]:
        """Drop all entries depending on any supplied reference/edit key."""

        dependencies = {str(value) for value in dependency_keys}
        if not dependencies:
            return ()
        removed = tuple(
            key
            for key, entry in self._entries.items()
            if dependencies.intersection(entry.dependency_keys)
            or dependencies.intersection(entry.references)
        )
        for key in removed:
            self._entries.pop(key, None)
        self._invalidations += len(removed)
        return removed

    def clear(self) -> tuple[str, ...]:
        removed = tuple(self._entries)
        self._entries.clear()
        self._invalidations += len(removed)
        return removed

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "invalidations": self._invalidations,
        }


__all__ = ["EmbedCache", "TrajectoryCache", "TrajectoryCacheEntry"]
