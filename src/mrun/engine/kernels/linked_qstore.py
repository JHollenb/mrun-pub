"""Base-pinned linked/overlay QStore views.

``OverlayQStore`` is the cold/debug implementation of the linked-image ABI.
It delegates each logical block to either an immutable base store or an
extension store and exposes the established QStore duck-typed interface.
There is intentionally no per-forward residual arithmetic.  The resolved
view pre-binds the page table once so callers retain the ordinary QStore ABI.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class LinkedQStoreError(RuntimeError):
    """Raised when two stores cannot be linked safely."""


def _blocks(store: Any) -> Mapping[str, Any]:
    blocks = getattr(store, "blocks", {})
    return blocks if isinstance(blocks, Mapping) else {}


def _physical_name(store: Any, name: str) -> str:
    """Resolve one logical block to its physical allocation before serving it."""

    key = str(name)
    table = _blocks(store)
    if not table:
        return key
    seen: set[str] = set()
    while True:
        if key in seen:
            raise LinkedQStoreError(f"cyclic QStore alias at {name!r}")
        seen.add(key)
        raw = table.get(key)
        if not isinstance(raw, Mapping):
            raise LinkedQStoreError(f"QStore has no block {key!r}")
        alias = raw.get("alias")
        if alias is None:
            return key
        key = str(alias)


def _descriptor(store: Any, physical: str) -> Mapping[str, Any] | None:
    raw = _blocks(store).get(physical)
    return raw if isinstance(raw, Mapping) else None


def _validate_compatible_descriptors(
    logical: str,
    base: Any,
    extension: Any,
    base_physical: str,
    extension_physical: str,
) -> None:
    """Reject a route whose target shape or storage kind cannot replace the base."""

    base_block = _descriptor(base, base_physical)
    extension_block = _descriptor(extension, extension_physical)
    if base_block is None or extension_block is None:
        return
    for field in ("kind", "shape"):
        if base_block.get(field) != extension_block.get(field):
            raise LinkedQStoreError(
                f"overlay block {logical!r} changes {field}: "
                f"{base_block.get(field)!r} != {extension_block.get(field)!r}"
            )


@dataclass(frozen=True, slots=True)
class _ResolvedRoute:
    provider: Any
    physical: str
    fp32: Callable[..., torch.Tensor]
    weight: Callable[..., torch.Tensor]
    matmul: Callable[..., torch.Tensor]
    matmul_row_stable: Callable[..., torch.Tensor] | None
    embed_rows: Callable[..., torch.Tensor]
    row_blocks: Callable[..., Iterable[tuple[int, int, torch.Tensor]]]


class OverlayQStore:
    """A base-pinned, immutable logical overlay over two QStore-like stores.

    The extension store must contain the complete encoded block for every name
    in ``overlay_names``.  Calls for all other names go to the base.  This is a
    deliberate cold/debug path: each operation pays one Python provider
    dispatch.  The hot linked-image path should resolve the same mapping into
    ordinary contiguous QStore pages before inference begins.
    """

    def __init__(
        self,
        base: Any,
        extension: Any,
        overlay_names: Iterable[str],
        *,
        image_id: str,
        close_underlying: bool = True,
    ) -> None:
        self.base = base
        self.extension = extension
        self.overlay_names = frozenset(str(name) for name in overlay_names)
        self.image_id = str(image_id)
        self.close_underlying = bool(close_underlying)
        self._closed = False

        base_identity = getattr(base, "store_identity", {}) or {}
        extension_identity = getattr(extension, "store_identity", {}) or {}
        base_source = getattr(base, "source_checkpoint_sha256", None)
        extension_source = getattr(extension, "source_checkpoint_sha256", None)
        if base_source is not None and extension_source is not None:
            # The source checkpoints may differ by design, but the extension
            # store must advertise the base it was linked against when it has
            # that field.  Do not infer compatibility from tensor shapes.
            declared_base = extension_identity.get("base_source_checkpoint_sha256")
            if declared_base is not None and declared_base != base_source:
                raise LinkedQStoreError(
                    f"extension base identity {declared_base!r} does not match "
                    f"resident base {base_source!r}"
                )
        for name in sorted(self.overlay_names):
            if not bool(extension.has(name)):
                raise LinkedQStoreError(f"extension store has no declared overlay block {name!r}")

        self.device = getattr(base, "device", "cpu")
        self.cfg = getattr(base, "cfg", None)
        self.man = dict(getattr(base, "man", {}) or {})
        self.man["linked_image_id"] = self.image_id
        self.man["base_store_identity"] = base_identity
        self.man["extension_store_identity"] = extension_identity
        self.man["dtype"] = self.man.get("dtype", getattr(base, "storage_dtype", None))
        self.storage_dtype = getattr(base, "storage_dtype", None)
        self.compute_dtype = getattr(base, "compute_dtype", None)
        self.max_block_bytes = max(
            int(getattr(base, "max_block_bytes", 0)),
            int(getattr(extension, "max_block_bytes", 0)),
        )
        self.directory = Path(getattr(base, "directory", "."))
        self.vocab = getattr(base, "vocab", None)
        self.source_checkpoint_sha256 = base_source
        self.derived_store_sha256 = f"linked:{self.image_id}"
        provider_content_verified = bool(
            base_identity.get("content_identity_verified", False)
            and extension_identity.get("content_identity_verified", False)
        )
        # The overlay view is not itself a content-addressed QStore manifest. Keep the
        # provider result as evidence, but do not promote the composed view to a verified
        # identity until a linker emits and verifies a composite certificate.
        self.store_identity = {
            "linked_image_id": self.image_id,
            "base_store_identity": base_identity,
            "extension_store_identity": extension_identity,
            "overlay_blocks": sorted(self.overlay_names),
            "provider_content_identity_verified": provider_content_verified,
            "content_identity_verified": False,
            "identity_status": "linked-overlay-not-content-addressed",
        }

    def _store(self, name: str) -> Any:
        return self.extension if str(name) in self.overlay_names else self.base

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("linked QStore is closed")

    def has(self, name: str) -> bool:
        self._check_open()
        return bool(self._store(name).has(name))

    def fp32(self, name: str) -> torch.Tensor:
        self._check_open()
        return self._store(name).fp32(name)

    def weight(self, name: str) -> torch.Tensor:
        self._check_open()
        return self._store(name).weight(name)

    def matmul(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self._check_open()
        return self._store(name).matmul(name, value)

    def matmul_row_stable(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self._check_open()
        provider = self._store(name)
        method = getattr(provider, "matmul_row_stable", None)
        if not callable(method):
            raise LinkedQStoreError(f"provider has no matmul_row_stable for {name!r}")
        return method(name, value)

    def embed_rows(self, name: str, ids: np.ndarray | torch.Tensor) -> torch.Tensor:
        self._check_open()
        return self._store(name).embed_rows(name, ids)

    def row_blocks(self, name: str, bs: int = 8192) -> Iterator[tuple[int, int, torch.Tensor]]:
        self._check_open()
        yield from self._store(name).row_blocks(name, bs=bs)

    def set_cache_budget(self, cache_mb: float) -> None:
        self._check_open()
        for provider in (self.base, self.extension):
            setter = getattr(provider, "set_cache_budget", None)
            if callable(setter):
                setter(cache_mb)

    def snapshot(self) -> dict[str, Any]:
        self._check_open()
        snapshots = {}
        for role, provider in (("base", self.base), ("extension", self.extension)):
            snapshot = getattr(provider, "snapshot", None)
            snapshots[role] = snapshot() if callable(snapshot) else None
        return {
            "schema": "linked-qstore-overlay-v1",
            "image_id": self.image_id,
            "overlay_blocks": sorted(self.overlay_names),
            "providers": snapshots,
            "store_identity": self.store_identity,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.close_underlying:
            self.extension.close()
            if self.extension is not self.base:
                self.base.close()

    def __enter__(self) -> OverlayQStore:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class ResolvedLinkedQStore:
    """Hot base-plus-extension QStore view with a pre-resolved logical page table.

    ``OverlayQStore`` is retained as a deliberately simple cold/debug reference.  This
    implementation resolves every logical name, including alias closure, once during
    construction and stores bound provider methods in a route table.  The forward loop
    therefore pays neither the overlay membership branch nor QStore alias walking.  A
    caller may give it a sparse extension QStore: all non-overlaid pages continue to be
    served from the resident base without copying them into a second image.

    The view is intentionally QStore-shaped rather than a new forward kernel.  It keeps
    the numerical contract of the selected provider, exposes the dense CUDA compact-page
    hooks when present, and makes no residual arithmetic in the forward path.
    """

    def __init__(
        self,
        base: Any,
        extension: Any,
        overlay_names: Iterable[str],
        *,
        image_id: str,
        close_underlying: bool = True,
        cache_mb: float | None = None,
    ) -> None:
        self.base = base
        self.extension = extension
        self.overlay_names = frozenset(str(name) for name in overlay_names)
        if any(not name for name in self.overlay_names):
            raise LinkedQStoreError("overlay block names must be non-empty")
        self.image_id = str(image_id)
        if not self.image_id:
            raise LinkedQStoreError("linked extension ID must be non-empty")
        self.close_underlying = bool(close_underlying)
        self._closed = False

        self._validate_store_contracts()
        self._routes = self._build_routes()
        if not self._routes:
            raise LinkedQStoreError("linked QStore has no logical blocks")
        self._route_names = frozenset(self._routes)
        self._provider_cache_bytes = self._estimate_provider_cache_bytes()

        base_identity = dict(getattr(base, "store_identity", {}) or {})
        extension_identity = dict(getattr(extension, "store_identity", {}) or {})
        self.device = getattr(base, "device", "cpu")
        self.cfg = getattr(base, "cfg", None)
        self.storage_dtype = getattr(base, "storage_dtype", None)
        self.compute_dtype = getattr(base, "compute_dtype", None)
        self.max_block_bytes = max(
            int(getattr(base, "max_block_bytes", 0)),
            int(getattr(extension, "max_block_bytes", 0)),
        )
        base_directory = getattr(base, "directory", ".")
        self.directory = Path(base_directory)
        self.vocab = getattr(base, "vocab", None)
        self.source_checkpoint_sha256 = getattr(base, "source_checkpoint_sha256", None)
        base_derived = getattr(base, "derived_store_sha256", None)
        extension_derived = getattr(extension, "derived_store_sha256", None)
        self.derived_store_sha256 = (
            f"linked:{self.image_id}:{base_derived or 'unknown'}:{extension_derived or 'unknown'}"
        )
        provider_content_verified = bool(
            base_identity.get("content_identity_verified", False)
            and extension_identity.get("content_identity_verified", False)
        )
        # A composed route table has no derived-file certificate of its own. Providers are
        # guarded individually, but the view remains exploratory until a composite image
        # manifest binds the resolved block map.
        self.content_identity_verified = False
        self.identity_status = (
            "linked-resolved-route-providers-verified"
            if provider_content_verified
            else "linked-resolved-route-provider-status"
        )
        self.store_identity = {
            "linked_image_id": self.image_id,
            "representation": "pre_resolved_qstore_view",
            "base_store_identity": base_identity,
            "extension_store_identity": extension_identity,
            "overlay_blocks": sorted(self.overlay_names),
            "route_count": len(self._routes),
            "base_route_count": sum(
                route.provider is self.base for route in self._routes.values()
            ),
            "extension_route_count": sum(
                route.provider is self.extension for route in self._routes.values()
            ),
            "provider_content_identity_verified": provider_content_verified,
            "content_identity_verified": self.content_identity_verified,
            "identity_status": self.identity_status,
        }

        base_manifest = dict(getattr(base, "man", {}) or {})
        self.blocks = {
            logical: dict(_descriptor(route.provider, route.physical) or {})
            for logical, route in self._routes.items()
        }
        base_manifest["blocks"] = self.blocks
        linked_image = dict(
            (getattr(extension, "man", {}) or {}).get("linked_image", {}) or {}
        )
        linked_image.update(
            {
                "schema": "mrun-linked-qstore-image-v1",
                "extension_id": self.image_id,
                "representation": "pre_resolved_qstore_view",
                "overlay_blocks": sorted(self.overlay_names),
                "runtime": {
                    **dict(linked_image.get("runtime", {}) or {}),
                    "extra_forward_ops": 0,
                    "page_table": "resolved_before_forward",
                    "shared_base_pages": True,
                    "kv_policy": "image_identity_required",
                },
            }
        )
        base_manifest["linked_image"] = linked_image
        self.man = base_manifest
        self._cache_budget_mb: float | None = None
        if cache_mb is not None:
            self.set_cache_budget(cache_mb)

    def _validate_store_contracts(self) -> None:
        base_cfg = getattr(self.base, "cfg", None)
        extension_cfg = getattr(self.extension, "cfg", None)
        if isinstance(base_cfg, Mapping) and isinstance(extension_cfg, Mapping):
            if dict(base_cfg) != dict(extension_cfg):
                raise LinkedQStoreError("base and extension QStore configs do not match")
        for field in ("storage_dtype", "compute_dtype", "device"):
            base_value = getattr(self.base, field, None)
            extension_value = getattr(self.extension, field, None)
            if base_value is not None and extension_value is not None and str(base_value) != str(
                extension_value
            ):
                raise LinkedQStoreError(
                    f"base and extension QStore {field} do not match: "
                    f"{base_value!r} != {extension_value!r}"
                )

        base_source = getattr(self.base, "source_checkpoint_sha256", None)
        extension_identity = getattr(self.extension, "store_identity", {}) or {}
        extension_manifest = getattr(self.extension, "man", {}) or {}
        linked_image = extension_manifest.get("linked_image", {})
        declared_base_source = extension_identity.get("base_source_checkpoint_sha256")
        if isinstance(linked_image, Mapping):
            declared_base_source = linked_image.get(
                "base_source_provenance_sha256", declared_base_source
            )
        if base_source is not None and declared_base_source is not None:
            if str(base_source) != str(declared_base_source):
                raise LinkedQStoreError(
                    f"extension base identity {declared_base_source!r} does not match "
                    f"resident base {base_source!r}"
                )

        for name in sorted(self.overlay_names):
            if not bool(self.extension.has(name)):
                # An alias can be named by its physical target in a sparse extension,
                # but it must still be present under that exact logical name.
                raise LinkedQStoreError(
                    f"extension store has no declared overlay block {name!r}"
                )

    def _build_routes(self) -> dict[str, _ResolvedRoute]:
        base_blocks = _blocks(self.base)
        logical_names = set(str(name) for name in base_blocks)
        logical_names.update(self.overlay_names)
        routes: dict[str, _ResolvedRoute] = {}

        for logical in sorted(logical_names):
            base_has = bool(self.base.has(logical))
            base_physical = _physical_name(self.base, logical) if base_has else None
            extension_physical: str | None = None
            provider = self.base
            physical = base_physical
            if logical in self.overlay_names and bool(self.extension.has(logical)):
                provider = self.extension
                extension_physical = _physical_name(self.extension, logical)
                physical = extension_physical
            elif (
                base_physical is not None
                and base_physical in self.overlay_names
                and bool(self.extension.has(base_physical))
            ):
                # This is the important tied-readout case: base ``lm_head`` is an alias
                # of ``embed``, so an overlaid ``embed`` must also route ``lm_head``.
                provider = self.extension
                extension_physical = _physical_name(self.extension, base_physical)
                physical = extension_physical

            if physical is None:
                continue
            if provider is self.extension and base_physical is not None:
                _validate_compatible_descriptors(
                    logical,
                    self.base,
                    self.extension,
                    base_physical,
                    extension_physical or physical,
                )
            routes[logical] = _ResolvedRoute(
                provider=provider,
                physical=str(physical),
                fp32=provider.fp32,
                weight=provider.weight,
                matmul=provider.matmul,
                matmul_row_stable=getattr(provider, "matmul_row_stable", None),
                embed_rows=provider.embed_rows,
                row_blocks=provider.row_blocks,
            )
        return routes

    def _estimate_provider_cache_bytes(self) -> dict[int, tuple[Any, int]]:
        seen: set[tuple[int, str]] = set()
        estimates: dict[int, tuple[Any, int]] = {}
        for route in self._routes.values():
            identity = (id(route.provider), route.physical)
            if identity in seen:
                continue
            seen.add(identity)
            provider, total = estimates.get(id(route.provider), (route.provider, 0))
            block = _descriptor(provider, route.physical)
            if block is None or block.get("kind") not in {"qrow", "f32row"}:
                estimates[id(route.provider)] = (provider, total)
                continue
            shape = block.get("shape", ())
            try:
                elements = int(np.prod(tuple(int(value) for value in shape)))
            except (TypeError, ValueError):
                elements = 0
            dtype = getattr(provider, "compute_dtype", torch.float32) or torch.float32
            try:
                element_bytes = int(torch.empty((), dtype=dtype).element_size())
            except (TypeError, RuntimeError):
                element_bytes = 4
            estimates[id(provider)] = (provider, total + elements * element_bytes)
        return estimates

    def _route(self, name: str) -> _ResolvedRoute:
        self._check_open()
        try:
            return self._routes[str(name)]
        except KeyError as exc:
            raise KeyError(f"linked QStore has no block {name!r}") from exc

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("linked QStore is closed")

    def has(self, name: str) -> bool:
        self._check_open()
        return str(name) in self._route_names

    def fp32(self, name: str) -> torch.Tensor:
        route = self._route(name)
        return route.fp32(route.physical)

    def weight(self, name: str) -> torch.Tensor:
        route = self._route(name)
        return route.weight(route.physical)

    def matmul(self, name: str, value: torch.Tensor) -> torch.Tensor:
        route = self._route(name)
        return route.matmul(route.physical, value)

    def matmul_row_stable(self, name: str, value: torch.Tensor) -> torch.Tensor:
        route = self._route(name)
        if route.matmul_row_stable is None:
            raise LinkedQStoreError(f"provider has no matmul_row_stable for {name!r}")
        return route.matmul_row_stable(route.physical, value)

    def embed_rows(self, name: str, ids: np.ndarray | torch.Tensor) -> torch.Tensor:
        route = self._route(name)
        return route.embed_rows(route.physical, ids)

    def row_blocks(self, name: str, bs: int = 8192) -> Iterator[tuple[int, int, torch.Tensor]]:
        route = self._route(name)
        yield from route.row_blocks(route.physical, bs=bs)

    def compact_page(self, name: str, *args: Any, **kwargs: Any) -> Any:
        route = self._route(name)
        method = getattr(route.provider, "compact_page", None)
        if not callable(method):
            raise LinkedQStoreError(f"provider has no compact_page for {name!r}")
        return method(route.physical, *args, **kwargs)

    def fused_swiglu(
        self,
        gate_name: str,
        up_name: str,
        activations: torch.Tensor,
    ) -> Any:
        gate = self._route(gate_name)
        up = self._route(up_name)
        method = getattr(gate.provider, "fused_swiglu", None)
        if gate.provider is up.provider and callable(method):
            return method(gate.physical, up.physical, activations)
        # Mixed-provider pairs are uncommon for a selected extension.  Keep correctness
        # by falling back to two already-resolved projections; callers can inspect the
        # route metadata rather than silently assuming the fused CUDA path was used.
        output = torch.nn.functional.silu(self.matmul(gate_name, activations)) * self.matmul(
            up_name, activations
        )
        return output, "linked-mixed-provider-reference"

    def selected_rows_fp32(self, name: str, ids: np.ndarray | torch.Tensor) -> torch.Tensor:
        route = self._route(name)
        method = getattr(route.provider, "selected_rows_fp32", None)
        if not callable(method):
            raise LinkedQStoreError(f"provider has no selected_rows_fp32 for {name!r}")
        return method(route.physical, ids)

    def prepare_resident_exact_head(self, name: str = "lm_head", **kwargs: Any) -> Any:
        route = self._route(name)
        method = getattr(route.provider, "prepare_resident_exact_head", None)
        if not callable(method):
            raise LinkedQStoreError(f"provider has no resident exact head for {name!r}")
        return method(route.physical, **kwargs)

    def resident_exact_head_fp32(self, name: str = "lm_head") -> Any:
        route = self._route(name)
        method = getattr(route.provider, "resident_exact_head_fp32", None)
        if not callable(method):
            return None
        return method(route.physical)

    def set_cache_budget(self, cache_mb: float) -> None:
        self._check_open()
        budget = max(0.0, float(cache_mb))
        total_bytes = sum(total for _, total in self._provider_cache_bytes.values())
        providers_seen: set[int] = set()
        self._provider_cache_budgets_mb: dict[str, float] = {}
        for role, provider in (("base", self.base), ("extension", self.extension)):
            record = self._provider_cache_bytes.get(id(provider))
            if record is None:
                continue
            _, provider_bytes = record
            provider_id = id(provider)
            if provider_id in providers_seen:
                continue
            providers_seen.add(provider_id)
            share = budget * provider_bytes / total_bytes if total_bytes else 0.0
            setter = getattr(provider, "set_cache_budget", None)
            if callable(setter):
                setter(share)
            self._provider_cache_budgets_mb[role] = share
        self._cache_budget_mb = budget

    def assert_content_identity_unchanged(self) -> None:
        self._check_open()
        seen: set[int] = set()
        for provider in (self.base, self.extension):
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            checker = getattr(provider, "assert_content_identity_unchanged", None)
            if callable(checker):
                checker()

    def reverify_content_identity(self) -> dict[str, Any]:
        self._check_open()
        records: dict[str, Any] = {}
        seen: set[int] = set()
        for role, provider in (("base", self.base), ("extension", self.extension)):
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            verifier = getattr(provider, "reverify_content_identity", None)
            records[role] = verifier() if callable(verifier) else None
        return records

    def snapshot(self) -> dict[str, Any]:
        self._check_open()
        snapshots = {}
        for role, provider in (("base", self.base), ("extension", self.extension)):
            snapshot = getattr(provider, "snapshot", None)
            snapshots[role] = snapshot() if callable(snapshot) else None
        return {
            "schema": "linked-qstore-resolved-v1",
            "image_id": self.image_id,
            "overlay_blocks": sorted(self.overlay_names),
            "route_count": len(self._routes),
            "base_route_count": self.store_identity["base_route_count"],
            "extension_route_count": self.store_identity["extension_route_count"],
            "provider_cache_budgets_mb": dict(
                getattr(self, "_provider_cache_budgets_mb", {})
            ),
            "providers": snapshots,
            "store_identity": self.store_identity,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.close_underlying:
            self.extension.close()
            if self.extension is not self.base:
                self.base.close()

    def __enter__(self) -> ResolvedLinkedQStore:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = ["LinkedQStoreError", "OverlayQStore", "ResolvedLinkedQStore"]
