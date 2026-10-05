"""Experimental device-native block-paged BF16 K/V storage for MLX.

The pool owns fixed K/V slabs in MLX unified memory.  Logical caches hold only generation-bound
page authorities, so complete prefix pages can be shared by refcount during a fork.  A partial
fork tail is copied into a private page, and an append after trimming into a shared page performs
copy-on-write before modifying the tail.

The default ``update_and_fetch`` interface remains mlx-lm-compatible and materializes a dense
logical sequence when it spans multiple pages.  An explicitly admitted BF16/B1 paged-decode lane
instead uses ``update_for_paged_decode``: K=1 is appended transactionally and a physical-slab plus
logical-page-table view is returned without constructing dense K/V.  The two paths have separate
telemetry and the default path is unchanged when the lane is disabled.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .._compat import add_exception_note

MLX_PAGED_KV_CACHE_ABI = "mrun-mlx-block-paged-bf16-kv-v1"


class MlxPagedKVError(RuntimeError):
    """A paged K/V operation crossed capacity, identity, or authority boundaries."""


class MlxPagedKVCapacityError(MlxPagedKVError):
    """The fixed physical page pool cannot satisfy an append or private-tail copy."""


@dataclass(frozen=True, slots=True)
class MlxPagedAttentionInput:
    """Validated device inputs for one physical-page attention dispatch."""

    keys: Any
    values: Any
    page_slots: Any
    length: int
    logical_page_count: int
    page_size: int
    kv_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: Any

    def __post_init__(self) -> None:
        for field_name in (
            "length",
            "logical_page_count",
            "page_size",
            "kv_heads",
            "key_head_dim",
            "value_head_dim",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if self.logical_page_count != _ceil_pages(self.length, self.page_size):
            raise ValueError("paged-attention length and logical page count diverged")
        if self.page_size & (self.page_size - 1):
            raise ValueError("paged-attention page size must be a power of two")
        if (
            int(getattr(self.page_slots, "ndim", -1)) != 1
            or int(self.page_slots.shape[0]) != self.logical_page_count
        ):
            raise ValueError("paged-attention page slots differ from the logical page table")
        if str(getattr(self.page_slots, "dtype", "")).lower().rsplit(".", maxsplit=1)[-1] not in {
            "int32",
        }:
            raise ValueError("paged-attention page slots must be int32")
        key_shape = tuple(int(value) for value in getattr(self.keys, "shape", ()))
        value_shape = tuple(int(value) for value in getattr(self.values, "shape", ()))
        expected_key_tail = (self.kv_heads, self.page_size, self.key_head_dim)
        expected_value_tail = (self.kv_heads, self.page_size, self.value_head_dim)
        if (
            len(key_shape) != 4
            or len(value_shape) != 4
            or key_shape[0] <= 0
            or value_shape[0] != key_shape[0]
            or key_shape[1:] != expected_key_tail
            or value_shape[1:] != expected_value_tail
        ):
            raise ValueError("paged-attention physical slabs differ from the declared geometry")
        if (
            getattr(self.keys, "dtype", None) != self.dtype
            or getattr(self.values, "dtype", None) != self.dtype
        ):
            raise ValueError("paged-attention slab dtypes differ from the declared dtype")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _name(value: str, field_name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _ceil_pages(token_count: int, page_size: int) -> int:
    return 0 if token_count == 0 else (token_count + page_size - 1) // page_size


@dataclass(frozen=True, slots=True)
class MlxKVPageAuthority:
    """Generation-stamped authority over one physical page reference."""

    pool_id: str
    slot: int
    generation: int
    cache_abi: str = MLX_PAGED_KV_CACHE_ABI

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", _name(self.pool_id, "pool_id"))
        if self.cache_abi != MLX_PAGED_KV_CACHE_ABI:
            raise ValueError("unsupported MLX paged K/V authority ABI")
        for field_name in ("slot", "generation"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.generation == 0:
            raise ValueError("page authority generation must be positive")


@dataclass(frozen=True, slots=True)
class MlxPagedKVPoolTelemetry:
    """Exact physical/logical page accounting.

    ``physical = free + reserved`` counts unique slab pages.  ``live = reserved + shared``
    counts allocator-held references (logical cache references plus quarantines), so ``shared``
    is the physical byte saving from aliases. ``logical_capacity`` is exact admitted token address
    space and may exceed physical bytes; the fixed pool still fails an append before mutation
    whenever no private page is available.
    """

    pool_id: str
    page_size: int
    page_count: int
    page_bytes: int
    physical_bytes: int
    free_pages: int
    free_bytes: int
    reserved_pages: int
    reserved_bytes: int
    live_page_references: int
    live_bytes: int
    shared_pages: int
    shared_bytes: int
    active_caches: int
    logical_capacity_pages: int
    logical_capacity_bytes: int
    quarantined_pages: int
    quarantined_bytes: int
    leaked_pages: int
    page_acquires: int
    page_reuses: int
    page_retains: int
    page_releases: int
    full_page_prefix_shares: int
    partial_tail_copy_tokens: int
    partial_tail_copy_bytes: int
    cow_pages: int
    cow_copy_tokens: int
    cow_copy_bytes: int
    update_calls: int
    appended_tokens: int
    trims: int
    resets: int
    cache_releases: int
    concatenate_materializations: int
    materialized_bytes: int
    paged_decode_appends: int
    page_table_materializations: int
    dense_materializations_avoided: int
    quarantine_reconciliations: int
    quarantine_reconciliation_failures: int
    failures: int
    reconciled: bool


@dataclass(frozen=True, slots=True)
class MlxPagedKVFactoryTelemetry:
    pool_count: int
    physical_bytes: int
    free_bytes: int
    reserved_bytes: int
    live_bytes: int
    shared_bytes: int
    logical_capacity_bytes: int
    active_caches: int
    quarantined_pages: int
    quarantined_bytes: int
    leaked_pages: int
    full_page_prefix_shares: int
    partial_tail_copy_bytes: int
    cow_copy_bytes: int
    concatenate_materializations: int
    materialized_bytes: int
    paged_decode_appends: int
    page_table_materializations: int
    dense_materializations_avoided: int
    quarantine_reconciliations: int
    quarantine_reconciliation_failures: int
    failures: int
    reconciled: bool


class MlxKVPagePool:
    """One fixed MLX slab for pages belonging to a single decoder-layer geometry."""

    def __init__(
        self,
        *,
        mx: Any,
        create_attention_mask: Any,
        page_size: int,
        page_count: int,
        kv_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        key_dtype: Any,
        value_dtype: Any,
        paged_decode_attention: bool = False,
        pool_id: str | None = None,
    ) -> None:
        self._mx = mx
        self._create_attention_mask = create_attention_mask
        self.page_size = _positive_int(page_size, "page_size")
        self.page_count = _positive_int(page_count, "page_count")
        self.kv_heads = _positive_int(kv_heads, "kv_heads")
        self.key_head_dim = _positive_int(key_head_dim, "key_head_dim")
        self.value_head_dim = _positive_int(value_head_dim, "value_head_dim")
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        if type(paged_decode_attention) is not bool:
            raise TypeError("paged_decode_attention must be boolean")
        self.paged_decode_attention = paged_decode_attention
        if self.paged_decode_attention:
            dtype_names = {
                str(value).lower().rsplit(".", maxsplit=1)[-1]
                for value in (self.key_dtype, self.value_dtype)
            }
            if dtype_names.isdisjoint({"bf16", "bfloat16"}) or len(dtype_names) != 1:
                raise MlxPagedKVError("paged decode requires matching BF16 K/V page slabs")
            if (
                self.key_head_dim != self.value_head_dim
                or self.key_head_dim % 32
                or not 32 <= self.key_head_dim <= 256
            ):
                raise MlxPagedKVError("paged decode requires matching SIMD-aligned K/V head widths")
            if self.page_size & (self.page_size - 1):
                raise MlxPagedKVError("paged decode requires a power-of-two page size")
        self.pool_id = _name(pool_id or f"mlx-kv-pool-{uuid4().hex}", "pool_id")
        for operation in ("zeros", "concatenate", "eval"):
            if not callable(getattr(mx, operation, None)):
                raise TypeError(f"MLX execution context lacks callable {operation!r}")
        if self.paged_decode_attention and (
            not callable(getattr(mx, "array", None)) or getattr(mx, "int32", None) is None
        ):
            raise TypeError("paged decode requires MLX int32 page-table construction")
        if not callable(create_attention_mask):
            raise TypeError("create_attention_mask must be callable")
        key_shape = (self.page_count, self.kv_heads, self.page_size, self.key_head_dim)
        value_shape = (self.page_count, self.kv_heads, self.page_size, self.value_head_dim)
        self._keys = mx.zeros(
            key_shape,
            dtype=key_dtype,
        )
        self._values = mx.zeros(
            value_shape,
            dtype=value_dtype,
        )
        self._array_type = type(self._keys)
        if (
            not isinstance(self._values, self._array_type)
            or tuple(int(value) for value in self._keys.shape) != key_shape
            or tuple(int(value) for value in self._values.shape) != value_shape
            or self._keys.dtype != key_dtype
            or self._values.dtype != value_dtype
        ):
            raise MlxPagedKVError("MLX page slab differs from its declared geometry")
        # ``mx.zeros`` is lazy.  Force the complete bounded slab to become resident now so pool
        # construction, rather than a later request, owns any device-allocation failure.
        mx.eval(self._keys, self._values)
        physical_bytes = int(self._keys.nbytes) + int(self._values.nbytes)
        if physical_bytes <= 0 or physical_bytes % self.page_count:
            raise MlxPagedKVError("MLX page slab has invalid physical byte accounting")
        self.page_bytes = physical_bytes // self.page_count
        self.bytes_per_token = self.page_bytes // self.page_size
        if self.bytes_per_token * self.page_size != self.page_bytes:
            raise MlxPagedKVError("MLX page byte geometry is not token-addressable")
        self._generations = [0] * self.page_count
        self._refcounts = [0] * self.page_count
        self._free = list(reversed(range(self.page_count)))
        self._cache_pages: dict[str, tuple[MlxKVPageAuthority, ...]] = {}
        self._cache_capacity_pages: dict[str, int] = {}
        self._cache_capacity_tokens: dict[str, int] = {}
        self._quarantined: dict[int, MlxKVPageAuthority] = {}
        self._closed = False
        self._lock = threading.RLock()
        self._page_acquires = 0
        self._page_reuses = 0
        self._page_retains = 0
        self._page_releases = 0
        self._full_page_prefix_shares = 0
        self._partial_tail_copy_tokens = 0
        self._partial_tail_copy_bytes = 0
        self._cow_pages = 0
        self._cow_copy_tokens = 0
        self._cow_copy_bytes = 0
        self._update_calls = 0
        self._appended_tokens = 0
        self._trims = 0
        self._resets = 0
        self._cache_releases = 0
        self._concatenate_materializations = 0
        self._materialized_bytes = 0
        self._paged_decode_appends = 0
        self._page_table_materializations = 0
        self._dense_materializations_avoided = 0
        self._quarantine_reconciliations = 0
        self._quarantine_reconciliation_failures = 0
        self._failures = 0

    @property
    def physical_bytes(self) -> int:
        return self.page_count * self.page_bytes

    @property
    def maximum_tokens(self) -> int:
        return self.page_count * self.page_size

    @property
    def geometry_signature(self) -> tuple[Any, ...]:
        return (
            MLX_PAGED_KV_CACHE_ABI,
            self.pool_id,
            self.page_size,
            self.page_count,
            self.kv_heads,
            self.key_head_dim,
            self.value_head_dim,
            str(self.key_dtype),
            str(self.value_dtype),
            self.paged_decode_attention,
            id(self._keys),
            id(self._values),
        )

    def _require_open_unlocked(self) -> None:
        if self._closed:
            raise MlxPagedKVError("MLX page pool is closed")

    def _validate_authority_unlocked(
        self,
        authority: MlxKVPageAuthority,
    ) -> int:
        if type(authority) is not MlxKVPageAuthority:
            raise TypeError("page operation requires MlxKVPageAuthority")
        if authority.pool_id != self.pool_id:
            raise MlxPagedKVError("page authority belongs to another pool")
        if authority.slot >= self.page_count:
            raise MlxPagedKVError("page authority slot lies outside the pool")
        slot = authority.slot
        if self._generations[slot] != authority.generation or self._refcounts[slot] <= 0:
            raise MlxPagedKVError("page authority is stale, released, or ABA-reused")
        return slot

    def page_refcount(self, authority: MlxKVPageAuthority) -> int:
        with self._lock:
            slot = self._validate_authority_unlocked(authority)
            return self._refcounts[slot]

    def _acquire_unlocked(self) -> MlxKVPageAuthority:
        self._require_open_unlocked()
        if not self._free:
            raise MlxPagedKVCapacityError("fixed MLX page pool is exhausted")
        slot = self._free.pop()
        if self._refcounts[slot] != 0:
            raise MlxPagedKVError("free-page ledger contains a referenced slot")
        if self._generations[slot] > 0:
            self._page_reuses += 1
        self._generations[slot] += 1
        self._refcounts[slot] = 1
        self._page_acquires += 1
        return MlxKVPageAuthority(
            pool_id=self.pool_id,
            slot=slot,
            generation=self._generations[slot],
        )

    def _retain_unlocked(self, authority: MlxKVPageAuthority) -> None:
        slot = self._validate_authority_unlocked(authority)
        self._refcounts[slot] += 1
        self._page_retains += 1

    def _release_unlocked(self, authority: MlxKVPageAuthority) -> None:
        slot = self._validate_authority_unlocked(authority)
        self._refcounts[slot] -= 1
        self._page_releases += 1
        if self._refcounts[slot] == 0:
            self._free.append(slot)

    def _page_arrays_unlocked(self, authority: MlxKVPageAuthority) -> tuple[Any, Any]:
        slot = self._validate_authority_unlocked(authority)
        return self._keys[slot : slot + 1], self._values[slot : slot + 1]

    def _eval_pages_unlocked(self, authorities: Sequence[MlxKVPageAuthority]) -> None:
        if not authorities:
            return
        arrays = tuple(
            value for authority in authorities for value in self._page_arrays_unlocked(authority)
        )
        self._mx.eval(*arrays)

    def _discard_unpublished_unlocked(
        self,
        authorities: Sequence[MlxKVPageAuthority],
        operation_error: BaseException,
    ) -> None:
        """Safely return unpublished pages, or quarantine them when device evaluation fails.

        MLX assignment is lazy.  Reusing a page whose failed transaction has not reached an
        evaluation boundary could let delayed writes corrupt its next owner.  In that case the
        authority intentionally remains live but absent from a cache page table: telemetry marks
        the slot leaked/unreconciled and pool shutdown fails closed instead of reusing it.
        """

        pages = tuple(authorities)
        if not pages:
            return
        try:
            self._eval_pages_unlocked(pages)
        except BaseException as cleanup_error:
            add_exception_note(
                operation_error,
                "unpublished MLX pages were quarantined after device evaluation failed: "
                f"{cleanup_error}"
            )
            self._quarantine_unlocked(pages, operation_error)
            return
        for authority in reversed(pages):
            try:
                self._release_unlocked(authority)
            except BaseException as cleanup_error:
                add_exception_note(
                    operation_error,
                    "unpublished MLX page release failed; quarantine was attempted: "
                    f"{cleanup_error}"
                )
                self._quarantine_unlocked((authority,), operation_error)

    def _quarantine_unlocked(
        self,
        authorities: Sequence[MlxKVPageAuthority],
        operation_error: BaseException,
    ) -> None:
        for authority in authorities:
            try:
                slot = self._validate_authority_unlocked(authority)
            except BaseException as quarantine_error:
                add_exception_note(
                    operation_error,
                    "failed MLX page could not be entered in the quarantine ledger: "
                    f"{quarantine_error}"
                )
                continue
            existing = self._quarantined.get(slot)
            if existing is not None and existing != authority:
                add_exception_note(operation_error, "MLX quarantine ledger contains a conflicting authority")
                continue
            self._quarantined[slot] = authority

    def reconcile_quarantined_pages(self) -> int:
        """Retry the device barrier and return safe quarantined pages to the fixed pool."""

        with self._lock:
            self._require_open_unlocked()
            pages = tuple(self._quarantined.values())
            if not pages:
                return 0
            try:
                self._eval_pages_unlocked(pages)
                for authority in pages:
                    slot = self._validate_authority_unlocked(authority)
                    if self._refcounts[slot] != 1:
                        raise MlxPagedKVError(
                            "quarantined page unexpectedly acquired another reference"
                        )
            except BaseException as reconciliation_error:
                self._quarantine_reconciliation_failures += 1
                self._failures += 1
                raise MlxPagedKVError("MLX page quarantine could not be reconciled") from (
                    reconciliation_error
                )
            for authority in reversed(pages):
                self._release_unlocked(authority)
                self._quarantined.pop(authority.slot)
            self._quarantine_reconciliations += 1
            return len(pages)

    def _copy_page_prefix_unlocked(
        self,
        source: MlxKVPageAuthority,
        target: MlxKVPageAuthority,
        token_count: int,
    ) -> None:
        if token_count < 0 or token_count > self.page_size:
            raise ValueError("page prefix copy lies outside one physical page")
        source_slot = self._validate_authority_unlocked(source)
        target_slot = self._validate_authority_unlocked(target)
        if source_slot == target_slot:
            raise MlxPagedKVError("private page copy target aliases its source")
        if token_count:
            self._keys[target_slot : target_slot + 1, :, :token_count, :] = self._keys[
                source_slot : source_slot + 1, :, :token_count, :
            ]
            self._values[target_slot : target_slot + 1, :, :token_count, :] = self._values[
                source_slot : source_slot + 1, :, :token_count, :
            ]

    def _register_cache_unlocked(self, cache_id: str, capacity: int) -> None:
        self._require_open_unlocked()
        if cache_id in self._cache_pages:
            raise MlxPagedKVError("logical cache identity is already registered")
        self._cache_pages[cache_id] = ()
        self._cache_capacity_pages[cache_id] = _ceil_pages(capacity, self.page_size)
        self._cache_capacity_tokens[cache_id] = capacity

    def _publish_cache_unlocked(
        self,
        cache_id: str,
        authorities: Sequence[MlxKVPageAuthority],
    ) -> None:
        if cache_id not in self._cache_pages:
            raise MlxPagedKVError("logical cache is not registered with this pool")
        pages = tuple(authorities)
        if len(pages) > self._cache_capacity_pages[cache_id]:
            raise MlxPagedKVError("logical cache page table exceeds its declared capacity")
        for authority in pages:
            self._validate_authority_unlocked(authority)
        self._cache_pages[cache_id] = pages

    def _unregister_cache_unlocked(self, cache_id: str) -> None:
        if self._cache_pages.get(cache_id) != ():
            raise MlxPagedKVError("logical cache still owns pages during unregister")
        if cache_id not in self._cache_capacity_pages:
            raise MlxPagedKVError("logical cache was already unregistered")
        self._cache_pages.pop(cache_id)
        self._cache_capacity_pages.pop(cache_id)
        self._cache_capacity_tokens.pop(cache_id)
        self._cache_releases += 1

    def new_cache(self, *, capacity: int) -> FixedMlxPagedKVCache:
        return FixedMlxPagedKVCache(pool=self, capacity=capacity)

    def _reconciled_unlocked(self) -> tuple[bool, int]:
        observed = [0] * self.page_count
        valid = (
            self._cache_pages.keys()
            == self._cache_capacity_pages.keys()
            == self._cache_capacity_tokens.keys()
        )
        for pages in self._cache_pages.values():
            for authority in pages:
                try:
                    slot = self._validate_authority_unlocked(authority)
                except (TypeError, MlxPagedKVError):
                    valid = False
                    continue
                observed[slot] += 1
        for authority in self._quarantined.values():
            try:
                slot = self._validate_authority_unlocked(authority)
            except (TypeError, MlxPagedKVError):
                valid = False
                continue
            observed[slot] += 1
        valid = valid and observed == self._refcounts
        free_set = set(self._free)
        valid = valid and len(free_set) == len(self._free)
        valid = valid and free_set == {
            slot for slot, refcount in enumerate(self._refcounts) if refcount == 0
        }
        # A leaked page is a physical slot with at least one live reference not represented by
        # a logical page table.  Count slots, not excess aliases, to match the physical allocator.
        leaked = sum(
            refcount > observed_count
            for refcount, observed_count in zip(self._refcounts, observed, strict=True)
        )
        return valid and leaked == 0, leaked

    def telemetry(self) -> MlxPagedKVPoolTelemetry:
        with self._lock:
            reserved_pages = sum(refcount > 0 for refcount in self._refcounts)
            live_references = sum(self._refcounts)
            shared_pages = sum(refcount > 1 for refcount in self._refcounts)
            reconciled, leaked_pages = self._reconciled_unlocked()
            return MlxPagedKVPoolTelemetry(
                pool_id=self.pool_id,
                page_size=self.page_size,
                page_count=self.page_count,
                page_bytes=self.page_bytes,
                physical_bytes=self.physical_bytes,
                free_pages=len(self._free),
                free_bytes=len(self._free) * self.page_bytes,
                reserved_pages=reserved_pages,
                reserved_bytes=reserved_pages * self.page_bytes,
                live_page_references=live_references,
                live_bytes=live_references * self.page_bytes,
                shared_pages=shared_pages,
                shared_bytes=(live_references - reserved_pages) * self.page_bytes,
                active_caches=len(self._cache_pages),
                logical_capacity_pages=sum(self._cache_capacity_pages.values()),
                logical_capacity_bytes=(
                    sum(self._cache_capacity_tokens.values()) * self.bytes_per_token
                ),
                quarantined_pages=len(self._quarantined),
                quarantined_bytes=len(self._quarantined) * self.page_bytes,
                leaked_pages=leaked_pages,
                page_acquires=self._page_acquires,
                page_reuses=self._page_reuses,
                page_retains=self._page_retains,
                page_releases=self._page_releases,
                full_page_prefix_shares=self._full_page_prefix_shares,
                partial_tail_copy_tokens=self._partial_tail_copy_tokens,
                partial_tail_copy_bytes=self._partial_tail_copy_bytes,
                cow_pages=self._cow_pages,
                cow_copy_tokens=self._cow_copy_tokens,
                cow_copy_bytes=self._cow_copy_bytes,
                update_calls=self._update_calls,
                appended_tokens=self._appended_tokens,
                trims=self._trims,
                resets=self._resets,
                cache_releases=self._cache_releases,
                concatenate_materializations=self._concatenate_materializations,
                materialized_bytes=self._materialized_bytes,
                paged_decode_appends=self._paged_decode_appends,
                page_table_materializations=self._page_table_materializations,
                dense_materializations_avoided=self._dense_materializations_avoided,
                quarantine_reconciliations=self._quarantine_reconciliations,
                quarantine_reconciliation_failures=(self._quarantine_reconciliation_failures),
                failures=self._failures,
                reconciled=reconciled,
            )

    def close(self) -> None:
        with self._lock:
            reconciled, leaked = self._reconciled_unlocked()
            if (
                self._cache_pages
                or self._quarantined
                or any(self._refcounts)
                or leaked
                or not reconciled
            ):
                raise MlxPagedKVError(
                    "cannot close MLX page pool with live, quarantined, or leaked pages"
                )
            self._closed = True


class FixedMlxPagedKVCache:
    """``mlx-lm`` cache facade backed by a generation-safe physical page table."""

    __slots__ = (
        "_cache_id",
        "_capacity",
        "_keys_view",
        "_last_prefix_copy_bytes",
        "_lock",
        "_pages",
        "_pool",
        "_released",
        "_values_view",
        "_views_offset",
        "offset",
    )

    def __init__(self, *, pool: MlxKVPagePool, capacity: int) -> None:
        if not isinstance(pool, MlxKVPagePool):
            raise TypeError("paged K/V cache requires MlxKVPagePool")
        self._pool = pool
        self._capacity = _positive_int(capacity, "capacity")
        if self._capacity > pool.maximum_tokens:
            raise MlxPagedKVCapacityError("logical cache capacity exceeds its physical pool")
        self._cache_id = f"mlx-kv-cache-{uuid4().hex}"
        self._pages: tuple[MlxKVPageAuthority, ...] = ()
        self.offset = 0
        self._released = False
        self._last_prefix_copy_bytes = 0
        self._lock = threading.RLock()
        with pool._lock:  # noqa: SLF001 - cache and pool form one authority domain
            pool._register_cache_unlocked(self._cache_id, self._capacity)  # noqa: SLF001
            try:
                self._keys_view, self._values_view = self._materialize_unlocked((), 0)
                self._views_offset = 0
            except BaseException:
                pool._unregister_cache_unlocked(self._cache_id)  # noqa: SLF001
                raise

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def page_size(self) -> int:
        return self._pool.page_size

    @property
    def paged_decode_attention_enabled(self) -> bool:
        return self._pool.paged_decode_attention

    @property
    def page_authorities(self) -> tuple[MlxKVPageAuthority, ...]:
        with self._lock:
            self._require_live_unlocked()
            return self._pages

    @property
    def last_prefix_copy_bytes(self) -> int:
        return self._last_prefix_copy_bytes

    @property
    def keys(self) -> Any:
        with self._lock, self._pool._lock:  # noqa: SLF001
            self._require_live_unlocked()
            self._ensure_materialized_unlocked()
            return self._keys_view

    @property
    def values(self) -> Any:
        with self._lock, self._pool._lock:  # noqa: SLF001
            self._require_live_unlocked()
            self._ensure_materialized_unlocked()
            return self._values_view

    @property
    def state(self) -> tuple[Any, Any]:
        return self.keys, self.values

    @property
    def nbytes(self) -> int:
        """Logical admitted bytes; physical/shared bytes are reported by pool telemetry."""

        return self._capacity * self._pool.bytes_per_token

    def _require_live_unlocked(self) -> None:
        if self._released:
            raise MlxPagedKVError("logical MLX paged cache was released")

    def _ensure_materialized_unlocked(self) -> None:
        if self._views_offset == self.offset:
            return
        self._keys_view, self._values_view = self._materialize_unlocked(
            self._pages,
            self.offset,
        )
        self._views_offset = self.offset

    def evaluation_arrays(self) -> tuple[Any, Any]:
        """Return slab dependencies without forcing a dense logical-sequence view."""

        with self._lock, self._pool._lock:  # noqa: SLF001
            self._require_live_unlocked()
            for authority in self._pages:
                self._pool._validate_authority_unlocked(authority)  # noqa: SLF001
            return self._pool._keys, self._pool._values  # noqa: SLF001

    def _validate_source(self, keys: Any, values: Any) -> int:
        pool = self._pool
        if not isinstance(keys, pool._array_type) or not isinstance(  # noqa: SLF001
            values,
            pool._array_type,  # noqa: SLF001
        ):
            raise TypeError("model K/V output must remain in the pool's device array domain")
        if int(getattr(keys, "ndim", -1)) != 4 or int(getattr(values, "ndim", -1)) != 4:
            raise ValueError("model K/V output must be rank-four MLX arrays")
        token_count = int(keys.shape[2])
        if token_count <= 0 or self.offset + token_count > self._capacity:
            raise OverflowError("native MLX paged K/V append exceeds logical capacity")
        if (
            int(keys.shape[0]) != 1
            or int(values.shape[0]) != 1
            or int(keys.shape[1]) != pool.kv_heads
            or int(values.shape[1]) != pool.kv_heads
            or int(keys.shape[3]) != pool.key_head_dim
            or int(values.shape[3]) != pool.value_head_dim
            or int(values.shape[2]) != token_count
            or keys.dtype != pool.key_dtype
            or values.dtype != pool.value_dtype
        ):
            raise ValueError("model K/V output differs from the paged pool geometry")
        return token_count

    def _materialize_unlocked(
        self,
        pages: Sequence[MlxKVPageAuthority],
        length: int,
    ) -> tuple[Any, Any]:
        pool = self._pool
        if length == 0:
            return (
                pool._keys[0:1, :, :0, :],  # noqa: SLF001
                pool._values[0:1, :, :0, :],  # noqa: SLF001
            )
        page_count = _ceil_pages(length, pool.page_size)
        if page_count != len(pages):
            raise MlxPagedKVError("logical offset and page table length diverged")
        key_parts: list[Any] = []
        value_parts: list[Any] = []
        for index, authority in enumerate(pages):
            keys, values = pool._page_arrays_unlocked(authority)  # noqa: SLF001
            stop = (
                pool.page_size
                if index < page_count - 1
                else length - (page_count - 1) * pool.page_size
            )
            key_parts.append(keys[:, :, :stop, :])
            value_parts.append(values[:, :, :stop, :])
        if page_count == 1:
            return key_parts[0], value_parts[0]
        materialized_keys = pool._mx.concatenate(tuple(key_parts), axis=2)  # noqa: SLF001
        materialized_values = pool._mx.concatenate(tuple(value_parts), axis=2)  # noqa: SLF001
        pool._concatenate_materializations += 1  # noqa: SLF001
        pool._materialized_bytes += length * pool.bytes_per_token  # noqa: SLF001
        return materialized_keys, materialized_values

    def _update(
        self,
        keys: Any,
        values: Any,
        *,
        materialize: bool,
    ) -> tuple[Any, Any] | MlxPagedAttentionInput:
        pool = self._pool
        with self._lock, pool._lock:  # noqa: SLF001 - one page-table transaction
            self._require_live_unlocked()
            token_count = self._validate_source(keys, values)
            if not materialize and (not pool.paged_decode_attention or token_count != 1):
                raise MlxPagedKVError(
                    "physical-page append requires an admitted paged-decode K=1 lane"
                )
            old_pages = self._pages
            old_offset = self.offset
            new_offset = old_offset + token_count
            needed_pages = _ceil_pages(new_offset, pool.page_size)
            missing_pages = needed_pages - len(old_pages)
            tail_count = old_offset % pool.page_size
            cow = bool(
                tail_count
                and old_pages
                and pool._refcounts[  # noqa: SLF001
                    pool._validate_authority_unlocked(old_pages[-1])  # noqa: SLF001
                ]
                > 1
            )
            allocation_count = missing_pages + int(cow)
            if allocation_count > len(pool._free):  # noqa: SLF001
                pool._failures += 1  # noqa: SLF001
                raise MlxPagedKVCapacityError("fixed MLX page pool lacks private append pages")
            acquired: list[MlxKVPageAuthority] = []
            old_tail: MlxKVPageAuthority | None = None
            published = False
            old_tail_released = False
            try:
                for _index in range(allocation_count):
                    acquired.append(pool._acquire_unlocked())  # noqa: SLF001
                candidate = list(old_pages)
                acquisition_index = 0
                if cow:
                    old_tail = candidate[-1]
                    private_tail = acquired[acquisition_index]
                    acquisition_index += 1
                    pool._copy_page_prefix_unlocked(  # noqa: SLF001
                        old_tail,
                        private_tail,
                        tail_count,
                    )
                    candidate[-1] = private_tail
                candidate.extend(acquired[acquisition_index:])

                source_start = 0
                logical_position = old_offset
                while source_start < token_count:
                    page_index = logical_position // pool.page_size
                    in_page = logical_position % pool.page_size
                    count = min(pool.page_size - in_page, token_count - source_start)
                    authority = candidate[page_index]
                    slot = pool._validate_authority_unlocked(authority)  # noqa: SLF001
                    pool._keys[slot : slot + 1, :, in_page : in_page + count, :] = keys[  # noqa: SLF001
                        :, :, source_start : source_start + count, :
                    ]
                    pool._values[slot : slot + 1, :, in_page : in_page + count, :] = values[  # noqa: SLF001
                        :, :, source_start : source_start + count, :
                    ]
                    source_start += count
                    logical_position += count
                candidate_pages = tuple(candidate)
                materialized = (
                    self._materialize_unlocked(candidate_pages, new_offset) if materialize else None
                )
                paged_input = None
                if materialized is None:
                    slots = pool._mx.array(  # noqa: SLF001
                        [authority.slot for authority in candidate_pages],
                        dtype=pool._mx.int32,  # noqa: SLF001
                    )
                    paged_input = MlxPagedAttentionInput(
                        keys=pool._keys,  # noqa: SLF001
                        values=pool._values,  # noqa: SLF001
                        page_slots=slots,
                        length=new_offset,
                        logical_page_count=len(candidate_pages),
                        page_size=pool.page_size,
                        kv_heads=pool.kv_heads,
                        key_head_dim=pool.key_head_dim,
                        value_head_dim=pool.value_head_dim,
                        dtype=pool.key_dtype,
                    )
                # Publication is the page-table commit point.  Cache fields are changed only
                # after every fallible device/authority operation has completed.
                pool._publish_cache_unlocked(self._cache_id, candidate_pages)  # noqa: SLF001
                published = True
                if old_tail is not None:
                    pool._release_unlocked(old_tail)  # noqa: SLF001
                    old_tail_released = True
                    pool._cow_pages += 1  # noqa: SLF001
                    pool._cow_copy_tokens += tail_count  # noqa: SLF001
                    pool._cow_copy_bytes += tail_count * pool.bytes_per_token  # noqa: SLF001
                self._pages = candidate_pages
                self.offset = new_offset
                if materialized is None:
                    self._views_offset = -1
                else:
                    self._keys_view, self._values_view = materialized
                    self._views_offset = new_offset
                pool._update_calls += 1  # noqa: SLF001
                pool._appended_tokens += token_count  # noqa: SLF001
                if materialized is not None:
                    return materialized
                pool._paged_decode_appends += 1  # noqa: SLF001
                pool._page_table_materializations += 1  # noqa: SLF001
                pool._dense_materializations_avoided += 1  # noqa: SLF001
                assert paged_input is not None
                return paged_input
            except BaseException as operation_error:
                rollback_succeeded = not published
                if published:
                    try:
                        if old_tail_released:
                            assert old_tail is not None
                            pool._retain_unlocked(old_tail)  # noqa: SLF001
                        pool._publish_cache_unlocked(self._cache_id, old_pages)  # noqa: SLF001
                        rollback_succeeded = True
                    except BaseException as rollback_error:
                        add_exception_note(
                            operation_error,
                            f"MLX page-table publication could not be rolled back: {rollback_error}"
                        )
                if rollback_succeeded:
                    pool._discard_unpublished_unlocked(  # noqa: SLF001
                        acquired,
                        operation_error,
                    )
                pool._failures += 1  # noqa: SLF001
                raise

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        result = self._update(keys, values, materialize=True)
        if not isinstance(result, tuple):
            raise AssertionError("dense MLX cache update returned paged-attention metadata")
        return result

    def update_for_paged_decode(self, keys: Any, values: Any) -> MlxPagedAttentionInput:
        result = self._update(keys, values, materialize=False)
        if not isinstance(result, MlxPagedAttentionInput):
            raise AssertionError("paged MLX cache update returned a dense sequence")
        return result

    def make_mask(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._require_live_unlocked()
            return self._pool._create_attention_mask(  # noqa: SLF001
                *args,
                offset=self.offset,
                **kwargs,
            )

    def _shrink_unlocked(self, offset: int, *, reset: bool) -> None:
        pool = self._pool
        if offset < 0 or offset > self.offset:
            raise ValueError("paged K/V rollback offset must not grow or escape the cache")
        if offset == self.offset:
            if reset:
                pool._resets += 1  # noqa: SLF001
            else:
                pool._trims += 1  # noqa: SLF001
            return
        kept_count = _ceil_pages(offset, pool.page_size)
        kept = self._pages[:kept_count]
        removed = self._pages[kept_count:]
        materialized = (
            None if pool.paged_decode_attention else self._materialize_unlocked(kept, offset)
        )
        pool._eval_pages_unlocked(removed)  # noqa: SLF001
        pool._publish_cache_unlocked(self._cache_id, kept)  # noqa: SLF001
        for authority in reversed(removed):
            pool._release_unlocked(authority)  # noqa: SLF001
        self._pages = kept
        self.offset = offset
        if materialized is None:
            self._views_offset = -1
        else:
            self._keys_view, self._values_view = materialized
            self._views_offset = offset
        if reset:
            pool._resets += 1  # noqa: SLF001
        else:
            pool._trims += 1  # noqa: SLF001

    def trim(self, count: int) -> int:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("trim count must be a non-negative integer")
        with self._lock, self._pool._lock:  # noqa: SLF001
            self._require_live_unlocked()
            trimmed = min(self.offset, count)
            self._shrink_unlocked(self.offset - trimmed, reset=False)
            return trimmed

    def reset(self, offset: int) -> None:
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError("reset offset must be an integer")
        with self._lock, self._pool._lock:  # noqa: SLF001
            self._require_live_unlocked()
            self._shrink_unlocked(offset, reset=True)

    def copy_committed_prefix_from(
        self,
        source: FixedMlxPagedKVCache,
        length: int,
    ) -> None:
        if not isinstance(source, FixedMlxPagedKVCache):
            raise TypeError("paged K/V fork source uses another cache ABI")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("paged K/V fork length must be non-negative")
        if source is self:
            raise MlxPagedKVError("paged K/V fork cannot target its source cache")
        if source._pool is not self._pool:  # noqa: SLF001
            raise MlxPagedKVError("paged K/V fork crosses a physical pool identity")
        first, second = sorted((self, source), key=lambda cache: cache._cache_id)  # noqa: SLF001
        pool = self._pool
        with first._lock, second._lock, pool._lock:  # noqa: SLF001
            self._require_live_unlocked()
            source._require_live_unlocked()  # noqa: SLF001
            if self.offset != 0 or self._pages:
                raise MlxPagedKVError("paged K/V fork target must be empty")
            if length > source.offset or length > self._capacity:
                raise OverflowError("paged K/V fork exceeds source or target capacity")
            full_pages, tail_tokens = divmod(length, pool.page_size)
            required_private = int(tail_tokens > 0)
            if required_private > len(pool._free):  # noqa: SLF001
                pool._failures += 1  # noqa: SLF001
                raise MlxPagedKVCapacityError("paged K/V fork lacks a private tail page")
            retained: list[MlxKVPageAuthority] = []
            acquired: list[MlxKVPageAuthority] = []
            published = False
            try:
                for authority in source._pages[:full_pages]:  # noqa: SLF001
                    pool._retain_unlocked(authority)  # noqa: SLF001
                    retained.append(authority)
                candidate = list(retained)
                if tail_tokens:
                    private_tail = pool._acquire_unlocked()  # noqa: SLF001
                    acquired.append(private_tail)
                    source_tail = source._pages[full_pages]  # noqa: SLF001
                    pool._copy_page_prefix_unlocked(  # noqa: SLF001
                        source_tail,
                        private_tail,
                        tail_tokens,
                    )
                    pool._eval_pages_unlocked((private_tail,))  # noqa: SLF001
                    candidate.append(private_tail)
                candidate_pages = tuple(candidate)
                materialized = (
                    None
                    if pool.paged_decode_attention
                    else self._materialize_unlocked(candidate_pages, length)
                )
                pool._publish_cache_unlocked(self._cache_id, candidate_pages)  # noqa: SLF001
                published = True
                self._pages = candidate_pages
                self.offset = length
                if materialized is None:
                    self._views_offset = -1
                else:
                    self._keys_view, self._values_view = materialized
                    self._views_offset = length
                self._last_prefix_copy_bytes = tail_tokens * pool.bytes_per_token
                pool._full_page_prefix_shares += full_pages  # noqa: SLF001
                pool._partial_tail_copy_tokens += tail_tokens  # noqa: SLF001
                pool._partial_tail_copy_bytes += self._last_prefix_copy_bytes  # noqa: SLF001
            except BaseException as operation_error:
                rollback_succeeded = not published
                if published:
                    try:
                        pool._publish_cache_unlocked(self._cache_id, ())  # noqa: SLF001
                        rollback_succeeded = True
                    except BaseException as rollback_error:
                        add_exception_note(
                            operation_error,
                            "fork page-table publication could not be rolled back: "
                            f"{rollback_error}"
                        )
                if rollback_succeeded:
                    pool._discard_unpublished_unlocked(  # noqa: SLF001
                        acquired,
                        operation_error,
                    )
                    for authority in reversed(retained):
                        try:
                            pool._release_unlocked(authority)  # noqa: SLF001
                        except BaseException as cleanup_error:
                            add_exception_note(
                                operation_error,
                                "retained fork page release failed and the pool is "
                                f"unreconciled: {cleanup_error}"
                            )
                pool._failures += 1  # noqa: SLF001
                raise

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return True

    def empty(self) -> bool:
        return self.offset == 0

    def storage_signature(self) -> tuple[Any, ...]:
        return (
            MLX_PAGED_KV_CACHE_ABI,
            self._pool.pool_id,
            self._cache_id,
            self._capacity,
            self._pool.geometry_signature,
        )

    def validate_release(self) -> None:
        with self._lock, self._pool._lock:  # noqa: SLF001
            self._require_live_unlocked()
            registered = self._pool._cache_pages.get(self._cache_id)  # noqa: SLF001
            if registered != self._pages:
                raise MlxPagedKVError("cache release crossed a corrupt pool page ledger")
            for authority in self._pages:
                self._pool._validate_authority_unlocked(authority)  # noqa: SLF001
            # Establish a device evaluation boundary before a factory begins its cross-layer
            # release.  Once every layer validates, page-ledger mutation is deterministic.
            self._pool._eval_pages_unlocked(self._pages)  # noqa: SLF001

    def _release(self, *, evaluate: bool) -> None:
        pool = self._pool
        with self._lock, pool._lock:  # noqa: SLF001
            self._require_live_unlocked()
            if pool._cache_pages.get(self._cache_id) != self._pages:  # noqa: SLF001
                raise MlxPagedKVError("cache release crossed a corrupt pool page ledger")
            for authority in self._pages:
                pool._validate_authority_unlocked(authority)  # noqa: SLF001
            if evaluate:
                pool._eval_pages_unlocked(self._pages)  # noqa: SLF001
            pool._publish_cache_unlocked(self._cache_id, ())  # noqa: SLF001
            for authority in reversed(self._pages):
                pool._release_unlocked(authority)  # noqa: SLF001
            self._pages = ()
            self.offset = 0
            self._views_offset = 0
            pool._unregister_cache_unlocked(self._cache_id)  # noqa: SLF001
            self._released = True

    def release(self) -> None:
        self._release(evaluate=True)


class MlxPagedKVCacheFactory:
    """Layer-ordered callable cache factory plus aggregate pool telemetry."""

    def __init__(self, pools: Sequence[MlxKVPagePool]) -> None:
        resolved = tuple(pools)
        if not resolved or any(not isinstance(pool, MlxKVPagePool) for pool in resolved):
            raise TypeError("paged cache factory requires non-empty MlxKVPagePool values")
        if len({pool.pool_id for pool in resolved}) != len(resolved):
            raise ValueError("paged cache factory pool identities must be unique")
        decode_modes = {pool.paged_decode_attention for pool in resolved}
        if len(decode_modes) != 1:
            raise ValueError("paged cache factory pools must share one decode-attention mode")
        if len({pool.page_size for pool in resolved}) != 1:
            raise ValueError("paged cache factory pools must share one page size")
        if len({pool.page_count for pool in resolved}) != 1:
            raise ValueError("paged cache factory pools must share one physical page count")
        self._pools = resolved
        self._lock = threading.RLock()
        self._closed = False

    @property
    def cache_abi(self) -> str:
        return MLX_PAGED_KV_CACHE_ABI

    @property
    def pools(self) -> tuple[MlxKVPagePool, ...]:
        return self._pools

    @property
    def paged_decode_attention(self) -> bool:
        return self._pools[0].paged_decode_attention

    @property
    def page_size(self) -> int:
        return self._pools[0].page_size

    @property
    def page_count(self) -> int:
        return self._pools[0].page_count

    def __call__(self, capacity: int) -> tuple[FixedMlxPagedKVCache, ...]:
        with self._lock:
            if self._closed:
                raise MlxPagedKVError("paged cache factory is closed")
            caches: list[FixedMlxPagedKVCache] = []
            try:
                for pool in self._pools:
                    caches.append(pool.new_cache(capacity=capacity))
            except BaseException as operation_error:
                for cache in reversed(caches):
                    try:
                        cache.release()
                    except BaseException as cleanup_error:
                        add_exception_note(
                            operation_error,
                            f"paged cache factory cleanup also failed: {cleanup_error}"
                        )
                raise
            return tuple(caches)

    def validate_release_caches(self, caches: Sequence[Any]) -> None:
        resolved = tuple(caches)
        if len(resolved) != len(self._pools):
            raise MlxPagedKVError("runtime cache release changed the layer count")
        for cache, pool in zip(resolved, self._pools, strict=True):
            if not isinstance(cache, FixedMlxPagedKVCache) or cache._pool is not pool:  # noqa: SLF001
                raise MlxPagedKVError("runtime cache release crossed a layer pool boundary")
            cache.validate_release()

    def release_caches(self, caches: Sequence[Any]) -> None:
        resolved = tuple(caches)
        with self._lock, ExitStack() as cache_locks:
            # Hold every logical-cache lock across the two-phase release.  Validation establishes
            # all device barriers first; mutation then cannot encounter a second lazy-eval failure.
            for cache in resolved:
                if isinstance(cache, FixedMlxPagedKVCache):
                    cache_locks.enter_context(cache._lock)  # noqa: SLF001
            self.validate_release_caches(resolved)
            for cache in reversed(resolved):
                cache._release(evaluate=False)  # noqa: SLF001

    def telemetry(self) -> MlxPagedKVFactoryTelemetry:
        snapshots = tuple(pool.telemetry() for pool in self._pools)
        return MlxPagedKVFactoryTelemetry(
            pool_count=len(snapshots),
            physical_bytes=sum(value.physical_bytes for value in snapshots),
            free_bytes=sum(value.free_bytes for value in snapshots),
            reserved_bytes=sum(value.reserved_bytes for value in snapshots),
            live_bytes=sum(value.live_bytes for value in snapshots),
            shared_bytes=sum(value.shared_bytes for value in snapshots),
            logical_capacity_bytes=sum(value.logical_capacity_bytes for value in snapshots),
            active_caches=sum(value.active_caches for value in snapshots),
            quarantined_pages=sum(value.quarantined_pages for value in snapshots),
            quarantined_bytes=sum(value.quarantined_bytes for value in snapshots),
            leaked_pages=sum(value.leaked_pages for value in snapshots),
            full_page_prefix_shares=sum(value.full_page_prefix_shares for value in snapshots),
            partial_tail_copy_bytes=sum(value.partial_tail_copy_bytes for value in snapshots),
            cow_copy_bytes=sum(value.cow_copy_bytes for value in snapshots),
            concatenate_materializations=sum(
                value.concatenate_materializations for value in snapshots
            ),
            materialized_bytes=sum(value.materialized_bytes for value in snapshots),
            paged_decode_appends=sum(value.paged_decode_appends for value in snapshots),
            page_table_materializations=sum(
                value.page_table_materializations for value in snapshots
            ),
            dense_materializations_avoided=sum(
                value.dense_materializations_avoided for value in snapshots
            ),
            quarantine_reconciliations=sum(value.quarantine_reconciliations for value in snapshots),
            quarantine_reconciliation_failures=sum(
                value.quarantine_reconciliation_failures for value in snapshots
            ),
            failures=sum(value.failures for value in snapshots),
            reconciled=all(value.reconciled for value in snapshots),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            snapshots = tuple(pool.telemetry() for pool in self._pools)
            if any(
                value.active_caches
                or value.quarantined_pages
                or value.leaked_pages
                or not value.reconciled
                for value in snapshots
            ):
                raise MlxPagedKVError("cannot close paged cache factory with live caches")
            for pool in self._pools:
                pool.close()
            self._closed = True


__all__ = [
    "MLX_PAGED_KV_CACHE_ABI",
    "FixedMlxPagedKVCache",
    "MlxKVPageAuthority",
    "MlxKVPagePool",
    "MlxPagedAttentionInput",
    "MlxPagedKVCacheFactory",
    "MlxPagedKVCapacityError",
    "MlxPagedKVError",
    "MlxPagedKVFactoryTelemetry",
    "MlxPagedKVPoolTelemetry",
]
