"""Exact-prefix session retention for backend-owned native model state.

The store never serializes or exposes cache tensors.  It owns one opaque retained-state handoff
per session and indexes complete committed token ledgers by content, allowing a new session ID to
fork the longest exact strict prefix under the same physical and semantic identity.  Each
continuation receives an independent child through the runtime's exact-prefix fork contract.  The
stored state covers ``committed_token_ids`` exactly, while ``pending_token_id`` is the model
output selected after that prefix and is deliberately *not* represented in retained state.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Protocol

from mrun.runtime.contracts import (
    ForkableModelRuntime,
    RuntimeRoute,
    StateForkResult,
    StateHandle,
    StateObservation,
)


class SessionStoreError(RuntimeError):
    """Base class for exact-prefix session failures."""


class SessionStoreClosed(SessionStoreError):
    """A lease was requested after the store stopped accepting work."""


class SessionIdentityError(SessionStoreError):
    """A caller attempted to cross a model/template/domain/route boundary."""


class SessionBusyError(SessionStoreError):
    """A second request attempted to mutate a session with an active lease."""


class SessionPrefixMismatch(SessionStoreError):
    """A known session's new prompt is not a strict extension of committed K/V."""


class SessionCapacityError(SessionStoreError):
    """The bounded retained-state budget cannot admit another session."""


class SessionIntegrityError(SessionStoreError):
    """Stored identity, ledger, observation, or handoff state failed validation."""


class SessionCleanupError(SessionStoreError):
    """An opaque state could not be deterministically released."""


class SessionCacheStatus(str, Enum):
    DISABLED = "disabled"
    MISS = "miss"
    HIT = "hit"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _name(value: str, field: str, *, maximum_bytes: int = 512) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field} must be a canonical non-empty string")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field} exceeds {maximum_bytes} UTF-8 bytes")
    return value


def _sha256(value: str, field: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest") from exc
    if value != value.lower():
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _nonnegative_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _token_row(values: tuple[int, ...], field: str) -> tuple[int, ...]:
    row = tuple(values)
    if not row:
        raise ValueError(f"{field} cannot be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
        for value in row
    ):
        raise ValueError(f"{field} must contain non-negative integer token IDs")
    return tuple(int(value) for value in row)


_TOKEN_PREFIX_DOMAIN = b"mrun.native-session.token-prefix.v1\0"


def _framed_token_id(value: int) -> bytes:
    width = max(1, (value.bit_length() + 7) // 8)
    encoded = value.to_bytes(width, "big")
    return width.to_bytes(8, "big") + encoded


def _token_prefix_sha256(values: tuple[int, ...]) -> str:
    """Return an incremental, unambiguous digest for one non-empty token row."""

    digest = hashlib.sha256(_TOKEN_PREFIX_DOMAIN)
    for value in values:
        digest.update(_framed_token_id(value))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _PrefixKey:
    identity_fingerprint: str
    runtime_id: str
    state_abi: str
    batch_size: int
    token_count: int
    token_prefix_sha256: str


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    """Complete semantic and physical identity for one in-memory session namespace."""

    model_id: str
    model_fingerprint: str
    chat_template_sha256: str
    semantic_token_count: int
    route_id: str
    runtime_id: str
    capability_fingerprint: str
    placement_fingerprint: str
    backend_id: str
    device_id: str
    state_abi: str
    execution_shape_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for field in ("model_id", "route_id", "runtime_id", "backend_id", "device_id", "state_abi"):
            object.__setattr__(self, field, _name(getattr(self, field), field))
        for field in (
            "model_fingerprint",
            "chat_template_sha256",
            "capability_fingerprint",
            "placement_fingerprint",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        object.__setattr__(
            self,
            "semantic_token_count",
            _positive_int(self.semantic_token_count, "semantic_token_count"),
        )
        if self.execution_shape_fingerprint is not None:
            object.__setattr__(
                self,
                "execution_shape_fingerprint",
                _sha256(
                    self.execution_shape_fingerprint,
                    "execution_shape_fingerprint",
                ),
            )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, str | int]:
        payload: dict[str, str | int] = {
            "model_id": self.model_id,
            "model_fingerprint": self.model_fingerprint,
            "chat_template_sha256": self.chat_template_sha256,
            "semantic_token_count": self.semantic_token_count,
            "route_id": self.route_id,
            "runtime_id": self.runtime_id,
            "capability_fingerprint": self.capability_fingerprint,
            "placement_fingerprint": self.placement_fingerprint,
            "backend_id": self.backend_id,
            "device_id": self.device_id,
            "state_abi": self.state_abi,
        }
        if self.execution_shape_fingerprint is not None:
            payload["execution_shape_fingerprint"] = self.execution_shape_fingerprint
        return payload

    def validate_route(self, route: RuntimeRoute) -> None:
        expected = (
            self.runtime_id,
            self.model_fingerprint,
            self.capability_fingerprint,
            self.placement_fingerprint,
            self.backend_id,
            self.device_id,
            self.execution_shape_fingerprint,
        )
        observed = (
            route.runtime_id,
            route.model_fingerprint,
            route.capability_fingerprint,
            route.placement_fingerprint,
            route.backend_id,
            route.device_id,
            route.execution_shape_fingerprint,
        )
        if observed != expected:
            raise SessionIdentityError("session identity does not bind the live runtime route")

    def validate_chat_boundary(
        self,
        *,
        model_id: str,
        chat_template_sha256: str,
        semantic_token_count: int,
        route_id: str,
    ) -> None:
        if (
            model_id != self.model_id
            or chat_template_sha256 != self.chat_template_sha256
            or semantic_token_count != self.semantic_token_count
            or route_id != self.route_id
        ):
            raise SessionIdentityError(
                "session identity does not bind the loaded model/template/domain/route"
            )


class _RetainedStateHandoff(Protocol):
    handoff_id: str
    runtime_id: str
    observation: StateObservation
    pending_token_id: int
    committed_token_ids: tuple[int, ...]

    @property
    def live(self) -> bool: ...

    def fork(
        self,
        runtime: ForkableModelRuntime,
        *,
        owner_id: str,
        capacity: int,
    ) -> StateForkResult: ...

    def release(self) -> bool: ...


@dataclass(slots=True)
class _Entry:
    entry_id: str
    session_id: str
    identity_fingerprint: str
    committed_token_ids: tuple[int, ...]
    ledger_sha256: str
    token_prefix_sha256: str
    pending_token_id: int
    observation: StateObservation
    handoff: _RetainedStateHandoff
    charge_bytes: int
    created_at: float
    last_access_at: float
    expires_at: float
    access_sequence: int
    pin_count: int = 0
    retired: bool = False


class SessionLease:
    """Opaque, single-use lease passed from the store to the generation coordinator.

    Public properties contain semantic IDs and accounting only.  The forked ``StateHandle`` is
    deliberately available solely through a private coordinator method.
    """

    def __init__(
        self,
        *,
        store: NativeSessionStore,
        lease_id: str,
        session_id: str,
        request_id: str,
        identity_fingerprint: str,
        status: SessionCacheStatus,
        prompt_token_ids: tuple[int, ...],
        prefix_token_count: int,
        state_capacity: int,
        reservation_bytes: int,
        reservation_slot: bool,
        cross_session_hit: bool = False,
    ) -> None:
        self.lease_id = lease_id
        self.session_id = session_id
        self.request_id = request_id
        self.identity_fingerprint = identity_fingerprint
        self.status = status
        self.prompt_token_ids = prompt_token_ids
        self.prefix_token_count = prefix_token_count
        self.state_capacity = state_capacity
        self._store = store
        self._reservation_bytes = reservation_bytes
        self._reservation_slot = reservation_slot
        self._cross_session_hit = cross_session_hit
        self._fork: StateForkResult | None = None
        self._claimed = False
        self._terminal = False
        self._lock = threading.Lock()

    @property
    def suffix_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids[self.prefix_token_count :]

    @property
    def hit(self) -> bool:
        return self.status is SessionCacheStatus.HIT

    @property
    def live(self) -> bool:
        with self._lock:
            return not self._terminal

    def _attach_fork(self, fork: StateForkResult) -> None:
        with self._lock:
            if (
                self._terminal
                or self._fork is not None
                or self.status is not SessionCacheStatus.HIT
            ):
                raise SessionIntegrityError("session lease cannot accept a fork receipt")
            self._fork = fork

    def _claim_state(
        self,
        store: NativeSessionStore,
        runtime: ForkableModelRuntime,
    ) -> StateHandle | None:
        with self._lock:
            if store is not self._store or runtime is not store._runtime:  # noqa: SLF001
                raise SessionIdentityError("session lease crossed its exact store/runtime owner")
            if self._terminal or self._claimed:
                raise SessionIntegrityError("session lease state was already consumed")
            self._claimed = True
            if self.status is SessionCacheStatus.MISS:
                return None
            if self._fork is None:
                raise SessionIntegrityError("prefix-hit lease is missing its native fork receipt")
            return self._fork.state

    def _finish(self) -> None:
        with self._lock:
            if self._terminal:
                raise SessionIntegrityError("session lease was already finalized")
            self._terminal = True


@dataclass(frozen=True, slots=True)
class SessionEntrySnapshot:
    session_id: str
    identity_fingerprint: str
    committed_token_count: int
    ledger_sha256: str
    pending_token_id: int
    state_capacity: int
    charge_bytes: int
    created_at: float
    last_access_at: float
    expires_at: float
    leased: bool


@dataclass(frozen=True, slots=True)
class SessionStoreTelemetry:
    identity_fingerprint: str
    entries: int
    active_leases: int
    stored_bytes: int
    reserved_bytes: int
    max_entries: int
    max_bytes: int
    hits: int
    misses: int
    installs: int
    aborts: int
    busy_rejections: int
    identity_rejections: int
    prefix_rejections: int
    capacity_rejections: int
    integrity_rejections: int
    forks: int
    fork_tokens: int
    fork_bytes: int
    ttl_evictions: int
    lru_evictions: int
    replacements: int
    manual_evictions: int
    cleanup_failures: int
    accepting: bool
    poisoned: bool
    reserved_slots: int = 0
    retired_entries: int = 0
    pinned_sources: int = 0
    cross_session_prefix_hits: int = 0
    cross_session_prefix_tokens: int = 0
    cross_session_prefix_bytes: int = 0

    @property
    def budget_reconciled(self) -> bool:
        return (
            self.stored_bytes + self.reserved_bytes <= self.max_bytes
            and self.entries + self.retired_entries + self.reserved_slots <= self.max_entries
        )


class NativeSessionStore:
    """Bounded TTL/LRU and content-prefix index for native committed state.

    Byte accounting charges allocated capacity rather than current logical length.  The formula
    supports both transformer state that grows per token and recurrent state that is fixed per
    row.  It is conservative for pageable runtimes and exact for fixed native arenas.
    """

    def __init__(
        self,
        runtime: ForkableModelRuntime,
        *,
        identity: SessionIdentity,
        state_bytes_per_token: int,
        state_fixed_bytes_per_row: int = 0,
        ttl_seconds: float = 900.0,
        max_entries: int = 128,
        max_bytes: int = 2 * 1024**3,
        clock: object | None = None,
    ) -> None:
        if not isinstance(runtime, ForkableModelRuntime):
            raise TypeError("session runtime must implement exact-prefix state fork")
        if not isinstance(identity, SessionIdentity):
            raise TypeError("identity must be SessionIdentity")
        identity.validate_route(runtime.route)
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise TypeError("ttl_seconds must be a finite positive number")
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("ttl_seconds must be a finite positive number")
        self._runtime = runtime
        self.identity = identity
        self._identity_fingerprint = identity.fingerprint
        self.state_bytes_per_token = _nonnegative_int(
            state_bytes_per_token,
            "state_bytes_per_token",
        )
        self.state_fixed_bytes_per_row = _nonnegative_int(
            state_fixed_bytes_per_row,
            "state_fixed_bytes_per_row",
        )
        if self.state_bytes_per_token == 0 and self.state_fixed_bytes_per_row == 0:
            raise ValueError("session state must have a per-token or fixed byte charge")
        self.ttl_seconds = ttl
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.max_bytes = _positive_int(max_bytes, "max_bytes")
        if clock is None:
            import time

            clock = time.monotonic
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._retired: dict[str, _Entry] = {}
        self._prefix_index: dict[_PrefixKey, dict[str, _Entry]] = {}
        self._entry_prefix_keys: dict[str, _PrefixKey] = {}
        self._prefix_lengths: dict[int, int] = {}
        self._active: dict[str, SessionLease] = {}
        self._stored_bytes = 0
        self._reserved_bytes = 0
        self._reserved_slots = 0
        self._sequence = 0
        self._entry_sequence = 0
        self._lease_sequence = 0
        self._last_now = 0.0
        self._accepting = True
        self._poisoned = False

        self._hits = 0
        self._misses = 0
        self._installs = 0
        self._aborts = 0
        self._busy_rejections = 0
        self._identity_rejections = 0
        self._prefix_rejections = 0
        self._capacity_rejections = 0
        self._integrity_rejections = 0
        self._forks = 0
        self._fork_tokens = 0
        self._fork_bytes = 0
        self._cross_session_prefix_hits = 0
        self._cross_session_prefix_tokens = 0
        self._cross_session_prefix_bytes = 0
        self._ttl_evictions = 0
        self._lru_evictions = 0
        self._replacements = 0
        self._manual_evictions = 0
        self._cleanup_failures = 0

    def _now(self) -> float:
        value = float(self._clock())  # type: ignore[operator]
        if not math.isfinite(value) or value < 0:
            raise SessionIntegrityError("session clock returned a non-finite negative timestamp")
        if value < self._last_now:
            self._poisoned = True
            raise SessionIntegrityError("session clock regressed")
        self._last_now = value
        return value

    def binds_runtime(self, runtime: object) -> bool:
        """Return whether ``runtime`` is the exact in-process state owner."""

        return runtime is self._runtime

    def _charge(self, capacity: int) -> int:
        return self.state_fixed_bytes_per_row + (
            _positive_int(capacity, "state_capacity") * self.state_bytes_per_token
        )

    def _assert_accounting_locked(self) -> None:
        retained = (*self._entries.values(), *self._retired.values())
        expected_prefix_lengths: dict[int, int] = {}
        for key in self._entry_prefix_keys.values():
            expected_prefix_lengths[key.token_count] = (
                expected_prefix_lengths.get(key.token_count, 0) + 1
            )
        valid = (
            self._stored_bytes == sum(entry.charge_bytes for entry in retained)
            and self._reserved_bytes
            == sum(lease._reservation_bytes for lease in self._active.values())  # noqa: SLF001
            and self._reserved_slots
            == sum(int(lease._reservation_slot) for lease in self._active.values())  # noqa: SLF001
            and self._stored_bytes >= 0
            and self._reserved_bytes >= 0
            and self._reserved_slots >= 0
            and len({entry.entry_id for entry in retained}) == len(retained)
            and len(self._entry_prefix_keys) == len(self._entries)
            and sum(len(bucket) for bucket in self._prefix_index.values()) == len(self._entries)
            and self._prefix_lengths == expected_prefix_lengths
        )
        if not valid:
            self._integrity_rejections += 1
            self._poisoned = True
            raise SessionIntegrityError("session byte/slot/index accounting did not reconcile")

    def _require_identity(self, identity: SessionIdentity) -> None:
        if not isinstance(identity, SessionIdentity) or identity != self.identity:
            self._identity_rejections += 1
            raise SessionIdentityError("session request crossed its bound identity")

    def _release_handoff(self, handoff: _RetainedStateHandoff) -> None:
        try:
            handoff.release()
        except BaseException as exc:
            self._cleanup_failures += 1
            self._poisoned = True
            raise SessionCleanupError(f"retained native state release failed: {exc}") from exc

    def _prefix_key(
        self,
        *,
        token_count: int,
        token_prefix_sha256: str,
    ) -> _PrefixKey:
        return _PrefixKey(
            identity_fingerprint=self._identity_fingerprint,
            runtime_id=self.identity.runtime_id,
            state_abi=self.identity.state_abi,
            batch_size=1,
            token_count=token_count,
            token_prefix_sha256=token_prefix_sha256,
        )

    def _index_entry_locked(self, entry: _Entry) -> None:
        key = self._prefix_key(
            token_count=len(entry.committed_token_ids),
            token_prefix_sha256=entry.token_prefix_sha256,
        )
        if entry.entry_id in self._entry_prefix_keys or entry.retired:
            self._poisoned = True
            raise SessionIntegrityError("session prefix index received a duplicate authority")
        bucket = self._prefix_index.setdefault(key, {})
        if entry.entry_id in bucket:
            self._poisoned = True
            raise SessionIntegrityError("session prefix index entry ID was reused")
        bucket[entry.entry_id] = entry
        self._entry_prefix_keys[entry.entry_id] = key
        self._prefix_lengths[key.token_count] = self._prefix_lengths.get(key.token_count, 0) + 1

    def _unindex_entry_locked(self, entry: _Entry) -> None:
        key = self._entry_prefix_keys.pop(entry.entry_id, None)
        removed = False
        if key is not None:
            length_count = self._prefix_lengths.get(key.token_count, 0)
            if length_count <= 1:
                self._prefix_lengths.pop(key.token_count, None)
            else:
                self._prefix_lengths[key.token_count] = length_count - 1
            bucket = self._prefix_index.get(key)
            if bucket is not None and bucket.get(entry.entry_id) is entry:
                bucket.pop(entry.entry_id)
                removed = True
                if not bucket:
                    self._prefix_index.pop(key)
        if removed:
            return

        # Preserve cleanup authority even if deliberate in-process corruption damaged the index.
        for candidate_key, bucket in tuple(self._prefix_index.items()):
            if bucket.get(entry.entry_id) is entry:
                bucket.pop(entry.entry_id)
                if not bucket:
                    self._prefix_index.pop(candidate_key)
                if key is None:
                    length_count = self._prefix_lengths.get(candidate_key.token_count, 0)
                    if length_count <= 1:
                        self._prefix_lengths.pop(candidate_key.token_count, None)
                    else:
                        self._prefix_lengths[candidate_key.token_count] = length_count - 1
                removed = True
        self._integrity_rejections += 1
        self._poisoned = True
        if not removed:
            raise SessionIntegrityError("session prefix index lost its entry authority")

    def _release_retired_entry_locked(self, entry: _Entry) -> None:
        if (
            not entry.retired
            or entry.pin_count != 0
            or self._retired.get(entry.entry_id) is not entry
        ):
            self._poisoned = True
            raise SessionIntegrityError("retired session authority failed its release gate")
        self._release_handoff(entry.handoff)
        self._retired.pop(entry.entry_id)
        self._stored_bytes -= entry.charge_bytes

    def _remove_entry_locked(self, session_id: str, *, reason: str) -> None:
        entry = self._entries.get(session_id)
        if entry is None:
            raise SessionIntegrityError("session removal lost its current entry authority")
        self._unindex_entry_locked(entry)
        self._entries.pop(session_id)
        entry.retired = True
        if entry.entry_id in self._retired:
            self._poisoned = True
            raise SessionIntegrityError("session entry retired through a reused authority ID")
        self._retired[entry.entry_id] = entry
        if reason == "ttl":
            self._ttl_evictions += 1
        elif reason == "lru":
            self._lru_evictions += 1
        elif reason == "manual":
            self._manual_evictions += 1
        if entry.pin_count == 0:
            self._release_retired_entry_locked(entry)

    def _unpin_entry_locked(self, entry: _Entry) -> None:
        if entry.pin_count <= 0:
            self._poisoned = True
            raise SessionIntegrityError("session source pin underflow")
        if entry.retired:
            if self._retired.get(entry.entry_id) is not entry:
                self._poisoned = True
                raise SessionIntegrityError("retired session source lost its ABA authority")
        elif self._entries.get(entry.session_id) is not entry:
            self._poisoned = True
            raise SessionIntegrityError("current session source lost its ABA authority")
        entry.pin_count -= 1
        if entry.retired and entry.pin_count == 0:
            self._release_retired_entry_locked(entry)

    def _prune_expired_locked(self, now: float) -> None:
        expired = sorted(
            (
                entry.expires_at,
                entry.session_id,
            )
            for entry in self._entries.values()
            if entry.expires_at <= now and entry.session_id not in self._active
        )
        for _expires, session_id in expired:
            self._remove_entry_locked(session_id, reason="ttl")

    def _reserve_locked(
        self,
        *,
        additional_bytes: int,
        additional_slots: int,
        protected_session_id: str,
        protected_entry_id: str | None,
    ) -> None:
        if additional_bytes > self.max_bytes or additional_slots > self.max_entries:
            self._capacity_rejections += 1
            raise SessionCapacityError("one retained state exceeds the session-store budget")
        while (
            self._stored_bytes + self._reserved_bytes + additional_bytes > self.max_bytes
            or len(self._entries) + len(self._retired) + self._reserved_slots + additional_slots
            > self.max_entries
        ):
            candidates = sorted(
                (
                    entry.access_sequence,
                    entry.session_id,
                )
                for entry in self._entries.values()
                if entry.session_id not in self._active
                and entry.session_id != protected_session_id
                and entry.entry_id != protected_entry_id
                and entry.pin_count == 0
            )
            if not candidates:
                self._capacity_rejections += 1
                raise SessionCapacityError(
                    "session-store budget is exhausted by active or protected entries"
                )
            self._remove_entry_locked(candidates[0][1], reason="lru")

    def _validate_entry_locked(self, entry: _Entry) -> None:
        try:
            ledger = _token_row(entry.committed_token_ids, "committed_token_ids")
            prefix_sha256 = _token_prefix_sha256(ledger)
            prefix_key = self._prefix_key(
                token_count=len(ledger),
                token_prefix_sha256=prefix_sha256,
            )
            valid = (
                self._entries.get(entry.session_id) is entry
                and not entry.retired
                and entry.pin_count >= 0
                and entry.entry_id.startswith("entry.")
                and entry.session_id in self._entries
                and entry.identity_fingerprint == self._identity_fingerprint
                and entry.ledger_sha256 == _canonical_sha256(list(ledger))
                and entry.token_prefix_sha256 == prefix_sha256
                and self._entry_prefix_keys.get(entry.entry_id) == prefix_key
                and self._prefix_index.get(prefix_key, {}).get(entry.entry_id) is entry
                and self._prefix_lengths.get(len(ledger), 0) > 0
                and entry.handoff.live
                and entry.handoff.runtime_id == self.identity.runtime_id
                and entry.handoff.observation == entry.observation
                and entry.handoff.committed_token_ids == ledger
                and entry.handoff.pending_token_id == entry.pending_token_id
                and entry.observation.runtime_id == self.identity.runtime_id
                and entry.observation.state_abi == self.identity.state_abi
                and entry.observation.batch_size == 1
                and entry.observation.lengths == (len(ledger),)
                and all(token < self.identity.semantic_token_count for token in ledger)
                and 0 <= entry.pending_token_id < self.identity.semantic_token_count
                and entry.charge_bytes == self._charge(entry.observation.capacity)
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            self._integrity_rejections += 1
            raise SessionIntegrityError(
                "stored session entry failed its exact identity/ledger gate"
            )

    def _longest_prefix_entry_locked(
        self,
        prompt: tuple[int, ...],
    ) -> _Entry | None:
        candidate_lengths = tuple(
            sorted(
                (token_count for token_count in self._prefix_lengths if token_count < len(prompt)),
                reverse=True,
            )
        )
        if not candidate_lengths:
            return None
        wanted = set(candidate_lengths)
        prefix_sha256s: dict[int, str] = {}
        digest = hashlib.sha256(_TOKEN_PREFIX_DOMAIN)
        maximum = candidate_lengths[0]
        for token_count, token_id in enumerate(prompt, start=1):
            if token_count > maximum:
                break
            digest.update(_framed_token_id(token_id))
            if token_count in wanted:
                prefix_sha256s[token_count] = digest.hexdigest()

        for token_count in candidate_lengths:
            key = self._prefix_key(
                token_count=token_count,
                token_prefix_sha256=prefix_sha256s[token_count],
            )
            bucket = self._prefix_index.get(key)
            if not bucket:
                continue
            for entry in sorted(bucket.values(), key=lambda item: (item.session_id, item.entry_id)):
                try:
                    self._validate_entry_locked(entry)
                except SessionIntegrityError:
                    self._remove_entry_locked(entry.session_id, reason="manual")
                    raise
                if entry.session_id in self._active:
                    continue
                # SHA-256 narrows the lookup; token equality remains authoritative.
                if entry.committed_token_ids == prompt[:token_count]:
                    return entry
        return None

    def acquire(
        self,
        *,
        session_id: str,
        request_id: str,
        identity: SessionIdentity,
        prompt_token_ids: tuple[int, ...],
        state_capacity: int,
    ) -> SessionLease:
        """Reserve one session mutation and optionally fork an exact committed prefix."""

        canonical_session = _name(session_id, "session_id", maximum_bytes=256)
        canonical_request = _name(request_id, "request_id", maximum_bytes=256)
        prompt = _token_row(prompt_token_ids, "prompt_token_ids")
        if any(token >= self.identity.semantic_token_count for token in prompt):
            raise SessionIdentityError("session prompt escaped the bound semantic token domain")
        capacity = _positive_int(state_capacity, "state_capacity")
        if capacity < len(prompt):
            raise SessionCapacityError("state capacity cannot hold the rendered session prompt")
        target_charge = self._charge(capacity)
        entry: _Entry | None = None
        cross_session_hit = False
        source_pinned = False
        with self._lock:
            now = self._now()
            self._require_identity(identity)
            if not self._accepting:
                raise SessionStoreClosed("session store is closed")
            if self._poisoned:
                raise SessionIntegrityError("session store is poisoned by a cleanup invariant")
            self._prune_expired_locked(now)
            if canonical_session in self._active:
                self._busy_rejections += 1
                raise SessionBusyError("session already has an active continuation lease")

            entry = self._entries.get(canonical_session)
            status = SessionCacheStatus.MISS
            prefix_count = 0
            reservation_slot = entry is None
            reservation_bytes = target_charge
            if entry is not None:
                try:
                    self._validate_entry_locked(entry)
                except SessionIntegrityError:
                    self._remove_entry_locked(canonical_session, reason="manual")
                    raise
                if entry.pin_count:
                    self._busy_rejections += 1
                    raise SessionBusyError(
                        "session source is pinned by a concurrent exact-prefix fork"
                    )
                prefix = entry.committed_token_ids
                if len(prompt) <= len(prefix) or prompt[: len(prefix)] != prefix:
                    self._prefix_rejections += 1
                    raise SessionPrefixMismatch(
                        "known session prompt must strictly extend its exact committed token ledger"
                    )
                status = SessionCacheStatus.HIT
                prefix_count = len(prefix)
                reservation_slot = False
                reservation_bytes = max(target_charge - entry.charge_bytes, 0)
            else:
                entry = self._longest_prefix_entry_locked(prompt)
                if entry is not None:
                    cross_session_hit = True
                    status = SessionCacheStatus.HIT
                    prefix_count = len(entry.committed_token_ids)

            if entry is not None:
                self._sequence += 1
                entry.access_sequence = self._sequence
                entry.last_access_at = now
                entry.expires_at = now + self.ttl_seconds

            self._reserve_locked(
                additional_bytes=reservation_bytes,
                additional_slots=int(reservation_slot),
                protected_session_id=canonical_session,
                protected_entry_id=None if entry is None else entry.entry_id,
            )
            if cross_session_hit:
                assert entry is not None
                entry.pin_count += 1
                source_pinned = True
            self._reserved_bytes += reservation_bytes
            self._reserved_slots += int(reservation_slot)
            self._lease_sequence += 1
            lease = SessionLease(
                store=self,
                lease_id=f"lease.{self._lease_sequence}",
                session_id=canonical_session,
                request_id=canonical_request,
                identity_fingerprint=self._identity_fingerprint,
                status=status,
                prompt_token_ids=prompt,
                prefix_token_count=prefix_count,
                state_capacity=capacity,
                reservation_bytes=reservation_bytes,
                reservation_slot=reservation_slot,
                cross_session_hit=cross_session_hit,
            )
            self._active[canonical_session] = lease
            if status is SessionCacheStatus.HIT:
                self._hits += 1
            else:
                self._misses += 1

        if entry is None:
            return lease
        fork: StateForkResult | None = None
        fork_attached = False
        try:
            fork = entry.handoff.fork(
                self._runtime,
                owner_id=f"session:{canonical_session}:{canonical_request}",
                capacity=capacity,
            )
            if (
                fork.runtime_id != self.identity.runtime_id
                or fork.source != entry.observation
                or fork.forked.runtime_id != self.identity.runtime_id
                or fork.forked.batch_size != 1
                or fork.forked.capacity != capacity
                or fork.forked.lengths != (prefix_count,)
                or fork.forked.state_abi != self.identity.state_abi
                or fork.state.observe() != fork.forked
            ):
                raise SessionIntegrityError(
                    "native state fork receipt drifted from the cache entry"
                )
            if source_pinned:
                with self._lock:
                    pin_count = entry.pin_count
                    try:
                        self._unpin_entry_locked(entry)
                    finally:
                        source_pinned = entry.pin_count == pin_count
            lease._attach_fork(fork)
            fork_attached = True
            with self._lock:
                self._forks += 1
                self._fork_tokens += prefix_count
                self._fork_bytes += fork.state_bytes_copied
                if cross_session_hit:
                    self._cross_session_prefix_hits += 1
                    self._cross_session_prefix_tokens += prefix_count
                    self._cross_session_prefix_bytes += fork.state_bytes_copied
            return lease
        except BaseException as exc:
            cleanup_errors: list[BaseException] = []
            if source_pinned:
                try:
                    with self._lock:
                        pin_count = entry.pin_count
                        try:
                            self._unpin_entry_locked(entry)
                        finally:
                            source_pinned = entry.pin_count == pin_count
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
            if fork is not None and not fork_attached:
                try:
                    self._runtime.release_state(fork.state)
                except BaseException as cleanup_exc:
                    with self._lock:
                        self._cleanup_failures += 1
                        self._poisoned = True
                    cleanup_errors.append(cleanup_exc)
            try:
                self.abort(lease)
            except BaseException as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
            if cleanup_errors:
                detail = "; ".join(str(error) for error in cleanup_errors)
                raise SessionCleanupError(
                    f"session fork failed ({exc}) and rollback failed ({detail})"
                ) from cleanup_errors[-1]
            if isinstance(exc, SessionStoreError):
                raise
            raise SessionIntegrityError(f"native exact-prefix fork failed: {exc}") from exc

    def claim_for_generation(self, lease: SessionLease) -> StateHandle | None:
        """Coordinator-only transfer of a forked child; no public cache payload is exposed."""

        if not isinstance(lease, SessionLease):
            raise TypeError("lease must be SessionLease")
        with self._lock:
            if self._active.get(lease.session_id) is not lease:
                raise SessionIntegrityError("session lease is stale or foreign")
        return lease._claim_state(self, self._runtime)

    def install(self, lease: SessionLease, handoff: _RetainedStateHandoff) -> None:
        """Atomically replace a leased entry with a successful extended state."""

        if not isinstance(lease, SessionLease):
            raise TypeError("lease must be SessionLease")
        ledger = _token_row(tuple(handoff.committed_token_ids), "committed_token_ids")
        observation = handoff.observation
        if (
            not handoff.live
            or handoff.runtime_id != self.identity.runtime_id
            or observation.runtime_id != self.identity.runtime_id
            or observation.state_abi != self.identity.state_abi
            or observation.batch_size != 1
            or observation.capacity != lease.state_capacity
            or observation.lengths != (len(ledger),)
            or ledger[: len(lease.prompt_token_ids)] != lease.prompt_token_ids
            or len(ledger) < len(lease.prompt_token_ids)
            or any(token >= self.identity.semantic_token_count for token in ledger)
            or handoff.pending_token_id < 0
            or handoff.pending_token_id >= self.identity.semantic_token_count
        ):
            self._integrity_rejections += 1
            raise SessionIntegrityError("completed session handoff failed exact ledger validation")

        old: _Entry | None
        with self._lock:
            if self._active.get(lease.session_id) is not lease or not lease.live:
                raise SessionIntegrityError("session completion used a stale or foreign lease")
            now = self._now()
            old = self._entries.get(lease.session_id)
            if lease._cross_session_hit and old is not None:  # noqa: SLF001
                raise SessionIntegrityError(
                    "cross-session target acquired a conflicting current authority"
                )
            if lease.status is SessionCacheStatus.MISS and old is not None:
                raise SessionIntegrityError("cache-miss target acquired a conflicting authority")
            if (
                lease.status is SessionCacheStatus.HIT
                and not lease._cross_session_hit  # noqa: SLF001
                and old is None
            ):
                raise SessionIntegrityError("same-session source disappeared before replacement")
            if old is not None:
                self._validate_entry_locked(old)
                if old.pin_count:
                    raise SessionIntegrityError(
                        "same-session replacement crossed an outstanding source pin"
                    )
                if old.handoff is handoff:
                    raise SessionIntegrityError(
                        "session replacement attempted to reuse one handoff authority"
                    )
            old_charge = old.charge_bytes if old is not None else 0
            new_charge = self._charge(observation.capacity)
            available = old_charge + lease._reservation_bytes  # noqa: SLF001
            if new_charge > available:
                raise SessionCapacityError("completed state exceeds its reserved session budget")
            self._sequence += 1
            self._entry_sequence += 1
            token_prefix_sha256 = _token_prefix_sha256(ledger)
            entry = _Entry(
                entry_id=f"entry.{self._entry_sequence}",
                session_id=lease.session_id,
                identity_fingerprint=self._identity_fingerprint,
                committed_token_ids=ledger,
                ledger_sha256=_canonical_sha256(list(ledger)),
                token_prefix_sha256=token_prefix_sha256,
                pending_token_id=int(handoff.pending_token_id),
                observation=observation,
                handoff=handoff,
                charge_bytes=new_charge,
                created_at=old.created_at if old is not None else now,
                last_access_at=now,
                expires_at=now + self.ttl_seconds,
                access_sequence=self._sequence,
            )
            if old is not None:
                self._remove_entry_locked(lease.session_id, reason="replacement")
            self._entries[lease.session_id] = entry
            self._index_entry_locked(entry)
            self._stored_bytes += new_charge
            self._reserved_bytes -= lease._reservation_bytes  # noqa: SLF001
            self._reserved_slots -= int(lease._reservation_slot)  # noqa: SLF001
            self._active.pop(lease.session_id)
            self._installs += 1
            if old is not None:
                self._replacements += 1
            lease._finish()
            if (
                self._stored_bytes + self._reserved_bytes > self.max_bytes
                or len(self._entries) + len(self._retired) + self._reserved_slots > self.max_entries
            ):
                self._poisoned = True

    def abort(self, lease: SessionLease, *, claimed_state_released: bool = False) -> bool:
        """End a failed/cancelled lease and retain the prior source entry, if any."""

        if not isinstance(lease, SessionLease):
            raise TypeError("lease must be SessionLease")
        fork_to_release: StateHandle | None = None
        with self._lock:
            current = self._active.get(lease.session_id)
            if current is None:
                return False
            if current is not lease:
                raise SessionIntegrityError("cannot abort a foreign session lease")
            with lease._lock:  # noqa: SLF001 - store owns lease lifecycle
                if lease._claimed and not claimed_state_released:  # noqa: SLF001
                    raise SessionIntegrityError(
                        "claimed session state must be released by the generation coordinator"
                    )
                if not lease._claimed and lease._fork is not None:  # noqa: SLF001
                    fork_to_release = lease._fork.state  # noqa: SLF001
            self._reserved_bytes -= lease._reservation_bytes  # noqa: SLF001
            self._reserved_slots -= int(lease._reservation_slot)  # noqa: SLF001
            self._active.pop(lease.session_id)
            self._aborts += 1
            lease._finish()
        if fork_to_release is not None:
            try:
                self._runtime.release_state(fork_to_release)
            except BaseException as exc:
                with self._lock:
                    self._cleanup_failures += 1
                    self._poisoned = True
                raise SessionCleanupError(f"forked session state release failed: {exc}") from exc
        return True

    def evict(self, session_id: str) -> bool:
        canonical = _name(session_id, "session_id", maximum_bytes=256)
        with self._lock:
            if canonical in self._active:
                raise SessionBusyError("cannot evict a session with an active lease")
            if canonical not in self._entries:
                return False
            self._remove_entry_locked(canonical, reason="manual")
            return True

    def inspect(self, session_id: str) -> SessionEntrySnapshot | None:
        canonical = _name(session_id, "session_id", maximum_bytes=256)
        with self._lock:
            now = self._now()
            self._prune_expired_locked(now)
            entry = self._entries.get(canonical)
            if entry is None:
                return None
            self._validate_entry_locked(entry)
            return SessionEntrySnapshot(
                session_id=entry.session_id,
                identity_fingerprint=entry.identity_fingerprint,
                committed_token_count=len(entry.committed_token_ids),
                ledger_sha256=entry.ledger_sha256,
                pending_token_id=entry.pending_token_id,
                state_capacity=entry.observation.capacity,
                charge_bytes=entry.charge_bytes,
                created_at=entry.created_at,
                last_access_at=entry.last_access_at,
                expires_at=entry.expires_at,
                leased=entry.session_id in self._active,
            )

    def telemetry(self) -> SessionStoreTelemetry:
        with self._lock:
            now = self._now()
            self._prune_expired_locked(now)
            self._assert_accounting_locked()
            return SessionStoreTelemetry(
                identity_fingerprint=self._identity_fingerprint,
                entries=len(self._entries),
                active_leases=len(self._active),
                stored_bytes=self._stored_bytes,
                reserved_bytes=self._reserved_bytes,
                reserved_slots=self._reserved_slots,
                retired_entries=len(self._retired),
                pinned_sources=sum(
                    entry.pin_count for entry in (*self._entries.values(), *self._retired.values())
                ),
                max_entries=self.max_entries,
                max_bytes=self.max_bytes,
                hits=self._hits,
                misses=self._misses,
                installs=self._installs,
                aborts=self._aborts,
                busy_rejections=self._busy_rejections,
                identity_rejections=self._identity_rejections,
                prefix_rejections=self._prefix_rejections,
                capacity_rejections=self._capacity_rejections,
                integrity_rejections=self._integrity_rejections,
                forks=self._forks,
                fork_tokens=self._fork_tokens,
                fork_bytes=self._fork_bytes,
                cross_session_prefix_hits=self._cross_session_prefix_hits,
                cross_session_prefix_tokens=self._cross_session_prefix_tokens,
                cross_session_prefix_bytes=self._cross_session_prefix_bytes,
                ttl_evictions=self._ttl_evictions,
                lru_evictions=self._lru_evictions,
                replacements=self._replacements,
                manual_evictions=self._manual_evictions,
                cleanup_failures=self._cleanup_failures,
                accepting=self._accepting,
                poisoned=self._poisoned,
            )

    def close(self) -> None:
        with self._lock:
            self._accepting = False
            if self._active:
                raise SessionBusyError("cannot close session store while leases are active")
            if any(entry.pin_count for entry in (*self._entries.values(), *self._retired.values())):
                self._poisoned = True
                raise SessionBusyError("cannot close session store while source pins remain")
            retired_before = tuple(sorted(self._retired))
            errors: list[BaseException] = []
            for session_id in sorted(tuple(self._entries)):
                try:
                    self._remove_entry_locked(session_id, reason="manual")
                except BaseException as exc:
                    errors.append(exc)
            for entry_id in retired_before:
                entry = self._retired.get(entry_id)
                if entry is None:
                    continue
                try:
                    self._release_retired_entry_locked(entry)
                except BaseException as exc:
                    errors.append(exc)
            if errors:
                detail = "; ".join(str(error) for error in errors)
                raise SessionCleanupError(
                    f"session store close retained {len(errors)} native authorities: {detail}"
                ) from errors[-1]
            self._assert_accounting_locked()


__all__ = [
    "NativeSessionStore",
    "SessionBusyError",
    "SessionCacheStatus",
    "SessionCapacityError",
    "SessionCleanupError",
    "SessionEntrySnapshot",
    "SessionIdentity",
    "SessionIdentityError",
    "SessionIntegrityError",
    "SessionLease",
    "SessionPrefixMismatch",
    "SessionStoreClosed",
    "SessionStoreError",
    "SessionStoreTelemetry",
]
