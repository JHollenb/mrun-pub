"""Public copy-on-write KV StateCuts over the bounded-memory paged QStore runtime.

The parent arena is allocated and prefetched once. Branches retain provisional K/V diffs only;
commit delegates to the established staged, epoch-checked atomic block primitive, while abandon
proves the parent arena never changed. Serializable records contain metadata and byte counts,
never the parent K/V tensors or branch tensor payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import torch

from .kernels import paged_forward as pf

STATECUT_SCHEMA = "mrun-paged-kv-statecut-v1"
STATECUT_CONTINUATION_SCHEMA = "mrun-paged-kv-statecut-continuation-v1"
STATECUT_RECEIPT_SCHEMA = "mrun-paged-kv-statecut-transaction-v1"
STATECUT_SCREENING_SCHEMA = "mrun-paged-kv-statecut-screening-v1"
STATECUT_ADJUDICATION_SCHEMA = "mrun-paged-kv-statecut-adjudication-v1"


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    try:
        version = int(tensor._version)  # noqa: SLF001 - exact same-process mutation guard
    except RuntimeError as exc:
        raise TypeError("StateCut parent tensors must track mutation versions") from exc
    return (
        id(tensor),
        int(tensor.data_ptr()),
        tuple(int(value) for value in tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        bool(tensor.is_contiguous()),
        version,
    )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _positive_byte_budget(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field} must be positive")
    return normalized


def _cuda_free_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("StateCut CUDA projection requested without an available CUDA device")
    return int(torch.cuda.mem_get_info(device)[0])


def _normalize_selected_rows(rows: Sequence[int], *, vocab_size: int) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in rows)
    if not normalized:
        raise ValueError("selected-row screening requires at least one vocabulary row")
    if len(normalized) != len(set(normalized)):
        raise ValueError("selected vocabulary rows must be unique")
    if min(normalized) < 0 or max(normalized) >= int(vocab_size):
        raise ValueError(f"selected vocabulary rows must be inside [0, {vocab_size})")
    return normalized


def _optional_rank(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("branch rank must be an integer or None")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError("branch rank must be positive")
    return normalized


def _optional_route(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value.strip() != value or len(value) > 128:
        raise ValueError(
            "branch route must be a trimmed non-empty string of at most 128 characters"
        )
    return value


def _optional_sign(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("branch sign must be -1, 0, 1, or None")
    normalized = int(value)
    if normalized not in {-1, 0, 1}:
        raise ValueError("branch sign must be -1, 0, 1, or None")
    return normalized


def _optional_dose(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("branch dose must be a finite non-negative number or None")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("branch dose must be a finite non-negative number or None")
    return normalized


@dataclass(frozen=True, slots=True)
class _FrozenLMHeadGuard:
    store_id: int
    kind: str
    identity: tuple[object, ...]

    @classmethod
    def capture(cls, store: Any) -> _FrozenLMHeadGuard:
        content_guard = getattr(store, "assert_content_identity_unchanged", None)
        if callable(content_guard):
            content_guard()
            return cls(
                store_id=id(store),
                kind="qstore-content-identity",
                identity=(
                    getattr(store, "source_checkpoint_sha256", None),
                    getattr(store, "derived_store_sha256", None),
                    getattr(store, "manifest_semantic_sha256", None),
                    tuple(getattr(store, "_verified_file_stats", ())),
                    str(getattr(store, "compute_dtype", torch.float32)),
                ),
            )
        head = getattr(store, "head", None)
        if isinstance(head, torch.Tensor):
            return cls(
                store_id=id(store),
                kind="tensor-version",
                identity=_tensor_signature(head),
            )
        raise NotImplementedError(
            "StateCut scoring requires a QStore content guard or a versioned tensor LM head"
        )

    def assert_unchanged(self, store: Any) -> None:
        if id(store) != self.store_id:
            raise RuntimeError("StateCut frozen LM head store changed")
        current = self.capture(store)
        if current != self:
            raise RuntimeError("StateCut frozen LM head changed after screening began")

    @property
    def sha256(self) -> str:
        return _sha256_json(
            {
                "store_id": self.store_id,
                "kind": self.kind,
                "identity": self.identity,
            }
        )


@dataclass(frozen=True, slots=True)
class _ParentGuard:
    cache_id: str
    epoch: int
    lengths: tuple[int, ...]
    key_signature: tuple[object, ...]
    value_signature: tuple[object, ...]

    @classmethod
    def capture(cls, cache: pf.BatchedPagedKVCache) -> _ParentGuard:
        if not isinstance(cache, pf.BatchedPagedKVCache):
            raise TypeError("StateCut parent must be a BatchedPagedKVCache")
        cache.assert_usable()
        return cls(
            cache_id=str(cache.cache_id),
            epoch=int(cache.epoch),
            lengths=tuple(int(value) for value in cache.lengths),
            key_signature=_tensor_signature(cache.k),
            value_signature=_tensor_signature(cache.v),
        )

    @property
    def sha256(self) -> str:
        return _sha256_json(
            {
                "cache_id": self.cache_id,
                "epoch": self.epoch,
                "lengths": self.lengths,
                "key_signature": self.key_signature,
                "value_signature": self.value_signature,
            }
        )


def _cache_allocated_bytes(cache: pf.BatchedPagedKVCache) -> int:
    return int(
        cache.k.numel() * cache.k.element_size()
        + cache.v.numel() * cache.v.element_size()
    )


def _cache_committed_bytes(cache: pf.BatchedPagedKVCache) -> int:
    per_token = (
        int(cache.k.shape[0])
        * int(cache.k.shape[3])
        * int(cache.k.shape[4])
        * 2
        * int(cache.k.element_size())
    )
    return int(sum(int(value) for value in cache.lengths) * per_token)


def _require_branch_id(branch_id: str) -> str:
    if (
        not isinstance(branch_id, str)
        or not branch_id
        or branch_id.strip() != branch_id
        or len(branch_id) > 128
    ):
        raise ValueError("branch_id must be a trimmed non-empty string of at most 128 characters")
    return branch_id


@dataclass(frozen=True, slots=True)
class PagedKVStateCutDescriptor:
    cut_id: str
    cache_id: str
    parent_epoch: int
    parent_lengths: tuple[int, ...]
    batch_size: int
    capacity: int
    numerical_contract: str
    parent_allocated_bytes: int
    parent_committed_bytes: int
    retention_budget_bytes: int
    runtime_storage_signature_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATECUT_SCHEMA,
            "cut_id": self.cut_id,
            "cache_id": self.cache_id,
            "parent_epoch": self.parent_epoch,
            "parent_lengths": self.parent_lengths,
            "batch_size": self.batch_size,
            "capacity": self.capacity,
            "numerical_contract": self.numerical_contract,
            "parent_allocated_bytes": self.parent_allocated_bytes,
            "parent_committed_bytes": self.parent_committed_bytes,
            "retention_budget_bytes": self.retention_budget_bytes,
            "runtime_storage_signature_sha256": self.runtime_storage_signature_sha256,
            "copy_on_write": True,
            "duplicated_parent_kv_bytes": 0,
            "serialized_parent_kv_bytes": 0,
        }

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class PagedKVStateCutBranch:
    cut_id: str
    branch_id: str
    parent_epoch: int
    parent_lengths: tuple[int, ...]
    rank: int | None = None
    route: str | None = None
    sign: int | None = None
    dose: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cut_id": self.cut_id,
            "branch_id": self.branch_id,
            "parent_epoch": self.parent_epoch,
            "parent_lengths": self.parent_lengths,
            "rank": self.rank,
            "route": self.route,
            "sign": self.sign,
            "dose": self.dose,
            "copy_on_write": True,
        }


@dataclass(frozen=True, slots=True)
class PagedKVStateCutContinuation:
    cut_id: str
    branch_ids: tuple[str, ...]
    parent_epoch: int
    parent_lengths: tuple[int, ...]
    output_contract: str
    batched_outputs: torch.Tensor
    outputs: tuple[torch.Tensor, ...]
    physical_forwards: int
    retained_diff_bytes: int
    per_branch_diff_bytes: int
    parent_copy_bytes: int
    retained_total_bytes: int
    arithmetic: str = "batch_invariant"

    @property
    def logical_branches(self) -> int:
        return len(self.branch_ids)

    @property
    def logical_suffix_rows(self) -> int:
        return len(self.branch_ids) * len(self.parent_lengths)

    def output_for(self, branch_id: str) -> torch.Tensor:
        normalized = _require_branch_id(branch_id)
        try:
            index = self.branch_ids.index(normalized)
        except ValueError as exc:
            raise KeyError(f"unknown StateCut branch {normalized!r}") from exc
        return self.outputs[index]

    def to_dict(self) -> dict[str, Any]:
        output_metadata = tuple(
            {
                "branch_id": branch_id,
                "shape": tuple(int(value) for value in output.shape),
                "dtype": str(output.dtype),
                "device": str(output.device),
                "serialized_tensor_bytes": 0,
            }
            for branch_id, output in zip(self.branch_ids, self.outputs, strict=True)
        )
        return {
            "schema": STATECUT_CONTINUATION_SCHEMA,
            "cut_id": self.cut_id,
            "branch_ids": self.branch_ids,
            "parent_epoch": self.parent_epoch,
            "parent_lengths": self.parent_lengths,
            "output_contract": self.output_contract,
            "outputs": output_metadata,
            "logical_branches": self.logical_branches,
            "logical_suffix_rows": self.logical_suffix_rows,
            "physical_forwards": self.physical_forwards,
            "arithmetic": self.arithmetic,
            "suffix_replays": 0,
            "retained_diff_bytes": self.retained_diff_bytes,
            "per_branch_diff_bytes": self.per_branch_diff_bytes,
            "parent_copy_bytes": self.parent_copy_bytes,
            "retained_total_bytes": self.retained_total_bytes,
            "metadata_only_serialization": True,
            "serialized_parent_kv_bytes": 0,
            "serialized_diff_tensor_bytes": 0,
        }


@dataclass(frozen=True, slots=True)
class PagedKVStateCutProjectionAccounting:
    logical_branches: int
    logical_parent_rows_per_branch: int
    logical_suffix_rows: int
    logical_score_values: int
    physical_suffix_forwards_total: int
    physical_suffix_forwards_this_call: int
    suffix_replays: int
    physical_selected_head_gathers_this_call: int
    physical_full_head_traversals_this_call: int
    physical_head_blocks_this_call: int

    def to_dict(self) -> dict[str, int]:
        return {
            "logical_branches": self.logical_branches,
            "logical_parent_rows_per_branch": self.logical_parent_rows_per_branch,
            "logical_suffix_rows": self.logical_suffix_rows,
            "logical_score_values": self.logical_score_values,
            "physical_suffix_forwards_total": self.physical_suffix_forwards_total,
            "physical_suffix_forwards_this_call": self.physical_suffix_forwards_this_call,
            "suffix_replays": self.suffix_replays,
            "physical_selected_head_gathers_this_call": (
                self.physical_selected_head_gathers_this_call
            ),
            "physical_full_head_traversals_this_call": (
                self.physical_full_head_traversals_this_call
            ),
            "physical_head_blocks_this_call": self.physical_head_blocks_this_call,
        }


@dataclass(frozen=True, slots=True)
class PagedKVStateCutProjectionMemory:
    retained_before_bytes: int
    retained_output_bytes: int
    retained_after_bytes: int
    retention_budget_bytes: int
    device_input_bytes: int
    device_weight_peak_bytes: int
    device_output_peak_bytes: int
    planned_device_scratch_peak_bytes: int
    device_scratch_budget_bytes: int
    cuda_free_before_bytes: int | None
    head_row_chunk_size: int

    def to_dict(self) -> dict[str, int | None]:
        return {
            "retained_before_bytes": self.retained_before_bytes,
            "retained_output_bytes": self.retained_output_bytes,
            "retained_after_bytes": self.retained_after_bytes,
            "retention_budget_bytes": self.retention_budget_bytes,
            "device_input_bytes": self.device_input_bytes,
            "device_weight_peak_bytes": self.device_weight_peak_bytes,
            "device_output_peak_bytes": self.device_output_peak_bytes,
            "planned_device_scratch_peak_bytes": self.planned_device_scratch_peak_bytes,
            "device_scratch_budget_bytes": self.device_scratch_budget_bytes,
            "cuda_free_before_bytes": self.cuda_free_before_bytes,
            "head_row_chunk_size": self.head_row_chunk_size,
        }


@dataclass(frozen=True, slots=True)
class PagedKVStateCutScreening:
    cut_id: str
    branch_ids: tuple[str, ...]
    branch_axis: tuple[PagedKVStateCutBranch, ...]
    selected_rows: tuple[int, ...]
    scores: torch.Tensor
    selected_winner_token_ids: torch.Tensor
    global_argmax_established: bool
    scope_statement: str
    frozen_lm_head_sha256: str
    accounting: PagedKVStateCutProjectionAccounting
    memory: PagedKVStateCutProjectionMemory

    def scores_for(self, branch_id: str) -> torch.Tensor:
        normalized = _require_branch_id(branch_id)
        try:
            index = self.branch_ids.index(normalized)
        except ValueError as exc:
            raise KeyError(f"unknown StateCut branch {normalized!r}") from exc
        return self.scores[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATECUT_SCREENING_SCHEMA,
            "cut_id": self.cut_id,
            "branch_ids": self.branch_ids,
            "branch_axis": tuple(branch.to_dict() for branch in self.branch_axis),
            "selected_rows": self.selected_rows,
            "scores": {
                "shape": tuple(int(value) for value in self.scores.shape),
                "dtype": str(self.scores.dtype),
                "device": str(self.scores.device),
                "serialized_tensor_bytes": 0,
            },
            "selected_winner_token_ids": tuple(
                tuple(int(value) for value in row)
                for row in self.selected_winner_token_ids.tolist()
            ),
            "global_argmax_established": self.global_argmax_established,
            "scope_statement": self.scope_statement,
            "frozen_lm_head_sha256": self.frozen_lm_head_sha256,
            "accounting": self.accounting.to_dict(),
            "memory": self.memory.to_dict(),
            "suffix_reused": True,
            "metadata_only_serialization": True,
            "serialized_parent_kv_bytes": 0,
            "serialized_diff_tensor_bytes": 0,
            "serialized_score_tensor_bytes": 0,
        }


@dataclass(frozen=True, slots=True)
class PagedKVStateCutAdjudication:
    cut_id: str
    branch_ids: tuple[str, ...]
    branch_axis: tuple[PagedKVStateCutBranch, ...]
    logits: torch.Tensor
    global_argmax_token_ids: torch.Tensor
    global_argmax_established: bool
    screened_rows: tuple[int, ...]
    selected_screening_parity_verified: bool
    frozen_lm_head_sha256: str
    accounting: PagedKVStateCutProjectionAccounting
    memory: PagedKVStateCutProjectionMemory

    def logits_for(self, branch_id: str) -> torch.Tensor:
        normalized = _require_branch_id(branch_id)
        try:
            index = self.branch_ids.index(normalized)
        except ValueError as exc:
            raise KeyError(f"unknown StateCut branch {normalized!r}") from exc
        return self.logits[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATECUT_ADJUDICATION_SCHEMA,
            "cut_id": self.cut_id,
            "branch_ids": self.branch_ids,
            "branch_axis": tuple(branch.to_dict() for branch in self.branch_axis),
            "logits": {
                "shape": tuple(int(value) for value in self.logits.shape),
                "dtype": str(self.logits.dtype),
                "device": str(self.logits.device),
                "serialized_tensor_bytes": 0,
            },
            "global_argmax_token_ids": tuple(
                tuple(int(value) for value in row)
                for row in self.global_argmax_token_ids.tolist()
            ),
            "global_argmax_established": self.global_argmax_established,
            "screened_rows": self.screened_rows,
            "selected_screening_parity_verified": (
                self.selected_screening_parity_verified
            ),
            "frozen_lm_head_sha256": self.frozen_lm_head_sha256,
            "accounting": self.accounting.to_dict(),
            "memory": self.memory.to_dict(),
            "suffix_reused": True,
            "metadata_only_serialization": True,
            "serialized_parent_kv_bytes": 0,
            "serialized_diff_tensor_bytes": 0,
            "serialized_logit_tensor_bytes": 0,
        }


@dataclass(frozen=True, slots=True)
class PagedKVStateCutReceipt:
    action: Literal["commit", "abandon"]
    cut_id: str
    committed_branch_id: str | None
    abandoned_branch_ids: tuple[str, ...]
    parent_epoch_before: int
    parent_epoch_after: int
    parent_lengths_before: tuple[int, ...]
    parent_lengths_after: tuple[int, ...]
    transition_verified: bool
    exact_parent_restored: bool
    parent_storage_unchanged: bool
    parent_guard_before_sha256: str
    parent_guard_after_sha256: str
    retained_diff_bytes: int
    parent_allocated_bytes: int
    physical_forwards: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATECUT_RECEIPT_SCHEMA,
            "action": self.action,
            "cut_id": self.cut_id,
            "committed_branch_id": self.committed_branch_id,
            "abandoned_branch_ids": self.abandoned_branch_ids,
            "parent_epoch_before": self.parent_epoch_before,
            "parent_epoch_after": self.parent_epoch_after,
            "parent_lengths_before": self.parent_lengths_before,
            "parent_lengths_after": self.parent_lengths_after,
            "transition_verified": self.transition_verified,
            "exact_parent_restored": self.exact_parent_restored,
            "parent_storage_unchanged": self.parent_storage_unchanged,
            "parent_guard_before_sha256": self.parent_guard_before_sha256,
            "parent_guard_after_sha256": self.parent_guard_after_sha256,
            "retained_diff_bytes": self.retained_diff_bytes,
            "parent_allocated_bytes": self.parent_allocated_bytes,
            "physical_forwards": self.physical_forwards,
            "metadata_only_serialization": True,
            "serialized_parent_kv_bytes": 0,
            "serialized_diff_tensor_bytes": 0,
        }

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict())


class PagedKVStateCut:
    """One immutable parent, many copy-on-write branch blocks, one terminal decision."""

    def __init__(
        self,
        engine: Any,
        cache: pf.BatchedPagedKVCache,
        *,
        retention_budget_bytes: int,
    ) -> None:
        self._validate_engine(engine)
        if not isinstance(cache, pf.BatchedPagedKVCache):
            raise TypeError("StateCut cache must be a BatchedPagedKVCache")
        if isinstance(retention_budget_bytes, bool) or not isinstance(
            retention_budget_bytes, Integral
        ):
            raise TypeError("retention_budget_bytes must be an integer")
        if int(retention_budget_bytes) <= 0:
            raise ValueError("retention_budget_bytes must be positive")
        if _cache_allocated_bytes(cache) > int(retention_budget_bytes):
            raise MemoryError("prefilled parent KV exceeds the StateCut retention budget")
        self._engine = engine
        self._cache = cache
        self._budget = int(retention_budget_bytes)
        self._lock = threading.RLock()
        self._cut_id = uuid4().hex
        with cache._lock:  # noqa: SLF001 - binds exclusive StateCut ownership metadata
            cache.assert_usable()
            active = getattr(cache, "_active_statecut_id", None)
            if active is not None:
                raise RuntimeError("paged KV cache already has an active StateCut")
            cache._active_statecut_id = self._cut_id  # type: ignore[attr-defined]  # noqa: SLF001
            self._parent_guard = _ParentGuard.capture(cache)
        self._branches: dict[str, PagedKVStateCutBranch] = {}
        self._panel: pf.PagedKVForkPanel | None = None
        self._continuation: PagedKVStateCutContinuation | None = None
        self._frozen_lm_head: _FrozenLMHeadGuard | None = None
        self._screening: PagedKVStateCutScreening | None = None
        self._adjudication: PagedKVStateCutAdjudication | None = None
        self._terminal = False

    @staticmethod
    def _validate_engine(engine: Any) -> None:
        capabilities_method = getattr(engine, "capabilities", None)
        if not callable(capabilities_method):
            raise TypeError("StateCut engine must expose capabilities()")
        capabilities = capabilities_method()
        if getattr(capabilities, "transactional_kv", None) is not True:
            raise NotImplementedError("engine does not advertise transactional KV")
        if str(getattr(engine, "arch", "")) not in {"qwen2", "qwen3", "llama"}:
            raise NotImplementedError("paged KV StateCuts require qwen2, qwen3, or llama")
        store = getattr(engine, "store", None)
        if store is None or not isinstance(getattr(store, "cfg", None), Mapping):
            raise TypeError("StateCut engine must expose a configured paged QStore")
        required = (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        )
        missing = tuple(key for key in required if key not in store.cfg)
        if missing:
            raise ValueError(f"StateCut store config is missing fields: {missing}")
        if not callable(getattr(store, "matmul_row_stable", None)):
            raise NotImplementedError(
                "StateCut branches require the row-stable paged QStore arithmetic lane"
            )

    @classmethod
    def prefill(
        cls,
        engine: Any,
        ids_list: Sequence[np.ndarray | Sequence[int]],
        *,
        retention_budget_bytes: int,
        capacity: int | None = None,
    ) -> tuple[PagedKVStateCut, torch.Tensor]:
        """Allocate the smallest parent arena, prefill it once, then seal it as immutable."""

        cls._validate_engine(engine)
        if isinstance(retention_budget_bytes, bool) or not isinstance(
            retention_budget_bytes, Integral
        ):
            raise TypeError("retention_budget_bytes must be an integer")
        if int(retention_budget_bytes) <= 0:
            raise ValueError("retention_budget_bytes must be positive")
        rows = tuple(np.asarray(row, dtype=np.int64) for row in ids_list)
        if not rows:
            raise ValueError("StateCut prefill requires at least one parent row")
        if any(row.ndim != 1 or int(row.size) <= 0 for row in rows):
            raise ValueError("StateCut prefill rows must be non-empty and one-dimensional")
        minimum_capacity = max(int(row.size) for row in rows) + 1
        normalized_capacity = minimum_capacity if capacity is None else capacity
        if isinstance(normalized_capacity, bool) or not isinstance(normalized_capacity, Integral):
            raise TypeError("StateCut capacity must be an integer")
        normalized_capacity = int(normalized_capacity)
        if normalized_capacity < minimum_capacity:
            raise ValueError(
                "StateCut capacity must hold the complete parent plus one continuation token"
            )
        cfg = engine.store.cfg
        planned_parent_bytes = (
            2
            * int(cfg["num_hidden_layers"])
            * len(rows)
            * normalized_capacity
            * int(cfg["num_key_value_heads"])
            * int(cfg["head_dim"])
            * torch.float32.itemsize
        )
        prefill_output_bytes = (
            len(rows) * int(cfg["hidden_size"]) * torch.float32.itemsize
        )
        if planned_parent_bytes + prefill_output_bytes > int(retention_budget_bytes):
            raise MemoryError("planned parent KV exceeds the StateCut retention budget")
        cache = pf.BatchedPagedKVCache(
            int(cfg["num_hidden_layers"]),
            len(rows),
            int(cfg["num_key_value_heads"]),
            int(cfg["head_dim"]),
            normalized_capacity,
            getattr(engine.store, "device", "cpu"),
        )
        engine_lock = getattr(engine, "_execution_lock", None)
        if engine_lock is None:
            engine_lock = engine._execution_lock = threading.RLock()
        with engine_lock:
            prefill_output = pf.paged_forward_kv_batch(
                engine.store,
                list(rows),
                cache,
                output_contract="hidden_state_only",
            )
        return (
            cls(
                engine,
                cache,
                retention_budget_bytes=int(retention_budget_bytes),
            ),
            prefill_output,
        )

    @property
    def descriptor(self) -> PagedKVStateCutDescriptor:
        guard = self._parent_guard
        return PagedKVStateCutDescriptor(
            cut_id=self._cut_id,
            cache_id=guard.cache_id,
            parent_epoch=guard.epoch,
            parent_lengths=guard.lengths,
            batch_size=self._cache.B,
            capacity=self._cache.capacity,
            numerical_contract=str(
                getattr(self._engine, "numerical_contract", "paged-qstore-established")
            ),
            parent_allocated_bytes=_cache_allocated_bytes(self._cache),
            parent_committed_bytes=_cache_committed_bytes(self._cache),
            retention_budget_bytes=self._budget,
            runtime_storage_signature_sha256=guard.sha256,
        )

    @property
    def cache(self) -> pf.BatchedPagedKVCache:
        """The bound arena for low-level integrations; mutation invalidates this StateCut."""

        return self._cache

    def _require_open(self) -> None:
        if self._terminal:
            raise RuntimeError("StateCut transaction is terminal")

    def _assert_parent_unchanged(self) -> _ParentGuard:
        current = _ParentGuard.capture(self._cache)
        if current != self._parent_guard:
            raise RuntimeError("StateCut immutable parent changed outside the transaction")
        return current

    def fork(
        self,
        branch_id: str,
        *,
        rank: int | None = None,
        route: str | None = None,
        sign: int | None = None,
        dose: float | None = None,
    ) -> PagedKVStateCutBranch:
        with self._lock:
            self._require_open()
            self._assert_parent_unchanged()
            if self._panel is not None:
                raise RuntimeError("cannot add a branch after continuation")
            normalized = _require_branch_id(branch_id)
            if normalized in self._branches:
                raise ValueError(f"duplicate StateCut branch {normalized!r}")
            branch = PagedKVStateCutBranch(
                cut_id=self._cut_id,
                branch_id=normalized,
                parent_epoch=self._parent_guard.epoch,
                parent_lengths=self._parent_guard.lengths,
                rank=_optional_rank(rank),
                route=_optional_route(route),
                sign=_optional_sign(sign),
                dose=_optional_dose(dose),
            )
            self._branches[normalized] = branch
            return branch

    def continue_one(
        self,
        tokens_by_branch: Mapping[str, Sequence[int] | np.ndarray],
        *,
        output_contract: pf.PagedBlockOutputContract = "hidden_state_only",
        selected_rows: Sequence[int] = (),
        arithmetic: pf.PagedPooledArithmetic = "batch_invariant",
    ) -> PagedKVStateCutContinuation:
        """Continue every fork one token (see :meth:`continue_block` for ``arithmetic``)."""
        if not isinstance(tokens_by_branch, Mapping):
            raise TypeError("tokens_by_branch must be a branch-ID mapping")
        normalized: dict[str, np.ndarray] = {}
        for branch_id, tokens in tokens_by_branch.items():
            row = np.asarray(tokens, dtype=np.int64)
            if row.ndim != 1 or tuple(row.shape) != (self._cache.B,):
                raise ValueError(
                    "each branch must supply exactly one token per parent batch row"
                )
            normalized[branch_id] = row.reshape(self._cache.B, 1)
        return self.continue_block(
            normalized,
            output_contract=output_contract,
            selected_rows=selected_rows,
            arithmetic=arithmetic,
        )

    def continue_block(
        self,
        tokens_by_branch: Mapping[str, Sequence[Sequence[int]] | np.ndarray],
        *,
        output_contract: pf.PagedBlockOutputContract = "hidden_state_only",
        selected_rows: Sequence[int] = (),
        arithmetic: pf.PagedPooledArithmetic = "batch_invariant",
    ) -> PagedKVStateCutContinuation:
        """Continue every fork by the same positive K-token provisional block.

        No token becomes visible in the parent until :meth:`commit` admits the
        complete block. Abandon therefore restores the exact pre-block parent.

        ``arithmetic`` names the numerical contract and is recorded on the continuation.
        ``batch_invariant`` (default, ``batch-invariant-triton-fp32-v1``) packs all branches into
        one weight traversal with fixed-bracket kernels, so each branch's bits are independent of
        the panel width. ``row_stable_split`` reproduces the pre-2026-09-24 contract (independent
        cuBLAS B=1 bits, one request-local call per branch); select it to replay older receipts.
        """
        if arithmetic not in {"row_stable_split", "batch_invariant"}:
            raise ValueError("StateCut arithmetic must be 'row_stable_split' or 'batch_invariant'")

        with self._lock:
            self._require_open()
            self._assert_parent_unchanged()
            if self._panel is not None:
                raise RuntimeError("StateCut branches have already continued")
            if not self._branches:
                raise RuntimeError("StateCut continuation requires at least one fork")
            if not isinstance(tokens_by_branch, Mapping):
                raise TypeError("tokens_by_branch must be a branch-ID mapping")
            branch_ids = tuple(self._branches)
            if set(tokens_by_branch) != set(branch_ids):
                raise ValueError("continuation tokens must cover every fork exactly once")
            rows: list[np.ndarray] = []
            token_count: int | None = None
            for branch_id in branch_ids:
                row = np.asarray(tokens_by_branch[branch_id], dtype=np.int64)
                if row.ndim != 2 or int(row.shape[0]) != self._cache.B:
                    raise ValueError(
                        "each branch block must have shape [parent_batch,tokens]"
                    )
                if int(row.shape[1]) <= 0:
                    raise ValueError("each branch block must contain at least one token")
                if token_count is None:
                    token_count = int(row.shape[1])
                elif int(row.shape[1]) != token_count:
                    raise ValueError("every branch block must use the same token count")
                rows.append(row)
            if token_count is None:
                raise RuntimeError("StateCut continuation lost its branch geometry")
            branch_count = len(branch_ids)
            cfg = self._engine.store.cfg
            diff_bytes = (
                2
                * int(cfg["num_hidden_layers"])
                * branch_count
                * self._cache.B
                * int(cfg["num_key_value_heads"])
                * int(cfg["head_dim"])
                * torch.float32.itemsize
                * token_count
            )
            contract = str(output_contract)
            output_width = {
                "hidden_state_only": int(cfg["hidden_size"]),
                "selected_token_rows": len(tuple(selected_rows)),
                "full_logits": int(cfg["vocab_size"]),
            }.get(contract)
            if output_width is None:
                raise ValueError(f"unsupported StateCut output contract {contract!r}")
            output_bytes = branch_count * self._cache.B * output_width * torch.float32.itemsize
            retained_total = _cache_allocated_bytes(self._cache) + diff_bytes + output_bytes
            if retained_total > self._budget:
                raise MemoryError("StateCut parent + branch diff + output exceeds retention budget")
            engine_lock = self._engine._execution_lock
            with engine_lock:
                output, panel = pf.paged_forward_statecut_branch_blocks(
                    self._engine.store,
                    np.stack(rows),
                    self._cache,
                    output_contract=output_contract,
                    selected_rows=selected_rows,
                    arithmetic=arithmetic,
                    last_only=True,
                )
            self._assert_parent_unchanged()
            if panel.retained_bytes != diff_bytes:
                raise RuntimeError("StateCut branch diff retained an unexpected byte count")
            batched_outputs = output.detach().cpu()
            outputs = tuple(batched_outputs.unbind(0))
            continuation = PagedKVStateCutContinuation(
                cut_id=self._cut_id,
                branch_ids=branch_ids,
                parent_epoch=self._parent_guard.epoch,
                parent_lengths=self._parent_guard.lengths,
                output_contract=contract,
                batched_outputs=batched_outputs,
                outputs=outputs,
                physical_forwards=1,
                arithmetic=arithmetic,
                retained_diff_bytes=panel.retained_bytes,
                per_branch_diff_bytes=panel.per_branch_bytes,
                parent_copy_bytes=0,
                retained_total_bytes=retained_total,
            )
            self._panel = panel
            self._continuation = continuation
            return continuation

    def _hidden_continuation(self) -> PagedKVStateCutContinuation:
        continuation = self._continuation
        if continuation is None:
            raise RuntimeError("StateCut scoring requires a completed branch continuation")
        if continuation.output_contract != "hidden_state_only":
            raise RuntimeError(
                "StateCut screening/adjudication requires hidden_state_only continuation so "
                "the same suffix state can feed both heads without replay"
            )
        expected = (
            len(continuation.branch_ids),
            self._cache.B,
            int(self._engine.store.cfg["hidden_size"]),
        )
        if tuple(continuation.batched_outputs.shape) != expected:
            raise RuntimeError(
                f"StateCut hidden continuation has shape "
                f"{tuple(continuation.batched_outputs.shape)}, expected {expected}"
            )
        if continuation.batched_outputs.dtype != torch.float32:
            raise RuntimeError("StateCut hidden continuation must retain fp32 states")
        return continuation

    def _assert_frozen_lm_head(self) -> _FrozenLMHeadGuard:
        if self._frozen_lm_head is None:
            self._frozen_lm_head = _FrozenLMHeadGuard.capture(self._engine.store)
        else:
            self._frozen_lm_head.assert_unchanged(self._engine.store)
        return self._frozen_lm_head

    def _retained_runtime_bytes(self) -> int:
        continuation = self._hidden_continuation()
        retained = int(continuation.retained_total_bytes)
        if self._screening is not None:
            retained += _tensor_bytes(self._screening.scores)
        if self._adjudication is not None:
            retained += _tensor_bytes(self._adjudication.logits)
        return retained

    def _projection_memory(
        self,
        *,
        output_rows: int,
        device_scratch_budget_bytes: int,
        projection_chunk_rows: int | None = None,
    ) -> PagedKVStateCutProjectionMemory:
        continuation = self._hidden_continuation()
        budget = _positive_byte_budget(
            device_scratch_budget_bytes,
            field="device_scratch_budget_bytes",
        )
        logical_rows = continuation.logical_suffix_rows
        hidden_size = int(self._engine.store.cfg["hidden_size"])
        retained_before = self._retained_runtime_bytes()
        retained_output = logical_rows * int(output_rows) * torch.float32.itemsize
        retained_after = retained_before + retained_output
        if retained_after > self._budget:
            raise MemoryError(
                "StateCut projection output would exceed the retention budget before allocation"
            )

        # The projection lane keeps one fp32 hidden batch resident and streams the head.  The
        # per-row bound conservatively charges a source-width fp32 weight, an additional fp32
        # cast/dequant result, int8 codes + one scale, and the branch-batched score column.
        device_input = logical_rows * hidden_size * torch.float32.itemsize
        weight_row_peak = (
            2 * hidden_size * torch.float32.itemsize + hidden_size + torch.float32.itemsize
        )
        output_row_peak = logical_rows * torch.float32.itemsize
        per_head_row_peak = weight_row_peak + output_row_peak
        available_for_rows = budget - device_input
        if available_for_rows < per_head_row_peak:
            raise MemoryError(
                "StateCut projection cannot fit one head row inside the device scratch budget"
            )
        chunk_rows = min(
            int(output_rows if projection_chunk_rows is None else projection_chunk_rows),
            8192,
            available_for_rows // per_head_row_peak,
        )
        if chunk_rows <= 0:  # defensive; the bound above should make this unreachable
            raise MemoryError("StateCut projection produced an empty head-row chunk")
        device_weight_peak = chunk_rows * weight_row_peak
        device_output_peak = chunk_rows * output_row_peak
        planned_device_peak = device_input + device_weight_peak + device_output_peak
        device = torch.device(getattr(self._engine.store, "device", "cpu"))
        free_cuda = _cuda_free_bytes(device)
        if free_cuda is not None and planned_device_peak > free_cuda:
            raise MemoryError(
                "StateCut projection scratch exceeds currently free CUDA memory before allocation"
            )
        return PagedKVStateCutProjectionMemory(
            retained_before_bytes=retained_before,
            retained_output_bytes=retained_output,
            retained_after_bytes=retained_after,
            retention_budget_bytes=self._budget,
            device_input_bytes=device_input,
            device_weight_peak_bytes=device_weight_peak,
            device_output_peak_bytes=device_output_peak,
            planned_device_scratch_peak_bytes=planned_device_peak,
            device_scratch_budget_bytes=budget,
            cuda_free_before_bytes=free_cuda,
            head_row_chunk_size=chunk_rows,
        )

    def _head_weights_fp32(
        self,
        weights: torch.Tensor,
        *,
        rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        hidden_size = int(self._engine.store.cfg["hidden_size"])
        expected = (int(rows), hidden_size)
        if not isinstance(weights, torch.Tensor) or tuple(weights.shape) != expected:
            actual = None if not isinstance(weights, torch.Tensor) else tuple(weights.shape)
            raise RuntimeError(f"LM-head page has shape {actual}, expected {expected}")
        compute_dtype = getattr(self._engine.store, "compute_dtype", torch.float32)
        if not isinstance(compute_dtype, torch.dtype):
            raise TypeError("QStore compute_dtype must be a torch dtype")
        return weights.to(device=device, dtype=compute_dtype).to(dtype=torch.float32)

    @staticmethod
    def _row_stable_head_projection(
        hidden_rows: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            tuple(
                hidden_rows[row : row + 1].float() @ weights.T
                for row in range(int(hidden_rows.shape[0]))
            ),
            dim=0,
        )

    def _projection_accounting(
        self,
        *,
        score_width: int,
        selected_gathers: int,
        full_traversals: int,
        head_blocks: int,
    ) -> PagedKVStateCutProjectionAccounting:
        continuation = self._hidden_continuation()
        return PagedKVStateCutProjectionAccounting(
            logical_branches=continuation.logical_branches,
            logical_parent_rows_per_branch=len(continuation.parent_lengths),
            logical_suffix_rows=continuation.logical_suffix_rows,
            logical_score_values=continuation.logical_suffix_rows * int(score_width),
            physical_suffix_forwards_total=continuation.physical_forwards,
            physical_suffix_forwards_this_call=0,
            suffix_replays=0,
            physical_selected_head_gathers_this_call=int(selected_gathers),
            physical_full_head_traversals_this_call=int(full_traversals),
            physical_head_blocks_this_call=int(head_blocks),
        )

    @torch.no_grad()
    def screen_selected_rows(
        self,
        selected_rows: Sequence[int],
        *,
        device_scratch_budget_bytes: int,
    ) -> PagedKVStateCutScreening:
        """Screen branches through frozen selected LM-head rows without replaying the suffix.

        The returned winner is scoped strictly to ``selected_rows``. It cannot establish the
        global vocabulary argmax; call :meth:`adjudicate_full_vocabulary` for that claim.
        """

        with self._lock:
            self._require_open()
            self._assert_parent_unchanged()
            continuation = self._hidden_continuation()
            if self._screening is not None:
                raise RuntimeError("StateCut has already performed selected-row screening")
            if self._adjudication is not None:
                raise RuntimeError("StateCut selected-row screening must precede adjudication")
            rows = _normalize_selected_rows(
                selected_rows,
                vocab_size=int(self._engine.store.cfg["vocab_size"]),
            )
            vocab_size = int(self._engine.store.cfg["vocab_size"])
            # Use the same aligned row-block width as full-vocabulary adjudication. CUDA GEMM
            # reduction order can otherwise differ between an N-row selected projection and the
            # same rows embedded in an 8K-row streamed projection. The resulting few-ulp drift is
            # harmless for ranking but invalidates a bit-exact same-head contract. We therefore
            # gather only the aligned blocks containing requested rows, project each complete
            # block, and retain only the requested columns.
            memory = self._projection_memory(
                output_rows=len(rows),
                device_scratch_budget_bytes=device_scratch_budget_bytes,
                projection_chunk_rows=vocab_size,
            )
            head_guard = self._assert_frozen_lm_head()
            store = self._engine.store
            embed_rows = getattr(store, "embed_rows", None)
            if not callable(embed_rows):
                raise NotImplementedError("StateCut selected-row screening requires embed_rows")
            device = torch.device(getattr(store, "device", "cpu"))
            logical_shape = (
                continuation.logical_branches,
                len(continuation.parent_lengths),
                len(rows),
            )
            scores = torch.empty(logical_shape, dtype=torch.float32, device="cpu")
            hidden = continuation.batched_outputs.reshape(
                continuation.logical_suffix_rows,
                int(store.cfg["hidden_size"]),
            ).to(device=device, dtype=torch.float32)
            gather_count = 0
            engine_lock = self._engine._execution_lock
            with engine_lock:
                head_guard.assert_unchanged(store)
                selected_by_block: dict[int, list[tuple[int, int]]] = {}
                for output_index, token_id in enumerate(rows):
                    block_start = (token_id // memory.head_row_chunk_size) * (
                        memory.head_row_chunk_size
                    )
                    selected_by_block.setdefault(block_start, []).append(
                        (output_index, token_id)
                    )
                for start, selected in sorted(selected_by_block.items()):
                    end = min(start + memory.head_row_chunk_size, vocab_size)
                    chunk_ids = np.arange(start, end, dtype=np.int64)
                    weights = self._head_weights_fp32(
                        embed_rows("lm_head", chunk_ids),
                        rows=end - start,
                        device=device,
                    )
                    chunk_scores = self._row_stable_head_projection(hidden, weights)
                    shaped = chunk_scores.reshape(*logical_shape[:-1], end - start)
                    source_columns = torch.as_tensor(
                        [token_id - start for _, token_id in selected],
                        dtype=torch.long,
                        device=shaped.device,
                    )
                    output_columns = torch.as_tensor(
                        [output_index for output_index, _ in selected],
                        dtype=torch.long,
                    )
                    scores.index_copy_(
                        -1,
                        output_columns,
                        shaped.index_select(-1, source_columns).cpu(),
                    )
                    gather_count += 1
                    del weights, chunk_scores, shaped, source_columns, output_columns
                head_guard.assert_unchanged(store)
            selected_ids = torch.as_tensor(rows, dtype=torch.long)
            winner_ids = selected_ids[scores.argmax(dim=-1)]
            accounting = self._projection_accounting(
                score_width=len(rows),
                selected_gathers=gather_count,
                full_traversals=0,
                head_blocks=0,
            )
            result = PagedKVStateCutScreening(
                cut_id=self._cut_id,
                branch_ids=continuation.branch_ids,
                branch_axis=tuple(self._branches.values()),
                selected_rows=rows,
                scores=scores,
                selected_winner_token_ids=winner_ids,
                global_argmax_established=False,
                scope_statement=(
                    "Selected LM-head rows rank only the supplied candidate set and cannot "
                    "establish the global vocabulary argmax."
                ),
                frozen_lm_head_sha256=head_guard.sha256,
                accounting=accounting,
                memory=memory,
            )
            self._screening = result
            return result

    @torch.no_grad()
    def adjudicate_full_vocabulary(
        self,
        *,
        device_scratch_budget_bytes: int,
    ) -> PagedKVStateCutAdjudication:
        """Project the same retained branch states through the complete frozen LM head once."""

        with self._lock:
            self._require_open()
            self._assert_parent_unchanged()
            continuation = self._hidden_continuation()
            if self._adjudication is not None:
                raise RuntimeError("StateCut has already performed full-vocabulary adjudication")
            vocab_size = int(self._engine.store.cfg["vocab_size"])
            memory = self._projection_memory(
                output_rows=vocab_size,
                device_scratch_budget_bytes=device_scratch_budget_bytes,
            )
            head_guard = self._assert_frozen_lm_head()
            store = self._engine.store
            row_blocks = getattr(store, "row_blocks", None)
            if not callable(row_blocks):
                raise NotImplementedError(
                    "StateCut full-vocabulary adjudication requires streamed row_blocks"
                )
            device = torch.device(getattr(store, "device", "cpu"))
            logical_shape = (
                continuation.logical_branches,
                len(continuation.parent_lengths),
                vocab_size,
            )
            logits = torch.empty(logical_shape, dtype=torch.float32, device="cpu")
            hidden = continuation.batched_outputs.reshape(
                continuation.logical_suffix_rows,
                int(store.cfg["hidden_size"]),
            ).to(device=device, dtype=torch.float32)
            expected_start = 0
            block_count = 0
            engine_lock = self._engine._execution_lock
            with engine_lock:
                head_guard.assert_unchanged(store)
                for raw_start, raw_end, raw_weights in row_blocks(
                    "lm_head",
                    bs=memory.head_row_chunk_size,
                ):
                    start, end = int(raw_start), int(raw_end)
                    if start != expected_start or end <= start or end > vocab_size:
                        raise RuntimeError("LM-head row stream is not a complete ordered partition")
                    weights = self._head_weights_fp32(
                        raw_weights,
                        rows=end - start,
                        device=device,
                    )
                    chunk_scores = self._row_stable_head_projection(hidden, weights)
                    logits[..., start:end].copy_(
                        chunk_scores.reshape(*logical_shape[:-1], end - start).cpu()
                    )
                    expected_start = end
                    block_count += 1
                    del weights, chunk_scores
                if expected_start != vocab_size:
                    raise RuntimeError("LM-head row stream ended before the full vocabulary")
                head_guard.assert_unchanged(store)

            screened_rows: tuple[int, ...] = ()
            parity_verified = True
            if self._screening is not None:
                screened_rows = self._screening.selected_rows
                projected = logits.index_select(
                    -1,
                    torch.as_tensor(screened_rows, dtype=torch.long),
                )
                parity_verified = bool(torch.equal(projected, self._screening.scores))
                if not parity_verified:
                    max_error = float((projected - self._screening.scores).abs().max())
                    raise RuntimeError(
                        "selected-row screening disagrees with the same frozen full head "
                        f"(max_abs_error={max_error})"
                    )
            accounting = self._projection_accounting(
                score_width=vocab_size,
                selected_gathers=0,
                full_traversals=1,
                head_blocks=block_count,
            )
            result = PagedKVStateCutAdjudication(
                cut_id=self._cut_id,
                branch_ids=continuation.branch_ids,
                branch_axis=tuple(self._branches.values()),
                logits=logits,
                global_argmax_token_ids=logits.argmax(dim=-1),
                global_argmax_established=True,
                screened_rows=screened_rows,
                selected_screening_parity_verified=parity_verified,
                frozen_lm_head_sha256=head_guard.sha256,
                accounting=accounting,
                memory=memory,
            )
            self._adjudication = result
            return result

    def _release_ownership(self) -> None:
        with self._cache._lock:  # noqa: SLF001 - releases bound StateCut ownership metadata
            active = getattr(self._cache, "_active_statecut_id", None)
            if active != self._cut_id:
                raise RuntimeError("StateCut cache ownership changed unexpectedly")
            self._cache._active_statecut_id = None  # type: ignore[attr-defined]  # noqa: SLF001

    def commit(self, branch_id: str) -> PagedKVStateCutReceipt:
        with self._lock:
            self._require_open()
            before = self._assert_parent_unchanged()
            if self._panel is None or self._continuation is None:
                raise RuntimeError("StateCut commit requires a completed continuation")
            normalized = _require_branch_id(branch_id)
            try:
                branch_index = self._continuation.branch_ids.index(normalized)
            except ValueError as exc:
                raise KeyError(f"unknown StateCut branch {normalized!r}") from exc
            delta = self._panel.select(branch_index)
            retained_diff_bytes = self._panel.retained_bytes
            committed_token_count = int(self._panel.token_count)
            accepted_lengths = (committed_token_count,) * self._cache.B
            engine_lock = self._engine._execution_lock
            with engine_lock:
                accepted = pf.commit_block(
                    self._cache,
                    delta,
                    accepted_lengths,
                )
            expected_lengths = tuple(
                value + committed_token_count for value in before.lengths
            )
            after = _ParentGuard.capture(self._cache)
            transition_verified = bool(
                accepted == accepted_lengths
                and after.epoch == before.epoch + 1
                and after.lengths == expected_lengths
                and after.cache_id == before.cache_id
            )
            if not transition_verified:
                raise RuntimeError("StateCut commit did not produce its atomic transition")
            abandoned = tuple(
                value for value in self._continuation.branch_ids if value != normalized
            )
            self._release_ownership()
            self._panel = None
            self._continuation = None
            self._screening = None
            self._adjudication = None
            self._terminal = True
            return PagedKVStateCutReceipt(
                action="commit",
                cut_id=self._cut_id,
                committed_branch_id=normalized,
                abandoned_branch_ids=abandoned,
                parent_epoch_before=before.epoch,
                parent_epoch_after=after.epoch,
                parent_lengths_before=before.lengths,
                parent_lengths_after=after.lengths,
                transition_verified=True,
                exact_parent_restored=False,
                parent_storage_unchanged=False,
                parent_guard_before_sha256=before.sha256,
                parent_guard_after_sha256=after.sha256,
                retained_diff_bytes=retained_diff_bytes,
                parent_allocated_bytes=_cache_allocated_bytes(self._cache),
                physical_forwards=1,
            )

    def abandon(self) -> PagedKVStateCutReceipt:
        with self._lock:
            self._require_open()
            after = self._assert_parent_unchanged()
            retained_diff_bytes = 0 if self._panel is None else self._panel.retained_bytes
            physical_forwards = 1 if self._panel is not None else 0
            branch_ids = tuple(self._branches)
            self._release_ownership()
            self._panel = None
            self._continuation = None
            self._screening = None
            self._adjudication = None
            self._terminal = True
            return PagedKVStateCutReceipt(
                action="abandon",
                cut_id=self._cut_id,
                committed_branch_id=None,
                abandoned_branch_ids=branch_ids,
                parent_epoch_before=self._parent_guard.epoch,
                parent_epoch_after=after.epoch,
                parent_lengths_before=self._parent_guard.lengths,
                parent_lengths_after=after.lengths,
                transition_verified=True,
                exact_parent_restored=True,
                parent_storage_unchanged=True,
                parent_guard_before_sha256=self._parent_guard.sha256,
                parent_guard_after_sha256=after.sha256,
                retained_diff_bytes=retained_diff_bytes,
                parent_allocated_bytes=_cache_allocated_bytes(self._cache),
                physical_forwards=physical_forwards,
            )

    def __enter__(self) -> PagedKVStateCut:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if not self._terminal:
            self.abandon()


def prefill_paged_kv_statecut(
    engine: Any,
    ids_list: Sequence[np.ndarray | Sequence[int]],
    *,
    retention_budget_bytes: int,
    capacity: int | None = None,
) -> tuple[PagedKVStateCut, torch.Tensor]:
    """Public functional entry point matching :meth:`PagedKVStateCut.prefill`."""

    return PagedKVStateCut.prefill(
        engine,
        ids_list,
        retention_budget_bytes=retention_budget_bytes,
        capacity=capacity,
    )


__all__ = [
    "STATECUT_ADJUDICATION_SCHEMA",
    "STATECUT_CONTINUATION_SCHEMA",
    "STATECUT_RECEIPT_SCHEMA",
    "STATECUT_SCHEMA",
    "STATECUT_SCREENING_SCHEMA",
    "PagedKVStateCut",
    "PagedKVStateCutAdjudication",
    "PagedKVStateCutBranch",
    "PagedKVStateCutContinuation",
    "PagedKVStateCutDescriptor",
    "PagedKVStateCutProjectionAccounting",
    "PagedKVStateCutProjectionMemory",
    "PagedKVStateCutReceipt",
    "PagedKVStateCutScreening",
    "prefill_paged_kv_statecut",
]
