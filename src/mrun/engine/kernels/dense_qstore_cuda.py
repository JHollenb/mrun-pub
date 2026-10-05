"""Compact CUDA consumers and row-stable reductions for dense int8 QStores.

The compact int8 row codes and fp32 row scales remain the durable/hot representation.
CUDA BF16/FP16 projections dequantize inside a fixed-tile Triton matmul, avoiding a
complete expanded weight. The fixed-row RMSNorm and causal GQA kernels assign one
program to one logical row so changing request/token packing changes grid cardinality,
not reduction shape. They remain off by default because they define a separate
``row-stable-triton-v1`` numerical contract rather than reproducing the established
Torch-batched trajectory. That contract passed its own exact B1-to-packed model gate.

Torch fallbacks exist for unit and parity tests. Callers may set ``require_triton`` to
fail closed when a CUDA fast-path claim is required.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .qstore import QStore

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU and Apple installations do not require Triton.
    triton = None
    tl = None


__all__ = [
    "CompactQRowPage",
    "DenseQStore",
    "DenseQStoreCaptureBindings",
    "DenseQStoreResidentArena",
    "DenseQStoreStats",
    "dequantized_reference_matmul",
    "fused_qrow_swiglu",
    "fused_qrow_swiglu_reference",
    "fused_qrow_matmul",
    "fused_qrow_reranked_argmax",
    "fused_qrow_top2",
    "fused_residual_rms_norm",
    "fused_residual_rms_norm_reference",
    "qrow_top2_reference",
    "reranked_argmax_workspace_bytes",
    "rerank_qrow_candidates_fp32",
    "segmented_decode_attention",
    "segmented_decode_attention_reference",
    "stable_attention",
    "stable_attention_reference",
    "stable_rms_norm",
    "stable_rms_norm_reference",
]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _semantic_row_count(page: CompactQRowPage, value: int | None) -> int:
    if value is None:
        return page.out_features
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("semantic_row_count must be an integer or None")
    row_count = int(value)
    if row_count < 2 or row_count > page.out_features:
        raise ValueError("semantic_row_count must select at least two rows inside the compact page")
    return row_count


@dataclass(frozen=True)
class CompactQRowPage:
    name: str
    start_row: int
    end_row: int
    in_features: int
    codes: torch.Tensor
    scales: torch.Tensor

    @property
    def out_features(self) -> int:
        return self.end_row - self.start_row

    @property
    def compact_bytes(self) -> int:
        return _tensor_bytes(self.codes) + _tensor_bytes(self.scales)

    @property
    def expanded_fp32_bytes(self) -> int:
        return self.out_features * self.in_features * 4

    def expanded_compute_bytes(self, dtype: torch.dtype) -> int:
        return self.out_features * self.in_features * torch.empty((), dtype=dtype).element_size()


def reranked_argmax_workspace_bytes(
    *,
    rows: int,
    in_features: int,
    semantic_row_count: int,
    block_n: int = 64,
    candidate_count: int = 2,
) -> int:
    """Conservatively charge all eager tensors in compact top-k plus FP32 reranking."""

    for value, name in (
        (rows, "rows"),
        (in_features, "in_features"),
        (semantic_row_count, "semantic_row_count"),
        (block_n, "block_n"),
        (candidate_count, "candidate_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if semantic_row_count < candidate_count:
        raise ValueError("semantic rows must cover every rerank candidate")
    vocabulary_blocks = (semantic_row_count + block_n - 1) // block_n
    # FP32/int32 top-k partials and final top-k values/indices.
    top2_bytes = rows * (vocabulary_blocks * candidate_count * (4 + 4) + candidate_count * (4 + 4))
    # FP32 activation, int8 selected codes, FP32 weights/products/scales/scores,
    # int64 candidate/local/tie indices, and final value/index reductions.
    rerank_bytes = rows * (
        in_features * 4
        + candidate_count * in_features * (1 + 4 + 4)
        + candidate_count * (8 + 8 + 4 + 4 + 8)
        + 4
        + 8
    )
    return int(top2_bytes + rerank_bytes)


@dataclass
class DenseQStoreStats:
    projection_calls: int = 0
    triton_projection_calls: int = 0
    reference_projection_calls: int = 0
    paired_swiglu_calls: int = 0
    triton_paired_swiglu_calls: int = 0
    reference_paired_swiglu_calls: int = 0
    selected_head_calls: int = 0
    selected_head_rows: int = 0
    selected_head_compact_bytes: int = 0
    page_loads: int = 0
    cache_hits: int = 0
    compact_h2d_bytes: int = 0
    compact_logical_bytes: int = 0
    expanded_fp32_bytes_avoided: int = 0
    expanded_compute_bytes_avoided: int = 0
    peak_compact_resident_bytes: int = 0
    fp32_aux_cache_loads: int = 0
    resident_exact_head_loads: int = 0

    def snapshot(self, *, resident_bytes: int, cache_entries: int) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "compact_resident_bytes": int(resident_bytes),
                "cache_entries": int(cache_entries),
            }
        )
        return payload


class DenseQStoreCaptureBindings:
    """Capture-owned, non-evictable QStore resources.

    CUDA Graph replay dereferences the same virtual addresses on every launch.  The
    ordinary compact-page LRU therefore cannot be the owner of graph inputs: eviction
    would make the captured kernel arguments stale even if a later load happened to
    contain identical bytes.  This object owns every compact page and full-precision
    auxiliary tensor used by one captured selected-head graph, plus the exact FP32
    selected-head matrix.  Aliases map to one physical allocation.
    """

    def __init__(
        self,
        store: DenseQStore,
        *,
        qrow_pages: dict[str, CompactQRowPage],
        qrow_aliases: dict[str, str],
        fp32_tensors: dict[str, torch.Tensor],
        fp32_aliases: dict[str, str],
        selected_head_rows: tuple[int, ...],
        selected_head_weights: torch.Tensor | None,
        rebindable_head: bool,
        budget_bytes: int | None,
        estimated_resident_bytes: int,
        arena_owner: DenseQStoreResidentArena | None = None,
    ) -> None:
        self._store = store
        self._qrow_pages = qrow_pages
        self._qrow_aliases = qrow_aliases
        self._fp32_tensors = fp32_tensors
        self._fp32_aliases = fp32_aliases
        self.selected_head_rows = selected_head_rows
        self._selected_head_weights: torch.Tensor | None = selected_head_weights
        self.rebindable_head = bool(rebindable_head)
        self.budget_bytes = budget_bytes
        self.estimated_resident_bytes = int(estimated_resident_bytes)
        self._closed = False
        self._arena_owner = arena_owner

        tensors = [
            *(page.codes for page in qrow_pages.values()),
            *(page.scales for page in qrow_pages.values()),
            *fp32_tensors.values(),
        ]
        if selected_head_weights is not None:
            tensors.append(selected_head_weights)
        self.resident_bytes = sum(_tensor_bytes(tensor) for tensor in tensors)
        self.stable_address_count = len(tensors)
        self._stable_addresses = tuple(int(tensor.data_ptr()) for tensor in tensors)
        self.stable_addresses_verified = False
        self.qrow_logical_count = len(qrow_aliases)
        self.qrow_physical_count = len(qrow_pages)
        self.fp32_logical_count = len(fp32_aliases)
        self.fp32_physical_count = len(fp32_tensors)
        self.aliases_deduplicated = (
            self.qrow_logical_count
            - self.qrow_physical_count
            + self.fp32_logical_count
            - self.fp32_physical_count
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CUDA Graph QStore bindings are closed")

    def verify_stable_addresses(self) -> None:
        """Prove every capture-owned tensor still has its recorded data pointer."""

        self._require_open()
        tensors = [
            *(page.codes for page in self._qrow_pages.values()),
            *(page.scales for page in self._qrow_pages.values()),
            *self._fp32_tensors.values(),
        ]
        if self._selected_head_weights is not None:
            tensors.append(self._selected_head_weights)
        current = tuple(int(tensor.data_ptr()) for tensor in tensors)
        if current != self._stable_addresses:
            self.stable_addresses_verified = False
            raise RuntimeError("CUDA Graph QStore resource address changed after capture")
        self.stable_addresses_verified = True

    def has(self, name: str) -> bool:
        self._require_open()
        return self._store.has(name)

    def fp32(self, name: str) -> torch.Tensor:
        self._require_open()
        try:
            return self._fp32_tensors[self._fp32_aliases[name]]
        except KeyError as exc:
            raise RuntimeError(
                f"FP32 resource {name!r} was not pinned for CUDA Graph capture"
            ) from exc

    def compact_page(self, name: str) -> CompactQRowPage:
        self._require_open()
        try:
            return self._qrow_pages[self._qrow_aliases[name]]
        except KeyError as exc:
            raise RuntimeError(
                f"qrow resource {name!r} was not pinned for CUDA Graph capture"
            ) from exc

    def matmul(self, name: str, activations: torch.Tensor) -> torch.Tensor:
        self._require_open()
        return self._store._matmul_page(self.compact_page(name), activations)

    def fused_swiglu(
        self,
        gate_name: str,
        up_name: str,
        activations: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        self._require_open()
        return self._store._fused_swiglu_pages(
            self.compact_page(gate_name),
            self.compact_page(up_name),
            activations,
        )

    def embed_rows(
        self,
        name: str,
        ids: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        self._require_open()
        return self._store._embed_page_rows(self.compact_page(name), ids)

    def selected_rows_fp32(
        self,
        name: str,
        ids: np.ndarray | torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        self._require_open()
        if name != "lm_head":
            raise RuntimeError("captured selected rows are available only for lm_head")
        if self.rebindable_head:
            index = torch.as_tensor(
                ids,
                device=self.compact_page(name).codes.device,
                dtype=torch.long,
            )
            if index.ndim != 1 or int(index.numel()) != len(self.selected_head_rows):
                raise RuntimeError("rebindable selected-head row buffer has the wrong shape")
            page = self.compact_page(name)
            codes = page.codes.index_select(0, index).float()
            scales = page.scales.index_select(0, index).float()
            return codes * scales[:, None]
        selected = tuple(int(value) for value in torch.as_tensor(ids).reshape(-1).tolist())
        if selected != self.selected_head_rows:
            raise RuntimeError("captured selected-head rows do not match the prepared CUDA Graph")
        if self._selected_head_weights is None:
            raise RuntimeError("captured selected-head weights are closed")
        return self._selected_head_weights

    def evidence(self) -> dict[str, Any]:
        return {
            "capture_resident_bytes": self.resident_bytes,
            "capture_estimated_resident_bytes": self.estimated_resident_bytes,
            "capture_residency_budget_bytes": self.budget_bytes,
            "capture_qrow_logical_count": self.qrow_logical_count,
            "capture_qrow_physical_count": self.qrow_physical_count,
            "capture_fp32_logical_count": self.fp32_logical_count,
            "capture_fp32_physical_count": self.fp32_physical_count,
            "capture_aliases_deduplicated": self.aliases_deduplicated,
            "capture_selected_head_rows": len(self.selected_head_rows),
            "capture_rebindable_head_rows": self.rebindable_head,
            "capture_stable_address_count": self.stable_address_count,
            "capture_resource_addresses_verified": self.stable_addresses_verified,
            "capture_residency_non_evictable": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._qrow_pages.clear()
        self._qrow_aliases.clear()
        self._fp32_tensors.clear()
        self._fp32_aliases.clear()
        self._selected_head_weights = None
        self.stable_addresses_verified = False
        self._closed = True
        arena_owner = self._arena_owner
        self._arena_owner = None
        if arena_owner is not None:
            arena_owner._release_binding()  # noqa: SLF001 - explicit ownership handshake


class DenseQStoreResidentArena:
    """One immutable, address-stable QStore resource owner shared by graph templates."""

    def __init__(
        self,
        store: DenseQStore,
        *,
        qrow_pages: dict[str, CompactQRowPage],
        qrow_aliases: dict[str, str],
        fp32_tensors: dict[str, torch.Tensor],
        fp32_aliases: dict[str, str],
        budget_bytes: int | None,
        estimated_resident_bytes: int,
    ) -> None:
        self._store = store
        self._qrow_pages = qrow_pages
        self._qrow_aliases = qrow_aliases
        self._fp32_tensors = fp32_tensors
        self._fp32_aliases = fp32_aliases
        self.budget_bytes = budget_bytes
        self.estimated_resident_bytes = int(estimated_resident_bytes)
        tensors = [
            *(page.codes for page in qrow_pages.values()),
            *(page.scales for page in qrow_pages.values()),
            *fp32_tensors.values(),
        ]
        self.resident_bytes = sum(_tensor_bytes(tensor) for tensor in tensors)
        self._stable_addresses = tuple(int(tensor.data_ptr()) for tensor in tensors)
        self._lock = threading.RLock()
        self._active_bindings = 0
        self._closed = False

    @property
    def active_bindings(self) -> int:
        with self._lock:
            return self._active_bindings

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def verify_stable_addresses(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("resident QStore arena is closed")
            tensors = [
                *(page.codes for page in self._qrow_pages.values()),
                *(page.scales for page in self._qrow_pages.values()),
                *self._fp32_tensors.values(),
            ]
            if tuple(int(tensor.data_ptr()) for tensor in tensors) != self._stable_addresses:
                raise RuntimeError("resident QStore arena resource address changed")

    def bind_selected_rows(
        self,
        selected_head_rows: Sequence[int],
    ) -> DenseQStoreCaptureBindings:
        selected = tuple(int(value) for value in selected_head_rows)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("resident arena selected rows must be non-empty and unique")
        head = self._store._resolve("lm_head")
        row_count = int(head["shape"][0])
        if min(selected) < 0 or max(selected) >= row_count:
            raise ValueError(f"selected head row must be inside [0, {row_count})")
        with self._lock:
            self.verify_stable_addresses()
            self._active_bindings += 1
        return DenseQStoreCaptureBindings(
            self._store,
            qrow_pages=dict(self._qrow_pages),
            qrow_aliases=dict(self._qrow_aliases),
            fp32_tensors=dict(self._fp32_tensors),
            fp32_aliases=dict(self._fp32_aliases),
            selected_head_rows=selected,
            selected_head_weights=None,
            rebindable_head=True,
            budget_bytes=self.budget_bytes,
            estimated_resident_bytes=self.estimated_resident_bytes,
            arena_owner=self,
        )

    def _release_binding(self) -> None:
        with self._lock:
            if self._active_bindings <= 0:
                raise RuntimeError("resident QStore arena binding count underflow")
            self._active_bindings -= 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._active_bindings:
                raise RuntimeError("cannot close resident QStore arena with live graph bindings")
            self._qrow_pages.clear()
            self._qrow_aliases.clear()
            self._fp32_tensors.clear()
            self._fp32_aliases.clear()
            self._stable_addresses = ()
            self._closed = True


if triton is not None:

    @triton.jit
    def _qrow_w8a16_kernel(
        activation_ptr,
        weight_ptr,
        scale_ptr,
        output_ptr,
        rows,
        out_features: tl.constexpr,
        in_features: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        use_bf16: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * block_m + tl.arange(0, block_m)
        offsets_n = pid_n * block_n + tl.arange(0, block_n)
        row_scales = tl.load(
            scale_ptr + offsets_n,
            mask=offsets_n < out_features,
            other=0.0,
        )
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        for start_k in range(0, in_features, block_k):
            offsets_k = start_k + tl.arange(0, block_k)
            activations = tl.load(
                activation_ptr + offsets_m[:, None] * in_features + offsets_k[None, :],
                mask=(offsets_m[:, None] < rows) & (offsets_k[None, :] < in_features),
                other=0.0,
            )
            weight_codes = tl.load(
                weight_ptr + offsets_n[:, None] * in_features + offsets_k[None, :],
                mask=(offsets_n[:, None] < out_features) & (offsets_k[None, :] < in_features),
                other=0,
            )
            weights_fp32 = weight_codes.to(tl.float32) * row_scales[:, None]
            if use_bf16:
                weights = weights_fp32.to(tl.bfloat16)
            else:
                weights = weights_fp32.to(tl.float16)
            accumulator += tl.dot(activations, tl.trans(weights))
        tl.store(
            output_ptr + offsets_m[:, None] * out_features + offsets_n[None, :],
            accumulator,
            mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < out_features),
        )

    @triton.jit
    def _qrow_w8a16_paired_swiglu_kernel(
        activation_ptr,
        gate_weight_ptr,
        gate_scale_ptr,
        up_weight_ptr,
        up_scale_ptr,
        output_ptr,
        rows,
        out_features: tl.constexpr,
        in_features: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        """Paired W8A16 projections with explicit BF16 SwiGLU boundaries."""

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * block_m + tl.arange(0, block_m)
        offsets_n = pid_n * block_n + tl.arange(0, block_n)
        valid_m = offsets_m < rows
        valid_n = offsets_n < out_features
        gate_scales = tl.load(gate_scale_ptr + offsets_n, mask=valid_n, other=0.0)
        up_scales = tl.load(up_scale_ptr + offsets_n, mask=valid_n, other=0.0)
        gate_accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        up_accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        for start_k in range(0, in_features, block_k):
            offsets_k = start_k + tl.arange(0, block_k)
            valid_k = offsets_k < in_features
            activations = tl.load(
                activation_ptr + offsets_m[:, None] * in_features + offsets_k[None, :],
                mask=valid_m[:, None] & valid_k[None, :],
                other=0.0,
            )
            gate_codes = tl.load(
                gate_weight_ptr + offsets_n[:, None] * in_features + offsets_k[None, :],
                mask=valid_n[:, None] & valid_k[None, :],
                other=0,
            )
            up_codes = tl.load(
                up_weight_ptr + offsets_n[:, None] * in_features + offsets_k[None, :],
                mask=valid_n[:, None] & valid_k[None, :],
                other=0,
            )
            gate_weights = (gate_codes.to(tl.float32) * gate_scales[:, None]).to(tl.bfloat16)
            up_weights = (up_codes.to(tl.float32) * up_scales[:, None]).to(tl.bfloat16)
            gate_accumulator += tl.dot(activations, tl.trans(gate_weights))
            up_accumulator += tl.dot(activations, tl.trans(up_weights))

        # Preserve the established W8A16 projection and elementwise BF16 boundaries:
        # projection -> BF16, SiLU -> BF16, product -> BF16.
        gate_bf16 = gate_accumulator.to(tl.bfloat16)
        up_bf16 = up_accumulator.to(tl.bfloat16)
        gate_fp32 = gate_bf16.to(tl.float32)
        silu_bf16 = (gate_fp32 * tl.sigmoid(gate_fp32)).to(tl.bfloat16)
        product_bf16 = (silu_bf16.to(tl.float32) * up_bf16.to(tl.float32)).to(tl.bfloat16)
        tl.store(
            output_ptr + offsets_m[:, None] * out_features + offsets_n[None, :],
            product_bf16,
            mask=valid_m[:, None] & valid_n[None, :],
        )

    @triton.jit
    def _qrow_w8a16_block_top2_kernel(
        activation_ptr,
        weight_ptr,
        scale_ptr,
        partial_value_ptr,
        partial_index_ptr,
        rows,
        out_features: tl.constexpr,
        in_features: tl.constexpr,
        vocabulary_blocks: tl.constexpr,
        start_row: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        use_bf16: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * block_m + tl.arange(0, block_m)
        offsets_n = pid_n * block_n + tl.arange(0, block_n)
        valid_n = offsets_n < out_features
        row_scales = tl.load(scale_ptr + offsets_n, mask=valid_n, other=0.0)
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        for start_k in range(0, in_features, block_k):
            offsets_k = start_k + tl.arange(0, block_k)
            activations = tl.load(
                activation_ptr + offsets_m[:, None] * in_features + offsets_k[None, :],
                mask=(offsets_m[:, None] < rows) & (offsets_k[None, :] < in_features),
                other=0.0,
            )
            weight_codes = tl.load(
                weight_ptr + offsets_n[:, None] * in_features + offsets_k[None, :],
                mask=valid_n[:, None] & (offsets_k[None, :] < in_features),
                other=0,
            )
            weights_fp32 = weight_codes.to(tl.float32) * row_scales[:, None]
            if use_bf16:
                weights = weights_fp32.to(tl.bfloat16)
            else:
                weights = weights_fp32.to(tl.float16)
            accumulator += tl.dot(activations, tl.trans(weights))

        token_indices = start_row + pid_n * block_n + tl.arange(0, block_n)
        masked = tl.where(valid_n[None, :], accumulator, float("-inf"))
        first_values = tl.max(masked, axis=1)
        first_ties = tl.where(
            masked == first_values[:, None],
            token_indices[None, :],
            0x7FFFFFFF,
        )
        first_indices = tl.min(first_ties, axis=1)
        second_masked = tl.where(
            valid_n[None, :] & (token_indices[None, :] != first_indices[:, None]),
            accumulator,
            float("-inf"),
        )
        second_values = tl.max(second_masked, axis=1)
        second_ties = tl.where(
            second_masked == second_values[:, None],
            token_indices[None, :],
            0x7FFFFFFF,
        )
        second_indices = tl.min(second_ties, axis=1)
        partial_base = (offsets_m * vocabulary_blocks + pid_n) * 2
        valid_m = offsets_m < rows
        tl.store(partial_value_ptr + partial_base, first_values, mask=valid_m)
        tl.store(partial_index_ptr + partial_base, first_indices, mask=valid_m)
        tl.store(partial_value_ptr + partial_base + 1, second_values, mask=valid_m)
        tl.store(partial_index_ptr + partial_base + 1, second_indices, mask=valid_m)

    @triton.jit
    def _reduce_top2_blocks_kernel(
        partial_value_ptr,
        partial_index_ptr,
        output_value_ptr,
        output_index_ptr,
        candidate_count: tl.constexpr,
        reduction_tile: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, reduction_tile)
        valid = offsets < candidate_count
        partial_offsets = row * candidate_count + offsets
        values = tl.load(partial_value_ptr + partial_offsets, mask=valid, other=float("-inf"))
        indices = tl.load(partial_index_ptr + partial_offsets, mask=valid, other=0x7FFFFFFF)
        first_value = tl.max(values, axis=0)
        first_ties = tl.where(values == first_value, indices, 0x7FFFFFFF)
        first_index = tl.min(first_ties, axis=0)
        second_values = tl.where(indices != first_index, values, float("-inf"))
        second_value = tl.max(second_values, axis=0)
        second_ties = tl.where(second_values == second_value, indices, 0x7FFFFFFF)
        second_index = tl.min(second_ties, axis=0)
        output_base = row * 2
        tl.store(output_value_ptr + output_base, first_value)
        tl.store(output_index_ptr + output_base, first_index)
        tl.store(output_value_ptr + output_base + 1, second_value)
        tl.store(output_index_ptr + output_base + 1, second_index)

    @triton.jit
    def _stable_rms_norm_kernel(
        activation_ptr,
        weight_ptr,
        output_ptr,
        rows,
        width: tl.constexpr,
        eps: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_n)
        valid = offsets < width
        values = tl.load(
            activation_ptr + row * width + offsets,
            mask=(row < rows) & valid,
            other=0.0,
        ).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) / width
        inverse_rms = tl.rsqrt(mean_square + eps)
        weights = tl.load(weight_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        tl.store(
            output_ptr + row * width + offsets,
            values * inverse_rms * weights,
            mask=(row < rows) & valid,
        )

    @triton.jit
    def _residual_bf16_rms_norm_kernel(
        residual_ptr,
        update_ptr,
        weight_ptr,
        residual_output_ptr,
        normalized_output_ptr,
        rows,
        width: tl.constexpr,
        eps: tl.constexpr,
        block_n: tl.constexpr,
    ):
        """Add in FP32, round the residual to BF16, then normalize that exact row."""

        row = tl.program_id(0)
        offsets = tl.arange(0, block_n)
        valid = offsets < width
        row_offsets = row * width + offsets
        residual = tl.load(
            residual_ptr + row_offsets,
            mask=(row < rows) & valid,
            other=0.0,
        ).to(tl.float32)
        update = tl.load(
            update_ptr + row_offsets,
            mask=(row < rows) & valid,
            other=0.0,
        ).to(tl.float32)
        rounded = (residual + update).to(tl.bfloat16)
        rounded_fp32 = rounded.to(tl.float32)
        mean_square = tl.sum(rounded_fp32 * rounded_fp32, axis=0) / width
        inverse_rms = tl.rsqrt(mean_square + eps)
        weights = tl.load(weight_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        tl.store(
            residual_output_ptr + row_offsets,
            rounded,
            mask=(row < rows) & valid,
        )
        tl.store(
            normalized_output_ptr + row_offsets,
            rounded_fp32 * inverse_rms * weights,
            mask=(row < rows) & valid,
        )

    @triton.jit
    def _stable_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        lengths_ptr,
        output_ptr,
        stride_qb: tl.constexpr,
        stride_qt: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_ot: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        token_count: tl.constexpr,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        attention_scale: tl.constexpr,
        block_s: tl.constexpr,
        block_d: tl.constexpr,
    ):
        program = tl.program_id(0)
        head = program % num_heads
        row = program // num_heads
        token = row % token_count
        batch = row // token_count
        kv_head = head // (num_heads // num_kv_heads)
        committed = tl.load(lengths_ptr + batch)
        total = committed + token + 1

        offsets_s = tl.arange(0, block_s)
        offsets_d = tl.arange(0, block_d)
        valid_d = offsets_d < head_dim
        valid_s = offsets_s < total
        query_row = tl.load(
            query_ptr
            + batch * stride_qb
            + token * stride_qt
            + head * stride_qh
            + offsets_d * stride_qd,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        key_rows = tl.load(
            key_ptr
            + batch * stride_kb
            + offsets_s[:, None] * stride_ks
            + kv_head * stride_kh
            + offsets_d[None, :] * stride_kd,
            mask=valid_s[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key_rows * query_row[None, :], axis=1) * attention_scale
        scores = tl.where(valid_s, scores, float("-inf"))
        maximum = tl.max(scores, axis=0)
        numerator = tl.exp(scores - maximum)
        numerator = tl.where(valid_s, numerator, 0.0)
        probabilities = numerator / tl.sum(numerator, axis=0)
        value_rows = tl.load(
            value_ptr
            + batch * stride_kb
            + offsets_s[:, None] * stride_ks
            + kv_head * stride_kh
            + offsets_d[None, :] * stride_kd,
            mask=valid_s[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        context = tl.sum(probabilities[:, None] * value_rows, axis=0)
        tl.store(
            output_ptr
            + batch * stride_ob
            + token * stride_ot
            + head * stride_oh
            + offsets_d * stride_od,
            context,
            mask=valid_d,
        )

    @triton.jit
    def _segmented_decode_attention_kernel(
        query_ptr,
        cache_key_ptr,
        cache_value_ptr,
        key_new_ptr,
        value_new_ptr,
        lengths_ptr,
        cache_row_indices_ptr,
        output_ptr,
        stride_qb: tl.constexpr,
        stride_qt: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_ckb: tl.constexpr,
        stride_cks: tl.constexpr,
        stride_ckh: tl.constexpr,
        stride_ckd: tl.constexpr,
        stride_cvb: tl.constexpr,
        stride_cvs: tl.constexpr,
        stride_cvh: tl.constexpr,
        stride_cvd: tl.constexpr,
        stride_nkb: tl.constexpr,
        stride_nkt: tl.constexpr,
        stride_nkh: tl.constexpr,
        stride_nkd: tl.constexpr,
        stride_nvb: tl.constexpr,
        stride_nvt: tl.constexpr,
        stride_nvh: tl.constexpr,
        stride_nvd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_ot: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        max_committed,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        attention_scale: tl.constexpr,
        has_cache_row_indices: tl.constexpr,
        block_s: tl.constexpr,
        block_d: tl.constexpr,
    ):
        """One-token causal GQA over two physical segments with online softmax."""

        program = tl.program_id(0)
        head = program % num_heads
        batch = program // num_heads
        kv_head = head // (num_heads // num_kv_heads)
        committed = tl.load(lengths_ptr + batch)
        if has_cache_row_indices:
            cache_batch = tl.load(cache_row_indices_ptr + batch)
        else:
            cache_batch = batch

        offsets_d = tl.arange(0, block_d)
        valid_d = offsets_d < head_dim
        query_row = tl.load(
            query_ptr + batch * stride_qb + head * stride_qh + offsets_d * stride_qd,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)

        running_max = float("-inf")
        running_sum = 0.0
        accumulator = tl.zeros((block_d,), dtype=tl.float32)
        offsets_in_tile = tl.arange(0, block_s)
        for block_start in tl.range(0, max_committed, block_s):
            offsets_s = block_start + offsets_in_tile
            valid_s = offsets_s < committed
            key_rows = tl.load(
                cache_key_ptr
                + cache_batch * stride_ckb
                + offsets_s[:, None] * stride_cks
                + kv_head * stride_ckh
                + offsets_d[None, :] * stride_ckd,
                mask=valid_s[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(key_rows * query_row[None, :], axis=1) * attention_scale
            logits = tl.where(valid_s, logits, float("-inf"))
            tile_max = tl.max(logits, axis=0)
            has_rows = block_start < committed
            next_max = tl.maximum(running_max, tile_max)
            correction = tl.where(has_rows, tl.exp(running_max - next_max), 1.0)
            weights = tl.where(valid_s, tl.exp(logits - next_max), 0.0)
            value_rows = tl.load(
                cache_value_ptr
                + cache_batch * stride_cvb
                + offsets_s[:, None] * stride_cvs
                + kv_head * stride_cvh
                + offsets_d[None, :] * stride_cvd,
                mask=valid_s[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            tile_sum = tl.sum(weights, axis=0)
            tile_accumulator = tl.sum(weights[:, None] * value_rows, axis=0)
            running_sum = tl.where(
                has_rows,
                running_sum * correction + tile_sum,
                running_sum,
            )
            accumulator = tl.where(
                has_rows,
                accumulator * correction + tile_accumulator,
                accumulator,
            )
            running_max = tl.where(has_rows, next_max, running_max)

        key_new = tl.load(
            key_new_ptr + batch * stride_nkb + kv_head * stride_nkh + offsets_d * stride_nkd,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        value_new = tl.load(
            value_new_ptr + batch * stride_nvb + kv_head * stride_nvh + offsets_d * stride_nvd,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        new_logit = tl.sum(key_new * query_row, axis=0) * attention_scale
        final_max = tl.maximum(running_max, new_logit)
        past_correction = tl.exp(running_max - final_max)
        new_weight = tl.exp(new_logit - final_max)
        accumulator = accumulator * past_correction + value_new * new_weight
        running_sum = running_sum * past_correction + new_weight
        tl.store(
            output_ptr + batch * stride_ob + head * stride_oh + offsets_d * stride_od,
            accumulator / running_sum,
            mask=valid_d,
        )


def load_compact_qrow_page(
    store: QStore,
    name: str,
    *,
    start_row: int = 0,
    end_row: int | None = None,
) -> CompactQRowPage:
    block = store._resolve(name)
    if block.get("kind") != "qrow":
        raise ValueError(f"{name} is not a qrow block")
    out_features, in_features = (int(value) for value in block["shape"])
    start = int(start_row)
    stop = out_features if end_row is None else int(end_row)
    if start < 0 or stop <= start or stop > out_features:
        raise ValueError(f"invalid row range [{start}, {stop}) for {name}")

    weight_start = int(block["w_off"]) + start * in_features
    weight_stop = int(block["w_off"]) + stop * in_features
    codes_np = np.asarray(store.w[weight_start:weight_stop], dtype=np.int8).reshape(
        stop - start,
        in_features,
    )
    scale_start = int(block["s_off"]) // 4 + start
    scales_np = np.asarray(store.s[scale_start : scale_start + stop - start], dtype=np.float32)
    codes = torch.from_numpy(codes_np.copy())
    scales = torch.from_numpy(scales_np.copy())
    device = torch.device(store.device)
    if device.type != "cpu":
        codes = codes.to(device)
        scales = scales.to(device)
    return CompactQRowPage(
        name=name,
        start_row=start,
        end_row=stop,
        in_features=in_features,
        codes=codes,
        scales=scales,
    )


@torch.inference_mode()
def dequantized_reference_matmul(
    page: CompactQRowPage,
    activations: torch.Tensor,
) -> torch.Tensor:
    flattened = activations.reshape(-1, activations.shape[-1])
    weights = page.codes.float() * page.scales[:, None]
    if flattened.dtype in (torch.bfloat16, torch.float16):
        weights = weights.to(flattened.dtype)
    output = flattened @ weights.t()
    return output.reshape(*activations.shape[:-1], page.out_features)


def _validate_paired_qrow_pages(
    gate_page: CompactQRowPage,
    up_page: CompactQRowPage,
    activations: torch.Tensor,
) -> None:
    if activations.ndim < 2 or activations.dtype is not torch.bfloat16:
        raise ValueError("paired SwiGLU requires BF16 activation rows")
    if gate_page.in_features != activations.shape[-1]:
        raise ValueError("activation width does not match paired compact qrow pages")
    if (
        gate_page.in_features != up_page.in_features
        or gate_page.out_features != up_page.out_features
        or gate_page.start_row != up_page.start_row
        or gate_page.end_row != up_page.end_row
    ):
        raise ValueError("gate/up compact qrow pages must have identical geometry")
    tensors = (gate_page.codes, gate_page.scales, up_page.codes, up_page.scales)
    if any(tensor.device != activations.device for tensor in tensors):
        raise ValueError("paired compact qrow pages and activations must use one device")


@torch.inference_mode()
def fused_qrow_swiglu_reference(
    gate_page: CompactQRowPage,
    up_page: CompactQRowPage,
    activations: torch.Tensor,
) -> torch.Tensor:
    """Materialized reference for the paired W8A16 BF16 SwiGLU contract."""

    _validate_paired_qrow_pages(gate_page, up_page, activations)
    gate = dequantized_reference_matmul(gate_page, activations).to(torch.bfloat16)
    up = dequantized_reference_matmul(up_page, activations).to(torch.bfloat16)
    silu = (gate.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    return (silu.float() * up.float()).to(torch.bfloat16)


@torch.inference_mode()
def fused_qrow_swiglu(
    gate_page: CompactQRowPage,
    up_page: CompactQRowPage,
    activations: torch.Tensor,
    *,
    require_triton: bool = False,
    block_m: int = 16,
) -> tuple[torch.Tensor, str]:
    """Fuse two compact W8A16 projections and their BF16 SwiGLU epilogue."""

    _validate_paired_qrow_pages(gate_page, up_page, activations)
    if block_m not in {16, 32, 64}:
        raise ValueError("block_m must be one of 16, 32, or 64")
    flattened = activations.reshape(-1, activations.shape[-1]).contiguous()
    use_triton = flattened.device.type == "cuda" and triton is not None
    if not use_triton:
        if require_triton:
            raise RuntimeError("paired W8A16 SwiGLU requires CUDA BF16 with Triton")
        return (
            fused_qrow_swiglu_reference(gate_page, up_page, activations),
            "reference-materialized-paired-w8a16-swiglu-bf16-v1",
        )

    rows = int(flattened.shape[0])
    output = torch.empty(
        (rows, gate_page.out_features),
        device=flattened.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(rows, block_m), triton.cdiv(gate_page.out_features, 64))
    _qrow_w8a16_paired_swiglu_kernel[grid](
        flattened,
        gate_page.codes,
        gate_page.scales,
        up_page.codes,
        up_page.scales,
        output,
        rows,
        out_features=gate_page.out_features,
        in_features=gate_page.in_features,
        block_m=block_m,
        block_n=64,
        block_k=32,
        num_warps=4,
        num_stages=3,
    )
    return (
        output.reshape(*activations.shape[:-1], gate_page.out_features),
        "triton-paired-w8a16-swiglu-bf16-v1",
    )


@torch.inference_mode()
def fused_qrow_matmul(
    page: CompactQRowPage,
    activations: torch.Tensor,
    *,
    require_triton: bool = False,
    block_m: int = 16,
) -> tuple[torch.Tensor, str]:
    if activations.shape[-1] != page.in_features:
        raise ValueError("activation width does not match compact qrow page")
    if page.codes.device != activations.device or page.scales.device != activations.device:
        raise ValueError("compact qrow page and activations must use one device")
    if block_m not in {16, 32, 64}:
        raise ValueError("block_m must be one of 16, 32, or 64")
    flattened = activations.reshape(-1, activations.shape[-1]).contiguous()
    use_triton = (
        flattened.device.type == "cuda"
        and triton is not None
        and flattened.dtype in (torch.bfloat16, torch.float16)
    )
    if not use_triton:
        if require_triton:
            raise RuntimeError("fused qrow execution requires CUDA BF16/FP16 with Triton")
        return dequantized_reference_matmul(page, activations), "reference-materialized"

    rows = int(flattened.shape[0])
    output = torch.empty(
        (rows, page.out_features),
        device=flattened.device,
        dtype=flattened.dtype,
    )
    grid = (triton.cdiv(rows, block_m), triton.cdiv(page.out_features, 64))
    _qrow_w8a16_kernel[grid](
        flattened,
        page.codes,
        page.scales,
        output,
        rows,
        out_features=page.out_features,
        in_features=page.in_features,
        block_m=block_m,
        block_n=64,
        block_k=32,
        use_bf16=flattened.dtype is torch.bfloat16,
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(*activations.shape[:-1], page.out_features), "triton-w8a16-qrow-v1"


@torch.inference_mode()
def qrow_top2_reference(
    page: CompactQRowPage,
    activations: torch.Tensor,
    *,
    semantic_row_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_count = _semantic_row_count(page, semantic_row_count)
    logits = dequantized_reference_matmul(page, activations)[..., :row_count].float()
    first_values, first_offsets = logits.max(dim=-1)
    without_first = logits.clone()
    without_first.scatter_(-1, first_offsets.unsqueeze(-1), float("-inf"))
    second_values, second_offsets = without_first.max(dim=-1)
    indices = torch.stack((first_offsets, second_offsets), dim=-1) + page.start_row
    values = torch.stack((first_values, second_values), dim=-1)
    return indices, values


@torch.inference_mode()
def rerank_qrow_candidates_fp32(
    page: CompactQRowPage,
    activations: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a compact shortlist in FP32; this does not certify shortlist coverage."""

    return _rerank_qrow_candidates_fp32(
        page,
        activations,
        candidate_indices,
        validate_candidate_bounds=True,
    )


def _rerank_qrow_candidates_fp32(
    page: CompactQRowPage,
    activations: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    validate_candidate_bounds: bool,
) -> tuple[torch.Tensor, torch.Tensor]:

    if activations.ndim < 2 or activations.shape[-1] != page.in_features:
        raise ValueError("activation shape does not match compact head page")
    if candidate_indices.shape[:-1] != activations.shape[:-1]:
        raise ValueError("candidate rows must match the activation prefix shape")
    if candidate_indices.shape[-1] < 1:
        raise ValueError("at least one candidate is required")
    if candidate_indices.device != activations.device:
        raise ValueError("candidate indices and activations must use one device")
    if page.codes.device != activations.device or page.scales.device != activations.device:
        raise ValueError("head page and activations must use one device")

    flattened = activations.reshape(-1, page.in_features)
    candidates = candidate_indices.reshape(flattened.shape[0], -1).to(torch.long)
    local_indices = candidates - page.start_row
    if validate_candidate_bounds and bool(
        ((local_indices < 0) | (local_indices >= page.out_features)).any().item()
    ):
        raise ValueError("candidate index falls outside the compact head page")
    flat_local = local_indices.reshape(-1)
    selected_codes = page.codes.index_select(0, flat_local).reshape(
        flattened.shape[0],
        candidates.shape[1],
        page.in_features,
    )
    selected_scales = page.scales.index_select(0, flat_local).reshape(
        flattened.shape[0],
        candidates.shape[1],
    )
    weights = selected_codes.float() * selected_scales[..., None]
    scores = (flattened.float()[:, None, :] * weights).sum(dim=-1)
    best_values = scores.max(dim=-1).values
    tied_indices = torch.where(
        scores == best_values[:, None],
        candidates,
        torch.iinfo(torch.long).max,
    )
    best_indices = tied_indices.min(dim=-1).values
    output_shape = activations.shape[:-1]
    return best_indices.reshape(output_shape), best_values.reshape(output_shape)


@torch.inference_mode()
def fused_qrow_top2(
    page: CompactQRowPage,
    activations: torch.Tensor,
    *,
    semantic_row_count: int | None = None,
    require_triton: bool = False,
    block_m: int = 16,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, str, int]:
    """Return top two rows from a semantic prefix without materializing logits."""

    if activations.ndim < 2 or activations.shape[-1] != page.in_features:
        raise ValueError("activation shape does not match compact head page")
    if page.codes.device != activations.device or page.scales.device != activations.device:
        raise ValueError("head page and activations must use one device")
    if block_m not in {16, 32, 64} or block_n != 64:
        raise ValueError("supported head tiles are block_m in {16,32,64}, block_n=64")
    row_count = _semantic_row_count(page, semantic_row_count)
    flattened = activations.reshape(-1, page.in_features).contiguous()
    use_triton = (
        flattened.device.type == "cuda"
        and triton is not None
        and flattened.dtype in (torch.bfloat16, torch.float16)
    )
    if not use_triton:
        if require_triton:
            raise RuntimeError("fused qrow top-2 requires CUDA BF16/FP16 with Triton")
        indices, values = qrow_top2_reference(
            page,
            activations,
            semantic_row_count=row_count,
        )
        return indices, values, "reference-materialized-top2", 0

    rows = int(flattened.shape[0])
    vocabulary_blocks = triton.cdiv(row_count, block_n)
    partial_values = torch.empty(
        (rows, vocabulary_blocks, 2),
        device=flattened.device,
        dtype=torch.float32,
    )
    partial_indices = torch.empty(
        (rows, vocabulary_blocks, 2),
        device=flattened.device,
        dtype=torch.int32,
    )
    output_values = torch.empty((rows, 2), device=flattened.device, dtype=torch.float32)
    output_indices = torch.empty((rows, 2), device=flattened.device, dtype=torch.int32)
    grid = (triton.cdiv(rows, block_m), vocabulary_blocks)
    _qrow_w8a16_block_top2_kernel[grid](
        flattened,
        page.codes,
        page.scales,
        partial_values,
        partial_indices,
        rows,
        out_features=row_count,
        in_features=page.in_features,
        vocabulary_blocks=vocabulary_blocks,
        start_row=page.start_row,
        block_m=block_m,
        block_n=block_n,
        block_k=32,
        use_bf16=flattened.dtype is torch.bfloat16,
        num_warps=4,
        num_stages=3,
    )
    candidate_count = vocabulary_blocks * 2
    reduction_tile = triton.next_power_of_2(candidate_count)
    _reduce_top2_blocks_kernel[(rows,)](
        partial_values,
        partial_indices,
        output_values,
        output_indices,
        candidate_count=candidate_count,
        reduction_tile=reduction_tile,
        num_warps=8,
        num_stages=2,
    )
    working_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (partial_values, partial_indices, output_values, output_indices)
    )
    output_shape = (*activations.shape[:-1], 2)
    return (
        output_indices.reshape(output_shape),
        output_values.reshape(output_shape),
        "triton-w8a16-qrow-block-top2-v1",
        int(working_bytes),
    )


@torch.inference_mode()
def fused_qrow_reranked_argmax(
    page: CompactQRowPage,
    activations: torch.Tensor,
    *,
    semantic_row_count: int | None = None,
    require_triton: bool = False,
    block_m: int = 16,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, str, int]:
    """Fused top-two search followed by compact FP32 candidate reranking."""

    candidate_indices, _, implementation, top2_working_bytes = fused_qrow_top2(
        page,
        activations,
        semantic_row_count=semantic_row_count,
        require_triton=require_triton,
        block_m=block_m,
        block_n=block_n,
    )
    # These indices were emitted by the masked kernel/reference above. Avoid an otherwise
    # unnecessary device synchronization solely to revalidate our own bounded output.
    indices, values = _rerank_qrow_candidates_fp32(
        page,
        activations,
        candidate_indices,
        validate_candidate_bounds=False,
    )
    rows = activations.numel() // page.in_features
    candidate_count = int(candidate_indices.shape[-1])
    row_count = _semantic_row_count(page, semantic_row_count)
    working_bytes = reranked_argmax_workspace_bytes(
        rows=rows,
        in_features=page.in_features,
        semantic_row_count=row_count,
        block_n=block_n,
        candidate_count=candidate_count,
    )
    return (
        indices,
        values,
        f"{implementation}+fp32-candidate-rerank-v1",
        max(int(top2_working_bytes), working_bytes),
    )


def _validate_residual_rms_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
) -> int:
    if residual.ndim < 2 or update.shape != residual.shape:
        raise ValueError("residual and update must have identical row-major shapes")
    if residual.dtype is not torch.bfloat16 or update.dtype is not torch.bfloat16:
        raise ValueError("fused residual RMSNorm requires a BF16 residual boundary")
    width = int(residual.shape[-1])
    if weight.ndim != 1 or weight.numel() != width:
        raise ValueError("RMSNorm weight width does not match the residual")
    if residual.device != update.device or residual.device != weight.device:
        raise ValueError("residual, update, and RMSNorm weight must use one device")
    return width


@torch.inference_mode()
def fused_residual_rms_norm_reference(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add through FP32, round to BF16, and normalize the rounded residual."""

    _validate_residual_rms_norm(residual, update, weight)
    rounded = (residual.float() + update.float()).to(torch.bfloat16)
    normalized = rounded.float() * torch.rsqrt(
        rounded.float().pow(2).mean(-1, keepdim=True) + float(eps)
    )
    return rounded, (normalized * weight.float()).to(torch.bfloat16)


@torch.inference_mode()
def fused_residual_rms_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    require_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """One row-stable residual/BF16-boundary/RMSNorm operation."""

    width = _validate_residual_rms_norm(residual, update, weight)
    use_triton = residual.device.type == "cuda" and triton is not None
    if not use_triton:
        if require_triton:
            raise RuntimeError("fused residual RMSNorm requires CUDA BF16 with Triton")
        rounded, normalized = fused_residual_rms_norm_reference(
            residual,
            update,
            weight,
            eps,
        )
        return rounded, normalized, "torch-residual-bf16-rms-reference-v1"

    residual_rows = residual.reshape(-1, width).contiguous()
    update_rows = update.reshape(-1, width).contiguous()
    rounded = torch.empty_like(residual_rows)
    normalized = torch.empty_like(residual_rows)
    rows = int(residual_rows.shape[0])
    block_n = triton.next_power_of_2(width)
    _residual_bf16_rms_norm_kernel[(rows,)](
        residual_rows,
        update_rows,
        weight.contiguous(),
        rounded,
        normalized,
        rows,
        width=width,
        eps=float(eps),
        block_n=block_n,
        num_warps=8 if width >= 2048 else 4,
        num_stages=2,
    )
    return (
        rounded.reshape_as(residual),
        normalized.reshape_as(residual),
        "triton-residual-bf16-rms-v1",
    )


@torch.inference_mode()
def stable_rms_norm_reference(
    activations: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    normalized = activations.float() * torch.rsqrt(
        activations.float().pow(2).mean(-1, keepdim=True) + float(eps)
    )
    return (normalized * weight.float()).to(activations.dtype)


@torch.inference_mode()
def stable_rms_norm(
    activations: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    require_triton: bool = False,
) -> tuple[torch.Tensor, str]:
    if activations.ndim < 2 or weight.ndim != 1:
        raise ValueError("activations need rows and weight must be a vector")
    width = int(activations.shape[-1])
    if weight.numel() != width or activations.device != weight.device:
        raise ValueError("RMSNorm weight width/device does not match activations")
    use_triton = (
        activations.device.type == "cuda"
        and triton is not None
        and activations.dtype in (torch.bfloat16, torch.float16)
    )
    if not use_triton:
        if require_triton:
            raise RuntimeError("stable RMSNorm requires CUDA BF16/FP16 with Triton")
        return stable_rms_norm_reference(activations, weight, eps), "torch-row-reference"

    flattened = activations.reshape(-1, width).contiguous()
    output = torch.empty_like(flattened)
    rows = int(flattened.shape[0])
    block_n = triton.next_power_of_2(width)
    _stable_rms_norm_kernel[(rows,)](
        flattened,
        weight.contiguous(),
        output,
        rows,
        width=width,
        eps=float(eps),
        block_n=block_n,
        num_warps=4,
        num_stages=2,
    )
    return output.reshape_as(activations), "triton-row-stable-rms-v1"


def _validate_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor | Sequence[int],
    *,
    check_values: bool,
) -> torch.Tensor:
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("query and key/value need [B,T,H,D] and [B,S,Hkv,D]")
    batch, token_count, num_heads, head_dim = query.shape
    key_batch, max_total, num_kv_heads, key_dim = key.shape
    if min(batch, token_count, num_heads, head_dim, max_total, num_kv_heads) <= 0:
        raise ValueError("attention tensors cannot contain an empty dimension")
    if key_batch != batch or key_dim != head_dim or num_heads % num_kv_heads:
        raise ValueError("attention batch/head dimensions are incompatible")
    if query.device != key.device or value.device != key.device:
        raise ValueError("query, key, and value must use one device")
    if query.dtype != key.dtype or value.dtype != key.dtype:
        raise ValueError("query, key, and value must use one dtype")
    length_tensor = torch.as_tensor(lengths, device=query.device, dtype=torch.long)
    if tuple(length_tensor.shape) != (batch,):
        raise ValueError("lengths must contain one value per request")
    if check_values and (
        bool((length_tensor < 0).any()) or bool((length_tensor + token_count > max_total).any())
    ):
        raise ValueError("committed lengths plus token block exceed KV storage")
    return length_tensor.contiguous()


@torch.inference_mode()
def stable_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    length_tensor = _validate_attention(query, key, value, lengths, check_values=True)
    _, token_count, num_heads, head_dim = query.shape
    repeat = num_heads // int(key.shape[2])
    scale = head_dim**-0.5
    request_outputs: list[torch.Tensor] = []
    for request, committed in enumerate(length_tensor.tolist()):
        token_outputs: list[torch.Tensor] = []
        for position in range(token_count):
            total = int(committed) + position + 1
            query_row = query[request, position].float()
            key_heads = key[request, :total].repeat_interleave(repeat, dim=1).permute(1, 0, 2)
            value_heads = value[request, :total].repeat_interleave(repeat, dim=1).permute(1, 0, 2)
            scores = (query_row[:, None, :] * key_heads.float()).sum(dim=-1) * scale
            probabilities = torch.softmax(scores, dim=-1)
            token_outputs.append((probabilities[:, :, None] * value_heads.float()).sum(dim=1))
        request_outputs.append(torch.stack(token_outputs, dim=0))
    return torch.stack(request_outputs, dim=0).to(query.dtype)


@torch.inference_mode()
def stable_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor | Sequence[int],
    *,
    require_triton: bool = False,
    sequence_tile: int | None = None,
    validate_lengths: bool = True,
) -> tuple[torch.Tensor, str]:
    length_tensor = _validate_attention(
        query,
        key,
        value,
        lengths,
        check_values=validate_lengths,
    )
    if sequence_tile is not None and (
        sequence_tile < key.shape[1] or sequence_tile < 16 or sequence_tile & (sequence_tile - 1)
    ):
        raise ValueError("sequence_tile must be a power of two covering KV storage")
    use_triton = (
        query.device.type == "cuda"
        and triton is not None
        and query.dtype in (torch.bfloat16, torch.float16)
    )
    if not use_triton:
        if require_triton:
            raise RuntimeError("stable attention requires CUDA BF16/FP16 with Triton")
        return stable_attention_reference(query, key, value, length_tensor), "torch-row-reference"

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    batch, token_count, num_heads, head_dim = (int(item) for item in query.shape)
    num_kv_heads = int(key.shape[2])
    block_s = sequence_tile or max(16, triton.next_power_of_2(int(key.shape[1])))
    output = torch.empty_like(query)
    _stable_attention_kernel[(batch * token_count * num_heads,)](
        query,
        key,
        value,
        length_tensor,
        output,
        *query.stride(),
        *key.stride(),
        *output.stride(),
        token_count=token_count,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attention_scale=head_dim**-0.5,
        block_s=block_s,
        block_d=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=2,
    )
    return output, "triton-row-stable-gqa-v1"


def _validate_segmented_decode_attention(
    query: torch.Tensor,
    cache_key: torch.Tensor,
    cache_value: torch.Tensor,
    key_new: torch.Tensor,
    value_new: torch.Tensor,
    lengths: torch.Tensor | Sequence[int],
    cache_row_indices: torch.Tensor | None,
    *,
    check_values: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if query.ndim != 4:
        raise ValueError("decode query must have shape [B,1,H,D]")
    batch, token_count, num_heads, head_dim = query.shape
    if token_count != 1:
        raise ValueError("segmented decode attention accepts exactly one new token")
    if cache_key.ndim != 4 or cache_value.shape != cache_key.shape:
        raise ValueError("committed key/value cache must have shape [slots,S,Hkv,D]")
    cache_slots, cache_capacity, num_kv_heads, cache_head_dim = cache_key.shape
    expected_new = (batch, 1, num_kv_heads, head_dim)
    if tuple(key_new.shape) != expected_new or tuple(value_new.shape) != expected_new:
        raise ValueError(f"new key/value tensors must have shape {expected_new}")
    if min(batch, num_heads, head_dim, cache_slots, cache_capacity, num_kv_heads) <= 0:
        raise ValueError("segmented decode tensors cannot contain an empty dimension")
    if cache_head_dim != head_dim or num_heads % num_kv_heads:
        raise ValueError("segmented decode batch/head dimensions are incompatible")
    tensors = (cache_key, cache_value, key_new, value_new)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("segmented decode tensors must use one device")
    if any(tensor.dtype != query.dtype for tensor in tensors):
        raise ValueError("segmented decode tensors must use one dtype")

    length_tensor = torch.as_tensor(lengths, device=query.device, dtype=torch.long)
    if tuple(length_tensor.shape) != (batch,):
        raise ValueError("lengths must contain one value per decode request")
    if check_values and bool(((length_tensor < 0) | (length_tensor > cache_capacity)).any().item()):
        raise ValueError("committed lengths exceed segmented decode cache storage")

    row_indices = cache_row_indices
    if row_indices is None:
        if cache_slots != batch:
            raise ValueError(
                "cache slots must equal query batch when cache_row_indices is not supplied"
            )
    else:
        if not isinstance(row_indices, torch.Tensor):
            raise TypeError("cache_row_indices must be a tensor or None")
        if row_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("cache_row_indices must use int32 or int64")
        if row_indices.device != query.device:
            raise ValueError("cache_row_indices must use the attention device")
        if tuple(row_indices.shape) != (batch,):
            raise ValueError("cache_row_indices must have shape [B]")
        if check_values and bool(((row_indices < 0) | (row_indices >= cache_slots)).any().item()):
            raise ValueError("cache_row_indices select a slot outside committed KV storage")
        row_indices = row_indices.contiguous()
    return length_tensor.contiguous(), row_indices


@torch.inference_mode()
def segmented_decode_attention_reference(
    query: torch.Tensor,
    cache_key: torch.Tensor,
    cache_value: torch.Tensor,
    key_new: torch.Tensor,
    value_new: torch.Tensor,
    lengths: torch.Tensor | Sequence[int],
    *,
    cache_row_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure one-token GQA reference over committed and provisional K/V segments.

    Each head advances an FP32 online-softmax state ``(maximum, denominator,
    accumulator)`` one source row at a time. It intentionally never joins the two
    segments, repeats K/V heads, or materializes a score/probability vector.
    """

    length_tensor, row_indices = _validate_segmented_decode_attention(
        query,
        cache_key,
        cache_value,
        key_new,
        value_new,
        lengths,
        cache_row_indices,
        check_values=True,
    )
    batch, _, num_heads, head_dim = query.shape
    num_kv_heads = int(cache_key.shape[2])
    heads_per_kv = num_heads // num_kv_heads
    scale = head_dim**-0.5
    committed_lengths = tuple(int(value) for value in length_tensor.tolist())
    physical_rows = (
        tuple(range(batch))
        if row_indices is None
        else tuple(int(value) for value in row_indices.tolist())
    )
    request_outputs: list[torch.Tensor] = []
    for request, (physical_row, committed) in enumerate(
        zip(physical_rows, committed_lengths, strict=True)
    ):
        head_outputs: list[torch.Tensor] = []
        for head in range(num_heads):
            kv_head = head // heads_per_kv
            query_row = query[request, 0, head].float()
            running_max: torch.Tensor | None = None
            running_sum = torch.zeros((), device=query.device, dtype=torch.float32)
            accumulator = torch.zeros(head_dim, device=query.device, dtype=torch.float32)
            for source in range(committed + 1):
                if source == committed:
                    source_key = key_new[request, 0, kv_head].float()
                    source_value = value_new[request, 0, kv_head].float()
                else:
                    source_key = cache_key[physical_row, source, kv_head].float()
                    source_value = cache_value[physical_row, source, kv_head].float()
                logit = torch.sum(query_row * source_key) * scale
                if running_max is None:
                    running_max = logit
                    running_sum.fill_(1.0)
                    accumulator.copy_(source_value)
                    continue
                next_max = torch.maximum(running_max, logit)
                old_weight = torch.exp(running_max - next_max)
                new_weight = torch.exp(logit - next_max)
                running_sum = running_sum * old_weight + new_weight
                accumulator = accumulator * old_weight + source_value * new_weight
                running_max = next_max
            head_outputs.append(accumulator / running_sum)
        request_outputs.append(torch.stack(head_outputs, dim=0).unsqueeze(0))
    return torch.stack(request_outputs, dim=0).to(query.dtype)


@torch.inference_mode()
def segmented_decode_attention(
    query: torch.Tensor,
    cache_key: torch.Tensor,
    cache_value: torch.Tensor,
    key_new: torch.Tensor,
    value_new: torch.Tensor,
    lengths: torch.Tensor | Sequence[int],
    *,
    cache_row_indices: torch.Tensor | None = None,
    require_triton: bool = False,
    sequence_tile: int = 64,
    max_committed: int | None = None,
    validate_lengths: bool = True,
) -> tuple[torch.Tensor, str]:
    """Execute bounded, tiled one-token GQA without assembling a combined K/V tensor."""

    length_tensor, row_indices = _validate_segmented_decode_attention(
        query,
        cache_key,
        cache_value,
        key_new,
        value_new,
        lengths,
        cache_row_indices,
        check_values=validate_lengths,
    )
    if isinstance(sequence_tile, bool) or not isinstance(sequence_tile, int):
        raise TypeError("sequence_tile must be an integer")
    if sequence_tile < 16 or sequence_tile > 256 or sequence_tile & (sequence_tile - 1):
        raise ValueError("sequence_tile must be a power of two in [16, 256]")
    if max_committed is None:
        max_committed = int(length_tensor.max().item())
    if isinstance(max_committed, bool) or not isinstance(max_committed, int):
        raise TypeError("max_committed must be an integer or None")
    if max_committed < 0 or max_committed > int(cache_key.shape[1]):
        raise ValueError("max_committed must fit inside committed KV storage")
    if validate_lengths and bool((length_tensor > max_committed).any().item()):
        raise ValueError("max_committed does not cover every committed request length")

    use_triton = (
        query.device.type == "cuda"
        and triton is not None
        and query.dtype in (torch.bfloat16, torch.float16)
    )
    if not use_triton:
        if require_triton:
            raise RuntimeError("segmented decode attention requires CUDA BF16/FP16 with Triton")
        return (
            segmented_decode_attention_reference(
                query,
                cache_key,
                cache_value,
                key_new,
                value_new,
                length_tensor,
                cache_row_indices=row_indices,
            ),
            "torch-segmented-online-softmax-gqa-decode-v1",
        )

    output = torch.empty_like(query)
    batch, _, num_heads, head_dim = (int(item) for item in query.shape)
    num_kv_heads = int(cache_key.shape[2])
    row_pointer = length_tensor if row_indices is None else row_indices
    _segmented_decode_attention_kernel[(batch * num_heads,)](
        query,
        cache_key,
        cache_value,
        key_new,
        value_new,
        length_tensor,
        row_pointer,
        output,
        *query.stride(),
        *cache_key.stride(),
        *cache_value.stride(),
        *key_new.stride(),
        *value_new.stride(),
        *output.stride(),
        max_committed,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attention_scale=head_dim**-0.5,
        has_cache_row_indices=row_indices is not None,
        block_s=sequence_tile,
        block_d=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=2,
    )
    return output, "triton-segmented-flash-gqa-decode-v1"


class DenseQStore(QStore):
    """Int8 QStore with a byte-bounded compact page cache and fused projections."""

    def __init__(
        self,
        model_name: str,
        *,
        root: Path,
        device: str = "cuda",
        compute_dtype: str = "bf16",
        compact_cache_mb: float = 0.0,
        require_triton: bool = True,
        stable_block_m: int = 16,
        pin_fp32_aux: bool = False,
    ) -> None:
        requested = torch.device(device)
        super().__init__(
            model_name,
            root=root,
            cache_mb=0.0,
            compute_dtype=compute_dtype,
            explicit_device=str(requested),
            enable_prefetch_ring=False,
        )
        # The dense engine consumes compact_page/row_blocks only — never weight(), the
        # ring's sole consumer — so the base-class ring (default ON for CUDA-granted
        # stores) would pin 3 slots x largest-block host RAM (~0.4GB at 0.5B, ~1.6GB
        # at 7B) plus worker threads for zero hits. Keep it off.
        self.disable_ring()
        if requested.type == "cuda" and require_triton and triton is None:
            raise RuntimeError("dense QStore CUDA execution requires Triton")
        if stable_block_m not in {16, 32, 64}:
            raise ValueError("stable_block_m must be one of 16, 32, or 64")
        self.device = str(requested)
        self.require_triton = bool(require_triton)
        self.stable_block_m = int(stable_block_m)
        self.compact_cache_budget = int(float(compact_cache_mb) * 1e6)
        self.compact_cache: OrderedDict[str, CompactQRowPage] = OrderedDict()
        self.compact_cache_bytes = 0
        # Explicit linked-image pages are admitted once and kept outside the evicting
        # compact LRU. This gives the runtime stable codes/scales addresses without
        # changing the established projection kernels.
        self._prebound_pages: dict[str, CompactQRowPage] = {}
        self._prebound_compact_bytes = 0
        self.pin_fp32_aux = bool(pin_fp32_aux)
        self.fp32_aux_cache: dict[str, torch.Tensor] = {}
        self.fp32_aux_cache_bytes = 0
        self._prebound_fp32: dict[str, torch.Tensor] = {}
        self._prebound_fp32_bytes = 0
        self.resident_exact_heads: dict[str, torch.Tensor] = {}
        self.resident_exact_head_budget_bytes = 0
        self.resident_exact_head_bytes = 0
        self.stats = DenseQStoreStats()
        if self.pin_fp32_aux:
            try:
                self._pin_all_fp32_auxiliary()
            except BaseException:
                self.fp32_aux_cache.clear()
                self.fp32_aux_cache_bytes = 0
                super().close()
                raise

    def reset_stats(self) -> None:
        self.stats = DenseQStoreStats(
            peak_compact_resident_bytes=(
                self.compact_cache_bytes
                + self._prebound_compact_bytes
                + self.fp32_aux_cache_bytes
                + self._prebound_fp32_bytes
            )
        )

    def stats_snapshot(self) -> dict[str, Any]:
        snapshot = self.stats.snapshot(
            resident_bytes=self.compact_cache_bytes + self._prebound_compact_bytes,
            cache_entries=len(self.compact_cache) + len(self._prebound_pages),
        )
        snapshot["stable_block_m"] = self.stable_block_m
        snapshot["compact_cache_budget_bytes"] = self.compact_cache_budget
        snapshot["cache_kind"] = "compact-device-lru"
        snapshot["fp32_aux_pinning_enabled"] = self.pin_fp32_aux
        snapshot["fp32_aux_cache_entries"] = len(self.fp32_aux_cache)
        snapshot["fp32_aux_resident_bytes"] = self.fp32_aux_cache_bytes
        snapshot["resident_physical_bytes"] = (
            self.compact_cache_bytes
            + self._prebound_compact_bytes
            + self.fp32_aux_cache_bytes
            + self._prebound_fp32_bytes
            + self.resident_exact_head_bytes
        )
        snapshot["resident_exact_head_entries"] = len(self.resident_exact_heads)
        snapshot["resident_exact_head_budget_bytes"] = self.resident_exact_head_budget_bytes
        snapshot["resident_exact_head_bytes"] = self.resident_exact_head_bytes
        snapshot["prebound_page_entries"] = len(self._prebound_pages)
        snapshot["prebound_compact_bytes"] = self._prebound_compact_bytes
        snapshot["prebound_fp32_entries"] = len(self._prebound_fp32)
        snapshot["prebound_fp32_bytes"] = self._prebound_fp32_bytes
        return snapshot

    def _pin_all_fp32_auxiliary(self) -> None:
        physical_names = tuple(
            dict.fromkeys(
                self._physical_key(name)
                for name, block in self.blocks.items()
                if isinstance(block, dict) and self._resolve(name).get("kind") == "fp32"
            )
        )
        required = sum(int(self.blocks[name].get("e_len", 0)) for name in physical_names)
        if required > self.compact_cache_budget:
            raise MemoryError(
                "pinned FP32 auxiliary residency exceeds the provider cache budget "
                f"({required} > {self.compact_cache_budget} bytes)"
            )
        cache: dict[str, torch.Tensor] = {}
        for name in physical_names:
            cache[name] = super().fp32(name).contiguous()
        resident = sum(_tensor_bytes(tensor) for tensor in cache.values())
        if resident != required:
            raise RuntimeError(
                "pinned FP32 auxiliary bytes disagree with the verified manifest "
                f"({resident} != {required})"
            )
        self.fp32_aux_cache = cache
        self.fp32_aux_cache_bytes = resident
        self.stats.fp32_aux_cache_loads += len(cache)
        self.stats.peak_compact_resident_bytes = max(
            self.stats.peak_compact_resident_bytes,
            self.fp32_aux_cache_bytes,
        )

    def fp32(self, name: str) -> torch.Tensor:
        physical = self._physical_key(name)
        prebound = getattr(self, "_prebound_fp32", {}).get(physical)
        if prebound is not None:
            return prebound
        cached = self.fp32_aux_cache.get(physical)
        if cached is not None:
            return cached
        if self.pin_fp32_aux and self._resolve(name).get("kind") == "fp32":
            raise RuntimeError(f"pinned FP32 auxiliary {name!r} is unexpectedly absent")
        return super().fp32(name)

    def prebind(
        self,
        names: Sequence[str],
        *,
        max_resident_bytes: int | None = None,
    ) -> dict[str, int]:
        """Admit named compact pages/FP32 auxiliaries as non-evictable resources.

        Aliases collapse to one physical allocation, so a tied ``lm_head`` never doubles
        residency. The manifest is checked before new allocations when an explicit budget
        is supplied; the ordinary compact LRU remains available for all other pages.
        """

        requested = tuple(str(name) for name in names)
        if not requested:
            return {"qrow_pages": 0, "fp32_tensors": 0, "resident_bytes": 0}
        physical_names = tuple(dict.fromkeys(self._physical_key(name) for name in requested))
        new_qrows: list[str] = []
        new_fp32: list[str] = []
        required = 0
        for physical in physical_names:
            block = self._resolve(physical)
            kind = block.get("kind")
            if kind == "qrow":
                if physical not in self._prebound_pages:
                    new_qrows.append(physical)
                    required += int(block.get("w_len", 0)) + int(block.get("s_len", 0))
            elif kind == "fp32":
                if physical not in self._prebound_fp32:
                    new_fp32.append(physical)
                    required += int(block.get("e_len", 0))
            else:
                raise ValueError(f"cannot prebind unsupported block {physical!r}")

        if max_resident_bytes is not None:
            budget = int(max_resident_bytes)
            if budget <= 0:
                raise ValueError("prebound residency budget must be positive")
            current = self._prebound_compact_bytes + self._prebound_fp32_bytes
            if current + required > budget:
                raise MemoryError(
                    "prebound linked pages exceed their explicit residency budget "
                    f"({current + required} > {budget} bytes)"
                )
            device = torch.device(self.device)
            if device.type == "cuda" and required:
                free_bytes, _ = torch.cuda.mem_get_info(device)
                if current + required > int(free_bytes):
                    raise MemoryError(
                        "prebound linked pages exceed currently free device memory "
                        f"({current + required} > {int(free_bytes)} bytes)"
                    )

        loaded_qrows = 0
        loaded_fp32 = 0
        for physical in new_qrows:
            page = load_compact_qrow_page(self, physical)
            self._prebound_pages[physical] = page
            page_bytes = int(page.compact_bytes)
            self._prebound_compact_bytes += page_bytes
            self.stats.page_loads += 1
            if page.codes.device.type == "cuda":
                self.stats.compact_h2d_bytes += page_bytes
            self.stats.peak_compact_resident_bytes = max(
                self.stats.peak_compact_resident_bytes,
                self._prebound_compact_bytes + self._prebound_fp32_bytes,
            )
            loaded_qrows += 1
        for physical in new_fp32:
            tensor = super().fp32(physical).contiguous()
            self._prebound_fp32[physical] = tensor
            self._prebound_fp32_bytes += _tensor_bytes(tensor)
            self.stats.peak_compact_resident_bytes = max(
                self.stats.peak_compact_resident_bytes,
                self._prebound_compact_bytes + self._prebound_fp32_bytes,
            )
            loaded_fp32 += 1
        return {
            "qrow_pages": loaded_qrows,
            "fp32_tensors": loaded_fp32,
            "resident_bytes": self._prebound_compact_bytes + self._prebound_fp32_bytes,
        }

    def set_compact_cache_budget(self, cache_mb: float) -> None:
        """Resize the compact-page LRU and synchronously enforce the new byte ceiling."""

        budget = max(0, int(float(cache_mb) * 1e6))
        if budget < self.fp32_aux_cache_bytes:
            raise MemoryError(
                "provider cache budget cannot evict pinned FP32 auxiliaries "
                f"({budget} < {self.fp32_aux_cache_bytes} bytes)"
            )
        self.compact_cache_budget = budget
        while (
            self.compact_cache
            and self.compact_cache_bytes + self.fp32_aux_cache_bytes > self.compact_cache_budget
        ):
            _, evicted = self.compact_cache.popitem(last=False)
            self.compact_cache_bytes -= evicted.compact_bytes

    def prepare_fully_resident(self) -> None:
        """Materialize every physical component allocation in its device cache.

        This is a hard residency operation for native execution, not an advisory warmup.
        The complete verified footprint must fit the configured provider budget before any
        missing qrow page is loaded, FP32 auxiliaries must already use the non-evictable pin
        contract, and the resulting cache topology is checked after materialization.
        """

        qrow_names = tuple(
            dict.fromkeys(
                self._physical_key(name)
                for name, block in self.blocks.items()
                if isinstance(block, dict) and self._resolve(name).get("kind") == "qrow"
            )
        )
        fp32_names = tuple(
            dict.fromkeys(
                self._physical_key(name)
                for name, block in self.blocks.items()
                if isinstance(block, dict) and self._resolve(name).get("kind") == "fp32"
            )
        )
        qrow_required = sum(
            int(self.blocks[name].get("w_len", 0)) + int(self.blocks[name].get("s_len", 0))
            for name in qrow_names
        )
        fp32_required = sum(int(self.blocks[name].get("e_len", 0)) for name in fp32_names)
        required = qrow_required + fp32_required
        if required > self.compact_cache_budget:
            raise MemoryError(
                "complete component residency exceeds the provider cache budget "
                f"({required} > {self.compact_cache_budget} bytes)"
            )
        prebound_fp32 = getattr(self, "_prebound_fp32", {})
        missing_fp32 = set(fp32_names) - (
            set(self.fp32_aux_cache) | set(prebound_fp32)
        )
        if missing_fp32:
            raise RuntimeError(
                "complete component residency requires pinned FP32 auxiliaries; "
                f"missing={sorted(missing_fp32)!r}"
            )
        prebound_pages = getattr(self, "_prebound_pages", {})
        missing_qrows = set(qrow_names) - set(self.compact_cache) - set(prebound_pages)
        missing_bytes = sum(
            int(self.blocks[name].get("w_len", 0)) + int(self.blocks[name].get("s_len", 0))
            for name in missing_qrows
        )
        device = torch.device(self.device)
        if device.type == "cuda" and missing_bytes:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if missing_bytes > int(free_bytes):
                raise MemoryError(
                    "complete component residency exceeds currently free device memory "
                    f"({missing_bytes} > {int(free_bytes)} bytes)"
                )
        for name in qrow_names:
            self.compact_page(name)
        resident_qrows = set(qrow_names) <= (set(self.compact_cache) | set(prebound_pages))
        resident_fp32 = set(fp32_names) <= (
            set(self.fp32_aux_cache) | set(prebound_fp32)
        )
        if (
            not resident_qrows
            or not resident_fp32
            or self.compact_cache_bytes + getattr(self, "_prebound_compact_bytes", 0)
            != qrow_required
            or self.fp32_aux_cache_bytes + getattr(self, "_prebound_fp32_bytes", 0)
            != fp32_required
        ):
            raise RuntimeError(
                "complete component residency postcondition failed: "
                f"qrows={resident_qrows} fp32={resident_fp32} "
                f"resident={self.compact_cache_bytes + self.fp32_aux_cache_bytes} "
                f"required={required}"
            )

    def _compact_cache_key(self, name: str) -> str:
        key = name
        seen: set[str] = set()
        while "alias" in self.blocks[key]:
            if key in seen:
                raise ValueError(f"cyclic QStore alias at {name!r}")
            seen.add(key)
            key = str(self.blocks[key]["alias"])
        return key

    def _physical_key(self, name: str) -> str:
        """Resolve a logical block name to one physical allocation."""

        key = str(name)
        seen: set[str] = set()
        while True:
            if key in seen:
                raise ValueError(f"cyclic QStore alias at {name!r}")
            seen.add(key)
            block = self.blocks.get(key)
            if not isinstance(block, dict):
                raise KeyError(f"QStore has no block {key!r}")
            alias = block.get("alias")
            if alias is None:
                return key
            key = str(alias)

    def compact_page(self, name: str) -> CompactQRowPage:
        cache_key = self._compact_cache_key(name)
        prebound = getattr(self, "_prebound_pages", {}).get(cache_key)
        if prebound is not None:
            self.stats.cache_hits += 1
            return prebound
        cached = self.compact_cache.get(cache_key)
        if cached is not None:
            self.compact_cache.move_to_end(cache_key)
            self.stats.cache_hits += 1
            return cached
        page = load_compact_qrow_page(self, cache_key)
        self.stats.page_loads += 1
        if page.codes.device.type == "cuda":
            self.stats.compact_h2d_bytes += page.compact_bytes
        auxiliary_bytes = int(getattr(self, "fp32_aux_cache_bytes", 0))
        transient_peak = self.compact_cache_bytes + auxiliary_bytes + page.compact_bytes
        available = self.compact_cache_budget - auxiliary_bytes
        if available > 0 and page.compact_bytes <= available:
            self.compact_cache[cache_key] = page
            self.compact_cache.move_to_end(cache_key)
            self.compact_cache_bytes += page.compact_bytes
            while self.compact_cache_bytes > available:
                _, evicted = self.compact_cache.popitem(last=False)
                self.compact_cache_bytes -= evicted.compact_bytes
        self.stats.peak_compact_resident_bytes = max(
            self.stats.peak_compact_resident_bytes,
            transient_peak,
            self.compact_cache_bytes + auxiliary_bytes,
        )
        return page

    def prepare_resident_exact_head(
        self,
        name: str = "lm_head",
        *,
        max_resident_bytes: int,
    ) -> torch.Tensor:
        """Admit and materialize one exact FP32 qrow head on the target device.

        The resident tensor is scored in the same row blocks as the streamed established
        contract. Admission is checked from the verified manifest before allocation and from
        the resulting tensor afterward. There is no silent fallback from this explicit method.
        """

        if max_resident_bytes <= 0:
            raise ValueError("resident exact-head budget must be positive")
        physical = self._physical_key(name)
        existing = self.resident_exact_heads.get(physical)
        if existing is not None:
            if _tensor_bytes(existing) > max_resident_bytes:
                raise MemoryError("existing resident exact head exceeds the requested budget")
            return existing
        block = self._resolve(name)
        if block.get("kind") != "qrow":
            raise ValueError(f"resident exact head {name!r} is not a qrow block")
        rows, columns = (int(value) for value in block["shape"])
        required = rows * columns * torch.empty((), dtype=torch.float32).element_size()
        if required > max_resident_bytes:
            raise MemoryError(
                "resident exact FP32 head exceeds its admission budget "
                f"({required} > {max_resident_bytes} bytes)"
            )
        device = torch.device(self.device)
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if required > int(free_bytes):
                raise MemoryError(
                    "resident exact FP32 head exceeds currently free device memory "
                    f"({required} > {int(free_bytes)} bytes)"
                )
        page = self.compact_page(name)
        resident = (page.codes.float() * page.scales[:, None]).contiguous()
        resident_bytes = _tensor_bytes(resident)
        if resident_bytes != required:
            raise RuntimeError(
                "resident exact FP32 head allocation disagrees with manifest admission "
                f"({resident_bytes} != {required} bytes)"
            )
        self.resident_exact_heads[physical] = resident
        self.resident_exact_head_budget_bytes = int(max_resident_bytes)
        self.resident_exact_head_bytes += resident_bytes
        self.stats.resident_exact_head_loads += 1
        return resident

    def resident_exact_head_fp32(self, name: str = "lm_head") -> torch.Tensor | None:
        return self.resident_exact_heads.get(self._physical_key(name))

    def _matmul_page(
        self,
        page: CompactQRowPage,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        output, implementation = fused_qrow_matmul(
            page,
            activations,
            require_triton=self.require_triton,
            block_m=self.stable_block_m,
        )
        self.stats.projection_calls += 1
        self.stats.compact_logical_bytes += page.compact_bytes
        if implementation.startswith("triton"):
            self.stats.triton_projection_calls += 1
            self.stats.expanded_fp32_bytes_avoided += page.expanded_fp32_bytes
            self.stats.expanded_compute_bytes_avoided += page.expanded_compute_bytes(
                self.compute_dtype
            )
        else:
            self.stats.reference_projection_calls += 1
        return output

    def _fused_swiglu_pages(
        self,
        gate_page: CompactQRowPage,
        up_page: CompactQRowPage,
        activations: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        output, implementation = fused_qrow_swiglu(
            gate_page,
            up_page,
            activations,
            require_triton=self.require_triton,
            block_m=self.stable_block_m,
        )
        pages = (gate_page, up_page)
        self.stats.projection_calls += 2
        self.stats.paired_swiglu_calls += 1
        self.stats.compact_logical_bytes += sum(page.compact_bytes for page in pages)
        if implementation.startswith("triton"):
            self.stats.triton_projection_calls += 2
            self.stats.triton_paired_swiglu_calls += 1
            self.stats.expanded_fp32_bytes_avoided += sum(
                page.expanded_fp32_bytes for page in pages
            )
            self.stats.expanded_compute_bytes_avoided += sum(
                page.expanded_compute_bytes(self.compute_dtype) for page in pages
            )
        else:
            self.stats.reference_projection_calls += 2
            self.stats.reference_paired_swiglu_calls += 1
        return output, implementation

    def fused_swiglu(
        self,
        gate_name: str,
        up_name: str,
        activations: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        return self._fused_swiglu_pages(
            self.compact_page(gate_name),
            self.compact_page(up_name),
            activations,
        )

    def matmul(self, name: str, x: torch.Tensor) -> torch.Tensor:
        block = self._resolve(name)
        if block.get("kind") != "qrow":
            return super().matmul(name, x)
        return self._matmul_page(self.compact_page(name), x)

    def _embed_page_rows(
        self,
        page: CompactQRowPage,
        ids: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        index = torch.as_tensor(ids, device=page.codes.device, dtype=torch.long)
        flat = index.reshape(-1)
        codes = page.codes.index_select(0, flat).float()
        scales = page.scales.index_select(0, flat)
        rows = codes * scales[:, None]
        if self.compute_dtype is not torch.float32:
            rows = rows.to(self.compute_dtype)
        return rows.reshape(*index.shape, page.in_features)

    def embed_rows(self, name: str, ids: np.ndarray | torch.Tensor) -> torch.Tensor:
        return self._embed_page_rows(self.compact_page(name), ids)

    def selected_rows_fp32(
        self,
        name: str,
        ids: np.ndarray | torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        """Load only named qrow rows and dequantize them to FP32 on the target device.

        The dense vocabulary head intentionally accumulates in FP32.  Reusing
        :meth:`embed_rows` here would narrow the selected weights to BF16/FP16 and
        silently create a different numerical contract.  This path also avoids loading
        the complete compact vocabulary page: transfer and temporary storage are
        proportional to ``len(ids) * hidden_size``.
        """

        block = self._resolve(name)
        if block.get("kind") != "qrow":
            raise ValueError(f"{name} is not a qrow block")
        indices = torch.as_tensor(ids, dtype=torch.long).detach().cpu().numpy()
        if indices.ndim != 1 or not indices.size:
            raise ValueError("selected qrow IDs must be a non-empty one-dimensional array")
        out_features, in_features = (int(value) for value in block["shape"])
        if int(indices.min()) < 0 or int(indices.max()) >= out_features:
            raise IndexError(f"selected qrow ID outside [0, {out_features})")

        weight_offset = int(block["w_off"])
        weight_rows = np.asarray(
            self.w[weight_offset : weight_offset + out_features * in_features],
            dtype=np.int8,
        ).reshape(out_features, in_features)
        codes_np = np.asarray(weight_rows[indices], dtype=np.int8)
        scale_offset = int(block["s_off"]) // 4
        scales_np = np.asarray(self.s[scale_offset + indices], dtype=np.float32)
        codes = torch.from_numpy(codes_np.copy())
        scales = torch.from_numpy(scales_np.copy())
        device = torch.device(self.device)
        compact_bytes = int(codes.numel() + scales.numel() * scales.element_size())
        if device.type != "cpu":
            codes = codes.to(device)
            scales = scales.to(device)
            self.stats.compact_h2d_bytes += compact_bytes
        self.stats.selected_head_calls += 1
        self.stats.selected_head_rows += int(indices.size)
        self.stats.selected_head_compact_bytes += compact_bytes
        self.stats.compact_logical_bytes += compact_bytes
        self.stats.peak_compact_resident_bytes = max(
            self.stats.peak_compact_resident_bytes,
            self.compact_cache_bytes
            + int(getattr(self, "fp32_aux_cache_bytes", 0))
            + compact_bytes,
        )
        return codes.float() * scales[:, None]

    def prepare_resident_arena(
        self,
        *,
        qrow_names: Sequence[str],
        fp32_names: Sequence[str],
        max_resident_bytes: int | None = None,
    ) -> DenseQStoreResidentArena:
        """Pin one immutable resource arena for any number of graph templates."""

        logical_qrows = tuple(dict.fromkeys((*map(str, qrow_names), "lm_head")))
        logical_fp32 = tuple(dict.fromkeys(map(str, fp32_names)))
        if not logical_qrows:
            raise ValueError("resident QStore arena requires qrow resources")
        if max_resident_bytes is not None and int(max_resident_bytes) <= 0:
            raise ValueError("resident QStore arena budget must be positive")
        budget = None if max_resident_bytes is None else int(max_resident_bytes)
        qrow_aliases = {name: self._physical_key(name) for name in logical_qrows}
        fp32_aliases = {name: self._physical_key(name) for name in logical_fp32}
        qrow_physical = tuple(dict.fromkeys(qrow_aliases.values()))
        fp32_physical = tuple(dict.fromkeys(fp32_aliases.values()))
        estimated = 0
        for name in qrow_physical:
            block = self.blocks[name]
            if block.get("kind") != "qrow":
                raise ValueError(f"resident arena qrow resource {name!r} is not qrow")
            out_features, in_features = (int(value) for value in block["shape"])
            estimated += out_features * in_features + out_features * 4
        for name in fp32_physical:
            block = self.blocks[name]
            if block.get("kind") != "fp32":
                raise ValueError(f"resident arena FP32 resource {name!r} is not fp32")
            estimated += int(np.prod(block["shape"])) * 4
        if budget is not None and estimated > budget:
            raise MemoryError(
                f"resident QStore arena exceeds its budget ({estimated} > {budget} bytes)"
            )
        device = torch.device(self.device)
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if estimated > int(free_bytes):
                raise MemoryError(
                    "resident QStore arena exceeds currently free device memory "
                    f"({estimated} > {int(free_bytes)} bytes)"
                )
        qrow_pages = {name: load_compact_qrow_page(self, name) for name in qrow_physical}
        fp32_tensors = {
            name: super(DenseQStore, self).fp32(name).contiguous() for name in fp32_physical
        }
        arena = DenseQStoreResidentArena(
            self,
            qrow_pages=qrow_pages,
            qrow_aliases=qrow_aliases,
            fp32_tensors=fp32_tensors,
            fp32_aliases=fp32_aliases,
            budget_bytes=budget,
            estimated_resident_bytes=estimated,
        )
        if budget is not None and arena.resident_bytes > budget:
            arena.close()
            raise MemoryError(
                "resident QStore arena exceeded its budget after allocation "
                f"({arena.resident_bytes} > {budget} bytes)"
            )
        arena.verify_stable_addresses()
        return arena

    def prepare_capture_bindings(
        self,
        *,
        qrow_names: Sequence[str],
        fp32_names: Sequence[str],
        selected_head_rows: Sequence[int],
        max_resident_bytes: int | None = None,
        rebindable_head: bool = False,
    ) -> DenseQStoreCaptureBindings:
        """Pin one selected-head graph's immutable resource set.

        The regular LRU remains untouched.  Capture bindings are separately owned and
        cannot be evicted while their executor is alive.  The byte budget is checked
        from manifest shapes before any device allocation, then checked again against
        the exact tensors.
        """

        logical_qrows = tuple(dict.fromkeys(str(name) for name in qrow_names))
        logical_fp32 = tuple(dict.fromkeys(str(name) for name in fp32_names))
        selected = tuple(int(value) for value in selected_head_rows)
        if not logical_qrows:
            raise ValueError("CUDA Graph capture requires at least one qrow resource")
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("captured selected-head rows must be non-empty and unique")
        if max_resident_bytes is not None and int(max_resident_bytes) <= 0:
            raise ValueError("CUDA Graph residency budget must be positive")
        budget = None if max_resident_bytes is None else int(max_resident_bytes)

        if rebindable_head and "lm_head" not in logical_qrows:
            logical_qrows = (*logical_qrows, "lm_head")
        qrow_aliases = {name: self._physical_key(name) for name in logical_qrows}
        fp32_aliases = {name: self._physical_key(name) for name in logical_fp32}
        qrow_physical = tuple(dict.fromkeys(qrow_aliases.values()))
        fp32_physical = tuple(dict.fromkeys(fp32_aliases.values()))

        estimated = 0
        for name in qrow_physical:
            block = self.blocks[name]
            if block.get("kind") != "qrow":
                raise ValueError(f"capture qrow resource {name!r} is not qrow")
            out_features, in_features = (int(value) for value in block["shape"])
            estimated += out_features * in_features + out_features * 4
        for name in fp32_physical:
            block = self.blocks[name]
            if block.get("kind") != "fp32":
                raise ValueError(f"capture FP32 resource {name!r} is not fp32")
            estimated += int(np.prod(block["shape"])) * 4

        _, head = self._physical_key("lm_head"), self._resolve("lm_head")
        if head.get("kind") != "qrow":
            raise ValueError("captured selected head requires a qrow lm_head")
        head_rows, head_width = (int(value) for value in head["shape"])
        if min(selected) < 0 or max(selected) >= head_rows:
            raise ValueError(f"selected head row must be inside [0, {head_rows})")
        if not rebindable_head:
            estimated += len(selected) * head_width * 4
        if budget is not None and estimated > budget:
            raise MemoryError(
                f"CUDA Graph capture residency exceeds its budget ({estimated} > {budget} bytes)"
            )
        device = torch.device(self.device)
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if estimated > int(free_bytes):
                raise MemoryError(
                    "CUDA Graph capture residency exceeds currently free device memory "
                    f"({estimated} > {int(free_bytes)} bytes)"
                )

        qrow_pages = {name: load_compact_qrow_page(self, name) for name in qrow_physical}
        fp32_tensors = {
            name: super(DenseQStore, self).fp32(name).contiguous() for name in fp32_physical
        }
        selected_weights = (
            None
            if rebindable_head
            else self.selected_rows_fp32("lm_head", selected).contiguous()
        )
        bindings = DenseQStoreCaptureBindings(
            self,
            qrow_pages=qrow_pages,
            qrow_aliases=qrow_aliases,
            fp32_tensors=fp32_tensors,
            fp32_aliases=fp32_aliases,
            selected_head_rows=selected,
            selected_head_weights=selected_weights,
            rebindable_head=rebindable_head,
            budget_bytes=budget,
            estimated_resident_bytes=estimated,
        )
        if budget is not None and bindings.resident_bytes > budget:
            bindings.close()
            raise MemoryError(
                "CUDA Graph capture residency exceeded its budget after allocation "
                f"({bindings.resident_bytes} > {budget} bytes)"
            )
        return bindings

    def clear_compact_cache(self) -> None:
        self.compact_cache.clear()
        self.compact_cache_bytes = 0

    def close(self) -> None:
        self.clear_compact_cache()
        self._prebound_pages.clear()
        self._prebound_compact_bytes = 0
        self.fp32_aux_cache.clear()
        self.fp32_aux_cache_bytes = 0
        self._prebound_fp32.clear()
        self._prebound_fp32_bytes = 0
        self.resident_exact_heads.clear()
        self.resident_exact_head_bytes = 0
        super().close()
