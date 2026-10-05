"""Experimental dense CUDA runtime over compact int8 QStore pages.

This backend promotes the validated dense QStore building blocks into ``mrun``:

* compact int8 row codes and fp32 scales are streamed or byte-LRU cached on CUDA;
* W8A16 projections dequantize inside a fixed-tile Triton matmul;
* RMSNorm and causal GQA default to the established batched reduction contract;
* K/V state is persistent and provisional blocks are committed transactionally; and
* the vocabulary head defaults to the established exact streamed QStore implementation.

Fixed-row Triton reductions remain off by default because they do not reproduce the
established Torch trajectory. The named ``row-stable-triton-v1`` contract passed
119,880/119,880 B1-derived tokens and crossed 100x aggregate scaling at B=128, while
matching 487/544 Torch-default trace tokens. The backend is explicit-only and is an
inference/forward runtime, not an autograd or activation-intervention backend.

An experimental fused top-two vocabulary search with compact FP32 candidate reranking
is separately opt-in. Synthetic gates restored 32,768/32,768 FP32 QStore winners. A
bounded real-Qwen integration then preserved 48/48 tested tokens at 12.86x head and
1.47x complete-forward speedup after tied-page cache deduplication. This does not prove
that top two covers every hidden state or support full-logit/training contracts.
"""

from __future__ import annotations

import json
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from ..models import load_tokenizer, resolve_model, store_name
from ..paths import stores_root as default_stores_root
from ._base_impl import BaseEngine
from .base import EngineCapabilities
from .kernels.batch_invariant import batch_invariant_matmul
from .kernels.composite_qstore import (
    ComponentGraphError,
    CompositeQStore,
    ContractQStoreView,
)
from .kernels.dense_qstore_cuda import (
    DenseQStore,
    DenseQStoreCaptureBindings,
    DenseQStoreResidentArena,
    fused_qrow_reranked_argmax,
    fused_qrow_swiglu,
    fused_residual_rms_norm,
    segmented_decode_attention,
    stable_attention,
    stable_rms_norm,
    stable_rms_norm_reference,
)
from .kernels.lexical import (
    LexicalComponent,
    LexicalQStoreView,
    load_separated_lexical,
)

__all__ = [
    "CommitStats",
    "DenseForwardResult",
    "DenseSelectedForwardResult",
    "DenseSelectedLastCUDAGraphExecutor",
    "DenseQStoreCudaEngine",
    "DenseSourceCudaInt8CompactHeadEngine",
    "DenseSourceCudaInt8Engine",
    "DenseQStoreKVCache",
    "DenseQStoreTarget",
    "InstallStats",
    "KVDelta",
]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _same_torch_device(actual: torch.device | str, expected: torch.device | str) -> bool:
    """Compare concrete devices without accepting an ambiguous ordinal."""

    actual_device = torch.device(actual)
    expected_device = torch.device(expected)
    return actual_device == expected_device


def _concrete_cuda_device(device: torch.device | str) -> torch.device:
    requested = torch.device(device)
    if requested.type != "cuda":
        raise ValueError("dense-qstore-cuda requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("dense-qstore-cuda requires an available CUDA device")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is unavailable: {index}")
    return torch.device("cuda", index)


def _find_linked_store(root: Path, model_name: str, extension_id: str) -> Path:
    """Find one explicit linked image for a dense CUDA engine."""

    matches: list[Path] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(manifest, Mapping):
            continue
        linked_image = manifest.get("linked_image")
        if (
            str(manifest.get("model_name", "")) == model_name
            and isinstance(linked_image, Mapping)
            and str(linked_image.get("extension_id", "")) == extension_id
        ):
            matches.append(manifest_path.parent.resolve())
    if not matches:
        raise FileNotFoundError(
            f"no linked dense QStore image for model {model_name!r} and extension "
            f"{extension_id!r} under {root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple linked dense QStore images for model {model_name!r} and extension "
            f"{extension_id!r}: {matches}"
        )
    return matches[0]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


_ESTABLISHED_DECODE_ATTENTION_MODE = "established"
_SEGMENTED_FLASH_GQA_DECODE_MODE = "segmented-flash-gqa-decode-v1"
_ESTABLISHED_BODY_FUSION_MODE = "established"
_RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE = "residual-rms-swiglu-v1"


def _decode_attention_config(mode: str, sequence_tile: int) -> tuple[str, int]:
    if not isinstance(mode, str):
        raise TypeError("decode_attention_mode must be a string")
    if mode not in {_ESTABLISHED_DECODE_ATTENTION_MODE, _SEGMENTED_FLASH_GQA_DECODE_MODE}:
        raise ValueError(
            "decode_attention_mode must be 'established' or 'segmented-flash-gqa-decode-v1'"
        )
    if isinstance(sequence_tile, bool) or not isinstance(sequence_tile, int):
        raise TypeError("decode_attention_tile must be an integer")
    if sequence_tile < 16 or sequence_tile > 256 or sequence_tile & (sequence_tile - 1):
        raise ValueError("decode_attention_tile must be a power of two in [16, 256]")
    return mode, int(sequence_tile)


def _decode_numerical_contract(base: str, mode: str) -> str:
    if mode == _ESTABLISHED_DECODE_ATTENTION_MODE:
        return base
    return f"{base}+{_SEGMENTED_FLASH_GQA_DECODE_MODE}"


def _body_fusion_config(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError("body_fusion_mode must be a string")
    if mode not in {_ESTABLISHED_BODY_FUSION_MODE, _RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE}:
        raise ValueError("body_fusion_mode must be 'established' or 'residual-rms-swiglu-v1'")
    return mode


def _body_numerical_contract(base: str, mode: str) -> str:
    if mode == _ESTABLISHED_BODY_FUSION_MODE:
        return base
    return f"{base}+{_RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE}"


def _execution_numerical_contract(base: str, decode_mode: str, body_mode: str) -> str:
    return _body_numerical_contract(
        _decode_numerical_contract(base, decode_mode),
        body_mode,
    )


@dataclass(frozen=True)
class KVDelta:
    """Uncommitted K/V rows produced by one target forward block."""

    parent_epoch: int
    parent_lengths: tuple[int, ...]
    cache_id: str
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    token_count: int

    @property
    def batch_size(self) -> int:
        return len(self.parent_lengths)

    @property
    def byte_count(self) -> int:
        return sum(_tensor_bytes(tensor) for tensor in (*self.keys, *self.values))

    def select(self, indices: Sequence[int]) -> KVDelta:
        chosen = tuple(int(index) for index in indices)
        if not chosen:
            raise ValueError("delta selection cannot be empty")
        if min(chosen) < 0 or max(chosen) >= self.batch_size:
            raise IndexError("delta selection outside batch")
        device_indices = torch.as_tensor(chosen, device=self.keys[0].device)
        return KVDelta(
            parent_epoch=self.parent_epoch,
            parent_lengths=tuple(self.parent_lengths[index] for index in chosen),
            cache_id=self.cache_id,
            keys=tuple(tensor.index_select(0, device_indices).clone() for tensor in self.keys),
            values=tuple(tensor.index_select(0, device_indices).clone() for tensor in self.values),
            token_count=self.token_count,
        )


@dataclass(frozen=True)
class CommitStats:
    accepted_counts: tuple[int, ...]
    kv_write_bytes: int
    epoch_before: int
    epoch_after: int


@dataclass(frozen=True)
class InstallStats:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    installed_lengths: tuple[int, ...]
    kv_copy_bytes: int
    epoch_before: int
    epoch_after: int


class DenseQStoreKVCache:
    """Preallocated request-isolated GQA cache with epoch-checked commits."""

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        dimensions = (num_layers, batch_size, max_seq_len, num_kv_heads, head_dim)
        if min(int(value) for value in dimensions) <= 0:
            raise ValueError("cache dimensions must be positive")
        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.max_seq_len = int(max_seq_len)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        shape = (self.batch_size, self.max_seq_len, self.num_kv_heads, self.head_dim)
        with torch.inference_mode(False):
            self.keys = [
                torch.zeros(shape, device=device, dtype=dtype) for _ in range(self.num_layers)
            ]
            self.values = [
                torch.zeros(shape, device=device, dtype=dtype) for _ in range(self.num_layers)
            ]
        self.device = self.keys[0].device
        self.lengths = np.zeros(self.batch_size, dtype=np.int64)
        self.epoch = 0
        self.cache_id = uuid4().hex
        self._lock = threading.RLock()

    @classmethod
    def for_store(
        cls,
        store: DenseQStore,
        *,
        batch_size: int,
        max_seq_len: int,
    ) -> DenseQStoreKVCache:
        cfg = store.cfg
        heads = int(cfg["num_attention_heads"])
        return cls(
            num_layers=int(cfg["num_hidden_layers"]),
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=int(cfg.get("num_key_value_heads", heads)),
            head_dim=int(cfg.get("head_dim", int(cfg["hidden_size"]) // heads)),
            device=store.device,
            dtype=store.compute_dtype,
        )

    @property
    def committed_bytes(self) -> int:
        rows = int(self.lengths.sum()) * self.num_layers * self.num_kv_heads * self.head_dim * 2
        return rows * torch.empty((), dtype=self.dtype).element_size()

    @property
    def allocated_bytes(self) -> int:
        return sum(_tensor_bytes(tensor) for tensor in (*self.keys, *self.values))

    def clone(self) -> DenseQStoreKVCache:
        copied = self.select(range(self.batch_size))
        copied.epoch = self.epoch
        return copied

    def select(self, indices: Sequence[int]) -> DenseQStoreKVCache:
        chosen = tuple(int(index) for index in indices)
        if not chosen:
            raise ValueError("cache selection cannot be empty")
        if min(chosen) < 0 or max(chosen) >= self.batch_size:
            raise IndexError("cache selection outside batch")
        selected = DenseQStoreKVCache(
            num_layers=self.num_layers,
            batch_size=len(chosen),
            max_seq_len=self.max_seq_len,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        selected.lengths = self.lengths[list(chosen)].copy()
        selected.epoch = self.epoch
        device_indices = torch.as_tensor(chosen, device=self.device)
        for layer in range(self.num_layers):
            selected.keys[layer].copy_(self.keys[layer].index_select(0, device_indices))
            selected.values[layer].copy_(self.values[layer].index_select(0, device_indices))
        return selected

    def install_requests(
        self,
        source: DenseQStoreKVCache,
        *,
        source_indices: Sequence[int],
        target_indices: Sequence[int],
    ) -> InstallStats:
        """Copy committed prefixes into reusable request slots.

        Source indices may repeat. Target indices must be unique. Stale tails are not
        cleared because every read is masked by the installed committed length.
        """

        sources = tuple(int(index) for index in source_indices)
        targets = tuple(int(index) for index in target_indices)
        if not sources or len(sources) != len(targets):
            raise ValueError("source and target selections must have equal non-zero length")
        if len(set(targets)) != len(targets):
            raise ValueError("target indices must be unique")
        if min(sources) < 0 or max(sources) >= source.batch_size:
            raise IndexError("source selection outside batch")
        if min(targets) < 0 or max(targets) >= self.batch_size:
            raise IndexError("target selection outside batch")
        if source is self and set(sources).intersection(targets):
            raise ValueError("overlapping in-place installs require a separate template cache")
        for field in ("num_layers", "num_kv_heads", "head_dim", "dtype", "device"):
            if getattr(source, field) != getattr(self, field):
                raise ValueError(f"source cache {field} does not match target cache")

        installed_lengths = tuple(int(source.lengths[index]) for index in sources)
        if any(length > self.max_seq_len for length in installed_lengths):
            raise OverflowError("source prefix exceeds target cache capacity")
        groups: dict[int, list[tuple[int, int]]] = {}
        for source_index, target_index, length in zip(
            sources, targets, installed_lengths, strict=True
        ):
            if length:
                groups.setdefault(length, []).append((source_index, target_index))

        for length, pairs in groups.items():
            source_rows = torch.as_tensor(
                [pair[0] for pair in pairs], device=self.device, dtype=torch.long
            )
            target_rows = torch.as_tensor(
                [pair[1] for pair in pairs], device=self.device, dtype=torch.long
            )
            for layer in range(self.num_layers):
                keys = source.keys[layer].index_select(0, source_rows)[:, :length]
                values = source.values[layer].index_select(0, source_rows)[:, :length]
                self.keys[layer][:, :length].index_copy_(0, target_rows, keys)
                self.values[layer][:, :length].index_copy_(0, target_rows, values)

        self.lengths[list(targets)] = np.asarray(installed_lengths, dtype=np.int64)
        copied_rows = (
            sum(installed_lengths) * self.num_layers * self.num_kv_heads * self.head_dim * 2
        )
        epoch_before = self.epoch
        self.epoch += 1
        return InstallStats(
            source_indices=sources,
            target_indices=targets,
            installed_lengths=installed_lengths,
            kv_copy_bytes=copied_rows * torch.empty((), dtype=self.dtype).element_size(),
            epoch_before=epoch_before,
            epoch_after=self.epoch,
        )

    def _validate_delta(self, delta: KVDelta) -> None:
        if delta.cache_id != self.cache_id:
            raise RuntimeError("KV delta belongs to a different dense cache")
        if delta.parent_epoch != self.epoch:
            raise RuntimeError(
                f"stale KV delta epoch {delta.parent_epoch}; cache is at epoch {self.epoch}"
            )
        if delta.parent_lengths != tuple(int(value) for value in self.lengths):
            raise RuntimeError("KV delta parent lengths do not match committed cache")
        if delta.batch_size != self.batch_size:
            raise ValueError("KV delta batch does not match cache")
        if len(delta.keys) != self.num_layers or len(delta.values) != self.num_layers:
            raise ValueError("KV delta layer count does not match cache")
        expected = (self.batch_size, delta.token_count, self.num_kv_heads, self.head_dim)
        for key, value in zip(delta.keys, delta.values, strict=True):
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(f"KV delta tensor shape must be {expected}")
            if not _same_torch_device(key.device, self.device) or not _same_torch_device(
                value.device, self.device
            ):
                raise ValueError("KV delta device does not match cache")
            if key.dtype != self.dtype or value.dtype != self.dtype:
                raise ValueError("KV delta dtype does not match cache")

    def _commit_unlocked(self, delta: KVDelta, accepted_counts: Sequence[int]) -> CommitStats:

        self._validate_delta(delta)
        counts = tuple(int(value) for value in accepted_counts)
        if len(counts) != self.batch_size:
            raise ValueError("accepted count must be supplied for every request")
        if any(value < 0 or value > delta.token_count for value in counts):
            raise ValueError("accepted count outside provisional block")
        if any(
            int(self.lengths[index]) + count > self.max_seq_len
            for index, count in enumerate(counts)
        ):
            raise OverflowError("KV cache capacity exceeded")

        epoch_before = self.epoch
        if len(set(counts)) == 1 and counts[0] > 0:
            count = counts[0]
            request_rows = torch.arange(self.batch_size, device=self.device)[:, None]
            positions = torch.as_tensor(self.lengths, device=self.device)[:, None]
            positions = positions + torch.arange(count, device=self.device)[None, :]
            for layer in range(self.num_layers):
                self.keys[layer][request_rows, positions] = delta.keys[layer][:, :count]
                self.values[layer][request_rows, positions] = delta.values[layer][:, :count]
        else:
            for layer in range(self.num_layers):
                for request, count in enumerate(counts):
                    if not count:
                        continue
                    start = int(self.lengths[request])
                    end = start + count
                    self.keys[layer][request, start:end].copy_(delta.keys[layer][request, :count])
                    self.values[layer][request, start:end].copy_(
                        delta.values[layer][request, :count]
                    )

        self.lengths += np.asarray(counts, dtype=np.int64)
        self.epoch += 1
        rows = sum(counts) * self.num_layers * self.num_kv_heads * self.head_dim * 2
        return CommitStats(
            accepted_counts=counts,
            kv_write_bytes=rows * torch.empty((), dtype=self.dtype).element_size(),
            epoch_before=epoch_before,
            epoch_after=self.epoch,
        )

    def commit(self, delta: KVDelta, accepted_counts: Sequence[int]) -> CommitStats:
        """Atomically validate and commit an accepted prefix for every request row."""

        with self._lock:
            return self._commit_unlocked(delta, accepted_counts)


@dataclass(frozen=True)
class DenseForwardResult:
    top1: torch.Tensor
    delta: KVDelta
    hidden: torch.Tensor
    kv_read_bytes: int
    kv_delta_bytes: int
    wall_s: float
    logits: torch.Tensor | None = None
    phase_wall_s: dict[str, float] | None = None


@dataclass(frozen=True)
class DenseSelectedForwardResult:
    """Body result for a selected final-position head; no global top-1 is implied."""

    selected_logits: torch.Tensor
    delta: KVDelta | None
    hidden: torch.Tensor
    kv_read_bytes: int
    kv_delta_bytes: int
    wall_s: float
    phase_wall_s: dict[str, float] | None = None


class DenseQStoreTarget:
    """Stateful dense Qwen/Llama target over compact QStore projections."""

    def __init__(
        self,
        store: DenseQStore | ContractQStoreView,
        *,
        max_seq_len: int = 128,
        stable_reductions: bool = False,
        decode_attention_mode: str = _ESTABLISHED_DECODE_ATTENTION_MODE,
        decode_attention_tile: int = 64,
        body_fusion_mode: str = _ESTABLISHED_BODY_FUSION_MODE,
        experimental_reranked_head: bool = False,
        semantic_token_count: int | None = None,
        stable_head: bool = False,
    ) -> None:
        arch = str(store.man.get("arch", "qwen2"))
        if arch not in {"qwen2", "qwen3", "llama"}:
            raise ValueError("dense-qstore-cuda supports qwen2, qwen3, and llama stores")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.store = store
        self.cfg = store.cfg
        self.arch = arch
        self.max_seq_len = int(max_seq_len)
        self.device = torch.device(store.device)
        self.compute_dtype = store.compute_dtype
        self.stable_reductions = bool(stable_reductions)
        self.decode_attention_mode, self.decode_attention_tile = _decode_attention_config(
            decode_attention_mode,
            decode_attention_tile,
        )
        self.body_fusion_mode = _body_fusion_config(body_fusion_mode)
        if (
            self.body_fusion_mode == _RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE
            and self.compute_dtype is not torch.bfloat16
        ):
            raise ValueError("residual-rms-swiglu-v1 requires BF16 compute")
        if self.body_fusion_mode == _RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE and not callable(
            getattr(store, "compact_page", None)
        ):
            raise RuntimeError("residual-rms-swiglu-v1 requires compact qrow page access")
        self.segmented_decode_attention_calls = 0
        self.segmented_decode_prefill_fallback_calls = 0
        self.fused_residual_rms_calls = 0
        self.fused_gate_up_swiglu_calls = 0
        self.experimental_reranked_head = bool(experimental_reranked_head)
        # Batch-invariant exact head: fixed-bracket FP32 over the compact qrow codes, so a row's
        # logits do not depend on how many rows share the call (row-stable-triton-v2).
        self.stable_head = bool(stable_head)
        self.stable_head_calls = 0
        self.reduction_calls: dict[str, int] = {}
        self.token_domain_checks = 0
        self.trusted_generated_token_bypasses = 0
        self.resident_exact_head_calls = 0
        self.streamed_exact_head_calls = 0
        self.reranked_head_calls = 0
        self.reranked_head_working_bytes_peak = 0
        configured_vocab_rows = int(self.cfg.get("vocab_size", 0))
        semantic_token_count = (
            configured_vocab_rows if semantic_token_count is None else int(semantic_token_count)
        )
        if semantic_token_count <= 0 or semantic_token_count > configured_vocab_rows:
            raise ValueError(
                "semantic token count must be positive and no larger than configured "
                f"vocabulary rows ({semantic_token_count} > {configured_vocab_rows})"
            )
        self.semantic_token_count = semantic_token_count
        heads = int(self.cfg["num_attention_heads"])
        self.head_dim = int(self.cfg.get("head_dim", int(self.cfg["hidden_size"]) // heads))

    def empty_cache(self, batch_size: int) -> DenseQStoreKVCache:
        return DenseQStoreKVCache.for_store(
            self.store,
            batch_size=batch_size,
            max_seq_len=self.max_seq_len,
        )

    def selected_capture_resource_names(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the complete immutable body-resource set for selected scoring."""

        qrows = ["embed"]
        fp32 = ["norm.final"]
        for layer in range(int(self.cfg["num_hidden_layers"])):
            qrows.extend(
                f"L{layer}.{projection}"
                for projection in ("q", "k", "v", "o", "gate", "up", "down")
            )
            fp32.extend((f"L{layer}.ln1", f"L{layer}.ln2"))
            fp32.extend(
                name
                for name in (
                    f"L{layer}.q.bias",
                    f"L{layer}.k.bias",
                    f"L{layer}.v.bias",
                    f"L{layer}.o.bias",
                    f"L{layer}.q_norm",
                    f"L{layer}.k_norm",
                )
                if self.store.has(name)
            )
        missing_qrows = [name for name in qrows if not self.store.has(name)]
        missing_fp32 = [name for name in fp32 if not self.store.has(name)]
        if missing_qrows or missing_fp32:
            missing = ", ".join((*missing_qrows, *missing_fp32))
            raise RuntimeError(f"CUDA Graph capture is missing QStore resources: {missing}")
        return tuple(qrows), tuple(fp32)

    def _record(self, implementation: str) -> None:
        self.reduction_calls[implementation] = self.reduction_calls.get(implementation, 0) + 1

    def _norm_tensor(self, activations: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        eps = float(self.cfg["rms_norm_eps"])
        if self.stable_reductions:
            output, implementation = stable_rms_norm(
                activations,
                weight,
                eps,
                require_triton=self.store.require_triton,
            )
        else:
            output = stable_rms_norm_reference(activations, weight, eps)
            implementation = "torch-batched-rms"
        self._record(implementation)
        return output

    def _norm(
        self,
        activations: torch.Tensor,
        name: str,
        resources: DenseQStore | DenseQStoreCaptureBindings | None = None,
    ) -> torch.Tensor:
        bound = self.store if resources is None else resources
        return self._norm_tensor(activations, bound.fp32(name).float())

    def _fused_residual_norm(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
        name: str,
        resources: DenseQStore | DenseQStoreCaptureBindings,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rounded, normalized, implementation = fused_residual_rms_norm(
            residual,
            update,
            resources.fp32(name).float(),
            float(self.cfg["rms_norm_eps"]),
            require_triton=self.store.require_triton,
        )
        self.fused_residual_rms_calls += 1
        self._record(implementation)
        return rounded, normalized

    def _fused_gate_up_swiglu(
        self,
        normalized: torch.Tensor,
        layer: int,
        resources: DenseQStore | DenseQStoreCaptureBindings,
    ) -> torch.Tensor:
        fused_store_call = getattr(resources, "fused_swiglu", None)
        if callable(fused_store_call):
            output, implementation = fused_store_call(
                f"L{layer}.gate",
                f"L{layer}.up",
                normalized,
            )
        else:
            compact_page = getattr(resources, "compact_page", None)
            if not callable(compact_page):
                raise RuntimeError("body fusion lost compact gate/up page access")
            output, implementation = fused_qrow_swiglu(
                compact_page(f"L{layer}.gate"),
                compact_page(f"L{layer}.up"),
                normalized,
                require_triton=self.store.require_triton,
                block_m=self.store.stable_block_m,
            )
        self.fused_gate_up_swiglu_calls += 1
        self._record(implementation)
        return output

    def _rope(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        theta = float(self.cfg["rope_theta"])
        inverse_frequency = 1.0 / (
            theta
            ** (
                torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32)
                / self.head_dim
            )
        )
        frequencies = positions.float().unsqueeze(-1) * inverse_frequency
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = embedding.cos().unsqueeze(2)
        sine = embedding.sin().unsqueeze(2)
        half = self.head_dim // 2
        rotated = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
        return (values.float() * cosine + rotated.float() * sine).to(self.compute_dtype)

    def _attention(
        self,
        *,
        layer: int,
        query: torch.Tensor,
        key_new: torch.Tensor,
        value_new: torch.Tensor,
        cache: DenseQStoreKVCache,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        batch, token_count, _, _ = query.shape
        num_kv_heads = int(key_new.shape[2])
        max_past = int(cache.lengths.max(initial=0))
        cache_row_indices = getattr(cache, "row_indices", None)
        if self.decode_attention_mode == _SEGMENTED_FLASH_GQA_DECODE_MODE and token_count == 1:
            context, implementation = segmented_decode_attention(
                query,
                cache.keys[layer],
                cache.values[layer],
                key_new,
                value_new,
                lengths,
                cache_row_indices=cache_row_indices,
                require_triton=self.store.require_triton,
                sequence_tile=self.decode_attention_tile,
                max_committed=max_past,
                validate_lengths=False,
            )
            self.segmented_decode_attention_calls += 1
            self._record(implementation)
            element_size = torch.empty((), dtype=self.compute_dtype).element_size()
            read_rows = int(cache.lengths.sum())
            read_bytes = read_rows * num_kv_heads * self.head_dim * 2 * element_size
            return context.to(self.compute_dtype), read_bytes
        if cache_row_indices is not None:
            raise RuntimeError(
                "fixed-slot cache row mapping is supported only by one-token segmented decode"
            )
        if self.decode_attention_mode == _SEGMENTED_FLASH_GQA_DECODE_MODE:
            self.segmented_decode_prefill_fallback_calls += 1
        max_total = max_past + token_count
        shape = (batch, max_total, num_kv_heads, self.head_dim)
        key = torch.zeros(shape, device=self.device, dtype=self.compute_dtype)
        value = torch.zeros_like(key)
        if max_past:
            key[:, :max_past].copy_(cache.keys[layer][:, :max_past])
            value[:, :max_past].copy_(cache.values[layer][:, :max_past])
        requests = torch.arange(batch, device=self.device)[:, None]
        offsets = torch.arange(token_count, device=self.device)[None, :]
        positions = lengths[:, None] + offsets
        key[requests, positions] = key_new
        value[requests, positions] = value_new

        if self.stable_reductions:
            sequence_tile = max(16, 1 << (self.max_seq_len - 1).bit_length())
            context, implementation = stable_attention(
                query,
                key,
                value,
                lengths,
                require_triton=self.store.require_triton,
                sequence_tile=sequence_tile,
                validate_lengths=False,
            )
        else:
            repeat = int(query.shape[2]) // num_kv_heads
            if repeat > 1:
                key = key.repeat_interleave(repeat, dim=2)
                value = value.repeat_interleave(repeat, dim=2)
            offsets = torch.arange(token_count, device=self.device)[None, :, None]
            key_positions = torch.arange(max_total, device=self.device)[None, None, :]
            allowed = key_positions <= lengths[:, None, None] + offsets
            query_heads = query.transpose(1, 2).float()
            key_heads = key.transpose(1, 2).float()
            value_heads = value.transpose(1, 2).float()
            scores = torch.matmul(query_heads, key_heads.transpose(-1, -2))
            scores.mul_(self.head_dim**-0.5)
            scores.masked_fill_(~allowed[:, None, :, :], float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            context = torch.matmul(probabilities, value_heads).transpose(1, 2)
            implementation = "torch-batched-attention"
        self._record(implementation)
        element_size = torch.empty((), dtype=self.compute_dtype).element_size()
        read_rows = int(cache.lengths.sum())
        read_bytes = read_rows * num_kv_heads * self.head_dim * 2 * element_size
        return context.to(self.compute_dtype), read_bytes

    def _head(
        self,
        hidden: torch.Tensor,
        *,
        return_logits: bool,
        last_logits_only: bool = False,
        last_top1_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if last_logits_only and last_top1_only:
            raise ValueError("last-logit and last-top1 execution are mutually exclusive")
        if last_logits_only:
            if not return_logits:
                raise ValueError("last-only language-head execution requires returned logits")
            hidden = hidden[:, -1:, :]
        if last_top1_only:
            if return_logits:
                raise ValueError("last-top1 language-head execution cannot return logits")
            hidden = hidden[:, -1:, :]
        if self.experimental_reranked_head and not return_logits:
            page = self.store.compact_page("lm_head")
            if page.start_row != 0 or page.out_features < self.semantic_token_count:
                raise RuntimeError(
                    "compact head page does not cover the complete semantic vocabulary prefix"
                )
            top1, _, implementation, working_bytes = fused_qrow_reranked_argmax(
                page,
                hidden,
                semantic_row_count=self.semantic_token_count,
                require_triton=self.store.require_triton,
                block_m=self.store.stable_block_m,
            )
            self.reranked_head_calls += 1
            self.reranked_head_working_bytes_peak = max(
                self.reranked_head_working_bytes_peak,
                int(working_bytes),
            )
            self._record(implementation)
            return top1, None

        batch, token_count, _ = hidden.shape
        vocab_size = int(self.cfg["vocab_size"])
        output_vocab_size = self.semantic_token_count if last_logits_only else vocab_size
        if self.stable_head:
            page = self.store.compact_page("lm_head")
            if page.start_row != 0 or page.out_features < output_vocab_size:
                raise RuntimeError("stable head page does not cover the output vocabulary")
            self.stable_head_calls += 1
            all_logits = batch_invariant_matmul(hidden.float(), page.codes, page.scales)
            semantic = all_logits[..., : self.semantic_token_count]
            # torch.argmax returns the first maximal index, matching the streamed head's
            # strict-greater block merge.
            best_index = semantic.argmax(dim=-1)
            if not return_logits:
                return best_index, None
            return best_index, all_logits[..., :output_vocab_size].contiguous()
        best_value = torch.full(
            (batch, token_count),
            float("-inf"),
            device=self.device,
            dtype=torch.float32,
        )
        best_index = torch.zeros((batch, token_count), device=self.device, dtype=torch.long)
        logits = (
            torch.empty(
                (batch, token_count, output_vocab_size),
                device=self.device,
                dtype=torch.float32,
            )
            if return_logits
            else None
        )
        hidden_fp32 = hidden.float()
        resident_getter = getattr(self.store, "resident_exact_head_fp32", None)
        resident_head = resident_getter("lm_head") if callable(resident_getter) else None
        if resident_head is not None:
            self.resident_exact_head_calls += 1
            row_blocks = (
                (start, min(start + 8192, vocab_size), resident_head[start : start + 8192])
                for start in range(0, vocab_size, 8192)
            )
        else:
            self.streamed_exact_head_calls += 1
            row_blocks = self.store.row_blocks("lm_head")
        for start, end, weight in row_blocks:
            if last_logits_only and start >= self.semantic_token_count:
                break
            semantic_end = min(end, self.semantic_token_count)
            scored_weight = (
                weight[: semantic_end - start]
                if last_logits_only and semantic_end < end
                else weight
            )
            block_logits = hidden_fp32 @ scored_weight.float().T
            if start < semantic_end:
                semantic_logits = block_logits[..., : semantic_end - start]
                block_value, block_offset = semantic_logits.max(dim=-1)
                replace = block_value > best_value
                best_value = torch.where(replace, block_value, best_value)
                best_index = torch.where(replace, block_offset + start, best_index)
            if logits is not None:
                output_end = min(end, output_vocab_size)
                if start < output_end:
                    logits[:, :, start:output_end] = block_logits[..., : output_end - start]
        return best_index, logits

    def _selected_last_head(
        self,
        hidden: torch.Tensor,
        row_ids: Sequence[int] | torch.Tensor,
        resources: DenseQStore | DenseQStoreCaptureBindings | None = None,
    ) -> torch.Tensor:
        """Score only requested vocabulary rows for the final sequence position."""

        bound = self.store if resources is None else resources
        if (
            isinstance(row_ids, torch.Tensor)
            and isinstance(bound, DenseQStoreCaptureBindings)
            and bound.rebindable_head
        ):
            selected: Sequence[int] | torch.Tensor = row_ids
        else:
            selected_tuple = tuple(int(value) for value in row_ids)
            if not selected_tuple or len(selected_tuple) != len(set(selected_tuple)):
                raise ValueError("selected vocabulary rows must be non-empty and unique")
            if min(selected_tuple) < 0 or max(selected_tuple) >= self.semantic_token_count:
                raise ValueError(
                    f"selected vocabulary rows must be inside [0, {self.semantic_token_count})"
                )
            selected = selected_tuple
        weights = bound.selected_rows_fp32("lm_head", selected)
        scores = hidden[:, -1].float() @ weights.T
        return scores

    @torch.inference_mode()
    def _forward_impl(
        self,
        input_ids: np.ndarray | torch.Tensor,
        cache: DenseQStoreKVCache,
        *,
        return_logits: bool = False,
        profile_phases: bool = False,
        selected_row_ids: Sequence[int] | None = None,
        capture_bindings: DenseQStoreCaptureBindings | None = None,
        capture_position_base: torch.Tensor | None = None,
        materialize_kv_delta: bool = True,
        synchronize: bool = True,
        last_logits_only: bool = False,
        last_top1_only: bool = False,
        _trusted_generated_token: bool = False,
        _capture_inputs_prevalidated: bool = False,
    ) -> DenseForwardResult | DenseSelectedForwardResult:
        if last_logits_only and not return_logits:
            raise ValueError("last-only language-head execution requires returned logits")
        if last_top1_only and return_logits:
            raise ValueError("last-top1 language-head execution cannot return logits")
        if last_logits_only and last_top1_only:
            raise ValueError("last-logit and last-top1 execution are mutually exclusive")
        if return_logits and selected_row_ids is not None:
            raise ValueError("full logits and selected-row logits are mutually exclusive")
        if capture_bindings is not None and selected_row_ids is None:
            raise RuntimeError("CUDA Graph capture supports only selected-row scoring")
        if capture_bindings is not None and (return_logits or profile_phases):
            raise RuntimeError("full-logit/profile execution is not CUDA Graph safe")
        if capture_bindings is not None and any(int(length) for length in cache.lengths):
            raise RuntimeError("stateful/decode CUDA Graph capture is not implemented")
        if _capture_inputs_prevalidated and capture_bindings is None:
            raise RuntimeError("prevalidated capture inputs require capture bindings")
        resources = self.store if capture_bindings is None else capture_bindings
        ids = torch.as_tensor(input_ids, device=self.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch,tokens]")
        if _capture_inputs_prevalidated:
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.dtype != torch.long
                or input_ids.device != self.device
            ):
                raise RuntimeError("prevalidated capture IDs must be device-local int64 tensors")
        elif _trusted_generated_token:
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.dtype != torch.long
                or input_ids.device != self.device
            ):
                raise RuntimeError(
                    "trusted generated-token execution requires device-local int64 IDs"
                )
            self.trusted_generated_token_bypasses += 1
        else:
            self.token_domain_checks += 1
            if ids.numel() and (
                torch.any(ids < 0).item() or torch.any(ids >= self.semantic_token_count).item()
            ):
                raise ValueError(
                    f"input token IDs must be inside [0, {self.semantic_token_count}); "
                    "padded model rows are not tokens"
                )
        batch, token_count = (int(value) for value in ids.shape)
        if batch != cache.batch_size:
            raise ValueError("input batch does not match cache")
        if token_count <= 0:
            raise ValueError("input block cannot be empty")
        if any(int(length) + token_count > cache.max_seq_len for length in cache.lengths):
            raise OverflowError("input block exceeds KV cache capacity")

        if synchronize:
            _sync(self.device)
        started = time.perf_counter()
        phase_started = started
        phase_wall_s: dict[str, float] | None = {} if profile_phases else None
        cfg = self.cfg
        hidden_size = int(cfg["hidden_size"])
        num_layers = int(cfg["num_hidden_layers"])
        num_heads = int(cfg["num_attention_heads"])
        num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))

        trusted_embed = getattr(resources, "_embed_rows_trusted_generated", None)
        if _trusted_generated_token and callable(trusted_embed):
            hidden = trusted_embed("embed", ids.reshape(-1)).clone()
        else:
            hidden = resources.embed_rows("embed", ids.reshape(-1)).clone()
        hidden = hidden.view(batch, token_count, hidden_size).to(self.compute_dtype)
        if phase_wall_s is not None:
            _sync(self.device)
            finished = time.perf_counter()
            phase_wall_s["embedding"] = finished - phase_started
            phase_started = finished

        if capture_position_base is None:
            position_base = torch.as_tensor(cache.lengths.copy(), device=self.device)
        else:
            if (
                tuple(capture_position_base.shape) != (batch,)
                or not _same_torch_device(
                    capture_position_base.device,
                    self.device,
                )
                or capture_position_base.dtype != torch.long
            ):
                raise ValueError("captured position base must be CUDA int64 [batch]")
            position_base = capture_position_base
        positions = position_base[:, None] + torch.arange(token_count, device=self.device)[None, :]
        delta_keys: list[torch.Tensor] = []
        delta_values: list[torch.Tensor] = []
        kv_read_bytes = 0
        fused_body = self.body_fusion_mode == _RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE
        normalized_for_attention = self._norm(hidden, "L0.ln1", resources) if fused_body else None
        final_hidden: torch.Tensor | None = None

        for layer in range(num_layers):
            if fused_body:
                if normalized_for_attention is None:
                    raise RuntimeError("fused body lost its next-layer normalized state")
                normalized = normalized_for_attention
            else:
                normalized = self._norm(hidden, f"L{layer}.ln1", resources)
            query = resources.matmul(f"L{layer}.q", normalized)
            key = resources.matmul(f"L{layer}.k", normalized)
            value = resources.matmul(f"L{layer}.v", normalized)
            for tensor_name, tensor in (("q", query), ("k", key), ("v", value)):
                bias_name = f"L{layer}.{tensor_name}.bias"
                if resources.has(bias_name):
                    tensor.add_(resources.fp32(bias_name).to(self.compute_dtype))
            query = query.view(batch, token_count, num_heads, self.head_dim)
            key = key.view(batch, token_count, num_kv_heads, self.head_dim)
            value = value.view(batch, token_count, num_kv_heads, self.head_dim)
            if resources.has(f"L{layer}.q_norm"):
                query = self._norm_tensor(query, resources.fp32(f"L{layer}.q_norm").float())
                key = self._norm_tensor(key, resources.fp32(f"L{layer}.k_norm").float())
            query = self._rope(query, positions)
            key = self._rope(key, positions)
            if materialize_kv_delta:
                delta_keys.append(key.clone())
                delta_values.append(value.clone())
            context, layer_read_bytes = self._attention(
                layer=layer,
                query=query,
                key_new=key,
                value_new=value,
                cache=cache,
                lengths=position_base,
            )
            kv_read_bytes += layer_read_bytes
            context = context.contiguous().view(batch, token_count, num_heads * self.head_dim)
            attention_output = resources.matmul(f"L{layer}.o", context)
            output_bias = f"L{layer}.o.bias"
            if resources.has(output_bias):
                attention_output.add_(resources.fp32(output_bias).to(self.compute_dtype))
            if fused_body:
                hidden, normalized = self._fused_residual_norm(
                    hidden,
                    attention_output,
                    f"L{layer}.ln2",
                    resources,
                )
                mlp = self._fused_gate_up_swiglu(normalized, layer, resources)
            else:
                hidden = hidden + attention_output
                normalized = self._norm(hidden, f"L{layer}.ln2", resources)
                gate = resources.matmul(f"L{layer}.gate", normalized)
                up = resources.matmul(f"L{layer}.up", normalized)
                mlp = torch.nn.functional.silu(gate) * up
            down = resources.matmul(f"L{layer}.down", mlp)
            if fused_body:
                next_norm = "norm.final" if layer + 1 == num_layers else f"L{layer + 1}.ln1"
                hidden, next_normalized = self._fused_residual_norm(
                    hidden,
                    down,
                    next_norm,
                    resources,
                )
                if layer + 1 == num_layers:
                    final_hidden = next_normalized
                    normalized_for_attention = None
                else:
                    normalized_for_attention = next_normalized
            else:
                hidden = hidden + down

        if final_hidden is None:
            final_hidden = self._norm(hidden, "norm.final", resources)
        if phase_wall_s is not None:
            _sync(self.device)
            finished = time.perf_counter()
            phase_wall_s["transformer"] = finished - phase_started
            phase_started = finished
        selected_logits = None
        if selected_row_ids is None:
            top1, logits = self._head(
                final_hidden,
                return_logits=return_logits,
                last_logits_only=last_logits_only,
                last_top1_only=last_top1_only,
            )
        else:
            selected_logits = self._selected_last_head(
                final_hidden,
                selected_row_ids,
                resources,
            )
        if synchronize:
            _sync(self.device)
        finished = time.perf_counter()
        if phase_wall_s is not None:
            phase_wall_s["language_head"] = finished - phase_started
            phase_wall_s["total"] = finished - started
        delta = (
            KVDelta(
                parent_epoch=cache.epoch,
                parent_lengths=tuple(int(value) for value in cache.lengths),
                cache_id=cache.cache_id,
                keys=tuple(delta_keys),
                values=tuple(delta_values),
                token_count=token_count,
            )
            if materialize_kv_delta
            else None
        )
        common = {
            "delta": delta,
            "hidden": final_hidden,
            "kv_read_bytes": kv_read_bytes,
            "kv_delta_bytes": 0 if delta is None else delta.byte_count,
            "wall_s": finished - started,
            "phase_wall_s": phase_wall_s,
        }
        if selected_logits is not None:
            return DenseSelectedForwardResult(
                selected_logits=selected_logits,
                **common,
            )
        if delta is None:
            raise RuntimeError("full-head execution requires a materialized KV delta")
        return DenseForwardResult(
            top1=top1,
            logits=logits,
            **common,
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: np.ndarray | torch.Tensor,
        cache: DenseQStoreKVCache,
        *,
        return_logits: bool = False,
        profile_phases: bool = False,
    ) -> DenseForwardResult:
        """Run the established full-vocabulary forward contract."""

        with cache._lock:  # noqa: SLF001 - target and cache share one transaction lease
            result = self._forward_impl(
                input_ids,
                cache,
                return_logits=return_logits,
                profile_phases=profile_phases,
            )
        if not isinstance(result, DenseForwardResult):
            raise RuntimeError("full forward returned a selected-head result")
        return result

    @torch.inference_mode()
    def forward_last_logits(
        self,
        input_ids: np.ndarray | torch.Tensor,
        cache: DenseQStoreKVCache,
    ) -> DenseForwardResult:
        """Run the full body while materializing only final-position vocabulary logits."""

        with cache._lock:  # noqa: SLF001 - target and cache share one transaction lease
            result = self._forward_impl(
                input_ids,
                cache,
                return_logits=True,
                last_logits_only=True,
            )
        if not isinstance(result, DenseForwardResult):
            raise RuntimeError("last-logit forward returned a selected-head result")
        if result.logits is None or int(result.logits.shape[1]) != 1:
            raise RuntimeError("last-logit forward did not preserve its bounded output shape")
        return result

    @torch.inference_mode()
    def forward_last_top1(
        self,
        input_ids: np.ndarray | torch.Tensor,
        cache: DenseQStoreKVCache,
    ) -> DenseForwardResult:
        """Run the full body while scoring only the final hidden row for greedy output."""

        with cache._lock:  # noqa: SLF001 - target and cache share one transaction lease
            result = self._forward_impl(
                input_ids,
                cache,
                return_logits=False,
                last_top1_only=True,
            )
        if not isinstance(result, DenseForwardResult):
            raise RuntimeError("last-top1 forward returned a selected-head result")
        if result.logits is not None or int(result.top1.shape[1]) != 1:
            raise RuntimeError("last-top1 forward did not preserve its bounded output shape")
        return result

    @torch.inference_mode()
    def _forward_trusted_generated(
        self,
        input_ids: torch.Tensor,
        cache: DenseQStoreKVCache,
    ) -> DenseForwardResult:
        """Internal decode route for tokens emitted by this target's masked greedy head."""

        with cache._lock:  # noqa: SLF001 - target and cache share one transaction lease
            result = self._forward_impl(
                input_ids,
                cache,
                _trusted_generated_token=True,
            )
        if not isinstance(result, DenseForwardResult):
            raise RuntimeError("trusted generated-token forward returned a selected-head result")
        return result

    @torch.inference_mode()
    def forward_selected_last(
        self,
        input_ids: np.ndarray | torch.Tensor,
        cache: DenseQStoreKVCache,
        row_ids: Sequence[int],
        *,
        capture_bindings: DenseQStoreCaptureBindings | None = None,
        capture_position_base: torch.Tensor | None = None,
        materialize_kv_delta: bool = True,
        synchronize: bool = True,
        _capture_inputs_prevalidated: bool = False,
    ) -> DenseSelectedForwardResult:
        """Run one stateless body and score only named final-position head rows."""

        with cache._lock:  # noqa: SLF001 - target and cache share one transaction lease
            result = self._forward_impl(
                input_ids,
                cache,
                selected_row_ids=row_ids,
                capture_bindings=capture_bindings,
                capture_position_base=capture_position_base,
                materialize_kv_delta=materialize_kv_delta,
                synchronize=synchronize,
                _capture_inputs_prevalidated=_capture_inputs_prevalidated,
            )
        if not isinstance(result, DenseSelectedForwardResult):
            raise RuntimeError("selected forward returned a full-head result")
        return result


@dataclass
class _CapturedSelectedCall:
    graph: Any
    output: torch.Tensor
    capture_stream: Any


class _TorchCUDAGraphDriver:
    """Small injectable boundary around the beta PyTorch CUDA Graph API."""

    requires_cuda = True
    name = "torch.cuda.CUDAGraph"

    @torch.inference_mode()
    def capture(
        self,
        operation: Any,
        *,
        device: torch.device,
        warmup: int,
    ) -> _CapturedSelectedCall:
        origin = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(origin)
        with torch.cuda.stream(capture_stream):
            warmup_output = None
            for _ in range(warmup):
                warmup_output = operation()
            del warmup_output
        capture_stream.synchronize()
        origin.wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            stream=capture_stream,
            capture_error_mode="global",
        ):
            static_output = operation()
        origin.wait_stream(capture_stream)
        return _CapturedSelectedCall(
            graph=graph,
            output=static_output,
            capture_stream=capture_stream,
        )


class DenseSelectedLastCUDAGraphExecutor:
    """One fail-closed static CUDA Graph for selected final-position scores."""

    def __init__(
        self,
        engine: DenseQStoreCudaEngine,
        ids_list: Sequence[np.ndarray | Sequence[int]],
        token_ids: Sequence[int],
        *,
        warmup: int = 3,
        max_resident_bytes: int | None = None,
        rebindable: bool = False,
        arena: DenseQStoreResidentArena | None = None,
        _driver: Any | None = None,
    ) -> None:
        if warmup <= 0:
            raise ValueError("CUDA Graph capture requires at least one warmup")
        rows = tuple(np.asarray(ids, dtype=np.int64) for ids in ids_list)
        if not rows:
            raise ValueError("CUDA Graph capture requires at least one input row")
        if any(row.ndim != 1 or not row.size for row in rows):
            raise ValueError("CUDA Graph inputs must be non-empty one-dimensional rows")
        lengths = {int(row.size) for row in rows}
        if len(lengths) != 1:
            raise ValueError("CUDA Graph capture requires one exact sequence shape")
        sequence_length = next(iter(lengths))
        if sequence_length > int(engine.max_seq_len):
            raise ValueError("CUDA Graph input length exceeds engine max_seq_len")
        selected = tuple(int(value) for value in token_ids)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("CUDA Graph selected token IDs must be non-empty and unique")
        vocab_size = int(engine.cfg["vocab_size"])
        if min(selected) < 0 or max(selected) >= vocab_size:
            raise ValueError(f"selected token IDs must be inside [0, {vocab_size})")

        driver = _TorchCUDAGraphDriver() if _driver is None else _driver
        device = torch.device(engine.target.device)
        if bool(getattr(driver, "requires_cuda", True)):
            if device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("selected CUDA Graph capture requires an available CUDA device")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_allocated = 0
            baseline_reserved = 0
        setup_started = time.perf_counter()
        static_ids = torch.as_tensor(
            np.stack(rows),
            device=device,
            dtype=torch.long,
        ).contiguous()
        static_position_base = torch.zeros(
            len(rows),
            device=device,
            dtype=torch.long,
        )
        static_cache = engine.target.empty_cache(len(rows))
        if static_cache.epoch != 0 or any(int(value) for value in static_cache.lengths):
            raise RuntimeError("CUDA Graph scoring requires a fresh stateless KV cache")
        qrow_names, fp32_names = engine.target.selected_capture_resource_names()
        if arena is not None:
            if not rebindable:
                raise ValueError("a resident arena requires a rebindable graph executor")
            bindings = arena.bind_selected_rows(selected)
        elif rebindable:
            bindings = engine.store.prepare_capture_bindings(
                qrow_names=qrow_names,
                fp32_names=fp32_names,
                selected_head_rows=selected,
                max_resident_bytes=max_resident_bytes,
                rebindable_head=True,
            )
        else:
            bindings = engine.store.prepare_capture_bindings(
                qrow_names=qrow_names,
                fp32_names=fp32_names,
                selected_head_rows=selected,
                max_resident_bytes=max_resident_bytes,
            )
        bindings.verify_stable_addresses()
        static_row_ids = (
            torch.as_tensor(selected, device=device, dtype=torch.long).contiguous()
            if rebindable
            else None
        )
        static_input_addresses = (
            int(static_ids.data_ptr()),
            int(static_position_base.data_ptr()),
            *((int(static_row_ids.data_ptr()),) if static_row_ids is not None else ()),
            *(int(tensor.data_ptr()) for tensor in static_cache.keys),
            *(int(tensor.data_ptr()) for tensor in static_cache.values),
        )

        def operation() -> torch.Tensor:
            result = engine.target.forward_selected_last(
                static_ids,
                static_cache,
                static_row_ids if static_row_ids is not None else selected,
                capture_bindings=bindings,
                capture_position_base=static_position_base,
                materialize_kv_delta=False,
                synchronize=False,
                _capture_inputs_prevalidated=True,
            )
            return result.selected_logits

        try:
            captured = driver.capture(
                operation,
                device=device,
                warmup=int(warmup),
            )
        except Exception:
            bindings.close()
            raise
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        capture_setup_ms = (time.perf_counter() - setup_started) * 1000.0
        bindings.verify_stable_addresses()
        observed_static_addresses = (
            int(static_ids.data_ptr()),
            int(static_position_base.data_ptr()),
            *((int(static_row_ids.data_ptr()),) if static_row_ids is not None else ()),
            *(int(tensor.data_ptr()) for tensor in static_cache.keys),
            *(int(tensor.data_ptr()) for tensor in static_cache.values),
        )
        if observed_static_addresses != static_input_addresses:
            bindings.close()
            raise RuntimeError("CUDA Graph static input/KV address changed during capture")
        expected_shape = (len(rows), len(selected))
        if tuple(captured.output.shape) != expected_shape:
            bindings.close()
            raise RuntimeError(
                "captured selected-head output has the wrong shape "
                f"({tuple(captured.output.shape)} != {expected_shape})"
            )
        if captured.output.dtype is not torch.float32:
            bindings.close()
            raise RuntimeError("captured selected-head output must use FP32")
        if not _same_torch_device(captured.output.device, device):
            bindings.close()
            raise RuntimeError("captured selected-head output moved off the capture device")
        if device.type == "cuda":
            allocated_delta = max(
                0,
                int(torch.cuda.memory_allocated(device)) - baseline_allocated,
            )
            reserved_delta = max(
                0,
                int(torch.cuda.memory_reserved(device)) - baseline_reserved,
            )
            peak_allocated_delta = max(
                0,
                int(torch.cuda.max_memory_allocated(device)) - baseline_allocated,
            )
        else:
            allocated_delta = reserved_delta = peak_allocated_delta = 0
        total_residency_delta = max(
            allocated_delta,
            reserved_delta,
            peak_allocated_delta,
        )
        if max_resident_bytes is not None and total_residency_delta > max_resident_bytes:
            bindings.close()
            raise MemoryError(
                "CUDA Graph total residency exceeded its budget after capture "
                f"({total_residency_delta} > {max_resident_bytes} bytes)"
            )

        self._bindings: DenseQStoreCaptureBindings | None = bindings
        self._cache: DenseQStoreKVCache | None = static_cache
        self._captured: _CapturedSelectedCall | None = captured
        self._driver_name = str(getattr(driver, "name", type(driver).__name__))
        self._operation: Any | None = operation
        self._static_ids: torch.Tensor | None = static_ids
        self._static_position_base: torch.Tensor | None = static_position_base
        self._static_row_ids: torch.Tensor | None = static_row_ids
        self._lock = threading.Lock()
        self._closed = False
        self._eager_control_count = 0
        self._replay_count = 0
        self._rebind_count = 0
        self._generation = 0
        self._bound_request_id: str | None = None
        self._rebindable = bool(rebindable)
        self._warmup = int(warmup)
        self._shape = expected_shape
        self._semantic_token_count = int(
            getattr(engine, "semantic_token_count", engine.cfg["vocab_size"])
        )
        self._static_addresses = (
            *static_input_addresses,
            int(captured.output.data_ptr()),
        )
        self._base_evidence = {
            "graph_replay": True,
            "capture_ready": True,
            "capture_executed": True,
            "capture_backend": self._driver_name,
            "graph_backend": self._driver_name,
            "capture_mode": "selected-last-stateless-score",
            "capture_warmup_iterations": self._warmup,
            "capture_warmups": self._warmup,
            "capture_count": 1,
            "capture_static_shapes": True,
            "capture_rebindable": self._rebindable,
            "capture_stable_addresses": True,
            "stable_addresses_verified": bindings.stable_addresses_verified,
            "capture_graph_safe": True,
            "capture_input_shape": [len(rows), sequence_length],
            "capture_output_shape": list(expected_shape),
            "capture_output_dtype": "fp32",
            "capture_full_logits": False,
            "capture_stateful_kv": False,
            "capture_decode": False,
            "capture_matched_eager_control_available": True,
            "capture_matched_eager_control_basis": (
                "same-operation-static-inputs-kv-and-pinned-resources-without-graph-replay"
            ),
            "capture_static_kv_allocated_bytes": static_cache.allocated_bytes,
            "capture_static_input_bytes": (
                _tensor_bytes(static_ids) + _tensor_bytes(static_position_base)
            ),
            "capture_setup_ms": capture_setup_ms,
            "capture_memory_baseline_allocated_bytes": baseline_allocated,
            "capture_memory_baseline_reserved_bytes": baseline_reserved,
            "capture_total_allocated_delta_bytes": allocated_delta,
            "capture_total_reserved_delta_bytes": reserved_delta,
            "capture_peak_allocated_delta_bytes": peak_allocated_delta,
            "capture_total_residency_delta_bytes": total_residency_delta,
            **bindings.evidence(),
        }

    @property
    def evidence(self) -> MappingProxyType:
        with self._lock:
            payload = {
                **self._base_evidence,
                "capture_replay_count": self._replay_count,
                "capture_matched_eager_control_count": self._eager_control_count,
                "capture_rebind_count": self._rebind_count,
                "capture_binding_generation": self._generation,
                "capture_bound_request_id": self._bound_request_id,
                "capture_executor_closed": self._closed,
            }
        return MappingProxyType(payload)

    @property
    def replay_count(self) -> int:
        with self._lock:
            return self._replay_count

    def _verify_addresses(self) -> None:
        if (
            self._bindings is None
            or self._cache is None
            or self._captured is None
            or self._static_ids is None
            or self._static_position_base is None
        ):
            raise RuntimeError("CUDA Graph executor lost its static resources")
        self._bindings.verify_stable_addresses()
        observed = (
            int(self._static_ids.data_ptr()),
            int(self._static_position_base.data_ptr()),
            *((int(self._static_row_ids.data_ptr()),) if self._static_row_ids is not None else ()),
            *(int(tensor.data_ptr()) for tensor in self._cache.keys),
            *(int(tensor.data_ptr()) for tensor in self._cache.values),
            int(self._captured.output.data_ptr()),
        )
        if observed != self._static_addresses:
            raise RuntimeError("CUDA Graph static tensor address changed after capture")

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @torch.inference_mode()
    def rebind(
        self,
        ids_list: Sequence[np.ndarray | Sequence[int]],
        token_ids: Sequence[int],
        *,
        request_id: str,
    ) -> int:
        """Install one request into fixed-address prompt and selected-row buffers."""

        rows = tuple(np.asarray(ids, dtype=np.int64) for ids in ids_list)
        selected = tuple(int(value) for value in token_ids)
        if not request_id:
            raise ValueError("rebind request_id must be non-empty")
        with self._lock:
            if self._closed or not self._rebindable:
                raise RuntimeError("CUDA Graph executor is not open and rebindable")
            if self._static_ids is None or self._static_row_ids is None:
                raise RuntimeError("rebindable CUDA Graph buffers are missing")
            expected_input = tuple(self._static_ids.shape)
            if (
                len(rows) != expected_input[0]
                or any(row.ndim != 1 or int(row.size) != expected_input[1] for row in rows)
            ):
                raise ValueError("rebound prompt rows do not match the captured shape")
            if len(selected) != int(self._static_row_ids.numel()):
                raise ValueError("rebound selected rows do not match the captured shape")
            if len(selected) != len(set(selected)):
                raise ValueError("rebound selected rows must be unique")
            if (
                not selected
                or min(selected) < 0
                or max(selected) >= self._semantic_token_count
            ):
                raise ValueError("rebound selected row is outside the semantic vocabulary")
            self._verify_addresses()
            self._static_ids.copy_(torch.as_tensor(np.stack(rows), device=self._static_ids.device))
            self._static_row_ids.copy_(
                torch.as_tensor(selected, device=self._static_row_ids.device, dtype=torch.long)
            )
            self._generation += 1
            self._rebind_count += 1
            self._bound_request_id = request_id
            return self._generation

    @torch.inference_mode()
    def execute(
        self,
        *,
        expected_generation: int | None = None,
        request_id: str | None = None,
    ) -> torch.Tensor:
        """Replay once and return an ownership-safe CPU FP32 score matrix."""

        with self._lock:
            if self._closed or self._captured is None:
                raise RuntimeError("CUDA Graph executor is closed")
            if expected_generation is not None and int(expected_generation) != self._generation:
                raise RuntimeError("CUDA Graph binding generation is stale")
            if request_id is not None and request_id != self._bound_request_id:
                raise RuntimeError("CUDA Graph request ID does not own the installed binding")
            self._verify_addresses()
            self._captured.graph.replay()
            output = self._captured.output.detach().to(
                device="cpu",
                dtype=torch.float32,
                copy=True,
            )
            self._replay_count += 1
            return output

    @torch.inference_mode()
    def execute_eager_control(self) -> torch.Tensor:
        """Run the same static bindings eagerly for capture-only attribution."""

        with self._lock:
            if self._closed or self._operation is None:
                raise RuntimeError("CUDA Graph executor is closed")
            self._verify_addresses()
            output = (
                self._operation()
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                    copy=True,
                )
            )
            self._eager_control_count += 1
            return output

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            bindings = self._bindings
            self._bindings = None
            self._captured = None
            self._cache = None
            self._operation = None
            self._static_ids = None
            self._static_position_base = None
            self._static_row_ids = None
            self._static_addresses = ()
            self._closed = True
        if bindings is not None:
            bindings.close()


class DenseQStoreCudaEngine(BaseEngine):
    """Public explicit-only adapter for the dense compact QStore CUDA runtime."""

    backend = "dense-qstore-cuda"
    supports_batch = True

    TORCH_NUMERICAL_CONTRACT = "torch-batched-established"
    ROW_STABLE_NUMERICAL_CONTRACT = "row-stable-triton-v1"
    # v1 body plus a batch-invariant FP32 vocabulary head (logits no longer depend on B*T).
    ROW_STABLE_V2_NUMERICAL_CONTRACT = "row-stable-triton-v2"

    def __init__(
        self,
        model_name: str,
        *,
        stores_dir: str | Path | None = None,
        store_path: str | Path | None = None,
        linked_extension_id: str | None = None,
        prebind_linked_extension: bool = False,
        device: str = "cuda",
        compute_dtype: str = "bf16",
        compact_cache_mb: float = 0.0,
        component_graph: str | Path | None = None,
        output_contract: Any = "full_logits",
        component_cache_mb: Mapping[str, float] | None = None,
        lexical_component_path: str | Path | None = None,
        lexical_values_path: str | Path | None = None,
        lexical_weights_path: str | Path | None = None,
        lexical_binding_path: str | Path | None = None,
        pin_component_fp32_aux: bool = True,
        resident_exact_head_mb: float | None = None,
        max_seq_len: int = 128,
        stable_block_m: int = 16,
        stable_reductions: bool = False,
        decode_attention_mode: str = _ESTABLISHED_DECODE_ATTENTION_MODE,
        decode_attention_tile: int = 64,
        body_fusion_mode: str = _ESTABLISHED_BODY_FUSION_MODE,
        allow_failed_stable_reduction_gate: bool = False,
        numerical_contract: str = TORCH_NUMERICAL_CONTRACT,
        experimental_reranked_head: bool = False,
        require_triton: bool = True,
        **_ignored: Any,
    ) -> None:
        self._closed = False
        self.composite_store: CompositeQStore | None = None
        self.component_output_contract: str | None = None
        self.store_path: Path | None = None
        self.linked_extension_id: str | None = None
        self.lexical_component: LexicalComponent | None = None
        separated_lexical = (
            lexical_values_path,
            lexical_weights_path,
            lexical_binding_path,
        )
        if any(value is not None for value in separated_lexical) and not all(
            value is not None for value in separated_lexical
        ):
            raise ValueError(
                "lexical_values_path, lexical_weights_path, and lexical_binding_path "
                "must be supplied together"
            )
        if lexical_component_path is not None and any(
            value is not None for value in separated_lexical
        ):
            raise ValueError("combined and separated lexical artifacts are mutually exclusive")
        if lexical_component_path is not None and component_graph is not None:
            raise ValueError(
                "lexical components currently compose with body stores, not component graphs"
            )
        if component_graph is not None and any(value is not None for value in separated_lexical):
            raise ValueError(
                "lexical components currently compose with body stores, not component graphs"
            )
        if torch.device(device).type != "cuda":
            raise ValueError("dense-qstore-cuda requires a CUDA device")
        if compute_dtype not in {"bf16", "fp16"}:
            raise ValueError("dense-qstore-cuda compute_dtype must be 'bf16' or 'fp16'")
        decode_attention_mode, decode_attention_tile = _decode_attention_config(
            decode_attention_mode,
            decode_attention_tile,
        )
        body_fusion_mode = _body_fusion_config(body_fusion_mode)
        if body_fusion_mode == _RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE:
            if compute_dtype != "bf16":
                raise ValueError("residual-rms-swiglu-v1 requires BF16 compute")
            if not require_triton:
                raise RuntimeError("residual-rms-swiglu-v1 requires Triton execution")
            if component_graph is not None:
                raise NotImplementedError(
                    "residual-rms-swiglu-v1 is not yet authorized for composite QStore views"
                )
        raw_output_contract = str(getattr(output_contract, "value", output_contract))
        normalized_output_contract = {
            "full_logits": "full_logits",
            "last_token_logits": "full_logits",
            "loss_only": "full_logits",
        }.get(raw_output_contract, raw_output_contract)
        if component_graph is not None:
            if store_path is not None or linked_extension_id is not None:
                raise ValueError("component graphs cannot be combined with an explicit store image")
            if prebind_linked_extension:
                raise ValueError("linked-image prebinding requires an explicit dense store image")
            if stores_dir is not None:
                raise ValueError("stores_dir and component_graph are mutually exclusive")
            if not isinstance(component_cache_mb, (Mapping, type(None))):
                raise TypeError("component_cache_mb must map component roles to MiB budgets")
            if normalized_output_contract != "full_logits":
                raise NotImplementedError(
                    "dense component graphs currently support only the full-logit contract; "
                    "selected-row component execution needs a separately bounded head ABI"
                )
            if experimental_reranked_head:
                raise NotImplementedError(
                    "the experimental reranked head is not authorized for component graphs"
                )
            if resident_exact_head_mb is not None and float(resident_exact_head_mb) <= 0:
                raise ValueError("resident_exact_head_mb must be positive when provided")
        elif component_cache_mb is not None:
            raise ValueError("component_cache_mb requires component_graph")
        elif resident_exact_head_mb is not None:
            raise ValueError("resident_exact_head_mb currently requires component_graph")
        if prebind_linked_extension and linked_extension_id is None and store_path is None:
            raise ValueError(
                "prebind_linked_extension requires linked_extension_id or store_path"
            )
        aliases = {
            "torch": self.TORCH_NUMERICAL_CONTRACT,
            "established": self.TORCH_NUMERICAL_CONTRACT,
            "row-stable": self.ROW_STABLE_NUMERICAL_CONTRACT,
            "deterministic": self.ROW_STABLE_NUMERICAL_CONTRACT,
            "row-stable-v2": self.ROW_STABLE_V2_NUMERICAL_CONTRACT,
            "batch-invariant": self.ROW_STABLE_V2_NUMERICAL_CONTRACT,
        }
        numerical_contract = aliases.get(numerical_contract, numerical_contract)
        row_stable_contracts = {
            self.ROW_STABLE_NUMERICAL_CONTRACT,
            self.ROW_STABLE_V2_NUMERICAL_CONTRACT,
        }
        valid_contracts = {self.TORCH_NUMERICAL_CONTRACT, *row_stable_contracts}
        if numerical_contract not in valid_contracts:
            choices = ", ".join(sorted(valid_contracts))
            raise ValueError(f"numerical_contract must be one of: {choices}")
        if (
            stable_reductions
            and not allow_failed_stable_reduction_gate
            and (numerical_contract not in row_stable_contracts)
        ):
            raise RuntimeError(
                "the row-stable reduction chain is disabled: Beast gate job-238bd82cd387 "
                "matched only 22/34 established B1 trace tokens. Pass "
                "numerical_contract='row-stable-triton-v1' to select the separately gated "
                "deterministic variant, or allow_failed_stable_reduction_gate=True only for "
                "legacy diagnostics."
            )
        if stable_reductions and numerical_contract not in row_stable_contracts:
            numerical_contract = self.ROW_STABLE_NUMERICAL_CONTRACT
        stable_reductions = bool(stable_reductions or numerical_contract in row_stable_contracts)
        stable_head = numerical_contract == self.ROW_STABLE_V2_NUMERICAL_CONTRACT
        self.spec = resolve_model(model_name)
        self.name = self.spec.name
        opened: DenseQStore | CompositeQStore | None = None
        try:
            if component_graph is not None:
                concrete_device = str(_concrete_cuda_device(device))
                composite = CompositeQStore(
                    component_graph,
                    cache_mb=compact_cache_mb,
                    component_cache_mb=component_cache_mb,
                    compute_dtype=compute_dtype,
                    provider_backend="dense-qstore-cuda",
                    provider_device=concrete_device,
                    provider_require_triton=require_triton,
                    provider_stable_block_m=stable_block_m,
                    provider_pin_fp32_aux=pin_component_fp32_aux,
                )
                opened = composite
                if composite.graph.model_name != self.name:
                    raise ComponentGraphError(
                        "component graph model does not match requested engine "
                        f"({composite.graph.model_name!r} != {self.name!r})"
                    )
                if composite.graph.architecture != self.spec.family:
                    raise ComponentGraphError(
                        "component graph architecture does not match the model registry "
                        f"({composite.graph.architecture!r} != {self.spec.family!r})"
                    )
                self.store = composite.for_contract(output_contract)
                if self.store.output_contract != "full_logits":
                    raise NotImplementedError(
                        "dense component execution requires the full-logit contract"
                    )
                self.composite_store = composite
                self.component_output_contract = self.store.output_contract
                if resident_exact_head_mb is not None:
                    composite.prepare_resident_exact_head(resident_exact_head_mb)
            else:
                explicit_store_path: Path | None = None
                if store_path is not None:
                    explicit_store_path = Path(store_path).expanduser().resolve()
                    if explicit_store_path.name == "manifest.json":
                        explicit_store_path = explicit_store_path.parent
                    if not explicit_store_path.is_dir():
                        raise FileNotFoundError(
                            f"explicit dense QStore image is not a directory: {explicit_store_path}"
                        )
                    if not (explicit_store_path / "manifest.json").is_file():
                        raise FileNotFoundError(
                            f"explicit dense QStore image has no manifest.json: {explicit_store_path}"
                        )
                    root = explicit_store_path.parent
                    key = explicit_store_path.name
                else:
                    root = Path(stores_dir) if stores_dir is not None else default_stores_root()
                    if linked_extension_id is not None:
                        explicit_store_path = _find_linked_store(
                            root,
                            self.spec.name,
                            str(linked_extension_id),
                        )
                        key = explicit_store_path.name
                    else:
                        candidates = [
                            store_name(self.spec),
                            self.spec.name,
                            self.spec.hf_id.split("/")[-1],
                            str(model_name),
                        ]
                        key = next(
                            (
                                candidate
                                for candidate in dict.fromkeys(candidates)
                                if (root / candidate / "manifest.json").exists()
                            ),
                            None,
                        )
                        if key is None:
                            tried = ", ".join(dict.fromkeys(candidates))
                            raise FileNotFoundError(
                                f"no dense int8 QStore under {root} (tried: {tried}). Build one with "
                                f"`mrun build-store {self.spec.name}` or set MRUN_STORES_ROOT."
                            )
                concrete_device = str(_concrete_cuda_device(device))
                dense_store = DenseQStore(
                    str(key),
                    root=root,
                    device=concrete_device,
                    compute_dtype=compute_dtype,
                    compact_cache_mb=compact_cache_mb,
                    require_triton=require_triton,
                    stable_block_m=stable_block_m,
                )
                opened = dense_store
                self.store = dense_store
                actual_path = Path(dense_store.directory).resolve()
                if explicit_store_path is not None and actual_path != explicit_store_path:
                    raise RuntimeError(
                        "DenseQStore opened a different image than requested: "
                        f"{actual_path} != {explicit_store_path}"
                    )
                manifest_model = str(dense_store.man.get("model_name", ""))
                if manifest_model != self.spec.name:
                    raise ValueError(
                        f"explicit dense QStore model {manifest_model!r} does not match "
                        f"requested model {self.spec.name!r}"
                    )
                linked_image = dense_store.man.get("linked_image")
                actual_extension_id = (
                    str(linked_image.get("extension_id"))
                    if isinstance(linked_image, Mapping)
                    and linked_image.get("extension_id") is not None
                    else None
                )
                if linked_extension_id is not None and actual_extension_id != str(
                    linked_extension_id
                ):
                    raise ValueError(
                        f"explicit dense QStore extension {actual_extension_id!r} does not "
                        f"match requested {linked_extension_id!r}"
                    )
                if prebind_linked_extension:
                    overlay_blocks = (
                        linked_image.get("overlay_blocks")
                        if isinstance(linked_image, Mapping)
                        else None
                    )
                    if not isinstance(overlay_blocks, list) or not overlay_blocks:
                        raise ValueError(
                            "prebind_linked_extension requires linked_image.overlay_blocks"
                        )
                    dense_store.prebind(tuple(str(name) for name in overlay_blocks))
                self.store_path = actual_path
                self.linked_extension_id = actual_extension_id

            if lexical_component_path is not None:
                if experimental_reranked_head:
                    raise NotImplementedError(
                        "experimental reranked head does not support external lexical components"
                    )
                self.lexical_component = LexicalComponent.load(
                    lexical_component_path,
                    body_config=self.store.cfg,
                    architecture=str(self.store.man.get("arch", self.spec.family)),
                )
                self.store = LexicalQStoreView(self.store, self.lexical_component)
                self.tokenizer = self.lexical_component.tokenizer
            elif all(value is not None for value in separated_lexical):
                self.lexical_component = load_separated_lexical(
                    values_path=lexical_values_path,
                    weights_path=lexical_weights_path,
                    binding_path=lexical_binding_path,
                    body_config=self.store.cfg,
                    architecture=str(self.store.man.get("arch", self.spec.family)),
                )
                self.store = LexicalQStoreView(self.store, self.lexical_component)
                self.tokenizer = self.lexical_component.tokenizer
            else:
                self.tokenizer = load_tokenizer(self.spec)
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.composite_store is not None:
                self.composite_store.validate_tokenizer(self.tokenizer)
            self.cfg = self.store.cfg
            configured_vocab_rows = int(self.cfg.get("vocab_size", 0))
            if configured_vocab_rows <= 0:
                raise ValueError("model config must declare a positive vocab_size")
            if self.composite_store is not None:
                semantic_token_count = int(self.composite_store.vocab.token_count)
            else:
                try:
                    semantic_token_count = int(len(self.tokenizer))
                except (TypeError, AttributeError) as exc:
                    raise TypeError("model tokenizer must expose its semantic token count") from exc
            if semantic_token_count <= 0 or semantic_token_count > configured_vocab_rows:
                raise ValueError(
                    "semantic token count must be positive and no larger than configured "
                    f"vocabulary rows ({semantic_token_count} > {configured_vocab_rows})"
                )
            self.semantic_token_count = semantic_token_count
            self.target = DenseQStoreTarget(
                self.store,
                max_seq_len=max_seq_len,
                stable_reductions=stable_reductions,
                decode_attention_mode=decode_attention_mode,
                decode_attention_tile=decode_attention_tile,
                body_fusion_mode=body_fusion_mode,
                experimental_reranked_head=experimental_reranked_head,
                semantic_token_count=semantic_token_count,
                stable_head=stable_head,
            )
            self.arch = str(self.store.man.get("arch", "qwen2"))
            self.device = str(self.target.device)
            self.n_layer = int(self.cfg["num_hidden_layers"])
            self.inter = int(self.cfg["intermediate_size"])
            self.hidden = int(self.cfg["hidden_size"])
            self.max_seq_len = int(max_seq_len)
            self.numerical_contract = _execution_numerical_contract(
                numerical_contract,
                decode_attention_mode,
                body_fusion_mode,
            )
            self.subset_head_numerical_contract = f"{self.numerical_contract}+selected-head-fp32-v1"
            self.supported_numerical_contracts = (
                self.numerical_contract,
                self.subset_head_numerical_contract,
            )
            self._last_cache: DenseQStoreKVCache | None = None
            self._cuda_graph_executors: weakref.WeakSet[DenseSelectedLastCUDAGraphExecutor] = (
                weakref.WeakSet()
            )
        except BaseException:
            if opened is not None:
                opened.close()
            self.composite_store = None
            raise

    @property
    def working_set_mb(self) -> float:
        cache_bytes = self._last_cache.allocated_bytes if self._last_cache is not None else 0
        if self.composite_store is None:
            compact_peak = self.store.stats.peak_compact_resident_bytes
        else:
            providers = self.composite_store.snapshot()["providers"].values()
            compact_peak = sum(
                int((provider.get("store_stats") or {}).get("resident_physical_bytes", 0))
                for provider in providers
            )
        return float(cache_bytes + compact_peak) / 1e6

    @working_set_mb.setter
    def working_set_mb(self, _value: Any) -> None:
        return None

    def new_cache(self, batch_size: int) -> DenseQStoreKVCache:
        cache = self.target.empty_cache(batch_size)
        self._last_cache = cache
        return cache

    def forward_block(
        self,
        input_ids: np.ndarray | torch.Tensor,
        cache: DenseQStoreKVCache,
        *,
        return_logits: bool = False,
        profile_phases: bool = False,
    ) -> DenseForwardResult:
        """Run a provisional block. The returned K/V delta is not committed."""

        self._last_cache = cache
        return self.target.forward(
            input_ids,
            cache,
            return_logits=return_logits,
            profile_phases=profile_phases,
        )

    def forward_last_logits(
        self,
        input_ids: np.ndarray | torch.Tensor,
        cache: DenseQStoreKVCache,
    ) -> DenseForwardResult:
        """Run a provisional block with only ``[B,1,V]`` device-resident logits."""

        self._last_cache = cache
        return self.target.forward_last_logits(input_ids, cache)

    def forward_last_top1(
        self,
        input_ids: np.ndarray | torch.Tensor,
        cache: DenseQStoreKVCache,
    ) -> DenseForwardResult:
        """Run a provisional block and select only its final-position greedy token."""

        self._last_cache = cache
        return self.target.forward_last_top1(input_ids, cache)

    def forward_decode_slots(
        self,
        input_ids: np.ndarray | torch.Tensor,
        indexed_cache: DenseQStoreKVCache,
    ) -> DenseForwardResult:
        """Run one provisional decode token against indexed rows of a shared KV pool."""

        if self.target.decode_attention_mode != _SEGMENTED_FLASH_GQA_DECODE_MODE:
            raise RuntimeError("forward_decode_slots requires segmented-flash-gqa-decode-v1")
        rows = torch.as_tensor(input_ids)
        if rows.ndim != 2 or int(rows.shape[1]) != 1:
            raise ValueError("forward_decode_slots requires input_ids with shape [B,1]")
        batch = int(rows.shape[0])
        if int(getattr(indexed_cache, "batch_size", -1)) != batch:
            raise ValueError("indexed cache must expose one logical length per decode request")
        row_indices = getattr(indexed_cache, "row_indices", None)
        if not isinstance(row_indices, torch.Tensor):
            raise TypeError("indexed cache row_indices must be a device tensor")
        if row_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indexed cache row_indices must use int32 or int64")
        if tuple(row_indices.shape) != (batch,):
            raise ValueError("indexed cache row_indices must have shape [B]")
        if row_indices.device != indexed_cache.device:
            raise ValueError("indexed cache row_indices must use the cache device")
        layer_count = int(self.target.cfg["num_hidden_layers"])
        if (
            not indexed_cache.keys
            or len(indexed_cache.keys) != layer_count
            or len(indexed_cache.values) != layer_count
        ):
            raise ValueError("indexed cache must expose every layer of the shared KV pool")
        pool_slots = int(indexed_cache.keys[0].shape[0])
        if any(
            tuple(key.shape) != tuple(value.shape) or int(key.shape[0]) != pool_slots
            for key, value in zip(indexed_cache.keys, indexed_cache.values, strict=True)
        ):
            raise ValueError("indexed cache key/value layers must share one pool geometry")
        if pool_slots <= 0 or bool(((row_indices < 0) | (row_indices >= pool_slots)).any().item()):
            raise ValueError("indexed cache row_indices select outside the shared KV pool")
        epoch_before = int(indexed_cache.epoch)
        lengths_before = indexed_cache.lengths.copy()
        self._last_cache = indexed_cache
        result = self.target.forward_last_top1(rows, indexed_cache)
        if indexed_cache.epoch != epoch_before or not np.array_equal(
            indexed_cache.lengths,
            lengths_before,
        ):
            raise RuntimeError("provisional indexed decode mutated committed cache authority")
        return result

    def commit_block(
        self,
        cache: DenseQStoreKVCache,
        result: DenseForwardResult,
        accepted_counts: Sequence[int],
    ) -> CommitStats:
        return cache.commit(result.delta, accepted_counts)

    def prefill_ids(
        self,
        input_ids: np.ndarray | torch.Tensor,
        *,
        return_logits: bool = False,
        last_top1_only: bool = False,
    ) -> tuple[DenseQStoreKVCache, DenseForwardResult]:
        if return_logits and last_top1_only:
            raise ValueError("last-top1 prefill cannot return logits")
        rows = torch.as_tensor(input_ids)
        batch_size = 1 if rows.ndim == 1 else int(rows.shape[0])
        cache = self.new_cache(batch_size)
        result = (
            self.forward_last_top1(rows, cache)
            if last_top1_only
            else self.forward_block(rows, cache, return_logits=return_logits)
        )
        cache.commit(result.delta, [result.delta.token_count] * batch_size)
        return cache, result

    @torch.inference_mode()
    def logits(self, ids: np.ndarray) -> torch.Tensor:
        row = np.asarray(ids, dtype=np.int64)
        if row.ndim != 1 or not row.size:
            raise ValueError("ids must be a non-empty one-dimensional token array")
        _, result = self.prefill_ids(row, return_logits=True)
        if result.logits is None:
            raise RuntimeError("dense target did not return requested logits")
        return result.logits[0].cpu()

    @torch.inference_mode()
    def logits_batch(self, ids_list: list[np.ndarray]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor | None] = [None] * len(ids_list)
        by_length: dict[int, list[int]] = {}
        for index, ids in enumerate(ids_list):
            row = np.asarray(ids, dtype=np.int64)
            if row.ndim != 1 or not row.size:
                raise ValueError("each ids row must be non-empty and one-dimensional")
            by_length.setdefault(int(row.size), []).append(index)
        for indices in by_length.values():
            rows = np.stack([np.asarray(ids_list[index], dtype=np.int64) for index in indices])
            _, result = self.prefill_ids(rows, return_logits=True)
            if result.logits is None:
                raise RuntimeError("dense target did not return requested logits")
            logits = result.logits.cpu()
            for offset, index in enumerate(indices):
                outputs[index] = logits[offset]
        return [output for output in outputs if output is not None]

    @torch.inference_mode()
    def selected_last_logits_batch(
        self,
        ids_list: Sequence[np.ndarray],
        token_ids: Sequence[int],
    ) -> torch.Tensor:
        """Run each equal-length body batch once and score only selected final rows."""

        if self.composite_store is not None:
            raise NotImplementedError(
                "selected-row component execution requires a separately bounded head ABI"
            )

        selected = tuple(int(value) for value in token_ids)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("selected token IDs must be non-empty and unique")
        if min(selected) < 0 or max(selected) >= self.semantic_token_count:
            raise ValueError(f"selected token IDs must be inside [0, {self.semantic_token_count})")
        if not ids_list:
            raise ValueError("ids_list must contain at least one row")

        outputs: list[torch.Tensor | None] = [None] * len(ids_list)
        by_length: dict[int, list[int]] = {}
        for index, ids in enumerate(ids_list):
            row = np.asarray(ids, dtype=np.int64)
            if row.ndim != 1 or not row.size:
                raise ValueError("each ids row must be non-empty and one-dimensional")
            by_length.setdefault(int(row.size), []).append(index)
        for indices in by_length.values():
            rows = np.stack([np.asarray(ids_list[index], dtype=np.int64) for index in indices])
            cache = self.new_cache(len(indices))
            result = self.target.forward_selected_last(
                rows,
                cache,
                selected,
                materialize_kv_delta=False,
            )
            scores = result.selected_logits.cpu()
            for offset, index in enumerate(indices):
                outputs[index] = scores[offset]
        if any(output is None for output in outputs):
            raise RuntimeError("dense selected-row execution did not populate every output")
        return torch.stack([output for output in outputs if output is not None])

    def prepare_selected_last_cuda_graph(
        self,
        ids_list: Sequence[np.ndarray | Sequence[int]],
        token_ids: Sequence[int],
        *,
        warmup: int = 3,
        residency_budget_mb: float | None = None,
    ) -> DenseSelectedLastCUDAGraphExecutor:
        """Capture one exact-shape, stateless selected-row score program.

        Capture is required by this entry point.  It never falls back to eager
        execution, and it deliberately does not accept full logits, an existing KV
        cache, or decode state.
        """

        if self.composite_store is not None:
            raise NotImplementedError(
                "CUDA Graph selected-row capture is not implemented for component stores"
            )

        if residency_budget_mb is not None and float(residency_budget_mb) <= 0:
            raise ValueError("CUDA Graph residency budget must be positive")
        executor = DenseSelectedLastCUDAGraphExecutor(
            self,
            ids_list,
            token_ids,
            warmup=warmup,
            max_resident_bytes=(
                None if residency_budget_mb is None else int(float(residency_budget_mb) * 1e6)
            ),
        )
        self._cuda_graph_executors.add(executor)
        return executor

    def prepare_resident_qstore_arena(
        self,
        *,
        residency_budget_mb: float | None = None,
    ) -> DenseQStoreResidentArena:
        """Create one immutable QStore owner for a family of graph templates."""

        if self.composite_store is not None:
            raise NotImplementedError("resident CUDA Graph arenas require a dense QStore")
        if residency_budget_mb is not None and float(residency_budget_mb) <= 0:
            raise ValueError("resident QStore arena budget must be positive")
        qrow_names, fp32_names = self.target.selected_capture_resource_names()
        return self.store.prepare_resident_arena(
            qrow_names=qrow_names,
            fp32_names=fp32_names,
            max_resident_bytes=(
                None if residency_budget_mb is None else int(float(residency_budget_mb) * 1e6)
            ),
        )

    def prepare_rebindable_selected_last_cuda_graph(
        self,
        ids_list: Sequence[np.ndarray | Sequence[int]],
        token_ids: Sequence[int],
        *,
        arena: DenseQStoreResidentArena,
        warmup: int = 3,
    ) -> DenseSelectedLastCUDAGraphExecutor:
        """Capture a fixed-shape template whose prompt and selected rows are mutable."""

        if not isinstance(arena, DenseQStoreResidentArena):
            raise TypeError("arena must be a DenseQStoreResidentArena")
        executor = DenseSelectedLastCUDAGraphExecutor(
            self,
            ids_list,
            token_ids,
            warmup=warmup,
            rebindable=True,
            arena=arena,
        )
        self._cuda_graph_executors.add(executor)
        return executor

    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: np.ndarray | torch.Tensor,
        *,
        max_new_tokens: int = 48,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        rows = torch.as_tensor(input_ids, dtype=torch.long)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.ndim != 2 or rows.shape[1] == 0:
            raise ValueError("input_ids must have shape [tokens] or [batch,tokens]")
        if int(rows.shape[1]) + max_new_tokens - 1 > self.max_seq_len:
            raise OverflowError("prompt and generated tokens exceed max_seq_len")
        cache, result = self.prefill_ids(
            rows,
            last_top1_only=self.target.experimental_reranked_head,
        )
        token = result.top1[:, -1]
        generated = [token.cpu()]
        for _ in range(1, max_new_tokens):
            self._last_cache = cache
            result = self.target._forward_trusted_generated(token[:, None], cache)
            cache.commit(result.delta, [1] * cache.batch_size)
            token = result.top1[:, -1]
            generated.append(token.cpu())
        return torch.stack(generated, dim=1)

    def generate_batch(
        self,
        prompts: Sequence[str | Sequence[int] | np.ndarray],
        *,
        max_new_tokens: int = 48,
        add_special_tokens: bool = False,
        eos_token_id: int | None = None,
        return_text: bool = False,
    ) -> list[list[int]] | list[str]:
        encoded: list[np.ndarray] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                row = self.encode([prompt], add_special_tokens=add_special_tokens)[0]
            else:
                row = np.asarray(prompt, dtype=np.int64)
            if row.ndim != 1 or not row.size:
                raise ValueError("each prompt must produce a non-empty token row")
            encoded.append(row)
        outputs: list[list[int] | None] = [None] * len(encoded)
        by_length: dict[int, list[int]] = {}
        for index, row in enumerate(encoded):
            by_length.setdefault(int(row.size), []).append(index)
        for indices in by_length.values():
            rows = np.stack([encoded[index] for index in indices])
            generated = self.generate_ids(rows, max_new_tokens=max_new_tokens).tolist()
            for index, tokens in zip(indices, generated, strict=True):
                outputs[index] = [int(token) for token in tokens]

        eos = eos_token_id
        if eos is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        finalized: list[list[int]] = []
        for tokens in outputs:
            if tokens is None:
                raise RuntimeError("generation output was not populated")
            if eos is not None and eos in tokens:
                tokens = tokens[: tokens.index(eos) + 1]
            finalized.append(tokens)
        if return_text:
            return [self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in finalized]
        return finalized

    def generate(
        self,
        prompt: str | list[int] | np.ndarray,
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        return_text: bool = False,
    ) -> list[int] | str:
        generated = self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            add_special_tokens=add_special_tokens,
            return_text=return_text,
        )
        return generated[0]

    @torch.inference_mode()
    def parity_check(self, input_ids: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Compare packed execution with the established B1 contract over one QStore."""

        rows = torch.as_tensor(input_ids, dtype=torch.long)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.ndim != 2 or not rows.shape[1]:
            raise ValueError("input_ids must have shape [tokens] or [batch,tokens]")
        packed_cache = self.target.empty_cache(int(rows.shape[0]))
        packed = self.target.forward(rows, packed_cache)
        reference_target = DenseQStoreTarget(
            self.store,
            max_seq_len=self.max_seq_len,
            stable_reductions=False,
            experimental_reranked_head=False,
            semantic_token_count=self.semantic_token_count,
        )
        scalar_hidden: list[torch.Tensor] = []
        scalar_top1: list[torch.Tensor] = []
        for row in rows:
            result = reference_target.forward(row, reference_target.empty_cache(1))
            scalar_hidden.append(result.hidden)
            scalar_top1.append(result.top1)
        expected_hidden = torch.cat(scalar_hidden, dim=0)
        expected_top1 = torch.cat(scalar_top1, dim=0)
        difference = packed.hidden.float() - expected_hidden.float()
        relative_l2 = float(
            (difference.norm() / expected_hidden.float().norm().clamp_min(1e-12)).item()
        )
        return {
            "scope": "same-qstore packed-vs-established-B1 gate",
            "batch_size": int(rows.shape[0]),
            "token_count": int(rows.shape[1]),
            "candidate_stable_reductions": self.target.stable_reductions,
            "candidate_reranked_head": self.target.experimental_reranked_head,
            "top1_exact": bool(torch.equal(packed.top1, expected_top1)),
            "hidden_max_abs": float(difference.abs().max().item()),
            "hidden_relative_l2": relative_l2,
            "canonical_hf_reference": False,
        }

    def runtime_stats(self) -> dict[str, Any]:
        cache = self._last_cache
        graph_executors = tuple(getattr(self, "_cuda_graph_executors", ()))
        component_snapshot = (
            self.composite_store.snapshot() if self.composite_store is not None else None
        )
        resident_exact_head = bool(
            component_snapshot and component_snapshot.get("resident_exact_head", False)
        )
        return {
            "backend": self.backend,
            "experimental": True,
            "numerical_contract": self.numerical_contract,
            "reduction_backend": (
                "row-stable-triton-v1"
                if self.target.stable_reductions
                else "torch-batched-established"
            ),
            "row_stable_gate": (
                "passed-job-4ae1fd1dc59c-mx-c8d0a7cc99fc25ea-119880-of-119880-"
                "tokens-b128-123.13x-b1024-521.41x-torch-trace-487-of-544"
                if self.target.stable_reductions
                else "not-selected-default-torch-contract"
            ),
            "decode_attention": {
                "mode": self.target.decode_attention_mode,
                "abi": self.target.decode_attention_mode,
                "sequence_tile": self.target.decode_attention_tile,
                "segmented_decode_layer_calls": self.target.segmented_decode_attention_calls,
                "established_prefill_layer_calls": (
                    self.target.segmented_decode_prefill_fallback_calls
                ),
                "committed_and_new_kv_segments": (
                    self.target.decode_attention_mode == _SEGMENTED_FLASH_GQA_DECODE_MODE
                ),
                "fixed_slot_row_mapping": (
                    self.target.decode_attention_mode == _SEGMENTED_FLASH_GQA_DECODE_MODE
                ),
                "combined_kv_materialization": False
                if self.target.decode_attention_mode == _SEGMENTED_FLASH_GQA_DECODE_MODE
                else True,
            },
            "body_fusion": {
                "mode": self.target.body_fusion_mode,
                "abi": self.target.body_fusion_mode,
                "bf16_residual_boundary": (
                    self.target.body_fusion_mode == _RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE
                ),
                "fused_residual_rms_calls": self.target.fused_residual_rms_calls,
                "fused_gate_up_swiglu_calls": self.target.fused_gate_up_swiglu_calls,
                "paired_compact_gate_up": (
                    self.target.body_fusion_mode == _RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE
                ),
            },
            "head_backend": (
                "triton-top2-fp32-rerank-experimental"
                if self.target.experimental_reranked_head
                else "resident-exact-fp32-established-row-blocks"
                if resident_exact_head
                else "streamed-fp32-qstore-established"
            ),
            "residency_contract": (
                str(component_snapshot["residency_contract"])
                if self.composite_store is not None
                else "monolithic-compact-lru+streamed-exact-head"
            ),
            "semantic_token_count": self.semantic_token_count,
            "configured_vocab_rows": int(self.cfg["vocab_size"]),
            "component_output_contract": self.component_output_contract,
            "component_graph": component_snapshot,
            "token_domain": {
                "validated_untrusted_calls": self.target.token_domain_checks,
                "trusted_generated_token_bypasses": (self.target.trusted_generated_token_bypasses),
                "trusted_path": "internal-semantic-masked-greedy-only",
            },
            "head_calls": {
                "resident_exact": self.target.resident_exact_head_calls,
                "streamed_exact": self.target.streamed_exact_head_calls,
                "compact_reranked": self.target.reranked_head_calls,
                "compact_reranked_working_bytes_peak": (
                    self.target.reranked_head_working_bytes_peak
                ),
                "compact_reranked_semantic_rows": (
                    self.semantic_token_count if self.target.experimental_reranked_head else 0
                ),
            },
            "reranked_head_gate": (
                "passed-bounded-mx-6bca93b32632325b-48of48-real-qwen-"
                "12.86x-head-top2-not-universal-proof"
            ),
            "compact_store": (
                self.store.stats_snapshot() if self.composite_store is None else component_snapshot
            ),
            "reductions": dict(sorted(self.target.reduction_calls.items())),
            "kv_allocated_bytes": cache.allocated_bytes if cache is not None else 0,
            "kv_committed_bytes": cache.committed_bytes if cache is not None else 0,
            "kv_epoch": cache.epoch if cache is not None else None,
            "cuda_graph": {
                "active_executors": len(graph_executors),
                "replay_count": sum(executor.replay_count for executor in graph_executors),
                "capture_resident_bytes": sum(
                    int(executor.evidence["capture_resident_bytes"]) for executor in graph_executors
                ),
            },
        }

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            logits=True,
            logits_batch=True,
            mlp_acts=False,
            approximate_quantized=True,
            generation=True,
            generation_batch=True,
            persistent_kv=True,
            transactional_kv=True,
            speculative_blocks=True,
            compact_fused_weights=True,
            compiled_workplan=True,
            graph_replay=getattr(self, "composite_store", None) is None,
            resident_graph_rebinding=getattr(self, "composite_store", None) is None,
            profile_driven_fusion=(
                getattr(getattr(self, "target", None), "body_fusion_mode", None)
                == _RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE
            ),
        )

    def build_work_plan(self, ids_list: list[np.ndarray], **kwargs: Any):
        """Build a typed static plan for this already-open dense QStore target."""

        from ..compiler import build_dense_qstore_plan

        return build_dense_qstore_plan(self, ids_list, **kwargs)

    def assert_content_identity_unchanged(self) -> None:
        """Fail a warm-pool hit if graph, payload, or tokenizer identity drifted."""

        self.store.assert_content_identity_unchanged()
        if self.composite_store is not None:
            self.composite_store.validate_tokenizer(self.tokenizer)

    def forward_acts(self, _ids: np.ndarray) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError("dense-qstore-cuda does not expose activation taps")

    def forward_patched(
        self,
        ids: np.ndarray,
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
        **_ignored: Any,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]:
        if patch_ops_by_layer or selected_maps or collect_acts:
            raise NotImplementedError(
                "dense-qstore-cuda does not expose activation capture or interventions"
            )
        return self.logits(ids), [], {}

    def down_weight(self, _layer: int) -> torch.Tensor:
        raise NotImplementedError("dense-qstore-cuda does not materialize analysis weights")

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        graph_executors = tuple(getattr(self, "_cuda_graph_executors", ()))
        for executor in graph_executors:
            executor.close()
        if hasattr(self, "_cuda_graph_executors"):
            self._cuda_graph_executors.clear()
        if self.composite_store is not None:
            self.composite_store.close()
        else:
            self.store.close()
        self._last_cache = None
        self._closed = True


class _CanonicalCudaSourceGraphFacade:
    """Tokenizer/source identity surface consumed by the native serving boundary."""

    def __init__(
        self,
        source: Any,
        *,
        model_name: str,
        tokenizer_descriptor: Mapping[str, Any],
        canonical_template_sha256: str,
    ) -> None:
        self.source = source
        self.model_name = model_name
        self.architecture = "qwen2"
        self.custody_fingerprint_sha256 = source.artifact_id
        self.declared_fingerprint_sha256 = source.artifact_id
        self.tokenizer_semantic_sha256 = str(tokenizer_descriptor["semantic_sha256"])
        self.semantic_token_count = int(tokenizer_descriptor["length"])
        self.tokenizer_canonical_sha256 = canonical_template_sha256
        self.raw = {
            "tokenizer": {
                "chat_template_sha256": tokenizer_descriptor["chat_template_sha256"],
                "canonical_chat_template_sha256": canonical_template_sha256,
                "hash_kind": "canonical-source-plus-runtime-assets",
            }
        }

    def assert_unchanged(self) -> None:
        from mrun.decompiler.emitter import open_component_artifact

        reopened = open_component_artifact(self.source.directory)
        if (
            reopened.artifact_id != self.source.artifact_id
            or reopened.manifest_sha256 != self.source.manifest_sha256
        ):
            raise RuntimeError("canonical source changed after direct CUDA open")

    def close(self) -> None:
        return None


class DenseSourceCudaInt8Engine(DenseQStoreCudaEngine):
    """Native CUDA executor lowered directly from canonical Qwen2 source components."""

    backend = "cuda-source-int8"
    NUMERICAL_CONTRACT = "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1"
    HEAD_EXECUTION_MODE = "exact-fp32-row-blocks"
    HEAD_EXECUTION_ABI = "exact-fp32-row-blocks-v1"

    def __init__(
        self,
        model_name: str,
        *,
        source_artifact: str | Path,
        native_artifact: str | Path | None = None,
        verified_native_artifact: Any | None = None,
        native_root: str | Path | None = None,
        device: str = "cuda",
        compute_dtype: str = "bf16",
        compact_cache_mb: float = 0.0,
        component_cache_mb: Mapping[str, float] | None = None,
        pin_component_fp32_aux: bool = True,
        resident_exact_head_mb: float | None = None,
        output_contract: Any = "full_logits",
        max_seq_len: int = 128,
        stable_block_m: int = 16,
        stable_reductions: bool = False,
        decode_attention_mode: str = _ESTABLISHED_DECODE_ATTENTION_MODE,
        decode_attention_tile: int = 64,
        body_fusion_mode: str = _ESTABLISHED_BODY_FUSION_MODE,
        experimental_reranked_head: bool = False,
        require_triton: bool = True,
        **_ignored: Any,
    ) -> None:
        from transformers import AutoTokenizer

        from mrun.decompiler.cuda_native import (
            SOURCE_CUDA_INT8_NATIVE_SCHEMA,
            VerifiedSourceCudaInt8Artifact,
            build_source_cuda_int8_artifact,
        )
        from mrun.decompiler.emitter import open_component_artifact
        from mrun.engine.kernels.source_cuda_int8 import DirectSourceCudaInt8Store
        from mrun.engine.mlx_component import (
            _canonical_json_bytes,
            _hash_regular_file,
            _runtime_tokenizer_descriptor,
            _sha256_bytes,
        )

        self._closed = False
        self.composite_store = None
        self.component_output_contract = "full_logits"
        self.direct_artifact = None
        self.source_component_artifact = None
        self.head_execution_mode = self.HEAD_EXECUTION_MODE
        self.head_execution_abi = self.HEAD_EXECUTION_ABI
        if torch.device(device).type != "cuda":
            raise ValueError("cuda-source-int8 requires a CUDA device")
        if compute_dtype != "bf16":
            raise ValueError(
                "cuda-source-int8 v1 is gated only for BF16 compute; FP16 is a separate contract"
            )
        decode_attention_mode, decode_attention_tile = _decode_attention_config(
            decode_attention_mode,
            decode_attention_tile,
        )
        body_fusion_mode = _body_fusion_config(body_fusion_mode)
        if body_fusion_mode == _RESIDUAL_RMS_SWIGLU_BODY_FUSION_MODE and not require_triton:
            raise RuntimeError("residual-rms-swiglu-v1 requires Triton execution")
        raw_output_contract = str(getattr(output_contract, "value", output_contract))
        if raw_output_contract not in {"full_logits", "last_token_logits", "loss_only"}:
            raise NotImplementedError("cuda-source-int8 supports only the full-logit contract")
        if stable_reductions:
            raise NotImplementedError(
                "cuda-source-int8 v1 has not gated the row-stable reduction contract"
            )
        compact_head = self.head_execution_mode == ("semantic-prefix-w8a16-top2-fp32-rerank")
        if experimental_reranked_head and not compact_head:
            raise NotImplementedError(
                "cuda-source-int8 v1 does not authorize the experimental reranked head"
            )
        if compact_head and not experimental_reranked_head:
            raise RuntimeError("compact direct CUDA head must explicitly select reranked execution")
        if compact_head and resident_exact_head_mb is not None:
            raise ValueError("compact direct CUDA head forbids an expanded resident FP32 head")
        if component_cache_mb is not None and not isinstance(component_cache_mb, Mapping):
            raise TypeError("component_cache_mb must map direct component roles to MiB budgets")
        if resident_exact_head_mb is not None and float(resident_exact_head_mb) <= 0:
            raise ValueError("resident_exact_head_mb must be positive")

        source = open_component_artifact(source_artifact)
        opened: DirectSourceCudaInt8Store | None = None
        try:
            if native_artifact is None:
                root = (
                    Path(native_root).expanduser()
                    if native_root is not None
                    else Path("~/.cache/mrun/cuda-source-int8").expanduser()
                )
                native_path = build_source_cuda_int8_artifact(source, root).path
            else:
                native_path = Path(native_artifact)
            if verified_native_artifact is None:
                artifact = VerifiedSourceCudaInt8Artifact(
                    native_path,
                    source_artifact=source,
                    verify_quantized_values=True,
                )
                self.native_artifact_verification_reused = False
            else:
                if not isinstance(verified_native_artifact, VerifiedSourceCudaInt8Artifact):
                    raise TypeError(
                        "verified_native_artifact must be a VerifiedSourceCudaInt8Artifact"
                    )
                artifact = verified_native_artifact
                if artifact.path != Path(native_path).expanduser().resolve():
                    raise ValueError("verified native artifact path differs from native_artifact")
                if (
                    artifact.source.get("artifact_id") != source.artifact_id
                    or artifact.source.get("manifest_sha256") != source.manifest_sha256
                ):
                    raise ValueError("verified native artifact belongs to a different source")
                artifact.assert_unchanged()
                self.native_artifact_verification_reused = True
            if artifact.manifest.get("schema") != SOURCE_CUDA_INT8_NATIVE_SCHEMA:
                raise RuntimeError("cuda-source-int8 received an incompatible native artifact")
            self.spec = resolve_model(model_name)
            self.name = self.spec.name
            accepted_model_ids = {
                str(model_name).lower(),
                str(self.spec.name).lower(),
                str(self.spec.hf_id).lower(),
            }
            if str(source.source.source_id).lower() not in accepted_model_ids:
                raise ValueError("requested model differs from the canonical source identity")
            if self.spec.family != "qwen2" or artifact.source["architecture"] != "qwen2":
                raise ValueError("cuda-source-int8 v1 requires a registered Qwen2 model")

            role_budgets = dict(component_cache_mb or {})
            if role_budgets and set(role_budgets) != set(artifact.components):
                raise ValueError(
                    "direct CUDA component budgets must cover every and only artifact role"
                )
            if any(float(value) <= 0 for value in role_budgets.values()):
                raise ValueError("direct CUDA component budgets must be positive")
            cache_mb = sum(float(value) for value in role_budgets.values()) or float(
                compact_cache_mb
            )
            concrete_device = str(_concrete_cuda_device(device))
            store = DirectSourceCudaInt8Store(
                artifact,
                source_artifact=source,
                device=concrete_device,
                compute_dtype=compute_dtype,
                compact_cache_mb=cache_mb,
                require_triton=require_triton,
                stable_block_m=stable_block_m,
                pin_fp32_aux=pin_component_fp32_aux,
            )
            opened = store
            self.store = store
            self.direct_artifact = artifact
            self.source_component_artifact = source

            self.tokenizer = AutoTokenizer.from_pretrained(
                source.directory / "assets",
                local_files_only=True,
                trust_remote_code=False,
            )
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            descriptor = _runtime_tokenizer_descriptor(self.tokenizer)
            io = source.ir_bundle.io
            for item in io.tokenizer_assets:
                digest, size, _file_identity = _hash_regular_file(
                    source.directory / "assets" / item.path,
                    expected_size=item.byte_count,
                    expected_sha256=item.sha256,
                )
                if digest != item.sha256 or size != item.byte_count:
                    raise RuntimeError("direct CUDA runtime tokenizer asset custody changed")
            selected_template = next(
                (item for item in io.chat_templates if item.template_id == "default"), None
            )
            chat_template = str(getattr(self.tokenizer, "chat_template", "") or "")
            canonical_template = _sha256_bytes(_canonical_json_bytes({"content": chat_template}))
            if selected_template is None or canonical_template != selected_template.sha256:
                raise RuntimeError("direct CUDA tokenizer template differs from source IO")
            if int(descriptor["length"]) > int(artifact.config["vocab_size"]):
                raise RuntimeError("direct CUDA tokenizer exceeds physical vocabulary rows")
            self.graph = _CanonicalCudaSourceGraphFacade(
                source,
                model_name=self.name,
                tokenizer_descriptor=descriptor,
                canonical_template_sha256=canonical_template,
            )
            self.semantic_token_count = int(descriptor["length"])
            self.cfg = store.cfg
            self.target = DenseQStoreTarget(
                store,
                max_seq_len=max_seq_len,
                stable_reductions=False,
                decode_attention_mode=decode_attention_mode,
                decode_attention_tile=decode_attention_tile,
                body_fusion_mode=body_fusion_mode,
                experimental_reranked_head=compact_head,
                semantic_token_count=self.semantic_token_count,
            )
            self.arch = "qwen2"
            self.device = str(self.target.device)
            self.n_layer = int(self.cfg["num_hidden_layers"])
            self.inter = int(self.cfg["intermediate_size"])
            self.hidden = int(self.cfg["hidden_size"])
            self.max_seq_len = int(max_seq_len)
            self.numerical_contract = _execution_numerical_contract(
                self.NUMERICAL_CONTRACT,
                decode_attention_mode,
                body_fusion_mode,
            )
            self.subset_head_numerical_contract = f"{self.numerical_contract}+selected-head-fp32-v1"
            self.supported_numerical_contracts = (
                self.numerical_contract,
                self.subset_head_numerical_contract,
            )
            self._last_cache = None
            self._cuda_graph_executors = weakref.WeakSet()
            if role_budgets:
                store.prepare_fully_resident()
            if resident_exact_head_mb is not None:
                store.prepare_resident_exact_head(
                    max_resident_bytes=int(float(resident_exact_head_mb) * 1e6)
                )
        except BaseException:
            if opened is not None:
                opened.close()
            self._closed = True
            raise

    def assert_content_identity_unchanged(self) -> None:
        if self._closed:
            raise RuntimeError("direct CUDA engine is closed")
        self.graph.assert_unchanged()
        self.store.assert_content_identity_unchanged()

    def runtime_stats(self) -> dict[str, Any]:
        report = super().runtime_stats()
        snapshot = self.store.snapshot()
        report.update(
            {
                "direct_from_canonical_source": True,
                "intermediate_qstore": False,
                "source_artifact_id": self.source_component_artifact.artifact_id,
                "native_artifact_sha256": self.direct_artifact.artifact_sha256,
                "native_artifact_verification_reused": (
                    self.native_artifact_verification_reused
                ),
                "direct_component_store": snapshot,
                "residency_contract": snapshot["residency_contract"],
                "head_execution_mode": self.head_execution_mode,
                "head_execution_abi": self.head_execution_abi,
                "head_backend": (
                    "resident-compact-semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"
                    if self.target.experimental_reranked_head
                    else "resident-exact-fp32-established-row-blocks"
                    if snapshot["resident_exact_head"]
                    else "streamed-fp32-direct-role-files-established"
                ),
                "head_claim_boundary": {
                    "greedy_only": self.target.experimental_reranked_head,
                    "full_logits_use_exact_fp32_diagnostic_path": (
                        self.target.experimental_reranked_head
                    ),
                    "universal_exactness_proven": (not self.target.experimental_reranked_head),
                },
            }
        )
        return report


class DenseSourceCudaInt8CompactHeadEngine(DenseSourceCudaInt8Engine):
    """Greedy-only direct CUDA candidate over the resident compact vocabulary head."""

    backend = "cuda-source-int8-compact-head"
    NUMERICAL_CONTRACT = (
        "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1+"
        "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"
    )
    HEAD_EXECUTION_MODE = "semantic-prefix-w8a16-top2-fp32-rerank"
    HEAD_EXECUTION_ABI = "semantic-prefix-w8a16-top2-fp32-rerank-greedy-v1"

    def __init__(
        self,
        model_name: str,
        *,
        resident_exact_head_mb: float | None = None,
        **kwargs: Any,
    ) -> None:
        if resident_exact_head_mb is not None:
            raise ValueError("compact direct CUDA head forbids an expanded resident FP32 head")
        if "experimental_reranked_head" in kwargs:
            raise TypeError("compact direct CUDA head owns its reranked execution selection")
        if kwargs.get("require_triton", True) is not True:
            raise ValueError("compact direct CUDA head requires Triton execution")
        super().__init__(
            model_name,
            resident_exact_head_mb=None,
            experimental_reranked_head=True,
            **kwargs,
        )
