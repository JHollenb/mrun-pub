"""Memory-mapped paged weight-store readers.

Stores share one reader contract:
- explicit lossless float32 (`kind="f32row"`): source values widened exactly to fp32.
- int8 row-scaled (`kind="qrow"`): one fp32 scale per output row.
- int4 group-wise (`kind="qrow4"`): packed two nibbles/byte, one fp32 scale per
  G-wide group of a row (G default 128).
- int3 group-wise (`kind="qrow3"`): packed 3-bit codes with chunk-fused matmul.
- int2 experimental ternary (`kind="qrow2"`): packed 2-bit codes with chunk-fused matmul.

`get`-style accessors (`weight`/`embed_rows`/`row_blocks`) dequantize ONE block to a
fresh fp32 tensor; callers free it after use, so resident heap stays O(largest single
matrix). The store *root* is injected (no module-global path) so the same reader serves
the engine's `stores/` dir and any other caller.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import stat
import time
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...device import resolve as _resolve_device
from ...store_provenance import (
    inspect_semantic_store_identity,
    verify_derived_provenance,
)
from .qstore_build import QSTORE_FILES, QSTORE_SCHEMA
from .qstore_fp32_build import (
    QSTORE_FP32_FILES,
    QSTORE_FP32_SCHEMA,
    QSTORE_FP32_STORAGE,
)

GROUP = 128


def _row_stable_loaded_matmul(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply one immutable weight with the exact request-local B=1 matmul shape.

    The leading axis is the request axis.  Each slice deliberately retains that singleton
    dimension, so the arithmetic selected for request ``b`` is the same operation as
    ``value[b:b+1] @ weight.T`` in an independent B=1 forward.  The caller is responsible for
    loading/dequantizing ``weight`` once; this helper never copies or mutates it.

    This trades a packed GEMM for B request-local GEMMs.  It is therefore a numerical-contract
    lane, not an unconditional speed optimization.
    """

    if not isinstance(value, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("row-stable matmul operands must be torch tensors")
    if value.ndim < 2:
        raise ValueError("row-stable matmul input must have a leading request axis")
    if weight.ndim != 2:
        raise ValueError("row-stable matmul weight must have shape [out, in]")
    if int(value.shape[0]) <= 0:
        raise ValueError("row-stable matmul request batch cannot be empty")
    if int(value.shape[-1]) != int(weight.shape[1]):
        raise ValueError("row-stable matmul input width does not match weight width")
    if value.device != weight.device:
        raise ValueError("row-stable matmul operands must share one device")
    return torch.cat(
        tuple(value[row : row + 1] @ weight.T for row in range(int(value.shape[0]))),
        dim=0,
    )


_QSTORE_WEIGHT_FILES = {
    "fp32": "weights.f32",
    "int8": "weights.i8",
    "int4": "weights.i4",
    "int3": "weights.i3",
    "int2": "weights.i2",
}


def inspect_qstore_manifest_identity(manifest: object) -> dict[str, Any]:
    """Return the cheap, self-validated identity status for a paged QStore manifest."""

    return inspect_semantic_store_identity(
        manifest,
        required_store_schema=(
            QSTORE_FP32_SCHEMA
            if isinstance(manifest, Mapping)
            and manifest.get("schema_version") == QSTORE_FP32_SCHEMA
            else QSTORE_SCHEMA
        ),
    )


def _required_int(block: Mapping[str, Any], key: str, *, positive: bool = False) -> int:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"QStore block field {key!r} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise RuntimeError(f"QStore block field {key!r} must be {qualifier}")
    return value


def _required_shape(block: Mapping[str, Any], *, dimensions: int | None = None) -> tuple[int, ...]:
    raw = block.get("shape")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("QStore block shape must be a non-empty list")
    shape = tuple(raw)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        raise RuntimeError("QStore block shape dimensions must be positive integers")
    if dimensions is not None and len(shape) != dimensions:
        raise RuntimeError(f"QStore block shape must have {dimensions} dimensions")
    return shape


def _validate_int8_layout(directory: Path, manifest: Mapping[str, Any]) -> None:
    """Validate every executable block mapping before opening any memory map."""

    if manifest.get("dtype") != "int8":
        raise RuntimeError("semantic QStore v3 must declare dtype='int8'")
    config = manifest.get("config")
    blocks = manifest.get("blocks")
    if not isinstance(config, Mapping) or not isinstance(blocks, Mapping) or not blocks:
        raise RuntimeError("semantic QStore v3 requires config and a non-empty block table")

    intervals: dict[str, list[tuple[int, int, str]]] = {
        "weights.i8": [],
        "scales.f32": [],
        "extras.f32": [],
    }
    qrow_count = fp32_count = 0
    for name, raw_block in blocks.items():
        if not isinstance(name, str) or not name or not isinstance(raw_block, Mapping):
            raise RuntimeError("QStore block names and descriptors must be non-empty mappings")
        if "alias" in raw_block:
            target = raw_block.get("alias")
            if not isinstance(target, str) or not target:
                raise RuntimeError(f"QStore alias {name!r} has an invalid target")
            continue

        kind = raw_block.get("kind")
        if kind == "qrow":
            output_rows, input_columns = _required_shape(raw_block, dimensions=2)
            weight_offset = _required_int(raw_block, "w_off")
            weight_length = _required_int(raw_block, "w_len", positive=True)
            scale_offset = _required_int(raw_block, "s_off")
            scale_length = _required_int(raw_block, "s_len", positive=True)
            if weight_length != output_rows * input_columns:
                raise RuntimeError(f"QStore qrow {name!r} weight length does not match its shape")
            if scale_length != output_rows * 4 or scale_offset % 4:
                raise RuntimeError(f"QStore qrow {name!r} scale range does not match its shape")
            intervals["weights.i8"].append((weight_offset, weight_offset + weight_length, name))
            intervals["scales.f32"].append((scale_offset, scale_offset + scale_length, name))
            qrow_count += 1
        elif kind == "fp32":
            shape = _required_shape(raw_block)
            offset = _required_int(raw_block, "e_off")
            length = _required_int(raw_block, "e_len", positive=True)
            if length != math.prod(shape) * 4 or offset % 4:
                raise RuntimeError(f"QStore fp32 {name!r} range does not match its shape")
            intervals["extras.f32"].append((offset, offset + length, name))
            fp32_count += 1
        else:
            raise RuntimeError(f"QStore block {name!r} has unsupported kind {kind!r}")

    if not qrow_count or not fp32_count:
        raise RuntimeError("semantic QStore v3 requires qrow and fp32 blocks")

    for name, raw_block in blocks.items():
        if not isinstance(raw_block, Mapping) or "alias" not in raw_block:
            continue
        seen = {name}
        target = raw_block["alias"]
        while True:
            if target not in blocks:
                raise RuntimeError(f"QStore alias {name!r} targets missing block {target!r}")
            if target in seen:
                raise RuntimeError(f"QStore alias cycle contains {target!r}")
            seen.add(target)
            target_block = blocks[target]
            if not isinstance(target_block, Mapping):
                raise RuntimeError(f"QStore alias {name!r} targets an invalid block")
            next_target = target_block.get("alias")
            if next_target is None:
                break
            if not isinstance(next_target, str) or not next_target:
                raise RuntimeError(f"QStore alias {target!r} has an invalid target")
            target = next_target

    for filename, ranges in intervals.items():
        path = directory / filename
        if not path.is_file():
            raise RuntimeError(f"QStore derived file is missing: {filename}")
        file_size = path.stat().st_size
        if not ranges:
            raise RuntimeError(f"QStore block table does not reference {filename}")
        previous_end = 0
        for start, end, block_name in sorted(ranges):
            if end > file_size:
                raise RuntimeError(
                    f"QStore {filename} range for block {block_name!r} exceeds file bounds"
                )
            if start < previous_end:
                raise RuntimeError(f"QStore {filename} ranges overlap at block {block_name!r}")
            if start != previous_end:
                raise RuntimeError(
                    f"QStore {filename} has an unreferenced gap before block {block_name!r}"
                )
            previous_end = end
        if previous_end != file_size:
            raise RuntimeError(f"QStore {filename} block ranges do not cover the complete file")


def _validate_fp32_layout(directory: Path, manifest: Mapping[str, Any]) -> None:
    """Validate every lossless-store range before opening either memory map."""

    if manifest.get("dtype") != "float32":
        raise RuntimeError("lossless QStore must declare dtype='float32'")
    if manifest.get("storage_contract") != QSTORE_FP32_STORAGE:
        raise RuntimeError("lossless QStore storage contract mismatch")
    source_dtypes = manifest.get("source_dtypes")
    accepted_source_dtypes = set(QSTORE_FP32_STORAGE["accepted_source_dtypes"])
    if (
        not isinstance(source_dtypes, list)
        or not source_dtypes
        or source_dtypes != sorted(set(source_dtypes))
        or any(dtype not in accepted_source_dtypes for dtype in source_dtypes)
    ):
        raise RuntimeError("lossless QStore source dtype evidence is invalid")
    config = manifest.get("config")
    blocks = manifest.get("blocks")
    if not isinstance(config, Mapping) or not isinstance(blocks, Mapping) or not blocks:
        raise RuntimeError("lossless QStore requires config and a non-empty block table")

    intervals: dict[str, list[tuple[int, int, str]]] = {
        "weights.f32": [],
        "extras.f32": [],
    }
    row_count = extras_count = 0
    for name, raw_block in blocks.items():
        if not isinstance(name, str) or not name or not isinstance(raw_block, Mapping):
            raise RuntimeError("QStore block names and descriptors must be non-empty mappings")
        if "alias" in raw_block:
            target = raw_block.get("alias")
            if not isinstance(target, str) or not target:
                raise RuntimeError(f"QStore alias {name!r} has an invalid target")
            continue
        kind = raw_block.get("kind")
        if kind == "f32row":
            output_rows, input_columns = _required_shape(raw_block, dimensions=2)
            offset = _required_int(raw_block, "w_off")
            length = _required_int(raw_block, "w_len", positive=True)
            if length != output_rows * input_columns * 4 or offset % 4:
                raise RuntimeError(
                    f"QStore f32row {name!r} range does not match its shape"
                )
            intervals["weights.f32"].append((offset, offset + length, name))
            row_count += 1
        elif kind == "fp32":
            shape = _required_shape(raw_block)
            offset = _required_int(raw_block, "e_off")
            length = _required_int(raw_block, "e_len", positive=True)
            if length != math.prod(shape) * 4 or offset % 4:
                raise RuntimeError(f"QStore fp32 {name!r} range does not match its shape")
            intervals["extras.f32"].append((offset, offset + length, name))
            extras_count += 1
        else:
            raise RuntimeError(f"QStore block {name!r} has unsupported kind {kind!r}")

    if not row_count or not extras_count:
        raise RuntimeError("lossless QStore requires f32row and fp32 blocks")

    for name, raw_block in blocks.items():
        if not isinstance(raw_block, Mapping) or "alias" not in raw_block:
            continue
        seen = {name}
        target = raw_block["alias"]
        while True:
            if target not in blocks:
                raise RuntimeError(f"QStore alias {name!r} targets missing block {target!r}")
            if target in seen:
                raise RuntimeError(f"QStore alias cycle contains {target!r}")
            seen.add(target)
            target_block = blocks[target]
            if not isinstance(target_block, Mapping):
                raise RuntimeError(f"QStore alias {name!r} targets an invalid block")
            next_target = target_block.get("alias")
            if next_target is None:
                break
            if not isinstance(next_target, str) or not next_target:
                raise RuntimeError(f"QStore alias {target!r} has an invalid target")
            target = next_target

    for filename, ranges in intervals.items():
        path = directory / filename
        if not path.is_file():
            raise RuntimeError(f"QStore derived file is missing: {filename}")
        file_size = path.stat().st_size
        if not ranges:
            raise RuntimeError(f"QStore block table does not reference {filename}")
        previous_end = 0
        for start, end, block_name in sorted(ranges):
            if end > file_size:
                raise RuntimeError(
                    f"QStore {filename} range for block {block_name!r} exceeds file bounds"
                )
            if start < previous_end:
                raise RuntimeError(f"QStore {filename} ranges overlap at block {block_name!r}")
            if start != previous_end:
                raise RuntimeError(
                    f"QStore {filename} has an unreferenced gap before block {block_name!r}"
                )
            previous_end = end
        if previous_end != file_size:
            raise RuntimeError(f"QStore {filename} block ranges do not cover the complete file")


def _load_qstore_manifest(
    directory: Path,
    *,
    expected_dtype: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("QStore manifest must be a JSON object")
    expected_schema = QSTORE_FP32_SCHEMA if expected_dtype == "fp32" else QSTORE_SCHEMA
    if manifest.get("schema_version") != expected_schema:
        # Compatibility: legacy and v2 stores still run, but their declared digests do not
        # bind the semantic block table and therefore cannot become promotion evidence.  The
        # FP32 lane has no legacy format and must always be content addressed.
        if expected_dtype == "fp32":
            raise RuntimeError(
                f"lossless QStore schema mismatch: expected {QSTORE_FP32_SCHEMA!r}"
            )
        return manifest, inspect_qstore_manifest_identity(manifest)
    if expected_dtype not in {"int8", "fp32"}:
        raise RuntimeError(f"unsupported semantic QStore dtype {expected_dtype!r}")

    identity = inspect_semantic_store_identity(
        manifest,
        required_store_schema=expected_schema,
        strict=True,
    )
    if expected_dtype == "fp32":
        _validate_fp32_layout(directory, manifest)
        filenames = QSTORE_FP32_FILES
    else:
        _validate_int8_layout(directory, manifest)
        filenames = QSTORE_FILES
    semantic_manifest = {key: value for key, value in manifest.items() if key != "derived"}
    verify_derived_provenance(
        directory,
        manifest.get("derived"),
        expected_filenames=filenames,
        semantic_manifest=semantic_manifest,
    )
    return (
        manifest,
        {
            **identity,
            "content_identity_verified": True,
            "identity_status": "content-addressed-semantic-v2-files-verified",
            "blob_identity_verified": True,
        },
    )


def _install_identity(instance: Any, identity: Mapping[str, Any]) -> None:
    instance.store_identity = dict(identity)
    instance.identity_status = str(identity["identity_status"])
    instance.content_identity_verified = bool(identity["content_identity_verified"])
    instance.source_checkpoint_sha256 = identity.get("source_checkpoint_sha256")
    instance.derived_store_sha256 = identity.get("derived_store_sha256")
    instance.manifest_semantic_sha256 = identity.get("manifest_semantic_sha256")


def _dequant_qrow(q: np.ndarray, sc: np.ndarray) -> torch.Tensor:
    """int8 [out,in] + fp32 row-scales [out] -> fp32 [out,in] torch tensor, on CPU.

    Measured 3.5x faster than the numpy ``q.astype(f32)*sc[:,None]`` it replaces
    (345ms->99ms over one Qwen2.5-0.5B forward's qrow blocks, 7950X/M-series both) and
    BIT-IDENTICAL (max|Δ|=0 over every block): int8->float32 widen is exact and the
    per-row multiply is IEEE-754 single either way. ``torch.from_numpy`` aliases the
    read-only memmap slice, but ``.to(float32)`` copies to a fresh writable tensor, so the
    in-place ``mul_`` never touches the mapped file. Dequant is ~72-79% of the CPU
    weight-step, so this is a ~2x lever on the whole CPU paged forward.
    """
    W = torch.from_numpy(q).to(torch.float32)
    scale = sc[:, None] if sc.ndim == 1 else sc[..., None]
    W.mul_(torch.from_numpy(scale))
    return W


def _drop_mmap_pages(array: Any, offset: int, length: int) -> None:
    """Release consumed file-backed pages without invalidating the mapped view.

    A dense paged forward touches every int8 source block once. Linux otherwise keeps
    those read-only mmap pages in process RSS until memory pressure arrives, making a
    27 GB QStore look resident even though the runtime only retains one dequantized
    block. ``MADV_DONTNEED`` drops the physical cache pages while preserving the mapping.
    Unsupported platforms/filesystems simply retain the old behavior.
    """

    if length <= 0:
        return
    handle = getattr(array, "_mmap", None)
    advise = getattr(handle, "madvise", None)
    if not callable(advise):
        return
    page_size = getattr(mmap, "PAGESIZE", 4096)
    start = max(0, int(offset) - (int(offset) % page_size))
    end = ((int(offset) + int(length) + page_size - 1) // page_size) * page_size
    try:
        advise(mmap.MADV_DONTNEED, start, end - start)
    except (AttributeError, OSError, ValueError):
        # Windows and some network-backed mappings reject this advice. The QStore remains
        # correct; only the RSS optimization is unavailable.
        return


def _dequant_int4(packed: np.ndarray, scales: np.ndarray, inn: int, G: int = GROUP) -> np.ndarray:
    """packed[out,rb] uint8 + scales[out,ng] -> fp32 [out,in]."""
    out, rb = packed.shape
    nib = np.empty((out, rb * 2), np.uint8)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = packed >> 4
    q = nib[:, :inn].astype(np.float32) - 8.0  # values -7..7
    # expand per-group scales to per-column (repeat each scale G times, clip to in)
    col_scale = np.repeat(scales, G, axis=1)[:, :inn]  # [out,in]
    return (q * col_scale).astype(np.float32)


def _dequant_int3(packed: np.ndarray, scales: np.ndarray, inn: int, G: int = GROUP) -> np.ndarray:
    """packed[out,row_bytes] uint8 + scales[out,ng] -> fp32 [out,in]."""
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, : inn * 3]
    codes = bits.reshape(packed.shape[0], inn, 3)
    q = (codes[:, :, 0] | (codes[:, :, 1] << 1) | (codes[:, :, 2] << 2)).astype(np.float32) - 4.0
    col_scale = np.repeat(scales, G, axis=1)[:, :inn]
    return (q * col_scale).astype(np.float32)


def _dequant_int2(packed: np.ndarray, scales: np.ndarray, inn: int, G: int = GROUP) -> np.ndarray:
    """packed[out,row_bytes] uint8 + scales[out,ng] -> fp32 ternary [out,in]."""
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, : inn * 2]
    codes = bits.reshape(packed.shape[0], inn, 2)
    q = (codes[:, :, 0] | (codes[:, :, 1] << 1)).astype(np.float32) - 1.0
    col_scale = np.repeat(scales, G, axis=1)[:, :inn]
    return (q * col_scale).astype(np.float32)


class _RingSlot:
    """One staging lane: reusable pinned host buffers + per-prefetch device tensors."""

    __slots__ = ("pinned_q", "pinned_s", "dev_q", "dev_s", "event", "future", "name", "shape")

    def __init__(self) -> None:
        self.pinned_q: torch.Tensor | None = None  # int8, grows to largest block
        self.pinned_s: torch.Tensor | None = None  # fp32, grows to largest row count
        self.dev_q: torch.Tensor | None = None
        self.dev_s: torch.Tensor | None = None
        self.event: torch.cuda.Event | None = None
        self.future: Future | None = None
        self.name: str | None = None
        self.shape: tuple[int, int] | None = None

    def release(self) -> None:
        self.dev_q = None
        self.dev_s = None
        # Keep the H2D event: the NEXT _stage on this slot must wait for it before
        # overwriting the pinned source (job-c5ac0f8d8aa8: recorded-but-incomplete H2D
        # read a reused pinned buffer -> bit-exactness lost under real hit rates).
        self.future = None
        self.name = None
        self.shape = None


class _RingPrefetch:
    """Successor-oracle pinned prefetch ring for the CUDA paged path.

    The plain CUDA path ships each block's mmap-backed pageable int8 codes + scales
    through a synchronous ``.to(device)``: the mmap page faults, the staging copy, and
    the H2D all serialize with compute. Dense forwards touch blocks in a static order
    (and ``generate`` repeats that order every token), so the ring learns each block's
    successor, stages the predicted next blocks into reusable pinned buffers on a
    worker thread, and issues async H2D on a dedicated copy stream while the caller's
    matmul runs. A hit consumes the identical int8 codes + fp32 scales through the
    identical widen/scale math -> bit-exact vs the plain path. A mispredict is a plain
    miss. Default ON for CUDA-granted stores since 2026-07-24 (``MRUN_PAGED_RING=0``
    disables, an integer sizes the lane count, ``QStore.enable_ring`` forces); zero
    behaviour change when off.
    """

    def __init__(self, store: QStore, *, slots: int = 3, depth: int = 2):
        if store.device == "cpu":
            raise RuntimeError("ring prefetch requires a CUDA-granted paged store")
        self.store = store
        self.depth = max(1, int(depth))
        self.torch_device = torch.device(store.device)
        self.stream = torch.cuda.Stream(device=self.torch_device)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qstore-ring")
        # np.copyto releases the GIL; chunked parallel memcpy lifts the measured
        # 17 GB/s single-thread pageable->pinned wall toward the ~34 GB/s bus figure
        # (decisive-probes job-527d9fe78306), letting the ring reach the PCIe link.
        self.copy_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="qstore-ring-cp")
        self.slots = [_RingSlot() for _ in range(max(2, int(slots)))]
        self.inflight: OrderedDict[str, _RingSlot] = OrderedDict()
        self.successor: dict[str, str] = {}
        self.last_access: str | None = None
        self.stats = {
            "hits": 0,
            "late_hits": 0,
            "misses": 0,
            "prefetches": 0,
            "staged": 0,  # _stage completions; prefetches >> staged means worker stall
            "prefetch_bytes": 0,
            "evicted_unused": 0,
            "successor_entries": 0,
        }

    # -- lifecycle -----------------------------------------------------------------
    def close(self) -> None:
        for slot in self.inflight.values():
            if slot.future is not None:
                slot.future.result()
            slot.release()
        self.inflight.clear()
        self.pool.shutdown(wait=True)
        self.copy_pool.shutdown(wait=True)

    # -- serving -------------------------------------------------------------------
    def take(self, name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return prefetched (dev int8 [out,in], dev fp32 scales [out]) or None."""
        slot = self.inflight.pop(name, None)
        if slot is None:
            self.stats["misses"] += 1
            return None
        assert slot.future is not None
        if slot.future.done():
            self.stats["hits"] += 1
        else:
            self.stats["late_hits"] += 1  # partial overlap still beats the plain path
        slot.future.result()  # staging + H2D issue complete
        assert slot.event is not None and slot.dev_q is not None and slot.dev_s is not None
        cur = torch.cuda.current_stream(self.torch_device)
        cur.wait_event(slot.event)
        # dev tensors were allocated on the ring stream but are consumed on `cur`:
        # without record_stream the caching allocator may recycle their memory for the
        # next ring-stream prefetch while the consumer matmul still reads it
        # (job-4e73ad6c2cd7: bit-exactness still lost after the pinned-drain fix).
        slot.dev_q.record_stream(cur)
        slot.dev_s.record_stream(cur)
        out, inn = slot.shape  # type: ignore[misc]
        q_dev = slot.dev_q.view(out, inn)
        s_dev = slot.dev_s
        slot.release()
        return q_dev, s_dev

    def observe(self, name: str) -> None:
        """Record access order and launch prefetches for the predicted successors."""
        if self.last_access is not None and self.last_access != name:
            self.successor[self.last_access] = name
            self.stats["successor_entries"] = len(self.successor)
        self.last_access = name
        nxt = name
        for _ in range(self.depth):
            nxt = self.successor.get(nxt)  # type: ignore[assignment]
            if nxt is None:
                return
            self._prefetch(nxt)

    # -- staging -------------------------------------------------------------------
    def _free_slot(self) -> _RingSlot | None:
        for slot in self.slots:
            if slot.name is None:
                return slot
        # evict the oldest completed, unconsumed prefetch (mispredict)
        for key, slot in list(self.inflight.items()):
            if slot.future is not None and slot.future.done():
                self.inflight.pop(key)
                slot.release()
                self.stats["evicted_unused"] += 1
                return slot
        return None

    def _prefetch(self, name: str) -> None:
        if name in self.inflight:
            return
        try:
            block = self.store._resolve(name)
        except KeyError:
            return
        if block.get("kind") != "qrow":
            return
        slot = self._free_slot()
        if slot is None:
            return
        slot.name = name
        out, inn = block["shape"]
        slot.shape = (int(out), int(inn))
        slot.future = self.pool.submit(self._stage, slot, block)
        self.inflight[name] = slot  # v1 omitted this: take() could never hit
        self.stats["prefetches"] += 1
        self.stats["prefetch_bytes"] += int(out) * int(inn) + int(out) * 4

    def preallocate(self, max_codes: int, max_rows: int) -> None:
        """Allocate every slot's pinned buffers on the CALLING (main) thread.
        cudaHostAlloc from the worker thread while the main thread runs CUDA ops was
        observed to hang the first _stage (job-29ec302e0a22: 3 prefetches — one hung,
        two queued — zero completions). Buffers are sized once from the manifest."""
        for slot in self.slots:
            slot.pinned_q = torch.empty(max_codes, dtype=torch.int8, pin_memory=True)
            slot.pinned_s = torch.empty(max_rows, dtype=torch.float32, pin_memory=True)

    def _stage(self, slot: _RingSlot, block: Mapping[str, Any]) -> None:
        if slot.event is not None:
            slot.event.synchronize()  # prior H2D from this pinned buffer must drain
        out, inn = slot.shape  # type: ignore[misc]
        n = out * inn
        if slot.pinned_q is None or slot.pinned_q.numel() < n:
            slot.pinned_q = torch.empty(n, dtype=torch.int8, pin_memory=True)
        if slot.pinned_s is None or slot.pinned_s.numel() < out:
            slot.pinned_s = torch.empty(out, dtype=torch.float32, pin_memory=True)
        w_off = block["w_off"]
        s_off = block["s_off"] // 4
        # mmap fault + memcpy into pinned staging happens HERE, on the worker thread,
        # overlapping the caller's queued GPU work. Large blocks copy in parallel
        # chunks (numpy releases the GIL during the memcpy).
        dst = slot.pinned_q[:n].numpy()
        src = np.asarray(self.store.w[w_off : w_off + n], dtype=np.int8)
        chunk = 8 * 1024 * 1024
        if n > 2 * chunk:
            spans = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
            futures = [self.copy_pool.submit(np.copyto, dst[a:b], src[a:b]) for a, b in spans]
            for fut in futures:
                fut.result()
        else:
            np.copyto(dst, src)
        np.copyto(
            slot.pinned_s[:out].numpy(),
            np.asarray(self.store.s[s_off : s_off + out], dtype=np.float32),
        )
        _drop_mmap_pages(self.store.w, w_off, n)
        _drop_mmap_pages(self.store.s, block["s_off"], out * 4)
        with torch.cuda.stream(self.stream):
            slot.dev_q = torch.empty(n, dtype=torch.int8, device=self.torch_device)
            slot.dev_q.copy_(slot.pinned_q[:n], non_blocking=True)
            slot.dev_s = torch.empty(out, dtype=torch.float32, device=self.torch_device)
            slot.dev_s.copy_(slot.pinned_s[:out], non_blocking=True)
            slot.event = torch.cuda.Event()
            slot.event.record(self.stream)
        self.stats["staged"] += 1


class QStore:
    """Memory-mapped paged store.

    The public constructor remains the established int8 lane.  ``QStoreFP32`` below selects
    the explicit lossless format through the private ``_storage_dtype`` hook; both readers
    expose one matrix at a time, keeping resident heap O(largest single matrix).
    """

    _DTYPES = {
        None: torch.float32,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }

    def __init__(
        self,
        model_name: str,
        *,
        root: Path,
        cache_mb: float = 0.0,
        compute_dtype: str | None = None,
        explicit_device: str | None = None,
        enable_prefetch_ring: bool = True,
        _storage_dtype: str = "int8",
    ):
        d = (Path(root) / model_name).resolve()
        if _storage_dtype not in {"int8", "fp32"}:
            raise ValueError(f"unsupported QStore storage dtype {_storage_dtype!r}")
        if _storage_dtype == "fp32" and compute_dtype not in (None, "fp32", "bf16", "fp16"):
            raise ValueError("lossless FP32 QStore compute dtype must be fp32, bf16, or fp16")
        self.storage_dtype = _storage_dtype
        self._configure_runtime_files(d, expected_dtype=_storage_dtype)
        self.man, identity = _load_qstore_manifest(d, expected_dtype=_storage_dtype)
        _install_identity(self, identity)
        initial_file_stats = self._content_file_stats()
        self.blocks = self.man["blocks"]
        self.cfg = self.man["config"]
        self.w = np.memmap(
            d / _QSTORE_WEIGHT_FILES[_storage_dtype],
            dtype=np.dtype("<f4") if _storage_dtype == "fp32" else np.int8,
            mode="r",
        )
        self.s = (
            None
            if _storage_dtype == "fp32"
            else np.memmap(d / "scales.f32", dtype=np.float32, mode="r")
        )
        self.e = np.memmap(
            d / "extras.f32",
            dtype=np.dtype("<f4") if _storage_dtype == "fp32" else np.float32,
            mode="r",
        )
        self.max_block_bytes = 0
        # Resident hot-set: an optional byte-budgeted LRU cache of dequantized fp32 (or
        # on-device) qrow weights. Dequant is ~72-79% of the CPU weight-step and is paid
        # AGAIN every forward — so a re-used weight (generate: one forward/token; repeated
        # scoring) re-dequants needlessly. With a budget the cache holds the whole int8
        # model resident (dense-resident speed) and LRU-evicts for models bigger than the
        # budget, degrading gracefully back to pure streaming. Cache hits are the SAME
        # tensor object -> bit-exact; default budget 0 = off = zero behaviour change.
        self.set_cache_budget(cache_mb)
        # Ordinary paged callers remain arch-gated by GATHER_DEVICE_PAGED. Explicit-device
        # adapters (DenseQStore) already validate their supported architecture and must not be
        # silently downgraded by an unrelated process-global paged allowlist. Passing an
        # explicit device also suppresses the base paged ring: that adapter owns a separate
        # compact-page cache and never consumes this ring.
        if explicit_device is None:
            dev, archs = _resolve_device("paged")
            arch = self.man.get("arch", "qwen2")
            self.device = dev if dev != "cpu" and arch in archs else "cpu"
            if dev != "cpu" and self.device == "cpu":
                # Correct but ~8x slower (measured 7087 vs 842 ms/fwd on qwen3-4b) — the
                # classic bare-`GATHER_DEVICE_PAGED=cuda` trap. Loud by design.
                print(
                    f"warn: paged device flag grants {dev!r} but arch {arch!r} is not in the "
                    f"allowlist {archs!r} -> cpu (~8x slower; set e.g. "
                    f'GATHER_DEVICE_PAGED="cuda qwen2 llama qwen3")',
                    flush=True,
                )
        else:
            requested = torch.device(explicit_device)
            if requested.type not in {"cpu", "cuda"}:
                raise ValueError("explicit QStore device must be cpu or cuda")
            if requested.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("explicit QStore CUDA execution requires an available GPU")
            if requested.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            self.device = str(requested)
        # Compute/storage dtype for dequantized qrow weights. Dequant math is always done in
        # fp32 (int8->fp32 widen is exact, the row-scale multiply is IEEE single); the RESULT
        # is then cast to compute_dtype. bf16 halves resident-cache bytes (so a 7B/14B fits the
        # card) AND runs the base projection matmuls on the 4080's tensor cores. None/"fp32"
        # keeps the prior bit-exact behaviour. Norms/biases (`fp32()`) stay fp32 for stability.
        self.compute_dtype = self._DTYPES[compute_dtype]
        # Successor-oracle pinned prefetch ring (CUDA only). MRUN_PAGED_RING=<slots>
        # sizes the lane count. Off = zero behaviour change; hits are bit-exact vs the
        # plain path (same codes, same widen/scale math).
        # PROMOTED 2026-07-24: ring defaults ON for CUDA-granted stores (bit-exact gate
        # passed, job-fd2a9bd444e6: 366 hits/893 late/253 miss, logits x1.12, stack
        # x3.92). MRUN_PAGED_RING=0 disables; an integer sets the lane count.
        ring_env = os.environ.get("MRUN_PAGED_RING", "").strip()
        if (
            _storage_dtype == "int8"
            and enable_prefetch_ring
            and self.device != "cpu"
            and ring_env != "0"
        ):
            self.enable_ring(slots=int(ring_env) if ring_env.isdigit() and int(ring_env) > 1 else 3)
        self._finish_runtime_file_guard(initial_file_stats)

    def _configure_runtime_files(self, directory: Path, *, expected_dtype: str) -> None:
        try:
            weight_filename = _QSTORE_WEIGHT_FILES[expected_dtype]
        except KeyError as exc:  # pragma: no cover - constructors pass a fixed known format
            raise ValueError(f"unsupported QStore dtype {expected_dtype!r}") from exc
        self.directory = directory
        self._expected_dtype = expected_dtype
        self._content_filenames = (
            (weight_filename, "extras.f32")
            if expected_dtype == "fp32"
            else (weight_filename, "scales.f32", "extras.f32")
        )
        self._ring: _RingPrefetch | None = None

    def _finish_runtime_file_guard(
        self,
        initial_file_stats: tuple[tuple[str, int, int, int, int, int, int], ...],
    ) -> None:
        current_file_stats = self._content_file_stats()
        if current_file_stats != initial_file_stats:
            self._invalidate_content_identity("verified-store-files-changed-during-open")
            self.close()
            raise RuntimeError("QStore files changed while the store was being opened")
        self._verified_file_stats = current_file_stats
        self.content_verified_at_ns = time.time_ns()

    def enable_ring(self, *, slots: int = 3, depth: int = 2) -> None:
        """Enable the pinned prefetch ring on a CUDA-granted store (idempotent)."""
        if self.storage_dtype != "int8":
            raise RuntimeError("ring prefetch is defined only for row-int8 QStores")
        if self.device == "cpu":
            raise RuntimeError("ring prefetch requires the CUDA paged path")
        if getattr(self, "_ring", None) is None:
            self._ring = _RingPrefetch(self, slots=slots, depth=depth)
            max_codes = max_rows = 1
            for raw in self.blocks.values():
                if isinstance(raw, Mapping) and raw.get("kind") == "qrow":
                    out, inn = raw["shape"]
                    max_codes = max(max_codes, int(out) * int(inn))
                    max_rows = max(max_rows, int(out))
            self._ring.preallocate(max_codes, max_rows)  # main thread: see preallocate()

    def disable_ring(self) -> None:
        ring = getattr(self, "_ring", None)
        if ring is not None:
            ring.close()
            self._ring = None

    def ring_stats(self) -> dict[str, int] | None:
        ring = getattr(self, "_ring", None)
        return dict(ring.stats) if ring is not None else None

    def ring_allocated_bytes(self) -> int:
        """Pinned+device capacity ceiling retained by the prefetch ring."""

        ring = getattr(self, "_ring", None)
        if ring is None:
            return 0
        total = 0
        for slot in ring.slots:
            pinned = 0
            for tensor in (slot.pinned_q, slot.pinned_s):
                if tensor is not None:
                    pinned += int(tensor.numel()) * int(tensor.element_size())
            device = 0
            for tensor in (slot.dev_q, slot.dev_s):
                if tensor is not None:
                    device += int(tensor.numel()) * int(tensor.element_size())
            # Device buffers are first-touch allocated, but admission must charge their
            # preallocated-shape ceiling even before the first prefetch.
            total += pinned + max(pinned, device)
        return total

    def close(self) -> None:
        """Idempotently release ring workers, resident tensors, and mmap handles."""

        self.disable_ring()
        self.set_cache_budget(0.0)
        for name in ("w", "s", "e"):
            array = getattr(self, name, None)
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
            setattr(self, name, None)

    def _content_file_stats(self) -> tuple[tuple[str, int, int, int, int, int, int], ...]:
        names = ("manifest.json", *self._content_filenames)
        records = []
        for name in names:
            path = self.directory / name
            try:
                file_stat = path.lstat()
            except OSError as exc:
                raise RuntimeError(f"QStore runtime file is unavailable: {name}") from exc
            if not stat.S_ISREG(file_stat.st_mode):
                kind = "symlink" if stat.S_ISLNK(file_stat.st_mode) else "non-regular file"
                raise RuntimeError(f"QStore runtime file must be regular, not a {kind}: {name}")
            records.append(
                (
                    name,
                    int(file_stat.st_dev),
                    int(file_stat.st_ino),
                    int(file_stat.st_mode),
                    int(file_stat.st_size),
                    int(file_stat.st_mtime_ns),
                    int(file_stat.st_ctime_ns),
                )
            )
        return tuple(records)

    def assert_content_identity_unchanged(self) -> None:
        """Cheaply reject a store whose verified files were replaced or modified."""

        try:
            current_file_stats = self._content_file_stats()
        except Exception as exc:
            self._invalidate_content_identity("verified-store-files-changed")
            raise RuntimeError("QStore files changed after content verification") from exc
        if current_file_stats != self._verified_file_stats:
            self._invalidate_content_identity("verified-store-files-changed")
            raise RuntimeError("QStore files changed after content verification")

    def _invalidate_content_identity(self, status: str) -> None:
        self.content_identity_verified = False
        self.store_identity["content_identity_verified"] = False
        self.store_identity["blob_identity_verified"] = False
        self.identity_status = status
        self.store_identity["identity_status"] = status

    def reverify_content_identity(self) -> dict[str, Any]:
        """Rehash every v3 blob immediately before promotion-grade evidence."""

        try:
            manifest, identity = _load_qstore_manifest(
                self.directory,
                expected_dtype=self._expected_dtype,
            )
        except Exception:
            self._invalidate_content_identity("fresh-content-verification-failed")
            raise
        if manifest != self.man:
            self._invalidate_content_identity("verified-store-manifest-changed")
            raise RuntimeError("QStore manifest changed after the store was opened")
        if (
            identity.get("content_identity_verified") is not True
            or identity.get("blob_identity_verified") is not True
        ):
            self._invalidate_content_identity("fresh-content-verification-failed")
            raise RuntimeError("QStore did not pass fresh semantic/blob verification")
        _install_identity(self, identity)
        self._verified_file_stats = self._content_file_stats()
        self.content_verified_at_ns = time.time_ns()
        return dict(identity)

    def _resolve(self, name: str) -> dict:
        key = str(name)
        seen: set[str] = set()
        while True:
            if key in seen:
                raise RuntimeError(f"cyclic QStore alias at {name!r}")
            seen.add(key)
            try:
                block = self.blocks[key]
            except KeyError as exc:
                raise KeyError(f"QStore has no block {key!r}") from exc
            alias = block.get("alias")
            if alias is None:
                return block
            key = str(alias)

    def has(self, name: str) -> bool:
        return name in self.blocks

    # ---- resident weight cache (byte-budgeted; admission-capped by default) -----------
    def set_cache_budget(self, cache_mb: float) -> None:
        """(Re)size the resident dequantized-weight cache. 0 disables + drops it.

        Policy (``MRUN_QSTORE_CACHE_POLICY``): ``pin-fill`` (default) inserts until the
        budget is full and NEVER evicts; ``lru`` restores the prior evicting cache.
        Measured 2026-07-24 (resident-knapsack PoC): the paged forward's per-token block
        sequence is exactly periodic, so LRU with budget < model dequant footprint gets
        LITERALLY 0 hit bytes (every block evicted before reuse), while any static
        resident set gets the budget fraction and is Belady-optimal (uniform reuse ⇒
        knapsack degenerates to fill). Isolated weight cycle measured 2.30x at 50%
        budget. Keep ``lru`` only for genuinely non-cyclic callers."""
        self._cache_budget = int(float(cache_mb) * 1e6)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_bytes = 0
        self._cache_policy = (
            os.environ.get("MRUN_QSTORE_CACHE_POLICY", "pin-fill").strip().lower() or "pin-fill"
        )
        if self._cache_policy not in ("pin-fill", "lru"):
            # A typo'd policy silently meaning pin-fill would strand callers who NEED
            # eviction (genuinely non-cyclic access) with a 0-hit cache and no signal.
            print(
                f"warn: MRUN_QSTORE_CACHE_POLICY={self._cache_policy!r} unknown "
                f"-> pin-fill (want pin-fill|lru)",
                flush=True,
            )
            self._cache_policy = "pin-fill"

    def _cache_get(self, name: str) -> torch.Tensor | None:
        if self._cache_budget <= 0:
            return None
        W = self._cache.get(name)
        if W is not None:
            self._cache.move_to_end(name)  # mark most-recently-used
        return W

    def _cache_put(self, name: str, W: torch.Tensor) -> None:
        if self._cache_budget <= 0:
            return
        nb = W.numel() * W.element_size()
        if self._cache_policy == "lru":
            if nb > self._cache_budget:
                return  # a single block bigger than the whole budget
            self._cache[name] = W
            self._cache.move_to_end(name)
            self._cache_bytes += nb
            while self._cache_bytes > self._cache_budget and len(self._cache) > 1:
                _, ev = self._cache.popitem(last=False)  # LRU evict
                self._cache_bytes -= ev.numel() * ev.element_size()
            return
        # pin-fill: admit while it fits, never evict (Belady-optimal on the periodic
        # forward; see set_cache_budget). Reinserting an existing key KEEPS the cached
        # tensor and drops the new one — a caller re-putting an UPDATED tensor under the
        # same key (none exists today) would serve stale weights; invalidate first.
        if name in self._cache:
            return
        if nb > self._cache_budget - self._cache_bytes:
            return
        self._cache[name] = W
        self._cache_bytes += nb

    def fp32(self, name: str) -> torch.Tensor:
        b = self._resolve(name)
        assert b["kind"] == "fp32", name
        n = int(np.prod(b["shape"]))
        off = b["e_off"] // 4
        # PyTorch cannot represent NumPy's read-only flag.  Aliasing the read-only mmap
        # therefore emits a warning and hands callers a tensor whose apparent mutability
        # would have undefined behaviour.  FP32 extras are small norms/biases; give the
        # tensor owned writable storage before exposing it to any backend.
        extras = np.array(
            self.e[off : off + n],
            dtype=np.float32,
            copy=True,
        ).reshape(b["shape"])
        _drop_mmap_pages(self.e, b["e_off"], n * 4)
        t = torch.from_numpy(extras)
        return t.to(self.device) if self.device != "cpu" else t

    def weight(self, name: str) -> torch.Tensor:
        """Load one full matrix.  The lossless lane copies fp32 bytes; int8 dequantizes."""
        b = self._resolve(name)
        assert b["kind"] in {"qrow", "f32row"}, name
        cached = self._cache_get(name)
        if cached is not None:
            return cached
        out, inn = b["shape"]
        if b["kind"] == "f32row":
            offset = b["w_off"] // 4
            values = np.array(
                self.w[offset : offset + out * inn],
                dtype=np.float32,
                copy=True,
            ).reshape(out, inn)
            weight = torch.from_numpy(values)
            if self.device != "cpu":
                weight = weight.to(self.device)
            if self.compute_dtype is not torch.float32:
                weight = weight.to(self.compute_dtype)
            self.max_block_bytes = max(
                self.max_block_bytes,
                weight.numel() * weight.element_size(),
            )
            self._cache_put(name, weight)
            return weight
        if self.device != "cpu" and self._ring is not None:
            hit = self._ring.take(name)
            self._ring.observe(name)
            if hit is not None:
                q_dev, sc_dev = hit
                W = q_dev.float() * sc_dev[:, None]  # identical widen/scale -> bit-exact
                if self.compute_dtype is not torch.float32:
                    W = W.to(self.compute_dtype)
                self.max_block_bytes = max(self.max_block_bytes, W.numel() * W.element_size())
                self._cache_put(name, W)
                return W
        q = np.asarray(self.w[b["w_off"] : b["w_off"] + out * inn], dtype=np.int8).reshape(out, inn)
        so = b["s_off"] // 4
        sc = np.asarray(self.s[so : so + out], dtype=np.float32)
        if self.device != "cpu":
            # ship int8 + row scales, dequant ON device (dequant = 72% of the cpu forward)
            W = (
                torch.from_numpy(q).to(self.device).float()
                * torch.from_numpy(sc).to(self.device)[:, None]
            )
        else:
            W = _dequant_qrow(q, sc)  # torch int8->f32 mul_: 3.5x the old numpy path
        _drop_mmap_pages(self.w, b["w_off"], out * inn)
        _drop_mmap_pages(self.s, b["s_off"], out * 4)
        if self.compute_dtype is not torch.float32:
            W = W.to(self.compute_dtype)  # fp32-accurate dequant, then narrow (bf16/fp16)
        self.max_block_bytes = max(self.max_block_bytes, W.numel() * W.element_size())
        self._cache_put(name, W)
        return W

    def matmul(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Compute ``x @ weight(name).T``. Quantized subclasses may fuse this."""
        weight = self.weight(name)
        # RMSNorm and bias paths intentionally remain FP32 for numerical stability.  A
        # narrowed compute-dtype store therefore receives FP32 activations at this seam;
        # torch.matmul rejects mixed floating dtypes instead of promoting them.  Match the
        # activation to the loaded weight once, preserving the store's declared arithmetic
        # contract while keeping the stable normalization above the projection in FP32.
        if x.dtype is not weight.dtype:
            x = x.to(dtype=weight.dtype)
        return x @ weight.T

    def matmul_row_stable(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Load/dequantize ``name`` once, then preserve independent B=1 arithmetic per row."""

        return _row_stable_loaded_matmul(x, self.weight(name))

    def embed_rows(self, name: str, ids: np.ndarray) -> torch.Tensor:
        """Load only requested embedding rows, preserving the store's numerical contract."""
        b = self._resolve(name)
        out, inn = b["shape"]
        if b["kind"] == "f32row":
            offset = b["w_off"] // 4
            flat = self.w[offset : offset + out * inn].reshape(out, inn)
            selected = np.array(flat[ids], dtype=np.float32, copy=True)
            _drop_mmap_pages(self.w, b["w_off"], out * inn * 4)
            rows = torch.from_numpy(selected)
            if self.device != "cpu":
                rows = rows.to(self.device)
            return rows if self.compute_dtype is torch.float32 else rows.to(self.compute_dtype)
        assert b["kind"] == "qrow", name
        wflat = np.asarray(self.w[b["w_off"] : b["w_off"] + out * inn], dtype=np.int8).reshape(
            out, inn
        )
        so = b["s_off"] // 4
        sc = np.asarray(self.s[so : so + out], dtype=np.float32)
        if self.device != "cpu":
            selected_weights = torch.from_numpy(wflat[ids]).to(self.device).float()
            selected_scales = torch.from_numpy(sc[ids]).to(self.device)
            while selected_scales.ndim < selected_weights.ndim:
                selected_scales = selected_scales.unsqueeze(-1)
            rows = selected_weights * selected_scales
        else:
            rows = _dequant_qrow(wflat[ids], sc[ids])
        _drop_mmap_pages(self.w, b["w_off"], out * inn)
        _drop_mmap_pages(self.s, b["s_off"], out * 4)
        return rows

    def row_blocks(self, name: str, bs: int = 8192):
        """Stream a matrix [out,in] in output-row chunks, bounding RSS.

        Chunks participate in the resident cache (key ``name[start:end]``): the lm_head
        re-streams every generated token. Cache hits return the SAME tensor bit-exact;
        budget 0 keeps pure streaming."""
        b = self._resolve(name)
        out, inn = b["shape"]
        scale_offset = b["s_off"] // 4 if b["kind"] == "qrow" else None
        for start in range(0, out, bs):
            end = min(start + bs, out)
            key = f"{name}[{start}:{end}]"
            cached = self._cache_get(key)
            if cached is not None:
                yield start, end, cached
                continue
            if b["kind"] == "f32row":
                offset = b["w_off"] // 4
                chunk = np.array(
                    self.w[offset + start * inn : offset + end * inn],
                    dtype=np.float32,
                    copy=True,
                ).reshape(end - start, inn)
                W = torch.from_numpy(chunk)
                if self.device != "cpu":
                    W = W.to(self.device)
                if self.compute_dtype is not torch.float32:
                    W = W.to(self.compute_dtype)
            else:
                assert b["kind"] == "qrow", name
                q = np.asarray(
                    self.w[b["w_off"] + start * inn : b["w_off"] + end * inn],
                    dtype=np.int8,
                ).reshape(end - start, inn)
                assert scale_offset is not None and self.s is not None
                sc = np.asarray(
                    self.s[scale_offset + start : scale_offset + end],
                    dtype=np.float32,
                )
                if self.device != "cpu":
                    W = (
                        torch.from_numpy(q).to(self.device).float()
                        * torch.from_numpy(sc).to(self.device)[:, None]
                    )
                else:
                    W = _dequant_qrow(q, sc)
            if self.compute_dtype is not torch.float32:
                W = W.to(self.compute_dtype)
            self.max_block_bytes = max(self.max_block_bytes, W.numel() * W.element_size())
            self._cache_put(key, W)
            if b["kind"] == "f32row":
                _drop_mmap_pages(self.w, b["w_off"] + start * inn * 4, (end - start) * inn * 4)
            else:
                _drop_mmap_pages(self.w, b["w_off"] + start * inn, (end - start) * inn)
                _drop_mmap_pages(self.s, b["s_off"] + start * 4, (end - start) * 4)
            yield start, end, W


class QStoreFP32(QStore):
    """Explicit lossless FP32 QStore with the ordinary paged-reader API."""

    def __init__(
        self,
        model_name: str,
        *,
        root: Path,
        cache_mb: float = 0.0,
        explicit_device: str | None = None,
        compute_dtype: str = "fp32",
        suffix: str = "-fp32",
    ) -> None:
        super().__init__(
            f"{model_name}{suffix}",
            root=root,
            cache_mb=cache_mb,
            compute_dtype=compute_dtype,
            explicit_device=explicit_device,
            enable_prefetch_ring=False,
            _storage_dtype="fp32",
        )


class QStoreInt4(QStore):
    """int4 group-wise store.

    The row-stable lane intentionally inherits the base full-weight implementation, matching
    existing Int4 serial arithmetic at an O(largest live matrix) fp32 working set.  Int3/Int2,
    whose established serial paths are chunked, override row-stable execution below.
    """

    def __init__(self, model_name: str, *, root: Path, suffix: str = "-int4"):
        d = (Path(root) / f"{model_name}{suffix}").resolve()
        self._configure_runtime_files(d, expected_dtype="int4")
        self.man, identity = _load_qstore_manifest(d, expected_dtype="int4")
        _install_identity(self, identity)
        initial_file_stats = self._content_file_stats()
        self.blocks = self.man["blocks"]
        self.cfg = self.man["config"]
        self.G = self.man["group_size"]
        self.w = np.memmap(d / "weights.i4", dtype=np.uint8, mode="r")
        self.s = np.memmap(d / "scales.f32", dtype=np.float32, mode="r")
        self.e = np.memmap(d / "extras.f32", dtype=np.float32, mode="r")
        self.max_block_bytes = 0
        self.device = "cpu"  # int4 dequant not device-ported (cuda gate is int8-only)
        self.set_cache_budget(0.0)  # int-N readers dequant per-chunk; cache attrs kept well-formed
        self._finish_runtime_file_guard(initial_file_stats)

    def weight(self, name: str) -> torch.Tensor:
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        packed = np.asarray(self.w[b["w_off"] : b["w_off"] + out * rb], dtype=np.uint8).reshape(
            out, rb
        )
        so = b["s_off"] // 4
        scales = np.asarray(self.s[so : so + out * ng], dtype=np.float32).reshape(out, ng)
        W = _dequant_int4(packed, scales, inn, self.G)
        self.max_block_bytes = max(self.max_block_bytes, W.nbytes)
        return torch.from_numpy(W)

    def embed_rows(self, name: str, ids: np.ndarray) -> torch.Tensor:
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        wflat = np.asarray(self.w[b["w_off"] : b["w_off"] + out * rb], dtype=np.uint8).reshape(
            out, rb
        )
        so = b["s_off"] // 4
        scales = np.asarray(self.s[so : so + out * ng], dtype=np.float32).reshape(out, ng)
        W = _dequant_int4(wflat[ids], scales[ids], inn, self.G)
        return torch.from_numpy(W)

    def row_blocks(self, name: str, bs: int = 8192):
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        so = b["s_off"] // 4
        for start in range(0, out, bs):
            end = min(start + bs, out)
            packed = np.asarray(
                self.w[b["w_off"] + start * rb : b["w_off"] + end * rb], dtype=np.uint8
            ).reshape(end - start, rb)
            scales = np.asarray(self.s[so + start * ng : so + end * ng], dtype=np.float32).reshape(
                end - start, ng
            )
            W = _dequant_int4(packed, scales, inn, self.G)
            self.max_block_bytes = max(self.max_block_bytes, W.nbytes)
            yield start, end, torch.from_numpy(W)


class QStoreInt3(QStore):
    """int3 group-wise store with chunk-fused unpack/dequant/matmul."""

    def __init__(self, model_name: str, *, root: Path, suffix: str = "-int3"):
        d = (Path(root) / f"{model_name}{suffix}").resolve()
        self._configure_runtime_files(d, expected_dtype="int3")
        self.man, identity = _load_qstore_manifest(d, expected_dtype="int3")
        _install_identity(self, identity)
        initial_file_stats = self._content_file_stats()
        self.blocks = self.man["blocks"]
        self.cfg = self.man["config"]
        self.G = self.man["group_size"]
        self.w = np.memmap(d / "weights.i3", dtype=np.uint8, mode="r")
        self.s = np.memmap(d / "scales.f32", dtype=np.float32, mode="r")
        self.e = np.memmap(d / "extras.f32", dtype=np.float32, mode="r")
        self.max_block_bytes = 0
        self.device = "cpu"  # int3 chunk-fused dequant is CPU/numpy by design
        self.set_cache_budget(0.0)
        self.matmul_chunk_rows = int(self.man.get("matmul_chunk_rows", 1024))
        self._finish_runtime_file_guard(initial_file_stats)

    def weight(self, name: str) -> torch.Tensor:
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        packed = np.asarray(self.w[b["w_off"] : b["w_off"] + out * rb], dtype=np.uint8).reshape(
            out, rb
        )
        so = b["s_off"] // 4
        scales = np.asarray(self.s[so : so + out * ng], dtype=np.float32).reshape(out, ng)
        W = _dequant_int3(packed, scales, inn, self.G)
        self.max_block_bytes = max(self.max_block_bytes, W.nbytes)
        return torch.from_numpy(W)

    def matmul(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Chunked ``x @ W.T`` without ever materializing the full fp32 int3 matrix."""
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        so = b["s_off"] // 4
        x2 = x.reshape(-1, inn).to(dtype=torch.float32)
        y = torch.empty((x2.shape[0], out), dtype=torch.float32, device=x2.device)
        bs = max(1, int(self.matmul_chunk_rows))
        for start in range(0, out, bs):
            end = min(start + bs, out)
            packed = np.asarray(
                self.w[b["w_off"] + start * rb : b["w_off"] + end * rb], dtype=np.uint8
            ).reshape(end - start, rb)
            scales = np.asarray(self.s[so + start * ng : so + end * ng], dtype=np.float32).reshape(
                end - start, ng
            )
            W = _dequant_int3(packed, scales, inn, self.G)
            self.max_block_bytes = max(self.max_block_bytes, W.nbytes)
            Wt = torch.from_numpy(W).to(device=x2.device)
            y[:, start:end] = x2 @ Wt.T
            del Wt, W
        return y.reshape(*x.shape[:-1], out)

    def matmul_row_stable(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Chunk-bounded row-stable matmul; every compact weight chunk is dequantized once."""

        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        so = b["s_off"] // 4
        if x.ndim < 2 or int(x.shape[0]) <= 0 or int(x.shape[-1]) != int(inn):
            raise ValueError("row-stable matmul input must have shape [B, ..., in]")
        y = torch.empty((*x.shape[:-1], out), dtype=torch.float32, device=x.device)
        x_fp32 = x.to(dtype=torch.float32)
        bs = max(1, int(self.matmul_chunk_rows))
        for start in range(0, out, bs):
            end = min(start + bs, out)
            packed = np.asarray(
                self.w[b["w_off"] + start * rb : b["w_off"] + end * rb], dtype=np.uint8
            ).reshape(end - start, rb)
            scales = np.asarray(self.s[so + start * ng : so + end * ng], dtype=np.float32).reshape(
                end - start, ng
            )
            weight = _dequant_int3(packed, scales, inn, self.G)
            self.max_block_bytes = max(self.max_block_bytes, weight.nbytes)
            weight_tensor = torch.from_numpy(weight).to(device=x.device)
            # QStoreInt3's independent B=1 contract flattens all non-feature axes before
            # matmul.  Retain that exact 2-D shape per request rather than using the base
            # QStore's rank-preserving ``torch.matmul`` contract.
            y[..., start:end] = torch.cat(
                tuple(
                    (x_fp32[row : row + 1].reshape(-1, inn) @ weight_tensor.T).reshape(
                        1, *x.shape[1:-1], end - start
                    )
                    for row in range(int(x.shape[0]))
                ),
                dim=0,
            )
            del weight_tensor, weight
        return y

    def embed_rows(self, name: str, ids: np.ndarray) -> torch.Tensor:
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        idx = np.asarray(ids, dtype=np.int64)
        flat = idx.reshape(-1)
        wflat = np.asarray(self.w[b["w_off"] : b["w_off"] + out * rb], dtype=np.uint8).reshape(
            out, rb
        )
        so = b["s_off"] // 4
        scales = np.asarray(self.s[so : so + out * ng], dtype=np.float32).reshape(out, ng)
        W = _dequant_int3(wflat[flat], scales[flat], inn, self.G)
        return torch.from_numpy(W.reshape(*idx.shape, inn))

    def row_blocks(self, name: str, bs: int = 8192):
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        so = b["s_off"] // 4
        for start in range(0, out, bs):
            end = min(start + bs, out)
            packed = np.asarray(
                self.w[b["w_off"] + start * rb : b["w_off"] + end * rb], dtype=np.uint8
            ).reshape(end - start, rb)
            scales = np.asarray(self.s[so + start * ng : so + end * ng], dtype=np.float32).reshape(
                end - start, ng
            )
            W = _dequant_int3(packed, scales, inn, self.G)
            self.max_block_bytes = max(self.max_block_bytes, W.nbytes)
            yield start, end, torch.from_numpy(W)


class QStoreInt2(QStore):
    """Experimental ternary int2 store with chunk-fused unpack/dequant/matmul."""

    def __init__(self, model_name: str, *, root: Path, suffix: str = "-int2"):
        d = (Path(root) / f"{model_name}{suffix}").resolve()
        self._configure_runtime_files(d, expected_dtype="int2")
        self.man, identity = _load_qstore_manifest(d, expected_dtype="int2")
        _install_identity(self, identity)
        initial_file_stats = self._content_file_stats()
        self.blocks = self.man["blocks"]
        self.cfg = self.man["config"]
        self.G = self.man["group_size"]
        self.w = np.memmap(d / "weights.i2", dtype=np.uint8, mode="r")
        self.s = np.memmap(d / "scales.f32", dtype=np.float32, mode="r")
        self.e = np.memmap(d / "extras.f32", dtype=np.float32, mode="r")
        self.max_block_bytes = 0
        self.device = "cpu"  # int2 chunk-fused dequant is CPU/numpy by design
        self.set_cache_budget(0.0)
        self.matmul_chunk_rows = int(self.man.get("matmul_chunk_rows", 1024))
        self._finish_runtime_file_guard(initial_file_stats)

    def weight(self, name: str) -> torch.Tensor:
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        packed = np.asarray(self.w[b["w_off"] : b["w_off"] + out * rb], dtype=np.uint8).reshape(
            out, rb
        )
        so = b["s_off"] // 4
        scales = np.asarray(self.s[so : so + out * ng], dtype=np.float32).reshape(out, ng)
        W = _dequant_int2(packed, scales, inn, self.G)
        self.max_block_bytes = max(self.max_block_bytes, W.nbytes)
        return torch.from_numpy(W)

    def matmul(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Chunked ``x @ W.T`` without materializing the full fp32 int2 matrix."""
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        so = b["s_off"] // 4
        x2 = x.reshape(-1, inn).to(dtype=torch.float32)
        y = torch.empty((x2.shape[0], out), dtype=torch.float32, device=x2.device)
        bs = max(1, int(self.matmul_chunk_rows))
        for start in range(0, out, bs):
            end = min(start + bs, out)
            packed = np.asarray(
                self.w[b["w_off"] + start * rb : b["w_off"] + end * rb], dtype=np.uint8
            ).reshape(end - start, rb)
            scales = np.asarray(self.s[so + start * ng : so + end * ng], dtype=np.float32).reshape(
                end - start, ng
            )
            W = _dequant_int2(packed, scales, inn, self.G)
            self.max_block_bytes = max(self.max_block_bytes, W.nbytes)
            Wt = torch.from_numpy(W).to(device=x2.device)
            y[:, start:end] = x2 @ Wt.T
            del Wt, W
        return y.reshape(*x.shape[:-1], out)

    def matmul_row_stable(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Chunk-bounded row-stable matmul; every compact weight chunk is dequantized once."""

        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        so = b["s_off"] // 4
        if x.ndim < 2 or int(x.shape[0]) <= 0 or int(x.shape[-1]) != int(inn):
            raise ValueError("row-stable matmul input must have shape [B, ..., in]")
        y = torch.empty((*x.shape[:-1], out), dtype=torch.float32, device=x.device)
        x_fp32 = x.to(dtype=torch.float32)
        bs = max(1, int(self.matmul_chunk_rows))
        for start in range(0, out, bs):
            end = min(start + bs, out)
            packed = np.asarray(
                self.w[b["w_off"] + start * rb : b["w_off"] + end * rb], dtype=np.uint8
            ).reshape(end - start, rb)
            scales = np.asarray(self.s[so + start * ng : so + end * ng], dtype=np.float32).reshape(
                end - start, ng
            )
            weight = _dequant_int2(packed, scales, inn, self.G)
            self.max_block_bytes = max(self.max_block_bytes, weight.nbytes)
            weight_tensor = torch.from_numpy(weight).to(device=x.device)
            # Match QStoreInt2's independent B=1 flatten-to-2-D arithmetic exactly.
            y[..., start:end] = torch.cat(
                tuple(
                    (x_fp32[row : row + 1].reshape(-1, inn) @ weight_tensor.T).reshape(
                        1, *x.shape[1:-1], end - start
                    )
                    for row in range(int(x.shape[0]))
                ),
                dim=0,
            )
            del weight_tensor, weight
        return y

    def embed_rows(self, name: str, ids: np.ndarray) -> torch.Tensor:
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        idx = np.asarray(ids, dtype=np.int64)
        flat = idx.reshape(-1)
        wflat = np.asarray(self.w[b["w_off"] : b["w_off"] + out * rb], dtype=np.uint8).reshape(
            out, rb
        )
        so = b["s_off"] // 4
        scales = np.asarray(self.s[so : so + out * ng], dtype=np.float32).reshape(out, ng)
        W = _dequant_int2(wflat[flat], scales[flat], inn, self.G)
        return torch.from_numpy(W.reshape(*idx.shape, inn))

    def row_blocks(self, name: str, bs: int = 8192):
        b = self._resolve(name)
        out, inn = b["shape"]
        rb = b["row_bytes"]
        ng = b["n_groups"]
        so = b["s_off"] // 4
        for start in range(0, out, bs):
            end = min(start + bs, out)
            packed = np.asarray(
                self.w[b["w_off"] + start * rb : b["w_off"] + end * rb], dtype=np.uint8
            ).reshape(end - start, rb)
            scales = np.asarray(self.s[so + start * ng : so + end * ng], dtype=np.float32).reshape(
                end - start, ng
            )
            W = _dequant_int2(packed, scales, inn, self.G)
            self.max_block_bytes = max(self.max_block_bytes, W.nbytes)
            yield start, end, torch.from_numpy(W)
