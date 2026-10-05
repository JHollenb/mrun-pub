"""Demand-paged int8 linear weights for Diffusers denoisers.

The causal ``mrun.engine.paged`` runtime is deliberately architecture-specific: it
knows how to execute decoder blocks, KV caches, and token-row gathers.  A Diffusers
DiT does not have that ABI.  This module is the smaller, honest diffusion substrate:
it stores every 2-D checkpoint weight as a row-scaled int8 page and replaces matching
``torch.nn.Linear`` modules with a loader that materializes one page at a time.

The page is shared by all rows of one physical ``F.linear`` call, so a leading
Diffusers batch is genuinely fused at the projection seam.  Non-linear state, norms,
convolutions, and the scheduler remain owned by Diffusers.  The resulting contract is
``int8_weight_only_*`` rather than exact BF16 parity.

This is intentionally separate from the causal QStore schema.  A causal QStore can
execute a model because its manifest and engine agree on the whole decoder ABI; this
store only promises a paged linear-weight ABI and refuses to masquerade as a causal
model store.
"""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn

DIFFUSION_QSTORE_SCHEMA = "mrun-diffusion-qstore-int8-v1"
DIFFUSION_QSTORE_INTEGRITY_SCHEMA = "mrun-diffusion-qstore-page-integrity-sha256-v1"
DIFFUSION_QSTORE_CONTENT_SCHEMA = "mrun-diffusion-qstore-content-v1"
DIFFUSION_QSTORE_FILES = ("weights.i8", "scales.f32")
DIFFUSION_QSTORE_QUANTIZATION = {
    "codec": "symmetric-int8",
    "granularity": "per-output-channel",
    "scale_dtype": "float32",
    "zero_point": 0,
}
DIFFUSION_QSTORE_NUMERICAL_LANE = "int8_weight_only"


class DiffusionQStoreIntegrityError(RuntimeError):
    """An encoded QStore page does not match its manifest-bound digest."""


@dataclass(frozen=True, slots=True)
class _DiffusionPageSnapshot:
    """One copied encoded page used for both verification and materialization."""

    quantized: np.ndarray
    scales: np.ndarray
    shape: tuple[int, int]
    weight_bytes: int
    scale_bytes: int
    integrity_verified: bool


_ACTIVE_DIFFUSION_WEIGHT_LEASES: contextvars.ContextVar[tuple[DiffusionWeightPageLease, ...]] = (
    contextvars.ContextVar("mrun_active_diffusion_weight_leases", default=())
)


def _quantize_row_int8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a contiguous float32 [out, in] matrix row by row."""

    value = np.ascontiguousarray(weight, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"diffusion qstore weights must be 2-D, got {value.shape}")
    scale = np.max(np.abs(value), axis=1, keepdims=True) / np.float32(127.0)
    # A zero row has no representable information to preserve; scale 1 keeps its
    # dequantized value zero and avoids a NaN/inf page descriptor.
    scale[scale == 0] = 1.0
    quantized = np.rint(value / scale).clip(-127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized), np.ascontiguousarray(scale[:, 0], dtype=np.float32)


def _safetensor_files(component_dir: Path) -> list[Path]:
    files = sorted(path for path in component_dir.rglob("*.safetensors") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no safetensors found under diffusion component {component_dir}")
    return files


def _source_fingerprint(component_dir: Path, files: list[Path]) -> dict[str, Any]:
    """Bind the store to the source layout without rereading every multi-GB blob."""

    records = []
    for path in files:
        stat = path.stat()
        records.append(
            {
                "path": str(path.relative_to(component_dir)),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return {"component_dir": str(component_dir), "files": records}


def build(
    component_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    model_name: str,
    pipeline_class: str,
) -> Path:
    """Stream a Diffusers transformer component into a paged int8 store.

    Only 2-D tensors whose checkpoint key ends in ``.weight`` are written.  The
    runtime replaces only matching ``nn.Linear`` modules, so positional tables or
    other 2-D state are harmlessly available but never interpreted as a projection.
    The builder holds one source tensor plus one quantized tensor at a time.
    """

    from safetensors import safe_open

    component = Path(component_dir).resolve()
    out = Path(output_dir).resolve()
    files = _safetensor_files(component)
    if out.exists():
        manifest_path = out / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"refusing to overwrite incomplete diffusion QStore: {out}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError(f"existing diffusion QStore manifest is invalid: {out}")
        _validate_manifest(out, manifest)
        expected_source = _source_fingerprint(component, files)
        if (
            manifest.get("schema_version") != DIFFUSION_QSTORE_SCHEMA
            or manifest.get("model_name") != model_name
            or manifest.get("pipeline_class") != pipeline_class
            or manifest.get("source") != expected_source
        ):
            raise RuntimeError(
                f"existing diffusion QStore identity/source mismatch; refusing to reuse {out}"
            )
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
    blocks: dict[str, dict[str, Any]] = {}
    integrity_blocks: dict[str, dict[str, str]] = {}
    weight_offset = 0
    scale_offset = 0
    source_tensor_count = 0
    try:
        with (
            (temporary / "weights.i8").open("wb") as weight_file,
            (temporary / "scales.f32").open("wb") as scale_file,
        ):
            for source_file in files:
                with safe_open(str(source_file), framework="pt") as tensors:
                    for key in sorted(tensors.keys()):
                        if not key.endswith(".weight"):
                            continue
                        tensor = tensors.get_tensor(key)
                        if tensor.ndim != 2:
                            del tensor
                            continue
                        # Import torch only inside the model-backed builder.  The
                        # reader and package metadata remain import-light.
                        import torch

                        value = tensor.detach().to(dtype=torch.float32).contiguous().numpy()
                        del tensor
                        if key in blocks:
                            raise RuntimeError(f"duplicate diffusion qstore block {key!r}")
                        quantized, scales = _quantize_row_int8(value)
                        encoded_weight = quantized.tobytes(order="C")
                        encoded_scales = scales.tobytes(order="C")
                        weight_file.write(encoded_weight)
                        scale_file.write(encoded_scales)
                        blocks[key] = {
                            "kind": "qrow",
                            "shape": [int(value.shape[0]), int(value.shape[1])],
                            "w_off": weight_offset,
                            "w_len": int(quantized.nbytes),
                            "s_off": scale_offset,
                            "s_len": int(scales.nbytes),
                        }
                        integrity_blocks[key] = {
                            "weights_i8_sha256": hashlib.sha256(encoded_weight).hexdigest(),
                            "scales_f32_sha256": hashlib.sha256(encoded_scales).hexdigest(),
                        }
                        weight_offset += int(quantized.nbytes)
                        scale_offset += int(scales.nbytes)
                        source_tensor_count += 1
                        del value, quantized, scales
                print(
                    f"    {source_file.name}: blocks={len(blocks)}",
                    flush=True,
                )

        manifest = {
            "schema_version": DIFFUSION_QSTORE_SCHEMA,
            "model_name": model_name,
            "pipeline_class": pipeline_class,
            "component": "transformer",
            "dtype": "int8",
            "quantization": DIFFUSION_QSTORE_QUANTIZATION,
            "source": _source_fingerprint(component, files),
            "blocks": blocks,
            "integrity": {
                "schema_version": DIFFUSION_QSTORE_INTEGRITY_SCHEMA,
                "algorithm": "sha256",
                "blocks": integrity_blocks,
            },
            "stats": {
                "source_2d_weight_tensors": source_tensor_count,
                "weights_i8_bytes": weight_offset,
                "scales_f32_bytes": scale_offset,
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(out)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"  -> {out}", flush=True)
    return out


def _validate_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != DIFFUSION_QSTORE_SCHEMA:
        raise RuntimeError(
            f"expected diffusion QStore schema {DIFFUSION_QSTORE_SCHEMA!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    if manifest.get("dtype") != "int8":
        raise RuntimeError("diffusion QStore must declare dtype='int8'")
    blocks = manifest.get("blocks")
    if not isinstance(blocks, Mapping) or not blocks:
        raise RuntimeError("diffusion QStore has no block table")
    intervals = {name: [] for name in DIFFUSION_QSTORE_FILES}
    for name, raw in blocks.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise RuntimeError("diffusion QStore block names/descriptors are invalid")
        if raw.get("kind") != "qrow":
            raise RuntimeError(f"diffusion QStore block {name!r} is not a qrow")
        shape = raw.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape)
        ):
            raise RuntimeError(f"diffusion QStore block {name!r} has an invalid shape")
        out_rows, in_cols = shape
        w_off, w_len = raw.get("w_off"), raw.get("w_len")
        s_off, s_len = raw.get("s_off"), raw.get("s_len")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (w_off, s_off)
        ):
            raise RuntimeError(f"diffusion QStore block {name!r} has an invalid offset")
        if w_len != out_rows * in_cols or s_len != out_rows * 4:
            raise RuntimeError(f"diffusion QStore block {name!r} has an invalid byte length")
        intervals["weights.i8"].append((w_off, w_off + w_len, name))
        intervals["scales.f32"].append((s_off, s_off + s_len, name))
    for filename, ranges in intervals.items():
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"diffusion QStore file is missing or not regular: {filename}")
        previous = 0
        for start, end, name in sorted(ranges):
            if start != previous:
                raise RuntimeError(f"diffusion QStore {filename} has a gap/overlap at {name!r}")
            if end > path.stat().st_size:
                raise RuntimeError(f"diffusion QStore {filename} block {name!r} exceeds file size")
            previous = end
        if previous != path.stat().st_size:
            raise RuntimeError(f"diffusion QStore {filename} has unreferenced trailing bytes")

    if "integrity" not in manifest:
        return
    integrity = manifest["integrity"]
    if not isinstance(integrity, Mapping):
        raise RuntimeError("diffusion QStore integrity declaration must be an object")
    if integrity.get("schema_version") != DIFFUSION_QSTORE_INTEGRITY_SCHEMA:
        raise RuntimeError("diffusion QStore integrity schema is missing or unsupported")
    if integrity.get("algorithm") != "sha256":
        raise RuntimeError("diffusion QStore integrity algorithm must be sha256")
    integrity_blocks = integrity.get("blocks")
    if not isinstance(integrity_blocks, Mapping):
        raise RuntimeError("diffusion QStore integrity block table must be an object")
    if set(integrity_blocks) != set(blocks):
        missing = sorted(set(blocks) - set(integrity_blocks))
        extra = sorted(set(integrity_blocks) - set(blocks))
        raise RuntimeError(
            f"diffusion QStore integrity coverage mismatch: missing={missing} extra={extra}"
        )
    digest_fields = {"weights_i8_sha256", "scales_f32_sha256"}
    for name, raw in integrity_blocks.items():
        if not isinstance(raw, Mapping) or set(raw) != digest_fields:
            raise RuntimeError(f"diffusion QStore integrity descriptor for {name!r} is malformed")
        for field_name in sorted(digest_fields):
            digest = raw[field_name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(
                    f"diffusion QStore integrity digest {name!r}.{field_name} is malformed"
                )


def _identity_fingerprint(directory: Path, manifest: Mapping[str, Any]) -> str:
    """Fingerprint validated store metadata without rereading multi-GB page files."""

    identity = {
        "manifest": manifest,
        "files": {name: int((directory / name).stat().st_size) for name in DIFFUSION_QSTORE_FILES},
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_fingerprint(manifest: Mapping[str, Any]) -> str | None:
    """Stable encoded-content identity, available only with complete integrity metadata."""

    integrity = manifest.get("integrity")
    if not isinstance(integrity, Mapping):
        return None
    body = {
        "schema_version": DIFFUSION_QSTORE_CONTENT_SCHEMA,
        "qstore_schema_version": manifest["schema_version"],
        "dtype": manifest["dtype"],
        "quantization": manifest.get("quantization"),
        "blocks": {
            key: {
                "descriptor": dict(manifest["blocks"][key]),
                "integrity": dict(integrity["blocks"][key]),
            }
            for key in sorted(manifest["blocks"])
        },
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DiffusionWeightPageLease:
    """One exact ordered dynamic extent over pages in a ``DiffusionQStore``.

    The lease owns neither a store nor a persistent cache. A page may be retained
    after first demand only until the context exits. Repeated keys therefore express
    repeated ordered demand while avoiding a second physical transfer.
    """

    def __init__(
        self,
        store: DiffusionQStore,
        ordered_keys: tuple[str, ...],
        *,
        device: Any,
        dtype: Any,
        numerical_lane: str,
        require_integrity: bool,
        max_retained_device_bytes: int | None,
    ):
        import torch

        if numerical_lane != DIFFUSION_QSTORE_NUMERICAL_LANE:
            raise ValueError(
                "diffusion QStore leases only support the "
                f"{DIFFUSION_QSTORE_NUMERICAL_LANE!r} numerical lane"
            )
        if not ordered_keys:
            raise ValueError("diffusion QStore lease requires at least one ordered key")
        if require_integrity and not store.integrity_capable:
            raise DiffusionQStoreIntegrityError(
                "diffusion QStore lease requires page integrity, but this is a legacy store; "
                "use require_integrity=False only for explicitly non-authoritative execution"
            )
        if max_retained_device_bytes is not None and (
            isinstance(max_retained_device_bytes, bool)
            or not isinstance(max_retained_device_bytes, int)
            or max_retained_device_bytes <= 0
        ):
            raise ValueError("max_retained_device_bytes must be a positive integer or None")
        missing = [key for key in ordered_keys if not store.has(key)]
        if missing:
            raise KeyError(f"diffusion QStore lease has unknown store blocks: {missing!r}")
        self._store = store
        self._store_generation = store._generation
        self._store_fingerprint = store.identity_fingerprint
        self._content_fingerprint = store.content_fingerprint
        self._device = torch.device(device)
        self._dtype = dtype
        self._numerical_lane = numerical_lane
        self._require_integrity = bool(require_integrity)
        self._max_retained_device_bytes = max_retained_device_bytes
        self._ordered_keys = ordered_keys
        self._cursor = 0
        self._state = "new"
        self._token: contextvars.Token[tuple[DiffusionWeightPageLease, ...]] | None = None
        self._materialized: dict[str, Any] = {}
        self._materialized_bytes: dict[str, int] = {}
        self._demand_bytes = 0
        self._encoded_bytes_read = 0
        self._transfer_bytes = 0
        self._materializations = 0
        self._cache_hits = 0
        self._resident_bytes_current = 0
        self._resident_bytes_peak = 0
        self._integrity_verified_pages = 0
        self._integrity_verified_weight_bytes = 0
        self._integrity_verified_scale_bytes = 0
        self._integrity_verification_attempts = 0
        self._integrity_verification_failures = 0
        self._eviction_events: list[dict[str, Any]] = []

    @property
    def ordered_keys(self) -> tuple[str, ...]:
        return self._ordered_keys

    @property
    def store_fingerprint(self) -> str:
        return self._store_fingerprint

    @property
    def active(self) -> bool:
        return self._state == "active"

    def __enter__(self) -> DiffusionWeightPageLease:
        if self._state != "new":
            raise RuntimeError(f"diffusion QStore lease cannot enter from state {self._state!r}")
        self._store._ensure_open()
        if self._store_generation != self._store._generation:
            raise RuntimeError("diffusion QStore lease is stale")
        if (
            self._store.identity_fingerprint != self._store_fingerprint
            or self._store.content_fingerprint != self._content_fingerprint
        ):
            raise RuntimeError("diffusion QStore lease store fingerprint changed")
        active = _ACTIVE_DIFFUSION_WEIGHT_LEASES.get()
        self._token = _ACTIVE_DIFFUSION_WEIGHT_LEASES.set((*active, self))
        self._state = "active"
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._finish(require_complete=exc_type is None)
        return False

    def _ensure_current(self) -> None:
        if self._state != "active":
            raise RuntimeError(f"diffusion QStore lease is not active (state={self._state!r})")
        self._store._ensure_open()
        if self._store_generation != self._store._generation:
            raise RuntimeError("diffusion QStore lease is stale")
        if self not in _ACTIVE_DIFFUSION_WEIGHT_LEASES.get():
            raise RuntimeError("diffusion QStore lease is stale in this thread/context")

    def resolve(
        self,
        name: str,
        *,
        store: DiffusionQStore,
        device: Any,
        dtype: Any,
        numerical_lane: str = DIFFUSION_QSTORE_NUMERICAL_LANE,
    ) -> Any:
        """Resolve the next declared page, rejecting every binding mismatch."""

        import torch

        self._ensure_current()
        if (
            store is not self._store
            or store.identity_fingerprint != self._store_fingerprint
            or store.content_fingerprint != self._content_fingerprint
        ):
            raise RuntimeError("diffusion QStore lease store identity mismatch")
        if torch.device(device) != self._device:
            raise RuntimeError(
                f"diffusion QStore lease device mismatch: expected {self._device}, got {device}"
            )
        if dtype != self._dtype:
            raise RuntimeError(
                f"diffusion QStore lease dtype mismatch: expected {self._dtype}, got {dtype}"
            )
        if numerical_lane != self._numerical_lane:
            raise RuntimeError(
                "diffusion QStore lease numerical lane mismatch: "
                f"expected {self._numerical_lane!r}, got {numerical_lane!r}"
            )
        key = str(name)
        if self._cursor >= len(self._ordered_keys):
            raise RuntimeError(f"diffusion QStore lease has no remaining demand for {key!r}")
        expected = self._ordered_keys[self._cursor]
        if key != expected:
            raise RuntimeError(
                f"diffusion QStore lease demand mismatch at index {self._cursor}: "
                f"expected {expected!r}, got {key!r}"
            )

        cached = self._materialized.get(key)
        if cached is not None:
            block_bytes = self._store.block_stored_bytes(key)
            self._cursor += 1
            self._demand_bytes += block_bytes
            self._store._observe_demand(block_bytes)
            self._cache_hits += 1
            return cached

        page_device_bytes = self._store.block_materialized_bytes(key, dtype=self._dtype)
        budget = self._max_retained_device_bytes
        if budget is not None and page_device_bytes > budget:
            raise RuntimeError(
                f"diffusion QStore page {key!r} needs {page_device_bytes} retained device "
                f"bytes, exceeding lease budget {budget}"
            )
        if self._store.integrity_capable:
            self._integrity_verification_attempts += 1
        try:
            snapshot = self._store._page_snapshot(key, require_integrity=self._require_integrity)
        except DiffusionQStoreIntegrityError:
            self._integrity_verification_failures += 1
            raise
        # Integrity is checked against one copied raw snapshot before changing
        # the resident set. A corrupt incoming page therefore cannot evict a
        # previously verified resident page or advance successful demand state.
        self._evict_for_demand(page_device_bytes)
        result, weight_bytes, scale_bytes, transfer_bytes = self._store._materialize_snapshot(
            snapshot, device=self._device, dtype=self._dtype
        )
        block_bytes = weight_bytes + scale_bytes
        resident_bytes = int(result.numel()) * int(result.element_size())
        if resident_bytes != page_device_bytes:  # pragma: no cover - dtype/shape invariant
            raise RuntimeError("diffusion QStore materialized page byte estimate mismatch")
        self._cursor += 1
        self._demand_bytes += block_bytes
        self._store._observe_demand(block_bytes)
        self._store._observe_page_load(weight_bytes, scale_bytes, transfer_bytes)
        self._materialized[key] = result
        self._resident_bytes_current += resident_bytes
        self._materialized_bytes[key] = resident_bytes
        self._resident_bytes_peak = max(self._resident_bytes_peak, self._resident_bytes_current)
        self._encoded_bytes_read += weight_bytes + scale_bytes
        self._transfer_bytes += transfer_bytes
        self._materializations += 1
        if snapshot.integrity_verified:
            self._integrity_verified_pages += 1
            self._integrity_verified_weight_bytes += weight_bytes
            self._integrity_verified_scale_bytes += scale_bytes
        return result

    def _next_use_index(self, key: str) -> int | None:
        for index in range(self._cursor + 1, len(self._ordered_keys)):
            if self._ordered_keys[index] == key:
                return index
        return None

    def _evict_for_demand(self, incoming_bytes: int) -> None:
        budget = self._max_retained_device_bytes
        if budget is None:
            return
        while self._resident_bytes_current + incoming_bytes > budget:
            if not self._materialized:  # pragma: no cover - oversized page checked earlier
                raise RuntimeError("diffusion QStore retention budget has no evictable page")
            next_uses = {key: self._next_use_index(key) for key in self._materialized}
            victim = max(
                self._materialized,
                key=lambda key: (
                    next_uses[key] is None,
                    -1 if next_uses[key] is None else next_uses[key],
                    key,
                ),
            )
            evicted_bytes = self._materialized_bytes.pop(victim)
            del self._materialized[victim]
            self._resident_bytes_current -= evicted_bytes
            self._eviction_events.append(
                {
                    "demand_index": self._cursor,
                    "evicted_key": victim,
                    "evicted_device_bytes": evicted_bytes,
                    "next_use_index": next_uses[victim],
                }
            )

    def release(self) -> None:
        """Release all retained tensors; double and stale releases fail closed."""

        self._finish(require_complete=True)

    def _finish(self, *, require_complete: bool) -> None:
        if self._state != "active":
            raise RuntimeError(f"diffusion QStore lease cannot release from state {self._state!r}")
        active = _ACTIVE_DIFFUSION_WEIGHT_LEASES.get()
        if not active or active[-1] is not self:
            raise RuntimeError("diffusion QStore lease release order is stale")
        incomplete = self._cursor != len(self._ordered_keys)
        token = self._token
        self._materialized.clear()
        self._materialized_bytes.clear()
        self._resident_bytes_current = 0
        self._state = "released"
        self._token = None
        if token is None:  # pragma: no cover - internal invariant
            raise RuntimeError("diffusion QStore lease lost its context token")
        _ACTIVE_DIFFUSION_WEIGHT_LEASES.reset(token)
        if require_complete and incomplete:
            raise RuntimeError(
                "diffusion QStore lease released before consuming its exact demand: "
                f"consumed={self._cursor} declared={len(self._ordered_keys)}"
            )

    def telemetry(self) -> dict[str, Any]:
        declared_bytes = sum(self._store.block_stored_bytes(key) for key in self._ordered_keys)
        store_bytes = sum(
            int((self._store.directory / name).stat().st_size) for name in DIFFUSION_QSTORE_FILES
        )
        return {
            "schema_version": DIFFUSION_QSTORE_SCHEMA,
            "store_fingerprint": self.store_fingerprint,
            "content_fingerprint": self._content_fingerprint,
            "store_directory": str(self._store.directory),
            "numerical_lane": self._numerical_lane,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "ordered_keys": list(self._ordered_keys),
            "store_stored_bytes": store_bytes,
            "declared_demands": len(self._ordered_keys),
            "consumed_demands": self._cursor,
            "declared_demand_bytes": declared_bytes,
            "demand_bytes": self._demand_bytes,
            "encoded_bytes_read": self._encoded_bytes_read,
            "device_transfer_bytes": self._transfer_bytes,
            "materializations": self._materializations,
            "lease_cache_hits": self._cache_hits,
            "max_retained_device_bytes": self._max_retained_device_bytes,
            "retention_authority": (
                "bounded-belady-next-use"
                if self._max_retained_device_bytes is not None
                else "unbounded-discovery-only"
            ),
            "retention_evictions": len(self._eviction_events),
            "retention_eviction_events": list(self._eviction_events),
            "integrity_schema": (
                DIFFUSION_QSTORE_INTEGRITY_SCHEMA if self._store.integrity_capable else None
            ),
            "integrity_capable": self._store.integrity_capable,
            "integrity_required": self._require_integrity,
            "integrity_authority": (
                "sha256-page-verification-required"
                if self._require_integrity
                else "non-authoritative-explicit-opt-out"
            ),
            "store_integrity_status": self._store.integrity_status,
            "integrity_verification_attempts": self._integrity_verification_attempts,
            "integrity_verification_failures": self._integrity_verification_failures,
            "integrity_verified_pages": self._integrity_verified_pages,
            "integrity_verified_weight_bytes": self._integrity_verified_weight_bytes,
            "integrity_verified_scale_bytes": self._integrity_verified_scale_bytes,
            "integrity_verified_bytes": (
                self._integrity_verified_weight_bytes + self._integrity_verified_scale_bytes
            ),
            "residency_measurement_scope": "lease-owned-materialized-tensors",
            "measured_retained_device_bytes_current": self._resident_bytes_current,
            "measured_retained_device_bytes_peak": self._resident_bytes_peak,
            "state": self._state,
        }


class DiffusionQStore:
    """Memory-mapped diffusion qrow pages with per-call telemetry."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        expected_model_name: str | None = None,
        expected_pipeline_class: str | None = None,
    ):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError("diffusion QStore manifest must be an object")
        _validate_manifest(self.directory, manifest)
        if expected_model_name is not None and manifest.get("model_name") != expected_model_name:
            raise RuntimeError(
                f"diffusion QStore model mismatch: expected {expected_model_name!r}, "
                f"got {manifest.get('model_name')!r}"
            )
        if (
            expected_pipeline_class is not None
            and manifest.get("pipeline_class") != expected_pipeline_class
        ):
            raise RuntimeError(
                f"diffusion QStore pipeline mismatch: expected {expected_pipeline_class!r}, "
                f"got {manifest.get('pipeline_class')!r}"
            )
        self.manifest = manifest
        self.blocks = manifest["blocks"]
        self.integrity_capable = "integrity" in manifest
        self.identity_fingerprint = _identity_fingerprint(self.directory, manifest)
        self.content_fingerprint = _content_fingerprint(manifest)
        self.weights = np.memmap(self.directory / "weights.i8", dtype=np.int8, mode="r")
        self.scales = np.memmap(self.directory / "scales.f32", dtype=np.float32, mode="r")
        self._closed = False
        self._generation = 0
        self._stats_lock = threading.Lock()
        self._integrity_lock = threading.Lock()
        self._integrity_verified_keys: set[str] = set()
        self._integrity_failed_keys: set[str] = set()
        self._stats = {
            "page_loads": 0,
            "page_bytes_i8": 0,
            "scale_bytes": 0,
            "linear_calls": 0,
            "input_rows": 0,
            "demand_bytes": 0,
            "encoded_bytes_read": 0,
            "device_transfer_bytes": 0,
        }

    def has(self, name: str) -> bool:
        return str(name) in self.blocks

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("diffusion QStore is closed")

    @property
    def integrity_status(self) -> str:
        if not self.integrity_capable:
            return "legacy-no-integrity"
        with self._integrity_lock:
            if self._integrity_failed_keys:
                return "verification-failed"
            verified = len(self._integrity_verified_keys)
        if verified == 0:
            return "capable-unverified"
        if verified == len(self.blocks):
            return "fully-verified"
        return "partially-verified"

    def _record_integrity_success(self, key: str) -> None:
        with self._integrity_lock:
            self._integrity_failed_keys.discard(key)
            self._integrity_verified_keys.add(key)

    def _record_integrity_failure(self, key: str) -> None:
        with self._integrity_lock:
            self._integrity_verified_keys.discard(key)
            self._integrity_failed_keys.add(key)

    def block_stored_bytes(self, name: str) -> int:
        try:
            block = self.blocks[str(name)]
        except KeyError as exc:
            raise KeyError(f"diffusion QStore has no block {name!r}") from exc
        return int(block["w_len"]) + int(block["s_len"])

    def block_materialized_bytes(self, name: str, *, dtype: Any) -> int:
        import torch

        try:
            block = self.blocks[str(name)]
        except KeyError as exc:
            raise KeyError(f"diffusion QStore has no block {name!r}") from exc
        element_size = torch.empty((), dtype=dtype).element_size()
        return int(block["shape"][0]) * int(block["shape"][1]) * int(element_size)

    def lease(
        self,
        ordered_keys: Any,
        *,
        device: Any,
        dtype: Any,
        numerical_lane: str = DIFFUSION_QSTORE_NUMERICAL_LANE,
        require_integrity: bool = True,
        max_retained_device_bytes: int | None = None,
    ) -> DiffusionWeightPageLease:
        self._ensure_open()
        if isinstance(ordered_keys, (str, bytes)):
            raise TypeError("diffusion QStore lease ordered_keys must be a sequence of keys")
        keys = tuple(str(key) for key in ordered_keys)
        return DiffusionWeightPageLease(
            self,
            keys,
            device=device,
            dtype=dtype,
            numerical_lane=numerical_lane,
            require_integrity=require_integrity,
            max_retained_device_bytes=max_retained_device_bytes,
        )

    def current_lease(self) -> DiffusionWeightPageLease | None:
        for lease in reversed(_ACTIVE_DIFFUSION_WEIGHT_LEASES.get()):
            if lease._store is self and lease.active:
                return lease
        return None

    def weight(self, name: str, *, device: Any, dtype: Any) -> Any:
        """Load and dequantize exactly one projection page for one F.linear call."""

        snapshot = self._page_snapshot(name, require_integrity=self.integrity_capable)
        result, weight_bytes, scale_bytes, transfer_bytes = self._materialize_snapshot(
            snapshot, device=device, dtype=dtype
        )
        self._observe_demand(weight_bytes + scale_bytes)
        self._observe_page_load(weight_bytes, scale_bytes, transfer_bytes)
        return result

    def _page_snapshot(self, name: str, *, require_integrity: bool) -> _DiffusionPageSnapshot:
        """Copy once, verify that exact copy, then hand it to materialization."""

        self._ensure_open()
        key = str(name)
        try:
            block = self.blocks[key]
        except KeyError as exc:
            raise KeyError(f"diffusion QStore has no block {name!r}") from exc
        weight_start = int(block["w_off"])
        weight_end = weight_start + int(block["w_len"])
        scale_start = int(block["s_off"]) // 4
        scale_end = scale_start + int(block["s_len"]) // 4
        out_rows, in_cols = (int(value) for value in block["shape"])
        quantized = np.array(
            self.weights[weight_start:weight_end],
            dtype=np.int8,
            copy=True,
        )
        scales = np.array(
            self.scales[scale_start:scale_end],
            dtype=np.float32,
            copy=True,
        )
        integrity_verified = False
        if self.integrity_capable:
            expected = self.manifest["integrity"]["blocks"][key]
            measured_weight = hashlib.sha256(memoryview(quantized)).hexdigest()
            measured_scales = hashlib.sha256(memoryview(scales).cast("B")).hexdigest()
            if measured_weight != expected["weights_i8_sha256"]:
                self._record_integrity_failure(key)
                raise DiffusionQStoreIntegrityError(
                    f"diffusion QStore encoded weight integrity mismatch for {key!r}"
                )
            if measured_scales != expected["scales_f32_sha256"]:
                self._record_integrity_failure(key)
                raise DiffusionQStoreIntegrityError(
                    f"diffusion QStore encoded scale integrity mismatch for {key!r}"
                )
            self._record_integrity_success(key)
            integrity_verified = True
        elif require_integrity:
            raise DiffusionQStoreIntegrityError(
                "legacy diffusion QStore has no page integrity declaration"
            )
        return _DiffusionPageSnapshot(
            quantized=quantized.reshape(out_rows, in_cols),
            scales=scales,
            shape=(out_rows, in_cols),
            weight_bytes=int(block["w_len"]),
            scale_bytes=int(block["s_len"]),
            integrity_verified=integrity_verified,
        )

    @staticmethod
    def _materialize_snapshot(
        snapshot: _DiffusionPageSnapshot, *, device: Any, dtype: Any
    ) -> tuple[Any, int, int, int]:
        """Materialize only the already copied and, when required, verified bytes."""

        import torch

        q_device = torch.from_numpy(snapshot.quantized).to(device=device)
        scale_device = torch.from_numpy(snapshot.scales).to(device=device)
        result = q_device.to(dtype=torch.float32).mul_(scale_device[:, None])
        if dtype != torch.float32:
            result = result.to(dtype=dtype)
        weight_bytes = snapshot.weight_bytes
        scale_bytes = snapshot.scale_bytes
        transfer_bytes = weight_bytes + scale_bytes if torch.device(device).type != "cpu" else 0
        return result, weight_bytes, scale_bytes, transfer_bytes

    def _observe_page_load(self, weight_bytes: int, scale_bytes: int, transfer_bytes: int) -> None:
        encoded_bytes = int(weight_bytes) + int(scale_bytes)
        with self._stats_lock:
            self._stats["page_loads"] += 1
            self._stats["page_bytes_i8"] += int(weight_bytes)
            self._stats["scale_bytes"] += int(scale_bytes)
            self._stats["encoded_bytes_read"] += encoded_bytes
            self._stats["device_transfer_bytes"] += int(transfer_bytes)

    def _observe_demand(self, block_bytes: int) -> None:
        with self._stats_lock:
            self._stats["demand_bytes"] += int(block_bytes)

    def observe_linear_input(self, value: Any) -> None:
        with self._stats_lock:
            self._stats["linear_calls"] += 1
            rows = int(value.shape[0]) if getattr(value, "ndim", 0) >= 3 else 1
            self._stats["input_rows"] += rows

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def stats_since(self, before: Mapping[str, int]) -> dict[str, int]:
        after = self.stats()
        return {key: int(value) - int(before.get(key, 0)) for key, value in after.items()}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        for array in (self.weights, self.scales):
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()


class DiffusionPagedLinear(nn.Module):
    """Drop-in inference projection whose weight is a demand-loaded qrow page."""

    def __init__(
        self,
        source: Any,
        qstore: DiffusionQStore,
        block_name: str,
        *,
        lease_required: bool = False,
    ):
        if not isinstance(source, nn.Linear):
            raise TypeError("DiffusionPagedLinear source must be nn.Linear")
        super().__init__()
        self.qstore = qstore
        self.block_name = str(block_name)
        self.lease_required = bool(lease_required)
        self.numerical_lane = DIFFUSION_QSTORE_NUMERICAL_LANE
        self.in_features = int(source.in_features)
        self.out_features = int(source.out_features)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        self.weight_shape = (self.out_features, self.in_features)

    def forward(self, value: Any) -> Any:
        import torch.nn.functional as functional

        if value.shape[-1] != self.in_features:
            raise RuntimeError(
                f"paged diffusion linear {self.block_name!r} expected width "
                f"{self.in_features}, got {value.shape[-1]}"
            )
        self.qstore.observe_linear_input(value)
        if self.lease_required:
            lease = self.qstore.current_lease()
            if lease is None:
                raise RuntimeError(
                    f"paged diffusion linear {self.block_name!r} requires an active weight lease"
                )
            weight = lease.resolve(
                self.block_name,
                store=self.qstore,
                device=value.device,
                dtype=value.dtype,
                numerical_lane=self.numerical_lane,
            )
        else:
            weight = self.qstore.weight(
                self.block_name,
                device=value.device,
                dtype=value.dtype,
            )
        bias = self.bias
        if bias is not None and (bias.device != value.device or bias.dtype != value.dtype):
            bias = bias.to(device=value.device, dtype=value.dtype)
        return functional.linear(value, weight, bias)


def _set_child(root: Any, module_name: str, replacement: Any) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def install_paged_linears(
    denoiser: Any,
    qstore: DiffusionQStore,
    *,
    strict: bool = True,
    lease_required: bool = False,
) -> dict[str, Any]:
    """Replace every checkpoint-backed ``nn.Linear`` in a denoiser.

    ``strict=True`` is the runtime default: a partially lowered denoiser would
    silently mix resident BF16 and paged int8 arithmetic, making memory and
    numerical receipts ambiguous.
    """

    import torch
    from torch import nn

    coverage = inspect_linear_coverage(denoiser, qstore)
    missing = list(coverage["missing_blocks"])
    if strict and missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise RuntimeError(
            f"diffusion QStore did not cover {len(missing)} Linear modules: {preview}{suffix}"
        )
    replacements: list[str] = []
    for module_name, module in list(denoiser.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        block_name = f"{module_name}.weight" if module_name else "weight"
        if not qstore.has(block_name):
            continue
        descriptor = qstore.blocks[block_name]
        expected_shape = (int(module.out_features), int(module.in_features))
        if tuple(int(value) for value in descriptor["shape"]) != expected_shape:
            raise RuntimeError(
                f"diffusion QStore shape mismatch for {block_name!r}: "
                f"store={descriptor['shape']} module={expected_shape}"
            )
        _set_child(
            denoiser,
            module_name,
            DiffusionPagedLinear(module, qstore, block_name, lease_required=lease_required),
        )
        replacements.append(block_name)

    report = {
        "backend": "diffusion-qstore-paged-fused",
        "schema_version": DIFFUSION_QSTORE_SCHEMA,
        "paged_linear_modules": len(replacements),
        "resident_linear_modules": len(missing),
        "missing_blocks": missing,
        "strict": bool(strict),
        "lease_required": bool(lease_required),
        "numerical_lane": DIFFUSION_QSTORE_NUMERICAL_LANE,
    }
    denoiser._saturn_diffusion_qstore = qstore
    denoiser._saturn_diffusion_qstore_report = report
    # Keep a cheap torch-version check close to the lowering boundary.  The
    # runtime needs F.linear over arbitrary leading dimensions, supported by all
    # torch versions used by the worker, but an import failure should be explicit.
    if not isinstance(denoiser, torch.nn.Module):
        raise TypeError("diffusion qstore denoiser must be a torch module")
    return report


def inspect_linear_coverage(denoiser: Any, qstore: DiffusionQStore) -> dict[str, Any]:
    """Compare a real/empty Diffusers denoiser's Linear ABI with a store."""

    from torch import nn

    missing: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    expected = 0
    for module_name, module in denoiser.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        expected += 1
        block_name = f"{module_name}.weight" if module_name else "weight"
        if not qstore.has(block_name):
            missing.append(block_name)
            continue
        shape = tuple(int(value) for value in qstore.blocks[block_name]["shape"])
        module_shape = (int(module.out_features), int(module.in_features))
        if shape != module_shape:
            shape_mismatches.append(
                {"block": block_name, "store": list(shape), "module": list(module_shape)}
            )
    if shape_mismatches:
        preview = shape_mismatches[0]
        raise RuntimeError(f"diffusion QStore Linear shape mismatch: {preview}")
    return {
        "expected_linear_modules": expected,
        "covered_linear_modules": expected - len(missing),
        "missing_blocks": missing,
        "store_blocks": len(qstore.blocks),
    }


def validate_component_coverage(
    component_dir: str | os.PathLike[str],
    qstore_dir: str | os.PathLike[str],
    *,
    pipeline_class: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Instantiate an empty Diffusers transformer and verify its Linear ABI."""

    import diffusers

    component = Path(component_dir).resolve()
    config = json.loads((component / "config.json").read_text(encoding="utf-8"))
    transformer_class_name = config.get("_class_name")
    if not isinstance(transformer_class_name, str):
        raise RuntimeError(f"{component}/config.json has no transformer _class_name")
    transformer_class = getattr(diffusers, transformer_class_name, None)
    if transformer_class is None:
        raise RuntimeError(f"installed diffusers has no {transformer_class_name}")
    try:
        from accelerate import init_empty_weights

        context = init_empty_weights()
    except ImportError:
        import torch

        context = torch.device("meta")
    with context:
        denoiser = transformer_class.from_config(config)
    store = DiffusionQStore(
        qstore_dir,
        expected_model_name=model_name,
        expected_pipeline_class=pipeline_class,
    )
    try:
        report = inspect_linear_coverage(denoiser, store)
        if report["missing_blocks"]:
            preview = ", ".join(report["missing_blocks"][:8])
            raise RuntimeError(
                "diffusion QStore coverage is incomplete "
                f"({len(report['missing_blocks'])} missing): {preview}"
            )
        return {
            **report,
            "component": str(component),
            "transformer_class": transformer_class_name,
            "pipeline_class": pipeline_class,
            "qstore": str(Path(qstore_dir).resolve()),
        }
    finally:
        store.close()


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build a Diffusers transformer diffusion QStore")
    parser.add_argument("component_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--pipeline-class", required=True)
    parser.add_argument(
        "--validate-component",
        action="store_true",
        help="instantiate an empty transformer and check every Linear against the store",
    )
    args = parser.parse_args()
    if args.validate_component:
        report = validate_component_coverage(
            args.component_dir,
            args.output_dir,
            pipeline_class=args.pipeline_class,
            model_name=args.model_name,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    output = build(
        args.component_dir,
        args.output_dir,
        model_name=args.model_name,
        pipeline_class=args.pipeline_class,
    )
    print(json.dumps({"output": str(output), "schema_version": DIFFUSION_QSTORE_SCHEMA}))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by mrun build jobs
    raise SystemExit(_main())
