"""Runtime provider for direct-source, role-separated CUDA int8 components."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mrun.decompiler.cuda_native import VerifiedSourceCudaInt8Artifact
from mrun.decompiler.emitter import ComponentArtifact, open_component_artifact

from . import dense_qstore_cuda as dense_kernels
from .dense_qstore_cuda import CompactQRowPage, DenseQStore, DenseQStoreStats


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


class DirectSourceCudaInt8Store(DenseQStore):
    """Dense CUDA kernel provider backed only by a direct native component artifact.

    Inheritance is implementation reuse for the fused W8A16 kernels and their bounded device
    cache.  This constructor intentionally does not call ``QStore``/``DenseQStore.__init__``;
    no QStore manifest, aggregate QStore file, or QStore provenance participates in this lane.
    """

    def __init__(
        self,
        artifact: str | Path | VerifiedSourceCudaInt8Artifact,
        *,
        source_artifact: str | Path | ComponentArtifact,
        device: str = "cuda",
        compute_dtype: str = "bf16",
        compact_cache_mb: float = 0.0,
        require_triton: bool = True,
        stable_block_m: int = 16,
        pin_fp32_aux: bool = False,
    ) -> None:
        source = (
            source_artifact
            if isinstance(source_artifact, ComponentArtifact)
            else open_component_artifact(source_artifact)
        )
        verified = (
            artifact
            if isinstance(artifact, VerifiedSourceCudaInt8Artifact)
            else VerifiedSourceCudaInt8Artifact(artifact, source_artifact=source)
        )
        if isinstance(artifact, VerifiedSourceCudaInt8Artifact):
            if artifact.source.get("artifact_id") != source.artifact_id:
                raise RuntimeError("direct CUDA artifact belongs to another canonical source")
            if getattr(artifact, "_source_artifact", None) is None:
                # A verifier supplied by a caller may not have received a source. Reopen with
                # the source to establish deterministic code/scale custody at runtime.
                verified = VerifiedSourceCudaInt8Artifact(artifact.path, source_artifact=source)
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("direct-source CUDA execution requires an available CUDA device")
        if requested.type == "cuda" and require_triton and dense_kernels.triton is None:
            raise RuntimeError("direct-source CUDA execution requires Triton")
        if requested.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if compute_dtype not in {"bf16", "fp16"}:
            raise ValueError("direct CUDA compute_dtype must be bf16 or fp16")
        if stable_block_m not in {16, 32, 64}:
            raise ValueError("stable_block_m must be one of 16, 32, or 64")

        self.artifact = verified
        self.source_artifact = source
        self.directory = verified.path
        self.blocks = verified.blocks
        self.cfg = verified.config
        self.man = {
            "schema_version": verified.manifest["schema"],
            "arch": verified.source["architecture"],
            "config": self.cfg,
            "blocks": self.blocks,
            "direct_from_canonical_source": True,
            "intermediate_qstore": False,
        }
        self.device = str(requested)
        self.compute_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[compute_dtype]
        self.require_triton = bool(require_triton)
        self.stable_block_m = int(stable_block_m)
        self.max_block_bytes = 0
        self._ring = None
        self.compact_cache_budget = int(float(compact_cache_mb) * 1e6)
        self.compact_cache: OrderedDict[str, CompactQRowPage] = OrderedDict()
        self.compact_cache_bytes = 0
        # This adapter initializes the shared dense-store accounting contract
        # directly, including the optional prebound residency pools.
        self._prebound_pages: dict[str, CompactQRowPage] = {}
        self._prebound_compact_bytes = 0
        self._prebound_fp32: dict[str, torch.Tensor] = {}
        self._prebound_fp32_bytes = 0
        self.pin_fp32_aux = bool(pin_fp32_aux)
        self.fp32_aux_cache: dict[str, torch.Tensor] = {}
        self.fp32_aux_cache_bytes = 0
        self.resident_exact_heads: dict[str, torch.Tensor] = {}
        self.resident_exact_head_budget_bytes = 0
        self.resident_exact_head_bytes = 0
        self.stats = DenseQStoreStats()
        self._mapped: dict[tuple[str, str], np.memmap] = {}
        dtypes = {
            "weights.i8": np.int8,
            "scales.f32": np.float32,
            "extras.f32": np.float32,
        }
        try:
            for role, component in verified.components.items():
                for filename, record in component["blobs"].items():
                    self._mapped[(role, filename)] = np.memmap(
                        verified.path / record["path"], mode="r", dtype=dtypes[filename]
                    )
            if self.pin_fp32_aux:
                self._pin_all_fp32_auxiliary()
        except BaseException:
            self.close()
            raise

    def _array(self, block: dict[str, Any], filename: str) -> np.memmap:
        try:
            return self._mapped[(str(block["role"]), filename)]
        except KeyError as exc:
            raise RuntimeError(
                f"direct CUDA block has no role-local {filename!r} provider"
            ) from exc

    def _pin_all_fp32_auxiliary(self) -> None:
        physical_names = tuple(
            dict.fromkeys(
                self._physical_key(name)
                for name, block in self.blocks.items()
                if isinstance(block, dict) and self._resolve(name).get("kind") == "fp32"
            )
        )
        required = sum(int(self.blocks[name]["e_len"]) for name in physical_names)
        if required > self.compact_cache_budget:
            raise MemoryError(
                "pinned direct CUDA auxiliaries exceed the provider cache budget "
                f"({required} > {self.compact_cache_budget} bytes)"
            )
        cache = {name: self._load_fp32(name) for name in physical_names}
        resident = sum(_tensor_bytes(tensor) for tensor in cache.values())
        if resident != required:
            raise RuntimeError("direct CUDA auxiliary residency differs from verified layout")
        self.fp32_aux_cache = cache
        self.fp32_aux_cache_bytes = resident
        self.stats.fp32_aux_cache_loads += len(cache)
        self.stats.peak_compact_resident_bytes = max(
            self.stats.peak_compact_resident_bytes, resident
        )

    def _load_fp32(self, name: str) -> torch.Tensor:
        block = self._resolve(name)
        if block.get("kind") != "fp32":
            raise ValueError(f"{name!r} is not an FP32 auxiliary block")
        count = int(np.prod(block["shape"]))
        offset = int(block["e_off"]) // 4
        values = np.array(
            self._array(block, "extras.f32")[offset : offset + count],
            dtype=np.float32,
            copy=True,
        ).reshape(block["shape"])
        tensor = torch.from_numpy(values)
        device = torch.device(self.device)
        return tensor.to(device) if device.type != "cpu" else tensor

    def fp32(self, name: str) -> torch.Tensor:
        physical = self._physical_key(name)
        cached = self.fp32_aux_cache.get(physical)
        if cached is not None:
            return cached
        if self.pin_fp32_aux:
            raise RuntimeError(f"pinned direct CUDA auxiliary {name!r} is absent")
        return self._load_fp32(name)

    def _load_page(
        self,
        name: str,
        *,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> CompactQRowPage:
        block = self._resolve(name)
        if block.get("kind") != "qrow":
            raise ValueError(f"{name!r} is not a qrow block")
        out_features, in_features = (int(value) for value in block["shape"])
        start = int(start_row)
        stop = out_features if end_row is None else int(end_row)
        if start < 0 or stop <= start or stop > out_features:
            raise ValueError(f"invalid row range [{start}, {stop}) for {name}")
        weight_start = int(block["w_off"]) + start * in_features
        scale_start = int(block["s_off"]) // 4 + start
        codes_np = np.asarray(
            self._array(block, "weights.i8")[
                weight_start : weight_start + (stop - start) * in_features
            ],
            dtype=np.int8,
        ).reshape(stop - start, in_features)
        scales_np = np.asarray(
            self._array(block, "scales.f32")[scale_start : scale_start + stop - start],
            dtype=np.float32,
        )
        codes = torch.from_numpy(codes_np.copy())
        scales = torch.from_numpy(scales_np.copy())
        device = torch.device(self.device)
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

    def compact_page(self, name: str) -> CompactQRowPage:
        cache_key = self._compact_cache_key(name)
        cached = self.compact_cache.get(cache_key)
        if cached is not None:
            self.compact_cache.move_to_end(cache_key)
            self.stats.cache_hits += 1
            return cached
        page = self._load_page(cache_key)
        self.stats.page_loads += 1
        if page.codes.device.type == "cuda":
            self.stats.compact_h2d_bytes += page.compact_bytes
        auxiliary_bytes = self.fp32_aux_cache_bytes
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

    def weight(self, name: str) -> torch.Tensor:
        page = self.compact_page(name)
        value = page.codes.float() * page.scales[:, None]
        if self.compute_dtype is not torch.float32:
            value = value.to(self.compute_dtype)
        self.max_block_bytes = max(self.max_block_bytes, _tensor_bytes(value))
        return value

    def matmul_row_stable(self, name: str, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight(name)
        return torch.cat(
            tuple(value[row : row + 1] @ weight.T for row in range(int(value.shape[0]))),
            dim=0,
        )

    def selected_rows_fp32(
        self,
        name: str,
        ids: np.ndarray | torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        block = self._resolve(name)
        if block.get("kind") != "qrow":
            raise ValueError(f"{name!r} is not a qrow block")
        indices = torch.as_tensor(ids, dtype=torch.long).detach().cpu().numpy()
        if indices.ndim != 1 or not indices.size:
            raise ValueError("selected qrow IDs must be a non-empty one-dimensional array")
        out_features, in_features = (int(value) for value in block["shape"])
        if int(indices.min()) < 0 or int(indices.max()) >= out_features:
            raise IndexError(f"selected qrow ID outside [0, {out_features})")
        weight_offset = int(block["w_off"])
        rows = np.asarray(
            self._array(block, "weights.i8")[
                weight_offset : weight_offset + out_features * in_features
            ],
            dtype=np.int8,
        ).reshape(out_features, in_features)
        scale_offset = int(block["s_off"]) // 4
        scales_array = self._array(block, "scales.f32")
        codes = torch.from_numpy(np.asarray(rows[indices], dtype=np.int8).copy())
        scales = torch.from_numpy(
            np.asarray(scales_array[scale_offset + indices], dtype=np.float32).copy()
        )
        device = torch.device(self.device)
        compact_bytes = _tensor_bytes(codes) + _tensor_bytes(scales)
        if device.type != "cpu":
            codes = codes.to(device)
            scales = scales.to(device)
            self.stats.compact_h2d_bytes += compact_bytes
        self.stats.selected_head_calls += 1
        self.stats.selected_head_rows += int(indices.size)
        self.stats.selected_head_compact_bytes += compact_bytes
        self.stats.compact_logical_bytes += compact_bytes
        return codes.float() * scales[:, None]

    def row_blocks(self, name: str, bs: int = 8192):
        block = self._resolve(name)
        if block.get("kind") != "qrow":
            raise ValueError(f"{name!r} is not a qrow block")
        out_features = int(block["shape"][0])
        for start in range(0, out_features, bs):
            stop = min(start + bs, out_features)
            page = self._load_page(name, start_row=start, end_row=stop)
            weights = page.codes.float() * page.scales[:, None]
            yield start, stop, weights

    def snapshot(self) -> dict[str, Any]:
        qrows = {
            self._physical_key(name)
            for name in self.blocks
            if self._resolve(name).get("kind") == "qrow"
        }
        auxiliaries = {
            self._physical_key(name)
            for name in self.blocks
            if self._resolve(name).get("kind") == "fp32"
        }
        fully_resident = qrows <= set(self.compact_cache) and auxiliaries <= set(
            self.fp32_aux_cache
        )
        return {
            "schema": self.artifact.manifest["schema"],
            "artifact_sha256": self.artifact.artifact_sha256,
            "source_artifact_id": self.source_artifact.artifact_id,
            "direct_from_canonical_source": True,
            "intermediate_qstore": False,
            "fully_resident": fully_resident,
            "resident_exact_head": self.resident_exact_head_fp32() is not None,
            "resident_exact_head_bytes": self.resident_exact_head_bytes,
            "residency_contract": "direct-role-files+complete-device-residency-v1",
            "store_stats": self.stats_snapshot(),
        }

    def assert_content_identity_unchanged(self) -> None:
        self.artifact.assert_unchanged()

    def reverify_content_identity(self) -> dict[str, Any]:
        self.artifact = VerifiedSourceCudaInt8Artifact(
            self.artifact.path, source_artifact=self.source_artifact
        )
        return {
            "artifact_sha256": self.artifact.artifact_sha256,
            "source_artifact_id": self.source_artifact.artifact_id,
            "content_identity_verified": True,
            "blob_identity_verified": True,
            "direct_from_canonical_source": True,
            "intermediate_qstore": False,
        }

    def stats_snapshot(self) -> dict[str, Any]:
        result = super().stats_snapshot()
        result.update(
            {
                "storage_layout": self.artifact.manifest["schema"],
                "direct_from_canonical_source": True,
                "intermediate_qstore": False,
                "artifact_physical_bytes": self.artifact.physical_bytes,
            }
        )
        return result

    def close(self) -> None:
        self.clear_compact_cache()
        self.fp32_aux_cache.clear()
        self.fp32_aux_cache_bytes = 0
        self.resident_exact_heads.clear()
        self.resident_exact_head_bytes = 0
        for array in self._mapped.values():
            handle = getattr(array, "_mmap", None)
            if handle is not None:
                handle.close()
        self._mapped.clear()


__all__ = ["DirectSourceCudaInt8Store"]
