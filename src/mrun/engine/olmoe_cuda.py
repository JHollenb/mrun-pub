#!/usr/bin/env python3
"""All-layer OLMoE CUDA runtime with an FP8 QStore expert tier.

The runtime keeps the attention/router skeleton and compact FP8 expert pages resident
on CUDA, compiles real router outputs into expert-major row batches, and preserves KV
state across full-vocabulary greedy decode steps. A streamed-bf16 expert backend is
included as the model-level quality reference.

Promoted from
``experiments/2026-07-18-183639-beast-all-layer-100x/olmoe_cuda_runtime.py``
after its full-model quality gate and sustained B=512 decode gate passed. The
experiment entry point is now a compatibility wrapper around this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import platform
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from ..models import resolve_model, snapshot_dir
from ..paths import models_root, stores_root
from ..store_provenance import (
    build_builder_provenance,
    build_derived_provenance,
    build_source_provenance,
    verify_builder_provenance,
    verify_derived_provenance,
    verify_source_provenance,
)
from ._base_impl import BaseEngine
from .base import EngineCapabilities
from .kernels.expert_queue import (
    ExpertQueuePlan,
    build_expert_queue,
    build_expert_queue_device,
    build_expert_queue_flat,
    scatter_committed_device,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-side tests do not install Triton
    triton = None
    tl = None


DEFAULT_MODEL_DIR = models_root() / "OLMoE-1B-7B-0924"
DEFAULT_STORE_DIR = stores_root() / "OLMoE-1B-7B-0924-fp8-e4m3fn-v1"
PAGE_ALIGNMENT = 4096
FP8_MAX = 448.0
STORE_SCHEMA = 2
STORE_CODEC = "rowwise-e4m3fn-v1"
STORE_FILES = ("experts.fp8",)
STORE_QUANTIZATION = {
    "codec": STORE_CODEC,
    "weight_dtype": "float8_e4m3fn",
    "scale_dtype": "float32",
    "granularity": "per-output-row",
    "page_alignment": PAGE_ALIGNMENT,
}
PROMPT_TEMPLATES = (
    "Efficient inference preserves state and groups repeated expert operations "
    "for request {index}.",
    "A CUDA runtime should move compact weights once and reuse them across useful row {index}.",
    "Sparse routing exposes expert locality while attention preserves causal context "
    "number {index}.",
    "The benchmark counts complete generated tokens and keeps its quality gate explicit {index}.",
    "Paging controls capacity while tensor cores control compute throughput for batch {index}.",
    "Metadata is valuable when it removes calls without changing model semantics {index}.",
    "Persistent key value state avoids replaying the prefix for generation stream {index}.",
    "A strong baseline uses the same checkpoint precision and output head for case {index}.",
)

__all__ = [
    "BF16StreamExpertBackend",
    "FP8ExpertBackend",
    "FP8ExpertStore",
    "FP8_MAX",
    "ForwardResult",
    "LayerKV",
    "OLMoECudaEngine",
    "OLMoECudaRuntime",
    "ResidentSkeleton",
    "TensorReader",
    "align_up",
    "build_fp8_store",
    "greedy_tokens",
    "grouped_fp8_tile_config",
    "grouped_route_plan",
    "grouped_route_pruned_queue",
    "grouped_route_queue",
    "grouped_route_tensors",
    "grouped_route_tensors_with_inverse",
    "main",
    "parse_batch_sizes",
    "prune_router_topk",
    "quantize_fp8_weight",
    "resolve_olmoe_model_dir",
    "resolve_olmoe_store_dir",
    "run_decode_sweep",
    "run_quality",
]


if triton is not None:

    @triton.jit
    def _triton_route_count_kernel(
        experts_ptr,
        counts_ptr,
        assignments: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < assignments
        experts = tl.load(experts_ptr + offsets, mask=mask, other=0)
        tl.atomic_add(counts_ptr + experts, 1, mask=mask)

    @triton.jit
    def _triton_route_prefix_kernel(
        counts_ptr,
        starts_ptr,
        num_experts: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.arange(0, block)
        mask = offsets < num_experts
        counts = tl.load(counts_ptr + offsets, mask=mask, other=0)
        inclusive = tl.cumsum(counts, axis=0)
        tl.store(starts_ptr + offsets, inclusive - counts, mask=mask)

    @triton.jit
    def _triton_route_scatter_kernel(
        experts_ptr,
        weights_ptr,
        starts_ptr,
        cursors_ptr,
        token_ids_ptr,
        coefficients_ptr,
        inverse_ptr,
        assignments: tl.constexpr,
        top_k: tl.constexpr,
        emit_inverse: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < assignments
        experts = tl.load(experts_ptr + offsets, mask=mask, other=0)
        local_offsets = tl.atomic_add(cursors_ptr + experts, 1, mask=mask)
        destinations = tl.load(starts_ptr + experts, mask=mask, other=0) + local_offsets
        coefficients = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
        tl.store(token_ids_ptr + destinations, offsets // top_k, mask=mask)
        tl.store(coefficients_ptr + destinations, coefficients, mask=mask)
        if emit_inverse:
            # ``offsets`` is the original flattened [token, route-rank]
            # position and ``destinations`` is its expert-major queue row.  The
            # stable reducer otherwise has to rediscover this bijection with a
            # search, repeat-interleave, comparison matrix, argmax, and scatter.
            tl.store(inverse_ptr + offsets, destinations, mask=mask)

    @triton.jit
    def _triton_grouped_fp8_kernel(
        a_ptr,
        b_ptr,
        a_scale_ptr,
        b_scale_ptr,
        out_ptr,
        starts_ptr,
        counts_ptr,
        n: tl.constexpr,
        k: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        group = tl.program_id(0)
        tile = tl.program_id(1)
        tiles_n: tl.constexpr = tl.cdiv(n, block_n)
        tile_m = tile // tiles_n
        tile_n = tile % tiles_n
        count = tl.load(counts_ptr + group)
        start = tl.load(starts_ptr + group)
        offsets_m = tile_m * block_m + tl.arange(0, block_m)
        offsets_n = tile_n * block_n + tl.arange(0, block_n)
        if tile_m * block_m < count:
            accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
            for k_start in range(0, k, block_k):
                offsets_k = k_start + tl.arange(0, block_k)
                a_values = tl.load(
                    a_ptr + (start + offsets_m[:, None]) * k + offsets_k[None, :],
                    mask=(offsets_m[:, None] < count) & (offsets_k[None, :] < k),
                    other=0.0,
                )
                b_values = tl.load(
                    b_ptr + group * n * k + offsets_n[:, None] * k + offsets_k[None, :],
                    mask=(offsets_n[:, None] < n) & (offsets_k[None, :] < k),
                    other=0.0,
                )
                accumulator += tl.dot(a_values, tl.trans(b_values))
            a_scales = tl.load(
                a_scale_ptr + start + offsets_m,
                mask=offsets_m < count,
                other=0.0,
            )
            b_scales = tl.load(
                b_scale_ptr + group * n + offsets_n,
                mask=offsets_n < n,
                other=0.0,
            )
            output = accumulator * a_scales[:, None] * b_scales[None, :]
            tl.store(
                out_ptr + (start + offsets_m[:, None]) * n + offsets_n[None, :],
                output,
                mask=(offsets_m[:, None] < count) & (offsets_n[None, :] < n),
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def align_up(value: int, alignment: int = PAGE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def tensor_relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(1e-12)
    return float((numerator / denominator).item())


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, *, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def env_optional_int(name: str, *, minimum: int = 1) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def env_optional_float(name: str, *, minimum: float = 0.0) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite float, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def grouped_fp8_tile_config() -> tuple[int, int, int, int, int]:
    """Return Triton grouped-FP8 tile knobs.

    The defaults preserve the original Beast-validated kernel. Env knobs are intentionally
    narrow so executor sweeps can run through mrun without patching code for each trial.
    """

    return (
        env_int("MRUN_OLMOE_GROUPED_BLOCK_M", default=32),
        env_int("MRUN_OLMOE_GROUPED_BLOCK_N", default=64),
        env_int("MRUN_OLMOE_GROUPED_BLOCK_K", default=32),
        env_int("MRUN_OLMOE_GROUPED_WARPS", default=4),
        env_int("MRUN_OLMOE_GROUPED_STAGES", default=3),
    )


class TensorReader:
    """Lazy safetensors reader with explicit unmapping between streamed layers."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.handles: dict[str, Any] = {}

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        handle = self.handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
            self.handles[shard] = handle
        return handle.get_tensor(key)

    def release(self) -> None:
        self.handles.clear()


def quantize_fp8_weight(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row E4M3 quantization; returned codes/scales remain on CPU."""
    values = source.detach().float().contiguous()
    scales = values.abs().amax(dim=1).div(FP8_MAX).clamp_min(1e-12)
    quantized = (values / scales[:, None]).clamp(-FP8_MAX, FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).contiguous(), scales.contiguous()


def _write_padding(handle: Any, digest: Any) -> None:
    padding = (-handle.tell()) % PAGE_ALIGNMENT
    if padding:
        payload = b"\0" * padding
        handle.write(payload)
        digest.update(payload)


def _verify_existing_fp8_store(
    store_dir: Path,
    *,
    source: dict[str, Any],
    builder: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"refusing to overwrite incomplete OLMoE store: {store_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": STORE_SCHEMA,
        "codec": STORE_CODEC,
        "model_config_sha256": source["config"]["sha256"],
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"existing OLMoE store is legacy or mismatched: {actual} != {expected}")
    verify_source_provenance(manifest.get("source"), source)
    verify_builder_provenance(manifest.get("builder"), builder)
    verify_derived_provenance(store_dir, manifest.get("derived"), expected_filenames=STORE_FILES)
    derived_record = manifest["derived"]["files"][0]
    if manifest.get("data_file") != derived_record["name"]:
        raise RuntimeError("existing OLMoE data filename does not match derived provenance")
    if manifest.get("data_bytes") != derived_record["bytes"]:
        raise RuntimeError("existing OLMoE data size does not match derived provenance")
    if manifest.get("data_sha256") != derived_record["sha256"]:
        raise RuntimeError("existing OLMoE data hash does not match derived provenance")
    return manifest


def _materialize_fp8_store(
    model_dir: Path,
    temporary: Path,
    *,
    cfg: dict[str, Any],
    source: dict[str, Any],
    builder: dict[str, Any],
) -> dict[str, Any]:
    layers = int(cfg["num_hidden_layers"])
    experts = int(cfg["num_experts"])
    reader = TensorReader(model_dir)
    data_path = temporary / "experts.fp8"
    pages: dict[str, Any] = {}
    digest = hashlib.sha256()
    started = time.perf_counter()

    try:
        with data_path.open("wb") as output:
            for layer in range(layers):
                for expert in range(experts):
                    _write_padding(output, digest)
                    page_offset = output.tell()
                    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
                    gate = reader.get(f"{prefix}.gate_proj.weight")
                    up = reader.get(f"{prefix}.up_proj.weight")
                    down = reader.get(f"{prefix}.down_proj.weight")
                    source_matrices = {
                        "gate_up": torch.cat((gate, up), dim=0),
                        "down": down,
                    }
                    descriptors: dict[str, Any] = {}
                    for name, source_matrix in source_matrices.items():
                        quantized, scales = quantize_fp8_weight(source_matrix)
                        blob = quantized.view(torch.uint8).numpy().tobytes(order="C")
                        offset = output.tell()
                        output.write(blob)
                        digest.update(blob)
                        scale_blob = scales.numpy().tobytes(order="C")
                        scale_offset = output.tell()
                        output.write(scale_blob)
                        digest.update(scale_blob)
                        descriptors[name] = {
                            "offset": offset,
                            "nbytes": len(blob),
                            "shape": list(quantized.shape),
                            "scale_offset": scale_offset,
                            "scale_nbytes": len(scale_blob),
                            "scale_shape": list(scales.shape),
                        }
                        del quantized, scales, source_matrix
                    pages[f"L{layer}.E{expert}"] = {
                        "offset": page_offset,
                        "length": output.tell() - page_offset,
                        "matrices": descriptors,
                    }
                    del gate, up, down, source_matrices
                reader.release()
                print(
                    f"built FP8 layer {layer + 1}/{layers}: {output.tell() / (1024**3):.2f} GiB",
                    flush=True,
                )
    finally:
        reader.release()

    derived = build_derived_provenance(
        temporary,
        STORE_FILES,
        known_sha256={data_path.name: digest.hexdigest()},
    )
    manifest = {
        "schema_version": STORE_SCHEMA,
        "codec": STORE_CODEC,
        "created_utc": utc_now(),
        "model_dir": str(model_dir),
        "model_config_sha256": source["config"]["sha256"],
        "source": source,
        "builder": builder,
        "derived": derived,
        "data_file": data_path.name,
        "data_bytes": data_path.stat().st_size,
        "data_sha256": digest.hexdigest(),
        "page_alignment": PAGE_ALIGNMENT,
        "layers": layers,
        "num_experts": experts,
        "hidden_size": int(cfg["hidden_size"]),
        "intermediate_size": int(cfg["intermediate_size"]),
        "top_k": int(cfg["num_experts_per_tok"]),
        "build_s": time.perf_counter() - started,
        "pages": pages,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_fp8_store(
    model_dir: Path,
    store_dir: Path,
    *,
    model_name: str,
    hf_id: str,
    revision: str | None = None,
) -> dict[str, Any]:
    """Build an atomic, source-content-bound FP8 expert store."""
    model_dir = model_dir.expanduser().resolve()
    store_dir = store_dir.expanduser()
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty OLMoE weight map: {index_path}")
    source_shards = [
        model_dir / name for name in sorted({str(name) for name in weight_map.values()})
    ]
    source = build_source_provenance(
        model_dir,
        source_shards,
        model_name=model_name,
        hf_id=hf_id,
        revision=revision,
    )
    builder = build_builder_provenance(
        [Path(__file__)],
        name="mrun.engine.olmoe_cuda.build_fp8_store",
        schema_version=str(STORE_SCHEMA),
        quantization=STORE_QUANTIZATION,
    )
    if store_dir.exists():
        return _verify_existing_fp8_store(store_dir, source=source, builder=builder)

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("model_type") != "olmoe":
        raise ValueError(f"expected olmoe, found {cfg.get('model_type')!r}")
    store_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{store_dir.name}.building-", dir=store_dir.parent))
    try:
        manifest = _materialize_fp8_store(
            model_dir,
            temporary,
            cfg=cfg,
            source=source,
            builder=builder,
        )
        temporary.rename(store_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


@dataclass(frozen=True)
class DeviceExpertLayer:
    gate_up: torch.Tensor
    gate_up_scales: torch.Tensor
    down: torch.Tensor
    down_scales: torch.Tensor


class FP8ExpertStore:
    """Memory-mapped host QStore that can materialize one all-layer CUDA hot tier."""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.manifest = json.loads((store_dir / "manifest.json").read_text())
        self.handle = (store_dir / self.manifest["data_file"]).open("rb")
        self.mapping = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        self.mapping.close()
        self.handle.close()

    def _load_matrix(self, descriptor: dict[str, Any], device: str) -> torch.Tensor:
        codes = torch.frombuffer(
            self.mapping,
            dtype=torch.uint8,
            count=int(descriptor["nbytes"]),
            offset=int(descriptor["offset"]),
        )
        quantized = codes.view(torch.float8_e4m3fn).reshape(descriptor["shape"])
        return quantized.to(device=device, non_blocking=False)

    def _load_scales(self, descriptor: dict[str, Any], device: str) -> torch.Tensor:
        scales = torch.frombuffer(
            self.mapping,
            dtype=torch.float32,
            count=int(descriptor["scale_nbytes"]) // 4,
            offset=int(descriptor["scale_offset"]),
        ).reshape(descriptor["scale_shape"])
        return scales.to(device=device, non_blocking=False)

    def _load_device_layer(self, layer: int, device: str) -> DeviceExpertLayer:
        experts = int(self.manifest["num_experts"])
        gate_up: list[torch.Tensor] = []
        gate_up_scales: list[torch.Tensor] = []
        down: list[torch.Tensor] = []
        down_scales: list[torch.Tensor] = []
        for expert in range(experts):
            descriptor = self.manifest["pages"][f"L{layer}.E{expert}"]["matrices"]
            gate_up.append(self._load_matrix(descriptor["gate_up"], device))
            gate_up_scales.append(self._load_scales(descriptor["gate_up"], device))
            down.append(self._load_matrix(descriptor["down"], device))
            down_scales.append(self._load_scales(descriptor["down"], device))
        return DeviceExpertLayer(
            gate_up=torch.stack(gate_up),
            gate_up_scales=torch.stack(gate_up_scales),
            down=torch.stack(down),
            down_scales=torch.stack(down_scales),
        )

    def load_device_layers(
        self,
        layers: Sequence[int],
        device: str = "cuda",
    ) -> tuple[list[DeviceExpertLayer], dict[str, Any]]:
        """Materialize selected expert layers on CUDA without loading the full hot tier."""
        total_layers = int(self.manifest["layers"])
        selected = [int(layer) for layer in layers]
        if not selected:
            raise ValueError("at least one layer must be selected")
        if any(layer < 0 or layer >= total_layers for layer in selected):
            raise ValueError(f"selected layers must be within [0, {total_layers})")
        started = time.perf_counter()
        pages = []
        for layer in selected:
            pages.append(self._load_device_layer(layer, device))
            print(f"loaded FP8 expert layer {layer + 1}/{total_layers}", flush=True)
        torch.cuda.synchronize()
        return pages, {
            "load_s": time.perf_counter() - started,
            "layers": selected,
            "device_bytes": sum(
                layer.gate_up.numel()
                + layer.down.numel()
                + layer.gate_up_scales.numel() * layer.gate_up_scales.element_size()
                + layer.down_scales.numel() * layer.down_scales.element_size()
                for layer in pages
            ),
        }

    def load_device(self, device: str = "cuda") -> tuple[list[DeviceExpertLayer], dict[str, Any]]:
        layers = int(self.manifest["layers"])
        pages: list[DeviceExpertLayer] = []
        started = time.perf_counter()
        for layer in range(layers):
            pages.append(self._load_device_layer(layer, device))
            print(f"loaded FP8 expert layer {layer + 1}/{layers}", flush=True)
        torch.cuda.synchronize()
        return pages, {
            "load_s": time.perf_counter() - started,
            "device_bytes": sum(
                layer.gate_up.numel()
                + layer.down.numel()
                + layer.gate_up_scales.numel() * layer.gate_up_scales.element_size()
                + layer.down_scales.numel() * layer.down_scales.element_size()
                for layer in pages
            ),
        }


@dataclass(frozen=True)
class LayerWeights:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    router: torch.Tensor


class ResidentSkeleton:
    """All always-active OLMoE tensors resident in bf16 on CUDA."""

    def __init__(
        self,
        reader: TensorReader,
        cfg: dict[str, Any],
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        def load(name: str) -> torch.Tensor:
            return reader.get(name).to(device=device, dtype=dtype).contiguous()

        self.embedding = load("model.embed_tokens.weight")
        self.layers: list[LayerWeights] = []
        for layer in range(int(cfg["num_hidden_layers"])):
            prefix = f"model.layers.{layer}"
            self.layers.append(
                LayerWeights(
                    input_norm=load(f"{prefix}.input_layernorm.weight"),
                    post_norm=load(f"{prefix}.post_attention_layernorm.weight"),
                    q_norm=load(f"{prefix}.self_attn.q_norm.weight"),
                    k_norm=load(f"{prefix}.self_attn.k_norm.weight"),
                    q_proj=load(f"{prefix}.self_attn.q_proj.weight"),
                    k_proj=load(f"{prefix}.self_attn.k_proj.weight"),
                    v_proj=load(f"{prefix}.self_attn.v_proj.weight"),
                    o_proj=load(f"{prefix}.self_attn.o_proj.weight"),
                    router=load(f"{prefix}.mlp.gate.weight"),
                )
            )
        self.final_norm = load("model.norm.weight")
        self.lm_head = load("lm_head.weight")
        reader.release()


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class BackendStats:
    qmm_calls: int = 0
    grouped_kernel_calls: int = 0
    individual_kernel_calls: int = 0
    expert_dispatches: int = 0
    expert_rows: int = 0
    route_metadata_s: float = 0.0
    layer_rows: list[dict[str, Any]] = field(default_factory=list)
    grouped_routes: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.qmm_calls = 0
        self.grouped_kernel_calls = 0
        self.individual_kernel_calls = 0
        self.expert_dispatches = 0
        self.expert_rows = 0
        self.route_metadata_s = 0.0
        self.layer_rows.clear()
        self.grouped_routes.clear()

    def as_dict(self) -> dict[str, Any]:
        layers = list(self.layer_rows)
        grouped_dispatches = 0
        for record in self.grouped_routes:
            active_counts = [
                int(count) for count in record["counts"].detach().cpu().tolist() if count
            ]
            grouped_dispatches += len(active_counts)
            assignments = int(record.get("assignments", record["rows"] * record["top_k"]))
            layer_record = {
                "layer": record["layer"],
                "rows": record["rows"],
                "top_k": record["top_k"],
                "assignments": assignments,
                "original_assignments": int(
                    record.get("original_assignments", record["rows"] * record["top_k"])
                ),
                "active_experts": len(active_counts),
                "route_reuse_x": assignments / max(1, len(active_counts)),
                "rows_per_expert_mean": (
                    float(statistics.mean(active_counts)) if active_counts else 0.0
                ),
                "rows_per_expert_max": max(active_counts) if active_counts else 0,
                "metadata_s": record["metadata_s"],
                "route_compiler": record["route_compiler"],
                "grouped_kernel": record["grouped_kernel"],
                "scatter_reduce": record.get("scatter_reduce", "torch-index-add"),
            }
            if "route_prune" in record:
                layer_record["route_prune"] = record["route_prune"]
            layers.append(layer_record)
        return {
            "qmm_calls": self.qmm_calls,
            "grouped_kernel_calls": self.grouped_kernel_calls,
            "individual_kernel_calls": self.individual_kernel_calls,
            "expert_dispatches": self.expert_dispatches + grouped_dispatches,
            "expert_rows": self.expert_rows,
            "route_metadata_s": self.route_metadata_s,
            "layers": layers,
        }


def grouped_route_plan(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Compile `[rows, top_k]` router output into stable expert-major batches."""
    started = time.perf_counter()
    rows, top_k = top_indices.shape
    flat_experts = top_indices.reshape(-1)
    flat_tokens = (
        torch.arange(rows, device=top_indices.device, dtype=torch.long)
        .unsqueeze(1)
        .expand(rows, top_k)
        .reshape(-1)
    )
    flat_weights = top_weights.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    sorted_tokens = flat_tokens.index_select(0, order)
    sorted_weights = flat_weights.index_select(0, order)
    counts = torch.bincount(flat_experts, minlength=int(top_indices.max().item()) + 1)
    host_counts = counts.cpu().tolist()
    assignments: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    offset = 0
    active_counts: list[int] = []
    for expert, count in enumerate(host_counts):
        if count:
            assignments.append(
                (
                    expert,
                    sorted_tokens[offset : offset + count],
                    sorted_weights[offset : offset + count],
                )
            )
            active_counts.append(count)
        offset += count
    return assignments, {
        "rows": rows,
        "top_k": top_k,
        "assignments": rows * top_k,
        "active_experts": len(assignments),
        "route_reuse_x": rows * top_k / max(1, len(assignments)),
        "rows_per_expert_mean": float(statistics.mean(active_counts)),
        "rows_per_expert_max": max(active_counts),
        "metadata_s": time.perf_counter() - started,
    }


def expert_queue_assignments(
    plan: ExpertQueuePlan,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Convert an expert-major queue into per-expert row lists for the small-batch path."""
    assignments: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    offset = 0
    for expert, count in enumerate(plan.counts.detach().cpu().tolist()):
        count = int(count)
        if count:
            next_offset = offset + count
            assignments.append(
                (
                    expert,
                    plan.token_ids[offset:next_offset],
                    plan.coefficients[offset:next_offset],
                )
            )
            offset = next_offset
        else:
            offset += count
    return assignments


def route_stats_from_queue(
    plan: ExpertQueuePlan,
    *,
    metadata_s: float,
    original_assignments: int | None = None,
    route_prune: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_counts = [int(count) for count in plan.counts.detach().cpu().tolist() if count]
    stats: dict[str, Any] = {
        "rows": plan.num_rows,
        "top_k": plan.top_k,
        "assignments": plan.assignments,
        "original_assignments": (
            int(original_assignments)
            if original_assignments is not None
            else plan.num_rows * plan.top_k
        ),
        "active_experts": len(active_counts),
        "route_reuse_x": plan.assignments / max(1, len(active_counts)),
        "rows_per_expert_mean": float(statistics.mean(active_counts)) if active_counts else 0.0,
        "rows_per_expert_max": max(active_counts) if active_counts else 0,
        "metadata_s": metadata_s,
    }
    if route_prune is not None:
        stats["route_prune"] = route_prune
    return stats


def prune_router_topk(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    min_weight: float = 0.0,
    max_top_k: int | None = None,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Drop weak router packets while preserving at least the primary route per row.

    This is an opt-in approximate path inspired by SPF's dominance-routed reductions: if a token's
    router mass is concentrated in the first one or two experts, we can skip the tail expert MLPs
    before the expensive grouped GEMMs. The default backend does not call this function.
    """
    if top_indices.ndim != 2 or top_indices.shape[0] == 0 or top_indices.shape[1] == 0:
        raise ValueError("top_indices must have non-empty shape [rows, top_k]")
    if top_weights.shape != top_indices.shape:
        raise ValueError("top_weights must match top_indices")
    if not math.isfinite(min_weight) or min_weight < 0.0:
        raise ValueError("min_weight must be a finite non-negative float")
    rows, top_k = top_indices.shape
    if max_top_k is not None and (max_top_k < 1 or max_top_k > top_k):
        raise ValueError(f"max_top_k must be in [1, {top_k}], got {max_top_k}")
    if top_indices.device != top_weights.device:
        raise ValueError("top_indices and top_weights must be on the same device")

    slot_ids = torch.arange(top_k, device=top_indices.device, dtype=torch.int64)
    slot_keep = torch.ones(top_k, device=top_indices.device, dtype=torch.bool)
    if max_top_k is not None:
        slot_keep = slot_ids < max_top_k
    weight_keep = torch.ones_like(top_weights, dtype=torch.bool)
    if min_weight > 0.0:
        weight_keep = top_weights.float() >= min_weight
    keep = weight_keep & slot_keep.unsqueeze(0)
    keep[:, 0] = True

    token_grid = (
        torch.arange(rows, device=top_indices.device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(rows, top_k)
    )
    slot_grid = slot_ids.unsqueeze(0).expand(rows, top_k)
    coefficients_grid = top_weights.float()
    if renormalize:
        kept_mass = (coefficients_grid * keep.to(coefficients_grid.dtype)).sum(dim=1)
        coefficients_grid = coefficients_grid / kept_mass.clamp_min(1e-12)[:, None]

    token_ids = token_grid[keep].contiguous()
    expert_ids = top_indices.to(dtype=torch.int64)[keep].contiguous()
    route_slots = slot_grid[keep].contiguous()
    coefficients = coefficients_grid[keep].to(dtype=torch.float32).contiguous()
    original_assignments = int(top_indices.numel())
    kept_assignments = int(token_ids.numel())
    stats = {
        "enabled": True,
        "min_weight": float(min_weight),
        "max_top_k": max_top_k,
        "renormalize": bool(renormalize),
        "original_assignments": original_assignments,
        "kept_assignments": kept_assignments,
        "dropped_assignments": original_assignments - kept_assignments,
        "kept_fraction": kept_assignments / max(1, original_assignments),
        "mean_routes_per_row": kept_assignments / max(1, rows),
    }
    return token_ids, expert_ids, route_slots, coefficients, stats


def grouped_route_pruned_queue(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    num_experts: int,
    min_weight: float = 0.0,
    max_top_k: int | None = None,
    renormalize: bool = False,
) -> tuple[ExpertQueuePlan, dict[str, Any]]:
    """Build an expert-major queue after opt-in sparse route pruning."""
    started = time.perf_counter()
    token_ids, expert_ids, route_slots, coefficients, prune_stats = prune_router_topk(
        top_indices,
        top_weights,
        min_weight=min_weight,
        max_top_k=max_top_k,
        renormalize=renormalize,
    )
    rows, top_k = top_indices.shape
    epochs = torch.zeros(rows, device=top_indices.device, dtype=torch.int64)
    deadlines = torch.zeros_like(epochs)
    plan = build_expert_queue_flat(
        token_ids,
        expert_ids,
        route_slots,
        coefficients,
        num_rows=rows,
        num_experts=num_experts,
        top_k=top_k,
        epochs=epochs,
        deadlines=deadlines,
        validate=not top_indices.is_cuda,
    )
    prune_stats["metadata_s"] = time.perf_counter() - started
    return plan, prune_stats


def _grouped_route_tensors_impl(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    num_experts: int,
    include_inverse: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    float,
]:
    """Build the fixed-width route plan and optionally retain its inverse permutation."""
    started = time.perf_counter()
    rows, top_k = top_indices.shape
    flat_experts = top_indices.reshape(-1)
    flat_weights = top_weights.reshape(-1)
    assignments = flat_experts.numel()
    if top_indices.is_cuda:
        if triton is None:
            raise RuntimeError("CUDA route compaction requires Triton")
        counts = torch.zeros(num_experts, device=top_indices.device, dtype=torch.int32)
        block = 256
        grid = (triton.cdiv(assignments, block),)
        _triton_route_count_kernel[grid](
            flat_experts,
            counts,
            assignments=assignments,
            block=block,
        )
        starts = torch.empty_like(counts)
        prefix_block = triton.next_power_of_2(num_experts)
        _triton_route_prefix_kernel[(1,)](
            counts,
            starts,
            num_experts=num_experts,
            block=prefix_block,
            num_warps=1,
        )
        cursors = torch.zeros_like(counts)
        token_ids = torch.empty(
            assignments,
            device=top_indices.device,
            dtype=torch.long,
        )
        coefficients = torch.empty_like(flat_weights)
        inverse = (
            torch.empty(assignments, device=top_indices.device, dtype=torch.long)
            if include_inverse
            else None
        )
        _triton_route_scatter_kernel[grid](
            flat_experts,
            flat_weights,
            starts,
            cursors,
            token_ids,
            coefficients,
            token_ids if inverse is None else inverse,
            assignments=assignments,
            top_k=top_k,
            emit_inverse=include_inverse,
            block=block,
        )
        return (
            token_ids,
            coefficients,
            starts,
            counts,
            inverse,
            time.perf_counter() - started,
        )

    flat_tokens = (
        torch.arange(rows, device=top_indices.device, dtype=torch.long)
        .unsqueeze(1)
        .expand(rows, top_k)
        .reshape(-1)
    )
    order = torch.argsort(flat_experts, stable=True)
    token_ids = flat_tokens.index_select(0, order)
    coefficients = flat_weights.index_select(0, order)
    counts = torch.bincount(flat_experts, minlength=num_experts).to(torch.int32)
    ends = counts.cumsum(dim=0, dtype=torch.int32)
    starts = torch.cat(
        (
            torch.zeros(1, device=top_indices.device, dtype=torch.int32),
            ends[:-1],
        )
    )
    inverse = None
    if include_inverse:
        inverse = torch.empty(assignments, device=top_indices.device, dtype=torch.long)
        inverse.index_copy_(
            0,
            order,
            torch.arange(assignments, device=top_indices.device, dtype=torch.long),
        )
    return token_ids, coefficients, starts, counts, inverse, time.perf_counter() - started


def grouped_route_tensors(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build a fixed-width expert-major route plan without synchronizing to the CPU."""
    token_ids, coefficients, starts, counts, _inverse, metadata_s = _grouped_route_tensors_impl(
        top_indices,
        top_weights,
        num_experts=num_experts,
        include_inverse=False,
    )
    return token_ids, coefficients, starts, counts, metadata_s


def grouped_route_tensors_with_inverse(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """Build an expert-major route plus its original-rank-to-queue-row bijection.

    The CUDA scatter already computes both sides of this mapping.  Retaining it
    lets a deterministic route-rank reducer consume the plan directly rather
    than launching a second route compiler after both expert GEMMs.
    """
    token_ids, coefficients, starts, counts, inverse, metadata_s = _grouped_route_tensors_impl(
        top_indices,
        top_weights,
        num_experts=num_experts,
        include_inverse=True,
    )
    if inverse is None:  # pragma: no cover - closed by include_inverse=True
        raise RuntimeError("route compiler did not emit its inverse permutation")
    return token_ids, coefficients, starts, counts, inverse, metadata_s


def grouped_route_queue(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[ExpertQueuePlan, float]:
    """Build the shared Expert Exchange queue for an OLMoE grouped expert layer.

    Full OLMoE decode currently routes a synchronous batch without request-slot reuse, so every
    packet shares epoch zero and an unset deadline. Continuous serving can later replace these
    tensors with real request-slot epochs/deadlines without changing the packet layout.
    """
    started = time.perf_counter()
    rows = int(top_indices.shape[0])
    epochs = torch.zeros(rows, device=top_indices.device, dtype=torch.int64)
    deadlines = torch.zeros_like(epochs)
    if top_indices.is_cuda:
        plan = build_expert_queue_device(
            top_indices,
            top_weights,
            num_experts=num_experts,
            epochs=epochs,
            deadlines=deadlines,
        )
    else:
        plan = build_expert_queue(
            top_indices,
            top_weights,
            num_experts=num_experts,
            epochs=epochs,
            deadlines=deadlines,
        )
    return plan, time.perf_counter() - started


def quantize_fp8_activation(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = source.abs().amax(dim=1).float().div(FP8_MAX).clamp_min(1e-12)
    quantized = (source.float() / scales[:, None]).clamp(-FP8_MAX, FP8_MAX)
    return quantized.to(torch.float8_e4m3fn), scales


def scaled_fp8_mm(
    source: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    source_q, source_scale = quantize_fp8_activation(source)
    return torch._scaled_mm(
        source_q,
        weight.t(),
        scale_a=source_scale[:, None].contiguous(),
        scale_b=weight_scale[None, :].contiguous(),
        out_dtype=out_dtype,
        use_fast_accum=True,
    )


def grouped_fp8_mm(
    source_q: torch.Tensor,
    weights: torch.Tensor,
    source_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    *,
    max_rows: int,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, str]:
    capability = torch.cuda.get_device_capability(source_q.device)
    if capability[0] >= 9:
        ends = starts + counts
        return (
            torch._scaled_grouped_mm(
                source_q,
                weights.transpose(1, 2),
                source_scales,
                weight_scales,
                offs=ends,
                out_dtype=out_dtype,
                use_fast_accum=True,
            ),
            "torch-scaled-grouped-mm",
        )
    if triton is None:
        raise RuntimeError("Ada grouped FP8 execution requires Triton")
    groups, output_features, contraction = weights.shape
    output = torch.empty(
        (source_q.shape[0], output_features),
        device=source_q.device,
        dtype=out_dtype,
    )
    block_m, block_n, block_k, num_warps, num_stages = grouped_fp8_tile_config()
    grid = (
        groups,
        triton.cdiv(max_rows, block_m) * triton.cdiv(output_features, block_n),
    )
    _triton_grouped_fp8_kernel[grid](
        source_q,
        weights,
        source_scales,
        weight_scales,
        output,
        starts,
        counts,
        n=output_features,
        k=contraction,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output, f"triton-ada-grouped-fp8:m{block_m}n{block_n}k{block_k}w{num_warps}s{num_stages}"


class FP8ExpertBackend:
    def __init__(
        self,
        pages: list[DeviceExpertLayer],
        *,
        dtype: torch.dtype = torch.bfloat16,
        use_fused_expert_queue: bool | None = None,
        route_prune_min_weight: float | None = None,
        route_prune_top_k: int | None = None,
        route_prune_renormalize: bool | None = None,
    ):
        self.pages = pages
        self.dtype = dtype
        self.use_fused_expert_queue = (
            env_flag("MRUN_OLMOE_FUSED_EXPERT_QUEUE")
            if use_fused_expert_queue is None
            else use_fused_expert_queue
        )
        self.route_prune_min_weight = (
            env_optional_float("MRUN_OLMOE_ROUTE_PRUNE_MIN_WEIGHT")
            if route_prune_min_weight is None
            else route_prune_min_weight
        )
        self.route_prune_top_k = (
            env_optional_int("MRUN_OLMOE_ROUTE_PRUNE_TOP_K")
            if route_prune_top_k is None
            else route_prune_top_k
        )
        self.route_prune_renormalize = (
            env_flag("MRUN_OLMOE_ROUTE_PRUNE_RENORMALIZE")
            if route_prune_renormalize is None
            else route_prune_renormalize
        )
        if self.route_prune_min_weight is not None and (
            not math.isfinite(self.route_prune_min_weight) or self.route_prune_min_weight < 0.0
        ):
            raise ValueError("route_prune_min_weight must be a finite non-negative float")
        if self.route_prune_top_k is not None and self.route_prune_top_k < 1:
            raise ValueError("route_prune_top_k must be positive")
        self.stats = BackendStats()

    def _route_prune_enabled(self, top_k: int) -> bool:
        if self.route_prune_min_weight is not None and self.route_prune_min_weight > 0.0:
            return True
        if self.route_prune_top_k is not None:
            if self.route_prune_top_k > top_k:
                raise ValueError(
                    f"route_prune_top_k must be <= router top_k {top_k}, "
                    f"got {self.route_prune_top_k}"
                )
            return self.route_prune_top_k < top_k
        return False

    def moe(
        self,
        layer: int,
        source: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        output: torch.Tensor | None = None
        page = self.pages[layer]
        use_grouped = source.shape[0] >= 128
        executed_assignments = int(top_indices.numel())
        route_prune_stats: dict[str, Any] | None = None
        route_prune_enabled = self._route_prune_enabled(top_indices.shape[1])
        if use_grouped:
            queue_plan: ExpertQueuePlan | None = None
            if route_prune_enabled:
                queue_plan, prune_record = grouped_route_pruned_queue(
                    top_indices,
                    top_weights,
                    num_experts=page.gate_up.shape[0],
                    min_weight=self.route_prune_min_weight or 0.0,
                    max_top_k=self.route_prune_top_k,
                    renormalize=self.route_prune_renormalize,
                )
                metadata_s = float(prune_record["metadata_s"])
                route_prune_stats = {
                    key: value for key, value in prune_record.items() if key != "metadata_s"
                }
                token_ids = queue_plan.token_ids
                coefficients = queue_plan.coefficients
                starts = queue_plan.starts
                counts = queue_plan.counts
                route_compiler = (
                    "expert-queue-pruned-cuda-v1"
                    if top_indices.is_cuda
                    else "expert-queue-pruned-torch-v1"
                )
            elif self.use_fused_expert_queue:
                queue_plan, metadata_s = grouped_route_queue(
                    top_indices,
                    top_weights,
                    num_experts=page.gate_up.shape[0],
                )
                token_ids = queue_plan.token_ids
                coefficients = queue_plan.coefficients
                starts = queue_plan.starts
                counts = queue_plan.counts
                route_compiler = (
                    "expert-queue-device-v1" if top_indices.is_cuda else "expert-queue-torch-v1"
                )
            else:
                token_ids, coefficients, starts, counts, metadata_s = grouped_route_tensors(
                    top_indices,
                    top_weights,
                    num_experts=page.gate_up.shape[0],
                )
                route_compiler = (
                    "triton-atomic-compact-v1" if top_indices.is_cuda else "torch-stable-sort-v1"
                )
            executed_assignments = int(token_ids.numel())
            rows = source.index_select(0, token_ids)
            rows_q, row_scales = quantize_fp8_activation(rows)
            gate_up, grouped_impl = grouped_fp8_mm(
                rows_q,
                page.gate_up,
                row_scales,
                page.gate_up_scales,
                starts,
                counts,
                # Top-k contains each expert at most once per source row.
                max_rows=source.shape[0],
                out_dtype=self.dtype,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            activated_q, activated_scales = quantize_fp8_activation(activated)
            expert_output, down_impl = grouped_fp8_mm(
                activated_q,
                page.down,
                activated_scales,
                page.down_scales,
                starts,
                counts,
                max_rows=source.shape[0],
                out_dtype=self.dtype,
            )
            if down_impl != grouped_impl:
                raise RuntimeError(f"grouped kernel mismatch: {grouped_impl} != {down_impl}")
            scatter_reduce = "torch-index-add"
            if queue_plan is not None and expert_output.is_cuda:
                active_epochs = torch.zeros(
                    source.shape[0],
                    device=source.device,
                    dtype=torch.int64,
                )
                output, _valid = scatter_committed_device(
                    queue_plan,
                    expert_output,
                    active_epochs=active_epochs,
                )
                scatter_reduce = "triton-epoch-scatter-v1"
            else:
                output = torch.zeros(
                    (source.shape[0], source.shape[1]),
                    device=source.device,
                    dtype=torch.float32,
                )
                contribution = expert_output.float() * coefficients.float().unsqueeze(1)
                output.index_add_(0, token_ids, contribution)
            self.stats.qmm_calls += 2
            self.stats.grouped_kernel_calls += 2
            self.stats.route_metadata_s += metadata_s
            grouped_record: dict[str, Any] = {
                "layer": layer,
                "rows": source.shape[0],
                "top_k": top_indices.shape[1],
                "assignments": executed_assignments,
                "original_assignments": int(top_indices.numel()),
                "counts": counts,
                "metadata_s": metadata_s,
                "route_compiler": route_compiler,
                "grouped_kernel": grouped_impl,
                "scatter_reduce": scatter_reduce,
            }
            if route_prune_stats is not None:
                grouped_record["route_prune"] = route_prune_stats
            self.stats.grouped_routes.append(grouped_record)
        else:
            output = torch.zeros(
                (source.shape[0], source.shape[1]),
                device=source.device,
                dtype=torch.float32,
            )
            if route_prune_enabled:
                queue_plan, prune_record = grouped_route_pruned_queue(
                    top_indices,
                    top_weights,
                    num_experts=page.gate_up.shape[0],
                    min_weight=self.route_prune_min_weight or 0.0,
                    max_top_k=self.route_prune_top_k,
                    renormalize=self.route_prune_renormalize,
                )
                metadata_s = float(prune_record["metadata_s"])
                route_prune_stats = {
                    key: value for key, value in prune_record.items() if key != "metadata_s"
                }
                assignments = expert_queue_assignments(queue_plan)
                route_stats = route_stats_from_queue(
                    queue_plan,
                    metadata_s=metadata_s,
                    original_assignments=int(top_indices.numel()),
                    route_prune=route_prune_stats,
                )
            else:
                assignments, route_stats = grouped_route_plan(top_indices, top_weights)
            for expert, token_ids, coefficients in assignments:
                rows = source.index_select(0, token_ids)
                gate_up = scaled_fp8_mm(
                    rows,
                    page.gate_up[expert],
                    page.gate_up_scales[expert],
                    out_dtype=self.dtype,
                )
                gate, up = gate_up.chunk(2, dim=-1)
                activated = F.silu(gate) * up
                expert_output = scaled_fp8_mm(
                    activated,
                    page.down[expert],
                    page.down_scales[expert],
                    out_dtype=self.dtype,
                )
                contribution = expert_output.float() * coefficients.float().unsqueeze(1)
                output.index_add_(0, token_ids, contribution)
            self.stats.qmm_calls += 2 * len(assignments)
            self.stats.individual_kernel_calls += 2 * len(assignments)
            self.stats.expert_dispatches += len(assignments)
            self.stats.route_metadata_s += float(route_stats["metadata_s"])
            self.stats.layer_rows.append({"layer": layer, **route_stats})
            executed_assignments = int(route_stats["assignments"])
        self.stats.expert_rows += executed_assignments
        if output is None:
            raise RuntimeError("MoE backend did not produce an output tensor")
        return output.to(self.dtype)


class BF16StreamExpertBackend:
    """Quality reference that streams original expert tensors one expert at a time."""

    def __init__(
        self,
        reader: TensorReader,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.reader = reader
        self.device = device
        self.dtype = dtype
        self.stats = BackendStats()

    def moe(
        self,
        layer: int,
        source: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        assignments, route_stats = grouped_route_plan(top_indices, top_weights)
        output = torch.zeros(
            (source.shape[0], source.shape[1]),
            device=source.device,
            dtype=torch.float32,
        )
        for expert, token_ids, coefficients in assignments:
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            gate = self.reader.get(f"{prefix}.gate_proj.weight")
            up = self.reader.get(f"{prefix}.up_proj.weight")
            down = self.reader.get(f"{prefix}.down_proj.weight")
            gate_up_weight = torch.cat((gate, up), dim=0).to(device=self.device, dtype=self.dtype)
            down_weight = down.to(device=self.device, dtype=self.dtype)
            rows = source.index_select(0, token_ids)
            gate_up = rows @ gate_up_weight.t()
            gate_values, up_values = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate_values) * up_values
            expert_output = activated @ down_weight.t()
            contribution = expert_output.float() * coefficients.float().unsqueeze(1)
            output.index_add_(0, token_ids, contribution)
            del gate, up, down, gate_up_weight, down_weight
        self.reader.release()
        self.stats.qmm_calls += 2 * len(assignments)
        self.stats.expert_dispatches += len(assignments)
        self.stats.expert_rows += int(top_indices.numel())
        self.stats.route_metadata_s += float(route_stats["metadata_s"])
        self.stats.layer_rows.append({"layer": layer, **route_stats})
        return output.to(self.dtype)


def rms_norm(source: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    values = source.float()
    normalized = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + eps)
    return weight * normalized.to(source.dtype)


def apply_residual_patch_ops(source: torch.Tensor, ops: list | tuple) -> torch.Tensor:
    """Apply residual-stream interventions after a complete decoder block."""
    for op, vector, _values in ops:
        if op != "proj_remove":
            raise ValueError(f"unknown residual patch op {op!r}")
        direction = torch.as_tensor(vector, dtype=torch.float32, device=source.device)
        direction = direction / (direction.norm() + 1e-9)
        projection = (source.float() @ direction).unsqueeze(-1) * direction
        source = (source.float() - projection).to(source.dtype)
    return source


def rotate_half(source: torch.Tensor) -> torch.Tensor:
    half = source.shape[-1] // 2
    return torch.cat((-source[..., half:], source[..., :half]), dim=-1)


def rope_tables(
    positions: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimensions = (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim
    )
    inverse = 1.0 / torch.pow(torch.tensor(theta, device=positions.device), dimensions)
    frequencies = torch.outer(positions.float(), inverse)
    embeddings = torch.cat((frequencies, frequencies), dim=-1)
    return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


@dataclass
class ForwardResult:
    logits: torch.Tensor
    cache: list[LayerKV]
    routes: list[torch.Tensor]
    hidden_states: list[torch.Tensor]


class OLMoECudaRuntime:
    def __init__(
        self,
        skeleton: ResidentSkeleton,
        expert_backend: FP8ExpertBackend | BF16StreamExpertBackend,
    ):
        self.skeleton = skeleton
        self.backend = expert_backend
        self.cfg = skeleton.cfg
        self.dtype = skeleton.dtype
        self.device = skeleton.device
        self.heads = int(self.cfg["num_attention_heads"])
        self.kv_heads = int(self.cfg.get("num_key_value_heads", self.heads))
        self.hidden = int(self.cfg["hidden_size"])
        self.head_dim = self.hidden // self.heads
        self.top_k = int(self.cfg["num_experts_per_tok"])
        self.eps = float(self.cfg.get("rms_norm_eps") or 1e-5)
        self.theta = float(self.cfg.get("rope_theta") or 10000.0)
        self.norm_topk = bool(self.cfg.get("norm_topk_prob", False))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past: list[LayerKV] | None = None,
        all_logits: bool = False,
        capture_routes: bool = False,
        capture_hidden_states: bool = False,
        resid_patch_ops_by_layer: dict[int, list] | None = None,
        phase_times_ms: dict[str, list[float]] | None = None,
    ) -> ForwardResult:
        if input_ids.ndim != 2:
            raise ValueError(f"expected [batch,tokens], got {tuple(input_ids.shape)}")
        batch, tokens = input_ids.shape
        if past is not None and tokens != 1:
            raise NotImplementedError("cached continuation currently accepts one token per request")
        hidden = self.skeleton.embedding[input_ids]
        captured_hidden = [hidden.detach().cpu()] if capture_hidden_states else []
        offset = 0 if past is None else int(past[0].key.shape[2])
        positions = torch.arange(
            offset,
            offset + tokens,
            device=input_ids.device,
            dtype=torch.long,
        )
        cosine, sine = rope_tables(positions, self.head_dim, self.theta, self.dtype)
        cosine = cosine[None, :, None, :]
        sine = sine[None, :, None, :]
        new_cache: list[LayerKV] = []
        captured_routes: list[torch.Tensor] = []

        def timed(label: str, fn):
            if phase_times_ms is None:
                return fn()
            if input_ids.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                value = fn()
                end.record()
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
            else:
                started = time.perf_counter()
                value = fn()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            phase_times_ms.setdefault(label, []).append(elapsed_ms)
            return value

        for layer_index, layer in enumerate(self.skeleton.layers):

            def build_attention_inputs(
                hidden_in=hidden,
                layer_in=layer,
                layer_index_in=layer_index,
            ):
                normalized_local = rms_norm(hidden_in, layer_in.input_norm, self.eps)
                query_local = rms_norm(
                    normalized_local @ layer_in.q_proj.t(),
                    layer_in.q_norm,
                    self.eps,
                )
                key_local = rms_norm(
                    normalized_local @ layer_in.k_proj.t(),
                    layer_in.k_norm,
                    self.eps,
                )
                value_local = normalized_local @ layer_in.v_proj.t()
                query_local = query_local.reshape(batch, tokens, self.heads, self.head_dim)
                key_local = key_local.reshape(batch, tokens, self.kv_heads, self.head_dim)
                value_local = value_local.reshape(batch, tokens, self.kv_heads, self.head_dim)
                query_local = query_local * cosine + rotate_half(query_local) * sine
                key_local = key_local * cosine + rotate_half(key_local) * sine
                query_local = query_local.transpose(1, 2)
                key_local = key_local.transpose(1, 2)
                value_local = value_local.transpose(1, 2)
                if past is not None:
                    key_local = torch.cat((past[layer_index_in].key, key_local), dim=2)
                    value_local = torch.cat((past[layer_index_in].value, value_local), dim=2)
                return query_local, key_local, value_local

            query, key, value = timed("attention_qkv_kv_update", build_attention_inputs)
            new_cache.append(LayerKV(key=key, value=value))

            def run_attention(
                hidden_in=hidden,
                layer_in=layer,
                query_in=query,
                key_in=key,
                value_in=value,
            ):
                attention_key = key_in
                attention_value = value_in
                if self.kv_heads != self.heads:
                    repeat = self.heads // self.kv_heads
                    attention_key = attention_key.repeat_interleave(repeat, dim=1)
                    attention_value = attention_value.repeat_interleave(repeat, dim=1)
                context = F.scaled_dot_product_attention(
                    query_in,
                    attention_key,
                    attention_value,
                    dropout_p=0.0,
                    is_causal=past is None and tokens > 1,
                )
                context = context.transpose(1, 2).reshape(batch, tokens, self.hidden)
                return hidden_in + context @ layer_in.o_proj.t()

            residual = timed("attention_sdpa_out", run_attention)

            def run_router(residual_in=residual, layer_in=layer):
                moe_input = rms_norm(residual_in, layer_in.post_norm, self.eps)
                flat = moe_input.reshape(batch * tokens, self.hidden)
                router_logits = flat @ layer_in.router.t()
                probabilities = torch.softmax(router_logits.float(), dim=-1)
                weights, indices = probabilities.topk(self.top_k, dim=-1)
                if self.norm_topk:
                    weights = weights / weights.sum(-1, keepdim=True)
                return flat, indices, weights.to(self.dtype)

            flat_input, top_indices, top_weights = timed("router_topk", run_router)
            if capture_routes:
                captured_routes.append(top_indices.detach().cpu())

            def run_moe(
                layer_index_in=layer_index,
                flat_input_in=flat_input,
                top_indices_in=top_indices,
                top_weights_in=top_weights,
            ):
                return self.backend.moe(
                    layer_index_in,
                    flat_input_in,
                    top_indices_in,
                    top_weights_in,
                ).reshape(batch, tokens, self.hidden)

            moe_output = timed("moe_expert", run_moe)
            hidden = residual + moe_output
            hidden = apply_residual_patch_ops(
                hidden, (resid_patch_ops_by_layer or {}).get(layer_index, [])
            )
            if capture_hidden_states:
                captured_hidden.append(hidden.detach().cpu())

        projected = timed(
            "final_norm",
            lambda hidden_in=hidden: rms_norm(
                hidden_in,
                self.skeleton.final_norm,
                self.eps,
            ),
        )
        if capture_hidden_states:
            captured_hidden[-1] = projected.detach().cpu()
        if not all_logits:
            projected = projected[:, -1, :]
        logits = timed("lm_head", lambda: projected @ self.skeleton.lm_head.t())
        return ForwardResult(
            logits=logits,
            cache=new_cache,
            routes=captured_routes,
            hidden_states=captured_hidden,
        )


def exact_prompt_ids(
    tokenizer: Any,
    *,
    batch: int,
    context: int,
    device: str,
) -> torch.Tensor:
    rows: list[list[int]] = []
    for index in range(batch):
        template = PROMPT_TEMPLATES[index % len(PROMPT_TEMPLATES)]
        text = f"Request {index}. " + template.format(index=index)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise RuntimeError("tokenizer returned no token ids")
        repeats = (context + len(token_ids) - 1) // len(token_ids)
        rows.append((token_ids * repeats)[:context])
    return torch.tensor(rows, device=device, dtype=torch.long)


def next_token_ce(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    rows = logits[:, :-1, :].float().reshape(-1, logits.shape[-1])
    labels = input_ids[:, 1:].reshape(-1)
    return float(F.cross_entropy(rows, labels).item())


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    left = reference.float()
    right = candidate.float()
    relative_l2 = tensor_relative_l2(right, left)
    cosine = F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item()
    return {
        "relative_l2": relative_l2,
        "cosine": float(cosine),
        "max_abs": float((right - left).abs().amax().item()),
        "position_top1_agreement": float(
            (left.argmax(dim=-1) == right.argmax(dim=-1)).float().mean().item()
        ),
        "last_position_top1_agreement": float(
            (left[:, -1].argmax(dim=-1) == right[:, -1].argmax(dim=-1)).float().mean().item()
        ),
        "finite": bool(torch.isfinite(right).all().item()),
    }


def compare_routes(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> dict[str, Any]:
    slot_agreement: list[float] = []
    set_overlap: list[float] = []
    for left, right in zip(reference, candidate, strict=True):
        slot_agreement.append(float((left == right).float().mean().item()))
        row_overlaps: list[float] = []
        for left_row, right_row in zip(left.tolist(), right.tolist(), strict=True):
            row_overlaps.append(len(set(left_row) & set(right_row)) / len(left_row))
        set_overlap.append(float(statistics.mean(row_overlaps)))
    return {
        "slot_agreement_mean": float(statistics.mean(slot_agreement)),
        "topk_set_overlap_mean": float(statistics.mean(set_overlap)),
        "per_layer_slot_agreement": slot_agreement,
        "per_layer_topk_set_overlap": set_overlap,
    }


@torch.no_grad()
def greedy_tokens(
    runtime: OLMoECudaRuntime,
    input_ids: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    first = runtime.forward(input_ids)
    token = first.logits.argmax(dim=-1)
    generated = [token]
    cache = first.cache
    for _ in range(steps - 1):
        result = runtime.forward(token[:, None], past=cache)
        cache = result.cache
        token = result.logits.argmax(dim=-1)
        generated.append(token)
    return torch.stack(generated, dim=1)


def resolve_olmoe_model_dir(model_name: str | Path) -> Path:
    """Resolve an explicit checkpoint directory or an ``mrun`` model-registry name."""
    candidate = Path(model_name).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return snapshot_dir(str(model_name))


def resolve_olmoe_store_dir(
    model_dir: Path,
    store_dir: str | Path | None = None,
) -> Path:
    """Resolve the compiled FP8 store without hiding its model-derived identity."""
    if store_dir is not None:
        return Path(store_dir).expanduser()
    configured = os.environ.get("MRUN_OLMOE_STORE_DIR")
    if configured:
        return Path(configured).expanduser()
    return stores_root() / f"{model_dir.name}-fp8-e4m3fn-v1"


class OLMoECudaEngine(BaseEngine):
    """Public ``mrun`` adapter around the proven all-layer OLMoE FP8 runtime.

    The adapter exposes logits, greedy generation, residual states, and residual projection
    interventions. Dense-MLP activation taps and training remain unavailable and fail explicitly.
    Instantiate it directly or through ``open_engine(..., backend='olmoe-cuda')``.
    """

    backend = "olmoe-cuda"
    arch = "olmoe"
    supports_batch = True

    def __init__(
        self,
        model_name: str,
        *,
        store_dir: str | Path | None = None,
        source_model_name: str | None = None,
        source_hf_id: str | None = None,
        source_revision: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        build_store: bool = False,
        use_fused_expert_queue: bool | None = None,
        **_ignored: Any,
    ) -> None:
        if not device.startswith("cuda"):
            raise ValueError("the OLMoE FP8 runtime currently requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("the OLMoE FP8 runtime requires CUDA")

        model_candidate = Path(model_name).expanduser()
        if model_candidate.is_dir():
            if source_hf_id is None:
                raise ValueError("explicit OLMoE model paths require source_hf_id provenance")
            lineage_name = source_model_name or model_candidate.name
            lineage_hf_id = source_hf_id
        else:
            source_spec = resolve_model(model_name)
            lineage_name = source_model_name or source_spec.name
            lineage_hf_id = source_hf_id or source_spec.hf_id
        self.model_dir = resolve_olmoe_model_dir(model_name)
        self.store_dir = resolve_olmoe_store_dir(self.model_dir, store_dir)
        manifest_path = self.store_dir / "manifest.json"
        if not manifest_path.exists() and not build_store:
            raise FileNotFoundError(
                f"no OLMoE FP8 store at {self.store_dir}. Build it with "
                "`mrun olmoe --phase build-store --model-dir MODEL --store-dir STORE "
                "--output RESULT.json` or pass build_store=True."
            )
        self.manifest = build_fp8_store(
            self.model_dir,
            self.store_dir,
            model_name=lineage_name,
            hf_id=lineage_hf_id,
            revision=source_revision,
        )
        self.cfg = json.loads((self.model_dir / "config.json").read_text())
        self.skeleton: ResidentSkeleton | None = ResidentSkeleton(
            TensorReader(self.model_dir),
            self.cfg,
            device=device,
            dtype=dtype,
        )
        expert_store = FP8ExpertStore(self.store_dir)
        try:
            pages, self.load_info = expert_store.load_device(device)
        finally:
            expert_store.close()
        self.expert_backend: FP8ExpertBackend | None = FP8ExpertBackend(
            pages,
            dtype=dtype,
            use_fused_expert_queue=use_fused_expert_queue,
        )
        self.runtime: OLMoECudaRuntime | None = OLMoECudaRuntime(
            self.skeleton,
            self.expert_backend,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.name = str(model_name)
        self.device = device
        self.dtype = dtype
        self.n_layer = int(self.cfg["num_hidden_layers"])
        self.inter = int(self.cfg["intermediate_size"])
        self.hidden = int(self.cfg["hidden_size"])
        self.working_set_mb = float(self.load_info["device_bytes"]) / 1e6

    def _require_runtime(self) -> OLMoECudaRuntime:
        if self.runtime is None:
            raise RuntimeError("OLMoECudaEngine is closed")
        return self.runtime

    @torch.no_grad()
    def logits(self, ids: np.ndarray) -> torch.Tensor:
        row = np.asarray(ids, dtype=np.int64)
        if row.ndim != 1 or not row.size:
            raise ValueError("ids must be a non-empty one-dimensional token array")
        input_ids = torch.as_tensor(row, device=self.device, dtype=torch.long)[None, :]
        return self._require_runtime().forward(input_ids, all_logits=True).logits[0].cpu()

    @torch.no_grad()
    def logits_batch(self, ids_list: list[np.ndarray]) -> list[torch.Tensor]:
        """Fuse rows with equal lengths; never pad because this runtime has no mask input."""
        outputs: list[torch.Tensor | None] = [None] * len(ids_list)
        by_length: dict[int, list[int]] = {}
        for index, ids in enumerate(ids_list):
            row = np.asarray(ids, dtype=np.int64)
            if row.ndim != 1 or not row.size:
                raise ValueError("each ids row must be non-empty and one-dimensional")
            by_length.setdefault(int(row.size), []).append(index)
        for indices in by_length.values():
            batch = np.stack([np.asarray(ids_list[index], dtype=np.int64) for index in indices])
            input_ids = torch.as_tensor(batch, device=self.device, dtype=torch.long)
            logits = self._require_runtime().forward(input_ids, all_logits=True).logits.cpu()
            for offset, index in enumerate(indices):
                outputs[index] = logits[offset]
        return [output for output in outputs if output is not None]

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: np.ndarray | torch.Tensor,
        *,
        max_new_tokens: int = 48,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        rows = torch.as_tensor(input_ids, device=self.device, dtype=torch.long)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.ndim != 2 or rows.shape[1] == 0:
            raise ValueError("input_ids must have shape [tokens] or [batch,tokens]")
        return greedy_tokens(
            self._require_runtime(),
            rows,
            steps=max_new_tokens,
        ).cpu()

    def generate(
        self,
        prompt: str | list[int] | np.ndarray,
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        return_text: bool = False,
    ) -> list[int] | str:
        if isinstance(prompt, str):
            ids = self.tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
            )["input_ids"]
        else:
            ids = [int(token) for token in np.asarray(prompt).tolist()]
        generated = self.generate_ids(
            np.asarray(ids, dtype=np.int64),
            max_new_tokens=max_new_tokens,
        )[0].tolist()
        eos = eos_token_id
        if eos is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None and eos in generated:
            generated = generated[: generated.index(eos) + 1]
        if return_text:
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        return generated

    def generate_batch(
        self,
        prompts: Sequence[str | Sequence[int] | np.ndarray],
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        return_text: bool = False,
    ) -> list[list[int]] | list[str]:
        """Generate equal-length prompt groups through one persistent-KV decode."""

        encoded: list[np.ndarray] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                row = np.asarray(
                    self.tokenizer(
                        prompt,
                        add_special_tokens=add_special_tokens,
                    )["input_ids"],
                    dtype=np.int64,
                )
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

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            logits=True,
            logits_batch=True,
            mlp_acts=False,
            residual_tap=True,
            residual_tap_batch=True,
            approximate_quantized=True,
            generation=True,
            generation_batch=True,
            persistent_kv=True,
            compact_fused_weights=True,
            grouped_moe=True,
            fp8_execution=True,
            route_first_moe=True,
        )

    @torch.no_grad()
    def hidden_states(self, ids: np.ndarray) -> list[torch.Tensor]:
        row = np.asarray(ids, dtype=np.int64)
        if row.ndim != 1 or not row.size:
            raise ValueError("ids must be a non-empty one-dimensional token array")
        input_ids = torch.as_tensor(row, device=self.device, dtype=torch.long)[None, :]
        result = self._require_runtime().forward(
            input_ids,
            all_logits=True,
            capture_hidden_states=True,
        )
        return [state[0] for state in result.hidden_states]

    @torch.no_grad()
    def hidden_states_batch(self, ids_list: list[np.ndarray]) -> list[list[torch.Tensor]]:
        """Fuse equal-length residual captures and restore the caller's row order.

        OLMoE's runtime already carries a real batch dimension. We bucket instead of
        padding because its causal attention surface does not accept an attention mask.
        """

        rows = [np.asarray(ids, dtype=np.int64) for ids in ids_list]
        if not rows:
            return []
        for row in rows:
            if row.ndim != 1 or not row.size:
                raise ValueError("ids must be a non-empty one-dimensional token array")
        outputs: list[list[torch.Tensor] | None] = [None] * len(rows)
        by_length: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            by_length.setdefault(int(row.size), []).append(index)
        for indices in by_length.values():
            input_ids = torch.as_tensor(
                np.stack([rows[index] for index in indices]),
                device=self.device,
                dtype=torch.long,
            )
            result = self._require_runtime().forward(
                input_ids,
                all_logits=False,
                capture_hidden_states=True,
            )
            for batch_index, output_index in enumerate(indices):
                outputs[output_index] = [
                    state[batch_index].detach().cpu() for state in result.hidden_states
                ]
        if any(output is None for output in outputs):
            raise RuntimeError("batched residual capture did not populate every row")
        return [output for output in outputs if output is not None]

    def forward_acts(self, _ids: np.ndarray) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError("olmoe-cuda does not expose activation taps")

    def forward_patched(
        self,
        ids: np.ndarray,
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        resid_patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]:
        if patch_ops_by_layer or selected_maps or collect_acts:
            raise NotImplementedError(
                "olmoe-cuda exposes residual interventions, not dense-MLP activation taps"
            )
        row = np.asarray(ids, dtype=np.int64)
        if row.ndim != 1 or not row.size:
            raise ValueError("ids must be a non-empty one-dimensional token array")
        input_ids = torch.as_tensor(row, device=self.device, dtype=torch.long)[None, :]
        result = self._require_runtime().forward(
            input_ids,
            all_logits=True,
            resid_patch_ops_by_layer=resid_patch_ops_by_layer,
        )
        return result.logits[0].cpu(), [], {}

    @torch.no_grad()
    def forward_patched_batch(
        self,
        ids_list: list[np.ndarray],
        *,
        patch_ops_by_layer: dict[int, list] | None = None,
        head_patch_ops_by_layer: dict[int, list] | None = None,
        resid_patch_ops_by_layer: dict[int, list] | None = None,
        selected_maps: dict[int, dict[str, Any]] | None = None,
        collect_acts: bool = False,
    ) -> list[tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]]:
        """Apply one shared residual intervention to equal-length row buckets."""

        if patch_ops_by_layer or head_patch_ops_by_layer or selected_maps or collect_acts:
            raise NotImplementedError(
                "olmoe-cuda batched patching exposes residual interventions only"
            )
        rows = [np.asarray(ids, dtype=np.int64) for ids in ids_list]
        if not rows:
            return []
        for row in rows:
            if row.ndim != 1 or not row.size:
                raise ValueError("ids must be a non-empty one-dimensional token array")
        outputs: list[tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]] | None] = [
            None
        ] * len(rows)
        by_length: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            by_length.setdefault(int(row.size), []).append(index)
        for indices in by_length.values():
            input_ids = torch.as_tensor(
                np.stack([rows[index] for index in indices]),
                device=self.device,
                dtype=torch.long,
            )
            result = self._require_runtime().forward(
                input_ids,
                all_logits=True,
                resid_patch_ops_by_layer=resid_patch_ops_by_layer,
            )
            for batch_index, output_index in enumerate(indices):
                outputs[output_index] = (result.logits[batch_index].cpu(), [], {})
        if any(output is None for output in outputs):
            raise RuntimeError("batched residual patch did not populate every row")
        return [output for output in outputs if output is not None]

    def down_weight(self, _layer: int) -> torch.Tensor:
        raise NotImplementedError("an MoE layer has per-expert down weights")

    def close(self) -> None:
        self.runtime = None
        self.expert_backend = None
        self.skeleton = None


def run_quality(
    model_dir: Path,
    cfg: dict[str, Any],
    skeleton: ResidentSkeleton,
    fp8_backend: FP8ExpertBackend,
    tokenizer: Any,
    *,
    batch: int,
    context: int,
    greedy_steps: int,
) -> dict[str, Any]:
    input_ids = exact_prompt_ids(
        tokenizer,
        batch=batch,
        context=context,
        device=skeleton.device,
    )
    fp8_backend.stats.reset()
    fp8_runtime = OLMoECudaRuntime(skeleton, fp8_backend)
    fp8_result = fp8_runtime.forward(
        input_ids,
        all_logits=True,
        capture_routes=True,
    )
    torch.cuda.synchronize()

    reference_backend = BF16StreamExpertBackend(TensorReader(model_dir))
    reference_runtime = OLMoECudaRuntime(skeleton, reference_backend)
    reference_result = reference_runtime.forward(
        input_ids,
        all_logits=True,
        capture_routes=True,
    )
    torch.cuda.synchronize()

    fp8_ce = next_token_ce(fp8_result.logits, input_ids)
    reference_ce = next_token_ce(reference_result.logits, input_ids)
    logits = compare_logits(reference_result.logits, fp8_result.logits)
    routes = compare_routes(reference_result.routes, fp8_result.routes)

    fp8_backend.stats.reset()
    reference_backend.stats.reset()
    fp8_generated = greedy_tokens(fp8_runtime, input_ids[:1], steps=greedy_steps)
    reference_generated = greedy_tokens(reference_runtime, input_ids[:1], steps=greedy_steps)
    torch.cuda.synchronize()
    greedy_agreement = float((fp8_generated == reference_generated).float().mean().item())
    quality_pass = logits["finite"] and fp8_ce - reference_ce <= 0.05 and logits["cosine"] >= 0.995
    return {
        "phase": "quality",
        "batch": batch,
        "context": context,
        "greedy_steps": greedy_steps,
        "input_ids_sha256": hashlib.sha256(input_ids.detach().cpu().numpy().tobytes()).hexdigest(),
        "reference_next_token_ce": reference_ce,
        "fp8_next_token_ce": fp8_ce,
        "ce_delta": fp8_ce - reference_ce,
        "logits": logits,
        "routes": routes,
        "greedy_token_agreement": greedy_agreement,
        "fp8_generated": fp8_generated.detach().cpu().tolist(),
        "reference_generated": reference_generated.detach().cpu().tolist(),
        "fp8_backend": fp8_backend.stats.as_dict(),
        "reference_backend": reference_backend.stats.as_dict(),
        "quality_gate": {
            "ce_delta_max": 0.05,
            "logit_cosine_min": 0.995,
            "pass": quality_pass,
        },
        "pass": quality_pass,
    }


@torch.no_grad()
def decode_sample(
    runtime: OLMoECudaRuntime,
    input_ids: torch.Tensor,
    *,
    steps: int,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    prefill_started = time.perf_counter()
    prefill = runtime.forward(input_ids)
    token = prefill.logits.argmax(dim=-1)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - prefill_started
    cache = prefill.cache
    generated = [token]
    step_times: list[float] = []
    for _ in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = runtime.forward(token[:, None], past=cache)
        token = result.logits.argmax(dim=-1)
        cache = result.cache
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - started)
        generated.append(token)
    generated_tensor = torch.stack(generated, dim=1)
    decode_s = sum(step_times)
    return {
        "prefill_s": prefill_s,
        "time_to_first_token_s": prefill_s,
        "decode_step_s": step_times,
        "decode_total_s": decode_s,
        "decode_tokens_per_s": input_ids.shape[0] * steps / decode_s,
        "inter_token_latency_median_s": statistics.median(step_times),
        "inter_token_latency_p95_s": percentile(step_times, 0.95),
        "end_to_end_tokens_per_s": (input_ids.shape[0] * (steps + 1) / (prefill_s + decode_s)),
        "generated_sha256": hashlib.sha256(
            generated_tensor.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "finite": bool(torch.isfinite(prefill.logits).all().item()),
    }


def summarize_decode_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": samples,
        "decode_tokens_per_s_median": statistics.median(
            sample["decode_tokens_per_s"] for sample in samples
        ),
        "decode_tokens_per_s_min": min(sample["decode_tokens_per_s"] for sample in samples),
        "prefill_s_median": statistics.median(sample["prefill_s"] for sample in samples),
        "inter_token_latency_median_s": statistics.median(
            sample["inter_token_latency_median_s"] for sample in samples
        ),
        "inter_token_latency_p95_s": max(sample["inter_token_latency_p95_s"] for sample in samples),
        "end_to_end_tokens_per_s_median": statistics.median(
            sample["end_to_end_tokens_per_s"] for sample in samples
        ),
        "finite": all(sample["finite"] for sample in samples),
    }


def summarize_phase_times(samples: list[dict[str, list[float]]]) -> dict[str, Any]:
    labels = sorted({label for sample in samples for label in sample})
    phases: dict[str, dict[str, float]] = {}
    totals: list[float] = []
    for sample in samples:
        total = sum(sum(values) for values in sample.values())
        totals.append(total)
    median_total = statistics.median(totals)
    for label in labels:
        per_sample = [sum(sample.get(label, [])) for sample in samples]
        median_ms = statistics.median(per_sample)
        phases[label] = {
            "median_ms": median_ms,
            "min_ms": min(per_sample),
            "max_ms": max(per_sample),
            "share_of_profiled_median": median_ms / max(median_total, 1e-9),
        }
    ranked = sorted(
        ({"phase": label, **values} for label, values in phases.items()),
        key=lambda item: item["median_ms"],
        reverse=True,
    )
    return {
        "phases": phases,
        "ranked": ranked,
        "profiled_total_median_ms": median_total,
        "profiled_total_min_ms": min(totals),
        "profiled_total_max_ms": max(totals),
        "top_two_share": sum(item["median_ms"] for item in ranked[:2]) / max(median_total, 1e-9),
    }


def run_decode_sweep(
    skeleton: ResidentSkeleton,
    backend: FP8ExpertBackend,
    tokenizer: Any,
    *,
    batch_sizes: Iterable[int],
    context: int,
    steps: int,
    repeats: int,
) -> dict[str, Any]:
    runtime = OLMoECudaRuntime(skeleton, backend)
    rows: list[dict[str, Any]] = []
    for batch in batch_sizes:
        input_ids = exact_prompt_ids(
            tokenizer,
            batch=batch,
            context=context,
            device=skeleton.device,
        )
        backend.stats.reset()
        _ = decode_sample(runtime, input_ids, steps=1)
        backend.stats.reset()
        torch.cuda.reset_peak_memory_stats()
        samples = [decode_sample(runtime, input_ids, steps=steps) for _ in range(repeats)]
        summary = summarize_decode_samples(samples)
        summary.update(
            {
                "batch": batch,
                "context": context,
                "decode_steps": steps,
                "backend": backend.stats.as_dict(),
                "peak_cuda_allocated_mb": (torch.cuda.max_memory_allocated() / (1024 * 1024)),
                "peak_cuda_reserved_mb": (torch.cuda.max_memory_reserved() / (1024 * 1024)),
            }
        )
        rows.append(summary)
        print(
            f"decode B={batch}: {summary['decode_tokens_per_s_median']:.2f} tok/s "
            f"ITL={summary['inter_token_latency_median_s'] * 1e3:.2f} ms",
            flush=True,
        )

    baseline = rows[0]["decode_tokens_per_s_median"]
    if rows[0]["batch"] != 1:
        raise ValueError("decode sweep must begin with B=1")
    for row in rows:
        row["aggregate_speedup_vs_b1"] = row["decode_tokens_per_s_median"] / baseline
    best = max(rows, key=lambda row: row["aggregate_speedup_vs_b1"])
    return {
        "phase": "decode",
        "context": context,
        "decode_steps": steps,
        "repeats": repeats,
        "rows": rows,
        "b1_decode_tokens_per_s": baseline,
        "best_batch": best["batch"],
        "best_aggregate_speedup_vs_b1": best["aggregate_speedup_vs_b1"],
        "target_100x_pass": best["aggregate_speedup_vs_b1"] >= 100.0,
        "pass": all(row["finite"] for row in rows),
    }


def run_profile_sweep(
    skeleton: ResidentSkeleton,
    backend: FP8ExpertBackend,
    tokenizer: Any,
    *,
    batch_sizes: Iterable[int],
    context: int,
    repeats: int,
) -> dict[str, Any]:
    runtime = OLMoECudaRuntime(skeleton, backend)
    rows: list[dict[str, Any]] = []
    for batch in batch_sizes:
        input_ids = exact_prompt_ids(
            tokenizer,
            batch=batch,
            context=context,
            device=skeleton.device,
        )
        backend.stats.reset()
        prefill = runtime.forward(input_ids)
        token = prefill.logits.argmax(dim=-1)
        cache = prefill.cache
        samples: list[dict[str, list[float]]] = []
        wall_ms: list[float] = []
        for _ in range(repeats):
            phase_times: dict[str, list[float]] = {}
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = runtime.forward(
                token[:, None],
                past=cache,
                phase_times_ms=phase_times,
            )
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - started) * 1000.0)
            samples.append(phase_times)
            token = result.logits.argmax(dim=-1)
            cache = result.cache
        summary = summarize_phase_times(samples)
        summary.update(
            {
                "batch": batch,
                "context": context,
                "repeats": repeats,
                "wall_median_ms": statistics.median(wall_ms),
                "wall_min_ms": min(wall_ms),
                "wall_max_ms": max(wall_ms),
                "profiled_to_wall_ratio": (
                    summary["profiled_total_median_ms"] / max(statistics.median(wall_ms), 1e-9)
                ),
                "backend": backend.stats.as_dict(),
                "peak_cuda_allocated_mb": (torch.cuda.max_memory_allocated() / (1024 * 1024)),
                "peak_cuda_reserved_mb": (torch.cuda.max_memory_reserved() / (1024 * 1024)),
            }
        )
        rows.append(summary)
        print(
            f"profile B={batch}: wall={summary['wall_median_ms']:.2f} ms "
            f"top={summary['ranked'][0]['phase']} "
            f"{summary['ranked'][0]['median_ms']:.2f} ms",
            flush=True,
        )
    return {
        "phase": "profile",
        "context": context,
        "repeats": repeats,
        "rows": rows,
        "pass": all(0.5 <= row["profiled_to_wall_ratio"] <= 1.5 for row in rows),
    }


def gpu_snapshot() -> dict[str, Any]:
    query = (
        "name,memory.total,memory.used,utilization.gpu,"
        "pcie.link.gen.current,pcie.link.width.current"
    )
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
        return {
            "name": fields[0],
            "memory_total_mb": int(fields[1]),
            "memory_used_mb": int(fields[2]),
            "utilization_percent": int(fields[3]),
            "pcie_generation": int(fields[4]),
            "pcie_width": int(fields[5]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"error": repr(error)}


def parse_batch_sizes(value: str) -> list[int]:
    sizes = sorted(set(int(item.strip()) for item in value.split(",") if item.strip()))
    if not sizes or sizes[0] != 1 or any(size <= 0 for size in sizes):
        raise ValueError("batch sizes must be positive and include B=1")
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("build-store", "quality", "decode", "profile"),
        required=True,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--model-name",
        default="olmoe-1b-7b",
        help="mrun registry name recorded in source provenance",
    )
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,8,128")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--quality-batch", type=int, default=2)
    parser.add_argument("--quality-context", type=int, default=16)
    parser.add_argument("--quality-greedy-steps", type=int, default=4)
    parser.add_argument(
        "--fused-expert-queue",
        action="store_true",
        help=(
            "use the Expert Exchange fused device queue and epoch-gated scatter/reduce "
            "for grouped FP8 MoE layers"
        ),
    )
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_spec = resolve_model(args.model_name)
    manifest = build_fp8_store(
        args.model_dir,
        args.store_dir,
        model_name=source_spec.name,
        hf_id=source_spec.hf_id,
        revision=args.source_revision,
    )
    environment = {
        "created_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_before": gpu_snapshot(),
    }
    if args.phase == "build-store":
        phase_result: dict[str, Any] = {
            "phase": "build-store",
            "store_dir": str(args.store_dir),
            "data_bytes": int(manifest["data_bytes"]),
            "data_sha256": manifest["data_sha256"],
            "build_s": manifest["build_s"],
            "pass": True,
        }
        load_info = None
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        cfg = json.loads((args.model_dir / "config.json").read_text())
        skeleton_reader = TensorReader(args.model_dir)
        skeleton = ResidentSkeleton(skeleton_reader, cfg)
        expert_store = FP8ExpertStore(args.store_dir)
        pages, load_info = expert_store.load_device()
        expert_store.close()
        fp8_backend = FP8ExpertBackend(
            pages,
            use_fused_expert_queue=args.fused_expert_queue,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        if args.phase == "quality":
            phase_result = run_quality(
                args.model_dir,
                cfg,
                skeleton,
                fp8_backend,
                tokenizer,
                batch=args.quality_batch,
                context=args.quality_context,
                greedy_steps=args.quality_greedy_steps,
            )
        elif args.phase == "decode":
            phase_result = run_decode_sweep(
                skeleton,
                fp8_backend,
                tokenizer,
                batch_sizes=parse_batch_sizes(args.batch_sizes),
                context=args.context,
                steps=args.steps,
                repeats=args.repeats,
            )
        else:
            phase_result = run_profile_sweep(
                skeleton,
                fp8_backend,
                tokenizer,
                batch_sizes=parse_batch_sizes(args.batch_sizes),
                context=args.context,
                repeats=args.repeats,
            )
        torch.cuda.synchronize()

    result = {
        "schema_version": 1,
        "environment": environment,
        "model": {
            "path": str(args.model_dir),
            "config_sha256": sha256_file(args.model_dir / "config.json"),
        },
        "store": {
            "path": str(args.store_dir),
            "codec": manifest["codec"],
            "data_bytes": int(manifest["data_bytes"]),
            "data_sha256": manifest["data_sha256"],
            "load": load_info,
        },
        "result": phase_result,
        "elapsed_s": time.perf_counter() - started,
        "gpu_after": gpu_snapshot(),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0 if phase_result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
