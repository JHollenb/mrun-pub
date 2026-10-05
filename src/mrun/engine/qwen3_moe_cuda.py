from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

# Re-exported deliberately: `dequantize_block_fp8` is part of this module's reader surface and
# callers (and its tests) import it from here.
from ..fp8 import declared_block_size, dequantize_block_fp8, scale_key_for
from ..models import resolve_model, snapshot_dir
from ..paths import stores_root
from ..store_provenance import (
    build_builder_provenance,
    build_derived_provenance,
    build_source_provenance,
    canonical_sha256,
    validate_source_provenance,
    verify_builder_provenance,
    verify_derived_provenance,
    verify_source_provenance,
)
from ._base_impl import BaseEngine
from .base import KEEP, EngineCapabilities
from .kernels.segmented_gqa_decode import (
    scatter_segmented_decode_kv,
    segmented_gqa_decode,
    segmented_gqa_decode_cow,
)
from .olmoe_cuda import grouped_route_tensors, grouped_route_tensors_with_inverse

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


EXPERT_CODEC_FP8 = "fp8"
EXPERT_CODEC_W4 = "w4"
SUPPORTED_EXPERT_CODECS = frozenset({EXPERT_CODEC_FP8, EXPERT_CODEC_W4})

FP8_STORE_SCHEMA = "qwen3-moe-packed-expert-store-v1"
FP8_STORE_CODEC = "rowwise-e4m3fn-paged-v1"
FP8_STORE_FILE = "experts.fp8"
W4_STORE_SCHEMA = "qwen3-moe-packed-int4-expert-store-v1"
W4_STORE_CODEC = "symmetric-int4-offset8-g128-f32-paged-v1"
W4_STORE_FILE = "experts.i4"
W4_GROUP_SIZE = 128
# Exact-shape RTX 4080 tuner job-69ca89a00b83 selected this one static launch across
# gate/up and down at B1 and both diverse/reuse B8 route profiles. It was bit-exact
# to the former M16/N32/K32/W4/S3 output and 2.44x--3.30x faster in the sealed panel.
W4_GROUPED_LAUNCH_CONFIG = (32, 32, 32, 4, 2)
W4_PREDOT_BF16_ARITHMETIC_POLICY = "w4-g128-predot-bf16-v1"
W4_POSTSCALE_BF16_ARITHMETIC_POLICY = "w4-g128-postscale-bf16-v1"
DEFAULT_W4_ARITHMETIC_POLICY = W4_PREDOT_BF16_ARITHMETIC_POLICY
SUPPORTED_W4_ARITHMETIC_POLICIES = frozenset(
    {
        W4_PREDOT_BF16_ARITHMETIC_POLICY,
        W4_POSTSCALE_BF16_ARITHMETIC_POLICY,
    }
)
# Authoritative runner job-b5ea4ff60f1d swept these exact Qwen3-30B shapes. The tuple key is
# (logical expert groups, output features, contraction). Unseen shapes use the conservative
# measured N32 fallback below; this table changes launch geometry, never arithmetic.
W4_POSTSCALE_EXACT_LAUNCH_CONFIGS = {
    (8, 1536, 2048): (16, 64, 64, 4, 2),
    (48, 1536, 2048): (16, 64, 64, 4, 3),
    (32, 1536, 2048): (16, 32, 64, 4, 2),
    (8, 2048, 768): (16, 32, 64, 4, 2),
    (48, 2048, 768): (16, 32, 64, 4, 2),
    (32, 2048, 768): (16, 32, 64, 4, 2),
}
W4_POSTSCALE_FALLBACK_LAUNCH_CONFIG = (16, 32, 64, 4, 2)

# Backward-compatible names for the established public FP8 format surface.
STORE_SCHEMA = FP8_STORE_SCHEMA
STORE_CODEC = FP8_STORE_CODEC
STORE_FILE = FP8_STORE_FILE
DEFAULT_CACHE_MB = 7100.0
DEFAULT_MAX_ACTIVE_PAGES = 128
GLOBAL_LRU_CACHE_POLICY = "global-lru-v1"
LAYER_FREQUENCY_CACHE_POLICY = "layer-frequency-lru-v1"
SUPPORTED_CACHE_POLICIES = frozenset({GLOBAL_LRU_CACHE_POLICY, LAYER_FREQUENCY_CACHE_POLICY})
COMPACT_PAGE_BINDING_POLICY = "compact-copy-v1"
SLOT_INDIRECT_PAGE_BINDING_POLICY = "slot-indirect-v1"
DEFAULT_PAGE_BINDING_POLICY = SLOT_INDIRECT_PAGE_BINDING_POLICY
SUPPORTED_PAGE_BINDING_POLICIES = frozenset(
    {COMPACT_PAGE_BINDING_POLICY, SLOT_INDIRECT_PAGE_BINDING_POLICY}
)
CACHE_FILL_PREFILL_PAGE_POLICY = "cache-fill-v1"
TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY = "transient-frequency-v1"
SUPPORTED_PREFILL_PAGE_POLICIES = frozenset(
    {CACHE_FILL_PREFILL_PAGE_POLICY, TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY}
)
ATOMIC_ROUTE_REDUCTION_POLICY = "atomic-index-add-v1"
STABLE_ROUTE_REDUCTION_POLICY = "stable-route-rank-v1"
DEFAULT_ROUTE_REDUCTION_POLICY = STABLE_ROUTE_REDUCTION_POLICY
SUPPORTED_ROUTE_REDUCTION_POLICIES = frozenset(
    {ATOMIC_ROUTE_REDUCTION_POLICY, STABLE_ROUTE_REDUCTION_POLICY}
)
PAGE_TRACE_SCHEMA = "mrun-qwen3-moe-page-trace-v1"
PAGE_TRACE_HARD_MAX_EVENTS = 250_000
PAGE_TRACE_HARD_MAX_REQUESTS = 2_000_000
PAGE_TRACE_HARD_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
FP8_MAX = 448.0
PAGE_ALIGNMENT = 4096


def normalize_expert_codec(value: str) -> str:
    codec = str(value).strip().lower()
    if codec not in SUPPORTED_EXPERT_CODECS:
        raise ValueError(
            f"unsupported Qwen3 MoE expert codec {codec!r}; expected one of "
            f"{sorted(SUPPORTED_EXPERT_CODECS)}"
        )
    return codec


def normalize_w4_arithmetic_policy(value: str | None) -> str:
    policy = DEFAULT_W4_ARITHMETIC_POLICY if value is None else str(value).strip().lower()
    if policy not in SUPPORTED_W4_ARITHMETIC_POLICIES:
        raise ValueError(
            f"unsupported Qwen3 MoE W4 arithmetic policy {policy!r}; expected one of "
            f"{sorted(SUPPORTED_W4_ARITHMETIC_POLICIES)}"
        )
    return policy


def _w4_postscale_launch_config(
    groups: int,
    output_features: int,
    contraction: int,
) -> tuple[int, int, int, int, int]:
    return W4_POSTSCALE_EXACT_LAUNCH_CONFIGS.get(
        (int(groups), int(output_features), int(contraction)),
        W4_POSTSCALE_FALLBACK_LAUNCH_CONFIG,
    )


def _store_contract(expert_codec: str) -> tuple[str, str, str, str]:
    codec = normalize_expert_codec(expert_codec)
    if codec == EXPERT_CODEC_FP8:
        return FP8_STORE_SCHEMA, FP8_STORE_CODEC, FP8_STORE_FILE, "fp8-page"
    return W4_STORE_SCHEMA, W4_STORE_CODEC, W4_STORE_FILE, "int4-page"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def align_up(value: int, alignment: int = PAGE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1e9 if sys.platform == "darwin" else value / 1e6


@dataclass(frozen=True)
class ExpertPageLayout:
    hidden_size: int
    intermediate_size: int
    gate_up_codes_offset: int
    gate_up_codes_bytes: int
    gate_up_scales_offset: int
    gate_up_scales_bytes: int
    down_codes_offset: int
    down_codes_bytes: int
    down_scales_offset: int
    down_scales_bytes: int
    page_stride: int

    @classmethod
    def create(cls, hidden_size: int, intermediate_size: int) -> ExpertPageLayout:
        gate_up_rows = 2 * intermediate_size
        gate_up_codes_offset = 0
        gate_up_codes_bytes = gate_up_rows * hidden_size
        gate_up_scales_offset = gate_up_codes_offset + gate_up_codes_bytes
        gate_up_scales_bytes = gate_up_rows * 4
        down_codes_offset = gate_up_scales_offset + gate_up_scales_bytes
        down_codes_bytes = hidden_size * intermediate_size
        down_scales_offset = down_codes_offset + down_codes_bytes
        down_scales_bytes = hidden_size * 4
        page_stride = align_up(down_scales_offset + down_scales_bytes)
        return cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gate_up_codes_offset=gate_up_codes_offset,
            gate_up_codes_bytes=gate_up_codes_bytes,
            gate_up_scales_offset=gate_up_scales_offset,
            gate_up_scales_bytes=gate_up_scales_bytes,
            down_codes_offset=down_codes_offset,
            down_codes_bytes=down_codes_bytes,
            down_scales_offset=down_scales_offset,
            down_scales_bytes=down_scales_bytes,
            page_stride=page_stride,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExpertPageLayout:
        return cls(**{name: int(payload[name]) for name in cls.__dataclass_fields__})

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class Int4ExpertPageLayout:
    hidden_size: int
    intermediate_size: int
    group_size: int
    gate_up_codes_offset: int
    gate_up_codes_bytes: int
    gate_up_scales_offset: int
    gate_up_scales_bytes: int
    down_codes_offset: int
    down_codes_bytes: int
    down_scales_offset: int
    down_scales_bytes: int
    page_stride: int

    @classmethod
    def create(
        cls,
        hidden_size: int,
        intermediate_size: int,
        *,
        group_size: int = W4_GROUP_SIZE,
    ) -> Int4ExpertPageLayout:
        hidden_size = int(hidden_size)
        intermediate_size = int(intermediate_size)
        group_size = int(group_size)
        if hidden_size < 1 or intermediate_size < 1 or group_size < 1:
            raise ValueError("INT4 expert dimensions and group_size must be positive")
        if hidden_size % group_size or intermediate_size % group_size:
            raise ValueError(
                "INT4 expert contractions must be divisible by group_size: "
                f"hidden={hidden_size}, intermediate={intermediate_size}, "
                f"group_size={group_size}"
            )
        if hidden_size % 2 or intermediate_size % 2:
            raise ValueError("INT4 expert contractions must be even for nibble packing")

        gate_up_rows = 2 * intermediate_size
        gate_up_codes_offset = 0
        gate_up_codes_bytes = gate_up_rows * (hidden_size // 2)
        gate_up_scales_offset = gate_up_codes_offset + gate_up_codes_bytes
        gate_up_scales_bytes = gate_up_rows * (hidden_size // group_size) * 4
        down_codes_offset = gate_up_scales_offset + gate_up_scales_bytes
        down_codes_bytes = hidden_size * (intermediate_size // 2)
        down_scales_offset = down_codes_offset + down_codes_bytes
        down_scales_bytes = hidden_size * (intermediate_size // group_size) * 4
        page_stride = align_up(down_scales_offset + down_scales_bytes)
        return cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            group_size=group_size,
            gate_up_codes_offset=gate_up_codes_offset,
            gate_up_codes_bytes=gate_up_codes_bytes,
            gate_up_scales_offset=gate_up_scales_offset,
            gate_up_scales_bytes=gate_up_scales_bytes,
            down_codes_offset=down_codes_offset,
            down_codes_bytes=down_codes_bytes,
            down_scales_offset=down_scales_offset,
            down_scales_bytes=down_scales_bytes,
            page_stride=page_stride,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Int4ExpertPageLayout:
        return cls(**{name: int(payload[name]) for name in cls.__dataclass_fields__})

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


def quantize_fp8_weight(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = source.detach().float().contiguous()
    scales = values.abs().amax(dim=1).div(FP8_MAX).clamp_min(1e-12)
    codes = (values / scales[:, None]).clamp(-FP8_MAX, FP8_MAX)
    return codes.to(torch.float8_e4m3fn).contiguous(), scales.contiguous()


def quantize_int4_weight(
    source: torch.Tensor,
    *,
    group_size: int = W4_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row/per-K-group INT4 using the production QStore nibble ABI.

    Signed values are restricted to ``[-7, 7]`` and stored as offset-eight nibbles. The low
    nibble is the even K position and the high nibble is the odd K position. Scales remain
    FP32, matching :mod:`mrun.engine.kernels.qstore_int4` exactly.
    """

    if source.ndim != 2:
        raise ValueError(f"INT4 quantization expects a matrix, got {tuple(source.shape)}")
    rows, contraction = map(int, source.shape)
    group_size = int(group_size)
    if rows < 1 or contraction < 1 or group_size < 1:
        raise ValueError("INT4 matrix dimensions and group_size must be positive")
    if contraction % group_size:
        raise ValueError(
            f"INT4 contraction {contraction} is not divisible by group_size {group_size}"
        )
    if contraction % 2:
        raise ValueError("INT4 contraction must be even for nibble packing")

    values = source.detach().to(device="cpu", dtype=torch.float32).contiguous()
    grouped = values.reshape(rows, contraction // group_size, group_size)
    scales = grouped.abs().amax(dim=-1).div(7.0)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    signed = torch.round(grouped / scales.unsqueeze(-1)).clamp_(-7, 7).to(torch.int16)
    codes = (signed + 8).to(torch.uint8).reshape(rows, contraction)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), scales.to(torch.float32).contiguous()


def dequantize_int4_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = W4_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialized reference for tests and qualification; never used by the hot path."""

    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("packed INT4 weights must be a rank-2 uint8 tensor")
    if scales.ndim != 2 or not scales.dtype.is_floating_point:
        raise ValueError("INT4 scales must be a rank-2 floating-point tensor")
    if scales.device != packed.device:
        raise ValueError("packed INT4 weights and scales must share one device")
    rows, packed_k = map(int, packed.shape)
    contraction = packed_k * 2
    group_size = int(group_size)
    if group_size < 1 or contraction % group_size:
        raise ValueError("packed INT4 contraction must be divisible by group_size")
    expected_scales = (rows, contraction // group_size)
    if tuple(scales.shape) != expected_scales:
        raise ValueError(
            f"INT4 scales have shape {tuple(scales.shape)}, expected {expected_scales}"
        )
    codes = torch.empty(
        (rows, contraction),
        dtype=torch.uint8,
        device=packed.device,
    )
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    signed = codes.to(torch.int16) - 8
    expanded_scales = scales.float().repeat_interleave(group_size, dim=1)
    return (signed.float() * expanded_scales).to(dtype)


def _single_file_weight_map(model_dir: Path) -> dict[str, str]:
    """Weight map for a checkpoint that ships NO ``model.safetensors.index.json``.

    Small checkpoints (Qwen2.5-0.5B, gpt2, pythia-160m) are one unsharded ``model.safetensors``
    and carry no index at all. Reading a named tensor out of them is the same range-read as the
    sharded case, so synthesizing the map here means one reader serves every checkpoint shape
    instead of callers branching on whether an index file happens to exist.
    """
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors and no index under {model_dir}")
    weight_map: dict[str, str] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 — safetensors handle, not a dict
                weight_map[key] = shard.name
    return weight_map


class TensorReader:
    """Lazy, name-addressed safetensors reader over ONE checkpoint directory.

    Two capabilities beyond a bare ``safe_open``, both load-bearing for reading a handful of
    named tensors out of a checkpoint too large to instantiate:

    * INDEX-OPTIONAL — sharded (``model.safetensors.index.json``) and single-file checkpoints
      are both addressed by tensor name (see :func:`_single_file_weight_map`).
    * FP8-TRANSPARENT — when a key has a companion ``<key>_scale_inv``/``<key>_scale`` tensor,
      :meth:`get` returns the DEQUANTIZED weight rather than the raw e4m3 codes. Without this a
      caller reading an FP8 checkpoint silently gets code bytes reinterpreted as numbers.
    """

    def __init__(self, model_dir: Path, *, dequant_dtype: torch.dtype = torch.bfloat16):
        self.model_dir = model_dir
        self.dequant_dtype = dequant_dtype
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map: dict[str, str] = dict(index["weight_map"])
        else:
            self.weight_map = _single_file_weight_map(model_dir)
        self.handles: dict[str, Any] = {}
        # The checkpoint's own declared FP8 block geometry, when it has one. Shapes alone cannot
        # always recover it (see _infer_block), so read it rather than guess.
        config_path = model_dir / "config.json"
        self.block_size: Sequence[int] | None = (
            declared_block_size(json.loads(config_path.read_text(encoding="utf-8")))
            if config_path.is_file()
            else None
        )

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def scale_key(self, key: str) -> str | None:
        """The companion FP8 scale tensor's name, or ``None`` when ``key`` is stored unquantized."""
        return scale_key_for(self.weight_map, key)

    def raw(self, key: str) -> torch.Tensor:
        """The stored tensor exactly as written (FP8 codes stay codes). Prefer :meth:`get`."""
        shard = self.weight_map[key]
        handle = self.handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
            self.handles[shard] = handle
        return handle.get_tensor(key)

    def get(self, key: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        scale_key = self.scale_key(key)
        if scale_key is None:
            tensor = self.raw(key)
            return tensor if dtype is None else tensor.to(dtype)
        return dequantize_block_fp8(
            self.raw(key),
            self.raw(scale_key),
            dtype=dtype or self.dequant_dtype,
            block_size=self.block_size,
        )

    def release(self) -> None:
        self.handles.clear()


def _source_records(model_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors weight map: {index_path}")
    return index, [
        model_dir / name for name in sorted({str(value) for value in weight_map.values()})
    ]


def _store_builder_provenance(expert_codec: str) -> dict[str, Any]:
    normalized = normalize_expert_codec(expert_codec)
    schema, codec, _data_file, _page_kind = _store_contract(normalized)
    if normalized == EXPERT_CODEC_FP8:
        name = "mrun.engine.qwen3_moe_cuda.build_fp8_expert_store"
        quantization = {
            "codec": codec,
            "weight_dtype": "float8_e4m3fn",
            "scale_dtype": "float32",
            "granularity": "per-output-row",
            "page_alignment": PAGE_ALIGNMENT,
        }
    else:
        name = "mrun.engine.qwen3_moe_cuda.build_int4_expert_store"
        quantization = {
            "codec": codec,
            "weight_dtype": "symmetric-int4-offset8",
            "scale_dtype": "float32",
            "granularity": "per-output-row-per-k-group",
            "group_size": W4_GROUP_SIZE,
            "signed_levels": [-7, 7],
            "low_nibble": "even-k",
            "page_alignment": PAGE_ALIGNMENT,
        }
    return build_builder_provenance(
        [Path(__file__)],
        name=name,
        schema_version=schema,
        quantization=quantization,
    )


def _verify_existing_store(
    store_dir: Path,
    *,
    source: dict[str, Any],
    builder: dict[str, Any],
    layout: ExpertPageLayout | Int4ExpertPageLayout,
    expert_codec: str,
) -> dict[str, Any]:
    schema, codec, data_file, _page_kind = _store_contract(expert_codec)
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"refusing incomplete expert store: {store_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": schema,
        "codec": codec,
        "data_file": data_file,
        "layout": layout.as_dict(),
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"existing expert store identity mismatch: {actual} != {expected}")
    verify_source_provenance(manifest.get("source"), source)
    actual_builder = manifest.get("builder")
    try:
        verify_builder_provenance(actual_builder, builder)
    except RuntimeError:
        _verify_compatible_promoted_builder(actual_builder, builder)
    verify_derived_provenance(
        store_dir,
        manifest.get("derived"),
        expected_filenames=(data_file,),
    )
    expected_bytes = int(manifest["layers"]) * int(manifest["num_experts"]) * layout.page_stride
    if (store_dir / data_file).stat().st_size != expected_bytes:
        raise RuntimeError("expert store byte length does not match its fixed page geometry")
    return manifest


def _verify_compatible_promoted_builder(
    actual: object,
    expected: dict[str, Any],
) -> None:
    """Accept the measured experiment builder when its format contract is identical."""

    if not isinstance(actual, dict):
        raise RuntimeError("existing store has no publication-grade builder provenance")
    allowed_names = (
        {"mrun.engine.qwen3_moe_cuda.build_int4_expert_store"}
        if expected.get("build_schema_version") == W4_STORE_SCHEMA
        else {
            "generation_atlas.qwen3_moe_decode.build_fp8_expert_store",
            "mrun.engine.qwen3_moe_cuda.build_fp8_expert_store",
        }
    )
    if actual.get("name") not in allowed_names:
        raise RuntimeError("existing store was produced by an unsupported builder")
    required_equal = ("schema_version", "build_schema_version", "quantization")
    if any(actual.get(key) != expected.get(key) for key in required_equal):
        raise RuntimeError("existing store builder format contract mismatch")
    source_files = actual.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise RuntimeError("existing store builder source bundle is missing")
    source_content = {
        "schema_version": actual["schema_version"],
        "files": source_files,
    }
    if actual.get("source_bundle_sha256") != canonical_sha256(source_content):
        raise RuntimeError("existing store builder source digest is invalid")


def _build_expert_store(
    model_dir: Path,
    store_dir: Path,
    *,
    model_name: str,
    hf_id: str,
    expert_codec: str,
    revision: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_expert_codec(expert_codec)
    schema, codec, data_file, page_kind = _store_contract(normalized)
    model_dir = model_dir.expanduser().resolve()
    store_dir = store_dir.expanduser()
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("model_type") != "qwen3_moe":
        raise ValueError(f"expected qwen3_moe, found {cfg.get('model_type')!r}")
    layers = int(cfg["num_hidden_layers"])
    experts = int(cfg["num_experts"])
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["moe_intermediate_size"])
    layout: ExpertPageLayout | Int4ExpertPageLayout
    if normalized == EXPERT_CODEC_FP8:
        layout = ExpertPageLayout.create(hidden, intermediate)
    else:
        layout = Int4ExpertPageLayout.create(hidden, intermediate)
    _, shards = _source_records(model_dir)
    source = build_source_provenance(
        model_dir,
        shards,
        model_name=model_name,
        hf_id=hf_id,
        revision=revision,
    )
    builder = _store_builder_provenance(normalized)
    if store_dir.exists():
        return _verify_existing_store(
            store_dir,
            source=source,
            builder=builder,
            layout=layout,
            expert_codec=normalized,
        )

    store_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{store_dir.name}.building-", dir=store_dir.parent))
    reader = TensorReader(model_dir)
    data_path = temporary / data_file
    digest = hashlib.sha256()
    started = time.perf_counter()
    padding_bytes = layout.page_stride - (layout.down_scales_offset + layout.down_scales_bytes)
    padding = b"\0" * padding_bytes
    blocks: dict[str, dict[str, Any]] = {}
    try:
        with data_path.open("wb") as output:
            for layer in range(layers):
                for expert in range(experts):
                    page_offset = output.tell()
                    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
                    gate = reader.get(f"{prefix}.gate_proj.weight")
                    up = reader.get(f"{prefix}.up_proj.weight")
                    down = reader.get(f"{prefix}.down_proj.weight")
                    if normalized == EXPERT_CODEC_FP8:
                        gate_up_codes, gate_up_scales = quantize_fp8_weight(
                            torch.cat((gate, up), dim=0)
                        )
                        down_codes, down_scales = quantize_fp8_weight(down)
                    else:
                        gate_up_codes, gate_up_scales = quantize_int4_weight(
                            torch.cat((gate, up), dim=0)
                        )
                        down_codes, down_scales = quantize_int4_weight(down)
                    blobs = (
                        gate_up_codes.view(torch.uint8).numpy().tobytes(order="C"),
                        gate_up_scales.numpy().tobytes(order="C"),
                        down_codes.view(torch.uint8).numpy().tobytes(order="C"),
                        down_scales.numpy().tobytes(order="C"),
                        padding,
                    )
                    for blob in blobs:
                        output.write(blob)
                        digest.update(blob)
                    if output.tell() - page_offset != layout.page_stride:
                        raise RuntimeError("expert page writer violated fixed stride")
                    blocks[f"L{layer}.E{expert}"] = {
                        "kind": page_kind,
                        "offset": page_offset,
                        "length": layout.page_stride,
                    }
                    del (
                        gate,
                        up,
                        down,
                        gate_up_codes,
                        gate_up_scales,
                        down_codes,
                        down_scales,
                    )
                reader.release()
                print(
                    f"built Qwen3 {normalized.upper()} expert layer {layer + 1}/{layers}: "
                    f"{output.tell() / 1e9:.3f} GB RSS={rss_gb():.2f} GB",
                    flush=True,
                )
    except Exception:
        reader.release()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        reader.release()

    derived = build_derived_provenance(
        temporary,
        (data_file,),
        known_sha256={data_file: digest.hexdigest()},
    )
    manifest = {
        "schema_version": schema,
        "codec": codec,
        "dtype": "fp8" if normalized == EXPERT_CODEC_FP8 else "int4",
        "arch": "qwen3_moe",
        "model_name": model_name,
        "hf_id": hf_id,
        "created_at": utc_now(),
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "source": source,
        "builder": builder,
        "derived": derived,
        "data_file": data_file,
        "data_bytes": data_path.stat().st_size,
        "data_sha256": digest.hexdigest(),
        "layers": layers,
        "num_experts": experts,
        "top_k": int(cfg["num_experts_per_tok"]),
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "layout": layout.as_dict(),
        "blocks": blocks,
        "build_s": time.perf_counter() - started,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.rename(store_dir)
    return manifest


def build_fp8_expert_store(
    model_dir: Path,
    store_dir: Path,
    *,
    model_name: str,
    hf_id: str,
    revision: str | None = None,
) -> dict[str, Any]:
    return _build_expert_store(
        model_dir,
        store_dir,
        model_name=model_name,
        hf_id=hf_id,
        expert_codec=EXPERT_CODEC_FP8,
        revision=revision,
    )


def build_int4_expert_store(
    model_dir: Path,
    store_dir: Path,
    *,
    model_name: str,
    hf_id: str,
    revision: str | None = None,
) -> dict[str, Any]:
    return _build_expert_store(
        model_dir,
        store_dir,
        model_name=model_name,
        hf_id=hf_id,
        expert_codec=EXPERT_CODEC_W4,
        revision=revision,
    )


@dataclass(frozen=True)
class DeviceExpertLayer:
    gate_up: torch.Tensor
    gate_up_scales: torch.Tensor
    down: torch.Tensor
    down_scales: torch.Tensor


@dataclass(frozen=True)
class DeviceInt4ExpertLayer:
    gate_up: torch.Tensor
    gate_up_scales: torch.Tensor
    down: torch.Tensor
    down_scales: torch.Tensor


class PackedFP8ExpertStore:
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != STORE_SCHEMA:
            raise ValueError("unsupported Qwen3 expert store schema")
        if self.manifest.get("codec") != STORE_CODEC:
            raise ValueError("unsupported Qwen3 FP8 expert store codec")
        self.layout = ExpertPageLayout.from_dict(self.manifest["layout"])
        self.layers = int(self.manifest["layers"])
        self.experts = int(self.manifest["num_experts"])
        self.data_path = store_dir / self.manifest["data_file"]
        self.release_after_gather = False

    def _ensure_pages(self) -> np.memmap:
        """Persistent flat [layers*experts, stride] mapping. The prior per-gather
        memmap open + MADV_RANDOM + close forced every miss through fresh 4 KiB page
        faults with readahead disabled — measured ~85 MB/s effective cold fill against
        a 26.81 GB/s link. One long-lived mapping keeps the OS page cache and default
        readahead working."""
        pages = getattr(self, "_pages", None)
        if pages is None:
            pages = np.memmap(
                self.data_path,
                dtype=np.uint8,
                mode="r",
                shape=(self.layers * self.experts, self.layout.page_stride),
            )
            self._pages = pages
        return pages

    def global_page(self, layer: int, expert: int) -> int:
        return layer * self.experts + expert

    def gather(
        self,
        layer: int,
        expert_ids: list[int],
        target: torch.Tensor,
    ) -> None:
        if not expert_ids:
            return
        if target.device.type != "cpu" or target.dtype != torch.uint8:
            raise ValueError("expert page staging target must be a CPU uint8 tensor")
        view = target[: len(expert_ids)].numpy()
        pages = self._ensure_pages()
        rows = layer * self.experts + np.asarray(expert_ids, dtype=np.int64)
        np.take(pages, rows, axis=0, out=view)
        if self.release_after_gather:
            self.release()

    def view_pages(self, raw: torch.Tensor) -> DeviceExpertLayer:
        return expert_page_views(raw, self.layout)

    def release(self) -> None:
        pages = getattr(self, "_pages", None)
        if pages is not None:
            mapping = getattr(pages, "_mmap", None)
            self._pages = None
            if mapping is not None:
                mapping.close()


class PackedInt4ExpertStore(PackedFP8ExpertStore):
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != W4_STORE_SCHEMA:
            raise ValueError("unsupported Qwen3 INT4 expert store schema")
        if self.manifest.get("codec") != W4_STORE_CODEC:
            raise ValueError("unsupported Qwen3 INT4 expert store codec")
        if self.manifest.get("dtype") != "int4":
            raise ValueError("Qwen3 INT4 expert store must declare dtype='int4'")
        self.layout = Int4ExpertPageLayout.from_dict(self.manifest["layout"])
        expected_layout = Int4ExpertPageLayout.create(
            self.layout.hidden_size,
            self.layout.intermediate_size,
        )
        if self.layout != expected_layout:
            raise ValueError(
                "Qwen3 INT4 expert-store layout does not match its fixed codec geometry"
            )
        self.layers = int(self.manifest["layers"])
        self.experts = int(self.manifest["num_experts"])
        if self.layers < 1 or self.experts < 1:
            raise ValueError("Qwen3 INT4 expert store must contain layers and experts")
        if self.manifest.get("data_file") != W4_STORE_FILE:
            raise ValueError("Qwen3 INT4 expert-store data filename violates its codec ABI")
        self.data_path = store_dir / W4_STORE_FILE
        expected_bytes = self.layers * self.experts * self.layout.page_stride
        if not self.data_path.is_file() or self.data_path.stat().st_size != expected_bytes:
            raise ValueError("Qwen3 INT4 expert-store byte length violates fixed page geometry")
        self.release_after_gather = False

    def view_pages(self, raw: torch.Tensor) -> DeviceInt4ExpertLayer:
        return int4_expert_page_views(raw, self.layout)


class HostPageTier:
    """Budgeted host-RAM copy of expert pages between the CUDA LRU and the NVMe store.

    ``warm()`` fills the tier with one sequential pass over the store file (OS
    readahead engaged), so first-touch misses become host memcpys instead of random
    4 KiB mmap faults, and process restarts never pay the cold-disk cliff again.
    Pages are admitted on first miss when not pre-warmed. Fill-once, no eviction:
    capacity is the byte budget, and overflow pages simply stay disk-backed."""

    def __init__(
        self,
        store: PackedFP8ExpertStore | PackedInt4ExpertStore,
        *,
        host_cache_mb: float,
        try_pin: bool = True,
    ):
        self.store = store
        stride = store.layout.page_stride
        total_pages = store.layers * store.experts
        capacity = min(total_pages, int(float(host_cache_mb) * 1e6) // stride)
        if capacity < 1:
            raise ValueError("host_cache_mb smaller than one expert page")
        self.capacity = capacity
        self.pinned = False
        buf = None
        if try_pin:
            try:
                buf = torch.empty((capacity, stride), dtype=torch.uint8, pin_memory=True)
                self.pinned = True
            except RuntimeError:
                buf = None
        if buf is None:
            buf = torch.empty((capacity, stride), dtype=torch.uint8)
        self.buf = buf
        self._buf_np = buf.numpy()
        self._rows = np.full(total_pages, -1, dtype=np.int64)
        self._next_row = 0
        self.stats = {
            "capacity_pages": capacity,
            "pinned": self.pinned,
            "warmed_pages": 0,
            "warm_seconds": 0.0,
            "served_pages": 0,
            "admitted_pages": 0,
            "disk_pages": 0,
        }

    def warm(self) -> dict[str, Any]:
        """Sequential bulk fill of pages [0, capacity) via pread — NOT mmap.

        Reading through the persistent memmap doubled RSS (touched page-cache pages
        count toward the process: 22 GB tier + 22 GB mapped = 44 GB, killed_ram twice,
        jobs a34a4678e757 / 76c4d7c17e47). Unbuffered readinto lands bytes directly in
        the tier buffer; RSS = tier only."""
        started = time.perf_counter()
        stride = self.store.layout.page_stride
        total = self.capacity * stride
        flat = self._buf_np.reshape(-1)
        mv = memoryview(flat)  # uint8 contiguous
        chunk = 256 * 1024 * 1024
        with open(self.store.data_path, "rb", buffering=0) as fh:
            off = self._next_row * stride
            fh.seek(off)
            while off < total:
                n = fh.readinto(mv[off : off + min(chunk, total - off)])
                if not n:
                    break
                off += n
        filled = off // stride
        self._rows[self._next_row : filled] = np.arange(self._next_row, filled, dtype=np.int64)
        self._next_row = filled
        self.stats["warm_seconds"] = time.perf_counter() - started
        self.stats["warmed_pages"] = filled
        return dict(self.stats)

    def gather(
        self,
        layer: int,
        expert_ids: list[int],
        target: torch.Tensor,
    ) -> None:
        """Fill ``target[:len(expert_ids)]`` from the tier where present, disk
        otherwise (admitting disk pages while capacity remains)."""
        view = target[: len(expert_ids)].numpy()
        pages = None
        for i, expert in enumerate(expert_ids):
            gid = self.store.global_page(layer, expert)
            row = self._rows[gid]
            if row >= 0:
                np.copyto(view[i], self._buf_np[row])
                self.stats["served_pages"] += 1
                continue
            if pages is None:
                pages = self.store._ensure_pages()
            np.copyto(view[i], pages[gid])
            self.stats["disk_pages"] += 1
            if self._next_row < self.capacity:
                new_row = self._next_row
                self._next_row = new_row + 1
                np.copyto(self._buf_np[new_row], view[i])
                self._rows[gid] = new_row
                self.stats["admitted_pages"] += 1


@dataclass
class PageCacheStats:
    page_requests: int = 0
    page_hits: int = 0
    page_misses: int = 0
    host_to_device_bytes: int = 0
    layer_acquires: int = 0
    active_experts_sum: int = 0
    evictions: int = 0
    same_layer_evictions: int = 0
    over_quota_evictions: int = 0
    fallback_evictions: int = 0
    transient_prefill_acquires: int = 0
    transient_prefill_pages: int = 0
    transient_prefill_admitted_pages: int = 0
    transient_prefill_preserved_pages: int = 0
    transient_prefill_h2d_bytes: int = 0
    transient_prefill_resident_materialization_d2d_bytes: int = 0
    transient_prefill_admission_d2d_bytes: int = 0
    transient_prefill_replaced_pages: int = 0
    transient_prefill_skipped_protected_pages: int = 0
    decode_protected_promotions: int = 0
    decode_protected_demotions: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_requests": self.page_requests,
            "page_hits": self.page_hits,
            "page_misses": self.page_misses,
            "hit_rate": self.page_hits / max(1, self.page_requests),
            "host_to_device_bytes": self.host_to_device_bytes,
            "host_to_device_gb": self.host_to_device_bytes / 1e9,
            "layer_acquires": self.layer_acquires,
            "active_experts_sum": self.active_experts_sum,
            "mean_active_experts": self.active_experts_sum / max(1, self.layer_acquires),
            "evictions": self.evictions,
            "same_layer_evictions": self.same_layer_evictions,
            "over_quota_evictions": self.over_quota_evictions,
            "fallback_evictions": self.fallback_evictions,
            "transient_prefill_acquires": self.transient_prefill_acquires,
            "transient_prefill_pages": self.transient_prefill_pages,
            "transient_prefill_admitted_pages": self.transient_prefill_admitted_pages,
            "transient_prefill_preserved_pages": self.transient_prefill_preserved_pages,
            "transient_prefill_h2d_bytes": self.transient_prefill_h2d_bytes,
            "transient_prefill_resident_materialization_d2d_bytes": (
                self.transient_prefill_resident_materialization_d2d_bytes
            ),
            "transient_prefill_admission_d2d_bytes": (self.transient_prefill_admission_d2d_bytes),
            "transient_prefill_replaced_pages": self.transient_prefill_replaced_pages,
            "transient_prefill_skipped_protected_pages": (
                self.transient_prefill_skipped_protected_pages
            ),
            "decode_protected_promotions": self.decode_protected_promotions,
            "decode_protected_demotions": self.decode_protected_demotions,
        }


class Qwen3MoePageTraceLimitError(RuntimeError):
    """Raised before cache mutation when compact live page-trace custody would overflow."""


def _page_trace_limit(value: int, *, name: str, hard_maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if value > hard_maximum:
        raise ValueError(f"{name} cannot exceed hard maximum {hard_maximum:,}")
    return int(value)


@dataclass(frozen=True)
class _CompactPageTraceEvent:
    phase: str
    step: int
    layer: int
    pages: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "step": self.step,
            "layer": self.layer,
            "pages": [[expert, count] for expert, count in self.pages],
        }


class _CompactPageTrace:
    """Bounded metadata-only trace matching ``mrun-qwen3-moe-page-trace-v1`` exactly."""

    def __init__(
        self,
        *,
        layer_count: int,
        page_bytes: int,
        max_events: int,
        max_requests: int,
        max_manifest_bytes: int,
    ) -> None:
        self.layer_count = int(layer_count)
        self.page_bytes = int(page_bytes)
        self.max_events = _page_trace_limit(
            max_events,
            name="page_trace_max_events",
            hard_maximum=PAGE_TRACE_HARD_MAX_EVENTS,
        )
        self.max_requests = _page_trace_limit(
            max_requests,
            name="page_trace_max_requests",
            hard_maximum=PAGE_TRACE_HARD_MAX_REQUESTS,
        )
        self.max_manifest_bytes = _page_trace_limit(
            max_manifest_bytes,
            name="page_trace_max_manifest_bytes",
            hard_maximum=PAGE_TRACE_HARD_MAX_MANIFEST_BYTES,
        )
        self._events: list[_CompactPageTraceEvent] = []
        self._event_json_bytes = 0
        self._requests = 0
        self._steps: dict[tuple[str, int], int] = {}
        empty_with_hash = {
            **self._core_payload(events=[]),
            "trace_sha256": "0" * 64,
        }
        self._empty_manifest_bytes = len(
            json.dumps(
                empty_with_hash,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if self._empty_manifest_bytes > self.max_manifest_bytes:
            raise ValueError(
                "page_trace_max_manifest_bytes cannot hold the empty compact trace manifest"
            )

    def _core_payload(self, *, events: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema": PAGE_TRACE_SCHEMA,
            "layer_count": self.layer_count,
            "page_bytes": self.page_bytes,
            "source": {},
            "events": events,
        }

    @property
    def manifest_bytes(self) -> int:
        commas = max(0, len(self._events) - 1)
        return self._empty_manifest_bytes + self._event_json_bytes + commas

    @property
    def trace_sha256(self) -> str | None:
        payload = self.export()
        return None if payload is None else str(payload["trace_sha256"])

    def record(self, *, phase: str, layer: int, frequencies: Counter[int]) -> None:
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"unsupported Qwen3 MoE page-trace phase {phase!r}")
        if layer < 0 or layer >= self.layer_count:
            raise ValueError(f"page-trace layer {layer} is outside {self.layer_count} layers")
        pages = tuple(
            (int(expert), int(frequencies[expert])) for expert in sorted(frequencies)
        )
        if not pages or any(expert < 0 or count <= 0 for expert, count in pages):
            raise ValueError("page trace requires unique non-negative pages with positive counts")
        step_key = (phase, int(layer))
        event = _CompactPageTraceEvent(
            phase=phase,
            step=self._steps.get(step_key, 0),
            layer=int(layer),
            pages=pages,
        )
        event_bytes = len(
            json.dumps(
                event.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        candidate_events = len(self._events) + 1
        candidate_requests = self._requests + len(pages)
        candidate_manifest_bytes = self.manifest_bytes + event_bytes + bool(self._events)
        if candidate_events > self.max_events:
            raise Qwen3MoePageTraceLimitError(
                f"page trace exceeded configured event cap {self.max_events:,}"
            )
        if candidate_requests > self.max_requests:
            raise Qwen3MoePageTraceLimitError(
                f"page trace exceeded configured request cap {self.max_requests:,}"
            )
        if candidate_manifest_bytes > self.max_manifest_bytes:
            raise Qwen3MoePageTraceLimitError(
                "page trace exceeded configured manifest-byte cap "
                f"{self.max_manifest_bytes:,}"
            )
        self._events.append(event)
        self._event_json_bytes += event_bytes
        self._requests = candidate_requests
        self._steps[step_key] = event.step + 1

    def export(self) -> dict[str, Any] | None:
        if not self._events:
            return None
        events = [event.as_dict() for event in self._events]
        core = self._core_payload(events=events)
        payload = {**core, "trace_sha256": canonical_sha256(core)}
        encoded_bytes = len(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        if encoded_bytes != self.manifest_bytes:
            raise RuntimeError("page trace manifest byte accounting drifted")
        return payload

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema": PAGE_TRACE_SCHEMA,
            "events": len(self._events),
            "page_requests": self._requests,
            "manifest_bytes": self.manifest_bytes,
            "trace_sha256": self.trace_sha256,
            "caps": {
                "events": self.max_events,
                "page_requests": self.max_requests,
                "manifest_bytes": self.max_manifest_bytes,
            },
            "retention": "phase/step/layer plus unique expert IDs and route counts only",
        }

    def clear(self) -> None:
        self._events.clear()
        self._event_json_bytes = 0
        self._requests = 0
        self._steps.clear()

    def drain(self) -> dict[str, Any] | None:
        payload = self.export()
        self.clear()
        return payload


class ExpertPageCache:
    def __init__(
        self,
        store: PackedFP8ExpertStore | PackedInt4ExpertStore,
        *,
        device: str,
        cache_mb: float,
        max_active_pages: int,
        host_tier: HostPageTier | None = None,
        route_prefetch: bool | None = None,
        cache_policy: str | None = None,
        page_binding_policy: str | None = None,
        prefill_page_policy: str | None = None,
        page_trace: bool | None = None,
        page_trace_max_events: int = PAGE_TRACE_HARD_MAX_EVENTS,
        page_trace_max_requests: int = PAGE_TRACE_HARD_MAX_REQUESTS,
        page_trace_max_manifest_bytes: int = PAGE_TRACE_HARD_MAX_MANIFEST_BYTES,
    ):
        self.store = store
        self.host_tier = host_tier
        # Route-locality prefetch: temporal locality is measured (+0.13-0.19 hit rate
        # over a time-shuffle null on OLMoE), so the PREVIOUS decode step's routes for
        # layer L+1 predict the next acquire well. After serving layer L we stage those
        # pages on a side stream; a wrong guess wastes idle bus, never correctness.
        # Effective mainly in the cache-starved regime and with the host tier (stage
        # source is then a memcpy, not a disk fault). Opt-in.
        if route_prefetch is None:
            route_prefetch = os.environ.get("MRUN_QWEN3_MOE_ROUTE_PREFETCH", "").strip() not in (
                "",
                "0",
            )
        self.route_prefetch = bool(route_prefetch)
        self._prev_routes: dict[int, list[int]] = {}
        self._prefetch_stream = (
            torch.cuda.Stream(device=device)
            if self.route_prefetch and torch.cuda.is_available()
            else None
        )
        self.prefetch_stats = {"prefetched_pages": 0, "prefetch_rounds": 0}
        self.device = torch.device(device)
        self.layout = store.layout
        if page_trace is None:
            trace_env = os.environ.get("MRUN_QWEN3_MOE_PAGE_TRACE", "").strip().lower()
            page_trace = trace_env not in {"", "0", "false", "no", "off"}
        self._page_trace = (
            _CompactPageTrace(
                layer_count=int(store.layers),
                page_bytes=int(self.layout.page_stride),
                max_events=page_trace_max_events,
                max_requests=page_trace_max_requests,
                max_manifest_bytes=page_trace_max_manifest_bytes,
            )
            if page_trace
            else None
        )
        configured_policy = (
            cache_policy
            if cache_policy is not None
            else os.environ.get("MRUN_QWEN3_MOE_CACHE_POLICY", GLOBAL_LRU_CACHE_POLICY)
        )
        self.cache_policy = str(configured_policy).strip().lower()
        if self.cache_policy not in SUPPORTED_CACHE_POLICIES:
            raise ValueError(
                "unsupported Qwen3 MoE cache policy "
                f"{self.cache_policy!r}; expected one of {sorted(SUPPORTED_CACHE_POLICIES)}"
            )
        configured_binding = (
            page_binding_policy
            if page_binding_policy is not None
            else os.environ.get(
                "MRUN_QWEN3_MOE_PAGE_BINDING_POLICY",
                DEFAULT_PAGE_BINDING_POLICY,
            )
        )
        self.page_binding_policy = str(configured_binding).strip().lower()
        if self.page_binding_policy not in SUPPORTED_PAGE_BINDING_POLICIES:
            raise ValueError(
                "unsupported Qwen3 MoE page binding policy "
                f"{self.page_binding_policy!r}; expected one of "
                f"{sorted(SUPPORTED_PAGE_BINDING_POLICIES)}"
            )
        configured_prefill = (
            prefill_page_policy
            if prefill_page_policy is not None
            else os.environ.get(
                "MRUN_QWEN3_MOE_PREFILL_PAGE_POLICY",
                CACHE_FILL_PREFILL_PAGE_POLICY,
            )
        )
        self.prefill_page_policy = str(configured_prefill).strip().lower()
        if self.prefill_page_policy not in SUPPORTED_PREFILL_PAGE_POLICIES:
            raise ValueError(
                "unsupported Qwen3 MoE prefill page policy "
                f"{self.prefill_page_policy!r}; expected one of "
                f"{sorted(SUPPORTED_PREFILL_PAGE_POLICIES)}"
            )
        if (
            self.prefill_page_policy == TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY
            and self.cache_policy != LAYER_FREQUENCY_CACHE_POLICY
        ):
            raise ValueError(
                "transient-frequency-v1 prefill requires layer-frequency-lru-v1 "
                "so every layer has a bounded admission quota"
            )
        if (
            self.prefill_page_policy == TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY
            and self.route_prefetch
        ):
            raise ValueError(
                "transient-frequency-v1 prefill is incompatible with route-history "
                "prefetch because speculative pages can evict decode-protected residency"
            )
        requested_capacity = int(float(cache_mb) * 1e6) // self.layout.page_stride
        self.capacity = min(
            store.layers * store.experts,
            max(int(max_active_pages), requested_capacity),
        )
        self.max_active_pages = int(max_active_pages)
        self.cache = torch.empty(
            (self.capacity, self.layout.page_stride),
            dtype=torch.uint8,
            device=self.device,
        )
        self.stage = torch.empty(
            (self.max_active_pages, self.layout.page_stride),
            dtype=torch.uint8,
            pin_memory=self.device.type == "cuda",
        )
        self.miss_device = torch.empty(
            (self.max_active_pages, self.layout.page_stride),
            dtype=torch.uint8,
            device=self.device,
        )
        # The promoted route addresses physical cache slots directly. Keep the established
        # compact gather as an explicit fallback, but do not reserve its potentially hundreds
        # of MiB unless that fallback is selected.
        self.compact = (
            torch.empty_like(self.miss_device)
            if self.page_binding_policy == COMPACT_PAGE_BINDING_POLICY
            else None
        )
        self.prefetch_stage = torch.empty_like(self.stage) if self.route_prefetch else None
        self.prefetch_device = torch.empty_like(self.miss_device) if self.route_prefetch else None
        self.entries: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.decode_protected: set[tuple[int, int]] = set()
        self.free_slots = list(range(self.capacity - 1, -1, -1))
        layer_count = int(store.layers)
        quota, remainder = divmod(self.capacity, layer_count)
        self.layer_quotas = tuple(
            quota + (1 if layer < remainder else 0) for layer in range(layer_count)
        )
        self.layer_entry_counts = [0] * layer_count
        self.stats = PageCacheStats()

    @property
    def device_bytes(self) -> int:
        elements = self.cache.numel() + self.miss_device.numel()
        if self.compact is not None:
            elements += self.compact.numel()
        if self.prefetch_device is not None:
            elements += self.prefetch_device.numel()
        return int(elements * self.cache.element_size())

    @property
    def page_trace_sha256(self) -> str | None:
        tracer = self._page_trace
        return None if tracer is None else tracer.trace_sha256

    def page_trace_status(self) -> dict[str, Any]:
        tracer = self._page_trace
        return {"enabled": False} if tracer is None else tracer.status()

    def export_page_trace(self) -> dict[str, Any] | None:
        tracer = self._page_trace
        return None if tracer is None else tracer.export()

    def drain_page_trace(self) -> dict[str, Any] | None:
        tracer = self._page_trace
        return None if tracer is None else tracer.drain()

    def reset(self, *, clear_pages: bool) -> None:
        if self._prefetch_stream is not None:
            self._prefetch_stream.synchronize()
        self.stats = PageCacheStats()
        self.prefetch_stats = {"prefetched_pages": 0, "prefetch_rounds": 0}
        self._prev_routes.clear()
        if self._page_trace is not None:
            self._page_trace.clear()
        if clear_pages:
            self.entries.clear()
            self.decode_protected.clear()
            self.free_slots = list(range(self.capacity - 1, -1, -1))
            self.layer_entry_counts = [0] * len(self.layer_entry_counts)

    def _eviction_candidate(
        self,
        key: tuple[int, int],
        protected: set[tuple[int, int]],
    ) -> tuple[tuple[int, int] | None, str]:
        if self.cache_policy == GLOBAL_LRU_CACHE_POLICY:
            return (
                next(
                    (candidate for candidate in self.entries if candidate not in protected),
                    None,
                ),
                "fallback",
            )

        incoming_layer = key[0]
        same_layer = next(
            (
                candidate
                for candidate in self.entries
                if candidate[0] == incoming_layer and candidate not in protected
            ),
            None,
        )
        if (
            same_layer is not None
            and self.layer_entry_counts[incoming_layer] >= self.layer_quotas[incoming_layer]
        ):
            return same_layer, "same-layer"

        over_quota = next(
            (
                candidate
                for candidate in self.entries
                if candidate not in protected
                and self.layer_entry_counts[candidate[0]] > self.layer_quotas[candidate[0]]
            ),
            None,
        )
        if over_quota is not None:
            return over_quota, "over-quota"
        return (
            next(
                (candidate for candidate in self.entries if candidate not in protected),
                None,
            ),
            "fallback",
        )

    def _reserve_slot(
        self,
        key: tuple[int, int],
        protected: set[tuple[int, int]],
    ) -> int:
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            victim, reason = self._eviction_candidate(key, protected)
            if victim is None:
                raise RuntimeError("expert cache cannot evict a page from the active set")
            slot = self.entries.pop(victim)
            if victim in self.decode_protected:
                self.decode_protected.remove(victim)
                self.stats.decode_protected_demotions += 1
            self.layer_entry_counts[victim[0]] -= 1
            self.stats.evictions += 1
            if reason == "same-layer":
                self.stats.same_layer_evictions += 1
            elif reason == "over-quota":
                self.stats.over_quota_evictions += 1
            else:
                self.stats.fallback_evictions += 1
        self.entries[key] = slot
        self.layer_entry_counts[key[0]] += 1
        return slot

    def _acquire_resident(
        self,
        layer: int,
        top_indices: torch.Tensor,
        *,
        protect: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routed_experts = [int(value) for value in top_indices.detach().cpu().reshape(-1)]
        frequencies = Counter(routed_experts)
        expert_ids = sorted(frequencies)
        if len(expert_ids) > self.max_active_pages:
            raise RuntimeError(
                f"active expert set {len(expert_ids)} exceeds staging limit {self.max_active_pages}"
            )
        if self._prefetch_stream is not None:
            # order any in-flight prefetch index_copy_ before this acquire reads cache
            torch.cuda.current_stream(self.device).wait_stream(self._prefetch_stream)
        keys = [(int(layer), expert) for expert in expert_ids]
        if self._page_trace is not None:
            self._page_trace.record(
                phase="decode" if protect else "prefill",
                layer=int(layer),
                frequencies=frequencies,
            )
        protected = set(keys)
        misses: list[int] = []
        miss_slots: list[int] = []
        admission_keys = keys
        if self.cache_policy == LAYER_FREQUENCY_CACHE_POLICY:
            # Least-used routes enter first. The most frequently selected experts
            # become the most-recently-used pages and survive the next layer's spill.
            # `selected` remains sorted below because `_compact_route` uses searchsorted.
            admission_keys = sorted(
                keys,
                key=lambda item: (frequencies[item[1]], item[1]),
            )
        for key in admission_keys:
            if key in self.entries:
                self.entries.move_to_end(key)
                continue
            misses.append(key[1])
            miss_slots.append(self._reserve_slot(key, protected))

        if misses:
            count = len(misses)
            if self.host_tier is not None:
                self.host_tier.gather(layer, misses, self.stage[:count])
            else:
                self.store.gather(layer, misses, self.stage[:count])
            self.miss_device[:count].copy_(self.stage[:count], non_blocking=True)
            slot_tensor = torch.tensor(
                miss_slots,
                dtype=torch.long,
                device=self.device,
            )
            self.cache.index_copy_(0, slot_tensor, self.miss_device[:count])
            self.stats.host_to_device_bytes += count * self.layout.page_stride

        slots = torch.tensor(
            [self.entries[key] for key in keys],
            dtype=torch.long,
            device=self.device,
        )
        self.stats.page_requests += len(keys)
        self.stats.page_misses += len(misses)
        self.stats.page_hits += len(keys) - len(misses)
        self.stats.layer_acquires += 1
        self.stats.active_experts_sum += len(keys)
        if protect:
            new_protected = set(keys) - self.decode_protected
            self.decode_protected.update(keys)
            self.stats.decode_protected_promotions += len(new_protected)
        if self.route_prefetch:
            self._maybe_prefetch_next_layer(int(layer), protected)
            self._prev_routes[int(layer)] = expert_ids
        selected = torch.tensor(expert_ids, dtype=torch.long, device=self.device)
        return selected, slots

    def acquire(
        self,
        layer: int,
        top_indices: torch.Tensor,
        *,
        protect: bool = False,
    ) -> tuple[torch.Tensor, DeviceExpertLayer | DeviceInt4ExpertLayer]:
        """Acquire the established compact-copy binding.

        This compatibility method deliberately remains available even when a cache was opened
        with the slot-indirect candidate. In that case the fallback buffer is allocated lazily,
        so a failed hardware experiment can switch paths without reconstructing cache residency.
        """
        selected, slots = self._acquire_resident(
            layer,
            top_indices,
            protect=protect,
        )
        if self.compact is None:
            self.compact = torch.empty_like(self.miss_device)
        torch.index_select(
            self.cache,
            0,
            slots,
            out=self.compact[: int(selected.numel())],
        )
        return selected, self.store.view_pages(self.compact[: int(selected.numel())])

    def acquire_slot_binding(
        self,
        layer: int,
        top_indices: torch.Tensor,
        *,
        protect: bool = False,
    ) -> tuple[
        torch.Tensor,
        DeviceExpertLayer | DeviceInt4ExpertLayer,
        torch.Tensor,
    ]:
        """Bind active logical groups to physical cache slots without copying their pages."""
        selected, slots = self._acquire_resident(
            layer,
            top_indices,
            protect=protect,
        )
        return selected, self.store.view_pages(self.cache), slots

    def acquire_transient_prefill(
        self,
        layer: int,
        top_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        DeviceExpertLayer | DeviceInt4ExpertLayer,
        torch.Tensor,
    ]:
        """Execute a long prefill from transient pages and admit only its hottest routes.

        A layer-major prefill can touch every expert and turn a bounded decode cache into a
        one-pass scan buffer. This path transfers the layer's exact expert union once into the
        existing miss slab, executes directly from that slab, and admits only enough of the
        most frequently routed pages to fill the layer's quota. Existing quota-resident pages
        are preserved, so a new prompt cannot flush another request's decode working set.

        The execution-slab order puts newly admitted misses first, then remaining misses, then
        resident pages. ``group_slots`` maps the sorted logical expert IDs used by route
        compilation back to that physical order. Nonresident pages make one compact host gather
        and H2D copy; resident pages make one cache-to-slab D2D gather before any stale cache slot
        can be reused. Newly admitted pages therefore remain a contiguous slab prefix.
        """
        if self.prefill_page_policy != TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY:
            raise RuntimeError("transient prefill acquisition requires transient-frequency-v1")
        if self.cache_policy != LAYER_FREQUENCY_CACHE_POLICY:
            raise RuntimeError("transient-frequency-v1 prefill requires layer-frequency-lru-v1")
        routed_experts = [int(value) for value in top_indices.detach().cpu().reshape(-1)]
        frequencies = Counter(routed_experts)
        expert_ids = sorted(frequencies)
        count = len(expert_ids)
        if count > self.max_active_pages:
            raise RuntimeError(
                f"active expert set {count} exceeds staging limit {self.max_active_pages}"
            )
        if self._prefetch_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._prefetch_stream)

        keys = [(int(layer), expert) for expert in expert_ids]
        if self._page_trace is not None:
            self._page_trace.record(
                phase="prefill",
                layer=int(layer),
                frequencies=frequencies,
            )
        resident = {key for key in keys if key in self.entries}
        protected_layer = {
            key for key in self.decode_protected if key in self.entries and key[0] == int(layer)
        }
        seed_capacity = max(
            0,
            self.layer_quotas[int(layer)] - len(protected_layer),
        )
        ranked_seed_keys = sorted(
            (key for key in keys if key not in self.decode_protected),
            key=lambda item: (-frequencies[item[1]], item[1]),
        )[:seed_capacity]
        desired_seed_keys = set(ranked_seed_keys)
        stale_seed_keys = [
            key
            for key in self.entries
            if key[0] == int(layer)
            and key not in self.decode_protected
            and key not in desired_seed_keys
        ]
        admission_keys = [key for key in ranked_seed_keys if key not in self.entries]
        admission_set = set(admission_keys)
        nonresident_keys = admission_keys + [
            key for key in keys if key not in resident and key not in admission_set
        ]
        resident_keys = [key for key in keys if key in resident]
        transfer_keys = nonresident_keys + resident_keys
        transfer_experts = [key[1] for key in transfer_keys]
        miss_count = len(nonresident_keys)

        # Materialize resident bytes before stale entries are released. Their slots may be
        # immediately reused by admissions below, but current-prefill execution still observes
        # the exact pre-replacement page in its independent miss slab.
        if resident_keys:
            resident_slots = torch.tensor(
                [self.entries[key] for key in resident_keys],
                dtype=torch.long,
                device=self.device,
            )
            torch.index_select(
                self.cache,
                0,
                resident_slots,
                out=self.miss_device[miss_count:count],
            )

        for key in stale_seed_keys:
            self.free_slots.append(self.entries.pop(key))
            self.layer_entry_counts[int(layer)] -= 1

        if miss_count:
            miss_experts = transfer_experts[:miss_count]
            if self.host_tier is not None:
                self.host_tier.gather(layer, miss_experts, self.stage[:miss_count])
            else:
                self.store.gather(layer, miss_experts, self.stage[:miss_count])
            self.miss_device[:miss_count].copy_(
                self.stage[:miss_count],
                non_blocking=True,
            )

        admitted_keys: list[tuple[int, int]] = []
        admission_slots: list[int] = []
        if admission_keys:
            protected = self.decode_protected | set(keys)
            for key in admission_keys:
                try:
                    slot = self._reserve_slot(key, protected)
                except RuntimeError:
                    break
                admitted_keys.append(key)
                admission_slots.append(slot)
        if admission_slots:
            slot_tensor = torch.tensor(
                admission_slots,
                dtype=torch.long,
                device=self.device,
            )
            self.cache.index_copy_(
                0,
                slot_tensor,
                self.miss_device[: len(admitted_keys)],
            )

        physical_by_expert = {expert: position for position, expert in enumerate(transfer_experts)}
        group_slots = torch.tensor(
            [physical_by_expert[expert] for expert in expert_ids],
            dtype=torch.long,
            device=self.device,
        )
        selected = torch.tensor(expert_ids, dtype=torch.long, device=self.device)
        self.stats.page_requests += count
        self.stats.page_hits += len(resident_keys)
        self.stats.page_misses += miss_count
        self.stats.host_to_device_bytes += miss_count * self.layout.page_stride
        self.stats.layer_acquires += 1
        self.stats.active_experts_sum += count
        self.stats.transient_prefill_acquires += 1
        self.stats.transient_prefill_pages += count
        self.stats.transient_prefill_admitted_pages += len(admitted_keys)
        self.stats.transient_prefill_preserved_pages += len(resident.difference(stale_seed_keys))
        self.stats.transient_prefill_h2d_bytes += miss_count * self.layout.page_stride
        self.stats.transient_prefill_resident_materialization_d2d_bytes += (
            len(resident_keys) * self.layout.page_stride
        )
        self.stats.transient_prefill_admission_d2d_bytes += (
            len(admitted_keys) * self.layout.page_stride
        )
        self.stats.transient_prefill_replaced_pages += len(stale_seed_keys)
        self.stats.transient_prefill_skipped_protected_pages += len(admission_keys) - len(
            admitted_keys
        )
        return (
            selected,
            self.store.view_pages(self.miss_device[:count]),
            group_slots,
        )

    def _maybe_prefetch_next_layer(
        self,
        layer: int,
        protected: set[tuple[int, int]],
    ) -> None:
        """Stage the previous step's routes for layer+1 on the side stream."""
        nxt = layer + 1
        predicted = self._prev_routes.get(nxt)
        if not predicted or self._prefetch_stream is None:
            return
        wanted = [e for e in predicted if (nxt, e) not in self.entries]
        wanted = wanted[: self.max_active_pages]
        if not wanted:
            return
        slots = []
        keys = []
        for expert in wanted:
            key = (nxt, expert)
            try:
                slots.append(self._reserve_slot(key, protected))
            except RuntimeError:
                break  # cache saturated by the protected set; skip quietly
            keys.append(key)
        if not slots:
            return
        count = len(keys)
        stage = self.prefetch_stage
        assert stage is not None and self.prefetch_device is not None
        if self.host_tier is not None:
            self.host_tier.gather(nxt, [k[1] for k in keys], stage[:count])
        else:
            self.store.gather(nxt, [k[1] for k in keys], stage[:count])
        with torch.cuda.stream(self._prefetch_stream):
            self.prefetch_device[:count].copy_(stage[:count], non_blocking=True)
            slot_tensor = torch.tensor(slots, dtype=torch.long, device=self.device)
            self.cache.index_copy_(0, slot_tensor, self.prefetch_device[:count])
        self.stats.host_to_device_bytes += count * self.layout.page_stride
        self.prefetch_stats["prefetched_pages"] += count
        self.prefetch_stats["prefetch_rounds"] += 1


def expert_page_views(
    raw: torch.Tensor,
    layout: ExpertPageLayout,
) -> DeviceExpertLayer:
    groups = raw.shape[0]
    gate_up = raw[
        :,
        layout.gate_up_codes_offset : (layout.gate_up_codes_offset + layout.gate_up_codes_bytes),
    ].view(torch.float8_e4m3fn)
    gate_up = gate_up.reshape(
        groups,
        2 * layout.intermediate_size,
        layout.hidden_size,
    )
    gate_up_scales = raw[
        :,
        layout.gate_up_scales_offset : (layout.gate_up_scales_offset + layout.gate_up_scales_bytes),
    ].view(torch.float32)
    gate_up_scales = gate_up_scales.reshape(groups, 2 * layout.intermediate_size)
    down = raw[
        :,
        layout.down_codes_offset : layout.down_codes_offset + layout.down_codes_bytes,
    ].view(torch.float8_e4m3fn)
    down = down.reshape(groups, layout.hidden_size, layout.intermediate_size)
    down_scales = raw[
        :,
        layout.down_scales_offset : (layout.down_scales_offset + layout.down_scales_bytes),
    ].view(torch.float32)
    down_scales = down_scales.reshape(groups, layout.hidden_size)
    return DeviceExpertLayer(
        gate_up=gate_up,
        gate_up_scales=gate_up_scales,
        down=down,
        down_scales=down_scales,
    )


def int4_expert_page_views(
    raw: torch.Tensor,
    layout: Int4ExpertPageLayout,
) -> DeviceInt4ExpertLayer:
    if raw.ndim != 2 or raw.dtype != torch.uint8:
        raise ValueError("INT4 expert pages must be a rank-2 uint8 tensor")
    if int(raw.shape[1]) != layout.page_stride:
        raise ValueError(f"INT4 expert page stride {int(raw.shape[1])} != {layout.page_stride}")
    groups = int(raw.shape[0])
    gate_up = raw[
        :,
        layout.gate_up_codes_offset : (layout.gate_up_codes_offset + layout.gate_up_codes_bytes),
    ].reshape(
        groups,
        2 * layout.intermediate_size,
        layout.hidden_size // 2,
    )
    gate_up_scales = raw[
        :,
        layout.gate_up_scales_offset : (layout.gate_up_scales_offset + layout.gate_up_scales_bytes),
    ].view(torch.float32)
    gate_up_scales = gate_up_scales.reshape(
        groups,
        2 * layout.intermediate_size,
        layout.hidden_size // layout.group_size,
    )
    down = raw[
        :,
        layout.down_codes_offset : layout.down_codes_offset + layout.down_codes_bytes,
    ].reshape(
        groups,
        layout.hidden_size,
        layout.intermediate_size // 2,
    )
    down_scales = raw[
        :,
        layout.down_scales_offset : (layout.down_scales_offset + layout.down_scales_bytes),
    ].view(torch.float32)
    down_scales = down_scales.reshape(
        groups,
        layout.hidden_size,
        layout.intermediate_size // layout.group_size,
    )
    return DeviceInt4ExpertLayer(
        gate_up=gate_up,
        gate_up_scales=gate_up_scales,
        down=down,
        down_scales=down_scales,
    )


if triton is not None:

    @triton.jit
    def _grouped_fp8_strided_kernel(
        source_ptr,
        weight_ptr,
        source_scale_ptr,
        weight_scale_ptr,
        output_ptr,
        starts_ptr,
        counts_ptr,
        group_slots_ptr,
        weight_group_stride,
        weight_row_stride,
        scale_group_stride,
        n: tl.constexpr,
        k: tl.constexpr,
        use_group_slots: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        group = tl.program_id(0)
        physical_group = group
        if use_group_slots:
            physical_group = tl.load(group_slots_ptr + group)
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
                source = tl.load(
                    source_ptr + (start + offsets_m[:, None]) * k + offsets_k[None, :],
                    mask=(offsets_m[:, None] < count) & (offsets_k[None, :] < k),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + physical_group * weight_group_stride
                    + offsets_n[:, None] * weight_row_stride
                    + offsets_k[None, :],
                    mask=(offsets_n[:, None] < n) & (offsets_k[None, :] < k),
                    other=0.0,
                )
                accumulator += tl.dot(source, tl.trans(weight))
            source_scales = tl.load(
                source_scale_ptr + start + offsets_m,
                mask=offsets_m < count,
                other=0.0,
            )
            weight_scales = tl.load(
                weight_scale_ptr + physical_group * scale_group_stride + offsets_n,
                mask=offsets_n < n,
                other=0.0,
            )
            output = accumulator * source_scales[:, None] * weight_scales[None, :]
            tl.store(
                output_ptr + (start + offsets_m[:, None]) * n + offsets_n[None, :],
                output,
                mask=(offsets_m[:, None] < count) & (offsets_n[None, :] < n),
            )

    @triton.jit
    def _grouped_w4_strided_kernel(
        source_ptr,
        packed_ptr,
        scale_ptr,
        output_ptr,
        starts_ptr,
        counts_ptr,
        group_slots_ptr,
        packed_group_stride,
        packed_row_stride,
        scale_group_stride,
        scale_row_stride,
        n: tl.constexpr,
        k: tl.constexpr,
        group_size: tl.constexpr,
        use_group_slots: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        group = tl.program_id(0)
        physical_group = group
        if use_group_slots:
            physical_group = tl.load(group_slots_ptr + group)
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
                source = tl.load(
                    source_ptr + (start + offsets_m[:, None]) * k + offsets_k[None, :],
                    mask=(offsets_m[:, None] < count) & (offsets_k[None, :] < k),
                    other=0.0,
                )
                packed = tl.load(
                    packed_ptr
                    + physical_group * packed_group_stride
                    + offsets_n[:, None] * packed_row_stride
                    + offsets_k[None, :] // 2,
                    mask=(offsets_n[:, None] < n) & (offsets_k[None, :] < k),
                    other=0,
                )
                shift = (offsets_k[None, :] & 1) * 4
                codes = (packed >> shift) & 15
                scales = tl.load(
                    scale_ptr
                    + physical_group * scale_group_stride
                    + offsets_n[:, None] * scale_row_stride
                    + offsets_k[None, :] // group_size,
                    mask=(offsets_n[:, None] < n) & (offsets_k[None, :] < k),
                    other=0.0,
                )
                weights = ((codes.to(tl.float32) - 8.0) * scales).to(tl.bfloat16)
                accumulator += tl.dot(source, tl.trans(weights))
            tl.store(
                output_ptr + (start + offsets_m[:, None]) * n + offsets_n[None, :],
                accumulator,
                mask=(offsets_m[:, None] < count) & (offsets_n[None, :] < n),
            )

    @triton.jit
    def _grouped_w4_postscale_strided_kernel(
        source_ptr,
        packed_ptr,
        scale_ptr,
        output_ptr,
        starts_ptr,
        counts_ptr,
        group_slots_ptr,
        packed_group_stride,
        packed_row_stride,
        scale_group_stride,
        scale_row_stride,
        n: tl.constexpr,
        k: tl.constexpr,
        group_size: tl.constexpr,
        use_group_slots: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        """Dot exact signed BF16 codes, then apply one FP32 scale per complete G128."""
        group = tl.program_id(0)
        physical_group = group
        if use_group_slots:
            physical_group = tl.load(group_slots_ptr + group)
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
            for group_start in range(0, k, group_size):
                group_accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
                for inner_start in range(0, group_size, block_k):
                    offsets_k = group_start + inner_start + tl.arange(0, block_k)
                    source = tl.load(
                        source_ptr + (start + offsets_m[:, None]) * k + offsets_k[None, :],
                        mask=(offsets_m[:, None] < count) & (offsets_k[None, :] < k),
                        other=0.0,
                    )
                    packed = tl.load(
                        packed_ptr
                        + physical_group * packed_group_stride
                        + offsets_n[:, None] * packed_row_stride
                        + offsets_k[None, :] // 2,
                        mask=(offsets_n[:, None] < n) & (offsets_k[None, :] < k),
                        other=0,
                    )
                    shift = (offsets_k[None, :] & 1) * 4
                    codes = (packed >> shift) & 15
                    code_values = (codes.to(tl.float32) - 8.0).to(tl.bfloat16)
                    group_accumulator += tl.dot(source, tl.trans(code_values))
                weight_scale = tl.load(
                    scale_ptr
                    + physical_group * scale_group_stride
                    + offsets_n * scale_row_stride
                    + group_start // group_size,
                    mask=offsets_n < n,
                    other=0.0,
                )
                accumulator += group_accumulator * weight_scale[None, :]
            tl.store(
                output_ptr + (start + offsets_m[:, None]) * n + offsets_n[None, :],
                accumulator,
                mask=(offsets_m[:, None] < count) & (offsets_n[None, :] < n),
            )

    @triton.jit
    def _grouped_bf16_kernel(
        source_ptr,
        weight_ptr,
        output_ptr,
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
                source = tl.load(
                    source_ptr + (start + offsets_m[:, None]) * k + offsets_k[None, :],
                    mask=(offsets_m[:, None] < count) & (offsets_k[None, :] < k),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr + group * n * k + offsets_n[:, None] * k + offsets_k[None, :],
                    mask=(offsets_n[:, None] < n) & (offsets_k[None, :] < k),
                    other=0.0,
                )
                accumulator += tl.dot(source, tl.trans(weight))
            tl.store(
                output_ptr + (start + offsets_m[:, None]) * n + offsets_n[None, :],
                accumulator,
                mask=(offsets_m[:, None] < count) & (offsets_n[None, :] < n),
            )

    @triton.jit
    def _stable_route_reduce_kernel(
        expert_output_ptr,
        coefficients_ptr,
        inverse_assignments_ptr,
        output_ptr,
        hidden: tl.constexpr,
        top_k: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.program_id(1) * block + tl.arange(0, block)
        mask = offsets < hidden
        accumulator = tl.zeros((block,), dtype=tl.float32)
        for rank in range(top_k):
            assignment = tl.load(inverse_assignments_ptr + row * top_k + rank)
            value = tl.load(
                expert_output_ptr + assignment * hidden + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            coefficient = tl.load(coefficients_ptr + assignment).to(tl.float32)
            accumulator += value * coefficient
        tl.store(output_ptr + row * hidden + offsets, accumulator, mask=mask)


def quantize_fp8_activation(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = source.abs().amax(dim=1).float().div(FP8_MAX).clamp_min(1e-12)
    codes = (source.float() / scales[:, None]).clamp(-FP8_MAX, FP8_MAX)
    return codes.to(torch.float8_e4m3fn), scales


def _grouped_fp8_shape_contract(
    source: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    group_slots: torch.Tensor | None,
) -> tuple[int, int, int]:
    """Validate logical-group versus physical-page geometry without launching CUDA.

    ``weights`` may either be compact ``[logical_groups, N, K]`` storage or the whole page
    cache ``[physical_slots, N, K]``. The latter is legal only with one slot index per logical
    group. Keeping this contract outside Triton makes malformed bindings fail deterministically
    and gives CPU-only CI full coverage of the address geometry.
    """
    if source.ndim != 2:
        raise ValueError("grouped FP8 source must have shape [rows, contraction]")
    if weights.ndim != 3:
        raise ValueError("grouped FP8 weights must have shape [groups, output, contraction]")
    if starts.ndim != 1 or counts.ndim != 1 or starts.numel() != counts.numel():
        raise ValueError("grouped FP8 starts and counts must be equal-length vectors")
    logical_groups = int(starts.numel())
    physical_groups, output_features, contraction = map(int, weights.shape)
    if logical_groups < 1:
        raise ValueError("grouped FP8 execution requires at least one logical group")
    if int(source.shape[1]) != contraction:
        raise ValueError(
            "grouped FP8 source contraction differs from weight contraction: "
            f"{int(source.shape[1])} != {contraction}"
        )
    if tuple(weight_scales.shape) != (physical_groups, output_features):
        raise ValueError("grouped FP8 scales must have shape [physical_groups, output_features]")
    integral_dtypes = {torch.int32, torch.int64}
    if starts.dtype not in integral_dtypes or counts.dtype not in integral_dtypes:
        raise ValueError("grouped FP8 starts and counts must use int32 or int64")
    if weights.stride(2) != 1 or weight_scales.stride(1) != 1:
        raise ValueError("grouped FP8 weights and scales require contiguous innermost rows")
    tensors = (weights, weight_scales, starts, counts)
    if any(tensor.device != source.device for tensor in tensors):
        raise ValueError("grouped FP8 tensors must reside on the source device")
    if group_slots is None:
        if physical_groups != logical_groups:
            raise ValueError(
                "compact grouped FP8 weights require one physical group per logical group"
            )
    else:
        if group_slots.ndim != 1 or int(group_slots.numel()) != logical_groups:
            raise ValueError("grouped FP8 slot binding must have one index per logical group")
        if group_slots.dtype not in integral_dtypes:
            raise ValueError("grouped FP8 slot binding must use int32 or int64")
        if group_slots.device != source.device:
            raise ValueError("grouped FP8 slot binding must reside on the source device")
        if not group_slots.is_contiguous():
            raise ValueError("grouped FP8 slot binding must be contiguous")
        # CPU validation can prove bounds without adding a CUDA synchronization to the hot path.
        # CUDA bindings are minted internally by ExpertPageCache from its own slot ledger.
        if group_slots.device.type == "cpu" and group_slots.numel():
            minimum = int(group_slots.min().item())
            maximum = int(group_slots.max().item())
            if minimum < 0 or maximum >= physical_groups:
                raise ValueError(
                    "grouped FP8 slot binding addresses outside physical weight storage"
                )
    return logical_groups, output_features, contraction


def grouped_fp8_mm_strided(
    source: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    *,
    group_slots: torch.Tensor | None = None,
    max_rows: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    groups, output_features, contraction = _grouped_fp8_shape_contract(
        source,
        weights,
        weight_scales,
        starts,
        counts,
        group_slots,
    )
    if triton is None or not source.is_cuda:
        raise RuntimeError("grouped FP8 execution requires CUDA Triton")
    source_codes, source_scales = quantize_fp8_activation(source)
    output = torch.empty(
        (source.shape[0], output_features),
        device=source.device,
        dtype=out_dtype,
    )
    block_m, block_n, block_k = 16, 64, 64
    grid = (
        groups,
        triton.cdiv(max_rows, block_m) * triton.cdiv(output_features, block_n),
    )
    _grouped_fp8_strided_kernel[grid](
        source_codes,
        weights,
        source_scales,
        weight_scales,
        output,
        starts,
        counts,
        counts if group_slots is None else group_slots,
        weights.stride(0),
        weights.stride(1),
        weight_scales.stride(0),
        n=output_features,
        k=contraction,
        use_group_slots=group_slots is not None,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=3,
    )
    return output


def _grouped_w4_shape_contract(
    source: torch.Tensor,
    packed_weights: torch.Tensor,
    weight_scales: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    group_slots: torch.Tensor | None,
    *,
    group_size: int = W4_GROUP_SIZE,
) -> tuple[int, int, int]:
    """Validate compact or slot-indirect grouped W4 page geometry without CUDA."""

    if source.ndim != 2:
        raise ValueError("grouped W4 source must have shape [rows, contraction]")
    if packed_weights.ndim != 3 or packed_weights.dtype != torch.uint8:
        raise ValueError("grouped W4 weights must be uint8 with shape [groups, output, packed-k]")
    if weight_scales.ndim != 3 or weight_scales.dtype != torch.float32:
        raise ValueError("grouped W4 scales must be float32 with shape [groups, output, k-groups]")
    if starts.ndim != 1 or counts.ndim != 1 or starts.numel() != counts.numel():
        raise ValueError("grouped W4 starts and counts must be equal-length vectors")
    logical_groups = int(starts.numel())
    physical_groups, output_features, packed_k = map(int, packed_weights.shape)
    contraction = packed_k * 2
    group_size = int(group_size)
    if logical_groups < 1:
        raise ValueError("grouped W4 execution requires at least one logical group")
    if group_size < 1 or contraction % group_size:
        raise ValueError("grouped W4 contraction must be divisible by group_size")
    if int(source.shape[1]) != contraction:
        raise ValueError(
            "grouped W4 source contraction differs from packed weight contraction: "
            f"{int(source.shape[1])} != {contraction}"
        )
    expected_scales = (
        physical_groups,
        output_features,
        contraction // group_size,
    )
    if tuple(weight_scales.shape) != expected_scales:
        raise ValueError(
            f"grouped W4 scales have shape {tuple(weight_scales.shape)}, expected {expected_scales}"
        )
    integral_dtypes = {torch.int32, torch.int64}
    if starts.dtype not in integral_dtypes or counts.dtype not in integral_dtypes:
        raise ValueError("grouped W4 starts and counts must use int32 or int64")
    if source.stride(1) != 1 or source.stride(0) != contraction:
        raise ValueError("grouped W4 source must be row-major contiguous")
    if packed_weights.stride(2) != 1 or weight_scales.stride(2) != 1:
        raise ValueError("grouped W4 weights and scales require contiguous innermost rows")
    tensors = (packed_weights, weight_scales, starts, counts)
    if any(tensor.device != source.device for tensor in tensors):
        raise ValueError("grouped W4 tensors must reside on the source device")
    if group_slots is None:
        if physical_groups != logical_groups:
            raise ValueError(
                "compact grouped W4 weights require one physical group per logical group"
            )
    else:
        if group_slots.ndim != 1 or int(group_slots.numel()) != logical_groups:
            raise ValueError("grouped W4 slot binding must have one index per logical group")
        if group_slots.dtype not in integral_dtypes:
            raise ValueError("grouped W4 slot binding must use int32 or int64")
        if group_slots.device != source.device:
            raise ValueError("grouped W4 slot binding must reside on the source device")
        if not group_slots.is_contiguous():
            raise ValueError("grouped W4 slot binding must be contiguous")
        if group_slots.device.type == "cpu" and group_slots.numel():
            minimum = int(group_slots.min().item())
            maximum = int(group_slots.max().item())
            if minimum < 0 or maximum >= physical_groups:
                raise ValueError(
                    "grouped W4 slot binding addresses outside physical weight storage"
                )
    return logical_groups, output_features, contraction


def grouped_w4_mm_strided(
    source: torch.Tensor,
    packed_weights: torch.Tensor,
    weight_scales: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    *,
    group_slots: torch.Tensor | None = None,
    group_size: int = W4_GROUP_SIZE,
    arithmetic_policy: str | None = None,
    max_rows: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    groups, output_features, contraction = _grouped_w4_shape_contract(
        source,
        packed_weights,
        weight_scales,
        starts,
        counts,
        group_slots,
        group_size=group_size,
    )
    normalized_arithmetic = normalize_w4_arithmetic_policy(arithmetic_policy)
    if (
        normalized_arithmetic == W4_POSTSCALE_BF16_ARITHMETIC_POLICY
        and int(group_size) != W4_GROUP_SIZE
    ):
        raise ValueError(
            f"{W4_POSTSCALE_BF16_ARITHMETIC_POLICY} requires group_size={W4_GROUP_SIZE}"
        )
    if normalized_arithmetic == W4_POSTSCALE_BF16_ARITHMETIC_POLICY and (
        source.dtype != torch.bfloat16 or out_dtype != torch.bfloat16
    ):
        raise ValueError(f"{W4_POSTSCALE_BF16_ARITHMETIC_POLICY} requires BF16 source and output")
    if triton is None or not source.is_cuda:
        raise RuntimeError("grouped W4 execution requires CUDA Triton")
    output = torch.empty(
        (source.shape[0], output_features),
        device=source.device,
        dtype=out_dtype,
    )
    if normalized_arithmetic == W4_POSTSCALE_BF16_ARITHMETIC_POLICY:
        launch_config = _w4_postscale_launch_config(
            groups,
            output_features,
            contraction,
        )
    else:
        launch_config = W4_GROUPED_LAUNCH_CONFIG
    block_m, block_n, block_k, num_warps, num_stages = launch_config
    grid = (
        groups,
        triton.cdiv(max_rows, block_m) * triton.cdiv(output_features, block_n),
    )
    kernel = (
        _grouped_w4_postscale_strided_kernel
        if normalized_arithmetic == W4_POSTSCALE_BF16_ARITHMETIC_POLICY
        else _grouped_w4_strided_kernel
    )
    kernel[grid](
        source,
        packed_weights,
        weight_scales,
        output,
        starts,
        counts,
        counts if group_slots is None else group_slots,
        packed_weights.stride(0),
        packed_weights.stride(1),
        weight_scales.stride(0),
        weight_scales.stride(1),
        n=output_features,
        k=contraction,
        group_size=group_size,
        use_group_slots=group_slots is not None,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def grouped_bf16_mm(
    source: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    *,
    max_rows: int,
) -> torch.Tensor:
    if triton is None or not source.is_cuda:
        raise RuntimeError("grouped BF16 execution requires CUDA Triton")
    groups, output_features, contraction = weights.shape
    output = torch.empty(
        (source.shape[0], output_features),
        device=source.device,
        dtype=source.dtype,
    )
    block_m, block_n, block_k = 16, 64, 32
    grid = (
        groups,
        triton.cdiv(max_rows, block_m) * triton.cdiv(output_features, block_n),
    )
    _grouped_bf16_kernel[grid](
        source,
        weights,
        output,
        starts,
        counts,
        n=output_features,
        k=contraction,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=3,
    )
    return output


@dataclass
class ExpertBackendStats:
    grouped_kernel_calls: int = 0
    route_compiler_calls: int = 0
    expert_assignments: int = 0
    active_experts: int = 0
    addressed_source_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "grouped_kernel_calls": self.grouped_kernel_calls,
            "route_compiler_calls": self.route_compiler_calls,
            "expert_assignments": self.expert_assignments,
            "active_experts": self.active_experts,
            "addressed_source_bytes": self.addressed_source_bytes,
            "addressed_source_gb": self.addressed_source_bytes / 1e9,
        }


def _compact_route(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    selected_experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    compact_indices = torch.searchsorted(selected_experts, top_indices)
    token_ids, coefficients, starts, counts, _metadata_s = grouped_route_tensors(
        compact_indices,
        top_weights,
        num_experts=int(selected_experts.numel()),
    )
    return token_ids, coefficients, starts, counts


def _compact_route_with_inverse(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    selected_experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile a compact route and retain its stable-rank inverse for reduction."""
    compact_indices = torch.searchsorted(selected_experts, top_indices)
    token_ids, coefficients, starts, counts, inverse, _metadata_s = (
        grouped_route_tensors_with_inverse(
            compact_indices,
            top_weights,
            num_experts=int(selected_experts.numel()),
        )
    )
    return token_ids, coefficients, starts, counts, inverse


def stable_route_reduce(
    expert_output: torch.Tensor,
    coefficients: torch.Tensor,
    token_ids: torch.Tensor,
    counts: torch.Tensor,
    top_indices: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    inverse_assignments: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce expert-major rows in original top-k rank order without atomics.

    CUDA route compaction uses atomic cursors, so rows within one expert group may appear in a
    different order between launches. The token IDs and coefficients remain aligned, but a
    colliding ``index_add_`` can then sum a token's eight contributions in different orders.
    This helper reconstructs the unique original ``[token, route-rank]`` destination for every
    expert-major row and performs a fixed-rank FP32 reduction.
    """
    if top_indices.ndim != 2:
        raise ValueError("stable route reduction requires [rows, top_k] indices")
    rows, top_k = map(int, top_indices.shape)
    assignments = rows * top_k
    if expert_output.ndim != 2 or int(expert_output.shape[0]) != assignments:
        raise ValueError("expert output rows must equal rows * top_k")
    if token_ids.ndim != 1 or int(token_ids.numel()) != assignments:
        raise ValueError("stable route token IDs must cover every assignment")
    if coefficients.ndim != 1 or int(coefficients.numel()) != assignments:
        raise ValueError("stable route coefficients must cover every assignment")
    if counts.ndim != 1 or int(counts.numel()) != int(selected_experts.numel()):
        raise ValueError("stable route counts must match selected experts")
    tensors = (coefficients, token_ids, counts, top_indices, selected_experts)
    if any(tensor.device != expert_output.device for tensor in tensors):
        raise ValueError("stable route tensors must share one device")

    inverse = inverse_assignments
    if inverse is not None:
        if inverse.ndim != 1 or int(inverse.numel()) != assignments:
            raise ValueError("stable route inverse must cover every assignment")
        if inverse.dtype not in (torch.int32, torch.int64):
            raise ValueError("stable route inverse must use int32 or int64")
        if inverse.device != expert_output.device:
            raise ValueError("stable route inverse must share the expert-output device")
    else:
        compact_routes = torch.searchsorted(selected_experts, top_indices)
        group_ids = torch.repeat_interleave(
            torch.arange(
                int(selected_experts.numel()),
                device=expert_output.device,
                dtype=torch.long,
            ),
            counts.to(torch.long),
            output_size=assignments,
        )
        routed_groups = compact_routes.index_select(0, token_ids)
        matches = routed_groups == group_ids[:, None]
        if not expert_output.is_cuda and not bool((matches.sum(dim=1) == 1).all()):
            raise ValueError("each expert-major row must map to one original route rank")
        ranks = matches.to(torch.int64).argmax(dim=1)
        destinations = token_ids * top_k + ranks
        inverse = torch.empty(
            assignments,
            device=expert_output.device,
            dtype=torch.long,
        )
        inverse.index_copy_(
            0,
            destinations,
            torch.arange(assignments, device=expert_output.device, dtype=torch.long),
        )

    hidden = int(expert_output.shape[1])
    if expert_output.is_cuda:
        if triton is None:
            raise RuntimeError("stable CUDA route reduction requires Triton")
        output = torch.empty(
            (rows, hidden),
            device=expert_output.device,
            dtype=torch.float32,
        )
        block = 256
        grid = (rows, triton.cdiv(hidden, block))
        _stable_route_reduce_kernel[grid](
            expert_output,
            coefficients,
            inverse,
            output,
            hidden=hidden,
            top_k=top_k,
            block=block,
            num_warps=4,
        )
        return output

    ordered_output = expert_output.index_select(0, inverse).reshape(
        rows,
        top_k,
        hidden,
    )
    ordered_coefficients = coefficients.index_select(0, inverse).reshape(rows, top_k)
    contributions = ordered_output.float() * ordered_coefficients.float().unsqueeze(-1)
    output = contributions[:, 0]
    for rank in range(1, top_k):
        output = output + contributions[:, rank]
    return output


class PagedFP8ExpertBackend:
    def __init__(
        self,
        cache: ExpertPageCache,
        *,
        dtype: torch.dtype,
        route_reduction_policy: str | None = None,
    ):
        self.cache = cache
        self.dtype = dtype
        configured_reduction = (
            route_reduction_policy
            if route_reduction_policy is not None
            else os.environ.get(
                "MRUN_QWEN3_MOE_ROUTE_REDUCTION_POLICY",
                DEFAULT_ROUTE_REDUCTION_POLICY,
            )
        )
        self.route_reduction_policy = str(configured_reduction).strip().lower()
        if self.route_reduction_policy not in SUPPORTED_ROUTE_REDUCTION_POLICIES:
            raise ValueError(
                "unsupported Qwen3 MoE route reduction policy "
                f"{self.route_reduction_policy!r}; expected one of "
                f"{sorted(SUPPORTED_ROUTE_REDUCTION_POLICIES)}"
            )
        self.stats = ExpertBackendStats()

    def reset_stats(self, *, clear_pages: bool) -> None:
        self.stats = ExpertBackendStats()
        self.cache.reset(clear_pages=clear_pages)

    def moe(
        self,
        layer: int,
        source: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        *,
        prefill: bool = False,
    ) -> torch.Tensor:
        group_slots: torch.Tensor | None
        if prefill and self.cache.prefill_page_policy == TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY:
            selected, page, group_slots = self.cache.acquire_transient_prefill(
                layer,
                top_indices,
            )
        elif self.cache.page_binding_policy == SLOT_INDIRECT_PAGE_BINDING_POLICY:
            selected, page, group_slots = self.cache.acquire_slot_binding(
                layer,
                top_indices,
                protect=not prefill,
            )
        else:
            selected, page = self.cache.acquire(
                layer,
                top_indices,
                protect=not prefill,
            )
            group_slots = None
        inverse: torch.Tensor | None = None
        if self.route_reduction_policy == STABLE_ROUTE_REDUCTION_POLICY:
            token_ids, coefficients, starts, counts, inverse = _compact_route_with_inverse(
                top_indices,
                top_weights,
                selected,
            )
        else:
            token_ids, coefficients, starts, counts = _compact_route(
                top_indices,
                top_weights,
                selected,
            )
        rows = source.index_select(0, token_ids)
        gate_up = grouped_fp8_mm_strided(
            rows,
            page.gate_up,
            page.gate_up_scales,
            starts,
            counts,
            group_slots=group_slots,
            max_rows=source.shape[0],
            out_dtype=self.dtype,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        expert_output = grouped_fp8_mm_strided(
            activated,
            page.down,
            page.down_scales,
            starts,
            counts,
            group_slots=group_slots,
            max_rows=source.shape[0],
            out_dtype=self.dtype,
        )
        if self.route_reduction_policy == STABLE_ROUTE_REDUCTION_POLICY:
            output = stable_route_reduce(
                expert_output,
                coefficients,
                token_ids,
                counts,
                top_indices,
                selected,
                inverse_assignments=inverse,
            )
        else:
            output = torch.zeros(
                (source.shape[0], source.shape[1]),
                device=source.device,
                dtype=torch.float32,
            )
            output.index_add_(
                0,
                token_ids,
                expert_output.float() * coefficients.float().unsqueeze(1),
            )
        self.stats.grouped_kernel_calls += 2
        self.stats.route_compiler_calls += 1
        self.stats.expert_assignments += int(top_indices.numel())
        self.stats.active_experts += int(selected.numel())
        self.stats.addressed_source_bytes += int(selected.numel()) * self.cache.layout.page_stride
        return output.to(self.dtype)

    def report(self) -> dict[str, Any]:
        return {
            "kind": "paged-fp8-qstore",
            "kernel": (
                "triton-slot-indirect-grouped-e4m3fn"
                if self.cache.page_binding_policy == SLOT_INDIRECT_PAGE_BINDING_POLICY
                else "triton-strided-grouped-e4m3fn"
            ),
            "cache_policy": self.cache.cache_policy,
            "page_binding_policy": self.cache.page_binding_policy,
            "prefill_page_policy": self.cache.prefill_page_policy,
            "route_reduction_policy": self.route_reduction_policy,
            "compact_page_materialized": self.cache.compact is not None,
            "compact_page_bytes": (
                0
                if self.cache.compact is None
                else self.cache.compact.numel() * self.cache.compact.element_size()
            ),
            "backend": self.stats.as_dict(),
            "page_cache": self.cache.stats.as_dict(),
            "cache_capacity_pages": self.cache.capacity,
            "cache_device_bytes": self.cache.device_bytes,
            "page_stride": self.cache.layout.page_stride,
            "layer_quota_pages": {
                "minimum": min(self.cache.layer_quotas),
                "maximum": max(self.cache.layer_quotas),
            },
            "resident_pages_by_layer": list(self.cache.layer_entry_counts),
            "decode_protected_pages": len(self.cache.decode_protected),
        }


class PagedInt4ExpertBackend(PagedFP8ExpertBackend):
    def __init__(
        self,
        cache: ExpertPageCache,
        *,
        dtype: torch.dtype,
        route_reduction_policy: str | None = None,
        w4_arithmetic_policy: str | None = None,
    ):
        if not isinstance(cache.store, PackedInt4ExpertStore):
            raise ValueError("PagedInt4ExpertBackend requires a packed INT4 expert store")
        self.w4_arithmetic_policy = normalize_w4_arithmetic_policy(w4_arithmetic_policy)
        super().__init__(
            cache,
            dtype=dtype,
            route_reduction_policy=route_reduction_policy,
        )

    def moe(
        self,
        layer: int,
        source: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        *,
        prefill: bool = False,
    ) -> torch.Tensor:
        group_slots: torch.Tensor | None
        if prefill and self.cache.prefill_page_policy == TRANSIENT_FREQUENCY_PREFILL_PAGE_POLICY:
            selected, page, group_slots = self.cache.acquire_transient_prefill(
                layer,
                top_indices,
            )
        elif self.cache.page_binding_policy == SLOT_INDIRECT_PAGE_BINDING_POLICY:
            selected, page, group_slots = self.cache.acquire_slot_binding(
                layer,
                top_indices,
                protect=not prefill,
            )
        else:
            selected, page = self.cache.acquire(
                layer,
                top_indices,
                protect=not prefill,
            )
            group_slots = None
        if not isinstance(page, DeviceInt4ExpertLayer):
            raise RuntimeError("INT4 backend received pages from a different codec")
        inverse: torch.Tensor | None = None
        if self.route_reduction_policy == STABLE_ROUTE_REDUCTION_POLICY:
            token_ids, coefficients, starts, counts, inverse = _compact_route_with_inverse(
                top_indices,
                top_weights,
                selected,
            )
        else:
            token_ids, coefficients, starts, counts = _compact_route(
                top_indices,
                top_weights,
                selected,
            )
        rows = source.index_select(0, token_ids)
        gate_up = grouped_w4_mm_strided(
            rows,
            page.gate_up,
            page.gate_up_scales,
            starts,
            counts,
            group_slots=group_slots,
            group_size=self.cache.layout.group_size,
            arithmetic_policy=self.w4_arithmetic_policy,
            max_rows=source.shape[0],
            out_dtype=self.dtype,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        expert_output = grouped_w4_mm_strided(
            activated,
            page.down,
            page.down_scales,
            starts,
            counts,
            group_slots=group_slots,
            group_size=self.cache.layout.group_size,
            arithmetic_policy=self.w4_arithmetic_policy,
            max_rows=source.shape[0],
            out_dtype=self.dtype,
        )
        if self.route_reduction_policy == STABLE_ROUTE_REDUCTION_POLICY:
            output = stable_route_reduce(
                expert_output,
                coefficients,
                token_ids,
                counts,
                top_indices,
                selected,
                inverse_assignments=inverse,
            )
        else:
            output = torch.zeros(
                (source.shape[0], source.shape[1]),
                device=source.device,
                dtype=torch.float32,
            )
            output.index_add_(
                0,
                token_ids,
                expert_output.float() * coefficients.float().unsqueeze(1),
            )
        self.stats.grouped_kernel_calls += 2
        self.stats.route_compiler_calls += 1
        self.stats.expert_assignments += int(top_indices.numel())
        self.stats.active_experts += int(selected.numel())
        self.stats.addressed_source_bytes += int(selected.numel()) * self.cache.layout.page_stride
        return output.to(self.dtype)

    def report(self) -> dict[str, Any]:
        report = super().report()
        binding = (
            "slot-indirect"
            if self.cache.page_binding_policy == SLOT_INDIRECT_PAGE_BINDING_POLICY
            else "strided"
        )
        arithmetic = (
            "-postscale-bf16"
            if self.w4_arithmetic_policy == W4_POSTSCALE_BF16_ARITHMETIC_POLICY
            else ""
        )
        report.update(
            {
                "kind": "paged-w4-qstore",
                "codec": W4_STORE_CODEC,
                "arithmetic_policy": self.w4_arithmetic_policy,
                "kernel": f"triton-{binding}-grouped-w4a16-g128{arithmetic}",
            }
        )
        return report


class PackedBF16ExpertBackend:
    def __init__(
        self,
        reader: TensorReader,
        *,
        device: str,
        dtype: torch.dtype,
    ):
        self.reader = reader
        self.device = device
        self.dtype = dtype
        self.stats = ExpertBackendStats()

    def reset_stats(self, *, clear_pages: bool = True) -> None:
        del clear_pages
        self.stats = ExpertBackendStats()

    def moe(
        self,
        layer: int,
        source: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        selected_ids = sorted({int(value) for value in top_indices.detach().cpu().reshape(-1)})
        selected = torch.tensor(selected_ids, dtype=torch.long, device=source.device)
        token_ids, coefficients, starts, counts = _compact_route(
            top_indices,
            top_weights,
            selected,
        )
        gate_up_host: list[torch.Tensor] = []
        down_host: list[torch.Tensor] = []
        source_bytes = 0
        for expert in selected_ids:
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            gate = self.reader.get(f"{prefix}.gate_proj.weight")
            up = self.reader.get(f"{prefix}.up_proj.weight")
            down = self.reader.get(f"{prefix}.down_proj.weight")
            gate_up_host.append(torch.cat((gate, up), dim=0))
            down_host.append(down)
            source_bytes += sum(
                tensor.numel() * tensor.element_size() for tensor in (gate, up, down)
            )
        gate_up_weight = torch.stack(gate_up_host).to(
            device=self.device,
            dtype=self.dtype,
        )
        down_weight = torch.stack(down_host).to(
            device=self.device,
            dtype=self.dtype,
        )
        rows = source.index_select(0, token_ids)
        gate_up = grouped_bf16_mm(
            rows,
            gate_up_weight,
            starts,
            counts,
            max_rows=source.shape[0],
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        expert_output = grouped_bf16_mm(
            activated,
            down_weight,
            starts,
            counts,
            max_rows=source.shape[0],
        )
        output = torch.zeros(
            (source.shape[0], source.shape[1]),
            device=source.device,
            dtype=torch.float32,
        )
        output.index_add_(
            0,
            token_ids,
            expert_output.float() * coefficients.float().unsqueeze(1),
        )
        self.reader.release()
        self.stats.grouped_kernel_calls += 2
        self.stats.route_compiler_calls += 1
        self.stats.expert_assignments += int(top_indices.numel())
        self.stats.active_experts += len(selected_ids)
        self.stats.addressed_source_bytes += source_bytes
        return output.to(self.dtype)

    def report(self) -> dict[str, Any]:
        return {
            "kind": "packed-bf16-reference",
            "kernel": "triton-grouped-bf16",
            "backend": self.stats.as_dict(),
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
    # Production skeletons bind q/k/v as non-overlapping views of this one allocation. The
    # scalar path retains its three established projections; ragged decode consumes one GEMM.
    qkv_proj: torch.Tensor | None = None


class ResidentQwen3Skeleton:
    def __init__(
        self,
        reader: TensorReader,
        cfg: dict[str, Any],
        *,
        device: str,
        dtype: torch.dtype,
    ):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        def load(name: str) -> torch.Tensor:
            return (
                reader.get(name)
                .to(
                    device=device,
                    dtype=dtype,
                )
                .contiguous()
            )

        self.embedding = load("model.embed_tokens.weight")
        self.layers: list[LayerWeights] = []
        for layer in range(int(cfg["num_hidden_layers"])):
            prefix = f"model.layers.{layer}"
            q_proj = load(f"{prefix}.self_attn.q_proj.weight")
            k_proj = load(f"{prefix}.self_attn.k_proj.weight")
            v_proj = load(f"{prefix}.self_attn.v_proj.weight")
            qkv_proj = torch.cat((q_proj, k_proj, v_proj), dim=0).contiguous()
            q_rows = int(q_proj.shape[0])
            k_rows = int(k_proj.shape[0])
            q_view = qkv_proj[:q_rows]
            k_view = qkv_proj[q_rows : q_rows + k_rows]
            v_view = qkv_proj[q_rows + k_rows :]
            fused_storage = qkv_proj.untyped_storage().data_ptr()
            if (
                any(
                    view.untyped_storage().data_ptr() != fused_storage
                    for view in (q_view, k_view, v_view)
                )
                or q_view.storage_offset() != 0
                or k_view.storage_offset() != q_view.numel()
                or v_view.storage_offset() != q_view.numel() + k_view.numel()
                or q_view.numel() + k_view.numel() + v_view.numel() != qkv_proj.numel()
            ):
                raise RuntimeError("fused QKV views do not partition exactly one resident storage")
            self.layers.append(
                LayerWeights(
                    input_norm=load(f"{prefix}.input_layernorm.weight"),
                    post_norm=load(f"{prefix}.post_attention_layernorm.weight"),
                    q_norm=load(f"{prefix}.self_attn.q_norm.weight"),
                    k_norm=load(f"{prefix}.self_attn.k_norm.weight"),
                    q_proj=q_view,
                    k_proj=k_view,
                    v_proj=v_view,
                    o_proj=load(f"{prefix}.self_attn.o_proj.weight"),
                    router=load(f"{prefix}.mlp.gate.weight"),
                    qkv_proj=qkv_proj,
                )
            )
            reader.release()
        self.final_norm = load("model.norm.weight")
        # A tied checkpoint stores no lm_head at all; reusing the embedding is what
        # tie_word_embeddings MEANS, and without this the skeleton KeyErrors on tied MoE
        # checkpoints instead of running.
        self.tied_lm_head = not reader.has("lm_head.weight")
        self.lm_head = self.embedding if self.tied_lm_head else load("lm_head.weight")
        reader.release()

    @property
    def device_bytes(self) -> int:
        tensors = [self.embedding, self.final_norm]
        if not getattr(self, "tied_lm_head", False):
            tensors.append(self.lm_head)  # tied: same storage, do not double-count
        for layer in self.layers:
            tensors.extend(
                (
                    layer.input_norm,
                    layer.post_norm,
                    layer.q_norm,
                    layer.k_norm,
                    layer.q_proj,
                    layer.k_proj,
                    layer.v_proj,
                    layer.o_proj,
                    layer.router,
                )
            )
        return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class StaticKVCache:
    layers: list[LayerKV]
    batch_size: int
    capacity: int
    length: int = 0

    @property
    def device_bytes(self) -> int:
        return int(
            sum(
                item.key.numel() * item.key.element_size()
                + item.value.numel() * item.value.element_size()
                for item in self.layers
            )
        )


@dataclass
class ForkedKVCache:
    """One immutable B1 parent plus branch-local post-cut K/V.

    ``layers`` contains only the tokens written after the cut. Attention consumes parent and
    branch tensors as two physical segments; parent K/V is never copied or broadcast into branch
    storage. All branches currently advance in lockstep, matching the equal-horizon Saturn panel.
    """

    parent: StaticKVCache
    layers: list[LayerKV]
    batch_size: int
    capacity: int
    delta_capacity: int
    delta_length: int = 0

    @property
    def length(self) -> int:
        return int(self.parent.length + self.delta_length)

    @property
    def device_bytes(self) -> int:
        return int(
            sum(
                item.key.numel() * item.key.element_size()
                + item.value.numel() * item.value.element_size()
                for item in self.layers
            )
        )

    @property
    def committed_bytes(self) -> int:
        return int(
            sum(
                (
                    item.key[:, :, : self.delta_length].numel()
                    + item.value[:, :, : self.delta_length].numel()
                )
                * item.key.element_size()
                for item in self.layers
            )
        )


@dataclass
class ForwardResult:
    logits: torch.Tensor | None
    cache: StaticKVCache | ForkedKVCache
    routes: list[torch.Tensor] = field(default_factory=list)
    # Residual-stream taps (populated only when the caller asks for them). With an all-layer
    # capture, ``hidden_states`` is HF ``output_hidden_states`` layout. A selected capture keeps
    # only the requested boundaries; ``hidden_state_layers`` is then the exact parallel address
    # map. ``final_hidden`` is post-final-norm, i.e. what the LM head consumes.
    hidden_states: list[torch.Tensor] = field(default_factory=list)
    hidden_state_layers: tuple[int, ...] = ()
    hidden_capture_evidence: HiddenCaptureEvidence | None = None
    final_hidden: torch.Tensor | None = None


@dataclass(frozen=True)
class HiddenCaptureEvidence:
    """Small execution receipt proving that capture selection reached the forward loop.

    This is metadata, not another activation artifact. Boundary ``-1`` is the embedding output;
    boundary ``L`` is the residual after decoder block ``L``.
    """

    requested_boundaries: tuple[int, ...]
    available_boundary_count: int
    hidden_last_only: bool
    retained_tensors: int
    retained_elements: int
    retained_bytes: int
    all_boundary_tape_elements: int
    all_boundary_tape_bytes: int
    avoided_retained_bytes: int
    execution_time_pushdown: bool


@dataclass(frozen=True)
class RaggedDecodeEvidence:
    """Opt-in decode taps for a sealed parity gate, never the serving hot-path result.

    All tensors remain on the runtime device.  In particular, token selection is still performed
    by the same on-device ``argmax`` as the default path; the caller decides whether and when to
    synchronize/copy evidence after the measured region.
    """

    token_ids: torch.Tensor
    logits: torch.Tensor
    routes: tuple[torch.Tensor, ...]


def rms_norm(source: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    values = source.float()
    normalized = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + eps)
    return weight * normalized.to(source.dtype)


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
        torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
            device=positions.device,
        )
        / head_dim
    )
    inverse = 1.0 / torch.pow(
        torch.tensor(theta, device=positions.device),
        dimensions,
    )
    frequencies = torch.outer(positions.float(), inverse)
    embeddings = torch.cat((frequencies, frequencies), dim=-1)
    return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def _device_matches_runtime(actual: torch.device, configured: str | torch.device) -> bool:
    """Treat an unindexed device alias (``cuda``) as its live indexed device."""

    expected = torch.device(configured)
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


def _native_sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout_p: float,
    is_causal: bool,
) -> torch.Tensor:
    """Run SDPA without physically expanding grouped K/V heads.

    Native GQA keeps K/V at ``num_key_value_heads`` and lets the SDPA backend
    map query heads to their K/V group. A ``repeat_interleave`` fallback would
    expand the full resident KV prefix on every layer and decode step, so older
    PyTorch builds fail closed.
    """
    query_heads = int(query.shape[-3])
    key_heads = int(key.shape[-3])
    value_heads = int(value.shape[-3])
    if key_heads != value_heads:
        raise ValueError(
            "SDPA requires key and value to have the same number of heads; "
            f"got key={key_heads}, value={value_heads}"
        )
    if query_heads == key_heads:
        # Preserve the established MHA call, including compatibility with torch
        # releases that predate the enable_gqa keyword.
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
    if key_heads <= 0 or query_heads % key_heads:
        raise ValueError(
            "native grouped-query attention requires query heads to be an exact "
            f"multiple of K/V heads; got query={query_heads}, key/value={key_heads}"
        )
    try:
        if query.is_cuda:
            # The unconstrained dispatcher may select its math GQA backend,
            # whose documented definition physically repeats K/V. Production
            # CUDA therefore pins FlashAttention so compact K/V is a residency
            # invariant rather than a Python-level implementation detail.
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    enable_gqa=True,
                )
        # Qwen3MoeCudaEngine is CUDA-only; CPU stays available for parity tests.
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=True,
        )
    except (ImportError, TypeError, NotImplementedError, RuntimeError) as error:
        raise RuntimeError(
            "Qwen3 MoE grouped-query attention requires CUDA FlashAttention "
            "with native enable_gqa support; no math or repeat_interleave "
            f"fallback is allowed (torch={torch.__version__})"
        ) from error


class Qwen3MoeDecodeRuntime:
    def __init__(
        self,
        skeleton: ResidentQwen3Skeleton,
        expert_backend: (PagedFP8ExpertBackend | PagedInt4ExpertBackend | PackedBF16ExpertBackend),
    ):
        self.skeleton = skeleton
        self.backend = expert_backend
        self.cfg = skeleton.cfg
        self.device = skeleton.device
        self.dtype = skeleton.dtype
        self.layers = int(self.cfg["num_hidden_layers"])
        self.heads = int(self.cfg["num_attention_heads"])
        self.kv_heads = int(self.cfg["num_key_value_heads"])
        self.head_dim = int(self.cfg.get("head_dim") or self.cfg["hidden_size"] // self.heads)
        self.hidden = int(self.cfg["hidden_size"])
        self.top_k = int(self.cfg["num_experts_per_tok"])
        self.eps = float(self.cfg.get("rms_norm_eps") or 1e-6)
        self.theta = float(self.cfg.get("rope_theta") or 1e6)
        self.norm_topk = bool(self.cfg.get("norm_topk_prob", True))

    def new_cache(self, batch_size: int, capacity: int) -> StaticKVCache:
        layers = [
            LayerKV(
                key=torch.empty(
                    (
                        batch_size,
                        self.kv_heads,
                        capacity,
                        self.head_dim,
                    ),
                    device=self.device,
                    dtype=self.dtype,
                ),
                value=torch.empty(
                    (
                        batch_size,
                        self.kv_heads,
                        capacity,
                        self.head_dim,
                    ),
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
            for _ in range(self.layers)
        ]
        return StaticKVCache(
            layers=layers,
            batch_size=batch_size,
            capacity=capacity,
        )

    def new_fork_cache(
        self,
        parent: StaticKVCache,
        *,
        batch_size: int,
        capacity: int,
    ) -> ForkedKVCache:
        """Allocate only branch-local tail storage for an immutable B1 parent."""

        if not isinstance(parent, StaticKVCache):
            raise TypeError("COW parent must be a StaticKVCache")
        if parent.batch_size != 1 or parent.length <= 0:
            raise ValueError("COW parent must contain one non-empty committed request")
        if len(parent.layers) != self.layers:
            raise ValueError("COW parent layer geometry does not match this runtime")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("COW branch batch_size must be a positive integer")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("COW logical capacity must be an integer")
        delta_capacity = int(capacity) - int(parent.length)
        if delta_capacity <= 0:
            raise ValueError("COW logical capacity must leave at least one post-cut token")
        layers = [
            LayerKV(
                key=torch.empty(
                    (batch_size, self.kv_heads, delta_capacity, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                ),
                value=torch.empty(
                    (batch_size, self.kv_heads, delta_capacity, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
            for _ in range(self.layers)
        ]
        return ForkedKVCache(
            parent=parent,
            layers=layers,
            batch_size=batch_size,
            capacity=capacity,
            delta_capacity=delta_capacity,
        )

    def materialize_fork_branch(
        self,
        fork: ForkedKVCache,
        branch: int,
    ) -> StaticKVCache:
        """Commit one selected branch, copying one parent only at the terminal decision."""

        if not isinstance(fork, ForkedKVCache):
            raise TypeError("commit source must be a ForkedKVCache")
        if isinstance(branch, bool) or not isinstance(branch, int):
            raise TypeError("commit branch must be an integer")
        if branch < 0 or branch >= fork.batch_size:
            raise IndexError("commit branch is outside the fork panel")
        committed = self.new_cache(batch_size=1, capacity=fork.length)
        parent_length = int(fork.parent.length)
        delta_length = int(fork.delta_length)
        for parent_layer, fork_layer, committed_layer in zip(
            fork.parent.layers,
            fork.layers,
            committed.layers,
            strict=True,
        ):
            committed_layer.key[:, :, :parent_length].copy_(
                parent_layer.key[:, :, :parent_length]
            )
            committed_layer.value[:, :, :parent_length].copy_(
                parent_layer.value[:, :, :parent_length]
            )
            if delta_length:
                committed_layer.key[:, :, parent_length:].copy_(
                    fork_layer.key[branch : branch + 1, :, :delta_length]
                )
                committed_layer.value[:, :, parent_length:].copy_(
                    fork_layer.value[branch : branch + 1, :, :delta_length]
                )
        committed.length = fork.length
        return committed

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: StaticKVCache,
        capture_routes: bool = False,
        all_logits: bool = False,
        capture_hidden: bool = False,
        hidden_last_only: bool = True,
        hidden_capture_layers: Sequence[int] | None = None,
        return_final_hidden: bool = False,
        skip_lm_head: bool = False,
        ablate: tuple[int, str] | None = None,
        ablate_embed_direction: torch.Tensor | None = None,
        ablate_embed_alpha: float = 1.0,
    ) -> ForwardResult:
        """One prefill/decode step, with optional analysis taps.

        ``capture_hidden`` records the residual stream in HF ``output_hidden_states`` layout
        (``[embed_out, out_L0, ..., out_L{nL-1}]``). ``hidden_capture_layers`` instead selects
        exact residual boundaries during execution: boundary ``-1`` is the embedding output and
        boundary ``L`` is the output of decoder block ``L``. Unrequested boundaries never enter
        the clone/retention path. ``hidden_last_only`` keeps just the final sequence position,
        which is what a last-token probe reads and what keeps the capture from costing
        ``nL * B * T * d`` (1.6 GB on a 62-layer 6144-wide model at B=32/T=64).

        ``ablate=(layer, "attn"|"mlp")`` zeroes that component's WHOLE output contribution to
        the residual at every position — the streamed equivalent of a forward hook returning
        ``zeros_like(out)`` on ``layer.self_attn`` / ``layer.mlp``.

        ``ablate_embed_direction`` projects a unit direction out of the embedding rows before
        layer 0 (``h -> h - alpha (h.v) v``), the same edit an in-place
        ``embedding.weight -= (W v) vᵀ`` makes, without mutating resident weights.

        ``skip_lm_head`` returns no logits at all: for a 151936-row vocab the head is the single
        largest allocation in a forward, and a residual-stream measurement never reads it.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"expected [batch,tokens], got {tuple(input_ids.shape)}")
        batch, tokens = input_ids.shape
        if batch != cache.batch_size:
            raise ValueError("input batch does not match KV cache")
        start = cache.length
        end = start + tokens
        if end > cache.capacity:
            raise OverflowError("KV cache capacity exceeded")
        if start > 0 and tokens != 1:
            raise NotImplementedError("cached continuation currently accepts one token")
        capture_declared = bool(capture_hidden or hidden_capture_layers is not None)
        if hidden_capture_layers is None:
            requested_hidden_layers = (
                tuple(range(-1, self.layers)) if capture_hidden else ()
            )
            execution_time_pushdown = False
        else:
            requested_hidden_layers = tuple(
                sorted(set(int(layer) for layer in hidden_capture_layers))
            )
            execution_time_pushdown = True
        if any(
            layer < -1 or layer >= self.layers
            for layer in requested_hidden_layers
        ):
            raise ValueError(
                f"hidden capture layer must be in -1..{self.layers - 1}"
            )
        requested_hidden_set = frozenset(requested_hidden_layers)
        ablate_layer, ablate_kind = (-1, "")
        if ablate is not None:
            ablate_layer, ablate_kind = int(ablate[0]), str(ablate[1])
            if ablate_kind not in ("attn", "mlp"):
                raise ValueError(f"ablate kind must be 'attn' or 'mlp', got {ablate_kind!r}")
            if not 0 <= ablate_layer < self.layers:
                raise ValueError(f"ablate layer {ablate_layer} out of range 0..{self.layers - 1}")
        hidden = self.skeleton.embedding[input_ids]
        if ablate_embed_direction is not None:
            direction = ablate_embed_direction.to(device=hidden.device, dtype=torch.float32)
            direction = direction / (direction.norm() + 1e-9)
            # `embedding[input_ids]` is an advanced-index GATHER, i.e. already a fresh tensor —
            # the in-place subtract below can never reach the resident embedding weight.
            projected = hidden.float()
            projected -= ablate_embed_alpha * (projected @ direction).unsqueeze(-1) * direction
            hidden = projected.to(self.dtype)
        captured_hidden: list[torch.Tensor] = []
        captured_hidden_layers: list[int] = []

        def _tap(boundary: int, state: torch.Tensor) -> None:
            if boundary not in requested_hidden_set:
                return
            captured_hidden.append(
                self._clone_hidden_capture(
                    boundary,
                    state,
                    hidden_last_only=hidden_last_only,
                )
            )
            captured_hidden_layers.append(boundary)

        if requested_hidden_set:
            _tap(-1, hidden)
        positions = torch.arange(
            start,
            end,
            device=input_ids.device,
            dtype=torch.long,
        )
        cosine, sine = rope_tables(
            positions,
            self.head_dim,
            self.theta,
            self.dtype,
        )
        cosine = cosine[None, :, None, :]
        sine = sine[None, :, None, :]
        captured_routes: list[torch.Tensor] = []

        for layer_index, layer in enumerate(self.skeleton.layers):
            normalized = rms_norm(hidden, layer.input_norm, self.eps)
            query = (normalized @ layer.q_proj.t()).reshape(
                batch,
                tokens,
                self.heads,
                self.head_dim,
            )
            key = (normalized @ layer.k_proj.t()).reshape(
                batch,
                tokens,
                self.kv_heads,
                self.head_dim,
            )
            value = (normalized @ layer.v_proj.t()).reshape(
                batch,
                tokens,
                self.kv_heads,
                self.head_dim,
            )
            query = rms_norm(query, layer.q_norm, self.eps)
            key = rms_norm(key, layer.k_norm, self.eps)
            query = query * cosine + rotate_half(query) * sine
            key = key * cosine + rotate_half(key) * sine
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            layer_cache = cache.layers[layer_index]
            layer_cache.key[:, :, start:end].copy_(key)
            layer_cache.value[:, :, start:end].copy_(value)
            attention_key = layer_cache.key[:, :, :end]
            attention_value = layer_cache.value[:, :, :end]
            context = _native_sdpa_attention(
                query,
                attention_key,
                attention_value,
                dropout_p=0.0,
                is_causal=start == 0 and tokens > 1,
            )
            context = context.transpose(1, 2).reshape(
                batch,
                tokens,
                self.heads * self.head_dim,
            )
            attention_output = context @ layer.o_proj.t()
            if layer_index == ablate_layer and ablate_kind == "attn":
                attention_output = torch.zeros_like(attention_output)
            residual = hidden + attention_output
            moe_input = rms_norm(residual, layer.post_norm, self.eps)
            flat = moe_input.reshape(batch * tokens, self.hidden)
            router_logits = flat @ layer.router.t()
            probabilities = torch.softmax(router_logits.float(), dim=-1)
            top_weights, top_indices = probabilities.topk(self.top_k, dim=-1)
            if self.norm_topk:
                top_weights = top_weights / top_weights.sum(-1, keepdim=True)
            top_weights = top_weights.to(self.dtype)
            if capture_routes:
                captured_routes.append(top_indices.detach().cpu())
            if layer_index == ablate_layer and ablate_kind == "mlp":
                # Skip the expert gather entirely: the removed output is exactly zero, so
                # paging 8 experts in to multiply them by nothing would only cost bandwidth.
                hidden = residual
            else:
                if isinstance(self.backend, PagedFP8ExpertBackend):
                    moe_output = self.backend.moe(
                        layer_index,
                        flat,
                        top_indices,
                        top_weights,
                        prefill=start == 0,
                    )
                else:
                    moe_output = self.backend.moe(
                        layer_index,
                        flat,
                        top_indices,
                        top_weights,
                    )
                moe_output = moe_output.reshape(batch, tokens, self.hidden)
                hidden = residual + moe_output
            if requested_hidden_set:
                _tap(layer_index, hidden)

        cache.length = end
        normalized = rms_norm(hidden, self.skeleton.final_norm, self.eps)
        logits = None
        if not skip_lm_head:
            logits_source = normalized if all_logits else normalized[:, -1]
            logits = logits_source @ self.skeleton.lm_head.t()
        elements_per_boundary = int(
            batch * self.hidden if hidden_last_only else batch * tokens * self.hidden
        )
        bytes_per_boundary = elements_per_boundary * int(hidden.element_size())
        all_boundary_count = self.layers + 1
        retained_elements = elements_per_boundary * len(captured_hidden)
        capture_evidence = None
        if capture_declared:
            capture_evidence = HiddenCaptureEvidence(
                requested_boundaries=requested_hidden_layers,
                available_boundary_count=all_boundary_count,
                hidden_last_only=hidden_last_only,
                retained_tensors=len(captured_hidden),
                retained_elements=retained_elements,
                retained_bytes=bytes_per_boundary * len(captured_hidden),
                all_boundary_tape_elements=elements_per_boundary * all_boundary_count,
                all_boundary_tape_bytes=bytes_per_boundary * all_boundary_count,
                avoided_retained_bytes=(
                    bytes_per_boundary * (all_boundary_count - len(captured_hidden))
                ),
                execution_time_pushdown=execution_time_pushdown,
            )
        return ForwardResult(
            logits=logits,
            cache=cache,
            routes=captured_routes,
            hidden_states=captured_hidden,
            hidden_state_layers=tuple(captured_hidden_layers),
            hidden_capture_evidence=capture_evidence,
            final_hidden=(
                (normalized[:, -1] if hidden_last_only else normalized).detach()
                if return_final_hidden
                else None
            ),
        )

    @staticmethod
    def _clone_hidden_capture(
        _boundary: int,
        state: torch.Tensor,
        *,
        hidden_last_only: bool,
    ) -> torch.Tensor:
        """The sole residual capture allocation seam, kept observable for retention tests."""

        selected = state[:, -1] if hidden_last_only else state
        return selected.detach().clone()

    @torch.no_grad()
    def forward_decode_rows(
        self,
        input_token_ids: torch.Tensor,
        *,
        cache: StaticKVCache,
        physical_slots: Sequence[int],
        parent_lengths: Sequence[int],
        capture_evidence: bool = False,
    ) -> torch.Tensor | RaggedDecodeEvidence:
        """Decode one token for each independently sized physical K/V row.

        This is the native continuous-batch seam. ``physical_slots`` maps the dense logical
        request order to rows of one fixed ``StaticKVCache`` arena, while ``parent_lengths``
        gives the committed prefix length of each request. Every layer writes one provisional
        K/V position at that row's parent length, then consumes the compact prefixes with the
        segmented native-GQA kernel. The cache's scalar ``length`` is deliberately untouched;
        the serving arena owns the later per-request commit or abandon decision.

        Prefill is intentionally not accepted here. It remains on the established scalar B1
        path until a separate ragged/chunked-prefill contract is qualified. ``capture_evidence``
        retains logits and route indices on device for parity gates; its default is false so the
        serving result remains only the selected token tensor.
        """

        if not isinstance(capture_evidence, bool):
            raise TypeError("ragged decode capture_evidence must be a bool")

        if input_token_ids.ndim == 2 and int(input_token_ids.shape[1]) == 1:
            input_token_ids = input_token_ids[:, 0]
        if input_token_ids.ndim != 1 or not int(input_token_ids.shape[0]):
            raise ValueError("ragged decode input_token_ids must have shape [rows] or [rows,1]")
        if input_token_ids.dtype != torch.long:
            raise TypeError("ragged decode input_token_ids must use torch.long")
        if not _device_matches_runtime(input_token_ids.device, self.device):
            raise ValueError(
                "ragged decode input_token_ids must already reside on the runtime device"
            )

        def _indices(values: Sequence[int], name: str) -> tuple[int, ...]:
            normalized: list[int] = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"ragged decode {name} must contain strict integers")
                normalized.append(int(value))
            return tuple(normalized)

        slots = _indices(physical_slots, "physical_slots")
        lengths = _indices(parent_lengths, "parent_lengths")
        rows = int(input_token_ids.shape[0])
        if len(slots) != rows or len(lengths) != rows:
            raise ValueError("ragged decode slots and lengths must align with input rows")
        if len(set(slots)) != rows:
            raise ValueError("ragged decode physical slots must be unique")
        if cache.batch_size <= 0 or len(cache.layers) != self.layers:
            raise ValueError("ragged decode cache does not match the runtime layer geometry")
        if any(slot < 0 or slot >= cache.batch_size for slot in slots):
            raise IndexError("ragged decode physical slot is outside the K/V arena")
        if any(length <= 0 or length >= cache.capacity for length in lengths):
            raise ValueError(
                "ragged decode parent lengths must be positive and leave one tail position"
            )
        for layer_cache in cache.layers:
            expected = (cache.batch_size, self.kv_heads, cache.capacity, self.head_dim)
            if (
                tuple(layer_cache.key.shape) != expected
                or tuple(layer_cache.value.shape) != expected
            ):
                raise ValueError("ragged decode K/V tensor geometry drifted")
            if (
                layer_cache.key.device != input_token_ids.device
                or layer_cache.value.device != input_token_ids.device
                or layer_cache.key.dtype != self.dtype
                or layer_cache.value.dtype != self.dtype
            ):
                raise ValueError("ragged decode K/V tensors differ from the runtime device/dtype")

        slot_tensor = torch.tensor(slots, device=input_token_ids.device, dtype=torch.long)
        parent_tensor = torch.tensor(lengths, device=input_token_ids.device, dtype=torch.long)
        sequence_lengths = parent_tensor + 1
        max_sequence_length = max(lengths) + 1
        hidden = self.skeleton.embedding[input_token_ids]
        cosine, sine = rope_tables(
            parent_tensor,
            self.head_dim,
            self.theta,
            self.dtype,
        )
        cosine = cosine[:, None, :]
        sine = sine[:, None, :]
        captured_routes: list[torch.Tensor] = []

        for layer_index, layer in enumerate(self.skeleton.layers):
            normalized = rms_norm(hidden, layer.input_norm, self.eps)
            qkv_projection = layer.qkv_proj
            if qkv_projection is None:
                # Hand-built CPU fixtures predate the fused resident view. Production skeletons
                # always materialize it once at load time; never concatenate in a serving loop.
                qkv_projection = torch.cat(
                    (layer.q_proj, layer.k_proj, layer.v_proj),
                    dim=0,
                )
            qkv = normalized @ qkv_projection.t()
            query_values, key_values, value_values = qkv.split(
                (
                    self.heads * self.head_dim,
                    self.kv_heads * self.head_dim,
                    self.kv_heads * self.head_dim,
                ),
                dim=-1,
            )
            query = query_values.reshape(rows, self.heads, self.head_dim)
            key = key_values.reshape(rows, self.kv_heads, self.head_dim)
            value = value_values.reshape(rows, self.kv_heads, self.head_dim)
            query = rms_norm(query, layer.q_norm, self.eps)
            key = rms_norm(key, layer.k_norm, self.eps)
            query = query * cosine + rotate_half(query) * sine
            key = key * cosine + rotate_half(key) * sine

            layer_cache = cache.layers[layer_index]
            scatter_segmented_decode_kv(
                key,
                value,
                layer_cache.key,
                layer_cache.value,
                slot_tensor,
                parent_tensor,
            )
            context = segmented_gqa_decode(
                query,
                layer_cache.key,
                layer_cache.value,
                slot_tensor,
                sequence_lengths,
                max_sequence_length=max_sequence_length,
            )
            attention_output = context.reshape(rows, self.heads * self.head_dim) @ layer.o_proj.t()
            residual = hidden + attention_output
            moe_input = rms_norm(residual, layer.post_norm, self.eps)
            router_logits = moe_input @ layer.router.t()
            probabilities = torch.softmax(router_logits.float(), dim=-1)
            top_weights, top_indices = probabilities.topk(self.top_k, dim=-1)
            if self.norm_topk:
                top_weights = top_weights / top_weights.sum(-1, keepdim=True)
            top_weights = top_weights.to(self.dtype)
            if capture_evidence:
                captured_routes.append(top_indices.detach().clone())
            if isinstance(self.backend, PagedFP8ExpertBackend):
                moe_output = self.backend.moe(
                    layer_index,
                    moe_input,
                    top_indices,
                    top_weights,
                    prefill=False,
                )
            else:
                moe_output = self.backend.moe(
                    layer_index,
                    moe_input,
                    top_indices,
                    top_weights,
                )
            hidden = residual + moe_output.reshape(rows, self.hidden)

        normalized = rms_norm(hidden, self.skeleton.final_norm, self.eps)
        logits = normalized @ self.skeleton.lm_head.t()
        token_ids = torch.argmax(logits, dim=-1)
        if capture_evidence:
            return RaggedDecodeEvidence(
                token_ids=token_ids,
                logits=logits,
                routes=tuple(captured_routes),
            )
        return token_ids

    @torch.no_grad()
    def forward_cow_decode(
        self,
        input_token_ids: torch.Tensor,
        *,
        cache: ForkedKVCache,
        capture_routes: bool = False,
        return_final_hidden: bool = False,
        skip_lm_head: bool = False,
    ) -> ForwardResult:
        """Advance equal-horizon branches without materializing their shared parent.

        Projection and MoE work remain one fused branch panel. Each attention layer writes one
        compact branch-delta K/V row, then the COW Triton kernel reads the immutable parent and
        row-local delta directly in a single online-softmax traversal.
        """

        if input_token_ids.ndim == 2 and int(input_token_ids.shape[1]) == 1:
            input_token_ids = input_token_ids[:, 0]
        if input_token_ids.ndim != 1 or not int(input_token_ids.shape[0]):
            raise ValueError(
                "COW decode input_token_ids must have shape [branches] or [branches,1]"
            )
        if input_token_ids.dtype != torch.long:
            raise TypeError("COW decode input_token_ids must use torch.long")
        if not _device_matches_runtime(input_token_ids.device, self.device):
            raise ValueError("COW decode input_token_ids must already reside on the runtime device")
        if not isinstance(cache, ForkedKVCache):
            raise TypeError("COW decode requires a ForkedKVCache")
        rows = int(input_token_ids.shape[0])
        if rows != cache.batch_size:
            raise ValueError("COW decode rows do not match the fork batch")
        if cache.parent.batch_size != 1 or cache.parent.length <= 0:
            raise ValueError("COW decode requires one non-empty parent row")
        if cache.delta_length >= cache.delta_capacity:
            raise OverflowError("COW branch delta capacity exceeded")
        if len(cache.parent.layers) != self.layers or len(cache.layers) != self.layers:
            raise ValueError("COW cache layer geometry does not match the runtime")
        parent_length = int(cache.parent.length)
        branch_position = int(cache.delta_length)
        logical_position = parent_length + branch_position
        branch_slots = torch.arange(rows, device=input_token_ids.device, dtype=torch.long)
        branch_positions = torch.full(
            (rows,),
            branch_position,
            device=input_token_ids.device,
            dtype=torch.long,
        )
        branch_lengths = branch_positions + 1
        logical_positions = torch.full(
            (rows,),
            logical_position,
            device=input_token_ids.device,
            dtype=torch.long,
        )
        cosine, sine = rope_tables(
            logical_positions,
            self.head_dim,
            self.theta,
            self.dtype,
        )
        cosine = cosine[:, None, :]
        sine = sine[:, None, :]
        hidden = self.skeleton.embedding[input_token_ids]
        captured_routes: list[torch.Tensor] = []

        for layer_index, layer in enumerate(self.skeleton.layers):
            normalized = rms_norm(hidden, layer.input_norm, self.eps)
            qkv_projection = layer.qkv_proj
            if qkv_projection is None:
                qkv_projection = torch.cat((layer.q_proj, layer.k_proj, layer.v_proj), dim=0)
            qkv = normalized @ qkv_projection.t()
            query_values, key_values, value_values = qkv.split(
                (
                    self.heads * self.head_dim,
                    self.kv_heads * self.head_dim,
                    self.kv_heads * self.head_dim,
                ),
                dim=-1,
            )
            query = query_values.reshape(rows, self.heads, self.head_dim)
            key = key_values.reshape(rows, self.kv_heads, self.head_dim)
            value = value_values.reshape(rows, self.kv_heads, self.head_dim)
            query = rms_norm(query, layer.q_norm, self.eps)
            key = rms_norm(key, layer.k_norm, self.eps)
            query = query * cosine + rotate_half(query) * sine
            key = key * cosine + rotate_half(key) * sine

            parent_layer = cache.parent.layers[layer_index]
            branch_layer = cache.layers[layer_index]
            expected_parent = (1, self.kv_heads, cache.parent.capacity, self.head_dim)
            expected_branch = (
                rows,
                self.kv_heads,
                cache.delta_capacity,
                self.head_dim,
            )
            if (
                tuple(parent_layer.key.shape) != expected_parent
                or tuple(parent_layer.value.shape) != expected_parent
                or tuple(branch_layer.key.shape) != expected_branch
                or tuple(branch_layer.value.shape) != expected_branch
            ):
                raise ValueError("COW K/V tensor geometry drifted")
            scatter_segmented_decode_kv(
                key,
                value,
                branch_layer.key,
                branch_layer.value,
                branch_slots,
                branch_positions,
            )
            context = segmented_gqa_decode_cow(
                query,
                parent_layer.key,
                parent_layer.value,
                branch_layer.key,
                branch_layer.value,
                branch_lengths,
                parent_length=parent_length,
                max_branch_length=branch_position + 1,
            )
            attention_output = context.reshape(rows, self.heads * self.head_dim) @ layer.o_proj.t()
            residual = hidden + attention_output
            moe_input = rms_norm(residual, layer.post_norm, self.eps)
            router_logits = moe_input @ layer.router.t()
            probabilities = torch.softmax(router_logits.float(), dim=-1)
            top_weights, top_indices = probabilities.topk(self.top_k, dim=-1)
            if self.norm_topk:
                top_weights = top_weights / top_weights.sum(-1, keepdim=True)
            top_weights = top_weights.to(self.dtype)
            if capture_routes:
                captured_routes.append(top_indices.detach().cpu())
            if isinstance(self.backend, PagedFP8ExpertBackend):
                moe_output = self.backend.moe(
                    layer_index,
                    moe_input,
                    top_indices,
                    top_weights,
                    prefill=False,
                )
            else:
                moe_output = self.backend.moe(
                    layer_index,
                    moe_input,
                    top_indices,
                    top_weights,
                )
            hidden = residual + moe_output.reshape(rows, self.hidden)

        cache.delta_length += 1
        normalized = rms_norm(hidden, self.skeleton.final_norm, self.eps)
        logits = None if skip_lm_head else normalized @ self.skeleton.lm_head.t()
        return ForwardResult(
            logits=logits,
            cache=cache,
            routes=captured_routes,
            final_hidden=normalized.detach() if return_final_hidden else None,
        )


def _model_dir(model: str) -> Path:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return snapshot_dir(resolve_model(model)).resolve()


def resolve_qwen3_moe_store_dir(
    model_dir: Path,
    store_dir: str | Path | None = None,
    *,
    expert_codec: str = EXPERT_CODEC_FP8,
) -> Path:
    """Resolve an explicit store or the model-derived shared QStore location."""

    normalized = normalize_expert_codec(expert_codec)
    if store_dir is not None:
        return Path(store_dir).expanduser()
    environment_key = (
        "MRUN_QWEN3_MOE_STORE_DIR"
        if normalized == EXPERT_CODEC_FP8
        else "MRUN_QWEN3_MOE_W4_STORE_DIR"
    )
    configured = os.environ.get(environment_key)
    if configured:
        return Path(configured).expanduser()
    suffix = "fp8-paged-v1" if normalized == EXPERT_CODEC_FP8 else "w4-paged-v1"
    return stores_root() / f"{model_dir.name}-{suffix}"


def _validate_expert_store(
    model_dir: Path,
    store_dir: Path,
    *,
    expert_codec: str,
    verify_content: bool = False,
) -> dict[str, Any]:
    """Validate store geometry and lineage, optionally rehashing all source/store bytes.

    The default preflight is intentionally cheap enough for every engine open. It validates
    the manifest's self-consistent source identity, hashes the small config/index files, and
    checks every checkpoint/store filename and byte length. ``verify_content=True`` performs
    the publication-grade 90 GB-class rehash and should be used at promotion boundaries.
    """

    normalized = normalize_expert_codec(expert_codec)
    schema, codec, expected_data_file, _page_kind = _store_contract(normalized)
    model_dir = model_dir.expanduser().resolve()
    store_dir = store_dir.expanduser()
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("model_type") != "qwen3_moe":
        raise ValueError(f"expected qwen3_moe, found {cfg.get('model_type')!r}")
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != schema:
        raise RuntimeError("unsupported Qwen3 MoE expert-store schema")
    if manifest.get("codec") != codec:
        raise RuntimeError("unsupported Qwen3 MoE expert-store codec")

    layout: ExpertPageLayout | Int4ExpertPageLayout
    if normalized == EXPERT_CODEC_FP8:
        layout = ExpertPageLayout.create(
            int(cfg["hidden_size"]),
            int(cfg["moe_intermediate_size"]),
        )
    else:
        layout = Int4ExpertPageLayout.create(
            int(cfg["hidden_size"]),
            int(cfg["moe_intermediate_size"]),
        )
    expected_geometry = {
        "layers": int(cfg["num_hidden_layers"]),
        "num_experts": int(cfg["num_experts"]),
        "top_k": int(cfg["num_experts_per_tok"]),
        "hidden_size": int(cfg["hidden_size"]),
        "intermediate_size": int(cfg["moe_intermediate_size"]),
        "layout": layout.as_dict(),
    }
    actual_geometry = {key: manifest.get(key) for key in expected_geometry}
    if actual_geometry != expected_geometry:
        raise RuntimeError(
            f"Qwen3 MoE expert-store geometry mismatch: {actual_geometry} != {expected_geometry}"
        )

    data_name = manifest.get("data_file")
    if not isinstance(data_name, str) or Path(data_name).name != data_name:
        raise RuntimeError("expert-store data_file must be one local filename")
    if data_name != expected_data_file:
        raise RuntimeError(
            f"expert-store data_file {data_name!r} does not match codec contract "
            f"{expected_data_file!r}"
        )
    data_path = store_dir / data_name
    expected_bytes = (
        expected_geometry["layers"] * expected_geometry["num_experts"] * layout.page_stride
    )
    if not data_path.is_file() or data_path.stat().st_size != expected_bytes:
        raise RuntimeError("expert-store byte length does not match fixed page geometry")
    if int(manifest.get("data_bytes", -1)) != expected_bytes:
        raise RuntimeError("expert-store manifest data_bytes is inconsistent")

    source = validate_source_provenance(manifest.get("source"))
    index, shards = _source_records(model_dir)
    source_records = {str(record["name"]): record for record in source["safetensors"]}
    if sorted(source_records) != sorted({str(value) for value in index["weight_map"].values()}):
        raise RuntimeError("expert-store source shard set does not match the checkpoint index")
    for shard in shards:
        record = source_records[shard.name]
        if shard.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"expert-store source byte length mismatch: {shard.name}")
    small_files = (
        ("config", model_dir / "config.json"),
        ("index", model_dir / "model.safetensors.index.json"),
    )
    for key, path in small_files:
        record = source.get(key)
        if not isinstance(record, dict):
            raise RuntimeError(f"expert-store source record is missing {key}")
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"expert-store source identity mismatch: {path.name}")

    if verify_content:
        expected_source = build_source_provenance(
            model_dir,
            shards,
            model_name=str(source["model_name"]),
            hf_id=str(source["hf_id"]),
        )
        verify_source_provenance(source, expected_source)
        verify_derived_provenance(
            store_dir,
            manifest.get("derived"),
            expected_filenames=(data_name,),
        )
    manifest["validation"] = {
        "geometry_verified": True,
        "source_manifest_verified": True,
        "small_source_files_verified": True,
        "source_blob_bytes_verified": True,
        "content_hashes_verified": bool(verify_content),
        "identity_status": (
            "content-hashes-verified"
            if verify_content
            else "declared-content-plus-geometry-verified"
        ),
    }
    return manifest


def validate_fp8_expert_store(
    model_dir: Path,
    store_dir: Path,
    *,
    verify_content: bool = False,
) -> dict[str, Any]:
    return _validate_expert_store(
        model_dir,
        store_dir,
        expert_codec=EXPERT_CODEC_FP8,
        verify_content=verify_content,
    )


def validate_int4_expert_store(
    model_dir: Path,
    store_dir: Path,
    *,
    verify_content: bool = False,
) -> dict[str, Any]:
    return _validate_expert_store(
        model_dir,
        store_dir,
        expert_codec=EXPERT_CODEC_W4,
        verify_content=verify_content,
    )


def _load_skeleton(
    model_dir: Path,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[ResidentQwen3Skeleton, dict[str, Any]]:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    reader = TensorReader(model_dir)
    started = time.perf_counter()
    skeleton = ResidentQwen3Skeleton(
        reader,
        cfg,
        device=device,
        dtype=dtype,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return skeleton, {
        "load_s": time.perf_counter() - started,
        "device_bytes": skeleton.device_bytes,
    }


@torch.no_grad()
def greedy_tokens(
    runtime: Qwen3MoeDecodeRuntime,
    input_ids: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    cache = runtime.new_cache(
        batch_size=int(input_ids.shape[0]),
        capacity=int(input_ids.shape[1]) + steps,
    )
    first = runtime.forward(input_ids, cache=cache)
    token = first.logits.argmax(dim=-1)
    generated = [token]
    for _ in range(steps - 1):
        result = runtime.forward(token[:, None], cache=cache)
        token = result.logits.argmax(dim=-1)
        generated.append(token)
    return torch.stack(generated, dim=1)


class Qwen3MoeCudaEngine(BaseEngine):
    """Paged FP8 or symmetric-W4 QStore inference for Qwen3 MoE checkpoints on CUDA.

    Attention, routing, normalization, embeddings, and the language head remain resident
    in BF16. Routed expert pages are gathered from an explicit immutable codec store into
    a global CUDA LRU and consumed directly by two strided grouped Triton kernels.
    """

    backend = "qwen3-moe-cuda"
    arch = "qwen3_moe"
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
        compute_dtype: str = "bf16",
        expert_codec: str = EXPERT_CODEC_FP8,
        cache_mb: float | None = None,
        max_active_pages: int = DEFAULT_MAX_ACTIVE_PAGES,
        host_cache_mb: float | None = None,
        warm_host: bool | None = None,
        release_store_mapping_after_gather: bool | None = None,
        route_prefetch: bool | None = None,
        cache_policy: str | None = None,
        page_binding_policy: str | None = None,
        prefill_page_policy: str | None = None,
        route_reduction_policy: str | None = None,
        w4_arithmetic_policy: str | None = None,
        page_trace: bool | None = None,
        page_trace_max_events: int = PAGE_TRACE_HARD_MAX_EVENTS,
        page_trace_max_requests: int = PAGE_TRACE_HARD_MAX_REQUESTS,
        page_trace_max_manifest_bytes: int = PAGE_TRACE_HARD_MAX_MANIFEST_BYTES,
        build_store: bool = False,
        verify_store_content: bool = False,
        **_ignored: Any,
    ) -> None:
        if not device.startswith("cuda"):
            raise ValueError("qwen3-moe-cuda requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("qwen3-moe-cuda requires CUDA")
        if compute_dtype != "bf16":
            raise ValueError("qwen3-moe-cuda currently supports compute_dtype='bf16' only")
        if triton is None:
            raise RuntimeError("qwen3-moe-cuda requires Triton")
        if max_active_pages < 1:
            raise ValueError("max_active_pages must be positive")
        self.expert_codec = normalize_expert_codec(expert_codec)
        if self.expert_codec == EXPERT_CODEC_W4:
            self.w4_arithmetic_policy: str | None = normalize_w4_arithmetic_policy(
                w4_arithmetic_policy
            )
        elif w4_arithmetic_policy is not None:
            raise ValueError("w4_arithmetic_policy requires expert_codec='w4'")
        else:
            self.w4_arithmetic_policy = None

        candidate = Path(model_name).expanduser()
        if candidate.is_dir():
            spec = None
            lineage_name = source_model_name or candidate.name
            lineage_hf_id = source_hf_id
        else:
            spec = resolve_model(model_name)
            lineage_name = source_model_name or spec.name
            lineage_hf_id = source_hf_id or spec.hf_id
        self.model_dir = _model_dir(model_name)
        self.store_dir = resolve_qwen3_moe_store_dir(
            self.model_dir,
            store_dir,
            expert_codec=self.expert_codec,
        )
        manifest_path = self.store_dir / "manifest.json"
        if not manifest_path.exists():
            if not build_store:
                raise FileNotFoundError(
                    f"no Qwen3 MoE {self.expert_codec.upper()} expert store at "
                    f"{self.store_dir}. Build it with "
                    "`mrun qwen3-moe build-store --model MODEL --store-dir STORE "
                    f"--expert-codec {self.expert_codec} --output RESULT.json` or pass "
                    "build_store=True."
                )
            if lineage_hf_id is None:
                raise ValueError(
                    "building from an explicit model path requires source_hf_id provenance"
                )
            builder = (
                build_fp8_expert_store
                if self.expert_codec == EXPERT_CODEC_FP8
                else build_int4_expert_store
            )
            builder(
                self.model_dir,
                self.store_dir,
                model_name=lineage_name,
                hf_id=lineage_hf_id,
                revision=source_revision,
            )
        validator = (
            validate_fp8_expert_store
            if self.expert_codec == EXPERT_CODEC_FP8
            else validate_int4_expert_store
        )
        self.manifest = validator(
            self.model_dir,
            self.store_dir,
            verify_content=verify_store_content,
        )

        configured_cache = os.environ.get("MRUN_QWEN3_MOE_CACHE_MB")
        requested_cache = (
            float(cache_mb) if cache_mb is not None else float(configured_cache or DEFAULT_CACHE_MB)
        )
        if requested_cache <= 0:
            raise ValueError("cache_mb must be positive")
        self.device = device
        self.dtype = torch.bfloat16
        self.cache_mb = requested_cache
        self.max_active_pages = int(max_active_pages)
        self.skeleton, self.load_info = _load_skeleton(
            self.model_dir,
            device=device,
            dtype=self.dtype,
        )
        store_type = (
            PackedFP8ExpertStore if self.expert_codec == EXPERT_CODEC_FP8 else PackedInt4ExpertStore
        )
        self.store = store_type(self.store_dir)
        release_mapping_env = os.environ.get(
            "MRUN_QWEN3_MOE_RELEASE_MAPPING_AFTER_GATHER", ""
        ).strip().lower()
        self.release_store_mapping_after_gather = (
            bool(release_store_mapping_after_gather)
            if release_store_mapping_after_gather is not None
            else release_mapping_env not in {"", "0", "false"}
        )
        self.store.release_after_gather = self.release_store_mapping_after_gather
        # Optional host-RAM page tier between the CUDA LRU and NVMe. Warm it and the
        # cold-fill cliff (measured 78-83 s of random 4 KiB faults) collapses to one
        # sequential pass; misses then cost a host memcpy, never a disk fault.
        configured_host = os.environ.get("MRUN_QWEN3_MOE_HOST_CACHE_MB")
        requested_host = (
            float(host_cache_mb)
            if host_cache_mb is not None
            else float(configured_host)
            if configured_host
            else 0.0
        )
        pin_env = os.environ.get("MRUN_QWEN3_MOE_TIER_PIN", "auto").strip().lower()
        # auto: pin only small tiers — a 20-30 GB cudaHostAlloc spikes RSS during
        # allocation and tripped killed_ram twice (jobs 439a70735e70, a34a4678e757);
        # a pageable tier still eliminates disk faults (stage stays pinned for H2D).
        try_pin = requested_host <= 8000 if pin_env == "auto" else pin_env not in ("0", "false")
        self.host_tier = (
            HostPageTier(self.store, host_cache_mb=requested_host, try_pin=try_pin)
            if requested_host > 0
            else None
        )
        env_warm = os.environ.get("MRUN_QWEN3_MOE_WARM_HOST", "").strip()
        should_warm = warm_host if warm_host is not None else env_warm not in ("", "0")
        if self.host_tier is not None and should_warm:
            self.host_tier.warm()
        self.page_cache = ExpertPageCache(
            self.store,
            device=device,
            cache_mb=requested_cache,
            max_active_pages=self.max_active_pages,
            host_tier=self.host_tier,
            route_prefetch=route_prefetch,
            cache_policy=cache_policy,
            page_binding_policy=page_binding_policy,
            prefill_page_policy=prefill_page_policy,
            page_trace=page_trace,
            page_trace_max_events=page_trace_max_events,
            page_trace_max_requests=page_trace_max_requests,
            page_trace_max_manifest_bytes=page_trace_max_manifest_bytes,
        )
        if self.expert_codec == EXPERT_CODEC_W4:
            self.expert_backend: PagedFP8ExpertBackend = PagedInt4ExpertBackend(
                self.page_cache,
                dtype=self.dtype,
                route_reduction_policy=route_reduction_policy,
                w4_arithmetic_policy=self.w4_arithmetic_policy,
            )
        else:
            self.expert_backend = PagedFP8ExpertBackend(
                self.page_cache,
                dtype=self.dtype,
                route_reduction_policy=route_reduction_policy,
            )
        self.runtime = Qwen3MoeDecodeRuntime(
            self.skeleton,
            self.expert_backend,
        )

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cfg = self.skeleton.cfg
        self.name = str(model_name)
        self.n_layer = int(self.cfg["num_hidden_layers"])
        self.inter = int(self.cfg["moe_intermediate_size"])
        self.hidden = int(self.cfg["hidden_size"])
        self.working_set_mb = (self.skeleton.device_bytes + self.page_cache.device_bytes) / 1e6
        self.init_taps()

    def init_taps(self) -> None:
        """Reset every analysis-tap field to "no ablation" (see :meth:`ablation`).

        Public and idempotent on purpose: the single definition of this engine's tap state, so
        adding a tap cannot leave a construction path (or a test fixture that bypasses
        ``__init__`` to avoid the CUDA/Triton requirement) silently missing a field. Held on the
        engine rather than threaded through every call so measurement code written against the
        dense paged engine's `candidate_logits_batch(ids_list, cands)` runs here unchanged.
        """
        self.ablate_component: tuple[int, str] | None = None
        self.ablate_embed_direction: torch.Tensor | None = None
        self.ablate_embed_alpha: float = 1.0
        self.ablate_lm_head: bool = False
        self._last_selected_capture_report: dict[str, Any] | None = None

    def _require_runtime(self) -> Qwen3MoeDecodeRuntime:
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("Qwen3MoeCudaEngine is closed")
        return runtime

    @torch.no_grad()
    def prefill_statecut(
        self,
        input_ids: Sequence[int] | np.ndarray,
        *,
        retention_budget_bytes: int,
    ) -> tuple[Any, torch.Tensor]:
        """Compile one immutable B1 prefix for zero-parent-copy branch continuation."""

        from .qwen3_moe_statecut import Qwen3MoeKVStateCut

        return Qwen3MoeKVStateCut.prefill(
            self,
            input_ids,
            retention_budget_bytes=retention_budget_bytes,
        )

    @torch.no_grad()
    def logits(self, ids: np.ndarray) -> torch.Tensor:
        row = np.asarray(ids, dtype=np.int64)
        if row.ndim != 1 or not row.size:
            raise ValueError("ids must be a non-empty one-dimensional token array")
        input_ids = torch.as_tensor(row, device=self.device, dtype=torch.long)[None, :]
        runtime = self._require_runtime()
        cache = runtime.new_cache(batch_size=1, capacity=int(row.size))
        return (
            runtime.forward(
                input_ids,
                cache=cache,
                all_logits=True,
            )
            .logits[0]
            .cpu()
        )

    @torch.no_grad()
    def logits_batch(self, ids_list: list[np.ndarray]) -> list[torch.Tensor]:
        """Fuse equal-length rows without introducing unmasked padding tokens."""

        if not ids_list:
            return []
        rows = [np.asarray(ids, dtype=np.int64) for ids in ids_list]
        for row in rows:
            if row.ndim != 1 or not row.size:
                raise ValueError("each ids row must be non-empty and one-dimensional")
        outputs: list[torch.Tensor | None] = [None] * len(rows)
        by_length: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            by_length.setdefault(int(row.size), []).append(index)
        runtime = self._require_runtime()
        for indices in by_length.values():
            batch = np.stack([rows[index] for index in indices])
            input_ids = torch.as_tensor(batch, device=self.device, dtype=torch.long)
            cache = runtime.new_cache(
                batch_size=len(indices),
                capacity=int(input_ids.shape[1]),
            )
            logits = runtime.forward(
                input_ids,
                cache=cache,
                all_logits=True,
            ).logits.cpu()
            for batch_index, output_index in enumerate(indices):
                outputs[output_index] = logits[batch_index]
        if any(output is None for output in outputs):
            raise RuntimeError("batched logits did not populate every row")
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
        prompt: str | Sequence[int] | np.ndarray,
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
        """Generate each equal-length prompt group in one persistent-KV decode."""

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
            generated = self.generate_ids(
                rows,
                max_new_tokens=max_new_tokens,
            ).tolist()
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

    def reset_page_cache(self, *, clear_pages: bool = True) -> None:
        self.expert_backend.reset_stats(clear_pages=clear_pages)

    def cache_stats(self) -> dict[str, Any]:
        report = self.expert_backend.report()
        if self.host_tier is not None:
            report["host_tier"] = dict(self.host_tier.stats)
        if self.page_cache is not None and self.page_cache.route_prefetch:
            report["route_prefetch"] = dict(self.page_cache.prefetch_stats)
        return report

    @property
    def page_trace_sha256(self) -> str | None:
        return self.page_cache.page_trace_sha256

    def export_page_trace(self) -> dict[str, Any] | None:
        return self.page_cache.export_page_trace()

    def drain_page_trace(self) -> dict[str, Any] | None:
        return self.page_cache.drain_page_trace()

    def warm_host_tier(self) -> dict[str, Any]:
        if self.host_tier is None:
            raise RuntimeError("engine was opened without host_cache_mb")
        return self.host_tier.warm()

    def runtime_report(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "algorithm": (
                "resident BF16 skeleton + compact native-GQA static KV + route-first "
                f"{self.expert_codec.upper()} QStore pages + bounded CUDA cache + "
                "page-bound grouped Triton GEMM"
            ),
            "device": self.device,
            "compute_dtype": "bf16",
            "expert_codec": self.expert_codec,
            "w4_arithmetic_policy": self.w4_arithmetic_policy,
            "working_set_mb": self.working_set_mb,
            "skeleton": self.load_info,
            "attention": {
                "implementation": "torch-sdpa-native-gqa + triton-segmented-gqa-cow",
                "ordinary_cuda_backend_policy": "flash-attention-only",
                "statecut_cuda_backend_policy": "triton-online-softmax-over-parent-plus-delta",
                "query_heads": self.runtime.heads,
                "kv_heads": self.runtime.kv_heads,
                "physical_kv_head_expansion": False,
                "unsupported_native_gqa_policy": "fail-closed",
            },
            "statecut": {
                "schema": "mrun-qwen3-moe-kv-statecut-v1",
                "attention_abi": "qwen3-moe-segmented-gqa-cow-online-softmax-v1",
                "immutable_parent": True,
                "branch_parent_copy_bytes": 0,
                "branch_delta_storage": True,
                "commit_materializes_selected_branch_only": True,
                "continuation_horizon": "bounded multi-token greedy continuation per transaction",
            },
            "store": {
                "path": str(self.store_dir),
                "schema": self.manifest["schema_version"],
                "codec": self.manifest["codec"],
                "data_bytes": int(self.manifest["data_bytes"]),
                "validation": self.manifest["validation"],
                "release_mapping_after_gather": self.release_store_mapping_after_gather,
            },
            "expert_runtime": self.expert_backend.report(),
            "page_trace": self.page_cache.page_trace_status(),
            "selected_capture": self.selected_capture_report(),
            "claim_boundary": [
                "Inference and forward logits are implemented; backward and training are not.",
                "Generation is greedy and uses a persistent static KV cache per request.",
                "Equal-length rows fuse; variable-length rows are bucketed without padding.",
                "Warm-cache throughput depends on route locality and cache capacity.",
                "Per-gather mapping release bounds process RSS at the cost of reopening the store.",
            ],
        }

    def selected_capture_contract(self) -> dict[str, Any]:
        """Negotiate the bounded microscope ABI without branching on the backend name."""

        layers = int(getattr(self, "n_layer", 0))
        return {
            "schema": "mrun-selected-capture-capability-v1",
            "supported": True,
            "execution_time_pushdown": True,
            "unrequested_boundaries_cloned": False,
            "sites": ("embedding_output", "residual_output"),
            "layer_address_range": (-1, layers - 1),
            "token_selectors": ("last",),
            "returned_device": "cpu",
            "returned_dtype": "fp32",
            "batching": "equal-length-fused; ragged-length-bucketed",
            "writes": False,
            "full_sequence_selected_capture": False,
            "execution_receipt": "metadata-only; returned selected rows are caller-owned",
        }

    def selected_capture_report(self) -> dict[str, Any]:
        """Return the negotiated ABI plus an ownership-safe copy of the latest receipt."""

        last_execution = self._last_selected_capture_report
        return {
            **self.selected_capture_contract(),
            "last_execution": (
                dict(last_execution) if last_execution is not None else None
            ),
        }

    def warm_cache(
        self,
        prompts: Sequence[str | Sequence[int] | np.ndarray],
        *,
        max_new_tokens: int = 1,
    ) -> dict[str, Any]:
        self.generate_batch(prompts, max_new_tokens=max_new_tokens)
        return self.cache_stats()

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
            compact_fused_weights=True,
            grouped_moe=True,
            paged_experts=True,
            cache_telemetry=True,
            continuous_batching=True,
            fp8_execution=getattr(self, "expert_codec", EXPERT_CODEC_FP8) == EXPERT_CODEC_FP8,
            int4_execution=getattr(self, "expert_codec", EXPERT_CODEC_FP8) == EXPERT_CODEC_W4,
            route_first_moe=True,
            selected_capture=True,
            residual_tap=True,
            # No FUSED whole-residual batch tap: `hidden_states_batch` falls back to one forward
            # per row. The fused batch tap this engine does own is `hidden_last_batch` (last
            # position only), which is what keeps a 62-layer capture off the VRAM budget.
            residual_tap_batch=False,
        )

    def forward_acts(
        self,
        _ids: np.ndarray,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError("qwen3-moe-cuda does not expose dense MLP activation taps")

    # ------------------------------------------------------------------ residual-stream taps
    @contextmanager
    def ablation(
        self,
        *,
        component: tuple[int, str] | None | Any = KEEP,
        embed_direction: torch.Tensor | None | Any = KEEP,
        embed_alpha: float | Any = KEEP,
        lm_head: bool | None | Any = KEEP,
    ) -> Any:
        """Scope an ablation over every tap call made inside the block, then restore.

        ``component=(layer, "attn"|"mlp")`` removes that component's whole output;
        ``embed_direction`` projects a unit direction out of the embedding rows before layer 0;
        ``lm_head`` also removes that direction from the output head (which is what a TIED
        checkpoint's single shared tensor does implicitly), defaulting to the checkpoint's own
        tie flag. Any argument left unset keeps its current value, so a component ablation can
        be nested inside an embedding ablation without silently cancelling it; pass ``None``
        explicitly to clear one.
        """
        previous = (
            self.ablate_component,
            self.ablate_embed_direction,
            self.ablate_embed_alpha,
            self.ablate_lm_head,
        )
        if component is not KEEP:
            self.ablate_component = component
        if embed_direction is not KEEP:
            self.ablate_embed_direction = embed_direction
        if embed_alpha is not KEEP:
            self.ablate_embed_alpha = float(embed_alpha)
        if lm_head is not KEEP:
            tied = bool(getattr(self.skeleton, "tied_lm_head", False))
            self.ablate_lm_head = tied if lm_head is None else bool(lm_head)
        try:
            yield self
        finally:
            (
                self.ablate_component,
                self.ablate_embed_direction,
                self.ablate_embed_alpha,
                self.ablate_lm_head,
            ) = previous

    def _tap_kwargs(self) -> dict[str, Any]:
        return {
            "ablate": self.ablate_component,
            "ablate_embed_direction": self.ablate_embed_direction,
            "ablate_embed_alpha": self.ablate_embed_alpha,
        }

    @torch.no_grad()
    def hidden_states(self, ids: np.ndarray) -> list[torch.Tensor]:
        """Per-layer residual states in HF ``output_hidden_states`` layout: ``nL+1`` tensors
        ``[T, hidden]`` where ``[0]`` is the embedding output and ``[i]`` (i=1..nL) is the
        residual after decoder layer ``i-1``. As in HF (and ``PagedEngine.hidden_states``) only
        the LAST entry is final-RMSNormed; entries ``1..nL-1`` stay pre-norm, so a depth probe
        indexes the same tensor here as on the dense oracle."""
        row = np.asarray(ids, dtype=np.int64)
        if row.ndim != 1 or not row.size:
            raise ValueError("ids must be a non-empty one-dimensional token array")
        runtime = self._require_runtime()
        cache = runtime.new_cache(batch_size=1, capacity=int(row.size))
        result = runtime.forward(
            torch.as_tensor(row, device=self.device, dtype=torch.long)[None, :],
            cache=cache,
            capture_hidden=True,
            hidden_last_only=False,
            skip_lm_head=True,
            **self._tap_kwargs(),
        )
        states = [state[0].detach().to("cpu") for state in result.hidden_states]
        eps = float(self.cfg.get("rms_norm_eps") or 1e-6)
        weight = self.skeleton.final_norm.to(states[-1].device)
        states[-1] = rms_norm(states[-1].float(), weight.float(), eps)
        return states

    @torch.no_grad()
    def hidden_last_batch(self, ids_list: list[np.ndarray]) -> torch.Tensor:
        """Post-final-norm hidden state at each row's last token ``[B, d]``, WITHOUT ever
        building the ``[B, T, vocab]`` logits tensor (skip_lm_head). Rows are bucketed by length
        exactly as ``logits_batch`` does, so no unmasked padding token is ever attended to."""
        rows = [np.asarray(ids, dtype=np.int64) for ids in ids_list]
        for row in rows:
            if row.ndim != 1 or not row.size:
                raise ValueError("each ids row must be non-empty and one-dimensional")
        if not rows:
            return torch.empty((0, self.hidden), dtype=torch.float32)
        runtime = self._require_runtime()
        outputs: list[torch.Tensor | None] = [None] * len(rows)
        by_length: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            by_length.setdefault(int(row.size), []).append(index)
        for indices in by_length.values():
            batch = np.stack([rows[index] for index in indices])
            input_ids = torch.as_tensor(batch, device=self.device, dtype=torch.long)
            cache = runtime.new_cache(
                batch_size=len(indices),
                capacity=int(input_ids.shape[1]),
            )
            hidden = runtime.forward(
                input_ids,
                cache=cache,
                skip_lm_head=True,
                return_final_hidden=True,
                **self._tap_kwargs(),
            ).final_hidden
            assert hidden is not None
            hidden = hidden.float().cpu()
            for batch_index, output_index in enumerate(indices):
                outputs[output_index] = hidden[batch_index]
        if any(row is None for row in outputs):
            raise RuntimeError("batched hidden states did not populate every row")
        return torch.stack([row for row in outputs if row is not None])

    @torch.no_grad()
    def selected_hidden_last_batch(
        self,
        ids_list: list[np.ndarray],
        layers: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        """Capture selected last-position residual boundaries in fused length buckets.

        Layer ``-1`` is the token-embedding output; layer ``L`` is the residual after decoder
        block ``L``.  The final block is final-RMSNormed to match ``hidden_states`` and HF's
        ``output_hidden_states`` convention.  The runtime clones only one ``[B, hidden]`` row
        per layer while executing, so an all-layer temporal microscope avoids both the scalar
        per-prompt fallback and the ``layers * B * T * hidden`` full-sequence tape.

        The returned tensors are CPU FP32 ``[B, hidden]`` arrays in caller row order.  This is
        an observation surface only: it does not enable residual writes or change the engine's
        numerical contract.
        """

        wanted = tuple(sorted(set(int(layer) for layer in layers)))
        if not wanted:
            raise ValueError("selected hidden capture requires at least one layer")
        if any(layer < -1 or layer >= self.n_layer for layer in wanted):
            raise ValueError(
                f"selected hidden layer must be in -1..{self.n_layer - 1}"
            )
        rows = [np.asarray(ids, dtype=np.int64) for ids in ids_list]
        for row in rows:
            if row.ndim != 1 or not row.size:
                raise ValueError("each ids row must be non-empty and one-dimensional")
        if not rows:
            self._last_selected_capture_report = {
                "requested_boundaries": wanted,
                "logical_rows": 0,
                "physical_length_buckets": 0,
                "retained_tensors": 0,
                "retained_bytes": 0,
                "all_boundary_tape_bytes": 0,
                "avoided_retained_bytes": 0,
                "returned_selected_bytes": 0,
                "all_boundary_return_bytes": 0,
                "avoided_return_bytes": 0,
                "execution_time_pushdown": True,
            }
            return {
                layer: torch.empty((0, self.hidden), dtype=torch.float32)
                for layer in wanted
            }

        runtime = self._require_runtime()
        outputs: dict[int, list[torch.Tensor | None]] = {
            layer: [None] * len(rows) for layer in wanted
        }
        bucket_capture_evidence: list[HiddenCaptureEvidence] = []
        by_length: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            by_length.setdefault(int(row.size), []).append(index)
        for indices in by_length.values():
            input_ids = torch.as_tensor(
                np.stack([rows[index] for index in indices]),
                device=self.device,
                dtype=torch.long,
            )
            result = runtime.forward(
                input_ids,
                cache=runtime.new_cache(
                    batch_size=len(indices),
                    capacity=int(input_ids.shape[1]),
                ),
                capture_hidden=True,
                hidden_last_only=True,
                hidden_capture_layers=wanted,
                skip_lm_head=True,
                **self._tap_kwargs(),
            )
            if result.hidden_state_layers != wanted:
                raise RuntimeError(
                    "selected hidden capture returned a mismatched boundary map"
                )
            captured_by_layer = dict(
                zip(result.hidden_state_layers, result.hidden_states, strict=True)
            )
            for layer in wanted:
                captured = captured_by_layer[layer]
                if layer == self.n_layer - 1:
                    captured = rms_norm(
                        captured.float(),
                        self.skeleton.final_norm.float(),
                        float(self.cfg.get("rms_norm_eps") or 1e-6),
                    )
                captured = captured.detach().float().cpu()
                for batch_index, output_index in enumerate(indices):
                    outputs[layer][output_index] = captured[batch_index]

            evidence = result.hidden_capture_evidence
            if evidence is None or not evidence.execution_time_pushdown:
                raise RuntimeError("selected hidden capture did not prove runtime pushdown")
            bucket_capture_evidence.append(evidence)

        result_by_layer: dict[int, torch.Tensor] = {}
        for layer in wanted:
            values = outputs[layer]
            if any(value is None for value in values):
                raise RuntimeError("selected hidden capture did not populate every row")
            result_by_layer[layer] = torch.stack(
                [value for value in values if value is not None]
            )
        retained_bytes = sum(item.retained_bytes for item in bucket_capture_evidence)
        all_tape_bytes = sum(
            item.all_boundary_tape_bytes for item in bucket_capture_evidence
        )
        self._last_selected_capture_report = {
            "requested_boundaries": wanted,
            "logical_rows": len(rows),
            "physical_length_buckets": len(by_length),
            "retained_tensors": sum(
                item.retained_tensors for item in bucket_capture_evidence
            ),
            "retained_bytes": retained_bytes,
            "all_boundary_tape_bytes": all_tape_bytes,
            "avoided_retained_bytes": all_tape_bytes - retained_bytes,
            "returned_selected_bytes": (
                len(rows) * len(wanted) * self.hidden * torch.float32.itemsize
            ),
            "all_boundary_return_bytes": (
                len(rows) * (self.n_layer + 1) * self.hidden * torch.float32.itemsize
            ),
            "execution_time_pushdown": True,
        }
        self._last_selected_capture_report["avoided_return_bytes"] = (
            self._last_selected_capture_report["all_boundary_return_bytes"]
            - self._last_selected_capture_report["returned_selected_bytes"]
        )
        return result_by_layer

    @torch.no_grad()
    def lm_head_rows(self, token_ids: list[int] | np.ndarray) -> torch.Tensor:
        """Only the requested output-head rows ``[k, d]`` — never the whole 151936-row head."""
        index = torch.as_tensor(np.asarray(token_ids, dtype=np.int64), device=self.device)
        rows = self.skeleton.lm_head.index_select(0, index).float()
        direction = self.ablate_embed_direction
        if direction is not None and self.ablate_lm_head:
            unit = direction.to(device=rows.device, dtype=torch.float32)
            unit = unit / (unit.norm() + 1e-9)
            rows = rows - self.ablate_embed_alpha * (rows @ unit).unsqueeze(-1) * unit
        return rows

    @torch.no_grad()
    def selected_last_logits_batch(
        self,
        ids_list: list[np.ndarray],
        token_ids: list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        """``[B, len(token_ids)]`` last-position logits over a token SUBSET. Mirrors
        ``PagedEngine.selected_last_logits_batch``: one hidden pass, one head-row lookup."""
        hidden = self.hidden_last_batch(ids_list).float()
        weights = self.lm_head_rows(list(token_ids)).to(hidden.device).float()
        return (hidden @ weights.T).detach().cpu()

    @torch.no_grad()
    def candidate_logits_batch(
        self,
        ids_list: list[np.ndarray],
        candidate_token_ids: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Per-row candidate scores through one hidden pass and one union head lookup. Same
        signature and semantics as ``PagedEngine.candidate_logits_batch`` so forced-choice
        scoring code is engine-agnostic. The shared ``-logZ`` cancels inside a row's candidate
        set, so argmax and margins are exact without a full-vocab softmax."""
        union = tuple(
            dict.fromkeys(
                token for row_candidates in candidate_token_ids for token in row_candidates
            )
        )
        union_scores = self.selected_last_logits_batch(ids_list, union)
        offsets = {token: index for index, token in enumerate(union)}
        return [
            row_scores.index_select(
                0,
                torch.as_tensor(
                    [offsets[token] for token in row_candidates],
                    dtype=torch.long,
                    device=row_scores.device,
                ),
            )
            for row_scores, row_candidates in zip(
                union_scores,
                candidate_token_ids,
                strict=True,
            )
        ]

    def down_weight(self, _layer: int) -> torch.Tensor:
        raise NotImplementedError("Qwen3 MoE layers have per-expert down weights")

    def close(self) -> None:
        self.runtime = None
        self.expert_backend = None
        self.page_cache = None
        self.host_tier = None
        if self.store is not None:
            self.store.release()
        self.store = None
        self.skeleton = None
