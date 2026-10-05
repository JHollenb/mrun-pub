"""Content-bound q4 expert pages for the canonical classic Mixtral source schema.

This module deliberately does not produce a normal ``mlx-lm`` model directory.  It produces a
separate runtime artifact: all non-expert tensors remain in authenticated skeleton shards while
each routed expert's ``w1``/``w2``/``w3`` matrices live in one independently authenticated MLX
affine-q4 page.  The corresponding runtime is opt-in and lives in
``mrun.runtime.mlx_mixtral_expert_paged``; neither the resident MLX loader nor ``PlacementPlan``
implicitly selects this format.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mrun.engine.mlx_component import (
    _canonical_json_bytes,
    _fsync_directory,
    _hash_regular_file,
    _read_regular_json,
    _safe_model_slug,
    _sha256_bytes,
    _write_durable,
)

from .emitter import ComponentArtifact, open_component_artifact
from .errors import DecompilerError
from .mlx_native import _read_allocation, _validated_config

MIXTRAL_EXPERT_STORE_SCHEMA = "mrun-mlx-mixtral-expert-paged-q4-v1"
MIXTRAL_EXPERT_STORE_BUILDER_ABI = "mrun-canonical-mixtral-to-mlx-expert-pages-v2"
MIXTRAL_EXPERT_STORE_RUNTIME_ABI = "mrun-mlx-mixtral-sparse-block-q4-v1"
MIXTRAL_EXPERT_STORE_MAPPING_ABI = "mixtral-classic-w1-w2-w3-per-expert-v1"
MIXTRAL_EXPERT_STORE_CODEC = "mlx-affine-int4-g64-bf16-direct-canonical-source-v1"
MIXTRAL_EXPERT_STORE_BITS = 4
MIXTRAL_EXPERT_STORE_GROUP_SIZE = 64
MIXTRAL_EXPERT_STORE_MODE = "affine"
MIXTRAL_EXPERT_STORE_SKELETON_SHARD_BYTES = 256 * 1024**2
MIXTRAL_EXPERT_STORE_NUMERICAL_CONTRACT = (
    "mixtral-selected-softmax-renormalized-source-order-q4-approximate-v1"
)

_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>[0-9]+)\.block_sparse_moe\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>w1|w2|w3)\.weight$"
)
_SAFE_MEMBER = re.compile(r"^[A-Za-z0-9_.\-/]+$")
_SOURCE_DTYPES = frozenset({"BF16", "F16", "F32"})
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "status",
        "artifact_sha256",
        "build_key_sha256",
        "recipe",
        "source",
        "config",
        "topology",
        "skeleton",
        "expert_pages",
        "coverage",
        "execution_certified",
        "native_runtime_candidate",
        "production_runtime_eligible",
        "full_model_runtime",
        "performance_claim_valid",
    }
)
_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


class MixtralExpertStoreLoweringError(DecompilerError):
    """The canonical artifact is outside the explicit Mixtral expert-page support set."""

    code = "mixtral_expert_store_lowering_rejection"
    gate = "G14"


class MixtralExpertStoreArtifactError(DecompilerError):
    """A Mixtral expert-page artifact failed identity or shape verification."""

    code = "mixtral_expert_store_artifact_failure"
    gate = "G14"


@dataclass(frozen=True, slots=True)
class MixtralExpertStoreBuildRecord:
    path: Path
    artifact_sha256: str
    build_key_sha256: str
    source_artifact_id: str
    skeleton_tensor_bytes: int
    expert_page_count: int
    expert_store_tensor_bytes: int
    verified_reopen: bool = True
    execution_certified: bool = False
    production_runtime_eligible: bool = False
    schema_version: str = MIXTRAL_EXPERT_STORE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "artifact_sha256": self.artifact_sha256,
            "build_key_sha256": self.build_key_sha256,
            "source_artifact_id": self.source_artifact_id,
            "skeleton_tensor_bytes": self.skeleton_tensor_bytes,
            "expert_page_count": self.expert_page_count,
            "expert_store_tensor_bytes": self.expert_store_tensor_bytes,
            "verified_reopen": self.verified_reopen,
            "execution_certified": self.execution_certified,
            "production_runtime_eligible": self.production_runtime_eligible,
        }


def _build_lock(key: str) -> threading.Lock:
    with _BUILD_LOCKS_GUARD:
        return _BUILD_LOCKS.setdefault(key, threading.Lock())


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MixtralExpertStoreArtifactError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise MixtralExpertStoreArtifactError(f"{field} must be an array")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise MixtralExpertStoreArtifactError(f"{field} has an unexpected schema")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MixtralExpertStoreArtifactError(f"{field} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MixtralExpertStoreArtifactError(f"{field} must be a non-negative integer")
    return int(value)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return the same path identity tuple used by the guarded file readers."""

    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise MixtralExpertStoreArtifactError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _member(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE_MEMBER.fullmatch(value):
        raise MixtralExpertStoreArtifactError(f"{field} is not a safe artifact member")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MixtralExpertStoreArtifactError(f"{field} is not a portable relative path")
    return path.as_posix()


def _durable_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - exercised on non-Apple hosts
        raise MixtralExpertStoreLoweringError(
            "building the Mixtral expert store requires the optional MLX dependency"
        ) from exc
    return mx


def _tensor_bytes(shape: list[int] | tuple[int, ...], dtype: str) -> int:
    widths = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4}
    try:
        width = widths[dtype]
    except KeyError as exc:
        raise MixtralExpertStoreArtifactError(f"unsupported tensor dtype {dtype!r}") from exc
    return math.prod(int(value) for value in shape) * width


def _expert_coordinates(name: str) -> tuple[int, int, str] | None:
    match = _EXPERT_PATTERN.fullmatch(name)
    if match is None:
        return None
    return (
        int(match.group("layer")),
        int(match.group("expert")),
        str(match.group("projection")),
    )


def _expected_expert_shapes(hidden: int, intermediate: int) -> dict[str, tuple[int, int]]:
    return {
        "w1": (intermediate, hidden),
        "w2": (hidden, intermediate),
        "w3": (intermediate, hidden),
    }


def _quantized_shapes(rows: int, columns: int) -> dict[str, tuple[int, int]]:
    if columns % MIXTRAL_EXPERT_STORE_GROUP_SIZE:
        raise MixtralExpertStoreLoweringError(
            f"expert width {columns} is not divisible by q4 group size "
            f"{MIXTRAL_EXPERT_STORE_GROUP_SIZE}"
        )
    return {
        "weight": (rows, columns * MIXTRAL_EXPERT_STORE_BITS // 32),
        "scales": (rows, columns // MIXTRAL_EXPERT_STORE_GROUP_SIZE),
        "biases": (rows, columns // MIXTRAL_EXPERT_STORE_GROUP_SIZE),
    }


def _source_blob_records(artifact: ComponentArtifact) -> dict[str, Mapping[str, Any]]:
    records = artifact.manifest.get("allocations")
    if not isinstance(records, list):
        raise MixtralExpertStoreArtifactError("canonical allocation manifest is missing")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("allocation_id"), str):
            raise MixtralExpertStoreArtifactError("canonical allocation record is malformed")
        allocation_id = str(record["allocation_id"])
        if allocation_id in result:
            raise MixtralExpertStoreArtifactError("canonical allocation record is duplicated")
        result[allocation_id] = record
    return result


def _source_blob_sha256(record: Mapping[str, Any]) -> str:
    blob = record.get("blob")
    if not isinstance(blob, Mapping):
        raise MixtralExpertStoreArtifactError("canonical allocation blob record is malformed")
    return _sha256(blob.get("sha256"), "canonical source blob sha256")


def _quantize_matrix(tensor: torch.Tensor) -> tuple[dict[str, Any], float, float]:
    mx = _mx()
    if tensor.ndim != 2 or not bool(torch.isfinite(tensor.float()).all().item()):
        raise MixtralExpertStoreLoweringError("expert matrix must be finite and two-dimensional")
    rows, columns = (int(value) for value in tensor.shape)
    expected = _quantized_shapes(rows, columns)
    reference = mx.array(tensor.float().numpy()).astype(mx.float32)
    packed, scales, biases = mx.quantize(
        reference.astype(mx.bfloat16),
        group_size=MIXTRAL_EXPERT_STORE_GROUP_SIZE,
        bits=MIXTRAL_EXPERT_STORE_BITS,
        mode=MIXTRAL_EXPERT_STORE_MODE,
    )
    arrays = {"weight": packed, "scales": scales, "biases": biases}
    observed = {
        "weight": (tuple(int(value) for value in packed.shape), str(packed.dtype)),
        "scales": (tuple(int(value) for value in scales.shape), str(scales.dtype)),
        "biases": (tuple(int(value) for value in biases.shape), str(biases.dtype)),
    }
    wanted = {
        "weight": (expected["weight"], "mlx.core.uint32"),
        "scales": (expected["scales"], "mlx.core.bfloat16"),
        "biases": (expected["biases"], "mlx.core.bfloat16"),
    }
    if observed != wanted:
        raise MixtralExpertStoreArtifactError("MLX q4 quantizer returned unexpected geometry")
    restored = mx.dequantize(
        packed,
        scales,
        biases,
        group_size=MIXTRAL_EXPERT_STORE_GROUP_SIZE,
        bits=MIXTRAL_EXPERT_STORE_BITS,
        mode=MIXTRAL_EXPERT_STORE_MODE,
    ).astype(mx.float32)
    error = restored - reference
    maximum = mx.max(mx.abs(error))
    squared = mx.sum(mx.square(error))
    mx.eval(packed, scales, biases, maximum, squared)
    max_abs = float(maximum.item())
    sum_squared = float(squared.item())
    if not math.isfinite(max_abs) or not math.isfinite(sum_squared):
        raise MixtralExpertStoreArtifactError("q4 error evidence is non-finite")
    return arrays, max_abs, sum_squared


def _builder_recipe(
    artifact: ComponentArtifact,
    *,
    config_sha256: str,
    source_dtype: str,
) -> dict[str, Any]:
    builder_sha256, _size, _file_identity = _hash_regular_file(Path(__file__).resolve())
    try:
        quantizer_version = package_version("mlx")
    except PackageNotFoundError as exc:  # pragma: no cover - malformed MLX installation
        raise MixtralExpertStoreLoweringError(
            "the MLX quantizer version is unidentifiable"
        ) from exc
    return {
        "schema": MIXTRAL_EXPERT_STORE_SCHEMA,
        "builder_abi": MIXTRAL_EXPERT_STORE_BUILDER_ABI,
        "builder_sha256": builder_sha256,
        "runtime_abi": MIXTRAL_EXPERT_STORE_RUNTIME_ABI,
        "mapping_abi": MIXTRAL_EXPERT_STORE_MAPPING_ABI,
        "codec": MIXTRAL_EXPERT_STORE_CODEC,
        "bits": MIXTRAL_EXPERT_STORE_BITS,
        "group_size": MIXTRAL_EXPERT_STORE_GROUP_SIZE,
        "mode": MIXTRAL_EXPERT_STORE_MODE,
        "quantizer": "mlx.core.quantize",
        "quantizer_version": quantizer_version,
        "quantizer_input_dtype": "bfloat16",
        "source_dtype": source_dtype,
        "source_artifact_id": artifact.artifact_id,
        "source_manifest_sha256": artifact.manifest_sha256,
        "source_ir_fingerprint": artifact.ir_bundle.fingerprint,
        "effective_config_sha256": config_sha256,
        "skeleton_shard_bytes": MIXTRAL_EXPERT_STORE_SKELETON_SHARD_BYTES,
        "numerical_contract": MIXTRAL_EXPERT_STORE_NUMERICAL_CONTRACT,
    }


def _source_record(
    artifact: ComponentArtifact,
    *,
    tied: bool,
) -> dict[str, Any]:
    return {
        "artifact_schema": artifact.manifest["schema_version"],
        "artifact_id": artifact.artifact_id,
        "manifest_sha256": artifact.manifest_sha256,
        "source_fingerprint": artifact.source.fingerprint,
        "tensor_index_fingerprint": artifact.tensor_index.fingerprint,
        "ir_bundle_fingerprint": artifact.ir_bundle.fingerprint,
        "io_fingerprint": artifact.ir_bundle.io.fingerprint,
        "model_fingerprint": artifact.ir_bundle.model.fingerprint,
        "architecture_id": artifact.ir_bundle.model.architecture_id,
        "adapter_id": artifact.ir_bundle.model.adapter_id,
        "tied_lexical_allocation": tied,
        "direct_from_canonical_source": True,
        "intermediate_qstore": False,
    }


def _topology(config: Mapping[str, Any]) -> dict[str, int]:
    return {
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "num_local_experts": int(config["num_local_experts"]),
        "num_experts_per_token": int(config["num_experts_per_tok"]),
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
    }


def _parameter_record(
    allocation: Any,
    record: Mapping[str, Any],
    *,
    name: str | None = None,
) -> dict[str, Any]:
    tensor_name = allocation.source_tensor if name is None else name
    return {
        "name": tensor_name,
        "source_tensor": allocation.source_tensor,
        "dtype": allocation.stored_dtype,
        "shape": list(allocation.stored_shape),
        "logical_bytes": int(allocation.byte_length),
        "source_allocation_id": allocation.allocation_id,
        "source_blob_sha256": _source_blob_sha256(record),
    }


def _write_skeleton_shards(
    staging: Path,
    artifact: ComponentArtifact,
    allocations: list[Any],
    source_records: Mapping[str, Mapping[str, Any]],
    *,
    source_dtype: str,
) -> list[dict[str, Any]]:
    directory = staging / "skeleton"
    directory.mkdir()
    shards: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    parameters: list[dict[str, Any]] = []
    pending_bytes = 0
    shard_index = 0

    def flush() -> None:
        nonlocal tensors, parameters, pending_bytes, shard_index
        if not tensors:
            return
        shard_index += 1
        relative = f"skeleton/model-{shard_index:05d}.safetensors"
        path = staging / relative
        save_file(
            dict(sorted(tensors.items())),
            path,
            metadata={
                "mrun": json.dumps(
                    {
                        "format": "mlx",
                        "role": "resident-mixtral-skeleton",
                        "source_artifact": artifact.artifact_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            },
        )
        _durable_file(path)
        digest, size, _file_identity = _hash_regular_file(path)
        shards.append(
            {
                "filename": relative,
                "bytes": size,
                "sha256": digest,
                "tensor_bytes": sum(item["logical_bytes"] for item in parameters),
                "parameters": sorted(parameters, key=lambda item: item["name"]),
            }
        )
        tensors = {}
        parameters = []
        pending_bytes = 0

    for allocation in sorted(allocations, key=lambda item: item.source_tensor):
        if pending_bytes and (
            pending_bytes + allocation.byte_length > MIXTRAL_EXPERT_STORE_SKELETON_SHARD_BYTES
        ):
            flush()
        record = source_records.get(allocation.allocation_id)
        if not isinstance(record, Mapping):
            raise MixtralExpertStoreArtifactError("canonical allocation record is missing")
        tensor = _read_allocation(artifact, allocation, record)
        if allocation.source_tensor in tensors:
            raise MixtralExpertStoreLoweringError("skeleton tensor mapping is not one-to-one")
        tensors[allocation.source_tensor] = tensor
        parameters.append(_parameter_record(allocation, record))
        pending_bytes += int(allocation.byte_length)
    flush()
    if not shards:
        raise MixtralExpertStoreLoweringError("Mixtral resident skeleton cannot be empty")
    return shards


def _write_expert_pages(
    staging: Path,
    artifact: ComponentArtifact,
    expert_allocations: Mapping[tuple[int, int, str], Any],
    source_records: Mapping[str, Mapping[str, Any]],
    *,
    layers: int,
    experts: int,
) -> tuple[list[dict[str, Any]], float, float, int]:
    mx = _mx()
    directory = staging / "experts"
    directory.mkdir()
    pages: list[dict[str, Any]] = []
    maximum_error = 0.0
    total_squared_error = 0.0
    total_elements = 0
    for layer in range(layers):
        for expert in range(experts):
            arrays: dict[str, Any] = {}
            sources: list[dict[str, Any]] = []
            tensors: list[dict[str, Any]] = []
            page_maximum = 0.0
            page_squared = 0.0
            page_elements = 0
            for projection in ("w1", "w2", "w3"):
                allocation = expert_allocations[(layer, expert, projection)]
                record = source_records.get(allocation.allocation_id)
                if not isinstance(record, Mapping):
                    raise MixtralExpertStoreArtifactError(
                        "canonical expert allocation record is missing"
                    )
                source = _read_allocation(artifact, allocation, record)
                quantized, max_abs, squared = _quantize_matrix(source)
                page_maximum = max(page_maximum, max_abs)
                page_squared += squared
                page_elements += int(source.numel())
                sources.append(_parameter_record(allocation, record))
                for part in ("weight", "scales", "biases"):
                    name = f"{projection}.{part}"
                    value = quantized[part]
                    arrays[name] = value
                    dtype = "U32" if part == "weight" else "BF16"
                    shape = [int(item) for item in value.shape]
                    tensors.append(
                        {
                            "name": name,
                            "dtype": dtype,
                            "shape": shape,
                            "logical_bytes": _tensor_bytes(shape, dtype),
                        }
                    )
            relative = f"experts/layer-{layer:04d}-expert-{expert:04d}.safetensors"
            path = staging / relative
            mx.save_safetensors(
                path,
                dict(sorted(arrays.items())),
                metadata={
                    "mrun": json.dumps(
                        {
                            "codec": MIXTRAL_EXPERT_STORE_CODEC,
                            "expert": expert,
                            "format": "mlx",
                            "layer": layer,
                            "source_artifact": artifact.artifact_id,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                },
            )
            _durable_file(path)
            digest, size, _file_identity = _hash_regular_file(path)
            tensor_bytes = sum(item["logical_bytes"] for item in tensors)
            pages.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "filename": relative,
                    "bytes": size,
                    "sha256": digest,
                    "tensor_bytes": tensor_bytes,
                    "source_parameters": sorted(sources, key=lambda item: item["name"]),
                    "tensors": sorted(tensors, key=lambda item: item["name"]),
                    "quantization_error": {
                        "max_abs": page_maximum,
                        "rmse": math.sqrt(page_squared / page_elements),
                        "elements": page_elements,
                    },
                }
            )
            maximum_error = max(maximum_error, page_maximum)
            total_squared_error += page_squared
            total_elements += page_elements
    return pages, maximum_error, total_squared_error, total_elements


def _build_record(verified: VerifiedMixtralExpertStore) -> MixtralExpertStoreBuildRecord:
    return MixtralExpertStoreBuildRecord(
        path=verified.path,
        artifact_sha256=verified.artifact_sha256,
        build_key_sha256=verified.build_key_sha256,
        source_artifact_id=str(verified.source["artifact_id"]),
        skeleton_tensor_bytes=verified.skeleton_tensor_bytes,
        expert_page_count=len(verified.expert_pages),
        expert_store_tensor_bytes=verified.expert_store_tensor_bytes,
    )


def build_mixtral_mlx_expert_store(
    source_artifact: str | Path,
    output_root: str | Path,
) -> MixtralExpertStoreBuildRecord:
    """Split a verified classic Mixtral component into a resident skeleton and q4 pages."""

    artifact = open_component_artifact(source_artifact)
    config, architecture, tied = _validated_config(artifact)
    if architecture != "mixtral":
        raise MixtralExpertStoreLoweringError("expert paging requires canonical classic Mixtral")
    allocations = list(artifact.ir_bundle.physical_weights.allocations)
    dtypes = {allocation.stored_dtype for allocation in allocations}
    if len(dtypes) != 1 or not dtypes <= _SOURCE_DTYPES:
        raise MixtralExpertStoreLoweringError(
            "Mixtral expert paging requires one BF16, F16, or F32 source dtype"
        )
    source_dtype = next(iter(dtypes))
    topology = _topology(config)
    expected_shapes = _expected_expert_shapes(
        topology["hidden_size"], topology["intermediate_size"]
    )
    expert_allocations: dict[tuple[int, int, str], Any] = {}
    skeleton_allocations: list[Any] = []
    for allocation in allocations:
        coordinates = _expert_coordinates(allocation.source_tensor)
        if coordinates is None:
            skeleton_allocations.append(allocation)
            continue
        layer, expert, projection = coordinates
        if not 0 <= layer < topology["num_hidden_layers"]:
            raise MixtralExpertStoreLoweringError("expert layer is outside canonical topology")
        if not 0 <= expert < topology["num_local_experts"]:
            raise MixtralExpertStoreLoweringError("expert index is outside canonical topology")
        if tuple(allocation.stored_shape) != expected_shapes[projection]:
            raise MixtralExpertStoreLoweringError("expert matrix shape differs from topology")
        if coordinates in expert_allocations:
            raise MixtralExpertStoreLoweringError("expert matrix mapping is duplicated")
        _quantized_shapes(*expected_shapes[projection])
        expert_allocations[coordinates] = allocation
    expected_coordinates = {
        (layer, expert, projection)
        for layer in range(topology["num_hidden_layers"])
        for expert in range(topology["num_local_experts"])
        for projection in ("w1", "w2", "w3")
    }
    if set(expert_allocations) != expected_coordinates:
        raise MixtralExpertStoreLoweringError("expert allocation inventory is incomplete")
    router_names = {
        f"model.layers.{layer}.block_sparse_moe.gate.weight"
        for layer in range(topology["num_hidden_layers"])
    }
    if not router_names <= {item.source_tensor for item in skeleton_allocations}:
        raise MixtralExpertStoreLoweringError("resident skeleton does not contain every router")

    effective_config = json.loads(_canonical_json_bytes(config))
    effective_config["mrun_expert_paging"] = {
        "schema": MIXTRAL_EXPERT_STORE_SCHEMA,
        "runtime_abi": MIXTRAL_EXPERT_STORE_RUNTIME_ABI,
        "codec": MIXTRAL_EXPERT_STORE_CODEC,
        "expert_matrices_resident": False,
    }
    config_bytes = _canonical_json_bytes(effective_config)
    config_sha = _sha256_bytes(config_bytes)
    recipe = _builder_recipe(
        artifact,
        config_sha256=config_sha,
        source_dtype=source_dtype,
    )
    build_key = _sha256_bytes(_canonical_json_bytes(recipe))
    root = Path(output_root).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise MixtralExpertStoreArtifactError("expert-store output root must be a real directory")
    root = root.resolve()
    target = root / f"{_safe_model_slug(artifact.source.source_id)}-experts-q4-{build_key[:16]}"

    with _build_lock(build_key):
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=root))
        try:
            config_file_sha, config_file_bytes = _write_durable(
                staging / "config.json", config_bytes
            )
            source_records = _source_blob_records(artifact)
            skeleton_shards = _write_skeleton_shards(
                staging,
                artifact,
                skeleton_allocations,
                source_records,
                source_dtype=source_dtype,
            )
            pages, maximum_error, total_squared_error, total_elements = _write_expert_pages(
                staging,
                artifact,
                expert_allocations,
                source_records,
                layers=topology["num_hidden_layers"],
                experts=topology["num_local_experts"],
            )
            skeleton_tensor_bytes = sum(item["tensor_bytes"] for item in skeleton_shards)
            expert_store_tensor_bytes = sum(item["tensor_bytes"] for item in pages)
            source_expert_bytes = sum(
                allocation.byte_length for allocation in expert_allocations.values()
            )
            manifest: dict[str, Any] = {
                "schema": MIXTRAL_EXPERT_STORE_SCHEMA,
                "status": "native-expert-paged-approximate-unexecuted",
                "build_key_sha256": build_key,
                "recipe": recipe,
                "source": _source_record(artifact, tied=tied),
                "config": {
                    "filename": "config.json",
                    "bytes": config_file_bytes,
                    "file_sha256": config_file_sha,
                    "semantic_sha256": config_sha,
                },
                "topology": topology,
                "skeleton": {
                    "residency": "resident",
                    "contains_expert_matrices": False,
                    "tensor_bytes": skeleton_tensor_bytes,
                    "shards": skeleton_shards,
                },
                "expert_pages": pages,
                "coverage": {
                    "source_allocation_count": len(allocations),
                    "source_allocation_bytes": sum(item.byte_length for item in allocations),
                    "source_dtype": source_dtype,
                    "skeleton_parameter_count": len(skeleton_allocations),
                    "skeleton_tensor_bytes": skeleton_tensor_bytes,
                    "expert_source_parameter_count": len(expert_allocations),
                    "expert_source_tensor_bytes": source_expert_bytes,
                    "expert_page_count": len(pages),
                    "expert_store_tensor_bytes": expert_store_tensor_bytes,
                    "all_source_allocations_emitted_once": True,
                    "expert_matrices_excluded_from_skeleton": True,
                    "quantization_error": {
                        "max_abs": maximum_error,
                        "rmse": math.sqrt(total_squared_error / total_elements),
                        "elements": total_elements,
                    },
                },
                "execution_certified": False,
                "native_runtime_candidate": True,
                "production_runtime_eligible": False,
                "full_model_runtime": False,
                "performance_claim_valid": False,
            }
            manifest["artifact_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
            _write_durable(staging / "manifest.json", _canonical_json_bytes(manifest))
            _fsync_directory(staging / "skeleton")
            _fsync_directory(staging / "experts")
            _fsync_directory(staging)
            rebuilt = VerifiedMixtralExpertStore(staging)
            rebuilt.assert_unchanged()
            if target.exists():
                # A recipe/build key names the inputs and lowering contract, not the emitted
                # bytes.  A self-consistent manifest can therefore be rewritten while retaining
                # that key.  Reuse is safe only after deriving the candidate again from the
                # canonical source and comparing the complete content-bound artifact identity.
                existing = VerifiedMixtralExpertStore(target)
                existing.assert_unchanged()
                if (
                    existing.build_key_sha256 != build_key
                    or existing.artifact_sha256 != rebuilt.artifact_sha256
                ):
                    raise MixtralExpertStoreArtifactError(
                        "existing expert store differs from canonical-source rebuild"
                    )
                shutil.rmtree(staging)
                return _build_record(existing)
            try:
                os.replace(staging, target)
            except OSError as publish_error:
                # A different process may have published the same recipe after our existence
                # check.  Accept only the byte-identical winner; every other collision remains
                # a hard failure and the staging cleanup below is safe.
                if not target.exists():
                    raise
                winner = VerifiedMixtralExpertStore(target)
                winner.assert_unchanged()
                if (
                    winner.build_key_sha256 != build_key
                    or winner.artifact_sha256 != rebuilt.artifact_sha256
                ):
                    raise MixtralExpertStoreArtifactError(
                        "concurrent expert-store publisher produced different content"
                    ) from publish_error
                shutil.rmtree(staging)
                return _build_record(winner)
            _fsync_directory(root)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    return _build_record(VerifiedMixtralExpertStore(target))


def _validate_safetensors(
    path: Path,
    expected: Mapping[str, tuple[str, tuple[int, ...]]],
) -> None:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != set(expected):
                raise MixtralExpertStoreArtifactError(
                    f"safetensors inventory differs from manifest: {path}"
                )
            for name, (dtype, shape) in expected.items():
                view = handle.get_slice(name)
                if view.get_dtype() != dtype or tuple(view.get_shape()) != shape:
                    raise MixtralExpertStoreArtifactError(
                        f"safetensors shape/dtype differs from manifest: {name}"
                    )
    except MixtralExpertStoreArtifactError:
        raise
    except Exception as exc:
        raise MixtralExpertStoreArtifactError(f"cannot inspect safetensors member {path}") from exc


def _validate_finite_tensors(path: Path, names: set[str]) -> None:
    if not names:
        return
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in names:
                if not bool(torch.isfinite(handle.get_tensor(name).float()).all().item()):
                    raise MixtralExpertStoreArtifactError(
                        f"safetensors tensor contains non-finite values: {name}"
                    )
    except MixtralExpertStoreArtifactError:
        raise
    except Exception as exc:
        raise MixtralExpertStoreArtifactError(
            f"cannot inspect finite safetensors values in {path}"
        ) from exc


class VerifiedMixtralExpertStore:
    """Strict immutable view of a resident-skeleton plus low-bit expert-store artifact."""

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("verified expert-store views are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, path: str | Path):
        self._frozen = False
        self.path = Path(path).expanduser().absolute()
        if self.path.is_symlink() or not self.path.is_dir():
            raise MixtralExpertStoreArtifactError("expert-store path must be a real directory")
        self.path = self.path.resolve()
        self._directory_identity = _identity(self.path.lstat())
        try:
            manifest, manifest_identity, manifest_file_sha = _read_regular_json(
                self.path / "manifest.json"
            )
        except Exception as exc:
            raise MixtralExpertStoreArtifactError("cannot read expert-store manifest") from exc
        if set(manifest) != _TOP_LEVEL_FIELDS:
            raise MixtralExpertStoreArtifactError("expert-store manifest has an unexpected schema")
        if manifest_file_sha != _sha256_bytes(_canonical_json_bytes(manifest)):
            raise MixtralExpertStoreArtifactError("expert-store manifest is not canonical JSON")
        if manifest.get("schema") != MIXTRAL_EXPERT_STORE_SCHEMA:
            raise MixtralExpertStoreArtifactError("expert-store schema is unsupported")
        if (
            manifest.get("status") != "native-expert-paged-approximate-unexecuted"
            or manifest.get("execution_certified") is not False
            or manifest.get("native_runtime_candidate") is not True
            or manifest.get("production_runtime_eligible") is not False
            or manifest.get("full_model_runtime") is not False
            or manifest.get("performance_claim_valid") is not False
        ):
            raise MixtralExpertStoreArtifactError("expert-store claim boundary is invalid")
        artifact_sha = _sha256(manifest.get("artifact_sha256"), "artifact_sha256")
        unhashed = dict(manifest)
        unhashed.pop("artifact_sha256")
        if _sha256_bytes(_canonical_json_bytes(unhashed)) != artifact_sha:
            raise MixtralExpertStoreArtifactError("expert-store artifact identity is invalid")

        recipe = _require_dict(manifest.get("recipe"), "recipe")
        expected_recipe_fields = {
            "schema",
            "builder_abi",
            "builder_sha256",
            "runtime_abi",
            "mapping_abi",
            "codec",
            "bits",
            "group_size",
            "mode",
            "quantizer",
            "quantizer_version",
            "quantizer_input_dtype",
            "source_dtype",
            "source_artifact_id",
            "source_manifest_sha256",
            "source_ir_fingerprint",
            "effective_config_sha256",
            "skeleton_shard_bytes",
            "numerical_contract",
        }
        _require_exact_keys(recipe, expected_recipe_fields, "recipe")
        expected_recipe_values = {
            "schema": MIXTRAL_EXPERT_STORE_SCHEMA,
            "builder_abi": MIXTRAL_EXPERT_STORE_BUILDER_ABI,
            "runtime_abi": MIXTRAL_EXPERT_STORE_RUNTIME_ABI,
            "mapping_abi": MIXTRAL_EXPERT_STORE_MAPPING_ABI,
            "codec": MIXTRAL_EXPERT_STORE_CODEC,
            "bits": MIXTRAL_EXPERT_STORE_BITS,
            "group_size": MIXTRAL_EXPERT_STORE_GROUP_SIZE,
            "mode": MIXTRAL_EXPERT_STORE_MODE,
            "quantizer": "mlx.core.quantize",
            "quantizer_input_dtype": "bfloat16",
            "skeleton_shard_bytes": MIXTRAL_EXPERT_STORE_SKELETON_SHARD_BYTES,
            "numerical_contract": MIXTRAL_EXPERT_STORE_NUMERICAL_CONTRACT,
        }
        if any(recipe.get(key) != value for key, value in expected_recipe_values.items()):
            raise MixtralExpertStoreArtifactError("expert-store quantization/ABI recipe is invalid")
        if recipe.get("source_dtype") not in _SOURCE_DTYPES:
            raise MixtralExpertStoreArtifactError("expert-store source dtype is unsupported")
        quantizer_version = recipe.get("quantizer_version")
        if (
            not isinstance(quantizer_version, str)
            or not quantizer_version
            or quantizer_version.strip() != quantizer_version
        ):
            raise MixtralExpertStoreArtifactError("expert-store quantizer version is invalid")
        for field in (
            "builder_sha256",
            "source_artifact_id",
            "source_manifest_sha256",
            "source_ir_fingerprint",
            "effective_config_sha256",
        ):
            _sha256(recipe.get(field), f"recipe {field}")
        build_key = _sha256(manifest.get("build_key_sha256"), "build_key_sha256")
        if build_key != _sha256_bytes(_canonical_json_bytes(recipe)):
            raise MixtralExpertStoreArtifactError("expert-store build key is invalid")

        source = _require_dict(manifest.get("source"), "source")
        _require_exact_keys(
            source,
            {
                "artifact_schema",
                "artifact_id",
                "manifest_sha256",
                "source_fingerprint",
                "tensor_index_fingerprint",
                "ir_bundle_fingerprint",
                "io_fingerprint",
                "model_fingerprint",
                "architecture_id",
                "adapter_id",
                "tied_lexical_allocation",
                "direct_from_canonical_source",
                "intermediate_qstore",
            },
            "source",
        )
        for field in (
            "artifact_id",
            "manifest_sha256",
            "source_fingerprint",
            "tensor_index_fingerprint",
            "ir_bundle_fingerprint",
            "io_fingerprint",
            "model_fingerprint",
        ):
            _sha256(source.get(field), f"source {field}")
        if (
            source.get("artifact_id") != recipe.get("source_artifact_id")
            or source.get("manifest_sha256") != recipe.get("source_manifest_sha256")
            or source.get("ir_bundle_fingerprint") != recipe.get("source_ir_fingerprint")
            or source.get("architecture_id") != "mixtral-sparse-moe-causal-decoder"
            or source.get("adapter_id") != "mrun.hf.mixtral-sparse-moe"
            or type(source.get("tied_lexical_allocation")) is not bool
            or source.get("direct_from_canonical_source") is not True
            or source.get("intermediate_qstore") is not False
        ):
            raise MixtralExpertStoreArtifactError("expert-store source identity is inconsistent")

        identities: dict[str, tuple[int, ...]] = {"manifest.json": manifest_identity}
        file_records: dict[str, dict[str, Any]] = {}
        config_record = _require_dict(manifest.get("config"), "config")
        _require_exact_keys(
            config_record,
            {"filename", "bytes", "file_sha256", "semantic_sha256"},
            "config",
        )
        config_name = _member(config_record.get("filename"), "config filename")
        if config_name != "config.json":
            raise MixtralExpertStoreArtifactError("expert-store config filename is invalid")
        config_hash, config_size, config_identity = _hash_regular_file(
            self.path / config_name,
            expected_size=_positive_int(config_record.get("bytes"), "config bytes"),
            expected_sha256=_sha256(config_record.get("file_sha256"), "config file sha256"),
        )
        if config_hash != config_record.get("semantic_sha256"):
            raise MixtralExpertStoreArtifactError("expert-store config is not semantically bound")
        try:
            config, config_json_identity, config_file_sha = _read_regular_json(
                self.path / config_name
            )
        except Exception as exc:
            raise MixtralExpertStoreArtifactError("cannot read expert-store config") from exc
        if config_json_identity != config_identity or config_file_sha != config_hash:
            raise MixtralExpertStoreArtifactError(
                "expert-store config changed between verification and parsing"
            )
        if config_file_sha != _sha256_bytes(_canonical_json_bytes(config)):
            raise MixtralExpertStoreArtifactError("expert-store config is not canonical JSON")
        if config_hash != recipe.get("effective_config_sha256"):
            raise MixtralExpertStoreArtifactError("expert-store config differs from its recipe")
        identities[config_name] = config_identity
        file_records[config_name] = dict(config_record)

        topology = _require_dict(manifest.get("topology"), "topology")
        _require_exact_keys(
            topology,
            {
                "num_hidden_layers",
                "num_local_experts",
                "num_experts_per_token",
                "hidden_size",
                "intermediate_size",
            },
            "topology",
        )
        topology_values = {
            key: _positive_int(value, f"topology {key}") for key, value in topology.items()
        }
        if not 0 < topology_values["num_experts_per_token"] < topology_values["num_local_experts"]:
            raise MixtralExpertStoreArtifactError("expert-store top-k is outside expert domain")
        page_count = topology_values["num_hidden_layers"] * topology_values["num_local_experts"]
        if page_count > 65_536:
            raise MixtralExpertStoreArtifactError(
                "expert-store topology exceeds bounded page inventory"
            )
        config_expected = {
            "model_type": "mixtral",
            "num_hidden_layers": topology_values["num_hidden_layers"],
            "num_local_experts": topology_values["num_local_experts"],
            "num_experts_per_tok": topology_values["num_experts_per_token"],
            "hidden_size": topology_values["hidden_size"],
            "intermediate_size": topology_values["intermediate_size"],
            "hidden_act": "silu",
        }
        if any(config.get(key) != value for key, value in config_expected.items()):
            raise MixtralExpertStoreArtifactError("expert-store config differs from topology")
        paging_config = config.get("mrun_expert_paging")
        if paging_config != {
            "schema": MIXTRAL_EXPERT_STORE_SCHEMA,
            "runtime_abi": MIXTRAL_EXPERT_STORE_RUNTIME_ABI,
            "codec": MIXTRAL_EXPERT_STORE_CODEC,
            "expert_matrices_resident": False,
        }:
            raise MixtralExpertStoreArtifactError("expert-store config paging contract is invalid")

        expected_expert_shapes = _expected_expert_shapes(
            topology_values["hidden_size"], topology_values["intermediate_size"]
        )
        skeleton = _require_dict(manifest.get("skeleton"), "skeleton")
        _require_exact_keys(
            skeleton,
            {"residency", "contains_expert_matrices", "tensor_bytes", "shards"},
            "skeleton",
        )
        if (
            skeleton.get("residency") != "resident"
            or skeleton.get("contains_expert_matrices") is not False
        ):
            raise MixtralExpertStoreArtifactError("resident skeleton contract is invalid")
        skeleton_records = _require_list(skeleton.get("shards"), "skeleton shards")
        if not skeleton_records:
            raise MixtralExpertStoreArtifactError("resident skeleton has no shards")
        skeleton_names: set[str] = set()
        source_allocation_ids: set[str] = set()
        skeleton_tensor_bytes = 0
        normalized_skeleton: list[dict[str, Any]] = []
        required_routers = {
            f"model.layers.{layer}.block_sparse_moe.gate.weight"
            for layer in range(topology_values["num_hidden_layers"])
        }
        for raw_shard in skeleton_records:
            shard = _require_dict(raw_shard, "skeleton shard")
            _require_exact_keys(
                shard,
                {"filename", "bytes", "sha256", "tensor_bytes", "parameters"},
                "skeleton shard",
            )
            filename = _member(shard.get("filename"), "skeleton shard filename")
            if not filename.startswith("skeleton/") or filename in file_records:
                raise MixtralExpertStoreArtifactError("skeleton shard filename is invalid")
            parameters = _require_list(shard.get("parameters"), "skeleton parameters")
            expected_tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
            parameter_bytes = 0
            for raw_parameter in parameters:
                parameter = _require_dict(raw_parameter, "skeleton parameter")
                _require_exact_keys(
                    parameter,
                    {
                        "name",
                        "source_tensor",
                        "dtype",
                        "shape",
                        "logical_bytes",
                        "source_allocation_id",
                        "source_blob_sha256",
                    },
                    "skeleton parameter",
                )
                name = str(parameter.get("name"))
                if (
                    name != parameter.get("source_tensor")
                    or _expert_coordinates(name) is not None
                    or name in skeleton_names
                ):
                    raise MixtralExpertStoreArtifactError(
                        "expert matrix leaked into resident skeleton or name is duplicated"
                    )
                dtype = str(parameter.get("dtype"))
                if dtype != recipe.get("source_dtype"):
                    raise MixtralExpertStoreArtifactError("skeleton source dtype is inconsistent")
                shape_raw = parameter.get("shape")
                if (
                    not isinstance(shape_raw, list)
                    or not shape_raw
                    or any(
                        isinstance(value, bool) or not isinstance(value, int) or value <= 0
                        for value in shape_raw
                    )
                ):
                    raise MixtralExpertStoreArtifactError("skeleton tensor shape is invalid")
                shape = tuple(int(value) for value in shape_raw)
                if name in required_routers and shape != (
                    topology_values["num_local_experts"],
                    topology_values["hidden_size"],
                ):
                    raise MixtralExpertStoreArtifactError(
                        "resident router shape differs from topology"
                    )
                logical_bytes = _positive_int(
                    parameter.get("logical_bytes"), "skeleton logical bytes"
                )
                if logical_bytes != _tensor_bytes(shape, dtype):
                    raise MixtralExpertStoreArtifactError(
                        "skeleton tensor byte accounting is invalid"
                    )
                allocation_id = str(parameter.get("source_allocation_id"))
                _sha256(parameter.get("source_blob_sha256"), "skeleton source blob sha256")
                if allocation_id in source_allocation_ids:
                    raise MixtralExpertStoreArtifactError(
                        "source allocation appears more than once in expert-store artifact"
                    )
                skeleton_names.add(name)
                source_allocation_ids.add(allocation_id)
                parameter_bytes += logical_bytes
                expected_tensors[name] = (dtype, shape)
            if parameter_bytes != _positive_int(
                shard.get("tensor_bytes"), "skeleton shard tensor bytes"
            ):
                raise MixtralExpertStoreArtifactError("skeleton shard byte accounting is invalid")
            digest, _size, file_identity = _hash_regular_file(
                self.path / filename,
                expected_size=_positive_int(shard.get("bytes"), "skeleton shard bytes"),
                expected_sha256=_sha256(shard.get("sha256"), "skeleton shard sha256"),
            )
            _validate_safetensors(self.path / filename, expected_tensors)
            _validate_finite_tensors(
                self.path / filename,
                required_routers.intersection(expected_tensors),
            )
            _digest_after, _size_after, identity_after = _hash_regular_file(
                self.path / filename,
                expected_size=int(shard["bytes"]),
                expected_sha256=str(shard["sha256"]),
            )
            if identity_after != file_identity:
                raise MixtralExpertStoreArtifactError(
                    "skeleton shard changed during value verification"
                )
            identities[filename] = identity_after
            file_records[filename] = {"bytes": shard["bytes"], "sha256": digest}
            skeleton_tensor_bytes += parameter_bytes
            normalized_skeleton.append(dict(shard))
        if skeleton_tensor_bytes != _positive_int(
            skeleton.get("tensor_bytes"), "skeleton tensor bytes"
        ):
            raise MixtralExpertStoreArtifactError("skeleton total byte accounting is invalid")
        if not required_routers <= skeleton_names:
            raise MixtralExpertStoreArtifactError("resident skeleton is missing a router matrix")

        page_records = _require_list(manifest.get("expert_pages"), "expert pages")
        if len(page_records) != page_count:
            raise MixtralExpertStoreArtifactError(
                "expert page inventory differs from bounded topology"
            )
        expected_coordinates = {
            (layer, expert)
            for layer in range(topology_values["num_hidden_layers"])
            for expert in range(topology_values["num_local_experts"])
        }
        pages_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        expert_store_tensor_bytes = 0
        expert_source_tensor_bytes = 0
        page_error_maximum = 0.0
        page_error_squared = 0.0
        page_error_elements = 0
        for raw_page in page_records:
            page = _require_dict(raw_page, "expert page")
            _require_exact_keys(
                page,
                {
                    "layer",
                    "expert",
                    "filename",
                    "bytes",
                    "sha256",
                    "tensor_bytes",
                    "source_parameters",
                    "tensors",
                    "quantization_error",
                },
                "expert page",
            )
            layer = _nonnegative_int(page.get("layer"), "expert page layer")
            expert = _nonnegative_int(page.get("expert"), "expert page expert")
            key = (layer, expert)
            if key not in expected_coordinates or key in pages_by_key:
                raise MixtralExpertStoreArtifactError("expert page coordinate is invalid")
            filename = _member(page.get("filename"), "expert page filename")
            expected_filename = f"experts/layer-{layer:04d}-expert-{expert:04d}.safetensors"
            if filename != expected_filename or filename in file_records:
                raise MixtralExpertStoreArtifactError("expert page filename is invalid")
            source_parameters = _require_list(
                page.get("source_parameters"), "expert source parameters"
            )
            expected_source_names = {
                f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}.weight"
                for projection in ("w1", "w2", "w3")
            }
            observed_source_names: set[str] = set()
            for raw_parameter in source_parameters:
                parameter = _require_dict(raw_parameter, "expert source parameter")
                _require_exact_keys(
                    parameter,
                    {
                        "name",
                        "source_tensor",
                        "dtype",
                        "shape",
                        "logical_bytes",
                        "source_allocation_id",
                        "source_blob_sha256",
                    },
                    "expert source parameter",
                )
                name = str(parameter.get("name"))
                coordinates = _expert_coordinates(name)
                if name != parameter.get("source_tensor") or coordinates is None:
                    raise MixtralExpertStoreArtifactError("expert source parameter name is invalid")
                observed_layer, observed_expert, projection = coordinates
                shape = tuple(parameter.get("shape", ()))
                if (
                    observed_layer != layer
                    or observed_expert != expert
                    or shape != expected_expert_shapes[projection]
                    or parameter.get("dtype") != recipe.get("source_dtype")
                    or parameter.get("logical_bytes")
                    != _tensor_bytes(shape, str(parameter.get("dtype")))
                ):
                    raise MixtralExpertStoreArtifactError(
                        "expert source parameter shape/dtype is invalid"
                    )
                allocation_id = str(parameter.get("source_allocation_id"))
                _sha256(parameter.get("source_blob_sha256"), "expert source blob sha256")
                if allocation_id in source_allocation_ids:
                    raise MixtralExpertStoreArtifactError(
                        "source allocation appears more than once in expert-store artifact"
                    )
                source_allocation_ids.add(allocation_id)
                observed_source_names.add(name)
                expert_source_tensor_bytes += int(parameter["logical_bytes"])
            if observed_source_names != expected_source_names:
                raise MixtralExpertStoreArtifactError("expert source parameter set is incomplete")

            tensor_records = _require_list(page.get("tensors"), "expert page tensors")
            expected_tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
            page_tensor_bytes = 0
            for projection, source_shape in expected_expert_shapes.items():
                quantized_shapes = _quantized_shapes(*source_shape)
                for part, shape in quantized_shapes.items():
                    expected_tensors[f"{projection}.{part}"] = (
                        "U32" if part == "weight" else "BF16",
                        shape,
                    )
            observed_tensor_names: set[str] = set()
            for raw_tensor in tensor_records:
                tensor = _require_dict(raw_tensor, "expert page tensor")
                _require_exact_keys(
                    tensor, {"name", "dtype", "shape", "logical_bytes"}, "expert page tensor"
                )
                name = str(tensor.get("name"))
                dtype = str(tensor.get("dtype"))
                shape = tuple(tensor.get("shape", ()))
                if name not in expected_tensors or (dtype, shape) != expected_tensors[name]:
                    raise MixtralExpertStoreArtifactError(
                        "expert page tensor has invalid shape/dtype/quantization geometry"
                    )
                logical_bytes = _positive_int(
                    tensor.get("logical_bytes"), "expert page tensor bytes"
                )
                if logical_bytes != _tensor_bytes(shape, dtype) or name in observed_tensor_names:
                    raise MixtralExpertStoreArtifactError(
                        "expert page tensor byte accounting or identity is invalid"
                    )
                page_tensor_bytes += logical_bytes
                observed_tensor_names.add(name)
            if observed_tensor_names != set(expected_tensors):
                raise MixtralExpertStoreArtifactError("expert page tensor inventory is incomplete")
            if page_tensor_bytes != _positive_int(
                page.get("tensor_bytes"), "expert page tensor bytes"
            ):
                raise MixtralExpertStoreArtifactError("expert page byte accounting is invalid")
            error = _require_dict(page.get("quantization_error"), "quantization error")
            _require_exact_keys(error, {"max_abs", "rmse", "elements"}, "quantization error")
            if (
                type(error.get("max_abs")) not in {int, float}
                or type(error.get("rmse")) not in {int, float}
                or not math.isfinite(float(error["max_abs"]))
                or not math.isfinite(float(error["rmse"]))
                or float(error["max_abs"]) < 0
                or float(error["rmse"]) < 0
                or _positive_int(error.get("elements"), "quantization error elements")
                != sum(math.prod(shape) for shape in expected_expert_shapes.values())
            ):
                raise MixtralExpertStoreArtifactError("quantization error evidence is invalid")
            page_error_maximum = max(page_error_maximum, float(error["max_abs"]))
            page_error_elements += int(error["elements"])
            page_error_squared += float(error["rmse"]) ** 2 * int(error["elements"])
            digest, _size, file_identity = _hash_regular_file(
                self.path / filename,
                expected_size=_positive_int(page.get("bytes"), "expert page file bytes"),
                expected_sha256=_sha256(page.get("sha256"), "expert page sha256"),
            )
            _validate_safetensors(self.path / filename, expected_tensors)
            _validate_finite_tensors(
                self.path / filename,
                {
                    name
                    for name in expected_tensors
                    if name.endswith(".scales") or name.endswith(".biases")
                },
            )
            _digest_after, _size_after, identity_after = _hash_regular_file(
                self.path / filename,
                expected_size=int(page["bytes"]),
                expected_sha256=str(page["sha256"]),
            )
            if identity_after != file_identity:
                raise MixtralExpertStoreArtifactError(
                    "expert page changed during value verification"
                )
            identities[filename] = identity_after
            file_records[filename] = {"bytes": page["bytes"], "sha256": digest}
            pages_by_key[key] = dict(page)
            expert_store_tensor_bytes += page_tensor_bytes
        if set(pages_by_key) != expected_coordinates:
            raise MixtralExpertStoreArtifactError("expert page coordinate inventory is incomplete")

        expected_files = {"manifest.json", *file_records}
        observed_files: set[str] = set()
        observed_directories: set[str] = set()
        for member in self.path.rglob("*"):
            relative = member.relative_to(self.path).as_posix()
            if member.is_symlink():
                raise MixtralExpertStoreArtifactError("expert-store inventory contains a symlink")
            if member.is_dir():
                observed_directories.add(relative)
            elif member.is_file():
                observed_files.add(relative)
            else:
                raise MixtralExpertStoreArtifactError(
                    "expert-store inventory contains a non-regular member"
                )
        if observed_files != expected_files or observed_directories != {"experts", "skeleton"}:
            raise MixtralExpertStoreArtifactError("expert-store file inventory is not exact")

        coverage = _require_dict(manifest.get("coverage"), "coverage")
        _require_exact_keys(
            coverage,
            {
                "source_allocation_count",
                "source_allocation_bytes",
                "source_dtype",
                "skeleton_parameter_count",
                "skeleton_tensor_bytes",
                "expert_source_parameter_count",
                "expert_source_tensor_bytes",
                "expert_page_count",
                "expert_store_tensor_bytes",
                "all_source_allocations_emitted_once",
                "expert_matrices_excluded_from_skeleton",
                "quantization_error",
            },
            "coverage",
        )
        expert_source_parameter_count = len(expected_coordinates) * 3
        skeleton_parameter_count = len(skeleton_names)
        if (
            coverage.get("source_dtype") != recipe.get("source_dtype")
            or coverage.get("skeleton_parameter_count") != skeleton_parameter_count
            or coverage.get("skeleton_tensor_bytes") != skeleton_tensor_bytes
            or coverage.get("expert_source_parameter_count") != expert_source_parameter_count
            or coverage.get("expert_source_tensor_bytes") != expert_source_tensor_bytes
            or coverage.get("expert_page_count") != len(expected_coordinates)
            or coverage.get("expert_store_tensor_bytes") != expert_store_tensor_bytes
            or coverage.get("source_allocation_count")
            != skeleton_parameter_count + expert_source_parameter_count
            or coverage.get("all_source_allocations_emitted_once") is not True
            or coverage.get("expert_matrices_excluded_from_skeleton") is not True
        ):
            raise MixtralExpertStoreArtifactError("expert-store coverage accounting is invalid")
        if len(source_allocation_ids) != coverage.get("source_allocation_count"):
            raise MixtralExpertStoreArtifactError("source allocation identity coverage is invalid")
        expected_source_bytes = sum(
            int(parameter["logical_bytes"])
            for shard in normalized_skeleton
            for parameter in shard["parameters"]
        ) + sum(
            int(parameter["logical_bytes"])
            for page in pages_by_key.values()
            for parameter in page["source_parameters"]
        )
        if coverage.get("source_allocation_bytes") != expected_source_bytes:
            raise MixtralExpertStoreArtifactError("source byte coverage is invalid")
        global_error = _require_dict(
            coverage.get("quantization_error"), "coverage quantization error"
        )
        _require_exact_keys(
            global_error,
            {"max_abs", "rmse", "elements"},
            "coverage quantization error",
        )
        expected_rmse = math.sqrt(page_error_squared / page_error_elements)
        if (
            type(global_error.get("max_abs")) not in {int, float}
            or type(global_error.get("rmse")) not in {int, float}
            or global_error.get("elements") != page_error_elements
            or not math.isclose(
                float(global_error["max_abs"]), page_error_maximum, rel_tol=1e-12, abs_tol=0.0
            )
            or not math.isclose(
                float(global_error["rmse"]), expected_rmse, rel_tol=1e-12, abs_tol=0.0
            )
        ):
            raise MixtralExpertStoreArtifactError(
                "aggregate quantization error evidence is inconsistent"
            )

        self._manifest = _freeze_json(manifest)
        self._config = _freeze_json(config)
        self._source = _freeze_json(source)
        self._topology = _freeze_json(topology_values)
        self.artifact_sha256 = artifact_sha
        self.build_key_sha256 = build_key
        self._skeleton_shards = tuple(_freeze_json(item) for item in normalized_skeleton)
        self._expert_pages = tuple(_freeze_json(pages_by_key[key]) for key in sorted(pages_by_key))
        self.skeleton_tensor_names = frozenset(skeleton_names)
        self.skeleton_tensor_bytes = skeleton_tensor_bytes
        self.expert_store_tensor_bytes = expert_store_tensor_bytes
        self.expert_page_tensor_bytes = next(iter(pages_by_key.values()))["tensor_bytes"]
        if any(
            page["tensor_bytes"] != self.expert_page_tensor_bytes for page in pages_by_key.values()
        ):
            raise MixtralExpertStoreArtifactError("expert pages do not have uniform geometry")
        self._pages_by_key = MappingProxyType(
            {key: _freeze_json(value) for key, value in pages_by_key.items()}
        )
        self._file_records = MappingProxyType(
            {key: _freeze_json(value) for key, value in file_records.items()}
        )
        self._identities = MappingProxyType(dict(identities))
        self._assert_directory_identity()
        self._frozen = True

    @property
    def manifest(self) -> dict[str, Any]:
        return _thaw_json(self._manifest)

    @property
    def config(self) -> dict[str, Any]:
        return _thaw_json(self._config)

    @property
    def source(self) -> dict[str, Any]:
        return _thaw_json(self._source)

    @property
    def topology(self) -> dict[str, Any]:
        return _thaw_json(self._topology)

    @property
    def skeleton_shards(self) -> tuple[dict[str, Any], ...]:
        return tuple(_thaw_json(item) for item in self._skeleton_shards)

    @property
    def expert_pages(self) -> tuple[dict[str, Any], ...]:
        return tuple(_thaw_json(item) for item in self._expert_pages)

    def expert_page(self, layer: int, expert: int) -> dict[str, Any]:
        try:
            return _thaw_json(self._pages_by_key[(int(layer), int(expert))])
        except (KeyError, TypeError, ValueError) as exc:
            raise MixtralExpertStoreArtifactError(
                "expert page coordinate is outside topology"
            ) from exc

    def verify_member(self, filename: str) -> Path:
        self._assert_directory_identity()
        relative = _member(filename, "artifact member")
        record = self._file_records.get(relative)
        identity = self._identities.get(relative)
        if record is None or identity is None:
            raise MixtralExpertStoreArtifactError("artifact member is not declared")
        path = self.path / relative
        _digest, _size, current_identity = _hash_regular_file(
            path,
            expected_size=int(record["bytes"]),
            expected_sha256=str(record.get("sha256", record.get("file_sha256"))),
        )
        if current_identity != identity:
            raise MixtralExpertStoreArtifactError("artifact member identity changed after reopen")
        return path

    @contextmanager
    def open_verified_member(self, filename: str) -> Iterator[BinaryIO]:
        """Yield the exact authenticated descriptor and rehash it after consumption."""

        self._assert_directory_identity()
        relative = _member(filename, "artifact member")
        record = self._file_records.get(relative)
        identity = self._identities.get(relative)
        if record is None or identity is None:
            raise MixtralExpertStoreArtifactError("artifact member is not declared")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path / relative, flags)
            opened = os.fstat(descriptor)
        except OSError as exc:
            raise MixtralExpertStoreArtifactError(
                "authenticated expert-store member hash mismatch or identity changed"
            ) from exc
        opened_identity = _identity(opened)
        if opened_identity != identity:
            os.close(descriptor)
            raise MixtralExpertStoreArtifactError("artifact member identity changed after reopen")
        handle = os.fdopen(descriptor, "rb", closefd=True)
        try:
            yield handle
            initial = os.fstat(handle.fileno())
            os.lseek(handle.fileno(), 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while chunk := os.read(handle.fileno(), 8 * 1024 * 1024):
                digest.update(chunk)
            final = os.fstat(handle.fileno())
            if (
                _identity(initial) != opened_identity
                or _identity(final) != opened_identity
                or digest.hexdigest() != str(record.get("sha256", record.get("file_sha256")))
            ):
                raise MixtralExpertStoreArtifactError(
                    "artifact member changed while it was consumed"
                )
            current = (self.path / relative).lstat()
            if _identity(current) != opened_identity:
                raise MixtralExpertStoreArtifactError(
                    "artifact member path changed while it was consumed"
                )
        finally:
            handle.close()

    def _assert_directory_identity(self) -> None:
        try:
            current = _identity(self.path.lstat())
        except OSError as exc:
            raise MixtralExpertStoreArtifactError("expert-store directory disappeared") from exc
        if current != self._directory_identity or self.path.is_symlink() or not self.path.is_dir():
            raise MixtralExpertStoreArtifactError("expert-store directory identity changed")

    def assert_unchanged(self) -> None:
        """Rehash every declared member and reject path or inventory substitution."""

        self._assert_directory_identity()
        try:
            manifest, manifest_identity, manifest_file_sha = _read_regular_json(
                self.path / "manifest.json"
            )
        except Exception as exc:
            raise MixtralExpertStoreArtifactError("cannot reread expert-store manifest") from exc
        if (
            manifest_identity != self._identities["manifest.json"]
            or manifest_file_sha != _sha256_bytes(_canonical_json_bytes(manifest))
            or manifest != self.manifest
        ):
            raise MixtralExpertStoreArtifactError("expert-store manifest changed after reopen")
        for filename in sorted(self._file_records):
            self.verify_member(filename)
        expected_files = {"manifest.json", *self._file_records}
        observed_files: set[str] = set()
        observed_directories: set[str] = set()
        for member in self.path.rglob("*"):
            relative = member.relative_to(self.path).as_posix()
            if member.is_symlink():
                raise MixtralExpertStoreArtifactError("expert-store inventory contains a symlink")
            if member.is_dir():
                observed_directories.add(relative)
            elif member.is_file():
                observed_files.add(relative)
            else:
                raise MixtralExpertStoreArtifactError(
                    "expert-store inventory contains a non-regular member"
                )
        if observed_files != expected_files or observed_directories != {"experts", "skeleton"}:
            raise MixtralExpertStoreArtifactError("expert-store file inventory changed")

    def verify_expert_page(self, layer: int, expert: int) -> Path:
        return self.verify_member(self.expert_page(layer, expert)["filename"])


__all__ = [
    "MIXTRAL_EXPERT_STORE_BITS",
    "MIXTRAL_EXPERT_STORE_CODEC",
    "MIXTRAL_EXPERT_STORE_GROUP_SIZE",
    "MIXTRAL_EXPERT_STORE_MODE",
    "MIXTRAL_EXPERT_STORE_NUMERICAL_CONTRACT",
    "MIXTRAL_EXPERT_STORE_RUNTIME_ABI",
    "MIXTRAL_EXPERT_STORE_SCHEMA",
    "MixtralExpertStoreArtifactError",
    "MixtralExpertStoreBuildRecord",
    "MixtralExpertStoreLoweringError",
    "VerifiedMixtralExpertStore",
    "build_mixtral_mlx_expert_store",
]
