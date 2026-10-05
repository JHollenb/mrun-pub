"""Compiler-only direct canonical-source MLX affine q2/q3 artifacts.

This module intentionally does not modify or wrap the affine-q4 v1 implementation in
``mlx_native``.  Q4 recipes bind the complete byte content of that module, so leaving it untouched
preserves the existing q4 schema, build keys, shard bytes, and artifact identities.  Q2 and q3
share the strict parameterized implementation below, but have independent public schemas, codecs,
builder ABIs, numerical contracts, verifiers, records, and content-addressed target paths.

These artifacts are approximate compiler outputs.  Their builders and verifiers do not execute or
certify a model.  Separate, explicitly experimental runtime classes may consume a strictly matched
artifact, but neither precision is quality- or production-promoted.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from mrun.engine.mlx_component import (
    _canonical_json_bytes,
    _fsync_directory,
    _hash_regular_file,
    _identity,
    _read_regular_json,
    _safe_model_slug,
    _sha256_bytes,
    _write_durable,
)

from .emitter import ComponentArtifact, open_component_artifact
from .mlx_native import (
    SOURCE_MLX_MAPPING_ABI,
    SOURCE_MLX_SHARD_BYTES,
    MLXSourceComponentEngine,
    SourceMlxArtifactError,
    SourceMlxLoweringError,
    _build_lock,
    _parameter_role,
    _read_allocation,
    _tokenizer_custody_sha256,
    _validated_config,
)
from .reference import ReferenceLoweringError, lower_component_artifact_to_reference

SOURCE_MLX_Q3_NATIVE_SCHEMA = "mrun-mlx-source-component-q3-native-v1"
SOURCE_MLX_Q3_BUILDER_ABI = "mrun-canonical-source-to-mlx-affine-q3-native-v1"
SOURCE_MLX_Q3_CODEC = "mlx-affine-int3-g64-bf16-direct-canonical-source-v1"
SOURCE_MLX_Q3_GROUP_SIZE = 64
SOURCE_MLX_Q3_BITS = 3
SOURCE_MLX_Q3_MODE = "affine"
SOURCE_MLX_Q3_NUMERICAL_CONTRACT = (
    "mlx-source-component-q3g64-bf16-weight-approximate-source-aux-exact-v1"
)

SOURCE_MLX_Q2_NATIVE_SCHEMA = "mrun-mlx-source-component-q2-native-v1"
SOURCE_MLX_Q2_BUILDER_ABI = "mrun-canonical-source-to-mlx-affine-q2-native-v1"
SOURCE_MLX_Q2_CODEC = "mlx-affine-int2-g64-bf16-direct-canonical-source-v1"
SOURCE_MLX_Q2_GROUP_SIZE = 64
SOURCE_MLX_Q2_BITS = 2
SOURCE_MLX_Q2_MODE = "affine"
SOURCE_MLX_Q2_NUMERICAL_CONTRACT = (
    "mlx-source-component-q2g64-bf16-weight-approximate-source-aux-exact-v1"
)

_LOWBIT_ARCHITECTURES = {
    "qwen2-dense-causal-decoder": "qwen2",
    "qwen3-dense-causal-decoder": "qwen3",
    "llama-dense-causal-decoder": "llama",
}
_SOURCE_DTYPES = frozenset({"BF16", "F16", "F32"})
_SOURCE_DTYPE_BITS = {"BF16": 16, "F16": 16, "F32": 32}
_SAFE_TENSOR_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHARD_ROLES = frozenset({"body", "norm", "ingress", "egress", "lexical_shared"})


@dataclass(frozen=True, slots=True)
class _AffineLowBitSpec:
    label: str
    schema: str
    builder_abi: str
    codec: str
    group_size: int
    bits: int
    mode: str
    numerical_contract: str

    def __post_init__(self) -> None:
        if self.label != f"q{self.bits}" or self.bits not in {2, 3}:
            raise ValueError("low-bit spec label/bits are inconsistent")
        if self.group_size != 64 or self.mode != "affine":
            raise ValueError("low-bit source artifacts require affine g64")
        if self.group_size * self.bits % 32:
            raise ValueError("low-bit groups must occupy a whole number of uint32 words")

    @property
    def allocation_count_field(self) -> str:
        return f"{self.label}_allocation_count"

    @property
    def encoding(self) -> str:
        return f"mlx-affine-{self.label}-g{self.group_size}"

    def packed_columns(self, source_columns: int) -> int:
        numerator = source_columns * self.bits
        if numerator % 32:
            raise SourceMlxLoweringError(
                f"{self.label} source width {source_columns} cannot be packed into uint32 words"
            )
        return numerator // 32


_Q3 = _AffineLowBitSpec(
    label="q3",
    schema=SOURCE_MLX_Q3_NATIVE_SCHEMA,
    builder_abi=SOURCE_MLX_Q3_BUILDER_ABI,
    codec=SOURCE_MLX_Q3_CODEC,
    group_size=SOURCE_MLX_Q3_GROUP_SIZE,
    bits=SOURCE_MLX_Q3_BITS,
    mode=SOURCE_MLX_Q3_MODE,
    numerical_contract=SOURCE_MLX_Q3_NUMERICAL_CONTRACT,
)
_Q2 = _AffineLowBitSpec(
    label="q2",
    schema=SOURCE_MLX_Q2_NATIVE_SCHEMA,
    builder_abi=SOURCE_MLX_Q2_BUILDER_ABI,
    codec=SOURCE_MLX_Q2_CODEC,
    group_size=SOURCE_MLX_Q2_GROUP_SIZE,
    bits=SOURCE_MLX_Q2_BITS,
    mode=SOURCE_MLX_Q2_MODE,
    numerical_contract=SOURCE_MLX_Q2_NUMERICAL_CONTRACT,
)


def _is_lower_sha256(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _strict_positive_int(value: Any, field: str, spec: _AffineLowBitSpec) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceMlxArtifactError(f"direct-source {spec.label} {field} must be positive")
    return int(value)


def _artifact_error(spec: _AffineLowBitSpec, message: str) -> SourceMlxArtifactError:
    return SourceMlxArtifactError(f"direct-source {spec.label} {message}")


def _lowbit_config(config: Mapping[str, Any], spec: _AffineLowBitSpec) -> dict[str, Any]:
    effective = json.loads(_canonical_json_bytes(dict(config)))
    quantization = {
        "bits": spec.bits,
        "group_size": spec.group_size,
        "mode": spec.mode,
    }
    effective["quantization"] = dict(quantization)
    effective["quantization_config"] = dict(quantization)
    effective.pop("quantize_activations", None)
    return effective


def _lowbit_recipe(
    artifact: ComponentArtifact,
    *,
    source_dtype: str,
    config_sha256: str,
    spec: _AffineLowBitSpec,
) -> dict[str, Any]:
    builder_sha256, _size, _identity_value = _hash_regular_file(Path(__file__).resolve())
    return {
        "schema": spec.schema,
        "builder_abi": spec.builder_abi,
        "builder_sha256": builder_sha256,
        "mapping_abi": SOURCE_MLX_MAPPING_ABI,
        "codec": spec.codec,
        "group_size": spec.group_size,
        "bits": spec.bits,
        "mode": spec.mode,
        "packed_storage_dtype": "uint32",
        "packed_columns_expression": "source_columns*bits/32",
        "shard_bytes": SOURCE_MLX_SHARD_BYTES,
        "quantizer": "mlx.core.quantize",
        "quantizer_input_dtype": "bfloat16",
        "auxiliary_codec": f"safetensors-{source_dtype.lower()}-source-exact-v1",
        "error_reference": f"canonical-source-{source_dtype.lower()}-float32",
        "mlx_version": importlib.metadata.version("mlx"),
        "numerical_contract": spec.numerical_contract,
        "source_dtype": source_dtype,
        "source_artifact_id": artifact.artifact_id,
        "source_manifest_sha256": artifact.manifest_sha256,
        "source_fingerprint": artifact.source.fingerprint,
        "source_ir_fingerprint": artifact.ir_bundle.fingerprint,
        "source_io_fingerprint": artifact.ir_bundle.io.fingerprint,
        "source_model_fingerprint": artifact.ir_bundle.model.fingerprint,
        "source_tokenizer_custody_sha256": _tokenizer_custody_sha256(artifact),
        "effective_config_sha256": config_sha256,
        "direct_from_canonical_source": True,
        "intermediate_qstore": False,
    }


def _build_record_payload(record: Any) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "path": str(record.path),
        "artifact_sha256": record.artifact_sha256,
        "build_key_sha256": record.build_key_sha256,
        "source_artifact_id": record.source_artifact_id,
        "shard_count": record.shard_count,
        "shard_bytes": record.shard_bytes,
        "verified_reopen": record.verified_reopen,
        "max_abs_error": record.max_abs_error,
        "rmse": record.rmse,
        "direct_from_canonical_source": record.direct_from_canonical_source,
        "intermediate_qstore": record.intermediate_qstore,
        "approximate_quantized": record.approximate_quantized,
        "native_runtime_candidate": record.native_runtime_candidate,
        "production_runtime_eligible": record.production_runtime_eligible,
    }


@dataclass(frozen=True, slots=True)
class SourceMlxQ3BuildRecord:
    path: Path
    artifact_sha256: str
    build_key_sha256: str
    source_artifact_id: str
    shard_count: int
    shard_bytes: int
    verified_reopen: bool
    max_abs_error: float
    rmse: float
    direct_from_canonical_source: bool = True
    intermediate_qstore: bool = False
    approximate_quantized: bool = True
    native_runtime_candidate: bool = True
    production_runtime_eligible: bool = False
    schema_version: str = SOURCE_MLX_Q3_NATIVE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return _build_record_payload(self)


@dataclass(frozen=True, slots=True)
class SourceMlxQ2BuildRecord:
    path: Path
    artifact_sha256: str
    build_key_sha256: str
    source_artifact_id: str
    shard_count: int
    shard_bytes: int
    verified_reopen: bool
    max_abs_error: float
    rmse: float
    direct_from_canonical_source: bool = True
    intermediate_qstore: bool = False
    approximate_quantized: bool = True
    native_runtime_candidate: bool = True
    production_runtime_eligible: bool = False
    schema_version: str = SOURCE_MLX_Q2_NATIVE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return _build_record_payload(self)


@dataclass(frozen=True, slots=True)
class _LowBitBuildValues:
    path: Path
    artifact_sha256: str
    build_key_sha256: str
    source_artifact_id: str
    shard_count: int
    shard_bytes: int
    verified_reopen: bool
    max_abs_error: float
    rmse: float


@dataclass(frozen=True, slots=True)
class _DirectAffineLowBit:
    weight: Any
    scales: Any
    biases: Any
    max_abs_error: float
    sum_squared_error: float
    elements: int


def _quantize_source_affine_lowbit(
    tensor: torch.Tensor,
    *,
    mx: Any,
    spec: _AffineLowBitSpec,
) -> _DirectAffineLowBit:
    if tensor.ndim != 2:
        raise TypeError(f"direct affine {spec.label} source tensor must be two-dimensional")
    rows, columns = (int(value) for value in tensor.shape)
    if columns % spec.group_size:
        raise SourceMlxLoweringError(
            f"{spec.label} source width {columns} is not divisible by {spec.group_size}"
        )
    packed_columns = spec.packed_columns(columns)
    if not bool(torch.isfinite(tensor.float()).all().item()):
        raise SourceMlxLoweringError("canonical source tensor contains non-finite values")
    reference = mx.array(tensor.float().numpy()).astype(mx.float32)
    quantizer_input = reference.astype(mx.bfloat16)
    quantized = mx.quantize(
        quantizer_input,
        group_size=spec.group_size,
        bits=spec.bits,
        mode=spec.mode,
    )
    if len(quantized) != 3:
        raise _artifact_error(spec, "MLX affine quantizer returned an invalid tuple")
    weight, scales, biases = quantized
    expected_aux_shape = (rows, columns // spec.group_size)
    if (
        tuple(int(value) for value in weight.shape) != (rows, packed_columns)
        or weight.dtype != mx.uint32
        or tuple(int(value) for value in scales.shape) != expected_aux_shape
        or tuple(int(value) for value in biases.shape) != expected_aux_shape
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
    ):
        raise _artifact_error(spec, "MLX affine quantizer returned an unexpected packed shape")
    restored = mx.dequantize(
        weight,
        scales,
        biases,
        group_size=spec.group_size,
        bits=spec.bits,
        mode=spec.mode,
    )
    error = restored.astype(mx.float32) - reference
    maximum = mx.max(mx.abs(error))
    squared = mx.sum(mx.square(error))
    mx.eval(weight, scales, biases, maximum, squared)
    max_abs_error = float(maximum.item())
    sum_squared_error = float(squared.item())
    if not math.isfinite(max_abs_error) or not math.isfinite(sum_squared_error):
        raise _artifact_error(spec, "MLX affine error evidence is non-finite")
    return _DirectAffineLowBit(
        weight=weight,
        scales=scales,
        biases=biases,
        max_abs_error=max_abs_error,
        sum_squared_error=sum_squared_error,
        elements=rows * columns,
    )


def _source_auxiliary_array(
    tensor: torch.Tensor, source_dtype: str, *, mx: Any, spec: _AffineLowBitSpec
) -> Any:
    if not bool(torch.isfinite(tensor.float()).all().item()):
        raise SourceMlxLoweringError("canonical source tensor contains non-finite values")
    dtype = {
        "BF16": mx.bfloat16,
        "F16": mx.float16,
        "F32": mx.float32,
    }[source_dtype]
    value = mx.array(tensor.float().numpy()).astype(dtype)
    mx.eval(value)
    if tuple(int(dimension) for dimension in value.shape) != tuple(tensor.shape):
        raise _artifact_error(spec, "source-exact auxiliary changed shape")
    return value


class _VerifiedSourceMlxLowBitArtifact:
    """Strict immutable view shared only by the new affine q2/q3 artifact schemas."""

    _SPEC: ClassVar[_AffineLowBitSpec]
    _PARAMETER_FIELDS = {
        "name",
        "dtype",
        "shape",
        "part",
        "encoding",
        "source_allocation_id",
        "source_blob_sha256",
        "source_tensor",
        "source_dtype",
        "source_shape",
        "source_byte_count",
        "logical_names",
    }

    def __init__(self, path: str | Path) -> None:
        spec = self._SPEC
        self.path = Path(path).expanduser().absolute()
        if self.path.is_symlink() or not self.path.is_dir():
            raise _artifact_error(spec, "artifact must be a real directory")
        self.path = self.path.resolve()
        self._directory_identity = _identity(self.path.lstat())
        try:
            manifest, manifest_identity, _manifest_file_sha = _read_regular_json(
                self.path / "manifest.json"
            )
        except Exception as exc:
            raise _artifact_error(spec, "manifest cannot be verified") from exc
        if manifest.get("schema") != spec.schema:
            raise _artifact_error(spec, "artifact schema is unsupported")
        if (
            manifest.get("status") != "native-lowered-approximate-unexecuted"
            or manifest.get("execution_certified") is not False
            or manifest.get("native_runtime_candidate") is not True
            or manifest.get("production_runtime_eligible") is not False
            or manifest.get("approximate_quantized") is not True
            or manifest.get("numerical_contract") != spec.numerical_contract
        ):
            raise _artifact_error(spec, "promotion boundary is malformed")
        declared = manifest.get("artifact_sha256")
        if not _is_lower_sha256(declared):
            raise _artifact_error(spec, "artifact identity is malformed")
        unhashed = dict(manifest)
        unhashed.pop("artifact_sha256", None)
        if declared != _sha256_bytes(_canonical_json_bytes(unhashed)):
            raise _artifact_error(spec, "artifact identity mismatch")

        recipe = manifest.get("recipe")
        if not isinstance(recipe, Mapping):
            raise _artifact_error(spec, "artifact has no recipe")
        expected_recipe = {
            "schema": spec.schema,
            "builder_abi": spec.builder_abi,
            "mapping_abi": SOURCE_MLX_MAPPING_ABI,
            "codec": spec.codec,
            "group_size": spec.group_size,
            "bits": spec.bits,
            "mode": spec.mode,
            "packed_storage_dtype": "uint32",
            "packed_columns_expression": "source_columns*bits/32",
            "shard_bytes": SOURCE_MLX_SHARD_BYTES,
            "quantizer": "mlx.core.quantize",
            "quantizer_input_dtype": "bfloat16",
            "numerical_contract": spec.numerical_contract,
            "direct_from_canonical_source": True,
            "intermediate_qstore": False,
        }
        if any(recipe.get(key) != value for key, value in expected_recipe.items()):
            raise _artifact_error(spec, "recipe ABI is unsupported")
        source_dtype = str(recipe.get("source_dtype", ""))
        if source_dtype not in _SOURCE_DTYPES:
            raise _artifact_error(spec, "source dtype is unsupported")
        if (
            recipe.get("auxiliary_codec") != f"safetensors-{source_dtype.lower()}-source-exact-v1"
            or recipe.get("error_reference") != f"canonical-source-{source_dtype.lower()}-float32"
            or type(recipe.get("mlx_version")) is not str
            or not recipe.get("mlx_version")
            or not _is_lower_sha256(recipe.get("builder_sha256"))
        ):
            raise _artifact_error(spec, "quantizer contract is invalid")
        lineage_recipe_fields = (
            "source_artifact_id",
            "source_manifest_sha256",
            "source_fingerprint",
            "source_ir_fingerprint",
            "source_io_fingerprint",
            "source_model_fingerprint",
            "source_tokenizer_custody_sha256",
            "effective_config_sha256",
        )
        for field in lineage_recipe_fields:
            if not _is_lower_sha256(recipe.get(field)):
                raise _artifact_error(spec, f"recipe {field!r} is malformed")
        build_key = _sha256_bytes(_canonical_json_bytes(recipe))
        if manifest.get("build_key_sha256") != build_key:
            raise _artifact_error(spec, "build key mismatch")

        config_record = manifest.get("config")
        if not isinstance(config_record, Mapping) or config_record.get("filename") != "config.json":
            raise _artifact_error(spec, "config record is malformed")
        try:
            config, config_identity, config_file_sha = _read_regular_json(self.path / "config.json")
        except Exception as exc:
            raise _artifact_error(spec, "config cannot be verified") from exc
        config_sha = _sha256_bytes(_canonical_json_bytes(config))
        if (
            config_file_sha != config_record.get("file_sha256")
            or _strict_positive_int(config_record.get("bytes"), "config bytes", spec)
            != int(config_identity[3])
            or config_sha != config_record.get("semantic_sha256")
            or config_sha != recipe.get("effective_config_sha256")
        ):
            raise _artifact_error(spec, "config is not recipe-bound")
        quantization_config = {
            "bits": spec.bits,
            "group_size": spec.group_size,
            "mode": spec.mode,
        }
        if (
            config.get("quantization") != quantization_config
            or config.get("quantization_config") != quantization_config
        ):
            raise _artifact_error(spec, "config quantization is invalid")

        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise _artifact_error(spec, "source record is malformed")
        source_hash_fields = (
            "artifact_id",
            "manifest_sha256",
            "source_fingerprint",
            "ir_bundle_fingerprint",
            "io_fingerprint",
            "model_fingerprint",
            "tokenizer_custody_sha256",
        )
        for field in source_hash_fields:
            if not _is_lower_sha256(source.get(field)):
                raise _artifact_error(spec, f"source {field!r} is malformed")
        source_recipe_pairs = {
            "artifact_id": "source_artifact_id",
            "manifest_sha256": "source_manifest_sha256",
            "source_fingerprint": "source_fingerprint",
            "ir_bundle_fingerprint": "source_ir_fingerprint",
            "io_fingerprint": "source_io_fingerprint",
            "model_fingerprint": "source_model_fingerprint",
            "tokenizer_custody_sha256": "source_tokenizer_custody_sha256",
        }
        if any(
            source.get(left) != recipe.get(right) for left, right in source_recipe_pairs.items()
        ):
            raise _artifact_error(spec, "source lineage is not recipe-bound")
        architecture_id = source.get("architecture_id")
        if (
            source.get("direct_from_canonical_source") is not True
            or source.get("intermediate_qstore") is not False
            or source.get("source_dtype") != source_dtype
            or type(source.get("tied_lexical_allocation")) is not bool
            or architecture_id not in _LOWBIT_ARCHITECTURES
            or source.get("architecture") != _LOWBIT_ARCHITECTURES[architecture_id]
            or config.get("model_type") != source.get("architecture")
            or config.get("tie_word_embeddings") != source.get("tied_lexical_allocation")
        ):
            raise _artifact_error(spec, "topology lineage is inconsistent")

        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise _artifact_error(spec, "artifact has no shards")
        identities = {"manifest.json": manifest_identity, "config.json": config_identity}
        observed_filenames: set[str] = set()
        observed_parameter_names: set[str] = set()
        observed_roles: set[str] = set()
        allocation_parameters: dict[str, list[tuple[Mapping[str, Any], torch.Tensor, str]]] = (
            defaultdict(list)
        )
        shard_bytes = 0
        emitted_parameter_bytes = 0
        for shard in shards:
            if not isinstance(shard, Mapping):
                raise _artifact_error(spec, "shard record is malformed")
            filename = shard.get("filename")
            role = shard.get("role")
            if (
                type(filename) is not str
                or re.fullmatch(r"model-[a-z_]+-[0-9]{5}\.safetensors", filename) is None
                or filename in observed_filenames
                or role not in _SHARD_ROLES
                or not _is_lower_sha256(shard.get("sha256"))
            ):
                raise _artifact_error(spec, "shard name/role is invalid")
            expected_size = _strict_positive_int(
                shard.get("bytes"), f"shard {filename} bytes", spec
            )
            try:
                digest, size, identity = _hash_regular_file(
                    self.path / filename,
                    expected_size=expected_size,
                    expected_sha256=str(shard["sha256"]),
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise _artifact_error(
                    spec, f"shard hash verification failed: {filename!r}"
                ) from exc
            if digest != shard.get("sha256"):
                raise _artifact_error(spec, "shard hash drifted")
            parameters = shard.get("parameters")
            if not isinstance(parameters, list) or not parameters:
                raise _artifact_error(spec, "shard has no parameters")
            if any(not isinstance(item, Mapping) for item in parameters):
                raise _artifact_error(spec, "parameter record is malformed")
            from safetensors import safe_open

            with safe_open(self.path / filename, framework="pt", device="cpu") as handle:
                keys = tuple(handle.keys())
                declared_names = [str(item.get("name")) for item in parameters]
                if len(set(declared_names)) != len(declared_names) or set(keys) != set(
                    declared_names
                ):
                    raise _artifact_error(spec, "shard header differs")
                metadata = handle.metadata() or {}
                if (
                    metadata.get("format") != "mlx"
                    or metadata.get("mrun-codec") != spec.codec
                    or metadata.get("mrun-role") != role
                    or metadata.get("mrun-source-artifact") != source["artifact_id"]
                ):
                    raise _artifact_error(spec, "shard metadata drifted")
                for item in parameters:
                    if set(item) != self._PARAMETER_FIELDS:
                        raise _artifact_error(spec, "parameter inventory is not exact")
                    name = item["name"]
                    allocation_id = item["source_allocation_id"]
                    logical_names = item["logical_names"]
                    source_shape = item["source_shape"]
                    shape = item["shape"]
                    if (
                        type(name) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(name) is None
                        or name in observed_parameter_names
                        or type(allocation_id) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(allocation_id) is None
                        or not _is_lower_sha256(item["source_blob_sha256"])
                        or type(item["source_tensor"]) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(item["source_tensor"]) is None
                        or item["source_dtype"] != source_dtype
                        or not isinstance(logical_names, list)
                        or logical_names != sorted(set(logical_names))
                        or not logical_names
                        or any(type(value) is not str or not value for value in logical_names)
                        or not isinstance(source_shape, list)
                        or not source_shape
                        or any(type(value) is not int or value <= 0 for value in source_shape)
                        or not isinstance(shape, list)
                        or not shape
                        or any(type(value) is not int or value <= 0 for value in shape)
                        or _strict_positive_int(
                            item["source_byte_count"], "source parameter bytes", spec
                        )
                        != math.prod(source_shape) * (_SOURCE_DTYPE_BITS[source_dtype] // 8)
                    ):
                        raise _artifact_error(spec, "parameter descriptor is malformed")
                    if _parameter_role(logical_names) != role:
                        raise _artifact_error(spec, "parameter crossed its physical role")
                    value = handle.get_tensor(name)
                    dtype_map = {
                        "U32": torch.uint32,
                        "BF16": torch.bfloat16,
                        "F16": torch.float16,
                        "F32": torch.float32,
                    }
                    dtype = item["dtype"]
                    if (
                        dtype not in dtype_map
                        or value.dtype != dtype_map[dtype]
                        or list(value.shape) != shape
                    ):
                        raise _artifact_error(spec, "parameter shape/dtype drifted")
                    observed_parameter_names.add(name)
                    emitted_parameter_bytes += int(value.numel()) * int(value.element_size())
                    allocation_parameters[allocation_id].append((item, value, str(role)))
            observed_filenames.add(filename)
            observed_roles.add(str(role))
            identities[filename] = identity
            shard_bytes += size

        quantized_allocations: set[str] = set()
        auxiliary_allocations: set[str] = set()
        source_allocation_bytes = 0
        source_weight_elements = 0
        allocation_facts: dict[str, dict[str, Any]] = {}
        for allocation_id, entries in allocation_parameters.items():
            first = entries[0][0]
            lineage_fields = (
                "source_blob_sha256",
                "source_tensor",
                "source_dtype",
                "source_shape",
                "source_byte_count",
                "logical_names",
                "encoding",
            )
            if any(
                _canonical_json_bytes(entry[0].get(field))
                != _canonical_json_bytes(first.get(field))
                for entry in entries[1:]
                for field in lineage_fields
            ):
                raise _artifact_error(spec, "allocation descriptors disagree")
            source_shape = [int(value) for value in first["source_shape"]]
            source_tensor = str(first["source_tensor"])
            logical_names = list(first["logical_names"])
            source_allocation_bytes += int(first["source_byte_count"])
            allocation_facts[allocation_id] = {
                "source_tensor": source_tensor,
                "logical_names": logical_names,
                "source_shape": source_shape,
            }
            if first["encoding"] == spec.encoding:
                if len(source_shape) != 2 or not source_tensor.endswith(".weight"):
                    raise _artifact_error(spec, "encoding was applied to an ineligible tensor")
                rows, columns = source_shape
                if columns % spec.group_size:
                    raise _artifact_error(spec, "source width violates group size")
                packed_columns = spec.packed_columns(columns)
                base = source_tensor[: -len(".weight")]
                expected = {
                    f"{base}.weight": ("weight", "U32", [rows, packed_columns]),
                    f"{base}.scales": (
                        "scales",
                        "BF16",
                        [rows, columns // spec.group_size],
                    ),
                    f"{base}.biases": (
                        "biases",
                        "BF16",
                        [rows, columns // spec.group_size],
                    ),
                }
                observed = {
                    str(item["name"]): (item["part"], item["dtype"], item["shape"])
                    for item, _value, _role in entries
                }
                if observed != expected or len(entries) != 3:
                    raise _artifact_error(spec, "packed triple is incomplete")
                quantized_allocations.add(allocation_id)
                source_weight_elements += rows * columns
            elif first["encoding"] == "source-exact":
                if len(entries) != 1 or len(source_shape) != 1:
                    raise _artifact_error(spec, "source-exact auxiliary allocation is malformed")
                item = entries[0][0]
                if (
                    item["name"] != source_tensor
                    or item["part"] != "source_exact"
                    or item["dtype"] != source_dtype
                    or item["shape"] != source_shape
                ):
                    raise _artifact_error(spec, "source-exact auxiliary tensor drifted")
                auxiliary_allocations.add(allocation_id)
            else:
                raise _artifact_error(spec, "allocation encoding is unknown")

        expected_roles = (
            {"body", "norm", "lexical_shared"}
            if source["tied_lexical_allocation"]
            else {"body", "norm", "ingress", "egress"}
        )
        if observed_roles != expected_roles:
            raise _artifact_error(spec, "physical roles are incomplete")
        coverage = manifest.get("coverage")
        count_field = spec.allocation_count_field
        if (
            not isinstance(coverage, Mapping)
            or coverage.get("source_dtype") != source_dtype
            or coverage.get("all_source_allocations_emitted_once") is not True
            or coverage.get("physical_aliases_not_duplicated") is not True
            or coverage.get("auxiliary_source_exact") is not True
            or coverage.get("source_allocation_count") != len(allocation_parameters)
            or coverage.get(count_field) != len(quantized_allocations)
            or coverage.get("auxiliary_allocation_count") != len(auxiliary_allocations)
            or coverage.get("emitted_parameter_count") != len(observed_parameter_names)
            or coverage.get("source_allocation_bytes") != source_allocation_bytes
            or coverage.get("emitted_parameter_bytes") != emitted_parameter_bytes
        ):
            raise _artifact_error(spec, "coverage claim is inconsistent")

        quantization = manifest.get("quantization")
        if not isinstance(quantization, Mapping):
            raise _artifact_error(spec, "artifact has no error evidence")
        blocks = quantization.get("blocks")
        if not isinstance(blocks, list) or any(not isinstance(item, Mapping) for item in blocks):
            raise _artifact_error(spec, "block evidence is malformed")
        block_ids: set[str] = set()
        total_squared_error = 0.0
        total_elements = 0
        maximum_error = 0.0
        for block in blocks:
            allocation_id = block.get("source_allocation_id")
            facts = allocation_facts.get(str(allocation_id))
            elements = block.get("elements")
            max_abs = block.get("max_abs_error")
            squared = block.get("sum_squared_error")
            rmse = block.get("rmse")
            expected_elements = math.prod(facts["source_shape"]) if facts is not None else None
            if (
                allocation_id not in quantized_allocations
                or allocation_id in block_ids
                or facts is None
                or block.get("source_tensor") != facts["source_tensor"]
                or block.get("logical_names") != facts["logical_names"]
                or type(elements) is not int
                or elements <= 0
                or elements != expected_elements
                or type(max_abs) not in {int, float}
                or type(squared) not in {int, float}
                or type(rmse) not in {int, float}
                or not all(math.isfinite(float(value)) for value in (max_abs, squared, rmse))
                or float(max_abs) < 0
                or float(squared) < 0
                or float(rmse) < 0
                or not math.isclose(
                    float(rmse),
                    math.sqrt(float(squared) / elements),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise _artifact_error(spec, "block evidence is inconsistent")
            block_ids.add(str(allocation_id))
            total_elements += elements
            total_squared_error += float(squared)
            maximum_error = max(maximum_error, float(max_abs))
        aggregate_rmse = math.sqrt(total_squared_error / total_elements) if total_elements else -1.0
        if (
            block_ids != quantized_allocations
            or quantization.get("bits") != spec.bits
            or quantization.get("group_size") != spec.group_size
            or quantization.get("mode") != spec.mode
            or quantization.get("packed_storage_dtype") != "U32"
            or quantization.get("packed_columns_expression") != "source_columns*bits/32"
            or quantization.get("source_codec")
            != f"safetensors-{source_dtype.lower()}-canonical-source-v1"
            or quantization.get("quantizer") != "mlx.core.quantize"
            or quantization.get(count_field) != len(quantized_allocations)
            or quantization.get("elements") != source_weight_elements
            or total_elements != source_weight_elements
            or quantization.get("auxiliary_source_exact") is not True
            or quantization.get("auxiliary_allocation_count") != len(auxiliary_allocations)
            or not math.isclose(
                float(quantization.get("max_abs_error", -1.0)),
                maximum_error,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(quantization.get("rmse", -1.0)),
                aggregate_rmse,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise _artifact_error(spec, "aggregate evidence is inconsistent")
        if {entry.name for entry in self.path.iterdir()} != set(identities):
            raise _artifact_error(spec, "artifact has undeclared files")

        self.manifest = dict(manifest)
        self.config = config
        self.source = dict(source)
        self.artifact_sha256 = str(declared)
        self.build_key_sha256 = build_key
        self.codec = spec.codec
        self.bits = spec.bits
        self.group_size = spec.group_size
        self.source_dtype = source_dtype
        self.shard_bytes = shard_bytes
        self.max_abs_error = maximum_error
        self.rmse = aggregate_rmse
        self._identities = identities

    def assert_unchanged(self) -> None:
        spec = self._SPEC
        if _identity(self.path.lstat()) != self._directory_identity:
            raise _artifact_error(spec, "directory identity changed")
        if {entry.name for entry in self.path.iterdir()} != set(self._identities):
            raise _artifact_error(spec, "file inventory changed")
        for filename, identity in self._identities.items():
            if _identity((self.path / filename).lstat()) != identity:
                raise _artifact_error(spec, f"file changed: {filename}")


class VerifiedSourceMlxQ3Artifact(_VerifiedSourceMlxLowBitArtifact):
    """Strict verifier for a compiler-produced affine q3g64 MLX artifact."""

    _SPEC = _Q3


class VerifiedSourceMlxQ2Artifact(_VerifiedSourceMlxLowBitArtifact):
    """Strict verifier for a compiler-produced affine q2g64 MLX artifact."""

    _SPEC = _Q2


def _verifier_type(
    spec: _AffineLowBitSpec,
) -> type[VerifiedSourceMlxQ2Artifact] | type[VerifiedSourceMlxQ3Artifact]:
    return VerifiedSourceMlxQ3Artifact if spec is _Q3 else VerifiedSourceMlxQ2Artifact


def _build_values_from_verified(
    verified: _VerifiedSourceMlxLowBitArtifact,
    *,
    source_artifact_id: str,
) -> _LowBitBuildValues:
    return _LowBitBuildValues(
        path=verified.path,
        artifact_sha256=verified.artifact_sha256,
        build_key_sha256=verified.build_key_sha256,
        source_artifact_id=source_artifact_id,
        shard_count=len(verified.manifest["shards"]),
        shard_bytes=verified.shard_bytes,
        verified_reopen=True,
        max_abs_error=verified.max_abs_error,
        rmse=verified.rmse,
    )


def _build_source_mlx_lowbit_artifact(
    source_artifact: ComponentArtifact | str | Path,
    output_root: str | Path,
    *,
    spec: _AffineLowBitSpec,
) -> _LowBitBuildValues:
    try:
        reference = lower_component_artifact_to_reference(source_artifact)
    except ReferenceLoweringError as exc:
        raise SourceMlxLoweringError(str(exc), details=exc.details) from exc
    artifact = reference.artifact
    source_config, architecture, tied = _validated_config(artifact)
    if architecture not in set(_LOWBIT_ARCHITECTURES.values()):
        raise SourceMlxLoweringError(
            f"direct {spec.label} lowering has no certified mapping for this architecture"
        )
    allocations = artifact.ir_bundle.physical_weights.allocations
    dtypes = {item.stored_dtype for item in allocations}
    if len(dtypes) != 1:
        raise SourceMlxLoweringError(
            f"direct {spec.label} target requires one canonical source dtype"
        )
    source_dtype = next(iter(dtypes))
    if source_dtype not in _SOURCE_DTYPES:
        raise SourceMlxLoweringError(
            f"direct MLX affine {spec.label} lowering requires BF16, F16, or F32 canonical weights"
        )
    if any(".rotary_emb." in item.source_tensor for item in allocations):
        raise SourceMlxLoweringError(
            f"serialized RoPE tensors are not registered in the direct {spec.label} target"
        )

    views_by_allocation: dict[str, list[str]] = defaultdict(list)
    for view in artifact.ir_bundle.physical_weights.views:
        views_by_allocation[view.allocation_id].append(view.logical_name)
    for allocation in allocations:
        logical_names = views_by_allocation.get(allocation.allocation_id, [])
        if not logical_names:
            raise SourceMlxLoweringError("canonical allocation has no logical views")
        if len(allocation.stored_shape) == 2:
            if not allocation.source_tensor.endswith(".weight"):
                raise SourceMlxLoweringError(
                    f"two-dimensional {spec.label} allocation is not a weight"
                )
            columns = int(allocation.stored_shape[1])
            if columns % spec.group_size:
                raise SourceMlxLoweringError(
                    f"{spec.label} source width for {allocation.source_tensor!r} is not "
                    f"divisible by {spec.group_size}"
                )
            spec.packed_columns(columns)
        elif len(allocation.stored_shape) != 1:
            raise SourceMlxLoweringError(
                f"direct {spec.label} target supports only matrix weights and vector auxiliaries"
            )
    allocation_roles = {
        allocation.allocation_id: _parameter_role(views_by_allocation[allocation.allocation_id])
        for allocation in allocations
    }
    if tied and not any(role == "lexical_shared" for role in allocation_roles.values()):
        raise SourceMlxLoweringError("tied lexical allocation did not remain physically shared")

    config = _lowbit_config(source_config, spec)
    config_sha = _sha256_bytes(_canonical_json_bytes(config))
    try:
        recipe = _lowbit_recipe(
            artifact,
            source_dtype=source_dtype,
            config_sha256=config_sha,
            spec=spec,
        )
        import mlx.core as mx
    except ImportError as exc:
        raise SourceMlxArtifactError(
            f"mlx is required for direct affine {spec.label} lowering"
        ) from exc
    build_key = _sha256_bytes(_canonical_json_bytes(recipe))
    output_root = Path(output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise _artifact_error(spec, "output root must be a real directory")
    output_root = output_root.resolve()
    target = output_root / (
        f"{_safe_model_slug(artifact.source.source_id)}-{spec.label}-{build_key[:16]}"
    )
    verifier_type = _verifier_type(spec)

    with _build_lock(build_key):
        if target.exists() or target.is_symlink():
            verified = verifier_type(target)
            if verified.build_key_sha256 != build_key:
                raise _artifact_error(spec, "existing target has a foreign build key")
            return _build_values_from_verified(verified, source_artifact_id=artifact.artifact_id)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=output_root))
        try:
            config_file_sha, config_bytes = _write_durable(
                staging / "config.json", _canonical_json_bytes(config)
            )
            allocation_records = {
                str(item["allocation_id"]): item for item in artifact.manifest["allocations"]
            }
            by_role: dict[str, list[Any]] = defaultdict(list)
            for allocation in allocations:
                if _SAFE_TENSOR_NAME.fullmatch(allocation.source_tensor) is None:
                    raise SourceMlxLoweringError("source tensor name is unsafe for safetensors")
                by_role[allocation_roles[allocation.allocation_id]].append(allocation)

            shards: list[dict[str, Any]] = []
            error_records: list[dict[str, Any]] = []
            total_squared_error = 0.0
            total_elements = 0
            maximum_error = 0.0
            quantized_allocation_count = 0
            auxiliary_allocation_count = 0
            emitted_parameter_bytes = 0
            for role in sorted(by_role):
                shard_index = 0
                arrays: dict[str, Any] = {}
                parameters: list[dict[str, Any]] = []
                pending_bytes = 0

                def flush(role_value: str = role) -> None:
                    nonlocal shard_index, arrays, parameters, pending_bytes
                    if not arrays:
                        return
                    shard_index += 1
                    filename = f"model-{role_value}-{shard_index:05d}.safetensors"
                    path = staging / filename
                    mx.eval(*arrays.values())
                    mx.save_safetensors(
                        str(path),
                        dict(sorted(arrays.items())),
                        metadata={
                            "format": "mlx",
                            "mrun-codec": spec.codec,
                            "mrun-role": role_value,
                            "mrun-source-artifact": artifact.artifact_id,
                        },
                    )
                    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    digest, size, _identity_value = _hash_regular_file(path)
                    shards.append(
                        {
                            "filename": filename,
                            "role": role_value,
                            "bytes": size,
                            "sha256": digest,
                            "parameters": sorted(parameters, key=lambda item: item["name"]),
                        }
                    )
                    arrays = {}
                    parameters = []
                    pending_bytes = 0
                    mx.clear_cache()

                for allocation in sorted(by_role[role], key=lambda item: item.source_tensor):
                    record = allocation_records.get(allocation.allocation_id)
                    if not isinstance(record, Mapping) or not isinstance(
                        record.get("blob"), Mapping
                    ):
                        raise _artifact_error(spec, "canonical allocation manifest is incomplete")
                    tensor = _read_allocation(artifact, allocation, record)
                    logical_names = sorted(views_by_allocation[allocation.allocation_id])
                    common = {
                        "source_allocation_id": allocation.allocation_id,
                        "source_blob_sha256": record["blob"]["sha256"],
                        "source_tensor": allocation.source_tensor,
                        "source_dtype": source_dtype,
                        "source_shape": list(allocation.stored_shape),
                        "source_byte_count": allocation.byte_length,
                        "logical_names": logical_names,
                    }
                    new_arrays: list[tuple[str, Any, str, str, str]] = []
                    if tensor.ndim == 2:
                        packed = _quantize_source_affine_lowbit(tensor, mx=mx, spec=spec)
                        base = allocation.source_tensor[: -len(".weight")]
                        new_arrays.extend(
                            [
                                (
                                    f"{base}.weight",
                                    packed.weight,
                                    "U32",
                                    "weight",
                                    spec.encoding,
                                ),
                                (
                                    f"{base}.scales",
                                    packed.scales,
                                    "BF16",
                                    "scales",
                                    spec.encoding,
                                ),
                                (
                                    f"{base}.biases",
                                    packed.biases,
                                    "BF16",
                                    "biases",
                                    spec.encoding,
                                ),
                            ]
                        )
                        quantized_allocation_count += 1
                        maximum_error = max(maximum_error, packed.max_abs_error)
                        total_squared_error += packed.sum_squared_error
                        total_elements += packed.elements
                        error_records.append(
                            {
                                "source_allocation_id": allocation.allocation_id,
                                "source_tensor": allocation.source_tensor,
                                "logical_names": logical_names,
                                "elements": packed.elements,
                                "max_abs_error": packed.max_abs_error,
                                "sum_squared_error": packed.sum_squared_error,
                                "rmse": math.sqrt(packed.sum_squared_error / packed.elements),
                            }
                        )
                    else:
                        auxiliary = _source_auxiliary_array(tensor, source_dtype, mx=mx, spec=spec)
                        new_arrays.append(
                            (
                                allocation.source_tensor,
                                auxiliary,
                                source_dtype,
                                "source_exact",
                                "source-exact",
                            )
                        )
                        auxiliary_allocation_count += 1
                    added_bytes = sum(
                        int(array.nbytes) for _name, array, _dtype, _part, _encoding in new_arrays
                    )
                    if arrays and pending_bytes + added_bytes > SOURCE_MLX_SHARD_BYTES:
                        flush()
                    for name, value, dtype, part, encoding in new_arrays:
                        if name in arrays:
                            raise _artifact_error(
                                spec, f"multiple source allocations map to {name!r}"
                            )
                        arrays[name] = value
                        parameters.append(
                            {
                                **common,
                                "name": name,
                                "dtype": dtype,
                                "shape": [int(dimension) for dimension in value.shape],
                                "part": part,
                                "encoding": encoding,
                            }
                        )
                    pending_bytes += added_bytes
                    emitted_parameter_bytes += added_bytes
                flush()

            if (
                not quantized_allocation_count
                or not auxiliary_allocation_count
                or total_elements <= 0
            ):
                raise _artifact_error(
                    spec,
                    "lowering did not observe both matrix and auxiliary allocations",
                )
            reopened = open_component_artifact(artifact.directory)
            if (
                reopened.artifact_id != artifact.artifact_id
                or reopened.manifest_sha256 != artifact.manifest_sha256
            ):
                raise _artifact_error(spec, "canonical source changed during lowering")
            tokenizer_custody = _tokenizer_custody_sha256(artifact)
            source = {
                "artifact_schema": artifact.manifest["schema_version"],
                "artifact_id": artifact.artifact_id,
                "manifest_sha256": artifact.manifest_sha256,
                "source_fingerprint": artifact.source.fingerprint,
                "ir_bundle_fingerprint": artifact.ir_bundle.fingerprint,
                "io_fingerprint": artifact.ir_bundle.io.fingerprint,
                "model_fingerprint": artifact.ir_bundle.model.fingerprint,
                "tokenizer_custody_sha256": tokenizer_custody,
                "architecture_id": artifact.ir_bundle.model.architecture_id,
                "architecture": architecture,
                "source_dtype": source_dtype,
                "tied_lexical_allocation": tied,
                "direct_from_canonical_source": True,
                "intermediate_qstore": False,
            }
            rmse = math.sqrt(total_squared_error / total_elements)
            count_field = spec.allocation_count_field
            manifest: dict[str, Any] = {
                "schema": spec.schema,
                "status": "native-lowered-approximate-unexecuted",
                "build_key_sha256": build_key,
                "recipe": recipe,
                "source": source,
                "config": {
                    "filename": "config.json",
                    "bytes": config_bytes,
                    "file_sha256": config_file_sha,
                    "semantic_sha256": config_sha,
                },
                "quantization": {
                    "bits": spec.bits,
                    "group_size": spec.group_size,
                    "mode": spec.mode,
                    "packed_storage_dtype": "U32",
                    "packed_columns_expression": "source_columns*bits/32",
                    "source_codec": (f"safetensors-{source_dtype.lower()}-canonical-source-v1"),
                    "quantizer": "mlx.core.quantize",
                    count_field: quantized_allocation_count,
                    "elements": total_elements,
                    "max_abs_error": maximum_error,
                    "rmse": rmse,
                    "auxiliary_source_exact": True,
                    "auxiliary_allocation_count": auxiliary_allocation_count,
                    "blocks": sorted(error_records, key=lambda item: item["source_allocation_id"]),
                },
                "shards": sorted(shards, key=lambda item: item["filename"]),
                "coverage": {
                    "source_allocation_count": len(allocations),
                    count_field: quantized_allocation_count,
                    "auxiliary_allocation_count": auxiliary_allocation_count,
                    "emitted_parameter_count": sum(len(item["parameters"]) for item in shards),
                    "source_allocation_bytes": sum(item.byte_length for item in allocations),
                    "emitted_parameter_bytes": emitted_parameter_bytes,
                    "source_dtype": source_dtype,
                    "all_source_allocations_emitted_once": True,
                    "physical_aliases_not_duplicated": True,
                    "auxiliary_source_exact": True,
                },
                "numerical_contract": spec.numerical_contract,
                "approximate_quantized": True,
                "execution_certified": False,
                "native_runtime_candidate": True,
                "production_runtime_eligible": False,
            }
            manifest["artifact_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
            _write_durable(staging / "manifest.json", _canonical_json_bytes(manifest))
            _fsync_directory(staging)
            try:
                staging.rename(target)
            except FileExistsError:
                verified = verifier_type(target)
                if verified.build_key_sha256 != build_key:
                    raise _artifact_error(
                        spec, "concurrent build published a different recipe"
                    ) from None
                return _build_values_from_verified(
                    verified, source_artifact_id=artifact.artifact_id
                )
            _fsync_directory(output_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    verified = verifier_type(target)
    if verified.build_key_sha256 != build_key:
        raise _artifact_error(spec, "published artifact lost recipe identity")
    return _build_values_from_verified(verified, source_artifact_id=artifact.artifact_id)


def _q3_record(values: _LowBitBuildValues) -> SourceMlxQ3BuildRecord:
    return SourceMlxQ3BuildRecord(
        path=values.path,
        artifact_sha256=values.artifact_sha256,
        build_key_sha256=values.build_key_sha256,
        source_artifact_id=values.source_artifact_id,
        shard_count=values.shard_count,
        shard_bytes=values.shard_bytes,
        verified_reopen=values.verified_reopen,
        max_abs_error=values.max_abs_error,
        rmse=values.rmse,
    )


def _q2_record(values: _LowBitBuildValues) -> SourceMlxQ2BuildRecord:
    return SourceMlxQ2BuildRecord(
        path=values.path,
        artifact_sha256=values.artifact_sha256,
        build_key_sha256=values.build_key_sha256,
        source_artifact_id=values.source_artifact_id,
        shard_count=values.shard_count,
        shard_bytes=values.shard_bytes,
        verified_reopen=values.verified_reopen,
        max_abs_error=values.max_abs_error,
        rmse=values.rmse,
    )


def build_source_mlx_q3_artifact(
    source_artifact: ComponentArtifact | str | Path,
    output_root: str | Path,
) -> SourceMlxQ3BuildRecord:
    """Build and strictly reopen an approximate affine-q3g64 compiler artifact."""

    return _q3_record(_build_source_mlx_lowbit_artifact(source_artifact, output_root, spec=_Q3))


def build_source_mlx_q2_artifact(
    source_artifact: ComponentArtifact | str | Path,
    output_root: str | Path,
) -> SourceMlxQ2BuildRecord:
    """Build and strictly reopen an approximate affine-q2g64 compiler artifact."""

    return _q2_record(_build_source_mlx_lowbit_artifact(source_artifact, output_root, spec=_Q2))


class MLXSourceQ3Engine(MLXSourceComponentEngine):
    """Experimental affine-q3 Metal executor bound to the q3 artifact schema."""

    backend = "mlx-source-q3"
    artifact_schema = SOURCE_MLX_Q3_NATIVE_SCHEMA
    artifact_builder = staticmethod(build_source_mlx_q3_artifact)
    artifact_verifier = VerifiedSourceMlxQ3Artifact
    artifact_root_default = "~/.cache/mrun/mlx-source-q3"
    approximate_quantized_default = True
    compact_fused_weights_default = True
    experimental_runtime = True

    def _artifact_numerical_contract(self) -> str:
        return SOURCE_MLX_Q3_NUMERICAL_CONTRACT


class MLXSourceQ2Engine(MLXSourceComponentEngine):
    """Experimental affine-q2 Metal executor bound to the q2 artifact schema."""

    backend = "mlx-source-q2"
    artifact_schema = SOURCE_MLX_Q2_NATIVE_SCHEMA
    artifact_builder = staticmethod(build_source_mlx_q2_artifact)
    artifact_verifier = VerifiedSourceMlxQ2Artifact
    artifact_root_default = "~/.cache/mrun/mlx-source-q2"
    approximate_quantized_default = True
    compact_fused_weights_default = True
    experimental_runtime = True

    def _artifact_numerical_contract(self) -> str:
        return SOURCE_MLX_Q2_NUMERICAL_CONTRACT


__all__ = [
    "SOURCE_MLX_Q2_BITS",
    "SOURCE_MLX_Q2_BUILDER_ABI",
    "SOURCE_MLX_Q2_CODEC",
    "SOURCE_MLX_Q2_GROUP_SIZE",
    "SOURCE_MLX_Q2_MODE",
    "SOURCE_MLX_Q2_NATIVE_SCHEMA",
    "SOURCE_MLX_Q2_NUMERICAL_CONTRACT",
    "SOURCE_MLX_Q3_BITS",
    "SOURCE_MLX_Q3_BUILDER_ABI",
    "SOURCE_MLX_Q3_CODEC",
    "SOURCE_MLX_Q3_GROUP_SIZE",
    "SOURCE_MLX_Q3_MODE",
    "SOURCE_MLX_Q3_NATIVE_SCHEMA",
    "SOURCE_MLX_Q3_NUMERICAL_CONTRACT",
    "MLXSourceQ2Engine",
    "MLXSourceQ3Engine",
    "SourceMlxQ2BuildRecord",
    "SourceMlxQ3BuildRecord",
    "VerifiedSourceMlxQ2Artifact",
    "VerifiedSourceMlxQ3Artifact",
    "build_source_mlx_q2_artifact",
    "build_source_mlx_q3_artifact",
]
