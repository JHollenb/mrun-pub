"""Direct canonical-source lowering to role-separated native CUDA int8 components.

This module deliberately does not create, inspect, or consume a QStore.  The durable
representation is the CUDA executor's physical component layout: symmetric per-output-row
int8 codes, FP32 scales, and source-exact FP32 auxiliary vectors split by semantic role.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import tempfile
import threading
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
from .errors import DecompilerError
from .mlx_native import (
    SourceMlxLoweringError,
    _parameter_role,
    _read_allocation,
    _tokenizer_custody_sha256,
    _validated_config,
)
from .reference import ReferenceLoweringError, lower_component_artifact_to_reference

SOURCE_CUDA_INT8_NATIVE_SCHEMA = "mrun-cuda-source-component-int8-native-v1"
SOURCE_CUDA_INT8_BUILDER_ABI = "mrun-canonical-source-to-cuda-int8-native-v1"
SOURCE_CUDA_INT8_MAPPING_ABI = "mrun-qwen2-logical-name-to-cuda-dense-v1"
SOURCE_CUDA_INT8_LAYOUT = "mrun-cuda-source-role-files-v1"
SOURCE_CUDA_INT8_CODEC = "cuda-symmetric-int8-rowwise-fp32-scales-direct-source-v1"
SOURCE_CUDA_INT8_AUX_CODEC = "cuda-fp32-source-value-exact-v1"
SOURCE_CUDA_INT8_NUMERICAL_CONTRACT = (
    "cuda-source-int8-rowwise-symmetric-fp32-scale-bf16-compute-v1"
)
SOURCE_CUDA_INT8_ERROR_BOUND = "abs-error<=row-scale/2+1e-6"

_SOURCE_DTYPES = {"BF16", "F16", "F32"}
_SOURCE_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
_ROLES = frozenset({"body", "norm", "ingress", "egress", "lexical_shared"})
_BLOB_NAMES = frozenset({"weights.i8", "scales.f32", "extras.f32"})
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_RUNTIME_NAME = re.compile(
    r"^(?:embed|lm_head|norm\.final|L[0-9]+\.(?:q|k|v|o|gate|up|down|ln1|ln2)(?:\.bias)?)$"
)
_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


class SourceCudaInt8LoweringError(DecompilerError):
    """The source artifact is outside the direct Qwen2 CUDA int8 support set."""

    code = "source_cuda_int8_lowering_rejection"
    gate = "G9-CUDA"


class SourceCudaInt8ArtifactError(DecompilerError):
    """A direct-source CUDA int8 artifact failed strict verification."""

    code = "source_cuda_int8_artifact_failure"
    gate = "G9-CUDA"


def _is_sha256(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceCudaInt8ArtifactError(f"{field} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceCudaInt8ArtifactError(f"{field} must be a non-negative integer")
    return int(value)


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceCudaInt8ArtifactError(f"{field} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise SourceCudaInt8ArtifactError(f"{field} must be finite and non-negative")
    return result


def _build_lock(key: str) -> threading.Lock:
    with _BUILD_LOCKS_GUARD:
        return _BUILD_LOCKS.setdefault(key, threading.Lock())


def _runtime_name(logical_name: str) -> str:
    if logical_name == "token_embedding.weight":
        return "embed"
    if logical_name == "lm_head.weight":
        return "lm_head"
    if logical_name == "final_norm.weight":
        return "norm.final"
    match = re.fullmatch(r"layers\.(\d+)\.(attention|mlp)\.([a-z_]+)\.(weight|bias)", logical_name)
    if match is None:
        raise SourceCudaInt8LoweringError(
            f"canonical Qwen2 logical parameter is unsupported: {logical_name!r}"
        )
    layer, family, parameter, suffix = match.groups()
    mapped = {
        ("attention", "q_proj"): "q",
        ("attention", "k_proj"): "k",
        ("attention", "v_proj"): "v",
        ("attention", "o_proj"): "o",
        ("mlp", "gate_proj"): "gate",
        ("mlp", "up_proj"): "up",
        ("mlp", "down_proj"): "down",
    }.get((family, parameter))
    if mapped is None:
        raise SourceCudaInt8LoweringError(
            f"canonical Qwen2 logical parameter is unsupported: {logical_name!r}"
        )
    return f"L{layer}.{mapped}" + (".bias" if suffix == "bias" else "")


def _runtime_name_extended(logical_name: str) -> str:
    norm = re.fullmatch(r"layers\.(\d+)\.(attention_norm|mlp_norm)\.weight", logical_name)
    if norm is not None:
        return f"L{norm.group(1)}." + ("ln1" if norm.group(2) == "attention_norm" else "ln2")
    return _runtime_name(logical_name)


@dataclass(frozen=True, slots=True)
class _QuantizedRowMatrix:
    codes: torch.Tensor
    scales: torch.Tensor
    max_abs_error: float
    sum_squared_error: float
    element_count: int
    maximum_row_scale: float
    theoretical_max_abs_error: float


def _quantize_rowwise_int8(tensor: torch.Tensor) -> _QuantizedRowMatrix:
    if tensor.ndim != 2:
        raise TypeError("direct CUDA int8 quantization requires a matrix")
    reference = tensor.float().contiguous()
    if not bool(torch.isfinite(reference).all().item()):
        raise SourceCudaInt8LoweringError("canonical source tensor contains non-finite values")
    row_maximum = reference.abs().amax(dim=1)
    scales = row_maximum / 127.0
    scales = torch.where(row_maximum == 0, torch.ones_like(scales), scales).float().contiguous()
    codes = torch.round(reference / scales[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    restored = codes.float() * scales[:, None]
    error = restored - reference
    maximum = float(error.abs().max().item())
    squared = float(error.double().square().sum().item())
    maximum_scale = float(scales.max().item())
    theoretical = maximum_scale * 0.5 + 1e-6
    if maximum > theoretical:
        raise SourceCudaInt8ArtifactError(
            f"rowwise int8 quantization exceeded its declared bound ({maximum} > {theoretical})"
        )
    return _QuantizedRowMatrix(
        codes=codes,
        scales=scales,
        max_abs_error=maximum,
        sum_squared_error=squared,
        element_count=int(reference.numel()),
        maximum_row_scale=maximum_scale,
        theoretical_max_abs_error=theoretical,
    )


def _effective_config(source: Mapping[str, Any]) -> dict[str, Any]:
    config = json.loads(_canonical_json_bytes(dict(source)))
    config["mrun_native_quantization"] = {
        "codec": SOURCE_CUDA_INT8_CODEC,
        "zero_point": 0,
        "granularity": "per-output-row",
        "scale_dtype": "float32",
        "auxiliary_dtype": "float32",
        "error_bound": SOURCE_CUDA_INT8_ERROR_BOUND,
    }
    return config


def _recipe(
    artifact: ComponentArtifact, *, source_dtype: str, config_sha256: str
) -> dict[str, Any]:
    builder_sha256, _size, _file_identity = _hash_regular_file(Path(__file__).resolve())
    return {
        "schema": SOURCE_CUDA_INT8_NATIVE_SCHEMA,
        "builder_abi": SOURCE_CUDA_INT8_BUILDER_ABI,
        "builder_sha256": builder_sha256,
        "mapping_abi": SOURCE_CUDA_INT8_MAPPING_ABI,
        "layout": SOURCE_CUDA_INT8_LAYOUT,
        "codec": SOURCE_CUDA_INT8_CODEC,
        "auxiliary_codec": SOURCE_CUDA_INT8_AUX_CODEC,
        "quantizer": "torch-rowwise-symmetric-int8-v1",
        "zero_point": 0,
        "scale_dtype": "float32",
        "source_dtype": source_dtype,
        "error_bound": SOURCE_CUDA_INT8_ERROR_BOUND,
        "numerical_contract": SOURCE_CUDA_INT8_NUMERICAL_CONTRACT,
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


@dataclass(frozen=True, slots=True)
class SourceCudaInt8BuildRecord:
    path: Path
    artifact_sha256: str
    build_key_sha256: str
    source_artifact_id: str
    component_count: int
    physical_bytes: int
    max_abs_error: float
    rmse: float
    verified_reopen: bool = True
    direct_from_canonical_source: bool = True
    intermediate_qstore: bool = False
    approximate_quantized: bool = True
    native_runtime_candidate: bool = True
    production_runtime_eligible: bool = False
    schema_version: str = SOURCE_CUDA_INT8_NATIVE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "artifact_sha256": self.artifact_sha256,
            "build_key_sha256": self.build_key_sha256,
            "source_artifact_id": self.source_artifact_id,
            "component_count": self.component_count,
            "physical_bytes": self.physical_bytes,
            "max_abs_error": self.max_abs_error,
            "rmse": self.rmse,
            "verified_reopen": self.verified_reopen,
            "direct_from_canonical_source": self.direct_from_canonical_source,
            "intermediate_qstore": self.intermediate_qstore,
            "approximate_quantized": self.approximate_quantized,
            "native_runtime_candidate": self.native_runtime_candidate,
            "production_runtime_eligible": self.production_runtime_eligible,
        }


def _open_source(value: ComponentArtifact | str | Path) -> ComponentArtifact:
    return value if isinstance(value, ComponentArtifact) else open_component_artifact(value)


class VerifiedSourceCudaInt8Artifact:
    """Immutable verified view of a direct canonical-source CUDA int8 artifact."""

    _PHYSICAL_COMMON = {
        "kind",
        "shape",
        "role",
        "source_allocation_id",
        "source_blob_sha256",
        "source_tensor",
        "source_dtype",
        "source_shape",
        "source_byte_count",
        "logical_names",
    }

    def __init__(
        self,
        path: str | Path,
        *,
        source_artifact: ComponentArtifact | str | Path | None = None,
        verify_quantized_values: bool = True,
    ) -> None:
        self.path = Path(path).expanduser().absolute()
        if self.path.is_symlink() or not self.path.is_dir():
            raise SourceCudaInt8ArtifactError("direct CUDA artifact must be a real directory")
        self.path = self.path.resolve()
        self._directory_identity = _identity(self.path.lstat())
        try:
            manifest, manifest_identity, _manifest_file_hash = _read_regular_json(
                self.path / "manifest.json"
            )
        except Exception as exc:
            raise SourceCudaInt8ArtifactError("cannot verify direct CUDA manifest") from exc
        if manifest.get("schema") != SOURCE_CUDA_INT8_NATIVE_SCHEMA:
            raise SourceCudaInt8ArtifactError("unsupported direct CUDA artifact schema")
        if (
            manifest.get("status") != "native-lowered-approximate-unexecuted"
            or manifest.get("execution_certified") is not False
            or manifest.get("native_runtime_candidate") is not True
            or manifest.get("production_runtime_eligible") is not False
            or manifest.get("approximate_quantized") is not True
            or manifest.get("numerical_contract") != SOURCE_CUDA_INT8_NUMERICAL_CONTRACT
        ):
            raise SourceCudaInt8ArtifactError("direct CUDA promotion boundary is malformed")
        declared_artifact = manifest.get("artifact_sha256")
        if not _is_sha256(declared_artifact):
            raise SourceCudaInt8ArtifactError("direct CUDA artifact identity is malformed")
        unhashed = dict(manifest)
        unhashed.pop("artifact_sha256", None)
        if declared_artifact != _sha256_bytes(_canonical_json_bytes(unhashed)):
            raise SourceCudaInt8ArtifactError("direct CUDA artifact identity mismatch")

        recipe = manifest.get("recipe")
        if not isinstance(recipe, Mapping):
            raise SourceCudaInt8ArtifactError("direct CUDA artifact has no recipe")
        expected_recipe = {
            "schema": SOURCE_CUDA_INT8_NATIVE_SCHEMA,
            "builder_abi": SOURCE_CUDA_INT8_BUILDER_ABI,
            "mapping_abi": SOURCE_CUDA_INT8_MAPPING_ABI,
            "layout": SOURCE_CUDA_INT8_LAYOUT,
            "codec": SOURCE_CUDA_INT8_CODEC,
            "auxiliary_codec": SOURCE_CUDA_INT8_AUX_CODEC,
            "quantizer": "torch-rowwise-symmetric-int8-v1",
            "zero_point": 0,
            "scale_dtype": "float32",
            "error_bound": SOURCE_CUDA_INT8_ERROR_BOUND,
            "numerical_contract": SOURCE_CUDA_INT8_NUMERICAL_CONTRACT,
            "direct_from_canonical_source": True,
            "intermediate_qstore": False,
        }
        if any(recipe.get(key) != value for key, value in expected_recipe.items()):
            raise SourceCudaInt8ArtifactError("direct CUDA recipe ABI is unsupported")
        source_dtype = recipe.get("source_dtype")
        if source_dtype not in _SOURCE_DTYPES or not _is_sha256(recipe.get("builder_sha256")):
            raise SourceCudaInt8ArtifactError("direct CUDA source dtype/builder is unsupported")
        lineage_fields = (
            "source_artifact_id",
            "source_manifest_sha256",
            "source_fingerprint",
            "source_ir_fingerprint",
            "source_io_fingerprint",
            "source_model_fingerprint",
            "source_tokenizer_custody_sha256",
            "effective_config_sha256",
        )
        if any(not _is_sha256(recipe.get(field)) for field in lineage_fields):
            raise SourceCudaInt8ArtifactError("direct CUDA recipe lineage is malformed")
        build_key = _sha256_bytes(_canonical_json_bytes(recipe))
        if manifest.get("build_key_sha256") != build_key:
            raise SourceCudaInt8ArtifactError("direct CUDA build key mismatch")

        config_record = manifest.get("config")
        if not isinstance(config_record, Mapping) or config_record.get("filename") != "config.json":
            raise SourceCudaInt8ArtifactError("direct CUDA config record is malformed")
        try:
            config, config_identity, config_file_hash = _read_regular_json(
                self.path / "config.json"
            )
        except Exception as exc:
            raise SourceCudaInt8ArtifactError("cannot verify direct CUDA config") from exc
        config_semantic_hash = _sha256_bytes(_canonical_json_bytes(config))
        if (
            config_file_hash != config_record.get("file_sha256")
            or int(config_identity[3]) != _positive_int(config_record.get("bytes"), "config bytes")
            or config_semantic_hash != config_record.get("semantic_sha256")
            or config_semantic_hash != recipe.get("effective_config_sha256")
            or config.get("model_type") != "qwen2"
            or config.get("mrun_native_quantization")
            != {
                "codec": SOURCE_CUDA_INT8_CODEC,
                "zero_point": 0,
                "granularity": "per-output-row",
                "scale_dtype": "float32",
                "auxiliary_dtype": "float32",
                "error_bound": SOURCE_CUDA_INT8_ERROR_BOUND,
            }
        ):
            raise SourceCudaInt8ArtifactError("direct CUDA config is not recipe-bound")

        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise SourceCudaInt8ArtifactError("direct CUDA source lineage is malformed")
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
            not _is_sha256(source.get(left)) or source.get(left) != recipe.get(right)
            for left, right in source_recipe_pairs.items()
        ):
            raise SourceCudaInt8ArtifactError("direct CUDA source lineage is not recipe-bound")
        if (
            source.get("artifact_schema") != "mrun-native-source-component-v1"
            or source.get("architecture_id") != "qwen2-dense-causal-decoder"
            or source.get("architecture") != "qwen2"
            or source.get("source_dtype") != source_dtype
            or type(source.get("tied_lexical_allocation")) is not bool
            or source.get("direct_from_canonical_source") is not True
            or source.get("intermediate_qstore") is not False
            or config.get("tie_word_embeddings") != source.get("tied_lexical_allocation")
        ):
            raise SourceCudaInt8ArtifactError("direct CUDA source topology is inconsistent")

        components = manifest.get("components")
        blocks = manifest.get("blocks")
        if not isinstance(components, Mapping) or not components:
            raise SourceCudaInt8ArtifactError("direct CUDA artifact has no components")
        if not isinstance(blocks, Mapping) or not blocks:
            raise SourceCudaInt8ArtifactError("direct CUDA artifact has no block table")
        identities: dict[str, tuple[int, int, int, int, int, int]] = {
            "manifest.json": manifest_identity,
            "config.json": config_identity,
        }
        expected_files = {"manifest.json", "config.json"}
        blob_sizes: dict[tuple[str, str], int] = {}
        physical_bytes = 0
        for role, component in sorted(components.items()):
            if role not in _ROLES or not isinstance(component, Mapping):
                raise SourceCudaInt8ArtifactError("direct CUDA component role is invalid")
            if (
                component.get("component_id") != role
                or component.get("role") != role
                or component.get("layout") != SOURCE_CUDA_INT8_LAYOUT
                or component.get("codec") != SOURCE_CUDA_INT8_CODEC
            ):
                raise SourceCudaInt8ArtifactError("direct CUDA component identity drifted")
            blobs = component.get("blobs")
            if not isinstance(blobs, Mapping) or not blobs:
                raise SourceCudaInt8ArtifactError("direct CUDA component has no blobs")
            if not set(blobs) <= _BLOB_NAMES:
                raise SourceCudaInt8ArtifactError("direct CUDA component blob name is invalid")
            for filename, record in sorted(blobs.items()):
                expected_path = f"components/{role}/{filename}"
                if (
                    not isinstance(record, Mapping)
                    or record.get("path") != expected_path
                    or not _is_sha256(record.get("sha256"))
                ):
                    raise SourceCudaInt8ArtifactError("direct CUDA blob descriptor is malformed")
                size = _positive_int(record.get("bytes"), f"CUDA {expected_path} bytes")
                try:
                    digest, observed_size, identity = _hash_regular_file(
                        self.path / expected_path,
                        expected_size=size,
                        expected_sha256=str(record["sha256"]),
                    )
                except Exception as exc:
                    raise SourceCudaInt8ArtifactError(
                        f"direct CUDA blob hash verification failed: {expected_path}"
                    ) from exc
                if digest != record["sha256"] or observed_size != size:
                    raise SourceCudaInt8ArtifactError("direct CUDA blob identity drifted")
                expected_files.add(expected_path)
                identities[expected_path] = identity
                blob_sizes[(str(role), str(filename))] = size
                physical_bytes += size

        intervals: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
        physical_allocations: dict[str, str] = {}
        runtime_names_by_allocation: dict[str, set[str]] = defaultdict(set)
        source_allocation_bytes = 0
        matrix_elements = 0
        sum_squared_error = 0.0
        maximum_error = 0.0
        qrow_count = auxiliary_count = alias_count = 0
        for name, raw in sorted(blocks.items()):
            if type(name) is not str or _RUNTIME_NAME.fullmatch(name) is None:
                raise SourceCudaInt8ArtifactError("direct CUDA runtime block name is invalid")
            if not isinstance(raw, Mapping):
                raise SourceCudaInt8ArtifactError("direct CUDA block descriptor is malformed")
            if "alias" in raw:
                if set(raw) != {"alias", "role", "source_allocation_id"}:
                    raise SourceCudaInt8ArtifactError("direct CUDA alias descriptor is not exact")
                target = raw.get("alias")
                if (
                    type(target) is not str
                    or target not in blocks
                    or target == name
                    or raw.get("role") not in _ROLES
                    or type(raw.get("source_allocation_id")) is not str
                ):
                    raise SourceCudaInt8ArtifactError("direct CUDA alias target is invalid")
                alias_count += 1
                continue
            kind = raw.get("kind")
            expected_fields = set(self._PHYSICAL_COMMON)
            if kind == "qrow":
                expected_fields |= {
                    "weight_blob",
                    "scale_blob",
                    "w_off",
                    "w_len",
                    "s_off",
                    "s_len",
                    "quantization_error",
                }
            elif kind == "fp32":
                expected_fields |= {"extras_blob", "e_off", "e_len", "source_value_exact"}
            else:
                raise SourceCudaInt8ArtifactError("direct CUDA block kind is unsupported")
            if set(raw) != expected_fields:
                raise SourceCudaInt8ArtifactError("direct CUDA physical descriptor is not exact")
            role = raw.get("role")
            allocation_id = raw.get("source_allocation_id")
            logical_names = raw.get("logical_names")
            shape = raw.get("shape")
            source_shape = raw.get("source_shape")
            if (
                role not in components
                or type(allocation_id) is not str
                or _SAFE_NAME.fullmatch(allocation_id) is None
                or allocation_id in physical_allocations
                or not _is_sha256(raw.get("source_blob_sha256"))
                or type(raw.get("source_tensor")) is not str
                or _SAFE_NAME.fullmatch(str(raw.get("source_tensor"))) is None
                or raw.get("source_dtype") != source_dtype
                or not isinstance(logical_names, list)
                or logical_names != sorted(set(logical_names))
                or not logical_names
                or not isinstance(shape, list)
                or not shape
                or any(type(value) is not int or value <= 0 for value in shape)
                or source_shape != shape
                or _parameter_role(logical_names) != role
                or _positive_int(raw.get("source_byte_count"), "source allocation bytes")
                != math.prod(shape) * _SOURCE_DTYPE_BYTES[str(source_dtype)]
            ):
                raise SourceCudaInt8ArtifactError("direct CUDA block lineage is malformed")
            expected_runtime_names = {_runtime_name_extended(logical) for logical in logical_names}
            if name not in expected_runtime_names:
                raise SourceCudaInt8ArtifactError(
                    "direct CUDA physical name crossed source mapping"
                )
            physical_allocations[str(allocation_id)] = name
            runtime_names_by_allocation[str(allocation_id)].add(name)
            source_allocation_bytes += int(raw["source_byte_count"])
            if kind == "qrow":
                if len(shape) != 2:
                    raise SourceCudaInt8ArtifactError("direct CUDA qrow must be a matrix")
                rows, columns = (int(value) for value in shape)
                weight_blob = raw.get("weight_blob")
                scale_blob = raw.get("scale_blob")
                if weight_blob != "weights.i8" or scale_blob != "scales.f32":
                    raise SourceCudaInt8ArtifactError("direct CUDA qrow blob binding is invalid")
                w_off = _nonnegative_int(raw.get("w_off"), "qrow weight offset")
                w_len = _positive_int(raw.get("w_len"), "qrow weight length")
                s_off = _nonnegative_int(raw.get("s_off"), "qrow scale offset")
                s_len = _positive_int(raw.get("s_len"), "qrow scale length")
                if w_len != rows * columns or s_off % 4 or s_len != rows * 4:
                    raise SourceCudaInt8ArtifactError("direct CUDA qrow span differs from shape")
                intervals[(str(role), "weights.i8")].append((w_off, w_off + w_len, name))
                intervals[(str(role), "scales.f32")].append((s_off, s_off + s_len, name))
                error = raw.get("quantization_error")
                if not isinstance(error, Mapping) or set(error) != {
                    "element_count",
                    "max_abs_error",
                    "sum_squared_error",
                    "rmse",
                    "maximum_row_scale",
                    "theoretical_max_abs_error",
                    "bound",
                }:
                    raise SourceCudaInt8ArtifactError("direct CUDA error evidence is malformed")
                elements = _positive_int(error.get("element_count"), "qrow error elements")
                max_error = _finite_nonnegative(error.get("max_abs_error"), "qrow max error")
                squared_error = _finite_nonnegative(
                    error.get("sum_squared_error"), "qrow squared error"
                )
                rmse = _finite_nonnegative(error.get("rmse"), "qrow RMSE")
                maximum_scale = _finite_nonnegative(
                    error.get("maximum_row_scale"), "qrow maximum scale"
                )
                theoretical = _finite_nonnegative(
                    error.get("theoretical_max_abs_error"), "qrow theoretical error"
                )
                if (
                    elements != rows * columns
                    or error.get("bound") != SOURCE_CUDA_INT8_ERROR_BOUND
                    or not math.isclose(rmse, math.sqrt(squared_error / elements), rel_tol=1e-12)
                    or not math.isclose(theoretical, maximum_scale * 0.5 + 1e-6, rel_tol=1e-12)
                    or max_error > theoretical
                ):
                    raise SourceCudaInt8ArtifactError("direct CUDA error bound is inconsistent")
                matrix_elements += elements
                sum_squared_error += squared_error
                maximum_error = max(maximum_error, max_error)
                qrow_count += 1
            else:
                if len(shape) != 1 or raw.get("extras_blob") != "extras.f32":
                    raise SourceCudaInt8ArtifactError("direct CUDA auxiliary binding is invalid")
                e_off = _nonnegative_int(raw.get("e_off"), "auxiliary offset")
                e_len = _positive_int(raw.get("e_len"), "auxiliary length")
                if (
                    e_off % 4
                    or e_len != math.prod(shape) * 4
                    or raw.get("source_value_exact") is not True
                ):
                    raise SourceCudaInt8ArtifactError("direct CUDA auxiliary span is invalid")
                intervals[(str(role), "extras.f32")].append((e_off, e_off + e_len, name))
                auxiliary_count += 1

        for name, raw in sorted(blocks.items()):
            if "alias" not in raw:
                continue
            seen = {name}
            target = str(raw["alias"])
            while "alias" in blocks[target]:
                if target in seen:
                    raise SourceCudaInt8ArtifactError("direct CUDA alias graph is cyclic")
                seen.add(target)
                target = str(blocks[target]["alias"])
                if target not in blocks:
                    raise SourceCudaInt8ArtifactError("direct CUDA alias target is missing")
            physical = blocks[target]
            if (
                raw["source_allocation_id"] != physical["source_allocation_id"]
                or raw["role"] != physical["role"]
            ):
                raise SourceCudaInt8ArtifactError("direct CUDA alias crossed physical custody")
            runtime_names_by_allocation[str(raw["source_allocation_id"])].add(name)

        for allocation_id, physical_name in physical_allocations.items():
            logical_names = blocks[physical_name]["logical_names"]
            expected_names = {_runtime_name_extended(value) for value in logical_names}
            if runtime_names_by_allocation[allocation_id] != expected_names:
                raise SourceCudaInt8ArtifactError("direct CUDA alias inventory is incomplete")

        if qrow_count <= 0 or auxiliary_count <= 0:
            raise SourceCudaInt8ArtifactError(
                "direct CUDA layout requires qrow and auxiliary blocks"
            )
        if set(intervals) != set(blob_sizes):
            raise SourceCudaInt8ArtifactError("direct CUDA blob/block inventory differs")
        for key, ranges in intervals.items():
            previous = 0
            for start, end, block_name in sorted(ranges):
                if start != previous or end <= start or end > blob_sizes[key]:
                    raise SourceCudaInt8ArtifactError(
                        f"direct CUDA blob coverage is invalid at {block_name!r}"
                    )
                previous = end
            if previous != blob_sizes[key]:
                raise SourceCudaInt8ArtifactError("direct CUDA blob has unreferenced bytes")

        quantization = manifest.get("quantization")
        coverage = manifest.get("coverage")
        if not isinstance(quantization, Mapping) or not isinstance(coverage, Mapping):
            raise SourceCudaInt8ArtifactError("direct CUDA evidence summary is malformed")
        aggregate_rmse = math.sqrt(sum_squared_error / matrix_elements)
        if (
            quantization.get("codec") != SOURCE_CUDA_INT8_CODEC
            or quantization.get("error_bound") != SOURCE_CUDA_INT8_ERROR_BOUND
            or quantization.get("auxiliary_source_value_exact") is not True
            or quantization.get("qrow_count") != qrow_count
            or quantization.get("element_count") != matrix_elements
            or not math.isclose(
                _finite_nonnegative(quantization.get("max_abs_error"), "maximum error"),
                maximum_error,
                rel_tol=1e-12,
            )
            or not math.isclose(
                _finite_nonnegative(quantization.get("rmse"), "aggregate RMSE"),
                aggregate_rmse,
                rel_tol=1e-12,
            )
            or coverage.get("source_allocation_count") != len(physical_allocations)
            or coverage.get("qrow_count") != qrow_count
            or coverage.get("auxiliary_count") != auxiliary_count
            or coverage.get("alias_count") != alias_count
            or coverage.get("source_allocation_bytes") != source_allocation_bytes
            or coverage.get("emitted_physical_bytes") != physical_bytes
            or coverage.get("all_source_allocations_emitted_once") is not True
            or coverage.get("physical_aliases_not_duplicated") is not True
            or coverage.get("all_blob_bytes_covered_once") is not True
            or coverage.get("auxiliary_source_value_exact") is not True
        ):
            raise SourceCudaInt8ArtifactError("direct CUDA coverage/evidence is inconsistent")

        observed_files: set[str] = set()
        expected_directories = {"components", *(f"components/{role}" for role in components)}
        observed_directories: set[str] = set()
        for child in self.path.rglob("*"):
            relative = child.relative_to(self.path).as_posix()
            if child.is_symlink():
                raise SourceCudaInt8ArtifactError("direct CUDA artifact cannot contain symlinks")
            mode = child.lstat().st_mode
            if stat.S_ISDIR(mode):
                observed_directories.add(relative)
            elif stat.S_ISREG(mode):
                observed_files.add(relative)
            else:
                raise SourceCudaInt8ArtifactError("direct CUDA artifact has a non-regular entry")
        if observed_files != expected_files or observed_directories != expected_directories:
            raise SourceCudaInt8ArtifactError("direct CUDA artifact inventory is not exact")

        self.manifest = dict(manifest)
        self.recipe = dict(recipe)
        self.source = dict(source)
        self.config = dict(config)
        self.components = {str(key): dict(value) for key, value in components.items()}
        self.blocks = {str(key): dict(value) for key, value in blocks.items()}
        self.artifact_sha256 = str(declared_artifact)
        self.build_key_sha256 = build_key
        self.physical_bytes = physical_bytes
        self.max_abs_error = maximum_error
        self.rmse = aggregate_rmse
        self._file_identities = identities
        self._source_artifact: ComponentArtifact | None = None
        if source_artifact is not None:
            source_value = _open_source(source_artifact)
            self._verify_source(source_value, verify_values=verify_quantized_values)
            self._source_artifact = source_value

    def _mapped_arrays(self) -> dict[tuple[str, str], np.memmap]:
        arrays: dict[tuple[str, str], np.memmap] = {}
        dtypes = {"weights.i8": np.int8, "scales.f32": np.float32, "extras.f32": np.float32}
        for role, component in self.components.items():
            for filename, record in component["blobs"].items():
                arrays[(role, filename)] = np.memmap(
                    self.path / record["path"], mode="r", dtype=dtypes[filename]
                )
        return arrays

    def _verify_source(self, artifact: ComponentArtifact, *, verify_values: bool) -> None:
        source_pairs = {
            "artifact_id": artifact.artifact_id,
            "manifest_sha256": artifact.manifest_sha256,
            "source_fingerprint": artifact.source.fingerprint,
            "ir_bundle_fingerprint": artifact.ir_bundle.fingerprint,
            "io_fingerprint": artifact.ir_bundle.io.fingerprint,
            "model_fingerprint": artifact.ir_bundle.model.fingerprint,
            "tokenizer_custody_sha256": _tokenizer_custody_sha256(artifact),
        }
        if any(self.source.get(field) != value for field, value in source_pairs.items()):
            raise SourceCudaInt8ArtifactError("canonical source custody differs from CUDA lineage")
        try:
            source_config, architecture, tied = _validated_config(artifact)
        except SourceMlxLoweringError as exc:
            raise SourceCudaInt8ArtifactError(
                "canonical source config is outside the CUDA artifact contract",
                details=exc.details,
            ) from exc
        if architecture != "qwen2" or artifact.ir_bundle.model.architecture_id != (
            "qwen2-dense-causal-decoder"
        ):
            raise SourceCudaInt8ArtifactError("canonical source architecture differs from CUDA")
        if (
            self.config != _effective_config(source_config)
            or tied != self.source["tied_lexical_allocation"]
        ):
            raise SourceCudaInt8ArtifactError("canonical source config differs from CUDA config")
        allocations = artifact.ir_bundle.physical_weights.allocations
        views: dict[str, list[str]] = defaultdict(list)
        for view in artifact.ir_bundle.physical_weights.views:
            views[view.allocation_id].append(view.logical_name)
        physical = {
            str(record["source_allocation_id"]): (name, record)
            for name, record in self.blocks.items()
            if "alias" not in record
        }
        if set(physical) != {item.allocation_id for item in allocations}:
            raise SourceCudaInt8ArtifactError("canonical allocation coverage differs from CUDA")
        source_records = {
            str(record["allocation_id"]): record for record in artifact.manifest["allocations"]
        }
        mapped = self._mapped_arrays() if verify_values else {}
        try:
            for allocation in allocations:
                name, block = physical[allocation.allocation_id]
                logical_names = sorted(views[allocation.allocation_id])
                source_record = source_records[allocation.allocation_id]
                expected_runtime_names = {_runtime_name_extended(value) for value in logical_names}
                if (
                    block["logical_names"] != logical_names
                    or name not in expected_runtime_names
                    or block["source_tensor"] != allocation.source_tensor
                    or block["source_dtype"] != allocation.stored_dtype
                    or block["source_shape"] != list(allocation.stored_shape)
                    or block["source_byte_count"] != allocation.byte_length
                    or block["source_blob_sha256"] != source_record["blob"]["sha256"]
                ):
                    raise SourceCudaInt8ArtifactError(
                        "canonical allocation descriptor differs from CUDA block"
                    )
                if not verify_values:
                    continue
                tensor = _read_allocation(artifact, allocation, source_record)
                role = str(block["role"])
                if block["kind"] == "qrow":
                    quantized = _quantize_rowwise_int8(tensor)
                    rows, columns = (int(value) for value in block["shape"])
                    weight_start = int(block["w_off"])
                    scale_start = int(block["s_off"]) // 4
                    codes = np.asarray(
                        mapped[(role, "weights.i8")][weight_start : weight_start + rows * columns]
                    ).reshape(rows, columns)
                    scales = np.asarray(
                        mapped[(role, "scales.f32")][scale_start : scale_start + rows]
                    )
                    if not np.array_equal(codes, quantized.codes.numpy()) or not np.array_equal(
                        scales, quantized.scales.numpy()
                    ):
                        raise SourceCudaInt8ArtifactError(
                            "CUDA int8 codes/scales differ from deterministic source lowering"
                        )
                    evidence = block["quantization_error"]
                    observed = {
                        "element_count": quantized.element_count,
                        "max_abs_error": quantized.max_abs_error,
                        "sum_squared_error": quantized.sum_squared_error,
                        "rmse": math.sqrt(quantized.sum_squared_error / quantized.element_count),
                        "maximum_row_scale": quantized.maximum_row_scale,
                        "theoretical_max_abs_error": quantized.theoretical_max_abs_error,
                        "bound": SOURCE_CUDA_INT8_ERROR_BOUND,
                    }
                    if (
                        evidence.get("element_count") != observed["element_count"]
                        or evidence.get("bound") != observed["bound"]
                        or any(
                            not math.isclose(
                                float(evidence.get(field, float("nan"))),
                                float(observed[field]),
                                rel_tol=1e-7,
                                abs_tol=1e-12,
                            )
                            for field in (
                                "max_abs_error",
                                "sum_squared_error",
                                "rmse",
                                "maximum_row_scale",
                                "theoretical_max_abs_error",
                            )
                        )
                    ):
                        raise SourceCudaInt8ArtifactError(
                            "CUDA quantization evidence differs from canonical source"
                        )
                else:
                    start = int(block["e_off"]) // 4
                    count = math.prod(block["shape"])
                    stored = np.asarray(mapped[(role, "extras.f32")][start : start + count])
                    if not np.array_equal(stored, tensor.float().reshape(-1).numpy()):
                        raise SourceCudaInt8ArtifactError(
                            "CUDA FP32 auxiliary differs from canonical source values"
                        )
        finally:
            for array in mapped.values():
                handle = getattr(array, "_mmap", None)
                if handle is not None:
                    handle.close()

    def assert_unchanged(self) -> None:
        try:
            if _identity(self.path.lstat()) != self._directory_identity:
                raise SourceCudaInt8ArtifactError("direct CUDA artifact directory changed")
            for relative, expected in self._file_identities.items():
                if _identity((self.path / relative).lstat()) != expected:
                    raise SourceCudaInt8ArtifactError(
                        f"direct CUDA artifact file changed after open: {relative}"
                    )
            if self._source_artifact is not None:
                reopened = open_component_artifact(self._source_artifact.directory)
                if (
                    reopened.artifact_id != self._source_artifact.artifact_id
                    or reopened.manifest_sha256 != self._source_artifact.manifest_sha256
                ):
                    raise SourceCudaInt8ArtifactError("canonical source changed after CUDA open")
        except OSError as exc:
            raise SourceCudaInt8ArtifactError("direct CUDA artifact changed after open") from exc


def build_source_cuda_int8_artifact(
    source_artifact: ComponentArtifact | str | Path,
    output_root: str | Path,
) -> SourceCudaInt8BuildRecord:
    """Lower canonical Qwen2 source allocations directly into native CUDA role files."""

    try:
        reference = lower_component_artifact_to_reference(source_artifact)
    except ReferenceLoweringError as exc:
        raise SourceCudaInt8LoweringError(str(exc), details=exc.details) from exc
    artifact = reference.artifact
    try:
        source_config, architecture, tied = _validated_config(artifact)
    except SourceMlxLoweringError as exc:
        raise SourceCudaInt8LoweringError(
            "canonical artifact architecture has no direct CUDA target",
            details=exc.details,
        ) from exc
    if architecture != "qwen2" or artifact.ir_bundle.model.architecture_id != (
        "qwen2-dense-causal-decoder"
    ):
        raise SourceCudaInt8LoweringError(
            "direct CUDA int8 v1 is intentionally closed to dense Qwen2/Qwen2.5"
        )
    allocations = artifact.ir_bundle.physical_weights.allocations
    dtypes = {item.stored_dtype for item in allocations}
    if len(dtypes) != 1 or next(iter(dtypes)) not in _SOURCE_DTYPES:
        raise SourceCudaInt8LoweringError(
            "direct CUDA int8 requires one canonical BF16, F16, or F32 source dtype"
        )
    source_dtype = next(iter(dtypes))
    views: dict[str, list[str]] = defaultdict(list)
    for view in artifact.ir_bundle.physical_weights.views:
        views[view.allocation_id].append(view.logical_name)
    runtime_names: set[str] = set()
    roles: dict[str, str] = {}
    mapped_names: dict[str, list[str]] = {}
    for allocation in allocations:
        logical_names = sorted(views.get(allocation.allocation_id, ()))
        if not logical_names:
            raise SourceCudaInt8LoweringError("canonical allocation has no logical views")
        names = sorted(_runtime_name_extended(value) for value in logical_names)
        if len(set(names)) != len(names) or runtime_names.intersection(names):
            raise SourceCudaInt8LoweringError("canonical logical names collide in CUDA mapping")
        runtime_names.update(names)
        mapped_names[allocation.allocation_id] = names
        roles[allocation.allocation_id] = _parameter_role(logical_names)
        if roles[allocation.allocation_id] not in _ROLES:
            raise SourceCudaInt8LoweringError("canonical allocation has no CUDA component role")
        if len(allocation.stored_shape) == 2 and not allocation.source_tensor.endswith(".weight"):
            raise SourceCudaInt8LoweringError("two-dimensional CUDA allocation is not a weight")
        if len(allocation.stored_shape) not in {1, 2}:
            raise SourceCudaInt8LoweringError(
                "direct CUDA int8 supports only matrix weights and vector auxiliaries"
            )
    if tied and not any(value == "lexical_shared" for value in roles.values()):
        raise SourceCudaInt8LoweringError("tied lexical allocation lost physical sharing")

    config = _effective_config(source_config)
    config_sha = _sha256_bytes(_canonical_json_bytes(config))
    recipe = _recipe(artifact, source_dtype=source_dtype, config_sha256=config_sha)
    build_key = _sha256_bytes(_canonical_json_bytes(recipe))
    output_root = Path(output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise SourceCudaInt8ArtifactError("direct CUDA output root must be a real directory")
    output_root = output_root.resolve()
    target = output_root / (
        f"{_safe_model_slug(artifact.source.source_id)}-cuda-int8-{build_key[:16]}"
    )

    with _build_lock(build_key):
        if target.exists() or target.is_symlink():
            verified = VerifiedSourceCudaInt8Artifact(target, source_artifact=artifact)
            if verified.build_key_sha256 != build_key:
                raise SourceCudaInt8ArtifactError("existing direct CUDA target has foreign recipe")
            return SourceCudaInt8BuildRecord(
                path=target,
                artifact_sha256=verified.artifact_sha256,
                build_key_sha256=build_key,
                source_artifact_id=artifact.artifact_id,
                component_count=len(verified.components),
                physical_bytes=verified.physical_bytes,
                max_abs_error=verified.max_abs_error,
                rmse=verified.rmse,
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=output_root))
        try:
            config_file_hash, config_bytes = _write_durable(
                staging / "config.json", _canonical_json_bytes(config)
            )
            source_records = {
                str(record["allocation_id"]): record for record in artifact.manifest["allocations"]
            }
            by_role: dict[str, list[Any]] = defaultdict(list)
            for allocation in allocations:
                by_role[roles[allocation.allocation_id]].append(allocation)
            components: dict[str, Any] = {}
            blocks: dict[str, Any] = {}
            total_squared_error = 0.0
            total_elements = 0
            maximum_error = 0.0
            qrow_count = auxiliary_count = alias_count = 0
            source_allocation_bytes = 0
            emitted_physical_bytes = 0
            for role in sorted(by_role):
                directory = staging / "components" / role
                directory.mkdir(parents=True)
                handles: dict[str, Any] = {}
                offsets = {"weights.i8": 0, "scales.f32": 0, "extras.f32": 0}
                try:
                    for allocation in sorted(
                        by_role[role], key=lambda item: mapped_names[item.allocation_id][0]
                    ):
                        record = source_records.get(allocation.allocation_id)
                        if not isinstance(record, Mapping) or not isinstance(
                            record.get("blob"), Mapping
                        ):
                            raise SourceCudaInt8ArtifactError(
                                "canonical source allocation record is missing"
                            )
                        tensor = _read_allocation(artifact, allocation, record)
                        logical_names = sorted(views[allocation.allocation_id])
                        names = mapped_names[allocation.allocation_id]
                        primary = "embed" if "embed" in names else names[0]
                        common = {
                            "shape": list(allocation.stored_shape),
                            "role": role,
                            "source_allocation_id": allocation.allocation_id,
                            "source_blob_sha256": record["blob"]["sha256"],
                            "source_tensor": allocation.source_tensor,
                            "source_dtype": source_dtype,
                            "source_shape": list(allocation.stored_shape),
                            "source_byte_count": allocation.byte_length,
                            "logical_names": logical_names,
                        }
                        source_allocation_bytes += allocation.byte_length
                        if tensor.ndim == 2:
                            quantized = _quantize_rowwise_int8(tensor)
                            for filename, payload in (
                                ("weights.i8", quantized.codes.numpy().tobytes(order="C")),
                                ("scales.f32", quantized.scales.numpy().tobytes(order="C")),
                            ):
                                handle = handles.get(filename)
                                if handle is None:
                                    handle = (directory / filename).open("xb")
                                    handles[filename] = handle
                                handle.write(payload)
                            evidence = {
                                "element_count": quantized.element_count,
                                "max_abs_error": quantized.max_abs_error,
                                "sum_squared_error": quantized.sum_squared_error,
                                "rmse": math.sqrt(
                                    quantized.sum_squared_error / quantized.element_count
                                ),
                                "maximum_row_scale": quantized.maximum_row_scale,
                                "theoretical_max_abs_error": (quantized.theoretical_max_abs_error),
                                "bound": SOURCE_CUDA_INT8_ERROR_BOUND,
                            }
                            blocks[primary] = {
                                "kind": "qrow",
                                **common,
                                "weight_blob": "weights.i8",
                                "scale_blob": "scales.f32",
                                "w_off": offsets["weights.i8"],
                                "w_len": int(quantized.codes.numel()),
                                "s_off": offsets["scales.f32"],
                                "s_len": int(quantized.scales.numel()) * 4,
                                "quantization_error": evidence,
                            }
                            offsets["weights.i8"] += int(quantized.codes.numel())
                            offsets["scales.f32"] += int(quantized.scales.numel()) * 4
                            total_squared_error += quantized.sum_squared_error
                            total_elements += quantized.element_count
                            maximum_error = max(maximum_error, quantized.max_abs_error)
                            qrow_count += 1
                        else:
                            auxiliary = tensor.float().contiguous()
                            if not bool(torch.isfinite(auxiliary).all().item()):
                                raise SourceCudaInt8LoweringError(
                                    "canonical source auxiliary contains non-finite values"
                                )
                            payload = auxiliary.numpy().tobytes(order="C")
                            handle = handles.get("extras.f32")
                            if handle is None:
                                handle = (directory / "extras.f32").open("xb")
                                handles["extras.f32"] = handle
                            handle.write(payload)
                            blocks[primary] = {
                                "kind": "fp32",
                                **common,
                                "extras_blob": "extras.f32",
                                "e_off": offsets["extras.f32"],
                                "e_len": len(payload),
                                "source_value_exact": True,
                            }
                            offsets["extras.f32"] += len(payload)
                            auxiliary_count += 1
                        for alias in names:
                            if alias == primary:
                                continue
                            blocks[alias] = {
                                "alias": primary,
                                "role": role,
                                "source_allocation_id": allocation.allocation_id,
                            }
                            alias_count += 1
                finally:
                    for handle in handles.values():
                        handle.flush()
                        os.fsync(handle.fileno())
                        handle.close()
                blobs: dict[str, Any] = {}
                for filename in sorted(handles):
                    path = directory / filename
                    digest, size, _file_identity = _hash_regular_file(path)
                    blobs[filename] = {
                        "path": f"components/{role}/{filename}",
                        "bytes": size,
                        "sha256": digest,
                    }
                    emitted_physical_bytes += size
                components[role] = {
                    "component_id": role,
                    "role": role,
                    "layout": SOURCE_CUDA_INT8_LAYOUT,
                    "codec": SOURCE_CUDA_INT8_CODEC,
                    "blobs": blobs,
                }
                _fsync_directory(directory)
            _fsync_directory(staging / "components")
            if not qrow_count or not auxiliary_count or total_elements <= 0:
                raise SourceCudaInt8ArtifactError(
                    "direct CUDA lowering did not observe matrices and auxiliaries"
                )
            reopened = open_component_artifact(artifact.directory)
            if (
                reopened.artifact_id != artifact.artifact_id
                or reopened.manifest_sha256 != artifact.manifest_sha256
            ):
                raise SourceCudaInt8ArtifactError("canonical source changed during CUDA lowering")
            source = {
                "artifact_schema": artifact.manifest["schema_version"],
                "artifact_id": artifact.artifact_id,
                "manifest_sha256": artifact.manifest_sha256,
                "source_fingerprint": artifact.source.fingerprint,
                "ir_bundle_fingerprint": artifact.ir_bundle.fingerprint,
                "io_fingerprint": artifact.ir_bundle.io.fingerprint,
                "model_fingerprint": artifact.ir_bundle.model.fingerprint,
                "tokenizer_custody_sha256": _tokenizer_custody_sha256(artifact),
                "architecture_id": artifact.ir_bundle.model.architecture_id,
                "architecture": architecture,
                "source_dtype": source_dtype,
                "tied_lexical_allocation": tied,
                "direct_from_canonical_source": True,
                "intermediate_qstore": False,
            }
            rmse = math.sqrt(total_squared_error / total_elements)
            manifest: dict[str, Any] = {
                "schema": SOURCE_CUDA_INT8_NATIVE_SCHEMA,
                "status": "native-lowered-approximate-unexecuted",
                "build_key_sha256": build_key,
                "recipe": recipe,
                "source": source,
                "config": {
                    "filename": "config.json",
                    "bytes": config_bytes,
                    "file_sha256": config_file_hash,
                    "semantic_sha256": config_sha,
                },
                "components": dict(sorted(components.items())),
                "blocks": dict(sorted(blocks.items())),
                "quantization": {
                    "codec": SOURCE_CUDA_INT8_CODEC,
                    "error_bound": SOURCE_CUDA_INT8_ERROR_BOUND,
                    "qrow_count": qrow_count,
                    "element_count": total_elements,
                    "max_abs_error": maximum_error,
                    "rmse": rmse,
                    "auxiliary_source_value_exact": True,
                },
                "coverage": {
                    "source_allocation_count": len(allocations),
                    "qrow_count": qrow_count,
                    "auxiliary_count": auxiliary_count,
                    "alias_count": alias_count,
                    "source_allocation_bytes": source_allocation_bytes,
                    "emitted_physical_bytes": emitted_physical_bytes,
                    "all_source_allocations_emitted_once": True,
                    "physical_aliases_not_duplicated": True,
                    "all_blob_bytes_covered_once": True,
                    "auxiliary_source_value_exact": True,
                },
                "numerical_contract": SOURCE_CUDA_INT8_NUMERICAL_CONTRACT,
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
                verified = VerifiedSourceCudaInt8Artifact(target, source_artifact=artifact)
                if verified.build_key_sha256 != build_key:
                    raise SourceCudaInt8ArtifactError(
                        "concurrent direct CUDA build published a different recipe"
                    ) from None
                return SourceCudaInt8BuildRecord(
                    path=target,
                    artifact_sha256=verified.artifact_sha256,
                    build_key_sha256=build_key,
                    source_artifact_id=artifact.artifact_id,
                    component_count=len(verified.components),
                    physical_bytes=verified.physical_bytes,
                    max_abs_error=verified.max_abs_error,
                    rmse=verified.rmse,
                )
            _fsync_directory(output_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    verified = VerifiedSourceCudaInt8Artifact(target, source_artifact=artifact)
    if verified.build_key_sha256 != build_key:
        raise SourceCudaInt8ArtifactError("published direct CUDA artifact lost recipe identity")
    return SourceCudaInt8BuildRecord(
        path=target,
        artifact_sha256=verified.artifact_sha256,
        build_key_sha256=build_key,
        source_artifact_id=artifact.artifact_id,
        component_count=len(verified.components),
        physical_bytes=verified.physical_bytes,
        max_abs_error=verified.max_abs_error,
        rmse=verified.rmse,
    )


__all__ = [
    "SOURCE_CUDA_INT8_AUX_CODEC",
    "SOURCE_CUDA_INT8_BUILDER_ABI",
    "SOURCE_CUDA_INT8_CODEC",
    "SOURCE_CUDA_INT8_ERROR_BOUND",
    "SOURCE_CUDA_INT8_LAYOUT",
    "SOURCE_CUDA_INT8_MAPPING_ABI",
    "SOURCE_CUDA_INT8_NATIVE_SCHEMA",
    "SOURCE_CUDA_INT8_NUMERICAL_CONTRACT",
    "SourceCudaInt8ArtifactError",
    "SourceCudaInt8BuildRecord",
    "SourceCudaInt8LoweringError",
    "VerifiedSourceCudaInt8Artifact",
    "build_source_cuda_int8_artifact",
]
