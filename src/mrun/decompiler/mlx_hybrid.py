"""Role-aware direct-source MLX codecs for decomposed dense Qwen2 models.

The artifact deliberately keeps the transformer body in affine q4 while assigning a distinct
codec to the physically shared token embedding/output projection.  The lexical allocation is
either affine q8 or byte-exact BF16.  Norms and other vector auxiliaries always remain byte-exact
BF16.  Tied lexical aliases are emitted once and continue to serve both ingress and egress.
"""

from __future__ import annotations

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
from typing import Any, ClassVar, Literal

import torch
from safetensors import safe_open

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
    SOURCE_MLX_Q4_BITS,
    SOURCE_MLX_Q4_GROUP_SIZE,
    SOURCE_MLX_Q4_MODE,
    SOURCE_MLX_SHARD_BYTES,
    MLXSourceComponentEngine,
    SourceMlxArtifactError,
    SourceMlxLoweringError,
    _build_lock,
    _is_lower_sha256,
    _parameter_role,
    _read_allocation,
    _source_auxiliary_array,
    _strict_positive_int,
    _tokenizer_custody_sha256,
    _validated_config,
)
from .reference import ReferenceLoweringError, lower_component_artifact_to_reference

SOURCE_MLX_HYBRID_NATIVE_SCHEMA = "mrun-mlx-source-component-role-hybrid-native-v1"
SOURCE_MLX_HYBRID_BUILDER_ABI = "mrun-canonical-source-to-mlx-role-hybrid-native-v1"
SOURCE_MLX_HYBRID_Q8_CODEC = "mlx-role-hybrid-body-q4g64-lexical-q8g64-norm-bf16-direct-source-v1"
SOURCE_MLX_HYBRID_BF16_CODEC = "mlx-role-hybrid-body-q4g64-lexical-bf16-norm-bf16-direct-source-v1"
SOURCE_MLX_HYBRID_Q8_NUMERICAL_CONTRACT = (
    "mlx-source-role-hybrid-body-q4g64-lexical-q8g64-norm-bf16-v1"
)
SOURCE_MLX_HYBRID_BF16_NUMERICAL_CONTRACT = (
    "mlx-source-role-hybrid-body-q4g64-lexical-bf16-norm-bf16-v1"
)
SOURCE_MLX_HYBRID_LEXICAL_Q8_BITS = 8
SOURCE_MLX_HYBRID_LEXICAL_PRECISIONS = frozenset({"q8", "bf16"})
_SAFE_TENSOR_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_DTYPE_BITS = {"BF16": 16, "F16": 16, "F32": 32, "F64": 64}

LexicalPrecision = Literal["q8", "bf16"]


@dataclass(frozen=True, slots=True)
class _HybridProfile:
    lexical_precision: LexicalPrecision
    codec: str
    numerical_contract: str

    @property
    def lexical_encoding(self) -> str:
        return "mlx-affine-q8-g64" if self.lexical_precision == "q8" else "source-exact-bf16"

    @property
    def lexical_bits(self) -> int:
        return 8 if self.lexical_precision == "q8" else 16


_PROFILES: dict[str, _HybridProfile] = {
    "q8": _HybridProfile("q8", SOURCE_MLX_HYBRID_Q8_CODEC, SOURCE_MLX_HYBRID_Q8_NUMERICAL_CONTRACT),
    "bf16": _HybridProfile(
        "bf16",
        SOURCE_MLX_HYBRID_BF16_CODEC,
        SOURCE_MLX_HYBRID_BF16_NUMERICAL_CONTRACT,
    ),
}


def _profile(lexical_precision: str) -> _HybridProfile:
    try:
        return _PROFILES[lexical_precision]
    except KeyError as exc:
        raise SourceMlxLoweringError(
            "hybrid lexical precision must be exactly 'q8' or 'bf16'"
        ) from exc


def _hybrid_quantization(profile: _HybridProfile) -> dict[str, Any]:
    quantization: dict[str, Any] = {
        "bits": SOURCE_MLX_Q4_BITS,
        "group_size": SOURCE_MLX_Q4_GROUP_SIZE,
        "mode": SOURCE_MLX_Q4_MODE,
    }
    quantization["model.embed_tokens"] = (
        {
            "bits": SOURCE_MLX_HYBRID_LEXICAL_Q8_BITS,
            "group_size": SOURCE_MLX_Q4_GROUP_SIZE,
            "mode": SOURCE_MLX_Q4_MODE,
        }
        if profile.lexical_precision == "q8"
        else False
    )
    return quantization


def _hybrid_config(config: Mapping[str, Any], profile: _HybridProfile) -> dict[str, Any]:
    effective = json.loads(_canonical_json_bytes(dict(config)))
    quantization = _hybrid_quantization(profile)
    effective["quantization"] = json.loads(_canonical_json_bytes(quantization))
    effective["quantization_config"] = json.loads(_canonical_json_bytes(quantization))
    effective.pop("quantize_activations", None)
    return effective


def _hybrid_recipe(
    artifact: ComponentArtifact,
    *,
    profile: _HybridProfile,
    config_sha256: str,
) -> dict[str, Any]:
    import importlib.metadata

    builder_sha256, _size, _identity_value = _hash_regular_file(Path(__file__).resolve())
    return {
        "schema": SOURCE_MLX_HYBRID_NATIVE_SCHEMA,
        "builder_abi": SOURCE_MLX_HYBRID_BUILDER_ABI,
        "builder_sha256": builder_sha256,
        "mapping_abi": SOURCE_MLX_MAPPING_ABI,
        "codec": profile.codec,
        "role_codecs": {
            "body": "mlx-affine-q4-g64",
            "lexical_shared": profile.lexical_encoding,
            "norm": "source-exact-bf16",
        },
        "body_bits": SOURCE_MLX_Q4_BITS,
        "lexical_bits": profile.lexical_bits,
        "lexical_precision": profile.lexical_precision,
        "group_size": SOURCE_MLX_Q4_GROUP_SIZE,
        "mode": SOURCE_MLX_Q4_MODE,
        "shard_bytes": SOURCE_MLX_SHARD_BYTES,
        "quantizer": "mlx.core.quantize",
        "quantizer_input_dtype": "bfloat16",
        "source_dtype": "BF16",
        "mlx_version": importlib.metadata.version("mlx"),
        "numerical_contract": profile.numerical_contract,
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
class SourceMlxHybridBuildRecord:
    path: Path
    artifact_sha256: str
    build_key_sha256: str
    source_artifact_id: str
    shard_count: int
    shard_bytes: int
    verified_reopen: bool
    lexical_precision: LexicalPrecision
    body_max_abs_error: float
    body_rmse: float
    lexical_max_abs_error: float
    lexical_rmse: float
    direct_from_canonical_source: bool = True
    intermediate_qstore: bool = False
    approximate_quantized: bool = True
    native_runtime_candidate: bool = True
    production_runtime_eligible: bool = False
    schema_version: str = SOURCE_MLX_HYBRID_NATIVE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "artifact_sha256": self.artifact_sha256,
            "build_key_sha256": self.build_key_sha256,
            "source_artifact_id": self.source_artifact_id,
            "shard_count": self.shard_count,
            "shard_bytes": self.shard_bytes,
            "verified_reopen": self.verified_reopen,
            "lexical_precision": self.lexical_precision,
            "body_max_abs_error": self.body_max_abs_error,
            "body_rmse": self.body_rmse,
            "lexical_max_abs_error": self.lexical_max_abs_error,
            "lexical_rmse": self.lexical_rmse,
            "direct_from_canonical_source": self.direct_from_canonical_source,
            "intermediate_qstore": self.intermediate_qstore,
            "approximate_quantized": self.approximate_quantized,
            "native_runtime_candidate": self.native_runtime_candidate,
            "production_runtime_eligible": self.production_runtime_eligible,
        }


@dataclass(frozen=True, slots=True)
class _AffineResult:
    weight: Any
    scales: Any
    biases: Any
    max_abs_error: float
    sum_squared_error: float
    elements: int


def _quantize_affine(tensor: torch.Tensor, *, bits: int, mx: Any) -> _AffineResult:
    if tensor.ndim != 2 or bits not in {4, 8}:
        raise TypeError("hybrid affine source tensor must be a matrix with 4 or 8 bits")
    rows, columns = (int(value) for value in tensor.shape)
    if columns % SOURCE_MLX_Q4_GROUP_SIZE:
        raise SourceMlxLoweringError(
            f"hybrid source width {columns} is not divisible by {SOURCE_MLX_Q4_GROUP_SIZE}"
        )
    if not bool(torch.isfinite(tensor.float()).all().item()):
        raise SourceMlxLoweringError("canonical source tensor contains non-finite values")
    reference = mx.array(tensor.float().numpy()).astype(mx.float32)
    weight, scales, biases = mx.quantize(
        reference.astype(mx.bfloat16),
        group_size=SOURCE_MLX_Q4_GROUP_SIZE,
        bits=bits,
        mode=SOURCE_MLX_Q4_MODE,
    )
    restored = mx.dequantize(
        weight,
        scales,
        biases,
        group_size=SOURCE_MLX_Q4_GROUP_SIZE,
        bits=bits,
        mode=SOURCE_MLX_Q4_MODE,
    )
    error = restored.astype(mx.float32) - reference
    maximum = mx.max(mx.abs(error))
    squared = mx.sum(mx.square(error))
    mx.eval(weight, scales, biases, maximum, squared)
    max_abs_error = float(maximum.item())
    sum_squared_error = float(squared.item())
    if not math.isfinite(max_abs_error) or not math.isfinite(sum_squared_error):
        raise SourceMlxArtifactError("hybrid affine error evidence is non-finite")
    return _AffineResult(
        weight=weight,
        scales=scales,
        biases=biases,
        max_abs_error=max_abs_error,
        sum_squared_error=sum_squared_error,
        elements=rows * columns,
    )


class VerifiedSourceMlxHybridArtifact:
    """Strict immutable view of one role-aware direct-source hybrid artifact."""

    expected_lexical_precision: ClassVar[LexicalPrecision | None] = None
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
        self.path = Path(path).expanduser().absolute()
        if self.path.is_symlink() or not self.path.is_dir():
            raise SourceMlxArtifactError("direct-source hybrid artifact must be a real directory")
        self.path = self.path.resolve()
        self._directory_identity = _identity(self.path.lstat())
        try:
            manifest, manifest_identity, _manifest_sha = _read_regular_json(
                self.path / "manifest.json"
            )
        except Exception as exc:
            raise SourceMlxArtifactError("cannot verify direct-source hybrid manifest") from exc
        if manifest.get("schema") != SOURCE_MLX_HYBRID_NATIVE_SCHEMA:
            raise SourceMlxArtifactError("unsupported direct-source hybrid artifact schema")
        recipe = manifest.get("recipe")
        if not isinstance(recipe, Mapping):
            raise SourceMlxArtifactError("direct-source hybrid artifact has no recipe")
        lexical_precision = recipe.get("lexical_precision")
        if lexical_precision not in SOURCE_MLX_HYBRID_LEXICAL_PRECISIONS:
            raise SourceMlxArtifactError("hybrid artifact lexical precision is unsupported")
        profile = _PROFILES[str(lexical_precision)]
        if (
            self.expected_lexical_precision is not None
            and profile.lexical_precision != self.expected_lexical_precision
        ):
            raise SourceMlxArtifactError("hybrid artifact lexical precision differs from backend")
        if (
            manifest.get("status") != "native-lowered-approximate-unexecuted"
            or manifest.get("execution_certified") is not False
            or manifest.get("native_runtime_candidate") is not True
            or manifest.get("production_runtime_eligible") is not False
            or manifest.get("approximate_quantized") is not True
            or manifest.get("numerical_contract") != profile.numerical_contract
        ):
            raise SourceMlxArtifactError("direct-source hybrid promotion boundary is malformed")
        declared = manifest.get("artifact_sha256")
        if not _is_lower_sha256(declared):
            raise SourceMlxArtifactError("direct-source hybrid artifact identity is malformed")
        unhashed = dict(manifest)
        unhashed.pop("artifact_sha256", None)
        if declared != _sha256_bytes(_canonical_json_bytes(unhashed)):
            raise SourceMlxArtifactError("direct-source hybrid artifact identity mismatch")
        expected_recipe = {
            "schema": SOURCE_MLX_HYBRID_NATIVE_SCHEMA,
            "builder_abi": SOURCE_MLX_HYBRID_BUILDER_ABI,
            "mapping_abi": SOURCE_MLX_MAPPING_ABI,
            "codec": profile.codec,
            "role_codecs": {
                "body": "mlx-affine-q4-g64",
                "lexical_shared": profile.lexical_encoding,
                "norm": "source-exact-bf16",
            },
            "body_bits": 4,
            "lexical_bits": profile.lexical_bits,
            "group_size": 64,
            "mode": "affine",
            "shard_bytes": SOURCE_MLX_SHARD_BYTES,
            "quantizer": "mlx.core.quantize",
            "quantizer_input_dtype": "bfloat16",
            "source_dtype": "BF16",
            "numerical_contract": profile.numerical_contract,
            "direct_from_canonical_source": True,
            "intermediate_qstore": False,
        }
        if any(recipe.get(key) != value for key, value in expected_recipe.items()):
            raise SourceMlxArtifactError("direct-source hybrid recipe ABI is unsupported")
        if (
            not _is_lower_sha256(recipe.get("builder_sha256"))
            or type(recipe.get("mlx_version")) is not str
            or not recipe.get("mlx_version")
        ):
            raise SourceMlxArtifactError("direct-source hybrid quantizer identity is malformed")
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
        if any(not _is_lower_sha256(recipe.get(field)) for field in lineage_fields):
            raise SourceMlxArtifactError("direct-source hybrid recipe lineage is malformed")
        build_key = _sha256_bytes(_canonical_json_bytes(recipe))
        if manifest.get("build_key_sha256") != build_key:
            raise SourceMlxArtifactError("direct-source hybrid build key mismatch")

        config_record = manifest.get("config")
        if not isinstance(config_record, Mapping) or config_record.get("filename") != "config.json":
            raise SourceMlxArtifactError("direct-source hybrid config record is malformed")
        try:
            config, config_identity, config_file_sha = _read_regular_json(self.path / "config.json")
        except Exception as exc:
            raise SourceMlxArtifactError("cannot verify direct-source hybrid config") from exc
        config_sha = _sha256_bytes(_canonical_json_bytes(config))
        expected_quantization = _hybrid_quantization(profile)
        if (
            config_file_sha != config_record.get("file_sha256")
            or _strict_positive_int(config_record.get("bytes"), "hybrid config bytes")
            != int(config_identity[3])
            or config_sha != config_record.get("semantic_sha256")
            or config_sha != recipe.get("effective_config_sha256")
            or config.get("quantization") != expected_quantization
            or config.get("quantization_config") != expected_quantization
        ):
            raise SourceMlxArtifactError("direct-source hybrid config is not recipe-bound")

        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise SourceMlxArtifactError("direct-source hybrid source record is malformed")
        source_pairs = {
            "artifact_id": "source_artifact_id",
            "manifest_sha256": "source_manifest_sha256",
            "source_fingerprint": "source_fingerprint",
            "ir_bundle_fingerprint": "source_ir_fingerprint",
            "io_fingerprint": "source_io_fingerprint",
            "model_fingerprint": "source_model_fingerprint",
            "tokenizer_custody_sha256": "source_tokenizer_custody_sha256",
        }
        if any(
            not _is_lower_sha256(source.get(left)) or source.get(left) != recipe.get(right)
            for left, right in source_pairs.items()
        ):
            raise SourceMlxArtifactError("direct-source hybrid source lineage is not recipe-bound")
        if (
            source.get("architecture_id") != "qwen2-dense-causal-decoder"
            or source.get("architecture") != "qwen2"
            or source.get("source_dtype") != "BF16"
            or source.get("tied_lexical_allocation") is not True
            or source.get("direct_from_canonical_source") is not True
            or source.get("intermediate_qstore") is not False
            or config.get("model_type") != "qwen2"
            or config.get("tie_word_embeddings") is not True
        ):
            raise SourceMlxArtifactError("direct-source hybrid topology lineage is inconsistent")

        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise SourceMlxArtifactError("direct-source hybrid artifact has no shards")
        identities = {"manifest.json": manifest_identity, "config.json": config_identity}
        observed_files: set[str] = set()
        observed_names: set[str] = set()
        observed_roles: set[str] = set()
        allocation_entries: dict[str, list[tuple[Mapping[str, Any], torch.Tensor, str]]] = (
            defaultdict(list)
        )
        shard_bytes = 0
        emitted_bytes = 0
        for shard in shards:
            if not isinstance(shard, Mapping):
                raise SourceMlxArtifactError("direct-source hybrid shard record is malformed")
            filename, role = shard.get("filename"), shard.get("role")
            if (
                type(filename) is not str
                or re.fullmatch(r"model-[a-z_]+-[0-9]{5}\.safetensors", filename) is None
                or filename in observed_files
                or role not in {"body", "norm", "lexical_shared"}
                or not _is_lower_sha256(shard.get("sha256"))
            ):
                raise SourceMlxArtifactError("direct-source hybrid shard identity is invalid")
            expected_size = _strict_positive_int(shard.get("bytes"), "hybrid shard bytes")
            try:
                digest, size, identity = _hash_regular_file(
                    self.path / filename,
                    expected_size=expected_size,
                    expected_sha256=str(shard["sha256"]),
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise SourceMlxArtifactError("direct-source hybrid shard hash failed") from exc
            if digest != shard["sha256"]:
                raise SourceMlxArtifactError("direct-source hybrid shard hash drifted")
            parameters = shard.get("parameters")
            if not isinstance(parameters, list) or not parameters:
                raise SourceMlxArtifactError("direct-source hybrid shard has no parameters")
            with safe_open(self.path / filename, framework="pt", device="cpu") as handle:
                names = [str(item.get("name")) for item in parameters if isinstance(item, Mapping)]
                metadata = handle.metadata() or {}
                if (
                    len(names) != len(parameters)
                    or len(set(names)) != len(names)
                    or set(names) != set(handle.keys())
                    or metadata.get("format") != "mlx"
                    or metadata.get("mrun-codec") != profile.codec
                    or metadata.get("mrun-role") != role
                    or metadata.get("mrun-source-artifact") != source["artifact_id"]
                ):
                    raise SourceMlxArtifactError("direct-source hybrid shard header drifted")
                for item in parameters:
                    if not isinstance(item, Mapping) or set(item) != self._PARAMETER_FIELDS:
                        raise SourceMlxArtifactError("hybrid parameter inventory is not exact")
                    name = item["name"]
                    allocation_id = item["source_allocation_id"]
                    logical_names = item["logical_names"]
                    source_shape = item["source_shape"]
                    shape = item["shape"]
                    if (
                        type(name) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(name) is None
                        or name in observed_names
                        or type(allocation_id) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(allocation_id) is None
                        or not _is_lower_sha256(item["source_blob_sha256"])
                        or type(item["source_tensor"]) is not str
                        or _SAFE_TENSOR_NAME.fullmatch(item["source_tensor"]) is None
                        or item["source_dtype"] != "BF16"
                        or not isinstance(logical_names, list)
                        or logical_names != sorted(set(logical_names))
                        or not logical_names
                        or not isinstance(source_shape, list)
                        or not source_shape
                        or any(type(value) is not int or value <= 0 for value in source_shape)
                        or not isinstance(shape, list)
                        or not shape
                        or any(type(value) is not int or value <= 0 for value in shape)
                        or _strict_positive_int(
                            item["source_byte_count"], "hybrid source parameter bytes"
                        )
                        != math.prod(source_shape) * 2
                        or _parameter_role(logical_names) != role
                    ):
                        raise SourceMlxArtifactError("hybrid parameter descriptor is malformed")
                    value = handle.get_tensor(name)
                    dtypes = {
                        "U32": torch.uint32,
                        "BF16": torch.bfloat16,
                    }
                    if (
                        item["dtype"] not in dtypes
                        or value.dtype != dtypes[item["dtype"]]
                        or list(value.shape) != shape
                    ):
                        raise SourceMlxArtifactError("hybrid parameter shape/dtype drifted")
                    observed_names.add(name)
                    emitted_bytes += value.numel() * value.element_size()
                    allocation_entries[str(allocation_id)].append((item, value, str(role)))
            identities[str(filename)] = identity
            observed_files.add(str(filename))
            observed_roles.add(str(role))
            shard_bytes += size

        if observed_roles != {"body", "norm", "lexical_shared"}:
            raise SourceMlxArtifactError("direct-source hybrid physical roles are incomplete")
        q4_ids: set[str] = set()
        q8_ids: set[str] = set()
        exact_matrix_ids: set[str] = set()
        auxiliary_ids: set[str] = set()
        source_bytes = 0
        allocation_facts: dict[str, dict[str, Any]] = {}
        for allocation_id, entries in allocation_entries.items():
            first = entries[0][0]
            lineage = (
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
                for field in lineage
            ):
                raise SourceMlxArtifactError("hybrid allocation descriptors disagree")
            role = entries[0][2]
            shape = [int(value) for value in first["source_shape"]]
            source_tensor = str(first["source_tensor"])
            source_bytes += int(first["source_byte_count"])
            allocation_facts[allocation_id] = {
                "source_tensor": source_tensor,
                "logical_names": first["logical_names"],
                "role": role,
                "shape": shape,
            }
            encoding = first["encoding"]
            if encoding in {"mlx-affine-q4-g64", "mlx-affine-q8-g64"}:
                bits = 4 if encoding == "mlx-affine-q4-g64" else 8
                if (
                    len(shape) != 2
                    or not source_tensor.endswith(".weight")
                    or role != ("body" if bits == 4 else "lexical_shared")
                    or (bits == 8 and profile.lexical_precision != "q8")
                ):
                    raise SourceMlxArtifactError("hybrid affine encoding crossed its role")
                rows, columns = shape
                base = source_tensor.removesuffix(".weight")
                expected = {
                    f"{base}.weight": ("weight", "U32", [rows, columns * bits // 32]),
                    f"{base}.scales": ("scales", "BF16", [rows, columns // 64]),
                    f"{base}.biases": ("biases", "BF16", [rows, columns // 64]),
                }
                observed = {
                    str(item["name"]): (item["part"], item["dtype"], item["shape"])
                    for item, _value, _role in entries
                }
                if len(entries) != 3 or observed != expected:
                    raise SourceMlxArtifactError("hybrid affine packed triple is incomplete")
                (q4_ids if bits == 4 else q8_ids).add(allocation_id)
            elif encoding == "source-exact-bf16":
                if (
                    len(entries) != 1
                    or first["name"] != source_tensor
                    or first["part"] != "source_exact"
                    or first["dtype"] != "BF16"
                    or first["shape"] != shape
                ):
                    raise SourceMlxArtifactError("hybrid source-exact tensor drifted")
                if len(shape) == 2:
                    if role != "lexical_shared" or profile.lexical_precision != "bf16":
                        raise SourceMlxArtifactError("source-exact matrix crossed hybrid role")
                    exact_matrix_ids.add(allocation_id)
                elif len(shape) == 1:
                    auxiliary_ids.add(allocation_id)
                else:
                    raise SourceMlxArtifactError("hybrid source-exact rank is unsupported")
            else:
                raise SourceMlxArtifactError("hybrid allocation encoding is unknown")

        if len(q8_ids | exact_matrix_ids) != 1:
            raise SourceMlxArtifactError("hybrid artifact must have one tied lexical allocation")
        coverage = manifest.get("coverage")
        if (
            not isinstance(coverage, Mapping)
            or coverage.get("source_dtype") != "BF16"
            or coverage.get("all_source_allocations_emitted_once") is not True
            or coverage.get("physical_aliases_not_duplicated") is not True
            or coverage.get("norms_source_exact") is not True
            or coverage.get("source_allocation_count") != len(allocation_entries)
            or coverage.get("body_q4_allocation_count") != len(q4_ids)
            or coverage.get("lexical_q8_allocation_count") != len(q8_ids)
            or coverage.get("lexical_bf16_allocation_count") != len(exact_matrix_ids)
            or coverage.get("auxiliary_source_exact_count") != len(auxiliary_ids)
            or coverage.get("emitted_parameter_count") != len(observed_names)
            or coverage.get("source_allocation_bytes") != source_bytes
            or coverage.get("emitted_parameter_bytes") != emitted_bytes
        ):
            raise SourceMlxArtifactError("hybrid coverage claim is inconsistent")

        quantization = manifest.get("quantization")
        blocks = quantization.get("blocks") if isinstance(quantization, Mapping) else None
        if not isinstance(blocks, list):
            raise SourceMlxArtifactError("hybrid quantization evidence is malformed")
        expected_quantized_ids = q4_ids | q8_ids
        block_ids: set[str] = set()
        sums = {"body": 0.0, "lexical_shared": 0.0}
        elements = {"body": 0, "lexical_shared": 0}
        maxima = {"body": 0.0, "lexical_shared": 0.0}
        for block in blocks:
            if not isinstance(block, Mapping):
                raise SourceMlxArtifactError("hybrid quantization block is malformed")
            allocation_id = str(block.get("source_allocation_id", ""))
            facts = allocation_facts.get(allocation_id)
            count = block.get("elements")
            maximum = block.get("max_abs_error")
            squared = block.get("sum_squared_error")
            rmse = block.get("rmse")
            role = block.get("role")
            bits = block.get("bits")
            if (
                allocation_id not in expected_quantized_ids
                or allocation_id in block_ids
                or facts is None
                or role != facts["role"]
                or bits != (4 if role == "body" else 8)
                or block.get("source_tensor") != facts["source_tensor"]
                or block.get("logical_names") != facts["logical_names"]
                or type(count) is not int
                or count <= 0
                or type(maximum) not in {int, float}
                or type(squared) not in {int, float}
                or type(rmse) not in {int, float}
                or not all(
                    math.isfinite(float(v)) and float(v) >= 0 for v in (maximum, squared, rmse)
                )
                or not math.isclose(
                    float(rmse), math.sqrt(float(squared) / count), rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise SourceMlxArtifactError("hybrid quantization block is inconsistent")
            block_ids.add(allocation_id)
            sums[str(role)] += float(squared)
            elements[str(role)] += count
            maxima[str(role)] = max(maxima[str(role)], float(maximum))
        if block_ids != expected_quantized_ids:
            raise SourceMlxArtifactError("hybrid quantization block coverage is incomplete")
        body_rmse = math.sqrt(sums["body"] / elements["body"])
        lexical_rmse = (
            math.sqrt(sums["lexical_shared"] / elements["lexical_shared"])
            if elements["lexical_shared"]
            else 0.0
        )
        declared_summaries = quantization.get("roles")
        expected_summaries = {
            "body": {
                "codec": "mlx-affine-q4-g64",
                "allocation_count": len(q4_ids),
                "elements": elements["body"],
                "max_abs_error": maxima["body"],
                "rmse": body_rmse,
            },
            "lexical_shared": {
                "codec": profile.lexical_encoding,
                "allocation_count": len(q8_ids | exact_matrix_ids),
                "elements": elements["lexical_shared"],
                "max_abs_error": maxima["lexical_shared"],
                "rmse": lexical_rmse,
            },
            "norm": {
                "codec": "source-exact-bf16",
                "max_abs_error": 0.0,
                "rmse": 0.0,
            },
        }
        if _canonical_json_bytes(declared_summaries) != _canonical_json_bytes(expected_summaries):
            raise SourceMlxArtifactError("hybrid role error summaries are inconsistent")
        if {entry.name for entry in self.path.iterdir()} != set(identities):
            raise SourceMlxArtifactError("direct-source hybrid artifact has undeclared files")

        self.manifest = dict(manifest)
        self.config = config
        self.source = dict(source)
        self.artifact_sha256 = str(declared)
        self.build_key_sha256 = build_key
        self.codec = profile.codec
        self.numerical_contract = profile.numerical_contract
        self.lexical_precision = profile.lexical_precision
        self.bits = 4
        self.body_bits = 4
        self.lexical_bits = profile.lexical_bits
        self.source_dtype = "BF16"
        self.shard_bytes = shard_bytes
        self.body_max_abs_error = maxima["body"]
        self.body_rmse = body_rmse
        self.lexical_max_abs_error = maxima["lexical_shared"]
        self.lexical_rmse = lexical_rmse
        self._identities = identities

    def assert_unchanged(self) -> None:
        if _identity(self.path.lstat()) != self._directory_identity:
            raise SourceMlxArtifactError("direct-source hybrid directory identity changed")
        if {entry.name for entry in self.path.iterdir()} != set(self._identities):
            raise SourceMlxArtifactError("direct-source hybrid file inventory changed")
        for filename, identity in self._identities.items():
            if _identity((self.path / filename).lstat()) != identity:
                raise SourceMlxArtifactError(f"direct-source hybrid file changed: {filename}")


class VerifiedSourceMlxHybridQ8Artifact(VerifiedSourceMlxHybridArtifact):
    expected_lexical_precision = "q8"


class VerifiedSourceMlxHybridBF16Artifact(VerifiedSourceMlxHybridArtifact):
    expected_lexical_precision = "bf16"


def build_source_mlx_hybrid_artifact(
    source_artifact: ComponentArtifact | str | Path,
    output_root: str | Path,
    *,
    lexical_precision: LexicalPrecision = "q8",
) -> SourceMlxHybridBuildRecord:
    """Lower tied Qwen2 BF16 source to q4-body/q8-or-BF16-lexical MLX weights."""

    profile = _profile(lexical_precision)
    try:
        reference = lower_component_artifact_to_reference(source_artifact)
    except ReferenceLoweringError as exc:
        raise SourceMlxLoweringError(str(exc), details=exc.details) from exc
    artifact = reference.artifact
    source_config, architecture, tied = _validated_config(artifact)
    if architecture != "qwen2" or not tied:
        raise SourceMlxLoweringError(
            "role-hybrid lowering is certified only for tied dense Qwen2 artifacts"
        )
    allocations = artifact.ir_bundle.physical_weights.allocations
    if {item.stored_dtype for item in allocations} != {"BF16"}:
        raise SourceMlxLoweringError("role-hybrid lowering requires uniform canonical BF16")
    if any(".rotary_emb." in item.source_tensor for item in allocations):
        raise SourceMlxLoweringError("serialized RoPE is not registered in the hybrid target")
    views: dict[str, list[str]] = defaultdict(list)
    for view in artifact.ir_bundle.physical_weights.views:
        views[view.allocation_id].append(view.logical_name)
    roles: dict[str, str] = {}
    for allocation in allocations:
        logical_names = sorted(views.get(allocation.allocation_id, []))
        if not logical_names:
            raise SourceMlxLoweringError("canonical allocation has no logical views")
        role = _parameter_role(logical_names)
        roles[allocation.allocation_id] = role
        if role not in {"body", "norm", "lexical_shared"}:
            raise SourceMlxLoweringError("hybrid allocation crossed the tied-role boundary")
        if len(allocation.stored_shape) == 2:
            if not allocation.source_tensor.endswith(".weight"):
                raise SourceMlxLoweringError("hybrid matrix allocation is not a weight")
            if int(allocation.stored_shape[1]) % 64:
                raise SourceMlxLoweringError("hybrid matrix width must be divisible by 64")
        elif len(allocation.stored_shape) != 1:
            raise SourceMlxLoweringError("hybrid target supports only matrices and vectors")
    if list(roles.values()).count("lexical_shared") != 1:
        raise SourceMlxLoweringError("hybrid target requires one physically tied lexical matrix")
    if set(roles.values()) != {"body", "norm", "lexical_shared"}:
        raise SourceMlxLoweringError("hybrid target requires body, norm, and lexical roles")

    config = _hybrid_config(source_config, profile)
    config_sha = _sha256_bytes(_canonical_json_bytes(config))
    try:
        recipe = _hybrid_recipe(artifact, profile=profile, config_sha256=config_sha)
        import mlx.core as mx
    except ImportError as exc:
        raise SourceMlxArtifactError("mlx is required for hybrid lowering") from exc
    build_key = _sha256_bytes(_canonical_json_bytes(recipe))
    output_root = Path(output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise SourceMlxArtifactError("hybrid output root must be a real directory")
    output_root = output_root.resolve()
    target = output_root / (
        f"{_safe_model_slug(artifact.source.source_id)}-hybrid-q4-body-"
        f"{profile.lexical_precision}-lexical-{build_key[:16]}"
    )
    verifier = (
        VerifiedSourceMlxHybridQ8Artifact
        if profile.lexical_precision == "q8"
        else VerifiedSourceMlxHybridBF16Artifact
    )

    with _build_lock(build_key):
        if target.exists() or target.is_symlink():
            verified = verifier(target)
            if verified.build_key_sha256 != build_key:
                raise SourceMlxArtifactError("existing hybrid target has a foreign build key")
            return _hybrid_record(verified, artifact.artifact_id)
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
                    raise SourceMlxLoweringError("source tensor name is unsafe")
                by_role[roles[allocation.allocation_id]].append(allocation)
            shards: list[dict[str, Any]] = []
            blocks: list[dict[str, Any]] = []
            emitted_bytes = 0
            counts = {
                "body_q4": 0,
                "lexical_q8": 0,
                "lexical_bf16": 0,
                "auxiliary": 0,
            }
            stats = {
                "body": {"squared": 0.0, "elements": 0, "maximum": 0.0},
                "lexical_shared": {"squared": 0.0, "elements": 0, "maximum": 0.0},
            }
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
                            "mrun-codec": profile.codec,
                            "mrun-role": role_value,
                            "mrun-source-artifact": artifact.artifact_id,
                        },
                    )
                    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    digest, size, _file_identity = _hash_regular_file(path)
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
                        raise SourceMlxArtifactError("canonical allocation manifest is incomplete")
                    tensor = _read_allocation(artifact, allocation, record)
                    logical_names = sorted(views[allocation.allocation_id])
                    common = {
                        "source_allocation_id": allocation.allocation_id,
                        "source_blob_sha256": record["blob"]["sha256"],
                        "source_tensor": allocation.source_tensor,
                        "source_dtype": "BF16",
                        "source_shape": list(allocation.stored_shape),
                        "source_byte_count": allocation.byte_length,
                        "logical_names": logical_names,
                    }
                    values: list[tuple[str, Any, str, str, str]] = []
                    if tensor.ndim == 2 and not (
                        role == "lexical_shared" and profile.lexical_precision == "bf16"
                    ):
                        bits = 8 if role == "lexical_shared" else 4
                        packed = _quantize_affine(tensor, bits=bits, mx=mx)
                        encoding = f"mlx-affine-q{bits}-g64"
                        base = allocation.source_tensor.removesuffix(".weight")
                        values.extend(
                            (
                                (f"{base}.weight", packed.weight, "U32", "weight", encoding),
                                (f"{base}.scales", packed.scales, "BF16", "scales", encoding),
                                (f"{base}.biases", packed.biases, "BF16", "biases", encoding),
                            )
                        )
                        counts["lexical_q8" if bits == 8 else "body_q4"] += 1
                        stats[role]["squared"] += packed.sum_squared_error
                        stats[role]["elements"] += packed.elements
                        stats[role]["maximum"] = max(stats[role]["maximum"], packed.max_abs_error)
                        blocks.append(
                            {
                                "source_allocation_id": allocation.allocation_id,
                                "source_tensor": allocation.source_tensor,
                                "logical_names": logical_names,
                                "role": role,
                                "bits": bits,
                                "elements": packed.elements,
                                "max_abs_error": packed.max_abs_error,
                                "sum_squared_error": packed.sum_squared_error,
                                "rmse": math.sqrt(packed.sum_squared_error / packed.elements),
                            }
                        )
                    else:
                        exact = _source_auxiliary_array(tensor, "BF16", mx=mx)
                        values.append(
                            (
                                allocation.source_tensor,
                                exact,
                                "BF16",
                                "source_exact",
                                "source-exact-bf16",
                            )
                        )
                        counts["lexical_bf16" if tensor.ndim == 2 else "auxiliary"] += 1
                    added = sum(int(value.nbytes) for _n, value, _d, _p, _e in values)
                    if arrays and pending_bytes + added > SOURCE_MLX_SHARD_BYTES:
                        flush()
                    for name, value, dtype, part, encoding in values:
                        if name in arrays:
                            raise SourceMlxArtifactError("hybrid native parameter is duplicated")
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
                    emitted_bytes += added
                    pending_bytes += added
                flush()

            if counts["body_q4"] <= 0 or counts["auxiliary"] <= 0:
                raise SourceMlxArtifactError("hybrid lowering missed body or auxiliary allocations")
            if (counts["lexical_q8"], counts["lexical_bf16"]) not in {(1, 0), (0, 1)}:
                raise SourceMlxArtifactError("hybrid lowering did not emit one lexical allocation")
            reopened = open_component_artifact(artifact.directory)
            if (
                reopened.artifact_id != artifact.artifact_id
                or reopened.manifest_sha256 != artifact.manifest_sha256
            ):
                raise SourceMlxArtifactError("canonical source changed during hybrid lowering")

            def role_summary(role: str, codec: str, allocation_count: int) -> dict[str, Any]:
                values = stats[role]
                count = int(values["elements"])
                return {
                    "codec": codec,
                    "allocation_count": allocation_count,
                    "elements": count,
                    "max_abs_error": float(values["maximum"]),
                    "rmse": math.sqrt(float(values["squared"]) / count) if count else 0.0,
                }

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
                "source_dtype": "BF16",
                "tied_lexical_allocation": True,
                "direct_from_canonical_source": True,
                "intermediate_qstore": False,
            }
            manifest: dict[str, Any] = {
                "schema": SOURCE_MLX_HYBRID_NATIVE_SCHEMA,
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
                    "blocks": sorted(blocks, key=lambda item: item["source_allocation_id"]),
                    "roles": {
                        "body": role_summary("body", "mlx-affine-q4-g64", counts["body_q4"]),
                        "lexical_shared": role_summary(
                            "lexical_shared",
                            profile.lexical_encoding,
                            counts["lexical_q8"] + counts["lexical_bf16"],
                        ),
                        "norm": {
                            "codec": "source-exact-bf16",
                            "max_abs_error": 0.0,
                            "rmse": 0.0,
                        },
                    },
                },
                "shards": sorted(shards, key=lambda item: item["filename"]),
                "coverage": {
                    "source_allocation_count": len(allocations),
                    "body_q4_allocation_count": counts["body_q4"],
                    "lexical_q8_allocation_count": counts["lexical_q8"],
                    "lexical_bf16_allocation_count": counts["lexical_bf16"],
                    "auxiliary_source_exact_count": counts["auxiliary"],
                    "emitted_parameter_count": sum(len(item["parameters"]) for item in shards),
                    "source_allocation_bytes": sum(item.byte_length for item in allocations),
                    "emitted_parameter_bytes": emitted_bytes,
                    "source_dtype": "BF16",
                    "all_source_allocations_emitted_once": True,
                    "physical_aliases_not_duplicated": True,
                    "norms_source_exact": True,
                },
                "numerical_contract": profile.numerical_contract,
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
                verified = verifier(target)
                if verified.build_key_sha256 != build_key:
                    raise SourceMlxArtifactError(
                        "concurrent hybrid build published a different recipe"
                    ) from None
                return _hybrid_record(verified, artifact.artifact_id)
            _fsync_directory(output_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    verified = verifier(target)
    if verified.build_key_sha256 != build_key:
        raise SourceMlxArtifactError("published hybrid artifact lost recipe identity")
    return _hybrid_record(verified, artifact.artifact_id)


def _hybrid_record(
    artifact: VerifiedSourceMlxHybridArtifact, source_artifact_id: str
) -> SourceMlxHybridBuildRecord:
    return SourceMlxHybridBuildRecord(
        path=artifact.path,
        artifact_sha256=artifact.artifact_sha256,
        build_key_sha256=artifact.build_key_sha256,
        source_artifact_id=source_artifact_id,
        shard_count=len(artifact.manifest["shards"]),
        shard_bytes=artifact.shard_bytes,
        verified_reopen=True,
        lexical_precision=artifact.lexical_precision,
        body_max_abs_error=artifact.body_max_abs_error,
        body_rmse=artifact.body_rmse,
        lexical_max_abs_error=artifact.lexical_max_abs_error,
        lexical_rmse=artifact.lexical_rmse,
    )


def build_source_mlx_hybrid_q8_artifact(
    source_artifact: ComponentArtifact | str | Path, output_root: str | Path
) -> SourceMlxHybridBuildRecord:
    return build_source_mlx_hybrid_artifact(source_artifact, output_root, lexical_precision="q8")


def build_source_mlx_hybrid_bf16_artifact(
    source_artifact: ComponentArtifact | str | Path, output_root: str | Path
) -> SourceMlxHybridBuildRecord:
    return build_source_mlx_hybrid_artifact(source_artifact, output_root, lexical_precision="bf16")


class _MLXSourceHybridEngine(MLXSourceComponentEngine):
    artifact_schema = SOURCE_MLX_HYBRID_NATIVE_SCHEMA
    approximate_quantized_default = True
    compact_fused_weights_default = False

    def _artifact_numerical_contract(self) -> str:
        return self.artifact.numerical_contract

    def runtime_report(self) -> dict[str, Any]:
        report = super().runtime_report()
        report.update(
            {
                "role_aware_codec": True,
                "body_precision": "q4g64",
                "lexical_precision": self.artifact.lexical_precision,
                "norm_precision": "bf16-source-exact",
                "weight_bits_is_uniform": False,
                "body_weight_bits": self.artifact.body_bits,
                "lexical_weight_bits": self.artifact.lexical_bits,
                "role_codecs": self.artifact.manifest["recipe"]["role_codecs"],
            }
        )
        return report


class MLXSourceHybridQ8Engine(_MLXSourceHybridEngine):
    """Q4 transformer body with one tied affine-q8 lexical allocation."""

    backend = "mlx-source-hybrid-q4-body-q8-lexical"
    artifact_builder = staticmethod(build_source_mlx_hybrid_q8_artifact)
    artifact_verifier = VerifiedSourceMlxHybridQ8Artifact
    artifact_root_default = "~/.cache/mrun/mlx-source-hybrid-q8"


class MLXSourceHybridBF16Engine(_MLXSourceHybridEngine):
    """Q4 transformer body with one tied byte-exact BF16 lexical allocation."""

    backend = "mlx-source-hybrid-q4-body-bf16-lexical"
    artifact_builder = staticmethod(build_source_mlx_hybrid_bf16_artifact)
    artifact_verifier = VerifiedSourceMlxHybridBF16Artifact
    artifact_root_default = "~/.cache/mrun/mlx-source-hybrid-bf16"


__all__ = [
    "SOURCE_MLX_HYBRID_BF16_CODEC",
    "SOURCE_MLX_HYBRID_BF16_NUMERICAL_CONTRACT",
    "SOURCE_MLX_HYBRID_BUILDER_ABI",
    "SOURCE_MLX_HYBRID_LEXICAL_Q8_BITS",
    "SOURCE_MLX_HYBRID_NATIVE_SCHEMA",
    "SOURCE_MLX_HYBRID_Q8_CODEC",
    "SOURCE_MLX_HYBRID_Q8_NUMERICAL_CONTRACT",
    "MLXSourceHybridBF16Engine",
    "MLXSourceHybridQ8Engine",
    "SourceMlxHybridBuildRecord",
    "VerifiedSourceMlxHybridArtifact",
    "VerifiedSourceMlxHybridBF16Artifact",
    "VerifiedSourceMlxHybridQ8Artifact",
    "build_source_mlx_hybrid_artifact",
    "build_source_mlx_hybrid_bf16_artifact",
    "build_source_mlx_hybrid_q8_artifact",
]
