"""Header-only safetensors inventory with complete shard and byte-range proofs."""

from __future__ import annotations

import os
import re
import stat
import struct
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any

from ._json import (
    canonical_json,
    canonical_sha256,
    require_dict,
    require_exact_keys,
    require_int,
    require_list,
    require_sha256,
    require_str,
    strict_json_file,
    strict_json_loads,
)
from .errors import TensorIndexError
from .source import FrozenSourceBundle, SourceFile

TENSOR_INDEX_SCHEMA = "mrun-safetensors-index-v1"
_DEFAULT_MAX_HEADER_BYTES = 256 * 1024 * 1024
_DTYPE_BITS = {
    "BOOL": 8,
    "I8": 8,
    "U8": 8,
    "I16": 16,
    "U16": 16,
    "I32": 32,
    "U32": 32,
    "I64": 64,
    "U64": 64,
    "F4": 4,
    "I4": 4,
    "U4": 4,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E8M0": 8,
    "F16": 16,
    "BF16": 16,
    "F32": 32,
    "F64": 64,
}
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_EXPERT_RE = re.compile(r"(?:^|\.)(?:experts?|expert_bank)\.(\d+)(?:\.|$)")


@dataclass(frozen=True, slots=True)
class TensorRecord:
    source_name: str
    source_file: str
    byte_offset: int
    byte_length: int
    storage_dtype: str
    shape: tuple[int, ...]
    range_identity: str
    candidate_layer_indices: tuple[int, ...] = ()
    candidate_expert_indices: tuple[int, ...] = ()
    candidate_modalities: tuple[str, ...] = ()
    companion_candidates: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.source_name) is not str or not self.source_name:
            raise ValueError("tensor source_name must be a non-empty string")
        object.__setattr__(self, "source_file", _portable_path(self.source_file))
        if type(self.byte_offset) is not int or self.byte_offset < 0:
            raise ValueError("tensor byte_offset must be non-negative")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise ValueError("tensor byte_length must be non-negative")
        if type(self.storage_dtype) is not str or not self.storage_dtype:
            raise ValueError("tensor storage_dtype must be a non-empty string")
        shape = tuple(self.shape)
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError("tensor shape dimensions must be non-negative integers")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self,
            "range_identity",
            require_sha256(self.range_identity, field="range_identity"),
        )
        for field_name in ("candidate_layer_indices", "candidate_expert_indices"):
            values = tuple(getattr(self, field_name))
            if values != tuple(sorted(set(values))) or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise ValueError(f"{field_name} must contain sorted unique non-negative integers")
            object.__setattr__(self, field_name, values)
        for field_name in ("candidate_modalities", "companion_candidates", "suggestions"):
            values = tuple(getattr(self, field_name))
            if values != tuple(sorted(set(values))) or any(
                type(value) is not str or not value for value in values
            ):
                raise ValueError(f"{field_name} must contain sorted unique strings")
            object.__setattr__(self, field_name, values)

    @property
    def byte_end(self) -> int:
        return self.byte_offset + self.byte_length

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_file": self.source_file,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
            "storage_dtype": self.storage_dtype,
            "shape": list(self.shape),
            "range_identity": self.range_identity,
            "candidate_layer_indices": list(self.candidate_layer_indices),
            "candidate_expert_indices": list(self.candidate_expert_indices),
            "candidate_modalities": list(self.candidate_modalities),
            "companion_candidates": list(self.companion_candidates),
            "suggestions": list(self.suggestions),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TensorRecord:
        value = require_dict(payload, field="tensor record")
        expected = {
            "source_name",
            "source_file",
            "byte_offset",
            "byte_length",
            "storage_dtype",
            "shape",
            "range_identity",
            "candidate_layer_indices",
            "candidate_expert_indices",
            "candidate_modalities",
            "companion_candidates",
            "suggestions",
        }
        require_exact_keys(value, expected, field="tensor record")
        shape = require_list(value["shape"], field="tensor shape")
        layer_indices = require_list(
            value["candidate_layer_indices"], field="candidate layer indices"
        )
        expert_indices = require_list(
            value["candidate_expert_indices"], field="candidate expert indices"
        )
        modalities = require_list(value["candidate_modalities"], field="candidate modalities")
        companions = require_list(value["companion_candidates"], field="companion candidates")
        suggestions = require_list(value["suggestions"], field="suggestions")
        return cls(
            source_name=require_str(value["source_name"], field="tensor source name"),
            source_file=require_str(value["source_file"], field="tensor source file"),
            byte_offset=require_int(value["byte_offset"], field="tensor byte offset", minimum=0),
            byte_length=require_int(value["byte_length"], field="tensor byte length", minimum=0),
            storage_dtype=require_str(value["storage_dtype"], field="tensor dtype"),
            shape=tuple(require_int(item, field="tensor dimension", minimum=0) for item in shape),
            range_identity=require_str(value["range_identity"], field="range identity"),
            candidate_layer_indices=tuple(
                require_int(item, field="candidate layer index", minimum=0)
                for item in layer_indices
            ),
            candidate_expert_indices=tuple(
                require_int(item, field="candidate expert index", minimum=0)
                for item in expert_indices
            ),
            candidate_modalities=tuple(
                require_str(item, field="candidate modality") for item in modalities
            ),
            companion_candidates=tuple(
                require_str(item, field="companion candidate") for item in companions
            ),
            suggestions=tuple(require_str(item, field="suggestion") for item in suggestions),
        )


@dataclass(frozen=True, slots=True)
class SafetensorsShard:
    source_file: str
    source_sha256: str
    file_bytes: int
    header_bytes: int
    data_offset: int
    data_bytes: int
    tensor_names: tuple[str, ...]
    metadata_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_file", _portable_path(self.source_file))
        object.__setattr__(
            self,
            "source_sha256",
            require_sha256(self.source_sha256, field="shard sha256"),
        )
        for field_name in ("file_bytes", "header_bytes", "data_offset", "data_bytes"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.file_bytes <= 8 or self.header_bytes <= 0:
            raise ValueError("safetensors shard/header must be non-empty")
        if self.data_offset != 8 + self.header_bytes:
            raise ValueError("safetensors data_offset does not follow its header")
        if self.data_offset + self.data_bytes != self.file_bytes:
            raise ValueError("safetensors payload does not cover the complete shard")
        names = tuple(self.tensor_names)
        if names != tuple(sorted(set(names))):
            raise ValueError("shard tensor names must be sorted and unique")
        object.__setattr__(self, "tensor_names", names)
        metadata = strict_json_loads(self.metadata_json, field="safetensors metadata")
        if not isinstance(metadata, dict):
            raise TypeError("safetensors metadata must be an object")
        object.__setattr__(self, "metadata_json", canonical_json(metadata))

    @property
    def metadata(self) -> dict[str, Any]:
        value = strict_json_loads(self.metadata_json, field="safetensors metadata")
        assert isinstance(value, dict)
        return value

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "file_bytes": self.file_bytes,
            "header_bytes": self.header_bytes,
            "data_offset": self.data_offset,
            "data_bytes": self.data_bytes,
            "tensor_names": list(self.tensor_names),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> SafetensorsShard:
        value = require_dict(payload, field="safetensors shard")
        require_exact_keys(
            value,
            {
                "source_file",
                "source_sha256",
                "file_bytes",
                "header_bytes",
                "data_offset",
                "data_bytes",
                "tensor_names",
                "metadata",
            },
            field="safetensors shard",
        )
        names = require_list(value["tensor_names"], field="shard tensor names")
        metadata = require_dict(value["metadata"], field="safetensors metadata")
        return cls(
            source_file=require_str(value["source_file"], field="shard source file"),
            source_sha256=require_str(value["source_sha256"], field="shard sha256"),
            file_bytes=require_int(value["file_bytes"], field="shard bytes", minimum=0),
            header_bytes=require_int(value["header_bytes"], field="header bytes", minimum=0),
            data_offset=require_int(value["data_offset"], field="data offset", minimum=0),
            data_bytes=require_int(value["data_bytes"], field="data bytes", minimum=0),
            tensor_names=tuple(require_str(item, field="tensor name") for item in names),
            metadata_json=canonical_json(metadata),
        )


@dataclass(frozen=True, slots=True)
class TensorIndex:
    source_fingerprint: str
    tensors: tuple[TensorRecord, ...]
    shards: tuple[SafetensorsShard, ...]
    total_tensor_bytes: int
    _config_features_json: str
    fingerprint: str
    schema_version: str = TENSOR_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TENSOR_INDEX_SCHEMA:
            raise ValueError(f"unsupported tensor-index schema: {self.schema_version!r}")
        object.__setattr__(
            self,
            "source_fingerprint",
            require_sha256(self.source_fingerprint, field="source fingerprint"),
        )
        tensors = tuple(self.tensors)
        if tensors != tuple(sorted(tensors, key=lambda item: item.source_name)):
            raise ValueError("tensor index records must be sorted by source name")
        names = [record.source_name for record in tensors]
        if len(names) != len(set(names)):
            raise ValueError("tensor index contains duplicate source names")
        object.__setattr__(self, "tensors", tensors)
        shards = tuple(self.shards)
        if not shards or shards != tuple(sorted(shards, key=lambda item: item.source_file)):
            raise ValueError("tensor index shards must be non-empty and sorted")
        if len({item.source_file for item in shards}) != len(shards):
            raise ValueError("tensor index contains duplicate shards")
        object.__setattr__(self, "shards", shards)
        shard_by_file = {item.source_file: item for item in shards}
        tensors_by_file: dict[str, list[TensorRecord]] = {
            source_file: [] for source_file in shard_by_file
        }
        for tensor in tensors:
            shard = shard_by_file.get(tensor.source_file)
            if shard is None:
                raise ValueError("tensor record references an unknown safetensors shard")
            if tensor.byte_offset < shard.data_offset or tensor.byte_end > shard.file_bytes:
                raise ValueError("tensor record range escapes its safetensors payload")
            relative_start = tensor.byte_offset - shard.data_offset
            relative_end = relative_start + tensor.byte_length
            expected_identity = canonical_sha256(
                {
                    "source_file": shard.source_file,
                    "source_sha256": shard.source_sha256,
                    "relative_start": relative_start,
                    "relative_end": relative_end,
                    "dtype": tensor.storage_dtype,
                    "shape": list(tensor.shape),
                }
            )
            if tensor.range_identity != expected_identity:
                raise ValueError("tensor range identity does not match its shard-bound range")
            tensors_by_file[tensor.source_file].append(tensor)
        for source_file, shard_tensors in tensors_by_file.items():
            shard = shard_by_file[source_file]
            if tuple(sorted(item.source_name for item in shard_tensors)) != shard.tensor_names:
                raise ValueError("shard tensor-name inventory does not match tensor records")
            cursor = shard.data_offset
            for tensor in sorted(
                shard_tensors,
                key=lambda item: (item.byte_offset, item.byte_end, item.source_name),
            ):
                if tensor.byte_offset != cursor:
                    raise ValueError("tensor record ranges contain a gap or overlap")
                if tensor.byte_length:
                    cursor = tensor.byte_end
            if cursor != shard.file_bytes:
                raise ValueError("tensor record ranges do not cover the complete shard payload")
        expected_bytes = sum(record.byte_length for record in tensors)
        if self.total_tensor_bytes != expected_bytes:
            raise ValueError("total_tensor_bytes does not match tensor records")
        if self.total_tensor_bytes != sum(shard.data_bytes for shard in shards):
            raise ValueError("total_tensor_bytes does not match safetensors shard payloads")
        config = strict_json_loads(self._config_features_json, field="config features")
        if not isinstance(config, dict):
            raise TypeError("config features must be an object")
        object.__setattr__(self, "_config_features_json", canonical_json(config))
        object.__setattr__(
            self,
            "fingerprint",
            require_sha256(self.fingerprint, field="tensor index fingerprint"),
        )
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("tensor index fingerprint does not match its payload")

    @classmethod
    def scan(
        cls,
        source: FrozenSourceBundle,
        *,
        max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES,
    ) -> TensorIndex:
        return build_tensor_index(source, max_header_bytes=max_header_bytes)

    @property
    def config_features(self) -> dict[str, Any]:
        value = strict_json_loads(self._config_features_json, field="config features")
        assert isinstance(value, dict)
        return value

    def by_name(self) -> dict[str, TensorRecord]:
        return {record.source_name: record for record in self.tensors}

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "tensors": [record.as_dict() for record in self.tensors],
            "shards": [shard.as_dict() for shard in self.shards],
            "total_tensor_bytes": self.total_tensor_bytes,
            "config_features": self.config_features,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Any) -> TensorIndex:
        value = require_dict(payload, field="tensor index")
        require_exact_keys(
            value,
            {
                "schema_version",
                "source_fingerprint",
                "tensors",
                "shards",
                "total_tensor_bytes",
                "config_features",
                "fingerprint",
            },
            field="tensor index",
        )
        tensors = require_list(value["tensors"], field="tensors")
        shards = require_list(value["shards"], field="shards")
        return cls(
            schema_version=require_str(value["schema_version"], field="tensor-index schema"),
            source_fingerprint=require_str(value["source_fingerprint"], field="source fingerprint"),
            tensors=tuple(TensorRecord.from_dict(record) for record in tensors),
            shards=tuple(SafetensorsShard.from_dict(record) for record in shards),
            total_tensor_bytes=require_int(
                value["total_tensor_bytes"], field="total tensor bytes", minimum=0
            ),
            _config_features_json=canonical_json(
                require_dict(value["config_features"], field="config features")
            ),
            fingerprint=require_str(value["fingerprint"], field="tensor-index fingerprint"),
        )


def _portable_path(value: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"invalid source-relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid source-relative path: {value!r}")
    return path.as_posix()


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _expected_bytes(dtype: str, shape: tuple[int, ...]) -> int | None:
    bits = _DTYPE_BITS.get(dtype)
    if bits is None:
        return None
    return (_product(shape) * bits + 7) // 8


def _read_exact_at(descriptor: int, offset: int, length: int, *, field: str) -> bytes:
    chunks: list[bytes] = []
    cursor = offset
    remaining = length
    while remaining:
        chunk = os.pread(descriptor, remaining, cursor)
        if not chunk:
            raise TensorIndexError(f"short read while reading {field}")
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _indices(pattern: re.Pattern[str], name: str) -> tuple[int, ...]:
    return tuple(sorted({int(match) for match in pattern.findall(name)}))


def _modalities(name: str) -> tuple[str, ...]:
    lowered = name.lower()
    candidates: set[str] = set()
    if any(part in lowered for part in ("vision", "visual", "image", "patch_embed")):
        candidates.add("vision")
    if any(part in lowered for part in ("audio", "speech", "mel_")):
        candidates.add("audio")
    if "video" in lowered:
        candidates.add("video")
    return tuple(sorted(candidates))


def _suggestions(name: str) -> tuple[str, ...]:
    lowered = name.lower()
    suggestions: set[str] = set()
    if any(part in lowered for part in ("query_key_value", "qkv_proj", "wqkv")):
        suggestions.add("possible-fused-qkv")
    if any(part in lowered for part in ("qweight", "qzeros", "g_idx")):
        suggestions.add("possible-packed-quantization")
    if any(part in lowered for part in ("scale", "zero_point", "scales")):
        suggestions.add("possible-codec-companion")
    return tuple(sorted(suggestions))


def _companion_candidates(name: str, all_names: set[str]) -> tuple[str, ...]:
    stems = {name}
    for suffix in (".weight", ".qweight", ".weight_packed"):
        if name.endswith(suffix):
            stems.add(name[: -len(suffix)])
    candidates: set[str] = set()
    for stem in stems:
        for suffix in (
            ".scale",
            ".scales",
            ".weight_scale",
            ".weight_scale_inv",
            ".qzeros",
            ".zeros",
            ".zero_points",
            ".g_idx",
        ):
            candidate = stem + suffix
            if candidate in all_names and candidate != name:
                candidates.add(candidate)
    return tuple(sorted(candidates))


def _scan_shard(
    source: FrozenSourceBundle,
    record: SourceFile,
    *,
    max_header_bytes: int,
) -> tuple[SafetensorsShard, tuple[TensorRecord, ...]]:
    path = source.file_path(record.path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TensorIndexError(
            f"cannot safely open safetensors shard {record.path}",
            details={"source_file": record.path},
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != record.byte_count:
            raise TensorIndexError(f"safetensors shard changed before header scan: {record.path}")
        prefix = _read_exact_at(descriptor, 0, 8, field=f"{record.path} header length")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length <= 0 or header_length > max_header_bytes:
            raise TensorIndexError(
                f"invalid safetensors header length in {record.path}",
                details={"header_bytes": header_length, "limit": max_header_bytes},
            )
        data_offset = 8 + header_length
        if data_offset > before.st_size:
            raise TensorIndexError(f"safetensors header exceeds shard size: {record.path}")
        header_bytes = _read_exact_at(
            descriptor,
            8,
            header_length,
            field=f"{record.path} JSON header",
        )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise TensorIndexError(f"safetensors shard changed during header scan: {record.path}")
    try:
        header = strict_json_loads(header_bytes, field=f"{record.path} safetensors header")
    except ValueError as exc:
        raise TensorIndexError(str(exc)) from exc
    if not isinstance(header, dict):
        raise TensorIndexError(f"safetensors header must be an object: {record.path}")

    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict) or any(
        type(key) is not str or type(value) is not str for key, value in raw_metadata.items()
    ):
        raise TensorIndexError(f"invalid safetensors metadata in {record.path}")
    data_bytes = before.st_size - data_offset
    relative_ranges: list[tuple[int, int, str]] = []
    tensors: list[TensorRecord] = []
    for name in sorted(header):
        if type(name) is not str or not name or name.strip() != name:
            raise TensorIndexError(f"invalid tensor name in {record.path}: {name!r}")
        try:
            entry = require_dict(header[name], field=f"tensor header {name}")
            require_exact_keys(entry, {"dtype", "shape", "data_offsets"}, field=name)
            dtype = require_str(entry["dtype"], field=f"{name} dtype")
            raw_shape = require_list(entry["shape"], field=f"{name} shape")
            shape = tuple(
                require_int(item, field=f"{name} shape dimension", minimum=0) for item in raw_shape
            )
            if len(shape) > 64:
                raise ValueError(f"{name} shape has too many dimensions")
            raw_offsets = require_list(entry["data_offsets"], field=f"{name} data offsets")
            if len(raw_offsets) != 2:
                raise ValueError(f"{name} data_offsets must contain exactly two integers")
            start = require_int(raw_offsets[0], field=f"{name} range start", minimum=0)
            end = require_int(raw_offsets[1], field=f"{name} range end", minimum=0)
        except (TypeError, ValueError) as exc:
            raise TensorIndexError(f"invalid tensor header for {name}: {exc}") from exc
        if end < start or end > data_bytes:
            raise TensorIndexError(
                f"tensor range is out of bounds for {name}",
                details={"start": start, "end": end, "data_bytes": data_bytes},
            )
        byte_length = end - start
        expected = _expected_bytes(dtype, shape)
        if expected is not None and expected != byte_length:
            raise TensorIndexError(
                f"tensor byte length does not match dtype/shape for {name}",
                details={
                    "dtype": dtype,
                    "shape": list(shape),
                    "expected_bytes": expected,
                    "actual_bytes": byte_length,
                },
            )
        absolute_offset = data_offset + start
        range_identity = canonical_sha256(
            {
                "source_file": record.path,
                "source_sha256": record.sha256,
                "relative_start": start,
                "relative_end": end,
                "dtype": dtype,
                "shape": list(shape),
            }
        )
        relative_ranges.append((start, end, name))
        tensors.append(
            TensorRecord(
                source_name=name,
                source_file=record.path,
                byte_offset=absolute_offset,
                byte_length=byte_length,
                storage_dtype=dtype,
                shape=shape,
                range_identity=range_identity,
                candidate_layer_indices=_indices(_LAYER_RE, name),
                candidate_expert_indices=_indices(_EXPERT_RE, name),
                candidate_modalities=_modalities(name),
                suggestions=_suggestions(name),
            )
        )

    cursor = 0
    for start, end, name in sorted(relative_ranges, key=lambda item: (item[0], item[1], item[2])):
        if start != cursor:
            kind = "overlap" if start < cursor else "gap"
            raise TensorIndexError(
                f"safetensors payload contains a byte-range {kind} before {name}",
                details={"source_file": record.path, "cursor": cursor, "start": start},
            )
        if end > start:
            cursor = end
    if cursor != data_bytes:
        raise TensorIndexError(
            f"safetensors payload has an unindexed trailing range: {record.path}",
            details={"covered_bytes": cursor, "data_bytes": data_bytes},
        )
    shard = SafetensorsShard(
        source_file=record.path,
        source_sha256=record.sha256,
        file_bytes=before.st_size,
        header_bytes=header_length,
        data_offset=data_offset,
        data_bytes=data_bytes,
        tensor_names=tuple(sorted(header)),
        metadata_json=canonical_json(raw_metadata),
    )
    return shard, tuple(tensors)


def _load_weight_map(source: FrozenSourceBundle) -> dict[str, str] | None:
    indexes = source.files_with_role("safetensors-index")
    if not indexes:
        return None
    if len(indexes) != 1:
        raise TensorIndexError("source has multiple safetensors indexes")
    index_record = indexes[0]
    try:
        payload = strict_json_file(
            source.file_path(index_record.path),
            max_bytes=source.policy.json_parse_limit_bytes,
            field=index_record.path,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise TensorIndexError(f"cannot parse safetensors index: {exc}") from exc
    if not isinstance(payload, dict):
        raise TensorIndexError("safetensors index must be an object")
    unknown = set(payload) - {"metadata", "weight_map"}
    if unknown:
        raise TensorIndexError(
            "safetensors index contains unknown top-level fields",
            details={"fields": sorted(unknown)},
        )
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise TensorIndexError("safetensors index has no non-empty weight_map")
    normalized: dict[str, str] = {}
    for name, shard in weight_map.items():
        if type(name) is not str or not name:
            raise TensorIndexError("safetensors weight_map contains an invalid tensor name")
        if type(shard) is not str:
            raise TensorIndexError(f"safetensors weight_map path for {name!r} is not a string")
        try:
            normalized[name] = _portable_path(shard)
        except ValueError as exc:
            raise TensorIndexError(
                f"safetensors weight_map path for {name!r} is unsafe: {exc}"
            ) from exc
    return normalized


class SafetensorsTensorIndexer:
    """Build a complete index while reading only shard prefixes and JSON headers."""

    def __init__(self, *, max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES) -> None:
        if type(max_header_bytes) is not int or max_header_bytes <= 0:
            raise ValueError("max_header_bytes must be a positive integer")
        self.max_header_bytes = max_header_bytes

    def build(self, source: FrozenSourceBundle) -> TensorIndex:
        if not isinstance(source, FrozenSourceBundle):
            raise TypeError("source must be a FrozenSourceBundle")
        source.assert_stat_unchanged()
        shard_files = tuple(
            sorted(source.files_with_role("weight-shard"), key=lambda item: item.path)
        )
        weight_map = _load_weight_map(source)
        if weight_map is None and len(shard_files) != 1:
            raise TensorIndexError(
                "multiple safetensors shards require one coherent index",
                details={"shards": [record.path for record in shard_files]},
            )
        if weight_map is not None:
            expected_shards = sorted(set(weight_map.values()))
            actual_shards = [record.path for record in shard_files]
            if expected_shards != actual_shards:
                raise TensorIndexError(
                    "safetensors index/shard inventory mismatch",
                    details={"expected": expected_shards, "actual": actual_shards},
                )

        shards: list[SafetensorsShard] = []
        tensors: list[TensorRecord] = []
        seen: set[str] = set()
        for shard_file in shard_files:
            shard, shard_tensors = _scan_shard(
                source,
                shard_file,
                max_header_bytes=self.max_header_bytes,
            )
            duplicate = sorted(
                record.source_name for record in shard_tensors if record.source_name in seen
            )
            if duplicate:
                raise TensorIndexError(
                    "tensor names are duplicated across safetensors shards",
                    details={"tensors": duplicate},
                )
            seen.update(record.source_name for record in shard_tensors)
            shards.append(shard)
            tensors.extend(shard_tensors)

        if weight_map is not None:
            actual_names = {record.source_name for record in tensors}
            expected_names = set(weight_map)
            if actual_names != expected_names:
                raise TensorIndexError(
                    "safetensors weight_map does not cover the exact tensor set",
                    details={
                        "missing_from_shards": sorted(expected_names - actual_names),
                        "missing_from_index": sorted(actual_names - expected_names),
                    },
                )
            wrong_shard = sorted(
                record.source_name
                for record in tensors
                if weight_map[record.source_name] != record.source_file
            )
            if wrong_shard:
                raise TensorIndexError(
                    "safetensors weight_map assigns tensors to the wrong shard",
                    details={"tensors": wrong_shard},
                )

        names = {record.source_name for record in tensors}
        tensors = [
            replace(record, companion_candidates=_companion_candidates(record.source_name, names))
            for record in tensors
        ]
        tensors.sort(key=lambda item: item.source_name)
        shards.sort(key=lambda item: item.source_file)
        payload = {
            "schema_version": TENSOR_INDEX_SCHEMA,
            "source_fingerprint": source.fingerprint,
            "tensors": [record.as_dict() for record in tensors],
            "shards": [shard.as_dict() for shard in shards],
            "total_tensor_bytes": sum(record.byte_length for record in tensors),
            "config_features": source.config,
        }
        index = TensorIndex(
            source_fingerprint=source.fingerprint,
            tensors=tuple(tensors),
            shards=tuple(shards),
            total_tensor_bytes=payload["total_tensor_bytes"],
            _config_features_json=canonical_json(source.config),
            fingerprint=canonical_sha256(payload),
        )
        source.assert_stat_unchanged()
        return index


def build_tensor_index(
    source: FrozenSourceBundle,
    *,
    max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES,
) -> TensorIndex:
    return SafetensorsTensorIndexer(max_header_bytes=max_header_bytes).build(source)
