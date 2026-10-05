"""Atomic, content-addressed lowering from a verified QStore-v3 into ComponentGraph.

This is deliberately a *transitional* compiler stage.  Its source is an already-quantized,
self-authenticating ``mrun-qstore-int8-v3`` store; it is not the direct lowering from the
canonical source-component artifact.  Keeping that boundary explicit prevents QStore
quantization from being mistaken for a lossless model decompilation.

The lowerer preserves each physical allocation exactly once, keeps aliases logical, and
publishes only after the production :class:`ComponentGraph` reader has reopened every
manifest/blob.  A stricter lowering-specific inspection then reconstructs the three source
payload hashes from component-local spans in source-offset order.  Consequently a graph is
accepted only when its routing is a complete byte-for-byte partition of the verified source.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO

from ..store_provenance import (
    SEMANTIC_DERIVED_SCHEMA,
    inspect_semantic_store_identity,
    semantic_manifest_sha256,
)
from .kernels.composite_qstore import (
    COMPONENT_GRAPH_SCHEMA,
    COMPONENT_PAYLOAD_FILES,
    ComponentGraph,
    ComponentGraphError,
    VocabManifest,
    _allocation_topology,
    _body_abi_semantic_sha256,
    _component_semantic_digest,
    _expected_operation_contracts,
    _graph_fingerprint_payload,
    canonical_json_bytes,
    tokenizer_descriptor,
)
from .kernels.qstore import _load_qstore_manifest
from .kernels.qstore_build import QSTORE_SCHEMA

QSTORE_COMPONENT_LOWERING_SCHEMA = "mrun-qstore-v3-component-lowering-v1"
QSTORE_COMPONENT_ADDRESS_SCHEMA = "mrun-qstore-v3-component-address-v1"
QSTORE_COMPONENT_RESULT_SCHEMA = "mrun-qstore-v3-component-build-result-v1"
TRANSITIONAL_SCOPE = (
    "transitional-verified-qstore-v3-lowering-not-direct-source-artifact-native-lowering"
)
SUPPORTED_QSTORE_ARCHITECTURES = frozenset({"llama", "qwen2", "qwen3"})

__all__ = [
    "QSTORE_COMPONENT_ADDRESS_SCHEMA",
    "QSTORE_COMPONENT_LOWERING_SCHEMA",
    "QSTORE_COMPONENT_RESULT_SCHEMA",
    "SUPPORTED_QSTORE_ARCHITECTURES",
    "TRANSITIONAL_SCOPE",
    "QStoreComponentArtifact",
    "QStoreComponentBuildError",
    "inspect_qstore_v3_component_artifact",
    "lower_qstore_v3_to_component_graph",
    "main",
]

_PAYLOAD_LAYOUT = {
    "qrow": (
        ("weights.i8", "w_off", "w_len", 1),
        ("scales.f32", "s_off", "s_len", 4),
    ),
    "fp32": (("extras.f32", "e_off", "e_len", 4),),
}
_MINIMUM_COMPONENT_FILE_BYTES = {
    "weights.i8": 1,
    "scales.f32": 4,
    "extras.f32": 4,
}
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


class QStoreComponentBuildError(ComponentGraphError):
    """The verified-QStore lowering or its atomic publication failed closed."""


@dataclass(frozen=True)
class QStoreComponentArtifact:
    """Stable result returned by both build and strict-inspection entry points."""

    artifact_id: str
    root: Path
    graph_path: Path
    graph_fingerprint_sha256: str
    declared_graph_fingerprint_sha256: str
    source_checkpoint_sha256: str
    source_derived_store_sha256: str
    tokenizer_semantic_sha256: str
    architecture: str
    model: str
    roles: tuple[str, ...]
    logical_block_count: int
    source_payload_bytes: int
    created: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": QSTORE_COMPONENT_RESULT_SCHEMA,
            "scope": TRANSITIONAL_SCOPE,
            "artifact_id": self.artifact_id,
            "root": str(self.root),
            "graph_path": str(self.graph_path),
            "graph_fingerprint_sha256": self.graph_fingerprint_sha256,
            "declared_graph_fingerprint_sha256": self.declared_graph_fingerprint_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_derived_store_sha256": self.source_derived_store_sha256,
            "tokenizer_semantic_sha256": self.tokenizer_semantic_sha256,
            "architecture": self.architecture,
            "model": self.model,
            "roles": list(self.roles),
            "logical_block_count": self.logical_block_count,
            "source_payload_bytes": self.source_payload_bytes,
            "created": self.created,
        }


@dataclass(frozen=True)
class _VerifiedSource:
    directory: Path
    manifest: dict[str, Any]
    identity: dict[str, Any]
    file_stats: dict[str, tuple[int, int, int, int, int, int]]
    payloads: dict[str, dict[str, Any]]

    def assert_unchanged_and_reverify(self) -> None:
        _assert_source_stats(self.directory, self.file_stats)
        strict_manifest = _read_json_regular_strict(self.directory / "manifest.json")
        try:
            manifest, identity = _load_qstore_manifest(
                self.directory,
                expected_dtype="int8",
            )
        except Exception as exc:
            raise QStoreComponentBuildError(
                "source QStore-v3 failed fresh semantic/blob verification"
            ) from exc
        _assert_source_stats(self.directory, self.file_stats)
        if canonical_json_bytes(manifest) != canonical_json_bytes(strict_manifest):
            raise QStoreComponentBuildError("source QStore manifest parse is not canonical")
        if canonical_json_bytes(manifest) != canonical_json_bytes(self.manifest):
            raise QStoreComponentBuildError("source QStore manifest changed during lowering")
        if _source_identity_payload(identity) != _source_identity_payload(self.identity):
            raise QStoreComponentBuildError("source QStore identity changed during lowering")


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise QStoreComponentBuildError("lowering input is not canonical JSON data") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _require_sha256(value: Any, field: str) -> str:
    if not _is_sha256(value):
        raise QStoreComponentBuildError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def _stat_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise QStoreComponentBuildError(f"source QStore file is unavailable: {path.name}") from exc
    if not stat.S_ISREG(value.st_mode):
        kind = "symlink" if stat.S_ISLNK(value.st_mode) else "non-regular file"
        raise QStoreComponentBuildError(
            f"source QStore file must be regular, not a {kind}: {path.name}"
        )
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QStoreComponentBuildError(f"duplicate JSON key {key!r} in source manifest")
        result[key] = value
    return result


def _read_json_regular_strict(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QStoreComponentBuildError(f"cannot open source manifest {path}") from exc
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise QStoreComponentBuildError("source manifest must be a regular file")
        chunks: list[bytes] = []
        while value := os.read(descriptor, 1024 * 1024):
            chunks.append(value)
        final = os.fstat(descriptor)
        if (
            int(initial.st_dev),
            int(initial.st_ino),
            int(initial.st_size),
            int(initial.st_mtime_ns),
            int(initial.st_ctime_ns),
        ) != (
            int(final.st_dev),
            int(final.st_ino),
            int(final.st_size),
            int(final.st_mtime_ns),
            int(final.st_ctime_ns),
        ):
            raise QStoreComponentBuildError("source manifest changed while being read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks), object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QStoreComponentBuildError("source manifest is not valid strict JSON") from exc
    if not isinstance(value, dict):
        raise QStoreComponentBuildError("source manifest must be a JSON object")
    return value


def _source_stats(directory: Path) -> dict[str, tuple[int, int, int, int, int, int]]:
    return {
        filename: _stat_identity(directory / filename)
        for filename in ("manifest.json", *COMPONENT_PAYLOAD_FILES)
    }


def _assert_source_stats(
    directory: Path,
    expected: Mapping[str, tuple[int, int, int, int, int, int]],
) -> None:
    if _source_stats(directory) != dict(expected):
        raise QStoreComponentBuildError("source QStore files changed during lowering")


def _source_identity_payload(identity: Mapping[str, Any]) -> dict[str, str]:
    return {
        "source_checkpoint_sha256": _require_sha256(
            identity.get("source_checkpoint_sha256"),
            "source checkpoint identity",
        ),
        "derived_store_sha256": _require_sha256(
            identity.get("derived_store_sha256"),
            "source QStore identity",
        ),
        "manifest_semantic_sha256": _require_sha256(
            identity.get("manifest_semantic_sha256"),
            "source QStore semantic manifest identity",
        ),
    }


def _payload_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    derived = manifest.get("derived")
    records = derived.get("files") if isinstance(derived, Mapping) else None
    if not isinstance(records, list):
        raise QStoreComponentBuildError("source QStore has no verified derived-file records")
    normalized: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise QStoreComponentBuildError("source QStore derived-file record is malformed")
        name = str(raw.get("name", ""))
        if name in normalized:
            raise QStoreComponentBuildError(f"duplicate source payload record {name!r}")
        size = raw.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise QStoreComponentBuildError(f"source payload {name!r} has an invalid size")
        normalized[name] = {
            "bytes": size,
            "sha256": _require_sha256(raw.get("sha256"), f"source payload {name!r}"),
        }
    if set(normalized) != set(COMPONENT_PAYLOAD_FILES):
        raise QStoreComponentBuildError(
            "source QStore must bind exactly weights.i8, scales.f32, and extras.f32"
        )
    return {name: normalized[name] for name in COMPONENT_PAYLOAD_FILES}


def _portable_semantic_source_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the complete v3 semantic manifest without host-local discovery metadata."""

    portable = _json_clone({key: value for key, value in manifest.items() if key != "derived"})
    source = portable.get("source")
    if not isinstance(source, dict):
        raise QStoreComponentBuildError("source QStore has no source-checkpoint provenance")
    source.pop("source_dir", None)
    # Revision discovery is evidence, not semantic identity. Canonicalize it so equivalent
    # QStores reached through a snapshot path and a direct path yield identical artifacts.
    source["revision"] = {"kind": "identity-excluded", "commits": []}
    return portable


def _open_verified_source(source_dir: str | Path) -> _VerifiedSource:
    directory = Path(source_dir).expanduser().absolute()
    try:
        directory_stat = directory.lstat()
    except OSError as exc:
        raise QStoreComponentBuildError(
            f"source QStore directory is unavailable: {directory}"
        ) from exc
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        raise QStoreComponentBuildError(
            "source QStore path must be a real directory, not a symlink"
        )

    before = _source_stats(directory)
    strict_manifest = _read_json_regular_strict(directory / "manifest.json")
    try:
        manifest, identity = _load_qstore_manifest(directory, expected_dtype="int8")
    except Exception as exc:
        raise QStoreComponentBuildError(
            "source must be a fully verified mrun-qstore-int8-v3 store"
        ) from exc
    after = _source_stats(directory)
    if before != after:
        raise QStoreComponentBuildError("source QStore files changed during verification")
    if canonical_json_bytes(manifest) != canonical_json_bytes(strict_manifest):
        raise QStoreComponentBuildError("source QStore manifest parse is not canonical")
    model_name = manifest.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise QStoreComponentBuildError("source QStore model_name must be a non-empty string")
    if manifest.get("schema_version") != QSTORE_SCHEMA:
        raise QStoreComponentBuildError(
            f"source schema must be {QSTORE_SCHEMA!r}; legacy QStores cannot be lowered"
        )
    if str(manifest.get("dtype")) != "int8":
        raise QStoreComponentBuildError("verified-QStore component lowering supports int8 only")
    architecture = str(manifest.get("arch", ""))
    if architecture not in SUPPORTED_QSTORE_ARCHITECTURES:
        raise QStoreComponentBuildError(
            f"architecture {architecture!r} is outside the proved transitional lowering set "
            f"{sorted(SUPPORTED_QSTORE_ARCHITECTURES)!r}"
        )
    if (
        identity.get("content_identity_verified") is not True
        or identity.get("blob_identity_verified") is not True
    ):
        raise QStoreComponentBuildError("source QStore did not pass exact blob verification")
    _source_identity_payload(identity)
    return _VerifiedSource(
        directory=directory,
        manifest=_json_clone(manifest),
        identity=_json_clone(identity),
        file_stats=after,
        payloads=_payload_records(manifest),
    )


def _coerce_tokenizer_descriptor(
    tokenizer: Any,
    *,
    configured_row_count: int,
) -> dict[str, Any]:
    descriptor = (
        _json_clone(tokenizer)
        if isinstance(tokenizer, Mapping)
        else tokenizer_descriptor(tokenizer)
    )
    if not isinstance(descriptor, dict):  # pragma: no cover - clone guarantees this for mappings
        raise QStoreComponentBuildError("tokenizer descriptor must be an object")
    try:
        VocabManifest.from_descriptor(
            descriptor,
            configured_row_count=configured_row_count,
        )
    except ComponentGraphError as exc:
        raise QStoreComponentBuildError("tokenizer descriptor failed semantic validation") from exc
    return descriptor


def _address_payload(
    *,
    source_identity: Mapping[str, Any],
    tokenizer_semantic_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": QSTORE_COMPONENT_ADDRESS_SCHEMA,
        "lowering_schema": QSTORE_COMPONENT_LOWERING_SCHEMA,
        "scope": TRANSITIONAL_SCOPE,
        "output_graph_schema": COMPONENT_GRAPH_SCHEMA,
        "source_store_schema": QSTORE_SCHEMA,
        "source_identity": _source_identity_payload(source_identity),
        "tokenizer_semantic_sha256": _require_sha256(
            tokenizer_semantic_sha256,
            "tokenizer semantic identity",
        ),
    }


def _artifact_id(
    *,
    source_identity: Mapping[str, Any],
    tokenizer_semantic_sha256: str,
) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            _address_payload(
                source_identity=source_identity,
                tokenizer_semantic_sha256=tokenizer_semantic_sha256,
            )
        )
    )


def _source_lineage(
    *,
    source: _VerifiedSource,
    artifact_id: str,
    tokenizer_semantic_sha256: str,
) -> dict[str, Any]:
    identity = _source_identity_payload(source.identity)
    return {
        "identity_status": "verified-qstore-v3-content-and-blobs",
        "lowering_schema": QSTORE_COMPONENT_LOWERING_SCHEMA,
        "lowering_scope": TRANSITIONAL_SCOPE,
        "direct_source_artifact_lowering": False,
        "source_store_schema_version": QSTORE_SCHEMA,
        **identity,
        "source_semantic_manifest": _portable_semantic_source_manifest(source.manifest),
        "source_payloads": _json_clone(source.payloads),
        "tokenizer_semantic_sha256": tokenizer_semantic_sha256,
        "artifact_id": artifact_id,
    }


def _body_abi(manifest: Mapping[str, Any]) -> dict[str, Any]:
    blocks = manifest.get("blocks")
    config = manifest.get("config")
    if not isinstance(blocks, Mapping) or not isinstance(config, Mapping):
        raise QStoreComponentBuildError("source QStore has no block table or runtime config")
    architecture = str(manifest["arch"])
    tied = bool(manifest.get("tie_word_embeddings", False))
    semantic = _body_abi_semantic_sha256(
        architecture=architecture,
        config=config,
        tied=tied,
        logical_blocks=blocks,
    )
    raw_hidden_size = config.get("hidden_size")
    raw_vocab_size = config.get("vocab_size")
    if (
        isinstance(raw_hidden_size, bool)
        or not isinstance(raw_hidden_size, int)
        or isinstance(raw_vocab_size, bool)
        or not isinstance(raw_vocab_size, int)
    ):
        raise QStoreComponentBuildError(
            "source QStore runtime config must define hidden_size and vocab_size"
        )
    hidden_size = raw_hidden_size
    vocab_size = raw_vocab_size
    if hidden_size <= 0 or vocab_size <= 0:
        raise QStoreComponentBuildError("source QStore hidden/vocabulary sizes must be positive")
    return {
        "semantic_sha256": semantic,
        "architecture": architecture,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "logical_block_count": len(blocks),
        "config": _json_clone(config),
    }


def _open_source_payloads(source: _VerifiedSource) -> dict[str, int]:
    descriptors: dict[str, int] = {}
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for filename in COMPONENT_PAYLOAD_FILES:
            path = source.directory / filename
            descriptor = os.open(path, flags)
            file_stat = os.fstat(descriptor)
            identity = (
                int(file_stat.st_dev),
                int(file_stat.st_ino),
                int(file_stat.st_mode),
                int(file_stat.st_size),
                int(file_stat.st_mtime_ns),
                int(file_stat.st_ctime_ns),
            )
            if not stat.S_ISREG(file_stat.st_mode) or identity != source.file_stats[filename]:
                os.close(descriptor)
                raise QStoreComponentBuildError(
                    f"source payload changed before copying: {filename}"
                )
            descriptors[filename] = descriptor
    except Exception:
        for descriptor in descriptors.values():
            os.close(descriptor)
        raise
    return descriptors


def _copy_exact_span(
    *,
    source_descriptor: int,
    source_offset: int,
    length: int,
    destination: BinaryIO,
) -> None:
    if source_offset < 0 or length <= 0:
        raise QStoreComponentBuildError("source allocation has a negative offset or empty span")
    remaining = length
    cursor = source_offset
    while remaining:
        amount = min(remaining, _COPY_CHUNK_BYTES)
        try:
            value = os.pread(source_descriptor, amount, cursor)
        except OSError as exc:
            raise QStoreComponentBuildError("cannot read source QStore allocation") from exc
        if len(value) != amount:
            raise QStoreComponentBuildError("short read while copying source QStore allocation")
        destination.write(value)
        cursor += amount
        remaining -= amount


def _write_json_file(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise QStoreComponentBuildError(f"cannot write lowering artifact {path}") from exc


def _sha256_regular_file(path: Path) -> tuple[int, str]:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise QStoreComponentBuildError(f"lowering artifact is not a regular file: {path}")
        while value := os.read(descriptor, _COPY_CHUNK_BYTES):
            digest.update(value)
        final = os.fstat(descriptor)
        if (
            int(file_stat.st_dev),
            int(file_stat.st_ino),
            int(file_stat.st_size),
            int(file_stat.st_mtime_ns),
            int(file_stat.st_ctime_ns),
        ) != (
            int(final.st_dev),
            int(final.st_ino),
            int(final.st_size),
            int(final.st_mtime_ns),
            int(final.st_ctime_ns),
        ):
            raise QStoreComponentBuildError(f"lowering artifact changed while hashing: {path}")
        return int(file_stat.st_size), digest.hexdigest()
    finally:
        os.close(descriptor)


def _write_component(
    *,
    source: _VerifiedSource,
    source_descriptors: Mapping[str, int],
    role: str,
    names: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    source_blocks = source.manifest["blocks"]
    destination_blocks: dict[str, dict[str, Any]] = {}
    payload_bytes = 0
    padding_bytes = 0
    handles: dict[str, BinaryIO] = {}
    try:
        for filename in COMPONENT_PAYLOAD_FILES:
            handles[filename] = (output_dir / filename).open("xb")
        for name in sorted(str(value) for value in names):
            raw_block = source_blocks[name]
            block = _json_clone(raw_block)
            alias = block.get("alias")
            if alias is not None:
                target = str(alias)
                seen = {name}
                while "alias" in source_blocks[target]:
                    if target in seen:
                        raise QStoreComponentBuildError(f"cyclic source alias at {name!r}")
                    seen.add(target)
                    target = str(source_blocks[target]["alias"])
                if target not in names:
                    raise QStoreComponentBuildError(
                        f"alias {name!r}->{target!r} crosses component {role!r}"
                    )
                destination_blocks[name] = block
                continue

            kind = str(block.get("kind", ""))
            try:
                layout = _PAYLOAD_LAYOUT[kind]
            except KeyError as exc:
                raise QStoreComponentBuildError(
                    f"unsupported source allocation kind {kind!r} for {name!r}"
                ) from exc
            for filename, offset_key, length_key, alignment in layout:
                source_offset = int(block[offset_key])
                length = int(block[length_key])
                destination = handles[filename]
                destination_offset = int(destination.tell())
                if destination_offset % alignment:
                    raise QStoreComponentBuildError(
                        f"unaligned component offset for {name!r} in {filename}"
                    )
                _copy_exact_span(
                    source_descriptor=source_descriptors[filename],
                    source_offset=source_offset,
                    length=length,
                    destination=destination,
                )
                block[offset_key] = destination_offset
                block[length_key] = length
                payload_bytes += length
            destination_blocks[name] = block

        for filename, handle in handles.items():
            if handle.tell() == 0:
                amount = _MINIMUM_COMPONENT_FILE_BYTES[filename]
                handle.write(b"\x00" * amount)
                padding_bytes += amount
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        for handle in handles.values():
            handle.close()

    component_manifest = {
        "model_name": f"{source.manifest.get('model_name', source.directory.name)}-{role}",
        "arch": source.manifest["arch"],
        "dtype": "int8",
        "tie_word_embeddings": bool(source.manifest.get("tie_word_embeddings", False)),
        "config": _json_clone(source.manifest["config"]),
        "blocks": destination_blocks,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_file(manifest_path, component_manifest)
    manifest_bytes, manifest_sha256 = _sha256_regular_file(manifest_path)
    blobs: dict[str, dict[str, Any]] = {}
    for filename in COMPONENT_PAYLOAD_FILES:
        size, digest = _sha256_regular_file(output_dir / filename)
        blobs[filename] = {"bytes": size, "sha256": digest}
    return {
        "role": role,
        "relative_path": f"components/{role}",
        "allowed_names": sorted(str(value) for value in names),
        "manifest_sha256": manifest_sha256,
        "manifest_bytes": manifest_bytes,
        "blobs": blobs,
        "payload_bytes": payload_bytes,
        "padding_bytes": padding_bytes,
        "semantic_content_sha256": _component_semantic_digest(
            output_dir,
            component_manifest,
            sorted(str(value) for value in names),
        ),
    }


def _coverage_from_spans(
    *,
    logical_blocks: Mapping[str, Mapping[str, Any]],
    routes: Mapping[str, str],
    component_manifests: Mapping[str, Mapping[str, Any]],
    source_payloads: Mapping[str, Mapping[str, Any]],
    component_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    spans: dict[str, list[tuple[int, int, str, str, int]]] = {
        filename: [] for filename in COMPONENT_PAYLOAD_FILES
    }
    role_payload_bytes: dict[str, int] = defaultdict(int)
    aliases: dict[str, str] = {}
    for name, logical_block in logical_blocks.items():
        role = routes[name]
        physical_blocks = component_manifests[role].get("blocks")
        if not isinstance(physical_blocks, Mapping):
            raise QStoreComponentBuildError(f"component {role!r} has no block table")
        physical = physical_blocks.get(name)
        if not isinstance(physical, Mapping):
            raise QStoreComponentBuildError(f"component {role!r} does not own {name!r}")
        if "alias" in logical_block:
            alias = str(logical_block["alias"])
            if physical != {"alias": alias}:
                raise QStoreComponentBuildError(
                    f"component alias {name!r} differs from source alias topology"
                )
            aliases[name] = alias
            continue
        kind = str(logical_block.get("kind", ""))
        try:
            layout = _PAYLOAD_LAYOUT[kind]
        except KeyError as exc:
            raise QStoreComponentBuildError(
                f"unsupported logical allocation kind {kind!r} for {name!r}"
            ) from exc
        for filename, offset_key, length_key, _alignment in layout:
            source_offset = int(logical_block[offset_key])
            length = int(logical_block[length_key])
            destination_offset = int(physical[offset_key])
            if int(physical[length_key]) != length:
                raise QStoreComponentBuildError(
                    f"component allocation length differs from source for {name!r}"
                )
            spans[filename].append(
                (source_offset, source_offset + length, name, role, destination_offset)
            )
            role_payload_bytes[role] += length

    file_results: dict[str, Any] = {}
    descriptors: dict[tuple[str, str], int] = {}
    try:
        for filename in COMPONENT_PAYLOAD_FILES:
            expected = source_payloads.get(filename)
            if not isinstance(expected, Mapping):
                raise QStoreComponentBuildError(f"source lineage omits payload {filename}")
            expected_size = int(expected.get("bytes", 0))
            expected_sha = _require_sha256(
                expected.get("sha256"),
                f"source lineage payload {filename}",
            )
            cursor = 0
            digest = hashlib.sha256()
            ordered = sorted(spans[filename])
            for start, end, name, role, destination_offset in ordered:
                if start != cursor:
                    qualifier = "overlap" if start < cursor else "gap"
                    raise QStoreComponentBuildError(
                        f"source {filename} allocation {qualifier} at {name!r}: {cursor}->{start}"
                    )
                if end <= start:
                    raise QStoreComponentBuildError(
                        f"source {filename} allocation for {name!r} is empty"
                    )
                if component_paths is not None:
                    key = (role, filename)
                    descriptor = descriptors.get(key)
                    if descriptor is None:
                        path = component_paths[role] / filename
                        descriptor = os.open(
                            path,
                            os.O_RDONLY
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                        )
                        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                            os.close(descriptor)
                            raise QStoreComponentBuildError(
                                f"component payload is not regular: {role}/{filename}"
                            )
                        descriptors[key] = descriptor
                    remaining = end - start
                    offset = destination_offset
                    while remaining:
                        amount = min(remaining, _COPY_CHUNK_BYTES)
                        value = os.pread(descriptor, amount, offset)
                        if len(value) != amount:
                            raise QStoreComponentBuildError(
                                f"short component span for {role}/{name!r}"
                            )
                        digest.update(value)
                        remaining -= amount
                        offset += amount
                cursor = end
            if cursor != expected_size:
                raise QStoreComponentBuildError(
                    f"source {filename} coverage ended at {cursor}, expected {expected_size}"
                )
            if component_paths is not None and digest.hexdigest() != expected_sha:
                raise QStoreComponentBuildError(
                    f"component allocations do not reconstruct source payload {filename}"
                )
            file_results[filename] = {
                "source_bytes": expected_size,
                "assigned_bytes": cursor,
                "span_count": len(ordered),
                "complete": True,
            }
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)

    return {
        "logical_block_count": len(logical_blocks),
        "assigned_logical_block_count": len(routes),
        "aliases": dict(sorted(aliases.items())),
        "files": file_results,
        "component_payload_bytes": dict(sorted(role_payload_bytes.items())),
        "source_payload_bytes": sum(record["source_bytes"] for record in file_results.values()),
        "assigned_payload_bytes": sum(record["assigned_bytes"] for record in file_results.values()),
        "complete": True,
    }


def _build_graph(
    *,
    source: _VerifiedSource,
    tokenizer: Mapping[str, Any],
    artifact_id: str,
    temporary_root: Path,
) -> None:
    components_root = temporary_root / "components"
    components_root.mkdir(parents=False, exist_ok=False)
    blocks = source.manifest["blocks"]
    role_names, topology = _allocation_topology(
        blocks,
        declared_tied=bool(source.manifest.get("tie_word_embeddings", False)),
    )
    source_descriptors = _open_source_payloads(source)
    try:
        components = {
            role: _write_component(
                source=source,
                source_descriptors=source_descriptors,
                role=role,
                names=names,
                output_dir=components_root / role,
            )
            for role, names in role_names.items()
        }
    finally:
        for descriptor in source_descriptors.values():
            os.close(descriptor)

    routes = {name: role for role, names in role_names.items() for name in sorted(names)}
    component_manifests = {
        role: _json_clone(
            {
                "model_name": f"{source.manifest.get('model_name', source.directory.name)}-{role}",
                "arch": source.manifest["arch"],
                "dtype": "int8",
                "tie_word_embeddings": bool(source.manifest.get("tie_word_embeddings", False)),
                "config": source.manifest["config"],
                "blocks": {
                    name: _read_component_block(temporary_root, components[role], name)
                    for name in components[role]["allowed_names"]
                },
            }
        )
        for role in components
    }
    coverage = _coverage_from_spans(
        logical_blocks=blocks,
        routes=routes,
        component_manifests=component_manifests,
        source_payloads=source.payloads,
        component_paths={
            role: temporary_root / components[role]["relative_path"] for role in components
        },
    )
    graph: dict[str, Any] = {
        "schema": COMPONENT_GRAPH_SCHEMA,
        "model": str(source.manifest.get("model_name", source.directory.name)),
        "architecture": str(source.manifest["arch"]),
        "body_abi": _body_abi(source.manifest),
        "tokenizer": _json_clone(tokenizer),
        "topology": topology,
        "logical_blocks": _json_clone(blocks),
        "source_lineage": _source_lineage(
            source=source,
            artifact_id=artifact_id,
            tokenizer_semantic_sha256=str(tokenizer["semantic_sha256"]),
        ),
        "components": components,
        "routes": dict(sorted(routes.items())),
        "operation_contracts": _expected_operation_contracts(role_names),
        "coverage": coverage,
    }
    graph["composite_fingerprint_sha256"] = _sha256_bytes(
        canonical_json_bytes(_graph_fingerprint_payload(graph))
    )
    _write_json_file(temporary_root / "model-graph.json", graph)


def _read_component_block(
    graph_root: Path,
    component: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    manifest_path = graph_root / str(component["relative_path"]) / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _json_clone(value["blocks"][name])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise QStoreComponentBuildError(
            f"cannot reopen newly written component allocation {name!r}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_output_root(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    value = root.lstat()
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise QStoreComponentBuildError("component output root must be a real directory")
    return root


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _strict_inspect(
    graph_path: str | Path,
    *,
    require_addressed_parent: bool,
) -> tuple[QStoreComponentArtifact, ComponentGraph]:
    graph = ComponentGraph(graph_path)
    if graph.schema != COMPONENT_GRAPH_SCHEMA:
        raise QStoreComponentBuildError(
            "transitional lowering inspection requires the production component graph schema"
        )
    lineage = graph.raw.get("source_lineage")
    if not isinstance(lineage, Mapping):
        raise QStoreComponentBuildError("component graph has no QStore source lineage")
    required_lineage = {
        "identity_status": "verified-qstore-v3-content-and-blobs",
        "lowering_schema": QSTORE_COMPONENT_LOWERING_SCHEMA,
        "lowering_scope": TRANSITIONAL_SCOPE,
        "direct_source_artifact_lowering": False,
        "source_store_schema_version": QSTORE_SCHEMA,
    }
    for field, expected in required_lineage.items():
        if lineage.get(field) != expected:
            raise QStoreComponentBuildError(f"component graph source lineage mismatch at {field!r}")
    source_identity = {
        name: lineage.get(name)
        for name in (
            "source_checkpoint_sha256",
            "derived_store_sha256",
            "manifest_semantic_sha256",
        )
    }
    source_identity = _source_identity_payload(source_identity)
    tokenizer_sha = _require_sha256(
        lineage.get("tokenizer_semantic_sha256"),
        "source-lineage tokenizer identity",
    )
    if tokenizer_sha != graph.vocab.descriptor_semantic_sha256:
        raise QStoreComponentBuildError("source-lineage tokenizer identity differs from graph")
    observed_artifact_id = _artifact_id(
        source_identity=source_identity,
        tokenizer_semantic_sha256=tokenizer_sha,
    )
    if lineage.get("artifact_id") != observed_artifact_id:
        raise QStoreComponentBuildError("component artifact content address mismatch")
    if require_addressed_parent and graph.path.parent.name != observed_artifact_id:
        raise QStoreComponentBuildError(
            "component artifact directory name does not equal its content address"
        )
    source_payloads = lineage.get("source_payloads")
    if not isinstance(source_payloads, Mapping):
        raise QStoreComponentBuildError("source lineage has no exact source payload records")
    if set(source_payloads) != set(COMPONENT_PAYLOAD_FILES):
        raise QStoreComponentBuildError("source lineage payload set is not exact")

    source_semantic_manifest = lineage.get("source_semantic_manifest")
    if not isinstance(source_semantic_manifest, Mapping):
        raise QStoreComponentBuildError("source lineage has no complete semantic QStore manifest")
    if (
        semantic_manifest_sha256(source_semantic_manifest)
        != source_identity["manifest_semantic_sha256"]
    ):
        raise QStoreComponentBuildError("source semantic manifest identity mismatch")
    derived_records = [
        {
            "name": filename,
            "bytes": int(source_payloads[filename]["bytes"]),
            "sha256": _require_sha256(
                source_payloads[filename].get("sha256"),
                f"source lineage payload {filename}",
            ),
        }
        for filename in sorted(COMPONENT_PAYLOAD_FILES)
    ]
    reconstructed_source_manifest = {
        **_json_clone(source_semantic_manifest),
        "derived": {
            "schema_version": SEMANTIC_DERIVED_SCHEMA,
            "files": derived_records,
            "manifest_semantic_sha256": source_identity["manifest_semantic_sha256"],
            "derived_store_sha256": source_identity["derived_store_sha256"],
        },
    }
    try:
        reconstructed_identity = inspect_semantic_store_identity(
            reconstructed_source_manifest,
            required_store_schema=QSTORE_SCHEMA,
            strict=True,
        )
    except RuntimeError as exc:
        raise QStoreComponentBuildError(
            "source lineage is not a self-authenticating QStore-v3 identity"
        ) from exc
    if _source_identity_payload(reconstructed_identity) != source_identity:
        raise QStoreComponentBuildError("reconstructed QStore source identity differs")
    expected_semantic_fields = {
        "schema_version": QSTORE_SCHEMA,
        "model_name": graph.model_name,
        "arch": graph.architecture,
        "dtype": "int8",
        "tie_word_embeddings": bool(graph.topology["observed_tied"]),
        "config": graph.body_abi["config"],
        "blocks": graph.logical_blocks,
    }
    for field, expected in expected_semantic_fields.items():
        if canonical_json_bytes(source_semantic_manifest.get(field)) != canonical_json_bytes(
            expected
        ):
            raise QStoreComponentBuildError(
                f"source semantic manifest differs from component graph at {field!r}"
            )

    component_paths: dict[str, Path] = {}
    for role in graph.components:
        component_paths[role] = graph.verify_component_blobs(role)
    component_manifests = {
        role: _json_clone(graph._component_manifests[role])  # noqa: SLF001 - custody bridge
        for role in graph.components
    }
    coverage = _coverage_from_spans(
        logical_blocks=graph.logical_blocks,
        routes=graph.routes,
        component_manifests=component_manifests,
        source_payloads=source_payloads,
        component_paths=component_paths,
    )
    if canonical_json_bytes(graph.raw.get("coverage")) != canonical_json_bytes(coverage):
        raise QStoreComponentBuildError("stored allocation coverage differs from reconstruction")
    for role, record in graph.components.items():
        expected_payload_bytes = int(coverage["component_payload_bytes"][role])
        actual_file_bytes = sum(int(blob["bytes"]) for blob in record["blobs"].values())
        expected_padding = actual_file_bytes - expected_payload_bytes
        if record.get("payload_bytes") != expected_payload_bytes:
            raise QStoreComponentBuildError(f"component {role!r} payload accounting differs")
        if record.get("padding_bytes") != expected_padding:
            raise QStoreComponentBuildError(f"component {role!r} padding accounting differs")
        if record.get("manifest_bytes") != (component_paths[role] / "manifest.json").stat().st_size:
            raise QStoreComponentBuildError(f"component {role!r} manifest size differs")
    graph.assert_unchanged()
    return (
        QStoreComponentArtifact(
            artifact_id=observed_artifact_id,
            root=graph.path.parent,
            graph_path=graph.path,
            graph_fingerprint_sha256=graph.fingerprint,
            declared_graph_fingerprint_sha256=graph.declared_fingerprint,
            source_checkpoint_sha256=source_identity["source_checkpoint_sha256"],
            source_derived_store_sha256=source_identity["derived_store_sha256"],
            tokenizer_semantic_sha256=tokenizer_sha,
            architecture=graph.architecture,
            model=graph.model_name,
            roles=tuple(sorted(graph.components)),
            logical_block_count=len(graph.logical_blocks),
            source_payload_bytes=int(coverage["source_payload_bytes"]),
            created=False,
        ),
        graph,
    )


def inspect_qstore_v3_component_artifact(
    graph_path: str | Path,
    *,
    require_addressed_parent: bool = True,
) -> QStoreComponentArtifact:
    """Strictly reopen a transitional artifact and prove exact source reconstruction."""

    result, _graph = _strict_inspect(
        graph_path,
        require_addressed_parent=require_addressed_parent,
    )
    return result


def _assert_existing_matches_inputs(
    *,
    result: QStoreComponentArtifact,
    graph: ComponentGraph,
    source: _VerifiedSource,
    tokenizer: Mapping[str, Any],
) -> None:
    expected_id = _artifact_id(
        source_identity=source.identity,
        tokenizer_semantic_sha256=str(tokenizer["semantic_sha256"]),
    )
    if result.artifact_id != expected_id:
        raise QStoreComponentBuildError("existing addressed artifact belongs to different inputs")
    expected_lineage = _source_lineage(
        source=source,
        artifact_id=expected_id,
        tokenizer_semantic_sha256=str(tokenizer["semantic_sha256"]),
    )
    if canonical_json_bytes(graph.raw["source_lineage"]) != canonical_json_bytes(expected_lineage):
        raise QStoreComponentBuildError("existing artifact source lineage differs from input")
    if canonical_json_bytes(graph.raw["tokenizer"]) != canonical_json_bytes(tokenizer):
        raise QStoreComponentBuildError("existing artifact tokenizer differs from input")


def lower_qstore_v3_to_component_graph(
    *,
    source_dir: str | Path,
    tokenizer: Any,
    output_root: str | Path,
) -> QStoreComponentArtifact:
    """Lower one verified int8 QStore-v3 into an atomically published ComponentGraph.

    ``tokenizer`` may be a tokenizer object or an already-built descriptor accepted by
    :class:`VocabManifest`.  The returned directory is ``output_root/<artifact-id>``.  An
    existing directory is never overwritten: it must pass the same complete reopen and
    source/tokenizer binding checks or the call fails.
    """

    source = _open_verified_source(source_dir)
    config = source.manifest.get("config")
    if not isinstance(config, Mapping):
        raise QStoreComponentBuildError("source QStore has no runtime config")
    configured_rows = config.get("vocab_size")
    if (
        isinstance(configured_rows, bool)
        or not isinstance(configured_rows, int)
        or configured_rows <= 0
    ):
        raise QStoreComponentBuildError("source QStore has no valid vocabulary size")
    descriptor = _coerce_tokenizer_descriptor(
        tokenizer,
        configured_row_count=configured_rows,
    )
    artifact_id = _artifact_id(
        source_identity=source.identity,
        tokenizer_semantic_sha256=str(descriptor["semantic_sha256"]),
    )
    root = _ensure_output_root(output_root)
    destination = root / artifact_id
    graph_path = destination / "model-graph.json"
    if _path_exists(destination):
        if destination.is_symlink() or not destination.is_dir():
            raise QStoreComponentBuildError(
                "existing content-addressed artifact path is not a real directory"
            )
        result, graph = _strict_inspect(graph_path, require_addressed_parent=True)
        _assert_existing_matches_inputs(
            result=result,
            graph=graph,
            source=source,
            tokenizer=descriptor,
        )
        source.assert_unchanged_and_reverify()
        return result

    temporary = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.building-", dir=root))
    published = False
    try:
        _build_graph(
            source=source,
            tokenizer=descriptor,
            artifact_id=artifact_id,
            temporary_root=temporary,
        )
        source.assert_unchanged_and_reverify()
        staged, staged_graph = _strict_inspect(
            temporary / "model-graph.json",
            require_addressed_parent=False,
        )
        _assert_existing_matches_inputs(
            result=staged,
            graph=staged_graph,
            source=source,
            tokenizer=descriptor,
        )
        for role in staged.roles:
            _fsync_directory(temporary / "components" / role)
        _fsync_directory(temporary / "components")
        _fsync_directory(temporary)
        try:
            os.rename(temporary, destination)
            published = True
            _fsync_directory(root)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            result, graph = _strict_inspect(graph_path, require_addressed_parent=True)
            _assert_existing_matches_inputs(
                result=result,
                graph=graph,
                source=source,
                tokenizer=descriptor,
            )
            source.assert_unchanged_and_reverify()
            return result
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)

    result = inspect_qstore_v3_component_artifact(graph_path)
    source.assert_unchanged_and_reverify()
    return replace(result, created=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mrun.engine.component_builder",
        description=(
            "Transitional verified-QStore-v3 to production ComponentGraph lowering; "
            "this is not direct source-artifact native lowering."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="atomically lower one verified QStore-v3")
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--tokenizer", required=True, help="local path or Hugging Face ID")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument(
        "--local-files-only",
        action="store_true",
        help="forbid tokenizer downloads",
    )
    inspect = commands.add_parser("inspect", help="strictly reopen and verify one artifact")
    inspect.add_argument("graph", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Narrow module CLI; top-level ``mrun`` dispatch is intentionally owned elsewhere."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_qstore_v3_component_artifact(args.graph)
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer,
                trust_remote_code=False,
                use_fast=True,
                local_files_only=bool(args.local_files_only),
            )
            result = lower_qstore_v3_to_component_graph(
                source_dir=args.source_dir,
                tokenizer=tokenizer,
                output_root=args.output_root,
            )
    except (ComponentGraphError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"component-builder: {exc}\n")
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the callable CLI
    raise SystemExit(main())
