"""Content-addressed lineage for model-derived stores.

The checkpoint digest is intentionally independent of host-local paths and mutable model aliases.
It binds the authoritative config, optional safetensors index, and every source weight shard.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SOURCE_SCHEMA = "mrun-source-checkpoint-v1"
DERIVED_SCHEMA = "mrun-derived-store-v1"
SEMANTIC_DERIVED_SCHEMA = "mrun-derived-store-v2"
SEMANTIC_MANIFEST_SCHEMA = "mrun-store-semantic-manifest-v1"
BUILDER_SCHEMA = "mrun-store-builder-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _file_record(path: Path, *, name: str | None = None, sha256: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"provenance input is empty: {path}")
    return {
        "name": name or path.name,
        "bytes": size,
        "sha256": sha256 or sha256_file(path),
    }


def discover_revision(model_dir: Path) -> dict[str, Any]:
    """Return portable revision evidence when the local HF layout exposes it."""
    root = model_dir.resolve()
    parts = root.parts
    if "snapshots" in parts:
        position = parts.index("snapshots")
        if position + 1 < len(parts):
            return {"kind": "hf-hub-snapshot", "commits": [parts[position + 1]]}

    metadata_dir = root / ".cache" / "huggingface" / "download"
    commits: set[str] = set()
    if metadata_dir.is_dir():
        for metadata in metadata_dir.glob("*.metadata"):
            try:
                first_line = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
            except (OSError, UnicodeDecodeError, IndexError):
                continue
            if len(first_line) == 40 and all(ch in "0123456789abcdef" for ch in first_line.lower()):
                commits.add(first_line.lower())
    if commits:
        return {
            "kind": "hf-local-dir",
            "commits": sorted(commits),
            "coherent": len(commits) == 1,
        }
    return {"kind": "unknown", "commits": []}


def build_source_provenance(
    model_dir: Path,
    shards: Iterable[Path],
    *,
    model_name: str,
    hf_id: str,
    revision: str | None = None,
) -> dict[str, Any]:
    root = model_dir.resolve()
    config_path = root / "config.json"
    config = _file_record(config_path)

    index_paths = sorted(root.glob("*.safetensors.index.json"))
    if len(index_paths) > 1:
        raise ValueError(f"multiple safetensors indexes in {root}")
    index = _file_record(index_paths[0]) if index_paths else None

    # Keep the logical snapshot filenames even when HF materializes them as symlinks into blobs.
    shard_paths = sorted(
        (Path(path).expanduser().absolute() for path in shards), key=lambda path: path.name
    )
    if not shard_paths:
        raise FileNotFoundError(f"no source safetensors shards in {root}")
    if len({path.name for path in shard_paths}) != len(shard_paths):
        raise ValueError(f"duplicate source shard basenames in {root}")
    if any(path.parent.resolve() != root for path in shard_paths):
        raise ValueError(f"all source shards must be direct children of {root}")

    shard_records = [_file_record(path) for path in shard_paths]
    if index_paths:
        payload = json.loads(index_paths[0].read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors weight_map: {index_paths[0]}")
        expected = sorted({str(value) for value in weight_map.values()})
        actual = [record["name"] for record in shard_records]
        if actual != expected:
            raise ValueError(f"source shard/index mismatch: actual={actual} expected={expected}")

    content = {
        "schema_version": SOURCE_SCHEMA,
        "config": config,
        "index": index,
        "safetensors": shard_records,
    }
    revision_info = (
        {"kind": "declared", "commits": [revision]} if revision else discover_revision(root)
    )
    return {
        **content,
        "model_name": model_name,
        "hf_id": hf_id,
        "revision": revision_info,
        "source_dir": str(root),
        "source_checkpoint_sha256": canonical_sha256(content),
    }


def _source_content(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the portable content record bound by ``source_checkpoint_sha256``.

    Revision discovery and source paths are useful evidence, but they are deliberately not
    content identity. The same checkpoint can be reached through a Hugging Face snapshot on
    one host and a direct immutable model directory on another.
    """

    return {
        "schema_version": source.get("schema_version"),
        "config": source.get("config"),
        "index": source.get("index"),
        "safetensors": source.get("safetensors"),
    }


def validate_source_provenance(source: object) -> Mapping[str, Any]:
    """Validate the internal content address of one source-provenance record."""

    if not isinstance(source, Mapping):
        raise RuntimeError("store has no publication-grade source provenance")
    required = (
        "schema_version",
        "model_name",
        "hf_id",
        "revision",
        "config",
        "index",
        "safetensors",
        "source_checkpoint_sha256",
    )
    missing = [key for key in required if key not in source]
    if missing:
        raise RuntimeError(f"store source provenance is incomplete: {missing}")
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise RuntimeError("store source provenance schema mismatch")
    digest = source.get("source_checkpoint_sha256")
    if not _is_sha256(digest):
        raise RuntimeError("store source checkpoint digest is invalid")
    if digest != canonical_sha256(_source_content(source)):
        raise RuntimeError("store source checkpoint digest does not match its file records")
    return source


def verify_source_provenance(actual: object, expected: Mapping[str, Any]) -> None:
    actual = validate_source_provenance(actual)
    expected = validate_source_provenance(expected)
    # ``revision`` is discovery evidence, not identity. Requiring it to match made a
    # byte-identical checkpoint non-portable between an HF ``snapshots/<commit>`` path and
    # a direct model directory. The authoritative file records and their canonical digest
    # remain exact.
    required_equal = (
        "schema_version",
        "model_name",
        "hf_id",
        "config",
        "index",
        "safetensors",
        "source_checkpoint_sha256",
    )
    for key in required_equal:
        if actual[key] != expected[key]:
            raise RuntimeError(f"existing store source provenance mismatch at {key}")


def semantic_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the portable, execution-semantic portion of a store manifest.

    ``derived`` is excluded to avoid a self-hash. Source paths and revision-discovery
    evidence are normalized away, while the source file records/content digest and every
    other top-level field (including the block table and format-specific knobs) remain
    bound.
    """

    payload = {str(key): value for key, value in manifest.items() if key != "derived"}
    source = payload.get("source")
    if isinstance(source, Mapping):
        portable_source = dict(source)
        portable_source.pop("source_dir", None)
        portable_source.pop("revision", None)
        payload["source"] = portable_source
    return {
        "schema_version": SEMANTIC_MANIFEST_SCHEMA,
        "manifest": payload,
    }


def semantic_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return canonical_sha256(semantic_manifest_payload(manifest))


def build_derived_provenance(
    root: Path,
    filenames: Iterable[str],
    *,
    known_sha256: Mapping[str, str] | None = None,
    semantic_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    known = dict(known_sha256 or {})
    records = [
        _file_record(root / name, name=name, sha256=known.get(name))
        for name in sorted(set(filenames))
    ]
    if semantic_manifest is None:
        content = {"schema_version": DERIVED_SCHEMA, "files": records}
    else:
        content = {
            "schema_version": SEMANTIC_DERIVED_SCHEMA,
            "files": records,
            "manifest_semantic_sha256": semantic_manifest_sha256(semantic_manifest),
        }
    return {**content, "derived_store_sha256": canonical_sha256(content)}


def verify_derived_provenance(
    root: Path,
    actual: object,
    *,
    expected_filenames: Iterable[str],
    semantic_manifest: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(actual, dict):
        raise RuntimeError("existing store has no publication-grade derived-file provenance")
    expected_names = sorted(set(expected_filenames))
    records = actual.get("files")
    expected_schema = DERIVED_SCHEMA if semantic_manifest is None else SEMANTIC_DERIVED_SCHEMA
    if actual.get("schema_version") != expected_schema or not isinstance(records, list):
        raise RuntimeError("existing store derived-file provenance schema mismatch")
    if [record.get("name") for record in records] != expected_names:
        raise RuntimeError("existing store derived-file set mismatch")
    expected = build_derived_provenance(
        root,
        expected_names,
        semantic_manifest=semantic_manifest,
    )
    if actual != expected:
        raise RuntimeError("existing store derived-file hash mismatch")


def inspect_semantic_store_identity(
    manifest: object,
    *,
    required_store_schema: str,
    strict: bool = False,
) -> dict[str, Any]:
    """Inspect a self-authenticating semantic store identity without reading blob files.

    This is a cheap semantic preflight. It never reports ``content_identity_verified``:
    a loader must still call :func:`verify_derived_provenance` once per open to prove the
    declared blob hashes match disk, and only the loaded-store API may promote that verdict.

    Older manifests remain readable diagnostics, but they are never reported as verified
    merely because they happen to contain two 64-character strings.
    """

    declared_source = None
    declared_store = None
    if isinstance(manifest, Mapping):
        source = manifest.get("source")
        derived = manifest.get("derived")
        if isinstance(source, Mapping) and _is_sha256(source.get("source_checkpoint_sha256")):
            declared_source = source["source_checkpoint_sha256"]
        if isinstance(derived, Mapping) and _is_sha256(derived.get("derived_store_sha256")):
            declared_store = derived["derived_store_sha256"]
    has_declared_pair = declared_source is not None and declared_store is not None
    legacy = {
        "content_identity_verified": False,
        "semantic_identity_verified": False,
        "blob_identity_verified": False,
        "identity_status": (
            "declared-content-only-unverified" if has_declared_pair else "legacy-unverified"
        ),
        "source_checkpoint_sha256": declared_source,
        "derived_store_sha256": declared_store,
        "manifest_semantic_sha256": None,
    }
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != required_store_schema:
        return legacy

    try:
        source = validate_source_provenance(manifest.get("source"))
        builder = manifest.get("builder")
        if not isinstance(builder, Mapping):
            raise RuntimeError("store has no publication-grade builder provenance")
        if builder.get("schema_version") != BUILDER_SCHEMA:
            raise RuntimeError("store builder provenance schema mismatch")
        source_files = builder.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            raise RuntimeError("store builder provenance has no source files")
        builder_digest = builder.get("source_bundle_sha256")
        builder_content = {"schema_version": BUILDER_SCHEMA, "files": source_files}
        if not _is_sha256(builder_digest) or builder_digest != canonical_sha256(builder_content):
            raise RuntimeError("store builder source digest does not match its file records")

        derived = manifest.get("derived")
        if not isinstance(derived, Mapping):
            raise RuntimeError("store has no semantic derived provenance")
        if derived.get("schema_version") != SEMANTIC_DERIVED_SCHEMA:
            raise RuntimeError("store semantic derived provenance schema mismatch")
        records = derived.get("files")
        if not isinstance(records, list) or not records:
            raise RuntimeError("store semantic derived provenance has no files")
        names = [record.get("name") for record in records if isinstance(record, Mapping)]
        if len(names) != len(records) or names != sorted(set(names)):
            raise RuntimeError("store semantic derived file records are invalid")
        for record in records:
            if (
                not isinstance(record.get("bytes"), int)
                or record["bytes"] <= 0
                or not _is_sha256(record.get("sha256"))
            ):
                raise RuntimeError("store semantic derived file record is invalid")

        semantic_digest = derived.get("manifest_semantic_sha256")
        expected_semantic = semantic_manifest_sha256(manifest)
        if not _is_sha256(semantic_digest) or semantic_digest != expected_semantic:
            raise RuntimeError("store semantic manifest digest mismatch")
        store_digest = derived.get("derived_store_sha256")
        derived_content = {
            "schema_version": SEMANTIC_DERIVED_SCHEMA,
            "files": records,
            "manifest_semantic_sha256": semantic_digest,
        }
        if not _is_sha256(store_digest) or store_digest != canonical_sha256(derived_content):
            raise RuntimeError("store derived digest does not match its semantic/file records")
    except RuntimeError as error:
        if strict:
            raise
        return {
            **legacy,
            "identity_status": "invalid-semantic-identity",
            "validation_error": str(error),
        }

    return {
        "content_identity_verified": False,
        "semantic_identity_verified": True,
        "blob_identity_verified": False,
        "identity_status": "semantic-manifest-verified-blobs-unverified",
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "derived_store_sha256": derived["derived_store_sha256"],
        "manifest_semantic_sha256": derived["manifest_semantic_sha256"],
    }


def build_builder_provenance(
    module_paths: Iterable[Path],
    *,
    name: str,
    schema_version: str,
    quantization: Mapping[str, Any],
) -> dict[str, Any]:
    package_parent = Path(__file__).resolve().parent.parent
    paths = {Path(path).resolve() for path in module_paths}
    paths.add(Path(__file__).resolve())
    source_files = []
    for path in sorted(paths, key=str):
        try:
            logical_name = str(path.relative_to(package_parent))
        except ValueError:
            logical_name = path.name
        source_files.append(_file_record(path, name=logical_name))
    source_content = {"schema_version": BUILDER_SCHEMA, "files": source_files}
    return {
        "schema_version": BUILDER_SCHEMA,
        "name": name,
        "build_schema_version": schema_version,
        "source_files": source_files,
        "source_bundle_sha256": canonical_sha256(source_content),
        "quantization": dict(quantization),
    }


def verify_builder_provenance(actual: object, expected: Mapping[str, Any]) -> None:
    if not isinstance(actual, dict):
        raise RuntimeError("existing store has no publication-grade builder provenance")
    if actual != expected:
        raise RuntimeError("existing store builder/quantization provenance mismatch")
