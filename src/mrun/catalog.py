"""Portable model-artifact records used by agents and the scheduler.

The logical model registry in :mod:`mrun.models` answers "what model did the caller
mean?".  This module answers the separate placement question: "which validated bytes
are actually present on this host, where are they, and what variant are they?".

The scanner is deliberately conservative and cheap.  It reads small manifests and
file metadata; it does not hash weight files.  Existing inventory consumers can keep
using ``model``, ``kind``, ``bytes`` and ``path`` while newer consumers use the
artifact identity and locator fields.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError, TypeError):
        return None, None
    return (value if isinstance(value, dict) else None), _sha256_bytes(raw)


def _file_inventory(root: Path) -> tuple[tuple[str, int], ...]:
    """Return bounded identity inputs without reading model contents."""
    rows: list[tuple[str, int]] = []
    try:
        candidates = sorted(path for path in root.rglob("*") if path.is_file())
    except OSError:
        return ()
    for path in candidates:
        try:
            rows.append((path.relative_to(root).as_posix(), int(path.stat().st_size)))
        except (OSError, ValueError):
            continue
    return tuple(rows)


def _bytes(root: Path) -> int:
    return sum(size for _, size in _file_inventory(root))


def _mount_for(path: Path, mounts: Iterable[str]) -> str | None:
    resolved = path.expanduser().resolve()
    candidates = []
    for mount in mounts:
        value = str(mount).rstrip("/") or "/"
        mount_path = Path(value)
        if resolved == mount_path or str(resolved).startswith(value + "/"):
            candidates.append(value)
    return max(candidates, key=len) if candidates else None


def _logical_model_name(raw: Any, registry: Mapping[str, Any]) -> str:
    value = str(raw or "").strip()
    if not value:
        return "unknown"
    lowered = value.lower()
    for name, spec in registry.items():
        aliases = {
            str(name).lower(),
            str(getattr(spec, "name", "")).lower(),
            str(getattr(spec, "hf_id", "")).lower(),
            str(getattr(spec, "hf_id", "")).rsplit("/", 1)[-1].lower(),
        }
        if lowered in aliases:
            return str(name)
    return value


def _artifact_id(
    *,
    model: str,
    kind: str,
    variant: str,
    root: Path,
    manifest: Mapping[str, Any] | None,
    manifest_sha256: str | None,
) -> tuple[str, str | None]:
    derived = manifest.get("derived") if isinstance(manifest, Mapping) else None
    content_hash = None
    if isinstance(derived, Mapping):
        for key in ("derived_store_sha256", "manifest_semantic_sha256"):
            candidate = derived.get(key)
            if isinstance(candidate, str) and candidate:
                content_hash = candidate
                break
    if content_hash:
        return f"artifact:sha256:{content_hash}", content_hash

    # Source snapshots are intentionally inventory-addressed until a model resolver
    # supplies a content digest.  This avoids pretending that a filename/size scan is
    # a cryptographic weight hash while still making mirrors and variants distinct.
    payload = {
        "model": model,
        "kind": kind,
        "variant": variant,
        "manifest_sha256": manifest_sha256,
        "files": _file_inventory(root),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"artifact:inventory-sha256:{_sha256_bytes(encoded)}", None


def _artifact_kind(manifest: Mapping[str, Any], variant: str) -> str:
    text = " ".join(
        str(manifest.get(key, ""))
        for key in ("schema_version", "schema", "codec", "artifact_kind")
    ).lower()
    text = f"{text} {variant.lower()}"
    if "expert" in text or "olmoe" in text:
        return "expert-store"
    if "component" in text or "native" in text or "compiled" in text:
        return "native-components"
    return "qstore"


def _source_model(manifest: Mapping[str, Any]) -> str | None:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        return None
    for key in ("model_name", "source_id", "hf_id", "model_id"):
        value = source.get(key)
        if value:
            return str(value)
    return None


@dataclass(frozen=True)
class ArtifactLocator:
    """A host-local locator; the artifact identity remains portable."""

    host: str | None
    mount: str | None
    path: str

    def as_dict(self) -> dict[str, str | None]:
        return {"host": self.host, "mount": self.mount, "path": self.path}


@dataclass(frozen=True)
class ModelArtifact:
    """One physical model or derived-engine artifact advertised by a host."""

    model: str
    kind: str
    artifact_kind: str
    variant: str
    artifact_id: str
    bytes: int
    path: str
    mount: str | None
    manifest_sha256: str | None = None
    content_hash: str | None = None
    codec: str | None = None
    source_model: str | None = None
    host: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "kind": self.kind,
            "artifact_kind": self.artifact_kind,
            "variant": self.variant,
            "artifact_id": self.artifact_id,
            "bytes": self.bytes,
            "path": self.path,
            "mount": self.mount,
            "manifest_sha256": self.manifest_sha256,
            "content_hash": self.content_hash,
            "codec": self.codec,
            "source_model": self.source_model,
            "host": self.host,
            "locator": ArtifactLocator(self.host, self.mount, self.path).as_dict(),
        }


def artifact_from_directory(
    directory: str | Path,
    *,
    model: str,
    kind: str = "qstore",
    registry: Mapping[str, Any] | None = None,
    mounts: Iterable[str] = (),
    host: str | None = None,
) -> ModelArtifact | None:
    """Inspect one manifest-bearing directory without opening the model."""
    root = Path(directory).expanduser()
    manifest, manifest_sha256 = _manifest(root / "manifest.json")
    if manifest is None:
        return None
    registry = registry or {}
    raw_model = manifest.get("model_name") or manifest.get("model_id") or model
    logical_model = _logical_model_name(raw_model, registry)
    variant = root.name
    artifact_kind = _artifact_kind(manifest, variant)
    artifact_id, content_hash = _artifact_id(
        model=logical_model,
        kind=kind,
        variant=variant,
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    return ModelArtifact(
        model=logical_model,
        kind=kind,
        artifact_kind=artifact_kind,
        variant=variant,
        artifact_id=artifact_id,
        bytes=_bytes(root),
        path=str(root),
        mount=_mount_for(root, mounts),
        manifest_sha256=manifest_sha256,
        content_hash=content_hash,
        codec=(str(manifest.get("codec")) if manifest.get("codec") else None),
        source_model=_source_model(manifest),
        host=host,
    )


def scan_store_artifacts(
    root: str | Path,
    *,
    registry: Mapping[str, Any] | None = None,
    mounts: Iterable[str] = (),
    host: str | None = None,
) -> list[ModelArtifact]:
    """Discover complete first-level derived stores, including suffixed variants."""
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    out: list[ModelArtifact] = []
    try:
        directories = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        return []
    for directory in directories:
        if directory.name.startswith(".") or not (directory / "manifest.json").is_file():
            continue
        artifact = artifact_from_directory(
            directory,
            model=directory.name,
            registry=registry,
            mounts=mounts,
            host=host,
        )
        if artifact is not None:
            out.append(artifact)
    return out


def artifact_from_weights(
    files: Iterable[Path],
    *,
    model: str,
    mounts: Iterable[str] = (),
    host: str | None = None,
) -> ModelArtifact | None:
    """Create a cheap inventory identity for one complete HF snapshot."""
    paths = [Path(path) for path in files if Path(path).is_file()]
    if not paths:
        return None
    root = paths[0].parent
    entries = tuple(sorted((path.name, int(path.stat().st_size)) for path in paths))
    payload = json.dumps({"model": model, "kind": "weights", "files": entries}, sort_keys=True)
    artifact_id = f"artifact:inventory-sha256:{_sha256_bytes(payload.encode())}"
    return ModelArtifact(
        model=model,
        kind="weights",
        artifact_kind="hf-weights",
        variant="hf-snapshot",
        artifact_id=artifact_id,
        bytes=sum(size for _, size in entries),
        path=str(root),
        mount=_mount_for(root, mounts),
        host=host,
    )


__all__ = [
    "ArtifactLocator",
    "ModelArtifact",
    "artifact_from_directory",
    "artifact_from_weights",
    "scan_store_artifacts",
]
