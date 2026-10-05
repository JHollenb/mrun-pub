"""Durable, checksum-verified storage for compilation bundles."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import CompilationBundle

ARTIFACT_SCHEMA = "mrun-compilation-artifact-v1"


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_key(key: str) -> str:
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise ValueError("artifact key must be a lowercase SHA-256 hex digest")
    return key


@dataclass(frozen=True)
class ArtifactRecord:
    path: Path
    payload_sha256: str
    byte_count: int
    artifact_key: str
    executable_key: str
    bundle_fingerprint: str


class CompilationArtifactStore:
    """Atomic content-addressed store keyed by the complete bundle fingerprint."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, artifact_key: str) -> Path:
        return self.root / f"{_validate_key(artifact_key)}.json"

    def save(self, bundle: CompilationBundle) -> ArtifactRecord:
        key = _validate_key(bundle.fingerprint)
        bundle_payload = bundle.as_dict()
        payload_bytes = _canonical_json(bundle_payload)
        payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        wrapper = {
            "artifact_schema": ARTIFACT_SCHEMA,
            "payload_sha256": payload_sha256,
            "bundle": bundle_payload,
        }
        encoded = _canonical_json(wrapper)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.path_for(key)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{key}.",
            suffix=".tmp",
            dir=self.root,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return ArtifactRecord(
            path=destination,
            payload_sha256=payload_sha256,
            byte_count=len(encoded),
            artifact_key=key,
            executable_key=bundle.lowered.executable_key,
            bundle_fingerprint=bundle.fingerprint,
        )

    def load(self, artifact_key: str) -> CompilationBundle:
        path = self.path_for(artifact_key)
        decoded = json.loads(path.read_bytes())
        if not isinstance(decoded, dict) or decoded.get("artifact_schema") != ARTIFACT_SCHEMA:
            raise ValueError("unsupported compilation artifact schema")
        bundle_payload = decoded.get("bundle")
        if not isinstance(bundle_payload, dict):
            raise TypeError("compilation artifact bundle must be an object")
        actual_sha256 = hashlib.sha256(_canonical_json(bundle_payload)).hexdigest()
        if actual_sha256 != decoded.get("payload_sha256"):
            raise ValueError("compilation artifact checksum mismatch")
        bundle = CompilationBundle.from_dict(bundle_payload)
        if bundle.fingerprint != artifact_key:
            raise ValueError("compilation artifact key mismatch")
        return bundle
