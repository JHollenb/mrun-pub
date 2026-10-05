"""Deterministic semantic QStore identity fixtures for compiler unit tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..engine.kernels.qstore_build import QSTORE_SCHEMA
from ..store_provenance import (
    BUILDER_SCHEMA,
    SEMANTIC_DERIVED_SCHEMA,
    SOURCE_SCHEMA,
    canonical_sha256,
    semantic_manifest_sha256,
)


def verified_test_manifest(base: Mapping[str, Any]) -> dict[str, Any]:
    """Attach internally valid v3 provenance without creating model blobs on disk."""

    manifest = dict(base)
    manifest["schema_version"] = QSTORE_SCHEMA
    source_content = {
        "schema_version": SOURCE_SCHEMA,
        "config": {
            "name": "config.json",
            "bytes": 1,
            "sha256": "1" * 64,
        },
        "index": None,
        "safetensors": [
            {
                "name": "model.safetensors",
                "bytes": 1,
                "sha256": "2" * 64,
            }
        ],
    }
    manifest["source"] = {
        **source_content,
        "model_name": str(manifest.get("model_name", "tiny")),
        "hf_id": "tests/tiny",
        "revision": {"kind": "test", "commits": []},
        "source_checkpoint_sha256": canonical_sha256(source_content),
    }
    source_files = [
        {
            "name": "test_builder.py",
            "bytes": 1,
            "sha256": "3" * 64,
        }
    ]
    builder_content = {
        "schema_version": BUILDER_SCHEMA,
        "files": source_files,
    }
    manifest["builder"] = {
        "schema_version": BUILDER_SCHEMA,
        "name": "test-builder",
        "build_schema_version": QSTORE_SCHEMA,
        "source_files": source_files,
        "source_bundle_sha256": canonical_sha256(builder_content),
        "quantization": {"dtype": "int8"},
    }
    blob_records = [
        {"name": "extras.f32", "bytes": 1, "sha256": "4" * 64},
        {"name": "scales.f32", "bytes": 1, "sha256": "5" * 64},
        {"name": "weights.i8", "bytes": 1, "sha256": "6" * 64},
    ]
    semantic_sha256 = semantic_manifest_sha256(manifest)
    derived_content = {
        "schema_version": SEMANTIC_DERIVED_SCHEMA,
        "files": blob_records,
        "manifest_semantic_sha256": semantic_sha256,
    }
    manifest["derived"] = {
        **derived_content,
        "derived_store_sha256": canonical_sha256(derived_content),
    }
    return manifest


def install_verified_test_identity(store: Any) -> None:
    """Populate the loaded-store fields required by compiler identity binding."""

    manifest = store.man
    store.content_identity_verified = True
    store.identity_status = "content-addressed-semantic-v2-files-verified"
    store.source_checkpoint_sha256 = manifest["source"]["source_checkpoint_sha256"]
    store.derived_store_sha256 = manifest["derived"]["derived_store_sha256"]
    store.manifest_semantic_sha256 = manifest["derived"]["manifest_semantic_sha256"]
    store.store_identity = {
        "content_identity_verified": True,
        "semantic_identity_verified": True,
        "blob_identity_verified": True,
        "identity_status": store.identity_status,
        "source_checkpoint_sha256": store.source_checkpoint_sha256,
        "derived_store_sha256": store.derived_store_sha256,
        "manifest_semantic_sha256": store.manifest_semantic_sha256,
    }
    store.content_verified_at_ns = 1
    store.assert_content_identity_unchanged = lambda: None
    store.reverify_content_identity = lambda: dict(store.store_identity)
