"""Runtime-verified QStore identity binding for compiled artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..store_provenance import canonical_sha256, inspect_semantic_store_identity
from .ir import _is_sha256_digest

_DECLARED_UNVERIFIED_PREFIX = "declared-unverified:"
IDENTITY_CERTIFICATE_SCHEMA = "mrun-loaded-qstore-identity-certificate-v1"


def _declared(
    manifest: Mapping[str, Any],
    section: str,
    key: str,
) -> str | None:
    payload = manifest.get(section)
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    return str(value) if value else None


def _diagnostic_identity(value: str | None, fallback: str) -> str:
    if value:
        return f"{_DECLARED_UNVERIFIED_PREFIX}{value}"
    return fallback


@dataclass(frozen=True)
class BoundQStoreIdentity:
    """Identity values safe to embed in a WorkPlan.

    Bare SHA-256 values are emitted only after the loaded store verified both its
    semantic manifest and every referenced blob. Legacy declarations remain useful
    diagnostics, but their prefix prevents ``DenseWorkPlan`` from treating syntax as
    content verification.
    """

    model_name: str
    model_revision: str
    store_fingerprint: str
    content_identity_verified: bool
    source_identity_status: str
    store_identity_status: str
    identity_status: str
    identity_certificate_sha256: str | None = None
    manifest_semantic_sha256: str | None = None
    builder_source_bundle_sha256: str | None = None
    blob_records_sha256: str | None = None


def _identity_certificate_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the immutable semantic/blob certificate payload declared by QStore v3."""

    schema = str(manifest.get("schema_version", ""))
    if not schema:
        raise RuntimeError("QStore identity certificate requires a store schema")
    inspected = inspect_semantic_store_identity(
        manifest,
        required_store_schema=schema,
        strict=True,
    )
    if not inspected.get("semantic_identity_verified"):
        raise RuntimeError("QStore semantic identity is not verified")
    source = manifest.get("source")
    builder = manifest.get("builder")
    derived = manifest.get("derived")
    if not all(isinstance(value, Mapping) for value in (source, builder, derived)):
        raise RuntimeError("QStore identity certificate requires source, builder, and derived data")
    records = derived.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("QStore identity certificate requires derived blob records")
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("QStore identity certificate blob record is invalid")
        name = record.get("name")
        byte_count = record.get("bytes")
        digest = record.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not _is_sha256_digest(digest)
        ):
            raise RuntimeError("QStore identity certificate blob record is invalid")
        normalized_records.append(
            {
                "name": name,
                "bytes": byte_count,
                "sha256": str(digest),
            }
        )
    if [record["name"] for record in normalized_records] != sorted(
        {record["name"] for record in normalized_records}
    ):
        raise RuntimeError("QStore identity certificate blob records are not canonical")
    source_sha256 = source.get("source_checkpoint_sha256")
    store_sha256 = derived.get("derived_store_sha256")
    semantic_sha256 = derived.get("manifest_semantic_sha256")
    builder_sha256 = builder.get("source_bundle_sha256")
    for field_name, value in (
        ("source_checkpoint_sha256", source_sha256),
        ("derived_store_sha256", store_sha256),
        ("manifest_semantic_sha256", semantic_sha256),
        ("builder_source_bundle_sha256", builder_sha256),
    ):
        if not _is_sha256_digest(value):
            raise RuntimeError(f"QStore identity certificate has invalid {field_name}")
    return {
        "schema_version": IDENTITY_CERTIFICATE_SCHEMA,
        "store_schema_version": schema,
        "source_checkpoint_sha256": str(source_sha256),
        "derived_store_sha256": str(store_sha256),
        "manifest_semantic_sha256": str(semantic_sha256),
        "builder_source_bundle_sha256": str(builder_sha256),
        "blob_records": normalized_records,
        "blob_records_sha256": canonical_sha256(normalized_records),
    }


def identity_certificate_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    payload = _identity_certificate_payload(manifest)
    return payload, canonical_sha256(payload)


def validate_plan_identity_certificate(
    plan: Any,
    manifest: Mapping[str, Any] | None,
) -> None:
    """Require a verified plan's certificate to match the complete compilation manifest."""

    if not bool(getattr(plan, "content_identity_verified", False)):
        return
    if manifest is None:
        raise ValueError("verified WorkPlan compilation requires its QStore manifest")
    payload, certificate_sha256 = identity_certificate_from_manifest(manifest)
    metadata = dict(getattr(plan, "metadata", ()))
    expected = {
        "identity_certificate_sha256": certificate_sha256,
        "manifest_semantic_sha256": payload["manifest_semantic_sha256"],
        "builder_source_bundle_sha256": payload["builder_source_bundle_sha256"],
        "blob_records_sha256": payload["blob_records_sha256"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"WorkPlan {key} does not match its QStore certificate")
    if (
        getattr(plan, "model_revision", None) != payload["source_checkpoint_sha256"]
        or getattr(plan, "store_fingerprint", None) != payload["derived_store_sha256"]
    ):
        raise ValueError("WorkPlan content identities do not match its QStore certificate")


def bind_loaded_qstore_identity(engine: Any) -> BoundQStoreIdentity:
    store = getattr(engine, "store", None)
    manifest = getattr(store, "man", None)
    if not isinstance(manifest, Mapping):
        raise RuntimeError("runtime engine has no QStore manifest for identity binding")
    model_name = str(getattr(engine, "name", manifest.get("model_name", "unknown")))
    declared_source = _declared(
        manifest,
        "source",
        "source_checkpoint_sha256",
    )
    declared_store = _declared(
        manifest,
        "derived",
        "derived_store_sha256",
    )
    verified = bool(getattr(store, "content_identity_verified", False))
    identity_status = str(getattr(store, "identity_status", "legacy-unverified"))
    if verified:
        assert_unchanged = getattr(store, "assert_content_identity_unchanged", None)
        if not callable(assert_unchanged):
            raise RuntimeError("verified QStore has no post-verification file guard")
        assert_unchanged()
        store_identity = getattr(store, "store_identity", None)
        if (
            not isinstance(store_identity, Mapping)
            or store_identity.get("blob_identity_verified") is not True
            or store_identity.get("semantic_identity_verified") is not True
        ):
            raise RuntimeError("verified QStore is missing semantic/blob load verification")
        source = getattr(store, "source_checkpoint_sha256", None)
        derived = getattr(store, "derived_store_sha256", None)
        if not _is_sha256_digest(source) or not _is_sha256_digest(derived):
            raise RuntimeError("verified QStore identity is missing canonical SHA-256 values")
        if declared_source != source or declared_store != derived:
            raise RuntimeError("loaded QStore identity disagrees with its manifest")
        certificate, certificate_sha256 = identity_certificate_from_manifest(manifest)
        semantic_sha256 = getattr(store, "manifest_semantic_sha256", None)
        if semantic_sha256 != certificate["manifest_semantic_sha256"]:
            raise RuntimeError("loaded QStore semantic identity disagrees with its certificate")
        return BoundQStoreIdentity(
            model_name=model_name,
            model_revision=str(source),
            store_fingerprint=str(derived),
            content_identity_verified=True,
            source_identity_status="content-addressed-verified",
            store_identity_status="content-addressed-semantic-verified",
            identity_status=identity_status,
            identity_certificate_sha256=certificate_sha256,
            manifest_semantic_sha256=certificate["manifest_semantic_sha256"],
            builder_source_bundle_sha256=certificate["builder_source_bundle_sha256"],
            blob_records_sha256=certificate["blob_records_sha256"],
        )

    return BoundQStoreIdentity(
        model_name=model_name,
        model_revision=_diagnostic_identity(
            declared_source,
            f"unversioned:{model_name}",
        ),
        store_fingerprint=_diagnostic_identity(
            declared_store,
            f"unfingerprinted:{model_name}",
        ),
        content_identity_verified=False,
        source_identity_status=("declared-unverified" if declared_source else "legacy-unverified"),
        store_identity_status=("declared-unverified" if declared_store else "legacy-unverified"),
        identity_status=identity_status,
    )


def manifest_declaration_matches(bound_value: str, declared_value: object) -> bool:
    """Compare a WorkPlan identity with the manifest declaration it was bound from."""

    if not declared_value:
        return True
    declared = str(declared_value)
    return bound_value in {
        declared,
        f"{_DECLARED_UNVERIFIED_PREFIX}{declared}",
    }
