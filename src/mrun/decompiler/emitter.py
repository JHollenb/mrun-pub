"""Deterministic, byte-preserving component artifacts for decoded model sources.

This module is deliberately narrower than a backend lowering pass.  It turns a successful U2
``DecompileResult`` into an immutable set of independently addressable physical allocations while
preserving every source byte, logical view, classification, and alias class.  It does not convert
codecs and it does not claim that a backend has executed the artifact.

The supported source contract is exact:

* adapter codec ``raw-float`` version ``1.0.0``;
* safetensors storage dtype ``F16``, ``BF16``, ``F32``, or ``F64``;
* little-endian, unpacked, byte-preserving range extraction.

Every other codec, codec version, packing, or storage dtype is rejected.  In particular this is
not a GPTQ/AWQ/bitsandbytes/QStore converter.  Those formats require separately registered value
decoders and value-level certification.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ._json import (
    canonical_json_bytes,
    canonical_sha256,
    require_bool,
    require_dict,
    require_exact_keys,
    require_int,
    require_list,
    require_sha256,
    require_str,
    strict_json_loads,
)
from .compiler import decompile_source
from .errors import DecompilerError
from .ir import IRBundle
from .reports import DecompileReport, DecompileResult
from .source import FrozenSourceBundle, SourcePolicy
from .tensor_index import TensorIndex

COMPONENT_ARTIFACT_SCHEMA = "mrun-native-source-component-v1"
COMPONENT_BUILD_RECORD_SCHEMA = "mrun-component-build-record-v1"
COMPONENT_CERTIFICATION_SCHEMA = "mrun-component-certification-v1"
COMPONENT_ARTIFACT_KIND = "canonical-native-source-components"
COMPONENT_ARTIFACT_STATUS = "built-unexecuted"
EMITTER_ID = "mrun.decompiler.native-source-component-emitter"
EMITTER_VERSION = "1.0.0"
ARTIFACT_CODEC_ID = "native-source-range"
ARTIFACT_CODEC_VERSION = "1.0.0"
SUPPORTED_SOURCE_CODEC_ID = "raw-float"
SUPPORTED_SOURCE_CODEC_VERSION = "1.0.0"
SUPPORTED_STORED_DTYPES = ("BF16", "F16", "F32", "F64")

_JSON_LIMIT_BYTES = 512 * 1024 * 1024
_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_SAFE_ALLOCATION_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_PAYLOAD_FILENAMES = (
    "decompile-report.json",
    "ir.json",
    "source.json",
    "tensor-index.json",
)
_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_id",
    "artifact_kind",
    "status",
    "execution_certified",
    "source_lineage",
    "compiler_lineage",
    "artifact_codec",
    "payloads",
    "assets",
    "allocations",
    "views",
    "alias_classes",
    "classifications",
    "coverage",
    "decompile_pending_gates",
}


class ArtifactEmissionError(DecompilerError):
    """The decoded source cannot be emitted under this artifact contract."""

    code = "artifact_emission_failure"
    gate = "G7"


class ArtifactVerificationError(DecompilerError):
    """A component artifact failed strict structural or content verification."""

    code = "artifact_verification_failure"
    gate = "G7"


class ExecutionCertificationUnavailable(DecompilerError):
    """No execution runner exists for the canonical source-range artifact."""

    code = "execution_certification_unavailable"
    gate = "G8"


def _canonical_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, expected_bytes: int | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactVerificationError(
            "cannot open artifact payload without following links",
            details={"path": path.name},
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactVerificationError(
                "artifact payload is not a regular file", details={"path": path.name}
            )
        while True:
            block = os.read(descriptor, _COPY_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
        if _stat_signature(before) != _stat_signature(after):
            raise ArtifactVerificationError(
                "artifact payload changed while it was verified", details={"path": path.name}
            )
    finally:
        os.close(descriptor)
    if expected_bytes is not None and byte_count != expected_bytes:
        raise ArtifactVerificationError(
            "artifact payload byte count differs from its manifest",
            details={"path": path.name, "expected": expected_bytes, "actual": byte_count},
        )
    return digest.hexdigest(), byte_count


def _stat_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _write_exclusive(path: Path, payload: bytes) -> dict[str, Any]:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ArtifactEmissionError(
            "refusing to overwrite an artifact staging payload", details={"path": path.name}
        ) from exc
    return {"byte_count": len(payload), "sha256": _sha256_bytes(payload)}


def _copy_source_range(
    source: FrozenSourceBundle,
    *,
    source_file: str,
    byte_offset: int,
    byte_length: int,
    destination: Path,
) -> tuple[str, int]:
    """Copy one guarded absolute source range with bounded memory and descriptor rechecks."""

    source_record = source.file(source_file)
    if byte_offset < 0 or byte_length < 0 or byte_offset + byte_length > source_record.byte_count:
        raise ArtifactEmissionError(
            "physical allocation exceeds its frozen source file",
            details={
                "source_file": source_file,
                "byte_offset": byte_offset,
                "byte_length": byte_length,
                "source_file_bytes": source_record.byte_count,
            },
        )
    source_path = source.file_path(source_file)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source_path, flags)
    except OSError as exc:
        raise ArtifactEmissionError(
            "cannot open a frozen source shard without following links",
            details={"source_file": source_file},
        ) from exc
    digest = hashlib.sha256()
    copied = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != source_record.byte_count:
            raise ArtifactEmissionError(
                "source shard identity differs from the frozen inventory",
                details={"source_file": source_file},
            )
        os.lseek(descriptor, byte_offset, os.SEEK_SET)
        try:
            output = destination.open("xb")
        except FileExistsError as exc:
            raise ArtifactEmissionError(
                "refusing to overwrite an allocation blob",
                details={"allocation_blob": destination.name},
            ) from exc
        with output:
            remaining = byte_length
            while remaining:
                block = os.read(descriptor, min(_COPY_CHUNK_BYTES, remaining))
                if not block:
                    raise ArtifactEmissionError(
                        "source shard ended inside a declared allocation",
                        details={"source_file": source_file, "copied_bytes": copied},
                    )
                output.write(block)
                digest.update(block)
                copied += len(block)
                remaining -= len(block)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(descriptor)
        if _stat_signature(before) != _stat_signature(after):
            raise ArtifactEmissionError(
                "source shard changed while an allocation was emitted",
                details={"source_file": source_file},
            )
    finally:
        os.close(descriptor)
    if copied != byte_length:
        raise ArtifactEmissionError(
            "allocation copy was incomplete",
            details={"expected_bytes": byte_length, "copied_bytes": copied},
        )
    return digest.hexdigest(), copied


def _artifact_codec_manifest() -> dict[str, Any]:
    return {
        "artifact_codec_id": ARTIFACT_CODEC_ID,
        "artifact_codec_version": ARTIFACT_CODEC_VERSION,
        "transformation": "byte-preserving-absolute-source-range-extraction",
        "numerical_effect": "none",
        "supported_source_contracts": [
            {
                "codec_id": SUPPORTED_SOURCE_CODEC_ID,
                "codec_version": SUPPORTED_SOURCE_CODEC_VERSION,
                "stored_dtypes": list(SUPPORTED_STORED_DTYPES),
                "byte_order": "little",
                "packing": "none",
            }
        ],
        "unsupported_policy": (
            "reject every codec ID, codec version, dtype, byte order, or packing outside the "
            "single supported source contract; no implicit conversion"
        ),
        "backend_lowering_status": "not-lowered",
    }


def _validate_source_codec(allocation: Any) -> None:
    codec = allocation.codec
    parameters = codec.parameters
    supported = (
        codec.codec_id == SUPPORTED_SOURCE_CODEC_ID
        and codec.codec_version == SUPPORTED_SOURCE_CODEC_VERSION
        and codec.stored_dtype == allocation.stored_dtype
        and allocation.stored_dtype in SUPPORTED_STORED_DTYPES
        and parameters
        == {
            "byte_order": "little",
            "packing": "none",
            "value_semantics": "safetensors-native-float",
        }
    )
    if not supported:
        raise ArtifactEmissionError(
            "source allocation codec is outside the byte-preserving emitter contract",
            details={
                "allocation_id": allocation.allocation_id,
                "source_tensor": allocation.source_tensor,
                "codec": codec.as_dict(),
                "supported_codec_id": SUPPORTED_SOURCE_CODEC_ID,
                "supported_codec_version": SUPPORTED_SOURCE_CODEC_VERSION,
                "supported_stored_dtypes": list(SUPPORTED_STORED_DTYPES),
            },
        )


def _payload_descriptor(payload: bytes) -> dict[str, Any]:
    return {"byte_count": len(payload), "sha256": _sha256_bytes(payload)}


def _manifest_identity_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "artifact_id"}


def _artifact_relative_path(value: Any, *, field: str) -> str:
    candidate = require_str(value, field=field)
    if "\\" in candidate:
        raise ArtifactVerificationError(f"{field} is not a portable relative path")
    path = PurePosixPath(candidate)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactVerificationError(f"{field} is not a portable relative path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class ArtifactBuildRecord:
    path: Path
    artifact_id: str
    manifest_sha256: str
    allocation_count: int
    emitted_blob_bytes: int
    verified_reopen: bool
    execution_certified: bool = False
    schema_version: str = COMPONENT_BUILD_RECORD_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "artifact_id": self.artifact_id,
            "manifest_sha256": self.manifest_sha256,
            "allocation_count": self.allocation_count,
            "emitted_blob_bytes": self.emitted_blob_bytes,
            "verified_reopen": self.verified_reopen,
            "execution_certified": self.execution_certified,
        }


@dataclass(frozen=True, slots=True)
class ComponentArtifact:
    directory: Path
    _manifest_json: str
    manifest_sha256: str
    source: FrozenSourceBundle
    tensor_index: TensorIndex
    ir_bundle: IRBundle
    decompile_report: DecompileReport

    @property
    def manifest(self) -> dict[str, Any]:
        value = strict_json_loads(self._manifest_json, field="verified component manifest")
        assert isinstance(value, dict)
        return value

    @property
    def artifact_id(self) -> str:
        return str(self.manifest["artifact_id"])

    @property
    def execution_certified(self) -> bool:
        return False

    def report(self) -> dict[str, Any]:
        coverage = self.manifest["coverage"]
        model = self.ir_bundle.model
        return {
            "schema_version": "mrun-component-artifact-report-v1",
            "status": "verified-built-unexecuted",
            "artifact_id": self.artifact_id,
            "artifact_kind": self.manifest["artifact_kind"],
            "manifest_sha256": self.manifest_sha256,
            "source_fingerprint": self.source.fingerprint,
            "tensor_index_fingerprint": self.tensor_index.fingerprint,
            "ir_bundle_fingerprint": self.ir_bundle.fingerprint,
            "adapter_id": model.adapter_id,
            "adapter_version": model.adapter_version,
            "architecture_id": model.architecture_id,
            "allocation_count": coverage["emitted_allocation_count"],
            "logical_view_count": coverage["logical_view_count"],
            "alias_class_count": coverage["alias_class_count"],
            "emitted_blob_bytes": coverage["emitted_blob_bytes"],
            "source_tensor_bytes": coverage["source_tensor_bytes"],
            "auxiliary_asset_count": coverage["emitted_auxiliary_asset_count"],
            "auxiliary_asset_bytes": coverage["emitted_auxiliary_asset_bytes"],
            "codec": self.manifest["artifact_codec"],
            "structural_reopen_verified": True,
            "execution_certified": False,
            "certification_boundary": (
                "artifact custody and byte-preserving emission are verified; no backend "
                "execution, numerical parity, generation, or chat gate has run"
            ),
            "decompile_pending_gates": self.manifest["decompile_pending_gates"],
        }


@dataclass(frozen=True, slots=True)
class CertificationRecord:
    artifact_id: str
    manifest_sha256: str
    status: str
    scope: str
    checks: tuple[str, ...]
    execution_performed: bool
    execution_certified: bool
    certification_boundary: str
    fingerprint: str
    schema_version: str = COMPONENT_CERTIFICATION_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.artifact_id, field="artifact_id")
        require_sha256(self.manifest_sha256, field="manifest_sha256")
        if self.status != "passed" or self.scope != "artifact-integrity-only":
            raise ValueError("component certification has an unsupported status or scope")
        if self.execution_performed or self.execution_certified:
            raise ValueError("integrity-only certification cannot claim execution")
        if self.checks != tuple(sorted(set(self.checks))) or not self.checks:
            raise ValueError("certification checks must be sorted and unique")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ValueError("certification fingerprint does not match its payload")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "manifest_sha256": self.manifest_sha256,
            "status": self.status,
            "scope": self.scope,
            "checks": list(self.checks),
            "execution_performed": self.execution_performed,
            "execution_certified": self.execution_certified,
            "certification_boundary": self.certification_boundary,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def build(cls, artifact: ComponentArtifact) -> CertificationRecord:
        payload = {
            "schema_version": COMPONENT_CERTIFICATION_SCHEMA,
            "artifact_id": artifact.artifact_id,
            "manifest_sha256": artifact.manifest_sha256,
            "status": "passed",
            "scope": "artifact-integrity-only",
            "checks": tuple(
                sorted(
                    {
                        "alias-preservation",
                        "allocation-byte-hashes",
                        "canonical-json-payloads",
                        "complete-source-tensor-coverage",
                        "exact-file-inventory",
                        "lineage-cross-binding",
                        "manifest-content-identity",
                        "payload-byte-hashes",
                        "structural-ir-reopen",
                    }
                )
            ),
            "execution_performed": False,
            "execution_certified": False,
            "certification_boundary": (
                "passed custody, completeness, identity, and structural reopen only; no model "
                "backend was invoked and no execution or numerical parity is certified"
            ),
        }
        return cls(**payload, fingerprint=canonical_sha256(payload))


class NativeSourceComponentEmitter:
    """Emit and reopen exact source-range component artifacts."""

    def build(self, result: DecompileResult, output_root: str | Path) -> ArtifactBuildRecord:
        if not result.succeeded:
            raise ArtifactEmissionError(
                "component emission requires a successful U2 decompilation",
                details={"decompile_report": result.report.as_dict()},
            )
        source = result.source
        index = result.tensor_index
        bundle = result.ir_bundle
        assert source is not None and index is not None and bundle is not None
        if source.root is None:
            raise ArtifactEmissionError("component emission requires an attached source root")
        if result.report.coverage is None or not result.report.coverage.complete:
            raise ArtifactEmissionError("component emission requires complete source coverage")

        root = Path(output_root).absolute()
        source_root = source.root.resolve()
        unresolved_root = root.resolve(strict=False)
        if unresolved_root == source_root or source_root in unresolved_root.parents:
            raise ArtifactEmissionError("artifact output root cannot be inside the frozen source")
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ArtifactEmissionError("artifact output root must be a non-symlink directory")
        root = root.resolve()
        if root == source_root or source_root in root.parents:
            raise ArtifactEmissionError("artifact output root cannot be inside the frozen source")

        allocations = bundle.physical_weights.allocations
        if len(allocations) != len(index.tensors):
            raise ArtifactEmissionError(
                "physical allocations do not exactly cover source tensor records",
                details={"allocations": len(allocations), "source_tensors": len(index.tensors)},
            )
        for allocation in allocations:
            _validate_source_codec(allocation)
            if not _SAFE_ALLOCATION_ID.fullmatch(allocation.allocation_id):
                raise ArtifactEmissionError(
                    "allocation ID is unsafe for deterministic artifact naming",
                    details={"allocation_id": allocation.allocation_id},
                )

        stage = Path(tempfile.mkdtemp(prefix=".mrun-component-stage-", dir=root))
        published = False
        lock_path: Path | None = None
        try:
            blobs = stage / "blobs"
            blobs.mkdir()
            source.assert_stat_unchanged()
            emitted_allocations: list[dict[str, Any]] = []
            emitted_bytes = 0
            for allocation in allocations:
                blob_relative = f"blobs/{allocation.allocation_id}.bin"
                blob_path = stage / blob_relative
                blob_sha256, blob_bytes = _copy_source_range(
                    source,
                    source_file=allocation.source_file,
                    byte_offset=allocation.byte_offset,
                    byte_length=allocation.byte_length,
                    destination=blob_path,
                )
                source_file = source.file(allocation.source_file)
                emitted_allocations.append(
                    {
                        "allocation_id": allocation.allocation_id,
                        "source_tensor": allocation.source_tensor,
                        "source_file": allocation.source_file,
                        "source_file_sha256": source_file.sha256,
                        "source_byte_offset": allocation.byte_offset,
                        "source_byte_length": allocation.byte_length,
                        "source_range_fingerprint": allocation.content_fingerprint,
                        "stored_shape": list(allocation.stored_shape),
                        "stored_dtype": allocation.stored_dtype,
                        "source_codec": allocation.codec.as_dict(),
                        "blob": {
                            "path": blob_relative,
                            "byte_count": blob_bytes,
                            "sha256": blob_sha256,
                        },
                    }
                )
                emitted_bytes += blob_bytes
            source.assert_unchanged()

            payload_values = {
                "source.json": source.as_dict(),
                "tensor-index.json": index.as_dict(),
                "ir.json": bundle.as_dict(),
                "decompile-report.json": result.report.as_dict(),
            }
            payloads: dict[str, dict[str, Any]] = {}
            for name in _PAYLOAD_FILENAMES:
                encoded = _canonical_file_bytes(payload_values[name])
                payloads[name] = _write_exclusive(stage / name, encoded)

            emitted_assets: list[dict[str, Any]] = []
            emitted_asset_bytes = 0
            for source_file in source.files:
                if source_file.role == "weight-shard":
                    continue
                artifact_path = f"assets/{source_file.path}"
                destination = stage / artifact_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                asset_sha256, asset_bytes = _copy_source_range(
                    source,
                    source_file=source_file.path,
                    byte_offset=0,
                    byte_length=source_file.byte_count,
                    destination=destination,
                )
                if asset_sha256 != source_file.sha256 or asset_bytes != source_file.byte_count:
                    raise ArtifactEmissionError(
                        "auxiliary source asset changed during byte-preserving emission",
                        details={"source_file": source_file.path},
                    )
                emitted_assets.append(
                    {
                        "source_path": source_file.path,
                        "role": source_file.role,
                        "source_sha256": source_file.sha256,
                        "byte_count": source_file.byte_count,
                        "artifact_path": artifact_path,
                    }
                )
                emitted_asset_bytes += asset_bytes
            source.assert_unchanged()

            weights = bundle.physical_weights
            manifest_without_id: dict[str, Any] = {
                "schema_version": COMPONENT_ARTIFACT_SCHEMA,
                "artifact_kind": COMPONENT_ARTIFACT_KIND,
                "status": COMPONENT_ARTIFACT_STATUS,
                "execution_certified": False,
                "source_lineage": {
                    "source_id": source.source_id,
                    "resolved_revision": source.resolved_revision,
                    "revision_immutable": source.revision_immutable,
                    "source_fingerprint": source.fingerprint,
                    "tensor_index_fingerprint": index.fingerprint,
                    "source_asset_count": len(source.files),
                    "source_asset_bytes": sum(item.byte_count for item in source.files),
                },
                "compiler_lineage": {
                    "adapter_id": bundle.model.adapter_id,
                    "adapter_version": bundle.model.adapter_version,
                    "adapter_fingerprint": bundle.model.adapter_fingerprint,
                    "decompile_report_fingerprint": result.report.fingerprint,
                    "physical_weights_fingerprint": weights.fingerprint,
                    "model_ir_fingerprint": bundle.model.fingerprint,
                    "state_ir_fingerprint": bundle.state.fingerprint,
                    "io_ir_fingerprint": bundle.io.fingerprint,
                    "ir_bundle_fingerprint": bundle.fingerprint,
                    "emitter_id": EMITTER_ID,
                    "emitter_version": EMITTER_VERSION,
                },
                "artifact_codec": _artifact_codec_manifest(),
                "payloads": payloads,
                "assets": emitted_assets,
                "allocations": emitted_allocations,
                "views": [item.as_dict() for item in weights.views],
                "alias_classes": [item.as_dict() for item in weights.alias_classes],
                "classifications": [item.as_dict() for item in weights.classifications],
                "coverage": {
                    "source_tensor_count": len(index.tensors),
                    "emitted_allocation_count": len(emitted_allocations),
                    "source_tensor_bytes": index.total_tensor_bytes,
                    "emitted_blob_bytes": emitted_bytes,
                    "logical_view_count": len(weights.views),
                    "alias_class_count": len(weights.alias_classes),
                    "classification_count": len(weights.classifications),
                    "source_auxiliary_asset_count": len(emitted_assets),
                    "emitted_auxiliary_asset_count": len(emitted_assets),
                    "source_auxiliary_asset_bytes": emitted_asset_bytes,
                    "emitted_auxiliary_asset_bytes": emitted_asset_bytes,
                    "complete": True,
                },
                "decompile_pending_gates": list(result.report.pending_gates),
            }
            artifact_id = canonical_sha256(manifest_without_id)
            manifest = {**manifest_without_id, "artifact_id": artifact_id}
            manifest_payload = _canonical_file_bytes(manifest)
            _write_exclusive(stage / "manifest.json", manifest_payload)

            # Verify the complete staged artifact before it can become visible under its ID.
            staged = _open_component_artifact(stage, enforce_directory_identity=False)
            if staged.artifact_id != artifact_id:
                raise ArtifactEmissionError("staged artifact reopened under a different identity")
            source.assert_unchanged()

            destination = root / artifact_id
            if destination.exists() or destination.is_symlink():
                raise ArtifactEmissionError(
                    "refusing to overwrite an existing component artifact",
                    details={"artifact_id": artifact_id, "path": str(destination)},
                )
            lock_path = root / f".{artifact_id}.publish.lock"
            try:
                lock_descriptor = os.open(
                    lock_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                )
            except FileExistsError as exc:
                raise ArtifactEmissionError(
                    "another publisher already owns the artifact identity lock",
                    details={"artifact_id": artifact_id},
                ) from exc
            try:
                os.write(lock_descriptor, f"{artifact_id}\n".encode("ascii"))
                os.fsync(lock_descriptor)
            finally:
                os.close(lock_descriptor)
            if destination.exists() or destination.is_symlink():
                raise ArtifactEmissionError(
                    "refusing to overwrite an artifact created during publication",
                    details={"artifact_id": artifact_id},
                )
            source.assert_stat_unchanged()
            os.rename(stage, destination)
            published = True
            directory_descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            verified = open_component_artifact(destination)
            return ArtifactBuildRecord(
                path=destination,
                artifact_id=artifact_id,
                manifest_sha256=verified.manifest_sha256,
                allocation_count=len(emitted_allocations),
                emitted_blob_bytes=emitted_bytes,
                verified_reopen=True,
                execution_certified=False,
            )
        finally:
            if lock_path is not None and lock_path.exists():
                lock_path.unlink()
            if not published and stage.exists():
                shutil.rmtree(stage)


def _read_canonical_json(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        size = path.lstat().st_size
    except OSError as exc:
        raise ArtifactVerificationError(
            "required artifact JSON is absent", details={"path": path.name}
        ) from exc
    if size <= 0 or size > _JSON_LIMIT_BYTES:
        raise ArtifactVerificationError(
            "artifact JSON is empty or exceeds its parsing bound",
            details={"path": path.name, "byte_count": size, "limit": _JSON_LIMIT_BYTES},
        )
    digest, observed = _sha256_file(path, expected_bytes=size)
    del digest, observed
    raw = path.read_bytes()
    try:
        value = strict_json_loads(raw, field=field)
    except (TypeError, ValueError) as exc:
        raise ArtifactVerificationError(
            "artifact JSON cannot be decoded strictly", details={"path": path.name}
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactVerificationError(
            "artifact JSON must contain an object", details={"path": path.name}
        )
    if raw != _canonical_file_bytes(value):
        raise ArtifactVerificationError(
            "artifact JSON is not in canonical byte form", details={"path": path.name}
        )
    return value, raw


def _validate_exact_inventory(directory: Path, manifest: dict[str, Any]) -> None:
    expected_files = {"manifest.json", *_PAYLOAD_FILENAMES}
    assets = require_list(manifest["assets"], field="artifact assets")
    for item in assets:
        record = require_dict(item, field="artifact asset")
        expected_files.add(_artifact_relative_path(record.get("artifact_path"), field="asset path"))
    allocations = require_list(manifest["allocations"], field="artifact allocations")
    for item in allocations:
        record = require_dict(item, field="artifact allocation")
        blob = require_dict(record.get("blob"), field="allocation blob")
        expected_files.add(_artifact_relative_path(blob.get("path"), field="allocation blob path"))
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ArtifactVerificationError(
                "component artifacts cannot contain symbolic links", details={"path": relative}
            )
        if stat.S_ISDIR(info.st_mode):
            actual_directories.add(relative)
        elif stat.S_ISREG(info.st_mode):
            actual_files.add(relative)
        else:
            raise ArtifactVerificationError(
                "component artifact contains a non-regular filesystem entry",
                details={"path": relative},
            )
    expected_directories = {"blobs"}
    for expected_file in expected_files:
        parent = Path(expected_file).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_directories != expected_directories or actual_files != expected_files:
        raise ArtifactVerificationError(
            "component artifact file inventory differs from its manifest",
            details={
                "expected_files": sorted(expected_files),
                "actual_files": sorted(actual_files),
                "expected_directories": sorted(expected_directories),
                "actual_directories": sorted(actual_directories),
            },
        )


def _validate_manifest_shape(manifest: dict[str, Any]) -> None:
    require_exact_keys(manifest, _MANIFEST_FIELDS, field="component manifest")
    if manifest["schema_version"] != COMPONENT_ARTIFACT_SCHEMA:
        raise ArtifactVerificationError("unsupported component artifact schema")
    if manifest["artifact_kind"] != COMPONENT_ARTIFACT_KIND:
        raise ArtifactVerificationError("unsupported component artifact kind")
    if manifest["status"] != COMPONENT_ARTIFACT_STATUS:
        raise ArtifactVerificationError("component artifact has an invalid build status")
    if require_bool(manifest["execution_certified"], field="execution_certified"):
        raise ArtifactVerificationError(
            "a source-range artifact cannot declare execution certification"
        )
    artifact_id = require_sha256(manifest["artifact_id"], field="artifact_id")
    if artifact_id != canonical_sha256(_manifest_identity_payload(manifest)):
        raise ArtifactVerificationError("component artifact ID does not match its manifest")
    if manifest["artifact_codec"] != _artifact_codec_manifest():
        raise ArtifactVerificationError("component artifact codec contract is not recognized")


def _validate_payloads(
    directory: Path, manifest: dict[str, Any]
) -> tuple[FrozenSourceBundle, TensorIndex, IRBundle, DecompileReport]:
    payloads = require_dict(manifest["payloads"], field="artifact payloads")
    if set(payloads) != set(_PAYLOAD_FILENAMES):
        raise ArtifactVerificationError("component artifact has an incomplete JSON payload set")
    decoded: dict[str, dict[str, Any]] = {}
    for name in _PAYLOAD_FILENAMES:
        descriptor = require_dict(payloads[name], field=f"payload descriptor {name}")
        require_exact_keys(descriptor, {"byte_count", "sha256"}, field="payload descriptor")
        byte_count = require_int(descriptor["byte_count"], field="payload bytes", minimum=1)
        expected_sha256 = require_sha256(descriptor["sha256"], field="payload sha256")
        value, raw = _read_canonical_json(directory / name, field=name)
        if len(raw) != byte_count or _sha256_bytes(raw) != expected_sha256:
            raise ArtifactVerificationError(
                "artifact JSON payload differs from its descriptor", details={"path": name}
            )
        decoded[name] = value
    try:
        source = FrozenSourceBundle.from_dict(decoded["source.json"])
        index = TensorIndex.from_dict(decoded["tensor-index.json"])
        bundle = IRBundle.from_dict(decoded["ir.json"])
        report = DecompileReport.from_dict(decoded["decompile-report.json"])
    except (TypeError, ValueError, DecompilerError) as exc:
        raise ArtifactVerificationError(
            "component artifact payload failed structural reopen"
        ) from exc
    if report.status != "decoded" or report.coverage is None or not report.coverage.complete:
        raise ArtifactVerificationError("artifact decompile report is not a complete U2 result")
    return source, index, bundle, report


def _validate_cross_lineage(
    manifest: dict[str, Any],
    source: FrozenSourceBundle,
    index: TensorIndex,
    bundle: IRBundle,
    report: DecompileReport,
) -> None:
    lineage = require_dict(manifest["source_lineage"], field="source lineage")
    require_exact_keys(
        lineage,
        {
            "source_id",
            "resolved_revision",
            "revision_immutable",
            "source_fingerprint",
            "tensor_index_fingerprint",
            "source_asset_count",
            "source_asset_bytes",
        },
        field="source lineage",
    )
    expected_source = {
        "source_id": source.source_id,
        "resolved_revision": source.resolved_revision,
        "revision_immutable": source.revision_immutable,
        "source_fingerprint": source.fingerprint,
        "tensor_index_fingerprint": index.fingerprint,
        "source_asset_count": len(source.files),
        "source_asset_bytes": sum(item.byte_count for item in source.files),
    }
    if lineage != expected_source or index.source_fingerprint != source.fingerprint:
        raise ArtifactVerificationError("artifact source lineage is not cross-bound")

    compiler = require_dict(manifest["compiler_lineage"], field="compiler lineage")
    expected_compiler = {
        "adapter_id": bundle.model.adapter_id,
        "adapter_version": bundle.model.adapter_version,
        "adapter_fingerprint": bundle.model.adapter_fingerprint,
        "decompile_report_fingerprint": report.fingerprint,
        "physical_weights_fingerprint": bundle.physical_weights.fingerprint,
        "model_ir_fingerprint": bundle.model.fingerprint,
        "state_ir_fingerprint": bundle.state.fingerprint,
        "io_ir_fingerprint": bundle.io.fingerprint,
        "ir_bundle_fingerprint": bundle.fingerprint,
        "emitter_id": EMITTER_ID,
        "emitter_version": EMITTER_VERSION,
    }
    if compiler != expected_compiler:
        raise ArtifactVerificationError("artifact compiler lineage is not cross-bound")
    if (
        report.source_fingerprint != source.fingerprint
        or report.tensor_index_fingerprint != index.fingerprint
        or report.selected_adapter_id != bundle.model.adapter_id
        or report.selected_adapter_version != bundle.model.adapter_version
        or report.selected_adapter_fingerprint != bundle.model.adapter_fingerprint
    ):
        raise ArtifactVerificationError("decompile report lineage differs from artifact IR")
    if manifest["decompile_pending_gates"] != list(report.pending_gates):
        raise ArtifactVerificationError("artifact changed the decompiler's pending gates")


def _validate_assets(
    directory: Path,
    manifest: dict[str, Any],
    source: FrozenSourceBundle,
) -> tuple[int, int]:
    records = require_list(manifest["assets"], field="artifact assets")
    source_assets = [item for item in source.files if item.role != "weight-shard"]
    if len(records) != len(source_assets):
        raise ArtifactVerificationError("artifact auxiliary asset inventory is incomplete")
    emitted_bytes = 0
    observed_paths: list[str] = []
    for record, source_file in zip(records, source_assets, strict=True):
        value = require_dict(record, field="artifact asset")
        require_exact_keys(
            value,
            {"source_path", "role", "source_sha256", "byte_count", "artifact_path"},
            field="artifact asset",
        )
        expected_path = f"assets/{source_file.path}"
        actual_path = _artifact_relative_path(value["artifact_path"], field="asset path")
        expected = {
            "source_path": source_file.path,
            "role": source_file.role,
            "source_sha256": source_file.sha256,
            "byte_count": source_file.byte_count,
            "artifact_path": expected_path,
        }
        if value != expected or actual_path != expected_path:
            raise ArtifactVerificationError(
                "artifact auxiliary asset differs from frozen source lineage",
                details={"source_path": source_file.path},
            )
        actual_sha256, actual_bytes = _sha256_file(
            directory / actual_path, expected_bytes=source_file.byte_count
        )
        if actual_sha256 != source_file.sha256 or actual_bytes != source_file.byte_count:
            raise ArtifactVerificationError(
                "artifact auxiliary asset content hash mismatch",
                details={"source_path": source_file.path},
            )
        observed_paths.append(source_file.path)
        emitted_bytes += actual_bytes
    if observed_paths != sorted(set(observed_paths)):
        raise ArtifactVerificationError("artifact auxiliary assets are not sorted and unique")
    return len(source_assets), emitted_bytes


def _validate_allocations(
    directory: Path,
    manifest: dict[str, Any],
    source: FrozenSourceBundle,
    index: TensorIndex,
    bundle: IRBundle,
    *,
    auxiliary_asset_count: int,
    auxiliary_asset_bytes: int,
) -> None:
    records = require_list(manifest["allocations"], field="artifact allocations")
    weights = bundle.physical_weights
    allocations = list(weights.allocations)
    if len(records) != len(allocations) or len(records) != len(index.tensors):
        raise ArtifactVerificationError("artifact allocation count is incomplete")
    emitted_bytes = 0
    observed_ids: list[str] = []
    for record, allocation in zip(records, allocations, strict=True):
        value = require_dict(record, field="artifact allocation")
        require_exact_keys(
            value,
            {
                "allocation_id",
                "source_tensor",
                "source_file",
                "source_file_sha256",
                "source_byte_offset",
                "source_byte_length",
                "source_range_fingerprint",
                "stored_shape",
                "stored_dtype",
                "source_codec",
                "blob",
            },
            field="artifact allocation",
        )
        expected_base = {
            "allocation_id": allocation.allocation_id,
            "source_tensor": allocation.source_tensor,
            "source_file": allocation.source_file,
            "source_file_sha256": source.file(allocation.source_file).sha256,
            "source_byte_offset": allocation.byte_offset,
            "source_byte_length": allocation.byte_length,
            "source_range_fingerprint": allocation.content_fingerprint,
            "stored_shape": list(allocation.stored_shape),
            "stored_dtype": allocation.stored_dtype,
            "source_codec": allocation.codec.as_dict(),
        }
        if {key: value[key] for key in expected_base} != expected_base:
            raise ArtifactVerificationError(
                "artifact allocation lineage differs from PhysicalWeightIR",
                details={"allocation_id": allocation.allocation_id},
            )
        _validate_source_codec(allocation)
        blob = require_dict(value["blob"], field="allocation blob")
        require_exact_keys(blob, {"path", "byte_count", "sha256"}, field="allocation blob")
        expected_path = f"blobs/{allocation.allocation_id}.bin"
        if require_str(blob["path"], field="allocation blob path") != expected_path:
            raise ArtifactVerificationError("allocation blob path is not canonical")
        byte_count = require_int(blob["byte_count"], field="allocation blob bytes", minimum=0)
        if byte_count != allocation.byte_length:
            raise ArtifactVerificationError("allocation blob length differs from source range")
        expected_sha256 = require_sha256(blob["sha256"], field="allocation blob sha256")
        actual_sha256, actual_bytes = _sha256_file(
            directory / expected_path, expected_bytes=byte_count
        )
        if actual_sha256 != expected_sha256 or actual_bytes != byte_count:
            raise ArtifactVerificationError(
                "allocation blob content hash mismatch",
                details={"allocation_id": allocation.allocation_id},
            )
        observed_ids.append(allocation.allocation_id)
        emitted_bytes += byte_count
    if observed_ids != sorted(set(observed_ids)):
        raise ArtifactVerificationError("artifact allocations are not sorted and unique")
    if manifest["views"] != [item.as_dict() for item in weights.views]:
        raise ArtifactVerificationError("artifact logical views differ from PhysicalWeightIR")
    if manifest["alias_classes"] != [item.as_dict() for item in weights.alias_classes]:
        raise ArtifactVerificationError("artifact alias classes differ from PhysicalWeightIR")
    if manifest["classifications"] != [item.as_dict() for item in weights.classifications]:
        raise ArtifactVerificationError("artifact classifications differ from PhysicalWeightIR")

    coverage = require_dict(manifest["coverage"], field="artifact coverage")
    expected_coverage = {
        "source_tensor_count": len(index.tensors),
        "emitted_allocation_count": len(allocations),
        "source_tensor_bytes": index.total_tensor_bytes,
        "emitted_blob_bytes": emitted_bytes,
        "logical_view_count": len(weights.views),
        "alias_class_count": len(weights.alias_classes),
        "classification_count": len(weights.classifications),
        "source_auxiliary_asset_count": auxiliary_asset_count,
        "emitted_auxiliary_asset_count": auxiliary_asset_count,
        "source_auxiliary_asset_bytes": auxiliary_asset_bytes,
        "emitted_auxiliary_asset_bytes": auxiliary_asset_bytes,
        "complete": True,
    }
    if coverage != expected_coverage or emitted_bytes != index.total_tensor_bytes:
        raise ArtifactVerificationError("artifact coverage is not byte-complete")


def _open_component_artifact(
    directory: str | Path, *, enforce_directory_identity: bool
) -> ComponentArtifact:
    root = Path(directory)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise ArtifactVerificationError("component artifact directory does not exist") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ArtifactVerificationError("component artifact root must be a non-symlink directory")
    root = root.resolve()
    manifest, manifest_raw = _read_canonical_json(
        root / "manifest.json", field="component manifest"
    )
    _validate_manifest_shape(manifest)
    if enforce_directory_identity and root.name != manifest["artifact_id"]:
        raise ArtifactVerificationError(
            "component artifact directory name differs from its content identity",
            details={"directory_name": root.name, "artifact_id": manifest["artifact_id"]},
        )
    _validate_exact_inventory(root, manifest)
    source, index, bundle, report = _validate_payloads(root, manifest)
    _validate_cross_lineage(manifest, source, index, bundle, report)
    auxiliary_count, auxiliary_bytes = _validate_assets(root, manifest, source)
    _validate_allocations(
        root,
        manifest,
        source,
        index,
        bundle,
        auxiliary_asset_count=auxiliary_count,
        auxiliary_asset_bytes=auxiliary_bytes,
    )
    return ComponentArtifact(
        directory=root,
        _manifest_json=canonical_json_bytes(manifest).decode("utf-8"),
        manifest_sha256=_sha256_bytes(manifest_raw),
        source=source,
        tensor_index=index,
        ir_bundle=bundle,
        decompile_report=report,
    )


def open_component_artifact(directory: str | Path) -> ComponentArtifact:
    """Strictly reopen and verify a published content-addressed artifact."""

    return _open_component_artifact(directory, enforce_directory_identity=True)


def certify_component_artifact(
    directory: str | Path, *, require_execution: bool = False
) -> CertificationRecord:
    """Run integrity certification; refuse to mislabel it as execution certification."""

    if require_execution:
        raise ExecutionCertificationUnavailable(
            "canonical source-range components have no registered execution certification runner",
            details={
                "required_next_step": (
                    "lower this artifact into a registered native backend, execute the G8-G12 "
                    "gate suite, and bind that runner's evidence separately"
                )
            },
        )
    return CertificationRecord.build(open_component_artifact(directory))


def build_component_artifact(
    source_root: str | Path,
    output_root: str | Path,
    *,
    source_id: str | None = None,
    resolved_revision: str | None = None,
    policy: SourcePolicy | None = None,
) -> ArtifactBuildRecord:
    """Decompile a local Qwen2/Qwen3/Llama source and publish its exact component artifact."""

    result = decompile_source(
        Path(source_root),
        source_id=source_id,
        resolved_revision=resolved_revision,
        policy=policy,
    )
    return NativeSourceComponentEmitter().build(result, output_root)


def inspect_component_eligibility(result: DecompileResult) -> dict[str, Any]:
    """Return the exact emitter support boundary without writing an artifact."""

    eligible = result.succeeded
    reasons: list[dict[str, Any]] = []
    if eligible:
        assert result.ir_bundle is not None
        for allocation in result.ir_bundle.physical_weights.allocations:
            try:
                _validate_source_codec(allocation)
            except ArtifactEmissionError as exc:
                eligible = False
                reasons.append(exc.as_dict())
    else:
        reasons.extend(item.as_dict() for item in result.report.failures)
    return {
        "eligible": eligible,
        "artifact_kind": COMPONENT_ARTIFACT_KIND,
        "artifact_schema": COMPONENT_ARTIFACT_SCHEMA,
        "artifact_codec": _artifact_codec_manifest(),
        "execution_certification_available": False,
        "rejections": reasons,
    }
