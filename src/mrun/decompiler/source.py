"""Immutable, path-independent source custody for model decompilation.

``freeze_source`` is intentionally architecture-blind.  It inventories every asset, rejects
unsafe filesystem and executable formats, hashes files with bounded memory, and retains local
stat guards only for mutation detection.  Local paths and inode metadata never enter the portable
fingerprint.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from ._json import (
    canonical_json,
    canonical_sha256,
    normalize_json,
    require_bool,
    require_dict,
    require_exact_keys,
    require_int,
    require_name,
    require_sha256,
    require_str,
    strict_json_file,
    strict_json_loads,
)
from .errors import SourceCustodyError, SourceMutationError, SourcePolicyError

FROZEN_SOURCE_SCHEMA = "mrun-frozen-source-bundle-v1"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_DEFAULT_JSON_LIMIT = 16 * 1024 * 1024
_PARSED_CONTROL_JSON = frozenset(
    {
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
    }
)
_PICKLE_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"})
_CODE_SUFFIXES = frozenset({".py", ".pyc", ".pyo", ".so", ".dylib", ".dll", ".wasm", ".jar"})


class RemoteCodePolicy(str, Enum):
    DENY = "deny"
    INVENTORY = "inventory-only"


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    """Acquisition policy; no setting in this core ever executes model repository code."""

    remote_code: RemoteCodePolicy = RemoteCodePolicy.DENY
    allow_pickle: bool = False
    require_immutable_revision: bool = False
    json_parse_limit_bytes: int = _DEFAULT_JSON_LIMIT

    def __post_init__(self) -> None:
        remote_code = self.remote_code
        if not isinstance(remote_code, RemoteCodePolicy):
            try:
                remote_code = RemoteCodePolicy(str(remote_code))
            except ValueError as exc:
                raise ValueError(f"unsupported remote-code policy: {self.remote_code!r}") from exc
            object.__setattr__(self, "remote_code", remote_code)
        if type(self.allow_pickle) is not bool:
            raise TypeError("allow_pickle must be a boolean")
        if type(self.require_immutable_revision) is not bool:
            raise TypeError("require_immutable_revision must be a boolean")
        if type(self.json_parse_limit_bytes) is not int or self.json_parse_limit_bytes <= 0:
            raise ValueError("json_parse_limit_bytes must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "remote_code": self.remote_code.value,
            "allow_pickle": self.allow_pickle,
            "require_immutable_revision": self.require_immutable_revision,
            "json_parse_limit_bytes": self.json_parse_limit_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> SourcePolicy:
        value = require_dict(payload, field="source policy")
        require_exact_keys(
            value,
            {
                "remote_code",
                "allow_pickle",
                "require_immutable_revision",
                "json_parse_limit_bytes",
            },
            field="source policy",
        )
        return cls(
            remote_code=RemoteCodePolicy(require_str(value["remote_code"], field="remote_code")),
            allow_pickle=require_bool(value["allow_pickle"], field="allow_pickle"),
            require_immutable_revision=require_bool(
                value["require_immutable_revision"],
                field="require_immutable_revision",
            ),
            json_parse_limit_bytes=require_int(
                value["json_parse_limit_bytes"],
                field="json_parse_limit_bytes",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: str
    byte_count: int
    sha256: str
    role: str

    def __post_init__(self) -> None:
        canonical = _canonical_relative_path(self.path)
        object.__setattr__(self, "path", canonical)
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError(f"source asset {canonical!r} must be non-empty")
        object.__setattr__(self, "sha256", require_sha256(self.sha256, field="file sha256"))
        object.__setattr__(self, "role", require_name(self.role, field="file role"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> SourceFile:
        value = require_dict(payload, field="source file")
        require_exact_keys(
            value,
            {"path", "byte_count", "sha256", "role"},
            field="source file",
        )
        return cls(
            path=require_str(value["path"], field="source file path"),
            byte_count=require_int(value["byte_count"], field="source file bytes", minimum=1),
            sha256=require_str(value["sha256"], field="source file sha256"),
            role=require_str(value["role"], field="source file role"),
        )


@dataclass(frozen=True, slots=True)
class _LocalFileGuard:
    path: str
    byte_count: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int
    symlink_target: str | None = None
    link_device: int | None = None
    link_inode: int | None = None
    link_size: int | None = None
    link_mtime_ns: int | None = None
    link_ctime_ns: int | None = None


@dataclass(frozen=True, slots=True)
class FrozenSourceBundle:
    """Portable source lock plus an optional local mutation guard."""

    source_id: str
    resolved_revision: str | None
    revision_immutable: bool
    policy: SourcePolicy
    files: tuple[SourceFile, ...]
    _config_json: str
    _documents_json: tuple[tuple[str, str], ...]
    fingerprint: str
    schema_version: str = FROZEN_SOURCE_SCHEMA
    _root: str | None = field(default=None, repr=False, compare=False)
    _guards: tuple[_LocalFileGuard, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != FROZEN_SOURCE_SCHEMA:
            raise ValueError(f"unsupported frozen-source schema: {self.schema_version!r}")
        object.__setattr__(self, "source_id", require_name(self.source_id, field="source_id"))
        if self.resolved_revision is not None:
            object.__setattr__(
                self,
                "resolved_revision",
                require_name(self.resolved_revision, field="resolved_revision"),
            )
        if type(self.revision_immutable) is not bool:
            raise TypeError("revision_immutable must be a boolean")
        if not isinstance(self.policy, SourcePolicy):
            raise TypeError("policy must be a SourcePolicy")
        files = tuple(self.files)
        if not files or any(not isinstance(record, SourceFile) for record in files):
            raise ValueError("frozen source must contain SourceFile records")
        if tuple(sorted(files, key=lambda item: item.path)) != files:
            raise ValueError("frozen source files must be sorted by path")
        paths = [record.path for record in files]
        if len(paths) != len(set(paths)):
            raise ValueError("frozen source contains duplicate paths")
        if len({path.casefold() for path in paths}) != len(paths):
            raise ValueError("frozen source contains case-insensitive path collisions")
        object.__setattr__(self, "files", files)
        by_path = {record.path: record for record in files}
        if "config.json" not in by_path or by_path["config.json"].role != "model-config":
            raise SourceCustodyError("frozen source has no root config.json")
        if not any(record.role == "weight-shard" for record in files):
            raise SourceCustodyError("frozen source has no safetensors weights")
        indexes = [record.path for record in files if record.role == "safetensors-index"]
        if len(indexes) > 1:
            raise SourceCustodyError(
                "frozen source contains multiple safetensors indexes",
                details={"indexes": indexes},
            )
        pickle_files = [record.path for record in files if record.role == "pickle-checkpoint"]
        if pickle_files and not self.policy.allow_pickle:
            raise SourcePolicyError(
                "pickle-bearing source violates its frozen policy",
                details={"files": pickle_files},
            )
        code_files = [record.path for record in files if record.role == "custom-code"]
        if code_files and self.policy.remote_code is RemoteCodePolicy.DENY:
            raise SourcePolicyError(
                "custom code violates the frozen deny policy",
                details={"files": code_files},
            )

        config = strict_json_loads(self._config_json, field="frozen config")
        if not isinstance(config, dict):
            raise TypeError("frozen config must be a JSON object")
        canonical_config = canonical_json(config)
        object.__setattr__(self, "_config_json", canonical_config)
        if (
            config.get("auto_map") not in (None, {}, [])
            and self.policy.remote_code is RemoteCodePolicy.DENY
        ):
            raise SourcePolicyError(
                "frozen config declares auto_map under deny policy",
                details={"auto_map": config.get("auto_map")},
            )

        documents: list[tuple[str, str]] = []
        for path, raw in self._documents_json:
            canonical_path = _canonical_relative_path(path)
            value = strict_json_loads(raw, field=f"frozen document {canonical_path}")
            documents.append((canonical_path, canonical_json(value)))
        if documents != sorted(documents):
            raise ValueError("frozen JSON documents must be sorted by path")
        if len({path for path, _ in documents}) != len(documents):
            raise ValueError("frozen JSON documents contain duplicate paths")
        expected_document_paths = {
            record.path
            for record in files
            if PurePosixPath(record.path).name in _PARSED_CONTROL_JSON
        }
        if {path for path, _ in documents} != expected_document_paths:
            raise SourceCustodyError(
                "frozen control-document inventory is incomplete",
                details={
                    "expected": sorted(expected_document_paths),
                    "actual": sorted(path for path, _ in documents),
                },
            )
        object.__setattr__(self, "_documents_json", tuple(documents))

        expected = canonical_sha256(self.identity_payload())
        object.__setattr__(
            self,
            "fingerprint",
            require_sha256(self.fingerprint, field="fingerprint"),
        )
        if self.fingerprint != expected:
            raise ValueError("frozen source fingerprint does not match its portable payload")
        if self.policy.require_immutable_revision and not self.revision_immutable:
            raise SourcePolicyError(
                "promotion policy requires an immutable resolved revision",
                details={"resolved_revision": self.resolved_revision},
            )

    @classmethod
    def freeze(
        cls,
        root: Path,
        *,
        source_id: str | None = None,
        resolved_revision: str | None = None,
        policy: SourcePolicy | None = None,
    ) -> FrozenSourceBundle:
        return freeze_source(
            root,
            source_id=source_id,
            resolved_revision=resolved_revision,
            policy=policy,
        )

    @property
    def root_digest(self) -> str:
        return self.fingerprint

    @property
    def config(self) -> dict[str, Any]:
        value = strict_json_loads(self._config_json, field="frozen config")
        assert isinstance(value, dict)
        return value

    @property
    def root(self) -> Path | None:
        return Path(self._root) if self._root is not None else None

    @property
    def custom_code_files(self) -> tuple[SourceFile, ...]:
        return tuple(record for record in self.files if record.role == "custom-code")

    def document(self, path: str) -> Any | None:
        canonical = _canonical_relative_path(path)
        for candidate, raw in self._documents_json:
            if candidate == canonical:
                return strict_json_loads(raw, field=f"frozen document {canonical}")
        return None

    def file(self, path: str) -> SourceFile:
        canonical = _canonical_relative_path(path)
        for record in self.files:
            if record.path == canonical:
                return record
        raise KeyError(canonical)

    def files_with_role(self, role: str) -> tuple[SourceFile, ...]:
        return tuple(record for record in self.files if record.role == role)

    def file_path(self, path: str) -> Path:
        if self._root is None:
            raise SourceCustodyError("frozen source has no attached local root")
        record = self.file(path)
        candidate = Path(self._root) / record.path
        return _resolve_safe_asset_path(
            candidate,
            root=Path(self._root),
            relative=record.path,
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "resolved_revision": self.resolved_revision,
            "revision_immutable": self.revision_immutable,
            "policy": self.policy.as_dict(),
            "files": [record.as_dict() for record in self.files],
            "config": strict_json_loads(self._config_json, field="frozen config"),
            "documents": {
                path: strict_json_loads(raw, field=f"frozen document {path}")
                for path, raw in self._documents_json
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Any, *, root: Path | None = None) -> FrozenSourceBundle:
        value = require_dict(payload, field="frozen source")
        require_exact_keys(
            value,
            {
                "schema_version",
                "source_id",
                "resolved_revision",
                "revision_immutable",
                "policy",
                "files",
                "config",
                "documents",
                "fingerprint",
            },
            field="frozen source",
        )
        raw_files = value["files"]
        if not isinstance(raw_files, list):
            raise TypeError("frozen source files must be an array")
        raw_documents = require_dict(value["documents"], field="frozen documents")
        revision = value["resolved_revision"]
        if revision is not None:
            revision = require_str(revision, field="resolved_revision")
        bundle = cls(
            schema_version=require_str(value["schema_version"], field="source schema"),
            source_id=require_str(value["source_id"], field="source_id"),
            resolved_revision=revision,
            revision_immutable=require_bool(
                value["revision_immutable"], field="revision_immutable"
            ),
            policy=SourcePolicy.from_dict(value["policy"]),
            files=tuple(SourceFile.from_dict(item) for item in raw_files),
            _config_json=canonical_json(require_dict(value["config"], field="config")),
            _documents_json=tuple(
                (path, canonical_json(raw_documents[path])) for path in sorted(raw_documents)
            ),
            fingerprint=require_str(value["fingerprint"], field="fingerprint"),
        )
        return bundle.attach_root(root) if root is not None else bundle

    def attach_root(self, root: Path) -> FrozenSourceBundle:
        root_path = _validate_root(root)
        scanned, guards = _scan_assets(root_path)
        expected = tuple(self.files)
        if scanned != expected:
            raise SourceMutationError(
                "attached source root does not match the frozen file inventory",
                details={
                    "expected": [record.as_dict() for record in expected],
                    "actual": [record.as_dict() for record in scanned],
                },
            )
        _verify_control_documents(self, root_path)
        return replace(self, _root=str(root_path), _guards=guards)

    def assert_unchanged(self) -> None:
        if self._root is None:
            raise SourceCustodyError("cannot verify a detached frozen source")
        root = _validate_root(Path(self._root))
        scanned, guards = _scan_assets(root)
        if scanned != self.files:
            raise SourceMutationError(
                "source assets changed after freeze",
                details={
                    "expected_fingerprint": self.fingerprint,
                    "changed_paths": _changed_paths(self.files, scanned),
                },
            )
        if self._guards and guards != self._guards:
            raise SourceMutationError(
                "source filesystem identity changed after freeze",
                details={"changed_paths": _changed_guard_paths(self._guards, guards)},
            )

    def assert_stat_unchanged(self) -> None:
        """Cheap between-stage guard; final custody still rehashes every asset."""

        if self._root is None:
            raise SourceCustodyError("cannot verify a detached frozen source")
        root = _validate_root(Path(self._root))
        paths = _list_asset_paths(root)
        if tuple(relative for relative, _, _ in paths) != tuple(
            record.path for record in self.files
        ):
            raise SourceMutationError("source asset set changed after freeze")
        if not self._guards:
            raise SourceCustodyError("attached frozen source has no local file guards")
        actual: list[_LocalFileGuard] = []
        expected_by_path = {guard.path: guard for guard in self._guards}
        for relative, entry_path, content_path in paths:
            info = content_path.lstat()
            entry = entry_path.lstat()
            expected = expected_by_path[relative]
            target = os.readlink(entry_path) if stat.S_ISLNK(entry.st_mode) else None
            actual.append(
                _LocalFileGuard(
                    path=relative,
                    byte_count=info.st_size,
                    sha256=expected.sha256,
                    device=info.st_dev,
                    inode=info.st_ino,
                    mtime_ns=info.st_mtime_ns,
                    ctime_ns=info.st_ctime_ns,
                    symlink_target=target,
                    link_device=entry.st_dev if target is not None else None,
                    link_inode=entry.st_ino if target is not None else None,
                    link_size=entry.st_size if target is not None else None,
                    link_mtime_ns=entry.st_mtime_ns if target is not None else None,
                    link_ctime_ns=entry.st_ctime_ns if target is not None else None,
                )
            )
        actual_tuple = tuple(actual)
        if actual_tuple != self._guards:
            raise SourceMutationError(
                "source filesystem identity changed after freeze",
                details={"changed_paths": _changed_guard_paths(self._guards, actual_tuple)},
            )


def _canonical_relative_path(value: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"invalid portable source path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid portable source path: {value!r}")
    return path.as_posix()


def _is_immutable_revision(value: str | None) -> bool:
    if value is None:
        return False
    candidate = value.lower()
    if candidate.startswith("sha256:"):
        candidate = candidate[7:]
    return len(candidate) in {40, 64} and all(ch in "0123456789abcdef" for ch in candidate)


def _discover_revision(root: Path) -> str | None:
    parts = root.resolve().parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            candidate = parts[index + 1]
            if _is_immutable_revision(candidate):
                return candidate.lower()
    metadata = root / ".cache" / "huggingface" / "download"
    commits: set[str] = set()
    if metadata.is_dir():
        for path in sorted(metadata.glob("*.metadata")):
            try:
                first = path.read_text(encoding="utf-8").splitlines()[0].strip().lower()
            except (OSError, UnicodeDecodeError, IndexError):
                continue
            if _is_immutable_revision(first):
                commits.add(first)
    if len(commits) > 1:
        raise SourceCustodyError(
            "source assets expose multiple immutable revisions",
            details={"revisions": sorted(commits)},
        )
    return next(iter(commits), None)


def _asset_role(relative: str) -> str:
    name = PurePosixPath(relative).name.lower()
    suffix = PurePosixPath(relative).suffix.lower()
    if name == "config.json":
        return "model-config"
    if name.endswith(".safetensors.index.json"):
        return "safetensors-index"
    if name.endswith(".safetensors"):
        return "weight-shard"
    if name == "generation_config.json":
        return "generation-config"
    if name == "tokenizer_config.json":
        return "tokenizer-config"
    if name == "special_tokens_map.json":
        return "special-tokens"
    if name == "added_tokens.json":
        return "added-tokens"
    if name.startswith("chat_template"):
        return "chat-template"
    if name in {"preprocessor_config.json", "processor_config.json"}:
        return "processor-config"
    if name in {"tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"}:
        return "tokenizer"
    if name.startswith(("vocab.", "sentencepiece.")):
        return "tokenizer"
    if name.startswith("readme") or name.startswith("license") or name == "notice":
        return "model-metadata"
    if suffix in _PICKLE_SUFFIXES:
        return "pickle-checkpoint"
    if suffix in _CODE_SUFFIXES:
        return "custom-code"
    return "metadata"


def _validate_root(root: Path) -> Path:
    candidate = Path(root).expanduser().absolute()
    try:
        mode = candidate.lstat().st_mode
    except FileNotFoundError as exc:
        raise SourceCustodyError(f"source root does not exist: {candidate}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise SourceCustodyError(f"source root must be a real directory: {candidate}")
    return candidate


def _hf_snapshot_blob_root(root: Path) -> Path | None:
    """Return the real blob directory for one immutable Hugging Face snapshot.

    Hugging Face snapshots are intentionally symlink farms: logical model filenames point at
    immutable files under the repository's sibling ``blobs`` directory.  Rejecting every symlink
    makes the default local representation unusable and encourages callers to create an
    unaudited copy.  We accept only this narrow, content-addressed layout; arbitrary links remain
    forbidden.
    """

    absolute = root.absolute()
    parts = absolute.parts
    positions = [index for index, part in enumerate(parts) if part == "snapshots"]
    if len(positions) != 1:
        return None
    index = positions[0]
    if index + 2 != len(parts) or not _is_immutable_revision(parts[index + 1]):
        return None
    repository = Path(*parts[:index])
    blob_root = repository / "blobs"
    try:
        info = blob_root.lstat()
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return None
    return blob_root.resolve()


def _resolve_safe_asset_path(path: Path, *, root: Path, relative: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise SourceMutationError(
            f"source asset disappeared: {relative}", details={"path": relative}
        ) from exc
    if stat.S_ISREG(info.st_mode):
        return path
    if not stat.S_ISLNK(info.st_mode):
        raise SourceCustodyError(
            f"source asset is not a regular non-symlink file: {relative}",
            details={"path": relative},
        )

    blob_root = _hf_snapshot_blob_root(root)
    raw_target = os.readlink(path)
    if blob_root is None or Path(raw_target).is_absolute():
        raise SourceCustodyError(
            f"source asset is not a regular non-symlink file: {relative}",
            details={"path": relative},
        )
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(blob_root)
    except (OSError, ValueError) as exc:
        raise SourceCustodyError(
            f"Hugging Face snapshot asset escapes its immutable blob store: {relative}",
            details={"path": relative},
        ) from exc
    blob_name = resolved.name.lower()
    if not _is_immutable_revision(blob_name):
        raise SourceCustodyError(
            f"Hugging Face snapshot target is not content addressed: {relative}",
            details={"path": relative, "target": raw_target},
        )
    target_info = resolved.lstat()
    if stat.S_ISLNK(target_info.st_mode) or not stat.S_ISREG(target_info.st_mode):
        raise SourceCustodyError(
            f"Hugging Face snapshot target is not a real regular file: {relative}",
            details={"path": relative},
        )
    return resolved


def _list_asset_paths(root: Path) -> tuple[tuple[str, Path, Path], ...]:
    collected: list[tuple[str, Path, Path]] = []
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path.relative_to(root).parts == (".cache", "huggingface"):
            # ``hf download --local-dir`` keeps transport locks and revision metadata here.
            # Transformers does not consume this subtree as model input; the decompiler reads
            # ``*.metadata`` separately only to discover a revision when one was not supplied.
            # Excluding the whole transport directory prevents zero-byte live lock files from
            # becoming model assets while keeping arbitrary sibling cache content fail-closed.
            if "download" in directories:
                metadata = current_path / "download"
                info = metadata.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    relative = metadata.relative_to(root).as_posix()
                    raise SourceCustodyError(
                        f"source contains an unsafe directory entry: {relative}",
                        details={"path": relative},
                    )
                directories.remove("download")
        for directory in sorted(directories):
            candidate = current_path / directory
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                relative = candidate.relative_to(root).as_posix()
                raise SourceCustodyError(
                    f"source contains an unsafe directory entry: {relative}",
                    details={"path": relative},
                )
        for filename in sorted(filenames):
            candidate = current_path / filename
            relative = _canonical_relative_path(candidate.relative_to(root).as_posix())
            content_path = _resolve_safe_asset_path(candidate, root=root, relative=relative)
            collected.append((relative, candidate, content_path))
    collected.sort(key=lambda item: item[0])
    paths = [relative for relative, _, _ in collected]
    if len(paths) != len(set(paths)) or len(paths) != len({item.casefold() for item in paths}):
        raise SourceCustodyError("source contains duplicate or case-colliding asset paths")
    if not collected:
        raise SourceCustodyError(f"source directory is empty: {root}")
    return tuple(collected)


def _hash_guarded(
    path: Path,
    *,
    relative: str,
    entry_path: Path,
) -> tuple[str, _LocalFileGuard]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceCustodyError(
            f"cannot safely open source asset: {relative}", details={"path": relative}
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise SourceCustodyError(
                f"source asset must be a non-empty regular file: {relative}",
                details={"path": relative, "bytes": before.st_size},
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise SourceMutationError(
            f"source asset changed while hashing: {relative}", details={"path": relative}
        )
    sha256 = digest.hexdigest()
    entry = entry_path.lstat()
    target = os.readlink(entry_path) if stat.S_ISLNK(entry.st_mode) else None
    return sha256, _LocalFileGuard(
        path=relative,
        byte_count=after.st_size,
        sha256=sha256,
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
        symlink_target=target,
        link_device=entry.st_dev if target is not None else None,
        link_inode=entry.st_ino if target is not None else None,
        link_size=entry.st_size if target is not None else None,
        link_mtime_ns=entry.st_mtime_ns if target is not None else None,
        link_ctime_ns=entry.st_ctime_ns if target is not None else None,
    )


def _scan_assets(root: Path) -> tuple[tuple[SourceFile, ...], tuple[_LocalFileGuard, ...]]:
    paths = _list_asset_paths(root)
    records: list[SourceFile] = []
    guards: list[_LocalFileGuard] = []
    for relative, entry_path, content_path in paths:
        digest, guard = _hash_guarded(
            content_path,
            relative=relative,
            entry_path=entry_path,
        )
        records.append(
            SourceFile(
                path=relative,
                byte_count=guard.byte_count,
                sha256=digest,
                role=_asset_role(relative),
            )
        )
        guards.append(guard)
    if tuple(relative for relative, _, _ in _list_asset_paths(root)) != tuple(
        record.path for record in records
    ):
        raise SourceMutationError("source asset set changed while it was frozen")
    return tuple(records), tuple(guards)


def _changed_paths(expected: tuple[SourceFile, ...], actual: tuple[SourceFile, ...]) -> list[str]:
    expected_by_path = {item.path: item for item in expected}
    actual_by_path = {item.path: item for item in actual}
    return sorted(
        path
        for path in set(expected_by_path) | set(actual_by_path)
        if expected_by_path.get(path) != actual_by_path.get(path)
    )


def _changed_guard_paths(
    expected: tuple[_LocalFileGuard, ...], actual: tuple[_LocalFileGuard, ...]
) -> list[str]:
    expected_by_path = {item.path: item for item in expected}
    actual_by_path = {item.path: item for item in actual}
    return sorted(
        path
        for path in set(expected_by_path) | set(actual_by_path)
        if expected_by_path.get(path) != actual_by_path.get(path)
    )


def _verify_control_documents(bundle: FrozenSourceBundle, root: Path) -> None:
    try:
        config = strict_json_file(
            _resolve_safe_asset_path(
                root / "config.json",
                root=root,
                relative="config.json",
            ),
            max_bytes=bundle.policy.json_parse_limit_bytes,
            field="config.json",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SourceMutationError(f"cannot verify frozen config.json: {exc}") from exc
    if canonical_json(config) != bundle._config_json:
        raise SourceMutationError("frozen config content does not match config.json")
    for relative, frozen_json in bundle._documents_json:
        try:
            document = strict_json_file(
                _resolve_safe_asset_path(
                    root / relative,
                    root=root,
                    relative=relative,
                ),
                max_bytes=bundle.policy.json_parse_limit_bytes,
                field=relative,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SourceMutationError(f"cannot verify frozen {relative}: {exc}") from exc
        if canonical_json(document) != frozen_json:
            raise SourceMutationError(f"frozen control content does not match {relative}")


def freeze_source(
    root: Path,
    *,
    source_id: str | None = None,
    resolved_revision: str | None = None,
    policy: SourcePolicy | None = None,
) -> FrozenSourceBundle:
    """Freeze a local declarative model bundle without importing model code."""

    selected_policy = policy or SourcePolicy()
    root_path = _validate_root(root)
    asset_paths = _list_asset_paths(root_path)
    files, guards = _scan_assets(root_path)
    content_by_relative = {relative: content for relative, _entry, content in asset_paths}
    by_path = {record.path: record for record in files}

    config_record = by_path.get("config.json")
    if config_record is None or config_record.role != "model-config":
        raise SourceCustodyError("source bundle has no root config.json")
    shard_records = tuple(record for record in files if record.role == "weight-shard")
    if not shard_records:
        raise SourceCustodyError("source bundle has no safetensors weights")
    indexes = tuple(record for record in files if record.role == "safetensors-index")
    if len(indexes) > 1:
        raise SourceCustodyError(
            "source bundle contains multiple safetensors indexes",
            details={"indexes": [record.path for record in indexes]},
        )

    pickle_files = [record.path for record in files if record.role == "pickle-checkpoint"]
    if pickle_files and not selected_policy.allow_pickle:
        raise SourcePolicyError(
            "pickle-bearing checkpoints are disabled for unattended decompilation",
            details={"files": pickle_files},
        )
    code_files = [record.path for record in files if record.role == "custom-code"]
    if code_files and selected_policy.remote_code is RemoteCodePolicy.DENY:
        raise SourcePolicyError(
            "custom code is present under deny policy",
            details={"files": code_files},
        )

    try:
        config = strict_json_file(
            content_by_relative["config.json"],
            max_bytes=selected_policy.json_parse_limit_bytes,
            field="config.json",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SourceCustodyError(f"cannot freeze config.json: {exc}") from exc
    if not isinstance(config, dict):
        raise SourceCustodyError("config.json must contain an object")
    config = normalize_json(config, field="config.json")
    assert isinstance(config, dict)
    auto_map = config.get("auto_map")
    if auto_map not in (None, {}, []) and selected_policy.remote_code is RemoteCodePolicy.DENY:
        raise SourcePolicyError(
            "config.json declares auto_map under deny policy",
            details={"auto_map": auto_map},
        )

    documents: list[tuple[str, str]] = []
    for record in files:
        if PurePosixPath(record.path).name not in _PARSED_CONTROL_JSON:
            continue
        try:
            document = strict_json_file(
                content_by_relative[record.path],
                max_bytes=selected_policy.json_parse_limit_bytes,
                field=record.path,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SourceCustodyError(f"cannot freeze {record.path}: {exc}") from exc
        documents.append((record.path, canonical_json(document)))

    revision = resolved_revision if resolved_revision is not None else _discover_revision(root_path)
    if revision is not None:
        revision = require_name(revision, field="resolved_revision")
    revision_immutable = _is_immutable_revision(revision)
    if selected_policy.require_immutable_revision and not revision_immutable:
        raise SourcePolicyError(
            "promotion policy requires an immutable resolved revision",
            details={"resolved_revision": revision},
        )

    portable = {
        "schema_version": FROZEN_SOURCE_SCHEMA,
        "source_id": source_id if source_id is not None else "local-content",
        "resolved_revision": revision,
        "revision_immutable": revision_immutable,
        "policy": selected_policy.as_dict(),
        "files": [record.as_dict() for record in files],
        "config": config,
        "documents": {
            path: strict_json_loads(raw, field=f"frozen document {path}") for path, raw in documents
        },
    }
    bundle = FrozenSourceBundle(
        source_id=require_name(portable["source_id"], field="source_id"),
        resolved_revision=revision,
        revision_immutable=revision_immutable,
        policy=selected_policy,
        files=files,
        _config_json=canonical_json(config),
        _documents_json=tuple(sorted(documents)),
        fingerprint=canonical_sha256(portable),
        _root=str(root_path),
        _guards=guards,
    )
    _verify_control_documents(bundle, root_path)
    bundle.assert_stat_unchanged()
    return bundle
