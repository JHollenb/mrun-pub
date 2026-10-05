"""Alias-aware, output-contract-routed QStore component graphs.

This module is the production reader for the exact disassembly proved by the July 27
component-graph experiment.  It intentionally keeps the experiment's legacy component
stores non-promotable: graph/component hashes are verified, but a legacy source is not
silently upgraded to QStore-v3 content lineage.

The central invariant is that physical allocation, logical parameter name, and permitted
operation are independent.  A tied ``embed``/``lm_head`` allocation is opened once, while
hidden-state execution can still deny every logical ``lm_head`` method before provider
lookup or telemetry changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import torch

from .qstore import QStore

COMPONENT_GRAPH_SCHEMA = "mrun-component-graph-v1"
POC_COMPONENT_GRAPH_SCHEMA = "mrun.disassembled-model-graph.poc.v2"
VOCAB_MANIFEST_SCHEMA = "mrun-vocab-manifest-v1"
COMPOSITE_CUSTODY_SCHEMA = "mrun-composite-custody-v1"
COMPONENT_PAYLOAD_FILES = ("weights.i8", "scales.f32", "extras.f32")
_COMPONENT_PAYLOAD_LAYOUT = {
    "qrow": (
        ("weights.i8", "w_off", "w_len"),
        ("scales.f32", "s_off", "s_len"),
    ),
    "fp32": (("extras.f32", "e_off", "e_len"),),
}
SUPPORTED_COMPONENT_GRAPH_SCHEMAS = frozenset({COMPONENT_GRAPH_SCHEMA, POC_COMPONENT_GRAPH_SCHEMA})


class ComponentGraphError(RuntimeError):
    """A component graph, component artifact, or route is invalid."""


class ComponentOutputContractError(ComponentGraphError):
    """A logical operation is not authorized by the output contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComponentGraphError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _identity_from_stat(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _open_regular_file(path: Path) -> tuple[int, tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ComponentGraphError(f"cannot open regular component artifact {path}") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ComponentGraphError(f"component artifact must be a regular file: {path}")
        return descriptor, _identity_from_stat(file_stat)
    except Exception:
        os.close(descriptor)
        raise


def _require_path_identity(path: Path, expected: tuple[int, int, int, int, int]) -> None:
    try:
        current = path.lstat()
    except OSError as exc:
        raise ComponentGraphError(
            f"component artifact disappeared during verification: {path}"
        ) from exc
    if not stat.S_ISREG(current.st_mode) or _identity_from_stat(current) != expected:
        raise ComponentGraphError(f"component artifact changed during verification: {path}")


def _read_json_verified(
    path: Path,
) -> tuple[dict[str, Any], tuple[int, int, int, int, int], str]:
    descriptor, initial = _open_regular_file(path)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            payload = handle.read()
            final = _identity_from_stat(os.fstat(handle.fileno()))
    except OSError as exc:
        raise ComponentGraphError(f"cannot read component graph JSON {path}") from exc
    if initial != final:
        raise ComponentGraphError(f"component graph JSON changed while being read: {path}")
    _require_path_identity(path, final)
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ComponentGraphError(f"cannot read component graph JSON {path}") from exc
    if not isinstance(value, dict):
        raise ComponentGraphError(f"component graph JSON must be an object: {path}")
    return value, final, _sha256_bytes(payload)


def _read_json(path: Path) -> dict[str, Any]:
    value, _identity, _digest = _read_json_verified(path)
    return value


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON contract used by the existing component-graph fingerprints."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ComponentGraphError("component graph is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file_verified(path: Path) -> tuple[str, tuple[int, int, int, int, int]]:
    digest = hashlib.sha256()
    descriptor, initial = _open_regular_file(path)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
            final = _identity_from_stat(os.fstat(handle.fileno()))
    except OSError as exc:
        raise ComponentGraphError(f"cannot hash component artifact {path}") from exc
    if initial != final:
        raise ComponentGraphError(f"component artifact changed while being hashed: {path}")
    _require_path_identity(path, final)
    return digest.hexdigest(), final


def _sha256_file(path: Path) -> str:
    digest, _identity = _sha256_file_verified(path)
    return digest


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _require_sha256(value: Any, field: str) -> str:
    if not _is_sha256(value):
        raise ComponentGraphError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise ComponentGraphError(f"cannot stat component artifact {path}") from exc
    return _identity_from_stat(stat)


def _resolve_alias(blocks: Mapping[str, Mapping[str, Any]], name: str) -> str:
    current = str(name)
    seen: set[str] = set()
    while True:
        if current in seen:
            raise ComponentGraphError(f"cyclic component-graph alias at {name!r}")
        seen.add(current)
        try:
            block = blocks[current]
        except KeyError as exc:
            raise ComponentGraphError(f"alias target {current!r} is absent") from exc
        alias = block.get("alias")
        if alias is None:
            return current
        current = str(alias)


def _logical_role(name: str) -> str:
    if name == "embed":
        return "ingress"
    if name == "lm_head":
        return "egress"
    if name in {"norm.final", "norm.final.bias"}:
        return "norm"
    return "body"


def _allocation_topology(
    blocks: Mapping[str, Mapping[str, Any]],
    *,
    declared_tied: bool,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    for required in ("embed", "lm_head", "norm.final"):
        if required not in blocks:
            raise ComponentGraphError(f"required logical block {required!r} is absent")

    by_root: dict[str, list[str]] = defaultdict(list)
    for name in blocks:
        by_root[_resolve_alias(blocks, str(name))].append(str(name))

    root_roles: dict[str, str] = {}
    for root, names in by_root.items():
        logical_roles = {_logical_role(name) for name in names}
        if logical_roles == {"ingress", "egress"} and {"embed", "lm_head"} <= set(names):
            root_roles[root] = "lexical_shared"
        elif len(logical_roles) == 1:
            root_roles[root] = next(iter(logical_roles))
        else:
            raise ComponentGraphError(
                f"unsupported cross-role alias class root={root!r} "
                f"names={sorted(names)} roles={sorted(logical_roles)}"
            )

    observed_tied = _resolve_alias(blocks, "embed") == _resolve_alias(blocks, "lm_head")
    if observed_tied != bool(declared_tied):
        raise ComponentGraphError(
            "tie declaration disagrees with observed alias topology: "
            f"declared={bool(declared_tied)} observed={observed_tied}"
        )

    role_names: dict[str, list[str]] = defaultdict(list)
    for name in blocks:
        role_names[root_roles[_resolve_alias(blocks, str(name))]].append(str(name))
    normalized_roles = {role: sorted(names) for role, names in sorted(role_names.items())}
    expected_roles = (
        {"body", "norm", "lexical_shared"}
        if observed_tied
        else {"body", "norm", "ingress", "egress"}
    )
    if set(normalized_roles) != expected_roles:
        raise ComponentGraphError(
            f"unexpected component roles {sorted(normalized_roles)}; "
            f"expected {sorted(expected_roles)}"
        )

    topology = {
        "declared_tied": bool(declared_tied),
        "observed_tied": observed_tied,
        "embed_physical_root": _resolve_alias(blocks, "embed"),
        "lm_head_physical_root": _resolve_alias(blocks, "lm_head"),
        "aliases": {
            name: str(block["alias"]) for name, block in sorted(blocks.items()) if "alias" in block
        },
        "alias_classes": {
            root: sorted(names) for root, names in sorted(by_root.items()) if len(names) > 1
        },
        "physical_roles": sorted(normalized_roles),
    }
    return normalized_roles, topology


def _expected_operation_contracts(
    role_names: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    tied = "lexical_shared" in role_names
    input_role = "lexical_shared" if tied else "ingress"
    base = ["body", "norm", input_role]
    full = base if tied else [*base, "egress"]
    return {
        "full_logits": {
            "required_components": sorted(full),
            "lm_head_methods": ["row_blocks"],
        },
        "lexical_hidden": {
            "required_components": sorted(base),
            "lm_head_methods": [],
        },
        "selected_rows": {
            "required_components": sorted(full),
            "lm_head_methods": ["embed_rows"],
        },
    }


@dataclass(frozen=True)
class VocabManifest:
    """Exact ordered tokenizer identity plus its configured model-row extent."""

    tokenizer_class: str
    token_count: int
    configured_row_count: int
    ordered_tokens_sha256: str
    backend_json_sha256: str
    chat_template_sha256: str
    special_token_ids: tuple[tuple[str, int | None], ...]
    descriptor_semantic_sha256: str
    semantic_sha256: str
    schema_version: str = VOCAB_MANIFEST_SCHEMA

    @classmethod
    def from_descriptor(
        cls,
        descriptor: Mapping[str, Any],
        *,
        configured_row_count: int,
    ) -> VocabManifest:
        token_count = int(descriptor.get("length", 0))
        if token_count <= 0:
            raise ComponentGraphError("tokenizer length must be positive")
        configured_row_count = int(configured_row_count)
        if configured_row_count < token_count:
            raise ComponentGraphError(
                "configured vocabulary rows cannot be smaller than tokenizer IDs"
            )
        tokenizer_class = str(descriptor.get("class", ""))
        if not tokenizer_class:
            raise ComponentGraphError("tokenizer class is absent")
        special = descriptor.get("special_token_ids")
        if not isinstance(special, Mapping):
            raise ComponentGraphError("tokenizer special_token_ids must be an object")
        normalized_special: list[tuple[str, int | None]] = []
        for name in ("bos", "cls", "eos", "mask", "pad", "sep", "unk"):
            value = special.get(name)
            if value is not None:
                value = int(value)
                if value < 0 or value >= token_count:
                    raise ComponentGraphError(
                        f"special token {name!r}={value} is outside tokenizer ID space"
                    )
            normalized_special.append((name, value))

        descriptor_payload = {
            "class": tokenizer_class,
            "length": token_count,
            "ordered_tokens_sha256": _require_sha256(
                descriptor.get("ordered_tokens_sha256"),
                "tokenizer ordered_tokens_sha256",
            ),
            "backend_json_sha256": _require_sha256(
                descriptor.get("backend_json_sha256"),
                "tokenizer backend_json_sha256",
            ),
            "special_token_ids": dict(normalized_special),
            "chat_template_sha256": _require_sha256(
                descriptor.get("chat_template_sha256"),
                "tokenizer chat_template_sha256",
            ),
        }
        descriptor_sha = _sha256_bytes(canonical_json_bytes(descriptor_payload))
        claimed = _require_sha256(
            descriptor.get("semantic_sha256"),
            "tokenizer semantic_sha256",
        )
        if claimed != descriptor_sha:
            raise ComponentGraphError("tokenizer semantic fingerprint mismatch")
        manifest_payload = {
            "schema_version": VOCAB_MANIFEST_SCHEMA,
            "tokenizer": descriptor_payload,
            "token_space": {
                "valid_ids": {"kind": "dense_range", "start": 0, "stop": token_count},
            },
            "row_space": {
                "configured_vocab_size": configured_row_count,
                "mapping": {
                    "kind": "identity_prefix",
                    "domain_size": token_count,
                    "codomain_size": configured_row_count,
                },
            },
        }
        return cls(
            tokenizer_class=tokenizer_class,
            token_count=token_count,
            configured_row_count=configured_row_count,
            ordered_tokens_sha256=descriptor_payload["ordered_tokens_sha256"],
            backend_json_sha256=descriptor_payload["backend_json_sha256"],
            chat_template_sha256=descriptor_payload["chat_template_sha256"],
            special_token_ids=tuple(normalized_special),
            descriptor_semantic_sha256=descriptor_sha,
            semantic_sha256=_sha256_bytes(canonical_json_bytes(manifest_payload)),
        )

    def validate_token_ids(
        self,
        token_ids: Sequence[int] | np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(token_ids, torch.Tensor):
            # Keep accelerator-resident IDs on their original device.  Converting through
            # NumPy here would both synchronize and make CUDA input impossible.
            ids = token_ids.to(dtype=torch.long)
            outside = bool(
                ids.numel()
                and (torch.any(ids < 0).item() or torch.any(ids >= self.token_count).item())
            )
        else:
            ids = np.asarray(token_ids, dtype=np.int64)
            outside = bool(ids.size and ((ids < 0).any() or (ids >= self.token_count).any()))
        if outside:
            raise ComponentOutputContractError(
                f"token IDs must be in [0, {self.token_count}); padded model rows are not tokens"
            )
        return ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tokenizer_class": self.tokenizer_class,
            "token_count": self.token_count,
            "configured_row_count": self.configured_row_count,
            "ordered_tokens_sha256": self.ordered_tokens_sha256,
            "backend_json_sha256": self.backend_json_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "special_token_ids": dict(self.special_token_ids),
            "descriptor_semantic_sha256": self.descriptor_semantic_sha256,
            "semantic_sha256": self.semantic_sha256,
        }


def tokenizer_descriptor(tokenizer: Any) -> dict[str, Any]:
    """Build the exact descriptor used by the Stage 4 component graphs."""

    ordered = hashlib.sha256()
    for token_id in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        encoded = ("" if token is None else str(token)).encode("utf-8", errors="surrogatepass")
        ordered.update(int(token_id).to_bytes(8, "little", signed=False))
        ordered.update(len(encoded).to_bytes(8, "little", signed=False))
        ordered.update(encoded)
    backend = tokenizer.backend_tokenizer.to_str().encode("utf-8")
    special_names = ("bos", "eos", "pad", "unk", "sep", "cls", "mask")
    special_ids = {name: getattr(tokenizer, f"{name}_token_id", None) for name in special_names}
    semantic_payload = {
        "class": type(tokenizer).__name__,
        "length": len(tokenizer),
        "ordered_tokens_sha256": ordered.hexdigest(),
        "backend_json_sha256": _sha256_bytes(backend),
        "special_token_ids": special_ids,
        "chat_template_sha256": _sha256_bytes(
            str(getattr(tokenizer, "chat_template", None) or "").encode("utf-8")
        ),
    }
    return {
        **semantic_payload,
        "semantic_sha256": _sha256_bytes(canonical_json_bytes(semantic_payload)),
    }


def _graph_fingerprint_payload(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Semantic payload shared with the Stage 4 v2 graph."""

    components = graph.get("components")
    if not isinstance(components, Mapping):
        raise ComponentGraphError("component graph components must be an object")
    return {
        "schema": graph["schema"],
        "model": graph["model"],
        "architecture": graph["architecture"],
        "body_abi_sha256": graph["body_abi"]["semantic_sha256"],
        "tokenizer_sha256": graph["tokenizer"]["semantic_sha256"],
        "topology": graph["topology"],
        "routes": graph["routes"],
        "components": {
            role: {
                "semantic_content_sha256": record["semantic_content_sha256"],
                "manifest_sha256": record["manifest_sha256"],
                "allowed_names": record["allowed_names"],
            }
            for role, record in sorted(components.items())
        },
        "operation_contracts": graph["operation_contracts"],
    }


def _custody_fingerprint_payload(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Relocatable production identity binding every runtime-semantic artifact record.

    Stage-4's legacy digest intentionally omitted blob records and the full logical block
    table.  We still verify that declared digest for compatibility, but never use it as a
    production pool/WorkPlan identity.  This stronger digest excludes only physical
    locators and build-time diagnostics, so relocating identical bytes is identity-stable.
    """

    components = graph.get("components")
    if not isinstance(components, Mapping):
        raise ComponentGraphError("component graph components must be an object")
    return {
        "schema": COMPOSITE_CUSTODY_SCHEMA,
        "source_graph_schema": graph["schema"],
        "declared_graph_fingerprint_sha256": graph["composite_fingerprint_sha256"],
        "model": graph["model"],
        "architecture": graph["architecture"],
        "body_abi": graph["body_abi"],
        "tokenizer": graph["tokenizer"],
        "source_lineage": graph.get("source_lineage", {}),
        "topology": graph["topology"],
        "logical_blocks": graph["logical_blocks"],
        "routes": graph["routes"],
        "components": {
            role: {
                "role": record["role"],
                "semantic_content_sha256": record["semantic_content_sha256"],
                "manifest_sha256": record["manifest_sha256"],
                "allowed_names": record["allowed_names"],
                "blobs": {
                    filename: {
                        "bytes": record["blobs"][filename]["bytes"],
                        "sha256": record["blobs"][filename]["sha256"],
                    }
                    for filename in COMPONENT_PAYLOAD_FILES
                },
            }
            for role, record in sorted(components.items())
        },
        "operation_contracts": graph["operation_contracts"],
    }


def _read_span(path: Path, offset: int, length: int) -> bytes:
    if offset < 0 or length < 0:
        raise ComponentGraphError(f"negative component payload span in {path}")
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            value = handle.read(length)
    except OSError as exc:
        raise ComponentGraphError(f"cannot read component payload span {path}") from exc
    if len(value) != length:
        raise ComponentGraphError(f"component payload span exceeds {path}")
    return value


def _component_semantic_digest(
    component_dir: Path,
    manifest: Mapping[str, Any],
    allowed_names: Sequence[str],
) -> str:
    """Recompute the Stage-4 layout-independent component semantic digest."""

    blocks = manifest.get("blocks")
    if not isinstance(blocks, Mapping):
        raise ComponentGraphError("component manifest has no block table")
    entries: list[dict[str, str]] = []
    for name in sorted(str(value) for value in allowed_names):
        block = blocks.get(name)
        if not isinstance(block, Mapping):
            raise ComponentGraphError(f"component semantic block {name!r} is absent")
        digest = hashlib.sha256()
        if "alias" in block:
            digest.update(canonical_json_bytes({"name": name, "alias": str(block["alias"])}))
        else:
            kind = str(block.get("kind", ""))
            shape = block.get("shape")
            if not isinstance(shape, list):
                raise ComponentGraphError(f"component block {name!r} has no shape")
            digest.update(
                canonical_json_bytes(
                    {"name": name, "kind": kind, "shape": [int(value) for value in shape]}
                )
            )
            try:
                layout = _COMPONENT_PAYLOAD_LAYOUT[kind]
            except KeyError as exc:
                raise ComponentGraphError(
                    f"unsupported component block kind {kind!r} for {name!r}"
                ) from exc
            for filename, offset_key, length_key in layout:
                length = int(block[length_key])
                value = _read_span(component_dir / filename, int(block[offset_key]), length)
                digest.update(canonical_json_bytes({"file_kind": filename, "bytes": length}))
                digest.update(value)
        entries.append({"name": name, "semantic_block_sha256": digest.hexdigest()})
    return _sha256_bytes(canonical_json_bytes(entries))


def _validate_component_int8_layout(
    component_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Validate a component's fixed int8 spans, allowing only zero tail padding."""

    if manifest.get("dtype") != "int8":
        raise ComponentGraphError("component manifest must declare dtype='int8'")
    blocks = manifest.get("blocks")
    if not isinstance(blocks, Mapping) or not blocks:
        raise ComponentGraphError("component manifest requires a non-empty block table")
    intervals: dict[str, list[tuple[int, int, str]]] = {
        filename: [] for filename in COMPONENT_PAYLOAD_FILES
    }
    for name, block in blocks.items():
        if not isinstance(name, str) or not name or not isinstance(block, Mapping):
            raise ComponentGraphError("component block descriptors must be named objects")
        alias = block.get("alias")
        if alias is not None:
            if not isinstance(alias, str) or not alias:
                raise ComponentGraphError(f"component alias {name!r} has an invalid target")
            continue
        shape = block.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
        ):
            raise ComponentGraphError(f"component block {name!r} has an invalid shape")
        kind = str(block.get("kind", ""))
        if kind == "qrow":
            if len(shape) != 2:
                raise ComponentGraphError(f"component qrow {name!r} must be a matrix")
            rows, columns = (int(value) for value in shape)
            expected = (
                ("weights.i8", "w_off", "w_len", rows * columns, 1),
                ("scales.f32", "s_off", "s_len", rows * 4, 4),
            )
        elif kind == "fp32":
            elements = int(np.prod(np.asarray(shape, dtype=np.int64)))
            expected = (("extras.f32", "e_off", "e_len", elements * 4, 4),)
        else:
            raise ComponentGraphError(f"unsupported component block kind {kind!r}")
        for filename, offset_key, length_key, expected_length, alignment in expected:
            offset = block.get(offset_key)
            length = block.get(length_key)
            if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0
                or isinstance(length, bool)
                or not isinstance(length, int)
                or length != expected_length
                or offset % alignment
            ):
                raise ComponentGraphError(
                    f"component block {name!r} has an invalid {filename} span"
                )
            intervals[filename].append((offset, offset + length, name))

    for name, block in blocks.items():
        if not isinstance(block, Mapping) or "alias" not in block:
            continue
        _resolve_alias(blocks, str(name))

    for filename, ranges in intervals.items():
        path = component_dir / filename
        if path.is_symlink() or not path.is_file():
            raise ComponentGraphError(f"component payload must be a regular file: {filename}")
        file_size = int(path.stat().st_size)
        previous_end = 0
        for start, end, block_name in sorted(ranges):
            if start != previous_end:
                qualifier = "overlaps" if start < previous_end else "has an internal gap before"
                raise ComponentGraphError(f"component {filename} {qualifier} block {block_name!r}")
            if end > file_size:
                raise ComponentGraphError(
                    f"component {filename} span for {block_name!r} exceeds file bounds"
                )
            previous_end = end
        padding = _read_span(path, previous_end, file_size - previous_end)
        if any(padding):
            raise ComponentGraphError(f"component {filename} has non-zero unreferenced padding")


def _validate_logical_int8_layout(
    blocks: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate the virtual monolithic spans consumed by memory admission.

    Component manifests use component-local offsets, while ``logical_blocks`` retains the
    source store's virtual monolithic layout. Runtime routing needs only kind/shape, but the
    compiler uses these logical spans to account scheduled and compact bytes. Consequently,
    a graph must not declare zero-length or overlapping spans while routing exact payload
    bytes from independently verified component manifests.
    """

    intervals: dict[str, list[tuple[int, int, str]]] = {
        filename: [] for filename in COMPONENT_PAYLOAD_FILES
    }
    for name, block in blocks.items():
        descriptor = _semantic_block_descriptor(block, name=name)
        if "alias" in descriptor:
            continue
        shape = descriptor["shape"]
        kind = descriptor["kind"]
        if kind == "qrow":
            if len(shape) != 2:
                raise ComponentGraphError(f"logical qrow block {name!r} must be a matrix")
            rows, columns = (int(value) for value in shape)
            expected = (
                ("weights.i8", "w_off", "w_len", rows * columns, 1),
                ("scales.f32", "s_off", "s_len", rows * 4, 4),
            )
        else:
            elements = 1
            for dimension in shape:
                elements *= int(dimension)
            expected = (("extras.f32", "e_off", "e_len", elements * 4, 4),)
        for filename, offset_key, length_key, expected_length, alignment in expected:
            offset = block.get(offset_key)
            length = block.get(length_key)
            if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0
                or isinstance(length, bool)
                or not isinstance(length, int)
                or length != expected_length
                or offset % alignment
            ):
                raise ComponentGraphError(f"logical block {name!r} has an invalid {filename} span")
            intervals[filename].append((offset, offset + length, name))

    for name in blocks:
        _resolve_alias(blocks, name)

    for filename, ranges in intervals.items():
        previous_end = 0
        previous_name: str | None = None
        for start, end, block_name in sorted(ranges):
            if start < previous_end:
                raise ComponentGraphError(
                    f"logical {filename} span for {block_name!r} overlaps {previous_name!r}"
                )
            previous_end = end
            previous_name = block_name


def _semantic_block_descriptor(block: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    alias = block.get("alias")
    if alias is not None:
        if not isinstance(alias, str) or not alias:
            raise ComponentGraphError(f"block {name!r} has an invalid alias target")
        return {"alias": alias}
    kind = block.get("kind")
    shape = block.get("shape")
    if kind not in {"qrow", "fp32"}:
        raise ComponentGraphError(f"block {name!r} has unsupported kind {kind!r}")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
        )
    ):
        raise ComponentGraphError(f"block {name!r} has an invalid shape")
    return {"kind": kind, "shape": list(shape)}


def _body_abi_semantic_sha256(
    *,
    architecture: str,
    config: Mapping[str, Any],
    tied: bool,
    logical_blocks: Mapping[str, Mapping[str, Any]],
) -> str:
    descriptors = {
        name: _semantic_block_descriptor(block, name=name)
        for name, block in sorted(logical_blocks.items())
    }
    return _sha256_bytes(
        canonical_json_bytes(
            {
                "architecture": architecture,
                "config": config,
                "tie_word_embeddings": bool(tied),
                "logical_blocks": descriptors,
            }
        )
    )


class ComponentGraph:
    """Validated, relocatable component graph with lazy blob verification."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute()
        self.root = self.path.parent.resolve()
        self.raw, self._graph_stat, _graph_sha256 = _read_json_verified(self.path)
        self.schema = str(self.raw.get("schema", ""))
        if self.schema not in SUPPORTED_COMPONENT_GRAPH_SCHEMAS:
            raise ComponentGraphError(f"unsupported component graph schema {self.schema!r}")
        self.declared_fingerprint = _require_sha256(
            self.raw.get("composite_fingerprint_sha256"),
            "component graph fingerprint",
        )
        expected = _sha256_bytes(canonical_json_bytes(_graph_fingerprint_payload(self.raw)))
        if self.declared_fingerprint != expected:
            raise ComponentGraphError("composite graph fingerprint mismatch")

        self.model_name = str(self.raw.get("model", ""))
        self.architecture = str(self.raw.get("architecture", ""))
        if not self.model_name or not self.architecture:
            raise ComponentGraphError("component graph model or architecture is absent")
        body_abi = self.raw.get("body_abi")
        tokenizer = self.raw.get("tokenizer")
        logical_blocks = self.raw.get("logical_blocks")
        components = self.raw.get("components")
        routes = self.raw.get("routes")
        contracts = self.raw.get("operation_contracts")
        topology = self.raw.get("topology")
        if not all(
            isinstance(value, Mapping)
            for value in (
                body_abi,
                tokenizer,
                logical_blocks,
                components,
                routes,
                contracts,
                topology,
            )
        ):
            raise ComponentGraphError("component graph contains a malformed object field")
        self.body_abi = _json_clone(body_abi)
        _require_sha256(self.body_abi.get("semantic_sha256"), "body ABI semantic_sha256")
        self.logical_blocks: dict[str, dict[str, Any]] = {
            str(name): _json_clone(block)
            for name, block in logical_blocks.items()
            if isinstance(block, Mapping)
        }
        if len(self.logical_blocks) != len(logical_blocks):
            raise ComponentGraphError("logical block table contains a non-object entry")

        declared_tied = bool(topology.get("declared_tied", False))
        role_names, observed_topology = _allocation_topology(
            self.logical_blocks,
            declared_tied=declared_tied,
        )
        if canonical_json_bytes(observed_topology) != canonical_json_bytes(topology):
            raise ComponentGraphError("stored topology differs from logical alias topology")
        self.topology = observed_topology

        self.components: dict[str, dict[str, Any]] = {}
        owned_names: dict[str, str] = {}
        for raw_role, raw_record in components.items():
            role = str(raw_role)
            if not isinstance(raw_record, Mapping):
                raise ComponentGraphError(f"component record {role!r} is not an object")
            record = _json_clone(raw_record)
            if str(record.get("role")) != role:
                raise ComponentGraphError(f"component record role mismatch for {role!r}")
            allowed_names = tuple(str(name) for name in record.get("allowed_names", ()))
            if not allowed_names or len(set(allowed_names)) != len(allowed_names):
                raise ComponentGraphError(f"component {role!r} has invalid allowed_names")
            for name in allowed_names:
                if name in owned_names:
                    raise ComponentGraphError(f"logical block {name!r} has multiple owners")
                owned_names[name] = role
            _require_sha256(record.get("manifest_sha256"), f"{role} manifest_sha256")
            _require_sha256(
                record.get("semantic_content_sha256"),
                f"{role} semantic_content_sha256",
            )
            blobs = record.get("blobs")
            if not isinstance(blobs, Mapping) or not blobs:
                raise ComponentGraphError(f"component {role!r} has no blob records")
            if set(blobs) != set(COMPONENT_PAYLOAD_FILES):
                raise ComponentGraphError(
                    f"component {role!r} must declare exactly {list(COMPONENT_PAYLOAD_FILES)!r}"
                )
            for filename, blob in blobs.items():
                if not isinstance(filename, str) or not isinstance(blob, Mapping):
                    raise ComponentGraphError(f"component {role!r} has an invalid blob record")
                if int(blob.get("bytes", 0)) <= 0:
                    raise ComponentGraphError(f"component {role!r}/{filename} has invalid size")
                _require_sha256(blob.get("sha256"), f"{role}/{filename} sha256")
            self.components[role] = record

        self.fingerprint = _sha256_bytes(
            canonical_json_bytes(_custody_fingerprint_payload(self.raw))
        )

        self.routes = {str(name): str(role) for name, role in routes.items()}
        if self.routes != dict(sorted(owned_names.items())):
            raise ComponentGraphError("graph routes and component ownership differ")
        if set(self.routes) != set(self.logical_blocks):
            raise ComponentGraphError("graph routes do not cover every logical block exactly once")
        if {role: sorted(record["allowed_names"]) for role, record in self.components.items()} != {
            role: sorted(names) for role, names in role_names.items()
        }:
            raise ComponentGraphError("component roles disagree with allocation-aware partition")

        expected_contracts = _expected_operation_contracts(role_names)
        if canonical_json_bytes(expected_contracts) != canonical_json_bytes(contracts):
            raise ComponentGraphError("operation contracts do not match graph topology")
        self.operation_contracts = expected_contracts

        self._manifest_stats: dict[str, tuple[int, int, int, int, int]] = {}
        self._component_manifests: dict[str, dict[str, Any]] = {}
        self._verified_blob_stats: dict[str, dict[str, tuple[int, int, int, int, int]]] = {}
        self._validate_component_manifests()
        _validate_logical_int8_layout(self.logical_blocks)

        config = self.body_abi.get("config")
        if not isinstance(config, Mapping):
            # Stage-4 PoC graphs intentionally kept the ABI small.  The complete runtime
            # configuration remains in each component's graph-hashed manifest, so that is
            # the authoritative compatibility bridge for the legacy schema.
            config = self._component_manifests["body"].get("config")
        if not isinstance(config, Mapping):
            raise ComponentGraphError("body ABI and body component have no runtime config")
        for role, manifest in self._component_manifests.items():
            component_config = manifest.get("config")
            if not isinstance(component_config, Mapping):
                raise ComponentGraphError(f"{role} component has no runtime config")
            if canonical_json_bytes(component_config) != canonical_json_bytes(config):
                raise ComponentGraphError(f"{role} component runtime config differs from body")
            if bool(manifest.get("tie_word_embeddings", False)) != bool(
                self.topology["observed_tied"]
            ):
                raise ComponentGraphError(
                    f"{role} component tie declaration differs from graph topology"
                )
        abi_arch = str(self.body_abi.get("architecture", self.architecture))
        if abi_arch != self.architecture:
            raise ComponentGraphError("body ABI architecture differs from component graph")
        abi_hidden = int(self.body_abi.get("hidden_size", config.get("hidden_size", 0)))
        if abi_hidden != int(config.get("hidden_size", 0)):
            raise ComponentGraphError("body ABI hidden size differs from runtime config")
        configured_rows = int(config.get("vocab_size", 0))
        abi_rows = int(self.body_abi.get("vocab_size", configured_rows))
        if abi_rows != configured_rows:
            raise ComponentGraphError("body ABI vocabulary size differs from runtime config")
        logical_count = int(self.body_abi.get("logical_block_count", len(self.logical_blocks)))
        if logical_count != len(self.logical_blocks):
            raise ComponentGraphError("body ABI logical block count differs from graph")
        observed_abi = _body_abi_semantic_sha256(
            architecture=self.architecture,
            config=config,
            tied=bool(self.topology["observed_tied"]),
            logical_blocks=self.logical_blocks,
        )
        if observed_abi != self.body_abi["semantic_sha256"]:
            raise ComponentGraphError(
                "body ABI semantic hash differs from runtime config and logical blocks"
            )
        self.vocab = VocabManifest.from_descriptor(
            tokenizer,
            configured_row_count=configured_rows,
        )
        embed = self.logical_blocks["embed"]
        lm_head = self.logical_blocks["lm_head"]
        embed_root = self.logical_blocks[_resolve_alias(self.logical_blocks, "embed")]
        head_root = self.logical_blocks[_resolve_alias(self.logical_blocks, "lm_head")]
        for name, block in (("embed", embed_root), ("lm_head", head_root)):
            shape = block.get("shape")
            if not isinstance(shape, list) or len(shape) != 2:
                raise ComponentGraphError(f"{name} has no matrix shape")
            if int(shape[0]) != configured_rows:
                raise ComponentGraphError(
                    f"{name} rows {shape[0]} disagree with configured vocab {configured_rows}"
                )
        del embed, lm_head

        self.source_identity_status = str(
            self.raw.get("source_lineage", {}).get("identity_status", "legacy-unverified")
        )

    def component_path(self, role: str) -> Path:
        try:
            record = self.components[str(role)]
        except KeyError as exc:
            raise ComponentGraphError(f"unknown component role {role!r}") from exc
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ComponentGraphError(f"component {role!r} has no relative locator")
        relative = Path(relative_path)
        if relative.is_absolute() or any(part == ".." for part in relative.parts):
            raise ComponentGraphError("component locator must be a relative in-root path")
        unresolved = self.root
        for part in relative.parts:
            if part in {"", "."}:
                continue
            unresolved /= part
            if unresolved.is_symlink():
                raise ComponentGraphError("component locator cannot traverse a symlink")
        component_dir = unresolved.resolve()
        try:
            component_dir.relative_to(self.root)
        except ValueError as exc:
            raise ComponentGraphError("component locator escapes the graph root") from exc
        if not component_dir.is_dir():
            raise ComponentGraphError(f"component locator is not a directory: {role!r}")
        return component_dir

    def _validate_component_manifests(self) -> None:
        for role, record in self.components.items():
            component_dir = self.component_path(role)
            manifest_path = component_dir / "manifest.json"
            manifest, manifest_stat, manifest_digest = _read_json_verified(manifest_path)
            if manifest_digest != record["manifest_sha256"]:
                raise ComponentGraphError(f"{role} component manifest hash mismatch")
            blocks = manifest.get("blocks")
            if not isinstance(blocks, Mapping):
                raise ComponentGraphError(f"{role} component has no block table")
            if set(blocks) != set(record["allowed_names"]):
                raise ComponentGraphError(f"{role} component block ownership mismatch")
            if str(manifest.get("arch", self.architecture)) != self.architecture:
                raise ComponentGraphError(f"{role} component architecture mismatch")
            _validate_component_int8_layout(component_dir, manifest)
            for name in record["allowed_names"]:
                logical = self.logical_blocks[name]
                physical = blocks[name]
                logical_descriptor = _semantic_block_descriptor(logical, name=name)
                physical_descriptor = _semantic_block_descriptor(physical, name=name)
                if logical_descriptor != physical_descriptor:
                    raise ComponentGraphError(
                        f"{role} component block {name!r} differs from its logical descriptor"
                    )
            self._component_manifests[role] = manifest
            self._manifest_stats[role] = manifest_stat

    def verify_component_blobs(self, role: str) -> Path:
        role = str(role)
        component_dir = self.component_path(role)
        record = self.components[role]
        current_manifest = component_dir / "manifest.json"
        _manifest, manifest_stat, manifest_digest = _read_json_verified(current_manifest)
        if (
            manifest_stat != self._manifest_stats[role]
            or manifest_digest != record["manifest_sha256"]
        ):
            raise ComponentGraphError(f"{role} component manifest changed")
        stats: dict[str, tuple[int, int, int, int, int]] = {}
        for filename in COMPONENT_PAYLOAD_FILES:
            expected = record["blobs"][filename]
            path = component_dir / filename
            if path.is_symlink():
                raise ComponentGraphError(
                    f"component payload cannot be a symlink: {role}/{filename}"
                )
            try:
                path.resolve().relative_to(component_dir)
            except ValueError as exc:
                raise ComponentGraphError(
                    f"component payload escapes its component directory: {role}/{filename}"
                ) from exc
            digest, file_stat = _sha256_file_verified(path)
            if file_stat[2] != int(expected["bytes"]):
                raise ComponentGraphError(f"{role}/{filename} size mismatch")
            if digest != expected["sha256"]:
                raise ComponentGraphError(f"{role}/{filename} content hash mismatch")
            stats[filename] = file_stat
        semantic = _component_semantic_digest(
            component_dir,
            self._component_manifests[role],
            record["allowed_names"],
        )
        if semantic != record["semantic_content_sha256"]:
            raise ComponentGraphError(f"{role} component semantic content hash mismatch")
        for filename, expected_stat in stats.items():
            if _stat_identity(component_dir / filename) != expected_stat:
                raise ComponentGraphError(f"{role}/{filename} changed during semantic verification")
        self._verified_blob_stats[role] = stats
        return component_dir

    def assert_store_matches_verified_component(self, role: str, store: QStore) -> None:
        """Bind QStore's mapped files to the exact inodes accepted by graph custody."""

        self.assert_unchanged()
        try:
            store_records = {
                str(record[0]): (
                    int(record[1]),
                    int(record[2]),
                    int(record[4]),
                    int(record[5]),
                    int(record[6]),
                )
                for record in store._verified_file_stats  # noqa: SLF001 - custody bridge
            }
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ComponentGraphError("opened QStore exposes no verified file custody") from exc
        expected = {"manifest.json": self._manifest_stats[role]}
        expected.update(self._verified_blob_stats[role])
        if store_records != expected:
            raise ComponentGraphError(
                f"{role} QStore mappings do not match the graph-verified component files"
            )

    def assert_unchanged(self) -> None:
        if _stat_identity(self.path) != self._graph_stat:
            raise ComponentGraphError("component graph file changed after validation")
        for role, expected in self._manifest_stats.items():
            if _stat_identity(self.component_path(role) / "manifest.json") != expected:
                raise ComponentGraphError(f"{role} component manifest changed after validation")
        for role, records in self._verified_blob_stats.items():
            component_dir = self.component_path(role)
            for filename, expected in records.items():
                if _stat_identity(component_dir / filename) != expected:
                    raise ComponentGraphError(
                        f"{role}/{filename} changed after content verification"
                    )


def inspect_component_graph_fingerprint(path: str | Path) -> str:
    """Validate graph structure/manifests and return its production custody fingerprint."""

    return ComponentGraph(path).fingerprint


def _qrow_compact_bytes(store: QStore, name: str, rows: int | None = None) -> int:
    block = store._resolve(name)  # noqa: SLF001 - physical telemetry needs resolved layout
    total_rows = int(block["shape"][0])
    selected = total_rows if rows is None else int(rows)
    weight_per_row = int(block.get("w_len", 0)) // total_rows
    scales_per_row = int(block.get("s_len", 0)) // total_rows
    return selected * (weight_per_row + scales_per_row)


class _ComponentProvider:
    """One component QStore with strict ownership and physical-read telemetry."""

    def __init__(self, *, role: str, store: QStore, allowed_names: Iterable[str]) -> None:
        self.role = str(role)
        self.store = store
        self.allowed_names = frozenset(str(name) for name in allowed_names)
        self.calls: Counter[str] = Counter()
        self.rows_read: Counter[str] = Counter()
        self.compact_bytes_addressed: Counter[str] = Counter()

    @property
    def max_block_bytes(self) -> int:
        return int(self.store.max_block_bytes)

    def _require(self, method: str, name: str) -> dict[str, Any]:
        if name not in self.allowed_names:
            raise ComponentGraphError(
                f"{self.role} component cannot serve out-of-role block {name!r}"
            )
        self.calls[f"{method}:{name}"] += 1
        return self.store._resolve(name)  # noqa: SLF001 - telemetry follows physical alias

    def has(self, name: str) -> bool:
        return name in self.allowed_names and self.store.has(name)

    def fp32(self, name: str) -> torch.Tensor:
        block = self._require("fp32", name)
        self.compact_bytes_addressed[name] += int(block.get("e_len", 0))
        return self.store.fp32(name)

    def weight(self, name: str) -> torch.Tensor:
        block = self._require("weight", name)
        rows = int(block["shape"][0])
        self.rows_read[name] += rows
        self.compact_bytes_addressed[name] += _qrow_compact_bytes(self.store, name)
        return self.store.weight(name)

    def matmul(self, name: str, value: torch.Tensor) -> torch.Tensor:
        block = self._require("matmul", name)
        rows = int(block["shape"][0])
        self.rows_read[name] += rows
        self.compact_bytes_addressed[name] += _qrow_compact_bytes(self.store, name)
        return self.store.matmul(name, value)

    def matmul_row_stable(self, name: str, value: torch.Tensor) -> torch.Tensor:
        block = self._require("matmul_row_stable", name)
        rows = int(block["shape"][0])
        self.rows_read[name] += rows
        self.compact_bytes_addressed[name] += _qrow_compact_bytes(self.store, name)
        return self.store.matmul_row_stable(name, value)

    def embed_rows(
        self,
        name: str,
        ids: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        self._require("embed_rows", name)
        count = int(ids.numel()) if isinstance(ids, torch.Tensor) else int(np.asarray(ids).size)
        self.rows_read[name] += count
        self.compact_bytes_addressed[name] += _qrow_compact_bytes(self.store, name, count)
        return self.store.embed_rows(name, ids)

    def row_blocks(self, name: str, bs: int = 8192):
        self._require("row_blocks", name)
        for start, end, value in self.store.row_blocks(name, bs=bs):
            count = int(end - start)
            self.rows_read[name] += count
            self.compact_bytes_addressed[name] += _qrow_compact_bytes(self.store, name, count)
            yield start, end, value

    def prepare_resident_exact_head(
        self,
        name: str,
        *,
        max_resident_bytes: int,
    ) -> torch.Tensor:
        self._require("prepare_resident_exact_head", name)
        prepare = getattr(self.store, "prepare_resident_exact_head", None)
        if not callable(prepare):
            raise ComponentGraphError(
                f"{self.role} provider cannot materialize a resident exact head"
            )
        return prepare(name, max_resident_bytes=max_resident_bytes)

    def resident_exact_head_fp32(self, name: str) -> torch.Tensor | None:
        self._require("resident_exact_head_fp32", name)
        resident = getattr(self.store, "resident_exact_head_fp32", None)
        return resident(name) if callable(resident) else None

    def snapshot(self) -> dict[str, Any]:
        compact_budget = getattr(self.store, "compact_cache_budget", None)
        compact_resident = getattr(self.store, "compact_cache_bytes", None)
        store_stats = getattr(self.store, "stats_snapshot", None)
        compact_keys: set[str] = set()
        auxiliary_keys: set[str] = set()
        compact_required_bytes = 0
        auxiliary_required_bytes = 0
        for name in self.allowed_names:
            physical = _resolve_alias(self.store.blocks, name)
            block = self.store.blocks[physical]
            if block.get("kind") == "qrow" and physical not in compact_keys:
                compact_keys.add(physical)
                compact_required_bytes += _qrow_compact_bytes(self.store, physical)
            elif block.get("kind") == "fp32" and physical not in auxiliary_keys:
                auxiliary_keys.add(physical)
                auxiliary_required_bytes += int(block.get("e_len", 0))
        resident_keys = set(getattr(self.store, "compact_cache", ()))
        resident_auxiliary_keys = set(getattr(self.store, "fp32_aux_cache", ()))
        compact_fully_resident = compact_keys <= resident_keys
        auxiliary_fully_resident = auxiliary_keys <= resident_auxiliary_keys
        auxiliary_resident_bytes = int(getattr(self.store, "fp32_aux_cache_bytes", 0))
        fully_resident = compact_fully_resident and auxiliary_fully_resident
        resident_exact_head_bytes = int(getattr(self.store, "resident_exact_head_bytes", 0))
        return {
            "role": self.role,
            "provider": ("dense-qstore-cuda" if compact_budget is not None else "paged-qstore"),
            "device": str(self.store.device),
            "compute_dtype": str(self.store.compute_dtype),
            "calls": dict(sorted(self.calls.items())),
            "rows_read": dict(sorted(self.rows_read.items())),
            "compact_bytes_addressed": dict(sorted(self.compact_bytes_addressed.items())),
            "cache_bytes": int(
                compact_resident
                if compact_resident is not None
                else getattr(self.store, "_cache_bytes", 0)
            ),
            "cache_budget_bytes": int(
                compact_budget
                if compact_budget is not None
                else getattr(self.store, "_cache_budget", 0)
            ),
            "cache_kind": (
                "compact-device-lru"
                if compact_budget is not None
                else str(getattr(self.store, "_cache_policy", "dequantized-weight-cache"))
            ),
            "store_stats": store_stats() if callable(store_stats) else None,
            "required_compact_qrow_bytes": compact_required_bytes,
            "resident_compact_qrow_bytes": int(compact_resident or 0),
            "compact_qrows_fully_resident": compact_fully_resident,
            "required_auxiliary_fp32_bytes": auxiliary_required_bytes,
            "resident_auxiliary_fp32_bytes": auxiliary_resident_bytes,
            "auxiliary_fp32_fully_resident": auxiliary_fully_resident,
            "required_physical_bytes": compact_required_bytes + auxiliary_required_bytes,
            "resident_physical_bytes": int(compact_resident or 0) + auxiliary_resident_bytes,
            "fully_resident": fully_resident,
            "resident_exact_head_bytes": resident_exact_head_bytes,
            "resident_exact_head": bool(resident_exact_head_bytes),
            "max_block_bytes": self.max_block_bytes,
        }

    def set_cache_budget(self, cache_mb: float) -> None:
        compact_setter = getattr(self.store, "set_compact_cache_budget", None)
        if callable(compact_setter):
            compact_setter(cache_mb)
        else:
            self.store.set_cache_budget(cache_mb)


def _normalize_output_contract(value: Any) -> str:
    raw = str(getattr(value, "value", value))
    return {
        "full_logits": "full_logits",
        "last_token_logits": "full_logits",
        "loss_only": "full_logits",
        "selected_token_rows": "selected_rows",
        "candidate_argmax_and_margin": "selected_rows",
        "hidden_state_only": "lexical_hidden",
        "lexical_hidden": "lexical_hidden",
        "selected_rows": "selected_rows",
    }.get(raw, raw)


class ContractQStoreView:
    """Immutable logical capability view over lazily shared physical providers."""

    def __init__(self, owner: CompositeQStore, output_contract: str) -> None:
        self.owner = owner
        self.output_contract = _normalize_output_contract(output_contract)
        try:
            contract = owner.graph.operation_contracts[self.output_contract]
        except KeyError as exc:
            raise ComponentOutputContractError(
                f"unknown component output contract {output_contract!r}"
            ) from exc
        self.required_components = frozenset(contract["required_components"])
        self.allowed_lm_head_methods = frozenset(contract["lm_head_methods"])
        self.route_counts: Counter[str] = Counter()

    @property
    def man(self) -> Mapping[str, Any]:
        return self.owner.man

    @property
    def cfg(self) -> Mapping[str, Any]:
        return self.owner.cfg

    @property
    def blocks(self) -> Mapping[str, Any]:
        return self.owner.blocks

    @property
    def device(self) -> str:
        return self.owner.device

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.owner.compute_dtype

    @property
    def require_triton(self) -> bool:
        return bool(getattr(self.owner._providers["body"].store, "require_triton", False))

    @property
    def stable_block_m(self) -> int:
        return int(getattr(self.owner._providers["body"].store, "stable_block_m", 16))

    @property
    def max_block_bytes(self) -> int:
        return self.owner.max_block_bytes

    @property
    def _cache_budget(self) -> int:
        return self.owner._cache_budget  # noqa: SLF001 - QStore compatibility surface

    @property
    def content_identity_verified(self) -> bool:
        return self.owner.content_identity_verified

    @property
    def identity_status(self) -> str:
        return self.owner.identity_status

    @property
    def store_identity(self) -> Mapping[str, Any]:
        return self.owner.store_identity

    @property
    def source_checkpoint_sha256(self) -> str | None:
        return self.owner.source_checkpoint_sha256

    @property
    def derived_store_sha256(self) -> str:
        return self.owner.derived_store_sha256

    def _guard(self, method: str, name: str) -> None:
        if name == "lm_head" and method not in self.allowed_lm_head_methods:
            raise ComponentOutputContractError(f"{self.output_contract} denies {method}({name!r})")

    def _provider(self, method: str, name: str) -> _ComponentProvider:
        self._guard(method, name)
        try:
            role = self.owner.graph.routes[name]
        except KeyError as exc:
            raise ComponentGraphError(f"component graph has no route for {name!r}") from exc
        if role not in self.required_components:
            raise ComponentOutputContractError(
                f"{self.output_contract} does not admit component {role!r} for {name!r}"
            )
        provider = self.owner._provider(role)  # noqa: SLF001 - immutable routed view
        self.route_counts[f"{role}:{method}:{name}"] += 1
        return provider

    def has(self, name: str) -> bool:
        if name == "lm_head" and not self.allowed_lm_head_methods:
            return False
        role = self.owner.graph.routes.get(name)
        return (
            role in self.required_components
            and name in self.owner.graph.components[str(role)]["allowed_names"]
        )

    def fp32(self, name: str) -> torch.Tensor:
        return self._provider("fp32", name).fp32(name)

    def weight(self, name: str) -> torch.Tensor:
        return self._provider("weight", name).weight(name)

    def matmul(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return self._provider("matmul", name).matmul(name, value)

    def matmul_row_stable(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return self._provider("matmul_row_stable", name).matmul_row_stable(name, value)

    def embed_rows(
        self,
        name: str,
        ids: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        token_ids: np.ndarray | torch.Tensor
        if name in {"embed", "lm_head"}:
            token_ids = self.owner.vocab.validate_token_ids(ids)
        elif isinstance(ids, torch.Tensor):
            token_ids = ids.to(dtype=torch.long)
        else:
            token_ids = np.asarray(ids, dtype=np.int64)
        return self._provider("embed_rows", name).embed_rows(name, token_ids)

    def _embed_rows_trusted_generated(
        self,
        name: str,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        """Internal-only route for IDs produced by this target's semantic-masked head."""

        if (
            not isinstance(ids, torch.Tensor)
            or ids.dtype != torch.long
            or str(ids.device) != str(self.device)
        ):
            raise RuntimeError("trusted generated IDs must be device-local int64 tensors")
        return self._provider("embed_rows", name).embed_rows(name, ids)

    def row_blocks(self, name: str, bs: int = 8192):
        provider = self._provider("row_blocks", name)
        yield from provider.row_blocks(name, bs=bs)

    def resident_exact_head_fp32(self, name: str = "lm_head") -> torch.Tensor | None:
        return self._provider("row_blocks", name).resident_exact_head_fp32(name)

    def set_cache_budget(self, cache_mb: float) -> None:
        self.owner.set_cache_budget(cache_mb)

    def assert_content_identity_unchanged(self) -> None:
        self.owner.assert_content_identity_unchanged()

    def ring_stats(self) -> dict[str, int] | None:
        return self.owner.ring_stats()

    def snapshot(self) -> dict[str, Any]:
        return {
            "output_contract": self.output_contract,
            "required_components": sorted(self.required_components),
            "allowed_lm_head_methods": sorted(self.allowed_lm_head_methods),
            "route_counts": dict(sorted(self.route_counts.items())),
            "opened_roles": sorted(self.owner.opened_roles),
        }

    def stats_snapshot(self) -> dict[str, Any]:
        return self.owner.snapshot()


class CompositeQStore:
    """QStore-compatible manager over an alias-aware component graph.

    The default duck-typed methods expose the complete-logit contract for legacy paged
    callers.  Output-aware code must call :meth:`for_contract` and receives a distinct,
    immutable capability view.  Providers are shared and opened lazily by physical role.
    """

    def __init__(
        self,
        graph_path: str | Path,
        *,
        cache_mb: float = 0.0,
        component_cache_mb: Mapping[str, float] | None = None,
        compute_dtype: str | None = None,
        provider_backend: str = "paged-qstore",
        provider_device: str | None = None,
        provider_require_triton: bool = True,
        provider_stable_block_m: int = 16,
        provider_pin_fp32_aux: bool = False,
    ) -> None:
        provider_backend = str(provider_backend).strip().lower().replace("_", "-")
        aliases = {
            "paged": "paged-qstore",
            "qstore": "paged-qstore",
            "dense-cuda": "dense-qstore-cuda",
        }
        provider_backend = aliases.get(provider_backend, provider_backend)
        if provider_backend not in {"paged-qstore", "dense-qstore-cuda"}:
            raise ValueError("provider_backend must be 'paged-qstore' or 'dense-qstore-cuda'")
        self.graph = ComponentGraph(graph_path)
        self.directory = self.graph.root
        self.composite_fingerprint_sha256 = self.graph.fingerprint
        self.vocab = self.graph.vocab
        self.vocab_manifest_sha256 = self.vocab.semantic_sha256
        self.content_identity_verified = False
        self.identity_status = (
            "component-graph-verified-source-" + self.graph.source_identity_status
        )
        self.source_checkpoint_sha256 = None
        self.derived_store_sha256 = self.graph.fingerprint
        self.store_identity = {
            "content_identity_verified": False,
            "blob_identity_verified": False,
            "semantic_identity_verified": True,
            "identity_status": self.identity_status,
            "component_graph_fingerprint": self.graph.fingerprint,
            "vocab_manifest_sha256": self.vocab.semantic_sha256,
        }
        self._compute_dtype_name = compute_dtype
        self.provider_backend = provider_backend
        self._provider_device = provider_device
        self._provider_require_triton = bool(provider_require_triton)
        self._provider_stable_block_m = int(provider_stable_block_m)
        self._provider_pin_fp32_aux = bool(provider_pin_fp32_aux)
        self._providers: dict[str, _ComponentProvider] = {}
        self._views: dict[str, ContractQStoreView] = {}
        self._resident_exact_head_admission: dict[str, Any] | None = None
        self._fully_resident_admission: dict[str, Any] | None = None
        self._closed = False
        self._lock = RLock()
        self._cache_budget = max(0, int(float(cache_mb) * 1e6))
        explicit = {
            str(role): max(0.0, float(value)) for role, value in (component_cache_mb or {}).items()
        }
        unknown = set(explicit) - set(self.graph.components)
        if unknown:
            raise ComponentGraphError(f"unknown component cache roles: {sorted(unknown)}")
        explicit_total = sum(explicit.values())
        if explicit and cache_mb and explicit_total > float(cache_mb) + 1e-9:
            raise ComponentGraphError("component cache budgets exceed the aggregate cache budget")
        if explicit:
            total_mb = float(cache_mb) if cache_mb else explicit_total
            remainder = max(0.0, total_mb - explicit_total)
            explicit["body"] = explicit.get("body", 0.0) + remainder
            self._cache_weights = (
                {role: value / total_mb for role, value in explicit.items()}
                if total_mb
                else {"body": 1.0}
            )
            self._cache_budget = int(total_mb * 1e6)
        else:
            self._cache_weights = {"body": 1.0}

        body = self._provider("body")
        self.cfg = _json_clone(body.store.cfg)
        self.blocks = _json_clone(self.graph.logical_blocks)
        logical_manifest = _json_clone(body.store.man)
        logical_manifest["model_name"] = self.graph.model_name
        logical_manifest["arch"] = self.graph.architecture
        logical_manifest["config"] = _json_clone(self.cfg)
        logical_manifest["blocks"] = _json_clone(self.blocks)
        logical_manifest["tie_word_embeddings"] = bool(self.graph.topology["observed_tied"])
        logical_manifest["vocab_manifest"] = self.vocab.as_dict()
        logical_manifest["component_graph"] = {
            "schema": self.graph.schema,
            "composite_fingerprint_sha256": self.graph.fingerprint,
            "declared_legacy_fingerprint_sha256": self.graph.declared_fingerprint,
            "custody_schema": COMPOSITE_CUSTODY_SCHEMA,
            "source_identity_status": self.graph.source_identity_status,
            "promotion_status": "blocked-legacy-component-lineage",
        }
        logical_manifest["derived"] = {
            "derived_store_sha256": self.graph.fingerprint,
        }
        self.man = logical_manifest
        self.device = body.store.device
        self.compute_dtype = body.store.compute_dtype
        self._default_view = self.for_contract("full_logits")

    @property
    def opened_roles(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._providers)

    @property
    def max_block_bytes(self) -> int:
        with self._lock:
            return max(
                (provider.max_block_bytes for provider in self._providers.values()),
                default=0,
            )

    def _role_cache_mb(self, role: str) -> float:
        return self._cache_budget * self._cache_weights.get(role, 0.0) / 1e6

    def _provider(self, role: str) -> _ComponentProvider:
        with self._lock:
            if self._closed:
                raise RuntimeError("CompositeQStore is closed")
            role = str(role)
            cached = self._providers.get(role)
            if cached is not None:
                return cached
            component_dir = self.graph.verify_component_blobs(role)
            if self.provider_backend == "dense-qstore-cuda":
                from .dense_qstore_cuda import DenseQStore

                store = DenseQStore(
                    component_dir.name,
                    root=component_dir.parent,
                    device=self._provider_device or "cuda",
                    compute_dtype=self._compute_dtype_name or "bf16",
                    compact_cache_mb=self._role_cache_mb(role),
                    require_triton=self._provider_require_triton,
                    stable_block_m=self._provider_stable_block_m,
                    pin_fp32_aux=self._provider_pin_fp32_aux,
                )
            else:
                store = QStore(
                    component_dir.name,
                    root=component_dir.parent,
                    cache_mb=self._role_cache_mb(role),
                    compute_dtype=self._compute_dtype_name,
                )
            try:
                self.graph.assert_store_matches_verified_component(role, store)
                if self.provider_backend == "dense-qstore-cuda" or role != "body":
                    store.disable_ring()
                if self._providers:
                    body = self._providers["body"].store
                    if canonical_json_bytes(store.cfg) != canonical_json_bytes(body.cfg):
                        raise ComponentGraphError(f"{role} component config differs from body")
                    if (
                        str(store.man.get("arch", self.graph.architecture))
                        != self.graph.architecture
                    ):
                        raise ComponentGraphError(
                            f"{role} component architecture differs from graph"
                        )
                    if str(store.device) != str(body.device):
                        raise ComponentGraphError(f"{role} component device differs from body")
                    if store.compute_dtype is not body.compute_dtype:
                        raise ComponentGraphError(
                            f"{role} component compute dtype differs from body"
                        )
                provider = _ComponentProvider(
                    role=role,
                    store=store,
                    allowed_names=self.graph.components[role]["allowed_names"],
                )
                self._providers[role] = provider
                return provider
            except Exception:
                store.close()
                raise

    def for_contract(self, output_contract: Any) -> ContractQStoreView:
        with self._lock:
            normalized = _normalize_output_contract(output_contract)
            view = self._views.get(normalized)
            if view is None:
                view = ContractQStoreView(self, normalized)
                self._views[normalized] = view
            return view

    def validate_tokenizer(self, tokenizer: Any) -> None:
        descriptor = tokenizer_descriptor(tokenizer)
        if descriptor["semantic_sha256"] != self.vocab.descriptor_semantic_sha256:
            raise ComponentGraphError("runtime tokenizer does not match component graph vocabulary")

    def assert_content_identity_unchanged(self) -> None:
        with self._lock:
            self.graph.assert_unchanged()
            for provider in self._providers.values():
                provider.store.assert_content_identity_unchanged()

    def set_cache_budget(self, cache_mb: float) -> None:
        with self._lock:
            proposed = max(0, int(float(cache_mb) * 1e6))
            for role, provider in self._providers.items():
                role_budget = proposed * self._cache_weights.get(role, 0.0)
                pinned = int(getattr(provider.store, "fp32_aux_cache_bytes", 0))
                if role_budget < pinned:
                    raise MemoryError(
                        f"{role} cache budget cannot retain pinned FP32 auxiliaries "
                        f"({int(role_budget)} < {pinned} bytes)"
                    )
            self._cache_budget = proposed
            for role, provider in self._providers.items():
                provider.set_cache_budget(self._role_cache_mb(role))

    def set_component_cache_budgets(self, budgets_mb: Mapping[str, float]) -> None:
        with self._lock:
            normalized = {str(role): max(0.0, float(value)) for role, value in budgets_mb.items()}
            unknown = set(normalized) - set(self.graph.components)
            if unknown:
                raise ComponentGraphError(f"unknown component cache roles: {sorted(unknown)}")
            total = sum(normalized.values())
            for role, provider in self._providers.items():
                role_budget = int(normalized.get(role, 0.0) * 1e6)
                pinned = int(getattr(provider.store, "fp32_aux_cache_bytes", 0))
                if role_budget < pinned:
                    raise MemoryError(
                        f"{role} cache budget cannot retain pinned FP32 auxiliaries "
                        f"({role_budget} < {pinned} bytes)"
                    )
            self._cache_budget = int(total * 1e6)
            self._cache_weights = (
                {role: value / total for role, value in normalized.items()}
                if total
                else {"body": 1.0}
            )
            for role, provider in self._providers.items():
                provider.set_cache_budget(self._role_cache_mb(role))

    def prepare_fully_resident_components(self) -> None:
        """Open and materialize every verified component provider on its target device.

        Native CUDA bindings cannot rely on a first forward to populate an LRU: placement
        claims full residency before execution starts.  This method therefore performs an
        aggregate preflight, opens every physical role, invokes each provider's hard residency
        operation, and validates the complete postcondition.  Partial budgets fail closed.
        """

        with self._lock:
            if self.provider_backend != "dense-qstore-cuda":
                raise ComponentGraphError(
                    "complete component residency requires dense-qstore-cuda providers"
                )
            role_required: dict[str, int] = {}
            role_budget: dict[str, int] = {}
            role_current: dict[str, int] = {}
            for role, manifest in self.graph._component_manifests.items():  # noqa: SLF001
                blocks = manifest["blocks"]
                physical = {_resolve_alias(blocks, str(name)) for name in blocks}
                required = sum(
                    int(blocks[name].get("w_len", 0))
                    + int(blocks[name].get("s_len", 0))
                    + int(blocks[name].get("e_len", 0))
                    for name in physical
                )
                budget = int(self._role_cache_mb(role) * 1e6)
                if required > budget:
                    raise MemoryError(
                        f"{role} complete residency exceeds its provider budget "
                        f"({required} > {budget} bytes)"
                    )
                provider = self._providers.get(role)
                current = 0
                if provider is not None:
                    snapshot = provider.snapshot()
                    current = int(snapshot["resident_physical_bytes"])
                role_required[role] = required
                role_budget[role] = budget
                role_current[role] = current

            future_bytes = sum(
                max(0, role_required[role] - role_current[role]) for role in role_required
            )
            body_device = torch.device(self.device)
            free_before: int | None = None
            if body_device.type == "cuda" and future_bytes:
                free_before, _ = torch.cuda.mem_get_info(body_device)
                if future_bytes > int(free_before):
                    raise MemoryError(
                        "complete component residency exceeds currently free device memory "
                        f"({future_bytes} > {int(free_before)} bytes)"
                    )

            self._fully_resident_admission = {
                "accepted": True,
                "component_role_required_bytes": role_required,
                "component_role_budget_bytes": role_budget,
                "component_role_current_resident_bytes": role_current,
                "future_component_resident_bytes": future_bytes,
                "cuda_free_before_bytes": free_before,
            }
            for role in sorted(self.graph.components):
                provider = self._provider(role)
                prepare = getattr(provider.store, "prepare_fully_resident", None)
                if not callable(prepare):
                    raise ComponentGraphError(
                        f"{role} provider cannot prepare complete device residency"
                    )
                prepare()

            snapshot = self.snapshot()
            providers = snapshot["providers"]
            nonresident = sorted(
                role
                for role in self.graph.components
                if role not in providers or not bool(providers[role].get("fully_resident"))
            )
            if nonresident:
                raise ComponentGraphError(
                    f"complete component residency postcondition failed: {nonresident!r}"
                )

    def prepare_resident_exact_head(self, max_resident_mb: float) -> torch.Tensor:
        with self._lock:
            max_resident_bytes = int(float(max_resident_mb) * 1e6)
            if max_resident_bytes <= 0:
                raise ValueError("resident exact-head budget must be positive")
            head = self.blocks[_resolve_alias(self.blocks, "lm_head")]
            if head.get("kind") != "qrow":
                raise ComponentGraphError("resident exact head requires a qrow lm_head")
            head_rows, head_columns = (int(value) for value in head["shape"])
            head_required_bytes = head_rows * head_columns * 4
            if head_required_bytes > max_resident_bytes:
                raise MemoryError(
                    "resident exact FP32 head exceeds its admission budget "
                    f"({head_required_bytes} > {max_resident_bytes} bytes)"
                )

            role_required: dict[str, int] = {}
            role_target: dict[str, int] = {}
            role_current: dict[str, int] = {}
            for role, manifest in self.graph._component_manifests.items():  # noqa: SLF001
                blocks = manifest["blocks"]
                physical = {_resolve_alias(blocks, str(name)) for name in blocks}
                required = sum(
                    int(blocks[name].get("w_len", 0))
                    + int(blocks[name].get("s_len", 0))
                    + int(blocks[name].get("e_len", 0))
                    for name in physical
                )
                budget = int(self._role_cache_mb(role) * 1e6)
                role_required[role] = required
                role_target[role] = min(required, budget)
                provider = self._providers.get(role)
                role_current[role] = (
                    0
                    if provider is None
                    else int(getattr(provider.store, "compact_cache_bytes", 0))
                    + int(getattr(provider.store, "fp32_aux_cache_bytes", 0))
                )
            future_component_bytes = sum(
                max(0, role_target[role] - role_current[role]) for role in role_target
            )
            body_device = torch.device(self.device)
            free_before: int | None = None
            if body_device.type == "cuda":
                free_before, _ = torch.cuda.mem_get_info(body_device)
                projected_increment = future_component_bytes + head_required_bytes
                if projected_increment > int(free_before):
                    raise MemoryError(
                        "resident component plus exact FP32 head exceeds currently free "
                        f"device memory ({projected_increment} > {int(free_before)} bytes)"
                    )
            self._resident_exact_head_admission = {
                "accepted": True,
                "head_required_bytes": head_required_bytes,
                "head_budget_bytes": max_resident_bytes,
                "component_role_required_bytes": role_required,
                "component_role_target_resident_bytes": role_target,
                "component_role_current_resident_bytes": role_current,
                "future_component_resident_bytes": future_component_bytes,
                "projected_incremental_device_bytes": (
                    future_component_bytes + head_required_bytes
                ),
                "cuda_free_before_bytes": free_before,
                "all_component_roles_fully_admitted": all(
                    role_target[role] == role_required[role] for role in role_required
                ),
            }
            if all(role_target[role] == role_required[role] for role in role_required):
                self.prepare_fully_resident_components()
            view = self.for_contract("full_logits")
            provider = view._provider(  # noqa: SLF001 - manager performs admitted route setup
                "row_blocks",
                "lm_head",
            )
            return provider.prepare_resident_exact_head(
                "lm_head",
                max_resident_bytes=max_resident_bytes,
            )

    def ring_stats(self) -> dict[str, int] | None:
        with self._lock:
            return self._providers["body"].store.ring_stats()

    def ring_allocated_bytes(self) -> int:
        with self._lock:
            return sum(
                provider.store.ring_allocated_bytes() for provider in self._providers.values()
            )

    def disable_ring(self) -> None:
        with self._lock:
            for provider in self._providers.values():
                provider.store.disable_ring()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for provider in self._providers.values():
                provider.store.close()
            self._providers.clear()
            self._views.clear()

    def has(self, name: str) -> bool:
        return self._default_view.has(name)

    def fp32(self, name: str) -> torch.Tensor:
        return self._default_view.fp32(name)

    def weight(self, name: str) -> torch.Tensor:
        return self._default_view.weight(name)

    def matmul(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return self._default_view.matmul(name, value)

    def matmul_row_stable(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return self._default_view.matmul_row_stable(name, value)

    def embed_rows(
        self,
        name: str,
        ids: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        return self._default_view.embed_rows(name, ids)

    def row_blocks(self, name: str, bs: int = 8192):
        yield from self._default_view.row_blocks(name, bs=bs)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            providers = {
                role: provider.snapshot() for role, provider in sorted(self._providers.items())
            }
            body = providers.get("body", {})
            body_fully_resident = bool(body.get("fully_resident", False))
            head_provider = providers.get(self.graph.routes.get("lm_head", ""), {})
            resident_exact_head = bool(head_provider.get("resident_exact_head", False))
            return {
                "schema": self.graph.schema,
                "component_graph_fingerprint": self.graph.fingerprint,
                "declared_legacy_graph_fingerprint": self.graph.declared_fingerprint,
                "vocab_manifest_sha256": self.vocab.semantic_sha256,
                "identity_status": self.identity_status,
                "content_identity_verified": False,
                "provider_backend": self.provider_backend,
                "residency_contract": (
                    "resident-body+resident-exact-fp32-head"
                    if self.provider_backend == "dense-qstore-cuda"
                    and body_fully_resident
                    and resident_exact_head
                    else "resident-body+streamed-exact-head"
                    if self.provider_backend == "dense-qstore-cuda" and body_fully_resident
                    else "component-compact-lru+resident-exact-fp32-head"
                    if self.provider_backend == "dense-qstore-cuda" and resident_exact_head
                    else "component-compact-lru+streamed-exact-head"
                    if self.provider_backend == "dense-qstore-cuda"
                    else "component-paged-qstore"
                ),
                "residency_guarantee": (
                    "measured-complete-body-residency"
                    if self.provider_backend == "dense-qstore-cuda" and body_fully_resident
                    else "byte-bounded-lru; body pages may stream and FP32 auxiliaries are on-demand"
                    if self.provider_backend == "dense-qstore-cuda"
                    else "configured-cache-policy"
                ),
                "body_fully_resident": body_fully_resident,
                "resident_exact_head": resident_exact_head,
                "resident_exact_head_bytes": int(head_provider.get("resident_exact_head_bytes", 0)),
                "resident_exact_head_admission": _json_clone(self._resident_exact_head_admission),
                "fully_resident_admission": _json_clone(self._fully_resident_admission),
                "body_required_physical_bytes": int(body.get("required_physical_bytes", 0)),
                "body_resident_physical_bytes": int(body.get("resident_physical_bytes", 0)),
                "opened_roles": sorted(self.opened_roles),
                "cache_budget_bytes": self._cache_budget,
                "component_cache_budget_bytes": {
                    role: int(self._role_cache_mb(role) * 1e6)
                    for role in sorted(self.graph.components)
                },
                "providers": providers,
                "views": {name: view.snapshot() for name, view in sorted(self._views.items())},
            }
