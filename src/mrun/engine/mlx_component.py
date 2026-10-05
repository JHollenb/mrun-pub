"""Native MLX execution for verified disassembled QStore component graphs.

The established :mod:`mrun.engine.kernels.composite_qstore` runtime deliberately exposes
PyTorch tensors because it feeds the CPU/CUDA Paged kernels.  This module is the independent
Apple lane: it verifies the same graph and component bytes with the standard library and
NumPy, repacks the existing symmetric row-int8 codes directly into MLX's affine q8 format,
and lets ``mlx-lm`` execute its normal fused Metal model and KV-cache implementation.

No weight passes through PyTorch.  The q8 conversion is lossless: signed QStore code ``q`` is
stored as unsigned ``q + 128`` with affine bias ``-128 * scale``.  Repeating the original
per-row scale over 64-column groups therefore makes MLX dequantize every value to the exact
QStore weight ``q * scale``.

The derived artifact is content addressed and role separated.  ``mlx-lm`` loads every
``model-*.safetensors`` shard, so body/norm/lexical physical ownership remains observable while
the execution path stays native.  A complete source ``config.json`` is required because the
legacy component graph intentionally omitted context-sensitive fields such as
``max_position_embeddings`` and ``rope_scaling``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..models import load_tokenizer, resolve_model, snapshot_dir

COMPONENT_GRAPH_SCHEMAS = frozenset(
    {"mrun-component-graph-v1", "mrun.disassembled-model-graph.poc.v2"}
)
COMPONENT_PAYLOAD_FILES = ("weights.i8", "scales.f32", "extras.f32")
COMPOSITE_CUSTODY_SCHEMA = "mrun-composite-custody-v1"
MLX_COMPONENT_NATIVE_SCHEMA = "mrun-mlx-component-native-v1"
MLX_COMPONENT_CODEC = "mlx-affine-int8-g64-exact-qstore-v1"
MLX_COMPONENT_Q4_NATIVE_SCHEMA = "mrun-mlx-component-q4-native-v1"
MLX_COMPONENT_Q4_CODEC = "mlx-int4-g64-bf16-full-model-requantized-from-qstore-v1"
MLX_COMPONENT_MAPPING_ABI = "mrun-mlx-component-qwen-llama-mapping-v1"
MLX_COMPONENT_GROUP_SIZE = 64
MLX_COMPONENT_BITS = 8
MLX_COMPONENT_MODE = "affine"


class MLXComponentError(RuntimeError):
    """Base error for native component verification, conversion, or execution."""


class MLXComponentGraphError(MLXComponentError):
    """The source component graph or one of its authenticated payloads is invalid."""


class MLXComponentMappingError(MLXComponentError):
    """A canonical logical block has no exact MLX parameter mapping."""


class MLXComponentArtifactError(MLXComponentError):
    """A derived native artifact is incomplete, corrupted, or belongs to another graph."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MLXComponentGraphError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MLXComponentGraphError("component metadata is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _require_sha256(value: Any, field: str) -> str:
    if not _is_sha256(value):
        raise MLXComponentGraphError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_regular_json(path: Path) -> tuple[dict[str, Any], tuple[int, ...], str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MLXComponentGraphError(f"cannot open component JSON {path}") from exc
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise MLXComponentGraphError(f"component JSON must be a regular file: {path}")
        chunks: list[bytes] = []
        while value := os.read(descriptor, 1024 * 1024):
            chunks.append(value)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(initial) != _identity(final):
        raise MLXComponentGraphError(f"component JSON changed while being read: {path}")
    try:
        current = path.lstat()
    except OSError as exc:
        raise MLXComponentGraphError(f"component JSON disappeared: {path}") from exc
    if _identity(current) != _identity(final) or not stat.S_ISREG(current.st_mode):
        raise MLXComponentGraphError(f"component JSON changed while being read: {path}")
    payload = b"".join(chunks)
    try:
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MLXComponentGraphError(f"cannot parse component JSON {path}") from exc
    if not isinstance(parsed, dict):
        raise MLXComponentGraphError(f"component JSON must contain an object: {path}")
    return parsed, _identity(final), _sha256_bytes(payload)


def _open_verified_blob(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> tuple[int, tuple[int, ...]]:
    if path.is_symlink():
        raise MLXComponentGraphError(f"component payload cannot be a symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MLXComponentGraphError(f"cannot open component payload {path}") from exc
    digest = hashlib.sha256()
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise MLXComponentGraphError(f"component payload must be regular: {path}")
        if int(initial.st_size) != int(expected_bytes):
            raise MLXComponentGraphError(f"component payload size mismatch: {path}")
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        final = os.fstat(descriptor)
        if _identity(initial) != _identity(final):
            raise MLXComponentGraphError(f"component payload changed while hashing: {path}")
        if digest.hexdigest() != expected_sha256:
            raise MLXComponentGraphError(f"component payload hash mismatch: {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        current = path.lstat()
        if _identity(current) != _identity(final):
            raise MLXComponentGraphError(f"component payload changed while opening: {path}")
        return descriptor, _identity(final)
    except BaseException:
        os.close(descriptor)
        raise


def _pread_exact(descriptor: int, offset: int, length: int, *, field: str) -> bytes:
    if isinstance(offset, bool) or isinstance(length, bool) or offset < 0 or length < 0:
        raise MLXComponentGraphError(f"{field} has a negative payload span")
    chunks: list[bytes] = []
    remaining = length
    position = offset
    while remaining:
        value = os.pread(descriptor, min(remaining, 8 * 1024 * 1024), position)
        if not value:
            raise MLXComponentGraphError(f"{field} payload span exceeds its file")
        chunks.append(value)
        remaining -= len(value)
        position += len(value)
    return b"".join(chunks)


def _resolve_alias(blocks: Mapping[str, Mapping[str, Any]], name: str) -> str:
    current = str(name)
    seen: set[str] = set()
    while True:
        if current in seen:
            raise MLXComponentGraphError(f"cyclic component alias at {name!r}")
        seen.add(current)
        block = blocks.get(current)
        if not isinstance(block, Mapping):
            raise MLXComponentGraphError(f"component alias target {current!r} is absent")
        target = block.get("alias")
        if target is None:
            return current
        if not isinstance(target, str) or not target:
            raise MLXComponentGraphError(f"component alias {current!r} is invalid")
        current = target


def _semantic_block_descriptor(block: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    alias = block.get("alias")
    if alias is not None:
        if not isinstance(alias, str) or not alias:
            raise MLXComponentGraphError(f"block {name!r} has an invalid alias")
        return {"alias": alias}
    kind = block.get("kind")
    shape = block.get("shape")
    if kind not in {"qrow", "fp32"}:
        raise MLXComponentGraphError(f"block {name!r} has unsupported kind {kind!r}")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
        )
    ):
        raise MLXComponentGraphError(f"block {name!r} has an invalid shape")
    return {"kind": str(kind), "shape": [int(value) for value in shape]}


def _graph_fingerprint_payload(graph: Mapping[str, Any]) -> dict[str, Any]:
    components = graph.get("components")
    if not isinstance(components, Mapping):
        raise MLXComponentGraphError("component graph components must be an object")
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
    components = graph["components"]
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


def _logical_role(name: str) -> str:
    if name == "embed":
        return "ingress"
    if name == "lm_head":
        return "egress"
    if name in {"norm.final", "norm.final.bias"}:
        return "norm"
    return "body"


def _expected_topology(
    blocks: Mapping[str, Mapping[str, Any]], declared_tied: bool
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    for required in ("embed", "lm_head", "norm.final"):
        if required not in blocks:
            raise MLXComponentGraphError(f"required logical block {required!r} is absent")
    by_root: dict[str, list[str]] = {}
    for name in blocks:
        by_root.setdefault(_resolve_alias(blocks, name), []).append(name)
    root_roles: dict[str, str] = {}
    for root, names in by_root.items():
        roles = {_logical_role(name) for name in names}
        if roles == {"ingress", "egress"} and {"embed", "lm_head"} <= set(names):
            root_roles[root] = "lexical_shared"
        elif len(roles) == 1:
            root_roles[root] = next(iter(roles))
        else:
            raise MLXComponentGraphError(f"unsupported cross-role alias class {sorted(names)!r}")
    tied = _resolve_alias(blocks, "embed") == _resolve_alias(blocks, "lm_head")
    if tied != bool(declared_tied):
        raise MLXComponentGraphError("tie declaration disagrees with alias topology")
    role_names: dict[str, list[str]] = {}
    for name in blocks:
        role = root_roles[_resolve_alias(blocks, name)]
        role_names.setdefault(role, []).append(name)
    role_names = {role: sorted(names) for role, names in sorted(role_names.items())}
    expected_roles = (
        {"body", "norm", "lexical_shared"}
        if tied
        else {
            "body",
            "norm",
            "ingress",
            "egress",
        }
    )
    if set(role_names) != expected_roles:
        raise MLXComponentGraphError("component graph has unexpected physical roles")
    topology = {
        "declared_tied": bool(declared_tied),
        "observed_tied": tied,
        "embed_physical_root": _resolve_alias(blocks, "embed"),
        "lm_head_physical_root": _resolve_alias(blocks, "lm_head"),
        "aliases": {
            name: str(block["alias"]) for name, block in sorted(blocks.items()) if "alias" in block
        },
        "alias_classes": {
            root: sorted(names) for root, names in sorted(by_root.items()) if len(names) > 1
        },
        "physical_roles": sorted(role_names),
    }
    return role_names, topology


def _expected_operation_contracts(role_names: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    tied = "lexical_shared" in role_names
    input_role = "lexical_shared" if tied else "ingress"
    base = ["body", "norm", input_role]
    full = base if tied else [*base, "egress"]
    return {
        "full_logits": {"required_components": sorted(full), "lm_head_methods": ["row_blocks"]},
        "lexical_hidden": {"required_components": sorted(base), "lm_head_methods": []},
        "selected_rows": {"required_components": sorted(full), "lm_head_methods": ["embed_rows"]},
    }


def _validate_logical_layout(blocks: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate the virtual monolithic spans retained for admission/accounting."""

    intervals: dict[str, list[tuple[int, int, str]]] = {
        filename: [] for filename in COMPONENT_PAYLOAD_FILES
    }
    for name, block in blocks.items():
        descriptor = _semantic_block_descriptor(block, name=name)
        if "alias" in descriptor:
            continue
        shape = descriptor["shape"]
        if descriptor["kind"] == "qrow":
            if len(shape) != 2:
                raise MLXComponentGraphError(f"logical qrow block {name!r} is not a matrix")
            rows, columns = shape
            expected = (
                ("weights.i8", "w_off", "w_len", rows * columns, 1),
                ("scales.f32", "s_off", "s_len", rows * 4, 4),
            )
        else:
            expected = (("extras.f32", "e_off", "e_len", math.prod(shape) * 4, 4),)
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
                raise MLXComponentGraphError(
                    f"logical block {name!r} has an invalid {filename} span"
                )
            intervals[filename].append((offset, offset + length, name))
    for name in blocks:
        _resolve_alias(blocks, name)
    for filename, ranges in intervals.items():
        previous_end = 0
        previous_name: str | None = None
        for start, end, name in sorted(ranges):
            if start < previous_end:
                raise MLXComponentGraphError(
                    f"logical {filename} span for {name!r} overlaps {previous_name!r}"
                )
            previous_end = end
            previous_name = name


@dataclass(frozen=True)
class PackedAffineQ8:
    """MLX affine-q8 parameters that exactly reconstruct one QStore qrow matrix."""

    weight: np.ndarray
    scales: np.ndarray
    biases: np.ndarray
    rows: int
    columns: int
    group_size: int = MLX_COMPONENT_GROUP_SIZE
    bits: int = MLX_COMPONENT_BITS
    mode: str = MLX_COMPONENT_MODE


def pack_symmetric_qrow_int8(
    codes: np.ndarray,
    row_scales: np.ndarray,
    *,
    group_size: int = MLX_COMPONENT_GROUP_SIZE,
) -> PackedAffineQ8:
    """Losslessly translate QStore signed row-int8 into MLX affine q8 groups.

    This function is intentionally NumPy-only so conversion never widens a complete model, or
    even one matrix, through Torch.  MLX packs four 8-bit values into each little-endian uint32.
    """

    values = np.asarray(codes)
    scales = np.asarray(row_scales)
    if values.dtype != np.int8 or values.ndim != 2:
        raise TypeError("QStore qrow codes must be a two-dimensional int8 array")
    rows, columns = (int(value) for value in values.shape)
    if group_size != MLX_COMPONENT_GROUP_SIZE:
        raise ValueError(f"native component codec requires group_size={MLX_COMPONENT_GROUP_SIZE}")
    if columns % group_size or columns % 4:
        raise ValueError("QStore qrow width must be divisible by the MLX group and pack widths")
    if scales.shape not in {(rows,), (rows, 1)}:
        raise ValueError("QStore qrow scales must contain exactly one value per output row")
    scales = np.asarray(scales, dtype=np.float32).reshape(rows)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("QStore qrow scales must be finite and positive")

    # Adding 128 modulo 256 is exactly a sign-bit flip.  Doing it in uint8 avoids the
    # temporary int16 matrix that would otherwise double peak memory for the vocabulary
    # embedding (the largest single tensor in these models).
    unsigned = np.bitwise_xor(values.view(np.uint8), np.uint8(0x80))
    packed = np.ascontiguousarray(unsigned).view(np.dtype("<u4")).reshape(rows, columns // 4)
    group_scales = np.repeat(scales[:, None], columns // group_size, axis=1)
    group_biases = np.asarray(-128.0 * group_scales, dtype=np.float32)
    return PackedAffineQ8(
        weight=packed.astype(np.uint32, copy=False),
        scales=np.array(group_scales, dtype=np.float32, copy=True, order="C"),
        biases=np.array(group_biases, dtype=np.float32, copy=True, order="C"),
        rows=rows,
        columns=columns,
    )


def dequantize_affine_q8_numpy(packed: PackedAffineQ8) -> np.ndarray:
    """Reference inverse used by pure unit tests and conversion audits."""

    unsigned = (
        np.ascontiguousarray(packed.weight).view(np.uint8).reshape(packed.rows, packed.columns)
    )
    groups = unsigned.reshape(packed.rows, -1, packed.group_size).astype(np.float32)
    restored = groups * packed.scales[..., None] + packed.biases[..., None]
    return restored.reshape(packed.rows, packed.columns)


@dataclass(frozen=True)
class MLXParameterTarget:
    logical_name: str
    kind: Literal["qrow", "fp32"]
    path: str


_LAYER_BLOCK_RE = re.compile(r"^L(?P<layer>[0-9]+)\.(?P<tail>[A-Za-z0-9_.]+)$")


def canonical_mlx_target(logical_name: str, architecture: str) -> MLXParameterTarget:
    """Map every canonical dense Qwen/Llama component block to an mlx-lm parameter."""

    architecture = str(architecture)
    if architecture not in {"qwen2", "qwen3", "llama"}:
        raise MLXComponentMappingError(
            f"native MLX components support qwen2/qwen3/llama, got {architecture!r}"
        )
    fixed = {
        "embed": MLXParameterTarget("embed", "qrow", "model.embed_tokens"),
        "lm_head": MLXParameterTarget("lm_head", "qrow", "lm_head"),
        "norm.final": MLXParameterTarget("norm.final", "fp32", "model.norm.weight"),
        "norm.final.bias": MLXParameterTarget("norm.final.bias", "fp32", "model.norm.bias"),
    }
    if logical_name in fixed:
        return fixed[logical_name]
    match = _LAYER_BLOCK_RE.fullmatch(str(logical_name))
    if match is None:
        raise MLXComponentMappingError(f"unsupported canonical block {logical_name!r}")
    layer = int(match.group("layer"))
    tail = match.group("tail")
    prefix = f"model.layers.{layer}"
    qrow_paths = {
        "q": f"{prefix}.self_attn.q_proj",
        "k": f"{prefix}.self_attn.k_proj",
        "v": f"{prefix}.self_attn.v_proj",
        "o": f"{prefix}.self_attn.o_proj",
        "gate": f"{prefix}.mlp.gate_proj",
        "up": f"{prefix}.mlp.up_proj",
        "down": f"{prefix}.mlp.down_proj",
    }
    fp32_paths = {
        "q.bias": f"{prefix}.self_attn.q_proj.bias",
        "k.bias": f"{prefix}.self_attn.k_proj.bias",
        "v.bias": f"{prefix}.self_attn.v_proj.bias",
        "o.bias": f"{prefix}.self_attn.o_proj.bias",
        "gate.bias": f"{prefix}.mlp.gate_proj.bias",
        "up.bias": f"{prefix}.mlp.up_proj.bias",
        "down.bias": f"{prefix}.mlp.down_proj.bias",
        "ln1": f"{prefix}.input_layernorm.weight",
        "ln1.bias": f"{prefix}.input_layernorm.bias",
        "ln2": f"{prefix}.post_attention_layernorm.weight",
        "ln2.bias": f"{prefix}.post_attention_layernorm.bias",
        "q_norm": f"{prefix}.self_attn.q_norm.weight",
        "k_norm": f"{prefix}.self_attn.k_norm.weight",
    }
    if tail in {"q_norm", "k_norm"} and architecture != "qwen3":
        raise MLXComponentMappingError(f"canonical block {logical_name!r} is only valid for qwen3")
    if tail in qrow_paths:
        return MLXParameterTarget(logical_name, "qrow", qrow_paths[tail])
    if tail in fp32_paths:
        return MLXParameterTarget(logical_name, "fp32", fp32_paths[tail])
    raise MLXComponentMappingError(f"unsupported canonical block {logical_name!r}")


@dataclass
class _OpenedComponent:
    role: str
    directory: Path
    manifest: dict[str, Any]
    manifest_identity: tuple[int, ...]
    descriptors: dict[str, int]
    blob_identities: dict[str, tuple[int, ...]]

    def close(self) -> None:
        for descriptor in self.descriptors.values():
            os.close(descriptor)
        self.descriptors.clear()


class VerifiedComponentGraphReader:
    """Torch-free, fail-closed reader over authenticated component qrow/fp32 spans."""

    def __init__(self, graph_path: str | Path) -> None:
        self.path = Path(graph_path).expanduser().absolute()
        self.root = self.path.parent.resolve()
        self.raw, self._graph_identity, _graph_file_sha = _read_regular_json(self.path)
        self._components: dict[str, _OpenedComponent] = {}
        self._closed = False
        try:
            self._validate_and_open()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> VerifiedComponentGraphReader:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def _validate_and_open(self) -> None:
        graph = self.raw
        self.schema = str(graph.get("schema", ""))
        if self.schema not in COMPONENT_GRAPH_SCHEMAS:
            raise MLXComponentGraphError(f"unsupported component graph schema {self.schema!r}")
        required_objects = (
            "body_abi",
            "tokenizer",
            "logical_blocks",
            "components",
            "routes",
            "operation_contracts",
            "topology",
        )
        if any(not isinstance(graph.get(field), Mapping) for field in required_objects):
            raise MLXComponentGraphError("component graph has a malformed object field")
        self.model_name = str(graph.get("model", ""))
        self.architecture = str(graph.get("architecture", ""))
        if not self.model_name or self.architecture not in {"qwen2", "qwen3", "llama"}:
            raise MLXComponentGraphError("native graph model or architecture is unsupported")
        declared = _require_sha256(
            graph.get("composite_fingerprint_sha256"), "component graph fingerprint"
        )
        observed = _sha256_bytes(_canonical_json_bytes(_graph_fingerprint_payload(graph)))
        if declared != observed:
            raise MLXComponentGraphError("component graph fingerprint mismatch")

        logical_raw = graph["logical_blocks"]
        self.logical_blocks = {
            str(name): dict(block)
            for name, block in logical_raw.items()
            if isinstance(name, str) and isinstance(block, Mapping)
        }
        if len(self.logical_blocks) != len(logical_raw):
            raise MLXComponentGraphError("logical block table contains an invalid entry")
        for name, block in self.logical_blocks.items():
            _semantic_block_descriptor(block, name=name)
            _resolve_alias(self.logical_blocks, name)
        _validate_logical_layout(self.logical_blocks)

        topology = graph["topology"]
        role_names, expected_topology = _expected_topology(
            self.logical_blocks, bool(topology.get("declared_tied", False))
        )
        if _canonical_json_bytes(topology) != _canonical_json_bytes(expected_topology):
            raise MLXComponentGraphError("stored topology differs from logical alias topology")
        self.topology = expected_topology
        expected_contracts = _expected_operation_contracts(role_names)
        if _canonical_json_bytes(graph["operation_contracts"]) != _canonical_json_bytes(
            expected_contracts
        ):
            raise MLXComponentGraphError("component operation contracts are invalid")

        components = graph["components"]
        owned: dict[str, str] = {}
        for raw_role, raw_record in components.items():
            role = str(raw_role)
            if not isinstance(raw_record, Mapping) or str(raw_record.get("role")) != role:
                raise MLXComponentGraphError(f"component role record {role!r} is invalid")
            allowed = raw_record.get("allowed_names")
            if (
                not isinstance(allowed, list)
                or not allowed
                or any(not isinstance(name, str) or not name for name in allowed)
                or len(set(allowed)) != len(allowed)
            ):
                raise MLXComponentGraphError(f"component {role!r} allowed_names are invalid")
            for name in allowed:
                if name in owned:
                    raise MLXComponentGraphError(f"logical block {name!r} has multiple owners")
                owned[name] = role
            if sorted(allowed) != role_names.get(role):
                raise MLXComponentGraphError(f"component {role!r} ownership differs from topology")
            blobs = raw_record.get("blobs")
            if not isinstance(blobs, Mapping) or set(blobs) != set(COMPONENT_PAYLOAD_FILES):
                raise MLXComponentGraphError(f"component {role!r} blob table is invalid")
            component_dir = self._component_path(role, raw_record)
            manifest_path = component_dir / "manifest.json"
            manifest, manifest_identity, manifest_sha = _read_regular_json(manifest_path)
            if manifest_sha != _require_sha256(
                raw_record.get("manifest_sha256"), f"{role} manifest hash"
            ):
                raise MLXComponentGraphError(f"component {role!r} manifest hash mismatch")
            if manifest.get("dtype") != "int8":
                raise MLXComponentGraphError(f"component {role!r} is not an int8 store")
            if str(manifest.get("arch", self.architecture)) != self.architecture:
                raise MLXComponentGraphError(f"component {role!r} architecture mismatch")
            if bool(manifest.get("tie_word_embeddings", False)) != bool(
                expected_topology["observed_tied"]
            ):
                raise MLXComponentGraphError(
                    f"component {role!r} tie declaration differs from graph topology"
                )
            blocks = manifest.get("blocks")
            if not isinstance(blocks, Mapping) or set(blocks) != set(allowed):
                raise MLXComponentGraphError(f"component {role!r} block ownership mismatch")
            for name in allowed:
                physical = blocks[name]
                if not isinstance(physical, Mapping):
                    raise MLXComponentGraphError(f"component block {name!r} is invalid")
                if _semantic_block_descriptor(physical, name=name) != _semantic_block_descriptor(
                    self.logical_blocks[name], name=name
                ):
                    raise MLXComponentGraphError(
                        f"component block {name!r} differs from the logical descriptor"
                    )
            self._validate_component_layout(component_dir, blocks)
            descriptors: dict[str, int] = {}
            identities: dict[str, tuple[int, ...]] = {}
            try:
                for filename in COMPONENT_PAYLOAD_FILES:
                    blob = blobs[filename]
                    if not isinstance(blob, Mapping):
                        raise MLXComponentGraphError(f"component blob {role}/{filename} is invalid")
                    descriptor, identity = _open_verified_blob(
                        component_dir / filename,
                        expected_bytes=int(blob.get("bytes", -1)),
                        expected_sha256=_require_sha256(
                            blob.get("sha256"), f"{role}/{filename} sha256"
                        ),
                    )
                    descriptors[filename] = descriptor
                    identities[filename] = identity
                semantic = self._component_semantic_digest(blocks, allowed, descriptors)
                if semantic != _require_sha256(
                    raw_record.get("semantic_content_sha256"),
                    f"{role} semantic content hash",
                ):
                    raise MLXComponentGraphError(f"component {role!r} semantic hash mismatch")
            except BaseException:
                for descriptor in descriptors.values():
                    os.close(descriptor)
                raise
            self._components[role] = _OpenedComponent(
                role=role,
                directory=component_dir,
                manifest=manifest,
                manifest_identity=manifest_identity,
                descriptors=descriptors,
                blob_identities=identities,
            )

        routes = {str(name): str(role) for name, role in graph["routes"].items()}
        if routes != dict(sorted(owned.items())) or set(routes) != set(self.logical_blocks):
            raise MLXComponentGraphError("component routes do not cover exact ownership")
        self.routes = routes
        configs = [component.manifest.get("config") for component in self._components.values()]
        if not configs or any(not isinstance(config, Mapping) for config in configs):
            raise MLXComponentGraphError("component graph is missing runtime config")
        first_config = dict(configs[0])
        if any(
            _canonical_json_bytes(config) != _canonical_json_bytes(first_config)
            for config in configs
        ):
            raise MLXComponentGraphError("component runtime configs differ")
        self.config = first_config
        self._validate_body_abi()
        self._validate_tokenizer_descriptor()
        self._validate_lexical_shapes()
        self.custody_fingerprint_sha256 = _sha256_bytes(
            _canonical_json_bytes(_custody_fingerprint_payload(graph))
        )
        self.declared_fingerprint_sha256 = declared

    def _component_path(self, role: str, record: Mapping[str, Any]) -> Path:
        relative_value = record.get("relative_path")
        if not isinstance(relative_value, str) or not relative_value:
            raise MLXComponentGraphError(f"component {role!r} has no relative path")
        relative = Path(relative_value)
        if relative.is_absolute() or any(part == ".." for part in relative.parts):
            raise MLXComponentGraphError("component path must be relative and in-root")
        current = self.root
        for part in relative.parts:
            if part in {"", "."}:
                continue
            current /= part
            if current.is_symlink():
                raise MLXComponentGraphError("component path cannot traverse a symlink")
        resolved = current.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise MLXComponentGraphError("component path escapes graph root") from exc
        if not resolved.is_dir():
            raise MLXComponentGraphError(f"component directory is absent: {role!r}")
        return resolved

    @staticmethod
    def _validate_component_layout(component_dir: Path, blocks: Mapping[str, Any]) -> None:
        intervals: dict[str, list[tuple[int, int, str]]] = {
            filename: [] for filename in COMPONENT_PAYLOAD_FILES
        }
        for name, raw_block in blocks.items():
            block = dict(raw_block)
            descriptor = _semantic_block_descriptor(block, name=str(name))
            if "alias" in descriptor:
                _resolve_alias(blocks, str(name))
                continue
            shape = descriptor["shape"]
            if descriptor["kind"] == "qrow":
                if len(shape) != 2:
                    raise MLXComponentGraphError(f"qrow block {name!r} is not a matrix")
                rows, columns = shape
                expected = (
                    ("weights.i8", "w_off", "w_len", rows * columns, 1),
                    ("scales.f32", "s_off", "s_len", rows * 4, 4),
                )
            else:
                elements = math.prod(shape)
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
                    raise MLXComponentGraphError(
                        f"component block {name!r} has an invalid {filename} span"
                    )
                intervals[filename].append((offset, offset + length, str(name)))
        for filename, ranges in intervals.items():
            path = component_dir / filename
            if path.is_symlink() or not path.is_file():
                raise MLXComponentGraphError(f"component payload is not regular: {path}")
            previous = 0
            for start, end, name in sorted(ranges):
                if start != previous:
                    raise MLXComponentGraphError(
                        f"component {filename} has a gap or overlap before {name!r}"
                    )
                previous = end
            size = int(path.stat().st_size)
            if previous > size:
                raise MLXComponentGraphError(f"component {filename} spans exceed the file")
            # Never materialize a complete model payload merely to inspect its usually tiny
            # alignment tail.  This stays bounded even for multi-gigabyte components.
            with path.open("rb") as handle:
                handle.seek(previous)
                while tail := handle.read(8 * 1024 * 1024):
                    if any(tail):
                        raise MLXComponentGraphError(
                            f"component {filename} has non-zero tail padding"
                        )

    @staticmethod
    def _component_semantic_digest(
        blocks: Mapping[str, Any],
        allowed_names: Sequence[str],
        descriptors: Mapping[str, int],
    ) -> str:
        entries: list[dict[str, str]] = []
        layout = {
            "qrow": (
                ("weights.i8", "w_off", "w_len"),
                ("scales.f32", "s_off", "s_len"),
            ),
            "fp32": (("extras.f32", "e_off", "e_len"),),
        }
        for name in sorted(allowed_names):
            block = blocks[name]
            digest = hashlib.sha256()
            if "alias" in block:
                digest.update(_canonical_json_bytes({"name": name, "alias": block["alias"]}))
            else:
                kind = str(block["kind"])
                digest.update(
                    _canonical_json_bytes(
                        {"name": name, "kind": kind, "shape": list(block["shape"])}
                    )
                )
                for filename, offset_key, length_key in layout[kind]:
                    length = int(block[length_key])
                    value = _pread_exact(
                        descriptors[filename], int(block[offset_key]), length, field=name
                    )
                    digest.update(_canonical_json_bytes({"file_kind": filename, "bytes": length}))
                    digest.update(value)
            entries.append({"name": name, "semantic_block_sha256": digest.hexdigest()})
        return _sha256_bytes(_canonical_json_bytes(entries))

    def _validate_body_abi(self) -> None:
        descriptors = {
            name: _semantic_block_descriptor(block, name=name)
            for name, block in sorted(self.logical_blocks.items())
        }
        observed = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "architecture": self.architecture,
                    "config": self.config,
                    "tie_word_embeddings": bool(self.topology["observed_tied"]),
                    "logical_blocks": descriptors,
                }
            )
        )
        body_abi = self.raw["body_abi"]
        if observed != _require_sha256(body_abi.get("semantic_sha256"), "body ABI hash"):
            raise MLXComponentGraphError("body ABI does not match runtime config and blocks")
        if int(body_abi.get("hidden_size", self.config.get("hidden_size", 0))) != int(
            self.config.get("hidden_size", 0)
        ):
            raise MLXComponentGraphError("body ABI hidden size mismatch")
        if int(body_abi.get("vocab_size", self.config.get("vocab_size", 0))) != int(
            self.config.get("vocab_size", 0)
        ):
            raise MLXComponentGraphError("body ABI vocabulary size mismatch")

    def _validate_tokenizer_descriptor(self) -> None:
        descriptor = self.raw["tokenizer"]
        payload = {
            "class": descriptor.get("class"),
            "length": descriptor.get("length"),
            "ordered_tokens_sha256": descriptor.get("ordered_tokens_sha256"),
            "backend_json_sha256": descriptor.get("backend_json_sha256"),
            "special_token_ids": descriptor.get("special_token_ids"),
            "chat_template_sha256": descriptor.get("chat_template_sha256"),
        }
        if (
            isinstance(payload["length"], bool)
            or not isinstance(payload["length"], int)
            or int(payload["length"]) <= 0
        ):
            raise MLXComponentGraphError("tokenizer descriptor length is invalid")
        for field in (
            "ordered_tokens_sha256",
            "backend_json_sha256",
            "chat_template_sha256",
        ):
            _require_sha256(payload[field], f"tokenizer {field}")
        observed = _sha256_bytes(_canonical_json_bytes(payload))
        if observed != _require_sha256(descriptor.get("semantic_sha256"), "tokenizer hash"):
            raise MLXComponentGraphError("tokenizer semantic hash mismatch")
        self.tokenizer_descriptor = payload
        self.tokenizer_semantic_sha256 = observed
        self.semantic_token_count = int(payload["length"])
        if self.semantic_token_count > int(self.config.get("vocab_size", 0)):
            raise MLXComponentGraphError("semantic token domain exceeds configured rows")

    def _validate_lexical_shapes(self) -> None:
        configured_rows = int(self.config.get("vocab_size", 0))
        hidden = int(self.config.get("hidden_size", 0))
        for name in ("embed", "lm_head"):
            root = _resolve_alias(self.logical_blocks, name)
            block = self.logical_blocks[root]
            if block.get("kind") != "qrow" or list(block.get("shape", ())) != [
                configured_rows,
                hidden,
            ]:
                raise MLXComponentGraphError(
                    f"{name} shape differs from configured vocabulary/hidden dimensions"
                )

    def _physical(self, name: str) -> tuple[_OpenedComponent, Mapping[str, Any], str]:
        if self._closed:
            raise MLXComponentGraphError("component reader is closed")
        root = _resolve_alias(self.logical_blocks, str(name))
        role = self.routes[str(name)]
        root_role = self.routes[root]
        if role != root_role:
            raise MLXComponentGraphError("component alias crosses an unsupported physical role")
        component = self._components[role]
        block = component.manifest["blocks"].get(root)
        if not isinstance(block, Mapping):
            raise MLXComponentGraphError(f"physical block {root!r} is absent")
        return component, block, root

    def qrow(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        component, block, root = self._physical(name)
        if block.get("kind") != "qrow":
            raise MLXComponentGraphError(f"component block {name!r} is not qrow")
        rows, columns = (int(value) for value in block["shape"])
        codes = np.frombuffer(
            _pread_exact(
                component.descriptors["weights.i8"],
                int(block["w_off"]),
                int(block["w_len"]),
                field=root,
            ),
            dtype=np.int8,
        ).reshape(rows, columns)
        scales = np.frombuffer(
            _pread_exact(
                component.descriptors["scales.f32"],
                int(block["s_off"]),
                int(block["s_len"]),
                field=root,
            ),
            dtype="<f4",
        ).reshape(rows)
        return codes, scales

    def fp32(self, name: str) -> np.ndarray:
        component, block, root = self._physical(name)
        if block.get("kind") != "fp32":
            raise MLXComponentGraphError(f"component block {name!r} is not fp32")
        shape = tuple(int(value) for value in block["shape"])
        return (
            np.frombuffer(
                _pread_exact(
                    component.descriptors["extras.f32"],
                    int(block["e_off"]),
                    int(block["e_len"]),
                    field=root,
                ),
                dtype="<f4",
            )
            .reshape(shape)
            .copy()
        )

    def role_names(self, role: str) -> tuple[str, ...]:
        component = self._components.get(str(role))
        if component is None:
            raise MLXComponentGraphError(f"unknown component role {role!r}")
        return tuple(sorted(str(name) for name in component.manifest["blocks"]))

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(self._components))

    def assert_unchanged(self) -> None:
        if self._closed:
            raise MLXComponentGraphError("component reader is closed")
        if _identity(self.path.lstat()) != self._graph_identity:
            raise MLXComponentGraphError("component graph changed after verification")
        for component in self._components.values():
            if (
                _identity((component.directory / "manifest.json").lstat())
                != component.manifest_identity
            ):
                raise MLXComponentGraphError(
                    f"component {component.role!r} manifest changed after verification"
                )
            for filename, descriptor in component.descriptors.items():
                if _identity(os.fstat(descriptor)) != component.blob_identities[filename]:
                    raise MLXComponentGraphError(
                        f"component {component.role!r}/{filename} changed after verification"
                    )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for component in self._components.values():
            component.close()
        self._components.clear()


def _runtime_tokenizer_descriptor(tokenizer: Any) -> dict[str, Any]:
    ordered = hashlib.sha256()
    for token_id in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        encoded = ("" if token is None else str(token)).encode("utf-8", errors="surrogatepass")
        ordered.update(int(token_id).to_bytes(8, "little", signed=False))
        ordered.update(len(encoded).to_bytes(8, "little", signed=False))
        ordered.update(encoded)
    backend = tokenizer.backend_tokenizer.to_str().encode("utf-8")
    names = ("bos", "eos", "pad", "unk", "sep", "cls", "mask")
    payload = {
        "class": type(tokenizer).__name__,
        "length": len(tokenizer),
        "ordered_tokens_sha256": ordered.hexdigest(),
        "backend_json_sha256": _sha256_bytes(backend),
        "special_token_ids": {name: getattr(tokenizer, f"{name}_token_id", None) for name in names},
        "chat_template_sha256": _sha256_bytes(
            str(getattr(tokenizer, "chat_template", None) or "").encode("utf-8")
        ),
    }
    return {**payload, "semantic_sha256": _sha256_bytes(_canonical_json_bytes(payload))}


MLX_COMPONENT_SHARD_BYTES = 256 * 1024 * 1024
MLX_COMPONENT_BUILDER_ABI = "mrun-mlx-component-builder-v1"
MLX_COMPONENT_Q4_BUILDER_ABI = "mrun-mlx-component-q4-builder-v1"
_BUILD_LOCKS_GUARD = threading.Lock()
_BUILD_LOCKS: dict[str, threading.Lock] = {}


def _build_lock(key: str) -> threading.Lock:
    with _BUILD_LOCKS_GUARD:
        return _BUILD_LOCKS.setdefault(key, threading.Lock())


def _safe_model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_name)).strip("-.")
    if not slug:
        raise MLXComponentArtifactError("model name has no safe artifact-directory form")
    return slug[:96]


def _hash_regular_file(
    path: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[str, int, tuple[int, ...]]:
    if path.is_symlink():
        raise MLXComponentArtifactError(f"native artifact file cannot be a symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MLXComponentArtifactError(f"cannot open native artifact file {path}") from exc
    digest = hashlib.sha256()
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise MLXComponentArtifactError(f"native artifact path is not regular: {path}")
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = _identity(final)
    if _identity(initial) != identity or _identity(path.lstat()) != identity:
        raise MLXComponentArtifactError(f"native artifact file changed while hashing: {path}")
    size = int(final.st_size)
    observed = digest.hexdigest()
    if expected_size is not None and size != int(expected_size):
        raise MLXComponentArtifactError(f"native artifact file size mismatch: {path}")
    if expected_sha256 is not None and observed != str(expected_sha256):
        raise MLXComponentArtifactError(f"native artifact file hash mismatch: {path}")
    return observed, size, identity


def _write_durable(path: Path, payload: bytes) -> tuple[str, int]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o644)
    try:
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _sha256_bytes(payload), len(payload)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_config_source(
    reader: VerifiedComponentGraphReader,
    model_config: Mapping[str, Any] | str | Path | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if isinstance(model_config, Mapping):
        config = json.loads(_canonical_json_bytes(dict(model_config)))
        digest = _sha256_bytes(_canonical_json_bytes(config))
        return config, digest, {"kind": "mapping", "semantic_sha256": digest}

    if model_config is None:
        try:
            config_path = snapshot_dir(reader.model_name) / "config.json"
        except (FileNotFoundError, ValueError) as exc:
            raise MLXComponentArtifactError(
                "a complete local config.json is required; pass model_config explicitly"
            ) from exc
        source_kind = "model-snapshot"
    else:
        config_path = Path(model_config).expanduser().absolute()
        source_kind = "explicit-path"
    # Hugging Face snapshots intentionally expose immutable blob objects through symlinks.
    # Resolve that locator once, then use the same no-follow verified reader on the blob itself.
    try:
        config_path = config_path.resolve(strict=True)
    except OSError as exc:
        raise MLXComponentArtifactError(f"complete config is absent: {config_path}") from exc
    try:
        config, _config_identity, file_sha256 = _read_regular_json(config_path)
    except MLXComponentGraphError as exc:
        raise MLXComponentArtifactError(f"cannot verify complete config {config_path}") from exc
    semantic = _sha256_bytes(_canonical_json_bytes(config))
    return (
        config,
        semantic,
        {
            "kind": source_kind,
            "semantic_sha256": semantic,
            "file_sha256": file_sha256,
        },
    )


def _normalized_config_value(config: Mapping[str, Any], key: str) -> Any:
    if key == "head_dim":
        return int(
            config.get("head_dim")
            or int(config.get("hidden_size", 0)) // int(config.get("num_attention_heads", 1))
        )
    if key in {"attention_bias", "mlp_bias"}:
        return bool(config.get(key, False))
    if key == "num_key_value_heads":
        return int(config.get(key, config.get("num_attention_heads", 0)))
    return config.get(key)


def _config_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and float(left) == float(right)
        )
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _effective_mlx_config(
    reader: VerifiedComponentGraphReader,
    source: Mapping[str, Any],
    *,
    bits: int = MLX_COMPONENT_BITS,
) -> tuple[dict[str, Any], bool]:
    required = {
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
    }
    missing = sorted(key for key in required if source.get(key) is None)
    if missing:
        raise MLXComponentArtifactError(
            f"complete config is missing context/model fields: {', '.join(missing)}"
        )
    if str(source.get("model_type")) != reader.architecture:
        raise MLXComponentArtifactError("complete config model_type differs from component graph")
    for key, expected in sorted(reader.config.items()):
        observed = _normalized_config_value(source, key)
        if observed is None or not _config_values_equal(observed, expected):
            raise MLXComponentArtifactError(
                f"complete config field {key!r} differs from graph runtime ABI: "
                f"{observed!r} != {expected!r}"
            )
    if int(source["max_position_embeddings"]) <= 0:
        raise MLXComponentArtifactError("max_position_embeddings must be positive")
    rope_scaling = source.get("rope_scaling")
    if rope_scaling is not None and not isinstance(rope_scaling, Mapping):
        raise MLXComponentArtifactError("rope_scaling must be an object or null")
    if reader.architecture == "qwen3" and source.get("head_dim") is None:
        raise MLXComponentArtifactError("qwen3 requires an explicit head_dim")

    effective = json.loads(_canonical_json_bytes(dict(source)))
    graph_tied = bool(reader.topology["observed_tied"])
    topology_override = bool(source.get("tie_word_embeddings", True)) != graph_tied
    effective["model_type"] = reader.architecture
    effective["head_dim"] = int(_normalized_config_value(source, "head_dim"))
    effective["num_key_value_heads"] = int(_normalized_config_value(source, "num_key_value_heads"))
    effective["tie_word_embeddings"] = graph_tied
    quantization = {
        "group_size": MLX_COMPONENT_GROUP_SIZE,
        "bits": int(bits),
        "mode": MLX_COMPONENT_MODE,
    }
    effective["quantization"] = dict(quantization)
    effective["quantization_config"] = dict(quantization)
    effective.pop("quantize_activations", None)
    return effective, topology_override


def _expected_model_blocks(
    architecture: str,
    config: Mapping[str, Any],
    *,
    tied: bool,
) -> dict[str, tuple[str, list[int]] | tuple[str, str]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    vocab = int(config["vocab_size"])
    expected: dict[str, tuple[str, list[int]] | tuple[str, str]] = {
        "embed": ("qrow", [vocab, hidden]),
        "lm_head": ("alias", "embed") if tied else ("qrow", [vocab, hidden]),
        "norm.final": ("fp32", [hidden]),
    }
    for layer in range(layers):
        prefix = f"L{layer}"
        expected.update(
            {
                f"{prefix}.q": ("qrow", [heads * head_dim, hidden]),
                f"{prefix}.k": ("qrow", [kv_heads * head_dim, hidden]),
                f"{prefix}.v": ("qrow", [kv_heads * head_dim, hidden]),
                f"{prefix}.o": ("qrow", [hidden, heads * head_dim]),
                f"{prefix}.gate": ("qrow", [intermediate, hidden]),
                f"{prefix}.up": ("qrow", [intermediate, hidden]),
                f"{prefix}.down": ("qrow", [hidden, intermediate]),
                f"{prefix}.ln1": ("fp32", [hidden]),
                f"{prefix}.ln2": ("fp32", [hidden]),
            }
        )
        if architecture == "qwen2":
            for tail, size in (
                ("q", heads * head_dim),
                ("k", kv_heads * head_dim),
                ("v", kv_heads * head_dim),
            ):
                expected[f"{prefix}.{tail}.bias"] = ("fp32", [size])
        elif architecture == "qwen3":
            expected[f"{prefix}.q_norm"] = ("fp32", [head_dim])
            expected[f"{prefix}.k_norm"] = ("fp32", [head_dim])
        elif architecture == "llama":
            if bool(config.get("attention_bias", False)):
                for tail, size in (
                    ("q", heads * head_dim),
                    ("k", kv_heads * head_dim),
                    ("v", kv_heads * head_dim),
                    ("o", hidden),
                ):
                    expected[f"{prefix}.{tail}.bias"] = ("fp32", [size])
            if bool(config.get("mlp_bias", False)):
                for tail, size in (("gate", intermediate), ("up", intermediate), ("down", hidden)):
                    expected[f"{prefix}.{tail}.bias"] = ("fp32", [size])
    return expected


def _validate_model_contract(
    reader: VerifiedComponentGraphReader,
    config: Mapping[str, Any],
) -> None:
    expected = _expected_model_blocks(
        reader.architecture,
        config,
        tied=bool(reader.topology["observed_tied"]),
    )
    if set(reader.logical_blocks) != set(expected):
        missing = sorted(set(expected) - set(reader.logical_blocks))
        extra = sorted(set(reader.logical_blocks) - set(expected))
        raise MLXComponentMappingError(
            f"logical model inventory differs from mlx-lm {reader.architecture}: "
            f"missing={missing!r}, extra={extra!r}"
        )
    for name, expected_descriptor in expected.items():
        block = reader.logical_blocks[name]
        if expected_descriptor[0] == "alias":
            if block != {"alias": expected_descriptor[1]}:
                raise MLXComponentMappingError(f"logical alias {name!r} is not canonical")
            continue
        descriptor = _semantic_block_descriptor(block, name=name)
        if descriptor != {
            "kind": expected_descriptor[0],
            "shape": expected_descriptor[1],
        }:
            raise MLXComponentMappingError(
                f"logical block {name!r} shape/kind differs from mlx-lm model contract"
            )
        target = canonical_mlx_target(name, reader.architecture)
        if target.kind != descriptor["kind"]:
            raise MLXComponentMappingError(f"logical block {name!r} maps to the wrong MLX kind")
        if descriptor["kind"] == "qrow" and int(descriptor["shape"][1]) % MLX_COMPONENT_GROUP_SIZE:
            raise MLXComponentMappingError(
                f"logical qrow {name!r} width is not divisible by {MLX_COMPONENT_GROUP_SIZE}"
            )


def _native_recipe(
    reader: VerifiedComponentGraphReader,
    *,
    effective_config_sha256: str,
    source_config_sha256: str,
    builder_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": MLX_COMPONENT_NATIVE_SCHEMA,
        "builder_abi": MLX_COMPONENT_BUILDER_ABI,
        "builder_sha256": builder_sha256,
        "mapping_abi": MLX_COMPONENT_MAPPING_ABI,
        "codec": MLX_COMPONENT_CODEC,
        "group_size": MLX_COMPONENT_GROUP_SIZE,
        "bits": MLX_COMPONENT_BITS,
        "mode": MLX_COMPONENT_MODE,
        "shard_bytes": MLX_COMPONENT_SHARD_BYTES,
        "source_custody_fingerprint_sha256": reader.custody_fingerprint_sha256,
        "source_config_sha256": source_config_sha256,
        "effective_config_sha256": effective_config_sha256,
    }


def _native_q4_recipe(
    reader: VerifiedComponentGraphReader,
    *,
    effective_config_sha256: str,
    source_config_sha256: str,
    builder_sha256: str,
) -> dict[str, Any]:
    import importlib.metadata

    return {
        "schema": MLX_COMPONENT_Q4_NATIVE_SCHEMA,
        "builder_abi": MLX_COMPONENT_Q4_BUILDER_ABI,
        "builder_sha256": builder_sha256,
        "mapping_abi": MLX_COMPONENT_MAPPING_ABI,
        "codec": MLX_COMPONENT_Q4_CODEC,
        "group_size": MLX_COMPONENT_GROUP_SIZE,
        "bits": 4,
        "mode": MLX_COMPONENT_MODE,
        "shard_bytes": MLX_COMPONENT_SHARD_BYTES,
        "quantizer": "mlx.core.quantize",
        "quantizer_input_dtype": "bfloat16",
        "auxiliary_dtype": "bfloat16",
        "error_reference": "qstore-int8-times-row-scale-float32",
        "mlx_version": importlib.metadata.version("mlx"),
        "source_custody_fingerprint_sha256": reader.custody_fingerprint_sha256,
        "source_config_sha256": source_config_sha256,
        "effective_config_sha256": effective_config_sha256,
    }


@dataclass(frozen=True)
class RequantizedAffineQ4:
    """One QStore qrow requantized by MLX to its native affine q4g64 format."""

    weight: Any
    scales: Any
    biases: Any
    max_abs_error: float
    sum_squared_error: float
    elements: int


def requantize_qrow_affine_q4_mlx(
    codes: np.ndarray,
    row_scales: np.ndarray,
    *,
    mx: Any | None = None,
) -> RequantizedAffineQ4:
    """Requantize one authenticated q8 row matrix to native affine q4g64.

    Source dequantization runs in MLX float32, then the speed lane deliberately rounds the
    quantizer input and affine metadata to bfloat16, matching mlx-lm's standard BF16-to-q4
    export. Error evidence remains measured against the original QStore float32 dequant. Peak
    expansion is bounded to one logical matrix; no Torch tensor or full model exists.
    """

    values = np.asarray(codes)
    scales = np.asarray(row_scales, dtype=np.float32)
    if values.dtype != np.int8 or values.ndim != 2:
        raise TypeError("QStore qrow codes must be a two-dimensional int8 array")
    rows, columns = (int(value) for value in values.shape)
    if columns % MLX_COMPONENT_GROUP_SIZE:
        raise ValueError(f"QStore qrow width must be divisible by {MLX_COMPONENT_GROUP_SIZE}")
    if scales.shape not in {(rows,), (rows, 1)}:
        raise ValueError("QStore qrow scales must contain exactly one value per output row")
    scales = scales.reshape(rows)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("QStore qrow scales must be finite and positive")
    if mx is None:
        try:
            import mlx.core as mx_module
        except ImportError as exc:
            raise MLXComponentArtifactError("mlx is required for q4 requantization") from exc
        mx = mx_module

    dense = mx.array(values).astype(mx.float32) * mx.array(scales)[:, None]
    quantizer_input = dense.astype(mx.bfloat16)
    quantized = mx.quantize(
        quantizer_input,
        group_size=MLX_COMPONENT_GROUP_SIZE,
        bits=4,
        mode=MLX_COMPONENT_MODE,
    )
    if len(quantized) != 3:
        raise MLXComponentArtifactError("MLX affine q4 did not return weight/scales/biases")
    weight, group_scales, group_biases = quantized
    restored = mx.dequantize(
        weight,
        group_scales,
        group_biases,
        group_size=MLX_COMPONENT_GROUP_SIZE,
        bits=4,
        mode=MLX_COMPONENT_MODE,
    )
    error = restored.astype(mx.float32) - dense
    max_abs = mx.max(mx.abs(error))
    sum_squared = mx.sum(mx.square(error))
    mx.eval(weight, group_scales, group_biases, max_abs, sum_squared)
    return RequantizedAffineQ4(
        weight=weight,
        scales=group_scales,
        biases=group_biases,
        max_abs_error=float(max_abs.item()),
        sum_squared_error=float(sum_squared.item()),
        elements=rows * columns,
    )


def _safetensors_header(
    path: Path,
) -> tuple[dict[str, Any], int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MLXComponentArtifactError(f"cannot inspect native shard {path}") from exc
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size < 10:
            raise MLXComponentArtifactError(f"native shard is not a safetensors file: {path}")
        raw_length = _pread_exact(descriptor, 0, 8, field=str(path))
        header_length = int.from_bytes(raw_length, "little", signed=False)
        if header_length <= 1 or 8 + header_length > int(initial.st_size):
            raise MLXComponentArtifactError(f"native shard has an invalid header: {path}")
        raw_header = _pread_exact(descriptor, 8, header_length, field=str(path))
        final = os.fstat(descriptor)
    except MLXComponentGraphError as exc:
        raise MLXComponentArtifactError(f"cannot inspect native shard {path}") from exc
    finally:
        os.close(descriptor)
    if _identity(initial) != _identity(final) or _identity(path.lstat()) != _identity(final):
        raise MLXComponentArtifactError(f"native shard changed during inspection: {path}")
    try:
        header = json.loads(raw_header.rstrip(b" \t\r\n"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError, MLXComponentGraphError) as exc:
        raise MLXComponentArtifactError(
            f"native shard has invalid safetensors JSON: {path}"
        ) from exc
    if not isinstance(header, dict):
        raise MLXComponentArtifactError(f"native shard header is not an object: {path}")
    data_bytes = int(final.st_size) - 8 - header_length
    return header, data_bytes, _identity(final)


class VerifiedMLXComponentArtifact:
    """Fail-closed verifier for a content-addressed native MLX model directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute()
        if self.path.is_symlink() or not self.path.is_dir():
            raise MLXComponentArtifactError("native artifact must be a real directory")
        self._directory_identity = _identity(self.path.lstat())
        try:
            manifest, manifest_identity, _manifest_file_sha = _read_regular_json(
                self.path / "manifest.json"
            )
        except MLXComponentGraphError as exc:
            raise MLXComponentArtifactError("cannot verify native artifact manifest") from exc
        self.manifest = manifest
        self._identities: dict[str, tuple[int, ...]] = {"manifest.json": manifest_identity}
        self.schema = str(manifest.get("schema", ""))
        if self.schema == MLX_COMPONENT_NATIVE_SCHEMA:
            builder_abi = MLX_COMPONENT_BUILDER_ABI
            codec = MLX_COMPONENT_CODEC
            bits = MLX_COMPONENT_BITS
        elif self.schema == MLX_COMPONENT_Q4_NATIVE_SCHEMA:
            builder_abi = MLX_COMPONENT_Q4_BUILDER_ABI
            codec = MLX_COMPONENT_Q4_CODEC
            bits = 4
        else:
            raise MLXComponentArtifactError("unsupported native artifact schema")
        self.codec = codec
        self.bits = bits
        declared_artifact = manifest.get("artifact_sha256")
        without_hash = dict(manifest)
        without_hash.pop("artifact_sha256", None)
        if declared_artifact != _sha256_bytes(_canonical_json_bytes(without_hash)):
            raise MLXComponentArtifactError("native artifact manifest hash mismatch")
        recipe = manifest.get("recipe")
        if not isinstance(recipe, Mapping):
            raise MLXComponentArtifactError("native artifact has no build recipe")
        self.build_key_sha256 = _sha256_bytes(_canonical_json_bytes(recipe))
        if manifest.get("build_key_sha256") != self.build_key_sha256:
            raise MLXComponentArtifactError("native artifact build key mismatch")
        expected_recipe_values = {
            "schema": self.schema,
            "builder_abi": builder_abi,
            "mapping_abi": MLX_COMPONENT_MAPPING_ABI,
            "codec": codec,
            "group_size": MLX_COMPONENT_GROUP_SIZE,
            "bits": bits,
            "mode": MLX_COMPONENT_MODE,
            "shard_bytes": MLX_COMPONENT_SHARD_BYTES,
        }
        if any(recipe.get(key) != value for key, value in expected_recipe_values.items()):
            raise MLXComponentArtifactError("native artifact recipe uses an incompatible ABI")
        if self.schema == MLX_COMPONENT_Q4_NATIVE_SCHEMA and (
            recipe.get("quantizer") != "mlx.core.quantize"
            or recipe.get("quantizer_input_dtype") != "bfloat16"
            or recipe.get("auxiliary_dtype") != "bfloat16"
            or recipe.get("error_reference") != "qstore-int8-times-row-scale-float32"
            or not isinstance(recipe.get("mlx_version"), str)
        ):
            raise MLXComponentArtifactError("native q4 artifact quantizer contract is invalid")

        config_record = manifest.get("config")
        if not isinstance(config_record, Mapping) or config_record.get("filename") != "config.json":
            raise MLXComponentArtifactError("native artifact config record is invalid")
        try:
            config, config_identity, config_file_sha = _read_regular_json(self.path / "config.json")
        except MLXComponentGraphError as exc:
            raise MLXComponentArtifactError("cannot verify native artifact config") from exc
        if config_file_sha != config_record.get("file_sha256"):
            raise MLXComponentArtifactError("native artifact config file hash mismatch")
        semantic_config_sha = _sha256_bytes(_canonical_json_bytes(config))
        if semantic_config_sha != config_record.get("semantic_sha256"):
            raise MLXComponentArtifactError("native artifact semantic config hash mismatch")
        if semantic_config_sha != recipe.get("effective_config_sha256"):
            raise MLXComponentArtifactError("native artifact config is not recipe-bound")
        quantization = config.get("quantization")
        if quantization != {
            "bits": bits,
            "group_size": MLX_COMPONENT_GROUP_SIZE,
            "mode": MLX_COMPONENT_MODE,
        }:
            raise MLXComponentArtifactError("native artifact quantization config is invalid")
        if self.schema == MLX_COMPONENT_Q4_NATIVE_SCHEMA:
            requantization = manifest.get("requantization")
            if (
                not isinstance(requantization, Mapping)
                or requantization.get("source_codec") != MLX_COMPONENT_CODEC
                or int(requantization.get("qrow_blocks", 0)) <= 0
                or int(requantization.get("elements", 0)) <= 0
                or int(requantization.get("auxiliary_fp32_blocks", 0)) <= 0
                or int(requantization.get("auxiliary_elements", 0)) <= 0
                or float(requantization.get("max_abs_error", -1.0)) < 0.0
                or float(requantization.get("rmse", -1.0)) < 0.0
                or float(requantization.get("auxiliary_max_abs_error", -1.0)) < 0.0
                or float(requantization.get("auxiliary_rmse", -1.0)) < 0.0
            ):
                raise MLXComponentArtifactError("native q4 requantization evidence is invalid")
        self.config = config
        self._identities["config.json"] = config_identity

        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise MLXComponentArtifactError("native artifact has no shards")
        observed_files = {"manifest.json", "config.json"}
        parameter_names: set[str] = set()
        self.shard_bytes = 0
        for record in shards:
            if not isinstance(record, Mapping):
                raise MLXComponentArtifactError("native shard record is not an object")
            filename = record.get("filename")
            role = record.get("role")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not re.fullmatch(r"model-[a-z_]+-[0-9]{5}\.safetensors", filename)
                or role not in {"body", "norm", "ingress", "egress", "lexical_shared"}
                or filename in observed_files
            ):
                raise MLXComponentArtifactError("native shard filename/role is invalid")
            path = self.path / filename
            _digest, size, identity = _hash_regular_file(
                path,
                expected_size=int(record.get("bytes", -1)),
                expected_sha256=str(record.get("sha256", "")),
            )
            header, data_bytes, header_identity = _safetensors_header(path)
            if identity != header_identity:
                raise MLXComponentArtifactError(f"native shard changed after hashing: {filename}")
            parameters = record.get("parameters")
            if not isinstance(parameters, list) or not parameters:
                raise MLXComponentArtifactError(f"native shard {filename} has no parameters")
            declared_parameters: dict[str, Mapping[str, Any]] = {}
            for parameter in parameters:
                if not isinstance(parameter, Mapping) or not isinstance(parameter.get("name"), str):
                    raise MLXComponentArtifactError(f"native shard {filename} parameter is invalid")
                name = str(parameter["name"])
                if name in declared_parameters or name in parameter_names:
                    raise MLXComponentArtifactError(f"duplicate native parameter {name!r}")
                declared_parameters[name] = parameter
                parameter_names.add(name)
            tensor_header = {key: value for key, value in header.items() if key != "__metadata__"}
            if set(tensor_header) != set(declared_parameters):
                raise MLXComponentArtifactError(
                    f"native shard {filename} header differs from its parameter manifest"
                )
            ranges: list[tuple[int, int, str]] = []
            for name, parameter in declared_parameters.items():
                tensor = tensor_header[name]
                if not isinstance(tensor, Mapping):
                    raise MLXComponentArtifactError(f"native tensor {name!r} header is invalid")
                if tensor.get("dtype") != parameter.get("dtype") or tensor.get(
                    "shape"
                ) != parameter.get("shape"):
                    raise MLXComponentArtifactError(f"native tensor {name!r} descriptor mismatch")
                offsets = tensor.get("data_offsets")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int) for value in offsets
                    )
                    or offsets[0] < 0
                    or offsets[1] <= offsets[0]
                    or offsets[1] > data_bytes
                ):
                    raise MLXComponentArtifactError(f"native tensor {name!r} span is invalid")
                ranges.append((int(offsets[0]), int(offsets[1]), name))
            previous = 0
            for start, end, name in sorted(ranges):
                if start != previous:
                    raise MLXComponentArtifactError(
                        f"native shard has a gap/overlap before tensor {name!r}"
                    )
                previous = end
            if previous != data_bytes:
                raise MLXComponentArtifactError(f"native shard {filename} has unowned data")
            observed_files.add(filename)
            self._identities[filename] = identity
            self.shard_bytes += size
        if {entry.name for entry in self.path.iterdir()} != observed_files:
            raise MLXComponentArtifactError("native artifact directory has undeclared files")
        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise MLXComponentArtifactError("native artifact source record is invalid")
        self.source = dict(source)
        self.artifact_sha256 = str(declared_artifact)

    def assert_unchanged(self) -> None:
        if _identity(self.path.lstat()) != self._directory_identity:
            raise MLXComponentArtifactError("native artifact directory changed after verification")
        if {entry.name for entry in self.path.iterdir()} != set(self._identities):
            raise MLXComponentArtifactError("native artifact file set changed after verification")
        for filename, expected in self._identities.items():
            if _identity((self.path / filename).lstat()) != expected:
                raise MLXComponentArtifactError(
                    f"native artifact file changed after verification: {filename}"
                )


def _builder_source_sha256() -> str:
    path = Path(__file__).resolve()
    digest, _size, _identity_value = _hash_regular_file(path)
    return digest


def build_mlx_component_artifact(
    graph_path: str | Path,
    output_root: str | Path,
    *,
    model_config: Mapping[str, Any] | str | Path | None = None,
) -> Path:
    """Verify, losslessly repack, and atomically publish one native MLX artifact."""

    output_root = Path(output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise MLXComponentArtifactError("native output root must be a real directory")
    output_root = output_root.resolve()

    with VerifiedComponentGraphReader(graph_path) as reader:
        source_config, source_config_sha256, source_descriptor = _read_config_source(
            reader, model_config
        )
        effective_config, topology_override = _effective_mlx_config(reader, source_config)
        _validate_model_contract(reader, effective_config)
        effective_config_sha256 = _sha256_bytes(_canonical_json_bytes(effective_config))
        builder_sha256 = _builder_source_sha256()
        recipe = _native_recipe(
            reader,
            effective_config_sha256=effective_config_sha256,
            source_config_sha256=source_config_sha256,
            builder_sha256=builder_sha256,
        )
        build_key = _sha256_bytes(_canonical_json_bytes(recipe))
        target = output_root / f"{_safe_model_slug(reader.model_name)}-{build_key[:16]}"
        with _build_lock(build_key):
            if target.exists() or target.is_symlink():
                artifact = VerifiedMLXComponentArtifact(target)
                if artifact.build_key_sha256 != build_key:
                    raise MLXComponentArtifactError(
                        "native artifact target belongs to another recipe"
                    )
                return artifact.path

            temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(output_root)))
            try:
                config_payload = _canonical_json_bytes(effective_config) + b"\n"
                config_file_sha256, config_bytes = _write_durable(
                    temporary / "config.json", config_payload
                )

                try:
                    import mlx.core as mx
                except ImportError as exc:
                    raise MLXComponentArtifactError(
                        "mlx is required to build native component safetensors"
                    ) from exc

                shard_records: list[dict[str, Any]] = []
                emitted_parameters: set[str] = set()
                for role in reader.roles:
                    shard_index = 0
                    chunk: dict[str, Any] = {}
                    chunk_parameters: list[dict[str, Any]] = []
                    chunk_bytes = 0

                    def flush_chunk(role_name: str = role) -> None:
                        nonlocal shard_index, chunk, chunk_parameters, chunk_bytes
                        if not chunk:
                            return
                        shard_index += 1
                        filename = f"model-{role_name}-{shard_index:05d}.safetensors"
                        shard_path = temporary / filename
                        mx.eval(*chunk.values())
                        mx.save_safetensors(
                            str(shard_path),
                            chunk,
                            metadata={
                                "format": "mlx",
                                "mrun-role": role_name,
                                "mrun-codec": MLX_COMPONENT_CODEC,
                            },
                        )
                        with shard_path.open("rb") as handle:
                            os.fsync(handle.fileno())
                        digest, size, _file_identity = _hash_regular_file(shard_path)
                        shard_records.append(
                            {
                                "filename": filename,
                                "role": role_name,
                                "bytes": size,
                                "sha256": digest,
                                "logical_names": sorted(
                                    {str(value["logical_name"]) for value in chunk_parameters}
                                ),
                                "parameters": sorted(
                                    chunk_parameters, key=lambda value: str(value["name"])
                                ),
                            }
                        )
                        chunk = {}
                        chunk_parameters = []
                        chunk_bytes = 0
                        mx.clear_cache()

                    for name in reader.role_names(role):
                        block = reader.logical_blocks[name]
                        if "alias" in block:
                            continue
                        target_parameter = canonical_mlx_target(name, reader.architecture)
                        new_arrays: list[tuple[str, np.ndarray, str]] = []
                        if target_parameter.kind == "qrow":
                            codes, row_scales = reader.qrow(name)
                            packed = pack_symmetric_qrow_int8(codes, row_scales)
                            new_arrays.extend(
                                (
                                    (f"{target_parameter.path}.weight", packed.weight, "U32"),
                                    (f"{target_parameter.path}.scales", packed.scales, "F32"),
                                    (f"{target_parameter.path}.biases", packed.biases, "F32"),
                                )
                            )
                        else:
                            new_arrays.append((target_parameter.path, reader.fp32(name), "F32"))
                        added_bytes = sum(int(array.nbytes) for _key, array, _dtype in new_arrays)
                        if chunk and chunk_bytes + added_bytes > MLX_COMPONENT_SHARD_BYTES:
                            flush_chunk()
                        for key, array, dtype in new_arrays:
                            if key in emitted_parameters:
                                raise MLXComponentMappingError(
                                    f"multiple logical blocks map to native parameter {key!r}"
                                )
                            emitted_parameters.add(key)
                            native = mx.array(array)
                            chunk[key] = native
                            chunk_parameters.append(
                                {
                                    "name": key,
                                    "logical_name": name,
                                    "dtype": dtype,
                                    "shape": [int(value) for value in native.shape],
                                }
                            )
                        chunk_bytes += added_bytes
                    flush_chunk()

                reader.assert_unchanged()
                source_record = {
                    "model": reader.model_name,
                    "architecture": reader.architecture,
                    "graph_schema": reader.schema,
                    "declared_graph_fingerprint_sha256": reader.declared_fingerprint_sha256,
                    "custody_fingerprint_sha256": reader.custody_fingerprint_sha256,
                    "body_abi_sha256": reader.raw["body_abi"]["semantic_sha256"],
                    "tokenizer_semantic_sha256": reader.tokenizer_semantic_sha256,
                    "source_identity_status": str(
                        reader.raw.get("source_lineage", {}).get(
                            "identity_status", "legacy-unverified"
                        )
                    ),
                }
                manifest_without_hash: dict[str, Any] = {
                    "schema": MLX_COMPONENT_NATIVE_SCHEMA,
                    "build_key_sha256": build_key,
                    "recipe": recipe,
                    "source": source_record,
                    "config": {
                        "filename": "config.json",
                        "bytes": config_bytes,
                        "file_sha256": config_file_sha256,
                        "semantic_sha256": effective_config_sha256,
                        "source": source_descriptor,
                        "topology_override": topology_override,
                    },
                    "shards": sorted(shard_records, key=lambda value: str(value["filename"])),
                }
                manifest = {
                    **manifest_without_hash,
                    "artifact_sha256": _sha256_bytes(_canonical_json_bytes(manifest_without_hash)),
                }
                _write_durable(temporary / "manifest.json", _canonical_json_bytes(manifest) + b"\n")
                _fsync_directory(temporary)
                try:
                    temporary.rename(target)
                except FileExistsError:
                    artifact = VerifiedMLXComponentArtifact(target)
                    if artifact.build_key_sha256 != build_key:
                        raise MLXComponentArtifactError(
                            "concurrent native build published a different recipe"
                        ) from None
                    return artifact.path
                _fsync_directory(output_root)
                artifact = VerifiedMLXComponentArtifact(target)
                if artifact.build_key_sha256 != build_key:
                    raise MLXComponentArtifactError(
                        "published native artifact lost recipe identity"
                    )
                return artifact.path
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)


def build_mlx_component_q4_artifact(
    graph_path: str | Path,
    output_root: str | Path,
    *,
    model_config: Mapping[str, Any] | str | Path | None = None,
) -> Path:
    """Publish a separately identified, lossy q4 speed artifact from verified QStore bytes."""

    output_root = Path(output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise MLXComponentArtifactError("native q4 output root must be a real directory")
    output_root = output_root.resolve()

    with VerifiedComponentGraphReader(graph_path) as reader:
        source_config, source_config_sha256, source_descriptor = _read_config_source(
            reader, model_config
        )
        effective_config, topology_override = _effective_mlx_config(reader, source_config, bits=4)
        _validate_model_contract(reader, effective_config)
        effective_config_sha256 = _sha256_bytes(_canonical_json_bytes(effective_config))
        builder_sha256 = _builder_source_sha256()
        recipe = _native_q4_recipe(
            reader,
            effective_config_sha256=effective_config_sha256,
            source_config_sha256=source_config_sha256,
            builder_sha256=builder_sha256,
        )
        build_key = _sha256_bytes(_canonical_json_bytes(recipe))
        target = output_root / f"{_safe_model_slug(reader.model_name)}-q4-{build_key[:16]}"
        with _build_lock(build_key):
            if target.exists() or target.is_symlink():
                artifact = VerifiedMLXComponentArtifact(target)
                if (
                    artifact.build_key_sha256 != build_key
                    or artifact.schema != MLX_COMPONENT_Q4_NATIVE_SCHEMA
                ):
                    raise MLXComponentArtifactError("native q4 target belongs to another recipe")
                return artifact.path

            temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(output_root)))
            try:
                config_payload = _canonical_json_bytes(effective_config) + b"\n"
                config_file_sha256, config_bytes = _write_durable(
                    temporary / "config.json", config_payload
                )
                try:
                    import mlx.core as mx
                except ImportError as exc:
                    raise MLXComponentArtifactError(
                        "mlx is required to build native q4 component safetensors"
                    ) from exc

                shard_records: list[dict[str, Any]] = []
                emitted_parameters: set[str] = set()
                error_records: list[dict[str, Any]] = []
                total_squared_error = 0.0
                total_elements = 0
                maximum_error = 0.0
                auxiliary_error_records: list[dict[str, Any]] = []
                auxiliary_squared_error = 0.0
                auxiliary_elements = 0
                auxiliary_maximum_error = 0.0
                for role in reader.roles:
                    shard_index = 0
                    chunk: dict[str, Any] = {}
                    chunk_parameters: list[dict[str, Any]] = []
                    chunk_bytes = 0

                    def flush_chunk(role_name: str = role) -> None:
                        nonlocal shard_index, chunk, chunk_parameters, chunk_bytes
                        if not chunk:
                            return
                        shard_index += 1
                        filename = f"model-{role_name}-{shard_index:05d}.safetensors"
                        shard_path = temporary / filename
                        mx.eval(*chunk.values())
                        mx.save_safetensors(
                            str(shard_path),
                            chunk,
                            metadata={
                                "format": "mlx",
                                "mrun-role": role_name,
                                "mrun-codec": MLX_COMPONENT_Q4_CODEC,
                            },
                        )
                        with shard_path.open("rb") as handle:
                            os.fsync(handle.fileno())
                        digest, size, _file_identity = _hash_regular_file(shard_path)
                        shard_records.append(
                            {
                                "filename": filename,
                                "role": role_name,
                                "bytes": size,
                                "sha256": digest,
                                "logical_names": sorted(
                                    {str(value["logical_name"]) for value in chunk_parameters}
                                ),
                                "parameters": sorted(
                                    chunk_parameters, key=lambda value: str(value["name"])
                                ),
                            }
                        )
                        chunk = {}
                        chunk_parameters = []
                        chunk_bytes = 0
                        mx.clear_cache()

                    for name in reader.role_names(role):
                        block = reader.logical_blocks[name]
                        if "alias" in block:
                            continue
                        target_parameter = canonical_mlx_target(name, reader.architecture)
                        new_arrays: list[tuple[str, Any, str]] = []
                        if target_parameter.kind == "qrow":
                            codes, row_scales = reader.qrow(name)
                            packed = requantize_qrow_affine_q4_mlx(codes, row_scales, mx=mx)
                            maximum_error = max(maximum_error, packed.max_abs_error)
                            total_squared_error += packed.sum_squared_error
                            total_elements += packed.elements
                            error_records.append(
                                {
                                    "logical_name": name,
                                    "elements": packed.elements,
                                    "max_abs_error": packed.max_abs_error,
                                    "rmse": math.sqrt(packed.sum_squared_error / packed.elements),
                                }
                            )
                            new_arrays.extend(
                                (
                                    (f"{target_parameter.path}.weight", packed.weight, "U32"),
                                    (f"{target_parameter.path}.scales", packed.scales, "BF16"),
                                    (f"{target_parameter.path}.biases", packed.biases, "BF16"),
                                )
                            )
                        else:
                            source_auxiliary = mx.array(reader.fp32(name))
                            rounded_auxiliary = source_auxiliary.astype(mx.bfloat16)
                            auxiliary_error = (
                                rounded_auxiliary.astype(mx.float32) - source_auxiliary
                            )
                            auxiliary_max_abs = mx.max(mx.abs(auxiliary_error))
                            auxiliary_sum_squared = mx.sum(mx.square(auxiliary_error))
                            mx.eval(
                                rounded_auxiliary,
                                auxiliary_max_abs,
                                auxiliary_sum_squared,
                            )
                            block_elements = math.prod(int(value) for value in block["shape"])
                            block_max_abs = float(auxiliary_max_abs.item())
                            block_sum_squared = float(auxiliary_sum_squared.item())
                            auxiliary_maximum_error = max(auxiliary_maximum_error, block_max_abs)
                            auxiliary_squared_error += block_sum_squared
                            auxiliary_elements += block_elements
                            auxiliary_error_records.append(
                                {
                                    "logical_name": name,
                                    "elements": block_elements,
                                    "max_abs_error": block_max_abs,
                                    "rmse": math.sqrt(block_sum_squared / block_elements),
                                }
                            )
                            new_arrays.append(
                                (
                                    target_parameter.path,
                                    rounded_auxiliary,
                                    "BF16",
                                )
                            )
                        added_bytes = sum(int(array.nbytes) for _key, array, _dtype in new_arrays)
                        if chunk and chunk_bytes + added_bytes > MLX_COMPONENT_SHARD_BYTES:
                            flush_chunk()
                        for key, array, dtype in new_arrays:
                            if key in emitted_parameters:
                                raise MLXComponentMappingError(
                                    f"multiple logical blocks map to native parameter {key!r}"
                                )
                            emitted_parameters.add(key)
                            chunk[key] = array
                            chunk_parameters.append(
                                {
                                    "name": key,
                                    "logical_name": name,
                                    "dtype": dtype,
                                    "shape": [int(value) for value in array.shape],
                                }
                            )
                        chunk_bytes += added_bytes
                        mx.clear_cache()
                    flush_chunk()

                if (
                    not error_records
                    or total_elements <= 0
                    or not auxiliary_error_records
                    or auxiliary_elements <= 0
                ):
                    raise MLXComponentArtifactError(
                        "native q4 build did not observe both qrow and auxiliary weights"
                    )
                reader.assert_unchanged()
                source_record = {
                    "model": reader.model_name,
                    "architecture": reader.architecture,
                    "graph_schema": reader.schema,
                    "declared_graph_fingerprint_sha256": reader.declared_fingerprint_sha256,
                    "custody_fingerprint_sha256": reader.custody_fingerprint_sha256,
                    "body_abi_sha256": reader.raw["body_abi"]["semantic_sha256"],
                    "tokenizer_semantic_sha256": reader.tokenizer_semantic_sha256,
                    "source_identity_status": str(
                        reader.raw.get("source_lineage", {}).get(
                            "identity_status", "legacy-unverified"
                        )
                    ),
                }
                requantization = {
                    "source_codec": MLX_COMPONENT_CODEC,
                    "quantizer": "mlx.core.quantize",
                    "qrow_blocks": len(error_records),
                    "elements": total_elements,
                    "max_abs_error": maximum_error,
                    "rmse": math.sqrt(total_squared_error / total_elements),
                    "auxiliary_fp32_blocks": len(auxiliary_error_records),
                    "auxiliary_elements": auxiliary_elements,
                    "auxiliary_max_abs_error": auxiliary_maximum_error,
                    "auxiliary_rmse": math.sqrt(auxiliary_squared_error / auxiliary_elements),
                    "blocks": sorted(error_records, key=lambda value: value["logical_name"]),
                    "auxiliary_blocks": sorted(
                        auxiliary_error_records,
                        key=lambda value: value["logical_name"],
                    ),
                }
                manifest_without_hash: dict[str, Any] = {
                    "schema": MLX_COMPONENT_Q4_NATIVE_SCHEMA,
                    "build_key_sha256": build_key,
                    "recipe": recipe,
                    "source": source_record,
                    "config": {
                        "filename": "config.json",
                        "bytes": config_bytes,
                        "file_sha256": config_file_sha256,
                        "semantic_sha256": effective_config_sha256,
                        "source": source_descriptor,
                        "topology_override": topology_override,
                    },
                    "requantization": requantization,
                    "shards": sorted(shard_records, key=lambda value: str(value["filename"])),
                }
                manifest = {
                    **manifest_without_hash,
                    "artifact_sha256": _sha256_bytes(_canonical_json_bytes(manifest_without_hash)),
                }
                _write_durable(temporary / "manifest.json", _canonical_json_bytes(manifest) + b"\n")
                _fsync_directory(temporary)
                try:
                    temporary.rename(target)
                except FileExistsError:
                    artifact = VerifiedMLXComponentArtifact(target)
                    if (
                        artifact.build_key_sha256 != build_key
                        or artifact.schema != MLX_COMPONENT_Q4_NATIVE_SCHEMA
                    ):
                        raise MLXComponentArtifactError(
                            "concurrent native q4 build published a different recipe"
                        ) from None
                    return artifact.path
                _fsync_directory(output_root)
                artifact = VerifiedMLXComponentArtifact(target)
                if (
                    artifact.build_key_sha256 != build_key
                    or artifact.schema != MLX_COMPONENT_Q4_NATIVE_SCHEMA
                ):
                    raise MLXComponentArtifactError(
                        "published native q4 artifact lost recipe identity"
                    )
                return artifact.path
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)


class MLXComponentEngine:
    """Resident Metal executor over a verified, role-separated component artifact."""

    backend = "mlx-component"
    supports_batch = True
    numerical_contract = "mlx-component-q8g64-qstore-weight-exact-v1"
    artifact_schema = MLX_COMPONENT_NATIVE_SCHEMA
    artifact_builder = staticmethod(build_mlx_component_artifact)
    artifact_root_environment = "MRUN_MLX_COMPONENT_ROOT"
    artifact_root_default = "~/.cache/mrun/mlx-component"

    def __init__(
        self,
        model_name: str,
        *,
        component_graph: str | Path | None = None,
        graph_path: str | Path | None = None,
        native_root: str | Path | None = None,
        native_artifact: str | Path | None = None,
        model_config: Mapping[str, Any] | str | Path | None = None,
        lazy: bool = False,
        **_ignored: Any,
    ) -> None:
        selected_graph = component_graph if component_graph is not None else graph_path
        if selected_graph is None or (component_graph is not None and graph_path is not None):
            raise ValueError("pass exactly one component_graph/graph_path")
        self._closed = False
        self.graph = VerifiedComponentGraphReader(selected_graph)
        try:
            self.spec = resolve_model(model_name)
            try:
                graph_spec = resolve_model(self.graph.model_name)
            except ValueError:
                graph_spec = None
            if graph_spec is not None:
                same_model = self.spec.hf_id.lower() == graph_spec.hf_id.lower()
            else:
                same_model = str(model_name).lower() == self.graph.model_name.lower()
            if not same_model:
                raise ValueError(
                    f"requested model {model_name!r} differs from component graph "
                    f"{self.graph.model_name!r}"
                )
            self.name = self.spec.name
            self.arch = self.graph.architecture

            if native_artifact is None:
                root = (
                    Path(native_root).expanduser()
                    if native_root is not None
                    else Path(
                        os.environ.get(
                            self.artifact_root_environment,
                            self.artifact_root_default,
                        )
                    ).expanduser()
                )
                artifact_path = self.artifact_builder(
                    selected_graph, root, model_config=model_config
                )
            else:
                artifact_path = Path(native_artifact)
            self.artifact = VerifiedMLXComponentArtifact(artifact_path)
            if self.artifact.schema != self.artifact_schema:
                raise MLXComponentArtifactError(
                    f"{self.backend} requires artifact schema {self.artifact_schema!r}, got "
                    f"{self.artifact.schema!r}"
                )
            if (
                self.artifact.source.get("custody_fingerprint_sha256")
                != self.graph.custody_fingerprint_sha256
                or self.artifact.source.get("tokenizer_semantic_sha256")
                != self.graph.tokenizer_semantic_sha256
                or self.artifact.source.get("model") != self.graph.model_name
            ):
                raise MLXComponentArtifactError(
                    "native artifact does not belong to the verified component graph"
                )

            self.tokenizer = load_tokenizer(self.spec, local_files_only=True)
            runtime_tokenizer = _runtime_tokenizer_descriptor(self.tokenizer)
            if runtime_tokenizer["semantic_sha256"] != self.graph.tokenizer_semantic_sha256:
                raise MLXComponentGraphError(
                    "runtime tokenizer differs from the graph-bound vocabulary contract"
                )

            try:
                import mlx.core as mx
                from mlx_lm.utils import load_model
            except ImportError as exc:
                raise RuntimeError("MLXComponentEngine requires mlx and mlx-lm") from exc
            self._mx = mx
            self.path = self.artifact.path
            active_memory_before_load = int(mx.get_active_memory())
            self.model, loaded_config = load_model(self.path, lazy=lazy, strict=True)
            self.model.eval()
            self.cfg = self.model.args
            if str(getattr(self.cfg, "model_type", "")) != self.graph.architecture:
                raise MLXComponentArtifactError("loaded mlx-lm architecture differs from graph")
            if _sha256_bytes(_canonical_json_bytes(loaded_config)) != _sha256_bytes(
                _canonical_json_bytes(self.artifact.config)
            ):
                raise MLXComponentArtifactError("mlx-lm changed the native artifact config")
            self.n_layer = int(self.cfg.num_hidden_layers)
            self.inter = int(self.cfg.intermediate_size)
            self.hidden = int(self.cfg.hidden_size)
            self.context_size = int(self.artifact.config["max_position_embeddings"])
            self.semantic_token_count = int(self.graph.semantic_token_count)
            self.artifact_bytes = int(self.artifact.shard_bytes)
            # Artifact bytes are a storage fact, not a resident-memory measurement.  MLX's
            # active counter is process-wide.  Record both the absolute observation and the
            # engine-local delta so a reference model loaded in the same process is not charged
            # to this route's placement plan.
            active_memory_after_load = int(mx.get_active_memory())
            self.mlx_engine_active_memory_bytes_at_load = max(
                0, active_memory_after_load - active_memory_before_load
            )
            self.mlx_active_memory_mb_at_load = active_memory_after_load / (1024.0 * 1024.0)
            self.mlx_peak_memory_mb_at_load = float(mx.get_peak_memory()) / (1024.0 * 1024.0)
            self.working_set_mb = self.mlx_engine_active_memory_bytes_at_load / (1024.0 * 1024.0)
            self.approximate_quantized = True
            self.compact_fused_weights = True
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> MLXComponentEngine:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        graph = getattr(self, "graph", None)
        if graph is not None:
            graph.close()
        mx = getattr(self, "_mx", None)
        if mx is not None:
            mx.clear_cache()

    def assert_content_identity_unchanged(self) -> None:
        if self._closed:
            raise MLXComponentError("native component engine is closed")
        self.graph.assert_unchanged()
        self.artifact.assert_unchanged()

    def capabilities(self):
        from .base import EngineCapabilities

        return EngineCapabilities(
            logits=True,
            logits_batch=True,
            mlp_acts=False,
            mlp_acts_batch=False,
            approximate_quantized=True,
            generation=True,
            generation_batch=True,
            persistent_kv=True,
            compact_fused_weights=True,
        )

    def encode(self, prompts: list[str], *, add_special_tokens: bool = False) -> list[np.ndarray]:
        encoded = self.tokenizer(prompts, add_special_tokens=add_special_tokens)
        return [np.asarray(ids, dtype=np.int64) for ids in encoded["input_ids"]]

    def prose_ids(self, max_len: int = 64) -> list[np.ndarray]:
        prompts = [
            "The capital of France is",
            "In a short proof, the key idea is",
            "A reliable experiment should",
            "When the model answers carefully, it",
        ]
        return [ids[:max_len] for ids in self.encode(prompts) if len(ids)]

    def _validate_ids(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        array = np.asarray(ids, dtype=np.int64)
        if array.ndim != 1 or not len(array):
            raise ValueError("token ids must be a non-empty one-dimensional sequence")
        if np.any(array < 0) or np.any(array >= self.semantic_token_count):
            raise ValueError("token ids escape the graph-bound semantic vocabulary")
        if len(array) > self.context_size:
            raise ValueError(f"prompt length {len(array)} exceeds context size {self.context_size}")
        return array

    def logits_native(self, ids: Sequence[int] | np.ndarray, *, cache: Any = None):
        array = self._validate_ids(ids)
        logits = self.model(self._mx.array(array)[None, :], cache=cache)[0]
        self._mx.eval(logits)
        return logits

    def logits_batch_native(self, ids_list: Sequence[Sequence[int] | np.ndarray]) -> list[Any]:
        from collections import defaultdict

        arrays = [self._validate_ids(ids) for ids in ids_list]
        output: list[Any | None] = [None] * len(arrays)
        by_length: dict[int, list[int]] = defaultdict(list)
        for index, array in enumerate(arrays):
            by_length[len(array)].append(index)
        for indices in by_length.values():
            batch = self._mx.array(np.stack([arrays[index] for index in indices]))
            logits = self.model(batch, cache=None)
            self._mx.eval(logits)
            for row, index in enumerate(indices):
                output[index] = logits[row]
        return output

    def logits(self, ids: np.ndarray):
        import torch

        native = self.logits_native(ids).astype(self._mx.float32)
        self._mx.eval(native)
        return torch.from_numpy(np.asarray(native))

    def logits_batch(self, ids_list: list[np.ndarray]) -> list[Any]:
        import torch

        output = []
        for native in self.logits_batch_native(ids_list):
            fp32 = native.astype(self._mx.float32)
            self._mx.eval(fp32)
            output.append(torch.from_numpy(np.ascontiguousarray(np.asarray(fp32))))
        return output

    def _greedy_sampler(self):
        semantic_tokens = self.semantic_token_count
        mx = self._mx
        return lambda logprobs: mx.argmax(logprobs[..., :semantic_tokens], axis=-1)

    def _prompt_ids(
        self,
        prompt: str | Sequence[int] | np.ndarray,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        if isinstance(prompt, str):
            values = self.encode([prompt], add_special_tokens=add_special_tokens)[0]
        else:
            values = self._validate_ids(prompt)
        return [int(value) for value in values.tolist()]

    def generate(
        self,
        prompt: str | Sequence[int] | np.ndarray,
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        stop_ids: tuple[int, ...] = (),
        return_text: bool = False,
        cache_mb: float | None = None,
        max_kv_size: int | None = None,
        prefill_step_size: int = 2048,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
    ) -> list[int] | str:
        del cache_mb
        if max_new_tokens <= 0:
            return "" if return_text else []
        ids = self._prompt_ids(prompt, add_special_tokens=add_special_tokens)
        required_context = len(ids) + int(max_new_tokens)
        if required_context > self.context_size:
            raise ValueError("prompt plus requested generation exceeds the model context size")
        if max_kv_size is not None and int(max_kv_size) < required_context:
            raise ValueError(
                "max_kv_size would evict globally attended context; this exact route does not "
                "declare a sliding-window contract"
            )
        eos = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.tokenizer, "eos_token_id", None)
        )
        stop = {int(value) for value in stop_ids}
        if eos is not None:
            stop.add(int(eos))
        from mlx_lm.generate import generate_step

        generated: list[int] = []
        for token, _logprobs in generate_step(
            self._mx.array(ids),
            self.model,
            max_tokens=int(max_new_tokens),
            sampler=self._greedy_sampler(),
            max_kv_size=max_kv_size,
            prefill_step_size=int(prefill_step_size),
            kv_bits=kv_bits,
            kv_group_size=int(kv_group_size),
        ):
            token_id = int(token.item() if hasattr(token, "item") else token)
            if token_id in stop:
                break
            generated.append(token_id)
        if return_text:
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        return generated

    def generate_batch(
        self,
        prompts: Sequence[str | Sequence[int] | np.ndarray],
        *,
        max_new_tokens: int = 48,
        eos_token_id: int | None = None,
        add_special_tokens: bool = False,
        stop_ids: tuple[int, ...] = (),
        return_text: bool = False,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        max_kv_size: int | None = None,
    ) -> list[list[int]] | list[str]:
        if not prompts:
            return []
        if max_new_tokens <= 0:
            return ["" for _ in prompts] if return_text else [[] for _ in prompts]
        encoded = [
            self._prompt_ids(prompt, add_special_tokens=add_special_tokens) for prompt in prompts
        ]
        required_contexts = [len(ids) + int(max_new_tokens) for ids in encoded]
        if any(required > self.context_size for required in required_contexts):
            raise ValueError("a prompt plus requested generation exceeds the context size")
        if max_kv_size is not None and any(
            int(max_kv_size) < required for required in required_contexts
        ):
            raise ValueError(
                "max_kv_size would evict globally attended context; this exact route does not "
                "declare a sliding-window contract"
            )
        eos = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.tokenizer, "eos_token_id", None)
        )
        stop = {int(value) for value in stop_ids}
        if eos is not None:
            stop.add(int(eos))
        from mlx_lm.generate import BatchGenerator

        generator = BatchGenerator(
            self.model,
            max_tokens=int(max_new_tokens),
            stop_tokens=[[value] for value in sorted(stop)],
            sampler=self._greedy_sampler(),
            completion_batch_size=max(1, int(completion_batch_size)),
            prefill_batch_size=max(1, int(prefill_batch_size)),
            prefill_step_size=max(1, int(prefill_step_size)),
            max_kv_size=max_kv_size,
        )
        outputs: list[list[int]] = [[] for _ in encoded]
        try:
            uids = generator.insert(encoded)
            index_by_uid = {uid: index for index, uid in enumerate(uids)}
            finished: set[int] = set()
            while len(finished) < len(uids):
                for response in generator.next_generated():
                    index = index_by_uid[int(response.uid)]
                    token = int(response.token)
                    if token not in stop:
                        outputs[index].append(token)
                    if response.finish_reason is not None:
                        finished.add(int(response.uid))
        finally:
            generator.close()
        if return_text:
            return [self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in outputs]
        return outputs

    def native_batch_generator(self, **kwargs: Any):
        """Expose mlx-lm's continuous-batching primitive for the chat serving layer."""

        from mlx_lm.generate import BatchGenerator

        kwargs.setdefault("sampler", self._greedy_sampler())
        return BatchGenerator(self.model, **kwargs)

    def runtime_report(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.name,
            "architecture": self.arch,
            "context_size": self.context_size,
            "semantic_token_count": self.semantic_token_count,
            "artifact_bytes": self.artifact_bytes,
            "mlx_active_memory_mb_at_load_process_wide": self.mlx_active_memory_mb_at_load,
            "mlx_engine_active_memory_bytes_at_load": self.mlx_engine_active_memory_bytes_at_load,
            "mlx_peak_memory_mb_at_load_process_wide": self.mlx_peak_memory_mb_at_load,
            "working_set_mb": self.working_set_mb,
            "component_graph_custody_sha256": self.graph.custody_fingerprint_sha256,
            "native_artifact_sha256": self.artifact.artifact_sha256,
            "native_build_key_sha256": self.artifact.build_key_sha256,
            "native_codec": self.artifact.codec,
            "weight_bits": self.artifact.bits,
            "numerical_contract": self.numerical_contract,
        }


class MLXComponentQ4Engine(MLXComponentEngine):
    """Lossy, separately accepted q4 speed lane over the same verified component graph."""

    backend = "mlx-component-q4"
    numerical_contract = "mlx-component-q4g64-bf16-full-model-requantized-from-qstore-v1"
    artifact_schema = MLX_COMPONENT_Q4_NATIVE_SCHEMA
    artifact_builder = staticmethod(build_mlx_component_q4_artifact)
    artifact_root_environment = "MRUN_MLX_COMPONENT_Q4_ROOT"
    artifact_root_default = "~/.cache/mrun/mlx-component-q4"
