"""Tokenizer-ID to lexical-row and QStore-byte address maps.

The tokenizer owns semantic IDs; a model owns lexical rows.  For a native QStore the
relationship is usually an identity prefix, but making it explicit gives callers one stable
lookup surface for tied and untied models, padded vocabulary rows, and later replacement
tokenizers.

The map is deliberately lexical-only.  It binds to the source QStore manifest and payload
hashes, but it does not copy embedding values.  A row address is a locator, not a tensor: the
caller still chooses whether to read the native QStore row or a separated lexical component.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .composite_qstore import VocabManifest, canonical_json_bytes, tokenizer_descriptor
from .qstore import _load_qstore_manifest

TOKEN_ADDRESS_MAP_SCHEMA = "mrun-token-address-map-v1"
TOKEN_ADDRESS_MAP_RESULT_SCHEMA = "mrun-token-address-map-result-v1"
_SOURCE_FILES = ("manifest.json", "weights.i8", "scales.f32", "extras.f32")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"token-address map manifest is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != TOKEN_ADDRESS_MAP_SCHEMA:
        raise ValueError("unsupported token-address map schema")
    claimed = value.get("semantic_sha256")
    unsigned = {key: item for key, item in value.items() if key != "semantic_sha256"}
    if claimed != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        raise ValueError("token-address map semantic fingerprint mismatch")
    return value


def _alias_root(blocks: Mapping[str, Mapping[str, Any]], name: str) -> str:
    seen: set[str] = set()
    current = str(name)
    while True:
        if current in seen:
            raise ValueError(f"cyclic QStore alias at {name!r}")
        seen.add(current)
        try:
            block = blocks[current]
        except KeyError as exc:
            raise ValueError(f"QStore alias target is missing for {name!r}") from exc
        alias = block.get("alias")
        if alias is None:
            return current
        current = str(alias)


def _lexical_address(
    manifest: Mapping[str, Any],
    logical_name: str,
    *,
    configured_row_count: int,
) -> dict[str, Any]:
    blocks = manifest.get("blocks")
    if not isinstance(blocks, Mapping):
        raise ValueError("QStore manifest has no block table")
    root = _alias_root(blocks, logical_name)
    block = blocks[root]
    if block.get("kind") != "qrow":
        raise ValueError(f"lexical block {logical_name!r} must be a qrow")
    shape = block.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"lexical block {logical_name!r} has an invalid shape")
    row_count, hidden_size = (int(shape[0]), int(shape[1]))
    if row_count < configured_row_count:
        raise ValueError(
            f"lexical block {logical_name!r} has {row_count} rows, below configured "
            f"vocabulary size {configured_row_count}"
        )
    if int(block.get("w_len", -1)) != row_count * hidden_size:
        raise ValueError(f"lexical block {logical_name!r} has invalid int8 row geometry")
    if int(block.get("s_len", -1)) != row_count * 4:
        raise ValueError(f"lexical block {logical_name!r} has invalid scale row geometry")
    return {
        "logical_name": logical_name,
        "physical_root": root,
        "kind": "qrow",
        "shape": [row_count, hidden_size],
        "row_count": row_count,
        "row_value_stride_bytes": hidden_size,
        "row_scale_stride_bytes": 4,
        "weights_file": "weights.i8",
        "scales_file": "scales.f32",
        "weights_offset": int(block["w_off"]),
        "scales_offset": int(block["s_off"]),
    }


def _source_hashes(root: Path) -> dict[str, str]:
    return {name: _sha256_file(root / name) for name in _SOURCE_FILES}


def _identity_rows(count: int) -> np.ndarray:
    return np.arange(int(count), dtype=np.int64)


def _validate_rows(
    rows: np.ndarray,
    *,
    domain_size: int,
    codomain_size: int,
    label: str,
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.int64)
    if values.shape != (int(domain_size),):
        raise ValueError(f"{label} must have shape [{int(domain_size)}]")
    if values.size and (values.min() < 0 or values.max() >= int(codomain_size)):
        raise ValueError(f"{label} contains a row outside [0, {int(codomain_size)})")
    return values


def _token_records(tokenizer: Any, input_rows: np.ndarray, output_rows: np.ndarray) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for token_id in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        records.append(
            {
                "token_id": int(token_id),
                "token": "" if token is None else str(token),
                "input_row": int(input_rows[token_id]),
                "output_row": int(output_rows[token_id]),
            }
        )
    return records


def _write_records(root: Path, records: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    tokens_path = root / "tokens.jsonl"
    lookup: dict[str, list[int]] = {}
    with tokens_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            token = str(record["token"])
            token_id = int(record["token_id"])
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
            lookup.setdefault(token, []).append(token_id)
    lookup_path = root / "token_lookup.json"
    _write_json(lookup_path, lookup)
    return _sha256_file(tokens_path), _sha256_file(lookup_path)


def _publish_map_artifact(
    root: Path,
    *,
    tokenizer: Any,
    input_rows: np.ndarray,
    output_rows: np.ndarray,
    token_count: int,
    output_token_count: int,
    manifest_body: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one already-validated map and bind its manifest to every generated file."""

    root.mkdir(parents=True)
    np.save(root / "input_rows.npy", input_rows)
    np.save(root / "output_rows.npy", output_rows)
    _write_records(root, _token_records(tokenizer, input_rows, output_rows))
    files = {
        "input_rows": "input_rows.npy",
        "output_rows": "output_rows.npy",
        "tokens": "tokens.jsonl",
        "token_lookup": "token_lookup.json",
    }
    file_sha256 = {key: _sha256_file(root / filename) for key, filename in files.items()}
    unsigned = {**dict(manifest_body), "files": files, "file_sha256": file_sha256}
    unsigned["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    _write_json(root / "manifest.json", unsigned)
    return unsigned


class TokenAddressMap:
    """Immutable tokenizer-ID, token-text, lexical-row, and QStore-address mapping."""

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        input_rows: np.ndarray,
        output_rows: np.ndarray,
    ) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.input_rows = np.asarray(input_rows, dtype=np.int64)
        self.output_rows = np.asarray(output_rows, dtype=np.int64)
        self.token_count = int(self.manifest["token_count"])
        self.output_token_count = int(self.manifest["output_token_count"])
        if self.input_rows.shape != (self.token_count,):
            raise ValueError("token-address input map does not cover tokenizer IDs")
        if self.output_rows.shape != (self.output_token_count,):
            raise ValueError("token-address output map does not cover output rows")

    @property
    def semantic_sha256(self) -> str:
        return str(self.manifest["semantic_sha256"])

    @property
    def model_name(self) -> str:
        return str(self.manifest["model_name"])

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        qstore_root: str | Path,
        tokenizer: Any,
        input_rows: Sequence[int] | np.ndarray | None = None,
        output_rows: Sequence[int] | np.ndarray | None = None,
        output_token_count: int | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> TokenAddressMap:
        root = Path(root).expanduser().resolve()
        qstore_root = Path(qstore_root).expanduser().resolve()
        if root.exists():
            raise FileExistsError(f"refusing to overwrite token-address map: {root}")

        manifest, _identity = _load_qstore_manifest(qstore_root, expected_dtype="int8")
        config = manifest.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("QStore manifest has no config")
        configured_row_count = int(config.get("vocab_size", 0))
        if configured_row_count <= 0:
            raise ValueError("QStore config has no positive vocab_size")
        token_count = int(len(tokenizer))
        if token_count <= 0 or token_count > configured_row_count:
            raise ValueError("tokenizer length must fit inside configured vocabulary rows")

        descriptor = tokenizer_descriptor(tokenizer)
        vocab = VocabManifest.from_descriptor(
            descriptor, configured_row_count=configured_row_count
        )
        embed_address = _lexical_address(
            manifest, "embed", configured_row_count=configured_row_count
        )
        lm_head_address = _lexical_address(
            manifest, "lm_head", configured_row_count=configured_row_count
        )
        requested_output_count = (
            configured_row_count if output_token_count is None else int(output_token_count)
        )
        if (
            requested_output_count < token_count
            or requested_output_count > lm_head_address["row_count"]
        ):
            raise ValueError("output_token_count exceeds the lexical output row space")
        input_map = _identity_rows(token_count) if input_rows is None else np.asarray(input_rows, dtype=np.int64)
        output_map = (
            _identity_rows(requested_output_count)
            if output_rows is None
            else np.asarray(output_rows, dtype=np.int64)
        )
        input_map = _validate_rows(
            input_map,
            domain_size=token_count,
            codomain_size=embed_address["row_count"],
            label="input_rows",
        )
        output_map = _validate_rows(
            output_map,
            domain_size=requested_output_count,
            codomain_size=lm_head_address["row_count"],
            label="output_rows",
        )
        source_hashes = _source_hashes(qstore_root)
        source_manifest_sha = source_hashes["manifest.json"]
        source_payload = {
            "model_name": str(manifest.get("model_name", qstore_root.name)),
            "architecture": str(manifest.get("arch", "")),
            "manifest_sha256": source_manifest_sha,
            "manifest_semantic_sha256": manifest.get("semantic_sha256"),
            "file_sha256": source_hashes,
        }
        manifest_body: dict[str, Any] = {
            "schema_version": TOKEN_ADDRESS_MAP_SCHEMA,
            "model_name": str(manifest.get("model_name", qstore_root.name)),
            "architecture": str(manifest.get("arch", "")),
            "source_qstore": source_payload,
            "tokenizer": descriptor,
            "vocab_manifest": vocab.as_dict(),
            "token_count": token_count,
            "output_token_count": requested_output_count,
            "configured_row_count": configured_row_count,
            "tied": embed_address["physical_root"] == lm_head_address["physical_root"],
            "row_mappings": {
                "input": {
                    "kind": "identity_prefix" if np.array_equal(input_map, _identity_rows(token_count)) else "explicit",
                    "domain_size": token_count,
                    "codomain_size": embed_address["row_count"],
                },
                "output": {
                    "kind": "identity_prefix" if np.array_equal(output_map, _identity_rows(requested_output_count)) else "explicit",
                    "domain_size": requested_output_count,
                    "codomain_size": lm_head_address["row_count"],
                },
            },
            "address_spaces": {"embed": embed_address, "lm_head": lm_head_address},
            "padded_output_rows": {
                "start": token_count,
                "stop": configured_row_count,
                "count": max(0, configured_row_count - token_count),
            },
            "provenance": dict(provenance or {}),
        }
        published = _publish_map_artifact(
            root,
            tokenizer=tokenizer,
            input_rows=input_map,
            output_rows=output_map,
            token_count=token_count,
            output_token_count=requested_output_count,
            manifest_body=manifest_body,
        )
        return cls(root, published, input_map, output_map)

    @classmethod
    def create_linked(
        cls,
        root: str | Path,
        *,
        base_qstore_root: str | Path,
        extension_qstore_root: str | Path,
        tokenizer: Any,
        input_rows: Sequence[int] | np.ndarray | None = None,
        output_rows: Sequence[int] | np.ndarray | None = None,
        output_token_count: int | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> TokenAddressMap:
        """Create a lexical address map for a resolved base-plus-extension image.

        Token IDs remain owned by the tokenizer and row mappings remain explicit.  Only the
        provider of each lexical address changes.  In a tied model, an ``embed`` overlay also
        routes the logical ``lm_head`` address to the extension, matching ``ResolvedLinkedQStore``.
        """

        root = Path(root).expanduser().resolve()
        base_root = Path(base_qstore_root).expanduser().resolve()
        extension_root = Path(extension_qstore_root).expanduser().resolve()
        if root.exists():
            raise FileExistsError(f"refusing to overwrite token-address map: {root}")

        base_manifest, _base_identity = _load_qstore_manifest(
            base_root, expected_dtype="int8"
        )
        extension_manifest, _extension_identity = _load_qstore_manifest(
            extension_root, expected_dtype="int8"
        )
        base_config = base_manifest.get("config")
        extension_config = extension_manifest.get("config")
        if not isinstance(base_config, Mapping) or not isinstance(extension_config, Mapping):
            raise ValueError("linked QStores must both declare a config")
        if dict(base_config) != dict(extension_config):
            raise ValueError("linked QStore configs do not match")
        if base_manifest.get("arch") != extension_manifest.get("arch"):
            raise ValueError("linked QStore architectures do not match")
        if base_manifest.get("dtype") != extension_manifest.get("dtype"):
            raise ValueError("linked QStore dtypes do not match")

        linked_image = extension_manifest.get("linked_image")
        if not isinstance(linked_image, Mapping):
            raise ValueError("extension QStore has no linked_image metadata")
        overlay_blocks = linked_image.get("overlay_blocks")
        if not isinstance(overlay_blocks, list) or not overlay_blocks:
            raise ValueError("extension QStore has no non-empty overlay_blocks")
        overlay = {str(name) for name in overlay_blocks}
        extension_id = str(linked_image.get("extension_id", ""))
        if not extension_id:
            raise ValueError("extension QStore has no extension_id")

        configured_row_count = int(base_config.get("vocab_size", 0))
        if configured_row_count <= 0:
            raise ValueError("linked QStore config has no positive vocab_size")
        token_count = int(len(tokenizer))
        if token_count <= 0 or token_count > configured_row_count:
            raise ValueError("tokenizer length must fit inside configured vocabulary rows")
        descriptor = tokenizer_descriptor(tokenizer)
        vocab = VocabManifest.from_descriptor(
            descriptor, configured_row_count=configured_row_count
        )

        base_embed = _lexical_address(
            base_manifest, "embed", configured_row_count=configured_row_count
        )
        base_lm_head = _lexical_address(
            base_manifest, "lm_head", configured_row_count=configured_row_count
        )
        extension_blocks = extension_manifest.get("blocks")
        if not isinstance(extension_blocks, Mapping):
            raise ValueError("extension QStore has no block table")

        def effective_address(logical_name: str, base_address: Mapping[str, Any]) -> dict[str, Any]:
            base_physical = str(base_address["physical_root"])
            overlaid = logical_name in overlay or base_physical in overlay
            if not overlaid:
                result = dict(base_address)
                result["provider"] = "base"
                return result
            candidate = logical_name if logical_name in extension_blocks else base_physical
            if candidate not in extension_blocks:
                raise ValueError(
                    f"extension overlays {logical_name!r} but has no lexical block "
                    f"{candidate!r}"
                )
            result = _lexical_address(
                extension_manifest,
                candidate,
                configured_row_count=configured_row_count,
            )
            if result["shape"] != base_address["shape"]:
                raise ValueError(f"extension lexical block {logical_name!r} changes shape")
            result["logical_name"] = logical_name
            result["provider"] = "extension"
            return result

        embed_address = effective_address("embed", base_embed)
        lm_head_address = effective_address("lm_head", base_lm_head)
        requested_output_count = (
            configured_row_count if output_token_count is None else int(output_token_count)
        )
        if requested_output_count < token_count or requested_output_count > lm_head_address["row_count"]:
            raise ValueError("output_token_count exceeds the linked lexical output row space")
        input_map = _identity_rows(token_count) if input_rows is None else np.asarray(input_rows, dtype=np.int64)
        output_map = (
            _identity_rows(requested_output_count)
            if output_rows is None
            else np.asarray(output_rows, dtype=np.int64)
        )
        input_map = _validate_rows(
            input_map,
            domain_size=token_count,
            codomain_size=embed_address["row_count"],
            label="input_rows",
        )
        output_map = _validate_rows(
            output_map,
            domain_size=requested_output_count,
            codomain_size=lm_head_address["row_count"],
            label="output_rows",
        )

        def source_payload(manifest: Mapping[str, Any], store_root: Path) -> dict[str, Any]:
            hashes = _source_hashes(store_root)
            return {
                "model_name": str(manifest.get("model_name", store_root.name)),
                "architecture": str(manifest.get("arch", "")),
                "manifest_sha256": hashes["manifest.json"],
                "manifest_semantic_sha256": manifest.get("semantic_sha256"),
                "file_sha256": hashes,
            }

        manifest_body: dict[str, Any] = {
            "schema_version": TOKEN_ADDRESS_MAP_SCHEMA,
            "map_kind": "linked_extension",
            "model_name": str(base_manifest.get("model_name", base_root.name)),
            "architecture": str(base_manifest.get("arch", "")),
            "source_qstore": source_payload(base_manifest, base_root),
            "extension_qstore": source_payload(extension_manifest, extension_root),
            "extension": {
                "extension_id": extension_id,
                "overlay_blocks": sorted(overlay),
                "selection": linked_image.get("selection"),
                "representation": linked_image.get("representation"),
            },
            "tokenizer": descriptor,
            "vocab_manifest": vocab.as_dict(),
            "token_count": token_count,
            "output_token_count": requested_output_count,
            "configured_row_count": configured_row_count,
            "tied": base_embed["physical_root"] == base_lm_head["physical_root"],
            "row_mappings": {
                "input": {
                    "kind": "identity_prefix" if np.array_equal(input_map, _identity_rows(token_count)) else "explicit",
                    "domain_size": token_count,
                    "codomain_size": embed_address["row_count"],
                },
                "output": {
                    "kind": "identity_prefix" if np.array_equal(output_map, _identity_rows(requested_output_count)) else "explicit",
                    "domain_size": requested_output_count,
                    "codomain_size": lm_head_address["row_count"],
                },
            },
            "address_spaces": {"embed": embed_address, "lm_head": lm_head_address},
            "padded_output_rows": {
                "start": token_count,
                "stop": configured_row_count,
                "count": max(0, configured_row_count - token_count),
            },
            "provenance": dict(provenance or {}),
        }
        published = _publish_map_artifact(
            root,
            tokenizer=tokenizer,
            input_rows=input_map,
            output_rows=output_map,
            token_count=token_count,
            output_token_count=requested_output_count,
            manifest_body=manifest_body,
        )
        return cls(root, published, input_map, output_map)

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        tokenizer: Any | None = None,
        qstore_root: str | Path | None = None,
        extension_qstore_root: str | Path | None = None,
    ) -> TokenAddressMap:
        root = Path(root).expanduser().resolve()
        manifest = _read_manifest(root)
        files = manifest.get("files")
        hashes = manifest.get("file_sha256")
        if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
            raise ValueError("token-address map file table is missing")
        for key in ("input_rows", "output_rows", "tokens", "token_lookup"):
            filename = root / str(files[key])
            if not filename.is_file() or _sha256_file(filename) != hashes[key]:
                raise ValueError(f"token-address map file is missing or changed: {key}")
        input_rows = np.load(root / str(files["input_rows"]), allow_pickle=False)
        output_rows = np.load(root / str(files["output_rows"]), allow_pickle=False)
        result = cls(root, manifest, input_rows, output_rows)
        if tokenizer is not None:
            descriptor = tokenizer_descriptor(tokenizer)
            if descriptor["semantic_sha256"] != manifest["tokenizer"]["semantic_sha256"]:
                raise ValueError("token-address map tokenizer fingerprint mismatch")
            if len(tokenizer) != result.token_count:
                raise ValueError("token-address map tokenizer length mismatch")
        if qstore_root is not None:
            if extension_qstore_root is None:
                result.assert_source_unchanged(qstore_root)
            else:
                result.assert_linked_sources_unchanged(qstore_root, extension_qstore_root)
        return result

    def address(self, token_id: int, *, space: str = "embed") -> dict[str, Any]:
        token_id = int(token_id)
        if space == "embed":
            if token_id < 0 or token_id >= self.token_count:
                raise IndexError(f"input token IDs must be inside [0, {self.token_count})")
            row = int(self.input_rows[token_id])
        elif space == "lm_head":
            if token_id < 0 or token_id >= self.output_token_count:
                raise IndexError(
                    f"output row IDs must be inside [0, {self.output_token_count})"
                )
            row = int(self.output_rows[token_id])
        else:
            raise ValueError("space must be 'embed' or 'lm_head'")
        descriptor = self.manifest["address_spaces"][space]
        return {
            "space": space,
            "token_id": token_id,
            "row": row,
            "provider": descriptor.get("provider", "source"),
            "physical_root": descriptor["physical_root"],
            "weights_file": descriptor["weights_file"],
            "weights_byte_offset": int(descriptor["weights_offset"]) + row * int(descriptor["row_value_stride_bytes"]),
            "scales_file": descriptor["scales_file"],
            "scale_byte_offset": int(descriptor["scales_offset"]) + row * int(descriptor["row_scale_stride_bytes"]),
        }

    def lookup_token(self, token: str) -> list[int]:
        lookup_path = self.root / str(self.manifest["files"]["token_lookup"])
        lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
        values = lookup.get(str(token), [])
        return [int(value) for value in values]

    def assert_source_unchanged(self, qstore_root: str | Path) -> None:
        root = Path(qstore_root).expanduser().resolve()
        expected = self.manifest["source_qstore"]["file_sha256"]
        actual = _source_hashes(root)
        if actual != expected:
            raise RuntimeError("source QStore changed after token-address map creation")

    def assert_linked_sources_unchanged(
        self,
        base_qstore_root: str | Path,
        extension_qstore_root: str | Path,
    ) -> None:
        """Recheck both providers for a linked-extension map."""

        base_root = Path(base_qstore_root).expanduser().resolve()
        extension_root = Path(extension_qstore_root).expanduser().resolve()
        expected_base = self.manifest["source_qstore"]["file_sha256"]
        expected_extension = self.manifest["extension_qstore"]["file_sha256"]
        if _source_hashes(base_root) != expected_base:
            raise RuntimeError("base QStore changed after linked token-address map creation")
        if _source_hashes(extension_root) != expected_extension:
            raise RuntimeError(
                "extension QStore changed after linked token-address map creation"
            )


def inspect_token_address_map(root: str | Path) -> dict[str, Any]:
    artifact = TokenAddressMap.load(root)
    manifest = artifact.manifest
    return {
        "schema_version": manifest["schema_version"],
        "model_name": artifact.model_name,
        "architecture": manifest["architecture"],
        "semantic_sha256": artifact.semantic_sha256,
        "token_count": artifact.token_count,
        "output_token_count": artifact.output_token_count,
        "configured_row_count": int(manifest["configured_row_count"]),
        "tied": bool(manifest["tied"]),
        "padded_output_rows": manifest["padded_output_rows"],
        "address_spaces": manifest["address_spaces"],
        "row_mappings": manifest["row_mappings"],
    }


__all__ = [
    "TOKEN_ADDRESS_MAP_RESULT_SCHEMA",
    "TOKEN_ADDRESS_MAP_SCHEMA",
    "TokenAddressMap",
    "inspect_token_address_map",
]
