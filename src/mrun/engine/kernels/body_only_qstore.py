"""Repack an int8 QStore without its lexical ingress/egress blocks.

The repacker copies the existing quantized bytes and scales; it does not dequantize or
re-quantize the transformer body.  The resulting image deliberately uses a separate legacy-style
schema because the source images used by the first experiment predate semantic QStore
provenance.  The body-only manifest carries its own content hash and source-manifest hash, while
the ordinary QStore reader continues to load it through its compatibility path.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BODY_ONLY_SCHEMA = "mrun-qstore-body-only-v1"
QSTORE_FILES = ("weights.i8", "scales.f32", "extras.f32")
LEXICAL_NAMES = frozenset({"embed", "lm_head"})


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _alias_root(blocks: Mapping[str, Mapping[str, Any]], name: str) -> str:
    seen: set[str] = set()
    current = name
    while True:
        if current in seen:
            raise ValueError(f"cyclic QStore alias at {name!r}")
        seen.add(current)
        block = blocks.get(current)
        if not isinstance(block, Mapping):
            raise ValueError(f"QStore alias {name!r} targets missing block {current!r}")
        alias = block.get("alias")
        if alias is None:
            return current
        if not isinstance(alias, str) or not alias:
            raise ValueError(f"QStore alias {current!r} has an invalid target")
        current = alias


def _copy_range(source: Path, destination: Any, start: int, length: int) -> None:
    if start < 0 or length <= 0:
        raise ValueError("QStore copy ranges must be positive")
    source_size = source.stat().st_size
    if start + length > source_size:
        raise ValueError(f"QStore copy range exceeds {source}: {start}+{length}>{source_size}")
    with source.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise OSError(f"unexpected EOF while copying {source}")
            destination.write(chunk)
            remaining -= len(chunk)


def _physical_blocks(
    blocks: Mapping[str, Mapping[str, Any]],
    *,
    kind: str,
    offset_key: str,
) -> list[tuple[str, Mapping[str, Any]]]:
    return sorted(
        (
            (name, block)
            for name, block in blocks.items()
            if isinstance(block, Mapping)
            and block.get("kind") == kind
            and _alias_root(blocks, name) not in LEXICAL_NAMES
        ),
        key=lambda item: (int(item[1][offset_key]), item[0]),
    )


def build_body_only_qstore(
    source_path: str | Path,
    output_path: str | Path,
    *,
    lexical_names: frozenset[str] = LEXICAL_NAMES,
) -> Path:
    """Create a physically stripped body image from one int8 QStore.

    ``embed`` and ``lm_head`` (including aliases that resolve to either physical root) are
    omitted. Every retained qrow/fp32 byte range is copied unchanged into compact files and its
    offsets are rewritten. The output is atomic and never overwrites an existing directory.
    """

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if source.name == "manifest.json":
        source = source.parent
    if output.name == "manifest.json":
        output = output.parent
    manifest_path = source / "manifest.json"
    if not source.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"source QStore is missing manifest.json: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite body-only QStore: {output}")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, dict):
        raise ValueError("source QStore manifest must be an object")
    if source_manifest.get("dtype") != "int8":
        raise ValueError("body-only repacking currently supports int8 QStores only")
    source_blocks = source_manifest.get("blocks")
    if not isinstance(source_blocks, dict) or not source_blocks:
        raise ValueError("source QStore has no block table")
    blocks = {
        str(name): block
        for name, block in source_blocks.items()
        if isinstance(name, str) and isinstance(block, Mapping)
    }
    if len(blocks) != len(source_blocks):
        raise ValueError("source QStore block table contains invalid entries")

    lexical_roots = frozenset(str(name) for name in lexical_names)
    removed_names = sorted(
        name for name in blocks if _alias_root(blocks, name) in lexical_roots
    )
    if not removed_names:
        raise ValueError("source QStore has no lexical blocks to remove")

    retained_qrows = sorted(
        (
            (name, block)
            for name, block in blocks.items()
            if block.get("kind") == "qrow"
            and _alias_root(blocks, name) not in lexical_roots
        ),
        key=lambda item: (int(item[1]["w_off"]), item[0]),
    )
    retained_fp32 = sorted(
        (
            (name, block)
            for name, block in blocks.items()
            if block.get("kind") == "fp32"
            and _alias_root(blocks, name) not in lexical_roots
        ),
        key=lambda item: (int(item[1]["e_off"]), item[0]),
    )
    if not retained_qrows or not retained_fp32:
        raise ValueError("body-only image must retain qrow and fp32 blocks")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    source_files = {name: source / name for name in QSTORE_FILES}
    new_blocks: dict[str, dict[str, Any]] = {}
    try:
        with (
            (temporary / "weights.i8").open("wb") as weights_out,
            (temporary / "scales.f32").open("wb") as scales_out,
            (temporary / "extras.f32").open("wb") as extras_out,
        ):
            weight_offset = 0
            scale_offset = 0
            for name, block in retained_qrows:
                copied = dict(block)
                weight_length = int(block["w_len"])
                scale_length = int(block["s_len"])
                _copy_range(
                    source_files["weights.i8"],
                    weights_out,
                    int(block["w_off"]),
                    weight_length,
                )
                _copy_range(
                    source_files["scales.f32"],
                    scales_out,
                    int(block["s_off"]),
                    scale_length,
                )
                copied["w_off"] = weight_offset
                copied["s_off"] = scale_offset
                new_blocks[name] = copied
                weight_offset += weight_length
                scale_offset += scale_length

            extra_offset = 0
            for name, block in retained_fp32:
                copied = dict(block)
                extra_length = int(block["e_len"])
                _copy_range(
                    source_files["extras.f32"],
                    extras_out,
                    int(block["e_off"]),
                    extra_length,
                )
                copied["e_off"] = extra_offset
                new_blocks[name] = copied
                extra_offset += extra_length

        # Preserve non-lexical aliases. In the Qwen source image lm_head aliases embed and is
        # therefore already absent; this loop matters for future tied/aliased architectures.
        for name, block in blocks.items():
            if name in new_blocks or name in removed_names:
                continue
            if "alias" in block:
                target = str(block["alias"])
                if _alias_root(blocks, target) not in lexical_roots:
                    new_blocks[name] = {"alias": target}

        body_manifest = {
            key: value
            for key, value in source_manifest.items()
            if key not in {"schema_version", "source", "builder", "derived", "blocks"}
        }
        body_manifest.update(
            {
                "schema_version": BODY_ONLY_SCHEMA,
                "blocks": new_blocks,
                "body_only": {
                    "removed_logical_blocks": removed_names,
                    "removed_physical_roots": sorted(
                        {_alias_root(blocks, name) for name in removed_names}
                    ),
                    "source_manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                    "source_qstore": str(source),
                },
            }
        )
        body_manifest["body_only"]["manifest_sha256"] = _canonical_sha256(body_manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(body_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return output


def inspect_body_only_qstore(path: str | Path) -> dict[str, Any]:
    """Return compact structural evidence that an image is physically body-only."""

    root = Path(path).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    blocks = manifest.get("blocks", {})
    lexical = sorted(
        name
        for name, block in blocks.items()
        if name in LEXICAL_NAMES or (isinstance(block, Mapping) and block.get("alias") in LEXICAL_NAMES)
    )
    return {
        "schema_version": manifest.get("schema_version"),
        "lexical_blocks_present": lexical,
        "body_only": manifest.get("body_only"),
        "file_bytes": {
            name: (root / name).stat().st_size
            for name in QSTORE_FILES
        },
        "block_count": len(blocks),
    }
