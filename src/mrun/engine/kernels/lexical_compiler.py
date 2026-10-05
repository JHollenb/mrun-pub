"""Compile one tokenizer vocabulary onto a frozen transformer body's lexical ABI.

The token-address map is the symbol table; this module is the linker/compiler that turns that
symbol table into body-coordinate input and output rows.  Shared token rows are copied exactly.
Tokens that do not exist in the source vocabulary are initialized from a deterministic
decomposition through the source tokenizer, with a statistics/UNK fallback when decomposition is
not available.  The body remains untouched.

This is an initialization compiler, not a claim of arbitrary-tokenizer parity.  The resulting
artifact is executable immediately and can then be improved with ``train_lexical_bridge`` while
the body stays frozen.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .composite_qstore import canonical_json_bytes
from .lexical import LexicalBinding, LexicalComponent, body_abi

LEXICAL_COMPILATION_SCHEMA = "mrun-lexical-compilation-v1"
LEXICAL_COMPILATION_MAPPING_SCHEMA = "mrun-lexical-compilation-mapping-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(payload + b"\n")
    return _sha256_bytes(payload + b"\n")


def _write_manifest(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned["semantic_sha256"] = _sha256_bytes(canonical_json_bytes(unsigned))
    path.write_text(json.dumps(unsigned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return unsigned


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"lexical compilation manifest is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != LEXICAL_COMPILATION_SCHEMA:
        raise ValueError("unsupported lexical compilation schema")
    claimed = value.get("semantic_sha256")
    unsigned = {key: item for key, item in value.items() if key != "semantic_sha256"}
    if claimed != _sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("lexical compilation semantic fingerprint mismatch")
    return value


def _token_text(tokenizer: Any, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return "" if token is None else str(token)


def _normalise_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = value.get("input_ids", [])
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if value is None:
        return []
    return [int(item) for item in value]


def _decode_one(tokenizer: Any, token_id: int) -> str:
    try:
        value = tokenizer.decode(
            [int(token_id)],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
    except TypeError:
        try:
            value = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        except Exception:  # noqa: BLE001 - tokenizer implementations vary
            return ""
    except Exception:  # noqa: BLE001 - a malformed individual token should use fallback
        return ""
    return "" if value is None else str(value)


def _decompose_token(source_tokenizer: Any, target_tokenizer: Any, target_id: int) -> list[int]:
    decoded = _decode_one(target_tokenizer, target_id)
    if not decoded:
        return []
    try:
        encoded = source_tokenizer.encode(decoded, add_special_tokens=False)
    except Exception:  # noqa: BLE001 - source tokenizers expose different encode signatures
        return []
    return _normalise_ids(encoded)


def _valid_source_ids(ids: list[int], token_count: int) -> list[int]:
    if not ids or any(token_id < 0 or token_id >= token_count for token_id in ids):
        return []
    return ids


@dataclass(frozen=True)
class LexicalCompilationResult:
    """Compiled target lexical component plus its immutable source-address map."""

    root: Path
    component: LexicalComponent
    manifest: dict[str, Any]
    mapping: tuple[dict[str, Any], ...]

    @property
    def semantic_sha256(self) -> str:
        return str(self.manifest["semantic_sha256"])

    def source_ids(self, target_id: int) -> tuple[int, ...]:
        token_id = int(target_id)
        if token_id < 0 or token_id >= len(self.mapping):
            raise IndexError(f"target token ID must be inside [0, {len(self.mapping)})")
        return tuple(int(value) for value in self.mapping[token_id]["source_token_ids"])


def _source_row_values(binding: LexicalBinding, source_ids: list[int], *, output: bool) -> np.ndarray:
    rows = binding.output_rows if output else binding.input_rows
    matrix = binding.weights.lm_head if output else binding.weights.embed
    row_ids = rows[np.asarray(source_ids, dtype=np.int64)]
    values = np.asarray(matrix[row_ids], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("lexical decomposition did not produce rank-2 source rows")
    return values.mean(axis=0, dtype=np.float32)


def _fallback_row(
    matrix: np.ndarray,
    semantic_rows: np.ndarray,
    *,
    unk_row: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    if unk_row is not None:
        return np.array(unk_row, dtype=np.float32, copy=True), "fallback_unk"
    source = np.asarray(matrix[semantic_rows], dtype=np.float32)
    mean = source.mean(axis=0, dtype=np.float32)
    std = float(source.std(dtype=np.float32))
    if not np.isfinite(std) or std <= 0.0:
        return np.array(mean, dtype=np.float32, copy=True), "fallback_mean"
    row = rng.normal(
        loc=mean.astype(np.float64),
        scale=std,
        size=(int(matrix.shape[1]),),
    ).astype(np.float32)
    return row, "fallback_stats"


def compile_lexical_component(
    root: str | Path,
    *,
    source_binding: LexicalBinding,
    target_tokenizer: Any,
    body_config: Mapping[str, Any],
    architecture: str,
    seed: int = 0,
    provenance: Mapping[str, Any] | None = None,
) -> LexicalCompilationResult:
    """Compile ``target_tokenizer`` rows into the source body's latent lexical space.

    ``source_binding`` supplies both the source token-address map and the source lexical values.
    The body ABI is checked before any output is published.  The target vocabulary may be smaller
    than the body's configured row count, but it may not exceed it.
    """

    root = Path(root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite lexical compilation: {root}")

    source_values = source_binding.values
    source_weights = source_binding.weights
    abi = body_abi(body_config, architecture)
    if source_weights.manifest.get("body_abi_sha256") != abi["semantic_sha256"]:
        raise ValueError("source lexical weights are incompatible with the target body ABI")

    source_token_count = int(source_values.token_count)
    target_token_count = int(len(target_tokenizer))
    configured_rows = int(body_config.get("vocab_size", 0))
    if source_token_count <= 0 or target_token_count <= 0:
        raise ValueError("source and target tokenizers must have positive vocabularies")
    if configured_rows <= 0 or target_token_count > configured_rows:
        raise ValueError(
            "target tokenizer vocabulary must fit inside the body's configured row space "
            f"({target_token_count} > {configured_rows})"
        )

    source_tokenizer = source_values.tokenizer
    source_by_text: dict[str, list[int]] = {}
    for source_id in range(source_token_count):
        source_by_text.setdefault(_token_text(source_tokenizer, source_id), []).append(source_id)

    input_matrix = np.asarray(source_weights.embed, dtype=np.float32)
    output_matrix = np.asarray(source_weights.lm_head, dtype=np.float32)
    source_input_rows = source_binding.input_rows[:source_token_count]
    source_output_rows = source_binding.output_rows[:source_token_count]
    semantic_input = input_matrix[source_input_rows]
    semantic_output = output_matrix[source_output_rows]
    if semantic_input.ndim != 2 or semantic_output.ndim != 2:
        raise ValueError("source lexical matrices do not cover semantic tokenizer rows")

    source_unk = getattr(source_tokenizer, "unk_token_id", None)
    unk_input = None
    unk_output = None
    if source_unk is not None and 0 <= int(source_unk) < source_token_count:
        unk_input = input_matrix[source_binding.input_rows[int(source_unk)]]
        unk_output = output_matrix[source_binding.output_rows[int(source_unk)]]

    rng = np.random.default_rng(int(seed))
    compiled_input = np.empty((target_token_count, input_matrix.shape[1]), dtype=np.float32)
    compiled_output = np.empty((target_token_count, output_matrix.shape[1]), dtype=np.float32)
    mapping: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()

    for target_id in range(target_token_count):
        target_text = _token_text(target_tokenizer, target_id)
        exact = source_by_text.get(target_text, [])
        if exact:
            source_ids = [int(exact[0])]
            kind = "exact"
            compiled_input[target_id] = input_matrix[source_binding.input_rows[source_ids[0]]]
            compiled_output[target_id] = output_matrix[source_binding.output_rows[source_ids[0]]]
        else:
            source_ids = _valid_source_ids(
                _decompose_token(source_tokenizer, target_tokenizer, target_id),
                source_token_count,
            )
            if source_ids:
                kind = "decomposed"
                compiled_input[target_id] = _source_row_values(
                    source_binding, source_ids, output=False
                )
                compiled_output[target_id] = _source_row_values(
                    source_binding, source_ids, output=True
                )
            else:
                source_ids = []
                compiled_input[target_id], input_kind = _fallback_row(
                    input_matrix,
                    source_input_rows,
                    unk_row=unk_input,
                    rng=rng,
                )
                compiled_output[target_id], output_kind = _fallback_row(
                    output_matrix,
                    source_output_rows,
                    unk_row=unk_output,
                    rng=rng,
                )
                kind = input_kind if input_kind == output_kind else "fallback"
        mapping.append(
            {
                "target_token_id": target_id,
                "source_token_ids": source_ids,
                "kind": kind,
            }
        )
        kinds[kind] += 1

    if source_weights.tied:
        compiled_output = compiled_input.copy()
        lm_head: np.ndarray | None = None
    else:
        lm_head = compiled_output

    root.mkdir(parents=True)
    component_provenance = {
        **dict(provenance or {}),
        "compiler": "mrun.lexical_compiler",
        "compiler_schema": LEXICAL_COMPILATION_SCHEMA,
        "source_values_semantic_sha256": source_values.semantic_sha256,
        "source_weights_semantic_sha256": source_weights.semantic_sha256,
        "source_binding_semantic_sha256": source_binding.semantic_sha256,
        "seed": int(seed),
        "initialization": "exact-rows-or-mean-source-decomposition-v1",
    }
    component = LexicalComponent.create(
        root,
        tokenizer=target_tokenizer,
        embed=compiled_input,
        lm_head=lm_head,
        body_config=body_config,
        architecture=architecture,
        provenance=component_provenance,
    )

    mapping_path = root / "target_to_source.json"
    mapping_payload = {
        "schema_version": LEXICAL_COMPILATION_MAPPING_SCHEMA,
        "source_tokenizer_semantic_sha256": source_values.manifest["tokenizer"]["semantic_sha256"],
        "target_tokenizer_semantic_sha256": component.manifest["tokenizer"]["semantic_sha256"],
        "records": mapping,
    }
    mapping_sha = _write_json(mapping_path, mapping_payload)
    compile_manifest = _write_manifest(
        root / "compile_manifest.json",
        {
            "schema_version": LEXICAL_COMPILATION_SCHEMA,
            "source": {
                "values_semantic_sha256": source_values.semantic_sha256,
                "weights_semantic_sha256": source_weights.semantic_sha256,
                "binding_semantic_sha256": source_binding.semantic_sha256,
                "tokenizer_semantic_sha256": source_values.manifest["tokenizer"]["semantic_sha256"],
                "token_count": source_token_count,
            },
            "target": {
                "tokenizer_semantic_sha256": component.manifest["tokenizer"]["semantic_sha256"],
                "token_count": target_token_count,
            },
            "body_abi_sha256": abi["semantic_sha256"],
            "component_semantic_sha256": component.manifest["semantic_sha256"],
            "mapping": {
                "file": mapping_path.name,
                "sha256": mapping_sha,
                "kind_counts": dict(sorted(kinds.items())),
            },
            "initialization": "exact-rows-or-mean-source-decomposition-v1",
            "provenance": dict(provenance or {}),
        },
    )
    return LexicalCompilationResult(root, component, compile_manifest, tuple(mapping))


def load_lexical_compilation(
    root: str | Path,
    *,
    body_config: Mapping[str, Any],
    architecture: str,
) -> LexicalCompilationResult:
    """Load and verify a compiled lexical component and its source-address map."""

    root = Path(root).expanduser().resolve()
    manifest = _read_manifest(root / "compile_manifest.json")
    component = LexicalComponent.load(root, body_config=body_config, architecture=architecture)
    if manifest.get("component_semantic_sha256") != component.manifest.get("semantic_sha256"):
        raise ValueError("lexical compilation component fingerprint mismatch")
    mapping_path = root / str(manifest["mapping"]["file"])
    if _sha256_file(mapping_path) != manifest["mapping"]["sha256"]:
        raise ValueError("lexical compilation mapping is missing or changed")
    mapping_manifest = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping_manifest.get("schema_version") != LEXICAL_COMPILATION_MAPPING_SCHEMA:
        raise ValueError("unsupported lexical compilation mapping schema")
    mapping = mapping_manifest.get("records")
    if not isinstance(mapping, list) or len(mapping) != component.token_count:
        raise ValueError("lexical compilation mapping does not cover target tokenizer IDs")
    return LexicalCompilationResult(root, component, manifest, tuple(mapping))


def inspect_lexical_compilation(root: str | Path) -> dict[str, Any]:
    """Return the verified compile manifest without loading weight matrices."""

    return _read_manifest(Path(root).expanduser().resolve() / "compile_manifest.json")
