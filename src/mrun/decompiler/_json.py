"""Strict canonical JSON and content-identity helpers for the decompiler.

The decompiler treats JSON as an artifact format, not as a convenient Python dump.  Duplicate
keys, non-finite numbers, non-string object keys, and unknown schema fields are therefore hard
errors.  Canonical bytes are shared by every fingerprint in this package.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"JSON contains non-finite number {value!r}")


def strict_json_loads(payload: str | bytes | bytearray, *, field: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {exc}") from exc


def strict_json_file(path: Path, *, max_bytes: int, field: str) -> Any:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"{field} is empty: {path}")
    if size > max_bytes:
        raise ValueError(f"{field} exceeds the {max_bytes}-byte parsing limit: {path}")
    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) != size:
        raise ValueError(f"{field} changed while it was read: {path}")
    return strict_json_loads(payload, field=field)


def normalize_json(value: Any, *, field: str = "value") -> Any:
    """Return a detached, JSON-native value while rejecting lossy coercions."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{field} object keys must be non-empty strings")
            if key in normalized:
                raise ValueError(f"{field} contains duplicate key {key!r}")
            normalized[key] = normalize_json(item, field=f"{field}.{key}")
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize_json(item, field=f"{field}[]") for item in value]
    raise TypeError(f"{field} contains non-JSON value {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        normalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_sha256(value: Any, *, field: str) -> str:
    if not is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def require_name(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def require_exact_keys(payload: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if any(type(key) is not str for key in payload):
        raise TypeError(f"{field} keys must be strings")
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{field} is missing fields: {missing!r}")
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {unknown!r}")


def require_dict(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value


def require_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array")
    return value


def require_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def require_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a boolean")
    return value


def require_str(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise TypeError(f"{field} must be {qualifier}")
    return value


def decode_canonical_object(value: str, *, field: str) -> dict[str, Any]:
    payload = strict_json_loads(value, field=field)
    if not isinstance(payload, dict):
        raise TypeError(f"{field} must decode to an object")
    return payload


def object_json(value: Mapping[str, Any], *, field: str) -> str:
    normalized = normalize_json(value, field=field)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field} must be an object")
    return canonical_json(normalized)


def model_json(model: Any) -> str:
    return json.dumps(model.as_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
