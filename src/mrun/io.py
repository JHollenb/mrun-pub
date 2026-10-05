"""Common IO and JSON serialization helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def raw(model: Any) -> Any:
    """Unwrap a compiled model to the underlying module when possible."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def sanitize(value: Any) -> Any:
    """Convert common scientific Python objects into JSON-safe values."""
    torch = _optional_torch()
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [sanitize(v) for v in value]
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, set):
        return [sanitize(v) for v in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if torch is not None and isinstance(value, torch.Tensor):
        return sanitize(value.detach().cpu().numpy())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    return value


def json_default(value: Any) -> Any:
    return sanitize(value)


def stable_json(data: Any) -> str:
    return json.dumps(sanitize(data), sort_keys=True, indent=2)


def compact_json(data: Any) -> str:
    return json.dumps(sanitize(data), sort_keys=True, separators=(",", ":"))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, *, sort_keys: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")


def _optional_torch() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch
