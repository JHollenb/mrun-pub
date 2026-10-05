"""Hydra-compatible YAML contract for mrun scientific runs.

The contract deliberately keeps the axes that affect comparability visible: model identity,
runtime/backend, benchmark type, serve type, batch/context geometry, and tracking policy.  The
loader uses OmegaConf when it is available in the shared workspace and falls back to PyYAML so the
worker remains usable from a minimal mrun environment.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .runtime import canonical_device, canonical_dtype, canonical_fabric

try:
    import yaml
except ImportError:  # pragma: no cover - the package declares PyYAML
    yaml = None  # type: ignore[assignment]

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - optional outside the shared workspace
    OmegaConf = None  # type: ignore[assignment]


class ScienceConfigError(ValueError):
    """Raised when a scientific benchmark config is not safe to execute."""


_SERVE_TYPES = frozenset({"inference", "forward", "training"})
_RUNTIME_KINDS = frozenset({"engine", "sciencegraph"})


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif OmegaConf is not None:
        value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    else:
        if yaml is None:
            raise ScienceConfigError("YAML configs require PyYAML or hydra-core/omegaconf")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScienceConfigError(f"config must resolve to an object: {path}")
    return dict(value)


def _parse_override(value: str) -> Any:
    if yaml is not None:
        parsed = yaml.safe_load(value)
        return value if parsed is None and value.strip().lower() not in {"null", "~"} else parsed
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _set_dotpath(config: dict[str, Any], raw: str) -> None:
    if "=" not in raw:
        raise ScienceConfigError(f"override must be key=value, got {raw!r}")
    key, value = raw.split("=", 1)
    parts: list[str | int] = []
    for segment in key.split("."):
        if not segment:
            continue
        match = re.fullmatch(r"([^\[\]]+)((?:\[\d+\])*)", segment)
        if match is None:
            raise ScienceConfigError(f"override has an invalid key: {raw!r}")
        base = match.group(1)
        parts.append(int(base) if base.isdigit() else base)
        parts.extend(int(index) for index in re.findall(r"\[(\d+)\]", match.group(2)))
    if not parts:
        raise ScienceConfigError(f"override has an empty key: {raw!r}")
    node: Any = config
    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        if isinstance(node, list):
            if not isinstance(part, int) or part >= len(node):
                raise ScienceConfigError(f"override list index is out of range: {raw!r}")
            child = node[part]
        else:
            if not isinstance(part, str):
                raise ScienceConfigError(f"override expected a list before index: {raw!r}")
            child = node.get(part)
            if child is None:
                child = [] if isinstance(next_part, int) else {}
                node[part] = child
        if not isinstance(child, (dict, list)):
            raise ScienceConfigError(f"override traverses a scalar value: {raw!r}")
        node = child
    final = parts[-1]
    parsed = _parse_override(value)
    if isinstance(node, list):
        if not isinstance(final, int) or final >= len(node):
            raise ScienceConfigError(f"override list index is out of range: {raw!r}")
        node[final] = parsed
    else:
        if not isinstance(final, str):
            raise ScienceConfigError(f"override expected a list before index: {raw!r}")
        node[final] = parsed


def _defaults(name: str) -> dict[str, Any]:
    return {
        "schema": "mrun-scientific-benchmark-v1",
        "experiment": {"name": name, "type": "benchmark"},
        "model": {"name": None, "path": None, "path_to_model": None},
        "runtimes": [],
        "test": {
            "type": "benchmark",
            "batch_sizes": [1],
            # ``context_size`` is the ergonomic public knob.  ``context_tokens`` remains the
            # canonical matrix field so existing configs can sweep multiple lengths.
            "context_size": None,
            "context_tokens": [128],
            "decode_tokens": 16,
            "warmup": 1,
            "repeats": 3,
            "prompts": ["A reliable model runtime should"],
            "reset_cache_each_repeat": False,
        },
        # ``auto`` is resolved on the worker after host capabilities are known.  Explicit
        # values remain available for parity/reference lanes and controlled comparisons.
        "serve": {"type": "inference", "fabric": "auto", "device": "auto", "dtype": "auto"},
        "execution": {
            "host": None,
            "prefer_host": None,
            "env_alias": "mrun",
            "timeout_s": 3600,
            "priority": 0,
            "output_root": "results/science",
            "failure_policy": "fail",
            "reservation": {},
            "needs": {},
        },
        "tracking": {"local": {"enabled": True}},
        "metadata": {},
    }


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScienceConfigError(f"{field} must be an object")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScienceConfigError(f"{field} must be a positive integer")
    return value


def _normalize_context_config(test: dict[str, Any]) -> None:
    """Resolve the scalar-friendly context alias into the benchmark matrix field.

    ``test.context_size`` intentionally wins when both names are present.  This makes a
    command-line override such as ``test.context_size=2048`` useful even when the checked-in
    config declares a default ``context_tokens`` matrix.
    """

    raw = test.get("context_size")
    if raw is None:
        raw = test.get("context_tokens")
    if raw is None:
        raw = [128]
    if isinstance(raw, int) and not isinstance(raw, bool):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ScienceConfigError("test.context_size/context_tokens must be an integer or list")
    # Store both names in a stable list form so the config hash and result manifest capture the
    # exact geometry that was executed.
    test["context_tokens"] = copy.deepcopy(values)
    test["context_size"] = copy.deepcopy(values)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized copy of a scientific benchmark config."""

    value = copy.deepcopy(config)
    if value.get("schema") != "mrun-scientific-benchmark-v1":
        raise ScienceConfigError("schema must be mrun-scientific-benchmark-v1")

    experiment = _require_mapping(value.get("experiment"), "experiment")
    if not isinstance(experiment.get("name"), str) or not experiment["name"].strip():
        raise ScienceConfigError("experiment.name is required")
    if experiment.get("type") != "benchmark":
        raise ScienceConfigError("experiment.type must be benchmark")

    model = _require_mapping(value.get("model"), "model")
    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise ScienceConfigError("model.name is required")
    model_path = model.get("path") or model.get("path_to_model")
    if model_path is not None and not isinstance(model_path, str):
        raise ScienceConfigError("model.path/model.path_to_model must be a string or null")
    model["path"] = model_path
    model["path_to_model"] = model_path

    runtimes = value.get("runtimes")
    if not isinstance(runtimes, list) or not runtimes:
        raise ScienceConfigError("runtimes must be a non-empty list")
    names: set[str] = set()
    for index, runtime in enumerate(runtimes):
        runtime = _require_mapping(runtime, f"runtimes[{index}]")
        name = runtime.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ScienceConfigError(f"runtimes[{index}].name is required")
        if name in names:
            raise ScienceConfigError(f"duplicate runtime name: {name!r}")
        names.add(name)
        kind = runtime.get("kind", "engine")
        if kind not in _RUNTIME_KINDS:
            raise ScienceConfigError(
                f"runtimes[{index}].kind must be one of {sorted(_RUNTIME_KINDS)}"
            )
        if not isinstance(runtime.get("backend"), str) or not runtime["backend"].strip():
            raise ScienceConfigError(f"runtimes[{index}].backend is required")
        if not isinstance(runtime.get("options", {}), dict):
            raise ScienceConfigError(f"runtimes[{index}].options must be an object")
        runtime.setdefault("kind", kind)
        runtime.setdefault("options", {})
        options = runtime["options"]
        for field, normalizer in (
            ("fabric", canonical_fabric),
            ("device", canonical_device),
            ("dtype", canonical_dtype),
            ("compute_dtype", canonical_dtype),
        ):
            if field in options:
                try:
                    options[field] = normalizer(options[field])
                except ValueError as error:
                    raise ScienceConfigError(
                        f"runtimes[{index}].options.{field} is invalid: {error}"
                    ) from error

    test = _require_mapping(value.get("test"), "test")
    _normalize_context_config(test)
    if test.get("type") != "benchmark":
        raise ScienceConfigError("test.type must be benchmark")
    for field in ("batch_sizes", "context_tokens"):
        values = test.get(field)
        if not isinstance(values, list) or not values:
            raise ScienceConfigError(f"test.{field} must be a non-empty list")
        for index, item in enumerate(values):
            _require_positive_int(item, f"test.{field}[{index}]")
    _require_positive_int(test.get("decode_tokens"), "test.decode_tokens")
    if (
        isinstance(test.get("warmup"), bool)
        or not isinstance(test.get("warmup"), int)
        or test["warmup"] < 0
    ):
        raise ScienceConfigError("test.warmup must be a non-negative integer")
    _require_positive_int(test.get("repeats"), "test.repeats")
    prompts = test.get("prompts")
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) and p for p in prompts)
    ):
        raise ScienceConfigError("test.prompts must be a non-empty list of strings")

    serve = _require_mapping(value.get("serve"), "serve")
    if serve.get("type") not in _SERVE_TYPES:
        raise ScienceConfigError(f"serve.type must be one of {sorted(_SERVE_TYPES)}")
    try:
        serve["fabric"] = canonical_fabric(serve.get("fabric", "auto"))
        serve["device"] = canonical_device(serve.get("device", "auto"))
        serve["dtype"] = canonical_dtype(serve.get("dtype", "auto"))
    except ValueError as error:
        raise ScienceConfigError(f"serve target is invalid: {error}") from error
    if serve["type"] == "inference" and test["decode_tokens"] <= 0:
        raise ScienceConfigError("inference benchmarks require test.decode_tokens > 0")

    execution = _require_mapping(value.get("execution"), "execution")
    for field in ("host", "prefer_host"):
        if execution.get(field) is not None and (
            not isinstance(execution[field], str) or not execution[field].strip()
        ):
            raise ScienceConfigError(f"execution.{field} must be a non-empty string or null")
    if execution.get("failure_policy") not in {"fail", "continue"}:
        raise ScienceConfigError("execution.failure_policy must be fail or continue")
    if not isinstance(execution.get("output_root"), str) or not execution["output_root"]:
        raise ScienceConfigError("execution.output_root is required")
    if not isinstance(execution.get("reservation", {}), dict):
        raise ScienceConfigError("execution.reservation must be an object")
    if not isinstance(execution.get("needs", {}), dict):
        raise ScienceConfigError("execution.needs must be an object")

    tracking = _require_mapping(value.get("tracking"), "tracking")
    if set(tracking) - {"local"}:
        raise ScienceConfigError("tracking supports local artifacts only")
    return value


def load_config(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """Load, compose, and validate a YAML/JSON scientific benchmark config."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = _read(source)
    config = _deep_merge(_defaults(source.stem), raw)
    for override in overrides:
        _set_dotpath(config, override)
    return validate_config(config)


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(canonical).hexdigest()
