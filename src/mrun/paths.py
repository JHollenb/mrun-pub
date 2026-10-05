"""Explicit MRUN configuration with portable user-local defaults."""
from __future__ import annotations
import os
import re
from pathlib import Path


def _env(name: str) -> str | None:
    return os.environ.get(f"MRUN_{name}")


def data_root() -> Path:
    default = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "mrun"
    return Path(_env("DATA_ROOT") or str(default)).expanduser()


def cache_root() -> Path:
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mrun"
    return Path(_env("CACHE_ROOT") or str(default)).expanduser()


def repo_root() -> Path:
    """Source checkout when present, otherwise the installed package directory."""
    override = _env("REPO_ROOT")
    if override:
        return Path(override).expanduser()
    package = Path(__file__).resolve().parent
    checkout = package.parent.parent
    return checkout if (checkout / "src/mrun").is_dir() else package


def models_root() -> Path:
    return Path(_env("MODELS_ROOT") or str(cache_root() / "models")).expanduser()


def artifact_root() -> Path:
    return Path(_env("ARTIFACT_ROOT") or str(data_root() / "artifacts")).expanduser()


def stores_root() -> Path:
    return Path(_env("STORES_ROOT") or str(models_root() / "qstores")).expanduser()


def fingerprint_cache_root() -> Path:
    return Path(_env("FINGERPRINT_CACHE") or str(artifact_root() / "fingerprints")).expanduser()


def safe_stem(label: str, *, max_len: int = 80) -> str:
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_")
    out = raw[-max_len:] if len(raw) > max_len else raw
    return out.lstrip("_") or "run"
