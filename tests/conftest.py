"""Public-boundary guard and explicit local scheduler test configuration."""
import builtins
import importlib
import os

import pytest

FORBIDDEN = frozenset({
    "manalysis", "common_harness", "model_experiments", "saturn", "saturn_pub",
    "mlflow", "boto3", "trackio", "minio",
})


_real_import = builtins.__import__
_real_import_module = importlib.import_module


def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in FORBIDDEN:
        raise ImportError(f"private/service dependency forbidden: {name}")
    return _real_import(name, *args, **kwargs)


def guarded_import_module(name, package=None):
    if name.split(".")[0] in FORBIDDEN:
        raise ImportError(f"private/service dependency forbidden: {name}")
    return _real_import_module(name, package)


# Allow libraries to probe absent optional dependencies with find_spec. Only an
# actual import is forbidden; a failed discovery probe can corrupt lazy imports.
builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")


@pytest.fixture(autouse=True)
def explicit_local_scheduler(monkeypatch):
    # Existing mechanics tests intentionally omit client tokens. Tests of the
    # production default explicitly remove this opt-in.
    monkeypatch.setenv("MRUN_ALLOW_UNAUTHENTICATED", "1")
