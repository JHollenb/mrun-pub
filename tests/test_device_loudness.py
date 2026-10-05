"""Loud-downgrade contract: a granted-but-unusable device must WARN, never silently cpu.

The trap class is measured, not hypothetical: bare GATHER_DEVICE_PAGED=cuda (empty arch
allowlist) produced argmax-correct answers ~8x slower (7087 vs 842 ms/fwd, qwen3-4b).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import mrun.engine.kernels.qstore as qstore_mod
from mrun.engine.kernels.dense_qstore_cuda import DenseQStore
from mrun.engine.kernels.qstore import QStore


def _tiny_store(tmp: Path, arch: str = "qwen2") -> Path:
    d = tmp / "s"
    d.mkdir()
    q = np.zeros((4, 4), np.int8)
    sc = np.ones(4, np.float32)
    q.reshape(-1).tofile(d / "weights.i8")
    sc.tofile(d / "scales.f32")
    np.zeros(1, np.float32).tofile(d / "extras.f32")
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "arch": arch,
                "config": {"hidden_size": 4},
                "blocks": {
                    "b": {
                        "kind": "qrow",
                        "shape": [4, 4],
                        "w_off": 0,
                        "w_len": 16,
                        "s_off": 0,
                        "s_len": 16,
                    }
                },
            }
        )
    )
    return d


def test_arch_gate_rejection_is_loud(tmp_path, monkeypatch, capsys):
    # Flag grants cuda but the store's arch is not allowlisted -> cpu + LOUD warning.
    _tiny_store(tmp_path, arch="qwen2")
    monkeypatch.setattr(qstore_mod, "_resolve_device", lambda scope: ("cuda", ("llama",)))
    store = QStore("s", root=tmp_path)
    assert store.device == "cpu"
    err_out = capsys.readouterr().out
    assert "allowlist" in err_out and "8x slower" in err_out


def test_arch_gate_grant_stays_quiet_on_cpu_resolution(tmp_path, monkeypatch, capsys):
    # Ordinary cpu resolution (the Mac) must NOT spam warnings.
    _tiny_store(tmp_path)
    monkeypatch.setattr(qstore_mod, "_resolve_device", lambda scope: ("cpu", ()))
    store = QStore("s", root=tmp_path)
    assert store.device == "cpu"
    assert "allowlist" not in capsys.readouterr().out


def test_dense_explicit_qwen2_cuda_bypasses_global_paged_allowlist(tmp_path, monkeypatch, capsys):
    """An explicit dense CUDA engine owns placement; the paged flag cannot downgrade it."""

    _tiny_store(tmp_path, arch="qwen2")
    resolutions = []
    monkeypatch.setattr(
        qstore_mod,
        "_resolve_device",
        lambda scope: resolutions.append(scope) or ("cuda", ("qwen3", "llama")),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    store = DenseQStore(
        "s",
        root=tmp_path,
        device="cuda:0",
        require_triton=False,
    )
    try:
        assert store.device == "cuda:0"
        assert resolutions == []
        assert store.ring_stats() is None
        assert "allowlist" not in capsys.readouterr().out
    finally:
        store.close()


def test_resolve_torch_failure_is_loud(monkeypatch, capsys):
    # A cuda grant with a broken torch must warn, not masquerade as an ordinary cpu run.
    import builtins

    import mrun.device as device_mod

    real_import = builtins.__import__

    def broken_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("simulated broken install")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", broken_torch)
    dev, archs = device_mod.resolve("paged", "cuda qwen2")
    assert dev == "cpu" and archs == ()
    assert "torch failed" in capsys.readouterr().out
