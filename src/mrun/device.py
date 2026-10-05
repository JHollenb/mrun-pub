"""Opt-in device resolution for sweep/gather legs (CUDA on beast, CPU everywhere else).

Precedence per scope (PAGED / SPECTRO / COMP):
  1. explicit CLI value (the caller passes it through as ``cli``)
  2. env ``GATHER_DEVICE_<SCOPE>``
  3. flag file ``/mnt/big/sweep-logs/GATHER_DEVICE_<SCOPE>`` (first line)
  4. ``"cpu"``

Value format: ``"<device> [arch ...]"``, e.g. ``"cuda qwen3 llama"``. The arch tokens are an
allowlist consumed by arch-gated callers (the paged QStore: an unported forward on a device
store would crash on device-mismatched constants); scopes with no arch gate ignore them.
Missing file/dir (the Mac), malformed content, or cuda-without-a-GPU all resolve to cpu, so
the SAME code runs unchanged everywhere and activation is purely additive.

Why flag files, not env: the running beast sweep's shell environment is frozen (children
inherit it), but each leg is a fresh python process — a file read at process start reaches
every future leg with zero orchestration edits. Activation = write the file AFTER the item's
parity gate passes; rollback = delete it.
"""
from __future__ import annotations

import os
from pathlib import Path

from .paths import data_root

FLAG_DIR = Path(os.environ.get("MRUN_DEVICE_FLAG_DIR", str(data_root() / "device-flags")))
SCOPES = ("paged", "spectro", "comp", "mlp")
_DEVICES = ("cpu", "cuda")


def resolve(scope: str, cli: str | None = None) -> tuple[str, tuple[str, ...]]:
    """Resolve the device for ``scope`` -> ``(device, allowed_archs)``.

    Whenever cuda is granted, TF32 is forced OFF (matmul + cudnn): TF32's ~1e-3 relative
    error breaches the fleet's 4-decimal-rounded stats; strict-fp32 CUDA measured
    rel ~5e-7 vs the CPU baseline.
    """
    scope = scope.lower()
    assert scope in SCOPES, f"unknown device scope {scope!r} (want one of {SCOPES})"
    raw = cli
    if raw is None:
        raw = os.environ.get(f"GATHER_DEVICE_{scope.upper()}")
    if raw is None:
        try:
            raw = (FLAG_DIR / f"GATHER_DEVICE_{scope.upper()}").read_text().splitlines()[0]
        except Exception:      # missing dir/file (the Mac), unreadable, empty
            raw = "cpu"
    parts = str(raw).split()
    dev = parts[0].lower() if parts else "cpu"
    archs = tuple(parts[1:])
    if dev not in _DEVICES:    # malformed -> cpu (mps stays a per-script explicit CLI choice)
        return "cpu", ()
    if dev == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print(f"warn: device_flag[{scope}] wants cuda but torch.cuda is unavailable -> cpu",
                      flush=True)
                return "cpu", ()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception as exc:
            # A cuda grant with a broken torch install must not masquerade as an
            # ordinary CPU run — that is a silent ~8x slowdown on the cuda host.
            print(f"warn: device_flag[{scope}] wants cuda but torch failed ({exc!r}) -> cpu",
                  flush=True)
            return "cpu", ()
    return dev, archs
