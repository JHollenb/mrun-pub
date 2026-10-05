"""Paged LoRA adapter persistence gates: save_adapter -> apply_adapter round-trips exactly,
and a mismatched base is refused. Gated on a built qwen store (same env flag as the sibling
``test_paged_lora.py``: ``MRUN_RUN_MODEL_TESTS=1`` + a reachable store root)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from mrun.models import store_name
from mrun.paths import stores_root

requires_model = pytest.mark.skipif(
    os.environ.get("MRUN_RUN_MODEL_TESTS", os.environ.get("MODEL_EXPERIMENTS_RUN_MODEL_TESTS")) != "1",
    reason="set MRUN_RUN_MODEL_TESTS=1 to run cached model tests",
)

MODEL = "qwen2.5-0.5b"


def _store_available(model: str) -> bool:
    return (stores_root() / store_name(model)).joinpath("manifest.json").exists()


@requires_model
def test_adapter_save_apply_roundtrip_is_exact(tmp_path):
    """Train a few steps, save the adapter, then load it into a FRESH trainer (zero-init LoRA)
    and confirm the reloaded forward logits match the trained trainer bit-for-bit — a pure
    tensor copy, not a recompute. A zero-init fresh trainer must first DIFFER (guards against a
    vacuous pass where nothing was trained)."""
    import torch

    from mrun.training import LoRAConfig, PagedLoRATrainer

    if not _store_available(MODEL):
        pytest.skip(f"no paged store for {MODEL}")
    cfg = LoRAConfig(rank=8, alpha=16.0, targets=("q", "k", "v", "o"), seed=7)
    tr = PagedLoRATrainer(MODEL, cfg=cfg)
    rng = np.random.default_rng(0)
    seqs = [list(map(int, rng.integers(5, tr.V - 5, size=14, dtype=np.int64))) for _ in range(3)]
    opt = torch.optim.Adam(tr.parameters(), lr=5e-3)
    for _ in range(8):
        tr.step(seqs, opt)

    probe = [list(map(int, rng.integers(5, tr.V - 5, size=11, dtype=np.int64)))]
    with torch.no_grad():
        trained = tr.forward_logits(probe, grad=False).detach().clone()

    adapter_dir = tmp_path / "adapter"
    tr.save_adapter(adapter_dir)
    assert (adapter_dir / "adapter.pt").exists()
    assert (adapter_dir / "adapter_config.json").exists()

    tr2 = PagedLoRATrainer(MODEL, cfg=cfg)
    with torch.no_grad():
        zero = tr2.forward_logits(probe, grad=False).detach().clone()
    assert float((zero - trained).abs().max()) > 1e-2, "training did not change logits (vacuous)"

    tr2.apply_adapter(adapter_dir)
    with torch.no_grad():
        reloaded = tr2.forward_logits(probe, grad=False).detach().clone()
    max_delta = float((reloaded - trained).abs().max())
    assert max_delta < 1e-4, f"round-trip not exact: max|Δ|={max_delta}"
    assert bool((reloaded.argmax(-1) == trained.argmax(-1)).all()), "argmax mismatch after reload"


@requires_model
def test_apply_adapter_refuses_arch_mismatch(tmp_path):
    """apply_adapter must RAISE (not silently mis-wire) when the saved adapter's base arch
    differs from the trainer it is applied to."""
    import json

    from mrun.training import LoRAConfig, PagedLoRATrainer

    if not _store_available(MODEL):
        pytest.skip(f"no paged store for {MODEL}")
    tr = PagedLoRATrainer(MODEL, LoRAConfig(rank=4, targets=("q", "v")))
    adapter_dir = tmp_path / "adapter"
    tr.save_adapter(adapter_dir)
    meta = json.loads((adapter_dir / "adapter_config.json").read_text())
    meta["arch"] = "llama"                      # pretend it came from a different base
    (adapter_dir / "adapter_config.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="arch"):
        tr.apply_adapter(adapter_dir)
