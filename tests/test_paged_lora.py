"""Paged LoRA trainer gates: init-parity, finite-diff grad check, overfit collapse.

Gated on a built qwen store (``MODEL_EXPERIMENTS_RUN_MODEL_TESTS=1`` + reachable store root).
"""
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
QWEN3_MODEL = "qwen3-0.6b"


def _store_available(model: str) -> bool:
    return (stores_root() / store_name(model)).joinpath("manifest.json").exists()


@requires_model
@pytest.mark.parametrize("model", [MODEL, QWEN3_MODEL])
def test_paged_lora_init_parity(model):
    """At init (LoRA B=0 ⇒ ΔW=0) the trainer forward == frozen batched_paged_logits (argmax-exact)."""
    import torch

    from mrun.engine.kernels.paged_forward import batched_paged_logits
    from mrun.training import LoRAConfig, PagedLoRATrainer

    if not _store_available(model):
        pytest.skip(f"no paged store for {model}")
    tr = PagedLoRATrainer(model, LoRAConfig(rank=4, targets=("q", "v")))
    rng = np.random.default_rng(0)
    ids = [rng.integers(5, tr.V - 5, size=L, dtype=np.int64) for L in (10, 7)]
    with torch.no_grad():
        tlog = tr.forward_logits(ids, grad=False)
    flog, lengths = batched_paged_logits(tr.store, ids, last_only=False)
    md, top1 = 0.0, 0
    for b in range(len(ids)):
        Lb = int(lengths[b])
        md = max(md, float((tlog[b, :Lb] - flog[b, :Lb]).abs().max()))
        top1 += int((tlog[b, :Lb].argmax(-1) == flog[b, :Lb].argmax(-1)).all())
    assert top1 == len(ids), f"init-parity argmax {top1}/{len(ids)}"
    assert md < 1e-4, f"init-parity max|Δ|={md}"


@requires_model
def test_paged_lora_grad_check():
    """Finite-difference vs autograd on LoRA B entries (B=0 at init ⇒ grad_B is the live target)."""
    import torch

    from mrun.training import LoRAConfig, PagedLoRATrainer

    if not _store_available(MODEL):
        pytest.skip(f"no paged store for {MODEL}")
    tr = PagedLoRATrainer(MODEL, LoRAConfig(rank=4, targets=("q", "v")))
    rng = np.random.default_rng(0)
    ids = [rng.integers(5, tr.V - 5, size=L, dtype=np.int64) for L in (10, 7)]
    tr.loss(ids).backward()
    Bp = tr.lora[(tr.layers[0], "q")]["B"]
    gB = Bp.grad.detach().clone()
    eps, fd, an = 1e-3, [], []
    for (i, j) in [(0, 0), (1, 2), (2, 1)]:
        if i >= Bp.shape[0] or j >= Bp.shape[1]:
            continue
        with torch.no_grad():
            Bp[i, j] += eps
        lp = float(tr.loss(ids).item())
        with torch.no_grad():
            Bp[i, j] -= 2 * eps
        lm = float(tr.loss(ids).item())
        with torch.no_grad():
            Bp[i, j] += eps
        fd.append((lp - lm) / (2 * eps)); an.append(float(gB[i, j]))
    atol, rtol = 0.02, 0.05
    ratios = [abs(f - a) / (atol + rtol * abs(a)) for f, a in zip(fd, an)]
    assert max(ratios) <= 1.0, f"grad-check ratios {ratios} (fd={fd}, an={an})"


@requires_model
def test_paged_lora_overfit_collapses():
    """One short sequence; CE must collapse over a few Adam steps ⇒ backward trains."""
    import torch

    from mrun.training import LoRAConfig, PagedLoRATrainer

    if not _store_available(MODEL):
        pytest.skip(f"no paged store for {MODEL}")
    tr = PagedLoRATrainer(MODEL, LoRAConfig(rank=8, alpha=16, targets=("q", "k", "v", "o")))
    rng = np.random.default_rng(0)
    seq = [rng.integers(5, tr.V - 5, size=12, dtype=np.int64)]
    opt = torch.optim.Adam(tr.parameters(), lr=5e-3)
    ce0 = float(tr.loss(seq).item())
    ce1 = ce0
    for _ in range(30):
        ce1 = tr.step(seq, opt)
    assert ce1 < 0.5 * ce0, f"overfit CE {ce0:.3f} -> {ce1:.3f}"
