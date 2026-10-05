"""forward_patched_rows / ablation_suite_ce — per-row conditions must equal scalar patching.

Model-gated (distilgpt2, cpu): skips when the snapshot isn't cached locally. The whole
claim is EXACTNESS: batching different conditions as rows changes nothing about each row's
math, so logits must match the scalar forward_patched per condition.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _engine():
    from mrun.engine import open_engine

    try:
        return open_engine("distilgpt2", backend="hf", device="cpu")
    except Exception as e:  # noqa: BLE001 — no snapshot / offline
        pytest.skip(f"distilgpt2 not loadable locally: {e}")


def _ids(eng, text):
    return np.asarray(eng.encode([text], add_special_tokens=False)[0], dtype=np.int64)


def test_rows_match_scalar_forward_patched():
    eng = _engine()
    ids = _ids(eng, "The lighthouse keeper counted the ships as they passed the point")
    conditions = [
        None,                                        # baseline row
        {1: [("zero", list(range(0, 96)), None)]},   # early-layer ablation
        {3: [("scale", list(range(200, 260)), 0.5)]},
        {1: [("zero", list(range(0, 48)), None)],    # multi-layer condition
         4: [("zero", list(range(700, 750)), None)]},
    ]
    batched = eng.forward_patched_rows([ids] * len(conditions), conditions)
    for cond, row_logits in zip(conditions, batched, strict=True):
        scalar_logits, _, _ = eng.forward_patched(ids, patch_ops_by_layer=cond)
        torch.testing.assert_close(row_logits, scalar_logits.float(), atol=1e-4, rtol=1e-4)


def test_rows_isolated_from_each_other():
    eng = _engine()
    ids = _ids(eng, "Rain hammered the tin roof through the night")
    heavy = {2: [("zero", list(range(0, 1500)), None)]}
    # baseline row next to a heavily-ablated row: baseline must be untouched
    both = eng.forward_patched_rows([ids, ids], [None, heavy])
    clean = eng.logits(ids)
    torch.testing.assert_close(both[0], clean.float(), atol=1e-5, rtol=1e-5)
    assert (both[1] - clean.float()).abs().max() > 1e-2  # ablation actually did something


def test_ablation_suite_ce_shapes_and_baseline():
    eng = _engine()
    bank = [_ids(eng, t) for t in
            ("The market opened lower after the announcement",
             "She tuned the violin before the concert began")]
    conditions = [None, {2: [("zero", list(range(0, 400)), None)]}]
    ces = eng.ablation_suite_ce(bank, conditions, max_batch=3)  # forces chunking (4 cells)
    assert len(ces) == 2
    # baseline mean-CE must equal directly-computed CE from logits_batch
    ref = 0.0
    for ids in bank:
        lg = eng.logits(ids)
        tgt = torch.as_tensor(ids[1:], dtype=torch.long)
        lp = torch.log_softmax(lg[:-1], dim=-1)
        ref += float(-lp[torch.arange(len(tgt)), tgt].mean())
    assert abs(ces[0] - ref / len(bank)) < 1e-4
    assert ces[1] > ces[0]  # zeroing 400 early-ish channels should hurt CE
