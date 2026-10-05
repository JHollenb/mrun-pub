"""HFEngine subset-lm_head scoring — winner+margin must match the full-vocab path.

Model-gated (distilgpt2, ~350MB, cpu): skips when the snapshot isn't cached locally.
The subset path's whole claim is EXACTNESS for single-token forced choice (the shared
softmax -logZ cancels), so the gate is winner flags identical + margins equal to fp32
reduction noise.
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


PROBES = [
    {"probe_id": "p0", "prompt": "The capital of France is",
     "correct": " Paris", "distractors": [" London", " Berlin"]},
    {"probe_id": "p1", "prompt": "Two plus two equals",
     "correct": " four", "distractors": [" five"]},
    {"probe_id": "p2", "prompt": "The sky on a clear day is",
     "correct": " blue", "distractors": [" green"]},
]


def test_subset_head_matches_full_path_winners_and_margins():
    eng = _engine()
    # Every candidate must be single-token under this tokenizer or the subset path
    # (correctly) refuses — the test is only meaningful on the cancelling case.
    assert eng._all_single_token(PROBES), "fixture candidates must be single-token"

    full = eng.score_forced_choice_many(PROBES)                      # full-vocab path
    subset = eng.score_forced_choice_many(PROBES, argmax_only=True)  # subset path
    assert "subset lm_head" in subset.get("note", "")

    for f_row, s_row in zip(full["rows"], subset["rows"]):
        assert f_row["correct_ranked_first"] == s_row["correct_ranked_first"]
        # logZ cancels within a probe: margins agree to fp32 reduction noise. (The full
        # path margin is in avg-LOGPROB units over the 1-token answer = logit - logZ;
        # the subset margin is raw-logit — the shared logZ drops out of the DIFFERENCE.)
        assert abs(f_row["margin"] - s_row["margin"]) < 1e-3


def test_hidden_last_batch_shape_and_head_rows():
    eng = _engine()
    ids = [np.asarray(eng.encode(["hello world"], add_special_tokens=False)[0]),
           np.asarray(eng.encode(["a much longer prompt for padding"], add_special_tokens=False)[0])]
    h = eng.hidden_last_batch(ids)
    assert h.shape == (2, eng.hidden)
    rows = eng.lm_head_rows([0, 5, 17])
    assert rows.shape == (3, eng.hidden)


def test_multi_token_candidates_raise():
    eng = _engine()
    bad = [{"prompt": "x", "correct": " unquestionably multitoken answer", "distractors": [" y"]}]
    with pytest.raises(ValueError, match="single-token"):
        eng.score_forced_choice_argmax_subset(bad)
