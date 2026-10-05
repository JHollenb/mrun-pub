"""Prefix-KV forced-choice scoring — parity vs the flat batched path.

Model-gated (distilgpt2, ~350MB, cpu): skips when the snapshot isn't cached locally.
The prefix-KV path runs the SAME computation (cached attention is exact), so every
candidate's avg_logprob must match the flat path to reduction noise — winners, margins,
and the calibrated logprobs themselves.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")


def _engine():
    from mrun.engine import open_engine

    try:
        return open_engine("distilgpt2", backend="hf", device="cpu")
    except Exception as e:  # noqa: BLE001 — no snapshot / offline
        pytest.skip(f"distilgpt2 not loadable locally: {e}")


# multi-token candidates + prompts long enough that the savings heuristic (>25% fewer
# tokens) selects the prefix path: flat = sum n_cand*(P+A), kv = sum P + n_cand*A.
PROBES = [
    {"probe_id": "p0",
     "prompt": "After weeks of debate the city council finally voted to rename the harbor "
               "bridge after",
     "correct": " the retired ferry captain",
     "distractors": [" a famous sea monster", " its original architect"]},
    {"probe_id": "p1",
     "prompt": "The recipe warns that if the caramel begins to smoke you should immediately",
     "correct": " remove the pan from the heat", "distractors": [" add more sugar quickly"]},
    {"probe_id": "p2",
     "prompt": "Grandmother kept every letter from the war years inside",
     "correct": " an old biscuit tin",
     "distractors": [" the neighbour's car", " a hollowed-out dictionary"]},
]


def test_prefix_kv_selected_and_matches_flat_path():
    eng = _engine()
    flat = eng._score_forced_choice_batched(PROBES)
    kv = eng.score_forced_choice_prefix_kv(PROBES)
    assert "prefix-kv" in kv.get("note", ""), "heuristic should select the prefix path here"
    for f_row, k_row in zip(flat["rows"], kv["rows"], strict=True):
        assert f_row["correct_ranked_first"] == k_row["correct_ranked_first"]
        assert abs(f_row["correct_avg_logprob"] - k_row["correct_avg_logprob"]) < 1e-4
        assert abs(f_row["margin"] - k_row["margin"]) < 1e-4
    assert flat["summary"]["accuracy"] == kv["summary"]["accuracy"]


def test_dispatch_uses_prefix_kv_and_env_disables():
    eng = _engine()
    via_many = eng.score_forced_choice_many(PROBES)
    assert "prefix-kv" in via_many.get("note", "")
    os.environ["MRUN_PREFIX_KV"] = "0"
    try:
        off = eng.score_forced_choice_many(PROBES)
        assert "prefix-kv" not in off.get("note", "")
        for a, b in zip(via_many["rows"], off["rows"], strict=True):
            assert abs(a["correct_avg_logprob"] - b["correct_avg_logprob"]) < 1e-4
    finally:
        os.environ.pop("MRUN_PREFIX_KV", None)


def test_short_prompt_falls_back_to_flat():
    eng = _engine()
    probes = [{"probe_id": "s0", "prompt": "Hi", "correct": " there", "distractors": [" now"]}]
    res = eng.score_forced_choice_prefix_kv(probes)
    assert "prefix-kv" not in res.get("note", "")  # savings <25% -> flat path
    assert len(res["rows"]) == 1


def test_prefix_kv_chunked_matches_unchunked(monkeypatch):
    """Chunking at the batch cap (the 15.07 GiB lm_head fix) must not change results:
    force a tiny cap so 6 probes chunk 3 ways, compare against one-shot rows."""
    eng = _engine()
    probes = PROBES * 2  # 6+ probes, stable probe_ids from the fixtures
    one = eng.score_forced_choice_prefix_kv(probes)
    monkeypatch.setenv("GATHER_MAX_BATCH", "2")
    chunked = eng.score_forced_choice_prefix_kv(probes)
    assert "chunked" in chunked["note"]
    for a, b in zip(one["rows"], chunked["rows"], strict=True):
        assert a["correct_ranked_first"] == b["correct_ranked_first"]
        assert abs(a["margin"] - b["margin"]) < 1e-4
