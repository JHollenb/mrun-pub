"""Paged backend parity + store-build correctness.

The pure-numpy quant test runs always; the engine parity test is gated on a built store
(``MODEL_EXPERIMENTS_RUN_MODEL_TESTS=1`` + a reachable ``MODEL_EXPERIMENTS_STORES_ROOT``).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from mrun.engine.kernels.qstore_build import _quant_row_int8
from mrun.models import store_name
from mrun.paths import stores_root


def test_quant_row_int8_roundtrip_is_per_row_symmetric():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 16)).astype(np.float32)
    q, scale = _quant_row_int8(W)
    assert q.dtype == np.int8 and q.shape == W.shape
    assert scale.shape == (8,)
    # dequant error is bounded by half a quantization step per row (symmetric int8, 127 levels)
    deq = q.astype(np.float32) * scale[:, None]
    per_row_step = np.abs(W).max(axis=1) / 127.0
    assert np.all(np.abs(W - deq) <= per_row_step[:, None] + 1e-6)


def test_quant_row_int8_zero_row_uses_unit_scale():
    W = np.zeros((2, 4), dtype=np.float32)
    q, scale = _quant_row_int8(W)
    assert np.all(q == 0) and np.all(scale == 1.0)


def _store_available(model: str) -> bool:
    return (stores_root() / store_name(model)).joinpath("manifest.json").exists()


requires_model = pytest.mark.skipif(
    os.environ.get("MRUN_RUN_MODEL_TESTS", os.environ.get("MODEL_EXPERIMENTS_RUN_MODEL_TESTS")) != "1",
    reason="set MRUN_RUN_MODEL_TESTS=1 to run cached model tests",
)


@requires_model
def test_open_engine_paged_missing_store_raises_clear_error():
    from mrun.engine import open_engine

    with pytest.raises(FileNotFoundError, match="build-store"):
        open_engine("qwen2.5-0.5b", backend="paged", stores_dir="/tmp/model-experiments-no-store")


@requires_model
def test_paged_vs_hf_int8_mechanics_parity():
    """Paged int8 == HF with matching per-output-channel int8 fake-quant, up to float
    reduction order (argmax-exact, max|Δlogit| ~1e-4). Catches every RoPE/GQA/RMSNorm bug."""
    import torch

    from mrun.engine import open_engine
    from mrun.models import load_hf_model, load_tokenizer

    model = "qwen2.5-0.5b"
    if not _store_available(model):
        pytest.skip(f"no paged store for {model} under {stores_root()}")

    @torch.no_grad()
    def fake_quant_int8(m):
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                W = mod.weight.data.float()
                sc = W.abs().amax(dim=1, keepdim=True) / 127.0
                sc = torch.where(sc == 0, torch.ones_like(sc), sc)
                mod.weight.data = (torch.round(W / sc).clamp(-127, 127) * sc).to(mod.weight.dtype)

    tok = load_tokenizer(model)
    hf = load_hf_model(model)
    fake_quant_int8(hf)
    prompts = [
        "The capital of France is",
        "In a short proof, the key idea is",
        "A reliable experiment should",
        "When the model answers carefully, it",
    ]
    with open_engine(model, backend="paged") as paged:
        match, max_abs = 0, 0.0
        for p in prompts:
            ids = np.asarray(tok(p, add_special_tokens=False)["input_ids"], dtype=np.int64)
            pl = paged.logits(ids)[-1]
            hl = hf(torch.from_numpy(ids)[None]).logits[0, -1].float()
            match += int(pl.argmax() == hl.argmax())
            max_abs = max(max_abs, float((pl - hl).abs().max()))
    assert match == len(prompts), f"argmax parity {match}/{len(prompts)}"
    assert max_abs < 0.5, f"max|Δlogit|={max_abs}"


@requires_model
def test_paged_batched_equals_scalar():
    """Each batched row equals the standalone scalar paged_logits (argmax-exact)."""
    from mrun.engine import open_engine

    model = "qwen2.5-0.5b"
    if not _store_available(model):
        pytest.skip(f"no paged store for {model} under {stores_root()}")
    with open_engine(model, backend="paged") as paged:
        batch = paged.encode([
            "The capital of France is",
            "Two plus two equals",
            "Water is made of hydrogen and",
        ])
        rows = paged.logits_batch(batch)
        for ids, row in zip(batch, rows):
            scalar = paged.logits(ids)
            assert row.shape == scalar.shape
            assert int(row[-1].argmax()) == int(scalar[-1].argmax())


# forced-choice probes whose every candidate is a single token under the Qwen tokenizer.
_FORCED_CHOICE_PROBES = (
    {"prompt": "The capital of France is", "correct": " Paris",
     "distractors": [" London", " Berlin", " Madrid"], "probe_id": "fc0"},
    {"prompt": "The opposite of hot is", "correct": " cold",
     "distractors": [" warm", " fast", " red"], "probe_id": "fc1"},
    {"prompt": "The sky is", "correct": " blue",
     "distractors": [" green", " loud", " square"], "probe_id": "fc2"},
    {"prompt": "Two plus two equals", "correct": " 4",
     "distractors": [" 5", " 3", " 7"], "probe_id": "fc3"},
)


def _winners_margins(result):
    return [(r["correct_ranked_first"], float(r["margin"])) for r in result["rows"]]


@requires_model
def test_score_forced_choice_batched_equals_scalar():
    """The batched score_forced_choice_many fast path matches the scalar loop: winners identical,
    margins equal to float-reduction order. This is the ~16x scoring lever — it must not change the
    answer, only amortize the weight stream."""
    from mrun.engine import open_engine

    model = "qwen2.5-0.5b"
    if not _store_available(model):
        pytest.skip(f"no paged store for {model} under {stores_root()}")
    probes = [dict(p) for p in _FORCED_CHOICE_PROBES]
    with open_engine(model, backend="paged") as paged:
        assert paged.supports_batch, "paged should fuse qwen2 batches"
        batched = paged.score_forced_choice_many(probes)
        paged.supports_batch = False                 # instance attr forces the scalar loop
        scalar = paged.score_forced_choice_many(probes)
    bw, sw = _winners_margins(batched), _winners_margins(scalar)
    assert [w for w, _ in bw] == [w for w, _ in sw], "winner mismatch batched vs scalar"
    max_dmargin = max(abs(b - s) for (_, b), (_, s) in zip(bw, sw))
    assert max_dmargin < 1.5e-4, f"max|Δmargin| batched-vs-scalar = {max_dmargin}"


@requires_model
def test_subset_lm_head_equals_full_softmax():
    """Candidate-subset lm_head gives the SAME winner and margin as full-softmax scoring for
    single-token forced choice (the shared -logZ cancels), at a strictly lower working-set floor."""
    from mrun.engine import open_engine

    model = "qwen2.5-0.5b"
    if not _store_available(model):
        pytest.skip(f"no paged store for {model} under {stores_root()}")
    with open_engine(model, backend="paged") as eng:
        # keep only probes whose every candidate is single-token under this tokenizer (the case
        # where logZ cancels); self-adjusts to the model like experiments/scoring_subset.
        def _single(c):
            return len(eng.encode([c], add_special_tokens=False)[0]) == 1
        probes = [dict(p) for p in _FORCED_CHOICE_PROBES
                  if all(_single(c) for c in (p["correct"], *p["distractors"]))]
        if not probes:
            pytest.skip("no probe has all single-token candidates under this tokenizer")
        full = eng.score_forced_choice_many(probes)
        floor_full = eng.working_set_mb
    with open_engine(model, backend="paged") as eng2:
        # exercise the public opt-in surface (argmax_only routes single-token probes to the subset
        # path), not just the backend method directly.
        sub = eng2.score_forced_choice_many(probes, argmax_only=True)
        floor_sub = eng2.working_set_mb

    fw, sw = _winners_margins(full), _winners_margins(sub)
    assert [w for w, _ in fw] == [w for w, _ in sw], "winner mismatch subset vs full"
    max_dmargin = max(abs(f - s) for (_, f), (_, s) in zip(fw, sw))
    assert max_dmargin < 1.5e-4, f"max|Δmargin| subset-vs-full = {max_dmargin}"
    assert floor_sub < floor_full, f"subset floor {floor_sub} not below full {floor_full}"
