"""MLX + ANE backend parity (Apple-only, opt-in).

Gated on ``MODEL_EXPERIMENTS_RUN_MODEL_TESTS=1`` plus the backend's library and a built store.
Both skip cleanly off-Apple / when the dependency is absent — correctness never depends on them
(the engine falls back to paged), so these only run where the accelerator exists.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from mrun.models import store_name
from mrun.paths import stores_root

requires_model = pytest.mark.skipif(
    os.environ.get("MRUN_RUN_MODEL_TESTS", os.environ.get("MODEL_EXPERIMENTS_RUN_MODEL_TESTS")) != "1",
    reason="set MRUN_RUN_MODEL_TESTS=1 to run cached model tests",
)


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _store_available(model: str) -> bool:
    return (stores_root() / store_name(model)).joinpath("manifest.json").exists()


REAL_PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
]


@requires_model
@pytest.mark.skipif(not _have("mlx"), reason="mlx not installed")
def test_mlx_argmax_matches_hf():
    from mrun.engine import open_engine

    model = "qwen2.5-0.5b"
    with open_engine(model, backend="hf") as hf, open_engine(model, backend="mlx") as mlx:
        for p in REAL_PROMPTS:
            ids = hf.encode([p])[0]
            assert int(mlx.logits(ids)[-1].argmax()) == int(hf.logits(ids)[-1].argmax())


@requires_model
@pytest.mark.skipif(not _have("coremltools"), reason="coremltools not installed")
def test_ane_argmax_matches_paged_on_real_text():
    """ANE fp16 is argmax-exact vs paged on REAL text (the gate). Random tokens are NOT a valid
    parity input (fp16 error compounds off-distribution) and are deliberately not tested here."""
    from mrun.engine import open_engine

    model = "qwen2.5-0.5b"
    if not _store_available(model):
        pytest.skip(f"no paged store for {model} under {stores_root()}")
    with open_engine(model, backend="paged") as paged, open_engine(model, backend="ane") as ane:
        if not ane._ane_ok:
            pytest.skip("ANE not available for this arch")
        tok = paged.tokenizer
        L = min(len(tok(p)["input_ids"]) for p in REAL_PROMPTS)
        ids = [np.asarray(tok(p)["input_ids"][:L], dtype=np.int64) for p in REAL_PROMPTS]
        ref = paged.logits_batch(ids)
        got = ane.logits_batch(ids)  # first call compiles (~20 s)
        match = sum(int(g[-1].argmax() == r[-1].argmax()) for g, r in zip(got, ref))
        corr = float(np.mean([np.corrcoef(g[-1].numpy(), r[-1].numpy())[0, 1] for g, r in zip(got, ref)]))
        assert match == len(ids), f"ANE argmax parity {match}/{len(ids)}"
        assert corr > 0.999, f"ANE logit corr {corr}"


def test_mlx_q4_resolution_error_names_convert_command(tmp_path, monkeypatch):
    """Without an export and without autoconvert, mlx-q4 resolution fails loudly with the
    exact convert command (deliberate-export policy) — no model or mlx needed."""
    pytest.importorskip("mlx")
    monkeypatch.setenv("MRUN_MLX_Q4_ROOT", str(tmp_path))
    monkeypatch.delenv("MRUN_MLX_Q4_AUTOCONVERT", raising=False)
    from mrun.engine.mlx import _resolve_mlx_q4_path

    with pytest.raises(FileNotFoundError, match="q-group-size 64"):
        _resolve_mlx_q4_path("qwen2.5-0.5b")


@requires_model
@pytest.mark.skipif(not _have("mlx"), reason="mlx not installed")
def test_mlx_q4_fc_winners_match_fp32_reference():
    """mlx-q4 gate is CAPABILITY (forced-choice winners), not per-position argmax: int4 flips
    low-margin positions at 0.5B while FC winners stay aligned (2026-07-23 PoC: 23/24)."""
    if os.environ.get("MRUN_MLX_Q4_ROOT") is None:
        pytest.skip("set MRUN_MLX_Q4_ROOT to a dir containing qwen2.5-0.5b-q4g64")
    from mrun.engine import open_engine

    probes = [
        {"prompt": p, "answers": [a, b], "correct": a}
        for p, a, b in [
            ("The capital of France is", " Paris", " London"),
            ("The chemical symbol for gold is", " Au", " Ag"),
            ("The largest planet in the solar system is", " Jupiter", " Mars"),
            ("The sun rises in the", " east", " west"),
        ]
    ]
    with open_engine("qwen2.5-0.5b", backend="hf") as hf:
        ref = hf.score_forced_choice_many(probes)
    with open_engine("qwen2.5-0.5b", backend="mlx-q4") as q4:
        got = q4.score_forced_choice_many(probes)
    ref_w = [r["correct_ranked_first"] for r in ref["rows"]]
    got_w = [r["correct_ranked_first"] for r in got["rows"]]
    assert got_w == ref_w
