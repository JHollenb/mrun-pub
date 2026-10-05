"""Multi-fabric split-batch dispatcher: shard math (always) + single-fabric delegation (gated)."""
from __future__ import annotations

import os

import pytest

from mrun.engine.multifabric import shard_sizes
from mrun.models import store_name
from mrun.paths import stores_root


def test_shard_sizes_sum_to_n_and_track_weights():
    # measured mlx/ane knees -> proportional split that sums exactly to n
    assert shard_sizes(96, [4208.0, 3412.0]) == [53, 43]
    assert sum(shard_sizes(100, [1.0, 1.0, 1.0])) == 100
    assert sum(shard_sizes(97, [3.0, 1.0])) == 97
    # the heavier weight never gets the smaller shard
    a, b = shard_sizes(50, [9.0, 1.0])
    assert a > b


def test_shard_sizes_handles_zero_weight_and_empty():
    assert shard_sizes(10, [1.0, 0.0]) == [10, 0]
    assert shard_sizes(0, [1.0, 1.0]) == [0, 0]


def _store_available(model: str) -> bool:
    return (stores_root() / store_name(model)).joinpath("manifest.json").exists()


requires_model = pytest.mark.skipif(
    os.environ.get("MRUN_RUN_MODEL_TESTS", os.environ.get("MODEL_EXPERIMENTS_RUN_MODEL_TESTS")) != "1",
    reason="set MRUN_RUN_MODEL_TESTS=1 to run cached model tests",
)


@requires_model
def test_multifabric_single_fabric_delegates_to_paged():
    """With one fabric the dispatcher must reproduce that backend's logits_batch exactly (the
    reassembly path is identity), so the multi-fabric wrapper is a strict superset of paged."""
    from mrun.engine import open_engine

    model = "qwen2.5-0.5b"
    if not _store_available(model):
        pytest.skip(f"no paged store for {model} under {stores_root()}")
    with open_engine(model, backend="multifabric", backends=("paged",)) as mf:
        assert mf.backend_names == ["paged"]
        batch = mf.encode([
            "The capital of France is",
            "Two plus two equals",
            "Water is made of hydrogen and",
        ])
        mf_rows = mf.logits_batch(batch)
        paged_rows = mf.engines[0].logits_batch(batch)
        assert len(mf_rows) == len(paged_rows)
        for a, b in zip(mf_rows, paged_rows):
            assert a.shape == b.shape
            assert int(a[-1].argmax()) == int(b[-1].argmax())


_PROC_PROBES = [
    {"prompt": "The capital of France is", "correct": " Paris", "distractors": [" London", " Berlin"], "probe_id": "q0"},
    {"prompt": "Two plus two equals", "correct": " 4", "distractors": [" 5", " 3"], "probe_id": "q1"},
    {"prompt": "The opposite of hot is", "correct": " cold", "distractors": [" warm", " red"], "probe_id": "q2"},
    {"prompt": "The sky is", "correct": " blue", "distractors": [" green", " loud"], "probe_id": "q3"},
]


@requires_model
def test_process_multifabric_single_fabric_matches_inproc_paged():
    """The process dispatcher must round-trip probes through a spawned worker and reassemble rows
    that match the in-process paged reference exactly (winners + margins to float order) — proves
    the IPC/pickle path and the shard reassembly are correct."""
    from mrun.engine import open_engine
    from mrun.engine.multifabric_proc import ProcessMultiFabric

    model = "qwen2.5-0.5b"
    if not _store_available(model):
        pytest.skip(f"no paged store for {model} under {stores_root()}")
    with open_engine(model, backend="paged") as ref:
        ref_rows = ref.score_forced_choice_many(_PROC_PROBES)["rows"]
    with ProcessMultiFabric(model, backends=("paged",)) as pmf:
        assert pmf.backend_names == ["paged"]
        proc_rows = pmf.score_forced_choice_many(_PROC_PROBES)["rows"]
    assert [r["probe_id"] for r in proc_rows] == [r["probe_id"] for r in ref_rows]
    assert [r["correct_ranked_first"] for r in proc_rows] == [r["correct_ranked_first"] for r in ref_rows]
    max_dmargin = max(abs(a["margin"] - b["margin"]) for a, b in zip(proc_rows, ref_rows))
    assert max_dmargin < 1e-4, f"process-vs-inproc margin drift {max_dmargin}"
