"""Parity gate for paged_lora: dense fp32 base vs paged int8 base, same LoRA / seed / batches.

Asserts the two properties the gate exists to prove, at a small (fast) config:
  * OUTCOME parity  — held-out forced-choice accuracy of the paged run matches the dense run
    within the run-to-run seed spread (int8 must not change *what* it learns).
  * MEMORY contract — the paged trainer's working set stays O(largest matrix) (the lm_head row
    floor), NOT O(model): far below the dense fp32 parameter footprint.
  * GEOMETRY ratio  — the int8-vs-dense adapter-subspace angle is recorded against the dense
    seed-noise floor. The ratio is reported (and loosely bounded) but NOT hard-gated tight: a
    "capability-only" split verdict is an ACCEPTED outcome, so the test asserts only that the gate
    ran and produced a finite ratio, plus the outcome+memory invariants.

Gated on a built qwen store (``MRUN_RUN_MODEL_TESTS=1`` + reachable store root).
"""
from __future__ import annotations

import os

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
def test_paged_lora_parity_outcome_and_memory():
    from mrun.training.paged_lora_parity import run_parity

    if not _store_available(MODEL):
        pytest.skip(f"no paged store for {MODEL}")
    # small + fast: enough steps for a non-trivial delta, cheap enough for CI-gated local.
    res = run_parity(MODEL, rank=8, targets=("q", "v"), n_train=48, n_eval=16,
                     steps=12, batch=6, verbose=False)

    # OUTCOME: int8 must not move the held-out outcome by more than the dense seed spread.
    assert res["outcome"]["parity"], res["outcome"]

    # MEMORY contract: paged working set is O(largest matrix), well under the dense fp32 params.
    mem = res["memory_mb"]
    assert mem["paged_working_set"] < 64.0, mem            # lm_head row floor ~29 MB on 0.5b
    assert mem["paged_working_set"] < 0.1 * mem["dense_param_fp32"], mem

    # GEOMETRY: the gate produced a finite, non-degenerate divergence ratio (verdict may be
    # geometry-clean OR capability-only — both are acceptable, so this is a sanity bound only).
    ratio = res["geometry"]["ratio_test_over_floor"]
    assert ratio == ratio and 0.0 <= ratio < 50.0, res["geometry"]  # finite, not absurd

    assert res["verdict"] in ("geometry+capability", "capability-only"), res["verdict"]
