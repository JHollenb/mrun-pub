"""Parity gate — check a non-HF backend against the HF reference before a canonical write.

Ported from discovery ``lab/engines.py``. A drift beyond ``tol`` means the fast path is NOT
canonical and its results must be labeled ``asserted`` rather than ``measured`` (Rule #2).
"""

from __future__ import annotations

from typing import Any


def assert_parity(
    model: str,
    workload: str = "forced_choice",
    *,
    backend: str,
    tol: float = 1e-4,
) -> dict[str, Any]:
    """Compare per-row forced-choice logprobs on a fixed tiny probe between ``backend``
    and HF; returns the measured drift and a ``canonical`` verdict."""
    from ..engine import open_engine

    probes = [
        {
            "id": f"p{i}",
            "category": "x",
            "prompt": f"Question {i}: two plus {i} equals",
            "correct": f" {i + 2}",
            "distractors": [f" {i + 3}"],
        }
        for i in range(3)
    ]
    ref = open_engine(model, backend="hf")
    alt = open_engine(model, backend=backend)
    try:
        ref_rows = ref.score_forced_choice_many(probes)["rows"]
        alt_rows = alt.score_forced_choice_many(probes)["rows"]
    finally:
        ref.close()
        alt.close()
    drift = max(
        abs(float(a["margin"]) - float(b["margin"]))
        for a, b in zip(ref_rows, alt_rows, strict=True)
    )
    return {
        "backend": backend,
        "model": model,
        "max_margin_drift": drift,
        "tol": tol,
        "canonical": bool(drift < tol),
    }
