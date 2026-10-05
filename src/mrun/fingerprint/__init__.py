"""mrun.fingerprint — embedding-provenance fingerprints: who is this checkpoint's substrate parent?

One tensor per model (the input embedding, HTTP-Range-read or mmap'd, streamed in row blocks),
reduced to a centered-Gram spectrum plus a cached top-k eigenbasis. The primary question is
always a LINEUP, never a threshold:

    from mrun.fingerprint import identify_parent, fingerprint, row_delta

    lineup = identify_parent("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", candidates=[
        "Qwen/Qwen2.5-Math-7B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-Coder-7B"])
    print(lineup.format())            # ranked with margins; bool(lineup) deliberately raises

    fp = fingerprint("Qwen/Qwen2.5-7B")          # n50/n90/n95/PR + top-256 eigenbasis, cached
    rd = row_delta("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", sample=20000)

Ported 2026-07-26 from ``experiments/2026-07-26-distillation-fingerprints`` (``fp_run.py``
spectrum leg, measured on 18 checkpoints; ``fp_pairs.py`` row leg, never run there and now
streamed) plus the range-reader from ``experiments/2026-07-25-embedding-concentration-regime``.
See :data:`mrun.fingerprint.spectrum.SCOPE` — SUBSTRATE lineage only; teacher attribution is out
of scope, and distilled-vs-instruction-tuned is not separable.
"""
from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = ("api", "reader", "spectrum", "lineup", "rows", "cli")
_API_NAMES = (
    "identify_parent", "fingerprint", "compare_models", "row_delta",
    "EmbeddingFingerprint", "ParentLineup", "RowDelta", "MatchedObjectError",
    "from_arrays", "subspace_alignment", "spectral_delta", "compare", "rank_candidates",
    "cache_dir", "cache_path", "SCOPE", "DEFAULT_K", "DEFAULT_LINEUP_K",
    "DEFAULT_AMBIGUITY_MARGIN",
)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _API_NAMES:
        return getattr(importlib.import_module(".api", __name__), name)
    raise AttributeError(f"module 'mrun.fingerprint' has no attribute {name!r}")


__all__ = [*_SUBMODULES, *_API_NAMES]
