"""Reusable paged-CUDA behavior harness — run a capability's forced-choice behavior leg on the
int8 PagedEngine (GPU, O(largest-matrix) RAM) instead of a dense model.

WHY THIS EXISTS. A capability's *behavior* is often a forced-choice score (true continuation vs
distractors) over a battery of prompts, optionally swept over a difficulty axis (nesting depth,
hop count) with scrambled / no-context null conditions. Forced-choice scoring is engine-clean —
`engine.score_forced_choice_many(probes)` needs no dense forward hooks — so it runs on the paged
engine at int8/CUDA. MEASURED 2026-07-17: Qwen3-14B coding bracket_match in ~55s on a 16GB card
(the model doesn't fit dense in bf16 at all); Qwen2.5-7B in ~29s vs ~5h dense.

This harness is CAPABILITY-AGNOSTIC and project-agnostic. The caller supplies a `build_batteries`
callback that, given the loaded engine, returns the depth->condition->probes structure (built with
the engine's own tokenizer). The harness loads the paged engine on CUDA, scores every condition,
aggregates the real component + lift over the null conditions, runs the degenerate-component guard,
stamps a code fingerprint, and returns a JSON-compatible record for the caller to persist.

Any project can use it: supply your probes, get a fast GPU behavior curve with provenance baked in.
See `discovery/experiments/understanding-gate/paged_coding_curve.py` / `paged_reasoning_curve.py`
for capability wrappers.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


# --------------------------------------------------------------------------- provenance + guard
def code_fingerprint(paths: list[str | Path]) -> dict:
    """Content sha256 of the ACTUAL source files that ran — immune to a recorded git_sha lying
    under rsync-drift (a fleet runs its own checkout; the stamp can disagree with the file). A curve
    assembled from tapes with different fingerprints is a code-path stitch and must be refused."""
    h = hashlib.sha256()
    used = []
    for p in paths:
        p = Path(p)
        if p.exists():
            h.update(p.read_bytes())
            used.append(p.name)
    return {"code_sha256": h.hexdigest()[:16], "files": used}


def _score_accuracy_chunked(eng, probes: list[dict], chunk_size: int) -> float | None:
    """Accuracy over ``probes`` via ``eng.score_forced_choice_many(argmax_only=True)``.

    ``argmax_only=True`` engages the SUBSET-LM_HEAD fast path for single-token candidates: the
    full-vocab softmax logZ cancels, so winner+margin come from candidate-only lm_head columns —
    the ``[B, Tmax, V]`` vocab tensor is NEVER built and the 544 MB unembedding is never streamed.
    That both fixes the OOM (long k-hop many-shot prompts × 152k vocab = 15+ GB otherwise) and lets
    one weight stream serve a large batch (the 17.5×-at-the-knee regime). We still chunk as a
    backstop for any battery whose candidates are multi-token (those fall through to the full
    batched path). Combines per-chunk accuracy weighted by n — equivalent to one big batch."""
    if not probes:
        return None
    num = 0.0
    den = 0
    for i in range(0, len(probes), max(1, chunk_size)):
        s = eng.score_forced_choice_many(probes[i:i + chunk_size], argmax_only=True)["summary"]
        n = int(s.get("n", 0))
        if n and s.get("accuracy") is not None:
            num += float(s["accuracy"]) * n
            den += n
    return round(num / den, 6) if den else None


def degenerate_flags(components: dict) -> dict:
    """Flag any component pinned at an exact boundary — a broken instrument, not data (the false
    "coding emergence" rode on a component that read exactly 0.000 from a decode bug). Exact 0.000
    -> floor; exact 1.000 -> ceiling. Surface these in the tape; never let a scalar hide a dead leg."""
    flags = {}
    for k, v in components.items():
        if isinstance(v, (int, float)):
            if v == 0.0:
                flags[k] = "degenerate_floor: exactly 0.000 — suspect broken measurement, not a real zero"
            elif v == 1.0:
                flags[k] = "degenerate_ceiling: exactly 1.000 — suspect saturation/cap"
    return flags


# --------------------------------------------------------------------------- the runner
def run_paged_forced_choice_curve(
    model: str,
    *,
    capability: str,
    component: str,
    build_batteries: Callable[[Any], dict[Any, dict[str, list[dict]]]],
    seed: int = 42,
    battery_tag: str = "paged",
    null_conditions: tuple[str, ...] = ("scrambled", "no_context"),
    chunk_size: int = 64,   # subset-lm_head path is low-VRAM (no vocab tensor); big batch = fewer
    #                         weight streams = faster. Drops automatically for multi-token batteries.

    fingerprint_paths: list[str | Path] | None = None,
    extra_config: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Score a forced-choice behavior curve on the int8 PagedEngine (CUDA) and emit a tape.

    ``build_batteries(engine)`` -> ``{level: {"real": [probe,...], "<null>": [...], ...}}`` where a
    probe is a dict ``{"prompt", "answer"/"correct", "distractors", ...}`` accepted by
    ``engine.score_forced_choice_many``. ``component`` is the tape key for the real accuracy
    (e.g. ``bracket_match_acc``). The real component is ``mean(real accuracy over levels)``; the
    lift is ``real - scrambled``. Returns the tape dict.
    """
    from .signature import load_paged_engine

    t0 = _now()
    eng, dev = load_paged_engine(model)
    if str(dev) != "cuda" and verbose:
        print(f"  WARN: paged engine on {dev} (not cuda) — correct but not the GPU fast path "
              f"(check GATHER_DEVICE_PAGED arch list + a free GPU)", flush=True)

    batteries = build_batteries(eng)  # caller builds probes with eng.tokenizer

    per_real: dict[Any, float] = {}
    per_null: dict[str, dict[Any, float]] = {n: {} for n in null_conditions}
    for level, conds in batteries.items():
        real = conds.get("real")
        if not real:
            continue
        per_real[level] = _score_accuracy_chunked(eng, real, chunk_size)
        for n in null_conditions:
            probes = conds.get(n)
            if probes:
                per_null[n][level] = _score_accuracy_chunked(eng, probes, chunk_size)

    def _mean(d):
        vals = [v for v in d.values() if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    comp_val = _mean(per_real) if per_real else 0.0
    null_means = {f"{component}_{n}_null": _mean(per_null[n]) for n in null_conditions}
    scr = null_means.get(f"{component}_scrambled_null")
    lift = round(comp_val - scr, 4) if (comp_val is not None and scr is not None) else None

    components = {component: comp_val, **null_means}
    flags = degenerate_flags({component: comp_val})
    if flags and verbose:
        print(f"  [GUARD] DEGENERATE {list(flags)} — broken instrument, not data", flush=True)

    fp = code_fingerprint(fingerprint_paths or [])
    tape = {
        "instrument": "paged-behavior-forced-choice-v1",
        "model": model, "model_name": model, "backend": "paged",
        "capability": capability,
        "coverage": {
            "behavior_accuracy": comp_val,
            "behavior_accuracy_clean": comp_val if not flags else None,
            "behavior_component_flags": flags,
            "behavior_components": components,
            f"{component}_lift_over_scrambled": lift,
            "accuracy_by_level": {str(k): per_real.get(k) for k in per_real},
        },
        "config": {"seed": seed, "device": str(dev), "dtype": "int8", "backend": "paged",
                   "code_fingerprint": fp, "battery": {"tag": battery_tag},
                   **(extra_config or {})},
        "tags": {"battery": battery_tag, "backend": "paged"},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(_now() - t0, 1),
    }
    if verbose:
        print(f"PAGED {model} {capability}/{component} seed={seed}: {comp_val} "
              f"lift={lift} ({dev}, {tape['elapsed_s']}s)", flush=True)

    return tape


def run_paged_behavior_legs(
    model: str,
    *,
    capability: str,
    compute_fn: Callable[[Any, int], dict],
    seed: int = 42,
    battery_tag: str = "paged",
    primary_component: str | None = None,
    fingerprint_paths: list[str | Path] | None = None,
    extra_config: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Generic paged behavior runner for probe/generate legs that are NOT forced-choice (metacognition
    AUROC probes, emotion valence probes, instruction-following via generate). ``compute_fn(engine,
    seed) -> {component: value}`` runs the behavior legs on the paged engine (via the engine's
    hidden_states()/generate()/score_forced_choice_many, no dense hooks) and returns the scalar
    components. The harness loads the engine on CUDA, runs the degenerate-component guard, stamps a
    code fingerprint, and emits a tape — same provenance discipline as the forced-choice curve."""
    from .signature import load_paged_engine
    t0 = _now()
    eng, dev = load_paged_engine(model)
    if str(dev) != "cuda" and verbose:
        print(f"  WARN: paged engine on {dev} (not cuda)", flush=True)
    components = compute_fn(eng, seed)
    flags = degenerate_flags(components)
    if flags and verbose:
        print(f"  [GUARD] DEGENERATE {list(flags)} — broken instrument, not data", flush=True)
    prim = primary_component or next(iter(components), None)
    prim_val = components.get(prim) if prim else None
    fp = code_fingerprint(fingerprint_paths or [])
    tape = {
        "instrument": "paged-behavior-legs-v1",
        "model": model, "model_name": model, "backend": "paged", "capability": capability,
        "coverage": {
            "behavior_accuracy": prim_val,
            "behavior_accuracy_clean": prim_val if not flags else None,
            "behavior_component_flags": flags,
            "behavior_components": components,
        },
        "config": {"seed": seed, "device": str(dev), "dtype": "int8", "backend": "paged",
                   "code_fingerprint": fp, "battery": {"tag": battery_tag}, **(extra_config or {})},
        "tags": {"battery": battery_tag, "backend": "paged"},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(_now() - t0, 1),
    }
    if verbose:
        print(f"PAGED {model} {capability}: {components} ({dev}, {tape['elapsed_s']}s)", flush=True)
    return tape


# --------------------------------------------------------------------------- io helpers
def _now() -> float:
    import time
    return time.time()
