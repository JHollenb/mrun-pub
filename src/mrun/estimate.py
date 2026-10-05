"""Estimate CPU/RAM cost of a run, before it runs.

Two estimators, designed to compose:

A. **First-principles RAM** (``estimate_memory``) — needs no history. Counts model
   parameters from the cached ``config.json`` (architecture-aware; falls back to the
   registry label) and applies the measured law ``rss_mb ~= weights_mb + overhead``.
   The slope/overhead were calibrated on real runs (see ``DEFAULT_OVERHEAD_MB``).

B. **History-backed** (``load_run_history`` + ``estimate_resources``) — every run writes
   ``resources`` into its ``manifest.json``. We read that growing dataset to (1) refit the
   memory law as more models are observed and (2) pull the nearest prior config's observed
   wall/CPU time, which is per-experiment and not predictable from model size alone.

Caveat: the memory law counts model weights + framework overhead. Activation memory
(batch x seq x hidden, plus attention seq^2) is excluded; it stays small at the seqlens
used here but will dominate if you push long contexts or large batches.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io import read_json
from .models import default_hub_root, hub_name, resolve_model
from .paths import models_root

log = logging.getLogger("mrun.estimate")

# Calibrated on two CPU runs (distilgpt2, qwen2.5-0.5b): a least-squares fit of
# rss_peak_mb ~= overhead + 1.0 * weights_mb gave overhead in [393, 431] MB. This is the
# fixed Python + torch + tokenizer floor; weights scale on top of it.
DEFAULT_OVERHEAD_MB = 410.0

# dtype name -> bytes per parameter.
_DTYPE_BYTES: dict[str, int] = {
    "float64": 8,
    "fp64": 8,
    "double": 8,
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int8": 1,
    "uint8": 1,
}


def dtype_bytes(dtype: str | int) -> int:
    """Bytes per parameter for a dtype name (``"bf16"``) or an explicit byte count."""
    if isinstance(dtype, int):
        return dtype
    key = str(dtype).lower().replace("torch.", "").strip()
    if key not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; known: {', '.join(sorted(_DTYPE_BYTES))}")
    return _DTYPE_BYTES[key]


# --------------------------------------------------------------------------- params


def _find_config_json(model: str) -> Path | None:
    """Locate a valid cached ``config.json`` without loading model weights."""
    spec = resolve_model(model)
    roots: list[Path] = []
    if spec.local_path:
        roots.append(Path(spec.local_path))
    hub = hub_name(spec)
    roots.extend([default_hub_root() / hub, models_root() / hub, models_root() / spec.name])
    for root in roots:
        if not root.exists():
            continue
        candidates = [root / "config.json", *sorted(root.glob("**/config.json"))]
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            # Hugging Face uses empty files under ``.no_exist`` as negative-cache
            # markers.  They are not model configs and must not poison estimation.
            if not candidate.is_file():
                continue
            try:
                decoded = read_json(candidate)
            except (OSError, TypeError, ValueError):
                continue
            if isinstance(decoded, dict):
                return candidate
    return None


def _params_from_dims(cfg: dict[str, Any]) -> int:
    """Transformer parameter count from config dims (verified vs distilgpt2 and qwen2.5)."""
    hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model") or 0)
    layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("n_layers") or 0)
    vocab = int(cfg.get("vocab_size") or 0)
    heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or 1)
    if not (hidden and layers and vocab):
        return 0
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    head_dim = int(cfg.get("head_dim") or (hidden // max(1, heads)))
    inter = int(cfg.get("intermediate_size") or cfg.get("n_inner") or (4 * hidden))
    gated = cfg.get("intermediate_size") is not None and cfg.get("num_key_value_heads") is not None

    embed = vocab * hidden
    attn = 2 * hidden * hidden + 2 * hidden * (kv_heads * head_dim)
    num_experts = int(cfg.get("num_experts") or 0)
    moe_inter = int(cfg.get("moe_intermediate_size") or 0)
    if num_experts and moe_inter:
        mlp = 3 * hidden * moe_inter * num_experts
        shared_inter = int(cfg.get("shared_expert_intermediate_size") or 0)
        mlp += 3 * hidden * shared_inter
        mlp += hidden * num_experts  # router
    else:
        mlp = (3 if gated else 2) * hidden * inter
    total = embed + layers * (attn + mlp)
    if cfg.get("tie_word_embeddings") is False:
        total += vocab * hidden  # untied lm_head is a second vocab x hidden matrix
    return int(total)


def _params_from_label(model: str) -> int:
    """Coarse fallback: parse the registry label, e.g. ``"82m"`` -> 82e6, ``"0.5b"`` -> 5e8."""
    try:
        label = resolve_model(model).label
    except ValueError:
        return 0
    match = re.match(
        r"\s*([\d.]+)\s*([mb])(?:\s*$|\s*[/_-])",
        label.lower(),
    )
    if not match:
        return 0
    scale = 1e6 if match.group(2) == "m" else 1e9
    return int(float(match.group(1)) * scale)


def model_param_count(model: str | Any) -> int:
    """Parameter count for a model name, spec, or a loaded torch model.

    Loaded model -> exact ``sum(p.numel())``. Name -> computed from cached ``config.json``
    when available, else parsed from the registry label. Returns 0 if nothing is known.
    """
    if hasattr(model, "parameters"):
        return int(sum(int(p.numel()) for p in model.parameters()))
    name = model.name if hasattr(model, "name") else str(model)
    config_path = _find_config_json(name)
    if config_path is not None:
        count = _params_from_dims(read_json(config_path))
        if count:
            return count
    return _params_from_label(name)


# --------------------------------------------------------------------- A2: activations

# What a task family holds on top of weights + activations. Initial heuristics; P2
# calibration (estimates rows carry est vs peak) refines these. `recorder` stays 1.0
# because its surcharge (attention maps for EVERY layer) is modeled analytically in
# estimate_activation_mb — checked against the one measured point (qwen2.5-0.5b
# physiology: est 8.3GB vs 6.4GB measured peak; conservative, not runaway).
TASK_RSS_FACTOR: dict[str, float] = {
    "forward": 1.0,
    "decode": 1.3,  # weight-decoder holds per-matrix workspaces
    # Recorder measured 2026-07-15: fp32 qwen0.5 @T=64 peaked 3936MB without the
    # attention leg (weights+overhead law said 2468) and 6370MB with it — the causal
    # dCE/ablation legs hold weight-matrix copies the analytic terms don't see.
    "recorder": 2.5,
    "train": 3.0,  # grads + optimizer state on top of weights
    "smoke": 1.5,  # smoke runs the same codepath, briefly — still needs the headroom
}

# Live-tensor multiple of one [B,T,hidden] fp32 activation during a forward (residual
# stream + attn workspace + mlp intermediate, coarse).
_HIDDEN_STREAM_FACTOR = 4


def _model_dims(model: str) -> dict[str, int]:
    """hidden/layers/heads/kv dims from the cached config.json (zeros when unknown)."""
    try:
        path = _find_config_json(model)
    except ValueError:  # not in the registry at all
        path = None
    cfg = read_json(path) if path is not None else {}
    hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model") or 0)
    heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or 0)
    return {
        "hidden": hidden,
        "layers": int(
            cfg.get("num_hidden_layers")
            or cfg.get("n_layer")
            or cfg.get("n_layers")
            or 0
        ),
        "heads": heads,
        "kv_heads": int(cfg.get("num_key_value_heads") or heads or 0),
        "head_dim": int(cfg.get("head_dim") or (hidden // heads if heads else 0)),
        "vocab": int(cfg.get("vocab_size") or 0),
    }


def estimate_activation_mb(
    model: str,
    *,
    seq_lens: list[int] | None = None,
    batch: int = 16,
    dtype: str = "float32",
    task: str = "forward",
    count_logits: bool = False,
) -> float:
    """Activation memory for one batched forward: hidden stream + attention maps + KV (+ optional
    LM-head logits).

    The old inline law (``B*T*hidden*4``) counted ONE live tensor; real runs hold several
    (residual stream, attention scores, MLP intermediates), and recorder runs materialize
    attention maps for every layer. Falls back to hidden=2048/layers=24/heads=16/vocab=32000 when
    the config is not cached (and LOGS that it did), so an unknown model still gets a non-trivial
    term.

    ``count_logits`` (opt-in) adds the LM-head LOGITS tensor ``[B, T, vocab]`` — the single
    largest live tensor at large vocab (Qwen's 152k logits alone can exceed a 0.5B model's
    weights). It is OFF by default because the fleet planner's task multipliers (``TASK_RSS_FACTOR``
    and the recorder attention term) were calibrated against MEASURED peaks that already absorb the
    logits empirically, so counting it there would double-count and over-reject hosts. A naive
    weights+activation consumer that does NOT have those calibrated multipliers (the manalysis
    run-harness reservation) passes ``count_logits=True`` to size for it explicitly.
    """
    dims = _model_dims(model)
    if not dims["hidden"]:
        # config.json not cached -> every dim below is a fallback guess; surface it (issue: a
        # silent hidden=2048/vocab=32000 substitution can badly mis-size an unknown model).
        log.warning(
            "estimate_activation_mb: no cached config for %r; using fallback dims "
            "(hidden=2048, layers=24, heads=16, vocab=32000) — cache config.json for accuracy",
            model,
        )
    hidden = dims["hidden"] or 2048
    layers = dims["layers"] or 24
    heads = dims["heads"] or 16
    kv_heads = dims["kv_heads"] or heads
    head_dim = dims["head_dim"] or (hidden // heads)
    tmax = max((int(x) for x in (seq_lens or [512])), default=512)
    nbytes = dtype_bytes(dtype)

    stream = batch * tmax * hidden * 4 * _HIDDEN_STREAM_FACTOR  # workspaces stay fp32
    # attention maps: recorder keeps them for every layer; a plain forward only has the
    # current layer's scores live.
    attn_layers = layers if task == "recorder" else 1
    attn = batch * heads * tmax * tmax * 4 * attn_layers
    kv = 2 * layers * batch * tmax * kv_heads * head_dim * nbytes
    total = stream + attn + kv
    if count_logits:
        # [B, T, vocab] in fp32 (loss / lm_head commonly upcast) — conservative bound.
        total += batch * tmax * (dims["vocab"] or 32000) * 4
    return round(total / 1e6, 1)


def task_rss_factor(task: str) -> float:
    return TASK_RSS_FACTOR.get(task, 1.0)


# --------------------------------------------------------------------------- A: memory


@dataclass(frozen=True)
class MemoryEstimate:
    model: str
    params: int
    dtype: str
    dtype_bytes: int
    weights_mb: float
    overhead_mb: float
    est_rss_mb: float
    param_source: str  # "config" | "label" | "exact" | "unknown"
    basis: str  # "first-principles" | "history-calibrated"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_memory(
    model: str | Any,
    *,
    dtype: str = "float32",
    overhead_mb: float = DEFAULT_OVERHEAD_MB,
    basis: str = "first-principles",
) -> MemoryEstimate:
    """Pre-run RAM estimate: ``weights + overhead``. No history needed."""
    params = model_param_count(model)
    if hasattr(model, "parameters"):
        source = "exact"
    elif params and _find_config_json(model.name if hasattr(model, "name") else str(model)):
        source = "config"
    elif params:
        source = "label"
    else:
        source = "unknown"
    nbytes = dtype_bytes(dtype)
    weights_mb = params * nbytes / 1e6
    name = model.name if hasattr(model, "name") else str(model)
    return MemoryEstimate(
        model=name,
        params=params,
        dtype=str(dtype),
        dtype_bytes=nbytes,
        weights_mb=round(weights_mb, 1),
        overhead_mb=round(overhead_mb, 1),
        est_rss_mb=round(weights_mb + overhead_mb, 1),
        param_source=source,
        basis=basis,
    )


# --------------------------------------------------------------------------- B: history


@dataclass(frozen=True)
class RunRecord:
    experiment: str
    run_id: str
    config: dict[str, Any]
    resources: dict[str, Any]
    path: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_run_history(outputs_root: str | Path = "outputs") -> list[RunRecord]:
    """Read every ``outputs/<experiment>/<run_id>/manifest.json`` into a list of records."""
    root = Path(outputs_root)
    records: list[RunRecord] = []
    for manifest in sorted(root.glob("*/*/manifest.json")):
        try:
            data = read_json(manifest)
        except (OSError, ValueError):
            continue
        if "resources" not in data:
            continue
        records.append(
            RunRecord(
                experiment=str(data.get("experiment", manifest.parent.parent.name)),
                run_id=str(data.get("run_id", manifest.parent.name)),
                config=dict(data.get("config", {})),
                resources=dict(data.get("resources", {})),
                path=str(manifest),
            )
        )
    return records


def calibrate_memory(history: list[RunRecord]) -> tuple[float, float] | None:
    """Refit ``rss_mb = overhead + slope * weights_mb`` from observed runs.

    Returns ``(overhead_mb, slope)`` once >=2 runs with distinct weight sizes are seen,
    else ``None`` (keep the first-principles defaults). ``slope`` should sit near 1.0.
    """
    points: list[tuple[float, float]] = []
    for rec in history:
        rss = rec.resources.get("rss_peak_mb")
        model = rec.config.get("model")
        if rss is None or not model:
            continue
        params = model_param_count(str(model))
        if not params:
            continue
        nbytes = dtype_bytes(str(rec.config.get("dtype", "float32")))
        points.append((params * nbytes / 1e6, float(rss)))
    xs = {round(x, 3) for x, _ in points}
    if len(points) < 2 or len(xs) < 2:
        return None
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return None
    slope = (n * sxy - sx * sy) / denom
    overhead = (sy - slope * sx) / n
    return round(overhead, 1), round(slope, 4)


def _config_similarity(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, list[str]]:
    """Fraction of shared keys with equal values; also the keys that differ."""
    keys = set(a) | set(b)
    keys.discard("run_id")
    if not keys:
        return 0.0, []
    differ = [k for k in sorted(keys) if a.get(k) != b.get(k)]
    return (len(keys) - len(differ)) / len(keys), differ


def nearest_run(
    config: dict[str, Any], *, name: str | None, history: list[RunRecord]
) -> tuple[RunRecord, float, list[str]] | None:
    """Most config-similar prior run (optionally restricted to experiment ``name``)."""
    pool = [r for r in history if name is None or r.experiment == name]
    if not pool:
        return None
    scored = [(r, *_config_similarity(config, r.config)) for r in pool]
    scored.sort(key=lambda t: t[1], reverse=True)
    best, score, differ = scored[0]
    return best, score, differ


# --------------------------------------------------------------------------- combined


def estimate_resources(
    config: dict[str, Any] | str,
    *,
    name: str | None = None,
    dtype: str | None = None,
    outputs_root: str | Path = "outputs",
) -> dict[str, Any]:
    """Best available CPU/RAM estimate: first-principles memory, refined by history.

    ``config`` may be a full experiment config dict (uses ``config["model"]``,
    ``config["dtype"]``) or a bare model name. History calibrates the memory law and
    supplies a wall/CPU estimate from the nearest prior config for experiment ``name``.
    """
    if isinstance(config, str):
        config = {"model": config}
    model = str(config.get("model", ""))
    resolved_dtype = dtype or str(config.get("dtype", "float32"))

    history = load_run_history(outputs_root)
    calibrated = calibrate_memory(history)
    if calibrated is not None:
        overhead_mb, slope = calibrated
        mem = estimate_memory(model, dtype=resolved_dtype, basis="history-calibrated")
        est_rss = round(slope * mem.weights_mb + overhead_mb, 1)
        memory = {
            **mem.as_dict(),
            "overhead_mb": overhead_mb,
            "slope": slope,
            "est_rss_mb": est_rss,
        }
    else:
        memory = estimate_memory(model, dtype=resolved_dtype).as_dict()

    notes: list[str] = []
    if memory["param_source"] in {"label", "unknown"}:
        src = memory["param_source"]
        notes.append(f"param count from {src} (coarse); cache config.json for accuracy")
    notes.append("activation memory excluded; valid at modest seqlen/batch")

    near = nearest_run(config, name=name, history=history)
    nearest_block: dict[str, Any] | None = None
    wall_s_est: float | None = None
    cpu_s_est: float | None = None
    if near is not None:
        rec, score, differ = near
        nearest_block = {
            "run_id": rec.run_id,
            "experiment": rec.experiment,
            "similarity": round(score, 3),
            "differing_keys": differ,
            "observed": {
                k: rec.resources.get(k)
                for k in ("wall_s", "cpu_s", "rss_peak_mb", "output_bytes")
            },
        }
        same_model = rec.config.get("model") == config.get("model")
        if not differ:  # exact config match -> observed numbers are the estimate
            wall_s_est = _as_float(rec.resources.get("wall_s"))
            cpu_s_est = _as_float(rec.resources.get("cpu_s"))
        elif rec.experiment == name and same_model:
            # same model + experiment, different knobs: same order of magnitude, not exact.
            wall_s_est = _as_float(rec.resources.get("wall_s"))
            cpu_s_est = _as_float(rec.resources.get("cpu_s"))
            notes.append(f"wall/cpu is a rough prior; knobs differ in: {', '.join(differ)}")
        elif rec.experiment == name:
            # wall/cpu scales with model size and pass count; no cross-model law from history.
            notes.append("wall/cpu unknown: nearest run uses a different model; not extrapolated")

    return {
        "model": model,
        "memory": memory,
        "est_rss_mb": memory["est_rss_mb"],
        "wall_s_estimate": wall_s_est,
        "cpu_s_estimate": cpu_s_est,
        "history": {
            "n_runs": len(history),
            "calibrated": calibrated is not None,
            "nearest": nearest_block,
        },
        "notes": notes,
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
