"""One engine report per run: what actually executed, and what it cost.

Motivation (issues I47/I48/I49). Engines already know backend, dtype, device, compiled
shapes, cache hits, fallback reasons and whether a batch API silently degraded to a
per-row loop — but each exposes a different subset through a differently-named method
(`execution_evidence`, `runtime_stats`, `runtime_report`, `cache_stats`) and most runs
persist none of it. Two failures in one day went unseen for exactly this reason: an "ane"
backend running 100% paged because coremltools was absent, and a Core ML path placing 98%
of its operations on the GPU rather than the Neural Engine.

This module does not add new measurement. It collects what engines already report into a
single stable shape, adds timing/throughput, and renders it as a graph so a slow or
mis-placed run is diagnosable from the artifact instead of by rerunning under a profiler.

Compute units: device-seconds are NOT comparable across fabrics, so summing them is a
category error. Each device carries an explicit weight and the weights are declared in
the report — a number nobody can trace is worse than no number.
"""

from __future__ import annotations

import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Relative throughput weights, paged-CPU = 1.0. Anchored to the measured standalone knees
# on one (model, T=32) point — mlx 4208 / coreml 3412 / paged 826 tok/s — and to the
# 2026-07-24 Core ML measurement (3.9-4.2x paged). These are COARSE and exist so that a
# 4080-second and an M3-second are not silently added; they are not a cost model.
DEVICE_WEIGHTS: dict[str, float] = {
    "cuda": 8.0,
    "mps": 5.0,
    "coreml": 4.0,
    "mlx": 5.0,
    "cpu": 1.0,
}

# A backend's `device` attribute does NOT identify its fabric: ANEPagedEngine subclasses
# PagedEngine and so reports device="cpu" while executing through Core ML. Weighting it as
# CPU understated its compute units by 4x in the first smoke run. Map backend name -> weight
# key explicitly rather than substring-matching whatever `device` happens to say.
BACKEND_FABRIC: dict[str, str] = {
    "ane": "coreml",
    "coreml": "coreml",
    "mlx": "mlx",
    "dense-qstore-cuda": "cuda",
    "cuda-source-int8": "cuda",
    "cuda-source-int8-compact-head": "cuda",
    "olmoe-cuda": "cuda",
    "qwen3-moe-cuda": "cuda",
    "moe-qstore-cuda": "cuda",
}


def _proc_stats() -> dict[str, Any]:
    """Process-tree CPU/RSS. Same psutil walk the agent guard already performs."""
    try:
        import psutil

        p = psutil.Process()
        cpu = p.cpu_times()
        rss = p.memory_info().rss
        for child in p.children(recursive=True):
            try:
                c = child.cpu_times()
                cpu_user, cpu_sys = cpu.user + c.user, cpu.system + c.system
                cpu = type(cpu)(cpu_user, cpu_sys, *tuple(cpu)[2:])
                rss += child.memory_info().rss
            except Exception:  # noqa: BLE001 — child died mid-walk
                continue
        return {"cpu_seconds": round(cpu.user + cpu.system, 3), "peak_rss_mb": round(rss / 1e6, 1)}
    except Exception:  # noqa: BLE001 — psutil absent: report nothing rather than guess
        return {}


def _gpu_stats() -> dict[str, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "gpu_util_pct": int(util.gpu),
            "gpu_mem_used_mb": round(mem.used / 1e6, 1),
            "gpu_name": pynvml.nvmlDeviceGetName(h),
        }
    except Exception:  # noqa: BLE001 — no cuda / no pynvml
        return {}


def engine_facts(engine: Any) -> dict[str, Any]:
    """Everything an engine already knows, under one shape.

    Reads whichever of the four historical reporting methods the engine implements; a
    method that raises is recorded as an error rather than swallowed, because a reporting
    surface that quietly returns nothing is how the coremltools-absent case stayed hidden.
    """
    facts: dict[str, Any] = {
        "backend": getattr(engine, "backend", "?"),
        "model": getattr(engine, "name", "?"),
        "arch": getattr(engine, "arch", "?"),
        "device": str(getattr(engine, "device", "cpu")),
        "n_layer": getattr(engine, "n_layer", None),
        "hidden": getattr(engine, "hidden", None),
        "supports_batch": bool(getattr(engine, "supports_batch", False)),
        "numerical_contract": getattr(engine, "numerical_contract", None),
        "reported_fabric": getattr(engine, "reported_fabric", None),
        "margin_floor": getattr(engine, "margin_floor", 0.0) or 0.0,
        "working_set_mb": getattr(engine, "working_set_mb", None),
    }
    for name in ("execution_evidence", "runtime_stats", "runtime_report", "cache_stats"):
        fn = getattr(engine, name, None)
        if not callable(fn):
            continue
        try:
            value = fn()
            if value:
                facts[name] = value
        except Exception as e:  # noqa: BLE001
            facts[name] = {"error": f"{type(e).__name__}: {e}"}
    # Degradation is the single most important field: a batch API that ran as a per-row
    # loop looks identical to a slow run unless it is stated outright.
    stats = getattr(engine, "scalar_fallback_stats", None)
    if callable(stats):
        fb = stats()
        facts["scalar_fallbacks"] = fb
        facts["degraded_to_scalar"] = bool(fb)
    # Which scoring path actually served the probes, and how fast. Recorded at the scoring
    # boundary so ORDINARY runs populate it — without this, stage timing only existed where a
    # benchmark opted in, i.e. never in the runs whose speed anyone actually wonders about.
    scoring = getattr(engine, "scoring_stats", None)
    if callable(scoring):
        sc = scoring()
        if sc:
            facts["scoring_paths"] = sc
    return facts


class EngineReport:
    """Accumulates stages, then emits JSON + a rendered graph.

    Usage:
        rep = EngineReport(engine, run_id="...")
        with rep.stage("prefill", tokens=B*T):
            engine.logits_batch(ids)
        rep.save(Path("results"))
    """

    def __init__(
        self,
        engine: Any,
        *,
        run_id: str | None = None,
        note: str | None = None,
        started_at: float | None = None,
        start_proc: dict[str, Any] | None = None,
    ):
        self.engine = engine
        self.facts = engine_facts(engine)
        self.run_id = run_id or os.environ.get("MRUN_JOB_ID") or f"local-{int(time.time())}"
        self.note = note
        self.stages: list[dict[str, Any]] = []
        self._t0 = started_at if started_at is not None else time.perf_counter()
        self._start_proc = _proc_stats() if start_proc is None else start_proc

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        tokens: int = 0,
        forwards: int = 0,
        cold: bool = False,
        bytes_moved: int = 0,
        **extra: Any,
    ):
        """Time one stage. Set ``cold=True`` for any stage that includes one-off setup —
        compile, disk-cache load, weight residency warm-up.

        This is not cosmetic. The first smoke run timed a single Core ML stage that included
        a 2.5 s package load and published **20.4 tok/s** for a backend measured at 1428 —
        a number 70x too low, presented as its throughput. Cold stages stay in the report
        (that is where cold-start cost is visible) but are excluded from the leaderboard's
        throughput claim.
        """
        t0 = time.perf_counter()
        err = None
        try:
            yield
        except Exception as e:  # noqa: BLE001 — a failed stage must still be recorded
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            dt = time.perf_counter() - t0
            entry = {
                "stage": name,
                "wall_s": round(dt, 4),
                "tokens": tokens,
                "forwards": forwards,
                "cold": bool(cold),
                **extra,
            }
            if tokens and dt > 0:
                entry["tok_per_s"] = round(tokens / dt, 1)
            if forwards and dt > 0:
                entry["s_per_forward"] = round(dt / forwards, 5)
            # Bandwidth is the denominator most of this project's paths are actually bound by:
            # CPU-paged is dequant/movement-bound (79% of a step at T=1), and Core ML's I/O
            # boundary moves a full-vocab logit tensor per call. A GB/s figure says whether a
            # stage is near its fabric's roofline or leaving headroom, which tok/s alone cannot.
            moved = int(bytes_moved) or self._auto_bytes(tokens, forwards)
            if moved and dt > 0:
                entry["bytes_moved"] = moved
                entry["gb_per_s"] = round(moved / dt / 1e9, 3)
                if tokens:
                    entry["bytes_per_token"] = int(moved / tokens)
            entry["compute_units"] = round(dt * self._device_weight(), 3)
            if err:
                entry["error"] = err
            self.stages.append(entry)

    def _auto_bytes(self, tokens: int, forwards: int) -> int:
        """Weight bytes a stage necessarily streamed, when the caller did not say.

        The paged engines stream the whole quantised store once per forward — that is the
        design (resident RAM is O(largest matrix), not O(model)) and it is also the dominant
        traffic. Store size is known, so the floor is derivable rather than guessed. Returns 0
        for resident backends, where weights are not re-streamed and a made-up number would be
        worse than none.
        """
        store = getattr(self.engine, "store", None)
        backend = str(self.facts.get("backend", ""))
        if store is None or not forwards or "paged" not in backend:
            return 0
        try:
            nbytes = int(getattr(store.w, "nbytes", 0)) + int(getattr(store.s, "nbytes", 0))
            return nbytes * int(forwards)
        except Exception:  # noqa: BLE001
            return 0

    def _device_weight(self) -> float:
        backend = str(self.facts.get("backend", "")).lower()
        if backend in BACKEND_FABRIC:
            return DEVICE_WEIGHTS[BACKEND_FABRIC[backend]]
        dev = str(self.facts.get("device", "cpu")).lower()
        for key, w in DEVICE_WEIGHTS.items():
            if key in dev:
                return w
        return DEVICE_WEIGHTS["cpu"]

    def summary(self) -> dict[str, Any]:
        # Engine facts change during execution: compiled shapes, cache hits, scoring paths,
        # fallback reasons, and scalar degradation all appear after construction.
        self.facts = engine_facts(self.engine)
        wall = time.perf_counter() - self._t0
        end_proc = _proc_stats()
        tokens = sum(s.get("tokens", 0) for s in self.stages)
        cpu_s = None
        if (
            end_proc.get("cpu_seconds") is not None
            and self._start_proc.get("cpu_seconds") is not None
        ):
            cpu_s = round(end_proc["cpu_seconds"] - self._start_proc["cpu_seconds"], 3)
        out = {
            "run_id": self.run_id,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": self.note,
            "host": platform.node().split(".")[0],
            "platform": platform.platform(),
            "wall_s": round(wall, 3),
            "cpu_seconds": cpu_s,
            "peak_rss_mb": end_proc.get("peak_rss_mb"),
            "tokens_total": tokens,
            "tok_per_s_overall": round(tokens / wall, 1) if tokens and wall else None,
            "bytes_moved_total": sum(s.get("bytes_moved", 0) for s in self.stages) or None,
            "gb_per_s_peak": max((s.get("gb_per_s", 0) for s in self.stages), default=0) or None,
            "compute_units_total": round(sum(s.get("compute_units", 0) for s in self.stages), 3),
            "device_weight": self._device_weight(),
            "device_weights_declared": DEVICE_WEIGHTS,
            "engine": self.facts,
            "stages": self.stages,
        }
        out.update({k: v for k, v in _gpu_stats().items()})
        return out

    def render_graph(self, payload: dict[str, Any] | None = None) -> str:
        """Small text graph of the path that executed. Deliberately not a picture:
        it must survive in a job log, a diff, and a terminal."""
        total = payload or self.summary()
        f = total["engine"]
        lines = [f"run {self.run_id}  [{f.get('model')}]"]
        fabric = f.get("reported_fabric") or f.get("device")
        chain = [f"backend={f.get('backend')}", f"device={f.get('device')}", f"fabric={fabric}"]
        lines.append("  " + " -> ".join(chain))
        ev = f.get("execution_evidence") or {}
        if ev.get("compute_units_request"):
            lines.append(f"    compute_units_request : {ev['compute_units_request']}")
        if ev.get("last_execution_path"):
            lines.append(f"    last_execution_path   : {ev['last_execution_path']}")
        if ev.get("placement_verified") is not None:
            lines.append(f"    placement_verified    : {ev['placement_verified']}")
        if ev.get("accelerator_unavailable_reason"):
            lines.append(f"    !! accelerator unavailable: {ev['accelerator_unavailable_reason']}")
        if ev.get("last_fallback_to_paged"):
            lines.append("    !! last call FELL BACK to paged")
        if ev.get("compiled_shapes"):
            lines.append(
                f"    compiled_shapes       : {ev['compiled_shapes']}"
                f"  (disk hit/miss {ev.get('disk_cache_hits')}/"
                f"{ev.get('disk_cache_misses')})"
            )
        for path, s in (f.get("scoring_paths") or {}).items():
            speed = f"  ({s['probes_per_s']} probes/s)" if s.get("probes_per_s") else ""
            lines.append(
                f"    scoring[{path}]".ljust(28)
                + f": {s['probes']} probes in {s['seconds']}s{speed}"
            )
        if f.get("degraded_to_scalar"):
            lines.append(f"    !! DEGRADED to per-row loops: {f['scalar_fallbacks']}")
        if f.get("margin_floor"):
            lines.append(
                f"    margin_floor          : {f['margin_floor']} (winners below this are flagged)"
            )
        for s in self.stages:
            bits = [f"{s['wall_s']}s", f"cu={s['compute_units']}"]
            if s.get("tok_per_s"):
                bits.append(f"{s['tok_per_s']} tok/s")
            if s.get("gb_per_s"):
                bits.append(f"{s['gb_per_s']} GB/s")
            if s.get("error"):
                bits.append(f"ERROR {s['error']}")
            lines.append(f"  · {s['stage']:<20} " + "  ".join(bits))
        lines.append(
            f"  = wall {total['wall_s']}s  cpu {total['cpu_seconds']}s  "
            f"compute_units {total['compute_units_total']}  "
            f"peak_rss {total['peak_rss_mb']}MB"
        )
        return "\n".join(lines)

    def save(self, out_dir: str | Path, *, print_graph: bool = True) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = self.summary()
        path = out_dir / f"engine-report-{self.run_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        graph = self.render_graph(payload)
        (out_dir / f"engine-graph-{self.run_id}.txt").write_text(graph)
        if print_graph:
            print(graph, flush=True)
        try:
            from .leaderboard import record as _record

            _record(payload)
        except Exception:  # noqa: BLE001 — leaderboard must never fail a run
            pass
        return path
