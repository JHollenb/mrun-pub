"""Append-only speed/size/compute leaderboard (issue I49).

Every "how fast is X" answer in this project has been re-derived by hand from scattered
experiment JSON, and half the recorded speed claims carry no fabric or batch context — which
is how "ANE 47x" survived as an ANE number while the work ran on the GPU. Rows here land
automatically from `EngineReport.save()`, never by hand, and every row carries the run id that
produced it plus the parity contract that was in force, so any number can be traced back.

Storage is JSONL under the stores root: append-only, greppable, no server, survives a scheduler
outage. `mrun leaderboard` renders it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

FIELDS = (
    "run_id", "ts", "host", "model", "backend", "device", "fabric",
    "batch", "seq_len", "tok_per_s", "probes_per_s", "scoring_path", "wall_s",
    "compute_units_total", "peak_rss_mb", "working_set_mb", "numerical_contract",
    "margin_floor", "degraded_to_scalar", "placement_verified", "note",
)


def leaderboard_path() -> Path:
    override = os.environ.get("MRUN_LEADERBOARD")
    if override:
        return Path(override)
    from ..paths import stores_root

    return Path(stores_root()).parent / "mrun-leaderboard.jsonl"


def record(report: dict[str, Any]) -> Path | None:
    """Fold one engine report into a leaderboard row. Never raises."""
    try:
        eng = report.get("engine", {})
        ev = eng.get("execution_evidence", {}) or {}
        stages = report.get("stages", [])
        # The throughput claim is the fastest WARM token-bearing stage. Cold stages carry
        # compile / package-load / residency cost: the first smoke run published a Core ML
        # backend at 20.4 tok/s because its only stage included a 2.5 s package load, against
        # 1428 tok/s measured warm. If every stage is cold there is no throughput claim to
        # make, and the row records wall time only rather than inventing one.
        scored = [s for s in stages if s.get("tok_per_s") and not s.get("cold")]
        best = max(scored, key=lambda s: s["tok_per_s"]) if scored else {}
        cold_only = bool([s for s in stages if s.get("tok_per_s")]) and not scored
        scoring_paths = eng.get("scoring_paths", {}) or {}
        measured_paths = [
            (name, stats)
            for name, stats in scoring_paths.items()
            if stats.get("probes_per_s")
        ]
        scoring_name, scoring = (
            max(measured_paths, key=lambda item: item[1]["probes_per_s"])
            if measured_paths
            else (None, {})
        )
        row = {
            "run_id": report.get("run_id"),
            "ts": report.get("ts"),
            "host": report.get("host"),
            "model": eng.get("model"),
            "backend": eng.get("backend"),
            "device": eng.get("device"),
            "fabric": eng.get("reported_fabric"),
            "batch": best.get("batch"),
            "seq_len": best.get("seq_len"),
            "tok_per_s": best.get("tok_per_s"),
            # Forced-choice probes are not generated tokens. Keep the units separate so an
            # ordinary scoring run contributes useful speed evidence without inflating tok/s.
            "probes_per_s": scoring.get("probes_per_s"),
            "scoring_path": scoring_name,
            "gb_per_s": best.get("gb_per_s"),
            "bytes_per_token": best.get("bytes_per_token"),
            "wall_s": report.get("wall_s"),
            "compute_units_total": report.get("compute_units_total"),
            "peak_rss_mb": report.get("peak_rss_mb"),
            "working_set_mb": eng.get("working_set_mb"),
            "numerical_contract": eng.get("numerical_contract"),
            "margin_floor": eng.get("margin_floor"),
            "degraded_to_scalar": eng.get("degraded_to_scalar"),
            "placement_verified": ev.get("placement_verified"),
            "stage": best.get("stage"),
            "cold_only": cold_only or None,
            "note": report.get("note"),
        }
        if row["tok_per_s"] is None and row["wall_s"] is None:
            return None
        path = leaderboard_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        return path
    except Exception:  # noqa: BLE001 — bookkeeping must never fail a run
        return None


def load() -> list[dict[str, Any]]:
    path = leaderboard_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def render(rows: list[dict[str, Any]] | None = None, *, sort: str = "tok_per_s",
           model: str | None = None, limit: int = 40) -> str:
    rows = load() if rows is None else rows
    if model:
        rows = [r for r in rows if model.lower() in str(r.get("model", "")).lower()]
    if not rows:
        return "leaderboard empty — run something through EngineReport.save()"
    rows = sorted(rows, key=lambda r: (r.get(sort) or 0), reverse=True)[:limit]
    head = (
        f"{'model':<22}{'backend':<10}{'fabric':<20}{'B':>4}{'T':>5}"
        f"{'tok/s':>10}{'probe/s':>10}{'GB/s':>8}{'CU':>9}{'RSS MB':>9}  flags"
    )
    out = [head, "-" * len(head)]
    for r in rows:
        flags = []
        if r.get("degraded_to_scalar"):
            flags.append("SCALAR-FALLBACK")
        if r.get("placement_verified") is False and r.get("fabric"):
            flags.append("placement-unverified")
        if r.get("margin_floor"):
            flags.append(f"margin>{r['margin_floor']}")
        if r.get("scoring_path"):
            flags.append(f"score={r['scoring_path']}")
        if r.get("cold_only"):
            flags.append("COLD-ONLY(no warm stage)")
        out.append(
            f"{str(r.get('model'))[:21]:<22}{str(r.get('backend'))[:9]:<10}"
            f"{str(r.get('fabric') or r.get('device'))[:19]:<20}"
            f"{r.get('batch') or '-':>4}{r.get('seq_len') or '-':>5}"
            f"{r.get('tok_per_s') or '-':>10}{r.get('probes_per_s') or '-':>10}"
            f"{r.get('gb_per_s') or '-':>8}"
            f"{r.get('compute_units_total') or '-':>9}"
            f"{r.get('peak_rss_mb') or '-':>9}  {' '.join(flags)}"
        )
    out.append("")
    out.append("CU = compute units (device-seconds x device weight; weights are coarse and "
               "declared in each engine report — do NOT compare across fabrics as if equal).")
    out.append(f"{len(load())} rows total at {leaderboard_path()}")
    return "\n".join(out)
