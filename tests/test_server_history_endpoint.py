from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import mrun.server.app as server_app
from mrun.server.db import DB


def _estimate(client_run_id: str, *, ram_peak_mb: float, model: str = "m/test") -> dict:
    return {
        "client_run_id": client_run_id,
        "experiment": "hist-exp",
        "model": model,
        "host": "beast",
        "status": "succeeded",
        "ram_peak_mb": ram_peak_mb,
        "vram_peak_mb": 512.0,
        "wall_s": 60.0,
        "backend": "hf",
        "dtype": "bf16",
        "task_family": "forward",
        "est_ram_mb": 8192.0,
    }


def _client(tmp_path, monkeypatch) -> TestClient:
    # Order-independence: fleet integration tests export MRUN_TOKEN into the
    # process env, which turns unauthenticated TestClient calls into 401s.
    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    db = DB(tmp_path / "mrun.db")
    monkeypatch.setattr(server_app, "db", db)
    return TestClient(server_app.app)


def test_history_returns_exact_row_after_estimate(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    server_app.db.add_estimate(_estimate("run-abc", ram_peak_mb=9677.0))
    resp = client.get("/api/history/run-abc")
    assert resp.status_code == 200
    body = resp.json()
    assert body["exact"]["ram_peak_mb"] == 9677.0
    assert body["exact"]["client_run_id"] == "run-abc"
    assert body["generalized"] is None


def test_history_nulls_when_absent(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.get("/api/history/never-ran")
    assert resp.status_code == 200
    assert resp.json() == {"exact": None, "exact_stats": None, "generalized": None}


def test_history_generalized_needs_family_key_and_three_runs(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    for i in range(3):
        row = _estimate(f"run-{i}", ram_peak_mb=1000.0 + i, model="m/gen")
        row["family_key"] = "model:m/gen:forward"
        server_app.db.add_estimate(row)
    resp = client.get("/api/history/run-0", params={"family_key": "model:m/gen:forward"})
    body = resp.json()
    assert body["generalized"]["n"] == 3
    assert body["generalized"]["p95_ram_mb"] >= 1000.0
    assert body["exact_stats"]["ram_peak_mb"] == 1000.0
