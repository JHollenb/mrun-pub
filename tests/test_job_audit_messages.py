from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import mrun.server.app as server_app
from mrun.server.db import DB


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    database = DB(tmp_path / "mrun.db")
    monkeypatch.setattr(server_app, "db", database)
    return TestClient(server_app.app)


def _request(*, note: str | None = None) -> dict:
    request = {
        "experiment": "audit-messages",
        "client_run_id": "audit-messages:0",
        "cmd": ["python", "worker.py"],
        "config": {"task_family": "test"},
        "needs": {"cuda": True},
        "payload_kind": "cmd",
    }
    if note is not None:
        request["note"] = note
    return request


def test_queue_note_and_cancellation_reason_are_durable(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    note = "One-row CUDA smoke; cancel if physical forwards exceed the declared two."
    submitted = client.post("/api/jobs", json=_request(note=note))
    assert submitted.status_code == 200
    job_id = submitted.json()["job_id"]
    assert submitted.json()["note"] == note

    queued = client.get(f"/api/jobs/{job_id}").json()
    assert queued["state"] == "queued"
    assert queued["meta"]["queue_note"] == note

    reason = "Observed scalar replay loop exceeded the declared forward budget."
    cancelled = client.post(
        f"/api/jobs/{job_id}/cancel",
        json={"reason": reason},
    )
    assert cancelled.status_code == 200
    assert cancelled.json() == {"state": "cancelled", "reason": reason}

    terminal = client.get(f"/api/jobs/{job_id}").json()
    assert terminal["meta"] == {
        "queue_note": note,
        "cancellation_reason": reason,
    }
    events = client.get(f"/api/jobs/{job_id}/events").json()
    assert any(
        event["kind"] == "job.submit" and event["payload"]["note"] == note
        for event in events
    )
    assert any(
        event["kind"] == "job.cancelled" and event["reason"] == reason
        for event in events
    )


@pytest.mark.parametrize(
    ("path", "body", "message"),
    [
        ("/api/jobs", _request(note=" "), "note must not be empty"),
        ("/api/jobs", _request(note="x" * 501), "note must be at most 500"),
    ],
)
def test_queue_note_validation(tmp_path, monkeypatch, path, body, message) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.post(path, json=body)
    assert response.status_code == 422
    assert message in response.text


def test_cancel_reason_validation_is_strict(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    job_id = client.post("/api/jobs", json=_request()).json()["job_id"]

    empty = client.post(f"/api/jobs/{job_id}/cancel", json={"reason": "  "})
    assert empty.status_code == 422
    assert "reason must not be empty" in empty.text

    unknown = client.post(
        f"/api/jobs/{job_id}/cancel",
        json={"reason": "stop", "silent": True},
    )
    assert unknown.status_code == 422
    assert "may contain only reason" in unknown.text


def test_reasonless_cancel_remains_backward_compatible(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    job_id = client.post("/api/jobs", json=_request()).json()["job_id"]

    cancelled = client.post(f"/api/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json() == {"state": "cancelled"}
    assert client.get(f"/api/jobs/{job_id}").json()["meta"] == {}
