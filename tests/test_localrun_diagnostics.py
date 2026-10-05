from __future__ import annotations

import pytest

from mrun.client.localrun import LocalRunHandle
from mrun.diagnostics import normalize_external_result


class _Api:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def json(self, method: str, path: str, *, json_body=None, **_kwargs):
        self.calls.append((method, path, json_body or {}))
        if path == "/api/local-runs":
            return {"job_id": "job-local-diagnostics"}
        return {"ok": True}


def test_external_result_adds_a_structured_failure_without_clobbering_fields():
    result = normalize_external_result(
        state="failed",
        result={
            "status": "failed",
            "returncode": 1,
            "stderr_tail": "ImportError: token=secret",
            "custom": "preserved",
        },
        command=["python", "run.py"],
        cwd="/tmp/experiment",
    )

    assert result["custom"] == "preserved"
    assert result["failure"]["kind"] == "external_process_exit"
    assert result["failure"]["returncode"] == 1
    assert result["failure"]["command"] == ["python", "run.py"]
    assert result["failure"]["cwd"] == "/tmp/experiment"
    assert "ImportError" in result["failure"]["log_tail"]
    assert "secret" not in result["failure"]["log_tail"]


def test_local_run_finish_sends_failure_for_legacy_status_returncode_callers():
    api = _Api()
    handle = LocalRunHandle(
        "mx:local-exp",
        command=["python", "run.py"],
        cwd="/tmp/experiment",
        api=api,
    )
    handle.register()
    handle.finish(
        "failed",
        result={"status": "failed", "returncode": 1},
    )

    finish = next(body for method, path, body in api.calls if path.endswith("/finish"))
    failure = finish["result"]["failure"]
    assert failure["kind"] == "external_process_exit"
    assert failure["message"] == "external process exited with return code 1"
    assert failure["command"] == ["python", "run.py"]
    assert failure["cwd"] == "/tmp/experiment"


def test_legacy_external_finish_is_explainable_without_command_metadata():
    api = _Api()
    handle = LocalRunHandle("mx:legacy-local-exp", api=api)
    handle.register()
    handle.finish("failed", result={"status": "failed", "returncode": 1})

    finish = next(body for method, path, body in api.calls if path.endswith("/finish"))
    failure = finish["result"]["failure"]
    assert failure["kind"] == "external_process_exit"
    assert failure["message"] == "external process exited with return code 1"
    assert "command" not in failure


def test_server_normalizes_raw_local_finish_payload(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import mrun.server.app as server_app
    from mrun.server.db import DB

    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    db = DB(tmp_path / "mrun.db")
    monkeypatch.setattr(server_app, "db", db)
    client = TestClient(server_app.app)

    registered = client.post(
        "/api/local-runs",
        json={
            "host": "testhost",
            "experiment": "mx:raw-local-failure",
            "command": ["python", "run.py"],
            "reservation": {"ram_mb": 128},
        },
    )
    assert registered.status_code == 200
    job_id = registered.json()["job_id"]

    finished = client.post(
        f"/api/local-runs/{job_id}/finish",
        json={"state": "failed", "result": {"status": "failed", "returncode": 1}},
    )
    assert finished.status_code == 200

    job = db.job(job_id)
    assert job is not None
    assert job["cmd"] == ["python", "run.py"]
    assert job["result"]["failure"]["kind"] == "external_process_exit"
    assert job["result"]["failure"]["command"] == ["python", "run.py"]
