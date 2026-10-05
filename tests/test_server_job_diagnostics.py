from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import mrun.server.app as server_app
from mrun.server.db import DB


def _job() -> dict:
    return {
        "job_id": "job-diagnostics",
        "client_run_id": "run-diagnostics",
        "experiment": "diagnostics",
        "state": "queued",
        "needs": {},
        "reservation": {"ram_mb": 128.0, "vram_mb": 0.0, "cpu_threads": 1},
        "cmd": ["python", "run.py"],
        "config": {},
    }


def _guarded_debug_job(job_id: str) -> dict:
    payload = f"sealed payload for {job_id}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    job = _job()
    job.update(
        {
            "job_id": job_id,
            "client_run_id": f"run-{job_id}",
            "needs": {
                "payload_custody_v2": True,
                "saturn_debug_credential_v1": True,
            },
            "custody_required": True,
            "payload_declared_sha256": digest,
            "payload_declared_size": len(payload),
            "payload_sealed_sha256": digest,
            "payload_sealed_size": len(payload),
            "payload_sealed_ts": time.time(),
        }
    )
    return job


def _claim_guarded_debug(
    db: DB,
    job_id: str,
    *,
    credential_id: str,
    credential: str,
) -> None:
    assert db.claim_job(
        job_id,
        host="testhost",
        lease_expires_ts=time.time() + 60.0,
        reservation={"ram_mb": 128.0, "vram_mb": 0.0, "cpu_threads": 1},
        plan={"backend": "hf"},
        lease_identity=f"lease-{job_id}",
        lease_capability=f"agent-{job_id}",
        debug_credential_id=credential_id,
        debug_credential=credential,
        debug_scopes=server_app._SATURN_DEBUG_SCOPES,
    )


def test_queue_refresh_publishes_and_clears_cooperative_resident_yield(tmp_path, monkeypatch):
    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    db = DB(tmp_path / "mrun.db")
    monkeypatch.setattr(server_app, "db", db)
    db.upsert_host(
        "beast",
        {
            "host": "beast",
            "os": "linux",
            "arch": "x86_64",
            "caps": {"cpu": True, "cuda": True},
            "ram_total_mb": 66_509,
            "vram_total_mb": 16_376,
            "cpu_threads": 32,
            "disk_total_gb": 2_000,
            "agent_version": "test",
            "protocol_version": 2,
        },
    )
    db.record_telemetry(
        "beast",
        {
            "cpu_pct": 0,
            "ram_free_mb": 48_509,
            "vram_free_mb": 2_992,
            "disk_free_gb": 500,
            "swap_used_mb": 0,
            "running": [
                {
                    "job_id": "job-resident",
                    "tree_rss_mb": 18_000,
                    "vram_mb": 13_384,
                }
            ],
        },
    )
    resident = {
        **_job(),
        "job_id": "job-resident",
        "client_run_id": "run-resident",
        "experiment": "resident-service",
        "state": "running",
        "needs": {"cuda": True, "host": "beast"},
        "reservation": {
            "ram_mb": 24_000,
            "vram_mb": 13_500,
            "cpu_threads": 4,
        },
        "priority": 0,
        "config": {
            "resident_worker": True,
            "preemptible_resident": True,
            "resident_yield_after_s": 60,
        },
    }
    db.insert_job(resident)
    db.update_job(
        "job-resident",
        assigned_host="beast",
        started_ts=time.time(),
        lease_expires_ts=time.time() + 60,
    )
    finite = {
        **_job(),
        "job_id": "job-finite",
        "client_run_id": "run-finite",
        "experiment": "finite-proof",
        "needs": {"cuda": True, "host": "beast"},
        "reservation": {
            "ram_mb": 4_000,
            "vram_mb": 2_000,
            "cpu_threads": 4,
        },
        "priority": 20,
        "created_ts": time.time(),
    }
    db.insert_job(finite)

    server_app._refresh_queue_admission_details()

    request = db.job("job-resident")["meta"]["resident_yield_request"]
    assert request["blocker_job_id"] == "job-finite"
    assert request["blocked_reason"].startswith("vram:")
    assert any(
        event["kind"] == "job.resident_yield_requested"
        for event in db.events(job_id="job-resident")
    )

    # Live free-memory telemetry drifts by a few MiB every sample.  That changes
    # the explanatory reason but must not republish the same resident/blocker
    # signal or reset its original request timestamp.
    requested_ts = request["requested_ts"]
    db.record_telemetry(
        "beast",
        {
            "cpu_pct": 0,
            "ram_free_mb": 48_500,
            "vram_free_mb": 2_980,
            "disk_free_gb": 500,
            "swap_used_mb": 0,
            "running": [
                {
                    "job_id": "job-resident",
                    "tree_rss_mb": 18_009,
                    "vram_mb": 13_396,
                }
            ],
        },
    )
    server_app._refresh_queue_admission_details()
    repeated = db.job("job-resident")["meta"]["resident_yield_request"]
    assert repeated["requested_ts"] == requested_ts
    assert sum(
        event["kind"] == "job.resident_yield_requested"
        for event in db.events(job_id="job-resident")
    ) == 1

    # A scheduler restart can briefly have no fresh host view.  Keep the
    # already-issued request sticky until the blocker leaves the queue.
    server_app._refresh_resident_yield_requests(
        db.jobs(state="queued"),
        all_hosts=[],
        active_by_host={"beast": [db.job("job-resident")]},
    )
    assert db.job("job-resident")["meta"]["resident_yield_request"] == repeated
    assert not any(
        event["kind"] == "job.resident_yield_cleared"
        for event in db.events(job_id="job-resident")
    )

    db.update_job("job-finite", state="cancelled", finished_ts=time.time())
    server_app._refresh_queue_admission_details()

    assert db.job("job-resident")["meta"]["resident_yield_request"] is None
    assert any(
        event["kind"] == "job.resident_yield_cleared" for event in db.events(job_id="job-resident")
    )


def test_agent_phase_and_failure_are_durable(tmp_path, monkeypatch):
    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    db = DB(tmp_path / "mrun.db")
    monkeypatch.setattr(server_app, "db", db)
    db.insert_job(_job())
    assert db.claim_job(
        "job-diagnostics",
        host="testhost",
        lease_expires_ts=time.time() + 60.0,
        reservation={"ram_mb": 128.0, "vram_mb": 0.0, "cpu_threads": 1},
        plan={"backend": "hf"},
    )
    client = TestClient(server_app.app)

    phase = client.post(
        "/api/jobs/job-diagnostics/events",
        json={
            "phase": "process.launch",
            "detail": "starting child",
            "result": {"command": ["python", "run.py"]},
        },
    )
    assert phase.status_code == 200

    failure = {
        "kind": "process_exit",
        "phase": "process.finalize",
        "message": "child exited with return code 7",
        "returncode": 7,
        "log_tail": "traceback tail",
    }
    terminal = client.post(
        "/api/jobs/job-diagnostics/events",
        json={
            "state": "failed",
            "detail": "child exited with return code 7",
            "result": {"status": "failed", "returncode": 7, "failure": failure},
        },
    )
    assert terminal.status_code == 200

    job = db.job("job-diagnostics")
    assert job is not None
    assert job["result"]["failure"]["kind"] == "process_exit"
    events = db.events(job_id="job-diagnostics")
    assert any(event["kind"] == "job.agent_phase" for event in events)
    assert any(event["kind"] == "job.finished" for event in events)


def test_debuggable_job_discovery_tracks_live_registration(tmp_path, monkeypatch):
    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    db = DB(tmp_path / "mrun.db")
    monkeypatch.setattr(server_app, "db", db)
    monkeypatch.setattr(server_app, "_saturn_debug_channels", {})
    db.insert_job(_job())
    client = TestClient(server_app.app)

    empty = client.get("/api/jobs?debuggable=true")
    assert empty.status_code == 200
    assert empty.json() == []

    capability = {
        "schema": "saturn-debugger-capability-v1",
        "job_id": "job-diagnostics",
        "session_id": "session-diagnostics",
        "family": "autoregressive",
    }
    registered = client.post(
        "/api/jobs/job-diagnostics/debug/register",
        json={"capability": capability},
    )
    assert registered.status_code == 200
    jobs = client.get("/api/jobs?debuggable=true").json()
    assert [job["job_id"] for job in jobs] == ["job-diagnostics"]
    assert jobs[0]["debugger_capable"] is True


def test_guarded_debug_credential_is_narrow_job_and_session_scoped(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MRUN_TOKEN", "client-secret")
    db = DB(tmp_path / "mrun.db")
    monkeypatch.setattr(server_app, "db", db)
    monkeypatch.setattr(server_app, "_saturn_debug_channels", {})
    db.insert_job(_guarded_debug_job("job-a"))
    db.insert_job(_guarded_debug_job("job-b"))
    _claim_guarded_debug(
        db,
        "job-a",
        credential_id="dbg-a",
        credential="debug-secret-a",
    )
    _claim_guarded_debug(
        db,
        "job-b",
        credential_id="dbg-b",
        credential="debug-secret-b",
    )
    client = TestClient(server_app.app)
    debug_headers = {
        "X-MRun-Debug-Credential-ID": "dbg-a",
        "X-MRun-Debug-Credential": "debug-secret-a",
    }
    client_headers = {"X-MRun-Token": "client-secret"}
    capability = {
        "schema": "saturn-debugger-capability-v1",
        "job_id": "job-a",
        "session_id": "session-a",
        "family": "diffusion",
    }

    registered = client.post(
        "/api/jobs/job-a/debug/register",
        headers=debug_headers,
        json={"capability": capability},
    )
    assert registered.status_code == 200
    repeated = client.post(
        "/api/jobs/job-a/debug/register",
        headers=debug_headers,
        json={"capability": capability},
    )
    assert repeated.status_code == 200
    assert repeated.json()["idempotent"] is True

    rebound = client.post(
        "/api/jobs/job-a/debug/register",
        headers=debug_headers,
        json={"capability": {**capability, "session_id": "session-other"}},
    )
    assert rebound.status_code == 409
    cross_job = client.get(
        "/api/jobs/job-b/debug", headers=debug_headers
    )
    assert cross_job.status_code == 403

    # This bearer cannot issue client commands or read any other job route.
    denied_command = client.post(
        "/api/jobs/job-a/debug",
        headers=debug_headers,
        json={"request_id": "request-denied", "operation": "status"},
    )
    assert denied_command.status_code == 401
    assert (
        client.get("/api/jobs/job-a/events", headers=debug_headers).status_code
        == 401
    )

    command_result: dict[str, object] = {}

    def send_command() -> None:
        response = client.post(
            "/api/jobs/job-a/debug",
            headers=client_headers,
            json={"request_id": "request-1", "operation": "status"},
        )
        command_result["status"] = response.status_code
        command_result["body"] = response.json()

    sender = threading.Thread(target=send_command)
    sender.start()
    deadline = time.time() + 5.0
    request = None
    while request is None and time.time() < deadline:
        polled = client.get(
            "/api/jobs/job-a/debug?wait_s=0.1", headers=debug_headers
        )
        assert polled.status_code == 200
        request = polled.json()["request"]
    assert request is not None
    responded = client.post(
        "/api/jobs/job-a/debug/respond",
        headers=debug_headers,
        json={
            "request_id": "request-1",
            "schema": "saturn-debugger-response-v1",
            "ok": True,
            "result": {"state": "paused"},
        },
    )
    assert responded.status_code == 200
    sender.join(timeout=5.0)
    assert not sender.is_alive()
    assert command_result["status"] == 200
    assert command_result["body"]["result"] == {"state": "paused"}

    serialized = json.dumps(
        {
            "job": db.job("job-a"),
            "jobs": db.jobs(),
            "events": db.events(job_id="job-a"),
        }
    )
    assert "debug-secret-a" not in serialized
    assert "debug_credential_hash" not in serialized


def test_guarded_debug_credential_expires_with_lease_and_terminal_state(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    db = DB(tmp_path / "mrun.db")
    monkeypatch.setattr(server_app, "db", db)
    monkeypatch.setattr(server_app, "_saturn_debug_channels", {})
    db.insert_job(_guarded_debug_job("job-expiring"))
    _claim_guarded_debug(
        db,
        "job-expiring",
        credential_id="dbg-expiring",
        credential="debug-secret-expiring",
    )
    authorized = db.authorize_guarded_debug_credential(
        "job-expiring",
        credential_id="dbg-expiring",
        credential="debug-secret-expiring",
        scope="debug:register",
    )
    assert authorized["job_id"] == "job-expiring"
    client = TestClient(server_app.app)
    headers = {
        "X-MRun-Debug-Credential-ID": "dbg-expiring",
        "X-MRun-Debug-Credential": "debug-secret-expiring",
    }
    registered = client.post(
        "/api/jobs/job-expiring/debug/register",
        headers=headers,
        json={
            "capability": {
                "schema": "saturn-debugger-capability-v1",
                "job_id": "job-expiring",
                "session_id": "session-expiring",
            }
        },
    )
    assert registered.status_code == 200
    db.update_job("job-expiring", state="succeeded", finished_ts=time.time())
    with pytest.raises(Exception, match="inactive"):
        db.authorize_guarded_debug_credential(
            "job-expiring",
            credential_id="dbg-expiring",
            credential="debug-secret-expiring",
            scope="debug:register",
        )
    routed = client.post(
        "/api/jobs/job-expiring/debug",
        json={"request_id": "terminal-command", "operation": "status"},
    )
    assert routed.status_code == 409
    assert "job-expiring" not in server_app._saturn_debug_channels


def test_stale_debugger_mailbox_is_pruned_before_command_routing(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    db = DB(tmp_path / "mrun.db")
    monkeypatch.setattr(server_app, "db", db)
    monkeypatch.setattr(server_app, "_saturn_debug_channels", {})
    db.insert_job(_guarded_debug_job("job-stale-mailbox"))
    _claim_guarded_debug(
        db,
        "job-stale-mailbox",
        credential_id="dbg-stale",
        credential="debug-secret-stale",
    )
    client = TestClient(server_app.app)
    registered = client.post(
        "/api/jobs/job-stale-mailbox/debug/register",
        headers={
            "X-MRun-Debug-Credential-ID": "dbg-stale",
            "X-MRun-Debug-Credential": "debug-secret-stale",
        },
        json={
            "capability": {
                "schema": "saturn-debugger-capability-v1",
                "job_id": "job-stale-mailbox",
                "session_id": "session-stale",
            }
        },
    )
    assert registered.status_code == 200
    server_app._saturn_debug_channels["job-stale-mailbox"]["last_seen_ts"] = (
        time.time() - server_app._SATURN_DEBUG_RETENTION_S - 1.0
    )

    routed = client.post(
        "/api/jobs/job-stale-mailbox/debug",
        json={"request_id": "stale-command", "operation": "status"},
    )
    assert routed.status_code == 410
    assert "job-stale-mailbox" not in server_app._saturn_debug_channels


def test_guarded_lease_mints_debug_secret_only_in_agent_response(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MRUN_TOKEN", "client-secret")
    monkeypatch.setenv("MRUN_AGENT_TOKEN", "agent-secret")
    db = DB(tmp_path / "mrun.db")
    monkeypatch.setattr(server_app, "db", db)
    db.upsert_host(
        "testhost",
        {
            "os": "linux",
            "arch": "x86_64",
            "caps": {
                "cpu": True,
                "payload_custody_v2": True,
                "saturn_debug_credential_v1": True,
            },
            "ram_total_mb": 16_000,
            "vram_total_mb": 0,
            "cpu_threads": 8,
            "disk_total_gb": 100,
            "agent_version": "test",
            "protocol_version": 1,
        },
    )
    db.record_telemetry(
        "testhost",
        {
            "cpu_pct": 1,
            "ram_free_mb": 12_000,
            "vram_free_mb": 0,
            "disk_free_gb": 90,
            "load1": 0,
            "swap_used_mb": 0,
            "running": [],
        },
    )
    db.insert_job(_guarded_debug_job("job-leased-debug"))
    client = TestClient(server_app.app)
    leased = client.post(
        "/api/agents/testhost/lease",
        headers={
            "X-MRun-Token": "client-secret",
            "X-MRun-Agent-Token": "agent-secret",
        },
    )

    assert leased.status_code == 200
    body = leased.json()
    debug_authorization = body["debug_authorization"]
    assert debug_authorization["schema"] == "mrun.job-debug-credential-v1"
    assert set(debug_authorization["scopes"]) == set(server_app._SATURN_DEBUG_SCOPES)
    raw_secret = debug_authorization["credential"]
    assert raw_secret
    public = client.get(
        "/api/jobs/job-leased-debug", headers={"X-MRun-Token": "client-secret"}
    ).json()
    events = client.get(
        "/api/jobs/job-leased-debug/events",
        headers={"X-MRun-Token": "client-secret"},
    ).json()
    serialized = json.dumps({"public": public, "events": events})
    assert raw_secret not in serialized
    assert "debug_authorization" not in serialized
    assert "debug_credential_hash" not in serialized
