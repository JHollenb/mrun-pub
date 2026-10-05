"""End-to-end fleet loop on one machine: in-process uvicorn + agent thread.

Covers: submit -> lease -> run -> logs -> succeeded; killed_ram at reservation;
queueing when reservations exceed capacity; cancel; estimates calibration
(second submit of the same config gets source=history).
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

PORT = 9925
CLIENT_TOKEN = "fleet-general-client-token"
AGENT_TOKEN = "fleet-independent-agent-token"


@pytest.fixture(scope="module")
def fleet(tmp_path_factory):
    module_patch = pytest.MonkeyPatch()
    data = tmp_path_factory.mktemp("server-data")
    module_patch.setenv("MRUN_SERVER_DATA", str(data))
    module_patch.setenv("MRUN_TOKEN", CLIENT_TOKEN)
    module_patch.setenv("MRUN_AGENT_TOKEN", AGENT_TOKEN)
    module_patch.setenv("MRUN_URL", f"http://127.0.0.1:{PORT}")
    # The integration host is the developer machine running pytest. It may already
    # have harmless swap in use, which would make admission refuse every synthetic
    # test job. Pure scheduler tests cover swap-pressure refusal directly.
    module_patch.setenv("MRUN_SWAP_MAX_MB", "999999")

    from mrun.server import app as server_app
    from mrun.server.db import DB

    # Other test modules may already have imported the application. Bind a fresh
    # database explicitly instead of assuming the import re-reads environment.
    module_patch.setattr(server_app, "db", DB(data / "mrun.db"))
    app = server_app.app

    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    st = threading.Thread(target=server.run, daemon=True)
    st.start()

    from mrun.client.api import Api

    api = Api(f"http://127.0.0.1:{PORT}", token=CLIENT_TOKEN)
    for _ in range(100):
        try:
            api.json("GET", "/healthz")
            break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("server did not start")

    from mrun.agent import main as agent_main
    from mrun.agent.config import AgentConfig
    from mrun.agent.main import Agent

    # Keep the synthetic fleet independent of whichever developer machine runs
    # pytest.  In particular, admission should not start failing merely because
    # the real home/model volume has less than the production disk headroom.
    real_telemetry_payload = agent_main.telemetry_payload

    def synthetic_telemetry_payload(*args, **kwargs):
        payload = real_telemetry_payload(*args, **kwargs)
        # Admission mechanics must not depend on other applications occupying
        # the developer's RAM. The workers below allocate only bounded toy data.
        import psutil
        payload["ram_free_mb"] = psutil.virtual_memory().total / 1e6 * 0.95
        payload["mem_pressure"] = 1
        payload["disk_free_gb"] = 90.0
        payload["disks"] = [
            {**row, "free_gb": 90.0}
            for row in payload.get("disks", [])
        ]
        return payload

    agent_main.telemetry_payload = synthetic_telemetry_payload

    cfg = AgentConfig(
        server_url=f"http://127.0.0.1:{PORT}",
        token=CLIENT_TOKEN,
        agent_token=AGENT_TOKEN,
        max_concurrent=2,
        os_memory_limit_mode="off",
        host="testhost",
        work_root=str(tmp_path_factory.mktemp("agent-work")),
    )
    agent = Agent(cfg)
    agent.register()
    tt = threading.Thread(target=agent.telemetry_loop, daemon=True)
    lt = threading.Thread(target=agent.lease_loop, daemon=True)
    tt.start()
    lt.start()

    yield api

    agent._stop.set()
    agent_main.telemetry_payload = real_telemetry_payload
    server.should_exit = True
    st.join(timeout=5)
    module_patch.undo()


def _wait_state(api, job_id, states, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = api.json("GET", f"/api/jobs/{job_id}")
        if job["state"] in states:
            return job
        time.sleep(0.5)
    raise AssertionError(f"{job_id} never reached {states}; last={job['state']}")


def _submit(api, experiment, code, *, ram_mb=500, config=None, **kw):
    from mrun.client.submit import submit

    return submit(
        experiment=experiment,
        cmd=[sys.executable, "-c", code],
        config=config or {"code": code},
        reservation={"ram_mb": ram_mb, "cpu_threads": 1},
        env_alias=None,
        api=api,
        **kw,
    )


def _register_synthetic_host(api, host: str, *, guarded: bool) -> None:
    from mrun.protocol import PROTOCOL_VERSION

    api.json(
        "POST",
        "/api/agents/register",
        headers=(
            {"X-MRun-Agent-Token": AGENT_TOKEN}
            if guarded
            else None
        ),
        json_body={
            "host": host,
            "os": "linux",
            "arch": "x86_64",
            "caps": {
                "cpu": True,
                "cuda": False,
                "payload_custody_v2": guarded,
            },
            "ram_total_mb": 16_000,
            "vram_total_mb": 0,
            "cpu_threads": 8,
            "disk_total_gb": 100,
            "agent_version": "test",
            "protocol_version": PROTOCOL_VERSION,
        },
    )
    api.json(
        "POST",
        f"/api/agents/{host}/telemetry",
        headers=(
            {"X-MRun-Agent-Token": AGENT_TOKEN}
            if guarded
            else None
        ),
        json_body={
            "cpu_pct": 0,
            "ram_free_mb": 14_000,
            "vram_free_mb": 0,
            "disk_free_gb": 90,
            "load1": 0,
            "swap_used_mb": 0,
            "swap_total_mb": 0,
            "running": [],
        },
    )


def test_scheduler_admission_claim_endpoint_is_owner_scoped(fleet):
    from mrun.client.api import ApiError

    api = fleet
    claim = {
        "claim_key": "atlas-suite:deadbeef",
        "owner_token": "claim-owner-a",
        "ttl_s": 60.0,
        "scope": {
            "experiment": "atlas-suite",
            "config_selector": {"suite_fingerprint": "deadbeef"},
        },
        "metadata": {"suite": "atlas"},
    }
    acquired = api.json("POST", "/api/admission-claims/acquire", json_body=claim)
    assert acquired["schema"] == "mrun.admission-claim.v2"
    assert acquired["claim_key"] == claim["claim_key"]
    assert acquired["owner_token"] == claim["owner_token"]

    # The same owner may renew, but a second process cannot cross the boundary.
    renewed = api.json("POST", "/api/admission-claims/acquire", json_body=claim)
    assert renewed["expires_ts"] >= acquired["expires_ts"]
    competing = {**claim, "owner_token": "claim-owner-b"}
    with pytest.raises(ApiError, match="HTTP 409"):
        api.json("POST", "/api/admission-claims/acquire", json_body=competing)
    with pytest.raises(ApiError, match="HTTP 409"):
        api.json(
            "POST",
            "/api/admission-claims/release",
            json_body={
                "claim_key": claim["claim_key"],
                "owner_token": "claim-owner-b",
                "fencing_epoch": acquired["fencing_epoch"],
            },
        )

    released = api.json(
        "POST",
        "/api/admission-claims/release",
        json_body={
            "claim_key": claim["claim_key"],
            "owner_token": claim["owner_token"],
            "fencing_epoch": acquired["fencing_epoch"],
        },
    )
    assert released["released"] is True
    acquired_by_b = api.json(
        "POST", "/api/admission-claims/acquire", json_body=competing
    )
    assert acquired_by_b["owner_token"] == "claim-owner-b"
    api.json(
        "POST",
        "/api/admission-claims/release",
        json_body={
            "claim_key": claim["claim_key"],
            "owner_token": "claim-owner-b",
            "fencing_epoch": acquired_by_b["fencing_epoch"],
        },
    )


def test_guarded_agent_capability_is_isolated_bound_and_never_disclosed(fleet):
    from mrun.client.api import Api, ApiError

    client = fleet
    agent = Api(
        f"http://127.0.0.1:{PORT}",
        token=CLIENT_TOKEN,
        agent_token=AGENT_TOKEN,
    )
    wrong_agent = Api(
        f"http://127.0.0.1:{PORT}",
        token=CLIENT_TOKEN,
        agent_token="wrong-agent-token",
    )
    host = "guarded-auth-host"
    _register_synthetic_host(client, host, guarded=True)
    packed = b"guarded endpoint payload bytes"
    packed_sha256 = hashlib.sha256(packed).hexdigest()
    scope = {
        "experiment": "guarded-agent-isolation",
        "config_selector": {"suite_fingerprint": "agent-auth-v1"},
    }
    claim_request = {
        "claim_key": "fleet:guarded-agent-isolation:v1",
        "owner_token": "fleet-agent-isolation-owner",
        "ttl_s": 300.0,
        "scope": scope,
        "metadata": {"test": "agent-isolation"},
    }
    claim = client.json(
        "POST", "/api/admission-claims/acquire", json_body=claim_request
    )
    plan = {"backend": "python-test", "dtype": "none", "threads": 1}
    # Protected scopes are always shipped-payload jobs.  A command-only guarded
    # admission would have no byte identity from which to bind lease authority.
    with pytest.raises(ApiError, match="HTTP 422"):
        client.json(
            "POST",
            "/api/jobs/guarded",
            json_body={
                "experiment": scope["experiment"],
                "client_run_id": "guarded-agent-isolation:cell-0",
                "cmd": [sys.executable, "worker.py"],
                "config": {
                    "suite_fingerprint": "agent-auth-v1",
                    "task_family": "test",
                },
                "needs": {"host": host},
                "reservation": {"ram_mb": 500, "cpu_threads": 1},
                "plans": {host: plan},
                "payload_kind": "cmd",
                "timeout_s": 30,
                "admission": {
                    "claim_key": claim["claim_key"],
                    "owner_token": claim["owner_token"],
                    "fencing_epoch": claim["fencing_epoch"],
                    "idempotency_key": "cell-0",
                },
            },
        )
    receipt = client.json(
        "POST",
        "/api/jobs/guarded",
        json_body={
            "experiment": scope["experiment"],
            "client_run_id": "guarded-agent-isolation:cell-0",
            "cmd": [sys.executable, "worker.py"],
            "config": {
                "suite_fingerprint": "agent-auth-v1",
                "task_family": "test",
            },
            "needs": {"host": host},
            "reservation": {"ram_mb": 500, "cpu_threads": 1},
            "plans": {host: plan},
            "payload_kind": "shipped",
            "payload": {"sha256": packed_sha256, "size_bytes": len(packed)},
            "timeout_s": 30,
            "admission": {
                "claim_key": claim["claim_key"],
                "owner_token": claim["owner_token"],
                "fencing_epoch": claim["fencing_epoch"],
                "idempotency_key": "cell-0",
            },
        },
    )
    job_id = receipt["job_id"]
    seal_status, _body, _headers = client.request(
        "PUT",
        f"/api/jobs/{job_id}/payload",
        raw_body=packed,
        headers={
            "X-MRun-Admission-Owner": claim["owner_token"],
            "X-MRun-Admission-Epoch": str(claim["fencing_epoch"]),
        },
    )
    assert seal_status == 200

    leased = agent.json("POST", f"/api/agents/{host}/lease", timeout_s=5.0)
    authority = leased.pop("agent_authorization")
    assert set(authority) == {"lease_id", "capability", "expires_ts"}
    capability = authority["capability"]
    lease_headers = {
        "X-MRun-Lease-ID": authority["lease_id"],
        "X-MRun-Lease-Capability": capability,
    }

    # The normal client token sees receipts/status but cannot perform agent actions.
    normal_status = client.json("GET", f"/api/jobs/{job_id}")
    normal_receipt = client.json("GET", f"/api/jobs/{job_id}/receipt")
    public = json.dumps({"status": normal_status, "receipt": normal_receipt})
    assert capability not in public
    assert "agent_authorization" not in public
    assert "lease_capability_hash" not in public
    initial_expiry = normal_status["lease_expires_ts"]
    with pytest.raises(ApiError, match="HTTP 403"):
        client.json(
            "POST",
            f"/api/agents/{host}/telemetry",
            json_body={"running": [{"job_id": job_id, "tree_rss_mb": 1}]},
        )
    assert client.json("GET", f"/api/jobs/{job_id}")["lease_expires_ts"] == initial_expiry
    host_status = next(
        row for row in client.json("GET", "/api/hosts") if row["name"] == host
    )
    assert host_status["telemetry"]["running"] == []
    assert not any(
        event["kind"] == "job.telemetry"
        for event in client.json("GET", f"/api/jobs/{job_id}/events")
    )
    agent.json(
        "POST",
        f"/api/agents/{host}/telemetry",
        json_body={
            "running": [
                {
                    "job_id": job_id,
                    "tree_rss_mb": 1,
                    "lease_authorization": {
                        "lease_id": authority["lease_id"],
                        "capability": capability,
                    },
                }
            ]
        },
    )
    assert client.json("GET", f"/api/jobs/{job_id}")["lease_expires_ts"] >= initial_expiry
    public_hosts = json.dumps(client.json("GET", "/api/hosts"))
    assert capability not in public_hosts
    assert "lease_authorization" not in public_hosts
    assert client.request("GET", f"/api/payloads/{job_id}")[0] == 403
    with pytest.raises(ApiError, match="HTTP 403"):
        client.json(
            "POST",
            f"/api/jobs/{job_id}/events",
            json_body={"state": "preparing"},
        )
    with pytest.raises(ApiError, match="HTTP 403"):
        client.json(
            "POST",
            f"/api/jobs/{job_id}/payload/executed",
            json_body={
                "host": host,
                "sha256": packed_sha256,
                "size_bytes": len(packed),
            },
        )
    assert (
        wrong_agent.request(
            "GET", f"/api/payloads/{job_id}", headers=lease_headers
        )[0]
        == 403
    )
    assert (
        agent.request(
            "GET",
            f"/api/payloads/{job_id}",
            headers={**lease_headers, "X-MRun-Lease-Capability": "wrong"},
        )[0]
        == 403
    )

    agent.json(
        "POST",
        f"/api/jobs/{job_id}/events",
        json_body={"state": "preparing"},
        headers=lease_headers,
    )
    download_status, downloaded, _download_headers = agent.request(
        "GET", f"/api/payloads/{job_id}", headers=lease_headers
    )
    assert download_status == 200
    assert downloaded == packed
    agent.json(
        "POST",
        f"/api/jobs/{job_id}/payload/executed",
        json_body={
            "host": host,
            "sha256": packed_sha256,
            "size_bytes": len(packed),
        },
        headers=lease_headers,
    )
    agent.json(
        "POST",
        f"/api/jobs/{job_id}/events",
        json_body={"state": "running"},
        headers=lease_headers,
    )
    # The legacy external-run routes must not cross-terminate or renew an agent job,
    # even after exact payload execution has been attested.
    with pytest.raises(ApiError, match="HTTP 409"):
        client.json(
            "POST",
            f"/api/local-runs/{job_id}/heartbeat",
            json_body={"tree_rss_mb": 999_999},
        )
    with pytest.raises(ApiError, match="HTTP 409"):
        client.json(
            "POST",
            f"/api/local-runs/{job_id}/finish",
            json_body={
                "state": "succeeded",
                "result": {"status": "controller-forged"},
            },
        )
    assert client.json("GET", f"/api/jobs/{job_id}")["state"] == "running"
    log_status, _log_body, _log_headers = agent.request(
        "POST",
        f"/api/jobs/{job_id}/logs?offset=0",
        raw_body=b"guarded log\n",
        headers=lease_headers,
    )
    assert log_status == 200
    agent.json(
        "POST",
        f"/api/jobs/{job_id}/events",
        json_body={
            "state": "succeeded",
            "result": {"status": "ok", "returncode": 0},
        },
        headers=lease_headers,
    )
    # Terminal transition immediately revokes the lease capability; terminal replays
    # are explicitly rejected rather than resurrecting or duplicating estimates.
    with pytest.raises(ApiError, match="HTTP 403"):
        agent.json(
            "POST",
            f"/api/jobs/{job_id}/events",
            json_body={"state": "succeeded"},
            headers=lease_headers,
        )


def test_guarded_endpoint_takeover_stale_replay_expiry_and_terminal_convergence(fleet):
    from mrun.client.api import ApiError

    client = fleet
    packed = b"endpoint takeover payload"
    scope = {
        "experiment": "guarded-endpoint-takeover",
        "config_selector": {"suite_fingerprint": "takeover-v1"},
    }
    claim_a_request = {
        "claim_key": "fleet:guarded-endpoint-takeover:v1",
        "owner_token": "endpoint-owner-a",
        "ttl_s": 300.0,
        "scope": scope,
        "metadata": {"owner": "a"},
    }
    claim_a = client.json(
        "POST", "/api/admission-claims/acquire", json_body=claim_a_request
    )

    def guarded_request(claim: dict) -> dict:
        return {
            "experiment": scope["experiment"],
            "client_run_id": "guarded-endpoint-takeover:cell-0",
            "cmd": [sys.executable, "worker.py"],
            "config": {"suite_fingerprint": "takeover-v1"},
            "needs": {"host": "takeover-never-host"},
            "reservation": {"ram_mb": 128, "cpu_threads": 1},
            "plans": {"takeover-never-host": {"backend": "test"}},
            "payload_kind": "shipped",
            "payload": {
                "sha256": hashlib.sha256(packed).hexdigest(),
                "size_bytes": len(packed),
            },
            "admission": {
                "claim_key": claim["claim_key"],
                "owner_token": claim["owner_token"],
                "fencing_epoch": claim["fencing_epoch"],
                "idempotency_key": "cell-0",
            },
        }

    first = client.json(
        "POST", "/api/jobs/guarded", json_body=guarded_request(claim_a)
    )
    client.json(
        "POST",
        "/api/admission-claims/release",
        json_body={
            "claim_key": claim_a["claim_key"],
            "owner_token": claim_a["owner_token"],
            "fencing_epoch": claim_a["fencing_epoch"],
        },
    )
    claim_b = client.json(
        "POST",
        "/api/admission-claims/acquire",
        json_body={**claim_a_request, "owner_token": "endpoint-owner-b"},
    )
    with pytest.raises(ApiError, match="HTTP 409"):
        client.json(
            "POST", "/api/jobs/guarded", json_body=guarded_request(claim_a)
        )
    takeover = client.json(
        "POST", "/api/jobs/guarded", json_body=guarded_request(claim_b)
    )
    assert takeover["job_id"] == first["job_id"]
    assert takeover["created"] is False
    assert takeover["admission"]["fencing_epoch"] == claim_b["fencing_epoch"]
    assert len(
        client.json(
            "GET",
            "/api/jobs?client_run_id=guarded-endpoint-takeover:cell-0",
        )
    ) == 1

    stale_status, _stale_body, _stale_headers = client.request(
        "PUT",
        f"/api/jobs/{first['job_id']}/payload",
        raw_body=packed,
        headers={
            "X-MRun-Admission-Owner": claim_a["owner_token"],
            "X-MRun-Admission-Epoch": str(claim_a["fencing_epoch"]),
        },
    )
    assert stale_status == 409
    seal_status, _seal_body, _seal_headers = client.request(
        "PUT",
        f"/api/jobs/{first['job_id']}/payload",
        raw_body=packed,
        headers={
            "X-MRun-Admission-Owner": claim_b["owner_token"],
            "X-MRun-Admission-Epoch": str(claim_b["fencing_epoch"]),
        },
    )
    assert seal_status == 200
    client.json(
        "POST",
        "/api/admission-claims/release",
        json_body={
            "claim_key": claim_b["claim_key"],
            "owner_token": claim_b["owner_token"],
            "fencing_epoch": claim_b["fencing_epoch"],
        },
    )
    expired_replay, _expired_body, _expired_headers = client.request(
        "PUT",
        f"/api/jobs/{first['job_id']}/payload",
        raw_body=packed,
        headers={
            "X-MRun-Admission-Owner": claim_b["owner_token"],
            "X-MRun-Admission-Epoch": str(claim_b["fencing_epoch"]),
        },
    )
    assert expired_replay == 409
    assert client.json("POST", f"/api/jobs/{first['job_id']}/cancel") == {
        "state": "cancelled"
    }
    terminal = client.json(
        "POST", "/api/jobs/guarded", json_body=guarded_request(claim_a)
    )
    assert terminal["job_id"] == first["job_id"]
    assert terminal["state"] == "cancelled"
    assert terminal["created"] is False


def test_general_token_old_agent_remains_compatible_for_ordinary_job(fleet):
    client = fleet
    host = "ordinary-legacy-host"
    _register_synthetic_host(client, host, guarded=False)
    submitted = client.json(
        "POST",
        "/api/jobs",
        json_body={
            "experiment": "ordinary-rolling-compat",
            "client_run_id": "ordinary-rolling-compat:0",
            "cmd": [sys.executable, "-c", "print('legacy')"],
            "config": {"compat": "ordinary"},
            "needs": {"host": host},
            "reservation": {"ram_mb": 128, "cpu_threads": 1},
        },
    )
    leased = client.json("POST", f"/api/agents/{host}/lease", timeout_s=5.0)
    assert leased["job_id"] == submitted["job_id"]
    assert "agent_authorization" not in leased
    for state in ("preparing", "running", "succeeded"):
        response = client.json(
            "POST",
            f"/api/jobs/{submitted['job_id']}/events",
            json_body={"state": state, "result": {}},
        )
        assert response == {"ok": True}


def test_guarded_shipped_job_is_fenced_sealed_and_executed_exactly(fleet):
    from mrun.client.api import ApiError

    api = fleet
    script = b"print('guarded-custody-ok')\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("worker.py")
        member.size = len(script)
        member.mtime = 0
        archive.addfile(member, io.BytesIO(script))
    packed = buffer.getvalue()
    packed_sha256 = hashlib.sha256(packed).hexdigest()
    scope = {
        "experiment": "guarded-fleet-custody",
        "config_selector": {"suite_fingerprint": "fleet-v2"},
    }
    claim_request = {
        "claim_key": "fleet:guarded-custody:v2",
        "owner_token": "fleet-owner-v2",
        "ttl_s": 300.0,
        "scope": scope,
        "metadata": {"test": True},
    }
    claim = api.json("POST", "/api/admission-claims/acquire", json_body=claim_request)
    plan = {"backend": "python-test", "dtype": "none", "threads": 1}
    guarded_request = {
        "experiment": scope["experiment"],
        "client_run_id": "guarded-fleet-custody:cell-0",
        "cmd": [sys.executable, "worker.py"],
        "config": {"suite_fingerprint": "fleet-v2", "task_family": "test"},
        "needs": {"host": "testhost"},
        "reservation": {"ram_mb": 500, "cpu_threads": 1},
        "client_estimate": None,
        "plans": {"testhost": plan},
        "payload_kind": "shipped",
        "payload": {"sha256": packed_sha256, "size_bytes": len(packed)},
        "env_alias": None,
        "timeout_s": 30,
        "priority": 1,
        "admission": {
            "claim_key": claim["claim_key"],
            "owner_token": claim["owner_token"],
            "fencing_epoch": claim["fencing_epoch"],
            "idempotency_key": "cell-0",
        },
    }
    receipt = api.json("POST", "/api/jobs/guarded", json_body=guarded_request)
    assert receipt["state"] == "awaiting_payload"
    assert receipt["created"] is True
    assert receipt["submitted_request"]["plans"] == {"testhost": plan}
    encoded_request = json.dumps(
        receipt["submitted_request"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(encoded_request).hexdigest() == receipt["submitted_request_sha256"]
    job_id = receipt["job_id"]
    time.sleep(0.5)
    assert api.json("GET", f"/api/jobs/{job_id}")["state"] == "awaiting_payload"
    unknown_status, _unknown_body, _unknown_headers = api.request(
        "PUT", "/api/jobs/job-does-not-exist/payload", raw_body=packed
    )
    assert unknown_status == 404

    # The persistent protected scope cannot be reached through the old endpoint.
    legacy = {
        key: value
        for key, value in guarded_request.items()
        if key not in {"admission", "payload"}
    }
    with pytest.raises(ApiError, match="HTTP 409"):
        api.json("POST", "/api/jobs", json_body=legacy)

    status, _body, _headers = api.request(
        "PUT",
        f"/api/jobs/{job_id}/payload",
        raw_body=b"tampered",
        headers={
            "X-MRun-Admission-Owner": claim["owner_token"],
            "X-MRun-Admission-Epoch": str(claim["fencing_epoch"]),
        },
    )
    assert status == 409
    status, body, _headers = api.request(
        "PUT",
        f"/api/jobs/{job_id}/payload",
        raw_body=packed,
        headers={
            "X-MRun-Admission-Owner": claim["owner_token"],
            "X-MRun-Admission-Epoch": str(claim["fencing_epoch"]),
        },
    )
    assert status == 200
    seal = json.loads(body)
    assert seal["created"] is True
    assert seal["payload_custody"]["sealed"]["sha256"] == packed_sha256

    # Exact replay is harmless while queued; different bytes remain forbidden.
    status, body, _headers = api.request(
        "PUT",
        f"/api/jobs/{job_id}/payload",
        raw_body=packed,
        headers={
            "X-MRun-Admission-Owner": claim["owner_token"],
            "X-MRun-Admission-Epoch": str(claim["fencing_epoch"]),
        },
    )
    # If the agent has not claimed it yet, replay is exact-idempotent. Once assigned,
    # even byte-identical PUTs are immutable by contract and return 409.
    assert status in {200, 409}
    if status == 200:
        assert json.loads(body)["idempotent_replay"] is True

    terminal = _wait_state(api, job_id, {"succeeded", "failed"})
    assert terminal["state"] == "succeeded"
    custody = terminal["payload_custody"]
    assert custody["declared"]["sha256"] == packed_sha256
    assert custody["sealed"]["sha256"] == packed_sha256
    assert custody["executed"]["sha256"] == packed_sha256
    assert custody["executed"]["host"] == "testhost"
    final_receipt = api.json("GET", f"/api/jobs/{job_id}/receipt")
    binding = final_receipt["resolved_execution"]["plan_binding"]
    assert binding["assigned_host"] == "testhost"
    assert binding["selected_plan"] == plan
    assert binding["submitted_plan"] == plan
    assert binding["matches_submitted_plan"] is True

    status, _body, _headers = api.request(
        "PUT",
        f"/api/jobs/{job_id}/payload",
        raw_body=packed,
        headers={
            "X-MRun-Admission-Owner": claim["owner_token"],
            "X-MRun-Admission-Epoch": str(claim["fencing_epoch"]),
        },
    )
    assert status == 409


def test_echo_job_succeeds_with_logs(fleet):
    api = fleet
    job_id = _submit(api, "echo", "import time; print('hello-fleet'); time.sleep(1.5)")
    job = _wait_state(api, job_id, {"succeeded", "failed"}, timeout=90)
    assert job["state"] == "succeeded", job
    status, body, _ = api.request("GET", f"/api/jobs/{job_id}/logs?offset=0")
    assert b"hello-fleet" in body
    assert job["result"]["peak_rss_mb"] > 0


def test_failed_job_has_structured_diagnostics(fleet):
    api = fleet
    code = "import sys; print('intentional-failure', flush=True); sys.exit(7)"
    job_id = _submit(api, "diagnostics", code)
    job = _wait_state(api, job_id, {"failed"}, timeout=90)
    failure = (job.get("result") or {}).get("failure") or {}
    assert failure["kind"] == "process_exit"
    assert failure["phase"] == "process.finalize"
    assert failure["returncode"] == 7
    assert "intentional-failure" in failure["log_tail"]
    events = api.json("GET", f"/api/jobs/{job_id}/events?limit=100")
    assert any(event["kind"] == "job.agent_phase" for event in events)


def test_memory_hog_killed_at_reservation(fleet):
    api = fleet
    # kill ceiling = max(res*1.1, res+512MB): a 250MB reservation is enforced at
    # 762MB, so the hog must overshoot the absolute grace, not just the 1.1 factor.
    # Incompressible bytes + an active touch loop — zero-filled or idle pages get
    # compressed/swapped out of RSS on a pressured macOS host and silently fit.
    hog = (
        "import os, time\n"
        "x = bytearray(900*1024*1024)\n"
        "for i in range(0, len(x), 65536): x[i:i+65536] = os.urandom(min(65536, len(x)-i))\n"
        "t = time.time()\n"
        "while time.time() - t < 60:\n"
        "    for i in range(0, len(x), 4096): x[i] ^= 1\n"
    )
    job_id = _submit(api, "hog", hog, ram_mb=250)
    job = _wait_state(api, job_id, {"killed_ram", "failed", "succeeded"}, timeout=90)
    assert job["state"] == "killed_ram", job
    assert "ceiling" in (job.get("status_detail") or "")


def test_oversized_pair_queues_then_runs(fleet):
    api = fleet
    # size off the LIVE headroom (total - baseline - margin), so one fits and two don't
    h = None
    for _ in range(30):  # first telemetry tick may not have landed yet
        h = next(x for x in api.json("GET", "/api/hosts") if x["name"] == "testhost")
        if h.get("telemetry"):
            break
        time.sleep(0.5)
    margin = max(0.15 * h["ram_total_mb"], 2048.0)
    headroom = h["telemetry"]["ram_free_mb"] - margin
    if headroom <= 2500:
        pytest.skip(f"host too loaded for this test (headroom {headroom:.0f}MB)")
    # Each current kill ceiling fits comfortably; two exceed the stable headroom.
    # Leave a wide margin rather than relying on a historical absolute grace.
    big = headroom * 0.6
    code = "import time; time.sleep(6); print('done-big')"
    j1 = _submit(api, "big-a", code, ram_mb=big, config={"n": 1, "code": code})
    j2 = _submit(api, "big-b", code, ram_mb=big, config={"n": 2, "code": code})
    try:
        _wait_state(api, j1, {"running", "succeeded"}, timeout=60)
    except AssertionError:
        # sized off a headroom snapshot — if the dev box's baseline grew past it
        # mid-test, "queued under load" is not the behavior under test
        detail = api.json("GET", f"/api/jobs/{j1}").get("status_detail") or ""
        if detail.startswith("waiting:"):
            api.json("POST", f"/api/jobs/{j1}/cancel")
            api.json("POST", f"/api/jobs/{j2}/cancel")
            pytest.skip(f"host baseline grew mid-test: {detail[:120]}")
        raise
    # while j1 runs, j2 must still be queued
    job2 = api.json("GET", f"/api/jobs/{j2}")
    assert job2["state"] == "queued", job2["state"]
    job2 = _wait_state(api, j2, {"succeeded"}, timeout=120)
    assert job2["state"] == "succeeded"


def test_small_jobs_coadmit_past_legacy_slot_cap(fleet):
    api = fleet
    code = "import time; print('coadmit-start', flush=True); time.sleep(12); print('coadmit-done')"
    ids = [
        _submit(api, "coadmit", code, ram_mb=200, config={"code": code, "idx": i})
        for i in range(3)
    ]
    deadline = time.time() + 45
    last: dict[str, str] = {}
    while time.time() < deadline:
        jobs = {jid: api.json("GET", f"/api/jobs/{jid}") for jid in ids}
        last = {jid: job["state"] for jid, job in jobs.items()}
        if sum(job["state"] == "running" for job in jobs.values()) == 3:
            break
        time.sleep(0.25)
    else:
        raise AssertionError(f"three fitting jobs never co-admitted; last states={last}")

    for jid in ids:
        job = _wait_state(api, jid, {"succeeded", "failed"}, timeout=90)
        assert job["state"] == "succeeded", job


def test_cancel_running_job(fleet):
    api = fleet
    job_id = _submit(api, "sleeper", "import time; time.sleep(120)")
    _wait_state(api, job_id, {"running"}, timeout=60)
    api.json("POST", f"/api/jobs/{job_id}/cancel")
    job = _wait_state(api, job_id, {"cancelled"}, timeout=60)
    assert job["state"] == "cancelled"


def test_history_reservation_on_resubmit(fleet):
    api = fleet
    # An instant print can exit between RSS samples on a fast Linux worker.
    # Hold this calibration process through at least one agent sampling tick.
    code = "import time; print('calib'); time.sleep(1.5)"
    cfg = {"model": None, "code": code, "tag": "calib"}
    j1 = _submit(api, "calib", code, ram_mb=400, config=cfg)
    measured = _wait_state(api, j1, {"succeeded"}, timeout=90)
    assert measured["result"]["peak_rss_mb"] > 0, measured["result"]
    # resubmit the SAME config without a declared reservation -> history should win
    from mrun.client.submit import submit

    j2 = submit(
        experiment="calib",
        cmd=[sys.executable, "-c", code],
        config=cfg,
        env_alias=None,
        api=api,
    )
    job2 = api.json("GET", f"/api/jobs/{j2}")
    assert job2["reservation"]["source"] == "history", job2["reservation"]
    _wait_state(api, j2, {"succeeded"}, timeout=90)


def test_hosts_endpoint_reports_testhost(fleet):
    api = fleet
    hosts = api.json("GET", "/api/hosts")
    names = [h["name"] for h in hosts]
    assert "testhost" in names
    h = next(h for h in hosts if h["name"] == "testhost")
    assert h["telemetry"] is not None
    assert h["ram_total_mb"] > 1000


# --- adversarial regressions (found by doc red-team 2026-07-14) -----------------


def test_submit_rejects_poison_reservation(fleet):
    """A str ram_mb used to poison the queue and 500 every lease poll."""
    api = fleet
    from mrun.client.api import ApiError

    body = {
        "experiment": "poison",
        "client_run_id": "poison-1",
        "cmd": ["echo", "hi"],
        "reservation": {"ram_mb": "lots"},
    }
    with pytest.raises(ApiError) as exc:
        api.json("POST", "/api/jobs", json_body=body)
    assert exc.value.status == 422


def test_submit_validation_4xx_not_500(fleet):
    api = fleet
    from mrun.client.api import ApiError

    cases = [
        {"experiment": "x", "cmd": ["echo"]},  # no client_run_id
        {"experiment": "x", "client_run_id": "c1"},  # no cmd
        {"experiment": "x", "client_run_id": "c1", "cmd": ["echo"], "needs": "cuda"},
        {"experiment": "x", "client_run_id": "c1", "cmd": ["echo"], "priority": "high"},
        {"experiment": "x", "client_run_id": "c1", "cmd": ["echo"], "timeout_s": -5},
    ]
    for body in cases:
        with pytest.raises(ApiError) as exc:
            api.json("POST", "/api/jobs", json_body=body)
        assert exc.value.status == 422, body


def test_priority_clamped(fleet):
    api = fleet
    resp = api.json(
        "POST",
        "/api/jobs",
        json_body={
            "experiment": "prio",
            "client_run_id": "prio-1",
            "cmd": ["echo"],
            "needs": {"host": "no-such-host"},  # never scheduled
            "priority": 10**18,
        },
    )
    job = api.json("GET", f"/api/jobs/{resp['job_id']}")
    assert job["priority"] == 100
    assert "unschedulable" in (job.get("status_detail") or "")
    api.json("POST", f"/api/jobs/{resp['job_id']}/cancel")


def test_inadmissible_high_priority_head_does_not_block_later_lease(fleet):
    api = fleet
    blocked = api.json(
        "POST",
        "/api/jobs",
        json_body={
            "experiment": "inadmissible-head",
            "client_run_id": "inadmissible-head-1",
            "cmd": ["echo", "blocked"],
            "config": {"task_family": "forward"},
            "needs": {"host": "testhost", "cuda": True},
            "reservation": {
                "ram_mb": 1_000_000_000,
                "vram_mb": 1_000_000_000,
                "cpu_threads": 1,
            },
            "priority": 100,
            "allow_unschedulable": True,
        }
    )
    blocked_id = blocked["job_id"]
    ready_id = None
    try:
        blocked_status = api.json("GET", f"/api/jobs/{blocked_id}")
        assert blocked_status["state"] == "queued"
        assert blocked_status["priority"] == 100
        assert blocked_status["status_detail"].startswith("unschedulable:")
        assert blocked["admission_outlook"]["status"] == "unschedulable"

        ready_id = _submit(
            api,
            "later-admissible",
            "print('later-admissible')",
            ram_mb=200,
            config={"code": "print('later-admissible')", "task_family": "forward"},
            needs={"host": "testhost"},
        )
        ready_status = api.json("GET", f"/api/jobs/{ready_id}")
        assert blocked_status["created_ts"] < ready_status["created_ts"]
        assert blocked_status["priority"] > ready_status["priority"]

        ready_status = _wait_state(api, ready_id, {"succeeded"}, timeout=90)
        assert ready_status["state"] == "succeeded"
        ready_events = api.json("GET", f"/api/jobs/{ready_id}/events?limit=50")
        assert any(event["kind"] == "job.claimed" for event in ready_events)

        blocked_status = api.json("GET", f"/api/jobs/{blocked_id}")
        assert blocked_status["state"] == "queued"
        assert blocked_status["status_detail"].startswith("unschedulable:")
        blocked_events = api.json("GET", f"/api/jobs/{blocked_id}/events?limit=50")
        assert any(
            event["kind"] == "admission.status"
            and event["reason"] == "unschedulable"
            for event in blocked_events
        )
    finally:
        if api.json("GET", f"/api/jobs/{blocked_id}")["state"] == "queued":
            api.json("POST", f"/api/jobs/{blocked_id}/cancel")


def test_no_terminal_resurrection(fleet):
    api = fleet
    job_id = _submit(api, "resurrect", "print('short')")
    _wait_state(api, job_id, {"succeeded"}, timeout=90)
    from mrun.client.api import ApiError

    with pytest.raises(ApiError) as exc:
        api.json("POST", f"/api/jobs/{job_id}/events", json_body={"state": "running"})
    assert exc.value.status == 409
    with pytest.raises(ApiError) as exc:
        api.json("POST", f"/api/jobs/{job_id}/events", json_body={"state": "succeeded"})
    assert exc.value.status == 409
    assert api.json("GET", f"/api/jobs/{job_id}")["state"] == "succeeded"


def test_no_events_on_queued_job(fleet):
    api = fleet
    resp = api.json(
        "POST",
        "/api/jobs",
        json_body={
            "experiment": "qevent",
            "client_run_id": "qevent-1",
            "cmd": ["echo"],
            "needs": {"host": "no-such-host"},
        },
    )
    from mrun.client.api import ApiError

    with pytest.raises(ApiError) as exc:
        api.json(
            "POST", f"/api/jobs/{resp['job_id']}/events", json_body={"state": "succeeded"}
        )
    assert exc.value.status == 409
    job = api.json("GET", f"/api/jobs/{resp['job_id']}")
    assert "pinned host" in (job.get("status_detail") or "")
    api.json("POST", f"/api/jobs/{resp['job_id']}/cancel")


def test_negative_log_offset_ok(fleet):
    api = fleet
    status, body, _ = api.request("GET", "/api/jobs/no-such-job/logs?offset=-5")
    assert status == 200
    assert body == b""


def test_log_long_polls_do_not_starve_health(fleet):
    from concurrent.futures import ThreadPoolExecutor

    api = fleet
    follower_count = 48
    start = threading.Barrier(follower_count + 1)

    def follow_empty_log():
        start.wait()
        return api.request(
            "GET",
            "/api/jobs/no-such-job/logs?offset=0&wait_s=1.5",
            timeout_s=5.0,
        )

    with ThreadPoolExecutor(max_workers=follower_count) as pool:
        followers = [pool.submit(follow_empty_log) for _ in range(follower_count)]
        start.wait()
        time.sleep(0.25)
        started = time.monotonic()
        assert api.json("GET", "/healthz", timeout_s=2.0) == {"ok": True}
        elapsed = time.monotonic() - started
        for future in followers:
            status, body, _headers = future.result(timeout=5.0)
            assert status == 200
            assert body == b""

    assert elapsed < 0.75


# --- local (external) runs — P0 crash-safety ------------------------------------


def test_local_run_lifecycle(fleet):
    api = fleet
    resp = api.json(
        "POST",
        "/api/local-runs",
        json_body={
            "host": "testhost",
            "experiment": "mx:local-exp",
            "reservation": {"ram_mb": 700, "cpu_threads": 1},
            "config": {"model": "distilgpt2"},
        },
    )
    job_id = resp["job_id"]
    job = api.json("GET", f"/api/jobs/{job_id}")
    assert job["state"] == "running"
    assert job["assigned_host"] == "testhost"
    assert job["payload_kind"] == "external"

    api.json(
        "POST",
        f"/api/local-runs/{job_id}/heartbeat",
        json_body={"tree_rss_mb": 512.0, "tree_vram_mb": 128.0},
    )
    job = api.json("GET", f"/api/jobs/{job_id}")
    assert job["external_rss_mb"] == 512.0
    assert job["external_vram_mb"] == 128.0

    api.json(
        "POST",
        f"/api/local-runs/{job_id}/finish",
        json_body={"state": "succeeded", "result": {"peak_rss_mb": 640.0, "elapsed_s": 3.0}},
    )
    job = api.json("GET", f"/api/jobs/{job_id}")
    assert job["state"] == "succeeded"
    assert job["result"]["peak_rss_mb"] == 640.0
    # terminal is final for local runs too
    from mrun.client.api import ApiError

    with pytest.raises(ApiError) as exc:
        api.json("POST", f"/api/local-runs/{job_id}/finish", json_body={"state": "failed"})
    assert exc.value.status == 409
    # and the terminal event wrote an estimates row (resubmit-style history visible via
    # a job with same client_run_id would pick it up; here just sanity-check hosts view)


def test_job_events_endpoint_records_lifecycle(fleet):
    api = fleet
    job_id = _submit(api, "events", "print('eventful')")
    job = _wait_state(api, job_id, {"succeeded"}, timeout=90)
    events = api.json("GET", f"/api/jobs/{job_id}/events?limit=50")
    kinds = [e["kind"] for e in events]

    assert job["state"] == "succeeded"
    assert "job.submit" in kinds
    assert "job.claimed" in kinds
    assert "job.finished" in kinds
    assert any(e["kind"] == "job.finished" and e["state"] == "succeeded" for e in events)



def test_local_run_reservation_blocks_fleet_job(fleet):
    api = fleet
    h = next(x for x in api.json("GET", "/api/hosts") if x["name"] == "testhost")
    margin = max(0.15 * h["ram_total_mb"], 2048.0)
    headroom = h["telemetry"]["ram_free_mb"] - margin
    assert headroom > 500, f"host too loaded for this test (headroom {headroom:.0f}MB)"
    big = headroom * 0.7
    resp = api.json(
        "POST",
        "/api/local-runs",
        json_body={"host": "testhost", "experiment": "mx:hog", "reservation": {"ram_mb": big}},
    )
    local_id = resp["job_id"]
    try:
        code = "print('after-local')"
        j = _submit(api, "blocked-by-local", code, ram_mb=big, config={"code": code, "n": 3})
        time.sleep(3.0)  # let a few lease polls happen
        assert api.json("GET", f"/api/jobs/{j}")["state"] == "queued"
    finally:
        api.json("POST", f"/api/local-runs/{local_id}/finish", json_body={"state": "succeeded"})
    _wait_state(api, j, {"succeeded"}, timeout=90)


def test_abandoned_local_run_goes_lost(fleet):
    api = fleet
    resp = api.json(
        "POST",
        "/api/local-runs",
        json_body={"host": "testhost", "experiment": "mx:crashy", "reservation": {"ram_mb": 300}},
    )
    job_id = resp["job_id"]
    # no heartbeats: lease (60s) must expire and free the reservation. Fast-forward by
    # rewinding the lease instead of sleeping a minute.
    from mrun.server.app import db as server_db

    server_db.update_job(job_id, lease_expires_ts=time.time() - 1)
    server_db.expire_leases()
    job = api.json("GET", f"/api/jobs/{job_id}")
    assert job["state"] == "lost"


# --- P1: inventory + per-host plans -----------------------------------------------


def test_inventory_roundtrip(fleet):
    api = fleet
    rows = [
        {"model": "qwen2.5-0.5b", "kind": "weights", "bytes": 1_000_000, "path": "/m/q"},
        {"model": "qwen2.5-0.5b", "kind": "qstore", "bytes": 300_000, "path": "/m/qs"},
    ]
    api.json(
        "POST",
        "/api/agents/testhost/inventory",
        json_body={"models": rows},
        headers={"X-MRun-Agent-Token": AGENT_TOKEN},
    )
    inv = api.json("GET", "/api/models?host=testhost")
    got = {(r["model"], r["kind"]) for r in inv}
    assert ("qwen2.5-0.5b", "weights") in got and ("qwen2.5-0.5b", "qstore") in got
    h = next(x for x in api.json("GET", "/api/hosts") if x["name"] == "testhost")
    assert any(m["model"] == "qwen2.5-0.5b" for m in h["models"])
    # full replace: pushing a new list drops the old rows
    api.json(
        "POST",
        "/api/agents/testhost/inventory",
        json_body={"models": []},
        headers={"X-MRun-Agent-Token": AGENT_TOKEN},
    )
    assert api.json("GET", "/api/models?host=testhost") == []


def test_plan_flows_to_child_env(fleet):
    api = fleet

    code = (
        "import os, json; p = os.environ.get('MRUN_PLAN');"
        "print('PLAN_DTYPE=' + (json.loads(p)['dtype'] if p else 'none'));"
        "print('BATCH=' + os.environ.get('GATHER_MAX_BATCH', 'none'))"
    )
    plans = {
        "testhost": {
            "model": "m-test",
            "backend": "hf",
            "device": "cpu",
            "dtype": "bfloat16",
            "max_batch": 4,
            "threads": 1,
            "ram_limit_mb": 400.0,
            "est_ram_mb": 300.0,
            "est_vram_mb": 0.0,
            "weights_gb": 0.01,
        }
    }
    body = {
        "experiment": "plan-env",
        "client_run_id": "plan-env-1",
        "cmd": [sys.executable, "-c", code],
        "config": {"model": "m-test"},
        "plans": plans,
    }
    resp = api.json("POST", "/api/jobs", json_body=body)
    job = _wait_state(api, resp["job_id"], {"succeeded", "failed"}, timeout=90)
    assert job["state"] == "succeeded", job
    # reservation was re-stamped from the plan at lease time
    assert job["reservation"]["source"] == "plan"
    assert job["reservation"]["ram_mb"] == 400.0
    assert job["plan"]["dtype"] == "bfloat16"
    _, body_bytes, _ = api.request("GET", f"/api/jobs/{resp['job_id']}/logs?offset=0")
    assert b"PLAN_DTYPE=bfloat16" in body_bytes
    assert b"BATCH=4" in body_bytes


def test_register_requires_host(fleet):
    api = fleet
    from mrun.client.api import ApiError
    from mrun.protocol import PROTOCOL_VERSION

    with pytest.raises(ApiError) as exc:
        api.json(
            "POST", "/api/agents/register", json_body={"protocol_version": PROTOCOL_VERSION}
        )
    assert exc.value.status == 422


# --- UI endpoints: priority reorder, run-link meta, host removal -----------------


def _submit_pinned(api, experiment, run_suffix, priority=0):
    """A job pinned to a nonexistent host stays queued forever (cancel to clean up)."""
    resp = api.json(
        "POST",
        "/api/jobs",
        json_body={
            "experiment": experiment,
            "client_run_id": f"{experiment}-{run_suffix}",
            "cmd": ["echo"],
            "needs": {"host": "no-such-host"},
            "priority": priority,
        },
    )
    return resp["job_id"]


def test_control_drain_blocks_new_leases_then_resume_runs(fleet):
    api = fleet
    job_id = None
    api.json("POST", "/api/control", json_body={"draining": True, "reason": "test-drain"})
    try:
        code = "print('after-drain')"
        job_id = _submit(api, "drain", code, config={"code": code, "tag": "drain"})
        time.sleep(2.0)
        job = api.json("GET", f"/api/jobs/{job_id}")
        assert job["state"] == "queued"

        control = api.json("GET", "/api/control")
        assert control["draining"] is True
        assert control["queued"] >= 1
        assert control["safe_to_shutdown"] == (control["active"] == 0)
    finally:
        api.json("POST", "/api/control", json_body={"draining": False, "reason": "test-resume"})

    if job_id:
        job = _wait_state(api, job_id, {"succeeded"}, timeout=90)
        assert job["state"] == "succeeded"


def test_restart_terminal_job_clones_queued_job(fleet):
    api = fleet
    code = "print('restart-source')"
    source_id = _submit(api, "restart-src", code, config={"code": code, "tag": "restart"})
    _wait_state(api, source_id, {"succeeded"}, timeout=90)

    api.json("POST", "/api/control", json_body={"draining": True, "reason": "test-restart"})
    try:
        restarted = api.json("POST", f"/api/jobs/{source_id}/restart")
        new_id = restarted["job_id"]
        assert new_id != source_id
        job = api.json("GET", f"/api/jobs/{new_id}")
        assert job["state"] == "queued"
        assert job["experiment"] == "restart-src"
        assert job["cmd"] == [sys.executable, "-c", code]
        assert job["meta"]["restart_of"] == source_id

        from mrun.client.api import ApiError

        with pytest.raises(ApiError) as exc:
            api.json("POST", f"/api/jobs/{new_id}/restart")
        assert exc.value.status == 409
    finally:
        api.json("POST", "/api/control", json_body={"draining": False, "reason": "test-resume"})

    job = _wait_state(api, new_id, {"succeeded"}, timeout=90)
    assert job["state"] == "succeeded"


def test_priority_endpoint_reorders_queue(fleet):
    api = fleet
    a = _submit_pinned(api, "reorder", "a")
    b = _submit_pinned(api, "reorder", "b")
    try:
        queued = [j["job_id"] for j in api.json("GET", "/api/jobs?state=queued")]
        assert queued.index(a) < queued.index(b)  # same priority -> created order

        api.json("POST", f"/api/jobs/{b}/priority", json_body={"priority": 50})
        queued = [j["job_id"] for j in api.json("GET", "/api/jobs?state=queued")]
        assert queued.index(b) < queued.index(a)
        assert api.json("GET", f"/api/jobs/{b}")["priority"] == 50

        # clamp + validation
        api.json("POST", f"/api/jobs/{b}/priority", json_body={"priority": 10**9})
        assert api.json("GET", f"/api/jobs/{b}")["priority"] == 100
        from mrun.client.api import ApiError

        with pytest.raises(ApiError) as exc:
            api.json("POST", f"/api/jobs/{b}/priority", json_body={"priority": "high"})
        assert exc.value.status == 422
    finally:
        for j in (a, b):
            api.json("POST", f"/api/jobs/{j}/cancel")

    # terminal jobs refuse re-prioritization
    from mrun.client.api import ApiError

    with pytest.raises(ApiError) as exc:
        api.json("POST", f"/api/jobs/{a}/priority", json_body={"priority": 1})
    assert exc.value.status == 409


def test_job_metadata(fleet):
    api = fleet
    job_id = _submit_pinned(api, "meta", "1")
    try:
        api.json(
            "POST",
            f"/api/jobs/{job_id}/meta",
            json_body={"run_id": "abc123def4567890", "label": "px",
                       "nested": {"dropped": True}},
        )
        job = api.json("GET", f"/api/jobs/{job_id}")
        assert job["meta"]["run_id"] == "abc123def4567890"
        assert "nested" not in job["meta"]  # non-scalar values dropped

        # merge, not clobber
        api.json("POST", f"/api/jobs/{job_id}/meta", json_body={"experiment_tag": "e1"})
        meta = api.json("GET", f"/api/jobs/{job_id}")["meta"]
        assert meta["run_id"] == "abc123def4567890" and meta["experiment_tag"] == "e1"
    finally:
        api.json("POST", f"/api/jobs/{job_id}/cancel")






def test_host_delete(fleet):
    api = fleet
    from mrun.client.api import ApiError
    from mrun.protocol import PROTOCOL_VERSION

    api.json(
        "POST",
        "/api/agents/register",
        json_body={"host": "ghosthost", "protocol_version": PROTOCOL_VERSION,
                   "ram_total_mb": 1000},
    )
    assert "ghosthost" in {h["name"] for h in api.json("GET", "/api/hosts")}

    # a host with an active job refuses deletion
    run = api.json(
        "POST",
        "/api/local-runs",
        json_body={"host": "ghosthost", "experiment": "pin",
                   "reservation": {"ram_mb": 100}},
    )
    with pytest.raises(ApiError) as exc:
        api.json("DELETE", "/api/hosts/ghosthost")
    assert exc.value.status == 409
    api.json(
        "POST", f"/api/local-runs/{run['job_id']}/finish", json_body={"state": "succeeded"}
    )

    api.json("DELETE", "/api/hosts/ghosthost")
    assert "ghosthost" not in {h["name"] for h in api.json("GET", "/api/hosts")}
    with pytest.raises(ApiError) as exc:
        api.json("DELETE", "/api/hosts/ghosthost")
    assert exc.value.status == 404


def test_host_limits_put_flips_admission_live(fleet):
    api = fleet
    # disable testhost from the operator API — a fresh submit must stay queued
    out = api.json("PUT", "/api/hosts/testhost/limits", json_body={"enabled": False})
    assert out["effective"]["enabled"] is False
    code = "print('limits-live')"
    jid = _submit(api, "limits-live", code, ram_mb=200, config={"code": code, "v": 1})
    time.sleep(2.5)
    job = api.json("GET", f"/api/jobs/{jid}")
    assert job["state"] == "queued", job["state"]
    assert "disabled" in (job.get("status_detail") or "")
    # re-enable (clear the limit set) — the queued job must now run to completion
    out = api.json("PUT", "/api/hosts/testhost/limits", json_body={})
    assert out["effective"]["enabled"] is True
    job = _wait_state(api, jid, {"succeeded"}, timeout=60)
    assert job["state"] == "succeeded"


def test_host_limits_validation(fleet):
    api = fleet
    from mrun.client.api import ApiError

    with pytest.raises(ApiError) as e:
        api.json("PUT", "/api/hosts/testhost/limits", json_body={"ram_margin_mb": 100})
    assert "1024" in str(e.value)
    with pytest.raises(ApiError):
        api.json("PUT", "/api/hosts/testhost/limits", json_body={"bogus_key": 1})
    with pytest.raises(ApiError):
        api.json("PUT", "/api/hosts/nope/limits", json_body={})


def test_ceiling_kill_auto_retries_with_grown_reservation(fleet):
    api = fleet
    # ~1.2GB incompressible hog vs 250MB reservation (ceiling 762MB) -> killed_ram ->
    # auto-clone(s) grown from the measured peak until it fits (max 2 attempts)
    hog = (
        "import os, time\n"
        "x = bytearray(os.urandom(1200*1024*1024))\n"
        "t = time.time()\n"
        "while time.time() - t < 3:\n"
        "    for i in range(0, len(x), 4096): x[i] ^= 1\n"
        "print('grew-ok')\n"
    )
    jid = _submit(api, "auto-grow", hog, ram_mb=250, config={"code": hog, "v": 3})
    job = _wait_state(api, jid, {"killed_ram", "failed", "succeeded"}, timeout=90)
    assert job["state"] == "killed_ram", job

    def follow_retry(job_id):
        deadline = time.time() + 10
        while time.time() < deadline:
            j = api.json("GET", f"/api/jobs/{job_id}")
            rid = (j.get("meta") or {}).get("auto_retry_job_id")
            if rid:
                return rid
            time.sleep(0.5)
        return None

    cur, final, prev_ram = jid, None, 250.0
    for _ in range(2):  # kill-sample timing decides whether one growth step suffices
        retry_id = follow_retry(cur)
        assert retry_id, f"killed job {cur} did not auto-retry"
        retry = api.json("GET", f"/api/jobs/{retry_id}")
        res = retry["reservation"]
        assert res["source"] == "grown-on-kill"
        assert res["ram_mb"] > prev_ram
        prev_ram = res["ram_mb"]
        final = _wait_state(api, retry_id, {"succeeded", "failed", "killed_ram"}, timeout=120)
        if final["state"] != "killed_ram":
            break
        cur = retry_id
    assert final is not None and final["state"] == "succeeded", final
    assert int((final.get("meta") or {}).get("auto_retry_attempt")) >= 1


def test_estimates_rows_written_for_model_less_jobs(fleet):
    api = fleet
    code = "print('famkey')"
    jid = _submit(api, "famkey-exp", code, ram_mb=300, config={"code": code})
    _wait_state(api, jid, {"succeeded"}, timeout=60)
    # family history accumulates for cmd-shaped (model-less) work
    import sqlite3, os
    dbp = os.path.join(os.environ["MRUN_SERVER_DATA"], "mrun.db")
    con = sqlite3.connect(dbp)
    row = con.execute(
        "SELECT family_key, reserved_ram_mb FROM estimates WHERE experiment='famkey-exp'"
    ).fetchone()
    con.close()
    assert row is not None
    assert row[0] and row[0].startswith("cmd:famkey-exp:")
    assert row[1] == 300.0


def test_launch_intent_first_end_to_end(fleet):
    """launch(): no pin, no backend knobs — submit, stream, return the result.

    Uses a small declared ask: the 4096MB probe default's kill ceiling can exceed a
    loaded dev box's live headroom and this test must not depend on machine load
    (probe sizing itself is unit-covered in test_reservation/test_scheduler).
    """
    api = fleet
    from mrun.client.submit import launch

    out = launch(
        [sys.executable, "-c", "print('intent-first-ok')"],
        experiment="intent-first",
        ram_mb=300,
        env_alias=None,
        api=api,
    )
    assert out.ok, out.job["state"]
    job = api.json("GET", f"/api/jobs/{out.job_id}")
    assert job["reservation"]["source"] == "declared"
    assert job["needs"].get("host") is None
