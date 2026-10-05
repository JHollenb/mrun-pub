from __future__ import annotations

import hashlib
import json
import threading
import time

from mrun.server.db import DB


def _queued_job() -> dict:
    return {
        "job_id": "job-atomic-claim",
        "client_run_id": "atomic-claim",
        "experiment": "lease-race",
        "state": "queued",
        "needs": {},
        "reservation": {"ram_mb": 128},
        "cmd": ["true"],
        "config": {},
    }


def test_two_same_host_agents_can_claim_queued_job_once(tmp_path):
    db = DB(tmp_path / "mrun.db")
    db.insert_job(_queued_job())
    barrier = threading.Barrier(2)
    results: list[bool] = []

    def claim() -> None:
        barrier.wait()
        results.append(
            db.claim_job(
                "job-atomic-claim",
                host="mbp1",
                lease_expires_ts=1234.0,
                reservation={"ram_mb": 128},
                plan={"backend": "hf"},
            )
        )

    agents = [threading.Thread(target=claim) for _ in range(2)]
    for agent in agents:
        agent.start()
    for agent in agents:
        agent.join(timeout=5)

    assert sorted(results) == [False, True]
    claimed = db.job("job-atomic-claim")
    assert claimed is not None
    assert claimed["state"] == "assigned"
    assert claimed["assigned_host"] == "mbp1"
    assert claimed["plan"] == {"backend": "hf"}


def test_distributed_admission_claim_is_atomic_owner_scoped_and_expiring(tmp_path):
    db = DB(tmp_path / "mrun.db")
    scope = {
        "experiment": "atlas-suite",
        "config_selector": {"suite_fingerprint": "exact-fingerprint"},
    }
    scope_sha256 = hashlib.sha256(
        json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    barrier = threading.Barrier(2)
    results: list[tuple[str, bool]] = []

    def acquire(owner: str) -> None:
        barrier.wait()
        record = db.acquire_admission_claim(
            "suite:exact-fingerprint",
            owner,
            ttl_s=60.0,
            scope=scope,
            scope_sha256=scope_sha256,
            metadata={"owner": owner},
            now=1000.0,
        )
        results.append((owner, record is not None))

    clients = [threading.Thread(target=acquire, args=(owner,)) for owner in ("a", "b")]
    for client in clients:
        client.start()
    for client in clients:
        client.join(timeout=5)

    assert sorted(acquired for _owner, acquired in results) == [False, True]
    winner = next(owner for owner, acquired in results if acquired)
    loser = next(owner for owner, acquired in results if not acquired)
    live = db.admission_claim("suite:exact-fingerprint", now=1001.0)
    assert live is not None
    assert live["owner_token"] == winner
    assert live["metadata"] == {"owner": winner}

    renewed = db.acquire_admission_claim(
        "suite:exact-fingerprint",
        winner,
        ttl_s=120.0,
        scope=scope,
        scope_sha256=scope_sha256,
        metadata={"renewed": True},
        now=1002.0,
    )
    assert renewed is not None
    assert renewed["acquired_ts"] == 1000.0
    assert renewed["expires_ts"] == 1122.0
    assert (
        db.acquire_admission_claim(
            "suite:exact-fingerprint",
            loser,
            ttl_s=60.0,
            scope=scope,
            scope_sha256=scope_sha256,
            now=1003.0,
        )
        is None
    )
    assert not db.release_admission_claim("suite:exact-fingerprint", loser, 1, now=1004.0)
    assert db.release_admission_claim(
        "suite:exact-fingerprint", winner, 1, now=1004.0
    )

    after_release = db.acquire_admission_claim(
        "suite:exact-fingerprint",
        loser,
        ttl_s=10.0,
        scope=scope,
        scope_sha256=scope_sha256,
        now=2000.0,
    )
    assert after_release is not None
    assert after_release["fencing_epoch"] == 2
    assert db.admission_claim("suite:exact-fingerprint", now=2010.0) is None


def test_history_stats_include_vram_p95(tmp_path):
    db = DB(tmp_path / "mrun.db")
    for idx, (ram, vram, wall) in enumerate(
        [(100.0, 10.0, 1.0), (200.0, 20.0, 2.0), (300.0, 30.0, 3.0)]
    ):
        db.add_estimate(
            {
                "client_run_id": f"hist-{idx}",
                "experiment": "history",
                "model": "qwen-moe",
                "host": "beast",
                "status": "succeeded",
                "ram_peak_mb": ram,
                "vram_peak_mb": vram,
                "wall_s": wall,
                "task_family": "recorder",
                "family_key": "model:qwen-moe:recorder",
            }
        )

    stats = db.history_stats("model:qwen-moe:recorder")

    assert stats is not None
    assert stats["n"] == 3
    assert stats["p95_ram_mb"] == 300.0
    assert stats["p95_vram_mb"] == 30.0
    assert stats["p50_wall_s"] == 2.0


def test_events_roundtrip_and_filter(tmp_path):
    db = DB(tmp_path / "mrun.db")
    db.add_event(
        "job.submit",
        job_id="job-a",
        host="beast",
        state="queued",
        payload={"reservation": {"ram_mb": 128}},
    )
    db.add_event("host.telemetry", host="beast", payload={"ram_free_mb": 1000.0})

    job_events = db.events(job_id="job-a")
    host_events = db.events(host="beast")

    assert len(job_events) == 1
    assert job_events[0]["kind"] == "job.submit"
    assert job_events[0]["payload"]["reservation"]["ram_mb"] == 128
    assert {e["kind"] for e in host_events} == {"job.submit", "host.telemetry"}


def test_settings_roundtrip(tmp_path):
    db = DB(tmp_path / "mrun.db")

    assert db.get_setting("control", {"draining": False}) == {"draining": False}
    db.set_setting("control", {"draining": True, "reason": "test"})
    assert db.get_setting("control") == {"draining": True, "reason": "test"}
    db.set_setting("control", {"draining": False})
    assert db.get_setting("control") == {"draining": False}


def test_status_detail_compare_set(tmp_path):
    db = DB(tmp_path / "mrun.db")
    db.insert_job(_queued_job())

    assert db.set_status_detail_if_changed("job-atomic-claim", "waiting: ram")
    assert not db.set_status_detail_if_changed("job-atomic-claim", "waiting: ram")
    assert db.job("job-atomic-claim")["status_detail"] == "waiting: ram"  # type: ignore[index]
    assert db.set_status_detail_if_changed(
        "job-atomic-claim", None, clear_prefixes=("waiting:", "unschedulable:")
    )
    assert db.job("job-atomic-claim")["status_detail"] is None  # type: ignore[index]


def test_expire_leases_logs_lost_event(tmp_path):
    db = DB(tmp_path / "mrun.db")
    db.insert_job(_queued_job())
    assert db.claim_job(
        "job-atomic-claim",
        host="beast",
        lease_expires_ts=time.time() - 1.0,
        reservation={"ram_mb": 128},
        plan={"backend": "hf"},
    )

    expired = db.expire_leases()
    events = db.events(job_id="job-atomic-claim")

    assert expired == ["job-atomic-claim"]
    assert db.job("job-atomic-claim")["state"] == "lost"  # type: ignore[index]
    assert events[0]["kind"] == "job.lost"
    assert events[0]["reason"] == "lease-expired"
