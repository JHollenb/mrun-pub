from __future__ import annotations

import hashlib
import json
import threading

import pytest

from mrun.server.db import (
    DB,
    GuardedAdmissionError,
    ProtectedScopeError,
)

CLAIM_KEY = "atlas:sana:exact-design"
SCOPE = {
    "experiment": "atlas-sana-controls",
    "config_selector": {"suite_fingerprint": "exact-design"},
}
SCOPE_SHA256 = hashlib.sha256(
    json.dumps(SCOPE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _acquire(db: DB, owner: str, *, now: float, ttl_s: float = 60.0) -> dict:
    claim = db.acquire_admission_claim(
        CLAIM_KEY,
        owner,
        ttl_s=ttl_s,
        scope=SCOPE,
        scope_sha256=SCOPE_SHA256,
        metadata={"owner": owner},
        now=now,
    )
    assert claim is not None
    return claim


def _request(*, body: bytes | None = None, marker: str = "canonical") -> tuple[str, str]:
    request = {
        "schema": "mrun.normalized-job-request.v2",
        "experiment": SCOPE["experiment"],
        "client_run_id": f"atlas-cell:{marker}",
        "cmd": ["python", "worker.py"],
        "needs": {"host": "beast", "payload_custody_v2": True},
        "reservation": {"ram_mb": 1024.0, "vram_mb": 512.0},
        "client_estimate": None,
        "payload_kind": "shipped" if body is not None else "cmd",
        "payload": (
            {
                "sha256": hashlib.sha256(body).hexdigest(),
                "size_bytes": len(body),
            }
            if body is not None
            else None
        ),
        "env_alias": "image-analysis",
        "config": {"suite_fingerprint": "exact-design", "marker": marker},
        "timeout_s": 300.0,
        "priority": 5,
        "plans": {"beast": {"backend": "cuda", "dtype": "bf16"}},
    }
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return encoded, hashlib.sha256(encoded.encode()).hexdigest()


def _job(
    job_id: str,
    *,
    body: bytes | None = None,
    marker: str = "canonical",
) -> tuple[dict, str]:
    request_json, request_sha256 = _request(body=body, marker=marker)
    return (
        {
            "job_id": job_id,
            "client_run_id": f"atlas-cell:{marker}",
            "experiment": SCOPE["experiment"],
            "state": "awaiting_payload" if body is not None else "queued",
            "needs": {"host": "beast", "payload_custody_v2": body is not None},
            "reservation": {
                "ram_mb": 1024.0,
                "vram_mb": 512.0,
                "cpu_threads": 1,
                "disk_gb": 1.0,
                "source": "declared",
            },
            "payload_kind": "shipped" if body is not None else "cmd",
            "env_alias": "image-analysis",
            "cmd": ["python", "worker.py"],
            "config": {"suite_fingerprint": "exact-design", "marker": marker},
            "timeout_s": 300.0,
            "priority": 5,
            "plans": {"beast": {"backend": "cuda", "dtype": "bf16"}},
            "request_json": request_json,
            "request_sha256": request_sha256,
            "custody_required": body is not None,
            "payload_declared_sha256": (
                hashlib.sha256(body).hexdigest() if body is not None else None
            ),
            "payload_declared_size": len(body) if body is not None else None,
        },
        request_sha256,
    )


def test_fence_idempotency_and_delayed_stale_post_converge(tmp_path):
    db = DB(tmp_path / "mrun.db")
    first = _acquire(db, "owner-a", now=100.0)
    job, request_sha256 = _job("job-first")
    inserted, created = db.admit_guarded_job(
        job,
        claim_key=CLAIM_KEY,
        owner_token="owner-a",
        fencing_epoch=first["fencing_epoch"],
        idempotency_key="cell-0",
        request_sha256=request_sha256,
        now=101.0,
    )
    assert created is True

    assert db.release_admission_claim(CLAIM_KEY, "owner-a", first["fencing_epoch"], now=102.0)
    second = _acquire(db, "owner-b", now=103.0)
    assert second["fencing_epoch"] == first["fencing_epoch"] + 1

    # A delayed epoch-1 POST arriving after takeover returns the already committed job;
    # it cannot insert another row and does not need a still-live old lease to reconcile.
    delayed, delayed_created = db.admit_guarded_job(
        _job("job-delayed")[0],
        claim_key=CLAIM_KEY,
        owner_token="owner-a",
        fencing_epoch=first["fencing_epoch"],
        idempotency_key="cell-0",
        request_sha256=request_sha256,
        now=104.0,
    )
    assert delayed_created is False
    assert delayed["job_id"] == inserted["job_id"] == "job-first"

    changed, changed_sha256 = _job("job-changed", marker="changed")
    with pytest.raises(GuardedAdmissionError, match="different normalized request"):
        db.admit_guarded_job(
            changed,
            claim_key=CLAIM_KEY,
            owner_token="owner-b",
            fencing_epoch=second["fencing_epoch"],
            idempotency_key="cell-0",
            request_sha256=changed_sha256,
            now=104.0,
        )

    with pytest.raises(GuardedAdmissionError, match="expired|stale|another client"):
        stale, stale_sha256 = _job("job-stale", marker="stale")
        db.admit_guarded_job(
            stale,
            claim_key=CLAIM_KEY,
            owner_token="owner-a",
            fencing_epoch=first["fencing_epoch"],
            idempotency_key="cell-stale",
            request_sha256=stale_sha256,
            now=104.0,
        )


def test_two_clients_same_idempotency_key_create_one_canonical_job(tmp_path):
    db_path = tmp_path / "mrun.db"
    db = DB(db_path)
    claim = _acquire(db, "owner-a", now=100.0)
    clients_db = [db, DB(db_path)]  # separate connections emulate two server workers
    barrier = threading.Barrier(2)
    results: list[tuple[str, bool]] = []

    def submit(index: int) -> None:
        job, request_sha256 = _job(f"job-race-{index}")
        barrier.wait()
        admitted, created = clients_db[index].admit_guarded_job(
            job,
            claim_key=CLAIM_KEY,
            owner_token="owner-a",
            fencing_epoch=claim["fencing_epoch"],
            idempotency_key="cell-race",
            request_sha256=request_sha256,
            now=101.0,
        )
        results.append((admitted["job_id"], created))

    clients = [threading.Thread(target=submit, args=(index,)) for index in range(2)]
    for client in clients:
        client.start()
    for client in clients:
        client.join(timeout=5)

    assert sorted(created for _job_id, created in results) == [False, True]
    assert len({job_id for job_id, _created in results}) == 1
    assert len(db.jobs()) == 1


def test_persistent_scope_rejects_legacy_bypass(tmp_path):
    db = DB(tmp_path / "mrun.db")
    claim = _acquire(db, "owner-a", now=100.0, ttl_s=1.0)
    assert claim["scope_sha256"] == SCOPE_SHA256
    ordinary, _request_sha256 = _job("job-bypass")
    ordinary["admission_claim_key"] = None
    with pytest.raises(ProtectedScopeError, match="protected"):
        db.insert_job(ordinary)

    # Protection outlives expiry; an old client cannot wait out the lease.
    assert db.admission_claim(CLAIM_KEY, now=102.0) is None
    with pytest.raises(ProtectedScopeError, match="protected"):
        db.insert_job({**ordinary, "job_id": "job-bypass-after-expiry"})


def test_scope_acquisition_refuses_matching_preexisting_ordinary_job(tmp_path):
    db = DB(tmp_path / "mrun.db")
    ordinary, _request_sha256 = _job("job-preexisting-ordinary")
    db.insert_job(ordinary)

    with pytest.raises(GuardedAdmissionError, match="existing ordinary nonterminal job"):
        _acquire(db, "owner-a", now=100.0)

    # The failed acquisition is atomic: it leaves neither a persistent protected
    # scope nor a live claim behind.
    assert db.protected_scope_for_job(SCOPE["experiment"], ordinary["config"]) is None
    assert db.admission_claim(CLAIM_KEY, now=100.0) is None

    # Historical terminal ordinary rows do not prevent protecting future work.
    db.update_job(ordinary["job_id"], state="cancelled", finished_ts=101.0)
    claim = _acquire(db, "owner-a", now=102.0)
    assert claim["scope_sha256"] == SCOPE_SHA256


def test_payload_seal_is_create_once_fenced_and_executed_digest_is_host_bound(tmp_path):
    db = DB(tmp_path / "mrun.db")
    body = b"exact packed execution bytes"
    claim = _acquire(db, "owner-a", now=100.0)
    job, request_sha256 = _job("job-payload", body=body)
    admitted, created = db.admit_guarded_job(
        job,
        claim_key=CLAIM_KEY,
        owner_token="owner-a",
        fencing_epoch=claim["fencing_epoch"],
        idempotency_key="payload-cell",
        request_sha256=request_sha256,
        now=101.0,
    )
    assert created is True
    assert admitted["state"] == "awaiting_payload"
    payload_path = tmp_path / "job-payload.tgz"
    digest = hashlib.sha256(body).hexdigest()

    with pytest.raises(GuardedAdmissionError, match="stale"):
        db.seal_job_payload(
            "job-payload",
            owner_token="owner-b",
            fencing_epoch=claim["fencing_epoch"],
            sha256=digest,
            size_bytes=len(body),
            body=body,
            path=payload_path,
            now=102.0,
        )
    barrier = threading.Barrier(2)
    seal_results: list[tuple[dict, bool]] = []

    def seal() -> None:
        barrier.wait()
        seal_results.append(
            db.seal_job_payload(
                "job-payload",
                owner_token="owner-a",
                fencing_epoch=claim["fencing_epoch"],
                sha256=digest,
                size_bytes=len(body),
                body=body,
                path=payload_path,
                now=102.0,
            )
        )

    putters = [threading.Thread(target=seal) for _ in range(2)]
    for putter in putters:
        putter.start()
    for putter in putters:
        putter.join(timeout=5)
    assert sorted(created for _record, created in seal_results) == [False, True]
    sealed = next(record for record, created in seal_results if created)
    assert sealed["state"] == "queued"
    assert sealed["payload_custody"]["declared"] == {
        "sha256": digest,
        "size_bytes": len(body),
    }
    assert sealed["payload_custody"]["sealed"]["sha256"] == digest

    replay, replay_created = db.seal_job_payload(
        "job-payload",
        owner_token="owner-a",
        fencing_epoch=claim["fencing_epoch"],
        sha256=digest,
        size_bytes=len(body),
        body=body,
        path=payload_path,
        now=103.0,
    )
    assert replay_created is False
    assert replay["payload_custody"] == sealed["payload_custody"]
    with pytest.raises(GuardedAdmissionError, match="declared|different"):
        tampered = b"different packed execution bytes"
        db.seal_job_payload(
            "job-payload",
            owner_token="owner-a",
            fencing_epoch=claim["fencing_epoch"],
            sha256=hashlib.sha256(tampered).hexdigest(),
            size_bytes=len(tampered),
            body=tampered,
            path=payload_path,
            now=103.0,
        )
    with pytest.raises(KeyError):
        db.seal_job_payload(
            "job-unknown",
            owner_token="owner-a",
            fencing_epoch=claim["fencing_epoch"],
            sha256=digest,
            size_bytes=len(body),
            body=body,
            path=tmp_path / "unknown.tgz",
            now=103.0,
        )

    assert db.claim_job(
        "job-payload",
        host="beast",
        lease_expires_ts=200.0,
        reservation=sealed["reservation"],
        plan={"backend": "cuda", "dtype": "bf16"},
        lease_identity="lease-payload",
        lease_capability="unguessable-capability",
        now=103.0,
    )
    authorized = db.authorize_guarded_agent_lease(
        "job-payload",
        lease_identity="lease-payload",
        lease_capability="unguessable-capability",
        now=104.0,
    )
    assert authorized["assigned_host"] == "beast"
    assert "lease_capability" not in authorized
    assert "lease_capability_hash" not in authorized
    with pytest.raises(GuardedAdmissionError, match="capability"):
        db.authorize_guarded_agent_lease(
            "job-payload",
            lease_identity="lease-payload",
            lease_capability="wrong-capability",
            now=104.0,
        )
    with pytest.raises(GuardedAdmissionError, match="expired"):
        db.authorize_guarded_agent_lease(
            "job-payload",
            lease_identity="lease-payload",
            lease_capability="unguessable-capability",
            now=200.0,
        )
    with pytest.raises(GuardedAdmissionError, match="assigned host"):
        db.report_executed_payload(
            "job-payload",
            host="other-host",
            sha256=digest,
            size_bytes=len(body),
            now=104.0,
        )
    with pytest.raises(GuardedAdmissionError, match="sealed bytes"):
        db.report_executed_payload(
            "job-payload",
            host="beast",
            sha256="0" * 64,
            size_bytes=len(body),
            now=104.0,
        )
    executed, executed_created = db.report_executed_payload(
        "job-payload",
        host="beast",
        sha256=digest,
        size_bytes=len(body),
        now=104.0,
    )
    assert executed_created is True
    assert executed["payload_custody"]["executed"] == {
        "sha256": digest,
        "size_bytes": len(body),
        "reported_ts": 104.0,
        "host": "beast",
    }

    db.update_job("job-payload", state="succeeded", finished_ts=105.0)
    with pytest.raises(GuardedAdmissionError, match="immutable"):
        db.seal_job_payload(
            "job-payload",
            owner_token="owner-a",
            fencing_epoch=claim["fencing_epoch"],
            sha256=digest,
            size_bytes=len(body),
            body=body,
            path=payload_path,
            now=106.0,
        )


def test_newer_live_owner_takes_over_only_the_same_unsealed_idempotent_job(tmp_path):
    db = DB(tmp_path / "mrun.db")
    body = b"takeover archive"
    digest = hashlib.sha256(body).hexdigest()
    first = _acquire(db, "owner-a", now=100.0)
    job, request_sha256 = _job("job-takeover", body=body)
    canonical, created = db.admit_guarded_job(
        job,
        claim_key=CLAIM_KEY,
        owner_token="owner-a",
        fencing_epoch=first["fencing_epoch"],
        idempotency_key="takeover-cell",
        request_sha256=request_sha256,
        now=101.0,
    )
    assert created is True
    assert db.release_admission_claim(CLAIM_KEY, "owner-a", first["fencing_epoch"], now=102.0)
    second = _acquire(db, "owner-b", now=103.0)

    with pytest.raises(GuardedAdmissionError, match="stale"):
        db.admit_guarded_job(
            _job("job-stale-replay", body=body)[0],
            claim_key=CLAIM_KEY,
            owner_token="owner-a",
            fencing_epoch=first["fencing_epoch"],
            idempotency_key="takeover-cell",
            request_sha256=request_sha256,
            now=104.0,
        )

    taken_over, replay_created = db.admit_guarded_job(
        _job("job-new-owner", body=body)[0],
        claim_key=CLAIM_KEY,
        owner_token="owner-b",
        fencing_epoch=second["fencing_epoch"],
        idempotency_key="takeover-cell",
        request_sha256=request_sha256,
        now=104.0,
    )
    assert replay_created is False
    assert taken_over["job_id"] == canonical["job_id"]
    assert taken_over["admission"]["fencing_epoch"] == second["fencing_epoch"]
    assert len(db.jobs()) == 1

    payload_path = tmp_path / "takeover.tgz"
    with pytest.raises(GuardedAdmissionError, match="stale"):
        db.seal_job_payload(
            canonical["job_id"],
            owner_token="owner-a",
            fencing_epoch=first["fencing_epoch"],
            sha256=digest,
            size_bytes=len(body),
            body=body,
            path=payload_path,
            now=105.0,
        )
    sealed, sealed_created = db.seal_job_payload(
        canonical["job_id"],
        owner_token="owner-b",
        fencing_epoch=second["fencing_epoch"],
        sha256=digest,
        size_bytes=len(body),
        body=body,
        path=payload_path,
        now=105.0,
    )
    assert sealed_created is True
    assert sealed["state"] == "queued"

    # Once sealed/terminal, exact POST replays only reconcile the canonical identity;
    # they cannot transfer authority or create a duplicate.
    db.update_job(canonical["job_id"], state="failed", finished_ts=106.0)
    terminal, terminal_created = db.admit_guarded_job(
        _job("job-terminal-replay", body=body)[0],
        claim_key=CLAIM_KEY,
        owner_token="owner-a",
        fencing_epoch=first["fencing_epoch"],
        idempotency_key="takeover-cell",
        request_sha256=request_sha256,
        now=107.0,
    )
    assert terminal_created is False
    assert terminal["job_id"] == canonical["job_id"]
    assert terminal["state"] == "failed"
    assert len(db.jobs()) == 1


def test_restart_recovery_seals_only_exact_guarded_declared_bytes(tmp_path, monkeypatch):
    data = tmp_path / "server-data"
    data.mkdir()
    monkeypatch.setenv("MRUN_SERVER_DATA", str(data))
    db_path = data / "mrun.db"
    db = DB(db_path)
    claim = _acquire(db, "owner-a", now=100.0)

    exact_body = b"durable exact rename before sqlite commit"
    exact_job, exact_request_sha = _job("job-recover-exact", body=exact_body)
    db.admit_guarded_job(
        exact_job,
        claim_key=CLAIM_KEY,
        owner_token="owner-a",
        fencing_epoch=claim["fencing_epoch"],
        idempotency_key="recover-exact",
        request_sha256=exact_request_sha,
        now=101.0,
    )
    mismatch_declared = b"declared bytes that never reached disk"
    mismatch_job, mismatch_request_sha = _job(
        "job-recover-mismatch", body=mismatch_declared, marker="mismatch"
    )
    db.admit_guarded_job(
        mismatch_job,
        claim_key=CLAIM_KEY,
        owner_token="owner-a",
        fencing_epoch=claim["fencing_epoch"],
        idempotency_key="recover-mismatch",
        request_sha256=mismatch_request_sha,
        now=101.0,
    )
    payloads = data / "payloads"
    payloads.mkdir(exist_ok=True)
    (payloads / "job-recover-exact.tgz").write_bytes(exact_body)
    (payloads / "job-recover-mismatch.tgz").write_bytes(b"wrong durable bytes")

    recovered = DB(db_path)
    exact = recovered.job("job-recover-exact")
    assert exact is not None
    assert exact["state"] == "queued"
    assert exact["payload_custody"]["declared"] == {
        "sha256": hashlib.sha256(exact_body).hexdigest(),
        "size_bytes": len(exact_body),
    }
    assert exact["payload_custody"]["sealed"]["sha256"] == hashlib.sha256(exact_body).hexdigest()

    mismatch = recovered.job("job-recover-mismatch")
    assert mismatch is not None
    assert mismatch["state"] == "awaiting_payload"
    assert mismatch["payload_custody"]["declared"] == {
        "sha256": hashlib.sha256(mismatch_declared).hexdigest(),
        "size_bytes": len(mismatch_declared),
    }
    assert mismatch["payload_custody"]["sealed"] is None
    assert "recovery refused" in mismatch["status_detail"]


def test_migration_seals_existing_unguarded_shipped_job_without_requeue(tmp_path, monkeypatch):
    data = tmp_path / "server-data"
    data.mkdir()
    monkeypatch.setenv("MRUN_SERVER_DATA", str(data))
    db_path = data / "mrun.db"
    db = DB(db_path)
    legacy = {
        "job_id": "job-existing-queued",
        "client_run_id": "existing-queued",
        "experiment": "legacy-shipped",
        "state": "queued",
        "needs": {"host": "beast"},
        "reservation": {"ram_mb": 128},
        "payload_kind": "shipped",
        "cmd": ["python", "worker.py"],
        "config": {},
    }
    db.insert_job(legacy)
    body = b"already queued before custody v2 rollout"
    payload_path = data / "payloads" / "job-existing-queued.tgz"
    payload_path.parent.mkdir()
    payload_path.write_bytes(body)

    migrated = DB(db_path).job("job-existing-queued")
    assert migrated is not None
    assert migrated["state"] == "queued"
    assert migrated["payload_custody"]["required"] is False
    assert migrated["payload_custody"]["sealed"]["sha256"] == hashlib.sha256(body).hexdigest()
    assert migrated["payload_custody"]["sealed"]["size_bytes"] == len(body)
