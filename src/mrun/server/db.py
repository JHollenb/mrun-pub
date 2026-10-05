"""SQLite store for the scheduler — WAL mode, one connection behind a lock.

Write volume is ~1 row / 5 s / host; no ORM needed. Logs and payloads are FILES under
the data dir (log offset == file size), never DB rows.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..protocol import TERMINAL_STATES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hosts (
    name TEXT PRIMARY KEY,
    os TEXT, arch TEXT,
    caps_json TEXT,
    ram_total_mb REAL, vram_total_mb REAL, cpu_threads INTEGER, disk_total_gb REAL,
    agent_version TEXT, protocol_version INTEGER,
    models_mount TEXT,
    last_seen_ts REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS telemetry (
    host TEXT PRIMARY KEY,
    ts REAL,
    cpu_pct REAL, ram_free_mb REAL, vram_free_mb REAL, disk_free_gb REAL, load1 REAL,
    swap_used_mb REAL, swap_total_mb REAL, mem_pressure INTEGER,
    disks_json TEXT,
    running_json TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    client_run_id TEXT,
    experiment TEXT,
    state TEXT,
    needs_json TEXT,
    reservation_json TEXT,
    payload_kind TEXT,
    env_alias TEXT,
    cmd_json TEXT,
    config_json TEXT,
    timeout_s REAL,
    priority INTEGER DEFAULT 0,
    assigned_host TEXT,
    lease_expires_ts REAL,
    kill_requested INTEGER DEFAULT 0,
    created_ts REAL, started_ts REAL, finished_ts REAL,
    result_json TEXT,
    status_detail TEXT,
    external_rss_mb REAL,
    external_vram_mb REAL,
    plans_json TEXT,
    plan_json TEXT,
    meta_json TEXT,
    admission_claim_key TEXT,
    admission_owner_token TEXT,
    admission_fencing_epoch INTEGER,
    idempotency_key TEXT,
    request_json TEXT,
    request_sha256 TEXT,
    scope_sha256 TEXT,
    custody_required INTEGER DEFAULT 0,
    payload_declared_sha256 TEXT,
    payload_declared_size INTEGER,
    payload_sealed_sha256 TEXT,
    payload_sealed_size INTEGER,
    payload_sealed_ts REAL,
    payload_executed_sha256 TEXT,
    payload_executed_size INTEGER,
    payload_executed_ts REAL,
    payload_executed_host TEXT,
    lease_identity TEXT,
    lease_capability_hash TEXT,
    lease_payload_sha256 TEXT,
    lease_payload_size INTEGER,
    lease_issued_ts REAL,
    debug_credential_id TEXT,
    debug_credential_hash TEXT,
    debug_scopes_json TEXT,
    debug_issued_ts REAL
);
CREATE TABLE IF NOT EXISTS host_models (
    host TEXT, model TEXT, kind TEXT,
    bytes INTEGER, path TEXT, ts REAL,
    artifact_id TEXT, artifact_kind TEXT, variant TEXT, mount TEXT,
    manifest_sha256 TEXT, content_hash TEXT, source_model TEXT, locator_json TEXT,
    PRIMARY KEY (host, model, kind)
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS jobs_client_run ON jobs(client_run_id);
CREATE TABLE IF NOT EXISTS estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_run_id TEXT,
    experiment TEXT,
    model TEXT,
    host TEXT,
    status TEXT,
    ram_peak_mb REAL, vram_peak_mb REAL, wall_s REAL,
    backend TEXT, dtype TEXT, task_family TEXT,
    est_ram_mb REAL,
    ts REAL,
    family_key TEXT,
    reserved_ram_mb REAL,
    reserved_vram_mb REAL,
    kill_state TEXT
);
CREATE INDEX IF NOT EXISTS estimates_run ON estimates(client_run_id);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    kind TEXT,
    job_id TEXT,
    host TEXT,
    state TEXT,
    reason TEXT,
    detail TEXT,
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id, ts);
CREATE INDEX IF NOT EXISTS events_host ON events(host, ts);
CREATE INDEX IF NOT EXISTS events_kind ON events(kind, ts);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT,
    updated_ts REAL
);
CREATE TABLE IF NOT EXISTS admission_claims (
    claim_key TEXT PRIMARY KEY,
    owner_token TEXT NOT NULL,
    fencing_epoch INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT,
    acquired_ts REAL NOT NULL,
    expires_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS admission_claims_expiry ON admission_claims(expires_ts);
CREATE TABLE IF NOT EXISTS admission_scopes (
    claim_key TEXT PRIMARY KEY,
    scope_json TEXT NOT NULL,
    scope_sha256 TEXT NOT NULL UNIQUE,
    created_ts REAL NOT NULL
);
"""


class ProtectedScopeError(RuntimeError):
    """An unguarded insertion matched a permanently protected logical scope."""


class GuardedAdmissionError(RuntimeError):
    """A fenced/idempotent guarded admission failed closed."""


def _lease_capability_digest(
    capability: str,
    *,
    job_id: str,
    host: str,
    lease_identity: str,
    payload_sha256: str,
    payload_size: int,
) -> str:
    """One-way binding of an unguessable lease secret to its complete authority."""

    binding = "\0".join(
        (
            "mrun.guarded-lease-capability.v1",
            job_id,
            host,
            lease_identity,
            payload_sha256,
            str(payload_size),
            capability,
        )
    )
    return hashlib.sha256(binding.encode("utf-8")).hexdigest()


def _debug_credential_digest(
    credential: str,
    *,
    credential_id: str,
    job_id: str,
    host: str,
    lease_identity: str,
    payload_sha256: str,
    payload_size: int,
    scopes: tuple[str, ...],
) -> str:
    """Bind a narrow debugger bearer to one exact guarded job lease.

    The raw bearer only appears in the successful lease response.  Its stored
    digest cannot be replayed against another job, host, lease attempt, sealed
    payload, credential identity, or scope set.
    """

    binding = "\0".join(
        (
            "mrun.guarded-job-debug-credential.v1",
            credential_id,
            job_id,
            host,
            lease_identity,
            payload_sha256,
            str(payload_size),
            ",".join(scopes),
            credential,
        )
    )
    return hashlib.sha256(binding.encode("utf-8")).hexdigest()


def data_dir() -> Path:
    d = Path(os.environ.get("MRUN_SERVER_DATA", str(Path.home() / ".mrun" / "server")))
    (d / "logs").mkdir(parents=True, exist_ok=True)
    (d / "payloads").mkdir(parents=True, exist_ok=True)
    return d


class DB:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (data_dir() / "mrun.db")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Idempotent column adds for DBs created before the schema grew (SQLite has no
        ADD COLUMN IF NOT EXISTS)."""
        wanted = {
            "telemetry": [
                ("swap_used_mb", "REAL"),
                ("swap_total_mb", "REAL"),
                ("mem_pressure", "INTEGER"),
                ("disks_json", "TEXT"),
                ("top_external_json", "TEXT"),
                ("pressure_external", "INTEGER"),
            ],
            "jobs": [
                ("external_rss_mb", "REAL"),
                ("external_vram_mb", "REAL"),
                ("plans_json", "TEXT"),
                ("plan_json", "TEXT"),
                ("meta_json", "TEXT"),
                ("admission_claim_key", "TEXT"),
                ("admission_owner_token", "TEXT"),
                ("admission_fencing_epoch", "INTEGER"),
                ("idempotency_key", "TEXT"),
                ("request_json", "TEXT"),
                ("request_sha256", "TEXT"),
                ("scope_sha256", "TEXT"),
                ("custody_required", "INTEGER DEFAULT 0"),
                ("payload_declared_sha256", "TEXT"),
                ("payload_declared_size", "INTEGER"),
                ("payload_sealed_sha256", "TEXT"),
                ("payload_sealed_size", "INTEGER"),
                ("payload_sealed_ts", "REAL"),
                ("payload_executed_sha256", "TEXT"),
                ("payload_executed_size", "INTEGER"),
                ("payload_executed_ts", "REAL"),
                ("payload_executed_host", "TEXT"),
                ("lease_identity", "TEXT"),
                ("lease_capability_hash", "TEXT"),
                ("lease_payload_sha256", "TEXT"),
                ("lease_payload_size", "INTEGER"),
                ("lease_issued_ts", "REAL"),
                ("debug_credential_id", "TEXT"),
                ("debug_credential_hash", "TEXT"),
                ("debug_scopes_json", "TEXT"),
                ("debug_issued_ts", "REAL"),
            ],
            "admission_claims": [("fencing_epoch", "INTEGER NOT NULL DEFAULT 1")],
            "hosts": [("models_mount", "TEXT")],
            "host_models": [
                ("artifact_id", "TEXT"),
                ("artifact_kind", "TEXT"),
                ("variant", "TEXT"),
                ("mount", "TEXT"),
                ("manifest_sha256", "TEXT"),
                ("content_hash", "TEXT"),
                ("source_model", "TEXT"),
                ("locator_json", "TEXT"),
            ],
            "estimates": [
                ("backend", "TEXT"),
                ("dtype", "TEXT"),
                ("task_family", "TEXT"),
                ("est_ram_mb", "REAL"),
                ("family_key", "TEXT"),
                ("reserved_ram_mb", "REAL"),
                ("reserved_vram_mb", "REAL"),
                ("kill_state", "TEXT"),
            ],
        }
        for table, cols in wanted.items():
            have = {
                r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")
            }
            for name, typ in cols:
                if name not in have:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")
        # indexes on migrated columns must come AFTER the ALTERs (a pre-migration DB
        # lacks the column, and executescript(_SCHEMA) would die before _migrate ran)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS estimates_model ON estimates(model, task_family)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS estimates_family ON estimates(family_key, status, ts)"
        )
        # One-time in-place backfill: model-keyed rows written before family_key existed
        # join their family so generalized history engages immediately after upgrade.
        self._conn.execute(
            """UPDATE estimates
               SET family_key = 'model:' || model || ':' || COALESCE(task_family, 'forward')
               WHERE family_key IS NULL AND model IS NOT NULL"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS events_ts ON events(ts)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS events_job ON events(job_id, ts)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS events_host ON events(host, ts)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS events_kind ON events(kind, ts)")
        self._conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS jobs_guarded_idempotency
               ON jobs(admission_claim_key, idempotency_key)
               WHERE admission_claim_key IS NOT NULL AND idempotency_key IS NOT NULL"""
        )
        # Reconcile an archive rename that became durable before the DB transaction
        # committed. Guarded declarations are immutable: exact bytes converge to a
        # seal and QUEUED; a mismatch stays unsealed/unschedulable and requires operator
        # intervention. Ordinary pre-v2 shipped jobs retain rolling compatibility by
        # adopting their already-present archive as a non-custodial seal.
        legacy_shipped = self._conn.execute(
            """SELECT job_id, state, custody_required,
                      payload_declared_sha256, payload_declared_size
               FROM jobs
               WHERE payload_kind='shipped' AND payload_sealed_sha256 IS NULL"""
        ).fetchall()
        for row in legacy_shipped:
            path = data_dir() / "payloads" / f"{row['job_id']}.tgz"
            if not path.is_file():
                continue
            body = path.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            # This timestamps the recovery/seal decision, not the original submit.
            recovered_ts = time.time()
            if bool(row["custody_required"]):
                declared_sha = row["payload_declared_sha256"]
                declared_size = row["payload_declared_size"]
                exact = (
                    declared_sha is not None
                    and declared_size is not None
                    and declared_sha == digest
                    and int(declared_size) == len(body)
                )
                if exact and row["state"] in ("awaiting_payload", "queued"):
                    self._conn.execute(
                        """UPDATE jobs SET payload_sealed_sha256=?,
                               payload_sealed_size=?, payload_sealed_ts=?,
                               state='queued', status_detail=NULL
                           WHERE job_id=? AND payload_sealed_sha256 IS NULL
                             AND payload_declared_sha256=?
                             AND payload_declared_size=?""",
                        (
                            digest,
                            len(body),
                            recovered_ts,
                            row["job_id"],
                            digest,
                            len(body),
                        ),
                    )
                else:
                    self._conn.execute(
                        """UPDATE jobs SET status_detail=?
                           WHERE job_id=? AND payload_sealed_sha256 IS NULL""",
                        (
                            "payload recovery refused: on-disk archive does not "
                            "exactly match immutable guarded declaration",
                            row["job_id"],
                        ),
                    )
                continue
            self._conn.execute(
                """UPDATE jobs SET
                       payload_declared_sha256=COALESCE(payload_declared_sha256, ?),
                       payload_declared_size=COALESCE(payload_declared_size, ?),
                       payload_sealed_sha256=?, payload_sealed_size=?, payload_sealed_ts=?
                   WHERE job_id=? AND payload_sealed_sha256 IS NULL
                     AND custody_required=0""",
                (
                    digest,
                    len(body),
                    digest,
                    len(body),
                    recovered_ts,
                    row["job_id"],
                ),
            )

    @contextmanager
    def tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    @contextmanager
    def immediate_tx(self):
        """Cross-connection/process write serialization for custody invariants.

        The regular store uses one connection behind a process-local lock. Guarded
        admission additionally needs the first read fenced against another uvicorn
        worker or process opening the same SQLite file, so acquire the SQLite RESERVED
        lock before reading idempotency/claim/payload state.
        """

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ---- hosts / telemetry -------------------------------------------------
    def upsert_host(self, name: str, payload: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO hosts(name, os, arch, caps_json, ram_total_mb, vram_total_mb,
                       cpu_threads, disk_total_gb, agent_version, protocol_version,
                       models_mount, last_seen_ts)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(name) DO UPDATE SET os=excluded.os, arch=excluded.arch,
                       caps_json=excluded.caps_json, ram_total_mb=excluded.ram_total_mb,
                       vram_total_mb=excluded.vram_total_mb, cpu_threads=excluded.cpu_threads,
                       disk_total_gb=excluded.disk_total_gb, agent_version=excluded.agent_version,
                       protocol_version=excluded.protocol_version,
                       models_mount=excluded.models_mount,
                       last_seen_ts=excluded.last_seen_ts""",
                (
                    name,
                    payload.get("os"),
                    payload.get("arch"),
                    json.dumps(payload.get("caps") or {}),
                    payload.get("ram_total_mb"),
                    payload.get("vram_total_mb", 0.0),
                    payload.get("cpu_threads"),
                    payload.get("disk_total_gb"),
                    payload.get("agent_version"),
                    payload.get("protocol_version"),
                    payload.get("models_mount"),
                    time.time(),
                ),
            )

    def record_telemetry(self, host: str, payload: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO telemetry(host, ts, cpu_pct, ram_free_mb, vram_free_mb,
                       disk_free_gb, load1, swap_used_mb, swap_total_mb, mem_pressure,
                       disks_json, running_json, top_external_json, pressure_external)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(host) DO UPDATE SET ts=excluded.ts, cpu_pct=excluded.cpu_pct,
                       ram_free_mb=excluded.ram_free_mb, vram_free_mb=excluded.vram_free_mb,
                       disk_free_gb=excluded.disk_free_gb, load1=excluded.load1,
                       swap_used_mb=excluded.swap_used_mb,
                       swap_total_mb=excluded.swap_total_mb,
                       mem_pressure=excluded.mem_pressure,
                       disks_json=excluded.disks_json,
                       running_json=excluded.running_json,
                       top_external_json=excluded.top_external_json,
                       pressure_external=excluded.pressure_external""",
                (
                    host,
                    time.time(),
                    payload.get("cpu_pct"),
                    payload.get("ram_free_mb"),
                    payload.get("vram_free_mb"),
                    payload.get("disk_free_gb"),
                    payload.get("load1"),
                    payload.get("swap_used_mb"),
                    payload.get("swap_total_mb"),
                    payload.get("mem_pressure"),
                    json.dumps(payload.get("disks") or []),
                    json.dumps(payload.get("running") or []),
                    json.dumps(payload.get("top_external") or []),
                    1 if payload.get("pressure_external") else 0,
                ),
            )
            c.execute("UPDATE hosts SET last_seen_ts=? WHERE name=?", (time.time(), host))

    def host_rows(self) -> list[dict[str, Any]]:
        with self.tx() as c:
            hosts = [dict(r) for r in c.execute("SELECT * FROM hosts")]
            tel = {r["host"]: dict(r) for r in c.execute("SELECT * FROM telemetry")}
            limit_rows = {
                r["key"]: r["value_json"]
                for r in c.execute(
                    "SELECT key, value_json FROM settings WHERE key LIKE 'host_limits:%'"
                )
            }
        for h in hosts:
            h["caps"] = json.loads(h.pop("caps_json") or "{}")
            t = tel.get(h["name"])
            if t:
                t["running"] = json.loads(t.pop("running_json") or "[]")
                t["disks"] = json.loads(t.pop("disks_json", None) or "[]")
                t["top_external"] = json.loads(t.pop("top_external_json", None) or "[]")
                t["pressure_external"] = bool(t.get("pressure_external"))
            h["telemetry"] = t
            try:
                h["limits"] = json.loads(
                    limit_rows.get(f"host_limits:{h['name']}") or "{}"
                ) or {}
            except json.JSONDecodeError:
                h["limits"] = {}
        return hosts

    def delete_host(self, name: str) -> None:
        """Forget a host entirely (registration, telemetry, warm-model inventory).
        For stale duplicate rows (a renamed agent re-registers under its new name and
        the old row lingers forever). Jobs keep their assigned_host string — history
        stays intact."""
        with self.tx() as c:
            c.execute("DELETE FROM hosts WHERE name=?", (name,))
            c.execute("DELETE FROM telemetry WHERE host=?", (name,))
            c.execute("DELETE FROM host_models WHERE host=?", (name,))

    # ---- events -------------------------------------------------------------
    def add_event(
        self,
        kind: str,
        *,
        job_id: str | None = None,
        host: str | None = None,
        state: str | None = None,
        reason: str | None = None,
        detail: str | None = None,
        payload: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> None:
        """Append a durable scheduler event.

        This is the audit/heuristics stream: transitions, admission decisions,
        telemetry/resource snapshots, inventory and metadata changes. Log bodies stay in
        flat files; the DB only stores compact structured context.
        """
        with self.tx() as c:
            c.execute(
                """INSERT INTO events(ts, kind, job_id, host, state, reason, detail,
                       payload_json)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    ts if ts is not None else time.time(),
                    kind,
                    job_id,
                    host,
                    state,
                    reason,
                    detail,
                    json.dumps(payload, default=str) if payload is not None else None,
                ),
            )

    def events(
        self,
        *,
        job_id: str | None = None,
        host: str | None = None,
        kind: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM events WHERE 1=1", []
        if job_id:
            q += " AND job_id=?"
            args.append(job_id)
        if host:
            q += " AND host=?"
            args.append(host)
        if kind:
            q += " AND kind=?"
            args.append(kind)
        q += " ORDER BY ts DESC, id DESC LIMIT ?"
        args.append(max(1, min(int(limit), 10_000)))
        with self.tx() as c:
            rows = c.execute(q, args).fetchall()
        return [_event_dict(r) for r in rows]

    # ---- settings -----------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.tx() as c:
            row = c.execute("SELECT value_json FROM settings WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value_json"] or "null")
        except json.JSONDecodeError:
            return default

    def set_setting(self, key: str, value: Any) -> None:
        now = time.time()
        with self.tx() as c:
            c.execute(
                """INSERT INTO settings(key, value_json, updated_ts)
                   VALUES(?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                       value_json=excluded.value_json,
                       updated_ts=excluded.updated_ts""",
                (key, json.dumps(value, default=str), now),
            )

    # ---- distributed admission claims -------------------------------------
    def acquire_admission_claim(
        self,
        claim_key: str,
        owner_token: str,
        *,
        ttl_s: float,
        scope: dict[str, Any],
        scope_sha256: str,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Atomically register a protected scope and acquire/renew its fenced lease.

        Claim rows are never deleted: preserving the last fencing epoch prevents ABA
        after release/expiry.  A live renewal by the same owner keeps its epoch; every
        takeover increments it.  The logical scope is immutable and globally unique.
        """

        current = time.time() if now is None else float(now)
        expires_ts = current + float(ttl_s)
        scope_json = _canonical_json(scope)
        with self.immediate_tx() as c:
            # Scope creation and ordinary job insertion use the same immediate
            # transaction boundary.  Whichever wins first forces the other to fail:
            # a pre-existing ordinary job cannot become a protected-scope execution
            # path retroactively, and a registered scope blocks later ordinary jobs.
            selector = scope.get("config_selector") or {}
            ordinary_rows = c.execute(
                """SELECT job_id, state, config_json FROM jobs
                   WHERE experiment=? AND admission_claim_key IS NULL""",
                (scope.get("experiment"),),
            ).fetchall()
            for ordinary in ordinary_rows:
                if ordinary["state"] in TERMINAL_STATES:
                    continue
                config = json.loads(ordinary["config_json"] or "{}")
                if all(
                    key in config and config[key] == value
                    for key, value in selector.items()
                ):
                    raise GuardedAdmissionError(
                        "protected scope matches existing ordinary nonterminal job "
                        f"{ordinary['job_id']!r} in state {ordinary['state']!r}"
                    )
            registered = c.execute(
                "SELECT * FROM admission_scopes WHERE claim_key=?", (claim_key,)
            ).fetchone()
            duplicate_scope = c.execute(
                "SELECT claim_key FROM admission_scopes WHERE scope_sha256=?",
                (scope_sha256,),
            ).fetchone()
            if registered is not None:
                if (
                    registered["scope_sha256"] != scope_sha256
                    or registered["scope_json"] != scope_json
                ):
                    raise GuardedAdmissionError(
                        "claim_key is permanently bound to a different protected scope"
                    )
            elif duplicate_scope is not None:
                raise GuardedAdmissionError(
                    "protected scope is already bound to another claim_key"
                )
            else:
                for prior in c.execute("SELECT * FROM admission_scopes").fetchall():
                    prior_scope = json.loads(prior["scope_json"])
                    if prior_scope.get("experiment") != scope.get("experiment"):
                        continue
                    prior_selector = prior_scope.get("config_selector") or {}
                    selector = scope.get("config_selector") or {}
                    common = set(prior_selector) & set(selector)
                    # No conflicting shared predicate means at least one config could
                    # match both selectors, making guarded ownership ambiguous.
                    if all(prior_selector[key] == selector[key] for key in common):
                        raise GuardedAdmissionError(
                            "protected scope overlaps an existing claim scope"
                        )
                c.execute(
                    """INSERT INTO admission_scopes(
                           claim_key, scope_json, scope_sha256, created_ts
                       ) VALUES(?,?,?,?)""",
                    (claim_key, scope_json, scope_sha256, current),
                )
            row = c.execute(
                "SELECT * FROM admission_claims WHERE claim_key=?", (claim_key,)
            ).fetchone()
            if (
                row is not None
                and float(row["expires_ts"]) > current
                and row["owner_token"] != owner_token
            ):
                return None
            if row is None:
                c.execute(
                    """INSERT INTO admission_claims(
                           claim_key, owner_token, fencing_epoch, metadata_json,
                           acquired_ts, expires_ts
                       ) VALUES(?,?,?,?,?,?)""",
                    (
                        claim_key,
                        owner_token,
                        1,
                        json.dumps(metadata or {}, sort_keys=True),
                        current,
                        expires_ts,
                    ),
                )
            elif float(row["expires_ts"]) > current:
                # Same live owner renewal: retaining the epoch lets in-flight uploads
                # complete without creating an artificial stale-fence transition.
                c.execute(
                    """UPDATE admission_claims
                       SET metadata_json=?, expires_ts=?
                       WHERE claim_key=? AND owner_token=? AND fencing_epoch=?""",
                    (
                        json.dumps(metadata or {}, sort_keys=True),
                        expires_ts,
                        claim_key,
                        owner_token,
                        int(row["fencing_epoch"]),
                    ),
                )
            else:
                # Expiry or explicit release is a takeover even when a caller reuses an
                # owner token.  The epoch, not token uniqueness, is the ABA defense.
                c.execute(
                    """UPDATE admission_claims
                       SET owner_token=?, fencing_epoch=?, metadata_json=?,
                           acquired_ts=?, expires_ts=?
                       WHERE claim_key=?""",
                    (
                        owner_token,
                        int(row["fencing_epoch"]) + 1,
                        json.dumps(metadata or {}, sort_keys=True),
                        current,
                        expires_ts,
                        claim_key,
                    ),
                )
            claimed = c.execute(
                "SELECT * FROM admission_claims WHERE claim_key=?", (claim_key,)
            ).fetchone()
            scope_row = c.execute(
                "SELECT * FROM admission_scopes WHERE claim_key=?", (claim_key,)
            ).fetchone()
        return (
            _admission_claim_dict(claimed, scope_row=scope_row)
            if claimed is not None
            else None
        )

    def release_admission_claim(
        self,
        claim_key: str,
        owner_token: str,
        fencing_epoch: int,
        *,
        now: float | None = None,
    ) -> bool:
        """Expire only the exact live owner+epoch while preserving its fence row."""

        current = time.time() if now is None else float(now)
        with self.immediate_tx() as c:
            cursor = c.execute(
                """UPDATE admission_claims SET expires_ts=?
                   WHERE claim_key=? AND owner_token=? AND fencing_epoch=?
                     AND expires_ts>?""",
                (current, claim_key, owner_token, fencing_epoch, current),
            )
        return cursor.rowcount == 1

    def admission_claim(
        self, claim_key: str, *, now: float | None = None
    ) -> dict[str, Any] | None:
        """Return a live claim for diagnostics/tests without destroying fence history."""

        current = time.time() if now is None else float(now)
        with self.tx() as c:
            row = c.execute(
                "SELECT * FROM admission_claims WHERE claim_key=? AND expires_ts>?",
                (claim_key, current),
            ).fetchone()
            scope_row = c.execute(
                "SELECT * FROM admission_scopes WHERE claim_key=?", (claim_key,)
            ).fetchone()
        return (
            _admission_claim_dict(row, scope_row=scope_row) if row is not None else None
        )

    def protected_scope_for_job(
        self, experiment: str, config: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return the persistent protected scope matching this logical job, if any."""

        with self.tx() as c:
            row = self._protected_scope_for_job(c, experiment, config)
        return _admission_scope_dict(row) if row is not None else None

    # ---- jobs ---------------------------------------------------------------
    def insert_job(self, job: dict[str, Any]) -> None:
        """Insert an ordinary job, atomically refusing every protected scope."""

        with self.immediate_tx() as c:
            scope = self._protected_scope_for_job(
                c, str(job["experiment"]), job.get("config") or {}
            )
            if scope is not None:
                raise ProtectedScopeError(
                    f"scope is protected by admission claim {scope['claim_key']!r}"
                )
            self._insert_job_row(c, job)

    @staticmethod
    def _protected_scope_for_job(
        c: sqlite3.Connection, experiment: str, config: dict[str, Any]
    ) -> sqlite3.Row | None:
        rows = c.execute("SELECT * FROM admission_scopes").fetchall()
        for row in rows:
            scope = json.loads(row["scope_json"])
            if scope.get("experiment") != experiment:
                continue
            selector = scope.get("config_selector") or {}
            if all(key in config and config[key] == value for key, value in selector.items()):
                return row
        return None

    @staticmethod
    def _insert_job_row(c: sqlite3.Connection, job: dict[str, Any]) -> None:
        c.execute(
            """INSERT INTO jobs(
                   job_id, client_run_id, experiment, state, needs_json,
                   reservation_json, payload_kind, env_alias, cmd_json, config_json,
                   timeout_s, priority, plans_json, created_ts,
                   admission_claim_key, admission_owner_token,
                   admission_fencing_epoch, idempotency_key, request_json,
                   request_sha256, scope_sha256, custody_required,
                   payload_declared_sha256, payload_declared_size,
                   payload_sealed_sha256, payload_sealed_size, payload_sealed_ts,
                   payload_executed_sha256, payload_executed_size,
                   payload_executed_ts, payload_executed_host,
                   lease_identity, lease_capability_hash, lease_payload_sha256,
                   lease_payload_size, lease_issued_ts, meta_json
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                job["job_id"],
                job["client_run_id"],
                job["experiment"],
                job["state"],
                json.dumps(job.get("needs") or {}),
                json.dumps(job.get("reservation") or {}),
                job.get("payload_kind", "cmd"),
                job.get("env_alias"),
                json.dumps(job.get("cmd") or []),
                json.dumps(job.get("config") or {}),
                job.get("timeout_s"),
                job.get("priority", 0),
                json.dumps(job.get("plans")) if job.get("plans") is not None else None,
                job.get("created_ts", time.time()),
                job.get("admission_claim_key"),
                job.get("admission_owner_token"),
                job.get("admission_fencing_epoch"),
                job.get("idempotency_key"),
                job.get("request_json"),
                job.get("request_sha256"),
                job.get("scope_sha256"),
                int(bool(job.get("custody_required"))),
                job.get("payload_declared_sha256"),
                job.get("payload_declared_size"),
                job.get("payload_sealed_sha256"),
                job.get("payload_sealed_size"),
                job.get("payload_sealed_ts"),
                job.get("payload_executed_sha256"),
                job.get("payload_executed_size"),
                job.get("payload_executed_ts"),
                job.get("payload_executed_host"),
                job.get("lease_identity"),
                job.get("lease_capability_hash"),
                job.get("lease_payload_sha256"),
                job.get("lease_payload_size"),
                job.get("lease_issued_ts"),
                json.dumps(job.get("meta") or {}),
            ),
        )

    def admit_guarded_job(
        self,
        job: dict[str, Any],
        *,
        claim_key: str,
        owner_token: str,
        fencing_epoch: int,
        idempotency_key: str,
        request_sha256: str,
        now: float | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Fence validation, idempotency reservation, and insertion in one transaction.

        Existing exact requests normally win before lease validation. An unsealed
        guarded upload is the exception: only the current live claim may replay it, and
        a strictly newer epoch atomically takes over sealing authority. Once bytes are
        sealed (or a job is terminal), exact stale replays may still converge on the
        canonical identity but can no longer mutate it.
        """

        current = time.time() if now is None else float(now)
        with self.immediate_tx() as c:
            existing = c.execute(
                """SELECT * FROM jobs
                   WHERE admission_claim_key=? AND idempotency_key=?""",
                (claim_key, idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["request_sha256"] != request_sha256:
                    raise GuardedAdmissionError(
                        "idempotency key is already bound to a different normalized request"
                    )
                awaiting_unsealed = (
                    bool(existing["custody_required"])
                    and existing["state"] == "awaiting_payload"
                    and existing["payload_sealed_sha256"] is None
                )
                if awaiting_unsealed:
                    claim = c.execute(
                        "SELECT * FROM admission_claims WHERE claim_key=?",
                        (claim_key,),
                    ).fetchone()
                    if claim is None or float(claim["expires_ts"]) <= current:
                        raise GuardedAdmissionError(
                            "live admission claim required to replay an unsealed job"
                        )
                    if (
                        claim["owner_token"] != owner_token
                        or int(claim["fencing_epoch"]) != fencing_epoch
                    ):
                        raise GuardedAdmissionError(
                            "unsealed job replay owner or fencing epoch is stale"
                        )
                    stored_epoch = int(existing["admission_fencing_epoch"])
                    stored_owner = existing["admission_owner_token"]
                    if fencing_epoch < stored_epoch:
                        raise GuardedAdmissionError(
                            "unsealed job replay fencing epoch is stale"
                        )
                    if fencing_epoch == stored_epoch and owner_token != stored_owner:
                        raise GuardedAdmissionError(
                            "unsealed job replay owner conflicts at the current epoch"
                        )
                    if fencing_epoch > stored_epoch:
                        c.execute(
                            """UPDATE jobs
                               SET admission_owner_token=?, admission_fencing_epoch=?
                               WHERE job_id=? AND state='awaiting_payload'
                                 AND payload_sealed_sha256 IS NULL
                                 AND admission_fencing_epoch<?""",
                            (owner_token, fencing_epoch, existing["job_id"], fencing_epoch),
                        )
                        existing = c.execute(
                            "SELECT * FROM jobs WHERE job_id=?", (existing["job_id"],)
                        ).fetchone()
                        if existing is None:  # pragma: no cover - row held by transaction
                            raise RuntimeError("guarded takeover job disappeared")
                return _job_dict(existing), False

            claim = c.execute(
                "SELECT * FROM admission_claims WHERE claim_key=?", (claim_key,)
            ).fetchone()
            if claim is None:
                raise GuardedAdmissionError("admission claim does not exist")
            if float(claim["expires_ts"]) <= current:
                raise GuardedAdmissionError("admission claim is expired")
            if claim["owner_token"] != owner_token:
                raise GuardedAdmissionError("admission claim is owned by another client")
            if int(claim["fencing_epoch"]) != fencing_epoch:
                raise GuardedAdmissionError("admission fencing epoch is stale")
            scope = c.execute(
                "SELECT * FROM admission_scopes WHERE claim_key=?", (claim_key,)
            ).fetchone()
            if scope is None:
                raise GuardedAdmissionError("admission claim has no protected scope")
            matched = self._protected_scope_for_job(
                c, str(job["experiment"]), job.get("config") or {}
            )
            if matched is None or matched["claim_key"] != claim_key:
                raise GuardedAdmissionError(
                    "normalized request does not match the claim's protected scope"
                )
            job["admission_claim_key"] = claim_key
            job["admission_owner_token"] = owner_token
            job["admission_fencing_epoch"] = fencing_epoch
            job["idempotency_key"] = idempotency_key
            job["scope_sha256"] = scope["scope_sha256"]
            job["request_sha256"] = request_sha256
            self._insert_job_row(c, job)
            inserted = c.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job["job_id"],)
            ).fetchone()
        if inserted is None:  # pragma: no cover - SQLite INSERT invariant
            raise RuntimeError("guarded job insert disappeared")
        return _job_dict(inserted), True

    def seal_job_payload(
        self,
        job_id: str,
        *,
        owner_token: str | None,
        fencing_epoch: int | None,
        sha256: str,
        size_bytes: int,
        body: bytes,
        path: Path,
        now: float | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Create-once payload seal; make the job schedulable only after durable bytes.

        File promotion happens while the SQLite write lock is held and before the state
        transition to ``queued``.  A crash after rename but before commit is recovered
        only when the next PUT supplies byte-identical content.
        """

        current = time.time() if now is None else float(now)
        if len(body) != size_bytes or hashlib.sha256(body).hexdigest() != sha256:
            raise GuardedAdmissionError(
                "payload body does not match the supplied digest and size"
            )
        created = False
        with self.immediate_tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["payload_kind"] != "shipped":
                raise GuardedAdmissionError("job does not accept a shipped payload")
            if row["state"] not in ("awaiting_payload", "queued"):
                raise GuardedAdmissionError(
                    f"payload is immutable once job state is {row['state']!r}"
                )
            custody_required = bool(row["custody_required"])
            if custody_required:
                if owner_token is None or fencing_epoch is None:
                    raise GuardedAdmissionError(
                        "guarded payload PUT requires owner and fencing epoch"
                    )
                claim = c.execute(
                    "SELECT * FROM admission_claims WHERE claim_key=?",
                    (row["admission_claim_key"],),
                ).fetchone()
                if (
                    claim is None
                    or float(claim["expires_ts"]) <= current
                    or claim["owner_token"] != owner_token
                    or int(claim["fencing_epoch"]) != fencing_epoch
                    or row["admission_owner_token"] != owner_token
                    or int(row["admission_fencing_epoch"]) != fencing_epoch
                ):
                    raise GuardedAdmissionError(
                        "payload owner or fencing epoch is stale"
                    )
            declared_sha = row["payload_declared_sha256"]
            declared_size = row["payload_declared_size"]
            if custody_required and (declared_sha is None or declared_size is None):
                raise GuardedAdmissionError(
                    "guarded payload has no immutable declared identity"
                )
            if declared_sha is not None and (
                declared_size is None
                or declared_sha != sha256
                or int(declared_size) != size_bytes
            ):
                raise GuardedAdmissionError("payload bytes differ from declared digest")
            sealed_sha = row["payload_sealed_sha256"]
            if sealed_sha is not None:
                if sealed_sha != sha256 or int(row["payload_sealed_size"]) != size_bytes:
                    raise GuardedAdmissionError("payload was already sealed with different bytes")
                if not path.is_file() or path.read_bytes() != body:
                    raise GuardedAdmissionError(
                        "sealed payload file is absent or differs from its database digest"
                    )
            else:
                if path.exists():
                    if not path.is_file() or path.read_bytes() != body:
                        raise GuardedAdmissionError(
                            "uncommitted payload file exists with different bytes"
                        )
                else:
                    tmp = path.with_name(
                        f".{path.name}.{os.getpid()}.{threading.get_ident()}."
                        f"{time.time_ns()}.tmp"
                    )
                    try:
                        with open(tmp, "xb") as handle:
                            handle.write(body)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(tmp, path)
                        # Persist the directory entry before advertising QUEUED.
                        dir_fd = os.open(path.parent, os.O_RDONLY)
                        try:
                            os.fsync(dir_fd)
                        finally:
                            os.close(dir_fd)
                    finally:
                        if tmp.exists():
                            tmp.unlink()
                if custody_required:
                    c.execute(
                        """UPDATE jobs
                           SET payload_sealed_sha256=?, payload_sealed_size=?,
                               payload_sealed_ts=?, state='queued', status_detail=NULL
                           WHERE job_id=? AND payload_declared_sha256=?
                             AND payload_declared_size=?""",
                        (sha256, size_bytes, current, job_id, sha256, size_bytes),
                    )
                else:
                    c.execute(
                        """UPDATE jobs
                           SET payload_declared_sha256=COALESCE(payload_declared_sha256, ?),
                               payload_declared_size=COALESCE(payload_declared_size, ?),
                               payload_sealed_sha256=?, payload_sealed_size=?,
                               payload_sealed_ts=?, state='queued', status_detail=NULL
                           WHERE job_id=?""",
                        (sha256, size_bytes, sha256, size_bytes, current, job_id),
                    )
                created = True
            updated = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if updated is None:  # pragma: no cover - row held throughout transaction
            raise RuntimeError("payload seal job disappeared")
        return _job_dict(updated), created

    def report_executed_payload(
        self,
        job_id: str,
        *,
        host: str,
        sha256: str,
        size_bytes: int,
        now: float | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Persist the exact archive downloaded by the assigned agent before launch."""

        current = time.time() if now is None else float(now)
        created = False
        with self.immediate_tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["state"] not in ("assigned", "preparing"):
                raise GuardedAdmissionError(
                    f"executed payload report is invalid in state {row['state']!r}"
                )
            if row["assigned_host"] != host:
                raise GuardedAdmissionError("payload reporter is not the assigned host")
            if (
                row["payload_sealed_sha256"] != sha256
                or row["payload_sealed_size"] is None
                or int(row["payload_sealed_size"]) != size_bytes
            ):
                raise GuardedAdmissionError("executed payload differs from sealed bytes")
            if row["payload_executed_sha256"] is not None:
                if (
                    row["payload_executed_sha256"] != sha256
                    or int(row["payload_executed_size"]) != size_bytes
                    or row["payload_executed_host"] != host
                ):
                    raise GuardedAdmissionError(
                        "executed payload was already reported with different custody"
                    )
            else:
                c.execute(
                    """UPDATE jobs SET payload_executed_sha256=?,
                           payload_executed_size=?, payload_executed_ts=?,
                           payload_executed_host=? WHERE job_id=?""",
                    (sha256, size_bytes, current, host, job_id),
                )
                created = True
            updated = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if updated is None:  # pragma: no cover
            raise RuntimeError("executed payload job disappeared")
        return _job_dict(updated), created

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self.tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _job_dict(row) if row else None

    def jobs(self, *, state: str | None = None, client_run_id: str | None = None,
             host: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM jobs WHERE 1=1", []
        if state:
            q += " AND state=?"
            args.append(state)
        if client_run_id:
            q += " AND client_run_id=?"
            args.append(client_run_id)
        if host:
            q += " AND assigned_host=?"
            args.append(host)
        q += " ORDER BY priority DESC, created_ts ASC"
        with self.tx() as c:
            rows = c.execute(q, args).fetchall()
        return [_job_dict(r) for r in rows]

    def active_jobs_on(self, host: str) -> list[dict[str, Any]]:
        with self.tx() as c:
            rows = c.execute(
                "SELECT * FROM jobs WHERE assigned_host=? AND state IN "
                "('assigned','preparing','running')",
                (host,),
            ).fetchall()
        return [_job_dict(r) for r in rows]

    def merge_meta(
        self, job_id: str, patch: dict[str, Any], *, only_if_absent: bool = False
    ) -> dict[str, Any]:
        """Read-merge-write of jobs.meta_json inside ONE transaction. Two concurrent
        writers exist by design (harness phone-home POST vs the links log-scrape cache);
        a read→merge→write spanning transactions loses one of them. ``only_if_absent``
        is for the scrape path: a scraped run_id must never overwrite the exact one the
        harness phoned home."""
        with self.tx() as c:
            row = c.execute(
                "SELECT meta_json FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            meta = json.loads(row["meta_json"] or "{}")
            for k, v in patch.items():
                if only_if_absent and k in meta:
                    continue
                meta[k] = v
            c.execute("UPDATE jobs SET meta_json=? WHERE job_id=?", (json.dumps(meta), job_id))
        return meta

    def update_job(self, job_id: str, **fields: Any) -> None:
        keys, args = [], []
        for k, v in fields.items():
            if k.endswith("_json_obj"):
                keys.append(k.replace("_json_obj", "_json") + "=?")
                args.append(json.dumps(v))
            else:
                keys.append(f"{k}=?")
                args.append(v)
        args.append(job_id)
        with self.tx() as c:
            c.execute(f"UPDATE jobs SET {', '.join(keys)} WHERE job_id=?", args)

    def set_status_detail_if_changed(
        self,
        job_id: str,
        detail: str | None,
        *,
        clear_prefixes: tuple[str, ...] = (),
    ) -> bool:
        """Atomically update ``status_detail`` only when the visible value changes.

        Queue-admission refreshes can run from several request handlers at once. Keeping
        the compare+set under the DB lock prevents duplicate "same reason" events.
        """
        with self.tx() as c:
            row = c.execute(
                "SELECT status_detail FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                return False
            prev = row["status_detail"]
            if detail is None and clear_prefixes:
                if not (isinstance(prev, str) and prev.startswith(clear_prefixes)):
                    return False
            if prev == detail:
                return False
            c.execute("UPDATE jobs SET status_detail=? WHERE job_id=?", (detail, job_id))
            return True

    def claim_job(
        self,
        job_id: str,
        *,
        host: str,
        lease_expires_ts: float,
        reservation: dict[str, Any],
        plan: dict[str, Any],
        lease_identity: str | None = None,
        lease_capability: str | None = None,
        debug_credential_id: str | None = None,
        debug_credential: str | None = None,
        debug_scopes: tuple[str, ...] = (),
        now: float | None = None,
    ) -> bool:
        """Atomically move one queued job to one agent.

        Multiple live agents may share a host identity. Their scheduler reads can pick
        the same queued row, so the state predicate belongs in the write transaction.
        Guarded jobs additionally bind an unguessable one-lease capability to the host,
        lease identity, and exact sealed payload inside the same transaction.
        """
        issued_ts = time.time() if now is None else float(now)
        with self.immediate_tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["state"] != "queued":
                return False
            guarded = bool(row["custody_required"])
            needs = json.loads(row["needs_json"] or "{}")
            debug_requested = bool(needs.get("saturn_debug_credential_v1"))
            capability_hash: str | None = None
            debug_credential_hash: str | None = None
            stored_debug_scopes: tuple[str, ...] = ()
            payload_sha256: str | None = None
            payload_size: int | None = None
            if guarded:
                if not lease_identity or not lease_capability:
                    raise GuardedAdmissionError(
                        "guarded claim requires a fresh lease identity and capability"
                    )
                payload_sha256 = row["payload_sealed_sha256"]
                raw_payload_size = row["payload_sealed_size"]
                if payload_sha256 is None or raw_payload_size is None:
                    raise GuardedAdmissionError(
                        "guarded claim requires an exact sealed payload"
                    )
                payload_size = int(raw_payload_size)
                if (
                    row["payload_declared_sha256"] != payload_sha256
                    or row["payload_declared_size"] is None
                    or int(row["payload_declared_size"]) != payload_size
                ):
                    raise GuardedAdmissionError(
                        "guarded claim declaration and seal differ"
                    )
                capability_hash = _lease_capability_digest(
                    lease_capability,
                    job_id=job_id,
                    host=host,
                    lease_identity=lease_identity,
                    payload_sha256=payload_sha256,
                    payload_size=payload_size,
                )
                if debug_requested:
                    if not debug_credential_id or not debug_credential:
                        raise GuardedAdmissionError(
                            "guarded debugger claim requires a fresh job debugger credential"
                        )
                    stored_debug_scopes = tuple(sorted(set(debug_scopes)))
                    if not stored_debug_scopes:
                        raise GuardedAdmissionError(
                            "guarded debugger claim requires at least one narrow scope"
                        )
                    debug_credential_hash = _debug_credential_digest(
                        debug_credential,
                        credential_id=debug_credential_id,
                        job_id=job_id,
                        host=host,
                        lease_identity=lease_identity,
                        payload_sha256=payload_sha256,
                        payload_size=payload_size,
                        scopes=stored_debug_scopes,
                    )
                elif debug_credential_id is not None or debug_credential is not None:
                    raise GuardedAdmissionError(
                        "job did not request guarded debugger authority"
                    )
            elif any(
                value is not None
                for value in (
                    lease_identity,
                    lease_capability,
                    debug_credential_id,
                    debug_credential,
                )
            ):
                raise GuardedAdmissionError(
                    "ordinary job must not receive guarded lease authority"
                )
            result = c.execute(
                """UPDATE jobs
                   SET state='assigned', assigned_host=?, lease_expires_ts=?,
                       reservation_json=?, plan_json=?, lease_identity=?,
                       lease_capability_hash=?, lease_payload_sha256=?,
                       lease_payload_size=?, lease_issued_ts=?,
                       debug_credential_id=?, debug_credential_hash=?,
                       debug_scopes_json=?, debug_issued_ts=?
                   WHERE job_id=? AND state='queued'
                   AND (custody_required=0 OR payload_sealed_sha256 IS NOT NULL)""",
                (
                    host,
                    lease_expires_ts,
                    json.dumps(reservation),
                    json.dumps(plan),
                    lease_identity,
                    capability_hash,
                    payload_sha256,
                    payload_size,
                    issued_ts if guarded else None,
                    debug_credential_id if debug_requested else None,
                    debug_credential_hash,
                    json.dumps(stored_debug_scopes) if debug_requested else None,
                    issued_ts if debug_requested else None,
                    job_id,
                ),
            )
        return result.rowcount == 1

    def authorize_guarded_agent_lease(
        self,
        job_id: str,
        *,
        lease_identity: str,
        lease_capability: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Validate one sensitive agent action against the current guarded lease.

        Authority ends at lease expiry or the first terminal transition. The raw
        capability is never stored; its digest is recomputed against all bound fields.
        """

        current = time.time() if now is None else float(now)
        with self.tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if not bool(row["custody_required"]):
            raise GuardedAdmissionError("job does not use guarded lease authority")
        if row["state"] not in ("assigned", "preparing", "running"):
            raise GuardedAdmissionError(
                f"guarded lease is inactive in state {row['state']!r}"
            )
        if row["lease_expires_ts"] is None or float(row["lease_expires_ts"]) <= current:
            raise GuardedAdmissionError("guarded lease capability is expired")
        if not lease_identity or row["lease_identity"] != lease_identity:
            raise GuardedAdmissionError("guarded lease identity does not match")
        host = row["assigned_host"]
        payload_sha256 = row["payload_sealed_sha256"]
        payload_size = row["payload_sealed_size"]
        if (
            not isinstance(host, str)
            or payload_sha256 is None
            or payload_size is None
            or row["lease_payload_sha256"] != payload_sha256
            or row["lease_payload_size"] is None
            or int(row["lease_payload_size"]) != int(payload_size)
        ):
            raise GuardedAdmissionError("guarded lease payload binding is invalid")
        expected = _lease_capability_digest(
            lease_capability,
            job_id=job_id,
            host=host,
            lease_identity=lease_identity,
            payload_sha256=str(payload_sha256),
            payload_size=int(payload_size),
        )
        stored = row["lease_capability_hash"]
        if not isinstance(stored, str) or not hmac.compare_digest(stored, expected):
            raise GuardedAdmissionError("guarded lease capability does not match")
        return _job_dict(row)

    def authorize_guarded_debug_credential(
        self,
        job_id: str,
        *,
        credential_id: str,
        credential: str,
        scope: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Authorize one job-child debugger action and nothing else.

        This deliberately does not accept the general agent credential or its
        lease bearer.  Authority ends with the current lease and is rebound on
        every retry/takeover.  This is API-level least privilege: it does not
        isolate the bearer from hostile processes running under the same Unix UID.
        """

        current = time.time() if now is None else float(now)
        with self.tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if not bool(row["custody_required"]):
            raise GuardedAdmissionError("job does not use guarded debugger authority")
        needs = json.loads(row["needs_json"] or "{}")
        if not needs.get("saturn_debug_credential_v1"):
            raise GuardedAdmissionError("job did not request debugger authority")
        if row["state"] not in ("assigned", "preparing", "running"):
            raise GuardedAdmissionError(
                f"debugger credential is inactive in state {row['state']!r}"
            )
        if row["lease_expires_ts"] is None or float(row["lease_expires_ts"]) <= current:
            raise GuardedAdmissionError("debugger credential is expired")
        if not credential_id or row["debug_credential_id"] != credential_id:
            raise GuardedAdmissionError("debugger credential identity does not match")
        lease_identity = row["lease_identity"]
        host = row["assigned_host"]
        payload_sha256 = row["payload_sealed_sha256"]
        payload_size = row["payload_sealed_size"]
        if (
            not isinstance(lease_identity, str)
            or not isinstance(host, str)
            or payload_sha256 is None
            or payload_size is None
            or row["lease_payload_sha256"] != payload_sha256
            or row["lease_payload_size"] is None
            or int(row["lease_payload_size"]) != int(payload_size)
        ):
            raise GuardedAdmissionError("debugger credential lease binding is invalid")
        try:
            scopes = tuple(json.loads(row["debug_scopes_json"] or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GuardedAdmissionError("debugger credential scopes are invalid") from exc
        if scope not in scopes:
            raise GuardedAdmissionError(
                f"debugger credential does not grant scope {scope!r}"
            )
        expected = _debug_credential_digest(
            credential,
            credential_id=credential_id,
            job_id=job_id,
            host=host,
            lease_identity=lease_identity,
            payload_sha256=str(payload_sha256),
            payload_size=int(payload_size),
            scopes=tuple(sorted(scopes)),
        )
        stored = row["debug_credential_hash"]
        if not isinstance(stored, str) or not hmac.compare_digest(stored, expected):
            raise GuardedAdmissionError("debugger credential does not match")
        return _job_dict(row)

    def expire_leases(self) -> list[str]:
        """Mark active jobs whose lease lapsed as lost; return their ids."""
        now = time.time()
        with self.tx() as c:
            rows = c.execute(
                "SELECT job_id, assigned_host FROM jobs "
                "WHERE state IN ('assigned','preparing','running') "
                "AND lease_expires_ts IS NOT NULL AND lease_expires_ts < ?",
                (now,),
            ).fetchall()
            ids = [r["job_id"] for r in rows]
            if ids:
                c.executemany(
                    "UPDATE jobs SET state='lost', finished_ts=?, status_detail=? WHERE job_id=?",
                    [(now, "lease expired (agent silent)", j) for j in ids],
                )
                c.executemany(
                    """INSERT INTO events(ts, kind, job_id, host, state, reason, detail,
                           payload_json)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    [
                        (
                            now,
                            "job.lost",
                            r["job_id"],
                            r["assigned_host"],
                            "lost",
                            "lease-expired",
                            "lease expired (agent silent)",
                            None,
                        )
                        for r in rows
                    ],
                )
        return ids

    # ---- model inventory (warm-cache map; MLflow stays the canonical registry) ----
    def replace_inventory(self, host: str, rows: list[dict[str, Any]]) -> None:
        now = time.time()
        with self.tx() as c:
            c.execute("DELETE FROM host_models WHERE host=?", (host,))
            c.executemany(
                """INSERT OR REPLACE INTO host_models(
                    host, model, kind, bytes, path, ts, artifact_id, artifact_kind,
                    variant, mount, manifest_sha256, content_hash, source_model, locator_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        host,
                        r.get("model"),
                        # The legacy primary key has no variant column. Encode the
                        # variant in qstore kind so multiple suffixed stores can
                        # coexist without a destructive schema rewrite. Consumers
                        # treat qstore:<variant> as a qstore warm kind.
                        (
                            f"qstore:{r.get('variant')}"
                            if r.get("kind", "weights") == "qstore" and r.get("variant")
                            else r.get("kind", "weights")
                        ),
                        r.get("bytes"),
                        r.get("path"),
                        now,
                        r.get("artifact_id"),
                        r.get("artifact_kind"),
                        r.get("variant"),
                        r.get("mount"),
                        r.get("manifest_sha256"),
                        r.get("content_hash"),
                        r.get("source_model"),
                        json.dumps(r.get("locator"), sort_keys=True)
                        if isinstance(r.get("locator"), dict)
                        else None,
                    )
                    for r in rows
                    if r.get("model")
                ],
            )

    def hosts_with_model(self, model: str) -> dict[str, list[str]]:
        """host -> kinds present (e.g. ['weights', 'qstore'])."""
        with self.tx() as c:
            rows = c.execute(
                "SELECT host, kind FROM host_models WHERE model=?", (model,)
            ).fetchall()
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["host"], []).append(r["kind"])
        return out

    def hosts_with_artifacts(self, model: str) -> dict[str, list[str]]:
        """host -> legacy warm kinds plus exact artifact IDs when available."""
        with self.tx() as c:
            rows = c.execute(
                "SELECT host, kind, artifact_id FROM host_models WHERE model=?", (model,)
            ).fetchall()
        out: dict[str, list[str]] = {}
        for row in rows:
            values = out.setdefault(row["host"], [])
            if row["kind"] not in values:
                values.append(row["kind"])
            artifact_id = row["artifact_id"]
            if artifact_id and artifact_id not in values:
                values.append(artifact_id)
        return out

    def inventory(self, host: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM host_models", []
        if host:
            q += " WHERE host=?"
            args.append(host)
        with self.tx() as c:
            rows = [dict(r) for r in c.execute(q + " ORDER BY host, model, kind", args)]
        for row in rows:
            raw_locator = row.pop("locator_json", None)
            if raw_locator:
                try:
                    row["locator"] = json.loads(raw_locator)
                except (TypeError, ValueError):
                    row["locator"] = None
            else:
                row["locator"] = None
        return rows

    # ---- estimates ----------------------------------------------------------
    def add_estimate(self, row: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO estimates(client_run_id, experiment, model, host, status,
                       ram_peak_mb, vram_peak_mb, wall_s, backend, dtype, task_family,
                       est_ram_mb, ts, family_key, reserved_ram_mb, reserved_vram_mb,
                       kill_state)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row.get("client_run_id"),
                    row.get("experiment"),
                    row.get("model"),
                    row.get("host"),
                    row.get("status"),
                    row.get("ram_peak_mb"),
                    row.get("vram_peak_mb"),
                    row.get("wall_s"),
                    row.get("backend"),
                    row.get("dtype"),
                    row.get("task_family"),
                    row.get("est_ram_mb"),
                    time.time(),
                    row.get("family_key"),
                    row.get("reserved_ram_mb"),
                    row.get("reserved_vram_mb"),
                    row.get("kill_state"),
                ),
            )

    def history_stats(self, family_key: str) -> dict[str, Any] | None:
        """Generalized history over succeeded runs of a reservation family
        (``protocol.family_key_for`` — model+task, or experiment+script for the
        model-less majority): p95 RAM, p95 VRAM, median wall. Needs n>=3 so one lucky
        small run can't set the reservation."""
        with self.tx() as c:
            rows = [
                dict(r)
                for r in c.execute(
                    "SELECT ram_peak_mb, vram_peak_mb, wall_s FROM estimates "
                    "WHERE family_key=? AND status='succeeded' "
                    "AND ram_peak_mb IS NOT NULL ORDER BY ram_peak_mb",
                    (family_key,),
                )
            ]
        if len(rows) < 3:
            return None
        rams = [float(r["ram_peak_mb"]) for r in rows]
        vrams = sorted(
            float(r["vram_peak_mb"]) for r in rows if r["vram_peak_mb"] is not None
        )
        walls = sorted(float(r["wall_s"]) for r in rows if r["wall_s"] is not None)
        p95 = rams[min(len(rams) - 1, int(0.95 * len(rams)))]
        p95_vram = vrams[min(len(vrams) - 1, int(0.95 * len(vrams)))] if vrams else None
        return {
            "n": len(rows),
            "p95_ram_mb": p95,
            "p95_vram_mb": p95_vram,
            "max_ram_mb": rams[-1],
            "max_vram_mb": vrams[-1] if vrams else None,
            "p50_wall_s": walls[len(walls) // 2] if walls else None,
        }

    def exact_stats(self, client_run_id: str) -> dict[str, Any] | None:
        """Aggregate history for this EXACT config: n, p95-or-max peak, latest wall.

        Feeds the n-aware exact rung of the sizing ladder: with n>=3 the p95 is a
        stable basis; below that the max peak with a fatter factor covers 1s-sampling
        undercount (measured 976 vs 2162MB).
        """
        with self.tx() as c:
            rows = [
                dict(r)
                for r in c.execute(
                    "SELECT ram_peak_mb, vram_peak_mb, wall_s FROM estimates "
                    "WHERE client_run_id=? AND status='succeeded' "
                    "AND ram_peak_mb IS NOT NULL ORDER BY ts DESC LIMIT 20",
                    (client_run_id,),
                )
            ]
        if not rows:
            return None
        rams = sorted(float(r["ram_peak_mb"]) for r in rows)
        vrams = sorted(
            float(r["vram_peak_mb"]) for r in rows if r["vram_peak_mb"] is not None
        )
        n = len(rows)
        ram_basis = rams[min(n - 1, int(0.95 * n))] if n >= 3 else rams[-1]
        vram_basis = (
            vrams[min(len(vrams) - 1, int(0.95 * len(vrams)))]
            if len(vrams) >= 3
            else (vrams[-1] if vrams else None)
        )
        return {
            "n": n,
            "ram_peak_mb": ram_basis,
            "vram_peak_mb": vram_basis,
            "max_ram_mb": rams[-1],
            "max_vram_mb": vrams[-1] if vrams else None,
            "wall_s": rows[0].get("wall_s"),
        }

    def calibration(self) -> list[dict[str, Any]]:
        """Per-task-family undershoot ratios (peak / est) — the feedback that tunes
        estimate_activation_mb and TASK_RSS_FACTOR."""
        with self.tx() as c:
            rows = c.execute(
                """SELECT task_family, COUNT(*) AS n,
                          AVG(ram_peak_mb / est_ram_mb) AS mean_ratio,
                          MAX(ram_peak_mb / est_ram_mb) AS max_ratio
                   FROM estimates
                   WHERE status='succeeded' AND est_ram_mb > 0 AND ram_peak_mb IS NOT NULL
                   GROUP BY task_family"""
            ).fetchall()
        return [dict(r) for r in rows]

    def history_for(self, client_run_id: str) -> dict[str, Any] | None:
        """Latest successful observation for this exact config."""
        with self.tx() as c:
            row = c.execute(
                "SELECT * FROM estimates WHERE client_run_id=? AND status='succeeded' "
                "ORDER BY ts DESC LIMIT 1",
                (client_run_id,),
            ).fetchone()
        return dict(row) if row else None


def _job_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["needs"] = json.loads(d.pop("needs_json") or "{}")
    d["reservation"] = json.loads(d.pop("reservation_json") or "{}")
    d["cmd"] = json.loads(d.pop("cmd_json") or "[]")
    d["config"] = json.loads(d.pop("config_json") or "{}")
    d["result"] = json.loads(d.pop("result_json") or "null")
    d["plans"] = json.loads(d.pop("plans_json", None) or "null")
    d["plan"] = json.loads(d.pop("plan_json", None) or "null")
    d["meta"] = json.loads(d.pop("meta_json", None) or "null")
    claim_key = d.pop("admission_claim_key", None)
    owner_token = d.pop("admission_owner_token", None)
    fencing_epoch = d.pop("admission_fencing_epoch", None)
    idempotency_key = d.pop("idempotency_key", None)
    scope_sha256 = d.pop("scope_sha256", None)
    d["admission"] = (
        {
            "claim_key": claim_key,
            "fencing_epoch": fencing_epoch,
            "idempotency_key": idempotency_key,
            "scope_sha256": scope_sha256,
        }
        if claim_key is not None
        else None
    )
    # The bearer credential remains server-side; callers receive only its immutable
    # ownership/fence identity through the admission object above.
    del owner_token
    d["submitted_request"] = json.loads(d.pop("request_json", None) or "null")
    d["submitted_request_sha256"] = d.pop("request_sha256", None)
    required = bool(d.pop("custody_required", 0))
    declared_sha = d.pop("payload_declared_sha256", None)
    declared_size = d.pop("payload_declared_size", None)
    sealed_sha = d.pop("payload_sealed_sha256", None)
    sealed_size = d.pop("payload_sealed_size", None)
    sealed_ts = d.pop("payload_sealed_ts", None)
    executed_sha = d.pop("payload_executed_sha256", None)
    executed_size = d.pop("payload_executed_size", None)
    executed_ts = d.pop("payload_executed_ts", None)
    executed_host = d.pop("payload_executed_host", None)
    d["payload_custody"] = {
        "required": required,
        "declared": (
            {"sha256": declared_sha, "size_bytes": declared_size}
            if declared_sha is not None
            else None
        ),
        "sealed": (
            {"sha256": sealed_sha, "size_bytes": sealed_size, "sealed_ts": sealed_ts}
            if sealed_sha is not None
            else None
        ),
        "executed": (
            {
                "sha256": executed_sha,
                "size_bytes": executed_size,
                "reported_ts": executed_ts,
                "host": executed_host,
            }
            if executed_sha is not None
            else None
        ),
    }
    # Lease capabilities are intentionally absent from every ordinary job/status/
    # receipt serialization. The scheduler adds the raw, one-use authority only to the
    # successful agent lease response that created it.
    d.pop("lease_identity", None)
    d.pop("lease_capability_hash", None)
    d.pop("lease_payload_sha256", None)
    d.pop("lease_payload_size", None)
    d.pop("lease_issued_ts", None)
    d.pop("debug_credential_id", None)
    d.pop("debug_credential_hash", None)
    d.pop("debug_scopes_json", None)
    d.pop("debug_issued_ts", None)
    return d


def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["payload"] = json.loads(d.pop("payload_json") or "null")
    return d


def _admission_claim_dict(
    row: sqlite3.Row, *, scope_row: sqlite3.Row | None = None
) -> dict[str, Any]:
    d = dict(row)
    d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
    if scope_row is not None:
        d["scope"] = json.loads(scope_row["scope_json"])
        d["scope_sha256"] = scope_row["scope_sha256"]
    return d


def _admission_scope_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["scope"] = json.loads(d.pop("scope_json"))
    return d


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
