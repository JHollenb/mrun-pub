from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrun.client.preflight import (
    CheckResult,
    PreflightReceipt,
    check_cmd_path_closure,
    check_hashed_root_hygiene,
    check_reservation_sanity,
    check_verifier_dry_run,
    check_workload_geometry,
    run_preflight,
    write_receipt,
)


def _payload(tmp_path: Path) -> Path:
    root = tmp_path / "payload"
    (root / "src").mkdir(parents=True)
    (root / "frozen").mkdir()
    (root / "src" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "frozen" / "policy.v1.json").write_text("{}\n", encoding="utf-8")
    return root


# -- check 1: cmd path closure -------------------------------------------------


def test_path_closure_flags_dangling_relative_custody_path(tmp_path):
    root = _payload(tmp_path)
    cmd = [
        "python",
        "src/run.py",
        "--parent-root",
        "/home/beast/.mrun/work/job-9b8d42c662df",
        "--parent-verification",
        "custody/native-parent/native-target-competition-verification.v1.json",
    ]
    result = check_cmd_path_closure(cmd, root)
    assert result.status == "failed"
    assert "custody/native-parent" in result.detail
    assert result.evidence["missing"] == [
        "custody/native-parent/native-target-competition-verification.v1.json"
    ]


def test_path_closure_accepts_absolute_and_payload_paths(tmp_path):
    root = _payload(tmp_path)
    cmd = [
        "/home/beast/domains/mrun/.venv/bin/python",
        "src/run.py",
        "--policy",
        "frozen/policy.v1.json",
        "--artifact-root",
        "/mnt/big/experiment-artifacts/foo",
        "--seed=7",
    ]
    result = check_cmd_path_closure(cmd, root)
    assert result.status == "passed"
    assert "/mnt/big/experiment-artifacts/foo" in result.evidence["absolute_unverified"]


def test_path_closure_parses_env_assignments(tmp_path):
    root = _payload(tmp_path)
    (root / "results" / "run").mkdir(parents=True)
    result = check_cmd_path_closure(
        [
            "/usr/bin/env",
            "MODEL_DIR=/mnt/models/flux",
            "PAYLOAD_ROOT=results/run",
            "python",
            "src/run.py",
        ],
        root,
    )
    assert result.status == "passed"
    assert result.evidence["absolute_unverified"] == ["/usr/bin/env", "/mnt/models/flux"]


def test_path_closure_flags_missing_relative_env_assignment(tmp_path):
    root = _payload(tmp_path)
    result = check_cmd_path_closure(
        ["/usr/bin/env", "PAYLOAD_ROOT=results/missing/run", "python", "src/run.py"],
        root,
    )
    assert result.status == "failed"
    assert result.evidence["missing"] == ["results/missing/run"]


def test_path_closure_rejects_parent_escape(tmp_path):
    root = _payload(tmp_path)
    result = check_cmd_path_closure(["python", "../outside/run.py"], root)
    assert result.status == "failed"
    assert result.evidence["escapes"] == ["../outside/run.py"]


def test_path_allowlist_suppresses_false_positive(tmp_path):
    root = _payload(tmp_path)
    cmd = ["python", "src/run.py", "--tag", "a/b"]
    assert check_cmd_path_closure(cmd, root).status == "failed"
    assert check_cmd_path_closure(cmd, root, allowlist=["a/b"]).status == "passed"


def test_path_closure_cmd_only_job_rejects_relative_paths():
    result = check_cmd_path_closure(["python", "src/run.py"], None)
    assert result.status == "failed"


# -- check 2: hashed-root hygiene ----------------------------------------------


def test_hashed_root_rejects_root_containing_work_dir(tmp_path):
    root = _payload(tmp_path)
    result = check_hashed_root_hygiene(root, ["."])
    assert result.status == "failed"
    assert "job.log" in result.detail or "job.log" in json.dumps(result.evidence)


def test_hashed_root_rejects_missing_declared_root(tmp_path):
    root = _payload(tmp_path)
    result = check_hashed_root_hygiene(root, ["results/hashed"])
    assert result.status == "failed"
    assert result.evidence["dangling"] == ["results/hashed"]


def test_hashed_root_accepts_clean_subdir(tmp_path):
    root = _payload(tmp_path)
    (root / "results" / "hashed").mkdir(parents=True)
    result = check_hashed_root_hygiene(root, ["results/hashed"])
    assert result.status == "passed"


def test_hashed_root_skipped_when_undeclared(tmp_path):
    assert check_hashed_root_hygiene(_payload(tmp_path), []).status == "skipped"


# -- check 3: verifier dry-run ---------------------------------------------------


def _verify_cfg(root: Path, script: str) -> dict:
    (root / "verify.py").write_text(script, encoding="utf-8")
    (root / "preflight").mkdir()
    (root / "preflight" / "skeleton.json").write_text(
        json.dumps({"files": {"result.json": {"json": {"ok": True}}}}),
        encoding="utf-8",
    )
    return {
        "cmd": ["python", "verify.py", "--root", "{skeleton}"],
        "skeleton": "preflight/skeleton.json",
        "timeout_s": 30,
    }


def test_verifier_dry_run_materializes_skeleton_and_passes(tmp_path):
    root = _payload(tmp_path)
    cfg = _verify_cfg(
        root,
        "import json,sys\n"
        "root = sys.argv[sys.argv.index('--root')+1]\n"
        "json.load(open(root + '/result.json'))\n",
    )
    assert check_verifier_dry_run(root, cfg).status == "passed"


def test_verifier_dry_run_gates_on_nonzero_exit(tmp_path):
    root = _payload(tmp_path)
    cfg = _verify_cfg(
        root,
        "import sys\nsys.stderr.write('missing custody file')\nsys.exit(1)\n",
    )
    result = check_verifier_dry_run(root, cfg)
    assert result.status == "failed"
    assert "missing custody file" in result.evidence["output_tail"]


def test_verifier_dry_run_skipped_without_config(tmp_path):
    assert check_verifier_dry_run(_payload(tmp_path), None).status == "skipped"


def test_verifier_dry_run_rejects_config_paths_outside_payload(tmp_path):
    root = _payload(tmp_path)
    cfg = _verify_cfg(root, "raise SystemExit(0)\n")
    cfg["skeleton"] = "../outside/skeleton.json"
    result = check_verifier_dry_run(root, cfg)
    assert result.status == "failed"
    assert "escapes the payload" in result.detail


def test_verifier_dry_run_rejects_command_paths_outside_payload(tmp_path):
    root = _payload(tmp_path)
    cfg = _verify_cfg(root, "raise SystemExit(0)\n")
    cfg["cmd"] = ["python", "../outside.py", "--root", "{skeleton}"]
    result = check_verifier_dry_run(root, cfg)
    assert result.status == "failed"
    assert "not payload-closed" in result.detail


# -- check 4: workload geometry ---------------------------------------------------


def _geometry_cfg(root: Path, body: str) -> dict:
    (root / "jobs").mkdir(exist_ok=True)
    (root / "jobs" / "predicates.py").write_text(body, encoding="utf-8")
    (root / "jobs" / "rows.json").write_text(json.dumps([{"answer": "a"}]), encoding="utf-8")
    return {"module": "jobs/predicates.py", "function": "check_rows", "rows": "jobs/rows.json"}


def test_geometry_predicate_blocks_impossible_spec(tmp_path):
    root = _payload(tmp_path)
    cfg = _geometry_cfg(
        root,
        "def check_rows(rows):\n"
        "    ok = len(rows) >= 2\n"
        "    return {'ok': ok, 'detail': 'need >=2 rows for a cyclic derangement'}\n",
    )
    result = check_workload_geometry(root, cfg)
    assert result.status == "failed"
    assert "derangement" in result.detail


def test_geometry_predicate_passes_constructible_spec(tmp_path):
    root = _payload(tmp_path)
    cfg = _geometry_cfg(
        root, "def check_rows(rows):\n    return {'ok': True, 'detail': 'ok'}\n"
    )
    assert check_workload_geometry(root, cfg).status == "passed"


def test_geometry_predicate_raising_fails_closed(tmp_path):
    root = _payload(tmp_path)
    cfg = _geometry_cfg(root, "def check_rows(rows):\n    raise RuntimeError('boom')\n")
    result = check_workload_geometry(root, cfg)
    assert result.status == "failed"
    assert "boom" in result.evidence["output_tail"]


def test_geometry_rejects_config_paths_outside_payload(tmp_path):
    root = _payload(tmp_path)
    cfg = _geometry_cfg(root, "def check_rows(rows):\n    return {'ok': True}\n")
    cfg["module"] = "../outside/predicates.py"
    result = check_workload_geometry(root, cfg)
    assert result.status == "failed"
    assert "escapes the payload" in result.detail


# -- check 5: reservation sanity --------------------------------------------------


class _HistoryApi:
    def __init__(self, body):
        self.body = body
        self.last_path = None

    def json(self, method, path, **kwargs):
        self.last_path = path
        return self.body


class _BrokenApi:
    def json(self, method, path, **kwargs):
        raise ConnectionError("no scheduler reachable")


def test_reservation_sanity_fails_below_measured_peak():
    api = _HistoryApi({"exact": {"ram_peak_mb": 9677.0}})
    result = check_reservation_sanity("run-x", {"ram_mb": 8192}, api=api)
    assert result.status == "failed"
    assert result.evidence["proposed_ram_mb"] == 11129


def test_reservation_sanity_prefers_n_aware_exact_stats_and_sends_family_key():
    api = _HistoryApi(
        {"exact": {"ram_peak_mb": 900.0}, "exact_stats": {"ram_peak_mb": 2000.0}}
    )
    result = check_reservation_sanity(
        "run-x", {"ram_mb": 1000}, family_key="model:m:forward", api=api
    )
    assert result.status == "failed"  # basis is the n-aware 2000, not the latest 900
    assert "family_key=model:m:forward" in api.last_path


def test_reservation_sanity_warns_without_declared_reservation():
    api = _HistoryApi({"exact": {"ram_peak_mb": 1000.0}})
    result = check_reservation_sanity("run-x", None, api=api)
    assert result.status == "warned"
    assert result.evidence["proposed_ram_mb"] == 1150


def test_reservation_sanity_passes_at_or_above_peak():
    api = _HistoryApi({"exact": {"ram_peak_mb": 1000.0}})
    assert check_reservation_sanity("run-x", {"ram_mb": 1000}, api=api).status == "passed"


def test_reservation_sanity_degrades_offline_and_on_error():
    assert check_reservation_sanity("run-x", {"ram_mb": 1}, api=None).status == "skipped"
    assert check_reservation_sanity("run-x", {"ram_mb": 1}, api=_BrokenApi()).status == "skipped"


# -- receipt ---------------------------------------------------------------------


def test_run_preflight_receipt_round_trip_and_tamper_raises(tmp_path, monkeypatch):
    root = _payload(tmp_path)
    monkeypatch.setenv("MRUN_PREFLIGHT_DIR", str(tmp_path / "receipts"))
    receipt = run_preflight(
        experiment="pf-exp",
        cmd=["python", "src/run.py"],
        config={"model": "m/test"},
        payload=root,
    )
    assert receipt.verdict == "passed"
    assert receipt.payload_sha256 is not None
    path = write_receipt(receipt)
    stored = json.loads(path.read_text(encoding="utf-8"))
    round_tripped = PreflightReceipt.from_dict(stored)
    assert round_tripped.fingerprint == receipt.fingerprint
    stored["verdict"] = "rejected"
    with pytest.raises(ValueError):
        PreflightReceipt.from_dict(stored)


def test_receipt_verdict_must_match_checks():
    check = CheckResult("cmd-path-closure", "failed", "boom", {})
    with pytest.raises(ValueError):
        PreflightReceipt(
            experiment="e",
            client_run_id="r",
            cmd=("x",),
            verdict="passed",
            checks=(check,),
        )


def test_run_preflight_rejects_on_dangling_path(tmp_path):
    root = _payload(tmp_path)
    receipt = run_preflight(
        experiment="pf-exp",
        cmd=["python", "src/run.py", "--parent-verification", "custody/missing.json"],
        payload=root,
    )
    assert receipt.verdict == "rejected"
    failed = [c for c in receipt.checks if c.status == "failed"]
    assert failed and failed[0].check == "cmd-path-closure"
