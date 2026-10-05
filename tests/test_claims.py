from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from mrun.claims import (
    ClaimRow,
    claim_from_cuda_graph_decision,
    claim_from_promotion_report,
    export_claim,
)

_SHA = "a" * 64


def _row(**overrides) -> ClaimRow:
    base = dict(
        claim_type="workplan-promotion",
        verdict="promotable",
        scope={"model": "m/test"},
        custody={"bundle_fingerprint": _SHA},
        detail="ok",
        created_ts=1.0,
    )
    base.update(overrides)
    return ClaimRow(**base)


def test_claim_row_rejects_bad_digest_and_verdict():
    with pytest.raises(ValueError):
        _row(custody={"bundle_fingerprint": "nope"})
    with pytest.raises(ValueError):
        _row(verdict="maybe")


def test_claim_id_is_content_addressed_and_tamper_detected():
    row = _row()
    same = _row(created_ts=99.0)  # created_ts excluded from the id
    assert row.claim_id == same.claim_id
    payload = row.as_dict()
    payload["detail"] = "changed"
    with pytest.raises(ValueError):
        ClaimRow.from_dict(payload)
    assert ClaimRow.from_dict(row.as_dict()).claim_id == row.claim_id


def test_export_claim_idempotent_and_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("MRUN_CLAIMS_DIR", str(tmp_path / "claims"))
    row = _row()
    first = export_claim(row)
    second = export_claim(row)
    assert first["created"] is True and second["created"] is False
    assert first["claim_id"] == row.claim_id
    log_lines = (tmp_path / "claims" / "log.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == 1
    stored = json.loads((tmp_path / "claims" / "rows" / f"{row.claim_id}.json").read_text())
    assert stored["verdict"] == "promotable"


def test_export_claim_concurrent_is_single_publish(tmp_path, monkeypatch):
    monkeypatch.setenv("MRUN_CLAIMS_DIR", str(tmp_path / "claims"))
    row = _row()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: export_claim(row), range(16)))
    assert sum(bool(result.get("created")) for result in results) == 1
    log_lines = (tmp_path / "claims" / "log.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == 1


def test_export_claim_never_raises(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a dir")
    monkeypatch.setenv("MRUN_CLAIMS_DIR", str(blocker))
    result = export_claim(_row())
    assert "error" in result
    assert "claim export skipped" in capsys.readouterr().err




def test_claim_from_promotion_report_blocked():
    report = SimpleNamespace(
        target="candidate",
        promotable=False,
        satisfied_requirements=("content-identity-verified",),
        blockers=("execution placement is not verified",),
    )
    row = claim_from_promotion_report(report, _SHA, scope={"model": "m/test"})
    assert row.verdict == "blocked"
    assert row.scope["target"] == "candidate"
    assert "placement" in row.detail


def test_claim_from_cuda_graph_decision_refusal():
    decision = SimpleNamespace(
        selected=False,
        reason="no promotion record for this hardware",
        promotion=None,
        blockers=("missing promotion registry entry",),
    )
    row = claim_from_cuda_graph_decision(decision, scope={"backend": "dense-qstore-cuda"})
    assert row.verdict == "refused"
    assert row.custody == {}
    assert "promotion registry" in row.detail
