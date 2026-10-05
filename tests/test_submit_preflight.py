from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mrun.client.preflight import PreflightRejected
from mrun.client.submit import submit


class PreflightRecordingApi:
    """Handles the history GET (preflight check 5) plus the ordinary submit POST/PUT."""

    def __init__(self) -> None:
        self.body: dict[str, Any] | None = None
        self.payload_bytes: bytes | None = None

    def json(self, method: str, path: str, **kwargs: Any) -> Any:
        if method == "GET" and path.startswith("/api/history/"):
            return {"exact": None, "generalized": None}
        assert method == "POST" and path == "/api/jobs"
        self.body = kwargs.get("json_body")
        return {"job_id": "job-test", "reservation": {"ram_mb": 8000}}

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        assert method == "PUT"
        self.payload_bytes = kwargs.get("raw_body")
        return (200, b"{}", {})


def _payload(tmp_path: Path) -> Path:
    root = tmp_path / "payload"
    (root / "src").mkdir(parents=True)
    (root / "src" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    return root


def test_submit_preflight_true_blocks_before_post(tmp_path, monkeypatch):
    monkeypatch.setenv("MRUN_PREFLIGHT_DIR", str(tmp_path / "receipts"))
    api = PreflightRecordingApi()
    with pytest.raises(PreflightRejected) as excinfo:
        submit(
            experiment="pf-block",
            cmd=["python", "src/run.py", "--parent-verification", "custody/missing.json"],
            payload=_payload(tmp_path),
            env_alias=None,
            api=api,
            preflight=True,
        )
    assert api.body is None
    assert api.payload_bytes is None
    receipts = list((tmp_path / "receipts").glob("preflight-*.json"))
    assert len(receipts) == 1
    assert excinfo.value.receipt.verdict == "rejected"


def test_submit_default_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("MRUN_PREFLIGHT_DIR", str(tmp_path / "receipts"))
    api = PreflightRecordingApi()
    job_id = submit(
        experiment="pf-off",
        cmd=["python", "src/run.py", "--parent-verification", "custody/missing.json"],
        payload=_payload(tmp_path),
        env_alias=None,
        api=api,
    )
    assert job_id == "job-test"
    assert api.body is not None
    assert not (tmp_path / "receipts").exists()


def test_submit_preflight_warn_submits_anyway(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MRUN_PREFLIGHT_DIR", str(tmp_path / "receipts"))
    api = PreflightRecordingApi()
    job_id = submit(
        experiment="pf-warn",
        cmd=["python", "src/run.py", "--parent-verification", "custody/missing.json"],
        payload=_payload(tmp_path),
        env_alias=None,
        api=api,
        preflight="warn",
    )
    assert job_id == "job-test"
    assert api.body is not None
    assert "preflight WARN" in capsys.readouterr().err
    assert list((tmp_path / "receipts").glob("preflight-*.json"))


def test_submit_preflight_clean_payload_passes_and_posts(tmp_path, monkeypatch):
    monkeypatch.setenv("MRUN_PREFLIGHT_DIR", str(tmp_path / "receipts"))
    api = PreflightRecordingApi()
    job_id = submit(
        experiment="pf-pass",
        cmd=["python", "src/run.py"],
        payload=_payload(tmp_path),
        env_alias=None,
        api=api,
        preflight=True,
    )
    assert job_id == "job-test"
    assert api.payload_bytes is not None
