from __future__ import annotations

from mrun.diagnostics import failure_record, log_tail, redact_command, redact_text


def test_redact_command_and_text_remove_common_credentials():
    command = redact_command(
        ["python", "run.py", "--token", "abc123", "--api-key=xyz", "Bearer secret"]
    )
    assert command == ["python", "run.py", "--token", "<redacted>", "--api-key=<redacted>", "Bearer <redacted>"]
    assert redact_text("password=hidden token:secret") == (
        "password=<redacted> token:<redacted>"
    )


def test_failure_record_contains_exception_and_log_tail(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("boot\nreal failure: token=hidden\n", encoding="utf-8")
    try:
        raise ValueError("bad shape")
    except ValueError as exc:
        record = failure_record(
            kind="agent_exception",
            phase="process.prepare",
            message="launch failed",
            command=["python", "run.py", "--password", "secret"],
            cwd=tmp_path,
            exception=exc,
            log_path=log,
            resources={"peak_rss_mb": 12.0},
        )
    assert record["kind"] == "agent_exception"
    assert record["phase"] == "process.prepare"
    assert record["command"][-2:] == ["--password", "<redacted>"]
    assert record["exception"]["type"] == "ValueError"
    assert "token=<redacted>" in record["log_tail"]
    assert log_tail(tmp_path / "missing.log") is None
