"""Guards: in-process ram_guard/check_rss and the run_stage subprocess kill ceiling."""

from __future__ import annotations

import sys

import pytest

import mrun.guard as guard_module
from mrun.guard import check_rss, ram_guard, rss_mb, set_thread_env
from mrun.orchestrate import run_stage


def test_rss_mb_positive():
    assert rss_mb() > 10.0


def test_check_rss_under_limit_returns_value():
    assert check_rss("test", limit_mb=1_000_000) > 0


def test_check_rss_over_limit_raises():
    with pytest.raises(MemoryError):
        check_rss("test", limit_mb=1.0)


def test_ram_guard_context(capsys):
    with ram_guard("phase"):
        pass


def test_ram_guard_checks_exit_boundary(monkeypatch):
    readings = iter((10.0, 30.0))
    monkeypatch.setattr(guard_module, "rss_mb", lambda: next(readings))
    with pytest.raises(MemoryError, match=r"phase \(exit\)"):
        with ram_guard("phase", limit_mb=20.0, verbose=False):
            pass


def test_ram_guard_preserves_body_exception(monkeypatch):
    readings = iter((10.0, 30.0))
    monkeypatch.setattr(guard_module, "rss_mb", lambda: next(readings))
    with pytest.raises(RuntimeError, match="body failed"):
        with ram_guard("phase", limit_mb=20.0, verbose=False):
            raise RuntimeError("body failed")


def test_check_rss_observes_limit_set_after_import(monkeypatch):
    monkeypatch.setattr(guard_module, "rss_mb", lambda: 30.0)
    monkeypatch.setenv("RSS_LIMIT_MB", "20")
    with pytest.raises(MemoryError, match="limit 20 MB"):
        check_rss("late plan")


def test_set_thread_env(monkeypatch):
    import os

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    set_thread_env(3)
    assert os.environ["OMP_NUM_THREADS"] == "3"


def test_run_stage_ok(tmp_path):
    log = tmp_path / "ok.log"
    res = run_stage([sys.executable, "-c", "print('hi')"], log_path=log)
    assert res.status == "ok"
    assert res.returncode == 0
    assert "hi" in log.read_text()


def test_run_stage_kills_memory_hog(tmp_path):
    hog = "x = bytearray(800 * 1024 * 1024); import time; time.sleep(30)"
    res = run_stage(
        [sys.executable, "-c", hog],
        log_path=tmp_path / "hog.log",
        ram_limit_mb=200,
        poll_s=0.2,
    )
    assert res.status == "killed_ram"
    assert res.elapsed_s < 30
    assert res.failure is not None
    assert res.failure["kind"] == "killed_ram"
    assert res.failure["resources"]["ceiling_mb"] == 200


def test_run_stage_timeout(tmp_path):
    res = run_stage(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        log_path=tmp_path / "slow.log",
        timeout_s=1.0,
        poll_s=0.2,
    )
    assert res.status == "timeout"
    assert res.elapsed_s < 10
    assert res.failure is not None
    assert res.failure["kind"] == "timeout"


def test_run_stage_failed(tmp_path):
    res = run_stage(
        [sys.executable, "-c", "print('failure-output'); raise SystemExit(3)"],
        log_path=tmp_path / "f.log",
    )
    assert res.status == "failed"
    assert res.returncode == 3
    assert res.failure is not None
    assert res.failure["kind"] == "process_exit"
    assert res.failure["phase"] == "process"
    assert "failure-output" in res.failure["log_tail"]
    assert res.diagnostics["command"][0] == sys.executable


def test_run_stage_spawn_failure_is_diagnosable(tmp_path):
    res = run_stage([str(tmp_path / "does-not-exist")], log_path=tmp_path / "spawn.log")
    assert res.status == "failed"
    assert res.failure is not None
    assert res.failure["kind"] == "spawn_error"
    assert res.failure["exception"]["type"] == "FileNotFoundError"
