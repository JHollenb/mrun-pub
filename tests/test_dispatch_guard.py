"""Unit tests for the dispatch-time guards (GAP 1 VRAM, GAP 2 payload retry).

All time is faked: ``clock`` reads a mutable list and ``sleep`` advances it, so the
backoff/window logic is exercised deterministically with zero wall-clock cost.
"""

from __future__ import annotations

import pytest

from mrun.agent.dispatch_guard import (
    GuardConfig,
    PayloadRetryConfig,
    check_vram_before_launch,
    fetch_payload_with_retry,
)


class FakeTime:
    """Monotonic clock whose value only moves when sleep() is called."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


# ------------------------------------------------------------------- GAP 1: VRAM


def test_vram_launches_immediately_when_free():
    ft = FakeTime()
    logs: list[str] = []
    d = check_vram_before_launch(
        8000.0, lambda: 20000.0,
        cfg=GuardConfig(safety_margin_mb=500.0),
        sleep=ft.sleep, clock=ft.clock, log=logs.append, job_id="job-x",
    )
    assert d.launch is True
    assert d.attempts == 1
    assert ft.sleeps == []  # never held
    assert any("VRAM-OK" in m for m in logs)


def test_vram_zero_declared_skips_query():
    calls = {"n": 0}

    def q():
        calls["n"] += 1
        return 0.0

    ft = FakeTime()
    d = check_vram_before_launch(0.0, q, sleep=ft.sleep, clock=ft.clock)
    assert d.launch is True
    assert calls["n"] == 0  # a job that needs no VRAM never queries the GPU


def test_vram_holds_then_launches():
    # occupied for the first 2 checks (finishing job's context), then frees.
    seq = iter([2000.0, 4000.0, 12000.0])
    ft = FakeTime()
    logs: list[str] = []
    d = check_vram_before_launch(
        8000.0, lambda: next(seq),
        cfg=GuardConfig(safety_margin_mb=500.0, hold_window_s=120.0, backoff_s=(5.0, 10.0)),
        sleep=ft.sleep, clock=ft.clock, log=logs.append, job_id="job-hold",
    )
    assert d.launch is True
    assert d.attempts == 3
    assert ft.sleeps == [5.0, 10.0]  # two holds with the configured backoff
    assert d.waited_s == 15.0
    assert sum("VRAM-HOLD" in m for m in logs) == 2
    assert any("VRAM-OK" in m for m in logs)


def test_vram_requeue_when_never_frees():
    ft = FakeTime()
    logs: list[str] = []
    d = check_vram_before_launch(
        8000.0, lambda: 1000.0,  # GPU stays occupied forever
        cfg=GuardConfig(safety_margin_mb=500.0, hold_window_s=30.0, backoff_s=(5.0, 10.0)),
        sleep=ft.sleep, clock=ft.clock, log=logs.append, job_id="job-stuck",
    )
    assert d.launch is False
    assert "after 30" in d.reason
    assert d.free_mb == 1000.0
    assert any("VRAM-RELEASE" in m for m in logs)
    # never slept past the window
    assert sum(ft.sleeps) <= 30.0


def test_vram_backoff_clamped_to_window():
    ft = FakeTime()
    d = check_vram_before_launch(
        8000.0, lambda: 100.0,
        cfg=GuardConfig(hold_window_s=12.0, backoff_s=(5.0, 60.0)),
        sleep=ft.sleep, clock=ft.clock,
    )
    assert d.launch is False
    # first backoff 5s (t=5), second clamped to remaining 7s (t=12), then window spent
    assert ft.sleeps == [5.0, 7.0]


def test_vram_unqueryable_fails_open():
    ft = FakeTime()
    logs: list[str] = []
    d = check_vram_before_launch(
        8000.0, lambda: None,  # no nvml / nvidia-smi
        sleep=ft.sleep, clock=ft.clock, log=logs.append, job_id="job-blind",
    )
    assert d.launch is True  # never worse than the pre-guard blind launch
    assert d.free_mb is None
    assert any("VRAM-UNQUERYABLE" in m for m in logs)


# ---------------------------------------------------------------- GAP 2: payload


def test_payload_succeeds_first_try():
    ft = FakeTime()
    body = fetch_payload_with_retry(
        lambda: (200, b"PAYLOAD"), sleep=ft.sleep, clock=ft.clock,
    )
    assert body == b"PAYLOAD"
    assert ft.sleeps == []


def test_payload_retries_then_succeeds():
    # 404 twice (PUT not landed), then 200 — the instant-dispatch race.
    seq = iter([(404, b""), (404, b""), (200, b"OK")])
    ft = FakeTime()
    logs: list[str] = []
    body = fetch_payload_with_retry(
        lambda: next(seq),
        cfg=PayloadRetryConfig(attempts=6, window_s=30.0),
        sleep=ft.sleep, clock=ft.clock, log=logs.append, job_id="job-race",
    )
    assert body == b"OK"
    assert len(ft.sleeps) == 2  # two backoff waits before success
    assert any("retry 1/6" in m for m in logs)


def test_payload_retries_transport_exception_then_succeeds():
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("scheduler disconnected")
        return (200, b"OK")

    ft = FakeTime()
    logs: list[str] = []
    body = fetch_payload_with_retry(fetch, sleep=ft.sleep, clock=ft.clock, log=logs.append)

    assert body == b"OK"
    assert ft.sleeps == [1.0]
    assert any("transport ConnectionError" in m for m in logs)


def test_payload_exhausts_and_raises():
    ft = FakeTime()
    with pytest.raises(RuntimeError) as ei:
        fetch_payload_with_retry(
            lambda: (404, b""),
            cfg=PayloadRetryConfig(attempts=4, window_s=100.0),
            sleep=ft.sleep, clock=ft.clock,
        )
    msg = str(ei.value)
    assert "payload download failed: HTTP 404" in msg  # legacy prefix preserved
    assert "after 4 attempts" in msg


def test_payload_window_bounds_attempts():
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return (503, b"")

    ft = FakeTime()
    with pytest.raises(RuntimeError):
        fetch_payload_with_retry(
            fetch,
            cfg=PayloadRetryConfig(attempts=100, window_s=6.0, base_delay_s=1.0, max_delay_s=8.0),
            sleep=ft.sleep, clock=ft.clock,
        )
    # window 6s with delays 1,2,4 -> stops well before 100 attempts
    assert calls["n"] < 100
    assert sum(ft.sleeps) <= 6.0


def test_payload_non_retryable_fails_fast():
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return (403, b"forbidden")

    ft = FakeTime()
    with pytest.raises(RuntimeError) as ei:
        fetch_payload_with_retry(fetch, sleep=ft.sleep, clock=ft.clock)
    assert "HTTP 403" in str(ei.value)
    assert calls["n"] == 1  # a real 4xx is not the race — no retry
    assert ft.sleeps == []
