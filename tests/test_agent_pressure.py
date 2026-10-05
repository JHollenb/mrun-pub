from __future__ import annotations

from mrun.agent.main import _swap_growth_is_critical
from mrun.protocol import SWAP_SENTINEL_GROWTH_MB


def test_darwin_swap_growth_remains_a_hard_signal() -> None:
    assert _swap_growth_is_critical(
        system="Darwin",
        growth_mb=SWAP_SENTINEL_GROWTH_MB + 1,
        available_mb=12_000,
        total_mb=20_000,
    )


def test_linux_stale_swap_with_abundant_available_ram_is_not_critical() -> None:
    assert not _swap_growth_is_critical(
        system="Linux",
        growth_mb=SWAP_SENTINEL_GROWTH_MB + 2_000,
        available_mb=53_000,
        total_mb=64_000,
    )


def test_linux_swap_growth_with_low_available_ram_is_critical() -> None:
    assert _swap_growth_is_critical(
        system="Linux",
        growth_mb=SWAP_SENTINEL_GROWTH_MB + 1,
        available_mb=3_000,
        total_mb=64_000,
    )


def test_small_swap_growth_is_never_critical() -> None:
    assert not _swap_growth_is_critical(
        system="Darwin",
        growth_mb=SWAP_SENTINEL_GROWTH_MB,
        available_mb=1_000,
        total_mb=20_000,
    )


class _FakeRun:
    def __init__(self, job_id: str, rss_mb: float) -> None:
        import threading

        self.job_id = job_id
        self.cur_rss_mb = rss_mb
        self.pressure_kill = threading.Event()


def _agent_with_runs(monkeypatch, runs):
    from mrun.agent.config import AgentConfig
    from mrun.agent.main import Agent

    agent = Agent(AgentConfig(server_url="http://test", host="t"))
    agent.runs = {r.job_id: r for r in runs}
    agent._swap_floor_mb = 0.0

    class _Swap:
        used = (SWAP_SENTINEL_GROWTH_MB + 2000) * 1e6

    class _Vm:
        available = 1_000e6
        total = 20_000e6

    import mrun.agent.main as agent_main

    monkeypatch.setattr(agent_main, "platform", __import__("platform"))
    import psutil

    monkeypatch.setattr(psutil, "swap_memory", lambda: _Swap)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Vm)
    monkeypatch.setattr(agent_main, "mem_pressure_level", lambda: 4)
    monkeypatch.setattr(
        agent_main.platform, "system", lambda: "Darwin"
    )
    return agent


def test_sentinel_skips_implausible_victim(monkeypatch) -> None:
    # a 61MB watcher cannot be why the host is drowning — measured collateral
    # kill x2 (2026-07-30); pressure must be flagged out-of-band, nobody killed
    watcher = _FakeRun("job-watcher", 61.0)
    agent = _agent_with_runs(monkeypatch, [watcher])
    agent.pressure_watch()
    assert not watcher.pressure_kill.is_set()
    assert agent._pressure_external is True


def test_sentinel_kills_plausible_largest_victim(monkeypatch) -> None:
    watcher = _FakeRun("job-watcher", 61.0)
    hog = _FakeRun("job-hog", 9000.0)
    agent = _agent_with_runs(monkeypatch, [watcher, hog])
    agent.pressure_watch()
    assert hog.pressure_kill.is_set()
    assert not watcher.pressure_kill.is_set()
    assert agent._pressure_external is False
