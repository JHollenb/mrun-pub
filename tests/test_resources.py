import pytest

from mrun import resources
from mrun.resources import ResourceMonitor, require_free_space


def test_free_space_guard_raises_when_below_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "disk_free_bytes", lambda _path: 10)
    with pytest.raises(RuntimeError, match="below required"):
        require_free_space(tmp_path, min_free_bytes=11)


def test_resource_monitor_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "hf_cache_bytes", lambda: 123)
    with ResourceMonitor(disk_path=tmp_path, output_paths=[tmp_path]) as monitor:
        (tmp_path / "out.txt").write_text("abc", encoding="utf-8")

    payload = monitor.metrics.as_dict()
    assert set(payload) == {
        "wall_s",
        "cpu_s",
        "rss_peak_mb",
        "rss_current_mb",
        "disk_free_before_bytes",
        "disk_free_after_bytes",
        "output_bytes",
        "hf_cache_bytes",
    }
    assert payload["output_bytes"] >= 3
    assert payload["hf_cache_bytes"] == 123
