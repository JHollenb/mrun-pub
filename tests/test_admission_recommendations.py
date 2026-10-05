from __future__ import annotations

from mrun.server.app import _reservation_recommendations


def test_reservation_recommendation_is_conservative_and_host_specific():
    host = {
        "name": "beast",
        "caps": {"cuda": True},
        "cpu_threads": 32,
        "ram_total_mb": 66508.0,
        "vram_total_mb": 16376.0,
        "limits": {"ram_margin_mb": 2048.0, "vram_margin_mb": 300.0},
    }
    job = {"needs": {"cuda": True}}

    recommendations = _reservation_recommendations(job, [host])

    assert recommendations[0]["host"] == "beast"
    reservation = recommendations[0]["reservation"]
    assert reservation["vram_mb"] == 14336.0
    assert reservation["ram_mb"] > 0
    assert reservation["cpu_threads"] == 32
