from __future__ import annotations

from mrun.engine.profiles import (
    engine_profiles,
    profile_for_backend,
    profile_id_for_backend,
)
from mrun.policy import HostCaps, plan_run


def test_builtin_profiles_keep_backend_aliases_explicit():
    assert profile_id_for_backend("moe-qstore-cuda") == "qwen3-moe-cuda"
    assert profile_for_backend("dense-cuda").backend == "dense-qstore-cuda"
    assert {profile.profile_id for profile in engine_profiles()} >= {
        "hf-reference",
        "paged-int8",
        "dense-qstore-cuda",
        "qwen3-moe-cuda",
    }


def test_profile_probe_is_fail_closed_for_wrong_host_or_artifact():
    profile = profile_for_backend("qwen3-moe-cuda")
    assert profile is not None
    result = profile.probe(
        {"artifact_kind": "expert-store"},
        {"caps": {"cuda": False}},
        {"device": "cuda", "dtype": "bf16"},
    )
    assert result["eligible"] is False
    assert any("CUDA" in reason for reason in result["reasons"])


def test_run_plan_can_bind_artifact_without_changing_legacy_selection():
    host = HostCaps(name="beast", ram_mb=61_000, vram_mb=16_376, has_cuda=True, cpus=32)
    plan = plan_run("qwen3-14b", host=host, backend="paged", device="cuda", dtype="bf16")
    bound = plan.bind_artifact(
        {
            "artifact_id": "artifact:fp8",
            "artifact_kind": "qstore",
            "bytes": 7_000_000_000,
            "mount": "/mnt/ssd1tb",
            "path": "/mnt/ssd1tb/qstores/qwen3-fp8",
        }
    )

    assert bound.backend == plan.backend == "paged"
    assert bound.engine_profile == "paged-int8"
    assert bound.artifact_id == "artifact:fp8"
    assert bound.artifact_mount == "/mnt/ssd1tb"
    assert bound.weights_gb == 7.0
    assert bound.engine_kwargs()["store_path"] == "/mnt/ssd1tb/qstores/qwen3-fp8"
