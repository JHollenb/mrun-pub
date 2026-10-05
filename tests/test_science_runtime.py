from __future__ import annotations

import pytest

import mrun.engine.paged as paged_module
from mrun.client.submit import _envelope_plans
from mrun.engine import _open_engine_impl
from mrun.policy import HostCaps
from mrun.science.benchmark import _dtype_kwargs
from mrun.science.cli import _scheduler_request
from mrun.science.runtime import ScienceRuntimeError, resolve_runtime_execution


def _config(*, fabric: str = "auto", device: str = "auto", dtype: str = "auto") -> dict:
    return {
        "model": {"name": "toy", "path": None},
        "serve": {"type": "inference", "fabric": fabric, "device": device, "dtype": dtype},
    }


CUDA_HOST = HostCaps(
    name="beast", ram_mb=64_000, vram_mb=16_000, has_cuda=True, cpus=16
)
CPU_HOST = HostCaps(name="cpu", ram_mb=64_000, cpus=8)
METAL_HOST = HostCaps(name="mbp1", ram_mb=32_000, has_mps=True, has_ane=True, cpus=12)


def test_auto_hf_uses_cuda_bf16_on_cuda_host() -> None:
    result = resolve_runtime_execution(_config(), {"name": "hf", "backend": "hf"}, host=CUDA_HOST)
    assert result["fabric"] == "cuda"
    assert result["device"] == "cuda:0"
    assert result["dtype"] == "bf16"


def test_auto_hf_uses_cpu_fp32_on_cpu_host() -> None:
    result = resolve_runtime_execution(_config(), {"name": "hf", "backend": "hf"}, host=CPU_HOST)
    assert result["fabric"] == "cpu"
    assert result["device"] == "cpu"
    assert result["dtype"] == "fp32"


def test_metal_aliases_resolve_to_mps() -> None:
    result = resolve_runtime_execution(
        _config(fabric="metal"), {"name": "mlx", "backend": "mlx"}, host=METAL_HOST
    )
    assert result["fabric"] == "metal"
    assert result["device"] == "mps"


@pytest.mark.parametrize(
    ("backend", "dtype"),
    [("paged-fp16", "fp16"), ("paged-bf16", "bf16"), ("paged-fp32", "fp32")],
)
def test_paged_dtype_lanes_are_explicit(
    backend: str, dtype: str
) -> None:
    result = resolve_runtime_execution(
        _config(device="cpu"), {"name": backend, "backend": backend}, host=CPU_HOST
    )
    assert result["dtype"] == dtype


def test_runtime_options_override_top_level_target() -> None:
    result = resolve_runtime_execution(
        _config(fabric="cpu", device="cpu", dtype="fp32"),
        {
            "name": "dense",
            "backend": "dense-qstore-cuda",
            "options": {"fabric": "gpu", "device": "cuda:0", "compute_dtype": "fp16"},
        },
        host=CUDA_HOST,
    )
    assert result["device"] == "cuda:0"
    assert result["dtype"] == "fp16"


def test_multifabric_science_keeps_paged_child_on_cpu() -> None:
    config = _config(fabric="metal")
    runtime = {
        "name": "multi",
        "backend": "multifabric",
        "options": {"backends": ["paged", "mlx"]},
    }
    execution = resolve_runtime_execution(config, runtime, host=METAL_HOST)
    assert execution["fabric"] == "metal"
    assert execution["device"] == "mps"
    assert _dtype_kwargs(runtime, config["serve"], execution)["device"] == "cpu"


def test_auto_runtime_device_inherits_top_level_fabric() -> None:
    config = _config(fabric="metal")
    runtime = {
        "name": "multi",
        "backend": "multifabric",
        "options": {"device": "auto", "backends": ["paged", "mlx"]},
    }
    execution = resolve_runtime_execution(config, runtime, host=METAL_HOST)
    assert execution["fabric"] == "metal"
    assert execution["device"] == "mps"


def test_cuda_runtime_fails_without_cuda() -> None:
    with pytest.raises(ScienceRuntimeError, match="requires CUDA"):
        resolve_runtime_execution(
            _config(), {"name": "dense", "backend": "dense-qstore-cuda"}, host=CPU_HOST
        )


@pytest.mark.parametrize(
    ("backend", "expected_dtype"), [("paged-fp16", "fp16"), ("paged-bf16", "bf16")]
)
def test_paged_alias_opens_lossless_store_with_fixed_compute_dtype(
    monkeypatch: pytest.MonkeyPatch, backend: str, expected_dtype: str
) -> None:
    captured: dict[str, object] = {}

    class FakePaged:
        def __init__(self, model: str, **kwargs: object) -> None:
            captured["model"] = model
            captured.update(kwargs)

    monkeypatch.setattr(paged_module, "PagedEngine", FakePaged)
    _open_engine_impl("toy", backend=backend, device="cpu")
    assert captured["fp32"] is True
    assert captured["compute_dtype"] == expected_dtype


def test_paged_alias_rejects_conflicting_compute_dtype() -> None:
    with pytest.raises(ValueError, match="fixed to fp16"):
        _open_engine_impl("toy", backend="paged-fp16", compute_dtype="bf16")


def test_sequential_runtime_plan_envelope_uses_peak_memory_and_total_wall_time() -> None:
    envelope = _envelope_plans(
        [
            {
                "backend": "hf",
                "device": "cpu",
                "max_batch": 1,
                "threads": 4,
                "ram_limit_mb": 2_000,
                "est_ram_mb": 1_500,
                "est_vram_mb": 0,
                "est_wall_s": 3,
                "weights_gb": 8,
                "reasons": ["hf"],
            },
            {
                "backend": "dense-qstore-cuda",
                "device": "cuda",
                "max_batch": 8,
                "threads": 2,
                "ram_limit_mb": 4_000,
                "est_ram_mb": 3_000,
                "est_vram_mb": 12_000,
                "est_wall_s": 5,
                "weights_gb": 8,
                "reasons": ["cuda"],
            },
        ]
    )
    assert envelope["ram_limit_mb"] == 4_000
    assert envelope["est_vram_mb"] == 12_000
    assert envelope["threads"] == 4
    assert envelope["est_wall_s"] == 8
    assert envelope["device"] == "cuda"
    assert "science runtime envelope: hf, dense-qstore-cuda" in envelope["reasons"]


def test_science_scheduler_request_contains_every_runtime_plan() -> None:
    config = {
        "experiment": {"name": "matrix"},
        "model": {"name": "toy"},
        "runtimes": [
            {"name": "reference", "backend": "hf", "options": {}},
            {"name": "paged", "backend": "paged-fp32", "options": {}},
        ],
        "serve": {"type": "forward", "fabric": "cuda", "device": "auto", "dtype": "auto"},
        "execution": {"host": "beast", "prefer_host": None, "needs": {}},
        "test": {"batch_sizes": [1, 8], "context_tokens": [128]},
    }
    request = _scheduler_request(config)
    assert [item["backend"] for item in request["plan_variants"]] == ["hf", "paged-fp32"]
    assert request["needs"]["cuda"] is True
