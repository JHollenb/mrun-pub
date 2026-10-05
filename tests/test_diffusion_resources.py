"""Regression tests for explicit FLUX diffusion reservations."""

from __future__ import annotations

import pytest

from mrun.diffusion import (
    DiffusionResourceEstimate,
    estimate_diffusion_resources,
    estimate_flux_resources,
)


def test_flux_resource_aliases_and_task_scaling() -> None:
    forward = estimate_flux_resources("FLUX.2-klein-4B", task="forward")
    measured = estimate_flux_resources("flux2-klein-4b", task="measure", capture_sites=True)
    train = estimate_flux_resources("flux2-klein-4b-distilled", task="train")

    assert forward.model_id == "flux2-klein-4b"
    assert measured.ram_mb > forward.ram_mb
    assert train.ram_mb > measured.ram_mb
    assert train.vram_mb == forward.vram_mb
    assert train.reservation()["source"] == "mrun diffusion FLUX component/offload envelope"
    assert train.as_plan(dtype="bfloat16", device="cuda")["backend"] == "diffusers"
    assert train.as_plan(dtype="bfloat16", device="cuda")["threads"] == 8


def test_flux_resource_scales_with_resolution() -> None:
    small = estimate_flux_resources("flux1-schnell", height=512, width=512)
    large = estimate_flux_resources("flux1-schnell", height=1024, width=1024)

    assert large.ram_mb > small.ram_mb
    assert large.vram_mb > small.vram_mb


def test_resident_klein_profile_uses_fast_worker_envelope() -> None:
    resident = estimate_flux_resources(
        "flux2-klein-4b", height=1024, width=1024, steps=8, phase_cuda=True, resident=True
    )

    assert resident.ram_mb == 20_000
    assert resident.vram_mb == 14_000


def test_nonresident_profile_keeps_offload_envelope() -> None:
    ordinary = estimate_flux_resources("flux2-klein-4b", height=1024, width=1024)

    assert ordinary.ram_mb == 41_600
    assert ordinary.ram_mb > 20_000


def test_unknown_flux_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown FLUX resource profile"):
        estimate_flux_resources("not-a-flux-model")


def test_sdxl_and_krea_resource_profiles_are_explicit() -> None:
    illustrious = estimate_flux_resources(
        "John6666/illustrious-xl10-improved-uncensored-v30-sdxl",
        height=1024,
        width=1024,
        steps=28,
        phase_cuda=True,
        resident=True,
    )
    pony = estimate_flux_resources("Runware/Pony_Diffusion_V6_XL", steps=25)
    pony_resident = estimate_flux_resources(
        "Runware/Pony_Diffusion_V6_XL", height=1024, width=1024, steps=25, resident=True
    )
    krea = estimate_flux_resources("krea/Krea-2-Turbo", height=1024, width=1024, steps=8)
    krea_resident = estimate_flux_resources(
        "krea/Krea-2-Turbo", height=1024, width=1024, steps=8, phase_cuda=True, resident=True
    )

    assert pony.model_id == "pony-xl-v6"
    assert pony.vram_mb == 14_000
    assert pony_resident.ram_mb == 24_000
    assert pony_resident.vram_mb == 14_000
    assert pony_resident.disk_gb == 1
    assert illustrious.model_id == "illustrious-xl"
    assert illustrious.ram_mb == 24_000
    assert illustrious.vram_mb == 14_000
    assert illustrious.disk_gb == 1
    assert krea.model_id == "krea2-turbo"
    assert krea.vram_mb > 16_000
    assert krea_resident.ram_mb == 36_000
    assert krea_resident.vram_mb == 12_000

    chroma = estimate_flux_resources(
        "lodestones/Chroma1-HD", height=1024, width=1024, steps=40, phase_cuda=True, resident=True
    )
    wai = estimate_flux_resources(
        "John6666/wai-nsfw-illustrious-sdxl-v150-sdxl",
        height=1024,
        width=1024,
        steps=28,
        phase_cuda=True,
        resident=True,
    )
    assert chroma.model_id == "chroma1-hd"
    assert chroma.ram_mb == 32_000
    assert chroma.vram_mb == 8_000
    assert chroma.disk_gb == 1
    assert wai.model_id == "wai-nsfw-illustrious-v150"
    assert wai.ram_mb == 24_000
    assert wai.vram_mb == 14_000
    assert wai.disk_gb == 1


def test_family_neutral_estimator_is_the_same_central_contract() -> None:
    generic = estimate_diffusion_resources(
        "krea/Krea-2-Turbo",
        height=512,
        width=512,
        steps=4,
        phase_cuda=True,
        capture_sites=True,
        resident=True,
    )
    legacy = estimate_flux_resources(
        "krea/Krea-2-Turbo",
        height=512,
        width=512,
        steps=4,
        phase_cuda=True,
        capture_sites=True,
        resident=True,
    )

    assert generic == legacy
    assert generic.model_id == "krea2-turbo"
    assert (generic.ram_mb, generic.vram_mb) == (36_000, 12_000)


@pytest.mark.parametrize(
    ("model_id", "ram_mb", "vram_mb", "profile_id"),
    [
        (
            "John6666/illustrious-xl10-improved-uncensored-v30-sdxl",
            11_776,
            8_192,
            "finite-restart-proof/illustrious-xl-v2",
        ),
        (
            "Runware/Pony_Diffusion_V6_XL",
            16_384,
            8_192,
            "finite-restart-proof/pony-xl-v1",
        ),
        (
            "John6666/wai-nsfw-illustrious-sdxl-v150-sdxl",
            11_776,
            8_192,
            "finite-restart-proof/wai-nsfw-illustrious-v150-v2",
        ),
        ("krea/Krea-2-Turbo", 26_624, 3_584, "finite-restart-proof/krea2-turbo-v2"),
        ("lodestones/Chroma1-HD", 32_512, 3_584, "finite-restart-proof/chroma1-hd-v2"),
    ],
)
def test_finite_restart_proof_uses_model_specific_sequential_profiles(
    model_id: str, ram_mb: int, vram_mb: int, profile_id: str
) -> None:
    estimate = estimate_diffusion_resources(
        model_id,
        task="forward",
        height=512,
        width=512,
        steps=4,
        phase_cuda=True,
        capture_sites=False,
        resident=False,
        workload_kind="finite_restart_proof",
        offload_strategy="sequential_cpu",
        batch_size=1,
        child_processes=2,
    )

    assert isinstance(estimate, DiffusionResourceEstimate)
    assert (estimate.ram_mb, estimate.vram_mb) == (ram_mb, vram_mb)
    assert (estimate.cpu_threads, estimate.disk_gb) == (4, 1)
    assert estimate.workload_kind == "finite_restart_proof"
    assert estimate.offload_strategy == "sequential_cpu"
    assert (estimate.batch_size, estimate.child_processes) == (1, 2)
    assert estimate.profile_id == profile_id
    assert estimate.history_key == (
        f"model:{estimate.model_id}:finite_restart_proof:sequential_cpu:512x512:b1:s4:c2"
    )
    assert estimate.metadata()["ram_kill_ceiling_mb"] == ram_mb * 1.1
    assert estimate.metadata()["vram_kill_ceiling_mb"] == vram_mb * 1.1
    reservation = estimate.reservation()
    assert reservation == {
        "ram_mb": ram_mb,
        "vram_mb": vram_mb,
        "cpu_threads": 4,
        "disk_gb": 1,
        "source": estimate.basis,
    }


def test_finite_restart_proof_metadata_is_stable_and_distinct_from_resident() -> None:
    kwargs = {
        "height": 512,
        "width": 512,
        "steps": 4,
        "phase_cuda": True,
        "workload_kind": "finite_restart_proof",
        "offload_strategy": "sequential_cpu",
        "batch_size": 1,
        "child_processes": 2,
    }
    first = estimate_diffusion_resources("illustrious-xl", **kwargs)
    second = estimate_diffusion_resources("illustrious-xl", **kwargs)
    resident = estimate_diffusion_resources(
        "illustrious-xl", height=512, width=512, steps=4, resident=True
    )

    assert first.as_dict() == second.as_dict()
    assert first.metadata() == second.metadata()
    assert first.ram_mb < resident.ram_mb
    assert first.vram_mb < resident.vram_mb
    assert first.history_key != "model:illustrious-xl:diffusion"


def test_finite_restart_proof_v2_profiles_cover_accepted_and_historical_peaks() -> None:
    def estimate(model_id: str) -> DiffusionResourceEstimate:
        result = estimate_diffusion_resources(
            model_id,
            task="forward",
            height=512,
            width=512,
            steps=4,
            phase_cuda=True,
            capture_sites=False,
            resident=False,
            workload_kind="finite_restart_proof",
            offload_strategy="sequential_cpu",
            batch_size=1,
            child_processes=2,
        )
        assert isinstance(result, DiffusionResourceEstimate)
        return result

    krea = estimate("krea/Krea-2-Turbo")
    chroma = estimate("lodestones/Chroma1-HD")

    # Keep these acceptance receipts as lower bounds on future profile edits.
    assert krea.metadata()["ram_kill_ceiling_mb"] >= 23_424
    assert krea.metadata()["vram_kill_ceiling_mb"] >= 1_714
    assert chroma.metadata()["ram_kill_ceiling_mb"] >= 19_602.8
    assert chroma.metadata()["vram_kill_ceiling_mb"] >= 838

    # A single low Chroma receipt does not supersede the prior matching
    # family/QStore high-water observation; retain no-kill coverage for it.
    assert chroma.metadata()["ram_kill_ceiling_mb"] >= 28.73 * 1024
    assert chroma.metadata()["vram_kill_ceiling_mb"] >= 2.88 * 1024


@pytest.mark.parametrize(
    "overrides",
    [
        {"resident": True},
        {"phase_cuda": False},
        {"capture_sites": True},
        {"offload_strategy": "model_cpu"},
        {"batch_size": 2},
        {"child_processes": 1},
        {"height": 1024},
        {"steps": 8},
    ],
)
def test_finite_restart_proof_rejects_unmeasured_geometry_or_execution(
    overrides: dict[str, object],
) -> None:
    kwargs: dict[str, object] = {
        "height": 512,
        "width": 512,
        "steps": 4,
        "phase_cuda": True,
        "capture_sites": False,
        "resident": False,
        "workload_kind": "finite_restart_proof",
        "offload_strategy": "sequential_cpu",
        "batch_size": 1,
        "child_processes": 2,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match="finite_restart_proof"):
        estimate_diffusion_resources("illustrious-xl", **kwargs)


def test_family_neutral_standard_path_preserves_flux_result_type() -> None:
    estimate = estimate_diffusion_resources("flux2-klein-4b", height=512, width=512)
    legacy = estimate_flux_resources("flux2-klein-4b", height=512, width=512)

    assert type(estimate) is type(legacy) is not DiffusionResourceEstimate
    assert estimate == legacy


def test_finite_fields_require_explicit_workload_kind() -> None:
    with pytest.raises(ValueError, match="require workload_kind"):
        estimate_diffusion_resources(
            "illustrious-xl",
            offload_strategy="sequential_cpu",
            batch_size=1,
            child_processes=2,
        )
