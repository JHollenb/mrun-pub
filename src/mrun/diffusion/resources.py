"""Resource contracts for local Diffusers jobs.

The language-model estimator in :mod:`mrun.estimate` counts one model's
weights.  A Diffusers FLUX pipeline is a multi-component workload: text
encoder, transformer, VAE, offload staging, and (for measurements) retained
activations all contribute to the process envelope.  These values are the
conservative, already-used Atlas reservations expressed as a small mrun
contract so submitters do not hand-copy them.

This module is deliberately torch/diffusers-free.  It is safe to import in the
scheduler, submitter, and agent environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..protocol import RAM_KILL_FACTOR, kill_ceiling_mb


@dataclass(frozen=True, slots=True)
class FluxResourceEstimate:
    """Reservation and geometry for one decomposed FLUX workload."""

    model_id: str
    task: str
    ram_mb: int
    vram_mb: int
    cpu_threads: int
    disk_gb: int
    height: int
    width: int
    steps: int
    phase_cuda: bool
    capture_sites: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def reservation(self) -> dict[str, object]:
        return {
            "ram_mb": self.ram_mb,
            "vram_mb": self.vram_mb,
            "cpu_threads": self.cpu_threads,
            "disk_gb": self.disk_gb,
            "source": "mrun diffusion FLUX component/offload envelope",
        }

    def as_plan(self, *, dtype: str, device: str) -> dict[str, object]:
        """Return a worker-local plan-shaped record for FLUX telemetry.

        This has the familiar ``RunPlan`` fields so result consumers can compare
        diffusion and language-model runs, but it is not fed to the language-model
        engine.  Scheduler admission uses :meth:`reservation`.
        """
        return {
            "model": self.model_id,
            "backend": "diffusers",
            "dtype": str(dtype),
            "device": str(device),
            "max_batch": 1,
            "threads": self.cpu_threads,
            "ram_limit_mb": float(self.ram_mb),
            "est_ram_mb": float(self.ram_mb),
            "est_vram_mb": float(self.vram_mb),
            "est_wall_s": None,
            "weights_gb": 0.0,
            "engine_options": {
                "flux_task": self.task,
                "height": self.height,
                "width": self.width,
                "steps": self.steps,
                "phase_cuda": self.phase_cuda,
                "capture_sites": self.capture_sites,
            },
            "reasons": [
                "mrun.diffusion.resources measured FLUX component/offload envelope",
                "scheduler reservation is explicit; no language-model RunPlan applied",
            ],
            "reservation": self.reservation(),
        }


@dataclass(frozen=True, slots=True)
class DiffusionResourceEstimate(FluxResourceEstimate):
    """Resource estimate with an explicit family-neutral workload identity.

    The legacy :class:`FluxResourceEstimate` is intentionally left unchanged:
    callers of ``estimate_flux_resources`` continue to receive the same result
    type and reservation contract.  The family-neutral estimator returns this
    subtype only for a workload lane that has its own measured envelope.
    """

    workload_kind: str
    offload_strategy: str
    batch_size: int
    child_processes: int
    profile_id: str
    basis: str
    history_key: str

    def metadata(self) -> dict[str, Any]:
        """Return auditable workload/profile metadata for a scheduler request."""

        return {
            "workload_kind": self.workload_kind,
            "offload_strategy": self.offload_strategy,
            "batch_size": self.batch_size,
            "child_processes": self.child_processes,
            "profile_id": self.profile_id,
            "basis": self.basis,
            "history_key": self.history_key,
            "ram_kill_ceiling_mb": kill_ceiling_mb(self.ram_mb),
            "vram_kill_ceiling_mb": float(self.vram_mb) * RAM_KILL_FACTOR,
        }

    def reservation(self) -> dict[str, object]:
        """Return scheduler fields with the finite-profile basis as its source."""

        # ``@dataclass(slots=True)`` recreates the class object, so explicit
        # qualification keeps this inherited call valid on Python 3.10+.
        reservation = FluxResourceEstimate.reservation(self)
        reservation["source"] = self.basis
        return reservation


# These are the same model-specific envelopes used by the Atlas submission
# path.  They describe process residency, not parameter count; the scheduler
# must not infer a smaller reservation from the denoiser's nominal size.
_BASE: dict[str, tuple[int, int, int]] = {
    "flux1-schnell": (44_000, 13_000, 5),
    "flux2-klein-4b": (32_000, 13_000, 22),
    "flux2-klein-4b-distilled": (32_000, 13_000, 22),
    "flux2-klein-9b": (44_000, 13_000, 42),
    "flux2-klein-9b-kv": (44_000, 13_000, 42),
    "flux2-dev": (52_000, 13_000, 125),
    # SDXL's resident UNet/VAE envelope measured on the Beast 16 GB card.
    "illustrious-xl": (44_000, 14_000, 25),
    "pony-xl-v6": (44_000, 14_000, 25),
    # Krea 2 Turbo is a 12B BF16 pipeline.  The ordinary envelope is the
    # resident/full-component estimate; the resident worker uses the explicit
    # sequential component-offload cap below so the model can page submodules
    # through CUDA while retaining the checkpoint in host RAM.
    "krea2-turbo": (58_000, 28_000, 50),
    # Chroma1-HD is an 8.9B T5 + DiT Diffusers pipeline.  Its ordinary
    # component estimate is intentionally conservative; the resident worker
    # uses sequential component offload on the 16 GB Beast card.
    "chroma1-hd": (48_000, 28_000, 48),
    # WAI-NSFW Illustrious v1.5 is a conventional SDXL export.  Keep its
    # physical generation scalar on Beast even though logical sibling runs are
    # still supported by the viewer.
    "wai-nsfw-illustrious-v150": (24_000, 14_000, 8),
}

# Resident Saturn workers keep the denoiser and VAE on the CUDA device and use
# low-CPU-memory loading.  The generic Diffusers envelope above is intentionally
# conservative for one-shot/offload jobs; applying it to a resident worker
# strands otherwise-admissible work behind a needless 1.1x scheduler ceiling.
# This cap is bounded by the measured Beast worker envelope and can still be
# raised by mrun's exact-history calibration if a future run actually needs it.
# The first 1024px uncensored-Klein resident smoke reached 16.28 GiB RSS and
# was then killed at the 17.6 GiB cgroup ceiling because uv/package page cache
# is charged separately from the process RSS telemetry.  Keep the resident
# profile small enough to share the Beast scheduler, but leave measured
# headroom for the community Qwen encoder and the upload/idle loop.
RESIDENT_KLEIN_4B_RAM_MB = 20_000
RESIDENT_KLEIN_4B_VRAM_MB = 14_000
RESIDENT_ILLUSTRIOUS_XL_RAM_MB = 24_000
RESIDENT_ILLUSTRIOUS_XL_VRAM_MB = 14_000
RESIDENT_PONY_XL_RAM_MB = 24_000
RESIDENT_PONY_XL_VRAM_MB = 14_000
# Calibrated against resident-worker history on Beast.  Krea's observed p95
# was 35,535.8 MB RSS / 10,074 MB VRAM; 36/12 GB leaves mrun's 1.1x guard at
# 39.6/13.2 GB.  Keep the reservation tied to measured residency rather than
# the much larger one-shot component estimate above.
RESIDENT_KREA2_TURBO_RAM_MB = 36_000
RESIDENT_KREA2_TURBO_VRAM_MB = 12_000
# Chroma's observed resident history reached 28,725.2 MB RSS / 2,882 MB VRAM
# at 1024px.  The 32/8 GB profile leaves 35.2/8.8 GB after mrun's 1.1x guard
# for geometry not represented in the small history sample.
RESIDENT_CHROMA1_HD_RAM_MB = 32_000
RESIDENT_CHROMA1_HD_VRAM_MB = 8_000
RESIDENT_WAI_NSFW_ILLUSTRIOUS_RAM_MB = 24_000
RESIDENT_WAI_NSFW_ILLUSTRIOUS_VRAM_MB = 14_000
# Resident Diffusers workers use an already-staged local checkpoint and run
# offline; uv/Torch caches also live outside the job workspace. Three measured
# Beast resident payloads occupied 18--24 MB, while generated images stream to
# object storage. One GB therefore leaves more than 40x measured scratch
# headroom without needlessly blocking a staged model on the models mount.
RESIDENT_DIFFUSION_SCRATCH_DISK_GB = 1

# Finite restart proofs are not resident Diffusion View workers.  They run one
# 512px, batch-1, four-step specimen through sequential CPU offload, with two
# fresh child processes executed serially under one outer lease.  Keep this
# lane's measured envelopes separate from the resident profiles above so a
# short proof does not inherit the resident worker's queue-blocking envelope.
FINITE_RESTART_PROOF_WORKLOAD_KIND = "finite_restart_proof"
FINITE_RESTART_PROOF_OFFLOAD_STRATEGY = "sequential_cpu"
FINITE_RESTART_PROOF_BATCH_SIZE = 1
FINITE_RESTART_PROOF_CHILD_PROCESSES = 2
FINITE_RESTART_PROOF_WIDTH = 512
FINITE_RESTART_PROOF_HEIGHT = 512
FINITE_RESTART_PROOF_STEPS = 4
_FINITE_RESTART_PROOF_BASIS = (
    "approved finite sequential restart-proof family-calibrated envelope; mrun 1.1 guard, "
    "1s-sampled/page-cache headroom, and two sequential child loads; distinct "
    "from resident-worker history"
)
_FINITE_RESTART_PROOF_PROFILES: dict[str, tuple[str, int, int]] = {
    # Two real sequential-offload proofs peaked at 9,445.6 and 10,708 MB RSS.
    # 11,776 MB is the next 512-MB unit above 1.2x the newer measurement and
    # still leaves mrun's 1.1 kill ceiling above both observations. Keep the
    # VRAM reservation until interval-level CUDA telemetry replaces the
    # current 1-second 516-632 MB samples.
    "illustrious-xl": ("finite-restart-proof/illustrious-xl-v2", 11_776, 8_192),
    "pony-xl-v6": ("finite-restart-proof/pony-xl-v1", 16_384, 8_192),
    # WAI is the same conventional SDXL execution shape as Illustrious. Reuse
    # the accepted 11,776-MB envelope; Pony's observed 15,333.1-MB peak is
    # model-specific and must not inflate every SDXL-family proof.
    "wai-nsfw-illustrious-v150": (
        "finite-restart-proof/wai-nsfw-illustrious-v150-v2",
        11_776,
        8_192,
    ),
    # The proof children are sequential, not concurrent. These v2 QStore
    # envelopes retain roughly 25% guarded headroom over the worst matching
    # mrun samples without carrying native/compare-lane peaks into this lane.
    # The accepted Krea receipt job-9f1e8a3e7dd0 measured 23,424 MB RSS /
    # 1,714 MB VRAM / 275.15 s under this profile.  The accepted Chroma
    # receipt job-bacf0e455296 measured 19,602.8 MB RSS / 838 MB VRAM /
    # 310.84 s (including 47.71 s of uv environment setup).  Chroma's prior
    # matching family/QStore history reached about 28.73 GiB RSS / 2.88 GiB
    # VRAM, so one lower receipt is retained as a trend and does not replace
    # that high-water envelope until repeated exact-key cold receipts do.
    "krea2-turbo": ("finite-restart-proof/krea2-turbo-v2", 26_624, 3_584),
    "chroma1-hd": ("finite-restart-proof/chroma1-hd-v2", 32_512, 3_584),
}


def _normalize_model_id(model_id: str) -> str:
    value = str(model_id).strip().lower()
    aliases = {
        "flux.1-schnell": "flux1-schnell",
        "flux.2-klein-4b": "flux2-klein-4b",
        "flux.2-klein-9b": "flux2-klein-9b",
        "flux.2-klein-9b-kv": "flux2-klein-9b-kv",
        "flux.2-dev": "flux2-dev",
        "black-forest-labs/flux.1-schnell": "flux1-schnell",
        "black-forest-labs/flux.2-klein-4b": "flux2-klein-4b",
        "black-forest-labs/flux.2-klein-9b": "flux2-klein-9b",
        "black-forest-labs/flux.2-klein-9b-kv": "flux2-klein-9b-kv",
        "black-forest-labs/flux.2-dev": "flux2-dev",
        "john6666/illustrious-xl10-improved-uncensored-v30-sdxl": "illustrious-xl",
        "illustrious-xl": "illustrious-xl",
        "runware/pony_diffusion_v6_xl": "pony-xl-v6",
        "pony-xl-v6": "pony-xl-v6",
        "krea/krea-2-turbo": "krea2-turbo",
        "krea2-turbo": "krea2-turbo",
        "lodestones/chroma1-hd": "chroma1-hd",
        "chroma1-hd": "chroma1-hd",
        "john6666/wai-nsfw-illustrious-sdxl-v150-sdxl": "wai-nsfw-illustrious-v150",
        "wai-nsfw-illustrious-v150": "wai-nsfw-illustrious-v150",
    }
    return aliases.get(value, value)


def estimate_flux_resources(
    model_id: str,
    *,
    task: str = "forward",
    height: int = 512,
    width: int = 512,
    steps: int = 4,
    phase_cuda: bool = False,
    capture_sites: bool = False,
    resident: bool = False,
) -> FluxResourceEstimate:
    """Return a conservative mrun reservation for one FLUX job.

    Resolution scales activation headroom, while capture and training add
    explicit process slack.  The returned values are admission inputs, not a
    performance prediction; wall time is learned from the exact mrun
    experiment/config identity after the first run.
    """

    normalized = _normalize_model_id(model_id)
    try:
        base_ram, base_vram, disk = _BASE[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown FLUX resource profile {model_id!r}; expected one of {sorted(_BASE)}"
        ) from exc
    if isinstance(height, bool) or isinstance(width, bool) or height < 64 or width < 64:
        raise ValueError("FLUX height and width must be integers >= 64")
    if height % 8 or width % 8:
        raise ValueError("FLUX height and width must be divisible by 8")
    if isinstance(steps, bool) or steps <= 0:
        raise ValueError("FLUX steps must be a positive integer")
    task_name = str(task).strip().lower()
    if task_name in {"training", "backward"}:
        task_name = "train"
    if task_name not in {"forward", "generate", "measure", "capture", "train"}:
        raise ValueError(f"unsupported FLUX task {task!r}")

    pixel_scale = max(1.0, (height * width) / float(512 * 512))
    ram = float(base_ram) * (1.0 + 0.10 * (pixel_scale - 1.0))
    if capture_sites or task_name in {"measure", "capture"}:
        ram *= 1.20
    if task_name == "train":
        ram *= 1.35
    vram = float(base_vram) * (1.0 + 0.05 * (pixel_scale - 1.0))
    if resident and normalized in {"flux2-klein-4b", "flux2-klein-4b-distilled"}:
        ram = min(ram, float(RESIDENT_KLEIN_4B_RAM_MB))
        vram = min(vram, float(RESIDENT_KLEIN_4B_VRAM_MB))
    if resident and normalized == "illustrious-xl":
        ram = min(ram, float(RESIDENT_ILLUSTRIOUS_XL_RAM_MB))
        vram = min(vram, float(RESIDENT_ILLUSTRIOUS_XL_VRAM_MB))
    if resident and normalized == "pony-xl-v6":
        ram = min(ram, float(RESIDENT_PONY_XL_RAM_MB))
        vram = min(vram, float(RESIDENT_PONY_XL_VRAM_MB))
    if resident and normalized == "krea2-turbo":
        # Krea's worker enables Diffusers sequential CPU offload.  This is an
        # image-pipeline component pager, not the language-model QStore
        # `paged` backend, so reserve host RAM for the full checkpoint and only
        # the measured/provisional CUDA working envelope.
        ram = min(ram, float(RESIDENT_KREA2_TURBO_RAM_MB))
        vram = min(vram, float(RESIDENT_KREA2_TURBO_VRAM_MB))
    if resident and normalized == "chroma1-hd":
        # Chroma's T5 encoder and DiT are also loaded through the worker's
        # sequential component-offload contract; do not pass its ordinary
        # full-component VRAM estimate to Beast's 16 GB admission guard.
        ram = min(ram, float(RESIDENT_CHROMA1_HD_RAM_MB))
        vram = min(vram, float(RESIDENT_CHROMA1_HD_VRAM_MB))
    if resident and normalized == "wai-nsfw-illustrious-v150":
        ram = min(ram, float(RESIDENT_WAI_NSFW_ILLUSTRIOUS_RAM_MB))
        vram = min(vram, float(RESIDENT_WAI_NSFW_ILLUSTRIOUS_VRAM_MB))
    if resident:
        # The worker's model_path is a verified local Beast path and its Hub
        # environment is offline.  Only uv/runtime staging and output scratch
        # belong in this reservation; model bytes are not downloaded by mrun.
        disk = RESIDENT_DIFFUSION_SCRATCH_DISK_GB
    return FluxResourceEstimate(
        model_id=normalized,
        task=task_name,
        ram_mb=int(round(ram)),
        vram_mb=int(round(vram)),
        cpu_threads=8 if task_name != "evaluate" else 6,
        disk_gb=disk,
        height=int(height),
        width=int(width),
        steps=int(steps),
        phase_cuda=bool(phase_cuda),
        capture_sites=bool(capture_sites),
    )


def estimate_diffusion_resources(
    model_id: str,
    *,
    task: str = "forward",
    height: int = 512,
    width: int = 512,
    steps: int = 4,
    phase_cuda: bool = False,
    capture_sites: bool = False,
    resident: bool = False,
    workload_kind: str | None = None,
    offload_strategy: str | None = None,
    batch_size: int | None = None,
    child_processes: int | None = None,
) -> FluxResourceEstimate | DiffusionResourceEstimate:
    """Return the centralized resource estimate for any registered diffusion family.

    ``estimate_flux_resources`` remains the backward-compatible implementation;
    this family-neutral entry point is the authority for new submitters.  The
    ``finite_restart_proof`` lane is deliberately explicit about its offload,
    physical batch, and fresh-child geometry because its reservation is not a
    resident-worker envelope.
    """

    if workload_kind is None:
        if any(
            value is not None
            for value in (offload_strategy, batch_size, child_processes)
        ):
            raise ValueError(
                "offload_strategy, batch_size, and child_processes require workload_kind"
            )
        return estimate_flux_resources(
            model_id,
            task=task,
            height=height,
            width=width,
            steps=steps,
            phase_cuda=phase_cuda,
            capture_sites=capture_sites,
            resident=resident,
        )

    normalized_kind = str(workload_kind).strip().lower().replace("-", "_")
    if normalized_kind != FINITE_RESTART_PROOF_WORKLOAD_KIND:
        raise ValueError(f"unsupported diffusion workload_kind {workload_kind!r}")
    if resident:
        raise ValueError("finite_restart_proof cannot use resident=True")
    task_name = str(task).strip().lower()
    if task_name not in {"forward", "generate"}:
        raise ValueError("finite_restart_proof requires task='forward' or task='generate'")
    if not phase_cuda:
        raise ValueError("finite_restart_proof requires phase_cuda=True")
    if capture_sites:
        raise ValueError(
            "finite_restart_proof does not cover retained capture sites; use a measured lane"
        )
    normalized_offload = str(offload_strategy or "").strip().lower().replace("-", "_")
    if normalized_offload != FINITE_RESTART_PROOF_OFFLOAD_STRATEGY:
        raise ValueError(
            "finite_restart_proof requires offload_strategy='sequential_cpu'"
        )
    if batch_size != FINITE_RESTART_PROOF_BATCH_SIZE:
        raise ValueError("finite_restart_proof supports only batch_size=1")
    if child_processes != FINITE_RESTART_PROOF_CHILD_PROCESSES:
        raise ValueError("finite_restart_proof supports only child_processes=2")
    if (
        type(height) is not int
        or type(width) is not int
        or height != FINITE_RESTART_PROOF_HEIGHT
        or width != FINITE_RESTART_PROOF_WIDTH
    ):
        raise ValueError("finite_restart_proof supports only 512x512 geometry")
    if type(steps) is not int or steps != FINITE_RESTART_PROOF_STEPS:
        raise ValueError("finite_restart_proof supports only steps=4")

    normalized_model = _normalize_model_id(model_id)
    try:
        profile_id, ram_mb, vram_mb = _FINITE_RESTART_PROOF_PROFILES[normalized_model]
    except KeyError as exc:
        raise ValueError(
            f"unknown finite restart-proof diffusion profile {model_id!r}; expected one of "
            f"{sorted(_FINITE_RESTART_PROOF_PROFILES)}"
        ) from exc
    history_key = (
        f"model:{normalized_model}:{FINITE_RESTART_PROOF_WORKLOAD_KIND}:"
        f"{normalized_offload}:{width}x{height}:b{batch_size}:s{steps}:c{child_processes}"
    )
    return DiffusionResourceEstimate(
        model_id=normalized_model,
        task=task_name,
        ram_mb=ram_mb,
        vram_mb=vram_mb,
        cpu_threads=4,
        disk_gb=1,
        height=height,
        width=width,
        steps=steps,
        phase_cuda=True,
        capture_sites=False,
        workload_kind=FINITE_RESTART_PROOF_WORKLOAD_KIND,
        offload_strategy=FINITE_RESTART_PROOF_OFFLOAD_STRATEGY,
        batch_size=FINITE_RESTART_PROOF_BATCH_SIZE,
        child_processes=FINITE_RESTART_PROOF_CHILD_PROCESSES,
        profile_id=profile_id,
        basis=_FINITE_RESTART_PROOF_BASIS,
        history_key=history_key,
    )


__all__ = [
    "DiffusionResourceEstimate",
    "FINITE_RESTART_PROOF_BATCH_SIZE",
    "FINITE_RESTART_PROOF_CHILD_PROCESSES",
    "FINITE_RESTART_PROOF_HEIGHT",
    "FINITE_RESTART_PROOF_OFFLOAD_STRATEGY",
    "FINITE_RESTART_PROOF_STEPS",
    "FINITE_RESTART_PROOF_WIDTH",
    "FINITE_RESTART_PROOF_WORKLOAD_KIND",
    "FluxResourceEstimate",
    "estimate_diffusion_resources",
    "estimate_flux_resources",
]
