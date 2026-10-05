"""Run policy — the single source of truth for backend/dtype/batch/thread decisions.

``plan_run`` turns (model, task, host capabilities) into an explicit, auditable ``RunPlan``:
which engine backend, which dtype, how big a batch, how many threads, and the RAM ceiling a
guard should enforce. The same function runs client-side (to estimate a reservation for
scheduler admission) and agent-side (to enforce it), so a developer agent cannot silently
pick fp32-on-a-16GB-host or forget that a 14B model is paged territory.

Encoded rules (each emits a ``reasons`` line):
- dtype: bf16 default on cuda hosts (parity-gated 2026-07-15; argmax 12/12, content-lift
  Δ0.03%) and on unified-memory Macs, EXCEPT ``task="train"`` which stays fp32 until the
  acquisition gauntlet gates bf16 gradients. ``MRUN_TORCH_DTYPE``/arg overrides; fp32+eager
  remains the parity oracle.
- cuda ⇒ TF32 forced off (TF32's ~1e-3 error breaches 4-decimal-rounded stats).
- backend ``hf`` default (batched, exact, canonical); a paged backend when the calibrated RAM
  law or the host VRAM guard says a resident model won't fit; ``mlx``/``ane`` are
  explicit-only and must be parity-gated before canonical use.
- ``max_batch`` via the activation-memory-capped ``resolve_max_batch``.
- ``ram_limit_mb`` = estimated RSS x safety factor — the direct local ``run_stage`` ceiling
  and the fleet reservation base. Fleet admission and its agent kill line apply the separate
  ``RAM_KILL_FACTOR`` margin to that reservation.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any

from .estimate import estimate_activation_mb, estimate_memory, task_rss_factor
from .guard import set_thread_env

DEFAULT_THREADS = 4
RAM_SAFETY_FACTOR = 1.3
# Fraction of host RAM a single run may plan to use (leaves room for OS + other work).
HOST_RAM_BUDGET_FRACTION = 0.7
# Fraction of VRAM a single run may plan to use (allocator fragmentation + cublas
# workspaces sit outside the weights+activations estimate).
VRAM_BUDGET_FRACTION = 0.85
_DENSE_CUDA_BACKENDS = {
    "dense-qstore-cuda",
    "dense-cuda",
}
_QWEN3_MOE_CUDA_BACKENDS = {
    "qwen3-moe-cuda",
    "moe-qstore-cuda",
}
_APPLE_METAL_BACKENDS = {"mlx", "mlx-q4"}
_APPLE_BACKENDS = _APPLE_METAL_BACKENDS | {"ane"}
# Measured Beast process envelope for the routed CUDA runtime: the 30B FP8 smoke
# peaked at 7.6--8.4GB RSS after loading the skeleton, tokenizer, page tier, and CUDA
# runtime. Keep the model bytes on the QStore tier, but reserve enough host runtime RAM
# that the scheduler does not kill a valid forward before the measured retry can learn.
QWEN3_MOE_CUDA_BASE_RAM_MB = 6_144.0
QWEN3_MOE_CUDA_DEFAULT_CACHE_MB = 7_100.0
QWEN3_MOE_CUDA_FIXED_MB = 3_188.0
QWEN3_MOE_FP8_EXPERT_PAGE_MB = 4.734976
QWEN3_MOE_W4_EXPERT_PAGE_MB = 2.506752
QWEN3_MOE_TOTAL_EXPERT_PAGES = 48 * 128
QWEN3_MOE_DEFAULT_MAX_ACTIVE_PAGES = 128
QWEN3_MOE_DEFAULT_EXPERT_CODEC = "fp8"
QWEN3_MOE_DEFAULT_PAGE_BINDING_POLICY = "slot-indirect-v1"
QWEN3_MOE_COMPACT_PAGE_BINDING_POLICY = "compact-copy-v1"
QWEN3_MOE_DEFAULT_W4_ARITHMETIC_POLICY = "w4-g128-predot-bf16-v1"
QWEN3_MOE_SUPPORTED_W4_ARITHMETIC_POLICIES = frozenset(
    {
        QWEN3_MOE_DEFAULT_W4_ARITHMETIC_POLICY,
        "w4-g128-postscale-bf16-v1",
    }
)
# 48 layers * (K + V) * 4 KV heads * 128 head dim * BF16.
QWEN3_MOE_KV_BYTES_PER_TOKEN_PER_STREAM = 48 * 2 * 4 * 128 * 2
# A generic CUDA-paged QStore streams one matrix at a time. Until an artifact-specific
# working-set estimate is bound, reserve at least 2 GiB for the dequantized matrix, CUDA ring,
# and logits, while also charging the activation envelope. This is deliberately much smaller
# than a resident model estimate and is refined by exact run history after the first execution.
PAGED_CUDA_MIN_WORKING_SET_MB = 2_048.0
# NVML charges the CUDA process/context above tensor bytes. Installed public
# Torch 2.13/CUDA 13 Qwen0.5 BF16 startup job-b8317b2a5809 reached 1,266MB
# against a 1,014.9MB tensor estimate before its first forward. Reserve 512MB
# (over twice that observed excess) before fit/admission; live guards and
# exact-history sizing still refine the workload-specific envelope.
CUDA_HF_PROCESS_OVERHEAD_MB = 512.0


def _auto_paged_backend(model: str, host: HostCaps) -> str:
    """Return the paged profile appropriate for a model family.

    Qwen3 MoE has a first-class routed expert pager; dense families use the ordinary QStore
    pager. Keep this decision in policy so ``backend=auto`` cannot accidentally send a model
    that exceeds the card through resident HF CUDA.
    """

    try:
        from .models import resolve_model

        family = resolve_model(model).family
    except Exception:  # noqa: BLE001 — unknown aliases keep the generic safe fallback
        family = ""
    if family == "qwen3_moe" and host.has_cuda:
        return "qwen3-moe-cuda"
    return "paged"


def _is_dense_cuda_backend(backend: str) -> bool:
    return str(backend).strip().lower().replace("_", "-") in _DENSE_CUDA_BACKENDS


def _is_qwen3_moe_cuda_backend(backend: str) -> bool:
    return str(backend).strip().lower().replace("_", "-") in _QWEN3_MOE_CUDA_BACKENDS


def _canonical_backend_name(backend: str) -> str:
    normalized = str(backend).strip().lower().replace("_", "-")
    if normalized in {"apple", "apple-speed"}:
        return "mlx"
    if normalized == "coreml":
        return "ane"
    if normalized in _QWEN3_MOE_CUDA_BACKENDS:
        return "qwen3-moe-cuda"
    return normalized


def _dense_cuda_engine_kwargs(dtype: str, device: str) -> dict[str, str]:
    """Translate a RunPlan's canonical dtype into the dense engine's public contract."""

    normalized_device = str(device).strip().lower()
    if not normalized_device.startswith("cuda"):
        raise ValueError(
            f"dense-qstore-cuda requires a CUDA device; the RunPlan selected {device!r}"
        )
    compute_dtype = {
        "bfloat16": "bf16",
        "bf16": "bf16",
        "float16": "fp16",
        "fp16": "fp16",
    }.get(str(dtype).strip().lower())
    if compute_dtype is None:
        raise ValueError(
            f"dense-qstore-cuda requires bfloat16 or float16; the RunPlan selected {dtype!r}"
        )
    return {
        "device": normalized_device,
        "compute_dtype": compute_dtype,
    }


def _qwen3_moe_cuda_engine_kwargs(
    dtype: str,
    device: str,
    engine_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_device = str(device).strip().lower()
    if not normalized_device.startswith("cuda"):
        raise ValueError(f"qwen3-moe-cuda requires a CUDA device; the RunPlan selected {device!r}")
    if str(dtype).strip().lower() not in {"bfloat16", "bf16"}:
        raise ValueError(
            f"qwen3-moe-cuda currently requires bfloat16; the RunPlan selected {dtype!r}"
        )
    options = dict(engine_options or {})
    reserved = {"device", "compute_dtype"}
    collisions = sorted(reserved & options.keys())
    if collisions:
        raise ValueError(f"qwen3-moe-cuda backend_options cannot override {collisions}")
    return {
        **options,
        "device": normalized_device,
        "compute_dtype": "bf16",
    }


def _qwen3_moe_cuda_resident_mb(
    engine_options: dict[str, Any] | None,
) -> tuple[float, float, float, int, str, float, int, float]:
    options = dict(engine_options or {})
    expert_codec = str(options.get("expert_codec", QWEN3_MOE_DEFAULT_EXPERT_CODEC)).strip().lower()
    page_mb_by_codec = {
        "fp8": QWEN3_MOE_FP8_EXPERT_PAGE_MB,
        "w4": QWEN3_MOE_W4_EXPERT_PAGE_MB,
    }
    if expert_codec not in page_mb_by_codec:
        raise ValueError(f"qwen3-moe-cuda expert_codec must be one of {sorted(page_mb_by_codec)}")
    requested_w4_arithmetic = options.get("w4_arithmetic_policy")
    if requested_w4_arithmetic is not None:
        w4_arithmetic_policy = str(requested_w4_arithmetic).strip().lower()
        if w4_arithmetic_policy not in QWEN3_MOE_SUPPORTED_W4_ARITHMETIC_POLICIES:
            raise ValueError(
                "qwen3-moe-cuda w4_arithmetic_policy must be one of "
                f"{sorted(QWEN3_MOE_SUPPORTED_W4_ARITHMETIC_POLICIES)}"
            )
        if expert_codec != "w4":
            raise ValueError("qwen3-moe-cuda w4_arithmetic_policy requires expert_codec='w4'")
    expert_page_mb = page_mb_by_codec[expert_codec]
    configured_cache = os.environ.get("MRUN_QWEN3_MOE_CACHE_MB")
    cache_mb = float(
        options.get(
            "cache_mb",
            configured_cache or QWEN3_MOE_CUDA_DEFAULT_CACHE_MB,
        )
    )
    max_active_pages = int(
        options.get(
            "max_active_pages",
            QWEN3_MOE_DEFAULT_MAX_ACTIVE_PAGES,
        )
    )
    if cache_mb <= 0:
        raise ValueError("qwen3-moe-cuda cache_mb must be positive")
    if max_active_pages <= 0:
        raise ValueError("qwen3-moe-cuda max_active_pages must be positive")
    effective_cache_mb = max(
        cache_mb,
        max_active_pages * expert_page_mb,
    )

    configured_binding = os.environ.get(
        "MRUN_QWEN3_MOE_PAGE_BINDING_POLICY",
        QWEN3_MOE_DEFAULT_PAGE_BINDING_POLICY,
    )
    page_binding_policy = (
        str(options.get("page_binding_policy", configured_binding)).strip().lower()
    )
    supported_page_bindings = {
        QWEN3_MOE_DEFAULT_PAGE_BINDING_POLICY,
        QWEN3_MOE_COMPACT_PAGE_BINDING_POLICY,
    }
    if page_binding_policy not in supported_page_bindings:
        raise ValueError(
            f"qwen3-moe-cuda page_binding_policy must be one of {sorted(supported_page_bindings)}"
        )

    configured_prefetch = os.environ.get("MRUN_QWEN3_MOE_ROUTE_PREFETCH", "")
    route_prefetch = options.get("route_prefetch")
    if route_prefetch is None:
        route_prefetch_enabled = configured_prefetch.strip().lower() not in {
            "",
            "0",
            "false",
        }
    elif isinstance(route_prefetch, str):
        route_prefetch_enabled = route_prefetch.strip().lower() not in {
            "",
            "0",
            "false",
        }
    else:
        route_prefetch_enabled = bool(route_prefetch)

    # The promoted slot-indirect ABI always needs one miss slab. The established compact
    # fallback needs a second slab, and opt-in route-history prefetch needs a third. CPU
    # staging tensors are charged to host RAM rather than VRAM.
    staging_slabs = 1
    if page_binding_policy == QWEN3_MOE_COMPACT_PAGE_BINDING_POLICY:
        staging_slabs += 1
    if route_prefetch_enabled:
        staging_slabs += 1
    staging_mb = staging_slabs * max_active_pages * expert_page_mb
    resident_mb = effective_cache_mb + QWEN3_MOE_CUDA_FIXED_MB + staging_mb

    configured_host = os.environ.get("MRUN_QWEN3_MOE_HOST_CACHE_MB")
    host_cache_mb = float(
        options.get(
            "host_cache_mb",
            configured_host or 0.0,
        )
    )
    if host_cache_mb < 0:
        raise ValueError("qwen3-moe-cuda host_cache_mb must be non-negative")
    # HostPageTier caps its allocation at the actual encoded store size.
    effective_host_cache_mb = min(
        host_cache_mb,
        QWEN3_MOE_TOTAL_EXPERT_PAGES * expert_page_mb,
    )
    return (
        round(resident_mb, 1),
        cache_mb,
        round(effective_cache_mb, 1),
        max_active_pages,
        expert_codec,
        expert_page_mb,
        staging_slabs,
        round(effective_host_cache_mb, 1),
    )


@dataclass(frozen=True)
class HostCaps:
    """What a host can do. ``detect()`` probes the local machine (agent-side)."""

    name: str
    ram_mb: float
    vram_mb: float = 0.0
    has_cuda: bool = False
    has_mps: bool = False
    has_ane: bool = False
    cpus: int = 8

    @classmethod
    def detect(cls) -> HostCaps:
        import platform

        import psutil

        ram_mb = psutil.virtual_memory().total / 1e6
        cpus = os.cpu_count() or 8
        has_cuda = False
        vram_mb = 0.0
        try:
            import torch

            has_cuda = torch.cuda.is_available()
            if has_cuda:
                vram_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
            has_mps = (
                bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
            )
        except Exception:
            has_mps = platform.system() == "Darwin" and platform.machine() == "arm64"
        is_asi = platform.system() == "Darwin" and platform.machine() == "arm64"
        return cls(
            name=platform.node().split(".")[0],
            ram_mb=ram_mb,
            vram_mb=vram_mb,
            has_cuda=has_cuda,
            has_mps=has_mps,
            has_ane=is_asi,
            cpus=cpus,
        )


@dataclass(frozen=True)
class RunPlan:
    model: str
    backend: str  # auto chooses resident HF or a compatible paged profile
    dtype: str  # float32 (canonical) | bfloat16 (opt-in)
    device: str  # cpu | cuda | mps
    max_batch: int
    threads: int
    ram_limit_mb: float  # local kill ceiling; fleet reservation base (agent adds kill factor)
    est_ram_mb: float
    est_vram_mb: float = 0.0
    est_wall_s: float | None = None
    weights_gb: float = 0.0  # on-disk size of the model; the server charges this against
    # a host's models-mount disk when the host does NOT already have the bytes (cold)
    engine_options: dict[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    # Additive identity fields. Legacy callers may leave these unset; newer planners
    # and agents can bind a concrete artifact/profile without changing backend aliases.
    engine_profile: str | None = None
    artifact_id: str | None = None
    artifact_kind: str | None = None
    artifact_mount: str | None = None
    artifact_locator: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        from dataclasses import asdict

        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RunPlan:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        known["reasons"] = tuple(known.get("reasons") or ())
        return cls(**known)

    def bind_artifact(self, artifact: dict[str, Any]) -> RunPlan:
        """Bind a discovered artifact without changing legacy planning decisions."""
        locator = dict(artifact.get("locator") or {})
        if artifact.get("path") and not locator.get("path"):
            locator["path"] = artifact["path"]
        mount = artifact.get("mount") or locator.get("mount")
        size_gb = float(artifact.get("bytes") or 0.0) / 1e9
        return replace(
            self,
            artifact_id=str(artifact.get("artifact_id")) if artifact.get("artifact_id") else None,
            artifact_kind=(
                str(artifact.get("artifact_kind")) if artifact.get("artifact_kind") else None
            ),
            artifact_mount=str(mount) if mount else None,
            artifact_locator=locator,
            weights_gb=round(size_gb, 3) if size_gb > 0 else self.weights_gb,
            reasons=(*self.reasons, f"artifact bound: {artifact.get('artifact_id', 'unknown')}"),
        )

    def engine_kwargs(self) -> dict:
        """kwargs for ``open_engine(self.model, backend=self.backend, **kwargs)``."""
        kwargs: dict[str, Any]
        if self.backend == "hf":
            kwargs = {
                "device": self.device,
                "model_kwargs": {"torch_dtype": self.dtype},
            }
        elif _is_dense_cuda_backend(self.backend):
            kwargs = _dense_cuda_engine_kwargs(self.dtype, self.device)
        elif _is_qwen3_moe_cuda_backend(self.backend):
            kwargs = _qwen3_moe_cuda_engine_kwargs(
                self.dtype,
                self.device,
                self.engine_options,
            )
        elif self.backend in {"paged", "paged-fp16", "paged-bf16", "paged-fp32"}:
            kwargs = {}
            if self.device != "cpu":
                kwargs["device"] = self.device
            if self.backend == "paged" and self.device != "cpu":
                kwargs["compute_dtype"] = {
                    "bfloat16": "bf16",
                    "bf16": "bf16",
                    "float16": "fp16",
                    "fp16": "fp16",
                    "float32": "fp32",
                    "fp32": "fp32",
                }[self.dtype]
        else:
            kwargs = {}

        locator_path = str(
            self.artifact_locator.get("path") or self.artifact_locator.get("relative_path") or ""
        )
        if locator_path:
            if self.backend in {"paged", "paged-fp16", "paged-bf16", "paged-fp32"}:
                kwargs.setdefault("store_path", locator_path)
            elif _is_dense_cuda_backend(self.backend):
                kwargs.setdefault("store_path", locator_path)
            elif _is_qwen3_moe_cuda_backend(self.backend) or self.backend == "olmoe-cuda":
                kwargs.setdefault("store_dir", locator_path)
        return kwargs


def resolve_max_batch(device: str, seq_lens: list[int], hidden: int) -> int:
    """Device-aware bucket cap for batched forwards (ported from discovery).

    ``GATHER_MAX_BATCH`` env wins when set (operator override, any device). Otherwise the
    proven default stays 16 on cpu; on cuda the cap is 64 — MEASURED on beast 2026-07-15
    (idle card, 2026-07-15-beast-rooflines): the hf-path bf16 throughput plateau spans
    B=8..64 (~9.4k tok/s at T=128) and drops past it; the path is logits-D2H-bound over
    the Gen4 x4 link (6.4 GB/s pinned), NOT FLOP-bound, so raising the cap buys nothing —
    the earlier "knee at B~128" note was a CPU-curve assertion that does not transfer.
    Activation-memory guard: a batched forward holds a [B, Tmax, hidden] fp32 tensor per
    tap; past 4 GiB at B=64 the bump is skipped and the cap stays 16."""
    env = os.environ.get("GATHER_MAX_BATCH", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if device not in ("cuda", "mps"):
        return 16
    # mps: 64 measured mildly better than 16 (3.4s vs 4.1s per 512 scored probes,
    # 2026-07-24 G5 serve PoC, fresh-process warm-Metal); same activation guard as cuda.
    tmax = max((int(x) for x in seq_lens), default=1)
    if 64 * tmax * max(int(hidden), 1) * 4 > 4 * (1 << 30):  # B * Tmax * hidden * 4B > 4 GiB
        return 16
    return 64


def _dtype_name(
    dtype: str | None,
    host: HostCaps | None = None,
    task: str = "forward",
    model: str | None = None,
) -> tuple[str, str]:
    """(canonical dtype name, reason). Unified-memory Macs default to bf16: halving the
    weight footprint is what keeps an 18GB host out of swap; canonical numbers run on the
    cuda host at fp32 or pass the parity gate first.

    Mac EXCEPTION (measured 2026-07-24, G5 serve PoC arm E): torch CPU bf16 compute is NOT
    Accelerate bf16 — batched forced-choice scoring ran 99.3s bf16 vs 21.0s fp32 (4.7x
    regression) on qwen2.5-0.5b. So non-train tasks pick fp32 whenever the fp32 footprint
    comfortably fits host RAM; bf16 stays for training (native-bf16 load, ~1.37x LoRA,
    half RAM) and for models where fp32 would risk swap."""
    if dtype in (None, "auto"):
        env = os.environ.get("MRUN_TORCH_DTYPE") or os.environ.get("PANR_TORCH_DTYPE") or ""
        env = env.strip().lower()
        if env in ("bf16", "bfloat16"):
            return "bfloat16", f"dtype=bfloat16 (env opt-in {env!r})"
        if env in ("fp32", "float32"):
            return "float32", f"dtype=float32 (env opt-in {env!r})"
        if host is not None and host.has_mps and not host.has_cuda:
            if model is not None and task != "train":
                try:
                    fp32_mb = float(estimate_memory(model, dtype="float32").est_rss_mb)
                except Exception:  # noqa: BLE001 — unknown model: keep the bf16 default
                    fp32_mb = float("inf")
                if fp32_mb <= 0.35 * host.ram_mb:
                    return "float32", (
                        "dtype=float32 (mac cpu, fits in RAM: torch-cpu-bf16 scoring is a "
                        "measured 4.7x regression vs fp32/Accelerate — 2026-07-24 G5 arm E; "
                        "bf16 kept for train and RAM-tight models, opt-in via dtype/env)"
                    )
            return "bfloat16", (
                "dtype=bfloat16 (unified-memory mac default; canonical numbers go to the "
                "cuda host fp32 or must pass the parity gate)"
            )
        if host is not None and host.has_cuda:
            if task == "train":
                # Inference/scoring bf16 is gated (2026-07-15 parity run, argmax 12/12 +
                # content-lift Δ0.03%); TRAINING under bf16 gradients is not — the gate
                # there is the acquisition gauntlet (operator must still mint), pending.
                return "float32", (
                    "dtype=float32 (train stays fp32 until the acquisition-gauntlet "
                    "gate passes for bf16 gradients; bf16 opt-in via dtype/env)"
                )
            return "bfloat16", (
                "dtype=bfloat16 (cuda tensor-core default; gated 2026-07-15: argmax 12/12, "
                "fc winners 6/6, content-lift Δ0.03%, det exact — see "
                "2026-07-15-cuda-dtype-parity; fp32 oracle via dtype='fp32' or "
                "MRUN_TORCH_DTYPE=fp32)"
            )
        return "float32", "dtype=float32 (canonical reference; bf16 is opt-in only)"
    d = str(dtype).strip().lower()
    if d in ("bf16", "bfloat16"):
        return "bfloat16", "dtype=bfloat16 (explicit opt-in — not the canonical path)"
    if d in ("fp16", "float16"):
        return "float16", "dtype=float16 (explicit opt-in — accelerator path, parity-gate it)"
    return "float32", "dtype=float32"


def plan_run(
    model: str,
    task: str = "forward",
    *,
    host: HostCaps | None = None,
    seq_lens: list[int] | None = None,
    backend: str = "auto",
    dtype: str | None = "auto",
    device: str | None = None,
    threads: int | None = None,
    backend_options: dict[str, Any] | None = None,
    workload: str | None = None,
) -> RunPlan:
    """Decide how ``model`` should run on ``host`` for ``task``. Pure policy — does not load
    anything. ``backend``/``dtype``/``device`` override the auto decisions (overrides are
    recorded in ``reasons`` so deviation from canon is visible).

    ``workload`` (opt-in) names the execution regime when it changes the right backend:
    the measured backend inversions (bandwidth-roofline PoC 2026-07-23/24) are regime
    facts, not model facts — int4 is x2.34 at B=1 decode but LOSES to bf16 on batched
    scoring and prefill. Currently wired: ``workload="decode"`` on a Metal-only host
    selects ``mlx-q4`` (B=1 decode; 0.5B 213 tok/s ~93x paged-cpu, 7B at the bandwidth
    roofline). ``None`` (default) keeps every existing decision unchanged.

    When the fleet scheduler already chose a plan for this job (``MRUN_PLAN`` env, set by
    the agent from the job's stored plan), that plan wins — the whole point is that the
    same decision drives admission AND execution."""
    if workload is not None:
        # Validate BEFORE the MRUN_PLAN early return — otherwise a typo'd workload
        # raises or not depending on ambient env state.
        workload = workload.strip().lower()
        if workload not in {"decode", "prefill", "score", "batch"}:
            raise ValueError(
                f"unknown workload {workload!r} (want decode/prefill/score/batch or None)"
            )
    pre = os.environ.get("MRUN_PLAN")
    if pre and backend == "auto" and dtype in (None, "auto") and device is None:
        try:
            import json

            plan = RunPlan.from_dict(json.loads(pre))
            if plan.model == model:
                extra = [f"workload={workload} ignored (scheduler plan wins)"] if workload else []
                return RunPlan.from_dict(
                    {
                        **plan.as_dict(),
                        "reasons": [*plan.reasons, "plan from scheduler (MRUN_PLAN)", *extra],
                    }
                )
        except Exception:  # noqa: BLE001 — a malformed env must not break planning
            pass
    requested_backend = str(backend)
    backend = _canonical_backend_name(backend)
    host = host or HostCaps.detect()
    backend_options = {
        key: os.fspath(value) if isinstance(value, os.PathLike) else value
        for key, value in dict(backend_options or {}).items()
    }
    seq_lens = seq_lens or [512]
    reasons: list[str] = [
        f"host={host.name} ram={host.ram_mb:.0f}MB cuda={host.has_cuda} "
        f"mps={host.has_mps} ane={host.has_ane} task={task}"
    ]
    if backend != requested_backend.strip().lower().replace("_", "-"):
        reasons.append(f"backend alias {requested_backend!r} canonicalized to {backend!r}")

    if workload is not None:
        reasons.append(f"workload={workload}")
    if (
        workload == "decode"
        and backend == "auto"
        and dtype in (None, "auto")
        and device is None
        and host.has_mps
        and not host.has_cuda
        and task.strip().lower() not in {"train", "training", "backward", "score"}
    ):
        # Regime-gated speed path (measured, bandwidth-roofline PoC): int4-resident Metal
        # decode sits at the memory roofline (0.5B 213 tok/s, 7B 29.1 tok/s ~= 122 GB/s
        # streamed) where bf16 is x2.34 slower at B=1 — and the SAME lever inverts on
        # batched scoring/prefill, which is why this branch keys on workload, not host.
        # mlx-q4 is approximate-quantized: canonical numbers still need the fp32 oracle
        # or a capability gate (capabilities().exact_reference is False).
        backend = "mlx-q4"
        workload_selected_backend = True
        reasons.append(
            "backend=mlx-q4 (workload=decode on Metal host: B=1 decode at the bandwidth "
            "roofline, x2.34 vs bf16; INVERTS on prefill/batched — approximate-quantized, "
            "fp32 oracle stays canonical)"
        )
    else:
        workload_selected_backend = False

    if backend in _APPLE_METAL_BACKENDS and not host.has_mps:
        raise ValueError(f"{backend} requires an Apple Metal host; {host.name!r} has no MPS")
    if backend == "ane" and not host.has_ane:
        raise ValueError(
            f"ane/coreml requires an Apple Silicon host; {host.name!r} has no ANE capability"
        )
    if backend in _APPLE_BACKENDS and task.strip().lower() in {"train", "training", "backward"}:
        raise ValueError(
            f"{backend} is an inference/forward backend; backward and training are not implemented"
        )

    # mac SCORE fast path (measured + gated 2026-07-24, G5 serve PoC): fp16-mps batched
    # scoring with prefix-KV is 6x the fp32-cpu path (3.4s vs 21.0s per 512 probes) and
    # passes the content-lift gate essentially exactly (lift 0.8461 vs fp32 0.8459,
    # paired delta +0.0003 +/- 0.0002). Scope is deliberately narrow: task="score" only —
    # canonical/instrument/train paths keep their existing dtypes; fp32-cpu oracle stays
    # one dtype="fp32" away.
    mac_score_fast = (
        dtype in (None, "auto")
        and device in (None, "mps")
        and host is not None
        and host.has_mps
        and not host.has_cuda
        and task == "score"
        and not (os.environ.get("MRUN_TORCH_DTYPE") or os.environ.get("PANR_TORCH_DTYPE"))
    )
    if mac_score_fast:
        try:
            fp16_mb = float(estimate_memory(model, dtype="float16").est_rss_mb)
        except Exception:  # noqa: BLE001 — unknown model: fall through to normal policy
            fp16_mb = float("inf")
        mac_score_fast = fp16_mb <= 0.35 * host.ram_mb
    if mac_score_fast:
        dtype_name = "float16"
        device = "mps"
        reasons.append(
            "dtype=float16 device=mps (mac score fast path, gated 2026-07-24: content-lift "
            "delta +0.0003+/-0.0002 vs fp32, 6x scoring with prefix-KV; fp32 oracle via "
            "dtype='fp32')"
        )
    elif workload_selected_backend:
        # Don't attach a cpu-scoring dtype rationale to a q4 Metal plan — the Metal
        # engine ignores plan dtype entirely; float16 is the nominal activation dtype.
        dtype_name = "float16"
        reasons.append(
            "dtype=float16 nominal (mlx-q4 is int4-g64 resident; Metal engine ignores plan dtype)"
        )
    else:
        dtype_name, why = _dtype_name(dtype, host, task, model)
        reasons.append(why)
    if backend == "ane":
        if dtype not in (None, "auto", "fp16", "float16"):
            raise ValueError(
                "ane/coreml uses a fixed float16 Core ML graph; dtype must be auto or float16"
            )
        dtype_name = "float16"
        reasons.append("dtype=float16 (fixed Core ML graph and I/O contract)")

    mem = estimate_memory(model, dtype=dtype_name)
    est_ram_mb = float(mem.est_rss_mb)
    if backend == "mlx-q4":
        # Size the unified-memory guard for what actually resides: int4 g64 weights are
        # 4.5 bits/weight vs float16's 16 (asserted bits math, not a measured RSS —
        # conservative because activations/KV stay fp16 and are counted separately).
        # Without this, a fp32/bf16-based estimate refuses the 7B decode case the
        # backend measurably runs at the bandwidth roofline. Re-estimate on an explicit
        # fp16 basis so the scaling holds for BOTH the workload path (already fp16) and
        # an explicit backend="mlx-q4" whose auto dtype may have landed on fp32.
        mem = estimate_memory(model, dtype="float16")
        q4_weights_mb = float(mem.weights_mb) * (4.5 / 16.0)
        est_ram_mb = float(mem.est_rss_mb) - float(mem.weights_mb) + q4_weights_mb
        reasons.append(
            f"est_ram sized at int4-g64 weights {q4_weights_mb:.0f}MB "
            f"(fp16 basis {mem.weights_mb:.0f}MB x 4.5/16)"
        )
    hidden = _hidden_dim(model)

    # device
    if device is None:
        if backend in _APPLE_BACKENDS:
            device = "mps"
            reasons.append(f"device=mps ({backend} requires Apple accelerator hardware)")
        elif host.has_cuda:
            device = "cuda"
            reasons.append("device=cuda (host has cuda)")
        else:
            device = "cpu"
            reasons.append("device=cpu (canonical; mps/mlx are explicit opt-ins)")
    elif not mac_score_fast:
        reasons.append(f"device={device} (explicit)")
    if backend in _APPLE_BACKENDS and device != "mps":
        raise ValueError(f"{backend} requires device='mps'; the RunPlan selected {device!r}")
    if device == "cuda":
        reasons.append("TF32 forced OFF on cuda (breaches 4-decimal stats)")

    max_batch = resolve_max_batch(device, seq_lens, hidden)
    act_mb = estimate_activation_mb(
        model, seq_lens=seq_lens, batch=max_batch, dtype=dtype_name, task=task
    )
    factor = task_rss_factor(task)
    if factor != 1.0:
        reasons.append(f"task={task} rss factor x{factor}")
    est_ram_mb = est_ram_mb * factor
    act_mb = act_mb * factor
    budget_mb = host.ram_mb * HOST_RAM_BUDGET_FRACTION

    # Unified-memory guard: Metal allocations are invisible to the RSS guard AND to
    # nvml-style VRAM telemetry — an explicit mps device on a big model is exactly the
    # crash the admission math cannot see. Downgrade rather than trust it.
    if device == "mps" and host.has_mps and not host.has_cuda:
        if est_ram_mb + act_mb > 0.5 * host.ram_mb:
            if backend in _APPLE_BACKENDS:
                raise ValueError(
                    f"{backend} unified-memory estimate {est_ram_mb + act_mb:.0f}MB exceeds "
                    f"50% of {host.ram_mb:.0f}MB; the RSS guard cannot see Metal allocations"
                )
            else:
                device = "cpu"
                reasons.append(
                    "device mps->cpu (unified-memory guard: Metal memory is invisible to the "
                    f"RSS kill ceiling; est {est_ram_mb + act_mb:.0f}MB > 50% of host RAM)"
                )

    # backend
    if backend != "auto":
        if not workload_selected_backend:
            reasons.append(f"backend={backend} (explicit)")
        if backend in _APPLE_METAL_BACKENDS:
            reasons.append(
                "Apple production speed path: resident MLX/Metal; task metrics remain "
                "parity-gated against the canonical reference"
            )
        if backend == "ane":
            # DEMOTED 2026-07-24. Core ML exists to reach the Neural Engine; MLComputePlan
            # measured 98% of ops placed on the GPU (NE fraction 0.011), and mlx beats it on
            # the same work — 2196.8 vs 1694.5 tok/s warm, 0.62 s vs 66.93 s cold. Prefer mlx
            # unless the run is specifically investigating Core ML.
            reasons.append(
                "backend=ane DEMOTED: coreml measured slower than mlx (1694.5 vs 2196.8 tok/s) "
                "and places ~98% of ops on the GPU, not the Neural Engine; placement is "
                "unverified per run and margins below 0.5 require reference re-scoring"
            )
    elif est_ram_mb + act_mb > budget_mb:
        backend = _auto_paged_backend(model, host)
        reasons.append(
            f"backend={backend} (est {est_ram_mb:.0f}+act {act_mb:.0f}MB > budget "
            f"{budget_mb:.0f}MB = {HOST_RAM_BUDGET_FRACTION:.0%} of host RAM; "
            "paged artifact/profile selected)"
        )
        if backend == "paged":
            # Generic QStore can use the CUDA page stream when the host has CUDA; otherwise it
            # remains the bounded CPU pager. The specialized MoE profile sets its own host tier
            # and device-staging estimate below.
            if not host.has_cuda:
                device = "cpu"
            est_ram_mb = min(est_ram_mb, 2048.0)  # streamed working set + activations, not weights
    else:
        auto_paged_backend = _auto_paged_backend(model, host)
        resident_vram_overflow = bool(
            host.vram_mb
            and device == "cuda"
            and float(mem.weights_mb) + act_mb > host.vram_mb * VRAM_BUDGET_FRACTION
        )
        if (
            auto_paged_backend == "qwen3-moe-cuda"
            and resident_vram_overflow
        ):
            # Select the measured routed-expert pager before its profile-specific cache and KV
            # sizing block below. This keeps qwen3 MoE from falling through the generic HF OOM
            # downgrade path when RAM is ample but the resident expert bank is not.
            backend = auto_paged_backend
            reasons.append(
                f"backend={backend} (resident MoE VRAM estimate "
                f"{float(mem.weights_mb) + act_mb:.0f}MB exceeds host budget; "
                "paged expert profile selected)"
            )
        else:
            backend = "hf"
            reasons.append(
                f"backend=hf (batched canonical; est {est_ram_mb:.0f}+act {act_mb:.0f}MB "
                f"fits budget {budget_mb:.0f}MB)"
            )

    if _is_dense_cuda_backend(backend):
        if not host.has_cuda:
            raise ValueError(
                f"dense-qstore-cuda requires a CUDA host; {host.name!r} has no CUDA capability"
            )
        # Validate the exact constructor contract now rather than recording a plan whose
        # device or precision would later be discarded by ``engine_from_plan``.
        _dense_cuda_engine_kwargs(dtype_name, device)
    if _is_qwen3_moe_cuda_backend(backend):
        if not host.has_cuda:
            raise ValueError(
                f"qwen3-moe-cuda requires a CUDA host; {host.name!r} has no CUDA capability"
            )
        if task.strip().lower() in {"train", "training", "backward"}:
            raise ValueError(
                "qwen3-moe-cuda is an inference/forward backend; "
                "backward, optimizer state, and training are not implemented"
            )
        _qwen3_moe_cuda_engine_kwargs(
            dtype_name,
            device,
            backend_options,
        )
        (
            qwen_resident_mb,
            qwen_cache_mb,
            qwen_effective_cache_mb,
            qwen_max_active_pages,
            qwen_expert_codec,
            qwen_expert_page_mb,
            qwen_staging_slabs,
            qwen_host_cache_mb,
        ) = _qwen3_moe_cuda_resident_mb(backend_options)
        max_batch = min(max_batch, 8)
        qwen_context_tokens = max(int(length) for length in seq_lens)
        qwen_kv_mb = max_batch * qwen_context_tokens * QWEN3_MOE_KV_BYTES_PER_TOKEN_PER_STREAM / 1e6
        if host.vram_mb:
            vram_budget = host.vram_mb * VRAM_BUDGET_FRACTION
            overflow_mb = qwen_resident_mb + qwen_kv_mb - vram_budget
            cache_was_explicit = "cache_mb" in backend_options or bool(
                os.environ.get("MRUN_QWEN3_MOE_CACHE_MB")
            )
            if overflow_mb > 0 and not cache_was_explicit:
                minimum_cache_mb = qwen_max_active_pages * qwen_expert_page_mb
                adjusted_cache_mb = max(
                    minimum_cache_mb,
                    qwen_cache_mb - overflow_mb,
                )
                if adjusted_cache_mb < qwen_cache_mb:
                    # Never round an admission-derived cache upward across the hard VRAM
                    # ceiling. Runtime takes decimal MB, so quantize down to that precision.
                    backend_options["cache_mb"] = math.floor(adjusted_cache_mb * 10.0) / 10.0
                    (
                        qwen_resident_mb,
                        qwen_cache_mb,
                        qwen_effective_cache_mb,
                        qwen_max_active_pages,
                        qwen_expert_codec,
                        qwen_expert_page_mb,
                        qwen_staging_slabs,
                        qwen_host_cache_mb,
                    ) = _qwen3_moe_cuda_resident_mb(backend_options)
        est_ram_mb = QWEN3_MOE_CUDA_BASE_RAM_MB + qwen_host_cache_mb
        act_mb = 0.0
        reasons.extend(
            (
                "qwen3-moe-cuda is profile-gated: auto only for qwen3_moe VRAM overflow; "
                "explicit for other callers pending diverse-route quality/traffic gates",
                (
                    f"resident estimate={qwen_resident_mb:.1f}MB "
                    f"(cache requested/effective={qwen_cache_mb:.1f}/"
                    f"{qwen_effective_cache_mb:.1f}MB, "
                    f"codec={qwen_expert_codec}, page={qwen_expert_page_mb:.6f}MB, "
                    f"max_active_pages={qwen_max_active_pages}, "
                    f"device_staging_slabs={qwen_staging_slabs}, "
                    "BF16 skeleton + staging)"
                ),
                f"host estimate={est_ram_mb:.1f}MB "
                f"(base={QWEN3_MOE_CUDA_BASE_RAM_MB:.1f}MB + "
                f"effective expert tier={qwen_host_cache_mb:.1f}MB)",
                f"KV reservation={qwen_kv_mb:.1f}MB "
                f"(batch={max_batch}, capacity={qwen_context_tokens}, "
                f"{QWEN3_MOE_KV_BYTES_PER_TOKEN_PER_STREAM} bytes/token/stream)",
                "max_batch capped at measured B8 decode envelope",
            )
        )

    # VRAM fit guard: a resident hf+cuda plan whose weights+activations exceed the card is a
    # guaranteed CUDA OOM crash (the 7B-on-16GB case). Route automatic placement through the
    # compatible paged profile instead of silently keeping HF and moving it to CPU. Explicit
    # backend choices remain explicit; the auto policy is the safety boundary.
    est_vram_mb = float(mem.weights_mb) + act_mb if device == "cuda" else 0.0
    if backend == "hf" and device == "cuda":
        est_vram_mb += CUDA_HF_PROCESS_OVERHEAD_MB
        reasons.append(f"CUDA HF process/context allowance={CUDA_HF_PROCESS_OVERHEAD_MB:.0f}MB (NVML beyond tensor bytes)")
    if _is_qwen3_moe_cuda_backend(backend):
        est_vram_mb = qwen_resident_mb + qwen_kv_mb
        if host.vram_mb:
            vram_budget = host.vram_mb * VRAM_BUDGET_FRACTION
            if est_vram_mb > vram_budget:
                raise ValueError(
                    f"qwen3-moe-cuda needs about {est_vram_mb:.0f}MB VRAM for its "
                    f"expert cache plus B{max_batch}/C{qwen_context_tokens} KV reservation, "
                    f"above the {vram_budget:.0f}MB admission budget on {host.name!r}; "
                    "lower backend_options['cache_mb'], batch/concurrency, or context capacity"
                )
    if backend == "hf" and device == "cuda" and host.vram_mb:
        vram_budget = host.vram_mb * VRAM_BUDGET_FRACTION
        if est_vram_mb > vram_budget:
            if requested_backend.strip().lower().replace("_", "-") == "auto":
                backend = _auto_paged_backend(model, host)
                reasons.append(
                    f"backend={backend} (resident HF VRAM estimate {est_vram_mb:.0f}MB "
                    f"> budget {vram_budget:.0f}MB; overflow routes to paged execution)"
                )
                if backend == "paged":
                    # CUDA paging keeps activations and one dequantized working set on the
                    # accelerator while weights remain in the host-side QStore. If the
                    # activation envelope alone is too large, use the CPU pager instead.
                    if act_mb > vram_budget:
                        device = "cpu"
                        reasons.append(
                            f"device cuda->cpu (paged activation estimate {act_mb:.0f}MB "
                            f"> budget {vram_budget:.0f}MB)"
                        )
                    else:
                        reasons.append(
                            f"device=cuda (paged working-set reservation capped at "
                            f"{min(vram_budget, max(PAGED_CUDA_MIN_WORKING_SET_MB, act_mb)):.0f}MB)"
                        )
                    est_ram_mb = min(est_ram_mb, 2048.0)
                # qwen3-moe-cuda is handled by its measured resident/cache calculation above;
                # this branch is primarily for dense paged QStore models.
            else:
                device = "cpu"
                reasons.append(
                    "explicit HF backend retained; device cuda->cpu because resident weights "
                    "exceed the host VRAM budget"
                )
            reasons.append(
                f"resident HF estimate was {est_vram_mb:.0f}MB = weights "
                f"{mem.weights_mb:.0f}+act {act_mb:.0f}MB against {vram_budget:.0f}MB "
                f"({VRAM_BUDGET_FRACTION:.0%} of {host.vram_mb:.0f}MB)"
            )
            est_vram_mb = 0.0
            if backend == "paged" and device == "cuda":
                est_vram_mb = round(
                    min(vram_budget, max(PAGED_CUDA_MIN_WORKING_SET_MB, act_mb)),
                    1,
                )
            max_batch = resolve_max_batch(device, seq_lens, hidden)

    if backend in {"paged", "paged-fp16", "paged-bf16", "paged-fp32"}:
        if device == "cuda" and host.vram_mb:
            vram_budget = host.vram_mb * VRAM_BUDGET_FRACTION
            if act_mb > vram_budget:
                device = "cpu"
                est_vram_mb = 0.0
                max_batch = resolve_max_batch(device, seq_lens, hidden)
                reasons.append(
                    f"device cuda->cpu (paged activation estimate {act_mb:.0f}MB > "
                    f"VRAM budget {vram_budget:.0f}MB)"
                )
            else:
                est_vram_mb = round(
                    min(vram_budget, max(PAGED_CUDA_MIN_WORKING_SET_MB, act_mb)),
                    1,
                )
        else:
            est_vram_mb = 0.0

    threads = threads or min(DEFAULT_THREADS, host.cpus)
    ram_limit_mb = (est_ram_mb + act_mb) * RAM_SAFETY_FACTOR
    # Floor for whole-model loads: torch runtime + loader transients sit above the
    # calibrated 410MB overhead on some codepaths (measured: bf16 qwen0.5 recorder@64
    # peaked >=2082MB vs a 2046MB ceiling — a marginal kill nobody wants).
    if backend == "hf":
        ram_limit_mb = max(ram_limit_mb, 3072.0)
    elif _is_qwen3_moe_cuda_backend(backend):
        ram_limit_mb = max(ram_limit_mb, 6144.0)
    reasons.append(
        f"max_batch={max_batch} threads={threads} ram_limit={ram_limit_mb:.0f}MB "
        f"(est x {RAM_SAFETY_FACTOR})"
    )
    try:
        from .engine.profiles import profile_id_for_backend
        engine_profile = profile_id_for_backend(backend)
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        # Profile identity is additive. Resource policy must stay usable on
        # service hosts that do not install the model execution runtime.
        engine_profile = None

    return RunPlan(
        model=model,
        backend=backend,
        dtype=dtype_name,
        device=device,
        max_batch=max_batch,
        threads=threads,
        ram_limit_mb=round(ram_limit_mb, 1),
        est_ram_mb=round(est_ram_mb, 1),
        est_vram_mb=round(est_vram_mb, 1),
        weights_gb=round(mem.weights_mb * 1.2 / 1000, 2),  # + tokenizer/safetensors slack
        engine_options=backend_options,
        reasons=tuple(reasons),
        engine_profile=engine_profile,
    )


def apply_plan(plan: RunPlan) -> None:
    """Enforce a plan's environment side: thread pinning, RSS limit, TF32-off."""
    set_thread_env(plan.threads)
    os.environ["RSS_LIMIT_MB"] = str(int(plan.ram_limit_mb))
    if plan.device == "cuda":
        try:
            import torch

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


def engine_from_plan(plan: RunPlan):
    """Open the engine a plan calls for (after ``apply_plan``)."""
    from .engine import open_engine

    apply_plan(plan)
    return open_engine(plan.model, backend=plan.backend, **plan.engine_kwargs())


def _hidden_dim(model: str) -> int:
    """Best-effort hidden size from the local snapshot config (falls back to 2048)."""
    import json

    from .models import resolve_model, snapshot_dir

    try:
        snap = snapshot_dir(resolve_model(model))
        raw = json.loads((snap / "config.json").read_text())
        return int(raw.get("hidden_size") or raw.get("n_embd") or raw.get("d_model") or 2048)
    except Exception:
        return 2048
