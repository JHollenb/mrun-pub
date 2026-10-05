"""Resolve the actual target and arithmetic for a scientific runtime.

The scheduler needs a small, conservative request before a worker is selected, while the
worker needs one authoritative answer after it can inspect the host.  This module is that
worker-side answer.  It intentionally returns JSON-safe strings so the resolved choice can be
written into the result manifest and compared in MLflow.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..policy import HostCaps, _dtype_name


class ScienceRuntimeError(ValueError):
    """Raised when a science runtime requests an impossible target or dtype."""


_CUDA_BACKENDS = frozenset(
    {
        "dense-qstore-cuda",
        "dense-cuda",
        "cuda-source-int8",
        "cuda-source-int8-compact-head",
        "olmoe-cuda",
        "qwen3-moe-cuda",
        "moe-qstore-cuda",
    }
)
_METAL_BACKENDS = frozenset(
    {
        "mlx",
        "mlx-q4",
        "mlx-component",
        "mlx-component-q4",
        "metal-component",
        "metal-component-q4",
        "ane",
        "coreml",
        "apple",
        "apple-speed",
    }
)
_PAGED_BACKENDS = frozenset(
    {"paged", "paged-fp32", "paged-lossless", "paged-fp16", "paged-bf16"}
)


def canonical_fabric(value: Any) -> str:
    normalized = str(value or "auto").strip().lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "cpu": "cpu",
        "gpu": "cuda",
        "cuda": "cuda",
        "metal": "metal",
        "mps": "metal",
        "apple": "metal",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ScienceRuntimeError(
            f"unknown serve.fabric={value!r}; use auto, cpu, gpu/cuda, or metal/mps"
        ) from error


def canonical_device(value: Any) -> str:
    normalized = str(value or "auto").strip().lower().replace("_", "-")
    if normalized in {"auto", "default"}:
        return "auto"
    if normalized in {"cpu"}:
        return "cpu"
    if normalized in {"gpu", "cuda"}:
        return "cuda:0"
    if normalized in {"metal", "mps", "apple"}:
        return "mps"
    if normalized.startswith("cuda:"):
        suffix = normalized.split(":", 1)[1]
        if suffix.isdigit():
            return normalized
    raise ScienceRuntimeError(
        f"unknown runtime device={value!r}; use auto, cpu, cuda[:N], or metal/mps"
    )


def canonical_dtype(value: Any) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "")
    aliases = {
        "auto": "auto",
        "fp32": "fp32",
        "float32": "fp32",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ScienceRuntimeError(
            f"unknown runtime dtype={value!r}; use auto, fp32, bf16, or fp16"
        ) from error


def _backend_name(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def _device_family(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "mps":
        return "metal"
    if device.startswith("cuda"):
        return "cuda"
    raise ScienceRuntimeError(f"unsupported canonical device {device!r}")


def _backend_family(backend: str, host: HostCaps) -> str:
    if backend in _CUDA_BACKENDS:
        return "cuda"
    if backend in _METAL_BACKENDS:
        return "metal"
    if backend in _PAGED_BACKENDS:
        return "cuda" if host.has_cuda else "cpu"
    if backend == "multifabric":
        if host.has_cuda:
            return "cuda"
        if host.has_mps:
            return "metal"
        return "cpu"
    # HF and the streaming MoE runner can use the best accelerator available.  MPS is a valid
    # HF target, while the ordinary paged reader deliberately has no Metal implementation.
    if host.has_cuda:
        return "cuda"
    if host.has_mps:
        return "metal"
    return "cpu"


def _check_target_available(family: str, host: HostCaps, backend: str) -> None:
    if family == "cuda" and not host.has_cuda:
        raise ScienceRuntimeError(
            f"backend={backend!r} requires CUDA, but host {host.name!r} has no CUDA device"
        )
    if family == "metal" and not host.has_mps:
        raise ScienceRuntimeError(
            f"backend={backend!r} requires Metal/MPS, but host {host.name!r} has no MPS device"
        )


def _policy_dtype(
    requested: str,
    *,
    host: HostCaps,
    family: str,
    model: str,
    task: str,
) -> tuple[str, str]:
    """Use the central mrun dtype policy after constraining it to this target."""

    adapted = replace(
        host,
        has_cuda=family == "cuda" and host.has_cuda,
        has_mps=family == "metal" and host.has_mps,
        vram_mb=host.vram_mb if family == "cuda" else 0.0,
    )
    dtype, reason = _dtype_name(None if requested == "auto" else requested, adapted, task, model)
    return canonical_dtype(dtype), reason


def resolve_runtime_execution(
    config: dict[str, Any], runtime: dict[str, Any], *, host: HostCaps | None = None
) -> dict[str, Any]:
    """Resolve one runtime's fabric, device, and dtype for the current worker.

    Runtime options override the top-level ``serve`` defaults.  ``auto`` is resolved here, never
    passed to an engine.  The result is deliberately stable and serializable because it is part
    of the scientific evidence, not merely an implementation detail.
    """

    host = host or HostCaps.detect()
    serve = dict(config.get("serve") or {})
    options = dict(runtime.get("options") or {})
    backend = _backend_name(runtime.get("backend", ""))
    if not backend:
        raise ScienceRuntimeError("scientific runtime backend cannot be empty")

    # A runtime-specific device is a complete per-runtime override of the top-level target.  A
    # runtime may therefore compare a CPU paged leg beside a CUDA HF leg in one config.  An
    # explicit per-runtime fabric remains available when the device is left as ``auto``.
    option_device = canonical_device(options.get("device", "auto"))
    fabric_value = (
        options["fabric"]
        if "fabric" in options
        else serve.get("fabric", "auto") if option_device == "auto" else "auto"
    )
    requested_fabric = canonical_fabric(fabric_value)
    requested_device = canonical_device(
        serve.get("device", "auto") if option_device == "auto" else option_device
    )
    inferred_family = _backend_family(backend, host)
    if requested_device == "auto":
        family = inferred_family if requested_fabric == "auto" else requested_fabric
        if family == "cuda":
            device = "cuda:0"
        elif family == "metal":
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested_device
        family = _device_family(device)

    # A device is more specific than a fabric alias, but contradictory explicit declarations
    # should fail loudly instead of silently moving a measurement to another accelerator.
    if requested_fabric != "auto" and requested_fabric != family:
        raise ScienceRuntimeError(
            f"runtime {runtime.get('name', backend)!r} requests fabric={requested_fabric!r} "
            f"but device={device!r} selects {family!r}"
        )
    if requested_fabric == "cpu" and inferred_family != "cpu" and requested_device == "auto":
        # ``fabric=cpu`` is an explicit opt-out from a CUDA-specialized backend and cannot make
        # that backend work on the CPU.  Generic HF/paged runtimes are allowed below.
        if backend in _CUDA_BACKENDS:
            raise ScienceRuntimeError(f"backend={backend!r} has no CPU runtime")
    if backend in _METAL_BACKENDS and family != "metal":
        raise ScienceRuntimeError(f"backend={backend!r} requires fabric=metal/mps")
    if backend in _CUDA_BACKENDS and family != "cuda":
        raise ScienceRuntimeError(f"backend={backend!r} requires fabric=gpu/cuda")
    if backend in _PAGED_BACKENDS and family == "metal":
        raise ScienceRuntimeError(
            "paged backends are CPU/CUDA readers; use hf or an MLX backend for Metal"
        )
    if backend == "paged" and any(options.get(key) for key in ("int2", "int3", "int4")):
        if family != "cpu":
            raise ScienceRuntimeError(
                "paged int2/int3/int4 readers are CPU-only; use row-int8 or a lossless CUDA lane"
            )
    _check_target_available(family, host, backend)

    requested_dtype = canonical_dtype(
        options.get("compute_dtype", options.get("dtype", serve.get("dtype", "auto")))
    )
    if backend == "paged-fp16":
        if requested_dtype not in {"auto", "fp16"}:
            raise ScienceRuntimeError("paged-fp16 is fixed to FP16 arithmetic")
        dtype, reason = "fp16", "paged-fp16 alias: FP32 storage with FP16 arithmetic"
    elif backend == "paged-bf16":
        if requested_dtype not in {"auto", "bf16"}:
            raise ScienceRuntimeError("paged-bf16 is fixed to BF16 arithmetic")
        dtype, reason = "bf16", "paged-bf16 alias: FP32 storage with BF16 arithmetic"
    elif backend in {"paged-fp32", "paged-lossless"} and requested_dtype == "auto":
        dtype, reason = "fp32", "paged-fp32 reference: FP32 storage and FP32 arithmetic"
    elif backend in {
        "dense-qstore-cuda",
        "dense-cuda",
        "cuda-source-int8",
        "cuda-source-int8-compact-head",
    }:
        dtype = "bf16" if requested_dtype == "auto" else requested_dtype
        reason = "dense CUDA fused path defaults to BF16 tensor-core arithmetic"
        if dtype not in {"bf16", "fp16"}:
            raise ScienceRuntimeError(f"{backend} requires BF16 or FP16 arithmetic")
    elif backend in {"qwen3-moe-cuda", "moe-qstore-cuda"}:
        if requested_dtype not in {"auto", "bf16"}:
            raise ScienceRuntimeError("qwen3-moe-cuda currently requires BF16 arithmetic")
        dtype, reason = "bf16", "qwen3-moe-cuda runtime contract requires BF16"
    elif backend == "olmoe-cuda":
        dtype = "bf16" if requested_dtype == "auto" else requested_dtype
        reason = "OLMoE CUDA runtime defaults to BF16 expert/skeleton arithmetic"
        if dtype not in {"bf16", "fp16", "fp32"}:
            raise ScienceRuntimeError("olmoe-cuda dtype must be BF16, FP16, or FP32")
    elif backend in {"ane", "coreml"} and requested_dtype == "auto":
        dtype, reason = "fp16", "Core ML compiled graph uses its FP16 I/O path"
    elif backend in {"mlx-q4"} and requested_dtype == "auto":
        dtype, reason = "fp16", "MLX q4 fused matmul path is quantized with FP16 boundary"
    elif backend in {
        "mlx",
        "mlx-component",
        "mlx-component-q4",
        "metal-component",
        "metal-component-q4",
    } and requested_dtype == "auto":
        dtype, reason = "bf16", "MLX/Metal runtime reports BF16 model arithmetic"
    elif requested_dtype == "auto":
        task = "train" if str(serve.get("type", "inference")).lower() == "training" else "forward"
        dtype, reason = _policy_dtype(
            requested_dtype,
            host=host,
            family=family,
            model=str(
                config.get("model", {}).get("path")
                or config.get("model", {}).get("name", "")
            ),
            task=task,
        )
    else:
        dtype, reason = requested_dtype, "dtype explicitly requested"

    return {
        "backend": backend,
        "fabric": family,
        "device": device,
        "dtype": dtype,
        "host_detected": host.name,
        "reason": reason,
    }


__all__ = [
    "ScienceRuntimeError",
    "canonical_device",
    "canonical_dtype",
    "canonical_fabric",
    "resolve_runtime_execution",
]
