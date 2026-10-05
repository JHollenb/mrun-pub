"""Compatibility engine-profile registry.

Profiles describe the contract around an existing backend; they do not replace the
dispatch table in :mod:`mrun.engine` or change how an engine is opened.  This gives
the planner and inventory code a stable metadata surface while old callers continue
to use ``open_engine(model, backend=...)`` unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EngineProfile:
    profile_id: str
    backend: str
    artifact_kinds: tuple[str, ...] = ()
    devices: tuple[str, ...] = ("cpu",)
    dtypes: tuple[str, ...] = ()
    workloads: tuple[str, ...] = ()
    promotion_status: str = "experimental"
    numerical_contract: str | None = None
    speed_prior: float = 1.0

    def probe(
        self,
        artifact: Mapping[str, Any] | None = None,
        host: Mapping[str, Any] | None = None,
        workload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return eligibility without importing torch or opening model bytes."""
        reasons: list[str] = []
        artifact = artifact or {}
        host = host or {}
        workload = workload or {}
        artifact_kind = str(artifact.get("artifact_kind") or "")
        if self.artifact_kinds and artifact_kind and artifact_kind not in self.artifact_kinds:
            reasons.append(
                f"artifact kind {artifact_kind!r} is not supported by {self.profile_id}"
            )
        device = str(workload.get("device") or "").strip().lower()
        if device and device not in self.devices:
            reasons.append(f"device {device!r} is not supported by {self.profile_id}")
        dtype = str(workload.get("dtype") or "").strip().lower()
        if self.dtypes and dtype and dtype not in self.dtypes:
            reasons.append(f"dtype {dtype!r} is not supported by {self.profile_id}")
        task = str(workload.get("task") or "").strip().lower()
        if self.workloads and task and task not in self.workloads:
            reasons.append(f"workload {task!r} is not supported by {self.profile_id}")
        caps = host.get("caps") if isinstance(host.get("caps"), Mapping) else host
        if "cuda" in self.devices and device.startswith("cuda") and not caps.get("cuda"):
            reasons.append("host does not advertise CUDA")
        if device.startswith("mps") and not caps.get("mps"):
            reasons.append("host does not advertise MPS")
        return {
            "eligible": not reasons,
            "profile_id": self.profile_id,
            "backend": self.backend,
            "promotion_status": self.promotion_status,
            "reasons": reasons,
        }

    def estimate(
        self,
        artifact: Mapping[str, Any] | None = None,
        host: Mapping[str, Any] | None = None,
        workload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a transparent artifact/resource hint for candidate ranking.

        Detailed model-specific estimates remain in ``mrun.policy``.  This method
        intentionally reports only profile facts so introducing the registry cannot
        silently alter existing reservation math.
        """
        del host, workload
        artifact = artifact or {}
        bytes_on_disk = float(artifact.get("bytes") or 0.0)
        return {
            "profile_id": self.profile_id,
            "backend": self.backend,
            "speed_prior": self.speed_prior,
            "weights_gb": round(bytes_on_disk / 1e9, 3),
            "promotion_status": self.promotion_status,
        }

    def preflight(
        self,
        artifact: Mapping[str, Any] | None = None,
        host: Mapping[str, Any] | None = None,
        workload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate metadata and an optional host-local locator, fail closed."""
        result = self.probe(artifact, host, workload)
        locator = (artifact or {}).get("locator") or {}
        path = locator.get("path") or (artifact or {}).get("path")
        if result["eligible"] and path and not Path(str(path)).exists():
            result = {
                **result,
                "eligible": False,
                "reasons": [*result["reasons"], f"artifact locator does not exist: {path}"],
            }
        return result

    def open(self, model: str, **kwargs: Any) -> Any:
        """Open through the legacy engine facade; this is an opt-in adapter."""
        from . import open_engine

        return open_engine(model, backend=self.backend, **kwargs)

    def report(self, engine: Any) -> dict[str, Any]:
        facts = getattr(engine, "runtime_report", None)
        payload = facts() if callable(facts) else {}
        if not isinstance(payload, Mapping):
            payload = {"runtime_report": payload}
        return {
            "profile_id": self.profile_id,
            "backend": self.backend,
            "promotion_status": self.promotion_status,
            **dict(payload),
        }


def _profile(
    profile_id: str,
    backend: str,
    *,
    artifact_kinds: tuple[str, ...] = (),
    devices: tuple[str, ...] = ("cpu",),
    dtypes: tuple[str, ...] = (),
    promotion_status: str = "experimental",
    numerical_contract: str | None = None,
    speed_prior: float = 1.0,
) -> EngineProfile:
    return EngineProfile(
        profile_id=profile_id,
        backend=backend,
        artifact_kinds=artifact_kinds,
        devices=devices,
        dtypes=dtypes,
        promotion_status=promotion_status,
        numerical_contract=numerical_contract,
        speed_prior=speed_prior,
    )


_PROFILES = (
    _profile(
        "hf-reference",
        "hf",
        artifact_kinds=("hf-weights",),
        devices=("cpu", "cuda", "mps"),
        promotion_status="reference",
        numerical_contract="hf-reference",
        speed_prior=2.0,
    ),
    _profile(
        "paged-int8",
        "paged",
        artifact_kinds=("qstore", "expert-store", "native-components"),
        devices=("cpu", "cuda"),
        promotion_status="qualified",
        numerical_contract="paged-int8",
    ),
    _profile(
        "paged-fp32",
        "paged-fp32",
        artifact_kinds=("qstore",),
        devices=("cpu", "cuda"),
        dtypes=("float32", "fp32"),
        promotion_status="qualified",
        numerical_contract="paged-fp32-source-exact-storage-fp32-arithmetic-v1",
    ),
    _profile(
        "paged-lossless-fp16",
        "paged-fp16",
        artifact_kinds=("qstore",),
        devices=("cpu", "cuda"),
        dtypes=("float16", "fp16"),
        promotion_status="experimental",
        numerical_contract="paged-fp16-source-exact-storage-fp16-arithmetic-v1",
    ),
    _profile(
        "paged-lossless-bf16",
        "paged-bf16",
        artifact_kinds=("qstore",),
        devices=("cpu", "cuda"),
        dtypes=("bfloat16", "bf16"),
        promotion_status="experimental",
        numerical_contract="paged-bf16-source-exact-storage-bf16-arithmetic-v1",
    ),
    _profile(
        "dense-qstore-cuda",
        "dense-qstore-cuda",
        artifact_kinds=("qstore", "native-components"),
        devices=("cuda",),
        dtypes=("bfloat16", "bf16", "float16", "fp16"),
        speed_prior=3.0,
    ),
    _profile(
        "qwen3-moe-cuda",
        "qwen3-moe-cuda",
        artifact_kinds=("expert-store", "qstore"),
        devices=("cuda",),
        dtypes=("bfloat16", "bf16", "float16", "fp16"),
        speed_prior=3.0,
    ),
    _profile(
        "olmoe-cuda",
        "olmoe-cuda",
        artifact_kinds=("expert-store",),
        devices=("cuda",),
        dtypes=("bfloat16", "bf16", "float16", "fp16"),
        speed_prior=3.0,
    ),
    _profile(
        "cuda-source-int8",
        "cuda-source-int8",
        artifact_kinds=("native-components", "qstore"),
        devices=("cuda",),
        dtypes=("bfloat16", "bf16", "float16", "fp16"),
        speed_prior=3.0,
    ),
    _profile(
        "cuda-source-int8-compact-head",
        "cuda-source-int8-compact-head",
        artifact_kinds=("native-components", "qstore"),
        devices=("cuda",),
        dtypes=("bfloat16", "bf16", "float16", "fp16"),
        speed_prior=3.0,
    ),
    _profile(
        "moe-stream",
        "moe-stream",
        artifact_kinds=("expert-store", "qstore"),
        devices=("cuda",),
        dtypes=("bfloat16", "bf16", "float16", "fp16"),
        speed_prior=2.0,
    ),
    _profile(
        "mlx",
        "mlx",
        artifact_kinds=("hf-weights", "native-components"),
        devices=("mps",),
        speed_prior=4.0,
    ),
    _profile(
        "mlx-q4",
        "mlx-q4",
        artifact_kinds=("hf-weights", "native-components"),
        devices=("mps",),
        speed_prior=5.0,
    ),
    _profile(
        "coreml-ane",
        "ane",
        artifact_kinds=("hf-weights", "native-components"),
        devices=("mps",),
        speed_prior=4.0,
    ),
    _profile(
        "mlx-component",
        "mlx-component",
        artifact_kinds=("native-components", "hf-weights"),
        devices=("mps",),
        speed_prior=4.0,
    ),
    _profile(
        "mlx-component-q4",
        "mlx-component-q4",
        artifact_kinds=("native-components", "hf-weights"),
        devices=("mps",),
        speed_prior=5.0,
    ),
    _profile(
        "multifabric",
        "multifabric",
        devices=("cpu", "cuda", "mps"),
        promotion_status="experimental",
        speed_prior=4.0,
    ),
)


_ALIASES = {
    "apple": "mlx",
    "apple-speed": "mlx",
    "coreml": "ane",
    "dense-cuda": "dense-qstore-cuda",
    "moe-qstore-cuda": "qwen3-moe-cuda",
    "paged-lossless": "paged-fp32",
}


def canonical_backend(backend: str) -> str:
    value = str(backend).strip().lower().replace("_", "-")
    return _ALIASES.get(value, value)


def engine_profiles() -> tuple[EngineProfile, ...]:
    return _PROFILES


def profile_for_backend(backend: str) -> EngineProfile | None:
    canonical = canonical_backend(backend)
    for profile in _PROFILES:
        if profile.backend == canonical:
            return profile
    return None


def profile_id_for_backend(backend: str) -> str:
    profile = profile_for_backend(backend)
    return profile.profile_id if profile is not None else f"backend:{canonical_backend(backend)}"


__all__ = [
    "EngineProfile",
    "canonical_backend",
    "engine_profiles",
    "profile_for_backend",
    "profile_id_for_backend",
]
