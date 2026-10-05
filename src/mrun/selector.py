"""Artifact-aware engine selection.

The legacy :func:`mrun.policy.plan_run` function remains the resource-policy
authority.  This module adds the missing placement layer: when a host advertises
concrete model artifacts, it asks the registered engine profiles which artifacts
can actually run and ranks those candidates.  If inventory is unavailable or a
profile is not eligible, selection deliberately falls back to the legacy policy.

That fallback is important for older agents and for experiments that predate the
artifact catalog.  A missing inventory signal must not silently turn a runnable
submission into a new hard dependency on the catalog protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .engine.profiles import EngineProfile, profile_for_backend
from .models import resolve_model
from .policy import HostCaps, RunPlan, plan_run
from .protocol import RAM_KILL_FACTOR, kill_ceiling_mb


class SelectionError(ValueError):
    """Raised for malformed selector inputs, not for an unavailable artifact."""


@dataclass(frozen=True)
class EngineCandidate:
    """One artifact/profile pair that passed metadata and resource policy."""

    profile_id: str
    backend: str
    artifact_id: str
    artifact: dict[str, Any]
    plan: RunPlan
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionResult:
    """The selected plan and the candidates considered by the selector."""

    plan: RunPlan
    candidates: tuple[EngineCandidate, ...] = ()
    used_fallback: bool = False
    reason: str = ""


_PROMOTED_STATUSES = frozenset({"reference", "qualified"})
_PAGED_BACKENDS = frozenset({"paged", "paged-fp16", "paged-bf16", "paged-fp32"})


def _canonical_model_aliases(model: str) -> set[str]:
    """Return stable aliases for matching host inventory to a logical model."""

    raw = str(model).strip().lower()
    aliases = {raw}
    try:
        spec = resolve_model(model)
    except (TypeError, ValueError):
        spec = None
    if spec is not None:
        aliases.update(
            value.lower()
            for value in (
                spec.name,
                spec.hf_id,
                spec.hf_id.rsplit("/", 1)[-1],
            )
            if value
        )
    return aliases


def _model_matches(model: str, artifact: Mapping[str, Any]) -> bool:
    aliases = _canonical_model_aliases(model)
    requested_spec = None
    try:
        requested_spec = resolve_model(model)
    except (TypeError, ValueError):
        pass
    values = []
    for key in ("model", "source_model", "hf_id", "name"):
        value = artifact.get(key)
        if value:
            values.append(str(value).strip().lower())
    if not values:
        # An inventory row without a logical identity is not safe to bind.
        return False
    for value in values:
        # A registered sibling is a distinct logical checkpoint, even when its
        # name starts with the requested name (for example base vs instruct).
        # Resolve before applying the materialization-suffix compatibility rule.
        # Otherwise ``qwen2.5-0.5b-instruct`` is incorrectly accepted for the
        # request ``qwen2.5-0.5b``.
        if requested_spec is not None:
            try:
                candidate_spec = resolve_model(value)
            except (TypeError, ValueError):
                candidate_spec = None
            if candidate_spec is not None and candidate_spec.name != requested_spec.name:
                continue
        if value in aliases:
            return True
        # Store directories commonly append a materialization suffix, for example
        # Qwen3-30B-A3B-fp8-paged-v1.  Only accept a suffix after a complete alias;
        # this avoids treating a different model with a shared label as identical.
        for alias in aliases:
            if value.startswith(alias) and len(value) > len(alias):
                suffix = value[len(alias)]
                if suffix in {"-", "_", "."}:
                    return True
    return False


def _artifact_kind(artifact: Mapping[str, Any]) -> str:
    """Normalize legacy inventory rows to the profile vocabulary."""

    explicit = str(artifact.get("artifact_kind") or "").strip().lower()
    if explicit in {"hf-weights", "qstore", "expert-store", "native-components"}:
        return explicit
    text = " ".join(
        str(artifact.get(key) or "").strip().lower()
        for key in ("kind", "variant", "artifact_id", "path")
    )
    if str(artifact.get("kind") or "").strip().lower() in {
        "weights",
        "hf",
        "safetensors",
    }:
        return "hf-weights"
    if "expert" in text or "olmoe" in text:
        return "expert-store"
    if "native" in text or "component" in text or "compiled" in text:
        return "native-components"
    if "qstore" in text or str(artifact.get("kind") or "").startswith("qstore"):
        return "qstore"
    return explicit


def _qstore_storage_contract(artifact: Mapping[str, Any]) -> str:
    """Infer the lossless/int8 contract exposed by a catalogued QStore.

    Current inventory rows predate a dedicated ``storage_dtype`` field, but the
    lossless builder deliberately publishes a ``-fp32`` materialization suffix.
    Keep the inference narrow and fail closed for lossless backends so an
    explicit numerical contract can never bind the smaller int8 sibling merely
    because it sorts first by byte size.
    """

    text = " ".join(
        str(artifact.get(key) or "").strip().lower()
        for key in ("storage_dtype", "variant", "kind", "path")
    ).replace("_", "-")
    if "fp32" in text or "float32" in text:
        return "fp32"
    return "quantized-or-unknown"


def _artifact_matches_explicit_backend(
    backend: str,
    artifact: Mapping[str, Any],
    backend_options: Mapping[str, Any] | None = None,
) -> bool:
    """Enforce backend-specific QStore storage semantics during exact binding."""

    kind = _artifact_kind(artifact)
    if kind == "qstore":
        contract = _qstore_storage_contract(artifact)
        if backend in {"paged-fp16", "paged-bf16", "paged-fp32"}:
            return contract == "fp32"
        if backend == "paged":
            return contract != "fp32"
    if backend in {"qwen3-moe-cuda", "moe-qstore-cuda"}:
        options = dict(backend_options or {})
        requested_codec = str(options.get("expert_codec") or "").strip().lower()
        artifact_codec = _expert_codec(artifact)
        if requested_codec and artifact_codec != requested_codec:
            return False
        requested_store = str(options.get("store_dir") or "").rstrip("/")
        if requested_store:
            locator = artifact.get("locator")
            artifact_store = str(
                artifact.get("path")
                or (locator.get("path") if isinstance(locator, Mapping) else "")
                or ""
            ).rstrip("/")
            if artifact_store != requested_store:
                return False
    return True


def _normalized_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(artifact)
    kind = _artifact_kind(row)
    if kind:
        row["artifact_kind"] = kind
    locator = dict(row.get("locator") or {})
    if row.get("path") and not locator.get("path"):
        locator["path"] = row["path"]
    if row.get("mount") and not locator.get("mount"):
        locator["mount"] = row["mount"]
    if row.get("host") and not locator.get("host"):
        locator["host"] = row["host"]
    if locator:
        row["locator"] = locator
    return row


def _has_concrete_locator(artifact: Mapping[str, Any]) -> bool:
    """Do not treat legacy warm-kind rows as exact artifact records."""

    locator = artifact.get("locator")
    return bool(
        artifact.get("artifact_id")
        or artifact.get("path")
        or (isinstance(locator, Mapping) and locator.get("path"))
    )


def _expert_codec(artifact: Mapping[str, Any]) -> str | None:
    """Infer the Qwen MoE page codec from the manifest-derived inventory row."""

    text = " ".join(
        str(artifact.get(key) or "").strip().lower()
        for key in ("codec", "variant", "path")
    )
    if any(token in text for token in ("int4", "i4", "w4")):
        return "w4"
    if any(token in text for token in ("fp8", "e4m3")):
        return "fp8"
    return None


def _host_payload(host: HostCaps) -> dict[str, Any]:
    return {
        "name": host.name,
        "caps": {
            "cuda": host.has_cuda,
            "mps": host.has_mps,
            "ane": host.has_ane,
            "cpu": True,
        },
        "ram_total_mb": host.ram_mb,
        "vram_total_mb": host.vram_mb,
        "cpu_threads": host.cpus,
    }


def _plan_fits_host(plan: RunPlan, host: HostCaps) -> bool:
    """Reject artifact candidates whose selected resource contract cannot be admitted.

    ``plan_run`` can intentionally describe an explicit resident fallback that is useful
    for diagnostics, even when it exceeds the host's static memory budget. That is not a
    valid automatic artifact candidate: accepting it here ships an impossible plan to the
    scheduler, which then sees a concrete per-host plan and rejects the submission before
    the safe paged legacy fallback can run.
    """

    ram_margin = max(float(host.ram_mb) * 0.15, 2048.0)
    ram_cap = max(0.0, float(host.ram_mb) - ram_margin)
    if kill_ceiling_mb(float(plan.ram_limit_mb)) > ram_cap:
        return False
    if plan.est_vram_mb > 0 and host.vram_mb > 0:
        vram_cap = max(0.0, float(host.vram_mb) - 1024.0)
        if float(plan.est_vram_mb) * RAM_KILL_FACTOR > vram_cap:
            return False
    return True


def _evidence_for(
    evidence: Mapping[str, Mapping[str, Any]] | None,
    profile: EngineProfile,
) -> Mapping[str, Any]:
    if not evidence:
        return {}
    for key in (profile.profile_id, profile.backend):
        value = evidence.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _score(
    profile: EngineProfile,
    plan: RunPlan,
    evidence: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any],
    pager_available: bool,
) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    throughput = evidence.get("throughput_tok_s")
    if throughput is None:
        throughput = evidence.get("throughput")
    measured = False
    try:
        measured_value = float(throughput)
        measured = measured_value > 0
    except (TypeError, ValueError):
        measured_value = 0.0
    if measured:
        score = measured_value
        reasons.append(f"measured throughput={measured_value:g} tok/s")
    else:
        score = float(profile.speed_prior) * 100.0
        reasons.append(f"profile speed prior={profile.speed_prior:g}")

    cold_start = evidence.get("cold_start_s")
    try:
        if cold_start is not None:
            penalty = min(max(float(cold_start), 0.0) / 10.0, 25.0)
            score -= penalty
            reasons.append(f"cold-start penalty={penalty:g}")
    except (TypeError, ValueError):
        pass

    # A resident HF plan that has already been forced to CPU on a CUDA host is a
    # safe last resort, but a real CUDA pager is preferable whenever its artifact
    # exists.  This is a placement preference, not a second memory heuristic.
    if plan.backend == "hf" and plan.device == "cpu" and pager_available:
        score -= 25.0
        reasons.append("CPU fallback penalty while a CUDA pager artifact is present")
    if not measured:
        # Smaller derived artifacts normally cold-start faster. Keep this a tiny
        # deterministic tie-breaker so evidence remains the dominant signal.
        try:
            size_gb = float(artifact.get("bytes") or 0.0) / 1e9
            score -= min(size_gb / 100.0, 5.0)
        except (TypeError, ValueError):
            pass
    return score, tuple(reasons)


def _fallback(
    plan: RunPlan,
    reason: str,
    *,
    candidates: Sequence[EngineCandidate] = (),
) -> SelectionResult:
    annotated = replace(plan, reasons=(*plan.reasons, f"selector fallback: {reason}"))
    return SelectionResult(
        plan=annotated,
        candidates=tuple(candidates),
        used_fallback=True,
        reason=reason,
    )


def select_run_plan(
    model: str,
    task: str = "forward",
    *,
    host: HostCaps,
    artifacts: Sequence[Mapping[str, Any]] | None = None,
    backend: str = "auto",
    dtype: str | None = "auto",
    device: str | None = None,
    seq_lens: list[int] | None = None,
    backend_options: Mapping[str, Any] | None = None,
    workload: str | None = None,
    evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> SelectionResult:
    """Select a runnable artifact/profile pair for one host.

    ``artifacts is None`` means the caller has no inventory signal.  That is
    intentionally different from an inventory row with no matching artifact: in
    both cases the legacy policy remains the safe compatibility answer, but the
    returned reason makes the distinction observable to callers and receipts.
    """

    if not str(model).strip():
        raise SelectionError("model must be non-empty")
    if not isinstance(host, HostCaps):
        raise SelectionError("host must be a HostCaps instance")

    options = dict(backend_options or {})
    requested_backend = str(backend).strip().lower().replace("_", "-")
    legacy = plan_run(
        model,
        task,
        host=host,
        seq_lens=seq_lens,
        backend=backend,
        dtype=dtype,
        device=device,
        backend_options=options,
        workload=workload,
    )

    # A scheduler-provided plan is already the admission/execution contract. Do
    # not re-rank it in a worker merely because local inventory is available.
    if any("plan from scheduler (MRUN_PLAN)" in reason for reason in legacy.reasons):
        return _fallback(legacy, "scheduler plan is authoritative")

    if requested_backend != "auto":
        if artifacts is None:
            return _fallback(legacy, "explicit backend; artifact inventory unavailable")
        matching = [
            _normalized_artifact(row)
            for row in artifacts
            if (
                isinstance(row, Mapping)
                and _model_matches(model, row)
                and _has_concrete_locator(row)
            )
        ]
        profile = profile_for_backend(legacy.backend)
        if profile is not None:
            compatible = [
                row
                for row in matching
                if _artifact_kind(row) in profile.artifact_kinds
                and _artifact_matches_explicit_backend(legacy.backend, row, options)
            ]
            if compatible:
                # Explicit backend semantics remain unchanged; bind only the
                # smallest matching artifact as an additive locator.
                chosen = min(compatible, key=lambda row: float(row.get("bytes") or 0.0))
                return SelectionResult(
                    plan=legacy.bind_artifact(chosen),
                    candidates=(),
                    used_fallback=False,
                    reason="explicit backend artifact bound",
                )
        return _fallback(legacy, "explicit backend has no matching local artifact")

    if artifacts is None:
        return _fallback(legacy, "artifact inventory unavailable; legacy policy")

    matching = [
        _normalized_artifact(row)
        for row in artifacts
        if (
            isinstance(row, Mapping)
            and _model_matches(model, row)
            and _has_concrete_locator(row)
        )
    ]
    if not matching:
        return _fallback(legacy, "no matching local artifact; legacy policy")

    # These are the first safe automatic profiles.  More specialized profiles can
    # join once their parity/promotion records are published; they should not become
    # an implicit numerical contract merely because a directory exists.
    profile_specs: list[tuple[EngineProfile, str | None]] = []
    hf_profile = profile_for_backend("hf")
    paged_profile = profile_for_backend("paged")
    if hf_profile is not None:
        profile_specs.append((hf_profile, None))
    if paged_profile is not None:
        profile_specs.append((paged_profile, None))
    qwen_profile = profile_for_backend("qwen3-moe-cuda")
    try:
        is_qwen_moe = resolve_model(model).family == "qwen3_moe"
    except (TypeError, ValueError):
        is_qwen_moe = False
    if is_qwen_moe and host.has_cuda and legacy.backend == "qwen3-moe-cuda" and qwen_profile:
        profile_specs.append((qwen_profile, "qwen3_moe_overflow"))

    pager_available = any(
        _artifact_kind(row) in ("qstore", "expert-store", "native-components")
        for row in matching
    )
    candidates: list[EngineCandidate] = []
    for profile, gate in profile_specs:
        if profile.promotion_status not in _PROMOTED_STATUSES and gate is None:
            continue
        supported = [
            row
            for row in matching
            if _artifact_kind(row) in profile.artifact_kinds
        ]
        for artifact in supported:
            candidate_dtype = dtype
            candidate_device = device
            candidate_options = dict(options)
            if profile.backend == "paged":
                candidate_device = candidate_device or ("cuda" if host.has_cuda else "cpu")
            elif profile.backend == "qwen3-moe-cuda":
                candidate_device = candidate_device or "cuda"
                if candidate_dtype in (None, "auto"):
                    candidate_dtype = "bf16"
                artifact_codec = _expert_codec(artifact)
                requested_codec = str(options.get("expert_codec") or "").strip().lower()
                if artifact_codec and requested_codec and artifact_codec != requested_codec:
                    continue
                if artifact_codec:
                    candidate_options["expert_codec"] = artifact_codec
            try:
                candidate_plan = plan_run(
                    model,
                    task,
                    host=host,
                    seq_lens=seq_lens,
                    backend=profile.backend,
                    dtype=candidate_dtype,
                    device=candidate_device,
                    backend_options=candidate_options,
                    workload=workload,
                )
            except (TypeError, ValueError):
                continue
            # ``plan_run(backend="paged")`` treats an explicit backend as a
            # caller-forced legacy request and may charge the resident source weights.
            # If that explicit estimate cannot fit, ask policy to make the normal auto
            # decision so its bounded-memory pager estimate is retained. Small models
            # keep the explicit paged estimate, preserving measured profile comparisons.
            if profile.backend == "paged" and not _plan_fits_host(candidate_plan, host):
                try:
                    candidate_plan = plan_run(
                        model,
                        task,
                        host=host,
                        seq_lens=seq_lens,
                        backend="auto",
                        dtype=candidate_dtype,
                        device=candidate_device,
                        backend_options=candidate_options,
                        workload=workload,
                    )
                except (TypeError, ValueError):
                    continue
            if candidate_plan.backend != profile.backend:
                continue
            if not _plan_fits_host(candidate_plan, host):
                continue
            probe = profile.probe(
                artifact,
                _host_payload(host),
                {"device": candidate_plan.device, "dtype": candidate_plan.dtype, "task": task},
            )
            if not probe.get("eligible"):
                continue
            candidate_evidence = _evidence_for(evidence, profile)
            if candidate_evidence.get("parity_valid") is False:
                continue
            if candidate_evidence.get("promotion_status") not in (
                None,
                *_PROMOTED_STATUSES,
            ):
                continue
            # HF on CUDA that was demoted to CPU is only a fallback when no pager
            # artifact is available. This avoids selecting a slow, resident path for
            # a model the host already has in a bounded-memory format.
            if (
                profile.backend == "hf"
                and candidate_plan.device == "cpu"
                and host.has_cuda
                and pager_available
            ):
                continue
            score, score_reasons = _score(
                profile,
                candidate_plan,
                candidate_evidence,
                artifact=artifact,
                pager_available=pager_available,
            )
            bound = candidate_plan.bind_artifact(artifact)
            reason = (
                f"selector candidate profile={profile.profile_id} artifact="
                f"{artifact.get('artifact_id', 'unknown')}"
            )
            if gate:
                reason += f" gate={gate}"
            bound = replace(bound, reasons=(*bound.reasons, reason, *score_reasons))
            candidates.append(
                EngineCandidate(
                    profile_id=profile.profile_id,
                    backend=profile.backend,
                    artifact_id=str(artifact.get("artifact_id") or ""),
                    artifact=artifact,
                    plan=bound,
                    score=score,
                    reasons=score_reasons,
                )
            )

    if not candidates:
        return _fallback(legacy, "no eligible artifact/profile; legacy policy")

    # Stable profile/artifact ordering makes receipts reproducible when candidates
    # have equal prior/evidence scores.
    candidates.sort(key=lambda item: (-item.score, item.profile_id, item.artifact_id))
    selected = candidates[0]
    selected_plan = replace(
        selected.plan,
        reasons=(
            *selected.plan.reasons,
            f"selector selected {selected.profile_id} score={selected.score:.3f}",
        ),
    )
    return SelectionResult(
        plan=selected_plan,
        candidates=tuple(candidates),
        used_fallback=False,
        reason="artifact-aware profile selection",
    )


__all__ = [
    "EngineCandidate",
    "SelectionError",
    "SelectionResult",
    "select_run_plan",
]
