"""remote_run — blocking submit to the fleet scheduler, streaming logs.

The synchronous-experiment UX: write a stub, call ``remote_run``, watch it run. Ctrl-C
DETACHES (the run keeps going; re-attach with ``mrun attach <job_id>`` or by calling
``attach``). Cancel is always explicit (``mrun cancel <job_id>``).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import sys
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ids import config_run_id
from ..protocol import MAX_PAYLOAD_BYTES, TERMINAL_STATES
from .api import Api, ApiError


@dataclass
class RemoteResult:
    job_id: str
    state: str
    job: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.state == "succeeded"

    @property
    def result(self) -> dict[str, Any]:
        return self.job.get("result") or {}


class Detached(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__(
            f"detached from {job_id} (still running) — re-attach: mrun attach {job_id}"
        )
        self.job_id = job_id


def _pack_payload(path: Path) -> bytes:
    tar_bytes = io.BytesIO()
    ignored_directories = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "outputs",
    }

    def normalized_metadata(member: tarfile.TarInfo) -> tarfile.TarInfo:
        # The archive identity is part of guarded admission idempotency. Normalize
        # host- and clock-derived metadata so an unchanged payload repacks to the
        # same bytes after a controller restart.
        member.uid = 0
        member.gid = 0
        member.uname = ""
        member.gname = ""
        member.mtime = 0
        member.pax_headers = {}
        return member

    with tarfile.open(fileobj=tar_bytes, mode="w", format=tarfile.PAX_FORMAT) as tar:
        if path.is_dir():
            for p in sorted(path.rglob("*")):
                relative_path = p.relative_to(path)
                if any(part in ignored_directories for part in relative_path.parts):
                    continue
                if p.is_dir():
                    continue
                tar.add(
                    p,
                    arcname=str(relative_path),
                    recursive=False,
                    filter=normalized_metadata,
                )
        else:
            tar.add(
                path,
                arcname=path.name,
                recursive=False,
                filter=normalized_metadata,
            )
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as compressed:
        compressed.write(tar_bytes.getvalue())
    data = buf.getvalue()
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"payload {len(data) / 1e6:.1f}MB exceeds {MAX_PAYLOAD_BYTES / 1e6:.0f}MB — "
            "ship the stub, not the data (models/datasets resolve on-host)"
        )
    return data


def _guarded_request_identity(
    *,
    experiment: str,
    cmd: Sequence[str],
    config: Mapping[str, Any],
    needs: Mapping[str, Any],
    reservation: Mapping[str, Any] | None,
    payload_sha256: str,
    payload_size: int,
    env_alias: str | None,
    model: str | None,
    task: str,
    backend: str,
    dtype: str | None,
    device: str | None,
    backend_options: Mapping[str, Any],
    timeout_s: float | None,
    priority: int,
) -> str:
    """Return the stable logical identity for a guarded shipped submission.

    The scheduler's v2 request digest also covers resolved plans and reservation
    output. Those values can legitimately change as fleet state/history changes;
    the guarded client identity must instead describe the caller's immutable
    logical request and exact payload bytes. The server then rejects a reused
    idempotency key if the normalized request has changed, rather than creating a
    second job silently.
    """

    identity = {
        "schema": "mrun-guarded-client-identity-v1",
        "experiment": experiment,
        "cmd": [str(value) for value in cmd],
        "config": dict(config),
        "needs": dict(needs),
        "reservation": None if reservation is None else dict(reservation),
        "payload": {"sha256": payload_sha256, "size_bytes": int(payload_size)},
        "env_alias": env_alias,
        "model": model,
        "task": task,
        "backend": backend,
        "dtype": dtype,
        "device": device,
        "backend_options": dict(backend_options),
        "timeout_s": timeout_s,
        "priority": int(priority),
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _guarded_submit(
    *,
    api: Api,
    body: dict[str, Any],
    packed_payload: bytes,
    identity: str,
    experiment: str,
    ttl_s: float,
) -> str:
    """Atomically admit, seal, and return one idempotent shipped job.

    mrun's guarded protocol is intentionally the only path here. A failed claim
    or guarded endpoint is surfaced to the caller; falling back to ordinary
    append-only submission would reintroduce duplicate jobs during replay.
    """

    claim_key = f"mrun:{identity}"
    # Deterministic ownership lets a controller restart renew the same logical
    # claim while the request is being sealed. Fencing epochs still protect the
    # payload if the claim expires and a later owner takes over.
    owner_token = f"mrun-owner:{identity}"
    idempotency_key = f"mrun:{identity}"
    scope = {
        "experiment": experiment,
        "config_selector": {"mrun_guarded_request_sha256": identity},
    }
    claim = api.json(
        "POST",
        "/api/admission-claims/acquire",
        json_body={
            "claim_key": claim_key,
            "owner_token": owner_token,
            "ttl_s": float(ttl_s),
            "scope": scope,
            "metadata": {
                "client": "mrun.client.submit",
                "request_identity": identity,
            },
        },
    )
    if not isinstance(claim, Mapping) or not claim.get("fencing_epoch"):
        raise RuntimeError("mrun guarded admission returned no fencing epoch")
    fencing_epoch = int(claim["fencing_epoch"])
    admission = {
        "claim_key": claim_key,
        "owner_token": owner_token,
        "fencing_epoch": fencing_epoch,
        "idempotency_key": idempotency_key,
    }
    guarded_body = {**body, "admission": admission}
    try:
        receipt = api.json(
            "POST", "/api/jobs/guarded", json_body=guarded_body
        )
        if not isinstance(receipt, Mapping) or not receipt.get("job_id"):
            raise RuntimeError("mrun guarded admission returned no job_id")
        job_id = str(receipt["job_id"])
        custody = receipt.get("payload_custody")
        sealed = custody.get("sealed") if isinstance(custody, Mapping) else None
        already_sealed = (
            isinstance(sealed, Mapping)
            and str(sealed.get("sha256") or "")
            == str(body["payload"]["sha256"])
            and int(sealed.get("size_bytes") or -1) == len(packed_payload)
        )
        if not already_sealed:
            status, raw, _headers = api.request(
                "PUT",
                f"/api/jobs/{job_id}/payload",
                raw_body=packed_payload,
                headers={
                    "X-MRun-Admission-Owner": owner_token,
                    "X-MRun-Admission-Epoch": str(fencing_epoch),
                },
            )
            if status >= 400:
                raise ApiError(status, raw.decode(errors="replace")[:500])
        print(
            f"mrun: guarded submission {job_id} ({experiment})",
            file=sys.stderr,
        )
        _warn_admission_outlook(
            receipt.get("admission_outlook")
            if isinstance(receipt, Mapping)
            else None
        )
        return job_id
    finally:
        try:
            api.json(
                "POST",
                "/api/admission-claims/release",
                json_body={
                    "claim_key": claim_key,
                    "owner_token": owner_token,
                    "fencing_epoch": fencing_epoch,
                },
            )
        except Exception as exc:  # noqa: BLE001 - expiry preserves the fence
            print(
                f"mrun: WARNING guarded admission release failed ({exc}); "
                "the claim will expire",
                file=sys.stderr,
            )


def _config_seq_lens(config: dict[str, Any]) -> list[int] | None:
    """Sequence-length knob from the experiment config, so activation estimates use the
    run's REAL context, not the generic 512 default."""
    for key in ("seq_lens", "max_len", "seq_len", "max_seq_len", "context_len"):
        val = config.get(key)
        if isinstance(val, (list, tuple)) and val:
            return [int(x) for x in val]
        if isinstance(val, (int, float)) and val:
            return [int(val)]
    return None


def _envelope_plans(plans: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine sequential runtime plans into one conservative admission plan.

    A science matrix opens and closes each runtime serially, so RAM/VRAM/threads use the
    maximum phase requirement rather than the sum. Wall time is additive because every phase
    runs. The first plan remains the representative engine identity for backwards-compatible
    receipts; the reasons record that the resource fields are an envelope.
    """

    if not plans:
        raise ValueError("cannot envelope an empty runtime plan list")
    envelope = dict(plans[0])
    for field in (
        "max_batch",
        "threads",
        "ram_limit_mb",
        "est_ram_mb",
        "est_vram_mb",
        "weights_gb",
    ):
        values = [float(plan.get(field) or 0.0) for plan in plans]
        value = max(values)
        envelope[field] = int(value) if field in {"max_batch", "threads"} else round(value, 1)
    wall_times = [float(plan["est_wall_s"]) for plan in plans if plan.get("est_wall_s")]
    envelope["est_wall_s"] = round(sum(wall_times), 1) if wall_times else None
    devices = {str(plan.get("device")) for plan in plans}
    if "cuda" in devices:
        envelope["device"] = "cuda"
    elif "mps" in devices:
        envelope["device"] = "mps"
    reasons: list[str] = []
    for plan in plans:
        reasons.extend(str(reason) for reason in plan.get("reasons") or ())
    reasons.append(
        "science runtime envelope: "
        + ", ".join(str(plan.get("backend", "unknown")) for plan in plans)
    )
    envelope["reasons"] = list(dict.fromkeys(reasons))
    if len(plans) > 1:
        # A combined plan has no single backend option dictionary. Keep only the resource
        # envelope; the science worker opens each runtime from its own resolved config.
        envelope["engine_options"] = {}
        # The phases may intentionally use different artifacts (for example an HF
        # reference followed by a paged engine). Do not let the representative first
        # plan make the scheduler charge or the worker open one artifact for every phase.
        envelope["artifact_id"] = None
        envelope["artifact_kind"] = None
        envelope["artifact_mount"] = None
        envelope["artifact_locator"] = {}
        envelope["reasons"] = [
            *envelope["reasons"],
            "multi-runtime envelope: artifact identity cleared",
        ]
    return envelope


def _plans_for_hosts(
    model: str | None,
    task: str,
    api: Api,
    seq_lens: list[int] | None = None,
    *,
    backend: str = "auto",
    dtype: str | None = "auto",
    device: str | None = None,
    backend_options: dict[str, Any] | None = None,
    plan_variants: Sequence[dict[str, Any]] | None = None,
    error_sink: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """One RunPlan per registered fleet host, planned against that host's REAL caps.

    The scheduler is torch-free and has no model configs, so plans are computed here (the
    submit host has the config cached) and shipped in the submit body; the server picks
    the host and stores the chosen plan, and the agent executes exactly it (``MRUN_PLAN``).
    """
    if not model:
        return None
    try:
        from ..estimate import estimate_memory
        from ..policy import HostCaps
        try:
            from ..selector import select_run_plan
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            # Service-only clients deliberately omit the inference runtime.
            # Keep intent-first sizing through the canonical metadata policy
            # when importing engine profiles would eagerly require torch.
            from types import SimpleNamespace
            from ..policy import plan_run

            def select_run_plan(model, task, *, artifacts=None, **kwargs):
                return SimpleNamespace(plan=plan_run(model, task, **kwargs))

        if estimate_memory(model).param_source == "unknown":
            return None
        variants = list(plan_variants or ()) or [
            {
                "backend": backend,
                "dtype": dtype,
                "device": device,
                "backend_options": backend_options,
            }
        ]
        plans: dict[str, dict[str, Any]] = {}
        for h in api.json("GET", "/api/hosts") or []:
            caps = h.get("caps") or {}
            normalized_device = (device or "").strip().lower()
            if normalized_device == "mps" and not caps.get("mps"):
                continue
            if normalized_device.startswith("cuda") and not caps.get("cuda"):
                continue
            hcaps = HostCaps(
                name=h["name"],
                ram_mb=float(h.get("ram_total_mb") or 0.0),
                vram_mb=float(h.get("vram_total_mb") or 0.0),
                has_cuda=bool(caps.get("cuda")),
                has_mps=bool(caps.get("mps")),
                has_ane=bool(caps.get("ane")),
                cpus=int(h.get("cpu_threads") or 8),
            )
            try:
                host_plans = [
                    select_run_plan(
                        model,
                        task,
                        host=hcaps,
                        artifacts=(
                            h.get("models")
                            if isinstance(h.get("models"), list)
                            else None
                        ),
                        seq_lens=variant.get("seq_lens") or seq_lens,
                        backend=str(variant.get("backend", backend)),
                        dtype=variant.get("dtype", dtype),
                        device=variant.get("device", device),
                        backend_options=dict(variant.get("backend_options") or {}),
                    ).plan.as_dict()
                    for variant in variants
                ]
                plans[h["name"]] = _envelope_plans(host_plans)
            except ValueError:
                # Every runtime must be plannable on the same host. ``needs`` prevents the
                # scheduler from selecting an incompatible host, while this rejects a host
                # that can run only part of a sequential matrix.
                continue
        return plans or None
    except Exception as exc:  # noqa: BLE001 — planning never blocks a submit, but it
        # must not fail SILENTLY either: a caller who thinks policy ran when it didn't
        # ships an unplanned job (measured: 99.6% of fleet traffic bypassed planning).
        print(
            f"mrun: WARNING per-host planning unavailable ({exc}); "
            "server will size from history/default",
            file=sys.stderr,
        )
        if error_sink is not None:
            error_sink["_plan_error"] = str(exc)[:300]
        return None


def _client_estimate(
    model: str | None,
    needs: dict[str, Any],
    *,
    task: str = "forward",
    seq_lens: list[int] | None = None,
    backend: str = "auto",
    dtype: str | None = "auto",
    device: str | None = None,
    backend_options: dict[str, Any] | None = None,
    plan_variants: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """First-principles reservation from the local policy layer (the submit host usually
    has the model config cached even when the run happens elsewhere). Fallback for hosts
    that were unknown at submit time; per-host plans (``_plans_for_hosts``) win."""
    if not model:
        return None
    try:
        from ..estimate import estimate_memory
        from ..policy import HostCaps, plan_run

        # An unknown model estimates as ~overhead-only (410MB) — a falsely TINY
        # reservation that admission would happily grant and the 1.1x guard would kill
        # instantly (or worse, locally crash). Let the server's conservative default win.
        if estimate_memory(model).param_source == "unknown":
            return None

        # plan against the TARGET-ish caps: cuda hosts if needs.cuda, else generic 32GB
        host = HostCaps(
            name="target",
            ram_mb=61_000 if needs.get("cuda") else 32_000,
            vram_mb=16_000 if needs.get("cuda") else 0,
            has_cuda=bool(needs.get("cuda")),
            has_mps=bool(needs.get("mps")),
            has_ane=bool(needs.get("mps")),
        )
        variants = list(plan_variants or ()) or [
            {
                "backend": backend,
                "dtype": dtype,
                "device": device,
                "backend_options": backend_options,
            }
        ]
        plan = _envelope_plans(
            [
                plan_run(
                    model,
                    task,
                    host=host,
                    seq_lens=variant.get("seq_lens") or seq_lens,
                    backend=str(variant.get("backend", backend)),
                    dtype=variant.get("dtype", dtype),
                    device=variant.get("device", device),
                    backend_options=dict(variant.get("backend_options") or {}),
                ).as_dict()
                for variant in variants
            ]
        )
        return {
            "ram_mb": plan["ram_limit_mb"],
            "vram_mb": plan["est_vram_mb"],
            "cpu_threads": plan["threads"],
            "source": "estimated",
        }
    except Exception as exc:  # noqa: BLE001
        print(
            f"mrun: WARNING client estimate unavailable ({exc}); "
            "server will size from history/default",
            file=sys.stderr,
        )
        return None


def _is_flux_submission(model: str | None, config: dict[str, Any]) -> bool:
    """Whether this job uses the multi-component Diffusers FLUX envelope.

    FLUX is in the model registry so experiment identity and history remain stable,
    but it must not inherit the language-model planner's single-weight estimate or
    engine plan.  The worker/submitter supplies the diffusion-specific reservation.
    """
    if config.get("flux_mode") or config.get("flux_boundary_schema"):
        return True
    if not model:
        return False
    try:
        from ..models import resolve_model

        return resolve_model(model).family == "flux"
    except (ImportError, ValueError):
        return False


def _flux_client_estimate(
    model: str | None,
    config: dict[str, Any],
    *,
    task: str,
) -> dict[str, Any] | None:
    """Estimate a FLUX job without importing torch or Diffusers."""
    if not model:
        return None
    try:
        from ..diffusion import estimate_flux_resources

        # The diffusion viewer keeps ``task_family=diffusion`` so its queue
        # history stays separate from language-model work.  The FLUX resource
        # contract intentionally has a smaller vocabulary (forward/generate/
        # measure/capture/train), so never let the UI/history label silently
        # turn a real image job into the 4 GiB probe-default fallback.
        mode = str(config.get("flux_mode") or task or "forward").strip().lower()
        if mode in {"diffusion", "image", "inference"}:
            mode = "generate"
        estimate = estimate_flux_resources(
            model,
            task=mode,
            height=int(config.get("height", 512)),
            width=int(config.get("width", 512)),
            steps=int(config.get("steps", 4)),
            phase_cuda=bool(config.get("phase_cuda", False)),
            capture_sites=bool(
                config.get("capture_sites", mode.strip().lower() in {"measure", "capture"})
            ),
            resident=bool(config.get("resident_worker")),
        )
        return estimate.reservation()
    except (ImportError, TypeError, ValueError):
        # A caller may use a custom model path/alias before adding a registry entry;
        # explicit reservations still work and the server can fall back to history.
        return None


def submit(
    *,
    experiment: str,
    cmd: list[str],
    config: dict[str, Any] | None = None,
    needs: dict[str, Any] | None = None,
    host: str | None = None,
    prefer_host: str | None = None,
    reservation: dict[str, Any] | None = None,
    payload: str | Path | None = None,
    payload_pack_fn: Callable[[Path], bytes] | None = None,
    env_alias: str | None = None,
    model: str | None = None,
    task: str = "forward",
    backend: str = "auto",
    dtype: str | None = "auto",
    device: str | None = None,
    backend_options: dict[str, Any] | None = None,
    plan_variants: Sequence[dict[str, Any]] | None = None,
    timeout_s: float | None = None,
    priority: int = 0,
    note: str | None = None,
    retry_on_kill: bool | None = None,
    api: Api | None = None,
    preflight: bool | str = False,
    guarded: bool = False,
    guarded_ttl_s: float = 900.0,
) -> str:
    """Submit and return the job_id (non-blocking half of remote_run).

    ``host=`` stays a HARD pin (existing scripts rely on host-local paths);
    ``prefer_host=`` is the soft preference — scored up in placement but the job
    migrates to any admissible host once starved.

    ``preflight=True`` runs the client-side admission checks (path closure,
    hashed-root hygiene, verifier dry-run, workload geometry, reservation
    sanity) before anything reaches the scheduler and raises
    ``PreflightRejected`` on failure; ``preflight="warn"`` prints blockers but
    submits anyway. A fingerprinted receipt is always written when enabled.
    These run BEFORE the server's own capacity preflight (422) and clamps —
    client checks catch semantic payload defects; the server owns admission.

    ``guarded=True`` is a shipped-payload-only path. It requires strict client
    preflight, acquires mrun's fenced admission claim, submits with an exact
    payload declaration, seals those same bytes, and releases the claim. It
    never falls back to ordinary append-only submission.

    ``note=`` is bounded operator context for queue triage. It is stored in job
    metadata and submission events but intentionally excluded from guarded
    scientific request identity.
    """
    if preflight not in (False, True, "warn"):
        raise ValueError("preflight must be False, True, or 'warn'")
    if guarded and payload is None:
        raise ValueError("guarded submissions require a shipped payload")
    if guarded and preflight is not True:
        raise ValueError("guarded submissions require preflight=True")
    if guarded and not (0.0 < float(guarded_ttl_s) <= 900.0):
        raise ValueError("guarded_ttl_s must be in (0, 900]")
    api = api or Api()
    needs = dict(needs or {})
    if host:
        needs["host"] = host
    if prefer_host:
        needs["prefer_host"] = prefer_host
    config = dict(config or {})
    if retry_on_kill is not None:
        config["retry_on_kill"] = retry_on_kill
    if model and "model" not in config:
        config["model"] = model
    # ``task`` drives both the client-side planner and the workload's resource
    # geometry.  Persist it in the submitted config when a caller did not name a
    # more specific family, otherwise server-side history would silently group a
    # ``task="training"`` job under the default ``forward`` family.  An explicit
    # task_family remains authoritative for workloads that need a narrower
    # measured-history bucket than the planner's broad task label.
    configured_task_family = str(config.get("task_family") or task).strip()
    if not configured_task_family:
        raise ValueError("task/task_family must be a non-empty string")
    config.setdefault("task_family", configured_task_family)
    task = configured_task_family
    if backend == "auto" and config.get("backend"):
        backend = str(config["backend"])
    if dtype in (None, "auto") and config.get("dtype"):
        dtype = str(config["dtype"])
    if device is None and config.get("device"):
        device = str(config["device"])
    configured_options = config.get("backend_options")
    if configured_options is not None and not isinstance(configured_options, dict):
        raise ValueError("config.backend_options must be an object")
    backend_options = {
        **dict(configured_options or {}),
        **dict(backend_options or {}),
    }
    backend_options = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in backend_options.items()
    }
    normalized_backend = backend.strip().lower().replace("_", "-")
    if normalized_backend in {"qwen3-moe-cuda", "moe-qstore-cuda"}:
        for key in (
            "expert_codec",
            "cache_mb",
            "max_active_pages",
            "host_cache_mb",
            "warm_host",
            "store_dir",
            "route_prefetch",
            "cache_policy",
            "page_binding_policy",
            "prefill_page_policy",
            "route_reduction_policy",
            "w4_arithmetic_policy",
            "verify_store_content",
        ):
            if key in config and key not in backend_options:
                backend_options[key] = config[key]
    if normalized_backend in {
        "dense-qstore-cuda",
        "dense-cuda",
        "olmoe-cuda",
        "qwen3-moe-cuda",
        "moe-qstore-cuda",
    }:
        needs["cuda"] = True
    if normalized_backend in {
        "mlx",
        "mlx-q4",
        "apple",
        "apple-speed",
        "ane",
        "coreml",
    }:
        # The scheduler understands the MPS capability bit. Per-host policy additionally
        # requires has_ane for Core ML, so incompatible Apple hosts receive no plan.
        needs["mps"] = True
    normalized_device = (device or "").strip().lower()
    if normalized_device == "mps":
        needs["mps"] = True
    elif normalized_device.startswith("cuda"):
        needs["cuda"] = True
    if backend != "auto" and "backend" not in config:
        config["backend"] = backend
    if dtype not in (None, "auto") and "dtype" not in config:
        config["dtype"] = dtype
    if device is not None and "device" not in config:
        config["device"] = device
    if backend_options:
        config["backend_options"] = backend_options
    packed_payload: bytes | None = None
    guarded_identity: str | None = None
    if guarded:
        packer = payload_pack_fn or _pack_payload
        packed_payload = packer(Path(payload))
        if not isinstance(packed_payload, bytes):
            raise TypeError("payload_pack_fn must return bytes")
        needs["payload_custody_v2"] = True
        guarded_identity = _guarded_request_identity(
            experiment=experiment,
            cmd=cmd,
            config=config,
            needs=needs,
            reservation=reservation,
            payload_sha256=hashlib.sha256(packed_payload).hexdigest(),
            payload_size=len(packed_payload),
            env_alias=env_alias,
            model=model,
            task=task,
            backend=backend,
            dtype=dtype,
            device=device,
            backend_options=backend_options,
            timeout_s=timeout_s,
            priority=priority,
        )
        config["mrun_guarded_request_sha256"] = guarded_identity
    if preflight:
        from .preflight import PreflightRejected, run_preflight, write_receipt

        preflight_pack_fn = payload_pack_fn
        if guarded:
            # The archive was already created for guarded admission. Reuse those
            # exact bytes so the receipt, declaration, and PUT cannot diverge.
            def preflight_pack(_path: Path) -> bytes:
                if packed_payload is None:  # pragma: no cover - guarded prepared above
                    raise RuntimeError("guarded preflight payload was not prepared")
                return packed_payload

            preflight_pack_fn = preflight_pack
        receipt = run_preflight(
            experiment=experiment,
            cmd=cmd,
            config=config,
            payload=payload,
            reservation=reservation,
            model=model or config.get("model"),
            task_family=config.get("task_family") or task,
            api=api,
            payload_pack_fn=preflight_pack_fn,
        )
        receipt_path = write_receipt(receipt)
        if receipt.verdict == "rejected":
            if preflight == "warn":
                blockers = "; ".join(
                    f"{c.check}: {c.detail}"
                    for c in receipt.checks
                    if c.status == "failed"
                )
                print(
                    f"mrun: preflight WARN ({blockers}) receipt={receipt_path}",
                    file=sys.stderr,
                )
            else:
                raise PreflightRejected(receipt)
    flux_submission = _is_flux_submission(model or config.get("model"), config)
    client_estimate = (
        _flux_client_estimate(model or config.get("model"), config, task=task)
        if flux_submission
        else _client_estimate(
            model or config.get("model"),
            needs,
            task=task,
            seq_lens=_config_seq_lens(config),
            backend=backend,
            dtype=dtype,
            device=device,
            backend_options=backend_options,
            plan_variants=plan_variants,
        )
    )
    plans = (
        None
        if flux_submission
        else _plans_for_hosts(
            model or config.get("model"),
            task,
            api,
            seq_lens=_config_seq_lens(config),
            backend=backend,
            dtype=dtype,
            device=device,
            backend_options=backend_options,
            plan_variants=plan_variants,
            error_sink=config,
        )
    )
    body = {
        "experiment": experiment,
        "client_run_id": config_run_id(experiment, config, cmd),
        "cmd": cmd,
        "config": config,
        "needs": needs,
        "reservation": reservation,
        "client_estimate": client_estimate,
        "plans": plans,
        "payload_kind": "shipped" if payload else "cmd",
        "env_alias": env_alias,
        "timeout_s": timeout_s,
        "priority": priority,
    }
    if note is not None:
        body["note"] = note
    if guarded:
        if packed_payload is None or guarded_identity is None:  # pragma: no cover
            raise RuntimeError("guarded submission payload was not prepared")
        body["payload"] = {
            "sha256": hashlib.sha256(packed_payload).hexdigest(),
            "size_bytes": len(packed_payload),
        }
        return _guarded_submit(
            api=api,
            body=body,
            packed_payload=packed_payload,
            identity=guarded_identity,
            experiment=experiment,
            ttl_s=float(guarded_ttl_s),
        )
    resp = api.json("POST", "/api/jobs", json_body=body)
    job_id = resp["job_id"]
    if payload:
        packed_payload = (
            payload_pack_fn(Path(payload))
            if payload_pack_fn is not None
            else _pack_payload(Path(payload))
        )
        api.request("PUT", f"/api/jobs/{job_id}/payload", raw_body=packed_payload)
    print(
        f"mrun: submitted {job_id} ({experiment}) reservation={resp['reservation']}",
        file=sys.stderr,
    )
    _warn_admission_outlook(resp.get("admission_outlook"))
    return job_id


def _warn_admission_outlook(outlook: dict[str, Any] | None) -> None:
    """Surface server-side clamps and a bleak queue outlook at submit time — a job
    that will never run must not fail silently into ``queued``."""
    if not outlook:
        return
    for clamp in outlook.get("clamps") or []:
        print(f"mrun: WARNING {clamp.get('why') or clamp}", file=sys.stderr)
    status = outlook.get("status")
    if status in ("waiting", "unschedulable"):
        reasons = "; ".join(outlook.get("reasons") or []) or "no host fits right now"
        print(f"mrun: WARNING queued but {status}: {reasons}", file=sys.stderr)
    for recommendation in outlook.get("recommendations") or []:
        host = recommendation.get("host") or "?"
        reservation = recommendation.get("reservation") or {}
        print(
            f"mrun: TIP host={host} try reservation="
            f"ram_mb={reservation.get('ram_mb')} vram_mb={reservation.get('vram_mb')} "
            f"cpu_threads={reservation.get('cpu_threads')}",
            file=sys.stderr,
        )


def _print_failure_diagnostics(job: dict[str, Any], *, include_tail: bool = False) -> None:
    result = job.get("result") or {}
    failure = result.get("failure")
    if not isinstance(failure, dict):
        return
    summary = [
        f"kind={failure.get('kind') or '-'}",
        f"phase={failure.get('phase') or '-'}",
        f"message={failure.get('message') or '-'}",
    ]
    if failure.get("returncode") is not None:
        summary.append(f"returncode={failure['returncode']}")
    print("mrun: failure diagnostics: " + " ".join(summary), file=sys.stderr)
    exception = failure.get("exception")
    if isinstance(exception, dict):
        print(
            f"mrun: exception {exception.get('type')}: {exception.get('message')}",
            file=sys.stderr,
        )
    if include_tail and failure.get("log_tail"):
        print("mrun: failure log tail:", file=sys.stderr)
        print(str(failure["log_tail"]).rstrip(), file=sys.stderr)
    elif failure.get("log_tail"):
        print(f"mrun: failure log tail available via `mrun why {job['job_id']}`", file=sys.stderr)


def attach(job_id: str, *, api: Api | None = None, echo: bool = True) -> RemoteResult:
    """Block until the job finishes, streaming logs. Ctrl-C detaches (raises Detached)."""
    api = api or Api()
    offset = 0
    state = "?"
    try:
        while True:
            job = api.retry_json("GET", f"/api/jobs/{job_id}")
            if job["state"] != state:
                state = job["state"]
                print(f"mrun: {job_id} -> {state}", file=sys.stderr)
            status, chunk, headers = api.request(
                "GET", f"/api/jobs/{job_id}/logs?offset={offset}&wait_s=10", timeout_s=30.0
            )
            if status < 400 and chunk:
                if echo:
                    sys.stdout.write(chunk.decode(errors="replace"))
                    sys.stdout.flush()
                offset = int(headers.get("X-Next-Offset", offset + len(chunk)))
            if state in TERMINAL_STATES:
                job = api.retry_json("GET", f"/api/jobs/{job_id}")
                # A ceiling-killed (or lost) job may have been auto-cloned with a grown
                # reservation — ride the retry instead of reporting a dead end.
                retry_id = (job.get("meta") or {}).get("auto_retry_job_id")
                if not retry_id and state in ("killed_ram", "killed_vram", "lost"):
                    # the retry link is written moments after the terminal state —
                    # give the server one beat before declaring a dead end
                    time.sleep(1.5)
                    job = api.retry_json("GET", f"/api/jobs/{job_id}")
                    retry_id = (job.get("meta") or {}).get("auto_retry_job_id")
                if retry_id:
                    print(
                        f"mrun: {job_id} {state} -> auto-retry {retry_id} "
                        f"(grown reservation)",
                        file=sys.stderr,
                    )
                    job_id, state, offset = retry_id, "?", 0
                    continue
                result = job.get("result") or {}
                result_summary = {
                    key: result.get(key)
                    for key in ("status", "returncode", "elapsed_s", "peak_rss_mb", "peak_vram_mb")
                    if key in result
                }
                print(
                    f"mrun: {job_id} finished: {state} result={result_summary}",
                    file=sys.stderr,
                )
                _print_failure_diagnostics(job, include_tail=not echo)
                return RemoteResult(job_id=job_id, state=state, job=job)
            if not chunk:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print(
            f"\nmrun: detached — run continues. re-attach: mrun attach {job_id} "
            f"(cancel: mrun cancel {job_id})",
            file=sys.stderr,
        )
        raise Detached(job_id) from None


def remote_run(
    experiment: str,
    cmd: list[str],
    *,
    config: dict[str, Any] | None = None,
    needs: dict[str, Any] | None = None,
    host: str | None = None,
    reservation: dict[str, Any] | None = None,
    payload: str | Path | None = None,
    env_alias: str | None = None,
    model: str | None = None,
    task: str = "forward",
    backend: str = "auto",
    dtype: str | None = "auto",
    device: str | None = None,
    backend_options: dict[str, Any] | None = None,
    timeout_s: float | None = None,
    priority: int = 0,
    note: str | None = None,
    api: Api | None = None,
    preflight: bool | str = False,
) -> RemoteResult:
    """Blocking submit: queue on the best host, stream logs, return the result."""
    api = api or Api()
    job_id = submit(
        experiment=experiment,
        cmd=cmd,
        config=config,
        needs=needs,
        host=host,
        reservation=reservation,
        payload=payload,
        env_alias=env_alias,
        model=model,
        task=task,
        backend=backend,
        dtype=dtype,
        device=device,
        backend_options=backend_options,
        timeout_s=timeout_s,
        priority=priority,
        note=note,
        api=api,
        preflight=preflight,
    )
    return attach(job_id, api=api)


def launch(
    cmd: list[str] | str,
    *,
    experiment: str | None = None,
    model: str | None = None,
    task: str = "forward",
    payload: str | Path | None = None,
    prefer_host: str | None = None,
    pin: str | None = None,
    ram_mb: float | None = None,
    vram_mb: float | None = None,
    disk_gb: float | None = None,
    cpu_threads: int | None = None,
    est_wall_s: float | None = None,
    gpu: bool | None = None,
    needs: Mapping[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    env_alias: str | None = None,
    priority: int = 0,
    note: str | None = None,
    timeout_s: float | None = None,
    detach: bool = False,
    retry_on_kill: bool = True,
    api: Api | None = None,
    preflight: bool | str = False,
    guarded: bool = False,
    guarded_ttl_s: float = 900.0,
) -> RemoteResult | str:
    """Intent-first submit: say WHAT to run; mrun picks host, settings and reservation.

    The preferred entry point. No reservation dict, no host pin, no backend knobs —
    sizing comes from the policy plan (when ``model`` is known), measured history for
    this cmd/experiment family, or a small probe default that grows on kill. Escape
    hatches: ``ram_mb``/``vram_mb`` become a declared ask (subject to server clamps),
    ``pin=`` a hard host pin, ``prefer_host=`` a soft one.

    Blocking by default (returns RemoteResult); ``detach=True`` returns the job_id.
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    if not cmd:
        raise ValueError("cmd must not be empty")
    if experiment is None:
        script = next((c for c in cmd if str(c).endswith(".py")), cmd[0])
        experiment = Path(str(script)).stem
    reservation = None
    if cpu_threads is not None and (
        isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1
    ):
        raise ValueError("cpu_threads must be a positive integer")
    if est_wall_s is not None and (
        isinstance(est_wall_s, bool) or not isinstance(est_wall_s, (int, float))
        or not math.isfinite(est_wall_s) or est_wall_s <= 0
    ):
        raise ValueError("est_wall_s must be positive and finite")
    if (cpu_threads is not None or est_wall_s is not None) and ram_mb is None:
        raise ValueError("cpu_threads and est_wall_s require a declared ram_mb")
    if ram_mb is not None or vram_mb is not None or disk_gb is not None:
        reservation = {"ram_mb": ram_mb, "vram_mb": vram_mb or 0.0}
        if disk_gb is not None:
            reservation["disk_gb"] = disk_gb
        if cpu_threads is not None:
            reservation["cpu_threads"] = cpu_threads
        if est_wall_s is not None:
            reservation["est_wall_s"] = est_wall_s
    resolved_needs = dict(needs or {})
    if gpu:
        resolved_needs["cuda"] = True
    api = api or Api()
    job_id = submit(
        experiment=experiment,
        cmd=[str(c) for c in cmd],
        config=config,
        needs=resolved_needs or None,
        host=pin,
        prefer_host=prefer_host,
        reservation=reservation,
        payload=payload,
        env_alias=env_alias,
        model=model,
        task=task,
        timeout_s=timeout_s,
        priority=priority,
        note=note,
        retry_on_kill=retry_on_kill,
        api=api,
        preflight=preflight,
        guarded=guarded,
        guarded_ttl_s=guarded_ttl_s,
    )
    if detach:
        return job_id
    return attach(job_id, api=api)
