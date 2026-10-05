"""Client-side admission preflight — fail before the scheduler, not on the GPU.

Every check here encodes a real lane-killing failure class from the 2026-07-31
frontier runs: a bare-relative custody path that dangled in the extracted payload,
a runtime ``job.log`` inside a hashed result root, a workload whose required row
geometry was unconstructible, and a reservation guessed below a measured peak.

The receipt is part of the proof: it binds the exact packed payload bytes and the
exact command that were checked, with a self-verifying fingerprint in the
``CudaGraphPromotionRecord`` style. An impossible request must fail preflight, not
remain queued (AGENTS.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..ids import config_run_id
from ..io import compact_json, stable_json
from ..protocol import family_key_for

PREFLIGHT_RECEIPT_SCHEMA = "mrun-preflight-receipt-v1"
PREFLIGHT_CONFIG_SCHEMA = "mrun-preflight-config-v1"
PREFLIGHT_CONFIG_NAME = "preflight.v1.json"

CHECK_NAMES = (
    "cmd-path-closure",
    "hashed-root-hygiene",
    "verifier-dry-run",
    "workload-geometry",
    "reservation-sanity",
)
CHECK_STATUSES = ("passed", "failed", "warned", "skipped")

# The agent writes its live log into the extracted-payload work dir; any hashed
# result root that contains the work dir will therefore contain runtime-mutable
# bytes (agent/executor.py).
RUNTIME_MUTABLE_NAMES = ("job.log",)

_PATHLIKE_EXTENSIONS = (
    ".py",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".txt",
    ".npz",
    ".safetensors",
    ".csv",
)
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=(.*)$")

RESERVATION_HEADROOM = 1.15


def _sha256_hex(value: object, field_name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class CheckResult:
    check: str
    status: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.check not in CHECK_NAMES:
            raise ValueError(f"unknown preflight check: {self.check!r}")
        if self.status not in CHECK_STATUSES:
            raise ValueError(f"unknown preflight status: {self.status!r}")
        if not isinstance(self.evidence, dict):
            raise ValueError("evidence must be a JSON-safe mapping")

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


class PreflightRejected(RuntimeError):
    def __init__(self, receipt: PreflightReceipt) -> None:
        blockers = "; ".join(
            f"{c.check}: {c.detail}" for c in receipt.checks if c.status == "failed"
        )
        super().__init__(f"preflight rejected ({blockers})")
        self.receipt = receipt


@dataclass(frozen=True)
class PreflightReceipt:
    experiment: str
    client_run_id: str
    cmd: tuple[str, ...]
    verdict: str
    checks: tuple[CheckResult, ...]
    payload_root: str | None = None
    payload_sha256: str | None = None
    created_ts: float = 0.0
    schema_version: str = PREFLIGHT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PREFLIGHT_RECEIPT_SCHEMA:
            raise ValueError(f"unsupported preflight receipt schema: {self.schema_version}")
        if not self.experiment or not self.client_run_id:
            raise ValueError("experiment and client_run_id must be non-empty")
        if self.verdict not in ("passed", "rejected"):
            raise ValueError(f"unknown preflight verdict: {self.verdict!r}")
        checks = tuple(self.checks)
        if not checks:
            raise ValueError("a preflight receipt must carry at least one check")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "cmd", tuple(str(t) for t in self.cmd))
        any_failed = any(c.status == "failed" for c in checks)
        if (self.verdict == "rejected") != any_failed:
            raise ValueError("verdict must be 'rejected' iff any check failed")
        if self.payload_sha256 is not None:
            object.__setattr__(
                self,
                "payload_sha256",
                _sha256_hex(self.payload_sha256, "payload_sha256"),
            )

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment": self.experiment,
            "client_run_id": self.client_run_id,
            "cmd": list(self.cmd),
            "verdict": self.verdict,
            "checks": [c.as_dict() for c in self.checks],
            "payload_root": self.payload_root,
            "payload_sha256": self.payload_sha256,
        }

    @property
    def fingerprint(self) -> str:
        canonical = compact_json(self._fingerprint_payload())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        payload = self._fingerprint_payload()
        payload["created_ts"] = self.created_ts
        payload["receipt_fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreflightReceipt:
        claimed = payload.get("receipt_fingerprint")
        receipt = cls(
            experiment=payload["experiment"],
            client_run_id=payload["client_run_id"],
            cmd=tuple(payload.get("cmd") or ()),
            verdict=payload["verdict"],
            checks=tuple(
                CheckResult(
                    check=c["check"],
                    status=c["status"],
                    detail=c.get("detail", ""),
                    evidence=dict(c.get("evidence") or {}),
                )
                for c in payload.get("checks") or ()
            ),
            payload_root=payload.get("payload_root"),
            payload_sha256=payload.get("payload_sha256"),
            created_ts=float(payload.get("created_ts") or 0.0),
            schema_version=payload.get("schema_version", PREFLIGHT_RECEIPT_SCHEMA),
        )
        if claimed is not None and claimed != receipt.fingerprint:
            raise ValueError("preflight receipt fingerprint mismatch")
        return receipt


def default_receipt_dir() -> Path:
    env = os.environ.get("MRUN_PREFLIGHT_DIR")
    if env:
        return Path(env)
    return Path.home() / ".config" / "mrun" / "preflight"


def write_receipt(receipt: PreflightReceipt, directory: Path | None = None) -> Path:
    directory = directory or default_receipt_dir()
    directory.mkdir(parents=True, exist_ok=True)
    name = f"preflight-{receipt.client_run_id}-{receipt.fingerprint[:12]}.json"
    target = directory / name
    tmp = directory / (name + ".tmp")
    tmp.write_text(stable_json(receipt.as_dict()) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


# -- experiment-declared configuration ---------------------------------------


def load_preflight_config(payload_root: Path) -> dict[str, Any]:
    """Read the optional ``preflight.v1.json`` at the payload root. Fail-closed on
    a malformed file; an absent file simply disables the declared checks."""

    path = payload_root / PREFLIGHT_CONFIG_NAME
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{PREFLIGHT_CONFIG_NAME} must be a JSON object")
    schema = raw.get("schema_version", PREFLIGHT_CONFIG_SCHEMA)
    if schema != PREFLIGHT_CONFIG_SCHEMA:
        raise ValueError(f"unsupported preflight config schema: {schema}")
    allow = raw.get("path_token_allowlist", [])
    roots = raw.get("hashed_roots", [])
    if not isinstance(allow, list) or not all(isinstance(t, str) for t in allow):
        raise ValueError("path_token_allowlist must be a list of strings")
    if not isinstance(roots, list) or not all(isinstance(t, str) for t in roots):
        raise ValueError("hashed_roots must be a list of strings")
    verify = raw.get("verify")
    if verify is not None:
        if (
            not isinstance(verify, dict)
            or not isinstance(verify.get("cmd"), list)
            or not all(isinstance(t, str) for t in verify["cmd"])
            or not isinstance(verify.get("skeleton"), str)
        ):
            raise ValueError("verify must declare cmd (list of str) and skeleton (str)")
    geometry = raw.get("geometry")
    if geometry is not None:
        if not isinstance(geometry, dict) or not all(
            isinstance(geometry.get(k), str) for k in ("module", "function", "rows")
        ):
            raise ValueError("geometry must declare module, function, and rows paths")
    return raw


def _payload_child(payload_root: Path, relative: str, label: str) -> Path:
    """Resolve a config-declared path and require it to remain inside the payload."""
    root = payload_root.resolve()
    resolved = (root / relative).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"declared {label} escapes the payload: {relative}")
    return resolved


# -- check 1: cmd path closure ------------------------------------------------


def _candidate_path_tokens(cmd: list[str]) -> list[str]:
    tokens: list[str] = []
    for raw_token in cmd:
        token = raw_token
        if token.startswith("--") and "=" in token:
            token = token.split("=", 1)[1]
        assignment = _ENV_ASSIGNMENT.match(token)
        if assignment is not None:
            token = assignment.group(1)
        if not token or "{env}" in token or "://" in token:
            continue
        if token.startswith("-"):
            continue
        pathlike = "/" in token or token.endswith(_PATHLIKE_EXTENSIONS)
        if pathlike:
            tokens.append(token)
    return tokens


def check_cmd_path_closure(
    cmd: list[str],
    payload_root: Path | None,
    allowlist: list[str] | None = None,
) -> CheckResult:
    allow = set(allowlist or [])
    missing: list[str] = []
    escapes: list[str] = []
    absolute: list[str] = []
    for token in _candidate_path_tokens(cmd):
        if token in allow:
            continue
        if os.path.isabs(token):
            absolute.append(token)
            continue
        if payload_root is None:
            missing.append(token)
            continue
        root = payload_root.resolve()
        resolved = (root / token).resolve()
        if resolved != root and root not in resolved.parents:
            escapes.append(token)
        elif not resolved.exists():
            missing.append(token)
    evidence: dict[str, Any] = {"absolute_unverified": absolute}
    if missing or escapes:
        evidence.update({"missing": missing, "escapes": escapes})
        parts = []
        if missing:
            parts.append(f"dangling relative path(s): {', '.join(missing)}")
        if escapes:
            parts.append(f"payload-escaping path(s): {', '.join(escapes)}")
        return CheckResult("cmd-path-closure", "failed", "; ".join(parts), evidence)
    return CheckResult(
        "cmd-path-closure",
        "passed",
        "all relative path tokens resolve inside the payload",
        evidence,
    )


# -- check 2: hashed-root hygiene ----------------------------------------------


def check_hashed_root_hygiene(
    payload_root: Path | None, hashed_roots: list[str]
) -> CheckResult:
    if not hashed_roots:
        return CheckResult(
            "hashed-root-hygiene", "skipped", "no hashed_roots declared", {}
        )
    if payload_root is None:
        return CheckResult(
            "hashed-root-hygiene",
            "skipped",
            "cmd-only job: hashed roots cannot be resolved client-side",
            {},
        )
    root = payload_root.resolve()
    contaminated: list[str] = []
    dangling: list[str] = []
    absolute: list[str] = []
    for declared in hashed_roots:
        if os.path.isabs(declared):
            absolute.append(declared)
            continue
        resolved = (root / declared).resolve()
        # The work dir IS the extracted payload root; the agent writes job.log
        # there. A hashed root that equals or contains the work dir will hash
        # runtime-mutable bytes.
        if resolved == root or resolved in root.parents:
            contaminated.append(declared)
        elif not resolved.exists():
            dangling.append(declared)
    evidence: dict[str, Any] = {"absolute_unverified": absolute}
    if contaminated or dangling:
        evidence.update(
            {
                "contains_work_dir": contaminated,
                "runtime_mutable_names": list(RUNTIME_MUTABLE_NAMES),
                "dangling": dangling,
            }
        )
        parts = []
        if contaminated:
            parts.append(
                "hashed root contains the job work dir (runtime job.log would be hashed): "
                + ", ".join(contaminated)
            )
        if dangling:
            parts.append(f"declared hashed root missing from payload: {', '.join(dangling)}")
        return CheckResult("hashed-root-hygiene", "failed", "; ".join(parts), evidence)
    return CheckResult(
        "hashed-root-hygiene", "passed", "hashed roots are clean of runtime files", evidence
    )


# -- check 3: verifier dry-run --------------------------------------------------


def _materialize_skeleton(spec_path: Path, target: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    files = spec.get("files")
    if not isinstance(files, dict):
        raise ValueError("skeleton spec must contain a 'files' object")
    for relpath, content in files.items():
        out = target / relpath
        if target.resolve() not in out.resolve().parents:
            raise ValueError(f"skeleton path escapes the skeleton dir: {relpath}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, dict) and "json" in content:
            out.write_text(stable_json(content["json"]) + "\n", encoding="utf-8")
        elif isinstance(content, dict) and "text" in content:
            out.write_text(str(content["text"]), encoding="utf-8")
        else:
            raise ValueError(f"skeleton file {relpath} must declare 'json' or 'text'")


def check_verifier_dry_run(
    payload_root: Path | None, verify_cfg: dict[str, Any] | None
) -> CheckResult:
    if not verify_cfg or payload_root is None:
        return CheckResult("verifier-dry-run", "skipped", "no verify config declared", {})
    skeleton_rel = verify_cfg["skeleton"]
    try:
        skeleton_path = _payload_child(payload_root, skeleton_rel, "verifier skeleton")
    except ValueError as exc:
        return CheckResult("verifier-dry-run", "failed", str(exc), {})
    if not skeleton_path.exists():
        return CheckResult(
            "verifier-dry-run",
            "failed",
            f"declared skeleton spec missing: {skeleton_rel}",
            {"skeleton": skeleton_rel},
        )
    timeout_s = float(verify_cfg.get("timeout_s", 60.0))
    command_closure = check_cmd_path_closure(list(verify_cfg["cmd"]), payload_root)
    if command_closure.status == "failed":
        return CheckResult(
            "verifier-dry-run",
            "failed",
            f"verifier command is not payload-closed: {command_closure.detail}",
            command_closure.evidence,
        )
    with tempfile.TemporaryDirectory(prefix="mrun-preflight-skel-") as tmp:
        try:
            _materialize_skeleton(skeleton_path, Path(tmp))
        except ValueError as exc:
            return CheckResult(
                "verifier-dry-run", "failed", f"bad skeleton spec: {exc}", {}
            )
        cmd = [t.replace("{skeleton}", tmp) for t in verify_cfg["cmd"]]
        try:
            proc = subprocess.run(
                cmd,
                cwd=payload_root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                "verifier-dry-run",
                "failed",
                f"verifier dry-run exceeded {timeout_s:.0f}s",
                {"cmd": cmd},
            )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            return CheckResult(
                "verifier-dry-run",
                "failed",
                f"verifier exited {proc.returncode} on the synthetic skeleton",
                {"cmd": cmd, "output_tail": tail},
            )
    return CheckResult(
        "verifier-dry-run", "passed", "verifier accepts the synthetic result skeleton", {}
    )


# -- check 4: workload geometry --------------------------------------------------


_GEOMETRY_RUNNER = """
import importlib.util, json, sys
module_path, function_name, rows_path = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("mrun_preflight_geometry", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = json.load(open(rows_path, encoding="utf-8"))
result = getattr(module, function_name)(rows)
print(json.dumps(result))
"""


def check_workload_geometry(
    payload_root: Path | None, geometry_cfg: dict[str, Any] | None
) -> CheckResult:
    if not geometry_cfg or payload_root is None:
        return CheckResult(
            "workload-geometry", "skipped", "no geometry predicate declared", {}
        )
    try:
        module_path = _payload_child(payload_root, geometry_cfg["module"], "geometry module")
        rows_path = _payload_child(payload_root, geometry_cfg["rows"], "geometry rows")
    except ValueError as exc:
        return CheckResult("workload-geometry", "failed", str(exc), {})
    for label, path in (("module", module_path), ("rows", rows_path)):
        if not path.exists():
            return CheckResult(
                "workload-geometry",
                "failed",
                f"declared geometry {label} missing: {path.name}",
                {label: str(path)},
            )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _GEOMETRY_RUNNER,
                str(module_path),
                geometry_cfg["function"],
                str(rows_path),
            ],
            capture_output=True,
            text=True,
            timeout=float(geometry_cfg.get("timeout_s", 60.0)),
        )
    except subprocess.TimeoutExpired:
        return CheckResult("workload-geometry", "failed", "geometry predicate timed out", {})
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-500:]
        return CheckResult(
            "workload-geometry",
            "failed",
            "geometry predicate raised",
            {"output_tail": tail},
        )
    try:
        verdict = json.loads((proc.stdout or "").strip().splitlines()[-1])
        ok = bool(verdict["ok"])
        detail = str(verdict.get("detail", ""))
    except Exception:  # noqa: BLE001
        return CheckResult(
            "workload-geometry",
            "failed",
            "geometry predicate produced malformed output",
            {"stdout_tail": (proc.stdout or "").strip()[-500:]},
        )
    if not ok:
        return CheckResult(
            "workload-geometry",
            "failed",
            f"workload geometry unconstructible: {detail}",
            {"detail": detail},
        )
    return CheckResult("workload-geometry", "passed", detail or "geometry predicate holds", {})


# -- check 5: reservation sanity --------------------------------------------------


def check_reservation_sanity(
    client_run_id: str,
    reservation: dict[str, Any] | None,
    *,
    family_key: str | None = None,
    api: Any | None = None,
) -> CheckResult:
    if api is None:
        return CheckResult(
            "reservation-sanity", "skipped", "offline: no scheduler API available", {}
        )
    # Model identifiers can contain slashes and callers may provide custom family keys. Encode
    # the query value so a model name cannot change the request path or query structure.
    params = f"?family_key={quote(family_key, safe=':')}" if family_key else ""
    try:
        body = api.json("GET", f"/api/history/{client_run_id}{params}")
    except Exception as exc:  # noqa: BLE001 — older server or unreachable: never block
        return CheckResult(
            "reservation-sanity",
            "skipped",
            f"history unavailable ({type(exc).__name__}); server may predate the endpoint",
            {},
        )
    # Prefer the server's n-aware exact basis (p95 with n>=3, else max peak — covers
    # the 1s-sampling undercount); fall back to the latest succeeded observation for
    # servers that predate exact_stats.
    exact_stats = (body or {}).get("exact_stats") or {}
    exact = (body or {}).get("exact") or {}
    peak = exact_stats.get("ram_peak_mb", exact.get("ram_peak_mb"))
    if peak is None:
        return CheckResult(
            "reservation-sanity", "passed", "no measured history for this exact config", {}
        )
    peak = float(peak)
    proposed = int(math.ceil(peak * RESERVATION_HEADROOM))
    declared = (reservation or {}).get("ram_mb")
    evidence = {
        "measured_peak_mb": peak,
        "proposed_ram_mb": proposed,
        "declared_ram_mb": declared,
    }
    if declared is None:
        return CheckResult(
            "reservation-sanity",
            "warned",
            f"no declared ram_mb; measured peak {peak:.0f}MB suggests {proposed}MB",
            evidence,
        )
    if float(declared) < peak:
        return CheckResult(
            "reservation-sanity",
            "failed",
            f"declared ram_mb {float(declared):.0f} is below measured peak "
            f"{peak:.0f}MB; never reserve below a measured peak (propose {proposed}MB)",
            evidence,
        )
    return CheckResult(
        "reservation-sanity",
        "passed",
        f"declared ram_mb covers the measured peak ({peak:.0f}MB)",
        evidence,
    )


# -- orchestrator -----------------------------------------------------------------


def run_preflight(
    *,
    experiment: str,
    cmd: list[str],
    config: dict[str, Any] | None = None,
    payload: str | Path | None = None,
    reservation: dict[str, Any] | None = None,
    model: str | None = None,
    task_family: str | None = None,
    api: Any | None = None,
    payload_pack_fn: Callable[[Path], bytes] | None = None,
) -> PreflightReceipt:
    """Run all five checks and return a fingerprinted receipt. The receipt binds
    the exact packed payload bytes (same packing as submission) so a later reader
    can tell precisely what was checked."""

    config = dict(config or {})
    payload_root = Path(payload).resolve() if payload else None
    declared = load_preflight_config(payload_root) if payload_root else {}
    checks = [
        check_cmd_path_closure(
            list(cmd), payload_root, declared.get("path_token_allowlist")
        ),
        check_hashed_root_hygiene(payload_root, declared.get("hashed_roots") or []),
        check_verifier_dry_run(payload_root, declared.get("verify")),
        check_workload_geometry(payload_root, declared.get("geometry")),
    ]
    client_run_id = config_run_id(experiment, config, list(cmd))
    family_config = dict(config)
    if model and "model" not in family_config:
        family_config["model"] = model
    if task_family and "task_family" not in family_config:
        family_config["task_family"] = task_family
    checks.append(
        check_reservation_sanity(
            client_run_id,
            reservation,
            family_key=family_key_for(experiment, list(cmd), family_config),
            api=api,
        )
    )
    payload_sha256 = None
    if payload_root is not None:
        if payload_pack_fn is None:
            from .submit import _pack_payload  # lazy: submit lazily imports this module

            payload_pack_fn = _pack_payload

        payload_sha256 = hashlib.sha256(payload_pack_fn(payload_root)).hexdigest()
    verdict = "rejected" if any(c.status == "failed" for c in checks) else "passed"
    return PreflightReceipt(
        experiment=experiment,
        client_run_id=client_run_id,
        cmd=tuple(cmd),
        verdict=verdict,
        checks=tuple(checks),
        payload_root=str(payload_root) if payload_root else None,
        payload_sha256=payload_sha256,
        created_ts=time.time(),
    )
