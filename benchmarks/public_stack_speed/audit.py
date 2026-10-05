"""Bind immutable reports to their sealed request and checkpoint bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def audit(root, stage, *, ssh_host=None):
    from mrun.client.submit import _pack_payload
    report_path = root / "report.json"
    report = json.loads(report_path.read_text())
    job = json.loads((root / "job.json").read_text())
    request = json.loads((stage / "saturn_debug_request.json").read_text())
    seal = json.loads((stage / "payload-seal.json").read_text())
    config = json.loads((stage / "saturn_run_config.json").read_text())
    checks = {
        "job_identity": report["job_id"] == job["job_id"],
        "request_marker": canonical_digest(request) == job["config"]["saturn_request_sha256"],
        "request_config": request["config"] == config == report["config"],
        "sealed_config": sha256(stage / "saturn_run_config.json") == seal["config_bytes_sha256"],
        "sealed_worker": sha256(stage / seal["worker"]) == seal["worker_sha256"],
        "payload_archive": hashlib.sha256(_pack_payload(stage)).hexdigest()
        == job["payload_custody"]["sealed"]["sha256"],
    }
    for name, expected in {**seal.get("files", {}), **seal["wheels"]}.items():
        checks["sealed_file/" + name] = sha256(stage / name) == expected
    binding = report.get("source_binding") or {}
    if report.get("saturn_request_sha256"):
        checks["reported_request"] = report["saturn_request_sha256"] == canonical_digest(request)
        checks["reported_seal"] = report["payload_seal"] == seal
    plan = job.get("plan") or {}
    checkpoint = (plan.get("artifact_locator") or {}).get("path") or report.get("executed_checkpoint_root")
    if not checkpoint:
        raise ValueError("checkpoint root is not bound to this job")
    # Older workers did not report their resolved checkpoint path. A separate
    # read-only audit hashes the admitted root; the historical report stays intact.
    if report.get("executed_checkpoint_root"):
        checks["checkpoint_root"] = report["executed_checkpoint_root"] == checkpoint
    mode = "worker-enforced"
    hardware_audit = None
    if binding.get("checks") and report.get("executed_checkpoint_root"):
        checks.update({"worker/" + key: value is True for key, value in binding["checks"].items()})
    else:
        if not ssh_host:
            raise ValueError("older reports require --ssh-host for a checkpoint-byte audit")
        admitted_host = (plan.get("artifact_locator") or {}).get("host", job["assigned_host"])
        if ssh_host != job["assigned_host"] or ssh_host != admitted_host:
            raise ValueError("checkpoint audit SSH target must match the assigned/admitted host")
        checks["checkpoint_audit_host"] = True
        mode = "post-job-checkpoint-byte-audit"
        code = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); result={}
for p in sorted(root.rglob("*")):
 if not p.is_file() or p.suffix not in {".safetensors",".json",".txt",".model"}: continue
 h=hashlib.sha256()
 with p.open("rb") as stream:
  for block in iter(lambda:stream.read(8<<20),b""): h.update(block)
 result[p.relative_to(root).as_posix()]=h.hexdigest()
cpu=pathlib.Path("/proc/cpuinfo")
cpu_model=next((x.split(":",1)[1].strip() for x in cpu.read_text().splitlines() if x.startswith("model name")),None) if cpu.is_file() else None
print(json.dumps({"weight_files":result,"cpu_model":cpu_model},sort_keys=True))
'''
        command = "python3 -c " + shlex.quote(code) + " " + shlex.quote(checkpoint)
        result = subprocess.run(["ssh", ssh_host, command], check=True, capture_output=True, text=True)
        observed = json.loads(result.stdout)
        checks["admitted_checkpoint_bytes"] = bool(observed["weight_files"]) and observed["weight_files"] == report["weight_files"]
        hardware_audit = {"host": ssh_host, "cpu_model": observed["cpu_model"], "scope": "post-job read-only host identity"}
    record = {"schema": "mrun-public-speed-source-audit-v3", "job_id": job["job_id"],
              "report_sha256": sha256(report_path), "mode": mode, "checkpoint_root": checkpoint,
              "request_sha256": canonical_digest(request), "payload_seal": seal, "checks": checks,
              "hardware_audit": hardware_audit,
              "boundary": "sealed source/configuration plus checkpoint identity; no model quality claim"}
    target = root / "source-audit-v3.json"
    if target.exists():
        raise ValueError("source audit already exists; preserve prior evidence")
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--ssh-host")
    args = parser.parse_args()
    record = audit(args.root, args.stage, ssh_host=args.ssh_host)
    print(json.dumps({"job_id": record["job_id"], "mode": record["mode"], "checks": record["checks"]}))
    return 0 if all(record["checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
