"""Collect exactly one job's logs/receipt; never follow retry descendants."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def evidence_surfaces(root, report):
    """Add derived observation streams and a named runtime-integrity receipt."""
    streams, gates = [], {}
    for row in report["rows"]:
        label = row["configuration"]
        for index, seconds in enumerate(row["seconds"]):
            streams.append({"configuration": label, "repeat": index, "model_id": report["model"]["model_id"],
                            "metrics": {"wall_seconds": seconds}})
        for field in ("outputs_repeat_exact", "partial_replay_exact", "abort_exact", "noop_exact"):
            if field in row:
                gates[label + "/" + field] = row[field]
    gates["requested_configurations_completed"] = not bool(report.get("errors"))
    (root / "metrics.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in streams))
    report_sha = hashlib.sha256((root / "report.json").read_bytes()).hexdigest()
    receipt = {"schema": "mrun-public-speed-mechanics-receipt-v1", "job_id": report["job_id"],
               "stage": "runtime-integrity", "mechanics_status": report["mechanics_status"],
               "gates": gates, "report": {"path": "report.json", "sha256": report_sha},
               "gate_contract": {"consumer": "unchanged native continuation in its declared numerical lane",
                                 "observable": "exact repeated/restore/abort/no-op output and declared configuration completion",
                                 "direction": "equality", "threshold": 0,
                                 "calibration": "same-program exact state/output contract",
                                 "split": "same prompt and common parent; no held-out semantic claim",
                                 "aggregation": "individual applicable mechanics rows retained",
                                 "failure_action": "retain outputs, mark mechanics incomplete; no semantic null"}}
    (root / "mechanics-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def main():
    from mrun.client.api import Api
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url", default=os.environ.get("MRUN_URL"))
    args = parser.parse_args()
    api = Api(base_url=args.url)
    job = api.json("GET", "/api/jobs/" + args.job_id)
    if job["job_id"] != args.job_id:
        raise ValueError("scheduler response belongs to a different job")
    if job["state"] in {"queued", "leased", "running"}:
        print(json.dumps({"job_id": args.job_id, "state": job["state"], "collected": False}))
        return 2
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    status, raw, _ = api.request("GET", f"/api/jobs/{args.job_id}/logs?offset=0&wait_s=1")
    if status != 200:
        raise RuntimeError("scheduler refused direct-job log collection")
    logs = raw.decode()
    (args.out / "worker.log").write_bytes(raw)
    marker = "PUBLIC_STACK_REPORT="
    report = next((json.loads(line[len(marker):]) for line in logs.splitlines()
                   if line.startswith(marker)), None)
    if report is not None:
        if report["job_id"] != args.job_id:
            raise ValueError("worker report belongs to a different job; raw logs retained")
        marker = report.get("saturn_request_sha256")
        if marker and marker != job["config"].get("saturn_request_sha256"):
            raise ValueError("worker request differs from admitted request; raw logs retained")
        (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        evidence_surfaces(args.out, report)
    custody = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
               for path in args.out.iterdir() if path.is_file()}
    (args.out / "custody.json").write_text(json.dumps(custody, indent=2) + "\n")
    print(json.dumps({"job_id": args.job_id, "state": job["state"], "report": report is not None}))
    return 0 if job["state"] == "succeeded" and report else 2


if __name__ == "__main__":
    raise SystemExit(main())
