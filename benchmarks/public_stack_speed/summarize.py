"""Recompute public speed tables from collected immutable worker reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def timing_checks(report, job, source, report_sha):
    """Named runtime/custody checks govern publishing this bounded timing claim."""
    custody = job.get("payload_custody") or {}
    payloads = [custody.get(key, {}).get("sha256") for key in ("declared", "sealed")]
    payloads.append((job.get("result") or {}).get("executed_payload", {}).get("sha256"))
    checks = {"job_identity": report["job_id"] == job["job_id"],
              "scheduler_success": job["state"] == "succeeded",
              "mechanics_valid": report["mechanics_status"] == "valid",
              "no_errors": not bool(report.get("errors")),
              "payload_custody": bool(payloads[0]) and len(set(payloads)) == 1,
              "source_audit_identity": source.get("job_id") == job["job_id"]
              and source.get("report_sha256") == report_sha,
              "source_audit_present": bool(source.get("checks")),
              "source_payload_archive": source.get("checks", {}).get("payload_archive") is True,
              "model_spec": report["model"].get("name", report["model"].get("registry_name")) == report["config"]["model"],
              "cpu_threads": report["cpu_threads"] == 4,
              "tf32_disabled": report["tf32"] is False,
              "deterministic": report["deterministic_algorithms"] is True,
              "admitted_plan": report["admitted_plan"] == (job.get("plan") or {})}
    for key, value in source.get("checks", {}).items():
        checks["source/" + key] = value is True
    for field, value in report["config"].items():
        checks["config/" + field] = value == job["config"].get(field)
    plan = report["admitted_plan"]
    if plan:
        for key, value in {"model": report["config"]["model"], "device": report["config"]["device"],
                           "backend": report["config"]["backend"], "max_batch": 1, "threads": 4,
                           "dtype": report["dtype"].removeprefix("torch.")}.items():
            checks["plan/" + key] = plan.get(key) == value
    else:
        # Image workers use a phase-runtime reservation rather than an LM plan.
        checks["phase_geometry"] = all(report["config"].get(key) for key in ("height", "width", "steps"))
        reservation = job.get("reservation") or {}
        checks["phase_reservation"] = all(reservation.get(key, 0) > 0 for key in
                                         ("ram_mb", "vram_mb", "disk_gb", "cpu_threads"))
        profile = report.get("requested_resource_profile")
        if profile:
            checks["phase_profile_binding"] = all(reservation.get(key) == value for key, value in profile.items())
    if report["config"]["device"] == "cuda":
        checks["cuda_admission"] = job["needs"].get("cuda") is True
    checks["no_retry"] = job["config"].get("retry_on_kill") is False
    family = report["model"]["family"]
    ar = family in {"qwen2", "mamba"}
    expected = {"native-eager" if family == "qwen2" else "native-slow" if ar else "native-phase"}
    variants = [("native", "legacy"), ("native", "journal")]
    if family == "qwen2":
        variants += [("virtual", "journal")]
        if report["dtype"] != "torch.bfloat16":
            variants = []
    publications = report["config"].get("mdb_publications", ["legacy", "journal"])
    expected.update("mdb-" + lane + "-" + mode for lane, mode in variants if mode in publications)
    if ar:
        expected.update("saturn-" + mode for mode in report["config"].get("saturn_residencies", ["resident"]))
    labels = [row["configuration"] for row in report["rows"]]
    checks["requested_configurations_completed"] = set(labels) == expected and len(labels) == len(expected)
    for row in report["rows"]:
        label = row["configuration"]
        required = {"outputs_repeat_exact"}
        if label.startswith(("mdb-", "saturn-")):
            required.add("partial_replay_exact")
        if label.startswith("mdb-"):
            required.add("abort_exact")
            if label.endswith("-journal"):
                required.add("noop_exact")
        for field in required:
            checks[label + "/" + field] = row.get(field) is True
        for field in ("outputs_repeat_exact", "partial_replay_exact", "abort_exact", "noop_exact"):
            if field in row:
                checks[label + "/" + field] = row[field] is True
        samples = row["seconds"]
        checks[label + "/samples"] = len(samples) == report["config"]["repeats"] and all(
            math.isfinite(sample) and sample > 0 for sample in samples)
        checks[label + "/median"] = bool(samples) and statistics.median(samples) == row["median_seconds"]
        if label.startswith("mdb-"):
            verification = row.get("verification") or {}
            checks[label + "/byte_custody"] = verification.get("mechanics_status") == "valid" and (
                verification.get("payload_audit") or {}).get("failures") == 0
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    records = []
    for path in args.reports:
        raw = path.read_bytes()
        report = json.loads(raw)
        job = json.loads(path.with_name("job.json").read_text())
        source_path = path.with_name("source-audit-v3.json")
        source = json.loads(source_path.read_text()) if source_path.exists() else {}
        report_sha = hashlib.sha256(raw).hexdigest()
        checks = timing_checks(report, job, source, report_sha)
        records.append({"report_sha256": report_sha, "source_audit": source,
                        "timing_eligible": all(checks.values()), "timing_mechanics_checks": checks,
                        "host": job.get("assigned_host"),
                        "started_ts": job.get("started_ts"), "finished_ts": job.get("finished_ts"),
                        "job_id": report["job_id"], "scheduler_state": job["state"],
                        "payload_sha256": job.get("payload_custody", {}).get("sealed", {}).get("sha256"),
                        "resources": {k: (job.get("result") or {}).get(k) for k in
                                      ("elapsed_s", "peak_rss_mb", "peak_vram_mb")},
                        "report": report})
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps({"schema": "mrun-public-speed-evidence-v1", "runs": records},
                                        indent=2, sort_keys=True, allow_nan=False) + "\n")
    dates = sorted({datetime.fromtimestamp(r["started_ts"], timezone.utc).date().isoformat()
                    for r in records if r.get("started_ts")})
    period = "–".join(dict.fromkeys([dates[0], dates[-1]])) if dates else "date unavailable"
    eligible = [record for record in records if record["timing_eligible"]]
    lines = ["# Measured public-stack speeds", "",
             f"Measured {period} (UTC). These qualify the sealed installed package payloads below on an existing compatible fleet; scheduler/agent services were not redeployed.", "",
             "Fresh installed-package observations from the public mrun, MDB and Saturn stack. "
             "Each row retains three warm wall-time observations and its checkpoint, payload and output identities.", "",
             f"{len(eligible)} finite jobs qualified {sum(len(record['report']['rows']) for record in eligible)} "
             "measured configurations. Applicable repeated outputs, partial replay, abort, journal no-op "
             "and byte-custody checks passed; cross-lane comparisons are recorded separately.", "",
             "Hardware is recorded per job below; four CPU threads; batch one. "
             "Language workloads use the prompt `The capital of France is`; the tables declare each greedy output budget. "
             "Image workloads use seed 17, 512×512, four steps, guidance 1.0 and `A red cube on a white table.`", "",
             "CUDA timings synchronize before and after the call. Loading, package bootstrap, checkpoint/package identity hashing, "
             "adapter construction and final benchmark report/log emission are excluded from warm timings. "
             "MDB durable cut publication and native/MDB output hashing are included. The files are locally cached; "
             "load times are not cold-download measurements. These are short research calls on a shared fleet, "
             "not sustained serving throughput or quality scores.", ""]
    lines += ["| Job | Host | Measured device | Packages | Timing eligible |", "| --- | --- | --- | --- | --- |"]
    for record in records:
        r = record["report"]
        versions = ", ".join(f"{k} {v}" for k, v in r["packages"].items())
        hardware = r["hardware"]
        if r["config"]["device"] == "cpu" and record["source_audit"].get("hardware_audit"):
            hardware = record["source_audit"]["hardware_audit"]["cpu_model"] + " (host identity audited after job)"
        lines.append(f'| `{r["job_id"]}` | {record["host"]} | {hardware} | {versions} | {record["timing_eligible"]} |')
    lines += [""]
    tables = {
        "Native language execution": [], "MDB prepared language continuation": [],
        "Saturn language execution": [], "Image execution": []}
    for record in records:
        report = record["report"]
        config = report["config"]
        if config["repeats"] != 3 or not record["timing_eligible"]:
            continue
        for row in report["rows"]:
            label = row["configuration"]
            model = config["model"]
            dtype = report["dtype"].removeprefix("torch.")
            device = config["device"]
            median = row["median_seconds"]
            spread = f'{min(row["seconds"]):.3f}–{max(row["seconds"]):.3f}'
            if "steps" in config:
                section = "Image execution"
                prep = row.get("prepare_seconds")
                tables[section].append(f'| {model} | {label} | {dtype} | {median:.3f} | {spread} | '
                                       f'{prep:.3f} |' if prep is not None else
                                       f'| {model} | {label} | {dtype} | {median:.3f} | {spread} | cached lookup only |')
            else:
                section = ("MDB prepared language continuation" if label.startswith("mdb-") else
                           "Saturn language execution" if label.startswith("saturn-") else
                           "Native language execution")
                cells = f'| {model} | {device} | {dtype} | {config["tokens"]} | {label} | {median:.3f} | {spread} | {config["tokens"] / median:.2f} |'
                if section == "MDB prepared language continuation":
                    cells += f' {row["prepare_seconds"]:.3f} | {statistics.median(row["restore_seconds"]):.3f} |'
                tables[section].append(cells)
    scopes = {
        "Native language execution": "Includes prompt prefill and the declared cached transitions, retaining the readout after the final emitted token. Each full-vocabulary readout is copied and hashed inside the timed call.",
        "MDB prepared language continuation": "Includes the declared transitions, full-vocabulary readout hashes, durable cuts and the final next-token readout. Prompt preparation and restoration are separate. Native and virtual Qwen are different numerical lanes.",
        "Saturn language execution": "Includes session creation/prompt execution and the declared greedy token commits, returning token IDs without per-token readout hashing. It stops at that commit, one readout earlier than the native/MDB timing scope; these are distinct research operations.",
        "Image execution": "Native phase rows include a warm conditioning-cache lookup, seeded sampling, denoising, VAE and the RGB copy/digest. The unreported warm-up populates that cache, so warm rows do not remeasure text-encoder computation. MDB rows time the prepared schedule with durable cuts and the RGB digest; preparation includes its first conditioning computation and is reported separately. Adding it gives a first-preparation call, with a different cache boundary from the native warm row."}
    for section, rows in tables.items():
        lines += ["## " + section, "", scopes[section], ""]
        if section == "Image execution":
            lines += ["| Model | Configuration | Precision | Median s/image | Range s | Preparation s |",
                      "| --- | --- | --- | ---: | ---: | ---: |"]
        else:
            header = "| Model | Device | Precision | Output tokens | Configuration | Median s | Range s | Output tokens/s |"
            separator = "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |"
            if section == "MDB prepared language continuation":
                header += " Preparation s | Restore median s |"
                separator += " ---: | ---: |"
            lines += [header, separator]
        lines += rows + [""]
    lines += ["## Execution checks", "",
              "The worker checks repeated outputs, typed partial-cut replay, residual preview/abort, "
              "journal-mode no-op commits and byte custody. It also retains a committed 0.9 residual-scale "
              "candidate and its unchanged-parent reference. Candidate differences are observations of the "
              "whole continuation; later autoregressive inputs can diverge. This does not certify a semantic circuit.", "",
              "| Job | Model | Device / precision | Status | Loading s | Peak RSS, reported MB | Peak VRAM, reported MB |",
              "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for record in records:
        r = record["report"]
        lines.append(f'| `{r["job_id"]}` | {r["config"]["model"]} | {r["config"]["device"]} / '
                     f'{r["dtype"].removeprefix("torch.")} | {record["scheduler_state"]}; '
                     f'{r["mechanics_status"]} | {r["load_seconds"]:.3f} | '
                     f'{record["resources"]["peak_rss_mb"]} | {record["resources"]["peak_vram_mb"]} |')
    lines += ["", "MDB Qwen requires BF16. FP32 native/Saturn rows do not imply FP32 MDB support. "
              "Same-lane exact replay is checked independently of cross-lane logits or token agreement.", "",
              "| Job | Configuration | Native output-token agreement | Native readout/image bytes | Same-lane replay |",
              "| --- | --- | --- | --- | --- |"]
    for record in records:
        for row in record["report"]["rows"]:
            lines.append(f'| `{record["job_id"]}` | {row["configuration"]} | {row.get("native_token_exact", "—")} | '
                         f'{row.get("native_output_exact", "—")} | {row.get("partial_replay_exact", "—")} |')
    lines += ["",
              "Full per-row preparation, restoration, setup, memory, publication costs, native comparisons, "
              "interventions, errors and model hashes are in the [evidence record](../benchmarks/public_stack_speed/evidence.json). "
              "The collector retains immutable raw reports and their SHA256 locally; model/state archives stay outside Git.", "",
              "## Calibration and stopped attempts", "",
              "The [attempt ledger](../benchmarks/public_stack_speed/attempts.json) retains cancelled requests, "
              "the BF16-only MDB refusal, CUDA startup sizing, the FP32 streaming RAM stop and harness corrections. "
              "Their partial measurements remain evidence and are excluded from successful three-repeat tables. "
              "The HF CUDA process allowance and the workload-matched streaming reservation corrected the two observed guard stops; guards and automatic-retry refusal remained active.", "",
              "## Reproduce", "", "See the [benchmark protocol](../benchmarks/public_stack_speed/README.md) "
              "for installation, offline worker configuration, launch and direct-job collection.", ""]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
