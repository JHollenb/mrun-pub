"""Command-line entry point."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

def _forward_records(engine: Any, prompts: list[str]) -> list[dict[str, Any]]:
    ids_list = engine.encode(prompts)
    records = []
    for prompt, ids in zip(prompts, ids_list, strict=True):
        logits = engine.logits(ids)[-1].detach().float().numpy()
        top = np.argsort(-logits)[:5]
        records.append(
            {
                "prompt": prompt,
                "n_tok": int(len(ids)),
                "argmax": int(top[0]),
                "next_token": engine.tokenizer.decode([int(top[0])])
                if hasattr(engine, "tokenizer")
                else "",
                "top5": [int(x) for x in top],
                "logits_last": logits.astype(np.float32),
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "fingerprint":
        from .fingerprint.cli import main as fingerprint_main
        return fingerprint_main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "olmoe":
        from .engine.olmoe_cuda import main as olmoe_main

        return olmoe_main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "qwen3-moe":
        from .engine.qwen3_moe_cli import main as qwen3_moe_main

        return qwen3_moe_main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "decompile":
        from .decompiler.cli import main as decompiler_main

        return decompiler_main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "inference":
        from .inference.cli import main as inference_main

        return inference_main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "science":
        from .science.cli import main as science_main

        return science_main(raw_argv[1:])

    parser = argparse.ArgumentParser(prog="mrun")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="run a lightweight smoke test")
    smoke.add_argument("--no-model", action="store_true")
    smoke.add_argument("--model", default="distilgpt2")

    run = sub.add_parser("run", help="run a prompt through a model")
    run.add_argument("model")
    run.add_argument(
        "--backend",
        default="auto",
        help=(
            "engine backend (auto, hf, paged, paged-fp32, dense-qstore-cuda, "
            "olmoe-cuda, qwen3-moe-cuda, mlx, ...)"
        ),
    )
    run.add_argument("--prompt", action="append", required=True)
    run.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="generate this many tokens instead of returning next-token logits",
    )
    run.add_argument("--store-dir", type=Path)
    run.add_argument(
        "--expert-codec",
        choices=("fp8", "w4"),
        default=None,
        help="Qwen3 MoE expert-store codec; omitted preserves the backend default (FP8)",
    )
    run.add_argument("--cache-mb", type=float)
    run.add_argument("--max-active-pages", type=int)
    run.add_argument(
        "--host-cache-mb",
        type=float,
        default=None,
        help="Qwen3 MoE host-RAM expert-page tier budget",
    )
    run.add_argument(
        "--warm-host",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="prewarm the Qwen3 MoE host page tier",
    )
    run.add_argument(
        "--route-prefetch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable Qwen3 MoE route-locality prefetch",
    )
    run.add_argument(
        "--cache-policy",
        choices=("global-lru-v1", "layer-frequency-lru-v1"),
        default=None,
        help="Qwen3 MoE CUDA expert-page cache policy",
    )
    run.add_argument(
        "--page-binding-policy",
        choices=("compact-copy-v1", "slot-indirect-v1"),
        default=None,
        help="Qwen3 MoE routed-page binding policy",
    )
    run.add_argument(
        "--prefill-page-policy",
        choices=("cache-fill-v1", "transient-frequency-v1"),
        default=None,
        help="Qwen3 MoE prefill page-admission policy",
    )
    run.add_argument(
        "--route-reduction-policy",
        choices=("atomic-index-add-v1", "stable-route-rank-v1"),
        default=None,
        help="Qwen3 MoE expert-reduction numerical ABI",
    )
    run.add_argument(
        "--w4-arithmetic-policy",
        choices=("w4-g128-predot-bf16-v1", "w4-g128-postscale-bf16-v1"),
        default=None,
        help="Qwen3 MoE W4 grouped-GEMM numerical ABI",
    )
    run.add_argument("--verify-store-content", action="store_true")
    run.add_argument("--out", type=Path)

    est = sub.add_parser("estimate", help="estimate CPU/RAM cost before running")
    est.add_argument("model")
    est.add_argument("--experiment", default=None, help="match wall/cpu to this experiment")
    est.add_argument("--dtype", default="float32")
    est.add_argument("--outputs", type=Path, default=Path("outputs"), help="run-history root")
    est.add_argument("--json", action="store_true", help="emit the full estimate as JSON")
    est.add_argument(
        "--host",
        default=None,
        help="preview the RunPlan for a registered fleet host (e.g. beast)",
    )
    est.add_argument("--task", default="forward", help="task family (forward/recorder/train)")

    lb = sub.add_parser("leaderboard", help="speed/size/compute rows recorded by engine reports")
    lb.add_argument("--model", default=None, help="substring filter on model name")
    lb.add_argument(
        "--sort",
        default="tok_per_s",
        choices=(
            "tok_per_s",
            "probes_per_s",
            "compute_units_total",
            "wall_s",
            "peak_rss_mb",
        ),
    )
    lb.add_argument("--limit", type=int, default=40)
    lb.add_argument("--json", action="store_true", help="emit raw rows")

    bs = sub.add_parser("build-store", help="build a paged QStore for a model")
    bs.add_argument("model")
    qgroup = bs.add_mutually_exclusive_group()
    qgroup.add_argument(
        "--fp32",
        action="store_true",
        help="build an explicit lossless float32 paged store instead",
    )
    qgroup.add_argument("--int4", action="store_true", help="build a group-wise int4 store instead")
    qgroup.add_argument(
        "--int2", action="store_true", help="build an experimental ternary int2 store instead"
    )
    qgroup.add_argument(
        "--int3", action="store_true", help="build a fused group-wise int3 store instead"
    )
    bs.add_argument("--out", type=Path, default=None, help="store root (default: MRUN_STORES_ROOT)")

    dbs = sub.add_parser(
        "build-diffusion-store",
        help="build a demand-paged int8 store for a Diffusers transformer component",
    )
    dbs.add_argument("component_dir", type=Path)
    dbs.add_argument("output_dir", type=Path)
    dbs.add_argument("--model-name", required=True)
    dbs.add_argument("--pipeline-class", required=True)
    dbs.add_argument("--validate-component", action="store_true")

    wp = sub.add_parser(
        "workplan",
        help="compile, persist, and benchmark a CPU-paged static WorkPlan",
    )
    wp.add_argument("model")
    wp.add_argument("--prompt", action="append", required=True)
    wp_quant = wp.add_mutually_exclusive_group()
    wp_quant.add_argument("--int4", action="store_true")
    wp_quant.add_argument("--int3", action="store_true")
    wp_quant.add_argument("--int2", action="store_true")
    wp.add_argument(
        "--output-contract",
        choices=(
            "full_logits",
            "last_token_logits",
            "selected_token_rows",
            "candidate_argmax_and_margin",
            "loss_only",
            "hidden_state_only",
        ),
        default="last_token_logits",
    )
    wp.add_argument(
        "--output-row",
        action="append",
        type=int,
        default=[],
        help="vocabulary row for selected_token_rows; repeat as needed",
    )
    wp.add_argument(
        "--candidate-token-id",
        action="append",
        type=int,
        default=[],
        help="candidate ID applied to every prompt row; repeat at least twice",
    )
    wp.add_argument("--warmup", type=int, default=1)
    wp.add_argument("--trials", type=int, default=3)
    wp.add_argument(
        "--verify-full-head",
        action="store_true",
        help="also compare an optimized output contract with the slower full-head path",
    )
    wp.add_argument("--artifact-dir", type=Path)
    wp.add_argument("--out", type=Path)
    wp.add_argument(
        "--peak-gops",
        type=float,
        help="optional explicit design assumption for roofline cost estimation",
    )
    wp.add_argument(
        "--bandwidth-gbps",
        type=float,
        help="optional explicit design assumption for roofline cost estimation",
    )
    wp.add_argument("--launch-us", type=float, default=0.0)
    wp.add_argument("--boundary-us", type=float, default=0.0)

    campaign = sub.add_parser(
        "campaign",
        help="compile and benchmark overlapping candidate readouts over one static input",
    )
    campaign.add_argument("model")
    campaign.add_argument("--prompt", required=True)
    campaign.add_argument(
        "--backend",
        choices=("auto", "paged", "dense-qstore-cuda"),
        default="auto",
        help=(
            "candidate runtime; auto uses CUDA Graph only when an exact installed "
            "hardware promotion passes identity, shape, amortization, and VRAM gates"
        ),
    )
    campaign_quant = campaign.add_mutually_exclusive_group()
    campaign_quant.add_argument("--int4", action="store_true")
    campaign_quant.add_argument("--int3", action="store_true")
    campaign_quant.add_argument("--int2", action="store_true")
    campaign.add_argument(
        "--query",
        action="append",
        required=True,
        help="logical readout as QUERY_ID=TOKEN_ID,TOKEN_ID,...; repeat at least twice",
    )
    campaign.add_argument("--warmup", type=int, default=1)
    campaign.add_argument("--trials", type=int, default=5)
    campaign.add_argument(
        "--expected-replays",
        type=int,
        default=None,
        help=(
            "expected lifetime replays for CUDA Graph setup amortization; defaults to "
            "the benchmark correctness + warmup + trial executions"
        ),
    )
    campaign.add_argument(
        "--compact-cache-mb",
        type=float,
        default=0.0,
        help="dense CUDA compact-page cache budget; ignored by paged",
    )
    campaign.add_argument(
        "--compute-dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="dense CUDA projection dtype",
    )
    campaign_graph = campaign.add_mutually_exclusive_group()
    campaign_graph.add_argument(
        "--cuda-graph",
        dest="cuda_graph",
        action="store_true",
        help=(
            "require real CUDA Graph capture/replay for the dense CUDA compiled leg; "
            "fails closed instead of falling back to eager execution"
        ),
    )
    campaign_graph.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="disable CUDA Graph selection when --backend auto is used",
    )
    campaign.set_defaults(cuda_graph=None)
    campaign.add_argument(
        "--minimum-improvement-percent",
        type=float,
        default=1.0,
        help="minimum paired independent/compiled speedup percentage for the evidence gate",
    )
    campaign.add_argument(
        "--factorial-ablation",
        action="store_true",
        help=(
            "also measure full/selected output by independent/compiled-union "
            "as a paired 2x2 ablation"
        ),
    )
    campaign.add_argument(
        "--campaign-artifact",
        type=Path,
        help="optional path for the serialized, content-addressed campaign",
    )
    campaign.add_argument("--out", type=Path)

    campaign_replay = sub.add_parser(
        "campaign-replay",
        help="validate, prepare once, and replay a serialized candidate campaign",
    )
    campaign_replay.add_argument("artifact", type=Path)
    campaign_replay.add_argument("--prompt", required=True)
    campaign_replay.add_argument("--replays", type=int, default=1)
    campaign_replay.add_argument(
        "--compact-cache-mb",
        type=float,
        default=0.0,
        help="dense CUDA compact-page cache budget",
    )
    campaign_replay.add_argument(
        "--compute-dtype",
        choices=("bf16", "fp16"),
        default=None,
        help=(
            "optional dense CUDA dtype assertion; replay derives the dtype from "
            "the artifact and refuses a mismatch"
        ),
    )
    campaign_replay.add_argument("--out", type=Path)

    sciencegraph = sub.add_parser(
        "sciencegraph",
        help="execute or benchmark a serialized Intervention ScienceGraph",
    )
    sciencegraph.add_argument("model")
    sciencegraph.add_argument("artifact", type=Path)
    sciencegraph.add_argument("--numerical-contract", required=True)
    sciencegraph.add_argument("--store-dir", type=Path)
    sciencegraph.add_argument("--device", default="cpu")
    sciencegraph.add_argument(
        "--fp32",
        action="store_true",
        help="use the explicit lossless paged-fp32 store (never selected implicitly)",
    )
    sciencegraph.add_argument(
        "--payload",
        action="append",
        default=[],
        help="authenticated tensor operand as PAYLOAD_ID=PATH.npy; repeat as needed",
    )
    sciencegraph.add_argument("--max-branch-batch", type=int)
    sciencegraph.add_argument("--benchmark", action="store_true")
    sciencegraph.add_argument("--warmup", type=int, default=1)
    sciencegraph.add_argument("--trials", type=int, default=5)
    sciencegraph.add_argument("--atol", type=float, default=2e-4)
    sciencegraph.add_argument("--minimum-speedup", type=float, default=1.05)
    sciencegraph.add_argument("--out", type=Path)

    olmoe = sub.add_parser(
        "olmoe",
        help="run the reusable OLMoE FP8 build, quality, or decode harness",
    )
    olmoe.add_argument(
        "olmoe_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to mrun.engine.olmoe_cuda",
    )

    qwen3_moe = sub.add_parser(
        "qwen3-moe",
        help="build, validate, quality-gate, or run paged Qwen3 MoE CUDA",
    )
    qwen3_moe.add_argument(
        "qwen3_moe_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to mrun.engine.qwen3_moe_cli",
    )

    sm = sub.add_parser(
        "submit",
        help="intent-first fleet submit: mrun picks host, settings and reservation",
    )
    sm.add_argument("-e", "--experiment", default=None, help="experiment name (default: cmd stem)")
    sm.add_argument("--model", default=None, help="model id — enables policy planning")
    sm.add_argument("--task", default="forward")
    sm.add_argument("--payload", default=None, help="directory/file to ship as the payload")
    sm.add_argument("--prefer", default=None, help="soft host preference (migrates when starved)")
    sm.add_argument("--pin", default=None, help="HARD host pin (job runs there or not at all)")
    sm.add_argument(
        "--ram",
        type=float,
        default=None,
        help="declared RAM MB (subject to server clamps)",
    )
    sm.add_argument("--vram", type=float, default=None, help="declared VRAM MB")
    sm.add_argument("--gpu", action="store_true", help="require a cuda host")
    sm.add_argument("--priority", type=int, default=0)
    sm.add_argument("--note", default=None, help="operator context shown by mrun why")
    sm.add_argument("--timeout", type=float, default=None, help="job timeout seconds")
    sm.add_argument("--env-alias", default=None)
    sm.add_argument("--detach", action="store_true", help="print job id and exit (no log stream)")
    sm.add_argument("--no-retry", action="store_true", help="disable auto-retry-on-kill")
    sm.add_argument(
        "--preflight",
        choices=("off", "warn", "strict"),
        default="off",
        help="client-side semantic checks before the server's capacity preflight",
    )
    sm.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run (prefix with --)")

    pf = sub.add_parser(
        "preflight",
        help="client-side admission checks: path closure, hashed-root hygiene, "
        "verifier dry-run, workload geometry, reservation sanity",
    )
    pf.add_argument("payload", nargs="?", default=None, help="payload dir (omit for cmd-only)")
    pf.add_argument("--experiment", required=True)
    pf.add_argument("--config", default=None, help="JSON file with the job config")
    pf.add_argument("--ram-mb", type=float, default=None)
    pf.add_argument("--vram-mb", type=float, default=None)
    pf.add_argument("--model", default=None)
    pf.add_argument("--task", default=None, help="task family for generalized history")
    pf.add_argument("--offline", action="store_true", help="skip scheduler-backed checks")
    pf.add_argument("--json", action="store_true", help="print the full receipt JSON")
    pf.add_argument("--receipt-dir", default=None)
    pf.add_argument("--cmd", nargs=argparse.REMAINDER, required=True, help="job command")

    at = sub.add_parser("attach", help="re-attach to a fleet job and stream its logs")
    at.add_argument("job_id")

    ca = sub.add_parser("cancel", help="cancel a fleet job")
    ca.add_argument("job_id")
    ca.add_argument("--reason", default=None, help="durable cancellation reason")

    sub.add_parser("hosts", help="show fleet hosts, telemetry and committed reservations")

    jl = sub.add_parser("jobs", help="list fleet jobs")
    jl.add_argument("--state", default=None)

    why = sub.add_parser("why", help="explain a fleet job's state and recent events")
    why.add_argument("job_id")
    why.add_argument("--events", type=int, default=20, help="number of recent events to show")

    mo = sub.add_parser("models", help="warm-model map: which host has which bytes")
    mo.add_argument("--host", default=None)

    sub.add_parser("ui", help="open the fleet web UI in a browser")

    sub.add_parser("agent", help="run the host agent (foreground)")
    sub.add_parser("server", help="run the scheduler server (foreground)")

    args = parser.parse_args(raw_argv)
    if args.command == "smoke":
        return _smoke(args)
    if args.command == "run":
        return _run(args)
    if args.command == "estimate":
        return _estimate(args)
    if args.command == "leaderboard":
        return _leaderboard(args)
    if args.command == "build-store":
        return _build_store(args)
    if args.command == "build-diffusion-store":
        return _build_diffusion_store(args)
    if args.command == "workplan":
        return _workplan(args)
    if args.command == "campaign":
        return _campaign(args)
    if args.command == "campaign-replay":
        return _campaign_replay(args)
    if args.command == "sciencegraph":
        return _sciencegraph(args)
    if args.command == "olmoe":
        from .engine.olmoe_cuda import main as olmoe_main

        return olmoe_main(args.olmoe_args)
    if args.command == "qwen3-moe":
        from .engine.qwen3_moe_cli import main as qwen3_moe_main

        return qwen3_moe_main(args.qwen3_moe_args)
    if args.command == "submit":
        return _submit_cli(args)
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "attach":
        return _attach(args)
    if args.command == "cancel":
        return _cancel(args)
    if args.command == "hosts":
        return _hosts(args)
    if args.command == "jobs":
        return _jobs(args)
    if args.command == "why":
        return _why(args)
    if args.command == "models":
        return _models(args)
    if args.command == "ui":
        import webbrowser

        from .client.api import Api

        api = Api()
        url = api._resolve() + "/ui"
        print(url)
        webbrowser.open(url)
        return 0
    if args.command == "agent":
        from .agent.main import main as agent_main

        agent_main()
        return 0
    if args.command == "server":
        from .server.__main__ import main as server_main

        server_main()
        return 0
    return 2


def _preflight(args: argparse.Namespace) -> int:
    import json as _json

    from .client.preflight import run_preflight, write_receipt
    from .io import read_json

    api = None
    if not args.offline:
        from .client.api import Api

        api = Api()
    config = read_json(args.config) if args.config else {}
    reservation: dict[str, float] = {}
    if args.ram_mb is not None:
        reservation["ram_mb"] = args.ram_mb
    if args.vram_mb is not None:
        reservation["vram_mb"] = args.vram_mb
    receipt = run_preflight(
        experiment=args.experiment,
        cmd=list(args.cmd or []),
        config=config,
        payload=args.payload,
        reservation=reservation or None,
        model=args.model or config.get("model"),
        task_family=args.task,
        api=api,
    )
    path = write_receipt(
        receipt, Path(args.receipt_dir) if args.receipt_dir else None
    )
    for check in receipt.checks:
        print(f"{check.status:>7}  {check.check}: {check.detail}")
    print(f"preflight: {receipt.verdict} (receipt {path})")
    if args.json:
        print(_json.dumps(receipt.as_dict(), indent=2, sort_keys=True))
    return 0 if receipt.verdict == "passed" else 1


@contextlib.contextmanager
def _exclusive_attach_lock(
    job_id: str, *, lock_root: Path | None = None
) -> Iterator[bool]:
    """Allow only one CLI log follower for a job from this user account."""
    root = lock_root or Path.home() / ".cache" / "mrun" / "attach"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    digest = hashlib.sha256(job_id.encode()).hexdigest()[:24]
    path = root / f"{digest}.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield True
    finally:
        os.close(fd)


def _attach(args: argparse.Namespace) -> int:
    from .client.api import ApiError
    from .client.submit import Detached, attach

    with _exclusive_attach_lock(args.job_id) as acquired:
        if not acquired:
            print(f"mrun: attach already running for {args.job_id} on this host")
            return 0
        try:
            res = attach(args.job_id)
        except Detached:
            return 0
        except ApiError as exc:
            print(f"mrun: {exc}")
            return 1
        return 0 if res.ok else 1


def _submit_cli(args: argparse.Namespace) -> int:
    from .client.api import ApiError
    from .client.preflight import PreflightRejected
    from .client.submit import Detached, launch

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("mrun submit: no command given (usage: mrun submit [opts] -- CMD...)")
        return 2
    try:
        out = launch(
            cmd,
            experiment=args.experiment,
            model=args.model,
            task=args.task,
            payload=args.payload,
            prefer_host=args.prefer,
            pin=args.pin,
            ram_mb=args.ram,
            vram_mb=args.vram,
            gpu=args.gpu,
            env_alias=args.env_alias,
            priority=args.priority,
            note=args.note,
            timeout_s=args.timeout,
            detach=args.detach,
            retry_on_kill=not args.no_retry,
            preflight=(
                False if args.preflight == "off" else
                True if args.preflight == "strict" else "warn"
            ),
        )
    except PreflightRejected as exc:
        # Client-side semantic rejection — the job never reached the scheduler.
        print(f"mrun submit: {exc}")
        return 1
    except ApiError as exc:
        # a preflight 422 carries per-host reasons in the body — show them verbatim
        print(f"mrun submit: {exc}")
        return 1
    except Detached:
        return 0
    if args.detach:
        print(out)
        return 0
    return 0 if out.ok else 1


def _cancel(args: argparse.Namespace) -> int:
    from .client.api import Api, ApiError

    try:
        body = {"reason": args.reason} if args.reason is not None else None
        print(Api().json("POST", f"/api/jobs/{args.job_id}/cancel", json_body=body))
    except ApiError as exc:
        print(f"mrun: {exc}")
        return 1
    return 0


def _hosts(args: argparse.Namespace) -> int:
    import time as _time

    from .client.api import Api

    for h in Api().json("GET", "/api/hosts"):
        tel = h.get("telemetry") or {}
        age = f"{_time.time() - tel['ts']:.0f}s ago" if tel.get("ts") else "never"
        eta = h.get("eta_s")
        print(
            f"{h['name']:<12} "
            f"ram {tel.get('ram_free_mb', 0):>8.0f}/{h['ram_total_mb']:>8.0f}MB free"
            f"  committed {h.get('committed_ram_mb', 0):>7.0f}MB"
            f"  jobs {len(h.get('active_jobs') or []):d}"
            f"  eta {'?' if eta is None else f'{eta:.0f}s'}"
            f"  caps {','.join(k for k, v in (h.get('caps') or {}).items() if v)}"
            f"  seen {age}"
        )
    return 0


def _jobs(args: argparse.Namespace) -> int:
    from .client.api import Api

    path = "/api/jobs" + (f"?state={args.state}" if args.state else "")
    for j in Api().json("GET", path):
        print(
            f"{j['job_id']}  {j['state']:<10} {j['experiment']:<28} "
            f"host={j.get('assigned_host') or '-':<8} "
            f"ram={j['reservation'].get('ram_mb', 0):.0f}MB({j['reservation'].get('source')})"
        )
    return 0


def _why(args: argparse.Namespace) -> int:
    import json
    import time as _time

    from .client.api import Api, ApiError

    api = Api()
    try:
        job = api.json("GET", f"/api/jobs/{args.job_id}")
        events = api.json("GET", f"/api/jobs/{args.job_id}/events?limit={args.events}")
    except ApiError as exc:
        print(f"mrun: {exc}")
        return 1

    res = job.get("reservation") or {}
    print(
        f"{job['job_id']}  {job['state']}  {job['experiment']}  "
        f"host={job.get('assigned_host') or '-'}"
    )
    if job.get("status_detail"):
        print(f"detail: {job['status_detail']}")
    print(
        "reservation: "
        f"ram={float(res.get('ram_mb') or 0):.0f}MB "
        f"vram={float(res.get('vram_mb') or 0):.0f}MB "
        f"cpu={int(res.get('cpu_threads') or 0)} "
        f"source={res.get('source') or '-'}"
    )
    if job.get("needs"):
        print(f"needs: {json.dumps(job['needs'], sort_keys=True)}")
    if job.get("plan"):
        plan = job["plan"]
        print(
            "plan: "
            f"{plan.get('backend') or '-'} / {plan.get('dtype') or '-'} / "
            f"{plan.get('device') or '-'}"
        )
    if job.get("result"):
        result = job["result"]
        print(
            "result: "
            f"peak_rss={result.get('peak_rss_mb')}MB "
            f"peak_vram={result.get('peak_vram_mb')}MB "
            f"elapsed={result.get('elapsed_s')}s"
        )
        backend = result.get("os_memory_limit_backend")
        if backend:
            # honesty about the never-OOM guarantee: cgroup is a strict boundary,
            # anything else is 1s-sampled advisory enforcement (macOS)
            strict = "strict (kernel boundary)" if "cgroup" in str(backend) else \
                "advisory (1s-sampled tree RSS)"
            print(f"memory enforcement: {backend} — {strict}")
        failure = result.get("failure")
        if isinstance(failure, dict):
            print("failure diagnostics:")
            print(json.dumps(failure, indent=2, sort_keys=True))
        diagnostics = result.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics.get("phases"):
            phases = [str(item.get("name")) for item in diagnostics["phases"]]
            print("execution phases: " + " -> ".join(phases))
    if job.get("config", {}).get("_plan_error"):
        print(f"planning: FAILED client-side — {job['config']['_plan_error']}")
    meta = dict(job.get("meta") or {})
    if meta.get("queue_note"):
        print(f"queue note: {meta.pop('queue_note')}")
    if meta.get("cancellation_reason"):
        print(f"cancellation reason: {meta.pop('cancellation_reason')}")
    if meta:
        print(f"meta: {json.dumps(meta, sort_keys=True)}")

    print(f"recent events (newest first, {len(events)}):")
    now = _time.time()
    for ev in events:
        age = "?"
        if ev.get("ts") is not None:
            age = f"{now - float(ev['ts']):.0f}s ago"
        fields = [str(ev.get("kind") or "?")]
        if ev.get("host"):
            fields.append(f"host={ev['host']}")
        if ev.get("state"):
            fields.append(f"state={ev['state']}")
        if ev.get("reason"):
            fields.append(f"reason={ev['reason']}")
        if ev.get("detail"):
            fields.append(f"detail={ev['detail']}")
        print(f"  {age:<8} " + " ".join(fields))
    return 0


def _models(args: argparse.Namespace) -> int:
    from .client.api import Api

    path = "/api/models" + (f"?host={args.host}" if args.host else "")
    rows = Api().json("GET", path)
    if not rows:
        print("no inventory reported yet")
        return 0
    for r in rows:
        size = f"{(r.get('bytes') or 0) / 1e9:.1f}GB"
        print(f"{r['host']:<10} {r['model']:<28} {r['kind']:<8} {size:>8}  {r.get('path') or ''}")
    return 0


def _smoke(args: argparse.Namespace) -> int:
    from .paths import artifact_root, models_root, stores_root

    print("mrun import ok")
    print(f"models_root={models_root()}")
    print(f"artifact_root={artifact_root()}")
    print(f"stores_root={stores_root()}")
    if args.no_model:
        return 0
    from .engine import open_engine

    with open_engine(args.model, backend="hf") as engine:
        rows = _forward_records(engine, ["The capital of France is"])
    print(rows[0]["next_token"])
    return 0


def _run(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .engine import open_engine
    from .io import write_json

    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    selected_backend = args.backend
    selected_plan = None
    if args.backend == "auto":
        from .agent.hostinfo import model_inventory
        from .policy import HostCaps, apply_plan
        from .selector import select_run_plan

        host = HostCaps.detect()
        try:
            local_artifacts = model_inventory(host.name)
        except Exception as exc:  # noqa: BLE001 — inventory is an optimization, not a gate
            print(f"mrun: WARNING local artifact inventory unavailable ({exc})", file=sys.stderr)
            local_artifacts = None
        selected = select_run_plan(
            args.model,
            host=host,
            artifacts=local_artifacts,
            backend="auto",
        )
        selected_plan = selected.plan
        selected_backend = selected_plan.backend
        apply_plan(selected_plan)

    engine_kwargs: dict[str, Any] = (
        selected_plan.engine_kwargs() if selected_plan is not None else {}
    )
    for argument, key in (
        (args.store_dir, "store_dir"),
        (args.expert_codec, "expert_codec"),
        (args.cache_mb, "cache_mb"),
        (args.max_active_pages, "max_active_pages"),
        (args.host_cache_mb, "host_cache_mb"),
        (args.warm_host, "warm_host"),
        (args.route_prefetch, "route_prefetch"),
        (args.cache_policy, "cache_policy"),
        (args.page_binding_policy, "page_binding_policy"),
        (args.prefill_page_policy, "prefill_page_policy"),
        (args.route_reduction_policy, "route_reduction_policy"),
        (args.w4_arithmetic_policy, "w4_arithmetic_policy"),
    ):
        if argument is not None:
            engine_kwargs[key] = argument
    if args.verify_store_content:
        engine_kwargs["verify_store_content"] = True

    with open_engine(
        args.model,
        backend=selected_backend,
        **engine_kwargs,
    ) as engine:
        capabilities = engine.capabilities()
        if args.max_new_tokens:
            if not capabilities.generation:
                raise ValueError(
                    f"backend {selected_backend!r} does not advertise generation support"
                )
            generate_batch = getattr(engine, "generate_batch", None)
            if callable(generate_batch):
                generated = generate_batch(
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                generated = [
                    engine.generate(prompt, max_new_tokens=args.max_new_tokens)
                    for prompt in args.prompt
                ]
            rows = [
                {
                    "prompt": prompt,
                    "generated_ids": tokens,
                    "generated_text": engine.tokenizer.decode(
                        tokens,
                        skip_special_tokens=True,
                    ),
                }
                for prompt, tokens in zip(args.prompt, generated, strict=True)
            ]
        else:
            rows = _forward_records(engine, args.prompt)
        runtime_report = getattr(engine, "runtime_report", None)
        runtime = runtime_report() if callable(runtime_report) else None
    payload: dict[str, Any] = {
        "model": args.model,
        "backend": selected_backend,
        "capabilities": asdict(capabilities),
        "rows": rows,
    }
    if selected_plan is not None:
        payload["plan"] = selected_plan.as_dict()
    if runtime is not None:
        payload["runtime"] = runtime
    if args.out:
        write_json(args.out, payload)
    else:
        for row in rows:
            if args.max_new_tokens:
                print(f"{row['prompt']!r} -> {row['generated_text']!r}")
            else:
                print(f"{row['prompt']!r} -> {row['next_token']!r} top5={row['top5']}")
    return 0


def _estimate(args: argparse.Namespace) -> int:
    from .estimate import estimate_resources
    from .io import stable_json

    try:
        return _estimate_inner(args, estimate_resources, stable_json)
    except ValueError as exc:
        print(f"mrun: {exc}")
        return 1


def _estimate_inner(args: argparse.Namespace, estimate_resources, stable_json) -> int:
    if args.host:
        from .client.api import Api
        from .policy import HostCaps
        from .selector import select_run_plan

        hosts = {h["name"]: h for h in Api().json("GET", "/api/hosts")}
        h = hosts.get(args.host)
        if h is None:
            print(f"mrun: unknown fleet host {args.host!r} (known: {', '.join(hosts)})")
            return 1
        caps = h.get("caps") or {}
        selection = select_run_plan(
            args.model,
            args.task,
            host=HostCaps(
                name=h["name"],
                ram_mb=float(h.get("ram_total_mb") or 0.0),
                vram_mb=float(h.get("vram_total_mb") or 0.0),
                has_cuda=bool(caps.get("cuda")),
                has_mps=bool(caps.get("mps")),
                has_ane=bool(caps.get("ane")),
                cpus=int(h.get("cpu_threads") or 8),
            ),
            artifacts=h.get("models") if isinstance(h.get("models"), list) else None,
        )
        plan = selection.plan
        if args.json:
            print(stable_json(plan.as_dict()))
            return 0
        print(
            f"{args.model} on {args.host}: {plan.backend}/{plan.dtype}/{plan.device} "
            f"batch={plan.max_batch} ram_limit={plan.ram_limit_mb:.0f}MB"
        )
        for r in plan.reasons:
            print(f"  - {r}")
        return 0
    est = estimate_resources(
        {"model": args.model, "dtype": args.dtype},
        name=args.experiment,
        dtype=args.dtype,
        outputs_root=args.outputs,
    )
    if args.json:
        print(stable_json(est))
        return 0

    mem = est["memory"]
    basis = mem["basis"]
    params_m = mem["params"] / 1e6
    print(f"{args.model}  ({params_m:.0f}M params, {args.dtype}, source={mem['param_source']})")
    print(
        f"  RAM  ~= {est['est_rss_mb']:.0f} MB  "
        f"(weights {mem['weights_mb']:.0f} + overhead {mem['overhead_mb']:.0f}; {basis})"
    )
    wall = est["wall_s_estimate"]
    cpu = est["cpu_s_estimate"]
    if wall is not None:
        print(f"  wall ~= {wall:.0f} s   cpu ~= {cpu:.0f} s   (from history)")
    else:
        print("  wall/cpu: no comparable run in history yet")
    hist = est["history"]
    near = hist["nearest"]
    if near is not None:
        diff = ", ".join(near["differing_keys"]) or "exact match"
        ref = f"{near['experiment']}/{near['run_id']}"
        print(f"  nearest: {ref}  sim={near['similarity']}  diff=[{diff}]")
    law = "refit" if hist["calibrated"] else "first-principles"
    print(f"  history: {hist['n_runs']} run(s), memory law {law}")
    for note in est["notes"]:
        print(f"  note: {note}")
    return 0


def _workplan(args: argparse.Namespace) -> int:
    from .compiler import (
        CompilationArtifactStore,
        CostAssumptions,
        EvidenceRecord,
        OutputContract,
        benchmark_eager_plan,
        benchmark_output_slicing,
        compile_work_plan,
        evaluate_promotion,
        evidence_payload_sha256,
        trace_qstore_graph_execution,
        verify_output_pushdown_parity,
    )
    from .engine import open_engine
    from .io import stable_json, write_json

    if (args.peak_gops is None) != (args.bandwidth_gbps is None):
        print("mrun: --peak-gops and --bandwidth-gbps must be supplied together")
        return 1
    assumptions = None
    if args.peak_gops is not None and args.bandwidth_gbps is not None:
        assumptions = CostAssumptions(
            peak_compute_ops_per_s=args.peak_gops * 1e9,
            memory_bandwidth_bytes_per_s=args.bandwidth_gbps * 1e9,
            launch_overhead_s=args.launch_us * 1e-6,
            boundary_overhead_s=args.boundary_us * 1e-6,
            label="cli-user-supplied",
        )

    engine_kwargs = {
        "device": "cpu",
        "int2": bool(args.int2),
        "int3": bool(args.int3),
        "int4": bool(args.int4),
    }
    with open_engine(args.model, backend="paged", **engine_kwargs) as engine:
        rows = engine.encode(args.prompt)
        contract = OutputContract(args.output_contract)
        if contract is OutputContract.SELECTED_TOKEN_ROWS and not args.output_row:
            print("mrun: selected_token_rows requires at least one --output-row")
            return 1
        if (
            contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
            and len(set(args.candidate_token_id)) < 2
        ):
            print("mrun: candidate_argmax_and_margin requires at least two candidate IDs")
            return 1
        plan = engine.build_work_plan(
            rows,
            output_contract=contract,
            required_output_rows=tuple(args.output_row),
            candidate_token_ids=(
                tuple(tuple(args.candidate_token_id) for _ in rows)
                if contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN
                else ()
            ),
        )
        bundle = compile_work_plan(
            plan,
            "paged-qstore",
            manifest=engine.store.man,
            cost_assumptions=assumptions,
        )
        pushdown_parity = None
        slicing_benchmark = None
        if args.verify_full_head and bool(dict(plan.metadata).get("head_output_pushdown", False)):
            pushdown_parity = verify_output_pushdown_parity(engine, plan, rows)
            slicing_benchmark = benchmark_output_slicing(
                engine,
                plan,
                bundle.lowered,
                rows,
                warmup=args.warmup,
                trials=args.trials,
                graph_compilation=bundle.graph,
            )
        benchmark = benchmark_eager_plan(
            engine,
            plan,
            bundle.lowered,
            rows,
            warmup=args.warmup,
            trials=args.trials,
        )
        graph_resource_trace = None
        if bundle.graph is not None and bundle.graph.rewrite_certificate is not None:
            graph_resource_trace = trace_qstore_graph_execution(
                engine,
                plan,
                bundle.lowered,
                bundle.graph,
                rows,
            )

        artifact_payload = None
        evidence_subject = bundle.fingerprint
        benchmark_fingerprint = evidence_payload_sha256(benchmark.as_dict())
        slicing_fingerprint = (
            None
            if slicing_benchmark is None
            else evidence_payload_sha256(slicing_benchmark.as_dict())
        )
        graph_trace_fingerprint = (
            None
            if graph_resource_trace is None
            else evidence_payload_sha256(graph_resource_trace.as_dict())
        )
        evidence = [
            EvidenceRecord(
                "content-addressed-model-and-store",
                plan.content_identity_verified,
                "WorkPlan manifest identity",
                subject_fingerprint=evidence_subject,
                result_fingerprint=dict(plan.metadata).get("identity_certificate_sha256"),
            ),
            EvidenceRecord(
                "same-qstore-direct-parity",
                benchmark.parity.exact,
                "local eager benchmark",
                f"max_abs_error={benchmark.parity.max_abs_error}",
                subject_fingerprint=evidence_subject,
                result_fingerprint=benchmark_fingerprint,
            ),
            EvidenceRecord(
                "equal-batch-baseline",
                slicing_benchmark is not None and slicing_benchmark.parity.allclose,
                "local full-head versus sliced paired benchmark",
                (
                    "not requested"
                    if slicing_benchmark is None
                    else (f"speedup={slicing_benchmark.speedup},trials={slicing_benchmark.trials}")
                ),
                subject_fingerprint=evidence_subject,
                result_fingerprint=slicing_fingerprint,
            ),
            EvidenceRecord(
                "observed-max-dequantized-block",
                benchmark.max_dequant_block_mb is not None,
                "paged engine telemetry",
                f"max_dequant_block_mb={benchmark.max_dequant_block_mb}",
                subject_fingerprint=evidence_subject,
                result_fingerprint=benchmark_fingerprint,
            ),
            EvidenceRecord(
                "graph-resource-trace",
                graph_resource_trace is not None and graph_resource_trace.complete,
                "local QStore resource interposition",
                (
                    "unavailable"
                    if graph_resource_trace is None
                    else (
                        f"expected={graph_resource_trace.expected_count},"
                        f"observed={graph_resource_trace.observed_count},"
                        f"exact_order={graph_resource_trace.exact_order}"
                    )
                ),
                subject_fingerprint=evidence_subject,
                result_fingerprint=graph_trace_fingerprint,
            ),
        ]
        if pushdown_parity is not None:
            evidence.append(
                EvidenceRecord(
                    "full-head-output-contract-parity",
                    pushdown_parity.allclose,
                    "local full-head comparison",
                    (
                        f"exact={pushdown_parity.exact},"
                        f"max_abs_error={pushdown_parity.max_abs_error}"
                    ),
                    subject_fingerprint=evidence_subject,
                    result_fingerprint=evidence_payload_sha256(pushdown_parity.as_dict()),
                )
            )
        if args.artifact_dir is not None:
            record = CompilationArtifactStore(args.artifact_dir).save(bundle)
            artifact_payload = {
                "path": str(record.path),
                "payload_sha256": record.payload_sha256,
                "byte_count": record.byte_count,
                "artifact_key": record.artifact_key,
                "executable_key": record.executable_key,
                "bundle_fingerprint": record.bundle_fingerprint,
            }
            evidence.append(
                EvidenceRecord(
                    "compilation-artifact-checksum",
                    True,
                    str(record.path),
                    record.payload_sha256,
                    subject_fingerprint=evidence_subject,
                    result_fingerprint=record.payload_sha256,
                )
            )

        promotion_reports = {
            target: evaluate_promotion(bundle, evidence, target=target)
            for target in ("reference", "candidate")
        }
        from .claims import claim_from_promotion_report, export_claim

        artifact_sha256 = (
            artifact_payload.get("payload_sha256")
            if isinstance(artifact_payload, dict)
            else None
        )
        claim_scope = {"model": bundle.plan.model_name}
        for report in promotion_reports.values():
            claim = claim_from_promotion_report(
                report,
                bundle.fingerprint,
                scope=claim_scope,
                artifact_sha256=artifact_sha256,
            )
            export_claim(claim)

        payload = {
            "plan": plan.as_dict(),
            "plan_fingerprint": plan.fingerprint,
            "executable_contract_fingerprint": plan.executable_contract_fingerprint,
            "bundle_fingerprint": bundle.fingerprint,
            "lowered": bundle.lowered.as_dict(),
            "graph": None if bundle.graph is None else bundle.graph.as_dict(),
            "graph_fingerprint": (None if bundle.graph is None else bundle.graph.fingerprint),
            "work_floor": (None if bundle.work_floor is None else bundle.work_floor.as_dict()),
            "memory": None if bundle.memory is None else bundle.memory.as_dict(),
            "cost": None if bundle.cost is None else bundle.cost.as_dict(),
            "benchmark": benchmark.as_dict(),
            "full_head_output_parity": (
                None if pushdown_parity is None else pushdown_parity.as_dict()
            ),
            "output_slicing_benchmark": (
                None if slicing_benchmark is None else slicing_benchmark.as_dict()
            ),
            "graph_resource_trace": (
                None if graph_resource_trace is None else graph_resource_trace.as_dict()
            ),
            "artifact": artifact_payload,
            "reference_promotion": promotion_reports["reference"].as_dict(),
            "candidate_promotion": promotion_reports["candidate"].as_dict(),
        }
    if args.out is not None:
        write_json(args.out, payload)
    else:
        print(stable_json(payload))
    return 0


def _resolve_campaign_runtime(
    args: argparse.Namespace,
    *,
    cuda_available: bool | None = None,
) -> dict[str, object]:
    """Resolve the narrow candidate-scoring primary runtime.

    This deliberately does not affect ``mrun run``: CUDA Graph execution currently
    implements only static, stateless, selected-vocabulary scoring.
    """

    requested_backend = str(args.backend)
    capture_option = getattr(args, "cuda_graph", None)
    packed_variant = any((args.int2, args.int3, args.int4))
    if cuda_available is None:
        import torch

        cuda_available = bool(torch.cuda.is_available())

    if requested_backend == "auto":
        if packed_variant:
            backend = "paged"
            reason = "packed int2/int3/int4 campaign requires the paged reference"
        elif cuda_available:
            backend = "dense-qstore-cuda"
            reason = (
                "CUDA available; dense selected-score runtime chosen, with graph capture "
                "pending an exact promotion gate"
            )
        else:
            backend = "paged"
            reason = "CUDA unavailable; selected-score primary runtime is paged"
        capture_requested = backend == "dense-qstore-cuda" and capture_option is True
    else:
        backend = requested_backend
        capture_requested = backend == "dense-qstore-cuda" and capture_option is True
        reason = f"explicit backend={backend}"

    if capture_option is True and backend != "dense-qstore-cuda":
        raise ValueError("--cuda-graph requires a CUDA dense-QStore campaign")
    return {
        "requested_backend": requested_backend,
        "backend": backend,
        "capture_requested": capture_requested,
        "cuda_available": bool(cuda_available),
        "reason": reason,
        "scope": "static-stateless-selected-vocabulary-campaign",
        "selection_mode": (
            "explicit-cuda-graph"
            if capture_option is True
            else "disabled"
            if capture_option is False
            else "promotion-gated-auto"
            if requested_backend == "auto" and backend == "dense-qstore-cuda"
            else "explicit-eager"
        ),
        "selected_for_primary_runtime": False,
        "promotion": None,
    }


def _campaign(args: argparse.Namespace) -> int:
    from .compiler import (
        CandidateReadout,
        benchmark_candidate_campaign,
        benchmark_candidate_campaign_factorial,
        benchmark_cuda_graph_campaign,
        compile_candidate_campaign,
        select_cuda_graph_promotion,
    )
    from .engine import open_engine
    from .io import stable_json, write_json

    readouts: list[CandidateReadout] = []
    try:
        for raw_query in args.query:
            query_id, separator, raw_tokens = str(raw_query).partition("=")
            if not separator:
                raise ValueError(f"query {raw_query!r} must use QUERY_ID=TOKEN_ID,TOKEN_ID,...")
            token_parts = [value.strip() for value in raw_tokens.split(",")]
            if any(not value for value in token_parts):
                raise ValueError(f"query {query_id!r} contains an empty token ID")
            readouts.append(
                CandidateReadout(
                    query_id=query_id.strip(),
                    candidate_token_ids=tuple(int(value) for value in token_parts),
                )
            )
        if len(readouts) < 2:
            raise ValueError("campaign requires at least two --query values")
    except ValueError as exc:
        print(f"mrun: {exc}")
        return 1

    try:
        runtime_selection = _resolve_campaign_runtime(args)
    except ValueError as exc:
        print(f"mrun: {exc}")
        return 1
    selected_backend = str(runtime_selection["backend"])
    capture_requested = bool(runtime_selection["capture_requested"])

    if selected_backend == "dense-qstore-cuda" and any((args.int2, args.int3, args.int4)):
        print("mrun: dense-qstore-cuda does not support int2/int3/int4 campaign stores")
        return 1
    engine_kwargs = (
        {
            "device": "cpu",
            "int2": bool(args.int2),
            "int3": bool(args.int3),
            "int4": bool(args.int4),
        }
        if selected_backend == "paged"
        else {
            "device": "cuda",
            "compute_dtype": args.compute_dtype,
            "compact_cache_mb": args.compact_cache_mb,
        }
    )
    try:
        with open_engine(args.model, backend=selected_backend, **engine_kwargs) as engine:
            token_ids = engine.encode(
                [args.prompt],
                add_special_tokens=False,
            )[0]
            expected_replays = (
                1 + int(args.warmup) + int(args.trials)
                if args.expected_replays is None
                else int(args.expected_replays)
            )
            if expected_replays <= 0:
                raise ValueError("--expected-replays must be positive")
            if (
                str(runtime_selection["requested_backend"]) == "auto"
                and selected_backend == "dense-qstore-cuda"
                and args.cuda_graph is None
            ):
                union_token_ids = tuple(
                    dict.fromkeys(
                        token for readout in readouts for token in readout.candidate_token_ids
                    )
                )
                decision = select_cuda_graph_promotion(
                    engine,
                    token_ids,
                    union_token_ids,
                    expected_replays=expected_replays,
                    minimum_improvement_percent=args.minimum_improvement_percent,
                )
                from .claims import claim_from_cuda_graph_decision, export_claim

                claim = claim_from_cuda_graph_decision(
                    decision,
                    scope={"backend": selected_backend, "expected_replays": expected_replays},
                )
                export_claim(claim)
                runtime_selection["promotion"] = decision.as_dict()
                capture_requested = decision.selected
                runtime_selection["capture_requested"] = capture_requested
                runtime_selection["selected_for_primary_runtime"] = decision.selected
                runtime_selection["reason"] = f"{decision.reason}; " + (
                    "primary runtime is promoted CUDA Graph replay"
                    if decision.selected
                    else "primary runtime falls back to dense selected-score eager"
                )
            elif selected_backend == "dense-qstore-cuda" and args.cuda_graph is False:
                runtime_selection["reason"] = (
                    "CUDA Graph explicitly disabled; using dense selected-score eager"
                )
            runtime_selection["expected_replays"] = expected_replays
            compiled = compile_candidate_campaign(
                engine,
                token_ids,
                tuple(readouts),
                capture_requested=capture_requested,
            )
            benchmark = benchmark_candidate_campaign(
                engine,
                compiled,
                token_ids,
                warmup=args.warmup,
                trials=args.trials,
                minimum_improvement_percent=args.minimum_improvement_percent,
            )
            capture_benchmark = (
                benchmark_cuda_graph_campaign(
                    engine,
                    compiled,
                    token_ids,
                    warmup=args.warmup,
                    trials=args.trials,
                    minimum_improvement_percent=args.minimum_improvement_percent,
                )
                if capture_requested
                else None
            )
            factorial = (
                benchmark_candidate_campaign_factorial(
                    engine,
                    compiled,
                    token_ids,
                    warmup=args.warmup,
                    trials=args.trials,
                    minimum_improvement_percent=args.minimum_improvement_percent,
                )
                if args.factorial_ablation
                else None
            )
            runtime_gate_passed = benchmark.improvement_demonstrated and (
                capture_benchmark is None or capture_benchmark.capture_improvement_demonstrated
            )
            runtime_selection["runtime_gate_passed"] = runtime_gate_passed
            if capture_requested and not runtime_gate_passed:
                runtime_selection["selected_for_primary_runtime"] = False
                runtime_selection["reason"] = (
                    "CUDA Graph diagnostic ran, but parity/improvement evidence did not "
                    "pass; route is not primary"
                )
            payload = {
                "input": {
                    "prompt": args.prompt,
                    "token_ids": [int(value) for value in token_ids],
                    "binding": compiled.input_binding.as_dict(),
                },
                "runtime_selection": runtime_selection,
                "campaign": compiled.as_dict(),
                "benchmark": benchmark.as_dict(),
            }
            if capture_benchmark is not None:
                payload["cuda_graph_benchmark"] = capture_benchmark.as_dict()
            if factorial is not None:
                payload["factorial_ablation"] = factorial.as_dict()
            if args.campaign_artifact is not None:
                if capture_requested and not runtime_gate_passed:
                    payload["campaign_artifact"] = {
                        "path": str(args.campaign_artifact),
                        "written": False,
                        "reason": "CUDA Graph runtime gate failed",
                    }
                else:
                    write_json(
                        args.campaign_artifact,
                        compiled.as_dict(),
                        sort_keys=True,
                    )
                    payload["campaign_artifact"] = {
                        "path": str(args.campaign_artifact),
                        "written": True,
                        "campaign_fingerprint": compiled.fingerprint,
                    }
    except (RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
        print(f"mrun: {exc}")
        return 1

    if args.out is not None:
        write_json(args.out, payload, sort_keys=True)
    else:
        print(stable_json(payload))
    capture_was_requested = bool(payload["runtime_selection"]["capture_requested"])
    return (
        0
        if not capture_was_requested or bool(payload["runtime_selection"]["runtime_gate_passed"])
        else 2
    )


def _campaign_replay(args: argparse.Namespace) -> int:
    import hashlib
    import time

    from .compiler import (
        CandidateCampaign,
        prepare_candidate_campaign,
        select_cuda_graph_promotion,
    )
    from .engine import open_engine
    from .io import stable_json, write_json

    if args.replays <= 0:
        print("mrun: --replays must be positive")
        return 1
    if args.compact_cache_mb < 0:
        print("mrun: --compact-cache-mb must be non-negative")
        return 1
    try:
        raw_artifact = args.artifact.read_bytes()
        campaign = CandidateCampaign.from_json(raw_artifact)
        lowering_backend = campaign.base_bundle.lowered.backend
        engine_backend = {
            "paged-qstore": "paged",
            "cuda-qstore": "dense-qstore-cuda",
        }.get(lowering_backend)
        if engine_backend is None:
            raise RuntimeError(f"campaign lowerer {lowering_backend!r} has no replay engine")
        plan = campaign.base_bundle.plan
        metadata = dict(plan.metadata)
        compiled_device = str(metadata.get("engine_device", ""))
        activation_dtype = plan.precision.activation_dtype
        weight_dtype = plan.precision.weight_dtype
        if not compiled_device:
            raise RuntimeError("campaign artifact has no compiled engine device")
        if engine_backend == "dense-qstore-cuda":
            if activation_dtype not in {"bf16", "fp16"}:
                raise RuntimeError("dense CUDA campaign activation dtype must be bf16 or fp16")
            if weight_dtype != "int8":
                raise RuntimeError("dense CUDA campaign replay requires an int8 QStore artifact")
            if args.compute_dtype is not None and args.compute_dtype != activation_dtype:
                raise RuntimeError(
                    "--compute-dtype does not match the campaign artifact "
                    f"({args.compute_dtype!r} != {activation_dtype!r})"
                )
            engine_kwargs = {
                "device": compiled_device,
                "compute_dtype": activation_dtype,
                "compact_cache_mb": args.compact_cache_mb,
            }
        else:
            if args.compute_dtype is not None:
                raise RuntimeError("--compute-dtype applies only to dense CUDA campaigns")
            if args.compact_cache_mb:
                raise RuntimeError("--compact-cache-mb applies only to dense CUDA campaigns")
            quantized_flags = {
                "int8": {},
                "int2": {"int2": True},
                "int3": {"int3": True},
                "int4": {"int4": True},
            }
            if weight_dtype not in quantized_flags:
                raise RuntimeError(
                    f"paged campaign QStore dtype {weight_dtype!r} is not replayable"
                )
            engine_kwargs = {
                "device": compiled_device,
                **quantized_flags[weight_dtype],
            }
        with open_engine(
            campaign.base_bundle.plan.model_name,
            backend=engine_backend,
            **engine_kwargs,
        ) as engine:
            token_ids = engine.encode(
                [args.prompt],
                add_special_tokens=False,
            )[0]
            promotion_decision = None
            if plan.capture.requested:
                promotion_decision = select_cuda_graph_promotion(
                    engine,
                    token_ids,
                    campaign.union_token_ids,
                    expected_replays=args.replays,
                )
                from .claims import claim_from_cuda_graph_decision, export_claim

                export_claim(
                    claim_from_cuda_graph_decision(
                        promotion_decision,
                        scope={"route": "serialized-campaign"},
                    )
                )
                if not promotion_decision.selected:
                    raise RuntimeError(
                        "serialized CUDA Graph campaign is not primary-runtime eligible: "
                        f"{promotion_decision.reason}; "
                        f"blockers={list(promotion_decision.blockers)}"
                    )
            results = []
            samples_ms = []
            with prepare_candidate_campaign(
                engine,
                campaign,
                token_ids,
            ) as prepared:
                for _ in range(args.replays):
                    started = time.perf_counter()
                    result = prepared.execute()
                    samples_ms.append((time.perf_counter() - started) * 1000.0)
                    results.append(result.as_dict())
        canonical = results[0]
        runtime_evidence = canonical["evidence"]
        payload = {
            "artifact": {
                "path": str(args.artifact),
                "sha256": hashlib.sha256(raw_artifact).hexdigest(),
            },
            "campaign_fingerprint": campaign.fingerprint,
            "compilation_fingerprint": (
                None
                if campaign.base_bundle.graph is None
                else campaign.base_bundle.graph.fingerprint
            ),
            "engine_backend": engine_backend,
            "compiled_reported_fabric": (campaign.base_bundle.lowered.reported_fabric),
            "reported_fabric": runtime_evidence["reported_fabric"],
            "execution": {
                "runtime_implementation_status": runtime_evidence["runtime_implementation_status"],
                "graph_replay": runtime_evidence["graph_replay"],
                "capture_requested": runtime_evidence["capture_requested"],
                "capture_ready": runtime_evidence["capture_ready"],
                "capture_executed": runtime_evidence["capture_executed"],
                "capture_metadata": runtime_evidence["capture_metadata"],
            },
            "runtime_configuration": {
                key: runtime_evidence[key]
                for key in (
                    "runtime_activation_dtype",
                    "runtime_weight_dtype",
                    "runtime_device",
                    "runtime_fabric",
                )
            },
            "replay_configuration": {
                "compiled_device": compiled_device,
                "activation_dtype": activation_dtype,
                "weight_dtype": weight_dtype,
                "compact_cache_mb": float(args.compact_cache_mb),
                "compute_dtype_assertion": args.compute_dtype,
                "promotion": (None if promotion_decision is None else promotion_decision.as_dict()),
            },
            "content_identity_verified": (campaign.base_bundle.plan.content_identity_verified),
            "input_binding": campaign.input_binding.as_dict(),
            "replay_count": args.replays,
            "replay_samples_ms": samples_ms,
            "deterministic_public_results": all(
                value["readouts"] == canonical["readouts"]
                and value["union_logits"] == canonical["union_logits"]
                for value in results[1:]
            ),
            "result": canonical,
        }
    except (OSError, RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
        print(f"mrun: {exc}")
        return 1

    if args.out is not None:
        write_json(args.out, payload, sort_keys=True)
    else:
        print(stable_json(payload))
    return 0


def _sciencegraph(args: argparse.Namespace) -> int:
    """Replay a checksum-bearing Intervention ScienceGraph on one bound paged engine."""

    import numpy as np
    import torch

    from .compiler import (
        InterventionScienceGraph,
        benchmark_intervention_sciencegraph,
        bind_sciencegraph_model_identity,
        execute_intervention_sciencegraph,
    )
    from .engine import open_engine
    from .io import stable_json, write_json

    try:
        graph = InterventionScienceGraph.read_json(args.artifact)
        if graph.branch_pack.numerical_contract != args.numerical_contract:
            raise ValueError(
                "--numerical-contract does not match the ScienceGraph artifact"
            )
        payloads: dict[str, torch.Tensor] = {}
        for raw in args.payload:
            payload_id, separator, raw_path = str(raw).partition("=")
            if not separator or not payload_id or not raw_path:
                raise ValueError("--payload must use PAYLOAD_ID=PATH.npy")
            if payload_id in payloads:
                raise ValueError(f"duplicate payload binding {payload_id!r}")
            path = Path(raw_path)
            if path.suffix != ".npy":
                raise ValueError("ScienceGraph CLI payloads must be .npy files")
            array = np.load(path, allow_pickle=False)
            if array.dtype.hasobject:
                raise ValueError("object arrays are forbidden as ScienceGraph payloads")
            payloads[payload_id] = torch.from_numpy(np.array(array, copy=True))
        engine_kwargs: dict[str, Any] = {"device": args.device}
        if args.store_dir is not None:
            engine_kwargs["stores_dir"] = args.store_dir
        backend = "paged-fp32" if args.fp32 else "paged"
        with open_engine(args.model, backend=backend, **engine_kwargs) as engine:
            if not engine.capabilities().intervention_sciencegraph:
                raise RuntimeError("engine does not advertise Intervention ScienceGraph support")
            model_identity = bind_sciencegraph_model_identity(engine)
            if model_identity != graph.branch_pack.model_identity:
                raise ValueError("loaded model/store identity does not match the ScienceGraph")
            if args.benchmark:
                result = benchmark_intervention_sciencegraph(
                    engine,
                    graph,
                    model_identity=model_identity,
                    numerical_contract=args.numerical_contract,
                    payload_bindings=payloads,
                    warmups=args.warmup,
                    repeats=args.trials,
                    atol=args.atol,
                    min_speedup=args.minimum_speedup,
                    max_branch_batch=args.max_branch_batch,
                )
            else:
                result = execute_intervention_sciencegraph(
                    engine,
                    graph,
                    model_identity=model_identity,
                    numerical_contract=args.numerical_contract,
                    payload_bindings=payloads,
                    max_branch_batch=args.max_branch_batch,
                )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"mrun: {exc}")
        return 1
    if args.out:
        write_json(args.out, result)
    else:
        print(stable_json(result))
    if args.benchmark and not result["qualified"]:
        return 2
    return 0


def _leaderboard(args: argparse.Namespace) -> int:
    from .engine import leaderboard as lb

    rows = lb.load()
    if args.model:
        rows = [r for r in rows if args.model.lower() in str(r.get("model", "")).lower()]
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    print(lb.render(rows, sort=args.sort, limit=args.limit))
    return 0


def _build_store(args: argparse.Namespace) -> int:
    from .paths import stores_root

    out_root = args.out or stores_root()
    if args.fp32:
        from .engine.kernels.qstore_fp32_build import build as build_fp32

        path = build_fp32(args.model, out_root=out_root)
    elif args.int2:
        from .engine.kernels.qstore_int2 import build as build_int2

        path = build_int2(args.model, out_root=out_root)
    elif args.int3:
        from .engine.kernels.qstore_int3 import build as build_int3

        path = build_int3(args.model, out_root=out_root)
    elif args.int4:
        from .engine.kernels.qstore_int4 import build as build_int4

        path = build_int4(args.model, out_root=out_root)
    else:
        from .engine.kernels.qstore_build import build

        path = build(args.model, out_root=out_root)
    print(f"store -> {path}")
    return 0


def _build_diffusion_store(args: argparse.Namespace) -> int:
    from .diffusion.qstore import build, validate_component_coverage

    if args.validate_component:
        print(
            json.dumps(
                validate_component_coverage(
                    args.component_dir,
                    args.output_dir,
                    model_name=args.model_name,
                    pipeline_class=args.pipeline_class,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        path = build(
            args.component_dir,
            args.output_dir,
            model_name=args.model_name,
            pipeline_class=args.pipeline_class,
        )
        print(f"diffusion store -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
