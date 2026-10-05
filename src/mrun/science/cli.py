"""CLI for local and fleet-backed scientific mrun benchmarks."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..client.submit import attach, submit
from ..io import stable_json, write_json
from .benchmark import run_benchmark
from .config import config_sha256, load_config
from .runtime import canonical_device, canonical_dtype, canonical_fabric


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrun science")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("benchmark", "run a resolved benchmark config in this process"),
        ("submit", "ship a resolved config to the mrun fleet"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
        command.add_argument("--dry-run", action="store_true")
    submit_parser = sub.choices["submit"]
    submit_parser.add_argument("--detach", action="store_true")
    submit_parser.add_argument("--host", default=None, help="hard host pin; normally omit")
    submit_parser.add_argument("--prefer-host", default=None, help="soft host preference")
    return parser


def _scheduler_request(config: dict[str, Any]) -> dict[str, Any]:
    runtime = config["runtimes"][0]
    serve = config["serve"]
    first_options = dict(runtime.get("options") or {})
    option_device = canonical_device(first_options.get("device", "auto"))
    device = option_device if option_device != "auto" else canonical_device(serve["device"])
    option_dtype = canonical_dtype(
        first_options.get("compute_dtype", first_options.get("dtype", "auto"))
    )
    requested_dtype = (
        option_dtype if option_dtype != "auto" else canonical_dtype(serve["dtype"])
    )
    backend = str(runtime["backend"])
    planner_options = dict(first_options)
    # Target/dtype controls are sent through the scheduler's typed fields.  Leaving them in
    # backend_options can look like a forbidden collision to specialized policy (notably
    # qwen3-moe-cuda's device/compute_dtype contract).
    for key in ("fabric", "device", "dtype", "compute_dtype"):
        planner_options.pop(key, None)
    needs = dict(config["execution"].get("needs") or {})
    cuda_backends = {
        "dense-qstore-cuda",
        "dense-cuda",
        "cuda-source-int8",
        "cuda-source-int8-compact-head",
        "olmoe-cuda",
        "qwen3-moe-cuda",
        "moe-qstore-cuda",
    }
    metal_backends = {
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
    fabrics: set[str] = set()
    plan_variants: list[dict[str, Any]] = []
    for item in config["runtimes"]:
        options = dict(item.get("options") or {})
        item_backend = str(item["backend"]).lower().replace("_", "-")
        option_device = canonical_device(options.get("device", "auto"))
        item_device = option_device
        if option_device == "auto":
            item_device = canonical_device(serve["device"])
        item_fabric_value = (
            options["fabric"]
            if "fabric" in options
            else serve.get("fabric", "auto") if option_device == "auto" else "auto"
        )
        item_fabric = canonical_fabric(item_fabric_value)
        fabrics.add(item_fabric)
        option_dtype = canonical_dtype(
            options.get("compute_dtype", options.get("dtype", "auto"))
        )
        item_dtype = option_dtype if option_dtype != "auto" else canonical_dtype(serve["dtype"])
        runtime_planner_options = dict(options)
        for key in ("fabric", "device", "dtype", "compute_dtype"):
            runtime_planner_options.pop(key, None)
        plan_variants.append(
            {
                "backend": item_backend,
                "dtype": None if item_dtype == "auto" else item_dtype,
                "device": None if item_device == "auto" else item_device,
                "backend_options": runtime_planner_options,
            }
        )
        if item_device.startswith("cuda") or item_backend in cuda_backends:
            needs["cuda"] = True
        if item_device == "mps" or item_backend in metal_backends:
            needs["mps"] = True
        if item_backend == "multifabric":
            child_backends = options.get("backends", ("paged", "mlx"))
            if isinstance(child_backends, (list, tuple)) and any(
                str(child).lower().replace("_", "-")
                in {"mlx", "mlx-q4", "mlx-component", "mlx-component-q4", "ane", "coreml"}
                for child in child_backends
            ):
                needs["mps"] = True
    if "cuda" in fabrics:
        needs["cuda"] = True
    if "metal" in fabrics:
        needs["mps"] = True
    execution = config["execution"]
    # ``auto`` must remain None for the scheduler planner; the worker resolves it after the
    # lease is placed.  Literal "auto" is not a valid RunPlan device/dtype.
    planner_dtype = None if requested_dtype == "auto" else requested_dtype
    planner_device = None if device == "auto" else device
    return {
        "experiment": config["experiment"]["name"],
        "model": config["model"]["name"],
        "task": "forward",
        "backend": backend,
        "dtype": planner_dtype,
        "device": planner_device,
        "requested_dtype": requested_dtype,
        "requested_device": device,
        "fabric": canonical_fabric(serve.get("fabric", "auto")),
        "backend_options": planner_options,
        "plan_variants": plan_variants,
        "runtime_backends": [item["backend"] for item in config["runtimes"]],
        "needs": needs,
        "host": execution.get("host"),
        "prefer_host": execution.get("prefer_host"),
        "science_config_sha256": config_sha256(config),
        "batch_sizes": config["test"]["batch_sizes"],
        "context_tokens": config["test"]["context_tokens"],
        "serve_type": config["serve"]["type"],
    }


def _submit(config: dict[str, Any], args: argparse.Namespace) -> int:
    request = _scheduler_request(config)
    if args.dry_run:
        print(stable_json(request))
        return 0
    with tempfile.TemporaryDirectory(prefix="mrun-science-") as directory:
        payload = Path(directory)
        write_json(payload / "science-config.json", config, sort_keys=True)
        # The worker's editable mrun checkout may lag the launcher's dirty tree. Ship only the
        # science and MLOps packages and overlay them onto the worker's installed mrun package at
        # process start; engine/scheduler modules continue to come from the worker environment.
        science_source = Path(__file__).resolve().parent
        shutil.copytree(science_source, payload / "mrun" / "science")
        shutil.copytree(science_source.parent / "mlops", payload / "mrun" / "mlops")
        execution = config["execution"]
        job_id = submit(
            experiment=f"science-{config['experiment']['name']}",
            cmd=[
                "uv",
                "run",
                "--project",
                "{env}",
                "python",
                "-c",
                (
                    "import mrun,os,sys; "
                    "mrun.__path__.insert(0, os.path.join(os.getcwd(), 'mrun')); "
                    "from mrun.science.cli import main; sys.exit(main(sys.argv[1:]))"
                ),
                "benchmark",
                "--config",
                "science-config.json",
            ],
            config={**request, "task_family": "forward"},
            needs=request["needs"],
            host=args.host or execution.get("host"),
            prefer_host=args.prefer_host or execution.get("prefer_host"),
            reservation=execution.get("reservation"),
            payload=payload,
            env_alias=execution.get("env_alias", "mrun"),
            model=config["model"]["name"],
            task="forward",
            backend=request["backend"],
            dtype=request["dtype"],
            device=request["device"],
            backend_options=request["backend_options"],
            plan_variants=request["plan_variants"],
            timeout_s=execution.get("timeout_s"),
            priority=int(execution.get("priority", 0)),
            note=(
                "scientific benchmark; "
                f"runtimes={','.join(item['name'] for item in config['runtimes'])}; "
                f"serve={config['serve']['type']}; batch={config['test']['batch_sizes']}; "
                f"context={config['test']['context_tokens']}"
            ),
            preflight=True,
        )
        print(json.dumps({"job_id": job_id, "config_sha256": request["science_config_sha256"]}))
        if args.detach:
            return 0
        result = attach(job_id)
        return 0 if result.state == "succeeded" else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config, args.override)
    if args.command == "benchmark":
        if args.dry_run:
            print(
                stable_json(
                    {
                        "config_sha256": config_sha256(config),
                        "experiment": config["experiment"],
                        "runtimes": config["runtimes"],
                        "serve": config["serve"],
                        "test": config["test"],
                    }
                )
            )
            return 0
        result = run_benchmark(config)
        print(stable_json({"run_id": result["run_id"], "run_dir": result["run_dir"]}))
        return 0
    return _submit(config, args)
