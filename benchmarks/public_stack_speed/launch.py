"""Seal three installed public packages and admit one finite hardware worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    from mdb.execution_environment import intent
    from mdb.bench_loading import resolve_bench_model
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import build_runtime_wheel, seal_package, stage_worker, worker_command
    from mrun.client.api import Api

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--url", default=os.environ.get("MRUN_URL"))
    parser.add_argument("--host")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    # The worker is a caller-owned HF/phase runtime, never an automatic QStore engine.
    config.setdefault("backend", "hf")
    config.setdefault("seq_len", 64)
    os.environ["GATHER_MAX_BATCH"] = "1"
    resource_kwargs = dict(config.get("resource_profile") or {})
    resource_basis = config.get("resource_basis")
    model_selector = resolve_bench_model(config["model"]).to_dict()
    if model_selector["family"] == "flux":
        from mdb.bench_resources import diffusion_resources
        profile, resource_basis = diffusion_resources(
            config["model"], height=config["height"], width=config["width"],
            steps=config["steps"], state_budget_bytes=512 << 20)
        resource_kwargs = {field: getattr(profile, field) for field in
                           ("ram_mb", "vram_mb", "disk_gb", "cpu_threads")}
        if resource_basis.get("phase_vram_budget_decimal_mb"):
            config["phase_vram_budget_mb"] = resource_basis["phase_vram_budget_decimal_mb"]
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    cuda = config.get("device", "cuda") == "cuda"
    request = {"schema": "mrun-public-stack-speed-request-v1", "config": config,
               "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
               "execution_environment": intent(), "needs": {"cuda": True} if cuda else {},
               "model_selector": model_selector, "resource_profile": resource_kwargs,
               "resource_basis": resource_basis,
               "scheduling_contract": one_lease_scheduling_contract(
                   logical_branches=8, logical_operations=256, logical_checkpoint_replays=12)}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if args.stage.exists():
        raise ValueError("staging directory must be new")
    worker = Path(__file__).with_name("worker.py")
    stage_worker(args.stage, worker=worker, config=config, request=request, config_bytes=raw)
    wheel = build_runtime_wheel(args.stage, package="saturn_pub", distribution="saturn-pub", version="0.4.0")
    seal_path = args.stage / "payload-seal.json"
    seal = json.loads(seal_path.read_text())
    seal["packages"]["saturn_pub"] = seal_package("saturn_pub")
    seal["wheels"][wheel.name] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n")
    command = worker_command(worker.name)
    index = command.index("python", command.index("uv"))
    command[index:index] = ["--with", "./" + wheel.name]
    # These deterministic settings are part of the admitted configuration.
    command.insert(1, "CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if args.dry_run:
        print(json.dumps({"command": command, "request": request, "stage": str(args.stage)}, indent=2))
        return
    api = Api(base_url=args.url)
    job = guarded_mrun_launch(command, request=request, experiment="public-stack-speed-" + config["model"],
                             task="generate", model=config["model"], payload=args.stage,
                             mrun_config={**config, "scheduling_contract": request["scheduling_contract"]},
                             prefer_host=args.host, pin=args.host if not cuda else None,
                             gpu=cuda, timeout_s=config.get("timeout_s", 840),
                             retry_on_kill=False, preflight=True, api=api,
                             **resource_kwargs,
                             note="Public wheels only; one resident model, batch 1, three warm repeats; bounded tokens/image geometry; exact replay/abort checks; no retries.")
    print(json.dumps({"job_id": job, "request_sha256": request["request_sha256"], "stage": str(args.stage)}))


if __name__ == "__main__":
    main()
