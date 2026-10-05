"""Finite installed-package hardware assay; no checkout imports or downloads."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fingerprint(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".safetensors", ".json", ".txt", ".model"}:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        result[path.relative_to(root).as_posix()] = digest.hexdigest()
    return result


def main():
    import torch
    from mdb.bench import BenchSession, tensor_digest
    from mdb.bench_loading import load_bench_model, local_checkpoint_root
    from mdb.bench_verify import verify
    from mdb.model_os_contracts import content_digest

    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    from mdb.job_support import request_sha256
    seal = json.loads(Path("payload-seal.json").read_text())
    root = Path(config["output_root"]) / os.environ["MRUN_JOB_ID"]
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = config.get("device", "cuda")
    if config.get("phase_vram_budget_mb"):
        os.environ["VRAM_LIMIT_MB"] = str(config["phase_vram_budget_mb"] * 1e6 / (1 << 20))
    admitted_plan = json.loads(os.environ.get("MRUN_PLAN", "{}"))
    source_binding = {"mode": "worker-enforced", "checks": {
        "request_config": request["config"] == config,
        "config_bytes": hashlib.sha256(Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()).hexdigest()
        == seal["config_bytes_sha256"],
    }}
    if admitted_plan:
        expected = {"model": config["model"], "device": device, "backend": config["backend"],
                    "max_batch": 1, "threads": 4}
        if config.get("dtype"):
            expected["dtype"] = config["dtype"]
        for key, value in expected.items():
            source_binding["checks"][key] = admitted_plan.get(key) == value
        if not all(source_binding["checks"].values()):
            raise RuntimeError("admitted model/backend/dtype/geometry differs from executed configuration")

    def timed(callback):
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        value = callback()
        if device == "cuda":
            torch.cuda.synchronize()
        return value, time.perf_counter() - start

    checkpoint = local_checkpoint_root(config["model"])
    locator = admitted_plan.get("artifact_locator") or {}
    if locator.get("path"):
        source_binding["checks"]["checkpoint_root"] = checkpoint.resolve() == Path(locator["path"]).resolve()
        if not source_binding["checks"]["checkpoint_root"]:
            raise RuntimeError("resolved checkpoint differs from admitted artifact")
    hashes, hash_seconds = timed(lambda: fingerprint(checkpoint))
    weights = content_digest(hashes)
    if config["model"].startswith("qwen") and config.get("dtype"):
        from mdb.ar.model import load_ar_model
        loader = lambda: load_ar_model(config["model"], device=device,
                                      torch_dtype=getattr(torch, config["dtype"]),
                                      attn_implementation="eager")
    else:
        loader = lambda: load_bench_model(config["model"], device=device)
    (spec, model, tokenizer), load_seconds = timed(loader)
    source_binding["checks"]["model_spec"] = getattr(spec, "name", spec.registry_name) == config["model"]
    for field in ("registry_name", "model_id", "revision"):
        if request.get("model_selector"):
            source_binding["checks"]["selector/" + field] = getattr(spec, field) == request["model_selector"][field]
    ar = spec.family in {"qwen2", "mamba"}
    if ar and config.get("dtype"):
        model.to(dtype=getattr(torch, config["dtype"]))
    repeats, tokens = config.get("repeats", 3), config.get("tokens", 8)
    prompt = config["prompt"]
    rows = []
    report = {
        "schema": "mrun-public-stack-speed-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "config": config, "model": spec.to_dict(), "weight_files": hashes,
        "checkpoint_sha256": weights, "hash_seconds": hash_seconds,
        "load_seconds": load_seconds, "python": platform.python_version(),
        "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
        "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("mrun-pub", "mdb-runtime", "saturn-pub", "torch", "transformers", "diffusers")},
        "rows": rows, "tf32": False, "deterministic_algorithms": True,
        "admitted_plan": admitted_plan,
        "source_binding": source_binding, "executed_checkpoint_root": str(checkpoint.resolve()),
        "saturn_request_sha256": request_sha256(request), "payload_seal": seal,
        "resource_basis": request.get("resource_basis"),
        "requested_resource_profile": request.get("resource_profile"),
        "torch_build": torch.__version__, "cuda_build": torch.version.cuda,
        "timing_scope": "synchronized host wall; batch one; warm resident weights; setup/loading/identity hashing excluded",
    }
    if device == "cpu" and Path("/proc/cpuinfo").is_file():
        report["hardware"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                   if line.startswith("model name")), report["hardware"])
    if ar:
        report["dtype"] = str(next(model.parameters()).dtype)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        report["prompt_token_ids"] = ids
        report["prompt_tokens"] = len(ids)
    else:
        report["dtype"] = str(next((model.transformer if spec.family == "flux" else model.unet).parameters()).dtype)
    if config.get("dtype"):
        source_binding["checks"]["loaded_dtype"] = report["dtype"] == "torch." + config["dtype"]
    if not all(source_binding["checks"].values()):
        raise RuntimeError("source/configuration binding failed")

    def save():
        dump(root / "report.json", report)

    def add(label, samples, outputs, *, scope, **extra):
        row = {"configuration": label, "seconds": samples,
               "median_seconds": statistics.median(samples), "min_seconds": min(samples),
               "max_seconds": max(samples), "repeats": len(samples), "timing_scope": scope,
               "outputs_repeat_exact": all(item == outputs[0] for item in outputs),
               "outputs": outputs, **extra}
        if ar:
            row["tokens_per_second"] = tokens / row["median_seconds"]
        else:
            row["images_per_second"] = 1 / row["median_seconds"]
        if device == "cuda":
            row["last_repeat_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (1 << 20)
            row["last_repeat_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / (1 << 20)
            observed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
            row["gpu_process_snapshot"] = observed.stdout.strip().splitlines()
            row["worker_pid"] = os.getpid()
        rows.append(row)
        save()
        print("SPEED_ROW=" + json.dumps({k: row[k] for k in
              ("configuration", "median_seconds", "outputs_repeat_exact")}), flush=True)
        return row

    if ar:
        # Explicitly use Transformers' registered slow Mamba lane in all comparisons.
        old_forwards = []
        if spec.family == "mamba":
            for layer in model.backbone.layers:
                old_forwards.append((layer.mixer, layer.mixer.forward))
                layer.mixer.forward = layer.mixer.slow_forward

        @torch.inference_mode()
        def native_ar():
            result = model(torch.tensor([ids], device=device), use_cache=True)
            output = []
            for _ in range(tokens):
                token = int(result.logits[:, -1].argmax(-1))
                kwargs = ({"cache_params": result.cache_params} if spec.family == "mamba"
                          else {"past_key_values": result.past_key_values})
                result = model(torch.tensor([[token]], device=device), use_cache=True, **kwargs)
                output.append([token, tensor_digest(result.logits[:, -1])])
            return output

        native_ar()  # Unreported warm-up.
        native_outputs, native_times = [], []
        for _ in range(repeats):
            output, seconds = timed(native_ar)
            native_outputs.append(output)
            native_times.append(seconds)
        add("native-eager" if spec.family == "qwen2" else "native-slow",
                         native_times, native_outputs, scope="prompt prefill plus fixed-token cached decode")
        for mixer, forward in old_forwards:
            mixer.forward = forward
        reference = native_outputs[0]
    else:
        from mrun.diffusion.phase import PhasePipeline
        phase = PhasePipeline.wrap(model, device=device)

        def native_image():
            embeds = phase.encode(prompt)
            result = phase.generate(embeds, generator=torch.Generator(device=device).manual_seed(17),
                                    height=config["height"], width=config["width"],
                                    num_inference_steps=config["steps"], guidance_scale=1.0,
                                    output_type="np")
            return tensor_digest(torch.from_numpy(result.images.copy()))

        native_image()
        native_outputs, native_times = [], []
        for _ in range(repeats):
            output, seconds = timed(native_image)
            native_outputs.append(output)
            native_times.append(seconds)
        add("native-phase", native_times, native_outputs,
                         scope="warm conditioning-cache lookup, seeded latent sampling, denoise and VAE; RGB digest copied")
        reference = native_outputs[0]
        # Phase ownership detaches encoders while denoising. Restore the caller's
        # pipeline before another adapter acquires it; keep this outside timing.
        phase.close()

    variants = ([('native', 'legacy'), ('native', 'journal'), ('virtual', 'journal')]
                if spec.family == "qwen2" else [('native', 'legacy'), ('native', 'journal')])
    if spec.family == "qwen2" and next(model.parameters()).dtype != torch.bfloat16:
        report["not_applicable"] = [{"component": "MDB Qwen", "reason": "direct-read KV ABI requires BF16"}]
        variants = []
    if config.get("mdb_publications"):
        variants = [(lane, mode) for lane, mode in variants if mode in config["mdb_publications"]]
    for lane, publication in variants:
        session = None
        label = f"mdb-{lane}-{publication}"
        try:
            print("WORKER_PHASE=" + json.dumps({"configuration": label, "phase": "setup"}), flush=True)
            kwargs = dict(root=root / label, model_id=spec.model_id, revision=spec.revision,
                          model_sha256=weights, session_id=f"{os.environ['MRUN_JOB_ID']}-{label}",
                          lane=lane, publication_mode=publication,
                          continuation_capacity=max(32, tokens * 2))
            if not ar:
                kwargs["resident_state_bytes"] = 512 << 20
            session, setup = timed(lambda: BenchSession.from_resident(model, tokenizer, **kwargs))

            def prepare():
                if ar:
                    return session.prefill(ids)
                return session.prepare(prompt, seed=17, height=config["height"], width=config["width"],
                                       num_inference_steps=config["steps"], guidance_scale=1.0)

            parent, prepare_seconds = timed(prepare)
            print("WORKER_PHASE=" + json.dumps({"configuration": label, "phase": "warmup",
                                                "setup_seconds": setup, "prepare_seconds": prepare_seconds}), flush=True)

            def finish():
                result = session.generate(tokens if ar else None)
                if ar:
                    return [[r["token_id"], r["logits_sha256"]] for r in result]
                return tensor_digest(session.resolve_value("image"))

            finish()  # Warm-up is retained as evidence but excluded from timings.
            outputs, samples, restore_samples = [], [], []
            for index in range(repeats):
                _, restored = timed(lambda: session.restore(parent))
                output, seconds = timed(finish)
                restore_samples.append(restored)
                outputs.append(output)
                samples.append(seconds)
                print("SPEED_SAMPLE=" + json.dumps({"configuration": label, "repeat": index,
                                                    "seconds": seconds}), flush=True)
            row = add(label, samples, outputs, scope="prepared fixed-budget suffix with durable cuts; preparation/restore excluded",
                      setup_seconds=setup, prepare_seconds=prepare_seconds,
                      restore_seconds=restore_samples,
                      native_output_exact=outputs[0] == reference,
                      native_token_exact=([r[0] for r in outputs[0]] == [r[0] for r in reference]) if ar else None,
                      prepare_plus_suffix_seconds=prepare_seconds + statistics.median(samples),
                      performance=session.inspect("performance"), memory=session.inspect("memory"))
            # A typed partial cut and a no-op/abort transaction test the microscope path.
            print("WORKER_PHASE=" + json.dumps({"configuration": label, "phase": "partial-cut"}), flush=True)
            session.restore(parent)
            if ar:
                session.step("embedding")
                session.step("layer")
            else:
                session.step("block")
            partial = session.checkpoint("partial-cut")
            print("WORKER_PHASE=" + json.dumps({"configuration": label, "phase": "partial-continuation"}), flush=True)
            first = finish()
            session.restore(partial)
            print("WORKER_PHASE=" + json.dumps({"configuration": label, "phase": "partial-replay"}), flush=True)
            second = finish()
            row["partial_replay_exact"] = first == second
            session.restore(partial)
            before = tensor_digest(session.resolve_value("residual"))
            edit = session.preview_edit("residual", 0.9, mode="scale")
            session.abort_edit(edit["act_id"])
            row["abort_exact"] = before == tensor_digest(session.resolve_value("residual"))
            if publication == "journal":
                edit = session.preview_edit("residual", 1.0, mode="scale")
                session.commit_edit(edit["act_id"])
                row["noop_exact"] = finish() == first
                session.restore(partial)
                edit = session.preview_edit("residual", 0.9, mode="scale")
                receipt = session.commit_edit(edit["act_id"])
                row["intervention"] = {"mode": "scale", "value": 0.9, "site": "residual",
                                       "receipt": receipt, "baseline": first, "candidate": finish(),
                                       "scope": "same partial parent, whole native continuation; divergent later contexts are total effects"}
                if not ar:
                    from PIL import Image
                    rgb = session.resolve_value("image")[0].float().clamp(0, 1).cpu().numpy()
                    Image.fromarray((rgb * 255).round().astype("uint8")).save(root / (label + "-candidate.png"))
            row["partial_cut"] = partial
            print("WORKER_PHASE=" + json.dumps({"configuration": label, "phase": "byte-custody"}), flush=True)
            row["verification"] = verify(session.root)
            save()
            print("WORKER_PHASE=" + json.dumps({"configuration": label, "phase": "integrity-complete"}), flush=True)
            if (not row["outputs_repeat_exact"] or not row["partial_replay_exact"]
                    or not row["abort_exact"] or not row.get("noop_exact", True)):
                raise RuntimeError("same-lane replay/abort integrity failed")
        except Exception as exc:
            report.setdefault("errors", []).append({"configuration": label, "error": str(exc),
                                                    "traceback": traceback.format_exc()})
            save()
            traceback.print_exc()
        finally:
            if session is not None:
                session.close()

    if ar:
        from saturn_pub.adapters.qwen import QwenAdapter
        from saturn_pub.adapters.mamba import MambaAdapter
        cls = QwenAdapter if spec.family == "qwen2" else MambaAdapter
        for residency in config.get("saturn_residencies", ["resident"]):
            label = "saturn-" + residency
            try:
                adapter, setup = timed(lambda: cls(model, residency=residency, device=device))

                def saturn_run():
                    session = adapter.session(ids)
                    adapter.generate(session, tokens)
                    return session.read("tokens")[0].tolist()[-tokens:]

                saturn_run()
                outputs, samples = [], []
                for _ in range(repeats):
                    output, seconds = timed(saturn_run)
                    outputs.append(output)
                    samples.append(seconds)
                row = add(label, samples, outputs, scope="session creation, prompt execution and fixed-token greedy decode",
                          setup_seconds=setup, native_token_exact=outputs[0] == [r[0] for r in reference],
                          execution_contract=adapter.execution)
                session = adapter.session(ids)
                session.continue_(2)
                parent = session.capture()
                left, right = session.fork(parent), session.fork(parent)
                adapter.generate(left, 2)
                adapter.generate(right, 2)
                comparison = left.compare(right)
                row["partial_replay_exact"] = torch.equal(left.read("logits"), right.read("logits"))
                row["comparison"] = comparison.to_dict() if hasattr(comparison, "to_dict") else {
                    "left_frame": left.capture().fingerprint, "right_frame": right.capture().fingerprint,
                    "same_logits": row["partial_replay_exact"]}
                save()
                if not row["outputs_repeat_exact"] or not row["partial_replay_exact"]:
                    raise RuntimeError("Saturn same-lane repeat/replay integrity failed")
            except Exception as exc:
                report.setdefault("errors", []).append({"configuration": label, "error": str(exc),
                                                        "traceback": traceback.format_exc()})
                save()
                traceback.print_exc()
    if not rows[0]["outputs_repeat_exact"]:
        report.setdefault("errors", []).append({"configuration": rows[0]["configuration"],
                                               "error": "native repeated output integrity failed"})
    report["mechanics_status"] = "valid" if not report.get("errors") else "incomplete"
    save()
    print("PUBLIC_STACK_REPORT=" + json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    if report.get("errors"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
