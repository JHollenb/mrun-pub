"""Config-driven, trackable model-runtime benchmarks."""

from __future__ import annotations

import hashlib
import json
import platform
import resource as process_resource
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from ..engine import open_engine
from ..engine.report import engine_facts
from ..io import sanitize, write_json
from ..paths import repo_root, safe_stem
from .config import config_sha256
from .runtime import resolve_runtime_execution


class ScienceBenchmarkError(RuntimeError):
    """Raised when a declared scientific benchmark cannot be executed."""


def _process_rss_peak_mb() -> float:
    value = float(process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024**2) if sys.platform == "darwin" else value / 1024


class _ResourceSampler:
    """Sample process and CUDA memory so mean as well as peak RSS is recorded."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _sample() -> dict[str, float]:
        result: dict[str, float] = {}
        try:
            import psutil

            result["rss_mb"] = float(psutil.Process().memory_info().rss / 1024**2)
        except Exception:  # pragma: no cover - psutil is a declared dependency
            pass
        try:
            import torch

            if torch.cuda.is_available():
                result["vram_allocated_mb"] = float(torch.cuda.memory_allocated() / 1024**2)
                result["vram_reserved_mb"] = float(torch.cuda.memory_reserved() / 1024**2)
        except Exception:  # noqa: BLE001 - telemetry never invalidates a measurement
            pass
        return result

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._sample()
            if sample:
                self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> _ResourceSampler:
        first = self._sample()
        if first:
            self.samples.append(first)
        self._thread = threading.Thread(
            target=self._run, name="mrun-science-resources", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        final = self._sample()
        if final:
            self.samples.append(final)

    def summary(self) -> dict[str, float | None]:
        keys = {key for sample in self.samples for key in sample}
        result: dict[str, float | None] = {}
        for key in sorted(keys):
            values = [sample[key] for sample in self.samples if key in sample]
            result[f"{key}_mean"] = float(mean(values)) if values else None
            result[f"{key}_max"] = float(max(values)) if values else None
            result[f"{key}_min"] = float(min(values)) if values else None
        return result


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _fixed_rows(
    engine: Any, prompts: list[str], batch_size: int, context_tokens: int
) -> list[np.ndarray]:
    encoded = engine.encode(prompts, add_special_tokens=False)
    rows: list[np.ndarray] = []
    for index in range(batch_size):
        row = np.asarray(encoded[index % len(encoded)], dtype=np.int64)
        if row.size == 0:
            raise ScienceBenchmarkError("tokenizer produced an empty prompt")
        if row.size >= context_tokens:
            row = row[-context_tokens:]
        else:
            # Repeating a deterministic prompt is preferable to padding with an undefined
            # token: every runtime sees exactly the requested physical-forward geometry.
            row = np.resize(row, context_tokens)
        rows.append(row)
    return rows


def _reset_cache(engine: Any) -> None:
    reset = getattr(engine, "reset_page_cache", None)
    if callable(reset):
        reset(clear_pages=True)


def _generate_rows(engine: Any, rows: list[np.ndarray], decode_tokens: int) -> list[list[int]]:
    generate_batch = getattr(engine, "generate_batch", None)
    if callable(generate_batch):
        return generate_batch(rows, max_new_tokens=decode_tokens)
    generate = getattr(engine, "generate", None)
    if callable(generate):
        return [generate(row, max_new_tokens=decode_tokens) for row in rows]
    # HFEngine deliberately exposes logits as its semantic primitive. For a benchmark that
    # explicitly asks for inference, use the underlying HF model's greedy generate surface.
    model = getattr(engine, "model", None)
    if model is None:
        raise ScienceBenchmarkError(
            f"backend {getattr(engine, 'backend', '?')!r} has no generation adapter"
        )
    try:
        import torch

        input_ids = torch.as_tensor(np.stack(rows), device=getattr(engine, "device", "cpu"))
        outputs = model.generate(
            input_ids,
            max_new_tokens=decode_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(getattr(engine, "tokenizer", None), "pad_token_id", None),
        )
        width = input_ids.shape[1]
        return [row[width:].detach().cpu().tolist() for row in outputs]
    except Exception as error:  # noqa: BLE001
        raise ScienceBenchmarkError(
            f"HF-style generation failed for {getattr(engine, 'backend', '?')}: {error}"
        ) from error


def _capabilities(engine: Any) -> dict[str, Any]:
    try:
        value = engine.capabilities()
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}
    if hasattr(value, "__dataclass_fields__"):
        return {key: bool(getattr(value, key)) for key in value.__dataclass_fields__}
    return sanitize(value)


_CONTEXT_LIMIT_KEYS = (
    "max_position_embeddings",
    "max_sequence_length",
    "max_seq_len",
    "context_length",
    "n_positions",
    "seq_length",
)


def _node_value(node: Any, key: str) -> Any:
    if isinstance(node, Mapping):
        return node.get(key)
    return getattr(node, key, None)


def _positive_context_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit() and int(value) > 0:
        return int(value)
    return None


def _context_info_from_node(
    node: Any, source: str, *, seen: set[int] | None = None
) -> dict[str, Any] | None:
    """Read a declared model context limit from HF-style config objects or dictionaries."""

    if node is None:
        return None
    seen = set() if seen is None else seen
    node_id = id(node)
    if node_id in seen:
        return None
    seen.add(node_id)
    for key in _CONTEXT_LIMIT_KEYS:
        value = _positive_context_value(_node_value(node, key))
        if value is not None:
            result: dict[str, Any] = {
                "max_context_tokens": value,
                "source": f"{source}.{key}",
            }
            sliding = _positive_context_value(_node_value(node, "sliding_window"))
            if sliding is not None:
                result["sliding_window_tokens"] = sliding
            return result
    # Qwen3.5 and similar composite checkpoints put the language-model limit under a nested
    # text/language config.  Keep the source path so a result is auditable rather than just a
    # naked integer.
    for key in ("text_config", "language_config", "llm_config", "model_config"):
        nested = _node_value(node, key)
        nested_result = _context_info_from_node(nested, f"{source}.{key}", seen=seen)
        if nested_result is not None:
            return nested_result
    return None


def _model_context_info(engine: Any) -> dict[str, Any]:
    """Return the model-declared context limit used to validate every benchmark cell."""

    candidates: list[tuple[Any, str]] = []
    model = getattr(engine, "model", None)
    if model is not None:
        candidates.append((getattr(model, "config", None), "engine.model.config"))
    candidates.extend(
        [
            (getattr(engine, "config", None), "engine.config"),
            (getattr(engine, "cfg", None), "engine.cfg"),
        ]
    )
    for node, source in candidates:
        result = _context_info_from_node(node, source)
        if result is not None:
            return result

    model_dirs: list[Path] = []
    for candidate in (
        getattr(engine, "model_dir", None),
        getattr(getattr(engine, "spec", None), "local_path", None),
    ):
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_dir() and path not in model_dirs:
                model_dirs.append(path)
    for model_dir in model_dirs:
        config_path = model_dir / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result = _context_info_from_node(config, str(config_path))
        if result is not None:
            return result
    return {"max_context_tokens": None, "source": "unavailable"}


def _torch_dtype(dtype: str) -> Any:
    import torch

    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype]


def _dtype_kwargs(
    runtime: dict[str, Any], serve: dict[str, Any], execution: dict[str, Any]
) -> dict[str, Any]:
    options = dict(runtime.get("options") or {})
    backend = str(runtime["backend"]).lower().replace("_", "-")
    dtype = str(execution["dtype"])
    # These engines expose a real torch device.  MLX/Core ML use their own placement and the
    # ANE compatibility adapter is backed by CPU paged tensors, so passing mps to it is wrong.
    if backend == "multifabric":
        # MultiFabricEngine owns a CPU-paged child plus an optional MLX child.  Its ``device``
        # argument is passed to the paged child; MPS is therefore a fabric selection, not a
        # valid child device.  CUDA can be passed through for the paged child when explicitly
        # requested on a CUDA host.
        options["device"] = "cuda:0" if execution["fabric"] == "cuda" else "cpu"
    elif backend not in {
        "mlx",
        "mlx-q4",
        "mlx-component",
        "mlx-component-q4",
        "metal-component",
        "metal-component-q4",
    }:
        if options.get("device") in {None, "auto"}:
            options["device"] = (
                "cpu" if backend in {"ane", "coreml"} else execution["device"]
            )
    if backend in {
        "paged",
        "paged-fp32",
        "paged-lossless",
        "paged-fp16",
        "paged-bf16",
        "qwen3-moe-cuda",
        "moe-qstore-cuda",
        "dense-qstore-cuda",
        "dense-cuda",
        "cuda-source-int8",
        "cuda-source-int8-compact-head",
    }:
        if options.get("compute_dtype") in {None, "auto"}:
            options["compute_dtype"] = dtype
    elif backend == "olmoe-cuda":
        options["dtype"] = _torch_dtype(dtype)
    elif backend == "moe-stream":
        if options.get("dtype") in {None, "auto"}:
            options["dtype"] = dtype
    elif backend == "hf":
        model_kwargs = dict(options.get("model_kwargs") or {})
        if model_kwargs.get("torch_dtype") in {None, "auto"}:
            model_kwargs["torch_dtype"] = dtype
        options["model_kwargs"] = model_kwargs
    # These are science-level controls, not constructor kwargs.  ``compute_dtype`` remains for
    # backends whose public contract uses it; ordinary ``dtype`` is handled above.
    options.pop("fabric", None)
    if backend not in {"moe-stream", "olmoe-cuda"}:
        options.pop("dtype", None)
    return {key: value for key, value in options.items() if value is not None}


def _run_case(
    engine: Any,
    *,
    serve_type: str,
    rows: list[np.ndarray],
    decode_tokens: int,
    warmup: int,
    repeats: int,
    reset_cache_each_repeat: bool,
) -> dict[str, Any]:
    def forward() -> float:
        started = time.perf_counter()
        engine.logits_batch(rows)
        return time.perf_counter() - started

    if serve_type == "training":
        raise ScienceBenchmarkError(
            "serve.type=training is part of the config contract but has no mrun engine adapter yet"
        )
    if serve_type not in {"forward", "inference"}:
        raise ScienceBenchmarkError(f"unsupported serve.type={serve_type!r}")

    for _ in range(warmup):
        if reset_cache_each_repeat:
            _reset_cache(engine)
        if serve_type == "forward":
            forward()
        else:
            _generate_rows(engine, rows, decode_tokens)

    prefill_s: list[float] = []
    total_s: list[float] = []
    generated_tokens: list[int] = []
    resource_summaries: list[dict[str, Any]] = []
    for _ in range(repeats):
        if reset_cache_each_repeat:
            _reset_cache(engine)
        with _ResourceSampler() as resources:
            prefill_elapsed = forward()
            prefill_s.append(prefill_elapsed)
            if serve_type == "forward":
                total_s.append(prefill_elapsed)
                generated_tokens.append(0)
            else:
                started = time.perf_counter()
                generated = _generate_rows(engine, rows, decode_tokens)
                total_s.append(time.perf_counter() - started)
                generated_tokens.append(
                    sum(
                        len(item) if hasattr(item, "__len__") else decode_tokens
                        for item in generated
                    )
                )
        resource_summaries.append(resources.summary())

    batch = len(rows)
    prompt_tokens = sum(len(row) for row in rows)
    prefill_metrics = _percentiles(prefill_s)
    total_metrics = _percentiles(total_s)
    decode_pairs = [
        (tokens, total - prefill)
        for tokens, total, prefill in zip(generated_tokens, total_s, prefill_s, strict=True)
        if total - prefill > 0
    ]
    decode_s = [value for _tokens, value in decode_pairs]
    prefill_tok_s = [prompt_tokens / value for value in prefill_s if value > 0]
    output_tok_s = [
        tokens / value for tokens, value in zip(generated_tokens, total_s, strict=True) if value > 0
    ]
    decode_tok_s = [tokens / value for tokens, value in decode_pairs if value > 0]
    e2e_tok_s = [
        (prompt_tokens + tokens) / value
        for tokens, value in zip(generated_tokens, total_s, strict=True)
        if value > 0
    ]
    resource_keys = {key for summary in resource_summaries for key in summary}
    resource = {
        key: float(
            mean([summary[key] for summary in resource_summaries if summary.get(key) is not None])
        )
        for key in resource_keys
        if any(summary.get(key) is not None for summary in resource_summaries)
    }
    rss_means = [
        summary["rss_mb_mean"]
        for summary in resource_summaries
        if summary.get("rss_mb_mean") is not None
    ]
    rss_maxima = [
        summary["rss_mb_max"]
        for summary in resource_summaries
        if summary.get("rss_mb_max") is not None
    ]
    if rss_means:
        resource["rss_mb_mean"] = float(mean(rss_means))
    if rss_maxima:
        resource["rss_mb_max"] = float(max(rss_maxima))
    resource["process_rss_peak_mb"] = _process_rss_peak_mb()
    fallback_stats = getattr(engine, "scalar_fallback_stats", lambda: {})()
    if serve_type == "forward":
        execution_mode = "logits_batch"
    elif fallback_stats:
        execution_mode = "scalar-fallback"
    elif callable(getattr(engine, "generate_batch", None)):
        execution_mode = "generate_batch"
    else:
        execution_mode = "scalar-generation"
    result = {
        "batch_size": batch,
        "context_tokens": len(rows[0]),
        "prompt_tokens": prompt_tokens,
        "decode_tokens_requested": decode_tokens,
        "generated_tokens_total": int(sum(generated_tokens)),
        "repeats": repeats,
        "warmup": warmup,
        "prefill_wall_s": prefill_metrics,
        "prefill_tok_s": _percentiles(prefill_tok_s),
        "generation_wall_s": total_metrics,
        "generation_tok_s": _percentiles(output_tok_s),
        "decode_wall_s_estimate": _percentiles(decode_s),
        "decode_tok_s": _percentiles(decode_tok_s),
        "end_to_end_tok_s": _percentiles(e2e_tok_s),
        "resource": resource,
        "timing_note": (
            "decode wall/tok is inferred as total generation time minus a separate prefill "
            "forward; it is comparable within this harness, but is not a native stage timer "
            "for every backend"
        ),
        "batch_execution_mode": execution_mode,
    }
    return result


def _source_snapshot() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        status_text = status.stdout or ""
        return {
            "repo": str(repo_root()),
            "commit": sha,
            "branch": branch,
            "dirty": bool(status_text.strip()),
            "working_tree_status_sha256": hashlib.sha256(status_text.encode()).hexdigest(),
        }
    except Exception as error:  # noqa: BLE001
        return {"repo": str(repo_root()), "error": f"{type(error).__name__}: {error}"}


def _model_artifact_identity(model_ref: str, model_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve a portable checkpoint identity without hashing frontier weight bytes."""

    identity: dict[str, Any] = {
        "requested": model_ref,
        "revision": str(model_config.get("revision") or "main"),
    }
    try:
        from ..artifacts import build_base_model_manifest
        from ..models import resolve_model, snapshot_dir

        spec = resolve_model(model_ref)
        root = (
            Path(model_config["path"]).expanduser().resolve()
            if model_config.get("path")
            else snapshot_dir(spec).resolve()
        )
        manifest = build_base_model_manifest(
            root,
            name=spec.name,
            hf_id=spec.hf_id,
            family=spec.family,
            revision=identity["revision"],
        )
        identity.update(
            {
                "name": spec.name,
                "hf_id": spec.hf_id,
                "family": spec.family,
                "path": str(root),
                "storage_uri": root.as_uri(),
                "content_hash": manifest["content_hash"],
                "weights": manifest["weights"],
                "metadata_files": manifest["metadata_files"],
                "validation": manifest["validation"],
            }
        )
    except Exception as error:  # noqa: BLE001 - identity failure is recorded with the run
        identity["error"] = f"{type(error).__name__}: {error}"
    return identity


def _tracking(config, run_id, run_dir, results, model_artifact):
    """All results are persisted in the caller-selected local output directory."""
    enabled = config.get("tracking", {}).get("local", {}).get("enabled", True)
    return {"local": {"status": "ok" if enabled else "disabled", "path": str(run_dir)}}


def run_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run all declared runtime/geometry cells and persist a reproducible result bundle."""
    sha = config_sha256(config)
    name = safe_stem(config["experiment"]["name"])
    run_id = f"science-{name}-{sha[:12]}"
    output_root = Path(config["execution"]["output_root"]).expanduser()
    run_dir = output_root / name / run_id
    for section in ("artifacts", "metrics", "reports", "notes"):
        (run_dir / section).mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "artifacts" / "config.json", config, sort_keys=True)
    model_ref = config["model"]["path"] or config["model"]["name"]
    model_artifact = _model_artifact_identity(model_ref, config["model"])

    all_results: list[dict[str, Any]] = []
    model_context_by_runtime: dict[str, dict[str, Any]] = {}
    execution_by_runtime: dict[str, dict[str, Any]] = {}
    for runtime in config["runtimes"]:
        if runtime.get("kind", "engine") == "sciencegraph":
            error = ScienceBenchmarkError(
                f"runtime {runtime['name']!r} declares sciencegraph, but no generic adapter "
                "is registered"
            )
            if config["execution"]["failure_policy"] == "fail":
                raise error
            model_context_by_runtime[runtime["name"]] = {
                "max_context_tokens": None,
                "source": "sciencegraph-adapter-unavailable",
            }
            execution_by_runtime[runtime["name"]] = {
                "backend": runtime["backend"],
                "fabric": "unavailable",
                "device": "unavailable",
                "dtype": "unavailable",
                "reason": "generic sciencegraph adapter is not registered",
            }
            write_json(
                run_dir / "artifacts" / f"runtime-{safe_stem(runtime['name'])}.json",
                {
                    "name": runtime["name"],
                    "backend": runtime["backend"],
                    "cases": [],
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                sort_keys=True,
            )
            continue
        try:
            execution = resolve_runtime_execution(config, runtime)
        except Exception as error:  # noqa: BLE001 - continue policy records an invalid leg
            if config["execution"]["failure_policy"] == "fail":
                raise
            execution = {
                "backend": str(runtime["backend"]),
                "fabric": "unavailable",
                "device": "unavailable",
                "dtype": "unavailable",
                "host_detected": platform.node(),
                "reason": f"resolution failed: {type(error).__name__}: {error}",
            }
            execution_by_runtime[runtime["name"]] = execution
            write_json(
                run_dir / "artifacts" / f"runtime-{safe_stem(runtime['name'])}.json",
                {
                    "name": runtime["name"],
                    "backend": runtime["backend"],
                    "execution": execution,
                    "cases": [],
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                sort_keys=True,
            )
            continue
        execution_by_runtime[runtime["name"]] = execution
        kwargs = _dtype_kwargs(runtime, config["serve"], execution)
        started = time.perf_counter()
        engine: Any | None = None
        load_s: float | None = None
        runtime_results: list[dict[str, Any]] = []
        runtime_error: dict[str, str] | None = None
        model_context = {"max_context_tokens": None, "source": "unavailable"}
        try:
            engine = open_engine(model_ref, backend=runtime["backend"], fresh=True, **kwargs)
            load_s = time.perf_counter() - started
            model_context = _model_context_info(engine)
            max_context = model_context.get("max_context_tokens")
            for batch_size in config["test"]["batch_sizes"]:
                for context_tokens in config["test"]["context_tokens"]:
                    if isinstance(max_context, int) and context_tokens > max_context:
                        raise ScienceBenchmarkError(
                            f"runtime {runtime['name']!r} requests "
                            f"context_tokens={context_tokens}, "
                            f"but the model declares max_context_tokens={max_context} "
                            f"({model_context['source']})"
                        )
                    rows = _fixed_rows(
                        engine,
                        config["test"]["prompts"],
                        batch_size,
                        context_tokens,
                    )
                    metrics = _run_case(
                        engine,
                        serve_type=config["serve"]["type"],
                        rows=rows,
                        decode_tokens=config["test"]["decode_tokens"],
                        warmup=config["test"]["warmup"],
                        repeats=config["test"]["repeats"],
                        reset_cache_each_repeat=config["test"]["reset_cache_each_repeat"],
                    )
                    row = {
                        "runtime": {
                            key: value for key, value in runtime.items() if key != "options"
                        },
                        "execution": execution,
                        "case": {
                            "batch_size": batch_size,
                            "context_tokens": context_tokens,
                            "model_context_limit_tokens": max_context,
                            "model_context_source": model_context["source"],
                        },
                        "metrics": metrics,
                    }
                    runtime_results.append(row)
                    all_results.append(row)
        except Exception as error:  # noqa: BLE001
            if config["execution"]["failure_policy"] == "fail":
                raise
            runtime_error = {"type": type(error).__name__, "message": str(error)}
        finally:
            close = getattr(engine, "close", None) if engine is not None else None
            if callable(close):
                close()
        model_context_by_runtime[runtime["name"]] = model_context
        runtime_record = {
            "name": runtime["name"],
            "backend": runtime["backend"],
            "execution": execution,
            "load_wall_s": load_s,
            "engine_facts": engine_facts(engine) if engine is not None else {},
            "capabilities": _capabilities(engine) if engine is not None else {},
            "model_context": model_context,
            "cases": runtime_results,
            "error": runtime_error,
        }
        write_json(
            run_dir / "artifacts" / f"runtime-{safe_stem(runtime['name'])}.json",
            runtime_record,
            sort_keys=True,
        )

    summary = {
        "schema": "mrun-scientific-result-v1",
        "run_id": run_id,
        "config_sha256": sha,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "source": _source_snapshot(),
        "model": config["model"],
        "model_artifact": model_artifact,
        "serve": config["serve"],
        "model_context_by_runtime": model_context_by_runtime,
        "execution_by_runtime": execution_by_runtime,
        "results": all_results,
    }
    write_json(run_dir / "artifacts" / "benchmark.json", summary, sort_keys=True)
    write_json(run_dir / "metrics" / "metrics.json", all_results, sort_keys=True)
    lines = [
        f"# {config['experiment']['name']}",
        "",
        f"Run: `{run_id}`",
        f"Config SHA-256: `{sha}`",
        "",
    ]
    for result in all_results:
        metric = result["metrics"]
        lines.append(
            f"- {result['runtime']['name']} B={result['case']['batch_size']} "
            f"T={result['case']['context_tokens']}: "
            f"prefill median {metric['prefill_tok_s']['median']:.2f} tok/s; "
            f"generation median {metric['generation_tok_s']['median']}; "
            f"RSS mean/max {metric['resource'].get('rss_mb_mean')}/"
            f"{metric['resource'].get('rss_mb_max')} MiB"
        )
    (run_dir / "reports" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        **summary,
        "tracking": _tracking(config, run_id, run_dir, all_results, model_artifact),
    }
    write_json(run_dir / "manifest.json", manifest, sort_keys=True)
    return {"run_id": run_id, "run_dir": str(run_dir), "manifest": manifest}
