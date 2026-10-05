"""Lifecycle CLI for the first-class paged Qwen3 MoE CUDA backend."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..io import write_json
from ..models import resolve_model
from .qwen3_moe_cuda import (
    DEFAULT_CACHE_MB,
    DEFAULT_MAX_ACTIVE_PAGES,
    EXPERT_CODEC_FP8,
    EXPERT_CODEC_W4,
    SUPPORTED_CACHE_POLICIES,
    SUPPORTED_PAGE_BINDING_POLICIES,
    SUPPORTED_PREFILL_PAGE_POLICIES,
    SUPPORTED_ROUTE_REDUCTION_POLICIES,
    SUPPORTED_W4_ARITHMETIC_POLICIES,
    PackedBF16ExpertBackend,
    Qwen3MoeCudaEngine,
    Qwen3MoeDecodeRuntime,
    TensorReader,
    _model_dir,
    build_fp8_expert_store,
    build_int4_expert_store,
    resolve_qwen3_moe_store_dir,
    validate_fp8_expert_store,
    validate_int4_expert_store,
)

RESULT_SCHEMA = "mrun.qwen3-moe-cuda.result.v1"


def _emit(result: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        write_json(output, result, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


def _engine(args: argparse.Namespace) -> Qwen3MoeCudaEngine:
    return Qwen3MoeCudaEngine(
        args.model,
        store_dir=args.store_dir,
        expert_codec=args.expert_codec,
        cache_mb=args.cache_mb,
        max_active_pages=args.max_active_pages,
        host_cache_mb=args.host_cache_mb,
        warm_host=args.warm_host,
        route_prefetch=args.route_prefetch,
        cache_policy=args.cache_policy,
        page_binding_policy=args.page_binding_policy,
        prefill_page_policy=args.prefill_page_policy,
        route_reduction_policy=args.route_reduction_policy,
        w4_arithmetic_policy=args.w4_arithmetic_policy,
        verify_store_content=getattr(args, "verify_content", False),
    )


def _exact_context_ids(
    engine: Qwen3MoeCudaEngine,
    prompts: list[str],
    context_tokens: int,
) -> torch.Tensor:
    if context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    rows = engine.tokenizer(
        prompts,
        add_special_tokens=False,
    )["input_ids"]
    exact: list[list[int]] = []
    for row in rows:
        if not row:
            raise ValueError("benchmark prompt tokenized to an empty row")
        repeats = (context_tokens + len(row) - 1) // len(row)
        exact.append((row * repeats)[-context_tokens:])
    return torch.as_tensor(exact, dtype=torch.long, device=engine.device)


def _decode_sample(
    engine: Qwen3MoeCudaEngine,
    input_ids: torch.Tensor,
    *,
    decode_steps: int,
    clear_pages: bool,
) -> dict[str, Any]:
    if decode_steps < 1:
        raise ValueError("decode_steps must be positive")
    engine.reset_page_cache(clear_pages=clear_pages)
    runtime = engine._require_runtime()
    cache = runtime.new_cache(
        batch_size=int(input_ids.shape[0]),
        capacity=int(input_ids.shape[1]) + decode_steps,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = runtime.forward(input_ids, cache=cache)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - started
    token = result.logits.argmax(dim=-1)
    generated = [token.detach().cpu()]
    step_ms: list[float] = []
    for _ in range(decode_steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = runtime.forward(token[:, None], cache=cache)
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - started) * 1000.0)
        token = result.logits.argmax(dim=-1)
        generated.append(token.detach().cpu())
    decode_s = sum(step_ms) / 1000.0
    generated_array = torch.stack(generated, dim=1).numpy()
    return {
        "batch_size": int(input_ids.shape[0]),
        "context_tokens": int(input_ids.shape[1]),
        "decode_steps_after_prefill": decode_steps,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tok_s": int(input_ids.shape[0]) * decode_steps / decode_s,
        "median_itl_ms": statistics.median(step_ms),
        "p95_itl_ms": _percentile(step_ms, 0.95),
        "generated_ids": generated_array.tolist(),
        "kv_cache_bytes": cache.device_bytes,
        "expert_runtime": engine.cache_stats(),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _trace(
    runtime: Qwen3MoeDecodeRuntime,
    input_ids: torch.Tensor,
    *,
    steps: int,
) -> dict[str, Any]:
    cache = runtime.new_cache(
        batch_size=int(input_ids.shape[0]),
        capacity=int(input_ids.shape[1]) + steps + 1,
    )
    logits: list[torch.Tensor] = []
    routes: list[list[torch.Tensor]] = []
    generated: list[torch.Tensor] = []
    result = runtime.forward(input_ids, cache=cache, capture_routes=True)
    logits.append(result.logits.detach().float().cpu())
    routes.append(result.routes)
    token = result.logits.argmax(dim=-1)
    generated.append(token.detach().cpu())
    for _ in range(steps):
        result = runtime.forward(
            token[:, None],
            cache=cache,
            capture_routes=True,
        )
        logits.append(result.logits.detach().float().cpu())
        routes.append(result.routes)
        token = result.logits.argmax(dim=-1)
        generated.append(token.detach().cpu())
    return {
        "logits": torch.stack(logits),
        "routes": routes,
        "generated": torch.stack(generated, dim=1),
    }


def _route_overlap(
    reference: list[list[torch.Tensor]],
    candidate: list[list[torch.Tensor]],
) -> float:
    intersections = 0
    unions = 0
    for reference_step, candidate_step in zip(reference, candidate, strict=True):
        for reference_layer, candidate_layer in zip(
            reference_step,
            candidate_step,
            strict=True,
        ):
            for reference_row, candidate_row in zip(
                reference_layer.reshape(-1, reference_layer.shape[-1]),
                candidate_layer.reshape(-1, candidate_layer.shape[-1]),
                strict=True,
            ):
                left = {int(value) for value in reference_row}
                right = {int(value) for value in candidate_row}
                intersections += len(left & right)
                unions += len(left | right)
    return intersections / max(1, unions)


def _compare_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    left = reference.float().reshape(-1)
    right = candidate.float().reshape(-1)
    delta = right - left
    return {
        "cosine": float(F.cosine_similarity(left, right, dim=0).item()),
        "relative_l2": float((delta.norm() / left.norm().clamp_min(1e-12)).item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
    }


def _build(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = _model_dir(args.model)
    candidate = Path(args.model).expanduser()
    if candidate.is_dir():
        if not args.source_hf_id:
            raise ValueError("--source-hf-id is required for an explicit checkpoint path")
        model_name = args.source_model_name or candidate.name
        hf_id = args.source_hf_id
    else:
        spec = resolve_model(args.model)
        model_name = args.source_model_name or spec.name
        hf_id = args.source_hf_id or spec.hf_id
    store_dir = resolve_qwen3_moe_store_dir(
        model_dir,
        args.store_dir,
        expert_codec=args.expert_codec,
    )
    builder = (
        build_fp8_expert_store if args.expert_codec == EXPERT_CODEC_FP8 else build_int4_expert_store
    )
    manifest = builder(
        model_dir,
        store_dir,
        model_name=model_name,
        hf_id=hf_id,
        revision=args.source_revision,
    )
    return {
        "schema": RESULT_SCHEMA,
        "phase": "build-store",
        "status": "ok",
        "model": args.model,
        "model_dir": str(model_dir),
        "store": {
            "path": str(store_dir),
            "schema": manifest["schema_version"],
            "codec": manifest["codec"],
            "data_bytes": int(manifest["data_bytes"]),
            "pages": int(manifest["layers"]) * int(manifest["num_experts"]),
            "page_stride": int(manifest["layout"]["page_stride"]),
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
            "derived_store_sha256": manifest["derived"]["derived_store_sha256"],
        },
    }


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = _model_dir(args.model)
    store_dir = resolve_qwen3_moe_store_dir(
        model_dir,
        args.store_dir,
        expert_codec=args.expert_codec,
    )
    validator = (
        validate_fp8_expert_store
        if args.expert_codec == EXPERT_CODEC_FP8
        else validate_int4_expert_store
    )
    manifest = validator(
        model_dir,
        store_dir,
        verify_content=args.verify_content,
    )
    return {
        "schema": RESULT_SCHEMA,
        "phase": "inspect",
        "status": "ok",
        "model": args.model,
        "model_dir": str(model_dir),
        "store": {
            "path": str(store_dir),
            "schema": manifest["schema_version"],
            "codec": manifest["codec"],
            "data_bytes": int(manifest["data_bytes"]),
            "validation": manifest["validation"],
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
            "derived_store_sha256": manifest["derived"]["derived_store_sha256"],
        },
    }


def _generate(args: argparse.Namespace) -> dict[str, Any]:
    with _engine(args) as engine:
        started = time.perf_counter()
        generated_ids = engine.generate_batch(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - started
        texts = [
            engine.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_ids
        ]
        return {
            "schema": RESULT_SCHEMA,
            "phase": "generate",
            "status": "ok",
            "model": args.model,
            "prompts": args.prompt,
            "generated_ids": generated_ids,
            "generated_text": texts,
            "max_new_tokens": args.max_new_tokens,
            "wall_s": wall_s,
            "aggregate_output_tok_s": (len(args.prompt) * args.max_new_tokens / wall_s),
            "runtime": engine.runtime_report(),
        }


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    with _engine(args) as engine:
        samples: list[dict[str, Any]] = []
        for batch_size in batch_sizes:
            prompts = [f"{args.prompt} Request {index}." for index in range(batch_size)]
            input_ids = _exact_context_ids(engine, prompts, args.context_tokens)
            for repeat in range(args.repeats):
                samples.append(
                    {
                        "temperature": "cold" if repeat == 0 else "warm",
                        "repeat": repeat,
                        **_decode_sample(
                            engine,
                            input_ids,
                            decode_steps=args.decode_steps,
                            clear_pages=repeat == 0,
                        ),
                    }
                )
        summary = []
        for batch_size in batch_sizes:
            selected = [row for row in samples if row["batch_size"] == batch_size]
            warm = [row for row in selected if row["temperature"] == "warm"] or selected
            summary.append(
                {
                    "batch_size": batch_size,
                    "cold_decode_tok_s": selected[0]["decode_tok_s"],
                    "warm_decode_tok_s_median": statistics.median(
                        row["decode_tok_s"] for row in warm
                    ),
                    "warm_itl_ms_median": statistics.median(row["median_itl_ms"] for row in warm),
                    "warm_page_hit_rate_median": statistics.median(
                        row["expert_runtime"]["page_cache"]["hit_rate"] for row in warm
                    ),
                    "warm_h2d_gb_median": statistics.median(
                        row["expert_runtime"]["page_cache"]["host_to_device_gb"] for row in warm
                    ),
                }
            )
        return {
            "schema": RESULT_SCHEMA,
            "phase": "benchmark",
            "status": "ok",
            "model": args.model,
            "input": {
                "prompt_template": args.prompt,
                "batch_sizes": batch_sizes,
                "context_tokens": args.context_tokens,
                "decode_steps_after_prefill": args.decode_steps,
                "repeats": args.repeats,
            },
            "summary": summary,
            "samples": samples,
            "runtime": engine.runtime_report(),
            "claim_boundary": [
                "decode_tok_s excludes prefill and counts complete generated tokens",
                "cold and warm expert-cache measurements are separate",
                "aggregate Bn throughput is not single-request latency",
            ],
        }


def _quality(args: argparse.Namespace) -> dict[str, Any]:
    with _engine(args) as engine:
        input_ids = _exact_context_ids(
            engine,
            [args.prompt],
            args.context_tokens,
        )
        reference_backend = PackedBF16ExpertBackend(
            TensorReader(engine.model_dir),
            device=engine.device,
            dtype=engine.dtype,
        )
        reference_runtime = Qwen3MoeDecodeRuntime(
            engine.skeleton,
            reference_backend,
        )
        reference = _trace(
            reference_runtime,
            input_ids,
            steps=args.decode_steps,
        )
        engine.reset_page_cache(clear_pages=True)
        candidate = _trace(
            engine._require_runtime(),
            input_ids,
            steps=args.decode_steps,
        )
        logits = _compare_logits(reference["logits"], candidate["logits"])
        route_overlap = _route_overlap(
            reference["routes"],
            candidate["routes"],
        )
        token_agreement = float(
            (reference["generated"] == candidate["generated"]).float().mean().item()
        )
        gates = {
            "logit_cosine_gte_0_995": logits["cosine"] >= 0.995,
            "relative_l2_lte_0_10": logits["relative_l2"] <= 0.10,
            "route_overlap_gte_0_95": route_overlap >= 0.95,
            "greedy_token_agreement_gte_0_75": token_agreement >= 0.75,
        }
        passed = all(gates.values())
        return {
            "schema": RESULT_SCHEMA,
            "phase": "quality",
            "status": "ok" if passed else "failed-gate",
            "passed": passed,
            "model": args.model,
            "input": {
                "prompt": args.prompt,
                "context_tokens": args.context_tokens,
                "decode_steps_after_prefill": args.decode_steps,
            },
            "comparison": {
                "logits": logits,
                "route_set_jaccard": route_overlap,
                "greedy_token_agreement": token_agreement,
                "reference_generated_ids": reference["generated"].tolist(),
                "candidate_generated_ids": candidate["generated"].tolist(),
            },
            "gates": gates,
            "runtime": engine.runtime_report(),
            "claim_boundary": [
                "The oracle is this runtime with original BF16 expert matrices.",
                "This is a bounded prompt and continuation gate, not a broad benchmark.",
            ],
        }


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="qwen3-30b-a3b")
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument(
        "--expert-codec",
        choices=(EXPERT_CODEC_FP8, EXPERT_CODEC_W4),
        default=EXPERT_CODEC_FP8,
    )
    parser.add_argument("--cache-mb", type=float, default=DEFAULT_CACHE_MB)
    parser.add_argument(
        "--max-active-pages",
        type=int,
        default=DEFAULT_MAX_ACTIVE_PAGES,
    )
    parser.add_argument(
        "--host-cache-mb",
        type=float,
        default=None,
        help="host-RAM expert-page tier budget; omitted defers to the engine/env default",
    )
    parser.add_argument(
        "--warm-host",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="sequentially prewarm the host page tier; omitted defers to the engine/env default",
    )
    parser.add_argument(
        "--route-prefetch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="prefetch the previous step's predicted next-layer routes",
    )
    parser.add_argument(
        "--cache-policy",
        choices=tuple(sorted(SUPPORTED_CACHE_POLICIES)),
        default=None,
        help="CUDA expert-page cache policy; omitted defers to the engine/env default",
    )
    parser.add_argument(
        "--page-binding-policy",
        choices=tuple(sorted(SUPPORTED_PAGE_BINDING_POLICIES)),
        default=None,
        help="bind routed experts by compact copy or physical cache-slot indirection",
    )
    parser.add_argument(
        "--prefill-page-policy",
        choices=tuple(sorted(SUPPORTED_PREFILL_PAGE_POLICIES)),
        default=None,
        help="expert-page admission policy for prefill",
    )
    parser.add_argument(
        "--route-reduction-policy",
        choices=tuple(sorted(SUPPORTED_ROUTE_REDUCTION_POLICIES)),
        default=None,
        help="expert-contribution reduction order and numerical ABI",
    )
    parser.add_argument(
        "--w4-arithmetic-policy",
        choices=tuple(sorted(SUPPORTED_W4_ARITHMETIC_POLICIES)),
        default=None,
        help="W4 grouped-GEMM numerical ABI; omitted preserves pre-dot BF16 reconstruction",
    )
    parser.add_argument("--verify-content", action="store_true")
    parser.add_argument("--output", type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mrun qwen3-moe",
        description="Build, validate, quality-gate, and run paged Qwen3 MoE CUDA inference",
    )
    sub = parser.add_subparsers(dest="phase", required=True)

    build = sub.add_parser("build-store", help="compile routed experts into paged FP8 or W4")
    build.add_argument("--model", default="qwen3-30b-a3b")
    build.add_argument("--store-dir", type=Path)
    build.add_argument(
        "--expert-codec",
        choices=(EXPERT_CODEC_FP8, EXPERT_CODEC_W4),
        default=EXPERT_CODEC_FP8,
    )
    build.add_argument("--source-model-name")
    build.add_argument("--source-hf-id")
    build.add_argument("--source-revision")
    build.add_argument("--output", type=Path)

    inspect = sub.add_parser("inspect", help="validate store geometry and lineage")
    inspect.add_argument("--model", default="qwen3-30b-a3b")
    inspect.add_argument("--store-dir", type=Path)
    inspect.add_argument(
        "--expert-codec",
        choices=(EXPERT_CODEC_FP8, EXPERT_CODEC_W4),
        default=EXPERT_CODEC_FP8,
    )
    inspect.add_argument("--verify-content", action="store_true")
    inspect.add_argument("--output", type=Path)

    generate = sub.add_parser("generate", help="run greedy persistent-KV inference")
    _add_runtime_options(generate)
    generate.add_argument("--prompt", action="append", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=16)

    benchmark = sub.add_parser(
        "benchmark",
        help="measure cold and warm post-prefill decode throughput",
    )
    _add_runtime_options(benchmark)
    benchmark.add_argument(
        "--prompt",
        default="Efficient inference routes only useful expert pages.",
    )
    benchmark.add_argument("--batch-sizes", default="1,8")
    benchmark.add_argument("--context-tokens", type=int, default=4)
    benchmark.add_argument("--decode-steps", type=int, default=4)
    benchmark.add_argument("--repeats", type=int, default=2)

    quality = sub.add_parser(
        "quality",
        help="compare paged quantized execution with on-demand BF16 experts",
    )
    _add_runtime_options(quality)
    quality.add_argument(
        "--prompt",
        default="Efficient inference routes only useful expert pages.",
    )
    quality.add_argument("--context-tokens", type=int, default=4)
    quality.add_argument("--decode-steps", type=int, default=2)

    args = parser.parse_args(argv)
    try:
        if args.phase == "build-store":
            result = _build(args)
        elif args.phase == "inspect":
            result = _inspect(args)
        elif args.phase == "generate":
            result = _generate(args)
        elif args.phase == "benchmark":
            result = _benchmark(args)
        else:
            result = _quality(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    _emit(result, args.output)
    if args.phase == "quality" and not result["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
