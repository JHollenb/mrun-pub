"""Hardware gate for the dense CUDA selected-row WorkPlan/campaign path.

The gate compares the new body-plus-selected-head route with the established
body-plus-full-logits route on the same engine, QStore, input rows, precision, and
process.  It also proves that selected execution does not touch ``row_blocks``, checks
single- and multi-row batches, exercises repeated determinism, and retains paired raw
latencies.  This is an eager CUDA gate; it makes no CUDA Graph claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ..compiler.benchmark import _paired_median_bootstrap_ci95
from ..engine import open_engine

DEFAULT_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _project_full(
    full_logits: Sequence[torch.Tensor],
    row_ids: Sequence[int],
) -> torch.Tensor:
    indices = torch.as_tensor(tuple(row_ids), dtype=torch.long)
    return torch.stack(
        [torch.as_tensor(logits)[-1].index_select(0, indices) for logits in full_logits]
    ).float()


def _comparison(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, object]:
    left = reference.detach().float().cpu()
    right = actual.detach().float().cpu()
    if left.shape != right.shape:
        return {
            "shape_equal": False,
            "allclose": False,
            "exact": False,
            "max_abs_error": float("inf"),
            "max_rel_error": float("inf"),
        }
    difference = (right - left).abs()
    denominator = left.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "shape_equal": True,
        "allclose": bool(torch.allclose(left, right, rtol=1e-5, atol=2e-5)),
        "exact": bool(torch.equal(left, right)),
        "max_abs_error": float(difference.max()) if difference.numel() else 0.0,
        "max_rel_error": (
            float((difference / denominator).max()) if difference.numel() else 0.0
        ),
        "winner_offsets_exact": bool(
            torch.equal(left.argmax(dim=-1), right.argmax(dim=-1))
        ),
        "compared_values": int(left.numel()),
    }


def _timed(callable_) -> tuple[object, float]:
    started = time.perf_counter()
    result = callable_()
    return result, (time.perf_counter() - started) * 1000.0


def gate(
    model_name: str,
    *,
    prompt: str,
    row_ids: Sequence[int] = DEFAULT_ROWS,
    compact_cache_mb: float = 600.0,
    compute_dtype: str = "bf16",
    warmup: int = 3,
    trials: int = 30,
    determinism_replays: int = 10,
) -> dict[str, object]:
    if warmup < 0 or trials <= 0 or determinism_replays <= 0:
        raise ValueError("warmup must be non-negative; trials/replays must be positive")
    selected_rows = tuple(int(value) for value in row_ids)
    if not selected_rows or len(selected_rows) != len(set(selected_rows)):
        raise ValueError("selected rows must be non-empty and unique")

    with open_engine(
        model_name,
        backend="dense-qstore-cuda",
        compute_dtype=compute_dtype,
        compact_cache_mb=compact_cache_mb,
        max_seq_len=128,
    ) as engine:
        token_ids = engine.encode([prompt], add_special_tokens=False)[0]
        single = [np.asarray(token_ids, dtype=np.int64)]
        multi = [
            single[0],
            np.ascontiguousarray(np.roll(single[0], 1)),
        ]

        full_single = engine.logits_batch(single)
        projected_single = _project_full(full_single, selected_rows)
        selected_single = engine.selected_last_logits_batch(single, selected_rows)
        single_parity = _comparison(projected_single, selected_single)

        full_multi = engine.logits_batch(multi)
        projected_multi = _project_full(full_multi, selected_rows)
        selected_multi = engine.selected_last_logits_batch(multi, selected_rows)
        multi_parity = _comparison(projected_multi, selected_multi)

        deterministic = all(
            torch.equal(
                selected_single,
                engine.selected_last_logits_batch(single, selected_rows),
            )
            for _ in range(determinism_replays)
        )

        def fail_row_blocks(*_args, **_kwargs):
            raise AssertionError("selected-row execution touched full-head row_blocks")

        with patch.object(engine.store, "row_blocks", side_effect=fail_row_blocks):
            no_full_head = engine.selected_last_logits_batch(single, selected_rows)
        no_full_head_proven = torch.equal(no_full_head, selected_single)

        for _ in range(warmup):
            engine.logits_batch(single)
            engine.selected_last_logits_batch(single, selected_rows)

        full_samples: list[float] = []
        selected_samples: list[float] = []
        for trial in range(trials):
            if trial % 2:
                _selected, selected_ms = _timed(
                    lambda: engine.selected_last_logits_batch(single, selected_rows)
                )
                _full, full_ms = _timed(lambda: engine.logits_batch(single))
            else:
                _full, full_ms = _timed(lambda: engine.logits_batch(single))
                _selected, selected_ms = _timed(
                    lambda: engine.selected_last_logits_batch(single, selected_rows)
                )
            full_samples.append(full_ms)
            selected_samples.append(selected_ms)

        ratios = tuple(
            full / selected
            for full, selected in zip(full_samples, selected_samples, strict=True)
        )
        ratio_median = float(np.median(np.asarray(ratios, dtype=np.float64)))
        ratio_ci95 = _paired_median_bootstrap_ci95(ratios)

        engine.store.reset_stats()
        telemetry_output = engine.selected_last_logits_batch(single, selected_rows)
        stats = engine.runtime_stats()
        compact = stats["compact_store"]
        selected_telemetry_ok = (
            compact["selected_head_calls"] == 1
            and compact["selected_head_rows"] == len(selected_rows)
            and compact["selected_head_compact_bytes"]
            == len(selected_rows) * (int(engine.hidden) + 4)
        )

        tf32_disabled = (
            not torch.backends.cuda.matmul.allow_tf32
            and not torch.backends.cudnn.allow_tf32
        )
        verdict = (
            bool(single_parity["allclose"])
            and bool(single_parity["winner_offsets_exact"])
            and bool(multi_parity["allclose"])
            and bool(multi_parity["winner_offsets_exact"])
            and deterministic
            and no_full_head_proven
            and selected_telemetry_ok
            and tf32_disabled
            and tuple(telemetry_output.shape) == (1, len(selected_rows))
            and telemetry_output.dtype is torch.float32
            and telemetry_output.device.type == "cpu"
        )
        return {
            "gate": "dense-cuda-selected-head-v1",
            "verdict": "PASS" if verdict else "FAIL",
            "model": model_name,
            "prompt": prompt,
            "token_ids": [int(value) for value in token_ids],
            "selected_row_ids": list(selected_rows),
            "selected_numerical_contract": engine.subset_head_numerical_contract,
            "single_batch_parity": single_parity,
            "multi_batch_parity": multi_parity,
            "deterministic_replays": {
                "count": determinism_replays,
                "exact": deterministic,
            },
            "full_head_row_blocks_bypassed": no_full_head_proven,
            "selected_output_contract": {
                "shape": list(telemetry_output.shape),
                "dtype": str(telemetry_output.dtype),
                "device": str(telemetry_output.device),
            },
            "timing_ms": {
                "full_samples": full_samples,
                "selected_samples": selected_samples,
                "paired_ratios": list(ratios),
                "full_median": float(np.median(full_samples)),
                "selected_median": float(np.median(selected_samples)),
                "paired_ratio_median": ratio_median,
                "paired_ratio_ci95": list(ratio_ci95),
                "paired_wins": sum(value > 1.0 for value in ratios),
                "speed_improvement_demonstrated": ratio_ci95[0] > 1.01,
                "warmup": warmup,
                "trials": trials,
            },
            "selected_head_telemetry": {
                "selected_head_calls": compact["selected_head_calls"],
                "selected_head_rows": compact["selected_head_rows"],
                "selected_head_compact_bytes": compact[
                    "selected_head_compact_bytes"
                ],
                "expected_compact_bytes": len(selected_rows)
                * (int(engine.hidden) + 4),
                "exact": selected_telemetry_ok,
            },
            "runtime": {
                "host": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "compact_cache_mb": compact_cache_mb,
                "compute_dtype": compute_dtype,
                "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
                "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
            },
            "claims": {
                "execution": "measured-dense-cuda-eager-selected-row",
                "cuda_graph_replay": False,
                "canonical_hf_quality": False,
                "energy": False,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen2.5-0.5b")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--row-id",
        type=int,
        action="append",
        default=[],
        help="selected vocabulary row; repeat (default 1..9)",
    )
    parser.add_argument("--compact-cache-mb", type=float, default=600.0)
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--determinism-replays", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = gate(
        args.model,
        prompt=args.prompt,
        row_ids=tuple(args.row_id) or DEFAULT_ROWS,
        compact_cache_mb=args.compact_cache_mb,
        compute_dtype=args.compute_dtype,
        warmup=args.warmup,
        trials=args.trials,
        determinism_replays=args.determinism_replays,
    )
    rendered = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    print(rendered)
    if args.out is not None:
        args.out.write_text(rendered + "\n")
        print(f"gate json -> {args.out}", flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
