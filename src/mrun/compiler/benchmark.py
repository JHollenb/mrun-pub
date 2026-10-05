"""Local eager parity and latency harness for executable WorkPlans."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from .executable import execute_direct_contract, execute_lowered_plan
from .graph_passes import GraphCompilation
from .ir import DenseWorkPlan, OutputContract
from .lowering import LoweredWorkPlan


@dataclass(frozen=True)
class OutputParity:
    allclose: bool
    exact: bool
    compared_values: int
    max_abs_error: float
    max_rel_error: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "allclose": self.allclose,
            "exact": self.exact,
            "compared_values": self.compared_values,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
        }


@dataclass(frozen=True)
class EagerBenchmarkResult:
    direct_ms: float
    plan_median_ms: float
    plan_p95_ms: float
    direct_to_plan_ratio: float
    trials: int
    warmup: int
    parity: OutputParity
    reported_fabric: str
    working_set_mb: float | None
    direct_max_dequant_block_mb: float | None = None
    plan_max_dequant_block_mb: float | None = None
    benchmark_basis: str = "measured-local-eager"

    @property
    def max_dequant_block_mb(self) -> float | None:
        """Precisely name the legacy ``working_set_mb`` engine observation.

        ``PagedEngine.working_set_mb`` tracks the largest dequantized weight block seen by
        its QStore. It is not process RSS, allocator high-water, or total working-set memory.
        The old field remains available for API compatibility.
        """

        leg_values = tuple(
            value
            for value in (
                self.direct_max_dequant_block_mb,
                self.plan_max_dequant_block_mb,
            )
            if value is not None
        )
        return max(leg_values) if leg_values else self.working_set_mb

    def as_dict(self) -> dict[str, Any]:
        return {
            "direct_ms": self.direct_ms,
            "plan_median_ms": self.plan_median_ms,
            "plan_p95_ms": self.plan_p95_ms,
            "direct_to_plan_ratio": self.direct_to_plan_ratio,
            "trials": self.trials,
            "warmup": self.warmup,
            "parity": self.parity.as_dict(),
            "reported_fabric": self.reported_fabric,
            "working_set_mb": self.working_set_mb,
            "max_dequant_block_mb": self.max_dequant_block_mb,
            "direct_max_dequant_block_mb": self.direct_max_dequant_block_mb,
            "plan_max_dequant_block_mb": self.plan_max_dequant_block_mb,
            "benchmark_basis": self.benchmark_basis,
        }


@dataclass(frozen=True)
class OutputSlicingBenchmarkResult:
    """Paired full-head-baseline versus output-sliced measurements.

    The baseline always runs ``engine.logits_batch`` and only then projects the full logits
    to the requested public output contract. The sliced leg runs the plan's specialized
    direct contract. This makes the comparison materially different from
    :class:`EagerBenchmarkResult`, whose two legs intentionally exercise the same numerical
    path to measure WorkPlan wrapper overhead.
    """

    output_contract: str
    baseline_samples_ms: tuple[float, ...]
    sliced_samples_ms: tuple[float, ...]
    paired_speedups: tuple[float, ...]
    baseline_median_ms: float
    baseline_p95_ms: float
    baseline_min_ms: float
    baseline_max_ms: float
    sliced_median_ms: float
    sliced_p95_ms: float
    sliced_min_ms: float
    sliced_max_ms: float
    baseline_to_sliced_ratio: float
    paired_speedup_median: float
    paired_speedup_ci95: tuple[float, float]
    paired_wins: int
    latency_reduction_percent: float
    minimum_improvement_percent: float
    improvement_demonstrated: bool
    trials: int
    warmup: int
    parity: OutputParity
    reported_fabric: str
    max_dequant_block_mb: float | None
    baseline_max_dequant_block_mb: float | None = None
    sliced_max_dequant_block_mb: float | None = None
    graph_compilation_fingerprint: str | None = None
    graph_rewrite_ids: tuple[str, ...] = ()
    manual_sliced_samples_ms: tuple[float, ...] = ()
    manual_sliced_median_ms: float | None = None
    manual_sliced_p95_ms: float | None = None
    manual_to_graph_ratio: float | None = None
    manual_to_graph_paired_ratios: tuple[float, ...] = ()
    manual_to_graph_paired_median: float | None = None
    manual_to_graph_paired_ci95: tuple[float, float] | None = None
    graph_binding_overhead_percent: float | None = None
    manual_graph_parity: OutputParity | None = None
    manual_max_dequant_block_mb: float | None = None
    benchmark_basis: str = "measured-full-head-baseline-vs-output-sliced"

    @property
    def speedup(self) -> float:
        """Median-latency ratio; values greater than one favor output slicing."""

        return self.baseline_to_sliced_ratio

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_contract": self.output_contract,
            "baseline_samples_ms": list(self.baseline_samples_ms),
            "sliced_samples_ms": list(self.sliced_samples_ms),
            "paired_speedups": list(self.paired_speedups),
            "baseline_median_ms": self.baseline_median_ms,
            "baseline_p95_ms": self.baseline_p95_ms,
            "baseline_min_ms": self.baseline_min_ms,
            "baseline_max_ms": self.baseline_max_ms,
            "sliced_median_ms": self.sliced_median_ms,
            "sliced_p95_ms": self.sliced_p95_ms,
            "sliced_min_ms": self.sliced_min_ms,
            "sliced_max_ms": self.sliced_max_ms,
            "baseline_to_sliced_ratio": self.baseline_to_sliced_ratio,
            "speedup": self.speedup,
            "paired_speedup_median": self.paired_speedup_median,
            "paired_speedup_ci95": list(self.paired_speedup_ci95),
            "paired_wins": self.paired_wins,
            "latency_reduction_percent": self.latency_reduction_percent,
            "minimum_improvement_percent": self.minimum_improvement_percent,
            "improvement_demonstrated": self.improvement_demonstrated,
            "trials": self.trials,
            "warmup": self.warmup,
            "parity": self.parity.as_dict(),
            "reported_fabric": self.reported_fabric,
            "max_dequant_block_mb": self.max_dequant_block_mb,
            "baseline_max_dequant_block_mb": self.baseline_max_dequant_block_mb,
            "sliced_max_dequant_block_mb": self.sliced_max_dequant_block_mb,
            "graph_compilation_fingerprint": self.graph_compilation_fingerprint,
            "graph_rewrite_ids": list(self.graph_rewrite_ids),
            "manual_sliced_samples_ms": list(self.manual_sliced_samples_ms),
            "manual_sliced_median_ms": self.manual_sliced_median_ms,
            "manual_sliced_p95_ms": self.manual_sliced_p95_ms,
            "manual_to_graph_ratio": self.manual_to_graph_ratio,
            "manual_to_graph_paired_ratios": list(self.manual_to_graph_paired_ratios),
            "manual_to_graph_paired_median": self.manual_to_graph_paired_median,
            "manual_to_graph_paired_ci95": (
                None
                if self.manual_to_graph_paired_ci95 is None
                else list(self.manual_to_graph_paired_ci95)
            ),
            "graph_binding_overhead_percent": self.graph_binding_overhead_percent,
            "manual_graph_parity": (
                None if self.manual_graph_parity is None else self.manual_graph_parity.as_dict()
            ),
            "manual_max_dequant_block_mb": self.manual_max_dequant_block_mb,
            "benchmark_basis": self.benchmark_basis,
        }


def _sync_engine(engine: Any) -> None:
    device = torch.device(getattr(engine, "device", "cpu"))
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _reset_max_dequant_block(engine: Any) -> bool:
    """Reset QStore block telemetry when no resident cache can hide block loads."""

    store = getattr(engine, "store", None)
    if store is None or not hasattr(store, "max_block_bytes"):
        return False
    if int(getattr(store, "_cache_budget", 0)) > 0:
        return False
    store.max_block_bytes = 0
    return True


def _max_dequant_block_mb(engine: Any) -> float | None:
    store = getattr(engine, "store", None)
    if store is None or not hasattr(store, "max_block_bytes"):
        return None
    return float(store.max_block_bytes) / 1e6


def _compare_outputs(
    reference: Any,
    actual: Any,
    *,
    rtol: float,
    atol: float,
) -> OutputParity:
    counts = 0
    max_abs = 0.0
    max_rel = 0.0
    allclose = True
    exact = True

    def visit(left: Any, right: Any) -> None:
        nonlocal counts, max_abs, max_rel, allclose, exact
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            left_tensor = torch.as_tensor(left).detach().cpu()
            right_tensor = torch.as_tensor(right).detach().cpu()
            if left_tensor.shape != right_tensor.shape:
                allclose = exact = False
                return
            left_float = left_tensor.double()
            right_float = right_tensor.double()
            difference = (left_float - right_float).abs()
            denominator = left_float.abs().clamp_min(torch.finfo(torch.float64).tiny)
            counts += left_tensor.numel()
            if difference.numel():
                max_abs = max(max_abs, float(difference.max()))
                max_rel = max(max_rel, float((difference / denominator).max()))
            allclose = allclose and bool(
                torch.allclose(left_float, right_float, rtol=rtol, atol=atol, equal_nan=True)
            )
            exact = exact and bool(torch.equal(left_tensor, right_tensor))
            return
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            visit(torch.as_tensor(left), torch.as_tensor(right))
            return
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                allclose = exact = False
                return
            for key in left:
                visit(left[key], right[key])
            return
        if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
            if len(left) != len(right):
                allclose = exact = False
                return
            for left_value, right_value in zip(left, right, strict=True):
                visit(left_value, right_value)
            return
        if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
            left_float = float(left)
            right_float = float(right)
            difference = abs(left_float - right_float)
            counts += 1
            max_abs = max(max_abs, difference)
            max_rel = max(max_rel, difference / max(abs(left_float), 1e-300))
            allclose = allclose and math.isclose(
                left_float,
                right_float,
                rel_tol=rtol,
                abs_tol=atol,
            )
            exact = exact and left == right
            return
        allclose = allclose and left == right
        exact = exact and left == right

    visit(reference, actual)
    return OutputParity(
        allclose=allclose,
        exact=exact,
        compared_values=counts,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
    )


def _project_full_logits(
    plan: DenseWorkPlan,
    logits: Sequence[torch.Tensor],
) -> Any:
    """Project materialized full logits to an output-slicing contract."""

    rows = [torch.as_tensor(row) for row in logits]
    if len(rows) != plan.shape.actual_batch:
        raise RuntimeError("engine did not return one full-logits tensor per input row")
    if any(row.ndim != 2 or row.shape[0] != plan.shape.sequence_length for row in rows):
        raise RuntimeError("full-head baseline logits do not match the planned sequence shape")

    if plan.output_contract is OutputContract.LAST_TOKEN_LOGITS:
        return torch.stack([row[-1] for row in rows])
    if plan.output_contract is OutputContract.SELECTED_TOKEN_ROWS:
        indices = torch.as_tensor(plan.required_output_rows, dtype=torch.long)
        return torch.stack([row[-1].index_select(0, indices.to(device=row.device)) for row in rows])
    if plan.output_contract is OutputContract.CANDIDATE_ARGMAX_AND_MARGIN:
        outputs: list[dict[str, Any]] = []
        for row, candidates in zip(rows, plan.candidate_token_ids, strict=True):
            indices = torch.as_tensor(candidates, dtype=torch.long, device=row.device)
            values = row[-1].index_select(0, indices).float()
            top = torch.topk(values, k=2)
            winner_offset = int(top.indices[0])
            runner_offset = int(top.indices[1])
            outputs.append(
                {
                    "winner_token_id": int(candidates[winner_offset]),
                    "runner_up_token_id": int(candidates[runner_offset]),
                    "winner_logit": float(top.values[0]),
                    "runner_up_logit": float(top.values[1]),
                    "margin": float(top.values[0] - top.values[1]),
                    "candidate_token_ids": tuple(int(token) for token in candidates),
                }
            )
        return tuple(outputs)
    raise ValueError(
        "output-slicing benchmark requires last_token_logits, selected_token_rows, "
        "or candidate_argmax_and_margin"
    )


def _full_head_baseline(
    engine: Any,
    plan: DenseWorkPlan,
    ids_list: Sequence[np.ndarray],
) -> Any:
    """Execute the deliberately unsliced baseline and project only after materialization."""

    return _project_full_logits(plan, list(engine.logits_batch(list(ids_list))))


def _duration_summary(samples: Sequence[float]) -> tuple[float, float, float, float]:
    ordered = sorted(float(value) for value in samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return (
        float(np.median(np.asarray(ordered, dtype=np.float64))),
        ordered[p95_index],
        ordered[0],
        ordered[-1],
    )


def _paired_median_bootstrap_ci95(
    samples: Sequence[float],
    *,
    resamples: int = 10_000,
) -> tuple[float, float]:
    """Return a deterministic paired-bootstrap interval for the median speedup."""

    values = np.asarray(samples, dtype=np.float64)
    if values.size == 1:
        value = float(values[0])
        return value, value
    random = np.random.default_rng(0)
    indices = random.integers(0, values.size, size=(resamples, values.size))
    medians = np.median(values[indices], axis=1)
    low, high = np.percentile(medians, (2.5, 97.5))
    return float(low), float(high)


def benchmark_eager_plan(
    engine: Any,
    plan: DenseWorkPlan,
    lowered: LoweredWorkPlan,
    ids_list: Sequence[np.ndarray | Sequence[int]],
    *,
    labels_list: Sequence[np.ndarray | Sequence[int]] | None = None,
    warmup: int = 1,
    trials: int = 5,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> EagerBenchmarkResult:
    """Measure direct eager vs WorkPlan dispatch and prove output-contract parity."""

    if warmup < 0 or trials <= 0:
        raise ValueError("warmup must be non-negative and trials must be positive")
    # Establish parity output before timing so one-time model/page initialization does not
    # privilege whichever path happens to be measured second.
    reference = execute_direct_contract(
        engine,
        plan,
        ids_list,
        labels_list=labels_list,
    )

    for _ in range(warmup):
        execute_direct_contract(
            engine,
            plan,
            ids_list,
            labels_list=labels_list,
        )
        execute_lowered_plan(
            engine,
            plan,
            lowered,
            ids_list,
            labels_list=labels_list,
        )
    direct_durations: list[float] = []
    plan_durations: list[float] = []
    direct_max_blocks: list[float] = []
    plan_max_blocks: list[float] = []
    actual: Any = None
    for trial in range(trials):
        order = ("direct", "plan") if trial % 2 == 0 else ("plan", "direct")
        for path in order:
            _sync_engine(engine)
            telemetry_reset = _reset_max_dequant_block(engine)
            started = time.perf_counter()
            if path == "direct":
                execute_direct_contract(
                    engine,
                    plan,
                    ids_list,
                    labels_list=labels_list,
                )
            else:
                result = execute_lowered_plan(
                    engine,
                    plan,
                    lowered,
                    ids_list,
                    labels_list=labels_list,
                )
                actual = result.outputs
            _sync_engine(engine)
            duration = (time.perf_counter() - started) * 1000
            (direct_durations if path == "direct" else plan_durations).append(duration)
            if telemetry_reset:
                observed = _max_dequant_block_mb(engine)
                if observed is not None:
                    (direct_max_blocks if path == "direct" else plan_max_blocks).append(observed)

    parity = _compare_outputs(reference, actual, rtol=rtol, atol=atol)
    sorted_durations = sorted(plan_durations)
    p95_index = min(len(sorted_durations) - 1, math.ceil(0.95 * len(sorted_durations)) - 1)
    direct_ms = float(np.median(np.asarray(direct_durations, dtype=np.float64)))
    median_ms = float(np.median(np.asarray(plan_durations, dtype=np.float64)))
    working_set = getattr(engine, "working_set_mb", None)
    direct_max_block = max(direct_max_blocks) if direct_max_blocks else None
    plan_max_block = max(plan_max_blocks) if plan_max_blocks else None
    return EagerBenchmarkResult(
        direct_ms=direct_ms,
        plan_median_ms=median_ms,
        plan_p95_ms=sorted_durations[p95_index],
        direct_to_plan_ratio=direct_ms / median_ms if median_ms else math.inf,
        trials=trials,
        warmup=warmup,
        parity=parity,
        reported_fabric=lowered.reported_fabric,
        working_set_mb=None if working_set is None else float(working_set),
        direct_max_dequant_block_mb=direct_max_block,
        plan_max_dequant_block_mb=plan_max_block,
    )


def benchmark_output_slicing(
    engine: Any,
    plan: DenseWorkPlan,
    lowered: LoweredWorkPlan,
    ids_list: Sequence[np.ndarray | Sequence[int]],
    *,
    warmup: int = 1,
    trials: int = 5,
    rtol: float = 1e-5,
    atol: float = 2e-5,
    minimum_improvement_percent: float = 1.0,
    graph_compilation: GraphCompilation | None = None,
) -> OutputSlicingBenchmarkResult:
    """Measure an honest unsliced full-head baseline against output slicing.

    The public result contract is identical in both legs:

    * baseline: full ``[B,T,V]`` logits, followed by host-side projection;
    * sliced: the plan's last-token, selected-row, or candidate-head specialization.

    With graph compilation, the benchmark rotates the full-head baseline, established
    manual specialization, and graph-bound specialization across trial order. Without it,
    the two available legs alternate. Samples at the same tuple index form a pair.
    Compilation time and process peak memory are outside this measurement. The optional
    memory observation is explicitly the largest dequantized QStore block only.
    """

    supported_contracts = {
        OutputContract.LAST_TOKEN_LOGITS,
        OutputContract.SELECTED_TOKEN_ROWS,
        OutputContract.CANDIDATE_ARGMAX_AND_MARGIN,
    }
    if plan.output_contract not in supported_contracts:
        raise ValueError(
            "output-slicing benchmark requires last_token_logits, selected_token_rows, "
            "or candidate_argmax_and_margin"
        )
    if not bool(dict(plan.metadata).get("output_pushdown", False)):
        raise ValueError("WorkPlan does not request output pushdown")
    method_name = {
        OutputContract.LAST_TOKEN_LOGITS: "last_logits_batch",
        OutputContract.SELECTED_TOKEN_ROWS: "selected_last_logits_batch",
        OutputContract.CANDIDATE_ARGMAX_AND_MARGIN: "candidate_logits_batch",
    }[plan.output_contract]
    if not callable(getattr(engine, method_name, None)):
        raise RuntimeError(f"engine does not implement required sliced method {method_name!r}")
    if lowered.plan_fingerprint != plan.fingerprint:
        raise ValueError("lowered schedule does not belong to this WorkPlan")
    if lowered.implementation_status != "eager-adapter":
        raise RuntimeError(
            f"{lowered.backend} is {lowered.implementation_status}; "
            "no executable sliced comparison is available"
        )
    compatible_engines = {
        "cuda-qstore": {"dense-qstore-cuda"},
        "paged-qstore": {"paged"},
    }
    engine_backend = str(getattr(engine, "backend", "unknown"))
    if engine_backend not in compatible_engines.get(
        lowered.backend,
        {engine_backend},
    ):
        raise RuntimeError(
            f"lowered backend {lowered.backend!r} is incompatible with engine {engine_backend!r}"
        )
    if warmup < 0 or trials <= 0:
        raise ValueError("warmup must be non-negative and trials must be positive")
    if rtol < 0 or atol < 0:
        raise ValueError("parity tolerances must be non-negative")
    if minimum_improvement_percent < 0:
        raise ValueError("minimum improvement threshold must be non-negative")

    rows = [np.asarray(ids, dtype=np.int64) for ids in ids_list]
    if len(rows) != plan.shape.actual_batch or any(
        row.ndim != 1 or int(row.size) != plan.shape.sequence_length for row in rows
    ):
        raise ValueError("runtime inputs do not match the planned batch and sequence shape")

    # Establish correctness before timing. When a graph is supplied, the manual leg
    # isolates compiler binding overhead from the already-established specialization.
    reference = _full_head_baseline(engine, plan, rows)
    manual_output = execute_direct_contract(engine, plan, rows)

    def run_manual() -> Any:
        return execute_direct_contract(engine, plan, rows)

    def run_sliced() -> Any:
        if graph_compilation is None:
            return run_manual()
        return execute_lowered_plan(
            engine,
            plan,
            lowered,
            rows,
            graph_compilation=graph_compilation,
        ).outputs

    sliced_output = run_sliced()
    paths = (
        ("baseline", "manual", "sliced")
        if graph_compilation is not None
        else ("baseline", "sliced")
    )

    for warmup_index in range(warmup):
        rotation = warmup_index % len(paths)
        order = paths[rotation:] + paths[:rotation]
        for path in order:
            if path == "baseline":
                _full_head_baseline(engine, plan, rows)
            elif path == "manual":
                run_manual()
            else:
                run_sliced()

    baseline_samples: list[float] = []
    manual_samples: list[float] = []
    sliced_samples: list[float] = []
    baseline_max_blocks: list[float] = []
    manual_max_blocks: list[float] = []
    sliced_max_blocks: list[float] = []
    for trial in range(trials):
        rotation = trial % len(paths)
        order = paths[rotation:] + paths[:rotation]
        for path in order:
            _sync_engine(engine)
            telemetry_reset = _reset_max_dequant_block(engine)
            started = time.perf_counter()
            if path == "baseline":
                _full_head_baseline(engine, plan, rows)
            elif path == "manual":
                manual_output = run_manual()
            else:
                sliced_output = run_sliced()
            _sync_engine(engine)
            duration = (time.perf_counter() - started) * 1000
            {
                "baseline": baseline_samples,
                "manual": manual_samples,
                "sliced": sliced_samples,
            }[path].append(duration)
            if telemetry_reset:
                observed = _max_dequant_block_mb(engine)
                if observed is not None:
                    {
                        "baseline": baseline_max_blocks,
                        "manual": manual_max_blocks,
                        "sliced": sliced_max_blocks,
                    }[path].append(observed)

    parity = _compare_outputs(reference, sliced_output, rtol=rtol, atol=atol)
    manual_graph_parity = (
        None
        if graph_compilation is None
        else _compare_outputs(manual_output, sliced_output, rtol=0.0, atol=0.0)
    )
    baseline_median, baseline_p95, baseline_min, baseline_max = _duration_summary(baseline_samples)
    sliced_median, sliced_p95, sliced_min, sliced_max = _duration_summary(sliced_samples)
    manual_median: float | None = None
    manual_p95: float | None = None
    if manual_samples:
        manual_median, manual_p95, _manual_min, _manual_max = _duration_summary(manual_samples)
    manual_to_graph_paired = (
        ()
        if not manual_samples
        else tuple(
            manual / graph if graph else math.inf
            for manual, graph in zip(manual_samples, sliced_samples, strict=True)
        )
    )
    manual_to_graph_paired_median = (
        None
        if not manual_to_graph_paired
        else float(np.median(np.asarray(manual_to_graph_paired, dtype=np.float64)))
    )
    manual_to_graph_paired_ci95 = (
        None
        if not manual_to_graph_paired
        else _paired_median_bootstrap_ci95(manual_to_graph_paired)
    )
    paired_speedups = tuple(
        baseline / sliced if sliced else math.inf
        for baseline, sliced in zip(baseline_samples, sliced_samples, strict=True)
    )
    paired_ci95 = _paired_median_bootstrap_ci95(paired_speedups)
    ratio = baseline_median / sliced_median if sliced_median else math.inf
    max_dequant_block = getattr(engine, "working_set_mb", None)
    baseline_max_block = max(baseline_max_blocks) if baseline_max_blocks else None
    sliced_max_block = max(sliced_max_blocks) if sliced_max_blocks else None
    manual_max_block = max(manual_max_blocks) if manual_max_blocks else None
    leg_max_blocks = tuple(
        value
        for value in (baseline_max_block, manual_max_block, sliced_max_block)
        if value is not None
    )
    overall_max_block = (
        max(leg_max_blocks)
        if leg_max_blocks
        else None
        if max_dequant_block is None
        else float(max_dequant_block)
    )
    required_ratio = 1.0 + minimum_improvement_percent / 100.0
    return OutputSlicingBenchmarkResult(
        output_contract=plan.output_contract.value,
        baseline_samples_ms=tuple(baseline_samples),
        sliced_samples_ms=tuple(sliced_samples),
        paired_speedups=paired_speedups,
        baseline_median_ms=baseline_median,
        baseline_p95_ms=baseline_p95,
        baseline_min_ms=baseline_min,
        baseline_max_ms=baseline_max,
        sliced_median_ms=sliced_median,
        sliced_p95_ms=sliced_p95,
        sliced_min_ms=sliced_min,
        sliced_max_ms=sliced_max,
        baseline_to_sliced_ratio=ratio,
        paired_speedup_median=float(np.median(np.asarray(paired_speedups, dtype=np.float64))),
        paired_speedup_ci95=paired_ci95,
        paired_wins=sum(value > 1.0 for value in paired_speedups),
        latency_reduction_percent=(1.0 - sliced_median / baseline_median) * 100.0
        if baseline_median
        else -math.inf,
        minimum_improvement_percent=minimum_improvement_percent,
        improvement_demonstrated=parity.allclose and paired_ci95[0] > required_ratio,
        trials=trials,
        warmup=warmup,
        parity=parity,
        reported_fabric=lowered.reported_fabric,
        max_dequant_block_mb=overall_max_block,
        baseline_max_dequant_block_mb=baseline_max_block,
        sliced_max_dequant_block_mb=sliced_max_block,
        graph_compilation_fingerprint=(
            None if graph_compilation is None else graph_compilation.fingerprint
        ),
        graph_rewrite_ids=(
            ()
            if graph_compilation is None or graph_compilation.rewrite_certificate is None
            else graph_compilation.rewrite_certificate.rewrite_ids
        ),
        manual_sliced_samples_ms=tuple(manual_samples),
        manual_sliced_median_ms=manual_median,
        manual_sliced_p95_ms=manual_p95,
        manual_to_graph_ratio=(
            None if manual_median is None or not sliced_median else manual_median / sliced_median
        ),
        manual_to_graph_paired_ratios=manual_to_graph_paired,
        manual_to_graph_paired_median=manual_to_graph_paired_median,
        manual_to_graph_paired_ci95=manual_to_graph_paired_ci95,
        graph_binding_overhead_percent=(
            None
            if manual_to_graph_paired_median is None or not manual_to_graph_paired_median
            else (1.0 / manual_to_graph_paired_median - 1.0) * 100.0
        ),
        manual_graph_parity=manual_graph_parity,
        manual_max_dequant_block_mb=manual_max_block,
        benchmark_basis=(
            "measured-full-head-baseline-vs-output-sliced"
            if graph_compilation is None
            else "measured-full-head-baseline-vs-graph-bound-output-sliced"
        ),
    )


def verify_output_pushdown_parity(
    engine: Any,
    plan: DenseWorkPlan,
    ids_list: Sequence[np.ndarray | Sequence[int]],
    *,
    rtol: float = 1e-5,
    atol: float = 2e-5,
) -> OutputParity:
    """Compare an optimized paged head contract with the established full-head path."""

    metadata = dict(plan.metadata)
    if not bool(metadata.get("output_pushdown", False)):
        raise ValueError("WorkPlan does not request output pushdown")
    metadata["output_pushdown"] = False
    reference_plan = replace(
        plan,
        numerical_contract=str(getattr(engine, "numerical_contract", plan.numerical_contract)),
        metadata=tuple(metadata.items()),
    )
    reference = execute_direct_contract(engine, reference_plan, ids_list)
    pushed = execute_direct_contract(engine, plan, ids_list)
    return _compare_outputs(reference, pushed, rtol=rtol, atol=atol)
