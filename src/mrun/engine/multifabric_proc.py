"""Process-based multi-fabric scoring — let CPU paged join the overlap without the GIL.

The thread-based `MultiFabricEngine` cannot overlap the CPU paged fabric: its 24-layer Python
loop holds the GIL, so its thread blocks the accelerator threads (measured: a 3-way thread split
falls *below* MLX alone). Separate **processes** have separate GILs, so paged runs fully parallel
to MLX and ANE.

The catch is IPC: shipping full logits ``[T, vocab]`` (≈19 MB/row at vocab 151936) across a process
boundary would dwarf the compute. So this dispatcher is for **forced-choice scoring**, where each
worker returns only the scored rows (winner + margin + per-candidate logprob — a few floats per
probe). Inputs are prompt strings + candidate strings (tiny). IPC stays negligible, so the
aggregate scoring throughput approaches the SUM of the fabrics' standalone throughput — now
*including* the CPU.

Design:
  - one ``ProcessPoolExecutor(max_workers=1)`` per backend, each with an initializer that opens
    that backend's engine ONCE in the worker (engine open is expensive — paged loads the store,
    mlx loads resident weights, ane compiles — so it must be amortized, not paid per call);
  - **spawn** start method (never fork): the parent has already imported torch / mlx / coremltools
    and started threads; forking that is a deadlock. Spawn re-imports cleanly in the child;
  - probes are split by throughput weight, each shard scored on its fabric's worker, rows
    reassembled in the original order;
  - a backend whose worker fails to start (or errors on a shard) is dropped and its shard is
    re-scored on the reference (paged) fabric, so correctness never depends on an accelerator.

Like the other accelerator paths this is **opt-in fp16 speed** for mlx/ane shards — gate any
content/CE leg on real-text parity vs paged (hw_parity), never random tokens.
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
from typing import Any

from .base import summarize_forced_choice_rows
from .multifabric import _DEFAULT_TPUT, shard_sizes

# Per-worker engine, opened once by the pool initializer and reused across shards.
_WORKER_ENGINE: Any = None


def _init_worker(model: str, backend: str, open_kwargs: dict[str, Any]) -> None:
    global _WORKER_ENGINE
    from . import open_engine

    _WORKER_ENGINE = open_engine(model, backend=backend, **open_kwargs)


def _score_shard(probes: list[dict[str, Any]], argmax_only: bool) -> list[dict[str, Any]]:
    # returns only the scored rows (small) — never logits
    return _WORKER_ENGINE.score_forced_choice_many(probes, argmax_only=argmax_only)["rows"]


def _resolved_backend() -> str:
    return getattr(_WORKER_ENGINE, "backend", "?")


class ProcessMultiFabric:
    """Forced-choice scoring split across fabrics in separate processes (separate GILs)."""

    def __init__(
        self,
        model: str,
        *,
        backends: tuple[str, ...] = ("paged", "mlx", "ane"),
        weights: dict[str, float] | None = None,
        device: str = "cpu",
        warmup_probe: dict[str, Any] | None = None,
        **open_kwargs: Any,
    ) -> None:
        self.model = model
        self._ctx = mp.get_context("spawn")
        open_kwargs = {"device": device, **open_kwargs}
        warmup = warmup_probe or {
            "prompt": "The capital of France is", "correct": " Paris",
            "distractors": [" London"], "probe_id": "warmup",
        }
        self.pools: dict[str, cf.ProcessPoolExecutor] = {}
        self.backend_names: list[str] = []
        self.open_errors: dict[str, str] = {}
        for b in backends:
            ex = cf.ProcessPoolExecutor(
                max_workers=1, mp_context=self._ctx,
                initializer=_init_worker, initargs=(model, b, open_kwargs),
            )
            try:
                # forces the worker to start + open its engine (amortized); drop the fabric on failure
                resolved = ex.submit(_score_and_report, warmup).result(timeout=600)
            except Exception as exc:
                self.open_errors[b] = f"{type(exc).__name__}: {exc}"
                ex.shutdown(wait=False, cancel_futures=True)
                continue
            actual = resolved[1]
            if actual in self.backend_names:        # `ane` may degrade to paged — don't double-count
                self.open_errors[b] = f"resolved to already-open backend {actual!r}"
                ex.shutdown(wait=False, cancel_futures=True)
                continue
            self.pools[actual] = ex
            self.backend_names.append(actual)
        if not self.backend_names:
            raise RuntimeError(f"no fabric process started for {model}: {self.open_errors}")
        weight_map = {**_DEFAULT_TPUT, **(weights or {})}
        self.weights = [float(weight_map.get(n, 1.0)) for n in self.backend_names]

    def set_weights(self, by_backend: dict[str, float]) -> None:
        self.weights = [float(by_backend.get(n, 1.0)) for n in self.backend_names]

    def score_forced_choice_many(
        self, probes: list[dict[str, Any]], *, argmax_only: bool = False
    ) -> dict[str, Any]:
        n = len(probes)
        if n == 0:
            return {"summary": summarize_forced_choice_rows([]), "rows": []}
        sizes = shard_sizes(n, self.weights)
        rows: list[dict[str, Any] | None] = [None] * n
        futures = []
        pos = 0
        for backend, sz in zip(self.backend_names, sizes):
            if sz <= 0:
                continue
            shard = probes[pos : pos + sz]
            futures.append((backend, pos, self.pools[backend].submit(_score_shard, shard, argmax_only)))
            pos += sz
        for backend, start, fut in futures:
            try:
                shard_rows = fut.result()
            except Exception:                       # fabric died mid-shard -> score it on paged in-proc
                shard_rows = self._fallback_rows(probes, start, sizes, backend)
            for j, row in enumerate(shard_rows):
                rows[start + j] = row
        return {"summary": summarize_forced_choice_rows(rows), "rows": rows}  # type: ignore[arg-type]

    def _fallback_rows(self, probes, start, sizes, backend) -> list[dict[str, Any]]:
        from . import open_engine

        sz = sizes[self.backend_names.index(backend)]
        with open_engine(self.model, backend="paged") as ref:
            return ref.score_forced_choice_many(probes[start : start + sz])["rows"]

    def close(self) -> None:
        for ex in self.pools.values():
            ex.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> ProcessMultiFabric:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _score_and_report(probe: dict[str, Any]) -> tuple[int, str]:
    """Warmup task: score one probe (opens the engine via the initializer) and report the resolved
    backend name so the parent can dedupe (e.g. ane degrading to paged)."""
    _WORKER_ENGINE.score_forced_choice_many([probe])
    return (1, _resolved_backend())


def open_process_multifabric(model: str, **kwargs: Any) -> ProcessMultiFabric:
    return ProcessMultiFabric(model, **kwargs)
