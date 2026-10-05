"""launch_overhead.py — measure the host/launch-overhead fraction of B=1 decode.

Decides whether CUDA-graph capture of the steady-state decode region is worth building
(the ANE-review takeaway: a fixed-function graph engine pays ONE dispatch per graph;
a GPU decode step pays one launch per kernel — hundreds per token on the paged path).
Pre-registered decision rule, same shape as PROFILE.md §5:

    overhead_fraction >= 0.15  -> graph capture is a real B=1 lever; build/promote it
    overhead_fraction <  0.15  -> launches hide under kernels; capture is NOT the next
                                  lever, spend the effort on fp8 / K4 instead

Method: run N warm single-token decode steps under ``torch.profiler`` and compare
wall time against the sum of on-device kernel time. The gap is host-side work:
Python dispatch, kernel launches, allocator, synchronization. This OVERSTATES pure
launch cost (Python is in the gap too), so a small measured fraction is decisive
("not launch-bound") while a large one justifies the follow-up capture experiment —
the cheap-probe-first discipline, not a certified attribution.

Run through mrun (fleet rules apply):

    mx run -- python -m mrun.tools.launch_overhead --model qwen2.5-0.5b --steps 32

Prints one JSON line: wall_s, device_kernel_s, overhead_fraction, steps, tok_s.
"""
from __future__ import annotations

import argparse
import json
import time


def measure(model: str, *, steps: int = 32, backend: str = "paged",
            prompt: str = "The quick brown fox jumps over the lazy dog. ") -> dict:
    """Profile ``steps`` warm B=1 decode tokens; return the overhead breakdown."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from ..engine import open_engine

    eng = open_engine(model, backend=backend)
    dev = str(getattr(eng, "device", "cpu"))
    if not dev.startswith("cuda"):
        raise SystemExit(
            f"launch_overhead needs a cuda-granted engine (got device={dev!r}); "
            "on beast set GATHER_DEVICE_PAGED=\"cuda qwen2 llama qwen3\""
        )

    # Warm pass: page/dequant caches, ring successor map, KV path — we are measuring
    # steady-state decode, not cold start.
    eng.generate(prompt, max_new_tokens=4)

    # Wall time from an UN-profiled pass: kineto instrumentation on hundreds of tiny
    # ops/token would inflate the overhead numerator against the pre-registered 0.15
    # threshold. The profiled pass below supplies only the device-kernel denominator.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    eng.generate(prompt, max_new_tokens=steps)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        eng.generate(prompt, max_new_tokens=steps)
        torch.cuda.synchronize()

    # Sum self device time over CUDA-side events ONLY — torch attributes each kernel's
    # time BOTH to the launching CPU op and to the kernel's own event, so summing over
    # every averaged event double-counts (this is the filter torch's own
    # "Self CUDA time total" applies).
    from torch.autograd import DeviceType

    device_us = sum(
        evt.self_device_time_total
        for evt in prof.key_averages()
        if evt.device_type == DeviceType.CUDA
    )
    device_s = device_us / 1e6
    overhead = max(0.0, wall_s - device_s)
    return {
        "model": model,
        "backend": backend,
        "device": dev,
        "steps": steps,
        "wall_s": round(wall_s, 4),
        "device_kernel_s": round(device_s, 4),
        "overhead_s": round(overhead, 4),
        "overhead_fraction": round(overhead / wall_s, 4) if wall_s > 0 else None,
        "tok_s": round(steps / wall_s, 2) if wall_s > 0 else None,
        "decision_rule": "overhead_fraction >= 0.15 -> build graph capture",
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--backend", default="paged")
    args = ap.parse_args(argv)
    print(json.dumps(measure(args.model, steps=args.steps, backend=args.backend)))


if __name__ == "__main__":
    main()
