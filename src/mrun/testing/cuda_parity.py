"""B0-CUDA gate — the paged forward on a CUDA-dequant QStore must match the proven
scalar-CPU forward BEFORE any fleet leg runs paged-on-GPU.

Port of the beast-proven gate (grok-mechanism-full-proof
``discovery/experiments/megagpu/cuda_parity.py``, PASS 2026-07-03, 8.14x/forward) onto
mrun's vendored kernels. Reference/test split on the DEVICE axis:

  leg A: scalar-CPU  vs scalar-CUDA   (the QStore device knob alone)
  leg B: scalar-CPU  vs batched-CUDA  (device x batching; mixed lengths exercise
                                       right-pad + key-pad mask on the GPU)

plus patch-op smoke (zero + global_mean apply and change output), the capture contract
(selected activations return cpu/fp16), 2x determinism, per-forward CPU-vs-CUDA timing,
VRAM peak, and the O(largest-matrix) working set. Criteria = the established B0 gate:
per-row argmax-exact at every real position, max|dlogit| < 0.5 (actual ~1e-3-class),
last_only agrees, deterministic 2x. CPU<->CUDA fp32 reduction order differs, so
bit-exactness is impossible by construction; 0.5 is the established bound.

Run (beast):
  OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
    python -m mrun.testing.cuda_parity --model qwen3-4b
(the gate sets GATHER_DEVICE_PAGED itself — env beats any flag file, so it is
deterministic regardless of what is already flagged on the box).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..engine.kernels import paged_forward as pf
from ..engine.paged import PagedEngine
from ..paths import stores_root

from ..paths import data_root

FLAG_DIR = Path(os.environ.get("MRUN_DEVICE_FLAG_DIR", str(data_root() / "device-flags")))
LENS = [12, 8, 16, 5, 11]      # B0's mixed lengths (padding exercised), rng seed 0


def _open_stores(model_name: str):
    """CPU + CUDA stores through the REAL env resolution path."""
    os.environ["GATHER_DEVICE_PAGED"] = "cpu"
    cpu = PagedEngine(model_name).store
    arch = cpu.man.get("arch", "qwen2")
    os.environ["GATHER_DEVICE_PAGED"] = f"cuda {arch}"
    gpu = PagedEngine(model_name).store
    return cpu, gpu, arch


def _cmp_rows(refs: list[torch.Tensor], tests: list[torch.Tensor]) -> dict:
    """Per-row all-positions comparison: argmax-exact count, max|dlogit|, max|dlogprob|."""
    top1, max_abs, lp_diff = 0, 0.0, 0.0
    for ref, t in zip(refs, tests):
        max_abs = max(max_abs, float((t - ref).abs().max()))
        top1 += int((t.argmax(-1) == ref.argmax(-1)).all().item())
        lp = (torch.log_softmax(t, -1) - torch.log_softmax(ref, -1)).abs().max().item()
        lp_diff = max(lp_diff, lp)
    return {"top1": top1, "max_abs": max_abs, "max_lp": lp_diff}


def gate(model_name: str) -> dict:
    res: dict = {"gate": "B0-CUDA", "model": model_name, "lengths": LENS}
    try:
        store_cpu, store_gpu, arch = _open_stores(model_name)
    except FileNotFoundError as e:
        root = stores_root()
        have = sorted(p.name for p in Path(root).glob("*") if p.is_dir())
        res.update({"B0_CUDA_verdict": "FAIL", "error": f"store not found: {e}",
                    "stores_present": have})
        return res
    finally:
        os.environ.pop("GATHER_DEVICE_PAGED", None)
    res["arch"] = arch
    res["store_device"] = store_gpu.device
    if store_gpu.device != "cuda":
        res.update({"B0_CUDA_verdict": "FAIL",
                    "error": "store did not resolve to cuda (torch.cuda unavailable, or arch gate)"})
        return res
    res["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)   # must be False

    rng = np.random.default_rng(0)
    Vtok = store_cpu.cfg["vocab_size"]
    nL = store_cpu.cfg["num_hidden_layers"]
    ids_list = [rng.integers(5, Vtok - 5, size=L, dtype=np.int64) for L in LENS]
    B = len(ids_list)

    # ---- scalar CPU reference (the proven M1 engine), per-forward timed ----
    refs, t_cpu = [], []
    for ids in ids_list:
        t0 = time.perf_counter()
        refs.append(pf.paged_logits(store_cpu, ids))
        t_cpu.append(time.perf_counter() - t0)

    # ---- leg A: scalar-CUDA (device knob alone), warmup then timed ----
    pf.paged_logits(store_gpu, ids_list[0])            # cuda context + allocator warmup
    torch.cuda.reset_peak_memory_stats()
    tests, t_gpu = [], []
    for ids in ids_list:
        t0 = time.perf_counter()
        out = pf.paged_logits(store_gpu, ids)          # returns cpu (the .cpu() is the sync)
        t_gpu.append(time.perf_counter() - t0)
        tests.append(out)
    A = _cmp_rows(refs, tests)
    det_scalar = torch.equal(tests[0], pf.paged_logits(store_gpu, ids_list[0]))

    # ---- leg B: batched-CUDA all-positions vs the same CPU refs ----
    t0 = time.perf_counter()
    blog, lengths = pf.batched_paged_logits(store_gpu, ids_list, last_only=False)  # [B,Tmax,V]
    t_batched = time.perf_counter() - t0
    Bcmp = _cmp_rows(refs, [blog[b, : len(ids)] for b, ids in enumerate(ids_list)])
    blog_last, _ = pf.batched_paged_logits(store_gpu, ids_list, last_only=True)    # [B,V]
    last_path_ok = all(
        blog_last[b].argmax().item() == blog[b, len(ids) - 1].argmax().item()
        for b, ids in enumerate(ids_list))
    blog2, _ = pf.batched_paged_logits(store_gpu, ids_list, last_only=False)
    det_batched = torch.equal(blog, blog2)

    # ---- patch-op smoke: apply without device errors, output must differ from unpatched ----
    L0, L1 = min(1, nL - 1), min(3, nL - 1)
    ops = {L0: [("zero", list(range(64)), None)],
           L1: [("global_mean", list(range(64, 96)), torch.full((32,), 0.25))]}   # cpu vals tensor
    patched = pf.paged_logits(store_gpu, ids_list[0], patch_ops_by_layer=ops)
    patch_diff = float((patched - tests[0]).abs().max())
    pb, _ = pf.batched_paged_logits(store_gpu, ids_list, last_only=False, patch_ops_by_layer=ops)
    patch_diff_b = float((pb[0, : LENS[0]] - blog[0, : LENS[0]]).abs().max())
    patch_ok = patch_diff > 0.0 and patch_diff_b > 0.0

    # ---- capture contract: selected activations come back cpu/fp16 ----
    cap_s: dict[int, torch.Tensor] = {}
    pf.paged_logits(store_gpu, ids_list[1],
                    capture_selected_maps={min(2, nL - 1): {"locals": [1, 2, 3]}},
                    captured_selected=cap_s)
    cap_b: dict[int, torch.Tensor] = {}
    pf.batched_paged_logits(store_gpu, ids_list, last_only=True,
                            capture_selected_maps={min(2, nL - 1): {"locals": [1, 2, 3]}},
                            captured_selected=cap_b)
    cap_ok = all(
        (min(2, nL - 1) in c
         and c[min(2, nL - 1)].device.type == "cpu"
         and c[min(2, nL - 1)].dtype == torch.float16)
        for c in (cap_s, cap_b))

    verdict = (A["top1"] == B and A["max_abs"] < 0.5
               and Bcmp["top1"] == B and Bcmp["max_abs"] < 0.5
               and last_path_ok and det_scalar and det_batched and patch_ok and cap_ok)
    med = lambda v: float(np.median(v))  # noqa: E731
    res.update({
        "scalar_cuda_vs_cpu": {"row_top1_all_pos": f"{A['top1']}/{B}",
                               "max_abs_logit_diff": round(A["max_abs"], 5),
                               "max_abs_logprob_diff": round(A["max_lp"], 6)},
        "batched_cuda_vs_scalar_cpu": {"row_top1_all_pos": f"{Bcmp['top1']}/{B}",
                                       "max_abs_logit_diff": round(Bcmp["max_abs"], 5),
                                       "max_abs_logprob_diff": round(Bcmp["max_lp"], 6)},
        "last_only_path_agrees": last_path_ok,
        "deterministic_2x": {"scalar": det_scalar, "batched": det_batched},
        "patch_smoke": {"scalar_max_abs_delta": round(patch_diff, 4),
                        "batched_max_abs_delta": round(patch_diff_b, 4), "ok": patch_ok},
        "capture_contract_cpu_fp16": cap_ok,
        "timing_s": {"scalar_cpu_per_forward": [round(t, 3) for t in t_cpu],
                     "scalar_cuda_per_forward": [round(t, 3) for t in t_gpu],
                     "scalar_cpu_median": round(med(t_cpu), 3),
                     "scalar_cuda_median": round(med(t_gpu), 3),
                     "end_to_end_scalar_speedup": round(med(t_cpu) / max(med(t_gpu), 1e-9), 2),
                     "batched_cuda_all_pos_B5": round(t_batched, 3)},
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "working_set_mb": round(store_gpu.max_block_bytes / 1e6, 1),
        "B0_CUDA_verdict": "PASS" if verdict else "FAIL",
    })
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--out", type=Path, default=None,
                    help="gate JSON path (default: <flag-dir>/cuda_parity_<model>.json "
                         "when that dir exists, else print-only)")
    a = ap.parse_args()
    res = gate(a.model)
    print(json.dumps(res, indent=2))
    out = a.out or (FLAG_DIR / f"cuda_parity_{a.model}.json" if FLAG_DIR.is_dir() else None)
    if out is not None:
        out.write_text(json.dumps(res, indent=2))
        print(f"gate json -> {out}", flush=True)
    return 0 if res.get("B0_CUDA_verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
