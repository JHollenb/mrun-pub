"""Capability-signature management — planning, loading, fleet submit, and the backend gate.

The capability signature (science: ``grok-mechanism-full-proof/discovery/src/
capability_signature.py``) measures a capability as a required-primitive checklist — per
primitive (present, dependence, specificity, depth), each null-controlled, plus a coverage
residual. Its OPERATIONAL half — which dtype/device a signature run gets, how its reservation
is sized, how jobs go to the fleet, and how a candidate engine backend (paged/qstore int8,
fused kernels, bf16, …) is judged against the dense reference — is mrun policy, and lives HERE.
The science batteries stay in the discovery repo (they back committed, measured findings);
``discovery/experiments/understanding-gate/mrun_signature.py`` is now a re-export shim over
this module, so existing callers (capability_signature.py, submit_signature.py) are unchanged.

House rule this module enforces (capability-metric-scramble-not-bpb): an engine lever is gated
on the CAPABILITY VECTOR — present/causal/depth legs plus adversarial nulls — never on CE or
argmax parity alone. ``compare_signatures`` is that gate: run the dense-fp32 reference arm and
the candidate-backend arm through the same battery, then compare payloads leg by leg.

Numbers below marked MEASURED come from the 2026-07-15 fleet runs (see the module docstrings
they were ported from).
"""

from __future__ import annotations

from dataclasses import replace

from ..policy import HostCaps, RunPlan, plan_run

# --------------------------------------------------------------------------- planning
# The order the signature prefers, FASTEST-that-admits first:
#   1. bf16/cuda   — the forward-heavy signature runs the dense HF module on the GPU instead of
#      leaving it idle; ~10x on the models that fit the card. MEASURED 2026-07-17: plan_run sizes
#      est_vram correctly (0.5B 2.8GB / 1.5B 5.6GB / 3B 9.5GB) and CLEANLY DOWNGRADES cuda->cpu for
#      anything that won't fit the 16GB card (4B/7B/14B), so the old "cuda RunPlan reports est_vram=0
#      -> bogus huge RAM reservation" fear does not occur — plan_run's VRAM guard (policy.py:293-303)
#      handles it. reservation_for reads p.est_vram_mb (real when cuda fits, 0 when it fell to cpu).
#   2. bf16/cpu    — the fallback when the GPU won't fit the model OR is busy with another job
#      (a fleet cuda job holds VRAM -> detect() reports low free VRAM -> plan_run downgrades this
#      rung to cpu; MEASURED 2026-07-17: 0.5B fell to cpu while operator-physiology held the GPU).
# UNIFORM bf16 on purpose: the old fp32/cpu rung made SMALL models fp32 and BIG models bf16 — the
# exact dtype split that (with a stale v1 code path) manufactured the false "coding emergence." One
# dtype across the whole curve removes that confound. fp32 is still available EXPLICITLY (--dtype
# fp32) for a canonical rerun. Dense HF on CUDA keeps full forward-hook support (the causal legs need
# hooks; only the PAGED engine lacks a hook API). Argmax forced-choice is bf16-robust and the
# mechanistic reductions upcast to fp32 (see measure_signature docstring).
_DTYPE_DEVICE_LADDER = (("bf16", "cuda"), ("bf16", "cpu"))

# Fit-budget fraction per dtype. Canonical fp32 is admitted only when it fits COMFORTABLY (0.70);
# the bf16 fallback is pushed to the real admission ceiling (~0.82 = total - 15% margin - baseline)
# so a model like 14B (bf16/cpu ceiling ~47GB, admits at ~52GB<54.5GB) is not falsely rejected.
_BUDGET_FRAC = {"fp32": 0.70, "bf16": 0.82}

# The signature is a COMPUTE-BOUND dense CPU run (no paging/GPU — interventions need forward
# hooks, which the paged engine has no API for). mrun's default `threads=4` throttles a >=7B
# dense matmul to ~2 cores of a 32-core box (MEASURED: 14B ran 5.5h at OMP=4, GPU idle). Pin to
# most of the box; capped because bf16-CPU matmul is memory-bandwidth-bound and scaling plateaus
# past ~mid-teens.
SIGNATURE_MAX_THREADS = 16

# The signature runs many forwards with `output_hidden_states=True` (all L+1 layers captured)
# plus per-layer hooks — recorder-class activation the plain `task="forward"` estimate
# under-counts. MEASURED 2026-07-15: 7B bf16 peaked 24584MB vs a `ram_limit` of 22165MB
# (+2419MB) -> killed_ram. The overflow is ACTIVATION/capture, ~independent of the
# (weight-dominated) base est, so add FIXED headroom, not a multiplier — a multiplier
# over-penalizes weight-heavy models (it pushed 14B's ceiling to 66GB > beast RAM, when 14B's
# real base is only ~42GB and fits). 5GB comfortably covers the measured 2.4GB 7B overflow and
# the larger-but-still-single-digit-GB capture at 14B.
SIGNATURE_HEADROOM_MB = 5000.0


def _target_host(host: HostCaps | None = None) -> HostCaps:
    """Use explicit target capabilities or the actual local host."""
    return host if host is not None else HostCaps.detect()


def plan_signature(model: str, host: HostCaps | None = None) -> RunPlan:
    """Largest dense-HF dtype that admits on the target host. fp32 (canonical) when it fits,
    else bf16. Raises if nothing dense fits (the model needs the paged-signature path)."""
    host = _target_host(host)
    tried = []
    for dtype, device in _DTYPE_DEVICE_LADDER:
        p = plan_run(model, task="forward", host=host, backend="hf", dtype=dtype, device=device)
        # Lift the ceiling above the measured all-layer-capture peak (SIGNATURE_HEADROOM_MB),
        # and un-throttle threads (mrun defaults to 4; this is a compute-bound dense CPU matmul).
        p = replace(
            p,
            ram_limit_mb=round(p.ram_limit_mb + SIGNATURE_HEADROOM_MB, 1),
            threads=min(max(host.cpus - 2, 1), SIGNATURE_MAX_THREADS),
        )
        cap = host.ram_mb * _BUDGET_FRAC[dtype]
        tried.append(f"{dtype}/{device} ceil={p.ram_limit_mb:.0f} cap={cap:.0f}")
        if p.ram_limit_mb * 1.1 <= cap:  # admission commits at the kill ceiling x1.1
            return p
    raise MemoryError(
        f"{model}: no dense-HF plan fits {host.name} ({'; '.join(tried)}); "
        "needs the paged-signature path"
    )


def reservation_for(model: str, host: HostCaps | None = None) -> tuple[dict, RunPlan]:
    """(reservation dict for ``submit``, the plan it came from). Sized by policy, not by hand."""
    p = plan_signature(model, host)
    resv = {
        "ram_mb": round(p.ram_limit_mb, 1),
        "vram_mb": round(p.est_vram_mb, 1),
        "cpu_threads": p.threads,
        "disk_gb": 2.0,
        "source": "plan_signature",
    }
    return resv, p


def paged_reservation(model: str, host: HostCaps | None = None) -> tuple[dict, None]:
    """Reservation for a paged-signature job. int8 paged is O(largest-matrix): the weight store
    is mmap'd (page cache, not RSS) and dequantized one block at a time on the GPU, so both the
    RAM working set and the VRAM block pool are small and model-size-INDEPENDENT (unlike the
    dense ceiling). Generous fixed sizes; these admit trivially and can even run concurrently."""
    resv = {"ram_mb": 8000.0, "vram_mb": 6000.0, "cpu_threads": 8, "disk_gb": 2.0,
            "source": "paged_signature"}
    return resv, None


# --------------------------------------------------------------------------- loading
# Arches whose paged forward is device-ported (argmax-exact on the beast CUDA path). The store's
# arch must appear in the GATHER_DEVICE_PAGED list for the int8 CUDA gather to engage; anything
# else silently (and correctly) falls back to the slow CPU dequant path.
_PAGED_CUDA_ARCHS = "qwen2 llama qwen3"


def load_paged_engine(model: str):
    """Open the int8 QStore PagedEngine on the arch-gated CUDA path (~8x/forward, argmax-exact,
    O(largest-matrix) RAM, weights streamed to the GPU). Returns ``(engine, device_str)``.

    The store reads GATHER_DEVICE_PAGED at open time; we default it to ``cuda <arches>`` so the
    gather engages for qwen2/llama/qwen3 (an explicit env still wins, e.g. to force cpu)."""
    import os

    from .. import open_engine

    os.environ.setdefault("GATHER_DEVICE_PAGED", f"cuda {_PAGED_CUDA_ARCHS}")
    eng = open_engine(model, backend="paged")
    return eng, eng.device


def load_for_signature(model: str, plan: RunPlan | None = None, host: HostCaps | None = None):
    """Resolve + load a DENSE HF module through mrun (canonical resolution, rope guard). Returns
    ``(hf_model, tokenizer, plan)``. dtype/device come from the signature plan, not the caller."""
    from .. import open_engine

    p = plan or plan_signature(model, host)
    eng = open_engine(p.model, backend=p.backend, **p.engine_kwargs())  # p.backend == "hf"
    hf_model = eng.model
    hf_model.eval()
    return hf_model, eng.tokenizer, p


# --------------------------------------------------------------------------- the backend gate
# Thresholds are POLICY defaults, override per call. Signs near zero are noise; a gate fires
# only on CLEAR reference signal (|ref| above the eps) followed by a candidate flip/drop.
DEP_EPS = 0.1        # reference dependence < -DEP_EPS  => "the task clearly NEEDS this"
SPEC_EPS = 0.1       # reference specificity > SPEC_EPS => "clearly localized here"
BEHAVIOR_TOL = 0.05  # capability-level behavior_accuracy may drop at most this much
COMPONENT_TOL = 0.10  # per-behavior-component drop tolerance
NULL_CEILING = 0.35  # a scrambled/no-context null above this in the CANDIDATE = resurrected
_DEPTH_KEYS = ("max_distance_surviving", "max_depth_tracked", "max_causal_chain_length",
               "depth_capacity")


def _null_accs(leg: dict) -> list[tuple[str, float]]:
    """(name, accuracy) for every adversarial-null summary present in a leg dict."""
    out = []
    for key in ("scrambled_null", "no_context_null"):
        val = leg.get(key)
        if isinstance(val, dict) and val.get("accuracy") is not None:
            out.append((key, float(val["accuracy"])))
    for key in ("scrambled_null_by_depth", "no_context_null_by_depth"):
        for depth, val in (leg.get(key) or {}).items():
            if isinstance(val, dict) and val.get("accuracy") is not None:
                out.append((f"{key}[{depth}]", float(val["accuracy"])))
    return out


def compare_signatures(reference: dict, candidate: dict, *,
                       dep_eps: float = DEP_EPS, spec_eps: float = SPEC_EPS,
                       behavior_tol: float = BEHAVIOR_TOL,
                       component_tol: float = COMPONENT_TOL,
                       null_ceiling: float = NULL_CEILING) -> dict:
    """Gate a candidate engine backend's signature payload against the dense reference's.

    ``reference``/``candidate`` are ``measure_signature`` payloads for the SAME model +
    capability + battery (reference = dense hf fp32 oracle; candidate = paged/qstore/fused/bf16
    arm). Returns ``{"pass": bool, "failures": [...], "warnings": [...], "per_primitive": ...}``.

    FAIL conditions (each cites its leg):
      * a leg measurable in the reference becomes unmeasurable in the candidate
      * present flips True -> False
      * reference dependence < -dep_eps but candidate dependence >= 0 (need lost)
      * reference specificity > spec_eps but candidate specificity <= 0 (localization lost)
      * any depth capacity shrinks
      * capability behavior_accuracy drops more than behavior_tol (components: component_tol)
      * an adversarial null reads above null_ceiling in the candidate (nulls are the canary:
        a resurrected null means the harness, not the model — halt, no verdicts)
    Warnings (reported, never failing): capability gained where the reference lacked it,
    unmeasurable-in-both legs, nulls elevated in the REFERENCE itself.
    """
    ref_cap, cand_cap = reference.get("capability"), candidate.get("capability")
    if ref_cap != cand_cap:
        raise ValueError(f"capability mismatch: {ref_cap!r} vs {cand_cap!r}")
    failures: list[str] = []
    warnings: list[str] = []
    per_primitive: dict[str, dict] = {}

    ref_prims = reference.get("primitives") or {}
    cand_prims = candidate.get("primitives") or {}
    for pname, ref in ref_prims.items():
        cand = cand_prims.get(pname)
        checks: list[str] = []
        if cand is None:
            failures.append(f"{pname}: missing from candidate payload")
            per_primitive[pname] = {"checks": ["missing"], "ok": False}
            continue
        for leg in ("present", "causal_use", "depth"):
            r, c = ref.get(leg) or {}, cand.get(leg) or {}
            if not r.get("measurable"):
                if c.get("measurable"):
                    warnings.append(f"{pname}.{leg}: measurable only in candidate (gained)")
                else:
                    checks.append(f"{leg}: skip (unmeasurable in both)")
                continue
            if not c.get("measurable"):
                failures.append(
                    f"{pname}.{leg}: measurable in reference, lost in candidate "
                    f"({c.get('reason', 'no reason')})"
                )
                continue
            if leg == "present" and "present" in r:
                if bool(r["present"]) and not bool(c.get("present")):
                    failures.append(f"{pname}.present: True -> False")
                elif not bool(r["present"]) and bool(c.get("present")):
                    warnings.append(f"{pname}.present: gained (False -> True)")
                else:
                    checks.append("present: ok")
            if leg == "causal_use":
                rd_, cd = r.get("dependence"), c.get("dependence")
                rs, cs = r.get("specificity"), c.get("specificity")
                if rd_ is not None and cd is not None and rd_ < -dep_eps and cd >= 0:
                    failures.append(f"{pname}.dependence: {rd_} -> {cd} (need lost)")
                else:
                    checks.append("dependence: ok")
                if rs is not None and cs is not None and rs > spec_eps and cs <= 0:
                    failures.append(f"{pname}.specificity: {rs} -> {cs} (localization lost)")
                else:
                    checks.append("specificity: ok")
            if leg == "depth":
                for dk in _DEPTH_KEYS:
                    if r.get(dk) is not None and c.get(dk) is not None:
                        if float(c[dk]) < float(r[dk]):
                            failures.append(f"{pname}.{dk}: {r[dk]} -> {c[dk]} (depth shrank)")
                        else:
                            checks.append(f"{dk}: ok")
            # nulls are the canary in EVERY leg that carries them
            for name, acc in _null_accs(c):
                if acc > null_ceiling:
                    failures.append(
                        f"{pname}.{leg}.{name}: candidate null at {acc} > {null_ceiling} "
                        "(null resurrected — harness artifact, halt)"
                    )
            for name, acc in _null_accs(r):
                if acc > null_ceiling:
                    warnings.append(f"{pname}.{leg}.{name}: REFERENCE null already at {acc}")
        per_primitive[pname] = {
            "checks": checks,
            "ok": not any(f.startswith(f"{pname}") for f in failures),
        }

    ref_cov, cand_cov = reference.get("coverage") or {}, candidate.get("coverage") or {}
    ra, ca = ref_cov.get("behavior_accuracy"), cand_cov.get("behavior_accuracy")
    if ra is not None and ca is not None and (ra - ca) > behavior_tol:
        failures.append(f"behavior_accuracy: {ra} -> {ca} (drop > {behavior_tol})")
    rcomp = ref_cov.get("behavior_components") or {}
    ccomp = cand_cov.get("behavior_components") or {}
    for key, rv in rcomp.items():
        cv = ccomp.get(key)
        if rv is not None and cv is not None and (float(rv) - float(cv)) > component_tol:
            failures.append(f"behavior.{key}: {rv} -> {cv} (drop > {component_tol})")

    return {
        "pass": not failures,
        "failures": failures,
        "warnings": warnings,
        "per_primitive": per_primitive,
        "reference": {"model": reference.get("model_name"),
                      "config": reference.get("config")},
        "candidate": {"model": candidate.get("model_name"),
                      "config": candidate.get("config")},
        "thresholds": {"dep_eps": dep_eps, "spec_eps": spec_eps,
                       "behavior_tol": behavior_tol, "component_tol": component_tol,
                       "null_ceiling": null_ceiling},
    }
