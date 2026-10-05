"""``mrun fingerprint`` — CLI for the embedding-provenance instrument.

    mrun fingerprint spectrum Qwen/Qwen2.5-7B [Qwen/Qwen2.5-Math-7B ...] [--json]
    mrun fingerprint lineup deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --candidate Qwen/Qwen2.5-Math-7B --candidate Qwen/Qwen2.5-7B \
        --candidate Qwen/Qwen2.5-Coder-7B [--k 64]
    mrun fingerprint rows Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct [--sample 20000]

Note there is no ``--threshold`` and no single-candidate mode: ``lineup`` requires >= 2
candidates because the alignment number is only interpretable relative to the others (see
``mrun.fingerprint.api``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _emit(rows: list[dict], out: Path | None) -> None:
    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        prog="mrun fingerprint",
        description="embedding-provenance fingerprints (SUBSTRATE lineage only; teacher "
                    "attribution is out of scope)")
    sub = p.add_subparsers(dest="leg", required=True)

    sp = sub.add_parser("spectrum", help="per-model centered-Gram spectrum + cached eigenbasis")
    sp.add_argument("models", nargs="+")
    sp.add_argument("--key", default=None, help="tensor key (default: family input embedding)")
    sp.add_argument("--k", type=int, default=256, help="eigenvectors to cache")
    sp.add_argument("--no-trim", action="store_true",
                    help="do NOT trim rows to len(tokenizer) (records trim=cfg)")

    ln = sub.add_parser("lineup", help="rank candidate substrate parents for a student")
    ln.add_argument("student")
    ln.add_argument("--candidate", action="append", required=True, dest="candidates",
                    help="repeat >=2 times; a lineup needs candidates, not a cutoff")
    ln.add_argument("--k", type=int, default=64, help="top-k eigenbasis for principal angles")
    ln.add_argument("--margin", type=float, default=None,
                    help="ambiguity margin on the rank1-rank2 GAP (default 0.02)")
    ln.add_argument("--strict-dtype", action="store_true")

    rw = sub.add_parser("rows", help="streamed per-row deltas + exact-identity check")
    rw.add_argument("left")
    rw.add_argument("right")
    rw.add_argument("--sample", type=int, default=None, help="deterministic N-row sample")
    rw.add_argument("--chunk", type=int, default=8192)

    for q in (sp, ln, rw):
        q.add_argument("--cache-dir", default=None)
        q.add_argument("--refresh", action="store_true", help="ignore the eigenbasis cache")
        q.add_argument("--remote", action="store_true",
                       help="force the HTTP-range path even if the hub cache has the model")
        q.add_argument("--json", action="store_true", help="print the raw JSON row(s)")
        q.add_argument("--out", type=Path, default=None, help="append JSON rows to this file")

    args = p.parse_args(argv)
    from . import api

    common = {"cache_dir": args.cache_dir, "refresh": args.refresh,
              "prefer_local": not args.remote}

    if args.leg == "spectrum":
        rows = []
        for model in args.models:
            fp = api.fingerprint(model, key=args.key, k=args.k, trim=not args.no_trim, **common)
            rows.append(fp.row())
            if not args.json:
                print(f"{fp.model}: V_cfg={fp.V_cfg} V_tok={fp.V_tok} d={fp.d} dtype={fp.dtype} "
                      f"trim={fp.trim} n50={fp.n50} ({fp.n50_d}d) n90={fp.n90} ({fp.n90_d}d) "
                      f"n95={fp.n95} PR={fp.pr_d}d source={fp.source}", flush=True)
        if args.json:
            print(json.dumps(rows, indent=1))
        _emit(rows, args.out)
        return 0

    if args.leg == "lineup":
        kw = {} if args.margin is None else {"ambiguity_margin": args.margin}
        lineup = api.identify_parent(args.student, args.candidates, k=args.k,
                                    strict_dtype=args.strict_dtype, **kw, **common)
        print(json.dumps(lineup.as_dict(), indent=1) if args.json else lineup.format())
        _emit([lineup.as_dict()], args.out)
        return 0

    rd = api.row_delta(args.left, args.right, sample=args.sample, chunk=args.chunk,
                       prefer_local=not args.remote)
    print(json.dumps(rd.as_dict(), indent=1) if args.json else
          f"{rd.left} vs {rd.right}: relF={rd.relF:.6g} med_row={rd.row_reld_median:.6g} "
          f"p90={rd.row_reld_p90:.6g} exact_rows={rd.exact_row_frac:.4f} "
          f"byte_identical={rd.byte_identical} V={rd.V_common} n={rd.n_rows} [{rd.note}]")
    _emit([rd.as_dict()], args.out)
    return 0
