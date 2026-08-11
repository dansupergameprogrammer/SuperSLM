#!/usr/bin/env python3
r"""T-1906 -- grade activation-width-curve captures through T-1777's own unmodified report
functions.

This is `t1809_grade_arms.py` with only its labels changed (T-1809 arm -> T-1906 point); the
comparison logic is untouched and reused rather than re-authored, per this ticket's own
comparability requirement.

Imports `t1777_retrieval_report`'s `load_pooled_vectors`, `compare_arms`,
`normal_ci_halfwidth`, and `discrete_step_recall` and calls them directly rather than
through the CLI, for the reason T-1805 recorded: the CLI fixes which directory plays
reference and which plays candidate in each of its two hardcoded comparisons, and the
`top1_within_top5` statistic is ASYMMETRIC in those roles. Calling `compare_arms` with the
roles named explicitly keeps T-1777's own reference (the true bf16 dumps) on the reference
side in every row, exactly as the engine baseline itself is scored.

Nothing in T-1777's worktree is written. Every figure printed here is computed by T-1777's
own functions on T-1777's own committed reference dumps.

T-1907 findings S1 and S2 (fixed, same commit as the dump tool's C1/S3 fixes):

S1 -- a `--arm NAME=DIR` width point's `.float.bin` files carry no width field of their own
(T-1777's own header format, kept unmodified, has none), so nothing previously stopped an
int8 directory from being graded and reported as though it were int16. Every width-point
directory is now required to carry `t1906_activation_width_dump.py`'s own `manifest.json`
sidecar (written by that tool, not by this one); this grader reads it, prints the bits it
found, and REFUSES -- exits non-zero without printing a verdict for that arm -- when the
manifest is missing, or when the operator's own `NAME` encodes a bit width (a trailing
`bits<N>`/`b<N>` token) that disagrees with the manifest's recorded `bits`. T-1777's own two
fixed baseline directories (the engine int8 dump, the fp32 self-consistency dump) are exempt
-- they are T-1777's own committed artifacts, never produced by this harness, and carry their
own provenance already checked by `compare_arms`'s fingerprint guard.

S2 -- `gap()` differenced two rows' `recall@k` without ever comparing their `n`, so a
width-point directory short of the full corpus (a `--limit` run, or documents lost to the
dump tool's own per-document failure path) was silently differenced against the fixed
239-document baseline rows and printed a resolution verdict anyway. `gap()` now refuses --
prints REFUSED, computes nothing -- whenever the two rows' populations differ.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_BITS_IN_NAME = re.compile(r"bits?(\d+)", re.IGNORECASE)


def _read_manifest_or_refuse(name: str, d: Path) -> int:
    """T-1907 S1. Returns the manifest's `bits`, or raises SystemExit (refuses) if the
    directory carries no manifest, or if `name` encodes a bit width that disagrees with it.
    """
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"REFUSED: --arm {name}={d} carries no manifest.json -- this grader cannot "
            f"confirm what activation width this directory was captured at, and grading it "
            f"would silently compare an unknown width against the named width points "
            f"(T-1907 finding S1). Regenerate it with t1906_activation_width_dump.py, which "
            f"writes manifest.json for every run.")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    bits = manifest.get("bits")
    if bits is None:
        raise SystemExit(f"REFUSED: {manifest_path} carries no 'bits' field.")
    m = _BITS_IN_NAME.search(name)
    if m is not None:
        claimed = int(m.group(1))
        if claimed != bits:
            raise SystemExit(
                f"REFUSED: --arm {name}={d} -- the name claims bits={claimed} but "
                f"{manifest_path} records bits={bits} (T-1907 finding S1). Two widths were "
                f"about to be compared as though they were the same arm.")
    if manifest.get("complete") is False:
        print(f"  [{name}] WARNING: manifest.json records complete=False -- this capture "
              f"had per-document failures; n_ok={manifest.get('n_ok')} "
              f"n_failed={manifest.get('n_failed')}", flush=True)
    return bits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1777-out", required=True,
                    help="T-1777's out/ directory (the committed reference dumps)")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR",
                    help="a width point to grade, e.g. bits16=out/t1906_arm_c_b16; repeatable")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    import t1777_retrieval_report as rr

    out = Path(args.t1777_out)
    manifest = out / "t1777_corpus" / "manifest.jsonl"
    labels, domains = [], {}
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                labels.append(rec["label"])
                domains[rec["label"]] = rec.get("domain", rec.get("population", "ALL"))

    ref_vecs, ref_fps = rr.load_pooled_vectors(labels, out / "t1777_full_float_bf16", "float")

    arms = {
        "engine int8 (canonical baseline)": (out / "t1777_full_int8", "int8", None),
        "reference self-consistency: true fp32": (out / "t1777_full_float_fp32", "float", None),
    }
    for spec in args.arm:
        name, _, d = spec.partition("=")
        d = Path(d)
        bits = _read_manifest_or_refuse(name, d)   # T-1907 S1 -- refuses on failure, else here
        arms[f"T-1906 {name} (bits={bits})"] = (d, "float", bits)

    rows = {}
    for name, (d, kind, bits) in arms.items():
        cand_vecs, cand_fps = rr.load_pooled_vectors(labels, d, kind)
        results, n_docs = rr.compare_arms(ref_vecs, cand_vecs, domains, ref_fps, cand_fps)
        row = {"n": n_docs}
        for k in (1, 5, 10):
            vals = np.array([r.recall[k] for r in results])
            mean = float(vals.mean())
            step = rr.discrete_step_recall(n_docs, k)
            ci = rr.normal_ci_halfwidth(mean, n_docs)
            row[f"recall@{k}"] = mean
            row[f"rp@{k}"] = float(max(step, ci))
            row[f"num@{k}"] = float(vals.sum())
        top1 = np.array([r.ref_top1_displacement for r in results])
        w5 = float((top1 < 5).mean())
        row["within_top5"] = w5
        row["within_top5_n"] = int((top1 < 5).sum())
        row["within_top5_rp"] = float(max(1.0 / n_docs, rr.normal_ci_halfwidth(w5, n_docs)))
        rhos = np.array([r.spearman_rho for r in results])
        row["spearman_mean"] = float(rhos.mean())
        row["spearman_min"] = float(rhos.min())
        rows[name] = row

        print(f"\n=== {name}  (N={n_docs}) ===")
        for k in (1, 5, 10):
            print(f"  recall@{k:<2} = {row[f'recall@{k}']:.6f}   "
                  f"(sum {row[f'num@{k}']:.4f} / {n_docs})   "
                  f"resolving power {row[f'rp@{k}']:.4f}")
        print(f"  top1-within-top5 = {w5:.6f} ({row['within_top5_n']}/{n_docs})   "
              f"resolving power {row['within_top5_rp']:.4f}")
        print(f"  spearman mean = {row['spearman_mean']:.4f}  min = {row['spearman_min']:.4f}")

    def gap(a, b, k):
        """A - B with the two rows' resolving powers summed, per StandardsDocument 5.4.

        T-1907 finding S2 (fixed): refuses -- returns (None, None, None) -- rather than
        computing a difference when the two rows were graded over different populations.
        `StandardsDocument.md` 5.4 requires a comparison be over the same population before
        it is evidence of anything; per 4, the check refuses rather than warns.
        """
        if rows[a]["n"] != rows[b]["n"]:
            return None, None, None
        d = rows[a][f"recall@{k}"] - rows[b][f"recall@{k}"]
        rp = rows[a][f"rp@{k}"] + rows[b][f"rp@{k}"]
        return d, rp, (abs(d) / rp if rp > 0 else float("nan"))

    print("\n=== pairwise gaps on recall@1 (the binding metric, D-SLM1455) ===")
    names = list(rows)
    any_refused = False
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d, rp, ratio = gap(a, b, 1)
            if d is None:
                any_refused = True
                print(f"  {a}\n    minus {b}\n    = REFUSED -- population mismatch "
                      f"(N={rows[a]['n']} vs N={rows[b]['n']}), not the same quantity "
                      f"(T-1907 finding S2)")
                continue
            verdict = ("RESOLVED %.2fx past combined resolving power" % ratio) if ratio > 1.0 \
                else "NOT DISTINGUISHABLE at this N"
            print(f"  {a}\n    minus {b}\n    = {d:+.6f}  combined RP {rp:.4f}  -> {verdict}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwritten: {args.json_out}")
    return 1 if any_refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
