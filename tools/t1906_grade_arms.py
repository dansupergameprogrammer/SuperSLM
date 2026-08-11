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

T-1907 round 1 findings S1 and S2 (fixed): a `--arm NAME=DIR` directory now requires a
`manifest.json` sidecar naming its own `bits`, and `gap()` refuses rather than differencing
two rows measured over different populations.

T-1907 round 2 findings (this commit):

C1 -- the round-1 manifest certified the RUN that last wrote into a directory, not the FILES
sitting in it: nothing cleared `--out-dir` before capture, so a `--limit`/`--bits` re-run into
a populated directory left a manifest describing the new run beside files an old run left
behind, and this grader trusted the manifest's `bits` without checking that its recorded
population matched what was actually graded. Two closes, both required now: the dump tool
refuses (or clears, with `--overwrite`) a populated `--out-dir` before capture, so a manifest
can only ever describe the directory it sits in; and this grader requires the manifest's
`labels` (the exact set the dump tool captured) to equal the label set `compare_arms` actually
graded, and its `n_ok` to equal the graded `n_docs` -- refusing before printing a verdict for
that arm if either disagrees. Label-SET equality is strictly stronger than an `n_ok`/`n_docs`
count match (two different 40-document subsets would pass a count check and fail this one).

S1 -- the name-vs-manifest guard matched `bits<N>` only, while this docstring (round 1) and
the build record claimed `b<N>` was also covered. `_BITS_IN_NAME` now matches `bits`, `bit`,
or a bare `b` prefix, anchored at a LEADING word boundary only (a trailing one broke the
`bits16_rerun`/`b16_final` cases the casebook itself lists, since `_` counts as a word
character and leaves no boundary between a digit and a following underscore -- caught by
executing the casebook's own 12-name test set before trusting the fix). `int16`, `16bit`, and
`width16` still do not match; `b8`, `bits16_rerun`, `b16_final`, `B16` all now do.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_BITS_IN_NAME = re.compile(r"\b(?:bits?|b)(\d+)", re.IGNORECASE)


def _read_manifest_or_refuse(name: str, d: Path) -> tuple:
    """T-1907 S1 (round 1, name/bits check) and C1 (round 2, population check deferred to the
    caller since it needs the graded results). Returns `(bits, expected_n_ok, expected_labels)`,
    or raises SystemExit (refuses) if the directory carries no manifest, the manifest is
    missing the fields round 2 requires, or `name` encodes a bit width that disagrees with the
    manifest's recorded `bits`.
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
    n_ok = manifest.get("n_ok")
    labels = manifest.get("labels")
    if n_ok is None or labels is None:
        raise SystemExit(
            f"REFUSED: --arm {name}={d} -- {manifest_path} carries no 'n_ok'/'labels' field "
            f"(T-1907 finding C1, round 2). This grader cannot confirm the graded population "
            f"matches what this directory's capture actually produced; either the run never "
            f"completed (manifest still reads complete=false) or it predates the C1 fix. "
            f"Regenerate the directory.")
    if manifest.get("complete") is False:
        print(f"  [{name}] WARNING: manifest.json records complete=False -- this capture "
              f"had per-document failures; n_ok={n_ok} n_failed={manifest.get('n_failed')}",
              flush=True)
    return bits, n_ok, set(labels)


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
        "engine int8 (canonical baseline)": (out / "t1777_full_int8", "int8", None, None, None),
        "reference self-consistency: true fp32": (
            out / "t1777_full_float_fp32", "float", None, None, None),
    }
    for spec in args.arm:
        name, _, d = spec.partition("=")
        d = Path(d)
        # T-1907 S1 (name/bits) refuses inside this call; C1's population check needs the
        # graded results and happens below, per-row, before that row is printed.
        bits, expected_n_ok, expected_labels = _read_manifest_or_refuse(name, d)
        arms[f"T-1906 {name} (bits={bits})"] = (d, "float", bits, expected_n_ok, expected_labels)

    rows = {}
    for name, (d, kind, bits, expected_n_ok, expected_labels) in arms.items():
        cand_vecs, cand_fps = rr.load_pooled_vectors(labels, d, kind)
        results, n_docs = rr.compare_arms(ref_vecs, cand_vecs, domains, ref_fps, cand_fps)

        if expected_labels is not None:   # T-1907 finding C1, round 2 -- width-point arms only
            graded_labels = {r.label for r in results}
            if n_docs != expected_n_ok or graded_labels != expected_labels:
                raise SystemExit(
                    f"REFUSED: --arm {name}={d} -- graded N={n_docs} over "
                    f"{len(graded_labels)} labels, but manifest.json records n_ok="
                    f"{expected_n_ok} over {len(expected_labels)} labels (T-1907 finding C1, "
                    f"round 2). The directory's contents do not match what its own manifest "
                    f"certifies -- symmetric difference: "
                    f"{len(graded_labels ^ expected_labels)} label(s). This is exactly the "
                    f"stale-directory scenario the manifest exists to catch; regenerate it.")

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
