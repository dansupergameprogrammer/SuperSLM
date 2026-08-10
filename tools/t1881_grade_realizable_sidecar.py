#!/usr/bin/env python3
r"""T-1881 -- grade the buildable (finite-stored-precision) Option E sidecar at site 4,
against T-1859's own ceiling and T-1777's committed true-bf16 reference.

Reads five width-1 pooled captures produced by `tools/t1859_option_e_measure.py
--single-arm <name> --k-cap <n>`, one config per invocation, this ticket's own
`run_t1881_captures.sh`:

    out/t1881_capture/shared/pooled/null.npz                    (k_cap irrelevant)
    out/t1881_capture/shared/pooled/base.npz                    (k_cap irrelevant)
    out/t1881_capture/kcap3/pooled/E_recovery_site4_only.npz     (coarser: 2-bit field)
    out/t1881_capture/kcap8/pooled/E_recovery_site4_only.npz     (M1's own: 4-bit field,
                                                                   T-1859's own ceiling)
    out/t1881_capture/kcap31/pooled/E_recovery_site4_only.npz    (finer: 5-bit field)

For each precision, computes:
  * recall@1 and the paired contrast against `base` (this cell's own baseline) --
    T-1859 S5.1's own convention, reproduced at width 1 instead of width 56;
  * the paired resolving power (1.96 * SE(delta)) and the ratio, per this campaign's own
    grading convention (`t1835_grade_sites.py`, `t1873_familywise_sweep.py`);
  * the fraction of the SAME-CONSTRUCTION ceiling this precision captures --
    frac = delta(this precision) / delta(k_cap=8, this run's own ceiling) -- both terms
    computed in this SAME width-1 cell, per D-SLM2128's own same-construction-denominator
    rule (T-1873 S3.2): the ceiling is this run's own k_cap=8 arm, not T-1859's filed
    width-56 figure, so no batch-width transfer assumption enters the fraction.

The reference is T-1777's own committed true-bf16 dump (independent of every arm this
ticket computes, unchanged since before this ticket existed) -- StandardsDocument.md S5.4's
comparison-validity requirement, satisfied the same way every prior ticket in this campaign
satisfies it.

Usage
    python tools\t1881_grade_realizable_sidecar.py \
        --root out\t1881_capture \
        --t1777-tools D:\SuperSLM\.worktrees\t1777-retrieval-agreement\tools \
        --t1777-out D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out \
        --json-out out\t1881_grades.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PRECISIONS = [
    ("kcap3", 3, "coarser (2-bit field)"),
    ("kcap8", 8, "M1's own (4-bit field, this run's own ceiling)"),
    ("kcap31", 31, "finer (5-bit field)"),
]


def _load_arm_recall1(npz_path: Path, rr, ref_vecs, ref_fps, domains, arm_key: str):
    if not npz_path.exists():
        raise SystemExit(f"missing pooled capture for {arm_key!r}: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    labs = [str(x) for x in z["labels"]]
    vecs = {l: z["vectors"][i] for i, l in enumerate(labs)}
    fps = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
    results, n_docs = rr.compare_arms(ref_vecs, vecs, domains, ref_fps, fps)
    order = [r.label for r in results]
    per_doc = np.array([r.recall[1] for r in results])
    return order, per_doc, n_docs


def _paired(order_a, a, order_b, b, z_crit):
    if order_a != order_b:
        raise SystemExit("document order mismatch -- not a valid pair")
    d = a - b
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    rp = 1.96 * se
    ratio = abs(mean) / rp if rp > 0 else float("nan")
    signed_ratio = mean / (z_crit * se) if se > 0 else float("nan")
    return {"n": n, "mean_delta": mean, "se": se, "paired_rp": rp, "ratio": ratio,
            "signed_ratio_z": signed_ratio}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="out/t1881_capture")
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1777-out", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    import t1777_retrieval_report as rr  # noqa: E402

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

    root = Path(args.root)
    order_null, r_null, n = _load_arm_recall1(
        root / "shared" / "pooled" / "null.npz", rr, ref_vecs, ref_fps, domains, "null")
    order_base, r_base, _ = _load_arm_recall1(
        root / "shared" / "pooled" / "base.npz", rr, ref_vecs, ref_fps, domains, "base")

    z_crit_single = 1.96  # m=1, this cell's own single-cell convention (matches t1859 S5.1)

    print(f"=== T-1881 width-1 cell, N={n} ===")
    print(f"null   recall@1 = {r_null.mean():.6f}")
    print(f"base   recall@1 = {r_base.mean():.6f}   (this cell's own baseline)")
    null_minus_base = _paired(order_null, r_null, order_base, r_base, z_crit_single)
    print(f"null - base (this cell's own accounting-basis denominator): "
          f"{null_minus_base['mean_delta']:.6f}")

    results = {}
    for tag, kcap, desc in PRECISIONS:
        p = root / tag / "pooled" / "E_recovery_site4_only.npz"
        order_e, r_e, n_e = _load_arm_recall1(p, rr, ref_vecs, ref_fps, domains,
                                               f"E_recovery_site4_only@{tag}")
        c = _paired(order_e, r_e, order_base, r_base, z_crit_single)
        verdict = "RESOLVED" if c["ratio"] >= 1.0 else "NOT RESOLVED"
        results[tag] = {"k_cap": kcap, "description": desc, "recall1": float(r_e.mean()),
                         **c, "verdict": verdict}
        print(f"\n--- {tag} (k_cap={kcap}, {desc}) ---")
        print(f"  recall@1 = {r_e.mean():.6f}")
        print(f"  delta vs base = {c['mean_delta']:+.6f}, paired RP = {c['paired_rp']:.4f}, "
              f"ratio = {c['ratio']:.2f}x -- {verdict}")
        print(f"  fraction of (null - base) accounting basis: "
              f"{c['mean_delta'] / null_minus_base['mean_delta'] * 100:.1f}%")

    ceiling_delta = results["kcap8"]["mean_delta"]
    print(f"\n=== fraction of THIS RUN'S OWN k_cap=8 ceiling captured, same-construction "
          f"denominator (D-SLM2128) ===")
    for tag, kcap, desc in PRECISIONS:
        frac = results[tag]["mean_delta"] / ceiling_delta if ceiling_delta else float("nan")
        results[tag]["frac_of_own_ceiling"] = frac
        print(f"  {tag} (k_cap={kcap}): {frac*100:.1f}% of the k_cap=8 ceiling's own recovery")

    # Cross-check against T-1859's filed width-56 figure (+0.087866, 1.41x) -- rough
    # magnitude, adjusted for the known width shift, per T-1879's own reproduction-gate
    # standard; not asserted bit-identical.
    print(f"\n=== cross-check vs T-1859's filed width-56 ceiling figure ===")
    print(f"  T-1859 filed (width 56): delta=+0.087866, ratio=1.41x")
    print(f"  this run, k_cap=8, width 1: delta={ceiling_delta:+.6f}, "
          f"ratio={results['kcap8']['ratio']:.2f}x")

    if args.json_out:
        payload = {
            "n": n,
            "null_recall1": float(r_null.mean()),
            "base_recall1": float(r_base.mean()),
            "null_minus_base": null_minus_base,
            "precisions": results,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
