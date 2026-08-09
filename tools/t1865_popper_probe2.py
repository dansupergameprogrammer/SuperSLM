#!/usr/bin/env python3
r"""T-1865 -- Popper probe, second pass.

A7  the same unmodified-engine construction measured three ways: the canonical
    engine int8 dump (batch width 1), stage B's in-harness `base` (width 56),
    and stage C's in-harness `base` (width 9).
A8  the recovery headroom, computed against each of those three anchors.
A6b the family-wise permutation null over the five attribution arms only
    (the `null` arm is a known-large sanity contrast, not an attribution arm).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def paired(cand, anchor):
    d = cand - anchor
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    rp = 1.96 * se
    return {"n": len(d), "delta_docs": mean * len(d), "rp_docs": rp * len(d),
            "ratio": abs(mean) / rp if rp > 0 else float("nan"),
            "verdict": ("RESOLVED" if (rp > 0 and abs(mean) / rp > 1.0) else "not resolved")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stagec", required=True)
    ap.add_argument("--stageb", required=True)
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1777-out", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1865)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    import t1777_retrieval_report as rr

    out = Path(args.t1777_out)
    labels, domains = [], {}
    with open(out / "t1777_corpus" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                labels.append(rec["label"])
                domains[rec["label"]] = rec.get("domain", rec.get("population", "ALL"))
    ref_vecs, ref_fps = rr.load_pooled_vectors(labels, out / "t1777_full_float_bf16", "float")

    def grade_dump(dump_dir, kind):
        v, f = rr.load_pooled_vectors(labels, out / dump_dir, kind)
        res, nd = rr.compare_arms(ref_vecs, v, domains, ref_fps, f)
        return {r.label: r.recall[1] for r in res}, [r.label for r in res], \
               {r.label: r.domain for r in res}

    def grade_cell(cell):
        per, order = {}, None
        for p in sorted(Path(cell).glob("pooled/*.npz")):
            z = np.load(p, allow_pickle=True)
            labs = [str(x) for x in z["labels"]]
            v = {l: z["vectors"][i] for i, l in enumerate(labs)}
            f = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
            res, nd = rr.compare_arms(ref_vecs, v, domains, ref_fps, f)
            order = order or [r.label for r in res]
            per[p.stem] = {r.label: r.recall[1] for r in res}
        return per, order

    canon, order, dom = grade_dump("t1777_full_int8", "int8")
    selfc, _, _ = grade_dump("t1777_full_float_fp32", "float")
    pc, order_c = grade_cell(args.stagec)
    pb, order_b = grade_cell(args.stageb)
    assert order == order_c == order_b, "document order differs across cells"

    V = lambda d: np.array([d[l] for l in order], dtype=float)
    report = {"n_docs": len(order)}

    # --- A7 -----------------------------------------------------------------
    print("=== A7  the SAME unmodified-engine construction, measured three ways ===")
    anchors = {
        "canonical engine int8 (batch width 1)": V(canon),
        "stage B in-harness base (batch width 56)": V(pb["base"]),
        "stage C in-harness base (batch width 9)": V(pc["base"]),
    }
    for name, a in anchors.items():
        print(f"  {name:44s} recall@1 = {a.sum():.0f} / {len(order)} docs  ({a.mean():.4f})")
    a7 = {k: {"docs": float(v.sum()), "recall1": float(v.mean())} for k, v in anchors.items()}
    spread = max(v.sum() for v in anchors.values()) - min(v.sum() for v in anchors.values())
    print(f"  spread across batch width alone: {spread:.0f} documents")
    a7["spread_docs_across_batch_width"] = float(spread)

    print("\n  pairwise, paired per document:")
    ks = list(anchors)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            s = paired(anchors[ks[j]], anchors[ks[i]])
            flips = int((anchors[ks[j]] != anchors[ks[i]]).sum())
            print(f"    {ks[j][:26]:26s} - {ks[i][:26]:26s} = {s['delta_docs']:+5.1f} docs "
                  f"({flips} flip) paired {s['ratio']:.2f}x ({s['verdict']})")
            a7[f"{ks[j]} MINUS {ks[i]}"] = {**s, "n_docs_flipped": flips}
    report["A7_base_measured_three_ways"] = a7

    # --- A8 -----------------------------------------------------------------
    print("\n=== A8  the composed arm's recovery, against each anchor ===")
    ceiling = V(selfc)
    cand = V(pc["m1m2_g128"])
    print(f"  reference self-consistency ceiling (fp32): {ceiling.sum():.0f} docs")
    print(f"  composed arm m1m2_g128 (batch width 9):    {cand.sum():.0f} docs")
    print(f"  composed arm middle    (batch width 56):   {V(pb['middle']).sum():.0f} docs")
    a8 = {"ceiling_docs": float(ceiling.sum()), "composed_w9_docs": float(cand.sum()),
          "composed_w56_docs": float(V(pb["middle"]).sum()), "per_anchor": {}}
    for name, a in anchors.items():
        head = ceiling.sum() - a.sum()
        rec = cand.sum() - a.sum()
        s = paired(cand, a)
        frac = rec / head if head else float("nan")
        print(f"  vs {name:44s} headroom={head:5.0f} docs  recovered={rec:+5.1f} docs "
              f"({frac*100:5.1f}%)  paired {s['ratio']:.2f}x ({s['verdict']})")
        a8["per_anchor"][name] = {"headroom_docs": float(head), "recovered_docs": float(rec),
                                  "fraction_of_headroom": float(frac), **s}
    report["A8_recovery_vs_anchor"] = a8

    # --- A6b ----------------------------------------------------------------
    arms = ["m1_g32", "m1_g128", "m2_residuals", "m2_site16", "m1m2_g128"]
    idx = {"pooled": list(range(len(order))),
           "id": [i for i, l in enumerate(order) if dom[l] == "id"],
           "ood": [i for i, l in enumerate(order) if dom[l] == "ood"]}
    base_c = V(pc["base"])
    diffs = {a: V(pc[a]) - base_c for a in arms}

    def ratio_from_d(d):
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        return abs(float(d.mean())) / (1.96 * se) if se > 0 else 0.0

    observed = {f"{a}/{c}": ratio_from_d(diffs[a][i]) for a in arms for c, i in idx.items()}
    obs_max = max(observed.values())
    rng = np.random.default_rng(args.seed)
    maxes = np.empty(args.perm)
    for t in range(args.perm):
        best = 0.0
        for a in arms:
            d = diffs[a] * rng.choice((-1.0, 1.0), size=len(order))
            for c, i in idx.items():
                r = ratio_from_d(d[i])
                if r > best:
                    best = r
        maxes[t] = best
    fw_p = float((maxes >= obs_max).mean())
    print(f"\n=== A6b  family-wise null over the FIVE attribution arms x 3 cells "
          f"({len(observed)} cells, {args.perm} sign-flip draws) ===")
    print(f"  observed max paired ratio: {obs_max:.2f}x  ({max(observed, key=observed.get)})")
    print(f"  null max ratio: median {np.percentile(maxes,50):.2f}x  "
          f"90th {np.percentile(maxes,90):.2f}x  95th {np.percentile(maxes,95):.2f}x")
    print(f"  family-wise p(max >= observed) = {fw_p:.4f}")
    print(f"  fraction of null draws whose max exceeds 1.0x = "
          f"{float((maxes > 1.0).mean()):.4f}")
    report["A6b_familywise"] = {
        "n_cells": len(observed), "n_perm": args.perm, "observed": observed,
        "observed_max": obs_max, "familywise_p": fw_p,
        "null_p50": float(np.percentile(maxes, 50)), "null_p90": float(np.percentile(maxes, 90)),
        "null_p95": float(np.percentile(maxes, 95)),
        "null_frac_max_over_1x": float((maxes > 1.0).mean()),
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
