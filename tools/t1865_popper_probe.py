#!/usr/bin/env python3
r"""T-1865 -- Popper probe against the T-1864 stage C result packet.

Read-only against the stage C and stage B cells. Re-grades using the SAME
T-1777 machinery the packet's own grader uses, then runs the attacks the
packet's grader does not run:

  A0  reproduce the packet's own reported numbers (pipeline agreement check)
  A1  grade the duplicate-base arms the packet's grader explicitly skips
  A2  the same, over stage B's 52 duplicate-base arms at batch width 56
  A3  cross-batch-width value reproduction: stage C m1m2_g128 vs stage B middle
  A4  fragility of every verdict, in whole documents
  A5  minimum resolvable effect per cell, in whole documents
  A6  family-wise null: permutation distribution of the max paired ratio

Usage
    python tools\t1865_popper_probe.py --stagec <dir> --stageb <dir> \
        --t1777-tools <dir> --t1777-out <dir> --json-out <path>
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
    return {
        "n": len(d),
        "delta": mean,
        "delta_docs": mean * len(d),
        "paired_rp": rp,
        "paired_rp_docs": rp * len(d),
        "paired_ratio": abs(mean) / rp if rp > 0 else float("nan"),
        "n_win": int((d > 0).sum()),
        "n_loss": int((d < 0).sum()),
        "n_tie": int((d == 0).sum()),
    }


def load_cell(cell: Path, rr, ref_vecs, ref_fps, domains):
    """-> {arm: {label: recall@1}}, {label: domain}, doc order"""
    per_doc, dom_of, order = {}, None, None
    for p in sorted(cell.glob("pooled/*.npz")):
        z = np.load(p, allow_pickle=True)
        labs = [str(x) for x in z["labels"]]
        vecs = {l: z["vectors"][i] for i, l in enumerate(labs)}
        fps = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
        results, nd = rr.compare_arms(ref_vecs, vecs, domains, ref_fps, fps)
        these = [r.label for r in results]
        if order is None:
            order, dom_of = these, {r.label: r.domain for r in results}
        elif these != order:
            raise SystemExit(f"{p.stem}: document order differs")
        per_doc[p.stem] = {r.label: r.recall[1] for r in results}
    return per_doc, dom_of, order


def raw_vectors(cell: Path, arm: str):
    z = np.load(cell / "pooled" / f"{arm}.npz", allow_pickle=True)
    return {str(l): z["vectors"][i] for i, l in enumerate(z["labels"])}


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

    C = Path(args.stagec)
    B = Path(args.stageb)
    pc, dom_c, order_c = load_cell(C, rr, ref_vecs, ref_fps, domains)
    report = {"stagec_arms": sorted(pc), "n_docs": len(order_c)}

    def vec(per_doc, arm, order):
        return np.array([per_doc[arm][l] for l in order], dtype=float)

    # --- binary check -------------------------------------------------------
    allvals = np.concatenate([vec(pc, a, order_c) for a in pc])
    report["recall1_is_binary"] = bool(set(np.unique(allvals)) <= {0.0, 1.0})
    print(f"[binary] recall@1 values observed: {sorted(set(np.unique(allvals)))}")

    # --- A0 / A1  every stage C arm, including the skipped duplicates -------
    print(f"\n=== A0/A1  stage C (batch width 9, N={len(order_c)}), every arm vs base ===")
    idx_id = [i for i, l in enumerate(order_c) if dom_c[l] == "id"]
    idx_ood = [i for i, l in enumerate(order_c) if dom_c[l] == "ood"]
    base_c = vec(pc, "base", order_c)
    a01 = {}
    for arm in sorted(pc):
        if arm == "base":
            continue
        v = vec(pc, arm, order_c)
        cells = {"pooled": paired(v, base_c),
                 "id": paired(v[idx_id], base_c[idx_id]),
                 "ood": paired(v[idx_ood], base_c[idx_ood])}
        a01[arm] = cells
        p = cells["pooled"]
        print(f"  {arm:14s} delta={p['delta']:+.6f} ({p['delta_docs']:+5.1f} docs)  "
              f"paired {p['paired_ratio']:.2f}x   RP={p['paired_rp_docs']:.1f} docs   "
              f"win/loss/tie={p['n_win']}/{p['n_loss']}/{p['n_tie']}")
    report["A01_stagec_vs_base"] = a01

    # duplicate-base arms: bit-level equality of the raw pooled vectors
    print("\n  [duplicate-base raw vector equality, stage C]")
    bv = raw_vectors(C, "base")
    dupeq = {}
    for arm in sorted(pc):
        if not arm.startswith("basedup"):
            continue
        av = raw_vectors(C, arm)
        exact = all(np.array_equal(bv[l], av[l]) for l in order_c)
        maxabs = max(float(np.abs(bv[l].astype(np.float64) - av[l].astype(np.float64)).max())
                     for l in order_c)
        dupeq[arm] = {"bit_identical_to_base": exact, "max_abs_diff": maxabs}
        print(f"    {arm}: bit-identical={exact}  max|diff|={maxabs:.3e}")
    report["A01_dup_raw_equality"] = dupeq

    # --- A2  stage B's 52 duplicate-base arms at batch width 56 -------------
    print(f"\n=== A2  stage B (batch width 56), duplicate-base arms vs base ===")
    pb, dom_b, order_b = load_cell(B, rr, ref_vecs, ref_fps, domains)
    base_b = vec(pb, "base", order_b)
    dups = [a for a in sorted(pb) if a.startswith("basedup")]
    dstats = []
    for arm in dups:
        dstats.append(paired(vec(pb, arm, order_b), base_b))
    dd = np.array([s["delta_docs"] for s in dstats])
    report["A2_stageb_dupbase"] = {
        "n_dup_arms": len(dups), "n_docs": len(order_b),
        "delta_docs_min": float(dd.min()) if len(dd) else None,
        "delta_docs_max": float(dd.max()) if len(dd) else None,
        "delta_docs_absmax": float(np.abs(dd).max()) if len(dd) else None,
        "n_nonzero": int((dd != 0).sum()),
    }
    print(f"  {len(dups)} duplicate-base arms, N={len(order_b)}: "
          f"delta range [{dd.min():+.1f}, {dd.max():+.1f}] docs, "
          f"{int((dd != 0).sum())} of {len(dups)} nonzero")

    # stage B's own real arms, for reference
    print("\n  [stage B real arms vs base]")
    b_real = {}
    for arm in sorted(pb):
        if arm in ("base",) or arm.startswith("basedup"):
            continue
        s = paired(vec(pb, arm, order_b), base_b)
        b_real[arm] = s
        print(f"    {arm:10s} delta={s['delta']:+.6f} ({s['delta_docs']:+5.1f} docs)  "
              f"paired {s['paired_ratio']:.2f}x  RP={s['paired_rp_docs']:.1f} docs")
    report["A2_stageb_real"] = b_real

    # --- A3  cross-batch-width reproduction --------------------------------
    print("\n=== A3  cross-batch-width: stage C (B=9) vs stage B (B=56), same arm ===")
    pairs = [("m1m2_g128", "middle"), ("base", "base"), ("null", "null")]
    a3 = {}
    common = [l for l in order_c if l in set(order_b)]
    for c_arm, b_arm in pairs:
        if c_arm not in pc or b_arm not in pb:
            continue
        cv, bvv = raw_vectors(C, c_arm), raw_vectors(B, b_arm)
        exact = all(np.array_equal(cv[l], bvv[l]) for l in common)
        maxabs = max(float(np.abs(cv[l].astype(np.float64) - bvv[l].astype(np.float64)).max())
                     for l in common)
        rc = np.array([pc[c_arm][l] for l in common])
        rb = np.array([pb[b_arm][l] for l in common])
        flips = int((rc != rb).sum())
        a3[f"{c_arm}_vs_{b_arm}"] = {
            "bit_identical": exact, "max_abs_diff": maxabs,
            "recall1_mean_stagec": float(rc.mean()), "recall1_mean_stageb": float(rb.mean()),
            "recall1_docs_stagec": float(rc.sum()), "recall1_docs_stageb": float(rb.sum()),
            "n_docs_flipped": flips, "n_common": len(common),
        }
        print(f"  {c_arm:11s} (B=9) vs {b_arm:7s} (B=56): bit-identical={exact}  "
              f"max|diff|={maxabs:.3e}  recall@1 {rc.sum():.0f} vs {rb.sum():.0f} docs  "
              f"({flips} documents flip)")
    report["A3_cross_batch_width"] = a3

    # --- A4  fragility, in whole documents ---------------------------------
    print("\n=== A4  fragility: document flips needed to cross the 1.0x threshold ===")
    def ratio_after_flips(v, anchor, k):
        """Flip the k documents that most support the verdict back to a tie
        (candidate := anchor) and recompute the paired ratio."""
        d = v - anchor
        supp = np.argsort(-np.abs(d))          # strongest contributors first
        vv = v.copy()
        for i in supp[:k]:
            vv[i] = anchor[i]
        return paired(vv, anchor)["paired_ratio"]

    a4 = {}
    for arm in sorted(pc):
        if arm == "base" or arm.startswith("basedup"):
            continue
        v = vec(pc, arm, order_c)
        r0 = paired(v, base_c)["paired_ratio"]
        k = 0
        if r0 > 1.0:
            while k < len(order_c) and ratio_after_flips(v, base_c, k) > 1.0:
                k += 1
            a4[arm] = {"ratio": r0, "flips_to_lose_resolution": k}
            print(f"  {arm:14s} {r0:.2f}x  -> loses RESOLVED after {k} document flip(s)")
        else:
            k = 0
            while k < len(order_c):
                dd2 = v - base_c
                order_supp = np.argsort(np.abs(dd2))
                vv = v.copy()
                added = 0
                for i in order_supp:
                    if added >= k:
                        break
                    if vv[i] == base_c[i]:
                        vv[i] = base_c[i] + np.sign(paired(v, base_c)["delta"] or 1.0)
                        added += 1
                if added < k:
                    break
                if paired(vv, base_c)["paired_ratio"] > 1.0:
                    break
                k += 1
            a4[arm] = {"ratio": r0, "flips_to_gain_resolution": k}
            print(f"  {arm:14s} {r0:.2f}x  -> would need {k} extra winning document(s) to RESOLVE")
    report["A4_fragility"] = a4

    # --- A5  minimum resolvable effect --------------------------------------
    print("\n=== A5  minimum resolvable effect (paired RP), in whole documents ===")
    a5 = {}
    for cellname, idx in (("pooled", list(range(len(order_c)))), ("id", idx_id), ("ood", idx_ood)):
        rps = {}
        for arm in sorted(pc):
            if arm == "base" or arm.startswith("basedup"):
                continue
            v = vec(pc, arm, order_c)[idx]
            s = paired(v, base_c[idx])
            rps[arm] = {"rp_recall1": s["paired_rp"], "rp_docs": s["paired_rp_docs"],
                        "delta_docs": s["delta_docs"], "n": s["n"]}
        a5[cellname] = rps
        vals = [r["rp_docs"] for r in rps.values()]
        print(f"  {cellname:7s} n={len(idx):3d}  RP spans {min(vals):.1f}-{max(vals):.1f} documents")
        for arm, r in rps.items():
            print(f"      {arm:14s} effect {r['delta_docs']:+5.1f} docs vs RP {r['rp_docs']:5.1f} docs")
    report["A5_min_resolvable_effect"] = a5

    # --- A6  family-wise null on the max paired ratio ------------------------
    print(f"\n=== A6  permutation null ({args.perm} draws) on the max paired ratio ===")
    rng = np.random.default_rng(args.seed)
    real_arms = [a for a in sorted(pc) if a != "base" and not a.startswith("basedup")]
    cells = [("pooled", list(range(len(order_c)))), ("id", idx_id), ("ood", idx_ood)]
    diffs = {a: vec(pc, a, order_c) - base_c for a in real_arms}
    observed = []
    for a in real_arms:
        for cn, idx in cells:
            observed.append((a, cn, paired(vec(pc, a, order_c)[idx], base_c[idx])["paired_ratio"]))
    obs_max = max(r for _, _, r in observed)

    def ratio_from_d(d):
        m = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        return abs(m) / (1.96 * se) if se > 0 else 0.0

    maxes = np.empty(args.perm)
    for t in range(args.perm):
        best = 0.0
        for a in real_arms:
            d = diffs[a] * rng.choice((-1.0, 1.0), size=len(order_c))
            for _, idx in cells:
                r = ratio_from_d(d[idx])
                if r > best:
                    best = r
        maxes[t] = best
    fw_p = float((maxes >= obs_max).mean())
    per_cell_p = {}
    for a, cn, r in observed:
        per_cell_p[f"{a}/{cn}"] = r
    report["A6_permutation"] = {
        "n_perm": args.perm, "observed_max_ratio": obs_max,
        "familywise_p_of_max": fw_p,
        "null_max_ratio_p50": float(np.percentile(maxes, 50)),
        "null_max_ratio_p95": float(np.percentile(maxes, 95)),
        "n_cells": len(observed), "observed_ratios": per_cell_p,
    }
    print(f"  observed max paired ratio across {len(observed)} base-anchored cells: {obs_max:.2f}x")
    print(f"  sign-flip null max ratio: median {np.percentile(maxes,50):.2f}x, "
          f"95th pct {np.percentile(maxes,95):.2f}x")
    print(f"  family-wise p(max >= observed) = {fw_p:.4f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
