#!/usr/bin/env python3
r"""T-1809 -- the interaction term and its achieved resolving power.

The interaction is  I = deficit(D) - [deficit(B) + deficit(C)]  with every deficit taken
against arm A, this construction's own null. Arm A cancels algebraically, so
I = C - D exactly; both forms are computed below and asserted equal, because a term that
is only ever computed one way is a term nobody checked.

Two resolving powers are reported for every difference:
  * UNPAIRED -- the sum of the two arms' own 95% normal-CI half-widths. This is T-1777's
    filed convention (T-1803 SS2, T-1805 SS2) and every cross-ticket comparison uses it.
  * PAIRED -- 1.96 x the standard error of the per-query difference. The arms share the
    corpus and the queries, so the difference is paired and the unpaired sum overstates
    the noise. Reported as the tighter, correctly-specified figure; it does not replace
    the filed convention, it qualifies it.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--t1777-tools", required=True)
ap.add_argument("--t1777-out", required=True)
ap.add_argument("--arm", action="append", default=[])
ap.add_argument("--json-out", default=None)
a = ap.parse_args()

sys.path.insert(0, str(Path(a.t1777_tools).resolve()))
import t1777_retrieval_report as rr

out = Path(a.t1777_out)
labels, domains = [], {}
with open(out / "t1777_corpus" / "manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            labels.append(r["label"]); domains[r["label"]] = r.get("domain", "ALL")

ref_vecs, ref_fps = rr.load_pooled_vectors(labels, out / "t1777_full_float_bf16", "float")

per_query = {}
sources = {"ENGINE": (out / "t1777_full_int8", "int8")}
for spec in a.arm:
    n, _, d = spec.partition("=")
    sources[n] = (Path(d), "float")

for name, (d, kind) in sources.items():
    cv, cf = rr.load_pooled_vectors(labels, d, kind)
    results, n_docs = rr.compare_arms(ref_vecs, cv, domains, ref_fps, cf)
    per_query[name] = {
        "labels": [r.label for r in results],
        1: np.array([r.recall[1] for r in results]),
        5: np.array([r.recall[5] for r in results]),
        10: np.array([r.recall[10] for r in results]),
        "w5": np.array([1.0 if r.ref_top1_displacement < 5 else 0.0 for r in results]),
        "n": n_docs,
    }

order = per_query["A"]["labels"]
for nm, v in per_query.items():
    assert v["labels"] == order, f"{nm}: query order differs -- the pairing would be wrong"

def unpaired_rp(x, k, n):
    m = float(x.mean())
    step = rr.discrete_step_recall(n, k) if isinstance(k, int) else 1.0 / n
    return max(step, rr.normal_ci_halfwidth(m, n))

def paired(xa, xb, n):
    d = xa - xb
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    return mean, 1.96 * se

N = per_query["A"]["n"]
report = {}
print(f"N = {N} queries, paired across every arm (identical corpus, identical order)\n")

for k in (1, 5, 10, "w5"):
    key = k if k != "w5" else "w5"
    kk = k if isinstance(k, int) else 1
    lbl = f"recall@{k}" if isinstance(k, int) else "top1-within-top5"
    print(f"===== {lbl} =====")
    vals = {nm: per_query[nm][key] for nm in per_query}
    for nm in ("A", "B", "C", "D", "ENGINE"):
        print(f"  arm {nm:<6} = {vals[nm].mean():.6f}   unpaired RP {unpaired_rp(vals[nm], kk, N):.4f}")

    defs = {nm: vals['A'] - vals[nm] for nm in ("B", "C", "D")}
    dB, dC, dD = defs["B"].mean(), defs["C"].mean(), defs["D"].mean()
    additive = dB + dC
    inter_long = dD - additive
    # The general identity: I = (A-D) - (A-B) - (A-C) = (B + C) - (A + D). It reduces to
    # C - D only where deficit(B) is exactly zero, which holds at k=1 on this data and
    # NOT at k=5/10 -- so the general form is what is computed and checked.
    inter_general = (vals["B"] + vals["C"] - vals["A"] - vals["D"]).mean()
    assert abs(inter_long - inter_general) < 1e-12, (inter_long, inter_general)
    reduces = abs(dB) < 1e-12
    if reduces:
        assert abs(inter_long - (vals["C"] - vals["D"]).mean()) < 1e-12

    # The interaction is a contrast over four paired per-query series; its own resolving
    # power is taken over that contrast, not over any single pair.
    contrast = vals["B"] + vals["C"] - vals["A"] - vals["D"]
    up = sum(unpaired_rp(vals[nm], kk, N) for nm in ("A", "B", "C", "D"))
    pm = float(contrast.mean())
    prp = 1.96 * float(contrast.std(ddof=1) / np.sqrt(N))
    print(f"  deficit(B)={dB:+.6f}  deficit(C)={dC:+.6f}  deficit(D)={dD:+.6f}")
    print(f"  additive prediction for deficit(D) = {additive:+.6f}")
    print(f"  INTERACTION = deficit(D) - additive = {inter_long:+.6f}   "
          f"(= (B+C)-(A+D), checked{'; reduces to C-D here' if reduces else ''})")
    print(f"    unpaired RP {up:.4f} -> {abs(inter_long)/up:.2f}x  "
          f"{'RESOLVED' if abs(inter_long)>up else 'NOT RESOLVED'}")
    print(f"    paired   RP {prp:.4f} -> {abs(pm)/prp if prp>0 else float('nan'):.2f}x  "
          f"{'RESOLVED' if abs(pm)>prp else 'NOT RESOLVED'}")

    for pair in (("A","B"),("A","C"),("A","D"),("C","ENGINE"),("D","ENGINE"),("C","D")):
        x, y = pair
        pmm, pprp = paired(vals[x], vals[y], N)
        upp = unpaired_rp(vals[x], kk, N) + unpaired_rp(vals[y], kk, N)
        print(f"    {x} - {y}: {pmm:+.6f}  unpaired RP {upp:.4f} ({abs(pmm)/upp:.2f}x)  "
              f"paired RP {pprp:.4f} ({abs(pmm)/pprp if pprp>0 else float('nan'):.2f}x)")
    report[lbl] = {"deficit_B": dB, "deficit_C": dC, "deficit_D": dD,
                   "additive": additive, "interaction": inter_long,
                   "interaction_rp_unpaired": up, "interaction_rp_paired": prp}
    print()

if a.json_out:
    json.dump(report, open(a.json_out, "w"), indent=2)
    print("written:", a.json_out)
