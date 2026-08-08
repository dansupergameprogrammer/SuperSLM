#!/usr/bin/env python3
r"""T-1835 -- the ranked per-site attribution table, over two independent numeric cells.

Two quantities per site, and they answer different questions:

  ISOLATED INJECTION (the `onlySS` arms).  One site quantized, seventeen float. Its final
  residual state's relative L2 against the `null` element of the SAME batch is the error that
  site alone injects, and its `recall@1` deficit against `null` is what that error costs the
  encoder oracle. This is the quantity the commission asks for and the one that discriminates:
  it is bounded below by zero, it is monotone in the site's own coarseness, and it does not
  ask the other seventeen sites' noise to cancel.

  MARGINAL REMOVAL (the `offSS` arms).  Seventeen sites quantized, one float. Its contrast
  with `base` is what removing that one site buys in the presence of all the others. On a
  mechanism that injects fresh independent noise at every site of every layer (D-SLM1570), a
  single site's removal is a small change to a large sum AND it re-rolls every rounding
  decision downstream of it, so this quantity is expected to be dominated by realization
  change rather than by the site. It is reported because the expectation is a claim that has
  to be measured, not assumed -- and because a site that DOES stand out here is carrying
  something the isolated figure cannot show.

The two cells (A at batch 59, B at batch 56) select different GEMM kernels and are therefore
different numeric realizations of the same 56 configurations. Their disagreement on a figure is
that figure's run-to-run resolving power, measured rather than modelled, and every claim below
is stated against it.

Usage
    python tools\t1835_attribution.py --cell-a out\t1835_cellA --cell-b out\t1835_cellB \
        --grades-a out\t1835_grades_cellA.json --grades-b out\t1835_grades_cellB.json \
        --json-out out\t1835_attribution.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

SITE_RE = re.compile(r"^(off|only)(\d\d)_(.+)$")
GROUP_RE = re.compile(r"^(offG|onlyG)_(.+)$")


def load_cell(root: Path):
    d = np.load(root / "drift" / "drift.npz", allow_pickle=True)
    arms = [str(a) for a in d["arms"]]
    rel = d["rel"]                     # (samples, B, rows)
    cos = d["cos"]
    positions = d["positions"]
    return {"arms": arms, "rel": rel, "cos": cos, "positions": positions,
            "labels": [str(x) for x in d["labels"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-a", required=True)
    ap.add_argument("--cell-b", required=True)
    ap.add_argument("--grades-a", required=True)
    ap.add_argument("--grades-b", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    cells = {"A": load_cell(Path(args.cell_a)), "B": load_cell(Path(args.cell_b))}
    grades = {"A": json.loads(Path(args.grades_a).read_text(encoding="utf-8")),
              "B": json.loads(Path(args.grades_b).read_text(encoding="utf-8"))}

    final_row = cells["A"]["rel"].shape[2] - 1
    # The graded embedding pools positions 1..n-1, so the drift figure that stands beside a
    # recall figure is taken over the same positions.
    def drift(cell, arm, row=final_row, pooled_positions=True):
        c = cells[cell]
        b = c["arms"].index(arm)
        sel = c["positions"] >= 1 if pooled_positions else slice(None)
        return c["rel"][sel, b, row]

    def stat(cell, arm, row=final_row):
        v = drift(cell, arm, row)
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "rms": float(np.sqrt((v ** 2).mean())), "n": int(v.size)}

    arms = cells["A"]["arms"]
    sites = {}
    for a in arms:
        m = SITE_RE.match(a)
        if m:
            kind, num, name = m.group(1), int(m.group(2)), m.group(3)
            sites.setdefault(num, {"name": name})[kind] = a
    groups = {}
    for a in arms:
        m = GROUP_RE.match(a)
        if m:
            kind, name = m.group(1), m.group(2)
            groups.setdefault(name, {})[kind] = a

    null_r1 = {c: grades[c]["rows"]["null"]["recall@1"] for c in ("A", "B")}
    base_r1 = {c: grades[c]["rows"]["base"]["recall@1"] for c in ("A", "B")}

    out = {"sites": {}, "groups": {}, "cells": {}, "additivity": {}}
    for c in ("A", "B"):
        out["cells"][c] = {
            "null_recall1": null_r1[c], "base_recall1": base_r1[c],
            "total_deficit": null_r1[c] - base_r1[c],
            "null_rp": grades[c]["rows"]["null"]["rp@1"],
            "base_rp": grades[c]["rows"]["base"]["rp@1"],
            "base_drift_final_mean": stat(c, "base")["mean"],
        }

    print("=== isolated injection: one site quantized, seventeen float ===")
    print(f"{'#':>3} {'site':22s} {'driftA':>8s} {'driftB':>8s} {'spread':>7s} "
          f"{'r@1 A':>8s} {'r@1 B':>8s} {'defA':>8s} {'shareA':>7s} {'pairedRP':>9s} {'res':>5s}")
    rows = []
    for num in sorted(sites):
        s = sites[num]
        only = s.get("only")
        if only is None:
            continue
        da, db = stat("A", only)["mean"], stat("B", only)["mean"]
        ga = grades["A"]["rows"][only]
        gb = grades["B"]["rows"][only]
        # deficit against the arm's own cell's `null`, and the contrast's own paired RP
        # (computed in the grader against `base`; recomputed here against `null`).
        defa = null_r1["A"] - ga["recall@1"]
        defb = null_r1["B"] - gb["recall@1"]
        total_a = null_r1["A"] - base_r1["A"]
        share = defa / total_a if total_a else float("nan")
        # paired RP of `only` against `base` is in the grades; against `null` we use the
        # unpaired sum of the two rows' own half-widths, which is the campaign's filed
        # convention when a paired series is not carried.
        rp = ga["rp@1"] + grades["A"]["rows"]["null"]["rp@1"]
        res = "yes" if abs(defa) > rp else "no"
        rows.append({"site": num, "name": s["name"], "arm": only,
                     "drift_A": da, "drift_B": db, "drift_spread": abs(da - db),
                     "recall1_A": ga["recall@1"], "recall1_B": gb["recall@1"],
                     "deficit_A": defa, "deficit_B": defb,
                     "share_of_total_A": share, "rp_vs_null_A": rp, "resolved": res})
    rows.sort(key=lambda r: -r["drift_A"])
    for r in rows:
        print(f"{r['site']:>3d} {r['name']:22s} {r['drift_A']:8.4f} {r['drift_B']:8.4f} "
              f"{r['drift_spread']:7.4f} {r['recall1_A']:8.4f} {r['recall1_B']:8.4f} "
              f"{r['deficit_A']:+8.4f} {r['share_of_total_A']:7.1%} {r['rp_vs_null_A']:9.4f} "
              f"{r['resolved']:>5s}")
    out["sites"] = rows

    print("\n=== marginal removal: seventeen sites quantized, one float (contrast vs base) ===")
    print(f"{'#':>3} {'site':22s} {'driftA':>8s} {'d-base':>8s} {'dr@1 A':>8s} {'dr@1 B':>8s} "
          f"{'pairedRP':>9s} {'ratio':>6s}")
    off_rows = []
    for num in sorted(sites):
        off = sites[num].get("off")
        if off is None:
            continue
        da = stat("A", off)["mean"]
        ca = grades["A"]["contrasts"][off]
        cb = grades["B"]["contrasts"][off]
        off_rows.append({"site": num, "name": sites[num]["name"], "arm": off,
                         "drift_A": da, "drift_minus_base": da - stat("A", "base")["mean"],
                         "d_recall1_A": ca["delta_recall1"], "d_recall1_B": cb["delta_recall1"],
                         "paired_rp_A": ca["paired_rp"], "paired_ratio_A": ca["paired_ratio"]})
    off_rows.sort(key=lambda r: -r["d_recall1_A"])
    for r in off_rows:
        print(f"{r['site']:>3d} {r['name']:22s} {r['drift_A']:8.4f} "
              f"{r['drift_minus_base']:+8.4f} {r['d_recall1_A']:+8.4f} {r['d_recall1_B']:+8.4f} "
              f"{r['paired_rp_A']:9.4f} {r['paired_ratio_A']:6.2f}")
    out["off_sites"] = off_rows

    print("\n=== groups ===")
    grp = []
    for name in sorted(groups):
        g = groups[name]
        row = {"group": name}
        for kind, arm in g.items():
            row[f"{kind}_drift_A"] = stat("A", arm)["mean"]
            row[f"{kind}_recall1_A"] = grades["A"]["rows"][arm]["recall@1"]
            row[f"{kind}_recall1_B"] = grades["B"]["rows"][arm]["recall@1"]
        grp.append(row)
        print(f"  {name:20s} onlyG drift {row.get('onlyG_drift_A', float('nan')):.4f}  "
              f"recall@1 {row.get('onlyG_recall1_A', float('nan')):.4f}   |   "
              f"offG drift {row.get('offG_drift_A', float('nan')):.4f}  "
              f"recall@1 {row.get('offG_recall1_A', float('nan')):.4f}")
    out["groups"] = grp

    print("\n=== additivity: do the isolated pieces sum to the whole? ===")
    # In error ENERGY, under independent injection the squared relative errors add.
    per_site_e = {r["site"]: r["drift_A"] ** 2 for r in rows}
    total_e = stat("A", "base")["mean"] ** 2
    summed = sum(per_site_e.values())
    print(f"  sum of the 18 isolated squared drifts = {summed:.4f}")
    print(f"  base's own squared drift              = {total_e:.4f}")
    print(f"  ratio (sum / whole)                   = {summed / total_e:.3f}")
    out["additivity"]["site_energy_sum"] = summed
    out["additivity"]["base_energy"] = total_e
    out["additivity"]["ratio"] = summed / total_e
    # The same question one scale down, using each group arm's OWN recorded site set rather
    # than a membership list restated here: a group of k sites measured together against the
    # sum of those k sites measured alone.
    arms_cfg = json.loads((Path(args.cell_a) / "arms.json").read_text(encoding="utf-8"))
    for name in sorted(groups):
        g = groups[name]
        if "onlyG" not in g:
            continue
        members = arms_cfg[g["onlyG"]]
        member_sum = sum(per_site_e[s] for s in members if s in per_site_e)
        whole = stat("A", g["onlyG"])["mean"] ** 2
        out["additivity"].setdefault("groups", {})[name] = {
            "members": members, "member_energy_sum": member_sum, "group_energy": whole,
            "ratio": member_sum / whole if whole else float("nan")}
        print(f"  {name:20s} {len(members):2d} sites: summed {member_sum:.4f} vs measured "
              f"together {whole:.4f}  (ratio {member_sum / whole if whole else float('nan'):.3f})")

    print("\n=== does the drift metric order the arms the same way recall@1 does? ===")
    # The drift figure is the high-resolution secondary metric and recall@1 is the binding
    # one. A rank agreement between them over all the arms is what licenses reading a drift
    # ordering where recall@1's own resolving power cannot separate two arms; a disagreement
    # would mean the drift ranking is about something the oracle does not score.
    from scipy.stats import spearmanr
    common = [a for a in arms if a in grades["A"]["rows"] and a != "null"]
    dv = np.array([stat("A", a)["mean"] for a in common])
    rv = np.array([grades["A"]["rows"][a]["recall@1"] for a in common])
    rho, pval = spearmanr(dv, rv)
    print(f"  Spearman(drift, recall@1) over {len(common)} arms = {rho:+.4f} (p = {pval:.3g})")
    out["drift_vs_recall_spearman"] = {"rho": float(rho), "p": float(pval),
                                       "n_arms": len(common)}

    print("\n=== per-layer profile of the top isolated sites (mean rel L2 by row) ===")
    top = [r["arm"] for r in rows[:5]]
    for arm in top + ["base"]:
        prof = [float(drift("A", arm, row=r).mean()) for r in (1, 5, 10, 15, 20, 25, 28)]
        print(f"  {arm:24s} " + "  ".join(f"L{r}={v:.4f}" for r, v in
                                          zip((1, 5, 10, 15, 20, 25, 28), prof)))
        out.setdefault("layer_profiles", {})[arm] = prof

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
