#!/usr/bin/env python3
"""Onset/amplification analysis over t1691_fulldepth_sitecomp_drive.py's own
fulldepth_results.tsv -- the same C(9,4)=126 exhaustive permutation test the
D-SLM763/764 arms used (rank of the ACTUAL mechanism-2 grouping among all 126
possible 4-of-9 splits, by control-minus-mech2 gap), applied PER ROW (1..28)
and PER SITE/QUANTITY, rather than only at the four previously-measured rows.

ONSET (per site/quantity): the first row (ascending from 1) at which the
actual mechanism-2 grouping's rank crosses two named thresholds -- rank
1/126 (p=0.0079, the test's own ceiling) and rank <=6/126 (p<=0.0476, the
threshold D-SLM763 itself used to call row 5 "an order of magnitude weaker,
non-disjoint" rather than null). Both are reported; neither is asserted as
"the" onset without the other, per StandardsDocument Sec5.4 (name the
quantity a threshold decides).

AMPLIFICATION: the gap-magnitude curve (control-mean minus mech2-mean),
per row, per site/quantity -- printed in full so where the curve actually
grows can be read off, not asserted from two endpoints.

Usage: python tools\\t1691_fulldepth_permtest.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
PATH = os.path.join(REPO_ROOT, "out", "t1691_fulldepth_sitecomp", "fulldepth_results.tsv")

rows = []
with open(PATH) as f:
    header = f.readline().strip().split("\t")
    for line in f:
        parts = line.rstrip("\n").split("\t")
        d = dict(zip(header, parts))
        rows.append(d)

labels = sorted(set(r["label"] for r in rows))
assert len(labels) == 9, labels
actual_mech2 = frozenset(r["label"] for r in rows if r["role"] == "mech2")
assert len(actual_mech2) == 4, actual_mech2
ALL_SPLITS = list(itertools.combinations(labels, 4))
assert len(ALL_SPLITS) == 126


def perm_test(site: str, quantity: str, row_number: int):
    by_label = {lbl: [] for lbl in labels}
    for r in rows:
        if r["site"] != site or r["quantity"] != quantity or int(r["row"]) != row_number:
            continue
        by_label[r["label"]].append(float(r["value"]))
    if any(len(v) == 0 for v in by_label.values()):
        return None
    label_mean = {lbl: float(np.mean(v)) for lbl, v in by_label.items()}

    def gap_for(group4):
        g = [label_mean[l] for l in group4]
        rest = [label_mean[l] for l in labels if l not in group4]
        return float(np.mean(rest) - np.mean(g))

    gaps = [(frozenset(split), gap_for(frozenset(split))) for split in ALL_SPLITS]
    gaps_sorted = sorted(gaps, key=lambda x: -x[1])
    rank = next(i for i, (grp, g) in enumerate(gaps_sorted, start=1) if grp == actual_mech2)
    actual_gap = dict(gaps)[actual_mech2]
    return rank, len(ALL_SPLITS), actual_gap


SITES_QUANTITIES = []
for site in sorted(set(r["site"] for r in rows)):
    for quantity in sorted(set(r["quantity"] for r in rows if r["site"] == site)):
        SITES_QUANTITIES.append((site, quantity))

ALL_ROWS = sorted(set(int(r["row"]) for r in rows))

print(f"Population: {len(labels)} prompts, mech2={sorted(actual_mech2)}, "
      f"126 possible 4-of-9 splits, permutation floor p=1/126=0.0079.\n")

onset_summary = {}
for site, quantity in SITES_QUANTITIES:
    curve = []
    first_ceiling_row = None   # rank 1/126, p=0.0079
    first_weak_row = None      # rank <=6/126, p<=0.0476
    for row_number in ALL_ROWS:
        result = perm_test(site, quantity, row_number)
        if result is None:
            continue
        rank, total, gap = result
        curve.append((row_number, rank, gap))
        if first_ceiling_row is None and rank == 1:
            first_ceiling_row = row_number
        if first_weak_row is None and rank <= 6:
            first_weak_row = row_number
    onset_summary[(site, quantity)] = (first_weak_row, first_ceiling_row, curve)

print("=== ONSET (first row, ascending, at which the actual mech2 grouping reaches each threshold) ===")
print(f"{'site':18s} {'quantity':16s} {'first row<=6/126 (p<=0.0476)':30s} {'first row=1/126 (p=0.0079)':28s}")
for site, quantity in SITES_QUANTITIES:
    first_weak, first_ceiling, _ = onset_summary[(site, quantity)]
    print(f"{site:18s} {quantity:16s} {str(first_weak):30s} {str(first_ceiling):28s}")

print("\n=== AMPLIFICATION -- gap (control-mean minus mech2-mean) curve per row, Q_full only, all 11 sites ===")
for site, quantity in SITES_QUANTITIES:
    if quantity != "q_full":
        continue
    _, _, curve = onset_summary[(site, quantity)]
    gap_str = " ".join(f"{g:+.3f}" for _row, _rank, g in curve)
    rank_str = " ".join(f"{rk:3d}" for _row, rk, _g in curve)
    print(f"\n{site} (q_full):")
    print(f"  rows {ALL_ROWS[0]}..{ALL_ROWS[-1]}")
    print(f"  gap:  {gap_str}")
    print(f"  rank: {rank_str}")

print("\n=== AMPLIFICATION -- Q_weight_alone vs Q_input_alone gap curves, 5 project_and_funnel sites ===")
for site in sorted(set(s for s, q in SITES_QUANTITIES if q in ("q_weight_alone", "q_input_alone"))):
    for quantity in ("q_weight_alone", "q_input_alone"):
        if (site, quantity) not in onset_summary:
            continue
        _, _, curve = onset_summary[(site, quantity)]
        gap_str = " ".join(f"{g:+.3f}" for _row, _rank, g in curve)
        print(f"{site:18s} {quantity:16s} gap: {gap_str}")
