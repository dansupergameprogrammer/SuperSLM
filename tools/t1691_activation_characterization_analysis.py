#!/usr/bin/env python3
"""T-1691 activation-characterization analysis (Brunel, 2026-08-04).

Reads tools/t1691_activation_characterization_drive.py's own output
(out/t1691_activation_characterization/activation_characterization_results.tsv,
52,668 cells: 9 prompts x 28 rows x 11 sites x (12 int8-path quantities + 7
float-path quantities)) and answers, precisely, at the resolving power this
data actually has:

  (A) GENERAL CURVE -- pooled across all 9 prompts (role-blind): does the
      activation state itself (max/median ratio, code occupancy/entropy,
      kurtosis, residual L2 norm) change shape at rows 20-22, as a property
      of the network's depth alone, independent of which prompt is running?
  (B) GROUP DIFFERENCE -- the same C(9,4)=126 exhaustive permutation test
      t1691_fulldepth_permtest.py used, applied to these NEW quantities: at
      which rows (if any) does the actual mechanism-2/control split rank
      unusually among all 126 possible splits, for max/median ratio, code
      entropy, kurtosis, n_outlier_channels? This is the CAUSE-vs-CORRELATE
      test: (A) alone cannot distinguish "this state change happens to
      every prompt, coincident with the timing" from "this state change is
      specifically what separates mechanism-2 prompts from controls."
  (C) CHANNEL PERSISTENCE -- across all (row, prompt) cells at a site, how
      concentrated is the argmax-channel distribution, and does the winning
      channel set change between the reversal band (rows 12-20) and the
      locus band (rows 21-28)?

Usage: python tools\\t1691_activation_characterization_analysis.py
"""
from __future__ import annotations

import itertools
import math
import os
from collections import Counter, defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
PATH = os.path.join(REPO_ROOT, "out", "t1691_activation_characterization",
                     "activation_characterization_results.tsv")
LOG_PATH = os.path.join(REPO_ROOT, "out", "t1691_activation_characterization",
                         "analysis_stdout.log")

rows = []
with open(PATH) as f:
    header = f.readline().strip().split("\t")
    for line in f:
        parts = line.rstrip("\n").split("\t")
        d = dict(zip(header, parts))
        d["row"] = int(d["row"])
        v = d["value"]
        d["value"] = float(v)
        rows.append(d)

labels = sorted(set(r["label"] for r in rows))
assert len(labels) == 9, labels
role_of = {r["label"]: r["role"] for r in rows}
actual_mech2 = frozenset(l for l in labels if role_of[l] == "mech2")
assert len(actual_mech2) == 4, actual_mech2
ALL_SPLITS = list(itertools.combinations(labels, 4))
assert len(ALL_SPLITS) == 126
ALL_ROWS = sorted(set(r["row"] for r in rows))
assert ALL_ROWS == list(range(1, 29)), ALL_ROWS
ALL_SITES = sorted(set(r["site"] for r in rows))
assert len(ALL_SITES) == 11, ALL_SITES

out_lines = []


def emit(s: str = ""):
    out_lines.append(s)
    print(s)


# Reversal band and locus band, named per D-SLM765/766 (not re-derived here,
# reused as the campaign's own established row bands).
REVERSAL_BAND = list(range(12, 21))  # 12..20
EARLY_BAND = list(range(1, 12))      # 1..11
LOCUS_BAND = list(range(21, 29))     # 21..28


def by_cell(path: str, quantity: str, site: str):
    """{(label, row) -> value}"""
    out = {}
    for r in rows:
        if r["path"] == path and r["quantity"] == quantity and r["site"] == site:
            out[(r["label"], r["row"])] = r["value"]
    return out


def perm_test(cellmap, row_number: int):
    """Same exhaustive C(9,4)=126 test as t1691_fulldepth_permtest.py, generalized
    to ANY per-(label,row) scalar quantity, not only q_full. Gap defined as
    (mean of the OTHER 5) - (mean of the named 4) -- i.e. positive gap means the
    named-4 group's mean is LOWER than the rest, matching the original tool's
    own convention (control_mean - mech2_mean when the named group is mech2)."""
    label_val = {lbl: cellmap.get((lbl, row_number)) for lbl in labels}
    if any(v is None for v in label_val.values()):
        return None

    def gap_for(group4):
        g = [label_val[l] for l in group4]
        rest = [label_val[l] for l in labels if l not in group4]
        return float(np.mean(rest) - np.mean(g))

    gaps = [(frozenset(split), gap_for(frozenset(split))) for split in ALL_SPLITS]
    gaps_sorted = sorted(gaps, key=lambda x: -x[1])
    rank_hi = next(i for i, (grp, g) in enumerate(gaps_sorted, start=1) if grp == actual_mech2)
    gaps_sorted_lo = sorted(gaps, key=lambda x: x[1])
    rank_lo = next(i for i, (grp, g) in enumerate(gaps_sorted_lo, start=1) if grp == actual_mech2)
    actual_gap = dict(gaps)[actual_mech2]
    return rank_hi, rank_lo, len(ALL_SPLITS), actual_gap


# =========================================================================
# (A) GENERAL CURVE -- role-blind, pooled mean across all 9 prompts, per row
# =========================================================================
emit("=" * 100)
emit("(A) GENERAL CURVE -- pooled mean across all 9 prompts (role-blind), per row")
emit("Tests whether the activation state itself changes shape at rows 20-22 as a")
emit("depth property, independent of mechanism-2 vs control classification.")
emit("=" * 100)

GENERAL_QUANTITIES = [
    ("int8", "max_median_ratio"),
    ("int8", "n_distinct_codes"),
    ("int8", "code_entropy_bits"),
    ("int8", "kurtosis"),
    ("int8", "n_outlier_channels"),
    ("float", "max_median_ratio"),
    ("float", "kurtosis"),
    ("float", "l2_norm"),
    ("float", "n_outlier_channels"),
]

REPRESENTATIVE_SITES = ["mlp_residual", "down_proj.requant", "attn_ctx", "mlp_act"]

for path, quantity in GENERAL_QUANTITIES:
    for site in REPRESENTATIVE_SITES:
        cellmap = by_cell(path, quantity, site)
        curve = []
        for row_number in ALL_ROWS:
            vals = [cellmap.get((lbl, row_number)) for lbl in labels]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            curve.append(float(np.mean(vals)) if vals else float("nan"))
        early_mean = np.nanmean(curve[0:11])
        reversal_mean = np.nanmean(curve[11:20])
        locus_mean = np.nanmean(curve[20:28])
        emit(f"\n{path:5s} {quantity:20s} site={site}:")
        emit(f"  band means: rows1-11={early_mean:.4f}  rows12-20={reversal_mean:.4f}  rows21-28={locus_mean:.4f}")
        emit("  row: " + " ".join(f"{r:5d}" for r in ALL_ROWS))
        emit("  val: " + " ".join(f"{v:5.2f}" if math.isfinite(v) else "  nan" for v in curve))

# =========================================================================
# (B) GROUP DIFFERENCE -- exhaustive permutation test, per row, at every site
# =========================================================================
emit("\n" + "=" * 100)
emit("(B) GROUP DIFFERENCE -- C(9,4)=126 exhaustive permutation test (rank of the")
emit("actual mechanism-2 grouping among all 126 possible 4-of-9 splits), applied to")
emit("the NEW activation-state quantities. rank_hi = rank by (rest-mean minus mech2-")
emit("mean), i.e. mech2 group LOWER; rank_lo = rank by the reverse (mech2 HIGHER).")
emit("Both floors are p=1/126=0.0079. Reports first row (ascending) reaching each")
emit("ceiling, for every site, at the quantities the commission names explicitly.")
emit("=" * 100)

TEST_QUANTITIES = [
    ("int8", "max_median_ratio"),
    ("int8", "n_distinct_codes"),
    ("int8", "code_entropy_bits"),
    ("int8", "kurtosis"),
    ("int8", "n_outlier_channels"),
    ("int8", "d_prime_engine"),
    ("float", "max_median_ratio"),
    ("float", "kurtosis"),
    ("float", "l2_norm"),
    ("float", "n_outlier_channels"),
]

onset_hits = defaultdict(list)  # (path,quantity) -> [(site, first_ceiling_row_hi, first_ceiling_row_lo)]

for path, quantity in TEST_QUANTITIES:
    emit(f"\n--- {path} / {quantity} ---")
    for site in ALL_SITES:
        cellmap = by_cell(path, quantity, site)
        first_hi = None
        first_lo = None
        row20_21_22 = {}
        for row_number in ALL_ROWS:
            result = perm_test(cellmap, row_number)
            if result is None:
                continue
            rank_hi, rank_lo, total, gap = result
            if first_hi is None and rank_hi == 1:
                first_hi = row_number
            if first_lo is None and rank_lo == 1:
                first_lo = row_number
            if row_number in (19, 20, 21, 22, 23):
                row20_21_22[row_number] = (rank_hi, rank_lo, gap)
        onset_hits[(path, quantity)].append((site, first_hi, first_lo))
        detail = " ".join(f"r{rn}:hi={h}/lo={l}(gap={g:+.3f})" for rn, (h, l, g) in sorted(row20_21_22.items()))
        emit(f"  {site:20s} first_rank1_hi(mech2-lower)={str(first_hi):>4s}  "
             f"first_rank1_lo(mech2-higher)={str(first_lo):>4s}  |  rows19-23: {detail}")

# =========================================================================
# (C) CHANNEL PERSISTENCE -- argmax-channel concentration, by band
# =========================================================================
emit("\n" + "=" * 100)
emit("(C) CHANNEL PERSISTENCE -- argmax-channel (per row,prompt cell) distribution,")
emit("split by band (early 1-11 / reversal 12-20 / locus 21-28), for the int8 path's")
emit("own wide-row argmax (the channel holding d_prime, i.e. the channel that sets")
emit("the row's own dynamic quantization scale) at representative sites.")
emit("=" * 100)

for site in REPRESENTATIVE_SITES:
    cellmap = by_cell("int8", "argmax_channel", site)
    emit(f"\nsite={site}:")
    for band_name, band_rows in (("early(1-11)", EARLY_BAND), ("reversal(12-20)", REVERSAL_BAND),
                                  ("locus(21-28)", LOCUS_BAND)):
        chans = [int(cellmap[(lbl, r)]) for lbl in labels for r in band_rows if (lbl, r) in cellmap]
        counts = Counter(chans)
        top5 = counts.most_common(5)
        n_distinct_channels = len(counts)
        n_total = len(chans)
        top1_frac = top5[0][1] / n_total if top5 else 0.0
        emit(f"  {band_name:16s}: {n_total} (row,prompt) cells, {n_distinct_channels} distinct argmax "
             f"channels seen, top channel={top5[0] if top5 else None} ({top1_frac:.1%} of cells), "
             f"top5={top5}")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
with open(LOG_PATH, "w") as f:
    f.write("\n".join(out_lines) + "\n")
print(f"\nfull log written to {LOG_PATH}")
