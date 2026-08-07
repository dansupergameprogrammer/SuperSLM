#!/usr/bin/env python3
"""Popper probe 4 -- two attacks on the T-1690 headline.

D. PRECISION ALONE, NO INT8. Grade the witness at bfloat16 against the
   witness at float32 -- the identical source file, identical weights,
   identical composition, one flag different. If the mechanism-2/other-five
   separation appears here, the shape is reproducible with no int8 engine
   and no library composition anywhere in the comparison.
E. RESOLVING POWER OF THE GAP STATISTIC. min(other5) - max(mech2) is an
   extreme-order statistic on n=4 vs n=5. Permute the 4/5 group labels over
   all C(9,4)=126 assignments and report where the campaign's own split
   falls.
"""
import struct
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

WIT = Path(r"D:\SuperSLM\.worktrees\t1690-third-witness\out\t1690")
LIB = Path(r"D:\SuperSLM\.worktrees\popper-t1802\out\popper_t1802")

MECH2 = ["digit_symbol_spaced", "digit_symbol_nospace", "plain_language_plus", "digit_symbol_mult"]
CONTROL5 = ["capital_of_germany", "capital_of_france", "capital_of_japan", "largest_planet", "days_in_week"]
PROMPTS = MECH2 + CONTROL5
LOCUS = list(range(21, 29))


def load_int8(path):
    with open(path, "rb") as f:
        rows, hs, fp, mode = struct.unpack("<QQQQ", f.read(32))
        codes = np.zeros((rows, hs), dtype=np.int8)
        for i in range(rows):
            struct.unpack("<qq", f.read(16))
            codes[i, :] = np.frombuffer(f.read(hs), dtype=np.int8)
    return codes


def load_float(path):
    with open(path, "rb") as f:
        rows, hs, fp, mode = struct.unpack("<QQQQ", f.read(32))
        data = np.frombuffer(f.read(), dtype=np.float32, count=rows * hs)
    return data.reshape(rows, hs)


def rho(a, b):
    r, _ = spearmanr(a.astype(np.float64), b.astype(np.float64))
    return float(r)


def main():
    int8 = {p: load_int8(WIT / f"{p}.int8.bin") for p in PROMPTS}
    lib = {p: load_float(LIB / f"{p}.libfloat.bin") for p in PROMPTS}
    wbf = {p: load_float(WIT / f"{p}.witness_bfloat16.bin") for p in PROMPTS}
    wfp = {p: load_float(WIT / f"{p}.witness_float32.bin") for p in PROMPTS}

    print("=== D. bfloat16 witness graded against float32 witness (same file, one flag) ===")
    print(f"  {'prompt':<22}{'group':<8}" + "".join(f"L{l:<6}" for l in LOCUS))
    vals = {}
    for p in PROMPTS:
        g = "mech2" if p in MECH2 else "other5"
        vals[p] = {l: rho(wbf[p][l], wfp[p][l]) for l in LOCUS}
        print(f"  {p:<22}{g:<8}" + "".join(f"{vals[p][l]:7.4f}" for l in LOCUS))
    print("\n  per-layer gap (min(other5) - max(mech2)) with NO int8 and NO library in the comparison:")
    for l in LOCUS:
        m2 = [vals[p][l] for p in MECH2]
        o5 = [vals[p][l] for p in CONTROL5]
        print(f"    L{l}: mech2 [{min(m2):.4f},{max(m2):.4f}]  other5 [{min(o5):.4f},{max(o5):.4f}]"
              f"  gap={min(o5)-max(m2):+.4f}  mean_diff={np.mean(o5)-np.mean(m2):+.4f}")

    print("\n=== D2. same, library(bf16) graded against witness(fp32) ===")
    for l in LOCUS:
        v = {p: rho(lib[p][l], wfp[p][l]) for p in PROMPTS}
        m2 = [v[p] for p in MECH2]
        o5 = [v[p] for p in CONTROL5]
        print(f"    L{l}: mech2 [{min(m2):.4f},{max(m2):.4f}]  other5 [{min(o5):.4f},{max(o5):.4f}]"
              f"  gap={min(o5)-max(m2):+.4f}")

    print("\n=== E. permutation of the 4/5 split, C(9,4)=126 assignments, int8 vs witness(fp32) ===")
    for l in (21, 24, 28):
        v = {p: rho(int8[p][l], wfp[p][l]) for p in PROMPTS}
        gaps = []
        for grp in combinations(PROMPTS, 4):
            other = [p for p in PROMPTS if p not in grp]
            gaps.append(min(v[p] for p in other) - max(v[p] for p in grp))
        gaps = np.array(gaps)
        observed = min(v[p] for p in CONTROL5) - max(v[p] for p in MECH2)
        n_ge = int((gaps >= observed - 1e-12).sum())
        print(f"  L{l}: observed gap={observed:+.4f}   permutation max={gaps.max():+.4f} "
              f"mean={gaps.mean():+.4f}   #splits >= observed: {n_ge}/126  (p={n_ge/126:.4f})")
        order = np.argsort(-gaps)
        top = [(sorted(list(combinations(PROMPTS, 4))[i]), gaps[i]) for i in order[:3]]
        for grp, g in top:
            print(f"       top split gap={g:+.4f}: {grp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
