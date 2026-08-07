#!/usr/bin/env python3
"""Popper probe 3 -- three questions the T-1690 packet does not answer:

A. What is the LIBRARY arm's own mech2/other5 gap on this machine today?
   (The packet's headline claims its float32 witness gap of +0.2098 at L28
   "matches a previously recorded gap of 0.2134" measured on the library
   arm. That prior number is re-derived here rather than cited.)
B. Does the layers-1-5 validation gate discriminate? Calibrated against a
   cross-prompt null: spearman(witness of prompt X, library of prompt Y).
C. Witness-vs-library agreement at the LOCUS layers 21-28 (the packet
   validates only at 1-5).
"""
import struct
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

WIT = Path(r"D:\SuperSLM\.worktrees\t1690-third-witness\out\t1690")
LIB = Path(r"D:\SuperSLM\.worktrees\popper-t1802\out\popper_t1802")

MECH2 = ["digit_symbol_spaced", "digit_symbol_nospace", "plain_language_plus", "digit_symbol_mult"]
CONTROL5 = ["capital_of_germany", "capital_of_france", "capital_of_japan", "largest_planet", "days_in_week"]
PROMPTS = MECH2 + CONTROL5
LOCUS = list(range(21, 29))
VALID = [1, 2, 3, 4, 5]


def load_int8(path):
    with open(path, "rb") as f:
        rows, hs, fp, mode = struct.unpack("<QQQQ", f.read(32))
        codes = np.zeros((rows, hs), dtype=np.int8)
        for i in range(rows):
            struct.unpack("<qq", f.read(16))
            codes[i, :] = np.frombuffer(f.read(hs), dtype=np.int8)
    return codes, fp


def load_float(path):
    with open(path, "rb") as f:
        rows, hs, fp, mode = struct.unpack("<QQQQ", f.read(32))
        data = np.frombuffer(f.read(), dtype=np.float32, count=rows * hs)
    return data.reshape(rows, hs), fp


def rho(a, b):
    r, _ = spearmanr(a.astype(np.float64), b.astype(np.float64))
    return float(r)


def gap(vals):
    return min(vals[p] for p in CONTROL5) - max(vals[p] for p in MECH2)


def main():
    int8, ifp = {}, {}
    for p in PROMPTS:
        int8[p], ifp[p] = load_int8(WIT / f"{p}.int8.bin")
    lib, lfp = {}, {}
    for p in PROMPTS:
        lib[p], lfp[p] = load_float(LIB / f"{p}.libfloat.bin")
    wit = {}
    for dt in ("bfloat16", "float32"):
        for p in PROMPTS:
            wit[(p, dt)], wfp = load_float(WIT / f"{p}.witness_{dt}.bin")
            assert wfp == ifp[p] == lfp[p], f"fingerprint mismatch {p} {dt}"
    print("provenance: all three arms agree on prompt fingerprint for all nine prompts\n")

    print("=== A. per-layer gap (min(other5) - max(mech2)), three float arms vs the SAME int8 dump ===")
    print(f"  {'layer':<7}{'library(bf16)':>15}{'witness(bf16)':>15}{'witness(fp32)':>15}")
    arms = {}
    for layer in LOCUS + [5]:
        row = {}
        row["library(bf16)"] = {p: rho(int8[p][layer], lib[p][layer]) for p in PROMPTS}
        for dt, name in (("bfloat16", "witness(bf16)"), ("float32", "witness(fp32)")):
            row[name] = {p: rho(int8[p][layer], wit[(p, dt)][layer]) for p in PROMPTS}
        arms[layer] = row
        print(f"  L{layer:<6}" + "".join(f"{gap(row[k]):+15.4f}" for k in
                                          ("library(bf16)", "witness(bf16)", "witness(fp32)")))

    print("\n  per-prompt values at L28:")
    print(f"    {'prompt':<22}{'group':<8}{'library(bf16)':>15}{'witness(bf16)':>15}{'witness(fp32)':>15}")
    for p in PROMPTS:
        g = "mech2" if p in MECH2 else "other5"
        print(f"    {p:<22}{g:<8}" + "".join(f"{arms[28][k][p]:15.4f}" for k in
                                             ("library(bf16)", "witness(bf16)", "witness(fp32)")))

    print("\n=== B. validation gate at layers 1-5: matched vs. CROSS-PROMPT null ===")
    for dt in ("bfloat16", "float32"):
        print(f"  dtype={dt}")
        for layer in VALID:
            matched = np.array([rho(wit[(p, dt)][layer], lib[p][layer]) for p in PROMPTS])
            null = np.array([rho(wit[(a, dt)][layer], lib[b][layer])
                             for a, b in product(PROMPTS, PROMPTS) if a != b])
            print(f"    L{layer}: matched mean={matched.mean():.4f} [{matched.min():.4f},{matched.max():.4f}]"
                  f"   cross-prompt NULL mean={null.mean():.4f} [{null.min():.4f},{null.max():.4f}]"
                  f" max={null.max():.4f}"
                  f"   separation(min matched - max null)={matched.min()-null.max():+.4f}")

    print("\n=== C. witness vs. library at the LOCUS layers (never validated by the packet) ===")
    for dt in ("bfloat16", "float32"):
        print(f"  dtype={dt}")
        for layer in LOCUS:
            m2 = np.array([rho(wit[(p, dt)][layer], lib[p][layer]) for p in MECH2])
            o5 = np.array([rho(wit[(p, dt)][layer], lib[p][layer]) for p in CONTROL5])
            null = np.array([rho(wit[(a, dt)][layer], lib[b][layer])
                             for a, b in product(PROMPTS, PROMPTS) if a != b])
            print(f"    L{layer}: mech2 [{m2.min():.4f},{m2.max():.4f}] other5 [{o5.min():.4f},{o5.max():.4f}]"
                  f"  gap={o5.min()-m2.max():+.4f}   cross-prompt null max={null.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
