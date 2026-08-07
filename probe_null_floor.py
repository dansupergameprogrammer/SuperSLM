#!/usr/bin/env python3
"""Popper probe 1 -- what does the Spearman instrument read on inputs that
share NO content? Calibrates the T-1690 headline scale against a
mismatched-prompt null: spearman(int8 codes of prompt X, witness float of
prompt Y) for X != Y, same layer.

Reads only the already-written dumps in the T-1690 worktree. Writes nothing
there."""
import struct
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DUMPS = Path(r"D:\SuperSLM\.worktrees\t1690-third-witness\out\t1690")

MECH2 = ["digit_symbol_spaced", "digit_symbol_nospace", "plain_language_plus", "digit_symbol_mult"]
CONTROL5 = ["capital_of_germany", "capital_of_france", "capital_of_japan", "largest_planet", "days_in_week"]
PROMPTS = MECH2 + CONTROL5
LAYERS = list(range(21, 29)) + [5, 1]


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
    int8 = {p: load_int8(DUMPS / f"{p}.int8.bin") for p in PROMPTS}
    wit = {}
    for dt in ("bfloat16", "float32"):
        for p in PROMPTS:
            wit[(p, dt)] = load_float(DUMPS / f"{p}.witness_{dt}.bin")

    for dt in ("bfloat16", "float32"):
        print(f"\n=== dtype={dt} ===")
        for layer in LAYERS:
            matched = np.array([rho(int8[p][layer], wit[(p, dt)][layer]) for p in PROMPTS])
            mismatched = np.array(
                [rho(int8[a][layer], wit[(b, dt)][layer]) for a, b in product(PROMPTS, PROMPTS) if a != b]
            )
            m2 = np.array([rho(int8[p][layer], wit[(p, dt)][layer]) for p in MECH2])
            o5 = np.array([rho(int8[p][layer], wit[(p, dt)][layer]) for p in CONTROL5])
            print(
                f"  L{layer:2d}: matched(same prompt) mean={matched.mean():.4f} "
                f"[{matched.min():.4f},{matched.max():.4f}]   "
                f"MISMATCHED(different prompt, n={len(mismatched)}) mean={mismatched.mean():.4f} "
                f"[{mismatched.min():.4f},{mismatched.max():.4f}] p95={np.percentile(mismatched,95):.4f}  "
                f"|| mech2 mean={m2.mean():.4f} other5 mean={o5.mean():.4f} "
                f"gap={o5.min()-m2.max():+.4f}"
            )

    # How much of the "other5 is high" signal is prompt-specific at all?
    print("\n=== per-prompt: matched vs. that prompt's own mismatched mean (float32) ===")
    dt = "float32"
    for layer in (21, 24, 28):
        print(f"  layer {layer}")
        for p in PROMPTS:
            grp = "mech2 " if p in MECH2 else "other5"
            matched = rho(int8[p][layer], wit[(p, dt)][layer])
            mism = np.array([rho(int8[p][layer], wit[(q, dt)][layer]) for q in PROMPTS if q != p])
            print(
                f"    {p:<22}{grp} matched={matched:.4f} mismatched_mean={mism.mean():.4f} "
                f"lift={matched-mism.mean():+.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
