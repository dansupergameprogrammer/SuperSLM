#!/usr/bin/env python3
"""T-1799: boundary geometry of the widened-accumulate-only candidate's residual sites,
against T-1796's production baseline and T-1797/T-1798's rotated-store baselines.

Convention identical to tools/t1797_boundary_analysis.py (T-1796's half-away-from-zero
distance-to-boundary metric, exact rational arithmetic): for each element of a captured
pre-quantization row, y = x * 127 * R / 2^(62 - s - k) is formed exactly, and its distance
to the nearest rounding boundary is reported in [0, 0.5] (0 = on the boundary, 0.5 = bin
center). Uniform null: mean 0.25, %within-eps ~= 2*eps.

Population here: 9 held-out prompts (p7-p15), last position, 28 layers x 2 residual sites
each (production k0 and the k8 candidate both captured by this ticket's own runs).
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

OUT = Path("out/t1799arms")
PROMPTS = range(7, 16)


def boundary_stats(path: Path, want_suffix: str):
    dists = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            site = parts[0]
            if not site.endswith(want_suffix):
                continue
            r = int(parts[2]); s = int(parts[4]); k = int(parts[6]); n = int(parts[8])
            xs = [int(v) for v in parts[9:9 + n]]
            e = 62 - s - k
            denom = 1 << e
            num_scale = 127 * r
            for x in xs:
                y = Fraction(abs(x) * num_scale, denom)
                fr = y - y.__floor__()
                d_center = min(fr, 1 - fr)
                dists.append(0.5 - float(d_center))
    return np.asarray(dists)


def main() -> int:
    print("== T-1799 boundary geometry (pooled 28 layers x 2 residual sites, 9 held-out prompts) ==")
    for label, arm in [("production residual sites (k0)", "k0"),
                        ("k8 candidate residual sites (widened accumulate)", "k8"),
                        ("k4 candidate residual sites", "k4")]:
        alld = []
        for p in PROMPTS:
            alld.append(boundary_stats(OUT / f"p{p}_{arm}_boundary.txt", "_residual"))
        per_prompt = [(d.mean(), (d < 0.01).mean(), (d < 0.05).mean(), (d < 0.10).mean())
                      for d in alld]
        pp = np.asarray(per_prompt)
        d = np.concatenate(alld)
        print(f"{label}: n={d.size}")
        print(f"  mean dist {d.mean():.4f}  within1% {100*(d<0.01).mean():.2f}%  "
              f"within5% {100*(d<0.05).mean():.2f}%  within10% {100*(d<0.10).mean():.2f}%")
        print(f"  per-prompt spread (sd over 9): mean {pp[:,0].std():.4f}, "
              f"w5% {100*pp[:,2].std():.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
