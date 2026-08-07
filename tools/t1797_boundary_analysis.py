#!/usr/bin/env python3
"""T-1797: boundary geometry and downstream code flips of the residual-rotation construction.

Boundary distance (T-1796's convention, half-away-from-zero geometry): for each element of
a captured pre-quantization row, the quotient y = x * 127 * R / 2^(62 - s - k) is formed
EXACTLY (python integers / Fraction), and its distance to the nearest rounding boundary
(half-integers of y) is reported in [0, 0.5]: 0 = on the boundary, 0.5 = bin center.
Uniform null: mean 0.25, %within-eps ~= 2*eps.

Sites: the k0 arm's boundary rows ARE production's residual-site geometry (k0 is dual-run
bit-verified against production). The rot arm captures both its fine16 stage and the
rot8 stage (the int8 quantization of the rotated row -- the construction's own storage).

Flips: last-position site codes (norm_attn / o_proj / down_proj int8; probs Q15) compared
arm vs k0, per layer pooled.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

OUT = Path("out/t1797arms")
PROMPTS = range(1, 7)


def boundary_stats(path: Path, want_suffix: str):
    """Returns per-element boundary distances pooled over rows whose site ends with suffix."""
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
                # distance to nearest integer (bin center), in [0, 0.5]
                d_center = min(fr, 1 - fr)
                dists.append(0.5 - float(d_center))
    return np.asarray(dists)


def main() -> int:
    print("== boundary geometry (pooled 28 layers x 2 residual sites, 6 prompts) ==")
    for label, arm, suffix in [
        ("production residual sites (k0)", "k0", "_residual"),
        ("rot: fine16 stage", "rot", ".fine16"),
        ("rot: rotated int8 store", "rot", ".rot8"),
    ]:
        alld = []
        for p in PROMPTS:
            alld.append(boundary_stats(OUT / f"p{p}_{arm}_boundary.txt", suffix))
        per_prompt = [(d.mean(), (d < 0.01).mean(), (d < 0.05).mean(), (d < 0.10).mean())
                      for d in alld]
        pp = np.asarray(per_prompt)
        d = np.concatenate(alld)
        print(f"{label}: n={d.size}")
        print(f"  mean dist {d.mean():.4f}  within1% {100*(d<0.01).mean():.2f}%  "
              f"within5% {100*(d<0.05).mean():.2f}%  within10% {100*(d<0.10).mean():.2f}%")
        print(f"  per-prompt spread (sd over 6): mean {pp[:,0].std():.4f}, "
              f"w5% {100*pp[:,2].std():.2f}pp")

    print()
    print("== downstream code flips at the last prompt token (arm vs k0), 6 prompts ==")
    for arm in ["k4", "k8", "rot"]:
        stats = {}
        for p in PROMPTS:
            ref = (OUT / f"p{p}_k0_sitecodes.txt").read_text().splitlines()
            got = (OUT / f"p{p}_{arm}_sitecodes.txt").read_text().splitlines()
            for lr, lg in zip(ref, got):
                pr, pg = lr.split(), lg.split()
                name = pr[0]
                a = np.asarray([int(v) for v in pr[2:]])
                b = np.asarray([int(v) for v in pg[2:]])
                assert a.size == b.size, (name, a.size, b.size)
                flips = (a != b).mean()
                mad = np.abs(a - b).mean()
                stats.setdefault(name, []).append((flips, mad))
        row = f"{arm}: "
        for name, v in stats.items():
            v = np.asarray(v)
            row += f"{name} flips {100*v[:,0].mean():.1f}%+-{100*v[:,0].std():.1f} " \
                   f"|d| {v[:,1].mean():.2f}; "
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
