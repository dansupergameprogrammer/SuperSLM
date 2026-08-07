#!/usr/bin/env python3
"""T-1797: grade the residual arms (prod/k4/k8/rot) against the float32 reference at the
last prompt token -- the fidelity tables in the solve record (D-SLM1384/D-SLM1385).

Inputs: out/t1797/p{i}_float.bin and p{i}_eng.bin (t1797_multipos_probe /
t1797_float_multipos), out/t1797arms/p{i}_{arm}.bin (t1797_residual_arms), i in 1..6.
"""

from __future__ import annotations

import sys

import numpy as np


def load(path):
    return np.fromfile(path, dtype=np.float32).astype(np.float64).reshape(-1, 29, 1536)


def main() -> int:
    states_report = list(range(29))
    arms = ["eng", "k4", "k8", "rot"]
    agg = {a: {s: [] for s in states_report} for a in arms}
    for i in range(1, 7):
        flt = load(f"out/t1797/p{i}_float.bin")
        for arm in arms:
            path = (f"out/t1797/p{i}_eng.bin" if arm == "eng"
                    else f"out/t1797arms/p{i}_{arm}.bin")
            e = load(path)
            assert e.shape == flt.shape, (i, arm, e.shape, flt.shape)
            t = e.shape[0] - 1  # last prompt token, T-1795's own cell
            for s in states_report:
                d = e[t, s] - flt[t, s]
                rel = np.linalg.norm(d) / np.linalg.norm(flt[t, s])
                cos = (e[t, s] @ flt[t, s] /
                       (np.linalg.norm(e[t, s]) * np.linalg.norm(flt[t, s])))
                agg[arm][s].append((rel, cos))
    print(f"{'st':>3}" + "".join(f" | {a:>6}: relL2+-sd cos" for a in arms))
    for s in states_report:
        row = f"{s:>3}"
        for arm in arms:
            v = np.array(agg[arm][s])
            row += f" | {v[:,0].mean():.3f}+-{v[:,0].std():.3f} {v[:,1].mean():.4f}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
