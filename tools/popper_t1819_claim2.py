#!/usr/bin/env python3
"""T-1819 attack on claim 2: "the rotation is concentrated at layer 0 and the final layer".

T-1795's concentration finding is read off rel_l2 = ||engine - float|| / ||float||, whose
denominator grows ~200x across the stack.  Two things are computed here that T-1795 did not:

  1. the same table in ABSOLUTE terms, with the per-boundary INCREMENT of absolute error
     and each boundary's share of the final error, so "concentration" can be read in the
     metric where the layers are commensurable;
  2. the final-layer spike decomposed into numerator and denominator: rel_l2 at state 28
     is recomputed against state 27's own float norm, isolating how much of the jump is
     the reference's own norm contracting rather than the error growing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_states(path: Path) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            n = int(p[0])
            rows.append(np.array([float(x) for x in p[1:1 + n]], dtype=np.float64))
    return np.stack(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", nargs="+", required=True)
    ap.add_argument("--float", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    E = [read_states(Path(p)) for p in a.engine]
    F = [read_states(Path(p)) for p in a.float]
    nP = len(E)
    ae = np.zeros((nP, 29)); fn = np.zeros((nP, 29)); rl = np.zeros((nP, 29))
    for p in range(nP):
        for k in range(29):
            ae[p, k] = np.linalg.norm(E[p][k] - F[p][k])
            fn[p, k] = np.linalg.norm(F[p][k])
            rl[p, k] = ae[p, k] / fn[p, k]
    A, N, R = ae.mean(0), fn.mean(0), rl.mean(0)
    tot = A[28]

    print(f"{'st':>3} {'rel_l2':>8} {'abs_err':>9} {'float_nrm':>10} {'d_abs':>9} "
          f"{'%of final':>10} {'d_rel':>9}")
    rows = []
    for k in range(29):
        d = A[k] - (A[k - 1] if k else 0.0)
        dr = R[k] - (R[k - 1] if k else 0.0)
        rows.append(dict(state=k, rel_l2=R[k], abs_err=A[k], float_norm=N[k],
                         d_abs=d, share_of_final=d / tot, d_rel=dr))
        print(f"{k:>3} {R[k]:>8.4f} {A[k]:>9.3f} {N[k]:>10.2f} {d:>9.3f} "
              f"{100*d/tot:>9.2f}% {dr:>9.4f}")

    rank_abs = sorted(((r["d_abs"], r["state"]) for r in rows), reverse=True)[:8]
    rank_rel = sorted(((r["d_rel"], r["state"]) for r in rows), reverse=True)[:8]
    print("\ntop-8 boundaries by ABSOLUTE increment:", [(s, round(v, 2)) for v, s in rank_abs])
    print("top-8 boundaries by RELATIVE increment:", [(s, round(v, 4)) for v, s in rank_rel])
    print(f"\nlayer 0 share of final absolute error: {100*A[1]/tot:.2f}%")
    print(f"layers 22-27 share of final absolute error: {100*(A[28]-A[22])/tot:.2f}%")
    neg = [r["state"] for r in rows if r["d_abs"] < 0]
    print(f"boundaries where absolute error DECREASES: {neg}")
    print(f"\nfloat norm 27 -> 28: {N[27]:.1f} -> {N[28]:.1f} ({100*(N[28]/N[27]-1):+.1f}%)")
    print(f"abs err  27 -> 28: {A[27]:.2f} -> {A[28]:.2f} ({100*(A[28]/A[27]-1):+.1f}%)")
    ctf = A[28] / N[27]
    print(f"rel_l2[28] against state-27 float norm: {ctf:.4f}  (reported {R[28]:.4f})")
    jump = R[28] - R[27]
    print(f"share of the final-layer rel_l2 jump due to the DENOMINATOR contracting: "
          f"{100*(R[28]-ctf)/jump:.1f}%")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print("written:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
