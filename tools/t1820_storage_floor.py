#!/usr/bin/env python3
"""T-1820 probe 1 -- the residual stream's own int8 STORAGE FLOOR, per state.

Reads T-1795's own dumps read-only (engine `p*_residual.txt`, float `p*_float.txt`,
29 states each, last prompt token) and computes, per state:

  q          the engine's own committed quantization step for that state, recovered from
             the dump itself (every value in a committed residual row is an integer
             multiple of one CarriedScale -- one scale per row, 1536 channels)
  code_max   the largest |code| the engine actually used at that state (int8 headroom used)
  floor      ||clamp(round(f/q),-128,127)*q - f|| / ||f||   -- the relative error an
             OTHERWISE-PERFECT engine would still incur purely by storing the float
             reference on that state's own grid at that state's own step
  floor_opt  the same with q chosen optimally for int8 (q = max|f|/127)
  observed   ||engine - float|| / ||f||, the measured relative L2

Discrimination claim: `floor` separates "the drift IS the residual stream's own int8
storage quantization" (floor ~ observed) from "the drift is accumulated upstream and
merely lands in that storage" (floor << observed). It does NOT separate a rotation from
dense noise, and no claim about either is made from it.

Read-only on every input. Writes nothing but its own stdout.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def read_vecs(path: Path) -> list[list[float]]:
    out = []
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            n = int(parts[0])
            vals = [float(x) for x in parts[1 : 1 + n]]
            if len(vals) != n:
                raise SystemExit(f"{path}: row declares {n} values, carries {len(vals)}")
            out.append(vals)
    return out


def recover_step(v: list[float]) -> tuple[float, float, int]:
    """Recover the row's own quantization step from the row itself.

    Every value is code*q with an integer code, so the step is the greatest common
    divisor of the values. Recovered as: the smallest positive magnitude present, then
    refined downward by any value whose ratio to it is not near-integer. Returns
    (q, worst_integrality_residual, max_abs_code).
    """
    mags = sorted({abs(x) for x in v if x != 0.0})
    if not mags:
        return 0.0, 0.0, 0
    # The dumps carry 6 significant digits, so integrality is checked at a tolerance scaled
    # to the largest magnitude in the row rather than at machine epsilon.
    tol = 1e-5 * mags[-1]

    def worst_residual(step: float) -> float:
        w = 0.0
        for m in mags:
            r = m / step
            w = max(w, abs(r - round(r)) * step)
        return w

    # The smallest magnitude present is an integer multiple of the true step; try each
    # divisor and keep the LARGEST step that makes every value integral.
    best = None
    for j in range(1, 33):
        q = mags[0] / j
        if worst_residual(q) <= tol:
            best = q
            break
    if best is None:
        best = mags[0] / 32
    return best, worst_residual(best) / best, int(round(mags[-1] / best))


def l2(v) -> float:
    return math.sqrt(sum(x * x for x in v))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True, help="directory holding p*_residual.txt and p*_float.txt")
    ap.add_argument("--prompts", default="p1,p2,p3")
    args = ap.parse_args(argv)

    d = Path(args.dump_dir)
    prompts = args.prompts.split(",")

    rows = {}
    for p in prompts:
        eng = read_vecs(d / f"{p}_residual.txt")
        flt = read_vecs(d / f"{p}_float.txt")
        if len(eng) != len(flt):
            raise SystemExit(f"{p}: engine has {len(eng)} states, float has {len(flt)}")
        for k, (e, f) in enumerate(zip(eng, flt)):
            q, integ, cmax = recover_step(e)
            nf = l2(f)
            obs = l2([a - b for a, b in zip(e, f)]) / nf
            # storage floor at the engine's OWN step
            fl = []
            for x in f:
                c = max(-128, min(127, round(x / q)))
                fl.append(c * q - x)
            floor = l2(fl) / nf
            # storage floor at the int8-optimal step for this vector
            qo = max(abs(x) for x in f) / 127.0
            flo = []
            for x in f:
                c = max(-128, min(127, round(x / qo)))
                flo.append(c * qo - x)
            floor_opt = l2(flo) / nf
            rows.setdefault(k, []).append(
                dict(q=q, integ=integ, cmax=cmax, obs=obs, floor=floor, floor_opt=floor_opt,
                     nf=nf, maxf=max(abs(x) for x in f))
            )

    def mean(xs):
        return sum(xs) / len(xs)

    print(f"prompts: {prompts}   states: {len(rows)}   (mean over prompts)")
    print("state |    q      code_max | observed  floor   floor_opt | obs/floor | max|f|/||f||")
    worst_integ = 0.0
    for k in sorted(rows):
        r = rows[k]
        worst_integ = max(worst_integ, max(x["integ"] for x in r))
        o, fl, flo = mean([x["obs"] for x in r]), mean([x["floor"] for x in r]), mean([x["floor_opt"] for x in r])
        print(
            f"{k:5d} | {mean([x['q'] for x in r]):.6g} {mean([x['cmax'] for x in r]):7.1f} |"
            f" {o:8.4f} {fl:7.4f} {flo:9.4f} | {o/fl:9.2f} | {mean([x['maxf']/x['nf'] for x in r]):.4f}"
        )
    print(f"\nstep-recovery integrality residual (worst over all rows): {worst_integ:.3e}"
          f"  [0 => every dumped value is exactly an integer multiple of the recovered step]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
