#!/usr/bin/env python3
"""T-1820 fourth analysis pass -- saturation against rounding, and PHASE I re-aggregated.

PHASE K   the int8 RANGE census. For each committed residual state the engine holds codes
          c_i and one scale q. A channel whose float value needs |f_i|/q > 127 cannot be
          represented at all: the code rails and the channel carries a deficit
          (|f_i|/q - 127)*q that no amount of rounding accuracy would remove. Every other
          channel carries at most q/2 of rounding error. This phase splits the observed error
          energy into the part sitting on railed channels and the part on the rest, and
          reports the share of error energy the railed channels carry.
          DISCRIMINATES: int8 RANGE saturation (a few channels too large for the shared
          scale) from int8 ROUNDING (every channel quantized at q). The two have different
          remedies and only one of them is fixed by a finer step.

PHASE I2  PHASE I re-aggregated. The first run's ||e_k|| and ||d_k|| columns were means over
          all 249 samples, and position 0 carries an error two orders of magnitude larger
          than any other position, so those two columns were position-0's own numbers wearing
          a population label (StandardsDocument.md 5.4, the aggregate species). The per-sample
          cosine was unaffected -- it is a mean of per-sample ratios, not a ratio of means --
          but it is reported here split by population so that can be seen rather than argued.

Read-only on every input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1820_mechanism import Capture  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def phase_k(caps, log, out):
    log("\n=== PHASE K -- int8 range saturation against int8 rounding ===")
    log("railed  = channels whose committed |code| is 127 (the int8 rail this engine uses;")
    log("          the census below reports the observed maximum |code| so that is not assumed).")
    log("over    = channels whose FLOAT value needs |f_i|/q > 127, i.e. cannot be represented")
    log("          on this state's own grid at all.")
    log("E_rail  = share of the state's total squared error carried by the railed channels.")
    log("deficit = sum over railed channels of (|f_i|/q - 127)*q, the part of the error that is")
    log("          range, not rounding -- an irreducible floor at this scale and this width.")
    log("Split by population: position 0 (the massive-activation token) and positions >= 1.")
    maxcode = max(int(np.abs(c.codes.astype(np.int32)).max()) for c in caps)
    log(f"\nobserved maximum |code| over every dumped state, all prompts: {maxcode}")
    res = {}
    for pop, sel in (("pos 0", lambda c: [0]), ("pos >=1", lambda c: list(range(1, c.T)))):
        log(f"\n  population: {pop}")
        log("  state | railed ch | over ch | E_rail share | deficit/||e|| | rel_l2")
        for s in range(caps[0].S):
            nr, no, er, df, rl = [], [], [], [], []
            for c in caps:
                for t in sel(c):
                    q = c.q[t, s]
                    f = c.flt[t, s]
                    e = c.err[t, s]
                    code = np.abs(c.codes[t, s].astype(np.int32))
                    rail = code >= 127
                    over = np.abs(f) / q > 127.0
                    e2 = e**2
                    nr.append(rail.sum())
                    no.append(over.sum())
                    er.append(e2[rail].sum() / e2.sum())
                    d = np.maximum(np.abs(f) / q - 127.0, 0.0) * q
                    df.append(np.linalg.norm(d) / np.linalg.norm(e))
                    rl.append(np.linalg.norm(e) / np.linalg.norm(f))
            res.setdefault(pop, {})[s] = dict(
                railed=float(np.mean(nr)), over=float(np.mean(no)),
                e_rail=float(np.mean(er)), deficit=float(np.mean(df)), rel=float(np.mean(rl)))
            log(f"  {s:5d} | {np.mean(nr):9.2f} | {np.mean(no):7.2f} | {np.mean(er):12.4f} |"
                f" {np.mean(df):13.4f} | {np.mean(rl):6.3f}")
    out["phase_k"] = res
    a = [res["pos >=1"][s]["e_rail"] for s in range(1, caps[0].S)]
    b = [res["pos 0"][s]["e_rail"] for s in range(1, caps[0].S)]
    log(f"\nK-RESULT states 1-28: railed channels carry {np.mean(a)*100:.2f}% of the squared error at")
    log(f"  positions >= 1 (range {min(a)*100:.2f}-{max(a)*100:.2f}%) and {np.mean(b)*100:.1f}% at position 0")
    log(f"  (range {min(b)*100:.1f}-{max(b)*100:.1f}%).")


def phase_i2(caps, log, out):
    log("\n=== PHASE I2 -- accumulation against amplification, re-aggregated ===")
    log("Per-sample quantities, then the median across samples -- never a ratio of means.")
    log("cos = cos(e_k, d_k) with d_k = e_{k+1} - e_k. Zero: the layer injects error orthogonal")
    log("to what it inherited (independent accumulation). Positive: the layer amplifies what it")
    log("inherited. Negative: the layer partly cancels what it inherited.")
    log("energy ratio = ||e_{k+1}||^2 / ||e_k||^2 per sample; > 1 is growth.")
    res = {}
    for pop, sel in (("pos 0", lambda c: [0]), ("pos >=1", lambda c: list(range(1, c.T)))):
        log(f"\n  population: {pop}")
        log("  layer | median ||d||/||e|| | median cos(e,d) +/- SD | median energy ratio")
        for k in range(caps[0].S - 1):
            r, cs, en = [], [], []
            for c in caps:
                for t in sel(c):
                    e0, e1 = c.err[t, k], c.err[t, k + 1]
                    d = e1 - e0
                    n0, nd = np.linalg.norm(e0), np.linalg.norm(d)
                    r.append(nd / n0)
                    cs.append(float(e0 @ d) / (n0 * nd))
                    en.append((np.linalg.norm(e1) / n0) ** 2)
            cs = np.array(cs)
            res.setdefault(pop, {})[k] = dict(ratio=float(np.median(r)), cos=float(np.median(cs)),
                                              cos_sd=float(cs.std()), energy=float(np.median(en)))
            log(f"  {k:5d} | {np.median(r):18.3f} | {np.median(cs):+16.3f} +/-{cs.std():.3f} |"
                f" {np.median(en):19.3f}")
    out["phase_i2"] = res
    c = [res["pos >=1"][k]["cos"] for k in range(1, caps[0].S - 1)]
    e = [res["pos >=1"][k]["energy"] for k in range(1, caps[0].S - 1)]
    log(f"\nI2-RESULT layers 1-27, positions >= 1: median cos(e_k,d_k) mean {np.mean(c):+.3f}, "
        f"range {min(c):+.3f}..{max(c):+.3f}; median per-layer error-energy ratio mean "
        f"{np.mean(e):.3f}, range {min(e):.3f}..{max(e):.3f}.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default="out/t1820")
    ap.add_argument("--prompts", default="p1,p2,p3,q1,q2,q3")
    ap.add_argument("--json", default="out/t1820/mechanism4.json")
    args = ap.parse_args(argv)
    caps = [Capture(p, Path(args.dump_dir)) for p in args.prompts.split(",")]

    def log(s=""):
        print(s, flush=True)

    out = {}
    phase_k(caps, log, out)
    phase_i2(caps, log, out)
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="ascii")
    log(f"\njson written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
