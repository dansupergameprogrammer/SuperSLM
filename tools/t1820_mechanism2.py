#!/usr/bin/env python3
"""T-1820 second analysis pass -- separating what PHASE B could not, and testing the
shared-scale hypothesis the first pass's results point at.

PHASE F   a fixed additive per-channel BIAS against a genuine LINEAR MAP. PHASE B fitted
          e ~ A f with no intercept, so a constant error vector could be absorbed by A
          wherever the float states share a common direction -- which they strongly do.
          Here the two are fitted separately and scored out-of-sample: first the training
          set's own mean error vector alone, then a centred linear map on top of it.
          DISCRIMINATES: a fixed per-channel bias (rounding that is systematically signed,
          a clamp, a stuck code) from a fixed linear operator (a rotation, a diagonal
          scaling, weight-quantization error). It does not separate either from the
          residual unexplained part, which PHASE B already showed dominates.

PHASE G   the position-0 state. Position 0 holds Qwen2.5's massive-activation channels,
          which is where a single shared per-vector quantization scale is most stressed.
          Reports, per state and position: ||f||, max|f|/||f||, the committed step q, the
          share of channels sitting at the int8 rail, and the error.
          DISCRIMINATES: an error whose size is set by the vector's LARGEST channel (a
          shared per-tensor scale) from one set by the vector's typical channel. Position 0
          and the ordinary positions differ by an order of magnitude in max|f|/||f||, so
          the two hypotheses make opposite predictions there.

PHASE H   the shared-scale regression. Across all (prompt,position) samples at each state,
          which predicts the ABSOLUTE error better: the vector's norm ||f||, or its largest
          channel max|f|? A per-channel scale would make error track ||f||; one shared scale
          set by the largest channel makes it track max|f|.
          DISCRIMINATES: per-channel-scaled quantization from shared-per-vector-scale
          quantization. It does not identify WHICH of the 18 activation-quantization sites
          contributes most; it establishes the shape common to all of them.

Read-only on every input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1820_mechanism import Capture, _synth  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def r2(true, pred):
    return float(1.0 - np.sum((true - pred) ** 2) / np.sum(true**2))


def kridge(Xtr, Etr, Xte, lam):
    K = Xtr @ Xtr.T
    W = np.linalg.solve(K + lam * np.eye(K.shape[0]), Etr)
    return (Xte @ Xtr.T) @ W


def phase_f(caps, shared, log, out, seed=20260807):
    log("\n=== PHASE F -- a fixed per-channel bias against a fixed linear map ===")
    log("Fit on p1,p2,p3 (all positions); score on q1,q2,q3 at positions >= %d." % shared)
    log("bias  = the training set's own mean error vector, applied unchanged to held-out samples.")
    log("lin   = a centred ridge map on top of that bias, best over lambda in {1e-6..1e4}.")
    log("The nulls are the same synthetic mechanisms PHASE B used, on the same real float states.")
    tr = [c for c in caps if c.pid in ("p1", "p2", "p3")]
    te = [c for c in caps if c.pid in ("q1", "q2", "q3")]
    lams = [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4]
    kinds = ["observed", "rotation", "diagonal", "sparse", "dense"]
    res = {}
    log("\n      |        observed       |     rotation      |     diagonal      |  sparse |   dense")
    log("state |  bias   lin   lin-bias |  bias   lin       |  bias   lin       |   lin   |    lin")
    for s in range(caps[0].S):
        Xtr = np.concatenate([c.flt[:, s] for c in tr])
        Xte = np.concatenate([c.flt[shared:, s] for c in te])
        Etr_obs = np.concatenate([c.err[:, s] for c in tr])
        Ete_obs = np.concatenate([c.err[shared:, s] for c in te])
        rel_tr = np.linalg.norm(Etr_obs, axis=1) / np.linalg.norm(Xtr, axis=1)
        rel_te = np.linalg.norm(Ete_obs, axis=1) / np.linalg.norm(Xte, axis=1)
        xbar = Xtr.mean(0)
        row = {}
        for kind in kinds:
            if kind == "observed":
                Etr, Ete = Etr_obs, Ete_obs
            else:
                rng = np.random.default_rng(seed + s)
                allf = np.concatenate([Xtr, Xte])
                allr = np.concatenate([rel_tr, rel_te])
                alle = _synth(rng, allf, allr, kind, caps[0].H)
                Etr, Ete = alle[: Xtr.shape[0]], alle[Xtr.shape[0]:]
            b = Etr.mean(0)
            rb = r2(Ete, np.broadcast_to(b, Ete.shape))
            rl = max(
                r2(Ete, b + kridge(Xtr - xbar, Etr - b, Xte - xbar, lam)) for lam in lams
            )
            row[kind] = dict(bias=rb, lin=rl)
        res[s] = row
        o, ro, di, sp, de = (row[k] for k in kinds)
        log(f"{s:5d} | {o['bias']:6.3f} {o['lin']:6.3f} {o['lin']-o['bias']:8.3f} |"
            f" {ro['bias']:6.3f} {ro['lin']:6.3f}      |"
            f" {di['bias']:6.3f} {di['lin']:6.3f}      |"
            f" {sp['lin']:7.3f} | {de['lin']:7.3f}")
    out["phase_f"] = res
    ob = [res[s]["observed"]["bias"] for s in res if s > 0]
    ol = [res[s]["observed"]["lin"] for s in res if s > 0]
    inc = [res[s]["observed"]["lin"] - res[s]["observed"]["bias"] for s in res if s > 0]
    rl = [res[s]["rotation"]["lin"] - res[s]["rotation"]["bias"] for s in res if s > 0]
    dl = [res[s]["diagonal"]["lin"] - res[s]["diagonal"]["bias"] for s in res if s > 0]
    log(f"\nF-RESULT over states 1-28: observed bias alone explains {np.mean(ob):.3f} of held-out")
    log(f"  error energy ({min(ob):.3f}..{max(ob):.3f}); adding a linear map adds {np.mean(inc):+.3f}")
    log(f"  ({min(inc):+.3f}..{max(inc):+.3f}). The same linear-map increment is {np.mean(rl):+.3f} for a")
    log(f"  true rotation and {np.mean(dl):+.3f} for a diagonal scaling on the same states.")


def phase_g(caps, log, out):
    log("\n=== PHASE G -- position 0 and the shared per-vector scale ===")
    log("rail% = share of the 1536 committed codes at |code| >= 127, the int8 rail.")
    log("All figures are means over the 6 prompts. Positions 0-23 are the shared token prefix,")
    log("where every prompt's state is bit-identical (GATE G1), so 'mean over prompts' there is")
    log("one value repeated six times, not an average over six independent draws.")
    picks = [0, 1, 2, 12, 23]
    res = {}
    for s in [0, 1, 2, 13, 26, 27, 28]:
        log(f"\n  state {s}")
        log("   position |    ||f||  max|f|/||f|| |      q    rail% | rel_l2 | abs err | ||e||/max|f|")
        rows = []
        for t in picks + ["mid", "last"]:
            if t == "mid":
                idx = [(c, tt) for c in caps for tt in range(24, c.T - 1)]
                label = "24..T-2"
            elif t == "last":
                idx = [(c, c.T - 1) for c in caps]
                label = "last"
            else:
                idx = [(c, t) for c in caps]
                label = str(t)
            nf = np.mean([np.linalg.norm(c.flt[tt, s]) for c, tt in idx])
            mx = np.mean([np.abs(c.flt[tt, s]).max() / np.linalg.norm(c.flt[tt, s]) for c, tt in idx])
            q = np.mean([c.q[tt, s] for c, tt in idx])
            rail = np.mean([np.mean(np.abs(c.codes[tt, s].astype(np.int32)) >= 127) for c, tt in idx])
            rel = np.mean([np.linalg.norm(c.err[tt, s]) / np.linalg.norm(c.flt[tt, s]) for c, tt in idx])
            ae = np.mean([np.linalg.norm(c.err[tt, s]) for c, tt in idx])
            emx = np.mean([np.linalg.norm(c.err[tt, s]) / np.abs(c.flt[tt, s]).max() for c, tt in idx])
            log(f"   {label:>8s} | {nf:8.2f}  {mx:11.4f} | {q:8.4f} {rail*100:6.2f} |"
                f" {rel:6.3f} | {ae:7.2f} | {emx:12.4f}")
            rows.append(dict(pos=label, nf=float(nf), maxratio=float(mx), q=float(q),
                             rail=float(rail), rel=float(rel), abserr=float(ae), e_over_max=float(emx)))
        res[s] = rows
    out["phase_g"] = res
    log("\nG-DISCRIMINATION: ||e||/max|f| is the quantity a shared per-vector scale predicts to be")
    log("  stable; rel_l2 = ||e||/||f|| is the quantity a per-channel scale predicts to be stable.")
    log("  Position 0 and the ordinary positions differ several-fold in max|f|/||f||, so whichever")
    log("  column moves LESS between them is the mechanism's own invariant.")


def phase_h(caps, log, out):
    log("\n=== PHASE H -- does the absolute error track ||f|| or max|f|? ===")
    log("Per state, over all 249 (prompt,position) samples: the coefficient of variation of")
    log("||e||/||f|| against that of ||e||/max|f|. The smaller CV names the quantity the error")
    log("is actually proportional to. Also: log-log regression of ||e|| on both predictors")
    log("jointly, whose exponents say which one carries the dependence.")
    res = {}
    log("\nstate | CV(||e||/||f||) | CV(||e||/max|f|) | winner   | exponent on ||f|| | on max|f|")
    for s in range(caps[0].S):
        ne, nf, mx = [], [], []
        for c in caps:
            ne += list(np.linalg.norm(c.err[:, s], axis=1))
            nf += list(np.linalg.norm(c.flt[:, s], axis=1))
            mx += list(np.abs(c.flt[:, s]).max(axis=1))
        ne, nf, mx = np.array(ne), np.array(nf), np.array(mx)
        a, b = ne / nf, ne / mx
        cva, cvb = a.std() / a.mean(), b.std() / b.mean()
        A = np.stack([np.ones_like(ne), np.log(nf), np.log(mx)], 1)
        coef, *_ = np.linalg.lstsq(A, np.log(ne), rcond=None)
        res[s] = dict(cv_norm=float(cva), cv_max=float(cvb), exp_norm=float(coef[1]),
                      exp_max=float(coef[2]))
        log(f"{s:5d} | {cva:15.4f} | {cvb:16.4f} | {'max|f|' if cvb<cva else '||f||  ':8s} |"
            f" {coef[1]:17.3f} | {coef[2]:9.3f}")
    out["phase_h"] = res
    cn = [res[s]["cv_norm"] for s in res if s > 0]
    cm = [res[s]["cv_max"] for s in res if s > 0]
    en = [res[s]["exp_norm"] for s in res if s > 0]
    em = [res[s]["exp_max"] for s in res if s > 0]
    log(f"\nH-RESULT over states 1-28: CV(||e||/||f||) mean {np.mean(cn):.4f}; "
        f"CV(||e||/max|f|) mean {np.mean(cm):.4f}. Joint log-log exponents: on ||f|| "
        f"{np.mean(en):+.3f}, on max|f| {np.mean(em):+.3f} (they sum to ~1 by scale invariance).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default="out/t1820")
    ap.add_argument("--prompts", default="p1,p2,p3,q1,q2,q3")
    ap.add_argument("--json", default="out/t1820/mechanism2.json")
    ap.add_argument("--phases", default="FGH")
    args = ap.parse_args(argv)

    d = Path(args.dump_dir)
    caps = [Capture(p, d) for p in args.prompts.split(",")]

    def log(s=""):
        print(s, flush=True)

    base = caps[0]
    shared = base.T
    for c in caps[1:]:
        n = min(shared, c.T)
        k = 0
        while k < n and base.tokens[k] == c.tokens[k]:
            k += 1
        shared = min(shared, k)
    log(f"shared token prefix: {shared} positions; prompts {[c.pid for c in caps]}; "
        f"positions {[c.T for c in caps]}")

    out = {}
    if "F" in args.phases:
        phase_f(caps, shared, log, out)
    if "G" in args.phases:
        phase_g(caps, log, out)
    if "H" in args.phases:
        phase_h(caps, log, out)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="ascii")
    log(f"\njson written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
