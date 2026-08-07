#!/usr/bin/env python3
"""T-1820 third analysis pass.

PHASE F2  a fixed per-channel BIAS against a fixed LINEAR MAP, on a POSITION-MATCHED
          population. The second pass fitted the bias on all positions and scored it on
          positions >= the shared prefix; position 0 carries an error two orders of magnitude
          larger than any other, so the training mean was not an estimate of the same
          quantity being scored (StandardsDocument.md 5.4, the population species). Here both
          sides are restricted to positions >= the shared prefix.
          DISCRIMINATES: a fixed additive per-channel bias from a fixed linear operator, and
          both from the unexplained remainder.

PHASE I   accumulation against amplification. Writing the per-layer error increment
          d_k = e_{k+1} - e_k, the growth of the error decomposes exactly as
          ||e_{k+1}||^2 = ||e_k||^2 + 2<e_k,d_k> + ||d_k||^2. Independent per-layer noise
          injection makes <e_k,d_k> ~ 0 and the error grow as the root-sum-square of the
          increments; a layer that AMPLIFIES the error it inherits makes <e_k,d_k> > 0 and the
          growth geometric.
          DISCRIMINATES: "independent quantization noise injected at each layer and
          accumulated" from "an early error amplified by every later layer". It does not
          attribute the injection to a particular site.

PHASE J   the massive-activation channels at position 0. Qwen2.5 develops very large,
          nearly-constant hidden-state channels at the first token from layer ~1 to the last
          layers, where they are removed. This phase tracks those specific channels' float and
          engine values across all 29 states at position 0.
          DISCRIMINATES: a position-0 failure that is the massive-activation channel being
          mis-cancelled from one that is spread over ordinary channels.

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


def phase_f2(caps, shared, log, out, seed=20260807):
    log("\n=== PHASE F2 -- fixed bias vs fixed linear map, position-matched populations ===")
    log(f"Both sides restricted to positions >= {shared}: fit on p1,p2,p3, score on q1,q2,q3.")
    log("bias = the training set's own mean error vector, applied unchanged out-of-sample.")
    log("lin  = a centred ridge map added on top, best over lambda in {1e-6..1e4}.")
    tr = [c for c in caps if c.pid in ("p1", "p2", "p3")]
    te = [c for c in caps if c.pid in ("q1", "q2", "q3")]
    lams = [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4]
    kinds = ["observed", "rotation", "diagonal", "sparse", "dense"]
    res = {}
    log("\n      |      observed        |  rotation      |  diagonal      | sparse | dense")
    log("state |  bias    lin   added |  bias    lin   |  bias    lin   |  lin   |  lin")
    for s in range(caps[0].S):
        Xtr = np.concatenate([c.flt[shared:, s] for c in tr])
        Xte = np.concatenate([c.flt[shared:, s] for c in te])
        Etr_o = np.concatenate([c.err[shared:, s] for c in tr])
        Ete_o = np.concatenate([c.err[shared:, s] for c in te])
        rel = np.concatenate([np.linalg.norm(Etr_o, axis=1) / np.linalg.norm(Xtr, axis=1),
                              np.linalg.norm(Ete_o, axis=1) / np.linalg.norm(Xte, axis=1)])
        xbar = Xtr.mean(0)
        row = {}
        for kind in kinds:
            if kind == "observed":
                Etr, Ete = Etr_o, Ete_o
            else:
                rng = np.random.default_rng(seed + s)
                allf = np.concatenate([Xtr, Xte])
                alle = _synth(rng, allf, rel, kind, caps[0].H)
                Etr, Ete = alle[: Xtr.shape[0]], alle[Xtr.shape[0]:]
            b = Etr.mean(0)
            rb = r2(Ete, np.broadcast_to(b, Ete.shape))
            rl = max(r2(Ete, b + kridge(Xtr - xbar, Etr - b, Xte - xbar, lam)) for lam in lams)
            row[kind] = dict(bias=rb, lin=rl)
        res[s] = row
        o, ro, di, sp, de = (row[k] for k in kinds)
        log(f"{s:5d} | {o['bias']:6.3f} {o['lin']:6.3f} {o['lin']-o['bias']:7.3f} |"
            f" {ro['bias']:6.3f} {ro['lin']:6.3f} | {di['bias']:6.3f} {di['lin']:6.3f} |"
            f" {sp['lin']:6.3f} | {de['lin']:6.3f}")
    out["phase_f2"] = res
    ob = [res[s]["observed"]["bias"] for s in res if s > 0]
    ol = [res[s]["observed"]["lin"] for s in res if s > 0]
    add = [res[s]["observed"]["lin"] - res[s]["observed"]["bias"] for s in res if s > 0]
    rb = [res[s]["rotation"]["bias"] for s in res if s > 0]
    rl = [res[s]["rotation"]["lin"] for s in res if s > 0]
    db = [res[s]["diagonal"]["bias"] for s in res if s > 0]
    dl = [res[s]["diagonal"]["lin"] for s in res if s > 0]
    sl = [res[s]["sparse"]["lin"] for s in res if s > 0]
    nl = [res[s]["dense"]["lin"] for s in res if s > 0]
    log(f"\nF2-RESULT over states 1-28, held out:")
    log(f"  observed  bias {np.mean(ob):+.3f} ({min(ob):+.3f}..{max(ob):+.3f})   "
        f"bias+linear {np.mean(ol):+.3f} ({min(ol):+.3f}..{max(ol):+.3f})   "
        f"the linear map adds {np.mean(add):+.3f}")
    log(f"  rotation  bias {np.mean(rb):+.3f}   bias+linear {np.mean(rl):+.3f}")
    log(f"  diagonal  bias {np.mean(db):+.3f}   bias+linear {np.mean(dl):+.3f}")
    log(f"  sparse    bias+linear {np.mean(sl):+.3f}      dense  bias+linear {np.mean(nl):+.3f}")


def phase_i(caps, shared, log, out):
    log("\n=== PHASE I -- accumulation against amplification ===")
    log("d_k = e_{k+1} - e_k, the per-layer error increment. Exactly:")
    log("  ||e_{k+1}||^2 = ||e_k||^2 + 2<e_k,d_k> + ||d_k||^2.")
    log("coh = 2<e_k,d_k> / (||e_{k+1}||^2 - ||e_k||^2), the share of the layer's error-energy")
    log("growth carried by the COHERENT cross term rather than by the increment's own energy.")
    log("cos = cos(e_k, d_k). Zero means the layer injects error orthogonal to what it")
    log("inherited (accumulation); positive means the layer amplifies what it inherited.")
    log("Cell: all 249 (prompt,position) samples; the SD is across those samples.")
    log("\nlayer | ||e_k|| | ||d_k|| | ||d||/||e|| | cos(e,d) +/-  | coherent share of growth")
    res = {}
    for k in range(caps[0].S - 1):
        ne, nd, cs, coh = [], [], [], []
        for c in caps:
            e0, e1 = c.err[:, k], c.err[:, k + 1]
            d = e1 - e0
            n0 = np.linalg.norm(e0, axis=1)
            nd_ = np.linalg.norm(d, axis=1)
            dot = np.sum(e0 * d, axis=1)
            ne += list(n0)
            nd += list(nd_)
            cs += list(dot / (n0 * nd_))
            growth = np.linalg.norm(e1, axis=1) ** 2 - n0**2
            ok = np.abs(growth) > 1e-12
            coh += list(2 * dot[ok] / growth[ok])
        cs = np.array(cs)
        res[k] = dict(ne=float(np.mean(ne)), nd=float(np.mean(nd)),
                      ratio=float(np.mean(nd) / np.mean(ne)), cos=float(cs.mean()),
                      cos_sd=float(cs.std()), coh=float(np.median(coh)))
        log(f"{k:5d} | {np.mean(ne):7.2f} | {np.mean(nd):7.2f} | {np.mean(nd)/np.mean(ne):11.3f} |"
            f" {cs.mean():+.3f} +/-{cs.std():.3f} | {np.median(coh):+.3f}")
    out["phase_i"] = res
    c = [res[k]["cos"] for k in res if k > 0]
    log(f"\nI-RESULT over layers 1-27: cos(e_k,d_k) mean {np.mean(c):+.3f}, "
        f"range {min(c):+.3f}..{max(c):+.3f}.")
    log("I-DISCRIMINATION: independent per-layer injection predicts cos ~ 0 (the increment is")
    log("  orthogonal to the inherited error, so error energy accumulates as a root-sum-square);")
    log("  amplification of the inherited error predicts cos > 0. The two are exclusive and the")
    log("  measured value falls in one of them. It does not name the injecting site.")


def phase_j(caps, log, out):
    log("\n=== PHASE J -- the massive-activation channels at position 0 ===")
    c = caps[0]
    ref = np.abs(c.flt[0, 13])
    top = np.argsort(ref)[::-1][:3]
    log(f"Channels ranked by |float| at position 0, state 13: {list(map(int, top))} "
        f"(values {[round(float(c.flt[0,13,i]),1) for i in top]}).")
    log("Every prompt is bit-identical at position 0 (GATE G1), so one prompt IS the population.")
    log("\nstate |" + "".join(f"   ch{int(i)} float    engine |" for i in top) +
        "  ||f|| all-ch | ||f|| ex-top3 | rel_l2 | rel_l2 ex-top3")
    rows = []
    for s in range(c.S):
        f, g = c.flt[0, s], c.eng[0, s]
        mask = np.ones(c.H, bool)
        mask[top] = False
        rel = np.linalg.norm(g - f) / np.linalg.norm(f)
        rel_ex = np.linalg.norm((g - f)[mask]) / np.linalg.norm(f[mask])
        cells = "".join(f" {f[i]:11.1f} {g[i]:9.1f} |" for i in top)
        log(f"{s:5d} |{cells} {np.linalg.norm(f):12.1f} | {np.linalg.norm(f[mask]):13.1f} |"
            f" {rel:6.3f} | {rel_ex:14.3f}")
        rows.append(dict(state=s, rel=float(rel), rel_ex=float(rel_ex),
                         nf=float(np.linalg.norm(f)), nf_ex=float(np.linalg.norm(f[mask])),
                         float_top=[float(f[i]) for i in top],
                         eng_top=[float(g[i]) for i in top]))
    out["phase_j"] = dict(channels=[int(i) for i in top], rows=rows)
    log("\nJ-DISCRIMINATION: excluding the three channels isolates whether position 0's")
    log("  divergence lives in the massive-activation channels or in the ordinary ones.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default="out/t1820")
    ap.add_argument("--prompts", default="p1,p2,p3,q1,q2,q3")
    ap.add_argument("--json", default="out/t1820/mechanism3.json")
    ap.add_argument("--phases", default="FIJ")
    args = ap.parse_args(argv)

    caps = [Capture(p, Path(args.dump_dir)) for p in args.prompts.split(",")]

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
    log(f"shared token prefix: {shared} positions; positions {[c.T for c in caps]}")

    out = {}
    if "F" in args.phases:
        phase_f2(caps, shared, log, out)
    if "I" in args.phases:
        phase_i(caps, shared, log, out)
    if "J" in args.phases:
        phase_j(caps, log, out)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="ascii")
    log(f"\njson written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
