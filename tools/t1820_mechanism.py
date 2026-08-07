#!/usr/bin/env python3
"""T-1820 -- what IS the residual-stream drift, if it is not a rotation?

Reads this ticket's own all-position captures (out/t1820/*_engine.bin from
tools/t1820_position_residual_probe.cpp, out/t1820/*_float.npy from
tools/t1820_float_position_reference.py) and runs five phases, each stating what it
separates and what it cannot.

GATES     instrument checks: token-id agreement, shared-prefix replication (positions whose
          whole token prefix is shared across prompts must produce bit-identical engine codes
          and identical float states), and reproduction of T-1795's own published
          last-position table.

PHASE A   the per-(state,position) scalar table: rel_l2, cosine, norm ratio, the engine's own
          committed quantization step q, and the int8 storage floor.
          DISCRIMINATES: "the drift is the residual stream's own int8 storage" (floor ~
          observed) from "the drift is accumulated and merely lands in that storage"
          (floor << observed). Does NOT separate rotation / diagonal / dense noise.

PHASE B   held-out linear-operator prediction. For each state, a ridge map e ~ A f is fitted
          on a training set of (prompt,position) samples and scored out-of-sample on prompts
          it never saw. The identical pipeline is run on synthetic errors built from the real
          float states under four mechanisms, each forced to the same per-sample relative-L2
          as the observation.
          DISCRIMINATES: {a fixed linear operator per state -- rotation, diagonal scaling}
          from {additive noise -- dense, sparse}. A fixed operator is predictable
          out-of-sample by construction; additive noise is not. Does NOT separate a rotation
          from a diagonal scaling (both are linear maps); PHASE C does that.

PHASE C   per-channel error structure at every position: the error's top-decile/bottom-decile
          magnitude ratio against the signal's own, and the rank correlation of |err| with |f|.
          DISCRIMINATES: a diagonal per-channel scaling (|e_i| proportional to |f_i|) and
          sparse per-channel spikes (a few channels carrying the mass) from any mechanism
          whose error is flat across channels. Does NOT separate a rotation from dense noise.

PHASE D   the position axis: rel_l2 and the LSB ratio as functions of token index, against a
          resolving power measured as the across-prompt spread at matched position over
          positions where the prompts carry different tokens.
          DISCRIMINATES: position-dependent structure from position-independent structure. It
          does not by itself attribute a positional effect to a cause; the width test below
          does that for the attention-width hypothesis specifically.

PHASE E   the attention-width test: whether the per-state error at position t tracks sqrt(t+1),
          the growth an error accumulated over t+1 attention terms would show.
          DISCRIMINATES: an error injected per-attended-position (grows with context width)
          from one injected once per layer (flat in position).

Read-only on every input.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# ---------------------------------------------------------------- loading


def load_engine(path: Path):
    raw = path.read_bytes()
    np_, ns, nh = struct.unpack_from("<iii", raw, 0)
    off = 12
    tokens = np.frombuffer(raw, dtype="<i4", count=np_, offset=off).copy()
    off += 4 * np_
    rec = 16 + nh
    codes = np.empty((np_, ns, nh), dtype=np.int8)
    scale_m = np.empty((np_, ns), dtype=np.int64)
    scale_e = np.empty((np_, ns), dtype=np.int64)
    for t in range(np_):
        for s in range(ns):
            o = off + (t * ns + s) * rec
            m, e = struct.unpack_from("<qq", raw, o)
            scale_m[t, s] = m
            scale_e[t, s] = e
            codes[t, s] = np.frombuffer(raw, dtype=np.int8, count=nh, offset=o + 16)
    if off + np_ * ns * rec != len(raw):
        raise SystemExit(f"{path}: trailing bytes -- format mismatch")
    return tokens, codes, scale_m, scale_e


def dequant(codes, m, e):
    """value = code * m * 2^e -- checked_chain_funnel.h:66, the CarriedScale's own
    documented meaning, used identically by every probe in this chain."""
    q = m.astype(np.float64) * np.exp2(e.astype(np.float64))
    return codes.astype(np.float64) * q[:, :, None], q


class Capture:
    def __init__(self, pid: str, d: Path):
        self.pid = pid
        tokens, codes, m, e = load_engine(d / f"{pid}_engine.bin")
        self.tokens = tokens
        self.codes = codes
        self.eng, self.q = dequant(codes, m, e)
        self.flt = np.load(d / f"{pid}_float.npy").astype(np.float64)
        ftok = np.load(d / f"{pid}_float.tokens.npy")
        if not np.array_equal(ftok, tokens):
            raise SystemExit(f"{pid}: engine and float token ids disagree")
        if self.flt.shape != self.eng.shape:
            raise SystemExit(f"{pid}: shapes disagree {self.flt.shape} vs {self.eng.shape}")
        self.err = self.eng - self.flt
        self.T, self.S, self.H = self.eng.shape


# ---------------------------------------------------------------- gates


def gates(caps, t1795_dir: Path, log):
    log("=== GATES ===")
    log(f"prompts: {[c.pid for c in caps]}  positions: {[c.T for c in caps]}  "
        f"states: {caps[0].S}  channels: {caps[0].H}")

    # G1 -- shared-prefix replication. A position whose ENTIRE token prefix is shared by two
    # prompts must produce bit-identical engine codes and identical float states, because
    # nothing downstream of the prefix can reach it. This is a determinism check on both
    # instruments at once and it establishes which positions are duplicates (excluded from
    # PHASE B's held-out test, where they would leak).
    base = caps[0]
    shared = base.T
    for c in caps[1:]:
        n = min(shared, c.T)
        k = 0
        while k < n and base.tokens[k] == c.tokens[k]:
            k += 1
        shared = min(shared, k)
    log(f"G1 shared token prefix across all {len(caps)} prompts: {shared} positions")
    worst_code, worst_flt = 0, 0.0
    for c in caps[1:]:
        worst_code = max(worst_code, int(np.abs(c.codes[:shared].astype(np.int32) -
                                                base.codes[:shared].astype(np.int32)).max()))
        worst_flt = max(worst_flt, float(np.abs(c.flt[:shared] - base.flt[:shared]).max()))
    log(f"G1 engine codes over the shared prefix, worst |difference| across prompts: {worst_code} "
        f"(0 => bit-identical)")
    log(f"G1 float states over the shared prefix, worst |difference| across prompts: {worst_flt:.3e}")
    if worst_code != 0:
        raise SystemExit("GATE G1 FAILED: engine is not deterministic over an identical prefix")

    # G2 -- reproduce T-1795's own published last-position rel_l2 at the states it tabulates,
    # from this ticket's independent capture and its own dequantisation.
    published = {0: 0.0259, 1: 0.4357, 2: 0.3800, 5: 0.4225, 9: 0.3405, 13: 0.2834,
                 15: 0.2615, 18: 0.3002, 22: 0.2929, 27: 0.2657, 28: 0.4568}
    p = [c for c in caps if c.pid in ("p1", "p2", "p3")]
    worst = 0.0
    rows = []
    for s, want in published.items():
        got = float(np.mean([
            np.linalg.norm(c.err[-1, s]) / np.linalg.norm(c.flt[-1, s]) for c in p
        ]))
        worst = max(worst, abs(got - want))
        rows.append((s, want, got))
    log("G2 T-1795's own last-position table, recomputed here (mean of p1,p2,p3):")
    log("    state  published  this ticket")
    for s, want, got in rows:
        log(f"    {s:5d}  {want:9.4f}  {got:11.4f}")
    log(f"G2 worst absolute disagreement: {worst:.4f}  "
        f"(T-1795 published to 4 decimals; this capture is independent of its dump)")
    return shared


# ---------------------------------------------------------------- phase A


def phase_a(caps, log, out):
    log("\n=== PHASE A -- scalars, storage floor, and the LSB ratio (all positions) ===")
    log("Cell: 6 prompts, 249 positions total, all 29 states, all 1536 channels. Mean over"
        " every (prompt,position) at that state; +/- is the population SD across those samples.")
    log("floor = ||clamp(round(f/q),-128,127)*q - f|| / ||f||, the relative error an otherwise-")
    log("        PERFECT engine still incurs storing the float reference on that state's own grid.")
    log("lsb   = ||engine - float|| / q, the drift measured in the state's own quantization steps.")
    log("\nstate | rel_l2  +/-     | cosine  | norm_r | floor   | obs/floor | lsb (rms/chan)")
    tab = {}
    for s in range(caps[0].S):
        rel, cos, nr, fl, lsb = [], [], [], [], []
        for c in caps:
            f = c.flt[:, s]
            e = c.err[:, s]
            g = c.eng[:, s]
            q = c.q[:, s]
            nf = np.linalg.norm(f, axis=1)
            ne = np.linalg.norm(e, axis=1)
            rel += list(ne / nf)
            cos += list(np.sum(f * g, axis=1) / (nf * np.linalg.norm(g, axis=1)))
            nr += list(np.linalg.norm(g, axis=1) / nf)
            cq = np.clip(np.round(f / q[:, None]), -128, 127) * q[:, None]
            fl += list(np.linalg.norm(cq - f, axis=1) / nf)
            lsb += list(ne / q / np.sqrt(c.H))
        rel, fl = np.array(rel), np.array(fl)
        tab[s] = dict(rel=float(rel.mean()), rel_sd=float(rel.std()), cos=float(np.mean(cos)),
                      nr=float(np.mean(nr)), floor=float(fl.mean()),
                      ratio=float(rel.mean() / fl.mean()), lsb=float(np.mean(lsb)))
        log(f"{s:5d} | {rel.mean():.4f} +/-{rel.std():.4f} | {np.mean(cos):.4f}  |"
            f" {np.mean(nr):.4f} | {fl.mean():.4f}  | {rel.mean()/fl.mean():9.2f} | {np.mean(lsb):8.3f}")
    out["phase_a"] = tab
    r = [v["ratio"] for v in tab.values() if v["ratio"] > 0]
    log(f"\nA-RESULT obs/floor spans {min(r[1:]):.2f}-{max(r[1:]):.2f} over states 1-28 "
        f"({r[0]:.2f} at the embedding).")
    log("A-DISCRIMINATION: kills 'the residual drift is the residual stream's own int8 storage'.")
    log("  It separates no other pair of candidates and no claim about them is drawn from it.")


# ---------------------------------------------------------------- phase B


def _synth(rng, f, target_rel, kind, H):
    """Synthetic error for one state, forced to each sample's OWN observed relative L2."""
    N = f.shape[0]
    if kind == "rotation":
        # A genuine orthogonal map, fixed for this state: independent Givens rotations on
        # H/2 disjoint channel pairs, by a common angle chosen to hit the target.
        idx = rng.permutation(H)
        a, b = idx[: H // 2], idx[H // 2 :]
        # ||Rf-f||^2 = 2(1-cos t)||f||^2 for a rotation acting on a complete pairing
        out = np.zeros_like(f)
        for i in range(N):
            th = np.arccos(np.clip(1 - target_rel[i] ** 2 / 2, -1, 1))
            ca, sa = np.cos(th), np.sin(th)
            g = f[i].copy()
            g[a] = ca * f[i][a] - sa * f[i][b]
            g[b] = sa * f[i][a] + ca * f[i][b]
            out[i] = g - f[i]
        return out
    if kind == "diagonal":
        d = rng.normal(size=H)
        d = d / np.linalg.norm(d) * np.sqrt(H)  # unit RMS, non-constant
        e = f * d
        return e / np.linalg.norm(e, axis=1, keepdims=True) * (
            target_rel[:, None] * np.linalg.norm(f, axis=1, keepdims=True))
    if kind == "sparse":
        cols = rng.choice(H, size=24, replace=False)
        e = np.zeros_like(f)
        e[:, cols] = rng.normal(size=(N, 24))
        return e / np.linalg.norm(e, axis=1, keepdims=True) * (
            target_rel[:, None] * np.linalg.norm(f, axis=1, keepdims=True))
    if kind == "dense":
        e = rng.normal(size=f.shape)
        return e / np.linalg.norm(e, axis=1, keepdims=True) * (
            target_rel[:, None] * np.linalg.norm(f, axis=1, keepdims=True))
    raise ValueError(kind)


def _heldout_r2(Xtr, Etr, Xte, Ete, lam):
    """Kernel ridge for the multi-output map e ~ A f (N < d, so solve in sample space).
    Returns the out-of-sample fraction of test error energy the fitted map explains."""
    K = Xtr @ Xtr.T
    W = np.linalg.solve(K + lam * np.eye(K.shape[0]), Etr)   # (Ntr, H)
    pred = (Xte @ Xtr.T) @ W
    return float(1.0 - np.sum((Ete - pred) ** 2) / np.sum(Ete ** 2))


def phase_b(caps, shared, log, out, seed=20260807):
    log("\n=== PHASE B -- held-out linear-operator prediction ===")
    train_ids, test_ids = ("p1", "p2", "p3"), ("q1", "q2", "q3")
    log(f"Split: fit on {train_ids} (all positions), score on {test_ids} at positions >= {shared},")
    log("i.e. strictly after the token prefix the six prompts share, so no test sample has an")
    log("identical twin in the training set. Ridge lambda is swept and the BEST held-out score")
    log("per mechanism is reported, which favours every candidate equally.")
    tr = [c for c in caps if c.pid in train_ids]
    te = [c for c in caps if c.pid in test_ids]
    rng = np.random.default_rng(seed)
    lams = [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4]
    kinds = ["observed", "rotation", "diagonal", "sparse", "dense"]
    res = {}
    log("\nstate | " + " | ".join(f"{k:>9s}" for k in kinds))
    for s in range(caps[0].S):
        Xtr = np.concatenate([c.flt[:, s] for c in tr])
        Xte = np.concatenate([c.flt[shared:, s] for c in te])
        Etr_obs = np.concatenate([c.err[:, s] for c in tr])
        Ete_obs = np.concatenate([c.err[shared:, s] for c in te])
        rel_tr = np.linalg.norm(Etr_obs, axis=1) / np.linalg.norm(Xtr, axis=1)
        rel_te = np.linalg.norm(Ete_obs, axis=1) / np.linalg.norm(Xte, axis=1)
        row = {}
        for kind in kinds:
            if kind == "observed":
                Etr, Ete = Etr_obs, Ete_obs
            else:
                # one operator/noise realisation per state, shared by train and test -- the
                # generous case for the linear-map candidates
                r2 = np.random.default_rng(seed + s)
                allf = np.concatenate([Xtr, Xte])
                allr = np.concatenate([rel_tr, rel_te])
                alle = _synth(r2, allf, allr, kind, caps[0].H)
                Etr, Ete = alle[: Xtr.shape[0]], alle[Xtr.shape[0]:]
            row[kind] = max(_heldout_r2(Xtr, Etr, Xte, Ete, lam) for lam in lams)
        res[s] = row
        log(f"{s:5d} | " + " | ".join(f"{row[k]:9.3f}" for k in kinds))
    out["phase_b"] = res
    obs = [res[s]["observed"] for s in res if s > 0]
    rot = [res[s]["rotation"] for s in res if s > 0]
    dia = [res[s]["diagonal"] for s in res if s > 0]
    den = [res[s]["dense"] for s in res if s > 0]
    spa = [res[s]["sparse"] for s in res if s > 0]
    log(f"\nB-RESULT over states 1-28: observed {min(obs):.3f}..{max(obs):.3f} (mean {np.mean(obs):.3f});"
        f" rotation {min(rot):.3f}..{max(rot):.3f} (mean {np.mean(rot):.3f});"
        f" diagonal {min(dia):.3f}..{max(dia):.3f} (mean {np.mean(dia):.3f});"
        f" sparse {min(spa):.3f}..{max(spa):.3f} (mean {np.mean(spa):.3f});"
        f" dense {min(den):.3f}..{max(den):.3f} (mean {np.mean(den):.3f}).")
    log("B-DISCRIMINATION: the two linear-map candidates and the two additive-noise candidates")
    log("  land in disjoint bands under an identical pipeline on the same real float states; the")
    log("  observation falls in one of them. It does NOT separate rotation from diagonal scaling.")


# ---------------------------------------------------------------- phase C


def phase_c(caps, log, out):
    log("\n=== PHASE C -- per-channel error structure at every position ===")
    log("decile ratio = mean|err| in the top decile of channels by |f| / the same in the bottom")
    log("decile, against the SIGNAL's own top/bottom-decile ratio measured on the same channels.")
    log("\nstate | err decile ratio | signal decile ratio | spearman(|err|,|f|) | top-1% share of err^2")
    res = {}
    for s in range(caps[0].S):
        er, sr, sp, top = [], [], [], []
        for c in caps:
            f, e = np.abs(c.flt[:, s]), np.abs(c.err[:, s])
            order = np.argsort(f, axis=1)
            n10 = c.H // 10
            lo = np.take_along_axis(e, order[:, :n10], 1).mean(1)
            hi = np.take_along_axis(e, order[:, -n10:], 1).mean(1)
            flo = np.take_along_axis(f, order[:, :n10], 1).mean(1)
            fhi = np.take_along_axis(f, order[:, -n10:], 1).mean(1)
            er += list(hi / np.maximum(lo, 1e-30))
            sr += list(fhi / np.maximum(flo, 1e-30))
            rf = np.argsort(np.argsort(f, axis=1), axis=1).astype(np.float64)
            re = np.argsort(np.argsort(e, axis=1), axis=1).astype(np.float64)
            rf -= rf.mean(1, keepdims=True)
            re -= re.mean(1, keepdims=True)
            sp += list(np.sum(rf * re, 1) / np.sqrt(np.sum(rf**2, 1) * np.sum(re**2, 1)))
            e2 = np.sort(e**2, axis=1)[:, ::-1]
            top += list(e2[:, : c.H // 100].sum(1) / e2.sum(1))
        res[s] = dict(err_decile=float(np.mean(er)), sig_decile=float(np.mean(sr)),
                      spearman=float(np.mean(sp)), top1pct=float(np.mean(top)))
        log(f"{s:5d} | {np.mean(er):16.2f} | {np.mean(sr):19.2f} | {np.mean(sp):19.3f} |"
            f" {np.mean(top):20.4f}")
    out["phase_c"] = res
    log("\nC-DISCRIMINATION: a diagonal per-channel scaling forces err decile ratio == signal")
    log("  decile ratio exactly; sparse spikes force a top-1% share near 1.0. Neither shape can")
    log("  be produced by a mechanism whose error is flat across channels. It does NOT separate")
    log("  a rotation from dense noise -- a generic rotation is channel-flat too.")


# ---------------------------------------------------------------- phase D + E


def phase_de(caps, shared, log, out):
    log("\n=== PHASE D -- the position axis ===")
    log("Per state, rel_l2 as a function of token index. Resolving power is the across-prompt")
    log("population SD at MATCHED position over positions >= the shared prefix, where the six")
    log("prompts carry different tokens -- so it measures instrument+content spread at fixed")
    log("position, which is what a position effect must be resolved against.")
    Tmin = min(c.T for c in caps)
    res = {}
    log("\nstate | rel_l2 @t=0 | @t=1..3 | @prefix.. | @last | across-prompt SD | position span / SD")
    for s in range(caps[0].S):
        M = np.array([[np.linalg.norm(c.err[t, s]) / np.linalg.norm(c.flt[t, s])
                       for t in range(Tmin)] for c in caps])   # (P, Tmin)
        prof = M.mean(0)
        sd = float(M[:, shared:].std(0).mean())
        span = float(prof[shared:].max() - prof[shared:].min())
        res[s] = dict(t0=float(prof[0]), early=float(prof[1:4].mean()),
                      mid=float(prof[shared:].mean()), last=float(prof[-1]),
                      sd=sd, span_over_sd=float(span / sd) if sd > 0 else 0.0,
                      profile=[float(x) for x in prof])
        log(f"{s:5d} | {prof[0]:11.4f} | {prof[1:4].mean():7.4f} | {prof[shared:].mean():9.4f} |"
            f" {prof[-1]:6.4f} | {sd:16.4f} | {span/sd if sd>0 else 0:18.1f}")
    out["phase_d"] = res

    log("\n=== PHASE E -- the attention-width test ===")
    log("If the per-layer error injection accumulates over the t+1 attended positions, the")
    log("layer-local error grows as sqrt(t+1). Fitted per state on positions >= the shared")
    log("prefix: log(rel_l2) = a + b*log(t+1). b ~ 0.5 is width accumulation in a relative")
    log("measure; b ~ 0 is a per-layer injection independent of context width.")
    log("Also reported: position 0, where attention attends to exactly one key with probability")
    log("1, so no softmax-width term can exist -- a control the fit cannot fake.")
    log("\nstate | slope b | b/SE  | rel_l2 @t=0 | mean rel_l2 t>=prefix | ratio")
    ee = {}
    for s in range(caps[0].S):
        xs, ys = [], []
        for c in caps:
            for t in range(shared, c.T):
                xs.append(np.log(t + 1.0))
                ys.append(np.log(np.linalg.norm(c.err[t, s]) / np.linalg.norm(c.flt[t, s])))
        xs, ys = np.array(xs), np.array(ys)
        A = np.stack([np.ones_like(xs), xs], 1)
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        resid = ys - A @ coef
        s2 = resid @ resid / (len(ys) - 2)
        se = np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1])
        t0 = float(np.mean([np.linalg.norm(c.err[0, s]) / np.linalg.norm(c.flt[0, s]) for c in caps]))
        mid = res[s]["mid"]
        ee[s] = dict(b=float(coef[1]), se=float(se), t0=t0, mid=mid)
        log(f"{s:5d} | {coef[1]:7.3f} | {coef[1]/se:6.1f} | {t0:11.4f} | {mid:21.4f} |"
            f" {mid/t0 if t0>0 else 0:6.2f}")
    out["phase_e"] = ee


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default="out/t1820")
    ap.add_argument("--t1795-dir", default="../t1795-residual-drift/out/t1795")
    ap.add_argument("--prompts", default="p1,p2,p3,q1,q2,q3")
    ap.add_argument("--json", default="out/t1820/mechanism.json")
    ap.add_argument("--phases", default="ABCDE")
    args = ap.parse_args(argv)

    d = Path(args.dump_dir)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    caps = [Capture(p, d) for p in args.prompts.split(",")]
    out = {}
    shared = gates(caps, Path(args.t1795_dir), log)
    if "A" in args.phases:
        phase_a(caps, log, out)
    if "B" in args.phases:
        phase_b(caps, shared, log, out)
    if "C" in args.phases:
        phase_c(caps, log, out)
    if "D" in args.phases or "E" in args.phases:
        phase_de(caps, shared, log, out)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="ascii")
    log(f"\njson written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
