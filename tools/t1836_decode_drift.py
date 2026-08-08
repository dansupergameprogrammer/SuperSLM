#!/usr/bin/env python3
r"""T-1836 -- does the prefill mechanism story hold during generation?

Reads the engine dumps this ticket's probe wrote (prompt positions AND generated positions,
committed int8 codes plus each state's own CarriedScale) and the float32 reference's own
per-position states, and re-runs T-1820's own mechanism measurements with the population split
by whether the position is a prompt token or a generated one.

Three questions, each answered with a discrimination claim:

  (a) IS DECODE DRIFT THE SAME MECHANISM AT THE SAME MAGNITUDE?
      T-1820's prefill signatures are: channel-flat (error top/bottom-decile ratio 1.31-2.05
      against a signal ratio of 46.5-85.3), dense (top-1% share 0.098-0.173), per-layer
      increment 65-101% of the inherited error, anti-aligned at cosine -0.206, and error
      energy growing ~26% per layer. Each is recomputed here on the decode population and on
      this run's own prefill population, so the comparison is within one instrument rather
      than against a filed number from another.

  (b) DOES ACCUMULATED K/V QUANTIZATION OVER GENERATED CONTEXT ADD A GROWING TERM?
      Drift against generation index, fitted on the same log-log form T-1820 §7.5 used for
      the prompt-position axis, with its slope, standard error and t-ratio. The control that
      makes it a discrimination rather than an observation: the SAME fit over this run's own
      prompt positions in the same absolute-index range. A decode-specific term shows up as a
      decode slope that exceeds the prefill slope at matched index; a shared position effect
      shows up as the two agreeing.

  (c) THE FIRST-TOKEN RAIL MECHANISM's DECODE-SIDE ANALOG.
      T-1820 found position 0's massive-activation channels railing at +/-127, carried ~25%
      low, frozen bit-exact for 19 states, carrying 34-36% of that position's squared error.
      The census is recomputed per position class, including the FIRST GENERATED position
      specifically -- the decode-side candidate for a "first token" effect.

Gates, executed here before any figure is read:
  * the engine and float sides consumed the same token at every position (the matched-input
    condition the whole interior comparison rests on);
  * this run's prompt-position records reproduce T-1820's own committed dump for the same
    prompt, byte for byte, where that dump is available -- a cross-ticket replication of the
    prefill half at the level of the raw codes rather than of a summary statistic.

Usage
    python tools\t1836_decode_drift.py --dump-dir out\t1836 --ids p1,p2,p3,q1,q2,q3 \
        --suffix _forced --t1820-dir D:\SuperSLM\.worktrees\t1820-drift-mechanism\out\t1820 \
        --json-out out\t1836\decode_drift.json
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


def load_engine_t1836(path: Path):
    raw = path.read_bytes()
    np_, ns, nh, prompt_len = struct.unpack_from("<iiii", raw, 0)
    off = 16
    consumed = np.frombuffer(raw, dtype="<i4", count=np_, offset=off).copy()
    off += 4 * np_
    argmax = np.frombuffer(raw, dtype="<i4", count=np_, offset=off).copy()
    off += 4 * np_
    rec = 16 + nh
    codes = np.empty((np_, ns, nh), dtype=np.int8)
    m = np.empty((np_, ns), dtype=np.int64)
    e = np.empty((np_, ns), dtype=np.int64)
    for t in range(np_):
        for s in range(ns):
            o = off + (t * ns + s) * rec
            mm, ee = struct.unpack_from("<qq", raw, o)
            m[t, s] = mm
            e[t, s] = ee
            codes[t, s] = np.frombuffer(raw, dtype=np.int8, count=nh, offset=o + 16)
    if off + np_ * ns * rec != len(raw):
        raise SystemExit(f"{path}: trailing bytes -- format mismatch")
    return dict(n=np_, ns=ns, nh=nh, prompt_len=prompt_len, consumed=consumed,
                argmax=argmax, codes=codes, m=m, e=e)


def load_engine_t1820(path: Path):
    """T-1820's own dump layout (no prompt_len / argmax fields)."""
    raw = path.read_bytes()
    np_, ns, nh = struct.unpack_from("<iii", raw, 0)
    off = 12
    tokens = np.frombuffer(raw, dtype="<i4", count=np_, offset=off).copy()
    off += 4 * np_
    rec = 16 + nh
    codes = np.empty((np_, ns, nh), dtype=np.int8)
    m = np.empty((np_, ns), dtype=np.int64)
    e = np.empty((np_, ns), dtype=np.int64)
    for t in range(np_):
        for s in range(ns):
            o = off + (t * ns + s) * rec
            mm, ee = struct.unpack_from("<qq", raw, o)
            m[t, s] = mm
            e[t, s] = ee
            codes[t, s] = np.frombuffer(raw, dtype=np.int8, count=nh, offset=o + 16)
    return dict(n=np_, tokens=tokens, codes=codes, m=m, e=e)


def dequant(codes, m, e):
    """value = code * m * 2^e -- checked_chain_funnel.h:66, the CarriedScale's own documented
    meaning, used identically by every probe in this chain."""
    q = m.astype(np.float64) * np.exp2(e.astype(np.float64))
    return codes.astype(np.float64) * q[:, :, None], q


def slope_fit(x, y):
    """Ordinary least squares of y on x; returns (slope, se, t)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    b = ((x - xm) * (y - ym)).sum() / sxx
    resid = y - (ym + b * (x - xm))
    s2 = (resid ** 2).sum() / (n - 2)
    se = float(np.sqrt(s2 / sxx))
    return float(b), se, float(b / se) if se > 0 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--float-suffix", default="")
    ap.add_argument("--t1820-dir", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    d = Path(args.dump_dir)
    ids = [s for s in args.ids.split(",") if s]
    out = {"gates": {}, "ids": ids, "suffix": args.suffix}

    caps = {}
    print("=== gates ===")
    for pid in ids:
        eng = load_engine_t1836(d / f"{pid}{args.suffix}_engine.bin")
        flt = np.load(d / f"{pid}{args.float_suffix}_float.npy").astype(np.float64)
        ftok = np.load(d / f"{pid}{args.float_suffix}_float.tokens.npy")
        T = min(eng["n"], flt.shape[0], len(ftok))
        if not np.array_equal(ftok[:T], eng["consumed"][:T]):
            first = int(np.flatnonzero(ftok[:T] != eng["consumed"][:T])[0])
            raise SystemExit(f"{pid}: engine and float consumed different tokens, first at "
                             f"position {first} (engine {eng['consumed'][first]}, "
                             f"float {ftok[first]})")
        engv, q = dequant(eng["codes"][:T], eng["m"][:T], eng["e"][:T])
        caps[pid] = {"eng": engv, "q": q, "flt": flt[:T], "codes": eng["codes"][:T],
                     "prompt_len": eng["prompt_len"], "T": T,
                     "argmax": eng["argmax"], "consumed": eng["consumed"][:T]}
        print(f"  {pid}: token agreement over all {T} positions "
              f"({eng['prompt_len']} prompt + {T - eng['prompt_len']} generated) -- PASS")

    if args.t1820_dir:
        t1820 = Path(args.t1820_dir)
        rep = {}
        for pid in ids:
            p = t1820 / f"{pid}_engine.bin"
            if not p.exists():
                continue
            old = load_engine_t1820(p)
            c = caps[pid]
            n = min(old["n"], c["prompt_len"])
            same_codes = bool(np.array_equal(old["codes"][:n], c["codes"][:n]))
            rep[pid] = {"positions": n, "codes_identical": same_codes}
            print(f"  {pid}: prompt-position codes vs T-1820's own dump over {n} positions -- "
                  f"{'IDENTICAL' if same_codes else 'DIFFER'}")
        out["gates"]["t1820_replication"] = rep

    # ---------------------------------------------------------------- errors
    # err[t, s, :] = engine - float at position t, state s.
    pref, dec = [], []          # (pid, t) index lists
    for pid, c in caps.items():
        for t in range(c["T"]):
            (pref if t < c["prompt_len"] else dec).append((pid, t))
    print(f"\npopulation: {len(pref)} prompt positions, {len(dec)} generated positions, "
          f"{len(ids)} prompts")
    out["population"] = {"prefill": len(pref), "decode": len(dec), "prompts": len(ids)}

    def gather(idx, fn):
        return np.array([fn(caps[pid], t) for pid, t in idx])

    def rel_l2(c, t):
        e = c["eng"][t] - c["flt"][t]
        return np.linalg.norm(e, axis=-1) / np.linalg.norm(c["flt"][t], axis=-1)

    def cosine(c, t):
        a, b = c["eng"][t], c["flt"][t]
        return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))

    def rel_l2_single(c, t, s):
        e = c["eng"][t, s] - c["flt"][t, s]
        return float(np.linalg.norm(e) / np.linalg.norm(c["flt"][t, s]))

    def cos_single(c, t, s):
        a, b = c["eng"][t, s], c["flt"][t, s]
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))

    rel_pref = gather(pref, rel_l2)          # (n_pref, 29)
    rel_dec = gather(dec, rel_l2)
    cos_pref = gather(pref, cosine)
    cos_dec = gather(dec, cosine)

    print("\n=== (a1) magnitude: relative L2 by state, prompt vs generated ===")
    print(f"{'state':>6s} {'prefill':>10s} {'prefill>=1':>11s} {'decode':>10s} {'ratio':>7s} "
          f"{'cos pref':>9s} {'cos dec':>9s}")
    pref_pos = np.array([t for _pid, t in pref])
    mask1 = pref_pos >= 1
    mag = []
    for s in (0, 1, 5, 10, 15, 20, 24, 27, 28):
        a = float(np.median(rel_pref[:, s]))
        a1 = float(np.median(rel_pref[mask1, s]))
        b = float(np.median(rel_dec[:, s]))
        mag.append({"state": s, "prefill_median": a, "prefill_ge1_median": a1,
                    "decode_median": b, "ratio": b / a1 if a1 else float("nan"),
                    "cos_prefill_ge1": float(np.median(cos_pref[mask1, s])),
                    "cos_decode": float(np.median(cos_dec[:, s]))})
        print(f"{s:>6d} {a:10.4f} {a1:11.4f} {b:10.4f} {b / a1 if a1 else float('nan'):7.3f} "
              f"{np.median(cos_pref[mask1, s]):9.4f} {np.median(cos_dec[:, s]):9.4f}")
    out["magnitude_by_state"] = mag

    print("\n=== (a2) accumulation geometry: is every layer still injecting fresh noise? ===")

    def geometry(idx):
        incr, cosd, energy = [], [], []
        for pid, t in idx:
            c = caps[pid]
            e = c["eng"][t] - c["flt"][t]                      # (29, H)
            for k in range(1, e.shape[0] - 1):
                ek, ek1 = e[k], e[k + 1]
                dk = ek1 - ek
                nk = np.linalg.norm(ek)
                if nk == 0:
                    continue
                incr.append(np.linalg.norm(dk) / nk)
                cosd.append(float(ek @ dk / (nk * max(np.linalg.norm(dk), 1e-300))))
                energy.append((np.linalg.norm(ek1) ** 2) / (nk ** 2))
        return np.array(incr), np.array(cosd), np.array(energy)

    geo = {}
    for name, idx in (("prefill(pos>=1)", [(p, t) for p, t in pref if t >= 1]),
                      ("decode", dec)):
        incr, cosd, energy = geometry(idx)
        geo[name] = {"increment_median": float(np.median(incr)),
                     "cos_median": float(np.median(cosd)),
                     "cos_mean": float(cosd.mean()),
                     "cos_sd": float(cosd.std(ddof=1)),
                     "cos_se": float(cosd.std(ddof=1) / np.sqrt(len(cosd))),
                     "energy_ratio_median": float(np.median(energy)),
                     "n": int(len(incr))}
        g = geo[name]
        print(f"  {name:16s} increment ||d_k||/||e_k|| median {g['increment_median']:.3f}   "
              f"cos(e_k,d_k) median {g['cos_median']:+.3f} mean {g['cos_mean']:+.3f} "
              f"(SE {g['cos_se']:.4f})   energy ratio median {g['energy_ratio_median']:.3f}   "
              f"n={g['n']}")
    out["geometry"] = geo

    print("\n=== (a3) channel structure: flat, or concentrated? ===")

    def channel_structure(idx):
        dec_ratio, sig_ratio, top1 = [], [], []
        for pid, t in idx:
            c = caps[pid]
            for s in range(1, c["eng"].shape[1]):
                f = c["flt"][t, s]
                e = np.abs(c["eng"][t, s] - f)
                order = np.argsort(-np.abs(f))
                nd = max(1, len(f) // 10)
                hi, lo = order[:nd], order[-nd:]
                if e[lo].mean() > 0:
                    dec_ratio.append(e[hi].mean() / e[lo].mean())
                if np.abs(f[lo]).mean() > 0:
                    sig_ratio.append(np.abs(f[hi]).mean() / np.abs(f[lo]).mean())
                e2 = e ** 2
                k = max(1, len(f) // 100)
                top1.append(np.sort(e2)[-k:].sum() / max(e2.sum(), 1e-300))
        return np.array(dec_ratio), np.array(sig_ratio), np.array(top1)

    chan = {}
    for name, idx in (("prefill(pos>=1)", [(p, t) for p, t in pref if t >= 1]),
                      ("decode", dec)):
        dr, sr, t1 = channel_structure(idx)
        chan[name] = {"error_decile_ratio_median": float(np.median(dr)),
                      "signal_decile_ratio_median": float(np.median(sr)),
                      "top1pct_share_median": float(np.median(t1))}
        c_ = chan[name]
        print(f"  {name:16s} error decile ratio {c_['error_decile_ratio_median']:.2f}   "
              f"signal decile ratio {c_['signal_decile_ratio_median']:.1f}   "
              f"top-1% error share {c_['top1pct_share_median']:.3f}")
    out["channel_structure"] = chan

    print("\n=== (a2)/(a3) resolving power: is the DECODE-vs-PREFILL gap in the mechanism")
    print("     signatures themselves resolved, or is the pooled SE above hiding prompt-level")
    print("     correlation? (each layer/position within one prompt is not an independent")
    print("     sample, so the pooled n above overstates precision -- paired per prompt, n=6)")

    def per_prompt_geo_chan(pid_idx_fn):
        cos_mean, incr_med, energy_med, err_dec, top1 = [], [], [], [], []
        for pid in ids:
            idx = pid_idx_fn(pid)
            incr, cosd, energy = geometry(idx)
            dr, _sr, t1 = channel_structure(idx)
            cos_mean.append(float(cosd.mean()) if len(cosd) else float("nan"))
            incr_med.append(float(np.median(incr)) if len(incr) else float("nan"))
            energy_med.append(float(np.median(energy)) if len(energy) else float("nan"))
            err_dec.append(float(np.median(dr)) if len(dr) else float("nan"))
            top1.append(float(np.median(t1)) if len(t1) else float("nan"))
        return (np.array(cos_mean), np.array(incr_med), np.array(energy_med), np.array(err_dec),
                np.array(top1))

    pre_idx = lambda pid: [(p, t) for p, t in pref if p == pid and t >= 1]
    dec_idx = lambda pid: [(p, t) for p, t in dec if p == pid]
    pre_cos, pre_incr, pre_energy, pre_errdec, pre_top1 = per_prompt_geo_chan(pre_idx)
    dec_cos, dec_incr, dec_energy, dec_errdec, dec_top1 = per_prompt_geo_chan(dec_idx)

    def paired_rp_arrays(name, pre_arr, dec_arr):
        d = dec_arr - pre_arr
        n = len(d)
        mean_d, sd_d = float(d.mean()), float(d.std(ddof=1))
        se_d = sd_d / np.sqrt(n)
        rp95 = 1.96 * se_d
        t_d = mean_d / se_d if se_d > 0 else float("nan")
        resolved = bool(abs(mean_d) > rp95)
        print(f"  {name:34s} prefill {pre_arr.mean():+.4f}  decode {dec_arr.mean():+.4f}  "
              f"diff {mean_d:+.4f}  RP95 {rp95:.4f}  "
              f"{'RESOLVED' if resolved else 'NOT RESOLVED'} (t={t_d:+.2f})")
        return {"prefill_per_prompt": pre_arr.tolist(), "decode_per_prompt": dec_arr.tolist(),
                "mean_diff": mean_d, "se_diff": se_d, "resolving_power_95": rp95, "t": t_d,
                "resolved_at_n6": resolved}

    geo_chan_rp = {
        "cos_ek_dk_mean": paired_rp_arrays("cos(e_k,d_k) mean, all layers", pre_cos, dec_cos),
        "increment_median": paired_rp_arrays("increment ||d_k||/||e_k|| median", pre_incr, dec_incr),
        "energy_ratio_median": paired_rp_arrays("energy ratio median", pre_energy, dec_energy),
        "error_decile_ratio_median": paired_rp_arrays("channel error decile ratio median",
                                                        pre_errdec, dec_errdec),
        "top1pct_share_median": paired_rp_arrays("top-1% error share median", pre_top1, dec_top1),
    }
    out["geometry_channel_resolving_power"] = geo_chan_rp

    print("\n=== (b) drift against index: generated vs prompt, same fit, same index range ===")
    fits = {}
    for s in (5, 15, 27, 28):
        # decode: index within the generated sequence, g = 0..N-1
        xs_d, ys_d = [], []
        for (pid, t), row in zip(dec, rel_dec):
            xs_d.append(np.log1p(t - caps[pid]["prompt_len"]))
            ys_d.append(np.log(max(row[s], 1e-300)))
        bd, sed, td = slope_fit(xs_d, ys_d)
        # decode against ABSOLUTE position, which is what a K/V-accumulation term would
        # track (the context length the position attends over).
        xs_a, ys_a = [], []
        for (pid, t), row in zip(dec, rel_dec):
            xs_a.append(np.log1p(t))
            ys_a.append(np.log(max(row[s], 1e-300)))
        ba, sea, ta = slope_fit(xs_a, ys_a)
        # prefill control, positions >= 24 (T-1820 §7.5's own cell -- strictly after the
        # 24-token chat-template prefix every prompt shares).
        xs_p, ys_p = [], []
        for (pid, t), row in zip(pref, rel_pref):
            if t >= 24:
                xs_p.append(np.log1p(t))
                ys_p.append(np.log(max(row[s], 1e-300)))
        bp, sep, tp = slope_fit(xs_p, ys_p)
        fits[s] = {"decode_vs_gen_index": [bd, sed, td],
                   "decode_vs_abs_position": [ba, sea, ta],
                   "prefill_vs_abs_position": [bp, sep, tp]}
        print(f"  state {s:2d}: decode vs generation index  b={bd:+.3f} (SE {sed:.3f}, "
              f"{td:+.1f} sigma)")
        print(f"            decode vs absolute position b={ba:+.3f} (SE {sea:.3f}, "
              f"{ta:+.1f} sigma)")
        print(f"            prefill vs absolute position (pos>=24) b={bp:+.3f} (SE {sep:.3f}, "
              f"{tp:+.1f} sigma)")
    out["index_fits"] = {str(k): v for k, v in fits.items()}

    print("\n=== (c) the rail census, by position class ===")

    def rail_census(idx):
        railed, unrep, share, deficit = [], [], [], []
        for pid, t in idx:
            c = caps[pid]
            for s in range(1, c["eng"].shape[1]):
                code = c["codes"][t, s].astype(np.int64)
                q = c["q"][t, s]
                f = c["flt"][t, s]
                e = c["eng"][t, s] - f
                at_rail = np.abs(code) == 127
                railed.append(int(at_rail.sum()))
                unrep.append(int((np.abs(f) / max(q, 1e-300) > 127).sum()))
                e2 = e ** 2
                share.append(e2[at_rail].sum() / max(e2.sum(), 1e-300))
                ne = np.linalg.norm(e)
                deficit.append(np.linalg.norm(e[at_rail]) / max(ne, 1e-300))
        return (np.array(railed), np.array(unrep), np.array(share), np.array(deficit))

    classes = {
        "position 0": [(p, t) for p, t in pref if t == 0],
        "prompt >= 1": [(p, t) for p, t in pref if t >= 1],
        "first generated": [(p, t) for p, t in dec if t == caps[p]["prompt_len"]],
        "generated >= 1": [(p, t) for p, t in dec if t > caps[p]["prompt_len"]],
    }
    cen = {}
    for name, idx in classes.items():
        if not idx:
            continue
        r, u, sh, df = rail_census(idx)
        cen[name] = {"railed_per_vector_mean": float(r.mean()),
                     "unrepresentable_mean": float(u.mean()),
                     "railed_error_share_mean": float(sh.mean()),
                     "railed_error_share_max": float(sh.max()),
                     "range_deficit_frac_mean": float(df.mean()),
                     "n_states": int(len(r))}
        c_ = cen[name]
        print(f"  {name:18s} railed/vector {c_['railed_per_vector_mean']:.2f}   "
              f"unrepresentable {c_['unrepresentable_mean']:.2f}   "
              f"railed share of squared error {c_['railed_error_share_mean']:.3f} "
              f"(max {c_['railed_error_share_max']:.3f})   "
              f"range deficit {c_['range_deficit_frac_mean']:.3f}")
    out["rail_census"] = cen

    print("\n=== (c) resolving power: does the first GENERATED position show the first-token")
    print("     rail catastrophe's analog, against a per-prompt paired n=6 floor? ===")
    pos0_share = np.array([rail_census([(pid, 0)])[2].mean() if any(p == pid and t == 0 for p, t in pref)
                            else float("nan") for pid in ids])
    fg_share = np.array([rail_census([(pid, caps[pid]["prompt_len"])])[2].mean() for pid in ids])
    rp_rail = paired_rp_arrays("railed error share: position0 vs first-generated", fg_share, pos0_share)
    verdict_rail = ("the first generated position does NOT reproduce the first-token rail "
                     "catastrophe -- it behaves like an ordinary interior position" if
                     abs(rp_rail["mean_diff"]) > rp_rail["resolving_power_95"] and
                     fg_share.mean() < pos0_share.mean()
                     else "inconclusive at this resolution")
    print(f"  -> {verdict_rail}")
    out["rail_position0_vs_first_generated_rp"] = rp_rail

    print("\n=== (d) resolving power: per-prompt paired contrast, prefill(pos>=1) vs decode, n=6 ===")
    print("Extension over the tool as committed at a4f25d1 -- the tool reports pooled medians\n"
          "(a1/a3) and OLS-fit standard errors (b) but no resolving power for the headline\n"
          "prefill-vs-decode CONTRAST itself. This section adds it, matching the campaign's own\n"
          "convention (T-1795/T-1796/T-1819/T-1820: inter-prompt spread at n=6 is the noise\n"
          "floor) rather than inventing a new one. Paired per prompt because prefill and decode\n"
          "are measured on the SAME six prompts, not independent samples.")

    def per_prompt_metric(pid, t_indices, fn):
        vals = [fn(caps[pid], t) for t in t_indices if t < caps[pid]["T"]]
        return float(np.median(vals)) if vals else float("nan")

    def paired_rp(name, fn, extra=""):
        pre, dec_ = [], []
        for pid in ids:
            c = caps[pid]
            pl = c["prompt_len"]
            pre.append(per_prompt_metric(pid, range(1, pl), fn))
            dec_.append(per_prompt_metric(pid, range(pl, c["T"]), fn))
        pre = np.array(pre)
        dec_ = np.array(dec_)
        d = dec_ - pre
        n = len(d)
        mean_d = float(d.mean())
        sd_d = float(d.std(ddof=1))
        se_d = sd_d / np.sqrt(n)
        rp95 = 1.96 * se_d
        t_d = mean_d / se_d if se_d > 0 else float("nan")
        resolved = bool(abs(mean_d) > rp95)
        rec = {"prefill_per_prompt": pre.tolist(), "decode_per_prompt": dec_.tolist(),
               "mean_diff": mean_d, "sd_diff": sd_d, "se_diff": se_d,
               "resolving_power_95": rp95, "t": t_d, "resolved_at_n6": resolved}
        print(f"  {name:34s} prefill median-of-medians {pre.mean():.4f}  decode median-of-medians "
              f"{dec_.mean():.4f}  diff {mean_d:+.4f}  RP95 {rp95:.4f}  "
              f"{'RESOLVED' if resolved else 'NOT RESOLVED'} (t={t_d:+.2f}) {extra}")
        return rec

    rp = {}
    rp["rel_l2_state28"] = paired_rp("final-state (28) rel_l2", lambda c, t: rel_l2_single(c, t, 28))
    rp["rel_l2_state27"] = paired_rp("state 27 rel_l2", lambda c, t: rel_l2_single(c, t, 27))
    rp["rel_l2_state5"] = paired_rp("state 5 rel_l2", lambda c, t: rel_l2_single(c, t, 5))
    rp["cos_state28"] = paired_rp("final-state (28) cosine", lambda c, t: cos_single(c, t, 28))
    out["resolving_power_prefill_vs_decode"] = rp

    print("\n=== (a)/(d) discrimination claim, stated against the resolving power just computed ===")
    for key, label in (("rel_l2_state28", "final-state relative L2"),
                        ("rel_l2_state27", "state-27 relative L2"),
                        ("rel_l2_state5", "state-5 relative L2"),
                        ("cos_state28", "final-state cosine")):
        r = rp[key]
        verdict = ("decode differs from prefill" if r["resolved_at_n6"]
                   else "no difference detected at this resolution -- not a proof of no "
                        "difference at any resolution")
        print(f"  {label}: diff {r['mean_diff']:+.4f} against RP95 {r['resolving_power_95']:.4f} "
              f"at n=6 prompts -- {verdict}")

    print("\n=== per-position final-state relative L2, generated positions ===")
    for pid in ids:
        c = caps[pid]
        vals = [float(np.linalg.norm(c["eng"][t, 28] - c["flt"][t, 28])
                      / np.linalg.norm(c["flt"][t, 28]))
                for t in range(c["prompt_len"], c["T"])]
        print(f"  {pid}: " + " ".join(f"{v:.3f}" for v in vals))
        out.setdefault("per_prompt_decode_final_rel", {})[pid] = vals

    print("\n=== free-running token agreement (engine's own argmax vs the float's tokens) ===")
    agree = {}
    for pid in ids:
        c = caps[pid]
        pl = c["prompt_len"]
        n = 0
        for g in range(c["T"] - pl):
            if c["argmax"][pl - 1 + g] != c["consumed"][pl + g]:
                break
            n += 1
        agree[pid] = {"agreeing_prefix": n, "generated": c["T"] - pl}
        print(f"  {pid}: the engine's own greedy choice matches the float's continuation for "
              f"the first {n}/{c['T'] - pl} generated tokens")
    out["free_running_agreement"] = agree

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
