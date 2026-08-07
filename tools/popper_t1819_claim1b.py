#!/usr/bin/env python3
"""T-1819 claim-1 follow-up: two measurements that DO discriminate between an
operator-like drift ("a rotation") and additive quantization noise, neither of which
T-1795 computed.

  1. Per-channel error-vs-signal scaling.  A rotation or a per-channel scaling is a
     LINEAR map: |err_i| scales with |f_i| across channels.  Additive quantization noise
     is set by the tensor's own quantization step: |err_i| is roughly flat in i and
     UNCORRELATED with |f_i|.  Reported as Spearman rank correlation of |err_i| against
     |f_i| across the 1536 channels, plus the ratio of mean |err| in the top-decile
     channels (by |f|) to the bottom-decile.

  2. Cross-prompt error-direction sharing.  If the drift is one operator R acting at a
     layer, err_p = (R - I) f_p for every prompt p, so the error directions inherit the
     states' own mutual geometry.  Compare cos(err_p, err_q) to cos(f_p, f_q).  Errors
     mutually orthogonal while the states are strongly aligned is evidence against a
     shared operator.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


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

    eng = [read_states(Path(p)) for p in a.engine]
    flt = [read_states(Path(p)) for p in a.float]
    nP = len(eng)

    print("=== 1. per-channel error vs signal (does |err_i| track |f_i|?) ===")
    print(f"{'st':>3} {'spearman':>9} {'topdec/botdec |err|':>20} {'meanerr_top':>12} {'meanerr_bot':>12}")
    rows = []
    for idx in range(29):
        rho, ratio, mt, mb = [], [], [], []
        for p in range(nP):
            f = flt[p][idx]
            err = np.abs(eng[p][idx] - f)
            af = np.abs(f)
            rho.append(stats.spearmanr(af, err).statistic)
            order = np.argsort(af)
            d = af.size // 10
            bot = err[order[:d]].mean()
            top = err[order[-d:]].mean()
            mt.append(top); mb.append(bot)
            ratio.append(top / bot if bot > 0 else np.nan)
        rows.append({"state": idx, "spearman": float(np.mean(rho)),
                     "top_bot_ratio": float(np.nanmean(ratio)),
                     "mean_err_topdecile": float(np.mean(mt)),
                     "mean_err_botdecile": float(np.mean(mb))})
        print(f"{idx:>3} {np.mean(rho):>9.4f} {np.nanmean(ratio):>20.3f} "
              f"{np.mean(mt):>12.5f} {np.mean(mb):>12.5f}")

    print()
    print("=== 2. cross-prompt geometry: errors vs states ===")
    print(f"{'st':>3} {'cos(f_p,f_q)':>13} {'cos(err_p,err_q)':>17}")
    xrows = []
    for idx in range(29):
        cf, ce = [], []
        for p, q in combinations(range(nP), 2):
            fp, fq = flt[p][idx], flt[q][idx]
            ep, eq = eng[p][idx] - fp, eng[q][idx] - fq
            cf.append(fp @ fq / (np.linalg.norm(fp) * np.linalg.norm(fq)))
            ce.append(ep @ eq / (np.linalg.norm(ep) * np.linalg.norm(eq)))
        xrows.append({"state": idx, "cos_states": float(np.mean(cf)),
                      "cos_errors": float(np.mean(ce))})
        print(f"{idx:>3} {np.mean(cf):>13.4f} {np.mean(ce):>17.4f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"channel_scaling": rows, "cross_prompt": xrows}, indent=2))
    print("written:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
