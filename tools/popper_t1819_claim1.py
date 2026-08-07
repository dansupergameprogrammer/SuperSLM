#!/usr/bin/env python3
"""T-1819 attack on claim 1: "the 28-layer residual accumulation is a rotation --
cosine drops while norm stays ~1".

Reads T-1795's own engine/float 29-state residual dumps and:

  1. Reproduces T-1795's pooled rel_l2 / cosine / norm_ratio table (independent code).
  2. Decomposes each state's error into the component PARALLEL to the float reference
     (radial / scaling) and the component ORTHOGONAL to it (tangential), and reports
     each as a share of squared error.  A "rotation" claim asserts the tangential share
     dominates; this is the quantity that was never computed.
  3. Measures whether the error is DENSE (rotation-like: mass spread over all channels)
     or CONCENTRATED (sparse spikes), by top-k share of squared error mass, against the
     same statistic for an isotropic-noise null of the same magnitude.
  4. Executes three counterexample mechanisms on the REAL float states, each calibrated
     to the state's own observed rel_l2, and reports the cosine / norm_ratio each
     produces:
       (a) DIAGONAL per-channel scaling, norm-matched exactly (a scaling, by definition,
           and provably not a rotation -- diag(d) with d not all equal is not orthogonal)
       (b) SPARSE spikes on k channels
       (c) DENSE isotropic additive noise
     If all three land inside the band T-1795 reports, the cosine/norm-ratio observation
     does not discriminate a rotation from its alternatives.
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
            parts = line.split()
            n = int(parts[0])
            v = np.array([float(x) for x in parts[1:1 + n]], dtype=np.float64)
            assert v.size == n
            rows.append(v)
    assert len(rows) == 29, len(rows)
    return np.stack(rows)


def metrics(e: np.ndarray, f: np.ndarray) -> dict:
    ne, nf = np.linalg.norm(e), np.linalg.norm(f)
    cos = float(e @ f / (ne * nf))
    rel = float(np.linalg.norm(e - f) / nf)
    return {"cos": cos, "ratio": float(ne / nf), "rel_l2": rel}


def radial_tangential(e: np.ndarray, f: np.ndarray) -> dict:
    """e = a*f + p, p _|_ f.  radial error = (a-1)*f, tangential error = p."""
    nf2 = float(f @ f)
    a = float(e @ f) / nf2
    p = e - a * f
    rad2 = (a - 1.0) ** 2 * nf2
    tan2 = float(p @ p)
    tot = rad2 + tan2
    return {
        "a_parallel": a,
        "radial_share_sq": rad2 / tot if tot > 0 else float("nan"),
        "tangential_share_sq": tan2 / tot if tot > 0 else float("nan"),
        "radial_rel": float(np.sqrt(rad2) / np.sqrt(nf2)),
        "tangential_rel": float(np.sqrt(tan2) / np.sqrt(nf2)),
    }


def topk_share(v: np.ndarray, frac: float) -> float:
    k = max(1, int(round(v.size * frac)))
    s = np.sort(v ** 2)[::-1]
    return float(s[:k].sum() / s.sum())


# ---------------- counterexample mechanisms ----------------

def ce_diagonal_norm_matched(f: np.ndarray, rel_target: float, rng, ratio_target: float):
    """e = D f, D = diag(d) -- a PURE PER-CHANNEL SCALING.  d is drawn iid, then
    rescaled so ||e||/||f|| equals the observed norm ratio EXACTLY.  Not a rotation:
    diag(d) is orthogonal only when every |d_i| == 1."""
    for sigma in np.linspace(0.01, 3.0, 600):
        d = 1.0 + sigma * rng.standard_normal(f.size)
        e = d * f
        e = e * (ratio_target * np.linalg.norm(f) / np.linalg.norm(e))
        if np.linalg.norm(e - f) / np.linalg.norm(f) >= rel_target:
            return e, sigma
    return e, sigma


def ce_sparse(f: np.ndarray, rel_target: float, rng, ratio_target: float, k: int = 24):
    """e = f + s, s supported on k channels only, then norm-matched."""
    idx = rng.choice(f.size, size=k, replace=False)
    s = np.zeros_like(f)
    s[idx] = rng.standard_normal(k) * np.abs(f).max()
    for scale in np.linspace(0.001, 50.0, 4000):
        e = f + scale * s
        e = e * (ratio_target * np.linalg.norm(f) / np.linalg.norm(e))
        if np.linalg.norm(e - f) / np.linalg.norm(f) >= rel_target:
            return e, scale
    return e, scale


def ce_dense_noise(f: np.ndarray, rel_target: float, rng, ratio_target: float):
    for scale in np.linspace(1e-4, 5.0, 4000):
        e = f + scale * rng.standard_normal(f.size) * np.linalg.norm(f) / np.sqrt(f.size)
        e = e * (ratio_target * np.linalg.norm(f) / np.linalg.norm(e))
        if np.linalg.norm(e - f) / np.linalg.norm(f) >= rel_target:
            return e, scale
    return e, scale


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", nargs="+", required=True)
    ap.add_argument("--float", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1819)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    eng = [read_states(Path(p)) for p in args.engine]
    flt = [read_states(Path(p)) for p in args.float]
    nP = len(eng)

    rows = []
    for idx in range(29):
        per = []
        for p in range(nP):
            e, f = eng[p][idx], flt[p][idx]
            m = metrics(e, f)
            rt = radial_tangential(e, f)
            err = e - f
            per.append({
                **m, **rt,
                "float_norm": float(np.linalg.norm(f)),
                "abs_err": float(np.linalg.norm(err)),
                "err_top1pct_share": topk_share(err, 0.01),
                "err_top10pct_share": topk_share(err, 0.10),
                "float_top1pct_share": topk_share(f, 0.01),
            })
        def agg(k):
            xs = [r[k] for r in per]
            return float(np.mean(xs)), float(np.std(xs))
        rows.append({"state": idx, **{k: agg(k)[0] for k in per[0]},
                     **{k + "_sd": agg(k)[1] for k in per[0]}})

    # isotropic-noise null for top-k share, dimension 1536
    null = rng.standard_normal((200, 1536))
    null_top1 = float(np.mean([topk_share(v, 0.01) for v in null]))
    null_top10 = float(np.mean([topk_share(v, 0.10) for v in null]))

    print("=== A: reproduction + radial/tangential decomposition (mean over %d prompts) ===" % nP)
    print(f"{'st':>3} {'rel_l2':>8} {'cos':>8} {'ratio':>7} | {'tan_share':>9} {'rad_share':>9} "
          f"{'rad_rel':>8} | {'err_top1%':>9} {'flt_top1%':>9}")
    for r in rows:
        print(f"{r['state']:>3} {r['rel_l2']:>8.4f} {r['cos']:>8.4f} {r['ratio']:>7.4f} | "
              f"{r['tangential_share_sq']:>9.4f} {r['radial_share_sq']:>9.4f} {r['radial_rel']:>8.4f} | "
              f"{r['err_top1pct_share']:>9.4f} {r['float_top1pct_share']:>9.4f}")
    print(f"isotropic-noise null: top1%%={null_top1:.4f} top10%%={null_top10:.4f}")

    print()
    print("=== B: counterexample mechanisms on the REAL float states ===")
    print("each calibrated to that state's own observed rel_l2 and norm ratio")
    print(f"{'st':>3} {'obs_cos':>8} {'obs_rat':>8} | {'diagCOS':>8} {'diagRAT':>8} | "
          f"{'sparCOS':>8} {'sparRAT':>8} | {'noisCOS':>8} {'noisRAT':>8}")
    ce_rows = []
    for idx in range(29):
        f = flt[0][idx]
        rel = rows[idx]["rel_l2"]
        rat = rows[idx]["ratio"]
        if np.linalg.norm(f) == 0:
            continue
        ed, _ = ce_diagonal_norm_matched(f, rel, rng, rat)
        es, _ = ce_sparse(f, rel, rng, rat)
        en, _ = ce_dense_noise(f, rel, rng, rat)
        md, ms, mn = metrics(ed, f), metrics(es, f), metrics(en, f)
        ce_rows.append({"state": idx, "obs": {"cos": rows[idx]["cos"], "ratio": rat, "rel_l2": rel},
                        "diagonal": md, "sparse": ms, "dense_noise": mn})
        print(f"{idx:>3} {rows[idx]['cos']:>8.4f} {rat:>8.4f} | {md['cos']:>8.4f} {md['ratio']:>8.4f} | "
              f"{ms['cos']:>8.4f} {ms['ratio']:>8.4f} | {mn['cos']:>8.4f} {mn['ratio']:>8.4f}")

    out = {"per_state": rows, "counterexamples": ce_rows,
           "null_topk": {"top1pct": null_top1, "top10pct": null_top10}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
