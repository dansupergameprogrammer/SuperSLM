#!/usr/bin/env python3
r"""T-1880 -- run T-1879's own pre-registered decision rule on a 2x2 in which the role
factor does not exist.

Four synthetic arms are built from T-1879's `null` capture. Each document's pooled vector
gets an INDEPENDENT ISOTROPIC GAUSSIAN displacement whose squared norm is set to that
document's own measured displacement in the corresponding real arm. So the synthetic 2x2
matches the real 2x2 document-for-document in displacement MAGNITUDE and carries no role
structure, no scale structure, and no consumer function at all -- the only thing that
distinguishes the four synthetic cells is how far each document moved.

T-1879's decision rule (T-1877 S4/S5: paired contrasts, m=2 Bonferroni z_crit=2.241403,
achieved family-wise RP at sigma_d=0.50, delta_min in {0.06, 0.10}) is then applied to the
synthetic 2x2 unchanged. If it returns the same verdict pattern it returned on the real
capture, the rule cannot distinguish "the role factor carries no independent weight" from
"the four cells have different displacement magnitudes".

Read-only on every T-1879 and T-1777 path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ("null", "only04_k_proj_landing", "only05_v_proj_landing", "onlyK_vscale", "onlyV_kscale")
NICE = {"only04_k_proj_landing": "onlyK_ownscale", "only05_v_proj_landing": "onlyV_ownscale",
        "onlyK_vscale": "onlyK_vscale", "onlyV_kscale": "onlyV_kscale", "null": "null"}
Z_CRIT = float(stats.norm.ppf(1 - 0.05 / (2 * 2)))
DELTA_MINS = {"delta_min=0.06": 0.06, "delta_min=0.10": 0.10}


def verdict_of(mean, se, achieved_rp):
    sr = mean / (Z_CRIT * se) if se > 0 else float("nan")
    outs = {}
    for lab, dm in DELTA_MINS.items():
        if abs(mean) < achieved_rp:
            outs[lab] = "UNDERPOWERED"
        elif sr > 1 and mean >= dm:
            outs[lab] = "RESOLVED-RECOVERY"
        elif sr < -1 and mean <= -dm:
            outs[lab] = "RESOLVED-REGRESSION"
        else:
            outs[lab] = "NOT RESOLVED"
    return sr, outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1777-out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    import t1777_retrieval_report as rr

    out = Path(args.t1777_out)
    labels, domains = [], {}
    with open(out / "t1777_corpus" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                labels.append(rec["label"])
                domains[rec["label"]] = rec.get("domain", rec.get("population", "ALL"))
    ref_vecs, ref_fps = rr.load_pooled_vectors(labels, out / "t1777_full_float_bf16", "float")

    cell = Path(args.cell)
    vecs, r1 = {}, {}
    docs = None
    for arm in ARMS:
        z = np.load(cell / "pooled" / f"{arm}.npz", allow_pickle=True)
        labs = [str(x) for x in z["labels"]]
        vecs[arm] = {l: np.asarray(z["vectors"][i], dtype=np.float64) for i, l in enumerate(labs)}
        fp = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
        res, _ = rr.compare_arms(ref_vecs, vecs[arm], domains, ref_fps, fp)
        r1[arm] = np.array([x.recall[1] for x in res])
        docs = [x.label for x in res]
    N = len(docs)

    # per-document displacement magnitude of each real arm against the float reference
    mag = {a: np.array([float(np.linalg.norm(vecs[a][l] - ref_vecs[l])) for l in docs])
           for a in ARMS}

    print(f"N={N}  real null recall@1={r1['null'].mean():.6f}")
    print("\nreal per-arm recall@1 and per-document displacement magnitude (mean):")
    for a in ARMS:
        print(f"  {NICE[a]:16s} recall@1={r1[a].mean():.6f}  mean||v-ref||={mag[a].mean():.4f}")

    achieved_rp = Z_CRIT * 0.50 / np.sqrt(N)
    print(f"\nz_crit(m=2)={Z_CRIT:.6f}  achieved family-wise RP (sigma_d=0.50, N={N})={achieved_rp:.4f}"
          "   -- T-1879's own values, unchanged")

    per_seed = []
    for s in range(args.seeds):
        rng = np.random.default_rng(18800 + s)
        syn = {}
        for a in ARMS:
            pert = {}
            for i, l in enumerate(docs):
                g = rng.standard_normal(ref_vecs[l].shape)
                g *= mag[a][i] / float(np.linalg.norm(g))
                pert[l] = ref_vecs[l] + g
            res, _ = rr.compare_arms(ref_vecs, pert, domains, ref_fps, {})
            assert [x.label for x in res] == docs
            syn[a] = np.array([x.recall[1] for x in res])
        per_seed.append(syn)

    print(f"\n=== synthetic 2x2 (isotropic displacement, magnitude-matched per document, "
          f"{args.seeds} seeds) ===")
    print("   the role factor DOES NOT EXIST in this construction")
    for a in ARMS:
        v = np.array([s[a].mean() for s in per_seed])
        print(f"  {NICE[a]:16s} synthetic recall@1 = {v.mean():.6f} "
              f"(sd over seeds {v.std(ddof=1):.6f})   real = {r1[a].mean():.6f}")

    contrasts = (("role_effect_at_V_scale", "onlyK_vscale", "only05_v_proj_landing"),
                 ("role_effect_at_K_scale", "only04_k_proj_landing", "onlyV_kscale"),
                 ("scale_effect_at_K_role", "only04_k_proj_landing", "onlyK_vscale"),
                 ("scale_effect_at_V_role", "onlyV_kscale", "only05_v_proj_landing"))

    print("\n=== T-1879's decision rule applied to the synthetic 2x2 ===")
    for name, a, b in contrasts:
        means, ses, srs, vds = [], [], [], []
        for s in per_seed:
            d = s[a] - s[b]
            m = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(N))
            sr, vd = verdict_of(m, se, achieved_rp)
            means.append(m); ses.append(se); srs.append(sr); vds.append(vd["delta_min=0.06"])
        print(f"  {name:24s} mean={np.mean(means):+.6f} (sd {np.std(means, ddof=1):.6f}) "
              f"SE={np.mean(ses):.6f} z={np.mean(means)/np.mean(ses):+.2f} "
              f"signed_ratio={np.mean(srs):+.2f}")
        print(f"  {'':24s} verdicts across seeds: {sorted(set(vds))}")

    print("\n=== the falsification criterion, evaluated on the synthetic 2x2 ===")
    for si, s in enumerate(per_seed):
        c1 = float((s['onlyK_vscale'] - s['only05_v_proj_landing']).mean())
        c2 = float((s['only04_k_proj_landing'] - s['onlyV_kscale']).mean())
        same = (c1 > 0) == (c2 > 0)
        print(f"  seed {si}: role_effect_at_V_scale={c1:+.6f}  role_effect_at_K_scale={c2:+.6f}  "
              f"same sign={same}  -> "
              f"{'CONFIRMS' if same else 'REFUTES (opposite directions)'} under T-1877 S2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
