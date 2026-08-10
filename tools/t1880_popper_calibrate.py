#!/usr/bin/env python3
r"""T-1880 -- calibrate T-1879's deciding metric against constructions of known inertness,
and place each of the four arms on a magnitude-only reference curve.

Two constructions are built from T-1879's own `null` capture, zero GPU:

  A. RANK-PRESERVING (exactly inert in the graded outcome, arbitrarily large in drift):
     each document's pooled vector is multiplied by its own positive scalar. Cosine
     similarity between any two documents is invariant under per-document positive
     scaling, so every ranking -- and therefore recall@1 -- is unchanged, while squared
     drift against the reference can be set to any value. The deciding rule must score
     this below its own threshold at every drift level, or it is not reading rank.

  B. ISOTROPIC (moves in a direction with no structure): each document's pooled vector
     gets independent Gaussian noise scaled to a target relative squared drift. Sweeping
     the target drift gives a magnitude-only reference curve -- how much recall@1 an
     unstructured perturbation of that size costs. Each of T-1879's four arms is then
     placed on that curve at its own measured drift, which separates "how far the pooled
     representation moved" from "in which direction it moved".

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
NICE = {"null": "null", "only04_k_proj_landing": "onlyK_ownscale",
        "only05_v_proj_landing": "onlyV_ownscale", "onlyK_vscale": "onlyK_vscale",
        "onlyV_kscale": "onlyV_kscale"}
FW_Z = 2.241403


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
    vecs, fps, r1 = {}, {}, {}
    for arm in ARMS:
        z = np.load(cell / "pooled" / f"{arm}.npz", allow_pickle=True)
        labs = [str(x) for x in z["labels"]]
        vecs[arm] = {l: np.asarray(z["vectors"][i], dtype=np.float64) for i, l in enumerate(labs)}
        fps[arm] = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
        res, _n = rr.compare_arms(ref_vecs, vecs[arm], domains, ref_fps, fps[arm])
        r1[arm] = np.array([x.recall[1] for x in res])
        docs = [x.label for x in res]
    N = len(docs)

    refn = {l: float(np.sum(ref_vecs[l] ** 2)) for l in docs}

    def rel_drift(v):
        return np.array([float(np.sum((v[l] - ref_vecs[l]) ** 2)) / refn[l] for l in docs])

    arm_drift = {a: float(rel_drift(vecs[a]).mean()) for a in ARMS}
    null_r1 = r1["null"].mean()

    def grade(v):
        res, _ = rr.compare_arms(ref_vecs, v, domains, ref_fps, {})
        return np.array([x.recall[1] for x in res])

    print(f"N={N}  null recall@1={null_r1:.6f}  ({int(r1['null'].sum())}/{N})")
    print("arm mean relative squared drift against the float reference:")
    for a in ARMS:
        print(f"  {NICE[a]:16s} {arm_drift[a]:.6g}   recall@1={r1[a].mean():.6f} "
              f"cost={null_r1 - r1[a].mean():+.6f}")

    print("\n=== CONSTRUCTION A: rank-preserving (exactly inert in the graded outcome) ===")
    print("   per-document positive scaling; cosine ranking is invariant by construction")
    rowsA = []
    for target in (1e-3, 1.1e-3, 1e-2, 0.116, 0.22, 1.9, 10.0):
        rng = np.random.default_rng(20260810)
        pert = {}
        for l in docs:
            # v -> c*v with c>0 chosen so ||c*v - ref||^2 / ||ref||^2 ~= target.
            # solve for c on the ray: minimise nothing, just take c = 1 + s with s from target
            v = vecs["null"][l]
            nv = float(np.sum(v * v))
            dot = float(np.sum(v * ref_vecs[l]))
            # ||c v - ref||^2 = c^2 nv - 2 c dot + refn ; want = target*refn
            aa, bb, cc = nv, -2.0 * dot, refn[l] - target * refn[l]
            disc = bb * bb - 4 * aa * cc
            c = (-bb + np.sqrt(disc)) / (2 * aa) if disc >= 0 else 1.0
            if c <= 0:
                c = 1.0
            pert[l] = c * v
        got = grade(pert)
        d = got - r1["null"]
        se = d.std(ddof=1) / np.sqrt(N)
        rp = FW_Z * d.std(ddof=1) / np.sqrt(N)
        verdict = "RESOLVED" if abs(d.mean()) > rp and rp > 0 else "not resolved (inert, correct)"
        print(f"  target rel drift {target:<9.4g} achieved {rel_drift(pert).mean():<9.4g} "
              f"recall@1={got.mean():.6f} delta_vs_null={d.mean():+.6f} SE={se:.6f} -> {verdict}")
        rowsA.append((target, float(rel_drift(pert).mean()), float(got.mean()), float(d.mean())))

    print("\n=== CONSTRUCTION B: isotropic perturbation -- magnitude-only reference curve ===")
    rowsB = []
    for target in (1e-4, 3e-4, 1e-3, 1.11e-3, 3e-3, 1e-2, 3e-2, 0.116, 0.218, 0.5, 1.0, 1.896, 4.0):
        costs = []
        for s in range(args.seeds):
            rng = np.random.default_rng(1880 * 1000 + s)
            pert = {}
            for l in docs:
                g = rng.standard_normal(vecs["null"][l].shape)
                g *= np.sqrt(target * refn[l] / float(np.sum(g * g)))
                pert[l] = vecs["null"][l] + g
            got = grade(pert)
            costs.append(float(null_r1 - got.mean()))
        costs = np.array(costs)
        print(f"  rel drift {target:<9.4g} isotropic recall@1 cost = {costs.mean():+.6f} "
              f"(sd over {args.seeds} seeds {costs.std(ddof=1):.6f})")
        rowsB.append((target, float(costs.mean()), float(costs.std(ddof=1))))

    print("\n=== placing each arm on the magnitude-only curve ===")
    tg = np.array([r[0] for r in rowsB])
    cv = np.array([r[1] for r in rowsB])
    for a in ARMS[1:]:
        d = arm_drift[a]
        pred = float(np.interp(np.log10(d), np.log10(tg), cv))
        actual = null_r1 - r1[a].mean()
        print(f"  {NICE[a]:16s} rel drift {d:<9.4g} isotropic-equivalent cost {pred:+.6f}  "
              f"measured cost {actual:+.6f}  ratio measured/isotropic = "
              f"{actual/pred if pred else float('nan'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
