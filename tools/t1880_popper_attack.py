#!/usr/bin/env python3
r"""T-1880 -- disposable attack probe against T-1879's result packet.

Reads T-1879's five width-1 pooled captures and T-1777's committed true-bf16 reference and
recomputes, independently of T-1879's own grading script:

  1. per-arm recall@1 (verifies the packet's headline figures);
  2. every paired contrast with its OWN achieved paired resolving power and discordant-pair
     (McNemar) counts, rather than a planning-sigma family-wise figure;
  3. the reproduction gate's two legs against their own achieved resolving power;
  4. the 2x2's identifiability algebra (what the four cells can and cannot separate);
  5. the same 2x2 on squared drift energy -- the metric T-1835's own record states is the
     additive one, where recall@1 is not;
  6. the id/ood domain split of both decision-bearing contrasts.

Read-only on every T-1879 and T-1777 path. Writes nothing but stdout / --json-out.
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


def paired(a, b, label, z_crit=1.959964, fw_z=2.241403):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    # achieved (not planning) resolving power, per-contrast and family-wise at m=2
    rp_own = z_crit * sd / np.sqrt(n)
    rp_fw_own = fw_z * sd / np.sqrt(n)
    n10 = int(((d > 0)).sum())
    n01 = int(((d < 0)).sum())
    # exact McNemar (binomial) on the discordant pairs
    nd = n10 + n01
    p_exact = float(stats.binomtest(min(n10, n01), nd, 0.5).pvalue) if nd else 1.0
    return {
        "label": label, "n": n, "mean": mean, "sd": sd, "se": se,
        "z": mean / se if se else float("nan"),
        "rp_paired_achieved_1.96": rp_own,
        "rp_paired_achieved_fw_m2": rp_fw_own,
        "resolved_vs_own_rp": abs(mean) > rp_own,
        "resolved_vs_own_fw_rp": abs(mean) > rp_fw_own,
        "discordant_a_better": n10, "discordant_b_better": n01,
        "mcnemar_exact_p": p_exact,
    }


def show(c):
    print(f"  {c['label']:52s} mean={c['mean']:+.6f} sd={c['sd']:.4f} SE={c['se']:.6f} "
          f"z={c['z']:+7.2f}")
    print(f"  {'':52s} achieved paired RP(1.96)={c['rp_paired_achieved_1.96']:.4f} "
          f"RP_fw(m=2)={c['rp_paired_achieved_fw_m2']:.4f} "
          f"-> {'RESOLVED' if c['resolved_vs_own_fw_rp'] else 'NOT RESOLVED / UNDERPOWERED'} (family-wise)")
    print(f"  {'':52s} discordant: {c['discordant_a_better']}+ / {c['discordant_b_better']}- "
          f"McNemar exact p={c['mcnemar_exact_p']:.3g}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1777-out", required=True)
    ap.add_argument("--json-out", default=None)
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
    r1, order, vecs = {}, {}, {}
    for arm in ARMS:
        z = np.load(cell / "pooled" / f"{arm}.npz", allow_pickle=True)
        labs = [str(x) for x in z["labels"]]
        v = {l: z["vectors"][i] for i, l in enumerate(labs)}
        fp = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
        res, n = rr.compare_arms(ref_vecs, v, domains, ref_fps, fp)
        order[arm] = [x.label for x in res]
        r1[arm] = np.array([x.recall[1] for x in res])
        vecs[arm] = v
    for arm in ARMS[1:]:
        assert order[arm] == order[ARMS[0]], "document order differs across arms"
    docs = order["null"]
    N = len(docs)
    dom = np.array([domains[l] for l in docs])

    print(f"=== ATTACK 1: per-arm recall@1, N={N} (independent recomputation) ===")
    for a in ARMS:
        print(f"  {NICE[a]:16s} recall@1 = {r1[a].mean():.6f}   ({int(r1[a].sum())}/{N} documents)")

    print("\n=== ATTACK 2: reproduction-gate legs against their OWN achieved resolving power ===")
    gate = {}
    for a in ("only04_k_proj_landing", "only05_v_proj_landing"):
        c = paired(r1["null"], r1[a], f"cost({NICE[a]}) = null - {NICE[a]}")
        gate[a] = c
        show(c)
    print("  T-1835 S1 filed: site4 +0.2008 (its own paired RP 0.0522); site5 +0.0418 (RP 0.0279)")
    kc, vc = gate["only04_k_proj_landing"]["mean"], gate["only05_v_proj_landing"]["mean"]
    print(f"  ratio here = {kc/vc:.2f}x   (filed ratio 0.2008/0.0418 = {0.2008/0.0418:.2f}x)")
    # what the ratio's denominator uncertainty does to the ratio
    lo = vc - gate["only05_v_proj_landing"]["rp_paired_achieved_fw_m2"]
    hi = vc + gate["only05_v_proj_landing"]["rp_paired_achieved_fw_m2"]
    print(f"  V cost +/- its own family-wise RP: [{lo:+.6f}, {hi:+.6f}] "
          f"-> ratio range [{kc/hi:.2f}x, {'inf' if lo <= 0 else f'{kc/lo:.2f}x'}]")

    print("\n=== ATTACK 3: the four contrasts, achieved RP + exact McNemar ===")
    contrasts = {
        "role_effect_at_V_scale (onlyK_vscale - onlyV_ownscale)":
            (r1["onlyK_vscale"], r1["only05_v_proj_landing"]),
        "role_effect_at_K_scale (onlyK_ownscale - onlyV_kscale)":
            (r1["only04_k_proj_landing"], r1["onlyV_kscale"]),
        "scale_effect_at_K_role (onlyK_ownscale - onlyK_vscale)":
            (r1["only04_k_proj_landing"], r1["onlyK_vscale"]),
        "scale_effect_at_V_role (onlyV_kscale - onlyV_ownscale)":
            (r1["onlyV_kscale"], r1["only05_v_proj_landing"]),
    }
    cres = {}
    for k, (a, b) in contrasts.items():
        c = paired(a, b, k)
        cres[k] = c
        show(c)

    print("\n=== ATTACK 4: identifiability of the 2x2 (costs against this run's own null) ===")
    cost = {a: float(r1["null"].mean() - r1[a].mean()) for a in ARMS[1:]}
    for a, v in cost.items():
        print(f"  cost({NICE[a]:16s}) = {v:+.6f}")
    R = cost["only04_k_proj_landing"] - cost["only05_v_proj_landing"]
    M_K = cost["onlyK_vscale"] - cost["only04_k_proj_landing"]
    M_V = cost["onlyV_kscale"] - cost["only05_v_proj_landing"]
    c1 = cost["onlyK_vscale"] - cost["only05_v_proj_landing"]
    c2 = cost["only04_k_proj_landing"] - cost["onlyV_kscale"]
    print(f"  own-scale gap            R   = cost(K_own) - cost(V_own)      = {R:+.6f}")
    print(f"  mis-scale penalty on K   M_K = cost(K_vscale) - cost(K_own)   = {M_K:+.6f}")
    print(f"  mis-scale penalty on V   M_V = cost(V_kscale) - cost(V_own)   = {M_V:+.6f}")
    print(f"  role_effect_at_V_scale (cost units) = R + M_K = {R + M_K:+.6f}  (measured {c1:+.6f})")
    print(f"  role_effect_at_K_scale (cost units) = R - M_V = {R - M_V:+.6f}  (measured {c2:+.6f})")
    print(f"  => the pre-registered confirmation region (both cost-contrasts > 0) requires")
    print(f"     R > M_V = {M_V:.6f} and R > -M_K = {-M_K:+.6f};")
    print(f"     i.e. a role effect larger than {M_V/abs(R):.2f}x the entire own-scale gap it explains.")

    print("\n=== ATTACK 5: the same 2x2 on squared drift energy (the additive metric) ===")
    # per-document squared L2 drift of the arm's pooled vector against the float reference,
    # and the same normalised by the reference's own squared norm.
    drift, ndrift = {}, {}
    for a in ARMS:
        d = np.array([float(np.sum((vecs[a][l] - ref_vecs[l]) ** 2)) for l in docs])
        nrm = np.array([float(np.sum(ref_vecs[l] ** 2)) for l in docs])
        drift[a] = d
        ndrift[a] = d / nrm
    for a in ARMS:
        print(f"  {NICE[a]:16s} mean sq drift = {drift[a].mean():.6g}   "
              f"mean relative sq drift = {ndrift[a].mean():.6g}")
    dc = {a: ndrift[a] - ndrift["null"] for a in ARMS[1:]}
    dR = float(dc["only04_k_proj_landing"].mean() - dc["only05_v_proj_landing"].mean())
    dMK = float(dc["onlyK_vscale"].mean() - dc["only04_k_proj_landing"].mean())
    dMV = float(dc["onlyV_kscale"].mean() - dc["only05_v_proj_landing"].mean())
    print(f"  relative-drift own-scale gap R_d   = {dR:+.6g}")
    print(f"  relative-drift mis-scale M_K       = {dMK:+.6g}")
    print(f"  relative-drift mis-scale M_V       = {dMV:+.6g}")
    print("  the two role contrasts on relative squared drift:")
    dres = {}
    for lab, (a, b) in (("role_effect_at_V_scale [drift]", ("onlyK_vscale", "only05_v_proj_landing")),
                        ("role_effect_at_K_scale [drift]", ("only04_k_proj_landing", "onlyV_kscale")),
                        ("scale_effect_at_K_role [drift]", ("only04_k_proj_landing", "onlyK_vscale")),
                        ("scale_effect_at_V_role [drift]", ("onlyV_kscale", "only05_v_proj_landing"))):
        c = paired(ndrift[a], ndrift[b], lab)
        dres[lab] = c
        print(f"    {lab:34s} mean={c['mean']:+.6g} SE={c['se']:.4g} z={c['z']:+.2f} "
              f"discordant {c['discordant_a_better']}+/{c['discordant_b_better']}-")

    print("\n=== ATTACK 6: id / ood domain split of the two decision-bearing contrasts ===")
    for name, (a, b) in list(contrasts.items())[:2]:
        print(f"  {name}")
        for d in ("id", "ood"):
            m = dom == d
            c = paired(a[m], b[m], f"    {d} (n={int(m.sum())})")
            print(f"    {d:4s} n={int(m.sum()):3d} mean={c['mean']:+.6f} SE={c['se']:.6f} "
                  f"z={c['z']:+7.2f} achieved RP_fw={c['rp_paired_achieved_fw_m2']:.4f} "
                  f"-> {'RESOLVED' if c['resolved_vs_own_fw_rp'] else 'NOT RESOLVED'}")
        for d in ("id", "ood"):
            m = dom == d
            print(f"    {d:4s} per-arm recall@1: " + "  ".join(
                f"{NICE[x]}={r1[x][m].mean():.4f}" for x in ARMS))

    print("\n=== ATTACK 7: bit-identity between arms (did the toggles change anything?) ===")
    for a in ARMS[1:]:
        same = sum(1 for l in docs if np.array_equal(vecs[a][l], vecs["null"][l]))
        print(f"  {NICE[a]:16s} pooled vectors bit-identical to null on {same}/{N} documents")

    if args.json_out:
        payload = {
            "n": N,
            "per_arm_recall1": {NICE[a]: float(r1[a].mean()) for a in ARMS},
            "per_arm_correct_docs": {NICE[a]: int(r1[a].sum()) for a in ARMS},
            "gate_legs": {NICE[a]: gate[a] for a in gate},
            "contrasts": cres,
            "algebra": {"R": R, "M_K": M_K, "M_V": M_V,
                        "c1_cost_units": c1, "c2_cost_units": c2},
            "drift": {"per_arm_mean_rel_sq_drift": {NICE[a]: float(ndrift[a].mean()) for a in ARMS},
                       "contrasts": dres},
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
