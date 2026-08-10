#!/usr/bin/env python3
r"""T-1879 -- grade the K-vs-V role x scale swap on the encoder oracle.

Reads the five width-1 pooled captures (`null`, `only04_k_proj_landing`,
`only05_v_proj_landing`, `onlyK_vscale`, `onlyV_kscale`) T-1879 captured with
`tools/t1835_site_toggle_dump.py --single-arm <name>`, one config per invocation, and
grades them per
`Claude/Vitruvius/t1877-kv-asymmetry-measurement-pricing-2026-08-09.md` S4/S5:

  * the reproduction gate -- `only04_k_proj_landing`/`only05_v_proj_landing` at width 1
    must reproduce T-1835 S1's direction and rough magnitude, adjusted for the known
    width-1 shift (D-SLM2141);
  * the two pre-registered, decision-bearing contrasts (m=2):
        role_effect_at_V_scale  = onlyK_vscale     - only05_v_proj_landing
        role_effect_at_K_scale  = only04_k_proj_landing - onlyV_kscale
  * the two informative-only contrasts completing the 2x2:
        scale_effect_at_K_role  = only04_k_proj_landing - onlyK_vscale
        scale_effect_at_V_role  = onlyV_kscale     - only05_v_proj_landing

The decision rule is T-1870 S6's signed family-wise ratio, z_crit recomputed for m=2, with
delta_min stated explicitly rather than assumed, and the UNDERPOWERED-vs-NOT-RESOLVED
distinction preserved (an effect below the achieved resolving power prints UNDERPOWERED,
never NOT RESOLVED, per `StandardsDocument.md` S5.4).

Usage
    python tools\t1879_grade_kv_swap.py --cell out\t1879_capture \
        --t1777-tools D:\SuperSLM\.worktrees\t1777-retrieval-agreement\tools \
        --t1777-out D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out \
        --json-out out\t1879_grades.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REQUIRED_ARMS = (
    "null",
    "only04_k_proj_landing",   # onlyK_ownscale
    "only05_v_proj_landing",   # onlyV_ownscale
    "onlyK_vscale",
    "onlyV_kscale",
)


def _load_arm_recall1(cell: Path, arm: str, rr, ref_vecs, ref_fps, domains):
    p = cell / "pooled" / f"{arm}.npz"
    if not p.exists():
        raise SystemExit(f"missing pooled capture for {arm!r}: {p}")
    z = np.load(p, allow_pickle=True)
    labs = [str(x) for x in z["labels"]]
    vecs = {l: z["vectors"][i] for i, l in enumerate(labs)}
    fps = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
    results, n_docs = rr.compare_arms(ref_vecs, vecs, domains, ref_fps, fps)
    order = [r.label for r in results]
    per_doc = np.array([r.recall[1] for r in results])
    return order, per_doc, n_docs


def _paired_contrast(name_a, a, name_b, b, order_a, order_b, z_crit, delta_mins, achieved_rp):
    if order_a != order_b:
        raise SystemExit(f"{name_a} vs {name_b}: document order differs -- not a valid pair")
    d = a - b
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    signed_ratio = mean / (z_crit * se) if se > 0 else float("nan")
    verdicts = {}
    for label, delta_min in delta_mins.items():
        if abs(mean) < achieved_rp:
            verdict = "UNDERPOWERED"
        elif signed_ratio > 1 and mean >= delta_min:
            verdict = "RESOLVED-RECOVERY"
        elif signed_ratio < -1 and mean <= -delta_min:
            verdict = "RESOLVED-REGRESSION"
        else:
            verdict = "NOT RESOLVED"
        verdicts[label] = verdict
    return {
        "contrast": f"{name_a} - {name_b}",
        "n": n,
        "mean_delta_recall1": mean,
        "se_paired": se,
        "signed_ratio_z": signed_ratio,
        "achieved_rp_fw": achieved_rp,
        "verdicts_by_delta_min": verdicts,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1777-out", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--sigma-d", type=float, default=0.50,
                    help="planning sigma_d for the achieved-RP figure (T-1877 S5 default)")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    import t1777_retrieval_report as rr  # noqa: E402

    out = Path(args.t1777_out)
    manifest = out / "t1777_corpus" / "manifest.jsonl"
    labels, domains = [], {}
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                labels.append(rec["label"])
                domains[rec["label"]] = rec.get("domain", rec.get("population", "ALL"))

    ref_vecs, ref_fps = rr.load_pooled_vectors(labels, out / "t1777_full_float_bf16", "float")

    cell = Path(args.cell)
    per_doc = {}
    orders = {}
    n_docs = None
    for arm in REQUIRED_ARMS:
        order, arr, n = _load_arm_recall1(cell, arm, rr, ref_vecs, ref_fps, domains)
        per_doc[arm] = arr
        orders[arm] = order
        n_docs = n

    z_crit = float(stats.norm.ppf(1 - 0.05 / (2 * 2)))  # m=2, two-sided Bonferroni
    n = n_docs
    achieved_rp_050 = z_crit * args.sigma_d / np.sqrt(n)
    achieved_rp_045 = z_crit * 0.45 / np.sqrt(n)
    delta_mins = {"delta_min=0.06 (T-1870 S3.2 default)": 0.06,
                  "delta_min=0.10 (N=238-matched table row, T-1877 S5)": 0.10}

    print(f"z_crit (m=2, Bonferroni, alpha=0.05, two-sided): {z_crit:.6f}")
    print(f"achieved family-wise RP at N={n}, sigma_d=0.50: {achieved_rp_050:.4f}")
    print(f"achieved family-wise RP at N={n}, sigma_d=0.45: {achieved_rp_045:.4f}")

    print("\n=== per-arm recall@1 (N={}) ===".format(n))
    for arm in REQUIRED_ARMS:
        print(f"  {arm:26s} recall@1 = {per_doc[arm].mean():.6f}")

    # Reproduction gate: only04/only05 vs null, direction/magnitude vs T-1835 S1's filed
    # figures (site4 +0.2008, site5 +0.0418), computed here as isolated-vs-null deltas --
    # T-1835 S1's own framing (isolated arm vs the null element of the same run).
    repro = {}
    for name, key in (("K (site4, only04_k_proj_landing)", "only04_k_proj_landing"),
                      ("V (site5, only05_v_proj_landing)", "only05_v_proj_landing")):
        d = per_doc[key] - per_doc["null"]
        repro[key] = {"mean_delta_vs_null": float(d.mean()),
                      "se": float(d.std(ddof=1) / np.sqrt(n))}
        print(f"\nreproduction check -- {name}: delta vs null = {d.mean():+.6f} "
              f"(T-1835 S1 filed: {'+0.2008' if key=='only04_k_proj_landing' else '+0.0418'})")

    print("\n=== T-1877 S4 decision-bearing contrasts (m=2) ===")
    role_v = _paired_contrast("onlyK_vscale", per_doc["onlyK_vscale"],
                              "onlyV_ownscale (only05_v_proj_landing)",
                              per_doc["only05_v_proj_landing"],
                              orders["onlyK_vscale"], orders["only05_v_proj_landing"],
                              z_crit, delta_mins, achieved_rp_050)
    role_k = _paired_contrast("onlyK_ownscale (only04_k_proj_landing)",
                              per_doc["only04_k_proj_landing"],
                              "onlyV_kscale", per_doc["onlyV_kscale"],
                              orders["only04_k_proj_landing"], orders["onlyV_kscale"],
                              z_crit, delta_mins, achieved_rp_050)
    for label, c in (("role_effect_at_V_scale", role_v), ("role_effect_at_K_scale", role_k)):
        print(f"\n{label}: {c['contrast']}")
        print(f"  mean delta = {c['mean_delta_recall1']:+.6f}, SE = {c['se_paired']:.6f}, "
              f"signed_ratio(z) = {c['signed_ratio_z']:.4f}")
        for dl, v in c["verdicts_by_delta_min"].items():
            print(f"  {dl}: {v}")

    print("\n=== informative-only contrasts (not decision-bearing) ===")
    scale_k = _paired_contrast("onlyK_ownscale", per_doc["only04_k_proj_landing"],
                               "onlyK_vscale", per_doc["onlyK_vscale"],
                               orders["only04_k_proj_landing"], orders["onlyK_vscale"],
                               z_crit, delta_mins, achieved_rp_050)
    scale_v = _paired_contrast("onlyV_kscale", per_doc["onlyV_kscale"],
                               "onlyV_ownscale", per_doc["only05_v_proj_landing"],
                               orders["onlyV_kscale"], orders["only05_v_proj_landing"],
                               z_crit, delta_mins, achieved_rp_050)
    for label, c in (("scale_effect_at_K_role", scale_k), ("scale_effect_at_V_role", scale_v)):
        print(f"\n{label}: {c['contrast']}")
        print(f"  mean delta = {c['mean_delta_recall1']:+.6f}, SE = {c['se_paired']:.6f}, "
              f"signed_ratio(z) = {c['signed_ratio_z']:.4f}")

    if args.json_out:
        payload = {
            "n_docs": n,
            "z_crit_m2": z_crit,
            "achieved_rp_fw_sigma050": achieved_rp_050,
            "achieved_rp_fw_sigma045": achieved_rp_045,
            "per_arm_recall1_mean": {a: float(per_doc[a].mean()) for a in REQUIRED_ARMS},
            "reproduction_check": repro,
            "role_effect_at_V_scale": role_v,
            "role_effect_at_K_scale": role_k,
            "scale_effect_at_K_role": scale_k,
            "scale_effect_at_V_role": scale_v,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
