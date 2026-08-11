#!/usr/bin/env python3
"""T-1893 DEBUNK PROBE (disposable). Decompose T-1891 leg 1's relative-L2 drift.

The packet's one DETECTABLE figure is `||pooled_int8 - pooled_float||_2 /
||pooled_float||_2`, computed on UNNORMALIZED pooled vectors, while the recall@1
figure it sits beside is computed from cosine similarity, which discards vector
magnitude entirely (t1777_retrieval_report.cosine_sim_matrix normalizes).

For rho = ||c||/||f|| and theta = angle(c, f):

    (||c - f|| / ||f||)^2 = (rho - cos t)^2 + sin^2 t

so relative L2 splits exactly into a RADIAL term (|rho - cos t|, a pure
magnitude-calibration error that cosine ranking cannot see) and an ANGULAR term
(sin t, the minimum relative error achievable over ALL rescalings of c, and the
only part any cosine-based retrieval can respond to).

If the fused arm's drift win lives in the radial term, "closer to the float
reference" is a claim about vector length, not about representation direction.
Every figure below is paired per document and carries its own 95% paired-t
half-width, the same construction t1891_paired_recall_delta.py uses.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1777_retrieval_report as trr  # noqa: E402

BASE = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out\t1891_capture")
MANIFEST = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out\t1777_corpus\manifest.jsonl")


def halfwidth(diffs, z=1.96):
    n = len(diffs)
    return z * float(np.std(diffs, ddof=1)) / math.sqrt(n)


def paired(name, a, b, labels):
    """a = old arm values, b = fused arm values; delta reported as old - fused
    (positive = fused is smaller/closer, matching the packet's own sign)."""
    d = [x - y for x, y in zip(a, b)]
    delta = float(np.mean(d))
    hw = halfwidth(d)
    verdict = "DETECTABLE" if abs(delta) > hw else "NOT DISTINGUISHABLE FROM ZERO"
    print(f"{name:<34} old={np.mean(a):.6f}  fused={np.mean(b):.6f}  "
          f"delta(old-fused)={delta:+.6f}  rp={hw:.6f}  {verdict}")
    return {"mean_old": float(np.mean(a)), "mean_fused": float(np.mean(b)),
            "paired_delta_old_minus_fused": delta, "resolving_power": hw,
            "verdict": verdict,
            "n_docs_fused_better": int(sum(1 for x in d if x > 0)),
            "n_docs_old_better": int(sum(1 for x in d if x < 0))}


def main() -> int:
    labels = [json.loads(l)["label"] for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    fl, _ = trr.load_pooled_vectors(labels, BASE / "float_bf16", "float")
    ol, _ = trr.load_pooled_vectors(labels, BASE / "int8_old", "int8")
    fu, _ = trr.load_pooled_vectors(labels, BASE / "int8_fused", "int8")
    common = sorted(set(fl) & set(ol) & set(fu))
    print(f"N={len(common)} documents present in all three arms\n")

    def comps(c, f):
        nf = float(np.linalg.norm(f))
        nc = float(np.linalg.norm(c))
        cos = float(np.dot(c, f) / (nc * nf))
        rho = nc / nf
        rel = float(np.linalg.norm(c - f)) / nf
        sin_t = math.sqrt(max(0.0, 1.0 - cos * cos))
        radial = abs(rho - cos)
        return rel, sin_t, radial, rho, cos

    O = [comps(ol[l], fl[l]) for l in common]
    F = [comps(fu[l], fl[l]) for l in common]

    out = {}
    print("--- the packet's headline figure, reproduced ---")
    out["relative_l2"] = paired("relative L2 (packet's metric)", [x[0] for x in O], [x[0] for x in F], common)
    print("\n--- its exact decomposition ---")
    out["angular_sin_theta"] = paired("ANGULAR sin(theta)  [scale-free]", [x[1] for x in O], [x[1] for x in F], common)
    out["radial_term"] = paired("RADIAL |rho - cos|  [magnitude]", [x[2] for x in O], [x[2] for x in F], common)
    print("\n--- the underlying quantities ---")
    out["cosine_to_float"] = paired("cosine(candidate, float)", [x[4] for x in O], [x[4] for x in F], common)
    out["norm_ratio_rho"] = paired("norm ratio ||c||/||f||", [x[3] for x in O], [x[3] for x in F], common)

    # How much of the drift delta is explained by the radial term alone?
    rel_d = out["relative_l2"]["paired_delta_old_minus_fused"]
    # decompose the mean-squared way too, to avoid a sqrt artefact
    rad_sq_o = float(np.mean([x[2] ** 2 for x in O])); rad_sq_f = float(np.mean([x[2] ** 2 for x in F]))
    ang_sq_o = float(np.mean([x[1] ** 2 for x in O])); ang_sq_f = float(np.mean([x[1] ** 2 for x in F]))
    print(f"\nmean-square split: radial^2 old={rad_sq_o:.6f} fused={rad_sq_f:.6f} (delta {rad_sq_o-rad_sq_f:+.6f})")
    print(f"                   angular^2 old={ang_sq_o:.6f} fused={ang_sq_f:.6f} (delta {ang_sq_o-ang_sq_f:+.6f})")
    share = (rad_sq_o - rad_sq_f) / ((rad_sq_o - rad_sq_f) + (ang_sq_o - ang_sq_f)) if ((rad_sq_o-rad_sq_f)+(ang_sq_o-ang_sq_f)) != 0 else float('nan')
    print(f"radial share of the mean-square drift improvement: {share*100:.1f}%")
    out["mean_square_split"] = {"radial_sq_old": rad_sq_o, "radial_sq_fused": rad_sq_f,
                                "angular_sq_old": ang_sq_o, "angular_sq_fused": ang_sq_f,
                                "radial_share_of_improvement": share}
    print(f"\nrelative-L2 delta reproduced from raw dumps: {rel_d:+.6f} "
          f"(packet: +0.030329)")

    Path(BASE / "t1893_drift_decomp.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {BASE / 't1893_drift_decomp.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
