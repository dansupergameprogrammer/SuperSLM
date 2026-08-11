#!/usr/bin/env python3
"""T-1891 capture leg 1 -- the paired old-vs-fused recall@1 delta, with its resolving
power (StandardsDocument.md 5.4), AND the mean relative L2 drift of the pooled final
state against the same-run bf16 reference.

DISPOSABLE. Not part of T-1777's own tooling; imports its functions rather than
modifying it (t1777_retrieval_report.py stays exactly what T-1777 built and T-1892
re-verified).

`t1777_retrieval_report.py` reports recall@1 for ONE candidate arm against the float
reference, with an INDEPENDENT Wilson-style resolving power. It does not compare two
candidate arms (old vs fused) against each other. This script does: for each of the
239 documents, whether int8-old's own top-1 nearest neighbour matches the float
reference's top-1 neighbour is a binary outcome (recall@1 at k=1 is exactly 0.0 or
1.0 per query); so is int8-fused's. The two outcomes are PAIRED (same query, same
reference ranking), so the delta's resolving power is computed from the McNemar-style
paired variance, not from two independent binomial CIs -- the correct construction
for "did old and fused answer the SAME retrieval question differently", not "are two
unrelated proportions different".

DRIFT (per coordinator scope ruling, 2026-08-10): this is NOT T-1820's mechanism
decomposition (kernel-ridge phases). It is the plain, continuous metric the campaign
has always called "final-state drift" (T-1835's own 0.6413->0.4948 for off04): the
mean relative L2 of the pooled final state against the same-run bf16 reference, per
document -- `||pooled_int8 - pooled_float||_2 / ||pooled_float||_2`. The pooled
vectors are the SAME ones `t1777_retrieval_report.py`'s own `pooled_final_layer_int8`/
`pooled_final_layer_float` produce for recall@1 (D-SLM447's ratified pooling scheme:
mean over positions 1..n, excluding position 0) -- one computation, reused for both
metrics, not two independent derivations of "the final state".

Usage:
    python tools/t1891_paired_recall_delta.py --int8-old-dir <dir> --int8-fused-dir <dir> \\
        --float-bf16-dir <dir> --manifest out/t1777_corpus/manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1777_retrieval_report as trr  # noqa: E402


def load_manifest(path: Path):
    labels, domains = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            labels.append(rec["label"])
            domains[rec["label"]] = rec.get("domain", "unknown")
    return labels, domains


def relative_l2(candidate: "np.ndarray", reference: "np.ndarray") -> float:
    """||candidate - reference||_2 / ||reference||_2 -- the plain relative-L2 drift of
    one pooled vector against its same-document bf16 reference. `reference`'s norm is
    the denominator (never `candidate`'s), matching every other "relative to the
    float reference" convention this codebase uses (StandardsDocument 5.4: the
    reference is independent of what it grades -- here the reference vector is float
    bf16, produced without reading either int8 arm)."""
    import numpy as np
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(candidate - reference)) / denom


def paired_mean_halfwidth(diffs: list, z: float = 1.96) -> float:
    """95% half-width on a paired CONTINUOUS difference (drift_old - drift_fused per
    document): the standard paired-t construction, SD(diffs)/sqrt(N), z (not t, N=239
    is large enough the two are within rounding) -- unlike recall@1's binary paired
    variance, this metric has no natural discrete-step floor (no report combines a
    fixed number of graded positions per document into one quantized value), so the
    resolving power here is exactly this CI half-width, stated as such rather than
    padded with a step term that would not mean anything for a continuous metric."""
    import numpy as np
    n = len(diffs)
    if n < 2:
        return float("nan")
    sd = float(np.std(diffs, ddof=1))
    return z * sd / math.sqrt(n)


def paired_mcnemar_halfwidth(p01: float, p10: float, n: int, z: float = 1.96) -> float:
    """95% half-width on a paired proportion difference (delta = p_fused_correct -
    p_old_correct = p01 - p10, where p01 = P(old wrong, fused right), p10 = P(old
    right, fused wrong)): Var(delta) = (p01 + p10 - (p01-p10)^2) / n, the standard
    paired-binary variance (this IS McNemar's test's own variance term, used here as
    a CI rather than a p-value, matching this codebase's own normal_ci_halfwidth
    convention elsewhere in this tool family)."""
    if n <= 0:
        return float("nan")
    delta = p01 - p10
    var = max(p01 + p10 - delta * delta, 0.0) / n
    return z * math.sqrt(var)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--int8-old-dir", required=True)
    ap.add_argument("--int8-fused-dir", required=True)
    ap.add_argument("--float-bf16-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    labels, domains = load_manifest(Path(args.manifest))
    float_vecs, float_fps = trr.load_pooled_vectors(labels, Path(args.float_bf16_dir), "float")
    old_vecs, old_fps = trr.load_pooled_vectors(labels, Path(args.int8_old_dir), "int8")
    fused_vecs, fused_fps = trr.load_pooled_vectors(labels, Path(args.int8_fused_dir), "int8")

    print(f"loaded: float={len(float_vecs)} old={len(old_vecs)} fused={len(fused_vecs)} "
          f"(of {len(labels)} manifest labels)")

    results_old, n_old = trr.compare_arms(float_vecs, old_vecs, domains, float_fps, old_fps, ks=(1,))
    results_fused, n_fused = trr.compare_arms(float_vecs, fused_vecs, domains, float_fps, fused_fps, ks=(1,))

    by_label_old = {r.label: r.recall[1] for r in results_old}
    by_label_fused = {r.label: r.recall[1] for r in results_fused}
    common = sorted(set(by_label_old) & set(by_label_fused))
    n = len(common)
    if n == 0:
        print("FAIL: no common labels between old and fused arms", file=sys.stderr)
        return 1

    old_correct = [by_label_old[l] >= 0.999 for l in common]
    fused_correct = [by_label_fused[l] >= 0.999 for l in common]

    p01 = sum(1 for o, f in zip(old_correct, fused_correct) if (not o) and f) / n   # old wrong, fused right
    p10 = sum(1 for o, f in zip(old_correct, fused_correct) if o and (not f)) / n   # old right, fused wrong
    p00 = sum(1 for o, f in zip(old_correct, fused_correct) if (not o) and (not f)) / n
    p11 = sum(1 for o, f in zip(old_correct, fused_correct) if o and f) / n

    mean_old = sum(old_correct) / n
    mean_fused = sum(fused_correct) / n
    delta = mean_fused - mean_old
    halfwidth = paired_mcnemar_halfwidth(p01, p10, n)
    step = 1.0 / n  # discrete step: one document flipping changes the paired count by 1/n
    resolving_power = max(step, halfwidth)
    verdict = "DETECTABLE" if abs(delta) > resolving_power else "NOT DISTINGUISHABLE FROM ZERO at this N"

    print(f"\n=== T-1891 leg 1: paired old-vs-fused recall@1 delta, N={n} (tuned-on corpus) ===")
    print(f"recall@1 old:   {mean_old:.4f}")
    print(f"recall@1 fused: {mean_fused:.4f}")
    print(f"paired delta (fused - old): {delta:+.4f}")
    print(f"discordant pairs: old-wrong/fused-right={round(p01*n)} ({p01:.4f})  "
          f"old-right/fused-wrong={round(p10*n)} ({p10:.4f})")
    print(f"concordant: both-right={round(p11*n)} ({p11:.4f})  both-wrong={round(p00*n)} ({p00:.4f})")
    print(f"discrete_step(1/N)={step:.5f}  95%_paired_CI_halfwidth={halfwidth:.4f}  "
          f"resolving_power={resolving_power:.4f}  {verdict}")

    # --- Drift: mean relative L2 of the pooled final state vs the bf16 reference ---
    # (coordinator scope ruling, 2026-08-10 -- the simple continuous metric, NOT
    # T-1820's mechanism decomposition). Same pooled vectors as the recall@1 arms
    # above (float_vecs/old_vecs/fused_vecs), so this is not a second capture.
    common_drift = sorted(set(float_vecs) & set(old_vecs) & set(fused_vecs))
    n_drift = len(common_drift)
    drift_old = [relative_l2(old_vecs[l], float_vecs[l]) for l in common_drift]
    drift_fused = [relative_l2(fused_vecs[l], float_vecs[l]) for l in common_drift]
    mean_drift_old = sum(drift_old) / n_drift
    mean_drift_fused = sum(drift_fused) / n_drift
    drift_diffs = [o - f for o, f in zip(drift_old, drift_fused)]  # old - fused (positive = fused improves)
    drift_delta = mean_drift_old - mean_drift_fused
    drift_halfwidth = paired_mean_halfwidth(drift_diffs)
    drift_verdict = ("DETECTABLE" if abs(drift_delta) > drift_halfwidth
                      else "NOT DISTINGUISHABLE FROM ZERO at this N")

    print(f"\n=== T-1891 leg 1: mean relative L2 drift of the pooled final state vs the "
          f"bf16 reference, N={n_drift} ===")
    print("(the simple continuous metric -- ||pooled_int8 - pooled_float||_2 / "
          "||pooled_float||_2, mean over documents -- NOT T-1820's mechanism decomposition)")
    print(f"mean relative L2, old:   {mean_drift_old:.4f}")
    print(f"mean relative L2, fused: {mean_drift_fused:.4f}")
    print(f"paired delta (old - fused): {drift_delta:+.4f}  "
          f"(positive = fused is CLOSER to the float reference)")
    print(f"95%_paired_CI_halfwidth={drift_halfwidth:.4f}  resolving_power={drift_halfwidth:.4f}  "
          f"{drift_verdict}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "recall_at_1": {
                "n": n, "recall_at_1_old": mean_old, "recall_at_1_fused": mean_fused,
                "paired_delta": delta, "p01_old_wrong_fused_right": p01,
                "p10_old_right_fused_wrong": p10, "p00_both_wrong": p00, "p11_both_right": p11,
                "discrete_step": step, "ci_95_halfwidth": halfwidth,
                "resolving_power": resolving_power, "verdict": verdict,
            },
            "drift_relative_l2": {
                "metric": "mean relative L2 of the pooled final state vs the bf16 reference "
                          "(the simple continuous metric, not T-1820's mechanism decomposition)",
                "n": n_drift, "mean_old": mean_drift_old, "mean_fused": mean_drift_fused,
                "paired_delta_old_minus_fused": drift_delta,
                "ci_95_halfwidth": drift_halfwidth, "resolving_power": drift_halfwidth,
                "verdict": drift_verdict,
            },
        }, indent=2))
        print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
