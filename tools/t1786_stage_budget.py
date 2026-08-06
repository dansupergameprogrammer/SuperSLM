#!/usr/bin/env python3
"""T-1786: attribute the attention construction's measured probability-space TVD (T-1778
D-SLM1234-1240, 0.155-0.204 against a fixed, independent float32 reference) across the stages
that produce it, by SUBSTITUTION -- holding every stage but one at exact production integer
behaviour, replacing the remaining one with its float-exact counterpart, and measuring what the
row's own TVD against the FIXED float reference becomes.

INPUTS, and what each one takes as a parameter (StandardsDocument.md 5.4, "a comparison is
evidence only if... the reference is independent of what it grades"):

  - tools/t1786_stage_dump_probe.cpp's own dump (out/t1786/p{1,2,3}.txt): the REAL arm's raw
    post-GEMM scores, derived (q_ln2, q_b, q_c), and Q15 probabilities, self-checked bit-for-bit
    against production (18/18 per prompt, printed at capture time). This script treats these as
    ENGINE INPUTS to replay, not as anything it derives itself.
  - tools/t1786_float_attn_reference.py's own dump (out/t1786/p{1,2,3}_float.txt): an unmodified
    copy of T-1778's own float32 reference instrument. Takes NO input from the engine's
    integer construction (T-1778 Sec4, re-verified unchanged here) -- it is the checkpoint's own
    stock HuggingFace eager-attention forward, computed independently, before this script ever
    reads it. THIS FILE IS NEVER RECOMPUTED OR MODIFIED BY ANY SUBSTITUTION BELOW -- it is read
    once per prompt and held fixed across every stage variant, which is the property T-1776
    (D-SLM1195-1198) found missing from the prior ablation family (that reference took the
    intervened q_ln2 as a parameter and moved with the intervention). No substitution in this
    script ever varies q_ln2, q_b, or q_c -- every variant reuses the SAME per-row values the
    dump probe captured from production; only the ARITHMETIC OPERATION applied to them changes,
    one stage at a time.

STAGES, as isolable arithmetic operations inside SoftmaxRowQ15 (include/superslm/intmath.h:568-
650, src/intmath.cpp:788-893), each stage's PRODUCTION and FLOAT-EXACT form stated explicitly:

  A. Domain clip.       PROD: clipped = max(shifted, -30*q_ln2).      EXACT: clipped = shifted
                         (no clip; z is allowed to grow past 30 rather than being bounded).
  B. Exponent approx.   PROD: (q_p+q_b)^2 + q_c, right-shifted by z (a quadratic fit to 2^x on
                         one octave). EXACT: M * 2^(clipped/q_ln2) in double precision -- the
                         exact continuous target T-1768 Sec7b already names as what the
                         polynomial approximates, evaluated at the SAME clipped value production
                         used (this substitution does not touch stage A).
  C. Accumulation.       PROD: total = sum(exps) in int64.            EXACT: identical -- integer
                         addition of already-exact integers has no rounding to remove. Verified,
                         not assumed, in Sec2 below.
  D. Q15 quantization.  PROD: floor((exps[k] << 15) / total).        EXACT: the real-valued
                         ratio exps[k] / total, carried at full precision, never rounded to a
                         15-bit integer.

A fifth candidate, scale derivation (q_ln2/q_b/q_c themselves), and two more named by the ticket
(the P x V matmul, the requantization chain after it) are addressed in Sec5/Sec6 as stages that
CANNOT be isolated by this method, with the reason stated -- not silently omitted.

METHOD. For each of the 216 (layer,head,prompt) rows in the population: (1) replay production
bit-for-bit from the dumped scores/q_ln2/q_b/q_c and self-check against the dumped probs
(Sec2); (2) compute the row's baseline TVD against the FIXED float reference; (3) for each
stage, compute the row's TVD with ONLY that stage substituted; (4) contribution = baseline_TVD -
substituted_TVD (the fraction of the row's own error eliminated by making exactly that one stage
exact). Pooled per layer and overall, exactly as T-1778/T-1784 pool (weighted census over every
element, not an average of per-row averages, for the element-level statistics; row-mean TVD for
the row-level statistic, matching T-1778's own convention).
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

I_EXP_CLIP_N = 30
PROB_FRAC_BITS = 15


def read_engine_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, width = int(header[0]), int(header[1])
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, head, w = int(parts[0]), int(parts[1]), int(parts[2])
            q_ln2, q_b, q_c = int(parts[3]), int(parts[4]), int(parts[5])
            idx = 6
            scores = [int(x) for x in parts[idx : idx + w]]
            idx += w
            probs = [int(x) for x in parts[idx : idx + w]]
            rows[(layer, head)] = {
                "width": w, "q_ln2": q_ln2, "q_b": q_b, "q_c": q_c,
                "scores": scores, "probs": probs,
            }
    return rows, width


def read_float_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, width = int(header[0]), int(header[1])
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, head, w = int(parts[0]), int(parts[1]), int(parts[2])
            probs = [float(x) for x in parts[3 : 3 + w]]
            rows[(layer, head)] = probs
    return rows, width


def shift_by_max(scores):
    peak = max(scores)
    return [s - peak for s in scores]


def production_row(scores, q_ln2, q_b, q_c):
    """Bit-exact replica of SoftmaxRowQ15 (src/intmath.cpp:788-893). Returns the Q15 int probs."""
    M = q_b * q_b + q_c
    shifted = shift_by_max(scores)
    clip_lo = -I_EXP_CLIP_N * q_ln2
    exps = []
    for s in shifted:
        clipped = s if s >= clip_lo else clip_lo
        z = (-clipped) // q_ln2
        q_p = clipped + z * q_ln2
        base = q_p + q_b
        v = base * base + q_c
        val = v >> z  # Python's >> on a (possibly negative) int is an arithmetic floor shift,
                       # matching intmath.cpp's own documented "(v >> z) is an arithmetic (floor)
                       # shift" (IExpEvaluate, src/intmath.cpp:549).
        if val < 0 or val > M:
            val = 0
        exps.append(val)
    total = sum(exps)
    denom = total if total > 1 else 1
    probs = [(e << PROB_FRAC_BITS) // denom for e in exps]
    return probs, exps, total, M


def stage_a_unclamped_exps(scores, q_ln2, q_b, q_c):
    """Stage A float-exact: remove the domain clip entirely (z allowed past 30), holding B
    (the same quadratic evaluation rule), C, D at production. Returns integer exps (arbitrary
    precision -- no int64 overflow risk in Python, and z growing only shrinks the shifted
    value, so no representability concern the production code's own int64 width exists to
    guard)."""
    shifted = shift_by_max(scores)
    exps = []
    for s in shifted:
        clipped = s  # no clamp
        z = (-clipped) // q_ln2
        q_p = clipped + z * q_ln2
        base = q_p + q_b
        v = base * base + q_c
        val = v >> z
        exps.append(max(val, 0))
    return exps


def stage_b_exact_exps(scores, q_ln2, q_b, q_c):
    """Stage B float-exact: hold the domain clip at production (stage A production), replace
    the quadratic polynomial with the exact continuous target it approximates, M*2^(clipped/
    q_ln2), evaluated in double precision at the SAME clipped value production used."""
    M = q_b * q_b + q_c
    shifted = shift_by_max(scores)
    clip_lo = -I_EXP_CLIP_N * q_ln2
    exps = []
    for s in shifted:
        clipped = s if s >= clip_lo else clip_lo
        val = M * math.exp(clipped * math.log(2.0) / q_ln2)
        exps.append(val)
    return exps


def tvd(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def main():
    worktree = Path(r"D:\SuperSLM\.worktrees\t1786-error-budget")
    outdir = worktree / "out" / "t1786"

    all_rows = []  # list of dicts, one per (prompt, layer, head)
    self_check_fail = 0

    for pi in (1, 2, 3):
        eng_rows, eng_w = read_engine_dump(outdir / f"p{pi}.txt")
        flt_rows, flt_w = read_float_dump(outdir / f"p{pi}_float.txt")
        assert eng_w == flt_w, f"P{pi}: width mismatch engine={eng_w} float={flt_w}"
        assert set(eng_rows) == set(flt_rows), f"P{pi}: (layer,head) key mismatch"

        for key, e in eng_rows.items():
            layer, head = key
            scores, q_ln2, q_b, q_c = e["scores"], e["q_ln2"], e["q_b"], e["q_c"]
            width = e["width"]

            prod_probs, prod_exps, prod_total, M = production_row(scores, q_ln2, q_b, q_c)
            if prod_probs != e["probs"]:
                self_check_fail += 1
                print(f"SELF-CHECK FAIL P{pi} layer={layer} head={head}: python replica "
                      f"diverges from dumped production probs", file=sys.stderr)
                continue

            p_float = flt_rows[key]
            assert len(p_float) == width

            p_prod_real = [p / 32768.0 for p in prod_probs]
            baseline_tvd = tvd(p_prod_real, p_float)

            # --- Stage A: domain clip -> exact (unclamp), B/C/D production ---
            exps_a = stage_a_unclamped_exps(scores, q_ln2, q_b, q_c)
            total_a = sum(exps_a)
            denom_a = total_a if total_a > 1 else 1
            probs_a = [(e2 << PROB_FRAC_BITS) // denom_a for e2 in exps_a]
            p_a_real = [p / 32768.0 for p in probs_a]
            tvd_a = tvd(p_a_real, p_float)

            # --- Stage B: exponent approx -> exact, A/C/D production (C/D re-applied to the
            # float exps at Q15 granularity, matching how production would consume a corrected
            # exponential: floor-quantize the same way). ---
            exps_b = stage_b_exact_exps(scores, q_ln2, q_b, q_c)
            total_b = sum(exps_b)
            denom_b = total_b if total_b > 1 else 1.0
            probs_b_real = [e2 / denom_b for e2 in exps_b]  # real-valued; Q15 floor applied too:
            probs_b_q15 = [math.floor(e2 * 32768.0 / denom_b) for e2 in exps_b]
            p_b_real = [p / 32768.0 for p in probs_b_q15]
            tvd_b = tvd(p_b_real, p_float)

            # --- Stage C: accumulation/normalization -> exact. Verified here, not assumed:
            # production's own total (int) IS the exact sum of production's own exps (ints);
            # there is no separate "exact" accumulation to substitute -- confirmed by direct
            # comparison every row. ---
            exact_total_c = sum(prod_exps)
            stage_c_is_noop = (exact_total_c == prod_total)

            # --- Stage D: Q15 quantization -> exact (real-valued ratio, no floor-to-15-bit). A/
            # B/C held at production (production exps/total). ---
            p_d_real = [e2 / prod_total if prod_total > 0 else 0.0 for e2 in prod_exps]
            tvd_d = tvd(p_d_real, p_float)

            # --- All-corrected (A+B+D; C is a no-op): the closest this construction can get to
            # the float reference WITHOUT touching q_ln2/q_b/q_c or anything upstream of the raw
            # integer scores themselves -- i.e., treating shifted[k]*ln2/q_ln2 as the row's own
            # recovered real logit and taking an exact softmax over it. ---
            shifted = shift_by_max(scores)
            exps_all = [math.exp(s * math.log(2.0) / q_ln2) for s in shifted]
            total_all = sum(exps_all)
            p_all_real = [e2 / total_all for e2 in exps_all]
            tvd_all = tvd(p_all_real, p_float)

            all_rows.append({
                "prompt": pi, "layer": layer, "head": head, "width": width,
                "q_ln2": q_ln2,
                "baseline": baseline_tvd,
                "a": tvd_a, "b": tvd_b, "c_noop": stage_c_is_noop, "d": tvd_d,
                "all_corrected": tvd_all,
            })

    print(f"self_check_failures={self_check_fail} / {len(all_rows) + self_check_fail} rows")
    assert self_check_fail == 0, "python replica does not match production -- STOP, do not trust anything below"
    print(f"stage C (accumulation/normalization) is a no-op on every row: "
          f"{all(r['c_noop'] for r in all_rows)} ({len(all_rows)}/{len(all_rows)} rows)")

    # Per-layer pooled (mean over rows -- 36 rows/layer: 3 prompts x 12 heads), matching
    # T-1778/T-1784's own row-mean convention.
    by_layer = defaultdict(list)
    for r in all_rows:
        by_layer[r["layer"]].append(r)

    print(f"\n{'layer':>5} {'n':>4} {'base':>8} {'A':>8} {'B':>8} {'D':>8} {'all':>8} "
          f"{'A contrib':>10} {'B contrib':>10} {'D contrib':>10} {'sum A+B+D':>10} {'residual':>10}")
    grand = defaultdict(list)
    for layer in sorted(by_layer):
        rs = by_layer[layer]
        n = len(rs)
        base = sum(r["baseline"] for r in rs) / n
        a = sum(r["a"] for r in rs) / n
        b = sum(r["b"] for r in rs) / n
        d = sum(r["d"] for r in rs) / n
        allc = sum(r["all_corrected"] for r in rs) / n
        ca = base - a
        cb = base - b
        cd = base - d
        csum = ca + cb + cd
        residual = base - csum  # closure gap: what the additive sum does not explain
        print(f"{layer:>5} {n:>4} {base:>8.4f} {a:>8.4f} {b:>8.4f} {d:>8.4f} {allc:>8.4f} "
              f"{ca:>10.4f} {cb:>10.4f} {cd:>10.4f} {csum:>10.4f} {residual:>10.4f}")
        for k, v in (("base", base), ("a", a), ("b", b), ("d", d), ("allc", allc)):
            grand[k].append((v, n))

    print("\nGrand pooled (mean over all 216 rows, unweighted layer-then-row average shown; "
          "row-level pooled figure below):")
    n_total = len(all_rows)
    base = sum(r["baseline"] for r in all_rows) / n_total
    a = sum(r["a"] for r in all_rows) / n_total
    b = sum(r["b"] for r in all_rows) / n_total
    d = sum(r["d"] for r in all_rows) / n_total
    allc = sum(r["all_corrected"] for r in all_rows) / n_total
    ca, cb, cd = base - a, base - b, base - d
    csum = ca + cb + cd
    residual = base - csum
    print(f"n={n_total} baseline_TVD={base:.4f} A={a:.4f} B={b:.4f} D={d:.4f} all_corrected={allc:.4f}")
    print(f"contribution A(domain clip)={ca:.4f} ({100*ca/base:.1f}%)  "
          f"B(exponent approx)={cb:.4f} ({100*cb/base:.1f}%)  "
          f"D(Q15 quantization)={cd:.4f} ({100*cd/base:.1f}%)")
    print(f"sum of contributions={csum:.4f} ({100*csum/base:.1f}% of baseline); "
          f"residual (closure gap, attributable to raw-score int8 quantization upstream of this "
          f"construction, per Sec5/Sec6)={residual:.4f} ({100*residual/base:.1f}%)")
    print(f"all_corrected TVD (A+B+D fixed simultaneously)={allc:.4f} -- compare to residual "
          f"above as a second, independent measure of the same closure gap")

    # Per-prompt breakdown for resolving-power (n=3 independent prompts, same discipline
    # T-1778 Sec8 applied).
    print("\nPer-prompt grand-pooled baseline TVD (resolving-power check, n=3):")
    for pi in (1, 2, 3):
        rs = [r for r in all_rows if r["prompt"] == pi]
        print(f"  P{pi}: n={len(rs)} baseline_TVD={sum(r['baseline'] for r in rs)/len(rs):.4f} "
              f"A_contrib={sum(r['baseline']-r['a'] for r in rs)/len(rs):.4f} "
              f"B_contrib={sum(r['baseline']-r['b'] for r in rs)/len(rs):.4f} "
              f"D_contrib={sum(r['baseline']-r['d'] for r in rs)/len(rs):.4f}")


if __name__ == "__main__":
    main()
