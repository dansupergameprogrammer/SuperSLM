#!/usr/bin/env python3
"""T-1755 (E6 re-measurement) -- op-level fidelity of the shipped i-exp softmax
construction (SoftmaxRowQ15 / IExpConstruct+IExpEvaluate, intmath.h, C7/C8/C9) against
an ideal (double-precision) softmax, on real self-checked score rows captured by
tools/t1755_softmax_probe.cpp from the compiled engine's own attention computation.

Dequantization: SoftmaxRowQ15's own composition comment (intmath.h) states
  s[k]  = ShiftByMax(scores)
  e[k]  = IExpEvaluate(IExpConstruct(s[k], q_ln2, q_b, q_c))     (C7/C8/C9)
  p[k]  = (e[k] << 15) / max(sum_k e[k], 1)
and IExpConstruct's own decomposition (intmath.h, verified at source):
  z = -clipped / q_ln2 ;  q_p = clipped + z*q_ln2 ;  value = ((q_p+q_b)^2+q_c) >> z
which is the standard exp(x) = 2^z * exp(residual) decomposition -- q_ln2 IS ln(2)
expressed in the row's own integer units (z counts "how many ln2's"), so
  real_x[k] = clipped_shifted_score[k] * (ln(2) / q_ln2)
converts a raw (post-max-subtraction, pre-clip) score into the real logit-difference
it represents. IExpEvaluate's own internal scale is undetermined in isolation ("out_scale
is never computed at runtime ... the nonlinear consumers are same-scale ratios and it
cancels", intmath.h) but is IDENTICAL for every element of one row (same q_ln2/q_b/q_c
triple), so it cancels exactly in softmax's own normalization -- no absolute scale is
needed to compute the row's ideal probability vector, only relative real_x values.
"""
import sys
import math

I_EXP_CLIP_N = 30

def ideal_probs(scores, q_ln2):
    mx = max(scores)
    shifted = [s - mx for s in scores]
    clip = -I_EXP_CLIP_N * q_ln2
    clipped = [max(s, clip) for s in shifted]
    real_x = [c * (math.log(2.0) / q_ln2) for c in clipped]
    exps = [math.exp(x) for x in real_x]
    total = sum(exps)
    return [e / total for e in exps]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out/t1755_softmax_rows.txt"
    with open(path) as f:
        lines = f.read().split("\n")
    idx = 0
    n_rows = int(lines[idx]); idx += 1

    per_layer = {}
    total_elems = 0
    total_diff2 = 0  # elements differing by >= 2 Q15 codes
    total_diff1 = 0  # elements differing by exactly 1 Q15 code
    max_delta_overall = 0
    all_deltas = []

    for r in range(n_rows):
        header = lines[idx].split(); idx += 1
        layer, head, width, q_ln2, q_b, q_c = (int(x) for x in header)
        scores = [int(x) for x in lines[idx].split()]; idx += 1
        probs_q15 = [int(x) for x in lines[idx].split()]; idx += 1
        assert len(scores) == width and len(probs_q15) == width

        ideal = ideal_probs(scores, q_ln2)
        ideal_q15 = [round(p * 32768.0) for p in ideal]

        row_max_delta = 0
        row_diffs = 0
        for k in range(width):
            delta = abs(probs_q15[k] - ideal_q15[k])
            all_deltas.append(delta)
            if delta > row_max_delta:
                row_max_delta = delta
            if delta >= 1:
                row_diffs += 1
            if delta == 1:
                total_diff1 += 1
            if delta >= 2:
                total_diff2 += 1
        total_elems += width
        max_delta_overall = max(max_delta_overall, row_max_delta)

        per_layer.setdefault(layer, []).append((row_max_delta, row_diffs, width))

        # Sanity: engine's own Q15 row should sum close to 32768 (a strong
        # independent check on the capture itself, not on the ideal reconstruction).
        engine_sum = sum(probs_q15)
        if r < 3:
            print(f"row {r}: layer={layer} head={head} width={width} q_ln2={q_ln2} "
                  f"engine_sum={engine_sum} (want ~32768) row_max_delta={row_max_delta} "
                  f"row_diffs={row_diffs}/{width}")

    pct_any_diff = 100.0 * sum(1 for d in all_deltas if d >= 1) / len(all_deltas)
    pct_diff_ge2 = 100.0 * total_diff2 / len(all_deltas)
    print()
    print(f"TOTAL: {n_rows} self-checked rows, {total_elems} Q15 elements")
    print(f"max delta (any element, any row): {max_delta_overall}")
    print(f"elements differing by >=1 code: {pct_any_diff:.4f}%")
    print(f"elements differing by >=2 codes: {pct_diff_ge2:.4f}%")
    print()
    print("per-checkpoint-layer max delta:")
    for layer in sorted(per_layer):
        rows = per_layer[layer]
        lmax = max(r[0] for r in rows)
        ldiffs = sum(r[1] for r in rows)
        lwidth_total = sum(r[2] for r in rows)
        print(f"  layer={layer}: max_delta={lmax} diffs={ldiffs}/{lwidth_total} "
              f"({100.0*ldiffs/lwidth_total:.4f}%)")

if __name__ == "__main__":
    main()
