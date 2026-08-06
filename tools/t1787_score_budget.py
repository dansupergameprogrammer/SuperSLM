#!/usr/bin/env python3
"""T-1787: attribute the RAW INTEGER attention score divergence (T-1786 D-SLM1286: 99.6% of the
measured 0.155-0.204 probability-space TVD survives an exact softmax over production's own raw
scores -- i.e. lives upstream of the softmax construction entirely) across the stages that
PRODUCE those raw scores, by the SAME substitution discipline T-1786 used one level downstream.

INPUTS (StandardsDocument.md 5.4, "a comparison is evidence only if..."):

  - tools/t1787_score_stage_probe.cpp's own dump (out/t1787/p{1,2,3}_{scores,qwide,kacc,
    rope}.txt): production's raw integer scores/q_ln2/q_b/q_c (identical format to T-1786's own
    dump), Query's pre-round projection accumulator (self-checked bit-for-bit against production
    q_codes at capture time, 1536/1536 elements per checkpoint layer), Key's pre-landing-rescale
    accumulator at EVERY prefill position (self-checked against production's own KV-cache write
    at capture time), and the RoPE cos/sin (Q30) tables. This script treats these as ENGINE
    INPUTS to replay, never as anything it derives itself.
  - tools/t1787_float_prescore_reference.py's own dump (out/t1787/p{1,2,3}_float.txt): the
    checkpoint's own float32 eager-attention forward's PRE-softmax scores and POST-softmax
    probabilities, independent of the engine's integer construction (verified at capture time: no
    import of superslm/intmath/dynamic_engine; two-path oracle cross-check; softmax(scores)==
    hook-captured weights within 1e-6). THIS FILE IS NEVER RECOMPUTED OR MODIFIED BY ANY
    SUBSTITUTION BELOW -- read once per prompt, held fixed across every variant.

STAGES, as isolable arithmetic operations between the RmsNorm output and the raw score GEMM.
Each PRODUCTION stage is replayed using the EXACT closed-form the compiled function computes (the
file:line each is copied from is named); each EXACT variant removes ONLY that stage's own
round/clamp, carrying a continuous (double) value through -- every OTHER stage's production
rounding still applies to whatever it is handed, matching T-1786's own stage-B convention ("re-
applied to the float exps at Q15 granularity, matching how production would consume a corrected
exponential"):

  Q. Query quantization.  PROD: RequantTokenCodeWide (src/intmath.cpp:468-482) --
     q = clamp(round_half_away((x_i*127*r)/2^(62-s)), -127, 127). EXACT: (x_i*127*r)/2^(62-s),
     no round, no clamp. r/s are read from the dump, RECOMPUTED FOR REAL by the C++ probe via
     MaxAbsReduceWide/NormalizeScale/CarriedScaleReciprocal and self-checked there against the
     real q_codes -- this script does not re-derive them.
  K. Key quantization (the landing-rescale requantization chain, mechanically distinct from
     Query's dynamic funnel). PROD: LandingRescale (src/forward/forward_sites.cpp:301-441) +
     ClampRopeCode -- k = clamp(round_half_away((kacc*m_a*r_t)/2^(62-(e_a-e_t))), -127, 127).
     EXACT: (kacc*m_a*r_t)/2^(62-(e_a-e_t)), no round, no clamp.
  R. RoPE. PROD: RopeApplyPair (src/intmath.cpp:774-784) + ClampRopeCode -- xr=x*cos-y*sin,
     yr=x*sin+y*cos, rotated=clamp(round_half_away(xr/2^30), round_half_away(yr/2^30), -127,127).
     EXACT: xr/2^30, yr/2^30, no round, no clamp. Applied identically to whichever pre-rotation
     value (production int8 or an EXACT quantization variant) the Q/K stage above produced.
  M. Score matmul accumulation. GemmInt8AccumulateRow -- int64 sum of int8xint8 products. A
     PREDICTED no-op (finite-domain integer accumulation, no rounding to remove), CHECKED directly
     below (Sec"self-check"), not assumed -- the same argument T-1786 Sec3 used for the softmax's
     own accumulation stage, now applied one level upstream.

METHOD, matching T-1786's own shape: for each row (layer, head, last-token query vs every cached
key position k), compute the row's baseline TVD (production probs, recovered to real logits via
q_ln2 -- ALREADY PROVEN EXACT, T-1768/D-SLM1164-1167, unaffected by any substitution below since
none of them touch q_ln2/q_b/q_c -- against the FIXED float reference); compute the TVD with
EXACTLY one stage substituted; contribution = baseline - substituted.

SELF-CHECK, before anything is trusted: this script's own Python replica of Q-quant+K-quant+RoPE+
matmul, run entirely in PRODUCTION mode (no substitution), must reproduce the DUMPED PRODUCTION
scores[k] EXACTLY (arbitrary-precision integer equality -- Python ints, no floating-point
involved on this path) on every (layer, head, k) row. A single mismatch anywhere hard-stops this
script before any budget number is computed or printed.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

HIDDEN_SIZE = 1536
HEAD_DIM = 128
NUM_HEADS = 12
NUM_KV_HEADS = 2
GROUP = NUM_HEADS // NUM_KV_HEADS
PAIRS = HEAD_DIM // 2
ROPE_FRAC_BITS = 30
PROB_FRAC_BITS = 15
CHECKPOINT_LAYERS = (0, 1, 2, 9, 18, 27)


def round_half_away_int(numerator: int, exponent: int) -> int:
    """round_half_away_from_zero(numerator / 2^exponent), exact arbitrary-precision integer
    arithmetic -- the same magnitude+sign construction RequantTokenCodeWide/LandingRescale use
    (src/intmath.cpp:468-482, src/forward/forward_sites.cpp:362-388: "round_half_away_from_zero(
    magnitude/2^k) == floor((2*magnitude+2^k)/2^(k+1))")."""
    if numerator == 0:
        return 0
    neg = numerator < 0
    mag = -numerator if neg else numerator
    q = (mag * 2 + (1 << exponent)) // (1 << (exponent + 1))
    return -q if neg else q


def round_half_away_float(v: float) -> int:
    """Same tie rule (ties away from zero) applied to a continuous double -- used only for the
    Q-exact/K-exact variants' OWN re-application of RoPE's production round, where the pre-
    rotation input is already a double (not an integer numerator/exponent pair)."""
    if v >= 0:
        return math.floor(v + 0.5)
    return math.ceil(v - 0.5)


def clamp127(v: int) -> int:
    if v > 127:
        return 127
    if v < -127:
        return -127
    return v


def read_scores_dump(path: Path):
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


def read_qwide_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, hidden_size = int(header[0]), int(header[1])
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, r, s = int(parts[0]), int(parts[1]), int(parts[2])
            wide = [int(x) for x in parts[3 : 3 + hidden_size]]
            rows[layer] = {"r": r, "s": s, "wide": wide}
    return rows


def read_kacc_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, head_dim = int(header[0]), int(header[1])
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, position, kv_head = int(parts[0]), int(parts[1]), int(parts[2])
            normed_m, normed_e, r_t, e_t = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
            kacc = [int(x) for x in parts[7 : 7 + head_dim]]
            rows[(layer, position, kv_head)] = {
                "normed_m": normed_m, "normed_e": normed_e, "r_t": r_t, "e_t": e_t, "kacc": kacc,
            }
    return rows


def read_rope_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        width, pairs = int(header[0]), int(header[1])
        for _ in range(width):
            parts = f.readline().split()
            pos = int(parts[0])
            vals = [int(x) for x in parts[1:]]
            cos_sin = [(vals[2 * i], vals[2 * i + 1]) for i in range(pairs)]
            rows[pos] = cos_sin
    return rows, width


def read_float_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, width = int(header[0]), int(header[1])
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, head, w = int(parts[0]), int(parts[1]), int(parts[2])
            idx = 3
            scores = [float(x) for x in parts[idx : idx + w]]
            idx += w
            probs = [float(x) for x in parts[idx : idx + w]]
            rows[(layer, head)] = {"scores": scores, "probs": probs}
    return rows, width


# --- Q quantization -----------------------------------------------------------------------

def q_prod_code(x_i: int, r: int, s: int) -> int:
    """RequantTokenCodeWide, PRODUCTION (src/intmath.cpp:468-482)."""
    exponent = 62 - s
    return clamp127(round_half_away_int(x_i * 127 * r, exponent))


def q_exact_real(x_i: int, r: int, s: int) -> float:
    """RequantTokenCodeWide's own continuous target -- no round, no clamp."""
    return (x_i * 127 * r) / (2.0 ** (62 - s))


# --- K quantization (landing-rescale requantization chain) --------------------------------

def k_prod_code(kacc_i: int, m_a: int, r_t: int, e_a: int, e_t: int) -> int:
    """LandingRescale + ClampRopeCode, PRODUCTION (src/forward/forward_sites.cpp:301-441)."""
    exponent = 62 - (e_a - e_t)
    numerator = kacc_i * m_a * r_t
    if exponent >= 0:
        return clamp127(round_half_away_int(numerator, exponent))
    # negative composite exponent: exact left shift, no rounding (forward_sites.cpp:389-413)
    return clamp127(numerator << (-exponent))


def k_exact_real(kacc_i: int, m_a: int, r_t: int, e_a: int, e_t: int) -> float:
    exponent = 62 - (e_a - e_t)
    return (kacc_i * m_a * r_t) / (2.0 ** exponent)


# --- RoPE -----------------------------------------------------------------------------------

def rope_pair_prod_mixed(x, y, cos_q30: int, sin_q30: int):
    """Same PRODUCTION rotation, but xr/yr may be float (when x,y are the Q/K-exact continuous
    pre-rotation values) -- round_half_away_float used in that case."""
    xr = x * cos_q30 - y * sin_q30
    yr = x * sin_q30 + y * cos_q30
    if isinstance(x, int) and isinstance(y, int):
        rx = round_half_away_int(xr, ROPE_FRAC_BITS)
        ry = round_half_away_int(yr, ROPE_FRAC_BITS)
    else:
        rx = round_half_away_float(xr / (2.0 ** ROPE_FRAC_BITS))
        ry = round_half_away_float(yr / (2.0 ** ROPE_FRAC_BITS))
    return clamp127(rx), clamp127(ry)


def rope_pair_exact(x, y, cos_q30: int, sin_q30: int):
    """RopeApplyPair's own continuous target -- no round, no clamp."""
    xr = x * cos_q30 - y * sin_q30
    yr = x * sin_q30 + y * cos_q30
    return xr / (2.0 ** ROPE_FRAC_BITS), yr / (2.0 ** ROPE_FRAC_BITS)


def apply_rope_vector(pre, position_cos_sin, mode: str):
    """pre: length-HEAD_DIM sequence (int codes or float exact values). mode in
    {'prod','ropeexact'}: production quantization stays as given in `pre`; only the ROTATION's
    own round/clamp is toggled."""
    out = [0] * HEAD_DIM
    for i in range(PAIRS):
        cos_q30, sin_q30 = position_cos_sin[i]
        x, y = pre[2 * i], pre[2 * i + 1]
        if mode == "prod":
            rx, ry = rope_pair_prod_mixed(x, y, cos_q30, sin_q30)
        elif mode == "ropeexact":
            rx, ry = rope_pair_exact(x, y, cos_q30, sin_q30)
        else:
            raise ValueError(mode)
        out[2 * i], out[2 * i + 1] = rx, ry
    return out


def dot(a, b) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def shift_by_max(scores):
    peak = max(scores)
    return [s - peak for s in scores]


def softmax_from_real_logits(logits):
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def tvd(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def main():
    worktree = Path(r"D:\SuperSLM\.worktrees\t1787-raw-score-divergence")
    outdir = worktree / "out" / "t1787"

    all_rows = []
    self_check_fail = 0
    matmul_mismatches = []

    # Per-prompt, per-position score-space comparison accumulators (ticket's own literal ask:
    # grade raw scores against the float reference's pre-softmax scores directly).
    score_space_pairs = []  # (prod_centered_logit, float_centered_score) per element

    for pi in (1, 2, 3):
        eng_rows, eng_w = read_scores_dump(outdir / f"p{pi}_scores.txt")
        qwide = read_qwide_dump(outdir / f"p{pi}_qwide.txt")
        kacc = read_kacc_dump(outdir / f"p{pi}_kacc.txt")
        rope, rope_w = read_rope_dump(outdir / f"p{pi}_rope.txt")
        flt_rows, flt_w = read_float_dump(outdir / f"p{pi}_float.txt")
        assert eng_w == flt_w == rope_w, f"P{pi}: width mismatch"
        width = eng_w
        last_pos = width - 1

        for (layer, head), e in eng_rows.items():
            q_ln2, q_b, q_c = e["q_ln2"], e["q_b"], e["q_c"]
            kv_head = head // GROUP

            qc = qwide[layer]
            r, s = qc["r"], qc["s"]
            q_wide_head = qc["wide"][head * HEAD_DIM : (head + 1) * HEAD_DIM]
            q_codes_prod = [q_prod_code(x, r, s) for x in q_wide_head]
            q_exact = [q_exact_real(x, r, s) for x in q_wide_head]

            q_rot_prod = apply_rope_vector(q_codes_prod, rope[last_pos], "prod")
            q_rot_qexact = apply_rope_vector(q_exact, rope[last_pos], "prod")
            q_rot_ropeexact_pre_prod = apply_rope_vector(q_codes_prod, rope[last_pos], "ropeexact")
            q_rot_allexact = apply_rope_vector(q_exact, rope[last_pos], "ropeexact")

            score_prod, score_qexact, score_kexact, score_ropeexact, score_allexact = [], [], [], [], []

            for k in range(width):
                kc = kacc[(layer, k, kv_head)]
                m_a, e_a, r_t, e_t = kc["normed_m"], kc["normed_e"], kc["r_t"], kc["e_t"]
                k_wide_head = kc["kacc"]

                k_codes_prod = [k_prod_code(v, m_a, r_t, e_a, e_t) for v in k_wide_head]
                k_exact = [k_exact_real(v, m_a, r_t, e_a, e_t) for v in k_wide_head]

                k_rot_prod = apply_rope_vector(k_codes_prod, rope[k], "prod")
                k_rot_kexact = apply_rope_vector(k_exact, rope[k], "prod")
                k_rot_ropeexact = apply_rope_vector(k_codes_prod, rope[k], "ropeexact")
                k_rot_allexact = apply_rope_vector(k_exact, rope[k], "ropeexact")

                score_prod.append(dot(q_rot_prod, k_rot_prod))
                score_qexact.append(dot(q_rot_qexact, k_rot_prod))
                score_kexact.append(dot(q_rot_prod, k_rot_kexact))
                score_ropeexact.append(dot(q_rot_ropeexact_pre_prod, k_rot_ropeexact))
                score_allexact.append(dot(q_rot_allexact, k_rot_allexact))

            # Self-check: the PRODUCTION-mode replica must reproduce the dumped integer scores
            # EXACTLY (arbitrary-precision Python ints -- both q_rot_prod/k_rot_prod are ints,
            # dot() of ints stays exact int arithmetic in Python).
            dumped = e["scores"]
            if score_prod != dumped:
                self_check_fail += 1
                mismatches = [(k, score_prod[k], dumped[k]) for k in range(width) if score_prod[k] != dumped[k]]
                print(f"SELF-CHECK FAIL P{pi} layer={layer} head={head}: replica diverges from "
                      f"dumped production scores at {len(mismatches)}/{width} positions, e.g. {mismatches[:3]}",
                      file=sys.stderr)
                continue
            else:
                # Score-matmul-accumulation no-op check: score_prod matched EXACTLY, which is
                # only possible if the accumulation step introduced zero discrepancy -- record
                # this row as a pass for the explicit "does accumulation contribute" check.
                matmul_mismatches.append(0)

            # --- Real-logit recovery (q_ln2, ALREADY PROVEN EXACT, T-1768) + softmax + TVD ---
            def to_probs(scores_row):
                shifted = shift_by_max(scores_row)
                logits = [x * math.log(2.0) / q_ln2 for x in shifted]
                return softmax_from_real_logits(logits)

            p_float = flt_rows[(layer, head)]["probs"]
            p_prod = to_probs(score_prod)
            p_qexact = to_probs(score_qexact)
            p_kexact = to_probs(score_kexact)
            p_ropeexact = to_probs(score_ropeexact)
            p_allexact = to_probs(score_allexact)

            baseline_tvd = tvd(p_prod, p_float)
            tvd_qexact = tvd(p_qexact, p_float)
            tvd_kexact = tvd(p_kexact, p_float)
            tvd_ropeexact = tvd(p_ropeexact, p_float)
            tvd_allexact = tvd(p_allexact, p_float)

            all_rows.append({
                "prompt": pi, "layer": layer, "head": head, "width": width,
                "baseline": baseline_tvd,
                "q": tvd_qexact, "k": tvd_kexact, "r": tvd_ropeexact, "allexact": tvd_allexact,
            })

            # --- Score-space comparison (ticket's literal ask): production's recovered real
            # logit vs the float reference's OWN pre-softmax score, both centered by their own
            # row max (softmax-invariance -- only relative differences are physically meaningful,
            # StandardsDocument.md 5.4's same-quantity requirement: an UNcentered comparison would
            # be comparing two arbitrary origins, not the same quantity). ---
            float_scores = flt_rows[(layer, head)]["scores"]
            prod_centered = shift_by_max([x * math.log(2.0) / q_ln2 for x in shift_by_max(score_prod)])
            float_centered = shift_by_max(float_scores)
            for a, b in zip(prod_centered, float_centered):
                score_space_pairs.append((a, b))

    print(f"self_check_failures={self_check_fail} / {len(all_rows) + self_check_fail} rows")
    assert self_check_fail == 0, "Python replica does not match production -- STOP, do not trust anything below"
    print(f"score-matmul accumulation no-op check: {len(matmul_mismatches)}/{len(matmul_mismatches)} rows "
          f"reproduce the dumped PRODUCTION integer score EXACTLY via independent Python re-accumulation "
          f"(arbitrary-precision integer dot product) -- accumulation contributes 0, confirmed by execution")

    # --- Score-space regression (ticket's literal ask) ---
    n = len(score_space_pairs)
    sum_x = sum(a for a, b in score_space_pairs)
    sum_y = sum(b for a, b in score_space_pairs)
    mean_x, mean_y = sum_x / n, sum_y / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in score_space_pairs)
    var_x = sum((a - mean_x) ** 2 for a, b in score_space_pairs)
    var_y = sum((b - mean_y) ** 2 for a, b in score_space_pairs)
    slope = cov / var_x if var_x > 0 else float("nan")
    corr = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else float("nan")
    rmse_raw = math.sqrt(sum((a - b) ** 2 for a, b in score_space_pairs) / n)
    rmse_scaled = math.sqrt(sum((slope * a - b) ** 2 for a, b in score_space_pairs) / n)
    print(f"\nscore-space comparison (production recovered real logit, row-max-centered, vs float "
          f"reference's own pre-softmax score, row-max-centered), n={n} elements:")
    print(f"  correlation={corr:.4f}  best-fit scale (float = slope * production)={slope:.4f}  "
          f"RMSE(raw, slope=1)={rmse_raw:.4f}  RMSE(after best-fit scale)={rmse_scaled:.4f}")
    print(f"  1/sqrt(head_dim)={1/math.sqrt(HEAD_DIM):.6f}  sqrt(head_dim)={math.sqrt(HEAD_DIM):.6f}  "
          f"(reference values in case the fitted scale lands near either)")
    std_y = math.sqrt(var_y / n)
    print(f"  context: float-side centered score std={std_y:.4f}, range=[{min(b for a,b in score_space_pairs):.2f}, "
          f"{max(b for a,b in score_space_pairs):.2f}] -- RMSE(after best-fit scale)={rmse_scaled:.4f} is "
          f"{100*rmse_scaled/std_y:.1f}% of that std")

    # Resolving power (n=3 prompts) for each stage's grand-pooled contribution.
    def per_prompt_contrib(key_a, key_b):
        vals = []
        for pi in (1, 2, 3):
            rs = [x for x in all_rows if x["prompt"] == pi]
            vals.append(sum(x[key_a] - x[key_b] for x in rs) / len(rs))
        mean = sum(vals) / 3
        var = sum((v - mean) ** 2 for v in vals) / 3
        return vals, mean, math.sqrt(var)

    print("\nResolving power (n=3 prompts, grand-pooled contribution mean vs inter-prompt std):")
    for label, key in (("Q (query quant)", "q"), ("K (key requant chain)", "k"), ("RoPE", "r")):
        vals, mean, std = per_prompt_contrib("baseline", key)
        ratio = mean / std if std > 0 else float("inf")
        print(f"  {label}: per-prompt={['%.4f' % v for v in vals]} mean={mean:.4f} std={std:.4f} "
              f"mean/std={ratio:.2f} -- {'resolved from zero' if abs(ratio) >= 2 else 'NOT resolved from zero at this n'}")

    # --- Per-layer budget table ---
    by_layer = defaultdict(list)
    for r in all_rows:
        by_layer[r["layer"]].append(r)

    print(f"\n{'layer':>5} {'n':>4} {'base':>8} {'Qex':>8} {'Kex':>8} {'Rex':>8} {'allex':>8} "
          f"{'Q contrib':>10} {'K contrib':>10} {'R contrib':>10} {'sum':>10} {'residual':>10}")
    for layer in sorted(by_layer):
        rs = by_layer[layer]
        n_l = len(rs)
        base = sum(x["baseline"] for x in rs) / n_l
        q_ = sum(x["q"] for x in rs) / n_l
        k_ = sum(x["k"] for x in rs) / n_l
        r_ = sum(x["r"] for x in rs) / n_l
        allex = sum(x["allexact"] for x in rs) / n_l
        cq, ck, cr = base - q_, base - k_, base - r_
        csum = cq + ck + cr
        residual = base - csum
        print(f"{layer:>5} {n_l:>4} {base:>8.4f} {q_:>8.4f} {k_:>8.4f} {r_:>8.4f} {allex:>8.4f} "
              f"{cq:>10.4f} {ck:>10.4f} {cr:>10.4f} {csum:>10.4f} {residual:>10.4f}")

    n_total = len(all_rows)
    base = sum(x["baseline"] for x in all_rows) / n_total
    q_ = sum(x["q"] for x in all_rows) / n_total
    k_ = sum(x["k"] for x in all_rows) / n_total
    r_ = sum(x["r"] for x in all_rows) / n_total
    allex = sum(x["allexact"] for x in all_rows) / n_total
    cq, ck, cr = base - q_, base - k_, base - r_
    csum = cq + ck + cr
    residual = base - csum
    print(f"\nGrand pooled (n={n_total}): baseline_TVD={base:.4f}  Q-exact={q_:.4f}  K-exact={k_:.4f}  "
          f"RoPE-exact={r_:.4f}  all-exact={allex:.4f}")
    print(f"contribution Q(query quantization)={cq:.4f} ({100*cq/base:.1f}%)  "
          f"K(key requant chain)={ck:.4f} ({100*ck/base:.1f}%)  RoPE={cr:.4f} ({100*cr/base:.1f}%)")
    print(f"sum of contributions={csum:.4f} ({100*csum/base:.1f}% of baseline); "
          f"residual (closure gap)={residual:.4f} ({100*residual/base:.1f}%)")
    print(f"all-exact TVD (Q+K+RoPE all exact simultaneously)={allex:.4f} -- compare to residual "
          f"above as a second, independent measure of the same closure gap")

    print("\nLayer 0 specifically, per-prompt (resolving power for the layer-0 all-exact effect, n=3):")
    for pi in (1, 2, 3):
        rs = [x for x in all_rows if x["prompt"] == pi and x["layer"] == 0]
        base_l0 = sum(x["baseline"] for x in rs) / len(rs)
        allex_l0 = sum(x["allexact"] for x in rs) / len(rs)
        print(f"  P{pi}: n={len(rs)} baseline={base_l0:.4f} all_exact={allex_l0:.4f} "
              f"delta={base_l0-allex_l0:.4f} ({100*(base_l0-allex_l0)/base_l0:.1f}% of layer-0 baseline)")

    print("\nPer-prompt grand-pooled baseline TVD (resolving-power check, n=3):")
    for pi in (1, 2, 3):
        rs = [x for x in all_rows if x["prompt"] == pi]
        print(f"  P{pi}: n={len(rs)} baseline_TVD={sum(x['baseline'] for x in rs)/len(rs):.4f} "
              f"Q_contrib={sum(x['baseline']-x['q'] for x in rs)/len(rs):.4f} "
              f"K_contrib={sum(x['baseline']-x['k'] for x in rs)/len(rs):.4f} "
              f"R_contrib={sum(x['baseline']-x['r'] for x in rs)/len(rs):.4f}")


if __name__ == "__main__":
    main()
