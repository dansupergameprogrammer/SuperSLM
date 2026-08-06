#!/usr/bin/env python3
"""T-1789: compute a construction that closes the attention-path fidelity gap, and measure what
any attention-path construction CAN close.

Two experiments, one machinery:

  E1 -- ATTRIBUTION (H1). Full-chain production replica (h[] -> RmsNorm -> Q/K projection ->
    fold -> bias -> funnel/landing -> RoPE -> score GEMM -> q_ln2 recovery), proven bit-exact
    against production dumps first (self-checks SC1-SC4 below), then run on the FLOAT MODEL'S OWN
    per-layer input hidden state (int8-injected at the engine's own representation). If the
    engine's attention path, handed the float model's own input, lands near the float reference,
    the measured divergence is upstream residual-stream drift, not attention-path arithmetic.
    A CONTROL arm (float attention with the checkpoint's true weights, on the SAME int8-injected
    hidden state) isolates the injection quantization's own cost.

  E2 -- CONSTRUCTION (H2/H3). Code-width sweep: carry F = 2^b extra fractional bits through the
    Q/K score path (requant target 127*F, landing target scaled by F, RoPE rounds to the F-fine
    code grid, clamp +/-127*F), production semantics otherwise IDENTICAL (same funnel r/s, same
    landing r_t/e_t, same Q30 tables, same ties-away rounding, same integer score GEMM; recovery
    divides by F^2). b in {2, 4, 8}; plus Q-only and K-only b=8 arms; plus the all-exact bound
    (T-1787's own construction) recomputed in-cell for continuity.

GRADING: every arm's scores are recovered to real logits (score * ln2 / q_ln2 / F^2), exact
softmax, TVD against the FIXED float32 reference probabilities (read once, never recomputed,
taking no input from the engine's construction -- T-1778 SS4-SS6's reference discipline, T-1786's
fixed-reference rule). The engine-side baseline is graded through the IDENTICAL pipeline so every
comparison is one-variable.

SELF-CHECK DISCIPLINE (StandardsDocument.md SS5.4): SC1 normed codes AND normed_scale reproduced
from h[] + site constant (validates FloorDivI64/ISqrt/funnel ports AND the empty-incoming scale
composition this ticket newly relies on); SC2 Q chain (GEMM vs dumped raw, fold vs dumped
folded_prebias, BIAS PORT vs dumped wide -- the piece T-1788 only reconciled by subtraction, here
re-derived from the dumped bias constants and checked; funnel r/s; q_ln2 triple); SC3 K chain
(same, at every position); SC4 full-path production scores bit-exact vs dump (T-1787's own gate,
extended one stage earlier to start from h[]). A failure anywhere hard-stops before any arm is
computed.
"""

from __future__ import annotations

import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKTREE = Path(r"D:\SuperSLM\.worktrees\t1789-attention-solve")
OUTDIR = WORKTREE / "out" / "t1789"

HIDDEN_SIZE = 1536
HEAD_DIM = 128
NUM_HEADS = 12
NUM_KV_HEADS = 2
GROUP = NUM_HEADS // NUM_KV_HEADS
KV_HIDDEN_SIZE = NUM_KV_HEADS * HEAD_DIM
PAIRS = HEAD_DIM // 2
ROPE_FRAC_BITS = 30
NORM_FRAC_BITS = 16
CHECKPOINT_LAYERS = (0, 1, 2, 9, 18, 27)
K_BIAS_Q_FORMAT = 30  # kBiasQFormat, include/superslm/intmath.h:537
RMS_NORM_EPS = 1e-06  # config.json rms_norm_eps (checked at source, see run log)

_KIEXP_PINNED_LN2 = float.fromhex("0x1.62e42fefa39efp-1")
K_IEXP_LN2_Q = int(_KIEXP_PINNED_LN2 * float(1 << 30))
K_IEXP_B_Q = int(1.353 * float(1 << 30))
K_IEXP_CA_Q = int((0.344 / 0.3585) * float(1 << 30))
INT32_MAX = 2**31 - 1
LN2 = math.log(2.0)


# ============================================================================================
# Ported primitives (T-1787/T-1788's own, verified there and re-verified by SC1-SC4 here)
# ============================================================================================

def clz64(n: int) -> int:
    assert n > 0
    return 64 - n.bit_length()


def normalize_scale(d_prime: int):
    p = 63 - clz64(d_prime)
    s = 30 - p
    dn = (d_prime << s) if s >= 0 else (d_prime >> 1)
    return dn, s


def dynamic_scale_reciprocal(dn: int) -> int:
    assert dn > 0
    return (2 * (1 << 62) + dn) // (2 * dn)


def round_half_away_int(numerator: int, exponent: int) -> int:
    if numerator == 0:
        return 0
    neg = numerator < 0
    mag = -numerator if neg else numerator
    q = (mag * 2 + (1 << exponent)) // (1 << (exponent + 1))
    return -q if neg else q


def clampF(v: int, limit: int) -> int:
    if v > limit:
        return limit
    if v < -limit:
        return -limit
    return v


def requant_code_wide_F(x_i: int, r: int, s: int, F: int) -> int:
    """RequantTokenCodeWide generalized: production is F=1 (target 127, clamp +/-127); the
    width-sweep candidates use F=2^b (target 127*F, clamp +/-127*F)."""
    exponent = 62 - s
    return clampF(round_half_away_int(x_i * 127 * F * r, exponent), 127 * F)


def landing_code_F(kacc_i: int, m_a: int, r_t: int, e_a: int, e_t: int, F: int) -> int:
    """LandingRescale + ClampRopeCode generalized the same way (production F=1)."""
    exponent = 62 - (e_a - e_t)
    numerator = kacc_i * m_a * r_t * F
    if exponent >= 0:
        return clampF(round_half_away_int(numerator, exponent), 127 * F)
    return clampF(numerator << (-exponent), 127 * F)


def bias_reconcile(b: int, r_a: int, e_a: int) -> int:
    """BiasReconcile (forward_sites.cpp:271-299): round_half_away(b * r_a / 2^(30 + 62 + e_a))."""
    exponent = K_BIAS_Q_FORMAT + 62 + e_a
    numerator = b * r_a
    if exponent >= 0:
        return round_half_away_int(numerator, exponent)
    return numerator << (-exponent)


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    result = (a * b + (1 << 30)) >> 31
    return INT32_MAX if result > INT32_MAX else result


def rounding_divide_by_pot_i32(x: int, exponent: int) -> int:
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    ux = x & 0xFFFFFFFF
    remainder = ux & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    shifted = x >> exponent
    return shifted + (1 if remainder > threshold else 0)


def apply_weight_scale_fold_int(acc: int, identity: int, mult: int, shift: int) -> int:
    if identity != 0:
        return acc
    acc32 = ((acc + 2**31) % 2**32) - 2**31
    hm = saturating_rounding_doubling_high_mul(acc32, mult)
    return rounding_divide_by_pot_i32(hm, shift)


def combine_carried_scale(am: int, ae: int, bm: int, be: int):
    e = ae + be + 31
    m = saturating_rounding_doubling_high_mul(((am + 2**31) % 2**32) - 2**31,
                                              ((bm + 2**31) % 2**32) - 2**31)
    if m < (1 << 30):
        m <<= 1
        e -= 1
    return m, e


def iexp_scale_constants(m: int, e: int):
    shift_ln2 = 30 + 62 + e
    shift_c = 30 + 124 + 2 * e
    r_m = dynamic_scale_reciprocal(m)
    num_ln2 = K_IEXP_LN2_Q * r_m
    num_b = K_IEXP_B_Q * r_m
    num_c = K_IEXP_CA_Q * r_m * r_m
    q_ln2 = num_ln2 >> shift_ln2 if shift_ln2 >= 0 else num_ln2 << (-shift_ln2)
    q_b = num_b >> shift_ln2 if shift_ln2 >= 0 else num_b << (-shift_ln2)
    q_c = num_c >> shift_c if shift_c >= 0 else num_c << (-shift_c)
    return q_ln2, q_b, q_c


def rope_rotate_codes_F(codes, cos_sin, F: int):
    """Production RopeApplyPair + ClampRopeCode on integer codes, generalized to the F-fine grid
    (production F=1: round to int, clamp +/-127)."""
    out = [0] * HEAD_DIM
    limit = 127 * F
    for i in range(PAIRS):
        c, s = cos_sin[i]
        x, y = codes[2 * i], codes[2 * i + 1]
        xr = x * c - y * s
        yr = x * s + y * c
        out[2 * i] = clampF(round_half_away_int(xr, ROPE_FRAC_BITS), limit)
        out[2 * i + 1] = clampF(round_half_away_int(yr, ROPE_FRAC_BITS), limit)
    return out


def rope_rotate_real(vec, cos_sin):
    out = [0.0] * HEAD_DIM
    for i in range(PAIRS):
        c, s = cos_sin[i]
        x, y = vec[2 * i], vec[2 * i + 1]
        out[2 * i] = (x * c - y * s) / (2.0 ** ROPE_FRAC_BITS)
        out[2 * i + 1] = (x * s + y * c) / (2.0 ** ROPE_FRAC_BITS)
    return out


def softmax_real(logits):
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def tvd(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def rmsnorm_chain(h_codes, gain, site_m, site_e):
    """RmsNormSite, complete: h[] -> normed codes + (r, s) + normed_scale.
    normed_scale composition: empty incoming -> combine(site_constant, (dn, -s)) -- validated by
    SC1 against production's own dumped normed_m/e before being trusted on foreign input."""
    sumsq = sum(int(hi) * int(hi) for hi in h_codes)
    root = math.isqrt((sumsq << (2 * NORM_FRAC_BITS)) // HIDDEN_SIZE)
    root = root if root > 1 else 1
    # FloorDivI64 is floor division; Python's // on ints IS floor division (both round toward
    # negative infinity), so the port is the direct expression:
    wide = [((int(h_codes[i]) << (2 * NORM_FRAC_BITS)) // root) * int(gain[i])
            for i in range(HIDDEN_SIZE)]
    d_prime = max(max((abs(x) for x in wide), default=0), 1)
    dn, s = normalize_scale(d_prime)
    r = dynamic_scale_reciprocal(dn)
    codes = [clampF(round_half_away_int(w * 127 * r, 62 - s), 127) for w in wide]
    nm, ne = combine_carried_scale(site_m, site_e, dn, -s)
    return codes, wide, r, s, nm, ne


# ============================================================================================
# Dump readers (T-1788 formats + this ticket's two additions)
# ============================================================================================

def read_normed_dump(path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        num_rows, hidden_size = (int(x) for x in f.readline().split())
        assert hidden_size == HIDDEN_SIZE
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, position = int(parts[0]), int(parts[1])
            nm, ne, r, s = (int(x) for x in parts[2:6])
            h = [int(x) for x in parts[6:6 + hidden_size]]
            normed = [int(x) for x in parts[6 + hidden_size:6 + 2 * hidden_size]]
            rows[(layer, position)] = {"h": h, "normed": normed, "normed_m": nm, "normed_e": ne,
                                       "r": r, "s": s}
    return rows


def read_qacc_dump(path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        num_rows, hidden_size = (int(x) for x in f.readline().split())
        for _ in range(num_rows):
            parts = f.readline().split()
            layer = int(parts[0])
            q_r, q_s, site_m, site_e = (int(x) for x in parts[1:5])
            raw = [int(x) for x in parts[5:5 + hidden_size]]
            folded = [int(x) for x in parts[5 + hidden_size:5 + 2 * hidden_size]]
            wide = [int(x) for x in parts[5 + 2 * hidden_size:5 + 3 * hidden_size]]
            rows[layer] = {"q_r": q_r, "q_s": q_s, "site_m": site_m, "site_e": site_e,
                           "raw": raw, "folded_prebias": folded, "wide": wide}
    return rows


def read_kacc_dump(path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        num_rows, head_dim = (int(x) for x in f.readline().split())
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, position, kv_head = int(parts[0]), int(parts[1]), int(parts[2])
            nm, ne, r_t, e_t = (int(x) for x in parts[3:7])
            raw = [int(x) for x in parts[7:7 + head_dim]]
            folded = [int(x) for x in parts[7 + head_dim:7 + 2 * head_dim]]
            kacc = [int(x) for x in parts[7 + 2 * head_dim:7 + 3 * head_dim]]
            rows[(layer, position, kv_head)] = {"normed_m": nm, "normed_e": ne, "r_t": r_t,
                                                "e_t": e_t, "raw": raw, "folded_prebias": folded,
                                                "kacc": kacc}
    return rows


def read_scores_dump(path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        num_rows, width = (int(x) for x in f.readline().split())
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, head, w = int(parts[0]), int(parts[1]), int(parts[2])
            q_ln2, q_b, q_c = (int(x) for x in parts[3:6])
            scores = [int(x) for x in parts[6:6 + w]]
            probs = [int(x) for x in parts[6 + w:6 + 2 * w]]
            rows[(layer, head)] = {"width": w, "q_ln2": q_ln2, "q_b": q_b, "q_c": q_c,
                                   "scores": scores, "probs": probs}
    return rows, width


def read_rope_dump(path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        width, pairs = (int(x) for x in f.readline().split())
        for _ in range(width):
            parts = f.readline().split()
            pos = int(parts[0])
            vals = [int(x) for x in parts[1:]]
            rows[pos] = [(vals[2 * i], vals[2 * i + 1]) for i in range(pairs)]
    return rows, width


def read_float_dump(path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        num_rows, width = (int(x) for x in f.readline().split())
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, head, w = int(parts[0]), int(parts[1]), int(parts[2])
            scores = [float(x) for x in parts[3:3 + w]]
            probs = [float(x) for x in parts[3 + w:3 + 2 * w]]
            rows[(layer, head)] = {"scores": scores, "probs": probs}
    return rows, width


def read_float_hidden_dump(path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        num_rows, hidden_size = (int(x) for x in f.readline().split())
        assert hidden_size == HIDDEN_SIZE
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, pos = int(parts[0]), int(parts[1])
            rows[(layer, pos)] = [float(x) for x in parts[2:2 + hidden_size]]
    return rows


def read_layerconst_dump(path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        n_layers, hidden_size, kv_hidden_size, num_kv_heads = (int(x) for x in f.readline().split())
        for _ in range(n_layers):
            parts = f.readline().split()
            idx = 0
            layer = int(parts[idx]); idx += 1
            q_id = [int(x) for x in parts[idx:idx + hidden_size]]; idx += hidden_size
            q_mult = [int(x) for x in parts[idx:idx + hidden_size]]; idx += hidden_size
            q_shift = [int(x) for x in parts[idx:idx + hidden_size]]; idx += hidden_size
            k_id = [int(x) for x in parts[idx:idx + kv_hidden_size]]; idx += kv_hidden_size
            k_mult = [int(x) for x in parts[idx:idx + kv_hidden_size]]; idx += kv_hidden_size
            k_shift = [int(x) for x in parts[idx:idx + kv_hidden_size]]; idx += kv_hidden_size
            khead_m = [int(x) for x in parts[idx:idx + num_kv_heads]]; idx += num_kv_heads
            khead_e = [int(x) for x in parts[idx:idx + num_kv_heads]]; idx += num_kv_heads
            has_qb = int(parts[idx]); idx += 1
            q_bias = None
            if has_qb:
                q_bias = [int(x) for x in parts[idx:idx + hidden_size]]; idx += hidden_size
            has_kb = int(parts[idx]); idx += 1
            k_bias = None
            if has_kb:
                k_bias = [int(x) for x in parts[idx:idx + kv_hidden_size]]; idx += kv_hidden_size
            gain = [int(x) for x in parts[idx:idx + hidden_size]]; idx += hidden_size
            # T-1789 additions (tail of line): attn_norm_site (m,e), q_site (m,e)
            norm_site_m, norm_site_e, q_site_m, q_site_e = (int(x) for x in parts[idx:idx + 4])
            idx += 4
            assert idx == len(parts), f"layerconst line not fully consumed: {idx} != {len(parts)}"
            rows[layer] = {"q_fold_identity": q_id, "q_fold_mult": q_mult, "q_fold_shift": q_shift,
                           "k_fold_identity": k_id, "k_fold_mult": k_mult, "k_fold_shift": k_shift,
                           "khead_m": khead_m, "khead_e": khead_e, "q_bias": q_bias,
                           "k_bias": k_bias, "gain": gain,
                           "norm_site_m": norm_site_m, "norm_site_e": norm_site_e,
                           "q_site_m": q_site_m, "q_site_e": q_site_e}
    return rows


def read_int8_weights(path):
    rows = {}
    data = path.read_bytes()
    off = 0
    for layer in CHECKPOINT_LAYERS:
        qn = HIDDEN_SIZE * HIDDEN_SIZE
        kn = KV_HIDDEN_SIZE * HIDDEN_SIZE
        qw = np.frombuffer(data, dtype=np.int8, count=qn, offset=off).astype(np.int64).reshape(
            HIDDEN_SIZE, HIDDEN_SIZE); off += qn
        kw = np.frombuffer(data, dtype=np.int8, count=kn, offset=off).astype(np.int64).reshape(
            KV_HIDDEN_SIZE, HIDDEN_SIZE); off += kn
        rows[layer] = {"q_weight": qw, "k_weight": kw}
    assert off == len(data)
    return rows


def rope_perm_indices(n_rows: int):
    """Engine RoPE row order from HF checkpoint row order: within each 128-row head block,
    engine row 2i is HF row i and engine row 2i+1 is HF row i+64 (interleaved-adjacent pairs vs
    HF's rotate-half pairing). VERIFIED against data before use: the engine's own int8 weight row
    j has |cosine| ~= 1.0000 against the HF row perm(j) and ~= 0.1 against HF row j, both Q and K,
    layers 0 and 27 (this ticket's own permutation check, run log). T-1788's true-weight arms did
    NOT apply this permutation and then rotated with interleaved pairing -- the mis-pairing this
    ticket's control arm initially reproduced and this function corrects."""
    idx = np.zeros(n_rows, dtype=int)
    for h in range(n_rows // HEAD_DIM):
        base = h * HEAD_DIM
        for i in range(HEAD_DIM // 2):
            idx[base + 2 * i] = base + i
            idx[base + 2 * i + 1] = base + i + HEAD_DIM // 2
    return idx


def read_true_weights(path):
    rows = {}
    data = path.read_bytes()
    off = 0
    pq = rope_perm_indices(HIDDEN_SIZE)
    pk = rope_perm_indices(KV_HIDDEN_SIZE)
    for layer in CHECKPOINT_LAYERS:
        qn = HIDDEN_SIZE * HIDDEN_SIZE
        kn = KV_HIDDEN_SIZE * HIDDEN_SIZE
        qw = np.frombuffer(data, dtype=np.float64, count=qn, offset=off).reshape(
            HIDDEN_SIZE, HIDDEN_SIZE); off += qn * 8
        qb = np.frombuffer(data, dtype=np.float64, count=HIDDEN_SIZE, offset=off).copy(); off += HIDDEN_SIZE * 8
        kw = np.frombuffer(data, dtype=np.float64, count=kn, offset=off).reshape(
            KV_HIDDEN_SIZE, HIDDEN_SIZE); off += kn * 8
        kb = np.frombuffer(data, dtype=np.float64, count=KV_HIDDEN_SIZE, offset=off).copy(); off += KV_HIDDEN_SIZE * 8
        # Permute into the ENGINE's own interleaved RoPE row order so the interleaved rotation
        # this script applies everywhere is the correct pairing for these weights too.
        rows[layer] = {"q_weight": qw[pq, :], "q_bias": qb[pq],
                       "k_weight": kw[pk, :], "k_bias": kb[pk]}
    assert off == len(data)
    return rows


def read_ln_gamma(path):
    rows = {}
    data = path.read_bytes()
    off = 0
    for layer in CHECKPOINT_LAYERS:
        rows[layer] = np.frombuffer(data, dtype=np.float64, count=HIDDEN_SIZE, offset=off).copy()
        off += HIDDEN_SIZE * 8
    assert off == len(data)
    return rows


# ============================================================================================
# The full attention-path chain (production semantics, parameterized by F and by input h)
# ============================================================================================

def inject_h(h_float):
    """Float hidden state -> engine int8 hidden codes at the engine's own representation
    (max-normalized signed int8, ties-away rounding -- the engine's own C3 rule)."""
    peak = max(abs(v) for v in h_float)
    if peak == 0.0:
        return [0] * HIDDEN_SIZE
    scale = peak / 127.0
    out = []
    for v in h_float:
        x = v / scale
        out.append(clampF(int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5)), 127))
    return out


def run_chain_for_layer(h_by_pos, width, lc, iw, rope, F_q, F_k, exact=False):
    """The engine's own attention path for ONE layer, from per-position h codes to per-head
    integer scores (or continuous scores when exact=True), production semantics throughout.
    Returns (scores_by_head, q_ln2_by_kvhead, F_q, F_k)."""
    last = width - 1
    # --- RMSNorm at every position ---
    normed = {}
    nscale = {}
    for pos in range(width):
        codes, wide, r, s, nm, ne = rmsnorm_chain(h_by_pos[pos], lc["gain"],
                                                  lc["norm_site_m"], lc["norm_site_e"])
        normed[pos] = np.array(codes, dtype=np.int64)
        nscale[pos] = (nm, ne)
    # --- K chain at every position ---
    k_rot = {}  # (pos, kv_head) -> rotated codes (int list) or real list
    kw = iw["k_weight"]
    for pos in range(width):
        nm, ne = nscale[pos]
        r_a = dynamic_scale_reciprocal(nm)
        kacc_np = kw @ normed[pos]
        kacc = [int(x) for x in kacc_np]
        for kv in range(NUM_KV_HEADS):
            off = kv * HEAD_DIM
            vec = []
            for d in range(HEAD_DIM):
                i = off + d
                a = apply_weight_scale_fold_int(kacc[i], lc["k_fold_identity"][i],
                                                lc["k_fold_mult"][i], lc["k_fold_shift"][i])
                if lc["k_bias"] is not None:
                    a += bias_reconcile(lc["k_bias"][i], r_a, ne)
                vec.append(a)
            r_t = lc["_r_t"][kv]
            e_t = lc["_e_t"][kv]
            if exact:
                kreal = [(v * nm * r_t) / (2.0 ** (62 - (ne - e_t))) for v in vec]
                k_rot[(pos, kv)] = rope_rotate_real(kreal, rope[pos])
            else:
                kcodes = [landing_code_F(v, nm, r_t, ne, e_t, F_k) for v in vec]
                k_rot[(pos, kv)] = rope_rotate_codes_F(kcodes, rope[pos], F_k)
    # --- Q chain at the last position ---
    nm, ne = nscale[last]
    r_a = dynamic_scale_reciprocal(nm)
    qw = iw["q_weight"]
    qacc_np = qw @ normed[last]
    q_wide = []
    for i in range(HIDDEN_SIZE):
        a = apply_weight_scale_fold_int(int(qacc_np[i]), lc["q_fold_identity"][i],
                                        lc["q_fold_mult"][i], lc["q_fold_shift"][i])
        if lc["q_bias"] is not None:
            a += bias_reconcile(lc["q_bias"][i], r_a, ne)
        q_wide.append(a)
    d_prime = max(max((abs(x) for x in q_wide), default=0), 1)
    dn, s = normalize_scale(d_prime)
    r = dynamic_scale_reciprocal(dn)
    qs0 = combine_carried_scale(nm, ne, lc["q_site_m"], lc["q_site_e"])
    qs = combine_carried_scale(qs0[0], qs0[1], dn, -s)
    q_ln2_by_kv = {}
    for kv in range(NUM_KV_HEADS):
        sm = combine_carried_scale(qs[0], qs[1], lc["khead_m"][kv], lc["khead_e"][kv])
        q_ln2_by_kv[kv] = iexp_scale_constants(sm[0], sm[1])[0]
    scores_by_head = {}
    for head in range(NUM_HEADS):
        kv = head // GROUP
        sl = q_wide[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        if exact:
            q_real = [(x * 127 * r) / (2.0 ** (62 - s)) for x in sl]
            q_rot = rope_rotate_real(q_real, rope[last])
            scores_by_head[head] = [sum(a * b for a, b in zip(q_rot, k_rot[(k, kv)]))
                                    for k in range(width)]
        else:
            q_codes = [requant_code_wide_F(x, r, s, F_q) for x in sl]
            q_rot = rope_rotate_codes_F(q_codes, rope[last], F_q)
            qv = np.array(q_rot, dtype=np.int64)
            scores_by_head[head] = [int(np.dot(qv, np.array(k_rot[(k, kv)], dtype=np.int64)))
                                    for k in range(width)]
    return scores_by_head, q_ln2_by_kv, (q_wide, r, s, nscale, normed)


def scores_to_probs(scores, q_ln2: int, F_q: int, F_k: int):
    logits = [x * LN2 / q_ln2 / (F_q * F_k) for x in scores]
    return softmax_real(logits)


# ============================================================================================
# Width-sweep arms recomputed from the PRODUCTION dumps (native input, no re-derivation of the
# funnel/landing scales -- exactly T-1787's substitution discipline, one variable per arm)
# ============================================================================================

def native_arm_scores(eng_rows, qacc, kacc, rope, width, layer, head, F_q, F_k, exact=False):
    kv = head // GROUP
    qc = qacc[layer]
    r, s = qc["q_r"], qc["q_s"]
    q_wide_head = qc["wide"][head * HEAD_DIM:(head + 1) * HEAD_DIM]
    last = width - 1
    if exact:
        q_rot = rope_rotate_real([(x * 127 * r) / (2.0 ** (62 - s)) for x in q_wide_head],
                                 rope[last])
    else:
        q_rot = rope_rotate_codes_F([requant_code_wide_F(x, r, s, F_q) for x in q_wide_head],
                                    rope[last], F_q)
    scores = []
    for k in range(width):
        kc = kacc[(layer, k, kv)]
        m_a, e_a, r_t, e_t = kc["normed_m"], kc["normed_e"], kc["r_t"], kc["e_t"]
        if exact:
            k_rot = rope_rotate_real(
                [(v * m_a * r_t) / (2.0 ** (62 - (e_a - e_t))) for v in kc["kacc"]], rope[k])
            scores.append(sum(a * b for a, b in zip(q_rot, k_rot)))
        else:
            k_rot = rope_rotate_codes_F(
                [landing_code_F(v, m_a, r_t, e_a, e_t, F_k) for v in kc["kacc"]], rope[k], F_k)
            scores.append(sum(a * b for a, b in zip(q_rot, k_rot)))
    return scores


# ============================================================================================
# Main
# ============================================================================================

def main():
    prompts = (1, 2, 3)
    layerconst = read_layerconst_dump(OUTDIR / "p1_layerconst.txt")
    int8_weights = read_int8_weights(OUTDIR / "weights.bin")
    true_weights = read_true_weights(OUTDIR / "true_weights.bin")
    ln_gamma = read_ln_gamma(OUTDIR / "ln_gamma.bin")

    checks = defaultdict(int)
    fails = 0

    all_rows = []  # one dict per (prompt, layer, head) with every arm's TVD

    for pi in prompts:
        normed_d = read_normed_dump(OUTDIR / f"p{pi}_normed.txt")
        qacc = read_qacc_dump(OUTDIR / f"p{pi}_qacc.txt")
        kacc = read_kacc_dump(OUTDIR / f"p{pi}_kacc.txt")
        eng_rows, eng_w = read_scores_dump(OUTDIR / f"p{pi}_scores.txt")
        rope, rope_w = read_rope_dump(OUTDIR / f"p{pi}_rope.txt")
        flt_rows, flt_w = read_float_dump(OUTDIR / f"p{pi}_float.txt")
        flt_hidden = read_float_hidden_dump(OUTDIR / f"p{pi}_float_hidden.txt")
        assert eng_w == flt_w == rope_w, f"P{pi} width mismatch"
        width = eng_w
        last = width - 1

        # r_t/e_t per (layer, kv_head), read off the kacc dump (position-invariant constants).
        for layer in CHECKPOINT_LAYERS:
            layerconst[layer]["_r_t"] = [kacc[(layer, 0, kv)]["r_t"] for kv in range(NUM_KV_HEADS)]
            layerconst[layer]["_e_t"] = [kacc[(layer, 0, kv)]["e_t"] for kv in range(NUM_KV_HEADS)]

        # ---------------- SC1: RMSNorm full chain incl. normed_scale ----------------
        for (layer, pos), nc in normed_d.items():
            lc = layerconst[layer]
            codes, wide, r, s, nm, ne = rmsnorm_chain(nc["h"], lc["gain"],
                                                      lc["norm_site_m"], lc["norm_site_e"])
            checks["sc1_normed_rows"] += 1
            if codes != nc["normed"] or (r, s) != (nc["r"], nc["s"]) or \
               (nm, ne) != (nc["normed_m"], nc["normed_e"]):
                fails += 1
                print(f"SC1 FAIL P{pi} layer={layer} pos={pos}: codes_eq={codes == nc['normed']} "
                      f"rs=({r},{s}) vs ({nc['r']},{nc['s']}) scale=({nm},{ne}) vs "
                      f"({nc['normed_m']},{nc['normed_e']})", file=sys.stderr)

        # ---------------- SC2/SC3/SC4: full-path replica vs production dumps ----------------
        h_by_pos_prod = {}
        for layer in CHECKPOINT_LAYERS:
            h_by_pos_prod[layer] = {pos: normed_d[(layer, pos)]["h"] for pos in range(width)}
        for layer in CHECKPOINT_LAYERS:
            lc = layerconst[layer]
            iw = int8_weights[layer]
            scores_by_head, q_ln2_by_kv, extras = run_chain_for_layer(
                h_by_pos_prod[layer], width, lc, iw, rope, 1, 1)
            q_wide_rep, r_rep, s_rep, nscale_rep, normed_rep = extras
            qc = qacc[layer]
            checks["sc2_q_layers"] += 1
            if q_wide_rep != qc["wide"]:
                fails += 1
                nbad = sum(1 for a, b in zip(q_wide_rep, qc["wide"]) if a != b)
                print(f"SC2 FAIL P{pi} layer={layer}: q_wide mismatch at {nbad}/{HIDDEN_SIZE}",
                      file=sys.stderr)
            for head in range(NUM_HEADS):
                e = eng_rows[(layer, head)]
                checks["sc4_score_rows"] += 1
                if scores_by_head[head] != e["scores"]:
                    fails += 1
                    nbad = sum(1 for a, b in zip(scores_by_head[head], e["scores"]) if a != b)
                    print(f"SC4 FAIL P{pi} layer={layer} head={head}: {nbad}/{width} scores differ",
                          file=sys.stderr)
                kv = head // GROUP
                checks["sc2_qln2_rows"] += 1
                if q_ln2_by_kv[kv] != e["q_ln2"]:
                    fails += 1
                    print(f"SC2 FAIL P{pi} layer={layer} head={head}: q_ln2 {q_ln2_by_kv[kv]} != "
                          f"{e['q_ln2']}", file=sys.stderr)
            # SC3: K chain spot equality is implied bit-for-bit by SC4 (scores are the dot of the
            # replica's own K codes); additionally check kacc reproduction explicitly:
            for kv in range(NUM_KV_HEADS):
                kc = kacc[(layer, last, kv)]
                nm, ne = nscale_rep[last]
                r_a = dynamic_scale_reciprocal(nm)
                off = kv * HEAD_DIM
                kacc_np = iw["k_weight"][off:off + HEAD_DIM, :] @ normed_rep[last]
                rep = []
                for d in range(HEAD_DIM):
                    i = off + d
                    a = apply_weight_scale_fold_int(int(kacc_np[d]), lc["k_fold_identity"][i],
                                                    lc["k_fold_mult"][i], lc["k_fold_shift"][i])
                    if lc["k_bias"] is not None:
                        a += bias_reconcile(lc["k_bias"][i], r_a, ne)
                    rep.append(a)
                checks["sc3_kacc_rows"] += 1
                if rep != kc["kacc"]:
                    fails += 1
                    nbad = sum(1 for a, b in zip(rep, kc["kacc"]) if a != b)
                    print(f"SC3 FAIL P{pi} layer={layer} kv={kv}: {nbad}/{HEAD_DIM} kacc differ",
                          file=sys.stderr)

        if fails:
            print(f"\n{fails} self-check failures -- STOPPING before any arm is computed",
                  file=sys.stderr)
            sys.exit(1)
        print(f"P{pi}: SC1-SC4 pass ({dict(checks)})", file=sys.stderr)

        # ---------------- Arms ----------------
        # Hybrid input: float model's own per-layer hidden state, int8-injected.
        h_by_pos_hyb = {}
        for layer in CHECKPOINT_LAYERS:
            h_by_pos_hyb[layer] = {pos: inject_h(flt_hidden[(layer, pos)]) for pos in range(width)}

        for layer in CHECKPOINT_LAYERS:
            lc = layerconst[layer]
            iw = int8_weights[layer]
            tw = true_weights[layer]
            gam = ln_gamma[layer]

            # E1 hybrid arms: production path (F=1), exact path, and W(8) path on injected h.
            hyb_prod, hyb_qln2, _ = run_chain_for_layer(h_by_pos_hyb[layer], width, lc, iw, rope, 1, 1)
            hyb_w8, hyb_w8_qln2, _ = run_chain_for_layer(h_by_pos_hyb[layer], width, lc, iw, rope,
                                                         256, 256)
            hyb_exact, hyb_ex_qln2, _ = run_chain_for_layer(h_by_pos_hyb[layer], width, lc, iw, rope,
                                                            1, 1, exact=True)

            # E1 control: float attention (true weights, float RMSNorm, exact rotation, 1/sqrt(d))
            # on the SAME injected-then-dequantized hidden state.
            ctrl_scores = {}
            deq = {}
            for pos in range(width):
                h_inj = h_by_pos_hyb[layer][pos]
                peak = max(abs(v) for v in flt_hidden[(layer, pos)])
                scale = (peak / 127.0) if peak > 0 else 0.0
                x = np.array(h_inj, dtype=np.float64) * scale
                rms = math.sqrt(float(np.mean(x * x)) + RMS_NORM_EPS)
                deq[pos] = (x / rms) * gam
            k_rot_ctrl = {}
            for pos in range(width):
                kvec = tw["k_weight"] @ deq[pos] + tw["k_bias"]
                for kv in range(NUM_KV_HEADS):
                    k_rot_ctrl[(pos, kv)] = rope_rotate_real(
                        list(kvec[kv * HEAD_DIM:(kv + 1) * HEAD_DIM]), rope[pos])
            qvec = tw["q_weight"] @ deq[last] + tw["q_bias"]
            scaling = HEAD_DIM ** -0.5
            for head in range(NUM_HEADS):
                kv = head // GROUP
                q_rot = rope_rotate_real(list(qvec[head * HEAD_DIM:(head + 1) * HEAD_DIM]),
                                         rope[last])
                ctrl_scores[head] = [scaling * sum(a * b for a, b in zip(q_rot, k_rot_ctrl[(k, kv)]))
                                     for k in range(width)]

            # we_native: T-1788's weight-exact experiment, CORRECTED (true weights in the
            # engine's own RoPE row order): true fp32 Q/K weight on the engine's own production
            # (drifted, int8-dequantized) activation, physical units, exact rotation, 1/sqrt(d).
            we_scores = {}
            normed_real = {}
            for pos in range(width):
                nc = normed_d[(layer, pos)]
                normed_real[pos] = np.array(nc["normed"], dtype=np.float64) * (
                    nc["normed_m"] * (2.0 ** nc["normed_e"]))
            k_rot_we = {}
            for pos in range(width):
                kvec = tw["k_weight"] @ normed_real[pos] + tw["k_bias"]
                for kv2 in range(NUM_KV_HEADS):
                    k_rot_we[(pos, kv2)] = rope_rotate_real(
                        list(kvec[kv2 * HEAD_DIM:(kv2 + 1) * HEAD_DIM]), rope[pos])
            qvec_we = tw["q_weight"] @ normed_real[last] + tw["q_bias"]
            for head in range(NUM_HEADS):
                kv2 = head // GROUP
                q_rot = rope_rotate_real(list(qvec_we[head * HEAD_DIM:(head + 1) * HEAD_DIM]),
                                         rope[last])
                we_scores[head] = [scaling * sum(a * b for a, b in zip(q_rot, k_rot_we[(k, kv2)]))
                                   for k in range(width)]

            for head in range(NUM_HEADS):
                kv = head // GROUP
                e = eng_rows[(layer, head)]
                p_float = flt_rows[(layer, head)]["probs"]
                q_ln2 = e["q_ln2"]

                row = {"prompt": pi, "layer": layer, "head": head}
                # native arms from dumps (one-variable substitutions on production state)
                row["baseline"] = tvd(scores_to_probs(e["scores"], q_ln2, 1, 1), p_float)
                for label, fq, fk in (("w2", 4, 4), ("w4", 16, 16), ("w8", 256, 256),
                                      ("w8_qonly", 256, 1), ("w8_konly", 1, 256)):
                    sc = native_arm_scores(eng_rows, qacc, kacc, rope, width, layer, head, fq, fk)
                    row[label] = tvd(scores_to_probs(sc, q_ln2, fq, fk), p_float)
                sc = native_arm_scores(eng_rows, qacc, kacc, rope, width, layer, head, 1, 1,
                                       exact=True)
                row["allexact"] = tvd(softmax_real([x * LN2 / q_ln2 for x in sc]), p_float)
                # hybrid arms
                row["hyb_prod"] = tvd(scores_to_probs(hyb_prod[head], hyb_qln2[kv], 1, 1), p_float)
                row["hyb_w8"] = tvd(scores_to_probs(hyb_w8[head], hyb_w8_qln2[kv], 256, 256), p_float)
                row["hyb_exact"] = tvd(softmax_real([x * LN2 / hyb_ex_qln2[kv] for x in
                                                     hyb_exact[head]]), p_float)
                row["control"] = tvd(softmax_real(ctrl_scores[head]), p_float)
                row["we_native"] = tvd(softmax_real(we_scores[head]), p_float)
                all_rows.append(row)
        print(f"P{pi}: arms computed ({sum(1 for r in all_rows if r['prompt'] == pi)} rows)",
              file=sys.stderr)

    print(f"\nTOTAL SELF-CHECKS: {dict(checks)}  FAILURES: {fails}\n")

    arm_labels = ["baseline", "w2", "w4", "w8", "w8_qonly", "w8_konly", "allexact",
                  "hyb_prod", "hyb_w8", "hyb_exact", "control", "we_native"]

    by_layer = defaultdict(list)
    for r in all_rows:
        by_layer[r["layer"]].append(r)

    header = f"{'layer':>5} {'n':>4}" + "".join(f" {a:>9}" for a in arm_labels)
    print(header)
    for layer in sorted(by_layer):
        rs = by_layer[layer]
        line = f"{layer:>5} {len(rs):>4}"
        for a in arm_labels:
            line += f" {sum(x[a] for x in rs) / len(rs):>9.4f}"
        print(line)
    n = len(all_rows)
    line = f"{'ALL':>5} {n:>4}"
    for a in arm_labels:
        line += f" {sum(x[a] for x in all_rows) / n:>9.4f}"
    print(line)

    print("\nPer-prompt grand-pooled (resolving power, n=3):")
    for a in arm_labels:
        vals = []
        for pi in prompts:
            rs = [x for x in all_rows if x["prompt"] == pi]
            vals.append(sum(x[a] for x in rs) / len(rs))
        mean = sum(vals) / 3
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / 3)
        print(f"  {a:>10}: per-prompt={['%.4f' % v for v in vals]} mean={mean:.4f} std={std:.4f}")

    # Key deltas with per-prompt resolving power
    print("\nKey deltas (mean over prompts, +/- inter-prompt std, mean/std):")
    for label, a, b in (("upstream drift share (baseline - hyb_prod)", "baseline", "hyb_prod"),
                        ("attention-path local (hyb_prod - control)", "hyb_prod", "control"),
                        ("injection cost (control)", "control", None),
                        ("W(8) native gain (baseline - w8)", "baseline", "w8"),
                        ("W(8) vs all-exact (w8 - allexact)", "w8", "allexact"),
                        ("hybrid W(8) residual (hyb_w8)", "hyb_w8", None),
                        ("weight-exact corrected gain (baseline - we_native)", "baseline",
                         "we_native")):
        vals = []
        for pi in prompts:
            rs = [x for x in all_rows if x["prompt"] == pi]
            va = sum(x[a] for x in rs) / len(rs)
            vb = sum(x[b] for x in rs) / len(rs) if b else 0.0
            vals.append(va - vb)
        mean = sum(vals) / 3
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / 3)
        ratio = mean / std if std > 0 else float("inf")
        print(f"  {label}: per-prompt={['%.4f' % v for v in vals]} mean={mean:.4f} std={std:.4f} "
              f"mean/std={ratio:.2f}")


if __name__ == "__main__":
    main()
