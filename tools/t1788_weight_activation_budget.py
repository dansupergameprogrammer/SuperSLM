#!/usr/bin/env python3
"""T-1788: does Q/K weight quantization and/or RMSNorm's own activation quantization explain the
80.1% raw-score-divergence residual T-1787 (D-SLM1296) left unexplained after removing Query
quantization, the Key requantization chain, and RoPE?

METHOD, extending T-1787's own substitution discipline one level further upstream. T-1787's own
"all-exact" variant (Q's own funnel-round removed, K's own landing-round removed, RoPE's own round
removed, but the SAME production int8 Q/K weight and the SAME production int8 RmsNorm output in
every variant) is THIS ticket's own baseline -- candidate 0 below, recomputed fresh on this
ticket's own 3 new held-out prompts from this ticket's own dumps, as a methodology-continuity
self-check before anything new is measured.

On top of that baseline, exactly two candidates are varied, individually (candidates 1, 2) and
jointly (candidate 3):

  CANDIDATE 1 -- WEIGHT-EXACT. The engine's own int8 Q/K weight is replaced by the checkpoint's
    TRUE float32 weight (`t1788_true_weights_dump.py`: an independent source, loaded fresh from
    the HF checkpoint, zero dependency on the engine's own int8 construction). Computed in
    PHYSICAL REAL UNITS throughout: normed_code * normed_scale -> real activation; true_weight @
    real_activation + true_bias -> real query/key; RoPE's own rotation applied exactly (rotation
    is unit-preserving: x*cos_real - y*sin_real is correct whatever units x/y carry, so the SAME
    cos_q30/sin_q30 tables, divided by 2^30, rotate a physical-unit vector correctly); dot product
    -> a score DIRECTLY comparable to the float reference's own pre-softmax score, no q_ln2
    recovery needed. This bypasses the WSC1 fold/funnel machinery entirely: there is no way to
    combine a physical-unit weight with the int8-calibrated fold constants (mult/shift are
    calibrated assuming an int8-magnitude accumulator; a true float32 weight's own magnitude is
    orders of magnitude different, so reusing those constants would not be a physically meaningful
    substitution -- StandardsDocument.md 5.4's "both sides the same quantity" requirement is what
    rules this out).

  CANDIDATE 2 -- RMSNORM-EXACT. RmsNorm's own final requantization round (wide[] -> int8 code) is
    removed, replaced by wide[]'s own EXACT (unrounded) code-space target -- the SAME formula
    T-1787 already used for Q's/K's own funnel rounding, applied one stage upstream. The engine's
    int8 Q/K weight STAYS at production. RmsNorm's own funnel scale (normed_m, normed_e) is a
    function of wide[]'s magnitude BEFORE the round (RequantChainChecked derives r/s from wide[],
    THEN rounds wide[] into codes -- the round never feeds back into r/s), so normed_scale is
    UNCHANGED between production and this candidate and needs no re-derivation. What DOES change
    is Q's own downstream funnel: a different (continuous) activation drives a different raw
    accumulator, hence a different q_wide magnitude, hence a different q_scale -- this candidate
    re-derives Q's own r/s/out_scale fresh from the modified accumulator (self-checked below
    against production's own r/s on the UNPERTURBED accumulator before being trusted on the
    perturbed one) and re-derives q_ln2/q_b/q_c (IExpScaleConstants) from the new q_scale to
    recover real logits. K's own landing-rescale needs no re-derivation: its own scale is the
    FIXED per-(layer,kv_head) (r_t, e_t) constant, unrelated to any accumulator's magnitude.

  CANDIDATE 3 -- JOINT. Both substitutions together: the checkpoint's true weight applied to the
    RMSNorm-exact continuous activation (physical units, same shape as candidate 1).

SELF-CHECK DISCIPLINE (StandardsDocument.md 5.4): every ported function (FloorDivI64/ISqrt via
reproducing normed[] bit-exact from h[]/gain; MaxAbsReduceWide/NormalizeScale/
DynamicScaleReciprocal via reproducing production's own dumped Q r/s; the exact-fold formula via
reproducing production's own dumped folded_prebias bit-exact; CombineCarriedScale/
IExpScaleConstants via reproducing production's own dumped q_ln2/q_b/q_c bit-exact) is checked
against a PRODUCTION (unperturbed) value the C++ probe captured via a real call to the compiled
function, before that port is trusted on any perturbed (candidate) input. A failure in any
self-check hard-stops this script before any budget number is printed.
"""

from __future__ import annotations

import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKTREE = Path(r"D:\SuperSLM\.worktrees\t1788-weight-activation-quant")
OUTDIR = WORKTREE / "out" / "t1788"

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

_KIEXP_PINNED_LN2 = float.fromhex("0x1.62e42fefa39efp-1")
_KIEXP_POLY_A = 0.3585
_KIEXP_POLY_B = 1.353
_KIEXP_POLY_C = 0.344
K_IEXP_LN2_Q = int(_KIEXP_PINNED_LN2 * float(1 << 30))
K_IEXP_B_Q = int(_KIEXP_POLY_B * float(1 << 30))
K_IEXP_CA_Q = int((_KIEXP_POLY_C / _KIEXP_POLY_A) * float(1 << 30))

INT32_MAX = 2**31 - 1


# ============================================================================================
# Ported primitives
# ============================================================================================

def floor_div_i64(a: int, b: int) -> int:
    return a // b


def isqrt_floor(n: int) -> int:
    return math.isqrt(n)


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


def max_abs_reduce_wide(row) -> int:
    d = max((abs(x) for x in row), default=0)
    return d if d >= 1 else 1


def round_half_away_float_to_int(v: float) -> int:
    """This codebase's own C3 tie rule (ties away from zero) applied to round a CONTINUOUS value
    into the int64 domain NormalizeScale/DynamicScaleReciprocal require. Used at exactly ONE
    boundary: where the RMSNorm-exact candidate's own continuous accumulator crosses into Q's
    funnel-scale derivation, which is an inherently integer bit-scan operation. This is the SAME
    treatment T-1787 already used at every continuous-to-discrete crossing in this campaign
    (RoPE's own production-style round applied to the Q-exact/K-exact continuous pre-rotation
    values, `rope_pair_prod_mixed`'s float branch) -- not a new convention introduced here."""
    if v >= 0:
        return math.floor(v + 0.5)
    return math.ceil(v - 0.5)


def round_half_away_int(numerator: int, exponent: int) -> int:
    if numerator == 0:
        return 0
    neg = numerator < 0
    mag = -numerator if neg else numerator
    q = (mag * 2 + (1 << exponent)) // (1 << (exponent + 1))
    return -q if neg else q


def clamp127(v):
    if v > 127:
        return 127
    if v < -127:
        return -127
    return v


def requant_token_code_wide(x_i: int, r: int, s: int) -> int:
    exponent = 62 - s
    return clamp127(round_half_away_int(x_i * 127 * r, exponent))


def rounding_divide_by_pot_i32(x: int, exponent: int) -> int:
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    ux = x & 0xFFFFFFFF
    remainder = ux & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    shifted = x >> exponent
    return shifted + (1 if remainder > threshold else 0)


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    ab = a * b
    result = (ab + (1 << 30)) >> 31
    return INT32_MAX if result > INT32_MAX else result


def apply_weight_scale_fold_int(acc: int, identity: int, mult: int, shift: int) -> int:
    if identity != 0:
        return acc
    acc32 = ((acc + 2**31) % 2**32) - 2**31
    hm = saturating_rounding_doubling_high_mul(acc32, mult)
    return rounding_divide_by_pot_i32(hm, shift)


def apply_weight_scale_fold_exact(acc_real: float, identity: int, mult: int, shift: int) -> float:
    if identity != 0:
        return acc_real
    return acc_real * mult / (2.0 ** (31 + shift))


def combine_carried_scale(am: int, ae: int, bm: int, be: int):
    ma = ((am + 2**31) % 2**32) - 2**31
    mb = ((bm + 2**31) % 2**32) - 2**31
    e = ae + be + 31
    m = saturating_rounding_doubling_high_mul(ma, mb)
    if m < (1 << 30):
        m <<= 1
        e -= 1
    return m, e


def iexp_scale_constants(m: int, e: int):
    shift_ln2 = 30 + 62 + e
    shift_b = 30 + 62 + e
    shift_c = 30 + 124 + 2 * e
    r_m = dynamic_scale_reciprocal(m)
    num_ln2 = K_IEXP_LN2_Q * r_m
    num_b = K_IEXP_B_Q * r_m
    num_c = K_IEXP_CA_Q * r_m * r_m
    q_ln2 = num_ln2 >> shift_ln2 if shift_ln2 >= 0 else num_ln2 << (-shift_ln2)
    q_b = num_b >> shift_b if shift_b >= 0 else num_b << (-shift_b)
    q_c = num_c >> shift_c if shift_c >= 0 else num_c << (-shift_c)
    return q_ln2, q_b, q_c


def rope_pair_exact(x, y, cos_q30: int, sin_q30: int):
    xr = x * cos_q30 - y * sin_q30
    yr = x * sin_q30 + y * cos_q30
    return xr / (2.0 ** ROPE_FRAC_BITS), yr / (2.0 ** ROPE_FRAC_BITS)


def apply_rope_vector_exact(pre, position_cos_sin):
    out = [0.0] * HEAD_DIM
    for i in range(PAIRS):
        cos_q30, sin_q30 = position_cos_sin[i]
        x, y = pre[2 * i], pre[2 * i + 1]
        rx, ry = rope_pair_exact(x, y, cos_q30, sin_q30)
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


def wide_from_h_gain(h, gain):
    """RmsNormSite's own construction (forward_sites.cpp:222-258): sumsq -> ISqrt(FloorDivI64) ->
    max(root,1) -> per-element FloorDivI64(h<<32,root)*gain."""
    sumsq = sum(hi * hi for hi in h)
    root = isqrt_floor(floor_div_i64(sumsq << (2 * NORM_FRAC_BITS), HIDDEN_SIZE))
    root = root if root > 1 else 1
    return [floor_div_i64(h[i] << (2 * NORM_FRAC_BITS), root) * gain[i] for i in range(HIDDEN_SIZE)], root


# ============================================================================================
# Dump readers
# ============================================================================================

def read_normed_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, hidden_size = int(header[0]), int(header[1])
        assert hidden_size == HIDDEN_SIZE
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, position = int(parts[0]), int(parts[1])
            normed_m, normed_e, r, s = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            idx = 6
            h = [int(x) for x in parts[idx: idx + hidden_size]]
            idx += hidden_size
            normed = [int(x) for x in parts[idx: idx + hidden_size]]
            rows[(layer, position)] = {
                "h": h, "normed": normed, "normed_m": normed_m, "normed_e": normed_e, "r": r, "s": s,
            }
    return rows


def read_qacc_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, hidden_size = int(header[0]), int(header[1])
        assert hidden_size == HIDDEN_SIZE
        for _ in range(num_rows):
            parts = f.readline().split()
            layer = int(parts[0])
            q_r, q_s, site_m, site_e = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            idx = 5
            raw = [int(x) for x in parts[idx: idx + hidden_size]]
            idx += hidden_size
            folded = [int(x) for x in parts[idx: idx + hidden_size]]
            idx += hidden_size
            wide = [int(x) for x in parts[idx: idx + hidden_size]]
            rows[layer] = {
                "q_r": q_r, "q_s": q_s, "site_m": site_m, "site_e": site_e,
                "raw": raw, "folded_prebias": folded, "wide": wide,
            }
    return rows


def read_kacc_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, head_dim = int(header[0]), int(header[1])
        assert head_dim == HEAD_DIM
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, position, kv_head = int(parts[0]), int(parts[1]), int(parts[2])
            normed_m, normed_e, r_t, e_t = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
            idx = 7
            raw = [int(x) for x in parts[idx: idx + head_dim]]
            idx += head_dim
            folded = [int(x) for x in parts[idx: idx + head_dim]]
            idx += head_dim
            kacc = [int(x) for x in parts[idx: idx + head_dim]]
            rows[(layer, position, kv_head)] = {
                "normed_m": normed_m, "normed_e": normed_e, "r_t": r_t, "e_t": e_t,
                "raw": raw, "folded_prebias": folded, "kacc": kacc,
            }
    return rows


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
            scores = [int(x) for x in parts[idx: idx + w]]
            idx += w
            probs = [int(x) for x in parts[idx: idx + w]]
            rows[(layer, head)] = {
                "width": w, "q_ln2": q_ln2, "q_b": q_b, "q_c": q_c, "scores": scores, "probs": probs,
            }
    return rows, width


def read_rope_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        width, pairs = int(header[0]), int(header[1])
        for _ in range(width):
            parts = f.readline().split()
            pos = int(parts[0])
            vals = [int(x) for x in parts[1:]]
            rows[pos] = [(vals[2 * i], vals[2 * i + 1]) for i in range(pairs)]
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
            scores = [float(x) for x in parts[idx: idx + w]]
            idx += w
            probs = [float(x) for x in parts[idx: idx + w]]
            rows[(layer, head)] = {"scores": scores, "probs": probs}
    return rows, width


def read_layerconst_dump(path: Path):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        n_layers, hidden_size, kv_hidden_size, num_kv_heads = (int(x) for x in header)
        assert hidden_size == HIDDEN_SIZE and kv_hidden_size == KV_HIDDEN_SIZE
        for _ in range(n_layers):
            parts = f.readline().split()
            idx = 0
            layer = int(parts[idx]); idx += 1
            q_id = [int(x) for x in parts[idx: idx + hidden_size]]; idx += hidden_size
            q_mult = [int(x) for x in parts[idx: idx + hidden_size]]; idx += hidden_size
            q_shift = [int(x) for x in parts[idx: idx + hidden_size]]; idx += hidden_size
            k_id = [int(x) for x in parts[idx: idx + kv_hidden_size]]; idx += kv_hidden_size
            k_mult = [int(x) for x in parts[idx: idx + kv_hidden_size]]; idx += kv_hidden_size
            k_shift = [int(x) for x in parts[idx: idx + kv_hidden_size]]; idx += kv_hidden_size
            khead_m = [int(x) for x in parts[idx: idx + num_kv_heads]]; idx += num_kv_heads
            khead_e = [int(x) for x in parts[idx: idx + num_kv_heads]]; idx += num_kv_heads
            has_qb = int(parts[idx]); idx += 1
            q_bias = None
            if has_qb:
                q_bias = [int(x) for x in parts[idx: idx + hidden_size]]; idx += hidden_size
            has_kb = int(parts[idx]); idx += 1
            k_bias = None
            if has_kb:
                k_bias = [int(x) for x in parts[idx: idx + kv_hidden_size]]; idx += kv_hidden_size
            gain = [int(x) for x in parts[idx: idx + hidden_size]]; idx += hidden_size
            rows[layer] = {
                "q_fold_identity": q_id, "q_fold_mult": q_mult, "q_fold_shift": q_shift,
                "k_fold_identity": k_id, "k_fold_mult": k_mult, "k_fold_shift": k_shift,
                "khead_m": khead_m, "khead_e": khead_e, "q_bias": q_bias, "k_bias": k_bias, "gain": gain,
            }
    return rows


def read_int8_weights(path: Path):
    rows = {}
    data = path.read_bytes()
    off = 0
    for layer in CHECKPOINT_LAYERS:
        qn = HIDDEN_SIZE * HIDDEN_SIZE
        kn = KV_HIDDEN_SIZE * HIDDEN_SIZE
        qw = list(struct.unpack_from(f"<{qn}b", data, off)); off += qn
        kw = list(struct.unpack_from(f"<{kn}b", data, off)); off += kn
        rows[layer] = {"q_weight": qw, "k_weight": kw}
    assert off == len(data), f"weights.bin size mismatch: read {off}, file has {len(data)}"
    return rows


def read_true_weights(path: Path):
    rows = {}
    data = path.read_bytes()
    off = 0
    for layer in CHECKPOINT_LAYERS:
        qn = HIDDEN_SIZE * HIDDEN_SIZE
        qw = list(struct.unpack_from(f"<{qn}d", data, off)); off += qn * 8
        qb = list(struct.unpack_from(f"<{HIDDEN_SIZE}d", data, off)); off += HIDDEN_SIZE * 8
        kn = KV_HIDDEN_SIZE * HIDDEN_SIZE
        kw = list(struct.unpack_from(f"<{kn}d", data, off)); off += kn * 8
        kb = list(struct.unpack_from(f"<{KV_HIDDEN_SIZE}d", data, off)); off += KV_HIDDEN_SIZE * 8
        rows[layer] = {"q_weight": qw, "q_bias": qb, "k_weight": kw, "k_bias": kb}
    assert off == len(data), f"true_weights.bin size mismatch: read {off}, file has {len(data)}"
    return rows


def gemm_row(activation, weight_flat, in_channels, out_channels, out_offset=0):
    out = [0.0] * out_channels
    for j in range(out_channels):
        base = (out_offset + j) * in_channels
        row = weight_flat[base: base + in_channels]
        out[j] = sum(activation[i] * row[i] for i in range(in_channels))
    return out


# ============================================================================================
# Vectorized per-prompt/per-layer computation
# ============================================================================================

class LayerWeightsNp:
    """Numpy views of the layer's own weight/fold data, built once and shared across prompts."""

    def __init__(self, layer, layerconst, int8_weights, true_weights):
        lc = layerconst[layer]
        iw = int8_weights[layer]
        tw = true_weights[layer]
        self.iw_q = np.array(iw["q_weight"], dtype=np.float64).reshape(HIDDEN_SIZE, HIDDEN_SIZE)
        self.iw_k = np.array(iw["k_weight"], dtype=np.float64).reshape(KV_HIDDEN_SIZE, HIDDEN_SIZE)
        self.tw_q = np.array(tw["q_weight"], dtype=np.float64).reshape(HIDDEN_SIZE, HIDDEN_SIZE)
        self.tw_qb = np.array(tw["q_bias"], dtype=np.float64)
        self.tw_k = np.array(tw["k_weight"], dtype=np.float64).reshape(KV_HIDDEN_SIZE, HIDDEN_SIZE)
        self.tw_kb = np.array(tw["k_bias"], dtype=np.float64)
        self.q_fold_identity = np.array(lc["q_fold_identity"], dtype=np.int64)
        self.q_fold_mult = np.array(lc["q_fold_mult"], dtype=np.float64)
        self.q_fold_shift = np.array(lc["q_fold_shift"], dtype=np.float64)
        self.k_fold_identity = np.array(lc["k_fold_identity"], dtype=np.int64)
        self.k_fold_mult = np.array(lc["k_fold_mult"], dtype=np.float64)
        self.k_fold_shift = np.array(lc["k_fold_shift"], dtype=np.float64)
        self.khead_m = lc["khead_m"]
        self.khead_e = lc["khead_e"]


def exact_fold_vec(raw, identity, mult, shift):
    """Vectorized apply_weight_scale_fold_exact: identity!=0 -> raw unchanged; else raw*mult/2^(31+shift)."""
    folded_branch = raw * mult / np.exp2(31.0 + shift)
    return np.where(identity != 0, raw, folded_branch)


def rope_rotate_matrix(mat, cos_mat, sin_mat):
    """mat: (n, HEAD_DIM). cos_mat/sin_mat: (n, PAIRS). Exact (no round, no clamp) rotation --
    unit-preserving, correct for code-units or physical-real values alike (this ticket's header)."""
    x = mat[:, 0::2]
    y = mat[:, 1::2]
    xr = x * cos_mat - y * sin_mat
    yr = x * sin_mat + y * cos_mat
    out = np.empty_like(mat)
    out[:, 0::2] = xr / (2.0 ** ROPE_FRAC_BITS)
    out[:, 1::2] = yr / (2.0 ** ROPE_FRAC_BITS)
    return out


class PromptData:
    def __init__(self, pi, layerconst, int8_weights, true_weights, layer_np_cache):
        self.pi = pi
        self.normed = read_normed_dump(OUTDIR / f"p{pi}_normed.txt")
        self.qacc = read_qacc_dump(OUTDIR / f"p{pi}_qacc.txt")
        self.kacc = read_kacc_dump(OUTDIR / f"p{pi}_kacc.txt")
        self.eng_rows, self.eng_w = read_scores_dump(OUTDIR / f"p{pi}_scores.txt")
        self.rope, self.rope_w = read_rope_dump(OUTDIR / f"p{pi}_rope.txt")
        self.flt_rows, self.flt_w = read_float_dump(OUTDIR / f"p{pi}_float.txt")
        assert self.eng_w == self.flt_w == self.rope_w, (
            f"P{pi}: width mismatch eng={self.eng_w} flt={self.flt_w} rope={self.rope_w}")
        self.width = self.eng_w
        self.last_pos = self.width - 1
        self.layerconst = layerconst
        self.int8_weights = int8_weights
        self.true_weights = true_weights
        self.layer_np = layer_np_cache

        self._wide_cache = {}
        self._normed_exact_cache = {}

        # RoPE cos/sin as (width, PAIRS) matrices, shared across every layer/candidate.
        self.cos_mat = np.array([[c for c, s in self.rope[k]] for k in range(self.width)], dtype=np.float64)
        self.sin_mat = np.array([[s for c, s in self.rope[k]] for k in range(self.width)], dtype=np.float64)

    def wide_at(self, layer, position):
        key = (layer, position)
        if key not in self._wide_cache:
            nc = self.normed[key]
            gain = self.layerconst[layer]["gain"]
            wide, root = wide_from_h_gain(nc["h"], gain)
            self._wide_cache[key] = wide
        return self._wide_cache[key]

    def normed_exact_at(self, layer, position):
        key = (layer, position)
        if key not in self._normed_exact_cache:
            wide = self.wide_at(layer, position)
            nc = self.normed[key]
            r, s = nc["r"], nc["s"]
            exponent = 62 - s
            self._normed_exact_cache[key] = [(w * 127 * r) / (2.0 ** exponent) for w in wide]
        return self._normed_exact_cache[key]

    def layer_matrices(self, layer):
        """Per-layer (width, HIDDEN_SIZE) activation matrices, computed once and reused across
        every head/kv_head in this layer: normed_real (production dequantized, physical),
        normed_exact_real (RMSNorm-exact dequantized with the SAME unchanged normed_scale,
        physical), normed_exact (RMSNorm-exact, code units, for the int8-weight-path candidate)."""
        normed_real = np.empty((self.width, HIDDEN_SIZE))
        normed_exact_real = np.empty((self.width, HIDDEN_SIZE))
        normed_exact_codeunits = np.empty((self.width, HIDDEN_SIZE))
        normed_m_arr = np.empty(self.width)
        normed_e_arr = np.empty(self.width)
        for k in range(self.width):
            nc = self.normed[(layer, k)]
            scale = nc["normed_m"] * (2.0 ** nc["normed_e"])
            normed_real[k, :] = np.array(nc["normed"], dtype=np.float64) * scale
            exact = self.normed_exact_at(layer, k)
            normed_exact_codeunits[k, :] = exact
            normed_exact_real[k, :] = np.array(exact) * scale
            normed_m_arr[k] = nc["normed_m"]
            normed_e_arr[k] = nc["normed_e"]
        return normed_real, normed_exact_real, normed_exact_codeunits, normed_m_arr, normed_e_arr

    def k_matrices(self, layer):
        """(width, KV_HIDDEN_SIZE) production kacc / folded_prebias, and the derived bias_reconciled."""
        kacc_prod = np.empty((self.width, KV_HIDDEN_SIZE))
        kacc_folded_prebias = np.empty((self.width, KV_HIDDEN_SIZE))
        r_t = np.empty(KV_HIDDEN_SIZE)
        e_t = np.empty(KV_HIDDEN_SIZE)
        for k in range(self.width):
            for kv_head in range(NUM_KV_HEADS):
                kc = self.kacc[(layer, k, kv_head)]
                off = kv_head * HEAD_DIM
                kacc_prod[k, off:off + HEAD_DIM] = kc["kacc"]
                kacc_folded_prebias[k, off:off + HEAD_DIM] = kc["folded_prebias"]
                if k == 0:
                    r_t[off:off + HEAD_DIM] = kc["r_t"]
                    e_t[off:off + HEAD_DIM] = kc["e_t"]
        bias_reconciled = kacc_prod - kacc_folded_prebias
        return kacc_prod, bias_reconciled, r_t, e_t


def main():
    prompts = (1, 2, 3)
    layerconst = read_layerconst_dump(OUTDIR / "p1_layerconst.txt")
    int8_weights = read_int8_weights(OUTDIR / "weights.bin")
    true_weights = read_true_weights(OUTDIR / "true_weights.bin")
    layer_np_cache = {layer: LayerWeightsNp(layer, layerconst, int8_weights, true_weights)
                       for layer in CHECKPOINT_LAYERS}

    self_check_fail = 0
    self_check_counts = defaultdict(int)
    all_rows = []

    for pi in prompts:
        pd = PromptData(pi, layerconst, int8_weights, true_weights, layer_np_cache)

        # --- Self-check A: normed[] reproduced bit-exact from h[]/gain. ---
        for (layer, position), nc in pd.normed.items():
            wide = pd.wide_at(layer, position)
            r, s = nc["r"], nc["s"]
            for i in range(HIDDEN_SIZE):
                got = requant_token_code_wide(wide[i], r, s)
                self_check_counts["normed_code"] += 1
                if got != nc["normed"][i]:
                    self_check_fail += 1
                    print(f"SELF-CHECK A (normed) FAIL P{pi} layer={layer} pos={position} i={i}: "
                          f"got={got} dumped={nc['normed'][i]}", file=sys.stderr)

        # --- Self-check B: ApplyWeightScaleFold, folded_prebias[] reproduced from raw[]. ---
        for layer in CHECKPOINT_LAYERS:
            lc = layerconst[layer]
            qc = pd.qacc[layer]
            for i in range(HIDDEN_SIZE):
                got = apply_weight_scale_fold_int(qc["raw"][i], lc["q_fold_identity"][i],
                                                   lc["q_fold_mult"][i], lc["q_fold_shift"][i])
                self_check_counts["fold_q"] += 1
                if got != qc["folded_prebias"][i]:
                    self_check_fail += 1
                    print(f"SELF-CHECK B (Q fold) FAIL P{pi} layer={layer} i={i}: "
                          f"got={got} dumped={qc['folded_prebias'][i]}", file=sys.stderr)
        for (layer, position, kv_head), kc in pd.kacc.items():
            lc = layerconst[layer]
            off = kv_head * HEAD_DIM
            for d in range(HEAD_DIM):
                got = apply_weight_scale_fold_int(kc["raw"][d], lc["k_fold_identity"][off + d],
                                                   lc["k_fold_mult"][off + d], lc["k_fold_shift"][off + d])
                self_check_counts["fold_k"] += 1
                if got != kc["folded_prebias"][d]:
                    self_check_fail += 1
                    print(f"SELF-CHECK B (K fold) FAIL P{pi} layer={layer} pos={position} "
                          f"kv_head={kv_head} d={d}: got={got} dumped={kc['folded_prebias'][d]}",
                          file=sys.stderr)

        # --- Self-check C: NormalizeScale/DynamicScaleReciprocal on Q's own wide[] -> (q_r,q_s). ---
        q_dn_s = {}
        for layer in CHECKPOINT_LAYERS:
            qc = pd.qacc[layer]
            d_prime = max_abs_reduce_wide(qc["wide"])
            dn, s = normalize_scale(d_prime)
            r = dynamic_scale_reciprocal(dn)
            q_dn_s[layer] = (dn, s)
            self_check_counts["q_rs"] += 1
            if r != qc["q_r"] or s != qc["q_s"]:
                self_check_fail += 1
                print(f"SELF-CHECK C (Q r/s) FAIL P{pi} layer={layer}: got r={r} s={s} "
                      f"dumped r={qc['q_r']} s={qc['q_s']}", file=sys.stderr)

        # --- Self-check D: CombineCarriedScale + IExpScaleConstants -> (q_ln2,q_b,q_c). ---
        # Q's own RequantChainChecked call is NOT RmsNorm's empty-incoming shape -- it threads
        # normed_scale through as `incoming` (ProjectAndFunnelCopyCapture's own
        # `const CarriedScale incoming[1] = {in_scale}` call, in_scale == normed_scale at the last
        # position), so q_scale folds THREE factors, left-associated per RequantChainChecked's own
        # C26 order: incoming (normed_scale) first, then site_constant, then the d_prime factor.
        q_scale_prod = {}
        for layer in CHECKPOINT_LAYERS:
            qc = pd.qacc[layer]
            dn, s = q_dn_s[layer]
            nc_last = pd.normed[(layer, pd.last_pos)]
            qs0_m, qs0_e = combine_carried_scale(nc_last["normed_m"], nc_last["normed_e"],
                                                  qc["site_m"], qc["site_e"])
            qs_m, qs_e = combine_carried_scale(qs0_m, qs0_e, dn, -s)
            q_scale_prod[layer] = (qs_m, qs_e)
            lc = layerconst[layer]
            for kv_head in range(NUM_KV_HEADS):
                sm_m, sm_e = combine_carried_scale(qs_m, qs_e, lc["khead_m"][kv_head], lc["khead_e"][kv_head])
                my_ln2, my_b, my_c = iexp_scale_constants(sm_m, sm_e)
                head0 = kv_head * GROUP
                row = pd.eng_rows[(layer, head0)]
                self_check_counts["q_ln2"] += 1
                if (my_ln2, my_b, my_c) != (row["q_ln2"], row["q_b"], row["q_c"]):
                    self_check_fail += 1
                    print(f"SELF-CHECK D (q_ln2) FAIL P{pi} layer={layer} kv_head={kv_head}: "
                          f"got=({my_ln2},{my_b},{my_c}) "
                          f"dumped=({row['q_ln2']},{row['q_b']},{row['q_c']})", file=sys.stderr)

        if self_check_fail:
            print(f"\nself_check_fail={self_check_fail} -- STOPPING before any budget number is computed",
                  file=sys.stderr)
            sys.exit(1)

        # === Vectorized per-layer computation ===
        for layer in CHECKPOINT_LAYERS:
            lnp = layer_np_cache[layer]
            qc = pd.qacc[layer]
            lc = layerconst[layer]

            normed_real, normed_exact_real, normed_exact_cu, normed_m_arr, normed_e_arr = \
                pd.layer_matrices(layer)
            kacc_prod, k_bias_reconciled, r_t_arr, e_t_arr = pd.k_matrices(layer)

            # ---- K, candidate 0 (baseline): kacc_prod already IS the exact code-space value's
            # own pre-round input at production's own rounding -- use the SAME k_exact_real
            # formula T-1787 established (per-position m_a/e_a, per-channel fixed r_t/e_t). ----
            exponent_k = 62.0 - (normed_e_arr[:, None] - e_t_arr[None, :])
            k_exact_base = kacc_prod * normed_m_arr[:, None] * r_t_arr[None, :] / np.exp2(exponent_k)
            k_rot_base = np.concatenate(
                [rope_rotate_matrix(k_exact_base[:, h * HEAD_DIM:(h + 1) * HEAD_DIM], pd.cos_mat, pd.sin_mat)
                 for h in range(NUM_KV_HEADS)], axis=1)

            # ---- K, candidate 1 (weight-exact): physical units, true weight. ----
            k_phys_we = normed_real @ lnp.tw_k.T + lnp.tw_kb[None, :]
            k_rot_we = np.concatenate(
                [rope_rotate_matrix(k_phys_we[:, h * HEAD_DIM:(h + 1) * HEAD_DIM], pd.cos_mat, pd.sin_mat)
                 for h in range(NUM_KV_HEADS)], axis=1)

            # ---- K, candidate 3 (joint): physical units, true weight, RMSNorm-exact activation. ----
            k_phys_joint = normed_exact_real @ lnp.tw_k.T + lnp.tw_kb[None, :]
            k_rot_joint = np.concatenate(
                [rope_rotate_matrix(k_phys_joint[:, h * HEAD_DIM:(h + 1) * HEAD_DIM], pd.cos_mat, pd.sin_mat)
                 for h in range(NUM_KV_HEADS)], axis=1)

            # ---- K, candidate 2 (RMSNorm-exact, int8 weight, code units, K needs no re-derivation). ----
            k_raw_re = normed_exact_cu @ lnp.iw_k.T
            k_folded_re = exact_fold_vec(k_raw_re, lnp.k_fold_identity[None, :], lnp.k_fold_mult[None, :],
                                          lnp.k_fold_shift[None, :])
            k_wide_re = k_folded_re + k_bias_reconciled
            k_exact_re = k_wide_re * normed_m_arr[:, None] * r_t_arr[None, :] / np.exp2(exponent_k)
            k_rot_re = np.concatenate(
                [rope_rotate_matrix(k_exact_re[:, h * HEAD_DIM:(h + 1) * HEAD_DIM], pd.cos_mat, pd.sin_mat)
                 for h in range(NUM_KV_HEADS)], axis=1)

            # ---- Q, candidate 2 (RMSNorm-exact): full 1536-dim, ONE derivation per layer. ----
            normed_exact_last = normed_exact_cu[pd.last_pos, :]
            q_raw_re = lnp.iw_q @ normed_exact_last
            q_folded_re = exact_fold_vec(q_raw_re, lnp.q_fold_identity, lnp.q_fold_mult, lnp.q_fold_shift)
            q_bias_reconciled = np.array(qc["wide"]) - np.array(qc["folded_prebias"])
            q_wide_re = q_folded_re + q_bias_reconciled
            q_wide_re_int = [round_half_away_float_to_int(v) for v in q_wide_re.tolist()]
            d_prime_new = max_abs_reduce_wide(q_wide_re_int)
            dn_new, s_new = normalize_scale(d_prime_new)
            r_new = dynamic_scale_reciprocal(dn_new)
            q_exact_re_full = (q_wide_re * 127.0 * r_new) / (2.0 ** (62 - s_new))
            # Q's own funnel folds in `incoming` (normed_scale) before site_constant and the
            # d_prime factor (RequantChainChecked's own C26 order) -- normed_scale ITSELF is
            # unchanged for this candidate (established in this ticket's own header: RmsNorm's own
            # r/s, hence normed_scale, are derived from wide[] BEFORE the round, so the round never
            # feeds back into them), so the SAME production nc_last is folded in here too.
            nc_last = pd.normed[(layer, pd.last_pos)]
            qs0_m_new, qs0_e_new = combine_carried_scale(nc_last["normed_m"], nc_last["normed_e"],
                                                          qc["site_m"], qc["site_e"])
            qs_m_new, qs_e_new = combine_carried_scale(qs0_m_new, qs0_e_new, dn_new, -s_new)

            # ---- Q, candidates 0/1/3 (full 1536-dim, once per layer). ----
            q_wide_base = np.array(qc["wide"])
            q_r, q_s = qc["q_r"], qc["q_s"]
            q_exact_base_full = (q_wide_base * 127.0 * q_r) / (2.0 ** (62 - q_s))

            normed_real_last = normed_real[pd.last_pos, :]
            q_phys_we_full = lnp.tw_q @ normed_real_last + lnp.tw_qb
            normed_exact_real_last = normed_exact_real[pd.last_pos, :]
            q_phys_joint_full = lnp.tw_q @ normed_exact_real_last + lnp.tw_qb

            for head in range(NUM_HEADS):
                kv_head = head // GROUP
                sl = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
                k_sl = slice(kv_head * HEAD_DIM, (kv_head + 1) * HEAD_DIM)
                last_cs = pd.cos_mat[pd.last_pos:pd.last_pos + 1, :]
                last_sn = pd.sin_mat[pd.last_pos:pd.last_pos + 1, :]

                q_rot_base = rope_rotate_matrix(q_exact_base_full[sl][None, :], last_cs, last_sn)[0]
                q_rot_we = rope_rotate_matrix(q_phys_we_full[sl][None, :], last_cs, last_sn)[0]
                q_rot_joint = rope_rotate_matrix(q_phys_joint_full[sl][None, :], last_cs, last_sn)[0]
                q_rot_re = rope_rotate_matrix(q_exact_re_full[sl][None, :], last_cs, last_sn)[0]

                score_base = k_rot_base[:, k_sl] @ q_rot_base
                # Physical-unit scores need the SAME attention-scaling factor the float reference
                # applies after its own matmul (t1788_float_prescore_reference.py's own
                # `oracle_recompute_attn`: "scaling = head_dim**-0.5; scores = matmul(...) *
                # scaling") -- the code-units candidates (baseline, RMSNorm-exact) never need this
                # explicitly because it is already folded into q_ln2's own derivation (T-1787
                # D-SLM1292: production's raw recovered logit best-fit scale against the float
                # reference is ~1, not 1/sqrt(head_dim) or sqrt(head_dim) -- i.e. q_ln2's recovery
                # already accounts for it). A genuinely physical-unit score, computed from a raw
                # unscaled dot product, has NOT had this factor applied and needs it explicitly.
                attn_scaling = HEAD_DIM ** -0.5
                score_we = (k_rot_we[:, k_sl] @ q_rot_we) * attn_scaling
                score_joint = (k_rot_joint[:, k_sl] @ q_rot_joint) * attn_scaling

                sm_m, sm_e = combine_carried_scale(qs_m_new, qs_e_new, lc["khead_m"][kv_head],
                                                    lc["khead_e"][kv_head])
                q_ln2_new, _, _ = iexp_scale_constants(sm_m, sm_e)
                score_re = k_rot_re[:, k_sl] @ q_rot_re

                e = pd.eng_rows[(layer, head)]
                q_ln2 = e["q_ln2"]
                p_float = pd.flt_rows[(layer, head)]["probs"]

                def to_probs_via_qln2(scores_row, ln2):
                    shifted = shift_by_max(list(scores_row))
                    logits = [x * math.log(2.0) / ln2 for x in shifted]
                    return softmax_from_real_logits(logits)

                p_baseline = to_probs_via_qln2(score_base, q_ln2)
                p_we = softmax_from_real_logits(shift_by_max(list(score_we)))
                p_joint = softmax_from_real_logits(shift_by_max(list(score_joint)))
                p_re = to_probs_via_qln2(score_re, q_ln2_new)

                all_rows.append({
                    "prompt": pi, "layer": layer, "head": head,
                    "baseline": tvd(p_baseline, p_float),
                    "weight_exact": tvd(p_we, p_float),
                    "rmsnorm_exact": tvd(p_re, p_float),
                    "joint": tvd(p_joint, p_float),
                })

        print(f"P{pi}: self-checks passed ({sum(self_check_counts.values())} checks so far), "
              f"{sum(1 for r in all_rows if r['prompt'] == pi)} rows computed", file=sys.stderr)

    print(f"\nTOTAL SELF-CHECKS: {dict(self_check_counts)}")
    print(f"TOTAL SELF-CHECK FAILURES: {self_check_fail}\n")

    # --- Per-layer budget table ---
    by_layer = defaultdict(list)
    for r in all_rows:
        by_layer[r["layer"]].append(r)

    print(f"{'layer':>5} {'n':>4} {'base':>8} {'W-exact':>9} {'R-exact':>9} {'joint':>9} "
          f"{'W contrib':>10} {'R contrib':>10} {'sum':>10} {'joint delta':>12}")
    for layer in sorted(by_layer):
        rs = by_layer[layer]
        n_l = len(rs)
        base = sum(x["baseline"] for x in rs) / n_l
        we = sum(x["weight_exact"] for x in rs) / n_l
        re_ = sum(x["rmsnorm_exact"] for x in rs) / n_l
        jt = sum(x["joint"] for x in rs) / n_l
        cw, cr = base - we, base - re_
        csum = cw + cr
        jdelta = base - jt
        print(f"{layer:>5} {n_l:>4} {base:>8.4f} {we:>9.4f} {re_:>9.4f} {jt:>9.4f} "
              f"{cw:>10.4f} {cr:>10.4f} {csum:>10.4f} {jdelta:>12.4f}")

    n_total = len(all_rows)
    base = sum(x["baseline"] for x in all_rows) / n_total
    we = sum(x["weight_exact"] for x in all_rows) / n_total
    re_ = sum(x["rmsnorm_exact"] for x in all_rows) / n_total
    jt = sum(x["joint"] for x in all_rows) / n_total
    cw, cr = base - we, base - re_
    csum = cw + cr
    jdelta = base - jt
    print(f"\nGrand pooled (n={n_total}): baseline={base:.4f} weight_exact={we:.4f} "
          f"rmsnorm_exact={re_:.4f} joint={jt:.4f}")
    print(f"contribution WEIGHT={cw:.4f} ({100*cw/base:.1f}% of baseline)  "
          f"RMSNORM={cr:.4f} ({100*cr/base:.1f}%)")
    print(f"sum of individual contributions={csum:.4f} ({100*csum/base:.1f}% of baseline)")
    print(f"joint delta (both together)={jdelta:.4f} ({100*jdelta/base:.1f}% of baseline) "
          f"-- compare to sum above for additivity")
    print(f"residual TVD surviving the joint substitution = {jt:.4f} "
          f"({100*jt/base:.1f}% of THIS ticket's own baseline)")

    # --- Resolving power, n=3 prompts ---
    print("\nResolving power (n=3 prompts, grand-pooled contribution mean vs inter-prompt std):")
    for label, key in (("WEIGHT (true fp32 vs int8)", "weight_exact"),
                        ("RMSNORM (exact vs int8 activation)", "rmsnorm_exact"),
                        ("JOINT delta", "joint")):
        vals = []
        for pi in prompts:
            rs = [x for x in all_rows if x["prompt"] == pi]
            contrib = sum(x["baseline"] - x[key] for x in rs) / len(rs)
            vals.append(contrib)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        ratio = mean / std if std > 0 else float("inf")
        print(f"  {label}: per-prompt={['%.4f' % v for v in vals]} mean={mean:.4f} std={std:.4f} "
              f"mean/std={ratio:.2f} -- {'resolved from zero' if abs(ratio) >= 2 else 'NOT resolved at this n'}")

    print("\nPer-prompt grand-pooled baseline (resolving-power check, n=3):")
    for pi in prompts:
        rs = [x for x in all_rows if x["prompt"] == pi]
        print(f"  P{pi}: n={len(rs)} baseline={sum(x['baseline'] for x in rs)/len(rs):.4f} "
              f"weight_exact={sum(x['weight_exact'] for x in rs)/len(rs):.4f} "
              f"rmsnorm_exact={sum(x['rmsnorm_exact'] for x in rs)/len(rs):.4f} "
              f"joint={sum(x['joint'] for x in rs)/len(rs):.4f}")


if __name__ == "__main__":
    main()
