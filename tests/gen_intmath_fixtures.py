#!/usr/bin/env python3
"""Generates sslm_intmath_fixtures.h from the pinned Python reference (D-SLM52).

Every expected value below is computed by IMPORTING and CALLING
Tools/superslm_spike/intmath.py — never hand-computed, never copied from any
C++ implementation. Re-running this script must reproduce
sslm_intmath_fixtures.h byte-for-byte (Poirot's reproducibility check).

Test-design records:
Claude/Curie/superslm-s2.1-intmath-test-design-2026-07-19.md (C1/C2/C3, C19-C22)
Claude/Curie/superslm-s2.2-nonlinear-test-design-2026-07-19.md (ISqrt/ISqrtTrace,
ShiftByMax, IExpConstruct/IExpEvaluate)
Claude/Curie/superslm-s2.3-rope-test-design-2026-07-19.md (RopeApplyPair)
"""

from __future__ import annotations

import math
import os
import random
import sys
from fractions import Fraction

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPIKE_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "Wizard", ".claude", "worktrees",
                 "superslm-dev-continue-d0b08e", "Tools", "superslm_spike")
)
_TOOLS_DIR = os.path.dirname(_SPIKE_DIR)
sys.path.insert(0, _SPIKE_DIR)
sys.path.insert(0, _TOOLS_DIR)

import intmath as im  # noqa: E402
import superslm_spike.rope as rope  # noqa: E402

INT32_MIN = im.INT32_MIN
INT32_MAX = im.INT32_MAX
OUT_PATH = os.path.join(_THIS_DIR, "sslm_intmath_fixtures.h")

# --------------------------------------------------------------------------
# C2 — SaturatingRoundingDoublingHighMul
# --------------------------------------------------------------------------

c2_cases: list[tuple[str, int, int, int]] = []


def add_c2(label: str, a: int, b: int) -> None:
    r = im.saturating_rounding_doubling_high_mul(a, b)
    c2_cases.append((label, a, b, r))


def find_c2_tie(target_n: int, negative: bool) -> tuple[int, int]:
    """(a, b) with a*b == +/-(target_n * 2**31 + 2**30) — an exact C2 tie.

    Searches factor pairs b = 2**shift so a stays an exact integer; verifies
    the tie condition programmatically rather than asserting it by
    construction alone.
    """
    magnitude = target_n * (1 << 31) + (1 << 30)
    val = -magnitude if negative else magnitude
    for shift in range(0, 32):
        b = 1 << shift
        if val % b == 0:
            a = val // b
            if INT32_MIN <= a <= INT32_MAX and INT32_MIN <= b <= INT32_MAX:
                # Confirm this really is an exact halfway point of a*b / 2**31.
                assert (a * b + (1 << 30)) % (1 << 31) == 0, "not an exact tie"
                return a, b
    raise RuntimeError(f"no int32 factor pair for target_n={target_n} negative={negative}")


add_c2("typical_pos", 1_000_000, 2_000_000)
add_c2("typical_neg", -1_000_000, 2_000_000)
add_c2("typical_neg_neg", -1_000_000, -2_000_000)
add_c2("zero_a", 0, 123_456)
add_c2("zero_b", 123_456, 0)
add_c2("zero_both", 0, 0)
add_c2("large_max_max", INT32_MAX, INT32_MAX)
add_c2("large_min_max", INT32_MIN, INT32_MAX)
add_c2("large_max_min", INT32_MAX, INT32_MIN)
add_c2("saturation_min_min", INT32_MIN, INT32_MIN)  # only pair whose exact result > INT32_MAX

# The negative-branch +infinity tie: away-from-zero would give -(n+1); the
# pinned rule (toward +infinity) gives -n. This is C2's load-bearing
# discriminator between the two tie rules.
_a, _b = find_c2_tie(1, negative=True)
add_c2("neg_tie_toward_plus_inf_n1", _a, _b)
_a, _b = find_c2_tie(3, negative=True)
add_c2("neg_tie_toward_plus_inf_n3", _a, _b)

# Positive-side tie for symmetry documentation (does not discriminate the
# rule — positive ties round the same way under both conventions).
_a, _b = find_c2_tie(1, negative=False)
add_c2("pos_tie_n1", _a, _b)

# --------------------------------------------------------------------------
# C1/C3 — RoundingDivideByPOT
# --------------------------------------------------------------------------

c1c3_cases: list[tuple[str, int, int, int]] = []


def add_c1c3(label: str, x: int, exponent: int) -> None:
    r = im.rounding_divide_by_pot(x, exponent)
    c1c3_cases.append((label, x, exponent, r))


def half_tie_value(exponent: int, negative: bool) -> int:
    """x = +/- 2**(exponent-1) — the exact-half point (x / 2**exponent has
    fractional part exactly 1/2), verified below via `fractions.Fraction`
    independently of `rounding_divide_by_pot`'s own remainder/threshold
    formula, so the construction does not just re-derive the function under
    test's internal boundary."""
    half = 1 << (exponent - 1)
    x = -half if negative else half
    frac = Fraction(x, 1 << exponent)
    floor_v = x >> exponent
    assert frac - floor_v == Fraction(1, 2), "not an exact half point"
    return x


for exp in (0, 1, 2, 5, 10, 16, 20, 30, 31):
    add_c1c3(f"typical_pos_exp{exp}", 123457 if exp < 20 else 999_999_937, exp)
    add_c1c3(f"typical_neg_exp{exp}", -123457 if exp < 20 else -999_999_937, exp)

# Exponent-0 identity, explicitly (already in the sweep above at exp=0, named again).
add_c1c3("identity_exp0_pos", 12345, 0)
add_c1c3("identity_exp0_neg", -12345, 0)
add_c1c3("identity_exp0_zero", 0, 0)

# Exact-half ties on both signs, across exponents (proves away-from-zero,
# not toward +infinity — the opposite convention from C2).
for exp in (1, 2, 5, 16, 31):
    if exp >= 1:
        tv = half_tie_value(exp, negative=False)
        if INT32_MIN <= tv <= INT32_MAX:
            add_c1c3(f"tie_pos_exp{exp}", tv, exp)
        tv = half_tie_value(exp, negative=True)
        if INT32_MIN <= tv <= INT32_MAX:
            add_c1c3(f"tie_neg_exp{exp}", tv, exp)

# The commission's literal worked example.
add_c1c3("commission_example_pos", 3, 1)
add_c1c3("commission_example_neg", -3, 1)

# INT32_MIN dividend across the exponent range, including exponent 31.
for exp in (0, 1, 5, 16, 30, 31):
    add_c1c3(f"int32_min_exp{exp}", INT32_MIN, exp)
add_c1c3("int32_max_exp31", INT32_MAX, 31)

# --------------------------------------------------------------------------
# MultiplyByQuantizedMultiplier — C1 composition
# --------------------------------------------------------------------------

mbqm_cases: list[tuple[str, int, int, int, int]] = []


def add_mbqm(label: str, x: int, m: int, shift: int) -> None:
    r = im.multiply_by_quantized_multiplier(x, m, shift)
    mbqm_cases.append((label, x, m, shift, r))


add_mbqm("half_scale_no_shift", 12345, 1 << 30, 0)  # multiplier == 0.5 in Q31
add_mbqm("half_scale_shift5_neg_x", -12345, 1 << 30, 5)
add_mbqm("near_one_scale_shift10_max_x", INT32_MAX, 2_000_000_000, 10)
add_mbqm("near_one_scale_shift10_min_x", INT32_MIN, 2_000_000_000, 10)
add_mbqm("zero_x", 0, 1_234_567, 3)
add_mbqm("small_multiplier_shift0", 7, 3, 0)
add_mbqm("realistic_requant_a", 500_000, 1_717_986_918, 12)  # ~0.8 in Q31
add_mbqm("realistic_requant_b", -500_000, 1_717_986_918, 12)
add_mbqm("shift31_min_min", INT32_MIN, INT32_MIN, 31)
add_mbqm("shift31_max_max", INT32_MAX, INT32_MAX, 31)

# --------------------------------------------------------------------------
# Clz64
# --------------------------------------------------------------------------

clz_cases: list[tuple[str, int, int]] = []


def add_clz(label: str, n: int) -> None:
    r = im.clz64(n)
    clz_cases.append((label, n, r))


add_clz("n_1", 1)
add_clz("n_2_pow_63", 1 << 63)
add_clz("n_2_pow_64_minus_1", (1 << 64) - 1)
for k in range(0, 64):
    add_clz(f"pow2_k{k}", 1 << k)
add_clz("d21_2_pow_30", 1 << 30)
add_clz("d21_2_pow_31", 1 << 31)
add_clz("mixed_general", 0xABCDEF0123456789 & ((1 << 64) - 1))
add_clz("mixed_general_2", 0x0000000100000001)

# --------------------------------------------------------------------------
# MaxAbsReduce
# --------------------------------------------------------------------------

maxabs_cases: list[tuple[str, list[int], int]] = []


def add_maxabs(label: str, xs: list[int]) -> None:
    r = im.max_abs_reduce(xs)
    maxabs_cases.append((label, xs, r))


add_maxabs("typical", [1, -5, 3, -10, 7])
add_maxabs("contains_int32_min", [5, INT32_MIN, -3])  # widen-before-abs: D' must be 2**31
add_maxabs("all_zero_guard", [0, 0, 0, 0])  # D' = 1 (the guard)
add_maxabs("single_positive", [42])
add_maxabs("single_negative", [-42])
add_maxabs("single_int32_min", [INT32_MIN])
add_maxabs("single_zero", [0])
add_maxabs("contains_int32_max", [1, INT32_MAX, -7])
_multiset = [3, -100, 7, -55, 100]
add_maxabs("order_perm_a", _multiset)
add_maxabs("order_perm_b", list(reversed(_multiset)))
_rng_perm = random.Random(77)
_shuffled = list(_multiset)
_rng_perm.shuffle(_shuffled)
add_maxabs("order_perm_c_shuffled", _shuffled)

# --------------------------------------------------------------------------
# NormalizeScale
# --------------------------------------------------------------------------

normscale_cases: list[tuple[str, int, int, int]] = []


def add_normscale(label: str, d_prime: int) -> None:
    dn, s = im.normalize_scale(d_prime)
    assert (1 << 30) <= dn < (1 << 31)
    normscale_cases.append((label, d_prime, dn, s))


add_normscale("d_prime_1", 1)
add_normscale("d_prime_2_pow_30", 1 << 30)  # s == 0
add_normscale("d_prime_2_pow_31", 1 << 31)  # s == -1, single right-shift, Dn == 2**30
add_normscale("d_prime_2_pow_31_minus_1", (1 << 31) - 1)
add_normscale("d_prime_2", 2)
add_normscale("d_prime_3", 3)
add_normscale("d_prime_100", 100)
add_normscale("d_prime_12345", 12345)
add_normscale("d_prime_2_pow_15", 1 << 15)
add_normscale("d_prime_2_pow_29", 1 << 29)
add_normscale("d_prime_2_pow_30_minus_1", (1 << 30) - 1)
add_normscale("d_prime_2_pow_30_plus_1", (1 << 30) + 1)
add_normscale("d_prime_2_pow_31_minus_2", (1 << 31) - 2)

# --------------------------------------------------------------------------
# DynamicScaleReciprocal
# --------------------------------------------------------------------------

dynrecip_named_cases: list[tuple[str, int, int]] = []


def add_dynrecip_named(label: str, dn: int) -> None:
    r = im.dynamic_scale_reciprocal(dn)
    dynrecip_named_cases.append((label, dn, r))


add_dynrecip_named("dn_2_pow_30_boundary", 1 << 30)  # R == 2**32 exactly
add_dynrecip_named("dn_2_pow_31_minus_1_boundary", (1 << 31) - 1)
add_dynrecip_named("dn_2_pow_30_plus_1", (1 << 30) + 1)
add_dynrecip_named("dn_midpoint", (1 << 30) + (1 << 29))
add_dynrecip_named("dn_2_pow_31_minus_2", (1 << 31) - 2)

# Dense sampled subset across [2**30, 2**31) — the ongoing-CI sample (a few
# thousand points); the exhaustive full-domain certifier is Brunel's separate
# tool, not this suite.
_rng = random.Random(20260719)
_DENSE_N = 3000
dynrecip_dense: list[tuple[int, int]] = []
_seen: set[int] = set()
_lo, _hi = 1 << 30, (1 << 31) - 1
while len(dynrecip_dense) < _DENSE_N:
    dn = _rng.randint(_lo, _hi)
    if dn in _seen:
        continue
    _seen.add(dn)
    dynrecip_dense.append((dn, im.dynamic_scale_reciprocal(dn)))
dynrecip_dense.sort()

# --------------------------------------------------------------------------
# RequantTokenCode
# --------------------------------------------------------------------------

requant_cases: list[tuple[str, int, int, int, int]] = []


def add_requant(label: str, x_i: int, r: int, s: int) -> None:
    code = im.requant_token_code(x_i, r, s)
    requant_cases.append((label, x_i, r, s, code))


def find_c22_tie(exponent: int, negative: bool) -> tuple[int, int]:
    """x_i, r such that x_i*127*r is an exact away-from-zero tie at `exponent`.

    127 is odd, so it is invertible mod 2**exponent; picks the smallest
    positive r solving 127*r == 2**(exponent-1) (mod 2**exponent) with
    x_i == -1 (negative) or x_i == 1 (positive), then verifies the tie
    condition programmatically.
    """
    denom = 1 << exponent
    half = denom >> 1
    inv127 = pow(127, -1, denom)
    r = (half * inv127) % denom
    if r == 0:
        r += denom
    x_i = -1 if negative else 1
    product = x_i * 127 * r
    assert (2 * abs(product)) % (2 * denom) == denom, "not an exact C22 tie"
    return x_i, r


# Typical +/- codes, realistic magnitudes.
add_requant("typical_pos_small", 1000, 3_000_000_000, 10)
add_requant("typical_neg_small", -1000, 3_000_000_000, 10)
add_requant("typical_pos_mid", 5_000_000, 4_294_967_296, 0)  # R == 2**32, s == 0
add_requant("typical_neg_mid", -5_000_000, 4_294_967_296, 0)
add_requant("zero_x_i", 0, 4_000_000_000, 20)

# Clamp saturation to +/-127.
add_requant("clamp_saturate_pos", INT32_MAX, 4_294_967_296, 0)
add_requant("clamp_saturate_neg", INT32_MIN, 4_294_967_296, 0)
add_requant("clamp_saturate_pos_large_r", 2_000_000_000, 4_294_967_000, 5)
add_requant("clamp_saturate_neg_large_r", -2_000_000_000, 4_294_967_000, 5)

# The ~2**70 width boundary: x_i = INT32_MIN, R = 2**32 (Dn = 2**30), s = 0.
add_requant("width_boundary_int32_min_r_2pow32", INT32_MIN, 1 << 32, 0)
# Same boundary pair at a non-zero s, to exercise the exponent arithmetic too.
add_requant("width_boundary_int32_min_r_2pow32_s30", INT32_MIN, 1 << 32, 30)

# Negative-side exact-tie corpus (chartered §17 G-8, D-SLM52): away-from-zero
# must give -(k+1) where toward-+infinity (C2's rule, the wrong one to reuse
# here) would give -k. Constructed at exponents spanning C22's realistic
# s-domain (exponent = 62 - s, s in [-1, 30] per NormalizeScale -> exponent in
# [32, 63]).
for exp in (32, 40, 50, 63):
    xi, r = find_c22_tie(exp, negative=True)
    add_requant(f"neg_tie_away_from_zero_exp{exp}", xi, r, 62 - exp)
    # Positive-side mirror for symmetry documentation only (positive ties
    # round the same way under both conventions, so these do NOT discriminate
    # the rule — the negative corpus above is what's load-bearing).
    xi, r = find_c22_tie(exp, negative=False)
    add_requant(f"pos_tie_exp{exp}", xi, r, 62 - exp)

# --------------------------------------------------------------------------
# Integrated pipeline: MaxAbsReduce -> NormalizeScale -> DynamicScaleReciprocal
# -> RequantTokenCode, over one realistic token vector, per-element codes.
# --------------------------------------------------------------------------

pipeline_cases: list[dict] = []


def add_pipeline(label: str, xs: list[int]) -> None:
    d_prime = im.max_abs_reduce(xs)
    dn, s = im.normalize_scale(d_prime)
    r = im.dynamic_scale_reciprocal(dn)
    codes = [im.requant_token_code(x, r, s) for x in xs]
    pipeline_cases.append(
        {"label": label, "xs": xs, "d_prime": d_prime, "dn": dn, "s": s, "r": r, "codes": codes}
    )


add_pipeline("small_typical", [100, -200, 300, -400, 500, 0, -50])
add_pipeline("contains_int32_min", [INT32_MIN, 100, -200, 300])
add_pipeline("all_zero", [0, 0, 0, 0, 0])
add_pipeline("wide_spread", [1, -1, 1000000, -1000000, INT32_MAX // 4, INT32_MIN // 4])

# --------------------------------------------------------------------------
# S2.2 — ISqrt / ISqrtTrace (C4/C5/C6)
# --------------------------------------------------------------------------

isqrt_cases: list[tuple[str, int, int, list[int]]] = []  # label, n, expected_root, expected_iterates


def add_isqrt(label: str, n: int) -> None:
    root = im.i_sqrt(n)
    trace = im.i_sqrt_trace(n)
    assert len(trace) == im.I_SQRT_ITERATIONS, "trace must be exactly I_SQRT_ITERATIONS long"
    assert trace[-1] == root, "trace's last iterate must equal i_sqrt's returned root"
    # Independent oracle: Python's own math.isqrt, double-sourcing the golden
    # against a floor-sqrt implementation that is not intmath.py's own.
    assert root == math.isqrt(n), f"i_sqrt disagrees with math.isqrt at n={n}"
    isqrt_cases.append((label, n, root, trace))


add_isqrt("zero", 0)
add_isqrt("one", 1)

for k in (2, 3, 7, 10, 1000, 1_000_000, 3_037_000_499):
    add_isqrt(f"perfect_square_k{k}", k * k)
    add_isqrt(f"just_below_square_k{k}", k * k - 1)
    if k * k + 1 <= (1 << 63) - 1:
        add_isqrt(f"just_above_square_k{k}", k * k + 1)

for n in (2, 3, 5, 10, 99, 123_456_789, 10 ** 12 + 7):
    add_isqrt(f"non_square_{n}", n)

add_isqrt("boundary_2_pow_62", 1 << 62)
add_isqrt("boundary_int64_max", (1 << 63) - 1)

# Every base-4 digit boundary the 32-iteration recurrence steps through.
for m in range(0, 32):
    add_isqrt(f"power_of_4_m{m}", 4 ** m)

# --------------------------------------------------------------------------
# S2.2 — ShiftByMax (C9)
# --------------------------------------------------------------------------

shiftmax_cases: list[tuple[str, list[int], list[int]]] = []


def add_shiftmax(label: str, logits: list[int]) -> None:
    out = im.shift_by_max(logits)
    shiftmax_cases.append((label, logits, out))


add_shiftmax("typical", [5, 3, 8, 1])
add_shiftmax("single_element", [42])
add_shiftmax("all_equal", [7, 7, 7, 7])
add_shiftmax("max_at_first", [10, 1, 2, 3])
add_shiftmax("max_at_last", [1, 2, 3, 10])
add_shiftmax("negative_logits", [-5, -10, -1, -20])
add_shiftmax("max_already_zero", [0, -5, -10])

_wide_spread = [3_000_000_000, -3_000_000_000, 0]
assert max(_wide_spread) - min(_wide_spread) > (1 << 32) - 1, (
    "wide_spread_exceeds_int32 must exceed int32's representable span to prove "
    "int64 widening is load-bearing"
)
add_shiftmax("wide_spread_exceeds_int32", _wide_spread)

# --------------------------------------------------------------------------
# S2.2 — the i-exp core (C7/C8), reached via IExpConstruct/IExpEvaluate since
# S-HARDEN-0. Constants are derived OFFLINE, in this
# generator ONLY, by reproducing i_exp's internal (q_ln2, q_b, q_c)
# derivation for a chosen float `scale` — the C++ under test consumes these
# as opaque pre-derived integers and never performs this float division
# itself (that derivation is out of scope for this slot per the commission).
# --------------------------------------------------------------------------

_POLY_A = im._POLY_A
_POLY_B = im._POLY_B
_POLY_C = im._POLY_C
_INT64_MAX = (1 << 63) - 1


def derive_constants(scale: float) -> tuple[int, int, int]:
    """Reproduces `i_exp`'s internal (q_ln2, q_b, q_c) derivation for `scale`."""
    q_ln2 = im.i_exp_ln2_quantum(scale)
    q_b = math.floor(_POLY_B / scale)
    q_c = math.floor(_POLY_C / (_POLY_A * scale ** 2))
    return q_ln2, q_b, q_c


iexp_cases: list[tuple[str, int, int, int, int, int]] = []  # label, q, q_ln2, q_b, q_c, expected


def add_iexp(label: str, q: int, q_ln2: int, q_b: int, q_c: int) -> None:
    r = im.i_exp_from_constants(q, q_ln2, q_b, q_c)
    assert -_INT64_MAX - 1 <= r <= _INT64_MAX, f"golden for {label} does not fit int64_t: {r}"
    iexp_cases.append((label, q, q_ln2, q_b, q_c, r))


# Realistic constant triples across plausible per-token activation scales.
_realistic_scales = [0.1, 0.05, 0.02, 0.005, 1.0 / 127.0]
for idx, scale in enumerate(_realistic_scales):
    q_ln2, q_b, q_c = derive_constants(scale)
    assert q_ln2 >= 1

    # Cross-check against i_exp(q, scale) itself — D-SLM52's other, float-entry
    # code path through the SAME decomposition — at q=0, double-sourcing this
    # triple's golden against an independent call into the reference rather
    # than only i_exp_from_constants agreeing with itself.
    cross_q_out, _ = im.i_exp(0, scale)
    assert im.i_exp_from_constants(0, q_ln2, q_b, q_c) == cross_q_out, (
        "i_exp_from_constants(0, ...) must match i_exp(0, scale)'s own derivation"
    )

    add_iexp(f"realistic_s{idx}_q_max_element", 0, q_ln2, q_b, q_c)
    add_iexp(f"realistic_s{idx}_q_neg1", -1, q_ln2, q_b, q_c)
    add_iexp(f"realistic_s{idx}_q_neg_one_ln2_step", -q_ln2, q_ln2, q_b, q_c)
    add_iexp(f"realistic_s{idx}_q_neg_five_ln2_steps", -5 * q_ln2, q_ln2, q_b, q_c)

    clip_q = -im.I_EXP_CLIP_N * q_ln2
    add_iexp(f"realistic_s{idx}_clip_boundary", clip_q, q_ln2, q_b, q_c)
    beyond_clip_q = clip_q - q_ln2
    r_clip = im.i_exp_from_constants(clip_q, q_ln2, q_b, q_c)
    r_beyond = im.i_exp_from_constants(beyond_clip_q, q_ln2, q_b, q_c)
    assert r_clip == r_beyond, "an input past the clip must clamp to the clip point's result"
    add_iexp(f"realistic_s{idx}_beyond_clip", beyond_clip_q, q_ln2, q_b, q_c)

# q_ln2 == 1, the minimum: scale chosen so floor(ln2/scale) == 1.
_scale_qln2_1 = 0.5
_q_ln2_min, _q_b_min, _q_c_min = derive_constants(_scale_qln2_1)
assert _q_ln2_min == 1, "scale must be chosen so the ln2 quantum is exactly 1"
add_iexp("qln2_min_q0", 0, _q_ln2_min, _q_b_min, _q_c_min)
add_iexp("qln2_min_q_neg1", -1, _q_ln2_min, _q_b_min, _q_c_min)
add_iexp("qln2_min_clip_boundary", -im.I_EXP_CLIP_N, _q_ln2_min, _q_b_min, _q_c_min)
add_iexp("qln2_min_beyond_clip", -im.I_EXP_CLIP_N - 5, _q_ln2_min, _q_b_min, _q_c_min)

# Width probe: the widest intermediate the reference's domain permits, found
# by shrinking `scale` until q_c (the dominant term, since it scales as
# 1/scale**2) approaches INT64_MAX while the FULL (q_p + q_b)**2 + q_c result
# still fits a signed 64-bit golden -- a wider intermediate could not be
# stored as the expected int64_t at all, so this is the largest value this
# fixture format can even express. Deterministic closed-form estimate plus a
# fixed refinement loop; no randomness.
_width_target = _INT64_MAX // 2  # headroom under the exact bound for rounding slop
_width_scale = math.sqrt(_POLY_C / (_POLY_A * _width_target))


def _width_probe_constants(scale: float) -> tuple[int, int, int, int]:
    q_ln2, q_b, q_c = derive_constants(scale)
    val = q_b * q_b + q_c  # the q=0 (z=0, no shift) result -- the widest point in this family
    return q_ln2, q_b, q_c, val


_wp_q_ln2, _wp_q_b, _wp_q_c, _wp_val = _width_probe_constants(_width_scale)
while _wp_val >= _width_target or _wp_q_ln2 < 1:
    _width_scale *= 1.01
    _wp_q_ln2, _wp_q_b, _wp_q_c, _wp_val = _width_probe_constants(_width_scale)
assert 0 < _wp_val < _INT64_MAX

add_iexp("width_probe_q0_max_element", 0, _wp_q_ln2, _wp_q_b, _wp_q_c)
add_iexp("width_probe_q_neg_one_ln2_step", -_wp_q_ln2, _wp_q_ln2, _wp_q_b, _wp_q_c)

WIDTH_PROBE_MAGNITUDE = _wp_val

# --------------------------------------------------------------------------
# S2.3 — RopeApplyPair (C11/C13)
# --------------------------------------------------------------------------

_INT64_MIN = -(1 << 63)
_INT64_MAX_ROPE = (1 << 63) - 1

rope_cases: list[tuple[str, int, int, int, int, int, int]] = []
# label, x, y, cos_q30, sin_q30, expected_x, expected_y


def add_rope(label: str, x: int, y: int, cos_q30: int, sin_q30: int) -> None:
    ex, ey = rope.rope_apply_pair(x, y, cos_q30, sin_q30)
    assert _INT64_MIN <= ex <= _INT64_MAX_ROPE and _INT64_MIN <= ey <= _INT64_MAX_ROPE, (
        f"{label}: golden ({ex}, {ey}) does not fit int64_t"
    )
    rope_cases.append((label, x, y, cos_q30, sin_q30, ex, ey))


ROPE_ONE = rope.ROPE_ONE
ROPE_FRAC_BITS = rope.ROPE_FRAC_BITS

# --- Identity / passthrough: angle 0 (cos=ROPE_ONE, sin=0) -> exact (x, y),
#     verified programmatically (the x*ROPE_ONE / ROPE_ONE division has zero
#     remainder, so no rounding fires). ---
for label, x, y in [
    ("identity_zero", 0, 0),
    ("identity_pos", 12345, 6789),
    ("identity_neg", -12345, -6789),
    ("identity_max", INT32_MAX, INT32_MAX),
    ("identity_min", INT32_MIN, INT32_MIN),
    ("identity_mixed_sign", INT32_MAX, INT32_MIN),
]:
    add_rope(label, x, y, ROPE_ONE, 0)
    ex, ey = rope_cases[-1][5], rope_cases[-1][6]
    assert (ex, ey) == (x, y), f"{label}: identity rotation must reproduce (x, y) exactly, got ({ex}, {ey})"

# --- Quarter turns: cos=0 isolates the pure-sin rotation. ---
for label, x, y in [
    ("typical", 100, 200),
    ("zero", 0, 0),
    ("negative", -300, 450),
    ("extreme", INT32_MAX, INT32_MIN),
]:
    add_rope(f"quarter_pos_sin_{label}", x, y, 0, ROPE_ONE)
    ex, ey = rope_cases[-1][5], rope_cases[-1][6]
    assert (ex, ey) == (-y, x), f"quarter_pos_sin_{label}: expected (-y, x) == ({-y}, {x}), got ({ex}, {ey})"

    add_rope(f"quarter_neg_sin_{label}", x, y, 0, -ROPE_ONE)
    ex, ey = rope_cases[-1][5], rope_cases[-1][6]
    assert (ex, ey) == (y, -x), f"quarter_neg_sin_{label}: expected (y, -x) == ({y}, {-x}), got ({ex}, {ey})"

# --- cos/sin edge corners not already covered by identity/quarter-turn. ---
add_rope("edge_cos_neg_one_sin_zero", 777, -333, -ROPE_ONE, 0)
_ex, _ey = rope_cases[-1][5], rope_cases[-1][6]
assert (_ex, _ey) == (-777, 333), "180-degree turn (cos=-1, sin=0) must give (-x, -y) exactly"

add_rope("edge_cos_zero_sin_zero", 555, -222, 0, 0)
_ex, _ey = rope_cases[-1][5], rope_cases[-1][6]
assert (_ex, _ey) == (0, 0), "degenerate cos=sin=0 must give (0, 0)"

add_rope("edge_cos_one_sin_one_typical", 100, -50, ROPE_ONE, ROPE_ONE)
add_rope("edge_cos_negone_sin_negone_typical", 100, -50, -ROPE_ONE, -ROPE_ONE)

# --- General angles: real Q2.30 rows from rope_tables (a Qwen2.5-1.5B-shaped
#     RoPE configuration: head_dim=128, so 64 pairs; context_cap=128;
#     theta=1000000.0), sampled across positions and pair indices, each row
#     paired with a rotating set of representative (x, y) activation values so
#     the general-angle class also exercises sign and magnitude diversity. ---
_ROPE_HEAD_DIM = 128
_ROPE_CONTEXT_CAP = 128
_ROPE_THETA = 1000000.0
_cos_table, _sin_table = rope.rope_tables(_ROPE_HEAD_DIM, _ROPE_CONTEXT_CAP, _ROPE_THETA)
_general_xy = [(37, -58), (0, 0), (-1000, 2000), (INT32_MAX // 4, -(INT32_MAX // 4)), (-777, -777), (999999, 1)]

_general_idx = 0
for _pos in (0, 1, 7, 63, 127):
    for _pair in (0, 5, 15, 31, 63):
        _x, _y = _general_xy[_general_idx % len(_general_xy)]
        _general_idx += 1
        _cos = _cos_table[_pos][_pair]
        _sin = _sin_table[_pos][_pair]
        add_rope(f"general_pos{_pos}_pair{_pair}", _x, _y, _cos, _sin)

# A row whose sin_q30 is negative and one whose cos_q30 is negative, found by
# scanning the same realistic table — proves the primitive on genuinely
# negative table entries, not only hand-picked ±ROPE_ONE/0 edges.
_neg_sin_row = None
_neg_cos_row = None
for _pos in range(_ROPE_CONTEXT_CAP):
    for _pair in range(_ROPE_HEAD_DIM // 2):
        if _neg_sin_row is None and _sin_table[_pos][_pair] < 0:
            _neg_sin_row = (_pos, _pair)
        if _neg_cos_row is None and _cos_table[_pos][_pair] < 0:
            _neg_cos_row = (_pos, _pair)
    if _neg_sin_row is not None and _neg_cos_row is not None:
        break
assert _neg_sin_row is not None, "expected at least one table row with sin_q30 < 0 across this domain"
assert _neg_cos_row is not None, "expected at least one table row with cos_q30 < 0 across this domain"

_p, _q = _neg_sin_row
add_rope("negative_sin_from_table", 4096, -8192, _cos_table[_p][_q], _sin_table[_p][_q])
assert rope_cases[-1][4] < 0, "negative_sin_from_table must carry sin_q30 < 0"

_p, _q = _neg_cos_row
add_rope("negative_cos_from_table", -8192, 4096, _cos_table[_p][_q], _sin_table[_p][_q])
assert rope_cases[-1][3] < 0, "negative_cos_from_table must carry cos_q30 < 0"

# --- Sign coverage: all four (x, y) quadrants against one fixed, realistic,
#     non-trivial table row. ---
_quad_cos, _quad_sin = _cos_table[10][3], _sin_table[10][3]
for label, x, y in [
    ("quadrant_pos_pos", 500, 300),
    ("quadrant_neg_pos", -500, 300),
    ("quadrant_pos_neg", 500, -300),
    ("quadrant_neg_neg", -500, -300),
]:
    add_rope(label, x, y, _quad_cos, _quad_sin)

# --- Wide inputs: x, y near INT32_MAX/INT32_MIN with cos/sin near +/-ROPE_ONE —
#     proves the int64 intermediate (~2^61) does not overflow AND that the
#     result can exceed int32 range (the int64 return type is load-bearing). ---
_wide_specs = [
    ("wide_max_min_cos1_sin1", INT32_MAX, INT32_MIN, ROPE_ONE, ROPE_ONE),
    ("wide_min_max_cos1_sin1", INT32_MIN, INT32_MAX, ROPE_ONE, ROPE_ONE),
    ("wide_max_max_cos1_negsin1", INT32_MAX, INT32_MAX, ROPE_ONE, -ROPE_ONE),
    ("wide_min_min_cos1_negsin1", INT32_MIN, INT32_MIN, ROPE_ONE, -ROPE_ONE),
    ("wide_max_min_negcos1_sin1", INT32_MAX, INT32_MIN, -ROPE_ONE, ROPE_ONE),
    ("wide_min_max_negcos1_negsin1", INT32_MIN, INT32_MAX, -ROPE_ONE, -ROPE_ONE),
    ("wide_min_min_negcos1_sin1", INT32_MIN, INT32_MIN, -ROPE_ONE, ROPE_ONE),
    ("wide_max_max_negcos1_negsin1", INT32_MAX, INT32_MAX, -ROPE_ONE, -ROPE_ONE),
]
_wide_exceeds_int32 = 0
for label, x, y, cos_q30, sin_q30 in _wide_specs:
    add_rope(label, x, y, cos_q30, sin_q30)
    ex, ey = rope_cases[-1][5], rope_cases[-1][6]
    if abs(ex) > INT32_MAX or abs(ey) > INT32_MAX:
        _wide_exceeds_int32 += 1
assert _wide_exceeds_int32 >= 1, (
    "the wide-input class must include at least one case whose result exceeds int32 range, "
    f"proving the int64 return type is load-bearing; found {_wide_exceeds_int32}"
)
WIDE_LARGEST_MAGNITUDE = max(
    max(abs(c[5]), abs(c[6])) for c in rope_cases if c[0].startswith("wide_")
)

# --- Tie behavior: construct inputs where x*cos - y*sin (and/or x*sin + y*cos)
#     is an exact half at 2^30 (a value == 2^29 mod 2^30), on both signs —
#     proving RoundingDivideByPOT's away-from-zero tie is inherited correctly.
#     Verified programmatically via Fraction, independent of rope_apply_pair's
#     own arithmetic. ---
_ROPE_HALF = 1 << (ROPE_FRAC_BITS - 1)  # 2**29


def _assert_exact_half(value: int, what: str) -> None:
    frac = Fraction(value, 1 << ROPE_FRAC_BITS)
    floor_v = value >> ROPE_FRAC_BITS
    assert frac - floor_v == Fraction(1, 2), f"{what} == {value} is not an exact half point at 2^{ROPE_FRAC_BITS}"


# x-component ties: y=0, sin_q30=0 isolates x*cos_q30 as the rounded term.
_assert_exact_half(_ROPE_HALF, "tie_x_pos target")
add_rope("tie_x_pos", 1, 0, _ROPE_HALF, 0)
_assert_exact_half(-_ROPE_HALF, "tie_x_neg target")
add_rope("tie_x_neg", -1, 0, _ROPE_HALF, 0)

# y-component ties: x=0, sin_q30=0 isolates y*cos_q30 as the rounded term of
# the y-output (x*sin + y*cos).
add_rope("tie_y_pos", 0, 1, _ROPE_HALF, 0)
add_rope("tie_y_neg", 0, -1, _ROPE_HALF, 0)

# Both components tie simultaneously in the same call: x=y=+/-1, cos_q30 =
# 2**29, sin_q30 = 0 -> x_out's rounded term is x*cos - y*sin = 2^29 (exact
# half) and y_out's rounded term is x*sin + y*cos = 2^29 (exact half) too.
add_rope("tie_both_pos", 1, 1, _ROPE_HALF, 0)
add_rope("tie_both_neg", -1, -1, _ROPE_HALF, 0)
for _tp, _tn in (("tie_x_pos", "tie_x_neg"), ("tie_y_pos", "tie_y_neg"), ("tie_both_pos", "tie_both_neg")):
    _pos_case = next(c for c in rope_cases if c[0] == _tp)
    _neg_case = next(c for c in rope_cases if c[0] == _tn)
    # Away-from-zero: the positive tie rounds UP (away from 0), the negative
    # tie rounds DOWN (away from 0) — never toward zero on either side.
    assert abs(_pos_case[5]) >= 1 or abs(_pos_case[6]) >= 1, f"{_tp}: away-from-zero rounding must move off 0"
    assert abs(_neg_case[5]) >= 1 or abs(_neg_case[6]) >= 1, f"{_tn}: away-from-zero rounding must move off 0"

# --------------------------------------------------------------------------
# Emit the C++ header.
# --------------------------------------------------------------------------


def cxx_i32(v: int) -> str:
    if v == INT32_MIN:
        return "superslm::kInt32Min"
    if v == INT32_MAX:
        return "superslm::kInt32Max"
    return f"INT32_C({v})"


def cxx_i64(v: int) -> str:
    return f"INT64_C({v})"


def cxx_u64(v: int) -> str:
    return f"UINT64_C({v})"


def cxx_str(s: str) -> str:
    return f'"{s}"'


lines: list[str] = []
emit = lines.append

emit("// GENERATED FILE. Do not hand-edit.")
emit("//")
emit("// Produced by tests/gen_intmath_fixtures.py from the pinned Python reference")
emit("// Tools/superslm_spike/intmath.py (D-SLM52). Every value here is the output of")
emit("// calling that reference's functions, not a hand computation. Re-running the")
emit("// generator must reproduce this file byte-for-byte.")
emit("//")
emit("// Test-design record:")
emit("// Claude/Curie/superslm-s2.1-intmath-test-design-2026-07-19.md")
emit("#ifndef SUPERSLM_TESTS_SSLM_INTMATH_FIXTURES_H")
emit("#define SUPERSLM_TESTS_SSLM_INTMATH_FIXTURES_H")
emit("")
emit('#include "superslm/intmath.h"')
emit("")
emit("#include <cstddef>")
emit("#include <cstdint>")
emit("")
emit("namespace superslm_test {")
emit("")

# --- C2 ---
emit("// --- C2: SaturatingRoundingDoublingHighMul -----------------------------------")
emit("")
emit("struct C2Case {")
emit("\tconst char* label;")
emit("\tint32_t a;")
emit("\tint32_t b;")
emit("\tint32_t expected;")
emit("};")
emit("")
emit(f"inline constexpr C2Case kC2Cases[] = {{")
for label, a, b, r in c2_cases:
    emit(f"\t{{{cxx_str(label)}, {cxx_i32(a)}, {cxx_i32(b)}, {cxx_i32(r)}}},")
emit("};")
emit(f"inline constexpr size_t kC2CasesCount = {len(c2_cases)};")
emit("")

# --- C1/C3 ---
emit("// --- C1/C3: RoundingDivideByPOT -----------------------------------------------")
emit("")
emit("struct C1C3Case {")
emit("\tconst char* label;")
emit("\tint32_t x;")
emit("\tint exponent;")
emit("\tint32_t expected;")
emit("};")
emit("")
emit("inline constexpr C1C3Case kC1C3Cases[] = {")
for label, x, exp, r in c1c3_cases:
    emit(f"\t{{{cxx_str(label)}, {cxx_i32(x)}, {exp}, {cxx_i32(r)}}},")
emit("};")
emit(f"inline constexpr size_t kC1C3CasesCount = {len(c1c3_cases)};")
emit("")

# --- MultiplyByQuantizedMultiplier ---
emit("// --- C1: MultiplyByQuantizedMultiplier (composition) --------------------------")
emit("")
emit("struct MbqmCase {")
emit("\tconst char* label;")
emit("\tint32_t x;")
emit("\tint32_t multiplier;")
emit("\tint shift;")
emit("\tint32_t expected;")
emit("};")
emit("")
emit("inline constexpr MbqmCase kMbqmCases[] = {")
for label, x, m, shift, r in mbqm_cases:
    emit(f"\t{{{cxx_str(label)}, {cxx_i32(x)}, {cxx_i32(m)}, {shift}, {cxx_i32(r)}}},")
emit("};")
emit(f"inline constexpr size_t kMbqmCasesCount = {len(mbqm_cases)};")
emit("")

# --- Clz64 ---
emit("// --- C21 helper: Clz64 ---------------------------------------------------------")
emit("")
emit("struct Clz64Case {")
emit("\tconst char* label;")
emit("\tuint64_t n;")
emit("\tint expected;")
emit("};")
emit("")
emit("inline constexpr Clz64Case kClz64Cases[] = {")
for label, n, r in clz_cases:
    emit(f"\t{{{cxx_str(label)}, {cxx_u64(n)}, {r}}},")
emit("};")
emit(f"inline constexpr size_t kClz64CasesCount = {len(clz_cases)};")
emit("")

# --- MaxAbsReduce ---
emit("// --- C20: MaxAbsReduce -----------------------------------------------------------")
emit("")
for idx, (label, xs, r) in enumerate(maxabs_cases):
    arr = ", ".join(cxx_i32(v) for v in xs) if xs else ""
    emit(f"inline constexpr int32_t kMaxAbsData{idx}[] = {{{arr if arr else '0'}}};  // {label}")
emit("")
emit("struct MaxAbsCase {")
emit("\tconst char* label;")
emit("\tconst int32_t* data;")
emit("\tsize_t n;")
emit("\tint64_t expected;")
emit("};")
emit("")
emit("inline constexpr MaxAbsCase kMaxAbsCases[] = {")
for idx, (label, xs, r) in enumerate(maxabs_cases):
    emit(f"\t{{{cxx_str(label)}, kMaxAbsData{idx}, {len(xs)}, {cxx_i64(r)}}},")
emit("};")
emit(f"inline constexpr size_t kMaxAbsCasesCount = {len(maxabs_cases)};")
emit("")

# --- NormalizeScale ---
emit("// --- C21: NormalizeScale ---------------------------------------------------------")
emit("")
emit("struct NormalizeScaleCase {")
emit("\tconst char* label;")
emit("\tint64_t d_prime;")
emit("\tint64_t expected_dn;")
emit("\tint expected_s;")
emit("};")
emit("")
emit("inline constexpr NormalizeScaleCase kNormalizeScaleCases[] = {")
for label, dp, dn, s in normscale_cases:
    emit(f"\t{{{cxx_str(label)}, {cxx_i64(dp)}, {cxx_i64(dn)}, {s}}},")
emit("};")
emit(f"inline constexpr size_t kNormalizeScaleCasesCount = {len(normscale_cases)};")
emit("")

# --- DynamicScaleReciprocal ---
emit("// --- C19: DynamicScaleReciprocal --------------------------------------------------")
emit("")
emit("struct DynRecipNamedCase {")
emit("\tconst char* label;")
emit("\tint64_t dn;")
emit("\tint64_t expected_r;")
emit("};")
emit("")
emit("inline constexpr DynRecipNamedCase kDynRecipNamedCases[] = {")
for label, dn, r in dynrecip_named_cases:
    emit(f"\t{{{cxx_str(label)}, {cxx_i64(dn)}, {cxx_i64(r)}}},")
emit("};")
emit(f"inline constexpr size_t kDynRecipNamedCasesCount = {len(dynrecip_named_cases)};")
emit("")
emit("struct DynRecipDenseCase {")
emit("\tint64_t dn;")
emit("\tint64_t expected_r;")
emit("};")
emit("")
emit(f"// Dense sampled subset across [2**30, 2**31) -- {len(dynrecip_dense)} points, seeded")
emit("// deterministically (Python random.Random(20260719)). This is the G-8")
emit("// ongoing-CI sample; the exhaustive full-domain proof is Brunel's separate")
emit("// certifier, not this suite.")
emit("inline constexpr DynRecipDenseCase kDynRecipDenseCases[] = {")
for dn, r in dynrecip_dense:
    emit(f"\t{{{cxx_i64(dn)}, {cxx_i64(r)}}},")
emit("};")
emit(f"inline constexpr size_t kDynRecipDenseCasesCount = {len(dynrecip_dense)};")
emit("")

# --- RequantTokenCode ---
emit("// --- C22: RequantTokenCode -----------------------------------------------------")
emit("")
emit("struct RequantCase {")
emit("\tconst char* label;")
emit("\tint32_t x_i;")
emit("\tint64_t r;")
emit("\tint s;")
emit("\tint8_t expected;")
emit("};")
emit("")
emit("inline constexpr RequantCase kRequantCases[] = {")
for label, xi, r, s, code in requant_cases:
    emit(f"\t{{{cxx_str(label)}, {cxx_i32(xi)}, {cxx_i64(r)}, {s}, INT8_C({code})}},")
emit("};")
emit(f"inline constexpr size_t kRequantCasesCount = {len(requant_cases)};")
emit("")

# --- Integrated pipeline ---
emit("// --- Integrated pipeline: MaxAbsReduce -> NormalizeScale ->")
emit("//     DynamicScaleReciprocal -> RequantTokenCode, per-element, over one")
emit("//     realistic token vector. Proves the four C19-C22 primitives compose to")
emit("//     the reference's end-to-end per-token codes, not merely that each")
emit("//     agrees with itself in isolation. ---")
emit("")
for idx, case in enumerate(pipeline_cases):
    arr = ", ".join(cxx_i32(v) for v in case["xs"])
    emit(f"inline constexpr int32_t kPipelineXs{idx}[] = {{{arr}}};  // {case['label']}")
    codes_arr = ", ".join(f"INT8_C({c})" for c in case["codes"])
    emit(f"inline constexpr int8_t kPipelineCodes{idx}[] = {{{codes_arr}}};")
emit("")
emit("struct PipelineCase {")
emit("\tconst char* label;")
emit("\tconst int32_t* xs;")
emit("\tconst int8_t* expected_codes;")
emit("\tsize_t n;")
emit("\tint64_t expected_d_prime;")
emit("\tint64_t expected_dn;")
emit("\tint expected_s;")
emit("\tint64_t expected_r;")
emit("};")
emit("")
emit("inline constexpr PipelineCase kPipelineCases[] = {")
for idx, case in enumerate(pipeline_cases):
    emit(
        f"\t{{{cxx_str(case['label'])}, kPipelineXs{idx}, kPipelineCodes{idx}, "
        f"{len(case['xs'])}, {cxx_i64(case['d_prime'])}, {cxx_i64(case['dn'])}, "
        f"{case['s']}, {cxx_i64(case['r'])}}},"
    )
emit("};")
emit(f"inline constexpr size_t kPipelineCasesCount = {len(pipeline_cases)};")
emit("")

# --- S2.2: ISqrt / ISqrtTrace (C4/C5/C6) ---
emit("// --- C4/C5/C6: ISqrt / ISqrtTrace -----------------------------------------------")
emit("")
for idx, (label, n, root, trace) in enumerate(isqrt_cases):
    arr = ", ".join(cxx_i64(v) for v in trace)
    emit(f"inline constexpr int64_t kISqrtIterates{idx}[] = {{{arr}}};  // {label}")
emit("")
emit("struct ISqrtCase {")
emit("\tconst char* label;")
emit("\tint64_t n;")
emit("\tint64_t expected_root;")
emit("\tconst int64_t* expected_iterates;  // I_SQRT_ITERATIONS entries")
emit("};")
emit("")
emit("inline constexpr ISqrtCase kISqrtCases[] = {")
for idx, (label, n, root, trace) in enumerate(isqrt_cases):
    emit(f"\t{{{cxx_str(label)}, {cxx_i64(n)}, {cxx_i64(root)}, kISqrtIterates{idx}}},")
emit("};")
emit(f"inline constexpr size_t kISqrtCasesCount = {len(isqrt_cases)};")
emit("")

# --- S2.2: ShiftByMax (C9) ---
emit("// --- C9: ShiftByMax --------------------------------------------------------------")
emit("")
for idx, (label, logits, expected) in enumerate(shiftmax_cases):
    larr = ", ".join(cxx_i64(v) for v in logits)
    earr = ", ".join(cxx_i64(v) for v in expected)
    emit(f"inline constexpr int64_t kShiftMaxLogits{idx}[] = {{{larr}}};  // {label}")
    emit(f"inline constexpr int64_t kShiftMaxExpected{idx}[] = {{{earr}}};")
emit("")
emit("struct ShiftByMaxCase {")
emit("\tconst char* label;")
emit("\tconst int64_t* logits;")
emit("\tconst int64_t* expected;")
emit("\tsize_t n;")
emit("};")
emit("")
emit("inline constexpr ShiftByMaxCase kShiftByMaxCases[] = {")
for idx, (label, logits, expected) in enumerate(shiftmax_cases):
    emit(
        f"\t{{{cxx_str(label)}, kShiftMaxLogits{idx}, kShiftMaxExpected{idx}, {len(logits)}}},"
    )
emit("};")
emit(f"inline constexpr size_t kShiftByMaxCasesCount = {len(shiftmax_cases)};")
emit("")

# --- S2.2: the i-exp core (C7/C8) ---
emit("// --- C7/C8: IExpFromConstants ------------------------------------------------------")
emit("")
emit(f"// Width probe: the widest intermediate constructed in this suite is")
emit(f"// {WIDTH_PROBE_MAGNITUDE} (case \"width_probe_q0_max_element\") -- flagged for the")
emit("// green-phase implementation's intermediate width (int64 vs a wider type).")
emit("")
emit("struct IExpCase {")
emit("\tconst char* label;")
emit("\tint64_t q;")
emit("\tint64_t q_ln2;")
emit("\tint64_t q_b;")
emit("\tint64_t q_c;")
emit("\tint64_t expected;")
emit("};")
emit("")
emit("inline constexpr IExpCase kIExpCases[] = {")
for label, q, q_ln2, q_b, q_c, r in iexp_cases:
    emit(
        f"\t{{{cxx_str(label)}, {cxx_i64(q)}, {cxx_i64(q_ln2)}, {cxx_i64(q_b)}, "
        f"{cxx_i64(q_c)}, {cxx_i64(r)}}},"
    )
emit("};")
emit(f"inline constexpr size_t kIExpCasesCount = {len(iexp_cases)};")
emit("")

# --- S2.3: RopeApplyPair (C11/C13) ---
emit("// --- C11/C13: RopeApplyPair -----------------------------------------------------")
emit("")
emit(f"// Widest magnitude constructed by the wide-input class: {WIDE_LARGEST_MAGNITUDE}")
emit("// (exceeds INT32_MAX -- the int64 return type is load-bearing).")
emit("")
emit("struct RopeCase {")
emit("\tconst char* label;")
emit("\tint32_t x;")
emit("\tint32_t y;")
emit("\tint32_t cos_q30;")
emit("\tint32_t sin_q30;")
emit("\tint64_t expected_x;")
emit("\tint64_t expected_y;")
emit("};")
emit("")
emit("inline constexpr RopeCase kRopeCases[] = {")
for label, x, y, cos_q30, sin_q30, ex, ey in rope_cases:
    emit(
        f"\t{{{cxx_str(label)}, {cxx_i32(x)}, {cxx_i32(y)}, {cxx_i32(cos_q30)}, {cxx_i32(sin_q30)}, "
        f"{cxx_i64(ex)}, {cxx_i64(ey)}}},"
    )
emit("};")
emit(f"inline constexpr size_t kRopeCasesCount = {len(rope_cases)};")
emit("")

emit("}  // namespace superslm_test")
emit("")
emit("#endif  // SUPERSLM_TESTS_SSLM_INTMATH_FIXTURES_H")
emit("")

text = "\n".join(lines)
# Pure ASCII, written as UTF-8 with an explicit newline. Both halves are load-bearing for
# the byte-for-byte reproducibility this fixture header depends on: open(..., "w") with no
# encoding= uses the platform's locale encoding, so a single non-ASCII character (e.g. an
# em dash in a comment) would serialize to different bytes on a Windows host (cp1252) than
# on a Linux one (UTF-8) for an unchanged fixture set, and would raise outright under a
# C/POSIX locale. Asserting ASCII here means the property cannot lapse by someone adding a
# typographic character to a comment above. (Same defect and same fix as
# tools/gen_matmul_golden.py and tools/gen_silu_lut_golden.py.)
text.encode("ascii")  # raises UnicodeEncodeError if a non-ASCII character crept in
with open(OUT_PATH, "w", newline="\n", encoding="utf-8") as f:
    f.write(text)

print(f"Wrote {OUT_PATH}")
print(f"C2: {len(c2_cases)} cases")
print(f"C1/C3: {len(c1c3_cases)} cases")
print(f"MultiplyByQuantizedMultiplier: {len(mbqm_cases)} cases")
print(f"Clz64: {len(clz_cases)} cases")
print(f"MaxAbsReduce: {len(maxabs_cases)} cases")
print(f"NormalizeScale: {len(normscale_cases)} cases")
print(f"DynamicScaleReciprocal named: {len(dynrecip_named_cases)} cases")
print(f"DynamicScaleReciprocal dense: {len(dynrecip_dense)} cases")
print(f"RequantTokenCode: {len(requant_cases)} cases")
print(f"Pipeline: {len(pipeline_cases)} cases")
print(f"ISqrt/ISqrtTrace: {len(isqrt_cases)} cases")
print(f"ShiftByMax: {len(shiftmax_cases)} cases")
print(f"IExpFromConstants: {len(iexp_cases)} cases")
print(f"IExpFromConstants width probe magnitude: {WIDTH_PROBE_MAGNITUDE}")
print(f"RopeApplyPair: {len(rope_cases)} cases")
print(f"RopeApplyPair widest constructed magnitude: {WIDE_LARGEST_MAGNITUDE}")
