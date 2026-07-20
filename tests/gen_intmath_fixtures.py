#!/usr/bin/env python3
"""Generates sslm_intmath_fixtures.h from the pinned Python reference (D-SLM52).

Every expected value below is computed by IMPORTING and CALLING
Tools/superslm_spike/intmath.py — never hand-computed, never copied from any
C++ implementation. Re-running this script must reproduce
sslm_intmath_fixtures.h byte-for-byte (Poirot's reproducibility check).

Test-design record: Claude/Curie/superslm-s2.1-intmath-test-design-2026-07-19.md
"""

from __future__ import annotations

import os
import random
import sys
from fractions import Fraction

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPIKE_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "Wizard", ".claude", "worktrees",
                 "superslm-dev-continue-d0b08e", "Tools", "superslm_spike")
)
sys.path.insert(0, _SPIKE_DIR)

import intmath as im  # noqa: E402

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
emit(f"// Dense sampled subset across [2**30, 2**31) — {len(dynrecip_dense)} points, seeded")
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

emit("}  // namespace superslm_test")
emit("")
emit("#endif  // SUPERSLM_TESTS_SSLM_INTMATH_FIXTURES_H")
emit("")

with open(OUT_PATH, "w", newline="\n") as f:
    f.write("\n".join(lines))

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
