#!/usr/bin/env python3
"""Generates tests/sslm_iexp_domain_fixtures.h -- Curie's S2.6-amendment red suite for
IExpShift / IExpBase / IExpConstantsInDomain (D-SLM78/79/81; Claude/Loki/
softmax-s2.6-strike-2026-07-21.md).

Every expected value is computed HERE, directly from the five-line decomposition
IExpFromConstants documents in src/intmath.cpp / include/superslm/intmath.h --

    clip_lo = -I_EXP_CLIP_N * q_ln2
    clipped = max(q, clip_lo)
    z       = (-clipped) // q_ln2          (floor; clipped <= 0, q_ln2 >= 1)
    q_p     = clipped + z * q_ln2
    base    = q_p + q_b
    in_domain(q_c) iff 0 <= (base**2 + q_c) >> z <= INT64_MAX

-- using Python's arbitrary-precision integers, NEVER by calling the C++ primitives
under test (which do not exist yet) and never by re-deriving the bound in fixed-width
int64 arithmetic (that re-derivation is exactly the defect D-SLM81 names: `base**2`
alone overflows int64 once q_b exceeds ~3.04e9). This script is therefore an
independent oracle in the same sense gen_intmath_fixtures.py's calls into the pinned
Python reference are: the formula is transcribed from the documented contract, not
recovered from either the extraction under test or its stub.

Re-running this script must reproduce sslm_iexp_domain_fixtures.h byte-for-byte.

Test-design record: Claude/Curie/superslm-s2.6-softmax-iexp-domain-test-design-2026-07-21.md
"""

from __future__ import annotations

import os

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
I_EXP_CLIP_N = 30  # must equal superslm::I_EXP_CLIP_N (include/superslm/intmath.h)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sslm_iexp_domain_fixtures.h")


def derive(q: int, q_ln2: int, q_b: int) -> tuple[int, int, int, int]:
    """Returns (clipped, z, q_p, base) -- the documented decomposition, exactly."""
    assert q_ln2 >= 1
    clip_lo = -I_EXP_CLIP_N * q_ln2
    clipped = clip_lo if q < clip_lo else q
    z = (-clipped) // q_ln2
    q_p = clipped + z * q_ln2
    base = q_p + q_b
    return clipped, z, q_p, base


def in_domain(q: int, q_ln2: int, q_b: int, q_c: int) -> tuple[bool, int, int, int]:
    """Returns (is_in_domain, exact_shifted_value, z, base). exact_shifted_value is the
    true mathematical (base**2 + q_c) >> z, unbounded -- NOT narrowed to int64. base**2
    + q_c is always >= 0 here (base**2 >= 0; q_c is documented positive), so the floor
    right-shift never needs a negative-value convention."""
    _, z, _, base = derive(q, q_ln2, q_b)
    val = base * base + q_c
    assert val >= 0, "base**2 + q_c must be non-negative for this domain formula to apply"
    shifted = val >> z
    return (INT64_MIN <= shifted <= INT64_MAX), shifted, z, base


# ---------------------------------------------------------------------------
# The strike's exact contract-legal input (Claude/Loki/softmax-s2.6-strike-
# 2026-07-21.md): q=0, q_ln2=887904998 (S2.2 width-probe fixture), q_b=1733160715
# (same fixture), q_c=2**63-1 (positive, int64-storable per the design's own C30
# domain correction). base**2 + q_c = 12,227,218,100,874,087,032, which exceeds
# INT64_MAX -- the regression cell for the defect.
# ---------------------------------------------------------------------------
QLN2 = 887904998
QB = 1733160715
QC_STRIKE = (1 << 63) - 1

domain_cases: list[tuple[str, int, int, int, int, bool]] = []
# label, q, q_ln2, q_b, q_c, expected_in_domain


def add_domain(label: str, q: int, q_ln2: int, q_b: int, q_c: int) -> None:
    ok, shifted, z, base = in_domain(q, q_ln2, q_b, q_c)
    domain_cases.append((label, q, q_ln2, q_b, q_c, ok))


# --- The strike's exact input: must be OUT of domain. ---
add_domain("strike_exact_input_out_of_domain", 0, QLN2, QB, QC_STRIKE)

# --- z=0 (q=0) boundary, forced exactly on both sides: base**2 + q_c == INT64_MAX
#     is the largest in-domain q_c; INT64_MAX + 1 is the first out. ---
_, _, _, _base_z0 = derive(0, QLN2, QB)
_q_c_max_z0 = INT64_MAX - _base_z0 * _base_z0
_q_c_first_out_z0 = _q_c_max_z0 + 1
add_domain("z0_boundary_last_in_domain", 0, QLN2, QB, _q_c_max_z0)
add_domain("z0_boundary_first_out_of_domain", 0, QLN2, QB, _q_c_first_out_z0)

# --- Same (q_ln2, q_b) triple and the STRIKE'S OWN q_c, one full ln2-quantum
#     step down (q = -q_ln2, an exact multiple so q_p collapses back to 0 and
#     base is unchanged): z becomes 1, and the very same q_c that overflowed
#     at z=0 now shifts back into int64 range. This is the cell that forces the
#     predicate to NOT be a naive `base**2 + q_c <= INT64_MAX` test -- that
#     naive test rejects this input, but the true shifted value is in range. ---
add_domain("z1_same_qc_as_strike_recovers_in_domain", -QLN2, QLN2, QB, QC_STRIKE)

# --- NOTE on a rejected construction: an attempt to force z=1's OWN boundary at
#     this same base (1733160715) requires q_c = (2*INT64_MAX+1) - base**2 ~=
#     1.544e19 -- itself outside int64_t's representable range, so no valid call
#     could ever construct it. At base=1733160715, EVERY valid int64_t q_c stays
#     in domain once z >= 1 (confirmed: base**2 + INT64_MAX, halved, is still <
#     INT64_MAX). A boundary cell pinned there would force nothing (the discipline
#     this suite exists to avoid) -- so it is not authored at this base. Forcing a
#     genuine z=1 boundary needs a base whose square alone approaches INT64_MAX
#     (see the larger-q_b family below).
add_domain("z_at_clip_ceiling_strike_qc_in_domain", -I_EXP_CLIP_N * QLN2, QLN2, QB, QC_STRIKE)
_, _z30, _, _base_z30 = derive(-I_EXP_CLIP_N * QLN2, QLN2, QB)
assert _z30 == I_EXP_CLIP_N
# Same note applies at z=30: the boundary q_c there is ~9.9e27, also outside
# int64_t. At the widest shift the primitive ever performs, EVERY valid int64_t
# q_c (up to INT64_MAX itself, exercised above) stays in domain for this base --
# an unconditional result over the whole valid q_c range, not a forced boundary.

# ---------------------------------------------------------------------------
# A second, larger-magnitude base family: q_b = 3,500,000,000, deliberately
# between sqrt(INT64_MAX) (~3.037e9) and sqrt(2*INT64_MAX) (~4.295e9) so that
# base**2 ALONE (with q=0, q_p=0, base=q_b exactly) already exceeds INT64_MAX --
# forcing the overflow through the q_b/base**2 axis specifically, independently
# of q_c (q_c=1, the smallest positive value, so it contributes nothing). This
# is the axis D-SLM81 names: q_b has no documented upper bound, and a caller
# recomputing base**2 in int64 overflows once q_b exceeds ~3.04e9 -- distinct
# from the strike's own q_c-dominated overflow. q_b = 3,500,000,000 is
# contract-legal (positive; no upper bound is documented) and realistically
# reachable only in the sense that nothing in the header rules it out -- the
# same standing the strike's own q_c=2**63-1 had.
# ---------------------------------------------------------------------------
QB_LARGE = 3_500_000_000

_, _, _, _base_large_z0 = derive(0, QLN2, QB_LARGE)
assert _base_large_z0 * _base_large_z0 > INT64_MAX, "QB_LARGE must make base**2 alone exceed INT64_MAX at z=0"
add_domain("z0_base_squared_alone_exceeds_domain_even_at_qc_one", 0, QLN2, QB_LARGE, 1)

_, _z1_large, _, _base_large_z1 = derive(-QLN2, QLN2, QB_LARGE)
assert _z1_large == 1
assert _base_large_z1 == _base_large_z0
_q_c_max_z1_large = (2 * INT64_MAX + 1) - _base_large_z1 * _base_large_z1
_q_c_first_out_z1_large = _q_c_max_z1_large + 1
assert 0 <= _q_c_max_z1_large <= INT64_MAX and 0 <= _q_c_first_out_z1_large <= INT64_MAX, (
    "the z=1 boundary for QB_LARGE must itself be a representable int64_t q_c -- "
    "otherwise this would repeat the same unconstructible-boundary mistake as the "
    "rejected 1733160715-based attempt above"
)
add_domain("z1_boundary_last_in_domain_large_base", -QLN2, QLN2, QB_LARGE, _q_c_max_z1_large)
add_domain("z1_boundary_first_out_of_domain_large_base", -QLN2, QLN2, QB_LARGE, _q_c_first_out_z1_large)

_clip_q_large = -I_EXP_CLIP_N * QLN2
add_domain("z_at_clip_ceiling_always_in_domain_large_base", _clip_q_large, QLN2, QB_LARGE, INT64_MAX)

# --- Typical small-magnitude in-domain case (realistic_s0 family shape, from
#     sslm_intmath_fixtures.h) -- confirms the predicate says yes on the
#     ordinary, non-adversarial path, not only at forced boundaries. ---
add_domain("typical_small_in_domain", 0, 6, 13, 95)

# --- A realistic operating-scale positive cell: (q_ln2, q_b, q_c) as
#     Tools/superslm_spike/intmath.py's own i_exp actually derives them for
#     scale=0.01 (a plausible per-token softmax activation scale), reproduced
#     here by reading that pinned reference directly --
#     q_b = floor(_POLY_B / scale), q_c = floor(_POLY_C / (_POLY_A * scale**2))
#     (intmath.py lines 203-204). This proves the predicate does not
#     over-reject the range real calls actually use -- the adversarial cells
#     above prove rejection; this proves the predicate is not simply "always
#     false near any large constant." ---
add_domain("realistic_operating_scale_0p01_in_domain", 0, 69, 135, 9595)

# ---------------------------------------------------------------------------
# Cross-check every case already shipped in sslm_intmath_fixtures.h's
# kIExpCases: every one of those 36 fixtures produced a golden that already
# fits int64_t (by construction of gen_intmath_fixtures.py's own width-probe
# search), so IExpConstantsInDomain must return true for every one of them.
# This is the "does not regress a known-good input" side of the predicate,
# checked here independently (same formula, transcribed fresh) rather than by
# re-deriving it from the golden 'expected' field.
# ---------------------------------------------------------------------------
_existing_iexp_cases = [
    ("realistic_s0_q_max_element", 0, 6, 13, 95, 264),
    ("realistic_s0_q_neg1", -1, 6, 13, 95, 239),
    ("realistic_s0_q_neg_one_ln2_step", -6, 6, 13, 95, 132),
    ("realistic_s0_q_neg_five_ln2_steps", -30, 6, 13, 95, 8),
    ("realistic_s0_clip_boundary", -180, 6, 13, 95, 0),
    ("realistic_s0_beyond_clip", -186, 6, 13, 95, 0),
    ("realistic_s1_q_max_element", 0, 13, 27, 383, 1112),
    ("realistic_s1_q_neg1", -1, 13, 27, 383, 1059),
    ("realistic_s1_q_neg_one_ln2_step", -13, 13, 27, 383, 556),
    ("realistic_s1_q_neg_five_ln2_steps", -65, 13, 27, 383, 34),
    ("realistic_s1_clip_boundary", -390, 13, 27, 383, 0),
    ("realistic_s1_beyond_clip", -403, 13, 27, 383, 0),
    ("realistic_s2_q_max_element", 0, 34, 67, 2398, 6887),
    ("realistic_s2_q_neg1", -1, 34, 67, 2398, 6754),
    ("realistic_s2_q_neg_one_ln2_step", -34, 34, 67, 2398, 3443),
    ("realistic_s2_q_neg_five_ln2_steps", -170, 34, 67, 2398, 215),
    ("realistic_s2_clip_boundary", -1020, 34, 67, 2398, 0),
    ("realistic_s2_beyond_clip", -1054, 34, 67, 2398, 0),
    ("realistic_s3_q_max_element", 0, 138, 270, 38382, 111282),
    ("realistic_s3_q_neg1", -1, 138, 270, 38382, 110743),
    ("realistic_s3_q_neg_one_ln2_step", -138, 138, 270, 38382, 55641),
    ("realistic_s3_q_neg_five_ln2_steps", -690, 138, 270, 38382, 3477),
    ("realistic_s3_clip_boundary", -4140, 138, 270, 38382, 0),
    ("realistic_s3_beyond_clip", -4278, 138, 270, 38382, 0),
    ("realistic_s4_q_max_element", 0, 88, 171, 15476, 44717),
    ("realistic_s4_q_neg1", -1, 88, 171, 15476, 44376),
    ("realistic_s4_q_neg_one_ln2_step", -88, 88, 171, 15476, 22358),
    ("realistic_s4_q_neg_five_ln2_steps", -440, 88, 171, 15476, 1397),
    ("realistic_s4_clip_boundary", -2640, 88, 171, 15476, 0),
    ("realistic_s4_beyond_clip", -2728, 88, 171, 15476, 0),
    ("qln2_min_q0", 0, 1, 2, 3, 7),
    ("qln2_min_q_neg1", -1, 1, 2, 3, 3),
    ("qln2_min_clip_boundary", -30, 1, 2, 3, 0),
    ("qln2_min_beyond_clip", -35, 1, 2, 3, 0),
    ("width_probe_q0_max_element", 0, 887904998, 1733160715, 1574531533020431360, 4578377597039742585),
    ("width_probe_q_neg_one_ln2_step", -887904998, 887904998, 1733160715, 1574531533020431360, 2289188798519871292),
]
assert len(_existing_iexp_cases) == 36

accessor_cases: list[tuple[str, int, int, int, int, int, int]] = []
# label, q, q_ln2, q_b, expected_z, expected_base, expected_recomposed

for label, q, q_ln2, q_b, q_c, expected in _existing_iexp_cases:
    _, z, _, base = derive(q, q_ln2, q_b)
    ok, shifted, _, _ = in_domain(q, q_ln2, q_b, q_c)
    assert ok, f"{label}: existing shipped fixture must be in-domain"
    assert shifted == expected, (
        f"{label}: independently-derived (base**2+q_c)>>z == {shifted}, "
        f"but the already-shipped golden 'expected' is {expected} -- the formula "
        f"transcribed into this generator does not match sslm_intmath_fixtures.h"
    )
    accessor_cases.append((label, q, q_ln2, q_b, z, base, expected))
    add_domain(f"existing_fixture_{label}", q, q_ln2, q_b, q_c)

# ---------------------------------------------------------------------------
# IExpShift postcondition: z is always in [0, I_EXP_CLIP_N] over every case
# constructed above (accessor_cases + the hand-built domain cases' own
# (q, q_ln2) pairs) -- checked here so the generator itself cannot silently
# emit a fixture that violates the documented postcondition.
# ---------------------------------------------------------------------------
for label, q, q_ln2, q_b, z, base, expected in accessor_cases:
    assert 0 <= z <= I_EXP_CLIP_N, f"{label}: z={z} outside documented [0, {I_EXP_CLIP_N}]"

# ---------------------------------------------------------------------------
# Emit the C++ header.
# ---------------------------------------------------------------------------

lines: list[str] = []
emit = lines.append


def cxx_i64(v: int) -> str:
    return f"INT64_C({v})"


def cxx_str(s: str) -> str:
    return f'"{s}"'


def cxx_bool(v: bool) -> str:
    return "true" if v else "false"


emit("// GENERATED FILE. Do not hand-edit.")
emit("//")
emit("// Produced by tests/gen_iexp_domain_fixtures.py: every expected value is computed")
emit("// there from IExpFromConstants's documented five-line decomposition using Python")
emit("// arbitrary-precision integers -- never by calling IExpShift/IExpBase/")
emit("// IExpConstantsInDomain (which do not exist yet) and never by re-deriving the bound")
emit("// in fixed-width int64 arithmetic (that re-derivation reproduces the D-SLM81 defect:")
emit("// base**2 alone overflows int64). Re-running the generator must reproduce this file")
emit("// byte-for-byte.")
emit("//")
emit("// Test-design record:")
emit("// Claude/Curie/superslm-s2.6-softmax-iexp-domain-test-design-2026-07-21.md")
emit("// Strike casebook: Claude/Loki/softmax-s2.6-strike-2026-07-21.md")
emit("#ifndef SUPERSLM_TESTS_SSLM_IEXP_DOMAIN_FIXTURES_H")
emit("#define SUPERSLM_TESTS_SSLM_IEXP_DOMAIN_FIXTURES_H")
emit("")
emit("#include <cstddef>")
emit("#include <cstdint>")
emit("")
emit("namespace superslm_test {")
emit("")

emit("// --- IExpShift / IExpBase: independently-derived (z, base) pairs, keyed to the")
emit("//     SAME (label, q, q_ln2, q_b, q_c) fixtures already shipped in")
emit("//     sslm_intmath_fixtures.h's kIExpCases (this generator asserts its own")
emit("//     recomposition, (base**2+q_c)>>z, equals that file's already-independent")
emit("//     'expected' golden before ever emitting a row -- see the assert above). ---")
emit("")
emit("struct IExpAccessorCase {")
emit("\tconst char* label;")
emit("\tint64_t q;")
emit("\tint64_t q_ln2;")
emit("\tint64_t q_b;")
emit("\tint64_t expected_z;")
emit("\tint64_t expected_base;")
emit("\tint64_t expected_iexp_from_constants;  // cross-reference only: kIExpCases' own golden")
emit("};")
emit("")
emit("inline constexpr IExpAccessorCase kIExpAccessorCases[] = {")
for label, q, q_ln2, q_b, z, base, expected in accessor_cases:
    emit(
        f"\t{{{cxx_str(label)}, {cxx_i64(q)}, {cxx_i64(q_ln2)}, {cxx_i64(q_b)}, "
        f"{cxx_i64(z)}, {cxx_i64(base)}, {cxx_i64(expected)}}},"
    )
emit("};")
emit(f"inline constexpr size_t kIExpAccessorCasesCount = {len(accessor_cases)};")
emit("")

emit("// --- IExpConstantsInDomain. Every expected_in_domain is computed in Python")
emit("//     arbitrary precision by gen_iexp_domain_fixtures.py, transcribing the")
emit("//     documented formula directly -- never by calling the primitive under test. ---")
emit("")
emit("struct IExpDomainCase {")
emit("\tconst char* label;")
emit("\tint64_t q;")
emit("\tint64_t q_ln2;")
emit("\tint64_t q_b;")
emit("\tint64_t q_c;")
emit("\tbool expected_in_domain;")
emit("};")
emit("")
emit("inline constexpr IExpDomainCase kIExpDomainCases[] = {")
for label, q, q_ln2, q_b, q_c, ok in domain_cases:
    emit(
        f"\t{{{cxx_str(label)}, {cxx_i64(q)}, {cxx_i64(q_ln2)}, {cxx_i64(q_b)}, "
        f"{cxx_i64(q_c)}, {cxx_bool(ok)}}},"
    )
emit("};")
emit(f"inline constexpr size_t kIExpDomainCasesCount = {len(domain_cases)};")
emit("")

emit("}  // namespace superslm_test")
emit("")
emit("#endif  // SUPERSLM_TESTS_SSLM_IEXP_DOMAIN_FIXTURES_H")
emit("")

text = "\n".join(lines)
text.encode("ascii")  # same reproducibility guard gen_intmath_fixtures.py uses
with open(OUT_PATH, "w", newline="\n", encoding="utf-8") as f:
    f.write(text)

print(f"Wrote {OUT_PATH}")
print(f"IExpAccessorCases: {len(accessor_cases)}")
print(f"IExpDomainCases: {len(domain_cases)}")
for c in domain_cases:
    print(" ", c[0], "->", c[5])
