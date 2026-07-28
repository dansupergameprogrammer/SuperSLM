#!/usr/bin/env python3
"""Generates tests/sslm_iexp_domain_fixtures.h -- Curie's S2.6-amendment red suite for
IExpConstruct / IExpEvaluate / IExpConstantsInDomain (D-SLM78/79/81; Claude/Loki/
softmax-s2.6-strike-2026-07-21.md), AND (from the `kIExpConstructCases` table down)
Curie's S-HARDEN-0 population suite for the checked `IExpConstruct`/`IExpEvaluate` entry
point (F9, F21, and Poirot's a1d7986 code-review Finding 1) -- against the FINAL API
(S-HARDEN-0, commits 78db1a7/cf2e426). This file serves two test-design records; see
each section's own docstring for which.

Every expected value is computed HERE, directly from the five-line decomposition
IExpConstruct/IExpEvaluate document in src/intmath.cpp / include/superslm/intmath.h --

    clip_lo = -I_EXP_CLIP_N * q_ln2
    clipped = max(q, clip_lo)
    z       = (-clipped) // q_ln2          (floor; clipped <= 0, q_ln2 >= 1)
    q_p     = clipped + z * q_ln2
    base    = q_p + q_b
    in_domain(q_c) iff q_ln2 <= INT64_MAX // I_EXP_CLIP_N
                       AND INT64_MIN <= (base**2 + q_c) >> z <= INT64_MAX

The first clause is `IExpConstantsInDomain`'s own first line (src/intmath.cpp:379,
`if (q_ln2 > kIExpMaxQLn2) return false;`, added to close Poirot's `7b668b2` review
finding 2): above that ceiling the clip bound `-I_EXP_CLIP_N * q_ln2` overflows int64
in C++ and the five-line decomposition below is meaningless, so the oracle rejects on
that clause alone, before evaluating the decomposition, exactly as the implementation
does -- rather than computing the decomposition anyway in Python's unbounded integers
(which would silently disagree with the implementation about which inputs are in
domain).

-- using Python's arbitrary-precision integers, NEVER by calling the C++ primitives
under test and never by re-deriving the bound in fixed-width int64 arithmetic (that
re-derivation is exactly the defect D-SLM81 names: `base**2` alone overflows int64
once q_b exceeds ~3.04e9). This script is therefore an independent oracle in the
same sense gen_intmath_fixtures.py's calls into the pinned Python reference are: the
formula is transcribed from the documented contract, never recomputed from the
C++ implementation under test.

Re-running this script must reproduce sslm_iexp_domain_fixtures.h byte-for-byte.

Test-design records:
  S2.6-amendment section (kIExpAccessorCases / kIExpDomainCases):
    Claude/Curie/superslm-s2.6-softmax-iexp-domain-test-design-2026-07-21.md
  S-HARDEN-0 section (kIExpConstructCases):
    Claude/Curie/superslm-s-harden-0-test-design-2026-07-21.md
"""

from __future__ import annotations

import os

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
I_EXP_CLIP_N = 30  # must equal superslm::I_EXP_CLIP_N (include/superslm/intmath.h)

# Matches src/intmath.cpp:362, kIExpMaxQLn2 = INT64_MAX / I_EXP_CLIP_N -- both operands
# positive, so C++ truncating division and Python floor division agree. Above this
# ceiling, IExpConstantsInDomain's first line rejects before evaluating the
# decomposition (Poirot's 7b668b2 review, finding 2; closed at a0e4850).
kIExpMaxQLn2 = INT64_MAX // I_EXP_CLIP_N

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sslm_iexp_domain_fixtures.h")


def wrap_to_int64(v: int) -> int:
    """Narrows an arbitrary-precision integer to the low 64 bits, reinterpreted as signed --
    the same operation IExpEvaluate's S128-to-int64 narrowing shift performs on a
    kNotRepresentable construction (src/intmath.cpp: `SShrToI64`). Confirmed against the
    committed golden below (D-SLM78's strike input): wrap_to_int64(1733160715**2 +
    (2**63-1)) == -6219525972835464584, the exact pinned value
    TestIExpConstructAndEvaluateProducesKnownWrappedValueForOutOfDomainConstants asserts.
    The true 128-bit-wide
    intermediate (base**2 + q_c, before the >> z narrowing) never exceeds roughly 2**93 for
    any int64 base/q_c/z this primitive can ever form (base**2 < 2**126 at worst, but the
    documented contract only reaches this function via IExpConstruct, whose base is itself
    bounded by int64 arithmetic that does not overflow -- see IExpConstruct's own comments),
    so Python's arbitrary-precision arithmetic followed by this mod-2**64 step reproduces the
    identical low-64-bits result the real S128 shift produces, without re-deriving the shift
    itself in fixed width."""
    m = v % (1 << 64)
    return m - (1 << 64) if m >= (1 << 63) else m


def derive(q: int, q_ln2: int, q_b: int) -> tuple[int, int, int, int]:
    """Returns (clipped, z, q_p, base) -- the documented decomposition, exactly."""
    assert q_ln2 >= 1
    clip_lo = -I_EXP_CLIP_N * q_ln2
    clipped = clip_lo if q < clip_lo else q
    z = (-clipped) // q_ln2
    q_p = clipped + z * q_ln2
    base = q_p + q_b
    return clipped, z, q_p, base


def in_domain(q: int, q_ln2: int, q_b: int, q_c: int) -> tuple[bool, int | None, int | None, int | None]:
    """Returns (is_in_domain, exact_shifted_value, z, base). exact_shifted_value is the
    true mathematical (base**2 + q_c) >> z, unbounded -- NOT narrowed to int64. base**2
    is always >= 0, but q_c is accepted here as a full int64_t (IExpConstantsInDomain's
    own contract validates only q<=0/q_ln2>=1 -- see Poirot's 7b668b2 review, finding 3 --
    coefficient positivity is a property of how C30/offline derive q_b/q_c, not an
    asserted precondition of this predicate), so val can be negative. Python's `>>` on an
    arbitrary-precision int is already an arithmetic (floor) shift for negative operands,
    the same convention SShrToI64 implements in C++, so no non-negativity assumption is
    needed for this formula to apply.

    Above kIExpMaxQLn2, `IExpConstantsInDomain` (src/intmath.cpp:379) rejects before ever
    computing the decomposition -- the clip bound `-I_EXP_CLIP_N * q_ln2` would overflow
    int64 there, so the decomposition is meaningless in the implementation even though
    Python's unbounded integers can compute it without overflowing. This oracle mirrors
    that short-circuit exactly: q_ln2, z, and base are not meaningful for the rejected
    region and are returned as None rather than a value the implementation never reaches."""
    if q_ln2 > kIExpMaxQLn2:
        return False, None, None, None
    _, z, _, base = derive(q, q_ln2, q_b)
    val = base * base + q_c
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
# The q_ln2 CEILING (Poirot's 7b668b2 review, finding 2; closed at a0e4850 by
# `if (q_ln2 > kIExpMaxQLn2) return false;`, src/intmath.cpp:379). Above
# kIExpMaxQLn2 the clip bound `-I_EXP_CLIP_N * q_ln2` overflows int64 in the
# implementation and the decomposition is meaningless -- before the guard
# existed, the predicate answered a false all-clear (`true`) at contract-legal
# q_ln2 the guard now rejects outright. This axis is independent of q_b/q_c
# (q_b=1, q_c=0, the review's own witness constants, isolate it).
#
# The generator's own `in_domain` mirrors the guard's short-circuit (added
# above): without that clause, Python's unbounded integers would compute the
# decomposition anyway and could disagree with the implementation about which
# side of the ceiling an input falls on. The cells below are only meaningful
# because the oracle now implements the same first-line rejection the code
# does -- this is the fixture-side half of closing N2; the second half is
# that this guard's own removal (not merely a shifted value) must make one of
# these cells fail, which is verified as a mutation at review/audit time and
# is not repeatable from this generator alone.
# ---------------------------------------------------------------------------
QB_CEILING_WITNESS = 1  # Poirot's 7b668b2/e6db8ea witness constants (q_b=1, q_c=0)
QC_CEILING_WITNESS = 0

add_domain("q_ln2_ceiling_last_in_domain", 0, kIExpMaxQLn2, QB_CEILING_WITNESS, QC_CEILING_WITNESS)
add_domain("q_ln2_ceiling_first_out_of_domain", 0, kIExpMaxQLn2 + 1, QB_CEILING_WITNESS, QC_CEILING_WITNESS)

# These two are the ones that actually force the guard: at the boundary cell
# above, the overflowing decomposition happens to also land out of int64
# range (z=-29), so a version of the predicate with the guard deleted would
# still answer false there BY COINCIDENCE and the cell would not detect the
# guard's removal. Further above the ceiling the overflow wraps differently
# and the pre-guard predicate answered true (a false all-clear) -- executed
# and recorded in Claude/Poirot/7b668b2-s2.2-iexp-domain-amendment-review-
# 2026-07-21.md sec. 3 and Claude/Poirot/e6db8ea-s2.2-iexp-amendment-fix-
# round-review-2026-07-21.md sec. 5.1. These are the cells a guard-removal
# mutation must fail.
add_domain("q_ln2_beyond_ceiling_2pow60_was_false_all_clear", 0, 1 << 60, QB_CEILING_WITNESS, QC_CEILING_WITNESS)
add_domain("q_ln2_beyond_ceiling_int64_max_was_false_all_clear", 0, INT64_MAX, QB_CEILING_WITNESS, QC_CEILING_WITNESS)

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

# ---------------------------------------------------------------------------
# The predicate's LOWER bound (Poirot's 7b668b2 review, finding 3): replacing
# `lower` with 0, or deleting the lower check entirely, left the suite green
# at 7435/0 -- this generator's own `in_domain` used to assert val >= 0,
# which made a negative-q_c fixture impossible to construct here in the first
# place. IExpConstantsInDomain's own documented preconditions are only q<=0
# and q_ln2>=1 (include/superslm/intmath.h: "this predicate answers the width
# question only, and does not validate those" -- it does not validate q_b/q_c's
# sign); coefficient positivity is a property of how C30/offline derive
# q_b/q_c (SuperSLM_Plan.md C7, "second-pass N2-5"), not an asserted
# precondition of this function. A negative q_c is therefore a legitimate
# call this predicate must still answer correctly, and this generator's own
# `assert val >= 0` was the reason no cell here could ever construct one.
#
# base**2 >= 0 always, and q_c >= INT64_MIN by its own type, so the smallest
# value the predicate can ever see is base**2 + INT64_MIN >= INT64_MIN (at
# base=0). The predicate's own lower bound at z=0 is exactly -2**63 ==
# INT64_MIN, and at z>0 it is -2**63 * 2**z -- strictly MORE negative. So the
# predicate's minimum reachable value (INT64_MIN) sits AT the z=0 lower bound
# and strictly ABOVE it for every z>0: no int64_t (base, q_c) pair can ever
# push the predicate's lower-bound check to reject, for any z. That is a
# consequence of q_c's own type, independent of whether it happens to be
# contract-typical (positive) or not.
#
# The three cells below sit at that exact, tightest reachable point (base=0,
# q_c=INT64_MIN, so val == INT64_MIN exactly) at z=0, z=1, and z=I_EXP_CLIP_N
# -- confirmed IN domain at every one. This forces the predicate's
# lower-bound comparison to be evaluated at genuine equality (z=0) and with
# growing slack (z=1, z=30) rather than never being reached, and catches a
# mutation that replaces the lower bound with a wrong value (e.g. 0): such a
# mutation answers OUT-of-domain at all three, where the real predicate
# (correctly) does not.
#
# Confirmed against a standalone compiled reproduction of the MSVC
# struct-S128 path (scratch-only, not part of this repository, deleted after
# use): the real predicate returns true at these cells and at every z in
# [0, 30] built the same way; a `lower = 0` mutation returns false at all of
# them (caught); a mutation that deletes the lower check entirely returns
# true at all of them, UNCHANGED from the real predicate -- because the
# check is never the reason a valid int64_t call is accepted or rejected.
# No cell, however constructed, can force that deletion to differ from the
# correct implementation; the cells below are the strongest assertion this
# predicate's lower bound admits. Routed to the builder/planner as a finding:
# the lower-bound branch is provably unreachable-as-false for any int64_t
# input, not merely untested by this suite's prior fixtures.
# ---------------------------------------------------------------------------
_, _z0_lb_check, _, _base0_lb_check = derive(0, 1, 0)
assert _z0_lb_check == 0 and _base0_lb_check == 0
add_domain("z0_lower_bound_equality_qc_min", 0, 1, 0, INT64_MIN)

_, _z1_lb_check, _, _base1_lb_check = derive(-7, 7, 0)
assert _z1_lb_check == 1 and _base1_lb_check == 0
add_domain("z1_lower_bound_slack_qc_min", -7, 7, 0, INT64_MIN)

_, _z30_lb_check, _, _base30_lb_check = derive(-I_EXP_CLIP_N * 7, 7, 0)
assert _z30_lb_check == I_EXP_CLIP_N and _base30_lb_check == 0
add_domain("z_at_clip_ceiling_lower_bound_slack_qc_min", -I_EXP_CLIP_N * 7, 7, 0, INT64_MIN)

# --- Typical small-magnitude in-domain case (realistic_s0 family shape, from
#     sslm_intmath_fixtures.h) -- confirms the predicate says yes on the
#     ordinary, non-adversarial path, not only at forced boundaries. ---
add_domain("typical_small_in_domain", 0, 6, 13, 95)

# --- A realistic operating-scale positive cell: (q_ln2, q_b, q_c) as
#     Tools/superslm_spike/intmath.py's own i_exp actually derives them for
#     scale=0.01 (a plausible per-token softmax activation scale) --
#     q_b = floor(_POLY_B / scale), q_c = floor(_POLY_C / (_POLY_A * scale**2))
#     (intmath.py lines 203-204). This script imports nothing outside the
#     standard library (see the module docstring), so the three values below
#     are computed from that formula by hand and hardcoded here as literal
#     integers, not read from the reference at generation time; they were
#     checked against Tools/superslm_spike/intmath.py directly at scale=0.01
#     (q_ln2=69, q_b=135, q_c=9595) before being hardcoded. This proves the
#     predicate does not over-reject the range real calls actually use -- the
#     adversarial cells above prove rejection; this proves the predicate is
#     not simply "always false near any large constant." ---
add_domain("realistic_operating_scale_0p01_in_domain", 0, 69, 135, 9595)

# ---------------------------------------------------------------------------
# The one-call-per-triple SHORTCUT CONDITION (IExpConstantsInDomain's cost
# note, include/superslm/intmath.h -- corrected at c33843d after two prior
# false versions of this paragraph; Claude/Poirot/93622d3-s2.2-iexp-amendment-
# close-round-review-2026-07-21.md, finding N7). The header claims the
# shortcut -- discharging an entire row with a single
# IExpConstantsInDomain(q=0, ...) call -- holds IFF `2*q_b >= q_ln2 - 1`.
# Over a row (z=0, q_p ranging across (-q_ln2, 0]) the two candidate worst
# points are q_p=0 (q=0, base=q_b) and q_p=-(q_ln2-1) (q=-(q_ln2-1),
# base=q_b-q_ln2+1); the claim is exactly that the first end's |base| is
# `>=` the second's.
#
# Four scenarios at q_ln2 = 3,000,000,000, each with q_c placed via
# INT64_MAX - dominant_base**2 so one end sits exactly at its own in-domain
# boundary -- forcing the OTHER end's status to be the genuine, executed
# evidence for whether the header's claim is right at that q_b, not an
# assumption:
#
#   A -- q_b = 1,800,000,000 (condition HOLDS, interior of the region):
#        q_c placed at the far end's own boundary; q=0 -- the dominant end
#        under the claim -- then answers FALSE (the row's true, tighter
#        answer) while the far end answers TRUE. This is the exact witness
#        Poirot's N7 review executed (93622d3-....md sec. 5.4): discharging
#        at q=0 is sound here, and checking only the far end would be a
#        false all-clear.
#   B -- q_b = 1,000,000,000 (condition FAILS, interior of the region):
#        q_c placed at q=0's own boundary; q=0 answers TRUE while the far
#        end -- the row's true worst point when the condition fails --
#        answers FALSE. Discharging at q=0 alone is UNSOUND here.
#   C -- q_b = 1,500,000,000, the LEAST q_b for which the condition holds at
#        this q_ln2 (2*q_b == q_ln2 - 1 + 1; confirmed by binary search
#        against the shipped IExpBase, same review). q_c placed at q=0's own
#        boundary; both ends answer TRUE -- the satisfying side of the
#        boundary, forced exactly.
#   D -- q_b = 1,499,999,999, one less than C: the condition fails by the
#        single integer separating it from C. Same q_c construction; q=0
#        answers TRUE, the far end answers FALSE -- the failing side of the
#        SAME boundary, forced exactly, one q_b away from C.
#
# These eight points are also swept generically by
# TestIExpConstantsInDomainAcrossCorpus below. The dedicated cross-check that
# ties them to the header's claimed condition --  and that fails if the
# claimed condition is replaced by the wrong, stronger one this amendment
# removed (`q_b >= q_ln2`) -- is
# TestIExpConstantsInDomainShortcutConditionMatchesHeaderClaim in
# test_main.cpp, which looks these exact labels up by name.
# ---------------------------------------------------------------------------
QLN2_SHORTCUT = 3_000_000_000
_far_q_shortcut = -(QLN2_SHORTCUT - 1)


def _shortcut_witness(label: str, q_b: int, qc_from_q0: bool) -> None:
    base0 = q_b
    base_far = q_b - QLN2_SHORTCUT + 1
    dominant_base = base0 if qc_from_q0 else base_far
    q_c = INT64_MAX - dominant_base * dominant_base
    assert 0 <= q_c <= INT64_MAX, f"{label}: constructed q_c must itself be a valid int64_t"
    add_domain(f"shortcut_{label}_q0", 0, QLN2_SHORTCUT, q_b, q_c)
    add_domain(f"shortcut_{label}_far_end", _far_q_shortcut, QLN2_SHORTCUT, q_b, q_c)


assert 2 * 1_800_000_000 >= QLN2_SHORTCUT - 1, "scenario A must satisfy the header's condition"
_shortcut_witness("holds_interior_A", 1_800_000_000, qc_from_q0=False)

assert not (2 * 1_000_000_000 >= QLN2_SHORTCUT - 1), "scenario B must fail the header's condition"
_shortcut_witness("fails_interior_B", 1_000_000_000, qc_from_q0=True)

assert 2 * 1_500_000_000 >= QLN2_SHORTCUT - 1, "scenario C (least satisfying q_b) must satisfy the condition"
assert not (2 * 1_499_999_999 >= QLN2_SHORTCUT - 1), "one less than C must fail the condition"
_shortcut_witness("boundary_holds_C", 1_500_000_000, qc_from_q0=True)
_shortcut_witness("boundary_fails_D", 1_499_999_999, qc_from_q0=True)

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
# IExpConstruct's z postcondition: always in [0, I_EXP_CLIP_N] over accessor_cases
# -- checked here so the generator itself cannot silently emit an accessor
# fixture that violates the documented postcondition.
#
# This sweeps accessor_cases only, deliberately not domain_cases. The
# postcondition itself is conditional on q_ln2 <= kIExpMaxQLn2 (the header,
# include/superslm/intmath.h:187: "z lies in [0, I_EXP_CLIP_N] only while
# q_ln2 <= INT64_MAX / I_EXP_CLIP_N"), and domain_cases now deliberately
# includes q_ln2 above that ceiling (the q_ln2_ceiling_* and
# q_ln2_beyond_ceiling_* cells above) specifically to force
# IExpConstantsInDomain's rejection of that region. Calling derive() on those
# rows to extract a z would either divide by the same overflow the cells
# exist to name, or (for the cells kept in Python's unbounded arithmetic
# on purpose) compute a z the postcondition never claimed to bound in the
# first place -- so widening this loop to include domain_cases would assert
# a postcondition against inputs the header excludes from it, not close a
# real gap. accessor_cases carries no such input (every accessor fixture is a
# kIExpCases row already proven in range), so this sweep is exactly the set
# for which the postcondition is a claim at all.
# ---------------------------------------------------------------------------
for label, q, q_ln2, q_b, z, base, expected in accessor_cases:
    assert 0 <= z <= I_EXP_CLIP_N, f"{label}: z={z} outside documented [0, {I_EXP_CLIP_N}]"

# ---------------------------------------------------------------------------
# RETIRED 2026-07-22 (this session, porting to the final API, D-SLM API commit
# 78db1a7/cf2e426): this file used to emit kIExpGuardOrderCases here -- a
# crash-probe population over the OLD two-function API (IExpFromConstants /
# IExpConstantsInDomain), distinguishing an "eval" call path from a "pred" call
# path because F9/F21's whole finding was that those two paths consulted the
# guard at different points. The final API collapses that distinction
# structurally: there is exactly ONE function that takes raw (q, q_ln2, q_b, q_c)
# -- IExpConstruct -- and it is TOTAL (asserts nothing, executes no UB on any
# input, in any build configuration), so the crash-probe subprocess isolation
# this population needed (because the OLD code could crash the whole process)
# has no subject left to isolate, and the "eval"/"pred" split has no second call
# path left to distinguish. Every witness value this population carried is
# ported, not dropped -- see kIExpConstructCases below: F9's four ceiling
# witnesses are badqln2_last_valid_ceiling / badqln2_first_invalid_ceiling /
# badqln2_interior_invalid_ceiling / badqln2_f9_witness_int64_max; F21's four
# q_b witnesses are badqb_f21_witness_qb_int64_min / badqb_boundary_first_unsafe
# / badqb_interior_unsafe / badqb_representable_boundary_square_not_representable
# (the last of these is a corrected label -- see its own comment on why the old
# "boundary_last_safe" name was accurate about q_p+q_b representability but not
# about overall domain membership). See test_main.cpp's comment at the deleted
# TestIExpGuardOrderCasesNeverExecuteUBAndAgreeWithIndependentOracle call site
# for the full account. Test-design record (superseded section):
# Claude/Curie/superslm-s-harden-0-test-design-2026-07-21.md.

# ---------------------------------------------------------------------------
# S-HARDEN-0 LAYER B: kIExpConstructCases, for the NEW `IExpConstruct` entry point
# the build lands (SuperSLM_Plan.md's S-HARDEN-0 sub-slot). References a type/function
# that does NOT EXIST at f078403 -- the emitted C++ struct and the test cells that
# consume it do not compile until Brunel builds `IExpConstruct` / `IExpConstruction` /
# `IExpDomain`. This is the documented, correct state for Layer B (see the test-design
# record); it is authored now so the red suite is complete and ready the moment the
# API lands, not invented at build time by whoever implements it.
#
# REVISED 2026-07-21 (Brunel, mid-build): the entry point returns a five-way
# `IExpDomain` enum, not a bool -- the wrapped-value cell's
# existing, unedited golden pins a DEFINED (narrowed, "meaningless" but not UB)
# wrapped value for one specific input, so the entry point must distinguish "the
# decomposition is well-formed but the final result does not fit int64"
# (kNotRepresentable, *out IS filled) from "the decomposition itself could not be
# formed" (kBadQ / kBadQLn2 / kBadQB, *out is left untouched). This resolves what the
# prior (bool) revision of this section called out as unresolvable: under the bool
# contract, an R4 (q_b) violation and an R5 (final-representability) violation were
# indistinguishable by return value, because both collapsed to `false`. Under the
# enum they are two different, independently observable outcomes, and R4/R5 are now
# each given their own witnesses below rather than R4 merely reusing F21's inputs to
# prove absence-of-crash alone.
#
# Five outcomes, per IExpConstruct's contract (validated before any arithmetic that
# could overflow): kOk (decomposition well-formed, result representable), kBadQ
# (q > 0), kBadQLn2 (q_ln2 < 1, OR q_ln2 > INT64_MAX/30 -- the plan's own two axes
# collapse to one outcome value, but each retains its own boundary/interior cells
# below since a mutation weakening one axis's check does not affect the other's),
# kBadQB (q_p + q_b not representable), kNotRepresentable (decomposition well-formed;
# (base^2 + q_c) >> z does not fit int64). Every outcome gets its own boundary pair
# (last-accepting / first-rejecting) plus an interior instance, so a mutation
# weakening any ONE clause is forced by a cell that isolates that clause.
# ---------------------------------------------------------------------------

construct_cases: list[tuple] = []
# (label, q, q_ln2, q_b, q_c, expected_domain, expected_z, expected_base, expected_value)
# expected_z/expected_base/expected_value are valid (and asserted) whenever
# expected_domain is "kOk" OR "kNotRepresentable" -- IExpConstruction is filled
# whenever the decomposition itself is well-formed, independent of whether the FINAL
# narrowed result fits int64, and IExpEvaluate is TOTAL over both outcomes (S-HARDEN-0
# final API: IExpEvaluate takes only the construction, never a caller-supplied q_c).
# expected_value is the exact value IExpEvaluate must return: for "kOk" this is the
# true (unnarrowed) shifted value, already representable; for "kNotRepresentable" this
# is that same value NARROWED to int64's low 64 bits (wrap_to_int64), matching the
# real S128-to-int64 narrowing shift IExpEvaluate performs (confirmed against the
# committed golden -- see wrap_to_int64's own docstring). For the three kBad* outcomes
# expected_z/expected_base/expected_value are 0 placeholders, never asserted (the
# untouched-*out contract is checked against a sentinel or a priming construction, not
# against these placeholders -- see the test-design record).

_DOMAIN_OK = "kOk"
_DOMAIN_NOT_REPRESENTABLE = "kNotRepresentable"
_DOMAIN_BAD_Q = "kBadQ"
_DOMAIN_BAD_QLN2 = "kBadQLn2"
_DOMAIN_BAD_QB = "kBadQB"


def add_construct(label: str, q: int, q_ln2: int, q_b: int, q_c: int) -> None:
    # in_domain()/derive() assume q <= 0 and q_ln2 >= 1 (documented preconditions of
    # the formula itself -- q_ln2 <= 0 divides by zero in derive(), and q > 0 makes
    # Python's floor-division z go negative, which val >> z then rejects outright).
    # IExpConstruct additionally owns rejecting q > 0 (kBadQ) and q_ln2 <= 0 (part of
    # kBadQLn2) BEFORE ever reaching that formula, so those rows are classified
    # directly here, by construction, never by calling in_domain()/derive() on an
    # input its own formula does not assume.
    if q > 0:
        construct_cases.append((label, q, q_ln2, q_b, q_c, _DOMAIN_BAD_Q, 0, 0, 0))
        return
    if q_ln2 < 1:
        construct_cases.append((label, q, q_ln2, q_b, q_c, _DOMAIN_BAD_QLN2, 0, 0, 0))
        return
    if q_ln2 > kIExpMaxQLn2:
        construct_cases.append((label, q, q_ln2, q_b, q_c, _DOMAIN_BAD_QLN2, 0, 0, 0))
        return
    # q<=0, 1<=q_ln2<=kIExpMaxQLn2: the decomposition's z/q_p are well-defined; base
    # (q_p+q_b) may or may not itself be representable -- check that BEFORE calling
    # in_domain() (which assumes base is already representable to square it).
    _, z, q_p, base = derive(q, q_ln2, q_b)
    if not (INT64_MIN <= base <= INT64_MAX):
        construct_cases.append((label, q, q_ln2, q_b, q_c, _DOMAIN_BAD_QB, 0, 0, 0))
        return
    ok, shifted, z2, base2 = in_domain(q, q_ln2, q_b, q_c)
    assert z2 == z and base2 == base  # in_domain() re-derives the same values
    if ok:
        construct_cases.append((label, q, q_ln2, q_b, q_c, _DOMAIN_OK, z, base, shifted))
    else:
        construct_cases.append(
            (label, q, q_ln2, q_b, q_c, _DOMAIN_NOT_REPRESENTABLE, z, base, wrap_to_int64(shifted))
        )


# --- kBadQ: q > 0. Reuses the existing "realistic_s0" witness family (already-shipped
#     golden 264 at q=0) as the last-valid boundary, so this cell cross-checks against
#     a value the suite has pinned since S2.2 rather than a fresh one. ---
add_construct("badq_last_valid_q_zero", 0, 6, 13, 95)
add_construct("badq_first_invalid_q_one", 1, 6, 13, 95)
# Confirmed by execution at f078403 (test-design record): calling the UNCHANGED
# IExpFromConstants(1000000, 6, 13, 95) crashed THERE with "shift exponent -166666 is
# negative" at src/intmath.cpp:50 -- a THIRD, currently-live UB class distinct from
# F9/F21, found the same way F21 was (by exploring the domain object, not named in
# either review finding), accepted by Brunel and filed to the planner's register as
# kBadQ. Not added to the Layer A population (out of the two named classes' explicit
# scope) but its closure rides this exact check, so this Layer B cell is also that
# finding's regression cell once IExpConstruct exists.
add_construct("badq_interior_invalid_large_positive_q", 1000000, 6, 13, 95)

# --- kBadQLn2, lower axis: q_ln2 >= 1. q_ln2=0 would divide by zero in the OLD
#     decomposition if ever reached (never executed here -- add_construct
#     short-circuits q_ln2<1 without calling in_domain/derive, so this generator
#     itself never divides by zero either). ---
add_construct("badqln2_last_valid_qln2_one", 0, 1, 5, 10)
add_construct("badqln2_first_invalid_qln2_zero", 0, 0, 5, 10)
add_construct("badqln2_interior_invalid_qln2_negative", 0, -1000, 5, 10)

# --- kBadQLn2, ceiling axis: q_ln2 <= INT64_MAX/30, reusing the Layer A witnesses
#     above (kIExpGuardOrderCases' f9_* rows) so the SAME inputs that crash today via
#     the unchanged evaluator are also proven, once IExpConstruct exists, to
#     construct cleanly (last-safe) or reject cleanly (first-over / interior / F9's
#     own extreme witness) through the new checked entry point. ---
add_construct("badqln2_last_valid_ceiling", 0, kIExpMaxQLn2, 1, 0)
add_construct("badqln2_first_invalid_ceiling", 0, kIExpMaxQLn2 + 1, 1, 0)
add_construct("badqln2_interior_invalid_ceiling", 0, 2 * kIExpMaxQLn2, 1, 0)
add_construct("badqln2_f9_witness_int64_max", 0, INT64_MAX, 1, 0)

# --- kBadQB: q_p + q_b representability, reusing F21's exact witnesses -- now
#     independently observable from kNotRepresentable under the enum (see this
#     block's own docstring above). Plus the interior point and the q_p+q_b
#     representable-but-square-overflows boundary the retired Layer A crash-probe
#     population (kIExpGuardOrderCases, f21_qb_boundary_last_safe /
#     f21_qb_interior_unsafe) carried and this port did not want to drop -- see
#     test_main.cpp's comment at the retired TestIExpGuardOrderCases... function for
#     why that population was retired rather than ported mechanically. ---
add_construct("badqb_f21_witness_qb_int64_min", -1, 1000, INT64_MIN, 0)
add_construct("badqb_boundary_first_unsafe", -999, 1000, INT64_MIN + 998, 0)
add_construct("badqb_interior_unsafe", -999, 1000, INT64_MIN + 500, 0)
# q_p + q_b == INT64_MIN exactly here (representable -- NOT kBadQB, confirmed by
# execution of this generator itself, not assumed), but base == INT64_MIN makes
# base**2 alone ~2**126, which the final (base**2+q_c)>>z step cannot represent at
# ANY z -- so this lands kNotRepresentable, not kOk. This is worth keeping precisely
# because it is a fact the prior Layer A analysis did not state: an R4-safe (q_p+q_b
# representable) input is NOT thereby R5-safe -- the two checks are independent in
# this direction, even though R4-UNSAFE always implies R5 would also reject (the
# direction the prior test-design record's R4/R5 note already proved).
add_construct("badqb_representable_boundary_square_not_representable", -999, 1000, INT64_MIN + 999, 0)

# --- kNotRepresentable: the decomposition is well-formed (q_p+q_b representable) but
#     (base^2+q_c)>>z does not fit int64. The witness reuses D-SLM78's original strike
#     input (Claude/Loki/softmax-s2.6-strike-2026-07-21.md) -- the SAME triple the
#     pre-existing, untouched `TestIExpFromConstantsAssertsOnOutOfDomainConstants`
#     already pins an exact wrapped IExpFromConstants value for under NDEBUG -- so
#     this cell's z/base=0/1733160715 is the construction that golden's wrapped value
#     is computed from, not an independently-chosen pair. The boundary pair reuses
#     the SAME z=0 boundary already forced in kIExpDomainCases above
#     (`z0_boundary_last_in_domain` / `z0_boundary_first_out_of_domain`), computed
#     from the same `_q_c_max_z0`/`_q_c_first_out_z0` this file already derived. ---
add_construct("notrepresentable_strike_witness", 0, QLN2, QB, QC_STRIKE)
add_construct("notrepresentable_boundary_last_ok", 0, QLN2, QB, _q_c_max_z0)
add_construct("notrepresentable_boundary_first_not_representable", 0, QLN2, QB, _q_c_first_out_z0)

# ---------------------------------------------------------------------------
# Finding 1 (Poirot's a1d7986 code review of this branch): two input strips executed
# NO undefined behaviour at f078403 -- they returned defined-but-meaningless values,
# not UB -- and are now refused by IExpConstruct. The previous suite (Layer A's
# kIExpGuardOrderCases, built to prove absence of UB) never included them, because
# they were never UB; they surfaced only from an executed differential against the
# old code, which is Poirot's review, not this suite. Boundary AND interior of each
# strip, so a mutation that narrows either rejection only partway is still caught.
# ---------------------------------------------------------------------------

# --- Strip A: 0 < q < q_ln2. At f078403, `clipped = q`; `(-q)/q_ln2` truncates
#     toward zero so z == 0; q_p == q; the final shift is by 0 -- no signed overflow,
#     no negative shift, no division by zero; the call returned q_b (offset by q_p)
#     squared plus q_c, a defined answer. Now: kBadQ, unconditionally, before any of
#     that arithmetic runs. ---
# Poirot's own smallest executed witness: IExpFromConstants(1, 1000, 0, 0) returned 1
# at f078403 (review, "Finding 1"); IExpConstruct(1, 1000, 0, 0, ...) now returns
# kBadQ and forms no construction at all.
add_construct("stripa_review_smallest_witness", 1, 1000, 0, 0)
add_construct("stripa_boundary_q_just_below_qln2", 999, 1000, 0, 0)  # q == q_ln2 - 1
add_construct("stripa_interior", 500, 1000, 0, 0)

# --- Strip B: q <= 0, -(INT64_MAX/30) <= q_ln2 <= -1. At f078403, `clip_lo =
#     -30*q_ln2` is positive and in range, `clipped = clip_lo`, z == 30, q_p == 0,
#     base == q_b, shift right by 30 -- all defined. Now: kBadQLn2, unconditionally,
#     before any of that arithmetic runs. Boundary at both ends of the strip's own
#     q_ln2 range (closest to 0, and the most negative value the strip still covers)
#     plus an interior point distinct from both and from the pre-existing
#     badqln2_interior_invalid_qln2_negative cell (q_ln2=-1000, already inside this
#     strip but not at either of its own edges). ---
add_construct("stripb_boundary_qln2_neg_one", 0, -1, 0, 0)
add_construct("stripb_boundary_qln2_most_negative", 0, -kIExpMaxQLn2, 0, 0)
add_construct("stripb_interior", 0, -(kIExpMaxQLn2 // 2), 0, 0)

assert len(construct_cases) == 23

# ---------------------------------------------------------------------------
# Emit the C++ header.
# ---------------------------------------------------------------------------

lines: list[str] = []
emit = lines.append


def cxx_i64(v: int) -> str:
    # INT64_MIN's magnitude (9223372036854775808) exceeds INT64_MAX as a positive
    # literal, so INT64_C(-9223372036854775808) parses the digits as unsigned first and
    # then negates them (MSVC C4146) -- correct in value but a build warning. Emit the
    # standard-library constant directly for that one value; every other value round-trips
    # through INT64_C unchanged.
    if v == INT64_MIN:
        return "INT64_MIN"
    return f"INT64_C({v})"


def cxx_str(s: str) -> str:
    return f'"{s}"'


def cxx_bool(v: bool) -> str:
    return "true" if v else "false"


emit("// GENERATED FILE. Do not hand-edit.")
emit("//")
emit("// Produced by tests/gen_iexp_domain_fixtures.py: every expected value is computed")
emit("// there from the documented five-line i-exp decomposition using Python")
emit("// arbitrary-precision integers -- never by calling IExpConstruct/IExpEvaluate/")
emit("// IExpConstantsInDomain and never by re-deriving the bound")
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

emit("// --- IExpConstruct's (z, base): independently-derived pairs, keyed to the")
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

emit("// --- S-HARDEN-0 LAYER B: kIExpConstructCases, for the checked `IExpConstruct` /")
emit("//     `IExpEvaluate` entry point (final API, S-HARDEN-0). expected_domain is one")
emit("//     of \"kOk\", \"kNotRepresentable\", \"kBadQ\", \"kBadQLn2\", \"kBadQB\" -- a")
emit("//     string, not superslm::IExpDomain directly, so this header stays free of a")
emit("//     compile-time dependency on that enum's exact values; the consuming test maps")
emit("//     IExpDomain to its name and string-compares.")
emit("//")
emit("//     expected_z/expected_base/expected_value are valid (IExpConstruction is")
emit("//     filled, and IExpEvaluate on it is asserted) whenever expected_domain is")
emit("//     \"kOk\" or \"kNotRepresentable\" -- the decomposition is well-formed for both,")
emit("//     and IExpEvaluate is TOTAL over both per the final API. expected_value is the")
emit("//     exact wrapped int64 IExpEvaluate must return -- see gen_iexp_domain_fixtures")
emit("//     .py's wrap_to_int64() docstring for how a kNotRepresentable row's value is")
emit("//     derived. All three fields are 0 placeholders, never asserted, for the three")
emit("//     kBad* rows, where *out is contractually left untouched (checked against a")
emit("//     priming construction, not against these placeholders).")
emit("//")
emit("//     Includes the ported S-HARDEN-0 population cells (retired kIExpGuardOrderCases")
emit("//     -- see this generator's own comment at the retirement site) and Poirot's")
emit("//     a1d7986 code-review Finding 1 strips (stripa_*/stripb_*). Test-design record:")
emit("//     Claude/Curie/superslm-s-harden-0-test-design-2026-07-21.md ---")
emit("")
emit("struct IExpConstructCase {")
emit("\tconst char* label;")
emit("\tint64_t q;")
emit("\tint64_t q_ln2;")
emit("\tint64_t q_b;")
emit("\tint64_t q_c;")
emit("\tconst char* expected_domain;")
emit("\tint64_t expected_z;")
emit("\tint64_t expected_base;")
emit("\tint64_t expected_value;  // IExpEvaluate(...); valid for kOk/kNotRepresentable only")
emit("};")
emit("")
emit("inline constexpr IExpConstructCase kIExpConstructCases[] = {")
for label, q, q_ln2, q_b, q_c, domain, z, base, value in construct_cases:
    emit(
        f"\t{{{cxx_str(label)}, {cxx_i64(q)}, {cxx_i64(q_ln2)}, {cxx_i64(q_b)}, "
        f"{cxx_i64(q_c)}, {cxx_str(domain)}, {cxx_i64(z)}, {cxx_i64(base)}, {cxx_i64(value)}}},"
    )
emit("};")
emit(f"inline constexpr size_t kIExpConstructCasesCount = {len(construct_cases)};")
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
