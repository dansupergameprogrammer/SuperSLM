#!/usr/bin/env python3
"""Generates sslm_s3_2_fixtures.h -- concrete, pinned witnesses for S3.2's own
red cells (Claude/Plans/SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.2; Sec5.1
C31, Sec4.4/Sec7.2 second limb C28, Sec7.2a third limb BIA1).

WHAT THIS PINS, AND WHAT IT DOES NOT (read before extending). Verified by direct
search (`grep -rln "FloorDiv\\|RmsNorm\\|ApplyWeightScale\\|Fold\\|attn_norm\\|
embed_entry\\|BiasCodeOutOfDomain\\|ValidateBiasesDomain\\|
RoundingDivideByPotExponentDomain" include/ src/`, zero hits except the removed
CheckRoundingDivideByPotExponentDomain's own history), S3.2's forward-composition
entry points -- the RMSNorm site, the WSC1 identity/near-identity fold-apply, the
C28 bias-reconciliation site, the embed entry with its token-id validation, and
the BIA1 value-domain descriptor -- do not exist anywhere in D:\\SuperSLM: not a
declaration, not a stub, not a status enum member. `CheckRoundingDivideByPot
ExponentDomain` (C28's own predicate wrapper) was staged once, during S3.1's
header-contract commit (32aca0c), and deliberately removed the same day (S7,
f98eee9) as "S3.2's own predicate" -- confirmed by `git log --all -S` against
this tree, not assumed from the commit message alone.

So NONE of S3.2's Coverage-Model cells can be authored as real, compiling,
red-or-green C++ test code this pass -- there is no shipped composition or
wrapper of any kind for this sub-slot to stand in for, unlike S3.1's C30/C34
cells (each anchored to an ALREADY-SHIPPED validator: IExpConstantsInDomain,
ValidateCompositionConstantsDomain). What this module does instead, matching
gen_s3_1_c30_iexp_domain_sweep_fixtures.py's own precedent for a fully-specified
but not-yet-droppable-in cell: derive and PIN, by execution against the vendored
reference (never re-derived by hand, never copied from a probe), the exact
witness data every S3.2 red cell will need the moment Brunel's header contract
lands. Every derivation below is verified in generate()/build_*() by an
assertion that fails loudly if the claim it makes is false -- this is the
generator's own mutation-provable claim, per StandardsDocument Sec4/D-SLM357: a
comment naming a mechanism is proven by breaking the mechanism and watching the
check fail (see the test-design record's mutation-verification log).

Re-running this script must reproduce the emitted header byte-for-byte.

Test-design record:
Claude/Curie/superslm-s3.2-weightless-and-projection-sites-test-design-2026-07-28.md
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "reference"))
_SPIKE_DIR = os.path.join(_REFERENCE_DIR, "superslm_spike")
sys.path.insert(0, _SPIKE_DIR)
sys.path.insert(0, _REFERENCE_DIR)

import intmath as im  # noqa: E402  (the vendored, hash-pinned reference)

OUT_PATH = os.path.join(_THIS_DIR, "sslm_s3_2_fixtures.h")

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
NORM_FRAC_BITS = 16


def _trunc_div(a: int, b: int) -> int:
    """C's `/` on signed integers: truncates toward zero. `b` is always > 0
    here (a hidden-row's ISqrt-derived root, guarded >= 1 by C31 itself)."""
    assert b > 0
    q = abs(a) // abs(b)
    return -q if a < 0 else q


def _round_half_up_ratio(numerator: int, denominator: int) -> int:
    """C2's tie rule (ties toward +infinity) -- the WRONG rule for C28's divide,
    used only to construct the sign-inverted negative control below."""
    return (2 * numerator + denominator) // (2 * denominator)


# ---------------------------------------------------------------------------
# C31 (Sec5.1) -- the unit cell: F-S3-2's own three-row witness table,
# transcribed verbatim (Sec4.2) and cross-checked against the vendored
# reference's own floor semantics (Python `//`), never against a hand-copied
# expected value.
# ---------------------------------------------------------------------------

def build_c31_unit_cases() -> list[dict]:
    # (hidden, roots) pairs named in F-S3-2, at NORM_FRAC_BITS=16 (numerator hidden<<32).
    pairs = [(-5, 7), (-127, 3), (-1, 2)]
    rows = []
    for hidden, roots in pairs:
        numerator = hidden << (2 * NORM_FRAC_BITS)
        floor_q = numerator // roots            # Python // is floor; roots > 0 always
        trunc_q = _trunc_div(numerator, roots)
        rows.append({
            "hidden": hidden, "roots": roots,
            "floor_q": floor_q, "trunc_q": trunc_q,
            "differs": floor_q != trunc_q,
        })
    # F-S3-2's own claim: the first two rows differ (inexact negative divide),
    # the third does not (exact divide, -1<<32 / 2 has no remainder).
    assert rows[0]["differs"] and rows[1]["differs"] and not rows[2]["differs"], (
        "F-S3-2's own three-row witness table no longer reproduces its stated "
        "divergence pattern -- the finding's own text is now wrong, or this "
        "transcription is"
    )
    assert rows[0]["floor_q"] == -3067833783 and rows[1]["floor_q"] == -181820282198 \
        and rows[2]["floor_q"] == -2147483648, (
        "F-S3-2's table values (Sec4.2) do not reproduce -- transcription error"
    )
    return rows


# ---------------------------------------------------------------------------
# C31 (Sec5.1) -- the site cell: its own witness, constructed at reachable
# (h, root) pairs (the unit table's pairs are unreachable at the site -- Sec5.1
# states root >= |h|*2**16/sqrt(H), which none of F-S3-2's three pairs satisfy
# at any plausible H). H=1536 (Qwen2.5-class hidden_size, Sec21); a sparse
# hidden/gain row (mostly zero, matching Sec4.7's own "ordinary rather than
# adversarial" sparse-row observation) with two nonzero, negative elements
# whose floor/truncation quotients diverge, and one positive element (which
# never diverges, since floor==trunc for non-negative numerators) as an
# in-band control on the same row.
# ---------------------------------------------------------------------------

def build_c31_site_case() -> dict:
    import math

    H = 1536
    nonzero = {0: (-5, 100), 1: (64, -30), 17: (-127, 5)}  # index -> (h, g)
    h = [0] * H
    g = [0] * H
    for idx, (hv, gv) in nonzero.items():
        h[idx] = hv
        g[idx] = gv

    sumsq = sum(hi * hi for hi in h)
    root_input = (sumsq << (2 * NORM_FRAC_BITS)) // H  # C31's own floordiv; H>0, numerator>=0 -> floor==trunc here
    root = max(im.i_sqrt(root_input), 1)

    # Reachability: root >= |h|*2^16/sqrt(H) at every nonzero element (Sec5.1's
    # own stated constraint on the site witness, checked rather than assumed).
    for idx, (hv, _gv) in nonzero.items():
        lower_bound = abs(hv) * (1 << 16) / math.sqrt(H)
        assert root >= lower_bound, (
            f"C31 site witness element {idx} (h={hv}): root={root} is below the "
            f"site's own reachability floor {lower_bound:.1f} -- this (h, root) pair "
            f"cannot occur at a real RMSNorm site (Sec5.1)"
        )

    elements = []
    any_divergence = False
    for idx in sorted(nonzero):
        hv, gv = nonzero[idx]
        num = hv << (2 * NORM_FRAC_BITS)
        floor_q = num // root
        trunc_q = _trunc_div(num, root)
        wide_floor = floor_q * gv
        wide_trunc = trunc_q * gv
        diverges = wide_floor != wide_trunc
        any_divergence = any_divergence or diverges
        elements.append({
            "index": idx, "h": hv, "g": gv,
            "floor_wide": wide_floor, "trunc_wide": wide_trunc,
            "diverges": diverges,
        })

    assert any_divergence, (
        "C31 site witness produced NO element where floor and truncation diverge -- "
        "the discriminating fixture Sec5.1 requires would prove nothing"
    )
    # The positive element (index 1) must NOT diverge -- floor and truncation
    # agree on every non-negative numerator, and this is the in-band control
    # that proves the divergence is really about sign, not about this row.
    positive_row = next(e for e in elements if e["index"] == 1)
    assert not positive_row["diverges"], (
        "C31 site witness's positive-h control element unexpectedly diverges "
        "between floor and truncation -- floor and trunc must agree on a "
        "non-negative numerator; the witness construction is wrong"
    )

    return {"H": H, "root": root, "sumsq": sumsq, "elements": elements}


# ---------------------------------------------------------------------------
# C24 (Sec4, master-plan row; Sec11 S3.2's own two-element-row cell) -- the
# identity/near-identity discrimination on the plan's own stated row. The
# reference (widest) channel's raw accumulator value is 2**30+1 -- a TRUE
# PASS-THROUGH (correct: no multiply, no shift) leaves it unchanged and sets
# the row's D'; a WRONG implementation that applies the near-identity
# multiplier (2**31-1, 0) to the identity channel too (instead of exempting
# it) folds it to one less. Because D' is shared by the whole row through the
# C19-22 chain ("the divergence class is TOKEN-WIDE through the shared D'",
# SuperSLM_Plan.md:2194-2197), that one-unit difference in the reference
# channel changes the requantization scale for every OTHER element too --
# including the fixed, already-folded second channel x=215593831, whose own
# raw value never changes between the two runs. This is why the witness is a
# two-element row rather than a single value: the fixture's whole point is
# that the near-identity control's error is invisible if you look only at the
# element it was applied to.
# ---------------------------------------------------------------------------

def build_c24_witness() -> dict:
    d_correct = (1 << 30) + 1     # the reference channel's raw value (pass-through, no fold)
    mult = (1 << 31) - 1
    shift = 0
    d_near_identity = im.multiply_by_quantized_multiplier(d_correct, mult, shift)
    assert d_near_identity == d_correct - 1, (
        f"the near-identity fold on the reference channel's own value no longer "
        f"reproduces the plan's stated 'off by one' — got {d_correct} -> "
        f"{d_near_identity}"
    )

    x = 215593831  # the plan's own second element (SuperSLM_Plan.md C24 row)

    def _run(d_ref: int) -> tuple[int, int, int, int]:
        d_prime = max(d_ref, x, 1)
        dn, s = im.normalize_scale(d_prime)
        r = im.dynamic_scale_reciprocal(dn)
        code_ref = im.requant_token_code(d_prime, r, s)
        code_x = im.requant_token_code(x, r, s)
        return d_prime, dn, s, code_ref, code_x

    d_prime_pt, dn_pt, s_pt, code_ref_pt, code_x_pt = _run(d_correct)
    d_prime_ni, dn_ni, s_ni, code_ref_ni, code_x_ni = _run(d_near_identity)

    assert (code_ref_pt, code_x_pt) == (127, 25), (
        f"pass-through codes no longer reproduce the plan's stated [127, 25] -- "
        f"got [{code_ref_pt}, {code_x_pt}]"
    )
    assert (code_ref_ni, code_x_ni) == (127, 26), (
        f"near-identity codes no longer reproduce the plan's stated [127, 26] -- "
        f"got [{code_ref_ni}, {code_x_ni}]"
    )
    assert code_x_pt != code_x_ni, (
        "the pass-through and near-identity runs must diverge on the SHARED "
        "second element's code -- that divergence, through the shared D', is "
        "the entire point of this witness"
    )

    return {
        "d_correct": d_correct, "d_near_identity": d_near_identity, "x": x,
        "code_ref_pass_through": code_ref_pt, "code_x_pass_through": code_x_pt,
        "code_ref_near_identity": code_ref_ni, "code_x_near_identity": code_x_ni,
    }


# ---------------------------------------------------------------------------
# BIA1 (Sec7.2a) -- the magnitude descriptor: |B[j]| such that B[j]*R_a stays
# inside int64, where R_a is C19's reciprocal over the incoming carried
# mantissa. R_a's own maximum is derived by calling the vendored C19
# reciprocal at its own domain's minimum (Dn = 2**30) -- never asserted from a
# closed-form guess -- because dynamic_scale_reciprocal is monotonically
# decreasing in Dn (R = round_half_up(2**62 / Dn)), so its extremum over
# Dn in [2**30, 2**31) is attained at the domain's left endpoint.
# ---------------------------------------------------------------------------

def build_bia1_bound() -> dict:
    # T-1657/D-SLM621/D-SLM665 (Poirot T-1657 review, Minor 2): the derived
    # bound this function used to compute (INT64_MAX // r_a_max) and assert tight
    # against R_a's own maximum is retired along with the load-time gate it fed --
    # there is no live constraint left that ties the fixture values below to R_a's
    # maximum, so re-deriving and asserting that relationship here proved a
    # constraint nothing enforces any more. r_a_max itself is kept and still
    # asserted: it is a genuine, still-true property of the vendored C19 reciprocal
    # (dynamic_scale_reciprocal is monotonically decreasing in Dn, so its maximum
    # over Dn in [2**30, 2**31) is attained at the domain's left endpoint,
    # Dn = 2**30) independent of any load-time gate, and this assertion is what
    # would catch that formula drifting.
    r_a_max = im.dynamic_scale_reciprocal(1 << 30)
    assert r_a_max == 1 << 32, (
        f"R_a's derived maximum is {r_a_max}, not 2**32 -- C19's reciprocal "
        f"formula or its vendored implementation changed"
    )
    # The three fixture values below are the FORMER load-time bound's own former-
    # boundary witnesses (INT32_MAX and one past it) -- fixed historical constants
    # now, kept only because design §10 cell 7 proves the REVERSE claim against
    # them: that all three load successfully, with no load-time gate left to
    # reject any of them.
    former_bound = INT32_MAX
    return {
        "r_a_max": r_a_max,
        "hostile_value": former_bound + 1,          # one past the FORMER bound
        "hostile_value_negated": -(former_bound + 1),
        "accept_boundary_value": former_bound,      # the former bound itself
    }


# ---------------------------------------------------------------------------
# C28 (Sec7.2 second limb, Sec4.4) -- the (q_B, e_a) site predicate's domain
# boundary, and the bias-reconciliation divide's sign-inverted negative
# control. q_B = 30 is F-S3-4's own verified constant ("q_b is uniform 30
# across dynamic_biases (verified)", tools/convert_model.py:149).
# ---------------------------------------------------------------------------

Q_B = 30


def build_c28_domain_boundary() -> dict:
    # k = q_B + 62 + e_a must be checked against [0, 63] (kRoundingDivideByPot
    # ExponentMinI64/MaxI64, intmath.h:59-60) before RoundingDivideByPOT is
    # called. Four boundary points, all four derived from k = Q_B + 62 + e_a
    # rather than guessed: k=-1 and k=64 are OUT (reject before the call);
    # k=0 and k=63 are IN (RoundingDivideByPOT's own inclusive endpoints).
    def e_a_for_k(k: int) -> int:
        return k - Q_B - 62

    points = {}
    for k, label in [(-1, "below_min"), (0, "at_min"), (63, "at_max"), (64, "above_max")]:
        e_a = e_a_for_k(k)
        computed_k = Q_B + 62 + e_a
        assert computed_k == k, "k/e_a arithmetic inversion is wrong"
        points[label] = {"e_a": e_a, "k": k, "in_domain": 0 <= k <= 63}

    assert points["below_min"]["in_domain"] is False
    assert points["at_min"]["in_domain"] is True
    assert points["at_max"]["in_domain"] is True
    assert points["above_max"]["in_domain"] is False
    return points


def build_c28_tie_witness() -> dict:
    """A constructed (B, R_a, e_a) triple landing the C28 divide on an EXACT
    half-integer tie, both signs, found by brute-force search over small B and
    a handful of R_a values drawn from C19's real reciprocal range
    [2**31+1, 2**32] (verified: dynamic_scale_reciprocal never returns a value
    outside this range over a 2000-point random sample plus both domain
    endpoints -- see the test-design record). Ties matter here specifically
    because C28 is C3 (away from zero), not C2 (toward +infinity) -- and the
    two rules disagree only on a tie, and only on its negative sign.
    """
    candidate_r_a = [2147483649, 3000000001, 2500000001, 4294967295, 3221225473]
    for e_a in range(-92, 30):
        k = Q_B + 62 + e_a
        if not (0 <= k <= 63) or k == 0:
            continue
        half = 1 << (k - 1)
        mod = 1 << k
        for r_a in candidate_r_a:
            for b in range(1, 200000):
                if (b * r_a) % mod == half:
                    return _finish_c28_tie_witness(b, r_a, e_a, k)
    raise AssertionError("no C28 tie witness found in the searched space -- widen the search")


def _finish_c28_tie_witness(b: int, r_a: int, e_a: int, k: int) -> dict:
    correct_pos = im.bias_reconcile(b, Q_B, r_a, e_a)
    correct_neg = im.bias_reconcile(-b, Q_B, r_a, e_a)

    def wrong_bias_reconcile(bb: int) -> int:
        numerator = bb * r_a
        if k >= 0:
            return _round_half_up_ratio(numerator, 1 << k)
        return numerator << (-k)

    wrong_pos = wrong_bias_reconcile(b)
    wrong_neg = wrong_bias_reconcile(-b)

    # The whole point of this witness: the two rounding rules MUST agree on
    # the positive tie (both round the same magnitude away from/at zero the
    # same way going toward +inf) and MUST disagree on the negative tie --
    # that disagreement is the sign-inverted negative control.
    assert wrong_pos == correct_pos, (
        "the wrong (round-half-up) candidate unexpectedly disagrees with the "
        "correct (away-from-zero) result on the POSITIVE tie -- this witness "
        "does not isolate the sign condition and must not be used as the "
        "negative control"
    )
    assert wrong_neg != correct_neg, (
        "the wrong (round-half-up) candidate unexpectedly AGREES with the "
        "correct (away-from-zero) result on the NEGATIVE tie -- the negative "
        "control has no discriminating power and must not be used"
    )
    assert abs(b) <= INT32_MAX, "the tie witness's B must itself be inside BIA1's own domain"

    return {
        "b": b, "q_b": Q_B, "r_a": r_a, "e_a": e_a, "k": k,
        "correct_pos": correct_pos, "correct_neg": correct_neg,
        "wrong_pos": wrong_pos, "wrong_neg": wrong_neg,
    }


# ---------------------------------------------------------------------------
# T-1657 (D-SLM621/641/645, design Sec10 cells 1/2/3/8) -- BiasReconcileWide's
# own widened-arithmetic witnesses. All four derived by EXECUTING the vendored
# reference (im.bias_reconcile, arbitrary precision), never by hand.
# ---------------------------------------------------------------------------

def build_bia1_wide_bit_exact() -> dict:
    """Sec10 cell 1: bit-exact against the reference at the strike's own worst-
    observed real BIA1 magnitude (9.938e13, past the deleted INT32_MAX bound),
    a canonical r_a (Dn = 2**30, the C19 domain's own left endpoint), and a
    mid-range composed exponent."""
    b = 99381983436726
    r_a = im.dynamic_scale_reciprocal(1 << 30)
    e_a = -52  # exponent = Q_B + 62 + e_a = 30 + 62 - 52 = 40
    exponent = Q_B + 62 + e_a
    assert exponent == 40
    expected = im.bias_reconcile(b, Q_B, r_a, e_a)
    assert INT64_MIN <= expected <= INT64_MAX, (
        "cell 1's own witness must land inside int64_t -- it is meant to "
        "demonstrate the widened arithmetic ACCEPTING a magnitude the deleted "
        "load-time bound used to reject, not a representability failure"
    )
    return {"b": b, "q_b": Q_B, "r_a": r_a, "e_a": e_a, "exponent": exponent, "expected": expected}


def build_bia1_wide_representability_boundary() -> dict:
    """Sec10 cell 2: the representability boundary, constructed exactly at
    int64_t's edge with exponent = 0 (the identity -- RoundingDivideByPOTWide
    applies no rounding at exponent 0, so both constructions are exact integer
    products, not rounding artifacts). e_a = -92 is Q_B + 62 + e_a = 0, the
    same domain floor the C28 boundary fixtures above already pin."""
    e_a = -Q_B - 62
    exponent = Q_B + 62 + e_a
    assert exponent == 0

    fits_b, fits_r_a = INT64_MAX, 1
    fits_expected = im.bias_reconcile(fits_b, Q_B, fits_r_a, e_a)
    assert fits_expected == INT64_MAX

    over_b, over_r_a = 1 << 62, 2
    over_expected = im.bias_reconcile(over_b, Q_B, over_r_a, e_a)
    assert over_expected == INT64_MAX + 1, (
        "the 'one past' construction must land exactly one past INT64_MAX, "
        "not merely past it, to prove the boundary is tight"
    )
    return {
        "q_b": Q_B, "e_a": e_a, "exponent": exponent,
        "fits_b": fits_b, "fits_r_a": fits_r_a, "fits_expected": fits_expected,
        "over_b": over_b, "over_r_a": over_r_a, "over_expected": over_expected,
    }


def build_bia1_wide_strike_witness() -> dict:
    """Sec10 cell 3: the strike's own raw-product-overflow, result-in-range
    witness (Claude/Loki/superslm-t1655-t1656-precondition-ruling-strike-
    2026-08-01.md) -- b = kBia1AcceptBoundaryValue (the FORMER load-time
    bound, INT32_MAX), r_a = 48507865471, at the composed exponent
    BiasReconcile's own formula produces for that witness's in_scale.e = -30.
    This is the case that discriminates T-1657's own claim from the retired
    BiasReconcileProductFitsInt64's: the RAW product overflows int64_t (so the
    retired guard would have wrongly rejected it) but the ROUNDED result does
    not."""
    b = INT32_MAX
    r_a = 48507865471
    e_a = -30
    exponent = Q_B + 62 + e_a
    assert exponent == 62
    raw_product = b * r_a
    assert raw_product > INT64_MAX, "the witness's own raw product must overflow int64_t"
    expected = im.bias_reconcile(b, Q_B, r_a, e_a)
    assert INT64_MIN <= expected <= INT64_MAX, (
        "the witness's own rounded result must fit int64_t -- that is the "
        "entire point of this discriminating cell"
    )
    return {"b": b, "q_b": Q_B, "r_a": r_a, "e_a": e_a, "exponent": exponent, "expected": expected}


def build_bia1_wide_tie_witness() -> dict:
    """Sec10 cell 8: a genuine C3 tie (remainder == threshold exactly) at a
    magnitude where |b * r_a| > INT64_MAX, forcing the genuinely-128-bit
    rounding path -- the existing narrow-width tie fixture (kC28Tie*) stays
    inside plain int64_t and cannot exercise this. Same search shape as
    build_c28_tie_witness above, widened: b ranges over int64_t magnitudes
    large enough that b * r_a exceeds int64_t for at least one candidate r_a,
    with the tie condition ((b * r_a) mod 2**k == half) checked in exact
    (unbounded) Python arithmetic."""
    candidate_r_a = [2147483649, 3000000001, 2500000001, 4294967295, 3221225473]
    base = 1 << 40  # a band of B large enough that B * R_a can exceed int64_t
    for e_a in range(-92, 30):
        k = Q_B + 62 + e_a
        if not (0 <= k <= 63) or k == 0:
            continue
        half = 1 << (k - 1)
        mod = 1 << k
        for r_a in candidate_r_a:
            for delta in range(200000):
                b = base + delta
                if b * r_a <= INT64_MAX or (b * r_a) % mod != half:
                    continue
                pos = im.bias_reconcile(b, Q_B, r_a, e_a)
                neg = im.bias_reconcile(-b, Q_B, r_a, e_a)
                if not (INT64_MIN <= pos <= INT64_MAX and INT64_MIN <= neg <= INT64_MAX):
                    continue  # representable-result candidates only (cell 2 owns the boundary)
                return _finish_bia1_wide_tie_witness(b, r_a, e_a, k)
    raise AssertionError("no wide C28 tie witness found in the searched space -- widen the search")


def _finish_bia1_wide_tie_witness(b: int, r_a: int, e_a: int, k: int) -> dict:
    correct_pos = im.bias_reconcile(b, Q_B, r_a, e_a)
    correct_neg = im.bias_reconcile(-b, Q_B, r_a, e_a)

    def wrong_bias_reconcile(bb: int) -> int:
        numerator = bb * r_a
        if k >= 0:
            return _round_half_up_ratio(numerator, 1 << k)
        return numerator << (-k)

    wrong_pos = wrong_bias_reconcile(b)
    wrong_neg = wrong_bias_reconcile(-b)
    assert b * r_a > INT64_MAX, "cell 8's own witness must exercise the genuinely-wide path"
    assert INT64_MIN <= correct_pos <= INT64_MAX and INT64_MIN <= correct_neg <= INT64_MAX, (
        "cell 8's own witness must itself be representable -- it tests the tie "
        "rule, not the representability boundary (that is cell 2's job)"
    )
    assert wrong_pos == correct_pos, (
        "the wrong (round-half-up) candidate unexpectedly disagrees with the "
        "correct (away-from-zero) result on the POSITIVE tie"
    )
    assert wrong_neg != correct_neg, (
        "the wrong (round-half-up) candidate unexpectedly AGREES with the "
        "correct (away-from-zero) result on the NEGATIVE tie -- no "
        "discriminating power"
    )
    return {
        "b": b, "q_b": Q_B, "r_a": r_a, "e_a": e_a, "k": k,
        "correct_pos": correct_pos, "correct_neg": correct_neg,
        "wrong_pos": wrong_pos, "wrong_neg": wrong_neg,
    }


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def _fmt_c31_unit(rows: list[dict]) -> list[str]:
    lines = ["inline constexpr C31UnitCase kC31UnitCases[] = {"]
    for r in rows:
        lines.append(
            "\t{ /*hidden=*/%dLL, /*roots=*/%dLL, /*floor_q=*/%dLL, "
            "/*trunc_q=*/%dLL, /*differs=*/%s },"
            % (r["hidden"], r["roots"], r["floor_q"], r["trunc_q"],
               "true" if r["differs"] else "false")
        )
    lines.append("};")
    return lines


def _fmt_c31_site(case: dict) -> list[str]:
    lines = [
        "inline constexpr int kC31SiteHiddenSize = %d;" % case["H"],
        "inline constexpr long long kC31SiteRoot = %dLL;" % case["root"],
        "inline constexpr long long kC31SiteSumSq = %dLL;" % case["sumsq"],
        "inline constexpr C31SiteElement kC31SiteElements[] = {",
    ]
    for e in case["elements"]:
        lines.append(
            "\t{ /*index=*/%d, /*h=*/%d, /*g=*/%d, /*floor_wide=*/%dLL, "
            "/*trunc_wide=*/%dLL, /*diverges=*/%s },"
            % (e["index"], e["h"], e["g"], e["floor_wide"], e["trunc_wide"],
               "true" if e["diverges"] else "false")
        )
    lines.append("};")
    return lines


def _fmt_c24(w: dict) -> list[str]:
    return [
        "inline constexpr long long kC24RefChannelPassThrough = %dLL;" % w["d_correct"],
        "inline constexpr long long kC24RefChannelNearIdentity = %dLL;  // wrong: off by one" % w["d_near_identity"],
        "inline constexpr long long kC24SharedElementX = %dLL;" % w["x"],
        "inline constexpr int kC24CodeRefPassThrough = %d;" % w["code_ref_pass_through"],
        "inline constexpr int kC24CodeXPassThrough = %d;" % w["code_x_pass_through"],
        "inline constexpr int kC24CodeRefNearIdentity = %d;" % w["code_ref_near_identity"],
        "inline constexpr int kC24CodeXNearIdentity = %d;  // differs from kC24CodeXPassThrough via the shared D'" % w["code_x_near_identity"],
    ]


def _fmt_bia1(bound: dict) -> list[str]:
    # T-1657/D-SLM621/D-SLM665: kBia1MagnitudeBound is retired along with the
    # load-time gate it named (kBia1MagnitudeBound/ValidateBiasesDomain/
    # BiasCodeOutOfDomain, all deleted from src/model.cpp and include/superslm/
    # model.h) -- BIA1's bias codes carry no load-time value-domain check any
    # longer, so there is no bound left to name here. The three fixture values
    # below (the former bound's own former-boundary witnesses) are kept: they
    # now prove the REVERSE claim, that all three load successfully.
    #
    # T-1657 Poirot review, Minor 2: `kBia1RaMax` (r_a_max, above) is not emitted
    # here -- it fed only the retired bound derivation, is referenced by no C++
    # code, and HEAD holds current truth (StandardsDocument §6.6): a constant with
    # no reader is not a fixture, it is a stale name. build_bia1_bound()'s own
    # assertion on r_a_max stays, as the live regression guard on the vendored
    # reciprocal's maximum.
    return [
        "inline constexpr long long kBia1HostileValue = %dLL;  // one past the FORMER bound -- must now accept" % bound["hostile_value"],
        "inline constexpr long long kBia1HostileValueNegated = %dLL;" % bound["hostile_value_negated"],
        "inline constexpr long long kBia1AcceptBoundaryValue = %dLL;  // == INT32_MAX, the former bound itself -- must accept" % bound["accept_boundary_value"],
    ]


def _fmt_c28_boundary(points: dict) -> list[str]:
    lines = []
    for label in ["below_min", "at_min", "at_max", "above_max"]:
        p = points[label]
        name = "kC28Boundary" + "".join(w.capitalize() for w in label.split("_"))
        lines.append("inline constexpr long long %sEA = %dLL;" % (name, p["e_a"]))
        lines.append("inline constexpr int %sK = %d;" % (name, p["k"]))
        lines.append("inline constexpr bool %sInDomain = %s;" % (name, "true" if p["in_domain"] else "false"))
    return lines


def _fmt_c28_tie(w: dict) -> list[str]:
    return [
        "inline constexpr long long kC28TieB = %dLL;" % w["b"],
        "inline constexpr long long kC28TieQB = %dLL;" % w["q_b"],
        "inline constexpr long long kC28TieRA = %dLL;" % w["r_a"],
        "inline constexpr long long kC28TieEA = %dLL;" % w["e_a"],
        "inline constexpr int kC28TieK = %d;" % w["k"],
        "inline constexpr long long kC28TieCorrectPos = %dLL;" % w["correct_pos"],
        "inline constexpr long long kC28TieCorrectNeg = %dLL;" % w["correct_neg"],
        "inline constexpr long long kC28TieWrongPos = %dLL;  // wrong candidate; must equal kC28TieCorrectPos" % w["wrong_pos"],
        "inline constexpr long long kC28TieWrongNeg = %dLL;  // wrong candidate; must NOT equal kC28TieCorrectNeg" % w["wrong_neg"],
    ]


def _fmt_bia1_wide_bit_exact(w: dict) -> list[str]:
    return [
        "inline constexpr long long kBia1WideB = %dLL;" % w["b"],
        "inline constexpr long long kBia1WideQB = %dLL;" % w["q_b"],
        "inline constexpr long long kBia1WideRA = %dLL;" % w["r_a"],
        "inline constexpr long long kBia1WideEA = %dLL;" % w["e_a"],
        "inline constexpr int kBia1WideExponent = %d;" % w["exponent"],
        "inline constexpr long long kBia1WideExpected = %dLL;" % w["expected"],
    ]


def _fmt_bia1_wide_boundary(w: dict) -> list[str]:
    return [
        "inline constexpr long long kBia1WideBoundaryQB = %dLL;" % w["q_b"],
        "inline constexpr long long kBia1WideBoundaryEA = %dLL;" % w["e_a"],
        "inline constexpr int kBia1WideBoundaryExponent = %d;" % w["exponent"],
        "inline constexpr long long kBia1WideBoundaryFitsB = %dLL;" % w["fits_b"],
        "inline constexpr long long kBia1WideBoundaryFitsRA = %dLL;" % w["fits_r_a"],
        "inline constexpr long long kBia1WideBoundaryFitsExpected = %dLL;  // == INT64_MAX" % w["fits_expected"],
        "inline constexpr long long kBia1WideBoundaryOverB = %dLL;" % w["over_b"],
        "inline constexpr long long kBia1WideBoundaryOverRA = %dLL;" % w["over_r_a"],
        "// kBia1WideBoundaryOverExpected == INT64_MAX + 1, one past int64_t -- not representable as a literal",
    ]


def _fmt_bia1_wide_strike_witness(w: dict) -> list[str]:
    return [
        "inline constexpr long long kBia1WideStrikeB = %dLL;  // == kBia1AcceptBoundaryValue (INT32_MAX)" % w["b"],
        "inline constexpr long long kBia1WideStrikeQB = %dLL;" % w["q_b"],
        "inline constexpr long long kBia1WideStrikeRA = %dLL;" % w["r_a"],
        "inline constexpr long long kBia1WideStrikeEA = %dLL;" % w["e_a"],
        "inline constexpr int kBia1WideStrikeExponent = %d;" % w["exponent"],
        "inline constexpr long long kBia1WideStrikeExpected = %dLL;" % w["expected"],
    ]


def _fmt_bia1_wide_tie(w: dict) -> list[str]:
    return [
        "inline constexpr long long kBia1WideTieB = %dLL;" % w["b"],
        "inline constexpr long long kBia1WideTieQB = %dLL;" % w["q_b"],
        "inline constexpr long long kBia1WideTieRA = %dLL;" % w["r_a"],
        "inline constexpr long long kBia1WideTieEA = %dLL;" % w["e_a"],
        "inline constexpr int kBia1WideTieK = %d;" % w["k"],
        "inline constexpr long long kBia1WideTieCorrectPos = %dLL;" % w["correct_pos"],
        "inline constexpr long long kBia1WideTieCorrectNeg = %dLL;" % w["correct_neg"],
        "inline constexpr long long kBia1WideTieWrongPos = %dLL;  // wrong candidate; must equal kBia1WideTieCorrectPos" % w["wrong_pos"],
        "inline constexpr long long kBia1WideTieWrongNeg = %dLL;  // wrong candidate; must NOT equal kBia1WideTieCorrectNeg" % w["wrong_neg"],
    ]


def generate() -> str:
    c31_unit = build_c31_unit_cases()
    c31_site = build_c31_site_case()
    c24_witness = build_c24_witness()
    bia1_bound = build_bia1_bound()
    c28_boundary = build_c28_domain_boundary()
    c28_tie = build_c28_tie_witness()
    bia1_wide_bit_exact = build_bia1_wide_bit_exact()
    bia1_wide_boundary = build_bia1_wide_representability_boundary()
    bia1_wide_strike = build_bia1_wide_strike_witness()
    bia1_wide_tie = build_bia1_wide_tie_witness()

    lines = [
        "// GENERATED FILE. Do not hand-edit.",
        "//",
        "// Produced by tests/gen_s3_2_fixtures.py. Every witness below is derived by",
        "// EXECUTING the vendored reference (tests/reference/superslm_spike/intmath.py)",
        "// or by arithmetic checked against it -- never copied from a probe, never",
        "// hand-computed.",
        "//",
        "// T-1657 Poirot review, Observation 1 (corrected): this file's own preamble",
        "// used to state that S3.2's forward-composition entry points (the RMSNorm",
        "// site, the WSC1 fold-apply, the C28 bias-reconciliation site, BIA1's",
        "// value-domain descriptor) \"do not exist in D:\\SuperSLM\" -- true only at the",
        "// S3.2 red-phase authoring of this generator, and false since the green",
        "// phase (src/forward/forward_sites.cpp, src/forward/checked_chain_funnel.cpp).",
        "// The witnesses below are consumed by tests/test_main.cpp's own S3.2/T-1657",
        "// cells against those real bodies, not staged for a future build.",
        "//",
        "// Re-running this script must reproduce this file byte-for-byte.",
        "//",
        "// Test-design record:",
        "// Claude/Curie/superslm-s3.2-weightless-and-projection-sites-test-design-2026-07-28.md",
        "#ifndef SUPERSLM_TESTS_SSLM_S3_2_FIXTURES_H",
        "#define SUPERSLM_TESTS_SSLM_S3_2_FIXTURES_H",
        "",
        "#include <cstddef>",
        "",
        "namespace superslm_test {",
        "",
        "// --- C31 (Sec5.1) unit cell: F-S3-2's own three-row witness table ---------",
        "struct C31UnitCase {",
        "\tlong long hidden;",
        "\tlong long roots;",
        "\tlong long floor_q;   // the pinned (correct) floor-division quotient",
        "\tlong long trunc_q;   // the naive-C++ (wrong) truncating quotient -- negative control",
        "\tbool differs;        // floor_q != trunc_q",
        "};",
        "",
    ]
    lines += _fmt_c31_unit(c31_unit)
    lines += [
        "",
        "inline constexpr size_t kC31UnitCasesCount = sizeof(kC31UnitCases) / sizeof(kC31UnitCases[0]);",
        "",
        "// --- C31 (Sec5.1) site cell: a reachable (h, root) witness at H=1536 -------",
        "struct C31SiteElement {",
        "\tint index;",
        "\tint h;",
        "\tint g;",
        "\tlong long floor_wide;  // the pinned (correct) floor-based wide value",
        "\tlong long trunc_wide;  // the naive-C++ (wrong) truncating wide value -- negative control",
        "\tbool diverges;",
        "};",
        "",
    ]
    lines += _fmt_c31_site(c31_site)
    lines += [
        "",
        "inline constexpr size_t kC31SiteElementsCount = sizeof(kC31SiteElements) / sizeof(kC31SiteElements[0]);",
        "",
        "// --- C24 identity/near-identity discrimination (the plan's own two-element row) ---",
    ]
    lines += _fmt_c24(c24_witness)
    lines += [
        "",
        "// --- BIA1 (Sec7.2a) magnitude descriptor -----------------------------------",
    ]
    lines += _fmt_bia1(bia1_bound)
    lines += [
        "",
        "// --- C28 (Sec7.2 2nd limb) (q_B, e_a) domain boundary ----------------------",
    ]
    lines += _fmt_c28_boundary(c28_boundary)
    lines += [
        "",
        "// --- C28 (Sec4.4) bias-reconciliation sign-inverted tie witness ------------",
    ]
    lines += _fmt_c28_tie(c28_tie)
    lines += [
        "",
        "// --- BIA1 widened core (T-1657, Sec10 cell 1): bit-exact at a magnitude",
        "// --- the deleted load-time bound used to reject -----------------------------",
    ]
    lines += _fmt_bia1_wide_bit_exact(bia1_wide_bit_exact)
    lines += [
        "",
        "// --- BIA1 widened core (T-1657, Sec10 cell 2): representability boundary ---",
    ]
    lines += _fmt_bia1_wide_boundary(bia1_wide_boundary)
    lines += [
        "",
        "// --- BIA1 widened core (T-1657, Sec10 cell 3): the strike's own raw-product-",
        "// --- overflow, result-in-range discriminating witness -----------------------",
    ]
    lines += _fmt_bia1_wide_strike_witness(bia1_wide_strike)
    lines += [
        "",
        "// --- BIA1 widened core (T-1657, Sec10 cell 8): a genuine C3 tie forced onto",
        "// --- the 128-bit path, both signs --------------------------------------------",
    ]
    lines += _fmt_bia1_wide_tie(bia1_wide_tie)
    lines += [
        "",
        "}  // namespace superslm_test",
        "",
        "#endif  // SUPERSLM_TESTS_SSLM_S3_2_FIXTURES_H",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    text = generate()
    with open(OUT_PATH, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
