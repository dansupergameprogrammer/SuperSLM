"""Arbitrary-precision reference oracles for the rung-1 dynamic per-token integer-scale
path (constructions C19-C22, SuperSLM_Plan.md S6.8 / S6.2 rung 1).

Promoted from, so the red suite imports one oracle rather than re-deriving the sweep:
  Tools/superslm_spike/loki_probes/c22_pin_owed_probe.py   (clz64, c20_c21, round_half_up,
      c19_reciprocal, quantize_token_row)
  Tools/superslm_spike/loki_probes/c22_adversarial_sweep.py (candidates_at, the tie-pair
      boundary construction)

Every function here is exact Python `int` / `fractions.Fraction` arithmetic -- NEVER a
float, and never a recode of intmath.py's pinned implementation STEPS (the 3 Newton
iterations + 2 correction steps, the bit-scan intrinsic). Per Curie's discipline and G-8(1)
("never bit-equality against a twin"): a cell built on this module fails when the
implementation under test is DETERMINISTICALLY WRONG, because the oracle recomputes the
pinned formula from its own definition, not because it disagrees with a second
implementation of the same defect.

Construction citations, all SuperSLM_Plan.md S6.8:
  C19 (D-SLM49) -- R = round_half_up(2**62 / Dn), Dn in [2**30, 2**31).
  C20 (D-SLM50) -- D = max_i|x_i|, every intermediate/output int64-domain; D' = max(D, 1).
  C21 (D-SLM51) -- p = 63 - clz64(D'), s = 30 - p, Dn = D'<<s if s>=0 else D'>>1.
  C22 (D-SLM52) -- q_i = clamp(round_half_away_from_zero(x_i*127*R / 2**(62-s)), -127, 127),
      R-COMPOSED semantics (R already carries C19's own rounding; the composite rounds a
      second time, ties away from zero per C3's RoundingDivideByPOT precedent).

Test-design record: Claude/Curie/SuperSLM_Rung1_RedSuite_TestDesign-2026-07-16.md
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

__all__ = [
    "INT32_MIN",
    "INT32_MAX",
    "clz64",
    "round_half_up_ratio",
    "round_half_away_from_zero_ratio",
    "max_abs_reduce_oracle",
    "normalize_scale_oracle",
    "reciprocal_oracle",
    "uncorrected_newton_reciprocal",
    "requant_code_oracle",
    "requant_code_single_rounding_oracle",
    "candidate_pair",
    "positive_tie_boundary_x",
    "negative_tie_boundary_x",
    "is_exact_composite_tie",
    "ALL_D_PRIME_BIT_LENGTHS",
]

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1

# The 31 C21 shift classes: bit-length 1 through 31 span D' in [1, 2**31 - 1]; D' == 2**31
# itself (bit-length 32) is C21's named negative-s edge and is handled as its own point,
# not as a 32nd class -- see normalize_scale_oracle's docstring.
ALL_D_PRIME_BIT_LENGTHS = tuple(range(1, 32))


# --- primitives ----------------------------------------------------------------


def clz64(n: int) -> int:
    """Count-leading-zeros over 64 bits. Defined for n in [1, 2**64 - 1] (C21's bit-scan)."""
    if not (1 <= n <= 2 ** 64 - 1):
        raise ValueError(f"clz64 is defined on n in [1, 2**64 - 1]; got {n}")
    return 64 - n.bit_length()


def round_half_up_ratio(numerator: int, denominator: int) -> int:
    """round_half_up(numerator / denominator) -- ties round UP (toward +infinity) --
    exact. denominator must be > 0; numerator may be any sign, but C19's own domain never
    calls this with a negative numerator (Dn > 0 always)."""
    if denominator <= 0:
        raise ValueError(f"round_half_up_ratio requires denominator > 0; got {denominator}")
    return (2 * numerator + denominator) // (2 * denominator)


def round_half_away_from_zero_ratio(numerator: int, denominator: int) -> int:
    """round_half_away_from_zero(numerator / denominator), sign-symmetric -- exact.
    denominator must be > 0. This is C22's composite tie rule (D-SLM52: C3's
    RoundingDivideByPOT precedent, NOT C2's toward-+infinity rule)."""
    if denominator <= 0:
        raise ValueError(
            f"round_half_away_from_zero_ratio requires denominator > 0; got {denominator}"
        )
    magnitude = (2 * abs(numerator) + denominator) // (2 * denominator)
    return magnitude if numerator >= 0 else -magnitude


# --- C20 -------------------------------------------------------------------------


def max_abs_reduce_oracle(x: Sequence[int]) -> int:
    """C20's oracle: D' = max(max_i|x_i|, 1). Fraction/int arithmetic has no width to
    overflow, so this is exact by construction -- the property intmath.max_abs_reduce's
    int64-widened-before-the-abs pin exists to reproduce, not merely approximate, at the
    D = 2**31 (INT32_MIN-present) boundary.

    Raises ValueError on an empty sequence. NOTE (Japp audit item 13 / matrix note): the
    real zero-length rejection is a loader-level, dim-2/dim-5 trust-boundary cell, not a
    runtime-reduction cell -- this oracle's own empty-input rejection exists only so it
    fails loudly rather than returning a wrong default if a packet passes it one by
    mistake, and no rung-1 packet in this pass targets it as a cell.
    """
    if len(x) == 0:
        raise ValueError(
            "max_abs_reduce_oracle is undefined on an empty sequence "
            "(zero-length rejection is a loader-level cell, out of scope here)"
        )
    return max(max(abs(int(v)) for v in x), 1)


# --- C21 -------------------------------------------------------------------------


def normalize_scale_oracle(d_prime: int) -> tuple[int, int]:
    """C21's oracle: (Dn, s) from D' in [1, 2**31] (2**31 inclusive -- C20's own domain
    attains it exactly via an INT32_MIN-present token; that is C21's one negative-s edge,
    not an out-of-domain input)."""
    if not (1 <= d_prime <= 2 ** 31):
        raise ValueError(f"normalize_scale_oracle is defined on D' in [1, 2**31]; got {d_prime}")
    p = 63 - clz64(d_prime)
    s = 30 - p
    dn = (d_prime << s) if s >= 0 else (d_prime >> 1)
    assert 2 ** 30 <= dn < 2 ** 31, f"C21 postcondition violated: Dn={dn} for D'={d_prime}"
    return dn, s


# --- C19 -------------------------------------------------------------------------


def reciprocal_oracle(dn: int) -> int:
    """C19's oracle: R = round_half_up(2**62 / Dn) -- the mathematically unique
    correctly-rounded integer (D-SLM49) that the pinned 3-Newton-iteration +
    2-correction-step construction must reproduce EXACTLY, not merely approximate."""
    if not (2 ** 30 <= dn < 2 ** 31):
        raise ValueError(f"reciprocal_oracle is defined on Dn in [2**30, 2**31); got {dn}")
    return round_half_up_ratio(1 << 62, dn)


def uncorrected_newton_reciprocal(dn: int, iterations: int = 3) -> int:
    """The NEGATIVE CONTROL construction G-8(2) requires the suite to FAIL: fixed-count
    Newton with NO correction step, seeded with the classic degree-1 minimax
    approximation to 1/d on d in [0.5, 1) -- coefficients 48/17 and 32/17 (Laplace's own
    named seed family, `dynamic-pertoken-scale-reciprocal-solve-2026-07-16.md`,
    "integer 48/17-32/17 seed... no correction step... refuted").

    Computed here in exact Fraction arithmetic rather than the Laplace harness's int64
    fixed-point emulation -- sufficient to demonstrate the DEFECT CLASS (Newton's
    intrinsic approximation error at 3 iterations falls far short of the ~62 bits of
    precision R needs), not a bit-exact reproduction of the eventual C++ kernel's rounding
    at every intermediate step. Pinned in this pass (Curie) precisely because no bit-exact
    int64 harness for the negative control exists yet; see the design doc's
    "Pinned in this pass" list.

    Dn in [2**30, 2**31) normalizes to d = Dn / 2**31 in [0.5, 1). Newton iterates
    y <- y*(2 - d*y) toward 1/d; R_estimate = round_half_up(2**31 * y).
    """
    if not (2 ** 30 <= dn < 2 ** 31):
        raise ValueError(f"uncorrected_newton_reciprocal is defined on Dn in [2**30, 2**31); got {dn}")
    d = Fraction(dn, 2 ** 31)
    y = Fraction(48, 17) - Fraction(32, 17) * d
    for _ in range(iterations):
        y = y * (2 - d * y)
    estimate = 2 ** 31 * y
    return round_half_up_ratio(estimate.numerator, estimate.denominator)


# --- C22 -------------------------------------------------------------------------


def requant_code_oracle(x_i: int, r: int, s: int) -> int:
    """C22's oracle, R-COMPOSED semantics (the pinned rule, D-SLM52):
    q_i = clamp(round_half_away_from_zero(x_i * 127 * R / 2**(62-s)), -127, 127).

    `62 - s` is always non-negative over C21's domain (s <= 30), so the division is
    always by a positive power of two; no left-shift branch is needed here (unlike the
    Loki probes, which handled a negative-exponent branch defensively -- this oracle
    proves that branch is unreachable at the type level via `round_half_away_from_zero_ratio`'s
    own denominator > 0 guard).
    """
    exponent = 62 - s
    if exponent < 0:
        raise ValueError(f"C22's exponent (62 - s) must be non-negative over C21's domain; got s={s}")
    q = round_half_away_from_zero_ratio(x_i * 127 * r, 1 << exponent)
    return max(-127, min(127, q))


def requant_code_single_rounding_oracle(x_i: int, d_prime: int) -> int:
    """The OTHER textually-named candidate (S6.2's own prose, NOT the pin): the exact
    rational x_i*127/D', rounded once, R and s never consulted. Retained as an oracle
    function (not a conformance target) because G-8(6a)/(6b)'s tie corpora exist to prove
    the pinned R-composed rule diverges from this one at constructed ties -- the negative
    control for the tie-corpus packets, not an alternate correct answer."""
    q = round_half_away_from_zero_ratio(x_i * 127, d_prime)
    return max(-127, min(127, q))


def candidate_pair(x: int, d_prime: int, dn: int, s: int, r: int) -> tuple[int, int]:
    """(R-composed, single-rounding) -- both C22 candidates at one (x, D') point, promoted
    from loki_probes/c22_adversarial_sweep.py's `candidates_at`. Used by the tie-corpus
    packets (5, 6) to assert the pinned rule (index 0) against the corpus AND to show the
    unpinned alternative (index 1) actually differs at the constructed tie -- the
    negative-control shape for those cells."""
    return requant_code_oracle(x, r, s), requant_code_single_rounding_oracle(x, d_prime)


def positive_tie_boundary_x(d_prime: int, k: int) -> int:
    """The integer NEAREST the exact half-integer tie of x*127/D' for POSITIVE output
    code k in [0, 126] -- promoted from c22_adversarial_sweep.py's boundary construction
    (`boundary = (2*k+1)*d_prime; x_center = boundary // (2*127)`).

    TIE-ADJACENT, not exact (amended 2026-07-16, bench-measured): the boundary
    (2k+1)*D'/254 is usually not integer-attainable, so the returned x usually does NOT
    sit exactly on the k+0.5 tie -- use `is_exact_composite_tie` to test genuine exactness.
    Test the integers within +/-2 of this value (a rounding disagreement between the two
    C22 candidates, if one exists, lives in that neighbourhood)."""
    if not (0 <= k <= 126):
        raise ValueError(f"positive_tie_boundary_x is defined for k in [0, 126]; got {k}")
    boundary = (2 * k + 1) * d_prime
    return boundary // (2 * 127)


def negative_tie_boundary_x(d_prime: int, k: int) -> int:
    """The mirrored NEGATIVE-side tie boundary for output code -(k+1), k in [0, 126]
    (G-8(6b) / S6.2's sign-symmetry argument: at a negative EXACT tie the pinned
    away-from-zero rule gives -(k+1) where toward-+infinity gives -k).

    TIE-ADJACENT, not exact, same caveat as positive_tie_boundary_x (amended 2026-07-16):
    most returned points are near but not on a mathematical tie, and at a non-exact point
    the candidate tie rules AGREE -- discrimination holds only at points where
    `is_exact_composite_tie(x, r, s)` is True (measured over the pinned sweep: 151 of
    603,504 in-domain boundary points are exact; all 151 discriminate; zero non-exact
    points do). Filter with the predicate before asserting discrimination."""
    return -positive_tie_boundary_x(d_prime, k)


def is_exact_composite_tie(x_i: int, r: int, s: int) -> bool:
    """True iff C22's un-rounded composite ratio x_i*127*R / 2**(62-s) sits EXACTLY on a
    half-integer tie -- the one place candidate tie rules can disagree.

    With num = x_i*127*r and den = 2**(62-s): exact tie iff (2*num) % (2*den) == den.
    Equivalently v2(x_i) + v2(R) == 61 - s (2-adic valuations; 127 is odd). The predicate
    is independent of BOTH rounding rules -- it tests where the ratio sits, not what any
    rule returns -- which is what makes filter-by-predicate-then-assert-discrimination a
    genuine control rather than a tautology (packet 6, amended 2026-07-16)."""
    exponent = 62 - s
    if exponent < 0:
        raise ValueError(f"C22's exponent (62 - s) must be non-negative over C21's domain; got s={s}")
    num = x_i * 127 * r
    den = 1 << exponent
    return (2 * num) % (2 * den) == den
