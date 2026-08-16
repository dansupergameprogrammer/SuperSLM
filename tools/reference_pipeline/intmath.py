"""Integer arithmetic primitives for the T-066 spike harness.

Realizes SuperSLM_Plan.md §6.2 (requantization pinned to the gemmlowp reference
definitions) and §6.3's i-sqrt (digit recurrence, D-SLM34) and i-exp (the I-BERT
second-order polynomial integer exponential, on max-shifted logits).

Every value on the reproducible path is an exact Python ``int``. Floats appear
only as offline constants — the activation scale and the polynomial's
coefficients — in the same discipline §6.4 states for the RoPE tables: the
derived quantities are integers, and the arithmetic that consumes them is
integer.

Test-design record: Claude/Curie/superslm-t066-harness-test-design-2026-07-14.md
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = [
    "INT32_MIN",
    "INT32_MAX",
    "saturating_rounding_doubling_high_mul",
    "rounding_divide_by_pot",
    "multiply_by_quantized_multiplier",
    "I_SQRT_ITERATIONS",
    "i_sqrt",
    "i_sqrt_trace",
    "I_EXP_CLIP_N",
    "i_exp",
    "i_exp_ln2_quantum",
    "shift_by_max",
    "clz64",
    "max_abs_reduce",
    "normalize_scale",
    "DYNAMIC_RECIPROCAL_NEWTON_ITERATIONS",
    "DYNAMIC_RECIPROCAL_CORRECTION_STEPS",
    "dynamic_scale_reciprocal",
    "dynamic_scale_reciprocal_trace",
    "requant_token_code",
    "requant_token_code_trace",
    "scale_mul",
    "carried_scale_product",
    "residual_reconcile",
    "bias_reconcile",
    "iexp_scale_constants",
    "i_exp_from_constants",
]

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1


# --- §6.2 requantization ------------------------------------------------------


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    """gemmlowp's ``SaturatingRoundingDoublingHighMul``: 2*a*b / 2**32, rounded.

    Ties go toward positive infinity, not away from zero. gemmlowp's negative
    nudge is ``1 - (1 << 30)`` — one short of the symmetric ``-(1 << 30)`` — and
    the C++ division truncates toward zero, which together make the negative
    branch round the same direction as the positive one. The plan's §15 gloss
    ("ties away from zero") describes `rounding_divide_by_pot` and is wrong for
    this primitive; §6.2's normative deferral to the gemmlowp reference is what
    is implemented here.

    ``(INT32_MIN, INT32_MIN)`` is the only int32 pair whose defined result
    exceeds INT32_MAX, and it saturates.
    """
    result = (a * b + (1 << 30)) >> 31
    return INT32_MAX if result > INT32_MAX else result


def rounding_divide_by_pot(x: int, exponent: int) -> int:
    """gemmlowp's ``RoundingDivideByPOT``: x / 2**exponent, ties away from zero.

    Defined for ``exponent`` in [0, 31]; exponent 0 is the identity. The
    threshold carries the ``+1`` on the negative branch, which is what turns the
    floor-shift's half-up behaviour into half-away-from-zero.
    """
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def multiply_by_quantized_multiplier(x: int, quantized_multiplier: int, shift: int) -> int:
    """§6.2's requant: the two primitives composed, in that order, with both roundings."""
    return rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(x, quantized_multiplier), shift
    )


# --- §6.3 i-sqrt --------------------------------------------------------------

# §6.8 C5. One iteration per base-4 digit of an int64 radicand: the count follows
# from the type, not from a convergence measurement, so it cannot be data-dependent.
I_SQRT_ITERATIONS = 32

# §6.8 C6. The first digit's weight, 4**31.
_I_SQRT_INITIAL_BIT = 1 << 62


def _i_sqrt_iterates(n: int) -> list[int]:
    """The root after each of the 32 digit steps (§6.3's listing verbatim).

    Compare, subtract, shift — no division. Newton's ``n // x`` is what §18's named
    risk lands on (int64 division emulation computing wrong bits on integrated and
    mobile GPUs); the construction is digit recurrence partly to keep i-sqrt off
    that row, and a divide reintroduces the risk while still passing every value
    cell in the suite.

    All 32 iterations run unconditionally. Skipping the leading zero digits (the
    conventional ``while bit > n: bit >>= 2`` prologue) is value-correct and makes
    the op count a function of the radicand, which is exactly what §14's
    cost-determinism contract forbids. Below the radicand's leading digit the
    comparison simply takes the else branch.

    ``n == 0`` needs no guard: every digit decision takes the else branch and the
    root stays 0. A guard here would mean the construction is a divide-based one.
    """
    if n < 0:
        raise ValueError(f"i_sqrt is defined on non-negative sums (n >= 0); got n={n}")

    remainder = n
    root = 0
    bit = _I_SQRT_INITIAL_BIT
    iterates: list[int] = []
    for _ in range(I_SQRT_ITERATIONS):
        trial = root + bit
        if remainder >= trial:
            remainder -= trial
            root = (root >> 1) + bit
        else:
            root >>= 1
        bit >>= 2
        iterates.append(root)
    return iterates


def i_sqrt(n: int) -> int:
    """floor(sqrt(n)) over [0, 2**63 - 1], by restoring shift-and-subtract."""
    return _i_sqrt_iterates(n)[-1]


def i_sqrt_trace(n: int) -> list[int]:
    """`i_sqrt`'s iterate sequence — one entry per executed iteration.

    The artifact D-SLM29's cost claim is falsifiable against: the length is
    ``I_SQRT_ITERATIONS`` for every input, so a data-dependent trip count is a
    test failure rather than a reading of the source.
    """
    return _i_sqrt_iterates(n)


# --- §6.3 i-exp ---------------------------------------------------------------

# The I-BERT second-order polynomial: exp(p) ~= A*(p + B)**2 + C on [-ln2, 0].
_POLY_A = 0.3585
_POLY_B = 1.353
_POLY_C = 0.344

# The exponent decomposition's clip. I-BERT clamps the shifted logit at
# -n*q_ln2, which makes the far tail total: past the clip the result is the
# clip point's result, not a divergent shift. n = 30 is the I-BERT
# construction's value; §6.3 names the construction without restating it.
I_EXP_CLIP_N = 30

_LN2 = math.log(2)


def i_exp_ln2_quantum(scale: float) -> int:
    """floor(ln2 / scale) — the quantum the exponent decomposition is stated in."""
    return math.floor(_LN2 / scale)


def i_exp(q: int, scale: float) -> tuple[int, float]:
    """exp(q * scale) in fixed point, as ``(q_out, out_scale)``.

    ``q`` is a max-shifted logit, so it is non-positive; a positive ``q`` drives
    the decomposition's ``z`` negative and is rejected rather than shifted the
    wrong way.

    ``scale`` must leave the ln2 quantum non-zero. The decomposition is stated in
    units of ``floor(ln2 / scale)`` and divides by it, so a scale coarse enough to
    floor that quantum to 0 has no decomposition to state.

    ``q * scale`` decomposes as ``-z*ln2 + p`` with ``p`` in (-ln2, 0]. The
    polynomial evaluates ``p``'s factor in integers and the ``2**-z`` factor is
    a right shift, so ``out_scale`` does not vary with ``z``.
    """
    if q > 0:
        raise ValueError(f"i_exp is defined on max-shifted logits (q <= 0); got q={q}")

    q_ln2 = i_exp_ln2_quantum(scale)
    if q_ln2 < 1:
        raise ValueError(
            f"i_exp is defined where the ln2 quantum floor(ln2/scale) is positive "
            f"(scale < ln2 ~= 0.6931); got scale={scale}, quantum={q_ln2}"
        )
    q_b = math.floor(_POLY_B / scale)
    q_c = math.floor(_POLY_C / (_POLY_A * scale ** 2))
    out_scale = _POLY_A * scale ** 2

    clipped = max(q, -I_EXP_CLIP_N * q_ln2)
    z = -clipped // q_ln2
    q_p = clipped + z * q_ln2
    q_out = ((q_p + q_b) ** 2 + q_c) >> z
    return q_out, out_scale


def shift_by_max(logits: Sequence[int]) -> list[int]:
    """§6.3's max-subtraction: an integer op that puts the maximum at 0."""
    values = list(logits)
    if not values:
        raise ValueError("shift_by_max is undefined on an empty sequence")
    peak = max(values)
    return [int(v) - peak for v in values]


# --- §6.2 rung 1: dynamic per-token scales (C19–C22) ---------------------------


def clz64(n: int) -> int:
    """Count leading zeros over 64 bits (§6.8 C21).

    Defined for n in [1, 2^64 - 1]. The bit-scan intrinsic pinned to scalar-
    reference behavior per §18.
    """
    if not (1 <= n <= 2 ** 64 - 1):
        raise ValueError(f"clz64 is defined on n in [1, 2**64 - 1]; got {n}")
    return 64 - n.bit_length()


def max_abs_reduce(x: Sequence[int]) -> int:
    """C20: max of |x_i| widened to int64 before the abs, with max(D, 1) guard.

    D = max_i |x_i| over int32 activations, widened before abs to avoid int32
    overflow (x_i = INT32_MIN present forces |x_i| = 2^31). The reduction is
    order-free (max applies no mid-stream rounding); every element is visited
    unconditionally (no early exit). D' = max(D, 1) guards the all-zero token
    and is exact on that class.
    """
    d = 0
    for x_i in x:
        # Widen to int64 before taking absolute value
        x_i_int64 = int(x_i)
        abs_x_i = abs(x_i_int64)
        d = max(d, abs_x_i)
    # Guard the all-zero token
    d_prime = max(d, 1)
    return d_prime


def normalize_scale(d_prime: int) -> tuple[int, int]:
    """C21: normalize D' ∈ [1, 2^31] to Dn ∈ [2^30, 2^31) via pure shift.

    p = 63 − clz64(D'), s = 30 − p, Dn = s ≥ 0 ? D' << s : D' >> 1.
    Calls clz64 exactly once via the module global (cost-determinism pin).
    """
    if not (1 <= d_prime <= 2 ** 31):
        raise ValueError(f"normalize_scale is defined on D' in [1, 2**31]; got {d_prime}")

    # Call clz64 exactly once, via module global reference
    p = 63 - clz64(d_prime)
    s = 30 - p

    if s >= 0:
        dn = d_prime << s
    else:
        dn = d_prime >> 1

    assert 2 ** 30 <= dn < 2 ** 31, f"C21 postcondition violated: Dn={dn} for D'={d_prime}"
    return dn, s


# --- C19: dynamic per-token scale reciprocal (Newton + correction) --------

DYNAMIC_RECIPROCAL_NEWTON_ITERATIONS = 3
DYNAMIC_RECIPROCAL_CORRECTION_STEPS = 2


def _round_half_up_ratio(numerator: int, denominator: int) -> int:
    """Helper: round_half_up(numerator / denominator) -- ties round toward +infinity."""
    return (2 * numerator + denominator) // (2 * denominator)


def dynamic_scale_reciprocal_trace(dn: int) -> list[int]:
    """C19 trace: all 5 iterates (3 Newton + 2 correction), last == returned value.

    R = round_half_up(2^62 / Dn) for Dn ∈ [2^30, 2^31). Computed via corrected
    Newton: 3 fixed Newton iterations + 2 fixed correction steps (the C++ port
    implements the correction branch-free per the casebook; output-exactness is
    the conformance bar here).
    The seed is not pinned; any seed reaching output-exactness is conformant.
    """
    if not (2 ** 30 <= dn < 2 ** 31):
        raise ValueError(f"dynamic_scale_reciprocal is defined on Dn in [2**30, 2**31); got {dn}")

    iterates: list[int] = []

    # Seed: y_0 ≈ 1/d where d = Dn / 2^31, using 48/17 - 32/17 * d
    # In Q31 fixed-point (implicit scaling by 2^31):
    # y_0 = round_half_up((48 * 2^31 - 32 * Dn) / 17)
    c32 = _round_half_up_ratio(48 << 31, 17)
    c32_2 = _round_half_up_ratio(32 << 31, 17)
    y = c32 - ((c32_2 * dn) >> 31)
    # Seed is not counted in the 5 iterates

    # Newton iterations: y <- y * (2 - d*y) = y * (2^32 - Dn*y) / 2^31
    for _ in range(DYNAMIC_RECIPROCAL_NEWTON_ITERATIONS):
        # Compute: y * (2^32 - (Dn * y >> 31)) >> 31
        dn_y = (dn * y) >> 31
        delta = (1 << 32) - dn_y
        y = (y * delta) >> 31
        iterates.append(y)

    # Correction steps: ensure y = round_half_up(2^62 / Dn)
    # Predicate: q is correctly rounded iff -Dn <= 2*(2^62 - q*Dn) < Dn
    for _ in range(DYNAMIC_RECIPROCAL_CORRECTION_STEPS):
        residual_2x = 2 * ((1 << 62) - y * dn)
        if residual_2x >= dn:  # Residual out of range on the high side
            y += 1
        elif residual_2x < -dn:  # Residual out of range on the low side
            y -= 1
        iterates.append(y)

    return iterates


def dynamic_scale_reciprocal(dn: int) -> int:
    """C19: R = round_half_up(2^62 / Dn) for Dn ∈ [2^30, 2^31)."""
    return dynamic_scale_reciprocal_trace(dn)[-1]


# --- C22: per-token requant composition --------------------------------

def _round_half_away_from_zero_ratio(numerator: int, denominator: int) -> int:
    """Helper: round_half_away_from_zero(numerator / denominator), sign-symmetric."""
    if denominator <= 0:
        raise ValueError(f"denominator must be > 0; got {denominator}")
    magnitude = (2 * abs(numerator) + denominator) // (2 * denominator)
    return magnitude if numerator >= 0 else -magnitude


def requant_token_code_trace(x_i: int, r: int, s: int) -> list[int]:
    """C22 trace: decomposition steps for q_i = clamp(round_half_away_from_zero(...), -127, 127).

    R-composed semantics (D-SLM52):
    q_i = clamp(round_half_away_from_zero((x_i · 127 · R) / 2^(62−s)), −127, 127)

    The trace represents the decomposition of the wide intermediate |x_i*127*r|.
    Returns a constant-length list (one entry per decomposition step) ending with the final value.
    """
    if s > 62:
        raise ValueError(f"C22's exponent (62 - s) must be non-negative; got s={s}")

    exponent = 62 - s
    product = x_i * 127 * r

    # This sim computes the product exactly in arbitrary-precision Python
    # integers; the wide-intermediate obligation is the C++ port's. Conformance
    # is by output exactness, and the trace records the two fixed steps
    # (round, clamp).

    decomposed: list[int] = []

    # Round the composite: round_half_away_from_zero(product / 2^exponent)
    q = _round_half_away_from_zero_ratio(product, 1 << exponent)
    decomposed.append(q)

    # Clamp to [-127, 127]
    q_clamped = max(-127, min(127, q))
    decomposed.append(q_clamped)

    return decomposed


def requant_token_code(x_i: int, r: int, s: int) -> int:
    """C22: q_i = clamp(round_half_away_from_zero((x_i · 127 · R) / 2^(62−s)), −127, 127).

    R-composed semantics (D-SLM52): R from C19, s from C21, ties away from zero (C3).
    """
    trace = requant_token_code_trace(x_i, r, s)
    return trace[-1]


# --- C26: canonical carried-scale runtime product + residual reconciliation ----


def scale_mul(m_a: int, e_a: int, m_b: int, e_b: int) -> tuple[int, int]:
    """C26's runtime scale product: `(m_a, e_a) * (m_b, e_b)` in canonical (m, e) form.

    ONE C2-class high-mul on the mantissas (`saturating_rounding_doubling_high_mul`,
    ties toward +infinity), then an EXACT left-shift renormalization back into
    `[2^30, 2^31)` — the renormalization moves whole bits and rounds nothing further;
    only the high-mul rounds.
    """
    high = saturating_rounding_doubling_high_mul(m_a, m_b)
    e = e_a + e_b + 31
    if high < (1 << 30):
        high <<= 1
        e -= 1
    assert (1 << 30) <= high < (1 << 31), f"C26 postcondition violated: m={high}"
    return high, e


def carried_scale_product(factors) -> tuple[int, int]:
    """D-SLM57: the multi-factor runtime scale product, LEFT-ASSOCIATED in composition
    order, the incoming carried scale first — `((S_in × C_site) × D'-factor) × …`, one
    C1/C2 high-mul per multiplication with an exact single-shift renormalization
    (`scale_mul` per step). The association order is a pin, not a preference: the
    high-mul is not associative (executed, Curie A-5 §14.1 — left vs right diverge on
    40.5% of seeded triples), so this module-global helper exists to make the order
    instrumentable (the rung-1 clz64 / P-1 pin class) rather than an inlined convention.

    `factors` is a sequence of canonical `(m, e)` pairs in composition order; each step
    calls `scale_mul` through the module-global name.
    """
    factors = list(factors)
    if not factors:
        raise ValueError("carried_scale_product is defined on at least one factor")
    m, e = factors[0]
    for m_b, e_b in factors[1:]:
        m, e = scale_mul(m, e, m_b, e_b)
    return m, e


def residual_reconcile(branch_code: int, m_b: int, r_h: int, e_b: int, e_h: int) -> int:
    """C26's residual-add reconciliation composite: the branch code rescaled into the
    stream's home scale —

        round_half_away_from_zero((branch_code · m_b · R_h) / 2^(62 − (e_b − e_h)))

    `R_h` is C19's reciprocal over the home mantissa `m_h` (`dynamic_scale_reciprocal(m_h)`),
    computed once per row and passed in here rather than recomputed per element. Ties away
    from zero (C3) — the composite is a rounding divide by a power of two, C13/C22's
    precedent. A negative composite exponent is an EXACT left shift, no rounding involved.
    """
    numerator = branch_code * m_b * r_h
    k = 62 - (e_b - e_h)
    if k >= 0:
        return _round_half_away_from_zero_ratio(numerator, 1 << k)
    return numerator << (-k)


# --- C28: projection bias reconciliation ----------------------------------------


def bias_reconcile(b: int, q_b: int, r_a: int, e_a: int) -> int:
    """C28's CORRECTED bias reconciliation (second-pass S2-1): the offline int bias `B[j]`,
    stored in Q-format `q_b` at the projection's reference scale, folded into the runtime
    accumulator (post-C25 fold, pre-C22 requant) —

        round_half_away_from_zero((B[j] · R_a) / 2^(q_B + 62 + e_a))

    `R_a` is C19's reciprocal over the incoming activation's carried mantissa, `e_a` its
    exponent — the exponent ADDS `e_a` and CARRIES the stored Q-format (the same convention
    C30's `q_b` uses). `B[j]` is signed, so the away-from-zero tie rule (C3) is load-bearing
    here, unlike C30's positive-only floor shifts.
    """
    numerator = b * r_a
    k = q_b + 62 + e_a
    if k >= 0:
        return _round_half_away_from_zero_ratio(numerator, 1 << k)
    return numerator << (-k)


# --- C30: per-token i-exp constant derivation (R-composed) ----------------------


def iexp_scale_constants(m: int, e: int, ln2_q: int, ln2_fmt: int,
                         b_q: int, b_fmt: int, ca_q: int, ca_fmt: int
                         ) -> tuple[int, int, int]:
    """C30's R-COMPOSED derivation of i-exp's per-token integer constants from the
    canonical carried scale `(m, e)` — the definition, not an approximation of the exact
    division (D-SLM52's precedent):

        R_m   = C19 over m
        q_ln2 = (LN2_Q · R_m)  >> (ln2_fmt + 62 + e)
        q_b   = (B_Q   · R_m)  >> (b_fmt   + 62 + e)
        q_c   = (CA_Q  · R_m²) >> (ca_fmt  + 124 + 2e)

    Truncating floor shifts: every operand above is positive (the coefficient integers ride
    C7/C8's positive-emission pin, N2-5), so the shift IS floor and no tie rule applies. A
    negative derivation shift is the FINE-scale (overflow-end) rejection — `q_c`'s threshold
    is earliest and sets the derivation's domain — and raises `ValueError`.
    """
    if not (ln2_q > 0 and b_q > 0 and ca_q > 0):
        raise ValueError(
            "C30's floor-shift claim holds on positive coefficients only (N2-5); "
            f"got ln2_q={ln2_q}, b_q={b_q}, ca_q={ca_q}"
        )
    shift_ln2 = ln2_fmt + 62 + e
    shift_b = b_fmt + 62 + e
    shift_c = ca_fmt + 124 + 2 * e
    if min(shift_ln2, shift_b, shift_c) < 0:
        raise ValueError(
            f"C30 fine-scale rejection: negative derivation shift at e={e} "
            f"(ln2={shift_ln2}, b={shift_b}, c={shift_c})"
        )
    r_m = dynamic_scale_reciprocal(m)
    return (
        (ln2_q * r_m) >> shift_ln2,
        (b_q * r_m) >> shift_b,
        (ca_q * r_m * r_m) >> shift_c,
    )


def i_exp_from_constants(q: int, q_ln2: int, q_b: int, q_c: int) -> int:
    """`i_exp`'s integer polynomial core, factored out so a caller holding C30-derived
    per-token constants (`iexp_scale_constants`) never has to route back through `i_exp`'s
    own float-`scale` derivation of `(q_ln2, q_b, q_c)` — that derivation is exactly the
    float `floor(ln2/scale)`-class arithmetic C30 exists to keep off a per-token dynamic
    scale's reproducible path. Same I-BERT decomposition as `i_exp`, `out_scale` omitted
    (C30: it is never computed at runtime — both nonlinear consumers are same-scale ratios
    and it cancels by construction).
    """
    if q > 0:
        raise ValueError(f"i_exp is defined on max-shifted logits (q <= 0); got q={q}")
    if q_ln2 < 1:
        raise ValueError(f"i_exp requires a positive ln2 quantum; got q_ln2={q_ln2}")
    clipped = max(q, -I_EXP_CLIP_N * q_ln2)
    z = -clipped // q_ln2
    q_p = clipped + z * q_ln2
    return ((q_p + q_b) ** 2 + q_c) >> z
