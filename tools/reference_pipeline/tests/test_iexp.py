"""Plan §6.3 — I-BERT second-order polynomial integer exponential (i-exp).

The construction: shift the logits by their max (an integer op), decompose
q*S = -z*ln2 + p with p in (-ln2, 0], approximate exp(p) with the second-order
polynomial, and factor 2^-z back in by a right shift.

Oracle strategy. The accuracy claim is an achievement claim, so it gets a feature
oracle grounded in `math.exp` — the definition — rather than a recode of the
polynomial. Two regions are treated differently, because the construction behaves
differently in them:

  * the z == 0 branch (q in [-q_ln2, 0]) is the polynomial acting alone, with no
    shift loss. Absolute error there is bounded by the polynomial's own
    approximation error plus scale quantization, and is asserted against math.exp.
  * the shifted branch (z > 0) loses precision to the `>> z` truncation, without
    bound in relative terms. Asserting an error bound there would either be vacuous
    or would encode a tolerance the plan does not state. It is pinned instead by the
    exponent law, which is exact.

Tolerances here are `red-uncalibrated` (finding C-4): §6.3 states no op-level
tolerance for i-exp, and §13.2 defers "stated tolerances" to spike data. The bounds
below are derived from the construction and are wide enough to pass a correct
implementation with headroom, while a dropped `+c` term (measured 3.5e-1) or a wrong
`b` coefficient (measured 1.5e-1) miss them by more than an order of magnitude.
"""

import math

import pytest

from conftest import api

MODULE = "reference_pipeline.intmath"

# Activation scales in the range a W8A8 attention pipeline calibrates into. The
# accuracy cells are stated over this set; the structural cells are stated over the
# whole domain.
SCALES = [0.005, 0.01]
WIDE_SCALES = [0.005, 0.01, 0.02, 0.05]

# Derived provisional bound (finding C-4). Correct construction measures 3.8e-3.
I_EXP_ABS_TOL = 5e-3


def value_of(pair):
    """(q_out, out_scale) -> the real number the fixed-point pair represents."""
    q_out, out_scale = pair
    return q_out * out_scale


# ==============================================================================
# The achievement claim — i-exp computes exp (feature oracle vs math.exp)
# ==============================================================================

@pytest.mark.parametrize("scale", SCALES)
def test_i_exp_of_zero_is_one(scale):
    """exp(0) == 1. The single input whose truth needs no tolerance argument to
    state, and the one a wrong polynomial constant misses immediately."""
    i_exp = api(MODULE, "i_exp")
    assert abs(value_of(i_exp(0, scale)) - 1.0) <= I_EXP_ABS_TOL


@pytest.mark.parametrize("scale", SCALES)
def test_i_exp_matches_exp_on_the_unshifted_branch(scale):
    """The polynomial branch, asserted pointwise against math.exp.

    Grounded in the definition of the exponential, not in a recode of the I-BERT
    polynomial — a test that recoded the polynomial would pass an implementation
    that recoded the same wrong coefficients.
    """
    i_exp, i_exp_ln2_quantum = api(MODULE, "i_exp", "i_exp_ln2_quantum")
    q_ln2 = i_exp_ln2_quantum(scale)
    worst = (0.0, None)
    for q in range(0, -q_ln2 - 1, -1):
        error = abs(value_of(i_exp(q, scale)) - math.exp(q * scale))
        if error > worst[0]:
            worst = (error, q)
    assert worst[0] <= I_EXP_ABS_TOL, (
        f"scale={scale}: worst absolute error {worst[0]:.3e} at q={worst[1]}, "
        f"bound {I_EXP_ABS_TOL:.3e}"
    )


@pytest.mark.parametrize("scale", SCALES)
def test_i_exp_obeys_the_exponent_law(scale):
    """exp(x - k*ln2) == exp(x) * 2^-k, which the construction realises as a right
    shift by k. Exact — no tolerance — so it holds where the accuracy cells cannot
    reach, and it is the cell a missing or misplaced `>> z` fails.
    """
    i_exp, i_exp_ln2_quantum = api(MODULE, "i_exp", "i_exp_ln2_quantum")
    q_ln2 = i_exp_ln2_quantum(scale)
    for q in range(0, -q_ln2, -1):
        base_q_out, base_scale = i_exp(q, scale)
        for k in range(1, 12):
            shifted_q_out, shifted_scale = i_exp(q - k * q_ln2, scale)
            assert shifted_q_out == base_q_out >> k, f"scale={scale} q={q} k={k}"
            assert shifted_scale == base_scale, "out_scale must not vary with z"


@pytest.mark.parametrize("scale", SCALES)
def test_i_exp_ln2_quantum_is_floor_ln2_over_scale(scale):
    """q_ln2 is the quantum the decomposition is stated in. Every branch boundary in
    this file is placed by it, so it is pinned rather than assumed."""
    i_exp_ln2_quantum = api(MODULE, "i_exp_ln2_quantum")
    quantum = i_exp_ln2_quantum(scale)
    assert type(quantum) is int
    assert quantum == math.floor(math.log(2) / scale)


# ==============================================================================
# Domain extremes
# ==============================================================================

@pytest.mark.parametrize("scale", WIDE_SCALES)
def test_i_exp_is_never_negative(scale):
    """exp is positive everywhere. A negative q_out means the fixed-point pipeline
    wrapped, and a wrapped probability poisons the whole softmax."""
    i_exp, i_exp_ln2_quantum = api(MODULE, "i_exp", "i_exp_ln2_quantum")
    q_ln2 = i_exp_ln2_quantum(scale)
    for q in range(0, -35 * q_ln2 - 1, -max(1, q_ln2 // 8)):
        q_out, _ = i_exp(q, scale)
        assert q_out >= 0, f"i_exp({q}, {scale}) -> {q_out}"


@pytest.mark.parametrize("scale", WIDE_SCALES)
def test_i_exp_is_monotone_non_increasing_as_q_falls(scale):
    """exp is strictly increasing, so the quantized form must be non-decreasing in q.
    A sign error in the z decomposition shows up here as a reversal."""
    i_exp, i_exp_ln2_quantum = api(MODULE, "i_exp", "i_exp_ln2_quantum")
    q_ln2 = i_exp_ln2_quantum(scale)
    previous = None
    for q in range(0, -8 * q_ln2 - 1, -1):
        q_out, _ = i_exp(q, scale)
        if previous is not None:
            assert q_out <= previous, f"i_exp({q}, {scale}) > i_exp({q + 1}, {scale})"
        previous = q_out


@pytest.mark.parametrize("scale", WIDE_SCALES)
def test_i_exp_far_tail_underflows_to_zero_without_wrapping(scale):
    """The negative domain extreme. exp(-large) is below the fixed-point resolution,
    so the defined result is 0 — reached by saturating clip and truncation, never by
    an exception and never by a wrap to a large or negative value."""
    i_exp = api(MODULE, "i_exp")
    for q in [-10 ** 4, -10 ** 6, -10 ** 9, -(2 ** 31)]:
        q_out, out_scale = i_exp(q, scale)
        assert q_out == 0, f"i_exp({q}, {scale}) -> {q_out}, expected 0"
        assert out_scale > 0


@pytest.mark.parametrize("scale", WIDE_SCALES)
def test_i_exp_is_clipped_below_the_pinned_bound(scale):
    """I-BERT clips the exponent decomposition at N. Past the clip the result must be
    constant, not divergent — the property that makes the far tail total.

    **The VALUE of N is not asserted, deliberately.** §6.8 C8 ("i-exp clip bound `N`") is
    **PIN OWED at S2**: the table does not yet carry a number, and §6.8's rule is that "a
    constant in the reproducible path that is not in this table is a specification defect,
    not an implementer's choice". While the row is owed there is nothing to assert against
    — any exact integer is as conformant as any other — so this cell pins the clip's
    *behaviour* and reads N from the module rather than restating a guess.

    The 2026-07-14 suite asserted `clip_n == 30`. That is the C-5 defect shape (record
    P-1): a test that pins a reading on an open question becomes a lock on the answer it
    guessed. C-5 was ruled against that suite's guess and the cell did not move; C8 is
    still open, so the guess is removed before it can harden. **When S2 pins C8, this cell
    gains one assertion against the table's number** — and nothing else moves.
    """
    i_exp, i_exp_ln2_quantum, clip_n = api(MODULE, "i_exp", "i_exp_ln2_quantum", "I_EXP_CLIP_N")
    assert type(clip_n) is int and clip_n > 0, "the clip bound is an exact positive integer"
    q_ln2 = i_exp_ln2_quantum(scale)
    at_clip = i_exp(-clip_n * q_ln2, scale)
    for q in [-(clip_n + 1) * q_ln2, -(clip_n + 8) * q_ln2, -10 ** 7]:
        assert i_exp(q, scale) == at_clip, f"i_exp({q}, {scale}) differs from the clip point"


@pytest.mark.parametrize("scale", SCALES)
def test_i_exp_rejects_positive_input(scale):
    """The construction is defined on shifted logits, which are <= 0 after the
    max-subtraction. A positive q is a caller defect and its handling is defined,
    not undefined — an unguarded positive q drives z negative and the shift with it.
    """
    i_exp = api(MODULE, "i_exp")
    for q in [1, 2, 10 ** 6]:
        with pytest.raises(ValueError):
            i_exp(q, scale)


# ==============================================================================
# Purity — the arithmetic is integer (Forge W6)
# ==============================================================================

@pytest.mark.parametrize("scale", SCALES)
def test_i_exp_returns_an_exact_int_and_a_positive_scale(scale):
    i_exp = api(MODULE, "i_exp")
    for q in [0, -1, -7, -100, -10 ** 5]:
        q_out, out_scale = i_exp(q, scale)
        assert type(q_out) is int, f"i_exp({q}, {scale}) q_out is {type(q_out).__name__}"
        assert isinstance(out_scale, float) and out_scale > 0


@pytest.mark.parametrize("scale", SCALES)
def test_i_exp_is_pure(scale):
    i_exp = api(MODULE, "i_exp")
    inputs = [0, -1, -50, -500, -5000]
    assert [i_exp(q, scale) for q in inputs] == [i_exp(q, scale) for q in inputs]


# ==============================================================================
# The max-subtraction — §6.3: "on shifted logits (max-subtraction is an integer op)"
# ==============================================================================

def test_shift_by_max_puts_the_maximum_at_zero():
    shift_by_max = api(MODULE, "shift_by_max")
    for logits in [[0], [5], [-5], [3, 1, 4, 1, 5], [-7, -7, -9], [2 ** 31 - 1, 0, -(2 ** 31)]]:
        shifted = shift_by_max(logits)
        assert max(shifted) == 0, f"shift_by_max({logits}) -> {shifted}"


def test_shift_by_max_makes_every_logit_non_positive():
    """The precondition i_exp's domain depends on."""
    shift_by_max = api(MODULE, "shift_by_max")
    for logits in [[3, 1, 4, 1, 5], [-7, -7, -9], [0, 0, 0], [2 ** 31 - 1, -(2 ** 31)]]:
        assert all(v <= 0 for v in shift_by_max(logits))


def test_shift_by_max_preserves_differences_exactly():
    """A subtraction, not a rescale: every pairwise gap survives unchanged. This is
    what makes the shift a no-op on the softmax it feeds."""
    shift_by_max = api(MODULE, "shift_by_max")
    logits = [3, 1, 4, 1, 5, -9, 2, 6]
    shifted = shift_by_max(logits)
    assert len(shifted) == len(logits)
    for i in range(len(logits)):
        for j in range(len(logits)):
            assert shifted[i] - shifted[j] == logits[i] - logits[j]


def test_shift_by_max_returns_exact_ints():
    shift_by_max = api(MODULE, "shift_by_max")
    for value in shift_by_max([3, 1, 4, 1, 5]):
        assert type(value) is int, f"{value!r} is {type(value).__name__}, not a plain int"


def test_shift_by_max_rejects_an_empty_sequence():
    shift_by_max = api(MODULE, "shift_by_max")
    with pytest.raises(ValueError):
        shift_by_max([])
