"""Plan §6.3 — integer square root over int64 sums, by restoring digit recurrence.

The construction is **restoring shift-and-subtract digit recurrence**, run for exactly
32 unconditional iterations (Dan, 2026-07-14 — replacing the I-BERT Newton iteration
this suite's findings C-1 and C-2 killed):

    res = 0; bit = 1 << 62
    repeat exactly 32 times:            # unconditional, data-independent
        if n >= res + bit: n -= res + bit; res = (res >> 1) + bit
        else:              res >>= 1
        bit >>= 2
    return res

Each iteration resolves one base-4 digit of the root, top down. 32 iterations cover
`bit` from 2^62 down to 2^0, which spans the whole int64 domain structurally rather
than empirically — so the two properties are no longer in tension:

  * There is no iterate-to-fixpoint, so C-1's idempotence question does not arise.
  * 32 steps is the digit count, not a convergence estimate, so C-2's "converges in
    <= 4 steps" defect cannot recur.

Two claims are pinned, and they are independent:

1. VALUE. `i_sqrt(n) == floor(sqrt(n))`, oracled against `math.isqrt` and against the
   defining inequality — the mathematical definition, never a recode of the recurrence.
2. COST. D-SLM29: the trip count is fixed and data-independent. The `if` above is a
   select on the digit, not a loop condition; the iteration count is a constant.

Testing only (1) would let a data-dependent trip count ship; testing only (2) would let
a fixed-count loop that returns the wrong number ship. Both, or neither is proven.
"""

import math

import pytest

from conftest import api

MODULE = "reference_pipeline.intmath"

INT64_MAX = 2 ** 63 - 1


def _domain():
    """Inputs spanning the domain and the recurrence's own structure.

    Digit recurrence resolves one base-4 digit per iteration, so powers of four are its
    carry boundaries — the values at which a digit decision flips. Perfect squares and
    their neighbours are where an off-by-one in the restoring subtraction lands. The
    int64 ceiling is the last input the 32-digit budget must still cover.
    """
    values = set(range(0, 130))
    for k in range(0, 32):                      # base-4 digit boundaries: the radix
        for delta in (-1, 0, 1):
            values.add(4 ** k + delta)
    for k in range(0, 63):                      # power-of-two boundaries
        for delta in (-1, 0, 1):
            values.add(2 ** k + delta)
    for root in [1, 2, 3, 7, 255, 256, 65535, 65536, 46341, 3037000499]:
        for delta in (-1, 0, 1):                # perfect squares +/- 1
            values.add(root * root + delta)
    values.update({
        10 ** 6, 10 ** 6 + 1,
        1085241185,             # C-2 witness: I-BERT Newton needed 6 steps here (int32)
        1736392818365009963,    # C-2 witness: I-BERT Newton needed 6 steps here (int64)
        2 ** 62,                # C-2 witness; also the recurrence's initial `bit`
        INT64_MAX - 1, INT64_MAX,
    })
    return sorted(v for v in values if 0 <= v <= INT64_MAX)


DOMAIN = _domain()

# A representative subset, for cells whose claim does not gain from the full sweep.
# The three full-domain sweeps below are the achievement claim and the cost claim,
# which do gain from it.
SAMPLE = [0, 1, 2, 3, 4, 15, 16, 63, 64, 255, 256, 10 ** 6,
          1085241185, 2 ** 31, 2 ** 62, INT64_MAX]

# The three inputs that could not be made green under the superseded Newton
# construction (findings C-1/C-2). They are called out so a regression to a
# convergence-predicated iteration is named, not merely counted.
FORMERLY_UNSATISFIABLE = [1085241185, 1736392818365009963, 2 ** 62]


# ==============================================================================
# Claim 1 — the value is the integer square root (the feature oracle)
# ==============================================================================

@pytest.mark.parametrize("n", DOMAIN)
def test_i_sqrt_equals_floor_sqrt(n):
    """The achievement claim, grounded in the definition of integer square root."""
    i_sqrt = api(MODULE, "i_sqrt")
    assert i_sqrt(n) == math.isqrt(n)


@pytest.mark.parametrize("n", DOMAIN)
def test_i_sqrt_satisfies_the_defining_inequality(n):
    """r == floor(sqrt(n)) iff r*r <= n < (r+1)*(r+1). Stated without math.isqrt, so a
    defect in the oracle library cannot hide a defect in the implementation."""
    i_sqrt = api(MODULE, "i_sqrt")
    r = i_sqrt(n)
    assert r * r <= n, f"i_sqrt({n}) == {r}, but {r}^2 > {n}"
    assert n < (r + 1) * (r + 1), f"i_sqrt({n}) == {r}, but ({r}+1)^2 <= {n}"


@pytest.mark.parametrize("n", FORMERLY_UNSATISFIABLE)
def test_i_sqrt_at_the_inputs_newton_could_not_reach(n):
    """The three witnesses that killed the superseded construction.

    Under I-BERT Newton at a fixed 4 iterations these were unreachable: the guarded
    loop needs 6 steps for n=1085241185 — inside int32, and the plan's stated domain
    was int64 (finding C-2). Digit recurrence reaches them because 32 base-4 digits
    span int64 by construction. Named separately so a regression to any
    convergence-estimated iteration count fails a cell that says why.
    """
    i_sqrt = api(MODULE, "i_sqrt")
    assert i_sqrt(n) == math.isqrt(n)


def test_i_sqrt_at_the_int64_ceiling():
    """The largest input the 32-digit budget must cover. floor(sqrt(2^63 - 1)) is
    3037000499 — a 32-bit root, which is why 32 base-4 digits suffice and why the
    budget is structural rather than measured."""
    i_sqrt = api(MODULE, "i_sqrt")
    assert i_sqrt(INT64_MAX) == 3037000499
    assert i_sqrt(2 ** 62) == 2 ** 31


def test_i_sqrt_at_base_four_digit_boundaries():
    """Each iteration resolves one base-4 digit, so 4^k is where a digit decision flips
    and the restoring subtraction either fires or does not. An off-by-one in `bit` or in
    the `(res >> 1) + bit` update surfaces at these values first."""
    i_sqrt = api(MODULE, "i_sqrt")
    for k in range(0, 32):
        boundary = 4 ** k
        for candidate in (boundary - 1, boundary, boundary + 1):
            if 0 <= candidate <= INT64_MAX:
                assert i_sqrt(candidate) == math.isqrt(candidate), (
                    f"at 4^{k} + {candidate - boundary}"
                )


def test_i_sqrt_of_zero_is_zero():
    """The domain floor. Digit recurrence needs no special case here — every digit
    decision simply takes the else branch — so a guard at n == 0 would be a sign the
    construction is not the one that was ruled."""
    i_sqrt = api(MODULE, "i_sqrt")
    assert i_sqrt(0) == 0


def test_i_sqrt_of_perfect_squares_is_exact():
    """The restoring subtraction must consume n exactly at a perfect square."""
    i_sqrt = api(MODULE, "i_sqrt")
    for root in [0, 1, 2, 3, 7, 255, 256, 65535, 65536, 46341, 3037000499]:
        assert i_sqrt(root * root) == root


def test_i_sqrt_just_below_a_perfect_square_does_not_round_up():
    """r*r - 1 must give r-1, not r. The most common off-by-one in an integer square
    root, and the one a `<=` / `<` slip in the digit comparison produces."""
    i_sqrt = api(MODULE, "i_sqrt")
    for root in [1, 2, 3, 7, 255, 256, 65535, 65536, 46341, 3037000499]:
        assert i_sqrt(root * root - 1) == root - 1


def test_i_sqrt_is_monotone_non_decreasing():
    i_sqrt = api(MODULE, "i_sqrt")
    previous = None
    for n in range(0, 4096):
        current = i_sqrt(n)
        if previous is not None:
            assert current >= previous, f"i_sqrt({n}) < i_sqrt({n - 1})"
        previous = current


@pytest.mark.parametrize("n", [0, 1, 3, 10 ** 6, 2 ** 62, INT64_MAX])
def test_i_sqrt_returns_a_non_negative_exact_int(n):
    """No float leaks out of the integer path (Forge W6)."""
    i_sqrt = api(MODULE, "i_sqrt")
    r = i_sqrt(n)
    assert type(r) is int, f"i_sqrt({n}) is {type(r).__name__}, not a plain int"
    assert r >= 0


def test_i_sqrt_rejects_negative_input():
    """RMSNorm sums of squares are non-negative by construction, so a negative input is
    a caller defect. Off-domain behaviour is defined, not undefined."""
    i_sqrt = api(MODULE, "i_sqrt")
    for n in [-1, -2, -(2 ** 31), -(2 ** 63)]:
        with pytest.raises(ValueError):
            i_sqrt(n)


def test_i_sqrt_is_pure():
    """The recurrence consumes its argument internally. Two calls with the same input
    must still agree — a caller's value is not a scratch register."""
    i_sqrt = api(MODULE, "i_sqrt")
    sample = [0, 1, 3, 10 ** 6, 1085241185, 2 ** 62, INT64_MAX]
    assert [i_sqrt(n) for n in sample] == [i_sqrt(n) for n in sample]


# ==============================================================================
# Claim 2 — the trip count is fixed and data-independent (D-SLM29)
# ==============================================================================

def test_i_sqrt_iterations_constant_is_thirty_two():
    """D-SLM29 pins the count; the digit-recurrence ruling sets it at 32 — one per
    base-4 digit of an int64. The constant is the artifact the cost claim is stated in,
    so it is asserted directly and a silent retune is a test failure.

    32 is not a tuning knob: fewer digits cannot reach the int64 ceiling (C-2's defect),
    and more would be dead iterations below `bit == 1`.
    """
    iterations = api(MODULE, "I_SQRT_ITERATIONS")
    assert iterations == 32


@pytest.mark.parametrize("n", DOMAIN)
def test_i_sqrt_trace_length_is_the_pinned_iteration_count(n):
    """The cost claim made falsifiable: one iterate per executed iteration, and the
    length must equal I_SQRT_ITERATIONS for EVERY input.

    Digit recurrence branches per digit — a select on `n >= res + bit` — but never on
    how many digits to process. An implementation that skips leading zero digits (the
    conventional `while bit > n: bit >>= 2` prologue) makes this length a function of n
    and nicks §14's cost-determinism contract, which is what D-SLM29 forbids.
    """
    i_sqrt_trace, iterations = api(MODULE, "i_sqrt_trace", "I_SQRT_ITERATIONS")
    trace = i_sqrt_trace(n)
    assert len(trace) == iterations, (
        f"i_sqrt_trace({n}) ran {len(trace)} iterations, not the pinned {iterations} — "
        f"the trip count is data-dependent"
    )


def test_i_sqrt_trace_length_is_identical_across_the_whole_domain():
    """The data-independence claim as one assertion over the domain, so the failure
    message names the spread rather than one arm of it. n == 0 processes 32 all-zero
    digits; n == 2^63 - 1 processes 32 significant ones; both must report 32."""
    i_sqrt_trace = api(MODULE, "i_sqrt_trace")
    lengths = {n: len(i_sqrt_trace(n)) for n in DOMAIN}
    distinct = sorted(set(lengths.values()))
    assert len(distinct) == 1, f"trip count varies with data: observed lengths {distinct}"


@pytest.mark.parametrize("n", SAMPLE)
def test_i_sqrt_trace_last_iterate_is_the_returned_value(n):
    """The trace must be of the real computation. Without this, a decoy trace of the
    right length could sit beside a data-dependent i_sqrt and the cells above would pass
    while the shipped trip count stayed a function of the data."""
    i_sqrt, i_sqrt_trace = api(MODULE, "i_sqrt", "i_sqrt_trace")
    trace = i_sqrt_trace(n)
    assert trace, f"i_sqrt_trace({n}) is empty"
    assert trace[-1] == i_sqrt(n)


@pytest.mark.parametrize("n", [0, 1, 3, 10 ** 6, 2 ** 62, INT64_MAX])
def test_i_sqrt_trace_iterates_are_non_negative_ints(n):
    i_sqrt_trace = api(MODULE, "i_sqrt_trace")
    for step, iterate in enumerate(i_sqrt_trace(n)):
        assert type(iterate) is int, f"i_sqrt_trace({n})[{step}] is {type(iterate).__name__}"
        assert iterate >= 0
