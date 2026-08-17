"""Plan §6.4 — RoPE: Q2.30 fixed-point sin/cos tables and the pairwise rotation.

"This is the plan's one genuinely novel numeric component" — no published integer
construction exists, so every cell here is derived from §6.4's own text:

    precomputed per-position fixed-point sin/cos tables (Q2.30 in int32, positions
    0..context cap), pairwise rotation (x*cos - y*sin, x*sin + y*cos) in int64
    intermediates with a single pinned rounding to the activation format. Table
    generation happens offline in double precision — the runtime only reads integers.

Oracles. The tables get a feature oracle against the double-precision reference §6.4
names, at the ulp of the stated format. The rotation gets exact feature oracles at the
three angles whose result is exactly known (0, 90, 180 degrees), plus a
double-precision oracle elsewhere, plus two cells that discriminate the two
constructions §6.4 forbids: rounding twice instead of once, and accumulating in int32
instead of int64.

Both of this file's open questions are now settled by §6.8's normative constants table,
and the cells cite the row rather than a reading:

  * **C12** pins the table length — `context_cap` entries, indices `[0, context_cap)`,
    upper bound **EXCLUSIVE**. The 2026-07-14 suite read §6.4's "positions 0..context
    cap" as inclusive and restated `context_cap + 1` in four places; the planner ruled
    exclusive and none of them moved, so a green test enforced the superseded reading
    until 2026-07-15 (record P-1). The length now lives in `c12_table_length` alone.
  * **C13** pins the rotation's single rounding as `RoundingDivideByPOT` — the reading
    the 2026-07-14 suite took and routed as C-6, since ratified.
"""

import math

import pytest

from conftest import api

MODULE = "reference_pipeline.rope"
INTMATH = "reference_pipeline.intmath"

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1

FRAC_BITS = 30
ONE_Q30 = 1 << 30

HEAD_DIM = 64
CONTEXT_CAP = 128
THETA = 10000.0


def c12_table_length(context_cap):
    """§6.8 C12 — **PINNED**: `context_cap` entries, indices `[0, context_cap)`.

    The single place this suite states the table length. Every cell reads it here, so a
    future ruling moves one line rather than four — which is the whole lesson of P-1: a
    length restated per-cell is a length that does not move when the table does.
    """
    return context_cap


def c12_last_position(context_cap):
    """The highest position the table carries. C12: "A sequence of `context_cap` tokens
    occupies positions `0 .. context_cap-1`; a position `>= context_cap` is the
    documented over-cap rejection (§17 dim 5), never a table read.\""""
    return context_cap - 1

# Q2.30 ulp tolerance for a table entry against the double-precision reference. A
# correct table lands within one ulp of the rounding; a wrong theta, a wrong
# exponent, or a swapped sin/cos misses by ~1e8 ulps.
TABLE_ULP_TOL = 2


def reference_angle(position, pair_index, head_dim=HEAD_DIM, theta=THETA):
    """§6.4's angle, in double precision — the reference the plan names."""
    inv_freq = theta ** (-2.0 * pair_index / head_dim)
    return position * inv_freq


# ==============================================================================
# The fixed-point format
# ==============================================================================

def test_q2_30_format_constants():
    """Q2.30: 30 fractional bits, so 1.0 is 2^30 and the format spans [-2, 2)."""
    frac_bits, one = api(MODULE, "ROPE_FRAC_BITS", "ROPE_ONE")
    assert frac_bits == FRAC_BITS
    assert one == ONE_Q30 == 1073741824
    assert one <= INT32_MAX  # 1.0 is representable in int32 at this radix point


# ==============================================================================
# The tables
# ==============================================================================

def test_rope_table_length_is_exactly_c12():
    """§6.8 C12, asserted on its own and read from one place.

    **The anti-regression cell.** C12 is PINNED: `context_cap` entries, indices
    `[0, context_cap)`, upper bound EXCLUSIVE. Its own rationale says the previous
    wording "is ambiguous by one entry and would silently produce two different table
    hashes" — and that is precisely what happened (record P-1): this suite guessed
    inclusive, the planner ruled exclusive, and the guess stayed green.

    Every length statement in this file routes through `c12_table_length`, so the next
    ruling moves one line. This cell is where a disagreement with C12 surfaces first and
    by name, rather than as a fencepost failure three cells away.
    """
    rope_tables = api(MODULE, "rope_tables")
    for cap in (1, 8, 16, 128, 4096):
        cos_table, sin_table = rope_tables(HEAD_DIM, cap, THETA)
        assert len(cos_table) == c12_table_length(cap), (
            f"cos table has {len(cos_table)} rows at context_cap={cap}; §6.8 C12 pins "
            f"{c12_table_length(cap)} — indices [0, {cap}), upper bound EXCLUSIVE"
        )
        assert len(sin_table) == c12_table_length(cap), (
            f"sin table has {len(sin_table)} rows at context_cap={cap}; §6.8 C12 pins "
            f"{c12_table_length(cap)} — indices [0, {cap}), upper bound EXCLUSIVE"
        )


def test_rope_tables_shape_is_length_by_pairs():
    """One row per position C12 admits, one column per rotated pair (head_dim // 2)."""
    rope_tables = api(MODULE, "rope_tables")
    cos_table, sin_table = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    assert len(cos_table) == len(sin_table) == c12_table_length(CONTEXT_CAP)
    for row in list(cos_table) + list(sin_table):
        assert len(row) == HEAD_DIM // 2


def test_rope_tables_position_zero_is_the_identity_rotation():
    """The one row whose truth is exact and needs no tolerance: angle 0 for every
    pair, so cos is exactly 1.0 (2^30) and sin is exactly 0. An off-by-one in the
    position index, or an inv_freq applied to the wrong axis, fails here."""
    rope_tables = api(MODULE, "rope_tables")
    cos_table, sin_table = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    assert list(cos_table[0]) == [ONE_Q30] * (HEAD_DIM // 2)
    assert list(sin_table[0]) == [0] * (HEAD_DIM // 2)


def test_rope_tables_match_the_double_precision_reference():
    """The achievement claim: the tables are the rounded Q2.30 sin/cos of §6.4's
    angle, checked against the double-precision reference the plan names as the
    generator. Grounded in the definition of the angle, not in a recode of whatever
    loop the implementation uses."""
    rope_tables = api(MODULE, "rope_tables")
    cos_table, sin_table = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    for position in range(c12_table_length(CONTEXT_CAP)):
        for pair_index in range(HEAD_DIM // 2):
            angle = reference_angle(position, pair_index)
            expected_cos = round(math.cos(angle) * ONE_Q30)
            expected_sin = round(math.sin(angle) * ONE_Q30)
            assert abs(cos_table[position][pair_index] - expected_cos) <= TABLE_ULP_TOL, (
                f"cos[{position}][{pair_index}]"
            )
            assert abs(sin_table[position][pair_index] - expected_sin) <= TABLE_ULP_TOL, (
                f"sin[{position}][{pair_index}]"
            )


def test_rope_tables_boundary_position_is_correct():
    """§17 dimension 6 names "RoPE table boundary positions". The last row is where a
    fencepost error in the generator lands.

    §6.8 C12: the last position the table carries is `context_cap - 1`. A table whose
    last row is `context_cap` has one row too many — the reading this suite held green
    until 2026-07-15 (record P-1).
    """
    rope_tables = api(MODULE, "rope_tables")
    cos_table, sin_table = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    last = c12_last_position(CONTEXT_CAP)
    for pair_index in range(HEAD_DIM // 2):
        angle = reference_angle(last, pair_index)
        assert abs(cos_table[last][pair_index] - round(math.cos(angle) * ONE_Q30)) <= TABLE_ULP_TOL
        assert abs(sin_table[last][pair_index] - round(math.sin(angle) * ONE_Q30)) <= TABLE_ULP_TOL


def test_rope_tables_carry_no_row_for_the_over_cap_position():
    """§6.8 C12: "a position `>= context_cap` is the documented over-cap rejection
    (§17 dim 5), **never a table read**."

    The table is the wrong place to serve an over-cap position, so it must not have a row
    for one. This is the cell that distinguishes C12's ruling from a table that merely
    happens to be indexed from zero.
    """
    rope_tables = api(MODULE, "rope_tables")
    cos_table, sin_table = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    assert len(cos_table) == CONTEXT_CAP
    for table, name in ((cos_table, "cos"), (sin_table, "sin")):
        with pytest.raises(IndexError):
            _ = table[CONTEXT_CAP]


def test_rope_tables_at_a_large_context_cap_boundary():
    """The boundary at a realistic cap, where the angle is large enough that a naive
    accumulate-per-position generator has drifted away from position*inv_freq."""
    rope_tables = api(MODULE, "rope_tables")
    large_cap = 4096
    cos_table, sin_table = rope_tables(HEAD_DIM, large_cap, THETA)
    assert len(cos_table) == c12_table_length(large_cap)
    for position in [c12_last_position(large_cap) - 1, c12_last_position(large_cap)]:
        for pair_index in [0, 1, HEAD_DIM // 4, HEAD_DIM // 2 - 1]:
            angle = reference_angle(position, pair_index)
            assert abs(cos_table[position][pair_index] - round(math.cos(angle) * ONE_Q30)) <= TABLE_ULP_TOL
            assert abs(sin_table[position][pair_index] - round(math.sin(angle) * ONE_Q30)) <= TABLE_ULP_TOL


def test_rope_table_entries_are_int32_and_within_the_q2_30_range():
    """"Q2.30 in int32": every entry is an exact int, fits int32, and |v| <= 2^30
    because sin and cos are bounded by 1. An entry above 2^30 means the radix point
    moved."""
    rope_tables = api(MODULE, "rope_tables")
    cos_table, sin_table = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    for name, table in [("cos", cos_table), ("sin", sin_table)]:
        for position, row in enumerate(table):
            for pair_index, value in enumerate(row):
                assert type(value) is int, (
                    f"{name}[{position}][{pair_index}] is {type(value).__name__}, not a plain int"
                )
                assert INT32_MIN <= value <= INT32_MAX
                assert abs(value) <= ONE_Q30, f"{name}[{position}][{pair_index}] == {value} exceeds 1.0"


def test_rope_tables_satisfy_the_pythagorean_identity():
    """cos^2 + sin^2 == 1 in the fixed-point domain, to within the rounding of the two
    entries. A per-entry rounding defect that happens to stay inside the ulp bound of
    one table still shows up as an off-unit-circle pair here."""
    rope_tables = api(MODULE, "rope_tables")
    cos_table, sin_table = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    bound = 4 * ONE_Q30  # each entry rounds by <= 0.5 ulp; the cross terms cost <= 2^31
    for position in range(c12_table_length(CONTEXT_CAP)):
        for pair_index in range(HEAD_DIM // 2):
            c = cos_table[position][pair_index]
            s = sin_table[position][pair_index]
            assert abs(c * c + s * s - ONE_Q30 ** 2) <= bound, f"[{position}][{pair_index}]"


def test_rope_tables_are_deterministic():
    """Two generations of the same configuration produce identical integers. The
    tables enter the artifact, so a nondeterministic generator would break §13.3's
    golden hashes at conversion time rather than at run time."""
    rope_tables = api(MODULE, "rope_tables")
    first = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    second = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    assert [list(r) for r in first[0]] == [list(r) for r in second[0]]
    assert [list(r) for r in first[1]] == [list(r) for r in second[1]]


def test_rope_tables_reject_an_odd_head_dim():
    """The rotation is pairwise, so head_dim must be even. Off-domain configuration
    is a defined rejection, not a silently truncated table."""
    rope_tables = api(MODULE, "rope_tables")
    with pytest.raises(ValueError):
        rope_tables(63, CONTEXT_CAP, THETA)


# ==============================================================================
# The pairwise rotation
# ==============================================================================

def test_rope_apply_pair_at_zero_degrees_is_the_identity():
    """cos == 1.0, sin == 0 must return the input unchanged and exactly. The single
    strongest cell for the shift: any rounding or shift width other than 30 fails."""
    rope_apply_pair = api(MODULE, "rope_apply_pair")
    for x, y in [(0, 0), (1, 0), (0, 1), (127, -128), (-128, 127), (12345, -6789)]:
        assert rope_apply_pair(x, y, ONE_Q30, 0) == (x, y)


def test_rope_apply_pair_at_ninety_degrees():
    """cos == 0, sin == 1.0 rotates (x, y) to (-y, x), exactly."""
    rope_apply_pair = api(MODULE, "rope_apply_pair")
    for x, y in [(1, 0), (0, 1), (127, -128), (-128, 127), (12345, -6789)]:
        assert rope_apply_pair(x, y, 0, ONE_Q30) == (-y, x)


def test_rope_apply_pair_at_one_eighty_degrees():
    """cos == -1.0, sin == 0 negates both components, exactly. Catches a sign error
    in the second component that the 0-degree cell cannot see."""
    rope_apply_pair = api(MODULE, "rope_apply_pair")
    for x, y in [(1, 0), (0, 1), (127, -128), (-128, 127), (12345, -6789)]:
        assert rope_apply_pair(x, y, -ONE_Q30, 0) == (-x, -y)


def test_rope_apply_pair_rounds_once_not_twice():
    """§6.4: "int64 intermediates with a SINGLE pinned rounding".

    The fixture is built so the two constructions disagree. With cos == 0.5 and
    sin == 0.25 and x == y == 1:
        rounding once:  (1*2^29 - 1*2^28) >> 30 == 0.25 -> 0
        rounding twice: round(0.5) - round(0.25) == 1 - 0  -> 1
    An implementation that rounds each product to the activation format before
    combining them returns 1 here. The single-rounding construction returns 0.
    """
    rope_apply_pair = api(MODULE, "rope_apply_pair")
    first, second = rope_apply_pair(1, 1, 1 << 29, 1 << 28)
    assert first == 0, "x*cos and y*sin were rounded separately before being combined"
    assert second == 1


def test_rope_apply_pair_accumulates_in_int64():
    """§6.4: "in int64 intermediates". x*cos for an int8 activation and a Q2.30 unit
    cosine is 127 * 2^30, which overflows int32. An int32 accumulator wraps it to
    -2^30 and returns -2 for the second component instead of 254.
    """
    rope_apply_pair = api(MODULE, "rope_apply_pair")
    assert 127 * ONE_Q30 > INT32_MAX  # the fixture does overflow int32
    first, second = rope_apply_pair(127, 127, ONE_Q30, ONE_Q30)
    assert (first, second) == (0, 254)


def test_rope_apply_pair_ties_follow_the_pinned_rounding():
    """§6.8 C13 — **PINNED**: the rotation's single rounding is `RoundingDivideByPOT`,
    which C3 pins as ties **away from zero**, in both directions.

    x*cos - y*sin == +2^29 is exactly +0.5 ulp of the output format; the negative
    fixture is exactly -0.5.
    """
    rope_apply_pair = api(MODULE, "rope_apply_pair")
    assert rope_apply_pair(1, 0, 1 << 29, 0)[0] == 1     # +0.5 -> 1, not 0
    assert rope_apply_pair(-1, 0, 1 << 29, 0)[0] == -1    # -0.5 -> -1, not 0


def test_rope_apply_pair_matches_the_double_precision_rotation():
    """The achievement claim at arbitrary angles: the rotated pair is the real
    rotation, to within the single rounding. Grounded in the definition of a rotation,
    driven through the real tables."""
    rope_tables, rope_apply_pair = api(MODULE, "rope_tables", "rope_apply_pair")
    cos_table, sin_table = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    for position in [0, 1, 2, 17, 64, c12_last_position(CONTEXT_CAP)]:
        for pair_index in [0, 1, 7, HEAD_DIM // 2 - 1]:
            angle = reference_angle(position, pair_index)
            c = cos_table[position][pair_index]
            s = sin_table[position][pair_index]
            for x, y in [(127, -128), (-128, 127), (100, 100), (1, -1), (0, 0)]:
                first, second = rope_apply_pair(x, y, c, s)
                assert abs(first - (x * math.cos(angle) - y * math.sin(angle))) <= 1.0
                assert abs(second - (x * math.sin(angle) + y * math.cos(angle))) <= 1.0


def test_rope_apply_pair_preserves_norm():
    """A rotation is an isometry. Holds at every table angle, to within the rounding
    of the two output components — a defect that scales the pair instead of rotating
    it passes the angle cells at 0 degrees and fails here."""
    rope_tables, rope_apply_pair = api(MODULE, "rope_tables", "rope_apply_pair")
    cos_table, sin_table = rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    for position in [0, 1, 5, 33, c12_last_position(CONTEXT_CAP)]:
        for pair_index in [0, 3, HEAD_DIM // 2 - 1]:
            c = cos_table[position][pair_index]
            s = sin_table[position][pair_index]
            for x, y in [(127, -128), (-100, 60), (40, 40)]:
                first, second = rope_apply_pair(x, y, c, s)
                before = math.hypot(x, y)
                after = math.hypot(first, second)
                assert abs(after - before) <= 1.5, f"pos={position} pair={pair_index} ({x},{y})"


def test_rope_apply_pair_returns_exact_ints():
    rope_apply_pair = api(MODULE, "rope_apply_pair")
    for value in rope_apply_pair(127, -128, 1 << 29, 1 << 28):
        assert type(value) is int, f"{value!r} is {type(value).__name__}, not a plain int"


def test_rope_apply_pair_is_pure():
    rope_apply_pair = api(MODULE, "rope_apply_pair")
    inputs = [(127, -128, 1 << 29, 1 << 28), (0, 0, ONE_Q30, 0), (-5, 9, -ONE_Q30, 0)]
    assert [rope_apply_pair(*i) for i in inputs] == [rope_apply_pair(*i) for i in inputs]


def test_rope_apply_pair_uses_the_shared_requant_rounding():
    """§6.8 C13 binds §6.4's single rounding to §6.2's primitive — "not a second rounding
    scheme". Composed against the intmath primitive so a divergence between the two is a
    failure rather than a quiet second dialect of rounding.
    """
    rope_apply_pair = api(MODULE, "rope_apply_pair")
    rounding_divide_by_pot = api(INTMATH, "rounding_divide_by_pot")
    cases = [(127, -128, 1 << 29, 1 << 28), (1, 1, 1 << 29, 1 << 28), (-7, 3, 12345678, -987654),
             (0, 0, ONE_Q30, 0), (-128, 127, -ONE_Q30, 1 << 20)]
    for x, y, c, s in cases:
        expected = (
            rounding_divide_by_pot(x * c - y * s, FRAC_BITS),
            rounding_divide_by_pot(x * s + y * c, FRAC_BITS),
        )
        assert rope_apply_pair(x, y, c, s) == expected, (x, y, c, s)
