"""Table-driven coverage for `t1906_grade_arms._BITS_IN_NAME` -- T-1907 finding S3, round 3.

This guard's coverage has been wrong in a NEW way in three consecutive review rounds, each
time defended by a docstring rather than pinned by a test that runs:

  round 1 -- shipped `bits?(\\d+)`; the docstring claimed `b<N>` was covered; it wasn't.
  round 2 -- widened to `\\b(?:bits?|b)(\\d+)`; the docstring enumerated four passing names and
             said nothing about the shape it still missed. A trailing `\\b` broke
             `bits16_rerun`/`b16_final` (caught before shipping, by executing round 1's own
             12-name set -- but only the trailing boundary was checked).
  round 3 -- the SAME fact (`_` is a word character, so `\\b` finds no boundary next to it)
             was never checked on the LEADING side: `arm_b8` and `run1_bits16` matched
             nothing, and `--arm run1_bits16=<a bits=8 directory>` graded silently at exit 0.

`StandardsDocument.md` SS4: a rule that has failed twice does not get a fourth prose
description, it gets a structure that fails the build when it regresses. This file is that
structure. Every name below is drawn from an actual review round's execution log (T-1907
rounds 1-3) -- none are invented for this file -- so a future change to `_BITS_IN_NAME` is
checked against the SAME cases three review rounds have already executed by hand, not against
a fresh guess at what might matter.

Run: `pytest tools/test_t1906_bits_in_name.py -v`
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1906_grade_arms as G  # noqa: E402


# name -> expected extracted bits under the CURRENT pattern, or None if the guard must not
# match at all. Every name is drawn from a T-1907 review round's own execution log; the value
# is what the pattern SHOULD return today (round 3), not a historical per-round value -- a
# regex has one current behavior, and a table with two rows for the same name and two
# different expectations would just be asserting a contradiction against itself. Where a
# name's expectation changed across rounds (round 1 did not cover bare `b<N>`; round 2 added
# it), the comment says so; the table carries only the value that must hold now.
CASES = [
    # --- T-1907 round 1's own 12-name set (0a1690c casebook, "12 plausible arm names") ---
    ("bits16", 16),
    ("bits8", 8),
    ("bits16_rerun", 16),
    ("BITS16", 16),
    ("bit16", 16),
    ("b16", 16),            # round 1's pattern (`bits?(\d+)`) did NOT cover bare `b<N>`;
    ("b8", 8),               # round 2 widened the pattern to cover it (T-1907 round 2 S1).
    ("B16", 16),             # These four rows assert round 2's (and round 3's) behavior --
    ("b16_final", 16),       # the CURRENT pattern -- not round 1's narrower original scope.
    ("int16", None),
    ("int12", None),
    ("16bit", None),
    ("width16", None),

    # --- T-1907 round 3's own defect cases (98d53c0 casebook, finding S3) ---
    ("arm_b8", 8),           # round 2's pattern matched NEITHER of these two (leading `_`
    ("run1_bits16", 16),     # left no `\b`); round 3's own fix is what makes them match.

    # --- T-1907 round 3's extended 20-name characterization set (98d53c0 execution #19) ---
    ("subbits16", None),   # preceded by a letter ('u') -- must not match mid-word
    ("abits16", None),     # preceded by a letter ('a')
    ("T1906b16", None),    # preceded by a digit ('6')
    ("1b16", None),        # preceded by a digit ('1')
    ("bf16", None),        # 'b' followed by a non-digit ('f') -- no digit run after the prefix
    ("batch16", None),     # 'b' followed by non-digit letters before any digit
    ("baseline8", None),   # same shape, longer word
    ("arm-b8", 8),         # '-' is a valid leading delimiter
    ("final.b12", 12),     # '.' is a valid leading delimiter
    ("kv-b16", 16),
    ("b16rerun", 16),      # trailing garbage after the digit run is NOT anchored against
    ("b16.2", 16),         # (round 2's own choice, re-affirmed -- only the LEADING side
    ("b012", 12),          # is anchored; round 3 touches nothing about the trailing side)
]


@pytest.mark.parametrize("name,expected", CASES, ids=[c[0] for c in CASES])
def test_bits_in_name(name, expected):
    m = G._BITS_IN_NAME.search(name)
    got = int(m.group(1)) if m is not None else None
    assert got == expected, (
        f"_BITS_IN_NAME.search({name!r}) -> {got!r}, expected {expected!r}. "
        f"This name is drawn from a T-1907 review round's own execution log; if this test now "
        f"fails, the guard's coverage just narrowed or widened relative to what a prior review "
        f"round verified by hand -- update the review record before changing this table."
    )


def test_cases_table_has_no_duplicate_names():
    """Every name in CASES appears exactly once. A duplicate would mean two rows silently
    assert two different things about the same input -- whichever pytest happened to collect
    second would mask the first rather than the table catching its own contradiction.
    """
    names = [name for name, _ in CASES]
    assert len(names) == len(set(names)), (
        f"CASES contains duplicate names: "
        f"{sorted({n for n in names if names.count(n) > 1})}"
    )
    assert len(names) >= 25, (
        f"expected at least 25 distinct names across three review rounds' worth of cases; "
        f"got {len(names)} -- has this table been trimmed?"
    )


def test_pattern_has_a_leading_anchor_that_is_not_a_bare_word_boundary():
    """A structural guard against reintroducing exactly round 3's own regression: a bare `\\b`
    on the leading side is silently wrong next to `_` (and any other word character), which is
    the whole content of finding S3. This does not re-derive the fix; it fails loudly if a
    future edit reverts the anchor to `\\b` without re-running the case table above -- the
    table itself would also fail, this is a second, cheaper tripwire on the pattern's own text.
    """
    pattern_text = G._BITS_IN_NAME.pattern
    assert not pattern_text.startswith(r"\b"), (
        "_BITS_IN_NAME starts with a bare \\b again -- this is exactly the T-1907 round 3 "
        "regression (a leading word-boundary anchor cannot match a token preceded by '_', "
        "since '_' is itself a word character). Use a negative lookbehind for an alphanumeric "
        "character instead, and re-run this file's full CASES table before shipping."
    )
