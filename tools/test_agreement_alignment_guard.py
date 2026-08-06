"""Red suite for `agreement_alignment_guard.py` (T-1756, casebook entry E7,
D-SLM346/347).

Two populations, per `StandardsDocument.md` Sec4's validation requirement
("a new structure is validated against an independently-found population
before it is trusted... a check shown able to fail on a population of one is
not shown to cover"):

1. A constructed fault (`test_constructed_fence_fault_fires`) -- built by this
   test, not found in the wild -- that proves the guard CAN fail: it isolates
   the exact mechanism (a constant leading-token offset) with no other
   variation, so a change that broke the shift search would be caught here
   even if the real-population fixture below happened not to exercise it.

2. An independently-found population (`test_reproduces_the_real_d_slm346_
   population`) -- the eight prompts' real ref/A/B token sequences from
   `Claude/Laplace/silu-lut-s15-harness/token_rerun_result.json`, the exact
   run D-SLM346 killed. This is not authored to make the guard pass; it is
   the actual data the historical debunk (`Claude/Popper/
   shift_align_probe.py`) was run against, vendored here (this repo has no
   access to the Wizard records repo at CI time) because a check that only
   ever sees data invented to satisfy it is not shown to cover the defect it
   claims to guard.
"""
from __future__ import annotations

import pytest

from agreement_alignment_guard import (
    AgreementResult,
    PossibleAlignmentArtifact,
    scored_agreement,
)


# --------------------------------------------------------------------------
# 1. Constructed fault: isolates the mechanism with no other variation.
# --------------------------------------------------------------------------

def test_constructed_fence_fault_fires():
    """A reference decode and a candidate decode that agree on every content
    token, but the candidate opens with a 3-token markdown fence the
    reference does not emit (the exact artifact named in the E7 casebook
    entry). Fixed-index (s=0) scoring should show near-total disagreement;
    the guard must refuse to return that number."""
    fence = [73594, 2236, 198]  # "```" "json" "\n" -- the real fence token ids
    content = [515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622]
    reference = content
    candidate = fence + content[:-3]  # same length as reference, offset by the fence

    # Naive fixed-index scoring on this construction: essentially total
    # disagreement, because every position after the fence is a different
    # content token than at that same index in the unshifted reference.
    naive = sum(1 for x, y in zip(candidate, reference) if x == y)
    assert naive <= 1, "constructed fault must reproduce near-total s=0 disagreement"

    with pytest.raises(PossibleAlignmentArtifact) as excinfo:
        scored_agreement(candidate, reference)

    err = excinfo.value
    assert err.s0_matches == naive
    assert err.best_shift == 3
    assert err.best_matches >= 12, "shift-aligned recovery should reach near-full content match"


def test_constructed_fence_fault_absent_when_prefix_shared():
    """Control for the constructed fault above: when both sides share the
    same leading fence (no offset introduced), s=0 scoring is already
    correct and the guard must not fire."""
    fence = [73594, 2236, 198]
    content = [515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622]
    reference = fence + content
    candidate = fence + content

    result = scored_agreement(candidate, reference)
    assert isinstance(result, AgreementResult)
    assert result.matches == result.length
    assert result.shift == 0


def test_genuine_divergence_no_shift_recovers_it():
    """Control: two sequences that genuinely disagree throughout, with no
    shift bringing them into better alignment (every position mismatches
    under every shift in the search window). The guard must not fire --
    a real content divergence is not the hazard this guard exists to catch,
    and flagging it would train callers to ignore the guard."""
    reference = list(range(100, 116))
    candidate = [v + 1000 for v in reference]  # disjoint value space at every shift

    result = scored_agreement(candidate, reference)
    assert isinstance(result, AgreementResult)
    assert result.matches == 0
    assert result.shift == 0


# --------------------------------------------------------------------------
# 2. Independently-found population: the real D-SLM346 run.
# --------------------------------------------------------------------------

# Vendored verbatim from `Claude/Laplace/silu-lut-s15-harness/
# token_rerun_result.json`'s `per_prompt` array (Wizard records repo,
# 2026-07-28 run) -- the eight-prompt, three-arm greedy-decode comparison
# D-SLM346 killed as a fixed-index alignment artifact. `ref` is the float
# reference decode; `A` and `B` are the two integer-forward arms compared
# against it. Token ids are Qwen2.5's own vocabulary; content is a JSON
# slot-filling scaffold, not natural-language text.
D_SLM346_POPULATION = [
    dict(id="plain-001", held_out=False,
         ref=[515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622],
         A=[515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622],
         B=[515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622]),
    dict(id="multi-001", held_out=False,
         ref=[515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622],
         A=[73594, 2236, 198, 515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341],
         B=[515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622]),
    dict(id="ho-indomain-1", held_out=True,
         ref=[515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622],
         A=[4913, 56431, 788, 330, 2190, 497, 330, 49767, 788, 5212, 18622, 788, 330, 359, 53434, 497],
         B=[73594, 2236, 198, 515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341]),
    dict(id="ho-indomain-2", held_out=True,
         ref=[515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622],
         A=[515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622],
         B=[73594, 2236, 198, 515, 220, 330, 56431, 788, 330, 24852, 5738, 756, 220, 330, 49767, 788]),
    dict(id="ho-indomain-3", held_out=True,
         ref=[515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622],
         A=[515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341, 262, 330, 18622],
         B=[73594, 2236, 198, 515, 220, 330, 56431, 788, 330, 2190, 756, 220, 330, 49767, 788, 341]),
    dict(id="ho-indomain-4", held_out=True,
         ref=[73594, 2236, 198, 515, 220, 330, 56431, 788, 330, 24852, 5738, 756, 220, 330, 49767, 788],
         A=[73594, 2236, 198, 515, 220, 330, 56431, 788, 330, 6555, 5738, 756, 220, 330, 49767, 788],
         B=[515, 220, 330, 56431, 788, 330, 6555, 5738, 756, 220, 330, 49767, 788, 341, 262, 330]),
    dict(id="ho-generic-1", held_out=True,
         ref=[515, 220, 330, 56431, 788, 330, 6555, 5738, 756, 220, 330, 49767, 788, 341, 262, 330],
         A=[515, 220, 330, 56431, 788, 330, 6555, 5738, 756, 220, 330, 49767, 788, 341, 262, 330],
         B=[73594, 2236, 198, 515, 220, 330, 56431, 788, 330, 94344, 756, 220, 330, 49767, 788, 341]),
    dict(id="ho-generic-2", held_out=True,
         ref=[73594, 2236, 198, 515, 220, 330, 56431, 788, 330, 6555, 5738, 756, 220, 330, 49767, 788],
         A=[73594, 2236, 198, 4913, 56431, 3252, 6555, 5738, 2198, 49767, 22317, 18622, 3252, 359, 53434, 2198],
         B=[515, 220, 330, 56431, 788, 330, 6555, 5738, 756, 220, 330, 49767, 788, 341, 262, 330]),
]

# Rows the guard must raise on -- the seven (prompt, arm) pairs whose shift
# search recovers the fence at exactly shift = +-3 (matching the 3-token
# fence length) with a large gain, reproducing what
# `Claude/Popper/shift_align_probe.py` found by hand on this exact run
# (`Claude/Decisions/DecisionLog.md` D-SLM346): a fixed-index score of 1
# recovering to 6-13 matches once the fence offset is accounted for.
EXPECTED_FLAGGED = {
    ("multi-001", "A"),
    ("ho-indomain-1", "B"),
    ("ho-indomain-2", "B"),
    ("ho-indomain-3", "B"),
    ("ho-indomain-4", "B"),
    ("ho-generic-1", "B"),
    ("ho-generic-2", "B"),
}

# One additional row the guard's default thresholds also flag:
# ho-indomain-1/A (s=0 matches=1, best=5 at shift=+2, gain=4, ratio=5x). This
# is NOT one of the fence-driven rows above -- its best shift (2) does not
# match the fence length (3), and neither `A` nor `ref` on this row carries a
# fence at all (both fence flags are False in `shift_align_probe.py`'s own
# output) -- so this is a genuine content divergence with a coincidental
# partial recovery under the shift search, not a reproduction of the E7
# artifact. The guard's default thresholds (relative gain >= 2x, absolute
# gain >= 3) flag it anyway, which is accepted and documented rather than
# tuned away: the guard's job is to force a second look whenever fixed-index
# scoring is not robust to a small shift, and erring toward an extra caution
# flag on a borderline case is the correct bias for a hazard that has
# already produced two killed headlines -- see `scored_agreement`'s own
# docstring.
EXPECTED_ADDITIONAL_CAUTION_FLAG = {("ho-indomain-1", "A")}


def _flag(row, arm):
    """True if `scored_agreement(row[arm], row['ref'])` raises."""
    try:
        scored_agreement(row[arm], row["ref"])
    except PossibleAlignmentArtifact:
        return True
    return False


def test_reproduces_the_real_d_slm346_population():
    """The guard must raise on every one of the seven fence-driven rows the
    historical debunk found, plus the one documented additional caution
    flag, and must NOT raise on any of the other eight (row, arm) cells --
    reproducing the independently-found population exactly, not a subset
    tuned to look sufficient."""
    flagged = set()
    for row in D_SLM346_POPULATION:
        for arm in ("A", "B"):
            if _flag(row, arm):
                flagged.add((row["id"], arm))

    expected = EXPECTED_FLAGGED | EXPECTED_ADDITIONAL_CAUTION_FLAG
    assert flagged == expected, (
        f"guard's flagged set diverged from the D-SLM346 population's expected "
        f"outcome.\nmissing (real hazard NOT caught): {expected - flagged}\n"
        f"unexpected (flagged beyond the documented caution row): {flagged - expected}"
    )

    # Every fence-driven row specifically must be among those caught -- the
    # load-bearing assertion. A regression that silently narrowed the shift
    # window or the gain thresholds enough to miss even one of these would
    # reproduce exactly the D-SLM346/347 failure this guard exists to close.
    missed_real_hazard = EXPECTED_FLAGGED - flagged
    assert not missed_real_hazard, (
        f"guard failed to catch real historical misalignment instance(s): {missed_real_hazard}"
    )


def test_no_shift_recovers_agreement_on_fully_aligned_rows():
    """Plain sanity check on the same population: the three rows where
    `A_vs_ref` was already a full-length or genuinely-unimproved match at
    s=0 (plain-001 both arms, ho-indomain-2/3/4's A arm, ho-generic-1's A
    arm) must not raise -- confirms the guard is not simply firing on every
    row regardless of content."""
    quiet_cells = [
        ("plain-001", "A"), ("plain-001", "B"),
        ("ho-indomain-2", "A"), ("ho-indomain-3", "A"), ("ho-indomain-4", "A"),
        ("ho-generic-1", "A"),
    ]
    by_id = {row["id"]: row for row in D_SLM346_POPULATION}
    for row_id, arm in quiet_cells:
        row = by_id[row_id]
        result = scored_agreement(row[arm], row["ref"])
        assert isinstance(result, AgreementResult), f"{row_id}/{arm} unexpectedly flagged"
