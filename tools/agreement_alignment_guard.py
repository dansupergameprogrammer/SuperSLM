"""Structural guard against the fixed-index next-token-agreement alignment artifact.

Casebook entry: `Claude/Linnaeus/preengine-behavior-measurements-audit-fact-sheet-2026-08-05.md`
E7 (D-SLM346/347). A position-wise, index-aligned ("s=0") comparison between two
independently-decoded token sequences reports content-identical continuations as
near-total disagreement whenever one sequence carries extra leading tokens the
other lacks -- e.g. one decode opens with a three-token markdown code-fence
("```json\\n") and the other does not, shifting every subsequent position by a
constant offset a fixed-index `zip()` never accounts for. The same shape produced
a killed headline twice: the C9 softmax campaign's row and the S3.0 SiLU-LUT
token-rerun both reported a large agreement gap that a shift-aligned re-score
(`Claude/Popper/shift_align_probe.py`) corrected to roughly a third of the
reported size (`Claude/Decisions/DecisionLog.md`, D-SLM346/347, 2026-07-28).

The casebook's own ruling on this entry is that the hazard is not a re-measurement
target -- it is a named failure mode that "would silently recur if a re-measurement
on the engine used the same fixed-index scoring," to be carried forward as a
caution for "whoever designs the next agreement metric against the engine." A
caution carried only in prose depends on the next author remembering to read it at
the moment it matters, which is exactly the shape of rule this project's own
standards single out as unreliable (a rule enforced by memory is vigilance, and
vigilance fails). This module makes the check structural instead: any future
next-token-agreement instrument that computes its headline through
`scored_agreement()` gets the shift search for free and cannot silently emit an
s=0 count the shift search itself flags as unreliable.

Import-independent: standard library only, no dependency on any cross-tree spike
or compiled engine artifact -- matching every other module directly under
tools/ (see sslm_convert_validate.py's own module docstring) -- so it runs on a
bare checkout and is collected by the existing `python -m pytest tools/
tests/reference/ -v` CI step (.github/workflows/tests.yml, job
`converter-validate`) with no separate wiring required.
"""
from __future__ import annotations

import dataclasses


class PossibleAlignmentArtifact(ValueError):
    """Raised by `scored_agreement()` when a nonzero shift in the search window
    recovers materially more matches than the requested fixed-index (s=0) score.

    This is the signature of the twice-repeated E7 artifact -- it is not proof
    that this particular case IS that artifact, only that the s=0 number cannot
    be trusted without alignment review, which is exactly the property a bare
    integer return value cannot carry. Carries the counts needed to decide.
    """

    def __init__(self, *, s0_matches: int, best_matches: int, best_shift: int, length: int):
        self.s0_matches = s0_matches
        self.best_matches = best_matches
        self.best_shift = best_shift
        self.length = length
        super().__init__(
            f"position-wise (s=0) agreement is {s0_matches}/{length}, but shift={best_shift} "
            f"recovers {best_matches}/{length} matches -- refusing to report the s=0 count as "
            f"an agreement figure without alignment review. Reference: casebook entry E7 "
            f"(D-SLM346/347) -- this exact shape has already produced two killed headlines. "
            f"Re-score with an alignment-aware method (best-shift, or fence/framing-stripped) "
            f"before reporting a next-token-agreement number."
        )


@dataclasses.dataclass(frozen=True)
class AgreementResult:
    """The only successful return shape from `scored_agreement()` -- the
    fixed-index (s=0) agreement count, returned only once the shift search has
    found no materially better alignment. `shift` is always 0: there is no
    return path that silently substitutes a shifted score for the requested
    one -- a caller that wants the shifted number computes it explicitly."""

    matches: int
    length: int
    shift: int


def _shift_overlap(a, b, shift: int) -> int:
    """Overlap between `a` and `b` under relative offset `shift` (b delayed by
    `shift` tokens relative to a, when shift > 0; advanced, when shift < 0),
    counted only over positions where both shifted sequences still have a
    token. Same convention as `Claude/Popper/shift_align_probe.py`'s
    `best_shift_overlap`, reused deliberately so a result computed by either
    tool means the same thing."""
    if shift >= 0:
        aa, bb = a[shift:], b[: len(a) - shift]
    else:
        aa, bb = a[: len(a) + shift], b[-shift:]
    n = min(len(aa), len(bb))
    return sum(1 for x, y in zip(aa[:n], bb[:n]) if x == y)


def scored_agreement(
    candidate,
    reference,
    *,
    max_shift: int = 4,
    min_relative_gain: float = 2.0,
    min_absolute_gain: int = 3,
) -> AgreementResult:
    """The guarded replacement for the pattern every next-token-agreement
    headline in this project's history has computed inline:
    `sum(1 for x, y in zip(candidate, reference) if x == y)`.

    Computes that same fixed-index (s=0) count, then searches every shift in
    `[-max_shift, max_shift]` (the window `shift_align_probe.py` established
    against the real S3.0/C9 artifacts) for a better-aligned overlap. If some
    nonzero shift recovers at least `min_absolute_gain` more matches AND at
    least `min_relative_gain` times the s=0 count, the s=0 number is refused
    -- `PossibleAlignmentArtifact` is raised instead of returned, because a
    caller receiving a plain integer here cannot distinguish a genuine
    content divergence from the fence-offset artifact that has already
    produced two killed headlines (D-SLM346/347).

    Default thresholds were chosen to reproduce, with margin, the seven
    fence-driven rows in the D-SLM346 population (`Claude/Laplace/
    silu-lut-s15-harness/token_rerun_result.json`, vendored as this module's
    own test fixture) -- every one of those rows shows a gain of at least 5
    matches at a relative gain of at least 6x. The same population contains
    one additional row (a non-fence divergence with a coincidental partial
    shift recovery, gain 4, ratio 5x) that these defaults also flag; that is
    a deliberate bias toward caution, not a missed exclusion -- see the test
    fixture's own docstring for the full accounting. A structure that misses
    a real instance to avoid an extra caution flag is the wrong trade for a
    hazard that has already cost two killed headlines.

    Raises `PossibleAlignmentArtifact` on suspected misalignment.
    Returns `AgreementResult` (the s=0 score, `shift=0`) when no shift
    materially outperforms it -- this is the metric's only successful return
    shape; there is no parameter or code path that hands back a number the
    shift search itself flagged.
    """
    n = min(len(candidate), len(reference))
    s0 = sum(1 for x, y in zip(candidate[:n], reference[:n]) if x == y)

    best_n, best_shift = s0, 0
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            continue
        overlap = _shift_overlap(candidate, reference, shift)
        if overlap > best_n:
            best_n, best_shift = overlap, shift

    if best_shift != 0:
        gain = best_n - s0
        ratio = best_n / max(s0, 1)
        if gain >= min_absolute_gain and ratio >= min_relative_gain:
            raise PossibleAlignmentArtifact(
                s0_matches=s0, best_matches=best_n, best_shift=best_shift, length=n
            )

    return AgreementResult(matches=s0, length=n, shift=0)
