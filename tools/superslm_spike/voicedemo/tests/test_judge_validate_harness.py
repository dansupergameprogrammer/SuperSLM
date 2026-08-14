"""P2 harness control-flow cells, run against `MockJudgeBackend` (see `judge_a/backend.py`'s module
docstring for why no real judge is wired up yet -- this proves the harness's own logic is correct,
independent of that standing blocker).
"""

from ..event_bank import build_entries
from ..judge_a.backend import MockJudgeBackend
from ..judge_a.validate import PRECISION_RECALL_BAR, check_corpus_sizing, run_validation
from ..validation_reactions import all_validation_reactions, attach_validation_reactions


def _entries():
    return attach_validation_reactions(build_entries())


def test_check_corpus_sizing_passes_on_the_real_corpus():
    check_corpus_sizing(_entries())  # must not raise


def test_perfect_judge_passes_with_precision_and_recall_1():
    entries = _entries()
    outcome = run_validation(entries, backend_factory=lambda i: MockJudgeBackend(error_rate=0.0))
    assert outcome.verdict == "PASS"
    assert len(outcome.iterations) == 1  # stops at the first iteration once it passes
    for field_result in outcome.best_iteration.field_results.values():
        assert field_result.precision_ci.point_estimate == 1.0
        assert field_result.recall_ci.point_estimate == 1.0
        assert field_result.precision_ci.lower >= PRECISION_RECALL_BAR
        assert field_result.recall_ci.lower >= PRECISION_RECALL_BAR


def test_ledger_accounts_for_every_reaction_twice_no_silent_drop():
    entries = _entries()
    n_reactions = len(all_validation_reactions(entries))
    outcome = run_validation(entries, backend_factory=lambda i: MockJudgeBackend(error_rate=0.0))
    # Two fields graded per reaction (info_present, attitude_present) -- plan's own "no silent drop"
    # ledger discipline (Sec.8 P5, applied here at P2's scale).
    assert len(outcome.best_iteration.ledger) == n_reactions * 2


def test_bad_judge_escalates_after_max_iterations():
    entries = _entries()
    # A high, fixed error rate that a real judge prompt iteration would not plausibly fix -- proves
    # the harness tries exactly MAX_ITERATIONS times and then reports ESCALATE with a ceiling,
    # rather than looping forever or silently lowering the bar.
    outcome = run_validation(entries, backend_factory=lambda i: MockJudgeBackend(error_rate=0.4, seed=i))
    assert outcome.verdict == "ESCALATE"
    assert len(outcome.iterations) == 3
    assert outcome.escalation_message != ""
    # The message must name the measured ceiling per field, not just say "failed".
    for field in ("info_present", "attitude_present"):
        assert field in outcome.escalation_message


def test_escalation_never_reports_a_passing_verdict():
    """The escalate-with-ceiling path must never claim PASS even if one field happened to clear the
    bar while the other did not -- both fields must pass for the overall verdict to be PASS (plan
    Sec.8 P2: "Precision >= 0.85 and recall >= 0.85 ... on both labels")."""

    entries = _entries()
    # error_rate chosen so info_present (which happens to have a simpler positive/negative split in
    # this corpus's construction) may occasionally look fine while attitude_present does not -- the
    # verdict must still require both.
    outcome = run_validation(entries, backend_factory=lambda i: MockJudgeBackend(error_rate=0.3, seed=i * 7))
    if outcome.verdict == "PASS":
        for field_result in outcome.best_iteration.field_results.values():
            assert field_result.passes
    else:
        assert outcome.verdict == "ESCALATE"
        assert not outcome.best_iteration.all_fields_pass
