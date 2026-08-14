"""T-2064: proves `score_answers.py`'s confusion-matrix construction and Wilson-CI wiring are
correct, using a small synthetic ground truth with a hand-computable confusion matrix per field --
independent of the real 120-item corpus, so these numbers are exact and verifiable by direct
arithmetic, not by trusting the scorer's own output.
"""

from __future__ import annotations

import pytest

from ..judge_a.stats import wilson_interval
from ..score_answers import AnswerSheetError, score

# 100 synthetic items -- large enough that a perfect score's Wilson lower bound clears 0.85 (at
# small n even a perfect point estimate has a wide CI: n=10 perfect has lower~0.72, verified against
# judge_a/judge_a/stats.py's own wilson_interval directly; n>=45 here comfortably clears 0.85).
# info_present: items 1-50 True, 51-100 False (50/50 split).
# attitude_present: items 1-60 True, 61-100 False (60/40 split) -- a different split than
# info_present, so a bug that swapped or shared state between the two fields' computations surfaces.
_IDS = [f"S{i:03d}" for i in range(1, 101)]

_GROUND_TRUTH = {
    item_id: {
        "info_present": i <= 50,
        "attitude_present": i <= 60,
    }
    for i, item_id in enumerate(_IDS, start=1)
}


def _all_correct_answers():
    return {item_id: dict(_GROUND_TRUTH[item_id]) for item_id in _IDS}


def _known_error_answers():
    """Deliberately flips a known, hand-counted set of answers:

    info_present (ground truth: S001-S050 True, S051-S100 False):
      - S001-S010 flipped True->False   (10 false negatives)
      - S051-S055 flipped False->True   (5 false positives)
      => tp=40, fp=5, tn=45, fn=10  =>  precision = 40/45 = 8/9, recall = 40/50 = 0.8

    attitude_present (ground truth: S001-S060 True, S061-S100 False):
      - S001-S015 flipped True->False   (15 false negatives)
      - no false positives
      => tp=45, fp=0, tn=40, fn=15  =>  precision = 45/45 = 1.0, recall = 45/60 = 0.75

    (Same precision/recall fractions as a smaller 8/1/9/2 and 9/0/8/3 design would give -- scaled
    5x so the all-correct sheet's own n is large enough to demonstrate a clean PASS, per the module
    docstring above.)
    """

    answers = {item_id: dict(_GROUND_TRUTH[item_id]) for item_id in _IDS}
    for i in range(1, 11):
        answers[f"S{i:03d}"]["info_present"] = False
    for i in range(51, 56):
        answers[f"S{i:03d}"]["info_present"] = True
    for i in range(1, 16):
        answers[f"S{i:03d}"]["attitude_present"] = False
    return answers


def test_all_correct_sheet_scores_precision_and_recall_1_on_both_fields():
    results = score(_GROUND_TRUTH, _all_correct_answers())

    info = results["info_present"]
    assert (info["tp"], info["fp"], info["tn"], info["fn"]) == (50, 0, 50, 0)
    assert info["precision"].point_estimate == 1.0
    assert info["recall"].point_estimate == 1.0
    assert info["passes"] is True

    attitude = results["attitude_present"]
    assert (attitude["tp"], attitude["fp"], attitude["tn"], attitude["fn"]) == (60, 0, 40, 0)
    assert attitude["precision"].point_estimate == 1.0
    assert attitude["recall"].point_estimate == 1.0
    assert attitude["passes"] is True


def test_known_error_sheet_produces_the_exact_hand_counted_confusion_matrix():
    results = score(_GROUND_TRUTH, _known_error_answers())

    info = results["info_present"]
    assert (info["tp"], info["fp"], info["tn"], info["fn"]) == (40, 5, 45, 10)
    assert info["precision"].point_estimate == pytest.approx(40 / 45)
    assert info["recall"].point_estimate == pytest.approx(40 / 50)

    attitude = results["attitude_present"]
    assert (attitude["tp"], attitude["fp"], attitude["tn"], attitude["fn"]) == (45, 0, 40, 15)
    assert attitude["precision"].point_estimate == pytest.approx(1.0)
    assert attitude["recall"].point_estimate == pytest.approx(45 / 60)


def test_known_error_sheet_ci_bounds_match_an_independent_direct_wilson_call():
    """Proves score()'s CI wiring (which (successes, n) pair it hands to wilson_interval for
    precision vs recall) is correct by comparing against an independently-constructed call, not by
    trusting score()'s own arithmetic."""

    results = score(_GROUND_TRUTH, _known_error_answers())

    info = results["info_present"]
    expected_precision_ci = wilson_interval(40, 45)   # tp=40, predicted_positive=tp+fp=45
    expected_recall_ci = wilson_interval(40, 50)      # tp=40, actual_positive=tp+fn=50
    assert info["precision"].lower == expected_precision_ci.lower
    assert info["precision"].upper == expected_precision_ci.upper
    assert info["recall"].lower == expected_recall_ci.lower
    assert info["recall"].upper == expected_recall_ci.upper

    attitude = results["attitude_present"]
    expected_precision_ci = wilson_interval(45, 45)   # tp=45, predicted_positive=45+0
    expected_recall_ci = wilson_interval(45, 60)      # tp=45, actual_positive=45+15
    assert attitude["precision"].lower == expected_precision_ci.lower
    assert attitude["recall"].lower == expected_recall_ci.lower
    assert attitude["recall"].upper == expected_recall_ci.upper


def test_known_error_sheet_fails_the_085_bar_on_info_present_recall():
    """info_present's recall lower-CI bound at 40/50=0.8 is well under 0.85 -- a real, verifiable
    FAIL, not just a low point estimate (mirrors judge_a/validate.py's own lower-CI-bound gate,
    Sec.8 P2)."""

    results = score(_GROUND_TRUTH, _known_error_answers())
    info = results["info_present"]
    assert info["recall"].lower < 0.85
    assert info["passes"] is False


def test_missing_item_in_answers_raises_rather_than_silently_scoring_partial():
    answers = _all_correct_answers()
    del answers["S005"]
    with pytest.raises(AnswerSheetError, match="missing"):
        score(_GROUND_TRUTH, answers)


def test_extra_item_in_answers_raises():
    answers = _all_correct_answers()
    answers["S999-not-a-real-item"] = {"info_present": True, "attitude_present": True}
    with pytest.raises(AnswerSheetError, match="not in ground truth"):
        score(_GROUND_TRUTH, answers)


def test_non_boolean_value_raises():
    answers = _all_correct_answers()
    answers["S001"]["info_present"] = "true"  # a string, not a JSON boolean
    with pytest.raises(AnswerSheetError, match="JSON boolean"):
        score(_GROUND_TRUTH, answers)


def test_missing_field_raises():
    answers = _all_correct_answers()
    del answers["S001"]["attitude_present"]
    with pytest.raises(AnswerSheetError, match="attitude_present"):
        score(_GROUND_TRUTH, answers)
