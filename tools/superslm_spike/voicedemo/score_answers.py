#!/usr/bin/env python3
"""T-2064: score a graded `answers.json` against the T-2064 held-out ground truth, per D-SLM3199's
protocol -- the same Wilson-CI precision/recall computation and the same >=0.85 lower-CI-bound
precision-AND-recall bar (both fields) that `judge_a/validate.py`'s harness already uses, applied to
an answer sheet produced by a grading agent working from the blind packet (`build_blind_packet.py`)
instead of by a live model call per question.

Usage:
  python -m voicedemo.score_answers --answers path/to/answers.json
  python -m voicedemo.score_answers --answers path/to/answers.json --ground-truth path/to/_t2064_ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from voicedemo.judge_a.stats import wilson_interval  # noqa: E402
from voicedemo.judge_a.validate import PRECISION_RECALL_BAR  # noqa: E402

FIELDS = ("info_present", "attitude_present")
DEFAULT_GROUND_TRUTH_PATH = os.path.join(_THIS_DIR, "_t2064_ground_truth.json")


class AnswerSheetError(ValueError):
    """The answer sheet does not account for every item, or contains a malformed value -- never
    silently scored partially (mirrors `judge_a/validate.py`'s own no-silent-drop ledger discipline,
    applied here to a human/agent-produced answer sheet instead of a live judge call)."""


def load_ground_truth(path: str) -> dict[str, dict[str, bool]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["ground_truth"]


def load_answers(path: str) -> dict[str, dict[str, bool]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score(ground_truth: dict[str, dict[str, bool]], answers: dict[str, dict[str, bool]]) -> dict:
    """Joins `answers` against `ground_truth` by item_id, computes tp/fp/tn/fn and Wilson CIs for
    each field. Raises `AnswerSheetError` if any item is missing, extra, or malformed -- a partial
    answer sheet is never silently scored as if it were complete."""

    gt_ids = set(ground_truth)
    ans_ids = set(answers)
    missing = gt_ids - ans_ids
    extra = ans_ids - gt_ids
    if missing:
        raise AnswerSheetError(f"answers.json is missing {len(missing)} item(s): {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        raise AnswerSheetError(f"answers.json has {len(extra)} item(s) not in ground truth: {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}")

    for item_id, ans in answers.items():
        for field in FIELDS:
            if field not in ans:
                raise AnswerSheetError(f"{item_id}: missing field {field!r}")
            if not isinstance(ans[field], bool):
                raise AnswerSheetError(f"{item_id}.{field}: expected a JSON boolean, got {ans[field]!r} ({type(ans[field]).__name__})")

    results = {}
    for field in FIELDS:
        tp = fp = tn = fn = 0
        for item_id in ground_truth:
            truth = ground_truth[item_id][field]
            pred = answers[item_id][field]
            if truth and pred:
                tp += 1
            elif (not truth) and pred:
                fp += 1
            elif (not truth) and (not pred):
                tn += 1
            else:
                fn += 1

        predicted_positive = tp + fp
        actual_positive = tp + fn
        precision_ci = wilson_interval(tp, predicted_positive) if predicted_positive > 0 else wilson_interval(0, 1)
        recall_ci = wilson_interval(tp, actual_positive) if actual_positive > 0 else wilson_interval(0, 1)
        passes = precision_ci.lower >= PRECISION_RECALL_BAR and recall_ci.lower >= PRECISION_RECALL_BAR

        results[field] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision_ci, "recall": recall_ci, "passes": passes,
        }

    n_total = len(ground_truth)
    for field in FIELDS:
        r = results[field]
        accounted = r["tp"] + r["fp"] + r["tn"] + r["fn"]
        if accounted != n_total:
            raise AssertionError(f"{field}: accounted for {accounted} items, expected {n_total} -- a silent drop occurred")

    return results


def print_report(results: dict) -> bool:
    """Prints the verdict table; returns True iff every field passes."""

    all_pass = True
    for field in FIELDS:
        r = results[field]
        p, rc = r["precision"], r["recall"]
        verdict = "PASS" if r["passes"] else "FAIL"
        all_pass = all_pass and r["passes"]
        print(
            f"  {field}: tp={r['tp']} fp={r['fp']} tn={r['tn']} fn={r['fn']}  "
            f"precision={p.point_estimate:.4f} (95% CI [{p.lower:.4f}, {p.upper:.4f}])  "
            f"recall={rc.point_estimate:.4f} (95% CI [{rc.lower:.4f}, {rc.upper:.4f}])  "
            f"{verdict} (bar: lower CI >= {PRECISION_RECALL_BAR})"
        )
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'} "
          f"(both fields must clear precision>=0.85 AND recall>=0.85, lower Wilson-CI bound)")
    return all_pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="path to the graded answers.json")
    ap.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH_PATH, help="path to the held-out ground-truth mapping")
    args = ap.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    answers = load_answers(args.answers)
    results = score(ground_truth, answers)
    print(f"Scored {len(ground_truth)} items against {args.ground_truth}\n")
    all_pass = print_report(results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
