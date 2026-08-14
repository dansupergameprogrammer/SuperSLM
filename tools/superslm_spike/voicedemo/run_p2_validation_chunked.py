#!/usr/bin/env python3
"""T-2061: chunked, resumable, checkpoint-durable P2 real validation runner.

Same judge (`OllamaJudgeBackend`), same prompts, same Wilson-CI scoring as `run_p2_validation.py` --
this script exists only to run it in bounded, foreground, actively-driven chunks (T-811's
no-unwatched-backgrounding practice, sharpened after an earlier run's watchdog-stall incident): each
invocation processes at most `--max-new-calls` judge calls, writes each ledger row to a JSON
checkpoint immediately (per-record durable, matching the plan's own P5 discipline), and exits cleanly
so the caller can inspect progress and re-invoke rather than block unboundedly.

Usage (repeat until it prints "ITERATION COMPLETE" for all iterations you want):
  python -m voicedemo.run_p2_validation_chunked --iteration 1 --max-new-calls 60
  python -m voicedemo.run_p2_validation_chunked --iteration 1 --max-new-calls 60   # resumes
  ...
  python -m voicedemo.run_p2_validation_chunked --iteration 1 --report            # print final report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from voicedemo.event_bank import build_entries  # noqa: E402
from voicedemo.judge_a.backend import OllamaJudgeBackend  # noqa: E402
from voicedemo.judge_a.stats import wilson_interval  # noqa: E402
from voicedemo.judge_a.validate import PRECISION_RECALL_BAR, _collect_reactions_with_entry  # noqa: E402
from voicedemo.run_p2_validation import _PROMPT_VARIANTS  # noqa: E402
from voicedemo.validation_reactions import attach_validation_reactions  # noqa: E402

CHECKPOINT_DIR = os.path.join(_THIS_DIR, "corpus", "p2_chunked_checkpoints")


def _checkpoint_path(iteration: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f"iteration_{iteration}.json")


def _plan_calls(entries) -> list[tuple[str, int, str, str, str]]:
    """One row per judge call, in a fixed deterministic order: (event_id, reaction_index, field,
    reaction_text, target). `target` is the required_info_item for "info", or the intended_attitude
    for "attitude" -- resolved once here so a resumed run never re-derives it differently."""

    rows = []
    for event_id, idx, vr, entry in _collect_reactions_with_entry(entries):
        if vr.info_target_index < 0:
            raise ValueError(f"{event_id}[{idx}]: info_target_index=-1 not supported (see validate.py)")
        info_item = entry.required_info[vr.info_target_index]
        rows.append((event_id, idx, "info_present", vr.text, info_item, vr.info_present))
        rows.append((event_id, idx, "attitude_present", vr.text, entry.intended_attitude, vr.attitude_present))
    return rows


def _load_checkpoint(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, path)  # atomic on both POSIX and Windows -- no half-written checkpoint on crash


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--max-new-calls", type=int, default=60)
    ap.add_argument("--report", action="store_true", help="print the final report instead of running more calls")
    args = ap.parse_args()

    entries = attach_validation_reactions(build_entries())
    plan = _plan_calls(entries)
    total_calls = len(plan)

    ckpt_path = _checkpoint_path(args.iteration)
    done_rows = _load_checkpoint(ckpt_path)
    done_keys = {(r["event_id"], r["reaction_index"], r["field"]) for r in done_rows}

    print(f"Iteration {args.iteration}: {len(done_rows)}/{total_calls} judge calls already checkpointed.")

    if args.report:
        if len(done_rows) < total_calls:
            print(f"NOT COMPLETE -- {total_calls - len(done_rows)} calls remain.")
            return 1
        _print_report(args.iteration, done_rows)
        return 0

    if len(done_rows) >= total_calls:
        print("ITERATION COMPLETE -- nothing left to run. Use --report to print the final numbers.")
        return 0

    info_fn, attitude_fn = _PROMPT_VARIANTS[args.iteration - 1]
    kwargs = {}
    if info_fn is not None:
        kwargs["info_prompt_fn"] = info_fn
    if attitude_fn is not None:
        kwargs["attitude_prompt_fn"] = attitude_fn
    print(f"Provisioning judge (iteration {args.iteration}, "
          f"{'canonical prompt' if info_fn is None else 'reworded prompt'})...")
    backend = OllamaJudgeBackend(**kwargs)
    print(f"Judge disclosed as: {backend.model_id}")

    remaining = [row for row in plan if (row[0], row[1], row[2]) not in done_keys]
    to_process = remaining[: args.max_new_calls]
    print(f"Processing {len(to_process)} new calls this invocation "
          f"({len(done_rows)} done, {total_calls - len(done_rows) - len(to_process)} will remain after)...")

    t0 = time.time()
    for i, (event_id, idx, field, text, target, ground_truth) in enumerate(to_process, 1):
        if field == "info_present":
            predicted = backend.answer_info(text, target)
        else:
            predicted = backend.answer_attitude(text, target)
        done_rows.append(
            {
                "event_id": event_id,
                "reaction_index": idx,
                "field": field,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": predicted == ground_truth,
            }
        )
        _save_checkpoint(ckpt_path, done_rows)  # per-record durable -- every call lands before the next starts
        if i % 20 == 0 or i == len(to_process):
            elapsed = time.time() - t0
            print(f"  ... {i}/{len(to_process)} this batch ({len(done_rows)}/{total_calls} total), "
                  f"{elapsed:.1f}s elapsed, unparseable={backend.unparseable_count}")

    print(f"Batch complete: {len(done_rows)}/{total_calls} total checkpointed for iteration {args.iteration}.")
    if len(done_rows) >= total_calls:
        print("ITERATION COMPLETE.")
        _print_report(args.iteration, done_rows)
    else:
        print(f"{total_calls - len(done_rows)} calls remain -- re-invoke to continue.")
    return 0


def _print_report(iteration: int, rows: list[dict]) -> None:
    print(f"\n=== Iteration {iteration} report ({len(rows)} ledger cells) ===")
    for field in ("info_present", "attitude_present"):
        field_rows = [r for r in rows if r["field"] == field]
        tp = sum(1 for r in field_rows if r["ground_truth"] and r["predicted"])
        fp = sum(1 for r in field_rows if (not r["ground_truth"]) and r["predicted"])
        tn = sum(1 for r in field_rows if (not r["ground_truth"]) and (not r["predicted"]))
        fn = sum(1 for r in field_rows if r["ground_truth"] and (not r["predicted"]))
        predicted_positive = tp + fp
        actual_positive = tp + fn
        precision_ci = wilson_interval(tp, predicted_positive) if predicted_positive > 0 else wilson_interval(0, 1)
        recall_ci = wilson_interval(tp, actual_positive) if actual_positive > 0 else wilson_interval(0, 1)
        passes = precision_ci.lower >= PRECISION_RECALL_BAR and recall_ci.lower >= PRECISION_RECALL_BAR
        print(
            f"  {field}: tp={tp} fp={fp} tn={tn} fn={fn}  "
            f"precision={precision_ci.point_estimate:.4f} (CI [{precision_ci.lower:.4f}, {precision_ci.upper:.4f}])  "
            f"recall={recall_ci.point_estimate:.4f} (CI [{recall_ci.lower:.4f}, {recall_ci.upper:.4f}])  "
            f"{'PASS' if passes else 'FAIL'} (bar: lower CI >= {PRECISION_RECALL_BAR})"
        )


if __name__ == "__main__":
    sys.exit(main())
