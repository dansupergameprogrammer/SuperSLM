#!/usr/bin/env python3
"""T-2064: build the blind grading packet for P2's validation set, per D-SLM3199 (Dan rules the
judge branch: an Opus agent grades a blind answer sheet; the local 8B judge is retired for grading,
the pin-and-disclose requirement is met by disclosure of the grading agent + fixed rubric + archived
judgments instead of local weights).

Emits two things:

1. The blind packet (`_t2064_packet/`, self-contained, safe to hand to a grading agent):
   - `instructions.md` -- the fixed per-field rubric (the exact entailment questions
     `judge_a/prompt.py`'s `INFO_PROMPT_TEMPLATE`/`ATTITUDE_PROMPT_TEMPLATE` already ask, restated for
     a one-pass-through-all-items agent rather than 240 separate API calls), the process discipline
     (one item at a time, no revisiting), and the exact `answers.json` output schema.
   - `items.json` -- all 120 validation reactions in randomized order under opaque IDs (`Q001..Q120`),
     each carrying only what the rubric needs to grade it: the reaction text, the specific
     `required_info_item` to check, and the `intended_attitude` to check. No label, no `label_kind`,
     no event id/kind, no split, no ordering hint back to the corpus's own construction (the corpus's
     `_make_four` always emits reactions in a fixed positive/neg-info/neg-attitude/neg-both order
     *per event* -- global randomization across all 120 breaks that pattern, not just a per-event
     shuffle, so no four-item cluster in the packet is recoverable as a unit).

2. The ground-truth mapping (`_t2064_ground_truth.json`, a **sibling** of the packet directory, NOT
   inside it) -- `{item_id: {info_present, attitude_present}}` for the same 120 items, used only by
   `score_answers.py` after a graded `answers.json` comes back. Never copied into the packet.

**Race context, considered and not included:** the coordinator's brief anticipated "event text and
race context." This corpus's `validation_reactions` are P2's calibration fixtures, and per
`events.py`'s own docstring the corpus entries they attach to are deliberately race-agnostic ("the
same event goes to all three voice adapters... race is a property of which adapter renders the
event, never of the event payload itself"). Neither `judge_a/prompt.py`'s templates nor this
20-reaction validation set reference race at all -- there is no race field to include without
inventing one the rubric does not ask for. (P5's later, race-specific adapter generations are a
different, not-yet-built set; this note applies to P2 only.)
"""

from __future__ import annotations

import json
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from voicedemo.event_bank import build_entries  # noqa: E402
from voicedemo.judge_a.validate import _collect_reactions_with_entry  # noqa: E402
from voicedemo.validation_reactions import attach_validation_reactions  # noqa: E402

PACKET_DIR = os.path.join(_THIS_DIR, "_t2064_packet")
GROUND_TRUTH_PATH = os.path.join(_THIS_DIR, "_t2064_ground_truth.json")  # sibling of PACKET_DIR, not inside it
SHUFFLE_SEED = 2064  # fixed and disclosed here -- reproducible packet generation, not a secret

INSTRUCTIONS_MD = """# T-2064 blind grading packet -- instructions

## What this is

120 items, each one validation reaction from the runtime-LoRA voice-demo's Judge-A calibration set
(`RuntimeLoRA_VoiceDemo_Plan.md` Sec.8 P2). For each item you make two independent yes/no judgments:
`info_present` and `attitude_present`. That is 240 judgments total.

This packet carries no labels and no hints about which items are "supposed to" be true or false.
Grade every item on its own text, using only the rubric below.

## Rubric -- `info_present`

You are given a **fact to check for** (`required_info_item`) and a **dialogue line**
(`reaction_text`).

**Question: does the dialogue convey this fact, directly or in different words?**

Answer `true` if the dialogue conveys the fact (even paraphrased, even briefly). Answer `false` if
the dialogue does not convey the fact -- including when the dialogue is on-topic in general but never
actually states or implies this specific fact.

## Rubric -- `attitude_present`

You are given a **tone to check for** (`intended_attitude`) and the same **dialogue line**
(`reaction_text`).

**Question: does the dialogue carry this tone?**

Answer `true` if the dialogue's tone matches the description. Answer `false` if it does not (including
a tone that is merely neutral, or a tone that is a plausible-sounding but different register).

## Process discipline

- Process items **one at a time, in the order given** in `items.json`.
- Judge each item **independently** -- do not revisit or change an earlier answer once made, and do
  not let a pattern you notice in earlier items' apparent balance (how many true/false so far) bias a
  later judgment. Each item's `reaction_text` plus its own `required_info_item`/`intended_attitude` is
  the entire basis for that item's two answers.
- `info_present` and `attitude_present` are graded independently of each other too -- a dialogue can
  convey the fact but carry the wrong tone, or vice versa.

## Output format

Write a single JSON file, `answers.json`, mapping every item's `item_id` to its two boolean answers:

```json
{
  "Q001": {"info_present": true, "attitude_present": false},
  "Q002": {"info_present": false, "attitude_present": true},
  ...
}
```

- Every one of the 120 `item_id`s in `items.json` must appear as a key. No item is skipped.
- Values are JSON booleans (`true`/`false`), not the strings `"YES"`/`"NO"` or `"True"`/`"False"`.
- No extra fields, no explanations, no confidence scores -- exactly `info_present` and
  `attitude_present` per item.
"""


def build_packet() -> None:
    entries = attach_validation_reactions(build_entries())
    reactions = _collect_reactions_with_entry(entries)  # (event_id, idx, vr, entry), corpus order

    rows = []
    for event_id, idx, vr, entry in reactions:
        if vr.info_target_index < 0:
            raise ValueError(f"{event_id}[{idx}]: info_target_index=-1 not supported by this packet builder")
        rows.append(
            {
                "reaction_text": vr.text,
                "required_info_item": entry.required_info[vr.info_target_index],
                "intended_attitude": entry.intended_attitude,
                "_ground_truth": {"info_present": vr.info_present, "attitude_present": vr.attitude_present},
            }
        )

    if len(rows) != 120:
        raise AssertionError(f"expected 120 validation reactions, got {len(rows)}")

    rng = random.Random(SHUFFLE_SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)  # global shuffle across all 120 -- breaks the corpus's own per-event 4-item pattern

    items = []
    ground_truth: dict[str, dict[str, bool]] = {}
    for i, row_idx in enumerate(order, start=1):
        item_id = f"Q{i:03d}"
        row = rows[row_idx]
        items.append(
            {
                "item_id": item_id,
                "reaction_text": row["reaction_text"],
                "required_info_item": row["required_info_item"],
                "intended_attitude": row["intended_attitude"],
            }
        )
        ground_truth[item_id] = row["_ground_truth"]

    os.makedirs(PACKET_DIR, exist_ok=True)

    items_path = os.path.join(PACKET_DIR, "items.json")
    with open(items_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"n_items": len(items), "items": items}, f, indent=2, sort_keys=False)
        f.write("\n")

    instructions_path = os.path.join(PACKET_DIR, "instructions.md")
    with open(instructions_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(INSTRUCTIONS_MD)

    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {
                "n_items": len(ground_truth),
                "shuffle_seed": SHUFFLE_SEED,
                "note": "held out of the packet; used only by score_answers.py after answers.json is returned",
                "ground_truth": ground_truth,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    # Structural self-check: confirm no ground-truth field leaked into the packet's own items.json.
    with open(items_path, "r", encoding="utf-8") as f:
        packet_text = f.read()
    for forbidden in ("ground_truth", "info_present", "attitude_present", "label_kind", "planted_negative"):
        if forbidden in packet_text:
            raise AssertionError(f"leak check failed: {forbidden!r} found in items.json")

    print(f"packet written: {PACKET_DIR}")
    print(f"  items.json: {len(items)} items, randomized (seed={SHUFFLE_SEED})")
    print(f"  instructions.md: {len(INSTRUCTIONS_MD)} chars")
    print(f"ground truth (outside packet): {GROUND_TRUTH_PATH}")
    print("leak check: passed -- no ground-truth field found anywhere in items.json")


if __name__ == "__main__":
    build_packet()
