# T-2064 blind grading packet -- instructions

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
