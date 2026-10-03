# T-3094 held-out annotation guide, frozen before model output

## Index

1. [Unit and segmentation](#1-unit-and-segmentation)
2. [Routing and status](#2-routing-and-status)
3. [Command fields](#3-command-fields)
4. [Independent round](#4-independent-round)

## 1. Unit and segmentation

Each command item is one independently executable game action. A condition that gates an action is its `trigger` metadata. Write a separate `wait` item only when the game actually schedules and tracks a waiting action with its own completion or cancellation. Preserve utterance order. Resolve shared arguments and pronouns to the intended subject and target. Mark dependency on the preceding item only when the later action requires the earlier action to complete.

## 2. Routing and status

Label each raw utterance `Executable`, `NoAction` or `Cancelled`. A cancelled utterance revokes an executable request and has no command query. Pure discussion, questions without an instruction, and zero-action transcripts are `NoAction`. A cancellation-like phrase followed by a new authorized instruction is `Executable`; annotate the current effective request. Above-cap requests retain their intended full gold action sequence and a consumer rejection expectation. Never delete an ambiguous row after seeing a prediction; flag it for adjudication.

## 3. Command fields

Use the frozen schema's key order. `action` is one of its five enum values. `subject` names the actor or system, `target` the object/location, `trigger` the event gating execution or a meaningful string such as `immediate`, and `depends_on_previous` is a boolean. Do not use empty or punctuation-only strings. A triggered fire without a scheduled wait is one `fire` item. A tracked wait followed by fire is two items, with fire dependent on wait when specified by the trace.

## 4. Independent round

The conductor sends the same frozen item IDs, raw utterances, source-class metadata, schema, this guide and game action traces to two independent annotators. Each annotator works alone and sees neither the other annotator's labels nor any model output, prompts, tuning results or grader verdict. Return per-ID route status, ordered command objects, ambiguity note and reason. The conductor compares exact labels, routes every disagreement to adjudication using the game action trace and the rule above, then freezes gold and split hashes before the first model output. Unresolved ambiguity stays visible and cannot be dropped or scored as correct. No person may author both annotation arms.
