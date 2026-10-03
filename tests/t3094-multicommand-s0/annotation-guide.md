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

The T-3096/T-3097 round for v2 is void, and v3 was held before annotation. Do not reuse labels, votes, adjudications or ambiguity decisions from either version. For v4, the conductor first sends `blind-annotation-input.jsonl`, this guide and `command.schema.json` to two new independent annotators. That first export has opaque case IDs and raw utterances only. Each annotator records whether the raw text requests an independently tracked wait or merely makes the signal a condition, plus route status, ordered command objects, ambiguity note and reason. The conductor locks those raw-only readings before either annotator receives `annotation-input.jsonl`, which adds the game event traces. The later trace resolves game facts and disagreements; a `signal_listener_registered` event gates a fire request, while a queued `signal_watch` task with its own completion callback is an independently tracked action. If the trace conflicts with a reasonable raw reading, flag the row and repair the corpus before gold rather than taking the trace as hidden instructions to the model.

The candidate model receives only each raw utterance string from `model-input.jsonl`; it never receives the JSONL wrapper, case ID, source class, pair ID or arm, split, game trace, annotation output or gold. Do not send annotators corpus-level diversity scores, the other arm's labels, model output, prompts, tuning results or grader verdicts. The utterances and sessions are synthetic; no recorded player audio is claimed. The conductor compares exact labels, adjudicates every disagreement and freezes new gold and split hashes before any candidate model output. Unresolved ambiguity stays visible and cannot be dropped or scored as correct. No person may author both annotation arms.
