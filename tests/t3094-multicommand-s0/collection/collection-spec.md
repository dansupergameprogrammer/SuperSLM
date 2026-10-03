# T-3103 natural-utterance collection specification (plan rev 7 §5; D-SLM8075 holds all labels)

2026-10-03 · test author · status: **collection open; nothing labeled, nothing frozen**.

## Index

1. [What is collected](#1-what-is-collected)
2. [What is withheld until the condition revision](#2-what-is-withheld-until-the-condition-revision)
3. [Strata and quotas](#3-strata-and-quotas)
4. [Writer commission](#4-writer-commission)
5. [Submission format and intake](#5-submission-format-and-intake)

## 1. What is collected

Complete, natural player requests addressed to an in-game squad-command interface, typed or transcribed, with transcript noise where the stratum calls for it. Each row is **text only**, with provenance. No row carries an intent, command, trigger, condition, cue, watch preference, route status, count, arm, or any other label. Labels are written later, by independent readers under the frozen guide, after the planner's condition revision (D-SLM8075) fixes the IntentDoc fields.

The T-3094 v3–v5 corpora, the rev-4 matched-pair drafts and the T-3100 boundary drafts are repair evidence, not candidate rows (plan rev 7 §8 S0). None of their text is reused.

## 2. What is withheld until the condition revision

- The IntentDoc field set and the representation of a gating condition (what is observed, and what state or event of it must hold).
- The condition strata the revision adds, with their quotas.
- The PC-2 reuse/new-watch pair challenge. Its two arms are defined by `watch_request`, which is part of the trigger fields under revision; pair commissioning waits for the revised field.
- Every label: intent gold, final-batch gold, cap class, route status, ambiguity adjudication.
- The candidate prompt and the annotation guide.

Rows whose meaning includes a gating condition ("fire when the beacon is lit") are **collected now, unlabeled**. Their wording is not steered toward any field representation.

## 3. Strata and quotas

Quotas are minimum counts of distinct held-out rows. Development rows come from separate sessions and do not count toward them.

| Stratum (commission key) | Held-out minimum | Meaning the writer is commissioned to express |
|---|---:|---|
| `conjunction` | 100 | two or more actions joined in one request |
| `punctuation_free` | 100 | transcript-style request with no punctuation |
| `correction` | 100 + 40 reserved | the player changes or corrects part of a request mid-utterance |
| `shared_argument` | 100 + 40 reserved | one actor or target shared across several actions |
| `pronoun` | 100 + 40 reserved | later actions refer back by pronoun |
| `nested_condition` | 100 + 40 reserved | an action gated on something observed in the world, possibly nested |
| `parallel` | 100 + 40 reserved | different actors given different simultaneous orders |
| `cancellation` | 100 | revokes a prior order and asks for nothing new |
| `zero_action` | 100 | talk, questions or remarks that ask for no action |
| `above_cap` | 100 | requests for many actions at once (more than four) |
| `at_cap` | 40 | asks for move, scan, hold and a fire that waits on something observed |
| `cap_plus_one` | 40 | the at-cap request plus one more separately requested action |
| `ambiguous_executable` | 40 | cancellation-like wording followed by a plain new request |
| development | ≥ 200 | any mix of the above, from sessions disjoint from held-out |

Total held-out minimum: 1,000 across the ten general strata (the 100 cancellation and 100 zero-action rows are also the dedicated no-action denominators, plan rev 7 §5) + 200 reserved compositional + 80 cap boundary + 40 ambiguous executable = **1,320 rows**, plus the condition strata the revision adds. Every count here is a collection target, not an observed population.

## 4. Writer commission

- **Independent writers.** At least two writers who have not seen T-3094, T-3100 or T-3101 text, traces, specs or labels. The test author does not write rows.
- **Meaning-only commission.** Each writer receives this file's §3 meaning column and the game's action vocabulary (`fire`, `move`, `scan`, `hold`, `wait`), never example sentences, fixed phrasings or label names.
- **Natural wording.** A writer writes as a player would type or say it. No fixed answer phrase, repeated clause, appended key, template or copied body across rows; no stratum may be recognizable from a stock phrase. Condition rows use the condition the player cares about in plain words ("when the beacon is lit", "once the gate is open", "if they start running"), varying the observed entity and its state or event.
- **Sessions.** A writer produces rows in sessions of 10–40 rows with a stable `speaker_id` per session. A whole session goes to one split.
- **Flags.** A writer flags any row they believe a player would not plausibly say; flagged rows are rewritten, not kept.
- **Naturalness review.** Two independent game-domain readers review every row for plausibility, blind to stratum (plan rev 7 §5); any flag sends the row back to its writer.

## 5. Submission format and intake

One JSONL file per writer. Each line has exactly these keys:

`row_id` (writer-unique string), `utterance` (nonempty string), `writer_id`, `session_id`, `speaker_id`, `source_kind` (`synthetic_writer` or `recorded_player`), `commission` (one stratum key from §3, or `development`), `writer_flag` (boolean), `writer_note` (string, may be empty).

`collection_intake.py <submission.jsonl>... [--quotas]` checks the format, refuses any label-bearing key, refuses duplicate rows after casefold and whitespace normalization across all submissions, refuses a session that spans two commissions' splits or two speakers, and reports counts per stratum. With `--quotas` it also refuses any §3 shortfall. Intake acceptance is a structural gate only; it is not a naturalness, identifiability or shortcut verdict, and the commission key never reaches a model or a first-pass reader.
