# IntentDoc v2 annotation guide — T-3103 S0, frozen

2026-10-03 · test author · plan rev 15 (`d628055561`) §1, §1.1, §1.2, §4.3, §5.2 · status: **frozen before any row is written or model output exists**.

## Index

1. [Your task, and what you may see](#1-your-task-and-what-you-may-see)
2. [Route status](#2-route-status)
3. [The document you write](#3-the-document-you-write)
4. [Condition atoms](#4-condition-atoms)
5. [Normalization: must and must not](#5-normalization-must-and-must-not)
6. [Role and polarity by construction (§1.2, verbatim)](#6-role-and-polarity-by-construction-12-verbatim)
7. [Acceptable-span sets](#7-acceptable-span-sets)
8. [Flags](#8-flags)
9. [Output format](#9-output-format)

## 1. Your task, and what you may see

You read one raw player utterance at a time. You see only what the model sees: the utterance text and this guide. You never see:
- a catalogue, game state or event trace;
- another annotator's labels;
- the row's stratum, commission or pair;
- any model output, prompt or score.

You write the gold IntentDoc v2 document the utterance requests, its route status, and the acceptable-span sets for every atom's `entity`, `state` and `value`. Alternatively you flag the row (§8). Your reading locks when you submit it. A later stage, run by a separate seat, derives final command batches from the locked intents and versioned game fixtures; it never changes your intent reading.

Two annotators label every row independently. No seat labels both arms. Disagreements are adjudicated against this guide, and a row that stays ambiguous is repaired, never dropped and never scored as correct.

## 2. Route status

- `Executable`: the utterance requests at least one action. Annotate the current effective request; a cancellation-like phrase followed by a new instruction is `Executable` (annotate the new instruction).
- `NoAction`: talk, questions, reports or remarks that request no action, including "when did Alpha arrive?".
- `Cancelled`: the utterance revokes an earlier order and requests nothing new.

`NoAction` and `Cancelled` rows have no document.

## 3. The document you write

One object, `{"intents":[...]}`, with at least one intent. Each intent has exactly these fields, in this order (plan §1):

| Field | Type and meaning |
|---|---|
| `action` | String enum `fire`, `move`, `scan`, `hold`, `wait`: requested goal, not a guaranteed executable game action. `wait` is used for a standalone watch goal; a watch that gates `fire` is a condition atom on the `fire` intent. For the §1.2 role rules, `fire` is **punctual** and `move`, `scan`, `hold` and `wait` are **durative**; this is a property of the closed action vocabulary, not of a game. |
| `subject`, `target` | Meaningful nonempty strings resolving the requested actor and object/location from the utterance. For a standalone `wait`, `target` names what is watched. |
| `conditions` | Homogeneous array, `minItems: 0`, of condition atoms (§1.1). `[]` means the action is not gated by any observed condition. Each atom gates **this** intent only. |
| `depends_on_previous` | Boolean expressing a requested dependency on the prior **intent**'s completion (“then”, “after that”). “And” between requested actions is not a dependency. Completion of the previous intent is never written as an atom. |

Write keys in the order shown. Every object has exactly these keys. Strings are meaningful: never empty, never punctuation only.

## 4. Condition atoms

Each atom has exactly these seven fields, in this order (plan §1.1):

| Field | Type and meaning |
|---|---|
| `role` | String enum `start`, `start_any`, `stop`, `during`. `start`: the action may begin only when the atom holds. `start_any`: one of two or more alternatives that can each start the action (“or”). `stop`: the durative action ends when the atom holds (“hold the bridge until the beacon is lit”). `during`: the action proceeds only while the atom holds (“keep scanning while the gate is open”). §1.2 assigns the role for every construction. |
| `entity` | Meaningful string: the observed thing as the player named it, with pronouns and possessives resolved to their referent by §1.2 rows 14–15 and one leading article or demonstrative removed. |
| `state` | Meaningful string: the state, event or measured property that must hold of the entity, in the player's words without copula, auxiliary or determiner: `lit`, `destroyed`, `arrives`, `open`, `health`, `in range`. |
| `cmp` | String enum `none`, `below`, `above`, `at_most`, `at_least`, `reaches`. `none` for a plain state or event. Otherwise the comparison applies to the measured property named by `state`, or, when `state` is a state or event, to the number of `entity` instances for which it holds (“when fewer than two turrets are standing”). `reaches` means the value is crossed from whichever side it starts on. |
| `value` | Meaningful string: the quantity as spoken (`half`, `30 percent`, `three`, `2`, `none`, `all`), or exactly the one-byte sentinel `-` when `cmp` is `none`. A spoken quantifier is copied, never converted: “when none of the turrets are standing” is `cmp: at_most`, `value: "none"`; “once all the turrets are down” is `cmp: at_least`, `value: "all"`. The sentinel cannot collide with a spoken quantity, because a spoken value is a word or numeral span of the utterance (§5.2 spoken-span check) and the semantic rules compare exact bytes. |
| `negated` | Boolean. `true` when the condition is that the state does **not** hold (“unless the gate is open”, “while the pulse is not done”, “don't fire while the pulse is pending”). Negation of the gated action itself is never written here: “don't fire until the beacon is lit” is a `start` atom with `negated: false`. |
| `watch` | String enum `auto`, `reuse`, `new`: the player's expressed cue-handling preference for this observation. It is a preference, **not** a claim that the game tracks a wait (D-SLM8048, D-SLM8067); §4.5 resolves it for every role. |

**Combination semantics — the contract the adapter and evaluator implement.** Within one intent: all `start` atoms must hold; if any `start_any` atoms are present, at least one of them must also hold; the action ends when any `stop` atom holds; the action proceeds only while all `during` atoms hold, and the catalogue's declared policy for the action decides whether a `during` atom ceasing to hold suspends or ends it. An atom holds when its predicate (with comparison) holds, inverted when `negated` is `true`. With no `start`, `start_any` or `during` atom the action may begin at once, subject to its requested dependency. Whether a `start` atom on a state already true at issue time is satisfied immediately (level) or waits for the next occurrence (edge) is a property of the catalogue predicate the adapter binds, not a model field. §4.5 states the evaluation exactly and gives it executed cells. This supports any conjunction of start conditions with one disjunctive group, stop on any of several conditions, and while-true gating.

**Semantic rules, checked on exact bytes after parse and before the resolver (S3).** (a) `cmp` is `"none"` exactly when `value` is exactly `"-"`. (b) `start_any` appears on zero or at least two atoms of an intent; a lone alternative is written `start`. (c) `entity` and `state` are never exactly `"-"`. No normalization, case folding or canonicalizer C is applied before these rules, so `"None"`, `"none"` and `" -"` are ordinary strings. A violation is the named whole-batch refusal `InvalidConditionAtom`. Atoms are written in the order the player mentioned them; graders are order-insensitive within an intent, and S3 does not reject order.

## 5. Normalization: must and must not

| The model MUST | The model MUST NOT |
|---|---|
| Resolve pronouns, possessives and ellipsis in `entity` (`it`, `there`, `your`, `my`, `its`, `their`) by §1.2 rows 14–15. | Map an entity or state onto a synonym it believes the game uses (`burning` → `lit`, `signal fire` → `beacon`). |
| Separate negation of the condition (`negated`) from negation of the premature action (“don't/hold fire until X” → `start`, `negated: false`). | Convert a quantity (`half` → `50%` or `0.5`, `none` → `0`), or invent a unit. |
| Choose `role` and `negated` from the construction by §1.2. | Decide edge versus level, whether a watch is tracked, or whether the entity exists in the world. |
| Normalize comparison direction into `cmp` (“drops below/falls under/less than” → `below`; “or less” → `at_most`; “hits/reaches” → `reaches`; “none of … are” → `at_most`; “all … are” → `at_least`). | Resolve a reference against game state (“the nearest tower”); it copies the phrase. |
| Place each atom on every intent it gates; a fronted condition over coordinated actions is copied onto each of them. | Invent a condition the player did not state, or turn “then/after that” between requested actions into an atom. |
| Split a conjunction into separate atoms, mark alternatives `start_any`, and write one atom per condition clause (§1.2 row 12). | Treat concurrency (“scan while reserve holds the tower”), a question (“when did Alpha arrive?”), a modifier (“fire at the lit beacon”) or a non-conditional connective use (§5.2 D7–D11) as a condition. |
| Strip one leading article or demonstrative from `entity` and the copula/auxiliary from `state` (“is lit” → `lit`). | Paraphrase `state` (“is lit” must not become `on`). |

**Out of the v2 contract.** A disjunction that needs more than one alternative group (“when A and B, or when C”), a condition on a condition (“once the beacon is lit after the gate opens”), explicit edge markers (“lit again”, “next time”), boundary inclusivity beyond the `cmp` enum, “until both A and B”, a deadline on an observed event (“fire before the gate opens”, §1.2 row 9), and repetition of a punctual action until a condition (“keep firing until it falls”, §1.2 row 17). These are measured in a reported stress stratum with no bar (§5.2); the adapter refuses what it can detect. They are recorded as claim limits (§9), not silently mis-scored.

## 6. Role and polarity by construction (§1.2, verbatim)

Every row of this table fixes `role` and `negated` for one construction, with one gold. "Punctual" is `fire`; "durative" is `move`, `scan`, `hold` and `wait`.

| # | Construction | Gold | Example and gold atom |
|---|---|---|---|
| 1 | `when`, `once`, `as soon as`, `the moment`, `if`, `after` + an observed state or event | `start`, `negated: false` | “fire when the beacon is lit” → `start` beacon/lit |
| 2 | `unless X`, `if not X`, `if X isn't …`, punctual action | `start`, `negated: true` | “fire at the bridge unless the gate is open” → `start` gate/open, `true` |
| 3 | `unless X`, `if not X`, durative action | `during`, `negated: true` | “keep scanning unless the alarm is sounding” → `during` alarm/sounding, `true` |
| 4 | A negated or withheld action + `until`/`till` X (“don't fire until”, “hold fire till”, “never fire until”) | `start`, `negated: false` | “don't fire until the beacon is lit” → `fire` intent, `start` beacon/lit |
| 5 | `not before X`, `no sooner than X`, `only after X` | `start`, `negated: false` | “fire not before the gate opens” → `start` gate/opens |
| 6 | Durative action + `until`/`till` X | `stop`, `negated: false` | “hold the bridge until the beacon is lit” → `hold` intent, `stop` beacon/lit |
| 7 | `while`/`as long as` X, X a state or property of an observed entity | durative: `during`, `false`; punctual: `start`, `false` | “keep scanning while the gate is open” → `during` gate/open; “fire while the gate is open” → `start` gate/open |
| 8 | A negated action + `while` X (“don't fire while the pulse is pending”) | punctual: `start`, `negated: true`; durative: `during`, `negated: true` | “don't fire while the pulse is pending” → `start` pulse/pending, `true`. Assigned to F5 only. |
| 9 | A **non-negated** action + `before X`, X an observed state or event (a negated or withheld action is row 16) | Out of contract (a deadline). Closest v2 reading, flagged, stress stratum S6: punctual `start`, `negated: true`; durative `during`, `negated: true`. The gold also records the expected resolver outcome: on an event predicate `NegatedEventUnsupported` unless the catalogue declares a negated-event policy (none in the reference fixtures); on a state predicate it resolves. | “fire before the gate opens” → `start` gate/opens, `true`, flag; expected `NegatedEventUnsupported` |
| 10 | `before`/`after` + another **requested** action | No atom; `depends_on_previous` on the later intent in spoken order of execution (decoy D4) | “scan, and fire after you scan” → `fire` with `depends_on_previous: true` |
| 11 | `while` + a friendly actor performing an action (“scan while reserve holds the tower”) | No atom; concurrency (decoy D1) | — |
| 12 | One gate stated in two clauses (“don't fire while the pulse is pending; fire once it completes”) | One atom per condition clause; no merging | → `start` pulse/pending `true` and `start` pulse/completes `false` |
| 13 | “hold fire” | Always a withheld `fire` intent (row 4), never a `hold` intent | “hold fire till the flare” → `fire`, `start` flare/… |
| 14 | Possessive pronoun on a compared or measured noun (“your health drops below half”, “my ammo”) | `entity` = the possessor's referent (the addressee's name for `your`, `player` for `my`, the antecedent for `its`/`their`); `state` = the noun | → `bravo squad` / `health` / `below` / `half` |
| 15 | Possessed part followed by its own state or event (“when the tower's shield is down”, “once Alpha's door opens”) | `entity` = possessor + part (`tower shield`); `state` = the predicate (`down`). C removes the possessive clitic, so the spoken copy `tower's shield` is the same canonical string and passes the spoken-span check (§5.1, §5.2 item 7). | → `start` tower shield/down |
| 16 | A negated or withheld action + `before X` (“don't fire before the horn sounds”, “keep fire off before the gate opens”) | `start`, `negated: false`, as row 4 | “do not fire at north ridge before the horn interval finishes” → `fire`, `start` horn interval/finishes |
| 17 | A non-negated **punctual** action + `until X` (“keep firing on the tower until it falls”) | Out of contract (repetition of a punctual action). Gold `stop` on the `fire` intent, flagged, stress stratum S7; expected resolver outcome `StopOnPunctualAction` | → `stop` tower/falls, flag; expected `StopOnPunctualAction` |
| 18 | An anaphoric `afterward`, `after that` or `then` whose referent is a condition clause already written as an atom (“do not fire at north ridge before the horn interval finishes … fire at north ridge afterward”) | No further atom, no dependency and no second intent: the clause restates the gated action of rows 4/16. When the referent is a previously requested action, it is `depends_on_previous` (row 10). | → one `fire` intent, `start` horn interval/finishes |

## 7. Acceptable-span sets

For each atom, give the set of strings that count as correct for `entity`, `state` and `value`.

- Every member is a **contiguous span of the utterance** after the grader's canonicalizer C. C applies case, one leading article or demonstrative, one leading copula or auxiliary on `state`, the possessive clitic, number words zero to twenty and percent. A member that is not a spoken span blocks the freeze, so never add a synonym.
- **Include** the spoken span and its article or copula variants (`the gate`, `gate`; `is lit`, `lit`).
- **Pronouns and possessives:** the set is the antecedent's span, the addressee's name for `your`, and `player` for `my` (§1.2 rows 14–15).
- **`value`:** the quantity exactly as spoken (`half`, `30 percent`, `three`, `none`, `all`). The only non-spoken value is the sentinel `-`, and only with `cmp: none`.

## 8. Flags

- `ambiguous`: more than one reading survives (scope, binding, role or count). State the readings in `reason`.
- `out_of_contract`: an §1.1 "Out of the v2 contract" form. Give the closest v2 reading as gold and name the form.
- `implausible`: a player would not plausibly say this in a squad-command interface.

## 9. Output format

One JSON object per row, written into the sealed store, never outside it:

`{"row": "<row id>", "annotator": "<seat id>", "route": "Executable|NoAction|Cancelled", "document": <IntentDoc v2 or null>, "spans": [[{"entity": [...], "state": [...], "value": [...]}, ...], ...], "flags": [...], "reason": "..."}`

`spans[i][j]` belongs to `document.intents[i].conditions[j]`.
