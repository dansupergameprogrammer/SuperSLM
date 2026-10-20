# Plan rev 15 tables, extracted verbatim (d628055561)

## Section 1.2 role and polarity by construction

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

## Section 1.1 what the model must and must not normalize

| The model MUST | The model MUST NOT |
|---|---|
| Resolve pronouns, possessives and ellipsis in `entity` (`it`, `there`, `your`, `my`, `its`, `their`) by §1.2 rows 14–15. | Map an entity or state onto a synonym it believes the game uses (`burning` → `lit`, `signal fire` → `beacon`). |
| Separate negation of the condition (`negated`) from negation of the premature action (“don't/hold fire until X” → `start`, `negated: false`). | Convert a quantity (`half` → `50%` or `0.5`, `none` → `0`), or invent a unit. |
| Choose `role` and `negated` from the construction by §1.2. | Decide edge versus level, whether a watch is tracked, or whether the entity exists in the world. |
| Normalize comparison direction into `cmp` (“drops below/falls under/less than” → `below`; “or less” → `at_most`; “hits/reaches” → `reaches`; “none of … are” → `at_most`; “all … are” → `at_least`). | Resolve a reference against game state (“the nearest tower”); it copies the phrase. |
| Place each atom on every intent it gates; a fronted condition over coordinated actions is copied onto each of them. | Invent a condition the player did not state, or turn “then/after that” between requested actions into an atom. |
| Split a conjunction into separate atoms, mark alternatives `start_any`, and write one atom per condition clause (§1.2 row 12). | Treat concurrency (“scan while reserve holds the tower”), a question (“when did Alpha arrive?”), a modifier (“fire at the lit beacon”) or a non-conditional connective use (§5.2 D7–D11) as a condition. |
| Strip one leading article or demonstrative from `entity` and the copula/auxiliary from `state` (“is lit” → `lit`). | Paraphrase `state` (“is lit” must not become `on`). |
