# T-3103 S0 abstract design matrix — MX-1

2026-10-03 · test author (eval-side reader, plan rev 15 §5.3) · plan rev 15 at `d628055561` (§0.1, §1.2, §5.2, §5.3, §8 R14-W1 and CA14-2) · status: **frozen for the conductor's partition**.

## Index

1. [What the matrix is](#1-what-the-matrix-is)
2. [Row unit and assignment units](#2-row-unit-and-assignment-units)
3. [Quotas per split](#3-quotas-per-split)
4. [Reserved-vocabulary assignment](#4-reserved-vocabulary-assignment)
5. [Probe reserve pool](#5-probe-reserve-pool)
6. [Detection arm — removable](#6-detection-arm--removable)
7. [Balance tolerance](#7-balance-tolerance)
8. [The partition](#8-the-partition)
9. [Findings for S0a and the planner](#9-findings-for-s0a-and-the-planner)
10. [Files](#10-files)

## 1. What the matrix is

`design-matrix.json` is the single abstract design matrix of R14-W1. It holds **14,059 abstract row slots** in **7,218 assignment units** under **2,038 signatures**. It holds no utterance text, no entity or state word and no gold string. Its columns are `row, stratum, construction, cmp, watch, action, aspect, role, negated, entity_type, placement, possessive, atoms, route, arm, reserved, member`, with `-` where an attribute does not apply. `build_design_matrix.py` generates the matrix deterministically (no randomness, no clock); `test_matrix.py` proves that a rebuild is byte-identical.

## 2. Row unit and assignment units

- **Row unit (R14-W1).** One utterance slot: stratum × construction × action × role × polarity × `cmp` × `watch` × placement × entity type × possessive. Role and polarity follow §1.2 from the construction and the action's aspect (`fire` is punctual; `move`, `scan`, `hold` and `wait` are durative).
- **Paraphrase family.** Four to seven slots that share one meaning: stratum, construction, `cmp`, `watch`, action, entity type and atom count. Members vary in placement and possessive. Sizes are 1,913 × 4, 82 × 5, 27 × 6 and 3 × 7. A family is one unit and never spans a split (§5.2).
- **Pairs.** Published minimal pairs and PC-2 arms are single units, so a pair never spans a split:

  | Pair type | Units |
  |---|---:|
  | PC-2 reuse/new arms | 453 |
  | D3–F1 | 39 |
  | D5–F2 | 39 |
  | D6–F1 | 37 |
  | D6–F2 | 2 |
  | F6 `dont_until`–F5 `unless` | 23 |
  | F6 `hold_fire_till`–F7 `until` | 23 |
  | S6–F6 `dont_before` | 5 |

- **Singles.** General, reserved-compositional, cap-boundary and ambiguous-executable slots, the arm's plain commands and every probe-reserve slot.
- **Signature.** The hash of a unit's kind, group and every abstract attribute of its rows. Units with one signature are identical abstract templates. The matrix gives each signature's unit count per split. It does not say which unit goes where; the partition decides that.

## 3. Quotas per split

Every count below is in rows. `coarse_quotas` in the JSON gives the full stratum × construction × `cmp` × `watch` table for every split.

| Split | Part | Rows | Composition |
|---|---|---:|---|
| G gate | eval | 560 + 460 arm | F1 40, F2 40, F3 50, F4 50, F5 90, F6 120, F7 90, F8 40, F11 40. All rows are single-atom; F3 and F4 are all `start`; F3 + F4 hold exactly 20 of each non-`none` `cmp`. 470 rows are start-side or `during`; F5 ∪ F6 = 210 and F6 ∪ F7 = 210. Watch: 77 `reuse`, 70 `new` (17 families each). |
| H held-out | eval | 3,880 | F 1,240 (F2 120, F3 120 per MX-1, others 100), with every one of the 37 named constructions at ≥ 20. B 5 × 100, D 11 × 40, S 7 × 20. General 10 × 100, with cancellation and zero-action as the no-action denominators. Reserved compositional 5 × 40. Cap 40 + 40, with cap-plus-one split 25 five-requested, 10 inserted-wait and 5 two-wait. PC-2 3 × 40 pairs. Ambiguous executable 40. |
| dev | eval | 404 | Two to three families per F cell (constructions round-robin), 12 rows per B cell, 4 per D family, 10 per general stratum, 4 per compositional stratum and ambiguous, 8 cap, 4 PC-2 pairs per condition. |
| RE | eval | 370 | Probe reserve (§5). |
| T | tv | 7,000 | Twice H for every covered stratum (F 2,480, B 1,000, D 880, eight general strata 1,600, compositional 400, cap 160, PC-2 480); every F construction ≥ 40. No no-action, ambiguous or S rows. |
| V | tv | 799 | 15% of T for single-atom strata and 9% otherwise, at least 4 per F construction; 385 single-atom rows. |
| RT | tv | 586 | Probe reserve (§5). |

Shared construction rules:
- **Placement (F rows).** Weights 23/23/23 for fronted, postposed and anaphoric, and 10/10/11 for the three mid-sentence sub-forms. Every H F cell clears 20/20/20 and 8/8/8.
- **Possessive.** 25% of rows in every F cell, so H F1–F5, F7 and F8 each have at least 20.
- **Watch.** 15% `reuse` and 15% `new` families per F and B cell outside the gate.

## 4. Reserved-vocabulary assignment

A row with `reserved == true` takes its entity and state words only from the sealed reserved list. The flag is set on whole units, because a family or pair shares its words. The list itself is not in the matrix; it is frozen later, eval-side, and never sent to the conductor or the adapter side.

| Split | Rule | Result |
|---|---|---|
| G | 40% of condition units | 227 of 560 atoms (≥ 210 required) |
| H | 35% of condition units | 1,047 of 2,973 atoms, 35.2% (≥ 30% required) |
| RE | 35% | 134 of 382 atoms |
| dev, T, V, RT | never | 0 |

## 5. Probe reserve pool

CA14-2's reserve: single slots across F, B, D, PC-2 and cap boundary, sized at 10% of the part's own rows per construction.
- **RE (370 rows)** is written by eval writers.
- **RT (586 rows)** is written by T/V writers.

Both are written before the masked-feature probe first runs and are never inspected. Rows are drawn only for the rule (b) re-test. If the pool holds fewer than 20 rows bearing a flagged value, blind writing tops it up (§8 CA14-2).

## 6. Detection arm — removable

The principal has not ruled on D-SLM8098. The arm's 460 gate rows carry `arm == true`:
- 230 plain commands (stratum `A0-plain`);
- 230 decoys: D1, D7, D8, D9 and D10 at 26 each; D3, D5, D6 and D11 at 25 each. D2 and D4 are excluded.

The arm's D3, D5 and D6 families are minimal pairs with gate F1/F2 families; those F rows are part of the 560 and stay whichever way the ruling goes.

- **Declined:** drop every `arm == true` row. No other row moves, and the partition summary already prints the assignment hash and per-split counts without the arm.
- **Adopted:** use the rows as dealt.

## 7. Balance tolerance

- **Where it applies.** F-cell rows (arm excluded), within aspect × measure strata (F3 measured, F4 count, other F cells plain), as the §5.2 item 1 legitimate determinants require.
- **What is crossed.**
  - Non-semantic attributes: action, placement, entity type and possessive. Entity type is exempt in F11/F12, where the form fixes it (MX-3).
  - Labels: role, negated, `cmp` and `watch`.
- **Statistic (frozen here, MX-5).** Bias-corrected Cramér's V (Bergsma 2013), with tolerance 0.10.
  - For family-level attributes (action, entity type), one observation per family; for row-level attributes (placement, possessive), one per row.
  - A table is computed only in strata of at least 100 rows.
  - Scopes that decide: the eval part, the T/V part, G, H and T.
  - Any table above 0.10 fails.
  - Each table also reports its resolving power and the uncorrected row-level V.
- **Matrix reading** (canonical deal, §8): corrected V is 0.000 on every table in every scope. Uncorrected row-level maxima: eval 0.187, T/V 0.144, G 0.059, H 0.146, T 0.180.
- **Resolved tables.** Eval 15 of 37, T/V 18 of 37, G 5 of 12, H 8 of 21, T 13 of 37. On the rest, the stratum is too small to resolve 0.10, so a pass means no association beyond chance was detected, not that V ≤ 0.10. These are mainly the F3/F4 durative strata, with about 25–55 families over 4 × 5 tables.
- **Vitality.** `test_balance_check_flags_a_planted_association` plants an entity × watch association in H and requires the check to flag it.

## 8. The partition

`partition_matrix.py` is run by the conductor. It works as follows:
1. Reads only the matrix.
2. Shuffles each signature's units with a SHA-256 counter stream seeded by the conductor's integer.
3. Deals them to splits in the signature's quota.
4. Checks the result:
   - every coarse quota per split;
   - the plan's floors, hard-coded from the plan text and independent of the builder;
   - that no unit spans a split;
   - that reserved rows appear only in G, H and RE;
   - the balance tolerance.
5. Reseeds on failure, with a recorded derived seed.

**What the seed changes.** Each split's abstract composition is fixed by the quotas, so it is the same for every seed. The seed only chooses which of several identical templates (unit IDs) goes to which split. The design aims every check at the matrix itself; the reseed path stays for a future matrix with free attributes. `test_matrix.py` confirms that the canonical (seedless) deal passes every check with and without the arm. The real partition was not run by this seat.

## 9. Findings for S0a and the planner

Each finding is my S0 freeze choice. S0a audits them; the planner is the owner if any needs a plan change.

- **MX-1 (plan arithmetic). F3 held-out is 120, not 100.**
  - §5.2 requires F3 + F4 to hold at least 40 atoms of each non-`none` `cmp`, and every named construction to have ≥ 20 rows.
  - `reaches` occurs only in F3's `reaches` construction, so it needs 40 rows there. The other four F3 constructions need 20 each, giving 120.
  - The F stratum is therefore 1,240 and T's F share is 2,480. The gate is unaffected (F3 50 with 20 `reaches`).
- **MX-2. F7's constructions are connective families.**
  - "hold until" is read as a durative action + `until` (§1.2 row 6), with the action free across `move`, `scan`, `hold` and `wait`. "keep … till" is the second construction.
  - The hold-fire minimal pair fixes `hold` on its F7 member, and F7's other families compensate so that `hold` keeps its share.
- **MX-3. Entity type is fixed by form in F11 and F12** (a running task, or the actor itself), so those cells are exempt from entity-type balance tables.
- **MX-4. A paraphrase family stays within one construction.** Connective wording varies within the construction's surface variants, so construction quotas are exact per split. §5.2's "varying … connective" is met within constructions, not across them.
- **MX-5. The balance statistic and its resolution, defined as in §7.**
  - The plan names a tolerance of 0.10 but not the estimator.
  - At these stratum sizes the uncorrected row-level V sits at 0.14–0.19 by granularity alone. Families of four identical rows over 4 × 5 tables cannot do better, even when perfectly balanced.
  - The S0a auditor rules on the estimator; the unresolved tables are listed in every partition summary.
- **Design counts (mine, not the plan's).**
  - Dev composition.
  - V factors 0.15 / 0.09.
  - Reserve factor 0.10.
  - Gate per-construction spread.
  - Gate watch families: 17 + 17.
  - Minimal-pair counts: every `dont_until`, `hold_fire_till` and S6 row is paired, and every D3, D5 and D6 family.
  - Arm decoy spread.
  - Family size 4–7.

## 10. Files

| Path (SuperSLM `tests/t3094-multicommand-s0/matrix/`) | Role |
|---|---|
| `design-matrix.json` | the matrix (also at `D:\_artifacts\SuperSLM\t3103\matrix\`) |
| `build_design_matrix.py` | deterministic builder |
| `partition_matrix.py` | the conductor's partition (also at `D:\_artifacts\SuperSLM\t3103\matrix\`) |
| `test_matrix.py` | builder determinism; no-text check; canonical-deal plan checks with and without the arm; partition mechanics on a synthetic matrix; planted-association vitality; CLI refusal and failure paths |
