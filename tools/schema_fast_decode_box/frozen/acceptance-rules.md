# Schema Fast Decode — Box Acceptance Rules (B1–B4), Frozen in Advance

**Seat:** the debunker. **Written:** 2026-10-20, before any B1–B4 reading exists or is read.
**Source:** `plans/superslm-schema-fast-decode/plan.md` rev 1.2.3, §0, §1, §4, §5, §8, §10, §11, §14 and §16, read
whole. The approval summary's Q1 wording (`approval-summary-rev1.2.md`) was also read. No build record, review record,
code-review record or box output was read. **Freeze:** the sha256 of this file is recorded before B1 runs. After that, any
edit to a rule is a new rule, dated, and it applies only to readings taken after the edit.

**What this document decides.** For each claim B1–B4 can test, it states the reading, the statistic, the noise floor
the reading must clear, and the result that counts as **un-refuted**, **refuted** or **inconclusive**. A verdict here
says how far a claim survived. It does not decide the release or the default; the plan and the principal do that.
Un-refuted is never written as proven.

## Index

1. The B4 quality margin (the default-on condition)
2. Method common to every timing rule
3. Commissioning: the controls that must fire before any timing verdict counts
4. B1 rules: batched-pass cost and finish cost
5. B2 rules: headless extraction, one thread
6. B3 rules: threaded decode at 4 tasks, and span size
7. B4 rules: quality
8. What each verdict means for the plan
9. Provenance of every operand
10. Appendix: the executed statistics

---

## 1. The B4 quality margin (the default-on condition)

The principal approved this condition: "on a labelled set of potion orders, skip-ahead gets the fields right at least as
often as today, within a margin the independent checker sets before looking at the results" (Q1 B).

**The margin is δ = 12.5 percentage points (5 of 40 prompts), for each field separately and for the whole order.** The
test is exact, one-sided and run at α = 0.05. All five endpoints must pass.

- **The quantity.** For each prompt and each field, the jump-forward-off arm and the jump-forward-on arm are each right
  or wrong against the frozen label. Let *b* be the prompts where off is right and on is wrong (regressions), and *c* the
  prompts where off is wrong and on is right (fixes). θ = (b − c)/40 is the drop in accuracy when jump-forward is on.
- **The test.** H0: θ ≥ δ. It uses the Tango score statistic with an exact unconditional p-value, taking the supremum
  over the nuisance parameter (§10). **Pass** iff p ≤ 0.05.
- **The endpoints:** `intent`, `item`, `quantity`, `polite`, and the whole order (all four fields right). The whole-order
  endpoint is there so that four single regressions spread across four fields cannot pass as four separate passes. The
  rule is pass-all, an intersection-union test, so the overall α stays 0.05 with no correction.

**What passes at N = 40, as executed (§10).** The table gives the fewest fixes *c* needed for each count of
regressions *b*:

| regressions *b* | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| fixes *c* needed | 0 | 0 | 2 | 3 | 5 | 7 | 8 | 10 | 11 |

- **In plain terms:** the on arm passes an endpoint if it loses at most one prompt net, or if every regression beyond the
  first is matched by a fix.
- **What a pass certifies:** at 95% confidence, the accuracy drop on the prompt population is under 12.5 points.

**Why 12.5 points, and not less.** This margin was not derived from any reading. It follows from N = 40, and N is fixed
by the plan.
- Even with zero discordant prompts, the smallest margin that 40 prompts can certify is **7.2 points** (executed, §10).
  Any margin under 7.2 points cannot pass, which would refuse the principal's condition by construction.
- At δ = 10 points, a tie of 2 regressions against 2 fixes fails. That is stricter than "at least as often as today".
- At δ = 12.5 points, every observed tie of up to 3 against 3 passes, and a net loss of one prompt passes. The
  population-level bound stays within one prompt in eight.
- **The limitation, stated before the result:** 40 prompts cannot certify a margin tighter than about 7 points. If the
  principal wants a tighter margin, the fix is a larger population: 120 prompts certify about 2.5 points at zero
  discordance. Tightening the threshold on these 40 prompts does not achieve it.

**Refuted and inconclusive.**
- **Refuted** (on is worse): an exact one-sided McNemar test of off over on, p ≤ 0.05, on any endpoint. At N = 40 this
  means b ≥ 5 with c = 0, b ≥ 7 with c ≤ 1, b ≥ 9 with c ≤ 2, and so on (§10).
- **Inconclusive:** neither a pass nor a refutation; for example b = 3 and c = 0. The condition is **not met**, and the
  default stays off.
- **Re-running:** an inconclusive result may be re-run only on a **new, separately frozen population of fresh prompts**,
  under these same rules. The first 40 prompts never enter that decision. Pooling the two sets, or extending the first
  set after it has been read, adds a second look at the data and is not allowed.

The clean-accept rule on `Prompt_Result` (§7.4) has the same form, at its own N.

---

## 2. Method common to every timing rule

- **Pairs.** n = 15 interleaved pairs, plus one warm-up pair that is discarded and never read. Model load is excluded.
  Odd pairs run arm A first and even pairs run arm B first (ABBA), so drift and cache warmth cancel. No outlier is
  removed for any reason. A crashed run fails the step; it is not dropped.
- **Statistic.** Per pair, R_i = t(baseline) / t(candidate), so R > 1 means the candidate is faster. The plan requires
  the median and the IQR, and both are reported (numpy default quantiles).
- **The decision interval.** Decisions use the sign-test interval for the median, **[R(4), R(12)]**, the 4th and 12th
  order statistics of the 15 ratios. Its coverage is **96.48%** (executed, §10). The IQR is reported and is never the
  decision interval.
- **Classification.** If the interval lies wholly inside the un-refuted region, the claim is **un-refuted**. If it lies
  wholly inside the refuted region, the claim is **refuted**. If it straddles a boundary, the result is
  **inconclusive**. Inconclusive is reported as "no result at this resolution", and it never counts as "no difference".
- **Minimum effect size.** A speed-up below **1.03×** is never acted on, however tight its interval. For a difference
  in ms (B3 spans), the minimum is **2 ms per span**. For a finish (B1), it is **0.05 ms**.
- **Required resolving power.** A rule can decide only when the interval's half-width is smaller than the gap between
  the predicted value and the nearest refutation edge. That gap is stated for each rule. If the half-width is wider,
  the verdict is **inconclusive before classification**, whatever the median.
- **The timed region** is generation only: from the first generation call after the prompt prefill (on the on arm, that
  is the start span's verb call) to the −2 dead end. Prompt prefill is timed and reported separately. §8's arithmetic
  covers generation only, so an end-to-end timing is a different quantity. If B2 or B3 time only end to end, the
  magnitude rules are inconclusive.
- **Same binary.** The jump-forward off and on arms run from **one library binary** (equal sha256), with the setting as
  a runtime switch. `main` and M1 are different binaries by nature; both hashes and both commits are printed beside
  every reading (§10 of the plan).
- **Cell coordinates reported with every reading:** thread count and hook bits, prompt depth, the schema, the model
  sha256, and the box's power plan. A verdict holds for the cell it was measured in.

---

## 3. Commissioning: the controls that must fire before any timing verdict counts

The timing comparison is an instrument, and its verdicts are **quarantined** until both of its controls score
correctly. I authored these controls without seeing any builder control. **They are additions to B2 and B3 that the
driver and the bench must carry. If a control is not run, every speed verdict in that step is "uncommissioned":
recorded, but never quoted or acted on.**

| Control | Step | Construction | Must score |
|---|---|---|---|
| **AA** (must-accept) | B2, and B3 at 4 tasks | jump-forward off against jump-forward off, labelled as two arms, interleaved like the real pairs | the interval contains 1.000 and its half-width is ≤ 0.03 |
| **2X** (must-fire) | B2 | arm A runs the off-arm extraction once; arm B runs it twice back to back, per sample. This is a known 2.00× slowdown on the real data path | the median is in [1.94, 2.06] and the interval contains 2.00 |
| **CB1** (cause removed) | B2 | jump-forward on at `chunk_budget` = 1 against jump-forward off. This keeps the token-count saving (29 → 22 tokens) and removes the batching | see §5.3 |

- **AA fails:** the box is noisier than the 3% minimum effect allows. The step is re-run once, since steps are
  resumable. If AA fails again, every timing verdict of that step is inconclusive.
- **2X fails:** the instrument does not measure time proportionally. The standing is **dead**, and the step's speed
  verdicts are withdrawn.
- **Order:** AA and 2X are read before any real pair. CB1 is read with the real pairs, because it is a claim test as
  well as a control.

---

## 4. B1 rules: batched-pass cost and finish cost

**The readings.**
- T(M) is the whole batched pass, all sites, on the real 0.5B artifact, for M = 1..8, at the prompt depth the
  extraction actually reaches. 15 rounds, with the order of M rotated each round.
- f_g(n) and f_d(n) are the whole finish (rows, re-mask and argmax) for greedy and for damped greedy, at n = 0, 1, 22,
  1,000 and V legal rows, with the same 15 rounds.
- r(M) = T(M)/T(1), computed **per round**, so that the per-round composites below have their own interval.

| # | Claim (plan section) | Rule | Un-refuted | Refuted | Required resolving power |
|---|---|---|---|---|---|
| 1.1 | K3: the real all-sites span cost is within 10% of the GEMM-only figure. Σ r over spans 3, 6, 6, 5, 2 = 9.47, so the edge is ≤ 10.42 (§8, §14) | S = r(3)+2·r(6)+r(5)+r(2) per round; interval [S(4), S(12)] | upper ≤ 10.42 | lower > 10.42 | half-width < 0.47 |
| 1.2 | §8's bounds: 1 ≤ r(M) ≤ M | for each M, the interval of r(M) | inside [1, M] | a resolved excursion at any M | — |
| 1.3 | §8: the greedy finish at the example's legal-row counts is about 0.25 ms; tolerance 2× for cell Z, where 0.50 ms is 1.4% of a layer pass | interval of f_g(22) | upper ≤ 0.50 ms | lower > 0.50 ms | half-width < 0.25 ms |
| 1.4 | §6: the dead-end finish under jump-forward costs about 0.25 ms with no layer pass | interval of f_g(0) | upper ≤ 0.50 ms | lower > 0.50 ms | as 1.3 |
| 1.5 | §8's input F ≈ 4.5 ms (the full finish, at 4 tasks, cell Z) | interval of f_g(V) at 4 tasks | inside [2.25, 6.75] ms (±50%) | wholly outside | — |
| 1.6 | **The mechanism, a must-fire check:** the finish cost tracks the legal-row count | f_g(V) − f_g(1000) and f_g(1000) − f_g(22), paired per round | both lower bounds > 0.05 ms | either one unresolved: **the bench is not running the legal-row path**, and rules 1.3–1.4 are withdrawn | — |
| 1.7 | §4.1: damped greedy keeps its O(V) work and claims no gain | f_d(n) − f_d(V) for each n | upper ≤ +0.05 ms (no regression) | lower > +0.05 ms at any n | — |

- **Rule 1.5 refuted** refutes §8's input on cell Z, not M1. §0 and §8 are then recomputed from the measured F before
  any release text quotes the 1.12×.
- **Composition check (B1 against B2).** P_on = T(3)+2·T(6)+T(5)+T(2) + f_g(0) + f_g at the four choice states'
  legal-row counts (16, 22, 15, 8; interpolated from the n grid, and marked so). If B2's on-arm median lies outside
  P_on ± 15%, the bench does not measure what the extraction does. Every B1-derived claim is then flagged an
  **artifact**, and is inconclusive until that is explained.
- **Reported, not gated:** T(7) against T(8) (§16 Q7; the tiled plan's lever) and the shape of r(M). B1 also feeds §16
  Q6 through rule 6.5.

---

## 5. B2 rules: headless extraction, one thread

### 5.1 Hard rules (any single failure refutes the claim; no statistics)

| # | Claim | Rule | Refuted by |
|---|---|---|---|
| 2.1 | C1: masked finish is exact | the token digest of every `main` run and every M1 run is identical (30 runs) | any one mismatch |
| 2.2 | §5.7: jump-forward is deterministic | the digest is equal across all 15 on runs, and across all 15 off runs | any one mismatch |
| 2.3 | C2 and cell 10.3: the output is schema-valid | every on run ends at the −2 dead end with the walk accepting and valid JSON under `potion_shop_order` | any one failure |
| 2.4 | §0: same choices, same text | if the on arm's four field values equal the off arm's, the JSON bytes are equal, and the on run emits 22 tokens if its choices are `a`/`health`/`one`/`true` | different bytes under the same choices |
| 2.5 | C2 and cell 6.2: chunk invariance on the real artifact | the CB1 arm's digest equals the default-budget on arm's | any one mismatch |

If the on arm's choices differ from the off arm's, that is a K1 event. The tokens, the digest and the field values are
recorded, and §0's 22-token arithmetic does not apply to that run. This is not a refutation of 2.4.

### 5.2 Speed rules

| # | Claim (plan section) | Predicted | Un-refuted region | Refuted | Required resolving power |
|---|---|---|---|---|---|
| 2.6 | M1 is faster (§0, §8: 1.12×) | 1.121 | lower bound > 1.03 **and** the interval within [1.06, 1.24], half to double the effect | upper < 1.06 (gain too small), or lower > 1.24 (the model is wrong; check for an artifact first) | half-width < 0.06 |
| 2.7 | Jump-forward is faster, against M1 (§8: 1,052 → 342–380 ms) | 2.77–3.07 | the interval overlaps [2.52, 3.38], the predicted range ±10% (K3's tolerance) | wholly below 2.52, or wholly above 3.38 | half-width < 0.25 |
| 2.8 | §8's bounds, 180–800 ms | — | inside [1.31, 5.78] | wholly outside | — |
| 2.9 | §0's headline against today, 3.1–3.5× | 3.10–3.45 | the conservative product interval overlaps [2.82, 3.79] | wholly outside | — |
| 2.10 | Cell check: today on cell Z ≈ 1,179 ms (§8) | 1,179 | the `main` arm's median within [943, 1,415] ms (±20%) | outside: the box is not the cell §8 priced. Absolute claims are inconclusive; the ratios 2.6–2.9 are still graded | — |

- **Rule 2.9's interval** is the product of the 2.6 and 2.7 endpoints. Their pairings are independent, so its coverage
  is at least 0.9648² = 0.931. It is tagged as derived.
- A refutation **above** the predicted range (2.6 or 2.7) is checked for an artifact before it is believed. The check
  confirms that the on arm emitted a complete, valid stream (rules 2.3 and 2.4) and that the off arm is not degraded
  (rule 2.10).

### 5.3 The cause-removal test (CB1)

§8 attributes the gain to two things: fewer tokens (29 → 22) and batched spans. At `chunk_budget` = 1, the run keeps the
first and loses the second.
- **Predicted:** R_CB1 = (29·36 + 30·0.25)/(22·36 + 5·0.25) = **1.326** [A].
- **Un-refuted:** the interval overlaps [1.21, 1.46], the prediction ±10%.
- **Refuted:** the interval is wholly above 1.46. Then the run without batching is faster than the token-count saving
  explains, so the headline is not caused by what §8 says. The interval is wholly below 1.21: the token-count saving is
  not real on cell Z.
- **Either refutation** makes 2.7 inconclusive as attributed. The speed may be real, but its stated cause is refuted.

---

## 6. B3 rules: threaded decode at 4 tasks, and span size

Hard rules 2.1–2.4 are re-applied at 4 tasks, with the same refutations. The AA control is run at 4 tasks (§3).

| # | Claim (plan section) | Predicted | Un-refuted region | Refuted | Required resolving power |
|---|---|---|---|---|---|
| 3.1 | M1, threaded (§8: 738 → 611 ms) | 1.209 | lower > 1.03 and the interval within [1.10, 1.42] | upper < 1.10, or lower > 1.42 | half-width < 0.10 |
| 3.2 | Jump-forward against M1, threaded (611 → 342–380 ms) | 1.61–1.79 | overlaps [1.46, 1.96] | wholly outside | half-width < 0.15 |
| 3.3 | Bounds, threaded | — | inside [0.76, 3.36] | wholly outside | — |
| 3.4 | §0's threaded headline against today, ≈ 2× | 1.94–2.16 | the conservative product interval overlaps [1.77, 2.37] | wholly outside | — |
| 3.5 | Cell check: today threaded ≈ 738 ms | 738 | the `main` median within [591, 886] ms | outside: absolute claims are inconclusive | — |
| 3.6 | G10 and §8: spans with M ≥ 2 run single-thread, so "both, threaded" equals "both, one thread" | — | the B3 on-arm median within ±10% of the B2 on-arm median | outside ±10%: G10's consequence is refuted on cell Z | — |

**Span size against threaded one-row passes (§16 Q6, K4).**
- **The reading.** For each M = 1..8 there are 15 interleaved pairs, and D(M) = T_span(M) − M·T_1row,threaded,
  in ms.
- **The comparator** is the one-row pass that Q6 C would actually run: a one-token threaded admission. B3's text says
  "decodes". A one-row decode adds an M1 finish (about 0.25 ms) per row. That biases D against the span by M × f_g, up
  to about 2 ms at M = 8, which equals the minimum effect. If decodes are used, the bias is reported beside D, and any
  verdict within 2 ms of a boundary is inconclusive.
- **The statistic** is the interval of the 15 paired differences, [D(4), D(12)].

| # | Claim | Un-refuted | Refuted |
|---|---|---|---|
| 3.7 | **Must-accept control: M = 1.** The span and the one-row pass are the same work | the interval contains 0, or \|median\| < 2 ms | a resolved difference ≥ 2 ms: the two sides are not the same quantity, and 3.8–3.10 are inconclusive |
| 3.8 | K4: the span loses at M = 2, by ≈ 8 ms on cell Z | lower bound > 0 and median ≥ 2 ms; magnitude: the interval overlaps [4, 16] ms | upper < 0 (the span wins); or wholly outside [4, 16] (the magnitude is refuted) |
| 3.9 | §8: the span wins at M = 3 (55 against 62 ms) | upper < 0 | lower > 0 and median ≥ 2 ms |
| 3.10 | Q6 C's threshold k | — | — |

**Q6 C's k (rule 3.10) is set by the rule, not chosen.** k is the largest M such that a loss is resolved (lower bound
> 0, median ≥ 2 ms) at **every** M′ from 2 to M. If no loss is resolved at M = 2, then k = 0, which is the plan's
default.

---

## 7. B4 rules: quality

### 7.1 Prerequisites, checked on the frozen population before any run

The population is input, not a reading, so checking it does not break the blindness.
- **Size and freeze.** N = 40 prompts, labelled `intent`/`item`/`quantity`/`polite` by a seat other than the builder.
  The population file and the scorer are hashed (sha256) and recorded before the first B4 run.
- **No labels added later.** A prompt the author judged ambiguous is removed **before** the freeze, never after. A label
  error found after the read is reported, but the frozen labels decide.
- **Discrimination.** For each field, the majority label covers ≤ 60% of the 40 prompts (`polite` needs ≥ 16 of the
  minority value). Each enum field shows ≥ 3 distinct labels. A field that fails this cannot show a regression; its
  endpoint is **inconclusive by construction**, and the whole-order endpoint carries on.

### 7.2 Scoring

- **Configuration.** Greedy (the plugin decodes greedy only, §5.8). One run per prompt per arm, both arms from the same
  library. B2's rule 2.2 is the determinism evidence: if 2.2 failed, B4 is void.
- **The scorer.** A field is right iff the output's field value equals the label byte for byte. A run that is not valid
  JSON, or not schema-valid, scores all four fields wrong. **Any such run on the on arm also refutes C2** (as rule 2.3).
- **Same choices, same text,** applied to all 40 prompts. Wherever the on arm's four values equal the off arm's, the
  bytes must be equal. One violation refutes §0's claim.
- **Secondary readout, not gated:** off-vs-on agreement (the K1 rate).

### 7.3 The decision: per-field accuracy against the labels

This is §1's rule: δ = 12.5 points, an exact Tango-score non-inferiority test, α = 0.05, five endpoints, all of which
must pass.

**Commissioning of the B4 instrument.** These controls are run on the off arm's outputs alone, **before the on arm's
outputs are unsealed**:

| Control | Construction | Must score |
|---|---|---|
| **B4-A** (must-accept) | "on" := off: b = c = 0 | pass on all five endpoints. This is already executed as arithmetic (§10, b = 0 passes) and is re-run through the real scorer |
| **B4-R1** (must-reject) | "on" := the off arm's own values for each field, deranged across prompts (a cyclic shift by 1). The result is a legal enum value the real model emitted, wrong for this prompt: the production-feasible shape of a K1 regression | fail (no pass) on every field where the construction is large, b − c ≥ 8 (≥ 20 points). A field where the construction is not that large is uncommissioned, and its endpoint is inconclusive |
| **B4-R2** (must-reject) | "on" := a constant predictor, the majority label for each field | fail on every field where b − c ≥ 8 |

If B4-R1 or B4-R2 passes the non-inferiority rule on a field where it was large, the instrument is **dead**, and no B4
verdict is acted on.

### 7.4 Clean accept on the `Prompt_Result` population (te368)

- **The quantity.** Paired, per prompt: clean accept, off against on, graded by the population's existing grader. The
  grader is hashed and frozen before the run, and it is unmodified.
- **The rule.** The same test as §1, at δ = 12.5 points and α = 0.05, and at the population's own N. The acceptance
  table for that N is computed with §10's script **before the readings are read**, and recorded beside them.
- **The resolving-power floor.** At zero discordance, the smallest certifiable margin is 13.9 points at N = 20, 9.5
  points at N = 30, and 7.2 points at N = 40. **At N ≤ 21, this rule cannot pass at δ = 12.5, and it is declared
  inconclusive by construction now.**
- **Refuted:** the exact one-sided McNemar test, p ≤ 0.05, off over on.
- **The standing of this readout.** Q1 B's condition names only the labelled potion accuracy, and this document does
  not add a fifth condition. A refuted clean accept is a measurement that contradicts the premise that jump-forward does
  not change model choices, on a second schema. It is surfaced to the principal as evidence that reopens Q1
  (StandardsDocument §5.6). It does not silently block anything.

---

## 8. What each verdict means for the plan

These are the plan's own consequences, stated so the reading is not reinterpreted afterwards:

- **A refutation of any hard rule** (2.1–2.5, the B3 repeats, the C2 scoring in §7.2) falsifies a §1 constraint on
  the real artifact. B-eng is not accepted (§9 of the plan: R depends on "B-eng accepted").
- **A refuted speed magnitude** (2.6–2.9, 3.1–3.4, 1.1, 1.5) corrects §0 and §8. The correction lands before any release
  text quotes the figure.
- **Existence refutations** are surfaced to the principal, because they refute the plan's purpose. They are: 2.6 with
  an upper bound < 1.0, and 2.7 or 3.2 with an upper bound < 1.03.
- **An inconclusive result** is reported as "no result at this resolution", with the interval and the required
  resolving power side by side.
- **B4 pass** satisfies one of Q1 B's four default-on conditions. **B4 inconclusive or refuted** means that condition
  is not met, and the default stays off.
- **K4 and Q6 C:** k is set by rule 3.10. If k = 0, Q6 A stands (nothing is built).
- **Out of scope here:** B5 (the in-game CPU wall, and its "beyond the measurement noise" condition) and B6 need their
  own rules, frozen before B5 runs.

---

## 9. Provenance of every operand

No bound in this document is produced by the artifact it grades.

| Operand | Source | Produced by the artifact under test? |
|---|---|---|
| Predicted ratios, bands and bounds (§4–§6) | the plan's §0 and §8 arithmetic (L = 36, F = 4.5, f = 0.25, 20.8 ms threaded roof, Σ r = 9.47) and K3's 10% | no; set from on-file figures and the plan's arithmetic |
| Minimum effect sizes (3%, 2 ms, 0.05 ms) and the 2× finish tolerance | stated here, from what changes a decision | no |
| The interval [x(4), x(12)] and its 96.48% coverage | the binomial(15, ½) distribution | no |
| δ = 12.5 points and α = 0.05 | the principal's wording and N = 40 (§1) | no |
| The labels | a seat other than the builder, frozen before any run | no |
| The scorer and the te368 grader | frozen and hashed before any run | no |
| Commissioning constructions (AA, 2X, CB1, B4-A, B4-R1, B4-R2) | authored by this seat, blind to builder controls | built **from** the off arm's readings (B4-R1, B4-R2), but each must fail or pass a bound fixed here |

---

## 10. Appendix: the executed statistics

Executed 2026-10-20 in Python 3, standard library only. The script is reproduced so the tables can be recomputed at
any N.

**Executed results:**
- **The n = 15 sign-test intervals:** [x(3), x(13)] has 0.9926 coverage; **[x(4), x(12)] has 0.9648**; [x(5), x(11)]
  has 0.8815.
- **The smallest margin certifiable at b = c = 0** (α = 0.05, one-sided): N = 20, 0.1391; N = 30, 0.0950;
  **N = 40, 0.0722**; N = 60, 0.0487; N = 80, 0.0368; N = 120, 0.0247.
- **N = 40, δ = 0.125, the fewest fixes *c* that pass for each *b*** (b = 0 to 12): 0, 0, 2, 3, 5, 7, 8, 10, 11, 13,
  14, 15, 17.
- **N = 40, the comparison margins:** at δ = 0.10 the fewest fixes are 0, 1, 3, 4, 6 (b = 0 to 4). At δ = 0.05 they are
  1, 3, 5, 7, and b = 4 needs more than 7.
- **Inferiority** (exact McNemar, one-sided, p ≤ 0.05): resolved at b = 5 or 6 with c = 0; b = 7 or 8 with c ≤ 1; b = 9
  with c ≤ 2; b = 10 or 11 with c ≤ 3; b = 12 with c ≤ 4.
- **Derived bands:** today 1,179.0 ms; M1 1,051.5 ms; R_M1 1.121; threaded today 738.2 ms; threaded M1 610.7 ms; R_M1,thr
  1.209; R_CB1 1.326; the K3 edge 10.417.

```python
import math
from math import comb, sqrt

def tango_z(b, c, N, d):  # H0: theta=(p_b-p_c) >= d; restricted MLE of p_c under p_b = p_c + d
    A = 2*N; B = -b - c + (2*N - b + c)*d; C = -c*d*(1 - d)
    pc = (-B + sqrt(max(B*B - 4*A*C, 0))) / (2*A)
    var = N*(2*pc + d*(1 - d))
    return (b - c - N*d) / sqrt(var if var > 0 else 1e-12)

def ni_pvalue(b0, c0, N, d, grid=200):  # exact unconditional: sup over nuisance p_c
    z0 = tango_z(b0, c0, N, d)
    region = [(b, c) for b in range(N+1) for c in range(N+1-b) if tango_z(b, c, N, d) <= z0 + 1e-12]
    sup = 0.0
    for i in range(grid+1):
        pc = (1 - d)/2*i/grid; pb = pc + d
        if pb + pc > 1: continue
        pr = 1 - pb - pc
        p = sum(math.factorial(N)/(math.factorial(b)*math.factorial(c)*math.factorial(N-b-c))
                * pb**b * pc**c * pr**(N-b-c) for b, c in region)
        sup = max(sup, p)
    return sup   # pass (non-inferior at margin d) iff sup <= alpha

def mcnemar_inferior(b, c, alpha=0.05):  # off better than on, exact one-sided
    D = b + c
    return D > 0 and sum(comb(D, i) for i in range(b, D+1)) / 2**D <= alpha
```
