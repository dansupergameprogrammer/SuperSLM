# A0: pre-licence coverage re-audit of the re-bound E4 population (2026-10-23)

2026-10-23 · coverage auditor · plan of record **rev 1.18.13 / 13a**, §15.33 (read whole), with T-3071
(`Claude/Popper/t3071-precache-refs-commission-2026-10-01.md`, read whole; identical in all 11 Wizard worktrees) and
D-SLM8030 (`DecisionLog.md`). Records only, with no box time. The run sheet's A1 remains E4's green gate.

The audit read:
- SuperSLM `47bdcc9e` (the generator) and `889e6f9a` (the branch tip; documents only above `47bdcc9e`), in a full clone
  (`--is-shallow-repository` = `false`);
- the v2 sidecars at `origin/curie/t3035-precache-refs` `972d5b7b`, `tests/t3035-precache-refs/discrimination-v2/`;
- `e4-cloud-2026-10-22.md` §6–§7 and `e4-box-runsheet-2026-10-22.md` (B3–B6, A0, L1, A1).

Mutants were applied in a detached worktree and reverted, and nothing was committed. Receipts are in
`a0-preliicence-reaudit-2026-10-23-receipts/`.

**Engine identity.** `git diff 1948e1f2 47bdcc9e -- src include CMakeLists.txt` and `git diff e0fe1aea 47bdcc9e -- …`
are both empty. So the E2 re-audit at `e0fe1aea` (`coverage-reaudit-post-m1-2026-10-22.md`) grades the same engine
this population anchors to. Between `e0fe1aea` and `47bdcc9e`, the only change to an E2 cell is
`e2_failures.cpp` (G1's 2-page confirmation).

## 1. The re-bound population and its bindings

| Check | Finding |
|---|---|
| Generator (`t3035_reference.cpp` at `47bdcc9e`) | `RecordPath` requires `{id}` exactly once, substitutes the stem, refuses a missing file, and the header writes `FileSha(<that file>)` as `discrimination_record_sha256`. There is no default and no directory form. Nothing else in the header or rows changed (diff read whole) |
| What it binds (rev 1.18.13a) | The run sheet's B3/B4/B5 set `$E4Record = '{id}.stream.json'`. L1's `record_hashes` is exactly the 20 stream hashes; the 20 `byte.json` are provenance in the manifest. This matches 13a |
| Cloud receipt (stand-in) | 20 pairs, run1 = run2; 20/20 headers bind their stem's stream sidecar. I recomputed two of the cited hashes from the sidecar bytes: flat00 `e120b5b8…` and nested00 `00d3a92d…` equal the record's figures |
| The 40 sidecars (all read) | Every one names `model_sha256` `4b27e2d6…` (T-3071's real 0.5B), `reducer_source_sha256` `457f0e9b…` (T-3071's commissioned reducer) and `page_positions` 16. The `instrument` fields are `T-3035-precache-stream-v2` and `T-3035-precache-byte-v1`, both COMMISSIONED per D-SLM8030. `length` is each stem's prompt length (flat 159–2,000; nested 1,195–1,223 = world 1,012 + persona). flat00's `reference_sha256` `91dfbe91…` and `target_ids_sha256` `ef7fae58…` equal T-3071 §2's pinned values |
| Standing | Each sidecar's JSON says `QUARANTINED` and "Observation only". The run sheet's L1 note reads standing from the registry, as §15.33 (a) and T-3071 §1 require. Correct |

**What the stream records say about the population (all 20 read).** Of 24 changed-state mutants per stem, the 64-token
streams kill between 7 and 23. For example, flat05 has 17 survivors and nested00 has 15. The two population summaries
(P1, P2) show:
- **every page-pair swap (`M-swap`) survives the streams: 0 killed of 10 in P1 and 0 of 10 in P2;**
- `row:noninitial` is killed 121/170 (P1) and 138/170 (P2).

The byte-v1 sidecars refuse 24/24 (flat09: 23/23) by exact state. The summaries themselves say "Positive stream
witnesses only; exact-state cells and independent commissioning remain separate". So the token references, by
themselves, cannot see a large class of restore defects. The licence is sound only if exact-state cells grade those
classes on the same engine and geometry. §2 and §3 check that.

## 2. The cells that grade it: 10.1 and 10.2

I read `Real101` and `Real102` at `47bdcc9e` (`e4_real.cpp:156-225`, `:531-638`, unchanged from `e8c029ed`).

The restore process asserts:
- the rule B against the engine and the helpers;
- 50 adopters per cell (both adopter modes, greedy and damped), each equal to the pinned reference stream, with
  `graded == 50`;
- `sslm_pc_test_only_prefix_prefill_calls() == 0` (10.1);
- `*out_shared_pages == floor(|world|/B)` and the fill-probe draw (10.2).

**Finding A (the plan's premise is false; no blocker).** §15.33 (5) and 13a speak of "10.1/10.2's exact-state
assertions". **10.1 and 10.2 have none.** Neither re-saves the restored prefix against the captured blob, and neither
reads an adopter's `SSB*` state. They grade token streams, plus share and draw for 10.2. On their own they would miss
every stream survivor in §1, page swaps included.

The exact-state obligation for this engine and geometry is carried by other cells, which I ran under state mutants
(§3). I accept that as discharging it for the licence. The planner should fold the correction (§5).

Adding a restore re-save identity to 10.1/10.2 would be cheap in code. But `e4_real.cpp` is the timing instrument's
hashed source, and X1/B10 name `e8c029ed`, so it would cost T1, B8 and X1/B10 again. I do **not** require it.

## 3. T-3071's named rows (6.1, 6.4, 6.7, 8.4, 8.5) and exact state at B = 128

**Already cleared at this engine by the E2 re-audit (`coverage-reaudit-post-m1-2026-10-22.md` §1):**
- 6.4: killed by `first-compatible-schema`; its isolated `current_token` mutant was run on GNU in the cloud and on
  MSVC on the box (`cc28c734`).
- 6.7 and 8.4: killed by `no-schema-walk` and `first-compatible-schema`.

**Not cleared there: 6.1 and 8.5,** which are not in `run_e2_mutants.sh`'s 28 cells. Run now. GNU Release,
`e0fe1aea` worktree, which has the same engine as `47bdcc9e` and unchanged `e2_integrity.cpp` and
`e2_composition.cpp`. Auditor mutants are in `a0_mutant.py`. Baseline, all green: 6.1 35,929 checks, 6.4 110,
6.7 203, 8.4 203, 8.5 28.

| Mutant (restore side) | 6.1 | 6.4 | 6.7 | 8.4 | 8.5 |
|---|---|---|---|---|---|
| `restore-drop-rope-k-count` | **RED** (SSB6 offset 148; re-save; "count 4 lost") | RED | RED | RED | green (tokens do not depend on counts) |
| `restore-swap-q-k-counts` | **RED** (offsets 140, 148; counts 3 and 4) | RED | RED | RED | green |
| `restore-drop-scale-e` | **RED** (offset 84; `-3` read back as 0) | RED | RED | RED | green (S20: `e` does not reach tokens) |
| `restore-drop-residual` | **RED** | RED | RED | RED | **RED** ("restored prefix under adapter differs") |
| `restore-walk-zero` | **RED** | RED | RED (30) | RED (30) | green (unbound) |
| `scatter-unrotated` (K/V scatter) | **RED** ("392 / 6,976 / 8 K/V rows differ") | RED | RED | RED | **RED** |
| `first-compatible-schema` | green (its prefixes are unbound) | RED | RED | RED | green |
| `SUPERSLM_PRECACHE_CURRENT_TOKEN_MUTANT=1` | green (SSB6 sentinel, per 6.1's text) | **RED**, 10 at `e2_root.cpp:109`/`:111` | green | green | green |

Every row is killed by the cell whose subject it is. 8.5 is killed by both state defects that can reach tokens
(residual, K/V scatter). It is correctly green on fields that cannot reach tokens (S20), and 6.1/6.4 grade those. These
cells run at the pinned B = 16.

**At the population's B = 128** (pkv_def's rule B equals the real 0.5B's), the rule-B cells were run under the same
defect classes. Baseline: 4.7/E2:pkv_def green (2,182), 9.8/E2:pkv_def green (1,987).

| Mutant | 4.7/E2:pkv_def | 9.8/E2:pkv_def |
|---|---|---|
| `scatter-unrotated` | RED (13) | RED ("re-saves different bytes" at L = 129, 4,095, 4,096) |
| `restore-drop-residual` | RED (15) | RED (re-save, L = 1, 128, 129) |
| `restore-swap-q-k-counts` | RED (7) | RED (re-save at L = 128, 129, 4,095, 4,096) |

`restore-b-from-model` and A22 were cleared on this engine in the E2 re-audit, §2. On the real artifact itself,
`6.6/E4:rule-b` asserts restore and re-save identity (`e4_real.cpp:746`); its box receipt is B2's log, which belongs to
A1. **So the stream survivors of §1 are killed at B = 128 by exact-state cells on this engine:** 4.7/9.8 E2 at rule B,
6.1/6.4 at B = 16, and 6.6 rule-B on the real artifact.

## 4. The planner's "discrimination measured at B = 16" (§15.33 (4))

**Accepted: no re-measure for this licence.** Two of the planner's reasons hold as stated:
- The references are prefill-built token streams that no restore can influence, and they are token-identical at
  B = 128. The debunker's recompute and the cloud's "every row equals" receipt show this. That is the B-invariance a
  licence needs.
- The byte-v1 refusals compare restored state exactly, so page size cannot weaken them.

Two parts of the reasoning are not right and should be corrected in the fold:
- **"Page constructions only get coarser at B = 128 … easier to detect" holds for exact-state detection only, not for
  streams.** At B = 16 the streams already miss every page swap (§1). A coarser page does not make a swap visible to a
  token stream. The B = 16 stream kill rates are therefore no estimate of B = 128 stream vitality, and the licence must
  not be read as claiming one. It does not need to: §3 carries the kill.
- **"B8's mutants" is a mis-cite.** B8a–B8d are 10.3's timing slices and run no restore-defect mutant. The B = 128
  kills are:
  - `apply_e4_mutant.py` on 10.2: `hint-s-at-2B` and `hint-share-ceil`, cloud stand-in at B = 128;
  - the E2 rule-B cells (§3);
  - 6.6/E4:rule-b's re-save identity on the box (B2).

  A1 should cite those.

Provenance limit, not a blocker: every per-stem sidecar measured one adopter configuration (`SSB5`, legacy, greedy).
T-3071's commissioning constructions covered SSB6 budget too, on flat00. The manifest's B = 16 note should also say
"legacy greedy adopter".

## 5. Owed, by owner (none blocks L1)

- **Planner:** fold Finding A. §15.33 (5) and 13a should name the exact-state cells that actually grade the stream
  survivors: 6.1, 6.4, 6.7, 8.4 (B = 16), 4.7/9.8 E2 at rule B, and 6.6/E4:rule-b. 10.1/10.2 grade streams, share and
  draw only. Fold §4's two corrections as well: exact-state only, and not "B8's mutants".
- **Debunker at L1 (suggested checks, one line each):**
  - each bound stream sidecar's `model_sha256` equals the box artifact's, and its `length` equals the stem's
    prompt-ids length;
  - its `reference_sha256` names the `5d5ca58` reference whose rows equal the new reference's rows. That is the
    token-identity link the binding relies on, because each sidecar names the old population's file.
- **Box (precondition of this verdict):** B5's log reads 20 `pair-equal`, `pairs differing: 0`, 20 `binds-v2`, and
  `headers not bound to their v2 sidecar: 0`. If it reads otherwise, this audit does not stand for that population.
- **A1:** include B2's `6.6/E4:rule-b` re-save receipt and the E4 mutant receipts (§4).

## Commands

```
git -C /home/claude/SuperSLM fetch origin              # separate clone; tip 889e6f9a, generator 47bdcc9e
git diff 1948e1f2 47bdcc9e -- src include CMakeLists.txt ; git diff e0fe1aea 47bdcc9e -- src include CMakeLists.txt   # both empty
git show 47bdcc9e -- tests/precache/reference/t3035_reference.cpp
git show 47bdcc9e:tests/precache/e4_real.cpp           # Real101, Real102, Parallel66RuleB
git show origin/curie/t3035-precache-refs:tests/t3035-precache-refs/discrimination-v2/<grp>/<stem>.{stream,byte}.json   # all 40 + 2 summaries
export SUPERSLM_PAGED_KV_FIXTURE_DIR=<cov>/pkv SUPERSLM_PRECACHE_SYNTHETIC_DIR=<scratch>/synth
superslm_precache_e1 =6.1/E2 ; =6.4/E2 ; =6.7/E2 ; =8.4/E2 ; =8.5/E2                     # baseline, then per mutant
python3 a0_mutant.py <wt>/src/sslm_abi.cpp <mutant>      # rebuild, run, git checkout -- src
python3 tests/precache/unpinned/apply_mutant.py <wt>/src/sslm_abi.cpp scatter-unrotated
python3 tests/precache/apply_e2_mutant.py <wt> first-compatible-schema
SUPERSLM_PRECACHE_CURRENT_TOKEN_MUTANT=1 superslm_precache_e1 =6.4/E2 …
superslm_precache_unpinned =4.7/E2:pkv_def ; =9.8/E2:pkv_def                              # baseline, then 3 mutants
```

**Verdict: CLEAR.** L1 may issue against the 20 stream-v2 bindings, on the condition above (B5 reads 20/20). The
planner fold of Finding A and §4 is owed, but it does not gate L1.
